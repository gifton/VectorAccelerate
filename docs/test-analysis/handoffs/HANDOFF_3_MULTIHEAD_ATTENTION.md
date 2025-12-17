# Handoff 3: Multi-Head Attention Optimization

## Problem Summary

Multi-head attention has 6.6x overhead compared to single-head attention, making it impractical for transformer-based similarity computations with multiple heads.

## Current Performance

| Configuration | Pairs/sec | vs Single-Head |
|---------------|-----------|----------------|
| 1 head | 4,070,340 | 1.0x (baseline) |
| 4 heads | 1,198,049 | 0.29x (3.4x slower) |
| 8 heads | 745,098 | 0.18x (5.5x slower) |
| 12 heads | 613,086 | 0.15x (6.6x slower) |

**Expected**: Multi-head should be ~1.5-2x slower than single-head, not 6.6x.

Test source: `AttentionSimilarityBenchmarkTests`

## Architecture Overview

Attention similarity computes:
```
similarity(q, k) = (q @ Wq) · (k @ Wk) / temperature
```

For multi-head:
```
similarity(q, k) = mean(
    (q @ Wq[0]) · (k @ Wk[0]),
    (q @ Wq[1]) · (k @ Wk[1]),
    ...
    (q @ Wq[h-1]) · (k @ Wk[h-1])
) / temperature
```

## Root Cause Analysis

The current `multihead_attention_similarity_kernel` processes heads **sequentially within each thread**:

```metal
kernel void multihead_attention_similarity_kernel(
    device const float* queries [[buffer(0)]],          // [N, D]
    device const float* keys [[buffer(1)]],             // [M, D]
    device const float* queryProjection [[buffer(2)]],  // [numHeads * H, D]
    device const float* keyProjection [[buffer(3)]],    // [numHeads * H, D]
    device float* similarities [[buffer(4)]],           // [N, M]
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    // One thread per (query, key) pair
    // PROBLEM: Sequential loop over heads
    for (uint head = 0; head < numHeads; ++head) {
        // Project query for this head
        for (uint j = 0; j < headDim; ++j) { ... }

        // Project key for this head
        for (uint j = 0; j < headDim; ++j) { ... }

        // Compute dot product
        totalSimilarity += dotProduct(projQuery, projKey);
    }
}
```

**Problems**:
1. Each head requires full D-dimensional projection (768 * headDim * 2 ops)
2. Heads processed serially - no parallelism across heads
3. Weight matrices read numHeads times per (q, k) pair
4. Thread-local storage limits headDim to 64 for multi-head

## Relevant Source Files

### Swift Kernel Wrapper
```
Sources/VectorAccelerate/Kernels/Metal4/AttentionSimilarityKernel.swift
```

### Metal Shader
```
Sources/VectorAccelerate/Metal/Shaders/AttentionSimilarity.metal
```

## Optimization Strategies

### Strategy A: Fused Multi-Head Projection

Project query/key once through concatenated weight matrix, then split into heads:

```metal
// Instead of: h separate [headDim, D] projections
// Use: one [numHeads * headDim, D] projection, then reshape

kernel void multihead_attention_fused(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* queryProjection [[buffer(2)]],  // [numHeads * headDim, D]
    device const float* keyProjection [[buffer(3)]],
    device float* similarities [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint queryIdx = tid.x;
    const uint keyIdx = tid.y;

    // Project query ONCE into all head dimensions
    float projQuery[MAX_TOTAL_HEAD_DIM];  // e.g., 12 * 64 = 768
    projectFull(queries + queryIdx * D, queryProjection, projQuery, D, numHeads * headDim);

    // Project key ONCE
    float projKey[MAX_TOTAL_HEAD_DIM];
    projectFull(keys + keyIdx * D, keyProjection, projKey, D, numHeads * headDim);

    // Compute dot products per head and average
    float totalSim = 0.0f;
    for (uint h = 0; h < numHeads; ++h) {
        float headSim = dotProduct(
            projQuery + h * headDim,
            projKey + h * headDim,
            headDim
        );
        totalSim += headSim;
    }

    similarities[queryIdx * M + keyIdx] = totalSim / (numHeads * temperature);
}
```

This halves the projection work (compute each projection once, not per-head).

### Strategy B: 3D Grid with Head Parallelism

Add a third grid dimension for heads:

```metal
// Grid: (numQueries, numKeys, numHeads)
kernel void multihead_attention_3d(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* queryProjection [[buffer(2)]],
    device const float* keyProjection [[buffer(3)]],
    device float* perHeadSimilarities [[buffer(4)]],  // [N, M, H]
    constant AttentionParams& params [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint queryIdx = tid.x;
    uint keyIdx = tid.y;
    uint headIdx = tid.z;

    // Each thread computes ONE head's contribution
    device const float* wq = queryProjection + headIdx * headDim * D;
    device const float* wk = keyProjection + headIdx * headDim * D;

    float projQuery[64], projKey[64];
    project(queries + queryIdx * D, wq, projQuery, D, headDim);
    project(keys + keyIdx * D, wk, projKey, D, headDim);

    float sim = dotProduct(projQuery, projKey, headDim);
    perHeadSimilarities[(queryIdx * M + keyIdx) * H + headIdx] = sim;
}

// Second pass: reduce across heads
kernel void reduce_heads(
    device const float* perHeadSims [[buffer(0)]],
    device float* similarities [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint idx = tid.x * M + tid.y;
    float sum = 0.0f;
    for (uint h = 0; h < H; ++h) {
        sum += perHeadSims[idx * H + h];
    }
    similarities[idx] = sum / (H * temperature);
}
```

Pros: Full parallelism across heads
Cons: Requires intermediate buffer, two kernel dispatches

### Strategy C: Simdgroup Cooperative Processing

Use simdgroup operations to parallelize head computation within a warp:

```metal
kernel void multihead_attention_simd(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* queryProjection [[buffer(2)]],
    device const float* keyProjection [[buffer(3)]],
    device float* similarities [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint queryIdx = gid.x;
    uint keyIdx = gid.y;

    // Each simd lane handles a different head (up to 32 heads)
    uint headIdx = simd_lane % numHeads;

    if (simd_lane < numHeads) {
        // Compute this head's similarity
        float headSim = computeHeadSimilarity(
            queries + queryIdx * D,
            keys + keyIdx * D,
            queryProjection + headIdx * headDim * D,
            keyProjection + headIdx * headDim * D,
            D, headDim
        );

        // Simd reduction to sum across heads
        float totalSim = simd_sum(headSim) / (numHeads * temperature);

        // Only lane 0 writes result
        if (simd_lane == 0) {
            similarities[queryIdx * M + keyIdx] = totalSim;
        }
    }
}
```

This uses 32-wide SIMD to parallelize up to 32 heads within a single warp.

### Strategy D: Precomputed Key Projections

For database search (many queries, fixed keys):

```swift
// Swift side: Precompute projected keys
let projectedKeys = preprojectKeys(keys: databaseKeys, weights: keyProjection)

// Metal: Skip key projection during search
kernel void attention_with_preprojected_keys(
    device const float* queries [[buffer(0)]],
    device const float* projectedKeys [[buffer(1)]],  // Already projected!
    device const float* queryProjection [[buffer(2)]],
    device float* similarities [[buffer(3)]],
    ...)
```

This is the most effective optimization for the common search use case.

## Swift Integration Points

### AttentionSimilarityKernel.swift Changes

```swift
public struct AttentionSimilarityKernel: Metal4Kernel {
    // Add head-parallel pipelines
    private var multiheadPipeline3D: MTLComputePipelineState?
    private var reduceHeadsPipeline: MTLComputePipelineState?

    // Add precomputed key support
    public func preprojectKeys(
        keys: MTLBuffer,
        numKeys: Int
    ) async throws -> MTLBuffer

    // Kernel selection based on configuration
    public func computeSimilarity(
        queries: MTLBuffer,
        keys: MTLBuffer,
        output: MTLBuffer,
        params: AttentionParams
    ) async throws {
        if params.numHeads == 1 {
            // Use single-head kernel
        } else if preprojectedKeys != nil {
            // Use preprojected path
        } else if params.numHeads <= 32 {
            // Use simdgroup cooperative
        } else {
            // Use 3D grid + reduction
        }
    }
}
```

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| 12-head throughput | 613K pairs/s | >2M pairs/s |
| Multi/Single ratio | 6.6x slower | <2x slower |
| Memory overhead | 1x | 2x acceptable (for preprojected keys) |

## Testing Commands

```bash
# Run attention benchmarks
swift test --filter "AttentionSimilarity"
swift test --filter "testMultiHeadScaling"

# Run ML integration tests
swift test --filter "MLIntegration"
```

## Key Constraints

1. **Head Dimension**: Typically 64 (768/12 for BERT)
2. **Max Heads**: Up to 16 for most models
3. **Thread-Local Limit**: 256 floats for projections (fits 4 heads * 64 dim)
4. **Simdgroup Size**: 32 on Apple Silicon (perfect for up to 32 heads)

## Deliverables

1. Optimized `AttentionSimilarity.metal` with multi-head kernel
2. Updated `AttentionSimilarityKernel.swift` with kernel selection
3. Optional: Preprojected key support for database search
4. Benchmark showing 3x+ improvement for 12-head configuration
