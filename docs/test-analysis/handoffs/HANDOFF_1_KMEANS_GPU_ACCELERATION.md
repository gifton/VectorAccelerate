# Handoff 1: KMeans GPU Acceleration

## Problem Summary

Product Quantization (PQ) training is critically slow due to CPU-bound KMeans clustering. The KMeans algorithm runs during:
1. PQ codebook training (8-16 codebooks, each with 256 centroids)
2. IVF index training (clustering vectors into nlist partitions)

## Current Performance (CRITICAL)

| Operation | Samples | Codebooks | Time | Target |
|-----------|---------|-----------|------|--------|
| PQ training | 200 | 8 | 7,128ms | <500ms |
| PQ training | 100 | 8 | 153ms | <50ms |
| PQ training | 50 | 16 | 510ms | <100ms |
| PQ training | 30 | 8 | 1,246ms | <200ms |
| IVF insert 10K | N/A | N/A | 10.03s | <0.5s |

The bottleneck is in KMeans assignment and update phases.

## Root Cause Analysis

The current `assign_to_centroids` kernel processes ONE vector-to-ALL-centroids per thread. For PQ with 256 centroids per codebook:
- Each thread computes 256 distance calculations
- No shared memory utilization for centroids
- Memory bandwidth limited by repeated centroid loads

## Relevant Source Files

### Swift Kernel Wrappers
```
Sources/VectorAccelerate/Index/Kernels/Clustering/
├── KMeansPipeline.swift          # Main orchestration (lines 92-200 critical)
├── KMeansAssignKernel.swift      # Assignment kernel wrapper
├── KMeansUpdateKernel.swift      # Centroid update wrapper
└── KMeansConvergenceKernel.swift # Convergence check
```

### Metal Shaders
```
Sources/VectorAccelerate/Metal/Shaders/ClusteringShaders.metal
```

## Current Shader Implementation

### Legacy Assignment Kernel (SLOW)
```metal
kernel void assign_to_centroids(
    constant float* vectors [[buffer(0)]],
    constant float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimensions [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;

    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    uint min_centroid = 0;

    // PROBLEM: Serial loop over ALL centroids per thread
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;

        // PROBLEM: No shared memory - centroids reloaded per thread
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }

        if (distance < min_distance) {
            min_distance = distance;
            min_centroid = c;
        }
    }

    assignments[id] = min_centroid;
    distances[id] = sqrt(min_distance);
}
```

### Existing Tiled Kernel (INCOMPLETE)
There's a `tiled_kmeans_distance` kernel that attempts 2D tiling but:
1. It's not currently wired up in `KMeansAssignKernel.swift`
2. The tile sizes (TILE_SIZE_Q=32, TILE_SIZE_C=8) may not be optimal
3. Shared memory allocation is external (threadgroup buffers)

## Optimization Strategy

### Option A: Fix and Enable Tiled Kernel
1. Wire `tiled_kmeans_distance` into `KMeansAssignKernel.swift`
2. Optimize tile sizes for Apple Silicon (M1/M2/M3):
   - TILE_SIZE_Q: 64-128 (queries per tile)
   - TILE_SIZE_C: 32-64 (centroids per tile)
   - TILE_DIM: 32-64 (dimensions per shared memory load)
3. Add proper threadgroup memory sizing calculation

### Option B: Warp-Cooperative Distance Matrix
Modern approach using simdgroup operations:
```metal
kernel void kmeans_assign_simd(
    device const float* vectors [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& num_centroids [[buffer(4)]],
    constant uint& dimensions [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup processes one vector against multiple centroids
    // Use simd_sum for parallel reduction
    // Use simd_shuffle for warp-level communication
}
```

### Option C: Batched Distance Matrix + ArgMin
Split into two kernels:
1. `compute_all_distances`: Compute full [N, K] distance matrix
2. `find_argmin`: Parallel reduction to find minimum per row

This is memory-intensive but highly parallelizable.

## Swift Integration Points

### KMeansAssignKernel.swift Changes Needed
```swift
// Current: Uses "assign_to_centroids" function
// Change to: Use optimized kernel selection based on problem size

public func assign(
    vectors: MTLBuffer,
    centroids: MTLBuffer,
    assignments: MTLBuffer,
    distances: MTLBuffer,
    numVectors: Int,
    numCentroids: Int,
    dimension: Int
) async throws {
    // TODO: Select optimal kernel based on numVectors, numCentroids, dimension
    // - Small (N*K < 50K): Use basic kernel
    // - Medium: Use tiled kernel
    // - Large: Use batched distance matrix approach
}
```

### KMeansPipeline.swift Changes Needed
The `fit()` method (line 99+) should:
1. Profile initial kernel selection
2. Consider mini-batch KMeans for N > 10K
3. Early termination if convergence detected

## Success Criteria

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| PQ 200 samples/8cb | 7,128ms | <500ms | GPU assignment |
| PQ 100 samples/8cb | 153ms | <50ms | GPU assignment |
| IVF 10K insert | 10.03s | <0.5s | Batch assignment |
| Memory overhead | N/A | <2x input | Tiled approach |

## Testing Commands

```bash
# Run specific benchmark
swift test --filter "testProductQuantizationRoundtrip"
swift test --filter "testProductQuantizationTraining"
swift test --filter "testIVFSearch10K_128D"

# Run all quantization tests
swift test --filter "Quantization"
```

## Key Constraints

1. **Metal 4 Required**: Shaders must compile with `-std=metal3.0` minimum
2. **No Persistent Threadgroups**: Avoid features unavailable on all Apple Silicon
3. **Shared Memory Limit**: 32KB per threadgroup on most devices
4. **Thread Limit**: 1024 threads per threadgroup maximum
5. **Float32 Only**: No half-precision in current implementation

## Deliverables

1. Optimized `ClusteringShaders.metal` with new/fixed tiled kernel
2. Updated `KMeansAssignKernel.swift` to use optimized kernel
3. Benchmark comparison showing improvement
4. Documentation of tile size selection rationale
