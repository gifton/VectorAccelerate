# 2.1 L2 Distance on GPU

> **The workhorse kernel—Euclidean distance computation at GPU scale.**

---

## The Concept

L2 (Euclidean) distance measures the straight-line distance between two vectors:

```
L2(q, d) = √Σᵢ (qᵢ - dᵢ)²

Or squared (avoids expensive sqrt, same ranking):
L2²(q, d) = Σᵢ (qᵢ - dᵢ)²
```

On CPU, you'd compute this with SIMD4. On GPU, we parallelize across:
1. **All (query, database) pairs**: Each thread handles one pair
2. **Within each pair**: Use float4 for vectorized accumulation

---

## Why It Matters

L2 distance is the default for most embedding models. A single search might require:

```
100 queries × 1,000,000 database vectors = 100,000,000 distances

Each distance at D=768:
  768 subtractions + 768 multiplications + 768 additions + 1 sqrt
  ≈ 2,300 FLOPs per distance

Total: 230 billion FLOPs for one search batch!

This is why GPU matters.
```

---

## The Technique: Generic L2 Kernel

Let's examine the generic L2 kernel that handles any dimension:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:34-91

kernel void l2_distance_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    // Bounds checking (required when using dispatchThreads)
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Calculate vector pointers using strides
    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Use a float4 accumulator to improve ILP and vectorize
    float4 sum4 = float4(0.0f);
    const uint dimension = params.dimension;

    // Process 4 elements at a time
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

    for (uint i = 0; i < simd_blocks; ++i) {
        float4 diff = query4[i] - database4[i];
        // Fused multiply-add for precision and performance
        sum4 = fma(diff, diff, sum4);
    }

    // Horizontal reduction of the vector accumulator
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;

    // Handle remaining elements
    if (remainder > 0) {
        device const float* query_tail = query + (simd_blocks * 4);
        device const float* database_tail = database + (simd_blocks * 4);

        for (uint i = 0; i < remainder; ++i) {
            float diff = query_tail[i] - database_tail[i];
            sum = fma(diff, diff, sum);
        }
    }

    // Apply sqrt if requested
    float distance = params.computeSqrt ? sqrt(sum) : sum;

    // Store result
    const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
    distances[outputIdx] = distance;
}
```

### Key Optimizations Explained

#### 1. float4 Vectorization

```metal
float4 diff = query4[i] - database4[i];
```

Instead of processing one float at a time, we process 4. This:
- Reduces loop iterations by 4×
- Enables better instruction pipelining
- Matches GPU memory transaction sizes

#### 2. Fused Multiply-Add (FMA)

```metal
sum4 = fma(diff, diff, sum4);
// Equivalent to: sum4 = diff * diff + sum4
// But FMA:
// - Uses single instruction (faster)
// - Has better numerical precision
// - Enables better pipelining
```

#### 3. Loop + Remainder Pattern

```metal
const uint simd_blocks = dimension / 4;
const uint remainder = dimension % 4;

// Main loop: fast float4 path
for (uint i = 0; i < simd_blocks; ++i) { ... }

// Cleanup: handle odd dimensions
for (uint i = 0; i < remainder; ++i) { ... }
```

This handles non-multiple-of-4 dimensions correctly while keeping the fast path vectorized.

---

## The Technique: Dimension-Optimized Kernels

For common embedding dimensions, VectorAccelerate provides hand-optimized kernels:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:97-147

// Optimized for D=384 (MiniLM, Sentence-BERT)
kernel void l2_distance_384_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Hardcoded stride for optimal address calculation
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 384);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 384);

    // Two interleaved accumulators for Instruction-Level Parallelism
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    // Unrolled by 8: 96 iterations become 12 loops
    for (uint i = 0; i < 96; i += 8) {
        float4 diff0 = query4[i+0] - database4[i+0];
        float4 diff1 = query4[i+1] - database4[i+1];
        float4 diff2 = query4[i+2] - database4[i+2];
        float4 diff3 = query4[i+3] - database4[i+3];
        float4 diff4 = query4[i+4] - database4[i+4];
        float4 diff5 = query4[i+5] - database4[i+5];
        float4 diff6 = query4[i+6] - database4[i+6];
        float4 diff7 = query4[i+7] - database4[i+7];

        // Interleaved accumulation hides FMA latency
        acc0 = fma(diff0, diff0, acc0);
        acc1 = fma(diff1, diff1, acc1);
        acc0 = fma(diff2, diff2, acc0);
        acc1 = fma(diff3, diff3, acc1);
        acc0 = fma(diff4, diff4, acc0);
        acc1 = fma(diff5, diff5, acc1);
        acc0 = fma(diff6, diff6, acc0);
        acc1 = fma(diff7, diff7, acc1);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1;
    float sum = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    float distance = params.computeSqrt ? sqrt(sum) : sum;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}
```

### Why Multiple Accumulators?

```
Single accumulator:
  fma(diff0, diff0, acc)  ← Must wait for acc to be ready
  fma(diff1, diff1, acc)  ← Must wait for previous fma
  fma(diff2, diff2, acc)  ← Must wait for previous fma
  ...
  Latency-bound! Each FMA waits for the previous.

Two accumulators (interleaved):
  fma(diff0, diff0, acc0)  ← acc0 starts computing
  fma(diff1, diff1, acc1)  ← acc1 starts while acc0 computes
  fma(diff2, diff2, acc0)  ← acc0 is ready now
  fma(diff3, diff3, acc1)  ← acc1 is ready now
  ...
  Throughput-bound! FMAs execute in parallel.
```

### Dimension Coverage

| Dimension | float4 blocks | Unroll factor | Accumulators |
|-----------|--------------|---------------|--------------|
| 384 | 96 | 8 | 2 |
| 512 | 128 | 8 | 2 |
| 768 | 192 | 12 | 3 |
| 1536 | 384 | 16 | 4 |

---

## Pipeline Selection in Swift

The Swift wrapper automatically selects the optimal kernel:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:203-218

/// Select the optimal pipeline for a given dimension.
private func selectPipeline(for dimension: UInt32) -> (pipeline: any MTLComputePipelineState, name: String) {
    switch dimension {
    case 384:
        return (pipeline384, "l2_distance_384_kernel")
    case 512:
        return (pipeline512, "l2_distance_512_kernel")
    case 768:
        return (pipeline768, "l2_distance_768_kernel")
    case 1536:
        return (pipeline1536, "l2_distance_1536_kernel")
    default:
        return (genericPipeline, "l2_distance_kernel")
    }
}
```

---

## 🔗 VectorCore Connection

VectorCore's L2 kernel uses similar patterns on CPU:

```swift
// VectorCore: CPU SIMD4 L2 distance
public func l2DistanceSquared<V: VectorProtocol>(
    _ a: V, _ b: V
) -> Float where V.Scalar == Float {
    var sum = SIMD4<Float>.zero

    a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            let count = aPtr.count
            let simdCount = count / 4

            for i in 0..<simdCount {
                let aVec = SIMD4<Float>(
                    aPtr[i*4], aPtr[i*4+1], aPtr[i*4+2], aPtr[i*4+3]
                )
                let bVec = SIMD4<Float>(
                    bPtr[i*4], bPtr[i*4+1], bPtr[i*4+2], bPtr[i*4+3]
                )
                let diff = aVec - bVec
                sum += diff * diff
            }

            // Handle remainder...
        }
    }

    return sum.x + sum.y + sum.z + sum.w
}
```

| Aspect | VectorCore (CPU) | VectorAccelerate (GPU) |
|--------|-----------------|------------------------|
| Vectorization | SIMD4 (4-wide) | float4 (4-wide per thread) |
| Parallelism | 8-12 cores | Thousands of threads |
| Unrolling | Compiler-controlled | Hand-tuned per dimension |
| Memory | L1/L2 cache | Device memory + registers |

---

## 🔗 VectorIndex Connection

VectorIndex's FlatIndex uses distance computation as its core operation:

```swift
// VectorIndex: FlatIndex search
public func search(query: Vector<D>, k: Int) -> [SearchResult] {
    var results: [(index: Int, distance: Float)] = []

    for (i, vector) in vectors.enumerated() {
        let dist = distance(query, vector)  // ← This is the hot path
        results.append((i, dist))
    }

    return results.sorted { $0.distance < $1.distance }.prefix(k)
}
```

VectorAccelerate replaces this sequential loop with parallel GPU execution:

```swift
// VectorAccelerate: GPU batch distance
let distances = try await l2Kernel.compute(
    queries: [query],
    database: vectors  // All distances computed in parallel
)
```

---

## Using L2DistanceKernel

### Basic Usage

```swift
import VectorAccelerate

// Create kernel
let context = try await Metal4Context()
let kernel = try await L2DistanceKernel(context: context)

// Compute distances
let queries: [[Float]] = [/* 100 queries × 768D */]
let database: [[Float]] = [/* 10,000 vectors × 768D */]

let distances = try await kernel.compute(
    queries: queries,
    database: database,
    computeSqrt: false  // Squared distance is faster, same ranking
)
// distances: [[Float]] with shape [100][10,000]
```

### Low-Level Buffer API

```swift
// For maximum control, use the buffer API
let queryBuffer = try context.bufferFactory.makeBuffer(
    from: flatQueries,
    label: "queries"
)
let dbBuffer = try context.bufferFactory.makeBuffer(
    from: flatDatabase,
    label: "database"
)

let parameters = L2DistanceParameters(
    numQueries: 100,
    numDatabase: 10_000,
    dimension: 768,
    computeSqrt: false
)

let distanceBuffer = try await kernel.execute(
    queries: queryBuffer,
    database: dbBuffer,
    parameters: parameters
)
```

---

## Performance Characteristics

### Compute vs Memory Bound

```
L2 distance per pair (D=768):
  Memory reads:  768 × 2 × 4 bytes = 6 KB
  Compute:       768 × 3 FLOPs ≈ 2.3 KFLOPs

Arithmetic intensity: 2.3 KFLOPs / 6 KB ≈ 0.38 FLOPs/byte

Apple M2 Max:
  Memory BW: 400 GB/s
  Compute: 13.6 TFLOPS

At 0.38 FLOPs/byte:
  Memory-limited throughput: 400 × 0.38 = 152 GFLOPS
  Peak compute: 13,600 GFLOPS

L2 distance is MEMORY BOUND on GPU!
```

This is why dimension-optimized kernels help—they reduce memory transactions through better register utilization.

---

## Key Takeaways

1. **float4 vectorization**: Process 4 elements per instruction for memory efficiency

2. **FMA (fused multiply-add)**: Better precision and performance than separate ops

3. **Multiple accumulators**: Hide FMA latency through instruction-level parallelism

4. **Dimension specialization**: Hand-tuned kernels for common embedding sizes provide 20-40% speedup

5. **Squared distance**: Avoid sqrt when possible (same ranking, faster)

6. **Memory bound**: L2 distance is limited by memory bandwidth, not compute

---

## Next Up

Cosine similarity has different optimization strategies:

**[→ 2.2 Cosine Similarity on GPU](./02-Cosine-On-GPU.md)**

---

*Guide 2.1 of 2.4 • Chapter 2: Distance Kernels*
