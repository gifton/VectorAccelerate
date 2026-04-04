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

## The Technique: Hierarchical SIMD Reduction

In version 0.4.0, VectorAccelerate moved from naive per-thread atomics to a **Hierarchical Reduction** model. This is the gold standard for Apple Silicon, maximizing throughput by reducing global memory contention.

Let's examine the 4-phase architecture:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/DistanceShaders.metal

kernel void l2_distance(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* distances [[buffer(2)]],
    // ... position attributes ...
) {
    // PHASE 1: Local float4 Accumulation
    // Threads load 128-bit chunks from the UMA bus into registers.
    float sq_diff = 0.0;
    for (uint i = lid; i < vec_dim; i += threads_per_tg) {
        float4 diff = q4[i] - t4[i];
        sq_diff += dot(diff, diff);
    }

    // PHASE 2: SIMD-group reduction
    // High-speed parallel sum across 32 threads using hardware shuffle.
    float simd_result = simd_sum(sq_diff);

    // PHASE 3: Threadgroup Consolidation
    // Cross-warp sum using high-speed shared memory.
    threadgroup float shared_sums[32]; 
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // PHASE 4: Exactly ONE write per result
    // Consolidation into a single result with zero contention.
    if (lid == 0) {
        float final_sum = 0.0;
        for (uint i = 0; i < active_simd_groups; i++) {
            final_sum += shared_sums[i];
        }
        distances[query_idx] = final_sum;
    }
}
```

### Key Optimizations Explained

#### 1. Warp-Level Parallelism (`simd_sum`)

Instead of each thread trying to write to global memory, we use the `simd_sum()` instruction. This allows 32 threads to sum their results in a single GPU cycle without using memory at all.

#### 2. Shared Memory Consolidation

For massive threadgroups, we use `threadgroup` memory to bridge the gap between SIMD-groups. This ensures that even if we have 1024 threads working on one vector, only **one** global memory write occurs at the end.

#### 3. Register Reuse

By pulling chunks into `float4` and accumulating locally, we saturate the 128-bit memory bus while keeping the hottest data in the absolute fastest part of the GPU: the registers.

---

## The Technique: Dynamic Threadgroup Sizing

Hardcoded threadgroup sizes are fragile. 0.4.0 dispatchers calculate them dynamically:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift

// Query the hardware's execution width (always 32 on Apple Silicon)
let w = pipelineState.threadExecutionWidth
let h = pipelineState.maxTotalThreadsPerThreadgroup / w
let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

// Dispatch exactly one threadgroup per vector pair
let threadgroups = MTLSizeMake(numQueries, numDatabase, 1)
```

This ensures the kernel automatically runs at peak occupancy whether it's on an iPhone (A-series) or a Mac Studio (M-Ultra).

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
// For maximum control, use the buffer API with pooling
let queryToken = try await context.getBuffer(for: flatQueries)
let dbToken = try await context.getBuffer(for: flatDatabase)

let parameters = L2DistanceParameters(
    numQueries: 100,
    numDatabase: 10_000,
    dimension: 768,
    computeSqrt: false
)

let distanceBuffer = try await kernel.execute(
    queries: queryToken.buffer,
    database: dbToken.buffer,
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
