# 1.1 Why GPU for Search

> **Understanding when and why GPUs accelerate vector search.**

---

## The Concept

Vector similarity search is fundamentally about computing **many independent distances**. When you search for nearest neighbors:

```
Query vector (1 × D)  ×  Database (N × D)  =  N distances to compute
```

Each distance computation is independent of the others. This is the key insight: **embarrassingly parallel workloads are perfect for GPUs**.

---

## Why It Matters

CPUs are designed for **latency**—making individual operations fast. GPUs are designed for **throughput**—making many operations complete in aggregate.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CPU Architecture                             │
│                                                                      │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│    │    Core 0    │ │    Core 1    │ │    Core 2    │  ...          │
│    │              │ │              │ │              │               │
│    │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │               │
│    │  │ SIMD4  │  │ │  │ SIMD4  │  │ │  │ SIMD4  │  │               │
│    │  └────────┘  │ │  └────────┘  │ │  └────────┘  │               │
│    │              │ │              │ │              │               │
│    │  Deep cache  │ │  Deep cache  │ │  Deep cache  │               │
│    │  Branch pred │ │  Branch pred │ │  Branch pred │               │
│    │  Speculation │ │  Speculation │ │  Speculation │               │
│    └──────────────┘ └──────────────┘ └──────────────┘               │
│                                                                      │
│    8-12 powerful cores, each optimized for single-thread speed       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Architecture                             │
│                                                                      │
│    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐    │
│    │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │    │
│    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤    │
│    │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │    │
│    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤    │
│    │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │ EU │    │
│    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤    │
│    │ ... hundreds more execution units ...                     │    │
│    └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘    │
│                                                                      │
│    Thousands of simple EUs, optimized for parallel throughput        │
└─────────────────────────────────────────────────────────────────────┘
```

### The Numbers

Apple Silicon GPU capabilities (approximate):

| Chip | GPU Cores | FP32 TFLOPS | Memory BW |
|------|-----------|-------------|-----------|
| M1 | 8 | 2.6 | 68 GB/s |
| M1 Pro | 16 | 5.2 | 200 GB/s |
| M1 Max | 32 | 10.4 | 400 GB/s |
| M2 | 10 | 3.6 | 100 GB/s |
| M2 Pro | 19 | 6.8 | 200 GB/s |
| M2 Max | 38 | 13.6 | 400 GB/s |
| M3 Max | 40 | 14.2 | 400 GB/s |

Compare to CPU SIMD (M1 CPU ~0.5 TFLOPS for vectorized code). The GPU offers **10-30× more raw compute**.

---

## The Technique: Understanding Parallelism

### Data Parallelism in Distance Computation

When computing L2 distance between query `q` and database vector `d`:

```
distance² = Σᵢ (qᵢ - dᵢ)²
```

On CPU (VectorCore approach):
```swift
// CPU: Process 4 elements at a time with SIMD4
var sum = SIMD4<Float>.zero
for i in stride(from: 0, to: dimension, by: 4) {
    let diff = q[i..<i+4] - d[i..<i+4]
    sum += diff * diff
}
return sum.sum()
```

On GPU, we think differently. Instead of one thread processing one vector pair with SIMD, we have **many threads each processing part of the work**:

```metal
// GPU: Each thread processes one (query, database) pair
kernel void l2_distance_kernel(
    device const float* queries,
    device const float* database,
    device float* distances,
    uint3 tid [[thread_position_in_grid]]
) {
    uint queryIdx = tid.x;
    uint dbIdx = tid.y;

    // This thread computes ONE distance value
    float sum = 0.0;
    for (uint d = 0; d < dimension; d++) {
        float diff = queries[queryIdx * dimension + d]
                   - database[dbIdx * dimension + d];
        sum += diff * diff;
    }
    distances[queryIdx * numDatabase + dbIdx] = sum;
}
```

With 100 queries × 10,000 database vectors, we dispatch **1,000,000 threads**—each computing one distance.

### Scaling Analysis

```
┌──────────────────────────────────────────────────────────────────────┐
│                    WORK DISTRIBUTION COMPARISON                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  CPU (8 cores, SIMD4):                                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Core 0: [████████████████████████████████████████████████████] │  │
│  │ Core 1: [████████████████████████████████████████████████████] │  │
│  │ Core 2: [████████████████████████████████████████████████████] │  │
│  │ ...                                                            │  │
│  │ Core 7: [████████████████████████████████████████████████████] │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  Each core processes N/8 distances sequentially                       │
│  Time: O(N × D / (8 × 4))  [4 from SIMD]                             │
│                                                                       │
│  GPU (1000s of threads):                                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ T0:[█] T1:[█] T2:[█] T3:[█] ... T999:[█] T1000:[█] ...         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  Thousands of threads run in parallel                                │
│  Time: O(D) + overhead  [All N distances computed "at once"]         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 VectorCore Connection

In VectorCore, you learned that SIMD4 processes 4 floats simultaneously:

```swift
// VectorCore: SIMD4 for 4-wide parallelism
let diff = SIMD4<Float>(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3])
let squared = diff * diff
```

GPU SIMD groups extend this to **32 threads** executing in lockstep:

```metal
// VectorAccelerate: 32 threads per SIMD group
// Each thread handles its portion, then they reduce together
float partial = /* this thread's contribution */;
float total = simd_sum(partial);  // Sum across 32 threads
```

| Concept | VectorCore (CPU) | VectorAccelerate (GPU) |
|---------|-----------------|------------------------|
| SIMD width | 4 floats | 32 threads |
| Instruction | Single SIMD4 op | SIMD group instruction |
| Synchronization | Implicit | Explicit (`simd_shuffle`, `simd_sum`) |

---

## 🔗 VectorIndex Connection

VectorIndex taught you that brute-force search is O(N×D) per query:

```swift
// VectorIndex: Flat search on CPU
func search(query: [Float], k: Int) -> [SearchResult] {
    var distances: [(index: Int, distance: Float)] = []
    for (i, vector) in database.enumerated() {
        let dist = l2Distance(query, vector)  // O(D)
        distances.append((i, dist))
    }
    return distances.sorted().prefix(k)  // O(N log N)
}
```

VectorAccelerate parallelizes this across the GPU:

```swift
// VectorAccelerate: Flat search on GPU
func search(query: [Float], k: Int) async throws -> [SearchResult] {
    // 1. Compute ALL N distances in parallel on GPU
    let distances = try await l2Kernel.compute(
        queries: [query],
        database: database
    )

    // 2. Find top-K in parallel on GPU
    let topK = try await topKKernel.select(
        distances: distances,
        k: k
    )

    return topK
}
```

The O(N×D) work is now distributed across thousands of GPU threads.

---

## When GPU Wins (and When It Doesn't)

### GPU Wins: Batch Queries

```
Single query to 1M database:
  CPU:  850ms
  GPU:  50ms + 10ms transfer = 60ms  ✅ GPU wins

10 queries to 1M database:
  CPU:  8,500ms (10 × 850ms)
  GPU:  60ms + 10ms transfer = 70ms  ✅ GPU wins big (121×)

Transfer cost amortized across queries!
```

### GPU Loses: Single Low-Latency Query

```
Single query to 10K database:
  CPU:  ~8ms (fast!)
  GPU:  ~50μs compute + ~1ms launch overhead  ❌ CPU wins on latency

The GPU launch overhead (~50-100μs) dominates for small work.
```

### Decision Framework

```swift
// Pseudocode for CPU/GPU selection
func shouldUseGPU(numQueries: Int, numDatabase: Int, dimension: Int) -> Bool {
    let workSize = numQueries * numDatabase * dimension

    // Heuristics based on empirical testing:
    if workSize < 10_000_000 {
        return false  // GPU overhead not worth it
    }

    if numQueries >= 10 {
        return true   // Batch queries amortize transfer
    }

    if numDatabase >= 100_000 {
        return true   // Large database benefits from parallelism
    }

    return false
}
```

---

## In VectorAccelerate

VectorAccelerate implements adaptive routing:

📍 See: `Sources/VectorAccelerate/Configuration/AdaptiveThresholds.swift`

```swift
/// Adaptive thresholds for CPU/GPU routing
public struct AdaptiveThresholds: Sendable {
    /// Minimum work size for GPU acceleration
    public let minGPUWorkSize: Int

    /// Minimum batch size to prefer GPU
    public let minBatchSize: Int

    /// Minimum database size for single-query GPU
    public let minDatabaseForSingleQuery: Int

    public static let `default` = AdaptiveThresholds(
        minGPUWorkSize: 10_000_000,
        minBatchSize: 8,
        minDatabaseForSingleQuery: 50_000
    )
}
```

---

## Key Takeaways

1. **GPUs trade latency for throughput**: Individual operations are slower, but aggregate throughput is much higher

2. **Vector search is embarrassingly parallel**: Each distance is independent—perfect for GPU

3. **Transfer costs matter**: GPU wins scale with batch size because transfer overhead is amortized

4. **CPU and GPU are complementary**: Use CPU for single low-latency queries, GPU for batch throughput

5. **Measure, don't assume**: The crossover point depends on your specific hardware and workload

---

## Next Up

Now that you understand *why* GPUs help, let's learn *how* they work:

**[→ 1.2 Metal Compute Basics](./02-Metal-Compute-Basics.md)**

---

*Guide 1.1 of 1.3 • Chapter 1: GPU Fundamentals*
