# 5.4 CPU vs GPU Decision Framework

> **When to accelerate—practical heuristics for routing decisions.**

---

## The Concept

GPU acceleration isn't always beneficial. This guide provides a framework for deciding when to use GPU vs. CPU.

```
The Decision:
  CPU (VectorCore/VectorIndex): Lower latency, simpler code
  GPU (VectorAccelerate): Higher throughput, more complexity

Neither is universally better - it depends on your workload.
```

---

## Why It Matters

Wrong routing wastes resources:

```
GPU for single small query:
  Overhead: ~100 μs (kernel launch, buffer setup)
  Compute: ~10 μs
  Total: 110 μs
  CPU would take: 50 μs
  ⚠️ 2× slower on GPU!

CPU for large batch:
  Per-query: 50 μs × 1000 = 50 ms
  GPU batch: 5 ms
  ⚠️ 10× slower on CPU!
```

---

## The Framework

### Decision Tree

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CPU VS GPU DECISION TREE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Is data already on GPU?                                            │
│       │                                                              │
│       ├── YES → Use GPU (avoid transfer)                            │
│       │                                                              │
│       └── NO → Continue ↓                                           │
│                                                                      │
│  Is this a batch operation (>10 queries)?                           │
│       │                                                              │
│       ├── YES → Use GPU (amortize overhead)                         │
│       │                                                              │
│       └── NO → Continue ↓                                           │
│                                                                      │
│  Is dataset large (>50K vectors)?                                   │
│       │                                                              │
│       ├── YES → Use GPU (more parallelism pays off)                 │
│       │                                                              │
│       └── NO → Continue ↓                                           │
│                                                                      │
│  Is latency critical (<1ms required)?                               │
│       │                                                              │
│       ├── YES → Use CPU (no GPU launch overhead)                    │
│       │                                                              │
│       └── NO → Use GPU (throughput wins)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Numerical Thresholds

Based on M-series Apple Silicon benchmarks:

| Criterion | GPU Better When | CPU Better When |
|-----------|----------------|-----------------|
| Database Size | > 50,000 vectors | < 10,000 vectors |
| Batch Size | > 10 queries | 1-2 queries |
| Dimension | > 256 | < 128 |
| Latency Req | > 5 ms acceptable | < 1 ms required |
| K (results) | Any | Any |

---

## Adaptive Routing Implementation

```swift
// 📍 See conceptual implementation

/// Adaptive router that chooses CPU or GPU based on workload
public actor AdaptiveSearchRouter {
    private let cpuIndex: FlatIndex<D768>
    private let gpuIndex: AcceleratedVectorIndex

    public struct Thresholds: Sendable {
        let minGPUBatchSize: Int = 10
        let minGPUDatabaseSize: Int = 50_000
        let maxLatencyForCPU: TimeInterval = 0.001  // 1ms

        public static let `default` = Thresholds()
    }

    private let thresholds: Thresholds

    /// Route query to appropriate backend
    public func search(
        queries: [[Float]],
        k: Int,
        maxLatency: TimeInterval?
    ) async throws -> [[SearchResult]] {

        let shouldUseGPU = selectBackend(
            batchSize: queries.count,
            databaseSize: await gpuIndex.count,
            maxLatency: maxLatency
        )

        if shouldUseGPU {
            return try await gpuIndex.search(queries: queries, k: k)
        } else {
            return try await cpuBatchSearch(queries: queries, k: k)
        }
    }

    private func selectBackend(
        batchSize: Int,
        databaseSize: Int,
        maxLatency: TimeInterval?
    ) -> Bool {
        // Latency constraint forces CPU
        if let maxLatency, maxLatency < thresholds.maxLatencyForCPU {
            return false
        }

        // Batch queries favor GPU
        if batchSize >= thresholds.minGPUBatchSize {
            return true
        }

        // Large database favors GPU
        if databaseSize >= thresholds.minGPUDatabaseSize {
            return true
        }

        // Default to CPU for small workloads
        return false
    }
}
```

---

## Workload-Specific Guidelines

### Real-Time Autocomplete

```
Requirements: <10ms latency, single query
Database: 10K-100K documents

Recommendation: CPU (VectorIndex)
  - GPU launch overhead too high for single query
  - HNSW provides <2ms search
  - GPU doesn't help at this scale
```

### Batch Embedding Search

```
Requirements: High throughput, 100s of queries
Database: 1M+ vectors

Recommendation: GPU (VectorAccelerate)
  - Batch amortizes overhead
  - GPU parallelism shines at scale
  - Can process 100 queries in <20ms
```

### Hybrid Retrieval-Augmented Generation (RAG)

```
Requirements: Moderate latency (50-100ms ok), variable query count
Database: 100K-1M documents

Recommendation: Adaptive
  - Single query: CPU (fast response)
  - Batch reranking: GPU (high throughput)
```

---

## Measuring Your Workload

```swift
/// Benchmark to determine optimal routing
func benchmarkRouting(
    queries: [[Float]],
    k: Int
) async throws -> (cpu: TimeInterval, gpu: TimeInterval) {

    // Benchmark CPU
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for query in queries {
        _ = try await cpuIndex.search(query: query, k: k)
    }
    let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

    // Benchmark GPU
    let gpuStart = CFAbsoluteTimeGetCurrent()
    _ = try await gpuIndex.search(queries: queries, k: k)
    let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

    return (cpuTime, gpuTime)
}

// Use results to tune thresholds
let (cpu, gpu) = try await benchmarkRouting(queries, k: 10)
let crossover = /* find where GPU becomes faster */
```

---

## 🔗 VectorCore Connection

VectorCore is always CPU—use it for:
- Low-level distance functions
- Custom algorithms
- Single-vector operations
- When VectorIndex is overkill

---

## 🔗 VectorIndex Connection

VectorIndex provides CPU-optimized structures:
- HNSW for low-latency single queries
- IVF for large-scale with good recall
- Flat for simplicity

Use VectorIndex when GPU overhead exceeds benefit.

---

## Key Takeaways

1. **Batch size matters most**: GPU overhead needs amortization

2. **Database size is secondary**: Larger = more GPU parallelism

3. **Latency constraints override**: Ultra-low latency needs CPU

4. **Measure don't guess**: Benchmark your actual workload

5. **Adaptive is best**: Route dynamically based on query characteristics

---

## Chapter Summary

You've learned pipeline optimization:

- ✅ Kernel fusion for memory reduction
- ✅ Async compute for CPU/GPU overlap
- ✅ Profiling with Instruments
- ✅ CPU vs GPU decision framework

Time for the capstone—building a complete accelerated system.

**[→ Chapter 6: Capstone](../06-Capstone/README.md)**

---

*Guide 5.4 of 5.4 • Chapter 5: Pipeline Optimization*
