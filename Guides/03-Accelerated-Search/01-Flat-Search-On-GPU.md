# 3.1 Flat Search on GPU

> **When brute force wins—GPU-powered exhaustive search.**

---

## The Concept

Flat (brute-force) search computes distances to *every* vector in the database:

```
Query q → Compare to [d₀, d₁, d₂, ..., dₙ] → Return K nearest

No index structure, no approximation, 100% recall.
```

On CPU, this is O(N×D) per query—prohibitively slow for large N. But GPUs change the equation by parallelizing across all N comparisons.

---

## Why It Matters

GPU flat search can be **faster than CPU approximate search** for many use cases:

```
Scenario: 100K vectors, 768 dimensions, K=10

CPU HNSW (ef=64):    ~2 ms, 98% recall
CPU Flat:            ~68 ms, 100% recall
GPU Flat:            ~1.2 ms, 100% recall  ← Faster AND perfect recall!

GPU flat search is competitive up to ~1M vectors.
```

### When to Use GPU Flat Search

✅ **Use flat search when:**
- Dataset < 1M vectors
- You need 100% recall (no approximation acceptable)
- Batch queries amortize GPU overhead
- Index update frequency is high (no structure to maintain)

❌ **Consider approximate methods when:**
- Dataset > 10M vectors
- Latency critical (single query < 1ms)
- Memory constrained

---

## The Technique: Two-Phase Search

### Phase 1: Distance Computation

Use the distance kernels from Chapter 2:

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:795-850

private func searchUnfiltered(query: [Float], k: Int) async throws -> [IndexSearchResult] {
    guard let kernel = fusedL2TopKKernel,
          let datasetBuffer = storage.buffer else {
        throw IndexError.gpuNotInitialized(operation: "search")
    }

    // Create query buffer
    let queryBuffer = try context.bufferFactory.makeBuffer(
        from: query,
        label: "search.query"
    )

    // Single fused operation: distance + top-K
    let parameters = FusedL2TopKParameters(
        numQueries: 1,
        numDataset: storage.allocatedSlots,
        dimension: configuration.dimension,
        k: k
    )

    let gpuResult = try await kernel.execute(
        queries: queryBuffer,
        dataset: datasetBuffer,
        parameters: parameters
    )

    // Convert to search results
    return gpuResult.results(for: 0).map { (index, distance) in
        IndexSearchResult(
            handle: handleAllocator.handle(for: UInt32(index))!,
            distance: distance
        )
    }
}
```

### Phase 2: Top-K Selection

After computing distances, select the K smallest:

```
Distance row for query 0:
[0.23, 1.45, 0.01, 0.89, 0.12, 2.34, 0.05, ...]
                 ↓
            Top-K Selection (K=3)
                 ↓
Results: [(index=2, dist=0.01), (index=6, dist=0.05), (index=4, dist=0.12)]
```

VectorAccelerate fuses these phases (see [3.3 Fused Distance+TopK](./03-Fused-Distance-TopK.md)).

---

## Batch Search: Amortizing Overhead

Single query has GPU launch overhead. Batching amortizes it:

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:664-700

public func search(
    queries: [[Float]],
    k: Int
) async throws -> [[IndexSearchResult]] {
    guard !queries.isEmpty else { return [] }

    // All queries in single GPU dispatch
    let flatQueries = queries.flatMap { $0 }
    let queryBuffer = try context.bufferFactory.makeBuffer(
        from: flatQueries
    )

    let parameters = FusedL2TopKParameters(
        numQueries: queries.count,     // Batch all queries
        numDataset: storage.allocatedSlots,
        dimension: configuration.dimension,
        k: k
    )

    let gpuResult = try await kernel.execute(
        queries: queryBuffer,
        dataset: datasetBuffer,
        parameters: parameters
    )

    // Unpack results per query
    return (0..<queries.count).map { queryIdx in
        gpuResult.results(for: queryIdx).compactMap { (index, distance) in
            guard let handle = handleAllocator.handle(for: UInt32(index)) else {
                return nil
            }
            return IndexSearchResult(handle: handle, distance: distance)
        }
    }
}
```

### Batch Size Impact

```
100 queries × 1M database:

Individual queries:
  100 × (50μs compute + 100μs overhead) = 15 ms

Batched queries:
  1 × (5ms compute + 100μs overhead) = 5.1 ms

Batching is 3× faster!
```

---

## Memory Layout for Batch Search

```
Queries [Q × D]:
┌─────────────────────────────────────────────┐
│ q₀: [f₀, f₁, f₂, ..., f₇₆₇]               │ ← Query 0
│ q₁: [f₀, f₁, f₂, ..., f₇₆₇]               │ ← Query 1
│ ...                                         │
│ qₙ: [f₀, f₁, f₂, ..., f₇₆₇]               │ ← Query Q-1
└─────────────────────────────────────────────┘

Database [N × D]:
┌─────────────────────────────────────────────┐
│ d₀: [f₀, f₁, f₂, ..., f₇₆₇]               │
│ d₁: [f₀, f₁, f₂, ..., f₇₆₇]               │
│ ...                                         │
│ dₙ: [f₀, f₁, f₂, ..., f₇₆₇]               │
└─────────────────────────────────────────────┘

Results [Q × K]:
┌─────────────────────────────────────────────┐
│ r₀: [(idx, dist), (idx, dist), ..., K items]│ ← Results for q₀
│ r₁: [(idx, dist), (idx, dist), ..., K items]│ ← Results for q₁
│ ...                                         │
└─────────────────────────────────────────────┘
```

---

## Handling Deleted Vectors

AcceleratedVectorIndex uses lazy deletion with a mask:

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:837-849

// Convert GPU results to IndexSearchResults, filtering deleted vectors
var results: [IndexSearchResult] = []
results.reserveCapacity(effectiveK)

for (rawIndex, distance) in gpuResult.results(for: 0) {
    // Skip deleted vectors
    if deletionMask.isDeleted(rawIndex) { continue }

    // Get valid handle
    guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else {
        continue
    }

    results.append(IndexSearchResult(handle: handle, distance: distance))

    if results.count >= effectiveK { break }
}
```

### Over-fetching Strategy

When vectors may be deleted, fetch more than K:

```swift
// Request more results to account for deleted vectors
let fetchK = min(effectiveK + deletionMask.deletedCount, storage.allocatedSlots)
```

---

## 🔗 VectorCore Connection

VectorCore's distance functions are the building blocks:

```swift
// VectorCore: Single distance computation
let dist = l2DistanceSquared(query, vector)

// VectorAccelerate: GPU computes ALL distances in parallel
// Same algorithm, thousands of threads
```

---

## 🔗 VectorIndex Connection

VectorIndex's FlatIndex is the CPU equivalent:

```swift
// VectorIndex: FlatIndex search (CPU)
public func search(query: Vector<D>, k: Int) -> [SearchResult] {
    var distances: [(Int, Float)] = []

    for (i, vector) in vectors.enumerated() {
        let dist = distance(query, vector)
        distances.append((i, dist))
    }

    return distances
        .sorted { $0.1 < $1.1 }
        .prefix(k)
        .map { SearchResult(index: $0.0, distance: $0.1) }
}
```

VectorAccelerate achieves the same result with GPU parallelism:

```swift
// VectorAccelerate: GPU flat search
let results = try await index.search(query: query, k: k)
// Same output, 50-100× faster
```

---

## Performance Characteristics

### Scaling with Database Size

```
100 queries × N database × 768D (M2 Max):

N = 10K:    0.3 ms (CPU: 7 ms)    → 23× speedup
N = 100K:   1.2 ms (CPU: 68 ms)   → 57× speedup
N = 1M:     12 ms (CPU: 680 ms)   → 57× speedup
N = 10M:    120 ms (CPU: 6.8 s)   → 57× speedup

Speedup is relatively constant - both scale linearly with N.
GPU just has a much better constant factor.
```

### Scaling with Query Batch Size

```
Q queries × 1M database × 768D (M2 Max):

Q = 1:      50 ms (high overhead per query)
Q = 10:     52 ms (5.2 ms per query)
Q = 100:    85 ms (0.85 ms per query)
Q = 1000:   750 ms (0.75 ms per query)

Per-query latency improves with batch size!
```

---

## In VectorAccelerate

The high-level flat search API:

📍 See: `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:626-653`

```swift
/// Search for nearest neighbors.
public func search(
    query: consuming [Float],
    k: Int,
    filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)? = nil
) async throws -> [IndexSearchResult] {
    guard query.count == configuration.dimension else {
        throw IndexError.dimensionMismatch(
            expected: configuration.dimension,
            got: query.count
        )
    }

    guard handleAllocator.occupiedCount > 0 else {
        return []
    }

    // Route to appropriate implementation
    guard let filter = filter else {
        return try await searchUnfiltered(query: query, k: k)
    }

    return try await searchFiltered(query: query, k: k, filter: filter)
}
```

---

## Key Takeaways

1. **GPU flat search is competitive**: Often faster than CPU approximate for < 1M vectors

2. **Batch queries for efficiency**: Amortize GPU launch overhead

3. **Handle deletions with over-fetching**: Fetch K + deletedCount, filter on CPU

4. **Fused kernels are faster**: Combine distance and selection (next guide)

5. **100% recall is free**: No approximation trade-off needed

---

## Next Up

Let's look at how Top-K selection is parallelized on GPU:

**[→ 3.2 Top-K Selection on GPU](./02-TopK-On-GPU.md)**

---

*Guide 3.1 of 3.4 • Chapter 3: Accelerated Search*
