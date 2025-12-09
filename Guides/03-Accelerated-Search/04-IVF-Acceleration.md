# 3.4 IVF Acceleration

> **GPU-accelerated approximate search—scaling beyond what flat search can handle.**

---

## The Concept

IVF (Inverted File Index) partitions vectors into clusters:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          IVF STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Centroids: [c₀, c₁, c₂, ..., c₂₅₅]   (256 cluster centers)        │
│                                                                      │
│  Inverted Lists:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Cluster 0:   [v₁₂, v₄₅, v₈₉, ...]     ~4000 vectors           │ │
│  │ Cluster 1:   [v₃, v₂₇, v₁₀₃, ...]     ~4000 vectors           │ │
│  │ Cluster 2:   [v₇, v₅₆, v₂₀₁, ...]     ~4000 vectors           │ │
│  │ ...                                                             │ │
│  │ Cluster 255: [v₁₅, v₉₈, v₃₄₅, ...]    ~4000 vectors           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Search: Query → Find nearest centroids → Search those clusters     │
└─────────────────────────────────────────────────────────────────────┘
```

Instead of searching all N vectors, IVF searches only `nprobe` clusters.

---

## Why It Matters

For large datasets, IVF provides a better tradeoff:

```
Dataset: 10M vectors × 768D, K=10

Flat Search (GPU):
  Time: ~120 ms per query batch
  Recall: 100%

IVF Search (GPU, nprobe=16 of 256 clusters):
  Time: ~8 ms per query batch
  Recall: ~95%
  Speedup: 15×
```

IVF trades some recall for much faster search.

---

## The Technique: Two-Phase GPU Search

### Phase 1: Centroid Search

Find the `nprobe` nearest cluster centroids:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift

/// Find nearest centroids for query vectors
private func findNearestCentroids(
    queries: [[Float]],
    structure: IVFGPUStructure
) async throws -> [[Int]] {
    // Compute distances to all centroids
    let centroidDistances = try await distanceKernel.compute(
        queries: queries,
        database: structure.centroids,
        computeSqrt: false
    )

    // Select top nprobe centroids per query
    let nearestCentroids = try await topKKernel.select(
        distances: centroidDistances,
        k: configuration.nprobe
    )

    return nearestCentroids.map { $0.map { $0.index } }
}
```

This is a small flat search: Q queries × 256 centroids.

### Phase 2: List Search

Search vectors in selected clusters:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift

/// Search within selected inverted lists
private func searchLists(
    queries: [[Float]],
    nearestCentroids: [[Int]],
    structure: IVFGPUStructure,
    k: Int
) async throws -> IVFSearchResult {
    // Gather vectors from selected clusters
    var candidateIndices: [[Int]] = []
    var candidateVectors: [[[Float]]] = []

    for (queryIdx, centroids) in nearestCentroids.enumerated() {
        var queryIndices: [Int] = []
        var queryVectors: [[Float]] = []

        for centroidIdx in centroids {
            let list = structure.invertedLists[centroidIdx]
            queryIndices.append(contentsOf: list.indices)
            queryVectors.append(contentsOf: list.vectors)
        }

        candidateIndices.append(queryIndices)
        candidateVectors.append(queryVectors)
    }

    // Batch distance computation for all candidates
    // (This is the expensive part - GPU accelerates it)
    let results = try await computeDistancesAndSelectTopK(
        queries: queries,
        candidates: candidateVectors,
        indices: candidateIndices,
        k: k
    )

    return results
}
```

---

## GPU Acceleration Points

IVF has multiple GPU acceleration opportunities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IVF ACCELERATION POINTS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CENTROID DISTANCE (GPU)                                         │
│     Q queries × 256 centroids × D                                   │
│     Small but parallelizable                                        │
│                                                                      │
│  2. TOP-NPROBE SELECTION (GPU)                                      │
│     Select nprobe nearest centroids                                 │
│     Warp-optimized selection                                        │
│                                                                      │
│  3. CANDIDATE DISTANCE (GPU) ← Main speedup!                        │
│     Q queries × (~nprobe × N/nlist) candidates × D                  │
│     This is where most time is spent                                │
│                                                                      │
│  4. FINAL TOP-K SELECTION (GPU)                                     │
│     Select K from candidates per query                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## IVF Training on GPU

K-Means clustering can also be GPU-accelerated:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Internal/IVFStructure.swift:165-212

public func train(vectors: [[Float]], context: Metal4Context) async throws {
    guard !isTrained else { return }
    guard vectors.count >= numClusters else {
        throw IndexError.invalidInput(
            message: "Need at least \(numClusters) vectors for training"
        )
    }

    // GPU-accelerated K-Means
    let kmeans = try await GPUKMeans(
        context: context,
        k: numClusters,
        dimension: dimension,
        maxIterations: 20
    )

    // Train on GPU
    centroids = try await kmeans.fit(vectors: vectors)

    // Assign vectors to clusters (also on GPU)
    let assignments = try await kmeans.predict(vectors: vectors)

    // Build inverted lists
    for (vectorIdx, clusterIdx) in assignments.enumerated() {
        invertedLists[clusterIdx].append(
            index: vectorIdx,
            vector: vectors[vectorIdx]
        )
    }

    isTrained = true
}
```

---

## Memory Layout for IVF

Efficient GPU access requires careful memory organization:

```
GPU Buffers:

Centroids [nlist × D]:
┌─────────────────────────────────────────────┐
│ c₀: [f₀, f₁, ..., f₇₆₇]                    │
│ c₁: [f₀, f₁, ..., f₇₆₇]                    │
│ ...                                         │
│ c₂₅₅: [f₀, f₁, ..., f₇₆₇]                  │
└─────────────────────────────────────────────┘

Inverted List Offsets [nlist + 1]:
┌─────────────────────────────────────────────┐
│ [0, 4123, 8201, 12456, ...]                 │
│  ↑     ↑      ↑       ↑                     │
│ list0 list1 list2   list3 ...              │
│ start start start  start                    │
└─────────────────────────────────────────────┘

Inverted List Data [N × D]:
┌─────────────────────────────────────────────┐
│ List 0 vectors (contiguous)                 │
│ List 1 vectors (contiguous)                 │
│ ...                                         │
└─────────────────────────────────────────────┘

Inverted List Indices [N]:
┌─────────────────────────────────────────────┐
│ Original indices for each vector            │
└─────────────────────────────────────────────┘
```

---

## IVF Search Pipeline

The complete pipeline:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift

public func search(
    queries: [[Float]],
    structure: IVFGPUStructure,
    k: Int
) async throws -> IVFSearchResult {
    // Phase 1: Find nearest centroids (GPU)
    let nearestCentroids = try await findNearestCentroids(
        queries: queries,
        structure: structure
    )

    // Phase 2: Gather candidate vectors from selected lists
    let candidates = gatherCandidates(
        nearestCentroids: nearestCentroids,
        structure: structure
    )

    // Phase 3: Compute distances to candidates (GPU - main cost)
    let distances = try await computeCandidateDistances(
        queries: queries,
        candidates: candidates
    )

    // Phase 4: Select final top-K (GPU)
    return try await selectTopK(
        distances: distances,
        candidates: candidates,
        k: k
    )
}
```

---

## 🔗 VectorCore Connection

VectorCore provides the distance primitives:

```swift
// VectorCore distance used in IVF centroid search
let centroidDistances = centroids.map { centroid in
    l2DistanceSquared(query, centroid)
}
```

VectorAccelerate GPU-accelerates this for batch queries.

---

## 🔗 VectorIndex Connection

VectorIndex's IVFIndex follows the same algorithm:

```swift
// VectorIndex: CPU IVF search
public func search(query: [Float], k: Int) -> [SearchResult] {
    // Find nearest centroids
    let nearestCentroids = centroids
        .enumerated()
        .map { ($0.offset, distance(query, $0.element)) }
        .sorted { $0.1 < $1.1 }
        .prefix(nprobe)
        .map { $0.0 }

    // Search selected lists
    var candidates: [(Int, Float)] = []
    for centroidIdx in nearestCentroids {
        for (idx, vector) in lists[centroidIdx] {
            candidates.append((idx, distance(query, vector)))
        }
    }

    return candidates.sorted { $0.1 < $1.1 }.prefix(k)
}
```

VectorAccelerate achieves the same with GPU parallelism.

---

## Using AcceleratedVectorIndex with IVF

```swift
import VectorAccelerate

// Create IVF index
let index = try await AcceleratedVectorIndex(
    configuration: .ivf(
        dimension: 768,
        nlist: 256,      // Number of clusters
        nprobe: 16,      // Clusters to search
        capacity: 1_000_000
    )
)

// Insert vectors (auto-trains when enough data)
for vector in trainingVectors {
    _ = try await index.insert(vector)
}

// Or manually trigger training
try await index.train()

// Search uses IVF automatically
let results = try await index.search(query: queryVector, k: 10)
```

---

## Performance Tuning

### nlist (Number of Clusters)

```
Rule of thumb: nlist ≈ √N to 4×√N

N = 1M → nlist = 1000-4000
N = 10M → nlist = 3000-12000

More clusters = faster search, but:
- More memory for centroids
- Smaller lists (less parallelism)
- Potentially lower recall
```

### nprobe (Clusters to Search)

```
Recall vs. Speed tradeoff:

nprobe = 1:   Very fast, ~60% recall
nprobe = 8:   Fast, ~90% recall
nprobe = 16:  Moderate, ~95% recall
nprobe = 32:  Slower, ~98% recall
nprobe = 64:  Much slower, ~99% recall

Start with nprobe = nlist/16, adjust based on recall needs
```

---

## Hybrid CPU/GPU Strategy

For some workloads, a hybrid approach works best:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID IVF STRATEGY                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Single query, low latency needed:                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CPU: Centroid search (256 distances is fast on CPU)        │    │
│  │  CPU: List gathering                                         │    │
│  │  GPU: Candidate distance (if >10K candidates)               │    │
│  │  CPU: Final top-K                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Batch queries, throughput focus:                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  GPU: All phases                                             │    │
│  │  Amortize overhead across queries                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **IVF enables scale**: Search 10M+ vectors efficiently

2. **Two-phase search**: Centroid search → List search

3. **GPU accelerates both phases**: But list search is the main win

4. **nlist/nprobe tune recall vs. speed**: More probes = better recall, slower search

5. **GPU K-Means for training**: Cluster training is also parallelizable

---

## Chapter Summary

You've learned how VectorAccelerate accelerates search:

- ✅ GPU flat search for perfect recall
- ✅ Parallel Top-K selection algorithms
- ✅ Fused kernels to avoid memory bottlenecks
- ✅ IVF acceleration for large-scale approximate search

Next, we'll dive into memory management—the key to production performance.

**[→ Chapter 4: Memory Management](../04-Memory-Management/README.md)**

---

*Guide 3.4 of 3.4 • Chapter 3: Accelerated Search*
