# VectorAccelerate GPU Acceleration Opportunities

> **Status:** Proposal
> **Priority:** P2 (After SwiftTopics algorithmic fixes)
> **Target:** VectorAccelerate 0.4.0+

## Overview

This document outlines GPU acceleration opportunities in VectorAccelerate that would benefit SwiftTopics cluster extraction and post-processing phases. These are **secondary optimizations**—the primary bottleneck is algorithmic complexity in SwiftTopics (see [CLUSTER_EXTRACTION_OPTIMIZATION.md](./CLUSTER_EXTRACTION_OPTIMIZATION.md)).

After fixing the O(n²) algorithmic issues, these GPU opportunities could provide additional 2-10x speedups for specific operations.

---

## Context: Current GPU Usage in HDBSCAN

VectorAccelerate already provides excellent GPU acceleration for:

| Phase | Kernel | Speedup |
|-------|--------|---------|
| Core distances | `computeKNNDistances` | 40-100x |
| Mutual reachability | `MutualReachabilityKernel` | 50x |
| MST construction | `BoruvkaMSTKernel` | 40-100x |

The cluster extraction phase is currently CPU-only because it involves tree traversal, which is inherently sequential. However, some sub-operations are parallelizable.

---

## Opportunity 1: Batch Centroid Distance Computation

### Use Case

Computing outlier scores requires calculating distances from each point to its cluster's centroid.

```swift
// Current: ClusterExtraction.swift - outlier scoring
for i in 0..<n {
    let clusterLabel = pointLabels[i]
    // Compute average core distance in cluster (nested loops)
    for (clusterID, points) in clusterPoints {
        if clusterIDToLabel[clusterID] == clusterLabel {
            for pointIdx in points {
                clusterCoreSum += coreDistances[pointIdx]
            }
        }
    }
}
```

### Proposed VectorAccelerate API

```swift
/// Computes distances from each point to its assigned cluster centroid
///
/// - Parameters:
///   - embeddings: Point embeddings [N × D]
///   - clusterLabels: Cluster assignment for each point [N], -1 = outlier
///   - clusterCount: Number of clusters (excluding outliers)
/// - Returns: Distance to centroid for each point [N], outliers get max distance
public func computeClusterCentroidDistances(
    embeddings: MTLBuffer,  // [N × D] Float32
    clusterLabels: MTLBuffer,  // [N] Int32
    clusterCount: Int,
    dimension: Int
) async throws -> [Float]
```

### Metal Kernel Sketch

```metal
// Phase 1: Compute cluster centroids (parallel reduction per cluster)
kernel void computeClusterCentroids(
    device const float* embeddings [[buffer(0)]],    // [N × D]
    device const int* labels [[buffer(1)]],          // [N]
    device atomic<float>* centroidSums [[buffer(2)]], // [K × D]
    device atomic<int>* clusterCounts [[buffer(3)]],  // [K]
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;

    int label = labels[tid];
    if (label < 0) return;  // Skip outliers

    // Atomically accumulate to cluster centroid
    for (uint d = 0; d < D; d++) {
        float val = embeddings[tid * D + d];
        atomic_fetch_add_explicit(&centroidSums[label * D + d], val, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&clusterCounts[label], 1, memory_order_relaxed);
}

// Phase 2: Normalize centroids (K threads)
kernel void normalizeCentroids(
    device float* centroidSums [[buffer(0)]],
    device const int* clusterCounts [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= K) return;

    int count = clusterCounts[tid];
    if (count == 0) return;

    float invCount = 1.0f / float(count);
    for (uint d = 0; d < D; d++) {
        centroidSums[tid * D + d] *= invCount;
    }
}

// Phase 3: Compute distances to centroids (N threads)
kernel void computeDistancesToCentroids(
    device const float* embeddings [[buffer(0)]],     // [N × D]
    device const float* centroids [[buffer(1)]],      // [K × D]
    device const int* labels [[buffer(2)]],           // [N]
    device float* distances [[buffer(3)]],            // [N]
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& maxDistance [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;

    int label = labels[tid];
    if (label < 0) {
        distances[tid] = maxDistance;
        return;
    }

    float dist = 0.0f;
    for (uint d = 0; d < D; d++) {
        float diff = embeddings[tid * D + d] - centroids[label * D + d];
        dist += diff * diff;
    }
    distances[tid] = sqrt(dist);
}
```

### Expected Speedup

| Points | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| 1,000 | ~5ms | ~0.5ms | 10x |
| 5,000 | ~50ms | ~2ms | 25x |
| 10,000 | ~200ms | ~5ms | 40x |

### Priority

**P2** - This is a small portion of the overall extraction time. Worth adding after algorithmic fixes are complete.

---

## Opportunity 2: Parallel Stability Accumulation

### Use Case

Stability computation sums contributions from all leaf descendants. After the algorithmic fix (bottom-up DP), each node's stability is computed by iterating over its leaves.

```swift
// After Fix 3, this loop is O(n) total but could still benefit from GPU
for leafDeathDistance in leafInfo.deathDistances {
    let contribution = max(0, lambdaBirth - effectiveLambdaDeath)
    stability += contribution
}
```

### Proposed VectorAccelerate API

```swift
/// Computes stability contributions for all nodes in parallel
///
/// - Parameters:
///   - birthLevels: Birth level for each node [M]
///   - deathLevels: Death level for each node [M]
///   - leafDeathDistances: Flattened array of leaf death distances
///   - leafOffsets: Start offset into leafDeathDistances for each node [M+1]
/// - Returns: Stability score for each node [M]
public func computeStabilities(
    birthLevels: MTLBuffer,
    deathLevels: MTLBuffer,
    leafDeathDistances: MTLBuffer,
    leafOffsets: MTLBuffer,  // CSR-style offsets
    nodeCount: Int
) async throws -> [Float]
```

### Metal Kernel Sketch

```metal
kernel void computeStabilityContributions(
    device const float* birthLevels [[buffer(0)]],
    device const float* deathLevels [[buffer(1)]],
    device const float* leafDeaths [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device float* stabilities [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    float birth = birthLevels[tid];
    float death = deathLevels[tid];

    // Convert to lambda space
    float lambdaBirth = birth > 1e-7f ? 1.0f / birth : INFINITY;
    float lambdaDeath = (death > 1e-7f && death < INFINITY) ? 1.0f / death : 0.0f;

    uint start = offsets[tid];
    uint end = offsets[tid + 1];

    float stability = 0.0f;
    for (uint i = start; i < end; i++) {
        float leafDeath = leafDeaths[i];
        float leafLambdaDeath = (leafDeath > 1e-7f && leafDeath < INFINITY)
            ? 1.0f / leafDeath : 0.0f;
        float effective = max(leafLambdaDeath, lambdaDeath);
        stability += max(0.0f, lambdaBirth - effective);
    }

    stabilities[tid] = stability;
}
```

### Expected Speedup

| Internal Nodes | CPU Time | GPU Time | Speedup |
|----------------|----------|----------|---------|
| 1,000 | ~2ms | ~0.3ms | 7x |
| 5,000 | ~15ms | ~1ms | 15x |

### Priority

**P3** - Low priority. The CPU version after algorithmic fixes will be fast enough (~15ms for 5000 points).

---

## Opportunity 3: Batched Soft Membership Probabilities

### Use Case

`MembershipProbabilityCalculator.compute()` calculates probability distributions over all clusters for each point.

```swift
// Current: O(n × k × d) nested loops
for i in 0..<n {
    for c in 0..<k {
        for d in 0..<dimension {
            let diff = embeddings[i].vector[d] - centroids[c][d]
            dist += diff * diff
        }
    }
}
```

### Proposed VectorAccelerate API

This is essentially an all-pairs distance matrix followed by softmax normalization. VectorAccelerate may already support this via existing APIs:

```swift
/// Computes soft cluster membership probabilities using distance-based softmax
///
/// - Parameters:
///   - embeddings: Point embeddings [N × D]
///   - centroids: Cluster centroids [K × D]
///   - temperature: Softmax temperature (lower = sharper probabilities)
/// - Returns: Probability matrix [N × K]
public func computeSoftMembership(
    embeddings: MTLBuffer,
    centroids: MTLBuffer,
    pointCount: Int,
    clusterCount: Int,
    dimension: Int,
    temperature: Float = 1.0
) async throws -> [[Float]]
```

### Implementation Notes

This is a composition of existing kernels:
1. **Distance matrix**: Use existing `computeDistanceMatrix()` for N×K distances
2. **Row-wise softmax**: Apply softmax normalization per row

### Expected Speedup

| N | K | CPU Time | GPU Time | Speedup |
|---|---|----------|----------|---------|
| 1,000 | 10 | ~20ms | ~1ms | 20x |
| 5,000 | 20 | ~200ms | ~5ms | 40x |

### Priority

**P2** - Moderate priority. This is useful for soft clustering scenarios but not critical for hard assignment.

---

## Opportunity 4: Parallel Union-Find (Research)

### Background

The hierarchy building phase uses Union-Find for cluster merging. Traditional Union-Find is sequential, but there's research on parallel/GPU Union-Find:

- **Shiloach-Vishkin algorithm**: O(log n) parallel iterations
- **Label propagation**: Iterative GPU-friendly approach
- **Concurrent Union-Find**: Lock-free with atomic operations

### Current Bottleneck

```swift
// ClusterHierarchyBuilder.swift - sequential merging
for edge in sortedEdges {  // Must process in order!
    clusterState.mergePoints(edge.source, edge.target, atDistance: edge.weight)
}
```

### GPU Opportunity

While full parallel Union-Find is complex, **batch processing** is feasible:

1. Group edges with same/similar weights
2. Process each batch in parallel (no ordering constraints within batch)
3. Synchronize between batches

### Proposed Approach

```swift
/// GPU-accelerated connected components using label propagation
///
/// - Parameters:
///   - edges: MST edges sorted by weight
///   - pointCount: Number of points
/// - Returns: Component labels for each point at each merge step
public func parallelConnectedComponents(
    edges: [(source: Int, target: Int, weight: Float)],
    pointCount: Int
) async throws -> [[Int]]
```

### Expected Speedup

Unknown - requires prototyping. Potential for 5-20x on large graphs.

### Priority

**P4** - Research priority. The current hierarchy building (~5s for 5000 points) is acceptable after other fixes.

---

## Summary: Priority Matrix

| Opportunity | Priority | Effort | Impact | Dependency |
|-------------|----------|--------|--------|------------|
| 1. Centroid Distances | P2 | Medium | Low | SwiftTopics Fix 4 |
| 2. Stability Accumulation | P3 | Medium | Very Low | SwiftTopics Fix 3 |
| 3. Soft Membership | P2 | Low (existing kernels) | Medium | None |
| 4. Parallel Union-Find | P4 | High (research) | Unknown | None |

---

## Recommended Implementation Order

1. **Wait for SwiftTopics algorithmic fixes** (100x improvement)
2. **Profile post-optimization bottlenecks** to identify actual GPU opportunities
3. **Implement Opportunity 3** (Soft Membership) - low effort, reuses existing kernels
4. **Implement Opportunity 1** (Centroid Distances) - moderate effort, clear API
5. **Evaluate Opportunities 2 & 4** based on profiling data

---

## Integration with SwiftTopics

Once VectorAccelerate provides these APIs, SwiftTopics would use them via `TopicsGPUContext`:

```swift
// Example integration
public actor ClusterExtractor {
    private let gpuContext: TopicsGPUContext?

    func computeOutlierScores(
        embeddings: [Embedding],
        clusterLabels: [Int],
        clusterCount: Int
    ) async throws -> [Float] {
        if let gpu = gpuContext {
            // Use GPU acceleration
            return try await gpu.computeClusterCentroidDistances(
                embeddings: embeddings,
                clusterLabels: clusterLabels,
                clusterCount: clusterCount
            )
        } else {
            // CPU fallback
            return computeOutlierScoresCPU(...)
        }
    }
}
```

---

## Related Documents

- [CLUSTER_EXTRACTION_OPTIMIZATION.md](./CLUSTER_EXTRACTION_OPTIMIZATION.md) - Primary optimization plan (algorithmic)
- VectorAccelerate kernel specs (in `.build/checkouts/VectorAccelerate/docs/kernel-specs/`)
- [BENCHMARK_SUITE_PLAN.md](./BENCHMARK_SUITE_PLAN.md) - Benchmarking framework
