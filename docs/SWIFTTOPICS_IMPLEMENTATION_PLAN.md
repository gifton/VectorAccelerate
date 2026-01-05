# VectorAccelerate Enhancement Plan for SwiftTopics

**Document Version**: 1.0
**Date**: January 2026
**Status**: Approved for implementation

---

## Context

SwiftTopics is a pure-Swift topic modeling library implementing BERTopic-style algorithms (HDBSCAN, UMAP, c-TF-IDF, NPMI). They already use VectorAccelerate for:
- `L2DistanceKernel` - Pairwise distances
- `FusedL2TopKKernel` - k-NN search (also computes core distances)
- `MatrixMultiplyKernel` - PCA covariance/projection
- `StatisticsKernel` - Mean centering

### SwiftTopics Scale & Requirements

| Parameter | Value |
|-----------|-------|
| Typical corpus | ~500 documents |
| Maximum corpus | ~5,000 documents |
| Embedding dimensions | 384-dim typical |
| Vocabulary size | <10,000 terms |
| Core distances | Pre-computed via FusedL2TopKKernel |

### Performance Bottleneck Analysis (from SwiftTopics)

| Component | Complexity | CPU Time (1K docs) | CPU Time (10K docs) |
|-----------|------------|-------------------|---------------------|
| **MutualReachability** | O(n²) | ~500ms | ~50s |
| UMAP Gradients | O(epochs × k × n) | ~300ms | ~3s |
| c-TF-IDF | O(vocab × topics) | ~10ms | ~50ms |
| NPMI Coherence | O(keywords² × topics) | <1ms | <5ms |

**Clear winner**: MutualReachability is the #1 bottleneck due to O(n²) scaling.

---

## Implementation Plan

### Phase 1: MutualReachabilityKernel + GPU MST (Priority 1)

**Formula**: `mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))`

**Design Decisions** (based on SwiftTopics feedback):
- Core distances are **pre-computed** via FusedL2TopKKernel
- Sparse mode is **primary** (pairs for MST construction)
- Dense mode **also needed** (full pairwise then filter approach)
- **Include Borůvka's MST algorithm** for end-to-end GPU acceleration
- Auto-select dimension-optimized variants (like L2DistanceKernel)

---

#### 1.1 MutualReachability Metal Shader

```
Sources/VectorAccelerate/Shaders/MutualReachability.metal
```

**Kernels**:
1. `mutual_reachability_dense_kernel` - N×N output matrix
2. `mutual_reachability_sparse_kernel` - Compute for specific pairs
3. `mutual_reachability_384_kernel` - Dimension-optimized (most common)
4. `mutual_reachability_512_kernel`, `768_kernel`, `1536_kernel`

---

#### 1.2 Borůvka's MST Algorithm (GPU)

Borůvka's is GPU-friendly because each iteration processes ALL components in parallel:

```
Algorithm:
1. Initialize: Each point is its own component
2. Repeat until single component:
   a. For each component, find minimum outgoing edge (parallel)
   b. Contract edges, merge components
   c. ~log(n) iterations total
```

**GPU Kernels needed**:
1. `boruvka_find_min_edge_kernel` - Find min edge per component
2. `boruvka_contract_kernel` - Merge components via edge contraction

```
Sources/VectorAccelerate/Shaders/BoruvkaMST.metal
```

**Swift API**:
```swift
public struct BoruvkaMSTKernel {
    /// Compute MST using Borůvka's algorithm with mutual reachability distances
    public func computeMST(
        embeddings: MTLBuffer,     // [N, D]
        coreDistances: MTLBuffer,  // [N]
        n: Int,
        d: Int
    ) async throws -> MSTResult

    /// Incremental MST for new points
    public func updateMST(
        existing: MSTResult,
        newEmbeddings: MTLBuffer,
        newCoreDistances: MTLBuffer
    ) async throws -> MSTResult
}

public struct MSTResult {
    /// Edges in MST: [(source, target, weight)]
    public let edges: [(Int, Int, Float)]
    /// Total MST weight
    public let totalWeight: Float
}
```

---

#### 1.3 Combined HDBSCAN Distance Module

High-level API that combines the kernels:

```swift
public struct HDBSCANDistanceModule {
    /// Full pipeline: core distances → mutual reachability → MST
    public func computeMST(
        embeddings: [[Float]],
        minSamples: Int = 5  // k for core distance
    ) async throws -> MSTResult
}
```

---

#### Files to Create/Modify

| File | Action |
|------|--------|
| `Sources/VectorAccelerate/Shaders/MutualReachability.metal` | CREATE |
| `Sources/VectorAccelerate/Shaders/BoruvkaMST.metal` | CREATE |
| `Sources/VectorAccelerate/Kernels/Distance/MutualReachabilityKernel.swift` | CREATE |
| `Sources/VectorAccelerate/Kernels/Graph/BoruvkaMSTKernel.swift` | CREATE |
| `Sources/VectorAccelerate/Modules/HDBSCANDistanceModule.swift` | CREATE |
| `Tests/VectorAccelerateTests/MutualReachabilityTests.swift` | CREATE |
| `Tests/VectorAccelerateTests/BoruvkaMSTTests.swift` | CREATE |

---

### Phase 2: UMAPGradientKernel (Priority 2)

**Challenge**: Requires atomic operations (new pattern for VectorAccelerate)

**Recommended Approach**: Segmented reduction (avoid atomics)
1. Sort edges by source vertex
2. Compute gradients per edge (parallel)
3. Segment-reduce by source (parallel via prefix sums)

This avoids atomic float add contention while maintaining GPU parallelism.

#### Files to Create/Modify

| File | Action |
|------|--------|
| `Sources/VectorAccelerate/Shaders/UMAPGradient.metal` | CREATE |
| `Sources/VectorAccelerate/Kernels/ML/UMAPGradientKernel.swift` | CREATE |
| `Tests/VectorAccelerateTests/UMAPGradientTests.swift` | CREATE |

---

### Phase 3: Utility Kernels (Priority 3)

Simple additions that support the main kernels:

| Kernel | Purpose | Complexity |
|--------|---------|------------|
| `BatchMax3Kernel` | `max(a, b, c)` elementwise | LOW |
| `LogSumExpKernel` | Numerically stable softmax | LOW |

---

### Phase 4: SparseLogTFIDFKernel (Priority 4)

Given c-TF-IDF is only ~50ms for 10K docs, this is **low priority**.
Consider lightweight implementation without full sparse matrix infrastructure.

---

### Phase 5: CooccurrenceCountKernel (Priority 5 / Optional)

Given NPMI is <5ms for 10K docs and vocab <10K, this may not need GPU.
**Recommendation**: Keep on CPU unless profiling shows otherwise.
Small vocab + exact counts = CPU is likely faster than atomic contention

---

## Expected Performance Improvement

### MutualReachabilityKernel (Primary Win)

| Corpus Size | Current CPU | Expected GPU | Speedup |
|-------------|-------------|--------------|---------|
| 500 docs | ~125ms | ~5ms | ~25x |
| 1,000 docs | ~500ms | ~15ms | ~33x |
| 5,000 docs | ~12.5s | ~100ms | ~125x |

**Conservative estimates** based on L2DistanceKernel benchmarks. The mutual reachability kernel adds only a max operation on top of L2, so overhead is minimal.

### Memory Requirements

| Mode | 1K docs | 5K docs | 10K docs |
|------|---------|---------|----------|
| Dense N×N | 4 MB | 100 MB | 400 MB |
| Sparse (15-NN) | 60 KB | 300 KB | 600 KB |

Sparse mode is memory-efficient and recommended for MST construction.

---

## Summary

**Confirmed scope for Phase 1** (HDBSCAN acceleration):

| Component | Description |
|-----------|-------------|
| `MutualReachabilityKernel` | Dense + sparse modes, dimension-optimized |
| `BoruvkaMSTKernel` | GPU-parallel MST via Borůvka's algorithm |
| `HDBSCANDistanceModule` | High-level API combining both |

**Files to create**: 7 files (2 shaders, 3 Swift kernels, 2 test files)

**Deferred to future phases**:
- UMAPGradientKernel (needs atomic/segmented decision)
- SparseLogTFIDFKernel (low priority)
- CooccurrenceCountKernel (likely stays CPU)

---

## Verification Checklist

After implementation, verify:
- [ ] Dense mode produces correct N×N matrix
- [ ] Sparse mode matches dense mode for same pairs
- [ ] Borůvka's produces valid MST (n-1 edges, connected)
- [ ] MST weights match CPU Prim's implementation
- [ ] Performance: 10-50x speedup over CPU at 1K docs
- [ ] Memory: Sparse mode uses <1MB for 5K docs

---

## Reference Documents

- Original proposals: `docs/VECTORACCELERATE_PROPOSALS.md`
- SwiftTopics SPEC: `/Users/goftin/dev/real/GournalV2/SwiftTopics/SPEC.md`

---

*Plan created January 2026*
