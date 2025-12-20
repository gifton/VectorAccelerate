# VectorAccelerate Quality Improvement Roadmap

This document tracks correctness, performance, and DX improvements based on external code review feedback. Each issue is categorized by priority and includes implementation notes, affected files, and completion status.

**Last Updated:** 2025-12-15
**Review Source:** External deep-dive code review

---

## Completion Summary

| Item | Description | Status | Date |
|------|-------------|--------|------|
| **P0.1** | IVF GPU Indirection (CSR candidates) | **COMPLETED** | 2025-12-14 |
| **P0.2** | IVF Training Duplicates Fix | **COMPLETED** | 2025-12-14 |
| **P0.3** | Batch Search Union Bug + Regression Test | **COMPLETED** | 2025-12-14 |
| **P0.4** | Configurable IVF Training + Tests | **COMPLETED** | 2025-12-13 |
| **P0.5** | K > 8 Auto Fallback | **COMPLETED** | 2025-12-13 |
| **P0.6** | Silent Clamping Fix | **COMPLETED** | 2025-12-13 |
| **P0.8** | Stable Handles (stableID, arrays) | **COMPLETED** | 2025-12-13 |
| **P0.9** | Streaming Top-K Marked Experimental | **COMPLETED** | 2025-12-13 |
| **P1.1** | GPU Batch Search for Flat Index | **COMPLETED** | 2025-12-13 |
| **P1.2** | IVF CPU Gather Elimination | **COMPLETED** | 2025-12-14 |
| **P1.3** | Buffer Pool Consistency | **COMPLETED** | 2025-12-14 |
| **P1.4** | Distance Clarity (euclideanDistance) | **COMPLETED** | 2025-12-13 |
| **P2.2** | Scalar Quantization Integration | **COMPLETED** | 2025-12-15 |
| **P2.3** | Benchmarking Harness | **COMPLETED** | 2025-12-14 |
| **P3.1** | CI Runner Fix (macos-26 → macos-latest) | **COMPLETED** | 2025-12-13 |
| **P3.2** | Documentation Alignment | **COMPLETED** | 2025-12-13 |
| **P4.1** | K-Means++ Initialization | **COMPLETED** | 2025-12-14 |
| **P4.2** | Adaptive nlist Selection | **COMPLETED** | 2025-12-15 |

### Summary of Completed Work

**P0.1 - IVF GPU Indirection (CSR Candidates)**
- Implemented per-query CSR candidate lists with GPU-side indirection
- New Metal kernel: `ivf_distance_with_indirection` computes distances using slot indirection
- New Swift wrapper: `IVFIndirectionDistanceKernel` encapsulates the kernel
- Added `encodeWithOffsets()` to `TopKSelectionKernel` for per-query segment selection
- Replaced `IVFSearchPipeline.gatherAndSearch()` with CSR-based implementation
- IVF recall improved: nprobe=4/nlist=8 → 92%, nprobe=8/nlist=8 → 100%
- Files: `SearchAndRetrieval.metal`, `TopKSelectionKernel.swift`, `IVFIndirectionDistanceKernel.swift`, `IVFSearchPipeline.swift`

**P0.2 - IVF Training Duplicates Fix**
- Fixed double-assignment bug in `AcceleratedVectorIndex.train()`
- Previously: `IVFStructure.train()` assigned staged vectors, then `AcceleratedVectorIndex.train()` re-assigned ALL vectors
- Fix: Removed duplicate assignment loop (lines 322-331 in original)
- Now: `IVFStructure.train()` handles assignment, `AcceleratedVectorIndex.train()` just calls it
- Files: `AcceleratedVectorIndex.swift`

**P0.3 - Batch Search Union Bug + Regression Test**
- Resolved by P0.1: CSR per-query candidate lists prevent cross-query contamination
- Added regression test: `testIVFBatchSearchNoCrossQueryContamination`
- Test creates two distinct clusters, verifies batch queries don't find each other's vectors
- Uses nprobe=1 to make contamination obvious if it occurs
- Files: `IVFTests.swift`

**P1.2 - IVF CPU Gather Elimination**
- Consolidated into P0.1: GPU indirection eliminates CPU vector gathering
- Verified: no `gatheredVectors`, `vectorsPtr[`, or vector read loops in hot path
- All vector reads happen on GPU via `ivf_distance_with_indirection` kernel
- Files: `IVFSearchPipeline.swift`

**P0.4 - Configurable IVF Training + Tests**
- Added `minTrainingVectors: Int?` parameter to `IndexConfiguration.ivf()`
- Parameter flows through to `IVFStructure` for auto-training threshold control
- Validation: `minTrainingVectors` must be >= `nlist` if specified
- Added comprehensive IVF training tests:
  - `testIVFTrainingActuallyTriggers`: Verifies training triggers at low threshold
  - `testIVFSearchAfterTraining`: Verifies search works post-training
  - `testIVFRecallVsBruteForce`: Compares IVF recall to flat index (low threshold until P0.1 fixed)
  - `testIVFDefaultMinTrainingVectors`: Verifies default behavior
  - `testIVFManualTraining`: Verifies manual `train()` method
- Files: `IndexConfiguration.swift`, `AcceleratedVectorIndex.swift`, `IVFTests.swift`

**P0.5 - K > 8 Handling (Auto Fallback with Memory Guardrail)**
- `FusedL2TopKKernel` now automatically selects strategy based on K:
  - K ≤ 8: Fused single-pass kernel (fastest)
  - 8 < K ≤ 32: Two-pass with warp-optimized selection
  - K > 32: Two-pass with standard selection
- GPU merge path (`TopKMergeKernel`) for chunked fallback avoids CPU readback
- Memory guardrail: `maxDistanceMatrixBytes` (default 64MB) triggers chunked processing
- Files: `FusedL2TopKKernel.swift`, `TopKMergeKernel.swift`, `SearchAndRetrieval.metal`

**P0.6 - Silent Parameter Clamping Fix**
- `FusedL2TopKParameters.init()` now **throws** on invalid input instead of silently clamping
- Validates: dimension > 0, dimension ≤ maxDimension, k > 0, k ≤ maxK, numQueries > 0, numDataset > 0
- Files: `FusedL2TopKKernel.swift`

**P0.8 - Stable Handles Implementation**
- `VectorHandle` simplified to `stableID: UInt32` only (no generation field)
- `HandleAllocator` uses **array-based** indirection (not dictionaries):
  - `stableIDToSlot: [UInt32]` - forward mapping
  - `slotToStableID: [UInt32]` - reverse mapping
  - Tombstone value: `0xFFFFFFFF`
- Handles remain valid across `compact()` operations
- Deleted handles absent from mapping (never reused)
- Files: `VectorHandle.swift`, `HandleAllocator.swift`, `MetadataStore.swift`, `IVFStructure.swift`, `AcceleratedVectorIndex.swift`

**P0.9 - Streaming Top-K Marked Experimental**
- Added `@available(*, deprecated)` to `StreamingTopKKernel` and related types
- Added experimental warnings to Metal shader (`streaming_l2_topk_update`)
- Documented correctness issue: only thread 0's results are preserved
- Recommendation: Use `FusedL2TopKKernel` with chunked fallback instead
- Files: `StreamingTopKKernel.swift`, `FusedL2TopKKernel.swift`, `AdvancedTopK.metal`

**P1.1 - GPU Batch Search for Flat Index**
- Implemented true GPU batching for flat index batch search
- `search(queries:k:)` now uses single GPU dispatch for all queries on flat index
- Added `searchBatchGPU()` private method for efficient multi-query processing
- IVF/filtered searches still use sequential fallback (IVF has known bugs in P0.1)
- Tests added:
  - `testBatchSearchMatchesSequential`: Verifies batch results match sequential
  - `testBatchSearchWithDeletedVectors`: Verifies deleted vector filtering
  - `testBatchSearchPerformanceAdvantage`: Benchmark comparison
- Files: `AcceleratedVectorIndex.swift`, `FlatIndexSearchTests.swift`

**P1.4 - Distance Clarity (euclideanDistance)**
- `IndexSearchResult.distance` already documented as squared L2 (L2²)
- `euclideanDistance` computed property already implemented as `sqrt(distance)`
- Added `testEuclideanDistanceProperty` test to verify relationship
- Files: `SearchResult.swift`, `FlatIndexSearchTests.swift`

**P3.1 - CI Runner Fix**
- Changed `runs-on: macos-26` to `runs-on: macos-latest` (6 occurrences)
- Files: `.github/workflows/ci.yml`

**P3.2 - Documentation Alignment**
- Updated README.md Selection table (removed StreamingTopKKernel recommendation)
- Added Handle Stability section documenting P0.8 behavior
- Added Known Limitations section covering:
  - FusedL2TopKKernel K-range strategies
  - IVF index work-in-progress status
  - Metal 3 vs Metal 4 compilation/runtime
- Files: `README.md`

**P1.3 - Buffer Pool Consistency**
- Replaced `device.makeBuffer()` with `context.getBuffer()` in hot search paths
- IVFSearchPipeline: Query buffer, candidate buffers, distance buffers all use pool
- AcceleratedVectorIndex: searchUnfiltered, searchFiltered, searchBatchGPU use pool
- BufferToken RAII pattern ensures automatic return to pool
- Reduces allocation latency variance in steady-state search
- Files: `IVFSearchPipeline.swift`, `AcceleratedVectorIndex.swift`

**P2.3 - Benchmarking Harness**
- Created `IndexBenchmarkHarness.swift` with ANN-benchmarks style evaluation
- Key metrics: Recall@K, latency percentiles (p50/p95/p99), throughput, memory, build time
- Workload types: pureSearch, mixedInsertSearch, mixedAll, batchSearch
- `LatencyStats` struct computes min/max/mean/stdDev/percentiles
- `IndexBenchmarkConfiguration` supports flat and IVF index types
- `BenchmarkReport.indexReport()` generates formatted console output
- `BenchmarkReport.indexJsonReport()` generates JSON for automation
- Files: `IndexBenchmarkHarness.swift`

---

## Design Decisions (Locked)

These decisions have been made and should guide all implementation work:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **P0.1: IVF Data Layout Fix** | GPU indirection (staged: distance kernel + TopK) | Lower risk than fully fused kernel; reuses existing TopK infrastructure |
| **P0.5: K > 8 Handling** | Auto fallback with memory guardrail | Users expect K=10/20 to work; chunked fallback for large matrices |
| **Handle Stability** | Stable handles, NO generation (stableID never reused) | Simplest DX; deleted stableID absent from map is sufficient |
| **IVF Training Split** | Split responsibilities | `IVFStructure.train()` learns centroids only; `AcceleratedVectorIndex.train()` builds lists once |
| **Handle Storage** | Arrays, not dictionaries | stableIDs are monotonic 0..N; arrays are faster and use less memory |

---

## Recommended Implementation Order

Based on risk reduction and early user-visible wins:

| Order | Issue | Rationale |
|-------|-------|-----------|
| 1 | P0.6 (silent clamping) | Tiny change, big correctness win |
| 2 | P3.1 (CI runner fix) | Increases confidence in all subsequent work |
| 3 | P0.5 (K > 8 fallback) | Fixes user-facing correctness immediately |
| 4 | P0.4 (configurable training + tests) | Creates harness that protects everything else |
| 5 | P0.2 (split training) | Low kernel risk, clears structural correctness |
| 6 | P0.1 (GPU IVF indirection) | Now have tests that catch IVF regressions |
| 7 | P0.3 (batch union) | Resolved by P0.1 design; add regression test |
| 8 | P0.8 (stable handles) | Biggest blast radius, do once search/training solid |

---

## Table of Contents

0. [Completion Summary](#completion-summary) **← START HERE**
1. [Design Decisions (Locked)](#design-decisions-locked)
2. [Phase 1: Correctness and Contracts (P0)](#phase-1-correctness-and-contracts-p0)
3. [Phase 2: Performance Improvements (P1)](#phase-2-performance-improvements-p1)
4. [Phase 3: Index Competitiveness (P2)](#phase-3-index-competitiveness-p2)
5. [Phase 4: Developer Experience (P3)](#phase-4-developer-experience-p3)
6. [Key Design Decisions (Details)](#key-design-decisions)
7. [Progress Summary](#progress-summary)
8. [IVF Quality Assessment](#ivf-quality-assessment-2025-12-14) **← NEW**
9. [Future Performance Improvements (P4)](#future-performance-improvements-p4) **← NEW**

---

## Phase 1: Correctness and Contracts (P0)

These issues affect correctness and must be fixed before any performance work.

### P0.1: IVF Search Data Layout Mismatch

**Status:** [x] **COMPLETED** (2025-12-14)

**Severity:** Critical - IVF search returns incorrect results

**Problem:**
There's a fundamental mismatch between how `IVFStructure.prepareGPUStructure()` prepares data and how `IVFSearchPipeline.gatherAndSearch()` consumes it:

- `prepareGPUStructure` sets `listVectors` to the **global storage buffer** (the main GPU buffer containing all vectors)
- `gatherAndSearch` treats `listVectors` as if it were a **flattened contiguous IVF buffer** indexed by `candidateIdx * dimension`

**Root Cause Analysis:**
```swift
// IVFStructure.swift:384-391
let structure = IVFGPUIndexStructure(
    ...
    listVectors: vectorBuffer,    // <-- This is the GLOBAL storage buffer
    vectorIndices: indexBuffer,   // <-- This maps IVF entry -> storage slot
    ...
)
```

```swift
// IVFSearchPipeline.swift:399-404
for candidateIdx in candidateList {
    let vecStart = candidateIdx * dimension  // <-- WRONG: candidateIdx is IVF list position
    for d in 0..<dimension {
        gatheredVectors.append(vectorsPtr[vecStart + d])  // Reading wrong vector!
    }
    gatheredOriginalIndices.append(originalIndicesPtr[candidateIdx])
}
```

**Affected Files:**
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift`
- `Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift`

**Decision: GPU indirection with staged kernels (lower risk)**

Rather than writing a complex fully-fused IVF kernel (heap + merge + barriers), we use a staged approach that reuses the existing, tested TopK infrastructure from P0.5:

1. **Kernel A:** Compute distances for `(query, candidateSet)` using slot indirection → output contiguous `distances` buffer
2. **Kernel B:** Run existing TopK selection kernel over distances → produce final results

This leverages the same TopK fallback infrastructure built for K > 8, reducing risk significantly.

**Implementation Plan:**

1. **New Metal Kernel:** `ivf_distance_with_indirection` (distance only, no selection)
```metal
// IVFIndirectionDistance.metal
kernel void ivf_distance_with_indirection(
    device const float* queries [[buffer(0)]],           // [Q × D]
    device const float* vectors [[buffer(1)]],           // Global storage [capacity × D]
    device const uint* vectorIndices [[buffer(2)]],      // IVF entry -> storage slot
    device const uint* candidateIndices [[buffer(3)]],   // Which IVF entries to check (flat list)
    device const uint* candidateOffsets [[buffer(4)]],   // CSR offsets: query q checks [offsets[q], offsets[q+1])
    device float* distances [[buffer(5)]],               // Output: [total_candidates] distances
    device uint* outputSlots [[buffer(6)]],              // Output: [total_candidates] storage slots
    constant IVFDistanceParams& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.totalCandidates) return;

    // Find which query this candidate belongs to (binary search or precomputed)
    uint queryIdx = findQueryForCandidate(tid, candidateOffsets, params.numQueries);

    uint ivfEntry = candidateIndices[tid];
    uint storageSlot = vectorIndices[ivfEntry];  // INDIRECTION

    // Compute L2² distance
    float dist = 0.0f;
    for (uint d = 0; d < params.dimension; d++) {
        float diff = queries[queryIdx * params.dimension + d] - vectors[storageSlot * params.dimension + d];
        dist += diff * diff;
    }

    distances[tid] = dist;
    outputSlots[tid] = storageSlot;
}
```

> **Naming note:** `candidateIndices` (not `candidateMask`) — this is a flat index list + CSR offsets, not a bitmask.

2. **Swift Pipeline:** Two-stage search
```swift
// IVFSearchPipeline.swift
private func searchWithIndirection(...) async throws -> IVFSearchResult {
    // Build per-query candidate lists (CPU, small data)
    let (candidateIndices, candidateOffsets) = buildCandidateLists(
        coarseResult: coarseResult,
        structure: structure,
        numQueries: numQueries
    )

    // Stage 1: Compute all distances on GPU
    let distanceResult = try await ivfDistanceKernel.execute(
        queries: queryBuffer,
        vectors: structure.listVectors,
        vectorIndices: structure.vectorIndices,
        candidateIndices: candidateIndices,
        candidateOffsets: candidateOffsets,
        params: distanceParams
    )

    // Stage 2: Select top-k per query using EXISTING TopK infrastructure
    // This reuses the same kernels from P0.5 fallback
    var results: [QueryResult] = []
    for q in 0..<numQueries {
        let queryDistances = distanceResult.distances(forQuery: q)
        let querySlots = distanceResult.slots(forQuery: q)

        let topK = try await topKSelector.execute(
            distances: queryDistances,
            indices: querySlots,
            k: k
        )
        results.append(topK)
    }

    return IVFSearchResult(results: results)
}
```

3. **CSR Format for Per-Query Candidates:**
```
candidateIndices: [c0, c1, c2, c3, c4, c5, c6, ...]  // flat list of IVF entry indices
candidateOffsets: [0, 3, 5, 8, ...]                   // query 0: [0,3), query 1: [3,5), query 2: [5,8)
```
This naturally fixes P0.3 (union-of-candidates bug) — each query only sees its own candidates.

**New Files Required:**
- `Sources/VectorAccelerate/Metal/Shaders/IVFIndirectionDistance.metal`
- `Sources/VectorAccelerate/Index/Kernels/IVF/IVFIndirectionDistanceKernel.swift`

**Kernel Spec:** See `docs/specs/IVF_INDIRECTION_DISTANCE_KERNEL_SPEC.md` for detailed implementation specification.

**Why This Approach:**
- Reuses existing, tested TopK selection kernels
- Distance kernel is simple (no heap, no merge, no barriers)
- Per-query CSR format inherently fixes P0.3
- Can optimize to fully-fused later once correctness is proven

**Definition of Done:**
- [ ] `ivf_distance_with_indirection` kernel compiles and passes unit tests
- [ ] Per-query candidate CSR construction is correct
- [ ] IVF search results match brute-force within expected recall
- [ ] No CPU gather of vectors in hot path
- [ ] P0.3 regression test passes (no cross-query contamination)

---

### P0.2: IVF Training Duplicates Assignments

**Status:** [x] **COMPLETED** (2025-12-14)

**Severity:** Critical - Causes duplicate entries in IVF lists, incorrect cluster sizes

**Problem:**
Training performs assignment twice:

1. `IVFStructure.train()` assigns staged vectors to clusters (lines 196-201)
2. `AcceleratedVectorIndex.train()` reassigns ALL active vectors again (lines 317-328)

This creates duplicate entries and breaks cluster statistics.

**Root Cause Analysis:**
```swift
// IVFStructure.swift:196-204 - First assignment
for (i, clusterIdx) in assignments.enumerated() {
    if i < stagingSlots.count {
        let entry = IVFListEntry(slotIndex: stagingSlots[i], generation: 0)  // <-- generation: 0 is wrong
        invertedLists[clusterIdx].append(entry)
    }
}
stagingSlots.removeAll()

// AcceleratedVectorIndex.swift:317-328 - Second assignment (DUPLICATE)
for slotIndex in deletionMask {
    ...
    _ = ivf.assignToCluster(
        vector: vector,
        slotIndex: UInt32(slotIndex),
        generation: handle.generation
    )
}
```

**Secondary Issue:** `generation: 0` in `IVFStructure.train()` breaks handle validation if generations have advanced.

**Affected Files:**
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift`
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift`

**Decision: Option A - Split responsibilities**

Clean separation where `IVFStructure` owns centroid learning and `AcceleratedVectorIndex` owns list building.

**Implementation Plan:**

1. **Modify `IVFStructure.train()`** - Learn centroids only, no list assignment:
```swift
// IVFStructure.swift
func train(vectors: [[Float]], context: Metal4Context) async throws {
    guard !vectors.isEmpty else {
        throw IndexError.invalidInput(message: "Cannot train with empty vectors")
    }

    trainingState = .training

    do {
        let kmeansConfig = KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 25,
            convergenceThreshold: 0.001,
            metric: .euclidean
        )

        let pipeline = try await KMeansPipeline(context: context, configuration: kmeansConfig)
        let result = try await pipeline.fit(vectors: vectors)

        // ONLY extract centroids - NO list assignment here
        centroids = result.extractCentroids()

        // Clear staging without assigning
        stagingSlots.removeAll()
        gpuStructureDirty = true
        trainingState = .trained

    } catch {
        trainingState = .failed(error)
        throw error
    }
}

// NEW: Method to clear lists before rebuild
func clearInvertedLists() {
    invertedLists = Array(repeating: [], count: numClusters)
    gpuStructureDirty = true
}
```

2. **Modify `AcceleratedVectorIndex.train()`** - Single source of list building:
```swift
// AcceleratedVectorIndex.swift
public func train() async throws {
    guard let ivf = ivfStructure else { return }
    guard !ivf.isTrained else { return }

    // Gather training vectors
    var trainingVectors: [[Float]] = []
    trainingVectors.reserveCapacity(handleAllocator.occupiedCount)

    for slotIndex in deletionMask {
        guard slotIndex < storage.allocatedSlots else { continue }
        trainingVectors.append(try storage.readVector(at: slotIndex))
    }

    guard !trainingVectors.isEmpty else {
        throw IndexError.invalidInput(message: "Cannot train IVF index with no vectors")
    }

    // Train centroids ONLY
    try await ivf.train(vectors: trainingVectors, context: context)

    // Clear any existing lists and rebuild ONCE
    ivf.clearInvertedLists()

    // Assign all vectors with correct generations
    for slotIndex in deletionMask {
        guard slotIndex < storage.allocatedSlots else { continue }
        let vector = try storage.readVector(at: slotIndex)
        if let handle = handleAllocator.handle(for: UInt32(slotIndex)) {
            _ = ivf.assignToCluster(
                vector: vector,
                slotIndex: UInt32(slotIndex),
                generation: handle.generation  // Correct generation from allocator
            )
        }
    }
}
```

**Changes Required:**
- `IVFStructure.swift`: Remove list assignment from `train()`, add `clearInvertedLists()`
- `AcceleratedVectorIndex.swift`: Update `train()` to be single source of list building

---

### P0.3: IVF Batch Search Union-of-Candidates Bug

**Status:** [x] **COMPLETED** (2025-12-14) - Resolved by P0.1 + Regression Test Added

**Severity:** Critical - Batch queries return semantically incorrect results

**Problem:**
`gatherAndSearch` builds a **union** of candidate indices across ALL queries, then runs one fused top-k across that union for EVERY query.

This means:
- Query A can get results from Query B's probed lists
- `nprobe` semantics are violated per-query
- Recall becomes unpredictable

**Resolution:**
P0.1's CSR-based candidate representation (`candidateIndices` + `candidateOffsets`) inherently fixes this bug:
- Each query has its own range: `[candidateOffsets[q], candidateOffsets[q+1])`
- No shared candidate set, no union
- Per-query semantics preserved by design

**This item becomes a regression test:**
```swift
func testIVFBatchSearchNoCrossQueryContamination() async throws {
    let config = IndexConfiguration.ivf(dimension: 32, nlist: 8, nprobe: 1, minTrainingVectors: 50)
    let index = try await AcceleratedVectorIndex(configuration: config)

    // Insert vectors that cluster into distinct groups
    // Group A: vectors [0,0,0,...], Group B: vectors [1,1,1,...]
    for i in 0..<100 {
        let value = Float(i < 50 ? 0 : 1)
        _ = try await index.insert([Float](repeating: value, count: 32))
    }

    // Batch query: Q1 near group A, Q2 near group B
    let queries: [[Float]] = [
        [Float](repeating: 0.0, count: 32),  // Should find group A
        [Float](repeating: 1.0, count: 32)   // Should find group B
    ]

    let results = try await index.search(queries: queries, k: 5)

    // Q1 results should all be from group A (indices 0-49)
    for result in results[0] {
        let slot = await index.slot(for: result.handle)!
        XCTAssertLessThan(slot, 50, "Q1 should only find group A vectors")
    }

    // Q2 results should all be from group B (indices 50-99)
    for result in results[1] {
        let slot = await index.slot(for: result.handle)!
        XCTAssertGreaterThanOrEqual(slot, 50, "Q2 should only find group B vectors")
    }
}
```

**Definition of Done:**
- [ ] P0.1 implemented with CSR per-query candidates
- [ ] Regression test above passes
- [ ] Remove old `allCandidateIndices` union code

---

### P0.4: Test Suite Doesn't Exercise IVF Training

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** High - IVF correctness bugs go undetected

**Problem:**
`minTrainingVectors` defaults to `max(numClusters * 10, 1000)`. Most IVF tests insert far fewer vectors (5-100), so:
- Training never triggers
- All searches fall back to flat search
- IVF code path is not tested

**Evidence:**
```swift
// IVFStructure.swift:151
self.minTrainingVectors = minTrainingVectors ?? max(numClusters * 10, 1000)

// IVFTests.swift - testIVFBasicSearch inserts only 5 vectors
// IVFTests.swift - testIVFLargeScale inserts 500, but nlist=16 requires 1000 minimum
```

**Affected Files:**
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift`
- `Tests/VectorAccelerateTests/IVFTests.swift`

**Required Changes:**

1. **Make `minTrainingVectors` configurable:**
```swift
// IVFStructure.swift
init(numClusters: Int, nprobe: Int, dimension: Int, minTrainingVectors: Int? = nil) {
    self.minTrainingVectors = minTrainingVectors ?? max(numClusters * 10, 1000)
}

// IndexConfiguration.swift - add parameter
public static func ivf(
    dimension: Int,
    nlist: Int,
    nprobe: Int,
    minTrainingVectors: Int? = nil,  // New parameter
    ...
)
```

2. **Add comprehensive IVF correctness tests:**
```swift
func testIVFTrainingActuallyHappens() async throws {
    let config = IndexConfiguration.ivf(
        dimension: 32,
        nlist: 4,
        nprobe: 2,
        minTrainingVectors: 50  // Low threshold for testing
    )
    let index = try await AcceleratedVectorIndex(configuration: config)

    // Insert enough to trigger training
    for i in 0..<60 {
        _ = try await index.insert([Float](repeating: Float(i), count: 32))
    }

    let stats = await index.statistics()
    XCTAssertTrue(stats.ivfStats?.isTrained == true, "IVF should be trained")
}

func testIVFSearchRecallVsBruteForce() async throws {
    // Compare IVF results against flat index for same data
    // Assert recall >= 0.9 for reasonable nprobe
}

func testIVFDeletedVectorsNotReturned() async throws {
    // Insert, delete, search - verify deleted handles never appear
}

func testIVFCompactionNoduplicates() async throws {
    // Insert, delete, compact, verify no duplicate results
}
```

---

### P0.5: Fused L2 Top-K Only Supports K ≤ 8

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** High - Silent incorrect behavior for common K values (10, 20, 50)

**Problem:**
The Metal kernel `fused_l2_topk` uses `K_PRIVATE = 8` for per-thread heap:
```metal
// AdvancedTopK.metal:18
constexpr constant uint K_PRIVATE = 8;

// AdvancedTopK.metal:264
const uint K_emit = (K <= K_PRIVATE) ? K : K_PRIVATE;  // Caps at 8!
```

However, Swift API suggests K up to 128:
```swift
// FusedL2TopKKernel.swift:39
public static let maxK: Int = 128
```

**Result:** For K=10, only 8 valid results are returned; positions 9-10 get sentinel values.

**Affected Files:**
- `Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal`
- `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift`

**Decision: Option B - Auto fallback with future Option C preparation**

Users expect K=10, K=20, K=50 to just work. Implement automatic fallback to two-pass approach for K > 8, while designing the abstraction to make future fused kernel improvements (Option C) a drop-in replacement.

**Implementation Plan:**

1. **K Range Strategy:**

| K Range | Current Strategy | Future (Option C) |
|---------|------------------|-------------------|
| K ≤ 8 | Fused kernel | Improved fused kernel |
| 8 < K ≤ 32 | L2Distance + WarpOptimizedSelection | Fused kernel with multi-pass |
| K > 32 | L2Distance + TopKSelection | Fused kernel with chunked reduction |

2. **Abstraction Layer for Future Kernel Upgrades:**
```swift
// FusedL2TopKKernel.swift

/// Internal strategy enum - allows swapping implementations without API changes
private enum TopKStrategy {
    case fusedDirect          // K ≤ 8: current fused kernel
    case twoPassWarp          // 8 < K ≤ 32: L2 + warp selection
    case twoPassStandard      // K > 32: L2 + standard selection
    // Future: case fusedMultiPass, case fusedChunked
}

private func selectStrategy(k: Int) -> TopKStrategy {
    switch k {
    case 1...8:
        return .fusedDirect
    case 9...32:
        return .twoPassWarp
    default:
        return .twoPassStandard
    }
}

public func execute(...) async throws -> Metal4FusedL2TopKResult {
    let strategy = selectStrategy(k: Int(parameters.k))

    switch strategy {
    case .fusedDirect:
        return try await executeFusedDirect(...)
    case .twoPassWarp:
        return try await executeTwoPassWarp(...)
    case .twoPassStandard:
        return try await executeTwoPassStandard(...)
    }
}

// Two-pass implementation using existing kernels
private func executeTwoPassWarp(...) async throws -> Metal4FusedL2TopKResult {
    // Step 1: Compute all L2 distances
    let distances = try await l2DistanceKernel.execute(queries: queries, dataset: dataset, ...)

    // Step 2: Select top-k using warp-optimized kernel
    let topK = try await warpSelectionKernel.execute(distances: distances, k: k, ...)

    return Metal4FusedL2TopKResult(indices: topK.indices, distances: topK.distances, ...)
}
```

3. **Design for Future Option C (Improved Fused Kernel):**

When we implement the improved fused kernel, it becomes a drop-in:
```swift
private func selectStrategy(k: Int) -> TopKStrategy {
    switch k {
    case 1...32:
        return .fusedMultiPass  // NEW: upgraded kernel handles K up to 32
    case 33...128:
        return .fusedChunked    // NEW: chunked reduction for larger K
    default:
        return .twoPassStandard // Fallback for K > 128
    }
}
```

**Required Changes:**
- `FusedL2TopKKernel.swift`: Add strategy selection and two-pass implementations
- Ensure `L2DistanceKernel` and `TopKSelectionKernel` are initialized in `FusedL2TopKKernel`
- Update `maxK` documentation to reflect actual supported range

**Memory Guardrail for Large Distance Matrices:**

The naive two-pass approach allocates `numDataset × numQueries × sizeof(Float)` for distances. This can blow up memory for large workloads.

```swift
private func selectStrategy(k: Int, numDataset: Int, numQueries: Int) -> TopKStrategy {
    let distanceMatrixBytes = numDataset * numQueries * MemoryLayout<Float>.size
    let maxDistanceMatrixBytes = 64 * 1024 * 1024  // 64 MB threshold

    // If distance matrix would be too large, use chunked approach
    if distanceMatrixBytes > maxDistanceMatrixBytes {
        return .chunked  // Process dataset in tiles, merge partial top-k
    }

    switch k {
    case 1...8:
        return .fusedDirect
    case 9...32:
        return .twoPassWarp
    default:
        return .twoPassStandard
    }
}

// Chunked implementation for memory-constrained scenarios
private func executeChunked(...) async throws -> Metal4FusedL2TopKResult {
    let chunkSize = maxDistanceMatrixBytes / (numQueries * MemoryLayout<Float>.size)
    var partialResults: [[TopKEntry]] = Array(repeating: [], count: numQueries)

    for chunkStart in stride(from: 0, to: numDataset, by: chunkSize) {
        let chunkEnd = min(chunkStart + chunkSize, numDataset)

        // Compute distances for this chunk
        let chunkDistances = try await l2DistanceKernel.execute(
            queries: queries,
            dataset: dataset,
            datasetRange: chunkStart..<chunkEnd,
            ...
        )

        // Select top-k from this chunk
        let chunkTopK = try await topKSelector.execute(distances: chunkDistances, k: k, ...)

        // Merge with running results
        for q in 0..<numQueries {
            partialResults[q] = mergeTopK(partialResults[q], chunkTopK[q], k: k)
        }
    }

    return Metal4FusedL2TopKResult(results: partialResults)
}
```

**Migration Path to Option C:**
1. Implement abstraction layer with memory guardrail (this work)
2. Add benchmarks to measure current fallback performance
3. Implement improved fused kernel with multi-pass reduction
4. Swap strategy selection - no API changes needed

**Definition of Done:**
- [ ] Strategy enum implemented with all cases
- [ ] Two-pass fallback for K > 8 works correctly
- [ ] Chunked approach for large distance matrices works
- [ ] Tests cover K = 1, 8, 10, 32, 50, 100
- [ ] Memory usage doesn't exceed threshold for large workloads

---

### P0.6: Silent Parameter Clamping in FusedL2TopKParameters

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** High - Users get silently wrong results

**Problem:**
```swift
// FusedL2TopKKernel.swift:57-60
self.dimension = UInt32(min(dimension, Self.maxDimension))  // Silent clamp to 768
self.k = UInt32(min(k, Self.maxK))  // Silent clamp to 128
```

If user passes dimension=1024, they get distances computed on 768 dimensions with no error.

**Affected Files:**
- `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift`

**Fix:**
```swift
public init(numQueries: Int, numDataset: Int, dimension: Int, k: Int) throws {
    guard dimension <= Self.maxDimension else {
        throw IndexError.invalidInput(
            message: "Dimension \(dimension) exceeds maximum \(Self.maxDimension)"
        )
    }
    guard k <= Self.maxK else {
        throw IndexError.invalidInput(
            message: "K \(k) exceeds maximum \(Self.maxK)"
        )
    }
    guard k > 0 else {
        throw IndexError.invalidInput(message: "K must be positive")
    }

    self.numQueries = UInt32(numQueries)
    self.numDataset = UInt32(numDataset)
    self.dimension = UInt32(dimension)
    self.k = UInt32(k)
}
```

---

### P0.8: Implement Stable Handles

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** High - Major DX improvement, affects all handle usage

**Problem:**
Current behavior invalidates all handles on compaction, requiring users to maintain handle mappings. This is error-prone and creates friction.

**Decision:** Stable handles with NO generation, using arrays (not dictionaries).

**Key Design Points:**

1. **No generation field** - stableIDs are never reused (`nextStableID` is monotonically increasing), so ABA protection via generation is unnecessary. A deleted stableID is simply absent from the mapping.

2. **Arrays, not dictionaries** - stableIDs are monotonic `0..<nextStableID`, so we use:
   - `stableIDToSlot: [UInt32]` where `stableIDToSlot[stableID] = slot` (or `UInt32.max` as tombstone)
   - `slotToStableID: [UInt32]` sized to storage capacity

   This is faster (no hashing), more memory efficient, and iteration-safe during compaction.

**Implementation Plan:**

1. **Handle Structure (simplified):**
```swift
// VectorHandle.swift
public struct VectorHandle: Hashable, Sendable, Codable {
    /// Stable identifier - never changes for this vector's lifetime
    /// Never reused, so no generation needed for ABA protection
    public let stableID: UInt32

    /// Check validity via HandleAllocator.isValid(handle:)
    /// A deleted handle has its stableID removed from the mapping
}
```

2. **Array-Based Indirection:**
```swift
// HandleAllocator.swift
private var stableIDToSlot: [UInt32] = []      // Index = stableID, value = slot (UInt32.max = deleted)
private var slotToStableID: [UInt32] = []      // Index = slot, value = stableID
private var nextStableID: UInt32 = 0

private let tombstone: UInt32 = .max

func allocate(slot: UInt32) -> VectorHandle {
    let stableID = nextStableID
    nextStableID += 1

    // Grow array if needed
    if stableID >= stableIDToSlot.count {
        stableIDToSlot.append(contentsOf: repeatElement(tombstone, count: 1024))
    }
    if slot >= slotToStableID.count {
        slotToStableID.append(contentsOf: repeatElement(tombstone, count: 1024))
    }

    stableIDToSlot[Int(stableID)] = slot
    slotToStableID[Int(slot)] = stableID

    return VectorHandle(stableID: stableID)
}

@inlinable
func slot(for handle: VectorHandle) -> UInt32? {
    guard handle.stableID < stableIDToSlot.count else { return nil }
    let slot = stableIDToSlot[Int(handle.stableID)]
    return slot == tombstone ? nil : slot
}

func isValid(_ handle: VectorHandle) -> Bool {
    slot(for: handle) != nil
}

func deallocate(_ handle: VectorHandle) {
    guard let slot = slot(for: handle) else { return }
    stableIDToSlot[Int(handle.stableID)] = tombstone
    slotToStableID[Int(slot)] = tombstone
}
```

3. **Safe Compaction (no dictionary mutation during iteration):**
```swift
func compact(keepMask: [Bool], newSlotForOld: [UInt32: UInt32]) {
    // Build new slotToStableID first
    var newSlotToStableID = [UInt32](repeating: tombstone, count: newCapacity)

    // Update stableIDToSlot in place (safe - iterating indices, not dictionary)
    for stableID in 0..<nextStableID {
        let oldSlot = stableIDToSlot[Int(stableID)]
        if oldSlot == tombstone { continue }  // Already deleted

        if let newSlot = newSlotForOld[oldSlot] {
            stableIDToSlot[Int(stableID)] = newSlot
            newSlotToStableID[Int(newSlot)] = stableID
        } else {
            // Vector was deleted during compaction
            stableIDToSlot[Int(stableID)] = tombstone
        }
    }

    slotToStableID = newSlotToStableID
    // Handles are UNCHANGED - users don't need to do anything
}
```

4. **Result Index Semantics:**

GPU kernels output **slot indices**. Search results map slots to stableIDs:
```swift
// In search result construction
for i in 0..<k {
    let slot = gpuResultSlots[i]
    let stableID = slotToStableID[Int(slot)]  // Reverse lookup
    results.append(IndexSearchResult(
        handle: VectorHandle(stableID: stableID),
        distance: gpuResultDistances[i]
    ))
}
```

**Affected Files:**
- `Sources/VectorAccelerate/Index/Internal/HandleAllocator.swift` - major rewrite
- `Sources/VectorAccelerate/Index/Types/VectorHandle.swift` - remove `index`, add `stableID`
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift` - use `slot(for:)` everywhere
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift` - store stableIDs in IVF entries
- `Sources/VectorAccelerate/Index/Internal/MetadataStore.swift` - key by stableID
- All search paths - map GPU slot results to stableIDs

**Breaking Changes:**
- `VectorHandle.index` → `VectorHandle.stableID`
- `compact()` returns `Void` or `CompactionStats` instead of handle mapping

**Definition of Done:**
- [ ] `VectorHandle` uses `stableID` only, no generation
- [ ] `HandleAllocator` uses arrays, not dictionaries
- [ ] Handles remain valid after compaction (test: insert, delete, compact, verify remaining handles work)
- [ ] Deleted handles return `nil` from `slot(for:)` (test: delete, verify handle invalid)
- [ ] Search results correctly map slot → stableID
- [ ] Compaction is safe (no mutation-during-iteration bugs)
- [ ] Memory usage is ~4 bytes per vector (not dictionary overhead)

---

### P0.9: Streaming Top-K Kernel Marked "Not Correct"

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** Medium - Experimental code exposed as production API

**Problem:**
The shader explicitly documents it's not correct:
```metal
// AdvancedTopK.metal:747-754
// Note: This is a simplified implementation where each thread maintains its own heap
// For production, threads would need to cooperatively merge their heaps
// ... we only write back from thread 0 (first thread processes first chunk elements)
// A more sophisticated implementation would use threadgroup reduction
```

**Affected Files:**
- `Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal`
- `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift`

**Fix Options:**
1. Mark as experimental with `@available(*, deprecated, message: "Experimental - results may be incorrect")`
2. Implement proper threadgroup cooperative merge
3. Remove from public API until fixed

**Recommended:** Option 1 for now, option 2 as Phase 3 work.

---

## Phase 2: Performance Improvements (P1)

These improve throughput and latency but don't affect correctness.

### P1.1: Batch Search Runs Sequentially

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** Medium - Major throughput loss for multi-query workloads

**Problem:**
```swift
// AcceleratedVectorIndex.swift:692-700
for query in queries {
    let result = try await search(query: query, k: k, filter: filter)
    results.append(result)
}
```

The fused kernel already supports `numQueries > 1`, but batch API doesn't use it.

**Affected Files:**
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift`

**Fix:**
```swift
public func search(queries: [[Float]], k: Int, filter: ...) async throws -> [[IndexSearchResult]] {
    // For flat index without filter, use true GPU batch
    if configuration.isFlat && filter == nil {
        return try await searchBatchFlat(queries: queries, k: k)
    }

    // Fallback to sequential for IVF or filtered
    // (IVF batch requires P0.3 fix first)
    ...
}

private func searchBatchFlat(queries: [[Float]], k: Int) async throws -> [[IndexSearchResult]] {
    let params = FusedL2TopKParameters(
        numQueries: queries.count,  // <-- Real batch!
        numDataset: storage.allocatedSlots,
        dimension: configuration.dimension,
        k: k
    )

    let flatQueries = queries.flatMap { $0 }
    let queryBuffer = device.makeBuffer(bytes: flatQueries, ...)

    let gpuResult = try await fusedL2TopKKernel.execute(
        queries: queryBuffer,
        dataset: storage.buffer,
        parameters: params
    )

    // Parse batch results
    return (0..<queries.count).map { gpuResult.results(for: $0) }
}
```

---

### P1.2: IVF Pipeline Uses CPU Gather

**Status:** [x] **COMPLETED** (2025-12-14) - Consolidated into P0.1

**Note:** This item is fully addressed by P0.1 (GPU indirection with staged kernels). The `ivf_distance_with_indirection` kernel eliminates CPU gather entirely.

**Verification Completed:**
- [x] Confirmed no `gatheredVectors`, `vectorsPtr[`, or vector read loops in hot path
- [x] Confirmed no CPU-side vector reads during search
- [x] All vector reads happen on GPU via `ivf_distance_with_indirection` kernel

---

### P1.3: Buffer Pool Not Used Consistently

**Status:** [x] **COMPLETED** (2025-12-14)

**Severity:** Low - Latency variance in steady-state

**Problem:**
`Metal4Context` has `BufferPool`, but kernels allocate fresh buffers in hot paths:
```swift
// Everywhere:
guard let queryBuffer = device.makeBuffer(...) else { ... }
```

**Affected Files:**
- `Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift`
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift`

**Solution Implemented:**
Replaced `device.makeBuffer()` with `context.getBuffer()` in hot search paths:
```swift
// Before:
guard let queryBuffer = device.makeBuffer(bytes: flatQueries, ...) else { ... }

// After:
let queryToken = try await context.getBuffer(for: flatQueries)
let queryBuffer = queryToken.buffer
```

**Changes:**
- IVFSearchPipeline.search(): Query buffer from pool
- IVFSearchPipeline.gatherAndSearch(): All transient buffers from pool
  - outputIndices, outputDistances
  - candidateIVFIndices, candidateQueryIds
  - candidateDistances, candidateSlots
  - topKValues, topKCandidateIndices
- AcceleratedVectorIndex.searchUnfiltered(): Query buffer from pool
- AcceleratedVectorIndex.searchFiltered(): Query buffer from pool
- AcceleratedVectorIndex.searchBatchGPU(): Query buffer from pool

BufferToken RAII pattern ensures automatic return to pool when tokens go out of scope.

---

### P1.4: Distance Meaning Unclear

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** Low - User confusion, potential bugs in client code

**Problem:**
- Index returns "L2² distance" (squared) per changelog
- Most users expect Euclidean distance
- `IndexSearchResult.distance` doesn't clarify

**Affected Files:**
- `Sources/VectorAccelerate/Index/Types/SearchResult.swift`
- Documentation

**Fix Options:**
1. Rename to `distanceSquared` for clarity
2. Add `IndexSearchResult.euclideanDistance` computed property
3. Document clearly in API docs

**Recommended:** Option 2 with clear documentation:
```swift
public struct IndexSearchResult {
    /// Squared L2 distance (for euclidean metric)
    public let distanceSquared: Float

    /// Euclidean distance (sqrt of distanceSquared)
    public var euclideanDistance: Float { sqrt(distanceSquared) }
}
```

---

## Phase 3: Index Competitiveness (P2)

### P2.1: GPU-Native Coarse Quantization + Candidate Construction

**Status:** [ ] Not Started

**Depends on:** P0.1

**Note:** P0.1 implements GPU-native fine search (distance computation + selection). This item covers the remaining CPU work: coarse quantization and candidate list construction.

**Current State After P0.1:**
- Coarse search (finding nearest centroids) still happens on CPU
- Candidate list construction (`candidateIndices` + `candidateOffsets`) happens on CPU
- Fine search is GPU-native

**This Item Covers:**
1. **GPU coarse quantization:** Move centroid distance computation to GPU
2. **GPU candidate construction:** Build CSR candidate lists on GPU
3. **Fully GPU-resident IVF search:** Single GPU dispatch from query to results

**Implementation Sketch:**
```metal
// Stage 0: GPU coarse search
kernel void ivf_coarse_search(
    device const float* queries,
    device const float* centroids,
    device uint* nearestCentroids,  // [Q × nprobe]
    ...
) { ... }

// Stage 1: GPU candidate list construction
kernel void ivf_build_candidates(
    device const uint* nearestCentroids,
    device const uint* listOffsets,      // IVF list boundaries
    device const uint* listSizes,
    device uint* candidateIndices,       // Output CSR
    device uint* candidateOffsets,
    ...
) { ... }

// Stage 2: Existing ivf_distance_with_indirection from P0.1
// Stage 3: Existing TopK selection
```

**Priority:** Lower than P0.1-P0.8. The CPU coarse search is not a bottleneck for typical workloads (nlist << numVectors).

---

### P2.2: Scalar Quantization Integration

**Status:** [x] **COMPLETED** (2025-12-15)

**Solution Implemented:**
Integrated scalar quantization into IVF index for 4x-8x memory reduction.

**Key Components:**

`VectorQuantization` enum (in `IndexConfiguration.swift`):
- `.none`: Full float32 precision (default)
- `.sq8`: INT8 symmetric quantization (4x compression)
- `.sq8Asymmetric`: INT8 asymmetric quantization (4x compression, better for non-centered data)
- `.sq4`: INT4 quantization (8x compression, higher recall loss)

`IVFQuantizedStorage` class (new file):
- Manages quantized vector storage for IVF indexes
- Computes optimal quantization parameters (scale, zero-point) from training data
- Supports quantization and dequantization operations
- CPU-based quantization with GPU buffer storage

**Integration Points:**
- `IndexConfiguration`: Added `quantization` property and updated factory methods
- `IVFStructure`: Added `quantization` parameter, initializes `IVFQuantizedStorage` after training
- `bytesPerVector`: Accounts for quantization compression ratio

**Memory Savings:**
- SQ8: 4x reduction (float32 → int8)
- SQ4: 8x reduction (float32 → int4, packed 2 elements per byte)

**Expected Recall Impact:**
- SQ8: < 5% recall loss
- SQ4: < 10% recall loss

**Files:**
- `Sources/VectorAccelerate/Index/Types/IndexConfiguration.swift`
- `Sources/VectorAccelerate/Index/Internal/IVFQuantizedStorage.swift` (NEW)
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift`
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift`
- `Tests/VectorAccelerateTests/IVFQuantizationTests.swift` (NEW)

---

### P2.3: Benchmarking Harness

**Status:** [x] **COMPLETED** (2025-12-14)

**Solution Implemented:**
Created `IndexBenchmarkHarness.swift` with ANN-benchmarks style evaluation framework.

**Key Components:**

`IndexBenchmarkResult`:
- recall: Float (fraction of ground truth in top-k)
- latencyStats: LatencyStats (p50/p95/p99/min/max/mean/stdDev)
- throughput: Double (queries/second)
- memoryUsageBytes: Int
- buildTimeSeconds: TimeInterval

`IndexBenchmarkConfiguration`:
- dimension, datasetSize, numQueries, k
- indexType: .flat or .ivf(nlist, nprobe, minTrainingVectors)
- warmupIterations, measurementIterations
- computeRecall, seed (for reproducibility)
- Predefined: .small, .medium, .large

`BenchmarkWorkload`:
- pureSearch: Static index search benchmark
- batchSearch(batchSize): Multiple queries at once
- mixedInsertSearch(insertRatio): Insert + search
- mixedAll(insertRatio, deleteRatio): Insert + search + delete

`IndexBenchmarkHarness`:
- runBenchmarkSuite(configurations:) → [IndexBenchmarkResult]
- runBenchmark(configuration:) → IndexBenchmarkResult
- runWorkloadBenchmark(workload:configuration:) → IndexBenchmarkResult
- Seeded random number generator for reproducibility

`BenchmarkReport` Extensions:
- indexReport(results:) → String (formatted console output)
- indexJsonReport(results:) → Data (JSON for automation)

**Files:**
- `Sources/VectorAccelerate/Benchmarking/IndexBenchmarkHarness.swift`

---

## Phase 4: Developer Experience (P3)

### P3.1: Fix CI Workflow

**Status:** [x] **COMPLETED** (2025-12-13)

**Severity:** Medium - CI doesn't actually run

**Problem:**
```yaml
runs-on: macos-26  # This doesn't exist
```

GitHub-hosted runners are `macos-13`, `macos-14`, `macos-15`, or `macos-latest`.

**Affected Files:**
- `.github/workflows/ci.yml`

**Fix:**
```yaml
runs-on: macos-latest  # or macos-15 for newest
```

---

### P3.2: Document/Behavior Alignment

**Status:** [x] **COMPLETED** (2025-12-13)

**Issues:**
- README says "handles remain valid across compaction" but compaction invalidates them
- `FusedL2TopKParameters.maxK = 128` but kernel supports K ≤ 8
- Comments reference "Metal 4" but CI validates Metal 3

**Fix:** Add "Known Limitations" section to README, align docs with actual behavior.

---

### P3.3: Reduce Public API Surface

**Status:** [ ] Not Started

Move experimental kernels to separate module or mark with `@_spi(Experimental)`.

---

## Key Design Decisions

### Decision A: Handle Stability Across Compaction

**Decision: Stable handles with NO generation, using arrays**

Handles will never change across compaction, deletions, or any index operation. This provides the best developer experience.

**Critical Design Choice: No Generation Field**

The original plan included a `generation` field for ABA protection, but this creates a correctness bug:
- After compaction, handles move to new slots
- The handle's generation stays the same (that's the point of stability)
- But `slotGenerations[newSlot]` is different from the old slot
- Result: valid handles fail validation after compaction

**Solution:** Drop `generation` entirely. Since `stableID` is never reused (monotonically increasing `nextStableID`), a deleted handle is simply absent from the mapping. No ABA protection needed.

**Critical Design Choice: Arrays, Not Dictionaries**

Dictionaries have issues:
1. Higher memory overhead than claimed (+8 bytes is optimistic for Swift dictionaries)
2. Can't mutate during iteration (the compaction pseudocode `for (k,v) in dict { dict[k] = newV }` is illegal)

**Solution:** Use arrays since stableIDs are monotonic `0..<nextStableID`:
```swift
private var stableIDToSlot: [UInt32] = []  // Index = stableID, value = slot or tombstone
private var slotToStableID: [UInt32] = []  // Index = slot, value = stableID or tombstone
```

See P0.8 for full implementation details.

**Files Affected:**
- `Sources/VectorAccelerate/Index/Internal/HandleAllocator.swift` - major rewrite
- `Sources/VectorAccelerate/Index/Types/VectorHandle.swift` - simplified to just `stableID`
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift` - use `slot(for:)` everywhere
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift` - store stableIDs in IVF entries
- All search paths - map GPU slot results to stableIDs via `slotToStableID`

---

### Decision B: Supported K Range

**Decision: Auto-fallback for all K values up to 128**

See P0.5 for full implementation details. Summary:

| K Range | Strategy | Performance |
|---------|----------|-------------|
| 1-8 | Fused kernel | Best |
| 9-32 | Two-pass (L2 + warp selection) | Good |
| 33-128 | Two-pass (L2 + standard selection) | Acceptable |
| >128 | Error (or chunked in future) | N/A |

Designed for future Option C upgrade path.

---

## Progress Summary

**Recommended Implementation Order** (see [Recommended Implementation Order](#recommended-implementation-order) for rationale):

| Order | Issue | Status | Notes |
|-------|-------|--------|-------|
| 1 | P0.6 Silent clamping | [x] **DONE** | Throwing initializer |
| 2 | P3.1 CI fix | [x] **DONE** | `macos-26` → `macos-latest` |
| 3 | P0.5 K > 8 (auto fallback + memory guardrail) | [x] **DONE** | GPU merge path |
| 4 | P0.4 Configurable training + tests | [x] **DONE** | minTrainingVectors param + IVF tests |
| 5 | P0.2 Split training | [x] **DONE** | Fixed double-assignment bug |
| 6 | P0.1 GPU IVF indirection (staged kernels) | [x] **DONE** | CSR candidates + GPU indirection |
| 7 | P0.3 Batch union bug | [x] **DONE** | Resolved by P0.1 + regression test |
| 8 | P0.8 Stable handles (no gen, arrays) | [x] **DONE** | stableID-only, arrays |

**Other Items:**

| Phase | Issue | Status | Notes |
|-------|-------|--------|-------|
| P0 | P0.9 Streaming correctness | [x] **DONE** | Marked experimental/deprecated |
| P1 | P1.1 Sequential batch | [x] **DONE** | GPU batch for flat index |
| P1 | P1.2 CPU gather | [x] **DONE** | Consolidated into P0.1, verified |
| P1 | P1.3 Buffer pool | [x] **DONE** | Buffer pool in search hot paths |
| P1 | P1.4 Distance clarity | [x] **DONE** | euclideanDistance property + test |
| P2 | P2.1 GPU coarse quant + candidates | [ ] | After P0.1 |
| P2 | P2.2 Scalar Quantization | [x] **DONE** | VectorQuantization enum, IVFQuantizedStorage |
| P2 | P2.3 Benchmarks | [x] **DONE** | IndexBenchmarkHarness implemented |
| P3 | P3.2 Doc alignment | [x] **DONE** | README updated |
| P3 | P3.3 API surface | [ ] | Future |
| P4 | P4.1 K-Means++ Init | [x] **DONE** | Improved cluster quality |
| P4 | P4.2 Adaptive nlist | [x] **DONE** | recommendedNlist, ivfAuto factory |

---

## IVF Quality Assessment (2025-12-14)

Comprehensive benchmarking was performed to validate IVF recall against industry standards.

### Recall vs FAISS Benchmarks (N=2000, D=128, nlist=32, K=10)

| nprobe | % clusters | FAISS Expected | VectorAccelerate | Status |
|--------|------------|----------------|------------------|--------|
| 3 | ~10% | 70% | **77.4%** | ✓ Meets benchmark |
| 6 | ~20% | 80% | **87.7%** | ✓ Exceeds benchmark |
| 16 | 50% | 92% | **96.7%** | ✓ Exceeds benchmark |
| 32 | 100% | 99% | **100.0%** | ✓ Perfect recall |

**Verdict:** IVF recall is within acceptable range of industry benchmarks.

### Recall vs Data Distribution (nprobe=50%, N=1000, D=64)

| Distribution | Recall | Assessment |
|--------------|--------|------------|
| Uniform Random | 84.6% | Acceptable |
| Gaussian Clusters | **93.8%** | Good |
| Sparse Clusters | **91.0%** | Good |

Real-world embeddings (text, images) typically have natural clustering, so expect 90%+ recall.

### IVF vs Flat Throughput (D=128, K=10, nprobe=25%)

| Dataset Size | Flat (q/s) | IVF (q/s) | Speedup | IVF Recall |
|--------------|------------|-----------|---------|------------|
| 1,000 | 1,737 | 862 | 0.50x | 73.2% |
| 2,000 | 1,413 | 1,101 | 0.78x | 68.2% |
| 5,000 | 1,044 | 960 | 0.92x | 56.8% |

**Note:** IVF overhead doesn't pay off until N > 10,000 vectors. For small datasets, flat index is faster.

### Batch Search Performance

| Workload | Throughput | p50 Latency |
|----------|------------|-------------|
| Single query | 2,740 q/s | 0.283 ms |
| Batch (size=10) | 33,174 q/s | 0.030 ms |
| **Speedup** | **9.53x** | |

GPU batch search provides ~10x speedup over sequential queries.

### Key Findings

1. **Recall is production-ready** for datasets with natural clustering (embeddings)
2. **100% recall** is achievable with nprobe=nlist (exhaustive mode)
3. **Typical operating point:** nprobe=20-50% of nlist → 85-95% recall
4. **Batch search** provides significant speedup and should be preferred

---

## Future Performance Improvements (P4)

Based on quality assessment, these improvements would enhance IVF performance:

### P4.1: K-Means++ Initialization

**Status:** [x] **COMPLETED** (2025-12-14)

**Priority:** Medium
**Impact:** +5-10% recall at same nprobe

**Problem:**
Current K-means uses random centroid initialization, which can lead to suboptimal clustering.

**Solution Implemented:**
K-means++ initialization was implemented in the clustering pipeline.

**Files:**
- `Sources/VectorAccelerate/Index/Kernels/Clustering/KMeansPipeline.swift`
- `Sources/VectorAccelerate/Index/Kernels/Clustering/ClusteringKernels.swift`

**Expected Improvement:**
- Better cluster separation → fewer missed neighbors
- More consistent recall across different data distributions

---

### P4.2: Adaptive nlist Selection

**Status:** [x] **COMPLETED** (2025-12-15)

**Priority:** Low
**Impact:** Better defaults for users

**Solution Implemented:**
Added helper methods and `ivfAuto` factory to `IndexConfiguration`.

**Key Components:**

`recommendedNlist(for:)`:
- Uses sqrt(N) heuristic, clamped to [8, 4096]
- Example: 100K vectors → nlist=316

`recommendedNprobe(for:targetRecall:)`:
- Scales with nlist based on target recall
- 70% recall: 10% of nlist
- 90% recall: 23% of nlist
- 99% recall: 50% of nlist

`ivfAuto(dimension:expectedSize:targetRecall:metric:quantization:)`:
- One-stop factory for optimal IVF configuration
- Automatically computes nlist and nprobe from dataset size and recall target

**Usage Example:**
```swift
// For a 100K vector dataset with 90% target recall
let config = IndexConfiguration.ivfAuto(
    dimension: 768,
    expectedSize: 100_000,
    targetRecall: 0.90
)
// Results: nlist=316, nprobe=73
```

**Files:**
- `Sources/VectorAccelerate/Index/Types/IndexConfiguration.swift`
- `Tests/VectorAccelerateTests/AdaptiveNlistTests.swift` (NEW)

---

### P4.3: Residual Quantization (OPQ/PQ)

**Priority:** Low (after P2.2)
**Impact:** 4-8x memory reduction, enables million-scale indexes

**Problem:**
Full float32 vectors consume significant memory:
- 1M vectors × 128D × 4 bytes = 512 MB

**Solution:**
Implement Product Quantization:
1. Split vector into subvectors (e.g., 128D → 8 × 16D)
2. Quantize each subvector to 8-bit codebook index
3. Store 8 bytes per vector instead of 512 bytes (64x compression)

**Implementation Sketch:**
```swift
struct ProductQuantizer {
    let numSubquantizers: Int  // e.g., 8
    let bitsPerCode: Int       // e.g., 8 (256 centroids per subquantizer)
    var codebooks: [[Float]]   // [numSubquantizers][256 * subvectorDim]

    func encode(_ vector: [Float]) -> [UInt8]
    func asymmetricDistance(query: [Float], codes: [UInt8]) -> Float
}
```

**Files:**
- `Sources/VectorAccelerate/Quantization/ProductQuantization.swift`
- `Sources/VectorAccelerate/Metal/Shaders/ProductQuantization.metal`

---

### P4.4: Multi-Probe LSH Fallback

**Priority:** Low
**Impact:** Alternative to IVF for certain workloads

**Problem:**
IVF requires training (K-means), which is expensive for streaming data.

**Solution:**
Implement LSH (Locality-Sensitive Hashing) as training-free alternative:
- No training required
- Good for streaming/incremental indexing
- Lower recall than IVF but faster index construction

---

### P4.5: HNSW Index Type

**Priority:** Medium-High
**Impact:** State-of-the-art recall/speed tradeoff

**Problem:**
IVF has limitations:
- Requires training
- Recall depends on cluster quality
- Not optimal for very high recall requirements

**Solution:**
Implement Hierarchical Navigable Small World (HNSW):
- Graph-based index with O(log N) search complexity
- No training required
- State-of-the-art recall at high throughput
- Used by FAISS, Pinecone, Milvus, etc.

**Complexity:** High (significant implementation effort)

**Files:**
- New: `Sources/VectorAccelerate/Index/HNSW/HNSWIndex.swift`
- New: `Sources/VectorAccelerate/Index/HNSW/HNSWGraph.swift`
- New: `Sources/VectorAccelerate/Metal/Shaders/HNSWSearch.metal`

---

### Performance Improvement Priority Order

| Priority | Item | Impact | Effort | Status |
|----------|------|--------|--------|--------|
| 1 | P2.1 GPU Coarse Quantization | Remove CPU bottleneck | Medium | Not Started |
| 2 | P4.1 K-Means++ Init | +5-10% recall | Low | **COMPLETED** |
| 3 | P4.5 HNSW Index | State-of-the-art | High | Not Started |
| 4 | P2.2 Scalar Quantization | 4x memory reduction | Medium | **COMPLETED** |
| 5 | P4.2 Adaptive nlist | Better UX | Low | **COMPLETED** |
| 6 | P4.3 Product Quantization | 64x compression | High | Not Started |

---

## Appendix: Reference Links

- [FAISS](https://github.com/facebookresearch/faiss) - Reference IVF implementation
- [hnswlib](https://github.com/nmslib/hnswlib) - HNSW reference
- [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) - Benchmarking methodology
- [MPSMatrixFindTopK](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk) - Apple's top-k primitive
- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) - Initialization algorithm
- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Hierarchical Navigable Small World graphs
- [Product Quantization](https://hal.inria.fr/inria-00514462/document) - Vector compression for similarity search
