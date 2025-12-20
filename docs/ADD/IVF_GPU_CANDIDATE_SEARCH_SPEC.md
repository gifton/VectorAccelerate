# GPU-Based IVF Candidate Search Kernel Specification

## Executive Summary

We need a Metal compute kernel that performs IVF (Inverted File Index) search directly on GPU without CPU-side candidate gathering. The current implementation is 27x slower than flat search due to CPU overhead from gathering candidates into intermediate buffers.

**Goal**: Eliminate CPU bottleneck by searching directly on inverted list buffers in GPU.

---

## Current Architecture (What's Wrong)

### Current Flow (Slow)
```
1. CPU: Read selected cluster indices from coarse quantizer
2. CPU: Loop through each query × nprobe × list_vectors (62,400 iterations for 100 queries)
3. CPU: Build Set<Int> of unique candidates
4. CPU: Sort candidates
5. CPU: Copy candidate vectors element-by-element to new buffer (80,000 float copies)
6. CPU: Allocate new Metal buffer
7. GPU: Run L2 + TopK on copied data
8. CPU: Map results back to original indices
```

### Performance Impact
- 100 queries, nprobe=8, 78 vectors/list
- **Flat search**: 0.042s (2361 q/s)
- **IVF search**: 1.137s (88 q/s)
- **Slowdown**: 27x

### Location of Problem Code
`Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift`
- Method: `gatherAndSearch()` (lines 520-744)
- CPU loops: lines 577-592
- Element copy: lines 639-645

---

## Proposed Architecture (GPU-Based)

### New Flow
```
1. GPU: Coarse quantizer returns [numQueries × nprobe] cluster indices
2. GPU: New kernel reads cluster indices + list offsets
3. GPU: For each query, iterate through selected lists directly
4. GPU: Compute L2 distances to vectors in those lists
5. GPU: Maintain per-query top-K heap
6. GPU: Output final [numQueries × K] results
```

**Key insight**: Vectors are already contiguous per-list in `listVectors` buffer. We don't need to copy them - just compute distances directly using offset arithmetic.

---

## Data Structures

### Input Buffers (Already Exist)

#### 1. Query Buffer
```
Type: device float*
Layout: [numQueries × dimension] row-major
Size: numQueries * dimension * sizeof(float)
Label: "IVFSearchPipeline.queries"
```

#### 2. List Vectors Buffer
```
Type: device float*
Layout: [totalVectors × dimension] row-major, CSR-ordered by cluster
Size: totalVectors * dimension * sizeof(float)
Label: "IVFStructure.listVectors"

Structure: Vectors are stored contiguously per cluster.
  Cluster 0 vectors: [offset[0]..offset[1])
  Cluster 1 vectors: [offset[1]..offset[2])
  ...
  Cluster N vectors: [offset[N]..offset[N+1])
```

#### 3. List Offsets Buffer (CSR Format)
```
Type: device uint32_t*
Layout: [numCentroids + 1] - CSR row pointers
Size: (numCentroids + 1) * sizeof(uint32_t)
Label: "IVFStructure.listOffsets"

Usage:
  list_start = offsets[cluster_id]
  list_end = offsets[cluster_id + 1]
  list_size = list_end - list_start

  vector[i] in cluster is at: listVectors[(list_start + i) * dimension]
```

#### 4. Vector Indices Buffer (Original Index Mapping)
```
Type: device uint32_t*
Layout: [totalVectors] - maps CSR position to original vector index
Size: totalVectors * sizeof(uint32_t)
Label: "IVFStructure.vectorIndices"

Usage:
  csr_position = list_start + local_index
  original_index = vectorIndices[csr_position]
```

#### 5. Coarse Quantizer Result
```
Type: device uint32_t*
Layout: [numQueries × nprobe] - selected cluster IDs per query
Size: numQueries * nprobe * sizeof(uint32_t)
Label: "IVFCoarseQuantizer.listIndices"

Note: Invalid entries marked with 0xFFFFFFFF sentinel
```

### Output Buffers

#### 1. Result Indices
```
Type: device uint32_t*
Layout: [numQueries × K] - original vector indices (not CSR positions)
Size: numQueries * K * sizeof(uint32_t)
Initialize: 0xFFFFFFFF (sentinel for "no result")
```

#### 2. Result Distances
```
Type: device float*
Layout: [numQueries × K] - L2 distances (squared, not sqrt)
Size: numQueries * K * sizeof(float)
Initialize: INFINITY
```

---

## Algorithm Specification

### Option A: Per-Query Thread Group (Recommended for nprobe < 16)

```metal
kernel void ivf_list_search(
    device const float* queries           [[buffer(0)]],  // [Q × D]
    device const float* listVectors       [[buffer(1)]],  // [N × D]
    device const uint32_t* listOffsets    [[buffer(2)]],  // [nlist + 1]
    device const uint32_t* vectorIndices  [[buffer(3)]],  // [N]
    device const uint32_t* selectedLists  [[buffer(4)]],  // [Q × nprobe]
    device uint32_t* outIndices           [[buffer(5)]],  // [Q × K]
    device float* outDistances            [[buffer(6)]],  // [Q × K]
    constant IVFSearchParams& params      [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one query
    uint queryIdx = tgid;
    if (queryIdx >= params.numQueries) return;

    // Load query into shared memory for reuse
    threadgroup float sharedQuery[MAX_DIMENSION];
    // ... cooperative load ...

    // Initialize thread-local top-K heap
    float localDistances[K];
    uint32_t localIndices[K];
    // ... init to infinity/sentinel ...

    // Iterate through selected lists for this query
    for (uint p = 0; p < params.nprobe; p++) {
        uint32_t clusterId = selectedLists[queryIdx * params.nprobe + p];
        if (clusterId == 0xFFFFFFFF) continue;

        uint32_t listStart = listOffsets[clusterId];
        uint32_t listEnd = listOffsets[clusterId + 1];

        // Threads cooperatively process vectors in this list
        for (uint i = listStart + ltid; i < listEnd; i += tgSize) {
            // Compute L2 distance
            float dist = 0.0f;
            for (uint d = 0; d < params.dimension; d++) {
                float diff = sharedQuery[d] - listVectors[i * params.dimension + d];
                dist += diff * diff;
            }

            // Get original index
            uint32_t origIdx = vectorIndices[i];

            // Insert into thread-local heap if better
            heap_insert(localDistances, localIndices, dist, origIdx, K);
        }
    }

    // Reduce thread-local heaps to threadgroup result
    // ... threadgroup reduction ...

    // Thread 0 writes final result
    if (ltid == 0) {
        for (uint k = 0; k < params.k; k++) {
            outIndices[queryIdx * params.k + k] = finalIndices[k];
            outDistances[queryIdx * params.k + k] = finalDistances[k];
        }
    }
}
```

### Option B: Two-Phase Approach (For Large nprobe or Many Candidates)

**Phase 1: Compute all distances**
```metal
// Output: [numQueries × maxCandidates] distances + indices
// Each thread handles one (query, candidate) pair
```

**Phase 2: TopK selection**
```metal
// Use existing TopKSelectionKernel on phase 1 output
```

### Parameters Struct
```metal
struct IVFSearchParams {
    uint32_t numQueries;
    uint32_t numCentroids;
    uint32_t dimension;
    uint32_t nprobe;
    uint32_t k;
    uint32_t maxCandidatesPerQuery;  // nprobe * max_list_size (for bounds)
};
```

---

## Performance Requirements

| Metric | Current | Target |
|--------|---------|--------|
| 100 queries @ N=5K | 1.137s | < 0.05s |
| Throughput | 88 q/s | > 2000 q/s |
| vs Flat speedup | 0.04x | > 1.5x |

### Memory Access Patterns

- **Query access**: Each query loaded once, reused for all its candidates
- **List vector access**: Sequential within list (coalesced)
- **Cross-list access**: Random (different lists have different offsets)

### Threadgroup Size Recommendations

- **Dimension ≤ 128**: 256 threads/group, load query cooperatively
- **Dimension > 128**: Consider tiling or multiple loads
- **Lists per query**: nprobe typically 4-32

---

## Integration Points

### Existing Coarse Quantizer
```swift
// Already exists: IVFCoarseQuantizerKernel
let coarseResult = try await coarseQuantizer.findNearestCentroids(
    queries: queryBuffer,
    centroids: structure.centroids,
    numQueries: numQueries,
    numCentroids: structure.numCentroids,
    dimension: dimension,
    nprobe: nprobe
)
// Returns: coarseResult.listIndices (device buffer)
```

### New Kernel Should Replace
```swift
// DELETE: gatherAndSearch() method (lines 520-744)
// REPLACE WITH:
let searchResult = try await ivfListSearchKernel.search(
    queries: queryBuffer,
    structure: structure,  // Contains listVectors, listOffsets, vectorIndices
    selectedLists: coarseResult.listIndices,
    numQueries: numQueries,
    nprobe: nprobe,
    k: k
)
```

### Existing Kernel Pattern to Follow
See `FusedL2TopKKernel.swift` for:
- Buffer management patterns
- Command buffer/encoder setup
- Result extraction
- Error handling

---

## Edge Cases to Handle

1. **Empty lists**: Some clusters may have 0 vectors
   - `listOffsets[c] == listOffsets[c+1]` → skip

2. **K > total candidates**: Return fewer than K results
   - Output sentinel (0xFFFFFFFF) for missing slots

3. **Invalid cluster ID**: Coarse quantizer may return sentinel
   - Check for 0xFFFFFFFF before accessing offsets

4. **Large dimension**: D > shared memory capacity
   - Tile query loading or use registers

5. **Variable list sizes**: Lists can range from 0 to 10x average
   - Load balancing may suffer; consider work-stealing

---

## Files to Reference

| File | Purpose |
|------|---------|
| `IVFSearchPipeline.swift` | Current implementation to replace |
| `IVFCoarseQuantizerKernel.swift` | Coarse quantizer interface |
| `FusedL2TopKKernel.swift` | Pattern for L2+TopK in one kernel |
| `TopKSelectionKernel.swift` | Standalone TopK if needed |
| `IVFStructure.swift` | Data structure definitions |
| `Shaders/L2Distance.metal` | L2 distance computation patterns |
| `Shaders/AdvancedTopK.metal` | Heap-based TopK patterns |

---

## Deliverables

1. **Metal shader**: `IVFListSearch.metal`
   - Kernel function(s) as specified above
   - Optimized for M1/M2/M3 GPU architecture

2. **Swift wrapper**: `IVFListSearchKernel.swift`
   - Follows existing kernel patterns
   - Handles buffer setup, dispatch, result extraction

3. **Integration**: Update `IVFSearchPipeline.swift`
   - Replace `gatherAndSearch()` with new kernel call
   - Remove CPU gathering code

---

## Validation

After implementation, run:
```bash
swift test --filter IVFValidationTests
```

All Priority 1 tests should still pass:
- testRecallIncreasesMonotonicallyWithNprobe
- testFullNprobeGivesNearPerfectRecall
- testDistancesMatchFlatIndex
- testIndexMappingIsCorrect
- testResultsAreSortedByDistance

And `testIVFFasterThanFlatAtScale` should now pass with IVF faster than flat.

---

## Existing Shader Patterns to Reference

### From AdvancedTopK.metal

The existing `fused_l2_topk` kernel in `Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal` is the closest reference. Key patterns:

```metal
// Constants
constexpr constant uint K_PRIVATE = 8;         // Per-thread heap size
constexpr constant uint MAX_TGS = 256;         // Max threadgroup size
constexpr constant uint MAX_D = 768;           // Max dimension
constexpr constant uint SENTINEL_INDEX = 0xFFFFFFFF;

// Candidate structure
struct Candidate {
    float distance;
    uint index;
};

// L2 distance computation (vectorized)
inline float calculate_l2_squared(const threadgroup float* query, device const float* vec, uint D) {
    float acc = 0.0f;
    uint d = 0;
    for (; d + 3 < D; d += 4) {
        float4 q = reinterpret_cast<const threadgroup float4*>(query + d)[0];
        float4 v = safe_load_float4(vec, d, D);
        float4 diff = q - v;
        acc += dot(diff, diff);
    }
    for (; d < D; ++d) {
        float diff = query[d] - vec[d];
        acc = fma(diff, diff, acc);
    }
    return acc;
}

// Per-thread heap update
inline void update_private_heap_sorted(thread Candidate* heap, float dist, uint id) {
    if (dist < heap[K_PRIVATE-1].distance) {
        heap[K_PRIVATE-1] = {dist, id};
        // Insertion sort to maintain order
        for (uint i = K_PRIVATE-1; i > 0; --i) {
            if (is_better(heap[i], heap[i-1])) {
                swap(heap[i], heap[i-1]);
            } else break;
        }
    }
}
```

### Key Difference for IVF

The existing kernel iterates:
```metal
for (uint n_idx = tid; n_idx < N; n_idx += tgs) {  // ALL vectors
    ...
}
```

The IVF kernel should iterate:
```metal
for (uint p = 0; p < nprobe; p++) {
    uint cluster = selectedLists[q * nprobe + p];
    if (cluster == SENTINEL_INDEX) continue;

    uint listStart = listOffsets[cluster];
    uint listEnd = listOffsets[cluster + 1];

    for (uint i = listStart + tid; i < listEnd; i += tgs) {
        // Compute L2 to listVectors[i * D ... (i+1)*D]
        // Map back: originalIndex = vectorIndices[i]
        // Update heap with (dist, originalIndex)
    }
}
```

---

## Questions for Implementer

1. **Heap vs Sort for TopK**: Preference for maintaining K-heap during scan vs sorting at end?

2. **Query tiling**: For batch queries, process multiple queries per threadgroup or one?

3. **SIMD utilization**: Use `simd_sum` for distance reduction or manual reduction?

4. **Memory residency**: Should we hint `MTLResourceOptions` for list buffers?

5. **Profiling hooks**: Add GPU timestamps for phase timing?

6. **Load balancing**: Lists have variable sizes. Should we use:
   - Simple: Each threadgroup handles one query, threads stripe across all its lists
   - Advanced: Work-stealing or dynamic scheduling across queries
