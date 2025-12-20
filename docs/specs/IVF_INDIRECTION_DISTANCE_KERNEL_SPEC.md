# Kernel Specification: IVF Distance with Indirection

**Kernel Name:** `ivf_distance_with_indirection`
**Purpose:** Compute L2² distances between queries and candidate vectors using slot indirection
**Priority:** P0.1 (Critical correctness fix)

---

## Context

### Problem Being Solved

The current IVF search pipeline has a data layout bug where it treats a global storage buffer as if it were a flattened IVF-specific buffer. The fix requires computing distances while performing slot indirection on the GPU.

### Architecture Decision

Rather than building a complex fully-fused kernel (distance + heap + merge), we use a **staged approach**:

1. **This kernel:** Compute distances for all candidates with slot indirection
2. **Existing kernel:** Use already-tested TopK selection kernel on the distance output

This reduces implementation risk and reuses proven infrastructure.

### Data Flow

```
Input:
  queries[Q × D]              - Query vectors
  vectors[capacity × D]       - Global storage buffer (sparse, may have gaps)
  vectorIndices[totalIVF]     - Maps IVF entry index → storage slot
  candidateIndices[totalCand] - Flat list of IVF entry indices to check
  candidateOffsets[Q + 1]     - CSR format: query q checks [offsets[q], offsets[q+1])

Output:
  distances[totalCand]        - L2² distance for each candidate
  outputSlots[totalCand]      - Storage slot for each candidate (for result mapping)
```

---

## Kernel Signature

```metal
#include <metal_stdlib>
using namespace metal;

struct IVFDistanceParams {
    uint32_t numQueries;       // Q
    uint32_t dimension;        // D
    uint32_t totalCandidates;  // Total entries in candidateIndices
    uint32_t storageCapacity;  // Size of vectors buffer (for bounds checking)
};

kernel void ivf_distance_with_indirection(
    device const float* queries           [[buffer(0)]],  // [Q × D]
    device const float* vectors           [[buffer(1)]],  // [capacity × D] global storage
    device const uint32_t* vectorIndices  [[buffer(2)]],  // [totalIVF] IVF entry → slot
    device const uint32_t* candidateIndices [[buffer(3)]],// [totalCand] which IVF entries
    device const uint32_t* candidateOffsets [[buffer(4)]],// [Q + 1] CSR offsets
    device float* distances               [[buffer(5)]],  // [totalCand] output distances
    device uint32_t* outputSlots          [[buffer(6)]],  // [totalCand] output slots
    constant IVFDistanceParams& params    [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
);
```

---

## Algorithm

### Per-Thread Work

Each thread processes one candidate entry:

```
1. Check bounds: if (tid >= params.totalCandidates) return;

2. Find which query this candidate belongs to:
   queryIdx = binarySearch(candidateOffsets, tid, params.numQueries)
   // Such that candidateOffsets[queryIdx] <= tid < candidateOffsets[queryIdx + 1]

3. Get the IVF entry index:
   ivfEntry = candidateIndices[tid]

4. Perform INDIRECTION to get storage slot:
   storageSlot = vectorIndices[ivfEntry]

5. Compute L2² distance:
   dist = 0.0
   for d in 0..<D:
       diff = queries[queryIdx * D + d] - vectors[storageSlot * D + d]
       dist += diff * diff

6. Write outputs:
   distances[tid] = dist
   outputSlots[tid] = storageSlot
```

### Binary Search for Query Index

```metal
inline uint32_t findQueryForCandidate(
    uint32_t tid,
    device const uint32_t* offsets,
    uint32_t numQueries
) {
    // Binary search to find q such that offsets[q] <= tid < offsets[q+1]
    uint32_t lo = 0;
    uint32_t hi = numQueries;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= tid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
```

---

## Optimization Requirements

### 1. Vectorized Distance Computation

For dimensions that are multiples of 4, use SIMD:

```metal
float computeL2Squared_SIMD(
    device const float* query,
    device const float* vector,
    uint32_t dimension
) {
    float4 sum = float4(0.0f);
    uint32_t d = 0;

    // Process 4 floats at a time
    for (; d + 4 <= dimension; d += 4) {
        float4 q = float4(query[d], query[d+1], query[d+2], query[d+3]);
        float4 v = float4(vector[d], vector[d+1], vector[d+2], vector[d+3]);
        float4 diff = q - v;
        sum += diff * diff;
    }

    float result = sum.x + sum.y + sum.z + sum.w;

    // Handle remainder
    for (; d < dimension; d++) {
        float diff = query[d] - vector[d];
        result += diff * diff;
    }

    return result;
}
```

### 2. Memory Access Pattern

- **Queries:** Each query is accessed by multiple threads (all candidates for that query). Consider query caching in threadgroup memory for high-candidate-count scenarios.
- **Vectors:** Random access pattern due to indirection. No optimization possible here.
- **CandidateIndices/Offsets:** Sequential access, good cache behavior.

### 3. Occupancy Considerations

- Thread count = `totalCandidates`
- Typical: 1000-100,000 candidates per search
- Use 1D grid dispatch: `(totalCandidates + threadgroupSize - 1) / threadgroupSize`
- Recommended threadgroup size: 256 or 512

---

## Edge Cases and Validation

### Input Validation

```metal
// Bounds check on candidate index
if (tid >= params.totalCandidates) return;

// The following should be guaranteed by Swift caller, but defensive:
// - ivfEntry < length(vectorIndices)
// - storageSlot < params.storageCapacity
// - queryIdx < params.numQueries
```

### Numerical Considerations

- L2² can overflow for large dimensions with large values. For typical embedding dimensions (64-1536) and normalized vectors, this is not a concern.
- No special handling for NaN/Inf required (garbage in, garbage out is acceptable).

### Empty Candidate Sets

If a query has zero candidates (`candidateOffsets[q] == candidateOffsets[q+1]`), no threads will process it. The Swift caller must handle this case (return empty results for that query).

---

## Test Cases

### 1. Basic Correctness

```
Setup:
  - 2 queries, dimension 4
  - 10 vectors in storage at slots [0, 2, 4, 6, 8, ...] (sparse)
  - vectorIndices maps IVF entries 0-9 to slots [0, 2, 4, 6, 8, ...]
  - Query 0 checks IVF entries [0, 1, 2]
  - Query 1 checks IVF entries [3, 4]

Input:
  queries = [[1,0,0,0], [0,1,0,0]]
  vectors[slot=0] = [1,0,0,0]  // exact match for Q0
  vectors[slot=2] = [0,1,0,0]  // exact match for Q1
  ...
  candidateIndices = [0, 1, 2, 3, 4]
  candidateOffsets = [0, 3, 5]  // Q0: [0,3), Q1: [3,5)

Expected:
  distances[0] = 0.0  // Q0 vs slot 0 (exact match)
  distances[3] should be small for Q1 vs slot 6 if that's close
  outputSlots = [0, 2, 4, 6, 8]  // The storage slots
```

### 2. Single Query

```
Setup:
  - 1 query, 1000 candidates
  - Verify all distances computed correctly
  - Verify outputSlots match vectorIndices lookup
```

### 3. Many Queries, Few Candidates Each

```
Setup:
  - 100 queries, 10 candidates each
  - Verify binary search correctly identifies query for each thread
  - Verify no cross-query contamination
```

### 4. Large Dimension

```
Setup:
  - dimension = 768 or 1536 (typical embedding sizes)
  - Verify SIMD path handles remainder correctly
  - Verify no precision issues
```

### 5. Sparse Storage

```
Setup:
  - Storage has gaps (deleted vectors)
  - vectorIndices points to non-contiguous slots
  - Verify indirection produces correct distances
```

---

## Swift Integration

### Kernel Wrapper Interface

```swift
public struct IVFIndirectionDistanceKernel {
    private let pipelineState: MTLComputePipelineState
    private let device: MTLDevice

    public init(context: Metal4Context) throws {
        // Load kernel from library
    }

    public func execute(
        queries: MTLBuffer,           // [Q × D] float
        vectors: MTLBuffer,           // [capacity × D] float (global storage)
        vectorIndices: MTLBuffer,     // [totalIVF] uint32
        candidateIndices: MTLBuffer,  // [totalCand] uint32
        candidateOffsets: MTLBuffer,  // [Q + 1] uint32
        params: IVFDistanceParams,
        commandBuffer: MTLCommandBuffer
    ) throws -> (distances: MTLBuffer, slots: MTLBuffer) {
        // Allocate output buffers
        // Encode compute command
        // Return output buffers
    }
}

public struct IVFDistanceParams {
    public let numQueries: UInt32
    public let dimension: UInt32
    public let totalCandidates: UInt32
    public let storageCapacity: UInt32
}
```

### Usage in IVF Pipeline

```swift
// In IVFSearchPipeline.swift
func searchWithIndirection(...) async throws -> IVFSearchResult {
    // 1. Build candidate lists (CPU, small data)
    let (candidateIndices, candidateOffsets) = buildCandidateLists(...)

    // 2. Compute distances (GPU, this kernel)
    let (distances, slots) = try await ivfDistanceKernel.execute(
        queries: queryBuffer,
        vectors: storage.buffer,
        vectorIndices: ivfStructure.vectorIndicesBuffer,
        candidateIndices: candidateIndicesBuffer,
        candidateOffsets: candidateOffsetsBuffer,
        params: params,
        commandBuffer: commandBuffer
    )

    // 3. Select top-k per query (GPU, existing kernel)
    var results: [QueryResult] = []
    for q in 0..<numQueries {
        let start = candidateOffsets[q]
        let end = candidateOffsets[q + 1]
        let count = end - start

        // Slice the distances/slots for this query
        let queryDistances = distances.slice(start..<end)
        let querySlots = slots.slice(start..<end)

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

---

## File Locations

- **Metal shader:** `Sources/VectorAccelerate/Metal/Shaders/IVFIndirectionDistance.metal`
- **Swift wrapper:** `Sources/VectorAccelerate/Index/Kernels/IVF/IVFIndirectionDistanceKernel.swift`
- **Tests:** `Tests/VectorAccelerateTests/IVFIndirectionDistanceKernelTests.swift`

---

## Acceptance Criteria

- [ ] Kernel compiles without warnings
- [ ] Basic correctness test passes (exact match has distance 0)
- [ ] Binary search for query index is correct
- [ ] SIMD vectorization for dimension % 4 == 0
- [ ] Handles arbitrary dimensions (not just multiples of 4)
- [ ] outputSlots correctly reflects the indirection lookup
- [ ] No cross-query contamination (each thread only accesses its query)
- [ ] Performance: >1M candidates/second on M1/M2 for D=768

---

## Notes for Implementer

1. **This kernel does NOT do top-k selection.** It only computes distances. Selection is handled by an existing, tested kernel.

2. **The binary search is critical for correctness.** Each thread must identify which query it belongs to based on the CSR offsets.

3. **Memory layout is column-major for queries/vectors** (standard for this codebase). Element `[row, col]` is at index `row * stride + col`.

4. **The `vectorIndices` buffer may be larger than `totalCandidates`** — it contains mappings for ALL IVF entries, not just the current candidates.

5. **Output slots are storage slots, not IVF entry indices.** The Swift layer uses these to map back to stable handles via `slotToStableID`.
