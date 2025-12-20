# 3.2 Top-K Selection on GPU

> **Parallel algorithms for finding the K smallest values—the second half of search.**

---

## The Concept

Given N distance values, find the K smallest:

```
Input:  [0.23, 1.45, 0.01, 0.89, 0.12, 2.34, 0.05, 0.67, ...]  (N values)
Output: [(2, 0.01), (6, 0.05), (4, 0.12)]                       (K smallest)
```

On CPU, this is O(N log K) with a heap. On GPU, we need different approaches that exploit parallelism.

---

## Why It Matters

Selection seems simple, but it's a significant portion of search time:

```
1M vectors, K=100:

Distance computation: ~12 ms (highly parallel)
Top-K selection (CPU): ~45 ms (sequential heap)
Top-K selection (GPU): ~0.8 ms (parallel)

Without GPU selection, you'd spend 79% of time on selection!
```

---

## The Techniques: Selection Algorithms

VectorAccelerate provides GPU selection algorithms optimized for different scenarios:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELECTION ALGORITHM CHOICE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RECOMMENDED: Use FusedL2TopKKernel (combines distance + selection) │
│                                                                      │
│  For standalone selection from pre-computed distances:              │
│                                                                      │
│  K ≤ 32?                                                            │
│    │                                                                 │
│    └── YES → WARP-OPTIMIZED SELECTION                               │
│              Uses SIMD group operations                              │
│              O(N/32) per query                                       │
│                                                                      │
│    └── NO → K ≤ 128?                                                │
│                │                                                     │
│                └── YES → HEAP-BASED (via TopKSelectionKernel)       │
│                          Private max-heap per thread                 │
│                          O(N log K) per thread                       │
│                                                                      │
│                └── NO → BITONIC SORT                                │
│                          Full parallel sort, extract K               │
│                          O(N log² N) total                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Recommendation**: For most use cases, prefer `FusedL2TopKKernel` over separate distance + selection kernels. It avoids materializing the full distance matrix and provides better performance.

---

## Algorithm 1: Warp-Optimized Selection (K ≤ 32)

For small K, we use **Metal 4 SIMD group intrinsics** to cooperatively find the best. This leverages `simd_shuffle_xor` for register-level communication between threads:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:534-589

kernel void warp_select_small_k_ascending(
    device const float* distances [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& queryCount [[buffer(3)]],
    constant uint& candidateCount [[buffer(4)]],
    constant uint& k_param [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint query_idx = gid.y;
    if (query_idx >= queryCount) return;

    const uint K = min(k_param, candidateCount);

    // Each lane maintains its own sorted list of K candidates
    IndexDistance local_k[K4_MAX_K];

    // Initialize with worst possible values
    for (uint i = 0; i < K; ++i) {
        local_k[i] = {SENTINEL_INDEX, INFINITY};
    }

    // Each lane processes every 32nd element
    const device float* query_distances = distances + query_idx * candidateCount;

    for (uint i = lane_id; i < candidateCount; i += 32) {
        float dist = query_distances[i];
        IndexDistance candidate = {i, dist};

        // Insert into sorted local list
        insert_local_topk(local_k, K, candidate);
    }

    // SIMD group merge: combine 32 local lists into one
    IndexDistance partner_k[K4_MAX_K];

    for (uint stride = 1; stride < 32; stride *= 2) {
        // Exchange with partner lane
        for (uint i = 0; i < K; ++i) {
            partner_k[i].index = simd_shuffle_xor(local_k[i].index, stride);
            partner_k[i].distance = simd_shuffle_xor(local_k[i].distance, stride);
        }

        // Merge two sorted lists of K elements → keep K best
        k4_merge_registers(local_k, partner_k, K);
    }

    // Lane 0 has the final result
    if (lane_id < K) {
        indices[query_idx * k_param + lane_id] = local_k[lane_id].index;
        values[query_idx * k_param + lane_id] = local_k[lane_id].distance;
    }
}
```

### How It Works

```
Step 1: Each lane processes N/32 candidates
┌────┬────┬────┬────────┬────┐
│L0  │L1  │L2  │ ...    │L31 │
│top8│top8│top8│        │top8│
└────┴────┴────┴────────┴────┘

Step 2: Pairwise merge with simd_shuffle_xor
Round 1 (stride=1):  L0↔L1, L2↔L3, ...
Round 2 (stride=2):  L0↔L2, L1↔L3, ...
Round 3 (stride=4):  L0↔L4, ...
Round 4 (stride=8):  L0↔L8, ...
Round 5 (stride=16): L0↔L16, ...

After 5 rounds: Lane 0 has global top-K
```

---

## Algorithm 2: Private Heap Selection (K ≤ 128)

> ⚠️ **Note**: The standalone `StreamingTopKKernel` is deprecated due to correctness issues. For production use, prefer `FusedL2TopKKernel` which combines distance computation and Top-K selection in a single pass. The heap concept shown below is still used internally by the fused kernel.

For medium K, each thread maintains a max-heap:

```metal
// Heap concept used in FusedL2TopKKernel's private heap approach

/// Max-heap: largest element at root
/// We track K smallest by keeping a max-heap of size K
/// New candidate enters only if smaller than root (the K-th smallest so far)

struct PrivateMaxHeap {
    float distances[MAX_K_PRIVATE];
    uint indices[MAX_K_PRIVATE];

    void insert(float distance, uint index, uint K) {
        // Only insert if better than current worst (root)
        if (distance < distances[0]) {
            distances[0] = distance;
            indices[0] = index;
            sink_down(distances, indices, K, 0);  // Restore heap property
        }
    }
};
```

The `FusedL2TopKKernel` uses this heap pattern internally with K_PRIVATE=8 elements per thread, then merges heaps in shared memory for the final Top-K.

### Heap Operations

```
Max-heap for K=4:

        [0.45]           ← Root (largest of K smallest)
       /      \
    [0.23]   [0.12]
     /
  [0.01]

Insert 0.08:
  0.08 < 0.45 (root), so replace root
  Sink down: [0.08] swaps with [0.23]
  Result:
        [0.23]
       /      \
    [0.08]   [0.12]
     /
  [0.01]
```

---

## Algorithm 3: Bitonic Sort (K > 128)

For large K, we sort all candidates and take the first K:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:106-127

inline void block_bitonic_sort(
    threadgroup Candidate* data,
    const uint N_PoT,  // N rounded up to power of 2
    const uint tid,
    const uint tgs     // threads per group
) {
    // Bitonic sort: O(log² N) parallel phases
    for (uint k = 2; k <= N_PoT; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < N_PoT; i += tgs) {
                uint partner = i ^ j;

                if (i < partner) {
                    bool direction_ascending = ((i & k) == 0);
                    Candidate c_i = data[i];
                    Candidate c_p = data[partner];

                    bool p_is_better = is_better(c_p, c_i);
                    bool should_swap = (direction_ascending == p_is_better);

                    if (should_swap) {
                        data[i] = c_p;
                        data[partner] = c_i;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

### Bitonic Network Visualization

```
Bitonic sort for N=8:

Initial: [7, 3, 5, 1, 8, 2, 4, 6]

Phase 1 (k=2): Compare pairs
  [3,7] [1,5] [2,8] [4,6]
  [3, 7, 1, 5, 2, 8, 4, 6]

Phase 2 (k=4): Bitonic merge
  j=2: [1,7] [3,5] [2,6] [4,8]
  j=1: [1,3] [5,7] [2,4] [6,8]
  [1, 3, 5, 7, 2, 4, 6, 8]

Phase 3 (k=8): Final merge
  j=4: [1,2] [3,4] [5,6] [7,8]
  j=2: [1,2] [3,4] [5,6] [7,8]
  j=1: [1,2] [3,4] [5,6] [7,8]
  [1, 2, 3, 4, 5, 6, 7, 8]

Fully sorted in O(log² N) parallel steps!
```

---

## Selection Kernel Usage

The TopKSelectionKernel wraps these algorithms:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/TopKSelectionKernel.swift

public final class TopKSelectionKernel: @unchecked Sendable {

    /// Select top-K from pre-computed distances
    public func select(
        distances: any MTLBuffer,
        k: Int,
        numQueries: Int,
        numCandidates: Int,
        ascending: Bool = true  // True for distances (smaller = better)
    ) async throws -> Metal4TopKResult {

        // Choose algorithm based on K
        if k <= 32 {
            return try await warpSelect(...)
        } else if k <= 128 {
            return try await heapSelect(...)
        } else {
            return try await bitonicSelect(...)
        }
    }
}
```

---

## 🔗 VectorCore Connection

CPU selection uses standard library:

```swift
// CPU: Use partial sort or heap
let topK = distances.enumerated()
    .sorted { $0.element < $1.element }
    .prefix(k)

// Or with heap for better complexity:
var heap = Heap<(Int, Float)>(comparator: { $0.1 > $1.1 })
for (i, dist) in distances.enumerated() {
    heap.insert((i, dist))
    if heap.count > k { heap.removeRoot() }
}
```

GPU selection achieves O(N/P) where P is parallelism level.

---

## 🔗 VectorIndex Connection

VectorIndex uses CPU selection after distance computation:

```swift
// VectorIndex: FlatIndex selection
let distances = database.map { distance(query, $0) }
let sorted = distances.enumerated().sorted { $0.element < $1.element }
return sorted.prefix(k)
```

VectorAccelerate parallelizes this:

```swift
// VectorAccelerate: GPU selection
let topK = try await topKKernel.select(
    distances: distanceBuffer,
    k: k,
    numQueries: Q,
    numCandidates: N
)
```

---

## Performance Comparison

Selection of K=10 from 1M candidates:

| Algorithm | CPU | GPU Warp | GPU Heap | GPU Bitonic |
|-----------|-----|----------|----------|-------------|
| Time | 45 ms | 0.8 ms | 1.2 ms | 3.5 ms |
| Speedup | 1× | 56× | 38× | 13× |

For small K, warp selection is dramatically faster.

---

## Key Takeaways

1. **Algorithm choice matters**: K determines optimal approach

2. **Warp selection for small K**: SIMD group operations are extremely fast

3. **Heap for medium K**: Private heaps avoid synchronization

4. **Bitonic for large K**: Fully parallel sort when needed

5. **Avoid full sorts**: Selection is O(N), sorting is O(N log N)

---

## Next Up

The ultimate optimization: fusing distance and selection:

**[→ 3.3 Fused Distance+TopK](./03-Fused-Distance-TopK.md)**

---

*Guide 3.2 of 3.4 • Chapter 3: Accelerated Search*
