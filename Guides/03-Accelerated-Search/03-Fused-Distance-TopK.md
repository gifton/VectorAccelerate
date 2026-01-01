# 3.3 Fused Distance+TopK

> **Single-pass search—computing distances and selecting top-K without the distance matrix.**

---

## The Concept

Instead of:
1. Compute Q×N distance matrix
2. Select top-K from each row

Fused kernels do both in a single pass:
1. Each threadgroup processes one query against all database vectors
2. Maintains a running top-K as it goes
3. No distance matrix ever materialized

---

## Why It Matters

The two-pass approach has a critical flaw: **memory bandwidth**

```
Two-pass approach:
  Pass 1: Compute distances, WRITE Q×N matrix to device memory
  Pass 2: Select top-K, READ Q×N matrix from device memory

  For Q=100, N=1M:
    Write: 400 MB
    Read:  400 MB
    Total: 800 MB memory traffic

Fused approach:
  Single pass: Compute distance, update local heap (registers)
  Only write: Q×K results

  For Q=100, N=1M, K=10:
    Write: 100 × 10 × 8 = 8 KB
    Total: 8 KB memory traffic

800 MB vs 8 KB = 100,000× less memory traffic!
```

This is why fused kernels are much faster for typical search workloads.

---

## The Technique: Threadgroup Per Query

The key insight: **one threadgroup processes one query**

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:204-330

kernel void fused_l2_topk(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    device uint* result_indices [[buffer(2)]],
    device float* result_distances [[buffer(3)]],
    constant uint& Q [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint q_id [[threadgroup_position_in_grid]],  // Query index
    uint tid [[thread_index_in_threadgroup]],    // Thread within group
    uint tgs [[threads_per_threadgroup]]         // Threadgroup size
) {
    if (q_id >= Q) return;

    // Shared memory for this threadgroup
    threadgroup float query_cached[MAX_D];
    threadgroup Candidate shared_candidates[MAX_SHARED_CANDIDATES_POT];

    // Each thread maintains a private heap of K_PRIVATE candidates
    Candidate private_heap[K_PRIVATE];  // K_PRIVATE = 8
    for (uint i = 0; i < K_PRIVATE; ++i) {
        private_heap[i] = {INFINITY, SENTINEL_INDEX};
    }

    // Cache query in shared memory (cooperative load)
    device const float* query_ptr = queries + q_id * D;
    for (uint d = tid; d < D; d += tgs) {
        query_cached[d] = query_ptr[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process ALL database vectors
    for (uint n_idx = tid; n_idx < N; n_idx += tgs) {
        // Compute distance
        float dist = calculate_l2_squared(query_cached, dataset + n_idx * D, D);

        // Update private heap
        update_private_heap_sorted(private_heap, dist, n_idx);
    }

    // Merge all private heaps in shared memory
    // ... (see below)

    // Write final K results
    // ...
}
```

### Step-by-Step Execution

```
Threadgroup for Query 0 (256 threads):

Step 1: Cache query in shared memory
┌─────────────────────────────────────────────────────────────────┐
│  Thread 0 loads query[0]                                        │
│  Thread 1 loads query[1]                                        │
│  ...                                                            │
│  Thread 255 loads query[255]                                    │
│  (Thread 256+ load remaining dimensions in next round)          │
└─────────────────────────────────────────────────────────────────┘
↓ BARRIER ↓

Step 2: Each thread processes N/256 database vectors
┌─────────────────────────────────────────────────────────────────┐
│  Thread 0: d[0], d[256], d[512], ... → private_heap[0]         │
│  Thread 1: d[1], d[257], d[513], ... → private_heap[1]         │
│  ...                                                            │
│  Thread 255: d[255], d[511], d[767], ... → private_heap[255]   │
└─────────────────────────────────────────────────────────────────┘

Step 3: Merge 256 private heaps (each with 8 candidates)
┌─────────────────────────────────────────────────────────────────┐
│  256 × 8 = 2048 candidates → sort → take top K                 │
└─────────────────────────────────────────────────────────────────┘

Step 4: Write K results to output
```

---

## Private Heap Maintenance

Each thread maintains a small, sorted list of best candidates:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:86-103

inline void update_private_heap_sorted(
    thread Candidate* heap,
    float new_dist,
    uint new_id
) {
    // Check if new candidate is better than worst in heap
    Candidate worst = heap[K_PRIVATE - 1];
    if (new_dist < worst.distance ||
        (new_dist == worst.distance && new_id < worst.index)) {

        // Replace worst with new candidate
        heap[K_PRIVATE - 1] = {new_dist, new_id};

        // Insertion sort to maintain sorted order
        #pragma unroll
        for (uint i = K_PRIVATE - 1; i > 0; --i) {
            if (is_better(heap[i], heap[i-1])) {
                // Swap
                Candidate tmp = heap[i];
                heap[i] = heap[i-1];
                heap[i-1] = tmp;
            } else {
                break;  // Already in position
            }
        }
    }
}
```

With K_PRIVATE=8, this is O(1) amortized—most candidates are rejected immediately.

---

## Heap Merging Strategies

After processing, we need to merge 256 heaps (each with 8 candidates):

### For Small K (≤32): Parallel Reduction

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:266-294

if (K_emit <= 32) {
    threadgroup BestCand scratch[MAX_TGS];

    for (uint sel = 0; sel < K_emit; ++sel) {
        // Each thread finds minimum in its portion
        BestCand local = {INFINITY, SENTINEL_INDEX, 0};
        for (uint i = tid; i < pow2_size; i += tgs) {
            Candidate c = shared_candidates[i];
            BestCand cur = {c.distance, c.index, i};
            local = reduce_min(local, cur);
        }

        // Parallel reduction across threads
        BestCand winner = parallel_min_reduce(scratch, local, tid, tgs);

        // Thread 0 writes result and marks candidate as used
        if (tid == 0) {
            result_indices[output_offset + sel] = winner.index;
            result_distances[output_offset + sel] = winner.distance;
            shared_candidates[winner.pos].distance = INFINITY;  // Mark used
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
```

### For Large K (>32): Bitonic Sort

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:303-321

else {
    // Sort all candidates
    block_bitonic_sort(shared_candidates, pow2_size, tid, tgs);

    // Extract top K
    for (uint k = tid; k < K_emit; k += tgs) {
        Candidate result = shared_candidates[k];
        result_indices[output_offset + k] = result.index;
        result_distances[output_offset + k] = result.distance;
    }
}
```

---

## Using FusedL2TopKKernel

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift

import VectorAccelerate

let context = try await Metal4Context()
let kernel = try await FusedL2TopKKernel(context: context)

// Single function: distances + selection
let result = try await kernel.execute(
    queries: queryBuffer,    // [Q × D]
    dataset: datasetBuffer,  // [N × D]
    parameters: FusedL2TopKParameters(
        numQueries: 100,
        numDataset: 1_000_000,
        dimension: 768,
        k: 10
    )
)

// Result contains indices and distances
for queryIdx in 0..<100 {
    let topK = result.results(for: queryIdx)
    // topK: [(index: Int, distance: Float)]
}
```

---

## Performance Comparison

100 queries × 1M database × 768D × K=10 (M2 Max):

| Approach | Distance | Selection | Total | Memory Traffic |
|----------|----------|-----------|-------|----------------|
| Two-pass | 10 ms | 4 ms | 14 ms | 800 MB |
| Fused | - | - | 8 ms | ~8 KB |

**43% faster** with fused kernel, and much better for memory-constrained scenarios.

---

## 🔗 VectorCore Connection

VectorCore doesn't have fused kernels—it's a building-block library. But the concept applies:

```swift
// VectorCore: Separate operations
let distances = database.map { l2DistanceSquared(query, $0) }
let topK = distances.enumerated().sorted { $0.1 < $1.1 }.prefix(k)

// Ideal: Fused operation (not in VectorCore)
// var topK = Heap()
// for vec in database {
//     let dist = l2DistanceSquared(query, vec)
//     topK.insertIfBetter(dist)
// }
```

The GPU fused kernel achieves this but with thousands of parallel workers.

---

## 🔗 VectorIndex Connection

VectorIndex's FlatIndex could benefit from fusion:

```swift
// Current VectorIndex: Two passes
public func search(query: [Float], k: Int) -> [SearchResult] {
    let distances = vectors.map { distance(query, $0) }  // Pass 1
    return distances.sorted().prefix(k)                   // Pass 2
}

// VectorAccelerate: Single fused pass
let results = try await fusedKernel.execute(query, database, k)
```

---

## Limitations and Trade-offs

### When Fused Kernels Are Best

✅ Small to medium K (≤ 128)
✅ Standard L2 distance
✅ Don't need the distance matrix for other purposes

### When Two-Pass Is Better

❌ Need distance matrix for multiple different K values
❌ Custom distance metrics not in fused kernel
❌ Very large K where bitonic sort in fused kernel is slow
❌ Need to reuse distances for other operations

---

## Key Takeaways

1. **Avoid materializing the distance matrix**: Fused kernels eliminate the biggest memory bottleneck

2. **One threadgroup per query**: Natural parallelism across queries

3. **Private heaps reduce synchronization**: Each thread works independently until merge

4. **Merge strategy depends on K**: Reduction for small K, sort for large K

5. **43% faster**: Typical speedup from fusion

---

## Next Up

Accelerating approximate search with IVF:

**[→ 3.4 IVF Acceleration](./04-IVF-Acceleration.md)**

---

*Guide 3.3 of 3.4 • Chapter 3: Accelerated Search*
