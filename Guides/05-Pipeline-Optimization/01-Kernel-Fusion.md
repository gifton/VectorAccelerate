# 5.1 Kernel Fusion

> **Combining operations to eliminate memory bottlenecks.**

---

## The Concept

Kernel fusion combines multiple operations into a single GPU dispatch, eliminating intermediate memory traffic:

```
Unfused (separate kernels):
  Kernel 1: Read A, Compute B, Write B to memory
  Kernel 2: Read B from memory, Compute C, Write C

Fused (single kernel):
  Kernel: Read A, Compute B (in registers), Compute C, Write C

Memory traffic reduced by half (or more)!
```

---

## Why It Matters

Most vector search workloads are **memory-bound**:

```
L2 distance → Top-K pipeline:

Unfused:
  1. Distance kernel: Read Q×D + N×D, Write Q×N distances
  2. Top-K kernel: Read Q×N distances, Write Q×K results

  For Q=100, N=1M, D=768:
    Distance reads:  (100×768 + 1M×768) × 4 = 3.1 GB
    Distance writes: 100×1M × 4 = 400 MB
    Top-K reads:     400 MB
    Top-K writes:    100×10 × 8 = 8 KB
    ────────────────────────────────────────────
    Total: ~3.9 GB memory traffic

Fused:
  Single kernel: Read Q×D + N×D, Write Q×K results
    Reads:  3.1 GB
    Writes: 8 KB
    ────────────────────────────────────────────
    Total: ~3.1 GB memory traffic

20% reduction just from fusion!
```

---

## The Technique: FusedL2TopKKernel

VectorAccelerate's `fused_l2_topk` kernel is the prime example:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:202-330

kernel void fused_l2_topk(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    device uint* result_indices [[buffer(2)]],
    device float* result_distances [[buffer(3)]],
    // ...
) {
    // Each threadgroup handles one query
    threadgroup float query_cached[MAX_D];
    Candidate private_heap[K_PRIVATE];

    // Cache query (shared by all threads in group)
    for (uint d = tid; d < D; d += tgs) {
        query_cached[d] = queries[q_id * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process ALL database vectors, maintaining heap
    for (uint n_idx = tid; n_idx < N; n_idx += tgs) {
        // Compute distance (in registers)
        float dist = calculate_l2_squared(query_cached, dataset + n_idx * D, D);

        // Update heap immediately (no memory write)
        update_private_heap_sorted(private_heap, dist, n_idx);
    }

    // Only write final K results
    // ...
}
```

### Key Optimizations

1. **Query cached in threadgroup memory**: Loaded once, used for all database vectors
2. **Distance in registers**: Never written to device memory
3. **Heap in registers**: Only final K results written out

---

## When Fusion Helps

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FUSION DECISION TREE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Are operations producer-consumer?                                  │
│  (Output of A is only input to B)                                   │
│       │                                                              │
│       ├── NO → Fusion doesn't help much                             │
│       │                                                              │
│       └── YES → Is intermediate data large?                         │
│                      │                                               │
│                      ├── NO → Fusion helps a little                 │
│                      │        (reduces kernel launch overhead)       │
│                      │                                               │
│                      └── YES → FUSION HELPS A LOT                   │
│                               (eliminates memory traffic)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Good Candidates for Fusion

- **Distance + Top-K**: Large intermediate (Q×N distances)
- **Normalize + Distance**: Normalized vectors reused immediately
- **Distance + Threshold**: Filter before expensive operations

### Poor Candidates

- **Distance + Multiple Different Selections**: Need to reuse distances
- **Independent Operations**: No data dependency

---

## Implementing Custom Fusion

Pattern for fusing operations in Metal:

```metal
// Template for fused kernels
kernel void fused_operation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    // ...
) {
    // Step 1: Load input into fast memory
    threadgroup float shared_input[MAX_SIZE];
    // Cooperative load...
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Perform operation A (in registers)
    float intermediate_result = /* operation A on shared_input */;

    // Step 3: Perform operation B (in registers)
    float final_result = /* operation B on intermediate_result */;

    // Step 4: Write only final result
    if (should_output) {
        output[out_idx] = final_result;
    }
}
```

---

## Fusion in VectorAccelerate

Available fused kernels:

| Kernel | Operations Fused | Memory Savings |
|--------|-----------------|----------------|
| `FusedL2TopKKernel` | L2 distance + Top-K | ~50% (no distance matrix) |

📍 See: `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift`

---

## 🔗 VectorCore Connection

VectorCore's operations are inherently unfused:

```swift
// VectorCore: Separate operations
let distances = database.map { l2DistanceSquared(query, $0) }  // Pass 1
let topK = selectTopK(distances, k: k)                          // Pass 2
```

This is fine for CPU where cache handles intermediate data well.

---

## 🔗 VectorIndex Connection

VectorIndex could benefit from fusion but operates on CPU:

```swift
// VectorIndex: Typically unfused
let distances = computeDistances(query, database)
let sorted = distances.sorted()
let topK = sorted.prefix(k)
```

VectorAccelerate's GPU fusion eliminates the memory bottleneck.

---

## Performance Impact

100 queries × 1M database × 768D:

| Approach | Distance | Top-K | Total | Memory Traffic |
|----------|----------|-------|-------|----------------|
| Unfused | 10 ms | 4 ms | 14 ms | ~3.9 GB |
| Fused | - | - | 8 ms | ~3.1 GB |

**43% faster** from fusion alone.

---

## Key Takeaways

1. **Fusion eliminates intermediate writes**: Huge win for memory-bound operations

2. **Cache inputs in fast memory**: Threadgroup memory for shared data, registers for per-thread

3. **Only write final results**: Skip all intermediate outputs

4. **Best for producer-consumer pairs**: Output of A only used by B

---

## Next Up

Overlapping CPU and GPU work:

**[→ 5.2 Async Compute](./02-Async-Compute.md)**

---

*Guide 5.1 of 5.4 • Chapter 5: Pipeline Optimization*
