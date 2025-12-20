// VectorAccelerate: Search and Retrieval Shaders
//
// GPU kernels for top-k selection and search operations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Common Structs

// Helper struct for carrying value and index
struct IndexedValue {
    float value;
    uint index;
};

// MARK: - Part 1: Top-K Selection (Heap-based O(N logK))

enum SelectionMode : uint8_t {
    SELECT_MIN = 0,
    SELECT_MAX = 1
};

struct TopKBatchParams {
    uint32_t batch_size;       // Q
    uint32_t num_elements;     // N
    uint32_t k;                // K
    uint32_t input_stride;
    uint32_t output_stride;
    uint8_t mode;
    uint8_t sorted;
    uint8_t padding[2];
};

// Comparison for Heap: Returns true if 'a' should be closer to the root than 'b'.
// SELECT_MIN uses a Max-Heap (root is the largest/worst).
// SELECT_MAX uses a Min-Heap (root is the smallest/worst).
bool compare_heap(float a, float b, SelectionMode mode) {
    return (mode == SELECT_MIN) ? (a > b) : (a < b);
}

// Heapify down (sift down) operation
void heapify_down(thread IndexedValue* heap, uint k, uint idx, SelectionMode mode) {
    while (true) {
        uint left = 2 * idx + 1;
        uint right = 2 * idx + 2;
        uint prioritized = idx;

        if (left < k && compare_heap(heap[left].value, heap[prioritized].value, mode)) {
            prioritized = left;
        }
        if (right < k && compare_heap(heap[right].value, heap[prioritized].value, mode)) {
            prioritized = right;
        }

        if (prioritized != idx) {
            // Swap
            IndexedValue temp = heap[idx];
            heap[idx] = heap[prioritized];
            heap[prioritized] = temp;
            idx = prioritized;
        } else {
            break;
        }
    }
}

// Sort the final heap using Heap Sort (in-place) if sorted output is requested.
void sort_heap(thread IndexedValue* heap, uint k, SelectionMode mode) {
    // We perform the extraction phase of Heap Sort.
    // The heap is already built. We repeatedly extract the root (worst element)
    // and place it at the end of the array, then re-heapify the remaining elements.
    for (int i = k - 1; i > 0; i--) {
        // Move current root to the end
        IndexedValue temp = heap[0];
        heap[0] = heap[i];
        heap[i] = temp;

        // Call heapify on the reduced heap
        heapify_down(heap, i, 0, mode);
    }
    // The result is now sorted from best to worst.
}

// (Spec Section: Metal Kernel Signatures - topk_select_batch_kernel)
// Batched top-k using a register-based heap. Optimized for K <= 128.
kernel void topk_select_batch_kernel(
    device const float* distances [[buffer(0)]],
    device float* topk_values [[buffer(1)]],
    device uint* topk_indices [[buffer(2)]],
    constant TopKBatchParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]] // 1D dispatch over batch_size (Q)
) {
    if (tid >= params.batch_size) return;

    const uint K = params.k;
    const uint N = params.num_elements;
    const SelectionMode MODE = SelectionMode(params.mode);

    if (K == 0) return;

    // Constraint: Heap must fit efficiently in registers.
    constexpr uint MAX_K = 128;
    if (K > MAX_K) return;

    IndexedValue heap[MAX_K];
    uint heap_size = 0;

    device const float* input_row = distances + tid * params.input_stride;

    // Phase 1: Fill the initial heap (O(K))
    for (uint i = 0; i < K && i < N; ++i) {
        heap[i] = {input_row[i], i};
        heap_size++;
    }

    // Build the heap (Bottom-up construction: O(K))
    for (int i = heap_size / 2 - 1; i >= 0; --i) {
        heapify_down(heap, heap_size, i, MODE);
    }

    // Phase 2: Stream the remaining elements (O((N-K) * logK))
    for (uint i = K; i < N; ++i) {
        float val = input_row[i];
        
        // Check if the new element is better than the root (the worst element so far)
        bool should_insert = (MODE == SELECT_MIN) ? (val < heap[0].value) : (val > heap[0].value);

        if (should_insert) {
            // Replace the root and sift down
            heap[0] = {val, i};
            heapify_down(heap, heap_size, 0, MODE);
        }
    }

    // Phase 3: Sort the results (if requested) (O(K logK))
    if (params.sorted) {
        sort_heap(heap, heap_size, MODE);
    }

    // Phase 4: Write results to global memory
    device float* output_val_row = topk_values + tid * params.output_stride;
    device uint* output_idx_row = topk_indices + tid * params.output_stride;

    for (uint i = 0; i < heap_size; ++i) {
        output_val_row[i] = heap[i].value;
        output_idx_row[i] = heap[i].index;
    }
    
    // Handle case where N < K (pad the output)
    for (uint i = heap_size; i < K; ++i) {
        output_val_row[i] = (MODE == SELECT_MIN) ? INFINITY : -INFINITY;
        output_idx_row[i] = UINT_MAX;
    }
}

// MARK: - Part 2: Parallel Reduction (Optimized Tree Reduction with SIMD)

enum ReductionOp : uint8_t {
    REDUCE_SUM = 0,
    REDUCE_MIN = 2,
    REDUCE_MAX = 3,
    REDUCE_ARGMIN = 4,
    REDUCE_ARGMAX = 5,
};

struct ReductionParams {
    uint32_t num_elements;
    uint32_t stride; // Stride=1 assumed in this implementation.
    uint8_t operation;
    uint8_t return_index; // 1 if ArgMin/ArgMax
    uint8_t padding[2];
    float initial_value;
};

// MARK: Reduction Helpers

// Function to apply the reduction operation on IndexedValue
IndexedValue apply_op(IndexedValue a, IndexedValue b, ReductionOp op) {
    // Handle identity elements (represented by index UINT_MAX)
    if (a.index == UINT_MAX) return b;
    if (b.index == UINT_MAX) return a;

    switch (op) {
        case REDUCE_SUM:
            return {a.value + b.value, 0}; // Index is irrelevant for SUM
        
        // Stable reduction: prefer smaller index if values are equal
        case REDUCE_MIN:
        case REDUCE_ARGMIN:
            if (b.value < a.value || (b.value == a.value && b.index < a.index)) {
                return b;
            }
            return a;
            
        case REDUCE_MAX:
        case REDUCE_ARGMAX:
            if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
                return b;
            }
            return a;
            
        default:
            return a;
    }
}

// Optimized warp-level reduction using SIMD group functions (shuffle)
IndexedValue warp_reduce(IndexedValue value, ReductionOp op) {
    // Butterfly reduction pattern within the SIMD group (typically 32 threads)
    // Assumes SIMD group size is 32.
    for (uint offset = 16; offset > 0; offset >>= 1) {
        IndexedValue other;
        // Shuffle both value and index simultaneously
        other.value = simd_shuffle_down(value.value, offset);
        other.index = simd_shuffle_down(value.index, offset);
        value = apply_op(value, other, op);
    }
    return value;
}

// (Spec Section: Metal Kernel Signatures - parallel_reduce_kernel)
// Generic reduction kernel (Handles one pass). Supports multi-pass ArgMin/ArgMax.
kernel void parallel_reduce_kernel(
    device const float* input_values [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]], // Input indices (for multi-pass ArgMin/Max)
    device float* output_values [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],      // Output indices
    constant ReductionParams& params [[buffer(4)]],
    threadgroup IndexedValue* shared_data [[threadgroup(0)]],
    uint3 tid3 [[thread_position_in_grid]],
    uint3 local_id3 [[thread_position_in_threadgroup]],
    uint3 group_id3 [[threadgroup_position_in_grid]],
    uint3 grid_size3 [[threads_per_grid]],
    uint3 threadgroup_size3 [[threads_per_threadgroup]]
) {
    // Extract scalar components for 1D dispatch usage
    const uint tid = tid3.x;
    const uint local_id = local_id3.x;
    const uint group_id = group_id3.x;
    const uint grid_size = grid_size3.x;
    const uint threadgroup_size = threadgroup_size3.x;

    const uint N = params.num_elements;
    const ReductionOp OP = ReductionOp(params.operation);

    // Phase 1: Sequential reduction per thread (Grid-stride loop)
    // Initialize index to UINT_MAX (identity element marker).
    IndexedValue thread_result = {params.initial_value, UINT_MAX};

    // Coalesced memory access (assuming stride=1)
    for (uint i = tid; i < N; i += grid_size) {
        float val = input_values[i];
        uint idx = i;
        
        // If performing indexed reduction and input indices are provided (multi-pass), use them.
        if (params.return_index && input_indices != nullptr) {
            idx = input_indices[i];
        }
        
        IndexedValue current = {val, idx};
        thread_result = apply_op(thread_result, current, OP);
    }

    // Phase 2: Store to shared memory
    shared_data[local_id] = thread_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Tree reduction in shared memory (down towards warp size 32)
    // Host ensures threadgroup_size is a power of 2 and >= 32.
    for (uint s = threadgroup_size / 2; s >= 32; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] = apply_op(shared_data[local_id], shared_data[local_id + s], OP);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 4: Final warp reduction
    if (local_id < 32) {
        IndexedValue val = shared_data[local_id];

        // Use optimized SIMD shuffle reduction
        val = warp_reduce(val, OP);

        // Phase 5: Write result
        if (local_id == 0) {
            output_values[group_id] = val.value;
            if (params.return_index && output_indices != nullptr) {
                output_indices[group_id] = val.index;
            }
        }
    }
}

// MARK: - Part 3: Top-K Merge (Sorted Lists)

// Merges two sorted (best-to-worst) top-k lists into a single top-k list.
//
// This is used by the chunked K>8 fallback path to avoid CPU merges:
// - running_* holds the best-so-far [Q × K]
// - chunk_* holds the best for the current chunk [Q × chunkK]
//
// Output is sorted (best-to-worst) by value, with stable tie-break on index.

struct TopKMergeParams {
    uint32_t num_queries;   // Q
    uint32_t k;             // K (output length)
    uint32_t chunk_k;       // chunkK (input length)
    uint32_t chunk_base;    // base offset to add to chunk indices
};

inline bool va_merge_is_better(float distA, uint idxA, float distB, uint idxB) {
    if (distA < distB) return true;
    if (distA > distB) return false;
    return idxA < idxB;
}

// (Spec Section: Metal Kernel Signatures - merge_topk_sorted_kernel)
kernel void merge_topk_sorted_kernel(
    device const uint* running_indices [[buffer(0)]],
    device const float* running_distances [[buffer(1)]],
    device const uint* chunk_indices [[buffer(2)]],
    device const float* chunk_distances [[buffer(3)]],
    device uint* out_indices [[buffer(4)]],
    device float* out_distances [[buffer(5)]],
    constant TopKMergeParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_queries) return;

    // Use standard constants (VA_INVALID_INDEX = 0xFFFFFFFF, VA_INFINITY = INFINITY)
    constexpr uint INVALID_INDEX = 0xFFFFFFFF;

    const uint K = params.k;
    const uint Kc = params.chunk_k;
    const uint base = params.chunk_base;

    // Strides: row-major contiguous
    const uint running_row = tid * K;
    const uint chunk_row = tid * Kc;

    uint i = 0;
    uint j = 0;

    for (uint out = 0; out < K; ++out) {
        // Running candidate
        uint idxA = INVALID_INDEX;
        float distA = INFINITY;
        if (i < K) {
            idxA = running_indices[running_row + i];
            distA = running_distances[running_row + i];
            if (idxA == INVALID_INDEX) {
                distA = INFINITY;
            }
        }

        // Chunk candidate (convert to global index)
        uint idxB = INVALID_INDEX;
        float distB = INFINITY;
        if (j < Kc) {
            uint local = chunk_indices[chunk_row + j];
            distB = chunk_distances[chunk_row + j];

            if (local != INVALID_INDEX) {
                idxB = local + base;
            } else {
                idxB = INVALID_INDEX;
                distB = INFINITY;
            }
        }

        const bool takeA = va_merge_is_better(distA, idxA, distB, idxB);

        if (takeA) {
            out_indices[running_row + out] = idxA;
            out_distances[running_row + out] = distA;
            i += 1;
        } else {
            out_indices[running_row + out] = idxB;
            out_distances[running_row + out] = distB;
            j += 1;
        }
    }
}

// MARK: - Part 4: IVF Candidate Distance (Indirection-Aware)

// Computes L2 squared distances from queries to a per-query candidate set, where
// candidates are expressed as IVF entry indices that must be mapped through
// `vectorIndices` to obtain the underlying storage slot in the global vector
// buffer.
//
// This kernel intentionally avoids any CPU-side vector gathering. The CPU builds
// a compact candidate list (CSR) and a parallel `candidateQueryIds` array so each
// candidate can be processed independently on GPU.

struct IVFIndirectionDistanceParams {
    uint32_t dimension;         // D
    uint32_t total_candidates;  // total number of candidate entries across all queries
    uint32_t num_queries;       // Q
    uint32_t total_ivf_entries; // length of vectorIndices (safety bound)
    uint32_t storage_capacity;  // number of slots in the global vector buffer
    uint32_t padding0;
    uint32_t padding1;
    uint32_t padding2;
};

kernel void ivf_distance_with_indirection(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device const uint* vectorIndices [[buffer(2)]],
    device const uint* candidateIVFIndices [[buffer(3)]],
    device const uint* candidateQueryIds [[buffer(4)]],
    device float* outDistances [[buffer(5)]],
    device uint* outSlots [[buffer(6)]],
    constant IVFIndirectionDistanceParams& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.total_candidates) return;

    uint q = candidateQueryIds[tid];
    uint ivfEntry = candidateIVFIndices[tid];

    // Safety guards to avoid OOB reads if CPU inputs are malformed.
    if (q >= params.num_queries || ivfEntry >= params.total_ivf_entries) {
        outDistances[tid] = INFINITY;
        outSlots[tid] = UINT_MAX;
        return;
    }

    uint slot = vectorIndices[ivfEntry];
    if (slot >= params.storage_capacity) {
        outDistances[tid] = INFINITY;
        outSlots[tid] = UINT_MAX;
        return;
    }

    const uint D = params.dimension;
    const uint qBase = q * D;
    const uint vBase = slot * D;

    float dist = 0.0f;
    for (uint d = 0; d < D; ++d) {
        float diff = queries[qBase + d] - vectors[vBase + d];
        dist += diff * diff;
    }

    outDistances[tid] = dist;
    outSlots[tid] = slot;
}
