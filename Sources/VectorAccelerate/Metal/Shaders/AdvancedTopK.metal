// VectorAccelerate: Advanced Top-K Selection Kernels
//
// GPU-accelerated selection with fusion, streaming, and warp optimization
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Features:
// - Fused L2 distance + top-K selection
// - Streaming top-K for large datasets
// - Warp-optimized selection for small K
// - Bitonic sort for large K values

#include "Metal4Common.h"

// MARK: - Configuration Constants

constexpr constant uint K_PRIVATE = 8;                // Per-thread register heap size
constexpr constant uint MAX_TGS = 256;                // Maximum threadgroup size
constexpr constant uint MAX_D = 768;                  // Maximum dimension for query caching (supports BERT-768)
constexpr constant uint MAX_SHARED_CANDIDATES_POT = 2048; // Max candidates in shared memory
constexpr constant uint SENTINEL_INDEX = 0xFFFFFFFF;  // Invalid index marker
constexpr constant uint MAX_K_PRIVATE = 128;          // Maximum K for streaming
constexpr constant uint K4_MAX_K = 32;                // Maximum K for warp-optimized kernel
constexpr constant uint K4_WARP_SIZE = 32;            // SIMD group size

// MARK: - Common Structures

struct Candidate {
    float distance;
    uint index;
};

struct IndexDistance {
    uint index;
    float distance;
};

struct StreamConfig {
    uint Q;                     // Total number of queries
    uint chunk_size;            // Vectors/distances in this chunk
    uint k_value;               // Final K to return
    ulong chunk_base_index;     // Global index offset (64-bit for large datasets)
};

// MARK: - Helper Functions

inline bool is_better(Candidate a, Candidate b) {
    if (a.distance < b.distance) return true;
    if (a.distance > b.distance) return false;
    return a.index < b.index;
}

inline float4 safe_load_float4(device const float* base, uint offset, uint max) {
    if (offset + 3 < max) {
        return reinterpret_cast<const device float4*>(base + offset)[0];
    }
    float4 v = float4(0.0f);
    for (uint i = 0; i < 4 && offset + i < max; ++i) {
        v[i] = base[offset + i];
    }
    return v;
}

inline float calculate_l2_squared(const threadgroup float* query_cached, device const float* vector_ptr, uint D) {
    float accumulator = 0.0f;
    uint d = 0;
    
    // Vectorized processing
    for (; d + 3 < D; d += 4) {
        float4 q_data = reinterpret_cast<const threadgroup float4*>(query_cached + d)[0];
        float4 v_data = safe_load_float4(vector_ptr, d, D);
        float4 diff = q_data - v_data;
        accumulator += dot(diff, diff);
    }
    
    // Handle remainder
    for (; d < D; ++d) {
        float diff = query_cached[d] - vector_ptr[d];
        accumulator = metal::fma(diff, diff, accumulator);
    }
    
    return accumulator;
}

inline void update_private_heap_sorted(thread Candidate* heap, float new_dist, uint new_id) {
    Candidate worst = heap[K_PRIVATE - 1];
    if (new_dist < worst.distance || (new_dist == worst.distance && new_id < worst.index)) {
        heap[K_PRIVATE - 1] = {new_dist, new_id};
        
        // Insertion sort to maintain order
        #pragma unroll
        for (uint i = K_PRIVATE - 1; i > 0; --i) {
            if (is_better(heap[i], heap[i-1])) {
                Candidate tmp = heap[i];
                heap[i] = heap[i-1];
                heap[i-1] = tmp;
            } else {
                break;
            }
        }
    }
}

// Bitonic sort for shared memory candidates
inline void block_bitonic_sort(threadgroup Candidate* data, const uint N_PoT, const uint tid, const uint tgs) {
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

// Parallel reduction for finding minimum
struct BestCand {
    float distance;
    uint index;
    uint pos;
};

/// Min-heap sink-down helper for descending selection (tracks K largest values)
/// Uses min-heap so root contains smallest of K largest - easy to compare against new candidates
inline void min_heap_sink_down(thread float* heap_data, thread uint* heap_indices, uint heap_size, uint parent_idx) {
    float parent_val = heap_data[parent_idx];
    uint parent_id = heap_indices[parent_idx];

    while (true) {
        uint child_idx = 2 * parent_idx + 1;
        if (child_idx >= heap_size) break;

        // Find smaller child (min-heap)
        if (child_idx + 1 < heap_size && heap_data[child_idx] > heap_data[child_idx + 1]) {
            child_idx++;
        }

        if (parent_val <= heap_data[child_idx]) break;

        heap_data[parent_idx] = heap_data[child_idx];
        heap_indices[parent_idx] = heap_indices[child_idx];
        parent_idx = child_idx;
    }

    heap_data[parent_idx] = parent_val;
    heap_indices[parent_idx] = parent_id;
}

inline BestCand reduce_min(BestCand a, BestCand b) {
    if (a.distance < b.distance) return a;
    if (b.distance < a.distance) return b;
    if (a.index < b.index) return a;
    return b;
}

inline BestCand warp_reduce_min(BestCand v) {
    for (int off = 16; off > 0; off >>= 1) {
        float d = simd_shuffle_down(v.distance, off);
        uint i = simd_shuffle_down(v.index, off);
        uint p = simd_shuffle_down(v.pos, off);
        BestCand w = {d, i, p};
        v = reduce_min(v, w);
    }
    return v;
}

inline BestCand parallel_min_reduce(threadgroup BestCand* scratch, BestCand local, uint tid, uint tgs) {
    // Compute SIMD indices from thread ID (Apple GPUs use 32-wide SIMD groups)
    uint lane_id = tid & 31;  // tid % 32
    uint warp_id = tid >> 5;  // tid / 32

    BestCand warp_min = warp_reduce_min(local);
    if (lane_id == 0) {
        scratch[warp_id] = warp_min;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    BestCand res = {INFINITY, SENTINEL_INDEX, 0};
    uint num_warps = (tgs + 31) >> 5;
    if (warp_id == 0) {
        if (tid < num_warps) {
            res = scratch[tid];
        }
        res = warp_reduce_min(res);
    }
    return res;
}

// MARK: - Kernel 1: Fused L2 Distance + Top-K

kernel void fused_l2_topk(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    device uint* result_indices [[buffer(2)]],
    device float* result_distances [[buffer(3)]],
    constant uint& Q [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint q_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    if (q_id >= Q) return;
    if (D > MAX_D || tgs > MAX_TGS || K == 0) return;

    threadgroup float query_cached[MAX_D];
    threadgroup Candidate shared_candidates[MAX_SHARED_CANDIDATES_POT];

    // Initialize private heap
    Candidate private_heap[K_PRIVATE];
    #pragma unroll
    for (uint i = 0; i < K_PRIVATE; ++i) {
        private_heap[i] = {INFINITY, SENTINEL_INDEX};
    }

    // Cache query in shared memory
    device const float* query_ptr = queries + (ulong)q_id * D;
    for (uint d = tid; d < D; d += tgs) {
        query_cached[d] = query_ptr[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process dataset and maintain private heap
    for (uint n_idx = tid; n_idx < N; n_idx += tgs) {
        const device float* vector_ptr = dataset + (ulong)n_idx * D;
        float dist = calculate_l2_squared(query_cached, vector_ptr, D);
        update_private_heap_sorted(private_heap, dist, n_idx);
    }

    // Merge private heaps in shared memory
    const uint num_valid_candidates = tgs * K_PRIVATE;
    uint pow2_size = 1;
    while (pow2_size < num_valid_candidates) { pow2_size <<= 1; }
    pow2_size = metal::min(pow2_size, (uint)MAX_SHARED_CANDIDATES_POT);

    // Copy private heap to shared memory
    for (uint k = 0; k < K_PRIVATE; ++k) {
        uint shared_idx = k * tgs + tid;
        if (shared_idx < pow2_size) {
            shared_candidates[shared_idx] = private_heap[k];
        }
    }
    
    // Pad with sentinels
    for (uint i = num_valid_candidates + tid; i < pow2_size; i += tgs) {
        shared_candidates[i] = {INFINITY, SENTINEL_INDEX};
    }

    const ulong output_offset = (ulong)q_id * K;
    const uint K_emit = (K <= K_PRIVATE) ? K : K_PRIVATE;

    if (K_emit <= 32) {
        // Use parallel reduction for small K
        threadgroup BestCand scratch[MAX_TGS];
        
        for (uint sel = 0; sel < K_emit; ++sel) {
            BestCand local = {INFINITY, SENTINEL_INDEX, 0};
            for (uint i = tid; i < pow2_size; i += tgs) {
                Candidate c = shared_candidates[i];
                BestCand cur = {c.distance, c.index, i};
                local = reduce_min(local, cur);
            }
            
            BestCand winner = parallel_min_reduce(scratch, local, tid, tgs);
            if (tid == 0) {
                if (winner.index != SENTINEL_INDEX) {
                    result_indices[output_offset + sel] = winner.index;
                    if (result_distances != nullptr) {
                        result_distances[output_offset + sel] = winner.distance;
                    }
                    shared_candidates[winner.pos].distance = INFINITY;
                } else {
                    result_indices[output_offset + sel] = SENTINEL_INDEX;
                    if (result_distances != nullptr) {
                        result_distances[output_offset + sel] = INFINITY;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Pad remaining outputs
        for (uint k = K_emit + tid; k < K; k += tgs) {
            result_indices[output_offset + k] = SENTINEL_INDEX;
            if (result_distances != nullptr) {
                result_distances[output_offset + k] = INFINITY;
            }
        }
    } else {
        // Use bitonic sort for larger K
        block_bitonic_sort(shared_candidates, pow2_size, tid, tgs);
        
        for (uint k = tid; k < K_emit; k += tgs) {
            if (k < pow2_size) {
                Candidate result = shared_candidates[k];
                result_indices[output_offset + k] = result.index;
                if (result_distances != nullptr) {
                    result_distances[output_offset + k] = result.distance;
                }
            } else {
                result_indices[output_offset + k] = SENTINEL_INDEX;
                if (result_distances != nullptr) {
                    result_distances[output_offset + k] = INFINITY;
                }
            }
        }
        
        // Pad remaining outputs
        for (uint k = K_emit + tid; k < K; k += tgs) {
            result_indices[output_offset + k] = SENTINEL_INDEX;
            if (result_distances != nullptr) {
                result_distances[output_offset + k] = INFINITY;
            }
        }
    }
}

// MARK: - Kernel 2: Streaming Top-K Management

// Max-heap helper for streaming
template<typename T_Dist, typename T_Idx>
METAL_FUNC void sink_down(thread T_Dist* heap_data, thread T_Idx* heap_indices, uint K, uint parent_idx) {
    T_Dist parent_val = heap_data[parent_idx];
    T_Idx parent_id = heap_indices[parent_idx];

    while (true) {
        uint child_idx = 2 * parent_idx + 1;
        if (child_idx >= K) break;

        if (child_idx + 1 < K && heap_data[child_idx] < heap_data[child_idx + 1]) {
            child_idx++;
        }

        if (parent_val >= heap_data[child_idx]) break;

        heap_data[parent_idx] = heap_data[child_idx];
        heap_indices[parent_idx] = heap_indices[child_idx];
        parent_idx = child_idx;
    }

    heap_data[parent_idx] = parent_val;
    heap_indices[parent_idx] = parent_id;
}

struct PrivateMaxHeap {
    float distances[MAX_K_PRIVATE];
    uint indices[MAX_K_PRIVATE];

    void load(device const float* global_dist, device const uint* global_indices, uint K) {
        for (uint i = 0; i < K; ++i) {
            distances[i] = global_dist[i];
            indices[i] = global_indices[i];
        }
    }

    void store(device float* global_dist, device uint* global_indices, uint K) {
        for (uint i = 0; i < K; ++i) {
            global_dist[i] = distances[i];
            global_indices[i] = indices[i];
        }
    }

    void insert(float distance, uint index, uint K) {
        if (distance < distances[0]) {
            distances[0] = distance;
            indices[0] = index;
            sink_down(distances, indices, K, 0);
        }
    }
};

// Initialize streaming buffers
kernel void streaming_topk_init(
    device float* running_distances [[buffer(0)]],
    device uint* running_indices [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    constant uint& num_queries [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const ulong total_elements = (ulong)num_queries * K;
    if ((ulong)tid >= total_elements) return;

    running_distances[tid] = INFINITY;
    running_indices[tid] = 0xFFFFFFFF;
}

// Process a chunk of data
kernel void streaming_topk_process_chunk(
    device const float* chunk_distances [[buffer(0)]],
    device float* running_distances [[buffer(2)]],
    device uint* running_indices [[buffer(3)]],
    constant StreamConfig& config [[buffer(5)]],
    uint tid_in_grid [[thread_position_in_grid]]
) {
    const uint q_idx = tid_in_grid;
    if (q_idx >= config.Q) return;

    const uint K = config.k_value;
    const uint CHUNK_SIZE = config.chunk_size;
    const ulong BASE_INDEX = config.chunk_base_index;

    if (K == 0 || K > MAX_K_PRIVATE || CHUNK_SIZE == 0) return;

    PrivateMaxHeap heap;
    
    ulong global_k_offset = (ulong)q_idx * K;
    device float* q_running_dist = running_distances + global_k_offset;
    device uint* q_running_indices = running_indices + global_k_offset;

    heap.load(q_running_dist, q_running_indices, K);

    ulong global_chunk_offset = (ulong)q_idx * CHUNK_SIZE;
    device const float* q_chunk_distances = chunk_distances + global_chunk_offset;

    for (uint i = 0; i < CHUNK_SIZE; ++i) {
        float distance = q_chunk_distances[i];
        ulong global_index_long = BASE_INDEX + i;
        uint global_index = (uint)global_index_long;
        heap.insert(distance, global_index, K);
    }

    heap.store(q_running_dist, q_running_indices, K);
}

// Sort final results
kernel void streaming_topk_finalize(
    device float* distances [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& K_const [[buffer(2)]],
    constant uint& Q [[buffer(3)]],
    uint tid_in_grid [[thread_position_in_grid]]
) {
    const uint q_idx = tid_in_grid;
    const uint K = K_const;

    if (q_idx >= Q) return;
    if (K == 0 || K > MAX_K_PRIVATE) return;

    PrivateMaxHeap heap;
    ulong global_offset = (ulong)q_idx * K;
    device float* q_dist = distances + global_offset;
    device uint* q_indices = indices + global_offset;

    heap.load(q_dist, q_indices, K);

    // In-place heapsort
    uint current_size = K;
    while (current_size > 1) {
        current_size--;
        
        float root_dist = heap.distances[0];
        uint root_idx = heap.indices[0];
        
        heap.distances[0] = heap.distances[current_size];
        heap.indices[0] = heap.indices[current_size];
        
        heap.distances[current_size] = root_dist;
        heap.indices[current_size] = root_idx;

        sink_down(heap.distances, heap.indices, current_size, 0);
    }

    heap.store(q_dist, q_indices, K);
}

// MARK: - Kernel 3: Warp-Optimized Selection for Small K

template <bool Ascending>
struct Property {
    inline bool is_better(thread const IndexDistance& a, thread const IndexDistance& b) const {
        if (Ascending) {
            return (a.distance < b.distance) || (a.distance == b.distance && a.index < b.index);
        } else {
            return (a.distance > b.distance) || (a.distance == b.distance && a.index < b.index);
        }
    }

    float init_dist() const {
        return Ascending ? INFINITY : -INFINITY;
    }
};

template <bool Ascending, uint MAX_K>
void insert_local_topk(thread IndexDistance (&local_k)[MAX_K], uint K, thread const IndexDistance& candidate, Property<Ascending> prop) {
    if (prop.is_better(candidate, local_k[K-1])) {
        local_k[K-1] = candidate;

        for (int i = K - 2; i >= 0; --i) {
            if (prop.is_better(local_k[i+1], local_k[i])) {
                // Manual swap (Metal doesn't have swap)
                IndexDistance temp = local_k[i];
                local_k[i] = local_k[i+1];
                local_k[i+1] = temp;
            } else {
                break;
            }
        }
    }
}

template <bool Ascending>
void k4_merge_registers(thread IndexDistance (&list_A)[K4_MAX_K], const thread IndexDistance (&list_B)[K4_MAX_K], uint K, Property<Ascending> prop) {
    IndexDistance merged[K4_MAX_K];
    uint ptr_A = 0;
    uint ptr_B = 0;

    for (uint i = 0; i < K; ++i) {
        if (prop.is_better(list_A[ptr_A], list_B[ptr_B])) {
            merged[i] = list_A[ptr_A++];
        } else {
            merged[i] = list_B[ptr_B++];
        }
    }
    
    for (uint i = 0; i < K; ++i) {
        list_A[i] = merged[i];
    }
}

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
    if (k_param > K4_MAX_K) return;

    const uint K = metal::min(k_param, candidateCount);
    if (K == 0) return;

    Property<true> prop;
    
    IndexDistance local_k[K4_MAX_K];
    const float boundary_dist = prop.init_dist();
    
    for (uint i = 0; i < K; ++i) {
        local_k[i] = IndexDistance{SENTINEL_INDEX, boundary_dist};
    }

    const device float* query_distances = distances + (ulong)query_idx * candidateCount;
    
    for (uint i = lane_id; i < candidateCount; i += K4_WARP_SIZE) {
        float dist = query_distances[i];
        IndexDistance candidate = IndexDistance{i, dist};
        insert_local_topk<true, K4_MAX_K>(local_k, K, candidate, prop);
    }

    IndexDistance partner_k[K4_MAX_K];
    
    for (uint stride = 1; stride < K4_WARP_SIZE; stride *= 2) {
        for (uint i = 0; i < K; ++i) {
            partner_k[i].index = simd_shuffle_xor(local_k[i].index, stride);
            partner_k[i].distance = simd_shuffle_xor(local_k[i].distance, stride);
        }
        k4_merge_registers(local_k, partner_k, K, prop);
    }

    const ulong output_offset = (ulong)query_idx * k_param;
    
    if (lane_id < K) {
        IndexDistance result = local_k[lane_id];
        if (result.index != SENTINEL_INDEX) {
            indices[output_offset + lane_id] = result.index;
            if (values != nullptr) {
                values[output_offset + lane_id] = result.distance;
            }
        }
    }
}

kernel void warp_select_small_k_descending(
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
    if (k_param > K4_MAX_K) return;

    const uint K = metal::min(k_param, candidateCount);
    if (K == 0) return;

    Property<false> prop;

    IndexDistance local_k[K4_MAX_K];
    const float boundary_dist = prop.init_dist();

    for (uint i = 0; i < K; ++i) {
        local_k[i] = IndexDistance{SENTINEL_INDEX, boundary_dist};
    }

    const device float* query_distances = distances + (ulong)query_idx * candidateCount;

    for (uint i = lane_id; i < candidateCount; i += K4_WARP_SIZE) {
        float dist = query_distances[i];
        IndexDistance candidate = IndexDistance{i, dist};
        insert_local_topk<false, K4_MAX_K>(local_k, K, candidate, prop);
    }

    IndexDistance partner_k[K4_MAX_K];

    for (uint stride = 1; stride < K4_WARP_SIZE; stride *= 2) {
        for (uint i = 0; i < K; ++i) {
            partner_k[i].index = simd_shuffle_xor(local_k[i].index, stride);
            partner_k[i].distance = simd_shuffle_xor(local_k[i].distance, stride);
        }
        k4_merge_registers(local_k, partner_k, K, prop);
    }

    const ulong output_offset = (ulong)query_idx * k_param;

    if (lane_id < K) {
        IndexDistance result = local_k[lane_id];
        if (result.index != SENTINEL_INDEX) {
            indices[output_offset + lane_id] = result.index;
            if (values != nullptr) {
                values[output_offset + lane_id] = result.distance;
            }
        }
    }
}

// MARK: - Kernel 4: Streaming L2 Top-K Update (Fused Distance + Update)
//
// ⚠️ EXPERIMENTAL - KNOWN CORRECTNESS ISSUE ⚠️
//
// This kernel has a fundamental correctness bug: each thread maintains its own
// private heap, but only thread 0 writes back results. This means results from
// other threads are DISCARDED, leading to incorrect top-k selection.
//
// The kernel only processes chunk elements that happen to map to thread 0's
// loop iterations. For production use, threads would need to cooperatively
// merge their heaps using threadgroup reduction.
//
// Recommendation: Use the chunked two-pass fallback path in FusedL2TopKKernel
// instead, which correctly handles large datasets via GPU merge.
//
// See: QUALITY_IMPROVEMENT_ROADMAP.md P0.9 for details.
//

/// Parameters for streaming L2 top-k update
struct StreamingL2Params {
    uint Q;            // Number of queries
    uint chunk_size;   // Number of vectors in this chunk
    uint D;            // Dimension of vectors
    uint K;            // Number of top-k results to maintain
    uint offset;       // Global index offset for this chunk
};

/// Update running top-k results with a new chunk of dataset vectors
/// Computes L2 distances and updates max-heap based top-k in-place
kernel void streaming_l2_topk_update(
    device const float* queries [[buffer(0)]],          // [Q × D]
    device const float* chunk [[buffer(1)]],            // [chunk_size × D]
    device uint* running_indices [[buffer(2)]],         // [Q × K] - running top-k indices
    device float* running_distances [[buffer(3)]],      // [Q × K] - running top-k distances
    constant StreamingL2Params& params [[buffer(4)]],
    uint q_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    if (q_id >= params.Q) return;

    const uint D = params.D;
    const uint K = params.K;
    const uint chunk_size = params.chunk_size;
    const uint global_offset = params.offset;

    // Bounds check
    if (K == 0 || chunk_size == 0 || K > MAX_K_PRIVATE) return;

    // Load query into thread-local registers (limited dimension)
    const device float* query_ptr = queries + (ulong)q_id * D;

    // Load current top-k into private memory (max heap: largest distance at root)
    ulong topk_offset = (ulong)q_id * K;
    float heap_distances[MAX_K_PRIVATE];
    uint heap_indices[MAX_K_PRIVATE];

    for (uint i = 0; i < K; ++i) {
        heap_distances[i] = running_distances[topk_offset + i];
        heap_indices[i] = running_indices[topk_offset + i];
    }

    // Process chunk vectors in parallel across threads
    for (uint n = tid; n < chunk_size; n += tgs) {
        const device float* vec_ptr = chunk + (ulong)n * D;

        // Compute L2 squared distance
        float dist_sq = 0.0f;
        uint d = 0;
        for (; d + 3 < D; d += 4) {
            float4 q_data = float4(query_ptr[d], query_ptr[d+1], query_ptr[d+2], query_ptr[d+3]);
            float4 v_data = float4(vec_ptr[d], vec_ptr[d+1], vec_ptr[d+2], vec_ptr[d+3]);
            float4 diff = q_data - v_data;
            dist_sq += dot(diff, diff);
        }
        for (; d < D; ++d) {
            float diff = query_ptr[d] - vec_ptr[d];
            dist_sq = fma(diff, diff, dist_sq);
        }

        uint global_index = global_offset + n;

        // Update max-heap if this distance is better than the worst in heap
        if (dist_sq < heap_distances[0]) {
            // Replace root with new candidate
            heap_distances[0] = dist_sq;
            heap_indices[0] = global_index;

            // Sink down to restore heap property
            uint parent = 0;
            while (true) {
                uint left = 2 * parent + 1;
                uint right = 2 * parent + 2;
                uint largest = parent;

                if (left < K && heap_distances[left] > heap_distances[largest]) {
                    largest = left;
                }
                if (right < K && heap_distances[right] > heap_distances[largest]) {
                    largest = right;
                }
                if (largest == parent) break;

                // Swap
                float tmp_d = heap_distances[parent];
                uint tmp_i = heap_indices[parent];
                heap_distances[parent] = heap_distances[largest];
                heap_indices[parent] = heap_indices[largest];
                heap_distances[largest] = tmp_d;
                heap_indices[largest] = tmp_i;
                parent = largest;
            }
        }
    }

    // Note: This is a simplified implementation where each thread maintains its own heap
    // For production, threads would need to cooperatively merge their heaps
    // Since the Swift code dispatches one threadgroup per query with many threads,
    // we only write back from thread 0 (first thread processes first chunk elements)
    // A more sophisticated implementation would use threadgroup reduction

    // Write back updated heap (only thread 0 to avoid race conditions)
    // In a production implementation, you'd want threadgroup cooperation
    if (tid == 0) {
        for (uint i = 0; i < K; ++i) {
            running_distances[topk_offset + i] = heap_distances[i];
            running_indices[topk_offset + i] = heap_indices[i];
        }
    }
}

// MARK: - Kernel 5: Batch Selection for General K

/// Batch select k smallest values (ascending order)
/// Used for k values > 32 where warp optimization doesn't apply
kernel void batch_select_k_nearest_ascending(
    device const float* distances [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& batchSize [[buffer(3)]],
    constant uint& queryCount [[buffer(4)]],
    constant uint& candidateCount [[buffer(5)]],
    constant uint& k_param [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint query_idx = gid.y;

    if (batch_idx >= batchSize || query_idx >= queryCount) return;
    if (k_param == 0 || k_param > MAX_K_PRIVATE) return;

    const uint K = metal::min(k_param, candidateCount);

    // Calculate input/output offsets
    const ulong input_offset = ((ulong)batch_idx * queryCount + query_idx) * candidateCount;
    const ulong output_offset = ((ulong)batch_idx * queryCount + query_idx) * k_param;
    const device float* query_distances = distances + input_offset;

    // Use max-heap to track K smallest (root = largest of the K smallest)
    float heap_dist[MAX_K_PRIVATE];
    uint heap_idx[MAX_K_PRIVATE];

    // Initialize with first K elements (thread 0 only for simplicity)
    if (tid == 0) {
        for (uint i = 0; i < K; ++i) {
            if (i < candidateCount) {
                heap_dist[i] = query_distances[i];
                heap_idx[i] = i;
            } else {
                heap_dist[i] = INFINITY;
                heap_idx[i] = SENTINEL_INDEX;
            }
        }

        // Heapify (build max heap)
        for (int i = (int)K / 2 - 1; i >= 0; --i) {
            sink_down(heap_dist, heap_idx, K, (uint)i);
        }

        // Process remaining candidates
        for (uint i = K; i < candidateCount; ++i) {
            float dist = query_distances[i];
            if (dist < heap_dist[0]) {
                heap_dist[0] = dist;
                heap_idx[0] = i;
                sink_down(heap_dist, heap_idx, K, 0);
            }
        }

        // Sort the heap (convert to ascending order)
        uint current_size = K;
        while (current_size > 1) {
            current_size--;
            float tmp_d = heap_dist[0];
            uint tmp_i = heap_idx[0];
            heap_dist[0] = heap_dist[current_size];
            heap_idx[0] = heap_idx[current_size];
            heap_dist[current_size] = tmp_d;
            heap_idx[current_size] = tmp_i;
            sink_down(heap_dist, heap_idx, current_size, 0);
        }

        // Write output
        for (uint i = 0; i < k_param; ++i) {
            if (i < K) {
                indices[output_offset + i] = heap_idx[i];
                if (values != nullptr) {
                    values[output_offset + i] = heap_dist[i];
                }
            } else {
                indices[output_offset + i] = SENTINEL_INDEX;
                if (values != nullptr) {
                    values[output_offset + i] = INFINITY;
                }
            }
        }
    }
}

/// Batch select k largest values (descending order)
kernel void batch_select_k_nearest_descending(
    device const float* distances [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& batchSize [[buffer(3)]],
    constant uint& queryCount [[buffer(4)]],
    constant uint& candidateCount [[buffer(5)]],
    constant uint& k_param [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint query_idx = gid.y;

    if (batch_idx >= batchSize || query_idx >= queryCount) return;
    if (k_param == 0 || k_param > MAX_K_PRIVATE) return;

    const uint K = metal::min(k_param, candidateCount);

    // Calculate input/output offsets
    const ulong input_offset = ((ulong)batch_idx * queryCount + query_idx) * candidateCount;
    const ulong output_offset = ((ulong)batch_idx * queryCount + query_idx) * k_param;
    const device float* query_distances = distances + input_offset;

    // Use min-heap to track K largest (root = smallest of the K largest)
    float heap_dist[MAX_K_PRIVATE];
    uint heap_idx[MAX_K_PRIVATE];

    // Initialize with first K elements (thread 0 only for simplicity)
    if (tid == 0) {
        for (uint i = 0; i < K; ++i) {
            if (i < candidateCount) {
                heap_dist[i] = query_distances[i];
                heap_idx[i] = i;
            } else {
                heap_dist[i] = -INFINITY;
                heap_idx[i] = SENTINEL_INDEX;
            }
        }

        // Heapify (build min heap)
        for (int i = (int)K / 2 - 1; i >= 0; --i) {
            min_heap_sink_down(heap_dist, heap_idx, K, (uint)i);
        }

        // Process remaining candidates
        for (uint i = K; i < candidateCount; ++i) {
            float dist = query_distances[i];
            if (dist > heap_dist[0]) {  // Larger than smallest in heap
                heap_dist[0] = dist;
                heap_idx[0] = i;
                min_heap_sink_down(heap_dist, heap_idx, K, 0);
            }
        }

        // Sort the heap (convert to descending order using max extraction from min-heap)
        // Actually we need to sort descending, so extract min repeatedly and reverse
        float sorted_dist[MAX_K_PRIVATE];
        uint sorted_idx[MAX_K_PRIVATE];

        for (uint i = 0; i < K; ++i) {
            sorted_dist[i] = heap_dist[i];
            sorted_idx[i] = heap_idx[i];
        }

        // Simple insertion sort for descending order (K is small enough)
        for (uint i = 1; i < K; ++i) {
            float key_d = sorted_dist[i];
            uint key_i = sorted_idx[i];
            int j = (int)i - 1;
            while (j >= 0 && sorted_dist[j] < key_d) {  // Descending order
                sorted_dist[j + 1] = sorted_dist[j];
                sorted_idx[j + 1] = sorted_idx[j];
                j--;
            }
            sorted_dist[j + 1] = key_d;
            sorted_idx[j + 1] = key_i;
        }

        // Write output
        for (uint i = 0; i < k_param; ++i) {
            if (i < K) {
                indices[output_offset + i] = sorted_idx[i];
                if (values != nullptr) {
                    values[output_offset + i] = sorted_dist[i];
                }
            } else {
                indices[output_offset + i] = SENTINEL_INDEX;
                if (values != nullptr) {
                    values[output_offset + i] = -INFINITY;
                }
            }
        }
    }
}