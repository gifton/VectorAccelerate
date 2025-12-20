// VectorAccelerate: IVF GPU Candidate Search Kernel
//
// Performs IVF search directly on GPU without CPU-side candidate gathering.
//
// Kernel: ivf_list_search
// - One threadgroup per query
// - Iterates over selected inverted lists (CSR ordered) directly in `listVectors`
// - Computes L2 (squared) distances and returns top-K per query
//
// NOTE: This shader is self-contained with IVF-prefixed types to avoid conflicts
// when combined with other shaders during runtime compilation.

#include <metal_stdlib>
using namespace metal;

// MARK: - IVF Configuration Constants

constexpr constant uint IVF_K_PRIVATE = 8;                    // Per-thread register heap size
constexpr constant uint IVF_MAX_TGS = 256;                    // Maximum threadgroup size
constexpr constant uint IVF_MAX_D_CACHED = 2048;              // Max dimension for query caching
constexpr constant uint IVF_MAX_SHARED_CANDIDATES_POT = 2048; // Max candidates in shared memory
constexpr constant uint IVF_SENTINEL_INDEX = 0xFFFFFFFF;      // Invalid index marker

// MARK: - IVF Data Structures

/// Must match `IVFListSearchParameters` in Swift.
struct IVFSearchParams {
    uint numQueries;
    uint numCentroids;
    uint dimension;
    uint nprobe;
    uint k;
    uint maxCandidatesPerQuery;
};

/// Candidate for top-K selection
struct IVFCandidate {
    float distance;
    uint index;
};

/// Best candidate with position tracking for parallel reduction
struct IVFBestCand {
    float distance;
    uint index;
    uint pos;
};

// MARK: - Helper Functions

inline bool ivf_is_better(IVFCandidate a, IVFCandidate b) {
    if (a.distance < b.distance) return true;
    if (a.distance > b.distance) return false;
    return a.index < b.index;
}

inline float4 ivf_safe_load_float4(device const float* base, uint offset, uint max) {
    if (offset + 3 < max) {
        return reinterpret_cast<const device float4*>(base + offset)[0];
    }
    float4 v = float4(0.0f);
    for (uint i = 0; i < 4 && offset + i < max; ++i) {
        v[i] = base[offset + i];
    }
    return v;
}

inline float ivf_calculate_l2_squared_cached(
    const threadgroup float* query_cached,
    device const float* vector_ptr,
    uint D
) {
    float accumulator = 0.0f;
    uint d = 0;

    // Vectorized processing
    for (; d + 3 < D; d += 4) {
        float4 q_data = reinterpret_cast<const threadgroup float4*>(query_cached + d)[0];
        float4 v_data = ivf_safe_load_float4(vector_ptr, d, D);
        float4 diff = q_data - v_data;
        accumulator += dot(diff, diff);
    }

    // Remainder
    for (; d < D; ++d) {
        float diff = query_cached[d] - vector_ptr[d];
        accumulator = fma(diff, diff, accumulator);
    }
    return accumulator;
}

inline float ivf_calculate_l2_squared_global(
    device const float* query_ptr,
    device const float* vector_ptr,
    uint D
) {
    float accumulator = 0.0f;
    uint d = 0;

    for (; d + 3 < D; d += 4) {
        float4 q_data = ivf_safe_load_float4(query_ptr, d, D);
        float4 v_data = ivf_safe_load_float4(vector_ptr, d, D);
        float4 diff = q_data - v_data;
        accumulator += dot(diff, diff);
    }

    for (; d < D; ++d) {
        float diff = query_ptr[d] - vector_ptr[d];
        accumulator = fma(diff, diff, accumulator);
    }

    return accumulator;
}

inline void ivf_update_private_heap_sorted(thread IVFCandidate* heap, float new_dist, uint new_id) {
    IVFCandidate worst = heap[IVF_K_PRIVATE - 1];
    if (new_dist < worst.distance || (new_dist == worst.distance && new_id < worst.index)) {
        heap[IVF_K_PRIVATE - 1] = {new_dist, new_id};

        // Insertion sort to maintain order
        #pragma unroll
        for (uint i = IVF_K_PRIVATE - 1; i > 0; --i) {
            if (ivf_is_better(heap[i], heap[i-1])) {
                IVFCandidate tmp = heap[i];
                heap[i] = heap[i-1];
                heap[i-1] = tmp;
            } else {
                break;
            }
        }
    }
}

// Bitonic sort for shared memory candidates
inline void ivf_block_bitonic_sort(threadgroup IVFCandidate* data, const uint N_PoT, const uint tid, const uint tgs) {
    for (uint k = 2; k <= N_PoT; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < N_PoT; i += tgs) {
                uint partner = i ^ j;
                if (i < partner) {
                    bool direction_ascending = ((i & k) == 0);
                    IVFCandidate c_i = data[i];
                    IVFCandidate c_p = data[partner];
                    bool p_is_better = ivf_is_better(c_p, c_i);
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

inline IVFBestCand ivf_reduce_min(IVFBestCand a, IVFBestCand b) {
    if (a.distance < b.distance) return a;
    if (b.distance < a.distance) return b;
    if (a.index < b.index) return a;
    return b;
}

inline IVFBestCand ivf_warp_reduce_min(IVFBestCand v) {
    for (int off = 16; off > 0; off >>= 1) {
        float d = simd_shuffle_down(v.distance, off);
        uint i = simd_shuffle_down(v.index, off);
        uint p = simd_shuffle_down(v.pos, off);
        IVFBestCand w = {d, i, p};
        v = ivf_reduce_min(v, w);
    }
    return v;
}

inline IVFBestCand ivf_parallel_min_reduce(threadgroup IVFBestCand* scratch, IVFBestCand local, uint tid, uint tgs) {
    uint lane_id = tid & 31;  // tid % 32
    uint warp_id = tid >> 5;  // tid / 32

    IVFBestCand warp_min = ivf_warp_reduce_min(local);
    if (lane_id == 0) {
        scratch[warp_id] = warp_min;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    IVFBestCand res = {INFINITY, IVF_SENTINEL_INDEX, 0};
    uint num_warps = (tgs + 31) >> 5;
    if (warp_id == 0) {
        if (tid < num_warps) {
            res = scratch[tid];
        }
        res = ivf_warp_reduce_min(res);
    }
    return res;
}

// MARK: - Kernel

kernel void ivf_list_search(
    device const float* queries [[buffer(0)]],              // [Q × D]
    device const float* listVectors [[buffer(1)]],          // [N × D] CSR-ordered
    device const uint* listOffsets [[buffer(2)]],           // [nlist + 1]
    device const uint* vectorIndices [[buffer(3)]],         // [N] CSR position -> original index
    device const uint* selectedLists [[buffer(4)]],         // [Q × nprobe]
    device uint* outIndices [[buffer(5)]],                  // [Q × K] original indices
    device float* outDistances [[buffer(6)]],               // [Q × K] squared L2
    constant IVFSearchParams& params [[buffer(7)]],
    uint q_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    if (q_id >= params.numQueries) return;
    if (tgs > IVF_MAX_TGS) return;

    const uint D = params.dimension;
    const uint K = params.k;

    // Always write valid output for this query even if inputs are degenerate.
    const ulong outBase = (ulong)q_id * K;
    if (K == 0 || D == 0) {
        for (uint i = tid; i < K; i += tgs) {
            outIndices[outBase + i] = IVF_SENTINEL_INDEX;
            outDistances[outBase + i] = INFINITY;
        }
        return;
    }

    device const float* query_ptr = queries + (ulong)q_id * D;

    // Query cache (threadgroup) when dimension fits.
    const bool cacheQuery = (D <= IVF_MAX_D_CACHED);
    threadgroup float query_cached[IVF_MAX_D_CACHED];
    if (cacheQuery) {
        for (uint d = tid; d < D; d += tgs) {
            query_cached[d] = query_ptr[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread heap (sorted ascending by distance).
    IVFCandidate private_heap[IVF_K_PRIVATE];
    #pragma unroll
    for (uint i = 0; i < IVF_K_PRIVATE; ++i) {
        private_heap[i] = {INFINITY, IVF_SENTINEL_INDEX};
    }

    // Scan selected lists for this query.
    const ulong selectedBase = (ulong)q_id * params.nprobe;
    for (uint p = 0; p < params.nprobe; ++p) {
        uint clusterId = selectedLists[selectedBase + p];
        if (clusterId == IVF_SENTINEL_INDEX) continue;
        if (clusterId >= params.numCentroids) continue;

        uint listStart = listOffsets[clusterId];
        uint listEnd = listOffsets[clusterId + 1];
        if (listEnd <= listStart) continue;

        for (uint i = listStart + tid; i < listEnd; i += tgs) {
            device const float* vec_ptr = listVectors + (ulong)i * D;
            float dist = cacheQuery
                ? ivf_calculate_l2_squared_cached(query_cached, vec_ptr, D)
                : ivf_calculate_l2_squared_global(query_ptr, vec_ptr, D);
            uint origIdx = vectorIndices[i];
            ivf_update_private_heap_sorted(private_heap, dist, origIdx);
        }
    }

    // Merge per-thread heaps into shared candidates.
    threadgroup IVFCandidate shared_candidates[IVF_MAX_SHARED_CANDIDATES_POT];

    const uint num_valid_candidates = tgs * IVF_K_PRIVATE;
    uint pow2_size = 1;
    while (pow2_size < num_valid_candidates) pow2_size <<= 1;
    pow2_size = min(pow2_size, IVF_MAX_SHARED_CANDIDATES_POT);

    // Layout: (k, tid) -> k * tgs + tid
    #pragma unroll
    for (uint kidx = 0; kidx < IVF_K_PRIVATE; ++kidx) {
        uint shared_idx = kidx * tgs + tid;
        if (shared_idx < pow2_size) {
            shared_candidates[shared_idx] = private_heap[kidx];
        }
    }

    // Pad to pow2 with sentinels.
    for (uint i = num_valid_candidates + tid; i < pow2_size; i += tgs) {
        shared_candidates[i] = {INFINITY, IVF_SENTINEL_INDEX};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Selection / sort.
    if (K <= 32) {
        threadgroup IVFBestCand scratch[IVF_MAX_TGS];

        if (tid == 0) {
            for (uint i = 0; i < K; ++i) {
                outIndices[outBase + i] = IVF_SENTINEL_INDEX;
                outDistances[outBase + i] = INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint out = 0; out < K; ++out) {
            IVFBestCand local = {INFINITY, IVF_SENTINEL_INDEX, 0};
            for (uint idx = tid; idx < pow2_size; idx += tgs) {
                IVFCandidate c = shared_candidates[idx];
                IVFBestCand bc = {c.distance, c.index, idx};
                local = ivf_reduce_min(local, bc);
            }
            IVFBestCand winner = ivf_parallel_min_reduce(scratch, local, tid, tgs);

            if (tid == 0) {
                if (winner.index != IVF_SENTINEL_INDEX) {
                    outIndices[outBase + out] = winner.index;
                    outDistances[outBase + out] = winner.distance;
                    shared_candidates[winner.pos].distance = INFINITY;
                    shared_candidates[winner.pos].index = IVF_SENTINEL_INDEX;
                } else {
                    outIndices[outBase + out] = IVF_SENTINEL_INDEX;
                    outDistances[outBase + out] = INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        ivf_block_bitonic_sort(shared_candidates, pow2_size, tid, tgs);

        for (uint out = tid; out < K; out += tgs) {
            if (out < pow2_size) {
                IVFCandidate c = shared_candidates[out];
                outIndices[outBase + out] = c.index;
                outDistances[outBase + out] = c.distance;
            } else {
                outIndices[outBase + out] = IVF_SENTINEL_INDEX;
                outDistances[outBase + out] = INFINITY;
            }
        }
    }
}
