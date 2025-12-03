//
//  HNSWShaders.metal
//  VectorIndexAcceleration
//
//  Metal 4 GPU kernels for HNSW index operations.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// =============================================================================
// Configuration Constants
// =============================================================================

constant uint MAX_DIM = 2048;
constant uint TILE_SIZE = 16;
constant uint K42_TGS = 256;  // Fixed for Blelloch scan
constexpr constant uint K41_SENTINEL_INDEX = 0xFFFFFFFF;

// PCG RNG constants
constant uint64_t PCG_MULTIPLIER = 6364136223846793005ull;
constant uint64_t PCG_INCREMENT = 1442695040888963407ull;

// =============================================================================
// Helper Structures
// =============================================================================

// Metric Types for template specialization
struct MetricL2 {};
struct MetricIP {};

// PCG RNG state
struct PCGState {
    uint64_t state;
};

// =============================================================================
// Helper Functions
// =============================================================================

// PCG RNG initialization
PCGState pcg_init(uint64_t seed, uint64_t sequence) {
    PCGState rng;
    rng.state = seed + sequence;
    rng.state = rng.state * PCG_MULTIPLIER + PCG_INCREMENT;
    return rng;
}

// PCG random uint32
uint pcg_random_uint(thread PCGState* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * PCG_MULTIPLIER + PCG_INCREMENT;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// PCG random float (0, 1]
float pcg_random_float(thread PCGState* rng) {
    uint r = pcg_random_uint(rng);
    return (float)(r + 1u) * 2.3283064365386963e-10f;
}

// Blelloch exclusive scan helper
void exclusive_scan_blelloch(threadgroup uint* data, uint tid) {
    int offset = 1;
    for (int d = K42_TGS >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            data[bi] += data[ai];
        }
        offset *= 2;
    }

    if (tid == 0) {
        data[K42_TGS - 1] = 0;
    }

    for (int d = 1; d < K42_TGS; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            uint t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Metric helper for distance computation
template <typename Metric>
struct MetricHelper {
    inline float boundary_value() const {
        return limits<float>::infinity();
    }

    inline float accumulate_vec4(float accumulator, float4 v1, float4 v2) const {
        if (is_same_v<Metric, MetricL2>) {
            float4 diff = v1 - v2;
            return accumulator + dot(diff, diff);
        } else {
            return accumulator + dot(v1, v2);
        }
    }

    inline float accumulate_scalar(float accumulator, float v1, float v2) const {
        if (is_same_v<Metric, MetricL2>) {
            float diff = v1 - v2;
            return metal::fma(diff, diff, accumulator);
        } else {
            return metal::fma(v1, v2, accumulator);
        }
    }

    inline float finalize(float accumulator) const {
        if (is_same_v<Metric, MetricIP>) {
            return -accumulator;
        }
        return accumulator;
    }
};

// =============================================================================
// Kernel: Distance Matrix
// =============================================================================

template <typename Metric>
void hnsw_distance_matrix_impl(
    device const float* query_vectors,
    device const float* candidate_vectors,
    device float* distance_matrix,
    device const uint* valid_candidates,
    constant uint& B,
    constant uint& N,
    constant uint& D,
    uint2 gid,
    uint2 tg_pos,
    uint2 tg_size,
    uint2 tid_in_tg
) {
    if (D > MAX_DIM) return;

    threadgroup float shared_query[TILE_SIZE][MAX_DIM];
    MetricHelper<Metric> helper;

    const uint TGS_FLAT = tg_size.x * tg_size.y;
    const uint tid_flat = tid_in_tg.y * tg_size.x + tid_in_tg.x;

    const uint total_elements = TILE_SIZE * D;
    const uint base_global_b = tg_pos.y * TILE_SIZE;

    for (uint i = tid_flat; i < total_elements; i += TGS_FLAT) {
        uint local_b = i / D;
        uint local_d = i % D;
        uint global_b = base_global_b + local_b;

        if (global_b < B) {
            ulong global_idx = (ulong)global_b * D + local_d;
            shared_query[local_b][local_d] = query_vectors[global_idx];
        } else {
            shared_query[local_b][local_d] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint global_b = gid.y;
    const uint global_n = gid.x;

    if (global_b >= B || global_n >= N) return;

    if (valid_candidates != nullptr) {
        const uint word_idx = global_n / 32;
        const uint bit_idx = global_n % 32;
        if (((valid_candidates[word_idx] >> bit_idx) & 1u) == 0) {
            ulong output_idx = (ulong)global_b * N + global_n;
            distance_matrix[output_idx] = helper.boundary_value();
            return;
        }
    }

    const uint local_b = tid_in_tg.y;
    const threadgroup float* query_ptr = shared_query[local_b];
    const device float* cand_ptr = candidate_vectors + (ulong)global_n * D;

    float accumulator = 0.0f;
    uint d = 0;

    for (; d + 3 < D; d += 4) {
        float4 q_data = reinterpret_cast<const threadgroup float4*>(query_ptr + d)[0];
        float4 c_data = reinterpret_cast<const device float4*>(cand_ptr + d)[0];
        accumulator = helper.accumulate_vec4(accumulator, q_data, c_data);
    }

    for (; d < D; ++d) {
        accumulator = helper.accumulate_scalar(accumulator, query_ptr[d], cand_ptr[d]);
    }

    ulong output_idx = (ulong)global_b * N + global_n;
    distance_matrix[output_idx] = helper.finalize(accumulator);
}

#define KERNEL_DISTANCE_MATRIX_ARGS \
    device const float* query_vectors [[buffer(0)]], \
    device const float* candidate_vectors [[buffer(1)]], \
    device float* distance_matrix [[buffer(2)]], \
    device const uint* valid_candidates [[buffer(3)]], \
    constant uint& batch_size [[buffer(4)]], \
    constant uint& num_candidates [[buffer(5)]], \
    constant uint& dimension [[buffer(6)]], \
    uint2 gid [[thread_position_in_grid]], \
    uint2 tg_pos [[threadgroup_position_in_grid]], \
    uint2 tg_size [[threads_per_threadgroup]], \
    uint2 tid_in_tg [[thread_index_in_threadgroup]]

kernel void hnsw_distance_matrix(KERNEL_DISTANCE_MATRIX_ARGS) {
    // Default to L2 distance
    hnsw_distance_matrix_impl<MetricL2>(
        query_vectors, candidate_vectors, distance_matrix, valid_candidates,
        batch_size, num_candidates, dimension,
        gid, tg_pos, tg_size, tid_in_tg
    );
}

kernel void hnsw_distance_matrix_l2(KERNEL_DISTANCE_MATRIX_ARGS) {
    hnsw_distance_matrix_impl<MetricL2>(
        query_vectors, candidate_vectors, distance_matrix, valid_candidates,
        batch_size, num_candidates, dimension,
        gid, tg_pos, tg_size, tid_in_tg
    );
}

kernel void hnsw_distance_matrix_inner_product(KERNEL_DISTANCE_MATRIX_ARGS) {
    hnsw_distance_matrix_impl<MetricIP>(
        query_vectors, candidate_vectors, distance_matrix, valid_candidates,
        batch_size, num_candidates, dimension,
        gid, tg_pos, tg_size, tid_in_tg
    );
}

// =============================================================================
// Kernel: Level Assignment
// =============================================================================

kernel void hnsw_assign_node_levels(
    device uint* node_levels [[buffer(0)]],
    device const uint* random_seeds [[buffer(1)]],
    device atomic_uint* level_counts [[buffer(2)]],
    device float* level_probabilities [[buffer(3)]],
    constant float& ml_factor [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    constant uint& max_level [[buffer(6)]],
    constant uint& global_random_seed [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]],
    uint num_tgs [[threadgroups_per_grid]]
) {
    const ulong grid_size = (ulong)tgs * num_tgs;

    for (ulong b_id = gid; b_id < B; b_id += grid_size) {
        uint64_t seed = (uint64_t)global_random_seed;
        uint64_t sequence = (random_seeds != nullptr) ? (uint64_t)random_seeds[b_id] : b_id;

        PCGState rng = pcg_init(seed, sequence);
        float rand_float = pcg_random_float(&rng);
        float level_f = metal::floor(-metal::log(rand_float) * ml_factor);

        uint level = 0;
        if (level_f >= 0.0f && level_f < (float)0xFFFFFFFFu) {
            level = (uint)level_f;
        }
        level = metal::min(level, max_level);

        node_levels[b_id] = level;

        if (level_probabilities != nullptr) {
            level_probabilities[b_id] = rand_float;
        }

        atomic_fetch_add_explicit(&level_counts[level], 1u, memory_order_relaxed);
    }
}

// =============================================================================
// Kernel: Visited Set Management
// =============================================================================

kernel void hnsw_clear_visited_flags(
    device uint* visited_flags [[buffer(0)]],
    constant uint& num_nodes [[buffer(1)]],
    constant uint& num_words [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]],
    uint num_tgs [[threadgroups_per_grid]]
) {
    const ulong num_vec4 = ((ulong)num_words + 3) / 4;
    const ulong grid_size = (ulong)tgs * num_tgs;

    device uint4* visited_flags_vec4 = reinterpret_cast<device uint4*>(visited_flags);
    const uint4 zero_vec = uint4(0);

    for (ulong i = gid; i < num_vec4; i += grid_size) {
        visited_flags_vec4[i] = zero_vec;
    }
}

kernel void hnsw_merge_visited_sets(
    device const uint* set_a [[buffer(0)]],
    device const uint* set_b [[buffer(1)]],
    device atomic_uint* merged [[buffer(2)]],
    constant uint& num_words [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]],
    uint num_tgs [[threadgroups_per_grid]]
) {
    const ulong num_vec4 = ((ulong)num_words + 3) / 4;
    const ulong grid_size = (ulong)tgs * num_tgs;

    device const uint4* set_a_vec4 = reinterpret_cast<device const uint4*>(set_a);
    device const uint4* set_b_vec4 = reinterpret_cast<device const uint4*>(set_b);

    for (ulong i = gid; i < num_vec4; i += grid_size) {
        uint4 data_a = set_a_vec4[i];
        uint4 data_b = set_b_vec4[i];
        uint4 merged_data = data_a | data_b;

        ulong base_idx = i * 4;

        atomic_fetch_or_explicit(&merged[base_idx], merged_data.x, memory_order_relaxed);

        if (base_idx + 1 < num_words) {
            atomic_fetch_or_explicit(&merged[base_idx + 1], merged_data.y, memory_order_relaxed);
        }
        if (base_idx + 2 < num_words) {
            atomic_fetch_or_explicit(&merged[base_idx + 2], merged_data.z, memory_order_relaxed);
        }
        if (base_idx + 3 < num_words) {
            atomic_fetch_or_explicit(&merged[base_idx + 3], merged_data.w, memory_order_relaxed);
        }
    }
}

// =============================================================================
// Kernel: Batch Edge Insertion
// =============================================================================

void insert_edge(
    uint src_id,
    uint dest_id,
    device uint* graph_edges,
    device atomic_uint* edge_counts,
    device const uint* edge_capacity,
    device const uint* edge_offsets,
    device atomic_uint* realloc_flags
) {
    uint reserved_slot_idx = atomic_fetch_add_explicit(&edge_counts[src_id], 1u, memory_order_relaxed);
    uint capacity = edge_capacity[src_id];

    if (reserved_slot_idx < capacity) {
        ulong base_offset = edge_offsets[src_id];
        ulong physical_idx = base_offset + reserved_slot_idx;
        graph_edges[physical_idx] = dest_id;
    } else {
        atomic_fetch_or_explicit(&realloc_flags[src_id], 1u, memory_order_relaxed);
    }
}

kernel void hnsw_batch_insert_edges(
    device uint* graph_edges [[buffer(0)]],
    device atomic_uint* edge_counts [[buffer(1)]],
    device const uint* edge_capacity [[buffer(2)]],
    device const uint* new_edges [[buffer(3)]],
    device const uint* source_nodes [[buffer(4)]],
    device const uint* edge_offsets [[buffer(5)]],
    device atomic_uint* realloc_flags [[buffer(6)]],
    constant uint& B [[buffer(7)]],
    constant uint& M [[buffer(8)]],
    constant uint& bidirectional [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint b_id = gid.y;
    const uint m_idx = gid.x;

    if (b_id >= B || m_idx >= M) return;

    const ulong flat_idx = (ulong)b_id * M + m_idx;

    uint src_id = source_nodes[b_id];
    uint dest_id = new_edges[flat_idx];

    if (src_id == dest_id || dest_id == K41_SENTINEL_INDEX || src_id == K41_SENTINEL_INDEX) return;

    insert_edge(src_id, dest_id, graph_edges, edge_counts, edge_capacity, edge_offsets, realloc_flags);

    if (bidirectional != 0) {
        insert_edge(dest_id, src_id, graph_edges, edge_counts, edge_capacity, edge_offsets, realloc_flags);
    }
}

// =============================================================================
// Kernel: Edge Pruning
// =============================================================================

kernel void hnsw_k42_pass1_count_surviving(
    device const uint* prune_flags [[buffer(3)]],
    device const uint* old_edge_offsets [[buffer(8)]],
    device uint* new_edge_counts [[buffer(2)]],
    constant uint& N [[buffer(6)]],
    uint n_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    if (n_id >= N) return;
    if (tgs > K42_TGS) return;

    ulong start = old_edge_offsets[n_id];
    ulong end = old_edge_offsets[n_id + 1];
    uint num_edges = (uint)(end - start);

    uint local_count = 0;
    for (uint i = tid; i < num_edges; i += tgs) {
        if (prune_flags[start + i] == 0) {
            local_count++;
        }
    }

    threadgroup uint shared_sum[K42_TGS];
    shared_sum[tid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < tgs; stride *= 2) {
        if (tid % (2 * stride) == 0 && (tid + stride) < tgs) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        new_edge_counts[n_id] = shared_sum[0];
    }
}

kernel void hnsw_k42_pass3_scan_and_compact(
    device const float* old_distances [[buffer(0)]],
    device const uint* old_graph_edges [[buffer(1)]],
    device const uint* prune_flags [[buffer(3)]],
    device const uint* old_edge_offsets [[buffer(8)]],
    device const uint* new_edge_offsets [[buffer(5)]],
    device uint* compacted_edges [[buffer(4)]],
    device float* compacted_distances [[buffer(9)]],
    constant uint& N [[buffer(6)]],
    uint n_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    if (n_id >= N) return;
    if (tgs != K42_TGS) return;

    threadgroup uint shared_scan[K42_TGS];

    ulong old_start = old_edge_offsets[n_id];
    ulong new_start = new_edge_offsets[n_id];
    uint num_edges = (uint)(old_edge_offsets[n_id+1] - old_start);

    if (num_edges > K42_TGS) return;

    if (tid < num_edges) {
        uint flag = prune_flags[old_start + tid];
        shared_scan[tid] = (flag == 0) ? 1u : 0u;
    } else {
        shared_scan[tid] = 0;
    }

    exclusive_scan_blelloch(shared_scan, tid);

    if (tid < num_edges) {
        if (prune_flags[old_start + tid] == 0) {
            uint local_dest_idx = shared_scan[tid];
            ulong global_dest_idx = new_start + local_dest_idx;

            uint edge_id = old_graph_edges[old_start + tid];
            compacted_edges[global_dest_idx] = edge_id;

            if (old_distances != nullptr && compacted_distances != nullptr) {
                compacted_distances[global_dest_idx] = old_distances[old_start + tid];
            }
        }
    }
}

// =============================================================================
// Kernel: Distance Cache Update
// =============================================================================

kernel void hnsw_update_distance_cache(
    device float* cache_distances [[buffer(0)]],
    device uint* cache_indices [[buffer(1)]],
    device const float* update_distances [[buffer(2)]],
    device const uint2* node_pairs [[buffer(3)]],
    device atomic_uint* counters [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]],
    uint num_tgs [[threadgroups_per_grid]]
) {
    const ulong grid_size = (ulong)tgs * num_tgs;

    for (ulong b_id = gid; b_id < B; b_id += grid_size) {
        float dist = update_distances[b_id];
        uint2 pair = node_pairs[b_id];
        uint nodeA = pair.x;
        uint nodeB = pair.y;

        // Update cache for nodeA -> nodeB
        uint slotA = atomic_fetch_add_explicit(&counters[nodeA], 1u, memory_order_relaxed) % C;
        ulong idxA = (ulong)nodeA * C + slotA;
        cache_distances[idxA] = dist;
        cache_indices[idxA] = nodeB;

        // Update cache for nodeB -> nodeA (symmetric)
        uint slotB = atomic_fetch_add_explicit(&counters[nodeB], 1u, memory_order_relaxed) % C;
        ulong idxB = (ulong)nodeB * C + slotB;
        cache_distances[idxB] = dist;
        cache_indices[idxB] = nodeA;
    }
}

// =============================================================================
// Kernel: Search Layer (Placeholder - requires visited set integration)
// =============================================================================

kernel void hnsw_search_layer_parallel(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device const uint* edges [[buffer(2)]],
    device const uint* edge_offsets [[buffer(3)]],
    device const uint* entry_points [[buffer(4)]],
    device uint* visited_flags [[buffer(6)]],
    device uint* result_indices [[buffer(7)]],
    device float* result_distances [[buffer(8)]],
    constant uint& B [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& D [[buffer(11)]],
    constant uint& ef_search [[buffer(12)]],
    constant uint& k [[buffer(13)]],
    constant uint& max_entry_points [[buffer(14)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one query
    if (tg_id >= B) return;

    // This is a simplified placeholder implementation
    // Full HNSW search requires:
    // 1. Priority queue management in shared memory
    // 2. Visited set tracking using atomic bit operations
    // 3. Layer-wise search with entry point selection
    // 4. Dynamic candidate list maintenance

    // For Phase 2, we initialize results to sentinel values
    // Full implementation will be added in subsequent phases
    if (tid < k) {
        result_indices[tg_id * k + tid] = K41_SENTINEL_INDEX;
        result_distances[tg_id * k + tid] = limits<float>::infinity();
    }
}
