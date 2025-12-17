#include "Metal4Common.h"
using namespace metal;

struct KMeansAssignArgs {
    uint dimension;
    uint numVectors;
    uint numCentroids;
    uint _padding;
};

constexpr uint VA_SIMD_WIDTH      = 32;
constexpr uint VA_CENTROID_TILE   = 32;

// Dimension tiling: 32 floats per tile = 8 float4s
constexpr uint VA_DIM_TILE        = 32;
constexpr uint VA_DIM_TILE4       = VA_DIM_TILE / 4;

// ------------------------------
// 512-threadgroup variant
// 16 simdgroups => 16 vectors per threadgroup
// ------------------------------
constexpr uint VA_VECTORS_PER_TG_512 = 16;
constexpr uint VA_TG_SIZE_512        = VA_VECTORS_PER_TG_512 * VA_SIMD_WIDTH;

kernel void assign_to_centroids_simd_tiled_512(
    device const float*      vectors     [[buffer(0)]],
    device const float*      centroids   [[buffer(1)]],
    device uint*             assignments [[buffer(2)]],
    device float*            distances   [[buffer(3)]],
    constant KMeansAssignArgs& args      [[buffer(4)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint   tid       [[thread_index_in_threadgroup]],
    uint   lane      [[thread_index_in_simdgroup]],
    uint   sgid      [[simdgroup_index_in_threadgroup]])
{
    const uint dimensions    = args.dimension;
    const uint num_vectors   = args.numVectors;
    const uint num_centroids = args.numCentroids;

    // One simdgroup processes one vector
    const uint vector_index  = tg_pos.x * VA_VECTORS_PER_TG_512 + sgid;
    const bool vector_active = (vector_index < num_vectors);

    // Threadgroup staging:
    // vectors staged as float4: [vectorsPerTG][DIM_TILE4]
    // centroids staged as float4, transposed by (d4, centroid): [DIM_TILE4][CENTROID_TILE]
    threadgroup float4 tg_vectors4[VA_VECTORS_PER_TG_512 * VA_DIM_TILE4];
    threadgroup float4 tg_centroids4[VA_DIM_TILE4 * VA_CENTROID_TILE];

    float best_dist = VA_INFINITY;
    uint  best_centroid = 0;

    for (uint c_base = 0; c_base < num_centroids; c_base += VA_CENTROID_TILE) {
        const uint centroid_index = c_base + lane;
        float dist = 0.0f;

        for (uint d_base = 0; d_base < dimensions; d_base += VA_DIM_TILE) {

            // 1) Load this vector's tile (8 float4s) into threadgroup memory.
            // Only first 8 lanes participate (one float4 per lane).
            if (lane < VA_DIM_TILE4) {
                const uint d = d_base + lane * 4;

                float4 v = float4(0.0f);
                if (vector_active) {
                    const ulong v_off = (ulong)vector_index * (ulong)dimensions;

                    if (d + 0 < dimensions) v[0] = vectors[v_off + (d + 0)];
                    if (d + 1 < dimensions) v[1] = vectors[v_off + (d + 1)];
                    if (d + 2 < dimensions) v[2] = vectors[v_off + (d + 2)];
                    if (d + 3 < dimensions) v[3] = vectors[v_off + (d + 3)];
                }
                tg_vectors4[sgid * VA_DIM_TILE4 + lane] = v;
            }

            // 2) Load centroid tile into threadgroup memory (float4 transposed layout)
            // Total elements per dim-tile: 32 centroids * 8 float4s = 256 float4s.
            for (uint i = tid; i < (VA_CENTROID_TILE * VA_DIM_TILE4); i += VA_TG_SIZE_512) {
                const uint c_local = i / VA_DIM_TILE4;           // [0..31]
                const uint d4      = i - c_local * VA_DIM_TILE4; // [0..7]

                const uint c = c_base + c_local;
                const uint d = d_base + d4 * 4;

                float4 cv = float4(0.0f);
                if (c < num_centroids) {
                    const ulong c_off = (ulong)c * (ulong)dimensions;

                    if (d + 0 < dimensions) cv[0] = centroids[c_off + (d + 0)];
                    if (d + 1 < dimensions) cv[1] = centroids[c_off + (d + 1)];
                    if (d + 2 < dimensions) cv[2] = centroids[c_off + (d + 2)];
                    if (d + 3 < dimensions) cv[3] = centroids[c_off + (d + 3)];
                }

                // Transposed by (d4, centroid): contiguous across lanes for a fixed d4
                tg_centroids4[d4 * VA_CENTROID_TILE + c_local] = cv;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 3) Accumulate partial distance for this lane's centroid
            if (vector_active && centroid_index < num_centroids) {
                const uint v_base = sgid * VA_DIM_TILE4;

                #pragma unroll
                for (uint d4 = 0; d4 < VA_DIM_TILE4; ++d4) {
                    const float4 v = tg_vectors4[v_base + d4];
                    const float4 c = tg_centroids4[d4 * VA_CENTROID_TILE + lane];
                    const float4 diff = v - c;
                    dist += dot(diff, diff);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Mask invalid
        if (!vector_active || centroid_index >= num_centroids) {
            dist = VA_INFINITY;
        }

        // simdgroup argmin reduction (distance, centroid_index)
        float min_dist = dist;
        uint  min_idx  = centroid_index;

        #pragma unroll
        for (uint offset = VA_SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            const float other_dist = simd_shuffle_down(min_dist, offset);
            const uint  other_idx  = simd_shuffle_down(min_idx,  offset);

            // Tie-break: choose lower centroid index deterministically
            if ((other_dist < min_dist) || ((other_dist == min_dist) && (other_idx < min_idx))) {
                min_dist = other_dist;
                min_idx  = other_idx;
            }
        }

        if (lane == 0 && vector_active) {
            if (min_dist < best_dist) {
                best_dist = min_dist;
                best_centroid = min_idx;
            }
        }
    }

    if (lane == 0 && vector_active) {
        assignments[vector_index] = best_centroid;
        distances[vector_index]   = sqrt(best_dist); // matches legacy semantics  [oai_citation:3‡HANDOFF_1_KMEANS_GPU_ACCELERATION.md](sediment://file_00000000a3cc722f91325c225716cffd)
    }
}

// ------------------------------
// 256-threadgroup variant
// 8 simdgroups => 8 vectors per threadgroup
// ------------------------------
constexpr uint VA_VECTORS_PER_TG_256 = 8;
constexpr uint VA_TG_SIZE_256        = VA_VECTORS_PER_TG_256 * VA_SIMD_WIDTH;

kernel void assign_to_centroids_simd_tiled_256(
    device const float*      vectors     [[buffer(0)]],
    device const float*      centroids   [[buffer(1)]],
    device uint*             assignments [[buffer(2)]],
    device float*            distances   [[buffer(3)]],
    constant KMeansAssignArgs& args      [[buffer(4)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint   tid       [[thread_index_in_threadgroup]],
    uint   lane      [[thread_index_in_simdgroup]],
    uint   sgid      [[simdgroup_index_in_threadgroup]])
{
    const uint dimensions    = args.dimension;
    const uint num_vectors   = args.numVectors;
    const uint num_centroids = args.numCentroids;

    const uint vector_index  = tg_pos.x * VA_VECTORS_PER_TG_256 + sgid;
    const bool vector_active = (vector_index < num_vectors);

    threadgroup float4 tg_vectors4[VA_VECTORS_PER_TG_256 * VA_DIM_TILE4];
    threadgroup float4 tg_centroids4[VA_DIM_TILE4 * VA_CENTROID_TILE];

    float best_dist = VA_INFINITY;
    uint  best_centroid = 0;

    for (uint c_base = 0; c_base < num_centroids; c_base += VA_CENTROID_TILE) {
        const uint centroid_index = c_base + lane;
        float dist = 0.0f;

        for (uint d_base = 0; d_base < dimensions; d_base += VA_DIM_TILE) {

            if (lane < VA_DIM_TILE4) {
                const uint d = d_base + lane * 4;

                float4 v = float4(0.0f);
                if (vector_active) {
                    const ulong v_off = (ulong)vector_index * (ulong)dimensions;

                    if (d + 0 < dimensions) v[0] = vectors[v_off + (d + 0)];
                    if (d + 1 < dimensions) v[1] = vectors[v_off + (d + 1)];
                    if (d + 2 < dimensions) v[2] = vectors[v_off + (d + 2)];
                    if (d + 3 < dimensions) v[3] = vectors[v_off + (d + 3)];
                }
                tg_vectors4[sgid * VA_DIM_TILE4 + lane] = v;
            }

            for (uint i = tid; i < (VA_CENTROID_TILE * VA_DIM_TILE4); i += VA_TG_SIZE_256) {
                const uint c_local = i / VA_DIM_TILE4;
                const uint d4      = i - c_local * VA_DIM_TILE4;

                const uint c = c_base + c_local;
                const uint d = d_base + d4 * 4;

                float4 cv = float4(0.0f);
                if (c < num_centroids) {
                    const ulong c_off = (ulong)c * (ulong)dimensions;

                    if (d + 0 < dimensions) cv[0] = centroids[c_off + (d + 0)];
                    if (d + 1 < dimensions) cv[1] = centroids[c_off + (d + 1)];
                    if (d + 2 < dimensions) cv[2] = centroids[c_off + (d + 2)];
                    if (d + 3 < dimensions) cv[3] = centroids[c_off + (d + 3)];
                }

                tg_centroids4[d4 * VA_CENTROID_TILE + c_local] = cv;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (vector_active && centroid_index < num_centroids) {
                const uint v_base = sgid * VA_DIM_TILE4;

                #pragma unroll
                for (uint d4 = 0; d4 < VA_DIM_TILE4; ++d4) {
                    const float4 v = tg_vectors4[v_base + d4];
                    const float4 c = tg_centroids4[d4 * VA_CENTROID_TILE + lane];
                    const float4 diff = v - c;
                    dist += dot(diff, diff);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (!vector_active || centroid_index >= num_centroids) {
            dist = VA_INFINITY;
        }

        float min_dist = dist;
        uint  min_idx  = centroid_index;

        #pragma unroll
        for (uint offset = VA_SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            const float other_dist = simd_shuffle_down(min_dist, offset);
            const uint  other_idx  = simd_shuffle_down(min_idx,  offset);

            if ((other_dist < min_dist) || ((other_dist == min_dist) && (other_idx < min_idx))) {
                min_dist = other_dist;
                min_idx  = other_idx;
            }
        }

        if (lane == 0 && vector_active) {
            if (min_dist < best_dist) {
                best_dist = min_dist;
                best_centroid = min_idx;
            }
        }
    }

    if (lane == 0 && vector_active) {
        assignments[vector_index] = best_centroid;
        distances[vector_index]   = sqrt(best_dist);
    }
}
