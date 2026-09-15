// VectorAccelerate: Clustering Shaders
//
// Metal shaders for K-means clustering operations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Optimized Tiled KMeans Assignment (Register-Cached)

// -----------------------------------------------------------------------------
// K-Means Assignment (Tiled Shared Memory + Register Caching)
// 1 Thread = 1 Vector (evaluating against all centroids in a loaded tile)
// -----------------------------------------------------------------------------
kernel void kmeans_assign_points(
    device const float* vectors [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimension [[buffer(6)]],
    constant uint& tile_capacity [[buffer(7)]],
    threadgroup float* tile_centroids [[threadgroup(0)]], // Dynamically bounded 32KB block
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    bool is_active = gid < num_vectors;
    
    // Inactive threads evaluate point 0 to prevent out-of-bounds global reads.
    // They MUST participate in the cooperative load and barriers below to prevent deadlock.
    uint safe_gid = is_active ? gid : 0;
    
    device const float* my_vec = vectors + (ulong)safe_gid * (ulong)dimension;
    uint vec_dim = dimension / 4;
    
    float min_dist = VA_INFINITY;
    uint min_idx = 0;
    
    // Whole-vector rows allow complete four-component tile loads; packed views
    // retain scalar storage alignment. Ragged rows use the scalar tile path.
    bool is_aligned = (dimension % 4 == 0);
    
    for (uint c_start = 0; c_start < num_centroids; c_start += tile_capacity) {
        uint c_end = min(c_start + tile_capacity, num_centroids);
        
        // Safely clamp to 32 to guarantee it fits in the static array registers
        uint num_in_tile = min(c_end - c_start, 32u);
        
        // ---------------------------------------------------------------------
        // PHASE 1: Cooperative Tile Load 
        // ---------------------------------------------------------------------
        uint total_floats = num_in_tile * dimension;
        
        if (is_aligned) {
            uint total_float4s = total_floats / 4;
            device const packed_float4* cent4 = (device const packed_float4*)(centroids + (ulong)c_start * (ulong)dimension);
            threadgroup packed_float4* tile4 = (threadgroup packed_float4*)tile_centroids;
            
            // Threads cooperate to safely stream the global memory tile into L1 Shared memory
            for (uint i = lid; i < total_float4s; i += threads_per_tg) {
                tile4[i] = cent4[i];
            }
        } else {
            device const float* cent = centroids + (ulong)c_start * (ulong)dimension;
            for (uint i = lid; i < total_floats; i += threads_per_tg) {
                tile_centroids[i] = cent[i];
            }
        }
        
        // Block until all threads finish loading the tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ---------------------------------------------------------------------
        // PHASE 2: Register-Cached Compute (Float4 Optimized)
        // ---------------------------------------------------------------------
        if (is_active) {
            float dists[32]; 
            for (uint c = 0; c < 32; c++) {
                dists[c] = 0.0;
            }
            
            if (is_aligned) {
                device const packed_float4* my_vec4 = (device const packed_float4*)my_vec;
                
                // Read vector chunks sequentially, reusing them against all centroids in the tile
                for (uint i = 0; i < vec_dim; i++) {
                    float4 v_val = my_vec4[i];
                    for (uint c = 0; c < num_in_tile; c++) {
                        threadgroup const packed_float4* c_vec4 = (threadgroup const packed_float4*)(tile_centroids + (ulong)c * (ulong)dimension);
                        float4 diff = v_val - c_vec4[i];
                        dists[c] += dot(diff, diff);
                    }
                }
            } else {
                for (uint i = 0; i < dimension; i++) {
                    float v_val = my_vec[i];
                    for (uint c = 0; c < num_in_tile; c++) {
                        float c_val = tile_centroids[(ulong)c * (ulong)dimension + i];
                        float diff = v_val - c_val;
                        dists[c] += diff * diff;
                    }
                }
            }
            
            // Local Top-1 Argmin 
            for (uint c = 0; c < num_in_tile; c++) {
                if (dists[c] < min_dist) {
                    min_dist = dists[c];
                    min_idx = c_start + c;
                }
            }
        }
        
        // Synchronize before loading the next centroid tile overwrite
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // -------------------------------------------------------------------------
    // PHASE 3: Exactly ONE uncontended write per vector
    // -------------------------------------------------------------------------
    if (is_active) {
        assignments[gid] = min_idx;
        distances[gid] = sqrt(min_dist);
    }
}

// MARK: - K-means++ Initialization

/// Compute minimum distances to existing centroids for K-means++
kernel void compute_min_distances(
    constant float* vectors [[buffer(0)]],
    constant float* centroids [[buffer(1)]],
    device float* min_distances [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& num_centroids [[buffer(4)]],
    constant uint& dimensions [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    ulong vector_offset = (ulong)id * dimensions;
    float min_distance = INFINITY;
    
    // Find minimum distance to any existing centroid
    for (uint c = 0; c < num_centroids; c++) {
        ulong centroid_offset = (ulong)c * dimensions;
        float distance = 0.0f;
        
        // Squared Euclidean distance
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        min_distance = min(min_distance, distance);
    }
    
    // Store squared distance for probability-weighted selection
    min_distances[id] = min_distance;
}

// MARK: - Optimized GPU K-Means Update (2-Pass)

// -----------------------------------------------------------------------------
// Pass 1: K-Means Accumulate (Cooperative Gather Topology)
// 1 Threadgroup = 1 Cluster & 1 Dimension Chunk (Float4)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Pass 1: K-Means Accumulate (Cooperative Gather Topology)
// 1 Threadgroup = 1 Cluster & 1 Dimension Chunk (Float4)
// -----------------------------------------------------------------------------
kernel void kmeans_update_accumulate(
    device const float* vectors [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device atomic_float* cluster_sums [[buffer(2)]],
    device atomic_uint* cluster_counts [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& dimension [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 threads_per_tg [[threads_per_threadgroup]]
) {
    uint cluster_id = tgid.x;
    uint chunk_idx = tgid.y;
    
    uint base_dim = chunk_idx * 4;
    if (base_dim >= dimension) return;
    
    uint dims_to_read = min(4u, dimension - base_dim);
    bool is_aligned = (dimension % 4 == 0);
    
    float4 local_sum = 0.0;
    uint local_count = 0;
    
    // Phase 1: Cooperative Gather 
    for (uint i = lid.x; i < num_vectors; i += threads_per_tg.x) {
        if (assignments[i] == cluster_id) {
            // Only chunk 0 tracks vector assignment counts to prevent inflation
            if (chunk_idx == 0) {
                local_count++;
            }
            
            float4 v = 0.0;
            if (is_aligned && dims_to_read == 4) {
                device const packed_float4* vec4 = (device const packed_float4*)(vectors + (ulong)i * (ulong)dimension);
                v = vec4[chunk_idx];
            } else {
                ulong v_offset = (ulong)i * dimension + base_dim;
                if (dims_to_read >= 1) v.x = vectors[v_offset];
                if (dims_to_read >= 2) v.y = vectors[v_offset + 1];
                if (dims_to_read >= 3) v.z = vectors[v_offset + 2];
                if (dims_to_read == 4) v.w = vectors[v_offset + 3];
            }
            local_sum += v;
        }
    }
    
    // Phase 2: SIMD Sum 
    float4 simd_sum_val;
    simd_sum_val.x = simd_sum(local_sum.x);
    simd_sum_val.y = simd_sum(local_sum.y);
    simd_sum_val.z = simd_sum(local_sum.z);
    simd_sum_val.w = simd_sum(local_sum.w);
    uint simd_count = simd_sum(local_count);
    
    // Phase 3: Threadgroup sum via shared memory
    threadgroup float4 shared_sums[32];
    threadgroup uint shared_counts[32];
    
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_sum_val;
        if (chunk_idx == 0) shared_counts[simd_group_id] = simd_count;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 4: Exactly ONE atomic write block per dimension chunk per threadgroup
    if (lid.x == 0) {
        uint active_simd_groups = (threads_per_tg.x + 31) / 32;
        float4 final_sum = 0.0;
        uint final_count = 0;
        
        for (uint i = 0; i < active_simd_groups; i++) {
            final_sum += shared_sums[i];
            if (chunk_idx == 0) final_count += shared_counts[i];
        }
        
        ulong out_offset = (ulong)cluster_id * dimension + base_dim;
        if (dims_to_read >= 1) atomic_fetch_add_explicit(&cluster_sums[out_offset], final_sum.x, memory_order_relaxed);
        if (dims_to_read >= 2) atomic_fetch_add_explicit(&cluster_sums[out_offset + 1], final_sum.y, memory_order_relaxed);
        if (dims_to_read >= 3) atomic_fetch_add_explicit(&cluster_sums[out_offset + 2], final_sum.z, memory_order_relaxed);
        if (dims_to_read == 4) atomic_fetch_add_explicit(&cluster_sums[out_offset + 3], final_sum.w, memory_order_relaxed);
        
        if (chunk_idx == 0 && final_count > 0) {
            atomic_fetch_add_explicit(&cluster_counts[cluster_id], final_count, memory_order_relaxed);
        }
    }
}

// -----------------------------------------------------------------------------
// Pass 2: K-Means Normalize 
// -----------------------------------------------------------------------------
kernel void kmeans_update_normalize(
    device const float* cluster_sums [[buffer(0)]],
    device const uint* cluster_counts [[buffer(1)]],
    device float* new_centroids [[buffer(2)]],
    device const float* old_centroids [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    uint cluster_id = tgid;
    uint count = cluster_counts[cluster_id];
    
    device const float* sums = cluster_sums + (ulong)cluster_id * (ulong)dimension;
    device float* out_cent = new_centroids + (ulong)cluster_id * (ulong)dimension;
    device const float* prev_cent = old_centroids + (ulong)cluster_id * (ulong)dimension;
    
    if (count > 0) {
        float inv_count = 1.0f / float(count);
        for (uint i = lid; i < dimension; i += threads_per_tg) {
            out_cent[i] = sums[i] * inv_count;
        }
    } else {
        // Preserve existing centroids for empty clusters
        for (uint i = lid; i < dimension; i += threads_per_tg) {
            out_cent[i] = prev_cent[i];
        }
    }
}