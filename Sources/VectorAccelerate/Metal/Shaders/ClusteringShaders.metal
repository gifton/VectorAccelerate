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
    
    // Dynamically branch to ensure 16-byte alignment is valid before casting to float4
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
            device const float4* cent4 = (device const float4*)(centroids + (ulong)c_start * (ulong)dimension);
            threadgroup float4* tile4 = (threadgroup float4*)tile_centroids;
            
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
                device const float4* my_vec4 = (device const float4*)my_vec;
                
                // Read vector chunks sequentially, reusing them against all centroids in the tile
                for (uint i = 0; i < vec_dim; i++) {
                    float4 v_val = my_vec4[i];
                    for (uint c = 0; c < num_in_tile; c++) {
                        threadgroup const float4* c_vec4 = (threadgroup const float4*)(tile_centroids + (ulong)c * (ulong)dimension);
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

// MARK: - Legacy Tiled Distance Computation

// Tile dimensions optimized for Apple Silicon
constant int TILE_SIZE_Q = 32;  // Queries/vectors per tile
constant int TILE_SIZE_C = 8;   // Centroids per tile
constant int TILE_DIM = 32;     // Dimensions per tile for shared memory

// MARK: - Optimized 2D Tiled Distance Computation

/// High-performance 2D tiled distance computation for KMeans assignment
/// Processes Q×C distance matrix in tiles to maximize GPU utilization
kernel void tiled_kmeans_distance(
    device const float* vectors [[buffer(0)]],       // [num_vectors, dimensions]
    device const float* centroids [[buffer(1)]],     // [num_centroids, dimensions]
    device float* distances [[buffer(2)]],           // [num_vectors, num_centroids]
    device uint* assignments [[buffer(3)]],          // [num_vectors]
    constant uint& num_vectors [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimensions [[buffer(6)]],
    threadgroup float* shared_vectors [[threadgroup(0)]],   // Shared memory for vectors
    threadgroup float* shared_centroids [[threadgroup(1)]], // Shared memory for centroids
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    // Calculate global indices
    const uint global_vec = tgid.y * TILE_SIZE_Q + tid.y;
    const uint global_cent = tgid.x * TILE_SIZE_C + tid.x;
    
    // Accumulator for distance calculation
    float distance_accum = 0.0f;
    
    // Process dimension tiles
    for (uint dim_tile = 0; dim_tile < (dimensions + TILE_DIM - 1) / TILE_DIM; dim_tile++) {
        // Cooperative loading of vector tile into shared memory
        if (tid.x < TILE_DIM && global_vec < num_vectors) {
            uint dim_idx = dim_tile * TILE_DIM + tid.x;
            if (dim_idx < dimensions) {
                shared_vectors[tid.y * TILE_DIM + tid.x] = 
                    vectors[global_vec * dimensions + dim_idx];
            } else {
                shared_vectors[tid.y * TILE_DIM + tid.x] = 0.0f;
            }
        }
        
        // Cooperative loading of centroid tile into shared memory
        if (tid.y < TILE_SIZE_C && tid.x < TILE_DIM && global_cent < num_centroids) {
            uint dim_idx = dim_tile * TILE_DIM + tid.x;
            if (dim_idx < dimensions) {
                shared_centroids[tid.y * TILE_DIM + tid.x] = 
                    centroids[global_cent * dimensions + dim_idx];
            } else {
                shared_centroids[tid.y * TILE_DIM + tid.x] = 0.0f;
            }
        }
        
        // Synchronize to ensure shared memory is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial distances for this dimension tile
        if (global_vec < num_vectors && global_cent < num_centroids) {
            for (uint d = 0; d < TILE_DIM && (dim_tile * TILE_DIM + d) < dimensions; d++) {
                float diff = shared_vectors[tid.y * TILE_DIM + d] - 
                            shared_centroids[(tid.x % TILE_SIZE_C) * TILE_DIM + d];
                distance_accum += diff * diff;
            }
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final distance to global memory
    if (global_vec < num_vectors && global_cent < num_centroids) {
        distances[global_vec * num_centroids + global_cent] = sqrt(distance_accum);
    }
}

/// Find minimum distance and assignment from precomputed distance matrix
kernel void find_min_assignment(
    device const float* distances [[buffer(0)]],     // [num_vectors, num_centroids]
    device uint* assignments [[buffer(1)]],          // [num_vectors]
    device float* min_distances [[buffer(2)]],       // [num_vectors]
    constant uint& num_vectors [[buffer(3)]],
    constant uint& num_centroids [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    float min_dist = INFINITY;
    uint min_idx = 0;
    
    uint offset = id * num_centroids;
    for (uint c = 0; c < num_centroids; c++) {
        float dist = distances[offset + c];
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = c;
        }
    }
    
    assignments[id] = min_idx;
    min_distances[id] = min_dist;
}

// MARK: - Centroid Assignment (Legacy)

/// Assign each vector to its nearest centroid using Euclidean distance
kernel void assign_to_centroids(
    constant float* vectors [[buffer(0)]],
    constant float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimensions [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    // Early exit for threads beyond data bounds
    if (id >= num_vectors) return;
    
    // Calculate base offset for this vector
    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    uint min_centroid = 0;
    
    // Find nearest centroid
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;
        
        // Compute squared Euclidean distance
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        // Track the centroid with minimum distance
        if (distance < min_distance) {
            min_distance = distance;
            min_centroid = c;
        }
    }
    
    // Store results
    assignments[id] = min_centroid;
    distances[id] = sqrt(min_distance);
}

// MARK: - Optimized GPU Reduction for Centroid Updates

/// Parallel reduction for centroid updates using shared memory
/// Simplified version without threadgroup atomics for compatibility
kernel void gpu_reduce_centroids(
    device const float* vectors [[buffer(0)]],           // [num_vectors, dimensions]
    device const uint* assignments [[buffer(1)]],        // [num_vectors]
    device float* partial_sums [[buffer(2)]],           // [num_blocks, num_centroids, dimensions]
    device uint* partial_counts [[buffer(3)]],          // [num_blocks, num_centroids]
    constant uint& num_vectors [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimensions [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint vector_idx = gid.x;
    if (vector_idx >= num_vectors) return;
    
    uint assignment = assignments[vector_idx];
    uint block_idx = gid.y;  // Which reduction block this thread contributes to
    
    // Each thread processes one vector and updates the partial sums directly
    uint vec_offset = vector_idx * dimensions;
    uint partial_offset = (block_idx * num_centroids + assignment) * dimensions;
    
    // Update partial sums for this vector's assigned centroid
    for (uint d = 0; d < dimensions; d++) {
        // Use atomic operations on device memory
        device atomic_float* sum_ptr = (device atomic_float*)&partial_sums[partial_offset + d];
        atomic_fetch_add_explicit(sum_ptr, vectors[vec_offset + d], memory_order_relaxed);
    }
    
    // Update partial count
    device atomic_uint* count_ptr = (device atomic_uint*)&partial_counts[block_idx * num_centroids + assignment];
    atomic_fetch_add_explicit(count_ptr, 1u, memory_order_relaxed);
}

/// Combine partial sums from reduction blocks
kernel void combine_partial_centroids(
    device const float* partial_sums [[buffer(0)]],    // [num_blocks, num_centroids, dimensions]
    device const uint* partial_counts [[buffer(1)]],   // [num_blocks, num_centroids]
    device float* centroids [[buffer(2)]],             // [num_centroids, dimensions]
    device uint* centroid_counts [[buffer(3)]],        // [num_centroids]
    constant uint& num_blocks [[buffer(4)]],
    constant uint& num_centroids [[buffer(5)]],
    constant uint& dimensions [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x;  // Centroid index
    uint d = gid.y;  // Dimension index
    
    if (c >= num_centroids || d >= dimensions) return;
    
    // Sum across all blocks
    float sum = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        sum += partial_sums[(b * num_centroids + c) * dimensions + d];
    }
    
    // Count only needs to be done for first dimension
    if (d == 0) {
        uint count = 0;
        for (uint b = 0; b < num_blocks; b++) {
            count += partial_counts[b * num_centroids + c];
        }
        centroid_counts[c] = count;
        
        // Avoid division by zero
        if (count > 0) {
            centroids[c * dimensions + d] = sum / float(count);
        }
    } else {
        uint count = centroid_counts[c];
        if (count > 0) {
            centroids[c * dimensions + d] = sum / float(count);
        }
    }
}

// MARK: - Centroid Update (Legacy)

/// Update centroids based on assignments (step 1: accumulate)
kernel void accumulate_centroids(
    constant float* vectors [[buffer(0)]],
    constant uint* assignments [[buffer(1)]],
    device atomic_float* centroid_sums [[buffer(2)]],
    device atomic_uint* centroid_counts [[buffer(3)]],
    constant uint& dimensions [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    // Calculate offsets for this vector
    uint vector_offset = id * dimensions;
    uint assignment = assignments[id];
    uint centroid_offset = assignment * dimensions;
    
    // Accumulate this vector's values into assigned centroid
    for (uint d = 0; d < dimensions; d++) {
        atomic_fetch_add_explicit(
            &centroid_sums[centroid_offset + d],
            vectors[vector_offset + d],
            memory_order_relaxed
        );
    }
    
    // Increment count for this centroid
    atomic_fetch_add_explicit(
        &centroid_counts[assignment],
        1,
        memory_order_relaxed
    );
}

/// Update centroids based on assignments (step 2: divide)
kernel void finalize_centroids(
    device float* centroid_sums [[buffer(0)]],
    constant uint* centroid_counts [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    constant uint& num_centroids [[buffer(3)]],
    constant uint& dimensions [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    uint c = id.x;  // Centroid index
    uint d = id.y;  // Dimension index
    
    // Bounds checking
    if (c >= num_centroids || d >= dimensions) return;
    
    // Calculate linear index
    uint idx = c * dimensions + d;
    uint count = centroid_counts[c];
    
    if (count > 0) {
        // Compute mean
        centroids[idx] = centroid_sums[idx] / float(count);
    }
    // else: Keep existing centroid position for empty clusters
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
    
    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    
    // Find minimum distance to any existing centroid
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
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

// MARK: - Mini-batch K-means

/// Perform incremental centroid update for mini-batch K-means
kernel void update_centroids_incremental(
    device float* centroids [[buffer(0)]],
    constant float* batch_vectors [[buffer(1)]],
    constant uint* batch_assignments [[buffer(2)]],
    constant float& learning_rate [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& dimensions [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint dim = id.y;
    
    // Bounds check
    if (batch_idx >= batch_size || dim >= dimensions) return;
    
    // Get the centroid assignment for this batch vector
    uint assignment = batch_assignments[batch_idx];
    float vector_value = batch_vectors[batch_idx * dimensions + dim];
    
    // Incremental update using gradient descent
    uint centroid_idx = assignment * dimensions + dim;
    float old_value = centroids[centroid_idx];
    float new_value = old_value + learning_rate * (vector_value - old_value);
    
    // Update centroid position
    centroids[centroid_idx] = new_value;
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
                device const float4* vec4 = (device const float4*)(vectors + (ulong)i * (ulong)dimension);
                v = vec4[chunk_idx];
            } else {
                uint v_offset = i * dimension + base_dim;
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
        
        uint out_offset = cluster_id * dimension + base_dim;
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