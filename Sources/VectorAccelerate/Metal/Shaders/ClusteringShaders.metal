// VectorAccelerate: Clustering Shaders
//
// Metal shaders for K-means clustering operations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Optimized SIMD-Tiled KMeans Assignment Kernels

/// Arguments struct for optimized KMeans assignment kernels
struct KMeansAssignArgs {
    uint dimension;
    uint numVectors;
    uint numCentroids;
    uint _padding;
};

// Tiling constants for SIMD-tiled KMeans (VA_SIMD_WIDTH from Metal4Common.h)
constant uint VA_CENTROID_TILE   = 32;   // Centroids per tile (matches SIMD width)
constant uint VA_DIM_TILE        = 32;   // Dimensions per tile (8 float4s)
constant uint VA_DIM_TILE4       = 8;    // VA_DIM_TILE / 4

// 512-thread variant: 16 simdgroups => 16 vectors per threadgroup
constant uint VA_VECTORS_PER_TG_512 = 16;
constant uint VA_TG_SIZE_512        = 512; // VA_VECTORS_PER_TG_512 * VA_SIMD_WIDTH

/// High-performance simdgroup-cooperative KMeans assignment kernel (512-thread variant)
///
/// Each simdgroup processes one vector, with 32 lanes evaluating 32 centroids in parallel.
/// Uses tiled memory access for optimal cache utilization.
///
/// - Performance: ~10-20x faster than legacy kernel for typical PQ training workloads
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
        distances[vector_index]   = sqrt(best_dist);
    }
}

// 256-thread variant: 8 simdgroups => 8 vectors per threadgroup
constant uint VA_VECTORS_PER_TG_256 = 8;
constant uint VA_TG_SIZE_256        = 256; // VA_VECTORS_PER_TG_256 * VA_SIMD_WIDTH

/// High-performance simdgroup-cooperative KMeans assignment kernel (256-thread variant)
///
/// Smaller threadgroup variant for devices with limited occupancy.
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

// MARK: - Utility Functions

/// Compute inertia (sum of squared distances to assigned centroids)
kernel void compute_inertia(
    constant float* distances [[buffer(0)]],
    device atomic_float* inertia [[buffer(1)]],
    constant uint& num_vectors [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    // Square the distance (inertia uses squared Euclidean distance)
    float squared_distance = distances[id] * distances[id];
    
    // Atomically add to global inertia sum
    atomic_fetch_add_explicit(inertia, squared_distance, memory_order_relaxed);
}

/// Initialize atomic float buffers to zero
kernel void clear_atomic_buffers(
    device atomic_float* buffer [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    
    // Atomic store for proper initialization
    atomic_store_explicit(&buffer[id], 0.0f, memory_order_relaxed);
}

/// Initialize atomic uint buffers to zero
kernel void clear_atomic_uint_buffers(
    device atomic_uint* buffer [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    
    // Initialize to zero for count accumulation
    atomic_store_explicit(&buffer[id], 0, memory_order_relaxed);
}