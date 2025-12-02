// VectorAccelerate: Clustering Shaders
//
// Metal shaders for K-means clustering operations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

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