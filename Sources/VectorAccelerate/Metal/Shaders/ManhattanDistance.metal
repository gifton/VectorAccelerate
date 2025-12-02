// VectorAccelerate: Manhattan Distance Kernel (L1 Norm)
//
// GPU-accelerated Manhattan distance computation
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// =============================================================================
// Configuration Constants
// =============================================================================

#define TILE_Q 16           // Query tile height
#define TILE_N 16           // Dataset tile width
#define TILE_D 64           // Dimension tile size
#define TILE_D_VEC 16       // TILE_D / 4 for float4 operations

// =============================================================================
// Main Kernels
// =============================================================================

/// Computes Manhattan distance matrix (L1 norm)
/// Distance = Σ|x_i - y_i|
kernel void manhattan_distance_batch(
    device const float* queries [[buffer(0)]],      // [Q × D]
    device const float* dataset [[buffer(1)]],      // [N × D]
    device float* distances [[buffer(2)]],          // [Q × N]
    constant uint& Q [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],         // (n_idx, q_idx)
    uint2 tid [[thread_position_in_threadgroup]]   // (local_n, local_q)
) {
    // Shared memory for tiled computation
    threadgroup float shared_Q[TILE_Q * TILE_D];
    threadgroup float shared_N[TILE_N * TILE_D];
    
    // Vectorized access for better performance
    threadgroup float4* shared_Q_f4 = reinterpret_cast<threadgroup float4*>(shared_Q);
    threadgroup float4* shared_N_f4 = reinterpret_cast<threadgroup float4*>(shared_N);
    
    // Calculate tile boundaries
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    
    // Linear thread ID for cooperative loading
    const uint tid_linear = tid.y * TILE_N + tid.x;
    const uint tile_row = tid_linear / TILE_D_VEC;
    const uint tile_col_f4 = tid_linear % TILE_D_VEC;
    
    // Initialize accumulator
    float manhattan_dist = 0.0f;
    
    // Process dimensions in tiles
    for (uint d_start = 0; d_start < D; d_start += TILE_D) {
        
        // Cooperative loading of query vectors
        const uint global_d_start = d_start + tile_col_f4 * 4;
        const uint global_q_idx = start_q + tile_row;
        float4 q_data = float4(0.0f);
        
        if (global_q_idx < Q && global_d_start < D) {
            const device float* q_ptr = queries + (uint64_t)global_q_idx * D + global_d_start;
            
            if (global_d_start + 4 <= D) {
                // Fast path: aligned float4 load
                q_data = *(reinterpret_cast<const device float4*>(q_ptr));
            } else {
                // Slow path: partial load
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    q_data[i] = q_ptr[i];
                }
            }
        }
        
        if (tile_row < TILE_Q && tile_col_f4 < TILE_D_VEC) {
            shared_Q_f4[tile_row * TILE_D_VEC + tile_col_f4] = q_data;
        }
        
        // Cooperative loading of dataset vectors
        const uint global_n_idx = start_n + tile_row;
        float4 n_data = float4(0.0f);
        
        if (global_n_idx < N && global_d_start < D) {
            const device float* n_ptr = dataset + (uint64_t)global_n_idx * D + global_d_start;
            
            if (global_d_start + 4 <= D) {
                // Fast path: aligned float4 load
                n_data = *(reinterpret_cast<const device float4*>(n_ptr));
            } else {
                // Slow path: partial load
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    n_data[i] = n_ptr[i];
                }
            }
        }
        
        if (tile_row < TILE_N && tile_col_f4 < TILE_D_VEC) {
            shared_N_f4[tile_row * TILE_D_VEC + tile_col_f4] = n_data;
        }
        
        // Synchronize after loading
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Manhattan distance for this tile
        const uint tile_end = min((uint)TILE_D, D - d_start);
        
        // Vectorized computation when possible
        const uint vec_end = tile_end / 4;
        for (uint k = 0; k < vec_end; ++k) {
            float4 q_val = shared_Q_f4[tid.y * TILE_D_VEC + k];
            float4 n_val = shared_N_f4[tid.x * TILE_D_VEC + k];
            
            // Compute absolute differences
            float4 diff = abs(q_val - n_val);
            
            // Sum components
            manhattan_dist += diff.x + diff.y + diff.z + diff.w;
        }
        
        // Handle remaining elements
        for (uint d = vec_end * 4; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * TILE_D + d];
            float n_val = shared_N[tid.x * TILE_D + d];
            manhattan_dist += abs(q_val - n_val);
        }
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result to global memory
    if (gid.y < Q && gid.x < N) {
        distances[(uint64_t)gid.y * N + gid.x] = manhattan_dist;
    }
}

// =============================================================================
// Optimized Variants
// =============================================================================

/// Manhattan distance with FMA optimization for weighted L1
kernel void manhattan_distance_weighted(
    device const float* queries [[buffer(0)]],      // [Q × D]
    device const float* dataset [[buffer(1)]],      // [N × D]
    device const float* weights [[buffer(2)]],      // [D] - per-dimension weights
    device float* distances [[buffer(3)]],          // [Q × N]
    constant uint& Q [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_Q[TILE_Q * TILE_D];
    threadgroup float shared_N[TILE_N * TILE_D];
    threadgroup float shared_W[TILE_D];  // Shared weights
    
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    const uint tid_linear = tid.y * TILE_N + tid.x;
    
    float weighted_manhattan = 0.0f;
    
    for (uint d_start = 0; d_start < D; d_start += TILE_D) {
        
        // Load weights cooperatively
        if (tid_linear < TILE_D) {
            uint global_d = d_start + tid_linear;
            shared_W[tid_linear] = (global_d < D) ? weights[global_d] : 1.0f;
        }
        
        // Load queries and dataset (same as before)
        for (uint d = tid_linear; d < TILE_Q * TILE_D; d += TILE_Q * TILE_N) {
            uint q_idx = d / TILE_D;
            uint d_idx = d % TILE_D;
            uint global_q = start_q + q_idx;
            uint global_d = d_start + d_idx;
            
            shared_Q[d] = (global_q < Q && global_d < D) ?
                         queries[(uint64_t)global_q * D + global_d] : 0.0f;
        }
        
        for (uint d = tid_linear; d < TILE_N * TILE_D; d += TILE_Q * TILE_N) {
            uint n_idx = d / TILE_D;
            uint d_idx = d % TILE_D;
            uint global_n = start_n + n_idx;
            uint global_d = d_start + d_idx;
            
            shared_N[d] = (global_n < N && global_d < D) ?
                         dataset[(uint64_t)global_n * D + global_d] : 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute weighted Manhattan distance
        const uint tile_end = min((uint)TILE_D, D - d_start);
        for (uint d = 0; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * TILE_D + d];
            float n_val = shared_N[tid.x * TILE_D + d];
            float weight = shared_W[d];
            
            // Weighted absolute difference
            weighted_manhattan = fma(weight, abs(q_val - n_val), weighted_manhattan);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (gid.y < Q && gid.x < N) {
        distances[(uint64_t)gid.y * N + gid.x] = weighted_manhattan;
    }
}

// =============================================================================
// Single Vector Variants
// =============================================================================

/// Manhattan distance between two vectors
kernel void manhattan_distance_single(
    device const float* vectorA [[buffer(0)]],      // [D]
    device const float* vectorB [[buffer(1)]],      // [D]
    device float* distance [[buffer(2)]],           // [1]
    constant uint& D [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes a portion of the dimension
    const uint ELEMS_PER_THREAD = 16;
    const uint start = tid * ELEMS_PER_THREAD;
    
    float local_sum = 0.0f;
    
    // Vectorized processing
    for (uint i = start; i < min(start + ELEMS_PER_THREAD, D); i += 4) {
        if (i + 4 <= D) {
            float4 a = *(reinterpret_cast<const device float4*>(vectorA + i));
            float4 b = *(reinterpret_cast<const device float4*>(vectorB + i));
            float4 diff = abs(a - b);
            local_sum += diff.x + diff.y + diff.z + diff.w;
        } else {
            // Handle remainder
            for (uint j = i; j < min(i + 4, D) && j < D; ++j) {
                local_sum += abs(vectorA[j] - vectorB[j]);
            }
        }
    }
    
    // Atomic add to accumulate across threads
    atomic_fetch_add_explicit(
        reinterpret_cast<device atomic<float>*>(distance),
        local_sum,
        memory_order_relaxed
    );
}

// =============================================================================
// Normalized Manhattan Distance
// =============================================================================

/// Normalized Manhattan distance (divided by dimension)
kernel void manhattan_distance_normalized(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& Q [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Similar to manhattan_distance_batch but with normalization
    threadgroup float shared_Q[TILE_Q * TILE_D];
    threadgroup float shared_N[TILE_N * TILE_D];
    threadgroup float4* shared_Q_f4 = reinterpret_cast<threadgroup float4*>(shared_Q);
    threadgroup float4* shared_N_f4 = reinterpret_cast<threadgroup float4*>(shared_N);
    
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    const uint tid_linear = tid.y * TILE_N + tid.x;
    const uint tile_row = tid_linear / TILE_D_VEC;
    const uint tile_col_f4 = tid_linear % TILE_D_VEC;
    
    float manhattan_dist = 0.0f;
    
    for (uint d_start = 0; d_start < D; d_start += TILE_D) {
        const uint global_d_start = d_start + tile_col_f4 * 4;
        const uint global_q_idx = start_q + tile_row;
        float4 q_data = float4(0.0f);
        
        if (global_q_idx < Q && global_d_start < D) {
            const device float* q_ptr = queries + (uint64_t)global_q_idx * D + global_d_start;
            if (global_d_start + 4 <= D) {
                q_data = *(reinterpret_cast<const device float4*>(q_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    q_data[i] = q_ptr[i];
                }
            }
        }
        
        if (tile_row < TILE_Q && tile_col_f4 < TILE_D_VEC) {
            shared_Q_f4[tile_row * TILE_D_VEC + tile_col_f4] = q_data;
        }
        
        const uint global_n_idx = start_n + tile_row;
        float4 n_data = float4(0.0f);
        
        if (global_n_idx < N && global_d_start < D) {
            const device float* n_ptr = dataset + (uint64_t)global_n_idx * D + global_d_start;
            if (global_d_start + 4 <= D) {
                n_data = *(reinterpret_cast<const device float4*>(n_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    n_data[i] = n_ptr[i];
                }
            }
        }
        
        if (tile_row < TILE_N && tile_col_f4 < TILE_D_VEC) {
            shared_N_f4[tile_row * TILE_D_VEC + tile_col_f4] = n_data;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        const uint tile_end = min((uint)TILE_D, D - d_start);
        const uint vec_end = tile_end / 4;
        
        for (uint k = 0; k < vec_end; ++k) {
            float4 q_val = shared_Q_f4[tid.y * TILE_D_VEC + k];
            float4 n_val = shared_N_f4[tid.x * TILE_D_VEC + k];
            float4 diff = abs(q_val - n_val);
            manhattan_dist += diff.x + diff.y + diff.z + diff.w;
        }
        
        for (uint d = vec_end * 4; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * TILE_D + d];
            float n_val = shared_N[tid.x * TILE_D + d];
            manhattan_dist += abs(q_val - n_val);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize by dimension and write
    if (gid.y < Q && gid.x < N) {
        distances[(uint64_t)gid.y * N + gid.x] = manhattan_dist / float(D);
    }
}
