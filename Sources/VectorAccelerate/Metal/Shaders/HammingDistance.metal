// VectorAccelerate: Hamming Distance Kernel
//
// GPU-accelerated Hamming distance for binary vectors
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// =============================================================================
// Configuration Constants
// =============================================================================

// Tiling configuration optimized for binary operations
// Prefixed with HAMM_ to avoid conflicts when shaders are combined
#define HAMM_TILE_Q 16           // Query tile height
#define HAMM_TILE_N 16           // Dataset tile width
#define HAMM_VEC_WIDTH 4         // uint4 = 128 bits
#define HAMM_TILE_D_WORDS 64     // uint32 words per tile (2048 bits)
#define HAMM_TILE_D_VEC (HAMM_TILE_D_WORDS / HAMM_VEC_WIDTH)  // 16 uint4 vectors

// =============================================================================
// Main Kernels
// =============================================================================

/// Computes Hamming distance matrix for bit-packed binary vectors
/// Uses XOR and population count for efficient computation
kernel void hamming_distance_batch(
    device const uint* queries [[buffer(0)]],       // [Q × D_words]
    device const uint* dataset [[buffer(1)]],       // [N × D_words]
    device uint* distances [[buffer(2)]],           // [Q × N]
    constant uint& Q [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D_words [[buffer(5)]],          // D/32 (number of uint32 words)
    uint2 gid [[thread_position_in_grid]],         // (n_idx, q_idx)
    uint2 tid [[thread_position_in_threadgroup]]   // (local_n, local_q)
) {
    // Shared memory for tiled computation
    threadgroup uint shared_Q[HAMM_TILE_Q * HAMM_TILE_D_WORDS];
    threadgroup uint shared_N[HAMM_TILE_N * HAMM_TILE_D_WORDS];
    
    // Vectorized access pointers for 128-bit operations
    threadgroup uint4* shared_Q_u4 = reinterpret_cast<threadgroup uint4*>(shared_Q);
    threadgroup uint4* shared_N_u4 = reinterpret_cast<threadgroup uint4*>(shared_N);
    
    // Calculate tile boundaries
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    
    // Linear thread ID for cooperative loading
    const uint tid_linear = tid.y * HAMM_TILE_N + tid.x;
    const uint tile_row = tid_linear / HAMM_TILE_D_VEC;
    const uint tile_col_u4 = tid_linear % HAMM_TILE_D_VEC;
    const uint smem_idx = tid_linear;
    
    // Initialize accumulator
    uint hamming_dist = 0;
    
    // Process binary vectors in tiles
    for (uint d_start = 0; d_start < D_words; d_start += HAMM_TILE_D_WORDS) {
        
        // Cooperative loading of query vectors
        const uint global_d_start = d_start + tile_col_u4 * HAMM_VEC_WIDTH;
        uint global_q_idx = start_q + tile_row;
        uint4 q_data = uint4(0);
        
        if (global_q_idx < Q) {
            if (global_d_start + HAMM_VEC_WIDTH <= D_words) {
                // Fast path: aligned uint4 load (128 bits)
                device const uint* q_ptr = queries + (uint64_t)global_q_idx * D_words + global_d_start;
                q_data = *(reinterpret_cast<device const uint4*>(q_ptr));
            } else if (global_d_start < D_words) {
                // Slow path: partial load for remainder
                for (uint i = 0; i < HAMM_VEC_WIDTH && global_d_start + i < D_words; ++i) {
                    q_data[i] = queries[(uint64_t)global_q_idx * D_words + global_d_start + i];
                }
            }
        }
        shared_Q_u4[smem_idx] = q_data;
        
        // Cooperative loading of dataset vectors
        uint global_n_idx = start_n + tile_row;
        uint4 n_data = uint4(0);
        
        if (global_n_idx < N) {
            if (global_d_start + HAMM_VEC_WIDTH <= D_words) {
                // Fast path: aligned uint4 load
                device const uint* n_ptr = dataset + (uint64_t)global_n_idx * D_words + global_d_start;
                n_data = *(reinterpret_cast<device const uint4*>(n_ptr));
            } else if (global_d_start < D_words) {
                // Slow path: partial load
                for (uint i = 0; i < HAMM_VEC_WIDTH && global_d_start + i < D_words; ++i) {
                    n_data[i] = dataset[(uint64_t)global_n_idx * D_words + global_d_start + i];
                }
            }
        }
        shared_N_u4[smem_idx] = n_data;
        
        // Synchronize after loading
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Hamming distance using XOR and popcount
        #pragma unroll
        for (uint k = 0; k < HAMM_TILE_D_VEC; ++k) {
            // Load 128 bits from each vector
            uint4 q_val = shared_Q_u4[tid.y * HAMM_TILE_D_VEC + k];
            uint4 n_val = shared_N_u4[tid.x * HAMM_TILE_D_VEC + k];
            
            // XOR to find differing bits
            uint4 xor_result = q_val ^ n_val;
            
            // Count differing bits using population count
            uint4 counts = popcount(xor_result);
            
            // Accumulate bit differences
            hamming_dist += counts.x + counts.y + counts.z + counts.w;
        }
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result to global memory
    if (gid.y < Q && gid.x < N) {
        distances[(uint64_t)gid.y * N + gid.x] = hamming_dist;
    }
}

// =============================================================================
// Float Vector Variants
// =============================================================================

/// Hamming distance for float vectors (binarized with threshold)
/// Treats values > threshold as 1, <= threshold as 0
kernel void hamming_distance_float(
    device const float* queries [[buffer(0)]],      // [Q × D]
    device const float* dataset [[buffer(1)]],      // [N × D]
    device uint* distances [[buffer(2)]],           // [Q × N]
    constant uint& Q [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& threshold [[buffer(6)]],        // Binarization threshold
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Shared memory for float vectors
    threadgroup float shared_Q[HAMM_TILE_Q * 64];  // Process 64 dims at a time
    threadgroup float shared_N[HAMM_TILE_N * 64];
    
    const uint TILE_D = 64;
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    const uint tid_linear = tid.y * HAMM_TILE_N + tid.x;
    
    uint hamming_dist = 0;
    
    // Process dimensions in tiles
    for (uint d_start = 0; d_start < D; d_start += TILE_D) {
        
        // Cooperative loading of queries
        for (uint d = tid_linear; d < HAMM_TILE_Q * TILE_D; d += HAMM_TILE_Q * HAMM_TILE_N) {
            uint q_idx = d / TILE_D;
            uint d_idx = d % TILE_D;
            uint global_q = start_q + q_idx;
            uint global_d = d_start + d_idx;
            
            if (global_q < Q && global_d < D) {
                shared_Q[d] = queries[(uint64_t)global_q * D + global_d];
            } else {
                shared_Q[d] = 0.0f;
            }
        }
        
        // Cooperative loading of dataset
        for (uint d = tid_linear; d < HAMM_TILE_N * TILE_D; d += HAMM_TILE_Q * HAMM_TILE_N) {
            uint n_idx = d / TILE_D;
            uint d_idx = d % TILE_D;
            uint global_n = start_n + n_idx;
            uint global_d = d_start + d_idx;
            
            if (global_n < N && global_d < D) {
                shared_N[d] = dataset[(uint64_t)global_n * D + global_d];
            } else {
                shared_N[d] = 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute binary Hamming distance
        uint tile_end = min(TILE_D, D - d_start);
        for (uint d = 0; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * TILE_D + d];
            float n_val = shared_N[tid.x * TILE_D + d];
            
            // Binarize and compare
            bool q_bit = (q_val > threshold);
            bool n_bit = (n_val > threshold);
            
            // Count differences
            hamming_dist += (q_bit != n_bit) ? 1 : 0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (gid.y < Q && gid.x < N) {
        distances[(uint64_t)gid.y * N + gid.x] = hamming_dist;
    }
}

// =============================================================================
// Single Vector Variants
// =============================================================================

/// Hamming distance between two bit-packed vectors
kernel void hamming_distance_single(
    device const uint* vectorA [[buffer(0)]],       // [D_words]
    device const uint* vectorB [[buffer(1)]],       // [D_words]
    device uint* distance [[buffer(2)]],            // [1]
    constant uint& D_words [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes a portion of the vectors
    const uint WORDS_PER_THREAD = 16;
    const uint start = tid * WORDS_PER_THREAD;
    
    uint local_dist = 0;
    
    // Process assigned words
    for (uint i = start; i < min(start + WORDS_PER_THREAD, D_words); ++i) {
        uint xor_result = vectorA[i] ^ vectorB[i];
        local_dist += popcount(xor_result);
    }
    
    // Atomic add to accumulate across threads
    atomic_fetch_add_explicit((device atomic_uint*)distance, local_dist, memory_order_relaxed);
}

/// Normalized Hamming distance (returns value in [0, 1])
kernel void hamming_distance_normalized(
    device const uint* queries [[buffer(0)]],
    device const uint* dataset [[buffer(1)]],
    device float* distances [[buffer(2)]],          // Float output for normalized
    constant uint& Q [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D_words [[buffer(5)]],
    constant uint& D_bits [[buffer(6)]],            // Total number of bits
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Similar structure to hamming_distance_batch
    // but divides result by D_bits at the end
    
    threadgroup uint shared_Q[HAMM_TILE_Q * HAMM_TILE_D_WORDS];
    threadgroup uint shared_N[HAMM_TILE_N * HAMM_TILE_D_WORDS];
    threadgroup uint4* shared_Q_u4 = reinterpret_cast<threadgroup uint4*>(shared_Q);
    threadgroup uint4* shared_N_u4 = reinterpret_cast<threadgroup uint4*>(shared_N);
    
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    const uint tid_linear = tid.y * HAMM_TILE_N + tid.x;
    const uint tile_row = tid_linear / HAMM_TILE_D_VEC;
    const uint tile_col_u4 = tid_linear % HAMM_TILE_D_VEC;
    const uint smem_idx = tid_linear;
    
    uint hamming_dist = 0;
    
    for (uint d_start = 0; d_start < D_words; d_start += HAMM_TILE_D_WORDS) {
        const uint global_d_start = d_start + tile_col_u4 * HAMM_VEC_WIDTH;
        uint global_q_idx = start_q + tile_row;
        uint4 q_data = uint4(0);
        
        if (global_q_idx < Q && global_d_start < D_words) {
            if (global_d_start + HAMM_VEC_WIDTH <= D_words) {
                device const uint* q_ptr = queries + (uint64_t)global_q_idx * D_words + global_d_start;
                q_data = *(reinterpret_cast<device const uint4*>(q_ptr));
            } else {
                for (uint i = 0; i < HAMM_VEC_WIDTH && global_d_start + i < D_words; ++i) {
                    q_data[i] = queries[(uint64_t)global_q_idx * D_words + global_d_start + i];
                }
            }
        }
        shared_Q_u4[smem_idx] = q_data;
        
        uint global_n_idx = start_n + tile_row;
        uint4 n_data = uint4(0);
        
        if (global_n_idx < N && global_d_start < D_words) {
            if (global_d_start + HAMM_VEC_WIDTH <= D_words) {
                device const uint* n_ptr = dataset + (uint64_t)global_n_idx * D_words + global_d_start;
                n_data = *(reinterpret_cast<device const uint4*>(n_ptr));
            } else {
                for (uint i = 0; i < HAMM_VEC_WIDTH && global_d_start + i < D_words; ++i) {
                    n_data[i] = dataset[(uint64_t)global_n_idx * D_words + global_d_start + i];
                }
            }
        }
        shared_N_u4[smem_idx] = n_data;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        #pragma unroll
        for (uint k = 0; k < HAMM_TILE_D_VEC; ++k) {
            uint4 q_val = shared_Q_u4[tid.y * HAMM_TILE_D_VEC + k];
            uint4 n_val = shared_N_u4[tid.x * HAMM_TILE_D_VEC + k];
            uint4 xor_result = q_val ^ n_val;
            uint4 counts = popcount(xor_result);
            hamming_dist += counts.x + counts.y + counts.z + counts.w;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize by total bits and write
    if (gid.y < Q && gid.x < N) {
        float normalized = float(hamming_dist) / float(D_bits);
        distances[(uint64_t)gid.y * N + gid.x] = normalized;
    }
}