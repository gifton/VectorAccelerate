// VectorAccelerate: Minkowski Distance Kernel (Lp Norm)
//
// GPU-accelerated Minkowski distance computation
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// =============================================================================
// Configuration Constants
// =============================================================================

// Prefixed with MINK_ to avoid conflicts when shaders are combined
#define MINK_TILE_Q 16           // Query tile height
#define MINK_TILE_N 16           // Dataset tile width
#define MINK_TILE_D 64           // Dimension tile size
#define MINK_TILE_D_VEC 16       // MINK_TILE_D / 4 for float4 operations

// =============================================================================
// Helper Functions
// =============================================================================

// Nonnegative base, positive exponent. No artificial small-base cutoff or saturation.
// The batch path deliberately retains FP32 intermediate range limits (VA3-022).
// MinkowskiRangePolicyTests covers zero, tiny bases, exact powers, and overflow in both libraries.
inline float minkowski_pow(float base, float p) {
    const uint bits = as_type<uint>(base) & 0x7FFFFFFFu;
    if (bits == 0) return 0.0f;
    if (bits >= 0x7F800000u) return base; // Preserve an overflowed intermediate (or NaN).
    if (base == 1.0f || p == 0.0f) return 1.0f;
    if (p == 1.0f) return base;
    if (p == 2.0f) return base * base;
    if (p == 3.0f) return base * base * base;
    if (p == 4.0f) { float b2 = base * base; return b2 * b2; }
    return exp2(p * log2(base));
}

inline float4 pow4(float4 base, float p) {
    return float4(minkowski_pow(base.x, p), minkowski_pow(base.y, p),
                  minkowski_pow(base.z, p), minkowski_pow(base.w, p));
}

// For fractional p a ratio can underflow BEFORE its power makes it significant again.
// Compute that power in log space; direct division suffices for p >= 1.
inline float minkowski_normalized_power(float diff, float scale, float log_scale, float p) {
    if (diff == 0.0f) return 0.0f;
    if (diff == scale) return 1.0f;
    if (p < 1.0f) return exp2(p * (log2(diff) - log_scale));
    return minkowski_pow(precise::divide(diff, scale), p);
}

// =============================================================================
// Main Kernels
// =============================================================================

/// Computes Minkowski distance matrix (Lp norm)
/// Distance = (Σ|x_i - y_i|^p)^(1/p)
///
/// `chebyshev` (buffer 7) is an explicit host opt-in: nonzero switches the kernel to the
/// exact L∞ max-norm regardless of p. It exists for `Metal4MinkowskiConfig.chebyshev` /
/// the providers' `.chebyshev` metric ONLY — the kernel must never infer it from p. The
/// old `is_large_p = (p > 10)` inference silently substituted L∞ for true Lp (error up to
/// D^(1/p); AUDIT-3 VA3-007, red in MinkowskiLargePPolicyTests pre-fix). Without the flag,
/// large p takes the general power path with FP32 intermediate range limits; callers
/// wanting normalized computation select minkowski_distance_stable.
kernel void minkowski_distance_batch(
    device const float* queries [[buffer(0)]],      // [Q × D]
    device const float* dataset [[buffer(1)]],      // [N × D]
    device float* distances [[buffer(2)]],          // [Q × N]
    constant float& p [[buffer(3)]],                // Minkowski parameter
    constant uint& Q [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& chebyshev [[buffer(7)]],         // nonzero: exact L∞ (explicit opt-in)
    uint2 gid [[thread_position_in_grid]],         // (n_idx, q_idx)
    uint2 tid [[thread_position_in_threadgroup]]   // (local_n, local_q)
) {
    // Shared memory for tiled computation
    threadgroup float shared_Q[MINK_TILE_Q * MINK_TILE_D];
    threadgroup float shared_N[MINK_TILE_N * MINK_TILE_D];
    
    // Vectorized access for better performance
    threadgroup packed_float4* shared_Q_f4 = reinterpret_cast<threadgroup packed_float4*>(shared_Q);
    threadgroup packed_float4* shared_N_f4 = reinterpret_cast<threadgroup packed_float4*>(shared_N);
    
    // Calculate tile boundaries
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;
    
    // Linear thread ID for cooperative loading
    const uint tid_linear = tid.y * MINK_TILE_N + tid.x;
    const uint tile_row = tid_linear / MINK_TILE_D_VEC;
    const uint tile_col_f4 = tid_linear % MINK_TILE_D_VEC;
    
    // Initialize accumulator based on p value
    float accumulator = 0.0f;
    
    // Special handling for common cases. The Chebyshev flag is authoritative: when the
    // host opted in, p plays no role in path selection.
    const bool is_chebyshev = (chebyshev != 0);
    const bool is_manhattan = !is_chebyshev && (p == 1.0f);
    const bool is_euclidean = !is_chebyshev && (p == 2.0f);

    // For the opted-in Chebyshev (L∞) path
    float max_diff = 0.0f;
    
    // Process dimensions in tiles
    for (uint d_start = 0; d_start < D; d_start += MINK_TILE_D) {
        
        // Cooperative loading of query vectors
        const uint global_d_start = d_start + tile_col_f4 * 4;
        const uint global_q_idx = start_q + tile_row;
        float4 q_data = float4(0.0f);
        
        if (global_q_idx < Q && global_d_start < D) {
            const device float* q_ptr = queries + (uint64_t)global_q_idx * D + global_d_start;
            
            if (global_d_start + 4 <= D) {
                // Complete block: packed load permits scalar-aligned rows
                q_data = *(reinterpret_cast<const device packed_float4*>(q_ptr));
            } else {
                // Slow path: partial load
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    q_data[i] = q_ptr[i];
                }
            }
        }
        
        if (tile_row < MINK_TILE_Q && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_Q_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = q_data;
        }
        
        // Cooperative loading of dataset vectors
        const uint global_n_idx = start_n + tile_row;
        float4 n_data = float4(0.0f);
        
        if (global_n_idx < N && global_d_start < D) {
            const device float* n_ptr = dataset + (uint64_t)global_n_idx * D + global_d_start;
            
            if (global_d_start + 4 <= D) {
                // Complete block: packed load permits scalar-aligned rows
                n_data = *(reinterpret_cast<const device packed_float4*>(n_ptr));
            } else {
                // Slow path: partial load
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    n_data[i] = n_ptr[i];
                }
            }
        }
        
        if (tile_row < MINK_TILE_N && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_N_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = n_data;
        }
        
        // Synchronize after loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Minkowski distance for this tile
        const uint tile_end = min(uint(MINK_TILE_D), D - d_start);
        
        if (is_manhattan) {
            // Special case: p = 1 (Manhattan distance)
            const uint vec_end = tile_end / 4;
            for (uint k = 0; k < vec_end; ++k) {
                float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
                float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
                float4 diff = abs(q_val - n_val);
                accumulator += diff.x + diff.y + diff.z + diff.w;
            }
            
            // Handle remainder
            for (uint d = vec_end * 4; d < tile_end; ++d) {
                float q_val = shared_Q[tid.y * MINK_TILE_D + d];
                float n_val = shared_N[tid.x * MINK_TILE_D + d];
                accumulator += abs(q_val - n_val);
            }
        }
        else if (is_euclidean) {
            // Special case: p = 2 (Euclidean distance)
            const uint vec_end = tile_end / 4;
            for (uint k = 0; k < vec_end; ++k) {
                float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
                float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
                float4 diff = q_val - n_val;
                // Use FMA for better precision
                accumulator = fma(diff.x, diff.x, accumulator);
                accumulator = fma(diff.y, diff.y, accumulator);
                accumulator = fma(diff.z, diff.z, accumulator);
                accumulator = fma(diff.w, diff.w, accumulator);
            }
            
            // Handle remainder
            for (uint d = vec_end * 4; d < tile_end; ++d) {
                float q_val = shared_Q[tid.y * MINK_TILE_D + d];
                float n_val = shared_N[tid.x * MINK_TILE_D + d];
                float diff = q_val - n_val;
                accumulator = fma(diff, diff, accumulator);
            }
        }
        else if (is_chebyshev) {
            // Opted-in exact L∞: running max over |diff|
            const uint vec_end = tile_end / 4;
            for (uint k = 0; k < vec_end; ++k) {
                float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
                float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
                float4 diff = abs(q_val - n_val);
                float local_max = max(max(diff.x, diff.y), max(diff.z, diff.w));
                max_diff = max(max_diff, local_max);
            }
            
            for (uint d = vec_end * 4; d < tile_end; ++d) {
                float q_val = shared_Q[tid.y * MINK_TILE_D + d];
                float n_val = shared_N[tid.x * MINK_TILE_D + d];
                max_diff = max(max_diff, abs(q_val - n_val));
            }
        }
        else {
            // General case: arbitrary p
            const uint vec_end = tile_end / 4;
            for (uint k = 0; k < vec_end; ++k) {
                float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
                float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
                float4 diff = abs(q_val - n_val);
                float4 powered = pow4(diff, p);
                accumulator += powered.x + powered.y + powered.z + powered.w;
            }
            
            // Handle remainder
            for (uint d = vec_end * 4; d < tile_end; ++d) {
                float q_val = shared_Q[tid.y * MINK_TILE_D + d];
                float n_val = shared_N[tid.x * MINK_TILE_D + d];
                float diff = abs(q_val - n_val);
                accumulator += minkowski_pow(diff, p);
            }
        }
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Final computation and write
    if (gid.y < Q && gid.x < N) {
        float result;
        
        if (is_manhattan) {
            result = accumulator;  // No need for root
        }
        else if (is_euclidean) {
            result = sqrt(accumulator);
        }
        else if (is_chebyshev) {
            result = max_diff;  // exact L∞ (opted in)
        }
        else {
            // General case: take p-th root
            float inv_p = 1.0f / p;
            result = minkowski_pow(accumulator, inv_p);
        }
        
        distances[(uint64_t)gid.y * N + gid.x] = result;
    }
}

// =============================================================================
// Optimized Variants
// =============================================================================

/// Minkowski distance with numerical stability for large p
///
/// Uses two-pass normalization to prevent accumulation overflow:
/// 1. First pass: Find max|x_i - y_i| across all dimensions
/// 2. Second pass: Compute sum((|x_i - y_i| / max)^p)
/// 3. Final: max * sum^(1/p) = original Minkowski distance
///
/// Normalized powered terms are in [0, 1], and their sum is at most D.
/// Fractional powers use log differences and logarithmic final rescaling to avoid
/// losing terms or overflowing the root before the final result is formed.
kernel void minkowski_distance_stable(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant float& p [[buffer(3)]],
    constant uint& Q [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Shared memory for tiled computation
    threadgroup float shared_Q[MINK_TILE_Q * MINK_TILE_D];
    threadgroup float shared_N[MINK_TILE_N * MINK_TILE_D];

    // Vectorized access for better performance
    threadgroup packed_float4* shared_Q_f4 = reinterpret_cast<threadgroup packed_float4*>(shared_Q);
    threadgroup packed_float4* shared_N_f4 = reinterpret_cast<threadgroup packed_float4*>(shared_N);

    // Calculate tile boundaries
    const uint start_q = gid.y - tid.y;
    const uint start_n = gid.x - tid.x;

    // Linear thread ID for cooperative loading
    const uint tid_linear = tid.y * MINK_TILE_N + tid.x;
    const uint tile_row = tid_linear / MINK_TILE_D_VEC;
    const uint tile_col_f4 = tid_linear % MINK_TILE_D_VEC;

    float max_diff = 0.0f;
    float sum_normalized = 0.0f;

    // ==========================================================================
    // First pass: find maximum difference across all dimensions
    // ==========================================================================
    for (uint d_start = 0; d_start < D; d_start += MINK_TILE_D) {

        // Cooperative loading of query vectors
        const uint global_d_start = d_start + tile_col_f4 * 4;
        const uint global_q_idx = start_q + tile_row;
        float4 q_data = float4(0.0f);

        if (global_q_idx < Q && global_d_start < D) {
            const device float* q_ptr = queries + (uint64_t)global_q_idx * D + global_d_start;

            if (global_d_start + 4 <= D) {
                q_data = *(reinterpret_cast<const device packed_float4*>(q_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    q_data[i] = q_ptr[i];
                }
            }
        }

        if (tile_row < MINK_TILE_Q && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_Q_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = q_data;
        }

        // Cooperative loading of dataset vectors
        const uint global_n_idx = start_n + tile_row;
        float4 n_data = float4(0.0f);

        if (global_n_idx < N && global_d_start < D) {
            const device float* n_ptr = dataset + (uint64_t)global_n_idx * D + global_d_start;

            if (global_d_start + 4 <= D) {
                n_data = *(reinterpret_cast<const device packed_float4*>(n_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    n_data[i] = n_ptr[i];
                }
            }
        }

        if (tile_row < MINK_TILE_N && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_N_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = n_data;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find max difference in this tile using vectorized operations
        const uint tile_end = min(uint(MINK_TILE_D), D - d_start);
        const uint vec_end = tile_end / 4;

        for (uint k = 0; k < vec_end; ++k) {
            float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
            float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
            float4 diff = abs(q_val - n_val);
            float local_max = max(max(diff.x, diff.y), max(diff.z, diff.w));
            max_diff = max(max_diff, local_max);
        }

        // Handle remainder
        for (uint d = vec_end * 4; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * MINK_TILE_D + d];
            float n_val = shared_N[tid.x * MINK_TILE_D + d];
            max_diff = max(max_diff, abs(q_val - n_val));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Per-pair state cannot cause an early return: all lanes must reach pass-2 barriers.
    // Finite input subtraction can itself overflow; then the true distance also exceeds FP32.
    const bool degenerate = (max_diff == 0.0f);
    const bool overflowed_difference = (max_diff > FLT_MAX);
    const float scale = (degenerate || overflowed_difference) ? 1.0f : max_diff;
    const float log_scale = p < 1.0f ? log2(scale) : 0.0f;

    // ==========================================================================
    // Second pass: compute normalized sum with all values in [0, 1]
    // ==========================================================================
    for (uint d_start = 0; d_start < D; d_start += MINK_TILE_D) {

        // Cooperative loading of query vectors (same pattern as first pass)
        const uint global_d_start = d_start + tile_col_f4 * 4;
        const uint global_q_idx = start_q + tile_row;
        float4 q_data = float4(0.0f);

        if (global_q_idx < Q && global_d_start < D) {
            const device float* q_ptr = queries + (uint64_t)global_q_idx * D + global_d_start;

            if (global_d_start + 4 <= D) {
                q_data = *(reinterpret_cast<const device packed_float4*>(q_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    q_data[i] = q_ptr[i];
                }
            }
        }

        if (tile_row < MINK_TILE_Q && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_Q_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = q_data;
        }

        // Cooperative loading of dataset vectors
        const uint global_n_idx = start_n + tile_row;
        float4 n_data = float4(0.0f);

        if (global_n_idx < N && global_d_start < D) {
            const device float* n_ptr = dataset + (uint64_t)global_n_idx * D + global_d_start;

            if (global_d_start + 4 <= D) {
                n_data = *(reinterpret_cast<const device packed_float4*>(n_ptr));
            } else {
                for (uint i = 0; i < 4 && global_d_start + i < D; ++i) {
                    n_data[i] = n_ptr[i];
                }
            }
        }

        if (tile_row < MINK_TILE_N && tile_col_f4 < MINK_TILE_D_VEC) {
            shared_N_f4[tile_row * MINK_TILE_D_VEC + tile_col_f4] = n_data;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute normalized powered differences
        const uint tile_end = min(uint(MINK_TILE_D), D - d_start);
        const uint vec_end = tile_end / 4;

        for (uint k = 0; k < vec_end; ++k) {
            float4 q_val = shared_Q_f4[tid.y * MINK_TILE_D_VEC + k];
            float4 n_val = shared_N_f4[tid.x * MINK_TILE_D_VEC + k];
            float4 diff = abs(q_val - n_val);

            sum_normalized += minkowski_normalized_power(diff.x, scale, log_scale, p);
            sum_normalized += minkowski_normalized_power(diff.y, scale, log_scale, p);
            sum_normalized += minkowski_normalized_power(diff.z, scale, log_scale, p);
            sum_normalized += minkowski_normalized_power(diff.w, scale, log_scale, p);
        }

        // Handle remainder
        for (uint d = vec_end * 4; d < tile_end; ++d) {
            float q_val = shared_Q[tid.y * MINK_TILE_D + d];
            float n_val = shared_N[tid.x * MINK_TILE_D + d];
            sum_normalized += minkowski_normalized_power(abs(q_val - n_val), scale, log_scale, p);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final computation: denormalize by multiplying by max_diff
    // result = max_diff * (sum_normalized)^(1/p)
    if (gid.y < Q && gid.x < N) {
        float result;
        if (degenerate) result = 0.0f;
        else if (overflowed_difference) result = INFINITY;
        else if (sum_normalized == 1.0f) result = max_diff;
        else if (p < 1.0f) {
            // Preserve scale's significand: log2(scale) can round away the gap below
            // FLT_MAX after rescaling. Split the root's binary exponent instead of
            // materializing an overflowing factor (MinkowskiRangePolicyTests).
            const float root_exponent = precise::divide(log2(sum_normalized), p);
            // Any positive FP32 scale times 2^512 exceeds FP32. Guard before int conversion.
            if (root_exponent > 512.0f) result = INFINITY;
            else {
                int scale_exponent;
                const float mantissa = frexp(max_diff, scale_exponent);
                const int whole = int(floor(root_exponent));
                const float fraction = root_exponent - float(whole);
                result = ldexp(mantissa * exp2(fraction), scale_exponent + whole);
            }
        } else {
            // p >= 1 bounds the root by D; divide directly instead of multiplying by 1/p.
            result = max_diff * exp2(precise::divide(log2(sum_normalized), p));
        }
        distances[(uint64_t)gid.y * N + gid.x] = result;
    }
}

