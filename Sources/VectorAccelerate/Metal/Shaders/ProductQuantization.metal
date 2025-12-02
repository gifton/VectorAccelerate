// VectorAccelerate: Product Quantization Shaders
//
// GPU kernels for product quantization encoding and search
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"
#include <metal_atomic>

// Define atomic types required for training
// Wrapped in guards to avoid redefinition in combined compilation
#ifndef VA_ATOMIC_TYPES_DEFINED
#define VA_ATOMIC_TYPES_DEFINED
typedef atomic<float> atomic_float;
typedef atomic<uint> atomic_uint;
#endif

// MARK: - Common Structures and Parameters

// Configuration structure
struct PQConfig {
    uint32_t N;      // Number of vectors (N_train or N_database)
    uint32_t D;      // Full dimension
    uint32_t M;      // Number of subspaces
    uint32_t K;      // Centroids per subspace (e.g., 256)
    uint32_t D_sub;  // Dimensions per subspace (D/M)
};

// MARK: - Helper Functions

// Helper function for calculating L2 distance squared (Vectorized)
float calculate_l2_sq_dist(device const float* vec, device const float* centroid, uint D_sub) {
    float dist = 0.0f;
    // Vectorized distance calculation (float4)
    uint simd_blocks = D_sub / 4;
    device const float4* vec4 = (device const float4*)vec;
    device const float4* cent4 = (device const float4*)centroid;

    float4 dist_acc = 0.0f;

    for (uint d = 0; d < simd_blocks; ++d) {
        float4 diff = vec4[d] - cent4[d];
        // Use Fused Multiply-Add (FMA) for better precision and speed
        dist_acc = fma(diff, diff, dist_acc);
    }
    // Horizontal sum
    dist = dist_acc.x + dist_acc.y + dist_acc.z + dist_acc.w;

    // Handle remainder (if D_sub is not divisible by 4)
    for (uint d = simd_blocks * 4; d < D_sub; ++d) {
        float diff = vec[d] - centroid[d];
        dist += diff * diff;
    }
    return dist;
}

// MARK: - Phase 1 & 2: Assignment and Encoding

// This kernel serves both as the Encoding kernel (Phase 2) and the Assignment step (E-step) for Training (Phase 1).
kernel void pq_assignment_or_encoding(
    device const float* vectors [[buffer(0)]],      // [N × D]
    device const float* codebooks [[buffer(1)]],    // [M × K × D_sub]
    device uint8_t* assignments_or_codes [[buffer(2)]], // [N × M] (Assuming K<=256)
    constant PQConfig& config [[buffer(3)]],
    // 2D grid (N, M): Highly parallelized dispatch.
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vec_id = tid.x;
    const uint m = tid.y; // Subspace ID

    if (vec_id >= config.N || m >= config.M) return;

    const uint D_sub = config.D_sub;
    const uint K = config.K;

    // Pointers
    device const float* vec_sub = vectors + vec_id * config.D + m * D_sub;
    device const float* codebook_m = codebooks + m * K * D_sub;

    float min_dist = INFINITY;
    uint8_t best_centroid = 0;

    // Find nearest centroid K
    for (uint k = 0; k < K; ++k) {
        device const float* centroid_k = codebook_m + k * D_sub;
        float dist = calculate_l2_sq_dist(vec_sub, centroid_k, D_sub);

        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = (uint8_t)k;
        }
    }

    // Store the assignment/code
    assignments_or_codes[vec_id * config.M + m] = best_centroid;
}

// MARK: - Phase 1: Training Update Step (M-step)

// Kernel 1.2: Update Step Accumulation
// Accumulates vectors assigned to each centroid using device memory atomics.
kernel void pq_train_update_accumulate(
    device const float* training_data [[buffer(0)]],
    device const uint8_t* assignments [[buffer(1)]],
    // Accumulators must be zeroed out by the host before this call.
    device atomic_float* centroids_accum [[buffer(2)]], // [M × K × D_sub]
    device atomic_uint* centroid_counts [[buffer(3)]],  // [M × K]
    constant PQConfig& config [[buffer(4)]],
    // 2D grid (N_train, M): Highly parallelized dispatch.
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vec_id = tid.x;
    const uint m = tid.y;

    if (vec_id >= config.N || m >= config.M) return;

    const uint D_sub = config.D_sub;
    const uint K = config.K;

    // Get the assignment
    uint8_t assignment = assignments[vec_id * config.M + m];

    // Pointers
    device const float* vec_sub = training_data + vec_id * config.D + m * D_sub;
    device atomic_float* accum_target = centroids_accum + (m * K + assignment) * D_sub;

    // Accumulate vector values atomically
    for (uint d = 0; d < D_sub; ++d) {
        float val = vec_sub[d];
        // Requires hardware support for float atomics (e.g., Apple Silicon GPUs)
        atomic_fetch_add_explicit(&accum_target[d], val, memory_order_relaxed);
    }

    // Increment count atomically
    uint count_idx = m * K + assignment;
    atomic_fetch_add_explicit(&centroid_counts[count_idx], 1u, memory_order_relaxed);
}

// Kernel 1.3: Update Step Finalization
// Computes the new centroids and calculates movement (convergence).
kernel void pq_train_update_finalize(
    device float* codebooks [[buffer(0)]],                   // Input: Old, Output: New
    device const atomic_float* centroids_accum [[buffer(1)]],
    device const atomic_uint* centroid_counts [[buffer(2)]],
    // Stores movement per centroid for convergence check [M * K]
    device float* convergence_out [[buffer(3)]],
    constant PQConfig& config [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]] // 2D grid: (K, M)
) {
    const uint k = tid.x;
    const uint m = tid.y;

    if (k >= config.K || m >= config.M) return;

    const uint D_sub = config.D_sub;
    const uint count_idx = m * config.K + k;
    uint count = atomic_load_explicit(&centroid_counts[count_idx], memory_order_relaxed);

    device float* centroid_target = codebooks + count_idx * D_sub;
    device const atomic_float* accum_source = centroids_accum + count_idx * D_sub;

    float movement_sq = 0.0f;

    if (count > 0) {
        float inv_count = 1.0f / float(count);
        for (uint d = 0; d < D_sub; ++d) {
            float sum = atomic_load_explicit(&accum_source[d], memory_order_relaxed);
            float new_val = sum * inv_count;
            float old_val = centroid_target[d];
            
            // Update the codebook
            centroid_target[d] = new_val;
            
            // Calculate movement
            float diff = old_val - new_val;
            movement_sq += diff * diff;
        }
    }
    // Note: Empty clusters (count == 0) retain their old position. Robust implementations might re-seed them.

    // Store the movement for convergence checking on the host.
    if (convergence_out != nullptr) {
        convergence_out[count_idx] = movement_sq;
    }
}

// MARK: - Phase 3: Distance Computation (ADC)

// Kernel 3.1: Precompute Distance Table (for a single query)
kernel void pq_precompute_distance_table(
    device const float* query [[buffer(0)]],          // [D]
    device const float* codebooks [[buffer(1)]],      // [M × K × D_sub]
    device float* distance_table [[buffer(2)]],       // [M × K]
    constant PQConfig& config [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]] // 2D grid: (K, M)
) {
    const uint k = tid.x;
    const uint m = tid.y;

    if (k >= config.K || m >= config.M) return;

    // Pointers
    device const float* query_sub = query + m * config.D_sub;
    device const float* centroid_k = codebooks + (m * config.K + k) * config.D_sub;

    // Calculate L2 squared distance
    float dist = calculate_l2_sq_dist(query_sub, centroid_k, config.D_sub);

    // Store the precomputed squared L2 distance
    distance_table[m * config.K + k] = dist;
}

// Kernel 3.2: Compute Distances via Lookup (Optimized with Threadgroup Memory)
kernel void pq_compute_distances_adc(
    device const uint8_t* codes [[buffer(0)]],              // [N × M]
    device const float* distance_table [[buffer(1)]],       // [M × K]
    device float* distances [[buffer(2)]],                  // [N]
    constant PQConfig& config [[buffer(3)]],
    // Optimization: Use threadgroup memory for the distance table lookup.
    threadgroup float* shared_dist_table [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],                   // 1D grid: (N)
    uint local_id [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    const uint M = config.M;
    const uint K = config.K;
    const uint MK = M * K;

    // 1. Cooperatively load the distance table into threadgroup memory
    // This drastically reduces global memory reads during the scan phase.
    for(uint i = local_id; i < MK; i += threadgroup_size) {
        shared_dist_table[i] = distance_table[i];
    }
    // Ensure all threads have finished loading
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute the distance for the assigned vector
    if (tid >= config.N) return;

    device const uint8_t* vector_codes = codes + tid * M;
    float approx_dist_sq = 0.0f;

    // Accumulate distances by looking up the codes in the shared distance table
    // Unrolling this loop improves performance as M is typically small (e.g., 8, 16).
    #pragma unroll
    for (uint m = 0; m < M; ++m) {
        uint8_t code = vector_codes[m];
        // Access pattern in shared memory: [m * K + code]
        approx_dist_sq += shared_dist_table[m * K + code];
    }

    // Store the result (Squared L2 distance).
    distances[tid] = approx_dist_sq;
}