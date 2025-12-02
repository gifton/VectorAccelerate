// VectorAccelerate: L2 Normalization Shaders
//
// GPU kernels for vector L2 normalization
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Parameters Structure (Spec Section: Parameters Structure)

struct L2NormParams {
    uint32_t num_vectors;
    uint32_t dimension;
    uint32_t input_stride;
    uint32_t output_stride;
    float epsilon;
    uint8_t store_norms;
    uint8_t padding[3];    // Alignment padding to match Swift struct
};

// MARK: - Helper Functions (Implementation Requirements 1 & 2)

// Helper to compute the norm squared using SIMD and handle remainders
float compute_norm_sq(device const float* vector, uint dimension) {
    float norm_sq = 0.0f;
    const uint simd_blocks = dimension / 4;

    device const float4* vec4 = (device const float4*)vector;

    // Process 4 elements at a time
    for (uint i = 0; i < simd_blocks; ++i) {
        float4 v = vec4[i];
        norm_sq += dot(v, v);
    }

    // Handle remaining elements
    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        float v = vector[i];
        norm_sq += v * v;
    }

    return norm_sq;
}

// Helper to apply normalization and write out the vector
void apply_normalization(device const float* input, device float* output, uint dimension, float norm, float epsilon) {
    // Calculate inverse norm, handling division by zero (Numerical Stability)
    float inv_norm = (norm > epsilon) ? (1.0f / norm) : 0.0f;

    const uint simd_blocks = dimension / 4;

    device const float4* in4 = (device const float4*)input;
    device float4* out4 = (device float4*)output;

    // Vectorized normalization
    for (uint i = 0; i < simd_blocks; ++i) {
        out4[i] = in4[i] * inv_norm;
    }

    // Handle remaining elements
    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        output[i] = input[i] * inv_norm;
    }
}

// MARK: - General Kernel (Spec Section: Metal Kernel Signatures)

kernel void l2_normalize_general_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* norms [[buffer(2)]],
    constant L2NormParams& params [[buffer(3)]],
    // We use uint tid for 1D dispatch (one thread per vector)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_vectors) return;

    const uint input_offset = tid * params.input_stride;
    const uint output_offset = tid * params.output_stride;

    device const float* current_input = input + input_offset;
    device float* current_output = output + output_offset;

    // Phase 1: Compute Norm
    float norm_sq = compute_norm_sq(current_input, params.dimension);
    float norm = sqrt(norm_sq);

    // Store norm if requested (Metal safely handles nullptr if the buffer wasn't bound)
    if (params.store_norms && norms != nullptr) {
        norms[tid] = norm;
    }

    // Phase 2: Normalize and Write
    apply_normalization(current_input, current_output, params.dimension, norm, params.epsilon);
}

// MARK: - In-place Kernel (Spec Section: Metal Kernel Signatures)

kernel void l2_normalize_inplace_kernel(
    device float* vectors [[buffer(0)]],
    device float* norms [[buffer(1)]],
    constant L2NormParams& params [[buffer(2)]], // Note: Params at index 2
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_vectors) return;

    // Uses input_stride (host code ensures input_stride == output_stride for in-place)
    const uint offset = tid * params.input_stride;
    device float* current_vector = vectors + offset;

    // Phase 1: Compute Norm
    float norm_sq = compute_norm_sq(current_vector, params.dimension);
    float norm = sqrt(norm_sq);

    // Store norm if requested
    if (params.store_norms && norms != nullptr) {
        norms[tid] = norm;
    }

    // Phase 2: Normalize and Write (In-place)
    apply_normalization(current_vector, current_vector, params.dimension, norm, params.epsilon);
}

// MARK: - Optimized Kernels (Spec Section: Implementation Requirements 3)

// Template function for optimized kernels using 4 accumulators and 16-element unrolling (4xfloat4)
template <uint DIMENSION>
void l2_normalize_optimized_impl(
    device const float* input,
    device float* output,
    device float* norms,
    constant L2NormParams& params,
    uint tid
) {
    if (tid >= params.num_vectors) return;

    // Optimized kernels assume stride equals dimension (dense packing, verified on host)
    const uint offset = tid * DIMENSION;

    device const float4* in4 = (device const float4*)(input + offset);
    device float4* out4 = (device float4*)(output + offset);

    // Phase 1: Compute norm with 4 accumulators for Instruction Level Parallelism (ILP)
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    constexpr uint NUM_BLOCKS = DIMENSION / 4;
    constexpr uint UNROLL_FACTOR = 4; // Unrolling 4 float4s (16 elements)

    // Process 16 elements per iteration
    for (uint i = 0; i < NUM_BLOCKS; i += UNROLL_FACTOR) {
        float4 v0 = in4[i];
        float4 v1 = in4[i+1];
        float4 v2 = in4[i+2];
        float4 v3 = in4[i+3];

        // Use Fused Multiply-Add (FMA) for better performance and precision
        acc0 = fma(v0, v0, acc0);
        acc1 = fma(v1, v1, acc1);
        acc2 = fma(v2, v2, acc2);
        acc3 = fma(v3, v3, acc3);
    }

    // Final reduction
    float4 sum = acc0 + acc1 + acc2 + acc3;
    float norm_sq = sum.x + sum.y + sum.z + sum.w;
    float norm = sqrt(norm_sq);

    // Store norm if requested
    if (params.store_norms && norms != nullptr) {
        norms[tid] = norm;
    }

    // Phase 2: Normalize
    float inv_norm = (norm > params.epsilon) ? (1.0f / norm) : 0.0f;

    for (uint i = 0; i < NUM_BLOCKS; ++i) {
        out4[i] = in4[i] * inv_norm;
    }
}

// Kernel Instantiations
kernel void l2_normalize_512_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* norms [[buffer(2)]],
    constant L2NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    l2_normalize_optimized_impl<512>(input, output, norms, params, tid);
}

kernel void l2_normalize_768_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* norms [[buffer(2)]],
    constant L2NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    l2_normalize_optimized_impl<768>(input, output, norms, params, tid);
}

kernel void l2_normalize_1536_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* norms [[buffer(2)]],
    constant L2NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    l2_normalize_optimized_impl<1536>(input, output, norms, params, tid);
}

// Note: l2_normalize_batch_kernel is omitted as efficient calculation of global statistics
// (mean/std) requires complex parallel reduction techniques, beyond the scope of this normalization kernel.