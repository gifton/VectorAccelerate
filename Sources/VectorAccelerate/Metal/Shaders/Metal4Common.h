// VectorAccelerate: Metal 4 Common Header
//
// MSL 4.0 compatibility header for VectorAccelerate compute kernels
// This header provides:
// - Version detection and feature guards
// - Common constants and types
// - Metal 4 specific utilities
// - Backward compatibility shims
//
// Usage: Include at the top of all .metal shader files
//
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
// Minimum MSL Version: 3.1 (Metal 4 SDK)
//

#ifndef VECTORACCELERATE_METAL4_COMMON_H
#define VECTORACCELERATE_METAL4_COMMON_H

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>

// Optional: Include metal_tensor for ML tensor operations (Metal 4 feature)
// Uncomment when targeting Metal 4 exclusively and using tensor ops
// #include <metal_tensor>

using namespace metal;

// =============================================================================
// MARK: - Version Detection
// =============================================================================

// MSL version detection
// MSL 4.0 corresponds to __METAL_VERSION__ >= 400 (Metal 4 SDK)
// MSL 3.1 corresponds to __METAL_VERSION__ >= 310
// MSL 3.0 corresponds to __METAL_VERSION__ >= 300

#if __METAL_VERSION__ >= 400
    #define VA_MSL_4_0 1
    #define VA_METAL_4_AVAILABLE 1
#else
    #define VA_MSL_4_0 0
    #define VA_METAL_4_AVAILABLE 0
#endif

#if __METAL_VERSION__ >= 310
    #define VA_MSL_3_1 1
#else
    #define VA_MSL_3_1 0
#endif

// =============================================================================
// MARK: - Common Constants
// =============================================================================

// Numerical stability constants
constant float VA_EPSILON = 1e-7f;
constant float VA_EPSILON_HALF = 1e-4h;
constant float VA_INFINITY = INFINITY;

// Sentinel values for invalid indices
constant uint VA_INVALID_INDEX = 0xFFFFFFFF;
constant uint VA_SENTINEL_INDEX = 0xFFFFFFFF;

// Thread configuration limits
constant uint VA_MAX_THREADGROUP_SIZE = 1024;
constant uint VA_PREFERRED_THREADGROUP_SIZE = 256;
constant uint VA_SIMD_WIDTH = 32;  // Apple Silicon SIMD group width

// Vector dimension presets for embedding models
constant uint VA_DIM_MINILM = 384;      // MiniLM, all-MiniLM-L6-v2
constant uint VA_DIM_BERT_SMALL = 512;  // Small BERT variants
constant uint VA_DIM_BERT = 768;        // BERT-base, DistilBERT, MPNet
constant uint VA_DIM_OPENAI = 1536;     // OpenAI ada-002
constant uint VA_DIM_OPENAI_3 = 3072;   // OpenAI text-embedding-3-large

// =============================================================================
// MARK: - Common Structures
// =============================================================================

// Index-distance pair for top-k selection
struct VAIndexDistance {
    uint index;
    float distance;
};

// Candidate structure for sorting/selection
struct VACandidate {
    float distance;
    uint index;
};

// Vector parameters for kernel configuration
struct VAVectorParams {
    uint32_t numVectors;     // Number of vectors
    uint32_t dimension;      // Vector dimension
    uint32_t stride;         // Stride between vectors (0 = dense packing)
    uint32_t padding;        // Alignment padding
};

// Batch distance parameters
struct VABatchDistanceParams {
    uint32_t numQueries;      // Number of query vectors (Q)
    uint32_t numDatabase;     // Number of database vectors (N)
    uint32_t dimension;       // Vector dimension (D)
    uint32_t strideQuery;     // Stride between query vectors
    uint32_t strideDatabase;  // Stride between database vectors
    uint32_t strideOutput;    // Stride for output matrix
    uint8_t computeSqrt;      // 0 = squared distance, 1 = apply sqrt
    uint8_t padding[3];       // Alignment padding
};

// =============================================================================
// MARK: - Helper Functions
// =============================================================================

// Safe float4 load with bounds checking
inline float4 va_safe_load_float4(device const float* base, uint offset, uint max_elements) {
    if (offset + 3 < max_elements) {
        return reinterpret_cast<device const float4*>(base + offset)[0];
    }
    float4 result = float4(0.0f);
    for (uint i = 0; i < 4 && offset + i < max_elements; ++i) {
        result[i] = base[offset + i];
    }
    return result;
}

// Safe threadgroup float4 load
inline float4 va_safe_load_float4_tg(threadgroup const float* base, uint offset, uint max_elements) {
    if (offset + 3 < max_elements) {
        return reinterpret_cast<threadgroup const float4*>(base + offset)[0];
    }
    float4 result = float4(0.0f);
    for (uint i = 0; i < 4 && offset + i < max_elements; ++i) {
        result[i] = base[offset + i];
    }
    return result;
}

// Candidate comparison (ascending by distance, then by index for stability)
inline bool va_candidate_is_better(VACandidate a, VACandidate b) {
    if (a.distance < b.distance) return true;
    if (a.distance > b.distance) return false;
    return a.index < b.index;
}

// Index-distance comparison
inline bool va_index_distance_is_better_asc(VAIndexDistance a, VAIndexDistance b) {
    if (a.distance < b.distance) return true;
    if (a.distance > b.distance) return false;
    return a.index < b.index;
}

inline bool va_index_distance_is_better_desc(VAIndexDistance a, VAIndexDistance b) {
    if (a.distance > b.distance) return true;
    if (a.distance < b.distance) return false;
    return a.index < b.index;
}

// =============================================================================
// MARK: - Reduction Utilities
// =============================================================================

// SIMD group reduction for sum (requires metal_simdgroup)
inline float va_simd_sum(float value) {
    return simd_sum(value);
}

// SIMD group reduction for minimum
inline float va_simd_min(float value) {
    return simd_min(value);
}

// SIMD group reduction for maximum
inline float va_simd_max(float value) {
    return simd_max(value);
}

// SIMD group prefix sum (scan)
inline float va_simd_prefix_sum(float value) {
    return simd_prefix_exclusive_sum(value);
}

// =============================================================================
// MARK: - Vectorized Distance Helpers
// =============================================================================

// Compute L2 squared distance using float4 vectorization
inline float va_l2_squared_vectorized(
    device const float* vec_a,
    device const float* vec_b,
    uint dimension
) {
    float4 acc = float4(0.0f);

    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* a4 = reinterpret_cast<device const float4*>(vec_a);
    device const float4* b4 = reinterpret_cast<device const float4*>(vec_b);

    for (uint i = 0; i < simd_blocks; ++i) {
        float4 diff = a4[i] - b4[i];
        acc = fma(diff, diff, acc);
    }

    float sum = acc.x + acc.y + acc.z + acc.w;

    // Handle remainder
    if (remainder > 0) {
        device const float* a_tail = vec_a + (simd_blocks * 4);
        device const float* b_tail = vec_b + (simd_blocks * 4);
        for (uint i = 0; i < remainder; ++i) {
            float diff = a_tail[i] - b_tail[i];
            sum = fma(diff, diff, sum);
        }
    }

    return sum;
}

// Compute dot product using float4 vectorization
inline float va_dot_product_vectorized(
    device const float* vec_a,
    device const float* vec_b,
    uint dimension
) {
    float4 acc = float4(0.0f);

    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* a4 = reinterpret_cast<device const float4*>(vec_a);
    device const float4* b4 = reinterpret_cast<device const float4*>(vec_b);

    for (uint i = 0; i < simd_blocks; ++i) {
        acc = fma(a4[i], b4[i], acc);
    }

    float sum = acc.x + acc.y + acc.z + acc.w;

    // Handle remainder
    if (remainder > 0) {
        device const float* a_tail = vec_a + (simd_blocks * 4);
        device const float* b_tail = vec_b + (simd_blocks * 4);
        for (uint i = 0; i < remainder; ++i) {
            sum = fma(a_tail[i], b_tail[i], sum);
        }
    }

    return sum;
}

// =============================================================================
// MARK: - Metal 4 Feature Guards
// =============================================================================

// Use these macros to guard Metal 4 specific features
// Example:
// #if VA_METAL_4_AVAILABLE
//     // Metal 4 optimized path
//     use_argument_table(...)
// #else
//     // Metal 3 fallback path
//     setBuffer(...)
// #endif

// Barrier helper for unified encoder (Metal 4)
// In Metal 4, barriers can specify resource dependencies
// For Metal 3 compatibility, we use standard threadgroup barriers
#define VA_THREADGROUP_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define VA_DEVICE_BARRIER() threadgroup_barrier(mem_flags::mem_device)
#define VA_FULL_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)

// =============================================================================
// MARK: - Debug Utilities (Optional)
// =============================================================================

#ifdef VA_DEBUG_ENABLED
    // Debug output helpers (only for development builds)
    #define VA_DEBUG_ASSERT(condition) \
        if (!(condition)) { /* trigger debug break or log */ }
#else
    #define VA_DEBUG_ASSERT(condition) ((void)0)
#endif

#endif // VECTORACCELERATE_METAL4_COMMON_H
