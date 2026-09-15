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

// Numerical stability constants.
// VA_EPSILON is mirrored as a #define in KernelContext.runtimeCompilePreamble (the combined
// runtime build strips this header); the #ifndef lets that macro win if both are ever seen,
// matching the VA_NORM_* pattern below. PreambleParityTests guards numeric identity.
#ifndef VA_EPSILON
constant float VA_EPSILON = 1e-7f;
#endif
constant float VA_EPSILON_HALF = 1e-4h;
constant float VA_INFINITY = INFINITY;

// -----------------------------------------------------------------------------
// Normalization policy constants (BE3 §4.4 — CPU parity with VectorCore's
// NormalizeKernels). Every VectorAccelerate normalize kernel pre-scales by
// 1 / clamp(maxAbs, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM) before accumulating
// squares, and forms the output in that pre-scaled domain.
//
// !! DUPLICATED IN Sources/VectorAccelerate/Core/KernelContext.swift !!
// `compileMetalSourcesFromBundle` strips `#include "Metal4Common.h"` and prepends
// its own preamble, so the runtime-compilation load path (the ONLY path in release
// builds, where debug.metallib is not loaded) needs these values as #defines
// there. The `#ifndef` guards below let those macros win when this header is
// nonetheless included; the two definitions MUST stay numerically identical.
// -----------------------------------------------------------------------------

// Smallest positive *normal* float (0x1p-126f == FLT_MIN == Swift's
// Float.leastNormalMagnitude). Deliberately NOT the denormal minimum: it is the
// lower clamp that keeps `scale = 1/den` finite (<= 2^126) for subnormal-dominated
// vectors.
#ifndef VA_NORM_MIN_DENOM
constant float VA_NORM_MIN_DENOM = 0x1p-126f;
#endif

// Largest pre-scale denominator (0x1p126f). Upper clamp: without it, a vector
// whose maxAbs exceeds 2^126 yields a *subnormal* `scale`, which a GPU running
// with denormals-are-zero (Metal's default math mode) flushes to 0 — the whole
// sum of squares then collapses to 0 and the vector is misclassified as
// degenerate. Clamping costs nothing: with den == 2^126 the largest scaled
// component is maxAbs/2^126 <= FLT_MAX/2^126 < 4, so the sum of squares is still
// bounded by 16·dimension.
#ifndef VA_NORM_MAX_DENOM
constant float VA_NORM_MAX_DENOM = 0x1p126f;
#endif

// Guard on the pre-scaled norm sNorm = ||v|| / den, i.e. on ||v|| >= 2^-127.
//
// The exact representability condition for 1/||v|| in FP32 is ||v|| > 2^-128
// (= 1/FLT_MAX), which for den == 2^-126 is `sNorm > 0.25`. This constant is one
// binade stricter, keeping every operand of the final division comfortably normal,
// so the output is identical whether or not the GPU flushes denormals.
//
// Bounds, with den clamped to [2^-126, 2^126] and the guard passed:
//   scale     = 1/den ∈ [2^-126, 2^126]             (normal)
//   |v·scale| <= maxAbs/den <= 4                    (normal)
//   sumSq     <= 16 · dimension                     (no overflow)
//   sNorm     ∈ (0.5, 4·sqrt(dimension)]            (normal)
//   |out|     = |v_i| / ||v|| <= 1
// No operand of the output computation is subnormal. The output is formed as
// `precise::divide(v · scale, sNorm)` rather than a multiply by `1/sNorm`,
// because under fast math (Metal's default) the compiler may reassociate
// `(v · scale) · (1/sNorm)` into `v · (scale/sNorm)`, and that single factor IS
// subnormal whenever ||v|| > 2^126 — measured: it flushed a 4e36 vector to all
// zeros. The one value that can still leave the normal range is the *reported*
// norm `den · sNorm`, which saturates to +Inf when ||v|| exceeds FLT_MAX (the
// honest answer) and may read 0 for a subnormal ||v|| on an FTZ GPU.
//
// When den == maxAbs the largest scaled component is ±1, so sNorm >= 1 and the
// guard always passes; it can only fail for the zero vector and for vectors whose
// largest magnitude is subnormal.
#ifndef VA_NORM_MIN_SCALED
constant float VA_NORM_MIN_SCALED = 0.5f;
#endif

// Upper end of the same guard: sNorm must also be *finite*.
//
// A vector with a ±Inf component reduces to sNorm = +Inf, which passes a bare
// `sNorm > 0.5` and would then produce Inf/Inf = NaN at that lane and 0 at every
// other lane. Non-finite inputs must take the pass-through leg instead, matching
// VectorCore (whose `mag` comes out NaN and fails its own `mag > 0` guard) and
// matching this library's behavior before the upper `den` clamp existed — the
// clamp made `scale` finite, which is what let +Inf survive into the sum.
//
// Expressed as a plain magnitude comparison rather than `isfinite()`: under fast
// math (Metal's default) the compiler is permitted to assume no Inf/NaN operands
// and may fold `isfinite()` to a constant, whereas current toolchains do not fold
// a dynamic float compare (an empirical guarantee, not a spec one — the Inf
// fixtures in NormalizationParityTests fail loudly if a toolchain starts folding
// it). Any finite input satisfies sNorm <= 4·sqrt(dimension), so this
// bound (2^100 ≈ 1.27e30) is unreachable for a legitimate vector — a dimension of
// 2^196 would be needed. NaN inputs are already excluded by the lower comparison
// (`NaN > 0.5` is false).
#ifndef VA_NORM_MAX_SCALED
constant float VA_NORM_MAX_SCALED = 0x1p100f;
#endif

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
        return reinterpret_cast<device const packed_float4*>(base + offset)[0];
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
        return reinterpret_cast<threadgroup const packed_float4*>(base + offset)[0];
    }
    float4 result = float4(0.0f);
    for (uint i = 0; i < 4 && offset + i < max_elements; ++i) {
        result[i] = base[offset + i];
    }
    return result;
}

// VA3-030: rooted L2 keeps its fast accumulation, then rescues range failures.
// Squared outputs intentionally bypass this helper. EuclideanRangePolicyTests covers
// AoS/SoA layouts, both libraries, overflow, underflow, and nonfinite input controls.
inline float va_euclidean_finalize(float sum, device const float* a, device const float* b,
                                    uint dimension, ulong b_lane_stride = 4) {
    const uint sum_bits = as_type<uint>(sum);
    if (sum_bits >= 0x00800000u && sum_bits < 0x7F800000u) return sqrt(sum);

    float max_diff = 0.0f;
    bool has_nan = false;
    for (uint i = 0; i < dimension; ++i) {
        const ulong bi = (ulong)(i / 4) * b_lane_stride + (i & 3);
        const float diff = fabs(a[i] - b[bi]);
        const uint bits = as_type<uint>(diff) & 0x7FFFFFFFu;
        has_nan |= bits > 0x7F800000u;
        max_diff = max(max_diff, diff);
    }
    if (has_nan) return as_type<float>(0x7FC00000u);
    if (max_diff > FLT_MAX) return INFINITY; // Even one difference exceeds the output range.
    if (max_diff == 0.0f) return 0.0f;

    float scaled_sum = 0.0f;
    for (uint i = 0; i < dimension; ++i) {
        const ulong bi = (ulong)(i / 4) * b_lane_stride + (i & 3);
        const float normalized = precise::divide(a[i] - b[bi], max_diff);
        scaled_sum = fma(normalized, normalized, scaled_sum);
    }
    // Keep the large/tiny scale out of the squared arithmetic, including under fast-math.
    int exponent;
    const float mantissa = frexp(max_diff, exponent);
    return ldexp(mantissa * sqrt(scaled_sum), exponent);
}

// VA3-016: shared Top-K ordering. Integer NaN classification survives fast-math.
// Numeric values precede NaNs in both directions; ties (including +/-0 and NaNs)
// prefer the smaller original index. Invalid slots follow every real candidate.
// TopKNaNPolicyTests exercises admission, sorting, merging, and padding in both libraries.
inline bool va_topk_is_better(float a, uint ai, float b, uint bi, bool ascending) {
    if (ai == 0xFFFFFFFFu) return false;
    if (bi == 0xFFFFFFFFu) return true;
    const bool a_nan = (as_type<uint>(a) & 0x7FFFFFFFu) > 0x7F800000u;
    const bool b_nan = (as_type<uint>(b) & 0x7FFFFFFFu) > 0x7F800000u;
    if (a_nan != b_nan) return !a_nan;
    if (!a_nan) {
        // Order FP32 bit patterns without arithmetic: fast-math may flush subnormals.
        // Collapse signed zero, then reverse negatives and move positives above them.
        uint ab = as_type<uint>(a), bb = as_type<uint>(b);
        if ((ab & 0x7FFFFFFFu) == 0) ab = 0;
        if ((bb & 0x7FFFFFFFu) == 0) bb = 0;
        const uint ak = (ab & 0x80000000u) ? ~ab : (ab ^ 0x80000000u);
        const uint bk = (bb & 0x80000000u) ? ~bb : (bb ^ 0x80000000u);
        if (ak != bk) return ascending ? ak < bk : ak > bk;
    }
    return ai < bi;
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
// MARK: - Cosine Similarity Overflow/Underflow Rescue (AUDIT-2 VA2-008/VA2-009)
// =============================================================================
//
// Single-precision Σv² accumulation misclassifies finite vectors as degenerate outside roughly
// [1e-19, 1e19] component magnitude: squares overflow to +Inf around 1.8e19, and collapse to 0
// under the GPU's flush-to-zero handling of subnormal squares below ~1.1e-19. Every cosine
// kernel therefore finishes through this trio: detect an unreliable accumulator state, recompute
// in the pre-scaled domain of the normalization policy above (den = clamp(maxAbs,
// VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM), scale = 1/den — the per-vector scales cancel exactly in
// the similarity quotient), and finalize with a NaN-propagating, [-1, 1]-clamped division.
//
// NaN inputs do NOT trigger the rescue (isinf/== 0 are both false for NaN); they ride the
// primary accumulators into va_cosine_similarity_finalize, which propagates them. isinf()/isnan()
// under fast math are kept honest empirically by the huge/nanPoisoned classes of
// DifferentialKernelVsCPUTests, which fail loudly if a toolchain ever folds them.
//
// !! DUPLICATED IN Sources/VectorAccelerate/Core/KernelContext.swift (runtimeCompilePreamble) !!
// The runtime combined-source compile strips this header; the preamble carries a byte-identical
// copy of this guarded block (PreambleParityTests.testCosineRescueBlockIdentical enforces it).

#ifndef VA_COSINE_RESCUE_DEFINED
#define VA_COSINE_RESCUE_DEFINED

// True when the naive accumulators cannot represent the pair correctly: an overflowed (Inf)
// term, or a squared norm that collapsed to exactly 0 (zero vector — cheap to re-confirm — or
// flushed subnormal squares).
//
// Expressed as magnitude comparisons, NOT isinf(): under fast math a toolchain may fold
// isinf() to false (measured in the plugin-built metallib during the 2026-08 audit, while the
// same source runtime-compiled kept it), silently disabling the rescue. `x > FLT_MAX` is true
// exactly for +Inf, compiles as a dynamic compare, and is false for NaN — NaN deliberately
// rides the primary path into the NaN-propagating finalization.
inline bool va_cosine_accumulators_unreliable(float dotAB, float normSqA, float normSqB) {
    return fabs(dotAB) > FLT_MAX || normSqA > FLT_MAX || normSqB > FLT_MAX
        || normSqA == 0.0f || normSqB == 0.0f;
}

// Recompute (A·B, ‖A‖², ‖B‖²) in the pre-scaled domain. |a·aScale| ≤ 4 (≤ 1 when maxAbs ≤
// 2^126), so every accumulator is bounded by 16·dimension — no overflow, no subnormal collapse.
// max() drops a NaN operand, so maxAbs stays finite for NaN-poisoned vectors; the NaN itself
// still propagates through the scaled products. Cold path: serial, correctness over speed.
inline float3 va_cosine_rescaled_terms(
    device const float* a,
    device const float* b,
    uint dimension
) {
    float aMax = 0.0f;
    float bMax = 0.0f;
    for (uint i = 0; i < dimension; ++i) {
        aMax = max(aMax, fabs(a[i]));
        bMax = max(bMax, fabs(b[i]));
    }
    const float aScale = 1.0f / clamp(aMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
    const float bScale = 1.0f / clamp(bMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
    float dotAB = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    for (uint i = 0; i < dimension; ++i) {
        const float x = a[i] * aScale;
        const float y = b[i] * bScale;
        dotAB = fma(x, y, dotAB);
        aa = fma(x, x, aa);
        bb = fma(y, y, bb);
    }
    return float3(dotAB, aa, bb);
}

// Shared finalization: similarity = (dot/‖A‖)/‖B‖, NaN-propagating clamp to [-1, 1], FLT_MIN
// degenerate floor per norm (BE3 4.5: absolute, not precision-relative), zero-vector policy
// similarity = 0. Robust to fast-math comparison flips on NaN: both branches propagate NaN.
//
// The single-product denominator sqrt(aa)*sqrt(bb) is deliberately NOT formed: under fast math
// (Metal's default) the compiler reassociates it into sqrt(aa*bb), whose argument overflows to
// +Inf for |components| ≳ 1e18 and silently collapsed the similarity to 0 — measured on Apple
// silicon during the 2026-08 audit (AUDIT-2 VA2-008). The divisions are precise::divide, also
// measured necessary: plain `/` lets fast math rewrite (dot/normA)/normB into
// dot·rcp(normA·normB), and that reciprocal is SUBNORMAL for norm products ≳ 8.5e37 — flushed
// to zero, collapsing the similarity to 0 for |components| ≈ 1e19 even with every accumulator
// finite. Same lesson as the normalize kernels (see VA_NORM_MIN_SCALED notes above). With the
// two-stage precise divide every intermediate is exact-by-construction: |dotAB/normA| ≤ normB
// by Cauchy-Schwarz, so the final quotient lands in [-1, 1] up to rounding.
inline float va_cosine_similarity_finalize(float dotAB, float aa, float bb) {
    const float normA = sqrt(aa);
    const float normB = sqrt(bb);
    if (normA > FLT_MIN && normB > FLT_MIN) {
        const float raw = precise::divide(precise::divide(dotAB, normA), normB);
        return isnan(raw) ? raw : clamp(raw, -1.0f, 1.0f);
    }
    return (isnan(dotAB) || isnan(aa) || isnan(bb)) ? NAN : 0.0f;
}

#endif // VA_COSINE_RESCUE_DEFINED

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
// Scalar-backed storage uses packed vector pointers (scalar alignment). Keep
// vector arithmetic in registers; never assume a row/stride is 16-byte aligned.

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

    device const packed_float4* a4 = reinterpret_cast<device const packed_float4*>(vec_a);
    device const packed_float4* b4 = reinterpret_cast<device const packed_float4*>(vec_b);

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

    device const packed_float4* a4 = reinterpret_cast<device const packed_float4*>(vec_a);
    device const packed_float4* b4 = reinterpret_cast<device const packed_float4*>(vec_b);

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
