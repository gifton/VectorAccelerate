// VectorAccelerate: L2 Normalization Shaders
//
// GPU kernels for vector L2 normalization
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// -----------------------------------------------------------------------------
// Numerical policy (BE3 §4.4 — parity with VectorCore's NormalizeKernels)
// -----------------------------------------------------------------------------
// One thread owns one vector and runs the two-pass Kahan pre-scaled algorithm:
//
//   1. maxAbs = max |v_i|
//   2. den    = clamp(maxAbs, 2^-126, 2^126)       // VA_NORM_MIN/MAX_DENOM: keeps scale normal
//   3. scale  = 1 / den                            // normal: 2^-126 <= scale <= 2^126
//   4. sumSq  = Σ (v_i · scale)²                   // every term <= 16 => finite for any finite input
//   5. sNorm  = sqrt(sumSq) = ||v|| · scale
//   6. norm   = den · sNorm = ||v||                // reported only; saturates to +Inf above FLT_MAX
//   7. out    = precise::divide(v · scale, sNorm)  // = v/||v||, iff sNorm > 0.5 (VA_NORM_MIN_SCALED)
//
// Without step 1–3 the naive Σ v² underflows to 0 for subnormal-magnitude vectors
// (GPU returned an all-zero vector where the CPU returns a unit vector) and
// overflows to +Inf for vectors with huge components (GPU returned an all-zero
// vector after dividing by +Inf). Both are now gone: sumSq is bounded by
// 16·dimension, and every finite input with a representable reciprocal norm is
// normalized correctly — including ||v|| > FLT_MAX, where the output is still the
// exact unit vector because 1/||v|| is never formed. The reported `norm` is the
// true ||v||_2 whenever that is representable and +Inf above FLT_MAX.
//
// Degenerate inputs (step 7 guard fails — the true zero vector, and vectors whose
// magnitude is so small that 1/||v|| overflows FP32) are copied through
// **unchanged**, matching VectorCore's `NormalizeKernels.normalizeUnchecked`,
// which leaves the buffer untouched rather than poisoning it with Inf/NaN. For the
// zero vector "unchanged" and "zeroed" coincide, so this also preserves the
// historical L2 behavior.
//
// Denormal (FTZ) invariance: the step-7 guard compares only normal-range values,
// and the pass-through copies raw bits (`l2_copy_bits`), because Metal's default
// math mode flushes subnormals to zero — a float copy would rewrite a subnormal
// input as 0. A subnormal-magnitude vector reduces to maxAbs == 0 (denormals
// flushed) or to sNorm <= 0.5 (denormals honored); both land on the pass-through.
// Residual limitation: an all-subnormal vector whose norm still exceeds 2^-127 is
// normalizable on the CPU but passes through unchanged on a GPU that flushes
// denormals — normalizing it needs arithmetic on subnormal operands.
//
// `epsilon` is purely the caller's *explicit* degenerate threshold: vectors with
// 0 < ||v|| <= epsilon are zeroed. It is no longer load-bearing for numerical
// stability, so it defaults to 0 on the Swift side; no epsilon value (including 0)
// can produce Inf/NaN because the reciprocal is only ever formed after the
// `sNorm > VA_NORM_MIN_SCALED` guard has proved it representable.

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

/// Result of the pre-scaled norm computation for one vector.
struct L2NormFactor {
    float scale;        // 1 / clamp(maxAbs, 2^-126, 2^126) — always a normal float
    float scaled_norm;  // ||v||_2 · scale — normal-range whenever the vector is nonzero
    float norm;         // ||v||_2, reported only; saturates to +Inf above FLT_MAX
    bool normalizable;  // false ⇒ pass the input through unchanged
};

/// Maximum absolute component of a vector (pass 1 of the stable algorithm).
inline float l2_max_abs(device const float* vector, uint dimension) {
    const uint simd_blocks = dimension / 4;
    device const float4* vec4 = (device const float4*)vector;

    float4 acc = 0.0f;
    for (uint i = 0; i < simd_blocks; ++i) {
        acc = max(acc, fabs(vec4[i]));
    }
    float max_abs = max(max(acc.x, acc.y), max(acc.z, acc.w));

    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        max_abs = max(max_abs, fabs(vector[i]));
    }
    return max_abs;
}

/// Sum of squares of the pre-scaled vector: Σ (v_i · scale)² (pass 2).
///
/// The scale must be applied per element *before* squaring: folding it out as
/// `(Σ v²) · scale²` underflows (`scale² = 2^-252` for the largest vectors) and
/// overflows (`v²`) — the two failures this whole algorithm exists to prevent.
inline float l2_scaled_norm_sq(device const float* vector, uint dimension, float scale) {
    float norm_sq = 0.0f;
    const uint simd_blocks = dimension / 4;

    device const float4* vec4 = (device const float4*)vector;

    // Process 4 elements at a time
    for (uint i = 0; i < simd_blocks; ++i) {
        float4 s = vec4[i] * scale;
        norm_sq += dot(s, s);
    }

    // Handle remaining elements
    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        float s = vector[i] * scale;
        norm_sq += s * s;
    }

    return norm_sq;
}

/// Full two-pass norm computation for one vector (steps 1–7 of the policy above).
///
/// Everything the *output* depends on stays in the pre-scaled domain: `scale` and
/// `scaled_norm` are both normal floats, whereas `1/||v||` is subnormal for any
/// vector with `||v|| > 2^126` and would be flushed to zero — and the vector
/// misread as degenerate — on a GPU with denormals-are-zero. `norm` is reported
/// but never inverted, and the output is produced by `precise::divide` so that
/// fast-math cannot reassociate the two scalings back into that subnormal factor.
inline L2NormFactor l2_norm_factor(device const float* vector, uint dimension) {
    const float max_abs = l2_max_abs(vector, dimension);
    // Clamped on BOTH sides so `scale` is always normal (see VA_NORM_MIN/MAX_DENOM).
    const float den = clamp(max_abs, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);

    L2NormFactor f;
    f.scale = 1.0f / den;
    f.scaled_norm = sqrt(l2_scaled_norm_sq(vector, dimension, f.scale));
    f.norm = den * f.scaled_norm;   // +Inf when ||v|| genuinely exceeds FLT_MAX
    f.normalizable = (f.scaled_norm > VA_NORM_MIN_SCALED);
    return f;
}

/// Bit-exact copy of a whole vector (the degenerate pass-through).
///
/// Copies raw 32-bit patterns rather than floats: Metal's default math mode
/// flushes subnormals to zero, so a float copy would rewrite a subnormal input as
/// 0 and diverge from the CPU's "return the input unchanged".
inline void l2_copy_bits(device const float* input, device float* output, uint dimension) {
    const uint simd_blocks = dimension / 4;

    device const uint4* in4 = (device const uint4*)input;
    device uint4* out4 = (device uint4*)output;
    for (uint i = 0; i < simd_blocks; ++i) {
        out4[i] = in4[i];
    }

    device const uint* in1 = (device const uint*)input;
    device uint* out1 = (device uint*)output;
    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        out1[i] = in1[i];
    }
}

/// Write `precise::divide(input · scale, divisor)`.
///
/// `divisor < 0` requests the degenerate pass-through (a verbatim bitwise copy);
/// `divisor == 0` emits zeros — the caller's epsilon policy. Both operands of the
/// division are normal-range, and `precise::divide` prevents fast-math from
/// rewriting it as a multiply by the (subnormal) reciprocal.
void apply_normalization(
    device const float* input,
    device float* output,
    uint dimension,
    float scale,
    float divisor
) {
    if (divisor < 0.0f) {
        l2_copy_bits(input, output, dimension);
        return;
    }

    const uint simd_blocks = dimension / 4;

    device const float4* in4 = (device const float4*)input;
    device float4* out4 = (device float4*)output;

    if (divisor == 0.0f) {
        for (uint i = 0; i < simd_blocks; ++i) { out4[i] = float4(0.0f); }
        for (uint i = simd_blocks * 4; i < dimension; ++i) { output[i] = 0.0f; }
        return;
    }

    const float4 divisor4 = float4(divisor);

    // Vectorized normalization (both operands normal-range; see l2_norm_factor)
    for (uint i = 0; i < simd_blocks; ++i) {
        out4[i] = precise::divide(in4[i] * scale, divisor4);
    }

    // Handle remaining elements
    for (uint i = simd_blocks * 4; i < dimension; ++i) {
        output[i] = precise::divide(input[i] * scale, divisor);
    }
}

/// Resolve the divisor to apply, encoding the degenerate pass-through as -1.
///
/// - not normalizable (zero vector or deep subnormal) → copy the input through
///   unchanged, matching VectorCore CPU.
/// - `||v|| <= epsilon` → the caller's explicit degenerate threshold: emit zeros.
///   Tested as `sNorm > epsilon · scale`, which is `||v|| > epsilon` scaled into the
///   normal range on both sides (epsilon · scale saturating to +Inf simply means
///   epsilon dwarfs ||v||, which is the correct answer; underflowing to 0 means the
///   opposite, and both are handled correctly by the comparison).
/// - otherwise → `sNorm`, the divisor for the pre-scaled vector.
inline float l2_resolve_divisor(L2NormFactor f, float epsilon) {
    if (!f.normalizable) { return -1.0f; }             // pass-through sentinel
    return (f.scaled_norm > epsilon * f.scale) ? f.scaled_norm : 0.0f;
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

    // Phase 1: Compute Norm (pre-scaled, overflow- and underflow-safe)
    L2NormFactor f = l2_norm_factor(current_input, params.dimension);

    // Store norm if requested (Metal safely handles nullptr if the buffer wasn't bound)
    if (params.store_norms && norms != nullptr) {
        norms[tid] = f.norm;
    }

    // Phase 2: Normalize and Write
    apply_normalization(current_input, current_output, params.dimension,
                        f.scale, l2_resolve_divisor(f, params.epsilon));
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

    // Phase 1: Compute Norm (pre-scaled, overflow- and underflow-safe)
    L2NormFactor f = l2_norm_factor(current_vector, params.dimension);

    // Store norm if requested
    if (params.store_norms && norms != nullptr) {
        norms[tid] = f.norm;
    }

    // Phase 2: Normalize and Write (In-place)
    apply_normalization(current_vector, current_vector, params.dimension,
                        f.scale, l2_resolve_divisor(f, params.epsilon));
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

    constexpr uint NUM_BLOCKS = DIMENSION / 4;
    constexpr uint UNROLL_FACTOR = 4; // Unrolling 4 float4s (16 elements)

    // Phase 1a: max |v_i| with 4 accumulators for Instruction Level Parallelism (ILP)
    float4 m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    for (uint i = 0; i < NUM_BLOCKS; i += UNROLL_FACTOR) {
        m0 = max(m0, fabs(in4[i]));
        m1 = max(m1, fabs(in4[i+1]));
        m2 = max(m2, fabs(in4[i+2]));
        m3 = max(m3, fabs(in4[i+3]));
    }
    float4 m = max(max(m0, m1), max(m2, m3));
    const float max_abs = max(max(m.x, m.y), max(m.z, m.w));

    const float den = clamp(max_abs, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
    const float scale = 1.0f / den;

    // Phase 1b: Σ (v_i · scale)², again with 4 accumulators
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Process 16 elements per iteration
    for (uint i = 0; i < NUM_BLOCKS; i += UNROLL_FACTOR) {
        float4 v0 = in4[i] * scale;
        float4 v1 = in4[i+1] * scale;
        float4 v2 = in4[i+2] * scale;
        float4 v3 = in4[i+3] * scale;

        // Use Fused Multiply-Add (FMA) for better performance and precision
        acc0 = fma(v0, v0, acc0);
        acc1 = fma(v1, v1, acc1);
        acc2 = fma(v2, v2, acc2);
        acc3 = fma(v3, v3, acc3);
    }

    // Final reduction
    float4 sum = acc0 + acc1 + acc2 + acc3;
    float norm_sq = sum.x + sum.y + sum.z + sum.w;
    float scaled_norm = sqrt(norm_sq);

    L2NormFactor f;
    f.scale = scale;
    f.scaled_norm = scaled_norm;
    f.norm = den * scaled_norm;   // +Inf when ||v|| genuinely exceeds FLT_MAX
    f.normalizable = (scaled_norm > VA_NORM_MIN_SCALED);

    // Store norm if requested
    if (params.store_norms && norms != nullptr) {
        norms[tid] = f.norm;
    }

    // Phase 2: Normalize
    const float divisor = l2_resolve_divisor(f, params.epsilon);
    if (divisor < 0.0f) {
        l2_copy_bits(input + offset, output + offset, DIMENSION);
        return;
    }
    if (divisor == 0.0f) {
        for (uint i = 0; i < NUM_BLOCKS; ++i) { out4[i] = float4(0.0f); }
        return;
    }

    const float4 divisor4 = float4(divisor);
    for (uint i = 0; i < NUM_BLOCKS; ++i) {
        out4[i] = precise::divide(in4[i] * scale, divisor4);
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
