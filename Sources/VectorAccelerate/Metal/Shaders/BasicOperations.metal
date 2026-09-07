// VectorAccelerate: Basic Operations Shaders
//
// Core GPU kernels for fundamental vector and matrix operations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// This file contains the core compute kernels used throughout VectorAccelerate:
// - Euclidean distance (single pair and batch)
// - Cosine distance/similarity
// - Dot product
// - Vector normalization
// - Vector arithmetic (add, subtract, scale)
// - Matrix-vector multiplication

#include "Metal4Common.h"

// Use common constants from Metal4Common.h
// VA_EPSILON, VA_INFINITY, VA_INVALID_INDEX are available
// (The former `constant float EPSILON = VA_EPSILON;` alias lost its last user when
// batchNormalize2D was deleted in AUDIT-3 Group F and is gone; KernelContext's combined
// compile still strips that spelling if it ever reappears.)

// =============================================================================
// MARK: - Normalization Core (BE3 §4.4 — parity with VectorCore CPU)
// =============================================================================
//
// `vectorNormalize`, `normalizeVectors` and `batchNormalize` all run the same
// two-pass Kahan pre-scaled algorithm as VectorCore's `NormalizeKernels`, with
// both passes reduced across the threadgroup:
//
//   1. maxAbs = max |v_i|                            (threadgroup max reduction)
//   2. den    = clamp(maxAbs, 2^-126, 2^126)         // VA_NORM_MIN/MAX_DENOM: keeps scale normal
//   3. scale  = 1 / den                              // normal: 2^-126 <= scale <= 2^126
//   4. sumSq  = Σ (v_i · scale)²                     (threadgroup sum reduction; each term <= 16)
//   5. sNorm  = sqrt(sumSq) = ||v|| · scale
//   6. ||v||  = sNorm / scale                        // never materialized here
//   7. out    = precise::divide(v · scale, sNorm)    // = v/||v||, iff 0.5 < sNorm < 2^100
//
// The previous implementation accumulated Σ v² directly and fell back to
// `magnitude > EPSILON (1e-7)`, which diverged from the CPU three ways: vectors
// with subnormal components underflowed to magnitude 0, vectors with huge
// components overflowed to +Inf (then divided to all-zero), and any vector with a
// legitimately small norm (e.g. 1e-19) was silently returned unnormalized.
//
// Degenerate policy (step 7 guard fails): the true zero vector, vectors whose norm
// is too small for 1/||v|| to be representable in FP32 (||v|| <= 2^-127), and
// vectors containing a non-finite component (±Inf ⇒ sNorm +Inf; NaN ⇒ sNorm NaN)
// are copied through **unchanged**. That is exactly what VectorCore's
// `NormalizeKernels.normalizeUnchecked` / `normalizedUncheckedNNN` do — they
// leave the buffer untouched rather than scaling it by Inf/NaN. (VectorCore's
// *checked* `normalized()` returns `.failure` for these inputs; these kernels have
// no error channel, so they mirror the unchecked form.)
//
// Denormal (FTZ) invariance: every value reaching the step-7 guard is normal-range,
// ±Inf, or NaN — none affected by denormal flushing — so the guard's decision is
// identical whether or not the GPU flushes;
// the pass-through copies raw bits (`va_copy_bits`) rather than float values,
// because Metal's default math mode flushes subnormals to zero — a float copy
// would silently rewrite a subnormal input as 0 and diverge from the CPU. The
// decision itself is FTZ-invariant: a subnormal-magnitude vector either reduces to
// maxAbs == 0 (denormals flushed) or to sNorm <= 0.5 (denormals honored), and
// both land on the pass-through.
//
// Residual limitation: a vector whose components are *all* subnormal but whose
// norm still exceeds 2^-127 (e.g. every component 1e-38) is normalizable on the
// CPU but passes through unchanged on a GPU that flushes denormals — normalizing
// it requires arithmetic on subnormal operands, which such a GPU cannot do.

#define VA_NORM_REDUCE_LANES 256u

/// Bit-exact element copy used by the degenerate pass-through.
///
/// A float load/store would be flushed to zero for subnormal values under Metal's
/// default (denormals-are-zero) math mode, turning "return the input unchanged"
/// into "return zeros" and diverging from the CPU. Copying the raw 32-bit pattern
/// is immune to that.
inline void va_copy_bits(device const float* src, device float* dst, ulong i) {
    ((device uint*)dst)[i] = ((device const uint*)src)[i];
}

/// Threadgroup tree reduction (max). `scratch` must hold at least `lanes` floats.
///
/// Every thread of the threadgroup must call this — the barriers are uniform.
/// Threads with `lane >= lanes` contribute nothing and simply pass through.
/// Correct for any `lanes` in [1, VA_NORM_REDUCE_LANES], power of two or not:
/// the `lane + stride < lanes` guard folds the ragged tail into the low lanes on
/// the first pass, after which the tree is a clean power-of-two reduction.
inline float va_tg_reduce_max(float value, threadgroup float* scratch, uint lane, uint lanes) {
    if (lane < lanes) { scratch[lane] = value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = VA_NORM_REDUCE_LANES / 2; stride > 0; stride >>= 1) {
        if (lane < stride && lane + stride < lanes) {
            scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // every read completes before reuse
    return result;
}

/// Threadgroup tree reduction (sum). See `va_tg_reduce_max` for the contract.
inline float va_tg_reduce_add(float value, threadgroup float* scratch, uint lane, uint lanes) {
    if (lane < lanes) { scratch[lane] = value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = VA_NORM_REDUCE_LANES / 2; stride > 0; stride >>= 1) {
        if (lane < stride && lane + stride < lanes) {
            scratch[lane] += scratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // every read completes before reuse
    return result;
}

/// The pre-scale and the divisor that turn `v` into `v / ||v||_2`, as
/// `precise::divide(v_i · scale, scaled_norm)`.
///
/// Both operands are normal-range (`|v·scale| <= 4`, `scaled_norm > 0.5`), so the
/// result never depends on the GPU's denormal mode. Two things are deliberately
/// avoided here:
///
///  * a vector containing ±Inf reduces to `scaled_norm == +Inf`, which fails the
///    upper half of the guard (`VA_NORM_MAX_SCALED`) and is passed through
///    unchanged rather than producing `Inf/Inf = NaN`.
///  * `1/||v||` is never formed — it is subnormal for every vector with
///    `||v|| > 2^126` (components ≈ 4e36 upwards) and does not exist at all for
///    `||v|| > FLT_MAX`, so a denormals-are-zero GPU would flush it to 0 and
///    misread the vector as degenerate.
///  * the scaling is a *division*, not a multiplication by `1/scaled_norm`.
///    Under `-ffast-math` (Metal's default) the compiler is free to reassociate
///    `(v · scale) · (1/sNorm)` back into `v · (scale/sNorm)`, which reintroduces
///    exactly the subnormal multiplier this design removes — measured: every
///    component of a 4e36 vector came out as 0. `precise::divide` is IEEE-exact
///    and is not rewritten into a reciprocal-multiply.
struct VANormScales {
    float scale;        // 1 / clamp(maxAbs, 2^-126, 2^126)
    float scaled_norm;  // ||v|| · scale ∈ (0.5, 4·sqrt(dim)], or 0 ⇒ pass through unchanged
};

/// Steps 1–7 of the normalization policy above, computed cooperatively.
///
/// Every thread receives the same result; `lane`/`lanes` must be uniform across
/// the threadgroup.
inline VANormScales va_normalize_scales(
    device const float* v,
    uint dimension,
    threadgroup float* scratch,
    uint lane,
    uint lanes
) {
    float local_max = 0.0f;
    if (lane < lanes) {
        for (uint i = lane; i < dimension; i += lanes) {
            local_max = max(local_max, fabs(v[i]));
        }
    }
    const float max_abs = va_tg_reduce_max(local_max, scratch, lane, lanes);

    // Clamped on BOTH sides so `scale` is always a normal float: below 2^-126 the
    // reciprocal would overflow, above 2^126 the reciprocal would be subnormal
    // (and flushed to zero on a denormals-are-zero GPU).
    const float den = clamp(max_abs, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);

    VANormScales r;
    r.scale = 1.0f / den;

    // Do not "simplify" this to fma(v*v, scale*scale, ...): scale² underflows to
    // zero for the largest vectors (scale = 2^-126 ⇒ scale² = 2^-252) and v² overflows
    // for them, so the pre-scale must be applied per element, before squaring.
    float local_sum = 0.0f;
    if (lane < lanes) {
        for (uint i = lane; i < dimension; i += lanes) {
            const float s = v[i] * r.scale;
            local_sum = fma(s, s, local_sum);
        }
    }
    const float scaled_norm = sqrt(va_tg_reduce_add(local_sum, scratch, lane, lanes));

    // Range guard, both ends: too small ⇒ 1/||v|| is not representable; not finite
    // ⇒ the input contained ±Inf (or NaN, already excluded by the lower compare).
    // Both take the bit-exact pass-through leg.
    const bool normalizable = (scaled_norm > VA_NORM_MIN_SCALED) && (scaled_norm < VA_NORM_MAX_SCALED);
    r.scaled_norm = normalizable ? scaled_norm : 0.0f;
    return r;
}

// MARK: - Basic Distance Operations
//
// The three single-pair kernels below reduce through `va_tg_reduce_add` and are
// dispatch-robust: correct for ANY threadgroup width, power of two or not, and
// widths beyond VA_NORM_REDUCE_LANES clamp to a 256-lane cooperative pass instead
// of writing past the shared array. The previous implementations halved a raw
// `tgSize/2` stride, which silently orphans lanes on every odd halving (tgSize
// 100 → 50 → 25 → 12 drops lane 24, …) — wrong sums, not crashes — and
// `Metal4ComputeEngine` dispatches cosine/dot with `min(256, dimension)` threads,
// making every non-pow2 dimension in 17…255 a wrong answer (AUDIT-3 VA3-002).

/// Compute Euclidean distance between two vectors
/// Uses parallel reduction for optimal performance
kernel void euclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[VA_NORM_REDUCE_LANES];
    const uint lanes = min(tgSize, VA_NORM_REDUCE_LANES);

    // Accumulation strides by `lanes`, not tgSize: threads beyond the clamp do no
    // work (rather than gathering partial sums the reduction would never merge).
    float sum = 0.0f;
    if (tid < lanes) {
        for (uint i = tid; i < dimension; i += lanes) {
            float diff = vectorA[i] - vectorB[i];
            sum += diff * diff;
        }
    }
    const float total = va_tg_reduce_add(sum, partialSums, tid, lanes);

    if (tid == 0) {
        result[0] = va_euclidean_finalize(total, vectorA, vectorB, dimension);
    }
}

/// Compute cosine distance between two vectors
/// Returns 1 - cosine_similarity for distance metric
kernel void cosineDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float scratch[VA_NORM_REDUCE_LANES];
    const uint lanes = min(tgSize, VA_NORM_REDUCE_LANES);

    float localDot = 0.0f;
    float localNormA = 0.0f;
    float localNormB = 0.0f;
    if (tid < lanes) {
        for (uint i = tid; i < dimension; i += lanes) {
            float a = vectorA[i];
            float b = vectorB[i];
            localDot += a * b;
            localNormA += a * a;
            localNormB += b * b;
        }
    }

    // Three sequential tree reductions through one scratch array —
    // `va_tg_reduce_add`'s trailing barrier makes back-to-back reuse safe.
    float dot = va_tg_reduce_add(localDot, scratch, tid, lanes);
    float aa = va_tg_reduce_add(localNormA, scratch, tid, lanes);
    float bb = va_tg_reduce_add(localNormB, scratch, tid, lanes);

    if (tid == 0) {
        // Overflow/underflow rescue + shared finalization (Metal4Common.h, AUDIT-2 VA2-008):
        // recompute serially in the pre-scaled domain when the naive accumulators overflowed
        // (Inf) or collapsed to 0; finalize with the FLT_MIN degenerate floor (BE3 4.5) and the
        // NaN-propagating [-1, 1] clamp shared by every cosine kernel.
        if (va_cosine_accumulators_unreliable(dot, aa, bb)) {
            float3 rescued = va_cosine_rescaled_terms(vectorA, vectorB, dimension);
            dot = rescued.x;
            aa = rescued.y;
            bb = rescued.z;
        }
        result[0] = 1.0f - va_cosine_similarity_finalize(dot, aa, bb);
    }
}

/// Compute dot product between two vectors
kernel void dotProduct(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[VA_NORM_REDUCE_LANES];
    const uint lanes = min(tgSize, VA_NORM_REDUCE_LANES);

    float sum = 0.0f;
    if (tid < lanes) {
        for (uint i = tid; i < dimension; i += lanes) {
            sum += vectorA[i] * vectorB[i];
        }
    }
    const float total = va_tg_reduce_add(sum, partialSums, tid, lanes);

    if (tid == 0) {
        result[0] = total;
    }
}

// MARK: - Vector Operations

/// Batch normalization of multiple vectors in parallel
/// Each threadgroup processes one vector
kernel void batchNormalize(
    device const float* input [[buffer(0)]],      // [num_vectors, dimension]
    device float* output [[buffer(1)]],           // [num_vectors, dimension] 
    constant uint& num_vectors [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    threadgroup float* shared_sums [[threadgroup(0)]], // Shared memory for reduction
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each threadgroup processes one vector
    const uint vector_idx = tgid;
    
    if (vector_idx >= num_vectors) return;
    
    const ulong vector_offset = (ulong)vector_idx * dimension;

    // Phase 1: Compute ||v|| with the pre-scaled two-pass algorithm (see the
    // normalization policy above). `shared_sums` is the caller-provided scratch;
    // only its first min(tg_size, 256) elements are touched.
    const uint lanes = min(tg_size, VA_NORM_REDUCE_LANES);
    const VANormScales f = va_normalize_scales(
        input + vector_offset, dimension, shared_sums, tid, lanes);

    // Phase 2: Normalize all dimensions of this vector (degenerate → unchanged)
    for (uint d = tid; d < dimension; d += tg_size) {
        const ulong idx = vector_offset + d;
        if (f.scaled_norm > 0.0f) {
            output[idx] = precise::divide(input[idx] * f.scale, f.scaled_norm);
        } else {
            va_copy_bits(input, output, idx);
        }
    }
}

/// Normalize vector to unit length (single vector)
///
/// Two-pass pre-scaled algorithm — see the normalization policy at the top of
/// this file. Every threadgroup redundantly reduces the whole vector, so the
/// dispatch may cover `dimension` with any number of threadgroups.
///
/// Degenerate inputs (zero vector; ||v|| too small for 1/||v|| to be
/// representable) are copied through unchanged, matching VectorCore's
/// `NormalizeKernels.normalizeUnchecked`.
kernel void vectorNormalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[VA_NORM_REDUCE_LANES];
    const uint lanes = min(tgSize, VA_NORM_REDUCE_LANES);

    const VANormScales f = va_normalize_scales(input, dimension, partialSums, threadId, lanes);

    if (tid < dimension) {
        if (f.scaled_norm > 0.0f) {
            output[tid] = precise::divide(input[tid] * f.scale, f.scaled_norm);
        } else {
            va_copy_bits(input, output, tid);
        }
    }
}

/// Scale vector by scalar value
kernel void vectorScale(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = input[tid] * scalar;
}

/// Add two vectors element-wise
kernel void vectorAdd(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] + vectorB[tid];
}

// MARK: - Matrix Operations

/// Matrix-vector multiplication (y = Ax)
/// Each thread computes one output element
kernel void matrixVectorMultiply(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= rows) return;
    
    float sum = 0.0f;
    ulong rowOffset = (ulong)tid * cols;
    
    // Unroll loop for better performance with small vectors
    uint i = 0;
    for (; i + 3 < cols; i += 4) {
        sum += matrix[rowOffset + i] * vector[i];
        sum += matrix[rowOffset + i + 1] * vector[i + 1];
        sum += matrix[rowOffset + i + 2] * vector[i + 2];
        sum += matrix[rowOffset + i + 3] * vector[i + 3];
    }
    
    // Handle remaining elements
    for (; i < cols; i++) {
        sum += matrix[rowOffset + i] * vector[i];
    }
    
    output[tid] = sum;
}

// MARK: - Batch Operations

/// Batch Euclidean distance computation
/// Compute distances from one query to multiple database vectors
kernel void batchEuclideanDistance(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint dbIdx = id.x;
    if (dbIdx >= numDatabase) return;
    
    float sum = 0.0f;
    ulong dbOffset = (ulong)dbIdx * dimension;
    
    // Unrolled loop for better performance
    uint i = 0;
    for (; i + 3 < dimension; i += 4) {
        float diff0 = query[i] - database[dbOffset + i];
        float diff1 = query[i + 1] - database[dbOffset + i + 1];
        float diff2 = query[i + 2] - database[dbOffset + i + 2];
        float diff3 = query[i + 3] - database[dbOffset + i + 3];
        
        sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    
    // Handle remaining elements
    for (; i < dimension; i++) {
        float diff = query[i] - database[dbOffset + i];
        sum += diff * diff;
    }

    distances[dbIdx] = va_euclidean_finalize(sum, query, database + dbOffset, dimension);
}

/// Batch cosine DISTANCE (1 − similarity) from one query to multiple database vectors.
///
/// Dispatch-compatible with `batchEuclideanDistance` (same buffers, same `uint2` grid indexing) —
/// this is the kernel `Metal4ComputeEngine` requests for `metric == .cosine` batch/fused paths.
/// Before the 2026-08 hardening audit no kernel of this name existed anywhere (AUDIT-2 VA2-006/
/// VA2-008): every engine cosine batch dispatch threw `shaderNotFound` and was silently rescued
/// by the CPU fallback, so the GPU cosine fused path had never actually run.
kernel void batchCosineDistance(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint dbIdx = id.x;
    if (dbIdx >= numDatabase) return;

    device const float* candidate = database + (ulong)dbIdx * dimension;

    float dotAB = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    for (uint i = 0; i < dimension; i++) {
        float q = query[i];
        float d = candidate[i];
        dotAB = fma(q, d, dotAB);
        aa = fma(q, q, aa);
        bb = fma(d, d, bb);
    }

    // Overflow/underflow rescue + shared finalization (Metal4Common.h, AUDIT-2 VA2-008).
    if (va_cosine_accumulators_unreliable(dotAB, aa, bb)) {
        float3 rescued = va_cosine_rescaled_terms(query, candidate, dimension);
        dotAB = rescued.x;
        aa = rescued.y;
        bb = rescued.z;
    }
    distances[dbIdx] = 1.0f - va_cosine_similarity_finalize(dotAB, aa, bb);
}

/// Batch cosine SIMILARITY (not distance) from one query to multiple database vectors.
///
/// Finishes through the shared rescue trio like every other live cosine kernel. This one was
/// missed by the AUDIT-2 VA2-008 slice (AUDIT-3 VA3-006): it kept naive Float accumulators
/// (overflow/flush misclassification outside ~[1e-19, 1e19] component magnitude) and the
/// single-product denominator `queryNorm * dbNorm`, which fast math reassociates into
/// `sqrt(aa·bb)` — the exact overflow mechanism measured during the audit.
kernel void batchCosineSimilarity(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numDatabase) return;

    device const float* candidate = database + (ulong)id * dimension;

    float dotAB = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    for (uint i = 0; i < dimension; i++) {
        float q = query[i];
        float d = candidate[i];
        dotAB = fma(q, d, dotAB);
        aa = fma(q, q, aa);
        bb = fma(d, d, bb);
    }

    // Overflow/underflow rescue + shared finalization (Metal4Common.h, AUDIT-2 VA2-008):
    // NaN-propagating [-1, 1] clamp, FLT_MIN degenerate floor, zero-vector similarity 0.
    if (va_cosine_accumulators_unreliable(dotAB, aa, bb)) {
        float3 rescued = va_cosine_rescaled_terms(query, candidate, dimension);
        dotAB = rescued.x;
        aa = rescued.y;
        bb = rescued.z;
    }
    similarities[id] = va_cosine_similarity_finalize(dotAB, aa, bb);
}

// MARK: - Shader Aliases for Compatibility

/// Alias for vectorNormalize - some code expects "normalizeVectors"
/// Both kernels share `va_normalize_scales`, so they cannot drift apart.
kernel void normalizeVectors(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[VA_NORM_REDUCE_LANES];
    const uint lanes = min(tgSize, VA_NORM_REDUCE_LANES);

    const VANormScales f = va_normalize_scales(input, dimension, partialSums, threadId, lanes);

    if (tid < dimension) {
        if (f.scaled_norm > 0.0f) {
            output[tid] = precise::divide(input[tid] * f.scale, f.scaled_norm);
        } else {
            va_copy_bits(input, output, tid);
        }
    }
}