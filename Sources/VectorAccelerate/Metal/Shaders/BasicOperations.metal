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

// Local alias for backward compatibility
constant float EPSILON = VA_EPSILON;

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
//   7. out    = precise::divide(v · scale, sNorm)    // = v/||v||, iff sNorm > 0.5
//
// The previous implementation accumulated Σ v² directly and fell back to
// `magnitude > EPSILON (1e-7)`, which diverged from the CPU three ways: vectors
// with subnormal components underflowed to magnitude 0, vectors with huge
// components overflowed to +Inf (then divided to all-zero), and any vector with a
// legitimately small norm (e.g. 1e-19) was silently returned unnormalized.
//
// Degenerate policy (step 7 guard fails): the true zero vector and vectors whose
// norm is too small for 1/||v|| to be representable in FP32 are copied through
// **unchanged**. That is exactly what VectorCore's
// `NormalizeKernels.normalizeUnchecked` / `normalizedUncheckedNNN` do — they
// leave the buffer untouched rather than scaling it by Inf/NaN. (VectorCore's
// *checked* `normalized()` returns `.failure` for these inputs; these kernels have
// no error channel, so they mirror the unchecked form.)
//
// Denormal (FTZ) invariance: the step-7 guard compares only normal-range values,
// and the pass-through copies raw bits (`va_copy_bits`) rather than float values,
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
inline void va_copy_bits(device const float* src, device float* dst, uint i) {
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

    r.scaled_norm = (scaled_norm > VA_NORM_MIN_SCALED) ? scaled_norm : 0.0f;
    return r;
}

// MARK: - Basic Distance Operations

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
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = sqrt(partialSums[0]);
    }
}

/// Compute squared Euclidean distance (no sqrt for performance)
kernel void squaredEuclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = partialSums[0];
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
    threadgroup float dotProducts[256];
    threadgroup float normA[256];
    threadgroup float normB[256];
    
    float localDot = 0.0f;
    float localNormA = 0.0f;
    float localNormB = 0.0f;
    
    for (uint i = tid; i < dimension; i += tgSize) {
        float a = vectorA[i];
        float b = vectorB[i];
        localDot += a * b;
        localNormA += a * a;
        localNormB += b * b;
    }
    
    dotProducts[tid] = localDot;
    normA[tid] = localNormA;
    normB[tid] = localNormB;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for all three values
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            dotProducts[tid] += dotProducts[tid + stride];
            normA[tid] += normA[tid + stride];
            normB[tid] += normB[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        float dot = dotProducts[0];
        float magA = sqrt(normA[0]);
        float magB = sqrt(normB[0]);
        
        // FLT_MIN (leastNormalMagnitude) floor, not 1e-8: don't reject valid dense micro-vectors
        // as zero (parity with VectorCore BE3 4.5).
        if (magA > FLT_MIN && magB > FLT_MIN) {
            // Clamp to the valid cosine range so FP drift can't push the similarity past
            // 1.0 and produce a negative distance.
            float cosineSim = clamp(dot / (magA * magB), -1.0f, 1.0f);
            result[0] = 1.0f - cosineSim;
        } else if (isnan(dot) || isnan(magA) || isnan(magB)) {
            // Propagate NaN (consistent with the cosine_similarity kernel) rather than
            // collapsing a NaN-bearing pair to a finite distance.
            result[0] = NAN;
        } else {
            result[0] = 1.0f;  // zero-length vector
        }
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
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        sum += vectorA[i] * vectorB[i];
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = partialSums[0];
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
    
    const uint vector_offset = vector_idx * dimension;

    // Phase 1: Compute ||v|| with the pre-scaled two-pass algorithm (see the
    // normalization policy above). `shared_sums` is the caller-provided scratch;
    // only its first min(tg_size, 256) elements are touched.
    const uint lanes = min(tg_size, VA_NORM_REDUCE_LANES);
    const VANormScales f = va_normalize_scales(
        input + vector_offset, dimension, shared_sums, tid, lanes);

    // Phase 2: Normalize all dimensions of this vector (degenerate → unchanged)
    for (uint d = tid; d < dimension; d += tg_size) {
        const uint idx = vector_offset + d;
        if (f.scaled_norm > 0.0f) {
            output[idx] = precise::divide(input[idx] * f.scale, f.scaled_norm);
        } else {
            va_copy_bits(input, output, idx);
        }
    }
}

/// Optimized batch normalization for contiguous vectors
/// Uses 2D grid for efficient parallel processing
///
/// - Warning: Legacy/unused. This kernel accumulates the magnitude with device
///   atomics across threadgroups and then relies on a threadgroup barrier for
///   cross-threadgroup visibility, which Metal does not provide — the magnitude
///   it reads back is not guaranteed complete. It is therefore left on the old
///   `EPSILON` path and does **not** implement the normalization policy above.
///   Use `batchNormalize` (threadgroup-per-vector) or the `l2_normalize_*`
///   kernels, both of which are CPU-parity correct.
kernel void batchNormalize2D(
    device const float* input [[buffer(0)]],      // [num_vectors, dimension]
    device float* output [[buffer(1)]],           // [num_vectors, dimension]
    device float* magnitudes [[buffer(2)]],       // [num_vectors] - optional output
    constant uint& num_vectors [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint vector_idx = gid.x;
    const uint chunk_size = 32; // Process in chunks for better cache usage
    const uint chunk_idx = gid.y;
    
    if (vector_idx >= num_vectors) return;
    
    const uint vector_offset = vector_idx * dimension;
    const uint start_dim = chunk_idx * chunk_size;
    const uint end_dim = min(start_dim + chunk_size, dimension);
    
    // Step 1: Compute partial sum for this chunk
    float partial_sum = 0.0f;
    for (uint d = start_dim; d < end_dim; d++) {
        float val = input[vector_offset + d];
        partial_sum += val * val;
    }
    
    // Use atomic to accumulate across chunks (simple for small chunk counts)
    device atomic_float* atomic_magnitude = (device atomic_float*)&magnitudes[vector_idx];
    atomic_fetch_add_explicit(atomic_magnitude, partial_sum, memory_order_relaxed);
    
    // Synchronize using threadgroup barrier if within same threadgroup
    threadgroup_barrier(mem_flags::mem_device);
    
    // Step 2: Normalize this chunk
    float magnitude = sqrt(magnitudes[vector_idx]);
    for (uint d = start_dim; d < end_dim; d++) {
        if (magnitude > EPSILON) {
            output[vector_offset + d] = input[vector_offset + d] / magnitude;
        } else {
            output[vector_offset + d] = input[vector_offset + d];
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

/// Subtract two vectors element-wise
kernel void vectorSubtract(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] - vectorB[tid];
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
    uint rowOffset = tid * cols;
    
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
    uint dbOffset = dbIdx * dimension;
    
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
    
    distances[dbIdx] = sqrt(sum);
}

/// Batch cosine similarity computation
kernel void batchCosineSimilarity(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numDatabase) return;
    
    float dotProduct = 0.0f;
    float queryNorm = 0.0f;
    float dbNorm = 0.0f;
    
    uint dbOffset = id * dimension;
    
    for (uint i = 0; i < dimension; i++) {
        float q = query[i];
        float d = database[dbOffset + i];
        
        dotProduct += q * d;
        queryNorm += q * q;
        dbNorm += d * d;
    }
    
    queryNorm = sqrt(queryNorm);
    dbNorm = sqrt(dbNorm);
    
    // FLT_MIN floor (parity with VectorCore BE3 4.5; see cosineDistance above).
    if (queryNorm > FLT_MIN && dbNorm > FLT_MIN) {
        similarities[id] = dotProduct / (queryNorm * dbNorm);
    } else {
        similarities[id] = 0.0f;
    }
}

// MARK: - Utility Operations

/// Compute L2 norm of a vector
kernel void vectorNorm(
    device const float* vector [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float val = vector[i];
        sum += val * val;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = sqrt(partialSums[0]);
    }
}

/// Element-wise multiplication (Hadamard product)
kernel void elementwiseMultiply(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] * vectorB[tid];
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