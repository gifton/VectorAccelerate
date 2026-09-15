//
//  LogSumExp.metal
//  VectorAccelerate
//
//  GPU kernels for numerically stable log-sum-exp and softmax operations.
//
//  Mathematical Background:
//  The naive logsumexp(x) = log(sum(exp(x_i))) overflows for large values.
//  The stable form is: logsumexp(x) = max(x) + log(sum(exp(x_i - max(x))))
//
//  Kernels:
//  - logsumexp_row_kernel: Row-wise logsumexp (one thread per row)
//  - logsumexp_row_vectorized_kernel: Vectorized row-wise (float4)
//  - logsumexp_reduce_pass1_kernel: First pass of full reduction
//  - logsumexp_reduce_pass2_kernel: Second pass combining partial results
//  - softmax_row_kernel: Row-wise softmax via logsumexp
//
//  Primary use case: Topic probability distributions and attention scores
//

#include <metal_stdlib>
using namespace metal;

// Use dynamic FLT_MAX comparisons for infinity branches under fast math (VA3-015).
// FastMathPolicyTests exercises row/reduction/softmax behavior on both compile paths.

// Integer classification survives fast-math assumptions about floating operands.
// NaNReductionPolicyTests covers signs/payloads and both compilation paths.
inline bool va_lse_is_nan(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}

inline bool va_lse_any_nan(float4 value) {
    return any((as_type<uint4>(value) & 0x7fffffffu) > 0x7f800000u);
}

// MARK: - Row-wise LogSumExp

/// Numerically stable log-sum-exp along rows.
/// Each thread handles one row.
///
/// logsumexp(x) = max(x) + log(sum(exp(x_i - max(x))))
///
/// - Parameters:
///   - input: [N, D] input matrix (row-major)
///   - output: [N] logsumexp per row
///   - n: Number of rows
///   - d: Number of columns
kernel void logsumexp_row_kernel(
    device const float* input       [[buffer(0)]],  // [N, D]
    device float* output            [[buffer(1)]],  // [N]
    constant uint& n                [[buffer(2)]],
    constant uint& d                [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float* row = input + (ulong)tid * d;

    // Step 1: Find maximum for numerical stability
    float maxVal = -INFINITY;
    for (uint i = 0; i < d; i++) {
        float value = row[i];
        if (va_lse_is_nan(value)) {
            output[tid] = NAN;
            return;
        }
        maxVal = max(maxVal, value);
    }

    // Handle edge case: all -inf
    if (maxVal < -FLT_MAX) {
        output[tid] = -INFINITY;
        return;
    }

    // Handle edge case: contains +inf
    if (maxVal > FLT_MAX) {
        output[tid] = INFINITY;
        return;
    }

    // Step 2: Sum of exp(x - max)
    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(row[i] - maxVal);
    }

    // Step 3: log(sum) + max
    output[tid] = log(sumExp) + maxVal;
}

/// Vectorized row-wise logsumexp for D divisible by 4.
/// Uses float4 for better memory throughput.
///
/// - Parameters:
///   - input: [N, D/4] input as float4 (row-major)
///   - output: [N] logsumexp per row
///   - n: Number of rows
///   - d4: D / 4 (number of float4 elements per row)
kernel void logsumexp_row_vectorized_kernel(
    device const float4* input      [[buffer(0)]],  // [N, D/4]
    device float* output            [[buffer(1)]],  // [N]
    constant uint& n                [[buffer(2)]],
    constant uint& d4               [[buffer(3)]],  // D / 4
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float4* row = input + (ulong)tid * d4;

    // Find maximum using float4
    float4 maxVec = float4(-INFINITY);
    for (uint i = 0; i < d4; i++) {
        float4 value = row[i];
        if (va_lse_any_nan(value)) {
            output[tid] = NAN;
            return;
        }
        maxVec = max(maxVec, value);
    }
    float maxVal = max(max(maxVec.x, maxVec.y), max(maxVec.z, maxVec.w));

    // Handle edge cases
    if (maxVal < -FLT_MAX) {
        output[tid] = -INFINITY;
        return;
    }
    if (maxVal > FLT_MAX) {
        output[tid] = INFINITY;
        return;
    }

    // Sum of exp(x - max)
    float4 sumVec = float4(0.0f);
    for (uint i = 0; i < d4; i++) {
        sumVec += exp(row[i] - maxVal);
    }
    float sumExp = sumVec.x + sumVec.y + sumVec.z + sumVec.w;

    output[tid] = log(sumExp) + maxVal;
}

// MARK: - Full Reduction (Two-Pass)

/// First pass: compute partial logsumexp per threadgroup.
///
/// Each threadgroup computes its local max and sum of exp(x - localMax).
/// These partials are later combined in pass 2.
///
/// - Parameters:
///   - input: Input array
///   - partialMax: [numGroups] max values per group
///   - partialSumExp: [numGroups] sum of exp(x - groupMax) per group
///   - count: Total number of elements
///   - numThreadgroups: Total number of threadgroups being dispatched
kernel void logsumexp_reduce_pass1_kernel(
    device const float* input           [[buffer(0)]],
    device float* partialMax            [[buffer(1)]],  // [numGroups]
    device float* partialSumExp         [[buffer(2)]],  // [numGroups]
    constant uint& count                [[buffer(3)]],
    constant uint& numThreadgroups      [[buffer(4)]],  // Total threadgroups
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];
    threadgroup uint sharedNaN[256];

    // Dispatch-robust for ANY threadgroup width (AUDIT-3 VA3-014): lanes clamp to the
    // shared-array capacity, loads are lane-guarded, and the trees start at a fixed
    // power-of-two stride with a ragged-tail guard. The pre-fix trees halved a raw
    // `tsize/2` stride, whose `lid + s < tsize` guard prevented the out-of-bounds read
    // but silently orphaned lanes on every odd halving — an under-count, not a crash.
    // Under the host's fixed 256-wide dispatch the element coverage below is identical
    // to the original grid-stride loop.
    const uint lanes = min(tsize, 256u);
    uint gridSize = lanes * numThreadgroups;
    uint base = tgid * lanes + lid;

    // Load and find local max using grid-stride loop
    float localMax = -INFINITY;
    uint localNaN = 0u;
    if (lid < lanes) {
        uint idx = base;
        while (idx < count) {
            float value = input[idx];
            localNaN |= uint(va_lse_is_nan(value));
            localMax = max(localMax, value);
            idx += gridSize;
        }
        sharedMax[lid] = localMax;
        sharedNaN[lid] = localNaN;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < lanes) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
            sharedNaN[lid] |= sharedNaN[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float groupMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Uniform decision after the reduction: NaN dominates either infinity.
    if (sharedNaN[0] != 0u) {
        if (lid == 0) {
            partialMax[tgid] = NAN;
            partialSumExp[tgid] = NAN;
        }
        return;
    }

    // Handle edge case: all -inf (uniform across the threadgroup — no divergent barriers)
    if (groupMax < -FLT_MAX) {
        if (lid == 0) {
            partialMax[tgid] = -INFINITY;
            partialSumExp[tgid] = 0.0f;
        }
        return;
    }

    // Symbolic +Inf partial. Avoid Inf-Inf manufacturing a NaN sum that pass2
    // would correctly treat as poison. Its scale is immaterial when max is +Inf.
    if (groupMax > FLT_MAX) {
        if (lid == 0) {
            partialMax[tgid] = INFINITY;
            partialSumExp[tgid] = 1.0f;
        }
        return;
    }

    // Compute sum of exp(x - groupMax) using grid-stride loop
    float localSum = 0.0f;
    if (lid < lanes) {
        uint idx = base;
        while (idx < count) {
            float val = input[idx];
            // NaNs/+Inf handled above; exclude -Inf, retaining -FLT_MAX.
            if (val >= -FLT_MAX) {
                localSum += exp(val - groupMax);
            }
            idx += gridSize;
        }
        sharedSum[lid] = localSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < lanes) {
            sharedSum[lid] += sharedSum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results
    if (lid == 0) {
        partialMax[tgid] = groupMax;
        partialSumExp[tgid] = sharedSum[0];
    }
}

/// Second pass: combine partial results from all threadgroups.
///
/// The partial sums were computed with different max values, so we need
/// to adjust them to a global max before combining:
///   sumExp_global = sum(partialSum_i * exp(partialMax_i - globalMax))
///
/// - Parameters:
///   - partialMax: [numGroups] max values from pass 1
///   - partialSumExp: [numGroups] partial sums from pass 1
///   - output: [1] final logsumexp result
///   - numGroups: Number of partial results
kernel void logsumexp_reduce_pass2_kernel(
    device const float* partialMax      [[buffer(0)]],
    device const float* partialSumExp   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant uint& numGroups            [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];
    threadgroup uint sharedNaN[256];

    // Dispatch-robust for ANY threadgroup width and ANY numGroups (AUDIT-3 VA3-014): the
    // pre-fix trees ran a guardless fixed-128 stride over sharedMax[256], so dispatching
    // fewer than 256 threads folded UNINITIALIZED threadgroup memory into the global max
    // (observed: garbage max → every exp(partialMax − globalMax) underflows → output −inf),
    // and partials beyond lane 255 were silently dropped. Loads now stride over the full
    // partial range and the trees carry the ragged-tail guard.
    const uint lanes = min(tsize, 256u);

    // Find this lane's max over its strided slice of the partials
    float localMax = -INFINITY;
    uint localNaN = 0u;
    if (lid < lanes) {
        for (uint i = lid; i < numGroups; i += lanes) {
            localNaN |= uint(va_lse_is_nan(partialMax[i]) || va_lse_is_nan(partialSumExp[i]));
            localMax = max(localMax, partialMax[i]);
        }
        sharedMax[lid] = localMax;
        sharedNaN[lid] = localNaN;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < lanes) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
            sharedNaN[lid] |= sharedNaN[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float globalMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sharedNaN[0] != 0u) {
        if (lid == 0) output[0] = NAN;
        return;
    }

    // Handle edge case: all -inf (uniform — every thread takes the same branch)
    if (globalMax < -FLT_MAX) {
        if (lid == 0) {
            output[0] = -INFINITY;
        }
        return;
    }

    // Handle edge case: contains +inf
    if (globalMax > FLT_MAX) {
        if (lid == 0) {
            output[0] = INFINITY;
        }
        return;
    }

    // Adjust sums to global max and reduce
    // partialSum_i was computed as sum(exp(x - partialMax_i))
    // We need: sum(exp(x - globalMax))
    //        = sum(exp(x - partialMax_i) * exp(partialMax_i - globalMax))
    //        = partialSum_i * exp(partialMax_i - globalMax)
    float localSum = 0.0f;
    if (lid < lanes) {
        for (uint i = lid; i < numGroups; i += lanes) {
            if (partialMax[i] >= -FLT_MAX) {
                localSum += partialSumExp[i] * exp(partialMax[i] - globalMax);
            }
        }
        sharedSum[lid] = localSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < lanes) {
            sharedSum[lid] += sharedSum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final result: log(sum) + globalMax
    if (lid == 0) {
        output[0] = log(sharedSum[0]) + globalMax;
    }
}

// MARK: - Softmax

/// Compute softmax using logsumexp for numerical stability.
///
/// softmax(x)_i = exp(x_i) / sum(exp(x_j))
///             = exp(x_i - logsumexp(x))
///
/// Each thread computes one element of the output.
///
/// - Parameters:
///   - input: [N, D] input matrix (row-major)
///   - output: [N, D] softmax output
///   - n: Number of rows
///   - d: Number of columns
kernel void softmax_row_kernel(
    device const float* input       [[buffer(0)]],  // [N, D]
    device float* output            [[buffer(1)]],  // [N, D]
    constant uint& n                [[buffer(2)]],
    constant uint& d                [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= n || col >= d) return;

    device const float* rowPtr = input + (ulong)row * d;

    // Compute logsumexp for this row
    float maxVal = rowPtr[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, rowPtr[i]);
    }

    // Handle edge case
    if (maxVal < -FLT_MAX || maxVal > FLT_MAX) {
        if (maxVal > FLT_MAX) {
            // One or more +Inf: the softmax limit is uniform over the argmax (Inf) set, so each
            // +Inf position gets 1/count and the rest get 0 — keeping the row summed to 1. (The
            // previous code assigned 1.0 to every +Inf, so a row with k infinities summed to k.)
            uint infCount = 0;
            for (uint i = 0; i < d; i++) {
                if (rowPtr[i] > FLT_MAX) infCount++;
            }
            output[(ulong)row * d + col] = (input[(ulong)row * d + col] > FLT_MAX) ? (1.0f / float(infCount)) : 0.0f;
        } else {
            // All -inf would give 0/0 = NaN; define the row as all zeros.
            output[(ulong)row * d + col] = 0.0f;
        }
        return;
    }

    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(rowPtr[i] - maxVal);
    }

    float lse = log(sumExp) + maxVal;

    // Output softmax: exp(x_i - logsumexp)
    output[(ulong)row * d + col] = exp(input[(ulong)row * d + col] - lse);
}

// MARK: - Softmax (Efficient Row-per-Thread Variant)

/// Efficient softmax: one thread computes entire row.
/// Better for smaller D where redundant logsumexp work dominates.
///
/// - Parameters:
///   - input: [N, D] input matrix
///   - output: [N, D] softmax output
///   - n: Number of rows
///   - d: Number of columns
kernel void softmax_row_efficient_kernel(
    device const float* input       [[buffer(0)]],  // [N, D]
    device float* output            [[buffer(1)]],  // [N, D]
    constant uint& n                [[buffer(2)]],
    constant uint& d                [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float* rowIn = input + (ulong)tid * d;
    device float* rowOut = output + (ulong)tid * d;

    // Find max
    float maxVal = rowIn[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, rowIn[i]);
    }

    // Handle edge cases
    if (maxVal < -FLT_MAX) {
        for (uint i = 0; i < d; i++) {
            rowOut[i] = 0.0f;
        }
        return;
    }
    if (maxVal > FLT_MAX) {
        // Uniform over the argmax (Inf) set so the row sums to 1 (previously each +Inf got 1.0).
        uint infCount = 0;
        for (uint i = 0; i < d; i++) {
            if (rowIn[i] > FLT_MAX) infCount++;
        }
        for (uint i = 0; i < d; i++) {
            rowOut[i] = (rowIn[i] > FLT_MAX) ? (1.0f / float(infCount)) : 0.0f;
        }
        return;
    }

    // Compute sum of exp(x - max)
    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(rowIn[i] - maxVal);
    }

    // Compute softmax values
    float invSumExp = 1.0f / sumExp;
    for (uint i = 0; i < d; i++) {
        rowOut[i] = exp(rowIn[i] - maxVal) * invSumExp;
    }
}
