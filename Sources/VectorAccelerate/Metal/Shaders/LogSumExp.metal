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

    device const float* row = input + tid * d;

    // Step 1: Find maximum for numerical stability
    float maxVal = row[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, row[i]);
    }

    // Handle edge case: all -inf
    if (maxVal == -INFINITY) {
        output[tid] = -INFINITY;
        return;
    }

    // Handle edge case: contains +inf
    if (maxVal == INFINITY) {
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

    device const float4* row = input + tid * d4;

    // Find maximum using float4
    float4 maxVec = row[0];
    for (uint i = 1; i < d4; i++) {
        maxVec = max(maxVec, row[i]);
    }
    float maxVal = max(max(maxVec.x, maxVec.y), max(maxVec.z, maxVec.w));

    // Handle edge cases
    if (maxVal == -INFINITY) {
        output[tid] = -INFINITY;
        return;
    }
    if (maxVal == INFINITY) {
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
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Shared memory for reduction (256 is typical threadgroup size)
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];

    // Grid-stride loop: compute grid size manually
    uint gridSize = tsize * numThreadgroups;

    // Load and find local max using grid-stride loop
    float localMax = -INFINITY;
    uint idx = tid;
    while (idx < count) {
        localMax = max(localMax, input[idx]);
        idx += gridSize;
    }
    sharedMax[lid] = localMax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max within threadgroup
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (lid < s && lid + s < tsize) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float groupMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle edge case: all -inf
    if (groupMax == -INFINITY) {
        if (lid == 0) {
            partialMax[tgid] = -INFINITY;
            partialSumExp[tgid] = 0.0f;
        }
        return;
    }

    // Compute sum of exp(x - groupMax) using grid-stride loop
    float localSum = 0.0f;
    idx = tid;
    while (idx < count) {
        float val = input[idx];
        // Only add finite values to sum
        if (val > -INFINITY) {
            localSum += exp(val - groupMax);
        }
        idx += gridSize;
    }
    sharedSum[lid] = localSum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum within threadgroup
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (lid < s && lid + s < tsize) {
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
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];

    // Load partials (handle case where numGroups < threadgroup size)
    float localMax = (lid < numGroups) ? partialMax[lid] : -INFINITY;
    sharedMax[lid] = localMax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float globalMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle edge case: all -inf
    if (globalMax == -INFINITY) {
        if (lid == 0) {
            output[0] = -INFINITY;
        }
        return;
    }

    // Handle edge case: contains +inf
    if (globalMax == INFINITY) {
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
    if (lid < numGroups && partialMax[lid] > -INFINITY) {
        localSum = partialSumExp[lid] * exp(partialMax[lid] - globalMax);
    }
    sharedSum[lid] = localSum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
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

    device const float* rowPtr = input + row * d;

    // Compute logsumexp for this row
    float maxVal = rowPtr[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, rowPtr[i]);
    }

    // Handle edge case
    if (maxVal == -INFINITY || maxVal == INFINITY) {
        // If all -inf, output would be NaN (0/0), set to 0
        // If contains +inf, the +inf element gets 1, others 0
        if (maxVal == INFINITY) {
            output[row * d + col] = (input[row * d + col] == INFINITY) ? 1.0f : 0.0f;
        } else {
            output[row * d + col] = 0.0f;
        }
        return;
    }

    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(rowPtr[i] - maxVal);
    }

    float lse = log(sumExp) + maxVal;

    // Output softmax: exp(x_i - logsumexp)
    output[row * d + col] = exp(input[row * d + col] - lse);
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

    device const float* rowIn = input + tid * d;
    device float* rowOut = output + tid * d;

    // Find max
    float maxVal = rowIn[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, rowIn[i]);
    }

    // Handle edge cases
    if (maxVal == -INFINITY) {
        for (uint i = 0; i < d; i++) {
            rowOut[i] = 0.0f;
        }
        return;
    }
    if (maxVal == INFINITY) {
        for (uint i = 0; i < d; i++) {
            rowOut[i] = (rowIn[i] == INFINITY) ? 1.0f : 0.0f;
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
