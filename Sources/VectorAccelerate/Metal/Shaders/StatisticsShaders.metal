// Sources/VectorAccelerate/Metal/Shaders/StatisticsShaders.metal

#include <metal_stdlib>
#include <metal_math>
#include <metal_compute> // Provides constants like FLT_MAX

using namespace metal;

// MARK: - Constants and Definitions

// Epsilon for zero comparisons in floating point arithmetic
constant float EPSILON = 1e-7f;

// Maximum threadgroup size supported by the reduction kernels.
// This defines the static allocation size for threadgroup memory.
// 1024 is the typical maximum on modern Apple GPUs.
constant uint MAX_TG_SIZE = 1024;

// MARK: - Helper Structures and Functions

// Structure for stable Welford's algorithm reduction and basic stats
struct StatsAggregate {
    float mean;
    float M2;   // Sum of squares of differences from the mean
    float minVal;
    float maxVal;
    float sum;
    uint count;
};

// Helper function to merge two StatsAggregates using Welford's combination formula.
// This provides numerical stability required for parallel variance calculation.
StatsAggregate mergeStats(StatsAggregate A, StatsAggregate B) {
    // Handle cases where one aggregate is empty
    if (A.count == 0) return B;
    if (B.count == 0) return A;

    StatsAggregate R;
    R.count = A.count + B.count;
    
    // Use floats for counts in calculations to maintain precision
    float countA = (float)A.count;
    float countB = (float)B.count;
    float countR = (float)R.count;

    float delta = B.mean - A.mean;

    // Stable formulation for combined mean: meanR = meanA + delta * (countB / countR)
    R.mean = A.mean + delta * (countB / countR);
    
    // Stable formulation for combined M2: M2R = M2A + M2B + delta^2 * (countA * countB / countR)
    R.M2 = A.M2 + B.M2 + delta * delta * (countA * countB / countR);

    // Standard reductions for min, max, sum
    R.minVal = min(A.minVal, B.minVal);
    R.maxVal = max(A.maxVal, B.maxVal);
    R.sum = A.sum + B.sum;

    return R;
}

// Structure and helper for Higher Moments reduction (simple summation)
struct MomentsAggregate {
    float M3_sum; // Sum of (x-mean)^3
    float M4_sum; // Sum of (x-mean)^4
};

MomentsAggregate mergeMoments(MomentsAggregate A, MomentsAggregate B) {
    return {A.M3_sum + B.M3_sum, A.M4_sum + B.M4_sum};
}

// Parameter structure for higherMoments kernel (matching Swift definition)
struct MomentParams {
    uint N;
    float mean;
};

// MARK: - Kernel 1: computeBasicStatistics

/**
 * @brief Computes basic statistics (mean, variance components, min, max, sum) using Welford's algorithm.
 *
 * @discussion This kernel is designed to be dispatched with a SINGLE threadgroup for full reduction.
 *             It uses a thread-stride loop to process the entire input array efficiently
 *             and performs a parallel reduction in threadgroup memory.
 *
 * @param input Input data array [buffer(0)]
 * @param output Output array [buffer(1)]. Format: [mean, M2, min, max, sum, count_as_float]
 * @param params Input parameters (uint2) [buffer(2)]. params[0] = dimension (N).
 */
kernel void computeBasicStatistics(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    // Swift implementation passes SIMD2<UInt32> (uint2)
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    uint dimension = params[0];

    // 1. Initialize local stats aggregate in registers.
    // Initialize min/max with limits for correct reduction.
    StatsAggregate localStats = {0.0f, 0.0f, FLT_MAX, -FLT_MAX, 0.0f, 0};

    // 2. Local aggregation phase (Iterative Welford)
    // Thread-stride loop ensures coalesced memory access and covers all data.
    for (uint i = tid; i < dimension; i += tgSize) {
        float x = input[i];

        // Update local stats using Welford's sequential online algorithm steps (efficient for local aggregation)
        localStats.count += 1;
        float delta = x - localStats.mean;
        localStats.mean += delta / (float)localStats.count;
        float delta2 = x - localStats.mean;
        localStats.M2 += delta * delta2;

        // Update min, max, sum
        localStats.minVal = min(localStats.minVal, x);
        localStats.maxVal = max(localStats.maxVal, x);
        localStats.sum += x;
    }

    // 3. Store local results in threadgroup memory
    // Static allocation based on MAX_TG_SIZE.
    threadgroup StatsAggregate sharedStats[MAX_TG_SIZE];
    
    // Safety check (should be prevented by Swift dispatcher)
    if (tid < MAX_TG_SIZE) {
        sharedStats[tid] = localStats;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Parallel reduction phase (Parallel Welford Combination)
    // Optimized tree reduction pattern using the stable mergeStats helper.
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            // Robustness check for non-power-of-2 tgSize
            if (tid + stride < tgSize) {
                sharedStats[tid] = mergeStats(sharedStats[tid], sharedStats[tid + stride]);
            }
        }
        // Barrier ensures synchronization before the next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Final output by thread 0
    if (tid == 0) {
        StatsAggregate finalStats = sharedStats[0];

        // Handle the edge case where input was empty (dimension=0)
        if (finalStats.count == 0) {
            output[0] = 0.0f; // mean
            output[1] = 0.0f; // M2
            output[2] = 0.0f; // min
            output[3] = 0.0f; // max
            output[4] = 0.0f; // sum
            output[5] = 0.0f; // count
        } else {
            output[0] = finalStats.mean;
            output[1] = finalStats.M2;
            output[2] = finalStats.minVal;
            output[3] = finalStats.maxVal;
            output[4] = finalStats.sum;
            output[5] = (float)finalStats.count;
        }
    }
}

// MARK: - Kernel 2: computeHigherMoments

/**
 * @brief Computes the third and fourth central moments sums (M3_sum, M4_sum).
 *
 * @discussion Designed for a SINGLE threadgroup dispatch. Uses parallel summation reduction.
 *             M3_sum = Σ(x - mean)³, M4_sum = Σ(x - mean)⁴.
 *
 * @param input Input data array [buffer(0)]
 * @param output Output array [buffer(1)]. Format: [M3_sum, M4_sum]
 * @param params Input parameters (MomentParams struct) [buffer(2)].
 */
kernel void computeHigherMoments(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    // Swift implementation passes MomentParams struct
    constant MomentParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    uint dimension = params.N;
    float mean = params.mean;

    // 1. Initialize local moments
    MomentsAggregate localMoments = {0.0f, 0.0f};

    // 2. Local aggregation phase (Thread-stride loop)
    for (uint i = tid; i < dimension; i += tgSize) {
        float delta = input[i] - mean;
        
        // Optimized calculation of powers
        float delta2 = delta * delta;
        float delta3 = delta2 * delta;
        float delta4 = delta2 * delta2;

        localMoments.M3_sum += delta3;
        localMoments.M4_sum += delta4;
    }

    // 3. Store in threadgroup memory
    threadgroup MomentsAggregate sharedMoments[MAX_TG_SIZE];
    if (tid < MAX_TG_SIZE) {
        sharedMoments[tid] = localMoments;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Parallel reduction phase (simple sum reduction)
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (tid + stride < tgSize) {
                sharedMoments[tid] = mergeMoments(sharedMoments[tid], sharedMoments[tid + stride]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Final output by thread 0
    if (tid == 0) {
        output[0] = sharedMoments[0].M3_sum;
        output[1] = sharedMoments[0].M4_sum;
    }
}

// MARK: - Kernel 3: computeQuantiles

/**
 * @brief Computes quantiles using linear interpolation on pre-sorted data.
 *
 * @discussion Embarrassingly parallel: one thread per quantile level.
 *             Uses the standard R-7 linear interpolation method.
 *
 * @param sortedData Pre-sorted input array [buffer(0)]
 * @param levels Quantile levels [0.0-1.0] [buffer(1)]
 * @param output Output quantile values [buffer(2)]
 * @param params Input parameters (uint2) [buffer(3)]. params[0]=dataSize (N), params[1]=numLevels.
 * @param gid Thread position in grid (corresponds to index in levels/output)
 */
kernel void computeQuantiles(
    device const float* sortedData [[buffer(0)]],
    device const float* levels [[buffer(1)]],
    device float* output [[buffer(2)]],
    // Swift implementation passes SIMD2<UInt32> (uint2)
    constant uint2& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dataSize = params[0];
    uint numLevels = params[1];

    if (gid >= numLevels) {
        return;
    }

    // Handle edge cases: Empty or single element data
    if (dataSize == 0) {
        // Quantile of empty set is undefined.
        output[gid] = NAN;
        return;
    }
    if (dataSize == 1) {
        output[gid] = sortedData[0];
        return;
    }

    float p = levels[gid];
    // Safety clamp, although input should be validated by Swift side.
    p = clamp(p, 0.0f, 1.0f);

    // Algorithm: Linear Interpolation (R-7)
    // position = (N-1)p (0-based position index)
    float position = p * (float)(dataSize - 1);

    // Calculate indices
    uint lower_idx = (uint)floor(position);
    
    // Calculate interpolation fraction (gamma)
    float fraction = position - (float)lower_idx;

    // Determine upper index. It is lower_idx + 1, unless we are at the boundary (p=1.0).
    // We use min() to clamp the upper index efficiently.
    uint upper_idx = min(lower_idx + 1, dataSize - 1);

    float lower_val = sortedData[lower_idx];
    float upper_val = sortedData[upper_idx];

    // Interpolate: Q(p) = (1-gamma) * X[lower] + gamma * X[upper]
    // Using metal::mix for efficient linear interpolation (lerp)
    float result = mix(lower_val, upper_val, fraction);

    output[gid] = result;
}

// MARK: - Kernel 4: computeCorrelation

/**
 * @brief Computes the Pearson correlation and covariance matrices for multiple datasets.
 *
 * @discussion Uses a 2D grid where thread (i, j) computes the relationship between dataset i and j.
 *             Implements a numerically stable two-pass algorithm within the thread.
 *             Exploits matrix symmetry by only computing the upper triangle.
 *
 * @param datasets Flattened input datasets [buffer(0)]. Layout: [dataset_idx * dataSize + element_idx]
 * @param output Output matrices [buffer(1)]. Layout: [correlation_matrix..., covariance_matrix...]
 * @param params Input parameters (uint2) [buffer(2)]. params[0]=dataSize (N), params[1]=numDatasets (M).
 * @param gid 2D thread position in grid (dataset_i, dataset_j)
 */
kernel void computeCorrelation(
    device const float* datasets [[buffer(0)]],
    device float* output [[buffer(1)]],
    // Swift implementation passes SIMD2<UInt32> (uint2)
    constant uint2& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dataSize = params[0];
    uint numDatasets = params[1];
    
    uint i = gid.x;
    uint j = gid.y;

    if (i >= numDatasets || j >= numDatasets) {
        return;
    }

    // Optimization: Compute only the upper triangle (including diagonal)
    if (j < i) {
        return;
    }
    
    // Define output indices and matrix size
    uint matrix_size = numDatasets * numDatasets;
    uint idx_ij = i * numDatasets + j;
    uint idx_ji = j * numDatasets + i;

    // Handle insufficient data size (N < 2 required for sample statistics N-1)
    if (dataSize < 2) {
        float covariance = 0.0f;
        // Correlation is 1 if i==j (and N>=1), otherwise undefined (Spec requires 0.0).
        float correlation = (dataSize >= 1 && i == j) ? 1.0f : 0.0f;

        // Write results
        output[idx_ij] = correlation;
        output[matrix_size + idx_ij] = covariance;
        if (i != j) {
             output[idx_ji] = correlation;
             output[matrix_size + idx_ji] = covariance;
        }
        return;
    }

    // Calculate base addresses for datasets i and j
    device const float* data_i = datasets + i * dataSize;
    device const float* data_j = datasets + j * dataSize;

    // Phase 1: Compute means (Two-pass algorithm for stability)
    float sum_i = 0.0f;
    float sum_j = 0.0f;

    // Optimization: Variance case (i==j)
    if (i == j) {
        for (uint k = 0; k < dataSize; ++k) {
            sum_i += data_i[k];
        }
        sum_j = sum_i;
    } else {
        // General case
        for (uint k = 0; k < dataSize; ++k) {
            sum_i += data_i[k];
            sum_j += data_j[k];
        }
    }

    float N = (float)dataSize;
    float mean_i = sum_i / N;
    float mean_j = sum_j / N;

    // Phase 2: Compute sums for covariance and variances (M2 sums)
    float cov_sum = 0.0f; // Sum((X_k - mean_X) * (Y_k - mean_Y))
    float M2_i = 0.0f;
    float M2_j = 0.0f;

    if (i == j) {
        // Variance case
        for (uint k = 0; k < dataSize; ++k) {
            float delta_i = data_i[k] - mean_i;
            M2_i += delta_i * delta_i;
        }
        cov_sum = M2_i;
        M2_j = M2_i;
    } else {
        // Correlation case
        for (uint k = 0; k < dataSize; ++k) {
            float delta_i = data_i[k] - mean_i;
            float delta_j = data_j[k] - mean_j;

            cov_sum += delta_i * delta_j;
            M2_i += delta_i * delta_i;
            M2_j += delta_j * delta_j;
        }
    }

    // Phase 3: Final Calculation
    // Sample covariance (N-1 denominator, Bessel's correction)
    float covariance = cov_sum / (N - 1.0f);

    // Pearson correlation calculation
    // correlation = cov_sum / sqrt(M2_i * M2_j). (N-1) terms cancel out.
    float correlation = 0.0f;
    float corr_denom = sqrt(M2_i * M2_j);

    // Check for zero variance (constant data)
    if (corr_denom < EPSILON) {
        // If variance is zero, correlation is undefined. Return 0.0 as required by spec.
        correlation = 0.0f;
    } else {
        correlation = cov_sum / corr_denom;
        // Clamp result to [-1.0, 1.0] due to potential floating point inaccuracies
        correlation = clamp(correlation, -1.0f, 1.0f);
    }
    
    // Ensure diagonal is exactly 1.0 if variance exists, improving numerical robustness
    if (i == j && M2_i > EPSILON) {
        correlation = 1.0f;
    }

    // Write Output (Symmetrically)

    // Correlation matrix (first part of output buffer)
    output[idx_ij] = correlation;
    
    // Covariance matrix (second part of output buffer)
    output[matrix_size + idx_ij] = covariance;

    // Write lower triangle
    if (i != j) {
        output[idx_ji] = correlation;
        output[matrix_size + idx_ji] = covariance;
    }
}