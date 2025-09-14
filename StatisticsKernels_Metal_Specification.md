# StatisticsKernels Metal Implementation Specification

## Project Context

**VectorAccelerate** is a high-performance GPU-accelerated vector computation framework for Apple platforms using Metal. This specification defines 4 Metal compute kernels for statistical operations that will be called from a Swift wrapper class.

## Architecture Requirements

### File Structure
- **File**: `Sources/VectorAccelerate/Metal/Shaders/StatisticsShaders.metal`
- **Language**: Metal Shading Language (MSL)
- **Platform**: macOS 14+, iOS 17+

### Code Style Requirements
```metal
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// Constants
constant float EPSILON = 1e-7f;

// Function naming: Use descriptive names matching Swift API
kernel void computeBasicStatistics(/* parameters */) {
    // Implementation
}
```

### Performance Patterns
- **Parallel Reduction**: Use threadgroup memory for aggregation operations
- **Thread Safety**: Use `threadgroup_barrier(mem_flags::mem_threadgroup)`
- **Numerical Stability**: Implement Welford's algorithm for variance computation
- **Memory Access**: Optimize for coalesced memory access patterns

## Required Kernels

### 1. computeBasicStatistics
**Purpose**: Single-pass computation of mean, variance, min, max, sum using Welford's algorithm

**Signature**:
```metal
kernel void computeBasicStatistics(
    device const float* input [[buffer(0)]],           // Input data array
    device float* output [[buffer(1)]],                // Output: [mean, M2, min, max, sum, count]
    constant uint& dimension [[buffer(2)]],            // Array size
    uint tid [[thread_position_in_threadgroup]],       // Thread ID in threadgroup
    uint tgSize [[threads_per_threadgroup]]            // Threadgroup size
);
```

**Algorithm**: Welford's Online Algorithm
```
For each value x:
    count = count + 1
    delta = x - mean
    mean = mean + delta/count
    delta2 = x - mean  
    M2 = M2 + delta*delta2
    
variance = M2 / (count - 1)  // or M2/count for population variance
```

**Implementation Requirements**:
- Use `threadgroup float` arrays for reduction (size 256+)
- Implement tree reduction pattern for parallel aggregation
- Handle min/max tracking simultaneously with mean/variance
- Output 6 values: `[mean, M2, minimum, maximum, sum, count_as_float]`
- **Critical**: M2 is the sum of squared differences, NOT variance
- Use `threadgroup_barrier(mem_flags::mem_threadgroup)` between reduction phases

**Memory Pattern**:
```metal
threadgroup float means[256];
threadgroup float M2s[256]; 
threadgroup float mins[256];
threadgroup float maxs[256];
threadgroup float sums[256];
```

### 2. computeHigherMoments
**Purpose**: Compute third and fourth central moments for skewness/kurtosis

**Signature**:
```metal
kernel void computeHigherMoments(
    device const float* input [[buffer(0)]],           // Input data array
    device float* output [[buffer(1)]],                // Output: [M3_sum, M4_sum]
    constant uint& dimension [[buffer(2)]],            // Array size
    constant float& mean [[buffer(3)]],                // Pre-computed mean
    uint tid [[thread_position_in_threadgroup]],       // Thread ID
    uint tgSize [[threads_per_threadgroup]]            // Threadgroup size  
);
```

**Algorithm**:
```
M3_sum = Σ(x - mean)³
M4_sum = Σ(x - mean)⁴

// In Swift:
skewness = (M3_sum/N) / (variance^1.5)
kurtosis = (M4_sum/N) / (variance^2)
```

**Implementation Requirements**:
- Simple parallel reduction pattern
- Compute `(x - mean)^3` and `(x - mean)^4` simultaneously
- Use separate threadgroup arrays for M3 and M4 accumulation
- Output exactly 2 values: `[M3_sum, M4_sum]`

### 3. computeQuantiles  
**Purpose**: Linear interpolation on sorted data for percentile computation

**Signature**:
```metal
kernel void computeQuantiles(
    device const float* sortedData [[buffer(0)]],      // Pre-sorted input array
    device const float* levels [[buffer(1)]],          // Quantile levels [0.0-1.0]
    device float* output [[buffer(2)]],                // Output quantile values
    constant uint& dataSize [[buffer(3)]],             // Size of sorted data
    constant uint& numLevels [[buffer(4)]],            // Number of quantile levels
    uint tid [[thread_position_in_grid]]               // Thread ID (one per quantile)
);
```

**Algorithm**: Linear Interpolation
```
For quantile level p ∈ [0,1]:
    position = p * (N - 1)
    lower_idx = floor(position)  
    upper_idx = ceil(position)
    fraction = position - lower_idx
    
    if lower_idx == upper_idx:
        result = sortedData[lower_idx]
    else:
        result = sortedData[lower_idx] * (1 - fraction) + 
                sortedData[upper_idx] * fraction
```

**Implementation Requirements**:
- **Embarrassingly parallel**: One thread per quantile level
- Input data is **already sorted** on CPU using vDSP_vsort
- Handle edge cases: level=0.0 → first element, level=1.0 → last element  
- Use `clamp()` for index bounds safety
- No reduction needed - direct computation per thread

### 4. computeCorrelation
**Purpose**: Compute correlation and covariance matrices for multiple datasets

**Signature**:
```metal
kernel void computeCorrelation(
    device const float* datasets [[buffer(0)]],        // Flattened datasets [dataset][dimension]
    device float* output [[buffer(1)]],                // Output: [correlation_matrix, covariance_matrix]
    constant uint& dataSize [[buffer(2)]],             // Elements per dataset
    constant uint& numDatasets [[buffer(3)]],          // Number of datasets
    uint2 gid [[thread_position_in_grid]]              // 2D grid: [dataset_i, dataset_j]
);
```

**Algorithm**: Pearson Correlation
```
For datasets X and Y:
    mean_X = Σ(X_i) / N
    mean_Y = Σ(Y_i) / N
    
    covariance = Σ((X_i - mean_X) * (Y_i - mean_Y)) / (N - 1)
    
    std_X = sqrt(Σ((X_i - mean_X)²) / (N - 1))
    std_Y = sqrt(Σ((Y_i - mean_Y)²) / (N - 1))
    
    correlation = covariance / (std_X * std_Y)
```

**Implementation Requirements**:
- **2D Grid**: Each thread computes correlation between datasets (i,j)
- **Memory Layout**: 
  - Input: `datasets[dataset_idx * dataSize + element_idx]`
  - Output: First N²elements = correlation matrix, next N² elements = covariance matrix
- **Symmetry**: Exploit matrix symmetry - only compute upper triangle, copy to lower
- **Numerical Stability**: Use two-pass algorithm (compute means first, then covariances)
- **Self-Correlation**: When i==j, correlation=1.0, covariance=variance

**Memory Pattern**:
```metal
// For efficiency, process in phases:
// Phase 1: Compute means for both datasets
// Phase 2: Compute covariance and variances in single pass
// Phase 3: Compute correlation from covariance and standard deviations
```

## Integration Requirements

### Buffer Management
- All kernels expect `device const float*` for read-only input
- All kernels use `device float*` for output buffers
- Buffer binding follows Metal convention: inputs first, outputs, then constants

### Error Handling
- Metal kernels don't throw exceptions
- Handle edge cases gracefully (return 0.0, NaN, or appropriate default)
- Bounds checking using conditional execution, not assertions

### Thread Group Sizing
- Design kernels to work with threadgroup sizes 32-1024
- Use `threads_per_threadgroup` parameter, don't hardcode
- Optimize for GPU architectures (warps of 32 threads)

## Mathematical Accuracy Requirements

### Numerical Stability
- **Welford's Algorithm**: Mandatory for variance computation (not naive Σ(x²) - (Σx)²)
- **Two-Pass Algorithms**: Preferred for correlation (first pass: means, second pass: covariances)
- **Catastrophic Cancellation**: Avoid subtracting large similar numbers

### Edge Cases
- **Empty Data**: Handle gracefully (return 0 or NaN as appropriate)
- **Single Element**: variance=0, correlation undefined
- **Constant Data**: variance=0, correlation=NaN or 0
- **Zero Variance**: correlation undefined → return 0.0

### Precision Requirements
- Use `float` precision (32-bit) throughout
- Constants: `EPSILON = 1e-7f` for zero comparisons
- Avoid `double` - not well supported on all GPU architectures

## Performance Targets

### Throughput Expectations
- **Basic Statistics**: >100M elements/second for large arrays
- **Higher Moments**: >80M elements/second  
- **Quantiles**: >1M quantiles/second on sorted data
- **Correlation**: >10K dataset pairs/second (1000-element datasets)

### Memory Bandwidth
- Optimize for coalesced memory access
- Minimize shared memory bank conflicts
- Use appropriate threadgroup sizes (multiples of 32)

## Testing Requirements

### Correctness Validation
Each kernel must match results from reference implementations:
- Basic statistics: Compare against Accelerate framework vDSP functions
- Moments: Compare against statistical libraries (scipy.stats)
- Quantiles: Compare against numpy.percentile
- Correlation: Compare against numpy.corrcoef

### Edge Case Testing
- Empty arrays, single elements, constant data
- Very small/large values (numerical stability)
- All-zero datasets, datasets with outliers
- NaN/infinite input handling

## Example Usage Pattern

The kernels will be called from Swift like this:
```swift
// 1. Create command buffer and encoder
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(basicStatsKernel)

// 2. Set buffers and parameters
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)
encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 2)

// 3. Dispatch threads
let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
let threadgroupCount = MTLSize(width: 1, height: 1, depth: 1) // Single threadgroup for reduction
encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
```

## Deliverable

**Single file**: `StatisticsShaders.metal` containing all 4 kernel implementations with:
- Proper Metal headers and namespace
- Comprehensive inline documentation
- Optimized parallel algorithms
- Numerical stability considerations
- Edge case handling
- Performance-optimized memory access patterns

The kernels should integrate seamlessly with the existing Swift StatisticsKernel wrapper class that expects these exact function names and signatures.