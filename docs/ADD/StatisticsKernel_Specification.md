# StatisticsKernel Implementation Specification

## Project Context

**Project**: VectorAccelerate - High-performance GPU-accelerated vector operations framework for Swift  
**Version**: v0.1.0  
**Target Platform**: Apple Silicon (macOS/iOS) with Metal GPU compute  
**Language**: Swift 5.9+ with Metal Shading Language (MSL)  
**Dependencies**: Metal, Foundation, VectorCore (custom vector protocol library), QuartzCore, Accelerate

## Background

VectorAccelerate is a production GPU compute framework that provides high-performance mathematical operations for machine learning and scientific computing. The project follows strict architectural patterns established in v0.1.0, with consistent APIs across all compute kernels.

**Existing Kernel Examples in Codebase:**
- `MatrixMultiplyKernel` - Matrix operations with tiled GPU algorithms
- `HistogramKernel` - Distribution analysis with multiple binning strategies  
- `BinaryQuantizationKernel` - Bit-packing with Hamming distance computation
- `JaccardDistanceKernel` - Set similarity with GPU reduction patterns

## Task: Implement StatisticsKernel

Create a comprehensive statistics computation kernel that provides GPU-accelerated fundamental statistical operations (mean, variance, quantiles, correlation, etc.) following the established v0.1.0 architectural patterns.

## Mandatory Architectural Requirements

### 1. File Structure
**File Location**: `Sources/VectorAccelerate/Kernels/StatisticsKernel.swift`

**Required Imports**:
```swift
import Metal
import Foundation
import VectorCore
import QuartzCore  // For CACurrentMediaTime()
import Accelerate  // For CPU fallback operations
```

### 2. Class Declaration Pattern
```swift
public final class StatisticsKernel {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let basicStatsKernel: any MTLComputePipelineState
    private let momentsKernel: any MTLComputePipelineState
    private let quantilesKernel: any MTLComputePipelineState
    private let correlationKernel: any MTLComputePipelineState
}
```

**Critical**: All Metal protocol types MUST use `any` keyword (Swift concurrency requirement)

### 3. Constructor Pattern (MANDATORY)
```swift
public init(device: any MTLDevice) throws {
    self.device = device
    
    guard let queue = device.makeCommandQueue() else {
        throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
    }
    self.commandQueue = queue
    
    guard let library = device.makeDefaultLibrary() else {
        throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
    }
    
    // Load compute kernels
    guard let basicFunc = library.makeFunction(name: "basicStatistics") else {
        throw AccelerationError.shaderNotFound(name: "basicStatistics")
    }
    // ... load other kernels
    
    do {
        self.basicStatsKernel = try device.makeComputePipelineState(function: basicFunc)
        // ... create other pipeline states
    } catch {
        throw AccelerationError.computeFailed(reason: "Failed to create pipeline states: \(error)")
    }
    
    // Validate hardware support
    let maxThreadsPerThreadgroup = basicStatsKernel.maxTotalThreadsPerThreadgroup
    if maxThreadsPerThreadgroup < 256 {
        throw AccelerationError.unsupportedOperation(
            "Device does not support required threadgroup size for statistics computation"
        )
    }
}
```

### 4. Error Handling (EXACT Enum Usage Required)
**Critical**: Use ONLY these exact AccelerationError cases (other cases will cause compilation failures):

```swift
// Correct usage examples:
throw AccelerationError.deviceInitializationFailed("message")
throw AccelerationError.shaderNotFound(name: "kernelName")
throw AccelerationError.computeFailed(reason: "description")
throw AccelerationError.bufferAllocationFailed(size: Int)
throw AccelerationError.dimensionMismatch(expected: Int, actual: Int)
throw AccelerationError.invalidInput("message")
throw AccelerationError.unsupportedOperation("message")
```

**Wrong examples that will fail compilation:**
```swift
// DO NOT USE - these don't exist:
AccelerationError.kernelNotFound
AccelerationError.bufferCreationFailed  
AccelerationError.executionFailed
AccelerationError.invalidDimensions
```

### 5. Sendable Conformance (MANDATORY)
All result structs MUST be Sendable for Swift concurrency:

```swift
public struct StatisticsResult: Sendable {
    // All properties must be Sendable types
}
```

### 6. VectorCore Integration Pattern
The VectorCore library provides a `VectorProtocol` with these methods:
```swift
// Available methods:
vector.toArray() -> [Float]        // Convert to Swift array
vector.scalarCount -> Int          // Number of elements

// DO NOT USE - these don't exist:
vector.scalars  // This property doesn't exist
```

**Correct VectorCore integration pattern**:
```swift
public func computeStatistics<V: VectorProtocol>(
    data: [V],
    config: StatisticsConfig = .default
) throws -> StatisticsResult where V.Scalar == Float {
    let floatArrays = data.map { $0.toArray() }  // Correct
    return try computeStatistics(data: floatArrays, config: config)
}
```

### 7. Performance Timing Pattern
```swift
let startTime = CACurrentMediaTime()
// ... GPU operations ...
let executionTime = CACurrentMediaTime() - startTime
```

### 8. Async/Await Pattern (Required)
```swift
public func computeStatisticsAsync(
    data: [Float],
    config: StatisticsConfig = .default
) async throws -> StatisticsResult {
    return try await withCheckedThrowingContinuation { continuation in
        do {
            let result = try computeStatistics(data: data, config: config)
            continuation.resume(returning: result)
        } catch {
            continuation.resume(throwing: error)
        }
    }
}
```

## Required Data Structures

### 1. Configuration Structure
```swift
public struct StatisticsConfig: Sendable {
    public let computeHigherMoments: Bool        // Include skewness/kurtosis
    public let computeQuantiles: Bool            // Include percentiles/quartiles
    public let quantileLevels: [Float]           // Custom quantile levels [0.0-1.0]
    public let biasCorrection: Bool              // Apply Bessel's correction
    public let parallelReduction: Bool           // Use parallel reduction algorithms
    
    public init(
        computeHigherMoments: Bool = true,
        computeQuantiles: Bool = false,
        quantileLevels: [Float] = [0.25, 0.5, 0.75, 0.95],
        biasCorrection: Bool = true,
        parallelReduction: Bool = true
    ) {
        self.computeHigherMoments = computeHigherMoments
        self.computeQuantiles = computeQuantiles
        self.quantileLevels = quantileLevels
        self.biasCorrection = biasCorrection
        self.parallelReduction = parallelReduction
    }
    
    public static let `default` = StatisticsConfig()
}
```

### 2. Basic Statistics Result
```swift
public struct BasicStatistics: Sendable {
    public let count: Int
    public let mean: Float
    public let variance: Float
    public let standardDeviation: Float
    public let minimum: Float
    public let maximum: Float
    public let range: Float
    public let sum: Float
    public let executionTime: TimeInterval
    
    /// Derived properties
    public var coefficientOfVariation: Float { 
        mean != 0 ? standardDeviation / abs(mean) : 0 
    }
    
    public var standardError: Float { 
        standardDeviation / sqrt(Float(count)) 
    }
    
    /// 95% confidence interval for mean
    public func confidenceInterval(level: Float = 0.95) -> (lower: Float, upper: Float) {
        let tValue: Float = 1.96 // Approximate for large samples
        let margin = tValue * standardError
        return (mean - margin, mean + margin)
    }
}
```

### 3. Higher-Order Moments Result
```swift
public struct HigherMoments: Sendable {
    public let skewness: Float           // Third central moment
    public let kurtosis: Float           // Fourth central moment
    public let excessKurtosis: Float     // Kurtosis - 3.0
    public let executionTime: TimeInterval
    
    /// Distribution shape classification
    public var distributionShape: DistributionShape {
        if abs(skewness) < 0.5 { return .symmetric }
        else if skewness > 0.5 { return .rightSkewed }
        else { return .leftSkewed }
    }
    
    /// Tail behavior classification
    public var tailedness: Tailedness {
        if excessKurtosis > 1.0 { return .heavyTailed }
        else if excessKurtosis < -1.0 { return .lightTailed }
        else { return .mesokurtic }
    }
    
    public enum DistributionShape: String, Sendable {
        case symmetric = "Symmetric"
        case rightSkewed = "Right-skewed"
        case leftSkewed = "Left-skewed"
    }
    
    public enum Tailedness: String, Sendable {
        case lightTailed = "Light-tailed"
        case mesokurtic = "Normal-tailed"
        case heavyTailed = "Heavy-tailed"
    }
}
```

### 4. Quantiles Result
```swift
public struct QuantilesResult: Sendable {
    public let quantiles: [Float: Float]    // quantile_level: value
    public let executionTime: TimeInterval
    
    /// Standard quantiles
    public var median: Float? { quantiles[0.5] }
    public var firstQuartile: Float? { quantiles[0.25] }
    public var thirdQuartile: Float? { quantiles[0.75] }
    public var percentile95: Float? { quantiles[0.95] }
    
    /// Interquartile range
    public var iqr: Float? {
        guard let q1 = firstQuartile, let q3 = thirdQuartile else { return nil }
        return q3 - q1
    }
    
    /// Outlier detection bounds (1.5 * IQR rule)
    public var outlierBounds: (lower: Float, upper: Float)? {
        guard let q1 = firstQuartile, let q3 = thirdQuartile else { return nil }
        let iqrVal = q3 - q1
        return (q1 - 1.5 * iqrVal, q3 + 1.5 * iqrVal)
    }
}
```

### 5. Comprehensive Result
```swift
public struct StatisticsResult: Sendable {
    public let basic: BasicStatistics
    public let moments: HigherMoments?
    public let quantiles: QuantilesResult?
    public let totalExecutionTime: TimeInterval
    
    /// Convenience accessors
    public var mean: Float { basic.mean }
    public var standardDeviation: Float { basic.standardDeviation }
    public var median: Float? { quantiles?.median }
    public var skewness: Float? { moments?.skewness }
    
    /// Distribution normality assessment
    public var isApproximatelyNormal: Bool {
        guard let skew = moments?.skewness, 
              let excessKurt = moments?.excessKurtosis else { return false }
        return abs(skew) < 0.5 && abs(excessKurt) < 0.5
    }
    
    /// Generate summary report
    public func summary() -> String {
        var report = "Statistical Summary\\n"
        report += "==================\\n"
        report += "Count: \\(basic.count)\\n"
        report += "Mean: \\(String(format: \"%.6f\", basic.mean))\\n"
        report += "Std Dev: \\(String(format: \"%.6f\", basic.standardDeviation))\\n"
        report += "Range: [\\(String(format: \"%.6f\", basic.minimum)), \\(String(format: \"%.6f\", basic.maximum))]\\n"
        
        if let moments = moments {
            report += "Skewness: \\(String(format: \"%.6f\", moments.skewness))\\n"
            report += "Kurtosis: \\(String(format: \"%.6f\", moments.kurtosis))\\n"
            report += "Distribution: \\(moments.distributionShape.rawValue)\\n"
        }
        
        if let quantiles = quantiles {
            if let median = quantiles.median {
                report += "Median: \\(String(format: \"%.6f\", median))\\n"
            }
            if let iqr = quantiles.iqr {
                report += "IQR: \\(String(format: \"%.6f\", iqr))\\n"
            }
        }
        
        report += "Execution Time: \\(String(format: \"%.4f\", totalExecutionTime))s\\n"
        return report
    }
}
```

### 6. Batch Processing Result
```swift
public struct BatchStatisticsResult: Sendable {
    public let results: [StatisticsResult]
    public let totalExecutionTime: TimeInterval
    public let averageStatistics: BasicStatistics
    
    /// Get result at index
    public func result(at index: Int) -> StatisticsResult? {
        guard index >= 0 && index < results.count else { return nil }
        return results[index]
    }
    
    /// Identify outlier datasets using z-score
    public func outlierDatasets(threshold: Float = 2.0) -> [Int] {
        let means = results.map { $0.basic.mean }
        let overallMean = means.reduce(0, +) / Float(means.count)
        let variance = means.reduce(0) { sum, mean in
            let diff = mean - overallMean
            return sum + diff * diff
        } / Float(means.count)
        let stdDev = sqrt(variance)
        
        return means.enumerated().compactMap { index, mean in
            let zScore = abs(mean - overallMean) / stdDev
            return zScore > threshold ? index : nil
        }
    }
}
```

## Required Core Methods

### 1. Primary API Methods
```swift
/// Compute comprehensive statistics for single dataset
public func computeStatistics(
    data: [Float],
    config: StatisticsConfig = .default
) throws -> StatisticsResult

/// Compute statistics for multiple datasets
public func computeBatchStatistics(
    datasets: [[Float]],
    config: StatisticsConfig = .default
) throws -> BatchStatisticsResult

/// Compute correlation/covariance matrix
public func computeCorrelation(
    datasets: [[Float]]
) throws -> CorrelationResult
```

### 2. Specialized Methods
```swift
/// Basic statistics only (faster)
public func computeBasicStatistics(data: [Float]) throws -> BasicStatistics

/// Higher-order moments with pre-computed mean/variance
public func computeHigherMoments(
    data: [Float], 
    mean: Float, 
    variance: Float
) throws -> HigherMoments

/// Quantiles computation
public func computeQuantiles(
    data: [Float], 
    levels: [Float]
) throws -> QuantilesResult
```

### 3. Async Versions (Required)
```swift
public func computeStatisticsAsync(
    data: [Float],
    config: StatisticsConfig = .default
) async throws -> StatisticsResult

public func computeBatchStatisticsAsync(
    datasets: [[Float]],
    config: StatisticsConfig = .default
) async throws -> BatchStatisticsResult
```

### 4. VectorCore Integration (Required)
```swift
public func computeStatistics<V: VectorProtocol>(
    data: [V],
    config: StatisticsConfig = .default
) throws -> StatisticsResult where V.Scalar == Float

public func computeBatchStatistics<V: VectorProtocol>(
    datasets: [[V]],
    config: StatisticsConfig = .default
) throws -> BatchStatisticsResult where V.Scalar == Float
```

### 5. Performance Benchmarking (Required)
```swift
public func benchmark(
    dataSizes: [Int] = [1000, 10000, 100000, 1000000],
    configurations: [StatisticsConfig] = [.default]
) throws -> [BenchmarkResult]

public struct BenchmarkResult {
    public let dataSize: Int
    public let configuration: StatisticsConfig
    public let executionTime: TimeInterval
    public let throughputMEPS: Double        // Million elements per second
}
```

## GPU Implementation Requirements

### Metal Compute Kernels Needed

**1. basicStatistics kernel**
- Computes mean, variance, min, max, sum in single GPU pass
- Use parallel reduction with shared memory
- Input: data array, output: 6 float results [mean, variance, min, max, sum, count]

**2. higherMoments kernel**  
- Computes skewness and kurtosis given pre-computed mean/variance
- Input: data array, mean, variance; output: [skewness, kurtosis]

**3. quantileComputation kernel**
- Computes quantiles from sorted data
- Note: Data must be sorted on CPU first (use vDSP_vsort)
- Input: sorted data, quantile levels; output: quantile values

**4. correlationMatrix kernel**
- Computes correlation/covariance between multiple datasets
- Input: multiple datasets (flattened), output: correlation matrix

### Buffer Management Pattern
```swift
// Create buffers following this exact pattern:
guard let inputBuffer = device.makeBuffer(
    bytes: data,
    length: data.count * MemoryLayout<Float>.stride,
    options: .storageModeShared
) else {
    throw AccelerationError.bufferAllocationFailed(size: data.count * MemoryLayout<Float>.stride)
}

guard let outputBuffer = device.makeBuffer(
    length: resultSize,
    options: .storageModeShared
) else {
    throw AccelerationError.bufferAllocationFailed(size: resultSize)
}
```

### GPU Execution Pattern
```swift
// Create command buffer
guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else {
    throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
}

// Configure compute pass
encoder.setComputePipelineState(kernel)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)

// Set parameters
var params = SIMD2<UInt32>(UInt32(data.count), 0)
encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)

// Configure thread groups
let threadgroupSize = MTLSize(
    width: min(data.count, kernel.maxTotalThreadsPerThreadgroup),
    height: 1,
    depth: 1
)
let threadgroupCount = MTLSize(
    width: (data.count + threadgroupSize.width - 1) / threadgroupSize.width,
    height: 1,
    depth: 1
)

// Execute
let startTime = CACurrentMediaTime()
encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
encoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let executionTime = CACurrentMediaTime() - startTime

// Check for errors
if let error = commandBuffer.error {
    throw AccelerationError.computeFailed(reason: "Statistics computation failed: \\(error)")
}
```

## Algorithm Requirements

### Numerical Stability
- **Mean/Variance**: Use Welford's online algorithm for numerical stability
- **Higher Moments**: Use stable two-pass algorithm with pre-computed mean
- **Summation**: Consider Kahan summation for high precision when needed

### CPU Fallbacks
Use Accelerate framework for validation and CPU operations:
```swift
// Validate GPU results against Accelerate
var cpuMean: Float = 0
vDSP_meanv(data, 1, &cpuMean, vDSP_Length(data.count))

// Sort data for quantiles  
var sortedData = data
vDSP_vsort(&sortedData, vDSP_Length(data.count), 1)
```

## Testing Requirements

### Input Validation
- Empty arrays should throw `AccelerationError.invalidInput`
- Single-element arrays should return valid statistics
- All-identical values should handle gracefully
- NaN/infinite values should be detected and handled

### Edge Cases
```swift
// Test these scenarios:
let emptyData: [Float] = []                    // Should throw error
let singleElement: [Float] = [5.0]             // Valid statistics  
let constantData: [Float] = [3.0, 3.0, 3.0]   // Zero variance
let withNaN: [Float] = [1.0, Float.nan, 3.0]  // Handle NaN
```

### Performance Targets
- **Basic Statistics**: >50 MEPS for datasets >10K elements
- **Batch Processing**: Efficient memory reuse across datasets
- **Memory Usage**: Minimize allocations in hot paths

## Integration Notes

### VectorAccelerate Ecosystem
This kernel will work alongside existing kernels. Users might combine operations like:
```swift
// Example usage pattern
let stats = try statisticsKernel.computeStatistics(data: vectors)
let histogram = try histogramKernel.computeHistogram(data: vectors, 
                                                   config: .init(bins: 100))
```

### Error Handling Consistency
Follow the exact same error handling patterns as other kernels in the codebase. When in doubt, check how `MatrixMultiplyKernel` or `HistogramKernel` handle similar situations.

## Success Criteria

1. **Compilation**: Zero errors, zero warnings
2. **API Consistency**: Matches other kernel patterns exactly  
3. **Accuracy**: Results within 1e-6 of Accelerate framework results
4. **Performance**: Meets throughput targets
5. **Robustness**: Handles all edge cases gracefully
6. **Documentation**: Comprehensive inline documentation with examples

## File Dependencies Expected

The implementation should work with these existing types (already in codebase):
- `AccelerationError` enum with the exact cases listed above
- `VectorProtocol` from VectorCore with `.toArray()` method
- Metal compute infrastructure

**Note**: Do not assume any types exist beyond what's explicitly documented here. If you need additional types, define them within your implementation file.

---

**Delivery**: Complete Swift implementation in single file following all patterns above. The code should compile and integrate seamlessly with the existing VectorAccelerate v0.1.0 architecture.