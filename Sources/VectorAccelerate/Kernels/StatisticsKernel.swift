import Metal
import Foundation
import VectorCore
import QuartzCore
import Accelerate
import simd

// MARK: - Data Structures

// 1. Configuration Structure
public struct StatisticsConfig: Sendable, Hashable {
    public let computeHigherMoments: Bool        // Include skewness/kurtosis
    public let computeQuantiles: Bool            // Include percentiles/quartiles
    public let quantileLevels: [Float]           // Custom quantile levels [0.0-1.0]
    public let biasCorrection: Bool              // Apply Bessel's correction (and sample corrections for moments)
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
        // Ensure levels are validated (0.0-1.0) and sorted
        self.quantileLevels = quantileLevels.filter { $0 >= 0.0 && $0 <= 1.0 }.sorted()
        self.biasCorrection = biasCorrection
        self.parallelReduction = parallelReduction
    }
    
    public static let `default` = StatisticsConfig()
}

// 2. Basic Statistics Result
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
        count > 0 ? standardDeviation / sqrt(Float(count)) : 0
    }
    
    /// 95% confidence interval for mean
    public func confidenceInterval(level: Float = 0.95) -> (lower: Float, upper: Float) {
        let tValue: Float = 1.96 // Approximate for large samples
        let margin = tValue * standardError
        return (mean - margin, mean + margin)
    }
}

// 3. Higher-Order Moments Result
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

// 4. Quantiles Result
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

// 5. Comprehensive Result
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
        var report = "Statistical Summary\n"
        report += "==================\n"
        report += "Count: \(basic.count)\n"
        report += "Mean: \(String(format: "%.6f", basic.mean))\n"
        report += "Std Dev: \(String(format: "%.6f", basic.standardDeviation))\n"
        report += "Range: [\(String(format: "%.6f", basic.minimum)), \(String(format: "%.6f", basic.maximum))]\n"
        
        if let moments = moments {
            report += "Skewness: \(String(format: "%.6f", moments.skewness))\n"
            report += "Kurtosis: \(String(format: "%.6f", moments.kurtosis))\n"
            report += "Distribution: \(moments.distributionShape.rawValue)\n"
        }
        
        if let quantiles = quantiles {
            if let median = quantiles.median {
                report += "Median: \(String(format: "%.6f", median))\n"
            }
            if let iqr = quantiles.iqr {
                report += "IQR: \(String(format: "%.6f", iqr))\n"
            }
        }
        
        report += "Execution Time: \(String(format: "%.4f", totalExecutionTime))s\n"
        return report
    }
}

// 6. Batch Processing Result
public struct BatchStatisticsResult: Sendable {
    public let results: [StatisticsResult]
    public let totalExecutionTime: TimeInterval
    // Statistics calculated over the means of the individual datasets.
    public let averageStatistics: BasicStatistics
    
    /// Get result at index
    public func result(at index: Int) -> StatisticsResult? {
        guard index >= 0 && index < results.count else { return nil }
        return results[index]
    }
    
    /// Identify outlier datasets using z-score
    public func outlierDatasets(threshold: Float = 2.0) -> [Int] {
        let overallMean = averageStatistics.mean
        let stdDev = averageStatistics.standardDeviation

        // If stdDev is zero, all means are identical, thus no outliers.
        guard stdDev > 0 else { return [] }

        return results.enumerated().compactMap { index, result in
            let mean = result.basic.mean
            let zScore = abs(mean - overallMean) / stdDev
            return zScore > threshold ? index : nil
        }
    }
}

// 7. Correlation Result
public struct CorrelationResult: Sendable {
    public let matrix: [[Float]]
    public let covarianceMatrix: [[Float]]
    public let executionTime: TimeInterval
}

// 8. Statistics Benchmark Result
public struct StatisticsBenchmarkResult: Sendable {
    public let dataSize: Int
    public let configuration: StatisticsConfig
    public let executionTime: TimeInterval
    public var throughputMEPS: Double {
        guard executionTime > 0 else { return 0 }
        return (Double(dataSize) / executionTime) / 1_000_000.0
    }
}

// MARK: - StatisticsKernel Implementation

public final class StatisticsKernel: @unchecked Sendable {
    // MARK: Properties
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let basicStatsKernel: any MTLComputePipelineState
    private let momentsKernel: any MTLComputePipelineState
    private let quantilesKernel: any MTLComputePipelineState
    private let correlationKernel: any MTLComputePipelineState

    // MARK: Initialization
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
        func loadFunction(name: String) throws -> any MTLFunction {
            guard let function = library.makeFunction(name: name) else {
                throw AccelerationError.shaderNotFound(name: name)
            }
            return function
        }

        let basicFunc = try loadFunction(name: "computeBasicStatistics")
        let momentsFunc = try loadFunction(name: "computeHigherMoments")
        let quantilesFunc = try loadFunction(name: "computeQuantiles")
        let correlationFunc = try loadFunction(name: "computeCorrelation")

        do {
            self.basicStatsKernel = try device.makeComputePipelineState(function: basicFunc)
            self.momentsKernel = try device.makeComputePipelineState(function: momentsFunc)
            self.quantilesKernel = try device.makeComputePipelineState(function: quantilesFunc)
            self.correlationKernel = try device.makeComputePipelineState(function: correlationFunc)
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
    
    // MARK: - Primary API Methods
    
    /// Compute comprehensive statistics for single dataset
    public func computeStatistics(
        data: [Float],
        config: StatisticsConfig = .default
    ) throws -> StatisticsResult {
        let totalStartTime = CACurrentMediaTime()
        
        try validateInputData(data)
        
        // 1. Basic Statistics
        let basicStats = try computeBasicStatisticsInternal(data: data, config: config)
        
        // 2. Higher-Order Moments
        var moments: HigherMoments? = nil
        if config.computeHigherMoments {
            moments = try computeHigherMomentsInternal(
                data: data,
                mean: basicStats.mean,
                variance: basicStats.variance,
                config: config
            )
        }
        
        // 3. Quantiles
        var quantiles: QuantilesResult? = nil
        if config.computeQuantiles && !config.quantileLevels.isEmpty {
            quantiles = try computeQuantilesInternal(data: data, levels: config.quantileLevels)
        }
        
        let totalExecutionTime = CACurrentMediaTime() - totalStartTime
        
        return StatisticsResult(
            basic: basicStats,
            moments: moments,
            quantiles: quantiles,
            totalExecutionTime: totalExecutionTime
        )
    }

    /// Compute statistics for multiple datasets
    public func computeBatchStatistics(
        datasets: [[Float]],
        config: StatisticsConfig = .default
    ) throws -> BatchStatisticsResult {
        guard !datasets.isEmpty else {
            throw AccelerationError.invalidInput("Datasets array cannot be empty.")
        }
        
        let totalStartTime = CACurrentMediaTime()
        
        var results: [StatisticsResult] = []
        results.reserveCapacity(datasets.count)
        
        // Process sequentially
        for (index, data) in datasets.enumerated() {
            do {
                let result = try computeStatistics(data: data, config: config)
                results.append(result)
            } catch {
                throw AccelerationError.computeFailed(reason: "Failed processing dataset at index \(index): \(error)")
            }
        }
        
        // Calculate statistics of the means
        let means = results.map { $0.basic.mean }
        let averageStatsConfig = StatisticsConfig(biasCorrection: false)
        let averageStats = try computeBasicStatisticsInternal(data: means, config: averageStatsConfig)

        let totalExecutionTime = CACurrentMediaTime() - totalStartTime
        
        return BatchStatisticsResult(
            results: results,
            totalExecutionTime: totalExecutionTime,
            averageStatistics: averageStats
        )
    }

    /// Compute correlation/covariance matrix
    public func computeCorrelation(
        datasets: [[Float]]
    ) throws -> CorrelationResult {
        guard datasets.count >= 2 else {
            throw AccelerationError.invalidInput("At least two datasets are required for correlation.")
        }
        
        let N = datasets[0].count
        guard N > 1 else {
             throw AccelerationError.invalidInput("Datasets must contain more than one element.")
        }
        let numDatasets = datasets.count

        // Validate dimensions and data
        for data in datasets {
            if data.count != N {
                throw AccelerationError.countMismatch(expected: N, actual: data.count)
            }
            try validateInputData(data)
        }
        
        // Flatten data for GPU buffer
        let flattenedData = datasets.flatMap { $0 }
        
        // Prepare buffers
        let dataBuffer = try createBuffer(data: flattenedData)
        let matrixSize = numDatasets * numDatasets
        let resultBuffer = try createBuffer(length: 2 * matrixSize * MemoryLayout<Float>.stride)
        
        // Configure parameters: [N (elements per dataset), numDatasets]
        var params = SIMD2<UInt32>(UInt32(N), UInt32(numDatasets))
        
        // Execute GPU kernel
        let executionTime = try withUnsafeMutableBytes(of: &params) { paramsPtr in
            try executeKernel(
                pipelineState: correlationKernel,
                buffers: [dataBuffer, resultBuffer],
                bytes: [(UnsafeRawPointer(paramsPtr.baseAddress!), MemoryLayout<SIMD2<UInt32>>.size)],
                gridSize: MTLSize(width: numDatasets, height: numDatasets, depth: 1)
            )
        }
        
        // Read results
        let resultArray = retrieveFloats(from: resultBuffer, count: 2 * matrixSize)
        
        // Deconstruct the flat array back into matrices
        var correlationMatrix: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numDatasets), count: numDatasets)
        var covarianceMatrix: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numDatasets), count: numDatasets)

        // Assuming row-major layout
        for i in 0..<numDatasets {
            for j in 0..<numDatasets {
                correlationMatrix[i][j] = resultArray[i * numDatasets + j]
                covarianceMatrix[i][j] = resultArray[matrixSize + i * numDatasets + j]
            }
        }
        
        return CorrelationResult(
            matrix: correlationMatrix,
            covarianceMatrix: covarianceMatrix,
            executionTime: executionTime
        )
    }
    
    // MARK: - Specialized Methods (Public API)

    /// Basic statistics only (faster)
    public func computeBasicStatistics(data: [Float]) throws -> BasicStatistics {
        try validateInputData(data)
        return try computeBasicStatisticsInternal(data: data, config: .default)
    }

    /// Higher-order moments with pre-computed mean/variance
    public func computeHigherMoments(
        data: [Float],
        mean: Float,
        variance: Float
    ) throws -> HigherMoments {
        try validateInputData(data)
        return try computeHigherMomentsInternal(data: data, mean: mean, variance: variance, config: .default)
    }

    /// Quantiles computation
    public func computeQuantiles(
        data: [Float],
        levels: [Float]
    ) throws -> QuantilesResult {
        try validateInputData(data)
        guard !levels.isEmpty else {
            throw AccelerationError.invalidInput("Quantile levels cannot be empty.")
        }
        // Ensure levels are sorted and valid
        let sortedLevels = levels.filter { $0 >= 0.0 && $0 <= 1.0 }.sorted()
        if sortedLevels.isEmpty {
             throw AccelerationError.invalidInput("No valid quantile levels provided (must be 0.0-1.0).")
        }
        return try computeQuantilesInternal(data: data, levels: sortedLevels)
    }
    
    // MARK: - Internal Implementations

    private func computeBasicStatisticsInternal(data: [Float], config: StatisticsConfig) throws -> BasicStatistics {
        let N = data.count
        
        // Handle edge case: Single element
        if N == 1 {
            return BasicStatistics(
                count: 1, mean: data[0], variance: 0.0, standardDeviation: 0.0,
                minimum: data[0], maximum: data[0], range: 0.0, sum: data[0], executionTime: 0.0
            )
        }

        // Prepare buffers
        let inputBuffer = try createBuffer(data: data)
        
        // Expected output from kernel: [mean, M2, min, max, sum, count_float]
        let outputCount = 6
        let outputBuffer = try createBuffer(length: outputCount * MemoryLayout<Float>.stride)
        
        // Configure parameters
        var params = SIMD2<UInt32>(UInt32(N), 0)

        // Configure grid for reduction
        let threadgroupSizeWidth = min(N, basicStatsKernel.maxTotalThreadsPerThreadgroup)
        
        // Execute GPU kernel
        let executionTime = try withUnsafeMutableBytes(of: &params) { paramsPtr in
            try executeKernel(
                pipelineState: basicStatsKernel,
                buffers: [inputBuffer, outputBuffer],
                bytes: [(UnsafeRawPointer(paramsPtr.baseAddress!), MemoryLayout<SIMD2<UInt32>>.size)],
                gridSize: MTLSize(width: N, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threadgroupSizeWidth, height: 1, depth: 1)
            )
        }

        // Read results
        let results = retrieveFloats(from: outputBuffer, count: outputCount)
        
        let mean = results[0]
        let M2 = results[1]
        let minimum = results[2]
        let maximum = results[3]
        let sum = results[4]
        
        // Calculate variance and standard deviation
        let denominator = config.biasCorrection ? Float(N - 1) : Float(N)
        let variance = M2 / denominator
        let standardDeviation = sqrt(max(0, variance))
        
        return BasicStatistics(
            count: N,
            mean: mean,
            variance: variance,
            standardDeviation: standardDeviation,
            minimum: minimum,
            maximum: maximum,
            range: maximum - minimum,
            sum: sum,
            executionTime: executionTime
        )
    }

    private func computeHigherMomentsInternal(
        data: [Float],
        mean: Float,
        variance: Float,
        config: StatisticsConfig
    ) throws -> HigherMoments {
        let N = data.count
        
        // Handle edge cases
        if N < 2 {
             return HigherMoments(
                skewness: 0.0, kurtosis: 0.0, excessKurtosis: -3.0, executionTime: 0.0
            )
        }
        
        // Determine population variance
        let n = Float(N)
        let populationVariance: Float

        if config.biasCorrection {
            populationVariance = variance * (n - 1) / n
        } else {
            populationVariance = variance
        }

        // Handle constant data
        if populationVariance < 1e-9 {
             return HigherMoments(
                skewness: 0.0, kurtosis: 0.0, excessKurtosis: -3.0, executionTime: 0.0
            )
        }

        // Prepare buffers
        let inputBuffer = try createBuffer(data: data)
        
        // Expected output: [M3_sum, M4_sum]
        let outputCount = 2
        let outputBuffer = try createBuffer(length: outputCount * MemoryLayout<Float>.stride)
        
        // Configure parameters
        struct MomentParams {
            var N: UInt32
            var mean: Float
        }
        var params = MomentParams(N: UInt32(N), mean: mean)

        // Execute GPU kernel
        let executionTime = try withUnsafeMutableBytes(of: &params) { paramsPtr in
            try executeKernel(
                pipelineState: momentsKernel,
                buffers: [inputBuffer, outputBuffer],
                bytes: [(UnsafeRawPointer(paramsPtr.baseAddress!), MemoryLayout<MomentParams>.size)],
                gridSize: MTLSize(width: N, height: 1, depth: 1)
            )
        }

        // Read results
        let results = retrieveFloats(from: outputBuffer, count: outputCount)
        
        let M3_sum = results[0]
        let M4_sum = results[1]
        
        // Calculate moments
        let skewness = (M3_sum / n) / pow(populationVariance, 1.5)
        let kurtosis = (M4_sum / n) / pow(populationVariance, 2)
        
        var finalSkewness = skewness
        var finalKurtosis = kurtosis

        // Apply bias correction
        if config.biasCorrection {
            if N >= 3 {
                finalSkewness = (sqrt(n * (n - 1)) / (n - 2)) * skewness
            }
            if N >= 4 {
                let term1 = (n*n - 1) / ((n - 2) * (n - 3))
                let term2 = (kurtosis - 3.0 + (6.0 / (n + 1)))
                finalKurtosis = term1 * term2 + 3.0
            }
        }

        return HigherMoments(
            skewness: finalSkewness,
            kurtosis: finalKurtosis,
            excessKurtosis: finalKurtosis - 3.0,
            executionTime: executionTime
        )
    }

    private func computeQuantilesInternal(
        data: [Float],
        levels: [Float]
    ) throws -> QuantilesResult {
        let startTime = CACurrentMediaTime()
        let N = data.count
        
        // Sort data on CPU using Accelerate
        var sortedData = data
        vDSP_vsort(&sortedData, vDSP_Length(N), 1)

        // Prepare buffers for GPU interpolation
        let dataBuffer = try createBuffer(data: sortedData)
        let levelsBuffer = try createBuffer(data: levels)
        let outputBuffer = try createBuffer(length: levels.count * MemoryLayout<Float>.stride)
        
        // Configure parameters
        var params = SIMD2<UInt32>(UInt32(N), UInt32(levels.count))
        
        // Execute GPU kernel
        _ = try withUnsafeMutableBytes(of: &params) { paramsPtr in
            try executeKernel(
                pipelineState: quantilesKernel,
                buffers: [dataBuffer, levelsBuffer, outputBuffer],
                bytes: [(UnsafeRawPointer(paramsPtr.baseAddress!), MemoryLayout<SIMD2<UInt32>>.size)],
                gridSize: MTLSize(width: levels.count, height: 1, depth: 1)
            )
        }
        
        // Read results
        let results = retrieveFloats(from: outputBuffer, count: levels.count)
        
        var quantileMap: [Float: Float] = [:]
        for (index, level) in levels.enumerated() {
            quantileMap[level] = results[index]
        }
        
        let totalExecutionTime = CACurrentMediaTime() - startTime
        
        return QuantilesResult(
            quantiles: quantileMap,
            executionTime: totalExecutionTime
        )
    }
    
    // MARK: - Async/Await Pattern

    public func computeStatisticsAsync(
        data: [Float],
        config: StatisticsConfig = .default
    ) async throws -> StatisticsResult {
        return try await Task.detached(priority: .userInitiated) { [self] in
            return try self.computeStatistics(data: data, config: config)
        }.value
    }

    public func computeBatchStatisticsAsync(
        datasets: [[Float]],
        config: StatisticsConfig = .default
    ) async throws -> BatchStatisticsResult {
        return try await Task.detached(priority: .userInitiated) { [self] in
            return try self.computeBatchStatistics(datasets: datasets, config: config)
        }.value
    }
    
    // MARK: - VectorCore Integration

    public func computeStatistics<V: VectorProtocol>(
        data: [V],
        config: StatisticsConfig = .default
    ) throws -> StatisticsResult where V.Scalar == Float {
        let floatArray = data.flatMap { $0.toArray() }
        return try computeStatistics(data: floatArray, config: config)
    }

    public func computeBatchStatistics<V: VectorProtocol>(
        datasets: [[V]],
        config: StatisticsConfig = .default
    ) throws -> BatchStatisticsResult where V.Scalar == Float {
        let floatArrays = datasets.map { innerArray in
            innerArray.flatMap { $0.toArray() }
        }
        return try computeBatchStatistics(datasets: floatArrays, config: config)
    }
    
    // MARK: - Performance Benchmarking

    public func benchmark(
        dataSizes: [Int] = [1000, 10000, 100000, 1000000],
        configurations: [StatisticsConfig] = [.default]
    ) throws -> [StatisticsBenchmarkResult] {
        var results: [StatisticsBenchmarkResult] = []
        let iterations = 5
        
        for size in dataSizes {
            // Generate synthetic data
            let data = (0..<size).map { _ in Float.random(in: -100.0...100.0) }
            
            for config in configurations {
                var averageTime: TimeInterval = 0
                
                // Warm-up run
                 _ = try? computeStatistics(data: data, config: config)

                for _ in 0..<iterations {
                    let result = try computeStatistics(data: data, config: config)
                    averageTime += result.totalExecutionTime
                }
                
                averageTime /= Double(iterations)
                
                results.append(StatisticsBenchmarkResult(
                    dataSize: size,
                    configuration: config,
                    executionTime: averageTime
                ))
            }
        }
        
        return results
    }

    // MARK: - Private Helpers

    // MARK: Input Validation
    private func validateInputData(_ data: [Float]) throws {
        if data.isEmpty {
            throw AccelerationError.invalidInput("Input data array cannot be empty.")
        }
        
        if data.contains(where: { !$0.isFinite }) {
            throw AccelerationError.invalidInput("Input data contains NaN or infinite values.")
        }
    }
    
    // MARK: Buffer Management
    
    private func createBuffer(data: [Float]) throws -> any MTLBuffer {
        let length = data.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: length,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: length)
        }
        return buffer
    }
    
    private func createBuffer(length: Int) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: length,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: length)
        }
        return buffer
    }
    
    // MARK: Data Retrieval
    private func retrieveFloats(from buffer: any MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    // MARK: GPU Execution Pattern
    
    private func executeKernel(
        pipelineState: any MTLComputePipelineState,
        buffers: [any MTLBuffer],
        bytes: [(ptr: UnsafeRawPointer, length: Int)],
        gridSize: MTLSize,
        threadgroupSize: MTLSize? = nil
    ) throws -> TimeInterval {
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(pipelineState)
        
        // Set buffers
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        // Set parameters
        var currentIndex = buffers.count
        for (ptr, length) in bytes {
            encoder.setBytes(ptr, length: length, index: currentIndex)
            currentIndex += 1
        }
        
        // Configure thread groups
        let tgSize: MTLSize
        if let providedSize = threadgroupSize {
            tgSize = providedSize
        } else {
            let maxThreads = pipelineState.maxTotalThreadsPerThreadgroup
            if gridSize.height == 1 && gridSize.depth == 1 {
                tgSize = MTLSize(width: min(gridSize.width, maxThreads), height: 1, depth: 1)
            } else {
                let side = Int(sqrt(Double(maxThreads)))
                tgSize = MTLSize(width: side, height: side, depth: 1)
            }
        }
        
        let threadgroupCount = MTLSize(
            width: (gridSize.width + tgSize.width - 1) / tgSize.width,
            height: (gridSize.height + tgSize.height - 1) / tgSize.height,
            depth: (gridSize.depth + tgSize.depth - 1) / tgSize.depth
        )

        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "GPU computation failed: \(error)")
        }
        
        return executionTime
    }
}