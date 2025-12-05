//
//  StatisticsKernel.swift
//  VectorAccelerate
//
//  Metal 4 Statistics kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 5, Priority 6
//
//  Features:
//  - Basic statistics (mean, variance, stddev, min, max, sum)
//  - Higher-order moments (skewness, kurtosis)
//  - Quantile computation with GPU interpolation
//  - Correlation and covariance matrices
//  - Batch processing for multiple datasets

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore
import Accelerate

// MARK: - Configuration

/// Configuration for statistics computation.
public struct Metal4StatisticsConfig: Sendable, Hashable {
    /// Compute higher-order moments (skewness, kurtosis)
    public let computeHigherMoments: Bool
    /// Compute quantiles (percentiles, quartiles)
    public let computeQuantiles: Bool
    /// Custom quantile levels [0.0-1.0]
    public let quantileLevels: [Float]
    /// Apply Bessel's correction for sample statistics
    public let biasCorrection: Bool

    public init(
        computeHigherMoments: Bool = true,
        computeQuantiles: Bool = false,
        quantileLevels: [Float] = [0.25, 0.5, 0.75, 0.95],
        biasCorrection: Bool = true
    ) {
        self.computeHigherMoments = computeHigherMoments
        self.computeQuantiles = computeQuantiles
        self.quantileLevels = quantileLevels.filter { $0 >= 0.0 && $0 <= 1.0 }.sorted()
        self.biasCorrection = biasCorrection
    }

    public static let `default` = Metal4StatisticsConfig()
    public static let basic = Metal4StatisticsConfig(computeHigherMoments: false, computeQuantiles: false)
    public static let full = Metal4StatisticsConfig(computeHigherMoments: true, computeQuantiles: true)
}

// MARK: - Result Types

/// Basic statistics result.
public struct Metal4BasicStatistics: Sendable {
    public let count: Int
    public let mean: Float
    public let variance: Float
    public let standardDeviation: Float
    public let minimum: Float
    public let maximum: Float
    public let range: Float
    public let sum: Float
    public let executionTime: TimeInterval

    /// Coefficient of variation (CV).
    public var coefficientOfVariation: Float {
        mean != 0 ? standardDeviation / Swift.abs(mean) : 0
    }

    /// Standard error of the mean.
    public var standardError: Float {
        count > 0 ? standardDeviation / Foundation.sqrt(Float(count)) : 0
    }

    /// 95% confidence interval for mean.
    public func confidenceInterval(level: Float = 0.95) -> (lower: Float, upper: Float) {
        let tValue: Float = 1.96  // Approximate for large samples
        let margin = tValue * standardError
        return (mean - margin, mean + margin)
    }
}

/// Higher-order moments result.
public struct Metal4HigherMoments: Sendable {
    public let skewness: Float
    public let kurtosis: Float
    public let excessKurtosis: Float
    public let executionTime: TimeInterval

    /// Distribution shape classification.
    public var distributionShape: DistributionShape {
        if Swift.abs(skewness) < 0.5 { return .symmetric }
        else if skewness > 0.5 { return .rightSkewed }
        else { return .leftSkewed }
    }

    /// Tail behavior classification.
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

/// Quantiles result.
public struct Metal4QuantilesResult: Sendable {
    public let quantiles: [Float: Float]
    public let executionTime: TimeInterval

    public var median: Float? { quantiles[0.5] }
    public var firstQuartile: Float? { quantiles[0.25] }
    public var thirdQuartile: Float? { quantiles[0.75] }
    public var percentile95: Float? { quantiles[0.95] }

    public var iqr: Float? {
        guard let q1 = firstQuartile, let q3 = thirdQuartile else { return nil }
        return q3 - q1
    }

    /// Outlier detection bounds (1.5 * IQR rule).
    public var outlierBounds: (lower: Float, upper: Float)? {
        guard let q1 = firstQuartile, let q3 = thirdQuartile else { return nil }
        let iqrVal = q3 - q1
        return (q1 - 1.5 * iqrVal, q3 + 1.5 * iqrVal)
    }
}

/// Comprehensive statistics result.
public struct Metal4StatisticsResult: Sendable {
    public let basic: Metal4BasicStatistics
    public let moments: Metal4HigherMoments?
    public let quantiles: Metal4QuantilesResult?
    public let totalExecutionTime: TimeInterval

    public var mean: Float { basic.mean }
    public var standardDeviation: Float { basic.standardDeviation }
    public var median: Float? { quantiles?.median }
    public var skewness: Float? { moments?.skewness }

    /// Whether distribution is approximately normal.
    public var isApproximatelyNormal: Bool {
        guard let skew = moments?.skewness,
              let excessKurt = moments?.excessKurtosis else { return false }
        return Swift.abs(skew) < 0.5 && Swift.abs(excessKurt) < 0.5
    }

    /// Generate summary report.
    public func summary() -> String {
        var report = "Statistical Summary\n"
        report += "==================\n"
        report += "Count: \(basic.count)\n"
        report += String(format: "Mean: %.6f\n", basic.mean)
        report += String(format: "Std Dev: %.6f\n", basic.standardDeviation)
        report += String(format: "Range: [%.6f, %.6f]\n", basic.minimum, basic.maximum)

        if let moments = moments {
            report += String(format: "Skewness: %.6f\n", moments.skewness)
            report += String(format: "Kurtosis: %.6f\n", moments.kurtosis)
            report += "Distribution: \(moments.distributionShape.rawValue)\n"
        }

        if let quantiles = quantiles {
            if let median = quantiles.median {
                report += String(format: "Median: %.6f\n", median)
            }
            if let iqr = quantiles.iqr {
                report += String(format: "IQR: %.6f\n", iqr)
            }
        }

        report += String(format: "Execution Time: %.4fs\n", totalExecutionTime)
        return report
    }
}

/// Correlation result.
public struct Metal4CorrelationResult: Sendable {
    public let matrix: [[Float]]
    public let covarianceMatrix: [[Float]]
    public let executionTime: TimeInterval
}

/// Batch statistics result.
public struct Metal4BatchStatisticsResult: Sendable {
    public let results: [Metal4StatisticsResult]
    public let totalExecutionTime: TimeInterval
    public let averageStatistics: Metal4BasicStatistics

    public func result(at index: Int) -> Metal4StatisticsResult? {
        guard index >= 0 && index < results.count else { return nil }
        return results[index]
    }

    /// Identify outlier datasets using z-score.
    public func outlierDatasets(threshold: Float = 2.0) -> [Int] {
        let overallMean = averageStatistics.mean
        let stdDev = averageStatistics.standardDeviation
        guard stdDev > 0 else { return [] }

        return results.enumerated().compactMap { index, result in
            let zScore = Swift.abs(result.basic.mean - overallMean) / stdDev
            return zScore > threshold ? index : nil
        }
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Statistics kernel.
///
/// Computes comprehensive statistics for datasets using GPU acceleration.
///
/// ## Capabilities
///
/// - **Basic**: Mean, variance, standard deviation, min, max, sum, range
/// - **Moments**: Skewness (3rd moment), kurtosis (4th moment)
/// - **Quantiles**: Percentiles, quartiles with GPU interpolation
/// - **Correlation**: Correlation and covariance matrices
///
/// ## Usage
///
/// ```swift
/// let kernel = try await StatisticsKernel(context: context)
///
/// // Basic statistics
/// let basic = try await kernel.computeBasicStatistics(data)
///
/// // Full analysis
/// let result = try await kernel.computeStatistics(data, config: .full)
/// print(result.summary())
///
/// // Correlation matrix
/// let corr = try await kernel.computeCorrelation(datasets: [x, y, z])
/// ```
public final class StatisticsKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "StatisticsKernel"

    // MARK: - Pipelines

    private let basicStatsPipeline: any MTLComputePipelineState
    private let momentsPipeline: any MTLComputePipelineState
    private let quantilesPipeline: any MTLComputePipelineState
    private let correlationPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Statistics kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let basicFunc = library.makeFunction(name: "computeBasicStatistics"),
              let momentsFunc = library.makeFunction(name: "computeHigherMoments"),
              let quantilesFunc = library.makeFunction(name: "computeQuantiles"),
              let correlationFunc = library.makeFunction(name: "computeCorrelation") else {
            throw VectorError.shaderNotFound(
                name: "Statistics kernels. Ensure StatisticsShaders.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.basicStatsPipeline = try await device.makeComputePipelineState(function: basicFunc)
        self.momentsPipeline = try await device.makeComputePipelineState(function: momentsFunc)
        self.quantilesPipeline = try await device.makeComputePipelineState(function: quantilesFunc)
        self.correlationPipeline = try await device.makeComputePipelineState(function: correlationFunc)

        // Validate hardware support
        let maxThreads = basicStatsPipeline.maxTotalThreadsPerThreadgroup
        if maxThreads < 256 {
            throw VectorError.unsupportedGPUOperation(
                "Device does not support required threadgroup size for statistics"
            )
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Primary API

    /// Compute comprehensive statistics for dataset.
    public func computeStatistics(
        _ data: [Float],
        config: Metal4StatisticsConfig = .default
    ) async throws -> Metal4StatisticsResult {
        let totalStartTime = CACurrentMediaTime()

        try validateInput(data)

        // 1. Basic Statistics
        let basicStats = try await computeBasicStatisticsInternal(data: data, config: config)

        // 2. Higher-Order Moments
        var moments: Metal4HigherMoments? = nil
        if config.computeHigherMoments {
            moments = try await computeHigherMomentsInternal(
                data: data,
                mean: basicStats.mean,
                variance: basicStats.variance,
                config: config
            )
        }

        // 3. Quantiles
        var quantiles: Metal4QuantilesResult? = nil
        if config.computeQuantiles && !config.quantileLevels.isEmpty {
            quantiles = try await computeQuantilesInternal(data: data, levels: config.quantileLevels)
        }

        let totalExecutionTime = CACurrentMediaTime() - totalStartTime

        return Metal4StatisticsResult(
            basic: basicStats,
            moments: moments,
            quantiles: quantiles,
            totalExecutionTime: totalExecutionTime
        )
    }

    /// Compute basic statistics only (faster).
    public func computeBasicStatistics(_ data: [Float]) async throws -> Metal4BasicStatistics {
        try validateInput(data)
        return try await computeBasicStatisticsInternal(data: data, config: .default)
    }

    /// Compute higher-order moments with pre-computed mean/variance.
    public func computeHigherMoments(
        _ data: [Float],
        mean: Float,
        variance: Float
    ) async throws -> Metal4HigherMoments {
        try validateInput(data)
        return try await computeHigherMomentsInternal(data: data, mean: mean, variance: variance, config: .default)
    }

    /// Compute quantiles.
    public func computeQuantiles(
        _ data: [Float],
        levels: [Float]
    ) async throws -> Metal4QuantilesResult {
        try validateInput(data)
        guard !levels.isEmpty else {
            throw VectorError.invalidInput("Quantile levels cannot be empty")
        }
        let sortedLevels = levels.filter { $0 >= 0.0 && $0 <= 1.0 }.sorted()
        if sortedLevels.isEmpty {
            throw VectorError.invalidInput("No valid quantile levels (must be 0.0-1.0)")
        }
        return try await computeQuantilesInternal(data: data, levels: sortedLevels)
    }

    /// Compute statistics for multiple datasets.
    public func computeBatchStatistics(
        _ datasets: [[Float]],
        config: Metal4StatisticsConfig = .default
    ) async throws -> Metal4BatchStatisticsResult {
        guard !datasets.isEmpty else {
            throw VectorError.invalidInput("Datasets array cannot be empty")
        }

        let totalStartTime = CACurrentMediaTime()

        var results: [Metal4StatisticsResult] = []
        results.reserveCapacity(datasets.count)

        for (index, data) in datasets.enumerated() {
            do {
                let result = try await computeStatistics(data, config: config)
                results.append(result)
            } catch {
                throw VectorError.computeFailed(reason: "Failed processing dataset at index \(index): \(error)")
            }
        }

        // Calculate statistics of the means
        let means = results.map { $0.basic.mean }
        let averageStatsConfig = Metal4StatisticsConfig(biasCorrection: false)
        let averageStats = try await computeBasicStatisticsInternal(data: means, config: averageStatsConfig)

        let totalExecutionTime = CACurrentMediaTime() - totalStartTime

        return Metal4BatchStatisticsResult(
            results: results,
            totalExecutionTime: totalExecutionTime,
            averageStatistics: averageStats
        )
    }

    /// Compute correlation/covariance matrix.
    public func computeCorrelation(datasets: [[Float]]) async throws -> Metal4CorrelationResult {
        guard datasets.count >= 2 else {
            throw VectorError.invalidInput("At least two datasets required for correlation")
        }

        let n = datasets[0].count
        guard n > 1 else {
            throw VectorError.invalidInput("Datasets must contain more than one element")
        }
        let numDatasets = datasets.count

        // Validate dimensions
        for data in datasets {
            if data.count != n {
                throw VectorError.countMismatch(expected: n, actual: data.count)
            }
            try validateInput(data)
        }

        let device = context.device.rawDevice
        let flattenedData = datasets.flatMap { $0 }

        guard let dataBuffer = device.makeBuffer(
            bytes: flattenedData,
            length: flattenedData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flattenedData.count * MemoryLayout<Float>.size)
        }
        dataBuffer.label = "Correlation.input"

        let matrixSize = numDatasets * numDatasets
        let resultSize = 2 * matrixSize * MemoryLayout<Float>.size
        guard let resultBuffer = device.makeBuffer(length: resultSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: resultSize)
        }
        resultBuffer.label = "Correlation.output"

        let params = SIMD2<UInt32>(UInt32(n), UInt32(numDatasets))

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(correlationPipeline)
            encoder.label = "Correlation"

            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(resultBuffer, offset: 0, index: 1)
            var localParams = params
            encoder.setBytes(&localParams, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)

            let side = Int(Foundation.sqrt(Double(correlationPipeline.maxTotalThreadsPerThreadgroup)))
            let threadgroupSize = MTLSize(width: side, height: side, depth: 1)
            let threadgroupCount = MTLSize(
                width: (numDatasets + side - 1) / side,
                height: (numDatasets + side - 1) / side,
                depth: 1
            )

            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Extract results
        let resultPtr = resultBuffer.contents().bindMemory(to: Float.self, capacity: 2 * matrixSize)
        var correlationMatrix: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numDatasets), count: numDatasets)
        var covarianceMatrix: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numDatasets), count: numDatasets)

        for i in 0..<numDatasets {
            for j in 0..<numDatasets {
                correlationMatrix[i][j] = resultPtr[i * numDatasets + j]
                covarianceMatrix[i][j] = resultPtr[matrixSize + i * numDatasets + j]
            }
        }

        return Metal4CorrelationResult(
            matrix: correlationMatrix,
            covarianceMatrix: covarianceMatrix,
            executionTime: executionTime
        )
    }

    // MARK: - VectorCore Integration

    /// Compute statistics for VectorProtocol types.
    public func computeStatistics<V: VectorProtocol>(
        _ data: [V],
        config: Metal4StatisticsConfig = .default
    ) async throws -> Metal4StatisticsResult where V.Scalar == Float {
        let floatArray = data.flatMap { $0.toArray() }
        return try await computeStatistics(floatArray, config: config)
    }

    // MARK: - Private Implementation

    private func computeBasicStatisticsInternal(
        data: [Float],
        config: Metal4StatisticsConfig
    ) async throws -> Metal4BasicStatistics {
        let n = data.count

        // Edge case: Single element
        if n == 1 {
            return Metal4BasicStatistics(
                count: 1,
                mean: data[0],
                variance: 0.0,
                standardDeviation: 0.0,
                minimum: data[0],
                maximum: data[0],
                range: 0.0,
                sum: data[0],
                executionTime: 0.0
            )
        }

        let device = context.device.rawDevice

        guard let inputBuffer = device.makeBuffer(
            bytes: data,
            length: data.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: data.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "BasicStats.input"

        // Output: [mean, M2, min, max, sum, count_float]
        let outputCount = 6
        let outputSize = outputCount * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "BasicStats.output"

        let params = SIMD2<UInt32>(UInt32(n), 0)
        // Use power-of-2 threadgroup size for correct parallel reduction
        let maxThreads = basicStatsPipeline.maxTotalThreadsPerThreadgroup
        let threadgroupSizeWidth: Int = {
            var size = 1
            while size * 2 <= min(n, maxThreads) {
                size *= 2
            }
            // Ensure at least enough threads to cover all elements
            if size < n && size * 2 <= maxThreads {
                size *= 2
            }
            return size
        }()

        let threadgroupSize = MTLSize(width: threadgroupSizeWidth, height: 1, depth: 1)
        let threadgroupCount = MTLSize(width: 1, height: 1, depth: 1)  // Single threadgroup for reduction

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(basicStatsPipeline)
            encoder.label = "BasicStatistics"

            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            var localParams = params
            encoder.setBytes(&localParams, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)

            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        }
        let executionTime = CACurrentMediaTime() - startTime

        let resultPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        let mean = resultPtr[0]
        let m2 = resultPtr[1]
        let minimum = resultPtr[2]
        let maximum = resultPtr[3]
        let sum = resultPtr[4]

        let denominator = config.biasCorrection ? Float(n - 1) : Float(n)
        let variance = m2 / denominator
        let standardDeviation = Foundation.sqrt(max(0, variance))

        return Metal4BasicStatistics(
            count: n,
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
        config: Metal4StatisticsConfig
    ) async throws -> Metal4HigherMoments {
        let n = data.count

        if n < 2 {
            return Metal4HigherMoments(
                skewness: 0.0,
                kurtosis: 0.0,
                excessKurtosis: -3.0,
                executionTime: 0.0
            )
        }

        let floatN = Float(n)
        let populationVariance: Float
        if config.biasCorrection {
            populationVariance = variance * (floatN - 1) / floatN
        } else {
            populationVariance = variance
        }

        if populationVariance < 1e-9 {
            return Metal4HigherMoments(
                skewness: 0.0,
                kurtosis: 0.0,
                excessKurtosis: -3.0,
                executionTime: 0.0
            )
        }

        let device = context.device.rawDevice

        guard let inputBuffer = device.makeBuffer(
            bytes: data,
            length: data.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: data.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "Moments.input"

        // Output: [M3_sum, M4_sum]
        let outputCount = 2
        let outputSize = outputCount * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "Moments.output"

        struct MomentParams {
            var n: UInt32
            var mean: Float
        }
        let params = MomentParams(n: UInt32(n), mean: mean)

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(momentsPipeline)
            encoder.label = "HigherMoments"

            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            var localParams = params
            encoder.setBytes(&localParams, length: MemoryLayout<MomentParams>.size, index: 2)

            let config = Metal4ThreadConfiguration.linear(count: n, pipeline: momentsPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
        let executionTime = CACurrentMediaTime() - startTime

        let resultPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        let m3Sum = resultPtr[0]
        let m4Sum = resultPtr[1]

        var skewness = (m3Sum / floatN) / pow(populationVariance, 1.5)
        var kurtosis = (m4Sum / floatN) / pow(populationVariance, 2)

        // Apply bias correction
        if config.biasCorrection {
            if n >= 3 {
                skewness = (Foundation.sqrt(floatN * (floatN - 1)) / (floatN - 2)) * skewness
            }
            if n >= 4 {
                let term1 = (floatN * floatN - 1) / ((floatN - 2) * (floatN - 3))
                let term2 = (kurtosis - 3.0 + (6.0 / (floatN + 1)))
                kurtosis = term1 * term2 + 3.0
            }
        }

        return Metal4HigherMoments(
            skewness: skewness,
            kurtosis: kurtosis,
            excessKurtosis: kurtosis - 3.0,
            executionTime: executionTime
        )
    }

    private func computeQuantilesInternal(
        data: [Float],
        levels: [Float]
    ) async throws -> Metal4QuantilesResult {
        let startTime = CACurrentMediaTime()
        let n = data.count

        // Sort data on CPU using Accelerate
        var sortedData = data
        vDSP_vsort(&sortedData, vDSP_Length(n), 1)

        let device = context.device.rawDevice

        guard let dataBuffer = device.makeBuffer(
            bytes: sortedData,
            length: sortedData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: sortedData.count * MemoryLayout<Float>.size)
        }
        dataBuffer.label = "Quantiles.sortedData"

        guard let levelsBuffer = device.makeBuffer(
            bytes: levels,
            length: levels.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: levels.count * MemoryLayout<Float>.size)
        }
        levelsBuffer.label = "Quantiles.levels"

        let outputSize = levels.count * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "Quantiles.output"

        let params = SIMD2<UInt32>(UInt32(n), UInt32(levels.count))

        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(quantilesPipeline)
            encoder.label = "Quantiles"

            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(levelsBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            var localParams = params
            encoder.setBytes(&localParams, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: levels.count, pipeline: quantilesPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }

        let totalExecutionTime = CACurrentMediaTime() - startTime

        let resultPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: levels.count)
        var quantileMap: [Float: Float] = [:]
        for (index, level) in levels.enumerated() {
            quantileMap[level] = resultPtr[index]
        }

        return Metal4QuantilesResult(
            quantiles: quantileMap,
            executionTime: totalExecutionTime
        )
    }

    private func validateInput(_ data: [Float]) throws {
        if data.isEmpty {
            throw VectorError.invalidInput("Input data array cannot be empty")
        }
        if data.contains(where: { !$0.isFinite }) {
            throw VectorError.invalidInput("Input data contains NaN or infinite values")
        }
    }
}
