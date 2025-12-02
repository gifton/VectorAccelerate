//
//  Metal4HistogramKernel.swift
//  VectorAccelerate
//
//  Metal 4 Histogram kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 6a, Priority 6
//
//  Features:
//  - Multiple binning strategies (uniform, adaptive, logarithmic, custom)
//  - Comprehensive statistics computation
//  - 2D histogram support
//  - Batch processing for multiple datasets

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore
import Accelerate

// MARK: - Binning Strategy

/// Binning strategy for histogram computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4BinningStrategy: Sendable {
    case uniform(bins: Int)
    case adaptive(bins: Int, method: AdaptiveMethod)
    case logarithmic(bins: Int, base: Float)
    case custom(edges: [Float])

    public enum AdaptiveMethod: Sendable {
        case equalFrequency
        case quantileBased
        case varianceMinimizing
    }

    /// Number of bins for this strategy.
    public var binCount: Int {
        switch self {
        case .uniform(let bins), .adaptive(let bins, _), .logarithmic(let bins, _):
            return bins
        case .custom(let edges):
            return max(1, edges.count - 1)
        }
    }
}

// MARK: - Configuration

/// Configuration for histogram computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HistogramConfig: Sendable {
    /// Binning strategy
    public let binningStrategy: Metal4BinningStrategy
    /// Include values outside range
    public let includeOutliers: Bool
    /// Normalize histogram to probabilities
    public let normalizeByTotal: Bool
    /// Compute additional statistics
    public let computeStatistics: Bool
    /// Explicit range (nil = auto-detect)
    public let range: (min: Float, max: Float)?

    public init(
        binningStrategy: Metal4BinningStrategy = .uniform(bins: 100),
        includeOutliers: Bool = true,
        normalizeByTotal: Bool = false,
        computeStatistics: Bool = true,
        range: (min: Float, max: Float)? = nil
    ) {
        self.binningStrategy = binningStrategy
        self.includeOutliers = includeOutliers
        self.normalizeByTotal = normalizeByTotal
        self.computeStatistics = computeStatistics
        self.range = range
    }

    public static let `default` = Metal4HistogramConfig()
}

// MARK: - Statistics

/// Statistical measures derived from histogram.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HistogramStatistics: Sendable {
    public let totalCount: Float
    public let mean: Float
    public let standardDeviation: Float
    public let skewness: Float
    public let kurtosis: Float
    public let entropy: Float
    public let mode: Float?
    public let median: Float?
    public let quartiles: (q1: Float, q3: Float)?

    /// Coefficient of variation.
    public var coefficientOfVariation: Float {
        mean != 0 ? standardDeviation / abs(mean) : 0
    }

    /// Check if distribution is approximately normal.
    public var isApproximatelyNormal: Bool {
        abs(skewness) < 0.5 && abs(kurtosis - 3.0) < 0.5
    }

    /// Distribution shape classification.
    public var distributionShape: DistributionShape {
        if abs(skewness) < 0.5 { return .symmetric }
        else if skewness > 0.5 { return .rightSkewed }
        else { return .leftSkewed }
    }

    public enum DistributionShape: String, Sendable {
        case symmetric = "Symmetric"
        case rightSkewed = "Right-skewed"
        case leftSkewed = "Left-skewed"
    }
}

// MARK: - Result Types

/// Comprehensive histogram result with statistics.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HistogramResult: Sendable {
    /// Bin counts or probabilities
    public let bins: [Float]
    /// Bin boundaries (n+1 edges for n bins)
    public let binEdges: [Float]
    /// Actual data range
    public let range: (min: Float, max: Float)
    /// Statistical measures
    public let statistics: Metal4HistogramStatistics
    /// Execution time
    public let executionTime: TimeInterval

    /// Get bin centers for plotting.
    public var binCenters: [Float] {
        guard binEdges.count > 1 else { return [] }
        return (0..<bins.count).map { i in
            (binEdges[i] + binEdges[i + 1]) / 2.0
        }
    }

    /// Get bin widths.
    public var binWidths: [Float] {
        guard binEdges.count > 1 else { return [] }
        return (0..<bins.count).map { i in
            binEdges[i + 1] - binEdges[i]
        }
    }

    /// Get bin with maximum count.
    public var peakBin: (index: Int, count: Float, center: Float)? {
        guard !bins.isEmpty else { return nil }
        let maxIndex = bins.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let centers = binCenters
        return (maxIndex, bins[maxIndex], centers.isEmpty ? 0 : centers[maxIndex])
    }

    /// Get cumulative distribution.
    public var cumulativeDistribution: [Float] {
        var cumSum: Float = 0
        return bins.map { count in
            cumSum += count
            return cumSum
        }
    }

    /// Find bin index for a given value.
    public func binIndex(for value: Float) -> Int? {
        guard value >= range.min && value <= range.max else { return nil }

        for i in 0..<binEdges.count - 1 {
            if value >= binEdges[i] && value < binEdges[i + 1] {
                return i
            }
        }

        // Handle edge case for maximum value
        if value == range.max {
            return bins.count - 1
        }

        return nil
    }

    /// Generate summary report.
    public func summary() -> String {
        return """
        Histogram Summary
        =================
        Bins: \(bins.count)
        Range: [\(String(format: "%.6f", range.min)), \(String(format: "%.6f", range.max))]
        Total Count: \(String(format: "%.0f", statistics.totalCount))
        Peak: Bin \(peakBin?.index ?? -1) (\(String(format: "%.0f", peakBin?.count ?? 0)) counts)
        Mean: \(String(format: "%.6f", statistics.mean))
        Std Dev: \(String(format: "%.6f", statistics.standardDeviation))
        Execution Time: \(String(format: "%.4f", executionTime))s
        """
    }
}

/// Batch histogram result for multiple datasets.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchHistogramResult: Sendable {
    public let histograms: [Metal4HistogramResult]
    public let totalExecutionTime: TimeInterval
    public let averageStatistics: Metal4HistogramStatistics

    /// Get histogram at index.
    public func histogram(at index: Int) -> Metal4HistogramResult? {
        guard index >= 0 && index < histograms.count else { return nil }
        return histograms[index]
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Histogram kernel.
///
/// Computes histograms with configurable binning strategies and
/// comprehensive statistical analysis.
///
/// ## Binning Strategies
///
/// - **Uniform**: Equal-width bins across data range
/// - **Adaptive**: Data-driven bin sizes (equal frequency, quantile-based)
/// - **Logarithmic**: Logarithmic scaling for skewed data
/// - **Custom**: User-defined bin edges
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4HistogramKernel(context: context)
///
/// // Basic histogram
/// let result = try await kernel.computeHistogram(data: values)
/// print(result.summary())
///
/// // Custom binning
/// let config = Metal4HistogramConfig(
///     binningStrategy: .uniform(bins: 50),
///     normalizeByTotal: true
/// )
/// let normalized = try await kernel.computeHistogram(data: values, config: config)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4HistogramKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4HistogramKernel"

    // MARK: - Pipelines

    private let uniformPipeline: any MTLComputePipelineState
    private let adaptivePipeline: any MTLComputePipelineState
    private let logarithmicPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Histogram kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let uniformFunc = library.makeFunction(name: "uniformHistogram"),
              let adaptiveFunc = library.makeFunction(name: "adaptiveHistogram"),
              let logFunc = library.makeFunction(name: "logarithmicHistogram") else {
            throw VectorError.shaderNotFound(
                name: "Histogram kernels. Ensure StatisticsShaders.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.uniformPipeline = try await device.makeComputePipelineState(function: uniformFunc)
        self.adaptivePipeline = try await device.makeComputePipelineState(function: adaptiveFunc)
        self.logarithmicPipeline = try await device.makeComputePipelineState(function: logFunc)

        // Validate hardware support
        if uniformPipeline.maxTotalThreadsPerThreadgroup < 256 {
            throw VectorError.unsupportedGPUOperation(
                "Device does not support required threadgroup size for histogram computation"
            )
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Core Operations

    /// Compute histogram for a single dataset.
    public func computeHistogram(
        data: [Float],
        config: Metal4HistogramConfig = .default
    ) async throws -> Metal4HistogramResult {
        guard !data.isEmpty else {
            throw VectorError.invalidInput("Input data cannot be empty")
        }

        let startTime = CACurrentMediaTime()

        // Determine data range
        let dataRange: (min: Float, max: Float)
        if let explicitRange = config.range {
            dataRange = explicitRange
        } else {
            var minVal: Float = 0
            var maxVal: Float = 0
            vDSP_minv(data, 1, &minVal, vDSP_Length(data.count))
            vDSP_maxv(data, 1, &maxVal, vDSP_Length(data.count))
            dataRange = (minVal, maxVal)
        }

        // Generate bin edges based on strategy
        let binEdges = try generateBinEdges(
            for: data,
            range: dataRange,
            strategy: config.binningStrategy
        )

        // Compute histogram using GPU
        let histogram = try await computeHistogramGPU(
            data: data,
            binEdges: binEdges,
            strategy: config.binningStrategy,
            includeOutliers: config.includeOutliers
        )

        // Normalize if requested
        let normalizedHistogram = config.normalizeByTotal ? normalize(histogram) : histogram

        // Compute statistics if requested
        let statistics: Metal4HistogramStatistics
        if config.computeStatistics {
            statistics = computeStatistics(
                from: normalizedHistogram,
                binEdges: binEdges,
                originalData: data
            )
        } else {
            statistics = Metal4HistogramStatistics(
                totalCount: normalizedHistogram.reduce(0, +),
                mean: 0, standardDeviation: 0, skewness: 0, kurtosis: 0,
                entropy: 0, mode: nil, median: nil, quartiles: nil
            )
        }

        let executionTime = CACurrentMediaTime() - startTime

        return Metal4HistogramResult(
            bins: normalizedHistogram,
            binEdges: binEdges,
            range: dataRange,
            statistics: statistics,
            executionTime: executionTime
        )
    }

    /// Compute histograms for multiple datasets.
    public func computeHistograms(
        datasets: [[Float]],
        config: Metal4HistogramConfig = .default
    ) async throws -> Metal4BatchHistogramResult {
        guard !datasets.isEmpty else {
            throw VectorError.invalidInput("Datasets cannot be empty")
        }

        let startTime = CACurrentMediaTime()
        var histograms: [Metal4HistogramResult] = []
        histograms.reserveCapacity(datasets.count)

        for dataset in datasets {
            let result = try await computeHistogram(data: dataset, config: config)
            histograms.append(result)
        }

        let totalExecutionTime = CACurrentMediaTime() - startTime
        let averageStatistics = computeAverageStatistics(from: histograms)

        return Metal4BatchHistogramResult(
            histograms: histograms,
            totalExecutionTime: totalExecutionTime,
            averageStatistics: averageStatistics
        )
    }

    /// Compute histogram intersection for similarity measurement.
    public func histogramIntersection(_ h1: Metal4HistogramResult, _ h2: Metal4HistogramResult) -> Float {
        guard h1.bins.count == h2.bins.count else { return 0 }

        var intersection: Float = 0
        for i in 0..<h1.bins.count {
            intersection += min(h1.bins[i], h2.bins[i])
        }

        return intersection
    }

    // MARK: - VectorCore Integration

    /// Compute histogram using VectorCore protocol types.
    public func computeHistogram<V: VectorProtocol>(
        data: [V],
        config: Metal4HistogramConfig = .default
    ) async throws -> Metal4HistogramResult where V.Scalar == Float {
        let flatData = data.flatMap { v in v.withUnsafeBufferPointer { Array($0) } }
        return try await computeHistogram(data: flatData, config: config)
    }

    // MARK: - Private Implementation

    private func generateBinEdges(
        for data: [Float],
        range: (min: Float, max: Float),
        strategy: Metal4BinningStrategy
    ) throws -> [Float] {
        switch strategy {
        case .uniform(let bins):
            return generateUniformBinEdges(range: range, bins: bins)

        case .adaptive(let bins, let method):
            return generateAdaptiveBinEdges(data: data, bins: bins, method: method)

        case .logarithmic(let bins, let base):
            return try generateLogarithmicBinEdges(range: range, bins: bins, base: base)

        case .custom(let edges):
            return edges.sorted()
        }
    }

    private func generateUniformBinEdges(range: (min: Float, max: Float), bins: Int) -> [Float] {
        let width = (range.max - range.min) / Float(bins)
        return (0...bins).map { Float($0) * width + range.min }
    }

    private func generateAdaptiveBinEdges(
        data: [Float],
        bins: Int,
        method: Metal4BinningStrategy.AdaptiveMethod
    ) -> [Float] {
        let sortedData = data.sorted()
        let n = sortedData.count

        switch method {
        case .equalFrequency:
            var edges: [Float] = [sortedData[0]]
            for i in 1..<bins {
                let index = min(i * n / bins, n - 1)
                edges.append(sortedData[index])
            }
            edges.append(sortedData[n - 1])
            return edges

        case .quantileBased:
            var edges: [Float] = []
            for i in 0...bins {
                let quantile = Float(i) / Float(bins)
                let index = min(Int(quantile * Float(n - 1)), n - 1)
                edges.append(sortedData[index])
            }
            return edges

        case .varianceMinimizing:
            // Use equal frequency as approximation
            return generateAdaptiveBinEdges(data: data, bins: bins, method: .equalFrequency)
        }
    }

    private func generateLogarithmicBinEdges(
        range: (min: Float, max: Float),
        bins: Int,
        base: Float
    ) throws -> [Float] {
        guard range.min > 0 else {
            throw VectorError.invalidInput("Logarithmic binning requires positive values")
        }

        let logMin = log(range.min) / log(base)
        let logMax = log(range.max) / log(base)
        let logWidth = (logMax - logMin) / Float(bins)

        return (0...bins).map { i in
            pow(base, logMin + Float(i) * logWidth)
        }
    }

    private func computeHistogramGPU(
        data: [Float],
        binEdges: [Float],
        strategy: Metal4BinningStrategy,
        includeOutliers: Bool
    ) async throws -> [Float] {
        let binCount = binEdges.count - 1
        let device = context.device.rawDevice

        guard let dataBuffer = device.makeBuffer(
            bytes: data,
            length: data.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: data.count * MemoryLayout<Float>.size)
        }

        guard let edgesBuffer = device.makeBuffer(
            bytes: binEdges,
            length: binEdges.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: binEdges.count * MemoryLayout<Float>.size)
        }

        let histogramSize = binCount * MemoryLayout<UInt32>.size
        guard let histogramBuffer = device.makeBuffer(length: histogramSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: histogramSize)
        }

        // Clear histogram buffer
        histogramBuffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: binCount)

        // Select appropriate kernel
        let kernel: any MTLComputePipelineState
        switch strategy {
        case .uniform, .custom:
            kernel = uniformPipeline
        case .adaptive:
            kernel = adaptivePipeline
        case .logarithmic:
            kernel = logarithmicPipeline
        }

        let params = SIMD4<UInt32>(
            UInt32(data.count),
            UInt32(binCount),
            includeOutliers ? 1 : 0,
            0
        )

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(kernel)
            encoder.label = "Histogram"

            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(edgesBuffer, offset: 0, index: 1)
            encoder.setBuffer(histogramBuffer, offset: 0, index: 2)

            var localParams = params
            encoder.setBytes(&localParams, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)

            let threadgroupSize = MTLSize(
                width: min(data.count, kernel.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
            let threadgroups = MTLSize(
                width: (data.count + threadgroupSize.width - 1) / threadgroupSize.width,
                height: 1,
                depth: 1
            )

            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        }

        let histogramPtr = histogramBuffer.contents().bindMemory(to: UInt32.self, capacity: binCount)
        let histogram = Array(UnsafeBufferPointer(start: histogramPtr, count: binCount))

        return histogram.map { Float($0) }
    }

    private func normalize(_ histogram: [Float]) -> [Float] {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return histogram }
        return histogram.map { $0 / total }
    }

    private func computeStatistics(
        from histogram: [Float],
        binEdges: [Float],
        originalData: [Float]
    ) -> Metal4HistogramStatistics {
        let totalCount = histogram.reduce(0, +)

        // Use Accelerate for mean
        var mean: Float = 0
        vDSP_meanv(originalData, 1, &mean, vDSP_Length(originalData.count))

        // Standard deviation
        var variance: Float = 0
        var temp = originalData
        var result = [Float](repeating: 0, count: originalData.count)
        let meanVec = [Float](repeating: mean, count: originalData.count)
        vDSP_vsub(meanVec, 1, &temp, 1, &result, 1, vDSP_Length(originalData.count))
        vDSP_vsq(result, 1, &temp, 1, vDSP_Length(originalData.count))
        vDSP_meanv(temp, 1, &variance, vDSP_Length(originalData.count))
        let standardDeviation = sqrt(variance)

        // Higher-order moments
        let skewness = computeSkewness(data: originalData, mean: mean, stdDev: standardDeviation)
        let kurtosis = computeKurtosis(data: originalData, mean: mean, stdDev: standardDeviation)

        // Shannon entropy
        let entropy = computeEntropy(histogram: histogram)

        // Mode (bin center with highest count)
        let maxIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset
        let mode = maxIndex.map { (binEdges[$0] + binEdges[$0 + 1]) / 2 }

        // Estimated median and quartiles
        let (median, quartiles) = estimateQuantiles(histogram: histogram, binEdges: binEdges)

        return Metal4HistogramStatistics(
            totalCount: totalCount,
            mean: mean,
            standardDeviation: standardDeviation,
            skewness: skewness,
            kurtosis: kurtosis,
            entropy: entropy,
            mode: mode,
            median: median,
            quartiles: quartiles
        )
    }

    private func computeSkewness(data: [Float], mean: Float, stdDev: Float) -> Float {
        guard stdDev > 0 else { return 0 }
        let n = Float(data.count)
        var skewness: Float = 0

        for value in data {
            let z = (value - mean) / stdDev
            skewness += z * z * z
        }

        return skewness / n
    }

    private func computeKurtosis(data: [Float], mean: Float, stdDev: Float) -> Float {
        guard stdDev > 0 else { return 0 }
        let n = Float(data.count)
        var kurtosis: Float = 0

        for value in data {
            let z = (value - mean) / stdDev
            kurtosis += z * z * z * z
        }

        return kurtosis / n
    }

    private func computeEntropy(histogram: [Float]) -> Float {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return 0 }

        var entropy: Float = 0
        for count in histogram {
            if count > 0 {
                let p = count / total
                entropy -= p * log2(p)
            }
        }

        return entropy
    }

    private func estimateQuantiles(
        histogram: [Float],
        binEdges: [Float]
    ) -> (median: Float?, quartiles: (q1: Float, q3: Float)?) {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return (nil, nil) }

        // Build cumulative distribution
        var cumulative: Float = 0
        var cumulativeHist: [Float] = []
        for count in histogram {
            cumulative += count / total
            cumulativeHist.append(cumulative)
        }

        func findQuantile(_ q: Float) -> Float? {
            for i in 0..<cumulativeHist.count {
                if cumulativeHist[i] >= q {
                    let binCenter = (binEdges[i] + binEdges[i + 1]) / 2
                    return binCenter
                }
            }
            return nil
        }

        let median = findQuantile(0.5)
        let q1 = findQuantile(0.25)
        let q3 = findQuantile(0.75)

        let quartiles = (q1 != nil && q3 != nil) ? (q1!, q3!) : nil

        return (median, quartiles)
    }

    private func computeAverageStatistics(from histograms: [Metal4HistogramResult]) -> Metal4HistogramStatistics {
        guard !histograms.isEmpty else {
            return Metal4HistogramStatistics(
                totalCount: 0, mean: 0, standardDeviation: 0, skewness: 0, kurtosis: 0,
                entropy: 0, mode: nil, median: nil, quartiles: nil
            )
        }

        let stats = histograms.map { $0.statistics }
        let count = Float(stats.count)

        let avgMean = stats.reduce(0) { $0 + $1.mean } / count
        let avgStdDev = stats.reduce(0) { $0 + $1.standardDeviation } / count
        let avgSkewness = stats.reduce(0) { $0 + $1.skewness } / count
        let avgKurtosis = stats.reduce(0) { $0 + $1.kurtosis } / count
        let avgEntropy = stats.reduce(0) { $0 + $1.entropy } / count
        let totalCount = stats.reduce(0) { $0 + $1.totalCount }

        return Metal4HistogramStatistics(
            totalCount: totalCount,
            mean: avgMean,
            standardDeviation: avgStdDev,
            skewness: avgSkewness,
            kurtosis: avgKurtosis,
            entropy: avgEntropy,
            mode: nil,
            median: nil,
            quartiles: nil
        )
    }
}
