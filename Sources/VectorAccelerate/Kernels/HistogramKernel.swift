// Histogram Kernel
// GPU-accelerated histogram computation with configurable binning strategies

import Metal
import Foundation
import VectorCore
import QuartzCore
import Accelerate

// MARK: - Histogram Kernel

/// GPU-accelerated histogram computation
/// Optimized for large-scale data analysis with multiple binning strategies
public final class HistogramKernel {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let uniformBinningKernel: any MTLComputePipelineState
    private let adaptiveBinningKernel: any MTLComputePipelineState
    private let logarithmicBinningKernel: any MTLComputePipelineState
    
    // MARK: - Configuration Types
    
    /// Binning strategy for histogram computation
    public enum BinningStrategy: Sendable {
        case uniform(bins: Int)                          // Equal-width bins
        case adaptive(bins: Int, method: AdaptiveMethod) // Adaptive bin sizes
        case logarithmic(bins: Int, base: Float)         // Logarithmic scaling
        case custom(edges: [Float])                      // Custom bin edges
        
        public enum AdaptiveMethod: Sendable {
            case equalFrequency    // Equal number of elements per bin
            case quantileBased     // Quantile-based binning
            case varianceMinimizing // Minimize within-bin variance
        }
        
        /// Number of bins for this strategy
        public var binCount: Int {
            switch self {
            case .uniform(let bins), .adaptive(let bins, _), .logarithmic(let bins, _):
                return bins
            case .custom(let edges):
                return max(1, edges.count - 1)
            }
        }
    }
    
    /// Configuration for histogram computation
    public struct HistogramConfig: Sendable {
        public let binningStrategy: BinningStrategy
        public let includeOutliers: Bool        // Include values outside range
        public let normalizeByTotal: Bool       // Normalize histogram to probabilities
        public let computeStatistics: Bool      // Compute additional statistics
        public let range: (min: Float, max: Float)? // Explicit range (nil = auto-detect)
        
        public init(
            binningStrategy: BinningStrategy = .uniform(bins: 100),
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
        
        public static let `default` = HistogramConfig()
    }
    
    // MARK: - Result Types
    
    /// Comprehensive histogram result with statistics
    public struct HistogramResult: Sendable {
        public let bins: [Float]                    // Bin counts or probabilities
        public let binEdges: [Float]                // Bin boundaries (n+1 edges for n bins)
        public let range: (min: Float, max: Float)  // Actual data range
        public let statistics: HistogramStatistics
        public let executionTime: TimeInterval
        
        /// Get bin centers for plotting
        public var binCenters: [Float] {
            guard binEdges.count > 1 else { return [] }
            return (0..<bins.count).map { i in
                (binEdges[i] + binEdges[i + 1]) / 2.0
            }
        }
        
        /// Get bin widths
        public var binWidths: [Float] {
            guard binEdges.count > 1 else { return [] }
            return (0..<bins.count).map { i in
                binEdges[i + 1] - binEdges[i]
            }
        }
        
        /// Get bin with maximum count
        public var peakBin: (index: Int, count: Float, center: Float)? {
            guard !bins.isEmpty else { return nil }
            let maxIndex = bins.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            let centers = binCenters
            return (maxIndex, bins[maxIndex], centers.isEmpty ? 0 : centers[maxIndex])
        }
        
        /// Get cumulative distribution
        public var cumulativeDistribution: [Float] {
            var cumSum: Float = 0
            return bins.map { count in
                cumSum += count
                return cumSum
            }
        }
        
        /// Find bin index for a given value
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
        
        /// Get histogram summary
        public func summary() -> String {
            return """
            Histogram Summary
            ================
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
    
    /// Statistical measures derived from histogram
    public struct HistogramStatistics: Sendable {
        public let totalCount: Float
        public let mean: Float
        public let standardDeviation: Float
        public let skewness: Float
        public let kurtosis: Float
        public let entropy: Float               // Shannon entropy
        public let mode: Float?                 // Most frequent value (bin center)
        public let median: Float?               // Estimated median
        public let quartiles: (q1: Float, q3: Float)?
        
        /// Coefficient of variation
        public var coefficientOfVariation: Float {
            mean != 0 ? standardDeviation / abs(mean) : 0
        }
        
        /// Check if distribution is approximately normal (using skewness/kurtosis)
        public var isApproximatelyNormal: Bool {
            abs(skewness) < 0.5 && abs(kurtosis - 3.0) < 0.5
        }
        
        /// Distribution shape classification
        public var distributionShape: DistributionShape {
            if abs(skewness) < 0.5 {
                return .symmetric
            } else if skewness > 0.5 {
                return .rightSkewed
            } else {
                return .leftSkewed
            }
        }
        
        public enum DistributionShape: String, Sendable {
            case symmetric = "Symmetric"
            case rightSkewed = "Right-skewed"
            case leftSkewed = "Left-skewed"
        }
    }
    
    /// Batch histogram result for multiple datasets
    public struct BatchHistogramResult: Sendable {
        public let histograms: [HistogramResult]
        public let totalExecutionTime: TimeInterval
        public let averageStatistics: HistogramStatistics
        
        /// Get histogram at index
        public func histogram(at index: Int) -> HistogramResult? {
            guard index >= 0 && index < histograms.count else { return nil }
            return histograms[index]
        }
        
        /// Compare distributions using statistical tests
        public func distributionSimilarity() -> [[Float]] {
            let count = histograms.count
            var similarity = Array(repeating: Array(repeating: Float(0), count: count), count: count)
            
            for i in 0..<count {
                for j in 0..<count {
                    if i == j {
                        similarity[i][j] = 1.0
                    } else {
                        // Use Jensen-Shannon divergence for similarity
                        similarity[i][j] = jensenShannonSimilarity(
                            histograms[i].bins,
                            histograms[j].bins
                        )
                    }
                }
            }
            
            return similarity
        }
        
        private func jensenShannonSimilarity(_ p: [Float], _ q: [Float]) -> Float {
            guard p.count == q.count else { return 0 }
            
            let pSum = p.reduce(0, +)
            let qSum = q.reduce(0, +)
            guard pSum > 0 && qSum > 0 else { return 0 }
            
            // Normalize distributions
            let pNorm = p.map { $0 / pSum }
            let qNorm = q.map { $0 / qSum }
            
            // Calculate JS divergence
            var jsDiv: Float = 0
            for i in 0..<p.count {
                let m = (pNorm[i] + qNorm[i]) / 2
                if pNorm[i] > 0 && m > 0 {
                    jsDiv += pNorm[i] * log2(pNorm[i] / m)
                }
                if qNorm[i] > 0 && m > 0 {
                    jsDiv += qNorm[i] * log2(qNorm[i] / m)
                }
            }
            jsDiv /= 2
            
            // Convert to similarity (0 = identical, 1 = completely different)
            return 1.0 - sqrt(jsDiv)
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }
        
        // Load histogram kernels
        guard let uniformFunc = library.makeFunction(name: "uniformHistogram") else {
            throw AccelerationError.shaderNotFound(name: "uniformHistogram")
        }
        
        guard let adaptiveFunc = library.makeFunction(name: "adaptiveHistogram") else {
            throw AccelerationError.shaderNotFound(name: "adaptiveHistogram")
        }
        
        guard let logFunc = library.makeFunction(name: "logarithmicHistogram") else {
            throw AccelerationError.shaderNotFound(name: "logarithmicHistogram")
        }
        
        do {
            self.uniformBinningKernel = try device.makeComputePipelineState(function: uniformFunc)
            self.adaptiveBinningKernel = try device.makeComputePipelineState(function: adaptiveFunc)
            self.logarithmicBinningKernel = try device.makeComputePipelineState(function: logFunc)
        } catch {
            throw AccelerationError.computeFailed(reason: "Failed to create histogram pipeline states: \(error)")
        }
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = uniformBinningKernel.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < 256 {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required threadgroup size for histogram computation"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Compute histogram for a single dataset
    /// - Parameters:
    ///   - data: Input data values
    ///   - config: Histogram configuration
    /// - Returns: Comprehensive histogram result
    public func computeHistogram(
        data: [Float],
        config: HistogramConfig = .default
    ) throws -> HistogramResult {
        guard !data.isEmpty else {
            throw AccelerationError.invalidInput("Input data cannot be empty")
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
        
        // Compute histogram using appropriate kernel
        let histogram = try computeHistogramGPU(
            data: data,
            binEdges: binEdges,
            strategy: config.binningStrategy,
            includeOutliers: config.includeOutliers
        )
        
        // Normalize if requested
        let normalizedHistogram = config.normalizeByTotal ? normalize(histogram) : histogram
        
        // Compute statistics if requested
        let statistics: HistogramStatistics
        if config.computeStatistics {
            statistics = try computeStatistics(
                from: normalizedHistogram,
                binEdges: binEdges,
                originalData: data
            )
        } else {
            statistics = HistogramStatistics(
                totalCount: normalizedHistogram.reduce(0, +),
                mean: 0, standardDeviation: 0, skewness: 0, kurtosis: 0,
                entropy: 0, mode: nil, median: nil, quartiles: nil
            )
        }
        
        let executionTime = CACurrentMediaTime() - startTime
        
        return HistogramResult(
            bins: normalizedHistogram,
            binEdges: binEdges,
            range: dataRange,
            statistics: statistics,
            executionTime: executionTime
        )
    }
    
    /// Compute histograms for multiple datasets
    /// - Parameters:
    ///   - datasets: Array of input datasets
    ///   - config: Histogram configuration
    /// - Returns: Batch histogram results
    public func computeHistograms(
        datasets: [[Float]],
        config: HistogramConfig = .default
    ) throws -> BatchHistogramResult {
        guard !datasets.isEmpty else {
            throw AccelerationError.invalidInput("Datasets cannot be empty")
        }
        
        let startTime = CACurrentMediaTime()
        var histograms: [HistogramResult] = []
        
        // Process each dataset
        for dataset in datasets {
            let result = try computeHistogram(data: dataset, config: config)
            histograms.append(result)
        }
        
        let totalExecutionTime = CACurrentMediaTime() - startTime
        
        // Compute average statistics
        let averageStatistics = computeAverageStatistics(from: histograms)
        
        return BatchHistogramResult(
            histograms: histograms,
            totalExecutionTime: totalExecutionTime,
            averageStatistics: averageStatistics
        )
    }
    
    // MARK: - Specialized Histogram Types
    
    /// Compute 2D histogram (joint distribution)
    public func compute2DHistogram(
        xData: [Float],
        yData: [Float],
        xBins: Int = 50,
        yBins: Int = 50
    ) throws -> Array2D<Float> {
        guard xData.count == yData.count else {
            throw AccelerationError.countMismatch(expected: xData.count, actual: yData.count)
        }
        
        // For now, implement using CPU fallback (GPU implementation would be more complex)
        var xMin: Float = 0, xMax: Float = 0
        var yMin: Float = 0, yMax: Float = 0
        
        vDSP_minv(xData, 1, &xMin, vDSP_Length(xData.count))
        vDSP_maxv(xData, 1, &xMax, vDSP_Length(xData.count))
        vDSP_minv(yData, 1, &yMin, vDSP_Length(yData.count))
        vDSP_maxv(yData, 1, &yMax, vDSP_Length(yData.count))
        
        let xWidth = (xMax - xMin) / Float(xBins)
        let yWidth = (yMax - yMin) / Float(yBins)
        
        var histogram = Array2D<Float>(rows: yBins, cols: xBins, initialValue: 0)
        
        for i in 0..<xData.count {
            let xBin = min(Int((xData[i] - xMin) / xWidth), xBins - 1)
            let yBin = min(Int((yData[i] - yMin) / yWidth), yBins - 1)
            histogram[yBin, xBin] += 1
        }
        
        return histogram
    }
    
    /// Compute histogram intersection for similarity measurement
    public func histogramIntersection(_ h1: HistogramResult, _ h2: HistogramResult) -> Float {
        guard h1.bins.count == h2.bins.count else { return 0 }
        
        var intersection: Float = 0
        for i in 0..<h1.bins.count {
            intersection += min(h1.bins[i], h2.bins[i])
        }
        
        return intersection
    }
    
    // MARK: - Async Operations
    
    /// Async version of histogram computation
    public func computeHistogramAsync(
        data: [Float],
        config: HistogramConfig = .default
    ) async throws -> HistogramResult {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try computeHistogram(data: data, config: config)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - VectorCore Integration
    
    /// Compute histogram using VectorCore protocol types
    public func computeHistogram<V: VectorProtocol>(
        data: [V],
        config: HistogramConfig = .default
    ) throws -> HistogramResult where V.Scalar == Float {
        let flatData = data.flatMap { $0.toArray() }
        return try computeHistogram(data: flatData, config: config)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark histogram computation for different data sizes
    public func benchmark(
        dataSizes: [Int],
        binCounts: [Int] = [50, 100, 200]
    ) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for dataSize in dataSizes {
            for binCount in binCounts {
                // Generate test data
                let testData = (0..<dataSize).map { _ in Float.random(in: -10...10) }
                let config = HistogramConfig(binningStrategy: .uniform(bins: binCount))
                
                // Warm-up
                _ = try computeHistogram(data: Array(testData.prefix(min(1000, dataSize))), config: config)
                
                // Benchmark runs
                var times: [TimeInterval] = []
                for _ in 0..<3 {
                    let result = try computeHistogram(data: testData, config: config)
                    times.append(result.executionTime)
                }
                
                let avgTime = times.reduce(0, +) / Double(times.count)
                let throughput = Double(dataSize) / avgTime / 1e6 // Million elements per second
                
                results.append(BenchmarkResult(
                    dataSize: dataSize,
                    binCount: binCount,
                    executionTime: avgTime,
                    throughputMEPS: throughput
                ))
            }
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let dataSize: Int
        public let binCount: Int
        public let executionTime: TimeInterval
        public let throughputMEPS: Double
    }
    
    // MARK: - Private Utilities
    
    /// Generate bin edges based on binning strategy
    private func generateBinEdges(
        for data: [Float],
        range: (min: Float, max: Float),
        strategy: BinningStrategy
    ) throws -> [Float] {
        switch strategy {
        case .uniform(let bins):
            return generateUniformBinEdges(range: range, bins: bins)
            
        case .adaptive(let bins, let method):
            return try generateAdaptiveBinEdges(data: data, bins: bins, method: method)
            
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
        method: BinningStrategy.AdaptiveMethod
    ) throws -> [Float] {
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
            // Simplified implementation - use equal frequency as approximation
            return try generateAdaptiveBinEdges(data: data, bins: bins, method: .equalFrequency)
        }
    }
    
    private func generateLogarithmicBinEdges(
        range: (min: Float, max: Float),
        bins: Int,
        base: Float
    ) throws -> [Float] {
        guard range.min > 0 else {
            throw AccelerationError.invalidInput("Logarithmic binning requires positive values")
        }
        
        let logMin = log(range.min) / log(base)
        let logMax = log(range.max) / log(base)
        let logWidth = (logMax - logMin) / Float(bins)
        
        return (0...bins).map { i in
            pow(base, logMin + Float(i) * logWidth)
        }
    }
    
    /// Compute histogram using GPU
    private func computeHistogramGPU(
        data: [Float],
        binEdges: [Float],
        strategy: BinningStrategy,
        includeOutliers: Bool
    ) throws -> [Float] {
        let binCount = binEdges.count - 1
        
        // Create buffers
        let dataSize = data.count * MemoryLayout<Float>.stride
        guard let dataBuffer = device.makeBuffer(
            bytes: data,
            length: dataSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: dataSize)
        }
        
        let edgesSize = binEdges.count * MemoryLayout<Float>.stride
        guard let edgesBuffer = device.makeBuffer(
            bytes: binEdges,
            length: edgesSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: edgesSize)
        }
        
        let histogramSize = binCount * MemoryLayout<UInt32>.stride
        guard let histogramBuffer = device.makeBuffer(
            length: histogramSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: histogramSize)
        }
        
        // Clear histogram buffer
        histogramBuffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: binCount)
        
        // Select appropriate kernel
        let kernel: any MTLComputePipelineState
        switch strategy {
        case .uniform:
            kernel = uniformBinningKernel
        case .adaptive:
            kernel = adaptiveBinningKernel
        case .logarithmic:
            kernel = logarithmicBinningKernel
        case .custom:
            kernel = uniformBinningKernel // Use uniform kernel for custom edges
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(kernel)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(edgesBuffer, offset: 0, index: 1)
        encoder.setBuffer(histogramBuffer, offset: 0, index: 2)
        
        // Set parameters
        var params = SIMD4<UInt32>(
            UInt32(data.count),
            UInt32(binCount),
            includeOutliers ? 1 : 0,
            0 // padding
        )
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        
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
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Histogram computation failed: \(error)")
        }
        
        // Extract results
        let histogramPointer = histogramBuffer.contents().bindMemory(to: UInt32.self, capacity: binCount)
        let histogram = Array(UnsafeBufferPointer(start: histogramPointer, count: binCount))
        
        return histogram.map { Float($0) }
    }
    
    /// Normalize histogram to probabilities
    private func normalize(_ histogram: [Float]) -> [Float] {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return histogram }
        return histogram.map { $0 / total }
    }
    
    /// Compute comprehensive statistics from histogram
    private func computeStatistics(
        from histogram: [Float],
        binEdges: [Float],
        originalData: [Float]
    ) throws -> HistogramStatistics {
        let totalCount = histogram.reduce(0, +)
        
        // Use original data for accurate statistics
        var mean: Float = 0
        vDSP_meanv(originalData, 1, &mean, vDSP_Length(originalData.count))
        
        // Standard deviation using Accelerate
        var variance: Float = 0
        var temp = originalData
        var result = [Float](repeating: 0, count: originalData.count)
        let meanVec = [Float](repeating: mean, count: originalData.count)
        vDSP_vsub(meanVec, 1, &temp, 1, &result, 1, vDSP_Length(originalData.count))
        vDSP_vsq(result, 1, &temp, 1, vDSP_Length(originalData.count))
        vDSP_meanv(temp, 1, &variance, vDSP_Length(originalData.count))
        let standardDeviation = sqrt(variance)
        
        // Higher-order moments for skewness and kurtosis
        let skewness = computeSkewness(data: originalData, mean: mean, stdDev: standardDeviation)
        let kurtosis = computeKurtosis(data: originalData, mean: mean, stdDev: standardDeviation)
        
        // Shannon entropy
        let entropy = computeEntropy(histogram: histogram)
        
        // Mode (bin center with highest count)
        let maxIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset
        let mode = maxIndex.map { (binEdges[$0] + binEdges[$0 + 1]) / 2 }
        
        // Estimated median and quartiles
        let (median, quartiles) = estimateQuantiles(histogram: histogram, binEdges: binEdges)
        
        return HistogramStatistics(
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
        
        // Find quantiles
        func findQuantile(_ q: Float) -> Float? {
            for i in 0..<cumulativeHist.count {
                if cumulativeHist[i] >= q {
                    // Linear interpolation within bin
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
    
    private func computeAverageStatistics(from histograms: [HistogramResult]) -> HistogramStatistics {
        guard !histograms.isEmpty else {
            return HistogramStatistics(
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
        
        return HistogramStatistics(
            totalCount: totalCount,
            mean: avgMean,
            standardDeviation: avgStdDev,
            skewness: avgSkewness,
            kurtosis: avgKurtosis,
            entropy: avgEntropy,
            mode: nil, // Cannot meaningfully average modes
            median: nil, // Cannot meaningfully average medians
            quartiles: nil // Cannot meaningfully average quartiles
        )
    }
}

// MARK: - Supporting Types

/// 2D array structure for 2D histograms
public struct Array2D<T> {
    public let rows: Int
    public let cols: Int
    private var data: [T]
    
    public init(rows: Int, cols: Int, initialValue: T) {
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: initialValue, count: rows * cols)
    }
    
    public subscript(row: Int, col: Int) -> T {
        get {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            return data[row * cols + col]
        }
        set {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            data[row * cols + col] = newValue
        }
    }
    
    public func asArray() -> [[T]] {
        var result: [[T]] = []
        for row in 0..<rows {
            var rowData: [T] = []
            for col in 0..<cols {
                rowData.append(self[row, col])
            }
            result.append(rowData)
        }
        return result
    }
}