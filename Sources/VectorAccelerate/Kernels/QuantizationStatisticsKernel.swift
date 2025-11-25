// Quantization Statistics Kernel
// GPU-accelerated quantization quality metrics with comprehensive statistical analysis

import Metal
import Foundation
import VectorCore
import QuartzCore
import Accelerate

// MARK: - Quantization Statistics Kernel

/// GPU-accelerated quantization statistics computation
/// Optimized for batch processing with comprehensive quality metrics
public final class QuantizationStatisticsKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLComputePipelineState
    
    // MARK: - Result Types
    
    /// Quality metrics for a single vector comparison
    public struct VectorMetrics: Sendable {
        public let mse: Float                    // Mean Squared Error
        public let psnr: Float                   // Peak Signal-to-Noise Ratio (dB)
        public let maxError: Float?              // Maximum absolute error
        public let executionTime: TimeInterval   // Individual computation time
        
        /// Root Mean Squared Error
        public var rmse: Float { sqrt(mse) }
        
        /// Signal-to-Noise Ratio in dB
        public var snr: Float { 
            guard mse > 0 else { return Float.infinity }
            return 10.0 * log10(1.0 / mse)
        }
        
        /// Quality classification based on PSNR
        public var qualityClass: QualityClass {
            if psnr.isInfinite { return .perfect }
            else if psnr > 40 { return .excellent }
            else if psnr > 30 { return .good }
            else if psnr > 20 { return .fair }
            else { return .poor }
        }
        
        /// Check if vectors are numerically identical
        public var isPerfectMatch: Bool { mse == 0.0 }
        
        public enum QualityClass: String, Sendable {
            case perfect = "Perfect"
            case excellent = "Excellent"
            case good = "Good" 
            case fair = "Fair"
            case poor = "Poor"
        }
    }
    
    /// Comprehensive batch statistics with statistical analysis
    public struct BatchStatistics: Sendable {
        public let perVectorMetrics: [VectorMetrics]
        public let averageMSE: Float
        public let averagePSNR: Float
        public let worstCaseMSE: Float           // Maximum MSE
        public let bestCaseMSE: Float            // Minimum MSE
        public let percentile95MSE: Float?       // 95th percentile MSE
        public let standardDeviationMSE: Float   // MSE standard deviation
        public let totalExecutionTime: TimeInterval
        
        /// Average Root Mean Squared Error
        public var averageRMSE: Float { sqrt(averageMSE) }
        
        /// Quality distribution across the batch
        public var qualityDistribution: [VectorMetrics.QualityClass: Int] {
            var distribution: [VectorMetrics.QualityClass: Int] = [:]
            for metric in perVectorMetrics {
                distribution[metric.qualityClass, default: 0] += 1
            }
            return distribution
        }
        
        /// Percentage of vectors with excellent quality (PSNR > 40 dB)
        public var excellentQualityPercentage: Float {
            let excellentCount = perVectorMetrics.filter { $0.qualityClass == .excellent || $0.qualityClass == .perfect }.count
            return Float(excellentCount) / Float(perVectorMetrics.count) * 100.0
        }
        
        /// Get vectors with MSE above threshold
        public func poorQualityIndices(mseThreshold: Float) -> [Int] {
            return perVectorMetrics.enumerated().compactMap { index, metric in
                metric.mse > mseThreshold ? index : nil
            }
        }
        
        /// Generate comprehensive quality report
        public func qualityReport() -> String {
            let distribution = qualityDistribution
            let perfectCount = distribution[.perfect] ?? 0
            let excellentCount = distribution[.excellent] ?? 0
            let goodCount = distribution[.good] ?? 0
            let fairCount = distribution[.fair] ?? 0
            let poorCount = distribution[.poor] ?? 0
            let total = perVectorMetrics.count
            
            return """
            Quantization Quality Report
            ==========================
            Total Vectors: \(total)
            Average MSE: \(String(format: "%.6f", averageMSE))
            Average PSNR: \(String(format: "%.2f", averagePSNR)) dB
            Average RMSE: \(String(format: "%.6f", averageRMSE))
            
            Quality Distribution:
            - Perfect (∞ dB):   \(perfectCount) (\(String(format: "%.1f", Float(perfectCount)/Float(total)*100))%)
            - Excellent (>40):  \(excellentCount) (\(String(format: "%.1f", Float(excellentCount)/Float(total)*100))%)
            - Good (30-40):     \(goodCount) (\(String(format: "%.1f", Float(goodCount)/Float(total)*100))%)
            - Fair (20-30):     \(fairCount) (\(String(format: "%.1f", Float(fairCount)/Float(total)*100))%)
            - Poor (<20):       \(poorCount) (\(String(format: "%.1f", Float(poorCount)/Float(total)*100))%)
            
            Statistical Summary:
            - Best MSE: \(String(format: "%.6f", bestCaseMSE))
            - Worst MSE: \(String(format: "%.6f", worstCaseMSE))
            - MSE Std Dev: \(String(format: "%.6f", standardDeviationMSE))
            \(percentile95MSE != nil ? "- 95th Percentile MSE: \(String(format: "%.6f", percentile95MSE!))" : "")
            
            Execution Time: \(String(format: "%.4f", totalExecutionTime))s
            """
        }
    }
    
    /// Configuration for statistics computation
    public struct StatisticsConfig: Sendable {
        public let computePercentiles: Bool      // Compute percentile statistics
        public let computeMaxError: Bool         // Compute maximum absolute error (CPU intensive)
        public let includeIndividualTiming: Bool // Track per-vector timing
        public let parallelCPUProcessing: Bool   // Use parallel CPU processing for max error
        
        public init(
            computePercentiles: Bool = true,
            computeMaxError: Bool = false,
            includeIndividualTiming: Bool = false,
            parallelCPUProcessing: Bool = true
        ) {
            self.computePercentiles = computePercentiles
            self.computeMaxError = computeMaxError
            self.includeIndividualTiming = includeIndividualTiming
            self.parallelCPUProcessing = parallelCPUProcessing
        }
        
        public static let `default` = StatisticsConfig()
        public static let comprehensive = StatisticsConfig(
            computePercentiles: true,
            computeMaxError: true,
            includeIndividualTiming: true,
            parallelCPUProcessing: true
        )
    }
    
    /// Comparison report for multiple quantization methods
    public struct ComparisonReport: Sendable {
        public let methodResults: [String: BatchStatistics]
        public let bestMethod: String?
        public let comparisonMatrix: [[Float]]   // MSE comparison matrix
        
        /// Get the method with lowest average MSE
        public var bestMethodByMSE: String? {
            return methodResults.min { $0.value.averageMSE < $1.value.averageMSE }?.key
        }
        
        /// Get the method with highest average PSNR
        public var bestMethodByPSNR: String? {
            return methodResults.max { $0.value.averagePSNR < $1.value.averagePSNR }?.key
        }
        
        /// Generate comparison summary
        public func comparisonSummary() -> String {
            var summary = "Quantization Method Comparison\n"
            summary += "==============================\n\n"
            
            for (method, stats) in methodResults.sorted(by: { $0.value.averageMSE < $1.value.averageMSE }) {
                summary += "\(method):\n"
                summary += "  Average MSE: \(String(format: "%.6f", stats.averageMSE))\n"
                summary += "  Average PSNR: \(String(format: "%.2f", stats.averagePSNR)) dB\n"
                summary += "  Excellent Quality: \(String(format: "%.1f", stats.excellentQualityPercentage))%\n\n"
            }
            
            if let best = bestMethodByMSE {
                summary += "Best Method (by MSE): \(best)\n"
            }
            
            return summary
        }
    }
    
    /// Real-time quality monitor for streaming quantization
    public final class QualityMonitor {
        private let windowSize: Int
        private var metricsWindow: [VectorMetrics] = []
        private let lock = NSLock()
        
        fileprivate init(windowSize: Int) {
            self.windowSize = windowSize
        }
        
        /// Add new metrics to the sliding window
        public func addMetrics(_ metrics: VectorMetrics) {
            lock.lock()
            defer { lock.unlock() }
            
            metricsWindow.append(metrics)
            if metricsWindow.count > windowSize {
                metricsWindow.removeFirst()
            }
        }
        
        /// Get current window statistics
        public func getCurrentStats() -> BatchStatistics? {
            lock.lock()
            defer { lock.unlock() }
            
            guard !metricsWindow.isEmpty else { return nil }
            
            return QuantizationStatisticsKernel.calculateBatchStatistics(
                metrics: metricsWindow,
                config: StatisticsConfig.default,
                totalExecutionTime: 0
            )
        }
        
        /// Check if quality is degrading (MSE trend)
        public func isQualityDegrading(threshold: Float = 0.1) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            
            guard metricsWindow.count >= 10 else { return false }
            
            let recent = Array(metricsWindow.suffix(5))
            let older = Array(metricsWindow.prefix(5))
            
            let recentAvgMSE = recent.reduce(0) { $0 + $1.mse } / Float(recent.count)
            let olderAvgMSE = older.reduce(0) { $0 + $1.mse } / Float(older.count)
            
            return recentAvgMSE > olderAvgMSE + threshold
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        guard let function = library.makeFunction(name: "computeQuantizationStats") else {
            throw AccelerationError.shaderNotFound(name: "computeQuantizationStats")
        }
        
        do {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            throw AccelerationError.computeFailed(reason: "Failed to create pipeline state: \(error)")
        }
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < 64 {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required threadgroup size for statistics computation"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Compute quantization statistics for vector batches
    /// - Parameters:
    ///   - original: Original float vectors
    ///   - quantized: Quantized float vectors
    ///   - config: Statistics computation configuration
    /// - Returns: Comprehensive batch statistics
    public func computeStatistics(
        original: [[Float]],
        quantized: [[Float]],
        config: StatisticsConfig = .default
    ) async throws -> BatchStatistics {
        guard original.count == quantized.count else {
            throw AccelerationError.countMismatch(expected: original.count, actual: quantized.count)
        }
        
        let numVectors = original.count
        guard numVectors > 0, let dimension = original.first?.count, dimension > 0 else {
            return BatchStatistics(
                perVectorMetrics: [],
                averageMSE: 0,
                averagePSNR: 0,
                worstCaseMSE: 0,
                bestCaseMSE: 0,
                percentile95MSE: nil,
                standardDeviationMSE: 0,
                totalExecutionTime: 0
            )
        }
        
        // Validate all vectors have same dimension
        for (_, vector) in original.enumerated() {
            guard vector.count == dimension else {
                throw AccelerationError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        for (_, vector) in quantized.enumerated() {
            guard vector.count == dimension else {
                throw AccelerationError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        
        let totalStartTime = CACurrentMediaTime()
        
        // Prepare input data
        let flatOriginal = original.flatMap { $0 }
        let flatQuantized = quantized.flatMap { $0 }
        
        // Create buffers
        let originalSize = flatOriginal.count * MemoryLayout<Float>.stride
        guard let originalBuffer = device.makeBuffer(
            bytes: flatOriginal,
            length: originalSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: originalSize)
        }
        
        let quantizedSize = flatQuantized.count * MemoryLayout<Float>.stride
        guard let quantizedBuffer = device.makeBuffer(
            bytes: flatQuantized,
            length: quantizedSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: quantizedSize)
        }
        
        let resultSize = numVectors * MemoryLayout<Float>.stride
        guard let mseBuffer = device.makeBuffer(length: resultSize, options: MTLResourceOptions.storageModeShared),
              let psnrBuffer = device.makeBuffer(length: resultSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: resultSize * 2)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(originalBuffer, offset: 0, index: 0)
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 1)
        encoder.setBuffer(mseBuffer, offset: 0, index: 2)
        encoder.setBuffer(psnrBuffer, offset: 0, index: 3)
        
        // Set parameters
        var params = SIMD2<UInt32>(UInt32(dimension), UInt32(numVectors))
        encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
        
        // Configure thread groups (one thread per vector)
        let threadgroupSize = MTLSize(
            width: min(numVectors, pipelineState.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        let threadgroupCount = MTLSize(
            width: (numVectors + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        // Execute GPU computation
        let gpuStartTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        let gpuExecutionTime = CACurrentMediaTime() - gpuStartTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Statistics computation failed: \(error)")
        }
        
        // Extract GPU results
        let msePointer = mseBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        let psnrPointer = psnrBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        
        let mseValues = Array(UnsafeBufferPointer(start: msePointer, count: numVectors))
        let psnrValues = Array(UnsafeBufferPointer(start: psnrPointer, count: numVectors))
        
        // Calculate max errors if requested (CPU intensive)
        var maxErrors: [Float]? = nil
        if config.computeMaxError {
            let maxErrorStartTime = CACurrentMediaTime()
            maxErrors = calculateMaxErrors(
                original: original,
                quantized: quantized,
                dimension: dimension,
                useParallel: config.parallelCPUProcessing
            )
            let maxErrorTime = CACurrentMediaTime() - maxErrorStartTime
            print("Max error calculation took \(String(format: "%.4f", maxErrorTime))s")
        }
        
        // Assemble per-vector metrics
        var metrics: [VectorMetrics] = []
        metrics.reserveCapacity(numVectors)
        
        let individualTiming = config.includeIndividualTiming ? gpuExecutionTime / Double(numVectors) : 0
        
        for i in 0..<numVectors {
            metrics.append(VectorMetrics(
                mse: mseValues[i],
                psnr: psnrValues[i],
                maxError: maxErrors?[i],
                executionTime: individualTiming
            ))
        }
        
        let totalExecutionTime = CACurrentMediaTime() - totalStartTime
        
        // Calculate batch statistics
        return Self.calculateBatchStatistics(
            metrics: metrics,
            config: config,
            totalExecutionTime: totalExecutionTime
        )
    }
    
    /// Compute statistics for single vector pair
    /// - Parameters:
    ///   - original: Original float vector
    ///   - quantized: Quantized float vector
    /// - Returns: Detailed vector metrics
    public func computeStatistics(
        original: [Float],
        quantized: [Float]
    ) async throws -> VectorMetrics {
        let config = StatisticsConfig.comprehensive
        let batchStats = try await computeStatistics(
            original: [original],
            quantized: [quantized],
            config: config
        )
        return batchStats.perVectorMetrics[0]
    }
    
    // MARK: - Method Comparison
    
    /// Compare multiple quantization methods
    /// - Parameters:
    ///   - original: Original float vectors
    ///   - quantizedVersions: Dictionary of method name to quantized vectors
    ///   - config: Statistics configuration
    /// - Returns: Comprehensive comparison report
    public func compareQuantizationMethods(
        original: [[Float]],
        quantizedVersions: [String: [[Float]]],
        config: StatisticsConfig = .default
    ) async throws -> ComparisonReport {
        var methodResults: [String: BatchStatistics] = [:]
        
        // Compute statistics for each method
        for (methodName, quantized) in quantizedVersions {
            let stats = try await computeStatistics(
                original: original,
                quantized: quantized,
                config: config
            )
            methodResults[methodName] = stats
        }
        
        // Find best method (lowest average MSE)
        let bestMethod = methodResults.min { $0.value.averageMSE < $1.value.averageMSE }?.key
        
        // Create comparison matrix (MSE comparison)
        let methods = Array(methodResults.keys.sorted())
        var comparisonMatrix: [[Float]] = []
        
        for method1 in methods {
            var row: [Float] = []
            for method2 in methods {
                let mse1 = methodResults[method1]!.averageMSE
                let mse2 = methodResults[method2]!.averageMSE
                // Ratio of MSEs (> 1 means method1 is worse than method2)
                row.append(mse2 > 0 ? mse1 / mse2 : 1.0)
            }
            comparisonMatrix.append(row)
        }
        
        return ComparisonReport(
            methodResults: methodResults,
            bestMethod: bestMethod,
            comparisonMatrix: comparisonMatrix
        )
    }
    
    /// Create quality monitor for streaming quantization
    /// - Parameter windowSize: Size of sliding window for quality tracking
    /// - Returns: Quality monitor instance
    public func createQualityMonitor(windowSize: Int = 100) -> QualityMonitor {
        return QualityMonitor(windowSize: windowSize)
    }
    
    // MARK: - Async Operations
    
    /// Async version of statistics computation
    public func computeStatisticsAsync(
        original: [[Float]],
        quantized: [[Float]],
        config: StatisticsConfig = .default
    ) async throws -> BatchStatistics {
        return try await computeStatistics(original: original, quantized: quantized, config: config)
    }
    
    // MARK: - VectorCore Integration
    
    /// Compute statistics using VectorCore protocol types
    public func computeStatistics<V: VectorProtocol>(
        original: [V],
        quantized: [V],
        config: StatisticsConfig = .default
    ) async throws -> BatchStatistics where V.Scalar == Float {
        let originalArrays = original.map { $0.toArray() }
        let quantizedArrays = quantized.map { $0.toArray() }
        return try await computeStatistics(original: originalArrays, quantized: quantizedArrays, config: config)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark statistics computation for different vector sizes
    public func benchmark(
        dimensions: [Int],
        vectorCounts: [Int] = [100, 500, 1000]
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for dimension in dimensions {
            for vectorCount in vectorCounts {
                // Generate test data
                let original = (0..<vectorCount).map { _ in
                    (0..<dimension).map { _ in Float.random(in: -1...1) }
                }
                let quantized = original.map { vector in
                    vector.map { value in round(value * 8) / 8 } // Simple 3-bit quantization
                }
                
                // Warm-up
                _ = try await computeStatistics(original: Array(original.prefix(10)), quantized: Array(quantized.prefix(10)))

                // Benchmark runs
                var times: [TimeInterval] = []
                for _ in 0..<3 {
                    let result = try await computeStatistics(original: original, quantized: quantized)
                    times.append(result.totalExecutionTime)
                }
                
                let avgTime = times.reduce(0, +) / Double(times.count)
                let throughput = Double(vectorCount * dimension) / avgTime / 1e6
                
                results.append(BenchmarkResult(
                    dimension: dimension,
                    vectorCount: vectorCount,
                    executionTime: avgTime,
                    throughputMEPS: throughput
                ))
            }
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let dimension: Int
        public let vectorCount: Int
        public let executionTime: TimeInterval
        public let throughputMEPS: Double
    }
    
    // MARK: - Private Utilities
    
    /// Calculate maximum absolute errors using optimized CPU processing
    private func calculateMaxErrors(
        original: [[Float]],
        quantized: [[Float]],
        dimension: Int,
        useParallel: Bool
    ) -> [Float] {
        let numVectors = original.count

        if useParallel && numVectors > 100 {
            // Use parallel processing with unsafe pointer for large batches
            // This is safe because we write to disjoint indices
            let maxErrors = UnsafeMutablePointer<Float>.allocate(capacity: numVectors)
            defer { maxErrors.deallocate() }

            // Initialize to zero
            maxErrors.initialize(repeating: 0.0, count: numVectors)

            // Wrap pointer in Sendable type for concurrent access
            // This is safe: each iteration writes to a unique index
            let sendablePointer = UnsafeSendable(maxErrors)

            // Parallel computation - each thread writes to its own index
            DispatchQueue.concurrentPerform(iterations: numVectors) { i in
                sendablePointer.value[i] = Self.calculateMaxErrorForVector(
                    original: original[i],
                    quantized: quantized[i],
                    dimension: dimension
                )
            }

            return Array(UnsafeBufferPointer(start: maxErrors, count: numVectors))
        } else {
            // Sequential processing for smaller batches
            var maxErrors = Array(repeating: Float(0), count: numVectors)
            for i in 0..<numVectors {
                maxErrors[i] = Self.calculateMaxErrorForVector(
                    original: original[i],
                    quantized: quantized[i],
                    dimension: dimension
                )
            }
            return maxErrors
        }
    }
    
    /// Calculate maximum error for a single vector pair using Accelerate
    private static func calculateMaxErrorForVector(
        original: [Float],
        quantized: [Float],
        dimension: Int
    ) -> Float {
        // Calculate absolute differences using vDSP
        var diff = [Float](repeating: 0, count: dimension)
        vDSP_vsub(quantized, 1, original, 1, &diff, 1, vDSP_Length(dimension))
        vDSP_vabs(diff, 1, &diff, 1, vDSP_Length(dimension))
        
        // Find maximum
        var maxError: Float = 0
        vDSP_maxv(diff, 1, &maxError, vDSP_Length(dimension))
        
        return maxError
    }
    
    /// Calculate comprehensive batch statistics
    private static func calculateBatchStatistics(
        metrics: [VectorMetrics],
        config: StatisticsConfig,
        totalExecutionTime: TimeInterval
    ) -> BatchStatistics {
        guard !metrics.isEmpty else {
            return BatchStatistics(
                perVectorMetrics: [],
                averageMSE: 0,
                averagePSNR: 0,
                worstCaseMSE: 0,
                bestCaseMSE: 0,
                percentile95MSE: nil,
                standardDeviationMSE: 0,
                totalExecutionTime: totalExecutionTime
            )
        }
        
        let mseValues = metrics.map { $0.mse }
        let psnrValues = metrics.map { $0.psnr }
        
        // Calculate MSE statistics using Accelerate
        var avgMSE: Float = 0
        var minMSE: Float = 0
        var maxMSE: Float = 0
        
        vDSP_meanv(mseValues, 1, &avgMSE, vDSP_Length(mseValues.count))
        vDSP_minv(mseValues, 1, &minMSE, vDSP_Length(mseValues.count))
        vDSP_maxv(mseValues, 1, &maxMSE, vDSP_Length(mseValues.count))
        
        // Calculate MSE standard deviation
        let variance = mseValues.reduce(0.0) { sum, mse in
            let diff = mse - avgMSE
            return sum + diff * diff
        } / Float(mseValues.count)
        let stdDev = sqrt(variance)
        
        // Calculate average PSNR (handle infinities)
        let finitePSNRs = psnrValues.filter { !$0.isInfinite }
        var avgPSNR: Float = 0
        
        if !finitePSNRs.isEmpty {
            vDSP_meanv(finitePSNRs, 1, &avgPSNR, vDSP_Length(finitePSNRs.count))
        } else if !psnrValues.isEmpty {
            avgPSNR = Float.infinity
        }
        
        // Calculate 95th percentile if requested
        var percentile95MSE: Float? = nil
        if config.computePercentiles {
            let sortedMSE = mseValues.sorted()
            let index = Int(Float(sortedMSE.count - 1) * 0.95)
            percentile95MSE = sortedMSE[max(0, min(index, sortedMSE.count - 1))]
        }
        
        return BatchStatistics(
            perVectorMetrics: metrics,
            averageMSE: avgMSE,
            averagePSNR: avgPSNR,
            worstCaseMSE: maxMSE,
            bestCaseMSE: minMSE,
            percentile95MSE: percentile95MSE,
            standardDeviationMSE: stdDev,
            totalExecutionTime: totalExecutionTime
        )
    }
}
