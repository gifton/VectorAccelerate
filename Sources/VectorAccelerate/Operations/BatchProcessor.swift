//
//  BatchProcessor.swift
//  VectorAccelerate
//
//  Scalable batch operation coordinator with adaptive processing
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Batch processing strategy
public enum BatchStrategy: Sendable {
    case sequential
    case parallel(maxConcurrency: Int)
    case adaptive  // Automatically choose based on workload
    case streaming(chunkSize: Int)  // For memory-constrained processing
}

/// Batch processing configuration
public struct BatchConfiguration: Sendable {
    public let strategy: BatchStrategy
    public let maxBatchSize: Int
    public let memoryLimit: Int  // Bytes
    public let enablePipelining: Bool

    /// Minimum batch size for GPU (fallback when decisionEngine is nil)
    @available(*, deprecated, message: "Use decisionEngine for adaptive GPU routing")
    public let gpuThreshold: Int

    /// Optional decision engine for adaptive GPU/CPU routing.
    /// When provided, GPU decisions are delegated to the engine.
    /// When nil, falls back to gpuThreshold.
    public let decisionEngine: GPUDecisionEngine?

    public init(
        strategy: BatchStrategy = .adaptive,
        maxBatchSize: Int = 10000,
        memoryLimit: Int = 512 * 1024 * 1024,  // 512MB
        enablePipelining: Bool = true,
        gpuThreshold: Int = 100,
        decisionEngine: GPUDecisionEngine? = nil
    ) {
        self.strategy = strategy
        self.maxBatchSize = maxBatchSize
        self.memoryLimit = memoryLimit
        self.enablePipelining = enablePipelining
        self.gpuThreshold = gpuThreshold
        self.decisionEngine = decisionEngine
    }

    public static let `default` = BatchConfiguration()
    public static let memory = BatchConfiguration(
        strategy: .streaming(chunkSize: 1000),
        memoryLimit: 256 * 1024 * 1024
    )
    public static let performance = BatchConfiguration(
        strategy: .parallel(maxConcurrency: ProcessInfo.processInfo.activeProcessorCount),
        memoryLimit: 1024 * 1024 * 1024
    )
}

/// Result of batch operation
public struct BatchResult<T: Sendable>: Sendable {
    public let results: [T]
    public let processingTime: TimeInterval
    public let chunksProcessed: Int
    public let memoryUsed: Int
}

/// Batch operation protocol
public protocol BatchOperation: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable
    
    func process(_ batch: [Input]) async throws -> [Output]
    var estimatedMemoryPerItem: Int { get }
}

/// Main batch processor using Metal 4
public actor BatchProcessor {
    private let context: Metal4Context?
    private let simdFallback: SIMDFallback
    private let configuration: BatchConfiguration
    private let logger: Logger

    // Resource monitoring
    private var currentMemoryUsage: Int = 0
    private var activeOperations: Int = 0

    // Performance metrics
    private var totalBatchesProcessed: Int = 0
    private var totalProcessingTime: TimeInterval = 0

    // MARK: - Initialization

    public init(
        context: Metal4Context? = nil,
        configuration: BatchConfiguration = .default
    ) async {
        self.context = context
        self.configuration = configuration
        self.logger = Logger.shared
        self.simdFallback = SIMDFallback(configuration: .performance)
    }
    
    // MARK: - Batch Processing
    
    /// Process batch with automatic strategy selection
    public func processBatch<Op: BatchOperation>(
        _ data: [Op.Input],
        operation: Op
    ) async throws -> BatchResult<Op.Output> {
        let startTime = CFAbsoluteTimeGetCurrent()
        let measureToken = await logger.startMeasure("batchProcess")
        measureToken.addMetadata("batchSize", value: "\(data.count)")
        defer { measureToken.end() }
        
        // Select strategy based on data size and configuration
        let strategy = selectStrategy(for: data.count, itemSize: operation.estimatedMemoryPerItem)
        
        let (results, chunks) = try await executeWithStrategy(
            data: data,
            operation: operation,
            strategy: strategy
        )
        
        let processingTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Update metrics
        totalBatchesProcessed += 1
        totalProcessingTime += processingTime
        
        return BatchResult(
            results: results,
            processingTime: processingTime,
            chunksProcessed: chunks,
            memoryUsed: currentMemoryUsage
        )
    }
    
    /// Execute with selected strategy
    private func executeWithStrategy<Op: BatchOperation>(
        data: [Op.Input],
        operation: Op,
        strategy: BatchStrategy
    ) async throws -> ([Op.Output], Int) {
        switch strategy {
        case .sequential:
            return try await executeSequential(data: data, operation: operation)
            
        case .parallel(let maxConcurrency):
            return try await executeParallel(
                data: data,
                operation: operation,
                maxConcurrency: maxConcurrency
            )
            
        case .adaptive:
            let adaptedStrategy = await adaptStrategy(for: data.count, itemSize: operation.estimatedMemoryPerItem)
            return try await executeWithStrategy(
                data: data,
                operation: operation,
                strategy: adaptedStrategy
            )
            
        case .streaming(let chunkSize):
            return try await executeStreaming(
                data: data,
                operation: operation,
                chunkSize: chunkSize
            )
        }
    }
    
    /// Sequential execution
    private func executeSequential<Op: BatchOperation>(
        data: [Op.Input],
        operation: Op
    ) async throws -> ([Op.Output], Int) {
        await logger.debug("Executing batch sequentially", category: "BatchProcessor")
        
        let results = try await operation.process(data)
        return (results, 1)
    }
    
    /// Parallel execution with concurrency control
    private func executeParallel<Op: BatchOperation>(
        data: [Op.Input],
        operation: Op,
        maxConcurrency: Int
    ) async throws -> ([Op.Output], Int) {
        await logger.debug("Executing batch in parallel with concurrency: \(maxConcurrency)", category: "BatchProcessor")
        
        let chunkSize = max(1, data.count / maxConcurrency)
        var chunks: [(index: Int, data: [Op.Input])] = []
        
        // Divide into chunks with indices
        var chunkIndex = 0
        for i in stride(from: 0, to: data.count, by: chunkSize) {
            let end = min(i + chunkSize, data.count)
            chunks.append((index: chunkIndex, data: Array(data[i..<end])))
            chunkIndex += 1
        }
        
        // Process chunks in parallel, preserving order
        let results = try await withThrowingTaskGroup(of: (Int, [Op.Output]).self) { group in
            for chunk in chunks {
                group.addTask {
                    let results = try await operation.process(chunk.data)
                    return (chunk.index, results)
                }
            }
            
            // Collect results in order
            var orderedResults = [Int: [Op.Output]]()
            for try await (index, chunkResults) in group {
                orderedResults[index] = chunkResults
            }
            
            // Combine in correct order
            var allResults: [Op.Output] = []
            for i in 0..<chunks.count {
                if let chunkResults = orderedResults[i] {
                    allResults.append(contentsOf: chunkResults)
                }
            }
            return allResults
        }
        
        return (results, chunks.count)
    }
    
    /// Streaming execution for memory efficiency
    private func executeStreaming<Op: BatchOperation>(
        data: [Op.Input],
        operation: Op,
        chunkSize: Int
    ) async throws -> ([Op.Output], Int) {
        await logger.debug("Executing batch with streaming, chunk size: \(chunkSize)", category: "BatchProcessor")
        
        var allResults: [Op.Output] = []
        var chunksProcessed = 0
        
        for i in stride(from: 0, to: data.count, by: chunkSize) {
            // Process chunk
            let end = min(i + chunkSize, data.count)
            let chunk = Array(data[i..<end])
            
            // Check memory pressure
            let estimatedMemory = chunk.count * operation.estimatedMemoryPerItem
            if currentMemoryUsage + estimatedMemory > configuration.memoryLimit {
                await logger.warning("Memory pressure detected, waiting for resources", category: "BatchProcessor")
                try await Task.sleep(for: .milliseconds(100))
            }
            
            currentMemoryUsage += estimatedMemory
            let results = try await operation.process(chunk)
            allResults.append(contentsOf: results)
            currentMemoryUsage -= estimatedMemory
            
            chunksProcessed += 1
        }
        
        return (allResults, chunksProcessed)
    }
    
    // MARK: - Strategy Selection
    
    private func selectStrategy(for count: Int, itemSize: Int) -> BatchStrategy {
        // Estimate total memory requirement
        let totalMemory = count * itemSize
        
        // Check if we need streaming for memory constraints
        if totalMemory > configuration.memoryLimit {
            let chunkSize = configuration.memoryLimit / itemSize
            return .streaming(chunkSize: max(1, chunkSize))
        }
        
        // Use parallel for large batches
        if count > 10000 {
            let cores = ProcessInfo.processInfo.activeProcessorCount
            return .parallel(maxConcurrency: cores)
        }
        
        // Use sequential for small batches
        if count < 100 {
            return .sequential
        }
        
        // Default to adaptive parallel
        return .parallel(maxConcurrency: 4)
    }
    
    private func adaptStrategy(for count: Int, itemSize: Int, dimension: Int = 128) async -> BatchStrategy {
        // Similar to selectStrategy but with more nuanced decisions
        let totalMemory = count * itemSize
        let cores = ProcessInfo.processInfo.activeProcessorCount

        // Memory-constrained
        if totalMemory > configuration.memoryLimit * 2 {
            return .streaming(chunkSize: configuration.memoryLimit / itemSize)
        }

        // Determine if parallel processing is worthwhile
        let shouldUseParallel: Bool
        if let engine = configuration.decisionEngine {
            // Use decision engine to determine if GPU/parallel processing is beneficial
            shouldUseParallel = await engine.shouldUseGPU(
                operation: .batchSearch,
                vectorCount: count,
                candidateCount: count,
                k: 1,
                dimension: dimension
            )
        } else {
            // Fallback to hardcoded threshold
            shouldUseParallel = count >= configuration.gpuThreshold
        }

        if !shouldUseParallel {
            return .sequential
        }

        // Optimal parallel processing
        let optimalConcurrency = min(cores, count / 100)
        return .parallel(maxConcurrency: max(1, optimalConcurrency))
    }
    
    // MARK: - Specialized Batch Operations
    
    /// Batch vector similarity computation
    public func batchSimilarity(
        vectors: [[Float]],
        references: [[Float]],
        metric: SupportedDistanceMetric = .cosine
    ) async throws -> [[Float]] {
        let operation = SimilarityOperation(
            references: references,
            metric: metric,
            context: context,
            simdFallback: simdFallback
        )
        
        let result = try await processBatch(vectors, operation: operation)
        return result.results
    }
    
    /// Batch matrix multiplication
    public func batchMatrixMultiply(
        matrices: [([Float], Int, Int)],  // (data, rows, cols)
        vectors: [[Float]]
    ) async throws -> [[Float]] {
        guard matrices.count == vectors.count else {
            throw VectorError.dimensionMismatch(
                expected: matrices.count,
                actual: vectors.count
            )
        }
        
        let operation = MatrixVectorOperation(
            matrices: matrices,
            simdFallback: simdFallback
        )
        
        let result = try await processBatch(vectors, operation: operation)
        return result.results
    }
    
    // MARK: - Performance Metrics
    
    public func getPerformanceMetrics() -> (
        batchesProcessed: Int,
        averageTime: TimeInterval,
        currentMemoryUsage: Int
    ) {
        let avgTime = totalBatchesProcessed > 0 
            ? totalProcessingTime / Double(totalBatchesProcessed)
            : 0
        
        return (totalBatchesProcessed, avgTime, currentMemoryUsage)
    }
}

// MARK: - Concrete Batch Operations

/// Similarity computation operation
struct SimilarityOperation: BatchOperation {
    let references: [[Float]]
    let metric: SupportedDistanceMetric
    let context: Metal4Context?
    let simdFallback: SIMDFallback
    
    var estimatedMemoryPerItem: Int {
        references.first?.count ?? 0 * MemoryLayout<Float>.stride * references.count
    }
    
    func process(_ batch: [[Float]]) async throws -> [[Float]] {
        var results: [[Float]] = []
        
        for vector in batch {
            var distances: [Float] = []
            
            for reference in references {
                let distance: Float
                
                switch metric {
                case .euclidean:
                    distance = try await simdFallback.euclideanDistance(vector, reference)
                case .cosine:
                    let dot = try await simdFallback.dotProduct(vector, reference)
                    let normA = try await simdFallback.dotProduct(vector, vector)
                    let normB = try await simdFallback.dotProduct(reference, reference)
                    distance = 1.0 - (dot / sqrt(normA * normB))
                case .manhattan:
                    distance = computeManhattanDistance(vector, reference)
                case .dotProduct:
                    let dot = try await simdFallback.dotProduct(vector, reference)
                    distance = -dot  // Negate for distance (higher dot product = closer)
                case .chebyshev:
                    distance = computeChebyshevDistance(vector, reference)
                }
                
                distances.append(distance)
            }
            
            results.append(distances)
        }
        
        return results
    }
    
    private func computeManhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            sum += abs(a[i] - b[i])
        }
        return sum
    }

    private func computeChebyshevDistance(_ a: [Float], _ b: [Float]) -> Float {
        var maxDiff: Float = 0
        for i in 0..<a.count {
            maxDiff = max(maxDiff, abs(a[i] - b[i]))
        }
        return maxDiff
    }
}

/// Matrix-vector multiplication operation
struct MatrixVectorOperation: BatchOperation {
    let matrices: [([Float], Int, Int)]
    let simdFallback: SIMDFallback
    
    var estimatedMemoryPerItem: Int {
        matrices.first?.0.count ?? 0 * MemoryLayout<Float>.stride
    }
    
    func process(_ batch: [[Float]]) async throws -> [[Float]] {
        var results: [[Float]] = []
        
        for (index, vector) in batch.enumerated() {
            let (matrix, rows, cols) = matrices[index]
            let result = try await simdFallback.matrixVectorMultiply(
                matrix: matrix,
                rows: rows,
                columns: cols,
                vector: vector
            )
            results.append(result)
        }
        
        return results
    }
}