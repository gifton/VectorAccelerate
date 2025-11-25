//
//  ComputeEngine.swift
//  VectorAccelerate
//
//  High-level compute engine for executing vector operations on GPU
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// High-level engine for executing vector computations on Metal
public actor ComputeEngine: ComputeProvider {
    internal let context: MetalContext
    internal let shaderManager: ShaderManager
    private let accelerationConfig: AccelerationConfiguration
    private let thresholdManager: AdaptiveThresholdManager
    private let performanceMonitor: PerformanceMonitor
    private let logger: Logger
    private var operationCount: Int = 0
    private var totalComputeTime: TimeInterval = 0

    // ComputeProvider cached properties (nonisolated for protocol conformance)
    public nonisolated let device: ComputeDevice
    public nonisolated let maxConcurrency: Int
    public nonisolated let deviceInfo: ComputeDeviceInfo
    
    // Execution configuration
    public struct Configuration: Sendable {
        public let preferAsync: Bool
        public let maxBatchSize: Int
        public let enableProfiling: Bool
        
        public init(
            preferAsync: Bool = true,
            maxBatchSize: Int = 1024,
            enableProfiling: Bool = false
        ) {
            self.preferAsync = preferAsync
            self.maxBatchSize = maxBatchSize
            self.enableProfiling = enableProfiling
        }
        
        public static let `default` = Configuration()
    }
    
    public let configuration: Configuration
    
    // MARK: - Initialization
    
    public init(
        context: MetalContext,
        configuration: Configuration = .default,
        accelerationConfig: AccelerationConfiguration = .default
    ) async throws {
        self.context = context
        self.configuration = configuration
        self.accelerationConfig = accelerationConfig
        self.shaderManager = try await ShaderManager(device: context.device)
        self.thresholdManager = AdaptiveThresholdManager(configuration: accelerationConfig)
        self.performanceMonitor = PerformanceMonitor()
        self.logger = Logger(configuration: configuration.enableProfiling ? .debug : .default)

        // Initialize ComputeProvider properties by caching device info
        self.device = .gpu(index: 0)
        let capabilities = context.device.capabilities
        self.maxConcurrency = capabilities.maxThreadsPerThreadgroup
        self.deviceInfo = ComputeDeviceInfo(
            name: await context.device.name,
            availableMemory: capabilities.maxBufferLength,
            maxThreads: capabilities.maxThreadsPerThreadgroup,
            preferredChunkSize: 1024
        )

        await logger.info("ComputeEngine initialized", category: "Initialization")

        // Precompile common shaders
        if configuration.preferAsync {
            Task {
                await logger.debug("Precompiling common shaders", category: "Shaders")
                try? await shaderManager.precompileCommonShaders()
            }
        }
    }

    // MARK: - ComputeProvider Protocol Conformance

    /// Execute work on the GPU compute device
    public func execute<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        // Simple delegation - let the work closure handle GPU operations
        return try await work()
    }

    /// Execute work in parallel across items using GPU when beneficial
    ///
    /// This implementation uses Swift structured concurrency as the default.
    /// For GPU-specific batched operations, use the specialized batch methods
    /// like `batchEuclideanDistance()` which better leverage Metal parallelism.
    public func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        // For now, rely on the protocol's default implementation which uses TaskGroup
        // This provides good CPU-side parallelism for GPU operations
        // Future optimization: Detect GPU-batchable operations and execute via Metal
        try await withThrowingTaskGroup(of: (Int, T).self) { group in
            for i in items {
                group.addTask {
                    (i, try await work(i))
                }
            }

            var results = [(Int, T)]()
            results.reserveCapacity(items.count)
            for try await result in group {
                results.append(result)
            }

            // Sort by index to maintain order
            results.sort { $0.0 < $1.0 }
            return results.map { $0.1 }
        }
    }

    /// Execute side-effecting work in parallel across items
    public func parallelForEach(
        items: Range<Int>,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        // Use default implementation via TaskGroup
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in items {
                group.addTask {
                    try await body(i)
                }
            }
            for try await _ in group { }
        }
    }

    /// Execute side-effecting work in parallel with minimum chunk size hint
    public func parallelForEach(
        items: Range<Int>,
        minChunk: Int,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        // Implement chunking to better utilize GPU batch operations
        let chunkSize = max(minChunk, items.count / maxConcurrency)

        try await withThrowingTaskGroup(of: Void.self) { group in
            var start = items.lowerBound
            while start < items.upperBound {
                let end = min(start + chunkSize, items.upperBound)
                let chunk = start..<end
                group.addTask {
                    for i in chunk {
                        try await body(i)
                    }
                }
                start = end
            }
            for try await _ in group { }
        }
    }

    /// Execute a parallel reduction using GPU-aware chunking
    public func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        let total = items.count
        guard total > 0 else { return initial }

        // Use GPU-aware chunk sizing
        let concurrency = maxConcurrency
        let targetTasks = max(1, min(total, concurrency * 2))
        let chunkSize = max(1, (total + targetTasks - 1) / targetTasks)

        return try await withThrowingTaskGroup(of: R.self) { group in
            var start = items.lowerBound
            while start < items.upperBound {
                let end = min(start + chunkSize, items.upperBound)
                let range = start..<end
                group.addTask {
                    try await rangeWork(range)
                }
                start = end
            }

            var accumulator = initial
            for try await partial in group {
                accumulator = combine(accumulator, partial)
            }
            return accumulator
        }
    }

    /// Execute a parallel reduction with minimum chunk size hint
    public func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        minChunk: Int,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        // Honor the minChunk hint for GPU batch sizing
        let total = items.count
        guard total > 0 else { return initial }

        let chunkSize = max(minChunk, 1)

        return try await withThrowingTaskGroup(of: R.self) { group in
            var start = items.lowerBound
            while start < items.upperBound {
                let end = min(start + chunkSize, items.upperBound)
                let range = start..<end
                group.addTask {
                    try await rangeWork(range)
                }
                start = end
            }

            var accumulator = initial
            for try await partial in group {
                accumulator = combine(accumulator, partial)
            }
            return accumulator
        }
    }

    // MARK: - Distance Operations
    
    /// Compute Euclidean distance between two vectors
    public func euclideanDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "euclideanDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Calculate thread groups
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        
        // Track statistics
        operationCount += 1
        
        return result[0]
    }
    
    /// Compute cosine distance between two vectors
    public func cosineDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "cosineDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Use appropriate thread configuration
            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    /// Compute dot product between two vectors
    public func dotProduct(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        let startTime = Date()
        
        await logger.debug("Computing dot product for vectors of dimension \(dimension)", category: "DotProduct")
        
        // For small vectors, use CPU implementation for reliability
        if dimension <= 16 {
            var result: Float = 0
            for i in 0..<dimension {
                result += vectorA[i] * vectorB[i]
            }
            
            // Track statistics
            let duration = Date().timeIntervalSince(startTime)
            operationCount += 1
            totalComputeTime += duration
            
            await logger.debug("Dot product (CPU) result: \(result)", category: "DotProduct")
            return result
        }
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Debug: Log buffer contents
        await logger.verbose("Buffer A: \(bufferA.copyData(as: Float.self, count: min(5, dimension)))", category: "DotProduct")
        await logger.verbose("Buffer B: \(bufferB.copyData(as: Float.self, count: min(5, dimension)))", category: "DotProduct")
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "dotProduct")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Use appropriate thread configuration
            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        
        // Track statistics
        let duration = Date().timeIntervalSince(startTime)
        operationCount += 1
        totalComputeTime += duration
        
        await logger.debug("Dot product result: \(result[0]), computation time: \(String(format: "%.3f", duration * 1000))ms", category: "DotProduct")
        await performanceMonitor.recordOperation(
            type: .distanceComputation,
            path: .gpu,
            duration: duration,
            dataSize: dimension * MemoryLayout<Float>.size * 2
        )
        
        return result[0]
    }
    
    // MARK: - Batch Operations
    
    /// Compute distances from query to multiple candidates
    public func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard candidates.allSatisfy({ $0.count == query.count }) else {
            throw VectorError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }
        
        let dimension = query.count
        let candidateCount = candidates.count
        let startTime = Date()
        
        await logger.debug("Computing batch euclidean distances for \(candidateCount) candidates of dimension \(dimension)", category: "BatchDistance")
        
        // For small batches, use CPU implementation for reliability
        if candidateCount <= 10 && dimension <= 16 {
            var distances: [Float] = []
            for candidate in candidates {
                var sum: Float = 0
                for i in 0..<dimension {
                    let diff = query[i] - candidate[i]
                    sum += diff * diff
                }
                distances.append(sqrt(sum))
            }
            
            // Track statistics
            let duration = Date().timeIntervalSince(startTime)
            operationCount += 1
            totalComputeTime += duration
            
            await logger.debug("Batch distances (CPU): \(distances)", category: "BatchDistance")
            return distances
        }
        
        // Flatten candidates into single buffer
        var flatCandidates: [Float] = []
        flatCandidates.reserveCapacity(candidateCount * dimension)
        for candidate in candidates {
            flatCandidates.append(contentsOf: candidate)
        }
        
        // Get buffers
        let queryBuffer = try await context.getBuffer(for: query)
        let candidatesBuffer = try await context.getBuffer(for: flatCandidates)
        let distancesBuffer = try await context.getBuffer(size: candidateCount * MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "batchEuclideanDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(distancesBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            
            // 2D thread configuration for batch processing
            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        // Read results
        let results = distancesBuffer.copyData(as: Float.self, count: candidateCount)
        
        // Track statistics
        let duration = Date().timeIntervalSince(startTime)
        operationCount += 1
        totalComputeTime += duration
        
        await logger.debug("Batch distances (GPU) first few: \(Array(results.prefix(5)))", category: "BatchDistance")
        await performanceMonitor.recordOperation(
            type: .batchDistanceComputation,
            path: .gpu,
            duration: duration,
            dataSize: (candidateCount * dimension + dimension) * MemoryLayout<Float>.size
        )
        
        return results
    }
    
    // MARK: - Vector Operations
    
    /// Normalize vector to unit length
    public func normalize(_ vector: [Float]) async throws -> [Float] {
        let dimension = vector.count
        
        // Get buffers
        let inputBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: dimension * MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "vectorNormalize")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 2)
            
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: dimension)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        return outputBuffer.copyData(as: Float.self, count: dimension)
    }
    
    /// Scale vector by scalar
    public func scale(_ vector: [Float], by scalar: Float) async throws -> [Float] {
        let dimension = vector.count
        
        // Get buffers
        let inputBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: dimension * MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "vectorScale")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)
            
            var scalarValue = scalar
            var dim = UInt32(dimension)
            encoder.setBytes(&scalarValue, length: MemoryLayout<Float>.size, index: 2)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: dimension)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        return outputBuffer.copyData(as: Float.self, count: dimension)
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix-vector multiplication
    public func matrixVectorMultiply(
        matrix: [[Float]],
        vector: [Float]
    ) async throws -> [Float] {
        guard !matrix.isEmpty else {
            throw VectorError.invalidData("Matrix is empty")
        }
        guard matrix[0].count == vector.count else {
            throw VectorError.dimensionMismatch(expected: matrix[0].count, actual: vector.count)
        }
        
        let rows = matrix.count
        let cols = matrix[0].count
        
        // Flatten matrix
        var flatMatrix: [Float] = []
        flatMatrix.reserveCapacity(rows * cols)
        for row in matrix {
            flatMatrix.append(contentsOf: row)
        }
        
        // Get buffers
        let matrixBuffer = try await context.getBuffer(for: flatMatrix)
        let vectorBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: rows * MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "matrixVectorMultiply")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(matrixBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(vectorBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 2)
            
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&colCount, length: MemoryLayout<UInt32>.size, index: 4)
            
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: rows)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        return outputBuffer.copyData(as: Float.self, count: rows)
    }
    
    // MARK: - Metal API Access

    /// Create a Sendable buffer token from data (preferred API for concurrency safety)
    /// - Parameters:
    ///   - data: Array of data to create buffer from
    ///   - options: Metal resource options (currently ignored, uses defaults inside context)
    /// - Returns: BufferToken which returns the buffer to the pool on deinit
    public func createBufferToken<T>(from data: [T], options: MTLResourceOptions) async throws -> BufferToken where T: Sendable {
        // MetalContext handles allocation and initialization
        return try await context.getBuffer(for: data)
    }

    /// Legacy: Create a raw Metal buffer from data.
    /// Prefer `createBufferToken(from:options:)` for concurrency-safe usage.
    @preconcurrency
    public func createBuffer<T>(from data: [T], options: MTLResourceOptions) async throws -> any MTLBuffer where T: Sendable {
        let token = try await context.getBuffer(for: data)
        return token.buffer
    }

    /// Direct access to the Metal command queue for manual command buffer creation (legacy).
    /// Prefer `commandQueueHandle` to avoid returning non-Sendable types across actors.
    @preconcurrency
    public var commandQueue: any MTLCommandQueue {
        get async { context.commandQueue }
    }

    /// Sendable wrapper handle for the command queue to cross actor boundaries safely.
    public var commandQueueHandle: UnsafeSendable<any MTLCommandQueue> {
        get async { context.commandQueue.uncheckedSendable }
    }

    // MARK: - Performance & Statistics

    /// Get performance statistics
    public func getStatistics() async -> EngineStatistics {
        let contextStats = await context.getPerformanceStats()
        let shaderStats = await shaderManager.getStatistics()
        let poolStats = await context.getPoolStatistics()
        
        // Use our tracked values since we're actually tracking operations
        let computeTime = totalComputeTime > 0 ? totalComputeTime : contextStats.totalComputeTime
        let opCount = operationCount > 0 ? operationCount : contextStats.operationCount
        
        await logger.info("Engine Statistics - Operations: \(opCount), Total compute time: \(String(format: "%.3f", computeTime * 1000))ms", category: "Statistics")
        
        return EngineStatistics(
            computeTime: computeTime,
            operationCount: opCount,
            shaderCompilations: shaderStats.compilationCount,
            bufferHitRate: poolStats.hitRate,
            memoryUtilization: poolStats.memoryUtilization
        )
    }
}

/// Statistics for compute engine performance
public struct EngineStatistics: Sendable {
    public let computeTime: TimeInterval
    public let operationCount: Int
    public let shaderCompilations: Int
    public let bufferHitRate: Double
    public let memoryUtilization: Double
}
