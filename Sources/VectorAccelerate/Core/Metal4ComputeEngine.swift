//
//  Metal4ComputeEngine.swift
//  VectorAccelerate
//
//  Metal 4 compute engine with unified encoder and argument table support.
//  Uses Metal4ShaderCompiler for runtime shader compilation (no .metallib required).
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Configuration for Metal 4 compute engine
public struct Metal4ComputeEngineConfiguration: Sendable {
    public let preferAsync: Bool
    public let maxBatchSize: Int
    public let enableProfiling: Bool
    public let maxArgumentTables: Int
    public let enableFusedPipelines: Bool

    public init(
        preferAsync: Bool = true,
        maxBatchSize: Int = 1024,
        enableProfiling: Bool = false,
        maxArgumentTables: Int = 32,
        enableFusedPipelines: Bool = true
    ) {
        self.preferAsync = preferAsync
        self.maxBatchSize = maxBatchSize
        self.enableProfiling = enableProfiling
        self.maxArgumentTables = maxArgumentTables
        self.enableFusedPipelines = enableFusedPipelines
    }

    public static let `default` = Metal4ComputeEngineConfiguration()
}

/// Metal 4 compute engine with unified encoder and argument table support
///
/// This engine uses Metal 4 APIs exclusively:
/// - Argument tables instead of individual setBuffer() calls
/// - Metal4ShaderCompiler for runtime .metal file compilation
/// - Unified encoder for compute and blit operations
/// - Automatic barrier insertion for fused pipelines
/// - Integration with ResidencyManager for explicit resource management
///
/// ## Metal 4 Only
/// This implementation requires Metal 4 (iOS 26+, macOS 26+). There is no
/// fallback to older Metal versions or legacy shader loading.
///
/// Example:
/// ```swift
/// let context = try await Metal4Context()
/// let engine = try await Metal4ComputeEngine(context: context)
///
/// // Simple operation
/// let distance = try await engine.euclideanDistance(vectorA, vectorB)
///
/// // Fused pipeline (distance + selection in one encoder)
/// let topK = try await engine.fusedDistanceTopK(query: query, database: database, k: 10)
/// ```
public actor Metal4ComputeEngine {
    // MARK: - Properties

    /// The Metal 4 context
    internal let context: Metal4Context

    /// Argument table pool for efficient binding
    public let argumentTablePool: ArgumentTablePool

    /// Configuration
    public let configuration: Metal4ComputeEngineConfiguration

    /// Optional decision engine for adaptive GPU/CPU routing.
    /// When provided, replaces hardcoded thresholds with data-driven decisions.
    private let decisionEngine: GPUDecisionEngine?

    /// Hazard tracker for automatic barriers
    private let hazardTracker = HazardTracker()

    /// Performance tracking
    private var operationCount: Int = 0
    private var totalComputeTime: TimeInterval = 0
    private var fusedPipelineCount: Int = 0

    // MARK: - Initialization

    /// Create a new Metal 4 compute engine
    ///
    /// - Parameters:
    ///   - context: The Metal4Context for GPU operations
    ///   - configuration: Engine configuration options
    ///   - decisionEngine: Optional adaptive GPU/CPU routing engine.
    ///     When provided, replaces hardcoded dimension/batch thresholds with
    ///     data-driven decisions. Recommended for production use.
    public init(
        context: Metal4Context,
        configuration: Metal4ComputeEngineConfiguration = .default,
        decisionEngine: GPUDecisionEngine? = nil
    ) async throws {
        self.context = context
        self.configuration = configuration
        self.decisionEngine = decisionEngine

        // Create argument table pool
        self.argumentTablePool = ArgumentTablePool(
            device: context.device.rawDevice,
            maxTables: configuration.maxArgumentTables
        )

        // Pre-warm argument table pool
        await argumentTablePool.warmUp(count: 8)

        // Warm up pipeline cache with common shaders
        if configuration.preferAsync {
            Task {
                await context.warmUpPipelineCache()
            }
        }
    }

    // MARK: - GPU/CPU Routing

    /// Determines whether to use GPU for a single-vector operation.
    /// Consults the decision engine if available, otherwise uses hardcoded thresholds.
    private func shouldUseGPU(
        operation: GPUOperation,
        dimension: Int,
        candidateCount: Int = 1,
        fallbackThreshold: Int
    ) async -> Bool {
        if let engine = decisionEngine {
            return await engine.shouldUseGPU(
                operation: operation,
                vectorCount: candidateCount,
                candidateCount: candidateCount,
                k: 1,
                dimension: dimension
            )
        }
        // Hardcoded fallback: use CPU when dimension is at or below threshold
        return dimension > fallbackThreshold
    }

    /// Determines whether to use GPU for a batch operation.
    private func shouldUseBatchGPU(
        operation: GPUOperation,
        dimension: Int,
        candidateCount: Int,
        fallbackDimThreshold: Int = 16,
        fallbackCountThreshold: Int = 10
    ) async -> Bool {
        if let engine = decisionEngine {
            return await engine.shouldUseGPU(
                operation: operation,
                vectorCount: candidateCount,
                candidateCount: candidateCount,
                k: 1,
                dimension: dimension
            )
        }
        // Hardcoded fallback: use CPU for small batches with small dimensions
        return !(candidateCount <= fallbackCountThreshold && dimension <= fallbackDimThreshold)
    }

    // MARK: - Pipeline Access

    /// Get a compute pipeline by function name using Metal 4 shader compiler
    private func getPipeline(functionName: String) async throws -> any MTLComputePipelineState {
        try await context.getPipeline(functionName: functionName)
    }

    // MARK: - Distance Operations

    /// Compute Euclidean distance using argument tables
    public func euclideanDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        let dimension = vectorA.count

        // Route to CPU for small vectors (adaptive when decision engine is available)
        if await !shouldUseGPU(operation: .l2Distance, dimension: dimension, fallbackThreshold: 16) {
            return try AccelerateFallback.euclideanDistance(vectorA, vectorB)
        }

        let profilingStart = ContinuousClock.now

        // Get buffers (automatically registered with residency)
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        // Get pipeline using Metal 4 shader compiler
        let pipeline = try await getPipeline(functionName: "euclideanDistance")

        // Acquire argument table
        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        // Configure argument table
        argTable.setBuffer(bufferA.buffer, offset: 0, index: 0)
        argTable.setBuffer(bufferB.buffer, offset: 0, index: 1)
        argTable.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            // Apply argument table
            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            // Set dimension parameter
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            // Dispatch
            let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        // Track performance
        if configuration.enableProfiling {
            let elapsed = ContinuousClock.now - profilingStart
            totalComputeTime += Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
            operationCount += 1
        }

        return resultBuffer.readScalar(as: Float.self)
    }

    /// Compute cosine distance using argument tables
    public func cosineDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        let dimension = vectorA.count

        // Route to CPU for small vectors (adaptive when decision engine is available)
        if await !shouldUseGPU(operation: .cosineSimilarity, dimension: dimension, fallbackThreshold: 16) {
            return 1.0 - (try AccelerateFallback.cosineSimilarity(vectorA, vectorB))
        }

        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        // Get pipeline
        let pipeline = try await getPipeline(functionName: "cosineDistance")

        // Acquire argument table
        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        // Configure
        argTable.setBuffer(bufferA.buffer, offset: 0, index: 0)
        argTable.setBuffer(bufferB.buffer, offset: 0, index: 1)
        argTable.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return resultBuffer.readScalar(as: Float.self)
    }

    /// Compute dot product using argument tables
    public func dotProduct(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        let dimension = vectorA.count

        // Route to CPU for small vectors (adaptive when decision engine is available)
        if await !shouldUseGPU(operation: .dotProduct, dimension: dimension, fallbackThreshold: 16) {
            var result: Float = 0
            for i in 0..<dimension {
                result += vectorA[i] * vectorB[i]
            }
            return result
        }

        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        // Get pipeline
        let pipeline = try await getPipeline(functionName: "dotProduct")

        // Acquire argument table
        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(bufferA.buffer, offset: 0, index: 0)
        argTable.setBuffer(bufferB.buffer, offset: 0, index: 1)
        argTable.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return resultBuffer.readScalar(as: Float.self)
    }

    // MARK: - Additional Distance Metrics

    /// Compute Manhattan distance (L1 norm)
    public func manhattanDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        let dimension = vectorA.count

        // Route to CPU for small vectors (adaptive when decision engine is available)
        if await !shouldUseGPU(operation: .manhattanDistance, dimension: dimension, fallbackThreshold: 64) {
            var sum: Float = 0
            for i in 0..<dimension {
                sum += abs(vectorA[i] - vectorB[i])
            }
            return sum
        }

        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "manhattanDistance")

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return resultBuffer.readScalar(as: Float.self)
    }

    /// Compute Chebyshev distance (L∞ norm)
    public func chebyshevDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        let dimension = vectorA.count

        // Route to CPU for small vectors (adaptive when decision engine is available)
        if await !shouldUseGPU(operation: .chebyshevDistance, dimension: dimension, fallbackThreshold: 64) {
            var maxDiff: Float = 0
            for i in 0..<dimension {
                maxDiff = max(maxDiff, abs(vectorA[i] - vectorB[i]))
            }
            return maxDiff
        }

        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "chebyshevDistance")

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return resultBuffer.readScalar(as: Float.self)
    }

    /// Compute Minkowski distance with custom p parameter
    public func minkowskiDistance(
        _ vectorA: [Float],
        _ vectorB: [Float],
        p: Float
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        // Special cases
        if p == 1.0 {
            return try await manhattanDistance(vectorA, vectorB)
        } else if p == 2.0 {
            return try await euclideanDistance(vectorA, vectorB)
        } else if p == .infinity {
            return try await chebyshevDistance(vectorA, vectorB)
        }

        let dimension = vectorA.count

        // CPU fallback for now (Minkowski with arbitrary p is rarely GPU-accelerated)
        var sum: Float = 0
        for i in 0..<dimension {
            sum += pow(abs(vectorA[i] - vectorB[i]), p)
        }
        return pow(sum, 1.0 / p)
    }

    // MARK: - Batch Operations

    /// Compute batch distances using argument tables
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

        // Route to CPU for small batches (adaptive when decision engine is available)
        if await !shouldUseBatchGPU(operation: .l2Distance, dimension: dimension, candidateCount: candidateCount) {
            return candidates.map { candidate in
                var sum: Float = 0
                for i in 0..<dimension {
                    let diff = query[i] - candidate[i]
                    sum += diff * diff
                }
                return sqrt(sum)
            }
        }

        // Flatten candidates
        var flatCandidates: [Float] = []
        flatCandidates.reserveCapacity(candidateCount * dimension)
        for candidate in candidates {
            flatCandidates.append(contentsOf: candidate)
        }

        // Get buffers
        let queryBuffer = try await context.getBuffer(for: query)
        let candidatesBuffer = try await context.getBuffer(for: flatCandidates)
        let distancesBuffer = try await context.getBuffer(size: candidateCount * MemoryLayout<Float>.size)

        // Get pipeline
        let pipeline = try await getPipeline(functionName: "batchEuclideanDistance")

        // Acquire argument table (batch descriptor for more bindings)
        let argTable = try await argumentTablePool.acquire(descriptor: .batch)
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        argTable.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
        argTable.setBuffer(distancesBuffer.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        return distancesBuffer.copyData(as: Float.self, count: candidateCount)
    }

    /// Compute batch cosine distances
    public func batchCosineDistance(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard candidates.allSatisfy({ $0.count == query.count }) else {
            throw VectorError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }

        let dimension = query.count
        let candidateCount = candidates.count

        // Route to CPU for small batches (adaptive when decision engine is available)
        if await !shouldUseBatchGPU(operation: .cosineSimilarity, dimension: dimension, candidateCount: candidateCount) {
            return candidates.map { candidate in
                var dotProd: Float = 0
                var normA: Float = 0
                var normB: Float = 0
                for i in 0..<dimension {
                    dotProd += query[i] * candidate[i]
                    normA += query[i] * query[i]
                    normB += candidate[i] * candidate[i]
                }
                let denom = sqrt(normA) * sqrt(normB)
                if denom < 1e-8 { return 1.0 }
                return 1.0 - (dotProd / denom)
            }
        }

        // Flatten candidates
        var flatCandidates: [Float] = []
        flatCandidates.reserveCapacity(candidateCount * dimension)
        for candidate in candidates {
            flatCandidates.append(contentsOf: candidate)
        }

        let queryBuffer = try await context.getBuffer(for: query)
        let candidatesBuffer = try await context.getBuffer(for: flatCandidates)
        let distancesBuffer = try await context.getBuffer(size: candidateCount * MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "batchCosineDistance")

        let argTable = try await argumentTablePool.acquire(descriptor: .batch)
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        argTable.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
        argTable.setBuffer(distancesBuffer.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        return distancesBuffer.copyData(as: Float.self, count: candidateCount)
    }

    // MARK: - Fused Pipelines

    /// Fused distance + top-k selection in a single encoder
    ///
    /// This is more efficient than separate operations because:
    /// - Single command buffer submission
    /// - Barrier instead of encoder end/begin
    /// - Shared argument table where possible
    public func fusedDistanceTopK(
        query: [Float],
        database: [[Float]],
        k: Int,
        metric: Metal4DistanceMetric = .euclidean
    ) async throws -> [(index: Int, distance: Float)] {
        guard !database.isEmpty else { return [] }
        guard database.allSatisfy({ $0.count == query.count }) else {
            throw VectorError.dimensionMismatch(expected: query.count, actual: database[0].count)
        }

        let dimension = query.count
        let candidateCount = database.count
        let actualK = min(k, candidateCount)

        // Flatten database
        var flatDatabase: [Float] = []
        flatDatabase.reserveCapacity(candidateCount * dimension)
        for vector in database {
            flatDatabase.append(contentsOf: vector)
        }

        // Get buffers
        let queryBuffer = try await context.getBuffer(for: query)
        let databaseBuffer = try await context.getBuffer(for: flatDatabase)
        let distancesBuffer = try await context.getBuffer(size: candidateCount * MemoryLayout<Float>.size)

        // Get pipelines
        let distanceFunctionName = metric == .euclidean ? "batchEuclideanDistance" : "batchCosineDistance"
        let distancePipeline = try await getPipeline(functionName: distanceFunctionName)

        // Acquire argument tables
        let distanceArgTable = try await argumentTablePool.acquire(descriptor: .batch)
        defer { PendingTableReturns.shared.enqueue(table: distanceArgTable, pool: argumentTablePool) }

        distanceArgTable.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        distanceArgTable.setBuffer(databaseBuffer.buffer, offset: 0, index: 1)
        distanceArgTable.setBuffer(distancesBuffer.buffer, offset: 0, index: 2)

        // Execute fused pipeline
        try await context.executeAndWait { commandBuffer, encoder in
            // Stage 1: Distance computation
            encoder.setComputePipelineState(distancePipeline)

            if let metal4Table = distanceArgTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

            // Barrier before selection
            encoder.memoryBarrier(scope: .buffers)
        }

        // CPU-side top-k selection
        let distances = distancesBuffer.copyData(as: Float.self, count: candidateCount)
        let indexed = distances.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 < $1.1 }
        let topK = sorted.prefix(actualK)

        fusedPipelineCount += 1

        return topK.map { (index: $0.0, distance: $0.1) }
    }

    // MARK: - Vector Operations

    /// Normalize vector using argument tables
    public func normalize(_ vector: [Float]) async throws -> [Float] {
        let dimension = vector.count

        let inputBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: dimension * MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "vectorNormalize")

        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
        argTable.setBuffer(outputBuffer.buffer, offset: 0, index: 1)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 2)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: (dimension + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return outputBuffer.copyData(as: Float.self, count: dimension)
    }

    /// Scale vector by scalar
    public func scale(_ vector: [Float], by scalar: Float) async throws -> [Float] {
        let dimension = vector.count

        let inputBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: dimension * MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "vectorScale")

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)

            var scalarValue = scalar
            var dim = UInt32(dimension)
            encoder.setBytes(&scalarValue, length: MemoryLayout<Float>.size, index: 2)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: (dimension + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return outputBuffer.copyData(as: Float.self, count: dimension)
    }

    // MARK: - Matrix Operations

    /// Matrix-vector multiplication using argument tables
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

        let matrixBuffer = try await context.getBuffer(for: flatMatrix)
        let vectorBuffer = try await context.getBuffer(for: vector)
        let outputBuffer = try await context.getBuffer(size: rows * MemoryLayout<Float>.size)

        let pipeline = try await getPipeline(functionName: "matrixVectorMultiply")

        let argTable = try await argumentTablePool.acquire(descriptor: .matrix)
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(matrixBuffer.buffer, offset: 0, index: 0)
        argTable.setBuffer(vectorBuffer.buffer, offset: 0, index: 1)
        argTable.setBuffer(outputBuffer.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&colCount, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerGroup = MTLSize(width: min(256, rows), height: 1, depth: 1)
            let threadgroups = MTLSize(width: (rows + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return outputBuffer.copyData(as: Float.self, count: rows)
    }

    // MARK: - Statistics

    /// Get engine statistics
    public func getStatistics() async -> Metal4EngineStatistics {
        let poolStats = await argumentTablePool.getStatistics()
        let contextStats = await context.getPerformanceStats()

        return Metal4EngineStatistics(
            operationCount: operationCount,
            totalComputeTime: totalComputeTime,
            fusedPipelineCount: fusedPipelineCount,
            argumentTableStats: poolStats,
            shaderCompilations: 0,  // Tracked by pipeline cache now
            averageOperationTime: operationCount > 0 ? totalComputeTime / Double(operationCount) : 0,
            contextOperationCount: contextStats.operationCount,
            contextComputeTime: contextStats.totalComputeTime
        )
    }

    /// Reset statistics
    public func resetStatistics() {
        operationCount = 0
        totalComputeTime = 0
        fusedPipelineCount = 0
    }
}

// MARK: - Statistics

/// Statistics for Metal 4 compute engine
public struct Metal4EngineStatistics: Sendable {
    public let operationCount: Int
    public let totalComputeTime: TimeInterval
    public let fusedPipelineCount: Int
    public let argumentTableStats: ArgumentTablePoolStatistics
    public let shaderCompilations: Int
    public let averageOperationTime: TimeInterval
    public let contextOperationCount: Int
    public let contextComputeTime: TimeInterval
}

// MARK: - Distance Metric Type

/// Supported distance metric types for Metal 4 compute engine
public enum Metal4DistanceMetric: String, Sendable, CaseIterable {
    case euclidean
    case cosine
    case dotProduct
    case manhattan
    case chebyshev
}

// MARK: - Pre-Allocated Buffer API

public extension Metal4ComputeEngine {

    /// Allocate a reusable buffer for vector data.
    ///
    /// Use with the `*WithBuffers` methods below to avoid per-call allocation overhead.
    /// The returned token can be reused across multiple calls by writing new data into it.
    ///
    /// ```swift
    /// let queryBuf = try await engine.allocateVectorBuffer(dimension: 768)
    /// let dbBuf = try await engine.allocateVectorBuffer(dimension: 768, count: 1000)
    /// let resultBuf = try await engine.allocateResultBuffer(count: 1000)
    ///
    /// // Reuse across many queries
    /// for query in queries {
    ///     queryBuf.write(data: query)
    ///     try await engine.euclideanDistanceWithBuffers(
    ///         query: queryBuf, database: dbBuf, result: resultBuf,
    ///         dimension: 768, candidateCount: 1000
    ///     )
    ///     let distances = resultBuf.contents(as: Float.self)
    ///     // Read distances[0..<1000] directly from unified memory (zero-copy)
    /// }
    /// ```
    func allocateVectorBuffer(dimension: Int, count: Int = 1) async throws -> BufferToken {
        try await context.getBuffer(size: count * dimension * MemoryLayout<Float>.size)
    }

    /// Allocate a reusable result buffer for distance/similarity outputs.
    func allocateResultBuffer(count: Int = 1) async throws -> BufferToken {
        try await context.getBuffer(size: count * MemoryLayout<Float>.size)
    }

    /// Compute batch Euclidean distances using pre-allocated buffers.
    ///
    /// This avoids all per-call allocation overhead. The caller owns the buffers
    /// and can read results directly from `result.contents(as: Float.self)` without
    /// copying (zero-copy on Apple Silicon unified memory).
    ///
    /// - Parameters:
    ///   - query: Buffer containing the query vector (dimension floats). Must be pre-filled via `write(data:)`.
    ///   - database: Buffer containing flattened candidate vectors (candidateCount * dimension floats).
    ///   - result: Buffer for output distances (candidateCount floats).
    ///   - dimension: Vector dimensionality.
    ///   - candidateCount: Number of candidate vectors in the database buffer.
    func batchEuclideanDistanceWithBuffers(
        query: BufferToken,
        database: BufferToken,
        result: BufferToken,
        dimension: Int,
        candidateCount: Int
    ) async throws {
        let pipeline = try await getPipeline(functionName: "batchEuclideanDistance")

        let argTable = try await argumentTablePool.acquire(descriptor: .batch)
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(query.buffer, offset: 0, index: 0)
        argTable.setBuffer(database.buffer, offset: 0, index: 1)
        argTable.setBuffer(result.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }

    /// Compute batch cosine distances using pre-allocated buffers.
    func batchCosineDistanceWithBuffers(
        query: BufferToken,
        database: BufferToken,
        result: BufferToken,
        dimension: Int,
        candidateCount: Int
    ) async throws {
        let pipeline = try await getPipeline(functionName: "batchCosineDistance")

        let argTable = try await argumentTablePool.acquire(descriptor: .batch)
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(query.buffer, offset: 0, index: 0)
        argTable.setBuffer(database.buffer, offset: 0, index: 1)
        argTable.setBuffer(result.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }

    /// Compute single Euclidean distance using pre-allocated buffers.
    ///
    /// Reads the result directly from unified memory (zero-copy).
    func euclideanDistanceWithBuffers(
        vectorA: BufferToken,
        vectorB: BufferToken,
        result: BufferToken,
        dimension: Int
    ) async throws -> Float {
        let pipeline = try await getPipeline(functionName: "euclideanDistance")

        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(vectorA.buffer, offset: 0, index: 0)
        argTable.setBuffer(vectorB.buffer, offset: 0, index: 1)
        argTable.setBuffer(result.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return result.readScalar(as: Float.self)
    }

    /// Compute single dot product using pre-allocated buffers.
    func dotProductWithBuffers(
        vectorA: BufferToken,
        vectorB: BufferToken,
        result: BufferToken,
        dimension: Int
    ) async throws -> Float {
        let pipeline = try await getPipeline(functionName: "dotProduct")

        let argTable = try await argumentTablePool.acquire()
        defer { PendingTableReturns.shared.enqueue(table: argTable, pool: argumentTablePool) }

        argTable.setBuffer(vectorA.buffer, offset: 0, index: 0)
        argTable.setBuffer(vectorB.buffer, offset: 0, index: 1)
        argTable.setBuffer(result.buffer, offset: 0, index: 2)

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)

            if let metal4Table = argTable as? Metal4ArgumentTable {
                metal4Table.apply(to: encoder)
            }

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return result.readScalar(as: Float.self)
    }
}

// MARK: - Convenience Extensions

public extension Metal4ComputeEngine {
    /// Create with default Metal 4 context
    static func createDefault() async throws -> Metal4ComputeEngine {
        guard let context = await Metal4Context.createDefault() else {
            throw VectorError.deviceInitializationFailed("Metal 4 not available")
        }
        return try await Metal4ComputeEngine(context: context)
    }
}
