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

    /// Hazard tracker for automatic barriers
    private let hazardTracker = HazardTracker()

    /// Performance tracking
    private var operationCount: Int = 0
    private var totalComputeTime: TimeInterval = 0
    private var fusedPipelineCount: Int = 0

    // MARK: - Initialization

    /// Create a new Metal 4 compute engine
    public init(
        context: Metal4Context,
        configuration: Metal4ComputeEngineConfiguration = .default
    ) async throws {
        self.context = context
        self.configuration = configuration

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
        let startTime = CFAbsoluteTimeGetCurrent()

        // Get buffers (automatically registered with residency)
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        // Get pipeline using Metal 4 shader compiler
        let pipeline = try await getPipeline(functionName: "euclideanDistance")

        // Acquire argument table
        let argTable = try await argumentTablePool.acquire()
        defer { Task { await argumentTablePool.release(argTable) } }

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
            totalComputeTime += CFAbsoluteTimeGetCurrent() - startTime
            operationCount += 1
        }

        return resultBuffer.copyData(as: Float.self)[0]
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

        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        // Get pipeline
        let pipeline = try await getPipeline(functionName: "cosineDistance")

        // Acquire argument table
        let argTable = try await argumentTablePool.acquire()
        defer { Task { await argumentTablePool.release(argTable) } }

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

        return resultBuffer.copyData(as: Float.self)[0]
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

        // For small vectors, use CPU
        if dimension <= 16 {
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
        defer { Task { await argumentTablePool.release(argTable) } }

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

        return resultBuffer.copyData(as: Float.self)[0]
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

        // For small vectors, use CPU
        if dimension <= 64 {
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

        return resultBuffer.copyData(as: Float.self)[0]
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

        // For small vectors, use CPU
        if dimension <= 64 {
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

        return resultBuffer.copyData(as: Float.self)[0]
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

        // For small batches, use CPU
        if candidateCount <= 10 && dimension <= 16 {
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
        defer { Task { await argumentTablePool.release(argTable) } }

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

        // For small batches, use CPU
        if candidateCount <= 10 && dimension <= 16 {
            return try candidates.map { candidate in
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
        defer { Task { await argumentTablePool.release(argTable) } }

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
        defer { Task { await argumentTablePool.release(distanceArgTable) } }

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
        defer { Task { await argumentTablePool.release(argTable) } }

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
        defer { Task { await argumentTablePool.release(argTable) } }

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
