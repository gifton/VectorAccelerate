// VectorAccelerate: Learned Distance Kernel
//
// GPU-accelerated distance computation with learned projection matrices.
// Part of Phase 4: Experimental ML Integration.
//
// This kernel projects vectors through a learned transformation before
// computing distances, enabling:
// - Dimensionality reduction (e.g., 768 -> 128 for Matryoshka embeddings)
// - Learned metric spaces for domain-specific similarity
// - Efficient approximate nearest neighbor search

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

/// Learned distance computation kernel with projection matrix support.
///
/// This kernel extends standard distance computation by projecting vectors
/// through a learned weight matrix before computing distances. This is useful
/// for:
/// - Matryoshka-style embeddings with learned projections
/// - Domain-adapted distance metrics
/// - Dimensionality reduction for faster search
///
/// ## Usage
/// ```swift
/// let kernel = try await LearnedDistanceKernel(context: context)
///
/// // Load projection weights
/// let projection = try await kernel.loadProjection(
///     from: weightsURL,
///     inputDim: 768,
///     outputDim: 128
/// )
///
/// // Compute distances in projected space
/// let distances = try await kernel.compute(
///     queries: queries,
///     database: database,
///     projection: projection
/// )
/// ```
///
/// ## Fallback Behavior
/// When projection weights are unavailable or ML features are disabled,
/// use the standard `L2DistanceKernel` for unprojected distance computation.
public final class LearnedDistanceKernel: @unchecked Sendable, Metal4Kernel {
    // MARK: - Properties

    public let context: Metal4Context

    // Generic learned distance pipeline
    private let learnedL2Pipeline: any MTLComputePipelineState
    private let learnedCosinePipeline: any MTLComputePipelineState

    // Optimized pipelines for common dimension pairs
    private let learned768to128Pipeline: any MTLComputePipelineState
    private let learned384to64Pipeline: any MTLComputePipelineState

    // Batch projection pipelines
    private let batchProjectionPipeline: any MTLComputePipelineState
    private let batchNormalizePipeline: any MTLComputePipelineState

    // Tensor manager for weight storage
    private let tensorManager: TensorManager

    // MARK: - Metal4Kernel Protocol

    public var name: String { "LearnedDistanceKernel" }

    public func warmUp() async throws {
        // Pipelines are already compiled during init
    }

    // MARK: - Parameters

    /// Parameters for learned distance computation
    public struct Parameters: Sendable {
        public let numQueries: UInt32
        public let numDatabase: UInt32
        public let inputDimension: UInt32
        public let projectedDimension: UInt32
        public let strideQuery: UInt32
        public let strideDatabase: UInt32
        public let strideOutput: UInt32
        public let computeSqrt: UInt8
        public let normalizeProjected: UInt8
        private let padding: (UInt8, UInt8) = (0, 0)

        public init(
            numQueries: Int,
            numDatabase: Int,
            inputDimension: Int,
            projectedDimension: Int,
            computeSqrt: Bool = true,
            normalizeProjected: Bool = false
        ) {
            self.numQueries = UInt32(numQueries)
            self.numDatabase = UInt32(numDatabase)
            self.inputDimension = UInt32(inputDimension)
            self.projectedDimension = UInt32(projectedDimension)
            self.strideQuery = UInt32(inputDimension)
            self.strideDatabase = UInt32(inputDimension)
            self.strideOutput = UInt32(numDatabase)
            self.computeSqrt = computeSqrt ? 1 : 0
            self.normalizeProjected = normalizeProjected ? 1 : 0
        }
    }

    /// Parameters for batch projection
    public struct ProjectionParameters: Sendable {
        public let numVectors: UInt32
        public let inputDimension: UInt32
        public let outputDimension: UInt32
        public let stride: UInt32
        public let normalize: UInt8
        private let padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

        public init(
            numVectors: Int,
            inputDimension: Int,
            outputDimension: Int,
            normalize: Bool = false
        ) {
            self.numVectors = UInt32(numVectors)
            self.inputDimension = UInt32(inputDimension)
            self.outputDimension = UInt32(outputDimension)
            self.stride = UInt32(inputDimension)
            self.normalize = normalize ? 1 : 0
        }
    }

    // MARK: - Initialization

    /// Initialize the learned distance kernel
    ///
    /// - Parameter context: Metal4Context to use for computation
    /// - Throws: `VectorError` if shader compilation fails
    public init(context: Metal4Context) async throws {
        self.context = context
        self.tensorManager = TensorManager(device: context.device.rawDevice)

        // Load pipelines using Metal4 shader compiler
        self.learnedL2Pipeline = try await context.getPipeline(functionName: "learned_l2_distance_kernel")
        self.learnedCosinePipeline = try await context.getPipeline(functionName: "learned_cosine_similarity_kernel")
        self.learned768to128Pipeline = try await context.getPipeline(functionName: "learned_l2_768_to_128_kernel")
        self.learned384to64Pipeline = try await context.getPipeline(functionName: "learned_l2_384_to_64_kernel")
        self.batchProjectionPipeline = try await context.getPipeline(functionName: "batch_projection_kernel")
        self.batchNormalizePipeline = try await context.getPipeline(functionName: "batch_normalize_kernel")
    }

    // MARK: - Weight Loading

    /// Load projection weights from a file
    ///
    /// - Parameters:
    ///   - url: URL to binary weight file (row-major float32)
    ///   - inputDim: Input vector dimension
    ///   - outputDim: Output dimension after projection
    ///   - name: Optional name for the tensor (for retrieval)
    /// - Returns: TensorBuffer containing projection weights
    public func loadProjection(
        from url: URL,
        inputDim: Int,
        outputDim: Int,
        name: String = "projection"
    ) async throws -> TensorBuffer {
        let shape = TensorShape.projection(inputDim: inputDim, outputDim: outputDim)
        return try await tensorManager.loadWeights(
            from: url,
            name: name,
            shape: shape,
            dataType: .float32
        )
    }

    /// Load projection weights from Data
    public func loadProjection(
        from data: Data,
        inputDim: Int,
        outputDim: Int,
        name: String = "projection"
    ) async throws -> TensorBuffer {
        let shape = TensorShape.projection(inputDim: inputDim, outputDim: outputDim)
        return try await tensorManager.loadWeights(
            from: data,
            name: name,
            shape: shape,
            dataType: .float32
        )
    }

    /// Create projection from Float array
    ///
    /// Weight layout: row-major [outputDim, inputDim]
    /// - weights[j * inputDim + i] is the weight for input[i] -> output[j]
    public func createProjection(
        from weights: [Float],
        inputDim: Int,
        outputDim: Int,
        name: String = "projection"
    ) async throws -> TensorBuffer {
        let shape = TensorShape.projection(inputDim: inputDim, outputDim: outputDim)
        return try await tensorManager.createTensor(
            from: weights,
            name: name,
            shape: shape
        )
    }

    /// Create a random projection matrix (for testing or baseline)
    ///
    /// Uses Xavier/Glorot initialization for proper scaling.
    public func createRandomProjection(
        inputDim: Int,
        outputDim: Int,
        name: String = "random_projection"
    ) async throws -> TensorBuffer {
        return try await tensorManager.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim,
            name: name
        )
    }

    /// Get a previously loaded projection by name
    public func getProjection(name: String) async -> TensorBuffer? {
        await tensorManager.getTensor(name: name)
    }

    // MARK: - Pipeline Selection

    /// Select the optimal pipeline for given parameters
    private func selectPipeline(for parameters: Parameters) -> any MTLComputePipelineState {
        if parameters.inputDimension == 768 && parameters.projectedDimension == 128 {
            return learned768to128Pipeline
        } else if parameters.inputDimension == 384 && parameters.projectedDimension == 64 {
            return learned384to64Pipeline
        } else {
            return learnedL2Pipeline
        }
    }

    // MARK: - Distance Computation

    /// Compute learned L2 distances between query and database vectors
    ///
    /// - Parameters:
    ///   - queryVectors: Buffer containing query vectors [N, inputDim]
    ///   - databaseVectors: Buffer containing database vectors [M, inputDim]
    ///   - projectionWeights: Projection matrix buffer [outputDim, inputDim]
    ///   - distances: Output buffer for distances [N, M]
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func computeL2(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        projectionWeights: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        // Select optimized kernel for common dimension pairs
        let pipeline: any MTLComputePipelineState
        if parameters.inputDimension == 768 && parameters.projectedDimension == 128 {
            pipeline = learned768to128Pipeline
        } else if parameters.inputDimension == 384 && parameters.projectedDimension == 64 {
            pipeline = learned384to64Pipeline
        } else {
            pipeline = learnedL2Pipeline
        }

        encoder.setComputePipelineState(pipeline)
        encoder.label = "LearnedL2DistanceKernel"

        encoder.setBuffer(queryVectors, offset: 0, index: 0)
        encoder.setBuffer(databaseVectors, offset: 0, index: 1)
        encoder.setBuffer(projectionWeights, offset: 0, index: 2)
        encoder.setBuffer(distances, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 4)

        // 2D dispatch for query x database pairs
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadsPerSide = Int(sqrt(Double(min(maxThreads, 256))))
        let threadWidth = min(threadsPerSide, Int(parameters.numQueries))
        let threadHeight = min(threadsPerSide, Int(parameters.numDatabase))

        let threadsPerThreadgroup = MTLSize(
            width: threadWidth,
            height: threadHeight,
            depth: 1
        )

        let threadgroups = MTLSize(
            width: (Int(parameters.numQueries) + threadWidth - 1) / threadWidth,
            height: (Int(parameters.numDatabase) + threadHeight - 1) / threadHeight,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    /// Compute learned cosine similarity between query and database vectors
    public func computeCosine(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        projectionWeights: any MTLBuffer,
        similarities: any MTLBuffer,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        encoder.setComputePipelineState(learnedCosinePipeline)
        encoder.label = "LearnedCosineSimilarityKernel"

        encoder.setBuffer(queryVectors, offset: 0, index: 0)
        encoder.setBuffer(databaseVectors, offset: 0, index: 1)
        encoder.setBuffer(projectionWeights, offset: 0, index: 2)
        encoder.setBuffer(similarities, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 4)

        let maxThreads = learnedCosinePipeline.maxTotalThreadsPerThreadgroup
        let threadsPerSide = Int(sqrt(Double(min(maxThreads, 256))))
        let threadWidth = min(threadsPerSide, Int(parameters.numQueries))
        let threadHeight = min(threadsPerSide, Int(parameters.numDatabase))

        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(
            width: (Int(parameters.numQueries) + threadWidth - 1) / threadWidth,
            height: (Int(parameters.numDatabase) + threadHeight - 1) / threadHeight,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    // MARK: - High-Level API

    /// Compute L2 distances with automatic buffer management
    ///
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - database: Array of database vectors
    ///   - projection: Projection weight tensor
    ///   - computeSqrt: Whether to compute square root
    ///   - normalizeProjected: Whether to normalize after projection
    /// - Returns: 2D array of distances [queries.count][database.count]
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        projection: TensorBuffer,
        computeSqrt: Bool = true,
        normalizeProjected: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let inputDim = queries[0].count
        guard queries.allSatisfy({ $0.count == inputDim }) &&
              database.allSatisfy({ $0.count == inputDim }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        // Verify projection dimensions
        guard projection.shape.dimensions.count == 2,
              projection.shape.dimensions[1] == inputDim else {
            throw VectorError.invalidOperation(
                "Projection input dimension (\(projection.shape.dimensions.last ?? 0)) " +
                "does not match vector dimension (\(inputDim))"
            )
        }

        let outputDim = projection.shape.dimensions[0]
        let numQueries = queries.count
        let numDatabase = database.count

        // Create buffers
        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }
        let device = context.device.rawDevice

        guard let queryBuffer = flatQueries.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = flatDatabase.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferCreationFailed("Failed to create database buffer")
        }

        guard let distanceBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create distance buffer")
        }

        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            inputDimension: inputDim,
            projectedDimension: outputDim,
            computeSqrt: computeSqrt,
            normalizeProjected: normalizeProjected
        )

        // Capture pipeline and buffers for closure
        let pipeline = selectPipeline(for: parameters)
        let projectionBuffer = projection.buffer

        // Execute computation using Metal4 pattern
        try await context.executeAndWait { [parameters] commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "LearnedL2Distance"

            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(databaseBuffer, offset: 0, index: 1)
            encoder.setBuffer(projectionBuffer, offset: 0, index: 2)
            encoder.setBuffer(distanceBuffer, offset: 0, index: 3)

            var params = parameters
            encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 4)

            // 2D dispatch: one thread per (query, database) pair
            let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
            let threadWidth = min(Int(parameters.numQueries), 16)
            let threadHeight = min(maxThreads / threadWidth, Int(parameters.numDatabase))

            let threadsPerGroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
            let groups = MTLSize(
                width: (Int(parameters.numQueries) + threadWidth - 1) / threadWidth,
                height: (Int(parameters.numDatabase) + threadHeight - 1) / threadHeight,
                depth: 1
            )

            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        }

        // Extract results
        let distancePointer = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )

        var results: [[Float]] = []
        results.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numDatabase)
            for j in 0..<numDatabase {
                row.append(distancePointer[i * numDatabase + j])
            }
            results.append(row)
        }

        return results
    }

    /// Compute distances using VectorCore types
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        projection: TensorBuffer,
        computeSqrt: Bool = true,
        normalizeProjected: Bool = false
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty && !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let inputDim = queries[0].count
        guard queries.allSatisfy({ $0.count == inputDim }) &&
              database.allSatisfy({ $0.count == inputDim }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        guard projection.shape.dimensions.count == 2,
              projection.shape.dimensions[1] == inputDim else {
            throw VectorError.invalidOperation(
                "Projection dimension mismatch"
            )
        }

        let outputDim = projection.shape.dimensions[0]
        let numQueries = queries.count
        let numDatabase = database.count
        let device = context.device.rawDevice

        // Create buffers from VectorProtocol arrays
        let queryData = queries.flatMap { vector -> [Float] in
            var result = [Float](repeating: 0, count: vector.count)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<vector.count {
                    result[i] = ptr[i]
                }
            }
            return result
        }

        let databaseData = database.flatMap { vector -> [Float] in
            var result = [Float](repeating: 0, count: vector.count)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<vector.count {
                    result[i] = ptr[i]
                }
            }
            return result
        }

        guard let queryBuffer = queryData.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = databaseData.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferCreationFailed("Failed to create database buffer")
        }

        guard let distanceBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create distance buffer")
        }

        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            inputDimension: inputDim,
            projectedDimension: outputDim,
            computeSqrt: computeSqrt,
            normalizeProjected: normalizeProjected
        )

        // Capture pipeline and buffers for closure
        let pipeline = selectPipeline(for: parameters)
        let projectionBuffer = projection.buffer

        // Execute computation using Metal4 pattern
        try await context.executeAndWait { [parameters] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "LearnedL2Distance"

            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(databaseBuffer, offset: 0, index: 1)
            encoder.setBuffer(projectionBuffer, offset: 0, index: 2)
            encoder.setBuffer(distanceBuffer, offset: 0, index: 3)

            var params = parameters
            encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 4)

            let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
            let threadWidth = min(Int(parameters.numQueries), 16)
            let threadHeight = min(maxThreads / threadWidth, Int(parameters.numDatabase))

            let threadsPerGroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
            let groups = MTLSize(
                width: (Int(parameters.numQueries) + threadWidth - 1) / threadWidth,
                height: (Int(parameters.numDatabase) + threadHeight - 1) / threadHeight,
                depth: 1
            )

            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        }

        let distancePointer = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )

        var results: [[Float]] = []
        results.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numDatabase)
            for j in 0..<numDatabase {
                row.append(distancePointer[i * numDatabase + j])
            }
            results.append(row)
        }

        return results
    }

    // MARK: - Batch Projection

    /// Project a batch of vectors through the projection matrix
    ///
    /// Useful for pre-projecting a database for subsequent fast distance computation.
    ///
    /// - Parameters:
    ///   - vectors: Input vectors to project
    ///   - projection: Projection weight tensor
    ///   - normalize: Whether to L2 normalize output vectors
    /// - Returns: Projected vectors [numVectors, outputDim]
    public func projectBatch(
        vectors: [[Float]],
        projection: TensorBuffer,
        normalize: Bool = false
    ) async throws -> [[Float]] {
        guard !vectors.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let inputDim = vectors[0].count
        guard vectors.allSatisfy({ $0.count == inputDim }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        guard projection.shape.dimensions.count == 2,
              projection.shape.dimensions[1] == inputDim else {
            throw VectorError.invalidOperation("Projection dimension mismatch")
        }

        let outputDim = projection.shape.dimensions[0]
        let numVectors = vectors.count
        let device = context.device.rawDevice

        let flatVectors = vectors.flatMap { $0 }

        guard let inputBuffer = flatVectors.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }

        guard let outputBuffer = device.makeBuffer(
            length: numVectors * outputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create output buffer")
        }

        guard let commandBuffer = context.makeCommandBufferUnsafe() else {
            throw VectorError.invalidOperation("Failed to create command buffer")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        // Batch projection
        encoder.setComputePipelineState(batchProjectionPipeline)
        encoder.label = "BatchProjectionKernel"

        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(projection.buffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)

        var params = ProjectionParameters(
            numVectors: numVectors,
            inputDimension: inputDim,
            outputDimension: outputDim,
            normalize: normalize
        )
        encoder.setBytes(&params, length: MemoryLayout<ProjectionParameters>.size, index: 3)

        // 2D dispatch: (numVectors, outputDim)
        let maxThreads = batchProjectionPipeline.maxTotalThreadsPerThreadgroup
        let threadWidth = min(32, numVectors)
        let threadHeight = min(maxThreads / threadWidth, outputDim)

        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectors + threadWidth - 1) / threadWidth,
            height: (outputDim + threadHeight - 1) / threadHeight,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        // Normalize if requested
        if normalize {
            guard let normalizeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw VectorError.encoderCreationFailed()
            }

            normalizeEncoder.setComputePipelineState(batchNormalizePipeline)
            normalizeEncoder.label = "BatchNormalizeKernel"

            normalizeEncoder.setBuffer(outputBuffer, offset: 0, index: 0)
            normalizeEncoder.setBytes(&params, length: MemoryLayout<ProjectionParameters>.size, index: 1)

            let normalizeThreads = MTLSize(width: min(256, numVectors), height: 1, depth: 1)
            let normalizeGroups = MTLSize(
                width: (numVectors + normalizeThreads.width - 1) / normalizeThreads.width,
                height: 1,
                depth: 1
            )

            normalizeEncoder.dispatchThreadgroups(normalizeGroups, threadsPerThreadgroup: normalizeThreads)
            normalizeEncoder.endEncoding()
        }

        // Wait for completion using continuation (handler must be added before commit)
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }

        // Extract results
        let outputPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numVectors * outputDim
        )

        var results: [[Float]] = []
        results.reserveCapacity(numVectors)
        for i in 0..<numVectors {
            var row: [Float] = []
            row.reserveCapacity(outputDim)
            for j in 0..<outputDim {
                row.append(outputPointer[i * outputDim + j])
            }
            results.append(row)
        }

        return results
    }

    // MARK: - Statistics

    /// Get tensor manager statistics
    public func getTensorStatistics() async -> TensorManagerStatistics {
        await tensorManager.getStatistics()
    }

    /// Unload projection weights to free memory
    public func unloadProjection(name: String) async {
        await tensorManager.unload(name: name)
    }
}

// MARK: - Performance Extensions

extension LearnedDistanceKernel {
    /// Performance statistics for kernel execution
    public struct PerformanceStats: Sendable {
        public let computeTime: TimeInterval
        public let throughput: Double  // GFLOPS
        public let projectionOps: Int
        public let distanceOps: Int
    }

    /// Compute with performance monitoring
    public func computeWithStats(
        queries: [[Float]],
        database: [[Float]],
        projection: TensorBuffer,
        computeSqrt: Bool = true
    ) async throws -> (results: [[Float]], stats: PerformanceStats) {
        let startTime = CACurrentMediaTime()
        let results = try await compute(
            queries: queries,
            database: database,
            projection: projection,
            computeSqrt: computeSqrt
        )
        let endTime = CACurrentMediaTime()

        let computeTime = endTime - startTime
        let inputDim = queries[0].count
        let outputDim = projection.shape.dimensions[0]
        let numQueries = queries.count
        let numDatabase = database.count

        // Projection: 2 ops per multiply-add, inputDim * outputDim per vector
        let projectionOps = (numQueries + numDatabase) * inputDim * outputDim * 2

        // Distance: 2 ops per subtract-square, outputDim per pair
        let distanceOps = numQueries * numDatabase * outputDim * 2

        let totalOps = projectionOps + distanceOps
        let throughput = Double(totalOps) / (computeTime * 1e9)

        let stats = PerformanceStats(
            computeTime: computeTime,
            throughput: throughput,
            projectionOps: projectionOps,
            distanceOps: distanceOps
        )

        return (results, stats)
    }
}
