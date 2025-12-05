//
//  AttentionSimilarityKernel.swift
//  VectorAccelerate
//
//  Metal 4 Attention-based Similarity kernel with learned projections.
//
//  Phase 4: ML Integration - Attention Similarity (P3)
//
//  Features:
//  - Cross-attention between query and key vectors
//  - Learned query/key projections (Wq, Wk)
//  - Scaled dot-product attention scoring
//  - Multi-head attention support
//  - Temperature scaling for similarity sharpness
//
//  Key Advantages over Cosine Similarity:
//  - Learns asymmetric query-key relationships
//  - Projects to optimized attention space
//  - Supports domain-specific similarity patterns
//  - Multi-head attention captures diverse similarity aspects

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for attention-based similarity.
public struct Metal4AttentionSimilarityConfig: Sendable {
    /// Input dimension of vectors
    public let inputDimension: Int
    /// Head dimension (typically inputDim / numHeads)
    public let headDimension: Int
    /// Number of attention heads
    public let numHeads: Int
    /// Temperature for scaling (sqrt(headDim) by default)
    public let temperature: Float?
    /// Whether to normalize output similarities to [0, 1]
    public let normalizeSimilarities: Bool

    public init(
        inputDimension: Int,
        headDimension: Int? = nil,
        numHeads: Int = 1,
        temperature: Float? = nil,
        normalizeSimilarities: Bool = false
    ) {
        self.inputDimension = inputDimension
        self.headDimension = headDimension ?? inputDimension
        self.numHeads = numHeads
        self.temperature = temperature
        self.normalizeSimilarities = normalizeSimilarities
    }

    /// Effective temperature (default: sqrt(headDim))
    public var effectiveTemperature: Float {
        temperature ?? sqrt(Float(headDimension))
    }

    /// Common configuration for transformer-style embeddings (768-dim, 12 heads)
    public static func transformer768() -> Metal4AttentionSimilarityConfig {
        Metal4AttentionSimilarityConfig(
            inputDimension: 768,
            headDimension: 64,
            numHeads: 12
        )
    }

    /// Common configuration for MiniLM embeddings (384-dim, 6 heads)
    public static func miniLM() -> Metal4AttentionSimilarityConfig {
        Metal4AttentionSimilarityConfig(
            inputDimension: 384,
            headDimension: 64,
            numHeads: 6
        )
    }

    /// Single-head attention for simple similarity
    public static func singleHead(inputDim: Int, projectedDim: Int) -> Metal4AttentionSimilarityConfig {
        Metal4AttentionSimilarityConfig(
            inputDimension: inputDim,
            headDimension: projectedDim,
            numHeads: 1
        )
    }
}

// MARK: - Parameters

/// Parameters for attention similarity kernel (matches Metal struct).
public struct AttentionSimilarityParameters: Sendable {
    public var numQueries: UInt32
    public var numKeys: UInt32
    public var inputDimension: UInt32
    public var headDimension: UInt32
    public var numHeads: UInt32
    public var strideQuery: UInt32
    public var strideKey: UInt32
    public var strideOutput: UInt32
    public var temperature: Float
    public var normalizeSimilarities: UInt8
    private var padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

    public init(
        numQueries: Int,
        numKeys: Int,
        config: Metal4AttentionSimilarityConfig
    ) {
        self.numQueries = UInt32(numQueries)
        self.numKeys = UInt32(numKeys)
        self.inputDimension = UInt32(config.inputDimension)
        self.headDimension = UInt32(config.headDimension)
        self.numHeads = UInt32(config.numHeads)
        self.strideQuery = UInt32(config.inputDimension)
        self.strideKey = UInt32(config.inputDimension)
        self.strideOutput = UInt32(numKeys)
        self.temperature = config.effectiveTemperature
        self.normalizeSimilarities = config.normalizeSimilarities ? 1 : 0
    }
}

// MARK: - Result Types

/// Result from attention similarity computation.
public struct Metal4AttentionSimilarityResult: Sendable {
    /// Similarity scores [numQueries, numKeys]
    public let similarities: [[Float]]
    /// Computation time
    public let computeTime: TimeInterval
    /// Throughput in query-key pairs per second
    public let throughput: Double

    /// Get similarity between specific query and key
    public func similarity(query: Int, key: Int) -> Float {
        guard query < similarities.count, key < similarities[query].count else {
            return 0
        }
        return similarities[query][key]
    }

    /// Get top-k most similar keys for a query
    public func topK(forQuery query: Int, k: Int) -> [(index: Int, similarity: Float)] {
        guard query < similarities.count else { return [] }
        let sims = similarities[query]
        return sims.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map { (index: $0.offset, similarity: $0.element) }
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Attention-based Similarity kernel.
///
/// Computes similarity between query and key vectors using scaled dot-product
/// attention with learned projections. This enables capturing asymmetric and
/// domain-specific similarity patterns.
///
/// ## Architecture
///
/// For single-head attention:
/// ```
/// Query (D) → Wq (D→H) → q (H)
/// Key (D) → Wk (D→H) → k (H)
/// Similarity = (q · k) / temperature
/// ```
///
/// For multi-head attention:
/// ```
/// Query (D) → [Wq1..WqN] → [q1..qN] (N×H)
/// Key (D) → [Wk1..WkN] → [k1..kN] (N×H)
/// Similarity = mean(qi · ki) / temperature for i in 1..N
/// ```
///
/// ## Weight Format
///
/// - Query projection: [numHeads * headDim, inputDim] row-major float32
/// - Key projection: [numHeads * headDim, inputDim] row-major float32
///
/// ## Usage
///
/// ```swift
/// let kernel = try await AttentionSimilarityKernel(context: context)
///
/// // Load pre-trained attention weights
/// try await kernel.loadWeights(
///     queryProjectionURL: wqURL,
///     keyProjectionURL: wkURL,
///     config: .singleHead(inputDim: 768, projectedDim: 128)
/// )
///
/// // Compute similarity matrix
/// let result = try await kernel.computeSimilarities(
///     queries: queryVectors,
///     keys: keyVectors
/// )
///
/// // Get top-5 matches for first query
/// let matches = result.topK(forQuery: 0, k: 5)
/// ```
public final class AttentionSimilarityKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "AttentionSimilarityKernel"

    // MARK: - Pipelines

    private let singleHeadPipeline: any MTLComputePipelineState
    private let multiHeadPipeline: any MTLComputePipelineState

    // MARK: - Weight Management

    private let tensorManager: TensorManager
    private var currentConfig: Metal4AttentionSimilarityConfig?
    private var queryProjection: TensorBuffer?
    private var keyProjection: TensorBuffer?

    // MARK: - Initialization

    /// Create a Metal 4 Attention Similarity kernel.
    ///
    /// - Parameter context: Metal 4 context for execution
    /// - Throws: `VectorError` if shader compilation fails
    public init(context: Metal4Context) async throws {
        self.context = context
        self.tensorManager = TensorManager(device: context.device.rawDevice)

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let singleHeadFunc = library.makeFunction(name: "attention_similarity_kernel"),
              let multiHeadFunc = library.makeFunction(name: "multihead_attention_similarity_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Attention similarity kernels. Ensure AttentionSimilarity.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.singleHeadPipeline = try await device.makeComputePipelineState(function: singleHeadFunc)
        self.multiHeadPipeline = try await device.makeComputePipelineState(function: multiHeadFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Weight Loading

    /// Load query and key projection weights from files.
    ///
    /// Weight format: binary float32, row-major.
    /// - Query projection: [numHeads * headDim, inputDim]
    /// - Key projection: [numHeads * headDim, inputDim]
    ///
    /// - Parameters:
    ///   - queryProjectionURL: URL to query projection weight file
    ///   - keyProjectionURL: URL to key projection weight file
    ///   - config: Attention similarity configuration
    public func loadWeights(
        queryProjectionURL: URL,
        keyProjectionURL: URL,
        config: Metal4AttentionSimilarityConfig
    ) throws {
        let projectionDim = config.numHeads * config.headDimension
        let projectionShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: projectionDim
        )

        queryProjection = try tensorManager.loadWeights(
            from: queryProjectionURL,
            name: "attention_query",
            shape: projectionShape,
            dataType: .float32
        )

        keyProjection = try tensorManager.loadWeights(
            from: keyProjectionURL,
            name: "attention_key",
            shape: projectionShape,
            dataType: .float32
        )

        currentConfig = config
    }

    /// Load weights from Data objects.
    public func loadWeights(
        queryProjectionData: Data,
        keyProjectionData: Data,
        config: Metal4AttentionSimilarityConfig
    ) throws {
        let projectionDim = config.numHeads * config.headDimension
        let projectionShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: projectionDim
        )

        queryProjection = try tensorManager.loadWeights(
            from: queryProjectionData,
            name: "attention_query",
            shape: projectionShape,
            dataType: .float32
        )

        keyProjection = try tensorManager.loadWeights(
            from: keyProjectionData,
            name: "attention_key",
            shape: projectionShape,
            dataType: .float32
        )

        currentConfig = config
    }

    /// Load weights from Float arrays.
    ///
    /// - Parameters:
    ///   - queryProjection: Query projection weights [projDim * inputDim], row-major
    ///   - keyProjection: Key projection weights [projDim * inputDim], row-major
    ///   - config: Attention similarity configuration
    public func loadWeights(
        queryProjection: [Float],
        keyProjection: [Float],
        config: Metal4AttentionSimilarityConfig
    ) throws {
        let projectionDim = config.numHeads * config.headDimension
        let projectionShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: projectionDim
        )

        self.queryProjection = try tensorManager.createTensor(
            from: queryProjection,
            name: "attention_query",
            shape: projectionShape
        )

        self.keyProjection = try tensorManager.createTensor(
            from: keyProjection,
            name: "attention_key",
            shape: projectionShape
        )

        currentConfig = config
    }

    /// Create random weights for testing/initialization.
    ///
    /// Uses Xavier/Glorot initialization for proper scaling.
    public func createRandomWeights(
        config: Metal4AttentionSimilarityConfig
    ) throws {
        let projectionDim = config.numHeads * config.headDimension

        queryProjection = try tensorManager.createRandomProjection(
            inputDim: config.inputDimension,
            outputDim: projectionDim,
            name: "attention_query"
        )

        keyProjection = try tensorManager.createRandomProjection(
            inputDim: config.inputDimension,
            outputDim: projectionDim,
            name: "attention_key"
        )

        currentConfig = config
    }

    /// Check if weights are loaded.
    public var hasWeights: Bool {
        queryProjection != nil && keyProjection != nil
    }

    /// Unload weights to free memory.
    public func unloadWeights() {
        tensorManager.unload(name: "attention_query")
        tensorManager.unload(name: "attention_key")
        queryProjection = nil
        keyProjection = nil
        currentConfig = nil
    }

    // MARK: - Encode API

    /// Encode attention similarity computation into an encoder.
    @discardableResult
    public func encodeAttentionSimilarity(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        keys: any MTLBuffer,
        output: any MTLBuffer,
        parameters: AttentionSimilarityParameters
    ) throws -> Metal4EncodingResult {
        guard let queryProj = queryProjection, let keyProj = keyProjection else {
            throw VectorError.invalidOperation("Projection weights not loaded")
        }

        let pipeline = parameters.numHeads > 1 ? multiHeadPipeline : singleHeadPipeline
        let pipelineName = parameters.numHeads > 1 ? "multihead_attention_similarity_kernel" : "attention_similarity_kernel"

        encoder.setComputePipelineState(pipeline)
        encoder.label = "AttentionSimilarity"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(keys, offset: 0, index: 1)
        encoder.setBuffer(queryProj.buffer, offset: 0, index: 2)
        encoder.setBuffer(keyProj.buffer, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<AttentionSimilarityParameters>.size, index: 5)

        // 2D dispatch: (numQueries, numKeys)
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numQueries),
            numDatabase: Int(parameters.numKeys),
            pipeline: pipeline
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: pipelineName,
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute attention similarity as standalone operation.
    public func execute(
        queries: any MTLBuffer,
        keys: any MTLBuffer,
        parameters: AttentionSimilarityParameters
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let outputSize = Int(parameters.numQueries) * Int(parameters.numKeys) * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "AttentionSimilarity.output"

        try await context.executeAndWait { [self] _, encoder in
            try self.encodeAttentionSimilarity(
                into: encoder,
                queries: queries,
                keys: keys,
                output: outputBuffer,
                parameters: parameters
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Compute attention-based similarities between query and key vectors.
    ///
    /// - Parameters:
    ///   - queries: Query vectors [N, inputDim]
    ///   - keys: Key vectors [M, inputDim]
    /// - Returns: Similarity result with [N, M] similarity matrix
    public func computeSimilarities(
        queries: [[Float]],
        keys: [[Float]]
    ) async throws -> Metal4AttentionSimilarityResult {
        guard let config = currentConfig else {
            throw VectorError.invalidOperation("Weights not loaded. Call loadWeights first.")
        }

        guard !queries.isEmpty && !keys.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let inputDim = queries[0].count
        guard inputDim == config.inputDimension else {
            throw VectorError.countMismatch(expected: config.inputDimension, actual: inputDim)
        }
        guard keys[0].count == config.inputDimension else {
            throw VectorError.countMismatch(expected: config.inputDimension, actual: keys[0].count)
        }

        let device = context.device.rawDevice
        let numQueries = queries.count
        let numKeys = keys.count

        // Create input buffers
        let flatQueries = queries.flatMap { $0 }
        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "AttentionSimilarity.queries"

        let flatKeys = keys.flatMap { $0 }
        guard let keyBuffer = device.makeBuffer(
            bytes: flatKeys,
            length: flatKeys.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatKeys.count * MemoryLayout<Float>.size)
        }
        keyBuffer.label = "AttentionSimilarity.keys"

        let parameters = AttentionSimilarityParameters(
            numQueries: numQueries,
            numKeys: numKeys,
            config: config
        )

        let startTime = CACurrentMediaTime()
        let outputBuffer = try await execute(
            queries: queryBuffer,
            keys: keyBuffer,
            parameters: parameters
        )
        let computeTime = CACurrentMediaTime() - startTime

        // Extract results
        let outputPtr = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numKeys
        )

        var similarities: [[Float]] = []
        similarities.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numKeys)
            for j in 0..<numKeys {
                row.append(outputPtr[i * numKeys + j])
            }
            similarities.append(row)
        }

        let throughput = Double(numQueries * numKeys) / computeTime

        return Metal4AttentionSimilarityResult(
            similarities: similarities,
            computeTime: computeTime,
            throughput: throughput
        )
    }

    /// Compute similarities using VectorCore types.
    public func computeSimilarities<V: VectorProtocol>(
        queries: [V],
        keys: [V]
    ) async throws -> Metal4AttentionSimilarityResult where V.Scalar == Float {
        let queryArrays = queries.map { vec -> [Float] in
            vec.withUnsafeBufferPointer { Array($0) }
        }
        let keyArrays = keys.map { vec -> [Float] in
            vec.withUnsafeBufferPointer { Array($0) }
        }
        return try await computeSimilarities(queries: queryArrays, keys: keyArrays)
    }

    // MARK: - Statistics

    /// Get tensor manager statistics.
    public func getTensorStatistics() -> TensorManagerStatistics {
        tensorManager.getStatistics()
    }
}
