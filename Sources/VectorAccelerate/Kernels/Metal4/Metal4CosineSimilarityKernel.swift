//
//  Metal4CosineSimilarityKernel.swift
//  VectorAccelerate
//
//  Metal 4 Cosine Similarity kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 1, Priority 1
//
//  Features:
//  - Dimension-specific optimized pipelines (384, 512, 768, 1536)
//  - Separate pipelines for normalized vs general inputs
//  - Supports kernel fusion via encode() API
//  - Output as similarity (dot product of normalized) or distance (1 - similarity)

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Parameters

/// Parameters for Cosine Similarity kernel.
///
/// Memory layout must match the Metal shader's `CosineSimilarityParams` struct.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4CosineSimilarityParameters: Sendable {
    /// Number of query vectors (N)
    public let numQueries: UInt32

    /// Number of database vectors (M)
    public let numDatabase: UInt32

    /// Vector dimension (D)
    public let dimension: UInt32

    /// Stride between query vectors
    public let strideQuery: UInt32

    /// Stride between database vectors
    public let strideDatabase: UInt32

    /// Stride for output matrix rows
    public let strideOutput: UInt32

    /// Output mode: 0 = similarity [-1, 1], 1 = distance [0, 2]
    public let outputDistance: UInt8

    /// Whether inputs are pre-normalized: 0 = no, 1 = yes
    public let inputsNormalized: UInt8

    /// Padding for alignment
    private let padding: (UInt8, UInt8) = (0, 0)

    /// Create parameters for dense vector storage.
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors
    ///   - numDatabase: Number of database vectors
    ///   - dimension: Vector dimension
    ///   - outputDistance: If true, output 1-similarity. If false, output similarity.
    ///   - inputsNormalized: If true, skip normalization (inputs are unit vectors).
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(dimension)
        self.strideDatabase = UInt32(dimension)
        self.strideOutput = UInt32(numDatabase)
        self.outputDistance = outputDistance ? 1 : 0
        self.inputsNormalized = inputsNormalized ? 1 : 0
    }

    /// Create parameters with explicit strides.
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        strideQuery: Int,
        strideDatabase: Int,
        strideOutput: Int,
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(strideQuery)
        self.strideDatabase = UInt32(strideDatabase)
        self.strideOutput = UInt32(strideOutput)
        self.outputDistance = outputDistance ? 1 : 0
        self.inputsNormalized = inputsNormalized ? 1 : 0
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Cosine Similarity kernel.
///
/// Computes cosine similarity between all pairs of query and database vectors:
/// ```
/// similarity[i,j] = (query[i] · database[j]) / (||query[i]|| × ||database[j]||)
/// ```
///
/// Or cosine distance when `outputDistance` is true:
/// ```
/// distance[i,j] = 1 - similarity[i,j]
/// ```
///
/// ## Performance Optimizations
///
/// - **Pre-normalized path**: When `inputsNormalized` is true, skips norm computation
/// - **Dimension-specific pipelines**: Optimized for 384, 512, 768, 1536 dimensions
/// - **SIMD vectorization**: float4 operations for memory coalescing
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4CosineSimilarityKernel(context: context)
///
/// // For pre-normalized embeddings (faster)
/// let similarities = try await kernel.compute(
///     queries: normalizedQueries,
///     database: normalizedDatabase,
///     inputsNormalized: true
/// )
///
/// // For raw embeddings
/// let similarities = try await kernel.compute(
///     queries: rawQueries,
///     database: rawDatabase,
///     inputsNormalized: false
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4CosineSimilarityKernel: @unchecked Sendable, DimensionOptimizedKernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4CosineSimilarityKernel"

    public let optimizedDimensions: [Int] = [384, 512, 768, 1536]
    public let fusibleWith: [String] = ["TopKSelection", "L2Normalization"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    /// Pipeline for pre-normalized inputs (faster)
    private let pipelineNormalized: any MTLComputePipelineState

    /// Pipeline for general (unnormalized) inputs
    private let pipelineGeneral: any MTLComputePipelineState

    /// Dimension-specific optimized pipelines
    private let pipeline384: any MTLComputePipelineState
    private let pipeline512: any MTLComputePipelineState
    private let pipeline768: any MTLComputePipelineState
    private let pipeline1536: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Cosine Similarity kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        // Load kernel functions
        guard let normalizedFunc = library.makeFunction(name: "cosine_similarity_normalized_kernel"),
              let generalFunc = library.makeFunction(name: "cosine_similarity_general_kernel"),
              let func384 = library.makeFunction(name: "cosine_similarity_384_kernel"),
              let func512 = library.makeFunction(name: "cosine_similarity_512_kernel"),
              let func768 = library.makeFunction(name: "cosine_similarity_768_kernel"),
              let func1536 = library.makeFunction(name: "cosine_similarity_1536_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Cosine similarity kernels. Ensure CosineSimilarity.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipelineNormalized = try await device.makeComputePipelineState(function: normalizedFunc)
        self.pipelineGeneral = try await device.makeComputePipelineState(function: generalFunc)
        self.pipeline384 = try await device.makeComputePipelineState(function: func384)
        self.pipeline512 = try await device.makeComputePipelineState(function: func512)
        self.pipeline768 = try await device.makeComputePipelineState(function: func768)
        self.pipeline1536 = try await device.makeComputePipelineState(function: func1536)
    }

    // MARK: - Pipeline Selection

    /// Select optimal pipeline based on dimension and normalization state.
    private func selectPipeline(
        for dimension: UInt32,
        inputsNormalized: Bool
    ) -> (pipeline: any MTLComputePipelineState, name: String) {
        // Use dimension-specific pipelines when available (these handle both cases)
        switch dimension {
        case 384:
            return (pipeline384, "cosine_similarity_384_kernel")
        case 512:
            return (pipeline512, "cosine_similarity_512_kernel")
        case 768:
            return (pipeline768, "cosine_similarity_768_kernel")
        case 1536:
            return (pipeline1536, "cosine_similarity_1536_kernel")
        default:
            // Fall back to general or normalized pipeline
            if inputsNormalized {
                return (pipelineNormalized, "cosine_similarity_normalized_kernel")
            } else {
                return (pipelineGeneral, "cosine_similarity_general_kernel")
            }
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode cosine similarity computation into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        output: any MTLBuffer,
        parameters: Metal4CosineSimilarityParameters
    ) -> Metal4EncodingResult {
        let (pipeline, pipelineName) = selectPipeline(
            for: parameters.dimension,
            inputsNormalized: parameters.inputsNormalized == 1
        )

        encoder.setComputePipelineState(pipeline)
        encoder.label = "CosineSimilarity.\(pipelineName)"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(database, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Metal4CosineSimilarityParameters>.size, index: 3)

        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numQueries),
            numDatabase: Int(parameters.numDatabase),
            pipeline: pipeline
        )

        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )

        return Metal4EncodingResult(
            pipelineName: pipelineName,
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute cosine similarity as standalone operation.
    public func execute(
        queries: any MTLBuffer,
        database: any MTLBuffer,
        parameters: Metal4CosineSimilarityParameters
    ) async throws -> any MTLBuffer {
        let outputSize = Int(parameters.numQueries) * Int(parameters.numDatabase) * MemoryLayout<Float>.size
        guard let outputBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "CosineSimilarity.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                queries: queries,
                database: database,
                output: outputBuffer,
                parameters: parameters
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Compute cosine similarities from Float arrays.
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty, !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }),
              database.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let numQueries = queries.count
        let numDatabase = database.count
        let device = context.device.rawDevice

        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }

        guard let databaseBuffer = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDatabase.count * MemoryLayout<Float>.size)
        }

        let parameters = Metal4CosineSimilarityParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            outputDistance: outputDistance,
            inputsNormalized: inputsNormalized
        )

        let outputBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(from: outputBuffer, numQueries: numQueries, numDatabase: numDatabase)
    }

    /// Compute cosine similarities from VectorProtocol types.
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty, !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }),
              database.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let numQueries = queries.count
        let numDatabase = database.count
        let device = context.device.rawDevice

        let queryBuffer = try createBuffer(from: queries, device: device, label: "CosineSimilarity.queries")
        let databaseBuffer = try createBuffer(from: database, device: device, label: "CosineSimilarity.database")

        let parameters = Metal4CosineSimilarityParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            outputDistance: outputDistance,
            inputsNormalized: inputsNormalized
        )

        let outputBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(from: outputBuffer, numQueries: numQueries, numDatabase: numDatabase)
    }

    // MARK: - Private Helpers

    private func createBuffer<V: VectorProtocol>(
        from vectors: [V],
        device: any MTLDevice,
        label: String
    ) throws -> any MTLBuffer where V.Scalar == Float {
        let dimension = vectors[0].count
        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.size

        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: byteSize)
        }
        buffer.label = label

        let destination = buffer.contents().bindMemory(to: Float.self, capacity: totalCount)
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            vector.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                destination.advanced(by: offset).update(from: srcBase, count: min(srcPtr.count, dimension))
            }
        }

        return buffer
    }

    private func extractResults(
        from buffer: any MTLBuffer,
        numQueries: Int,
        numDatabase: Int
    ) -> [[Float]] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: numQueries * numDatabase)
        var results: [[Float]] = []
        results.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numDatabase)
            for j in 0..<numDatabase {
                row.append(pointer[i * numDatabase + j])
            }
            results.append(row)
        }
        return results
    }
}

// MARK: - Metal4DistanceKernel Conformance

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4CosineSimilarityKernel: Metal4DistanceKernel {
    public typealias Parameters = Metal4CosineSimilarityParameters

    /// Protocol conformance: encode into encoder with distances output buffer.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: Metal4CosineSimilarityParameters
    ) -> Metal4EncodingResult {
        // Delegate to the main encode method with output buffer
        encode(into: encoder, queries: queries, database: database, output: distances, parameters: parameters)
    }
}
