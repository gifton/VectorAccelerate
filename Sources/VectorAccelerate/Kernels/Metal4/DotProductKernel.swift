//
//  DotProductKernel.swift
//  VectorAccelerate
//
//  Metal 4 Dot Product kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 1, Priority 1
//
//  Features:
//  - Dimension-specific optimized pipelines (384, 512, 768, 1536)
//  - GEMV optimization for single-query scenarios
//  - Supports kernel fusion via encode() API
//  - Optional absolute value output

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Parameters

/// Parameters for Dot Product kernel.
///
/// Memory layout must match the Metal shader's `DotProductParams` struct.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct DotProductParameters: Sendable {
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

    /// Output absolute value: 0 = normal, 1 = |dot product|
    public let absoluteValue: UInt8

    /// Padding for alignment
    private let padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

    /// Create parameters for dense vector storage.
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors
    ///   - numDatabase: Number of database vectors
    ///   - dimension: Vector dimension
    ///   - absoluteValue: If true, output |dot product| instead of dot product
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        absoluteValue: Bool = false
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(dimension)
        self.strideDatabase = UInt32(dimension)
        // For GEMV (N=1), stride is 1; for GEMM (N>1), stride is numDatabase
        self.strideOutput = UInt32(numQueries > 1 ? numDatabase : 1)
        self.absoluteValue = absoluteValue ? 1 : 0
    }

    /// Create parameters with explicit strides.
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        strideQuery: Int,
        strideDatabase: Int,
        strideOutput: Int,
        absoluteValue: Bool = false
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(strideQuery)
        self.strideDatabase = UInt32(strideDatabase)
        self.strideOutput = UInt32(strideOutput)
        self.absoluteValue = absoluteValue ? 1 : 0
    }

    /// Whether this is a single-query (GEMV) scenario
    public var isGEMV: Bool {
        numQueries == 1
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Dot Product kernel.
///
/// Computes dot product between all pairs of query and database vectors:
/// ```
/// output[i,j] = query[i] · database[j] = Σ(query[i][k] × database[j][k])
/// ```
///
/// ## Performance Optimizations
///
/// - **GEMV path**: Single-query scenarios use optimized matrix-vector multiply
/// - **Dimension-specific pipelines**: Optimized for 384, 512, 768, 1536 dimensions
/// - **SIMD vectorization**: float4 operations for memory coalescing
///
/// ## Use Cases
///
/// - **Maximum Inner Product Search (MIPS)**: Find vectors with highest dot product
/// - **Attention scores**: Compute Q·K^T in attention mechanisms
/// - **Feature correlation**: Measure alignment between feature vectors
///
/// ## Usage
///
/// ```swift
/// let kernel = try await DotProductKernel(context: context)
///
/// // Standard batch computation
/// let products = try await kernel.compute(
///     queries: queries,
///     database: database
/// )
///
/// // Single query (uses GEMV optimization)
/// let scores = try await kernel.compute(
///     queries: [singleQuery],
///     database: database
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class DotProductKernel: @unchecked Sendable, DimensionOptimizedKernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "DotProductKernel"

    public let optimizedDimensions: [Int] = [384, 512, 768, 1536]
    public let fusibleWith: [String] = ["TopKSelection", "MatrixOps"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    /// Generic GEMM-style pipeline
    private let pipelineGeneric: any MTLComputePipelineState

    /// GEMV-optimized pipeline for single query
    private let pipelineGEMV: any MTLComputePipelineState

    /// Dimension-specific optimized pipelines
    private let pipeline384: any MTLComputePipelineState
    private let pipeline512: any MTLComputePipelineState
    private let pipeline768: any MTLComputePipelineState
    private let pipeline1536: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Dot Product kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let genericFunc = library.makeFunction(name: "dot_product_kernel"),
              let gemvFunc = library.makeFunction(name: "dot_product_gemv_kernel"),
              let func384 = library.makeFunction(name: "dot_product_384_kernel"),
              let func512 = library.makeFunction(name: "dot_product_512_kernel"),
              let func768 = library.makeFunction(name: "dot_product_768_kernel"),
              let func1536 = library.makeFunction(name: "dot_product_1536_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Dot product kernels. Ensure DotProduct.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipelineGeneric = try await device.makeComputePipelineState(function: genericFunc)
        self.pipelineGEMV = try await device.makeComputePipelineState(function: gemvFunc)
        self.pipeline384 = try await device.makeComputePipelineState(function: func384)
        self.pipeline512 = try await device.makeComputePipelineState(function: func512)
        self.pipeline768 = try await device.makeComputePipelineState(function: func768)
        self.pipeline1536 = try await device.makeComputePipelineState(function: func1536)
    }

    // MARK: - Pipeline Selection

    /// Select optimal pipeline based on dimension and query count.
    private func selectPipeline(
        for dimension: UInt32,
        isGEMV: Bool
    ) -> (pipeline: any MTLComputePipelineState, name: String) {
        // GEMV path for single query
        if isGEMV {
            return (pipelineGEMV, "dot_product_gemv_kernel")
        }

        // Dimension-specific for batch operations
        switch dimension {
        case 384:
            return (pipeline384, "dot_product_384_kernel")
        case 512:
            return (pipeline512, "dot_product_512_kernel")
        case 768:
            return (pipeline768, "dot_product_768_kernel")
        case 1536:
            return (pipeline1536, "dot_product_1536_kernel")
        default:
            return (pipelineGeneric, "dot_product_kernel")
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode dot product computation into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        output: any MTLBuffer,
        parameters: DotProductParameters
    ) -> Metal4EncodingResult {
        let (pipeline, pipelineName) = selectPipeline(
            for: parameters.dimension,
            isGEMV: parameters.isGEMV
        )

        encoder.setComputePipelineState(pipeline)
        encoder.label = "DotProduct.\(pipelineName)"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(database, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<DotProductParameters>.size, index: 3)

        // Thread configuration differs for GEMV vs GEMM
        let config: Metal4ThreadConfiguration
        if parameters.isGEMV {
            // 1D dispatch for GEMV
            config = Metal4ThreadConfiguration.linear(
                count: Int(parameters.numDatabase),
                pipeline: pipeline
            )
        } else {
            // 2D dispatch for GEMM
            config = Metal4ThreadConfiguration.forDistanceKernel(
                numQueries: Int(parameters.numQueries),
                numDatabase: Int(parameters.numDatabase),
                pipeline: pipeline
            )
        }

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

    /// Execute dot product as standalone operation.
    public func execute(
        queries: any MTLBuffer,
        database: any MTLBuffer,
        parameters: DotProductParameters
    ) async throws -> any MTLBuffer {
        let outputSize = Int(parameters.numQueries) * Int(parameters.numDatabase) * MemoryLayout<Float>.size
        guard let outputBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "DotProduct.output"

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

    /// Compute dot products from Float arrays.
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        absoluteValue: Bool = false
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

        let parameters = DotProductParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            absoluteValue: absoluteValue
        )

        let outputBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(from: outputBuffer, numQueries: numQueries, numDatabase: numDatabase)
    }

    /// Compute dot products from VectorProtocol types.
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        absoluteValue: Bool = false
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

        let queryBuffer = try createBuffer(from: queries, device: device, label: "DotProduct.queries")
        let databaseBuffer = try createBuffer(from: database, device: device, label: "DotProduct.database")

        let parameters = DotProductParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            absoluteValue: absoluteValue
        )

        let outputBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(from: outputBuffer, numQueries: numQueries, numDatabase: numDatabase)
    }

    /// Compute single query against database (GEMV-optimized).
    ///
    /// Returns a 1D array of dot products.
    public func computeSingle(
        query: [Float],
        database: [[Float]],
        absoluteValue: Bool = false
    ) async throws -> [Float] {
        let results = try await compute(
            queries: [query],
            database: database,
            absoluteValue: absoluteValue
        )
        return results[0]
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
extension DotProductKernel: Metal4DistanceKernel {
    public typealias Parameters = DotProductParameters

    /// Protocol conformance: encode into encoder with distances output buffer.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: DotProductParameters
    ) -> Metal4EncodingResult {
        // Delegate to the main encode method with output buffer
        encode(into: encoder, queries: queries, database: database, output: distances, parameters: parameters)
    }
}
