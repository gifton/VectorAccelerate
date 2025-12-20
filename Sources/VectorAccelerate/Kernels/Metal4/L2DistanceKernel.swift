//
//  L2DistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 L2 (Euclidean) distance kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 1, Priority 1
//
//  Features:
//  - Dimension-specific optimized pipelines (384, 512, 768, 1536)
//  - ArgumentTable-based buffer binding
//  - Supports kernel fusion via encode() API
//  - Automatic residency management
//  - VectorCore type integration

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Parameters

/// Parameters for L2 distance kernel.
///
/// Memory layout must match the Metal shader's `L2DistanceParams` struct.
/// Uses explicit padding for alignment.
public struct L2DistanceParameters: Sendable {
    /// Number of query vectors (N)
    public let numQueries: UInt32

    /// Number of database vectors (M)
    public let numDatabase: UInt32

    /// Vector dimension (D)
    public let dimension: UInt32

    /// Stride between query vectors (typically == dimension)
    public let strideQuery: UInt32

    /// Stride between database vectors (typically == dimension)
    public let strideDatabase: UInt32

    /// Stride for output matrix rows (typically == numDatabase)
    public let strideOutput: UInt32

    /// Whether to compute sqrt (1) or return squared distance (0)
    public let computeSqrt: UInt8

    /// Padding for 4-byte alignment
    private let padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

    /// Create parameters for dense (contiguous) vector storage.
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors
    ///   - numDatabase: Number of database vectors
    ///   - dimension: Vector dimension
    ///   - computeSqrt: If true, compute Euclidean distance. If false, squared distance.
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        computeSqrt: Bool = true
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(dimension)
        self.strideDatabase = UInt32(dimension)
        self.strideOutput = UInt32(numDatabase)
        self.computeSqrt = computeSqrt ? 1 : 0
    }

    /// Create parameters with explicit strides for non-contiguous layouts.
    ///
    /// Use this when vectors are part of a larger struct or have padding.
    public init(
        numQueries: Int,
        numDatabase: Int,
        dimension: Int,
        strideQuery: Int,
        strideDatabase: Int,
        strideOutput: Int,
        computeSqrt: Bool = true
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDatabase = UInt32(numDatabase)
        self.dimension = UInt32(dimension)
        self.strideQuery = UInt32(strideQuery)
        self.strideDatabase = UInt32(strideDatabase)
        self.strideOutput = UInt32(strideOutput)
        self.computeSqrt = computeSqrt ? 1 : 0
    }
}

// MARK: - Kernel Implementation

/// Metal 4 L2 (Euclidean) distance kernel.
///
/// Computes L2 distance between all pairs of query and database vectors:
/// ```
/// distance[i,j] = ||query[i] - database[j]||₂
/// ```
///
/// ## Performance Optimizations
///
/// - **Dimension-specific pipelines**: Hand-tuned kernels for common embedding
///   dimensions (384, 512, 768, 1536) with loop unrolling and register optimization
/// - **SIMD vectorization**: float4 operations for memory coalescing
/// - **2D dispatch**: Optimized thread group configuration for (query, database) pairs
///
/// ## Thread Safety
///
/// This kernel is thread-safe. All mutable state (pipelines) is initialized during
/// construction and never modified thereafter. The `encode()` method can be called
/// concurrently from multiple threads.
///
/// ## Embedding Dimension Coverage
///
/// | Dimension | Models | Kernel |
/// |-----------|--------|--------|
/// | 384 | MiniLM, all-MiniLM-L6-v2, Sentence-BERT | `l2_distance_384_kernel` |
/// | 512 | Small BERT variants | `l2_distance_512_kernel` |
/// | 768 | BERT-base, DistilBERT, MPNet | `l2_distance_768_kernel` |
/// | 1536 | OpenAI ada-002 | `l2_distance_1536_kernel` |
/// | Other | Any | `l2_distance_kernel` (generic) |
///
/// ## Usage
///
/// ### Standalone Execution
/// ```swift
/// let kernel = try await L2DistanceKernel(context: context)
/// let distances = try await kernel.execute(
///     queries: queryBuffer,
///     database: databaseBuffer,
///     parameters: .init(numQueries: 100, numDatabase: 10000, dimension: 768)
/// )
/// ```
///
/// ### Fused with Top-K Selection
/// ```swift
/// try await context.executeAndWait { commandBuffer, encoder in
///     try kernel.encode(into: encoder, ...)
///     encoder.memoryBarrier(scope: .buffers)
///     try topKKernel.encode(into: encoder, ...)
/// }
/// ```
public final class L2DistanceKernel: @unchecked Sendable, DimensionOptimizedKernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "L2DistanceKernel"

    public let optimizedDimensions: [Int] = [384, 512, 768, 1536]
    public let fusibleWith: [String] = ["TopKSelection", "L2Normalization"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines (Immutable After Init)

    /// Generic pipeline for arbitrary dimensions
    private let genericPipeline: any MTLComputePipelineState

    /// Dimension-specific optimized pipelines
    private let pipeline384: any MTLComputePipelineState
    private let pipeline512: any MTLComputePipelineState
    private let pipeline768: any MTLComputePipelineState
    private let pipeline1536: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 L2 distance kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    /// - Throws: `VectorError.shaderNotFound` if kernel functions are missing
    public init(context: Metal4Context) async throws {
        self.context = context

        // Load library
        let library = try await context.shaderCompiler.getDefaultLibrary()

        // Load kernel functions
        guard let genericFunc = library.makeFunction(name: "l2_distance_kernel"),
              let func384 = library.makeFunction(name: "l2_distance_384_kernel"),
              let func512 = library.makeFunction(name: "l2_distance_512_kernel"),
              let func768 = library.makeFunction(name: "l2_distance_768_kernel"),
              let func1536 = library.makeFunction(name: "l2_distance_1536_kernel") else {
            throw VectorError.shaderNotFound(
                name: "L2 distance kernels. Ensure L2Distance.metal is compiled."
            )
        }

        // Create pipeline states
        let device = context.device.rawDevice
        self.genericPipeline = try await device.makeComputePipelineState(function: genericFunc)
        self.pipeline384 = try await device.makeComputePipelineState(function: func384)
        self.pipeline512 = try await device.makeComputePipelineState(function: func512)
        self.pipeline768 = try await device.makeComputePipelineState(function: func768)
        self.pipeline1536 = try await device.makeComputePipelineState(function: func1536)
    }

    // MARK: - Pipeline Selection

    /// Select the optimal pipeline for a given dimension.
    private func selectPipeline(for dimension: UInt32) -> (pipeline: any MTLComputePipelineState, name: String) {
        switch dimension {
        case 384:
            return (pipeline384, "l2_distance_384_kernel")
        case 512:
            return (pipeline512, "l2_distance_512_kernel")
        case 768:
            return (pipeline768, "l2_distance_768_kernel")
        case 1536:
            return (pipeline1536, "l2_distance_1536_kernel")
        default:
            return (genericPipeline, "l2_distance_kernel")
        }
    }

    // MARK: - Warm Up

    /// Pre-warm pipelines (already done in init, but available for explicit warm-up).
    public func warmUp() async throws {
        // Pipelines are created in init, so this is a no-op
        // Could be extended to trigger initial dispatch for GPU shader cache
    }

    // MARK: - Encode API (for Fusion)

    /// Encode L2 distance computation into an existing encoder.
    ///
    /// This method does NOT create or end the encoder - it only adds dispatch commands.
    /// The encoder must have been created with `commandBuffer.makeComputeCommandEncoder()`.
    ///
    /// **Important**: If fusing with subsequent operations that read the distance buffer,
    /// insert `encoder.memoryBarrier(scope: .buffers)` after this call.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into
    ///   - queries: Query vectors buffer [N, D] - row-major float32
    ///   - database: Database vectors buffer [M, D] - row-major float32
    ///   - distances: Output buffer [N, M] - must be pre-allocated
    ///   - parameters: Execution parameters
    /// - Returns: Encoding result with dispatch configuration
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: L2DistanceParameters
    ) -> Metal4EncodingResult {
        // Select pipeline
        let (pipeline, pipelineName) = selectPipeline(for: parameters.dimension)

        // Configure encoder
        encoder.setComputePipelineState(pipeline)
        encoder.label = "L2Distance.\(pipelineName)"

        // Bind buffers directly (Metal 4 ArgumentTable can be used via context if needed)
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(database, offset: 0, index: 1)
        encoder.setBuffer(distances, offset: 0, index: 2)

        // Bind parameters
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<L2DistanceParameters>.size, index: 3)

        // Calculate thread configuration
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numQueries),
            numDatabase: Int(parameters.numDatabase),
            pipeline: pipeline
        )

        // Dispatch
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

    /// Encode L2 distance computation with explicit buffer offsets.
    ///
    /// This is primarily used for chunked processing, where the caller wants to compute
    /// distances for a subrange of the database buffer without copying.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - queries: Query vectors buffer
    ///   - queryOffset: Byte offset into `queries`
    ///   - database: Database vectors buffer
    ///   - databaseOffset: Byte offset into `database`
    ///   - distances: Output distance buffer
    ///   - distancesOffset: Byte offset into `distances`
    ///   - parameters: Execution parameters (must match the logical shapes)
    /// - Returns: Encoding result
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        queryOffset: Int,
        database: any MTLBuffer,
        databaseOffset: Int,
        distances: any MTLBuffer,
        distancesOffset: Int,
        parameters: L2DistanceParameters
    ) -> Metal4EncodingResult {
        // Select pipeline
        let (pipeline, pipelineName) = selectPipeline(for: parameters.dimension)

        // Configure encoder
        encoder.setComputePipelineState(pipeline)
        encoder.label = "L2Distance.\(pipelineName) (offsets)"

        // Bind buffers with offsets
        encoder.setBuffer(queries, offset: queryOffset, index: 0)
        encoder.setBuffer(database, offset: databaseOffset, index: 1)
        encoder.setBuffer(distances, offset: distancesOffset, index: 2)

        // Bind parameters
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<L2DistanceParameters>.size, index: 3)

        // Calculate thread configuration
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numQueries),
            numDatabase: Int(parameters.numDatabase),
            pipeline: pipeline
        )

        // Dispatch
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

    // MARK: - Execute API (Standalone)

    /// Execute L2 distance computation as a standalone operation.
    ///
    /// Creates a command buffer, encodes the operation, submits, and waits for completion.
    /// The output buffer is allocated automatically.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer [N, D]
    ///   - database: Database vectors buffer [M, D]
    ///   - parameters: Execution parameters
    /// - Returns: Buffer containing distances [N, M]
    public func execute(
        queries: any MTLBuffer,
        database: any MTLBuffer,
        parameters: L2DistanceParameters
    ) async throws -> any MTLBuffer {
        // Allocate output buffer
        let outputSize = Int(parameters.numQueries) * Int(parameters.numDatabase) * MemoryLayout<Float>.size
        guard let distanceBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        distanceBuffer.label = "L2Distance.output"

        // Execute via context
        try await context.executeAndWait { [self] commandBuffer, encoder in
            self.encode(
                into: encoder,
                queries: queries,
                database: database,
                distances: distanceBuffer,
                parameters: parameters
            )
        }

        return distanceBuffer
    }

    // MARK: - High-Level Array API

    /// Compute L2 distances from Float arrays.
    ///
    /// This is a convenience method that handles buffer allocation and data transfer.
    /// For maximum performance with large data, prefer the buffer-based APIs.
    ///
    /// - Parameters:
    ///   - queries: Query vectors as 2D array [numQueries][dimension]
    ///   - database: Database vectors as 2D array [numDatabase][dimension]
    ///   - computeSqrt: If true, return Euclidean distance. If false, squared distance.
    /// - Returns: Distance matrix [numQueries][numDatabase]
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        computeSqrt: Bool = true
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

        // Flatten arrays
        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        // Create buffers
        let device = context.device.rawDevice

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "L2Distance.queries"

        guard let databaseBuffer = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDatabase.count * MemoryLayout<Float>.size)
        }
        databaseBuffer.label = "L2Distance.database"

        // Execute
        let parameters = L2DistanceParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )

        let distanceBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        // Extract results
        return extractResults(
            from: distanceBuffer,
            numQueries: numQueries,
            numDatabase: numDatabase
        )
    }

    /// Compute L2 distances from VectorProtocol types.
    ///
    /// Uses zero-copy buffer creation when possible.
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        computeSqrt: Bool = true
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

        // Create buffers with zero-copy when possible
        let queryBuffer = try createBuffer(from: queries, device: device, label: "L2Distance.queries")
        let databaseBuffer = try createBuffer(from: database, device: device, label: "L2Distance.database")

        let parameters = L2DistanceParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )

        let distanceBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(
            from: distanceBuffer,
            numQueries: numQueries,
            numDatabase: numDatabase
        )
    }

    /// Compute L2 distances using StaticDimension vectors.
    ///
    /// Provides compile-time dimension safety.
    public func compute<D: StaticDimension>(
        queries: [Vector<D>],
        database: [Vector<D>],
        computeSqrt: Bool = true
    ) async throws -> [[Float]] {
        guard !queries.isEmpty, !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        // Dimension is known at compile time
        let dimension = D.value
        let numQueries = queries.count
        let numDatabase = database.count
        let device = context.device.rawDevice

        let queryBuffer = try createBuffer(from: queries, device: device, label: "L2Distance.queries")
        let databaseBuffer = try createBuffer(from: database, device: device, label: "L2Distance.database")

        let parameters = L2DistanceParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )

        let distanceBuffer = try await execute(
            queries: queryBuffer,
            database: databaseBuffer,
            parameters: parameters
        )

        return extractResults(
            from: distanceBuffer,
            numQueries: numQueries,
            numDatabase: numDatabase
        )
    }

    // MARK: - Private Helpers

    /// Create a Metal buffer from VectorProtocol array.
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

    /// Extract distance results from buffer into 2D array.
    private func extractResults(
        from buffer: any MTLBuffer,
        numQueries: Int,
        numDatabase: Int
    ) -> [[Float]] {
        let distancePointer = buffer.contents().bindMemory(
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
}

// MARK: - Metal4DistanceKernel Conformance

extension L2DistanceKernel: Metal4DistanceKernel {
    public typealias Parameters = L2DistanceParameters
}
