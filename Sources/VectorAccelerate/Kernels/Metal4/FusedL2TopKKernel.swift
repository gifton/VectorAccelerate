//
//  FusedL2TopKKernel.swift
//  VectorAccelerate
//
//  Metal 4 Fused L2 Distance + Top-K Selection kernel.
//
//  Phase 5: Kernel Migrations - Batch 2, Priority 2
//
//  Features:
//  - Single-pass L2 distance + top-k selection
//  - Avoids materializing full distance matrix
//  - Chunk-based streaming for large datasets
//  - Thread-safe for concurrent encoding

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Parameters

/// Parameters for Fused L2 Top-K kernel.
///
/// Memory layout must match the Metal shader expectations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct FusedL2TopKParameters: Sendable {
    /// Number of query vectors (Q)
    public let numQueries: UInt32

    /// Number of dataset vectors (N)
    public let numDataset: UInt32

    /// Vector dimension (D)
    public let dimension: UInt32

    /// Number of top elements to select (K)
    public let k: UInt32

    /// Maximum K supported by the kernel
    public static let maxK: Int = 128

    /// Maximum dimension supported by the kernel
    public static let maxDimension: Int = 512

    /// Create parameters for fused L2 + Top-K.
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors
    ///   - numDataset: Number of dataset vectors
    ///   - dimension: Vector dimension (must be <= 512)
    ///   - k: Number of nearest neighbors to find
    public init(
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int
    ) {
        self.numQueries = UInt32(numQueries)
        self.numDataset = UInt32(numDataset)
        self.dimension = UInt32(min(dimension, Self.maxDimension))
        self.k = UInt32(min(k, Self.maxK))
    }
}

// MARK: - Configuration

/// Configuration for fused operation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4FusedL2Config: Sendable {
    /// Whether to include distance values in output
    public let includeDistances: Bool

    /// Threadgroup size for kernel dispatch
    public let threadgroupSize: Int

    public init(includeDistances: Bool = true, threadgroupSize: Int = 256) {
        self.includeDistances = includeDistances
        self.threadgroupSize = min(threadgroupSize, 256)
    }

    /// Default configuration with distances included
    public static let `default` = Metal4FusedL2Config()
}

// MARK: - Result Type

/// Result from fused L2 + Top-K operation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4FusedL2TopKResult: Sendable {
    /// Buffer containing selected indices [numQueries × k]
    public let indices: any MTLBuffer

    /// Buffer containing selected distances [numQueries × k] (optional)
    public let distances: (any MTLBuffer)?

    /// Number of queries in batch
    public let numQueries: Int

    /// K value used
    public let k: Int

    /// Extract results for a specific query.
    public func results(for queryIndex: Int) -> [(index: Int, distance: Float)] {
        guard queryIndex < numQueries else { return [] }

        let offset = queryIndex * k
        let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)
        let distPtr = distances?.contents().bindMemory(to: Float.self, capacity: numQueries * k)

        var results: [(index: Int, distance: Float)] = []
        results.reserveCapacity(k)
        for i in 0..<k {
            let idx = indexPtr[offset + i]
            if idx != 0xFFFFFFFF {  // Skip sentinel values
                let dist = distPtr?[offset + i] ?? 0
                results.append((index: Int(idx), distance: dist))
            }
        }
        return results
    }

    /// Get all results as array of arrays.
    public func allResults() -> [[(index: Int, distance: Float)]] {
        (0..<numQueries).map { results(for: $0) }
    }
}

// MARK: - Streaming Update Parameters

/// Parameters for streaming update kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4StreamingL2Params: Sendable {
    var Q: UInt32
    var chunkSize: UInt32
    var D: UInt32
    var K: UInt32
    var offset: UInt32
}

// MARK: - Kernel Implementation

/// Metal 4 Fused L2 Distance + Top-K Selection kernel.
///
/// Computes L2 distances and selects top-k nearest neighbors in a single pass,
/// avoiding the memory overhead of materializing the full distance matrix.
///
/// ## Performance Characteristics
///
/// - **Memory**: O(Q × K) vs O(Q × N) for separate distance + selection
/// - **Algorithm**: Heap-based selection with O(N log K) complexity per query
/// - **Chunking**: Supports datasets larger than GPU memory via streaming
///
/// ## Usage
///
/// ```swift
/// let kernel = try await FusedL2TopKKernel(context: context)
///
/// // Single-pass nearest neighbor search
/// let result = try await kernel.findNearestNeighbors(
///     queries: queryVectors,
///     dataset: datasetVectors,
///     k: 10
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class FusedL2TopKKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "FusedL2TopKKernel"

    // MARK: - Pipelines

    /// Main fused L2 + Top-K pipeline
    private let fusedPipeline: any MTLComputePipelineState

    /// Streaming update pipeline for chunked processing
    private let streamingUpdatePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Fused L2 Top-K kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let fusedFunc = library.makeFunction(name: "fused_l2_topk") else {
            throw VectorError.shaderNotFound(
                name: "Fused L2 Top-K kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        guard let streamingFunc = library.makeFunction(name: "streaming_l2_topk_update") else {
            throw VectorError.shaderNotFound(
                name: "Streaming L2 Top-K update kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.fusedPipeline = try await device.makeComputePipelineState(function: fusedFunc)
        self.streamingUpdatePipeline = try await device.makeComputePipelineState(function: streamingFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode fused L2 + Top-K into an existing encoder.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - queries: Query vectors buffer [numQueries × dimension]
    ///   - dataset: Dataset vectors buffer [numDataset × dimension]
    ///   - outputIndices: Output buffer for selected indices [numQueries × k]
    ///   - outputDistances: Optional output buffer for distances [numQueries × k]
    ///   - parameters: Kernel parameters
    /// - Returns: Encoding result
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        outputIndices: any MTLBuffer,
        outputDistances: (any MTLBuffer)?,
        parameters: FusedL2TopKParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(fusedPipeline)
        encoder.label = "FusedL2TopK (K=\(parameters.k))"

        // Bind buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(outputIndices, offset: 0, index: 2)
        encoder.setBuffer(outputDistances, offset: 0, index: 3)

        // Bind parameters individually (matching Metal shader signature)
        var paramQ = parameters.numQueries
        var paramN = parameters.numDataset
        var paramD = parameters.dimension
        var paramK = parameters.k
        encoder.setBytes(&paramQ, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&paramN, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&paramD, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&paramK, length: MemoryLayout<UInt32>.size, index: 7)

        // Dispatch: one threadgroup per query
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: Int(parameters.numQueries), height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "fused_l2_topk",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    /// Encode streaming update for chunk processing.
    ///
    /// Used when processing dataset in chunks to update running top-k.
    @discardableResult
    public func encodeStreamingUpdate(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        chunk: any MTLBuffer,
        runningIndices: any MTLBuffer,
        runningDistances: any MTLBuffer,
        parameters: Metal4StreamingL2Params
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(streamingUpdatePipeline)
        encoder.label = "StreamingL2TopKUpdate"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(chunk, offset: 0, index: 1)
        encoder.setBuffer(runningIndices, offset: 0, index: 2)
        encoder.setBuffer(runningDistances, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Metal4StreamingL2Params>.size, index: 4)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: Int(parameters.Q), height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "streaming_l2_topk_update",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Execute fused L2 + Top-K as standalone operation.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer
    ///   - dataset: Dataset vectors buffer
    ///   - parameters: Kernel parameters
    ///   - config: Operation configuration
    /// - Returns: Fused result with indices and optional distances
    public func execute(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        parameters: FusedL2TopKParameters,
        config: Metal4FusedL2Config = .default
    ) async throws -> Metal4FusedL2TopKResult {
        let device = context.device.rawDevice
        let numQueries = Int(parameters.numQueries)
        let k = Int(parameters.k)

        // Validate parameters
        guard parameters.dimension <= FusedL2TopKParameters.maxDimension else {
            throw VectorError.invalidInput(
                "Dimension \(parameters.dimension) exceeds maximum \(FusedL2TopKParameters.maxDimension)"
            )
        }

        // Allocate output buffers
        let indicesSize = numQueries * k * MemoryLayout<UInt32>.size
        guard let indicesBuffer = device.makeBuffer(length: indicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: indicesSize)
        }
        indicesBuffer.label = "FusedL2TopK.indices"

        let distancesBuffer: (any MTLBuffer)?
        if config.includeDistances {
            let distancesSize = numQueries * k * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: distancesSize, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: distancesSize)
            }
            buffer.label = "FusedL2TopK.distances"
            distancesBuffer = buffer
        } else {
            distancesBuffer = nil
        }

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                queries: queries,
                dataset: dataset,
                outputIndices: indicesBuffer,
                outputDistances: distancesBuffer,
                parameters: parameters
            )
        }

        return Metal4FusedL2TopKResult(
            indices: indicesBuffer,
            distances: distancesBuffer,
            numQueries: numQueries,
            k: k
        )
    }

    // MARK: - High-Level API

    /// Find k nearest neighbors using fused L2 distance computation.
    ///
    /// - Parameters:
    ///   - queries: Query vectors as 2D Float array
    ///   - dataset: Dataset vectors as 2D Float array
    ///   - k: Number of nearest neighbors to find
    ///   - includeDistances: Whether to include distance values
    /// - Returns: Array of (index, distance) pairs for each query
    public func findNearestNeighbors(
        queries: [[Float]],
        dataset: [[Float]],
        k: Int,
        includeDistances: Bool = true
    ) async throws -> [[(index: Int, distance: Float)]] {
        guard !queries.isEmpty, !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input arrays")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }),
              dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        guard dimension <= FusedL2TopKParameters.maxDimension else {
            throw VectorError.invalidInput(
                "Dimension \(dimension) exceeds maximum \(FusedL2TopKParameters.maxDimension)"
            )
        }

        let device = context.device.rawDevice
        let flatQueries = queries.flatMap { $0 }
        let flatDataset = dataset.flatMap { $0 }

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "FusedL2TopK.queries"

        guard let datasetBuffer = device.makeBuffer(
            bytes: flatDataset,
            length: flatDataset.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDataset.count * MemoryLayout<Float>.size)
        }
        datasetBuffer.label = "FusedL2TopK.dataset"

        let parameters = FusedL2TopKParameters(
            numQueries: queries.count,
            numDataset: dataset.count,
            dimension: dimension,
            k: k
        )

        let result = try await execute(
            queries: queryBuffer,
            dataset: datasetBuffer,
            parameters: parameters,
            config: Metal4FusedL2Config(includeDistances: includeDistances)
        )

        return result.allResults()
    }

    /// Find nearest neighbors using VectorCore types.
    public func findNearestNeighbors<V: VectorProtocol>(
        queries: [V],
        dataset: [V],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        guard !queries.isEmpty, !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }),
              dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let device = context.device.rawDevice

        let queryBuffer = try createBuffer(from: queries, device: device, label: "FusedL2TopK.queries")
        let datasetBuffer = try createBuffer(from: dataset, device: device, label: "FusedL2TopK.dataset")

        let parameters = FusedL2TopKParameters(
            numQueries: queries.count,
            numDataset: dataset.count,
            dimension: dimension,
            k: k
        )

        let result = try await execute(
            queries: queryBuffer,
            dataset: datasetBuffer,
            parameters: parameters
        )

        return result.allResults()
    }

    // MARK: - Chunked Processing

    /// Process large dataset in chunks for datasets exceeding GPU memory.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer
    ///   - dataset: Dataset vectors buffer
    ///   - queryCount: Number of query vectors
    ///   - datasetCount: Total number of dataset vectors
    ///   - dimension: Vector dimension
    ///   - k: Number of nearest neighbors
    ///   - chunkSize: Vectors to process per chunk
    /// - Returns: Final top-k results
    public func chunkedFindNearestNeighbors(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        k: Int,
        chunkSize: Int = 100_000
    ) async throws -> Metal4FusedL2TopKResult {
        let device = context.device.rawDevice
        let actualK = min(k, FusedL2TopKParameters.maxK)

        // Process first chunk to initialize results
        let firstChunkSize = min(chunkSize, datasetCount)
        let firstParams = FusedL2TopKParameters(
            numQueries: queryCount,
            numDataset: firstChunkSize,
            dimension: dimension,
            k: actualK
        )

        let result = try await execute(
            queries: queries,
            dataset: dataset,
            parameters: firstParams
        )

        // Capture buffers for use in closures
        let resultIndices = result.indices
        guard let resultDistances = result.distances else {
            throw VectorError.invalidOperation("Chunked processing requires distances buffer")
        }

        // Process remaining chunks
        var processedCount = firstChunkSize
        while processedCount < datasetCount {
            let remainingCount = datasetCount - processedCount
            let currentChunkSize = min(chunkSize, remainingCount)

            // Create view into dataset at current offset
            let offset = processedCount * dimension * MemoryLayout<Float>.size
            let chunkLength = currentChunkSize * dimension * MemoryLayout<Float>.size
            let chunkPointer = dataset.contents().advanced(by: offset)

            guard let chunkBuffer = device.makeBuffer(
                bytes: chunkPointer,
                length: chunkLength,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: chunkLength)
            }

            let streamingParams = Metal4StreamingL2Params(
                Q: UInt32(queryCount),
                chunkSize: UInt32(currentChunkSize),
                D: UInt32(dimension),
                K: UInt32(actualK),
                offset: UInt32(processedCount)
            )

            try await context.executeAndWait { [self] _, encoder in
                self.encodeStreamingUpdate(
                    into: encoder,
                    queries: queries,
                    chunk: chunkBuffer,
                    runningIndices: resultIndices,
                    runningDistances: resultDistances,
                    parameters: streamingParams
                )
            }

            processedCount += currentChunkSize
        }

        return result
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
}
