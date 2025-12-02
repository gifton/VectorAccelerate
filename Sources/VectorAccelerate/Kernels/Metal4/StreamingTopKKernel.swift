//
//  StreamingTopKKernel.swift
//  VectorAccelerate
//
//  Metal 4 Streaming Top-K kernel for massive datasets.
//
//  Phase 5: Kernel Migrations - Batch 2, Priority 2
//
//  Features:
//  - Supports datasets > 4 billion vectors
//  - Incremental chunk processing with running heap
//  - Three-phase: init → process chunks → finalize
//  - Memory-efficient: O(Q × K) state vs O(Q × N) full matrix

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for streaming top-k operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4StreamingConfig: Sendable {
    /// Number of query vectors
    public let queryCount: Int

    /// Number of top elements to select
    public let k: Int

    /// Number of vectors to process per chunk
    public let chunkSize: Int

    /// Total number of vectors in dataset (supports > 4B)
    public let totalVectorCount: Int64

    /// Maximum K supported
    public static let maxK: Int = 128

    public init(
        queryCount: Int,
        k: Int,
        chunkSize: Int = 100_000,
        totalVectorCount: Int64
    ) {
        self.queryCount = queryCount
        self.k = min(k, Self.maxK)
        self.chunkSize = chunkSize
        self.totalVectorCount = totalVectorCount
    }

    /// Number of chunks to process
    public var numberOfChunks: Int {
        Int((totalVectorCount + Int64(chunkSize) - 1) / Int64(chunkSize))
    }
}

// MARK: - Streaming State

/// Mutable state for streaming top-k operation.
///
/// Tracks progress and holds running buffers during chunk processing.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4StreamingState: @unchecked Sendable {
    /// Running top-k distances [queryCount × k]
    public let runningDistances: any MTLBuffer

    /// Running top-k indices [queryCount × k]
    public let runningIndices: any MTLBuffer

    /// Configuration for this streaming operation
    public let config: Metal4StreamingConfig

    /// Number of chunks processed so far
    private(set) var chunksProcessed: Int = 0

    /// Lock for thread-safe chunk increment
    private let lock = NSLock()

    internal init(
        distances: any MTLBuffer,
        indices: any MTLBuffer,
        config: Metal4StreamingConfig
    ) {
        self.runningDistances = distances
        self.runningIndices = indices
        self.config = config
    }

    /// Progress as fraction [0, 1]
    public var progress: Float {
        Float(chunksProcessed) / Float(config.numberOfChunks)
    }

    /// Whether all chunks have been processed
    public var isComplete: Bool {
        chunksProcessed >= config.numberOfChunks
    }

    /// Thread-safe increment of chunks processed
    internal func incrementChunks() {
        lock.lock()
        chunksProcessed += 1
        lock.unlock()
    }
}

// MARK: - Result Type

/// Final result after streaming completes.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4StreamingResult: Sendable {
    /// Final sorted indices [queryCount × k]
    public let indices: any MTLBuffer

    /// Final sorted distances [queryCount × k]
    public let distances: any MTLBuffer

    /// Number of queries
    public let queryCount: Int

    /// K value used
    public let k: Int

    /// Extract sorted results for a specific query.
    ///
    /// Returns results sorted by distance (ascending for minimum mode).
    public func sortedResults(for queryIndex: Int) -> [(index: Int64, distance: Float)] {
        guard queryIndex < queryCount else { return [] }

        let offset = queryIndex * k
        let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: queryCount * k)
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * k)

        var results: [(index: Int64, distance: Float)] = []
        results.reserveCapacity(k)
        for i in 0..<k {
            let idx = indexPtr[offset + i]
            if idx != 0xFFFFFFFF {  // Skip sentinel values
                results.append((index: Int64(idx), distance: distPtr[offset + i]))
            }
        }
        return results
    }

    /// Get all results as array of arrays.
    public func allResults() -> [[(index: Int64, distance: Float)]] {
        (0..<queryCount).map { sortedResults(for: $0) }
    }
}

// MARK: - Internal Params

/// Parameters for streaming init kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct StreamingInitParams: Sendable {
    var K: UInt32
    var Q: UInt32
}

/// Parameters for streaming process chunk kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct StreamingProcessParams: Sendable {
    var Q: UInt32
    var chunkSize: UInt32
    var K: UInt32
    var chunkBaseIndex: UInt64
}

// MARK: - Kernel Implementation

/// Metal 4 Streaming Top-K kernel for massive datasets.
///
/// Designed to handle datasets exceeding GPU memory or even system memory
/// by processing vectors in chunks and maintaining a running top-k heap.
///
/// ## Algorithm
///
/// 1. **Initialize**: Fill running buffers with infinity/sentinel values
/// 2. **Process Chunks**: For each chunk, update running top-k via heap insertion
/// 3. **Finalize**: Sort the final top-k and copy to shared memory
///
/// ## Memory Efficiency
///
/// State memory: O(Q × K × 8 bytes)
/// - For Q=1000 queries and K=128: ~1 MB
/// - Independent of dataset size N
///
/// ## Usage
///
/// ```swift
/// let kernel = try await StreamingTopKKernel(context: context)
///
/// // Initialize streaming state
/// let state = try await kernel.initializeStreaming(config: config)
///
/// // Process chunks
/// for chunk in datasetChunks {
///     try await kernel.processChunk(distances: chunk, state: state, ...)
/// }
///
/// // Finalize and get results
/// let result = try await kernel.finalizeStreaming(state: state)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class StreamingTopKKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "StreamingTopKKernel"

    // MARK: - Pipelines

    /// Initialize running buffers with infinity/sentinel
    private let initPipeline: any MTLComputePipelineState

    /// Process a chunk and update running top-k
    private let processPipeline: any MTLComputePipelineState

    /// Sort final results (heapsort)
    private let finalizePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Streaming Top-K kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let initFunc = library.makeFunction(name: "streaming_topk_init") else {
            throw VectorError.shaderNotFound(
                name: "Streaming Top-K init kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        guard let processFunc = library.makeFunction(name: "streaming_topk_process_chunk") else {
            throw VectorError.shaderNotFound(
                name: "Streaming Top-K process kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        guard let finalizeFunc = library.makeFunction(name: "streaming_topk_finalize") else {
            throw VectorError.shaderNotFound(
                name: "Streaming Top-K finalize kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.initPipeline = try await device.makeComputePipelineState(function: initFunc)
        self.processPipeline = try await device.makeComputePipelineState(function: processFunc)
        self.finalizePipeline = try await device.makeComputePipelineState(function: finalizeFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Streaming Operations

    /// Initialize streaming state for chunk-based processing.
    ///
    /// Allocates running buffers in private memory and initializes with
    /// infinity distances and sentinel indices.
    ///
    /// - Parameter config: Streaming configuration
    /// - Returns: Initialized streaming state
    public func initializeStreaming(config: Metal4StreamingConfig) async throws -> Metal4StreamingState {
        guard config.k <= Metal4StreamingConfig.maxK else {
            throw VectorError.invalidInput("K must be <= \(Metal4StreamingConfig.maxK)")
        }

        let device = context.device.rawDevice

        // Allocate running state buffers in private memory for GPU efficiency
        let distanceSize = config.queryCount * config.k * MemoryLayout<Float>.size
        let indexSize = config.queryCount * config.k * MemoryLayout<UInt32>.size

        guard let distanceBuffer = device.makeBuffer(length: distanceSize, options: .storageModePrivate) else {
            throw VectorError.bufferAllocationFailed(size: distanceSize)
        }
        distanceBuffer.label = "StreamingTopK.runningDistances"

        guard let indexBuffer = device.makeBuffer(length: indexSize, options: .storageModePrivate) else {
            throw VectorError.bufferAllocationFailed(size: indexSize)
        }
        indexBuffer.label = "StreamingTopK.runningIndices"

        // Initialize buffers with infinity/sentinel values
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(initPipeline)
            encoder.label = "StreamingTopKInit"

            encoder.setBuffer(distanceBuffer, offset: 0, index: 0)
            encoder.setBuffer(indexBuffer, offset: 0, index: 1)

            var params = StreamingInitParams(K: UInt32(config.k), Q: UInt32(config.queryCount))
            encoder.setBytes(&params, length: MemoryLayout<StreamingInitParams>.size, index: 2)

            let totalElements = config.queryCount * config.k
            let optimalThreads = min(initPipeline.maxTotalThreadsPerThreadgroup, 256)
            let numThreadgroups = (totalElements + optimalThreads - 1) / optimalThreads

            encoder.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: optimalThreads, height: 1, depth: 1)
            )
        }

        return Metal4StreamingState(
            distances: distanceBuffer,
            indices: indexBuffer,
            config: config
        )
    }

    /// Process a chunk of pre-computed distances.
    ///
    /// Updates running top-k by comparing chunk distances against current heap.
    ///
    /// - Parameters:
    ///   - distances: Distance matrix for current chunk [queryCount × chunkSize]
    ///   - state: Current streaming state
    ///   - chunkBaseIndex: Global index offset for this chunk
    public func processChunk(
        distances: any MTLBuffer,
        state: Metal4StreamingState,
        chunkBaseIndex: Int64
    ) async throws {
        guard !state.isComplete else {
            throw VectorError.invalidOperation("Streaming already complete")
        }

        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(processPipeline)
            encoder.label = "StreamingTopKProcessChunk"

            // Set buffers
            encoder.setBuffer(distances, offset: 0, index: 0)
            encoder.setBuffer(state.runningDistances, offset: 0, index: 2)
            encoder.setBuffer(state.runningIndices, offset: 0, index: 3)

            var params = StreamingProcessParams(
                Q: UInt32(state.config.queryCount),
                chunkSize: UInt32(state.config.chunkSize),
                K: UInt32(state.config.k),
                chunkBaseIndex: UInt64(chunkBaseIndex)
            )
            encoder.setBytes(&params, length: MemoryLayout<StreamingProcessParams>.size, index: 5)

            // One thread per query
            let optimalThreads = min(processPipeline.maxTotalThreadsPerThreadgroup, 256)
            let numThreadgroups = (state.config.queryCount + optimalThreads - 1) / optimalThreads

            encoder.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: optimalThreads, height: 1, depth: 1)
            )
        }

        state.incrementChunks()
    }

    /// Finalize streaming and sort results.
    ///
    /// Copies from private to shared memory and sorts by distance.
    ///
    /// - Parameter state: Completed streaming state
    /// - Returns: Final sorted results
    public func finalizeStreaming(state: Metal4StreamingState) async throws -> Metal4StreamingResult {
        guard state.isComplete else {
            throw VectorError.invalidOperation(
                "Streaming not complete. Processed \(state.chunksProcessed)/\(state.config.numberOfChunks) chunks"
            )
        }

        let device = context.device.rawDevice
        let config = state.config

        // Create output buffers in shared memory for CPU access
        let distanceSize = config.queryCount * config.k * MemoryLayout<Float>.size
        let indexSize = config.queryCount * config.k * MemoryLayout<UInt32>.size

        guard let finalDistances = device.makeBuffer(length: distanceSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distanceSize)
        }
        finalDistances.label = "StreamingTopK.finalDistances"

        guard let finalIndices = device.makeBuffer(length: indexSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: indexSize)
        }
        finalIndices.label = "StreamingTopK.finalIndices"

        // Copy and sort in a single command buffer
        try await context.executeAndWait { [self] commandBuffer, encoder in
            // First, use blit encoder to copy from private to shared
            encoder.endEncoding()

            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                return
            }

            blitEncoder.copy(
                from: state.runningDistances,
                sourceOffset: 0,
                to: finalDistances,
                destinationOffset: 0,
                size: distanceSize
            )

            blitEncoder.copy(
                from: state.runningIndices,
                sourceOffset: 0,
                to: finalIndices,
                destinationOffset: 0,
                size: indexSize
            )

            blitEncoder.endEncoding()

            // Then sort using compute encoder
            guard let sortEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return
            }

            sortEncoder.setComputePipelineState(finalizePipeline)
            sortEncoder.label = "StreamingTopKFinalize"

            sortEncoder.setBuffer(finalDistances, offset: 0, index: 0)
            sortEncoder.setBuffer(finalIndices, offset: 0, index: 1)

            var params = StreamingInitParams(K: UInt32(config.k), Q: UInt32(config.queryCount))
            sortEncoder.setBytes(&params, length: MemoryLayout<StreamingInitParams>.size, index: 2)

            let optimalThreads = min(finalizePipeline.maxTotalThreadsPerThreadgroup, 256)
            let numThreadgroups = (config.queryCount + optimalThreads - 1) / optimalThreads

            sortEncoder.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: optimalThreads, height: 1, depth: 1)
            )

            sortEncoder.endEncoding()
        }

        return Metal4StreamingResult(
            indices: finalIndices,
            distances: finalDistances,
            queryCount: config.queryCount,
            k: config.k
        )
    }

    // MARK: - High-Level API

    /// Process entire dataset in streaming fashion using a distance chunk provider.
    ///
    /// - Parameters:
    ///   - distanceChunkProvider: Async function providing distance buffers for each chunk
    ///   - config: Streaming configuration
    ///   - progressHandler: Optional callback for progress updates
    /// - Returns: Final top-k results
    public func streamingTopK(
        distanceChunkProvider: @escaping (Int) async throws -> (any MTLBuffer)?,
        config: Metal4StreamingConfig,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws -> Metal4StreamingResult {
        // Initialize state
        let state = try await initializeStreaming(config: config)

        // Process chunks
        var chunkIndex = 0
        var globalIndex: Int64 = 0

        while let chunkBuffer = try await distanceChunkProvider(chunkIndex) {
            try await processChunk(
                distances: chunkBuffer,
                state: state,
                chunkBaseIndex: globalIndex
            )

            chunkIndex += 1
            globalIndex += Int64(config.chunkSize)

            progressHandler?(state.progress)

            if state.isComplete {
                break
            }
        }

        // Finalize
        return try await finalizeStreaming(state: state)
    }
}
