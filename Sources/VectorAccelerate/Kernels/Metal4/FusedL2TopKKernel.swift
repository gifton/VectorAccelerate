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

    /// Maximum dimension supported by the kernel (supports BERT-768)
    public static let maxDimension: Int = 768

    /// Create parameters for fused L2 + Top-K.
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors (must be > 0)
    ///   - numDataset: Number of dataset vectors (must be > 0)
    ///   - dimension: Vector dimension (must be > 0 and <= maxDimension)
    ///   - k: Number of nearest neighbors to find (must be > 0 and <= maxK)
    /// - Throws: `IndexError.invalidInput` if any parameter is invalid
    public init(
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int
    ) throws {
        guard numQueries > 0 else {
            throw IndexError.invalidInput(message: "numQueries must be positive, got \(numQueries)")
        }
        guard numDataset > 0 else {
            throw IndexError.invalidInput(message: "numDataset must be positive, got \(numDataset)")
        }
        guard dimension > 0 else {
            throw IndexError.invalidInput(message: "dimension must be positive, got \(dimension)")
        }
        guard dimension <= Self.maxDimension else {
            throw IndexError.invalidInput(
                message: "dimension \(dimension) exceeds maximum supported dimension \(Self.maxDimension)"
            )
        }
        guard k > 0 else {
            throw IndexError.invalidInput(message: "k must be positive, got \(k)")
        }
        guard k <= Self.maxK else {
            throw IndexError.invalidInput(
                message: "k \(k) exceeds maximum supported k \(Self.maxK)"
            )
        }

        self.numQueries = UInt32(numQueries)
        self.numDataset = UInt32(numDataset)
        self.dimension = UInt32(dimension)
        self.k = UInt32(k)
    }
}

// MARK: - Configuration

/// Configuration for fused operation.
public struct Metal4FusedL2Config: Sendable {
    /// Whether to include distance values in output
    public let includeDistances: Bool

    /// Threadgroup size for kernel dispatch
    public let threadgroupSize: Int

    /// Maximum number of bytes allowed for a materialized distance matrix.
    ///
    /// This limit is only relevant for the **two-pass fallback** strategies
    /// (L2Distance -> TopKSelection). The fused kernel (K ≤ 8) does not
    /// materialize a distance matrix.
    public let maxDistanceMatrixBytes: Int

    /// Enable chunked processing when the full distance matrix would exceed
    /// `maxDistanceMatrixBytes`.
    public let enableChunkedFallback: Bool

    /// Prefer GPU-side merge for the chunked fallback path.
    ///
    /// When enabled (default), `FusedL2TopKKernel` will maintain a running top-k
    /// on the GPU during chunked processing and merge each chunk's top-k results
    /// with a dedicated merge kernel. This avoids repeated CPU readback and
    /// sorting work for large datasets.
    ///
    /// If the merge shader is unavailable (e.g. older precompiled metallib),
    /// the kernel will automatically fall back to the CPU merge implementation.
    public let preferGPUMergeInChunkedFallback: Bool

    public init(
        includeDistances: Bool = true,
        threadgroupSize: Int = 256,
        maxDistanceMatrixBytes: Int = 64 * 1024 * 1024,
        enableChunkedFallback: Bool = true,
        preferGPUMergeInChunkedFallback: Bool = true
    ) {
        self.includeDistances = includeDistances
        self.threadgroupSize = min(threadgroupSize, 256)
        self.maxDistanceMatrixBytes = maxDistanceMatrixBytes
        self.enableChunkedFallback = enableChunkedFallback
        self.preferGPUMergeInChunkedFallback = preferGPUMergeInChunkedFallback
    }

    /// Default configuration with distances included
    public static let `default` = Metal4FusedL2Config()
}

// MARK: - Result Type

/// Result from fused L2 + Top-K operation.
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
///
/// - Warning: **EXPERIMENTAL** - The underlying `streaming_l2_topk_update` shader
///   has known correctness issues. Use the chunked two-pass fallback instead.
@available(*, deprecated, message: "Experimental: streaming_l2_topk_update has correctness issues. Use execute() which automatically uses chunked fallback.")
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
public final class FusedL2TopKKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "FusedL2TopKKernel"

    // MARK: - Pipelines

    /// Main fused L2 + Top-K pipeline
    private let fusedPipeline: any MTLComputePipelineState

    /// Streaming update pipeline for chunked processing
    private let streamingUpdatePipeline: any MTLComputePipelineState

    // MARK: - Fallback Kernels

    /// Two-pass fallback (distance matrix materialization)
    private let l2DistanceKernel: L2DistanceKernel
    private let topKSelectionKernel: TopKSelectionKernel
    private let warpSelectionKernel: WarpOptimizedSelectionKernel

    /// GPU merge kernel for chunked fallback (optional).
    ///
    /// If unavailable (e.g. older metallib), chunked fallback will use CPU merge.
    private let topKMergeKernel: TopKMergeKernel?

    // MARK: - Strategy Constants

    /// Hard limit of the current fused shader implementation (K_PRIVATE)
    private static let fusedMaxK: Int = 8

    /// Selection threshold for warp-optimized top-k
    private static let warpOptimizedMaxK: Int = 32

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

        // Fallback kernels (used automatically for K > 8)
        self.l2DistanceKernel = try await L2DistanceKernel(context: context)
        self.topKSelectionKernel = try await TopKSelectionKernel(context: context)
        self.warpSelectionKernel = try await WarpOptimizedSelectionKernel(context: context)

        // Optional GPU merge kernel for chunked fallback (K > 8).
        // If unavailable (e.g. older precompiled metallib), we fall back to CPU merge.
        self.topKMergeKernel = try? await TopKMergeKernel(context: context)
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
        assert(
            Int(parameters.k) <= Self.fusedMaxK,
            "fused_l2_topk shader is only guaranteed correct for K <= \(Self.fusedMaxK). Use execute() for K > \(Self.fusedMaxK) (two-pass fallback)."
        )

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
    /// - Warning: **EXPERIMENTAL** - The underlying `streaming_l2_topk_update` shader
    ///   has known correctness issues where only thread 0's results are preserved.
    ///   Use `execute()` instead, which automatically uses the correct chunked fallback.
    ///
    /// Used when processing dataset in chunks to update running top-k.
    @available(*, deprecated, message: "Experimental: has correctness issues. Use execute() which automatically uses chunked fallback.")
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
    /// Automatically selects the optimal strategy based on K:
    /// - K ≤ 8: Fused single-pass kernel (fastest, no distance matrix)
    /// - 8 < K ≤ 32: Two-pass with warp-optimized selection
    /// - K > 32: Two-pass with standard selection
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
        let numQueries = Int(parameters.numQueries)
        let numDataset = Int(parameters.numDataset)
        let dimension = Int(parameters.dimension)
        let k = Int(parameters.k)

        // Strategy selection
        if k <= Self.fusedMaxK {
            // Fused shader path (fastest, no distance matrix)
            return try await executeFusedDirect(
                queries: queries,
                dataset: dataset,
                parameters: parameters,
                config: config
            )
        }

        // Two-pass fallback paths
        let distanceMatrixBytes = numQueries * numDataset * MemoryLayout<Float>.size
        let shouldChunk = config.enableChunkedFallback && distanceMatrixBytes > config.maxDistanceMatrixBytes

        if shouldChunk {
            return try await executeChunkedTwoPass(
                queries: queries,
                dataset: dataset,
                numQueries: numQueries,
                numDataset: numDataset,
                dimension: dimension,
                k: k,
                maxDistanceMatrixBytes: config.maxDistanceMatrixBytes,
                config: config
            )
        }

        if k <= Self.warpOptimizedMaxK {
            return try await executeTwoPassWarp(
                queries: queries,
                dataset: dataset,
                numQueries: numQueries,
                numDataset: numDataset,
                dimension: dimension,
                k: k,
                config: config
            )
        }

        return try await executeTwoPassStandard(
            queries: queries,
            dataset: dataset,
            numQueries: numQueries,
            numDataset: numDataset,
            dimension: dimension,
            k: k,
            config: config
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

        let parameters = try FusedL2TopKParameters(
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

        let parameters = try FusedL2TopKParameters(
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
        // NOTE:
        // The original implementation used the `streaming_l2_topk_update` shader.
        // That shader is intentionally marked as experimental in the roadmap.
        //
        // This method now delegates to the robust chunked two-pass fallback:
        //   L2Distance (per chunk) -> Selection (per chunk) -> CPU merge.
        let params = try FusedL2TopKParameters(
            numQueries: queryCount,
            numDataset: datasetCount,
            dimension: dimension,
            k: k
        )

        // Convert chunkSize into a byte limit for the distance matrix:
        // distanceBytesPerChunk = Q * chunkSize * sizeof(Float)
        let maxBytes = max(1, queryCount * chunkSize * MemoryLayout<Float>.size)

        return try await execute(
            queries: queries,
            dataset: dataset,
            parameters: params,
            config: Metal4FusedL2Config(
                includeDistances: true,
                threadgroupSize: 256,
                maxDistanceMatrixBytes: maxBytes,
                enableChunkedFallback: true
            )
        )
    }

    // MARK: - Strategy Implementations

    private func executeFusedDirect(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        parameters: FusedL2TopKParameters,
        config: Metal4FusedL2Config
    ) async throws -> Metal4FusedL2TopKResult {
        let device = context.device.rawDevice
        let numQueries = Int(parameters.numQueries)
        let k = Int(parameters.k)

        // Allocate outputs
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

    private func executeTwoPassWarp(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int,
        config: Metal4FusedL2Config
    ) async throws -> Metal4FusedL2TopKResult {
        try await executeTwoPass(
            queries: queries,
            dataset: dataset,
            numQueries: numQueries,
            numDataset: numDataset,
            dimension: dimension,
            k: k,
            selection: .warp,
            config: config
        )
    }

    private func executeTwoPassStandard(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int,
        config: Metal4FusedL2Config
    ) async throws -> Metal4FusedL2TopKResult {
        try await executeTwoPass(
            queries: queries,
            dataset: dataset,
            numQueries: numQueries,
            numDataset: numDataset,
            dimension: dimension,
            k: k,
            selection: .standard,
            config: config
        )
    }

    private enum TwoPassSelectionKind {
        case warp
        case standard
    }

    private func executeTwoPass(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int,
        selection: TwoPassSelectionKind,
        config: Metal4FusedL2Config
    ) async throws -> Metal4FusedL2TopKResult {
        let device = context.device.rawDevice
        let actualK = min(k, FusedL2TopKParameters.maxK)

        // Allocate full distance matrix (guarded by strategy selector)
        let distancesSize = numQueries * numDataset * MemoryLayout<Float>.size
        guard let distanceMatrix = device.makeBuffer(length: distancesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distancesSize)
        }
        distanceMatrix.label = "FusedL2TopK.fallback.distances"

        // Allocate top-k outputs
        let outIndicesSize = numQueries * actualK * MemoryLayout<UInt32>.size
        let outValuesSize = numQueries * actualK * MemoryLayout<Float>.size
        guard let outIndices = device.makeBuffer(length: outIndicesSize, options: .storageModeShared),
              let outValues = device.makeBuffer(length: outValuesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outIndicesSize + outValuesSize)
        }
        outIndices.label = "FusedL2TopK.fallback.indices"
        outValues.label = "FusedL2TopK.fallback.values"

        let l2Params = L2DistanceParameters(
            numQueries: numQueries,
            numDatabase: numDataset,
            dimension: dimension,
            computeSqrt: false // match fused kernel: squared L2
        )

        try await context.executeAndWait { [self] _, encoder in
            // 1) Distance matrix
            self.l2DistanceKernel.encode(
                into: encoder,
                queries: queries,
                database: dataset,
                distances: distanceMatrix,
                parameters: l2Params
            )

            // 2) Barrier before selection
            encoder.memoryBarrier(scope: .buffers)

            // 3) Select top-k per query
            switch selection {
            case .warp:
                self.warpSelectionKernel.encode(
                    into: encoder,
                    distances: distanceMatrix,
                    outputIndices: outIndices,
                    outputValues: outValues,
                    queryCount: numQueries,
                    candidateCount: numDataset,
                    k: actualK,
                    mode: .ascending
                )
            case .standard:
                let topKParams = TopKParameters(
                    batchSize: numQueries,
                    numElements: numDataset,
                    k: actualK,
                    mode: .minimum,
                    sorted: true
                )
                self.topKSelectionKernel.encode(
                    into: encoder,
                    input: distanceMatrix,
                    outputValues: outValues,
                    outputIndices: outIndices,
                    parameters: topKParams
                )
            }
        }

        return Metal4FusedL2TopKResult(
            indices: outIndices,
            distances: config.includeDistances ? outValues : nil,
            numQueries: numQueries,
            k: actualK
        )
    }

    private struct TopKPair {
        var index: UInt32
        var distance: Float
    }

    private func executeChunkedTwoPass(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        numQueries: Int,
        numDataset: Int,
        dimension: Int,
        k: Int,
        maxDistanceMatrixBytes: Int,
        config: Metal4FusedL2Config
    ) async throws -> Metal4FusedL2TopKResult {
        let device = context.device.rawDevice
        let actualK = min(k, FusedL2TopKParameters.maxK)

        // Compute chunk size from guardrail: Q * chunkSize * sizeof(Float) <= maxDistanceMatrixBytes
        let bytesPerCandidate = max(1, numQueries * MemoryLayout<Float>.size)
        let maxCandidatesPerChunk = max(1, maxDistanceMatrixBytes / bytesPerCandidate)
        let chunkSize = min(numDataset, maxCandidatesPerChunk)

        let selectionKind: TwoPassSelectionKind = (actualK <= Self.warpOptimizedMaxK) ? .warp : .standard
        let sentinel: UInt32 = 0xFFFF_FFFF

        // =====================================================================
        // GPU Merge Path: Maintain running top-k on GPU, merge without CPU readback
        // =====================================================================
        if config.preferGPUMergeInChunkedFallback, let mergeKernel = self.topKMergeKernel {
            let runningIdxBytes = numQueries * actualK * MemoryLayout<UInt32>.size
            let runningDistBytes = numQueries * actualK * MemoryLayout<Float>.size

            guard let indicesA = device.makeBuffer(length: runningIdxBytes, options: .storageModeShared),
                  let distancesA = device.makeBuffer(length: runningDistBytes, options: .storageModeShared),
                  let indicesB = device.makeBuffer(length: runningIdxBytes, options: .storageModeShared),
                  let distancesB = device.makeBuffer(length: runningDistBytes, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: runningIdxBytes + runningDistBytes)
            }
            indicesA.label = "FusedL2TopK.chunked.running.indices[A]"
            distancesA.label = "FusedL2TopK.chunked.running.distances[A]"
            indicesB.label = "FusedL2TopK.chunked.running.indices[B]"
            distancesB.label = "FusedL2TopK.chunked.running.distances[B]"

            // Initialize running buffers to sentinel/+infinity so the first merge works
            let initCount = numQueries * actualK
            let initIdxA = indicesA.contents().bindMemory(to: UInt32.self, capacity: initCount)
            let initDistA = distancesA.contents().bindMemory(to: Float.self, capacity: initCount)
            let initIdxB = indicesB.contents().bindMemory(to: UInt32.self, capacity: initCount)
            let initDistB = distancesB.contents().bindMemory(to: Float.self, capacity: initCount)
            for i in 0..<initCount {
                initIdxA[i] = sentinel
                initDistA[i] = .infinity
                initIdxB[i] = sentinel
                initDistB[i] = .infinity
            }

            var runningIndices: any MTLBuffer = indicesA
            var runningDistances: any MTLBuffer = distancesA
            var scratchIndices: any MTLBuffer = indicesB
            var scratchDistances: any MTLBuffer = distancesB

            var chunkStart = 0
            while chunkStart < numDataset {
                let thisChunkSize = min(chunkSize, numDataset - chunkStart)
                let chunkK = min(actualK, thisChunkSize)

                // Per-chunk distance matrix
                let distanceBytes = numQueries * thisChunkSize * MemoryLayout<Float>.size
                guard let distanceMatrix = device.makeBuffer(length: distanceBytes, options: .storageModeShared) else {
                    throw VectorError.bufferAllocationFailed(size: distanceBytes)
                }
                distanceMatrix.label = "FusedL2TopK.chunked.distances[\(chunkStart)..<\(chunkStart + thisChunkSize)]"

                // Per-chunk top-k outputs (chunk-local indices)
                let outIndicesBytes = numQueries * chunkK * MemoryLayout<UInt32>.size
                let outValuesBytes = numQueries * chunkK * MemoryLayout<Float>.size
                guard let chunkIndices = device.makeBuffer(length: outIndicesBytes, options: .storageModeShared),
                      let chunkValues = device.makeBuffer(length: outValuesBytes, options: .storageModeShared) else {
                    throw VectorError.bufferAllocationFailed(size: outIndicesBytes + outValuesBytes)
                }
                chunkIndices.label = "FusedL2TopK.chunked.chunkIndices"
                chunkValues.label = "FusedL2TopK.chunked.chunkValues"

                let l2Params = L2DistanceParameters(
                    numQueries: numQueries,
                    numDatabase: thisChunkSize,
                    dimension: dimension,
                    computeSqrt: false
                )

                let databaseOffsetBytes = chunkStart * dimension * MemoryLayout<Float>.size
                let mergeParams = TopKMergeParameters(
                    numQueries: numQueries,
                    k: actualK,
                    chunkK: chunkK,
                    chunkBase: chunkStart
                )

                // Capture current buffer references for the closure (they'll be swapped after)
                let currentRunningIndices = runningIndices
                let currentRunningDistances = runningDistances
                let currentScratchIndices = scratchIndices
                let currentScratchDistances = scratchDistances

                try await context.executeAndWait { [self] _, encoder in
                    // 1) Distances for this chunk (database offset avoids copy)
                    self.l2DistanceKernel.encode(
                        into: encoder,
                        queries: queries,
                        queryOffset: 0,
                        database: dataset,
                        databaseOffset: databaseOffsetBytes,
                        distances: distanceMatrix,
                        distancesOffset: 0,
                        parameters: l2Params
                    )

                    encoder.memoryBarrier(scope: .buffers)

                    // 2) Top-k within this chunk
                    switch selectionKind {
                    case .warp:
                        self.warpSelectionKernel.encode(
                            into: encoder,
                            distances: distanceMatrix,
                            outputIndices: chunkIndices,
                            outputValues: chunkValues,
                            queryCount: numQueries,
                            candidateCount: thisChunkSize,
                            k: chunkK,
                            mode: .ascending
                        )
                    case .standard:
                        let topKParams = TopKParameters(
                            batchSize: numQueries,
                            numElements: thisChunkSize,
                            k: chunkK,
                            mode: .minimum,
                            sorted: true
                        )
                        self.topKSelectionKernel.encode(
                            into: encoder,
                            input: distanceMatrix,
                            outputValues: chunkValues,
                            outputIndices: chunkIndices,
                            parameters: topKParams
                        )
                    }

                    encoder.memoryBarrier(scope: .buffers)

                    // 3) Merge running top-k with this chunk's top-k (GPU)
                    mergeKernel.encode(
                        into: encoder,
                        runningIndices: currentRunningIndices,
                        runningDistances: currentRunningDistances,
                        chunkIndices: chunkIndices,
                        chunkDistances: chunkValues,
                        outputIndices: currentScratchIndices,
                        outputDistances: currentScratchDistances,
                        parameters: mergeParams
                    )
                }

                // Swap running and scratch for the next iteration
                swap(&runningIndices, &scratchIndices)
                swap(&runningDistances, &scratchDistances)

                chunkStart += thisChunkSize
            }

            return Metal4FusedL2TopKResult(
                indices: runningIndices,
                distances: config.includeDistances ? runningDistances : nil,
                numQueries: numQueries,
                k: actualK
            )
        }

        // =====================================================================
        // CPU Merge Path: Fallback when GPU merge is unavailable or disabled
        // =====================================================================

        // CPU-side running top-k per query (k ≤ 128)
        var best: [[TopKPair]] = Array(repeating: [], count: numQueries)
        for q in 0..<numQueries {
            best[q].reserveCapacity(actualK)
        }

        var chunkStart = 0
        while chunkStart < numDataset {
            let thisChunkSize = min(chunkSize, numDataset - chunkStart)
            let chunkK = min(actualK, thisChunkSize)

            // Per-chunk distance matrix
            let distanceBytes = numQueries * thisChunkSize * MemoryLayout<Float>.size
            guard let distanceMatrix = device.makeBuffer(length: distanceBytes, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: distanceBytes)
            }
            distanceMatrix.label = "FusedL2TopK.chunked.distances[\(chunkStart)..<\(chunkStart + thisChunkSize)]"

            // Per-chunk top-k outputs
            let outIndicesBytes = numQueries * chunkK * MemoryLayout<UInt32>.size
            let outValuesBytes = numQueries * chunkK * MemoryLayout<Float>.size
            guard let outIndices = device.makeBuffer(length: outIndicesBytes, options: .storageModeShared),
                  let outValues = device.makeBuffer(length: outValuesBytes, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: outIndicesBytes + outValuesBytes)
            }
            outIndices.label = "FusedL2TopK.chunked.indices"
            outValues.label = "FusedL2TopK.chunked.values"

            let l2Params = L2DistanceParameters(
                numQueries: numQueries,
                numDatabase: thisChunkSize,
                dimension: dimension,
                computeSqrt: false
            )

            let databaseOffsetBytes = chunkStart * dimension * MemoryLayout<Float>.size

            try await context.executeAndWait { [self] _, encoder in
                // 1) Distance matrix for this chunk (database offset avoids copy)
                self.l2DistanceKernel.encode(
                    into: encoder,
                    queries: queries,
                    queryOffset: 0,
                    database: dataset,
                    databaseOffset: databaseOffsetBytes,
                    distances: distanceMatrix,
                    distancesOffset: 0,
                    parameters: l2Params
                )

                encoder.memoryBarrier(scope: .buffers)

                // 2) Top-k within this chunk
                switch selectionKind {
                case .warp:
                    self.warpSelectionKernel.encode(
                        into: encoder,
                        distances: distanceMatrix,
                        outputIndices: outIndices,
                        outputValues: outValues,
                        queryCount: numQueries,
                        candidateCount: thisChunkSize,
                        k: chunkK,
                        mode: .ascending
                    )
                case .standard:
                    let topKParams = TopKParameters(
                        batchSize: numQueries,
                        numElements: thisChunkSize,
                        k: chunkK,
                        mode: .minimum,
                        sorted: true
                    )
                    self.topKSelectionKernel.encode(
                        into: encoder,
                        input: distanceMatrix,
                        outputValues: outValues,
                        outputIndices: outIndices,
                        parameters: topKParams
                    )
                }
            }

            // CPU merge: best-so-far (≤ k) with chunk top-k (≤ k)
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * chunkK)
            let valPtr = outValues.contents().bindMemory(to: Float.self, capacity: numQueries * chunkK)
            let chunkBase = UInt32(chunkStart)

            for q in 0..<numQueries {
                var merged = best[q]
                merged.reserveCapacity(merged.count + chunkK)

                let rowOffset = q * chunkK
                for i in 0..<chunkK {
                    let localIdx = idxPtr[rowOffset + i]
                    if localIdx == sentinel { continue }
                    let globalIdx = localIdx &+ chunkBase
                    merged.append(TopKPair(index: globalIdx, distance: valPtr[rowOffset + i]))
                }

                // Sort and truncate to k
                merged.sort {
                    if $0.distance == $1.distance { return $0.index < $1.index }
                    return $0.distance < $1.distance
                }
                if merged.count > actualK {
                    merged.removeLast(merged.count - actualK)
                }
                best[q] = merged
            }

            chunkStart += thisChunkSize
        }

        // Materialize final outputs
        let finalIndicesBytes = numQueries * actualK * MemoryLayout<UInt32>.size
        guard let finalIndices = device.makeBuffer(length: finalIndicesBytes, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: finalIndicesBytes)
        }
        finalIndices.label = "FusedL2TopK.chunked.final.indices"

        let finalDistances: (any MTLBuffer)?
        if config.includeDistances {
            let finalDistancesBytes = numQueries * actualK * MemoryLayout<Float>.size
            guard let buf = device.makeBuffer(length: finalDistancesBytes, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: finalDistancesBytes)
            }
            buf.label = "FusedL2TopK.chunked.final.distances"
            finalDistances = buf
        } else {
            finalDistances = nil
        }

        let outIdx = finalIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * actualK)
        let outDist = finalDistances?.contents().bindMemory(to: Float.self, capacity: numQueries * actualK)

        for q in 0..<numQueries {
            let rowBase = q * actualK
            let row = best[q]
            for i in 0..<actualK {
                if i < row.count {
                    outIdx[rowBase + i] = row[i].index
                    if let outDist {
                        outDist[rowBase + i] = row[i].distance
                    }
                } else {
                    outIdx[rowBase + i] = sentinel
                    if let outDist {
                        outDist[rowBase + i] = .infinity
                    }
                }
            }
        }

        return Metal4FusedL2TopKResult(
            indices: finalIndices,
            distances: finalDistances,
            numQueries: numQueries,
            k: actualK
        )
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
