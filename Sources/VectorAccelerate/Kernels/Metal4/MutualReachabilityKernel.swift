//
//  MutualReachabilityKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for computing mutual reachability distances (HDBSCAN).
//
//  Features:
//  - Dense mode: Full N×N mutual reachability matrix
//  - Sparse mode: Compute specific pairs only
//  - Dimension-optimized kernels (384, 512, 768, 1536)
//  - Fusion support via encode() API
//  - VectorProtocol integration

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Parameters

/// Parameters for mutual reachability kernel.
///
/// Memory layout must match the Metal shader's `MutualReachabilityParams` struct.
public struct MutualReachabilityParams: Sendable {
    /// Number of points (N)
    public let n: UInt32

    /// Embedding dimension (D)
    public let d: UInt32

    /// Stride between embeddings (typically == d)
    public let strideEmbed: UInt32

    /// Number of pairs for sparse mode
    public let pairCount: UInt32

    /// Create parameters for dense mode.
    ///
    /// - Parameters:
    ///   - n: Number of points
    ///   - d: Embedding dimension
    public init(n: Int, d: Int) {
        self.n = UInt32(n)
        self.d = UInt32(d)
        self.strideEmbed = UInt32(d)
        self.pairCount = 0
    }

    /// Create parameters for sparse mode.
    ///
    /// - Parameters:
    ///   - d: Embedding dimension
    ///   - pairCount: Number of pairs to compute
    public init(d: Int, pairCount: Int) {
        self.n = 0  // Not used in sparse mode
        self.d = UInt32(d)
        self.strideEmbed = UInt32(d)
        self.pairCount = UInt32(pairCount)
    }

    /// Create parameters with explicit stride.
    ///
    /// - Parameters:
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - strideEmbed: Stride between embeddings
    ///   - pairCount: Number of pairs (sparse mode)
    public init(n: Int, d: Int, strideEmbed: Int, pairCount: Int = 0) {
        self.n = UInt32(n)
        self.d = UInt32(d)
        self.strideEmbed = UInt32(strideEmbed)
        self.pairCount = UInt32(pairCount)
    }
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for computing mutual reachability distances.
///
/// Mutual reachability is defined as:
/// ```
/// mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
/// ```
///
/// This metric is used in HDBSCAN clustering to account for varying local densities.
/// Points in dense regions have small core distances; points in sparse regions have
/// large core distances.
///
/// ## Modes
///
/// - **Dense mode**: Computes full N×N matrix. Use when you need all pairwise distances.
/// - **Sparse mode**: Computes specific pairs only. More memory-efficient for MST construction.
///
/// ## Performance Optimizations
///
/// - **Dimension-specific pipelines**: Hand-tuned kernels for common embedding
///   dimensions (384, 512, 768, 1536) with loop unrolling and register optimization
/// - **SIMD vectorization**: float4 operations for memory coalescing
/// - **2D dispatch**: Optimized thread group configuration for (i, j) pairs
///
/// ## Thread Safety
///
/// This kernel is thread-safe. All mutable state (pipelines) is initialized during
/// construction and never modified thereafter.
///
/// ## Embedding Dimension Coverage
///
/// | Dimension | Models | Kernel |
/// |-----------|--------|--------|
/// | 384 | MiniLM, Sentence-BERT | `mutual_reachability_384_kernel` |
/// | 512 | Small BERT variants | `mutual_reachability_512_kernel` |
/// | 768 | BERT-base, DistilBERT, MPNet | `mutual_reachability_768_kernel` |
/// | 1536 | OpenAI ada-002 | `mutual_reachability_1536_kernel` |
/// | Other | Any | `mutual_reachability_dense_kernel` (generic) |
///
/// ## Usage
///
/// ### Dense Mode
/// ```swift
/// let kernel = try await MutualReachabilityKernel(context: context)
/// let distances = try await kernel.compute(
///     embeddings: embeddingBuffer,
///     coreDistances: coreBuffer,
///     n: 1000,
///     d: 384
/// )
/// ```
///
/// ### Sparse Mode
/// ```swift
/// let kernel = try await MutualReachabilityKernel(context: context)
/// let distances = try await kernel.computeSparse(
///     embeddings: embeddingBuffer,
///     coreDistances: coreBuffer,
///     pairs: pairsBuffer,
///     pairCount: 50000,
///     d: 384
/// )
/// ```
public final class MutualReachabilityKernel: @unchecked Sendable, Metal4Kernel, DimensionOptimizedKernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "MutualReachabilityKernel"
    public let optimizedDimensions: [Int] = [384, 512, 768, 1536]
    public let fusibleWith: [String] = ["BoruvkaMST", "MinimumSpanningTree"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    /// Generic dense pipeline (any dimension)
    private let densePipeline: any MTLComputePipelineState

    /// Sparse pipeline for computing specific pairs
    private let sparsePipeline: any MTLComputePipelineState

    /// Dimension-specific optimized pipelines
    private let pipeline384: (any MTLComputePipelineState)?
    private let pipeline512: (any MTLComputePipelineState)?
    private let pipeline768: (any MTLComputePipelineState)?
    private let pipeline1536: (any MTLComputePipelineState)?

    // MARK: - Initialization

    /// Create a mutual reachability kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    /// - Throws: `VectorError.shaderNotFound` if core kernel functions are missing
    public init(context: Metal4Context) async throws {
        self.context = context

        // Load library
        let library = try await context.shaderCompiler.getDefaultLibrary()

        // Load required dense kernel
        guard let denseFunc = library.makeFunction(name: "mutual_reachability_dense_kernel") else {
            throw VectorError.shaderNotFound(
                name: "mutual_reachability_dense_kernel. Ensure MutualReachability.metal is compiled."
            )
        }

        // Load required sparse kernel
        guard let sparseFunc = library.makeFunction(name: "mutual_reachability_sparse_kernel") else {
            throw VectorError.shaderNotFound(
                name: "mutual_reachability_sparse_kernel. Ensure MutualReachability.metal is compiled."
            )
        }

        // Create core pipeline states
        let device = context.device.rawDevice
        self.densePipeline = try await device.makeComputePipelineState(function: denseFunc)
        self.sparsePipeline = try await device.makeComputePipelineState(function: sparseFunc)

        // Load dimension-optimized kernels (optional - fall back to generic if unavailable)
        if let func384 = library.makeFunction(name: "mutual_reachability_384_kernel") {
            self.pipeline384 = try? await device.makeComputePipelineState(function: func384)
        } else {
            self.pipeline384 = nil
        }

        if let func512 = library.makeFunction(name: "mutual_reachability_512_kernel") {
            self.pipeline512 = try? await device.makeComputePipelineState(function: func512)
        } else {
            self.pipeline512 = nil
        }

        if let func768 = library.makeFunction(name: "mutual_reachability_768_kernel") {
            self.pipeline768 = try? await device.makeComputePipelineState(function: func768)
        } else {
            self.pipeline768 = nil
        }

        if let func1536 = library.makeFunction(name: "mutual_reachability_1536_kernel") {
            self.pipeline1536 = try? await device.makeComputePipelineState(function: func1536)
        } else {
            self.pipeline1536 = nil
        }
    }

    // MARK: - Pipeline Selection

    /// Select the optimal pipeline for a given dimension.
    private func selectPipeline(for dimension: Int) -> (pipeline: any MTLComputePipelineState, name: String) {
        switch dimension {
        case 384:
            if let p = pipeline384 { return (p, "mutual_reachability_384_kernel") }
        case 512:
            if let p = pipeline512 { return (p, "mutual_reachability_512_kernel") }
        case 768:
            if let p = pipeline768 { return (p, "mutual_reachability_768_kernel") }
        case 1536:
            if let p = pipeline1536 { return (p, "mutual_reachability_1536_kernel") }
        default:
            break
        }
        return (densePipeline, "mutual_reachability_dense_kernel")
    }

    // MARK: - Warm Up

    /// Pre-warm pipelines (already done in init).
    public func warmUp() async throws {
        // Pipelines are created in init, this is a no-op
    }

    // MARK: - Encode API (for Fusion)

    /// Encode dense mutual reachability computation into an existing encoder.
    ///
    /// This method does NOT create or end the encoder - it only adds dispatch commands.
    /// Use this for fusing multiple operations into a single command buffer.
    ///
    /// **Important**: If fusing with subsequent operations that read the output buffer,
    /// insert `encoder.memoryBarrier(scope: .buffers)` after this call.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into
    ///   - embeddings: Embedding vectors buffer [N, D] - row-major float32
    ///   - coreDistances: Core distances buffer [N] - float32
    ///   - output: Output buffer [N, N] - must be pre-allocated
    ///   - n: Number of points
    ///   - d: Embedding dimension
    /// - Returns: Encoding result with dispatch configuration
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        output: any MTLBuffer,
        n: Int,
        d: Int
    ) -> Metal4EncodingResult {
        // Select pipeline
        let (selectedPipeline, pipelineName) = selectPipeline(for: d)

        // Configure encoder
        encoder.setComputePipelineState(selectedPipeline)
        encoder.label = "MutualReachability.\(pipelineName)"

        // Bind buffers
        encoder.setBuffer(embeddings, offset: 0, index: 0)
        encoder.setBuffer(coreDistances, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        // Bind parameters
        var params = MutualReachabilityParams(n: n, d: d)
        encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 3)

        // Calculate thread configuration
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: n,
            numDatabase: n,
            pipeline: selectedPipeline
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

    /// Encode sparse mutual reachability computation into an existing encoder.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - embeddings: Embedding vectors buffer [N, D]
    ///   - coreDistances: Core distances buffer [N]
    ///   - pairs: Pairs buffer [P] as packed uint2
    ///   - output: Output buffer [P] - must be pre-allocated
    ///   - pairCount: Number of pairs
    ///   - d: Embedding dimension
    /// - Returns: Encoding result
    @discardableResult
    public func encodeSparse(
        into encoder: any MTLComputeCommandEncoder,
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        pairs: any MTLBuffer,
        output: any MTLBuffer,
        pairCount: Int,
        d: Int
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(sparsePipeline)
        encoder.label = "MutualReachability.sparse"

        // Bind buffers
        encoder.setBuffer(embeddings, offset: 0, index: 0)
        encoder.setBuffer(coreDistances, offset: 0, index: 1)
        encoder.setBuffer(pairs, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        // Bind parameters
        var params = MutualReachabilityParams(d: d, pairCount: pairCount)
        encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 4)

        // Calculate thread configuration (1D dispatch over P pairs)
        let config = Metal4ThreadConfiguration.linear(
            count: pairCount,
            pipeline: sparsePipeline
        )

        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )

        return Metal4EncodingResult(
            pipelineName: "mutual_reachability_sparse_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Dense Mode

    /// Computes full N×N mutual reachability matrix.
    ///
    /// Automatically selects dimension-optimized kernel for dimensions 384, 512, 768, 1536.
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N×D embedding matrix (row-major Float32).
    ///   - coreDistances: Buffer containing N core distances (Float32).
    ///   - n: Number of points.
    ///   - d: Embedding dimension.
    /// - Returns: Buffer containing N×N mutual reachability matrix (row-major Float32).
    /// - Throws: `VectorError` if execution fails.
    ///
    /// - Complexity: O(N² × D) compute, O(N²) memory
    public func compute(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> any MTLBuffer {
        // Allocate output buffer
        let outputSize = n * n * MemoryLayout<Float>.size
        guard let output = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        output.label = "MutualReach.denseOutput"

        // Execute via encode()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                embeddings: embeddings,
                coreDistances: coreDistances,
                output: output,
                n: n,
                d: d
            )
        }

        return output
    }

    // MARK: - Sparse Mode

    /// Computes mutual reachability for specific pairs only.
    ///
    /// This is more memory-efficient than dense mode when you only need a subset
    /// of pairwise distances (e.g., for MST construction).
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N×D embedding matrix (row-major Float32).
    ///   - coreDistances: Buffer containing N core distances (Float32).
    ///   - pairs: Buffer containing P pairs as packed uint2 (i, j indices).
    ///   - pairCount: Number of pairs (P).
    ///   - d: Embedding dimension.
    /// - Returns: Buffer containing P mutual reachability distances (Float32).
    /// - Throws: `VectorError` if execution fails.
    ///
    /// - Complexity: O(P × D) compute, O(P) memory
    public func computeSparse(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        pairs: any MTLBuffer,
        pairCount: Int,
        d: Int
    ) async throws -> any MTLBuffer {
        // Allocate output buffer
        let outputSize = pairCount * MemoryLayout<Float>.size
        guard let output = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        output.label = "MutualReach.sparseOutput"

        // Execute via encodeSparse()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeSparse(
                into: encoder,
                embeddings: embeddings,
                coreDistances: coreDistances,
                pairs: pairs,
                output: output,
                pairCount: pairCount,
                d: d
            )
        }

        return output
    }

    // MARK: - Convenience Methods (Swift Arrays)

    /// Computes full N×N mutual reachability matrix from Swift arrays.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix as nested arrays.
    ///   - coreDistances: N core distances.
    /// - Returns: N×N mutual reachability matrix as nested arrays.
    /// - Throws: `VectorError` if execution fails.
    public func compute(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> [[Float]] {
        let n = embeddings.count
        guard n > 0 else { return [] }
        let d = embeddings[0].count

        // Flatten embeddings to row-major
        let flatEmbeddings = embeddings.flatMap { $0 }
        let device = context.device.rawDevice

        // Create buffers
        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatEmbeddings.count * MemoryLayout<Float>.size)
        }
        embedBuffer.label = "MutualReach.embeddings"

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "MutualReach.coreDistances"

        // Execute
        let outputBuffer = try await compute(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )

        // Read back and reshape to 2D
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: n * n)
        var result: [[Float]] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(UnsafeBufferPointer(start: ptr + i * n, count: n)))
        }

        return result
    }

    /// Computes mutual reachability for specific pairs from Swift arrays.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix as nested arrays.
    ///   - coreDistances: N core distances.
    ///   - pairs: Array of (i, j) index tuples.
    /// - Returns: Array of mutual reachability distances for each pair.
    /// - Throws: `VectorError` if execution fails.
    public func computeSparse(
        embeddings: [[Float]],
        coreDistances: [Float],
        pairs: [(Int, Int)]
    ) async throws -> [Float] {
        let n = embeddings.count
        guard n > 0, !pairs.isEmpty else { return [] }
        let d = embeddings[0].count

        // Flatten embeddings
        let flatEmbeddings = embeddings.flatMap { $0 }
        let device = context.device.rawDevice

        // Convert pairs to packed uint2 format
        var packedPairs: [UInt32] = []
        packedPairs.reserveCapacity(pairs.count * 2)
        for (i, j) in pairs {
            packedPairs.append(UInt32(i))
            packedPairs.append(UInt32(j))
        }

        // Create buffers
        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatEmbeddings.count * MemoryLayout<Float>.size)
        }
        embedBuffer.label = "MutualReach.embeddings"

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "MutualReach.coreDistances"

        guard let pairsBuffer = device.makeBuffer(
            bytes: packedPairs,
            length: packedPairs.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: packedPairs.count * MemoryLayout<UInt32>.size)
        }
        pairsBuffer.label = "MutualReach.pairs"

        // Execute
        let outputBuffer = try await computeSparse(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            pairs: pairsBuffer,
            pairCount: pairs.count,
            d: d
        )

        // Read back
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: pairs.count)
        return Array(UnsafeBufferPointer(start: ptr, count: pairs.count))
    }

    // MARK: - VectorProtocol API

    /// Computes mutual reachability from VectorProtocol types.
    ///
    /// Uses zero-copy buffer creation when possible.
    ///
    /// - Parameters:
    ///   - embeddings: Array of VectorProtocol-conforming vectors
    ///   - coreDistances: Core distances for each point
    /// - Returns: N×N mutual reachability matrix
    public func compute<V: VectorProtocol>(
        embeddings: [V],
        coreDistances: [Float]
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !embeddings.isEmpty else {
            throw VectorError.invalidInput("Empty embeddings array")
        }

        let n = embeddings.count
        let d = embeddings[0].count
        let device = context.device.rawDevice

        // Create buffers using zero-copy pattern
        let embedBuffer = try createBuffer(from: embeddings, device: device, label: "MutualReach.embeddings")

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "MutualReach.coreDistances"

        let outputBuffer = try await compute(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )

        return extractResults(from: outputBuffer, n: n)
    }

    /// Computes mutual reachability using StaticDimension vectors.
    ///
    /// Provides compile-time dimension safety.
    public func compute<D: StaticDimension>(
        embeddings: [Vector<D>],
        coreDistances: [Float]
    ) async throws -> [[Float]] {
        guard !embeddings.isEmpty else {
            throw VectorError.invalidInput("Empty embeddings array")
        }

        let n = embeddings.count
        let dimension = D.value
        let device = context.device.rawDevice

        let embedBuffer = try createBuffer(from: embeddings, device: device, label: "MutualReach.embeddings")

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "MutualReach.coreDistances"

        let outputBuffer = try await compute(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: dimension
        )

        return extractResults(from: outputBuffer, n: n)
    }

    // MARK: - Private Helpers

    /// Create a Metal buffer from VectorProtocol array (zero-copy pattern).
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

    /// Extract results from buffer into 2D array.
    private func extractResults(from buffer: any MTLBuffer, n: Int) -> [[Float]] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: n * n)
        var results: [[Float]] = []
        results.reserveCapacity(n)
        for i in 0..<n {
            results.append(Array(UnsafeBufferPointer(start: ptr + i * n, count: n)))
        }
        return results
    }
}
