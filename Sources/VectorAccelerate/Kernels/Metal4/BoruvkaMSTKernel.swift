//
//  BoruvkaMSTKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for computing Minimum Spanning Tree using Boruvka's algorithm.
//
//  Phase 1: Core Foundation
//
//  Features:
//  - GPU-parallel MST computation via Boruvka's algorithm
//  - O(N) space complexity (no N^2 distance matrix stored)
//  - ~log(N) iterations for complete MST
//  - Mutual reachability distances computed on-the-fly

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Result Type

/// Result of MST computation using Boruvka's algorithm.
public struct MSTResult: Sendable {
    /// Edges in the MST as (source, target, weight) tuples.
    /// Always contains exactly n-1 edges for n points (n > 1).
    public let edges: [(source: Int, target: Int, weight: Float)]

    /// Total weight of the MST (sum of all edge weights).
    public let totalWeight: Float

    /// Number of Boruvka iterations performed.
    public let iterations: Int

    /// Number of points in the original dataset.
    public let pointCount: Int

    public init(
        edges: [(source: Int, target: Int, weight: Float)],
        totalWeight: Float,
        iterations: Int,
        pointCount: Int
    ) {
        self.edges = edges
        self.totalWeight = totalWeight
        self.iterations = iterations
        self.pointCount = pointCount
    }
}

// MARK: - Parameters

/// Parameters for Boruvka's algorithm kernels.
///
/// Memory layout must match the Metal shader's `BoruvkaParams` struct.
struct BoruvkaParams: Sendable {
    var n: UInt32
    var d: UInt32
    var iteration: UInt32
    var _padding: UInt32 = 0

    init(n: Int, d: Int, iteration: Int) {
        self.n = UInt32(n)
        self.d = UInt32(d)
        self.iteration = UInt32(iteration)
    }
}

/// GPU-side MST edge structure.
///
/// Memory layout must match the Metal shader's `MSTEdge` struct.
struct MSTEdgeGPU {
    var source: UInt32
    var target: UInt32
    var weight: Float
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for computing Minimum Spanning Tree using Boruvka's algorithm.
///
/// This kernel is designed for HDBSCAN clustering, computing the MST over
/// mutual reachability distances. Distances are computed on-the-fly to
/// avoid O(N^2) memory usage.
///
/// ## Algorithm
///
/// Boruvka's algorithm runs in O(log N) iterations, with each iteration:
/// 1. Finding the minimum outgoing edge for each component (parallel)
/// 2. Reducing to per-component minimum edges
/// 3. Adding selected edges to the MST and merging components
///
/// ## Why Boruvka's?
///
/// | Algorithm | GPU Suitability | Complexity | Parallelism |
/// |-----------|-----------------|------------|-------------|
/// | **Boruvka's** | Excellent | O(E log V) | O(V) per iteration |
/// | Prim's | Poor | O(E log V) | Sequential edge selection |
/// | Kruskal's | Medium | O(E log E) | Requires global sorting |
///
/// ## Complexity
///
/// - Time: O(N^2 x D x log N) - dominated by distance computation per iteration
/// - Space: O(N) - no N^2 distance matrix stored
/// - Iterations: ~log_2(N)
///
/// ## Thread Safety
///
/// This kernel is thread-safe. All mutable state (pipelines) is initialized during
/// construction and never modified thereafter.
///
/// ## Usage
///
/// ```swift
/// let kernel = try await BoruvkaMSTKernel(context: context)
/// let mst = try await kernel.computeMST(
///     embeddings: embeddingBuffer,
///     coreDistances: coreBuffer,
///     n: 1000,
///     d: 384
/// )
/// print("MST has \(mst.edges.count) edges with total weight \(mst.totalWeight)")
/// ```
// MARK: - Work Buffers

/// Work buffers for Borůvka iterations.
///
/// These buffers can be pre-allocated and reused across multiple MST computations
/// or iterations when using the fusion API.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct BoruvkaWorkBuffers: Sendable {
    /// Per-point minimum edge weight found
    public let pointMinWeight: any MTLBuffer

    /// Per-point minimum edge target
    public let pointMinTarget: any MTLBuffer

    /// Per-component minimum edge weight
    public let componentMinWeight: any MTLBuffer

    /// Per-component minimum edge source
    public let componentMinSource: any MTLBuffer

    /// Per-component minimum edge target
    public let componentMinTarget: any MTLBuffer

    /// Candidate edges collected during iteration
    public let candidateEdges: any MTLBuffer

    /// Count of candidate edges
    public let edgeCount: any MTLBuffer

    /// Component IDs for union-find
    public let componentIds: any MTLBuffer
}

// MARK: - Kernel Implementation

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class BoruvkaMSTKernel: @unchecked Sendable, Metal4Kernel, DimensionOptimizedKernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "BoruvkaMSTKernel"

    /// Dimensions that have specialized optimized pipelines
    public let optimizedDimensions: [Int] = [384, 512, 768, 1536]

    /// Types of kernels this can be fused with
    public let fusibleWith: [String] = ["MutualReachabilityKernel", "FusedL2TopKKernel"]

    /// Whether a barrier is required before this kernel's output is read
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    /// Generic pipeline for finding minimum outgoing edge per point
    private let findMinPipeline: any MTLComputePipelineState

    /// Dimension-optimized find-min pipelines
    private let findMinPipeline384: (any MTLComputePipelineState)?
    private let findMinPipeline512: (any MTLComputePipelineState)?
    private let findMinPipeline768: (any MTLComputePipelineState)?
    private let findMinPipeline1536: (any MTLComputePipelineState)?

    /// Pipeline for reducing to per-component minimum
    private let componentReducePipeline: any MTLComputePipelineState

    /// Pipeline for adding edges and merging components
    private let mergePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Boruvka's MST kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    /// - Throws: `VectorError.shaderNotFound` if kernel functions are missing
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let findMinFunc = library.makeFunction(name: "boruvka_find_min_edge_kernel") else {
            throw VectorError.shaderNotFound(
                name: "boruvka_find_min_edge_kernel. Ensure BoruvkaMST.metal is compiled."
            )
        }

        guard let reduceFunc = library.makeFunction(name: "boruvka_component_reduce_kernel") else {
            throw VectorError.shaderNotFound(
                name: "boruvka_component_reduce_kernel. Ensure BoruvkaMST.metal is compiled."
            )
        }

        guard let mergeFunc = library.makeFunction(name: "boruvka_merge_kernel") else {
            throw VectorError.shaderNotFound(
                name: "boruvka_merge_kernel. Ensure BoruvkaMST.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.findMinPipeline = try await device.makeComputePipelineState(function: findMinFunc)
        self.componentReducePipeline = try await device.makeComputePipelineState(function: reduceFunc)
        self.mergePipeline = try await device.makeComputePipelineState(function: mergeFunc)

        // Initialize dimension-optimized pipelines (graceful fallback if not available)
        if let func384 = library.makeFunction(name: "boruvka_find_min_edge_384_kernel") {
            self.findMinPipeline384 = try await device.makeComputePipelineState(function: func384)
        } else {
            self.findMinPipeline384 = nil
        }

        if let func512 = library.makeFunction(name: "boruvka_find_min_edge_512_kernel") {
            self.findMinPipeline512 = try await device.makeComputePipelineState(function: func512)
        } else {
            self.findMinPipeline512 = nil
        }

        if let func768 = library.makeFunction(name: "boruvka_find_min_edge_768_kernel") {
            self.findMinPipeline768 = try await device.makeComputePipelineState(function: func768)
        } else {
            self.findMinPipeline768 = nil
        }

        if let func1536 = library.makeFunction(name: "boruvka_find_min_edge_1536_kernel") {
            self.findMinPipeline1536 = try await device.makeComputePipelineState(function: func1536)
        } else {
            self.findMinPipeline1536 = nil
        }
    }

    // MARK: - Pipeline Selection

    /// Get the optimal find-min pipeline for a given dimension.
    ///
    /// Falls back to generic pipeline if no optimized version exists.
    private func getFindMinPipeline(for dimension: Int) -> any MTLComputePipelineState {
        switch dimension {
        case 384:
            return findMinPipeline384 ?? findMinPipeline
        case 512:
            return findMinPipeline512 ?? findMinPipeline
        case 768:
            return findMinPipeline768 ?? findMinPipeline
        case 1536:
            return findMinPipeline1536 ?? findMinPipeline
        default:
            return findMinPipeline
        }
    }

    // MARK: - Warm Up

    /// Pre-warm pipelines (already done in init).
    public func warmUp() async throws {
        // Pipelines are created in init, this is a no-op
    }

    // MARK: - Public API (MTLBuffer)

    /// Computes MST from embeddings and core distances.
    ///
    /// Mutual reachability distances are computed on-the-fly within the kernel
    /// to avoid O(N^2) memory usage.
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N x D embedding matrix (row-major Float32).
    ///   - coreDistances: Buffer containing N core distances (Float32).
    ///   - n: Number of points.
    ///   - d: Embedding dimension.
    /// - Returns: MST result with edges, total weight, and iteration count.
    /// - Throws: `VectorError` if execution fails.
    ///
    /// - Complexity: O(N^2 x D x log N) time, O(N) space
    public func computeMST(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MSTResult {
        // Handle edge cases
        guard n > 0 else {
            return MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0)
        }

        if n == 1 {
            return MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 1)
        }

        let device = context.device.rawDevice

        // Allocate intermediate buffers
        guard let componentIds = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentIds.label = "Boruvka.componentIds"

        guard let pointMinWeight = device.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<Float>.size)
        }
        pointMinWeight.label = "Boruvka.pointMinWeight"

        guard let pointMinTarget = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        pointMinTarget.label = "Boruvka.pointMinTarget"

        guard let componentMinWeight = device.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<Float>.size)
        }
        componentMinWeight.label = "Boruvka.componentMinWeight"

        guard let componentMinSource = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentMinSource.label = "Boruvka.componentMinSource"

        guard let componentMinTarget = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentMinTarget.label = "Boruvka.componentMinTarget"

        // Allocate buffer for candidate edges - may be larger than N-1 due to duplicates
        // Worst case: each component adds one edge per iteration = N edges per iteration
        // With ~log(N) iterations, max ~N*log(N) edges total, but N*2 is usually enough
        let candidateBufferSize = max(n * 2, n - 1)
        guard let candidateEdges = device.makeBuffer(
            length: candidateBufferSize * MemoryLayout<MSTEdgeGPU>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: candidateBufferSize * MemoryLayout<MSTEdgeGPU>.size)
        }
        candidateEdges.label = "Boruvka.candidateEdges"

        guard let edgeCount = device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<UInt32>.size)
        }
        edgeCount.label = "Boruvka.edgeCount"

        // Initialize component IDs (each point is its own component)
        initializeComponentIds(componentIds, n: n)

        // Initialize candidate edge count to 0
        let candidateCountPtr = edgeCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        candidateCountPtr.pointee = 0

        // Track actual MST edges (deduplicated)
        var mstEdges: [(source: Int, target: Int, weight: Float)] = []
        mstEdges.reserveCapacity(n - 1)

        var iterations = 0
        let maxIterations = Int(ceil(log2(Double(n)))) + 2  // Safety margin

        while iterations < maxIterations {
            // Check if we have all N-1 edges
            if mstEdges.count >= n - 1 {
                break
            }

            let newEdges = try await runIteration(
                embeddings: embeddings,
                coreDistances: coreDistances,
                componentIds: componentIds,
                pointMinWeight: pointMinWeight,
                pointMinTarget: pointMinTarget,
                componentMinWeight: componentMinWeight,
                componentMinSource: componentMinSource,
                componentMinTarget: componentMinTarget,
                candidateEdges: candidateEdges,
                candidateCount: edgeCount,
                mstEdges: &mstEdges,
                n: n,
                d: d,
                iteration: iterations
            )

            iterations += 1

            // If no new edges were added, we're done (all connected)
            if newEdges == 0 {
                break
            }
        }

        // Calculate total weight
        let totalWeight = mstEdges.reduce(0) { $0 + $1.weight }

        return MSTResult(
            edges: mstEdges,
            totalWeight: totalWeight,
            iterations: iterations,
            pointCount: n
        )
    }

    // MARK: - Convenience Methods (Swift Arrays)

    /// Computes MST from Swift arrays.
    ///
    /// - Parameters:
    ///   - embeddings: N x D embedding matrix as nested arrays.
    ///   - coreDistances: N core distances.
    /// - Returns: MST result with edges and total weight.
    /// - Throws: `VectorError` if execution fails.
    public func computeMST(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> MSTResult {
        let n = embeddings.count
        guard n > 0 else {
            return MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0)
        }
        let d = embeddings[0].count

        let flatEmbeddings = embeddings.flatMap { $0 }
        let device = context.device.rawDevice

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatEmbeddings.count * MemoryLayout<Float>.size)
        }
        embedBuffer.label = "Boruvka.embeddings"

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "Boruvka.coreDistances"

        return try await computeMST(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )
    }

    // MARK: - FusibleKernel API

    /// Create reusable work buffers for a given problem size.
    ///
    /// Pre-allocating work buffers allows for efficient reuse across multiple
    /// MST computations or when using the encode API for fusion scenarios.
    ///
    /// - Parameter n: Number of points in the dataset
    /// - Returns: Pre-allocated work buffers
    /// - Throws: `VectorError.bufferAllocationFailed` if allocation fails
    public func createWorkBuffers(n: Int) throws -> BoruvkaWorkBuffers {
        let device = context.device.rawDevice

        guard let pointMinWeight = device.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<Float>.size)
        }
        pointMinWeight.label = "Boruvka.workBuffers.pointMinWeight"

        guard let pointMinTarget = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        pointMinTarget.label = "Boruvka.workBuffers.pointMinTarget"

        guard let componentMinWeight = device.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<Float>.size)
        }
        componentMinWeight.label = "Boruvka.workBuffers.componentMinWeight"

        guard let componentMinSource = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentMinSource.label = "Boruvka.workBuffers.componentMinSource"

        guard let componentMinTarget = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentMinTarget.label = "Boruvka.workBuffers.componentMinTarget"

        // Candidate edges buffer - worst case N edges per iteration
        let candidateBufferSize = max(n * 2, n - 1)
        guard let candidateEdges = device.makeBuffer(
            length: candidateBufferSize * MemoryLayout<MSTEdgeGPU>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: candidateBufferSize * MemoryLayout<MSTEdgeGPU>.size)
        }
        candidateEdges.label = "Boruvka.workBuffers.candidateEdges"

        guard let edgeCount = device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<UInt32>.size)
        }
        edgeCount.label = "Boruvka.workBuffers.edgeCount"

        guard let componentIds = device.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
        }
        componentIds.label = "Boruvka.workBuffers.componentIds"

        return BoruvkaWorkBuffers(
            pointMinWeight: pointMinWeight,
            pointMinTarget: pointMinTarget,
            componentMinWeight: componentMinWeight,
            componentMinSource: componentMinSource,
            componentMinTarget: componentMinTarget,
            candidateEdges: candidateEdges,
            edgeCount: edgeCount,
            componentIds: componentIds
        )
    }

    /// Encode a single Borůvka iteration into an existing encoder.
    ///
    /// This is useful for fusing with other operations. The caller is responsible
    /// for managing the command buffer lifecycle and memory barriers.
    ///
    /// - Note: This encodes ONE iteration only. For full MST, use `computeMST`.
    ///         The caller must handle CPU-side union-find merging between iterations.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - embeddings: N×D embedding buffer
    ///   - coreDistances: N core distances buffer
    ///   - workBuffers: Pre-allocated work buffers from `createWorkBuffers`
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - iteration: Current iteration number
    /// - Returns: Encoding result with dispatch info
    @discardableResult
    public func encodeIteration(
        into encoder: any MTLComputeCommandEncoder,
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        workBuffers: BoruvkaWorkBuffers,
        n: Int,
        d: Int,
        iteration: Int
    ) -> Metal4EncodingResult {
        var params = BoruvkaParams(n: n, d: d, iteration: iteration)
        let pipeline = getFindMinPipeline(for: d)

        // Step 1: Find minimum outgoing edge per point
        encoder.setComputePipelineState(pipeline)
        encoder.label = "Boruvka.findMin[\(iteration)].d\(d)"
        encoder.setBuffer(embeddings, offset: 0, index: 0)
        encoder.setBuffer(coreDistances, offset: 0, index: 1)
        encoder.setBuffer(workBuffers.componentIds, offset: 0, index: 2)
        encoder.setBuffer(workBuffers.pointMinWeight, offset: 0, index: 3)
        encoder.setBuffer(workBuffers.pointMinTarget, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 5)
        dispatchLinear(encoder: encoder, pipeline: pipeline, count: n)

        encoder.memoryBarrier(scope: .buffers)

        // Step 2: Reduce to per-component minimum
        encoder.setComputePipelineState(componentReducePipeline)
        encoder.label = "Boruvka.reduce[\(iteration)]"
        encoder.setBuffer(workBuffers.componentIds, offset: 0, index: 0)
        encoder.setBuffer(workBuffers.pointMinWeight, offset: 0, index: 1)
        encoder.setBuffer(workBuffers.pointMinTarget, offset: 0, index: 2)
        encoder.setBuffer(workBuffers.componentMinWeight, offset: 0, index: 3)
        encoder.setBuffer(workBuffers.componentMinSource, offset: 0, index: 4)
        encoder.setBuffer(workBuffers.componentMinTarget, offset: 0, index: 5)
        encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
        dispatchLinear(encoder: encoder, pipeline: componentReducePipeline, count: n)

        encoder.memoryBarrier(scope: .buffers)

        // Step 3: Collect candidate edges
        encoder.setComputePipelineState(mergePipeline)
        encoder.label = "Boruvka.collectEdges[\(iteration)]"
        encoder.setBuffer(workBuffers.componentIds, offset: 0, index: 0)
        encoder.setBuffer(workBuffers.componentMinWeight, offset: 0, index: 1)
        encoder.setBuffer(workBuffers.componentMinSource, offset: 0, index: 2)
        encoder.setBuffer(workBuffers.componentMinTarget, offset: 0, index: 3)
        encoder.setBuffer(workBuffers.candidateEdges, offset: 0, index: 4)
        encoder.setBuffer(workBuffers.edgeCount, offset: 0, index: 5)
        encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
        dispatchLinear(encoder: encoder, pipeline: mergePipeline, count: n)

        return Metal4EncodingResult(
            pipelineName: "boruvka_iteration_\(iteration)",
            threadgroups: MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, n), height: 1, depth: 1)
        )
    }

    // MARK: - VectorProtocol API

    /// Compute MST from VectorProtocol-conforming embeddings.
    ///
    /// This method provides type-safe, ergonomic access for VectorCore users.
    ///
    /// - Parameters:
    ///   - embeddings: Array of vectors conforming to VectorProtocol
    ///   - coreDistances: Core distances for each point
    /// - Returns: MST result
    public func computeMST<V: VectorProtocol>(
        embeddings: [V],
        coreDistances: [Float]
    ) async throws -> MSTResult where V.Scalar == Float {
        guard !embeddings.isEmpty else {
            return MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0)
        }

        let n = embeddings.count
        let d = embeddings[0].count
        let device = context.device.rawDevice

        // Create buffer using zero-copy pattern
        let embedBuffer = try createBufferFromVectors(embeddings, device: device)
        embedBuffer.label = "Boruvka.embeddings.VectorProtocol"

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }
        coreBuffer.label = "Boruvka.coreDistances"

        return try await computeMST(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )
    }

    /// Compute MST using StaticDimension vectors for compile-time dimension safety.
    public func computeMST<D: StaticDimension>(
        embeddings: [Vector<D>],
        coreDistances: [Float]
    ) async throws -> MSTResult {
        guard !embeddings.isEmpty else {
            return MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0)
        }

        let n = embeddings.count
        let d = D.value
        let device = context.device.rawDevice

        let embedBuffer = try createBufferFromStaticVectors(embeddings, device: device)
        embedBuffer.label = "Boruvka.embeddings.Vector<D\(d)>"

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
        }

        return try await computeMST(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )
    }

    // MARK: - Private Methods

    /// Create buffer from VectorProtocol array using zero-copy pattern.
    private func createBufferFromVectors<V: VectorProtocol>(
        _ vectors: [V],
        device: any MTLDevice
    ) throws -> any MTLBuffer where V.Scalar == Float {
        let dimension = vectors[0].count
        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.size

        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: byteSize)
        }

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

    /// Create buffer from StaticDimension vectors.
    private func createBufferFromStaticVectors<D: StaticDimension>(
        _ vectors: [Vector<D>],
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        let dimension = D.value
        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.size

        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: byteSize)
        }

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

    // MARK: - Private Methods (Iteration)

    /// Run a single Boruvka iteration (find min, reduce, collect edges, merge on CPU).
    ///
    /// Returns the number of new MST edges added this iteration.
    private func runIteration(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        componentIds: any MTLBuffer,
        pointMinWeight: any MTLBuffer,
        pointMinTarget: any MTLBuffer,
        componentMinWeight: any MTLBuffer,
        componentMinSource: any MTLBuffer,
        componentMinTarget: any MTLBuffer,
        candidateEdges: any MTLBuffer,
        candidateCount: any MTLBuffer,
        mstEdges: inout [(source: Int, target: Int, weight: Float)],
        n: Int,
        d: Int,
        iteration: Int
    ) async throws -> Int {
        // Read candidate count before this iteration
        let candidateCountPtr = candidateCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        let candidatesBefore = Int(candidateCountPtr.pointee)

        // Select dimension-optimized pipeline
        let selectedFindMinPipeline = getFindMinPipeline(for: d)

        try await context.executeAndWait { [self] _, encoder in
            var params = BoruvkaParams(n: n, d: d, iteration: iteration)

            // Step 1: Find minimum outgoing edge per point (using dimension-optimized kernel)
            encoder.setComputePipelineState(selectedFindMinPipeline)
            encoder.label = "Boruvka.findMin[\(iteration)].d\(d)"
            encoder.setBuffer(embeddings, offset: 0, index: 0)
            encoder.setBuffer(coreDistances, offset: 0, index: 1)
            encoder.setBuffer(componentIds, offset: 0, index: 2)
            encoder.setBuffer(pointMinWeight, offset: 0, index: 3)
            encoder.setBuffer(pointMinTarget, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 5)
            dispatchLinear(encoder: encoder, pipeline: selectedFindMinPipeline, count: n)

            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Reduce to per-component minimum
            encoder.setComputePipelineState(componentReducePipeline)
            encoder.label = "Boruvka.reduce[\(iteration)]"
            encoder.setBuffer(componentIds, offset: 0, index: 0)
            encoder.setBuffer(pointMinWeight, offset: 0, index: 1)
            encoder.setBuffer(pointMinTarget, offset: 0, index: 2)
            encoder.setBuffer(componentMinWeight, offset: 0, index: 3)
            encoder.setBuffer(componentMinSource, offset: 0, index: 4)
            encoder.setBuffer(componentMinTarget, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
            dispatchLinear(encoder: encoder, pipeline: componentReducePipeline, count: n)

            encoder.memoryBarrier(scope: .buffers)

            // Step 3: Collect candidate edges (component merging done on CPU)
            encoder.setComputePipelineState(mergePipeline)
            encoder.label = "Boruvka.collectEdges[\(iteration)]"
            encoder.setBuffer(componentIds, offset: 0, index: 0)
            encoder.setBuffer(componentMinWeight, offset: 0, index: 1)
            encoder.setBuffer(componentMinSource, offset: 0, index: 2)
            encoder.setBuffer(componentMinTarget, offset: 0, index: 3)
            encoder.setBuffer(candidateEdges, offset: 0, index: 4)
            encoder.setBuffer(candidateCount, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
            dispatchLinear(encoder: encoder, pipeline: mergePipeline, count: n)
        }

        // Read candidate count after GPU execution
        let candidatesAfter = Int(candidateCountPtr.pointee)
        let newCandidatesCount = candidatesAfter - candidatesBefore

        // Step 4: Merge components on CPU and collect actual MST edges
        var newMSTEdgesCount = 0
        if newCandidatesCount > 0 {
            newMSTEdgesCount = mergeComponentsAndCollectEdges(
                componentIds: componentIds,
                candidateEdges: candidateEdges,
                startCandidate: candidatesBefore,
                endCandidate: candidatesAfter,
                mstEdges: &mstEdges,
                n: n
            )
        }

        return newMSTEdgesCount
    }

    /// Merge components on CPU and collect actual MST edges (deduplicating).
    ///
    /// Uses Union-Find with path compression for efficient merging.
    /// Only edges that actually connect different components are added to MST.
    ///
    /// - Returns: Number of new edges added to MST
    private func mergeComponentsAndCollectEdges(
        componentIds: any MTLBuffer,
        candidateEdges: any MTLBuffer,
        startCandidate: Int,
        endCandidate: Int,
        mstEdges: inout [(source: Int, target: Int, weight: Float)],
        n: Int
    ) -> Int {
        let compPtr = componentIds.contents().bindMemory(to: UInt32.self, capacity: n)
        let edgePtr = candidateEdges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: endCandidate)

        // Helper: Find root with path compression
        func find(_ x: Int) -> Int {
            var x = x
            while compPtr[x] != UInt32(x) {
                let parent = Int(compPtr[x])
                compPtr[x] = compPtr[parent]  // Path compression
                x = parent
            }
            return x
        }

        var newEdgesCount = 0

        // Process each candidate edge
        for i in startCandidate..<endCandidate {
            let edge = edgePtr[i]
            let rootA = find(Int(edge.source))
            let rootB = find(Int(edge.target))

            if rootA != rootB {
                // This edge connects different components - add to MST
                mstEdges.append((
                    source: Int(edge.source),
                    target: Int(edge.target),
                    weight: edge.weight
                ))
                newEdgesCount += 1

                // Union: merge smaller root into larger (by index for simplicity)
                let newRoot = min(rootA, rootB)
                let oldRoot = max(rootA, rootB)
                compPtr[oldRoot] = UInt32(newRoot)
            }
            // Else: edge is a duplicate (both endpoints already in same component) - skip
        }

        // Flatten all component IDs to their roots
        for i in 0..<n {
            compPtr[i] = UInt32(find(i))
        }

        return newEdgesCount
    }

    /// Initialize component IDs so each point is its own component.
    private func initializeComponentIds(_ buffer: any MTLBuffer, n: Int) {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            ptr[i] = UInt32(i)
        }
    }

    /// Dispatch a linear 1D kernel.
    private func dispatchLinear(
        encoder: any MTLComputeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        count: Int
    ) {
        let config = Metal4ThreadConfiguration.linear(count: count, pipeline: pipeline)
        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }
}
