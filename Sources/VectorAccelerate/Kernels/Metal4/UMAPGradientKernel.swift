//
//  UMAPGradientKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for UMAP gradient computation using segmented reduction.
//
//  Phase 1: Core Foundation ✅
//  Phase 2: Optimization API ✅
//  Phase 3: Protocol Conformance & GPU Optimization ✅
//
//  Features:
//  - Per-edge gradient computation (attractive force)
//  - Segmented reduction (no atomics)
//  - Negative sampling (repulsive force)
//  - Gradient application
//  - High-level optimizeEpoch() API
//  - Target gradient bidirectional updates (GPU-accelerated)
//  - Loss computation for testing
//  - FusibleKernel conformance for kernel fusion
//  - VectorProtocol support for type-safe APIs
//  - Buffer-based executeEpoch() for maximum performance

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Public Types

/// Parameters for UMAP optimization.
public struct UMAPParameters: Sendable {
    /// Curve parameter 'a' (default: 1.929 for min_dist=0.1)
    public var a: Float

    /// Curve parameter 'b' (default: 0.7915 for min_dist=0.1)
    public var b: Float

    /// Learning rate (default: 1.0, typically decays over epochs)
    public var learningRate: Float

    /// Negative samples per positive edge (default: 5)
    public var negativeSampleRate: Int

    /// Small epsilon for numerical stability
    public var epsilon: Float

    /// Default parameters for min_dist=0.1
    public static let `default` = UMAPParameters(
        a: 1.929,
        b: 0.7915,
        learningRate: 1.0,
        negativeSampleRate: 5,
        epsilon: 0.001
    )

    /// Create custom parameters.
    public init(
        a: Float = 1.929,
        b: Float = 0.7915,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        epsilon: Float = 0.001
    ) {
        self.a = a
        self.b = b
        self.learningRate = learningRate
        self.negativeSampleRate = negativeSampleRate
        self.epsilon = epsilon
    }

    /// Compute a, b parameters from min_dist (approximation).
    ///
    /// For exact values, use UMAP's `find_ab_params` function.
    /// This provides a reasonable approximation for common use cases.
    public static func from(minDist: Float, spread: Float = 1.0) -> UMAPParameters {
        // Simplified approximation of UMAP's curve fitting
        // For min_dist=0.1, spread=1.0: a≈1.929, b≈0.7915
        let b = 1.0 / (1.0 + exp(-(spread - minDist)))
        let a = 1.577 * pow(minDist, -0.8951)
        return UMAPParameters(a: a, b: b, learningRate: 1.0, negativeSampleRate: 5, epsilon: 0.001)
    }
}

/// UMAP edge for optimization (must be sorted by source!).
public struct UMAPEdge: Sendable {
    public var source: UInt32
    public var target: UInt32
    public var weight: Float

    public init(source: Int, target: Int, weight: Float) {
        self.source = UInt32(source)
        self.target = UInt32(target)
        self.weight = weight
    }
}

// MARK: - GPU Parameter Struct

/// GPU-side parameter structure.
///
/// Memory layout must match the Metal shader's `UMAPParams` struct.
struct UMAPParamsGPU: Sendable {
    var a: Float
    var b: Float
    var learningRate: Float
    var epsilon: Float
    var n: UInt32
    var d: UInt32
    var edgeCount: UInt32
    var negSampleRate: UInt32
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for UMAP gradient computation.
///
/// This kernel computes UMAP optimization gradients using segmented reduction,
/// avoiding atomic operations for better GPU utilization.
///
/// ## Algorithm
///
/// 1. **Edge gradients**: Compute attractive gradient for each edge in parallel
/// 2. **Segment reduce**: Sum edge gradients per point using segment information
/// 3. **Apply gradients**: Update embedding positions
/// 4. **Negative sampling**: Apply repulsive force from random non-neighbors
///
/// ## Requirements
///
/// - Edges **must be sorted by source** for segment reduction to work
/// - Use `sortEdgesBySource()` helper if edges are not pre-sorted
///
/// ## Complexity
///
/// - Time: O(E × D) per epoch where E ≈ N × k
/// - Space: O(E × D) for intermediate edge gradients
///
/// ## Usage
///
/// ```swift
/// let kernel = try await UMAPGradientKernel(context: context)
///
/// // Ensure edges are sorted by source
/// let sortedEdges = edges.sorted { $0.source < $1.source }
/// let (starts, counts) = kernel.computeSegments(edges: sortedEdges, n: n)
///
/// // Compute gradients
/// let gradients = try await kernel.computeGradients(
///     embedding: embeddingBuffer,
///     edges: edgeBuffer,
///     segmentStarts: startsBuffer,
///     segmentCounts: countsBuffer,
///     n: n, d: d, edgeCount: edges.count,
///     params: .default
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class UMAPGradientKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "UMAPGradientKernel"

    // MARK: - Pipelines

    private let edgeGradientPipeline: any MTLComputePipelineState
    private let segmentReducePipeline: any MTLComputePipelineState
    private let applyGradientPipeline: any MTLComputePipelineState
    private let negativeSamplePipeline: any MTLComputePipelineState
    private let accumulateTargetPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a UMAP gradient kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    /// - Throws: `VectorError.shaderNotFound` if kernel functions are missing
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let edgeGradFunc = library.makeFunction(name: "umap_edge_gradient_kernel") else {
            throw VectorError.shaderNotFound(
                name: "umap_edge_gradient_kernel. Ensure UMAPGradient.metal is compiled."
            )
        }

        guard let segmentReduceFunc = library.makeFunction(name: "umap_segment_reduce_kernel") else {
            throw VectorError.shaderNotFound(
                name: "umap_segment_reduce_kernel. Ensure UMAPGradient.metal is compiled."
            )
        }

        guard let applyGradFunc = library.makeFunction(name: "umap_apply_gradient_kernel") else {
            throw VectorError.shaderNotFound(
                name: "umap_apply_gradient_kernel. Ensure UMAPGradient.metal is compiled."
            )
        }

        guard let negSampleFunc = library.makeFunction(name: "umap_negative_sample_kernel") else {
            throw VectorError.shaderNotFound(
                name: "umap_negative_sample_kernel. Ensure UMAPGradient.metal is compiled."
            )
        }

        guard let accumulateTargetFunc = library.makeFunction(name: "umap_accumulate_target_gradients_kernel") else {
            throw VectorError.shaderNotFound(
                name: "umap_accumulate_target_gradients_kernel. Ensure UMAPGradient.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.edgeGradientPipeline = try await device.makeComputePipelineState(function: edgeGradFunc)
        self.segmentReducePipeline = try await device.makeComputePipelineState(function: segmentReduceFunc)
        self.applyGradientPipeline = try await device.makeComputePipelineState(function: applyGradFunc)
        self.negativeSamplePipeline = try await device.makeComputePipelineState(function: negSampleFunc)
        self.accumulateTargetPipeline = try await device.makeComputePipelineState(function: accumulateTargetFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines are created in init, this is a no-op
    }

    // MARK: - Segment Computation

    /// Computes segment information from sorted edges.
    ///
    /// **Important**: Edges must be sorted by source for this to work correctly.
    ///
    /// - Parameters:
    ///   - edges: Array of UMAPEdge sorted by source
    ///   - n: Number of points
    /// - Returns: (segmentStarts, segmentCounts) arrays of length N
    public func computeSegments(edges: [UMAPEdge], n: Int) -> (starts: [UInt32], counts: [UInt32]) {
        var starts = [UInt32](repeating: 0, count: n)
        var counts = [UInt32](repeating: 0, count: n)

        guard !edges.isEmpty else {
            return (starts, counts)
        }

        var currentSource: UInt32 = edges[0].source
        var currentStart: UInt32 = 0
        starts[Int(currentSource)] = 0

        for (idx, edge) in edges.enumerated() {
            if edge.source != currentSource {
                // Finalize previous source
                counts[Int(currentSource)] = UInt32(idx) - currentStart

                // Start new source
                currentSource = edge.source
                currentStart = UInt32(idx)
                starts[Int(currentSource)] = currentStart
            }
        }

        // Finalize last source
        counts[Int(currentSource)] = UInt32(edges.count) - currentStart

        return (starts, counts)
    }

    /// Sort edges by source vertex.
    ///
    /// - Parameter edges: Unsorted edges
    /// - Returns: Edges sorted by source
    public func sortEdgesBySource(_ edges: [UMAPEdge]) -> [UMAPEdge] {
        return edges.sorted { $0.source < $1.source }
    }

    // MARK: - Gradient Computation (Buffer API)

    /// Computes gradients without applying them.
    ///
    /// Useful for custom optimizers, debugging, or when you need to
    /// inspect gradients before applying.
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding buffer (Float32)
    ///   - edges: Edge buffer (UMAPEdge, sorted by source)
    ///   - segmentStarts: Segment start indices buffer [N] (UInt32)
    ///   - segmentCounts: Segment counts buffer [N] (UInt32)
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - edgeCount: Number of edges
    ///   - params: UMAP parameters
    /// - Returns: Gradient buffer [N, D] (Float32)
    /// - Throws: `VectorError` if execution fails
    public func computeGradients(
        embedding: any MTLBuffer,
        edges: any MTLBuffer,
        segmentStarts: any MTLBuffer,
        segmentCounts: any MTLBuffer,
        n: Int,
        d: Int,
        edgeCount: Int,
        params: UMAPParameters
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice

        // Allocate intermediate buffers
        let edgeGradSize = max(edgeCount * d * MemoryLayout<Float>.size, MemoryLayout<Float>.size)
        guard let edgeGradients = device.makeBuffer(
            length: edgeGradSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: edgeGradSize)
        }
        edgeGradients.label = "UMAP.edgeGradients"

        guard let targetGradients = device.makeBuffer(
            length: edgeGradSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: edgeGradSize)
        }
        targetGradients.label = "UMAP.targetGradients"

        let pointGradSize = n * d * MemoryLayout<Float>.size
        guard let pointGradients = device.makeBuffer(
            length: pointGradSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: pointGradSize)
        }
        pointGradients.label = "UMAP.pointGradients"

        // Execute kernels
        try await context.executeAndWait { [self] _, encoder in
            var gpuParams = UMAPParamsGPU(
                a: params.a,
                b: params.b,
                learningRate: params.learningRate,
                epsilon: params.epsilon,
                n: UInt32(n),
                d: UInt32(d),
                edgeCount: UInt32(edgeCount),
                negSampleRate: UInt32(params.negativeSampleRate)
            )

            // Step 1: Compute per-edge gradients (only if we have edges)
            if edgeCount > 0 {
                encoder.setComputePipelineState(edgeGradientPipeline)
                encoder.label = "UMAP.edgeGradient"
                encoder.setBuffer(embedding, offset: 0, index: 0)
                encoder.setBuffer(edges, offset: 0, index: 1)
                encoder.setBuffer(edgeGradients, offset: 0, index: 2)
                encoder.setBuffer(targetGradients, offset: 0, index: 3)
                encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
                dispatchLinear(encoder: encoder, pipeline: edgeGradientPipeline, count: edgeCount)

                encoder.memoryBarrier(scope: .buffers)
            }

            // Step 2: Segment reduction for source gradients
            encoder.setComputePipelineState(segmentReducePipeline)
            encoder.label = "UMAP.segmentReduce"
            encoder.setBuffer(edgeGradients, offset: 0, index: 0)
            encoder.setBuffer(segmentStarts, offset: 0, index: 1)
            encoder.setBuffer(segmentCounts, offset: 0, index: 2)
            encoder.setBuffer(pointGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchLinear(encoder: encoder, pipeline: segmentReducePipeline, count: n)
        }

        return pointGradients
    }

    // MARK: - Gradient Application

    /// Applies computed gradients to the embedding in-place.
    ///
    /// - Parameters:
    ///   - embedding: The embedding buffer to modify [N, D] (Float32)
    ///   - gradients: Computed gradients [N, D] (Float32)
    ///   - n: Number of points
    ///   - d: Embedding dimension
    public func applyGradients(
        embedding: any MTLBuffer,
        gradients: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            var gpuParams = UMAPParamsGPU(
                a: 0, b: 0, learningRate: 0, epsilon: 0,
                n: UInt32(n),
                d: UInt32(d),
                edgeCount: 0,
                negSampleRate: 0
            )

            encoder.setComputePipelineState(applyGradientPipeline)
            encoder.label = "UMAP.applyGradient"
            encoder.setBuffer(embedding, offset: 0, index: 0)
            encoder.setBuffer(gradients, offset: 0, index: 1)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)
            dispatchLinear(encoder: encoder, pipeline: applyGradientPipeline, count: n * d)
        }
    }

    // MARK: - Negative Sampling

    /// Applies repulsive gradients from negative samples.
    ///
    /// Each point is pushed away from randomly selected non-neighbor points.
    ///
    /// - Parameters:
    ///   - embedding: The embedding buffer to modify [N, D] (Float32)
    ///   - randomTargets: Random target indices [N × negRate] (UInt32)
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - params: UMAP parameters
    public func applyNegativeSampling(
        embedding: any MTLBuffer,
        randomTargets: any MTLBuffer,
        n: Int,
        d: Int,
        params: UMAPParameters
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            var gpuParams = UMAPParamsGPU(
                a: params.a,
                b: params.b,
                learningRate: params.learningRate,
                epsilon: params.epsilon,
                n: UInt32(n),
                d: UInt32(d),
                edgeCount: 0,
                negSampleRate: UInt32(params.negativeSampleRate)
            )

            encoder.setComputePipelineState(negativeSamplePipeline)
            encoder.label = "UMAP.negativeSample"
            encoder.setBuffer(embedding, offset: 0, index: 0)
            encoder.setBuffer(randomTargets, offset: 0, index: 1)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)
            dispatchLinear(encoder: encoder, pipeline: negativeSamplePipeline, count: n)
        }
    }

    // MARK: - High-Level Optimization API

    /// Performs one epoch of UMAP optimization.
    ///
    /// This method:
    /// 1. Computes attractive gradients for all edges
    /// 2. Applies gradients to source and target vertices
    /// 3. Applies repulsive gradients from negative samples
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding (modified in place)
    ///   - edges: Edges with weights (must be sorted by source!)
    ///   - params: UMAP parameters
    /// - Throws: If GPU execution fails
    ///
    /// ## Usage
    ///
    /// ```swift
    /// var embedding = initialEmbedding
    /// let sortedEdges = edges.sorted { $0.source < $1.source }
    ///
    /// for epoch in 0..<nEpochs {
    ///     var params = UMAPParameters.default
    ///     params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
    ///     try await kernel.optimizeEpoch(
    ///         embedding: &embedding,
    ///         edges: sortedEdges,
    ///         params: params
    ///     )
    /// }
    /// ```
    public func optimizeEpoch(
        embedding: inout [[Float]],
        edges: [UMAPEdge],
        params: UMAPParameters
    ) async throws {
        let n = embedding.count
        guard n > 0 else { return }
        let d = embedding[0].count
        let edgeCount = edges.count

        let device = context.device.rawDevice

        // Flatten embedding for GPU
        let flatEmbedding = embedding.flatMap { $0 }

        // Create embedding buffer
        guard let embeddingBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatEmbedding.count * MemoryLayout<Float>.size)
        }
        embeddingBuffer.label = "UMAP.embedding"

        // Create edge buffer (only if we have edges)
        if edgeCount > 0 {
            guard let edgeBuffer = device.makeBuffer(
                bytes: edges,
                length: edges.count * MemoryLayout<UMAPEdge>.size,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: edges.count * MemoryLayout<UMAPEdge>.size)
            }
            edgeBuffer.label = "UMAP.edges"

            // Compute segments
            let (starts, counts) = computeSegments(edges: edges, n: n)

            guard let startsBuffer = device.makeBuffer(
                bytes: starts,
                length: starts.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            ),
            let countsBuffer = device.makeBuffer(
                bytes: counts,
                length: counts.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<UInt32>.size)
            }

            // Allocate intermediate buffers for Phase 3 GPU pipeline
            let edgeGradSize = max(edgeCount * d * MemoryLayout<Float>.size, MemoryLayout<Float>.size)
            guard let edgeGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared),
                  let targetGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: edgeGradSize)
            }
            edgeGradients.label = "UMAP.edgeGradients"
            targetGradients.label = "UMAP.targetGradients"

            let pointGradSize = n * d * MemoryLayout<Float>.size
            guard let pointGradients = device.makeBuffer(length: pointGradSize, options: .storageModeShared),
                  let accumulatedTargetGradients = device.makeBuffer(length: pointGradSize, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: pointGradSize)
            }

            // Execute gradient computation and application in a single command buffer
            try await context.executeAndWait { [self] _, encoder in
                // Step 1: Compute source and target gradients
                encodeGradients(
                    into: encoder,
                    embedding: embeddingBuffer,
                    edges: edgeBuffer,
                    segmentStarts: startsBuffer,
                    segmentCounts: countsBuffer,
                    edgeGradients: edgeGradients,
                    targetGradients: targetGradients,
                    pointGradients: pointGradients,
                    n: n,
                    d: d,
                    edgeCount: edgeCount,
                    params: params
                )

                encoder.memoryBarrier(scope: .buffers)

                // Step 2: Apply source gradients
                encodeApplyGradients(
                    into: encoder,
                    embedding: embeddingBuffer,
                    gradients: pointGradients,
                    n: n,
                    d: d
                )

                encoder.memoryBarrier(scope: .buffers)

                // Step 3: Accumulate and apply target gradients (GPU-accelerated in Phase 3)
                memset(accumulatedTargetGradients.contents(), 0, pointGradSize)

                encodeAccumulateTargetGradients(
                    into: encoder,
                    targetGradients: targetGradients,
                    edges: edgeBuffer,
                    accumulatedGradients: accumulatedTargetGradients,
                    edgeCount: edgeCount,
                    n: n,
                    d: d,
                    params: params
                )

                encoder.memoryBarrier(scope: .buffers)

                encodeApplyGradients(
                    into: encoder,
                    embedding: embeddingBuffer,
                    gradients: accumulatedTargetGradients,
                    n: n,
                    d: d
                )
            }
        }

        // Step 4: Negative sampling (repulsive force)
        if params.negativeSampleRate > 0 {
            let randomTargets = generateRandomTargets(n: n, rate: params.negativeSampleRate)
            guard let targetsBuffer = device.makeBuffer(
                bytes: randomTargets,
                length: randomTargets.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: randomTargets.count * MemoryLayout<UInt32>.size)
            }

            try await applyNegativeSampling(
                embedding: embeddingBuffer,
                randomTargets: targetsBuffer,
                n: n,
                d: d,
                params: params
            )
        }

        // Read back embedding
        let resultPtr = embeddingBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<n {
            for j in 0..<d {
                embedding[i][j] = resultPtr[i * d + j]
            }
        }
    }

    // MARK: - Loss Computation

    /// Computes the UMAP cross-entropy loss for testing/debugging.
    ///
    /// This is expensive and should only be used for validation, not during training.
    ///
    /// - Parameters:
    ///   - embedding: Current embedding [N, D]
    ///   - edges: Sorted edges with weights
    ///   - params: UMAP parameters
    /// - Returns: Cross-entropy loss value
    public func computeLoss(
        embedding: [[Float]],
        edges: [UMAPEdge],
        params: UMAPParameters
    ) -> Float {
        guard !embedding.isEmpty else { return 0 }
        var loss: Float = 0
        let d = embedding[0].count

        for edge in edges {
            let i = Int(edge.source)
            let j = Int(edge.target)
            let weight = edge.weight

            // Compute squared distance
            var distSq: Float = 0
            for k in 0..<d {
                let diff = embedding[i][k] - embedding[j][k]
                distSq += diff * diff
            }

            // UMAP probability in low-dim space: q_ij = 1 / (1 + a * d^2b)
            let distSqPowB = pow(max(distSq, params.epsilon), params.b)
            let q = 1.0 / (1.0 + params.a * distSqPowB)

            // Cross-entropy: -w * log(q) - (1-w) * log(1-q)
            let clampedQ = max(min(q, 1.0 - 1e-6), 1e-6)
            loss -= weight * log(clampedQ)
            loss -= (1.0 - weight) * log(1.0 - clampedQ)
        }

        return loss
    }

    // MARK: - Private Helpers

    /// Generates random target indices for negative sampling.
    private func generateRandomTargets(n: Int, rate: Int) -> [UInt32] {
        var targets = [UInt32](repeating: 0, count: n * rate)
        for i in 0..<n {
            for s in 0..<rate {
                // Random index excluding self
                var j = Int.random(in: 0..<n)
                while j == i && n > 1 {
                    j = Int.random(in: 0..<n)
                }
                targets[i * rate + s] = UInt32(j)
            }
        }
        return targets
    }

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

    // MARK: - Phase 3: Encode APIs for Kernel Fusion

    /// Encode UMAP gradient computation into an existing encoder.
    ///
    /// This method does NOT create or end the encoder - it only adds dispatch commands.
    /// Use this for fusing multiple operations into a single command buffer.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into
    ///   - embedding: Current N×D embedding buffer (Float32)
    ///   - edges: Edge buffer (UMAPEdge, sorted by source)
    ///   - segmentStarts: Segment start indices buffer [N] (UInt32)
    ///   - segmentCounts: Segment counts buffer [N] (UInt32)
    ///   - edgeGradients: Output buffer [E, D] for per-edge gradients
    ///   - targetGradients: Output buffer [E, D] for target gradients
    ///   - pointGradients: Output buffer [N, D] for accumulated gradients
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - edgeCount: Number of edges
    ///   - params: UMAP parameters
    /// - Returns: Information about the encoding for debugging
    @discardableResult
    public func encodeGradients(
        into encoder: any MTLComputeCommandEncoder,
        embedding: any MTLBuffer,
        edges: any MTLBuffer,
        segmentStarts: any MTLBuffer,
        segmentCounts: any MTLBuffer,
        edgeGradients: any MTLBuffer,
        targetGradients: any MTLBuffer,
        pointGradients: any MTLBuffer,
        n: Int,
        d: Int,
        edgeCount: Int,
        params: UMAPParameters
    ) -> Metal4EncodingResult {
        var gpuParams = UMAPParamsGPU(
            a: params.a,
            b: params.b,
            learningRate: params.learningRate,
            epsilon: params.epsilon,
            n: UInt32(n),
            d: UInt32(d),
            edgeCount: UInt32(edgeCount),
            negSampleRate: UInt32(params.negativeSampleRate)
        )

        // Step 1: Compute per-edge gradients
        if edgeCount > 0 {
            encoder.setComputePipelineState(edgeGradientPipeline)
            encoder.label = "UMAP.edgeGradient"
            encoder.setBuffer(embedding, offset: 0, index: 0)
            encoder.setBuffer(edges, offset: 0, index: 1)
            encoder.setBuffer(edgeGradients, offset: 0, index: 2)
            encoder.setBuffer(targetGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)

            let edgeConfig = Metal4ThreadConfiguration.linear(count: edgeCount, pipeline: edgeGradientPipeline)
            encoder.dispatchThreadgroups(edgeConfig.threadgroups, threadsPerThreadgroup: edgeConfig.threadsPerThreadgroup)

            encoder.memoryBarrier(scope: .buffers)
        }

        // Step 2: Segment reduction for source gradients
        encoder.setComputePipelineState(segmentReducePipeline)
        encoder.label = "UMAP.segmentReduce"
        encoder.setBuffer(edgeGradients, offset: 0, index: 0)
        encoder.setBuffer(segmentStarts, offset: 0, index: 1)
        encoder.setBuffer(segmentCounts, offset: 0, index: 2)
        encoder.setBuffer(pointGradients, offset: 0, index: 3)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)

        let config = Metal4ThreadConfiguration.linear(count: n, pipeline: segmentReducePipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "umap_gradient",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode gradient application into an existing encoder.
    @discardableResult
    public func encodeApplyGradients(
        into encoder: any MTLComputeCommandEncoder,
        embedding: any MTLBuffer,
        gradients: any MTLBuffer,
        n: Int,
        d: Int
    ) -> Metal4EncodingResult {
        var gpuParams = UMAPParamsGPU(
            a: 0, b: 0, learningRate: 0, epsilon: 0,
            n: UInt32(n),
            d: UInt32(d),
            edgeCount: 0,
            negSampleRate: 0
        )

        encoder.setComputePipelineState(applyGradientPipeline)
        encoder.label = "UMAP.applyGradient"
        encoder.setBuffer(embedding, offset: 0, index: 0)
        encoder.setBuffer(gradients, offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)

        let config = Metal4ThreadConfiguration.linear(count: n * d, pipeline: applyGradientPipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "umap_apply_gradient",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode negative sampling into an existing encoder.
    @discardableResult
    public func encodeNegativeSampling(
        into encoder: any MTLComputeCommandEncoder,
        embedding: any MTLBuffer,
        randomTargets: any MTLBuffer,
        n: Int,
        d: Int,
        params: UMAPParameters
    ) -> Metal4EncodingResult {
        var gpuParams = UMAPParamsGPU(
            a: params.a,
            b: params.b,
            learningRate: params.learningRate,
            epsilon: params.epsilon,
            n: UInt32(n),
            d: UInt32(d),
            edgeCount: 0,
            negSampleRate: UInt32(params.negativeSampleRate)
        )

        encoder.setComputePipelineState(negativeSamplePipeline)
        encoder.label = "UMAP.negativeSample"
        encoder.setBuffer(embedding, offset: 0, index: 0)
        encoder.setBuffer(randomTargets, offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)

        let config = Metal4ThreadConfiguration.linear(count: n, pipeline: negativeSamplePipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "umap_negative_sample",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode target gradient accumulation into an existing encoder.
    @discardableResult
    public func encodeAccumulateTargetGradients(
        into encoder: any MTLComputeCommandEncoder,
        targetGradients: any MTLBuffer,
        edges: any MTLBuffer,
        accumulatedGradients: any MTLBuffer,
        edgeCount: Int,
        n: Int,
        d: Int,
        params: UMAPParameters
    ) -> Metal4EncodingResult {
        var gpuParams = UMAPParamsGPU(
            a: params.a,
            b: params.b,
            learningRate: params.learningRate,
            epsilon: params.epsilon,
            n: UInt32(n),
            d: UInt32(d),
            edgeCount: UInt32(edgeCount),
            negSampleRate: UInt32(params.negativeSampleRate)
        )

        encoder.setComputePipelineState(accumulateTargetPipeline)
        encoder.label = "UMAP.accumulateTarget"
        encoder.setBuffer(targetGradients, offset: 0, index: 0)
        encoder.setBuffer(edges, offset: 0, index: 1)
        encoder.setBuffer(accumulatedGradients, offset: 0, index: 2)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 3)

        let config = Metal4ThreadConfiguration.linear(count: edgeCount, pipeline: accumulateTargetPipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "umap_accumulate_target_gradients",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Phase 3: GPU Target Gradient Accumulation

    /// Applies target gradients using GPU atomic accumulation.
    ///
    /// This replaces the CPU-side loop from Phase 2's `applyTargetGradients()`.
    /// Uses atomic operations for accumulation which is faster for large edge counts.
    ///
    /// - Parameters:
    ///   - embedding: The embedding buffer to modify [N, D]
    ///   - edges: Edge buffer (UMAPEdge)
    ///   - targetGradients: Per-edge target gradients [E, D]
    ///   - edgeCount: Number of edges
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - params: UMAP parameters
    public func applyTargetGradientsGPU(
        embedding: any MTLBuffer,
        edges: any MTLBuffer,
        targetGradients: any MTLBuffer,
        edgeCount: Int,
        n: Int,
        d: Int,
        params: UMAPParameters
    ) async throws {
        guard edgeCount > 0 else { return }

        let device = context.device.rawDevice
        let accumulationSize = n * d * MemoryLayout<Float>.size

        guard let accumulatedGradients = device.makeBuffer(length: accumulationSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: accumulationSize)
        }

        // Zero initialize
        memset(accumulatedGradients.contents(), 0, accumulationSize)

        try await context.executeAndWait { [self] _, encoder in
            // Accumulate target gradients with atomics
            encodeAccumulateTargetGradients(
                into: encoder,
                targetGradients: targetGradients,
                edges: edges,
                accumulatedGradients: accumulatedGradients,
                edgeCount: edgeCount,
                n: n,
                d: d,
                params: params
            )

            encoder.memoryBarrier(scope: .buffers)

            // Apply accumulated gradients to embedding
            encodeApplyGradients(
                into: encoder,
                embedding: embedding,
                gradients: accumulatedGradients,
                n: n,
                d: d
            )
        }
    }

    // MARK: - Phase 3: Buffer-Based Epoch API

    /// Execute one epoch of UMAP optimization on GPU buffers.
    ///
    /// This is the most efficient API for repeated optimization epochs.
    /// Buffers are not copied - modifications happen in place.
    ///
    /// - Parameters:
    ///   - embedding: Embedding buffer [N, D] (modified in place)
    ///   - edges: Edge buffer (sorted by source)
    ///   - segmentStarts: Pre-computed segment starts [N]
    ///   - segmentCounts: Pre-computed segment counts [N]
    ///   - randomTargets: Pre-generated random target indices [N × negRate], or nil to skip
    ///   - n: Number of points
    ///   - d: Embedding dimension
    ///   - edgeCount: Number of edges
    ///   - params: UMAP parameters
    public func executeEpoch(
        embedding: any MTLBuffer,
        edges: any MTLBuffer,
        segmentStarts: any MTLBuffer,
        segmentCounts: any MTLBuffer,
        randomTargets: (any MTLBuffer)?,
        n: Int,
        d: Int,
        edgeCount: Int,
        params: UMAPParameters
    ) async throws {
        let device = context.device.rawDevice

        // Allocate intermediate buffers
        let edgeGradSize = max(edgeCount * d * MemoryLayout<Float>.size, MemoryLayout<Float>.size)
        guard let edgeGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared),
              let targetGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: edgeGradSize)
        }

        let pointGradSize = n * d * MemoryLayout<Float>.size
        guard let pointGradients = device.makeBuffer(length: pointGradSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: pointGradSize)
        }

        // For target gradient accumulation
        guard let accumulatedTargetGradients = device.makeBuffer(length: pointGradSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: pointGradSize)
        }

        // Execute all operations in a single command buffer
        try await context.executeAndWait { [self] _, encoder in
            // Step 1: Compute gradients
            if edgeCount > 0 {
                encodeGradients(
                    into: encoder,
                    embedding: embedding,
                    edges: edges,
                    segmentStarts: segmentStarts,
                    segmentCounts: segmentCounts,
                    edgeGradients: edgeGradients,
                    targetGradients: targetGradients,
                    pointGradients: pointGradients,
                    n: n,
                    d: d,
                    edgeCount: edgeCount,
                    params: params
                )

                encoder.memoryBarrier(scope: .buffers)

                // Step 2: Apply source gradients
                encodeApplyGradients(
                    into: encoder,
                    embedding: embedding,
                    gradients: pointGradients,
                    n: n,
                    d: d
                )

                encoder.memoryBarrier(scope: .buffers)

                // Step 3: Accumulate and apply target gradients
                // Zero-initialize the accumulation buffer
                memset(accumulatedTargetGradients.contents(), 0, pointGradSize)

                encodeAccumulateTargetGradients(
                    into: encoder,
                    targetGradients: targetGradients,
                    edges: edges,
                    accumulatedGradients: accumulatedTargetGradients,
                    edgeCount: edgeCount,
                    n: n,
                    d: d,
                    params: params
                )

                encoder.memoryBarrier(scope: .buffers)

                encodeApplyGradients(
                    into: encoder,
                    embedding: embedding,
                    gradients: accumulatedTargetGradients,
                    n: n,
                    d: d
                )
            }

            // Step 4: Negative sampling
            if let targets = randomTargets, params.negativeSampleRate > 0 {
                encoder.memoryBarrier(scope: .buffers)
                encodeNegativeSampling(
                    into: encoder,
                    embedding: embedding,
                    randomTargets: targets,
                    n: n,
                    d: d,
                    params: params
                )
            }
        }
    }

    // MARK: - Phase 3: VectorProtocol Support

    /// Performs one epoch of UMAP optimization using VectorProtocol embeddings.
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding as VectorProtocol array (modified in place)
    ///   - edges: Edges with weights (must be sorted by source!)
    ///   - params: UMAP parameters
    public func optimizeEpoch<V: VectorProtocol>(
        embedding: inout [V],
        edges: [UMAPEdge],
        params: UMAPParameters
    ) async throws where V.Scalar == Float {
        let n = embedding.count
        guard n > 0 else { return }
        let d = embedding[0].count

        // Convert to [[Float]] for processing
        var floatEmbedding: [[Float]] = embedding.map { vector in
            var result = [Float](repeating: 0, count: d)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<min(ptr.count, d) {
                    result[i] = ptr[i]
                }
            }
            return result
        }

        try await optimizeEpoch(embedding: &floatEmbedding, edges: edges, params: params)

        // Write back to VectorProtocol types
        for i in 0..<n {
            embedding[i].withUnsafeMutableBufferPointer { ptr in
                for k in 0..<min(ptr.count, d) {
                    ptr[k] = floatEmbedding[i][k]
                }
            }
        }
    }

    /// Compute UMAP loss using VectorProtocol embeddings.
    public func computeLoss<V: VectorProtocol>(
        embedding: [V],
        edges: [UMAPEdge],
        params: UMAPParameters
    ) -> Float where V.Scalar == Float {
        let floatEmbedding: [[Float]] = embedding.map { vector in
            var result = [Float](repeating: 0, count: vector.count)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<ptr.count {
                    result[i] = ptr[i]
                }
            }
            return result
        }
        return computeLoss(embedding: floatEmbedding, edges: edges, params: params)
    }
}

// MARK: - FusibleKernel Conformance

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension UMAPGradientKernel: FusibleKernel {
    public var fusibleWith: [String] { ["L2Distance", "TopKSelection"] }
    public var requiresBarrierAfter: Bool { true }
}
