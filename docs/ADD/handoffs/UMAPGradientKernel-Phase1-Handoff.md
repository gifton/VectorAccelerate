# UMAPGradientKernel Phase 1 Handoff

## Overview

This document provides context for implementing Phase 1 of the UMAPGradientKernel - the core foundation for GPU-accelerated UMAP gradient computation using segmented reduction (no atomic operations).

## Background

### Why GPU-Accelerated UMAP Gradients?

UMAP optimization is a bottleneck for dimensionality reduction in topic modeling pipelines. Each epoch requires computing gradients for all edges, which scales as O(E × D) where E ≈ N × k.

| Corpus Size | Edges (k=15) | CPU Time/Epoch | Expected GPU Time |
|-------------|--------------|----------------|-------------------|
| 1,000 docs | 15K | ~15ms | ~1ms |
| 5,000 docs | 75K | ~75ms | ~5ms |
| 10,000 docs | 150K | ~150ms | ~10ms |

### Challenge: Gradient Accumulation

Multiple edges update the same point's gradient. The naive approach uses atomic operations:

```metal
atomic_fetch_add(&gradients[i], grad);  // High contention!
```

**VectorAccelerate avoids atomics.** We use **segmented reduction** instead.

### Segmented Reduction Strategy

1. **Sort edges by source vertex** (preprocessing, one-time)
2. **Compute per-edge gradients** (fully parallel)
3. **Segment-reduce by source** (parallel per point)
4. **Apply gradients** (fully parallel)

This eliminates atomics while maintaining GPU parallelism.

### Mathematical Background

UMAP optimization involves two forces:

**Attractive gradient** (for edge (i, j) with weight w_ij):
```
grad_attract = -2ab × d^(2b-2) / (1 + a×d^(2b)) × (y_i - y_j) × w_ij × lr
```

**Repulsive gradient** (for negative sample):
```
grad_repel = 2b / ((ε + d²) × (1 + a×d^(2b))) × (y_i - y_j) × lr
```

Where:
- `a, b` = UMAP curve parameters (typically a=1.929, b=0.7915 for min_dist=0.1)
- `d²` = squared distance between low-dim embeddings y_i and y_j
- `ε` = small constant to prevent division by zero (0.001)
- `lr` = learning rate

---

## Implementation Phases Overview

| Phase | Focus | LOC | Deliverables |
|-------|-------|-----|--------------|
| **1** | Core Foundation | ~450 | Metal shaders, basic Swift wrapper, core tests |
| **2** | Optimization API | ~300 | `optimizeEpoch()`, loss computation, full test coverage |
| **3** | Protocol Conformance | ~250 | Metal4Kernel, FusibleKernel, VectorProtocol, benchmarks |

---

## Phase 1 Scope

Implement the core algorithm with all Metal kernels and basic Swift infrastructure.

### Deliverables

| Component | Description |
|-----------|-------------|
| `UMAPGradient.metal` | 4 GPU kernels + parameter structs |
| `UMAPGradientKernel.swift` | Basic Swift API with pipeline initialization |
| `UMAPGradientKernelTests.swift` | 6-8 correctness tests |

### Expected LOC: ~450

---

## Implementation Tasks

### 1. Create Metal Shader File

**File**: `Sources/VectorAccelerate/Metal/Shaders/UMAPGradient.metal`

```metal
//
//  UMAPGradient.metal
//  VectorAccelerate
//
//  GPU kernels for UMAP gradient computation using segmented reduction.
//
//  Phase 1: Core kernels
//
//  Kernels:
//  - umap_edge_gradient_kernel: Compute per-edge gradients (attractive force)
//  - umap_segment_reduce_kernel: Reduce edge gradients to point gradients
//  - umap_apply_gradient_kernel: Apply gradients to embedding
//  - umap_negative_sample_kernel: Compute repulsive gradients from negative samples

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameter Structures

struct UMAPParams {
    float a;              // Curve parameter a (default: 1.929)
    float b;              // Curve parameter b (default: 0.7915)
    float learningRate;   // Learning rate
    float epsilon;        // Small constant for numerical stability (0.001)
    uint n;               // Number of points
    uint d;               // Embedding dimension (typically 2-50)
    uint edgeCount;       // Number of edges
    uint negSampleRate;   // Negative samples per point
};

struct UMAPEdge {
    uint source;
    uint target;
    float weight;
};

// MARK: - Kernel 1: Edge Gradient Computation

/// Computes attractive gradients for all edges in parallel.
///
/// Each thread processes one edge and computes the gradient contribution
/// for both source and target points.
///
/// Output:
/// - edgeGradients: Gradient contribution for the source point
/// - targetGradients: Gradient contribution for the target point (negated)
kernel void umap_edge_gradient_kernel(
    device const float* embedding       [[buffer(0)]],  // [N, D]
    device const UMAPEdge* edges        [[buffer(1)]],  // [E]
    device float* edgeGradients         [[buffer(2)]],  // [E, D] grads for source
    device float* targetGradients       [[buffer(3)]],  // [E, D] grads for target (negated)
    constant UMAPParams& params         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.edgeCount) return;

    UMAPEdge edge = edges[tid];
    uint i = edge.source;
    uint j = edge.target;
    float weight = edge.weight;

    // Compute squared distance in low-dim space
    float distSq = 0.0f;
    for (uint k = 0; k < params.d; k++) {
        float diff = embedding[i * params.d + k] - embedding[j * params.d + k];
        distSq = fma(diff, diff, distSq);
    }

    // Attractive gradient coefficient
    // grad_coeff = -2ab × d^(2b-2) / (1 + a×d^(2b)) × weight × lr
    //
    // Note: d^(2b-2) = (d²)^(b-1) and d^(2b) = (d²)^b
    float distSqPowB = pow(max(distSq, params.epsilon), params.b);
    float distSqPowBm1 = pow(max(distSq, params.epsilon), params.b - 1.0f);
    float denom = 1.0f + params.a * distSqPowB;

    float gradCoeff = -2.0f * params.a * params.b * distSqPowBm1 / denom;
    gradCoeff *= weight * params.learningRate;

    // Clamp gradient coefficient to prevent numerical issues
    gradCoeff = clamp(gradCoeff, -4.0f, 4.0f);

    // Compute and store gradient for this edge
    for (uint k = 0; k < params.d; k++) {
        float diff = embedding[i * params.d + k] - embedding[j * params.d + k];
        float grad = gradCoeff * diff;
        edgeGradients[tid * params.d + k] = grad;
        targetGradients[tid * params.d + k] = -grad;  // Newton's third law
    }
}

// MARK: - Kernel 2: Segmented Reduction

/// Reduces per-edge gradients to per-point gradients using segment information.
///
/// Each thread handles one point and sums all edge gradients in its segment.
/// Edges must be sorted by source for this to work correctly.
kernel void umap_segment_reduce_kernel(
    device const float* edgeGradients   [[buffer(0)]],  // [E, D]
    device const uint* segmentStarts    [[buffer(1)]],  // [N]
    device const uint* segmentCounts    [[buffer(2)]],  // [N]
    device float* pointGradients        [[buffer(3)]],  // [N, D]
    constant UMAPParams& params         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint start = segmentStarts[tid];
    uint count = segmentCounts[tid];

    // Initialize gradient to zero
    for (uint k = 0; k < params.d; k++) {
        pointGradients[tid * params.d + k] = 0.0f;
    }

    // Sum all edge gradients in this segment
    for (uint e = 0; e < count; e++) {
        uint edgeIdx = start + e;
        for (uint k = 0; k < params.d; k++) {
            pointGradients[tid * params.d + k] += edgeGradients[edgeIdx * params.d + k];
        }
    }
}

// MARK: - Kernel 3: Apply Gradients

/// Applies accumulated gradients to the embedding.
///
/// Processes N×D elements in parallel (one thread per element).
kernel void umap_apply_gradient_kernel(
    device float* embedding             [[buffer(0)]],  // [N, D] in/out
    device const float* gradients       [[buffer(1)]],  // [N, D]
    constant UMAPParams& params         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n * params.d) return;

    embedding[tid] += gradients[tid];
}

// MARK: - Kernel 4: Negative Sampling

/// Computes repulsive gradients from random negative samples.
///
/// Each point is pushed away from randomly selected non-neighbor points.
/// Updates are applied directly to embedding (no accumulation needed since
/// each point has its own unique set of negative samples).
kernel void umap_negative_sample_kernel(
    device float* embedding             [[buffer(0)]],  // [N, D]
    device const uint* randomTargets    [[buffer(1)]],  // [N × negRate] random indices
    constant UMAPParams& params         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    for (uint s = 0; s < params.negSampleRate; s++) {
        uint j = randomTargets[tid * params.negSampleRate + s];
        if (j == tid) continue;  // Skip self

        // Compute squared distance
        float distSq = 0.0f;
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            distSq = fma(diff, diff, distSq);
        }

        // Repulsive gradient coefficient
        // grad_coeff = 2b / ((ε + d²) × (1 + a×d^(2b))) × lr
        float distSqPowB = pow(max(distSq, params.epsilon), params.b);
        float denom = (params.epsilon + distSq) * (1.0f + params.a * distSqPowB);
        float gradCoeff = 2.0f * params.b / denom * params.learningRate;

        // Clamp gradient coefficient
        gradCoeff = clamp(gradCoeff, -4.0f, 4.0f);

        // Apply repulsive gradient directly
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            embedding[tid * params.d + k] += gradCoeff * diff;
        }
    }
}
```

### 2. Register Shader in KernelContext

**File**: `Sources/VectorAccelerate/Core/KernelContext.swift`

Add `"UMAPGradient"` to the shader compile list (around line ~137 where other shaders are registered).

### 3. Create Swift Kernel

**File**: `Sources/VectorAccelerate/Kernels/Metal4/UMAPGradientKernel.swift`

```swift
//
//  UMAPGradientKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for UMAP gradient computation using segmented reduction.
//
//  Phase 1: Core Foundation
//
//  Features:
//  - Per-edge gradient computation (attractive force)
//  - Segmented reduction (no atomics)
//  - Negative sampling (repulsive force)
//  - Gradient application

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

        let device = context.device.rawDevice
        self.edgeGradientPipeline = try await device.makeComputePipelineState(function: edgeGradFunc)
        self.segmentReducePipeline = try await device.makeComputePipelineState(function: segmentReduceFunc)
        self.applyGradientPipeline = try await device.makeComputePipelineState(function: applyGradFunc)
        self.negativeSamplePipeline = try await device.makeComputePipelineState(function: negSampleFunc)
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
        let edgeGradSize = edgeCount * d * MemoryLayout<Float>.size
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

            // Step 1: Compute per-edge gradients
            encoder.setComputePipelineState(edgeGradientPipeline)
            encoder.label = "UMAP.edgeGradient"
            encoder.setBuffer(embedding, offset: 0, index: 0)
            encoder.setBuffer(edges, offset: 0, index: 1)
            encoder.setBuffer(edgeGradients, offset: 0, index: 2)
            encoder.setBuffer(targetGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchLinear(encoder: encoder, pipeline: edgeGradientPipeline, count: edgeCount)

            encoder.memoryBarrier(scope: .buffers)

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

    // MARK: - Private Helpers

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
```

### 4. Create Test File

**File**: `Tests/VectorAccelerateTests/UMAPGradientKernelTests.swift`

```swift
// VectorAccelerate: UMAPGradientKernel Tests
//
// Tests for GPU-accelerated UMAP gradient computation.
//
// Phase 1: Core Foundation Tests
// - Segment computation correctness
// - Gradient computation produces valid output
// - Gradient signs are correct (attractive pulls together)
// - Edge cases
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {

    /// CPU reference implementation for UMAP edge gradient.
    ///
    /// Computes the attractive gradient for a single edge.
    static func cpuUMAPEdgeGradient(
        embedding: [[Float]],
        edge: UMAPEdge,
        params: UMAPParameters
    ) -> [Float] {
        let i = Int(edge.source)
        let j = Int(edge.target)
        let d = embedding[0].count

        // Compute squared distance
        var distSq: Float = 0
        for k in 0..<d {
            let diff = embedding[i][k] - embedding[j][k]
            distSq += diff * diff
        }

        // Gradient coefficient
        let distSqClamped = max(distSq, params.epsilon)
        let distSqPowB = pow(distSqClamped, params.b)
        let distSqPowBm1 = pow(distSqClamped, params.b - 1.0)
        let denom = 1.0 + params.a * distSqPowB

        var gradCoeff = -2.0 * params.a * params.b * distSqPowBm1 / denom
        gradCoeff *= edge.weight * params.learningRate
        gradCoeff = max(-4.0, min(4.0, gradCoeff))  // Clamp

        // Compute gradient
        var grad = [Float](repeating: 0, count: d)
        for k in 0..<d {
            let diff = embedding[i][k] - embedding[j][k]
            grad[k] = gradCoeff * diff
        }

        return grad
    }

    /// Generate random UMAP edges for testing.
    static func randomUMAPEdges(n: Int, edgesPerPoint: Int = 5) -> [UMAPEdge] {
        var edges: [UMAPEdge] = []
        edges.reserveCapacity(n * edgesPerPoint)

        for i in 0..<n {
            var targets = Set<Int>()
            while targets.count < min(edgesPerPoint, n - 1) {
                let j = Int.random(in: 0..<n)
                if j != i {
                    targets.insert(j)
                }
            }
            for j in targets {
                edges.append(UMAPEdge(
                    source: i,
                    target: j,
                    weight: Float.random(in: 0.5...1.0)
                ))
            }
        }

        // Sort by source
        return edges.sorted { $0.source < $1.source }
    }

    /// Generate random low-dimensional embedding for UMAP testing.
    static func randomLowDimEmbedding(n: Int, d: Int = 2) -> [[Float]] {
        return (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -10...10) }
        }
    }
}

// MARK: - UMAPGradientKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class UMAPGradientKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: UMAPGradientKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await UMAPGradientKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Segment Computation Tests

    func testSegmentComputationBasic() async throws {
        // Simple case: 3 points, edges sorted by source
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 1.0),
            UMAPEdge(source: 0, target: 2, weight: 1.0),
            UMAPEdge(source: 1, target: 0, weight: 1.0),
            UMAPEdge(source: 2, target: 0, weight: 1.0),
        ]

        let (starts, counts) = kernel.computeSegments(edges: edges, n: 3)

        XCTAssertEqual(starts[0], 0, "Source 0 starts at index 0")
        XCTAssertEqual(counts[0], 2, "Source 0 has 2 edges")
        XCTAssertEqual(starts[1], 2, "Source 1 starts at index 2")
        XCTAssertEqual(counts[1], 1, "Source 1 has 1 edge")
        XCTAssertEqual(starts[2], 3, "Source 2 starts at index 3")
        XCTAssertEqual(counts[2], 1, "Source 2 has 1 edge")
    }

    func testSegmentComputationWithGaps() async throws {
        // Point 1 has no outgoing edges
        let edges = [
            UMAPEdge(source: 0, target: 2, weight: 1.0),
            UMAPEdge(source: 2, target: 0, weight: 1.0),
        ]

        let (starts, counts) = kernel.computeSegments(edges: edges, n: 3)

        XCTAssertEqual(counts[0], 1, "Source 0 has 1 edge")
        XCTAssertEqual(counts[1], 0, "Source 1 has 0 edges")
        XCTAssertEqual(counts[2], 1, "Source 2 has 1 edge")
    }

    func testSegmentComputationEmpty() async throws {
        let edges: [UMAPEdge] = []
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 5)

        XCTAssertEqual(starts.count, 5)
        XCTAssertEqual(counts.count, 5)
        for i in 0..<5 {
            XCTAssertEqual(counts[i], 0, "All counts should be 0")
        }
    }

    // MARK: - Gradient Computation Tests

    func testGradientComputationProducesOutput() async throws {
        let n = 20
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice

        // Create buffers
        let flatEmbedding = embedding.flatMap { $0 }
        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create embedding buffer")
            return
        }

        guard let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create edge buffer")
            return
        }

        guard let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create starts buffer")
            return
        }

        guard let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create counts buffer")
            return
        }

        // Compute gradients
        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: edges.count,
            params: .default
        )

        // Verify output has correct size
        XCTAssertEqual(gradBuffer.length, n * d * MemoryLayout<Float>.size)

        // Verify gradients are finite (no NaN or Inf)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<(n * d) {
            XCTAssertFalse(gradPtr[i].isNaN, "Gradient should not be NaN at index \(i)")
            XCTAssertFalse(gradPtr[i].isInfinite, "Gradient should not be Inf at index \(i)")
        }
    }

    func testGradientSignIsCorrect() async throws {
        // Two points far apart should have attractive gradient pulling them together
        let embedding: [[Float]] = [
            [0.0, 0.0],   // Point 0 at origin
            [10.0, 0.0],  // Point 1 far to the right
        ]
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 1.0),
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 2)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 2,
            d: 2,
            edgeCount: 1,
            params: .default
        )

        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 4)

        // Point 0's gradient in x-direction should be positive (pull toward point 1)
        // Because gradient = coeff * (x0 - x1) = coeff * (0 - 10) = coeff * -10
        // And coeff is negative (attractive), so gradient is positive
        XCTAssertGreaterThan(
            gradPtr[0], 0,
            "Point 0 should be pulled toward point 1 (positive x gradient)"
        )
    }

    func testGradientMatchesCPUReference() async throws {
        // Small test case for exact verification
        let n = 5
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 0.8),
            UMAPEdge(source: 0, target: 2, weight: 0.6),
            UMAPEdge(source: 1, target: 0, weight: 0.8),
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let params = UMAPParameters.default
        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: edges.count,
            params: params
        )

        // Compute CPU reference (sum edge gradients per point)
        var cpuGradients = [[Float]](repeating: [Float](repeating: 0, count: d), count: n)
        for edge in edges {
            let edgeGrad = Metal4KernelTestHelpers.cpuUMAPEdgeGradient(
                embedding: embedding,
                edge: edge,
                params: params
            )
            let src = Int(edge.source)
            for k in 0..<d {
                cpuGradients[src][k] += edgeGrad[k]
            }
        }

        // Compare GPU vs CPU
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<n {
            for k in 0..<d {
                XCTAssertEqual(
                    gradPtr[i * d + k],
                    cpuGradients[i][k],
                    accuracy: 1e-4,
                    "Gradient mismatch at point \(i), dim \(k)"
                )
            }
        }
    }

    // MARK: - Edge Cases

    func testSinglePointNoEdges() async throws {
        let embedding: [[Float]] = [[1.0, 2.0]]
        let edges: [UMAPEdge] = []
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 1)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            length: MemoryLayout<UMAPEdge>.size,  // Dummy buffer
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 1,
            d: 2,
            edgeCount: 0,
            params: .default
        )

        // Gradient should be zero (no edges)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 2)
        XCTAssertEqual(gradPtr[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(gradPtr[1], 0.0, accuracy: 1e-6)
    }

    func testZeroWeightEdge() async throws {
        let embedding: [[Float]] = [
            [0.0, 0.0],
            [5.0, 0.0],
        ]
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 0.0),  // Zero weight
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 2)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 2,
            d: 2,
            edgeCount: 1,
            params: .default
        )

        // Gradient should be zero (zero weight edge)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 4)
        XCTAssertEqual(gradPtr[0], 0.0, accuracy: 1e-6, "Zero-weight edge should produce zero gradient")
        XCTAssertEqual(gradPtr[1], 0.0, accuracy: 1e-6)
    }

    // MARK: - Sort Helper Test

    func testSortEdgesBySource() async throws {
        let unsorted = [
            UMAPEdge(source: 2, target: 0, weight: 1.0),
            UMAPEdge(source: 0, target: 1, weight: 1.0),
            UMAPEdge(source: 1, target: 2, weight: 1.0),
            UMAPEdge(source: 0, target: 2, weight: 1.0),
        ]

        let sorted = kernel.sortEdgesBySource(unsorted)

        XCTAssertEqual(sorted[0].source, 0)
        XCTAssertEqual(sorted[1].source, 0)
        XCTAssertEqual(sorted[2].source, 1)
        XCTAssertEqual(sorted[3].source, 2)
    }
}
```

---

## Key Implementation Notes

### 1. Edge Sorting Requirement

Edges **must be sorted by source** for segment reduction to work. The kernel provides `sortEdgesBySource()` helper, but ideally edges come pre-sorted from the UMAP graph construction phase.

### 2. Gradient Clamping

Gradients are clamped to [-4, 4] in the Metal shader to prevent numerical instability:

```metal
gradCoeff = clamp(gradCoeff, -4.0f, 4.0f);
```

### 3. Memory Barriers

Required between kernels to ensure buffer writes are visible:

```swift
encoder.memoryBarrier(scope: .buffers)
```

### 4. Power Function Edge Cases

The `pow()` function with negative base or fractional exponent can produce NaN. We use `max(distSq, epsilon)` to ensure positive base:

```metal
float distSqClamped = max(distSq, params.epsilon);
float distSqPowB = pow(distSqClamped, params.b);
```

### 5. Target Gradients

The edge gradient kernel also computes target gradients (negated), which will be used in Phase 2 for full bidirectional gradient application.

---

## Build & Test Commands

```bash
# Build
swift build

# Run UMAP tests only
swift test --filter UMAPGradientKernelTests

# Run specific test
swift test --filter testGradientMatchesCPUReference
```

---

## Verification Checklist

- [x] `UMAPGradient.metal` compiles (4 kernels)
- [x] Shader registered in `KernelContext.swift`
- [x] `UMAPGradientKernel.swift` compiles
- [x] `UMAPParameters` struct with default values
- [x] `UMAPEdge` struct defined
- [x] `computeSegments()` method implemented
- [x] `computeGradients()` buffer API implemented
- [x] `testSegmentComputationBasic` passes
- [x] `testSegmentComputationWithGaps` passes
- [x] `testSegmentComputationEmpty` passes
- [x] `testGradientComputationProducesOutput` passes (finite values)
- [x] `testGradientSignIsCorrect` passes (attractive pulls together)
- [x] `testGradientMatchesCPUReference` passes
- [x] `testSinglePointNoEdges` passes
- [x] `testZeroWeightEdge` passes
- [x] `testSortEdgesBySource` passes

**Phase 1 completed 2026-01-05. All 9 tests passing.**

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| Metal shaders | ~150 |
| Swift kernel | ~200 |
| Public types | ~80 |
| Tests + helpers | ~250 |
| **Total** | ~680 |

---

## Next Phases Preview

### Phase 2: Full Optimization API (~300 LOC)
- `optimizeEpoch()` method for [[Float]] arrays
- `applyGradients()` method
- Negative sampling integration
- Loss computation for testing
- Tests: optimization reduces loss, learning rate decay

### Phase 3: Protocol Conformance & Polish (~250 LOC)
- `FusibleKernel` protocol conformance (encode APIs)
- `Metal4Kernel` full conformance
- VectorProtocol support
- Performance benchmarks
- Integration example with SwiftTopics UMAP class

---

## Reference Files

```
VectorAccelerate/
├── Sources/VectorAccelerate/
│   ├── Metal/Shaders/
│   │   └── UMAPGradient.metal          # CREATE
│   ├── Kernels/Metal4/
│   │   ├── UMAPGradientKernel.swift    # CREATE
│   │   └── KernelProtocol.swift        # REFERENCE
│   └── Core/
│       └── KernelContext.swift         # MODIFY (register shader)
└── Tests/VectorAccelerateTests/
    └── UMAPGradientKernelTests.swift   # CREATE
```

---

## Contact

- Original spec: `docs/kernel-specs/03-UMAPGradientKernel.md`
- Reference implementations: `BoruvkaMSTKernel.swift`, `MutualReachabilityKernel.swift`

Phase 1 ready for implementation 2026-01-05.
