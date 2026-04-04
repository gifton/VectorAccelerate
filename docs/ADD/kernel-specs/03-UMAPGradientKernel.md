# UMAPGradientKernel Specification

**Version**: 1.0
**Status**: Approved
**Priority**: Phase 2
**Estimated LOC**: ~1,000 (200 Metal + 450 Swift + 350 support)

---

## 1. Overview

### 1.1 Purpose

The UMAPGradientKernel computes gradients for UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction. UMAP optimizes a low-dimensional embedding by minimizing cross-entropy between high-dimensional and low-dimensional fuzzy simplicial sets.

### 1.2 Mathematical Background

UMAP optimization involves two forces:
1. **Attractive force**: Pulls similar points together
2. **Repulsive force**: Pushes dissimilar points apart

**Attractive gradient** (for edge (i, j) with weight w_ij):
```
grad_attract = -2ab × d^(2b-2) / (1 + a×d^(2b)) × (y_i - y_j) × w_ij
```

**Repulsive gradient** (for negative sample):
```
grad_repel = 2b / ((ε + d²) × (1 + a×d^(2b))) × (y_i - y_j)
```

Where:
- `a, b` = UMAP curve parameters (typically a=1.929, b=0.7915 for min_dist=0.1)
- `d²` = squared distance between low-dim embeddings y_i and y_j
- `ε` = small constant to prevent division by zero (0.001)

### 1.3 Challenge: Gradient Accumulation

Multiple edges update the same embedding point's gradient. The naive approach uses atomic operations:

```metal
atomic_fetch_add(&gradients[i], grad);  // Contention!
```

VectorAccelerate avoids atomics. We use **segmented reduction** instead.

---

## 2. Approach: Segmented Reduction

### 2.1 Strategy

Instead of atomic accumulation, we:
1. **Sort edges by source vertex**
2. **Compute per-edge gradients** (fully parallel)
3. **Segment-reduce by source** (parallel prefix sums)
4. **Scatter to embedding gradients**

This eliminates atomics while maintaining GPU parallelism.

### 2.2 Data Flow

```
Input:
  edges: [(source, target, weight)] - sorted by source
  embedding: [N, D] - current low-dim positions

Step 1: Compute per-edge gradients
  edge_grads: [E, D] - gradient contribution per edge

Step 2: Segment boundaries
  segment_starts: [N] - where each source's edges begin
  segment_counts: [N] - how many edges per source

Step 3: Segmented reduction
  For each segment (source vertex):
    sum all edge_grads in segment → point_grad

Step 4: Apply gradients
  embedding[i] += learning_rate × point_grad[i]
```

---

## 3. Input/Output Specification

### 3.1 Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embedding` | `MTLBuffer` (Float32) | [N, D] | Current low-dim embedding (D typically 2-50) |
| `edges` | `MTLBuffer` (UMAPEdge) | [E] | Edges sorted by source |
| `segmentStarts` | `MTLBuffer` (UInt32) | [N] | Start index of each source's edges |
| `segmentCounts` | `MTLBuffer` (UInt32) | [N] | Number of edges per source |
| `params` | `UMAPParams` | struct | a, b, learningRate, negativeSampleRate |

**UMAPEdge structure**:
```swift
struct UMAPEdge {
    var source: UInt32
    var target: UInt32
    var weight: Float
}
```

### 3.2 Outputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `gradients` | `MTLBuffer` (Float32) | [N, D] | Accumulated gradients per point |

### 3.3 Parameters

```swift
public struct UMAPParameters {
    /// Curve parameter 'a' (default: 1.929 for min_dist=0.1)
    public var a: Float = 1.929

    /// Curve parameter 'b' (default: 0.7915 for min_dist=0.1)
    public var b: Float = 0.7915

    /// Learning rate (default: 1.0, decays over epochs)
    public var learningRate: Float = 1.0

    /// Negative samples per positive edge (default: 5)
    public var negativeSampleRate: Int = 5

    /// Small epsilon for numerical stability
    public var epsilon: Float = 0.001
}
```

---

## 4. Metal Shader Design

### 4.1 File Location

```
Sources/VectorAccelerate/Shaders/UMAPGradient.metal
```

### 4.2 Parameter Structure

```metal
struct UMAPParams {
    float a;
    float b;
    float learningRate;
    float epsilon;
    uint n;              // Number of points
    uint d;              // Embedding dimension
    uint edgeCount;      // Number of edges
    uint negSampleRate;  // Negative samples per edge
};

struct UMAPEdge {
    uint source;
    uint target;
    float weight;
};
```

### 4.3 Kernel 1: Compute Per-Edge Gradients

```metal
kernel void umap_edge_gradient_kernel(
    device const float* embedding       [[buffer(0)]],  // [N, D]
    device const UMAPEdge* edges        [[buffer(1)]],  // [E]
    device float* edgeGradients         [[buffer(2)]],  // [E, D] per-edge grads for source
    device float* targetGradients       [[buffer(3)]],  // [E, D] per-edge grads for target (negated)
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
        distSq += diff * diff;
    }

    // Attractive gradient coefficient
    // grad_coeff = -2ab × d^(2b-2) / (1 + a×d^(2b)) × weight × lr
    float dPowB = pow(distSq, params.b);
    float dPow2Bm2 = pow(distSq, params.b - 1.0f);  // d^(2b-2) = d^(2(b-1))
    float denom = 1.0f + params.a * dPowB;
    float gradCoeff = -2.0f * params.a * params.b * dPow2Bm2 / denom;
    gradCoeff *= weight * params.learningRate;

    // Compute and store gradient for this edge
    for (uint k = 0; k < params.d; k++) {
        float diff = embedding[i * params.d + k] - embedding[j * params.d + k];
        float grad = gradCoeff * diff;
        edgeGradients[tid * params.d + k] = grad;
        targetGradients[tid * params.d + k] = -grad;  // Newton's third law
    }
}
```

### 4.4 Kernel 2: Segmented Reduction

Reduce per-edge gradients to per-point gradients using segment information.

```metal
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
```

### 4.5 Kernel 3: Apply Gradients

```metal
kernel void umap_apply_gradient_kernel(
    device float* embedding             [[buffer(0)]],  // [N, D] in/out
    device const float* gradients       [[buffer(1)]],  // [N, D]
    constant UMAPParams& params         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n * params.d) return;

    embedding[tid] += gradients[tid];
}
```

### 4.6 Kernel 4: Negative Sampling (Optional)

Repulsive force from random negative samples:

```metal
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

        float distSq = 0.0f;
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            distSq += diff * diff;
        }

        // Repulsive gradient coefficient
        // grad_coeff = 2b / ((ε + d²) × (1 + a×d^(2b))) × lr
        float dPowB = pow(distSq, params.b);
        float denom = (params.epsilon + distSq) * (1.0f + params.a * dPowB);
        float gradCoeff = 2.0f * params.b / denom * params.learningRate;

        // Apply repulsive gradient directly (no accumulation needed, one update per pair)
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            embedding[tid * params.d + k] += gradCoeff * diff;
        }
    }
}
```

---

## 5. Swift API Design

### 5.1 File Location

```
Sources/VectorAccelerate/Kernels/ML/UMAPGradientKernel.swift
```

### 5.2 Public Interface

```swift
import Metal

/// Parameters for UMAP optimization.
public struct UMAPParameters: Sendable {
    public var a: Float
    public var b: Float
    public var learningRate: Float
    public var negativeSampleRate: Int
    public var epsilon: Float

    /// Default parameters for min_dist=0.1
    public static let `default` = UMAPParameters(
        a: 1.929,
        b: 0.7915,
        learningRate: 1.0,
        negativeSampleRate: 5,
        epsilon: 0.001
    )

    /// Compute a, b from min_dist parameter
    public static func from(minDist: Float, spread: Float = 1.0) -> UMAPParameters {
        // Curve fitting to match UMAP's find_ab_params
        let b = (spread - minDist) / spread
        let a = pow(2.0, 2.0 * b) - 1.0
        return UMAPParameters(a: a, b: b, learningRate: 1.0, negativeSampleRate: 5, epsilon: 0.001)
    }
}

/// UMAP edge for optimization.
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

/// Computes UMAP optimization gradients using segmented reduction.
///
/// This kernel avoids atomic operations by:
/// 1. Computing per-edge gradients in parallel
/// 2. Using segment-based reduction to accumulate per-point gradients
/// 3. Applying gradients to update embeddings
///
/// ## Usage
/// ```swift
/// let kernel = try UMAPGradientKernel(context: context)
///
/// // Prepare edges (must be sorted by source!)
/// let sortedEdges = edges.sorted { $0.source < $1.source }
///
/// for epoch in 0..<nEpochs {
///     params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
///     try await kernel.optimizeEpoch(
///         embedding: &embedding,
///         edges: sortedEdges,
///         params: params
///     )
/// }
/// ```
public struct UMAPGradientKernel: Metal4Kernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let edgeGradientPipeline: MTLComputePipelineState
    private let segmentReducePipeline: MTLComputePipelineState
    private let applyGradientPipeline: MTLComputePipelineState
    private let negativeSamplePipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(context: Metal4Context) throws {
        self.context = context
        self.edgeGradientPipeline = try context.makePipeline(function: "umap_edge_gradient_kernel")
        self.segmentReducePipeline = try context.makePipeline(function: "umap_segment_reduce_kernel")
        self.applyGradientPipeline = try context.makePipeline(function: "umap_apply_gradient_kernel")
        self.negativeSamplePipeline = try context.makePipeline(function: "umap_negative_sample_kernel")
    }

    // MARK: - Public API

    /// Performs one epoch of UMAP optimization.
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding (modified in place).
    ///   - edges: Edges with weights (must be sorted by source!).
    ///   - params: UMAP parameters.
    /// - Throws: If GPU execution fails.
    public func optimizeEpoch(
        embedding: inout [[Float]],
        edges: [UMAPEdge],
        params: UMAPParameters
    ) async throws {
        let n = embedding.count
        let d = embedding[0].count
        let e = edges.count

        // Prepare segment information
        let (segmentStarts, segmentCounts) = computeSegments(edges: edges, n: n)

        // Flatten embedding
        var flatEmbedding = embedding.flatMap { $0 }

        // Create buffers
        let embeddingBuffer = try context.makeBuffer(bytes: flatEmbedding, label: "UMAP.embedding")
        let edgeBuffer = try context.makeBuffer(bytes: edges, label: "UMAP.edges")
        let segmentStartsBuffer = try context.makeBuffer(bytes: segmentStarts, label: "UMAP.segmentStarts")
        let segmentCountsBuffer = try context.makeBuffer(bytes: segmentCounts, label: "UMAP.segmentCounts")

        // Intermediate buffers
        let edgeGradients = try context.makeBuffer(length: e * d * MemoryLayout<Float>.size, label: "UMAP.edgeGrads")
        let targetGradients = try context.makeBuffer(length: e * d * MemoryLayout<Float>.size, label: "UMAP.targetGrads")
        let pointGradients = try context.makeBuffer(length: n * d * MemoryLayout<Float>.size, label: "UMAP.pointGrads")

        // Execute kernels
        try await context.executeAndWait { commandBuffer, encoder in
            var gpuParams = UMAPParamsGPU(
                a: params.a,
                b: params.b,
                learningRate: params.learningRate,
                epsilon: params.epsilon,
                n: UInt32(n),
                d: UInt32(d),
                edgeCount: UInt32(e),
                negSampleRate: UInt32(params.negativeSampleRate)
            )

            // Step 1: Compute per-edge gradients
            encoder.setComputePipelineState(edgeGradientPipeline)
            encoder.setBuffer(embeddingBuffer, offset: 0, index: 0)
            encoder.setBuffer(edgeBuffer, offset: 0, index: 1)
            encoder.setBuffer(edgeGradients, offset: 0, index: 2)
            encoder.setBuffer(targetGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchThreads(encoder: encoder, pipeline: edgeGradientPipeline, count: e)

            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Segment reduction for source gradients
            encoder.setComputePipelineState(segmentReducePipeline)
            encoder.setBuffer(edgeGradients, offset: 0, index: 0)
            encoder.setBuffer(segmentStartsBuffer, offset: 0, index: 1)
            encoder.setBuffer(segmentCountsBuffer, offset: 0, index: 2)
            encoder.setBuffer(pointGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchThreads(encoder: encoder, pipeline: segmentReducePipeline, count: n)

            encoder.memoryBarrier(scope: .buffers)

            // Step 3: Apply source gradients
            encoder.setComputePipelineState(applyGradientPipeline)
            encoder.setBuffer(embeddingBuffer, offset: 0, index: 0)
            encoder.setBuffer(pointGradients, offset: 0, index: 1)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)
            dispatchThreads(encoder: encoder, pipeline: applyGradientPipeline, count: n * d)

            // Note: Target gradients and negative sampling can be added similarly
        }

        // Read back embedding
        let resultPtr = embeddingBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<n {
            for j in 0..<d {
                embedding[i][j] = resultPtr[i * d + j]
            }
        }
    }

    /// Computes gradients without applying them.
    ///
    /// Useful for custom optimizers or debugging.
    public func computeGradients(
        embedding: MTLBuffer,
        edges: MTLBuffer,
        segmentStarts: MTLBuffer,
        segmentCounts: MTLBuffer,
        n: Int,
        d: Int,
        edgeCount: Int,
        params: UMAPParameters
    ) async throws -> MTLBuffer {
        let edgeGradients = try context.makeBuffer(
            length: edgeCount * d * MemoryLayout<Float>.size,
            label: "UMAP.edgeGrads"
        )
        let targetGradients = try context.makeBuffer(
            length: edgeCount * d * MemoryLayout<Float>.size,
            label: "UMAP.targetGrads"
        )
        let pointGradients = try context.makeBuffer(
            length: n * d * MemoryLayout<Float>.size,
            label: "UMAP.pointGrads"
        )

        try await context.executeAndWait { commandBuffer, encoder in
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

            // Compute per-edge gradients
            encoder.setComputePipelineState(edgeGradientPipeline)
            encoder.setBuffer(embedding, offset: 0, index: 0)
            encoder.setBuffer(edges, offset: 0, index: 1)
            encoder.setBuffer(edgeGradients, offset: 0, index: 2)
            encoder.setBuffer(targetGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchThreads(encoder: encoder, pipeline: edgeGradientPipeline, count: edgeCount)

            encoder.memoryBarrier(scope: .buffers)

            // Segment reduction
            encoder.setComputePipelineState(segmentReducePipeline)
            encoder.setBuffer(edgeGradients, offset: 0, index: 0)
            encoder.setBuffer(segmentStarts, offset: 0, index: 1)
            encoder.setBuffer(segmentCounts, offset: 0, index: 2)
            encoder.setBuffer(pointGradients, offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
            dispatchThreads(encoder: encoder, pipeline: segmentReducePipeline, count: n)
        }

        return pointGradients
    }

    // MARK: - Private

    private func computeSegments(edges: [UMAPEdge], n: Int) -> ([UInt32], [UInt32]) {
        var starts = [UInt32](repeating: 0, count: n)
        var counts = [UInt32](repeating: 0, count: n)

        var currentSource: UInt32 = 0
        var currentStart: UInt32 = 0

        for (idx, edge) in edges.enumerated() {
            if edge.source != currentSource {
                // Finalize previous source
                if idx > 0 {
                    counts[Int(currentSource)] = UInt32(idx) - currentStart
                }
                // Start new source
                currentSource = edge.source
                currentStart = UInt32(idx)
                starts[Int(currentSource)] = currentStart
            }
        }
        // Finalize last source
        if !edges.isEmpty {
            counts[Int(currentSource)] = UInt32(edges.count) - currentStart
        }

        return (starts, counts)
    }

    private func dispatchThreads(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let config = Metal4ThreadConfiguration.grid1D(count: count, pipeline: pipeline)
        encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
    }
}

// MARK: - GPU Structures

struct UMAPParamsGPU {
    var a: Float
    var b: Float
    var learningRate: Float
    var epsilon: Float
    var n: UInt32
    var d: UInt32
    var edgeCount: UInt32
    var negSampleRate: UInt32
}
```

---

## 6. Complexity Analysis

### 6.1 Time Complexity

| Phase | Complexity | Notes |
|-------|------------|-------|
| Edge gradient | O(E × D) | Fully parallel |
| Segment reduce | O(N × avgEdgesPerNode × D) | Parallel per point |
| Apply gradient | O(N × D) | Fully parallel |
| Negative sampling | O(N × negRate × D) | Fully parallel |

**Total per epoch**: O(E × D) where E ≈ N × k (k = n_neighbors)

### 6.2 Space Complexity

| Buffer | Size |
|--------|------|
| embedding | O(N × D) |
| edges | O(E) |
| edgeGradients | O(E × D) |
| pointGradients | O(N × D) |
| **Total** | O(E × D) |

For N=5000, k=15, D=15: ~5000 × 15 × 15 × 4 bytes ≈ 4.5 MB

---

## 7. Performance Considerations

### 7.1 Edge Sorting Requirement

Edges **must be sorted by source** for segment reduction to work. This is a one-time O(E log E) cost before optimization.

### 7.2 Segment Load Imbalance

If some vertices have many more edges than others, segment reduction may have load imbalance. Mitigations:
- Use warp-level reduction for small segments
- Split large segments across multiple threadgroups

### 7.3 Expected Performance

| N | E (k=15) | D | Epochs | Estimated Time |
|---|----------|---|--------|----------------|
| 1000 | 15K | 15 | 200 | ~3s |
| 5000 | 75K | 15 | 200 | ~15s |
| 10000 | 150K | 15 | 200 | ~30s |

---

## 8. Testing Requirements

### 8.1 Correctness Tests

```swift
// Test: Gradients match CPU reference
func testGradientCorrectness() async throws {
    let gpuGrads = try await kernel.computeGradients(...)
    let cpuGrads = cpuUMAPGradients(...)
    XCTAssertEqual(gpuGrads, cpuGrads, accuracy: 1e-4)
}

// Test: Optimization reduces loss
func testLossDecreases() async throws {
    var embedding = initialEmbedding
    let initialLoss = computeUMAPLoss(embedding, edges)

    for _ in 0..<10 {
        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
    }

    let finalLoss = computeUMAPLoss(embedding, edges)
    XCTAssertLessThan(finalLoss, initialLoss)
}

// Test: Segment computation is correct
func testSegmentComputation() {
    let edges = [
        UMAPEdge(source: 0, target: 1, weight: 1.0),
        UMAPEdge(source: 0, target: 2, weight: 1.0),
        UMAPEdge(source: 1, target: 0, weight: 1.0),
        UMAPEdge(source: 2, target: 0, weight: 1.0),
    ]
    let (starts, counts) = computeSegments(edges: edges, n: 3)
    XCTAssertEqual(starts, [0, 2, 3])
    XCTAssertEqual(counts, [2, 1, 1])
}
```

### 8.2 Edge Cases

- Single point: No gradients
- Disconnected components: Each optimizes independently
- Zero-weight edges: No contribution
- All edges to same target: High gradient magnitude

---

## 9. Integration Notes

### 9.1 With SwiftTopics

```swift
// SwiftTopics UMAP class would use:
public class UMAP {
    private let kernel: UMAPGradientKernel

    public func fit(data: [[Float]]) async throws -> [[Float]] {
        // 1. Build k-NN graph (using FusedL2TopKKernel)
        let knnGraph = try await buildKNNGraph(data)

        // 2. Compute fuzzy simplicial set (edge weights)
        let edges = computeFuzzyEdges(knnGraph)

        // 3. Sort edges by source
        let sortedEdges = edges.sorted { $0.source < $1.source }

        // 4. Initialize low-dim embedding
        var embedding = initializeEmbedding(n: data.count, d: nComponents)

        // 5. Optimize
        for epoch in 0..<nEpochs {
            var params = UMAPParameters.default
            params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
            try await kernel.optimizeEpoch(embedding: &embedding, edges: sortedEdges, params: params)
        }

        return embedding
    }
}
```

---

## 10. Future Enhancements

### 10.1 Batch Processing

Process multiple epochs in a single GPU submission to reduce CPU-GPU sync overhead.

### 10.2 Gradient Clipping

Add optional gradient magnitude clipping for stability:
```metal
float gradMag = length(grad);
if (gradMag > maxGrad) {
    grad *= maxGrad / gradMag;
}
```

### 10.3 Momentum / Adam

Add support for momentum-based optimizers:
- Store velocity buffer
- Update: v = β×v + (1-β)×grad; embedding += lr×v
