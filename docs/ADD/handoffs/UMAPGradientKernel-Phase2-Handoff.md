# UMAPGradientKernel Phase 2 Handoff

## Overview

This document provides context for implementing Phase 2 of the UMAPGradientKernel - the full optimization API for GPU-accelerated UMAP gradient descent.

## Prerequisites

**Phase 1 is complete.** The following are already implemented:

| Component | Status |
|-----------|--------|
| `UMAPGradient.metal` | ✅ 4 kernels implemented |
| `UMAPGradientKernel.swift` | ✅ Basic Swift wrapper with `computeGradients()` |
| `UMAPGradientKernelTests.swift` | ✅ 9 tests passing |
| Shader registration | ✅ In `KernelContext.swift` |

## Phase 2 Scope

Implement the high-level optimization API that makes the kernel easy to use from Swift.

### Deliverables

| Component | Description |
|-----------|-------------|
| `optimizeEpoch()` | Full epoch optimization for `[[Float]]` arrays |
| `applyGradients()` | Apply computed gradients to embedding |
| Negative sampling | Integrate repulsive force computation |
| Target gradients | Apply gradients to target vertices (bidirectional) |
| Loss computation | Optional loss tracking for testing/debugging |
| Additional tests | ~6 new tests for optimization behavior |

### Expected LOC: ~300

---

## Implementation Tasks

### 1. Add `applyGradients()` Method

Add this method to `UMAPGradientKernel.swift` after the existing `computeGradients()` method:

```swift
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
```

### 2. Add `applyNegativeSampling()` Method

```swift
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
```

### 3. Add `optimizeEpoch()` High-Level API

This is the main entry point for UMAP optimization:

```swift
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
    var flatEmbedding = embedding.flatMap { $0 }

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

        // Step 1: Compute source gradients
        let sourceGradients = try await computeGradients(
            embedding: embeddingBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: edgeCount,
            params: params
        )

        // Step 2: Apply source gradients
        try await applyGradients(
            embedding: embeddingBuffer,
            gradients: sourceGradients,
            n: n,
            d: d
        )

        // Step 3: Compute and apply target gradients
        // Target gradients are computed implicitly in computeGradients but stored separately
        // For now, we apply bidirectional updates by also processing reverse edges
        // TODO: In Phase 3, optimize by using targetGradients buffer directly
        try await applyTargetGradients(
            embedding: embeddingBuffer,
            edges: edgeBuffer,
            edgeCount: edgeCount,
            n: n,
            d: d,
            params: params
        )
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

/// Applies target gradients (Newton's third law - equal and opposite).
///
/// For each edge (i, j), point j receives the negated gradient of point i.
private func applyTargetGradients(
    embedding: any MTLBuffer,
    edges: any MTLBuffer,
    edgeCount: Int,
    n: Int,
    d: Int,
    params: UMAPParameters
) async throws {
    // For Phase 2, we use a simple CPU-side accumulation for target gradients
    // Phase 3 will optimize this with a dedicated GPU kernel

    // This is acceptable for small-medium datasets (< 10K points)
    // The GPU is still doing the heavy lifting for source gradients and negative sampling

    let device = context.device.rawDevice

    // Allocate buffers for target gradient computation
    let edgeGradSize = max(edgeCount * d * MemoryLayout<Float>.size, MemoryLayout<Float>.size)
    guard let edgeGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared),
          let targetGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared) else {
        throw VectorError.bufferAllocationFailed(size: edgeGradSize)
    }

    // Compute edge gradients (this also computes targetGradients)
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

        encoder.setComputePipelineState(edgeGradientPipeline)
        encoder.setBuffer(embedding, offset: 0, index: 0)
        encoder.setBuffer(edges, offset: 0, index: 1)
        encoder.setBuffer(edgeGradients, offset: 0, index: 2)
        encoder.setBuffer(targetGradients, offset: 0, index: 3)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 4)
        dispatchLinear(encoder: encoder, pipeline: edgeGradientPipeline, count: edgeCount)
    }

    // Accumulate target gradients on CPU and apply
    // (GPU-optimized version in Phase 3)
    let embPtr = embedding.contents().bindMemory(to: Float.self, capacity: n * d)
    let tgtGradPtr = targetGradients.contents().bindMemory(to: Float.self, capacity: edgeCount * d)
    let edgesPtr = edges.contents().bindMemory(to: UMAPEdge.self, capacity: edgeCount)

    // For each edge, apply target gradient to the target vertex
    for e in 0..<edgeCount {
        let edge = edgesPtr[e]
        let j = Int(edge.target)
        for k in 0..<d {
            embPtr[j * d + k] += tgtGradPtr[e * d + k]
        }
    }
}
```

### 4. Add Loss Computation (Optional, for Testing)

```swift
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
```

### 5. Add Phase 2 Tests

Add these tests to `UMAPGradientKernelTests.swift`:

```swift
// MARK: - Phase 2: Optimization API Tests

func testOptimizeEpochReducesLoss() async throws {
    let n = 30
    let d = 2
    var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)

    let params = UMAPParameters.default
    let initialLoss = kernel.computeLoss(embedding: embedding, edges: edges, params: params)

    // Run a few optimization epochs
    for _ in 0..<5 {
        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
    }

    let finalLoss = kernel.computeLoss(embedding: embedding, edges: edges, params: params)

    XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease after optimization")
}

func testOptimizeEpochWithLearningRateDecay() async throws {
    let n = 20
    let d = 2
    var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 3)

    let nEpochs = 10
    let initialLR: Float = 1.0

    for epoch in 0..<nEpochs {
        var params = UMAPParameters.default
        params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
    }

    // Verify embedding values are finite
    for i in 0..<n {
        for k in 0..<d {
            XCTAssertFalse(embedding[i][k].isNaN, "Embedding should not be NaN after optimization")
            XCTAssertFalse(embedding[i][k].isInfinite, "Embedding should not be Inf after optimization")
        }
    }
}

func testOptimizeEpochNoEdges() async throws {
    let n = 5
    let d = 2
    var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let edges: [UMAPEdge] = []
    let originalEmbedding = embedding

    // With no edges and no negative sampling, embedding should change only due to negative sampling
    var params = UMAPParameters.default
    params.negativeSampleRate = 0  // Disable negative sampling

    try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

    // Embedding should be unchanged (no edges, no negative samples)
    for i in 0..<n {
        for k in 0..<d {
            XCTAssertEqual(embedding[i][k], originalEmbedding[i][k], accuracy: 1e-6)
        }
    }
}

func testNegativeSamplingPushesApart() async throws {
    // Two nearby points with no edges should be pushed apart by negative sampling
    var embedding: [[Float]] = [
        [0.0, 0.0],
        [0.1, 0.0],  // Very close to point 0
    ]
    let edges: [UMAPEdge] = []  // No attractive edges

    var params = UMAPParameters.default
    params.negativeSampleRate = 5
    params.learningRate = 1.0

    // Run optimization (only negative sampling will apply)
    try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

    // Compute distance after
    let distAfter = sqrt(pow(embedding[0][0] - embedding[1][0], 2) +
                         pow(embedding[0][1] - embedding[1][1], 2))

    // Points should be pushed farther apart (initial distance was 0.1)
    XCTAssertGreaterThan(distAfter, 0.1, "Negative sampling should push points apart")
}

func testApplyGradientsModifiesEmbedding() async throws {
    let n = 10
    let d = 2
    let device = context.device.rawDevice

    // Create embedding
    let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let flatEmbedding = embedding.flatMap { $0 }

    guard let embedBuffer = device.makeBuffer(
        bytes: flatEmbedding,
        length: flatEmbedding.count * MemoryLayout<Float>.size,
        options: .storageModeShared
    ) else {
        XCTFail("Failed to create embedding buffer")
        return
    }

    // Create known gradients
    let gradients = [Float](repeating: 0.5, count: n * d)
    guard let gradBuffer = device.makeBuffer(
        bytes: gradients,
        length: gradients.count * MemoryLayout<Float>.size,
        options: .storageModeShared
    ) else {
        XCTFail("Failed to create gradient buffer")
        return
    }

    // Apply gradients
    try await kernel.applyGradients(embedding: embedBuffer, gradients: gradBuffer, n: n, d: d)

    // Verify embedding was modified
    let resultPtr = embedBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
    for i in 0..<n {
        for k in 0..<d {
            let expected = flatEmbedding[i * d + k] + 0.5
            XCTAssertEqual(resultPtr[i * d + k], expected, accuracy: 1e-5)
        }
    }
}

func testLossComputation() async throws {
    let n = 10
    let d = 2
    let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 3)

    let loss = kernel.computeLoss(embedding: embedding, edges: edges, params: .default)

    // Loss should be finite and positive
    XCTAssertFalse(loss.isNaN)
    XCTAssertFalse(loss.isInfinite)
    XCTAssertGreaterThan(loss, 0)
}
```

---

## Key Implementation Notes

### 1. Bidirectional Gradient Application

UMAP applies gradients to both source and target vertices. Phase 2 handles this via:
- Source gradients: Computed via segmented reduction (GPU)
- Target gradients: Computed in edge gradient kernel, accumulated on CPU

Phase 3 will optimize target gradient accumulation with a dedicated GPU kernel.

### 2. Negative Sampling

Negative sampling provides the repulsive force that prevents all points from collapsing. The `umap_negative_sample_kernel` applies repulsive gradients directly (no accumulation needed since each point has unique samples).

### 3. Random Target Generation

For simplicity, Phase 2 generates random targets on CPU. Phase 3 could optimize this with GPU-based random number generation.

### 4. Loss Computation

The `computeLoss()` method is CPU-only and intended for testing. It computes:
```
loss = -Σ [w_ij * log(q_ij) + (1 - w_ij) * log(1 - q_ij)]
```
where `q_ij = 1 / (1 + a * d^2b)` is the low-dim similarity.

### 5. Learning Rate Decay

UMAP typically uses linear learning rate decay:
```swift
params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
```

---

## Build & Test Commands

```bash
# Build
swift build

# Run UMAP tests only
swift test --filter UMAPGradientKernelTests

# Run specific Phase 2 test
swift test --filter testOptimizeEpochReducesLoss
```

---

## Verification Checklist

- [x] `applyGradients()` method implemented
- [x] `applyNegativeSampling()` method implemented
- [x] `optimizeEpoch()` high-level API implemented
- [x] `computeLoss()` for testing implemented
- [x] Target gradient application implemented
- [x] Random target generation implemented
- [x] `testOptimizeEpochReducesLoss` passes
- [x] `testOptimizeEpochWithLearningRateDecay` passes
- [x] `testOptimizeEpochNoEdges` passes
- [x] `testNegativeSamplingPushesApart` passes
- [x] `testApplyGradientsModifiesEmbedding` passes
- [x] `testLossComputation` passes
- [x] All 15 tests pass (9 Phase 1 + 6 Phase 2)

**Phase 2 completed 2026-01-05. All 15 tests passing.**

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| `applyGradients()` | ~25 |
| `applyNegativeSampling()` | ~30 |
| `optimizeEpoch()` | ~120 |
| `applyTargetGradients()` | ~50 |
| `computeLoss()` | ~30 |
| Helper methods | ~25 |
| New tests | ~150 |
| **Total** | ~430 |

---

## Phase 3 Preview

Phase 3 (Protocol Conformance & Polish) will add:
- `FusibleKernel` protocol conformance (encode APIs)
- GPU-optimized target gradient accumulation
- VectorProtocol support for embeddings
- Performance benchmarks
- Integration example with SwiftTopics UMAP class

---

## Reference Files

```
VectorAccelerate/
├── Sources/VectorAccelerate/
│   ├── Metal/Shaders/
│   │   └── UMAPGradient.metal          # EXISTS (Phase 1)
│   └── Kernels/Metal4/
│       └── UMAPGradientKernel.swift    # MODIFY (add methods)
└── Tests/VectorAccelerateTests/
    └── UMAPGradientKernelTests.swift   # MODIFY (add tests)
```

---

## Contact

- Phase 1 handoff: `docs/handoffs/UMAPGradientKernel-Phase1-Handoff.md`
- Original spec: `docs/kernel-specs/03-UMAPGradientKernel.md`
- Reference implementations: `BoruvkaMSTKernel.swift`, `MutualReachabilityKernel.swift`

Phase 2 ready for implementation 2026-01-05.
