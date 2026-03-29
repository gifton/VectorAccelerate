# UMAPGradientKernel Phase 3 Handoff

## Overview

This document provides context for implementing Phase 3 of the UMAPGradientKernel - Protocol Conformance, GPU Optimization, and Integration.

## Prerequisites

**Phases 1 and 2 are complete.** The following are implemented:

| Component | Status |
|-----------|--------|
| `UMAPGradient.metal` | ✅ 4 kernels implemented |
| `UMAPGradientKernel.swift` | ✅ Core + Optimization API |
| `UMAPGradientKernelTests.swift` | ✅ 15 tests passing |
| `computeGradients()` | ✅ Buffer-level API |
| `applyGradients()` | ✅ GPU gradient application |
| `applyNegativeSampling()` | ✅ Repulsive force |
| `optimizeEpoch()` | ✅ High-level `[[Float]]` API |
| `computeLoss()` | ✅ CPU loss for testing |
| Target gradients | ⚠️ CPU-side accumulation (optimization target) |

## Phase 3 Scope

Complete the kernel with protocol conformance, performance optimizations, and production-ready features.

### Deliverables

| Component | Description |
|-----------|-------------|
| `FusibleKernel` conformance | `encode()` APIs for kernel fusion |
| GPU target gradient accumulation | New Metal kernel to replace CPU loop |
| VectorProtocol support | Type-safe embedding APIs |
| `Metal4DistanceKernel`-style API | Consistent with other VectorAccelerate kernels |
| Performance benchmarks | Verify GPU speedup |
| Integration example | SwiftTopics UMAP usage |

### Expected LOC: ~400

---

## Implementation Tasks

### 1. Add FusibleKernel Protocol Conformance

The kernel should support fusion with other operations in a shared command encoder.

```swift
// Add to UMAPGradientKernel class declaration
extension UMAPGradientKernel: FusibleKernel {
    public var fusibleWith: [String] { ["L2Distance", "TopKSelection"] }
    public var requiresBarrierAfter: Bool { true }
}
```

### 2. Add Encode API for Gradient Computation

Add `encode()` method following the pattern from `L2DistanceKernel`:

```swift
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

        let config = Metal4ThreadConfiguration.linear(count: edgeCount, pipeline: edgeGradientPipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

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
```

### 3. GPU-Optimized Target Gradient Accumulation

The current Phase 2 implementation accumulates target gradients on CPU. Add a GPU kernel.

#### 3.1 New Metal Kernel

Add to `UMAPGradient.metal`:

```metal
// MARK: - Kernel 5: Target Gradient Accumulation

/// Accumulates target gradients using atomic adds.
///
/// This kernel atomically adds each edge's target gradient to the corresponding
/// target point. While atomics have contention, this is typically faster than
/// CPU-side accumulation for large edge counts.
///
/// For very high-contention scenarios (many edges to same target), consider
/// the segment-reduce approach used for source gradients.
kernel void umap_accumulate_target_gradients_kernel(
    device const float* targetGradients   [[buffer(0)]],  // [E, D] per-edge target grads
    device const UMAPEdge* edges          [[buffer(1)]],  // [E]
    device atomic_float* pointGradients   [[buffer(2)]],  // [N, D] accumulated output
    constant UMAPParams& params           [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.edgeCount) return;

    UMAPEdge edge = edges[tid];
    uint j = edge.target;  // Target point index

    // Atomically accumulate gradient for target point
    for (uint k = 0; k < params.d; k++) {
        float grad = targetGradients[tid * params.d + k];
        atomic_fetch_add_explicit(
            &pointGradients[j * params.d + k],
            grad,
            memory_order_relaxed
        );
    }
}

/// Alternative: Segment-reduce for target gradients (no atomics).
///
/// Requires edges to be sorted by TARGET (not source).
/// Use when atomic contention is too high.
kernel void umap_segment_reduce_target_kernel(
    device const float* targetGradients   [[buffer(0)]],  // [E, D]
    device const uint* targetSegmentStarts [[buffer(1)]], // [N] start idx for each target
    device const uint* targetSegmentCounts [[buffer(2)]], // [N] count for each target
    device float* pointGradients          [[buffer(3)]],  // [N, D]
    constant UMAPParams& params           [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint start = targetSegmentStarts[tid];
    uint count = targetSegmentCounts[tid];

    // Initialize gradient to zero
    for (uint k = 0; k < params.d; k++) {
        pointGradients[tid * params.d + k] = 0.0f;
    }

    // Sum all target gradients in this segment
    for (uint e = 0; e < count; e++) {
        uint edgeIdx = start + e;
        for (uint k = 0; k < params.d; k++) {
            pointGradients[tid * params.d + k] += targetGradients[edgeIdx * params.d + k];
        }
    }
}
```

#### 3.2 Swift Integration

Add to `UMAPGradientKernel.swift`:

```swift
// Add to init
private let accumulateTargetPipeline: any MTLComputePipelineState

// In init:
guard let accumulateTargetFunc = library.makeFunction(name: "umap_accumulate_target_gradients_kernel") else {
    throw VectorError.shaderNotFound(
        name: "umap_accumulate_target_gradients_kernel. Ensure UMAPGradient.metal is compiled."
    )
}
self.accumulateTargetPipeline = try await device.makeComputePipelineState(function: accumulateTargetFunc)

/// Applies target gradients using GPU atomic accumulation.
///
/// Replaces the CPU-side loop in Phase 2's `applyTargetGradients()`.
public func applyTargetGradientsGPU(
    embedding: any MTLBuffer,
    edges: any MTLBuffer,
    targetGradients: any MTLBuffer,
    edgeCount: Int,
    n: Int,
    d: Int,
    params: UMAPParameters
) async throws {
    // Create temporary accumulation buffer (initialized to zero)
    let device = context.device.rawDevice
    let accumulationSize = n * d * MemoryLayout<Float>.size
    guard let accumulatedGradients = device.makeBuffer(length: accumulationSize, options: .storageModeShared) else {
        throw VectorError.bufferAllocationFailed(size: accumulationSize)
    }

    // Zero initialize
    memset(accumulatedGradients.contents(), 0, accumulationSize)

    // Accumulate target gradients with atomics
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

        encoder.setComputePipelineState(accumulateTargetPipeline)
        encoder.label = "UMAP.accumulateTarget"
        encoder.setBuffer(targetGradients, offset: 0, index: 0)
        encoder.setBuffer(edges, offset: 0, index: 1)
        encoder.setBuffer(accumulatedGradients, offset: 0, index: 2)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 3)
        dispatchLinear(encoder: encoder, pipeline: accumulateTargetPipeline, count: edgeCount)

        encoder.memoryBarrier(scope: .buffers)

        // Apply accumulated gradients to embedding
        encoder.setComputePipelineState(applyGradientPipeline)
        encoder.label = "UMAP.applyTargetGradient"
        encoder.setBuffer(embedding, offset: 0, index: 0)
        encoder.setBuffer(accumulatedGradients, offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<UMAPParamsGPU>.size, index: 2)
        dispatchLinear(encoder: encoder, pipeline: applyGradientPipeline, count: n * d)
    }
}
```

#### 3.3 Update `optimizeEpoch()` to use GPU Target Gradients

Replace the `applyTargetGradients()` call in `optimizeEpoch()`:

```swift
// Replace Step 3 in optimizeEpoch():
// OLD: try await applyTargetGradients(...)
// NEW:
try await applyTargetGradientsGPU(
    embedding: embeddingBuffer,
    edges: edgeBuffer,
    targetGradients: targetGradientsBuffer,  // Need to pass this from computeGradients
    edgeCount: edgeCount,
    n: n,
    d: d,
    params: params
)
```

**Note**: This requires modifying `computeGradients()` to return the `targetGradients` buffer, or computing edge gradients separately.

### 4. VectorProtocol Support

Add type-safe API for VectorProtocol embeddings:

```swift
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

    // Convert to [[Float]] for now (Phase 4 could optimize this)
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

    // Write back - this works for mutable VectorProtocol types
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
```

### 5. Buffer-Based API (Similar to L2DistanceKernel)

Add an `execute()` method for standalone buffer operations:

```swift
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
///   - randomTargets: Pre-generated random target indices [N × negRate]
///   - n: Number of points
///   - d: Embedding dimension
///   - edgeCount: Number of edges
///   - params: UMAP parameters
public func executeEpoch(
    embedding: any MTLBuffer,
    edges: any MTLBuffer,
    segmentStarts: any MTLBuffer,
    segmentCounts: any MTLBuffer,
    randomTargets: any MTLBuffer?,
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

            // Step 3: Apply target gradients (GPU atomic accumulation)
            // ... encode target gradient accumulation
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
```

### 6. Performance Benchmark Tests

Add benchmarks to measure GPU vs CPU speedup:

```swift
// MARK: - Phase 3: Performance Benchmarks

func testOptimizeEpochPerformance() async throws {
    // Test various sizes
    let sizes = [(100, 2), (500, 2), (1000, 2), (2000, 2)]

    for (n, d) in sizes {
        var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 15)
        let params = UMAPParameters.default

        // Warm up
        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

        // Measure
        let start = CACurrentMediaTime()
        let epochs = 10
        for _ in 0..<epochs {
            try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
        }
        let elapsed = CACurrentMediaTime() - start

        let msPerEpoch = (elapsed / Double(epochs)) * 1000
        print("UMAPGradient n=\(n), d=\(d): \(String(format: "%.2f", msPerEpoch)) ms/epoch")

        // Verify performance targets
        // At n=1000, should be < 10ms/epoch on M1/M2
        if n == 1000 {
            XCTAssertLessThan(msPerEpoch, 50, "Epoch should complete in < 50ms for n=1000")
        }
    }
}

func testTargetGradientGPUvsCSPU() async throws {
    // Compare GPU atomic accumulation vs CPU loop
    let n = 1000
    let d = 2
    let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
    let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 15)

    // ... setup buffers ...

    // Measure CPU path (current Phase 2)
    // Measure GPU path (new Phase 3)
    // Compare times and verify correctness matches
}
```

### 7. Integration Example

Add documentation example for SwiftTopics integration:

```swift
/// Example: Integrating UMAPGradientKernel with SwiftTopics
///
/// ```swift
/// class UMAP {
///     private let kernel: UMAPGradientKernel
///     private let context: Metal4Context
///
///     init() async throws {
///         self.context = try await Metal4Context()
///         self.kernel = try await UMAPGradientKernel(context: context)
///     }
///
///     func fit(
///         knnGraph: [(Int, Int, Float)],  // (source, target, similarity)
///         nComponents: Int = 2,
///         nEpochs: Int = 200,
///         learningRate: Float = 1.0,
///         minDist: Float = 0.1
///     ) async throws -> [[Float]] {
///         let n = Set(knnGraph.map { $0.0 }).count
///
///         // Initialize random embedding
///         var embedding = (0..<n).map { _ in
///             (0..<nComponents).map { _ in Float.random(in: -10...10) }
///         }
///
///         // Convert kNN graph to UMAP edges
///         var edges = knnGraph.map { (src, tgt, sim) in
///             UMAPEdge(source: src, target: tgt, weight: sim)
///         }
///         edges = kernel.sortEdgesBySource(edges)
///
///         // Compute a, b parameters from min_dist
///         var params = UMAPParameters.from(minDist: minDist)
///
///         // Optimization loop
///         for epoch in 0..<nEpochs {
///             // Linear learning rate decay
///             params.learningRate = learningRate * (1.0 - Float(epoch) / Float(nEpochs))
///
///             try await kernel.optimizeEpoch(
///                 embedding: &embedding,
///                 edges: edges,
///                 params: params
///             )
///         }
///
///         return embedding
///     }
/// }
/// ```
```

---

## Key Implementation Notes

### 1. Atomic Float Operations

Metal supports `atomic_float` with `atomic_fetch_add_explicit`. This is used for target gradient accumulation:

```metal
#include <metal_atomic>
typedef atomic<float> atomic_float;

atomic_fetch_add_explicit(&buffer[idx], value, memory_order_relaxed);
```

### 2. Memory Order

Use `memory_order_relaxed` for gradient accumulation since:
- Order between different dimensions doesn't matter
- All threads complete before the next kernel reads
- Memory barrier between kernels ensures visibility

### 3. Buffer Reuse

For maximum performance in repeated epochs, the caller should:
1. Pre-allocate all buffers once
2. Pre-compute segments once (edges don't change)
3. Only regenerate random targets each epoch
4. Call `executeEpoch()` with persistent buffers

### 4. Edge Sorting

Edges MUST be sorted by source for segment reduction. The kernel provides:
- `sortEdgesBySource()` - convenience method
- `computeSegments()` - compute segment info from sorted edges

For GPU-optimized target gradients with segment reduction (not atomics), edges would also need to be sorted by target, requiring a second copy of edges sorted differently.

---

## Build & Test Commands

```bash
# Build
swift build

# Run UMAP tests only
swift test --filter UMAPGradientKernelTests

# Run specific Phase 3 test
swift test --filter testOptimizeEpochPerformance
```

---

## Verification Checklist

- [x] `FusibleKernel` protocol conformance added
- [x] `encodeGradients()` method implemented
- [x] `encodeApplyGradients()` method implemented
- [x] `encodeNegativeSampling()` method implemented
- [x] `encodeAccumulateTargetGradients()` method implemented
- [x] GPU target gradient accumulation kernel added to Metal
- [x] `applyTargetGradientsGPU()` implemented
- [x] `optimizeEpoch()` updated to use GPU target gradients
- [x] VectorProtocol overload for `optimizeEpoch()` added
- [x] VectorProtocol overload for `computeLoss()` added
- [x] `executeEpoch()` buffer-based API added
- [x] Performance benchmark test added
- [x] GPU target gradient test (correctness vs CPU)
- [x] All tests pass (20 tests: 9 Phase 1 + 6 Phase 2 + 5 Phase 3)

**Phase 3 completed 2026-01-05. All 20 tests passing.**

### Performance Results

| Corpus Size | ms/epoch |
|-------------|----------|
| n=100 | 1.42 ms |
| n=500 | 2.35 ms |
| n=1000 | 4.89 ms |

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| FusibleKernel conformance | ~10 |
| `encodeGradients()` | ~50 |
| `encodeApplyGradients()` | ~25 |
| `encodeNegativeSampling()` | ~25 |
| Metal target accumulation kernel | ~35 |
| `applyTargetGradientsGPU()` | ~45 |
| VectorProtocol overloads | ~60 |
| `executeEpoch()` buffer API | ~60 |
| Performance benchmarks | ~80 |
| Integration example docs | ~40 |
| **Total** | ~430 |

---

## Future Considerations (Phase 4+)

- **GPU Random Number Generation**: Replace CPU `generateRandomTargets()` with GPU RNG
- **Double Buffering**: Overlap CPU buffer prep with GPU execution
- **Adaptive Negative Sampling**: Sample more negatives for close points
- **Early Termination**: Stop when loss plateaus
- **Spectral Initialization**: Better starting positions

---

## Reference Files

```
VectorAccelerate/
├── Sources/VectorAccelerate/
│   ├── Metal/Shaders/
│   │   └── UMAPGradient.metal          # MODIFY (add target accumulation kernel)
│   └── Kernels/Metal4/
│       ├── KernelProtocol.swift        # Reference for FusibleKernel
│       ├── L2DistanceKernel.swift      # Reference for encode() pattern
│       └── UMAPGradientKernel.swift    # MODIFY (add Phase 3 methods)
└── Tests/VectorAccelerateTests/
    └── UMAPGradientKernelTests.swift   # MODIFY (add Phase 3 tests)
```

---

## Contact

- Phase 1 handoff: `docs/handoffs/UMAPGradientKernel-Phase1-Handoff.md`
- Phase 2 handoff: `docs/handoffs/UMAPGradientKernel-Phase2-Handoff.md`
- Original spec: `docs/kernel-specs/03-UMAPGradientKernel.md`
- Reference implementations: `L2DistanceKernel.swift`, `MutualReachabilityKernel.swift`

Phase 3 ready for implementation 2026-01-05.
