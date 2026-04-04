# BoruvkaMSTKernel Phase 3 Handoff

## Overview

This document provides context for implementing Phase 3 of the BoruvkaMSTKernel - FusibleKernel conformance, VectorProtocol support, HDBSCANDistanceModule integration, and performance benchmarks.

## Phase 1 & 2 Recap (Completed)

| Phase | Deliverables | Status |
|-------|--------------|--------|
| Phase 1 | Core Borůvka's algorithm, 3 GPU kernels, hybrid GPU/CPU merge | ✅ Complete |
| Phase 2 | Dimension-optimized kernels (384/512/768/1536), CPU Prim's reference, correctness tests | ✅ Complete |

### Current Test Coverage

19 tests passing:
- Edge count, connectivity, duplicate detection
- Edge cases (empty, single, two points)
- Iteration count, weight sanity
- MTLBuffer API, common dimensions
- Weight matches Prim's (correctness verification)
- Dimension-optimized kernel correctness (384, 512, 768)
- DimensionOptimizedKernel protocol conformance

### Current Architecture

```
For each iteration (O(log N) total):
  1. GPU: boruvka_find_min_edge_*_kernel  - Find min edge per point (dimension-optimized)
  2. GPU: boruvka_component_reduce_kernel - Reduce to per-component min
  3. GPU: boruvka_merge_kernel           - Collect candidate edges
  4. CPU: Union-Find merge               - Deduplicate + merge components
```

---

## Phase 3 Scope

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `FusibleKernel` conformance | Enable fusion with MutualReachabilityKernel | HIGH |
| `VectorProtocol` support | Ergonomic API for VectorCore types | HIGH |
| `HDBSCANDistanceModule` | High-level module combining kernels | HIGH |
| Performance benchmarks | GPU vs CPU comparison | MEDIUM |

### Expected LOC: ~400

---

## Task 1: FusibleKernel Conformance

The `FusibleKernel` protocol allows kernels to share command encoders, reducing submission overhead.

### File: `BoruvkaMSTKernel.swift`

Add protocol conformance:

```swift
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension BoruvkaMSTKernel: FusibleKernel {
    /// Types of kernels this can be fused with
    public var fusibleWith: [String] {
        ["MutualReachabilityKernel", "FusedL2TopKKernel"]
    }

    /// Whether a barrier is required before this kernel's output is read
    public var requiresBarrierAfter: Bool {
        true  // MST edges buffer must be synchronized before CPU reads
    }
}
```

### Add Encode API

Add an `encode` method for fusion scenarios where the caller manages the command buffer:

```swift
/// Encode a single Borůvka iteration into an existing encoder.
///
/// This is useful for fusing with other operations. The caller is responsible
/// for managing the command buffer lifecycle and memory barriers.
///
/// - Note: This encodes ONE iteration only. For full MST, use `computeMST`.
///
/// - Parameters:
///   - encoder: The compute command encoder
///   - embeddings: N×D embedding buffer
///   - coreDistances: N core distances buffer
///   - componentIds: N component IDs buffer (mutable)
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
    componentIds: any MTLBuffer,
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
    encoder.setBuffer(componentIds, offset: 0, index: 2)
    encoder.setBuffer(workBuffers.pointMinWeight, offset: 0, index: 3)
    encoder.setBuffer(workBuffers.pointMinTarget, offset: 0, index: 4)
    encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 5)
    dispatchLinear(encoder: encoder, pipeline: pipeline, count: n)

    encoder.memoryBarrier(scope: .buffers)

    // Step 2: Reduce to per-component minimum
    encoder.setComputePipelineState(componentReducePipeline)
    encoder.label = "Boruvka.reduce[\(iteration)]"
    encoder.setBuffer(componentIds, offset: 0, index: 0)
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
    encoder.setBuffer(componentIds, offset: 0, index: 0)
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

/// Work buffers for Borůvka iterations.
public struct BoruvkaWorkBuffers: Sendable {
    public let pointMinWeight: any MTLBuffer
    public let pointMinTarget: any MTLBuffer
    public let componentMinWeight: any MTLBuffer
    public let componentMinSource: any MTLBuffer
    public let componentMinTarget: any MTLBuffer
    public let candidateEdges: any MTLBuffer
    public let edgeCount: any MTLBuffer
}

/// Create reusable work buffers for a given problem size.
public func createWorkBuffers(n: Int) throws -> BoruvkaWorkBuffers {
    let device = context.device.rawDevice

    guard let pointMinWeight = device.makeBuffer(
        length: n * MemoryLayout<Float>.size,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: n * MemoryLayout<Float>.size)
    }

    // ... (create all other buffers similarly)

    return BoruvkaWorkBuffers(
        pointMinWeight: pointMinWeight,
        pointMinTarget: pointMinTarget,
        componentMinWeight: componentMinWeight,
        componentMinSource: componentMinSource,
        componentMinTarget: componentMinTarget,
        candidateEdges: candidateEdges,
        edgeCount: edgeCount
    )
}
```

---

## Task 2: VectorProtocol Support

Add convenience methods that accept VectorCore types directly.

### File: `BoruvkaMSTKernel.swift`

```swift
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

    let embedBuffer = try createBufferFromVectors(embeddings, device: device)
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

// MARK: - Private Helpers

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
```

---

## Task 3: HDBSCANDistanceModule

Create a high-level module that combines FusedL2TopKKernel (for core distances) with BoruvkaMSTKernel.

### File: `Sources/VectorAccelerate/Modules/HDBSCANDistanceModule.swift`

```swift
//
//  HDBSCANDistanceModule.swift
//  VectorAccelerate
//
//  High-level module for HDBSCAN distance computations.
//  Combines core distance computation with MST construction.
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Result of HDBSCAN distance computation.
public struct HDBSCANDistanceResult: Sendable {
    /// Core distances for each point (k-th nearest neighbor distance)
    public let coreDistances: [Float]

    /// Minimum Spanning Tree over mutual reachability distances
    public let mst: MSTResult

    /// Number of points processed
    public let pointCount: Int

    /// k value used for core distance computation
    public let minSamples: Int
}

/// High-level module for HDBSCAN clustering distance computations.
///
/// This module provides a simplified API for computing the mutual reachability
/// MST required by HDBSCAN clustering. It combines:
/// 1. Core distance computation (k-th nearest neighbor distance)
/// 2. MST construction over mutual reachability distances
///
/// ## Usage
///
/// ```swift
/// let module = try await HDBSCANDistanceModule(context: context)
/// let result = try await module.computeMST(
///     embeddings: documentEmbeddings,
///     minSamples: 5
/// )
/// // result.mst contains the MST for cluster extraction
/// ```
///
/// ## Performance
///
/// | Corpus Size | Expected Time |
/// |-------------|---------------|
/// | 500 docs | ~50ms |
/// | 1,000 docs | ~150ms |
/// | 5,000 docs | ~2s |
///
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class HDBSCANDistanceModule: @unchecked Sendable {

    // MARK: - Properties

    public let context: Metal4Context

    private let topKKernel: FusedL2TopKKernel
    private let mstKernel: BoruvkaMSTKernel

    // MARK: - Initialization

    /// Create an HDBSCAN distance module.
    ///
    /// - Parameter context: Metal 4 context
    public init(context: Metal4Context) async throws {
        self.context = context
        self.topKKernel = try await FusedL2TopKKernel(context: context)
        self.mstKernel = try await BoruvkaMSTKernel(context: context)
    }

    // MARK: - Public API

    /// Compute MST over mutual reachability distances.
    ///
    /// This is the primary entry point for HDBSCAN distance computation.
    /// It computes core distances using k-NN search, then builds the MST.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix (row-major)
    ///   - minSamples: k value for core distance (default: 5)
    /// - Returns: Core distances and MST result
    public func computeMST(
        embeddings: [[Float]],
        minSamples: Int = 5
    ) async throws -> HDBSCANDistanceResult {
        let n = embeddings.count
        guard n > 0 else {
            return HDBSCANDistanceResult(
                coreDistances: [],
                mst: MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0),
                pointCount: 0,
                minSamples: minSamples
            )
        }

        // Step 1: Compute core distances (k-th nearest neighbor distance)
        let coreDistances = try await computeCoreDistances(
            embeddings: embeddings,
            k: minSamples
        )

        // Step 2: Compute MST over mutual reachability
        let mst = try await mstKernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        return HDBSCANDistanceResult(
            coreDistances: coreDistances,
            mst: mst,
            pointCount: n,
            minSamples: minSamples
        )
    }

    /// Compute MST from VectorProtocol embeddings.
    public func computeMST<V: VectorProtocol>(
        embeddings: [V],
        minSamples: Int = 5
    ) async throws -> HDBSCANDistanceResult where V.Scalar == Float {
        // Convert to [[Float]] for now; optimize later with direct buffer API
        let floatEmbeddings = embeddings.map { vector -> [Float] in
            var result = [Float](repeating: 0, count: vector.count)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<ptr.count {
                    result[i] = ptr[i]
                }
            }
            return result
        }
        return try await computeMST(embeddings: floatEmbeddings, minSamples: minSamples)
    }

    /// Compute MST with pre-computed core distances.
    ///
    /// Use this when core distances are already available (e.g., from a previous
    /// FusedL2TopKKernel call).
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix
    ///   - coreDistances: Pre-computed core distances
    /// - Returns: MST result
    public func computeMSTWithCoreDistances(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> MSTResult {
        return try await mstKernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )
    }

    // MARK: - Private Helpers

    /// Compute core distances using k-NN search.
    ///
    /// The core distance for point p is the distance to its k-th nearest neighbor.
    private func computeCoreDistances(
        embeddings: [[Float]],
        k: Int
    ) async throws -> [Float] {
        let n = embeddings.count
        guard n > 1 else {
            return [Float](repeating: 0, count: n)
        }

        // Use k+1 because the nearest neighbor of a point is itself (distance 0)
        let effectiveK = min(k + 1, n)

        // Compute all-pairs k-NN (query = database = embeddings)
        let results = try await topKKernel.search(
            queries: embeddings,
            database: embeddings,
            k: effectiveK,
            metric: .l2
        )

        // Extract k-th nearest neighbor distance (index k, since index 0 is self)
        var coreDistances = [Float](repeating: 0, count: n)
        for i in 0..<n {
            if results.distances[i].count >= effectiveK {
                // The k-th index (0-indexed) after self is at position min(k, count-1)
                let kIndex = min(k, results.distances[i].count - 1)
                coreDistances[i] = results.distances[i][kIndex]
            }
        }

        return coreDistances
    }
}
```

---

## Task 4: Performance Benchmarks

Add benchmark tests comparing GPU MST to CPU Prim's implementation.

### File: `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift`

Add to the test class:

```swift
// MARK: - Phase 3: Performance Benchmarks

func testPerformanceComparison_100Points() async throws {
    let n = 100
    let d = 384
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    // Warm up
    _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

    // Measure GPU
    let gpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<10 {
        _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    }
    let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0

    // Measure CPU
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<10 {
        _ = Metal4KernelTestHelpers.cpuPrimsMST(embeddings: embeddings, coreDistances: coreDistances)
    }
    let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0

    let speedup = cpuTime / gpuTime
    print("Performance (n=\(n), d=\(d)): GPU=\(gpuTime*1000)ms, CPU=\(cpuTime*1000)ms, Speedup=\(speedup)x")

    // GPU should be faster for this size
    XCTAssertGreaterThan(speedup, 1.0, "GPU should be faster than CPU for n=\(n)")
}

func testPerformanceComparison_500Points() async throws {
    let n = 500
    let d = 384
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    // Warm up
    _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

    // Measure GPU (fewer iterations due to longer runtime)
    let gpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<3 {
        _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    }
    let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 3.0

    // Measure CPU
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<3 {
        _ = Metal4KernelTestHelpers.cpuPrimsMST(embeddings: embeddings, coreDistances: coreDistances)
    }
    let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 3.0

    let speedup = cpuTime / gpuTime
    print("Performance (n=\(n), d=\(d)): GPU=\(gpuTime*1000)ms, CPU=\(cpuTime*1000)ms, Speedup=\(speedup)x")

    // Should see significant speedup at this scale
    XCTAssertGreaterThan(speedup, 5.0, "GPU should be >5x faster than CPU for n=\(n)")
}
```

---

## Task 5: Tests for New Functionality

### VectorProtocol API Tests

```swift
// MARK: - Phase 3: VectorProtocol API Tests

func testVectorProtocolAPI() async throws {
    let n = 20
    let d = 16

    // Create DynamicVector embeddings
    let embeddings: [DynamicVector] = (0..<n).map { _ in
        DynamicVector(Metal4KernelTestHelpers.randomVectors(count: 1, dimension: d)[0])
    }
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    let result = try await kernel.computeMST(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    XCTAssertEqual(result.edges.count, n - 1)
    XCTAssertTrue(Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n))
}
```

### HDBSCANDistanceModule Tests

```swift
// MARK: - HDBSCANDistanceModule Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class HDBSCANDistanceModuleTests: XCTestCase {

    var context: Metal4Context!
    var module: HDBSCANDistanceModule!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        module = try await HDBSCANDistanceModule(context: context)
    }

    func testBasicMSTComputation() async throws {
        let n = 50
        let d = 32
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        XCTAssertEqual(result.pointCount, n)
        XCTAssertEqual(result.minSamples, 5)
        XCTAssertEqual(result.coreDistances.count, n)
        XCTAssertEqual(result.mst.edges.count, n - 1)
    }

    func testCoreDistancesPositive() async throws {
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: 30, dimension: 16)

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        for cd in result.coreDistances {
            XCTAssertGreaterThanOrEqual(cd, 0, "Core distances should be non-negative")
        }
    }

    func testPrecomputedCoreDistances() async throws {
        let n = 30
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let mst = try await module.computeMSTWithCoreDistances(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(mst.edges.count, n - 1)
    }
}
```

---

## Verification Checklist

### FusibleKernel Conformance
- [x] `fusibleWith` property returns correct kernel names
- [x] `requiresBarrierAfter` returns true
- [x] `encodeIteration` method implemented
- [x] `BoruvkaWorkBuffers` struct created
- [x] `createWorkBuffers` method implemented

### VectorProtocol Support
- [x] `computeMST<V: VectorProtocol>` implemented
- [x] `computeMST<D: StaticDimension>` implemented
- [x] `createBufferFromVectors` helper implemented
- [x] Tests pass for DynamicVector input

### HDBSCANDistanceModule
- [x] Module created in `Sources/VectorAccelerate/Modules/`
- [x] `HDBSCANDistanceResult` struct defined
- [x] `computeMST(embeddings:minSamples:)` implemented
- [x] `computeMSTWithCoreDistances` implemented
- [x] Core distance extraction from k-NN correct
- [x] Tests pass

### Performance Benchmarks
- [x] 100-point benchmark shows GPU advantage (40x speedup achieved)
- [x] 500-point benchmark shows >5x speedup (344x speedup achieved!)
- [x] Results printed for documentation

---

## Build & Test Commands

```bash
# Build
swift build

# Run all BoruvkaMST tests
swift test --filter BoruvkaMSTKernelTests

# Run HDBSCANDistanceModule tests
swift test --filter HDBSCANDistanceModuleTests

# Run specific test
swift test --filter testPerformanceComparison_500Points
```

---

## Key Implementation Notes

### 1. FusibleKernel Pattern

Follow the pattern from `L2DistanceKernel`:
- `encode` method adds work to existing encoder
- Caller manages command buffer lifecycle
- Memory barriers inserted between dependent operations

### 2. VectorProtocol Zero-Copy

Use `withUnsafeBufferPointer` to avoid intermediate allocations:

```swift
vector.withUnsafeBufferPointer { ptr in
    destination.advanced(by: offset).update(from: ptr.baseAddress!, count: ptr.count)
}
```

### 3. Core Distance from k-NN

The k-th nearest neighbor distance excludes self:
- Query point i has distance 0 to itself at index 0
- Core distance is at index k (the k-th neighbor after self)

### 4. HDBSCANDistanceModule Simplicity

Keep the module simple - it's a composition layer:
- Don't duplicate logic from underlying kernels
- Provide convenient defaults (minSamples=5)
- Allow escape hatches (pre-computed core distances)

---

## Reference Files

- Current implementation: `Sources/VectorAccelerate/Kernels/Metal4/BoruvkaMSTKernel.swift`
- FusibleKernel pattern: `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift`
- VectorProtocol pattern: `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift`
- FusedL2TopKKernel: `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift`
- Tests: `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift`

---

## Phase 4 Preview

After Phase 3:
- Incremental MST updates (add new points without recomputing)
- Multi-GPU support for very large datasets
- Memory-efficient streaming for datasets exceeding GPU memory

---

## Status: COMPLETED 2026-01-05

### Implementation Summary
- **Total Tests**: 32 passing (25 BoruvkaMSTKernel + 7 HDBSCANDistanceModule)
- **Performance**: 40x speedup at n=100, 344x speedup at n=500
- **Files Modified**: `BoruvkaMSTKernel.swift`, `BoruvkaMSTKernelTests.swift`
- **Files Created**: `HDBSCANDistanceModule.swift`
