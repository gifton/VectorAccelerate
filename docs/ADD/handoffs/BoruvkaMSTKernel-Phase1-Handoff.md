# BoruvkaMSTKernel Phase 1 Handoff

## Overview

This document provides context for implementing Phase 1 of the BoruvkaMSTKernel - the core foundation for GPU-accelerated Minimum Spanning Tree computation using Borůvka's algorithm.

## Background

### Why Borůvka's Algorithm?

| Algorithm | GPU Suitability | Complexity | Parallelism |
|-----------|-----------------|------------|-------------|
| **Borůvka's** | Excellent | O(E log V) | O(V) per iteration |
| Prim's | Poor | O(E log V) | Sequential edge selection |
| Kruskal's | Medium | O(E log E) | Requires global sorting |

Borůvka's is GPU-friendly because:
1. Each iteration processes ALL components in parallel
2. Each component independently finds its minimum outgoing edge
3. Only ~log(N) iterations needed
4. Natural batch structure: O(N) work per iteration

### Algorithm Steps

```
Input: N points with mutual reachability distances
Output: MST with N-1 edges

1. Initialize: Each point is its own component (N components)
2. Repeat until 1 component remains:
   a. For each component, find the minimum-weight edge to another component (parallel)
   b. Add all found edges to MST (handles duplicates)
   c. Merge components connected by new edges
3. Return MST edges
```

**Iterations**: ~log₂(N)
- N=100: ~7 iterations
- N=1000: ~10 iterations
- N=5000: ~13 iterations

---

## Phase 1 Scope

Implement the core algorithm with generic kernels (no dimension optimization).

### Deliverables

| Component | Description |
|-----------|-------------|
| `BoruvkaMST.metal` | 3 GPU kernels + parameter structs |
| `BoruvkaMSTKernel.swift` | Swift API with iteration loop |
| `BoruvkaMSTKernelTests.swift` | 6-8 correctness tests |

### Expected LOC: ~400

---

## Implementation Tasks

### 1. Create Metal Shader File

**File**: `Sources/VectorAccelerate/Metal/Shaders/BoruvkaMST.metal`

```metal
//
//  BoruvkaMST.metal
//  VectorAccelerate
//
//  GPU kernels for Borůvka's MST algorithm.
//
//  Phase 1: Core kernels (generic dimension)
//
//  Kernels:
//  - boruvka_find_min_edge_kernel: Find minimum outgoing edge per point
//  - boruvka_component_reduce_kernel: Reduce to per-component minimum
//  - boruvka_merge_kernel: Add edges to MST and merge components

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameter Structures

struct BoruvkaParams {
    uint n;              // Number of points
    uint d;              // Embedding dimension
    uint iteration;      // Current iteration (for debugging)
    uint _padding;       // Alignment padding
};

struct MSTEdge {
    uint source;
    uint target;
    float weight;
};

// MARK: - Kernel 1: Find Minimum Outgoing Edge

/// Each thread handles one point, finding the minimum-weight edge to a different component.
///
/// Computes mutual reachability on-the-fly: max(core_i, core_j, euclidean(i, j))
kernel void boruvka_find_min_edge_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, D]
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current components
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;  // Self means no valid edge found

    // Check all other points
    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;  // Same component, skip

        // Compute L2 distance
        float distSq = 0.0f;
        for (uint k = 0; k < params.d; k++) {
            float diff = embeddings[tid * params.d + k] - embeddings[j * params.d + k];
            distSq = fma(diff, diff, distSq);
        }
        float dist = sqrt(distSq);

        // Mutual reachability = max(core_i, core_j, dist)
        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}

// MARK: - Kernel 2: Component-Level Reduce

/// Reduce per-point minimums to per-component minimums.
/// Only the "representative" of each component (smallest index) performs the reduction.
kernel void boruvka_component_reduce_kernel(
    device const uint* componentIds         [[buffer(0)]],  // [N]
    device const float* pointMinWeight      [[buffer(1)]],  // [N] per-point min
    device const uint* pointMinTarget       [[buffer(2)]],  // [N] per-point target
    device float* componentMinWeight        [[buffer(3)]],  // [N] per-component min
    device uint* componentMinSource         [[buffer(4)]],  // [N] source of min edge
    device uint* componentMinTarget         [[buffer(5)]],  // [N] target of min edge
    constant BoruvkaParams& params          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];

    // Only the "representative" of each component performs the reduction
    // Representative = smallest index in component
    bool isRepresentative = true;
    for (uint i = 0; i < tid; i++) {
        if (componentIds[i] == myComponent) {
            isRepresentative = false;
            break;
        }
    }

    if (!isRepresentative) {
        componentMinWeight[tid] = INFINITY;
        return;
    }

    // Find minimum across all points in this component
    float bestWeight = INFINITY;
    uint bestSource = tid;
    uint bestTarget = tid;

    for (uint i = 0; i < params.n; i++) {
        if (componentIds[i] != myComponent) continue;
        if (pointMinWeight[i] < bestWeight) {
            bestWeight = pointMinWeight[i];
            bestSource = i;
            bestTarget = pointMinTarget[i];
        }
    }

    componentMinWeight[tid] = bestWeight;
    componentMinSource[tid] = bestSource;
    componentMinTarget[tid] = bestTarget;
}

// MARK: - Kernel 3: Merge Components

/// Add selected edges to MST and merge components.
/// Uses atomic operations for edge count.
kernel void boruvka_merge_kernel(
    device uint* componentIds               [[buffer(0)]],  // [N] in/out
    device const float* componentMinWeight  [[buffer(1)]],  // [N]
    device const uint* componentMinSource   [[buffer(2)]],  // [N]
    device const uint* componentMinTarget   [[buffer(3)]],  // [N]
    device MSTEdge* mstEdges                [[buffer(4)]],  // [N-1] output edges
    device atomic_uint* edgeCount           [[buffer(5)]],  // [1] current edge count
    constant BoruvkaParams& params          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    // Only representatives add edges
    if (componentMinWeight[tid] == INFINITY) return;

    uint source = componentMinSource[tid];
    uint target = componentMinTarget[tid];
    float weight = componentMinWeight[tid];

    // Avoid duplicate edges: only add if source component < target component
    uint sourceComp = componentIds[source];
    uint targetComp = componentIds[target];
    if (sourceComp >= targetComp) return;

    // Add edge to MST
    uint idx = atomic_fetch_add_explicit(edgeCount, 1, memory_order_relaxed);
    mstEdges[idx].source = source;
    mstEdges[idx].target = target;
    mstEdges[idx].weight = weight;

    // Merge: all points in target component join source component
    // Use the smaller component ID as the new ID
    uint newId = min(sourceComp, targetComp);
    uint oldId = max(sourceComp, targetComp);

    for (uint i = 0; i < params.n; i++) {
        if (componentIds[i] == oldId) {
            componentIds[i] = newId;
        }
    }
}
```

### 2. Register Shader in KernelContext

**File**: `Sources/VectorAccelerate/Core/KernelContext.swift`

Add `"BoruvkaMST"` to the shader compile list (around line ~137 where other shaders are registered).

### 3. Create Swift Kernel

**File**: `Sources/VectorAccelerate/Kernels/Metal4/BoruvkaMSTKernel.swift`

```swift
//
//  BoruvkaMSTKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for computing Minimum Spanning Tree using Borůvka's algorithm.
//
//  Phase 1: Core Foundation
//
//  Features:
//  - GPU-parallel MST computation
//  - O(N) space (no N² distance matrix)
//  - ~log(N) iterations

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Result Type

/// Result of MST computation.
public struct MSTResult: Sendable {
    /// Edges in the MST as (source, target, weight) tuples.
    /// Always contains exactly n-1 edges for n points.
    public let edges: [(source: Int, target: Int, weight: Float)]

    /// Total weight of the MST (sum of all edge weights).
    public let totalWeight: Float

    /// Number of Borůvka iterations performed.
    public let iterations: Int

    /// Number of points in the original dataset.
    public let pointCount: Int
}

// MARK: - Parameters

/// Parameters for Borůvka's algorithm kernels.
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
struct MSTEdgeGPU {
    var source: UInt32
    var target: UInt32
    var weight: Float
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for computing Minimum Spanning Tree using Borůvka's algorithm.
///
/// This kernel is designed for HDBSCAN clustering, computing the MST over
/// mutual reachability distances. Distances are computed on-the-fly to
/// avoid O(N²) memory usage.
///
/// ## Algorithm
///
/// Borůvka's algorithm runs in O(log N) iterations, with each iteration:
/// 1. Finding the minimum outgoing edge for each component (parallel)
/// 2. Reducing to per-component minimum edges
/// 3. Adding selected edges to the MST and merging components
///
/// ## Complexity
///
/// - Time: O(N² × D × log N) - dominated by distance computation per iteration
/// - Space: O(N) - no N² distance matrix stored
/// - Iterations: ~log₂(N)
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
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class BoruvkaMSTKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "BoruvkaMSTKernel"

    // MARK: - Pipelines

    private let findMinPipeline: any MTLComputePipelineState
    private let componentReducePipeline: any MTLComputePipelineState
    private let mergePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Borůvka's MST kernel.
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
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines are created in init, this is a no-op
    }

    // MARK: - Public API

    /// Computes MST from embeddings and core distances.
    ///
    /// Mutual reachability distances are computed on-the-fly within the kernel
    /// to avoid O(N²) memory usage.
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N×D embedding matrix (row-major Float32).
    ///   - coreDistances: Buffer containing N core distances (Float32).
    ///   - n: Number of points.
    ///   - d: Embedding dimension.
    /// - Returns: MST result with edges, total weight, and iteration count.
    /// - Throws: `VectorError` if execution fails.
    ///
    /// - Complexity: O(N² × D × log N) time, O(N) space
    public func computeMST(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MSTResult {
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

        guard let mstEdges = device.makeBuffer(
            length: (n - 1) * MemoryLayout<MSTEdgeGPU>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: (n - 1) * MemoryLayout<MSTEdgeGPU>.size)
        }
        mstEdges.label = "Boruvka.mstEdges"

        guard let edgeCount = device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<UInt32>.size)
        }
        edgeCount.label = "Boruvka.edgeCount"

        // Initialize component IDs (each point is its own component)
        initializeComponentIds(componentIds, n: n)

        // Initialize edge count to 0
        let edgeCountPtr = edgeCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        edgeCountPtr.pointee = 0

        var iterations = 0
        let maxIterations = Int(ceil(log2(Double(n)))) + 2  // Safety margin

        while iterations < maxIterations {
            let currentEdgeCount = edgeCountPtr.pointee

            // Check if we have all N-1 edges
            if currentEdgeCount >= n - 1 {
                break
            }

            try await runIteration(
                embeddings: embeddings,
                coreDistances: coreDistances,
                componentIds: componentIds,
                pointMinWeight: pointMinWeight,
                pointMinTarget: pointMinTarget,
                componentMinWeight: componentMinWeight,
                componentMinSource: componentMinSource,
                componentMinTarget: componentMinTarget,
                mstEdges: mstEdges,
                edgeCount: edgeCount,
                n: n,
                d: d,
                iteration: iterations
            )

            iterations += 1
        }

        // Read back results
        return readResults(
            mstEdges: mstEdges,
            edgeCount: edgeCount,
            n: n,
            iterations: iterations
        )
    }

    // MARK: - Convenience Methods (Swift Arrays)

    /// Computes MST from Swift arrays.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix as nested arrays.
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

    // MARK: - Private Methods

    private func runIteration(
        embeddings: any MTLBuffer,
        coreDistances: any MTLBuffer,
        componentIds: any MTLBuffer,
        pointMinWeight: any MTLBuffer,
        pointMinTarget: any MTLBuffer,
        componentMinWeight: any MTLBuffer,
        componentMinSource: any MTLBuffer,
        componentMinTarget: any MTLBuffer,
        mstEdges: any MTLBuffer,
        edgeCount: any MTLBuffer,
        n: Int,
        d: Int,
        iteration: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            var params = BoruvkaParams(n: n, d: d, iteration: iteration)

            // Step 1: Find minimum outgoing edge per point
            encoder.setComputePipelineState(findMinPipeline)
            encoder.label = "Boruvka.findMin[\(iteration)]"
            encoder.setBuffer(embeddings, offset: 0, index: 0)
            encoder.setBuffer(coreDistances, offset: 0, index: 1)
            encoder.setBuffer(componentIds, offset: 0, index: 2)
            encoder.setBuffer(pointMinWeight, offset: 0, index: 3)
            encoder.setBuffer(pointMinTarget, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 5)
            dispatchLinear(encoder: encoder, pipeline: findMinPipeline, count: n)

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

            // Step 3: Add edges and merge components
            encoder.setComputePipelineState(mergePipeline)
            encoder.label = "Boruvka.merge[\(iteration)]"
            encoder.setBuffer(componentIds, offset: 0, index: 0)
            encoder.setBuffer(componentMinWeight, offset: 0, index: 1)
            encoder.setBuffer(componentMinSource, offset: 0, index: 2)
            encoder.setBuffer(componentMinTarget, offset: 0, index: 3)
            encoder.setBuffer(mstEdges, offset: 0, index: 4)
            encoder.setBuffer(edgeCount, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
            dispatchLinear(encoder: encoder, pipeline: mergePipeline, count: n)
        }
    }

    private func initializeComponentIds(_ buffer: any MTLBuffer, n: Int) {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            ptr[i] = UInt32(i)
        }
    }

    private func readResults(
        mstEdges: any MTLBuffer,
        edgeCount: any MTLBuffer,
        n: Int,
        iterations: Int
    ) -> MSTResult {
        let count = Int(edgeCount.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
        let edgePtr = mstEdges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: count)

        var edges: [(source: Int, target: Int, weight: Float)] = []
        edges.reserveCapacity(count)
        var totalWeight: Float = 0

        for i in 0..<count {
            let edge = edgePtr[i]
            edges.append((source: Int(edge.source), target: Int(edge.target), weight: edge.weight))
            totalWeight += edge.weight
        }

        return MSTResult(
            edges: edges,
            totalWeight: totalWeight,
            iterations: iterations,
            pointCount: n
        )
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
}
```

### 4. Create Test File

**File**: `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift`

```swift
// VectorAccelerate: BoruvkaMSTKernel Tests
//
// Tests for GPU-accelerated MST computation using Borůvka's algorithm.
//
// Phase 1: Core Foundation Tests
// - Edge count (N-1 edges)
// - Connectivity
// - No duplicate edges
// - Basic correctness
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {
    /// Verify MST is connected using Union-Find.
    static func verifyConnected(edges: [(source: Int, target: Int, weight: Float)], n: Int) -> Bool {
        guard n > 0 else { return true }
        if n == 1 { return edges.isEmpty }

        // Simple Union-Find
        var parent = Array(0..<n)

        func find(_ x: Int) -> Int {
            if parent[x] != x {
                parent[x] = find(parent[x])
            }
            return parent[x]
        }

        func union(_ x: Int, _ y: Int) {
            let px = find(x)
            let py = find(y)
            if px != py {
                parent[px] = py
            }
        }

        for edge in edges {
            union(edge.source, edge.target)
        }

        // Check all points have same root
        let root = find(0)
        for i in 1..<n {
            if find(i) != root {
                return false
            }
        }
        return true
    }

    /// Check for duplicate edges in MST.
    static func hasDuplicateEdges(edges: [(source: Int, target: Int, weight: Float)]) -> Bool {
        var seen = Set<String>()
        for edge in edges {
            let key = "\(min(edge.source, edge.target))-\(max(edge.source, edge.target))"
            if seen.contains(key) {
                return true
            }
            seen.insert(key)
        }
        return false
    }
}

// MARK: - BoruvkaMSTKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class BoruvkaMSTKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: BoruvkaMSTKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await BoruvkaMSTKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Edge Count Tests

    func testEdgeCountSmall() async throws {
        let n = 10
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, n - 1, "MST should have exactly N-1 edges")
        XCTAssertEqual(result.pointCount, n)
    }

    func testEdgeCount100() async throws {
        let n = 100
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, n - 1, "MST should have exactly N-1 edges")
    }

    // MARK: - Connectivity Tests

    func testConnectivity() async throws {
        let n = 50
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n),
            "MST should connect all points"
        )
    }

    // MARK: - Duplicate Edge Tests

    func testNoDuplicateEdges() async throws {
        let n = 50
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertFalse(
            Metal4KernelTestHelpers.hasDuplicateEdges(edges: result.edges),
            "MST should not have duplicate edges"
        )
    }

    // MARK: - Edge Cases

    func testSinglePoint() async throws {
        let embeddings: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]
        let coreDistances: [Float] = [0.5]

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 0, "Single point should have no edges")
        XCTAssertEqual(result.totalWeight, 0)
        XCTAssertEqual(result.pointCount, 1)
    }

    func testTwoPoints() async throws {
        let embeddings: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0]  // Distance = 5
        ]
        let coreDistances: [Float] = [0.5, 0.5]

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 1, "Two points should have exactly 1 edge")
        XCTAssertEqual(result.pointCount, 2)

        // Weight should be mutual_reach = max(0.5, 0.5, 5.0) = 5.0
        XCTAssertEqual(result.totalWeight, 5.0, accuracy: 1e-4)
    }

    func testEmptyInput() async throws {
        let embeddings: [[Float]] = []
        let coreDistances: [Float] = []

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 0)
        XCTAssertEqual(result.totalWeight, 0)
        XCTAssertEqual(result.pointCount, 0)
    }

    // MARK: - Iteration Count Tests

    func testIterationCount() async throws {
        let n = 100
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // Expected: ~log2(100) ≈ 7 iterations
        let expectedMax = Int(ceil(log2(Double(n)))) + 2
        XCTAssertLessThanOrEqual(
            result.iterations,
            expectedMax,
            "Should complete in O(log N) iterations"
        )
        XCTAssertGreaterThan(result.iterations, 0, "Should take at least 1 iteration")
    }

    // MARK: - Weight Sanity Tests

    func testTotalWeightPositive() async throws {
        let n = 20
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertGreaterThan(result.totalWeight, 0, "Total weight should be positive")

        // Verify individual edges have positive weights
        for edge in result.edges {
            XCTAssertGreaterThan(edge.weight, 0, "Edge weight should be positive")
        }
    }
}
```

---

## Key Implementation Notes

### 1. Atomic Operations

The merge kernel uses `atomic_fetch_add_explicit` for thread-safe edge count increment:

```metal
uint idx = atomic_fetch_add_explicit(edgeCount, 1, memory_order_relaxed);
mstEdges[idx] = ...;
```

### 2. Component ID Management

- Initialize: Each point `i` has `componentId[i] = i`
- After merge: All points in component B get reassigned to component A
- Representative: Smallest index in each component

### 3. Duplicate Edge Prevention

The merge kernel only adds edge if `sourceComp < targetComp`:

```metal
if (sourceComp >= targetComp) return;  // Skip reverse edges
```

### 4. Memory Barriers

Required between kernels to ensure buffer writes are visible:

```swift
encoder.memoryBarrier(scope: .buffers)
```

### 5. Buffer Reuse

All intermediate buffers are reused across iterations to minimize allocation.

---

## Build & Test Commands

```bash
# Build
swift build

# Run BoruvkaMST tests only
swift test --filter BoruvkaMSTKernelTests

# Run specific test
swift test --filter testEdgeCount100
```

---

## Verification Checklist

- [ ] `BoruvkaMST.metal` compiles (3 kernels)
- [ ] Shader registered in `KernelContext.swift`
- [ ] `BoruvkaMSTKernel.swift` compiles
- [ ] `MSTResult` struct defined with all fields
- [ ] `testEdgeCountSmall` passes (N-1 edges)
- [ ] `testEdgeCount100` passes
- [ ] `testConnectivity` passes (all points reachable)
- [ ] `testNoDuplicateEdges` passes
- [ ] `testSinglePoint` passes (0 edges)
- [ ] `testTwoPoints` passes (1 edge, correct weight)
- [ ] `testEmptyInput` passes
- [ ] `testIterationCount` passes (O(log N) iterations)
- [ ] `testTotalWeightPositive` passes

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| Metal shaders | ~150 |
| Swift kernel | ~250 |
| Tests | ~150 |
| **Total** | ~550 |

---

## Next Phases Preview

### Phase 2: Dimension Optimization
- Add dimension-specific find-min kernels (384, 512, 768, 1536)
- Implement CPU Prim's algorithm for reference
- Add `testWeightMatchesPrims` verification
- Add `DimensionOptimizedKernel` conformance

### Phase 3: API Polish + Integration
- `FusibleKernel` conformance
- `VectorProtocol` support
- `HDBSCANDistanceModule` high-level API
- Performance benchmarks

---

## Contact

- Original spec: `docs/kernel-specs/02-BoruvkaMSTKernel.md`
- Reference implementation: `MutualReachabilityKernel.swift`

Phase 1 ready for implementation 2026-01-04.
