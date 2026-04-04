# BoruvkaMSTKernel Specification

**Version**: 1.0
**Status**: Approved
**Priority**: Phase 1 (Highest)
**Estimated LOC**: ~750 (250 Metal + 500 Swift)

---

## 1. Overview

### 1.1 Purpose

The BoruvkaMSTKernel computes a Minimum Spanning Tree (MST) using Borůvka's algorithm, optimized for GPU execution. This is used in HDBSCAN to build a cluster hierarchy from mutual reachability distances.

### 1.2 Why Borůvka's Algorithm?

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

### 1.3 Algorithm Overview

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
- N=1000: ~10 iterations
- N=5000: ~13 iterations

---

## 2. Input/Output Specification

### 2.1 Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embeddings` | `MTLBuffer` (Float32) | [N, D] | Row-major embedding matrix |
| `coreDistances` | `MTLBuffer` (Float32) | [N] | Pre-computed core distances |
| `n` | `UInt32` | scalar | Number of points |
| `d` | `UInt32` | scalar | Embedding dimension |

**Note**: Mutual reachability distances are computed on-the-fly within the kernel to avoid O(N²) memory.

### 2.2 Outputs

```swift
public struct MSTResult {
    /// Edges in the MST as (source, target, weight) tuples.
    /// Always contains exactly n-1 edges for n points.
    public let edges: [(source: Int, target: Int, weight: Float)]

    /// Total weight of the MST (sum of all edge weights).
    public let totalWeight: Float

    /// Number of Borůvka iterations performed.
    public let iterations: Int
}
```

### 2.3 Intermediate Buffers (GPU-side)

| Buffer | Type | Shape | Description |
|--------|------|-------|-------------|
| `componentIds` | UInt32 | [N] | Component ID for each point |
| `minEdgeWeight` | Float32 | [N] | Min edge weight per component |
| `minEdgeTarget` | UInt32 | [N] | Target of min edge per component |
| `mstEdges` | (UInt32, UInt32, Float32) | [N-1] | Accumulated MST edges |
| `edgeCount` | UInt32 | [1] | Current number of MST edges |

---

## 3. Metal Shader Design

### 3.1 File Location

```
Sources/VectorAccelerate/Shaders/BoruvkaMST.metal
```

### 3.2 Parameter Structures

```metal
struct BoruvkaParams {
    uint n;              // Number of points
    uint d;              // Embedding dimension
    uint iteration;      // Current iteration (for debugging)
};

struct MSTEdge {
    uint source;
    uint target;
    float weight;
};
```

### 3.3 Kernel 1: Find Minimum Outgoing Edge

Each thread handles one point, finding the minimum-weight edge to a different component.

```metal
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

        // Compute mutual reachability distance
        float distSq = 0.0f;
        for (uint k = 0; k < params.d; k++) {
            float diff = embeddings[tid * params.d + k] - embeddings[j * params.d + k];
            distSq += diff * diff;
        }
        float dist = sqrt(distSq);
        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}
```

### 3.4 Kernel 2: Component-Level Minimum

Reduce per-point minimums to per-component minimums.

```metal
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
```

### 3.5 Kernel 3: Add Edges and Merge Components

Add selected edges to MST and merge components.

```metal
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

### 3.6 Optimized Find-Min Kernel (Dimension-Specific)

For D=384, use vectorized distance computation:

```metal
kernel void boruvka_find_min_edge_384_kernel(
    device const float* embeddings          [[buffer(0)]],
    device const float* coreDistances       [[buffer(1)]],
    device const uint* componentIds         [[buffer(2)]],
    device float* minEdgeWeight             [[buffer(3)]],
    device uint* minEdgeTarget              [[buffer(4)]],
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];
    device const float4* myEmbed = (device const float4*)(embeddings + tid * 384);

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* otherEmbed = (device const float4*)(embeddings + j * 384);

        // Vectorized L2 distance
        float4 acc0 = 0.0f, acc1 = 0.0f;
        for (uint k = 0; k < 96; k += 8) {
            float4 d0 = myEmbed[k+0] - otherEmbed[k+0];
            float4 d1 = myEmbed[k+1] - otherEmbed[k+1];
            float4 d2 = myEmbed[k+2] - otherEmbed[k+2];
            float4 d3 = myEmbed[k+3] - otherEmbed[k+3];
            float4 d4 = myEmbed[k+4] - otherEmbed[k+4];
            float4 d5 = myEmbed[k+5] - otherEmbed[k+5];
            float4 d6 = myEmbed[k+6] - otherEmbed[k+6];
            float4 d7 = myEmbed[k+7] - otherEmbed[k+7];
            acc0 += d0*d0 + d1*d1 + d2*d2 + d3*d3;
            acc1 += d4*d4 + d5*d5 + d6*d6 + d7*d7;
        }
        float4 sum = acc0 + acc1;
        float dist = sqrt(sum.x + sum.y + sum.z + sum.w);

        float mutualReach = max(max(myCore, coreDistances[j]), dist);
        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}
```

---

## 4. Swift API Design

### 4.1 File Location

```
Sources/VectorAccelerate/Kernels/Graph/BoruvkaMSTKernel.swift
```

### 4.2 Public Interface

```swift
import Metal

/// Result of MST computation.
public struct MSTResult: Sendable {
    /// Edges in the MST as (source, target, weight) tuples.
    public let edges: [(source: Int, target: Int, weight: Float)]

    /// Total weight of the MST.
    public let totalWeight: Float

    /// Number of iterations performed.
    public let iterations: Int

    /// Number of points in the original dataset.
    public let pointCount: Int
}

/// Computes Minimum Spanning Tree using GPU-accelerated Borůvka's algorithm.
///
/// This kernel is designed for HDBSCAN clustering, computing the MST over
/// mutual reachability distances. Distances are computed on-the-fly to
/// avoid O(N²) memory usage.
///
/// ## Algorithm
/// Borůvka's algorithm runs in O(log N) iterations, with each iteration:
/// 1. Finding the minimum outgoing edge for each component (parallel)
/// 2. Adding selected edges to the MST
/// 3. Merging connected components
///
/// ## Complexity
/// - Time: O(N² × D × log N) - dominated by distance computation per iteration
/// - Space: O(N) - no N² distance matrix stored
/// - Iterations: ~log₂(N)
public struct BoruvkaMSTKernel: Metal4Kernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let findMinPipelines: [Int: MTLComputePipelineState]
    private let componentReducePipeline: MTLComputePipelineState
    private let mergePipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(context: Metal4Context) throws {
        self.context = context

        // Dimension-optimized find-min kernels
        var findMinPipelines: [Int: MTLComputePipelineState] = [:]
        findMinPipelines[384] = try context.makePipeline(function: "boruvka_find_min_edge_384_kernel")
        findMinPipelines[0] = try context.makePipeline(function: "boruvka_find_min_edge_kernel")
        self.findMinPipelines = findMinPipelines

        self.componentReducePipeline = try context.makePipeline(function: "boruvka_component_reduce_kernel")
        self.mergePipeline = try context.makePipeline(function: "boruvka_merge_kernel")
    }

    // MARK: - Public API

    /// Computes MST from embeddings and core distances.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix buffer.
    ///   - coreDistances: N core distances buffer.
    ///   - n: Number of points.
    ///   - d: Embedding dimension.
    /// - Returns: MST result with edges and total weight.
    public func computeMST(
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MSTResult {
        // Allocate intermediate buffers
        let componentIds = try context.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            label: "Boruvka.componentIds"
        )
        let pointMinWeight = try context.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            label: "Boruvka.pointMinWeight"
        )
        let pointMinTarget = try context.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            label: "Boruvka.pointMinTarget"
        )
        let componentMinWeight = try context.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            label: "Boruvka.componentMinWeight"
        )
        let componentMinSource = try context.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            label: "Boruvka.componentMinSource"
        )
        let componentMinTarget = try context.makeBuffer(
            length: n * MemoryLayout<UInt32>.size,
            label: "Boruvka.componentMinTarget"
        )
        let mstEdges = try context.makeBuffer(
            length: (n - 1) * MemoryLayout<MSTEdgeGPU>.size,
            label: "Boruvka.mstEdges"
        )
        let edgeCount = try context.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            label: "Boruvka.edgeCount"
        )

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

    /// Convenience method for Swift arrays.
    public func computeMST(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> MSTResult {
        let n = embeddings.count
        let d = embeddings[0].count

        let flatEmbeddings = embeddings.flatMap { $0 }
        let embedBuffer = try context.makeBuffer(bytes: flatEmbeddings, label: "Boruvka.embeddings")
        let coreBuffer = try context.makeBuffer(bytes: coreDistances, label: "Boruvka.coreDistances")

        return try await computeMST(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )
    }

    // MARK: - Private

    private func runIteration(
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        componentIds: MTLBuffer,
        pointMinWeight: MTLBuffer,
        pointMinTarget: MTLBuffer,
        componentMinWeight: MTLBuffer,
        componentMinSource: MTLBuffer,
        componentMinTarget: MTLBuffer,
        mstEdges: MTLBuffer,
        edgeCount: MTLBuffer,
        n: Int,
        d: Int,
        iteration: Int
    ) async throws {
        try await context.executeAndWait { commandBuffer, encoder in
            var params = BoruvkaParams(n: UInt32(n), d: UInt32(d), iteration: UInt32(iteration))

            // Step 1: Find minimum outgoing edge per point
            let findMinPipeline = selectFindMinPipeline(for: d)
            encoder.setComputePipelineState(findMinPipeline)
            encoder.setBuffer(embeddings, offset: 0, index: 0)
            encoder.setBuffer(coreDistances, offset: 0, index: 1)
            encoder.setBuffer(componentIds, offset: 0, index: 2)
            encoder.setBuffer(pointMinWeight, offset: 0, index: 3)
            encoder.setBuffer(pointMinTarget, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 5)
            dispatchThreads(encoder: encoder, pipeline: findMinPipeline, count: n)

            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Reduce to per-component minimum
            encoder.setComputePipelineState(componentReducePipeline)
            encoder.setBuffer(componentIds, offset: 0, index: 0)
            encoder.setBuffer(pointMinWeight, offset: 0, index: 1)
            encoder.setBuffer(pointMinTarget, offset: 0, index: 2)
            encoder.setBuffer(componentMinWeight, offset: 0, index: 3)
            encoder.setBuffer(componentMinSource, offset: 0, index: 4)
            encoder.setBuffer(componentMinTarget, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
            dispatchThreads(encoder: encoder, pipeline: componentReducePipeline, count: n)

            encoder.memoryBarrier(scope: .buffers)

            // Step 3: Add edges and merge components
            encoder.setComputePipelineState(mergePipeline)
            encoder.setBuffer(componentIds, offset: 0, index: 0)
            encoder.setBuffer(componentMinWeight, offset: 0, index: 1)
            encoder.setBuffer(componentMinSource, offset: 0, index: 2)
            encoder.setBuffer(componentMinTarget, offset: 0, index: 3)
            encoder.setBuffer(mstEdges, offset: 0, index: 4)
            encoder.setBuffer(edgeCount, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<BoruvkaParams>.size, index: 6)
            dispatchThreads(encoder: encoder, pipeline: mergePipeline, count: n)
        }
    }

    private func selectFindMinPipeline(for dimension: Int) -> MTLComputePipelineState {
        findMinPipelines[dimension] ?? findMinPipelines[0]!
    }

    private func initializeComponentIds(_ buffer: MTLBuffer, n: Int) {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            ptr[i] = UInt32(i)
        }
    }

    private func readResults(
        mstEdges: MTLBuffer,
        edgeCount: MTLBuffer,
        n: Int,
        iterations: Int
    ) -> MSTResult {
        let count = edgeCount.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        let edgePtr = mstEdges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: Int(count))

        var edges: [(source: Int, target: Int, weight: Float)] = []
        var totalWeight: Float = 0

        for i in 0..<Int(count) {
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

    private func dispatchThreads(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let config = Metal4ThreadConfiguration.grid1D(count: count, pipeline: pipeline)
        encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
    }
}

// MARK: - GPU Structures

struct BoruvkaParams {
    var n: UInt32
    var d: UInt32
    var iteration: UInt32
}

struct MSTEdgeGPU {
    var source: UInt32
    var target: UInt32
    var weight: Float
}
```

---

## 5. Complexity Analysis

### 5.1 Time Complexity

| Phase | Per-Iteration | Total (log N iterations) |
|-------|---------------|--------------------------|
| Find min edge | O(N² × D) | O(N² × D × log N) |
| Component reduce | O(N²) | O(N² × log N) |
| Merge | O(N) | O(N × log N) |

**Dominant cost**: Distance computation in find-min phase

### 5.2 Space Complexity

| Buffer | Size | Notes |
|--------|------|-------|
| componentIds | O(N) | Reused each iteration |
| pointMinWeight/Target | O(N) | Reused each iteration |
| componentMin* | O(N) | Reused each iteration |
| mstEdges | O(N) | Grows to N-1 edges |
| **Total** | O(N) | No N² matrix! |

### 5.3 Iteration Count

| N | Expected Iterations | Actual (typical) |
|---|---------------------|------------------|
| 100 | 7 | 6-8 |
| 1000 | 10 | 9-11 |
| 5000 | 13 | 12-14 |

---

## 6. Performance Considerations

### 6.1 Bottleneck Analysis

The find-min kernel is the bottleneck:
- Each of N threads scans all N points
- Computes D-dimensional distance for each
- O(N² × D) work per iteration

### 6.2 Optimization Opportunities

1. **Early termination**: Skip points in same component (reduces work as components merge)
2. **Spatial indexing**: Use ball tree to prune distance computations
3. **Approximate nearest**: For very large N, sample instead of exhaustive search

### 6.3 Expected Performance

| N | D | Iterations | Estimated Time | Notes |
|---|---|------------|----------------|-------|
| 500 | 384 | ~9 | ~50ms | Includes all kernels |
| 1000 | 384 | ~10 | ~150ms | Distance dominates |
| 5000 | 384 | ~13 | ~2s | May benefit from spatial index |

---

## 7. Testing Requirements

### 7.1 Correctness Tests

```swift
// Test: MST has exactly N-1 edges
func testEdgeCount() async throws {
    let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    XCTAssertEqual(result.edges.count, n - 1)
}

// Test: MST is connected (all points reachable)
func testConnectivity() async throws {
    let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    let connected = verifyConnected(edges: result.edges, n: n)
    XCTAssertTrue(connected)
}

// Test: MST weight matches CPU Prim's implementation
func testWeightMatchesPrims() async throws {
    let gpuResult = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    let cpuResult = primsMST(embeddings: embeddings, coreDistances: coreDistances)
    XCTAssertEqual(gpuResult.totalWeight, cpuResult.totalWeight, accuracy: 1e-4)
}

// Test: No duplicate edges
func testNoDuplicates() async throws {
    let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    let edgeSet = Set(result.edges.map { EdgeKey(min($0.source, $0.target), max($0.source, $0.target)) })
    XCTAssertEqual(edgeSet.count, result.edges.count)
}

// Test: All edge weights are valid mutual reachability distances
func testEdgeWeights() async throws {
    let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
    for edge in result.edges {
        let expected = mutualReachability(embeddings[edge.source], embeddings[edge.target],
                                          coreDistances[edge.source], coreDistances[edge.target])
        XCTAssertEqual(edge.weight, expected, accuracy: 1e-5)
    }
}
```

### 7.2 Edge Cases

- Single point (n=1): Empty MST (0 edges)
- Two points (n=2): Single edge
- All identical embeddings: MST based on core distances only
- Collinear points: Verify correct chain structure

### 7.3 Performance Tests

```swift
func testPerformance1K() async throws {
    measure {
        _ = try! await kernel.computeMST(embeddings: embeddings1K, coreDistances: cores1K)
    }
    // Expected: <200ms
}
```

---

## 8. Integration with HDBSCAN

### 8.1 Usage in HDBSCANDistanceModule

```swift
public struct HDBSCANDistanceModule {
    private let mutualReachKernel: MutualReachabilityKernel
    private let mstKernel: BoruvkaMSTKernel
    private let topKKernel: FusedL2TopKKernel

    public func computeMST(
        embeddings: [[Float]],
        minSamples: Int = 5
    ) async throws -> MSTResult {
        // Step 1: Compute core distances (k-th NN distance)
        let coreDistances = try await topKKernel.computeCoreDistances(
            embeddings: embeddings,
            k: minSamples
        )

        // Step 2: Build MST on mutual reachability distances
        return try await mstKernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )
    }
}
```

### 8.2 Post-MST Processing (CPU)

After GPU MST computation, SwiftTopics performs on CPU:
1. Sort MST edges by weight (for dendrogram)
2. Build cluster hierarchy
3. Extract flat clustering via Excess of Mass (EOM)

---

## 9. Future Enhancements

### 9.1 Spatial Indexing

For N > 5000, consider integrating ball tree:
- Pre-build ball tree on GPU
- Use tree to prune distance computations
- Could reduce per-iteration cost from O(N²) to O(N log N)

### 9.2 Streaming Mode

For very large N:
- Process in batches
- Maintain partial MST across batches
- Merge batch MSTs at the end

### 9.3 Incremental Updates

When adding new points:
- Recompute only affected edges
- Efficiently update existing MST
