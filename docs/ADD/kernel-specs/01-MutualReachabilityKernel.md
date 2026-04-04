# MutualReachabilityKernel Specification

**Version**: 1.0
**Status**: Approved
**Priority**: Phase 1 (Highest)
**Estimated LOC**: ~700 (300 Metal + 400 Swift)

---

## 1. Overview

### 1.1 Purpose

The MutualReachabilityKernel computes mutual reachability distances for HDBSCAN clustering. This metric accounts for varying local densities by incorporating core distances into the distance calculation.

### 1.2 Mathematical Definition

```
mutual_reachability(a, b) = max(core_distance[a], core_distance[b], euclidean_distance(a, b))
```

Where:
- `core_distance[i]` = distance from point `i` to its k-th nearest neighbor
- `euclidean_distance(a, b)` = L2 distance between embeddings of points `a` and `b`

### 1.3 Use Case

HDBSCAN uses mutual reachability distances to build a minimum spanning tree (MST). Points in dense regions have small core distances; points in sparse regions have large core distances. The mutual reachability metric ensures that:
- Two points in a dense region: distance ≈ euclidean distance
- Point in dense + point in sparse: distance ≈ sparse point's core distance
- Two points in sparse regions: distance ≈ max of both core distances

---

## 2. Input/Output Specification

### 2.1 Dense Mode

Computes full N×N mutual reachability matrix.

**Inputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embeddings` | `MTLBuffer` (Float32) | [N, D] | Row-major embedding matrix |
| `coreDistances` | `MTLBuffer` (Float32) | [N] | Pre-computed core distances |
| `n` | `UInt32` | scalar | Number of points |
| `d` | `UInt32` | scalar | Embedding dimension |

**Outputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `output` | `MTLBuffer` (Float32) | [N, N] | Symmetric mutual reachability matrix |

**Memory**: 4 × N² bytes (e.g., 100 MB for N=5000)

### 2.2 Sparse Mode

Computes mutual reachability for specific pairs only.

**Inputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embeddings` | `MTLBuffer` (Float32) | [N, D] | Row-major embedding matrix |
| `coreDistances` | `MTLBuffer` (Float32) | [N] | Pre-computed core distances |
| `pairs` | `MTLBuffer` (UInt32) | [P, 2] | Pairs of indices to compute |
| `n` | `UInt32` | scalar | Number of points |
| `d` | `UInt32` | scalar | Embedding dimension |
| `pairCount` | `UInt32` | scalar | Number of pairs (P) |

**Outputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `output` | `MTLBuffer` (Float32) | [P] | Mutual reachability for each pair |

**Memory**: 4 × P bytes (e.g., 600 KB for P=150,000)

---

## 3. Metal Shader Design

### 3.1 File Location

```
Sources/VectorAccelerate/Shaders/MutualReachability.metal
```

### 3.2 Parameter Structure

```metal
struct MutualReachabilityParams {
    uint n;              // Number of points
    uint d;              // Embedding dimension
    uint strideEmbed;    // Stride between embeddings (default = D)
    uint pairCount;      // Number of pairs (sparse mode only)
};
```

### 3.3 Dense Kernel (Generic)

```metal
kernel void mutual_reachability_dense_kernel(
    device const float* embeddings      [[buffer(0)]],  // [N, D]
    device const float* coreDistances   [[buffer(1)]],  // [N]
    device float* output                [[buffer(2)]],  // [N, N]
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint j = tid.y;

    if (i >= params.n || j >= params.n) return;

    // Diagonal is always 0
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Compute L2 distance
    float distSq = 0.0f;
    for (uint k = 0; k < params.d; k++) {
        float diff = embeddings[i * params.strideEmbed + k]
                   - embeddings[j * params.strideEmbed + k];
        distSq += diff * diff;
    }
    float dist = sqrt(distSq);

    // Mutual reachability = max(core_i, core_j, dist)
    float coreI = coreDistances[i];
    float coreJ = coreDistances[j];
    float mutualReach = max(max(coreI, coreJ), dist);

    output[i * params.n + j] = mutualReach;
}
```

### 3.4 Dimension-Optimized Kernels

For common embedding dimensions, provide optimized variants with:
- Loop unrolling
- Multiple accumulators for ILP
- float4 vectorization

**Dimensions to optimize**:
| Dimension | Model Examples | Unroll Factor | Accumulators |
|-----------|----------------|---------------|--------------|
| 384 | MiniLM, Sentence-BERT | 8 | 2 |
| 512 | Small BERT variants | 8 | 2 |
| 768 | BERT-base, DistilBERT | 12 | 3 |
| 1536 | OpenAI ada-002 | 16 | 4 |

**Example (384-dim optimized)**:

```metal
kernel void mutual_reachability_384_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint j = tid.y;

    if (i >= params.n || j >= params.n) return;
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Vectorized L2 distance for D=384
    device const float4* vecI = (device const float4*)(embeddings + i * 384);
    device const float4* vecJ = (device const float4*)(embeddings + j * 384);

    float4 acc0 = 0.0f, acc1 = 0.0f;

    // 384 / 4 = 96 float4s, unroll by 8 = 12 iterations
    for (uint k = 0; k < 96; k += 8) {
        float4 d0 = vecI[k+0] - vecJ[k+0];
        float4 d1 = vecI[k+1] - vecJ[k+1];
        float4 d2 = vecI[k+2] - vecJ[k+2];
        float4 d3 = vecI[k+3] - vecJ[k+3];
        float4 d4 = vecI[k+4] - vecJ[k+4];
        float4 d5 = vecI[k+5] - vecJ[k+5];
        float4 d6 = vecI[k+6] - vecJ[k+6];
        float4 d7 = vecI[k+7] - vecJ[k+7];

        acc0 += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        acc1 += d4*d4 + d5*d5 + d6*d6 + d7*d7;
    }

    float4 sum = acc0 + acc1;
    float distSq = sum.x + sum.y + sum.z + sum.w;
    float dist = sqrt(distSq);

    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[i * params.n + j] = mutualReach;
}
```

### 3.5 Sparse Kernel

```metal
kernel void mutual_reachability_sparse_kernel(
    device const float* embeddings      [[buffer(0)]],  // [N, D]
    device const float* coreDistances   [[buffer(1)]],  // [N]
    device const uint2* pairs           [[buffer(2)]],  // [P] pairs
    device float* output                [[buffer(3)]],  // [P]
    constant MutualReachabilityParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.pairCount) return;

    uint i = pairs[tid].x;
    uint j = pairs[tid].y;

    // Compute L2 distance
    float distSq = 0.0f;
    for (uint k = 0; k < params.d; k++) {
        float diff = embeddings[i * params.d + k] - embeddings[j * params.d + k];
        distSq += diff * diff;
    }
    float dist = sqrt(distSq);

    // Mutual reachability
    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[tid] = mutualReach;
}
```

---

## 4. Swift API Design

### 4.1 File Location

```
Sources/VectorAccelerate/Kernels/Distance/MutualReachabilityKernel.swift
```

### 4.2 Public Interface

```swift
import Metal

/// Computes mutual reachability distances for HDBSCAN clustering.
///
/// Mutual reachability is defined as:
/// ```
/// mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
/// ```
///
/// This kernel supports both dense (N×N matrix) and sparse (specific pairs) modes.
public struct MutualReachabilityKernel: Metal4Kernel, DimensionOptimizedKernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let pipelines: [Int: MTLComputePipelineState]  // dimension -> pipeline
    private let sparsePipeline: MTLComputePipelineState

    // MARK: - Initialization

    /// Creates a new MutualReachabilityKernel.
    ///
    /// - Parameter context: The Metal context for GPU execution.
    /// - Throws: If pipeline creation fails.
    public init(context: Metal4Context) throws {
        self.context = context

        // Load dimension-optimized pipelines
        var pipelines: [Int: MTLComputePipelineState] = [:]
        pipelines[384] = try context.makePipeline(function: "mutual_reachability_384_kernel")
        pipelines[512] = try context.makePipeline(function: "mutual_reachability_512_kernel")
        pipelines[768] = try context.makePipeline(function: "mutual_reachability_768_kernel")
        pipelines[1536] = try context.makePipeline(function: "mutual_reachability_1536_kernel")
        pipelines[0] = try context.makePipeline(function: "mutual_reachability_dense_kernel")  // generic
        self.pipelines = pipelines

        self.sparsePipeline = try context.makePipeline(function: "mutual_reachability_sparse_kernel")
    }

    // MARK: - Dense Mode

    /// Computes full N×N mutual reachability matrix.
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N×D embedding matrix (row-major).
    ///   - coreDistances: Buffer containing N core distances.
    ///   - n: Number of points.
    ///   - d: Embedding dimension.
    /// - Returns: Buffer containing N×N mutual reachability matrix.
    /// - Throws: If GPU execution fails.
    ///
    /// - Complexity: O(N² × D) compute, O(N²) memory
    public func compute(
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MTLBuffer {
        let outputSize = n * n * MemoryLayout<Float>.size
        let output = try context.makeBuffer(length: outputSize, label: "MutualReach.output")

        try await execute(
            embeddings: embeddings,
            coreDistances: coreDistances,
            output: output,
            n: n,
            d: d
        )

        return output
    }

    /// Encodes dense mutual reachability computation into an existing encoder.
    ///
    /// Use this for kernel fusion with subsequent operations.
    public func encode(
        into encoder: MTLComputeCommandEncoder,
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        output: MTLBuffer,
        n: Int,
        d: Int
    ) {
        let pipeline = selectPipeline(for: d)
        encoder.setComputePipelineState(pipeline)

        var params = MutualReachabilityParams(
            n: UInt32(n),
            d: UInt32(d),
            strideEmbed: UInt32(d),
            pairCount: 0
        )

        encoder.setBuffer(embeddings, offset: 0, index: 0)
        encoder.setBuffer(coreDistances, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 3)

        let config = Metal4ThreadConfiguration.grid2D(
            width: n,
            height: n,
            pipeline: pipeline
        )
        encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
    }

    // MARK: - Sparse Mode

    /// Computes mutual reachability for specific pairs only.
    ///
    /// - Parameters:
    ///   - embeddings: Buffer containing N×D embedding matrix.
    ///   - coreDistances: Buffer containing N core distances.
    ///   - pairs: Buffer containing P pairs of indices as (UInt32, UInt32).
    ///   - pairCount: Number of pairs (P).
    ///   - d: Embedding dimension.
    /// - Returns: Buffer containing P mutual reachability distances.
    /// - Throws: If GPU execution fails.
    ///
    /// - Complexity: O(P × D) compute, O(P) memory
    public func computeSparse(
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        pairs: MTLBuffer,
        pairCount: Int,
        d: Int
    ) async throws -> MTLBuffer {
        let outputSize = pairCount * MemoryLayout<Float>.size
        let output = try context.makeBuffer(length: outputSize, label: "MutualReach.sparseOutput")

        try await context.executeAndWait { commandBuffer, encoder in
            encodeSparse(
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

    /// Encodes sparse mutual reachability computation.
    public func encodeSparse(
        into encoder: MTLComputeCommandEncoder,
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        pairs: MTLBuffer,
        output: MTLBuffer,
        pairCount: Int,
        d: Int
    ) {
        encoder.setComputePipelineState(sparsePipeline)

        var params = MutualReachabilityParams(
            n: 0,  // not used in sparse mode
            d: UInt32(d),
            strideEmbed: UInt32(d),
            pairCount: UInt32(pairCount)
        )

        encoder.setBuffer(embeddings, offset: 0, index: 0)
        encoder.setBuffer(coreDistances, offset: 0, index: 1)
        encoder.setBuffer(pairs, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 4)

        let config = Metal4ThreadConfiguration.grid1D(
            count: pairCount,
            pipeline: sparsePipeline
        )
        encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
    }

    // MARK: - High-Level Convenience

    /// Computes mutual reachability from Swift arrays.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix as nested arrays.
    ///   - coreDistances: N core distances.
    /// - Returns: N×N mutual reachability matrix.
    public func compute(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> [[Float]] {
        let n = embeddings.count
        let d = embeddings[0].count

        // Flatten embeddings
        let flatEmbeddings = embeddings.flatMap { $0 }

        let embedBuffer = try context.makeBuffer(
            bytes: flatEmbeddings,
            label: "MutualReach.embeddings"
        )
        let coreBuffer = try context.makeBuffer(
            bytes: coreDistances,
            label: "MutualReach.coreDistances"
        )

        let outputBuffer = try await compute(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )

        // Read back and reshape
        let flatOutput = outputBuffer.contents().bindMemory(to: Float.self, capacity: n * n)
        var result: [[Float]] = []
        for i in 0..<n {
            result.append(Array(UnsafeBufferPointer(start: flatOutput + i * n, count: n)))
        }
        return result
    }

    // MARK: - Private

    private func selectPipeline(for dimension: Int) -> MTLComputePipelineState {
        pipelines[dimension] ?? pipelines[0]!
    }

    private func execute(
        embeddings: MTLBuffer,
        coreDistances: MTLBuffer,
        output: MTLBuffer,
        n: Int,
        d: Int
    ) async throws {
        try await context.executeAndWait { commandBuffer, encoder in
            encode(
                into: encoder,
                embeddings: embeddings,
                coreDistances: coreDistances,
                output: output,
                n: n,
                d: d
            )
        }
    }
}

// MARK: - Parameter Structure

struct MutualReachabilityParams {
    var n: UInt32
    var d: UInt32
    var strideEmbed: UInt32
    var pairCount: UInt32
}
```

---

## 5. Thread Configuration

### 5.1 Dense Mode

| N | Grid Size | Threadgroup | Notes |
|---|-----------|-------------|-------|
| 500 | 500×500 | 16×16 | ~1K threadgroups |
| 1000 | 1000×1000 | 16×16 | ~4K threadgroups |
| 5000 | 5000×5000 | 16×16 | ~100K threadgroups |

Thread group size of 16×16 = 256 threads is optimal for:
- Good occupancy on Apple Silicon
- Efficient L2 cache usage
- Balanced workload distribution

### 5.2 Sparse Mode

| P (pairs) | Grid Size | Threadgroup | Notes |
|-----------|-----------|-------------|-------|
| 10K | 10,000 | 256 | Single pass |
| 100K | 100,000 | 256 | Single pass |
| 1M | 1,000,000 | 256 | May need chunking |

---

## 6. Performance Considerations

### 6.1 Memory Access Patterns

**Dense mode**:
- Each thread reads 2 full embeddings (2 × D × 4 bytes)
- For D=384: 3 KB per thread pair
- Benefit from L2 cache when same row accessed by multiple threads

**Sparse mode**:
- Random access pattern
- Less cache-friendly but fewer total accesses
- Prefer when P << N²

### 6.2 Numerical Precision

- Use `sqrt()` not `rsqrt()` for final distance (accuracy matters for MST)
- Accumulate in float32 (sufficient for embedding dimensions ≤ 2048)
- No special handling needed for denormals

### 6.3 Expected Performance

| N | D | Mode | Estimated Time | Throughput |
|---|---|------|----------------|------------|
| 1000 | 384 | Dense | ~15ms | 66M pairs/s |
| 5000 | 384 | Dense | ~100ms | 250M pairs/s |
| 1000 | 384 | Sparse (50K) | ~2ms | 25M pairs/s |

---

## 7. Testing Requirements

### 7.1 Correctness Tests

```swift
// Test: Dense output matches CPU reference
func testDenseCorrectness() async throws {
    let embeddings: [[Float]] = generateRandomEmbeddings(n: 100, d: 384)
    let coreDistances: [Float] = generateRandomCoreDistances(n: 100)

    let gpuResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    let cpuResult = cpuMutualReachability(embeddings: embeddings, coreDistances: coreDistances)

    XCTAssertEqual(gpuResult, cpuResult, accuracy: 1e-5)
}

// Test: Sparse mode matches dense mode
func testSparseMatchesDense() async throws {
    let embeddings = generateRandomEmbeddings(n: 100, d: 384)
    let coreDistances = generateRandomCoreDistances(n: 100)
    let pairs = generateAllPairs(n: 100)  // All n² pairs

    let denseResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    let sparseResult = try await kernel.computeSparse(embeddings: embeddings, coreDistances: coreDistances, pairs: pairs)

    // Verify sparse results match corresponding dense entries
    for (idx, pair) in pairs.enumerated() {
        XCTAssertEqual(sparseResult[idx], denseResult[pair.0][pair.1], accuracy: 1e-5)
    }
}

// Test: Symmetry (dense mode produces symmetric matrix)
func testSymmetry() async throws {
    let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    for i in 0..<n {
        for j in 0..<n {
            XCTAssertEqual(result[i][j], result[j][i], accuracy: 1e-6)
        }
    }
}

// Test: Diagonal is zero
func testDiagonalZero() async throws {
    let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    for i in 0..<n {
        XCTAssertEqual(result[i][i], 0.0)
    }
}

// Test: mutual_reach >= euclidean_distance (always)
func testMutualReachLowerBound() async throws {
    let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    let l2Distances = computeL2Distances(embeddings)

    for i in 0..<n {
        for j in 0..<n {
            XCTAssertGreaterThanOrEqual(result[i][j], l2Distances[i][j] - 1e-5)
        }
    }
}
```

### 7.2 Edge Cases

- Single point (n=1): 1×1 matrix with 0
- Two identical points: distance = max(core_a, core_b)
- Core distance = 0: falls back to euclidean distance
- Very large core distances: doesn't overflow

### 7.3 Performance Tests

```swift
func testPerformance1K() async throws {
    let embeddings = generateRandomEmbeddings(n: 1000, d: 384)
    let coreDistances = generateRandomCoreDistances(n: 1000)

    measure {
        _ = try! await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    }
    // Expected: <20ms
}

func testPerformance5K() async throws {
    let embeddings = generateRandomEmbeddings(n: 5000, d: 384)
    let coreDistances = generateRandomCoreDistances(n: 5000)

    measure {
        _ = try! await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    }
    // Expected: <150ms
}
```

---

## 8. Dependencies

### 8.1 Internal Dependencies

- `Metal4Context` - GPU context management
- `Metal4ThreadConfiguration` - Thread dispatch calculation
- `MetalBufferFactory` - Buffer allocation

### 8.2 External Dependencies

- Metal framework
- simd framework (for Swift math utilities)

---

## 9. Future Considerations

### 9.1 Potential Optimizations

- **Tiled computation**: For N > 10K, compute in tiles to reduce memory
- **Half precision**: FP16 accumulation for dimensions > 1024
- **Fused k-NN + mutual reach**: Combine core distance and mutual reach in one pass

### 9.2 API Extensions

- Batch processing for multiple embedding sets
- Streaming mode for very large N
- Integration with BoruvkaMSTKernel for end-to-end HDBSCAN
