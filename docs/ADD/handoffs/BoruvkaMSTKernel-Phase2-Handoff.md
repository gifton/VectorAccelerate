# BoruvkaMSTKernel Phase 2 Handoff

**Status: COMPLETED 2026-01-05**

## Overview

This document provides context for implementing Phase 2 of the BoruvkaMSTKernel - dimension-optimized variants and correctness verification against a CPU reference implementation.

## Phase 1 Recap (Completed)

Phase 1 delivered the core Borůvka's MST algorithm with:

| Component | Status | Location |
|-----------|--------|----------|
| Metal shader (3 kernels) | ✅ Complete | `Sources/VectorAccelerate/Metal/Shaders/BoruvkaMST.metal` |
| Swift kernel | ✅ Complete | `Sources/VectorAccelerate/Kernels/Metal4/BoruvkaMSTKernel.swift` |
| Tests (13 passing) | ✅ Complete | `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift` |

### Key Architecture Decisions from Phase 1

1. **Hybrid GPU/CPU approach**: GPU kernels find minimum edges; CPU performs Union-Find merge
2. **Why CPU merge?**: Avoids race conditions from parallel component ID updates
3. **Union-Find with path compression**: O(α(n)) amortized per operation
4. **Candidate edge deduplication**: GPU may emit A→B and B→A; CPU Union-Find naturally filters duplicates

### Current Algorithm Flow

```
For each iteration (O(log N) total):
  1. GPU: boruvka_find_min_edge_kernel     - Find min edge per point (O(N²×D))
  2. GPU: boruvka_component_reduce_kernel  - Reduce to per-component min (O(N²))
  3. GPU: boruvka_merge_kernel            - Collect candidate edges (O(N))
  4. CPU: Union-Find merge                - Deduplicate + merge components (O(N×α(N)))
```

---

## Phase 2 Scope

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| Dimension-optimized kernels | 384, 512, 768, 1536 variants | HIGH |
| CPU Prim's reference | For MST weight verification | HIGH |
| `testWeightMatchesPrims` | Verify GPU MST matches CPU | HIGH |
| `DimensionOptimizedKernel` conformance | Auto-select best kernel | MEDIUM |

### Expected LOC: ~500

---

## Task 1: Dimension-Optimized Metal Kernels

The `boruvka_find_min_edge_kernel` is the hotspot (O(N²×D) per iteration). Dimension-specific unrolling provides significant speedups.

### File: `BoruvkaMST.metal`

Add dimension-optimized variants after the existing generic kernel:

```metal
// MARK: - Dimension-Optimized Find Min Edge Kernels

/// 384-dimensional variant (most common for sentence embeddings)
/// Processes 96 float4 vectors per distance computation
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

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    // Precompute base pointer for this point
    device const float4* vec_i = (device const float4*)(embeddings + tid * 384);

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* vec_j = (device const float4*)(embeddings + j * 384);

        // Unrolled 384-dim distance: 96 float4 iterations
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);

        #pragma unroll
        for (uint k = 0; k < 24; k++) {
            uint base = k * 4;
            float4 d0 = vec_i[base + 0] - vec_j[base + 0];
            float4 d1 = vec_i[base + 1] - vec_j[base + 1];
            float4 d2 = vec_i[base + 2] - vec_j[base + 2];
            float4 d3 = vec_i[base + 3] - vec_j[base + 3];
            acc0 = fma(d0, d0, acc0);
            acc1 = fma(d1, d1, acc1);
            acc2 = fma(d2, d2, acc2);
            acc3 = fma(d3, d3, acc3);
        }

        float4 sum = acc0 + acc1 + acc2 + acc3;
        float distSq = sum.x + sum.y + sum.z + sum.w;
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

/// 512-dimensional variant (common for some models)
kernel void boruvka_find_min_edge_512_kernel(
    // Same signature as 384 variant
    // 128 float4 iterations, unroll factor 4
) { /* ... */ }

/// 768-dimensional variant (BERT-base)
kernel void boruvka_find_min_edge_768_kernel(
    // 192 float4 iterations, unroll factor 4
) { /* ... */ }

/// 1536-dimensional variant (OpenAI ada-002)
kernel void boruvka_find_min_edge_1536_kernel(
    // 384 float4 iterations, unroll factor 4
) { /* ... */ }
```

### Unroll Strategy

| Dimension | float4 Count | Unroll Factor | Inner Iterations |
|-----------|--------------|---------------|------------------|
| 384 | 96 | 4 | 24 |
| 512 | 128 | 4 | 32 |
| 768 | 192 | 4 | 48 |
| 1536 | 384 | 4 | 96 |

---

## Task 2: Swift Kernel Updates

### File: `BoruvkaMSTKernel.swift`

Add `DimensionOptimizedKernel` conformance:

```swift
// MARK: - DimensionOptimizedKernel Conformance

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension BoruvkaMSTKernel: DimensionOptimizedKernel {

    /// Supported optimized dimensions
    public static var optimizedDimensions: [Int] {
        [384, 512, 768, 1536]
    }

    /// Check if dimension has optimized kernel
    public static func hasOptimizedKernel(for dimension: Int) -> Bool {
        optimizedDimensions.contains(dimension)
    }
}
```

### Add Dimension-Specific Pipelines

```swift
// In BoruvkaMSTKernel class:

/// Dimension-optimized find-min pipelines
private var findMinPipeline384: (any MTLComputePipelineState)?
private var findMinPipeline512: (any MTLComputePipelineState)?
private var findMinPipeline768: (any MTLComputePipelineState)?
private var findMinPipeline1536: (any MTLComputePipelineState)?

/// Initialize dimension-optimized pipelines
private func initializeOptimizedPipelines(library: any MTLLibrary) async throws {
    let device = context.device.rawDevice

    if let func384 = library.makeFunction(name: "boruvka_find_min_edge_384_kernel") {
        findMinPipeline384 = try await device.makeComputePipelineState(function: func384)
    }
    if let func512 = library.makeFunction(name: "boruvka_find_min_edge_512_kernel") {
        findMinPipeline512 = try await device.makeComputePipelineState(function: func512)
    }
    if let func768 = library.makeFunction(name: "boruvka_find_min_edge_768_kernel") {
        findMinPipeline768 = try await device.makeComputePipelineState(function: func768)
    }
    if let func1536 = library.makeFunction(name: "boruvka_find_min_edge_1536_kernel") {
        findMinPipeline1536 = try await device.makeComputePipelineState(function: func1536)
    }
}

/// Get appropriate find-min pipeline for dimension
private func getFindMinPipeline(for dimension: Int) -> any MTLComputePipelineState {
    switch dimension {
    case 384: return findMinPipeline384 ?? findMinPipeline
    case 512: return findMinPipeline512 ?? findMinPipeline
    case 768: return findMinPipeline768 ?? findMinPipeline
    case 1536: return findMinPipeline1536 ?? findMinPipeline
    default: return findMinPipeline
    }
}
```

---

## Task 3: CPU Prim's Reference Implementation

For verifying MST correctness, implement Prim's algorithm on CPU.

### File: `BoruvkaMSTKernelTests.swift`

Add to test helpers:

```swift
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {

    /// CPU reference MST using Prim's algorithm with mutual reachability.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix
    ///   - coreDistances: N core distances
    /// - Returns: MST edges sorted by weight, and total weight
    static func cpuPrimsMST(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) -> (edges: [(source: Int, target: Int, weight: Float)], totalWeight: Float) {
        let n = embeddings.count
        guard n > 1 else { return ([], 0) }

        // Compute mutual reachability for edge (i, j)
        func mutualReachability(_ i: Int, _ j: Int) -> Float {
            var distSq: Float = 0
            for k in 0..<embeddings[i].count {
                let diff = embeddings[i][k] - embeddings[j][k]
                distSq += diff * diff
            }
            let dist = sqrt(distSq)
            return max(coreDistances[i], coreDistances[j], dist)
        }

        // Prim's algorithm
        var inMST = [Bool](repeating: false, count: n)
        var minWeight = [Float](repeating: .infinity, count: n)
        var parent = [Int](repeating: -1, count: n)

        minWeight[0] = 0

        var edges: [(source: Int, target: Int, weight: Float)] = []
        var totalWeight: Float = 0

        for _ in 0..<n {
            // Find minimum weight vertex not in MST
            var u = -1
            var minW: Float = .infinity
            for v in 0..<n {
                if !inMST[v] && minWeight[v] < minW {
                    minW = minWeight[v]
                    u = v
                }
            }

            guard u >= 0 else { break }
            inMST[u] = true

            // Add edge to MST (skip first vertex which has no parent)
            if parent[u] >= 0 {
                let weight = mutualReachability(parent[u], u)
                edges.append((source: parent[u], target: u, weight: weight))
                totalWeight += weight
            }

            // Update weights for adjacent vertices
            for v in 0..<n {
                if !inMST[v] {
                    let weight = mutualReachability(u, v)
                    if weight < minWeight[v] {
                        minWeight[v] = weight
                        parent[v] = u
                    }
                }
            }
        }

        return (edges, totalWeight)
    }
}
```

### Add Correctness Test

```swift
// MARK: - MST Weight Correctness Tests

func testWeightMatchesPrims() async throws {
    let n = 30
    let d = 16
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    // Compute GPU MST
    let gpuResult = try await kernel.computeMST(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    // Compute CPU reference MST
    let (_, cpuTotalWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    // MST total weight should match (any valid MST has the same total weight)
    XCTAssertEqual(
        gpuResult.totalWeight,
        cpuTotalWeight,
        accuracy: 1e-3,
        "GPU MST weight should match CPU Prim's MST weight"
    )
}

func testWeightMatchesPrimsLarger() async throws {
    let n = 100
    let d = 32
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    let gpuResult = try await kernel.computeMST(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    let (_, cpuTotalWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    XCTAssertEqual(
        gpuResult.totalWeight,
        cpuTotalWeight,
        accuracy: 1e-2,  // Slightly larger tolerance for larger n
        "GPU MST weight should match CPU Prim's MST weight"
    )
}

func testOptimizedDimensionCorrectness() async throws {
    // Test each optimized dimension matches CPU reference
    for d in [384, 512, 768] {  // Skip 1536 for speed
        let n = 20
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let (_, cpuWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuWeight,
            accuracy: 1e-2,
            "Dimension \(d): GPU MST weight should match CPU"
        )
    }
}
```

---

## Task 4: Performance Benchmark Test (Optional)

Add a performance comparison test:

```swift
func testDimensionOptimizedPerformance() async throws {
    let n = 200
    let d = 384
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

    // Warm up
    _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

    // Measure
    measure {
        let expectation = XCTestExpectation(description: "MST computation")
        Task {
            _ = try await kernel.computeMST(
                embeddings: embeddings,
                coreDistances: coreDistances
            )
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 10.0)
    }
}
```

---

## Verification Checklist

### Metal Shaders
- [x] `boruvka_find_min_edge_384_kernel` compiles
- [x] `boruvka_find_min_edge_512_kernel` compiles
- [x] `boruvka_find_min_edge_768_kernel` compiles
- [x] `boruvka_find_min_edge_1536_kernel` compiles

### Swift Kernel
- [x] `DimensionOptimizedKernel` conformance added
- [x] Optimized pipelines initialized in `init`
- [x] `getFindMinPipeline(for:)` selects correct pipeline
- [x] `runIteration` uses dimension-appropriate pipeline

### Tests
- [x] `cpuPrimsMST` helper implemented
- [x] `testWeightMatchesPrims` passes
- [x] `testWeightMatchesPrimsLarger` passes
- [x] `testOptimizedDimension384Correctness` passes
- [x] `testOptimizedDimension512Correctness` passes
- [x] `testOptimizedDimension768Correctness` passes
- [x] `testDimensionOptimizedKernelConformance` passes
- [x] All existing Phase 1 tests still pass (19 total tests)

---

## Build & Test Commands

```bash
# Build
swift build

# Run all BoruvkaMST tests
swift test --filter BoruvkaMSTKernelTests

# Run specific test
swift test --filter testWeightMatchesPrims
```

---

## Key Implementation Notes

### 1. MST Weight Uniqueness

Any valid MST of a graph has the same total weight. So even though GPU Borůvka and CPU Prim's may produce different edge sets, their total weights must match.

### 2. Numerical Precision

Use tolerance of 1e-2 to 1e-3 for weight comparisons due to:
- Float32 accumulation order differences
- GPU fast-math optimizations

### 3. Pipeline Selection Pattern

Follow the pattern from `L2DistanceKernel` for dimension-based pipeline selection:

```swift
// Reference: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift
private func getPipeline(for dimension: Int) -> any MTLComputePipelineState {
    switch dimension {
    case 384: return pipeline384 ?? genericPipeline
    // ...
    }
}
```

### 4. Graceful Fallback

If an optimized kernel isn't available, fall back to generic kernel silently.

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| Metal kernels (4 optimized) | ~250 |
| Swift dimension support | ~100 |
| CPU Prim's reference | ~60 |
| New tests | ~100 |
| **Total** | ~510 |

---

## Phase 3 Preview

After Phase 2:
- `FusibleKernel` conformance for pipeline fusion
- `VectorProtocol` support for ergonomic API
- Integration with `HDBSCANDistanceModule`
- Performance benchmarks vs CPU

---

## Reference Files

- Current implementation: `Sources/VectorAccelerate/Kernels/Metal4/BoruvkaMSTKernel.swift`
- Metal shaders: `Sources/VectorAccelerate/Metal/Shaders/BoruvkaMST.metal`
- Tests: `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift`
- Dimension-optimized pattern: `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift`
- Original spec: `docs/kernel-specs/02-BoruvkaMSTKernel.md`

---

## Phase 2 Completion Summary (2026-01-05)

### Files Modified

| File | Changes |
|------|---------|
| `Sources/VectorAccelerate/Metal/Shaders/BoruvkaMST.metal` | Added 4 dimension-optimized kernels (384, 512, 768, 1536) |
| `Sources/VectorAccelerate/Kernels/Metal4/BoruvkaMSTKernel.swift` | Added `DimensionOptimizedKernel` conformance, pipeline selection |
| `Tests/VectorAccelerateTests/BoruvkaMSTKernelTests.swift` | Added CPU Prim's reference, 6 new tests |

### Test Results

All 19 tests pass:
- 13 Phase 1 tests (edge count, connectivity, duplicates, edge cases, etc.)
- 6 Phase 2 tests (weight verification, dimension-optimized correctness, protocol conformance)

### LOC Added

| Component | LOC |
|-----------|-----|
| Metal kernels (4 optimized) | ~240 |
| Swift dimension support | ~50 |
| CPU Prim's reference | ~75 |
| New tests | ~120 |
| **Total** | ~485 |

---

Phase 2 completed 2026-01-05.
