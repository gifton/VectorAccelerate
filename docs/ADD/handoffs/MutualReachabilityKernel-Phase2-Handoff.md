# MutualReachabilityKernel Phase 2 Handoff

## Overview

This document provides context for implementing Phase 2 of the MutualReachabilityKernel - adding dimension-optimized kernels for common embedding dimensions.

## Phase 1 Summary (Completed)

Phase 1 implemented the core foundation:

### Files Created

| File | Purpose |
|------|---------|
| `Sources/VectorAccelerate/Metal/Shaders/MutualReachability.metal` | Generic dense + sparse kernels |
| `Sources/VectorAccelerate/Kernels/Metal4/MutualReachabilityKernel.swift` | Swift API wrapper |
| `Tests/VectorAccelerateTests/MutualReachabilityKernelTests.swift` | 12 passing tests |

### Files Modified

| File | Change |
|------|--------|
| `Sources/VectorAccelerate/Core/KernelContext.swift` | Added "MutualReachability" to shader compile list (line ~137) |

### Current API

```swift
public final class MutualReachabilityKernel: @unchecked Sendable, Metal4Kernel {
    // Dense mode
    public func compute(embeddings: MTLBuffer, coreDistances: MTLBuffer, n: Int, d: Int) async throws -> MTLBuffer

    // Sparse mode
    public func computeSparse(embeddings: MTLBuffer, coreDistances: MTLBuffer, pairs: MTLBuffer, pairCount: Int, d: Int) async throws -> MTLBuffer

    // Convenience (Swift arrays)
    public func compute(embeddings: [[Float]], coreDistances: [Float]) async throws -> [[Float]]
    public func computeSparse(embeddings: [[Float]], coreDistances: [Float], pairs: [(Int, Int)]) async throws -> [Float]
}
```

---

## Phase 2 Scope

Add dimension-optimized kernels with loop unrolling and multiple accumulators for common embedding dimensions.

### Target Dimensions

| Dimension | Model Examples | Unroll Factor | Accumulators |
|-----------|----------------|---------------|--------------|
| 384 | MiniLM, Sentence-BERT | 8 | 2 |
| 512 | Small BERT variants | 8 | 2 |
| 768 | BERT-base, DistilBERT | 12 | 3 |
| 1536 | OpenAI ada-002 | 16 | 4 |

### Expected Speedup

Dimension-optimized kernels typically achieve 2-5x speedup over generic kernels due to:
- Loop unrolling eliminates loop overhead
- Multiple accumulators enable instruction-level parallelism (ILP)
- float4 vectorization improves memory throughput

---

## Implementation Tasks

### 1. Add Metal Kernels to `MutualReachability.metal`

Add 4 new kernels after the existing `mutual_reachability_sparse_kernel`:

```metal
// Template for 384-dim kernel
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

Similar patterns for 512, 768, 1536 with appropriate unroll factors.

**Reference**: See `L2Distance.metal` for existing dimension-optimized kernels:
```
Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal
```

### 2. Update Swift Kernel

Modify `MutualReachabilityKernel.swift`:

1. Add pipeline properties:
```swift
private let pipeline384: (any MTLComputePipelineState)?
private let pipeline512: (any MTLComputePipelineState)?
private let pipeline768: (any MTLComputePipelineState)?
private let pipeline1536: (any MTLComputePipelineState)?
```

2. Load pipelines in `init`:
```swift
// Load dimension-optimized kernels (optional - may not exist)
self.pipeline384 = try? await device.makeComputePipelineState(
    function: library.makeFunction(name: "mutual_reachability_384_kernel")!
)
// ... similar for 512, 768, 1536
```

3. Add pipeline selection:
```swift
private func selectPipeline(for dimension: Int) -> any MTLComputePipelineState {
    switch dimension {
    case 384: return pipeline384 ?? densePipeline
    case 512: return pipeline512 ?? densePipeline
    case 768: return pipeline768 ?? densePipeline
    case 1536: return pipeline1536 ?? densePipeline
    default: return densePipeline
    }
}
```

4. Update `compute()` to use selected pipeline

5. Conform to `DimensionOptimizedKernel` protocol:
```swift
extension MutualReachabilityKernel: DimensionOptimizedKernel {
    public var optimizedDimensions: [Int] { [384, 512, 768, 1536] }
}
```

**Reference**: See `L2DistanceKernel.swift` for pattern:
```
Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift
```

### 3. Add Tests

Add to `MutualReachabilityKernelTests.swift`:

```swift
func testDimensionOptimizedMatchesGeneric() async throws {
    // Test each optimized dimension produces same results as generic
    for d in [384, 512, 768, 1536] {
        let n = 50
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
        let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Verify against CPU reference
        for i in 0..<n {
            for j in 0..<n {
                XCTAssertEqual(result[i][j], cpuResult[i][j], accuracy: 1e-3,
                    "Dimension \(d) mismatch at [\(i)][\(j)]")
            }
        }
    }
}

func testPerformanceComparison() async throws {
    // Compare optimized vs generic performance
    let n = 200
    let d = 384
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

    let start = CFAbsoluteTimeGetCurrent()
    _ = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    // 200x200 @ 384-dim should complete in < 100ms with optimized kernel
    XCTAssertLessThan(elapsed, 0.1, "384-dim optimized kernel too slow: \(elapsed)s")
}
```

---

## File Locations

```
VectorAccelerate/
├── Sources/VectorAccelerate/
│   ├── Metal/Shaders/
│   │   ├── MutualReachability.metal    # ADD: dimension-optimized kernels
│   │   └── L2Distance.metal            # REFERENCE: existing pattern
│   ├── Kernels/Metal4/
│   │   ├── MutualReachabilityKernel.swift  # MODIFY: add pipelines + selection
│   │   ├── L2DistanceKernel.swift          # REFERENCE: existing pattern
│   │   └── KernelProtocol.swift            # REFERENCE: DimensionOptimizedKernel
│   └── Core/
│       └── KernelContext.swift         # NO CHANGES NEEDED (shader already registered)
└── Tests/VectorAccelerateTests/
    └── MutualReachabilityKernelTests.swift  # ADD: dimension tests
```

---

## Key Patterns to Follow

### 1. Metal Kernel Naming
```
mutual_reachability_<dim>_kernel
```
Examples: `mutual_reachability_384_kernel`, `mutual_reachability_768_kernel`

### 2. Loop Unrolling Formula
- 384-dim: 96 float4s / 8 unroll = 12 iterations
- 512-dim: 128 float4s / 8 unroll = 16 iterations
- 768-dim: 192 float4s / 12 unroll = 16 iterations
- 1536-dim: 384 float4s / 16 unroll = 24 iterations

### 3. Accumulator Count
Use 2-4 accumulators to enable ILP:
```metal
float4 acc0 = 0.0f, acc1 = 0.0f;  // 2 accumulators for smaller dims
float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;  // 3 for 768
float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;  // 4 for 1536
```

### 4. Buffer Allocation
Use `context.device.rawDevice.makeBuffer()` (not bufferFactory):
```swift
guard let buffer = context.device.rawDevice.makeBuffer(
    length: size,
    options: .storageModeShared
) else {
    throw VectorError.bufferAllocationFailed(size: size)
}
buffer.label = "MutualReach.output"
```

---

## Build & Test Commands

```bash
# Build
swift build

# Run MutualReachability tests only
swift test --filter MutualReachabilityKernelTests

# Run specific test
swift test --filter testDimensionOptimizedMatchesGeneric
```

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| Metal kernels (4 dims) | ~200 |
| Swift changes | ~80 |
| New tests | ~50 |
| **Total** | ~330 |

---

## Verification Checklist

- [ ] All 4 dimension-optimized kernels compile
- [ ] Generic fallback still works for non-optimized dimensions
- [ ] Optimized kernels produce identical results to generic (within 1e-3 tolerance)
- [ ] Performance improvement visible (expect 2-5x for optimized dimensions)
- [ ] All existing tests still pass
- [ ] New dimension tests pass
- [ ] Conforms to `DimensionOptimizedKernel` protocol

---

## Contact

Original spec document: `docs/kernel-specs/01-MutualReachabilityKernel.md`

Phase 1 implementation completed 2026-01-04.
