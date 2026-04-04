# MutualReachabilityKernel Phase 3 Handoff

## Overview

This document provides context for implementing Phase 3 of the MutualReachabilityKernel - adding fusion support, VectorProtocol integration, and API polish.

## Phase 1 & 2 Summary (Completed)

### Phase 1: Core Foundation
- Generic dense kernel (`mutual_reachability_dense_kernel`)
- Sparse kernel (`mutual_reachability_sparse_kernel`)
- Swift wrapper with convenience APIs for `[[Float]]` arrays
- 12 tests covering correctness, edge cases, and performance

### Phase 2: Dimension Optimization
- 4 dimension-optimized kernels (384, 512, 768, 1536)
- `DimensionOptimizedKernel` protocol conformance
- Automatic pipeline selection based on dimension
- 4 additional tests (total: 16 tests passing)

### Files Created/Modified

| File | Status |
|------|--------|
| `Sources/VectorAccelerate/Metal/Shaders/MutualReachability.metal` | Complete (6 kernels) |
| `Sources/VectorAccelerate/Kernels/Metal4/MutualReachabilityKernel.swift` | Needs Phase 3 updates |
| `Tests/VectorAccelerateTests/MutualReachabilityKernelTests.swift` | Complete (16 tests) |
| `Sources/VectorAccelerate/Core/KernelContext.swift` | Complete (shader registered) |

---

## Phase 3 Scope

Add fusion support (`FusibleKernel`), VectorProtocol integration, and API polish to match other Metal4 kernels like `L2DistanceKernel`.

### Expected LOC: ~150

---

## Implementation Tasks

### 1. Add FusibleKernel Protocol Conformance

Update the class declaration:

```swift
public final class MutualReachabilityKernel: @unchecked Sendable, Metal4Kernel, DimensionOptimizedKernel, FusibleKernel {
    // ... existing properties ...

    // Add FusibleKernel properties
    public let fusibleWith: [String] = ["BoruvkaMST", "MinimumSpanningTree"]
    public let requiresBarrierAfter: Bool = true
```

### 2. Add Dense Encode API

Add an `encode()` method that encodes into an existing encoder (for kernel fusion):

```swift
// MARK: - Encode API (for Fusion)

/// Encode dense mutual reachability computation into an existing encoder.
///
/// This method does NOT create or end the encoder - it only adds dispatch commands.
/// Use this for fusing multiple operations into a single command buffer.
///
/// **Important**: If fusing with subsequent operations that read the output buffer,
/// insert `encoder.memoryBarrier(scope: .buffers)` after this call.
///
/// - Parameters:
///   - encoder: The compute command encoder to encode into
///   - embeddings: Embedding vectors buffer [N, D] - row-major float32
///   - coreDistances: Core distances buffer [N] - float32
///   - output: Output buffer [N, N] - must be pre-allocated
///   - n: Number of points
///   - d: Embedding dimension
/// - Returns: Encoding result with dispatch configuration
@discardableResult
public func encode(
    into encoder: any MTLComputeCommandEncoder,
    embeddings: any MTLBuffer,
    coreDistances: any MTLBuffer,
    output: any MTLBuffer,
    n: Int,
    d: Int
) -> Metal4EncodingResult {
    // Select pipeline
    let (pipeline, pipelineName) = selectPipeline(for: d)

    // Configure encoder
    encoder.setComputePipelineState(pipeline)
    encoder.label = "MutualReachability.\(pipelineName)"

    // Bind buffers
    encoder.setBuffer(embeddings, offset: 0, index: 0)
    encoder.setBuffer(coreDistances, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)

    // Bind parameters
    var params = MutualReachabilityParams(n: n, d: d)
    encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 3)

    // Calculate thread configuration
    let config = Metal4ThreadConfiguration.forDistanceKernel(
        numQueries: n,
        numDatabase: n,
        pipeline: pipeline
    )

    // Dispatch
    encoder.dispatchThreadgroups(
        config.threadgroups,
        threadsPerThreadgroup: config.threadsPerThreadgroup
    )

    return Metal4EncodingResult(
        pipelineName: pipelineName,
        threadgroups: config.threadgroups,
        threadsPerThreadgroup: config.threadsPerThreadgroup
    )
}
```

### 3. Add Sparse Encode API

```swift
/// Encode sparse mutual reachability computation into an existing encoder.
///
/// - Parameters:
///   - encoder: The compute command encoder
///   - embeddings: Embedding vectors buffer [N, D]
///   - coreDistances: Core distances buffer [N]
///   - pairs: Pairs buffer [P] as packed uint2
///   - output: Output buffer [P] - must be pre-allocated
///   - pairCount: Number of pairs
///   - d: Embedding dimension
/// - Returns: Encoding result
@discardableResult
public func encodeSparse(
    into encoder: any MTLComputeCommandEncoder,
    embeddings: any MTLBuffer,
    coreDistances: any MTLBuffer,
    pairs: any MTLBuffer,
    output: any MTLBuffer,
    pairCount: Int,
    d: Int
) -> Metal4EncodingResult {
    encoder.setComputePipelineState(sparsePipeline)
    encoder.label = "MutualReachability.sparse"

    encoder.setBuffer(embeddings, offset: 0, index: 0)
    encoder.setBuffer(coreDistances, offset: 0, index: 1)
    encoder.setBuffer(pairs, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)

    var params = MutualReachabilityParams(d: d, pairCount: pairCount)
    encoder.setBytes(&params, length: MemoryLayout<MutualReachabilityParams>.size, index: 4)

    let config = Metal4ThreadConfiguration.linear(
        count: pairCount,
        pipeline: sparsePipeline
    )

    encoder.dispatchThreadgroups(
        config.threadgroups,
        threadsPerThreadgroup: config.threadsPerThreadgroup
    )

    return Metal4EncodingResult(
        pipelineName: "mutual_reachability_sparse_kernel",
        threadgroups: config.threadgroups,
        threadsPerThreadgroup: config.threadsPerThreadgroup
    )
}
```

### 4. Add VectorProtocol Support

Add generic methods that work with any `VectorProtocol` type:

```swift
// MARK: - VectorProtocol API

/// Computes mutual reachability from VectorProtocol types.
///
/// Uses zero-copy buffer creation when possible.
///
/// - Parameters:
///   - embeddings: Array of VectorProtocol-conforming vectors
///   - coreDistances: Core distances for each point
/// - Returns: N×N mutual reachability matrix
public func compute<V: VectorProtocol>(
    embeddings: [V],
    coreDistances: [Float]
) async throws -> [[Float]] where V.Scalar == Float {
    guard !embeddings.isEmpty else {
        throw VectorError.invalidInput("Empty embeddings array")
    }

    let n = embeddings.count
    let d = embeddings[0].count
    let device = context.device.rawDevice

    // Create buffers using zero-copy pattern
    let embedBuffer = try createBuffer(from: embeddings, device: device, label: "MutualReach.embeddings")

    guard let coreBuffer = device.makeBuffer(
        bytes: coreDistances,
        length: coreDistances.count * MemoryLayout<Float>.size,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
    }
    coreBuffer.label = "MutualReach.coreDistances"

    let outputBuffer = try await compute(
        embeddings: embedBuffer,
        coreDistances: coreBuffer,
        n: n,
        d: d
    )

    return extractResults(from: outputBuffer, n: n)
}

// MARK: - Private Helpers

/// Create a Metal buffer from VectorProtocol array (zero-copy pattern).
private func createBuffer<V: VectorProtocol>(
    from vectors: [V],
    device: any MTLDevice,
    label: String
) throws -> any MTLBuffer where V.Scalar == Float {
    let dimension = vectors[0].count
    let totalCount = vectors.count * dimension
    let byteSize = totalCount * MemoryLayout<Float>.size

    guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
        throw VectorError.bufferAllocationFailed(size: byteSize)
    }
    buffer.label = label

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

/// Extract results from buffer into 2D array.
private func extractResults(from buffer: any MTLBuffer, n: Int) -> [[Float]] {
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: n * n)
    var results: [[Float]] = []
    results.reserveCapacity(n)
    for i in 0..<n {
        results.append(Array(UnsafeBufferPointer(start: ptr + i * n, count: n)))
    }
    return results
}
```

### 5. Add StaticDimension Support (Optional)

For compile-time dimension safety:

```swift
/// Computes mutual reachability using StaticDimension vectors.
///
/// Provides compile-time dimension safety.
public func compute<D: StaticDimension>(
    embeddings: [Vector<D>],
    coreDistances: [Float]
) async throws -> [[Float]] {
    guard !embeddings.isEmpty else {
        throw VectorError.invalidInput("Empty embeddings array")
    }

    let n = embeddings.count
    let dimension = D.value
    let device = context.device.rawDevice

    let embedBuffer = try createBuffer(from: embeddings, device: device, label: "MutualReach.embeddings")

    guard let coreBuffer = device.makeBuffer(
        bytes: coreDistances,
        length: coreDistances.count * MemoryLayout<Float>.size,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: coreDistances.count * MemoryLayout<Float>.size)
    }
    coreBuffer.label = "MutualReach.coreDistances"

    let outputBuffer = try await compute(
        embeddings: embedBuffer,
        coreDistances: coreBuffer,
        n: n,
        d: dimension
    )

    return extractResults(from: outputBuffer, n: n)
}
```

### 6. Update File Header

Update the file header to reflect completed phases:

```swift
//
//  MutualReachabilityKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for computing mutual reachability distances (HDBSCAN).
//
//  Features:
//  - Dense mode: Full N×N mutual reachability matrix
//  - Sparse mode: Compute specific pairs only
//  - Dimension-optimized kernels (384, 512, 768, 1536)
//  - Fusion support via encode() API
//  - VectorProtocol integration
```

### 7. Update `compute()` to Use `encode()`

Refactor the existing `compute()` to use the new `encode()` method internally:

```swift
public func compute(
    embeddings: any MTLBuffer,
    coreDistances: any MTLBuffer,
    n: Int,
    d: Int
) async throws -> any MTLBuffer {
    // Allocate output buffer
    let outputSize = n * n * MemoryLayout<Float>.size
    guard let output = context.device.rawDevice.makeBuffer(
        length: outputSize,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: outputSize)
    }
    output.label = "MutualReach.denseOutput"

    // Execute via encode()
    try await context.executeAndWait { [self] commandBuffer, encoder in
        self.encode(
            into: encoder,
            embeddings: embeddings,
            coreDistances: coreDistances,
            output: output,
            n: n,
            d: d
        )
    }

    return output
}
```

---

## Reference Files

```
VectorAccelerate/
├── Sources/VectorAccelerate/Kernels/Metal4/
│   ├── MutualReachabilityKernel.swift  # MODIFY: add encode APIs + VectorProtocol
│   ├── L2DistanceKernel.swift          # REFERENCE: FusibleKernel pattern
│   └── KernelProtocol.swift            # REFERENCE: FusibleKernel protocol
└── Tests/VectorAccelerateTests/
    └── MutualReachabilityKernelTests.swift  # ADD: fusion + VectorProtocol tests
```

---

## Key Patterns from L2DistanceKernel

### 1. FusibleKernel Properties

```swift
public let fusibleWith: [String] = ["TopKSelection", "L2Normalization"]
public let requiresBarrierAfter: Bool = true
```

### 2. Encode API Signature

```swift
@discardableResult
public func encode(
    into encoder: any MTLComputeCommandEncoder,
    queries: any MTLBuffer,
    database: any MTLBuffer,
    distances: any MTLBuffer,
    parameters: L2DistanceParameters
) -> Metal4EncodingResult
```

### 3. VectorProtocol Buffer Creation

```swift
private func createBuffer<V: VectorProtocol>(
    from vectors: [V],
    device: any MTLDevice,
    label: String
) throws -> any MTLBuffer where V.Scalar == Float {
    // Use withUnsafeBufferPointer for zero-copy
    for (i, vector) in vectors.enumerated() {
        vector.withUnsafeBufferPointer { srcPtr in
            // Direct memory copy
        }
    }
}
```

---

## New Tests to Add

```swift
// MARK: - Phase 3: Fusion & VectorProtocol Tests

func testEncodeAPIProducesCorrectResults() async throws {
    let n = 30
    let d = 384
    let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
    let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

    // Flatten and create buffers
    let device = context.device.rawDevice
    let flatEmbeddings = embeddings.flatMap { $0 }
    let embedBuffer = device.makeBuffer(bytes: flatEmbeddings, length: flatEmbeddings.count * 4, options: .storageModeShared)!
    let coreBuffer = device.makeBuffer(bytes: coreDistances, length: coreDistances.count * 4, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(length: n * n * 4, options: .storageModeShared)!

    // Use encode() API
    try await context.executeAndWait { [kernel] _, encoder in
        kernel.encode(
            into: encoder,
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            output: outputBuffer,
            n: n,
            d: d
        )
    }

    // Verify against CPU reference
    let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: n * n)
    for i in 0..<n {
        for j in 0..<n {
            XCTAssertEqual(ptr[i * n + j], cpuResult[i][j], accuracy: 1e-3)
        }
    }
}

func testFusibleKernelProperties() async throws {
    XCTAssertTrue(kernel.requiresBarrierAfter)
    XCTAssertFalse(kernel.fusibleWith.isEmpty)
}

func testVectorProtocolSupport() async throws {
    // Test with DynamicVector from VectorCore
    let n = 20
    let d = 128

    // Create DynamicVector embeddings
    var embeddings: [DynamicVector] = []
    for _ in 0..<n {
        let values = (0..<d).map { _ in Float.random(in: -1...1) }
        embeddings.append(DynamicVector(values))
    }
    let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

    // Should work with VectorProtocol
    let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

    XCTAssertEqual(result.count, n)
    XCTAssertEqual(result[0].count, n)

    // Diagonal should be zero
    for i in 0..<n {
        XCTAssertEqual(result[i][i], 0.0)
    }
}
```

---

## Build & Test Commands

```bash
# Build
swift build

# Run all MutualReachability tests
swift test --filter MutualReachabilityKernelTests

# Run specific Phase 3 tests
swift test --filter testEncodeAPIProducesCorrectResults
swift test --filter testVectorProtocolSupport
```

---

## Verification Checklist

- [x] `FusibleKernel` protocol conformance added
- [x] `encode()` method for dense mode added
- [x] `encodeSparse()` method added
- [x] `Metal4EncodingResult` returned from encode methods
- [x] VectorProtocol generic `compute()` method added
- [x] StaticDimension support added (optional)
- [x] File header updated to reflect all phases
- [x] Existing tests still pass (21 tests - Phase 1+2+3)
- [x] New fusion tests pass
- [x] New VectorProtocol tests pass

---

## Estimated LOC

| Component | LOC |
|-----------|-----|
| FusibleKernel conformance | ~10 |
| `encode()` method | ~40 |
| `encodeSparse()` method | ~30 |
| VectorProtocol support | ~50 |
| Private helpers | ~20 |
| New tests | ~60 |
| **Total** | ~210 |

---

## Contact

- Original spec: `docs/kernel-specs/01-MutualReachabilityKernel.md`
- Phase 2 handoff: `docs/handoffs/MutualReachabilityKernel-Phase2-Handoff.md`
- Reference implementation: `L2DistanceKernel.swift`

Phase 2 completed 2026-01-04.
Phase 3 completed 2026-01-05. All 21 tests passing.
