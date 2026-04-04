# BatchMax3Kernel Specification

**Version**: 1.0
**Status**: Approved
**Priority**: Phase 3 (Utility)
**Estimated LOC**: ~170 (50 Metal + 120 Swift)

---

## 1. Overview

### 1.1 Purpose

The BatchMax3Kernel computes element-wise maximum of three arrays. This is a simple utility kernel used in mutual reachability computation and other operations.

### 1.2 Mathematical Definition

```
output[i] = max(a[i], b[i], c[i])
```

### 1.3 Use Cases

1. **Mutual Reachability** (primary): `max(core_a, core_b, dist)`
2. **Clamping**: `max(min_val, max(a, b))`
3. **General reduction**: Any 3-way maximum operation

---

## 2. Input/Output Specification

### 2.1 Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `a` | `MTLBuffer` (Float32) | [N] | First input array |
| `b` | `MTLBuffer` (Float32) | [N] | Second input array |
| `c` | `MTLBuffer` (Float32) | [N] | Third input array |
| `count` | `UInt32` | scalar | Number of elements |

### 2.2 Outputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `output` | `MTLBuffer` (Float32) | [N] | Element-wise maximum |

---

## 3. Metal Shader Design

### 3.1 File Location

```
Sources/VectorAccelerate/Shaders/BatchMax.metal
```

### 3.2 Kernel Implementation

```metal
#include <metal_stdlib>
using namespace metal;

/// Element-wise maximum of three arrays.
kernel void batch_max3_kernel(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* c       [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& count        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = max(max(a[tid], b[tid]), c[tid]);
}

/// Vectorized version for better memory throughput.
kernel void batch_max3_vectorized_kernel(
    device const float4* a      [[buffer(0)]],
    device const float4* b      [[buffer(1)]],
    device const float4* c      [[buffer(2)]],
    device float4* output       [[buffer(3)]],
    constant uint& count4       [[buffer(4)]],  // count / 4
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count4) return;
    output[tid] = max(max(a[tid], b[tid]), c[tid]);
}

/// Two-array maximum (common case).
kernel void batch_max2_kernel(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = max(a[tid], b[tid]);
}

/// In-place maximum: a[i] = max(a[i], b[i])
kernel void batch_max_inplace_kernel(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    a[tid] = max(a[tid], b[tid]);
}
```

---

## 4. Swift API Design

### 4.1 File Location

```
Sources/VectorAccelerate/Kernels/Elementwise/BatchMaxKernel.swift
```

### 4.2 Public Interface

```swift
import Metal

/// Computes element-wise maximum of arrays.
///
/// Provides efficient GPU-accelerated max operations with
/// vectorized variants for optimal memory throughput.
public struct BatchMaxKernel: Metal4Kernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let max3Pipeline: MTLComputePipelineState
    private let max3VectorizedPipeline: MTLComputePipelineState
    private let max2Pipeline: MTLComputePipelineState
    private let maxInplacePipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(context: Metal4Context) throws {
        self.context = context
        self.max3Pipeline = try context.makePipeline(function: "batch_max3_kernel")
        self.max3VectorizedPipeline = try context.makePipeline(function: "batch_max3_vectorized_kernel")
        self.max2Pipeline = try context.makePipeline(function: "batch_max2_kernel")
        self.maxInplacePipeline = try context.makePipeline(function: "batch_max_inplace_kernel")
    }

    // MARK: - Three-Array Maximum

    /// Computes element-wise max(a, b, c).
    ///
    /// - Parameters:
    ///   - a: First input buffer.
    ///   - b: Second input buffer.
    ///   - c: Third input buffer.
    ///   - count: Number of elements.
    /// - Returns: Buffer containing max(a[i], b[i], c[i]).
    public func max3(
        a: MTLBuffer,
        b: MTLBuffer,
        c: MTLBuffer,
        count: Int
    ) async throws -> MTLBuffer {
        let output = try context.makeBuffer(
            length: count * MemoryLayout<Float>.size,
            label: "BatchMax.output"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            encode(into: encoder, a: a, b: b, c: c, output: output, count: count)
        }

        return output
    }

    /// Encodes max3 for kernel fusion.
    public func encode(
        into encoder: MTLComputeCommandEncoder,
        a: MTLBuffer,
        b: MTLBuffer,
        c: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) {
        // Use vectorized version if count is divisible by 4
        if count % 4 == 0 && count >= 16 {
            encoder.setComputePipelineState(max3VectorizedPipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var count4 = UInt32(count / 4)
            encoder.setBytes(&count4, length: MemoryLayout<UInt32>.size, index: 4)

            let config = Metal4ThreadConfiguration.grid1D(count: count / 4, pipeline: max3VectorizedPipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        } else {
            encoder.setComputePipelineState(max3Pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 4)

            let config = Metal4ThreadConfiguration.grid1D(count: count, pipeline: max3Pipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
    }

    // MARK: - Two-Array Maximum

    /// Computes element-wise max(a, b).
    public func max2(
        a: MTLBuffer,
        b: MTLBuffer,
        count: Int
    ) async throws -> MTLBuffer {
        let output = try context.makeBuffer(
            length: count * MemoryLayout<Float>.size,
            label: "BatchMax.output"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(max2Pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.grid1D(count: count, pipeline: max2Pipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }

        return output
    }

    // MARK: - In-Place Maximum

    /// Computes a[i] = max(a[i], b[i]) in-place.
    public func maxInplace(
        a: MTLBuffer,
        b: MTLBuffer,
        count: Int
    ) async throws {
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(maxInplacePipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 2)

            let config = Metal4ThreadConfiguration.grid1D(count: count, pipeline: maxInplacePipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
    }

    // MARK: - Convenience (Swift Arrays)

    /// Computes max(a, b, c) for Swift arrays.
    public func max3(
        a: [Float],
        b: [Float],
        c: [Float]
    ) async throws -> [Float] {
        precondition(a.count == b.count && b.count == c.count)
        let count = a.count

        let aBuffer = try context.makeBuffer(bytes: a, label: "BatchMax.a")
        let bBuffer = try context.makeBuffer(bytes: b, label: "BatchMax.b")
        let cBuffer = try context.makeBuffer(bytes: c, label: "BatchMax.c")

        let output = try await max3(a: aBuffer, b: bBuffer, c: cBuffer, count: count)

        let ptr = output.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
```

---

## 5. Performance Considerations

### 5.1 Memory Bandwidth

This kernel is **memory-bound**, not compute-bound:
- 3 reads + 1 write per element = 16 bytes
- Apple Silicon M1 Pro: ~200 GB/s
- Theoretical: 12.5 billion elements/second

### 5.2 Vectorization

Using float4 improves memory coalescing:
- 4 elements per thread reduces thread count
- Better utilization of 128-bit memory transactions

### 5.3 Expected Performance

| Count | Time | Throughput |
|-------|------|------------|
| 1M | ~0.1ms | 10B elem/s |
| 10M | ~1ms | 10B elem/s |
| 100M | ~10ms | 10B elem/s |

---

## 6. Testing Requirements

### 6.1 Correctness Tests

```swift
func testMax3Correctness() async throws {
    let a: [Float] = [1.0, 5.0, 3.0, 2.0]
    let b: [Float] = [4.0, 2.0, 6.0, 1.0]
    let c: [Float] = [2.0, 3.0, 1.0, 7.0]

    let result = try await kernel.max3(a: a, b: b, c: c)

    XCTAssertEqual(result, [4.0, 5.0, 6.0, 7.0])
}

func testMax2Correctness() async throws {
    let a: [Float] = [1.0, 5.0, 3.0]
    let b: [Float] = [4.0, 2.0, 6.0]

    let result = try await kernel.max2(a: a, b: b)

    XCTAssertEqual(result, [4.0, 5.0, 6.0])
}

func testVectorizedMatchesScalar() async throws {
    // Test that vectorized (count % 4 == 0) matches scalar
    let count = 1024
    let a = (0..<count).map { _ in Float.random(in: -10...10) }
    let b = (0..<count).map { _ in Float.random(in: -10...10) }
    let c = (0..<count).map { _ in Float.random(in: -10...10) }

    let result = try await kernel.max3(a: a, b: b, c: c)
    let expected = zip(zip(a, b), c).map { max(max($0.0, $0.1), $1) }

    for i in 0..<count {
        XCTAssertEqual(result[i], expected[i], accuracy: 1e-6)
    }
}
```

### 6.2 Edge Cases

- Empty arrays (count = 0)
- Single element
- All equal values
- Negative values
- Inf and NaN handling

---

## 7. Integration with MutualReachability

While MutualReachabilityKernel computes the max inline, BatchMax3Kernel can be used for:

1. **Verification**: Compare inline max with separate max3 call
2. **Staged computation**: When distances are pre-computed
3. **Debugging**: Isolate max operation for testing

```swift
// Alternative mutual reachability using BatchMax3:
let distances = try await l2Kernel.compute(embeddings: embeddings)
let coreA = expandCoreDistances(coreDistances, asRows: true, n: n)
let coreB = expandCoreDistances(coreDistances, asRows: false, n: n)
let mutualReach = try await batchMaxKernel.max3(a: coreA, b: coreB, c: distances)
```
