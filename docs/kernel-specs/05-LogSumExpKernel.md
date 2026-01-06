# LogSumExpKernel Specification

**Version**: 1.0
**Status**: Approved
**Priority**: Phase 3 (Utility)
**Estimated LOC**: ~230 (80 Metal + 150 Swift)

---

## 1. Overview

### 1.1 Purpose

The LogSumExpKernel computes the log-sum-exp function, which is essential for numerically stable softmax and log-probability computations. This is a building block for topic probability distributions.

### 1.2 Mathematical Definition

```
logsumexp(x) = log(Σ exp(x_i))
             = max(x) + log(Σ exp(x_i - max(x)))  // Numerically stable form
```

The naive form `log(Σ exp(x_i))` overflows for large x values. The stable form subtracts the maximum first.

### 1.3 Use Cases

1. **Softmax**: `softmax(x)_i = exp(x_i - logsumexp(x))`
2. **Log-probability normalization**: Normalize log-probs to sum to 1
3. **Topic distributions**: Convert topic scores to probabilities
4. **Attention mechanisms**: Stable attention score computation

---

## 2. Input/Output Specification

### 2.1 Row-wise LogSumExp

Compute logsumexp along rows of a matrix.

**Inputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `input` | `MTLBuffer` (Float32) | [N, D] | Input matrix (row-major) |
| `n` | `UInt32` | scalar | Number of rows |
| `d` | `UInt32` | scalar | Number of columns |

**Outputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `output` | `MTLBuffer` (Float32) | [N] | LogSumExp per row |

### 2.2 Full Reduction

Compute logsumexp of entire array.

**Inputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `input` | `MTLBuffer` (Float32) | [N] | Input array |
| `count` | `UInt32` | scalar | Number of elements |

**Outputs**:
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `output` | `MTLBuffer` (Float32) | [1] | Single logsumexp value |

---

## 3. Metal Shader Design

### 3.1 File Location

```
Sources/VectorAccelerate/Shaders/LogSumExp.metal
```

### 3.2 Row-wise Kernel

```metal
#include <metal_stdlib>
using namespace metal;

/// Numerically stable log-sum-exp along rows.
/// Each thread handles one row.
kernel void logsumexp_row_kernel(
    device const float* input       [[buffer(0)]],  // [N, D]
    device float* output            [[buffer(1)]],  // [N]
    constant uint& n                [[buffer(2)]],
    constant uint& d                [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float* row = input + tid * d;

    // Step 1: Find maximum
    float maxVal = row[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, row[i]);
    }

    // Step 2: Sum of exp(x - max)
    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(row[i] - maxVal);
    }

    // Step 3: log(sum) + max
    output[tid] = log(sumExp) + maxVal;
}

/// Vectorized row-wise logsumexp for D divisible by 4.
kernel void logsumexp_row_vectorized_kernel(
    device const float4* input      [[buffer(0)]],  // [N, D/4]
    device float* output            [[buffer(1)]],  // [N]
    constant uint& n                [[buffer(2)]],
    constant uint& d4               [[buffer(3)]],  // D / 4
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float4* row = input + tid * d4;

    // Find maximum using float4
    float4 maxVec = row[0];
    for (uint i = 1; i < d4; i++) {
        maxVec = max(maxVec, row[i]);
    }
    float maxVal = max(max(maxVec.x, maxVec.y), max(maxVec.z, maxVec.w));

    // Sum of exp(x - max)
    float4 sumVec = 0.0f;
    for (uint i = 0; i < d4; i++) {
        sumVec += exp(row[i] - maxVal);
    }
    float sumExp = sumVec.x + sumVec.y + sumVec.z + sumVec.w;

    output[tid] = log(sumExp) + maxVal;
}
```

### 3.3 Full Reduction Kernel

Two-pass reduction for large arrays.

```metal
/// First pass: compute partial logsumexp per threadgroup.
kernel void logsumexp_reduce_pass1_kernel(
    device const float* input           [[buffer(0)]],
    device float* partialMax            [[buffer(1)]],  // [numGroups]
    device float* partialSumExp         [[buffer(2)]],  // [numGroups]
    constant uint& count                [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];

    // Load and find local max
    float localMax = -INFINITY;
    uint idx = tid;
    while (idx < count) {
        localMax = max(localMax, input[idx]);
        idx += tsize * gridDim.x;  // Grid-stride loop
    }
    sharedMax[lid] = localMax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max within threadgroup
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float groupMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sum of exp(x - groupMax)
    float localSum = 0.0f;
    idx = tid;
    while (idx < count) {
        localSum += exp(input[idx] - groupMax);
        idx += tsize * gridDim.x;
    }
    sharedSum[lid] = localSum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum within threadgroup
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sharedSum[lid] += sharedSum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results
    if (lid == 0) {
        partialMax[tgid] = groupMax;
        partialSumExp[tgid] = sharedSum[0];
    }
}

/// Second pass: combine partial results.
kernel void logsumexp_reduce_pass2_kernel(
    device const float* partialMax      [[buffer(0)]],
    device const float* partialSumExp   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant uint& numGroups            [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];

    // Load partials
    float localMax = (lid < numGroups) ? partialMax[lid] : -INFINITY;
    sharedMax[lid] = localMax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < numGroups) {
            sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float globalMax = sharedMax[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Adjust sums to global max and reduce
    float localSum = 0.0f;
    if (lid < numGroups) {
        // Adjust: sumExp_i was computed with partialMax_i
        // Need: exp(x - globalMax) = exp(x - partialMax) * exp(partialMax - globalMax)
        localSum = partialSumExp[lid] * exp(partialMax[lid] - globalMax);
    }
    sharedSum[lid] = localSum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < numGroups) {
            sharedSum[lid] += sharedSum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = log(sharedSum[0]) + globalMax;
    }
}
```

### 3.4 Softmax Kernel (Convenience)

```metal
/// Compute softmax using logsumexp.
/// softmax(x)_i = exp(x_i - logsumexp(x))
kernel void softmax_row_kernel(
    device const float* input       [[buffer(0)]],  // [N, D]
    device float* output            [[buffer(1)]],  // [N, D]
    constant uint& n                [[buffer(2)]],
    constant uint& d                [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= n || col >= d) return;

    device const float* rowPtr = input + row * d;

    // Compute logsumexp for this row
    float maxVal = rowPtr[0];
    for (uint i = 1; i < d; i++) {
        maxVal = max(maxVal, rowPtr[i]);
    }

    float sumExp = 0.0f;
    for (uint i = 0; i < d; i++) {
        sumExp += exp(rowPtr[i] - maxVal);
    }

    float lse = log(sumExp) + maxVal;

    // Output softmax
    output[row * d + col] = exp(input[row * d + col] - lse);
}
```

---

## 4. Swift API Design

### 4.1 File Location

```
Sources/VectorAccelerate/Kernels/Reduction/LogSumExpKernel.swift
```

### 4.2 Public Interface

```swift
import Metal

/// Computes numerically stable log-sum-exp and related operations.
///
/// LogSumExp is defined as:
/// ```
/// logsumexp(x) = log(Σ exp(x_i))
/// ```
///
/// This kernel uses the stable formulation:
/// ```
/// logsumexp(x) = max(x) + log(Σ exp(x_i - max(x)))
/// ```
public struct LogSumExpKernel: Metal4Kernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let rowPipeline: MTLComputePipelineState
    private let rowVectorizedPipeline: MTLComputePipelineState
    private let reducePass1Pipeline: MTLComputePipelineState
    private let reducePass2Pipeline: MTLComputePipelineState
    private let softmaxPipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(context: Metal4Context) throws {
        self.context = context
        self.rowPipeline = try context.makePipeline(function: "logsumexp_row_kernel")
        self.rowVectorizedPipeline = try context.makePipeline(function: "logsumexp_row_vectorized_kernel")
        self.reducePass1Pipeline = try context.makePipeline(function: "logsumexp_reduce_pass1_kernel")
        self.reducePass2Pipeline = try context.makePipeline(function: "logsumexp_reduce_pass2_kernel")
        self.softmaxPipeline = try context.makePipeline(function: "softmax_row_kernel")
    }

    // MARK: - Row-wise LogSumExp

    /// Computes logsumexp along each row of a matrix.
    ///
    /// - Parameters:
    ///   - input: N×D input matrix buffer.
    ///   - n: Number of rows.
    ///   - d: Number of columns.
    /// - Returns: Buffer of N logsumexp values.
    public func rowwise(
        input: MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MTLBuffer {
        let output = try context.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            label: "LogSumExp.rowOutput"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            encodeRowwise(into: encoder, input: input, output: output, n: n, d: d)
        }

        return output
    }

    /// Encodes row-wise logsumexp for kernel fusion.
    public func encodeRowwise(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        n: Int,
        d: Int
    ) {
        let useVectorized = d % 4 == 0 && d >= 16

        if useVectorized {
            encoder.setComputePipelineState(rowVectorizedPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var nU32 = UInt32(n)
            var d4 = UInt32(d / 4)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&d4, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.grid1D(count: n, pipeline: rowVectorizedPipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        } else {
            encoder.setComputePipelineState(rowPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var nU32 = UInt32(n)
            var dU32 = UInt32(d)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&dU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.grid1D(count: n, pipeline: rowPipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
    }

    // MARK: - Full Reduction

    /// Computes logsumexp of entire array.
    ///
    /// - Parameters:
    ///   - input: Input array buffer.
    ///   - count: Number of elements.
    /// - Returns: Single logsumexp value.
    public func reduce(
        input: MTLBuffer,
        count: Int
    ) async throws -> Float {
        let numGroups = min((count + 255) / 256, 256)

        let partialMax = try context.makeBuffer(
            length: numGroups * MemoryLayout<Float>.size,
            label: "LogSumExp.partialMax"
        )
        let partialSum = try context.makeBuffer(
            length: numGroups * MemoryLayout<Float>.size,
            label: "LogSumExp.partialSum"
        )
        let output = try context.makeBuffer(
            length: MemoryLayout<Float>.size,
            label: "LogSumExp.output"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            // Pass 1: Compute partial logsumexp per group
            encoder.setComputePipelineState(reducePass1Pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(partialMax, offset: 0, index: 1)
            encoder.setBuffer(partialSum, offset: 0, index: 2)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: numGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )

            encoder.memoryBarrier(scope: .buffers)

            // Pass 2: Combine partials
            encoder.setComputePipelineState(reducePass2Pipeline)
            encoder.setBuffer(partialMax, offset: 0, index: 0)
            encoder.setBuffer(partialSum, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var numGroupsU32 = UInt32(numGroups)
            encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        return output.contents().bindMemory(to: Float.self, capacity: 1).pointee
    }

    // MARK: - Softmax

    /// Computes softmax along each row.
    ///
    /// softmax(x)_i = exp(x_i) / Σ exp(x_j)
    ///              = exp(x_i - logsumexp(x))
    public func softmax(
        input: MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> MTLBuffer {
        let output = try context.makeBuffer(
            length: n * d * MemoryLayout<Float>.size,
            label: "Softmax.output"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(softmaxPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var nU32 = UInt32(n)
            var dU32 = UInt32(d)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&dU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.grid2D(width: d, height: n, pipeline: softmaxPipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }

        return output
    }

    // MARK: - Convenience (Swift Arrays)

    /// Computes row-wise logsumexp for Swift arrays.
    public func rowwise(input: [[Float]]) async throws -> [Float] {
        let n = input.count
        let d = input[0].count
        let flat = input.flatMap { $0 }

        let inputBuffer = try context.makeBuffer(bytes: flat, label: "LogSumExp.input")
        let outputBuffer = try await rowwise(input: inputBuffer, n: n, d: d)

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Computes logsumexp of a 1D array.
    public func reduce(input: [Float]) async throws -> Float {
        let inputBuffer = try context.makeBuffer(bytes: input, label: "LogSumExp.input")
        return try await reduce(input: inputBuffer, count: input.count)
    }
}
```

---

## 5. Numerical Stability

### 5.1 Problem with Naive Implementation

```swift
// DANGEROUS: Overflows for large values
let naive = log(values.map { exp($0) }.reduce(0, +))

// Example: values = [1000, 1001, 1002]
// exp(1000) = Inf → sum = Inf → log(Inf) = Inf
```

### 5.2 Stable Implementation

```swift
// SAFE: Subtracts max first
let maxVal = values.max()!
let stable = maxVal + log(values.map { exp($0 - maxVal) }.reduce(0, +))

// Example: values = [1000, 1001, 1002]
// After shift: [0, 1, 2]
// exp: [1, e, e²] ≈ [1, 2.718, 7.389]
// sum ≈ 11.1
// result = 1002 + log(11.1) ≈ 1004.4
```

### 5.3 Edge Cases

- All -Inf: returns -Inf
- Contains +Inf: returns +Inf
- Contains NaN: returns NaN
- Single element: returns that element

---

## 6. Testing Requirements

### 6.1 Correctness Tests

```swift
func testRowwiseCorrectness() async throws {
    let input: [[Float]] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, -2.0, -3.0]
    ]

    let result = try await kernel.rowwise(input: input)

    // Expected: log(e^1 + e^2 + e^3), log(e^4 + e^5 + e^6), ...
    let expected: [Float] = [
        3.0 + log(exp(-2.0) + exp(-1.0) + 1.0),  // ≈ 3.407
        6.0 + log(exp(-2.0) + exp(-1.0) + 1.0),  // ≈ 6.407
        -1.0 + log(1.0 + exp(-1.0) + exp(-2.0))  // ≈ -0.593
    ]

    for i in 0..<3 {
        XCTAssertEqual(result[i], expected[i], accuracy: 1e-5)
    }
}

func testNumericalStability() async throws {
    // Large values that would overflow naive exp
    let input: [Float] = [1000, 1001, 1002]
    let result = try await kernel.reduce(input: input)

    // Should not be Inf or NaN
    XCTAssertFalse(result.isInfinite)
    XCTAssertFalse(result.isNaN)

    // Expected: 1002 + log(1 + e^(-1) + e^(-2)) ≈ 1002.41
    XCTAssertEqual(result, 1002.41, accuracy: 0.01)
}

func testSoftmaxSumsToOne() async throws {
    let input: [[Float]] = [
        [1.0, 2.0, 3.0, 4.0]
    ]
    let n = 1, d = 4

    let inputBuffer = try context.makeBuffer(bytes: input.flatMap { $0 }, label: "test")
    let output = try await kernel.softmax(input: inputBuffer, n: n, d: d)

    let ptr = output.contents().bindMemory(to: Float.self, capacity: d)
    let sum = (0..<d).map { ptr[$0] }.reduce(0, +)

    XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
}
```

---

## 7. Performance Considerations

### 7.1 Row-wise vs Full Reduction

- **Row-wise**: O(N) threads, each does O(D) work - efficient for matrix operations
- **Full reduction**: Two-pass algorithm, better for very large 1D arrays

### 7.2 Expected Performance

| Operation | Shape | Time |
|-----------|-------|------|
| Rowwise | 1000×50 | ~0.1ms |
| Rowwise | 10000×100 | ~1ms |
| Reduce | 1M elements | ~0.5ms |
| Softmax | 1000×50 | ~0.2ms |

---

## 8. Integration with SwiftTopics

### 8.1 Topic Probability Distribution

```swift
// Convert topic scores to probabilities
let topicScores: [[Float]] = ... // [numDocs, numTopics]
let topicProbs = try await logSumExpKernel.softmax(
    input: topicScoresBuffer,
    n: numDocs,
    d: numTopics
)
```

### 8.2 Log-Probability Normalization

```swift
// Normalize log-probabilities
let logProbs: [Float] = ... // Log-probabilities for topics
let logNormalizer = try await logSumExpKernel.reduce(input: logProbs)
let normalizedLogProbs = logProbs.map { $0 - logNormalizer }
```
