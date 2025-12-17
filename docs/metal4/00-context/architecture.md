# VectorAccelerate Architecture Overview

> **Purpose:** This document provides context for external agents working on Metal 4 migration tasks. It describes the current architecture, key abstractions, and code organization.

## Project Overview

VectorAccelerate is a high-performance GPU compute library for vector operations, optimized for embedding-based ML workloads on Apple Silicon. It provides:

- **Distance computations:** L2, cosine, dot product, Manhattan, Chebyshev, Minkowski, Hamming
- **Quantization:** Scalar (4/8/16-bit), binary, product quantization
- **Matrix operations:** Multiply, transpose, vector-matrix
- **Selection:** Top-K, streaming, warp-optimized
- **Statistics:** Mean, variance, histograms

## Technology Stack

- **Language:** Swift 6 with strict concurrency
- **GPU API:** Metal 3 (MSL 3.0)
- **Platforms:** macOS 14+, iOS 17+, tvOS 17+, visionOS 1+
- **Dependencies:** VectorCore (vector protocol abstractions)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API Layer                         │
│  ComputeEngine, BatchProcessor, QuantizationEngine              │
├─────────────────────────────────────────────────────────────────┤
│                       Kernel Layer (25+)                        │
│  L2DistanceKernel, CosineSimilarityKernel, TopKSelectionKernel │
├─────────────────────────────────────────────────────────────────┤
│                    Core Infrastructure                          │
│  MetalContext ─── MetalDevice ─── ShaderManager                 │
│       │               │               │                         │
│  BufferPool    MetalBufferFactory  Pipeline Cache               │
├─────────────────────────────────────────────────────────────────┤
│                     Metal Runtime                               │
│  MTLCommandQueue, MTLCommandBuffer, MTLComputeCommandEncoder    │
├─────────────────────────────────────────────────────────────────┤
│                   Metal Shaders (16 files)                      │
│  *.metal files with compute kernels                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MetalContext (Actor)

**File:** `Sources/VectorAccelerate/Core/MetalContext.swift`

The main entry point and resource manager. Handles:
- Device selection and capability detection
- Command queue management
- Buffer pool lifecycle
- Shader loading and caching

```swift
public actor MetalContext: AccelerationProvider {
    public let device: MetalDevice
    public let bufferPool: BufferPool
    internal let commandQueue: any MTLCommandQueue
    public nonisolated let bufferFactory: MetalBufferFactory

    // Execute compute operations
    public func execute<T: Sendable>(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T
}
```

### 2. MetalDevice

**File:** `Sources/VectorAccelerate/Core/MetalDevice.swift`

Wraps `MTLDevice` with capability detection and factory methods.

```swift
public actor MetalDevice {
    public let device: any MTLDevice
    public let capabilities: MetalDeviceCapabilities

    // Factory methods
    public func makeCommandQueue(label: String) async throws -> any MTLCommandQueue
    public func makeBuffer(length: Int, options: MTLResourceOptions) -> (any MTLBuffer)?
    public func makeLibrary(source: String) async throws -> any MTLLibrary
    public func makeComputePipelineState(function: any MTLFunction) async throws -> any MTLComputePipelineState
}
```

### 3. BufferPool (Actor)

**File:** `Sources/VectorAccelerate/Core/BufferPool.swift`

Token-based buffer pooling with RAII pattern.

```swift
public actor BufferPool: BufferProvider {
    // Bucket-based allocation for efficiency
    private let bucketSizes: [Int] = [1024, 4096, 16384, 65536, 262144, 1048576, ...]

    // Get buffer (returns token that auto-returns on deinit)
    public func getBuffer(size: Int) async throws -> BufferToken
}

public final class BufferToken: @unchecked Sendable {
    public let buffer: any MTLBuffer
    public let size: Int
    // Auto-returns to pool on deinit
}
```

### 4. ShaderManager (Actor)

**File:** `Sources/VectorAccelerate/Core/ShaderManager.swift`

Handles shader compilation and pipeline caching.

```swift
public actor ShaderManager {
    private var pipelineCache: [String: any MTLComputePipelineState] = [:]

    public func getPipelineState(functionName: String) async throws -> any MTLComputePipelineState
    public func getSpecializedPipeline(functionName: String, constantValues: MTLFunctionConstantValues) async throws -> any MTLComputePipelineState
}
```

### 5. ComputeEngine (Actor)

**File:** `Sources/VectorAccelerate/Core/ComputeEngine.swift`

High-level compute operations using kernels.

```swift
public actor ComputeEngine {
    public func computeL2Distance(queries: [[Float]], database: [[Float]]) async throws -> [[Float]]
    public func computeCosineSimilarity(...) async throws -> [[Float]]
    public func computeTopK(...) async throws -> [(index: Int, distance: Float)]
}
```

## Kernel Architecture

Each kernel follows a consistent pattern:

```swift
public actor SomeKernel {
    private let context: MetalContext
    private var pipeline: (any MTLComputePipelineState)?

    public init(context: MetalContext) {
        self.context = context
    }

    public func compute(/* params */) async throws -> Result {
        // 1. Ensure pipeline is loaded
        let pipeline = try await ensurePipeline()

        // 2. Get buffers from pool
        let inputBuffer = try await context.getBuffer(for: inputData)
        let outputBuffer = try await context.getBuffer(size: outputSize)

        // 3. Execute compute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)

            let threadgroups = calculateThreadgroups(...)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        // 4. Read results
        return outputBuffer.copyData(as: Float.self)
    }
}
```

## Metal Shader Organization

**Location:** `Sources/VectorAccelerate/Metal/Shaders/`

| File | Kernels | Purpose |
|------|---------|---------|
| `BasicOperations.metal` | 7 | euclideanDistance, cosineDistance, dotProduct, etc. |
| `L2Distance.metal` | 4 | Dimension-optimized L2 (384, 512, 768, 1536) |
| `CosineSimilarity.metal` | 4 | Dimension-optimized cosine |
| `DotProduct.metal` | 5 | Dot product + GEMV variants |
| `AdvancedTopK.metal` | 6 | Heap-based selection algorithms |
| `ProductQuantization.metal` | 4 | PQ encoding/decoding |
| `ClusteringShaders.metal` | 3 | K-means operations |
| ... | ... | ... |

### Shader Pattern

```metal
#include <metal_stdlib>
using namespace metal;

kernel void someKernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Parameters& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    // Threadgroup shared memory for reductions
    threadgroup float shared[THREADGROUP_SIZE];

    // Load data
    float value = input[tid];

    // Process
    shared[tid % tsize] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid % tsize < stride) {
            shared[tid % tsize] += shared[tid % tsize + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid % tsize == 0) {
        output[tgid] = shared[0];
    }
}
```

## Concurrency Model

- All core components are **actors** for thread safety
- Buffer access uses `@unchecked Sendable` with manual locking where needed
- Command buffer completion uses async/await via `withCheckedContinuation`
- Metal operations are inherently async (GPU execution)

## Error Handling

```swift
public enum VectorError: Error, Sendable {
    case deviceInitializationFailed(String)
    case shaderCompilationFailed(String)
    case shaderNotFound(name: String)
    case bufferAllocationFailed(size: Int)
    case invalidBufferSize(requested: Int, maximum: Int)
    case computeFailed(reason: String)
    case memoryPressure()
    case dimensionMismatch(expected: Int, got: Int)
}
```

## Key Patterns to Understand

### 1. Buffer Lifecycle
```swift
// Borrow from pool → use → auto-return on scope exit
let token = try await context.getBuffer(size: 1024)
// Use token.buffer
// Token returns to pool when it goes out of scope
```

### 2. Pipeline Caching
```swift
// Pipelines are compiled once and cached by function name
let pipeline = try await shaderManager.getPipelineState(functionName: "l2_distance_384")
```

### 3. Execution Pattern
```swift
try await context.executeAndWait { commandBuffer, encoder in
    // All encoding happens here
    // Completion is awaited automatically
}
```

### 4. Dimension-Specific Kernels
```swift
// Many kernels have optimized variants for common embedding dimensions
let kernelName = "l2_distance_\(dimension)"  // e.g., l2_distance_384, l2_distance_768
```

## File Inventory

See `file-inventory.md` for complete list of all files and their line counts.
