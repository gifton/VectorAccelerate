# Current Code Patterns

> **Purpose:** Reference for current Metal 3 patterns that need migration to Metal 4.

---

## 1. Command Queue & Buffer Creation

### Current Pattern (MetalContext.swift)

```swift
// Command queue created from device
self.commandQueue = try await device.makeCommandQueue(label: configuration.commandQueueLabel)

// Command buffer from queue
public func makeCommandBuffer() -> (any MTLCommandBuffer)? {
    commandQueue.makeCommandBuffer()
}
```

### Migration Target

```swift
// MTL4 queue with residency support
let descriptor = MTL4CommandQueueDescriptor()
self.commandQueue = device.makeCommandQueue(descriptor: descriptor) as! MTL4CommandQueue

// Command buffer from device (decoupled)
public func makeCommandBuffer() -> MTL4CommandBuffer {
    device.makeCommandBuffer() as! MTL4CommandBuffer
}

// Residency set attached to queue
commandQueue.addResidencySet(residencySet)
```

---

## 2. Compute Execution

### Current Pattern (MetalContext.swift:285-313)

```swift
public func execute<T: Sendable>(
    _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
) async throws -> T {
    guard let commandBuffer = makeCommandBuffer() else {
        throw VectorError.deviceInitializationFailed("Failed to create command buffer")
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw VectorError.deviceInitializationFailed("Failed to create compute encoder")
    }

    let result = try await operation(commandBuffer, encoder)

    encoder.endEncoding()
    commandBuffer.commit()

    return result
}
```

### Migration Target

```swift
public func execute<T: Sendable>(
    _ operation: @Sendable (MTL4CommandBuffer, MTL4ComputeCommandEncoder) async throws -> T
) async throws -> T {
    let commandBuffer = device.makeCommandBuffer() as! MTL4CommandBuffer

    // Attach residency set for this operation
    commandBuffer.useResidencySet(residencySet)

    let encoder = commandBuffer.makeComputeCommandEncoder() as! MTL4ComputeCommandEncoder

    let result = try await operation(commandBuffer, encoder)

    encoder.endEncoding()

    // Submit via queue
    commandQueue.commit([commandBuffer])

    // Signal event for completion tracking
    let eventValue = nextEventValue()
    commandQueue.signalEvent(completionEvent, value: eventValue)

    return result
}
```

---

## 3. Buffer Binding in Kernels

### Current Pattern (L2DistanceKernel.swift)

```swift
try await context.executeAndWait { commandBuffer, encoder in
    encoder.setComputePipelineState(selectedPipeline)
    encoder.setBuffer(queryVectors, offset: 0, index: 0)
    encoder.setBuffer(databaseVectors, offset: 0, index: 1)
    encoder.setBuffer(distances, offset: 0, index: 2)

    var params = parameters
    encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 3)

    let threadgroups = MTLSize(width: threadgroupCount, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
}
```

### Migration Target

```swift
try await context.executeAndWait { commandBuffer, encoder in
    encoder.setComputePipelineState(selectedPipeline)

    // Use argument table instead of individual setBuffer calls
    let argTable = argumentTablePool.acquire()
    argTable.setAddress(queryVectors.gpuAddress, index: 0)
    argTable.setAddress(databaseVectors.gpuAddress, index: 1)
    argTable.setAddress(distances.gpuAddress, index: 2)
    argTable.setAddress(paramsBuffer.gpuAddress, index: 3)

    encoder.setArgumentTable(argTable, stages: .compute)

    let threadgroups = MTLSize(width: threadgroupCount, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)

    argumentTablePool.release(argTable)
}
```

---

## 4. Shader Compilation

### Current Pattern (MetalDevice.swift:294-313)

```swift
public func makeLibrary(source: String) async throws -> any MTLLibrary {
    let options = MTLCompileOptions()
    options.fastMathEnabled = true
    options.languageVersion = .version3_0

    return try await withCheckedThrowingContinuation { continuation in
        device.makeLibrary(source: source, options: options) { library, error in
            if let library = library {
                continuation.resume(returning: library)
            } else {
                continuation.resume(throwing: VectorError.shaderCompilationFailed(
                    error?.localizedDescription ?? "Unknown error"
                ))
            }
        }
    }
}
```

### Migration Target

```swift
public actor Metal4Compiler {
    private let compiler: MTL4Compiler

    init(device: MTLDevice) throws {
        let descriptor = MTL4CompilerDescriptor()
        descriptor.qualityOfService = .userInteractive
        self.compiler = try device.makeCompiler(descriptor: descriptor)
    }

    func makeLibrary(source: String) async throws -> any MTLLibrary {
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0  // MSL 4.0

        return try await compiler.makeLibrary(source: source, options: options)
    }
}
```

---

## 5. Pipeline Creation

### Current Pattern (ShaderManager.swift:106-123)

```swift
public func getPipelineState(functionName: String) async throws -> any MTLComputePipelineState {
    let cacheKey = functionName
    if let cached = pipelineCache[cacheKey] {
        return cached
    }

    let function = try await getFunction(name: functionName)
    let pipelineState = try await device.makeComputePipelineState(function: function)
    pipelineCache[cacheKey] = pipelineState
    return pipelineState
}
```

### Migration Target

```swift
public func getPipelineState(functionName: String) async throws -> any MTLComputePipelineState {
    let cacheKey = functionName
    if let cached = pipelineCache[cacheKey] {
        return cached
    }

    // Descriptor-based function creation
    let functionDescriptor = MTL4LibraryFunctionDescriptor()
    functionDescriptor.name = functionName
    functionDescriptor.library = defaultLibrary

    let function = try await compiler.makeFunction(descriptor: functionDescriptor)

    // Pipeline creation
    let pipelineDescriptor = MTLComputePipelineDescriptor()
    pipelineDescriptor.computeFunction = function
    let pipelineState = try await device.makeComputePipelineState(descriptor: pipelineDescriptor,
                                                                   options: [])

    pipelineCache[cacheKey] = pipelineState.0
    return pipelineState.0
}
```

---

## 6. Completion Handler Synchronization

### Current Pattern (ConcurrencyShims.swift:43-52)

```swift
public extension MTLCommandBuffer {
    func commitAndWait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            self.addCompletedHandler { _ in
                continuation.resume()
            }
            self.commit()
        }
    }
}
```

### Migration Target

```swift
public extension MTL4CommandQueue {
    func commitAndWait(_ commandBuffer: MTL4CommandBuffer, event: MTLSharedEvent) async {
        let targetValue = event.signaledValue + 1

        // Commit via queue
        self.commit([commandBuffer])
        self.signalEvent(event, value: targetValue)

        // Wait for event
        await withCheckedContinuation { continuation in
            event.notifyListener(at: targetValue) { _, _ in
                continuation.resume()
            }
        }
    }
}
```

---

## 7. Buffer Pool Token

### Current Pattern (BufferPool.swift:40-54)

```swift
public final class BufferToken: @unchecked Sendable {
    public let buffer: any MTLBuffer
    private let pool: BufferPool?
    public let size: Int

    deinit {
        // Auto-return to pool
        Task.detached {
            await capturedPool?.returnBuffer(capturedBuffer, size: capturedSize)
        }
    }
}
```

### Migration Target

```swift
public final class BufferToken: @unchecked Sendable {
    public let buffer: any MTLBuffer
    private let pool: BufferPool?
    private let residencyManager: ResidencyManager?
    public let size: Int

    // GPU address for argument table binding
    public var gpuAddress: UInt64 {
        buffer.gpuAddress
    }

    init(buffer: any MTLBuffer, size: Int, pool: BufferPool?, residencyManager: ResidencyManager?) {
        self.buffer = buffer
        self.size = size
        self.pool = pool
        self.residencyManager = residencyManager

        // Register with residency manager
        residencyManager?.addAllocation(buffer)
    }

    deinit {
        // Remove from residency before returning to pool
        residencyManager?.removeAllocation(buffer)

        Task.detached {
            await capturedPool?.returnBuffer(capturedBuffer, size: capturedSize)
        }
    }
}
```

---

## 8. Metal Shader Structure

### Current Pattern (BasicOperations.metal)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void euclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];

    // Compute partial sum
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tsize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    partialSums[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        result[tgid] = sqrt(partialSums[0]);
    }
}
```

### Migration Target (MSL 4.0)

```metal
#include <metal_stdlib>
#include <metal_tensor>  // New in MSL 4.0
using namespace metal;

kernel void euclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partialSums[256];

    // Compute partial sum
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tsize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }

    // SIMD reduction first (more efficient)
    sum = simd_sum(sum);

    // Only first lane in each simdgroup writes
    if (simd_lane == 0) {
        partialSums[simd_group] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across simdgroups
    uint numSimdgroups = tsize / 32;  // 32 threads per simdgroup
    if (tid < numSimdgroups) {
        sum = partialSums[tid];
        sum = simd_sum(sum);

        if (tid == 0) {
            result[tgid] = sqrt(sum);
        }
    }
}
```

---

## 9. Kernel Variant Selection

### Current Pattern (L2DistanceKernel.swift)

```swift
let selectedPipeline: any MTLComputePipelineState
switch parameters.dimension {
case 384:
    selectedPipeline = try await getPipeline(for: "l2_distance_384")
case 512:
    selectedPipeline = try await getPipeline(for: "l2_distance_512")
case 768:
    selectedPipeline = try await getPipeline(for: "l2_distance_768")
case 1536:
    selectedPipeline = try await getPipeline(for: "l2_distance_1536")
default:
    selectedPipeline = try await getPipeline(for: "l2_distance_generic")
}
```

### Migration Target

```swift
// Same pattern, but pipelines can be harvested for faster loading
let selectedPipeline: any MTLComputePipelineState
let kernelName: String

switch parameters.dimension {
case 384:  kernelName = "l2_distance_384"
case 512:  kernelName = "l2_distance_512"
case 768:  kernelName = "l2_distance_768"
case 1536: kernelName = "l2_distance_1536"
default:   kernelName = "l2_distance_generic"
}

// Check harvested cache first
if let harvested = harvestedPipelines[kernelName] {
    selectedPipeline = harvested
} else {
    selectedPipeline = try await getPipeline(for: kernelName)
}
```

---

## Summary: Key Pattern Changes

| Pattern | Metal 3 | Metal 4 |
|---------|---------|---------|
| Command buffer source | `queue.makeCommandBuffer()` | `device.makeCommandBuffer()` |
| Resource visibility | Implicit | `ResidencySet` |
| Buffer binding | `encoder.setBuffer()` | `ArgumentTable` |
| Compilation | `device.makeLibrary()` | `MTL4Compiler` |
| Function creation | `library.makeFunction()` | `compiler.makeFunction(descriptor:)` |
| Completion sync | `addCompletedHandler` | `MTLSharedEvent` |
| MSL version | 3.0 | 4.0 |
| Concurrency | Opt-in | Default (barriers required) |
