# Metal 4 API Quick Reference

> **Purpose:** Concise reference for Metal 4 APIs relevant to VectorAccelerate migration.

## Core Type Changes

| Metal 3 | Metal 4 | Notes |
|---------|---------|-------|
| `MTLCommandQueue` | `MTL4CommandQueue` | Adds residency set support |
| `MTLCommandBuffer` | `MTL4CommandBuffer` | Decoupled from queue, reusable |
| `MTLComputeCommandEncoder` | `MTL4ComputeCommandEncoder` | Unified (compute + blit + accel) |
| Device compilation | `MTL4Compiler` | Separate compiler with QoS |
| `setBuffer()` | `MTL4ArgumentTable` | GPU address binding |
| Implicit retention | `MTLResidencySet` | Explicit resource residency |
| `addCompletedHandler` | `MTLSharedEvent` | Event-based synchronization |

---

## 1. Command Queue & Buffer

### Creating Command Queue
```swift
// Metal 3
let queue = device.makeCommandQueue()

// Metal 4
let descriptor = MTL4CommandQueueDescriptor()
descriptor.maxCommandBufferCount = 3
let queue = device.makeCommandQueue(descriptor: descriptor) as! MTL4CommandQueue
```

### Creating Command Buffer
```swift
// Metal 3 - from queue
let commandBuffer = queue.makeCommandBuffer()

// Metal 4 - from device (decoupled)
let commandBuffer = device.makeCommandBuffer() as! MTL4CommandBuffer
```

### Submitting Commands
```swift
// Metal 3
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Metal 4
queue.commit([commandBuffer])  // Submit via queue
// Use events for synchronization (see below)
```

---

## 2. Residency Sets

Resources must be explicitly made resident in Metal 4.

### Creating Residency Set
```swift
let descriptor = MTLResidencySetDescriptor()
descriptor.initialCapacity = 64

let residencySet = try device.makeResidencySet(descriptor: descriptor)
```

### Adding Resources
```swift
// Add individual resources
residencySet.addAllocation(buffer1)
residencySet.addAllocation(buffer2)
residencySet.addAllocation(texture1)

// Commit to make resident
residencySet.commit()
```

### Attaching to Queue/Buffer
```swift
// For all command buffers on this queue
queue.addResidencySet(residencySet)

// OR for specific command buffer
commandBuffer.useResidencySet(residencySet)
```

### Resource Lifecycle
```swift
// Resource stays resident while in ANY attached residency set
// Remove when no longer needed
residencySet.removeAllocation(buffer)
residencySet.commit()  // Commit changes
```

---

## 3. Argument Tables

Replace individual `setBuffer()` calls with argument tables.

### Creating Argument Table
```swift
let descriptor = MTL4ArgumentTableDescriptor()
descriptor.maxBufferBindCount = 16
descriptor.maxTextureBindCount = 8
descriptor.maxSamplerBindCount = 4

let argumentTable = try device.makeArgumentTable(descriptor: descriptor)
```

### Binding Resources
```swift
// Buffers - use GPU address
argumentTable.setAddress(buffer.gpuAddress, index: 0)
argumentTable.setAddress(buffer.gpuAddress + offset, index: 1)

// Textures - use GPU resource ID
argumentTable.setTexture(texture.gpuResourceID, index: 0)

// Samplers
argumentTable.setSampler(sampler.gpuResourceID, index: 0)
```

### Using in Encoder
```swift
// Attach to specific stages
encoder.setArgumentTable(argumentTable, stages: .compute)

// For render encoders
encoder.setArgumentTable(vertexTable, stages: .vertex)
encoder.setArgumentTable(fragmentTable, stages: .fragment)
```

---

## 4. Unified Compute Encoder

Single encoder for compute, blit, and acceleration structure operations.

### Getting Encoder
```swift
let encoder = commandBuffer.makeComputeCommandEncoder() as! MTL4ComputeCommandEncoder
```

### Compute Operations
```swift
encoder.setComputePipelineState(pipeline)
encoder.setArgumentTable(args, stages: .compute)
encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
```

### Blit Operations (same encoder!)
```swift
// Copy buffer
encoder.copy(from: srcBuffer, sourceOffset: 0,
             to: dstBuffer, destinationOffset: 0,
             size: byteCount)

// Fill buffer
encoder.fill(buffer: dstBuffer, range: 0..<size, value: 0)
```

### Barriers
```swift
// Wait for dispatch to complete before blit
encoder.barrier(resources: [buffer],
                beforeStages: .dispatch,
                afterStages: .blit)

// Pass barrier - all previous work before all subsequent
encoder.passBarrier()
```

---

## 5. Shader Compilation

### Creating Compiler
```swift
let descriptor = MTL4CompilerDescriptor()
descriptor.qualityOfService = .userInteractive  // or .utility for background

let compiler = try device.makeCompiler(descriptor: descriptor)
```

### Compiling Library
```swift
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.fastMathEnabled = true

let library = try await compiler.makeLibrary(source: source, options: options)
```

### Creating Functions (Descriptor-Based)
```swift
// Basic function
let functionDescriptor = MTL4LibraryFunctionDescriptor()
functionDescriptor.name = "kernelName"
functionDescriptor.library = library

// With specialization
let specializedDescriptor = MTL4SpecializedFunctionDescriptor()
specializedDescriptor.functionDescriptor = functionDescriptor
specializedDescriptor.constantValues = constantValues

let function = try compiler.makeFunction(descriptor: specializedDescriptor)
```

### Pipeline Harvesting (AOT Compilation)
```swift
// Harvest pipelines for faster loading
let harvestDescriptor = MTL4PipelineHarvestDescriptor()
harvestDescriptor.pipelines = [pipeline1, pipeline2]
let harvestData = try compiler.harvest(descriptor: harvestDescriptor)

// Save to disk
try harvestData.write(to: fileURL)

// Load pre-compiled
let precompiled = try compiler.makeComputePipelineStates(harvestData: harvestData)
```

---

## 6. Synchronization

### Shared Events
```swift
// Create event
let event = device.makeSharedEvent()!
var eventValue: UInt64 = 0

// Signal after work
eventValue += 1
queue.signalEvent(event, value: eventValue)

// Wait for completion
event.notifyListener(at: eventValue) { event, value in
    // Work completed
}

// Or synchronous wait
event.wait(untilSignaledValue: eventValue, timeoutMS: 1000)
```

### Barriers
```swift
// Resource barrier
encoder.barrier(resources: [buffer1, buffer2],
                beforeStages: .dispatch,
                afterStages: .dispatch)

// Memory barrier (all resources)
encoder.memoryBarrier(beforeStages: .dispatch, afterStages: .blit)

// Pass barrier
encoder.passBarrier()
```

---

## 7. MTLTensor (ML Integration)

### Creating Tensor Buffer
```swift
let tensorDescriptor = MTLTensorDescriptor()
tensorDescriptor.dataType = .float32
tensorDescriptor.shape = [batchSize, channels, height, width]

let tensor = device.makeTensor(descriptor: tensorDescriptor)
```

### In Shaders (MSL 4.0)
```metal
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

kernel void tensorKernel(
    device MTLTensor<float, 2>* weights [[buffer(0)]],
    device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Create inline tensor from input
    auto inputTensor = MTLTensor<float, dextents<1>>(input + tid * dim, dim);

    // Matrix multiply
    auto result = matmul(inputTensor, *weights);

    // Write output
    for (int i = 0; i < outDim; i++) {
        output[tid * outDim + i] = result[i];
    }
}
```

---

## 8. MSL 4.0 Changes

### New Headers
```metal
#include <metal_stdlib>
#include <metal_tensor>      // Tensor operations
#include <metal_raytracing>  // Ray tracing (if needed)
using namespace metal;
```

### Barrier Stages
```metal
// Compute encoder stages
.dispatch              // Compute kernel execution
.blit                  // Copy/fill operations
.accelerationStructure // RT acceleration structure builds
```

### Simdgroup Matrix Operations
```metal
simdgroup_matrix<float, 8, 8> matA, matB, matC;

// Load from memory
simdgroup_load(matA, srcA, stride);
simdgroup_load(matB, srcB, stride);

// Multiply-accumulate
simdgroup_multiply_accumulate(matC, matA, matB, matC);

// Store to memory
simdgroup_store(matC, dst, stride);
```

---

## Migration Checklist

### Per-File Changes

- [ ] Replace `MTLCommandQueue` → `MTL4CommandQueue`
- [ ] Replace `queue.makeCommandBuffer()` → `device.makeCommandBuffer()`
- [ ] Add `ResidencySet` for all buffers
- [ ] Replace `setBuffer()` → `ArgumentTable`
- [ ] Replace `addCompletedHandler` → `MTLSharedEvent`
- [ ] Update `MTLCompileOptions.languageVersion` → `.version4_0`
- [ ] Add barriers where needed

### Per-Shader Changes

- [ ] Add `#include <metal_tensor>` if using tensors
- [ ] Verify `threadgroup_barrier` compatibility
- [ ] Consider `simdgroup_matrix` for matrix ops
