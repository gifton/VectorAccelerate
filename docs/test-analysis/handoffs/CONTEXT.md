# VectorAccelerate Project Context

## Overview

VectorAccelerate is a high-performance Swift library for GPU-accelerated vector similarity search, designed for Apple Silicon. It provides Metal 4-based kernels for embedding operations, IVF indexing, quantization, and attention-based similarity computations.

## Technology Stack

- **Language**: Swift 6.0, Metal Shading Language (MSL) 4.0
- **Platform**: macOS 26+, iOS 26+, visionOS 3.0+
- **GPU API**: Metal 4 (Apple Silicon optimized)
- **Build**: Swift Package Manager

## Repository Structure

```
Sources/VectorAccelerate/
├── Core/                           # Metal context, buffer pools, compute engine
│   ├── Metal4Context.swift         # GPU context management
│   ├── BufferPool.swift            # Buffer allocation/reuse
│   └── ComputeEngine.swift         # Command encoding
├── Kernels/Metal4/                 # Swift kernel wrappers
│   ├── ProductQuantizationKernel.swift
│   ├── NeuralQuantizationKernel.swift
│   ├── AttentionSimilarityKernel.swift
│   ├── L2DistanceKernel.swift
│   └── [other kernels]
├── Metal/Shaders/                  # Metal shader sources (.metal)
│   ├── ClusteringShaders.metal     # KMeans kernels
│   ├── NeuralQuantization.metal    # Encoder/decoder kernels
│   ├── AttentionSimilarity.metal   # Attention kernels
│   ├── ProductQuantization.metal   # PQ kernels
│   └── [other shaders]
├── Index/                          # Vector indexing
│   ├── Internal/
│   │   └── IVFStructure.swift      # IVF index management
│   └── Kernels/
│       ├── Clustering/             # KMeans pipeline
│       │   ├── KMeansPipeline.swift
│       │   ├── KMeansAssignKernel.swift
│       │   └── KMeansUpdateKernel.swift
│       └── IVF/                    # IVF search kernels
└── ML/
    └── QuantizationEngine.swift    # Quantization orchestration
```

## Key Architectural Patterns

### 1. Kernel Protocol
All GPU kernels implement `Metal4Kernel` protocol:
```swift
public protocol Metal4Kernel: Sendable {
    var context: Metal4Context { get }
    func warmUp() async throws
}
```

### 2. Buffer Management
Buffers are managed via `BufferPool` with bucket-based allocation:
- Automatic size bucketing for reuse
- Thread-safe allocation
- Automatic cleanup on memory pressure

### 3. Shader Invocation
Typical kernel dispatch pattern:
```swift
let commandBuffer = context.commandQueue.makeCommandBuffer()
let encoder = commandBuffer.makeComputeCommandEncoder()
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 1)
encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

### 4. Data Layout Conventions
- Vectors: Row-major `[N, D]` where N=count, D=dimension
- Matrices: Row-major `[rows, cols]`
- All floating-point: `Float32` (32-bit)
- Quantized: `Int8` (signed) or `UInt8` (unsigned)

## Common Metal Shader Header

All shaders include `Metal4Common.h`:
```metal
#ifndef METAL4_COMMON_H
#define METAL4_COMMON_H

#include <metal_stdlib>
using namespace metal;

constant float VA_EPSILON = 1e-7f;
constant float VA_INFINITY = INFINITY;

#endif
```

## Performance Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Flat search 10K vectors | <1ms | 0.69ms |
| IVF search 10K vectors | <1ms | 0.89ms |
| Batch insert | >1M vec/s | 2.28M vec/s |
| Neural encode (768->128) | >500K vec/s | 968K vec/s |
| Attention similarity | >3M pairs/s | 4M pairs/s |

## Testing

Tests run with `swift test --parallel`. Performance benchmarks are in:
- `Tests/VectorAccelerateTests/PerformanceBenchmarks.swift`
- `Tests/VectorAccelerateTests/*BenchmarkTests.swift`

## Contact Points in Codebase

For each handoff task, the relevant files are documented in the specific handoff document.
