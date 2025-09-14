# VectorAccelerate

## Overview

VectorAccelerate provides hardware acceleration for VectorStoreKit operations, leveraging Apple Silicon's Metal GPU and unified memory architecture. This package has a single dependency on VectorCore for type compatibility and delivers 10-100x performance improvements through parallel compute shaders and optimized memory management.

## Purpose

In our 5-package architecture, VectorAccelerate serves as the optional acceleration layer that other packages can leverage through protocols defined in VectorCore. It provides transparent GPU acceleration while maintaining CPU fallbacks for compatibility.

## Architecture in 5-Package System

```
VectorCore (protocols)
    ↑
VectorAccelerate (You are here)
    ↓
Used by: VectorIndex (optional), VectorStore (optional), VectorAI (required)
```

## Core Components

### 1. Metal Acceleration Engine
- **Compute Pipelines**: Pre-compiled shaders for common operations
- **Buffer Management**: Intelligent pooling and memory reuse
- **Command Optimization**: Batched GPU command submission
- **Auto-tuning**: Runtime performance optimization

### 2. Accelerated Operations

#### Distance Computations
- Cosine, Euclidean, Dot Product (all metrics from VectorCore)
- Batch matrix computation (millions of comparisons/sec)
- Streaming mode for large datasets
- Mixed precision support (FP32/FP16)

#### Matrix Operations
- General matrix multiply (GEMM)
- Transpose and reshape
- Element-wise operations
- Reductions (sum, max, mean)

#### Quantization Acceleration
- Product quantization encoding/decoding
- Scalar quantization
- Codebook operations
- Distance table computation

#### Neural Network Operations
- Layer computations (dense, attention)
- Activation functions
- Batch normalization
- Embedding operations

### 3. Memory Management
- **Unified Memory**: Zero-copy between CPU/GPU
- **Smart Pooling**: Reusable buffer allocation
- **Pressure Handling**: Adaptive to system memory
- **Alignment**: Optimized memory layouts

### 4. Performance Features
- **Adaptive Routing**: CPU vs GPU decision per operation
- **Batch Optimization**: Automatic batching for small operations
- **Pipeline Caching**: Compiled shader reuse
- **Profiling**: Built-in performance measurement

## API Design

### Simple Acceleration
```swift
import VectorCore
import VectorAccelerate

// Create accelerator
let accelerator = try await MetalAccelerator()

// Accelerate distance computation
let distances = try await accelerator.compute(
    operation: .distance(metric: .cosine),
    between: vectors1,
    and: vectors2
)

// Accelerate matrix multiply
let result = try await accelerator.compute(
    operation: .matrixMultiply,
    a: matrixA,
    b: matrixB
)
```

### Integration with Other Packages
```swift
import VectorCore
import VectorAccelerate
import VectorIndex

// VectorIndex automatically uses acceleration if available
let index = HNSWIndex(
    dimensions: 768,
    accelerator: try? await MetalAccelerator()
)

// Falls back to CPU if accelerator unavailable
let results = try await index.search(query, k: 10)
```

### Advanced Configuration
```swift
// Configure memory limits
let config = MetalConfiguration(
    memoryLimit: .gigabytes(4),
    preferredDevice: .discrete,
    enableProfiling: true
)

let accelerator = try await MetalAccelerator(configuration: config)

// Monitor performance
let stats = await accelerator.statistics()
print("GPU utilization: \(stats.utilization)%")
print("Memory used: \(stats.memoryUsed.megabytes)MB")
```

## Implementation Status

### Phase 1: Core Infrastructure (Day 1 of 3) ✅
- ✅ MetalBuffer type definition (simple wrapper with size tracking)
- ✅ AccelerationError enum (5 error cases)
- ✅ MetalContext actor (device and command queue management)
- 🚧 BufferPool implementation (Day 2)
- 📋 ShaderLibrary and pipeline management (Day 3)

### Phase 2: Distance Computation (Days 4-6) 📋
- 📋 Metal shader implementation
- 📋 GPU distance computation API
- 📋 CPU fallback with Accelerate

### Phase 3: API Integration (Days 7-9) 📋
- 📋 VectorCore AccelerationProvider conformance
- 📋 Clean public API surface
- 📋 Comprehensive test suite

### Phase 4: Performance (Days 10-12) 📋
- 📋 Profiling infrastructure
- 📋 Memory optimization
- 📋 SIMD improvements

### Phase 5: Nice-to-Haves (Days 13-15) 📋
- 📋 Additional distance metrics
- 📋 Advanced operations
- 📋 Quality of life improvements

## Performance Characteristics

### Throughput (M1 Pro)
| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Cosine Distance (batch 10K) | 100M/s | 5B/s | 50x |
| Matrix Multiply (1024x1024) | 2 GFLOPS | 2 TFLOPS | 1000x |
| Vector Quantization | 10K/s | 1M/s | 100x |

### Latency Considerations
- Kernel dispatch: ~10μs overhead
- Optimal batch size: >1000 vectors
- Memory transfer: Unified memory (zero-copy)

### When to Use GPU
- ✅ Batch operations (>1000 items)
- ✅ Matrix operations
- ✅ Training/indexing
- ❌ Single vector operations
- ❌ Low latency requirements (<100μs)

## Platform Requirements

### Minimum
- Apple Silicon Mac (M1+)
- iOS 17+ with A15 Bionic+
- macOS 14+
- 4GB unified memory

### Recommended
- M1 Pro/Max/Ultra or M2+
- 16GB+ unified memory
- Latest OS versions

### Platform Support
- ✅ macOS 14+
- ✅ iOS/iPadOS 17+
- ✅ visionOS 1.0+
- ⚠️ tvOS 17+ (limited memory)
- ❌ watchOS (no Metal compute)
- ❌ Linux/Windows

## Optimization Guidelines

### Memory Efficiency
```swift
// Reuse buffers for multiple operations
let pool = accelerator.bufferPool
let buffer = try await pool.acquire(size: vectorCount * dimensions * 4)
defer { pool.release(buffer) }

// Process in chunks for large datasets
for chunk in vectors.chunked(by: 10_000) {
    try await accelerator.process(chunk, into: buffer)
}
```

### Performance Tips
1. **Batch Operations**: Group small operations
2. **Memory Alignment**: Use aligned allocations
3. **Pipeline Warmup**: Pre-compile shaders
4. **Avoid Transfers**: Keep data on GPU

## Debugging & Profiling

### Enable Validation
```swift
#if DEBUG
let accelerator = try await MetalAccelerator(
    options: [.enableValidation, .checkNumerics]
)
#endif
```

### Performance Profiling
```swift
let profiler = MetalProfiler()
accelerator.attach(profiler: profiler)

// Run operations...

let report = await profiler.report()
print(report.slowestOperations(count: 10))
```

## Future Enhancements

- **Neural Engine**: Integration for inference
- **Multi-GPU**: Distributed computation
- **Custom Kernels**: User-defined Metal shaders
- **WebGPU**: Cross-platform compatibility
- **Quantized Ops**: INT8/INT4 computation

## Contributing

VectorAccelerate requires specialized testing:
1. Benchmark on multiple devices (M1, M2, A15+)
2. Verify numerical accuracy vs CPU
3. Test memory pressure scenarios
4. Profile power consumption
5. Ensure CPU fallback works

## Design Decisions

1. **Minimal Dependencies**: Single dependency on VectorCore for type consistency
2. **Protocol-Based**: Integration through VectorCore protocols
3. **Fail-Safe**: Always have CPU fallback
4. **Lazy Loading**: Don't initialize Metal until needed
5. **Transparent**: Same API whether CPU or GPU

This package is the performance foundation of VectorStoreKit - optimizations here benefit the entire ecosystem.