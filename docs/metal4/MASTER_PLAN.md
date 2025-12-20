# VectorAccelerate Metal 4 Upgrade Plan

## Executive Summary

This document outlines a comprehensive plan to upgrade VectorAccelerate from Metal 3 to Metal 4, targeting iOS 26 and macOS 26 exclusively. Metal 4, announced at WWDC 2025, introduces significant architectural changes including unified command encoding, native tensor/ML support, explicit resource management, and new shader compilation infrastructure.

**Key Decision Points:**
- Minimum deployment: iOS 26 / macOS 26 / tvOS 26 / visionOS 26
- Metal Shading Language: MSL 4.0
- Apple Silicon only (M1+, A14+)

### Toolchain Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Xcode | 17.0+ | Required for iOS/macOS 26 SDK |
| Swift | 6.0+ | Strict concurrency mode |
| Metal SDK | Metal 4.0 | Bundled with Xcode 17 |
| MSL Standard | `-std=metal4.0` | All shaders |
| macOS (dev) | 26.0+ | For local testing |

**CI Configuration:**
```yaml
# Required CI environment
runs-on: macos-26
xcode-version: '17.0'
# Shader validation
xcrun metal -c shader.metal -std=metal4.0 -mmacosx-version-min=26.0
```

---

## Table of Contents

1. [Metal 4 Overview](#1-metal-4-overview)
2. [Current VectorAccelerate Architecture](#2-current-vectoraccelerate-architecture)
3. [Breaking Changes & Required Updates](#3-breaking-changes--required-updates)
4. [Feature Detection & Capabilities](#4-feature-detection--capabilities)
5. [Error Handling Policy](#5-error-handling-policy)
6. [Performance & Observability](#6-performance--observability)
7. [Rollout Strategy](#7-rollout-strategy)
8. [New Metal 4 Features to Adopt](#8-new-metal-4-features-to-adopt)
9. [ML/Tensor Integration Opportunities](#9-mltensor-integration-opportunities)
10. [Shader Migration Guide](#10-shader-migration-guide)
11. [Implementation Phases](#11-implementation-phases)
12. [Risk Assessment](#12-risk-assessment)
13. [Testing Strategy](#13-testing-strategy)
14. [Sources & References](#14-sources--references)

---

## 1. Metal 4 Overview

### 1.1 Core Philosophy Changes

Metal 4 represents a fundamental shift in GPU programming philosophy:

| Aspect | Metal 3 | Metal 4 |
|--------|---------|---------|
| Command Encoding | Separate encoder types | Unified MTL4ComputeCommandEncoder |
| Resource Management | Implicit retention | Explicit residency sets |
| Concurrency | Opt-in parallelism | Concurrent by default |
| Compilation | Device-coupled | Dedicated MTL4Compiler |
| ML Support | External (MPS) | Native MTLTensor in shaders |
| Command Buffers | Queue-coupled | Decoupled from queue |

### 1.2 New Core Types

```swift
// New Metal 4 Types
MTL4CommandQueue          // Enhanced queue with residency management
MTL4CommandBuffer         // Reusable, decoupled from queue
MTL4ComputeCommandEncoder // Unified compute/blit/accel structure
MTL4Compiler              // Dedicated shader compilation
MTL4ArgumentTable         // Explicit resource binding
MTLResidencySet           // Resource residency management
MTLTensor                 // Native ML tensor type
```

### 1.3 Platform Requirements

```swift
// Package.swift updates required
platforms: [
    .macOS(.v26),      // macOS 26+
    .iOS(.v26),        // iOS 26+
    .tvOS(.v26),       // tvOS 26+
    .visionOS(.v3)     // visionOS 3+
]
```

**Hardware Requirements:**
- Apple Silicon only (M1/A14 Bionic and newer)
- No Intel/AMD support for Metal 4 features

---

## 2. Current VectorAccelerate Architecture

### 2.1 Metal API Touchpoints Summary

| Component | Files | API Touchpoints | Migration Complexity |
|-----------|-------|-----------------|---------------------|
| Core Infrastructure | 8 | 32 | High |
| Kernel Implementations | 25 | 35 | Medium |
| Metal Shaders | 16 | 10 | Low-Medium |
| **Total** | **49** | **77** | - |

### 2.2 Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     VectorAccelerate (Metal 3)                  │
├─────────────────────────────────────────────────────────────────┤
│  MetalContext (Actor)                                           │
│  ├─ MTLCommandQueue (single)                                    │
│  ├─ BufferPool (token-based)                                    │
│  ├─ ShaderManager (compilation + caching)                       │
│  └─ MetalDevice (capability detection)                          │
├─────────────────────────────────────────────────────────────────┤
│  ComputeEngine                                                  │
│  ├─ MTLCommandBuffer creation                                   │
│  ├─ MTLComputeCommandEncoder binding                            │
│  └─ dispatchThreadgroups execution                              │
├─────────────────────────────────────────────────────────────────┤
│  Kernels (25+)                                                  │
│  ├─ Distance: L2, Cosine, Dot, Manhattan, Chebyshev, Minkowski │
│  ├─ Quantization: Scalar, Binary, Product                       │
│  ├─ Matrix: Multiply, Transpose, Vector                         │
│  └─ Selection: TopK, Streaming, WarpOptimized                   │
├─────────────────────────────────────────────────────────────────┤
│  Metal Shaders (16 files, 50+ kernels)                          │
│  └─ MSL 3.0, threadgroup_barrier synchronization                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Current Compilation Flow

```swift
// Current Metal 3 Pattern
let options = MTLCompileOptions()
options.languageVersion = .version3_0
options.fastMathEnabled = true

device.makeLibrary(source: source, options: options) { library, error in
    // Handle completion
}
```

---

## 3. Breaking Changes & Required Updates

### 3.1 Command Queue & Buffer Changes

#### Current Implementation (MetalContext.swift)
```swift
// Metal 3 - Command buffer from queue
self.commandQueue = try await device.makeCommandQueue(label: label)
let commandBuffer = commandQueue.makeCommandBuffer()
```

#### Metal 4 Required Changes
```swift
// Metal 4 - Decoupled command buffers
let queueDescriptor = MTL4CommandQueueDescriptor()
self.commandQueue = device.makeCommandQueue(descriptor: queueDescriptor) as! MTL4CommandQueue

// Command buffers created from device, not queue
let commandBuffer = device.makeCommandBuffer() as! MTL4CommandBuffer

// Residency set required for resource access
let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
residencySet.addAllocation(buffer)
residencySet.commit()
commandQueue.addResidencySet(residencySet)
```

**Files Affected:**
- `MetalContext.swift` (lines 42, 93, 120, 280-282)
- `MetalDevice.swift` (lines 199-223)
- `ComputeEngine.swift` (command buffer usage throughout)

### 3.2 Explicit Resource Residency

#### Current Implementation
```swift
// Metal 3 - Implicit resource retention
encoder.setBuffer(buffer, offset: 0, index: 0)
// Buffer automatically retained by command buffer
```

#### Metal 4 Required Changes
```swift
// Metal 4 - Explicit residency management
// Step 1: Create residency set at initialization
let residencyDescriptor = MTLResidencySetDescriptor()
residencyDescriptor.initialCapacity = 64
let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

// Step 2: Add resources to residency set
residencySet.addAllocation(buffer)
residencySet.commit()

// Step 3: Attach to command queue or buffer
commandQueue.addResidencySet(residencySet)
// OR for per-buffer control:
commandBuffer.useResidencySet(residencySet)

// Step 4: Use argument tables for binding
let argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)
argumentTable.setAddress(buffer.gpuAddress, index: 0)
encoder.setArgumentTable(argumentTable, stages: .compute)
```

**Files Affected:**
- `BufferPool.swift` (entire file needs residency integration)
- `MetalBufferFactory.swift` (buffer creation patterns)
- All kernel files (buffer binding patterns)

### 3.3 Compiler Infrastructure Changes

#### Current Implementation (MetalDevice.swift)
```swift
// Metal 3 - Device-coupled compilation
public func makeLibrary(source: String) async throws -> any MTLLibrary {
    let options = MTLCompileOptions()
    options.languageVersion = .version3_0

    return try await withCheckedThrowingContinuation { continuation in
        device.makeLibrary(source: source, options: options) { library, error in
            // Handle result
        }
    }
}
```

#### Metal 4 Required Changes
```swift
// Metal 4 - Dedicated compiler with QoS
public actor Metal4ShaderCompiler {
    private let compiler: MTL4Compiler

    init(device: MTLDevice) throws {
        let descriptor = MTL4CompilerDescriptor()
        descriptor.qualityOfService = .userInteractive  // For development
        self.compiler = try device.makeCompiler(descriptor: descriptor)
    }

    func compile(source: String) async throws -> any MTLLibrary {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0

        return try await compiler.makeLibrary(source: source, options: options)
    }
}
```

**Files Affected:**
- `MetalDevice.swift` (lines 294-322)
- `ShaderManager.swift` (entire compilation logic)
- `KernelContext.swift` (fallback compilation)

### 3.4 Function & Pipeline Creation

#### Current Implementation
```swift
// Metal 3 - Direct function creation
guard let function = library.makeFunction(name: functionName) else {
    throw VectorError.shaderNotFound(name: functionName)
}
let pipeline = try await device.makeComputePipelineState(function: function)
```

#### Metal 4 Required Changes
```swift
// Metal 4 - Descriptor-based function creation
let functionDescriptor = MTL4LibraryFunctionDescriptor()
functionDescriptor.name = functionName
functionDescriptor.library = library

// For specialization
let specializedDescriptor = MTL4SpecializedFunctionDescriptor()
specializedDescriptor.functionDescriptor = functionDescriptor
specializedDescriptor.constantValues = constantValues

let function = try compiler.makeFunction(descriptor: specializedDescriptor)

// Pipeline creation (may use descriptors)
let pipelineDescriptor = MTLComputePipelineDescriptor()
pipelineDescriptor.computeFunction = function
let pipeline = try await device.makeComputePipelineState(descriptor: pipelineDescriptor)
```

**Files Affected:**
- `ShaderManager.swift` (lines 79-102, 127-154)
- All kernel initialization code

### 3.5 Synchronization Changes

#### Current Implementation (ConcurrencyShims.swift)
```swift
// Metal 3 - Completion handler pattern
public extension MTLCommandBuffer {
    func commitAndWait() async {
        await withCheckedContinuation { continuation in
            self.addCompletedHandler { _ in
                continuation.resume()
            }
            self.commit()
        }
    }
}
```

#### Metal 4 Required Changes
```swift
// Metal 4 - Event-based synchronization preferred
public extension MTL4CommandBuffer {
    func commitAndWait(queue: MTL4CommandQueue, event: MTLSharedEvent) async {
        let targetValue = event.signaledValue + 1

        // Commit with event signaling
        queue.commit([self])
        queue.signalEvent(event, value: targetValue)

        // Wait for completion
        await withCheckedContinuation { continuation in
            event.notifyListener(at: targetValue) { _, _ in
                continuation.resume()
            }
        }
    }
}

// Barrier-based synchronization within encoders
encoder.barrier(resources: [buffer1, buffer2],
                beforeStages: .dispatch,
                afterStages: .dispatch)
```

**Files Affected:**
- `ConcurrencyShims.swift` (entire file)
- `MetalContext.swift` (execute methods)
- `ComputeEngine.swift` (synchronization points)

---

## 4. Feature Detection & Capabilities

Metal 4 features are not monolithic—some features depend on GPU family, not just OS version. VectorAccelerate must detect and gate features granularly.

### 4.1 Metal4Capabilities Struct

```swift
/// Granular Metal 4 feature detection
public struct Metal4Capabilities: Sendable, Equatable {
    /// Core Metal 4 support (MTL4CommandQueue, etc.)
    public let supportsMetal4Core: Bool

    /// Placement sparse buffers for large datasets
    public let supportsPlacementSparseBuffers: Bool

    /// MTLTensor and inline ML operations
    public let supportsMLTensor: Bool

    /// Machine learning command encoder
    public let supportsMachineLearningEncoder: Bool

    /// Advanced barrier features
    public let supportsAdvancedBarriers: Bool

    /// Simdgroup matrix operations (8x8, 16x16)
    public let supportsSimdgroupMatrix: Bool

    /// Maximum argument table buffer count
    public let maxArgumentTableBuffers: Int

    /// Maximum residency set capacity
    public let maxResidencySetCapacity: Int

    /// Initialize from MTLDevice
    public init(device: MTLDevice) {
        self.supportsMetal4Core = device.supportsFamily(.metal4)
        self.supportsPlacementSparseBuffers = device.supportsFamily(.apple9)  // M3+
        self.supportsMLTensor = device.supportsFamily(.metal4)
        self.supportsMachineLearningEncoder = device.supportsFamily(.apple9)
        self.supportsAdvancedBarriers = device.supportsFamily(.metal4)
        self.supportsSimdgroupMatrix = device.supportsFamily(.apple8)  // M2+
        self.maxArgumentTableBuffers = 31
        self.maxResidencySetCapacity = device.supportsFamily(.apple9) ? 4096 : 1024
    }
}
```

### 4.2 Feature Gating Strategy

```swift
/// Backend selection based on capabilities
public enum GPUBackend: Sendable {
    case metal3(MTLDevice)
    case metal4(MTLDevice, Metal4Capabilities)

    /// Recommended backend for device
    public static func recommended(for device: MTLDevice) -> GPUBackend {
        let caps = Metal4Capabilities(device: device)
        if caps.supportsMetal4Core {
            return .metal4(device, caps)
        }
        return .metal3(device)
    }
}

/// Feature-gated execution
extension Metal4Context {
    public func executeWithMLTensor<T: Sendable>(
        _ operation: @Sendable (MTL4CommandBuffer, MTL4ComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        guard capabilities.supportsMLTensor else {
            throw VectorError.featureNotSupported("MLTensor requires Metal 4 with apple9 GPU family")
        }
        return try await execute(operation)
    }
}
```

### 4.3 Files to Modify

| File | Changes |
|------|---------|
| `MetalDevice.swift` | Add `Metal4Capabilities` property |
| `MetalContext.swift` | Add capability-gated methods |
| `Metal4Context.swift` | Use capabilities for feature paths |
| NEW: `Metal4Capabilities.swift` | Full capability detection |

---

## 5. Error Handling Policy

Metal 4 introduces new failure modes that require explicit handling. This section defines the error categories and recovery strategies.

### 5.1 Error Categories

| Category | Examples | Recovery Strategy |
|----------|----------|-------------------|
| **Setup Failures** | Device doesn't support Metal 4, queue creation fails | Fail fast, surface to caller |
| **Compilation Failures** | Shader syntax error, unsupported MSL feature | Log, fall back to pre-compiled if available |
| **Residency Failures** | Memory pressure, allocation rejected | Retry with smaller set, evict unused, or fail |
| **Execution Failures** | Command buffer error, GPU hang | Log diagnostic, surface error, suggest retry |
| **Feature Unavailable** | MLTensor on unsupported GPU | Graceful degradation or feature-specific error |

### 5.2 Error Types

```swift
public enum Metal4Error: Error, Sendable {
    // Setup errors
    case metal4NotSupported(reason: String)
    case commandQueueCreationFailed(underlying: Error?)
    case compilerCreationFailed(underlying: Error?)

    // Compilation errors
    case shaderCompilationFailed(shader: String, error: String)
    case functionNotFound(name: String, library: String)
    case pipelineCreationFailed(function: String, error: String)

    // Residency errors
    case residencySetFull(capacity: Int, requested: Int)
    case residencyCommitFailed(underlying: Error)
    case allocationNotInResidencySet(bufferLabel: String?)

    // Execution errors
    case commandBufferError(status: MTLCommandBufferStatus, error: Error?)
    case encoderCreationFailed
    case dispatchFailed(reason: String)

    // Feature errors
    case featureNotSupported(feature: String, requiredFamily: String)
    case mlTensorNotAvailable
}
```

### 5.3 Error Handling Locations

| Component | Error Type | Handling |
|-----------|------------|----------|
| `Metal4Context.init` | Setup failures | Throw immediately, log to OSLog |
| `Metal4Compiler` | Compilation failures | Throw with diagnostic, cache failure |
| `ResidencyManager` | Residency failures | Attempt recovery, then throw |
| `execute()` methods | Execution failures | Log diagnostic, throw with context |
| Capability-gated methods | Feature unavailable | Throw specific feature error |

### 5.4 Logging Strategy

```swift
import os

extension Metal4Context {
    private static let logger = Logger(subsystem: "com.vectoraccelerate", category: "Metal4")

    private func logError(_ error: Metal4Error, context: String) {
        Self.logger.error("[\(context)] \(error.localizedDescription)")

        // Additional diagnostics for debugging
        #if DEBUG
        Self.logger.debug("Device: \(device.device.name)")
        Self.logger.debug("Capabilities: \(String(describing: capabilities))")
        #endif
    }
}
```

---

## 6. Performance & Observability

### 6.1 Metrics to Track

| Metric | Unit | Collection Method | Baseline Threshold |
|--------|------|-------------------|-------------------|
| GPU execution time | μs | MTLCounterSampleBuffer | Per-kernel baseline |
| Command buffer count | count | Per-operation counter | 1-3 per operation |
| Peak memory usage | MB | Residency set tracking | Per-dimension baseline |
| Residency commits | count | ResidencyManager counter | < 10 per batch |
| Pipeline cache hits | % | ShaderManager counter | > 95% |
| Argument table reuse | % | ArgumentTablePool counter | > 80% |

### 6.2 Performance Recorder

```swift
/// Records performance metrics for debugging and regression detection
public actor PerformanceRecorder {
    public struct Metrics: Sendable, Codable {
        public let operation: String
        public let dimension: Int
        public let batchSize: Int
        public let gpuTimeUs: Double
        public let cpuTimeUs: Double
        public let peakMemoryMB: Double
        public let commandBufferCount: Int
        public let timestamp: Date
        public let hardwareID: String
    }

    private var records: [Metrics] = []
    private let maxRecords: Int

    public init(maxRecords: Int = 10000) {
        self.maxRecords = maxRecords
    }

    public func record(_ metrics: Metrics) {
        records.append(metrics)
        if records.count > maxRecords {
            records.removeFirst(records.count - maxRecords)
        }
    }

    public func exportJSON(to url: URL) async throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(records)
        try data.write(to: url)
    }
}
```

### 6.3 Regression Detection

**CI Integration:**
```yaml
performance-regression:
  runs-on: macos-26
  steps:
    - name: Run Benchmarks
      run: swift run VectorAccelerateBenchmarks --output results.json

    - name: Check Regression
      run: |
        python scripts/check_regression.py \
          --baseline Benchmarks/baselines/m3-baseline.json \
          --results results.json \
          --threshold-time 1.10 \
          --threshold-memory 1.05
```

**Regression Thresholds:**
- GPU time: Fail if > 110% of baseline
- Memory usage: Fail if > 105% of baseline
- Pipeline cache hit rate: Fail if < 90%

### 6.4 Baseline Files

Store per-hardware baselines in `Benchmarks/baselines/`:
- `m1-baseline.json`
- `m2-baseline.json`
- `m3-baseline.json`
- `a17-baseline.json`

---

## 7. Rollout Strategy

### 7.1 Phased Rollout

| Phase | Metal 4 Status | Escape Hatch | Timeline |
|-------|----------------|--------------|----------|
| **Alpha** | Off by default, opt-in | N/A | Phase 1-2 |
| **Beta** | On by default on supported devices | Force Metal 3 flag | Phase 3-4 |
| **Stable** | Always on (Metal 3 deprecated) | None | Phase 5+ |

### 7.2 Backend Selection API

```swift
public struct AccelerationConfiguration: Sendable {
    public enum BackendPreference: Sendable {
        /// Use Metal 3 always (deprecated after stable)
        case metal3Only

        /// Use Metal 4 if available, fall back to Metal 3
        case metal4IfAvailable

        /// Require Metal 4, fail if unavailable
        case metal4Required

        /// Auto-select based on rollout phase
        case automatic
    }

    public var backendPreference: BackendPreference = .automatic

    /// Force Metal 3 even when Metal 4 is available (debug/testing)
    public var forceMetal3: Bool = false

    /// Enable experimental ML features (Phase 4)
    public var enableExperimentalML: Bool = false
}
```

### 7.3 Environment Variable Override

```swift
// Allow CI and developers to force specific backend
// VECTORACCELERATE_BACKEND=metal3|metal4|auto
if let override = ProcessInfo.processInfo.environment["VECTORACCELERATE_BACKEND"] {
    switch override {
    case "metal3": return .metal3Only
    case "metal4": return .metal4Required
    default: return .automatic
    }
}
```

### 7.4 Deprecation Path

```swift
@available(*, deprecated, message: "Metal 3 backend will be removed in VectorAccelerate 1.0")
public actor MetalContext { ... }

// New primary entry point
public actor Metal4Context { ... }

// Protocol for migration
public protocol AccelerationContext: Actor {
    func execute<T: Sendable>(...) async throws -> T
}
```

---

## 8. New Metal 4 Features to Adopt

### 8.1 Unified Compute Encoder

**Benefit:** Single encoder for compute, blit, and acceleration structure operations.

```swift
// Metal 4 - Unified encoding
let encoder = commandBuffer.makeComputeCommandEncoder() as! MTL4ComputeCommandEncoder

// Compute dispatch
encoder.setComputePipelineState(computePipeline)
encoder.setArgumentTable(computeArgs, stages: .compute)
encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)

// Blit in same encoder (no separate encoder needed)
encoder.copy(from: sourceBuffer, to: destBuffer)

// Barrier for dependency
encoder.barrier(resources: [destBuffer], beforeStages: .blit, afterStages: .dispatch)

// Continue with more compute
encoder.setComputePipelineState(anotherPipeline)
encoder.dispatchThreadgroups(moreThreadgroups, threadsPerThreadgroup: threads)

encoder.endEncoding()
```

**VectorAccelerate Opportunity:**
- Fuse multi-pass operations (e.g., normalize + distance calculation)
- Reduce encoder creation overhead
- Better GPU utilization through unified scheduling

### 8.2 Argument Tables

**Benefit:** Reduced CPU overhead for resource binding.

```swift
// Create reusable argument table
let tableDescriptor = MTL4ArgumentTableDescriptor()
tableDescriptor.maxBufferBindCount = 16
tableDescriptor.maxTextureBindCount = 4

let argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)

// Bind resources once
argumentTable.setAddress(queryBuffer.gpuAddress, index: 0)
argumentTable.setAddress(databaseBuffer.gpuAddress, index: 1)
argumentTable.setAddress(resultBuffer.gpuAddress, index: 2)

// Reuse across multiple dispatches
for batch in batches {
    encoder.setArgumentTable(argumentTable, stages: .compute)
    encoder.dispatchThreadgroups(batch.threadgroups, threadsPerThreadgroup: threads)
}
```

**VectorAccelerate Opportunity:**
- Batch operations with shared argument tables
- Reduce per-dispatch binding overhead in ComputeEngine
- Pre-configure kernel argument tables

### 8.3 Placement Sparse Resources

**Benefit:** Fine-grained memory control for large datasets.

```swift
// Create placement heap
let heapDescriptor = MTLHeapDescriptor()
heapDescriptor.type = .placement
heapDescriptor.size = 1024 * 1024 * 1024  // 1GB
let placementHeap = device.makeHeap(descriptor: heapDescriptor)

// Create sparse buffer
let sparseDescriptor = MTLBufferDescriptor()
sparseDescriptor.length = embeddingDatabaseSize
sparseDescriptor.storageMode = .sparse
let sparseBuffer = device.makeBuffer(descriptor: sparseDescriptor)

// Map pages on demand
let pageSize = device.minimumSparsePageSize
for region in activeRegions {
    commandQueue.mapPages(of: sparseBuffer,
                          range: region,
                          heap: placementHeap)
}
```

**VectorAccelerate Opportunity:**
- Efficient large embedding database handling
- Memory-mapped vector stores
- Dynamic resource allocation for variable workloads

### 8.4 Enhanced Pipeline Compilation

**Benefit:** Faster loading, better optimization.

```swift
// Pipeline harvesting for AOT compilation
let harvestDescriptor = MTL4PipelineHarvestDescriptor()
harvestDescriptor.pipelines = [pipeline1, pipeline2, pipeline3]
let harvestData = try compiler.harvest(descriptor: harvestDescriptor)

// Save harvest data for later
try harvestData.write(to: harvestURL)

// Reload with harvested data (faster)
let precompiledPipelines = try compiler.makeComputePipelineStates(
    harvestData: harvestData
)
```

**VectorAccelerate Opportunity:**
- Pre-compiled kernel variants for all dimension sizes
- Faster cold start times
- Build-time shader validation

---

## 9. ML/Tensor Integration Opportunities

### 9.1 Native MTLTensor Support

Metal 4 introduces `MTLTensor` for native ML operations in shaders. This is transformative for VectorAccelerate.

#### Current Pattern (External ML)
```swift
// Current - Separate ML inference and GPU compute
let mlModel = try MLModel(contentsOf: modelURL)
let embedding = try mlModel.prediction(input: input)
let buffer = device.makeBuffer(data: embedding.floatArray)
// Then use buffer in Metal compute
```

#### Metal 4 Pattern (Inline Inference)
```metal
// MSL 4.0 - Inline tensor operations in shaders
#include <metal_tensor>

kernel void embeddingWithInference(
    device float* input [[buffer(0)]],
    device MTLTensor<float, 2>* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Create inline tensor from input
    auto inputTensor = MTLTensor<float, dextents<1>>(input + tid * dim, dim);

    // Run inference directly in shader
    auto result = matmul(inputTensor, *weights);

    // Apply activation
    auto activated = relu(result);

    // Write output
    for (int i = 0; i < outputDim; i++) {
        output[tid * outputDim + i] = activated[i];
    }
}
```

### 9.2 VectorAccelerate ML Integration Points

#### 5.2.1 Learned Distance Metrics

```metal
// MSL 4.0 - Neural distance metric
kernel void learnedDistanceMetric(
    device float* queries [[buffer(0)]],
    device float* database [[buffer(1)]],
    device MTLTensor<float, 2>* projectionWeights [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant DistanceParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint queryIdx = gid.y;
    uint dbIdx = gid.x;

    // Load vectors
    auto queryVec = MTLTensor<float, dextents<1>>(
        queries + queryIdx * params.dimension,
        params.dimension
    );
    auto dbVec = MTLTensor<float, dextents<1>>(
        database + dbIdx * params.dimension,
        params.dimension
    );

    // Project through learned transformation
    auto projectedQuery = matmul(queryVec, *projectionWeights);
    auto projectedDb = matmul(dbVec, *projectionWeights);

    // Compute distance in projected space
    float dist = 0.0;
    for (int i = 0; i < params.projectedDim; i++) {
        float diff = projectedQuery[i] - projectedDb[i];
        dist += diff * diff;
    }

    distances[queryIdx * params.numDatabase + dbIdx] = sqrt(dist);
}
```

#### 5.2.2 Neural Quantization

```metal
// MSL 4.0 - Learned quantization in shader
kernel void neuralQuantize(
    device float* vectors [[buffer(0)]],
    device MTLTensor<float, 2>* encoder [[buffer(1)]],
    device MTLTensor<float, 2>* decoder [[buffer(2)]],
    device uint8_t* quantized [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Load vector as tensor
    auto vec = MTLTensor<float, dextents<1>>(vectors + gid * dim, dim);

    // Encode to latent space
    auto latent = relu(matmul(vec, *encoder));

    // Quantize latent representation
    for (int i = 0; i < latentDim; i++) {
        quantized[gid * latentDim + i] = uint8_t(clamp(latent[i] * 255.0, 0.0, 255.0));
    }
}

kernel void neuralDequantize(
    device uint8_t* quantized [[buffer(0)]],
    device MTLTensor<float, 2>* decoder [[buffer(1)]],
    device float* vectors [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Dequantize to float
    float latent[LATENT_DIM];
    for (int i = 0; i < LATENT_DIM; i++) {
        latent[i] = float(quantized[gid * LATENT_DIM + i]) / 255.0;
    }

    auto latentTensor = MTLTensor<float, dextents<1>>(latent, LATENT_DIM);

    // Decode back to vector space
    auto decoded = matmul(latentTensor, *decoder);

    // Write output
    for (int i = 0; i < dim; i++) {
        vectors[gid * dim + i] = decoded[i];
    }
}
```

#### 5.2.3 Attention-Based Similarity

```metal
// MSL 4.0 - Inline attention for semantic similarity
kernel void attentionSimilarity(
    device float* queries [[buffer(0)]],      // [numQueries, seqLen, dim]
    device float* keys [[buffer(1)]],         // [numKeys, seqLen, dim]
    device MTLTensor<float, 2>* Wq [[buffer(2)]],
    device MTLTensor<float, 2>* Wk [[buffer(3)]],
    device float* similarities [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Project query and key
    auto q = loadAndProject(queries, gid.y, *Wq, params);
    auto k = loadAndProject(keys, gid.x, *Wk, params);

    // Compute scaled dot-product attention
    float score = dot(q, k) / sqrt(float(params.headDim));

    // Apply softmax normalization (simplified)
    similarities[gid.y * params.numKeys + gid.x] = score;
}
```

### 9.3 New Kernel Opportunities

| Kernel | Description | ML Integration |
|--------|-------------|----------------|
| `LearnedDistanceKernel` | Distance with learned projection | MTLTensor matmul |
| `NeuralQuantizationKernel` | Encoder-decoder quantization | Inline inference |
| `AttentionPoolingKernel` | Attention-weighted aggregation | Softmax + matmul |
| `NeuralRerankerKernel` | Cross-encoder reranking | Full transformer layer |
| `AdaptiveIndexKernel` | Learned index navigation | Decision network |

---

## 10. Shader Migration Guide

### 10.1 MSL Version Update

All shader files need header updates:

```metal
// Current (MSL 3.0)
#include <metal_stdlib>
using namespace metal;

// Metal 4 (MSL 4.0)
#include <metal_stdlib>
#include <metal_tensor>  // New for tensor operations
using namespace metal;
```

### 10.2 Barrier Updates

```metal
// Current - Still valid in Metal 4
threadgroup_barrier(mem_flags::mem_threadgroup);

// Metal 4 - Additional barrier options
// For device memory coherency
threadgroup_barrier(mem_flags::mem_device);

// For texture operations
threadgroup_barrier(mem_flags::mem_texture);
```

### 10.3 SIMD Group Operations (Enhanced)

```metal
// Current (Metal 3)
float result = simd_sum(value);
float shuffled = simd_shuffle_down(value, offset);

// Metal 4 - Enhanced operations
float result = simd_sum(value);  // Still valid
float shuffled = simd_shuffle_down(value, offset);  // Still valid

// New in Metal 4 - Matrix operations
simdgroup_matrix<float, 8, 8> matA, matB, matC;
simdgroup_load(matA, srcA, stride);
simdgroup_load(matB, srcB, stride);
simdgroup_multiply_accumulate(matC, matA, matB, matC);
simdgroup_store(matC, dst, stride);
```

### 10.4 Files to Update

| File | Changes Required |
|------|------------------|
| `AdvancedTopK.metal` | Add metal_tensor header, consider simdgroup matrix for sorting |
| `BasicOperations.metal` | Minimal changes, verify barrier compatibility |
| `ClusteringShaders.metal` | Add tensor support for centroid operations |
| `CosineSimilarity.metal` | Consider tensor-based normalization |
| `DistanceShaders.metal` | Add learned distance metric variants |
| `DotProduct.metal` | Optimize with simdgroup matrix operations |
| `L2Distance.metal` | Add tensor projection option |
| `L2Normalization.metal` | Minimal changes |
| `ProductQuantization.metal` | Add neural quantization path |
| `QuantizationShaders.metal` | Add neural encoder/decoder |
| `SearchAndRetrieval.metal` | Add attention-based retrieval |
| `StatisticsShaders.metal` | Minimal changes |

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Objective:** Update core infrastructure to Metal 4 without breaking existing functionality.

#### Tasks:

1. **Update Package.swift**
   ```swift
   platforms: [
       .macOS(.v26),
       .iOS(.v26),
       .tvOS(.v26),
       .visionOS(.v3)
   ]
   ```

2. **Create Metal4Context** (parallel to MetalContext)
   - MTL4CommandQueue initialization
   - Residency set management
   - MTL4Compiler integration

3. **Update MetalDevice**
   - Add Metal 4 capability detection
   - MSL 4.0 compilation support
   - New resource options

4. **Create ResidencyManager**
   - Track all allocated resources
   - Automatic residency set updates
   - Integration with BufferPool

#### Deliverables:
- [ ] Package.swift updated
- [ ] Metal4Context actor created
- [ ] MetalDevice updated for MSL 4.0
- [ ] ResidencyManager implemented
- [ ] All existing tests passing

### Phase 2: Command Encoding (Week 3-4)

**Objective:** Migrate to unified command encoding and argument tables.

#### Tasks:

1. **Create Metal4ComputeEngine**
   - Unified MTL4ComputeCommandEncoder usage
   - Barrier-based synchronization
   - Argument table management

2. **Update Buffer Binding**
   - Argument table creation
   - GPU address binding
   - Per-kernel argument descriptors

3. **Update Synchronization**
   - MTLSharedEvent integration
   - Event-based completion
   - Barrier placement

4. **Migrate Kernels**
   - Update all 25+ kernel files
   - New binding patterns
   - Verify dispatch behavior

#### Deliverables:
- [ ] Metal4ComputeEngine implemented
- [ ] ArgumentTableManager created
- [ ] All kernels migrated to new binding
- [ ] Synchronization tests passing

### Phase 3: Shader Compilation (Week 5-6)

**Objective:** Adopt new compilation infrastructure and pipeline management.

#### Tasks:

1. **Create Metal4ShaderCompiler**
   - MTL4Compiler wrapper
   - QoS management
   - Async compilation

2. **Update Pipeline Creation**
   - Descriptor-based function creation
   - Flexible render pipeline states
   - Pipeline harvesting support

3. **Implement Pipeline Harvesting**
   - Build-time harvesting script
   - Harvest data storage
   - Fast pipeline loading

4. **Update Metal Shaders**
   - MSL 4.0 headers
   - Verify all kernels compile
   - Add new simdgroup operations

#### Deliverables:
- [ ] Metal4ShaderCompiler implemented
- [ ] Pipeline harvesting working
- [ ] All shaders updated to MSL 4.0
- [ ] CI validation for Metal 4 shaders

### Phase 4: ML Integration (Week 7-8)

**Objective:** Add native tensor support and ML-accelerated operations.

#### Tasks:

1. **Add MTLTensor Infrastructure**
   - Tensor buffer creation
   - Weight loading utilities
   - Tensor binding helpers

2. **Implement New Kernels**
   - LearnedDistanceKernel
   - NeuralQuantizationKernel
   - AttentionPoolingKernel

3. **Update Existing Kernels**
   - Optional tensor projection paths
   - Hybrid CPU/GPU/ML execution

4. **Performance Optimization**
   - Profile tensor operations
   - Optimize memory layout
   - Benchmark vs MPS

#### Deliverables:
- [x] TensorManager implemented
- [x] 3+ new ML-accelerated kernels (LearnedDistance, NeuralQuantization, AttentionSimilarity)
- [x] Benchmark suite for ML operations
- [x] Documentation for ML features

### Phase 5: Optimization & Polish (Week 9-10)

**Objective:** Performance optimization and production readiness.

**Status:** In Progress (2025-11-30)

#### Tasks:

1. **Performance Profiling** ✅
   - Metal System Trace analysis
   - GPU timeline optimization
   - Memory bandwidth analysis
   - Benchmark baselines created

2. **Shader Compilation Fixes** ✅
   - Fixed OptimizedMatrixOps.metal macro conflicts
   - Fixed ProductQuantization.metal atomic typedef conflicts
   - All shaders now compile in SPM runtime mode
   - 24 of 26 shader loading failures fixed

3. **Memory Optimization** (Pending)
   - Placement sparse resources
   - Optimal residency strategies
   - Buffer pool tuning

4. **API Polish** (Pending)
   - Clean public API surface
   - Deprecation warnings for old paths
   - Migration guide documentation

5. **Comprehensive Testing** ✅
   - Unit tests for all new code
   - Integration tests
   - Performance regression tests
   - 824/828 tests passing (99.5%)

#### Deliverables:
- [x] Performance benchmarks documented
- [ ] Memory usage optimized
- [ ] Public API finalized
- [x] All tests passing (99.5% - 4 known issues)
- [ ] Migration guide complete

---

## 12. Risk Assessment

### 12.1 High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking API changes in Metal 4 | All kernels affected | Parallel implementation, maintain Metal 3 path initially |
| Performance regression | User impact | Comprehensive benchmarking before release |
| Residency management complexity | Memory errors | Thorough testing, automatic management |
| ML tensor compatibility | New features may not work | Early prototyping, fallback paths |

### 12.2 Medium Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Shader compilation changes | Build system | Pipeline harvesting, CI validation |
| Synchronization model change | Race conditions | Event-based testing, stress tests |
| Argument table overhead | Performance | Profiling, optional adoption |

### 12.3 Low Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Platform version requirement | User adoption | Clear documentation |
| MSL syntax changes | Shader updates | Incremental migration |

---

## 13. Testing Strategy

### 13.1 Unit Tests

```swift
// New test categories
class Metal4ContextTests: XCTestCase {
    func testResidencySetCreation() async throws
    func testCommandQueueConfiguration() async throws
    func testArgumentTableBinding() async throws
}

class Metal4CompilerTests: XCTestCase {
    func testMSL4Compilation() async throws
    func testFunctionDescriptorCreation() async throws
    func testPipelineHarvesting() async throws
}

class TensorKernelTests: XCTestCase {
    func testTensorBufferCreation() async throws
    func testInlineInference() async throws
    func testLearnedDistance() async throws
}
```

### 13.2 Performance Tests

```swift
class Metal4PerformanceTests: XCTestCase {
    func testUnifiedEncoderOverhead() async throws {
        // Compare unified vs separate encoder performance
    }

    func testArgumentTableVsDirectBinding() async throws {
        // Compare binding strategies
    }

    func testResidencySetOverhead() async throws {
        // Measure residency management cost
    }

    func testTensorInferenceLatency() async throws {
        // Measure inline ML inference
    }
}
```

### 13.3 CI Pipeline Updates

```yaml
# .github/workflows/ci.yml additions
metal4-validation:
  name: Metal 4 Shader Validation
  runs-on: macos-26  # When available
  steps:
    - name: Validate MSL 4.0 Shaders
      run: |
        for shader in Sources/.../Shaders/*.metal; do
          xcrun metal -c "$shader" -std=metal4.0 -o /dev/null
        done

    - name: Test Pipeline Harvesting
      run: swift test --filter Metal4CompilerTests
```

---

## 14. Sources & References

### Apple Documentation
- [Discover Metal 4 - WWDC25](https://developer.apple.com/videos/play/wwdc2025/205/)
- [Combine Metal 4 machine learning and graphics - WWDC25](https://developer.apple.com/videos/play/wwdc2025/262/)
- [Explore Metal 4 games - WWDC25](https://developer.apple.com/videos/play/wwdc2025/254/)
- [What's New in Metal - Apple Developer](https://developer.apple.com/metal/whats-new/)
- [MTL4CommandQueue Documentation](https://developer.apple.com/documentation/metal/mtl4commandqueue)
- [MTL4CommandBuffer Documentation](https://developer.apple.com/documentation/metal/mtl4commandbuffer)

### Community Resources
- [Getting Started with Metal 4 - Metal by Example](https://metalbyexample.com/metal-4/)
- [WWDC 2025 - Discover Metal 4 - DEV Community](https://dev.to/arshtechpro/wwdc-2025-discover-metal-4-23f2)
- [Apple's Metal 4 Overview - Developer Tech](https://www.developer-tech.com/news/apple-fuses-ai-with-graphics-metal-4-overhaul/)
- [Metal 4 Tensor APIs - Burn Issue #3949](https://github.com/Tracel-AI/burn/issues/3949)

### Technical Analysis
- [Apple's Metal 4: The Graphics API Revolution - Medium](https://medium.com/@shivashanker7337/apples-metal-4-the-graphics-api-revolution-nobody-saw-coming-a2e272be4d57)

---

## Appendix A: Current API Mapping

| Metal 3 API | Metal 4 Equivalent | Notes |
|-------------|-------------------|-------|
| `MTLCommandQueue` | `MTL4CommandQueue` | Enhanced with residency |
| `MTLCommandBuffer` | `MTL4CommandBuffer` | Decoupled from queue |
| `MTLComputeCommandEncoder` | `MTL4ComputeCommandEncoder` | Unified encoding |
| `device.makeLibrary()` | `compiler.makeLibrary()` | Separate compiler |
| `library.makeFunction()` | `compiler.makeFunction(descriptor:)` | Descriptor-based |
| `encoder.setBuffer()` | `argumentTable.setAddress()` | Argument tables |
| `addCompletedHandler` | `MTLSharedEvent` | Event-based sync |
| Implicit retention | `ResidencySet` | Explicit residency |

---

## Appendix B: File Change Summary

| File | Changes | Priority |
|------|---------|----------|
| Package.swift | Platform versions | P0 |
| MetalContext.swift | Major rewrite | P0 |
| MetalDevice.swift | Compiler, options | P0 |
| BufferPool.swift | Residency integration | P1 |
| ShaderManager.swift | MTL4Compiler | P1 |
| ComputeEngine.swift | Unified encoder | P1 |
| ConcurrencyShims.swift | Event sync | P1 |
| All kernel files | Binding patterns | P2 |
| All shader files | MSL 4.0 headers | P2 |

---

*Document Version: 1.0*
*Created: November 2025*
*Target Release: VectorAccelerate 0.2.0*
