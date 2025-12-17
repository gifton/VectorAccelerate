# VectorAccelerate 0.3.0 Release Notes

**Release Date:** December 2025
**Swift:** 6.0+
**Platforms:** macOS 26.0+, iOS 26.0+, tvOS 26.0+, visionOS 3.0+

---

## Overview

VectorAccelerate 0.3.0 introduces a complete GPU-first vector index implementation, exposes low-level kernel primitives for custom ML pipelines, and delivers comprehensive Swift 6 strict concurrency compliance.

This release establishes a **two-layer architecture**:
1. **High-Level API**: `AcceleratedVectorIndex` for turnkey vector search
2. **Low-Level API**: Direct GPU kernel access for custom workflows

---

## What's New

### AcceleratedVectorIndex — GPU-First Vector Search

A complete, actor-based GPU vector index with automatic GPU/CPU routing:

```swift
// Create a flat index for 768-dimensional embeddings
let index = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768, capacity: 10_000)
)

// Insert vectors (returns stable handle)
let handle = try await index.insert(embedding)

// Search for similar vectors
let results = try await index.search(query: queryVector, k: 10)
```

**Features:**
- **Flat Index**: Sub-millisecond search (0.30ms @ 5K vectors, 128D)
- **IVF Index**: GPU-accelerated approximate nearest neighbor with K-means clustering
- **VectorHandle**: Generation-based opaque handles that survive compaction
- **Filtered Search**: Iterative fetch strategy for predicate-based filtering
- **Batch Operations**: ~21K vectors/sec insert throughput (128D)
- **Lazy Deletion**: Mark-for-delete with background compaction

### Low-Level Kernel API Exposure

Direct access to 24 GPU kernel primitives for custom ML pipelines:

**Distance Kernels:**
- `L2DistanceKernel` — Euclidean distance with dimension-optimized paths (384, 512, 768, 1536D)
- `CosineSimilarityKernel` — Cosine similarity/distance with optional normalization
- `DotProductKernel` — Inner product computation
- `MinkowskiDistanceKernel` — Generalized Lp distance (L1, L2, L∞)
- `HammingDistanceKernel` — Binary vector distance
- `JaccardDistanceKernel` — Set similarity

**Selection Kernels:**
- `TopKSelectionKernel` — Batch top-k from distance matrices
- `FusedL2TopKKernel` — Single-pass L2 + top-k (memory efficient)
- `StreamingTopKKernel` — Streaming top-k for large datasets
- `WarpOptimizedSelectionKernel` — SIMD group cooperative selection

**Quantization Kernels:**
- `ScalarQuantizationKernel` — INT8/INT4 quantization (4-8x compression)
- `BinaryQuantizationKernel` — 1-bit quantization (32x compression)
- `ProductQuantizationKernel` — PQ with learned codebooks (32-64x)
- `NeuralQuantizationKernel` — Learned neural compression

**Matrix Kernels:**
- `MatrixMultiplyKernel` — Tiled GEMM with simdgroup operations
- `MatrixVectorKernel` — Optimized mat-vec multiply
- `MatrixTransposeKernel` — Tiled transpose
- `BatchMatrixKernel` — Batched matrix operations

**Utility Kernels:**
- `StatisticsKernel` — Mean, variance, min, max, histogram
- `L2NormalizationKernel` — Vector normalization
- `ElementwiseKernel` — Element-wise operations
- `ParallelReductionKernel` — GPU reduction primitives
- `HistogramKernel` — Distribution analysis

### KernelDistanceProviders — VectorCore Integration

Seven new `DistanceProvider` implementations backed by Metal4 kernels:

```swift
// Create a GPU-backed distance provider
let provider = try await L2KernelDistanceProvider()

// Use with VectorCore APIs
let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)
let batch = try await provider.batchDistance(from: query, to: candidates, metric: .euclidean)
```

**Available Providers:**
- `L2KernelDistanceProvider`
- `CosineKernelDistanceProvider`
- `DotProductKernelDistanceProvider`
- `MinkowskiKernelDistanceProvider`
- `JaccardKernelDistanceProvider`
- `HammingKernelDistanceProvider`
- `UniversalKernelDistanceProvider` (auto-selects kernel by metric)

---

## Breaking Changes

None. This release is additive to 0.2.0.

---

## Swift 6 Strict Concurrency Compliance

This release achieves full Swift 6 strict concurrency compliance:

### TensorManager Refactored
- Converted from `actor` to `final class: @unchecked Sendable`
- Thread safety via `NSLock` for mutable state
- Methods are now synchronous (removed unnecessary `async`)

### Actor Isolation Fixes
- `Metal4Context`: Fixed completion handler capturing for `completionEvent`
- `ComputeEngine`: Added `nonisolated` cached `commandQueue` for cross-actor access
- `MetalContext`: Added `getCommandQueue()` for safe cross-actor queue sharing

### Command Buffer Patterns
- Fixed `commandBuffer.completed()` → `withCheckedContinuation` + `addCompletedHandler`
- Completion handlers registered BEFORE `commit()` to avoid race conditions

### Safe Pointer Handling
- `KernelProtocol`: `withUnsafeBytes(of:)` pattern for generic parameter encoding

---

## CI/CD Modernization

- **macOS 26 Runners**: All jobs now run on `macos-26`
- **Xcode 26**: Using `latest-stable` for Swift 6.2+ toolchain
- **Build Reliability**: Added `set -eo pipefail` for proper error propagation
- **Metal Shader Validation**: Compile-time shader verification

---

## Performance

| Operation | Dimension | Throughput/Latency |
|-----------|-----------|-------------------|
| Flat Insert | 128D | 21,866 vec/s |
| Flat Insert | 768D | 3,670 vec/s |
| Flat Search | 128D | 0.30ms (5K vectors) |
| Flat Search | 768D | 0.73ms (5K vectors) |
| IVF Search | 128D | 0.21ms (500 vectors) |

---

## Bug Fixes

- **Memory Pressure Handling**: Fixed test assertion that incorrectly required buffer retention under memory pressure
- **K-means Pipeline**: Removed unused dimension variable warning
- **IVF Training**: Fixed off-by-one error in DeletionMask iteration

---

## Test Suite

**826 tests** covering:
- Flat index search edge cases
- Handle lifecycle and generation tracking
- IVF training and search
- Kernel consumer integration
- Performance regression benchmarks
- Validation and error handling

---

## New Files

### Source Files
- `Sources/VectorAccelerate/VectorAccelerate.swift` — Module documentation and type aliases
- `Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift` — VectorCore providers
- `Sources/VectorAccelerate/Index/**` — Complete index implementation
- `Sources/VectorAccelerateBenchmarks/KernelUsageExamples.swift` — Usage examples

### Test Files
- `FlatIndexSearchTests.swift`
- `HandleLifecycleTests.swift`
- `IntrospectionTests.swift`
- `IVFTests.swift`
- `ValidationEdgeCaseTests.swift`
- `PerformanceBenchmarks.swift`
- `KernelConsumerTests.swift`
- `KernelDistanceProviderTests.swift`

### Documentation
- `docs/CLEAN_API_REDESIGN.md` — Architecture documentation

---

## Removed Files

- `EmbeddingEngine.swift` — Replaced by kernel-based approach
- `MinkowskiCalculator.swift` — Consolidated into kernel
- Legacy test files and disabled tests

---

## Migration Guide

### From 0.2.0

No breaking changes. New features are additive.

**To use the new vector index:**
```swift
// Add to your code
let index = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768)
)
```

**To use kernel providers with VectorCore:**
```swift
// Instead of custom distance implementations
let provider = try await L2KernelDistanceProvider()
```

---

## Contributors

- GPU index implementation and kernel API exposure
- Swift 6 concurrency compliance
- CI modernization for macOS 26

---

## Full Changelog

See `CHANGELOG.md` for detailed change history.
