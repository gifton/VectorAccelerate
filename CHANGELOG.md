# Changelog

All notable changes to VectorAccelerate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.6] - 2026-01-10

### Fixed

- **Metal Library Loading for SPM Transitive Dependencies** — Fixed a critical bug where `device.makeDefaultLibrary()` would return the host app's metallib instead of VectorAccelerate's bundled metallib when used as a transitive SPM dependency (e.g., `App -> PackageB -> VectorAccelerate`)
  - Added `isVectorAccelerateLibrary()` validation to verify loaded libraries contain expected shader functions
  - Changed loading order to prioritize VectorAccelerate's bundle before falling back to `makeDefaultLibrary()`
  - `Metal4ShaderCompiler.getDefaultLibrary()` now delegates to `KernelContext.getSharedLibrary()` for consistent behavior
  - Affected kernels: All GPU kernels (L2 normalization, distance calculations, cosine similarity, etc.)

### Changed

- Updated library version references to 0.3.6

---

## [0.3.5] - 2026-01-09

### Added

- **GPUHealthMonitor** — New actor for tracking GPU health and managing automatic CPU fallbacks
  - Per-operation failure tracking with configurable thresholds
  - Degradation levels: none → minor → moderate → severe
  - Automatic operation disabling after repeated failures
  - Time-based recovery with configurable disable duration
  - Configurable presets: default, aggressive, lenient
  - Thread-safe actor-based design consistent with VA patterns

### Changed

- Updated library version references to 0.3.5

---

## [0.3.0] - 2025-12-05

### Added

- **VectorIndexAcceleration Module** - New GPU-first vector index implementation
  - `AcceleratedVectorIndex` — Actor-based GPU vector index with Flat and IVF support
  - `VectorHandle` — Generation-based opaque handles for stable vector references
  - `IndexConfiguration` — Flexible index configuration with presets
  - `GPUIndexStats` — Comprehensive index introspection and statistics
  - `SearchResult` — Native GPU search results with L2² distances

- **Flat Index Features**
  - Sub-millisecond search on 5K+ vectors (0.30ms for 128D, 0.73ms for 768D)
  - Filtered search with iterative fetch strategy
  - Batch insert (~21K vectors/sec for 128D)
  - Lazy deletion with compaction support

- **IVF Index Features**
  - GPU-accelerated IVF (Inverted File) index
  - K-means clustering for coarse quantization
  - Configurable nlist and nprobe parameters

- **Handle Lifecycle**
  - Generation-based stale handle detection
  - Automatic handle remapping after compaction
  - Metadata preservation across operations

- **Comprehensive Test Suite** (109 new tests)
  - `FlatIndexSearchTests` — Search edge cases, filtering, batch operations
  - `HandleLifecycleTests` — Handle validity, compaction, generation tracking
  - `IntrospectionTests` — Statistics, memory reporting, configuration
  - `IVFTests` — IVF creation, training, search, compaction
  - `ValidationEdgeCaseTests` — Input validation, concurrency, error handling
  - `PerformanceBenchmarks` — Performance regression tests

### Fixed

- **IVF Training Bounds Check** — Fixed off-by-one error in IVF training where DeletionMask iteration could access unallocated vector slots

### Performance

| Operation | Dimension | Throughput |
|-----------|-----------|------------|
| Flat Insert | 128D | 21,866 vec/s |
| Flat Insert | 768D | 3,670 vec/s |
| Flat Search | 128D | 0.30ms (5K vectors) |
| Flat Search | 768D | 0.73ms (5K vectors) |
| IVF Search | 128D | 0.21ms (500 vectors) |

---

## [0.2.0] - 2025-12-01

### Breaking Changes

- **Metal 4 Required**: This release requires Metal 4 and drops support for older operating systems
  - macOS 26.0+ (was macOS 15.0)
  - iOS 26.0+ (was iOS 18.0)
  - tvOS 26.0+ (was tvOS 18.0)
  - visionOS 3.0+ (was visionOS 2.0)
  - Swift 6.0+ (was Swift 5.9)
- **Kernel Naming**: Dropped `Metal4` prefix from all public kernel classes
  - `Metal4L2DistanceKernel` → `L2DistanceKernel`
  - `Metal4CosineSimilarityKernel` → `CosineSimilarityKernel`
  - `Metal4MatrixMultiplyKernel` → `MatrixMultiplyKernel`
  - (and 21 other kernels)
- **Initialization Pattern**: All kernels now require `Metal4Context` and async initialization
  ```swift
  // Before (0.1.x)
  let kernel = try L2DistanceKernel(device: device)

  // After (0.2.0)
  let context = try await Metal4Context()
  let kernel = try await L2DistanceKernel(context: context)
  ```
- **Removed Legacy Kernels**: 23 legacy kernel files removed in favor of Metal 4 implementations

### Added

- **Metal 4 Infrastructure**
  - `Metal4Context` — Unified context for device, queues, and resource management
  - `Metal4ComputeEngine` — Unified command encoding
  - `ResidencyManager` — Explicit GPU memory residency management
  - `PipelineCache` — Thread-safe pipeline state caching
  - `PipelineHarvester` — Background pipeline compilation
  - `ArgumentTablePool` — Efficient argument table allocation
  - `TensorManager` — Tensor buffer management
  - `Metal4Capabilities` — Device capability detection
  - `Metal4ShaderCompiler` — Runtime shader compilation

- **Experimental ML Features** (Phase 4)
  - `LearnedDistanceKernel` — Projection-based learned distance metrics
  - `LearnedDistanceService` — High-level service with automatic fallback
  - `AttentionSimilarityKernel` — Attention-weighted similarity computation
  - `NeuralQuantizationKernel` — Neural network-based quantization

- **New Shaders**
  - `AttentionSimilarity.metal`
  - `LearnedDistance.metal`
  - `NeuralQuantization.metal`
  - `Metal4Common.h` — Shared MSL 4.0 header

- **Matrix Utilities**
  - `Matrix.random(rows:columns:range:)` — Random matrix generation
  - `Matrix.zeros(rows:columns:)` — Zero matrix creation
  - `Matrix.identity(size:)` — Identity matrix creation

### Changed

- All Metal shaders updated to MSL 4.0
- Kernel initialization is now async (`try await`)
- `dimension:` parameter removed from compute methods (now inferred from input)
- Internal protocols kept `Metal4` prefix for clarity:
  - `Metal4Kernel`, `Metal4DistanceKernel`
  - `Metal4EncodingResult`, `Metal4ThreadConfiguration`

### Fixed

- Fixed shared memory out-of-bounds access in `tiledMatrixMultiply` shader
  - Threads with `tid.x >= TILE_K` or `tid.y >= TILE_K` were writing outside allocated threadgroup memory
  - Caused incorrect results for small matrices

### Removed

- **Legacy Kernel Files** (23 files)
  - `L2DistanceKernel.swift` (old)
  - `CosineSimilarityKernel.swift` (old)
  - `DotProductKernel.swift` (old)
  - `MatrixMultiplyKernel.swift` (old)
  - `TopKSelectionKernel.swift` (old)
  - And 18 others

- **Legacy Test Files** (5 files)
  - `L2DistanceKernelTests.swift`
  - `CosineSimilarityKernelTests.swift`
  - `DotProductKernelTests.swift`
  - `FusedL2TopKKernelTests.swift`
  - `WarpOptimizedSelectionKernelTests.swift`

### Migration Guide

For users upgrading from 0.1.x:

1. Update minimum deployment targets to macOS 26.0 / iOS 26.0
2. Replace device-based initialization with context-based:
   ```swift
   // Old
   let device = MTLCreateSystemDefaultDevice()!
   let kernel = try L2DistanceKernel(device: device)

   // New
   let context = try await Metal4Context()
   let kernel = try await L2DistanceKernel(context: context)
   ```
3. Remove `dimension:` parameters from compute calls
4. Add `await` to kernel initialization and compute methods

---

## [0.1.x] - Previous Releases

For changes prior to 0.2.0, see git history. Version 0.1.x supports macOS 15+ / iOS 18+ with Metal 3.
