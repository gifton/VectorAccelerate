# Changelog

All notable changes to VectorAccelerate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-06-06

The GPU compute façade + VectorCore 0.3.0 integration, layered on the earlier remediation of an external architectural & numerical audit (17 findings: 13 fixed, 2 refuted as non-issues, 1 deprecated/broken kernel removed). The minor bump reflects the new `MetalComputeProvider` API, the deprecation of the scattered GPU surface, the removed deprecated kernel, and the behavioral change to fused activation.

> Release notes for 0.4.3–0.4.4 are recorded in the `Package.swift` header.

### Added
- **`MetalComputeProvider`** — a single GPU compute façade (`batchDistance`, `findNearest`/top-K, `distanceMatrix`, single `distance`) that routes GPU-vs-CPU through `GPUDecisionEngine`, falls back to Accelerate, and reuses the no-copy kernel staging. Conforms to VectorCore's **`BatchKernelProvider`** — the R4 dispatch hook shipped in VectorCore 0.3.0. Installed as `Operations.computeProvider`, it makes VectorCore's `Operations.findNearest` / `findNearestBatch` dispatch transparently to the GPU: euclidean/cosine run on the fused distance+top-K kernel, and every other metric falls back to that metric's own `batchDistance` so results never diverge from the CPU path.

### Deprecated
- **The scattered GPU distance/search surface**, all superseded by `MetalComputeProvider` and **scheduled for removal in 0.6.0**: `BatchOperations.findNearestGPU` / `batchDistancesGPU` / `pairwiseDistancesGPU` (now thin delegates to the provider — also fixing `pairwiseDistancesGPU`'s Chebyshev-as-Euclidean bug), `AcceleratedDistanceProvider`, the `acceleratedDistance(to:metric:)` / `acceleratedDistanceOptimized(...)` convenience extensions, and the `AcceleratedVectorFactory.createDefaultProviders` / `createProviders` / `VectorCoreIntegration.createDistanceProvider` vendors. (`AcceleratedVectorOperations` vector ops are unaffected.)

### Removed
- **`streaming_l2_topk_update` Metal shader and its Swift wiring** (`Metal4StreamingL2Params`, `FusedL2TopKKernel.encodeStreamingUpdate`, the streaming pipeline). The kernel was experimental and already deprecated: each thread kept a private heap but only thread 0 wrote back, discarding every other thread's results. Use `FusedL2TopKKernel.execute()`, which uses the correct chunked two-pass fallback.

### Fixed
- **Fused activation was a silent no-op.** `BatchMatrixKernel.multiplyFused(config:)` packed `activation` into the parameters but never sent it to the GPU, so it returned a plain linear GEMM. The activation code is now passed and `batchMatrixMultiplyFused` applies ReLU/tanh/sigmoid/GELU. *(Behavioral change for callers using `activation:`.)*
- **Scalar quantization collapse.** A single `+Inf` was remapped to ~`FLT_MAX`, which dominated the global min/max so every normal value quantized to 0. The scale is now derived from the vector's finite range (with `±Inf` pulled into it); a `/256` magnitude clamp keeps the dequantization round trip overflow-safe.
- **K-Means++ centroid selection.** Replaced `Array.contains` membership (O(N·K²)) with a boolean mask (O(N·K)); the probability sum now accumulates in `Double`, so Float32 mantissa absorption no longer starves tail points of selection probability.
- **Cosine distance could go negative.** Clamp similarity to `[-1, 1]` (NaN-preserving) across `cosine_similarity` and `BasicOperations` so floating-point drift can't yield a negative distance and break top-k min-heap invariants.
- **Softmax over infinities.** A row with *k* `+Inf` entries summed to *k*; mass is now distributed uniformly as `1/k` (both the per-element and per-row kernels) so rows sum to 1.
- **Batch distance providers** now reject dimension-mismatched candidates instead of staging past a buffer row; **`manhattanDistance`** guards empty input.

### Changed
- **Dependency: require VectorCore 0.3.0** (was 0.2.1) — for the frozen SoA layout contract and the `BatchKernelProvider` / R4 dispatch hook behind the GPU façade and the zero-copy bridge, and inheriting (via 0.2.2) the BE3 audit fixes: a `SwiftFloatSIMDProvider` SIMD8 heap-overflow (the root cause of intermittent softmax NaNs), a cosine-denominator infinity overflow, and an async `MemoryPool` leak. No source changes were required for the bump: VectorAccelerate uses neither the source-breaking `LinearQuantizationParams.zeroPoint` (`Int8`→`Int32`) nor the renamed SoA euclidean kernel.
- **Zero-copy distance staging.** Query/candidate storage is copied straight into Metal buffers via `withUnsafeBufferPointer`, eliminating per-vector `toArray()` allocations and intermediate flat arrays on the L2/Cosine provider batch paths.
- **1-D threadgroup dispatch** for the L2 and cosine kernels. They index by a scalar `thread_position_in_threadgroup`, so the previous 2-D `(w×h)` group ran the reduction at `1/h` width with `h`-fold redundant work (results were already correct).
- **CPU fallbacks:** small-batch Euclidean routes through Accelerate (vDSP); `SIMDFallback` dot/Euclidean/normalize use `loadUnaligned` instead of scalar element fills; `manhattanDistance` uses a stack scratch buffer instead of a per-call heap array.
- **`AttentionSimilarityParameters`** uses explicit `UInt8` padding fields (not a tuple) for a deterministic C-compatible layout, pinned by a `MemoryLayout` test.

### Added
- Regression tests: fused ReLU is actually applied; quantization survives a `+Inf` outlier; `AttentionSimilarityParameters` layout is pinned at 40 bytes; batch top-K K-scaling measurement.

### Notes
- The audit's two "Critical" infrastructure claims were **refuted**: the L2/cosine kernels were numerically correct (the dispatch item above was a performance, not a correctness, bug), and the buffer pool's pending-return queue drains from multiple call sites (no leak).
- Batch top-K (`topk_select_batch_kernel`) keeps its per-thread `heap[128]`; profiling shows large-K cost scales (K=128 ≈ 2.5× K=32), so a threadgroup-cooperative rewrite is a tracked follow-up if large-K lands on a hot path.

---

## [0.4.2] - 2026-04-05

### Added
- **Technical Audit Report:** Comprehensive `AUDIT.md` covering API design, performance architecture, correctness, testing gaps, benchmarking strategy, and prioritized improvement roadmap.
- **P0 Test Suite (81 tests):** New test files for `Metal4ComputeEngine`, `BatchDistanceEngine`, `AccelerateFallback`, `ArgumentTablePool`, and `GPUDecisionEngine` -- covering distance operations, batch routing, CPU fallbacks, pool lifecycle, and adaptive thresholds.
- **Crossover Benchmark Runner:** New `--crossover` CLI mode that sweeps batch sizes across dimensions and metrics to identify the CPU/GPU performance breakeven point. Outputs JSON for threshold calibration.
- **Pre-Allocated Buffer API:** `allocateVectorBuffer`, `allocateResultBuffer`, and `*WithBuffers` variants of batch Euclidean, batch cosine, single Euclidean, and single dot product -- enabling zero per-call allocation for hot-path reuse.
- **GPU Timestamp Profiling:** `GPUTimingInfo` struct capturing `MTLCommandBuffer.gpuStartTime`/`gpuEndTime` after every `executeAndWait`/`executeBlitAndWait`. Benchmark reports now show wall-clock p50/p95/p99, GPU compute p50/p95, and submission overhead.
- **`GPUOperation.chebyshevDistance`:** New enum case for proper adaptive routing of Chebyshev distance operations.

### Changed
- **Benchmark Timing:** Replaced `CFAbsoluteTimeGetCurrent()` with `ContinuousClock` across all benchmark and profiling paths (BenchmarkFramework, Metal4Context, Metal4ComputeEngine). Added p95/p99 percentiles via `LatencyStats` reuse. Added `blackHole()` to prevent dead-code elimination of benchmark results.
- **Unified Error Handling:** `AccelerateFallback` distance methods now throw `VectorError.dimensionMismatch` instead of returning `Float.nan`. `FallbackProvider` distance methods throw instead of returning `.infinity`. Batch wrappers use `try?` with appropriate fallback values.
- **Adaptive GPU/CPU Routing:** `Metal4ComputeEngine` now accepts an optional `GPUDecisionEngine` for data-driven routing. All 7 single/batch operations route through `shouldUseGPU`/`shouldUseBatchGPU` helpers. CPU fallback added to `euclideanDistance` and `cosineDistance` single-vector operations.

### Fixed
- **SIMD Path `withTaskGroup` Anti-Pattern:** Replaced per-candidate `Task` spawning in `batchEuclideanDistanceSIMD`, `batchCosineSimilaritySIMD`, and `batchManhattanDistanceSIMD` with direct `AccelerateFallback` calls. Eliminates ~500 task spawns per batch, estimated 10-100x throughput improvement on the SIMD path.
- **`BufferToken.deinit` / `ArgumentTableToken.deinit` `Task.detached`:** Replaced with synchronous `PendingBufferReturns`/`PendingTableReturns` queues drained on next pool access. Fixes non-deterministic buffer return, potential leaks on program exit, and degraded pool hit rate.
- **TOCTOU Double-Return Bug:** `BufferToken.deinit` and `ArgumentTableToken.deinit` now atomically set `isReturned`/`isReleased` under lock before enqueue, preventing double-return when `returnToPool()`/`release()` races with deinit.
- **Retain Cycle in Pending Return Queues:** `PendingBufferReturns` and `PendingTableReturns` now store `ObjectIdentifier` instead of strong pool references, breaking the singleton-to-pool retain cycle.
- **12 `defer { Task { release } }` in Metal4ComputeEngine:** All argument table release paths replaced with synchronous `PendingTableReturns.shared.enqueue`, consistent with the deinit fix.
- **Stale GPU Timing:** `lastGPUTiming` now set to `nil` when timestamps are invalid (`gpuStart == 0` or `gpuEnd < gpuStart`), preventing stale timing from prior iterations leaking into benchmark results.
- **Zero-Copy Scalar Reads:** Added `BufferToken.readScalar(as:)` and replaced all 11 single-value `copyData(as:)[0]` sites (5 in Metal4ComputeEngine, 4 in KernelDistanceProviders, 2 in pre-allocated API) with direct unified-memory reads. Eliminates per-call `[Float]` array allocation for scalar results.

---

## [0.4.1] - 2026-04-04

### Added
- **VectorCore 0.2.0 Integration:** Updated dependency and aligned with new standardized search result types.
- **Enhanced Search API:** `AcceleratedVectorIndex.search` now returns `SearchResults<VectorHandle>`, providing rich metadata including search timing and candidate counts.
- **Normalized Cosine Optimization:** Leveraged the new `IndexableVector.isNormalized` hint from VectorCore to bypass redundant GPU norm calculations in `CosineKernelDistanceProvider`.
- **Vector384Optimized Support:** Full integration with VectorCore's new optimized 384D type for high-performance MiniLM/BERT workflows.

---

## [0.4.0] - 2026-04-03

### Added
- **Hierarchical SIMD Reductions:** Overhauled L2 and Cosine kernels using a 4-phase reduction model (`Local -> Warp -> Threadgroup -> Global`), maximizing 128-bit memory bus saturation.
- **Tiled Shared Memory KMeans:** New hardware-adaptive assignment kernel that dynamically scales tiles to fit within Apple Silicon's 32KB shared memory limit.
- **Cooperative Gather Topology:** High-performance 2-pass K-Means update orchestration that eliminates global atomic contention during centroid re-calculation.
- **Tiled GEMM Neural Encoder:** New state-of-the-art encoder (Phase 5) using a Full-D register loop and shared memory padding to achieve 10-50x speedups by eliminating bank conflicts.
- **Vectorized Transposed Decoder:** Optimized dequantization path (Phase 4) using dual-accumulator latency hiding and dimension-specialized variants for a 2x throughput gain.
- **Eager Pipeline Pre-compilation:** Background pre-compilation of critical path kernels during `Metal4Context` initialization to eliminate first-use latency.
- **Advanced Distance Metrics:** GPU-accelerated kernels for Manhattan, Chebyshev, Minkowski, and Jaccard distances.

### Changed
- **Enforced Asynchronous Execution:** All GPU operations now utilize `await commitAndWait()` to suspend Swift tasks without blocking OS threads.
- **Buffer Pool Integration:** Transitioned all hot-path kernels to use the new `BufferPool` ring-buffer strategy, removing allocation overhead from compute loops.
- **Standardized Kernel Signatures:** Refactored all Metal 4 kernels to use consistent attribute dimensionality, ensuring compatibility with the latest Apple Silicon compilers.
- **Enhanced Buffer Safety:** Implemented `BufferToken` lifecycle anchoring via `.keepAlive(until:)` to synchronize memory reclamation with physical GPU completion.

### Fixed
- **Empty Cluster Stability:** Improved K-Means update logic to preserve existing centroids for clusters with zero assignments, preventing NaN corruption.
- **Minkowski Overflow:** Fixed numerical instability in high-power Minkowski distances by implementing log-space exponent clamping.

---

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
