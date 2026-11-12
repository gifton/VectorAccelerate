# Changelog

All notable changes to VectorAccelerate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The hardening audit of the Metal stack. `docs/audits/AUDIT-2.md` and `docs/audits/AUDIT-3-shaders.md` are the per-finding record, and `docs/stability/` holds the resulting contracts.

### Fixed
- **`MetalComputeProvider` was unusable in Release builds.** `SoADistance.metal` was missing from the runtime shader compile, the only load path in those builds, and several pipeline keys named kernels that don't exist. The GPU cosine fused and batch paths had never run in any build because their kernel didn't exist.
- **`batchDistance` could not reach the GPU under default configuration**, so its GPU legs ran on the CPU.
- **Numerics.** GPU cosine, rooted L2, Minkowski and correlation misbehaved on huge, tiny and mixed-scale finite inputs. See `docs/stability/DISTANCE-RANGE-CONTRACT.md` for what is rescued and what keeps FP32 limits.
- **NaN handling.** GPU Top-K now orders NaN last and breaks ties by smaller original index, matching VectorCore 0.3.3. LogSumExp and basic statistics propagate NaN.
- **Dispatch-geometry bugs.** Reduction trees, barriers and threadgroup initializers gave wrong results or undefined behaviour at non-power-of-two widths in Jaccard, Manhattan, Chebyshev, Minkowski, LogSumExp, statistics and the single-pair basic kernels.
- **Out-of-bounds access.** Bounds are now enforced in PQ codes and ADC tables, Borůvka candidates, fused IVF candidates, sparse c-TF-IDF tails, batch matrix bias and quantization statistics. Device row and address arithmetic no longer overflows 32 bits, and unaligned vector loads are gone.
- **UMAP negative sampling race.** Targets are now frozen for the pass, so sampling no longer reads the embedding while writing it.
- **Ignored and mishandled neural options.** `useActivation`, `normalizeProjected` and `normalizeLatent` are honoured, and per-vector reconstruction scales are preserved.
- **Buffer lifecycle.** Pool allocation bounds, return routing across pool lifetimes, argument-table ownership and cache-clear accounting.
- **Shader validation.** Fused Top-K and IVF list search now fit the instrumented threadgroup memory budget.
- **`WarmupManager` lost pauses and cancels.** The warm-up loop wrote `state` after suspending on a pipeline, so `reportUserActivity()`, `pause(reason:)` and thermal pauses were overwritten and warm-up kept running. A restart could also be overwritten with `.cancelled`, leaving `beginWarmup` waiting forever.
- **A pipeline compile error during `Metal4Context` warm-up aborted Debug processes.** Only a key naming a missing kernel asserts now. Compile errors are logged and the pipeline compiles on first use.
- **The test suite failed under Swift 6.4's default build system.** Swift Build packages the shaders as `default.metallib` with no sources, so the runtime compile returned a library with no kernels and 76 tests that asked for it failed. `default.metallib` is now the shipped library. `KernelContext.makeLibraryFromBundleSources` throws when the bundle has no sources or an incomplete set, and the hardening suites certify whatever libraries the bundle ships. See `docs/stability/SHADER-LIBRARY-PACKAGING-CONTRACT.md`.
- **`SoACandidateSet` with zero candidates threw `bufferAllocationFailed`.** Metal refuses a zero-length buffer. It now allocates one element and copies nothing, and both scoring methods return empty results.
- **`PerformanceMonitor` reads the task's resident size inside the `withUnsafeMutablePointer` borrow** instead of after it. This wasn't the miscompile below, but it was the shape closest to it in production code.
- **Chebyshev (L∞) distance propagates NaN on every VectorAccelerate path** (owner decision 2026-10-31, matching VectorCore 0.4.0's `ChebyshevDistance`, downstream upgrade row BH-15): a NaN coordinate, or the same infinity on both sides, now gives NaN where `max` used to drop it; an infinite difference stays +Inf and finite results are unchanged. Applies to `MetalComputeProvider` (`.chebyshev`, single and batch), `FallbackProvider`, `Metal4ComputeEngine.chebyshevDistance` on both its CPU branch and the `chebyshevDistance` kernel, `MinkowskiDistanceKernel` with `.chebyshev`, `BatchProcessor` and `MemoryMapManager`. The kernels detect NaN on the integer bit pattern so the policy survives fast math. Pinned against Core's witness by `ChebyshevNaNPolicyTests`.

### Changed
- **Dependency: VectorCore floor raised to 0.3.3** for the NaN-last Top-K contract and original-index ties.
- **Minkowski no longer silently returns Chebyshev for large `p`.** The approximation is the explicit opt-in `Metal4MinkowskiConfig.chebyshevApproximation`.
- **Unsupported sizes now throw instead of being silently capped**, including attention head dimension, learned projection dimension, neural latent dimension and IVF `nprobe`. `TopKParameters` no longer clamps `k`.
- **Basic statistics accept NaN input** and propagate it, where they previously threw.
- **`encodeNegativeSampling` now throws** and has an overload taking caller-owned scratch.
- **`Metal4NeuralEncodingResult` exposes `scales`, one per row.** `scale` is now a deprecated diagnostic average.
- **Raw kernel bindings changed** for callers encoding shaders directly: `minkowski_distance_batch` buffer(7), `neural_encode_pass1` buffer(8), the UMAP negative-sampling output at buffer(3), and capacity words in the Borůvka and fused IVF parameter structs.
- **`PipelineCache.warmUp(keys:)` returns `[PipelineWarmUpFailure]`** instead of tuples with an error string.
- **Wall-clock test assertions run only under `VECTORACCELERATE_STRICT_PERF=1`**, enforced by `PerfGateCoverageTests`.
- **GPU availability guards use a runtime Metal check.** `MetalComputeProvider`, `AcceleratedVectorOperations`, `AcceleratedVectorFactory.isAccelerationAvailable`, `VectorCoreIntegration` and `MetalDevice.isGPUAvailable()` now share `MetalDevice.isAvailable` instead of VectorCore's `ComputeDevice.gpu().isAvailable`, which is a compile-time constant in 0.3.3. A host with no Metal device now fails at the guard with `metalNotAvailable`.
- **Every consumer's build host needs the Metal Toolchain component** under Swift Build.
- **Normalization parity with VectorCore 0.4.0 is pinned with one known divergence.** VectorCore 0.4.0 normalizes a vector whose coordinates are all subnormal (its widened plan returns the exact unit vector); VectorAccelerate keeps the unified pass-through for that class because Metal's fast-math mode flushes the subnormal inputs to zero before any arithmetic and the CPU fallbacks mirror the GPU. `NormalizationParityTests` no longer uses that fixture as a cross-check and pins the divergence explicitly (`testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere`). Every other fixture still matches VectorCore's `NormalizeKernels.normalizeUnchecked` elementwise.

### Added
- **Verified against VectorCore 0.4.0-alpha.1** (pinned `exact:`): see `docs/migrations/VECTORCORE-0.4.0-ALPHA1.md` for the pins, commands, results, outcome differences and what was not checked.
- `MetalComputeProvider.RoutingTelemetry`, which reports whether the GPU actually served a call.
- `BatchBiasLayout` and `PipelineWarmUpFailure`.
- The permanent guard suites in `Tests/VectorAccelerateTests/Hardening/`.
- A CI job that runs the test suite in Release, the configuration where the missing shader went unnoticed.
- `CompilerLostStoreCanaryTests` and `LostStoreShapeGuardTests`, for a Swift 6.2 and 6.4.2 optimizer miscompile found by VectorCore. See `docs/stability/COMPILER-LOST-STORE-CONTRACT.md`.
- **`VectorCore040CompatTests`** (18 tests): GPU-execution evidence from `MetalComputeProvider.routingTelemetry()` paired with comparisons against VectorCore's own witnesses, each with a `preferGPU: false` CPU twin, for provider dispatch, the dot-product sign convention, unsupported-metric deferral, batch-vs-single equality, the frozen SoA layout golden case and borrow-mode lifetime, empty/single candidate sets, copy-on-write isolation of `DynamicVector`, Euclidean and cosine agreement including NaN/infinity/zero policy, exact-tie and NaN Top-K ordering through the fused GPU path, scale-gated routing, ragged-input rejection, odd dimensions, and the `Vector<D>.isNormalized` dot-path switch.

### Removed
- `encodeTiledEncode`, `encodeTiledEncodeQuantize`, `encodeTiledEncodeQuantizeV2`, their `isTiled…Available` properties and `TiledEncodeParameters`. Their kernels never existed, so they could only throw. `encodeTiledV3` is the shipping tiled path.
- `PipelineCacheKey.cosineSimilarity` and about 2400 lines of dead or unsound shaders.

## [0.6.0] - 2026-08-15

The zero-copy SoA scoring release, plus the removal promised in 0.5.0. The minor bump reflects the removed deprecated GPU surface, the `L2NormalizationParameters.epsilon` default change, and the new SoA scoring API. The normalization parity work was validated with measured GPU evidence (fast-math reassociation and FTZ boundary behavior probed on hardware) and is gated by a release-configuration test that also guards the runtime shader-compile path.

### Added
- **Zero-copy SoA scoring** — the release headline. `SoACandidateSet` builds VectorCore's page-aligned lane-major SoA once and bridges it into a zero-copy `MTLBuffer` (borrow mode; staged-copy fallback when zero-copy is unavailable), and `MetalComputeProvider` gains `batchDistance(query:against:metric:)` / `findNearest(query:in:k:metric:)` against a prebuilt set — build once, score many. Distances run on lane-major L2/cosine Metal kernels validated against VectorCore's golden layout fixture; `findNearest` selects top-K with VectorCore's zero-copy pointer `TopKSelection.select` directly on the GPU output buffer (O(N log k), no full distance-array materialization — a fused single-thread GPU selection variant was benchmarked at 1.8–6.7× slower and rejected). The borrow-mode lifetime contract is enforced structurally: the SoA allocation and `MTLBuffer` are internal, so the provider methods — which pin the set until GPU completion — are the only possible submitters.
- **End-to-end dispatch tests** (`OperationsDispatchTests`): proves VectorCore's `Operations.findNearest` / `findNearestBatch` route through `Operations.$computeProvider` to the installed `BatchKernelProvider`.

### Fixed
- **Normalization diverged from VectorCore's CPU three ways (BE3 §4.4).** Neither GPU shader family pre-scaled before accumulating `Σ v²`, so subnormal-magnitude vectors underflowed the sum to `0` and huge-magnitude vectors (`|v| ≳ 1e19`) overflowed it to `+Inf`; on top of that the two families disagreed with each other and with the CPU about what to do next. `vectorNormalize` / `normalizeVectors` (behind `Metal4ComputeEngine.normalize` → `AcceleratedVectorOperations.normalize`) returned the **input unchanged** below a hard-coded `EPSILON = 1e-7`, silently leaving legitimately small vectors (e.g. `‖v‖ = 2.3e-19`) unnormalized; the `l2_normalize_*` family (behind `L2NormalizationKernel`) returned a **zero vector** below its `epsilon`, and reported `+Inf` in the `norms` output for huge inputs; VectorCore's post-BE3 CPU (`NormalizeKernels`, Kahan pre-scale with a `max(maxAbs, .leastNormalMagnitude)` clamp) normalized all of those correctly, and returned the input untouched only for vectors whose reciprocal norm is genuinely unrepresentable in FP32.

  Both families — and VectorAccelerate's own CPU fallbacks (`AccelerateFallback.normalize`, `SIMDFallback.normalize`, `FallbackProvider.normalize`, now sharing one `StableNormalization` implementation) — run VectorCore's algorithm: `den = clamp(maxAbs, 0x1p-126, 0x1p126)`, `scale = 1/den`, `sNorm = √Σ (v·scale)² = ‖v‖·scale`, `out = (v·scale)/sNorm` ≡ `v/‖v‖`. Consequences: huge and micro-magnitude vectors now normalize correctly instead of collapsing to zeros or passing through — including vectors whose `‖v‖₂` is itself unrepresentable (elements ≥ 1e38), since `1/‖v‖₂` is never formed; and **no finite input can produce `Inf`/`NaN`**, for any `epsilon` including `0`.

  Two GPU-specific hazards are handled explicitly, because Metal's default math mode flushes subnormals to zero. The pre-scale denominator is clamped on **both** sides (`[2^-126, 2^126]`): unclamped, a vector with components ≳ 8.5e37 produced a *subnormal* `1/maxAbs` that flushed to `0`, collapsing the sum of squares and misreporting the norm as `0.0`. And the output is a `precise::divide(v·scale, ‖v‖·scale)`, not a multiply by `1/(‖v‖·scale)`: fast math is free to reassociate the two scalings into the single factor `1/‖v‖`, which is subnormal for every `‖v‖ > 2^126` and flushed a 4e36 vector to all zeros in testing. Stored `norms` are the true `‖v‖₂` over the whole normal range; they saturate to `+Inf` where `‖v‖₂` genuinely exceeds `.greatestFiniteMagnitude` (the honest answer — the pre-fix kernel reported a finite-looking `0.0` there) and, at the other end, may read `0` for a subnormal `‖v‖₂` on a GPU that flushes denormals (see the `VA_NORM_MIN_SCALED` note in `Metal4Common.h`; those vectors are the pass-through class anyway). The normalized output is exact in both cases. The CPU fallbacks use the identical clamp and division so all five implementations agree.

  The normalization constants live in `Metal4Common.h` **and** in `KernelContext`'s runtime-compilation preamble, which strips the header include and is the only Metal library load path in release builds; a new test compiles that path and asserts the normalize kernels are present, so a constant added to one place but not the other can no longer disable the library. The shared threadgroup reductions were additionally made correct for non-power-of-two threadgroup sizes (the old `stride = tgSize/2` tree dropped lanes for dimensions such as 3, 5 or 100).

  *Unified degenerate policy:* every path returns the **input unchanged** for the vectors it cannot normalize — the true zero vector and vectors too small for `1/‖v‖₂` to be representable (`‖v‖₂ ≤ 2^-127`), plus any vector containing a non-finite component (`±Inf`/`NaN`). Note this is a small-side criterion: `‖v‖₂` *above* `.greatestFiniteMagnitude` has no representable reciprocal either, yet those vectors normalize correctly because the reciprocal is never formed. This matches `VectorCore.NormalizeKernels.normalizeUnchecked` (for the zero vector "unchanged" and "zeroed" coincide, so zero-vector behavior is bit-identical to before). The GPU pass-through copies raw bit patterns, so subnormals survive Metal's denormals-are-zero math mode.

  *Behavioral change:* `L2NormalizationParameters.epsilon` (and the `epsilon:` parameter on `L2NormalizationKernel.normalize` / `normalizeSingle`) **now defaults to `0` instead of `1e-8`**. It is no longer load-bearing for numerical stability — it is purely the caller's explicit degenerate threshold (vectors with `0 < ‖v‖₂ ≤ epsilon` are emitted as zeros), and `0` is the value that matches the CPU. Callers who relied on small-norm vectors being zeroed should pass `epsilon: 1e-8` explicitly. Two unreferenced legacy kernels stay off the unified path and are now documented as such in the shader source: `batchNormalize2D` (its cross-threadgroup atomic accumulation is unsound) and `fastNormalize` in `OptimizedMatrixOps.metal` (naive `Σ v²` plus a `+1e-8` reciprocal bias).

### Removed
- **The scattered GPU surface deprecated in 0.5.0, as scheduled** (12 declarations, all in `VectorCoreIntegration.swift`): the `AcceleratedDistanceProvider` actor; `AcceleratedVectorFactory.createDefaultProviders` / `createProviders`; the `acceleratedDistance(to:metric:)` / `acceleratedDistanceOptimized(...)` convenience extensions; `VectorCoreIntegration.createDistanceProvider`; and `BatchOperations.findNearestGPU` / `batchDistancesGPU` / `pairwiseDistancesGPU` / `normalizeGPU` / `normalizeGPUUnchecked` / `scaleGPU`. The last three (`normalizeGPU` / `normalizeGPUUnchecked` / `scaleGPU`) were carved out of the 0.5.0 deprecation wave and go **directly from live to removed in this release**: they were unused, paid a fresh `Metal4Context` per call with a sequential per-vector loop, and `normalizeGPU` silently dropped vectors whose reconstruction failed (`try?`), so the output could be shorter than the input. Migrate distance and search calls to **`MetalComputeProvider`** (`distance(_:_:metric:)`, `batchDistance(query:candidates:metric:)`, `findNearest(query:in:k:metric:)`, `distanceMatrix(queries:candidates:metric:)`); migrate vector-op calls to **`AcceleratedVectorOperations`** (`normalize(_:)`, `scale(_:by:)`), which is unaffected and remains the supported vector-ops actor. The now-pointless `MetalComputeProviderTests.testDeprecatedBatchOperationsStillDelegateCorrectly` delegation-parity test was removed alongside its targets.

### Changed
- **CPU top-K selection is now VectorCore's adaptive `TopKSelection`** (heap select, O(N log k) for small k) instead of a full O(N log N) sort, across every `MetalComputeProvider.findNearest` CPU path. Ties now resolve deterministically to the smaller index (previously unstable-sort order); for `dotProduct` the larger-is-closer ordering is preserved via exact IEEE-754 negation.
- **Dependency: VectorCore floor raised 0.3.0 → 0.3.2.** `Package.resolved` is now tracked in git for reproducible resolution across local/CI/clones.
- **Transitive public surface growth.** Via the package's `@_exported import VectorCore`, consumers now transitively receive VectorCore 0.3.2's enlarged public surface (`LinearAlgebra`, `ManifoldLearning`, `AccelerateArraySIMDProvider`) and its new `VectorCoreC` C target builds as a transitive dependency.

## [0.5.0] - 2026-06-06

The GPU compute façade + VectorCore 0.3.0 integration, layered on the earlier remediation of an external architectural & numerical audit (17 findings: 13 fixed, 2 refuted as non-issues, 1 deprecated/broken kernel removed). The minor bump reflects the new `MetalComputeProvider` API, the deprecation of the scattered GPU surface, the removed deprecated kernel, and the behavioral change to fused activation.

> Release notes for 0.4.3–0.4.4 are recorded in the `Package.swift` header.

### Added
- **`MetalComputeProvider`** — a single GPU compute façade (`batchDistance`, `findNearest`/top-K, `distanceMatrix`, single `distance`) that routes GPU-vs-CPU through `GPUDecisionEngine`, falls back to Accelerate, and reuses the no-copy kernel staging. Conforms to VectorCore's **`BatchKernelProvider`** — the R4 dispatch hook shipped in VectorCore 0.3.0. Installed via `Operations.$computeProvider.withValue(provider) { … }`, it makes VectorCore's `Operations.findNearest` / `findNearestBatch` dispatch transparently to the GPU inside the scope: euclidean/cosine run on the fused distance+top-K kernel, and every other metric falls back to that metric's own `batchDistance` so results never diverge from the CPU path.

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
