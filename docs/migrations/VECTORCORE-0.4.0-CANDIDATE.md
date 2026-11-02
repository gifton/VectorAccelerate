# VectorAccelerate against the VectorCore 0.4.0 candidate

**Written:** 2026-10-30 · **Status:** candidate verification, not a release. The candidate is unmerged, untagged
and unreleased; nothing here moves the manifest floor.

## 1. What was pinned

| Repository | Revision | Notes |
|---|---|---|
| VectorAccelerate (this worktree) | base `853052d28f215b3ab02972a3fa7a57d5d2a1f873` on `gifton/metal-hardening-checkpoint`; work on branch `gifton/vectorcore-0.4.0-prep` in `.worktrees/vectorcore-0.4.0-prep` | the hardening checkpoint plus the 0.4.0 preparation below |
| VectorCore candidate | `379e610f4365e820454c8c5328c4f87dfaa2031b` (`codex/audit-top-six-2026-10-27`, `v0.3.3-29-g379e610`) | tested implementation `73d6a5bcd4fa1063e39d467284638b2812a84e7d` plus documentation; `VectorCoreVersion` still reports `0.3.3` (upgrade row LM-02) |
| VectorCore baseline | tag `v0.3.3`, `fca4b602383589c46b627d8a0de2b6b2a68cd07d` | what `Package.resolved` pins and what the baseline runs used |
| VectorCore checkout read | `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3` | HEAD is the candidate; `git status` clean before and after this work; not modified |

### Dependency override (isolated, local)

```
swift package edit VectorCore --path /Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3
```

This symlinks `Packages/VectorCore` (gitignored) to the checkout and marks the dependency `edited` in
`.build/workspace-state.json`. `Package.swift` is unchanged apart from a comment recording the SHAs; its floor
stays `from: "0.3.3"` because the candidate has no tag. SwiftPM drops an edited package from `Package.resolved`,
so that file is restored from git before any commit and must be re-restored after further builds under the
override. Undo with `swift package unedit VectorCore`. Builds read the Core checkout in place and write only to
this worktree's `.build`.

Pinning by revision (`.package(url:, revision:)`) is not possible until the commit is pushed.

### Environment

| Item | Value |
|---|---|
| Host | Apple M3 Max, 48 GiB (Mac15,9 class), macOS 27.0.1 (26A434) |
| Xcode / Swift | Xcode 27.2 (27B52), Swift 6.4.2 (swiftlang-6.4.2.1.9) |
| Metal Toolchain | 27B47, downloaded during this work with `xcodebuild -downloadComponent MetalToolchain` (it was absent after the Xcode 27.2 upgrade; every `swift build` failed at `CompileMetalFile` until then) |
| SwiftPM build system | default `swiftbuild` (Swift Build); `--build-system native` is available but deprecated. The two differ in how `.metal` resources are packaged, see section 6 |
| Perf gates | `VECTORACCELERATE_STRICT_PERF` unset (perf assertions skipped, as designed) |

## 2. Compilation and migrations

| Command (from the worktree, override active) | Result |
|---|---|
| `swift build --build-tests` (Debug, swiftbuild) | first attempt: one error, `MetalDevice.swift:462: value of type 'ComputeDevice' has no member 'isAvailable'` (the only compile break, row CB-02). After the migration below: `Build complete!`, 0 errors; the only warnings are VectorAccelerate's own pre-existing deprecations (`StreamingTopKKernel`, `Metal4StreamingConfig`, `resetForTesting()`) |
| `swift build -c release --build-tests` (swiftbuild) | library, benchmarks executable and VectorCore build; the `@testable` test target fails to link against the release module ("unable to resolve Swift module dependency to a compatible module: 'VectorAccelerate'"). This is a build-system limitation independent of the Core change; `swift test -c release` is the working form, see section 5 |
| `swift test --build-system native` (Debug, builds then runs) | builds with 0 errors; VectorCore compiled from the override in `.build/arm64-apple-macosx/debug` |
| Baseline control, pristine tree at 0.3.3: `swift build --build-tests --scratch-path <scratch>` | `Build complete!` (VectorCore resolved to `fca4b602`, 0.3.3) |

### Source changes

One compile break, upgrade row **CB-02**: `ComputeDevice.gpu().isAvailable` is removed. In 0.3.3 it was a
compile-time constant (`true` wherever Metal can be imported outside the simulator), so VectorAccelerate's seven
guards were constant `true` on every supported platform. They now share VectorAccelerate's own runtime check.

| File | Change |
|---|---|
| `Sources/VectorAccelerate/Core/MetalDevice.swift` | `isGPUAvailable()` returns `MetalDevice.isAvailable` (`MTLCreateSystemDefaultDevice() != nil`); doc comments no longer name the removed `DeviceCapabilities` |
| `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift` | `init(configuration:)` guard |
| `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift` | `AcceleratedVectorOperations.init()`, `AcceleratedVectorFactory.isAccelerationAvailable`, `VectorCoreIntegration.createVectorOperations()` |
| `Sources/VectorAccelerateBenchmarks/main.swift`, `CrossoverBenchmarkRunner.swift` | the executable's availability guards |
| `Package.swift` | comment recording the candidate SHAs and the override |
| `Tests/VectorAccelerateTests/VectorCore040CompatTests.swift` | new, section 4 |

Behavioural effect: on a Metal host nothing changes. On a host without a Metal device the guards now fail with
`metalNotAvailable` instead of failing later inside `Metal4Context()`. The migrated code compiles against 0.3.3
as well (verified by the baseline build), so the branch does not depend on the override to build.

## 3. Compatibility checklist (every applicable row of the downstream upgrade list)

Legend. **Hit**: VectorAccelerate reaches the changed surface; migrated and/or tested. **Watch**: compiles
unchanged, behaviour relevant, evidence recorded. **N/A**: no reference in `Sources/` or `Tests/` (mechanical
grep over the symbol list of the row), listed so the row is accounted for. Test names refer to
`VectorCore040CompatTests` unless another suite is named.

### 3.1 Compile-breaking (section 1)

| Row | Status | Evidence |
|---|---|---|
| CB-01 `StaticDimension.Storage` | N/A | `D.Storage` / `DimN.Storage` never named; the four `<D: StaticDimension>` kernels compile unchanged |
| CB-02 `ComputeDevice` discovery | **Hit** | seven sites migrated (section 2); `testGPUAvailabilityGuardsUseRuntimeMetalCheck` |
| CB-03 `Operations.bufferProvider`, `VectorCore.Configuration` | N/A | no reference |
| CB-04 flag helpers | N/A | no reference |
| CB-05 `SoA.memoryEfficiency`, `estimatePerformanceGain` | N/A | no reference; `SoA.memoryFootprint`/`layoutDescriptor` still used |
| CB-06 removed concrete metric overloads | Watch | VectorAccelerate calls Core metrics only through `any DistanceMetric<Float>` witnesses (`MetalComputeProvider.batchDistance`); numeric effect under BH-12/13 |
| CB-07 `SearchResult.computedScore` | N/A | VectorAccelerate constructs `SearchResult(id:distance:)` and never reads `computedScore` |
| CB-08 `SoAFP16Cache512` | N/A | no reference |
| CB-09 `fp16Storage` | N/A | no reference |
| CB-10 `TypedBufferPool.init` | N/A | VectorAccelerate has its own buffer pool |
| CB-11, CB-12, CB-13 auto-tuners | N/A | no reference to `KernelAutoTuner`, `MixedPrecisionAutoTuner`, `QuantizedAutoTuner`, `WorkloadCharacteristics`, `PerformanceProfile` |
| CB-14 `detectOverflow` (Intel) | N/A | no reference; Metal-only package |
| CB-15 `MutableVectorCollection.update` | N/A | no `VectorCollection` conformer anywhere |
| CB-19 `BufferError.invalidRequest` | N/A | the only `BufferError` in the tree is VectorAccelerate's own `Metal4Error` helper |
| CB-40, CB-41, CB-42 benchmarking | N/A | `VectorCoreBenchmarking` is not a dependency |
| CB-43 dynamic storage structs | **Hit** (indirect) | no direct reference; reached through `DynamicVector`: `testDynamicVectorCopyOnWriteIsolationThroughProvider` |
| CB-44 watchOS | N/A | watchOS already excluded (no Metal) |

### 3.2 Signature changes (section 2)

| Row | Status | Evidence |
|---|---|---|
| SG-01 `findNearestBatch(cpuStrategy:)` | Watch | call sites omit the parameter and compile; no full-name reference; `testFindNearestBatchThroughProviderEqualsPerQuery` |
| SG-02 `entropyFast` | N/A | no reference |
| SG-03 `centroid`/`statistics` throw | N/A | no reference |
| SG-04 `BatchOperations.updateConfiguration` | N/A | no reference |
| SG-05 `any DistanceMetric<Float>` facades | Watch | `BatchKernelProvider` keeps `any DistanceMetric` parameters at the candidate (read from `Sources/VectorCore/Protocols/BatchKernelProvider.swift`; only its doc comment changed), so `MetalComputeProvider`'s three witnesses still bind: `testOperationsDispatchReachesProviderWitnessesAndExecutesGPU`, `OperationsDispatchTests`. No `VectorCollection.search`, `TopKSelection.nearest/batchNearest`, `OptimizedDistanceMetrics.create` use |
| SG-06 `PCAModel.mean` | N/A | no reference |
| SG-07 `TypedBufferPool.acquire` | N/A | no reference |
| SG-08 `KernelAutoTuner` signatures | N/A | no reference |
| SG-09 `AutoTuning.calibrateIfNeeded` | N/A | no reference |
| SG-40, SG-41, SG-42 benchmarking | N/A | not a dependency |

### 3.3 Behaviour changes (section 3)

| Row | Status | Evidence |
|---|---|---|
| BH-01 batch search default route | Watch | with the provider installed Core delegates before any CPU route; `testFindNearestBatchThroughProviderEqualsPerQuery` pins per-query equality. Without a provider VectorAccelerate makes no batch-search call |
| BH-02 generic small-k streaming | Watch | the provider is reached first for every metric; for an unsupported metric it defers to the metric's `batchDistance` plus `TopKSelection`: `testUnsupportedMetricThroughProviderIsBitIdenticalToCoreCPURoute` (bit-identical to Core's new route) |
| BH-03 `VectorMath.nearestNeighbors` | N/A | no reference |
| BH-04 facades keep the metric | N/A | no facade use |
| BH-05 relevance scores on `SearchResult` | Watch | VectorAccelerate produces `SearchResult<VectorHandle>` with `score == nil` and never consumes Core-produced results; consumers comparing the two by `==` will see the new `score` |
| BH-06 `KNearestHeap` | N/A | no reference |
| BH-07 `TopKResult` decoding | N/A | VectorAccelerate's `TopKResult` alias is its own `Metal4TopKResult`; Core's is never decoded |
| BH-08 type-identity routing | N/A | no retroactive `VectorProtocol` conformance; the only custom conformer is the test-local `UploadWindowVector` |
| BH-09, BH-53 `pairwiseDistances` | N/A | no reference |
| BH-10 matrix diagonals, `cosineSimilarityMatrix` | N/A | no reference |
| BH-11 optimized-vector hashing | N/A | no `Set`/`Dictionary` keyed by `Vector*Optimized` |
| BH-12 Euclidean/dot arithmetic | **Hit** | `testEuclideanMatchesCoreWitnessOnGPUAndCPU`, `testEuclideanNonFiniteCoordinatesMatchCoreOnGPUAndCPU`, `testDotProductThroughProviderKeepsCoreNegatedConvention`; `SoAScoringParityTests`, `MetalComputeProviderTests` |
| BH-13 cosine arithmetic | **Hit** | `testCosineEdgePolicyMatchesCoreOnGPUAndCPU`, `testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance`, `testPreferGPUFalseServesOperationsDispatchOnCPU`; `SoAScoringParityTests` |
| BH-14 cosine NaN/Inf/zero policy | **Hit** | `testCosineEdgePolicyMatchesCoreOnGPUAndCPU` (GPU and CPU twins against the candidate's witness) |
| BH-15 Chebyshev/Hamming NaN | **Hit** (decided 2026-10-31, section 10) | through the Core hook `ChebyshevDistance` defers to Core's witness. VectorAccelerate's own Chebyshev paths used `max`, which drops NaN; all six now propagate NaN exactly as Core does, pinned against Core's witness by `ChebyshevNaNPolicyTests` (CPU: `MetalComputeProvider`, `FallbackProvider`, `Metal4ComputeEngine`'s CPU branch, `BatchProcessor`, `MemoryMapManager`; GPU: the `chebyshevDistance` kernel and the `minkowski_distance_batch` L∞ branch) |
| BH-16 NaN-propagating reductions | N/A | no Core reduction, clip or `minElement`/`maxElement` use |
| BH-17, BH-18 FP16 NaN canonicalization | N/A | no `MixedPrecisionKernels` use |
| BH-19 `Vector<D>.isNormalized` witness | **Hit** (decided 2026-10-31: accept as tested, section 10) | `KernelDistanceProviders` switch cosine to the dot kernel when both sides report `isNormalized`; normalized `Vector<Dim512>` inputs now take that path: `testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance`. `AcceleratedVectorIndex.insert<V: IndexableVector>` records the hint (`vectorHints(for:)`) and does not route on it. `DynamicVector` and `Vector*Optimized` unchanged. The owner accepted the behaviour as tested: both paths agree with Core within the FP32 bound, the index keeps recording the hint, and the data-dependence of the check is Core request C3 |
| BH-20 width-0 extrema | N/A | no zero-width dimension |
| BH-21 storage shape enforcement | N/A | VectorAccelerate never assigns a Core vector's `storage` |
| BH-22 dimension guards, factory | Watch | `AcceleratedVectorOperations` builds results with `try T(array)`; same throwing contract as 0.3.3 |
| BH-23 execution-domain validation | Watch | `MetalComputeProvider.maxConcurrency` is `maxThreadsPerThreadgroup.width` (positive); every `parallel*` default is inherited: `MetalComputeProviderTests.testConformsToComputeProviderWithGPUIdentity`, dispatch tests |
| BH-24 scheduling and cancellation | Watch | inherited defaults; no custom reduction |
| BH-25, BH-26, BH-27, BH-28, BH-29, BH-30 | N/A | configuration, collections, AutoTuning, FP16 wrappers, linear algebra, UMAP (VectorAccelerate's UMAP kernel is its own) not referenced |
| BH-31 `Logger`, BH-51 `Logger` name | N/A | VectorAccelerate's module-level `Logger` actor shadows both imports; `Core/Logger.swift` (`import os.log`) and `Persistence/WriteAheadLog.swift` (`import os`) compile |
| BH-32 `createVector` docs | N/A | no reference |
| BH-33 normalization range policy | **Hit** | `NormalizationParityTests` cross-checks `NormalizeKernels.normalizeUnchecked` on six fixtures; see section 5 |
| BH-34 `NormalizationHint` invalidation | Watch | `IndexableVectorInsertTests` (hint capture through the typed insert); see section 5 |
| BH-35 approximate equality | N/A | no reference |
| BH-36, BH-37, BH-38 statistics, entropy | N/A | no reference |
| BH-39, BH-59 INT8 quantization | N/A | VectorAccelerate's `ScalarQuantizationKernel` is its own; no Core INT8 type |
| BH-40, BH-41, BH-42, BH-60 tuners, FP16 diagnostics | N/A | no reference |
| BH-43 binary serialization | N/A | the WAL is VectorAccelerate's own format |
| BH-44, BH-45, BH-56, BH-57 pools | N/A | own buffer pool |
| BH-46 `SoAFP16Cache512` | N/A | no reference |
| BH-47 aligned memory | Watch | reached only inside Core's `SoA` allocation: `testSoALayoutContractGoldenThroughZeroCopyGPUPath`, `testSoABorrowModeKeepsOwnershipAndReleasesAfterGPUCompletion`, `testSoACandidateSetBoundaryCounts` |
| BH-48 tiered storage | N/A | no reference |
| BH-49 Intel FP16 | N/A | arm64 Metal host |
| BH-50 `Mirror`/`dump` | N/A | no snapshot of Core types |
| BH-52 defaulted requirements binding client methods | N/A | no `relevanceScore(for:)` or `search(query:k:metric:)` declared anywhere |
| BH-54 Release traps on undersized outputs | N/A | no `MixedPrecisionKernels`/`MatrixDistance` use |
| BH-55 hinted `ComputeProvider` defaults | Watch | `MetalComputeProvider` implements only `execute`; every hint-less and hinted default is inherited, so it is reached by every route |
| BH-58 manifold error kinds | N/A | no reference |
| BH-61 `normalizedUncheckedWithHint` | Watch | not referenced, but `acceleratedNormalizeUnchecked()`'s CPU branch calls `normalizedUnchecked()`, which now stops the process on zero or non-finite input (0.3.3 asserted in Debug only); unchanged API, sharper failure |
| BH-62 .. BH-67 audit repairs | Watch / N/A | BH-62 covered with CB-43; BH-63..67 not referenced |

### 3.4 ABI and layout (section 4)

Every row (AB-01 .. AB-09) requires a rebuild from source; the override builds Core from source, so all are
satisfied by the builds in section 2. No code reads `MemoryLayout<Vector*Optimized>`, a `scalarCount` child
through `Mirror`, or a Core vector's `storage`.

### 3.5 Performance (section 5)

PF-01 (slower default batch search) and PF-02 (per-row pairwise) apply only without a provider or to
`pairwiseDistances`, which VectorAccelerate does not call. PF-03 (O(D) hashing) is unused. No timing was
measured here (perf gates off); nothing in this document is a performance claim.

### 3.6 New API worth adopting (section 6)

| Row | Disposition |
|---|---|
| NA-01 `cpuStrategy: .gram(...)` | not adopted: VectorAccelerate's CPU fallback is its own Accelerate path and the provider is reached before Core's CPU strategies |
| NA-02 `ComputeDevice.isSupportedByCore` | candidate for `MetalDevice.asComputeDevice()` documentation; `.gpu(index: 0).isSupportedByCore == false` correctly says Core supplies no GPU provider. Not referenced yet so the tree still compiles against 0.3.3 |
| NA-03 `relevanceScore(for:)` | optional: `AcceleratedVectorIndex` could populate `score`; not done |
| NA-10 `VectorCoreCompiledFeatures` | optional diagnostics; not done |
| others | not applicable |

### 3.7 Benchmarking and C surface (section 7)

Not applicable: `VectorCoreBenchmarking` is not a dependency and no `VectorCoreC` header is included directly
(the target builds transitively, as before).

### 3.8 Limitations that bound this work (Appendix E)

LM-01: this document is the first downstream execution of the candidate. LM-02: pinned by local override, not
by version. LM-04 and LM-05: cosine extremes and GEMM routes are not asserted bit-equal anywhere here; the
cross-package tolerances are stated in `VectorCore040CompatTests.crossPackageTolerance`. LM-07: the only custom
`VectorProtocol` conformer is test-local.

## 4. The new evidence suite: `VectorCore040CompatTests`

Eighteen tests. Every GPU assertion reads `MetalComputeProvider.routingTelemetry()` and requires
`gpuKernel >= 1` with zero silent fallbacks; every GPU test has a `preferGPU: false` twin that requires
`gpuKernel == 0`. Providers are built with `fallbackToCPU: false`, so a GPU error fails the test instead of
producing a CPU result. Tolerance: `dim * 2^-23` relative (twice that for cosine), the FP32 accumulation bound
over `dim` products with a different summation order on each side; ties and NaN ordering are asserted exactly.

| Priority | Tests |
|---|---|
| 1 API and dispatch | `testGPUAvailabilityGuardsUseRuntimeMetalCheck`, `testOperationsDispatchReachesProviderWitnessesAndExecutesGPU`, `testDotProductThroughProviderKeepsCoreNegatedConvention`, `testUnsupportedMetricThroughProviderIsBitIdenticalToCoreCPURoute`, `testFindNearestBatchThroughProviderEqualsPerQuery` |
| 2 Ownership and GPU lifetimes | `testSoALayoutContractGoldenThroughZeroCopyGPUPath`, `testSoABorrowModeKeepsOwnershipAndReleasesAfterGPUCompletion`, `testSoACandidateSetBoundaryCounts`, `testDynamicVectorCopyOnWriteIsolationThroughProvider` |
| 3 Numerical and search contracts | `testEuclideanMatchesCoreWitnessOnGPUAndCPU`, `testCosineEdgePolicyMatchesCoreOnGPUAndCPU`, `testEuclideanNonFiniteCoordinatesMatchCoreOnGPUAndCPU`, `testTopKTiesAndNaNOrderingThroughGPUFusedAndCPUPaths`, `testSoAFindNearestTiesResolveToSmallerIndex` |
| 4 Routing and fallback | `testInstalledProviderRoutesByScaleAndOnlyInsideScope`, `testRaggedInputRejectedAndOddDimensionServedOnGPU`, `testPreferGPUFalseServesOperationsDispatchOnCPU` |
| BH-19 decision item | `testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance` |

Static audit for priority 2 (pointer lifetimes): every `withUnsafeBufferPointer` / `withUnsafeMutableBufferPointer`
over a Core vector in `Sources/` copies synchronously inside the closure (`Array($0)`, `update(from:)`,
`memcpy`); no Core storage pointer is retained across an `await`. The SoA bridge holds the `SoA` object (borrow
mode) and pins it in the command buffer's completion handler; `consumeAllocation()` is never called.

## 5. Results

All runs from the worktree on the host in section 1, `VECTORACCELERATE_STRICT_PERF` unset. "Baseline" means the
pristine tree at `853052d` against VectorCore 0.3.3 in a separate scratch path (`--scratch-path`, no override);
"candidate" means this branch against `379e610f` through the override. Logs are in the session scratchpad
(`baseline-*.log`, `candidate-*.log`).

### 5.1 Full suites

| Run | Command | Result |
|---|---|---|
| Baseline, Debug, swiftbuild | `swift test --scratch-path <scratch>` (build) then `swift test --skip-build --scratch-path <scratch>` | 1780 tests, **128 failures in 76 tests**, 4 skipped, 275 s. Every failure is the packaging finding of section 6 (runtime-compiled library has no kernels) |
| Candidate, Debug, swiftbuild | `swift build --build-tests` then `swift test --skip-build` | 1798 tests, 133 failures in 80 tests, 4 skipped, 237 s. The same 76 as the baseline plus 4 new: two defects in the first version of the new tests (FP32 arithmetic in an assertion; 601 candidates, below `minVectorsForGPU = 1000`), the empty-`SoACandidateSet` defect (fixed, section 2) and `NormalizationParityTests.testCPUFallbacksMatchVectorCore` on the all-subnormal fixture (the genuine Core change of row BH-33, now pinned as a known divergence). After those changes the filtered re-run of `VectorCore040CompatTests`, `NormalizationParityTests`, `SoACandidateSetTests`, `MetalComputeProviderSoATests`, `SoAScoringParityTests` reports 40 tests, 10 failures all in the single packaging test `testRuntimeCompiledLibraryContainsNormalizeKernels` |
| **Candidate, Debug, native** | `swift test --build-system native` | **1799 tests, 0 failures, 2 skipped** (`IVFValidationTests.testThroughputScalesWithNprobe`, `testThroughputStability`: perf-gated placeholders), 239 s. This is the configuration the owner's recorded gates correspond to; the bundle carries the 27 `.metal` sources and the hardening suite's runtime leg runs |
| Candidate, Release, swiftbuild | `swift test -c release` | 1800 tests, 128 failures in exactly the same 76 packaging tests as the Debug baseline, 4 skipped (the two IVF throughput placeholders and two `EpsilonCompileParityTests` that need the runtime library), 52 s. All 18 compatibility tests and both new normalization tests pass in Release; Release loads `default.metallib` here (section 6) |
| **Candidate, Release, native** | `swift test -c release --build-system native` | **1800 tests, 0 failures, 2 skipped** (the two IVF throughput placeholders), 55 s. Release here runs the runtime-compiled library from the 27 bundled `.metal` sources (no `default.metallib` in the native bundle), i.e. the configuration the owner's historical release gate measured |

The difference between 1780 and 1799 is the 18 new compatibility tests plus
`NormalizationParityTests.testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere`
(`testUpstreamNormalizationReportsD1D2D3ResolvedInVectorCore040` was added after the native run; it is covered by the
filtered runs below and by the Release runs).

### 5.2 CPU fallback and GPU execution, separately

Every GPU assertion in `VectorCore040CompatTests` requires `routingTelemetry().gpuKernel >= 1`,
`cpuFallbackAfterGPUError == 0` and `cpuFallbackEmptyGPUResult == 0`, with `fallbackToCPU: false`; the CPU
twins require `gpuKernel == 0`. Both legs passed in every candidate run. `RoutingProvenanceTests` (unchanged)
pins the scale gates independently. GPU-served calls under the candidate therefore include: Euclidean and cosine
`batchDistance` at 2000 x 128 through the `BatchKernelProvider` hook, fused distance + Top-K at 5000 x 64, k = 10
through `Operations.findNearest` and `findNearestBatch`, the lane-major SoA kernel (zero-copy), and a 13-dimensional
search.

### 5.3 The new tests against 0.3.3 (control)

`swift build --build-tests --scratch-path <scratch>` then
`swift test --skip-build --scratch-path <scratch> --filter 'VectorCore040CompatTests|NormalizationParityTests'`
against VectorCore 0.3.3: 30 tests, 4 failing. Exactly the three assertions marked "0.4.0 behaviour" fail in the
way the Core change predicts, plus the packaging test:

| Test | At 0.3.3 | Meaning |
|---|---|---|
| `NormalizationParityTests.testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere` | Core returns the input (norm 2.26e-39) | BH-33 is a real change |
| `NormalizationParityTests.testUpstreamNormalizationReportsD1D2D3ResolvedInVectorCore040` | D1 produces `+Inf` | the 0.6.0 reports describe 0.3.3; resolved at the candidate |
| `VectorCore040CompatTests.testCosineEdgePolicyMatchesCoreOnGPUAndCPU` | precondition: `CosineDistance` does not return NaN for non-finite coordinates | BH-14 is a real change |
| `NormalizationParityTests.testRuntimeCompiledLibraryContainsNormalizeKernels` | packaging, section 6 | unrelated |
| `VectorCore040CompatTests.testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance` (final version, run separately with `--filter`) | the `any IndexableVector` witness answers `false` for all 65 vectors while the concrete `Vector<Dim512>.isNormalized` computes the check (`\|‖v‖²−1\|` 0 to 2.4e-7 at 0.3.3); 0 of 64 report normalized, so the dot path is never taken | BH-19 is a real change in witness dispatch, not in the concrete member |

The first version of that test passed at 0.3.3 too, because it read `isNormalized` through the concrete type;
it now reads through the existential, the path `KernelDistanceProviders` use. Its first guarded-less version also
crashed the test process at 0.3.3 (indexing an empty array after the failed precondition); it now returns after the
precondition fails. Note the deviations differ by Core version because `normalized()` itself changed (0.3.3 divides
by the FP32 magnitude, the candidate uses `NormalizationCore.plan`): worst 2.4e-7 at 0.3.3, 1.19e-6 at the candidate.

### 5.4 Remaining failures

At the time of section 5.1: under the default `swiftbuild` build system the 76 packaging failures of section 6,
pre-existing and identical at the baseline; under the native build system none. No failure is attributable to the
Core candidate. Section 10 records the owner's packaging decision and the gates after it: none remain under either
build system.

## 6. Pre-existing environment finding: the runtime shader compile path is empty under Swift Build

This is independent of the Core change (identical at the pristine 0.3.3 baseline) and is recorded here because it
decides which library a test or a release binary actually runs.

Mechanism. `KernelContext.loadMetalLibrary` tries, in order, `debug.metallib` (Debug only), `default.metallib`,
a runtime compile of the bundled `.metal` sources, then the device default library. Under the previous
toolchain `swift build` (SwiftPM's native build system) copied the 30 `.metal` sources into the resource bundle
and shipped no `default.metallib`, so Release ran the runtime compile and the hardening suite's "runtime" leg
exercised it. Under Swift 6.4.2's default `swiftbuild` backend the `.process("Metal/Shaders")` rule compiles the
sources into `default.metallib` and copies none of them (the bundle holds `debug.metallib`, `default.metallib`
and `Metal4Common.h`). `makeLibraryFromBundleSources` then finds no sources, compiles only its preamble and
returns a library with no kernels, silently (`if let url = bundle.url(...)` skips missing files).

Consequences observed on this host:

| Build system | Debug library | Release library | Hardening "runtime" leg |
|---|---|---|---|
| `swiftbuild` (default) | `debug.metallib` | `default.metallib` | 76 tests in 18 classes fail with `XCTUnwrap failed: expected non-nil value of type "MTLFunction"` or "runtime-compiled library is missing 143 kernel(s)" (BoruvkaBoundsTests, EuclideanRangePolicyTests, TopKNaNPolicyTests, FastMathPolicyTests, VectorAlignmentTests, NaNReductionPolicyTests, MinkowskiRangePolicyTests, IVFListInstrumentationTests, PQBoundsTests, SparseTFIDFBoundsTests, IVFCandidateBoundsTests, FusedTopKInstrumentationTests, UMAPNegativeSamplingTests, ShaderLibraryCompletenessTests, NeuralLatentNormalizationTests, IVFBenchmarkFixtureGPUTests, IgnoredFlagTests, NormalizationParityTests.testRuntimeCompiledLibraryContainsNormalizeKernels) |
| `--build-system native` (deprecated) | `debug.metallib` | runtime compile | passes: the bundle holds the 27 `.metal` sources plus `debug.metallib` and no `default.metallib`; the full candidate suite ran 1799 / 0 failures / 2 skipped (section 5) |

Also under `swiftbuild`, `swift build -c release --build-tests` fails to compile the `@testable` test target
("unable to resolve Swift module dependency to a compatible module: 'VectorAccelerate'"); `swift test -c release`
is the working form and is the owner's historical gate command.

**Owner decision 2026-10-31: `default.metallib` is the shipped library** (option A). Implemented in section 10:
the runtime compile stays as a fallback that throws when the bundle carries no sources or an incomplete set, and
the hardening suites certify whichever libraries the bundle ships. The compiler plugin's `debug.metallib` still
ships in Release bundles (1.7 MB beside the 0.97 MB `default.metallib`); gating or dropping it is a separate
packaging change not made here.

## 7. Core requests and suspected Core defects

No executed suite exposed a wrong result in the candidate. Three items are worth sending upstream; none is a
crash or a wrong number, and VectorCore was not modified here.

**C1. `BatchKernelProvider` still takes `any DistanceMetric` while the Float facades moved to
`any DistanceMetric<Float>` (R01).** `Operations.findNearest` is constrained to `M.Scalar == Float`, so every
metric that reaches the provider is a Float metric, but the protocol erases that: `MetalComputeProvider` must
downcast (`metric as? any DistanceMetric<Float>`) and throw `invalidInput` on a mismatch that cannot occur
through `Operations`. Request: narrow the three `BatchKernelProvider` requirements to `any DistanceMetric<Float>`
in the same breaking release, with the witness-signature warning CB-15/SG-05 already carry (an un-narrowed
downstream override would otherwise become a dead overload). VectorAccelerate will change its three witnesses in
step. Evidence: `Sources/VectorCore/Protocols/BatchKernelProvider.swift` at `379e610f` (comment-only diff from
0.3.3); `MetalComputeProvider.batchDistance(query:candidates:metric:)`.

**C2. N01 / row BH-33: GPU consumers cannot follow the all-subnormal normalization.** Minimal reproducer (pure
Core, Debug or Release):

```swift
import VectorCore
var v = [Float](repeating: Float(1e-40 as Double), count: 512)   // every coordinate subnormal
v.withUnsafeMutableBufferPointer { NormalizeKernels.normalizeUnchecked($0.baseAddress!, dimension: $0.count) }
print(v[0])   // 0.3.3: 1e-40 (unchanged)   candidate: 0.044194173 == 1/sqrt(512)
```

The candidate's answer is the better one, but Metal's fast-math mode flushes subnormal inputs to zero before any
arithmetic, so a GPU kernel cannot form the scaled terms and VectorAccelerate keeps the pass-through for this
class (and mirrors it on its CPU fallbacks so its five implementations agree). Request: a sentence in the N01
CHANGELOG entry or `Docs/Memory_Alignment.md` stating that implementations under flush-to-zero may pass the
all-subnormal class through, so the divergence is documented on both sides. Pinned downstream by
`NormalizationParityTests.testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere`.

**C3. Row BH-19: the `Vector<D>.isNormalized` witness misses freshly normalized unit vectors at 512 dimensions.**
The check is `abs(magnitudeSquared - 1) < 1e-6` on an FP32 `magnitudeSquared`, whose accumulation error over
512 squared coordinates is of the same order as the threshold. Observed on this host with the candidate:
64 random 512-d vectors passed through `Vector<Dim512>.normalized()`: 63 report `isNormalized`, one does not,
worst `|‖v‖² − 1|` = 1.1920929e-06 (ten ulps of 1, against a threshold of 1e-6). `VectorCore040CompatTests.testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance`
asserts the witness is exactly the documented check and records the count in its failure message. Consequence for consumers that route on the hint (an index switching cosine to a
dot product): a batch of normalized `Vector<D>` takes the dot path only when every member happens to pass the
check, so the route is data-dependent rather than a property of "normalized". Minimal reproducer (pure Core):

```swift
import VectorCore
var seed: UInt64 = 1
func next() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float((seed >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1 }
let unit = try (0..<64).map { _ in try Vector<Dim512>((0..<512).map { _ in next() }).normalized().get() }
print(unit.filter(\.isNormalized).count, "of 64 report isNormalized;",
      "worst |‖v‖²−1| =", unit.map { abs($0.magnitudeSquared - 1) }.max()!)
```

Request: either a dimension-aware tolerance (for example `sqrt(D) * 2^-23`, or compute the check in Double),
or a documented statement that `Vector<D>.isNormalized` is a strict FP32 check that can report `false` for a
unit vector, so hint consumers do not read it as "is a unit vector". Not a wrong result: `CosineDistance`
itself is unaffected.

**C4. Row LM-02: a tag.** Until the candidate is tagged, `Package.swift` cannot pin it (`revision:` needs a
pushed commit), and every downstream build needs the local `swift package edit` override, which also drops the
dependency from `Package.resolved`.

## 9. Compiler finding VC-C001: VectorAccelerate exposure review (2026-10-30, evening)

VectorCore's compiler investigation (`/Users/goftin/dev/projects/VectorCore-compiler-investigation-2026-10-30-evidence`,
`REPORT.md`, `FINDINGS.md`) confirmed that Apple Swift 6.4.2 and 6.2 miscompile one shape at -O, -Osize and
-O with WMO: an array literal held in a local, a `withUnsafeMutableBufferPointer` borrow whose closure hands the
buffer to an out-of-line writer `inout`, then an element read, after or inside the borrow. RedundantLoadElimination
forwards the literal across the write. By-value writers and non-literal initializers keep the store. Core's Release
gate stays red by its own canary; the report asks VectorAccelerate for its own tests because a Core-only flag cannot
protect client code. This section records what was checked here and what was added.

### 9.1 Exposure review

| Check | Result |
|---|---|
| `inout Unsafe[Mutable][Raw][Buffer]Pointer` parameters in `Sources/` and `Tests/` | none before this work; the only one now is the canary's writer |
| Mutable borrows in `Sources/` | 8 sites in 6 files (`MatrixEngine` x3, `StableNormalization`, `SIMDFallback`, `QuantizationEngine`, `UMAPGradientKernel`, `PerformanceMonitor`); every owner is `[T](repeating:count:)` or `Data(count:)` and every body writes through vDSP, a C call or an inline loop; no literal owner, no `inout` hand-off |
| Mutable borrows in `Tests/` | `NormalizationParityTests` x4 (pointer passed by value to `NormalizeKernels.normalizeUnchecked`, owners non-literal), `FactoryUploadBoundsTests` x2 (a test conformer forwarding by value) |
| Core's lexical scanner (`Scripts/ci/tests/test_lost_store_reads.py` at the candidate) run over the whole VectorAccelerate tree (240 files, driver in the session scratchpad) | 1 post-borrow-read hit: `PerformanceMonitor.updateResourceMetrics::info`, the `task_info` out-pointer idiom (pointer passed by value into a C call, not the inout-writer trigger). 20 captured-variable hits, the form Core flags by class and never reproduced: `CPUTopK.search::heap`, `LearnedDistanceKernel.compute::result` x2, `UMAPGradientKernel` `result` x2, `QuantizationEngine` `result` x3, `HDBSCANDistanceModule.computeMST::result`, `MemoryMapManager.extractVector::result`, `SIMDFallback` `sum`/`sumSquared` x4, `SIMDOptimizer` `sum` x4, `StableNormalization.normalizedSIMD` `scale`/`sumSquares` |
| Full Release suite, native build system, candidate, with `-Xswiftc -Xllvm -Xswiftc -sil-disable-pass=redundant-load-elimination` applied to all ten Swift modules (verified in `.build/release.yaml`) | 1800 tests, 0 failures, 2 skipped: indistinguishable from the default-optimization run, so no VectorAccelerate test could tell the two compilers apart |

Half of the captured-variable sites are numerical accumulators in CPU fallbacks. They are the class VectorCore pinned as
keeping their writes (K1-K5, K7, K8) on one host; the VectorAccelerate Release suite that exercises them passes. They are
not rewritten here.

### 9.2 Added

- `Tests/VectorAccelerateTests/Hardening/CompilerLostStoreCanaryTests.swift`: the exact miscompiled shape (post-borrow read
  and the inside-borrow variant) plus three controls that pin the shapes VectorAccelerate relies on (by-value writer,
  out-pointer read inside the borrow, captured accumulator). In an optimized build on a compiler where the miscompile is
  demonstrated (`6.2 <= compiler < 6.5`) a lost store is an expected, non-strict failure, so the gate stays green and the
  log records the toolchain's condition; on any other compiler in Release, or in Debug, the store must be observed.
  **Owner decision 2026-10-30: keep this green-gate stance** (Core keeps its own gate red); the `XCTExpectFailure` branch
  stays, and a toolchain outside the demonstrated range that still loses the store will turn the gate red for
  re-qualification.
- `Tests/VectorAccelerateTests/Hardening/LostStoreShapeGuardTests.swift`: a Swift port of the Core scanner's rules that
  apply here, over `Sources/**/*.swift` and `Tests/**/*.swift` with comments and strings blanked: no `inout` unsafe-pointer
  parameter anywhere, no literal-initialized local read after a `withUnsafeMutableBufferPointer` borrow in its scope, no
  use of `x` after `withUnsafeMutablePointer(to: &x)` in its scope. Allowlist with reasons; stale entries fail; fixture
  controls prove the scanner sees the canary shapes and accepts the W2 inside-borrow read, by-value writers,
  `[T](repeating:count:)` owners and store-only follow-ups.
- `PerformanceMonitor.updateResourceMetrics` now reads `resident_size` inside the borrow (W2), so production code needs
  no allowlist entry.

### 9.3 Results

| Run | Result |
|---|---|
| Debug, swiftbuild: `swift test --filter 'CompilerLostStoreCanaryTests\|LostStoreShapeGuardTests'` | 9 tests, 0 failures; at -Onone both canary shapes observe the store (1.0) |
| Release, native: `swift test -c release --build-system native --filter ...` | 9 tests, 0 failures, 2 expected failures: both canary shapes observe `2.0 (bits 0x40000000)` where memory holds 1, i.e. **the VC-C001 miscompile reproduces inside VectorAccelerate's own optimized test module** and is recorded, not hidden. The three controls (by-value writer, out-pointer read inside the borrow, captured accumulator) pass |
| Release, swiftbuild: `swift test -c release --filter ...` | identical: 9 / 0 / 2 expected failures, same bits |
| Guard over the real tree (`testNoUnreviewedLostStoreShapeInSourcesOrTests`) | 242 files, 0 unreviewed hits, 3 allowlisted (all in the canary file), 0 stale entries; fixture controls prove the scanner flags the canary shapes and accepts the W2 read, by-value writers, `[T](repeating:count:)` owners and store-only follow-ups |
| Full gates with the new suites, native build system, candidate | `swift test -c release --build-system native`: **1809 tests, 0 failures, 2 skipped, 2 expected failures** (the canary). `swift test --build-system native`: **1809 tests, 0 failures, 2 skipped, 0 expected failures** (at -Onone the canary observes the store) |

Reading of the canary result: the compiler VectorAccelerate is built with does miscompile the shape, in VectorAccelerate's
own module, exactly as Core's investigation predicted; VectorAccelerate's production code does not contain the shape
(guard), its own optimized paths pass with and without the RLE pass (section 9.1), and the controls show the shapes it
does use keep their stores on this compiler. The canary is the signal that would change first on a toolchain update.

## 8. Follow-ups for the owner

1. When the candidate is tagged: `swift package unedit VectorCore`, raise the floor to `from: "0.4.0"`,
   re-resolve, commit `Package.resolved`, re-run the gates in sections 5 and 10.
2. Shader packaging, BH-19 and Chebyshev NaN were decided on 2026-10-31 (section 10). Open from that decision:
   whether the compiler plugin's `debug.metallib` should keep shipping in Release bundles, and the fact that every
   consumer build host now needs the Metal Toolchain component installed.
3. `acceleratedNormalizeUnchecked()`'s CPU branch now stops the process on zero or non-finite input (BH-61);
   consider routing it through `normalized()` or documenting the precondition.
4. Optional adoptions: `ComputeDevice.isSupportedByCore` in `MetalDevice.asComputeDevice()` docs (NA-02),
   `relevanceScore(for:)` to populate `IndexSearchResult.score` (NA-03), `VectorCoreCompiledFeatures` in
   diagnostics (NA-10). All are 0.4.0-only symbols and were left out so the tree still compiles against 0.3.3.
5. After any build under the override, `git checkout Package.resolved` before committing.

## 10. Decisions of 2026-10-31 and their implementation

Three decisions were put to the owner after sections 6 and 9; all three took option A.

### 10.1 `default.metallib` is the shipped library

- `KernelContext.makeLibraryFromBundleSources` now throws `shaderCompilationFailed` when the bundle carries no
  `.metal` sources, and when it carries an incomplete set (a missing listed file used to be skipped silently,
  AUDIT-2 VA2-001). `KernelContext.bundledShaderSourceURLs(in:)` exposes what the bundle holds.
  `loadMetalLibrary`'s order is unchanged: `debug.metallib` in Debug, `default.metallib`, the runtime compile,
  then the device default library; in production the throw is caught and the next step is tried.
- New test utility `Tests/VectorAccelerateTests/Utilities/ShippedMetalLibraries.swift` enumerates the libraries
  this build's bundle can load, labelled `plugin` (`debug.metallib`, Debug only), `spm` (`default.metallib`) and
  `runtime` (only when sources are bundled), and fails if none ships. The 18 hardening suites that drove a
  hard-coded "runtime" leg now iterate over it; their test bodies are unchanged.
- `ShaderLibraryCompletenessTests` asserts every kernel of every non-excluded shader file in every shipped library,
  and that every shipped library exposes the same kernel set, which under Swift Build in Debug compares the
  plugin's `debug.metallib` with SPM's `default.metallib` for the first time. `NormalizationParityTests` checks
  the normalize kernels in every shipped library. `EpsilonCompileParityTests` skips, with the reason stated,
  when there is no preamble-driven runtime compile to compare. `VectorAlignmentTests` reads the shader source it
  instruments from the repository instead of the bundle.
- What this costs is listed in section 6: every consumer build host needs the Metal Toolchain, the compile flags of
  the shipped library come from Xcode's defaults, and the plugin's debug library still ships in Release.

### 10.2 BH-19: accepted as tested

No code change. `testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance` and the index's hint recording stand
as the contract; the threshold question stays with Core as request C3.

### 10.3 Chebyshev propagates NaN

Every VectorAccelerate Chebyshev path now matches Core's `ChebyshevDistance`: a NaN coordinate, or the same infinity
on both sides, gives NaN; an infinite difference gives +Inf; finite input is the exact maximum absolute difference.
CPU paths check `isNaN` per difference; the two Metal kernels (`chebyshevDistance` in `DistanceShaders.metal`, the
L∞ branch of `minkowski_distance_batch`) detect NaN on the integer bit pattern and carry a flag, because `max` drops
NaN and fast math may fold `isnan`. `ChebyshevNaNPolicyTests` pins `MetalComputeProvider` (single and batch),
`FallbackProvider`, `Metal4ComputeEngine` on its CPU branch and through the kernel, and
`MinkowskiDistanceKernel.chebyshevDistances` at dimensions 8 and 259 against Core's witness; `BatchProcessor` and
`MemoryMapManager` carry the same loop and are not separately exercised.

### 10.4 Gates after the decisions

Full suite against the candidate, after the three decisions (1814 tests: the 1809 of section 9 plus
`ChebyshevNaNPolicyTests`). The two `IVFValidationTests` throughput placeholders are perf-gated skips in every row;
the two canary expected failures appear in every optimized row.

| Build system | Configuration | Command | Result |
|---|---|---|---|
| swiftbuild (default) | Debug | `swift test` | **1814 tests, 0 failures**, 4 skipped (the two IVF placeholders plus the two `EpsilonCompileParityTests`), 240 s |
| swiftbuild (default) | Release | `swift test -c release` | **1814 tests, 0 failures**, 4 skipped (the two IVF placeholders plus the two `EpsilonCompileParityTests`, which skip with the stated reason because no preamble compile exists in this bundle), 2 expected failures, 54 s |
| native | Debug | `swift test --build-system native` | **1814 tests, 0 failures**, 2 skipped, 236 s; `EpsilonCompileParityTests` runs here because the sources are bundled |
| native | Release | `swift test -c release --build-system native` | **1814 tests, 0 failures**, 2 skipped, 2 expected failures, 54 s |

What each row certifies: under swiftbuild the `spm` library (`default.metallib`) in Release and `plugin` plus `spm` in
Debug, including the first plugin-versus-SPM kernel-set comparison; under native the `runtime` compile in Release and
`plugin` plus `runtime` in Debug. The 76 failures of section 6 are gone on both build systems. The first swiftbuild
Debug run after the change had exactly one failure, `VectorAlignmentTests.testNeuralTailNeverRequestsWeightsPastBufferEnd`,
which read the shader source it instruments from the bundle; it now reads it from the repository.
