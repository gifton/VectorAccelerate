# VectorAccelerate against VectorCore 0.4.0-alpha.1: alpha-only upgrade rows

This file records the audit of the eleven rows that VectorCore's upgrade list (`DOWNSTREAM_UPGRADES.md`) added after the October candidate that `VECTORCORE-0.4.0-CANDIDATE.md` covers. The audit was run on the branch `gifton/vectorcore-0.4.0-alpha1` at the commit that adds this file (the parent of that commit is the tree that was searched; this file and the guard test `VectorCore040CompatTests/testLinkedVectorCoreIsTheAlpha` are the only changes). Every `Search run` cell is run from the worktree root. Hit counts are lines matched by that command (`| wc -l`).

| Row | What changed (one line, from the upgrade list) | Search run | Hits in Sources | Hits in Tests | Conclusion |
|---|---|---|---|---|---|
| CB-43 | Public dynamic storage classes (`AlignedDynamicArrayStorage`, `DynamicArrayStorage`) are now copy-on-write structs; mutation through `let` and class identity/weak references no longer compile. | `git grep -n -e 'AlignedDynamicArrayStorage' -e 'DynamicArrayStorage' -- Sources Tests` | 0 | 0 | not used |
| CB-44 | watchOS is no longer a supported platform (`Package.platforms`). | `sed -n '/platforms: \[/,/\]/p' Package.swift` prints `.macOS(.v26), .iOS(.v26), .tvOS(.v26), .visionOS(.v26)`; `git grep -n 'watchOS' -- Package.swift Sources Tests` returns only the comment `Package.swift:84: // watchOS removed - no Metal 4 support (no GPU)` | 0 | 0 | not used (this package's `platforms:` list names no watchOS) |
| CB-45 | `SIMDStorage.withUnsafeMutableBufferPointer` is replaced by `withUnsafeMutableElements`; no legacy fallback; concrete ContiguousArray stdlib calls remain available. | `git grep -n -e 'SIMDStorage' -e 'withUnsafeMutableElements' -e 'withUnsafeMutableBufferPointer' -- Sources Tests` | 6 | 18 | not used: no `SIMDStorage` or `withUnsafeMutableElements` hit; the 6 Sources hits are `[Float]` / `[[Float]]` element calls (`UMAPGradientKernel.swift:1191`, `MatrixEngine.swift:247,327,401`, `SIMDFallback.swift:304`, `StableNormalization.swift:178`), the stdlib array method the row says remains available |
| BH-62 | Copies of either public dynamic storage isolate scalar and mutable-buffer writes; the wrapper delegates COW. | `git grep -n -e 'AlignedDynamicArrayStorage' -e 'DynamicArrayStorage' -- Sources Tests` | 0 | 0 | not used |
| BH-63 | Collection replacement/clearing destroys displaced values after unlocking (`SimpleVectorCollection.update`, `removeAll`). | `git grep -n -e 'SimpleVectorCollection' -e 'MutableVectorCollection' -e 'VectorCollection' -- Sources Tests` | 0 | 1 | not used (the one Tests hit is a doc comment at `VectorCore040CompatTests.swift:119` saying this package does not call `VectorCollection.search`) |
| BH-64 | Finite interpolation uses widened arithmetic and preserves endpoint bits (`Vector.lerp`, `lerpUnclamped`, `linspace`). | `git grep -n -e 'lerp' -e 'linspace' -e 'VectorInterpolation' -- Sources Tests` | 1 | 0 | not used (the one hit is a comment in the Metal shader `Sources/VectorAccelerate/Metal/Shaders/StatisticsShaders.metal:308` about `metal::mix`, not a VectorCore call) |
| BH-65 | The squared-distance batch shortcut applies only to coordinate-zero queries (`EdgeCaseHandler.handleBatchDistanceEdgeCases`, reached only from `BatchKernels_SoA.safeBatchEuclidean512`). | `git grep -n -e 'handleBatchDistanceEdgeCases' -e 'safeBatchEuclidean512' -e 'EdgeCaseHandler' -e 'BatchKernels_SoA' -- Sources Tests` | 0 | 0 | not used |
| BH-66 | Precision diagnostics retain Double statistics through outlier classification (`MixedPrecisionKernels.analyzePrecision`). | `git grep -n -e 'analyzePrecision' -e 'AnalyzableVector' -e 'MixedPrecisionKernels' -- Sources Tests` | 0 | 0 | not used |
| AB-09 | Two public storage types changed from classes to structs; rebuild downstream binaries. | `git grep -n -e 'AlignedDynamicArrayStorage' -e 'DynamicArrayStorage' -- Sources Tests`; `git grep -n -e 'binaryTarget' -e 'xcframework' -- Package.swift` (0 hits) and `git ls-files '*.xcframework' '*.a' '*.dylib'` (0 lines) show no prebuilt binary; `Package.swift` takes VectorCore as `.package(url: "https://github.com/gifton/VectorCore", exact: "0.4.0-alpha.1")`, a source dependency built in the same build | 0 | 0 | not used (nothing links a prebuilt VectorCore binary; it is compiled from source with this package) |
| AB-10 | `SIMDStorage` witness requirements change and `ContiguousArray` gains `withUnsafeMutableElements(_:)`; rebuild VectorCore and downstream together. | `git grep -n -e 'SIMDStorage' -e 'withUnsafeMutableElements' -- Sources Tests`; prebuilt-binary check as in AB-09 (`git grep -n -e 'binaryTarget' -e 'xcframework' -- Package.swift`, 0 hits) | 0 | 0 | not used (same source-dependency build as AB-09) |
| NA-14 | Public initializers for `Tuning` and `CalibrationProbes`. | `git grep -n -e 'Tuning' -e 'CalibrationProbes' -- Sources Tests` | 0 | 0 | not used |

## Pins

| Item | Value | Source |
|---|---|---|
| VectorCore (baseline) | 0.3.3, revision `fca4b602383589c46b627d8a0de2b6b2a68cd07d` | `Package.resolved` before commit `92ffafa` |
| VectorCore (this branch) | `exact: "0.4.0-alpha.1"`, revision `44c9ffaf145a59688f420023b28cdcf87c158070` | `Package.swift`, `Package.resolved` |
| Base commit of this branch | `c768d7f7fc26689af175c88e637c13174de16538` (merge base with `origin/main` as checked at the final run; `origin/main` has since moved to `0eebd927399792a35bc24ea070c8e071eb868f52`, which the branch history does not contain) | `git merge-base HEAD origin/main` |
| Branch tip before this record's commit | `65156c2d524fc4a1b2636fd2c011e7f643a3aed0` | `git rev-parse HEAD` |

Commits on the branch after the plan commit `310bff4`: `92ffafa` (pin) and `5b39d01` (pin comment reworded), `bfa4095` (migration), `65156c2` (guard test and the audit table above).

## Environment

| Item | Value |
|---|---|
| Host | `mac.lan`, Apple M3 Max (`sysctl -n machdep.cpu.brand_string`) |
| macOS | ProductName macOS, ProductVersion 27.0.1, BuildVersion 26A434 (`sw_vers`) |
| Xcode | Xcode 27.2, Build version 27B52 (`xcodebuild -version`) |
| Swift | swift-driver version: 1.168.6 Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1), Target: arm64-apple-macosx27.0.0 (`swift --version`) |
| Metal Toolchain | present: `xcrun -f metal` printed `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.2.47.0.O9H9X8/Metal.xctoolchain/usr/bin/metal` |

Another job was using the GPU lightly on this machine while the suites ran. No test failed in any run, so no re-run was needed.

## Commands and totals

Debug is `swift test`; Release is `swift test -c release` (`swift build -c release --build-tests` does not link the `@testable` target). No filters or parallel flags. "Executed", "failed" and "skipped" are the runner's final `Executed N tests, with S tests skipped and F failures` line.

| Stage | Configuration | Executed | Failed | Skipped | Log |
|---|---|---|---|---|---|
| Baseline, VectorCore 0.3.3 | Debug | 1804 | 0 | 4 | `baseline-debug.log` |
| Baseline, VectorCore 0.3.3 | Release | 1804 | 0 | 4 | `baseline-release.log` |
| Alpha pinned, before migration (`5b39d01`) | Debug | 1804 | 1 | 4 | `premigration-debug.log` |
| Alpha pinned, before migration (`5b39d01`) | Release | 1804 | 1 | 4 | `premigration-release.log` |
| After migration (`bfa4095`) | Debug | 1829 | 0 | 4 | `postmigration-debug.log` |
| After migration (`bfa4095`) | Release | 1829 | 0 | 4 | `postmigration-release.log` |
| Final, at `65156c2` | Debug | 1830 | 0 | 4 | `alpha1-debug.log` |
| Final, at `65156c2` | Release | 1830 | 0 | 4 | `alpha1-release.log` |

Final-run summary lines: Debug `Executed 1830 tests, with 4 tests skipped and 0 failures (0 unexpected) in 241.330 (241.451) seconds`; Release `Executed 1830 tests, with 4 tests skipped and 0 failures (0 unexpected) in 53.526 (53.636) seconds`. Both commands exited 0. The logs are build evidence kept outside the repository (they are git-ignored); the numbers above were read from them.

Skips are the same four in the baseline and the final run, in both configurations (a sorted comparison of the `Test Case ... skipped` lines): `EpsilonCompileParityTests` `testHistogramDegenerateGateAgreesAcrossBuildPaths` and `testHistogramNormalRangeAgreesAcrossBuildPaths` (reason in the baseline: this build ships only precompiled metallibs, so the preamble compile path does not exist here), and `IVFValidationTests` `testThroughputScalesWithNprobe` and `testThroughputStability`. No new skip.

Compiler canary (`CompilerLostStoreCanaryTests`, Release): the Release log of the final run contains two `Expected failure in` lines, at `CompilerLostStoreCanaryTests.swift:146` (`testLiteralArrayStoreIsObservedAfterOutOfLineInoutWriter`) and `:150` (`testLiteralArrayStoreIsObservedWhenReadInsideBorrowAfterInoutWriter`), each reporting `VC-C001 ... observed 2.0 ... expected 1`. The test records a lost store on a compiler in the demonstrated range (6.2 <= compiler < 6.5) as an expected, non-strict failure (`isStrict = false`), so the runner counts those tests as passed. The same two lines appear in the baseline Release log. The Debug logs contain none. So "0 failures" in Release includes two declared expected failures that show the miscompile reproducing on this compiler; the suite is green by that design, not because the miscompile is absent. (`grep -c 'Expected failure in'`: Debug 0, Release 2.)

## First build against the alpha

`swift build --build-tests` after the pin, with no source change: built clean, `Build complete!` (13.30 s), exit 0, 0 errors, 52 warnings (`first-build.log`, `first-build.exit`). None of the warnings names a VectorCore symbol; they are of kinds that already existed (for example the deprecated `StreamingTopKKernel`). The 13 s duration suggests an incremental build, but `VectorCore` and `VectorCoreC` were recompiled and the test product linked. After the migration commit the same command gave `Build complete! (13.26 sec)` with 0 `error:` lines (`post-migration-build.log`).

## What changed outcome

Against the baseline, in both Debug and Release at the final run (sorted comparison of `Test Case ... passed|failed|skipped` lines, `baseline-*.log` against `alpha1-*.log`):

- No test that ran in the baseline has a different outcome. All 1804 baseline test cases are present and passed or skipped as before. (A line-by-line comparison of the Debug logs shows one apparent swap, `MatrixEngineEnhancedTests testBatchMatrixVectorMultiply` against `testBatchOperationPerformanceComparison`; it is console output of the first test running into the log's "passed" line. Both tests are reported passed in `alpha1-debug.log` at lines 3891 and 3893.)
- 26 test cases are new, all passed in both configurations (see "Tests added").

Intermediate state, not in the final result: on the alpha before the migration, one test failed in both configurations, `NormalizationParityTests.testCPUFallbacksMatchVectorCore` (`NormalizationParityTests.swift:317`: `VectorCore.normalizeUnchecked/all-subnormal(1e-40): max |Δ| = 0.044194173 at [0] (got 1e-40 vs 0.044194173)`). Cause: VectorCore 0.4.0 normalises a vector whose coordinates are all subnormal (upgrade row BH-33), and the old fixture still compared VectorAccelerate against it.

Tests changed (from the diff `5b39d01..bfa4095`; only `NormalizationParityTests.swift` among existing tests changed):

- `NormalizationParityTests`: the `all-subnormal(1e-40)` fixture now has `crossCheckVectorCore: false`. It is still tested for what VectorAccelerate does, which is to return the input bit-exactly unchanged (`isNormalizable: false`), on the GPU and the CPU paths. It is no longer compared against VectorCore. Comments in the file were updated to say the pass-through matches VectorCore 0.3.3's `normalizeUnchecked`, not 0.4.0's.
- How the class is treated now: VectorCore 0.4.0 normalises the all-subnormal class; VectorAccelerate's implementations pass it through; the difference is pinned by a named test rather than removed, `NormalizationParityTests.testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere`. It asserts that VectorCore's `NormalizeKernels.normalizeUnchecked` returns a unit vector (`1/sqrt(512)` per element) and that `StableNormalization.normalizedAccelerate` returns the input bit-exactly. Reason, as the test's own comment gives it: VectorAccelerate keeps the unified pass-through policy for this class because Metal's fast-math mode flushes the subnormal inputs to zero before any arithmetic, so the GPU cannot form the scaled terms, and the CPU fallbacks mirror the GPU so all five implementations agree with each other. This record adds no other reason and does not claim the divergence is acceptable; it states that it exists and is pinned.
- Source behaviour that came with the migration commit (not a test outcome change in the baseline suite): Chebyshev distance propagates NaN on VectorAccelerate's paths, matching VectorCore 0.4.0 (row BH-15); pinned by `ChebyshevNaNPolicyTests`.

## Tests added

| Suite | Added | Count in suite at the final run | Added by |
|---|---|---|---|
| `ChebyshevNaNPolicyTests` | whole suite | 5 | `bfa4095` |
| `NormalizationParityTests` | `testAllSubnormalClassIsNormalizedByVectorCore040ButPassedThroughHere`, `testUpstreamNormalizationReportsD1D2D3ResolvedInVectorCore040` | 12 (2 new) | `bfa4095` |
| `VectorCore040CompatTests` | whole suite (18) plus `testLinkedVectorCoreIsTheAlpha` (1) | 19 | `bfa4095`, `65156c2` |

5 + 2 + 19 = 26, which is the difference between 1804 and 1830. Counts were read from the `Test Suite '...' passed` lines of both final logs. `testLinkedVectorCoreIsTheAlpha` is a guard: it fails if the linked VectorCore is not the alpha.

## Conflict resolutions that departed from the rule

Task 3 resolved 14 conflict hunks in 10 files by four rules (keep this branch's reviewed text where it already carries the work; take the cherry-picked side for the ComputeDevice availability, Chebyshev, normalization and compatibility work; keep the exact pin in `Package.swift`; keep both sides' entries in `CHANGELOG.md`). None was left unclassified. Two decisions departed from a literal reading (`unclassified-hunks.md`):

1. `Sources/VectorAccelerate/Core/MetalDevice.swift`, hunk 2 (doc comment of `isGPUAvailable()`): the cherry-picked side was taken, as the rule says, but it states "the single availability check behind every VectorAccelerate guard" without naming a test. This branch's side named `GPUAvailabilityGuardTests`, so "(`GPUAvailabilityGuardTests`)" was appended so the claim names its test.
2. `CHANGELOG.md`, the single hunk spanning `[Unreleased]`: the cherry-picked side restated work this branch already carries in reviewed form and opened with "the manifest floor stays 0.3.3", which is false here. This branch's text was kept for those topics, and the three cherry-picked entries that are new here (Chebyshev NaN, normalization parity, `VectorCore040CompatTests`) were added. The cherry-picked intro paragraph was not carried.

## Not checked

- Platforms other than this Mac (macOS 27.0.1, Apple M3 Max): iOS, tvOS, visionOS, Intel Macs, other Apple GPUs. Nothing was built or run for them.
- Compilers other than Apple Swift 6.4.2 (Xcode 27.2). The canary's behaviour on other compilers is described by the test's comments only; it was not run here on any other.
- Performance. The suites ran with wall-clock assertions off (`VECTORACCELERATE_STRICT_PERF` was not set); no benchmark was run, so nothing here says the alpha is faster or slower. The per-run durations above include other GPU load and are not measurements.
- Behaviour of VectorCore 0.4.0-alpha.1 beyond what this package's suite and the eleven audited rows touch. The audit searched `Sources` and `Tests` text for the symbols named in each row; it did not review VectorCore's code and cannot see use through other names or from dependents of this package.
- The 0.4.0 upgrade rows other than the eleven alpha-only ones above are covered by `VECTORCORE-0.4.0-CANDIDATE.md` and by the tests listed here, not re-audited in this exercise.
- Runs were made once per configuration at each stage; run-to-run variation (including flaky tests, or the GPU contention noted above) was not measured.
- The four skipped tests were not run in any configuration, so nothing is known from this record about their behaviour on the alpha.
- Whether VectorCore will change between this alpha and 0.4.0 final. This branch is pinned to the alpha and verified only against it.
- Downstream packages that depend on this branch (for example SwiftTopics) were not built.
