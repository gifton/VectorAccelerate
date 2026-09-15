# Research Brief — Observability, Guardrails, and Reproducible Benchmarks for VectorAccelerate

**Date:** 2026-09-14
**Status:** Research (pre-RFC), with the owner's decisions recorded in §6a and the timing spike result in Appendix D (both 2026-09-14).
**Scope of this document:** what the problem actually is for this codebase, what already exists (in this repo and its siblings), what mature projects and the literature do, the principles that follow, a decomposition into sub-projects, and the open decisions.
**Environment this was researched on:** Apple M3 Max (12P+4E CPU, 40-core GPU, 48 GB), macOS 26.5.2 (25F84), Swift 6.3.3, Xcode 26.6, AC power, thermal state nominal.

---

## 0. The one-paragraph version

VectorAccelerate has mature measurement *plumbing* (DCE barrier, percentiles, GPU command-buffer timestamps, a seeded ANN-style index harness, a rigorous differential-oracle test pattern, 22 stability contracts backed by 33 guard suites) and **no closed loop**: no baseline is stored, no comparison runs, no number can fail a build, and no run carries hardware or thermal provenance. Four prior attempts (library-side framework, benchmark executable, the MASTER_PLAN §6 paper design, PerfGate-gated test assertions) each built plumbing and stopped before the loop closed. Three open backlog items are blocked on exactly this gap: VA3-024 (needs before/after), the two IVF throughput placeholders (need "a controlled benchmark design"), and the README speedup table (unreproducible). The research points to one strategic rule: **close the loop on a single vertical slice first** (one kernel, one metric, one per-device baseline, one comparison that can go red), then widen. The program decomposes into six sub-projects (§5); the foundation sub-project is small and unblocks everything else.

---

## 1. The problem, stated precisely

"Visibility, guardrails, reproducible metrics" are three different things that share one substrate. Conflating them is why the prior attempts stalled: each solved part of one pillar without the substrate.

| Pillar | Question it answers | Failure mode today |
|---|---|---|
| **Observability** | *What did the library actually do at runtime?* Which path ran (GPU/CPU, fused/separate), why, how long each stage took, what the pool and pipeline cache did. | Routing provenance exists only on `MetalComputeProvider`; the decision engine returns a bare `Bool`; kernels, `Metal4Context`, and the whole `Index/` tree are silent; GPU timing is single-slot and races. |
| **Guardrails** | *What must always be true, and how loudly does it fail?* Correctness contracts, numerical tolerance, resource budgets, determinism policy, validation layers. | Correctness contracts are strong (22 docs / 33 suites). Resource and determinism guardrails are weak: no pool high-water mark, no live-lease count, no stated policy for atomic-float variation (VA3-019), validation-layer runs are hand-typed and their evidence lives in `/private/tmp`. |
| **Reproducible metrics** | *Is this number trustworthy, and is it different from last time?* Provenance, seeded inputs, controlled environment, statistics, baselines, decision rule. | Three clocks, three RNGs, unseeded shipped generators, no baselines, no comparison tool, no thermal or power provenance, no CI perf job. |

**The shared substrate** every pillar writes into:

1. an **environment fingerprint** (chip, GPU cores, OS build, Metal toolchain, Swift, build config, power source, thermal state, quiet-machine probe);
2. **deterministic inputs** (one RNG, one generator, seed and stream recorded);
3. a **result schema** with versioning and provenance;
4. an **execution surface** (a CLI that any of the three pillars can be driven from, plus a library-side instrumentation API);
5. a **decision rule** (what makes a number red).

Build the substrate once; all three pillars become thin.

---

## 2. Current state — inventory

Facts only, verified against the working tree at `135f51b` plus the uncommitted slice-34 work. File references are `path:line`.

### 2.1 Size of the thing being observed

- `Sources/`: 55,870 lines of Swift, 10,796 lines of MSL across 27 `.metal` files, ~180 kernels. Swift 6.2 tools, strict concurrency, Metal 4 only (macOS 26+).
- `Tests/`: 47,772 lines, 109 XCTest files, zero Swift Testing. Current gate expectation 1764 passed / 0 failed / 2 skipped in both debug and release.
- Dual shader compile paths (plugin `debug.metallib` under `#if DEBUG` vs runtime-concatenated source in release) are a first-class testing axis (`EpsilonCompileParityTests`, `PreambleParityTests`, `FusedTopKInstrumentationTests.withPipelines`).

### 2.2 The four prior benchmark attempts and why each stopped

| Attempt | Where | What works | Where it stopped |
|---|---|---|---|
| (a) Library-side framework | `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift` (430 L, last touched 2026-04-05), `IndexBenchmarkHarness.swift` (758 L) | `blackHole` DCE barrier (:19); `ContinuousClock` wall timing (:290); p50/p95/p99; GPU timing read from `context.lastGPUTiming`; `submissionOverhead = wall p50 − GPU p50`; index harness has `SeededRandomNumberGenerator` (seed 42), brute-force ground truth, `LatencyStats` with interpolated percentiles, workloads enum | Ships inside the **product** (public API); `generateRandomVector` unseeded (:317); index harness uses `CACurrentMediaTime` (:304); `indexJsonReport` (:717) has zero callers; some paths hardcode `recall: 1.0` (:449, :548); no baseline, no comparison, no thermal tag, no adaptive warm-up. Predates the entire hardening epic (slices 1–34). |
| (b) Benchmark executable | `Sources/VectorAccelerateBenchmarks/` (5 files, 1,548 L) | `--crossover` sweep (dims × batch sizes, `ContinuousClock`, JSON out) meant to calibrate `GPUDecisionEngine` | Output never fed back into thresholds (AUDIT.md:118 still says crossovers "are unknown"); `KernelUsageExamples.swift` (557 L) unreachable from `main.swift`; `SwiftTopicsBenchmarkRunner` uses `CFAbsoluteTimeGetCurrent`; JSON written to CWD, neither tracked nor gitignored as an artifact; nothing runs it. |
| (c) Paper design | `docs/ADD/metal4/MASTER_PLAN.md:514-597` | Complete design: metric table with thresholds (pipeline cache hits >95%, argument-table reuse >80%, residency commits <10/batch), `PerformanceRecorder` actor, per-hardware baselines, a `performance-regression` CI job | **Zero of it exists.** No `PerformanceRecorder`, no `Benchmarks/baselines/`, no `scripts/check_regression.py`, no CI job. |
| (d) Test-suite perf assertions | `PerformanceBenchmarks.swift`, `MLIntegrationBenchmarkTests.swift` (1,551 L), `SwiftTopicsBenchmarks.swift`, `IndexBenchmark*Tests.swift` | The two `IndexBenchmark*` suites assert real recall (flat = 1.0, IVF > 0.9) and are valuable | All wall-clock and throughput floors are behind `PerfGate.strict` (`VECTORACCELERATE_STRICT_PERF=1`, off by default); GPU-stress suites skip under `CI`; `PerformanceBenchmarks.swift` prints and asserts nothing, no warm-up, unseeded. In practice **perf is never asserted anywhere automatically.** |

AUDIT.md §E (2026-04-05) rated the benchmarking strategy "Significant" and listed seven fixes; CHANGELOG 0.4.2 records fixes 1–4 done (ContinuousClock, p95/p99, `blackHole`, `GPUTimingInfo`). Fixes 6 (thermal awareness) and 7 (adaptive warm-up until CoV < 5%) were never done. `ThermalStateMonitor` exists and is unused by any benchmark.

### 2.3 Instrumentation that exists

- **GPU timing:** `Metal4Context.GPUTimingInfo` (:49–66) from `commandBuffer.gpuStartTime/gpuEndTime`, captured in `executeAndWait` (:436) and `executeBlitAndWait` (:512). Stored in a single slot `_lastGPUTiming`, overwritten per dispatch, unsafe under concurrent submissions. Command-buffer granularity only. **No `MTLCounterSampleBuffer`, no `MTL4CounterHeap`, no encoder-level timestamps anywhere.**
- **Profiling counters (opt-in):** `Metal4Configuration.enableProfiling` gates `totalComputeTime`/`computeOperationCount` → `Metal4Context.PerformanceStats` (:680). Mirrored on `Metal4ComputeEngineConfiguration`, `ClusteringConfiguration`, `IVFConfiguration` (→ `IVFPhaseTimings`), `KMeansPipeline` per-iteration timings.
- **Routing provenance (always on):** `MetalComputeProvider.RoutingTelemetry` (:49–63) — counters for `gpuKernel`, `cpuDecisionEngine`, `cpuPolicy`, `cpuFallbackAfterGPUError`, `cpuFallbackEmptyGPUResult`, `lastGPUErrorDescription`. Guarded by `Hardening/RoutingProvenanceTests.swift`. Rule established in the 2026-08-16 hardening plan: *"Telemetry observes; it must not alter which path runs."* Nothing routed via `Metal4ComputeEngine`, the index, or `BatchProcessor` is covered.
- **Audit trace:** `VA_AUDIT_TRACE=1` prints `[VA_AUDIT] gpu-submit` on every submit (`Metal4Context.swift:305-308`). The only env-driven runtime trace.
- **Pull-model statistics snapshots (13 structs):** `BufferPool.PoolStatistics`, `ResidencyStatistics`, `WarmupStatistics`, `PipelineCacheStatistics`, `ArchivePipelineCacheStatistics`, `ArgumentTablePoolStatistics`, `Metal4CompilationStatistics`, `TensorManagerStatistics`, `WALStatistics`, `GPUIndexStats`, `GPUPerformanceStats`, `AccelerationStatistics`, `LoggerPerformanceStats`. None push events; none carry timestamps.
- **Logging:** `Core/Logger.swift` — one os_log subsystem `com.vectoraccelerate`, one category, message interpolated into a single `%{public}@` (no structured fields). Consumers: `MatrixEngine`, `BatchProcessor`, `MemoryMapManager`, `SIMDFallback`, `QuantizationEngine`. **Zero kernels, zero `Metal4Context`, zero `Index/`.**
- **Absent entirely:** `os_signpost`, `OSSignposter`, `MetricKit`, Instruments custom packages.
- **Orphans:** `Configuration/PerformanceMonitor.swift` (210 L, zero callers, stubbed utilization getters, last touched 2025-09-14); `IndexAccelerationConfiguration.logDecisions` (declared, stored, never read).

### 2.4 Guardrails that exist

- **Enforcement model is thrown errors, not assertions:** 682 `throw` sites vs 12 `precondition`, 2 `assert`, 1 `assertionFailure`, 1 `fatalError`. Error surface `AccelerationError` (18 cases incl. `bufferPoolExhausted`, `invalidBufferSize`, `memoryPressure`, `dimensionMismatch`). No debug-only invariant mode exists.
- **Buffer pool accounting** (`Core/BufferPool.swift`): 64 MiB cap; throws on negative/oversize/budget/bucket/allocation failure; `PoolStatistics` exposes hit/miss/alloc counts, `currentMemoryUsage`, `maxMemoryLimit`, total/available buffers. **No high-water mark, no explicit live-lease count** (only inferable as `total − available`). Three hardening suites guard lifecycle and accounting.
- **Decision engine** (`Core/GPUDecisionEngine.swift`): `shouldUseGPU(...) -> Bool` (:333) — no reason code, no decision record. Adaptive `gpuPerformanceRatio` mutated globally across operations (:527–530).
- **Health and thermal:** `GPUHealthMonitor` (degradation levels, per-op failure counts, `shouldFallback`), `ThermalStateMonitor` (`shouldThrottle`, observers). Neither is consulted by any benchmark.
- **Differential oracle pattern** (`Hardening/DifferentialKernelVsCPUTests.swift`): ten adversarial value classes (zeros, duplicates, 1e19, 1e-20, subnormal 1e-40, mixed scale, NaN/±Inf poisoned); independent Double-accumulation oracles; three-tier comparator (`both NaN ⇒ equal`, `bitwise equal ⇒ equal`, else `|a−b| ≤ max(absTol, relTol·max(|a|,|b|))`, defaults `relTol = 2e-4`, `absTol = 1e-5`); class mismatch is a hard fail. Sweeps D ∈ {1,3,7,16,33,128,384,768} × N ∈ {1,2,33,257,1000}. Top-K tie order deliberately not pinned (AUDIT-2 VA2-007). **No ULP comparator anywhere; tolerances are per-file constants** (`2e-4/1e-5`, `1e-4` at `MLIntegrationBenchmarkTests:954`, bare `Float.ulpOfOne` at `NumericalStabilityTests:33`).
- **Validation layers:** `docs/stability/FUSED-TOPK-INSTRUMENTATION-CONTRACT.md` (uncommitted, slice 34) records the `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` run mode and real numbers (fused_l2_topk static threadgroup 22,528 B → 6,160 B; instrumented 45,056 B → 12,308 B). Open under validation: `ivf_list_search` at 55,296 B vs 32,768 B limit; `includeDistances: false` fused dispatch SIGABRTs on unbound buffer(3); unary elementwise `input_b` binding. **No repo-resident way to run this mode; the paired benchmark evidence is in `/private/tmp/va-fused-instrumentation/2026-09-14/`.**
- **Determinism policy is explicitly unresolved** (HARDENING-HANDOFF §6 Group E / VA3-019): relaxed `atomic_float` accumulation order in UMAP target gradients, PQ training, k-means update ⇒ run-to-run FP variation with no accept-and-bound decision.

### 2.5 Reproducibility primitives

- Canonical test RNG: `Tests/.../Utilities/TestRNG.swift` — 64-bit LCG, constructor warm-up discard, stream mixing, Box-Muller, top-24-bit float. Canonical generator: `TestDataGenerator.swift` (uniform, gaussian clusters, separated clusters, unit, sparse, skewed; perturbed/cluster/random queries; `DatasetStatistics`).
- Adoption is partial: `TestDataGenerator` used by 3 test files; `TestRNG` by ~10. **Three other seeded RNGs coexist** (`SeededRandomNumberGenerator` xorshift64 in shipped `IndexBenchmarkHarness.swift:665`; two private `SeededRNG` in tests). **Nine unseeded `Float.random` sites in shipped Sources** (`BenchmarkFramework:318`, `Core/Types:249`, `TensorManager:396`, `KMeansPlusPlusKernel:131`, `ProductQuantizationKernel:587`, `QuantizationEngine:593`, `UMAPGradientKernel:679`, `WarpOptimizedSelectionKernel:796/838`, `QuantizationStatisticsKernel:547`); none accept an injected generator.
- Metal-availability skipping: 149+ `XCTSkip` sites in four message spellings, each re-creating a device; no shared `requireMetalDevice()`.

### 2.6 Delivery infrastructure

- `.github/workflows/ci.yml`: six jobs on GitHub-hosted `macos-26` (virtualized GPU, Xcode 26.0.1): `metal-shaders`, `build`, `test` (retries once; comment documents nondeterministic command-buffer hangs and pipeline-compile errors on the virtual GPU), `build-ios`, `build-release`, `lint`. **No perf job. No release-config test job** (deferred by owner, AUDIT-2 decision 5).
- `Scripts/`: one file, `compile_metallib.sh` (note: `-std=metal3.2` vs CI's `metal4.0`).
- SwiftLint excludes `Sources/VectorAccelerateBenchmarks` but not the library-side `Benchmarking/`.

### 2.7 Performance claims currently on record

- README.md:582–586 — five CPU-vs-GPU speedups (82×, 97×, 60×, 45×, 52×) with **no hardware, OS, dataset, date, or reproduction script**. Also :5 "up to 100x", :274 "~0.3ms search", :298–299 insert/search rates, :574 "2x throughput gain".
- `docs/ADD/` targets (all estimates unless noted): IVF 100q@5K target <0.05 s and >2000 q/s; ">1M candidates/s at D=768 on M1/M2"; neural encode 10–50× (bandwidth-model 104× ceiling); neural decode 4–6×; per-kernel millisecond tables for mutual-reachability, Borůvka, UMAP, batch-max, logsumexp, sparse TF-IDF; `QUALITY_IMPROVEMENT_ROADMAP.md:1461-1502` is the only block presenting itself as measured (recall vs FAISS by nprobe; IVF slower than flat below N≈10K; batch 9.53×). HARDENING-HANDOFF records "UMAP GPU benchmark underperforms expectation (0.6–1.6× vs 2–5×)" with no follow-up.
- Fixed hardware constants assumed in docs and code: 32 KB threadgroup memory, 16 KB pages, ~200 GB/s UMA, 32-wide SIMD groups. Only one machine has ever been used: M3 Max. Multi-chip coverage (M1 Pro/M2 Max/M4/A17) exists only as aspiration in docs.

### 2.8 Prior art in the sibling repositories (the suite context)

- **VectorCore** (`/Users/goftin/dev/gsuite/VSK/VectorCore`): `VectorCoreBenchmarking` library target with a real schema — `BenchRun { metadata: BenchMetadata, results: [BenchCase], abComparisons }`; `BenchMetadata` carries package/version/gitSHA/date/os/arch/cpuCores/deviceModel/swiftVersion/buildConfiguration/deviceTag/binarySizes/suites/dims/runSeed/runLabel/flags/thresholds/filters; `BenchCase` carries iterations, ns/op, ns/unit, throughput, gflops, `suspicious`, mean/median/p90/stddev/RSD, `correctness {samples, maxAbs, meanAbs, maxRel, meanRel}`. `EnvCapture` reads sysctl/brand string/hostname; device tag `arch-os-host` or `VC_DEVICE_TAG`. File layout convention `.bench/runs/<device-tag>/<timestamp>_<sha>.json`. Case-ID grammar (`batch.<metric>.<dim>.N<count>.<variant>[.<mode>]`) with parse/round-trip. `benchmarks.yml`: `workflow_dispatch` on `macos-15`, records env, runs `--profile full --samples 5 --run-seed 1 --format json`, uploads `.bench/` as an artifact (30 days). Correctness thresholds documented: FP32 ≤1e-6 rel, FP16 ≤2e-3, FP16 cosine ≤3e-3. **No GPU block in the schema.** The 2026-era `Perf_Benchmarking.md` is marked historical, but its roadmap explicitly anticipated a cross-package aggregator (`vector-bench-runner --suite core|index|accelerated|all`) and a unified schema.
- **VectorIndex** (`/Users/goftin/dev/gsuite/VSK/VectorIndex`): single-file benchmark CLI; `.bench/baseline-<version>/` directories as comparison points; a written **gate rule** worth copying: Release config, same machine, quiet machine (probe: no-op `swift build` must finish <10 s, else wait), recall bit-identical for mechanical changes and within ±0.01 with a one-line explanation for accumulation-order changes, byte-identical graph determinism test at every task boundary. `VECTORINDEX_RELIABILITY_ASSESSMENT_2026-08-15.md` Tier 3 records CI filter groups silently running the full suite and a "skip audit" recommendation ("is anything else silently untested?") — directly applicable here given `PerfGate` and 149 Metal skips.
- **VectorBench** (`/Users/goftin/dev/gsuite/VSK/VectorBench`): a macOS SwiftUI app (SwiftData persistence, device profiling with thermal monitoring, Swift Charts, git-integrated baselines) that consumes VectorCore's JSON. Stalled at UI polish; "multi-package support (VectorIndex, VectorAccelerated)" listed as coming soon. Relevant because any new schema decision determines whether this app is a viewer for free or dead.

---

## 3. What mature projects and the literature do

### 3.1 Measurement statistics

**Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013)** is the reference. Its content that matters here:

- Variation is *hierarchical*: process launch, then steady-state phase, then iteration. Repeating only at the innermost level (the universal mistake) measures nothing about the outer levels. For this library the levels are (1) process + metallib load + PSO creation + pool warm-up, (2) command-buffer submission, (3) kernel execution.
- Run a one-time *variance-profiling* experiment to measure variance at each level; derive the repetition count per level from that; then fix those counts in configuration. Do not guess "50 iterations, 5 warm-up".
- Detect steady state before measuring (their approach: manual inspection or a change-point on the warm-up series). AUDIT.md's unimplemented fix 7 (adaptive warm-up until CoV of a sliding window < 5%) is the pragmatic form; bound it with a maximum and **record the warm-up count used** in the result.
- Report effect size with a **bootstrap confidence interval** on the ratio, not a point estimate. A "3% regression" without an interval is noise.

Practical consequence: the harness needs a `samples × iterations` two-level loop at minimum, with per-sample restart of the outer level for anything where pipeline or pool state matters.

### 3.2 Regression detection

Two families:

- **Threshold against a baseline.** Bencher's seven models are a good taxonomy: static, percentage, z-score (needs ≥30 history points, independent runs), t-test (prediction interval, small N), log-normal (positive-only data with outliers), IQR and delta-IQR (robust, adapts to trending data). Percentage-vs-baseline with a boundary of `max(5%, 2 × baseline RSD)` is the honest starting point for a series with fewer than ten points.
- **Change-point detection on history.** E-Divisive means (MongoDB's CI work → Hunter, Apache Otava, Nyrkiö) finds shifts in short series (4–7 points) where Mann-Whitney needs ~30, tolerates noise, and reports the change point rather than per-run alerts. Graduate to this once ≥20 historical runs exist per case.

Two rules from practice: **alert on improvements too** (a sudden 10× "win" is usually a broken benchmark or a dead-code-eliminated loop), and compare only within an identical environment key (§4 P4).

### 3.3 Apple Silicon environment control

- macOS offers **no core pinning**. QoS is the only lever: `userInteractive`/`userInitiated` biases scheduling to P-cores; background QoS lands on E-cores and is throttled first under thermal pressure. Benchmarks must run at high QoS and record it.
- `ProcessInfo.thermalState` has four states but the kernel has five pressure levels; "fair" masks both moderate and heavy pressure. It is a useful *reject* signal (anything above `.nominal` invalidates a run) but not a *measure*. `powermetrics` gives real thermal pressure and CPU/GPU frequency but needs `sudo`.
- Record per run: power source (AC/battery), Low Power Mode, thermal state at start and end, memory pressure, free memory, and a **quiet-machine probe**. VectorIndex's "no-op `swift build` < 10 s" is a coarse probe; a better one is a calibration micro-kernel (STREAM-like GPU copy and CPU copy) run at the start *and end* of every run — drift between the two above a few percent flags the run as thermally unstable, and the absolute number is the roofline calibration VectorCore's roadmap wanted.
- Measured noise on M-series with this discipline is small: per-trial noise ≤0.4% typical, ≤1.3% worst, in a published LLM benchmark; cross-machine variance dwarfs it. Baselines therefore must be **per device**, and a MacBook Pro M3 Max *will* throttle under sustained GPU load — duty-cycle long suites and tag runs.

### 3.4 GPU timing on Metal 4 — a hard constraint discovered in this research

- `gpuStartTime/gpuEndTime` (what `Metal4Context` uses today) is **whole-command-buffer** granularity. It cannot separate kernels in one buffer and cannot exclude blits.
- **Legacy `MTLCounterSampleBuffer` timestamps: partially usable, verified locally.** wgpu issue #9414 reports all-zero timestamps on macOS 26. The spike (Appendix D) refutes that for stage-boundary sampling on this M3 Max: `.atStageBoundary` is supported and returns non-zero nanosecond timestamps that match `gpuStartTime/gpuEndTime`; `.atDispatchBoundary`, `.atBlitBoundary`, and `.atDrawBoundary` are unsupported, so a client that assumes dispatch-boundary slots reads zeros (the likely cause of the report). One anomaly: blit-resolving a `.private` sample buffer returned one-run-stale values from the second run on. Net: the legacy path can time an encoder but not a dispatch, and `docs/ADD/metal4/MASTER_PLAN.md`'s "GPU exec μs via `MTLCounterSampleBuffer`" is at best encoder-level.
- **The Metal 4 counter-heap path works, verified locally (Appendix D).** `MTL4CounterHeap(.timestamp)` with `MTL4ComputeCommandEncoder.writeTimestamp(granularity:counterHeap:index:)` before and after a dispatch gave non-zero, monotonic stamps in 600 of 600 pairs at both granularities; CPU `resolveCounterRange(_:)` returns packed `UInt64` ticks; `queryTimestampFrequency()` is 24 MHz (41.7 ns per tick). Encoder-level kernel time for a 128 MiB copy was 0.361 ms at 2.8–3.4% RSD, stable to 0.4% across runs and granularities; `.precise` bought nothing measurable with one dispatch per encoder; any heap stamp in a command buffer adds 7–13 µs to the command-buffer span but nothing to the encoder span; a command-buffer-level stamp must never be subtracted from an encoder-level stamp, because their ordering is not guaranteed (6–14% of samples inverted).
- **Hard precondition discovered:** `Metal4Context` submits on a legacy `MTLCommandQueue`/`MTLCommandBuffer` (`Core/Metal4Context.swift:128`, `:233`; the comments there say Metal 4 "would" use `MTL4CommandQueueDescriptor`), and `MTL4CommandBuffer` has no `gpuStartTime/gpuEndTime` (its equivalent is the asynchronous `MTL4CommitFeedback`). Counter heaps cannot be written from a legacy encoder. So encoder-level kernel timing inside the library requires migrating the submit path to `MTL4CommandBuffer`/`MTL4ComputeCommandEncoder` first. Until then the harness isolates one kernel per command buffer and reports command-buffer GPU time with a measured bias of about +2 µs (no stamps) to +10 µs (with stamps) over the kernel span, which dominates for the library's small kernels.
- Three distinct numbers must be kept apart in the schema: **wall** (ContinuousClock around encode+submit+wait — what a caller experiences; the right number for CPU/GPU crossover), **command-buffer GPU time**, and **kernel time** (counter heap). `submissionOverhead = wall − commandBuffer` already exists conceptually in `BenchmarkFramework`.

### 3.5 Vector-search benchmark methodology (ann-benchmarks, big-ann-benchmarks, VectorDBBench)

- Core metrics: **recall@k** averaged over a query set, **QPS**, per-query latency **p50/p95/p99**, index **build time**, **memory**. The headline artifact is the **recall–QPS curve** per index configuration, and comparisons are "QPS at recall ≥ X", never raw QPS.
- **Recall definition matters for GPU float distances.** ID-set intersection punishes tie flips caused by accumulation order (exactly the VA3-019 issue). ann-benchmarks counts a returned neighbor as correct if its distance is ≤ the ground-truth k-th distance × (1 + ε). Use the ε form; record ε.
- Ground truth is a stored artifact (query IDs, k neighbor IDs, k distances), computed once by an exact Double-precision CPU oracle and keyed by dataset seed — never recomputed per run.
- Datasets: synthetic distributions are fine for regression tracking (TestDataGenerator already covers uniform/clustered/sparse/skewed) but claims about "real workloads" need at least one real embedding set (SIFT1M, GloVe, or a 384/768-D sentence-embedding set) with a checked-in manifest and hash.
- big-ann-benchmarks' streaming track uses **runbooks** (scripted insert/delete/search sequences) to score indices under mutation; VA's `BenchmarkWorkload.mixedInsertSearch` is the seed of this.
- Recent critiques (2025–26) of vector-DB benchmarks: hidden parameter sensitivity, single-threaded vs batched conflation, and filtered search are the usual blind spots. Report parameters exhaustively in the result; distinguish single-query latency from batch throughput as separate cases.

### 3.6 Tooling available in the Swift ecosystem

- **ordo-one/package-benchmark** (v1.35, June 2026): percentile-oriented output, per-benchmark absolute and relative thresholds, PR regression checks vs a stored baseline, custom metrics, malloc/ARC/syscall counters (jemalloc dependency dropped). Strong for CPU-side and allocation metrics; **no notion of GPU time** — it would need custom metrics fed from the counter heap. Worth adopting for CPU kernels and allocation regression; not sufficient alone.
- **XCTest `measure(metrics:)`** with `XCTClockMetric`/`XCTMemoryMetric`/`XCTCPUMetric` and baselines works only under Xcode/xcodebuild with `.xcresult`, not `swift test`. Not a fit for an SPM-first repo.
- **Continuous-benchmarking services:** Bencher (self-hosted allowed, seven threshold models), github-action-benchmark (simplest; publishes to gh-pages; alert on N% regression), Nyrkiö (E-Divisive as a GitHub Action; self-hosted runners currently org-only), CodSpeed (their own runners, no macOS GPU). All consume a JSON of `{name, value, unit}`; none measure Metal. Any of them can sit on top of the schema; none replace it.
- **Fuzzing:** `-sanitize=fuzzer` is not available on the stock Apple Silicon toolchain; FuzzCheck is unmaintained. Property-style coverage is better obtained from seeded parameter sweeps with the existing `TestRNG` (the `DifferentialKernelVsCPUTests` shape), which is already the house style.

### 3.7 Guardrail techniques

- **Metal validation layers in automation:** `MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`, `MTL_DEBUG_LAYER_ERROR_MODE=assert|nslog|ignore`, `MTL_DEBUG_LAYER_WARNING_MODE=nslog`; must be set **before the first device is created in the process**, so they belong in a wrapper script or CI env, not in test code. `man MetalValidation` is the reference. Shader validation also changes static threadgroup footprints (the slice-34 finding), so a validation run is a distinct gate, not a variant of the normal run.
- **Differential tolerance.** Mixed abs/rel is standard; ULP bounds are appropriate for kernels whose error is analyzable. For a Float32 sum of D terms the sequential bound is ≈ (D−1)·ε·Σ|xᵢ| with ε = 2⁻²⁴; a tree reduction (SIMD-group sums) has depth ⌈log₂D⌉ and a bound ≈ ⌈log₂D⌉·ε·Σ|xᵢ|. Tolerances should be **derived from D and the reduction shape**, not per-file constants — this is what turns "2e-4 because it passes" into a contract.
- **Soak testing:** long-running mixed workloads at fixed duty cycle; the signal is a **monotonic slope**, not a level — resident memory, pool `currentMemoryUsage`, live leases, pipeline-cache size, file descriptors, all sampled on a fixed period; report MB/hour and projected time-to-exhaustion. The pool's missing high-water mark and live-lease count are the two counters this needs.
- **Roofline as a guardrail:** for memory-bound kernels (all distance kernels), express achieved bandwidth as a fraction of the calibrated STREAM number from the preflight probe; a kernel dropping below its recorded fraction is a regression *independent of absolute time*, which makes the gate portable across devices. Published M-series figures (M1 68 GB/s … M3 Max ~400 GB/s, M4 Max 546 GB/s) are ceilings; the probe gives the real one.
- **Skip audit:** every `XCTSkip`, `PerfGate`, and env gate should be enumerated once and classified (environment gate vs unimplemented vs disabled-on-purpose), following VectorIndex's Tier-3 lesson. The 2 remaining skips are known; the 149 Metal skips are not classified.

### 3.8 Observability for an embedded library

- **os_signpost intervals** (`OSSignposter`, or `os_signpost` when a custom Instruments package needs the older API) with one subsystem and **categories per concern** (`routing`, `pool`, `pipeline`, `kernel`, `index`) make Instruments' Points of Interest and a custom instrument work for free. Intervals should bracket submit→scheduled→completed per command buffer with a submission ID, and each kernel dispatch.
- **Structured fields**, not interpolated strings: `%{public}s` per field so `log stream --predicate` can filter by kernel name, N, D, k, path.
- **Push vs pull:** the existing 13 snapshot structs are pull-model and fine for tests. Runtime consumers (an app deciding whether to trust GPU routing, or a dashboard) need push: a `RoutingDecision` record (`operation, n, d, k, chosen, reason, thresholdsSnapshot, thermalState, poolPressure`) delivered through a delegate or `AsyncStream`, mirrored to a signpost. OpenTelemetry-swift's `OSSignposterIntegration` shows the shape of bridging spans to signposts.
- **Zero cost when off:** a single `Instrumentation` configuration read once into a static flag; every hook is `@inlinable` with an early `guard` so release builds without instrumentation compile to nothing on the hot path. The 2026-08-16 rule stands: telemetry must never alter routing.

---

## 4. Principles that follow from the research

- **P1 — No number without provenance.** Every result carries the environment key (§4 P4), git SHA + dirty flag, seed, warm-up count actually used, sample/iteration counts, thermal state at start and end, and the preflight probe values. A result without these is discarded by the comparison tool, not merely warned about.
- **P2 — Three clocks, used on purpose.** Wall = `ContinuousClock`; command-buffer GPU = `gpuStartTime/gpuEndTime`; kernel = `MTL4CounterHeap`. `CFAbsoluteTimeGetCurrent` and `CACurrentMediaTime` are removed from all measurement code.
- **P3 — One RNG, one generator, one comparator, shared by tests and harness.** Promote `TestRNG`/`TestDataGenerator` and a tolerance comparator (abs/rel/ULP/class) into a target both tests and the harness import; retire the other three RNGs; give the nine unseeded shipped sites an injectable generator.
- **P4 — Baselines are keyed, comparisons are within-key.** Key = (chip model + brand string, GPU core count, OS build, Metal toolchain version, Swift version, build configuration, validation-layer on/off). Cross-key comparisons produce curves and tables, never pass/fail.
- **P5 — Correctness gates are exact or tolerance-typed; performance gates are statistical.** Never a bare "must be faster than 1 ms".
- **P6 — Observability is opt-in, zero-cost when off, and never changes routing.**
- **P7 — The harness is not product API.** `Sources/VectorAccelerate/Benchmarking/` moves out of the shipped library (breaking change, to be noted for 0.7.0); the harness becomes an executable plus a non-product library target.
- **P8 — CI runners produce correctness and schema validation, not trusted numbers.** The virtualized `macos-26` GPU is documented as nondeterministic. Trusted numbers come from a physical machine (the M3 Max, later a self-hosted runner) on a manual or scheduled trigger.
- **P9 — Claims are regenerated from artifacts or retired.** README:582–586 either gets a reproduction command and a date/hardware line or is replaced by a pointer to the latest run.
- **P10 — Close the loop before widening the matrix.** The first milestone is one kernel, one metric, one baseline, one comparison that can fail. Every prior attempt widened first.

---

## 5. Decomposition into sub-projects

The program is too large for one spec. Each sub-project below is one brainstorm → spec → plan → implementation cycle with its own milestone that stands alone if the program pauses. Dependencies are explicit; SP2 and SP3 can run in parallel after SP1.

### SP0 — Foundation (substrate)
**Delivers:** environment fingerprint + preflight probe; result schema v1 (a superset of VectorCore's `BenchRun`, adding `schemaVersion`, a `gpu` block, thermal/power fields, warm-up-used, and the preflight numbers); `.bench/runs/<device-key>/<timestamp>_<sha>.json` layout; a `VectorAccelerateTestKit` (name provisional) library target holding `TestRNG`, `TestDataGenerator`, the comparator, oracles, and `requireMetalDevice()`; the harness executable skeleton (`--list`, `--filter`, `--mode smoke|quick|full`, `--format json`, `--out`); deletion of the orphans (`PerformanceMonitor`, `KernelUsageExamples`, `logDecisions`, the unused `indexJsonReport`) and of the duplicate RNGs.
**Milestone M0:** one command produces a schema-valid JSON with full provenance for one existing kernel benchmark; a schema test fails on a missing provenance field; a "no orphan public type" mechanical test exists.
**Blocks:** everything.

### SP1 — Timing substrate
**Delivers:** the MTL4CounterHeap spike result and, if positive, encoder-level kernel timing inside `Metal4Context` behind the instrumentation flag; per-submission timing records (ring buffer or returned with the result) replacing the single slot; two-level `samples × iterations` loop with adaptive warm-up (bounded, recorded); bootstrap CI on medians; variance-profiling mode that reports per-level variance so repetition counts can be set from data.
**Milestone M1:** a known-cost kernel (a fixed-size copy) reports kernel time within a few percent of its bandwidth-derived expectation, wall/command-buffer/kernel are reported separately, and the CI half-width is printed.
**Depends on:** SP0.

### SP2 — Correctness and resource guardrails
**Delivers:** the shared comparator with D-derived tolerances; recall@k with distance-ε; a repo-resident validation-layer run mode (`Scripts/validate.sh` + a CI job) that is red until `ivf_list_search`, the unbound buffer(3) path, and `input_b` are fixed; buffer-pool high-water mark and live-lease counters with a lifecycle test; a stated **determinism policy for VA3-019** (§6 Q4) encoded as a test with an empirically derived bound; a soak mode in the harness with slope reporting; the skip audit (classify every `XCTSkip`/`PerfGate`/env gate).
**Milestone M2:** `Scripts/validate.sh` green on the suite; determinism policy documented in `docs/stability/` with its guard suite; soak mode runs 30 minutes with a flat slope.
**Depends on:** SP0 (SP1 only for the soak timings).

### SP3 — Observability
**Delivers:** `RoutingDecision` records with reason codes from `GPUDecisionEngine` and every routing site (not only `MetalComputeProvider`); os_signpost intervals for submit/scheduled/completed and per-kernel dispatch; structured os_log categories; a unified `Instrumentation` configuration that compiles to nothing when off; an "explain" API that returns the last N decisions.
**Milestone M3:** an Instruments trace of the index search shows nested intervals per kernel; `RoutingProvenanceTests` extended to the engine and index paths; a test proves the off-path adds no allocation (package-benchmark malloc counter is a good fit here).
**Depends on:** SP1 (timing records) and SP0.

### SP4 — Benchmark matrix, baselines, and regression rule
**Delivers:** the case grammar (extend VectorCore's) and the matrix: kernel micro (distance, top-K fused/warp/standard, normalize, quantize), index macro (flat/IVF recall–QPS curves at fixed nprobe sweep, build time, memory), crossover sweep feeding `GPUDecisionEngine` thresholds, streaming runbook; ground-truth artifacts; baseline capture (`bench baseline capture`) and comparison (`bench compare --against`) with the percentage-then-change-point rule; markdown report; the two IVF placeholders implemented as scaling-shape and RSD gates; VA3-024 measured before/after; README claims regenerated or retired.
**Milestone M4:** a deliberate 20% kernel slowdown on a branch is caught by `bench compare` on the M3 Max; the README table is replaced by generated content with hardware and date.
**Depends on:** SP1, SP2 (comparator, recall-ε).

### SP5 — Automation and reporting
**Delivers:** `workflow_dispatch` (later scheduled) trusted-runner job, or a local `Scripts/bench.sh` that does preflight → run → compare → store when no self-hosted runner exists; CI smoke job that runs `--mode smoke` on the virtual runner and validates the schema only; artifact retention and a history directory; optional viewers (revive VectorBench on the shared schema, or a static HTML report).
**Milestone M5:** every merge to `main` has a run record within a day; a regression opens a visible alert.
**Depends on:** SP4.

**Suggested order:** SP0 → SP1 → (SP2 ∥ SP3) → SP4 → SP5. SP0 is small (days, not weeks). SP1 contains the only real unknown (the counter heap) and should start with the spike.

---

## 6. Decisions the owner must make before the RFC

Each is listed with a recommendation; the RFC will be written against the answers.

**Q1 — Scope: VectorAccelerate-only harness, or a VSK-suite harness?**
Recommendation: implement in VectorAccelerate first, but make the schema a strict superset of VectorCore's `BenchRun` with `schemaVersion` and a `package` field, so VectorBench and a future aggregator consume all three packages. Do not build the aggregator now.

**Q2 — Where the harness lives.**
Recommendation: two pieces. (a) A thin instrumentation API inside the library (signposts, decision records, timing records, counters) — this must ship. (b) A `VectorAccelerateTestKit` library target (shared by tests and the harness) plus a `VectorAccelerateBench` executable — neither is a product. Remove `Sources/VectorAccelerate/Benchmarking/` from the product in 0.7.0 with a CHANGELOG entry. Alternative: a separate SPM package; rejected for now because the harness needs `@testable` access to internal statistics.

**Q3 — Trusted-runner policy.**
Recommendation: CI keeps correctness, adds the validation-layer job and a schema-only smoke job. Trusted numbers come from the M3 Max via `Scripts/bench.sh` with preflight enforcement, results stored under `.bench/runs/` and committed to a `bench-history` branch (or kept as artifacts). Add a self-hosted runner only if a second, dedicated Mac exists; do not run perf on the virtual GPU.

**Q4 — Determinism policy for VA3-019 (atomic-float accumulation in UMAP, PQ training, k-means).**
Options: (i) accept and bound — measure run-to-run variation per kernel on the M3 Max, publish the bound in a stability contract, gate on it; (ii) rework to deterministic reductions (segmented or two-pass) at a measurable cost; (iii) both, behind a `deterministicReductions` flag. Recommendation: (i) now, (iii) later only if a consumer needs bit-reproducible training. Whichever is chosen, recall gates use the distance-ε form so tie flips are not false regressions.

**Q5 — Regression decision rule.**
Recommendation: baseline = median of ≥5 runs' medians on one environment key; alert when the new median is outside `max(5%, 2 × baseline RSD)` in either direction; switch a case to E-Divisive once it has ≥20 history points; improvements alert too.

**Q6 — README claims.**
Recommendation: retire lines 582–586 immediately (replace with "see the latest run under `.bench/runs/`") rather than leaving unreproducible numbers in place until SP4; regenerate with hardware and date once M4 lands.

**Q7 — Dead code.**
Recommendation: delete `PerformanceMonitor.swift`, `KernelUsageExamples.swift`, `logDecisions`, the unused `indexJsonReport`, and the three duplicate RNGs in SP0, with a mechanical "no unreferenced public type" test so the class is closed, not the instance.

**Q8 — Hardware claims.**
Recommendation: claim only what has a run record (M3 Max today). The schema and baseline layout support many devices; multi-device coverage is a future acquisition or contribution problem, not a design problem.

**Q9 — Adopt ordo-one/package-benchmark for CPU-side metrics?**
Recommendation: yes, for CPU kernels and allocation/ARC regression (it is the only tool here with malloc counters), fed into the same schema; not as the primary GPU harness.

### 6a. Decisions recorded (owner, 2026-09-14)

| Q | Decision |
|---|---|
| Q1 | VectorAccelerate-only now; the kit is generic by discipline (no VectorAccelerate/VectorCore/Metal/XCTest imports, enforced by an import-lint test; schema field-compatible with VectorCore's `BenchRun`) so it can be lifted to VectorCore later with "careful disciplined consideration". |
| Q2 | Option A: non-product targets inside the package (`VectorTestKit` library, `VectorAccelerateBench` executable). Two written triggers: the first bench-only external dependency moves the harness to a `Benchmarks/` sub-package; VectorCore adopting the kit moves the kit out. No app. |
| Q3 | M3 Max is the only trusted runner for now. An M4 Pro and an M2 Pro exist and become additional environment keys later. |
| Q4 | Determinism (VA3-019): accept and bound now. |
| Q5 | Speed regressions alert but do not block. Per-case boundary `max(2.5%, 2 × baseline RSD)`, widening shown in the report; improvements alert too. |
| Q6 | README claims retired now (table at 582–586 and "up to 100x" at line 5). |
| Q7 | Delete dead code aggressively, with a mechanical orphan test. |
| Q8 | Claim only hardware with a run record (M3 Max). |
| Q9 | ordo-one/package-benchmark deferred to SP3's first allocation guardrail; CPU and allocation metrics only. |

---

## 7. Risks and unknowns

- **The counter-heap API is verified, but the library's submit path is legacy.** Moving `Metal4Context` to `MTL4CommandBuffer` is a hot-path change to the core of the library and is SP1's central design decision; the interim is one-kernel-per-command-buffer isolation with the measured bias above.
- **Virtualized CI GPU** constrains gating; already documented in `ci.yml`. Do not fight it.
- **Laptop thermals.** Sustained GPU suites on a MacBook Pro will throttle. Duty-cycle, tag, and reject runs whose start/end probe drift exceeds the bound. Expect the full matrix to need staging across sessions.
- **Validation-layer gate starts red.** Three known failures under `MTL_SHADER_VALIDATION=1`. Land the gate as a non-blocking job first; flip to blocking when green.
- **Breaking change** in removing `Benchmarking/` from the product. Consumers are unknown; the version-history header in `Package.swift` and CHANGELOG discipline cover it.
- **Solo-developer cost.** Each sub-project must leave the repo better if the program pauses after it; the milestones in §5 are chosen so that they do.
- **Toolchain drift in docs** (Swift 6.0 / 6.2 / 6.3.3 stated in different places). The fingerprint records the truth; the docs get one reconciliation pass in SP0.

---

## 8. Recommended next steps

1. Owner answers Q1–Q9 (or accepts the recommendations as written).
2. **Spike: done 2026-09-14** (Appendix D). The counter-heap API works; SP1's open design question is whether to migrate `Metal4Context` to `MTL4CommandBuffer` inside SP1 or defer that and ship the interim isolation mode first.
3. Brainstorm and write the **SP0 + SP1 design spec** (`docs/superpowers/specs/`) — the RFC — then the implementation plan via the writing-plans skill.
4. Run the **skip audit** in parallel (it is mechanical and informs SP2's scope).

---

## Appendix A — Local environment fingerprint (captured 2026-09-14)

| Field | Value |
|---|---|
| Chip | Apple M3 Max, `Mac15,9` |
| CPU | 16 cores (12 performance, 4 efficiency) |
| GPU | 40 cores |
| Memory | 48 GB |
| OS | macOS 26.5.2, build 25F84 |
| Swift | 6.3.3 (swiftlang-6.3.3.1.3, clang-2100.1.1.101) |
| Xcode | 26.6 (17F113) |
| Metal toolchain (per ledger) | 32023.883 |
| Power | AC; Low Power Mode off |
| Thermal state | nominal (`ProcessInfo.thermalState = 0`) |
| CI runner | `macos-26`, Xcode 26.0.1, virtualized GPU |

## Appendix B — Sources consulted

Measurement and statistics
- Kalibera & Jones, *Rigorous Benchmarking in Reasonable Time* (ISMM 2013): https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf
- Daly et al., *The Use of Change Point Detection to Identify Software Performance Regressions in a CI System* (ICPE 2020): https://arxiv.org/pdf/2003.00584
- Fleming et al., *Hunter: Using Change Point Detection to Hunt for Performance Regressions*: https://arxiv.org/pdf/2301.03034
- *8 Years of Optimizing Apache Otava* (E-Divisive engineering): https://arxiv.org/pdf/2505.06758
- Nyrkiö on GitHub Actions (MooBench study): https://arxiv.org/html/2510.11310
- Bencher threshold models: https://bencher.dev/docs/explanation/thresholds/

Apple Silicon environment
- ETH Zürich, *Benchmarking M-series Apple CPUs* (course notes): https://acl.inf.ethz.ch/teaching/fastcode/2025/benchmarking_m_series_apple_cpus.pdf
- *Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC*: https://arxiv.org/pdf/2502.05317
- apple-silicon-llm-bench (noise figures): https://github.com/john-rocky/apple-silicon-llm-bench
- Thermal-state granularity caveat: https://stanislas.blog/2025/12/macos-thermal-throttling-app/
- `powermetrics`: https://ss64.com/mac/powermetrics.html
- macOS memory-bandwidth tool: https://github.com/timoheimonen/macOS-memory-benchmark

Metal timing and validation
- wgpu issue #9414, legacy timestamp queries return zeros on macOS 26: https://github.com/gfx-rs/wgpu/issues/9414
- `MTL4CounterHeap.resolveCounterRange`: https://developer.apple.com/documentation/metal/mtl4counterheap/resolvecounterrange:
- Understanding the Metal 4 core API: https://developer.apple.com/documentation/Metal/understanding-the-metal-4-core-api
- GPU counters and counter sample buffers: https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers
- Validating your app's Metal shader usage: https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage/
- `MetalValidation(1)` man page: https://keith.github.io/xcode-man-pages/MetalValidation.1.html
- Flutter/Impeller, enabling Metal validation without Xcode: https://github.com/flutter/flutter/blob/main/docs/engine/impeller/docs/metal_validation.md

Vector-search benchmarking
- ANN-Benchmarks paper: https://arxiv.org/pdf/1807.05614
- Big ANN Benchmarks: https://big-ann-benchmarks.com/neurips21.html
- *ANN Search: Recall What Matters*: https://arxiv.org/pdf/2606.04522
- *Towards Robustness: A Critique of Current Vector Database Assessments*: https://arxiv.org/pdf/2507.00379
- Weaviate ANN benchmark methodology: https://docs.weaviate.io/weaviate/benchmarks/ann

Swift tooling
- ordo-one/package-benchmark: https://github.com/ordo-one/package-benchmark and https://www.swift.org/blog/benchmarks/
- jemalloc dependency dropped (1.35): https://forums.swift.org/t/ordo-one-benchmark-drops-dependency-on-jemalloc/87624
- Swift libFuzzer integration: https://github.com/apple/swift/blob/main/docs/libFuzzerIntegration.md
- github-action-benchmark: https://github.com/benchmark-action/github-action-benchmark
- Bencher: https://github.com/bencherdev/bencher
- CodSpeed on CI noise: https://codspeed.io/blog/benchmarks-in-ci-without-noise

Observability
- OpenTelemetry-swift signpost integration: https://github.com/open-telemetry/opentelemetry-swift/tree/main/Sources/Instrumentation/SignPostIntegration
- Custom Instruments packages: https://www.appspector.com/blog/building-custom-xcode-instruments-package

Numerical validation
- NVIDIA, *Floating Point and IEEE 754* (ULP/relative error, CPU-vs-GPU differences): https://docs.nvidia.com/cuda/floating-point/index.html
- Roofline model: https://modal.com/gpu-glossary/perf/roofline-model

Soak testing
- https://www.radview.com/blog/soak-testing-software-playbook-memory-leak-detection-stability/

## Appendix C — Repository references used in §2

`AUDIT.md` §E (190–226) · `CHANGELOG.md` 0.4.2/0.5.0/0.6.0 · `README.md` 5, 238, 274–300, 529–531, 568–586, 596–609, 626 · `docs/audits/HARDENING-HANDOFF.md` §3.3, §5, §6 · `docs/audits/REVIEW-2026-09-14-slice32.md` · `docs/stability/FUSED-TOPK-INSTRUMENTATION-CONTRACT.md` (uncommitted) · `docs/ADD/metal4/MASTER_PLAN.md` 514–597 · `docs/ADD/QUALITY_IMPROVEMENT_ROADMAP.md` 1461–1502 · `docs/superpowers/plans/2026-08-16-hardening-audit-phase0-1.md` · `.github/workflows/ci.yml` · `Sources/VectorAccelerate/Benchmarking/*` · `Sources/VectorAccelerateBenchmarks/*` · `Sources/VectorAccelerate/Core/{Metal4Context,BufferPool,GPUDecisionEngine,Logger,GPUHealthMonitor,ThermalStateMonitor}.swift` · `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift` · `Tests/VectorAccelerateTests/Utilities/{PerfGate,TestRNG,TestDataGenerator}.swift` · `Tests/VectorAccelerateTests/Hardening/{DifferentialKernelVsCPUTests,RoutingProvenanceTests,FusedTopKInstrumentationTests}.swift` · sibling repos: `VectorCore/Sources/VectorCoreBenchmarking/{Models,EnvCapture}.swift`, `VectorCore/.github/workflows/benchmarks.yml`, `VectorCore/docs/beta-evolution-2/Perf_Benchmarking.md`, `VectorIndex/docs/superpowers/plans/2026-07-31-vectorindex-0.2.0-phase3-perf.md`, `VSK/VECTORINDEX_RELIABILITY_ASSESSMENT_2026-08-15.md`, `VSK/VectorBench/README.md`.

## Appendix D — Timing spike (2026-09-14, throwaway)

Evidence directory: `docs/superpowers/research/spikes/2026-09-14-mtl4-timestamps/` (report, single-file source, three measured runs, one API-validation run). Machine as in Appendix A. Workload: runtime-compiled `float4` copy of 64 MiB (128 MiB traffic), one dispatch per `MTL4CommandBuffer`, 20 warm-up + 100 measured, QoS user-interactive.

| Clock | Brackets | Median (run 1) | RSD |
|---|---|---|---|
| wall (`ContinuousClock`) | encode + commit + signal + wait | 0.545–0.567 ms | 43–48% (outlier-driven) |
| command buffer (`MTL4CommitFeedback`) | whole buffer | 0.363 ms without stamps, 0.371 ms with | 3–12% |
| command-buffer heap stamps | cb-before → cb-after | 0.368–0.370 ms | 3.6–4.5% |
| **encoder heap stamps** | dispatch only | **0.361 ms** (372 GB/s implied) | **2.8–3.4%** |

Findings: encoder-level stamps non-zero and monotonic in 600/600 pairs at both granularities; `.precise` indistinguishable from `.relaxed` here; heap stamps add 7–13 µs to the command-buffer span only; never mix a command-buffer stamp with an encoder stamp in one delta; `queryTimestampFrequency()` = 24 MHz; legacy stage-boundary sampling also works (nanoseconds), dispatch/blit/draw boundaries unsupported on this device; `.private` legacy sample buffer + blit resolve returned one-run-stale values (not investigated). Precondition for library-side instrumentation: `Metal4Context` must move from the legacy `MTLCommandQueue` (`Core/Metal4Context.swift:128`, `:233`) to `MTL4CommandBuffer` encoders.
