# Metal improvements wish list — what the observability / benchmark harness needs from the library

**Date:** 2026-09-14
**Audience:** the agent (or person) doing Metal hardening and improvements in `Sources/VectorAccelerate`.
**Origin:** the research brief `docs/superpowers/research/2026-09-14-observability-harness-research.md` and the timing spike in `docs/superpowers/research/spikes/2026-09-14-mtl4-timestamps/`. Read Appendix D of the brief for the measured numbers behind HW-01 to HW-03.
**Status of the harness program:** SP0 (foundation: schema, fingerprint, seeded kit, harness executable) is being designed now and needs nothing from this list. SP1 (kernel-level timing) is blocked on HW-01. SP2/SP3 (guardrails, observability) consume HW-05 onward.

## Ground rules that apply to every item

1. **Telemetry observes; it never alters which path runs.** This is the rule set in the 2026-08-16 hardening plan and enforced by `RoutingProvenanceTests`. Nothing here may change routing, thresholds, or results.
2. **Zero cost when off.** Every hook is behind a configuration flag read once into a stored property; the off path must add no allocation and no command-buffer work. The harness will later gate this mechanically (malloc count on the submit path with instrumentation off).
3. **House discipline applies:** red-first test, a `docs/stability/*-CONTRACT.md` for anything that states a guarantee, a guard suite in `Tests/VectorAccelerateTests/Hardening/`, ledger entry, and the claim rule (a correctness claim in a comment names its test or is marked unverified).
4. **Both metallib paths.** Anything that touches pipelines is tested under the plugin `debug.metallib` and the runtime-compiled source library, per the existing convention.
5. **Do not add a third clock.** Wall time is `ContinuousClock`; GPU time is commit feedback or counter heaps. No `CFAbsoluteTimeGetCurrent`, no `CACurrentMediaTime` in new code.

## Summary

| ID | Priority | Item | Unblocks |
|---|---|---|---|
| HW-01 | P0 | Migrate the `Metal4Context` submit path from legacy `MTLCommandQueue`/`MTLCommandBuffer` to `MTL4CommandBuffer` with `MTL4CommitFeedback` | SP1 entirely |
| HW-02 | P0 | Opt-in per-encoder counter-heap timestamps | SP1 kernel time |
| HW-03 | P0 | Per-submission timing records with submission IDs (replace the single `_lastGPUTiming` slot) | SP1, SP3 |
| HW-04 | P0 | Verify the `executeAndWait` wake-up is exact | SP1 wall-time validity |
| HW-05 | P1 | Stable labels on every encoder and dispatch | SP1 attribution, SP3 signposts |
| HW-06 | P1 | `RoutingDecision` records with reason codes at every routing site | SP3, VA3-024 analysis |
| HW-07 | P1 | One `InstrumentationConfiguration` replacing the five scattered `enableProfiling` flags | SP3 |
| HW-08 | P1 | os_signpost intervals (submit, encoder, routing, pool) | SP3 |
| HW-09 | P1 | Buffer pool high-water mark, live-lease count, eviction count | SP2 soak mode |
| HW-10 | P1 | Pipeline and argument-table cold/warm counters; per-pipeline threadgroup-footprint introspection | SP2, SP4 first-call cost |
| HW-11 | P1 | Injectable RNG at the nine unseeded `Float.random` sites in shipped code | SP2 determinism, SP4 reproducibility |
| HW-12 | P1 | Validation-layer green: the three known `MTL_SHADER_VALIDATION=1` failures, plus a whole-library static-footprint gate | SP2 validation gate |
| HW-13 | P1 | Repetition hook for relaxed-atomic kernels so run-to-run variation can be measured (VA3-019 accept-and-bound) | SP2 |
| HW-14 | P2 | Per-kernel cost model (`bytesMoved`, `flops` as functions of shape) | SP4 roofline guardrail |
| HW-15 | P2 | Counters for command buffers per operation, encoders per buffer, residency commits, argument-table reuse | SP4 (MASTER_PLAN §6 targets become measurable) |
| HW-16 | P2 | Structured os_log categories instead of one interpolated string | SP3 |

---

## P0 — timing substrate

### HW-01 — Migrate the submit path to Metal 4 command buffers

**What.** `Metal4Context` creates a legacy queue (`Core/Metal4Context.swift:233`, `device.rawDevice.makeCommandQueue()`; stored as `any MTLCommandQueue` at `:128`) and reads `MTLCommandBuffer.gpuStartTime/gpuEndTime` (`:436`, `:512`). Move to `MTL4CommandQueue` + `MTL4CommandAllocator` + `MTL4CommandBuffer` + `MTL4ComputeCommandEncoder`, with GPU span taken from `MTL4CommitFeedback.gpuStartTime/gpuEndTime` delivered via `MTL4CommitOptions.addFeedbackHandler` on `queue.commit([cb], options:)`.

**Why the harness needs it.** Counter heaps (HW-02) cannot be written from a legacy encoder, and `MTL4CommandBuffer` has no `gpuStartTime`. Without this, the harness can only time whole command buffers with a measured +2 to +10 µs bias over the kernel span, which dominates for small kernels and cannot attribute time inside fused or multi-encoder pipelines.

**Suggested shape.** Transitional `submitPath: .legacy | .metal4` on `Metal4Configuration`, with the existing parity suites run against both, then remove `.legacy`. The spike's pattern (allocator `reset()` after the shared-event wait, then `beginCommandBuffer` again; `commit` + `signalEvent`; CPU `MTLSharedEvent.wait(untilSignaledValue:timeoutMS:)`) passed API validation over ~1,400 command buffers.

**Acceptance.** All 33 guard suites green on both paths in debug and release; `GPUTimingInfo` populated from commit feedback; `MTL_DEBUG_LAYER=1` run exits 0; a test proves the feedback handler fires exactly once per commit and that `gpuEnd >= gpuStart`.

### HW-02 — Opt-in per-encoder counter-heap timestamps

**What.** One `MTL4CounterHeap` of type `.timestamp` per context, a ring of two slots per encoder sized for the maximum in-flight submissions; `encoder.writeTimestamp(granularity: .relaxed, counterHeap:index:)` immediately before the first dispatch and after the last dispatch of each encoder; CPU `heap.resolveCounterRange(_:)` after the existing wait; conversion by `device.queryTimestampFrequency()` (24 MHz on M3 Max, 41.7 ns per tick).

**Why.** The encoder span is the only clock that measures the kernel itself: 0.361 ms at 2.8–3.4% RSD and stable to 0.4% across runs in the spike, unaffected by whether stamps are present. `.precise` bought nothing measurable with one dispatch per encoder and the header warns it may split encoders; use `.relaxed`.

**Rules.** Never subtract a command-buffer-level stamp from an encoder-level stamp (their order inverted in 6–14% of samples). Any stamp in a buffer adds 7–13 µs to the *buffer* span, so the feedback span must be reported as "with instrumentation" when stamps are on. Off means no heap is created.

**Acceptance.** With timestamps on, the copy-kernel encoder span is within 3% of the feedback span minus the documented stamp overhead; with timestamps off, no `MTL4CounterHeap` exists and the submit path performs zero allocations (test with `malloc_zone_statistics` before/after, or with package-benchmark once adopted); unwritten slots resolve as 0 and are never reported as a span.

### HW-03 — Per-submission timing records

**What.** Replace `_lastGPUTiming` (single slot, overwritten per dispatch, races under concurrent submits) with a monotonically increasing `SubmissionID` and a record `{ id, labels: [String], cpuEncodeNs, cpuWaitNs, feedback: (gpuStart, gpuEnd)?, encoderSpans: [(label, ticks)] }`. Return it with the operation result where the API allows, and keep the last N in a bounded ring readable by ID.

**Why.** The harness runs two-level sampling (samples × iterations) and must pair each measurement with its own submission; the index and batch paths issue several buffers per operation.

**Acceptance.** Two concurrent `executeAndWait` calls from separate tasks yield two distinct records with distinct IDs and no overwrite; ring capacity is bounded and documented.

### HW-04 — Verify the `executeAndWait` wake-up is exact

**What.** The memory ledger notes an unfiled "`Metal4Context.executeAndWait` early-wake hazard". If the CPU wait can return before the GPU has completed, every wall-time and feedback-based number the harness records is wrong. Confirm the wait is on a shared-event value that is signalled only after completion, or fix it.

**Acceptance.** A test that submits a deliberately slow kernel (large copy) and asserts that on return, feedback `gpuEnd` is populated and the output buffer is fully written.

---

## P1 — attribution and observability hooks

### HW-05 — Stable labels on every encoder and dispatch

**What.** Each encoder gets `label = "<kernelFunctionName> n=<N> d=<D> k=<K>"` (or a structured `KernelLabel`), and the label rides on the timing record and the signpost. Fused pipelines label each encoder distinctly.

**Acceptance.** A fused pipeline reports one encoder span per encoder with the expected labels; labels are stable across runs (no addresses, no timestamps in them).

### HW-06 — `RoutingDecision` records with reason codes

**What.** `GPUDecisionEngine.shouldUseGPU(...)` returns a bare `Bool` (`Core/GPUDecisionEngine.swift:333`). Add an `explain`/`decide` variant returning `RoutingDecision { operation, n, d, k, queryCount, chosen: .gpu/.cpu, reason: Reason, thresholds: snapshot, adaptiveRatio, thermalState, poolUtilization, healthLevel }` with `Reason` an enum (`belowThreshold`, `aboveThreshold`, `policyCPU`, `policyGPU`, `healthFallback`, `thermalThrottle`, `gpuError`, `emptyGPUResult`, `adaptiveRatio`). Emit it from every routing site: `MetalComputeProvider` (already has counters), `Metal4ComputeEngine`, `AcceleratedVectorIndex`, `BatchProcessor`. Bounded ring plus optional `AsyncStream`. The adaptive mutation of `gpuPerformanceRatio` (`:527–530`) must be visible in the record.

**Why.** VA3-024 and the crossover calibration need to know *why* a path was taken, not only that it was. Today anything not routed through `MetalComputeProvider` is invisible.

**Acceptance.** `RoutingProvenanceTests` extended to the engine, index, and batch paths; a test proves reading records does not change the next decision (P1 rule).

### HW-07 — One `InstrumentationConfiguration`

**What.** There are five separate `enableProfiling` flags (`Metal4Configuration`, `Metal4ComputeEngineConfiguration`, `ClusteringConfiguration`, `IVFConfiguration`, plus `IndexAccelerationConfiguration.enableProfiling`) and one dead `logDecisions`. Replace with one struct `{ timestamps, signposts, decisionRecords, ringCapacity }` reachable from all of them (or referenced by them), read once at context creation.

**Acceptance.** One switch turns everything on; the old flags are deprecated with a message naming the new one; the dead `logDecisions` is deleted.

### HW-08 — os_signpost intervals

**What.** `OSSignposter(subsystem: "com.vectoraccelerate", category: ...)` with categories `submit`, `kernel`, `routing`, `pool`, `pipeline`, `index`. Intervals: per command buffer submit → scheduled → completed keyed by `SubmissionID`; per encoder (using the label from HW-05); per routing decision as an event; per pipeline compile. Use `signposter.isEnabled` to skip formatting when off. Structured fields (`%{public}s` per field), never one interpolated string.

**Acceptance.** An Instruments trace of one index search shows nested intervals per kernel; the off path adds no allocation (same mechanism as HW-02).

---

## P1 — resource and determinism guardrails

### HW-09 — Buffer pool accounting

**What.** `BufferPool.PoolStatistics` (`Core/BufferPool.swift:680`) lacks a high-water mark and an explicit live-lease count. Add `highWaterMarkBytes`, `liveLeaseCount`, `peakLiveLeaseCount`, `evictionCount`, and a `generation` counter that increments on `reset()`; document reset semantics (retired leases excluded, as `:478` already says).

**Why.** The soak mode's signal is the slope of these counters over hours; without a high-water mark, a leak that returns to baseline between samples is invisible.

**Acceptance.** A lifecycle test with a scripted lease/return sequence asserts the exact high-water mark and peak live count; `BufferPoolLifecycleTests` extended, contract doc updated.

### HW-10 — Pipeline and argument-table cold/warm counters, footprint introspection

**What.** `PipelineCacheStatistics` and `ArgumentTablePoolStatistics` should distinguish cold compiles from cache hits with a count and cumulative compile time, so a run can report "first call included N compiles totalling T ms". Add an introspection API returning, per cached pipeline, `staticThreadgroupMemoryLength`, `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`.

**Why.** SP4 reports first-call cost separately from steady state; SP2's validation gate needs every kernel's static footprint (slice 34 showed instrumentation doubles it).

**Acceptance.** A test compiles every function in both metallibs and asserts `staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength` (extends `FusedTopKInstrumentationTests` to all ~180 kernels); cache statistics assert hit/miss counts for a warm second call.

### HW-11 — Injectable RNG at the unseeded sites

**What.** Nine `Float.random`/`Int.random` sites in shipped code have no injectable generator: `BenchmarkFramework.swift:318` (being deleted in SP0), `Core/Types.swift:249`, `Core/TensorManager.swift:396–397`, `KMeansPlusPlusKernel.swift:131`, `ProductQuantizationKernel.swift:587`, `ML/QuantizationEngine.swift:593`, `UMAPGradientKernel.swift:679–681`, `WarpOptimizedSelectionKernel.swift:796/838`, `QuantizationStatisticsKernel.swift:547`. Add a `RandomNumberGenerator` parameter (default `SystemRandomNumberGenerator()`) or a seed on the relevant configuration.

**Why.** Reproducible benchmarks of k-means++, PQ training, UMAP, and quantization statistics are impossible otherwise; the two IVF throughput placeholders depend on seeded k-means.

**Acceptance.** Same seed → identical output for the non-atomic paths (test per site); a class-closing grep test enumerates `.random(` occurrences in `Sources/` against an allowlist and fails on new unseeded sites.

### HW-12 — Validation-layer green

**What.** Under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` three failures are recorded in `docs/stability/FUSED-TOPK-INSTRUMENTATION-CONTRACT.md`: `ivf_list_search` at 55,296 B vs the 32,768 B limit (in progress as of this writing), the `includeDistances: false` fused dispatch leaving buffer(3) unbound (SIGABRT under API validation), and the unary elementwise `input_b` binding at index 1. Close all three.

**Why.** SP2 adds `Scripts/validate.sh` and a CI job that runs the suite under both variables; it starts non-blocking and flips to blocking when green. Note the variables must be set before the first device is created in the process, so they live in the script, not in test code.

**Acceptance.** `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test` exits 0 in debug and release; the HW-10 footprint test passes under instrumentation.

### HW-13 — Repetition hook for relaxed-atomic kernels (VA3-019 accept-and-bound)

**What.** The owner's decision is to accept run-to-run variation from relaxed `atomic_float` accumulation (UMAP target gradients, PQ training, k-means update) and bound it empirically. The harness needs to run each such kernel N times on identical inputs and read the outputs to compute the observed spread. Ensure these entry points are re-runnable on the same input without hidden state, and list them (a static `nondeterministicKernels` registry or a doc table).

**Acceptance.** A test runs each listed kernel 10× on a seeded input and asserts the spread is within the bound published in a new `docs/stability/ATOMIC-ACCUMULATION-CONTRACT.md`; the bound is derived from measurement, not chosen.

---

## P2 — cost model and counters

### HW-14 — Per-kernel cost model

**What.** A `KernelCostModel` per kernel: `bytesRead(shape)`, `bytesWritten(shape)`, `flops(shape)` as pure functions of `(n, d, k, queryCount)`.

**Why.** The harness computes achieved bandwidth as a fraction of the preflight STREAM number; a kernel dropping below its recorded fraction is a regression independent of absolute time, which is what makes the gate portable across the M2 Pro, M3 Max, and M4 Pro.

**Acceptance.** For the copy kernel the model reproduces the spike's 372 GB/s within 2%; the distance kernels' models are checked against the SoA layout contract's byte counts.

### HW-15 — Operation-level counters

**What.** Per operation: command buffers issued, encoders per buffer, residency-set commits, argument-table reuse vs allocation, pipeline-cache hits. These are exactly the `MASTER_PLAN.md §6` targets (1–3 buffers per op, <10 residency commits per batch, >95% cache hits, >80% argument-table reuse) that were never measurable.

**Acceptance.** Counters exposed on the timing record from HW-03; a test asserts a single flat search issues the documented number of command buffers.

### HW-16 — Structured os_log categories

**What.** `Core/Logger.swift` interpolates everything into one `%{public}@` under one category and reaches five files, none of them kernels or the index. Split by the categories in HW-08 and log structured fields; add loggers to `Metal4Context`, the kernels' error paths, and `Index/`.

**Acceptance.** `log stream --predicate 'subsystem == "com.vectoraccelerate" && category == "routing"'` shows decision records with fields; no new `print` in production code (existing SwiftLint rule).

---

## Not requested (and why)

- **Deterministic-reduction variants of the atomic kernels.** Deferred by owner decision (Q4); revisit only if a consumer needs bit-reproducible training.
- **`MTLCounterSampleBuffer` (legacy) sampling.** Works at stage boundaries on M3 Max but cannot time a dispatch; superseded by HW-02.
- **Anything that changes routing thresholds.** Calibration is a harness output (SP4), not a library change.
