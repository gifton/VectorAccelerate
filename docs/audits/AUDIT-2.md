# AUDIT-2 — Hardening Audit Findings Ledger

**Started:** 2026-08-16 · **Scope so far:** Phase 0 (honest baseline) + Phase 1.1 (GPU↔CPU differential harness) + Phase 1.2 (shader-compile duality closure), per `docs/superpowers/plans/2026-08-16-hardening-audit-phase0-1.md`.
**Baseline at audit start (branch `gifton/metal-compute-provider`, `029d250`):** debug suite green (1540 tests); **release-config suite RED: 37 failures / 1540** (all in `MetalComputeProviderTests`, `MetalComputeProviderSoATests`, `SoAFusedTopKTests`, `SoAScoringParityTests`, `SoAKernelGoldenTests`, `OperationsDispatchTests`). Release config had never been CI-tested.

Status legend: **fixed-in-tree** (fix + regression guard landed in this audit branch) · **pinned** (defect encoded as an `XCTExpectFailure` guard; fixing it flips the guard) · **decision-needed** (fix changes observable behavior/contract; owner call required) · **observation** (no action yet).

New permanent guards from this audit, all under `Tests/VectorAccelerateTests/Hardening/`:
`RoutingProvenanceTests` (which silicon served each provider call, via the new `MetalComputeProvider.RoutingTelemetry`), `ShaderLibraryCompletenessTests` (runtime library ↔ shader directory ↔ metallib, generated not hand-listed), `PreambleParityTests` (preamble ↔ `Metal4Common.h` numeric identity + symbol coverage), `DifferentialKernelVsCPUTests` (GPU legs vs `AccelerateFallback` vs Double oracle across adversarial value classes). Plus `VA_AUDIT_TRACE=1` (env) prints one line per GPU command-buffer submission for per-test attribution.

---

## VA2-001 — `SoADistance.metal` missing from the runtime shader compile: the 0.6.0 provider is unusable in release builds

**Severity:** CRITICAL · **Status:** fixed-in-tree

The runtime combined-source compile (`KernelContext.makeLibraryFromBundleSources`) assembled 27 hand-listed files; the shader directory has 30. In release builds this compile is the **only** load path — the resource bundle ships no `default.metallib` (MetalCompilerPlugin supplants SPM's metallib step) and `debug.metallib` is only consulted under `#if DEBUG`. `SoADistance.metal` (the 0.6.0 zero-copy kernels) was not in the list, so `SoADistanceKernel.init` threw — and since `MetalComputeProvider.init` constructs it eagerly, **every provider construction failed in every release process**.

- **Evidence:** pristine-HEAD `swift test -c release`: 37 failures, e.g. `VectorError.resourceUnavailable: Shader 'VectorAccelerate: soa_l2_distance/soa_cosine_distance not found'`; release bundle listing (30 `.metal` + `debug.metallib`, no `default.metallib`).
- **Why debug never saw it:** debug loads `debug.metallib` (all 187 kernels, compiled per-file by the plugin).
- **Fix:** `SoADistance` added to `KernelContext.runtimeCompileShaderFiles`; list hoisted to an internal constant with an explicit exclusion map; `ShaderLibraryCompletenessTests` now asserts (a) every kernel of every non-excluded file exists in the runtime-compiled library, (b) listed ∪ excluded exactly tiles the directory, (c) runtime library ≡ metallib. A forgotten file is now a red test, not a release-only outage.

## VA2-002 — Dead shaders with toxic macros: `ManhattanDistance.metal`, `ChebyshevDistance.metal`

**Severity:** MEDIUM · **Status:** **fixed-in-tree (deleted, owner-approved 2026-08-16)**

Nine kernels across these two files had **zero Swift call sites** (`MetalComputeProvider` computes both metrics on CPU by policy). They shipped in the metallib as dead weight, and they could not join the combined runtime compile: each `#define`d `TILE_Q/TILE_N/TILE_D/TILE_D_VEC` unguarded, and in the single combined translation unit those macros mangled identifiers in `OptimizedMatrixOps.metal` (empirically reproduced: `expected unqualified-id` killing all ~190 kernels). **Both files are deleted**; the exclusion map is empty again and the tiling test enforces that every remaining `.metal` file is compiled. If GPU L1/L∞ is ever wanted, rewrite against `Metal4Common.h` conventions.

## VA2-003 — The provider's batch-distance GPU path is unreachable under any default configuration

**Severity:** HIGH · **Status:** **fixed-in-tree (owner-approved 2026-08-16)** · verified by `RoutingProvenanceTests.testDefaultConfigBatchDistanceRoutesByScale`

`MetalComputeProvider.batchDistance` consulted `GPUDecisionEngine.shouldUseGPU` with `k: 0`. The engine hard-gated `k >= minKForGPU` (default 10) and `queryCount·candidateCount·k >= minOperationsForGPU` (default 50 000) — both **always false at k = 0**. Consequences: the "no-copy GPU kernel path" that headlined 0.5.0 was dead code through the façade; `distanceMatrix` (row-wise delegation) was CPU-always. **Fix:** the k-gates now apply only to selection-shaped operations (`topKSelection`, `bitonicSort`); distance-shaped operations gate their complexity on `queryCount·candidateCount·dimension` — the actual per-candidate work. Verified live by telemetry: `batchDistance` at 2000×128 dispatches the GPU kernel under default configuration, while 300×128 still routes to CPU (below `minCandidatesForGPU`). Side effect (intended): other k-less operations (`normalization`, `ivfAssignment`, …) that consulted the engine with small/zero k are likewise reachable now. The operation-specific floors in `evaluateOperationSpecificCriteria` remain non-threshold-relative — acceptable defaults, revisit only if a consumer needs to force small-workload GPU.

## VA2-004 — The provider test suite's "GPU vs CPU parity" tests compared CPU to CPU

**Severity:** HIGH (test debt) · **Status:** remediated by provenance guards

Every provider in `MetalComputeProviderTests` is built with default configuration, so (per VA2-003) its batch-distance "GPU" legs ran on CPU — which is why a fully broken release GPU stack (VA2-001) coexisted with a green debug suite. The new `RoutingTelemetry` + provenance assertions make "the GPU actually served this" an assertable fact; `VA_AUDIT_TRACE=1` gives per-test GPU-submission counts for auditing the rest of the suite.

## VA2-005 — `findNearest` silently falls back to CPU when the GPU returns an empty result

**Severity:** MEDIUM · **Status:** **fixed-in-tree (owner-approved 2026-08-16)**

`MetalComputeProvider.findNearest` treated an empty fused-GPU result as "quietly recompute on CPU" — even with `fallbackToCPU == false`. For valid inputs a correct kernel can never produce an empty result, so that branch existed only to mask a malfunction. **Fix:** an empty GPU result is now a `VectorError.computeFailed` — thrown when `fallbackToCPU == false`, counted (`cpuFallbackEmptyGPUResult`) and recorded in `lastGPUErrorDescription` when fallback rescues it. The differential harness asserts the condition never fires on the current kernels.

## VA2-006 — `PipelineCacheKey.functionName` derived five function names that exist in no library

**Severity:** HIGH · **Status:** **fixed-in-tree (owner-approved 2026-08-16)** · hard-guarded by `testCommonPipelineKeysResolveToRealFunctions`

Phantom names, all verified against both libraries:
- `"cosineSimilarity", dim 0` → `cosine_similarity_kernel` — the real generic is `cosine_similarity_general_kernel`; **fixed** (dimension variants were already correct);
- `"topK"` → `top_k_selection` — **fixed** to the real `topk_select_batch_kernel`;
- `"fused_l2_topk"` → underscore-stripping → `fusedl2topk` — **fixed**: `fused_*` operations pass through unchanged (they ARE literal kernel names);
- `.batch("euclideanDistance")` → `"batch" + tail.capitalized` = `batchEuclideandistance` (Swift's `.capitalized` lowercases the rest of the word) — **fixed** to uppercase only the first character;
- **`batchCosineDistance`** — the worst of the five, discovered while fixing VA2-008: not a derivation at all but a literal name `Metal4ComputeEngine` requests at three sites (including `fusedDistanceTopK(metric: .cosine)`) **for a kernel that was never written** — only `batchCosineSimilarity` existed, with different semantics. The GPU cosine batch/fused paths had therefore never executed in any build; every dispatch threw `shaderNotFound` and the silent CPU fallback ate it. **Fixed** by implementing `batchCosineDistance` (dispatch-compatible with `batchEuclideanDistance`, VA2-008-safe finalization); the fused harness now runs a cosine leg to keep it working.

Loudness (owner-approved): `Metal4ShaderCompiler.compileMultiple` now throws an aggregate error naming every failed key; `PipelineCache.warmUp` returns the failed keys (`@discardableResult`); `Metal4Context`'s init-time warm-up asserts on any failure in debug builds. The resolution probe is a hard assertion — a new phantom is a red test.

## VA2-007 — GPU-vote and CPU-vote `findNearest` used different tie-break/ordering semantics

**Severity:** MEDIUM · **Status:** fixed-in-tree (same change as VA2-010)

The engine's fused path selected via unstable `Array.sorted` (unspecified equal-distance order); the CPU path via VectorCore `TopKSelection` with deterministic `.smallerIndex` ties. Identical inputs could return differently-ordered neighbors depending on which side served the call. Both paths now share `TopKSelection`.

## VA2-008 — GPU cosine kernels misclassified finite vectors as degenerate outside ~[1e-19, 1e19] component range

**Severity:** CRITICAL (numerics) · **Status:** **fixed-in-tree (owner-approved 2026-08-16)** · hard-guarded across all value classes by the differential harness

The live cosine kernels (`cosine_similarity` in DistanceShaders, `cosineDistance` in BasicOperations, `soa_cosine_distance`, and the new `batchCosineDistance`) accumulated `Σv²` naively in FP32: finite `huge` inputs (|c| ≈ 1e19) overflowed to +Inf, `tiny` (≈1e-20) collapsed to 0 under flush-to-zero — both misclassified the vector as degenerate (distance 1.0 or NaN). 145 diverging harness cells at audit time.

**Fix — shared trio in `Metal4Common.h` (mirrored byte-identically in the runtime preamble, drift-guarded by `testCosineRescueBlockIdentical`):** `va_cosine_accumulators_unreliable` (fold-proof magnitude-compare trigger), `va_cosine_rescaled_terms` (pre-scaled recompute in the `VA_NORM_*` clamp domain — the 0.6.0 normalize algorithm, finally ported to distance), and `va_cosine_similarity_finalize` (NaN-propagating, `precise::divide`, per-norm FLT_MIN floor). Three additional fast-math mechanisms were **measured** en route and are documented in the header, because each independently silently corrupted results:

1. fast math reassociates `sqrt(aa)*sqrt(bb)` into `sqrt(aa*bb)`, whose argument overflows for |c| ≳ 1e18 → similarity 0;
2. fast math rewrites `(dot/normA)/normB` into `dot·rcp(normA·normB)`, whose reciprocal is **subnormal** for norm products ≳ 8.5e37 → flushed → similarity 0 even with every accumulator finite (hence `precise::divide` — the same lesson the normalize kernels already carried);
3. the plugin-built metallib **folds `isinf()` under fast math** while runtime compilation of identical source does not — the trigger now uses `x > FLT_MAX` dynamic compares.

Policy decision recorded: vectors whose largest magnitude is **subnormal** are degenerate (similarity 0) on BOTH legs — flush-to-zero makes them unreadable operands on the GPU, and the normalize family already treats them as unnormalizable; leg interchangeability wins over CPU-only extra range.

## VA2-009 — NaN policy divergence between GPU cosine and its CPU fallback

**Severity:** HIGH · **Status:** **fixed-in-tree (owner-approved 2026-08-16)** · hard-guarded by the harness NaN classes

`AccelerateFallback`'s cosine swallowed NaN inputs into similarity 0 (its `norm > 0` guard is false for NaN) while the GPU propagated NaN — results changed depending on which silicon served the call. **Fix:** both cosine functions now share `cosineSimilarityCore`: single-precision vDSP primary path gated on *normal* accumulators, Double-accumulation rescue for overflow/subnormal/zero-norm cases (mirroring the GPU's pre-scaled rescue), NaN propagation everywhere, `[-1, 1]` NaN-preserving clamp. Also fixed in passing: `batchCosineSimilarity` read out of bounds for ragged candidates (`vDSP_dotpr` over `query.count` against a shorter row) — ragged pairs now return NaN; and a subnormal *accumulator* (not just norm) routes to the rescue, since it carries as few as ~10 mantissa bits (measured 3.9e-5 error at dim 1).

## VA2-010 — One NaN distance scrambled the entire fused top-K selection

**Severity:** HIGH · **Status:** fixed-in-tree

`Metal4ComputeEngine.fusedDistanceTopK` (the provider's GPU-vote path — GPU distances, then CPU selection) sorted with `{ $0.1 < $1.1 }`. With any NaN distance present this comparator violates strict weak ordering and `Array.sorted`'s result is unspecified — measured: one NaN-poisoned candidate among 5000 made the "top-k" return **non-minimal neighbors with plausible distances** (9/9 poisoned harness cells; GPU per-candidate distances verified correct against the oracle — only the selection was corrupted). Fixed by selecting via `TopKSelection` (NaN-tolerant heap, `.smallerIndex` ties); harness green including poisoned classes. Note the kernel-side lesson: "fused" is a misnomer — selection happens on CPU; the doc comment still describes a fused GPU pipeline.

## VA2-011 — `Metal4Context.execute` never checks `commandBuffer.error`

**Severity:** MEDIUM · **Status:** confirmed, not yet fixed (Phase-2 P2 work package)

`execute` commits and waits (`commitAndWait`, despite its "submitted asynchronously" doc comment) but, unlike `executeAndWait` and `executeBlitAndWait`, never inspects `commandBuffer.error` — a faulted command buffer on this path returns normally and callers read stale/garbage buffer contents. Fix alongside the Phase-2 execution-core package (which also owns: per-call unretained `MTLSharedEventListener`, non-monotonic `signaledValue` assignment from concurrent completion handlers, no timeout on event waits, throw-mid-encode skipping `endEncoding`, and residency sets tracked but never attached).

## VA2-013 — MetalCompilerPlugin does not track header dependencies: header-only edits ship stale kernels

**Severity:** MEDIUM (build hygiene) · **Status:** confirmed 2026-08-16, workaround known, upstream/decision-needed

Editing ONLY `Metal4Common.h` does not trigger recompilation of `debug.metallib` — the plugin's dependency tracking apparently keys on the `.metal` files' content, not their includes. Measured during the VA2-008 fix: a finalize change made solely in the header left the deployed metallib on the previous version while freshly-compiled probes behaved correctly, which cost a full diagnostic loop chasing a "fixed but still failing" kernel. Workaround: `touch Sources/VectorAccelerate/Metal/Shaders/*.metal` after header edits. **Recommendation:** file upstream against MetalCompilerPlugin (or add the touch to a build script); any future shared-header work must assume this trap exists.

## VA2-012 — Observations (no action yet)

- The combined runtime compile's `'EPSILON' macro redefined` warning is now explained: `DistanceShaders.metal` `#define`s `VA_EPSILON 1e-8f`, and the compiler's textual `VA_EPSILON → EPSILON` replacement turns that into a bare `#define EPSILON 1e-8f` after the preamble's guarded `1e-7f`. Consequence: in the COMBINED build, files listed after DistanceShaders see `EPSILON = 1e-8` while their per-file metallib compiles see `1e-7` (via `constant float EPSILON = VA_EPSILON`) — a live metallib-vs-runtime numeric drift for any EPSILON-using kernel downstream of DistanceShaders in the list. Small blast radius (epsilon guards, not results), but it is exactly the drift class this audit exists to eliminate — normalize when the affected files are next touched.
- `UMAPGradientKernel` GPU benchmark underperforms its own expectations locally (0.6–1.6× vs 2–5× expected ⚠) in both debug and release baseline runs — queued for the Phase-2 UMAP package (which also owns the relaxed `atomic_float` determinism question).
- Skips on a GPU-present machine (21 debug / 34 release) are concentrated in `IVFValidationTests` and `NeuralQuantization*Tests` and are mostly `XCTSkip("Not yet implemented")` placeholders — dormant coverage, not availability-vacated GPU tests.
- Trace-measurement caveat: `VA_AUDIT_TRACE` attribution binds submissions to the test active at print time and keys by method name; asynchronous warm-up work can smear across boundaries. The per-test numbers below are directional except for synchronous provider paths, which bind exactly.

---

## Verification state after the remediation slice (2026-08-16, owner decisions executed)

All five owner decisions implemented and verified: **debug 1557 tests / 0 failures; release 1557 tests / 0 failures** (21 skipped in both — the pre-existing "Not yet implemented" placeholders). The differential harness passes with **every** adversarial value class as a hard assertion (no remaining expected-failure pins); the fused top-K harness now includes a cosine leg exercising the newly-written `batchCosineDistance`. Fixed this slice: VA2-002 (deleted), VA2-003, VA2-005, VA2-006 (all five phantoms + loud warm-up), VA2-008 (incl. three measured fast-math mechanisms), VA2-009. Newly recorded: VA2-013 (plugin header-dependency staleness). Remaining open: VA2-011 (Phase-2 P2), VA2-012 observations, CI release leg (deferred by owner).

## Verification state after the initial audit slice (2026-08-16)

- **Debug, full suite with all audit changes:** 1555 tests (1540 + 15 new guards), **0 failures**, 21 skipped; 2 expected-failure pins active (VA2-006, VA2-008/009).
- **GPU-submission measurement** (`VA_AUDIT_TRACE=1`, full debug suite): 11,714 command-buffer submissions through `Metal4Context`; IVF/recall suites dominate legitimately (≈700–1,400 each). Empirical VA2-004 confirmation: `test_batchEuclideanDistance_forceGPU`, `testBatchDistanceEuclideanParity_CPUandGPU`, and `testBatchDistanceCosineParity_CPUandGPU` each attributed **zero** GPU submissions.
- **Shader-validation leg** (`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, 52 tests across the differential harness + SoA/top-K/engine suites — validation confirmed enabled in-process): 0 failures, 0 validation faults. The adversarial dispatch matrix (dims 1–768, N 1–5000, k>N, NaN/Inf classes) triggers no out-of-bounds access on these paths.
- **Release config, fixed tree:** 1555 tests, **0 failures** (pristine HEAD was 37/1540 RED). The VA2-001 fix is verified end-to-end in the previously-broken configuration; all Hardening guards pass in release.
- Recommended CI addition: a release-configuration test leg (the 0.6.0 preamble regression and VA2-001 were both release-only and invisible to current CI).
