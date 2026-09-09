# AUDIT-3 — Independent Metal Shader Corpus Review

**Date:** 2026-08-16 · **Scope:** all 28 `.metal` files + `Metal4Common.h` (12,761 lines), read end-to-end, with Swift-side dispatch-geometry and liveness verification for every reachability-dependent finding.
**Method:** fresh full-file pass (not anchored on AUDIT-2), then per-finding verification against the actual Swift dispatch sites — a kernel bug's severity here reflects whether the *current* caller can trigger it.
**Relationship to AUDIT-2:** VA2 items that remain open are folded in where they belong; several AUDIT-2-era memory anchors were re-verified and a few turned out already hardened (noted inline).

Severity: **P1** = wrong results / UB / OOB reachable through live call paths · **P2** = wrong results behind one config/API step, policy inconsistencies, latent traps armed by dispatch changes · **P3** = hygiene, dead code, perf, documentation.

---

## Remediation slice 1 (2026-08-16, owner-approved: Group F + all P1s) — EXECUTED

**Verification: debug 1558 tests / 0 failures; release 1558 tests / 0 failures (two consecutive
clean runs after the flake fix below).** All work is red→green where the defect was
deterministically reproducible (VA3-001 published race values at dims 257/300/777; VA3-008 read
back poison bytes; VA3-010 leaked 1e6 padding into products); VA3-004/-005/-006 landed
fix+value-test together with the mechanism code-verified.

**Flake found while gating** (pre-existing, unrelated to this slice's changes):
`Priority2IntegrationTests.testBufferPoolAllocation` failed 2 of 5 release runs on
`allocationCount > 0` — the pool is process-shared, so when earlier tests have warmed the
matching size buckets, all three of the test's acquisitions are pure hits and zero fresh
allocations occur. Order-dependent assertion, not a product bug. Fixed to assert on request
activity (`hitCount + missCount >= 3`); five subsequent release-run results: 3 pre-fix
(2 fail), 2 post-fix (clean).

Fixed: **VA3-001** (jaccard rewritten dispatch-robust, one-threadgroup dispatch at both wrapper
sites, dimension-sweep regression), **VA3-003** (stable Minkowski uniform-flow degenerate flag;
the absolute 1e-8 cutoff became the honest `== 0` test, resolving the stable-kernel part of
VA3-022; mixed-tile + micro-scale pins), **VA3-004** (explicit `BatchBiasLayout` — shared `[N]`
vs per-batch `[batchSize×N]` — wired shader↔Swift; both layouts value-checked, bad lengths
throw), **VA3-005** (kernel takes `uint2 (dimension, numVectors)` + grid guard; 1500-vector
boundary test), **VA3-006** (`batchCosineSimilarity` finishes through the shared rescue trio;
new differential leg drives the live `BatchDistanceEngine` GPU branch across all ten adversarial
classes), **VA3-008** (both warp-select kernels write all `k_param` slots with sentinel padding),
**VA3-009** (all nine `threadgroup float … = 0.0f` initializers removed), **VA3-010**
(`DotProductKernel.selectPipeline` gained the `stride == dimension` guard; poison-padding
regression test).

**Liveness corrections found during execution** (the audit's original greps required
exact-quoted full names and missed prefix-named kernels):
* **VA3-008 was LIVE, not dead** — `WarpOptimizedSelectionKernel` loads the warp/batch select
  kernels and `FusedL2TopKKernel` uses it on the two-pass and chunked fallback paths (a short
  final chunk hits `candidateCount < k` exactly as diagnosed). Escalated to P1 and fixed.
* **`streaming_topk_*` are loaded by the deprecated-but-shipping `StreamingTopKKernel`** — not
  deletable without an API removal decision; struck from the Group F inventory.
* **VA3-010's MutualReachability half is structurally safe**: both
  `MutualReachabilityParams` construction sites pin `strideEmbed == d`, so dimension-only
  selection cannot currently misfire; the invariant is now documented on `selectPipeline`
  instead of guarded. DotProduct was genuinely exposed (public explicit-stride params) and got
  the guard.

**Group F executed** (~2,400 lines deleted): `CosineSimilarity.metal` removed wholesale; dead
kernels stripped from DistanceShaders (`minkowskiDistance`, SIMD trio), MinkowskiDistance
(`_single`, the stub `_fractional`), OptimizedMatrixOps (`fastNormalize`), BasicOperations
(`batchNormalize2D`, `squaredEuclideanDistance`, `vectorNorm`, `vectorSubtract`,
`elementwiseMultiply` — `vectorAdd`/`vectorScale`/`batchNormalize`/`normalizeVectors` verified
live/parity-tested and kept), ClusteringShaders (all eight legacy kernels incl. the broken
`tiled_kmeans_distance` and racy `combine_partial_centroids`; `compute_min_distances` verified
live and kept), QuantizationShaders (scalar/product pairs), L2Distance (the four specialized
variants), AttentionSimilarity (`768_to_64`, `batch_attention`), IVFCandidateBuilder (the pow2
Blelloch scan). `runtimeCompileShaderFiles` updated (27 files); `PipelineCacheKey` lost the
`cosineSimilarity` operation/constructor and derives every `l2Distance` key to the general
kernel; `commonKeys`/`embeddingModelKeys` now warm only real, live kernels (dot-product
specialized family). The doc-theater `OptimizedKernelsTests.swift` (asserted array counts,
"documented" the broken kernels) was deleted.

**Group D advance (pulled in because the deletions touched the same lines):** all eight
`PipelineRegistry` phantom operation names re-pointed to real kernels, deleted-key references
pruned, and a new `testBuiltInRegistryKeysResolveToRealFunctions` resolves every key of every
built-in registry against both libraries — the registry can no longer accumulate phantoms
silently. (`tiledTransposeInPlace` — VA3-021 — remains open for the next slice.)

---

## Remediation slice 2 (2026-08-16, owner-directed: VA3-002) — EXECUTED

**Verification: debug 1561 tests / 0 failures (21 skipped); release 1561 tests / 0 failures
(21 skipped).** Baseline was 1558/0 on both legs; the +3 are the new regression suite, which
passes under both the debug metallib and the release runtime combined-source compile.

**VA3-002 (red→green).** `TreeReductionDispatchRobustnessTests` (Hardening/) written before the
fix: engine-level `cosineDistance`/`dotProduct` sweeps (`Metal4ComputeEngine(context:)` with the
default nil decision engine → GPU for dim > 16) over non-pow2 dims {17, 20, 100, 200, 255} +
pow2 controls {32, 64, 128, 256} + large {300, 1000} against `AccelerateFallback`, plus a
kernel-direct `euclideanDistance` leg with explicit threadgroup widths
{17, 100, 255, 384, 512, 64, 256} — the engine's width-1 euclidean dispatch cannot expose the
tree (left untouched; VA3-024). The red run confirmed the orphaning mechanism exactly where
predicted: cosine diverged at all five non-pow2 dims (dim=20: GPU 1.105 vs CPU 0.922) with every
pow2 control agreeing, and euclidean under-counted deterministically (ones-vector at tg=100
returned √640 for a 1000-dim input; tg=512 returned √512 through the unclamped shared array).
Fix: all three kernels accumulate with `lanes = min(tgSize, 256)` striding — lanes-guarded, so
threads past the clamp gather nothing the tree would drop — and reduce through the file's own
`va_tg_reduce_add`; `cosineDistance` runs three sequential reductions through one scratch array
(the helper's trailing barrier makes back-to-back reuse safe) and its VA2-008 rescue/finalize
block is untouched. Kernel signatures and buffer indices unchanged; no header edits.

**VA3-031 (P1, was LIVE — exposed by the red run, fixed in-slice).** The dotProduct engine leg
did not red with the orphaning signature: it returned 0.0 / identical stale bytes at *every*
dim, pow2 controls included — the result buffer was never written. Attribution:
`getPipeline(functionName:)` on both `PipelineCache` and `ArchivePipelineCache` wraps the
literal name in `PipelineCacheKey(operation:)`, and `PipelineCacheKey.functionName`'s
`case "dotProduct"` rewrote it to `dot_product_kernel` — the *batch* kernel, dispatched with
single-pair bindings. Its params struct reads zeros past the engine's 4-byte `setBytes`
dimension, the grid guard exits, and callers receive whatever bytes the pooled result buffer
already held. The single-pair `dotProduct` kernel therefore had **no dispatch site at all**
(this ledger's LIVE-cond call for its tree was optimistic — the path diverted one layer above
the kernel), and `commonKeys`' `.distance("dotProduct")` entry, grouped under "Basic operations"
beside euclidean/cosine, silently warmed the batch kernel it never named. Invisible because
every pre-existing engine dotProduct test uses dim ≤ 16 — CPU-routed by the nil-engine fallback
(the routing observation, materialized). `euclideanDistance`/`cosineDistance` were already
identity-mapped during the VA2-006 fix; dotProduct was the one engine name still poisoned.
Fix: the keyed batch family's operation string is now `"dot_product"` (the
`.dotProduct(dimension:)` factory still resolves to `dot_product{_384,_512,_768,_1536}_kernel`),
the three literal single-pair names identity-map, and the switch carries a comment banning
rewriting cases that shadow kernel names. Three stale test assertions updated to the factory
key / new operation string (PipelineRegistryTests, PipelineCacheKeyTests ×2).

Gates beyond the new suite: guard suites (DifferentialKernelVsCPU, NormalizationParity,
PreambleParity, ShaderLibraryCompleteness) green; cache-key blast radius (PipelineCacheKey,
PipelineCache, PipelineHarvester, HarvestManifest, Metal4ShaderCompiler,
Metal4CompilerConfiguration, PipelineRegistry, WarmupManager, ArchivePipelineCache,
BinaryArchiveManager, Metal4ComputeEngine) green. Out of scope, recorded:
`getPipeline(functionName: "vectorMultiply")` (Metal4ComputeEngine matrix path) names a kernel
that exists in no library — pre-existing VA3-021-class phantom, fallback-shielded; VA3-013/-014
(same defect family as VA3-002) remain open pending owner scope call.

---

## Remediation slice 3 (2026-08-16, owner-directed: VA3-013 + VA3-014 — Group A complete) — EXECUTED

**Verification: debug 1567 tests / 0 failures (21 skipped); release 1567 tests / 0 failures
(21 skipped).** Baseline 1561/0 both legs; the +6 are `LatentReductionGeometryTests`
(Hardening/), red-first. Both findings are LATENT — correct only under the hosts' exact
dispatch geometry — so the red legs drive the kernels directly with the geometries the hosts
never use, while the live paths (engine manhattan/chebyshev at dims > 64, the LSE and
statistics wrappers' own geometries) ride along as must-stay-green controls: green before AND
after, proving both the shielding claim and the rewrite.

**Red evidence (mechanism confirmed per kernel):**
- `computeBasicStatistics` tg=100: count=64 of 100 (36 lanes orphaned), sum 3104 vs 4950,
  max 97 vs 99; tg=17: count=95. Pow2 controls clean — the host's "Use power-of-2 threadgroup
  size for correct parallel reduction" was the mask working as designed while the kernel's own
  comment claimed robustness.
- `computeHigherMoments` tg=100: M3 = 15,018,704 vs the exact 24,502,500 (every term and
  partial sum < 2²⁵, so M3 admits exact FP32 comparison) — 39% of the cubes never merged.
- `logsumexp_reduce_pass1_kernel` tsize=100: partialMax 9.7 vs 9.9 (the maximum's merged
  subtree orphaned), sumExp 7.28 vs 10.51; tsize=300 corrupted through the unclamped
  `sharedMax[lid]` writes.
- `logsumexp_reduce_pass2_kernel` tsize=64: output −inf — the guardless fixed-128 tree folded
  uninitialized threadgroup memory into globalMax, making every exp(partialMax − globalMax)
  underflow to zero, exactly as diagnosed; numGroups=300 at 256 threads: −94.45 vs 50.0,
  matching the hand-computed log(256) − 100 for the silently dropped dominant partial at
  index 299.
- `manhattanDistance` tg=100×1: 274.5 vs 653.4 (the hardcoded-256 stride leaves every element
  with i % 256 ≥ 100 uncovered AND thread 0 sums min(256, dim) shared slots, 156 of them never
  written); `chebyshevDistance`: a 7.5 spike at index 255 invisible (0.247 returned). The
  1024-dim 4×256 leg — every group racing its own `atomic_store` into result[0] — happened to
  flake-PASS the red run (group 0's correct store landed last): the nondeterminism itself,
  demonstrated. Post-fix it is deterministically correct.

**Fixes (buffer indices unchanged everywhere; attribute parameters changed where needed, which
is binding-neutral — the VA3-001 jaccard-rewrite precedent):**
- manhattan/chebyshev: rewritten on the jaccard template in the same file — ONE-threadgroup
  contract documented, `lanes = min(tgSize, 256)` clamp, lanes-strided accumulation, no early
  return (all threads reach the barrier), thread-0 serial fold over `lanes` (not
  `min(256, dimension)`), plain store replacing the per-group racing `atomic_store`. Extra
  threadgroups now redundantly compute the identical full result.
- LSE pass1: lanes clamp + lane-guarded loads; the grid-stride re-indexed to
  `tgid·lanes + lid` stepping `lanes·numThreadgroups` — element coverage is bit-identical
  under the host's fixed 256-wide dispatch; both trees now fixed-128 start with the
  `lid + s < lanes` ragged-tail guard. The uniform all-−inf early-out is retained.
- LSE pass2: gained `tsize [[threads_per_threadgroup]]`; loads stride the FULL partial range
  (`for i = lid; i < numGroups; i += lanes`), which also removes the silent 256-partial cap;
  both trees carry the ragged-tail guard. ±inf early-outs retained (uniform).
- Statistics kernels: trees start at the fixed pow2 `MAX_TG_SIZE/2` and the
  `tid + stride < tgSize` guard now does what the deleted comment claimed; accumulation is
  unchanged (every thread participates; tgSize ≤ MAX_TG_SIZE = 1024 is Metal's device
  maximum). `StatisticsKernel`'s pow2-forcing host workaround is no longer load-bearing
  (left in place — a reasonable width choice regardless).

**Also recorded (not fixed — VA3-021 class, third instance):** the engine's batch manhattan
path requests `getPipeline(functionName: "batchManhattanDistance")`, a kernel that exists in
no shader file — the path throws when reached (loud for direct engine users, fallback-rescued
behind the provider). Joins `tiledTransposeInPlace` and `vectorMultiply` as evidence for
VA3-021's proposed completeness-test extension over Swift name literals.

Gates beyond the new suite: the slice-1/-2 regression suites stay green; the kernel-wrapper
family (LogSumExp, Statistics, QuantizationStatistics, ParallelReduction, L2Distance,
L2Normalization, Minkowski, MutualReachability, TopKSelection, WarpOptimizedSelection, and the
rest of the KernelTests classes), DifferentialKernelVsCPU, NormalizationParity, PreambleParity,
ShaderLibraryCompleteness, Metal4ComputeEngine, and NumericalStability suites all green; full
debug + release above.

---

## Remediation slice 4 (2026-08-16, owner-directed: adversarial meta-review of slices 1–3) — EXECUTED

**Verification: debug 1577 tests / 0 failures (21 skipped); release 1577 tests / 0 failures
(21 skipped).** Baseline 1567; the +10 are the meta-review's regression and coverage legs.

**Method.** The audit methodology turned on the epic itself: three independent adversarial
reviewers over the full uncommitted diff (shader corpus / Swift infra / test honesty), each
armed with the masking-pattern checklist and required to map every robustness claim the fixes
added to a covering test leg. Every UNCOVERED claim was then closed empirically.

**Three live defects found in — or armed by — the epic, all fixed red→green:**

- **VA3-032 (P1, was LIVE): `BatchDistanceEngine.batchDotProduct`/`batchManhattanDistance` GPU
  branches dispatched phantom kernels.** `getPipeline(functionName: "batchDotProduct")` /
  `"batchManhattanDistance"` name kernels that have never existed in any library. The branches
  armed at gpuThreshold (1000) candidates with no decision engine — and the epic's own VA2-003
  k-gate exemption armed the engine-attached path too (`.dotProduct`/`.manhattanDistance` are
  exempt, GPUDecisionEngine.swift:423) — throwing shaderNotFound on public APIs where callers
  expected values. Red: both APIs threw at 1000×64 and under explicit `useGPU: true`. Fix:
  GPU-routed requests run the SIMD paths until a real batch kernel is wired
  (`dot_product_kernel` is the natural candidate for dot); the dead GPU encoders and their
  phantom literals are deleted.
- **VA3-033 (P2, was LIVE): `batchCosineSimilarity`'s sub-simdThreshold CPU leg diverged from
  the other two legs of the same API.** The inline naive formula (NaN swallowed to 0 by
  `queryNorm > 0`, product-denominator overflow → similarity 0, no clamp) survived
  VA2-008/-009 and VA3-006, which upgraded only the SIMD and GPU legs — a value discontinuity
  at the batch-size-100 boundary. Red: NaN-poisoned query → 0.0 at 99 candidates vs NaN at
  100; 1e19 components → 0.0/NaN garbage at 99 vs correct at 100. Fix: the leg delegates to
  `AccelerateFallback.batchCosineSimilarity` (rescued core); all routings agree.
- **VA3-034 (P2, was LIVE-cond): `warp_select_small_k_*` over-cap k resurrected the VA3-008
  stale-bytes symptom.** `if (k_param > K4_MAX_K) return;` left every output slot unwritten
  for k > 32 through the public `encodeWarp` (`selectTopK` routes such k elsewhere; the encode
  API bypasses that). Red: k=33 read back 0xDEADBEEF poison in every slot. Fix: over-cap
  requests sentinel-fill all k_param slots (index 0xFFFFFFFF, ±Inf per mode) before returning
  — an honest "no results".

**Corrections to this ledger's own record (caught by the Swift reviewer):** slice 2 located
`"vectorMultiply"` in "Metal4ComputeEngine matrix path, fallback-shielded" — it is
`VectorCoreIntegration.swift`'s public `multiply(_:_:)` and has always thrown (nothing shields
it); slice 3 called `"batchManhattanDistance"` "the engine's batch manhattan path …
fallback-rescued behind the provider" — it is `BatchDistanceEngine`'s, nothing rescued it, and
it is now fixed under VA3-032. Lesson: phantom reachability claims must be re-derived whenever
routing gates change — the VA2-003 exemption invalidated the k-gate shielding argument the day
it landed.

**The epic's own failure mode, found in the epic — untested robustness claims. All closed with
new legs that passed on first run (the fixes were correct; the claims were unproven):**
- LSE pass1's multi-threadgroup indexing (rewritten in slice 3) had ZERO value assertions at
  any count producing more than one group; a one-token gridSize regression would have
  double-counted the array on the live path under all prior tests. Now: end-to-end `reduce`
  at 257/1 000/100 000 vs a Double reference, kernel-direct multi-group legs at
  (100×3, 256×4, 64×5), and an exact all-ones structural leg.
- cosineDistance/dotProduct's >256-lanes clamp had no covering leg (euclidean only), and their
  non-pow2 coverage was routing-contingent on the unpinned `dimension > 16` fallback — the
  VA3-024 follow-up would have silently degraded those sweeps to CPU-vs-CPU. Now:
  routing-independent kernel-direct sweeps at widths {17, 100, 255, 384, 512, 64, 256}, result
  buffers poisoned pre-dispatch, and `where`-filters guarded by >256-retention asserts.
- Statistics' "ANY tgSize up to MAX_TG_SIZE" never executed its fixed-512/256 strides. Now:
  widths {300, 1000, 1024} over 2048-element exact-integer data, with retention asserts.
- LSE pass2's "ANY threadgroup width" never fired its ragged-tail guard (pow2 widths only).
  Now: tsize {17, 100} legs.
- The SoA lane-major cosine rescue (a third hand-rolled copy of the rescue) never saw its
  underflow-collapse trigger. Now: `.tiny`/`.mixedScale` classes in the SoA differential.
- VA3-005's regression test pinned the numVectors plumbing, not the OOB write. Now: a
  poisoned-tail leg under the wrapper's exact ceil-rounded dispatch.
- VA3-010's stride-guard test padded both strides at once (a one-conjunct regression passed).
  Now: mixed single-side legs, both directions.
- jaccard had no kernel-direct geometry coverage (wrapper geometry only). Now: swept beside
  manhattan/chebyshev in the geometry test.
- The Priority2 buffer-pool assertion was cumulative over a process-shared pool (vacuous in
  full-suite runs); it now asserts the delta from a pre-call snapshot.

**VA3-031 class closure (the ledger's own proposal, implemented in
ShaderLibraryCompletenessTests):** `testOperationDerivationNeverShadowsKernelNames` round-trips
every corpus kernel name through `PipelineCacheKey(operation:).functionName` (allowlisted:
`batch_select_k_nearest_*` — the two pre-existing reverse-shadow names, unarmed because they
are loaded via `makeFunction` directly); `testSwiftLiteralPipelineRequestsResolveToRealKernels`
resolves every `getPipeline(functionName:)`/`getPipelineState(functionName:)` literal in
Sources against the corpus (allowlisted: `vectorMultiply`, VA3-021 open). Growing either
allowlist silently is now impossible.

**Recorded, not fixed (filed under existing items):** VA3-024 addendum — the engine's
batch/fused distance dispatches use 16×16 threadgroups against kernels that consume only
`id.x` (16× redundant compute; `BatchDistanceEngine` dispatches the same kernels at 256×1
correctly). VA3-031 residual — `Metal4ShaderCompiler.getPipelineState(functionName:)` caches
literal-resolved pipelines under `PipelineCacheKey(operation:)` slots (zero production
callers). VA3-011 adjacent — `BatchMatrixKernel.encode` (raw-buffer API) cannot validate bias
length against `biasLayout`. Provider telemetry — the `any DistanceMetric` delegation branch
increments no counter. `PipelineCacheKey.quantized` derives only phantom `_q4/_q8/_binary/_pq`
names since Group F (zero callers; delete on next touch, along with its derivation-test
assertion). `AccelerateFallback` ragged-pair asymmetry (euclidean → +Inf, cosine → NaN;
provider-unreachable). VA3-012 re-affirmed open: this epic edited DistanceShaders and
deliberately did not centralize EPSILON — that fix is its own cross-cutting slice.

**Reviewer verdicts:** shader corpus — no Critical; every rewrite survives hand re-derivation
at lanes {1, 2, 17, 100, 255, 256, 300, 1024} and both build paths. Swift infra — no Critical
beyond VA3-032/-033; the VA3-031 split verified complete against all 142 kernel names;
persisted archives structurally cannot serve a wrong pipeline. Test honesty — slices 1–3's
fixes confirmed pinned with exact-valued assertions; gaps as listed above;
`OptimizedKernelsTests`' deletion confirmed lossless.

## Remediation slice 5 (2026-08-24, owner-directed: VA3-007 — max-norm becomes explicit opt-in, branch kept) — EXECUTED

**Verification: debug 1583 tests / 0 failures (21 skipped); release 1583 tests / 0 failures
(21 skipped).** This slice adds exactly 5 tests (`Hardening/MinkowskiLargePPolicyTests`,
red-first). Totals are +6 over the slice-4 record (1577); the extra +1 predates this slice
and is unattributed — the slice-4 number was a point-in-time record, and the gate here is
0 failures with this slice's 5 present in both configs.

**Red confirmed the mechanism exactly** (uniform-difference geometry, D = 768, where the
substitution error D^(1/p) is fully realized): p = 50 with a default config returned 1.0
(the max-norm) where true L50 = 768^(1/50) ≈ 1.1421; p = 11 with explicit
`useStableComputation: false` returned 1.0 where true L11 = 768^(1/11) ≈ 1.8294 — an 83%
error behind a flag that read as a numerics choice.

**Fix (owner decision: option B — keep the branch, gate it).**
- `minkowski_distance_batch` takes an explicit `chebyshev` opt-in at `buffer(7)`; the
  `is_large_p = (p > 10)` inference is gone, and the flag is authoritative over the
  p-derived manhattan/euclidean fast paths. Without the flag, large p runs the exact
  `safe_pow` general path.
- `Metal4MinkowskiConfig` gained `chebyshevApproximation` (default false). Auto-routing now
  sends ALL p > 10 to the stable kernel (the (10, 30] upper cap removed — unblocked by the
  VA3-003 fix); `useStableComputation` is a pure numerics choice on both legs. An
  (inconsistent) `chebyshevApproximation: true` + `useStableComputation: true` resolves in
  favor of the approximation, documented on the init.
- `isChebyshev`/`metricName` reflect the flag, not `p > 30` — a bare
  `Metal4MinkowskiConfig(p: 100)` now honestly reports and computes "Minkowski (L100.0)".
- The `.chebyshev` preset and all four provider `.chebyshev` sites
  (`MinkowskiKernelDistanceProvider` single + batch, `UniversalKernelDistanceProvider`
  single + batch) carry the flag; their exact-L∞ semantics are unchanged.
- Also reconciles the GPU kernel with `Metal4ComputeEngine.minkowskiDistance`'s CPU path,
  which always computed true Lp — the two surfaces previously disagreed for p > 30.

**Test-honesty note:** the preset/provider controls include a *discriminating*
uniform-difference leg — exact L∞ = 1.0 vs true L100 = 768^(1/100) ≈ 1.069 — because with a
single dominant coordinate, true L100 is numerically ≈ L∞ in FP32 and a dropped opt-in
would pass. A provider that forgets `chebyshevApproximation` goes red on that leg, not
green-by-luck.

With this slice, **every P1 in the AUDIT-3 ledger is FIXED**. Remaining open: VA3-011
(silent capability caps), VA3-012 (EPSILON dual-compile drift), VA3-021 (phantom-literal
completeness extension), and the Group B/C/E/G remainder.

---

## Remediation slice 6 (2026-08-24, owner-directed: VA3-012 — EPSILON dual-compile drift) — EXECUTED

**Verification: debug 1586 tests / 0 failures (21 skipped); release 1586 tests / 0 failures
(21 skipped).** Baseline 1583 + 3 red-first tests (`Hardening/EpsilonCompileParityTests`).

**Red confirmed the mechanism exactly, as a live two-library divergence:** the same
`uniformHistogram` kernel, driven with a bin range of 5e-8 (inside the (1e-8, 1e-7] drift
window), returned **[16, 0]** from the debug metallib (header `VA_EPSILON = 1e-7` → degenerate
branch) and **[8, 8]** from the runtime combined build (EPSILON silently redefined to 1e-8 by
DistanceShaders upstream) — same source, two numerical behaviors, demonstrated in one process.
The structural leg was red on both counts (DistanceShaders defines `VA_EPSILON`; the preamble
lacked it); the non-degenerate control was green throughout.

**Fix — single epsilon authority, no token surgery:**
- `DistanceShaders.metal`: the file-local `#define VA_EPSILON 1e-8f` shadow became
  `#define VA_JACCARD_UNION_EPSILON 1e-8f` (value preserved — it is file-local on BOTH build
  paths; renaming, not retuning). This kills the poison at its source: no shared-name macro
  shadow exists to be rewritten into a global redefinition.
- `KernelContext.swift`: the preamble now defines `#ifndef VA_EPSILON / #define VA_EPSILON
  1e-7f` under the shaders' real symbol name; the `VA_EPSILON → EPSILON` token rewrite and
  all four whitespace-fragile `constant float EPSILON = …;` exact-string strips are deleted
  (grep confirmed zero remaining targets — they were dead since the per-file declarations
  were removed). `VA_INVALID_INDEX → 0xFFFFFFFF` literal substitution retained (numerically
  identity-preserving, parity-mapped in tests).
- `Metal4Common.h`: `constant float VA_EPSILON = 1e-7f;` gained the `#ifndef` guard,
  matching the established VA_NORM_* pattern (the preamble macro wins if both are ever seen).
- `PreambleParityTests`: the replacement-era mapping `preambleConsts["VA_EPSILON"] =
  preambleConsts["EPSILON"]` was itself a latent trap — with the EPSILON macro gone,
  Swift's `dict[k] = nil` would have *removed* the parsed VA_EPSILON entry and silently
  skipped the parity check. Replaced with a hard `XCTAssertNotNil` on the preamble's own
  `VA_EPSILON`; the hardcoded provided-symbol allowances shrank to the two honest ones.

**Effect:** LearnedDistance normalize gates, NeuralQuantization `computeScale`/normalize
floors, and StatisticsShaders' histogram range gate now run at the header's 1e-7 on BOTH
build paths (previously 1e-8 in every release build). `EpsilonCompileParityTests` pins the
class permanently: the behavioral leg re-reds on any future two-library epsilon divergence,
and `testSingleEpsilonAuthority` re-reds if any compiled file ever (re)defines
`VA_EPSILON`/`EPSILON`.

---

## Remediation slice 7 (2026-08-28, owner-directed: VA3-011 — silent capability caps) — EXECUTED

**Verification: debug 1598 tests / 0 failures (21 skipped); release 1598 tests / 0 failures
(21 skipped).** Baseline 1586 + 12 red-first tests (`Hardening/CapabilityCapPolicyTests`).
Red run: 8 over-cap legs failed on mechanism (silent acceptance / stale poison bytes), 3
at-cap controls green throughout.

**Two defects found beyond the audit's inventory:**
- **TopKParameters.init silently clamped `k = min(k, 128)`** (both inits) — a top-200
  request became top-128 with no signal, before any kernel even ran. The audit's own class,
  living in the params type. Fixed: the inits preserve the request; over-cap k reads back as
  an all-sentinel row from the kernel; `select()` still throws.
- **The new host guards ARMED the dormant AUDIT-2 "throw-mid-encode" anchor**: a throw from
  inside `executeAndWait`'s open encoder left it un-ended, and Metal API validation ABORTED
  the process ("Command encoder released without endEncoding") — observed live as a signal-6
  crash of the new suite. Fixed in all three Metal4Context execution methods (`execute`,
  `executeAndWait`, `executeBlitAndWait`): a throwing closure now gets its encoder ended and
  the error propagates; nothing commits. Pinned by
  `testThrowingEncodeClosurePropagatesWithoutCrash`. (The epic's own lesson recurring: a
  correct gate change re-arms dormant defects downstream — re-derive reachability.)

**Host-side throws (Policy: caps are errors, never silent):**
- Attention: `encodeAttentionSimilarity` (the single dispatch choke) throws for
  headDimension > 256 (single-head) / > 64 (multi-head).
- Learned: `computeL2`/`computeCosine` throw for projectedDimension > 256 (guard placed
  BEFORE encoder creation — no leak on the error path).
- Neural: `validateCapability` (latentDimension ≤ 128, `maxLatentDimension`) at ALL FOUR
  config entry points — loadWeights ×3 (URL / Data / arrays — the audit knew of two;
  anchor-count discipline found the third) + createRandomWeights.
- IVF: `buildCandidates` throws for nprobe > `maxNprobe` (64), covering fused + three-pass.

**Kernel-side sentinel/NaN fills (never stale bytes):**
- `topk_select_batch_kernel` K > 128: publishes an all-sentinel row (±INF by mode,
  UINT_MAX) instead of the bare return that resurrected the VA3-008 stale-bytes symptom
  through raw `encode()`.
- `fused_l2_topk` D > 768 / tgs > 256: tid-strided sentinel-fill of the query's K slots
  (params type guards D host-side; this covers direct dispatch — driven kernel-direct in
  the test).
- NeuralQuantization: generic quantize kernel NaN-scales + zeroes codes over-cap (the old
  truncation also used the clamped value as the codes row stride — layout scramble);
  generic dequantize-decode NaN-fills the output row; all four 2d-tg dequantize variants'
  silent `latentDim4 > 32` returns now NaN-fill cooperatively (uniform condition, before
  all barriers, idempotent across Y-tiles). The 2d-tg and generic fills are
  defense-in-depth behind the tested host guards — verified by inspection, marked as such
  in-kernel (explicit unverified contract per claim discipline).
- Attention/Learned truncation clamps left in place as belt; unreachable over-cap through
  every dispatch path (all funnel through the throwing chokes).

With this slice, Group C's cap family is closed. Remaining open: VA3-021 (phantom-literal
completeness extension) and the Group B/E/G remainder.

---

## Remediation slice 8 (2026-09-04, owner-directed: VA3-021 — the phantom-literal class, closed) — EXECUTED

**Verification: debug 1590 tests / 0 failures (11 skipped); release 1590 tests / 0 failures
(11 skipped).** Prior baseline 1598 + 2 red-first tests − 10 deleted forever-skipping tests;
the standing skip count dropped 21 → 11 — both deltas exact.

**The class-closer paid for itself before the fix**: extending
`ShaderLibraryCompletenessTests` to resolve every `makeFunction(name:)` literal (part 3 of
the VA3-031/VA3-021 completeness family; parts 1–2 covered PipelineCacheKey derivations and
`getPipeline(functionName:)` literals) went red with **EIGHT phantoms** where the audit knew
of one:

- `tiledTransposeInPlace` (the known instance) — permanently-nil pipeline; `inPlace: true`
  silently ran the out-of-place kernel, so a caller who then read the INPUT buffer got
  untransposed data with no signal. Red-first behavioral leg:
  `KernelTests.testInPlaceRequestThrowsInsteadOfSilentDowngrade`.
- `tiledMatrixMultiply_512/768/1536` — dead specialized selection in MatrixMultiplyKernel;
  every K fell through to the generic kernel anyway (perf-fiction, correctness-neutral).
- `batchMatrixVector` — a pipeline property that was never even read.
- `neural_encode_tiled_kernel`, `neural_encode_quantize_tiled_kernel`,
  `neural_encode_quantize_tiled_v2_kernel` — three public APIs
  (`encodeTiledEncode`/`Quantize`/`QuantizeV2`) that could only throw "not available", with
  **ten tests that had skipped on `isTiled*Available` guards since the day they were
  written** — roughly half the suite's standing skip count, reading as coverage.

**Fix (Group F discipline — delete the fiction, keep the fact):** all eight optional loads
deleted. Transpose: `inPlace` parameter removed from `encode`, `execute` throws for
`config.inPlace` (field kept, documented). MatrixMultiply: specialized properties +
`selectPipeline` deleted, generic pipeline used directly (behavior identical — it always
was). MatrixVector: unread `batchPipeline` deleted. Neural: the tiled V1/V2 API trio,
`isTiled*Available` accessors, `TiledEncodeParameters` (no other consumers), and the ten
skipping tests deleted (~880 test lines); `encodeTiledV3` (neural_encode_pass1 /
neural_quantize_pass2 — kernels that exist) is the shipping tiled path and survives
untouched. Recorded, not fixed: `encodeTiledV3` has no dedicated test coverage — noted as a
coverage gap, out of this slice's scope.

**Class closed:** `testSwiftMakeFunctionLiteralsResolveToRealKernels` carries NO allowlist —
an optional load that never resolves is dead code by definition. Together with parts 1–2,
every route by which Swift can name a kernel (cache-key derivation, getPipeline literal,
makeFunction literal) is now mechanically checked against the corpus.

---

## Remediation slice 9 (2026-09-05, owner-directed: VA3-015 while VectorCore Top-K work proceeds) — EXECUTED

**Scope:** correlation finalization, four histogram finite-value gates, and LSE/softmax
infinity equality branches. Fast math remains enabled. No changes to VectorCore,
dependency resolution, VA3-016 NaN/tie policy, Minkowski policy, or accumulator-overflow rescue.

**Observed red:** `FastMathPolicyTests.testCorrelationFiniteAccumulatorsAtExtremeScales`
returned zero instead of −1 for perfectly anticorrelated datasets at scales `1e10`,
`1e18`, `6e18`, and `1e-18`: **16 assertion failures across plugin and runtime libraries**.
The individual M2 and covariance accumulators remained representable. Ordinary-scale,
constant-data, histogram, and LSE/softmax controls passed. Covariance checks use an
independent Double reference with cancellation error bounded by the input product scale.

**Fix:** compute the two norms separately; express the existing product `FLT_MIN` floor
as a division, with explicit finite zero handling; compute correlation with two
`precise::divide` calls. Independent review caught an intermediate-underflow regression
in the initial fixed-order divide: `[1e18,-1e18,1e-10,-1e-10]` versus
`[0,0,1e-18,-1e-18]` returned zero instead of approximately `1e-28`. The additional
relative-tolerance regression failed four assertions on both libraries. Dividing by the
**smaller finite norm first** fixes it; the intermediate is bounded by the larger norm
via Cauchy-Schwarz. Both dataset orders and both symmetric output entries are tested.
Nonfinite norms retain the quotient/clamp path and original division operand order.

**Preventive alignment, not reproduced defects:** all three histogram variants already
excluded nonfinite inputs on this toolchain, and the tested infinity branches already
worked. Their `isfinite(value)` gates now use `fabs(value) <= FLT_MAX`; LSE/softmax
infinity equality tests now use strict `> FLT_MAX` / `< -FLT_MAX` comparisons. Coverage
includes degenerate/ordinary histogram ranges, outliers on/off, finite endpoints, both
row LSE kernels, both reduction passes (including an empty partial group), and both
softmax kernels. Existing all−Inf and repeated +Inf softmax behavior is retained.
The two reduction inclusion gates involving `> -INFINITY` remain with VA3-016.

**Coverage and gates:** nine new tests execute the actual plugin and runtime libraries
in debug, and the runtime library in release. Missing required debug metallib/functions
fail instead of silently skipping; only absence of a Metal device skips. Direct dispatch
prevents CPU fallback masking. Correlation/LSE/softmax outputs are poisoned; atomic
histograms start from nonzero known counts. Final targeted gate: **53 tests / 0 failures**.
Full debug: **1599 tests / 0 failures / 11 skipped** (252.543 seconds); full release:
**1599 / 0 / 11** (50.923 seconds). Baseline 1590 + nine new tests; skips unchanged.
Independent review's division-order finding was reproduced, fixed, and re-reviewed.
Environment: Apple M3 Max, Metal 4, Apple metal compiler 32023.883. No performance
improvement is claimed; this slice changes numerical correctness while retaining fast math.
All changes remain uncommitted; the owner controls commits and release.

---

## Remediation slice 10 (2026-09-06, owner-approved: VA3-016 LSE/basic-statistics NaN policy) — EXECUTED

**Approved contract:** any input NaN dominates positive or negative infinity in scalar/vector
row LSE and full LSE reduction. Basic-statistics mean, M2/variance, min, max, and sum become
NaN; count includes every input. Derived standard deviation and range also remain NaN.
Histograms retain their existing nonfinite exclusion. Top-K is a separate part of VA3-016
and remains open pending VectorCore release/integration. Softmax policy, higher moments,
quantiles, correlation, and general overflow rescue are outside this slice.

**Red:** the new `NaNReductionPolicyTests` suite reproduced row/reduction NaN loss or
infinity precedence, numeric statistics extrema despite NaN input, and public basic-statistics
rejection of NaNs (406 failed assertions/errors in the initial eight-test run, including that
validation throw). Finite/empty basic-statistics kernel controls passed. Pass 1 also
manufactured NaN partial sums from legitimate +Inf input through `Inf - Inf`.

**Shader changes:** classify input NaNs through integer bit patterns, independently of
fast-math floating-point assumptions. Row kernels check before infinity branches. Each LSE
reduction carries a uint NaN flag through the same lane-clamped, ragged-tail reduction as
the maximum; all early decisions remain uniform after barriers. Pass 1 writes `(NaN, NaN)`
for poisoned groups, `(-Inf, 0)` for empty/all-negative-infinity groups, and symbolic
`(+Inf, 1)` for positive-infinity groups without NaN. Pass 2 checks both partial fields for
NaN before its infinity branches. Finite inclusion gates use `>= -FLT_MAX`, retaining the
negative finite endpoint. `StatsAggregate` carries a uint flag through count-preserving
merges and writes five NaNs at finalization; its 1024-entry shared array is now 28 KiB.

**Public surface:** `computeBasicStatistics` and basic-only `computeStatistics` configurations
now accept NaNs, including NaN mixed with infinity, instead of throwing. Singleton spread
fields and the variance-to-standard-deviation conversion preserve NaN. Empty input and
infinity without NaN still throw. Requests for higher moments, quantiles, and correlation
retain finite-only validation. The former `StatisticsKernelTests.testNaNInputThrows` now
checks the approved propagation contract. Public LSE documentation states NaN precedence.

**Coverage/review:** nine new direct-GPU/public-API tests cover NaN signs/payloads (including
a signaling encoding), every row position, singleton inputs, inactive lanes, multiple groups,
grid-stride positions, ragged widths, widths above the LSE shared-lane cap, and statistics
widths through 1024. Controls include finite endpoints, all ±Inf, constant finite data,
empty kernel input, and unchanged public validation. Debug exercises plugin and runtime
libraries; release exercises the runtime library. Independent review found no blocking
issues; its suggested two-pass finite-endpoint control was added. Targeted: **74 / 0**.
Full debug: **1608 tests / 0 failures / 11 skipped** (254.639 seconds). Full release:
**1608 / 0 / 11** (52.831 seconds). Baseline 1599 + nine new tests; the renamed existing
NaN-validation test does not change the count. No dependency updates or commits were made.
VA3-016 remains **PARTIAL**: LSE/basic-statistics policy is complete; GPU Top-K is pending.


## Remediation slice 11 (2026-09-06, owner-directed: VectorCore 0.3.3 upgrade) — EXECUTED

**Dependency:** raised the manifest minimum from 0.3.2 to 0.3.3 and resolved the corrected
`v0.3.3` tag at `fca4b602383589c46b627d8a0de2b6b2a68cd07d`. MetalCompilerPlugin remains
pinned at 0.1.6. Updated README dependency examples and the provider's obsolete NaN
behavior documentation. No provider algorithm or shader changes are part of this slice.

**Contract adoption:** three new `TopKSelectionAdoptionTests` use literal expected results
for numeric/NaN ordering in both directions, signed-zero and equal-value index ties,
heap admission/eviction with early NaNs, NaN tail fill, and pointer IDs that differ from
original positions. The existing maximizing adapter negates scores before and after CPU
selection; NaNs remain last under the new comparator without changing that adapter.

**Red/green evidence:** all three new tests failed against 0.3.2 (18 failed assertions;
the eight existing adoption tests passed). Against 0.3.3, the adoption, routing-provenance,
and GPU/CPU differential suites pass: **21 tests / 0 failures**. Full debug: **1611 / 0 failures / 11 skipped**
(254.169 seconds); full release: **1611 / 0 / 11** (52.300 seconds). Both commands
exited successfully. Baseline 1608 plus three new adoption tests.
Logs: `/private/tmp/vectorcore-033-upgrade/` (`red.log`, `resolve.log`, `targeted.log`,
`debug-full.log`, `release-full.log`).

**Remaining scope:** VA3-016 stays **PARTIAL**. The CPU contract is integrated; GPU Top-K
still needs NaN-last comparisons and original-index ties for membership and output order.
This upgrade does not establish GPU parity for NaNs. All changes remain uncommitted.


## Remediation slice 12 (2026-09-06–07, owner-directed: VA3-016 GPU Top-K contract) — EXECUTED

**Contract:** numeric values precede NaNs in both selection directions; numeric ties
(including signed zero) and NaNs prefer smaller original indices. Invalid-index padding
follows all real candidates, including NaNs. Admission, eviction, sorting, and merging
share the same order. Unsorted batch output preserves membership without promising order.

**Red evidence:** nine new `TopKNaNPolicyTests` failed before shader edits: **432 failed
assertions, zero unexpected failures**. Failures covered batch heap tie eviction, NaN
admission/output across warp and general selectors, streaming, sorted chunk merge, fused
selection, and IVF selection. Empty warp rows retained poisoned output bytes; fused
selection reused an infinity winner because removal changed its distance but retained
its valid index. Finite ordinary-score controls passed. Subnormal fixtures failed in both
libraries. A first implementation using integer NaN classification plus floating numeric
comparisons reduced failures to 12, all subnormal ordering: comparisons flushed scores
to zero. This is why the final comparator uses integer order keys for numeric values too.

**Implementation:** `va_topk_is_better` in `Metal4Common.h`, mirrored in the runtime
preamble, classifies NaNs by exponent/mantissa bits, normalizes signed zeros, and maps
non-NaN FP32 bits into monotonically ordered unsigned keys (negative complement, positive
sign-bit flip). Selection retains the input scores; NaN payload preservation is not
promised. The comparator is used by the
batch selector/merge in `SearchAndRetrieval.metal`, the warp/general/streaming/fused paths
in `AdvancedTopK.metal`, and IVF's candidate comparisons. Heap sinks reverse the same
comparator to keep the worst candidate at the root. Empty warp rows now reach padding;
fused winner removal invalidates its index. The missing fused shared-candidate publication
barrier was also added, matching IVF; the missing barrier was found by inspection, not
isolated as an empirical race failure.

**Public fallback alignment:** the optional CPU merge in `FusedL2TopKKernel` still used
NaN-unsafe sorting. A separate public chunked-API test reproduced **two failed assertions**
with CPU merge selected while GPU merge passed. The CPU sort now applies numeric-first,
NaN-last, global-index ties. Its public test exercises both configuration choices.

**Coverage/review:** nine direct-GPU tests exercise both plugin/runtime libraries in debug
and runtime in release; one public-API test exercises the default library in each build. Cases include NaN signs/payloads/signaling encodings, both infinities, signed
zero, subnormals, equal-score eviction, NaNs in initial heaps and across lanes/strides,
K=1/3/5/8/9/12/32/33/128 as applicable, padded multi-query strides, unsorted membership,
empty/all-NaN rows, reversed streaming chunk submission, real NaNs before sentinel slots,
and original IVF indices distinct from CSR order. Fused and IVF tests cover reduction and
bitonic branches. VectorCore 0.3.3 provides additional large-input parity checks alongside
literal fixtures. Independent review found no blocking issues; its fused bitonic coverage
suggestion was added. Targeted gate: **56 tests / 0 failures** (17.811 seconds).

**Boundaries:** this slice fixes selection ordering, not distance arithmetic or existing
capacity/dispatch limits. In particular, per-thread eight-candidate retention in direct
fused/IVF paths and the deprecated streaming host's known ABI/index-width issues are not
closed by these direct-kernel tests. No performance benchmark claim is made.
Full debug: **1621 tests / 0 failures / 11 skipped** (257.786 seconds).
Full release: **1621 / 0 / 11** (52.800 seconds). Both commands exited successfully.
Baseline 1611 plus ten new tests. VA3-016 is **FIXED** within the recorded scope.
Logs (`red-confirmed.log`, `green.log`, `cpu-merge-red.log`, `targeted-final.log`,
`debug-full.log`, `release-full.log`) and exact slice-start backups:
`/private/tmp/va3-016-topk/`. All work remains uncommitted.


## Remediation slice 13 (2026-09-07, owner-approved: VA3-022 Minkowski range policy) — EXECUTED

**Owner decision:** retain explicit `useStableComputation: false` as the faster path with
FP32 intermediate range limits; do not add automatic rescue. Keep fractional p support,
require finite p > 0, remove artificial cutoffs/clamps, and use exact special-case powers.
Automatic routing remains p > 10 to stable unless overridden; Chebyshev remains explicit.

**Red evidence:** six new `MinkowskiRangePolicyTests` initially produced **48 failed
assertions / zero unexpected failures** across both libraries and public validation.
One component differing by 1e20 at p=3/5/11 produced capped/distorted values (about
6.076e37 / 3.603e7 / 2721.9), rather than an honest overflowing fast-path intermediate.
The stable result for that fixture is 1e20. Tiny fractional/scalar-tail distances were
zeroed; near-1/near-2 exponents were silently substituted; fractional ratio underflow
and final rescaling lost representable results. Ordinary exact-metric/zero, large-scale
normalization, and explicit-Chebyshev controls passed. Positive-infinite p was accepted publicly.

**Implementation:** replace `safe_pow` with unclamped `minkowski_pow`, retaining only
exact zero/unit/common-power cases and integer nonfinite classification. Remove unused
fractional helper variants and their contradictory cutoff rules. Both scalar tails and
float4 components use the same helper. Host metric classification and shader specialization
now require exact p=1 or p=2. `execute` rejects nonfinite/nonpositive p; raw `encode` keeps
its nonthrowing API and documents the same caller precondition.

Stable mode uses `precise::divide(diff, scale)` for p >= 1 instead of a reciprocal that
can flush (preventive alignment with the established precise-division idiom; the large-scale
controls passed before this edit). For fractional p it computes normalized powers from log differences, avoiding
ratio underflow before a fractional power makes the term significant. Final fractional
rescaling preserves scale's significand with `frexp`, splits the root's binary exponent,
and applies `ldexp`; an exponent bound guards float-to-int conversion after overflow.
A single contributing component returns its scale directly, including extreme finite p.
Finite subtraction overflow produces infinity; degenerate pairs keep the uniform barrier
flow established in slice 1. This does not change the standalone L2/dot kernels (VA3-030).

**Review-driven correction:** the first rescale used `exp2(log2(scale) + log2(sum)/p)`.
Review suggested a finite-endpoint failure. Its initial p=1/2 fixture passed both libraries;
an expanded power-of-two sweep reproduced **eight failed assertions** at p=1/64 and
p=1/128, where the exact final value is FLT_MAX. The exponent-split implementation fixes
those cases; p=1/2 through p=1/32 remain controls. Independent follow-up review found no
blocking issues. This was tested before the full gates, not inferred from algebra alone.

**Coverage:** five direct-GPU methods exercise plugin and runtime libraries in debug,
runtime in release; one public method checks invalid/extreme p, exact metric labels, and
explicit stable/fast results. Cases include D=1/4/5/65 scalar/vector/tile tails, fractional
p, near-integer p, true overflow, fast underflow versus stable rescue, finite powers above
the old exp(87) cap, normalization near FLT_MAX, root-factor overflow with finite final
results, and fractional ratio underflow. Relative tolerances use Double references from
exact Float inputs. Final targeted gate: **36 tests / 0 failures** (12.688 seconds).

**Contract limits:** finite-input computations target the requested Lp formula; p<1 is
supported but is not a metric. Explicit fast mode can lose a representable final result
to subtraction/power/sum/root range limits. Stable mode avoids the reproduced normalization
and rescaling failures, but ordinary FP32 rounding, finite accumulation precision, and
flush-to-zero limitations remain. No all-input/full-range or performance guarantee is made.
Nonfinite data-input semantics and raw-encode validation are not broadened by this slice.
Full debug: **1627 tests / 0 failures / 11 skipped** (250.690 seconds).
Full release: **1627 / 0 / 11** (52.962 seconds). Both commands exited successfully.
Baseline 1621 plus six new tests. VA3-022 is **FIXED** under the approved range contract.
Logs (`red.log`, `green.log`, `rescale-endpoint-red.log` [passed control],
`rescale-sweep.log` [eight failures], `targeted-final.log`, `debug-full.log`,
`release-full.log`) and exact slice-start backups are in `/private/tmp/va3-022/`. All work remains uncommitted.


## Remediation slice 14 (2026-09-07, owner-approved: VA3-030 split range policy) — EXECUTED

**Decision:** direct rooted Euclidean/L2 gets automatic exceptional-range rescue while
ordinary FP32 accumulation remains intact. Squared L2 retains FP32 output/intermediate
limits. Dot products retain FP32 range and cancellation limits; scaling alone does not
establish a cancellation-accuracy contract. Minkowski's separate explicit-fast policy
from slice 13 is unchanged. The owner approved this split before implementation.

**Implementation:** `va_euclidean_finalize`, identically mirrored in `Metal4Common.h` and
`KernelContext`'s runtime preamble, returns the existing square root for normal finite
accumulators. Zero, subnormal, or nonfinite totals trigger a maximum-difference scan,
precisely divided normalized squares, and significand/exponent rescaling (`frexp`/`ldexp`).
Integer classification preserves NaN handling under fast-math; NaN differences dominate
infinite differences. Truly unrepresentable roots remain infinite. Existing reduction
barriers precede the exceptional scan; no additional buffer or dispatch is introduced.

The helper covers `euclideanDistance`, `batchEuclideanDistance`, and the rooted branches
of `l2_distance`, `l2_distance_kernel`, and `soa_l2_distance`. SoA passes its candidate
lane stride explicitly. Squared branches bypass the helper. CPU fallback finalization
recomputes exceptional sums in Double, converting operands before subtraction and the
result after the root. Accelerate, both SIMD fallback configurations, FallbackProvider,
engine/batch routes, mapped Euclidean search, and the CPU quantization centroid helper
share this finalizer. Public L2/dot documentation links the range contract.

**Scope:** see [`DISTANCE-RANGE-CONTRACT.md`](../stability/DISTANCE-RANGE-CONTRACT.md).
This does not promise all-input accuracy, bitwise CPU/GPU parity, or performance gains.
Normal finite totals retain accumulation error; GPU flush-to-zero can erase subnormal
operands/differences. Fused/index squared-score selection and later rooted presentations
(clustering, product quantization, HDBSCAN, SearchResult) retain squared-score limitations.
Learned projection distances retain projection/squared-sum limits, even with optional
rooted output. Weighted/transformed metrics, RMSE, and normalization are separate.
Independent review found no blocker in the changed paths and identified these scope
qualifications, which are explicitly recorded rather than claimed as rescued.

**Evidence:** eight new `EuclideanRangePolicyTests`; initial red **574 failed assertions,
0 unexpected failures**. GPU tests dispatch both plugin and runtime libraries in debug
and runtime in release, using Double/closed-form references and poisoned output guards.
They cover huge/tiny normal differences, subnormal squared totals, scalar tails, ragged
reduction widths, SoA lane stride, zero/ordinary cases, FLT_MAX endpoints, large common
offsets, genuine overflow, NaN/Inf, unchanged squared overflow, and dot cancellation
outside the retained range. CPU variants, public engine/batch routes, and mapped search
are exercised. Quantization's shared helper substitution has full-suite regression
coverage but no dedicated exceptional-range public fixture.

Targeted regression gate: **169 tests / 0 failures** (25.905 seconds).
Full debug: **1635 tests / 0 failures / 11 skipped** (256.978 seconds), exit 0.
Full release: **1635 / 0 / 11** (51.769 seconds), exit 0. Baseline 1627 plus eight new
tests. VA3-030 is **FIXED under the approved split range contract**; Group B is closed
with the documented limits retained. Logs and slice-start source backups are in
`/private/tmp/va3-030/`. All work remains uncommitted.

## Remediation slice 15 (2026-09-07, owner-approved: Group C VA3-018) — EXECUTED

**Scope:** fix the corpus-wide 32-bit row/address arithmetic class. Counts, dimensions,
strides, kernel parameter layouts, and stored IDs keep their existing ABI. Device-buffer
products promote an operand to `ulong` before multiplication; address locals and helper
arguments preserve the wide value. Independent products in batched/strided addresses
are each promoted, rather than relying on a wide enclosing sum.

**Implementation:** 21 shader files cover L2/dot/learned/attention distance, normalization,
basic batch operations, mutual reachability and Borůvka, UMAP, SoA, neural/product/scalar
quantization, clustering, elementwise strides, matrix operations, statistics/LSE,
search/IVF, and sparse TF-IDF output rows. Specialized dimension paths are included.
Normalization's downstream `idx` and `va_copy_bits` index parameter are widened too.
UMAP's flattened element-count comparison also uses a wide product. No shared-header,
runtime-preamble, Swift parameter, or binding changes are needed.

**Verification:** ten `Hardening/IndexWidthTests` extract actual shader expressions and
local/parameter types, compile GPU arithmetic probes with Metal 4 fast math, and compare
against UInt64 references below/at/above 2^32, realistic 3M×1536 and 66k×66k shapes, and
representable extreme products. Batch offsets, independent strided/unrolled products,
and downstream narrowing are covered. Initial probes reproduced production failures;
the finalized probes replayed against untouched slice-start shader copies report
**647 incorrect offsets**, and all ten tests pass on the fixed sources. A comment-only
match in the initial harness was removed before that final baseline replay.
Targeted integration: **114 tests / 0 failures** (23.571 seconds), then expanded review
probes **10 / 0**. Full debug: **1645 / 0 failures / 11 skipped** (257.727 seconds), exit 0.
Full release: **1645 / 0 / 11** (56.339 seconds), exit 0. Both shader compilation paths
are exercised by the regression suites. Focused independent review found no remaining
row/address-product or narrowing blockers; its matrix/helper coverage concerns are closed.

**Limits:** probes calculate offsets without dereferencing multi-gigabyte buffers; they
are not large-allocation tests or a throughput benchmark. Existing 32-bit count/grid/ID
limits and near-limit grid/loop arithmetic remain. In particular, deprecated streaming
Top-K's explicit global-ID `ulong`→`uint` conversion is a legacy ABI residual, not fixed
by address promotion. Alignment (VA3-017), PQ/candidate capacities, input validity, and
UMAP races remain separate findings. See
[the address-width contract](../stability/INDEX-WIDTH-CONTRACT.md).

VA3-018's row/address arithmetic class is **FIXED**. Group C remains open. Logs,
standalone baseline replay, exact slice-start copies, and the slice-only shader diff
are in `/private/tmp/va3-018/`. All work remains uncommitted.


## Remediation slice 16 (2026-09-07, owner-approved: continue with VA3-017) — EXECUTED

**Decision:** scalar-backed vector accesses use packed memory views; ordinary vector
register arithmetic, buffer parameter ABI, element sizes, strides, shared-array sizes,
and dispatch geometry remain intact. No new dimension/alignment rejection policy.

**Implementation:** 15 `.metal` files plus `Metal4Common.h` replace alignment-increasing
`float4`/`uint4`/`char4` pointer views with scalar-aligned packed equivalents. Includes
general and specialized distance/projection/normalization paths, neural/PQ, tiled
Minkowski/Hamming, IVF/Top-K helpers, clustering/Borůvka, and scalar threadgroup storage.
Explicitly vector-typed buffer arguments keep their existing alignment/layout contract.
The plugin header-dependency workaround was applied: touched all `.metal` files after
header edits. Runtime preamble and numerical formulas are unchanged.

**Additional confirmed defects in touched accesses:**
- Generic transposed neural decode requested four weights for the final 1-3-output
  block. At D=5/L=5, the last load requested elements 24–27 of a 25-element weight buffer.
  Value parity and `MTL_SHADER_VALIDATION=1` initially passed because unused lanes can
  mask the invalid request. A production-source footprint probe wraps all 20 full-block
  weight loads and records bounds before dereferencing; it failed in **20 cases**.
  Tail threads now compute only valid outputs with scalar weight reads after the final
  barrier, then return. Complete blocks retain their vector path. Tail accumulation
  order can differ from the dual-accumulator path within ordinary FP32 tolerance.
- Three specialized neural vector stores still computed `vectorIdx * INPUT_DIM` in
  32 bits: missed by slice 15's address inventory. A new IndexWidthTests probe failed
  with **15 incorrect offsets** before these stores were promoted to `ulong`. The new
  scalar-tail output offset is also covered. This closes that VA3-018 follow-up.

**Verification:** eight new `VectorAlignmentTests` combine offline compiler diagnostics
with real GPU tests through plugin/runtime libraries: all alignment residues in strided
L2/dot rows, paired reductions, odd-stride normalization and raw subnormal bit copies,
PQ subspaces, neural code/weight/output rows, Hamming odd-word tiles, and Minkowski tiles.
The baseline compiler check failed on actual alignment increases while the initial
value check passed. The final guard also normalizes explicit reinterpret casts to
semantically equivalent C-style casts in temporary copies because `-Wcast-align` otherwise
accepts them; a deliberately invalid reinterpret-cast canary verifies the diagnostic.
Whitespace in cast syntax is covered. Full-block neural footprint instrumentation must
match all 20 production loads to avoid silent partial coverage.

Targeted: **33 tests / 0 failures** (4.994 seconds). Shader validation: **8 / 0**
(3.388 seconds), no reported validation errors. Final strengthened diagnostic guard also
passes in isolation. Full debug: **1654 / 0 failures / 11 skipped** (259.638 seconds),
exit 0. Full release: **1654 / 0 / 11** (55.419 seconds), exit 0. Nine added test
methods over slice 15 (eight alignment tests plus one index-width test). The whitespace
canary refinement was verified separately after the debug run and in the full release run;
production shaders were unchanged after the targeted/validation gates. Independent
review found no remaining production blockers; tail and test-coverage findings addressed.

VA3-017 is **FIXED** for scalar-backed pointer conversions. These checks do not expand
vector-buffer ABI, capacity/count/ID limits, or prove arbitrary caller buffer offsets.
No performance claim. See [the vector alignment contract](../stability/VECTOR-ALIGNMENT-CONTRACT.md).
Logs, original source copies, footprint/index red runs, and slice-only shader diff are
in `/private/tmp/va3-017/`. Group C still has VA3-025/-026/-028 input/capacity contracts.
All work remains uncommitted.



## Remediation slice 17 (2026-09-07, owner-approved: VA3-025 sparse TF-IDF bounds) — EXECUTED

**Scope and implementation:** the vectorized c-TF-IDF shader retains its buffer ABI
and complete-group arithmetic. A final 1–3-entry group uses scalar reads, gathers, and
writes bounded by `nnz`; the base multiplication promotes to `ulong` first. Host routing
still selects vectorization only for `nnz >= 16 && nnz % 4 == 0`.

K=0 now returns before any buffer access in the Top-K shader. The nonthrowing encoder
requires UInt32-representable nonnegative counts and encodes no work for K=0 or zero
clusters. The standalone throwing API rejects invalid K before checking emptiness and
returns an empty list per cluster with zero timing/throughput for K=0. Positive-K ranking,
sentinels, and score arithmetic are unchanged.

**Reproduction:** the first five `SparseTFIDFBoundsTests` failed with **50 assertions**:
unused-ID gathers, overwritten tail output guards through both compile paths, a zero-K
shader reaching its first cluster read, the encoder reporting a nonzero dispatch, and
negative K accepted for empty input. Test-only gather instrumentation records violations
before dereferencing poisoned IDs. The zero-K probe intercepts the first cluster read
before the original underflow can fault. Those five tests then passed after the fix.
Three further integration tests cover exact allocations, real K=0/1/oversized-K dispatches,
empty clusters, output sentinels, extra threads, and public empty-result shape/metadata.

**Validation detail:** initial API+shader validation caught an invalid *test binding*:
`uint4*`/`float4*` arguments require at least 16 bound bytes, including nnz=1–3. The exact
allocation test now uses nnz=5/6/7/17/18/19; smaller tails retain ABI-valid backing storage
and poisoned unused lanes/output canaries. No shader logic change was needed. This ABI
minimum is explicit in source and the contract. Final validation: **8 tests / 0 failures**
(1.234 seconds), with both `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1`; exit 0,
no reported validation errors. The earlier targeted gate passed **42 / 0** (2.928 seconds).

Full debug: **1662 tests / 0 failures / 11 skipped** (261.782 seconds), exit 0.
Full release: **1662 / 0 / 11** (59.614 seconds), exit 0. Eight added tests over
slice 16. Final production and test sources were used for both full gates. VA3-025 is
**FIXED**; Group C still has VA3-026/-028 capacity contracts.
Independent read-only review found no production blockers. The encoder test checks its
reported dispatch count; the early return before encoder mutation is also source-reviewed.
These tests do not validate arbitrary malformed sparse inputs, broaden count/ID or
allocation limits, change nonfinite ranking policy, or establish a performance improvement.
See [the sparse TF-IDF bounds contract](../stability/SPARSE-TFIDF-BOUNDS-CONTRACT.md).
Logs and exact slice-start copies are in `/private/tmp/va3-025/`. The owner authorized
checkpointing slice 17 on `gifton/metal-hardening-checkpoint`, the ongoing branch for
all work before and after the handoff. Slices 1–16 are backed up in `d39eee0`.



## Remediation slice 18 (2026-09-07, owner-approved: VA3-026 PQ bounds) — EXECUTED

**Scope:** retain UInt8 codes, K=1...256, and the 32 KB ADC shared-table limit. The
existing host initializer already rejected K>256 (the original short audit description
omitted that guard). The missing bounds were raw shader capability checks, nonpositive-K
execution validation, the ADC allocation/dispatch limit, and code indices within K.

**Implementation:** assignment fills 0xff for unsupported K before input reads; centroid
255 remains valid for K=256. Training accumulation rejects unsupported K and skips codes
outside their subspace before pointer formation. ADC uses a group-uniform guard before
shared loading/barriers, with `M <= 8192 / K` checked only after positive valid K. Invalid
shapes fill live output distances with NaN. Invalid codes in otherwise valid tables fill
that vector with NaN after the barrier, before lookup. Full-byte codes and ordinary
at-cap arithmetic are preserved.

Host train/encode/ADC entry points reject nonpositive K before allocation; combined
training/encoding does so before staging. `computeDistances` checks the table bound
before product/allocation/UInt32 conversion/encoding, rounds dynamic shared binding to
16 bytes, and checks device memory minus static pipeline usage. Oversized ADC throws;
training/encoding retain larger-model support. Binding alignment is required by Apple's
Metal API, and the small/odd-table tests cover the rounded host path.

**Evidence so far:** four initial `PQBoundsTests` failed with **57 assertions**, covering
centroid 256 wrapping to byte zero, unsupported shapes reaching shared loads (including
a product wrapping at 2^32), invalid-code lookups, and cross-subspace/out-of-range atomic
writes. All four passed after the fix. Test-only source instrumentation intercepts bad
ADC accesses before dereferencing so the baseline cannot fault the GPU; raw accumulation
uses physically allocated guard capacity. Six further tests cover real library rejection,
at-cap/ragged dispatch, early host rejection including Int.max M, small table alignment,
retained larger-model train/encode, and nonpositive-K host validation. Both plugin and
runtime libraries are exercised in debug, runtime in release.

Targeted **47 tests / 0 failures** (2.824 seconds). Independent review found no production
blockers; added its requested accumulation controls for unsupported K=0/257 and valid
K=256/code255 to the existing test. Final API+shader validation **10 / 0** (0.277 seconds),
exit 0, no reported validation errors. Full debug: **1672 / 0 failures / 11 skipped**
(256.920 seconds), exit 0. Full release: **1672 / 0 / 11** (56.807 seconds), exit 0.
Both full gates use the final production and test sources. VA3-026 is **FIXED**; Group C
now has only VA3-028 remaining.

No general buffer/shape/model-compatibility or CPU decode validation claim; raw callers
still supply sufficient aligned dynamic shared memory and valid layouts. No performance
or deterministic-training claim. See [the PQ bounds contract](../stability/PQ-BOUNDS-CONTRACT.md).
Logs and exact slice-start sources are in `/private/tmp/va3-026/`. Work remains on the
single `gifton/metal-hardening-checkpoint` branch.



## Remediation slice 19 (2026-09-07, owner-approved: VA3-028 Borůvka bounds) — EXECUTED

**Implementation:** retain the 2N candidate allocation with the finite geometric bound:
for a valid undirected graph, complete merging halves the number of active components;
isolated components emit nothing. Repeated unmerged fusion rounds violate this premise,
so the collector now enforces capacity independently. The fourth/padding UInt32 parameter
word carries actual record capacity; total struct size and buffer binding indices stay
unchanged. CAS reserves unique in-range slots; full-buffer attempts monotonically mark
`capacity + 1` without writing, and capacity is limited to UInt32.max-1 so the counter
cannot wrap. Standalone count reads throw before dispatch/readback if overflowed.
Fusion exposes `candidateCapacity` and `readCandidateCount()` for safe post-completion
readback; its documentation explicitly requires initialization/reset and complete merging.
Allocation validates 2N representability and the device maximum buffer size first.

**Edge validity:** all five find-min variants use UInt32.max as no-target, and component
reduction propagates/clears endpoint sentinels for missing/non-representative edges.
Genuine +infinity weights are selected when no smaller usable edge exists; finite edges
can replace an initial infinite winner. Collector bounds-checks both endpoints before
component lookup and no longer filters edges by infinity weight. NaN candidate weights
retain comparison-style rejection via a fast-math-safe bit test. Input `max`/distance
arithmetic, FP32 squared range, and nondeterministic atomic candidate order are unchanged.

**Evidence so far:** five initial `BoruvkaBoundsTests` failed with **292 assertions**:
capacity over-writes/counter overflow, omitted infinite edges, stale non-representative
endpoints, and disconnected public MST output. Physical guard storage keeps the baseline
collector writes inside the allocation while testing the declared logical bound. Those
five tests passed after the fix. Five additional tests cover a saturated UInt32.max
counter, invalid endpoints, repeated unmerged fusion rounds (including 257 concurrent
candidates), geometric-bound rounds on a hierarchical 32-point fixture, and huge/invalid
allocation counts. The fixture independently checks the known 1-D MST weight.

Targeted **54 tests / 0 failures** (22.426 seconds). Independent review found no blockers;
removed the optional/default capacity argument as suggested so omitted future wiring is
a compile error. Expanded the existing selection/public tests with finite-after-infinite
and squared-overflow fixtures. Final API+shader validation **10 / 0** (0.324 seconds),
exit 0, no reported validation errors. Full debug: **1682 / 0 failures / 11 skipped**
(259.798 seconds), exit 0. Full release: **1682 / 0 / 11** (55.297 seconds), exit 0.
Both full gates use the final production and test sources. VA3-028 is **FIXED** and
**Group C is closed under its documented contracts**.

See [the Borůvka bounds contract](../stability/BORUVKA-BOUNDS-CONTRACT.md) for raw ABI,
count, initialization, overflow recovery, numerical-range, and determinism boundaries.
No throughput or arbitrary malformed-buffer claim. Logs, initial sources, slice diff,
and final production/test hashes are in `/private/tmp/va3-028/`. Work remains on the
single `gifton/metal-hardening-checkpoint` branch.


## Remediation slice 20 (2026-09-07, owner-approved: VA3-019 IVF candidate bounds) — EXECUTED

**Scope:** first bounded VA3-019 slice. The fused IVF builder previously reserved and
wrote beyond an estimated allocation. It also returned atomic per-query starts as CSR
boundaries, which is incorrect whenever allocation order differs from query order.
Both defects reproduced in two initial `IVFCandidateBoundsTests`: **446 assertions**,
zero unexpected failures. They passed after bounded reservations and CSR conversion.

The fused shader now reserves complete query segments within explicit physical capacity,
uses a non-wrapping overflow marker, and publishes invalid/empty descriptors on failure.
Host overflow recovery discards the attempt and runs exact count/prefix/build; allocation
hints never silently truncate recall. Completed unordered segments are validated and
GPU-blitted into query-ordered CSR, with zero-copy retention when already ordered.
The fourth UInt32 shader parameter now carries capacity; raw callers must supply it.
Three-pass prefix accumulation is wide and saturates to a rejected UInt32.max marker.

Expanded tests exposed a zero-query division-by-zero in dispatch configuration; empty
inputs now return before encoding. Independent review found that the pool can return
64 MiB for larger requests: fused capacity now uses actual lengths, and every exact
output/metadata allocation checks physical storage before writes. The global pool
behavior remains separate debt. Review approved after both issues were addressed.
Descriptor/counter pool leases are retained through GPU completion and CSR conversion.

Eight tests cover both compile paths, logical canaries, complete reservations, empty
queries, counter saturation, prefix overflow, skewed-list fallback, deliberate segment
permutations, invalid layouts/counts, and oversized pooled outputs. Targeted **139 / 0 failures / 11 skipped** (113.332s),
API+shader validation **8 / 0** (1.130s), exit 0 with no reported validation errors.
Restoring the old prefix arithmetic produced two assertion failures in its dedicated
regression (both compile paths); the final fix was restored before the gates.
Full debug **1690 / 0 / 11** (256.476s), full release **1690 / 0 / 11** (54.272s),
both exit 0, on identical final production/test sources. Read-only review approved.
See [the IVF candidate contract](../stability/IVF-CANDIDATE-BOUNDS-CONTRACT.md).
Logs and initial production sources are in `/private/tmp/va3-019/`. No performance claim.
**VA3-019 remains open for UMAP's embedding race and atomic accumulation policy.**


## Remediation slice 21 (2026-09-07, owner-approved: VA3-019 UMAP negative sampling) — EXECUTED

**Scope:** remove the true cross-thread embedding race. The shader reads all targets
from immutable pass-start coordinates, initializes a distinct output row per point,
and preserves sequential updates across that point's samples. A second GPU dispatch
publishes completed rows back to the embedding. Three buffer barriers cover prior
producers, sampling-to-publication, and later consumers/reuse on concurrent encoders.
Self and out-of-range IDs are skipped; addresses remain wide, with no fixed-D scratch cap.
The formula, clipping placement and FP32 math policy are unchanged.

The owner explicitly approved a source compatibility change: `encodeNegativeSampling`
now throws and has a caller-owned scratch overload. Convenience encoding allocates
private output (retained command buffers required); reusable scratch supports unretained
command buffers with explicit resource lifetime through completion. Negative sampling
validates UInt32 counts, device-limit products, physical lengths, device identity and
GPU-address overlap before encoding. Zero N/D/rate is a no-op; no pooled allocation is
used. High-level async signatures are unchanged. Raw shader callers must add distinct
output at buffer(3); UMAPParams stays 32 bytes. See
[the contract](../stability/UMAP-NEGATIVE-SAMPLING-CONTRACT.md) for migration and limits.

Two initial regression tests failed **1262 assertions** on the original implementation.
Nine final tests cover both compilation paths, analytic sequential-source expectations,
a Double reference with default curve parameters, output/input guards, ragged D,
self/invalid IDs, empty/invalid shapes, scratch overlap/length, concurrent encoder
ordering, explicit lifetime with unretained references, scratch reuse, and epoch order.
Targeted **40/0** (0.680s), API+shader validation **9/0** (1.090s), both exit 0.
Full debug **1699/0/11** (258.028s), release **1699/0/11** (55.735s), both exit 0,
on identical production/test sources.
Read-only review approved with no findings. Logs: `/private/tmp/va3-019-umap/`.

**VA3-019 remains open for atomic accumulation policy.** Fixing the embedding race does
not make whole epochs bitwise deterministic: floating target-gradient atomics and random
sample generation remain. Numerical/input-clamping backlog is separate. Scratch costs
4ND bytes and publication adds a GPU copy dispatch; no performance improvement claimed.


## Remediation slice 22 (2026-09-07, owner-authorized continuation: VA3-027 flags) — EXECUTED

**Scope:** the two audited ignored flags. `encodeTiledV3` forwards the dispatch's
`useActivation` byte as a UInt32 at new raw pass-1 buffer(8). The tiled affine projection
applies ReLU after bias only when requested; disabled activation preserves negative
intermediates and signed INT8 codes. Swift signatures and the neural parameter struct
layout are unchanged. Raw pass-1 callers must now bind the activation constant.

Both specialized learned L2 kernels honor `normalizeProjected` by materializing complete
projections and reusing the general projection, normalization and squared-difference
helpers. `computeSqrt` remains independent; unnormalized fused loops stay unchanged.
Existing FP32/epsilon and dense-layout limits remain; no performance improvement claimed.

Four new tests initially failed **5129 assertions** on the original code. Final coverage
includes both shader compilation paths, activation/bias combinations, partial tiles,
over-dispatch and canaries, signed public tiled codes/scales, per-dispatch flag forwarding,
both specialized learned dimensions and general reference, normalization/root combinations,
zero/tiny/parallel/opposite projections, padded output rows and public selection paths.
Targeted **42/0** (13.118s), API+shader validation **4/0** (5.406s), full debug
**1703/0/11** (258.458s), release **1703/0/11** (56.740s), all exit 0 on identical
production/test sources. Read-only review approved, no findings. Logs: `/private/tmp/va3-027/`.
See [the flag contract](../stability/IGNORED-FLAGS-CONTRACT.md).

This closes the original VA3-027 scope and the dedicated `encodeTiledV3` coverage gap.
**Adjacent residual:** source inspection shows `normalizeLatent` is read only in generic
`neural_encode_quantize_kernel`; specialized and tiled quantized neural encoders ignore
it. Its behavioral regression and normalization/scale parity fix remain a separate slice.
VA3-019 atomic accumulation policy remains open; no new determinism guarantee.


## Remediation slice 23 (2026-09-07, owner-authorized continuation: neural latent normalization) — EXECUTED

**Scope:** the adjacent `normalizeLatent` omission identified in slice 22. Three exported
specialized INT8 quantizers now call the generic normalization helper after affine/ReLU.
Tiled pass 2 receives the dispatch flag at new UInt32 buffer(5). One lane computes the
norm in sequential FMA order and publishes a reciprocal via a uniform barrier; all lanes
reduce normalized magnitudes and quantize normalized values. Input intermediates stay
immutable. Enabled paths match the generic computed-norm epsilon cutoff and minimum
scale. Disabled paths retain their prior behavior, including tiled zero/tiny scale rules.
Swift signatures, parameter struct layout and public latent cap remain unchanged.

Three tests initially failed **510 assertions**. Tests cover all raw specializations and
the generic reference, bias/ReLU/norm combinations, code and scale expectations, zero and
tiny rows, tiled L=1/3/33/128/257, widths 32/96/256, input immutability, over-dispatch guards,
and public tiled/generic parity with per-dispatch flags at four shapes. Raw larger-L
coverage does not widen public limits. Current public quantize dispatch uses the generic
shader; exported specializations are exercised directly.

Metal validation exposed a second local issue: the public generic quantizer left its
optional bias at buffer(4) unbound, causing an assertion despite the shader's null check.
It now binds a persistent 128-float zero fallback; real bias takes precedence. Allocation
failure uses the existing throwing initializer. The first full-debug attempt was stopped
after that validation failure; all final gates were rerun on the corrected sources.
Final targeted **61/0** (16.893s), API+shader validation **7/0** (4.508s, includes slice-22
tests), full debug **1706/0/11** (258.714s), release **1706/0/11** (56.135s), all exit 0.
Read-only review approved both the normalization change and bias correction. Final source
hashes unchanged across gates. Logs: `/private/tmp/va3-normalize-latent/` (including
`validation-red.log`). See [the contract](../stability/NEURAL-LATENT-NORMALIZATION-CONTRACT.md).

No robust full-range norm, bitwise parity, performance improvement or broader optional-
bias validation claim. **New source-confirmed residual:** high-level `encode()` averages
per-vector scales into its result; `decode()` fills every scale slot with that average.
Different row scales are lost. A result/API preservation fix and regression are next
concrete reconstruction work. Other optional-bias entry points remain candidates for
validation. VA3-019 atomic policy stays open.


## Remediation slice 24 (2026-09-08, owner-authorized continuation: neural result scales) — EXECUTED

**Scope:** the reconstruction defect found in slice 23. `Metal4NeuralEncodingResult`
now owns one scale per vector; both high-level decoders and two direct benchmark
consumers use that array. The previous `.scale` remains a deprecated diagnostic average.
Code-payload metrics retain their values and now explicitly exclude scale metadata.
GPU results are copied before pool leases end. Decoders validate dimensions, exact code
and scale counts, and output products; encoding rejects ragged rows. All high-level
neural pooled allocations check actual capacity before copying or dispatching, locally
containing the known oversized-request pool defect without changing the global pool.

The first three regression tests failed **64 assertions** on the original implementation.
Six final tests cover unequal row magnitudes through both decoders, all transposed
specializations, normalized/zero rows, scale ownership through pool reuse and reload,
malformed results, ragged input rejection and direct non-transposed decoder variants.
Exact INT8 coordinates isolate scale loss from quantization error.

Two adjacent failures were reproduced while validating the decoder surface:
- Metal API validation aborted on unbound decoderBias buffer(4) in transposed v2.
  Weight loading now prepares an output-sized zero-bias buffer; all decoder wrappers
  bind it when no real bias exists. Unloading releases it; real bias takes precedence.
- Direct L=3 tests failed **16 assertions** because optimized non-transposed float4
  kernels discarded the incomplete latent block. Swift selects those variants only
  for L divisible by four; other shapes use the existing scalar fallback. L=4 controls
  cover all four optimized widths, and a trailing output guard remains intact.

Final targeted **55/0** (24.566s), API+shader validation **6/0** (0.198s), full debug
**1712/0/11** (260.265s), release **1712/0/11** (56.602s), all exit 0. Final source
hashes were unchanged across full gates. Read-only review approved the implementation,
routing correction and contract. Logs: `/private/tmp/va3-neural-scales/`
(including original red, validation abort and ragged fallback red). See
[the result contract](../stability/NEURAL-ENCODING-SCALES-CONTRACT.md).

No raw ABI, public latent-cap, FP32 range, model-identity, concurrent weight-loading,
lossless reconstruction or performance guarantee is added. Other optional-bias APIs
remain validation candidates; the global buffer pool remains separate debt. VA3-019
atomic accumulation policy remains open alongside Group G and recorded residuals.


---

Liveness legend: **LIVE** (dispatched by shipping Swift), **LIVE-cond** (live behind a config or public-API parameter), **LATENT** (kernel defect shielded by the current caller's exact geometry), **DEAD** (no Swift dispatch site).

---

## Summary index

| ID | Severity | Liveness | One-liner |
|----|----------|----------|-----------|
| VA3-001 | P1 | **FIXED** | `jaccardDistance`: barrier-divergence UB (dim % 256 ≠ 0) + multi-threadgroup `atomic_store` race (dim > 256) |
| VA3-002 | P1 | **FIXED** | BasicOperations tree reductions drop lanes for non-pow2 threadgroups; live for `cosineDistance`/`dotProduct` at dims 17–255 when no decision engine |
| VA3-003 | P1 | **FIXED** | `minkowski_distance_stable`: divergent early-return before pass-2 barriers (UB on any tile containing a near-identical pair); auto-selected for p ∈ (10, 30] |
| VA3-004 | P1 | **FIXED** | `batchMatrixMultiplyFused` bias indexed `[row*N+col]` (M×N) vs Swift-accepted N / batchSize×N layouts → GPU OOB read, wrong bias |
| VA3-005 | P1 | **FIXED** | `computeQuantizationStats`: no grid guard + ceil dispatch → OOB device *writes* for numVectors > 1024 non-multiple |
| VA3-006 | P1 | **FIXED** | `batchCosineSimilarity` missed the VA2-008 remediation: naive accumulators, reassociation-prone product denominator, no clamp — live GPU path |
| VA3-007 | P1 | **FIXED** (slice 5: max-norm now explicit opt-in) | `minkowski_distance_batch` silently substituted Chebyshev for p > 10 (error up to D^(1/p), ~1.8× at p=11/D=768); was the default path for p > 30 |
| VA3-008 | P1 | **FIXED** (was LIVE — liveness corrected) | `warp_select_small_k_*`: uninitialized output tail when candidateCount < k; sentinel slots skipped entirely |
| VA3-009 | P1* | **FIXED** | `threadgroup float tgScale = 0.0f` initializer race in 9 NeuralQuantization sites — scale can nondeterministically read 0 (*probe to confirm codegen; fix regardless) |
| VA3-010 | P1 | **FIXED** (DotProduct guarded; MutualReach structurally safe, invariant documented) | Dimension-only specialized-kernel selection without stride guard: `DotProductKernel`, `MutualReachabilityKernel` (L2Normalization has the guard — copy it) |
| VA3-011 | P2 | **FIXED** (slice 7: host throws + kernel sentinel/NaN fills; found TopKParameters silent k-clamp + armed/fixed the throw-mid-encode encoder leak) | Silent capability caps: attention headDim (256/64), learned projectedDim (256), neural latentDim (128, **with stride corruption**), IVF nprobe (64), topk encode K (128) |
| VA3-012 | P2 | **FIXED** (slice 6: single epsilon authority, token surgery deleted) | EPSILON dual-compile drift confirmed with victims: files after DistanceShaders saw 1e-8 in the release combined build, 1e-7 in the debug metallib |
| VA3-013 | P2 | **FIXED** (was LATENT) | Guard-before-barrier + `id % 256` traps in manhattan/chebyshev — correct only under the engine's exact one-group dispatch |
| VA3-014 | P2 | **FIXED** (was LATENT) | LogSumExp/Statistics reductions: orphaning trees *with comments claiming non-pow2 robustness*; pass2 fixed-256 tree reads uninitialized shared for smaller groups |
| VA3-015 | P2 | **FIXED** (slice 9) | Correlation uses smaller-norm-first precise division; histogram and LSE/softmax gates use dynamic finite-limit comparisons; both compile paths covered |
| VA3-016 | P2 | **FIXED** (slices 10–12) | LSE/basic-statistics NaN propagation; VectorCore 0.3.3 CPU contract; shared GPU NaN-last/index-tie ordering and CPU chunk merge alignment |
| VA3-017 | P2 | FIXED (slice 16) | Scalar-backed device/threadgroup vector accesses use packed views; compiler alignment guard and odd-layout GPU tests; neural tail overread also fixed |
| VA3-018 | P2 | FIXED (slice 15) | Row/address products and downstream indices use `ulong` across 21 shader files; existing count/grid/ID limits retained |
| VA3-019 | P2 | LIVE | Float accumulation policy remains open; IVF bounds/CSR fixed in slice 20 and UMAP negative-sample race fixed in slice 21 |
| VA3-020 | P3 | **EXECUTED** (minus warp/batch/streaming select — live, see slice-1 notes) | Dead/broken kernel inventory for deletion — including `minkowski_distance_fractional` (tile load is a stub comment; reads uninitialized shared memory) and `tiled_kmeans_distance` (incoherent tile load) |
| VA3-021 | P2 | **FIXED** (slice 8: all 8 phantom makeFunction literals deleted, class test closes the family) | `tiledTransposeInPlace` — 6th phantom function name; in-place requests silently downgrade and write to the *output* buffer |
| VA3-022 | P2 | **FIXED** (slices 1, 13) | Removed power cutoffs/clamps and approximate p substitution; stable normalization/rescaling hardened; explicit fast-path FP32 limits retained |
| VA3-023 | P3 | LIVE | `use_fast_math=0` in DataTransformations doesn't disable fast math (whole library compiles `.fast`) — dishonest API flag |
| VA3-024 | P3 | LIVE | Perf pathologies: single-pair euclidean dispatches **one thread**; `batch_select_k_nearest` uses 1/256 threads; hamming-single 256× overdispatch; per-element softmax O(D²/row) |
| VA3-025 | P3 | **FIXED** | Slice 17: scalar-bounded c-TF-IDF vector tails; zero-K shader/host no-op; invalid K rejected; vector ABI and host routing retained |
| VA3-026 | P3 | **FIXED** | Slice 18: byte-code and ADC 32 KB bounds enforced; host throws before encoding, raw ADC NaN-fills; invalid assignments cannot cross subspaces; larger-model training/encoding retained |
| VA3-027 | P2 | FIXED (slice 22) | Tiled activation and specialized learned normalization flags honored; adjacent neural normalizeLatent omission fixed in slice 23 |
| VA3-028 | P3 | **FIXED** | Slice 19: bounded reservations and checked readback; explicit 2N proof; invalid endpoints distinguish missing edges from genuine infinity across all find/reduce/collect paths |
| VA3-029 | P3 | — | Header/hygiene: triplicated helper families (va_/ivf_/bare), `VA_EPSILON_HALF` type mismatch, misnamed prefix-sum, non-hygienic debug macro |
| VA3-030 | P3 | FIXED (slice 14) | Approved split policy: direct rooted-L2 exceptional-range rescue; squared L2/dot retain documented FP32 limits; derived squared-score roots and learned projections retain their limits |
| VA3-031 | P1 | **FIXED** (was LIVE — discovered by slice 2's red run) | `getPipeline(functionName: "dotProduct")` rewrote the literal kernel name to the batch `dot_product_kernel`: the engine's single-pair dot product never dispatched its kernel and returned stale pool bytes for every dim > 16 |
| VA3-032 | P1 | **FIXED** (was LIVE — found by meta-review slice 4) | `BatchDistanceEngine.batchDotProduct`/`batchManhattanDistance` GPU branches dispatched phantom kernels; VA2-003's k-gate exemption armed them — public APIs threw shaderNotFound where CPU results were expected |
| VA3-033 | P2 | **FIXED** (was LIVE — found by meta-review slice 4) | `batchCosineSimilarity`'s sub-100-candidate CPU leg kept the pre-VA2-008 naive formula — NaN/overflow answers flipped across the batch-size-100 boundary of one public API |
| VA3-034 | P2 | **FIXED** (was LIVE-cond — found by meta-review slice 4) | `warp_select_small_k_*` with k > 32 returned without writing any output slot (stale pool bytes read back as results) through the public `encodeWarp` |

Routing observation (no ID): with a decision engine attached, post-VA2-003 complexity gating (`1·1·dim ≥ 50 000`) makes the engine's single-pair GPU paths effectively unreachable — their bugs are invisible to provider tests but fully exposed to direct `Metal4ComputeEngine` users, who get `decisionEngine: nil` by default and the `dimension > 16` fallback.

---

## Group A — Reduction & barrier correctness (VA3-001, -002, -003, -013, -014)

**Amortized fix:** one shared, dispatch-robust threadgroup reduction helper (the pattern already exists: `va_tg_reduce_max/add` in BasicOperations.metal:89-120 starts at a fixed pow2 stride with a ragged-tail guard and is correct for any lane count) + a rule that *no thread returns before the last barrier* (clamp work, don't return — `kmeans_assign_points`' `is_active`/`safe_gid` pattern at ClusteringShaders.metal:30-36 is the reference).

### VA3-001 (P1, LIVE): `jaccardDistance` — divergence UB + cross-threadgroup race
`DistanceShaders.metal:119-159`, dispatched from `JaccardDistanceKernel.swift:226-232` with **fixed 256-thread groups and `ceil(dimension/256)` groups**.
- `if (id >= dimension) return;` precedes the `threadgroup_barrier` → for any dimension not a multiple of 256, the last group's tail threads exit while the rest wait: **undefined behavior** (empirically "works" on AGX today, which is exactly how it stays hidden).
- For dimension > 256, each group's `tid==0` computes a total from *its own* shared array — group 0's is the correct full sum (strides cover everything), later groups' cover only their tail — and every group `atomic_store`s to `result[0]`: **last-store-wins, nondeterministically wrong Jaccard distance**.
- Fix: rewrite on the shared reduction helper with one-group-per-result semantics (like `l2_distance` in the same file, which is clean), or per-group `atomic_add` after a host zero.

### VA3-002 (P1, LIVE-cond): BasicOperations tree reductions orphan lanes for non-pow2 threadgroups
`euclideanDistance:218-223`, `squaredEuclideanDistance:250-255` (dead), `dotProduct:341-346`, `cosineDistance:294-301`, `vectorNorm:683-688` (dead) — all use `for (stride = tgSize/2; stride > 0; stride /= 2)` with no ragged-tail guard: for tgSize=100 the element at index 24 (and others on later halvings) never merges → **wrong sum**. Also `partialSums[256]` is written at `[tid]` with no clamp → threadgroup-memory OOB if any caller ever passes tgSize > 256.
- Reachability: `Metal4ComputeEngine` dispatches `cosineDistance`/`dotProduct` with `width = min(256, dimension)` (Metal4ComputeEngine.swift:283-285, 337-339). With `decisionEngine` (default **nil** for direct engine users, Metal4ComputeEngine.swift:99) absent, GPU is taken for `dimension > 16` → **any non-pow2 dimension in 17…255 produces a wrong cosine distance / dot product** on that path. (With a decision engine attached the path is practically unreachable — see routing observation.) `euclideanDistance` is dispatched with width **1** (serial, so accidentally correct; see VA3-024).
- Note the mask: the AUDIT-2 differential harness exercised pow2/embedding dims, where `min(256, dim)` is always a power of two.

### VA3-003 (P1, LIVE): `minkowski_distance_stable` divergent early-return before pass-2 barriers
`MinkowskiDistance.metal:417-423`: `if (max_diff < 1e-8f) { …write 0…; return; }` — `max_diff` is per-(q,n)-pair, i.e. per-*thread*. Any 16×16 tile that mixes a near-identical pair (self-distance, duplicates — routine in all-pairs matrices) with differing pairs has some threads return while others proceed to the pass-2 `threadgroup_barrier`s at :474/:505 → **UB**. Selected automatically for p ∈ (10, 30] (`MinkowskiDistanceKernel.swift:44`) and for any explicit `useStableComputation: true`.
- Fix: replace the early return with a uniform flag (`skip = max_diff < eps`) and let all threads run the barriered loops, writing at the end; also see VA3-022 for the absolute epsilon.

### VA3-013 (P2, LATENT): manhattan/chebyshev correctness is an accident of dispatch
`DistanceShaders.metal:18-81`. `uint tid = id % 256` + hardcoded stride 256 + guard-before-barrier are all wrong in general; the engine's dispatch (`min(256, dimension)` threads, exactly one group — Metal4ComputeEngine.swift:382-384/427-429) happens to make the guard dead and the strides consistent. Any change (dispatchThreads, rounded groups, >1 group) silently reactivates VA3-001-style behavior. Their serial `tid==0` final sums are non-pow2-safe (unlike VA3-002). Harden the kernels to use `threads_per_threadgroup` and the shared helper, or pin the contract with an in-kernel comment + host assert.

### VA3-014 (P2, LATENT): LogSumExp / Statistics reductions — broken pattern with reassuring comments
- `logsumexp_reduce_pass1_kernel` (LogSumExp.metal:164-169, 199-204): `s = tsize/2` halving with `lid+s < tsize` guard **prevents OOB but orphans elements** for non-pow2 tsize. `computeBasicStatistics` (StatisticsShaders.metal:138-147) and `computeHigherMoments` (:220-227) use the same pattern under a comment that literally says "Robustness check for non-power-of-2 tgSize" — the check stops the crash, not the wrong answer. This is the purest instance of the masking pattern this audit hunts.
- `logsumexp_reduce_pass2_kernel` (:241-246, 281-286): hardcoded `s = 128…` tree over `sharedMax[256]` with **no `lid+s < tsize` guard** — dispatch with fewer than 256 threads reads uninitialized threadgroup memory into a max-reduce. Also silently ignores partials beyond index 255.
- Shielded today: host dispatches fixed 256 and caps `numGroups = min(ceil(count/256), 256)` (LogSumExpKernel.swift:407, :368-383). Fix by adopting the shared helper; delete the false comments.

---

## Group B — Numerics & fast-math policy (VA3-006, -007, -012, -015, -016, -022, -030)

**Amortized fix:** extend the AUDIT-2 idioms (dynamic `> FLT_MAX` compares, two-stage `precise::divide`, pre-scaled rescue, NaN-propagating clamp) to the stragglers; centralize EPSILON.

### VA3-006 (P1, LIVE): `batchCosineSimilarity` escaped the VA2-008 remediation
`BasicOperations.metal:626-660`, live via `BatchDistanceOperations.swift:192` (GPU branch of the public batch-similarity API). It still has: naive Float accumulators (overflow → garbage for |x| ≳ 1e19, FTZ-collapse for ≲ 1e-19), denominator `queryNorm * dbNorm` — the exact `sqrt(a)*sqrt(b)` product AUDIT-2 *measured* being reassociated to `sqrt(a·b)` and overflowing — no `[-1,1]` clamp, and a NaN policy that only accidentally matches. The AUDIT-2 memory even listed it among the live cosine kernels; the fix slice covered `cosineDistance`/`cosine_similarity`/`soa`/`batchCosineDistance` but not this one, and the differential harness never drove the `BatchDistanceOperations` entry point. **Port the rescue trio + finalize, and add this dispatch path to DifferentialKernelVsCPUTests.**

### VA3-007 (P1, FIXED in slice 5): silent Chebyshev substitution for p > 10
`MinkowskiDistance.metal:140-143, 240-256, 291-293`: `is_large_p = p > 10` switched the batch kernel to a max-norm "approximation" — no warning, no flag. True Lp exceeds L∞ by up to D^(1/p) (p=11, D=768 → 1.83×). Reachable: p > 30 auto-routed to the batch kernel (stable range was (10, 30], MinkowskiDistanceKernel.swift:44), as did explicit `useStableComputation: false` with p > 10.
**Fixed (owner decision: opt-in, branch kept).** The kernel takes an explicit `chebyshev` flag (buffer 7) and never infers the max-norm from p; `Metal4MinkowskiConfig` gained `chebyshevApproximation` (default false), auto-routes ALL p > 10 to the stable kernel (upper cap removed — unblocked by VA3-003), and treats `useStableComputation` as a pure numerics choice (explicit `false` + large p now runs the batch kernel's exact `safe_pow` path). The `.chebyshev` preset and all four provider `.chebyshev` sites (MinkowskiKernelDistanceProvider + UniversalKernelDistanceProvider, single + batch) carry the flag — their exact-L∞ semantics are unchanged and now *pinned* by a discriminating uniform-difference leg (exact L∞ = 1.0 vs true L100 ≈ 1.069 at D=768, so a dropped opt-in goes red). Red pre-fix in `Hardening/MinkowskiLargePPolicyTests`: p=50 default config returned 1.0 vs true 1.1421; p=11 non-stable returned 1.0 vs true 1.8294. `isChebyshev`/`metricName` now reflect the flag, not `p > 30`. Also reconciles the kernel with `Metal4ComputeEngine.minkowskiDistance`'s CPU path, which always computed true Lp.

### VA3-012 (P2, FIXED in slice 6): EPSILON dual-compile drift — confirmed victims
`KernelContext.swift:461-491` strips four exact spellings of `constant float EPSILON = …;` and rewrites every `VA_EPSILON` token to `EPSILON`. `DistanceShaders.metal:14`'s `#define VA_EPSILON 1e-8f` therefore becomes `#define EPSILON 1e-8f` in the combined TU, redefining the preamble's `1e-7f` for **every file after DistanceShaders in `runtimeCompileShaderFiles`** (4th of 28 — nearly everything is downstream): LearnedDistance `normalizeInPlace`/`batch_normalize_kernel` gates, NeuralQuantization `computeScale`/`normalizeVector`, StatisticsShaders histogram range gate all run with 1e-8 in the release combined build vs 1e-7 in the debug metallib. Small numbers, real divergence — the exact class this audit series exists for. Also: the exact-string strips are whitespace-fragile; one reformat → duplicate symbol → the whole release library fails to build (loud, but a bad way to find out). Fix: give every file `#ifndef`-guarded macros or a single shared constant; delete the per-file `EPSILON` declarations and the string surgery.
**Fixed in slice 6** (see the slice-6 section for the full mechanism): DistanceShaders' shadow renamed `VA_JACCARD_UNION_EPSILON` (1e-8 preserved, file-local both paths), preamble defines `VA_EPSILON 1e-7f` under the real symbol name, token rewrite + dead strips deleted, `Metal4Common.h` guard added, PreambleParityTests parity now on the real symbol. Red two-library divergence and the corpus-wide naming rule pinned by `Hardening/EpsilonCompileParityTests`.

### VA3-015 (P2, FIXED — slice 9): fast-math-fragile idioms outside the cosine family
The following is the original diagnosis. Slice 9 above records the reproduced correlation
failures, preventive histogram/LSE alignment, independent-review correction, and passing
debug/release gates. `Hardening/FastMathPolicyTests.swift` contains the permanent guards.

- `computeCorrelation` (StatisticsShaders.metal:419-422): `sqrt(M2_i) * sqrt(M2_j)` under a comment claiming it avoids the overflow — AUDIT-2 measured this exact claim false (reassociated to `sqrt(a·b)`); huge-variance data silently yields correlation 0.
- Histograms (StatisticsShaders.metal:547, 571, 648, 726): `isfinite(value)` — same fold-risk family as the `isinf()` that AUDIT-2 caught being folded in the plugin metallib; a folded `isfinite` bins NaN/Inf into real buckets. Use `fabs(x) <= FLT_MAX`.
- LogSumExp (:54, 60, 102, 106, 252, 260, 328, 391): `== ±INFINITY` compares — currently un-folded, but the corpus standard after AUDIT-2 is the `> FLT_MAX` dynamic-compare idiom; align.

### VA3-016 (P2, FIXED — slices 10–12): NaN and tie-break policy inconsistencies
**Current status:** LSE/basic-statistics propagation is implemented with the owner's
approved NaN-dominates-infinity policy, including the public basic-statistics API.
See slice 10 for reduction evidence and validation boundaries. Slice 12 completes GPU
Top-K ordering and its CPU chunk merge fallback against VectorCore 0.3.3. Existing
capacity/dispatch limits remain separate; see slice 12. The bullets below preserve the
original diagnosis.

- LogSumExp pass1 (:189) gates `val > -INFINITY` before summing → **drops NaN**; the row-wise kernels propagate NaN. Two entry points, two answers.
- `topk_select_batch_kernel` heap comparisons have no defined NaN semantics and **no index tie-break** (SearchAndRetrieval.metal:39-41, 115-135), while the CPU selection path standardized on VectorCore `TopKSelection` `.smallerIndex` (VA2-007/010). Equal-distance results can order differently CPU vs GPU.
- StatsAggregate min/max silently skip NaN while mean/M2 propagate it (StatisticsShaders.metal:105-122).
The owner approved NaN propagation for reductions and NaN-last ordering with
smaller-original-index ties for Top-K. VectorCore 0.3.3 supplies the CPU reference and is
integrated in slice 11. Slice 12 implements the GPU comparator and verifies both compile
paths, heap membership, sorting, sentinel handling, and public CPU/GPU chunk merging.

### VA3-022 (P2, FIXED — slices 1, 13): `safe_pow` and stable-Minkowski silent distortions
**Current status:** slice 13 implements the owner-approved range contract: no artificial
cutoffs/saturation or approximate-p substitution; explicit fast mode retains intermediate
FP32 limits; stable fractional normalization and exponent-split rescaling cover the
reproduced range failures. Public execution requires finite p > 0. See slice 13 for
evidence and limits. The following paragraph preserves the original diagnosis.
`MinkowskiDistance.metal:35-59`: the exp-clamp (`±87`) silently caps genuinely huge results at ~6e37 where +Inf is the honest answer; `base < 1e-10 → 0` under-reports fractional-p distances (tiny diffs *grow* under p < 1: (1e-9)^0.5 ≈ 3e-5); stable kernel's `max_diff < 1e-8` absolute early-exit (:418) zeroes distances for legitimate micro-scale data — the same absolute-vs-FLT_MIN lesson as BE3 §4.5.

### VA3-030 (P3, FIXED — slice 14): L2/dot split range policy
**Current status:** owner approved automatic exceptional-range rescue for direct rooted
L2, with ordinary FP32 accumulation retained. Squared L2 and dot retain documented range
and cancellation limits. Five Metal entry points and CPU fallback routes now share the
corresponding finalizers. Derived roots of squared scores and learned projection distances
retain their intermediate limits. See slice 14 and
[the range contract](../stability/DISTANCE-RANGE-CONTRACT.md) for scope and verification.
The following paragraph preserves the original decision item.

All L2/dot kernels overflow Σdiff² to Inf for components ≳ 1e19 even when the true (rooted) distance is representable. CPU legs overflow identically, so the differential harness can't see it. Either document the supported range as a contract or port the pre-scaled rescue as the cosine family did. Recording as a decision item, not a defect.

---

## Group C — Memory safety & dispatch contracts (VA3-004, -005, -009, -010, -011, -017, -018, -025, -026, -028)

**Amortized fix:** (a) every kernel guards its own grid bounds; (b) specialized fixed-stride kernels are selected only behind a `stride == dim` guard (copy `L2NormalizationKernel.selectPipeline`, L2NormalizationKernel.swift:220-233); (c) capability caps (`> 128` latent, `> 64` heads, `> 64` nprobe…) become thrown errors host-side and sentinel-fills kernel-side, never silent returns.

### VA3-004 (P1, LIVE): fused batch-GEMM bias is indexed against a layout nobody provides
Shader: `bias[row * N + col]` — an M×N matrix per… nothing (no batch offset either) (OptimizedMatrixOps.metal:235-237). Swift validates `bias.count == batchSize*N || N` (BatchMatrixKernel.swift:406) and passes the array verbatim (:442-447). With the common N-length bias and M > 1: **GPU OOB read**, garbage bias added to rows ≥ 1. With batchSize×N: reads the wrong entries and still overflows when M > batchSize. Decide the semantic (per-column `bias[col]` is the conventional one), fix shader + validation together.

### VA3-005 (P1, LIVE): `computeQuantizationStats` writes out of bounds on large batches
Kernel has no `numVectors` guard (QuantizationShaders.metal:192-229 — the Swift-side `SIMD2` param lands on a `uint&`, so only `dimension` is even visible to it); wrapper dispatches `ceil(numVectors/min(numVectors,1024))` full groups (QuantizationStatisticsKernel.swift:362-376). numVectors = 1500 → 2048 threads → threads 1500-2047 **write** `mse[1500…2047]` past a 1500-float buffer. Add the guard (and pass/read numVectors properly).

### VA3-009 (P1*, LIVE): `threadgroup` initializer race on the decode scale
Nine sites (`NeuralQuantization.metal:542, 634, 711, 787, 881, 1066, 1130, 1192` + `shared_scale` :1373): `threadgroup float tgScale = 0.0f;` — the comment says the initializer exists "to silence uninitialized warning", but a threadgroup-scope initializer is executed by *every* thread, unordered with thread 0's real `tgScale = scales[vectorIdx]` store before the barrier. A late zero-store can win → **scale reads 0 → entire vector decodes to zeros, nondeterministically**. The code already initializes correctly via thread 0 + barrier; the fix is deleting the initializers. (*Confirm the codegen with a probe before ranking the incident risk; the fix is unconditional either way.*)

### VA3-010 (P1, LIVE-cond): stride-blind specialized-kernel selection
- `DotProductKernel.selectPipeline` (DotProductKernel.swift:202-212): picks `dot_product_{384,512,768,1536}_kernel` on dimension alone; `DotProductParameters` has a public init taking explicit `strideQuery/strideDatabase` (:78-87). Strided input at those dims → the kernel hardcodes dense offsets → **silent wrong dot products**.
- `MutualReachabilityKernel.selectPipeline` (MutualReachabilityKernel.swift:223-237): same, and `MutualReachabilityParams` has a public explicit-stride init (:67).
- The corpus already contains the correct pattern: `L2NormalizationKernel.selectPipeline` guards `inputStride == dimension && outputStride == dimension` first. Copy it. (L2Distance/Cosine specialized variants are dead — no exposure there.)

### VA3-011 (P2, FIXED in slice 7): silent capability caps
- Attention: `effectiveHeadDim = min(headDim, 256)` single-head, `min(headDim, 64)` multi-head (AttentionSimilarity.metal:129, 189) — no host validation found; larger heads silently project truncated → wrong similarities. The multihead 64 cap is easy to exceed.
- Learned: `min(projectedDimension, 256)` (LearnedDistance.metal:158, 209).
- Neural: `min(latentDimension, 128)` (NeuralQuantization.metal:167, 258) — **worse than truncation**: the truncated value is then used as the codes/weights row stride, so L > 128 also scrambles the layout.
- IVF builders: `p < 64` loop caps drop probes ≥ 64 silently (IVFCandidateBuilder.metal:61, 224, 294) — no host validation found.
- `topk_select_batch_kernel` K > 128 and `fused_l2_topk` D > 768 / tgs > 256 are silent in-kernel no-ops (output untouched); hosts currently validate (TopKSelectionKernel.swift:386; FusedL2TopKParameters init) but the `encode()` APIs bypass those guards, and `FusedL2TopKKernel.encode`'s K bound is a debug-only `assert` (:315-318).
Policy: throw host-side; sentinel-fill kernel-side.
**Fixed in slice 7** (see the slice-7 section): host throws at every entry point (attention choke, learned computeL2/computeCosine, neural validateCapability ×4 entries, IVF buildCandidates), sentinel/NaN fills in topk_select_batch / fused_l2_topk / the neural kernel family, TopKParameters un-clamped, and the throw-mid-encode encoder leak the guards armed fixed in Metal4Context. Pinned by `Hardening/CapabilityCapPolicyTests` (12 tests, red-first).

### VA3-017 (P2, FIXED — slice 16): unaligned vector-load/store class
`reinterpret_cast<device const float4*>(base + offset)` where `offset % 4 ≠ 0` is possible: general-stride kernels (`l2_distance_kernel`, `dot_product_kernel`/gemv, DistanceShaders `l2_distance`/`cosine_similarity` row offsets when dim ≢ 0 mod 4 — e.g. dim 6 row 1), PQ `calculate_l2_sq_dist` (D_sub ≢ 0 mod 4, odd m), weight rows in LearnedDistance/AttentionSimilarity/NeuralQuantization (inputDim ≢ 0 mod 4, odd row), `va_safe_load_float4`/`ivf_safe_load_float4` helpers, `l2_copy_bits` uint4, neural `char4` (latentDim odd). Works-by-luck on current AGX at best. Two kernels already implement the fix (`kmeans_assign_points` `is_aligned` dual path, ClusteringShaders.metal:43; `neural_encode_pass1`, NeuralQuantization.metal:1267) — the inconsistency proves the hazard is known. Amortize: alignment-guarded dual path or `packed_float4` in general-stride kernels; alternatively host-side `dim % 4 == 0` routing guards.

**Fixed in slice 16:** packed scalar-aligned views replace the casts above, including
scalar threadgroup storage and uint/char vector copies. The original diagnosis is retained
for context. Compiler diagnostics and direct GPU tests cover the contract; the generic
neural weight-tail overread found during the sweep is also fixed. See slice 16.

### VA3-018 (P2, FIXED — slice 15): 32-bit row/address arithmetic class
`tid * stride`, `queryIdx * strideOutput + dbIdx`, `i * params.n + j` computed in `uint` overflow past 2³² elements: realistic at 3 M × 1536 normalize (4.6 G), 66 k × 66 k distance matrices. Offenders: L2Distance, DotProduct, CosineSimilarity(dead), LearnedDistance, AttentionSimilarity, L2Normalization, MutualReachability, UMAP, BasicOperations batch kernels. Already correct (`ulong`/`uint64_t`): DistanceShaders pair kernels, HammingDistance, IVFListSearch, AdvancedTopK, ClusteringShaders k-means, SearchAndRetrieval merge, NeuralQuantization (partial). Plus the known deprecated `streaming_topk_process_chunk` explicit `(uint)global_index_long` truncation (AdvancedTopK.metal:415-416). Amortize: promote row-offset math to `ulong` corpus-wide (mechanical).

**Fixed in slice 15:** row/address products promote before multiplication and preserve
wide locals/helper arguments across the corpus, including additional search, sparse,
Borůvka, and matrix sites found during the sweep. The diagnostic paragraph above records
the original finding. Existing count/grid limits and the deprecated streaming uint-ID
conversion remain separate limitations; see the slice-15 verification and width contract.

**Slice-16 follow-up:** three specialized neural vector output stores were missed by
slice 15. They now promote `vectorIdx * INPUT_DIM` before multiplication, with a red-first
boundary probe; the new generic neural scalar-tail output offset is also tested.

### VA3-025 (FIXED, slice 17) / VA3-026 (FIXED, slice 18) / VA3-028 (FIXED, slice 19): input-contract edges

- **VA3-025 FIXED — slice 17:** vectorized c-TF-IDF previously loaded/gathered/stored
  unused partial-tail lanes (shielded by host `nnz % 4 == 0` routing); raw and standalone
  Top-K lacked a K=0 guard. Scalar tail accesses and zero-K shader/host returns now close
  both defects. Invalid K is rejected; the vector ABI minimum remains 16 bound bytes.
  See slice 17 and [the bounds contract](../stability/SPARSE-TFIDF-BOUNDS-CONTRACT.md).
- **VA3-026 FIXED — slice 18:** raw assignment/accumulation/ADC enforce byte-code bounds;
  ADC rejects tables above 8192 floats before multiplication/shared access. Host ADC throws
  before allocation/encoding, aligns dynamic binding to 16 bytes, and checks device minus
  static pipeline memory. Training/encoding remain available above the ADC-only cap.
  The host initializer's existing K<=256 precondition is retained; execution also rejects
  nonpositive K. See [the PQ bounds contract](../stability/PQ-BOUNDS-CONTRACT.md).
- **VA3-028 FIXED — slice 19:** 2N allocation now has an explicit active-component
  halving proof, backed by independently bounded shader reservations and checked host
  counts. Overflow marks capacity+1 without an out-of-bounds write; fusion callers have
  a checked count reader. All find-min variants, reduction, and collection use endpoint
  validity rather than +infinity as the no-edge signal. See
  [the Borůvka contract](../stability/BORUVKA-BOUNDS-CONTRACT.md).

---

## Group D — Silent-fallback & phantom surface (VA3-021, VA3-031, remainder of -011)

### VA3-031 (P1, FIXED in slice 2): operation-name/function-name conflation hijacked the engine's dotProduct
`PipelineCache.getPipeline(functionName:)` and `ArchivePipelineCache.getPipeline(functionName:)`
both wrap the literal function name in `PipelineCacheKey(operation:)`, so every rewriting case in
`PipelineCacheKey.functionName` hijacks the identically-spelled kernel name. "dotProduct" →
`dot_product_kernel` sent `Metal4ComputeEngine.dotProduct` (and `dotProductWithBuffers`) to the
batch kernel with mismatched bindings — result buffer never written, stale pooled bytes returned
as the answer. Fixed by splitting the languages: batch family operation `"dot_product"`, literal
kernel names identity-mapped. Residual class risk: any future rewriting case whose operation
string equals a kernel name reintroduces this silently — the switch now carries a warning
comment, and the VA3-021 note's proposal (resolve every Swift `makeFunction(name:)`/
`getPipeline(functionName:)` literal in the completeness test) would close the class permanently
(slice 2 found one more such phantom: `"vectorMultiply"`, fallback-shielded).

### VA3-021 (P2, FIXED in slice 8): `tiledTransposeInPlace` — phantom function #6 (of EIGHT found)
`MatrixTransposeKernel.swift:143` optionally loads a function that exists in no `.metal` file → always nil → `canDoInPlace` always false → an `encode(inPlace: true)` request silently runs the out-of-place kernel **writing to the `output` buffer** while the caller expects `input` mutated (:166-176). Either write the kernel or delete the branch and make `inPlace: true` an error. Note: `testCommonPipelineKeysResolveToRealFunctions` can't see this class (optional `makeFunction` at kernel-wrapper level, not PipelineCacheKey); consider extending the completeness test to grep Swift for `makeFunction(name:)`/`getPipeline(functionName:)` literals and resolve them all — that closes the phantom class permanently.

---

## Group E — Determinism & races (VA3-019, part of -027)

### VA3-019 (P2, LIVE)
- **UMAP negative-sampling race FIXED — slice 21:** immutable pass-start targets, sequential per-source updates in distinct output, then synchronized GPU copy-back. Throwing fusion API and reusable scratch overload; raw callers now bind output at buffer(3). See [the contract](../stability/UMAP-NEGATIVE-SAMPLING-CONTRACT.md).
- Relaxed `atomic_float` accumulation orders: UMAP target gradients (:202-223), PQ training (ProductQuantization.metal:103-138), k-means update (ClusteringShaders.metal:492-580) — run-to-run nondeterminism by design; needs a stated determinism policy (the "no atomics" contract in the handoffs is currently false in 10 files).
- **Fused IVF FIXED — slice 20:** bounded whole-query reservations, physical capacity checks, exact-path overflow recovery, and query-ordered CSR conversion. Atomic segment starts were also incorrectly returned as CSR boundaries; this is now covered by scheduling-independent permutation tests. Raw segment order remains unspecified; see [the contract](../stability/IVF-CANDIDATE-BOUNDS-CONTRACT.md).
- UMAP kernel-1 gradient-clip point differs from reference UMAP (clips coefficient *after* lr/weight multiply, not the per-dim gradient) — the never-completed input-clamping backlog item from NUMERICAL_STABILITY_FINDINGS.

### VA3-027 (P2, FIXED — slice 22): silently dropped flags
`neural_encode_pass1` now receives the dispatch activation flag at buffer(8) and applies
ReLU conditionally after bias. Specialized `learned_l2_768_to_128`/`384_to_64` now honor
`normalizeProjected` with the general helper policy. Raw and public regressions cover
both flags and compilation paths; see [the contract](../stability/IGNORED-FLAGS-CONTRACT.md).
The adjacent neural `normalizeLatent` omissions were fixed in slice 23 across specialized
and tiled quantized encoders; see the normalization contract. High-level per-vector scale
preservation was subsequently fixed in slice 24. Slice 22 itself covered the original
two audited flags.

---

## Group F — Deletion inventory (VA3-020, P3) — all DEAD (no Swift dispatch site), verified 2026-08-16

| File | Kernels | Note |
|------|---------|------|
| CosineSimilarity.metal | entire specialized family (5 kernels, ~500 lines) | stale pre-VA2-008 numerics; if ever revived, silently wrong — AUDIT-2 recommendation stands |
| DistanceShaders.metal | `minkowskiDistance`, legacy SIMD trio (`batchCosineSimilaritySIMD`, `batchDotProductSIMD`, `batchEuclideanDistanceSIMD`) | trio's "Retained for Internal Dispatch Compatibility" is false; minkowski returns Σ|d|ᵖ without the root |
| MinkowskiDistance.metal | `minkowski_distance_single` (no root, host-zeroing contract), **`minkowski_distance_fractional`** | fractional's tile load is literally a `// ...` placeholder — it reads uninitialized threadgroup memory; a compiled, exported kernel that returns garbage by construction |
| OptimizedMatrixOps.metal | `fastNormalize` | documented unsound |
| BasicOperations.metal | `batchNormalize2D`, `squaredEuclideanDistance`, `vectorNorm`, `vectorAdd/Subtract/Scale`, `elementwiseMultiply` | batchNormalize2D documented unsound (cross-threadgroup barrier misuse); the rest unreferenced |
| ClusteringShaders.metal | `tiled_kmeans_distance` (**incoherent tile load** — `tid.x` used as both centroid and dimension index, stores a replicated diagonal; wrong distances if ever dispatched), `find_min_assignment`, `assign_to_centroids`, `accumulate_centroids`, `finalize_centroids`, `gpu_reduce_centroids` (per-block semantics unimplemented), `combine_partial_centroids` (**cross-thread race** on `centroid_counts`), `update_centroids_incremental` (RMW race) | the live k-means path (`kmeans_assign_points`, `kmeans_update_*`) is well-built; everything legacy below it is dead and several are broken |
| QuantizationShaders.metal | `scalarQuantize`, `scalarDequantize` (both unguarded), `productQuantize`/`productDequantize` (quantize parameterizes `codebookSize`, dequantize hardcodes 256 — silent garbage if they ever diverged) | superseded by DataTransformations int8/int4 and ProductQuantization.metal; keep only `binaryQuantize`/`binaryHammingDistance`/`computeQuantizationStats` (live) |
| AdvancedTopK.metal | `warp_select_small_k_ascending/descending` (VA3-008), `batch_select_k_nearest_ascending/descending` (thread-0-only), `streaming_topk_init/process_chunk/finalize` (deprecated, ulong→uint truncation) | |
| L2Distance.metal | `l2_distance_384/512/768/1536_kernel` | unreferenced (L2 uses the DistanceShaders pair kernel and `l2_distance_kernel`) |
| AttentionSimilarity.metal | `attention_similarity_768_to_64_kernel`, `batch_attention_similarity_kernel` | never selected (AttentionSimilarityKernel.swift:412 picks only single/multihead) |
| IVFCandidateBuilder.metal | `ivf_prefix_sum_candidates` | the pow2-Blelloch anchor — broken for non-pow2 tg **and** dead (host uses `ivf_prefix_sum_sequential`); delete rather than fix |

Deleting this inventory removes ~2,500 lines and the majority of this ledger's latent-trap surface. Precedent: the owner-approved Manhattan/Chebyshev deletion (VA2-002). All are covered by the tiling/completeness tests, so deletions are mechanical (`runtimeCompileShaderFiles` needs no change unless a whole file goes).

---

## Group G — Hygiene & performance (VA3-023, -024, -029)

- **VA3-023:** `ElementwiseParams.use_fast_math` selects `fast::` vs plain intrinsics, but the whole library compiles `mathMode = .fast` — the "precise" branch is fast-math too. Document or actually build a precise pipeline variant.
- **VA3-024:** engine single-pair `euclideanDistance` dispatches **one thread total** (Metal4ComputeEngine.swift:225-227) — the "parallel reduction" kernel runs serially on one GPU lane, slower than vDSP by orders of magnitude while telemetry reports a GPU hit. `batch_select_k_nearest_*` do all work on `tid == 0` of a full threadgroup. `hamming_distance_single` dispatches 256× the needed threads (HammingDistanceKernel.swift:477-481; harmless atomic adds of 0). `softmax_row_kernel` recomputes the row max+sum per *element*. `boruvka_component_reduce_kernel` is O(N²) serial scans. These are prime suspects for the known UMAP/overall GPU-underperformance observations.
- **VA3-029:** `VA_EPSILON_HALF` is a `float` initialized from a half literal; `va_simd_prefix_sum` returns the *exclusive* sum; `VA_DEBUG_ASSERT` is a non-hygienic `if` macro; helper triplication (`va_*` in Metal4Common vs `ivf_*` in IVFListSearch vs bare in AdvancedTopK — three copies of safe-load/bitonic/heap/is-better) is a drift machine: consolidate behind include-guards once the combined-TU strategy allows.

---

## Re-verified AUDIT-2 anchors that are already sound (no action)

- `IVFListSearch.metal` pow2-merge handles non-pow2 tgs correctly; ulong indexing throughout; barrier discipline clean; host dispatches fixed 256 (mult-of-32, so the simd-shuffle contract holds).
- `fused_l2_topk` merge/selection structure mirrors IVF and is sound; Swift caps fused K at 8 with a two-pass fallback and validates D ≤ 768 in the params init.
- `BasicOperations` normalize core (`va_tg_reduce_*`, `va_normalize_scales`) and all `l2_normalize_*` kernels: dispatch-robust, non-pow2-safe, FTZ-invariant — the reference implementation for Group A.
- `kmeans_assign_points` + `kmeans_update_*`: exemplary guard/barrier pattern; Swift caps tile_capacity at 32 (kernel-side stride-vs-clamp mismatch remains a latent trap if the Swift 32 is ever raised — noted under Group A hardening).
- `minkowski_distance_batch`/`stable` cooperative loads, `hamming_distance_batch`/`normalized` tiling, `merge_topk_sorted_kernel`, `ivf_distance_with_indirection` (model input validation), `parallel_reduce_kernel` (contract documented, host complies), histograms' atomic binning: all correct as dispatched.
- `MatrixMultiplyKernel`/`MatrixTransposeKernel` geometry matches shader tile constants exactly (32/32/8, 16×16+pad).
- BatchMax and SparseTFIDF hosts enforce the %4 vectorization contracts.
