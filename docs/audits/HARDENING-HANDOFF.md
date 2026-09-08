# VectorAccelerate Hardening Epic — Full Handoff

**Written:** 2026-09-04 · **For:** a successor model with zero context on this repository
**Historical state stamp (2026-09-04):** branch `gifton/metal-compute-provider`, 29 commits ahead of `main` (main's tip
is the 0.6.0 release stamp). The ENTIRE epic is **uncommitted working-tree state**: 57 files
changed (+2,063 / −4,717), 9 untracked (the `Tests/.../Hardening/` guard suite and this doc's
siblings). **Commit policy at that stamp:** leave changes uncommitted pending owner authorization.
The owner authorized a branch, commit, and remote checkpoint on 2026-09-07 (see below).
**Verification at stamp:** debug `swift test` → **1590 tests / 0 failures / 11 skipped**;
release `swift test -c release` → **1590 / 0 / 11**. Both must stay that way (or grow).

This document is self-contained, but the authoritative per-finding record is
`docs/audits/AUDIT-3-shaders.md` (findings VA3-001…034, groups A–G, remediation
slices 1–22) and `docs/audits/AUDIT-2.md` (findings VA2-001…013). Read them before deep work.
`docs/audits/REVIEW-PATTERNS.md` holds the project's adversarial-review pattern library.

---

## Current addendum — 2026-09-07

**Slice 9 completed: VA3-015 is FIXED.** Correlation finalization now avoids fast-math
variance-product overflow/underflow and divides by the smaller finite norm first so tiny
representable correlations survive. Histogram finite-value gates and LSE/softmax infinity
branches use dynamic `FLT_MAX` comparisons. Nine new direct-GPU tests cover both compile
paths.

**Slice 10 completed: VA3-016 reductions are fixed.** Input NaNs
dominate both infinities in scalar/vector row LSE and two-pass reduction. Basic statistics
propagate NaN through mean, M2/variance, extrema, sum, standard deviation, and range while
preserving count. `computeBasicStatistics` and basic-only `computeStatistics` now accept
NaN-containing input, including mixed NaN/Inf. Empty input and infinity without NaN still
throw; higher-moment, quantile, and correlation requests retain finite-only validation.
Histograms and softmax are unchanged. Nine further tests cover both shader paths and public APIs.

Current full gates: **debug 1690 / 0 failures / 11 skipped; release 1690 / 0 / 11**.
The original state stamp above is historical. On 2026-09-07, the owner authorized
checkpointing all project work through slice 16 on `gifton/metal-hardening-checkpoint`
and pushing that branch to `origin`. This checkpoint includes the hardening tests, audit
ledgers, contracts, and supporting project files. Local `.antigravitycli/` configuration
links are ignored. Subsequent implementation continues from this branch.

**Slice 11 completed: VectorCore 0.3.3 integration.** The manifest minimum and resolved dependency
now use 0.3.3 (`fca4b602383589c46b627d8a0de2b6b2a68cd07d`). Three new adoption tests
verify NaN-last ordering, heap membership, and original-position ties with pointer IDs.
Targeted tests pass (21 / 0); both full upgrade gates pass (1611 / 0 / 11).

**Slice 12 completed: VA3-016 is FIXED.** GPU Top-K now uses shared FP32 integer ordering
keys so NaNs and subnormals survive fast-math policy handling. Numeric values precede NaNs
in either mode; ties use smaller original indices; padding follows every real candidate.
Batch/warp/general/streaming/fused/IVF selection and chunk merge use this order. The public
CPU chunk merge matches. Empty warp padding and fused winner invalidation are fixed;
a missing fused shared-candidate publication barrier was added. Ten new tests; targeted
56 / 0; full debug/release 1621 / 0 / 11. Existing capacity/dispatch and deprecated
streaming-host issues remain outside this slice; see AUDIT-3 slice 12 for exact boundaries.

**Slice 13 completed: VA3-022 is FIXED.** The owner chose to retain FP32 intermediate
limits for explicit `useStableComputation: false`, with no automatic rescue. Power cutoffs,
exponent clamps, and approximate p=1/2 substitutions are removed. Public execution requires
finite p > 0, including fractional p. Stable normalization uses precise division or log
power differences; final fractional rescaling preserves the scale's significand with
exponent splitting. Defaults remain p > 10 auto-stable and Chebyshev explicit opt-in.
Six new tests; targeted 36 / 0; debug/release 1627 / 0 / 11. Ordinary rounding/FTZ limits
remain; see AUDIT-3 slice 13. No further Minkowski policy decision is pending.

**Slice 14 completed: VA3-030 is FIXED under the split range policy.** Owner approved
exceptional-range rescue for direct rooted Euclidean/L2, retaining ordinary FP32 arithmetic.
Five Metal entry points share a scaled finalizer; CPU fallback routes share Double
recomputation. Squared L2 and dot retain their documented FP32 range/precision limits.
Eight new tests; targeted 169 / 0; full debug/release 1635 / 0 / 11. Scope and retained limits
are in [the distance range contract](../stability/DISTANCE-RANGE-CONTRACT.md), including
squared-score selection/later roots and learned projection distances. Minkowski's explicit
fast-mode contract remains separate. **Group B is closed under these contracts.**

**Slice 15 completed: VA3-018 row/address arithmetic is FIXED.** Products promote before
multiplication and retain wide downstream indices across 21 shader files, including the
normalization bit-copy helper. Ten production-expression GPU boundary tests pass; final
baseline replay reproduces 647 incorrect offsets. Targeted 114 / 0; full debug/release
1645 / 0 / 11. Parameter layouts and count/grid/ID limits remain unchanged; see
[the address-width contract](../stability/INDEX-WIDTH-CONTRACT.md). Group C remains open.

**Slice 16 completed: VA3-017 is FIXED.** Scalar-backed device/threadgroup vector loads
and stores use packed memory views across 15 shader files plus the common header. Fixed
a generic neural weight-tail overread (instrumented red: 20 cases) and three specialized
neural store offsets missed by VA3-018 (red: 15 cases). Eight alignment tests plus one
index test; targeted 33 / 0; shader validation 8 / 0; debug/release 1654 / 0 / 11.
ABI/count/capacity limits remain; no performance claim. See
[the alignment contract](../stability/VECTOR-ALIGNMENT-CONTRACT.md). Group C still has
VA3-025/-026/-028 input/capacity contracts.

**Slice 17 completed: VA3-025 is FIXED.** Vectorized sparse c-TF-IDF processes partial
tails with scalar-bounded accesses; host routing stays divisible-by-four at nnz >= 16.
K=0 is a shader/encoder no-op and returns one empty list per cluster in the standalone
API. Invalid K is rejected. Eight new tests: initial red 50 assertions, targeted 42 / 0,
API+shader validation 8 / 0, full debug/release 1662 / 0 / 11. Vector buffer bindings
retain their 16-byte minimum and alignment; arbitrary sparse-input validation and
nonfinite ranking remain separate. See [the bounds contract](../stability/SPARSE-TFIDF-BOUNDS-CONTRACT.md).
The owner authorized checkpointing this slice on the same branch on 2026-09-07.
`gifton/metal-hardening-checkpoint` is the ongoing branch for all work before and after
the handoff; it includes the full `gifton/metal-compute-provider` ancestry.
Group C remainder: VA3-026/-028.


**Slice 18 completed: VA3-026 is FIXED.** PQ retains all 256 byte codes and a 32 KB
ADC-only table cap. Host ADC rejects oversize requests before allocation/encoding,
rounds shared bindings to 16 bytes, and checks device/static memory. Raw shaders reject
unsupported K/table sizes and invalid codes before unsafe accesses; training/encoding
retain larger-model support. Ten tests: initial red 57 assertions, targeted 47 / 0,
API+shader validation 10 / 0, full debug/release 1672 / 0 / 11. See
[the PQ bounds contract](../stability/PQ-BOUNDS-CONTRACT.md) for sentinels and retained
caller requirements. All work continues on `gifton/metal-hardening-checkpoint`.
**Group C remainder: VA3-028 only.**


**Slice 19 completed: VA3-028 is FIXED.** Borůvka keeps the proven 2N candidate allocation,
with bounded shader reservations and checked host readback. Overflow marks capacity+1;
fusion callers can use `readCandidateCount()` after GPU completion. Invalid endpoints
represent missing edges, so genuine infinite weights remain usable in every find/reduce/
collect path. Ten new tests: initial red 292 assertions, targeted 54 / 0, API+shader
validation 10 / 0, full debug/release 1682 / 0 / 11. Read-only review found no blockers;
capacity is now mandatory in the internal parameter initializer. See
[the Borůvka contract](../stability/BORUVKA-BOUNDS-CONTRACT.md) for raw ABI, initialization,
range and nondeterminism limits. **Group C is closed under its documented contracts.**
All work continues on the single `gifton/metal-hardening-checkpoint` branch.

**Slice 20 completed: VA3-019 IVF portion fixed.** Fused reservations enforce physical
capacity and signal overflow without wrapping; underestimated hints recover through the
exact three-pass builder. Atomic segments are converted into correct query-ordered CSR,
including a GPU reorder when needed. Zero-size dispatch and undersized pooled allocation
hazards were also closed locally. Eight new tests; initial red 446 assertions, targeted
139/0/11, API+shader validation 8/0, full debug/release 1690/0/11; review approved.
See [the IVF candidate contract](../stability/IVF-CANDIDATE-BOUNDS-CONTRACT.md).
**VA3-019 remains open for the UMAP race and atomic accumulation policy.** Same single
`gifton/metal-hardening-checkpoint` branch.


**Slice 21 completed: VA3-019 UMAP negative-sampling race fixed.** Targets are immutable
for each pass; source points still evolve in sample order. Separate output and a GPU
copy-back eliminate cross-thread embedding reads/writes. The owner approved making
`encodeNegativeSampling` throwing, with a reusable-scratch overload. Input counts,
physical lengths and GPU-address overlap are checked before encoding; invalid target
IDs are skipped. Nine new tests; original two-test red 1262 assertions, targeted 40/0,
API+shader validation 9/0, full debug/release 1699/0/11; review approved.
See [the UMAP negative-sampling contract](../stability/UMAP-NEGATIVE-SAMPLING-CONTRACT.md)
for the `try` migration, new raw output binding, scratch lifetime and synchronization.
**VA3-019 remains open for the atomic accumulation policy.**
Numerical/input-clamping backlog remains separate. Same single checkpoint branch.

**Slice 22 completed: VA3-027's two audited ignored flags fixed.** Tiled neural
encoding forwards `useActivation` and conditionally applies ReLU after bias. Specialized
learned L2 paths honor `normalizeProjected` using the general path's normalization
helpers and retain independent square-root selection. Four new tests; original red
5129 assertions, targeted 42/0, API+shader validation 4/0, full debug/release 1703/0/11;
review approved. See [the flag contract](../stability/IGNORED-FLAGS-CONTRACT.md) for the
new raw neural pass-1 buffer(8) UInt32 flag and retained numerical/layout limits.
Dedicated `encodeTiledV3` coverage is now present. Source inspection found adjacent
`normalizeLatent` omissions in specialized/tiled neural quantized encoders; recorded
below as follow-up, not silently included in this slice. VA3-019 atomic policy remains
open. Same single checkpoint branch.

## 1. What this project is

VectorAccelerate (VA) is the GPU-acceleration package of the VSK suite: Metal 4 compute
kernels for vector search and ML primitives (distance metrics, top-k selection, IVF
indexing, quantization — scalar/product/neural, clustering, UMAP, attention/learned
distances, statistics, log-sum-exp), exposed to Swift through kernel wrapper classes,
`Metal4ComputeEngine`, and VectorCore `DistanceProvider` implementations. Sibling packages:
VectorCore 0.3.3 (CPU/SIMD ground truth — its `TopKSelection`, normalize kernels, and BE3
numerics policy are the parity reference), VectorIndex (pins VA 0.3.1 — not in play here).

- Shader corpus: **27 `.metal` files** in `Sources/VectorAccelerate/Metal/Shaders/` +
  `Metal4Common.h` (shared constants/helpers, e.g. `VA_EPSILON`, `VA_NORM_*`, the cosine
  rescue trio under `VA_COSINE_RESCUE_DEFINED`).
- Swift: `Core/` (Metal4Context, Metal4ComputeEngine, KernelContext, PipelineCache/Key,
  PipelineRegistry, GPUDecisionEngine), `Kernels/Metal4/` (per-kernel wrappers),
  `Integration/` (MetalComputeProvider, KernelDistanceProviders), `Index/` (IVF pipeline).
- Tests: `Tests/VectorAccelerateTests/`, including `Hardening/` — **24 permanent guard
  suites** created by this epic (§5).

## 2. Architecture facts you must internalize first

**2.1 The dual shader-compile paths (the single most important fact).** There are TWO ways
kernels reach the GPU, and they MUST behave identically:
- **Debug builds:** `debug.metallib`, produced per-file by MetalCompilerPlugin, `#if DEBUG`
  gated. Each `.metal` file compiles alone with `#include "Metal4Common.h"`.
- **Release builds:** SPM ships **no** metallib. `KernelContext.makeLibraryFromBundleSources`
  concatenates the files in `KernelContext.runtimeCompileShaderFiles` into ONE translation
  unit, strips the `#include "Metal4Common.h"` lines, and prepends
  `KernelContext.runtimeCompilePreamble` — a hand-maintained mirror of the header. This is
  the ONLY path release users execute. One compile error strands ALL ~180 kernels; one
  numeric drift between preamble and header silently forks debug/release behavior.
  History: a file missing from the list made release throw on init for all of 0.6.0
  (VA2-001); the epsilon token-rewrite forked gate values between builds (VA3-012).
  Guards: `PreambleParityTests` (numeric identity + cosine-rescue byte identity),
  `ShaderLibraryCompletenessTests` (directory↔list tiling, per-kernel presence,
  metallib↔runtime parity), `EpsilonCompileParityTests` (behavioral two-library
  differential). **Any shader or preamble edit requires the release gate, not just debug.**
- Corollary (VA2-013, open): MetalCompilerPlugin does NOT track header dependencies —
  editing only `Metal4Common.h` ships a stale `debug.metallib`. Workaround: `touch
  Sources/VectorAccelerate/Metal/Shaders/*.metal` before building.

**2.2 Metal4Context execution core** (`Core/Metal4Context.swift`): `execute`,
`executeAndWait`, `executeBlitAndWait` create a command buffer + encoder, run your closure,
end encoding, commit, wait on a shared `MTLSharedEvent`. Since slice 7, a **throwing
closure gets its encoder ended and the error propagates** (before that fix, Metal API
validation aborted the process — "Command encoder released without endEncoding").
Still-open anchors in this file: residency sets tracked but never attached; per-call
unretained `MTLSharedEventListener`; shared-event `signaledValue` set from completion
handlers of potentially concurrent command buffers (non-monotonic risk). `VA_AUDIT_TRACE=1`
env prints a line per GPU submit.

**2.3 Silent-fallback masking (why tests lie here).** `MetalComputeProvider.Configuration
.fallbackToCPU = true` (default) swallows GPU errors into CPU results. `GPUDecisionEngine`
defaults route small-N to CPU (minVectorsForGPU=1000 etc.), so naive tests never touch GPU
paths; `Metal4ComputeEngine` single-pair ops CPU-route below dimension thresholds. Many
tests `XCTSkip` without Metal. Consequence: **a green suite does not prove GPU kernels
run** — that is the masking pattern this whole epic hunts. `RoutingProvenanceTests` +
`MetalComputeProvider.RoutingTelemetry` exist to pin which leg actually executed; the
`DifferentialKernelVsCPUTests` harness compares GPU vs CPU legs on adversarial inputs.

**2.4 Naming routes from Swift to kernels — all mechanically checked now.** Three routes,
each once a source of "phantom" names that existed in no library: (1) `PipelineCacheKey`
operation→functionName derivation (VA2-006, VA3-031 — literal names are identity-mapped;
NEVER add a rewriting case whose operation string equals a kernel name); (2)
`getPipeline(functionName:)` literals; (3) `makeFunction(name:)` literals (slice 8 found
EIGHT phantoms here). `ShaderLibraryCompletenessTests` closes all three routes with no
makeFunction allowlist and a shrinking getPipeline allowlist (currently: `vectorMultiply`).

**2.5 Fast-math is on everywhere** (`mathMode = .fast`, including the runtime compile).
Measured consequences already found: `sqrt(a)*sqrt(b)` reassociated to `sqrt(ab)` (overflow
→ silent 0); `(d/nA)/nB` rewritten to `d·rcp(nA·nB)` (subnormal reciprocal flush — fix is
`precise::divide`); `isinf()` folded to false in the plugin metallib but not runtime
(fix: `x > FLT_MAX` dynamic compares). These idioms are the corpus standard; extending them
to the VA3-015 stragglers was completed in slice 9, including smaller-norm-first division.

## 3. How this epic works (follow these conventions exactly)

1. **Owner-directed slices.** The owner (gifton) picks the next finding; you execute one
   bounded slice. Policy decisions (delete vs opt-in, NaN semantics, API breaks) are the
   owner's — surface options with a recommendation, wait for the call. If a slice looks
   massive, present the plan before executing (owner's standing request).
2. **Red-first TDD, mechanism-exact.** Every fix starts with a failing test that reproduces
   the *mechanism* (wrong value, stale poison bytes read back, silent substitution), run
   and observed red BEFORE the fix. Poison output buffers (`0xDEADBEEF` / `12345.0`) to
   expose no-op paths. At-cap / control legs must be green before AND after. Prefer tests
   that discriminate (e.g. data where the honest and the buggy answer differ in FP32).
3. **Gates.** After the fix: targeted suites, then FULL debug suite, then FULL release
   suite (`swift test -c release`) — release exercises the runtime-compiled library
   (§2.1). Record exact counts. Current expectation: 1690/0/11 both configs.
4. **Ledger + memory.** Append a "Remediation slice N" section to
   `docs/audits/AUDIT-3-shaders.md`, flip the finding's summary-table row and detail
   heading to **FIXED**, and append a dated paragraph to the session memory file
   (`~/.claude/projects/-Users-goftin-dev-gsuite-VSK-future-VectorAccelerate/memory/
   va-hardening-audit-focus.md`) + refresh `MEMORY.md`'s one-line hook.
5. **Claim discipline** (from the owner's global CLAUDE.md): any robustness claim in a
   comment must name the covering test, be one-step derivable, or be explicitly marked an
   unverified contract. Second occurrence of a defect anywhere = add the mechanical test
   that closes the CLASS, not just the instance. When a routing/config gate changes,
   re-derive reachability for everything it shielded (this has bitten twice — see §8).
6. **Commit policy.** The owner decides commits. The 2026-09-07 branch/commit/push
   checkpoint is explicitly authorized; obtain owner direction for later checkpoints.

## 4. History (what has been done)

**AUDIT-2 (2026-08-16, `docs/audits/AUDIT-2.md`, VA2-001…013).** Phase 0/1 hardening pass.
Headlines: release suite was RED at pristine HEAD (VA2-001, SoADistance missing from the
runtime list); four phantom cache-key derivations (VA2-006); provider batchDistance GPU
path unreachable under defaults (VA2-003); GPU cosine misclassified huge/tiny/mixed-scale
vectors via naive FP32 norms (VA2-008) with the measured fast-math mechanisms of §2.5; the
cosine "rescue trio" (unreliable-accumulator detect → pre-scaled recompute → NaN-propagating
clamped finalize) now lives in `Metal4Common.h` + byte-identical preamble mirror; NaN and
tie-break policy standardized on the CPU side to VectorCore `TopKSelection` `.smallerIndex`
(VA2-007/-010); dead Manhattan/Chebyshev specialized shader files deleted (VA2-002);
`batchCosineDistance` — requested at 3 engine sites — had NEVER existed (kernel written).

**AUDIT-3 (2026-08-16, VA3-001…034).** Independent end-to-end read of all shader files with
per-finding Swift dispatch/liveness verification, organized into groups: A reduction/barrier,
B numerics/fast-math policy, C memory-safety/dispatch contracts, D phantom surface,
E determinism, F deletion inventory, G hygiene. Then twenty-two remediation slices:

| Slice | Date | Scope | Highlights | Gate |
|---|---|---|---|---|
| 1 | 08-16 | Group F + all P1s | jaccard barrier-UB rewrite (VA3-001); stable-Minkowski uniform degenerate flag (VA3-003); batch-GEMM bias layout enum (VA3-004); quantization-stats grid guard (VA3-005); batchCosineSimilarity rescue port (VA3-006); warp-select sentinel padding (VA3-008, was LIVE); 9 threadgroup-initializer races removed (VA3-009); DotProduct stride guard (VA3-010); ~2,400 lines dead shaders deleted (Group F); pre-existing BufferPool flake fixed | 1558/0 ×2 |
| 2 | 08-16 | VA3-002 | BasicOperations single-pair tree reductions made width-robust; **VA3-031 found: cache-key operation/function conflation sent engine dotProduct to the batch kernel — stale pool bytes returned at EVERY dim**; literal names identity-mapped | 1561/0 |
| 3 | 08-16 | VA3-013/-014 (Group A done) | manhattan/chebyshev/LSE/statistics reductions rewritten dispatch-robust; false "robustness" comments deleted; 4th phantom literal found | 1567/0 |
| 4 | 08-16 | Adversarial meta-review of the epic itself | 3 live defects IN/ARMED-BY the epic fixed: VA3-032 (batch dot/manhattan GPU branches dispatched phantoms — armed by our own VA2-003 gate change), VA3-033 (cosine sub-100 CPU leg pre-rescue), VA3-034 (encodeWarp k>32 wrote nothing); every uncovered claim from slices 1–3 closed empirically; completeness tests extended | 1577/0 |
| 5 | 08-24 | VA3-007 | `minkowski_distance_batch` silently substituted Chebyshev for p>10 (error up to D^(1/p); measured 1.0 vs true 1.83 at p=11/D=768). Owner chose **opt-in**: explicit `chebyshevApproximation` flag at buffer(7); all p>10 auto-route to the stable kernel; `.chebyshev` preset + 4 provider sites carry the flag, pinned by a discriminating uniform-diff leg | 1583/0 |
| 6 | 08-24 | VA3-012 | EPSILON dual-compile drift: DistanceShaders' `#define VA_EPSILON 1e-8f` shadow + KernelContext's token rewrite poisoned every downstream file to 1e-8 in RELEASE while debug ran 1e-7. Red = live two-library divergence ([16,0] vs [8,8] histograms). Fix: single epsilon authority — shadow renamed `VA_JACCARD_UNION_EPSILON`, preamble defines `VA_EPSILON` under its real name, token surgery deleted. Bonus: PreambleParityTests' own mapping was a latent silent-skip (`dict[k]=nil` removes the key) | 1586/0 |
| 7 | 08-28 | VA3-011 | Silent capability caps → host throws (attention 256/64, learned 256, neural 128 ×4 entry points, IVF nprobe 64) + kernel sentinel/NaN fills (topk batch K>128, fused_l2_topk D>768, neural family). **Found beyond inventory:** `TopKParameters.init` silently clamped k=min(k,128); and the new guards ARMED the dormant throw-mid-encode encoder leak → process abort — fixed in Metal4Context (§2.2) | 1598/0 |
| 8 | 09-04 | VA3-021 | makeFunction-literal completeness test went red with **8 phantoms** (audit knew 1): transpose in-place (now throws; param deleted), matmul specialized trio, unread batchMatrixVector, neural tiled V1/V2 trio whose 3 public APIs only threw and whose **10 tests had skipped since birth** (~half the standing 21 skips). All deleted (~880 test lines); `encodeTiledV3` (real pass1/pass2 kernels) is the shipping tiled path | 1590/0/11 |
| 9 | 09-05 | VA3-015 | Correlation denominator overflow/underflow fixed with smaller-norm-first precise division; histogram/LSE/softmax dynamic comparison alignment; nine direct-GPU tests, both compile paths | 1599/0/11 |
| 10 | 09-06 | VA3-016 reductions | Owner-approved NaN propagation for LSE/basic statistics, integer flags through GPU reductions, explicit infinity partials, public basic-statistics validation/derived-value alignment; Top-K remains pending | 1608/0/11 |
| 11 | 09-06 | VectorCore 0.3.3 integration | Manifest/lockfile upgrade; three CPU adoption tests cover NaN-last ordering, heap membership, and pointer-ID ties; GPU Top-K remains pending | 1611/0/11 |
| 12 | 09-06–07 | VA3-016 GPU Top-K | Shared integer comparator across selection/merge paths; NaN-last/index ties; empty warp padding, fused winner invalidation + barrier; CPU chunk merge alignment | 1621/0/11 |
| 13 | 09-07 | VA3-022 Minkowski | Owner retained explicit fast-path limits; removed clamps/cutoffs and approximate powers; stable fractional normalization/rescaling + finite-p validation | 1627/0/11 |
| 14 | 09-07 | VA3-030 L2/dot policy | Direct rooted-L2 exceptional-range rescue; squared L2/dot retain documented FP32 limits; eight boundary/routing tests | 1635/0/11 |
| 15 | 09-07 | VA3-018 address width | Promote row products before multiplication; wide downstream locals/helper arguments across 21 shaders; ten GPU arithmetic tests, retained ABI limits | 1645/0/11 |
| 16 | 09-07 | VA3-017 alignment | Packed scalar-backed vector views; neural weight-tail bounds and three missed store offsets; compiler/footprint guards, odd-layout GPU tests | 1654/0/11 |
| 17 | 09-07 | VA3-025 sparse TF-IDF bounds | Scalar vector tails; zero-K no-op; invalid K guard; eight bounds/API regression tests | 1662/0/11 |
| 18 | 09-07 | VA3-026 PQ bounds | Byte-code guards, ADC-only 32 KB cap and aligned binding, invalid-code isolation; ten regression tests | 1672/0/11 |
| 19 | 09-07 | VA3-028 Borůvka bounds | Bounded reservations/readback; endpoint validity preserves infinite edges; geometric-bound and fusion regressions | 1682/0/11 |
| 20 | 09-07 | VA3-019 IVF portion | Physical capacity guards, overflow recovery, valid CSR ordering, prefix saturation and empty handling; UMAP/policy remain open | 1690/0/11 |
| 21 | 09-07 | VA3-019 UMAP race | Immutable targets, sequential source updates, separate output/copy-back, throwing fusion and reusable scratch; atomic policy remains open | 1699/0/11 |
| 22 | 09-07 | VA3-027 audited flags | Tiled activation and specialized learned normalization honored; raw/public regression coverage; adjacent normalizeLatent debt recorded | 1703/0/11 |

**Status:** every P1 fixed; groups A, B, C, D, F closed (B/C retain documented contracts
and limits). Groups E/G and the recorded residuals remain open (§6).

## 5. The permanent guard suites (`Tests/VectorAccelerateTests/Hardening/`)

| File | Class it closes |
|---|---|
| `ShaderLibraryCompletenessTests` | phantom names: directory↔list tiling, every kernel present in the runtime library, metallib↔runtime kernel-set parity, cache-key derivations, getPipeline literals, makeFunction literals (no allowlist) |
| `PreambleParityTests` | header↔preamble numeric identity; cosine-rescue block byte-identity; every VA_* symbol used is provided |
| `EpsilonCompileParityTests` | behavioral two-library differential (debug metallib vs runtime compile) on an epsilon-gated kernel; single-epsilon-authority rule |
| `RoutingProvenanceTests` | which leg (GPU/CPU) actually executed, via RoutingTelemetry |
| `DifferentialKernelVsCPUTests` | GPU vs CPU value parity across ten adversarial input classes on live dispatch paths |
| `TreeReductionDispatchRobustnessTests` | non-pow2 threadgroup widths on the BasicOperations reductions (VA3-002) |
| `LatentReductionGeometryTests` | adversarial dispatch geometries on manhattan/chebyshev/LSE/statistics (VA3-013/-014) |
| `WarpSelectionPaddingTests` | sentinel padding of warp-select output tails and over-cap k (VA3-008/-034); establishes the sentinel convention (±INF by mode, 0xFFFFFFFF) |
| `SpecializedKernelStrideGuardTests` | `stride == dim` guard before selecting fixed-stride specialized kernels (VA3-010) |
| `MinkowskiLargePPolicyTests` | true-Lp vs opted-in Chebyshev semantics (VA3-007) |
| `CapabilityCapPolicyTests` | over-cap throws + sentinel fills + throw-mid-encode propagation (VA3-011) |
| `FastMathPolicyTests` | finite correlation denominator extremes and division order; histogram nonfinite exclusion; LSE/softmax infinity branches on both compilation paths (VA3-015) |
| `NaNReductionPolicyTests` | NaN-dominates-infinity LSE/basic statistics across payloads, positions, lanes/groups/strides, both compile paths, and public API boundaries (VA3-016 reduction portion) |
| `TopKNaNPolicyTests` | NaN-last/index-tie membership and ordering, subnormals, sentinels, streaming/chunk merges, fused/IVF selection in both compile paths, public CPU/GPU merge agreement (VA3-016 Top-K portion) |
| `MinkowskiRangePolicyTests` | Tiny/fractional distances, exact p specializations, explicit fast range limits, stable ratio/root rescaling, FLT_MAX endpoints, public finite-p validation; both compile paths (VA3-022) |
| `EuclideanRangePolicyTests` | Direct rooted-L2 huge/tiny rescue, boundary/nonfinite values, tails/widths/SoA stride, CPU/GPU routes, mapped search, retained squared/dot limits; both compile paths (VA3-030) |
| `IndexWidthTests` | Production-source GPU arithmetic probes at 2^32, batch/strided products, and downstream narrowing (VA3-018); no large-buffer allocation claim |
| `VectorAlignmentTests` | Compiler alignment diagnostic including reinterpret-cast canary, odd-layout GPU checks, and instrumented neural weight-tail footprints (VA3-017) |
| `SparseTFIDFBoundsTests` | Partial-tail footprints/canaries, exact allocations within vector ABI, zero-K shader/encoder/public behavior, invalid K, sentinels and over-dispatch (VA3-025) |
| `PQBoundsTests` | Byte-code endpoints, invalid assignment/lookup isolation, ADC table cap/overflow, early host rejection, binding alignment, and retained larger-model train/encode (VA3-026) |
| `BoruvkaBoundsTests` | Capacity/overflow/wrap guards, endpoint validity and infinity across all variants, fusion count safety, geometric bound with complete merging (VA3-028) |
| `IVFCandidateBoundsTests` | Bounded whole-query reservations, skewed-list recovery, physical storage guards, prefix saturation and scheduling-independent CSR reordering (VA3-019 IVF portion) |
| `UMAPNegativeSamplingTests` | Frozen target reads, sequential source updates, raw output guards, concurrent fusion, scratch reuse/lifetime, epoch ordering, and invalid input rejection (VA3-019 UMAP race) |
| `IgnoredFlagTests` | Tiled ReLU on/off after bias, signed quantized outputs, specialized learned normalization/root flags, guards and both compilation paths (VA3-027) |

The 11 skips in the current Apple Silicon gates are explicit unimplemented
`IVFValidationTests` placeholders (including the missing retrieval API), not environment
gates. Slice 20 corrected the previous misclassification after inspecting the test logs.
The separate ten-test phantom neural class was deleted in slice 8.

## 6. Remaining work (the honest open list)

**Group B — numerics & fast-math policy (closed with documented limits):**
- **VA3-015 (P2, FIXED — slice 9):** correlation finalization uses smaller-norm-first
  precise division; histogram/LSE/softmax comparisons use dynamic finite-limit gates.
  Correlation failures reproduced; histogram/LSE changes are preventive alignment.
- **VA3-016 (P2, FIXED — slices 10–12):** LSE/basic-statistics NaN propagation and
  GPU Top-K NaN-last/index ties are complete against VectorCore 0.3.3. Integer ordering
  preserves subnormal scores; public CPU/GPU chunk merges agree. Existing selection
  capacity/dispatch limitations are separate (slice 12 boundaries).
- **VA3-022 (P2, FIXED — slices 1, 13):** removed artificial power cutoffs/clamps and
  approximate specializations; stable fractional normalization/rescaling hardened. Owner
  explicitly retained fast-mode intermediate range limits with no automatic rescue.
  Public p must be finite and positive; ordinary FP32 rounding/FTZ limits remain.
- **VA3-030 (P3, FIXED — slice 14):** direct rooted L2
  rescues exceptional accumulators on CPU/GPU; squared L2 and dot retain documented FP32
  limits. This includes limits on derived roots of squared scores and learned projection
  distances; see the distance range contract. No performance/full-range guarantee.

**Group C — closed under documented contracts.** P1 memory-safety fixes, capability
caps, VA3-018 address width, VA3-017 alignment, VA3-025 sparse TF-IDF, VA3-026 PQ bounds,
and VA3-028 Borůvka bounds are complete. Raw caller storage/layout, numerical-range,
count/ID, and synchronization requirements remain as documented in the contracts.

**Group E — determinism (needs OWNER POLICY):**
- **VA3-019 (P2, LIVE):** the remaining part is relaxed `atomic_float` accumulation
  order in UMAP target gradients, PQ training and k-means update. Run-to-run numerical
  variation needs a stated policy (accept + document, or rework); older broad "no atomics"
  claims are false. **The UMAP negative-sampling race is fixed in slice 21** through
  immutable targets, separate output and synchronized publication. See
  [the UMAP contract](../stability/UMAP-NEGATIVE-SAMPLING-CONTRACT.md).
- **Fused IVF portion fixed in slice 20:** bounded physical capacity, exact-path
  overflow recovery, and valid query-ordered CSR conversion. Raw atomic segment order
  stays unspecified; see [the contract](../stability/IVF-CANDIDATE-BOUNDS-CONTRACT.md).
- **VA3-027 (P2, FIXED — slice 22):** the two audited flags are honored: tiled neural
  `useActivation` and specialized learned-distance `normalizeProjected`. See
  [the contract](../stability/IGNORED-FLAGS-CONTRACT.md). The adjacent neural
  `normalizeLatent` omissions are recorded separately below.

**Group G — hygiene/perf:** VA3-023, VA3-029 (see ledger), and **VA3-024**: engine batch
dispatch uses 16×16 threadgroups for 1-D work (16× redundant compute in some batch paths)
+ the engine width-1 euclidean dispatch oddity. Perf-only; benchmark before/after.

**Recorded-not-fixed (slice-4 residuals, in the ledger):** `getPipelineState` residual
slot; `BatchMatrixKernel.encode` bias-length validation on the raw encode path; telemetry
blind spot for custom metrics; `PipelineCacheKey.quantized` derives only phantom names
(zero callers — DELETE on next touch, plus its derivation-test assertion); AccelerateFallback
ragged-pair asymmetry (euclidean→+Inf vs cosine→NaN, provider-unreachable).

**Coverage gaps / debt:** global buffer-pool requests above 64 MiB may receive undersized
storage (IVF now checks/rejects this locally; broader pool correction remains open);
11 unimplemented `IVFValidationTests` placeholders; neural `normalizeLatent` is read only
by the generic `neural_encode_quantize_kernel`, with specialized/tiled quantized encoders
ignoring it (source-confirmed in slice 22; behavioral regression/fix pending). The
previous `encodeTiledV3` dedicated-test gap was closed in slice 22.
VA2-013 plugin header-dep gap (workaround in §2.1); the CI leg for the release gate was
deferred by the owner (AUDIT-2 decision 5); UMAP GPU benchmark underperforms expectation
(0.6–1.6× vs 2–5×); `docs/stability/NUMERICAL_STABILITY_FINDINGS.md` backlog (UMAPGradient
input clamping, AttentionSimilarity stable sigmoid, LSE partialMax≤globalMax invariant);
deprecated `StreamingTopKKernel` still ships (its kernel truncates a `ulong` index to
`uint`); README "Known Limitations" doc drift.

## 7. Meta-lessons — the failure patterns to hunt (all observed here, most twice)

1. **Comments that claim robustness ARE the bug's camouflage** ("Robustness check for
   non-power-of-2 tgSize" guarded the crash, not the wrong answer). Treat every such claim
   as a test obligation.
2. **A correct gate change arms dormant defects.** Twice: VA2-003's k-gate exemption armed
   phantom dispatches (VA3-032); slice 7's guards armed the throw-mid-encode abort.
   After ANY routing/validation change, re-derive reachability of everything downstream.
3. **Phantom names accumulate wherever strings name kernels** — and optional loads
   (`if let makeFunction`) are worse than throwing ones: permanently-nil pipelines behind
   silent fallbacks. All three routes are now test-closed; keep them closed.
4. **Forever-skipping guard-gated tests read as coverage.** A skip that can never un-skip
   is worse than no test (10 of them hid the neural phantom trio for its whole life).
5. **Silent clamps in params types** (`k = min(k, maxK)`) defeat every downstream guard —
   validate-or-preserve, never quietly rewrite the request.
6. **Two compile paths = every numeric constant is a fork risk.** Single authority + parity
   tests + a behavioral two-library differential.
7. **Exact-string liveness greps miss prefix-named kernels** — always suffix-wildcard; and
   **anchor-count asserts on every scripted edit** (a "2-site" fix found a 3rd loadWeights
   variant only because the edit script asserted counts).
8. **Test-shim rot:** a compatibility mapping in a test (`dict[k] = old[k2]`) can silently
   become a check-deleter when the underlying feature changes (`dict[k]=nil` removes keys).
9. **Poison-buffer reads are the only honest test for "kernel wrote nothing"** paths.
10. **CPU and GPU legs overflowing identically hides overflow from differentials** — pick
    adversarial ranges where the reference is computable in Double closed form.

## 8. Reference map

- Ledgers: `docs/audits/AUDIT-3-shaders.md` (authoritative; slices 1–22 + all findings),
  `docs/audits/AUDIT-2.md`, `docs/audits/REVIEW-PATTERNS.md`.
- Plans: `docs/superpowers/plans/2026-08-16-hardening-audit-phase0-1.md` (epic origin).
- Numerics backlog: `docs/stability/NUMERICAL_STABILITY_FINDINGS.md`.
- Session memory (Claude-side): `~/.claude/projects/-Users-goftin-dev-gsuite-VSK-future-
  VectorAccelerate/memory/va-hardening-audit-focus.md` — dated per-slice records mirroring
  the ledger, plus `MEMORY.md` index. Keep both updated per slice (§3.4).
- Key source files: `Core/KernelContext.swift` (runtime compile list + preamble),
  `Core/Metal4Context.swift` (execution core), `Core/PipelineCacheKey.swift` (name
  derivation — beware rewriting cases), `Metal/Shaders/Metal4Common.h` (shared constants +
  cosine rescue; mirrored in the preamble).

**Suggested next slice:** VA3-019 atomic accumulation policy (owner decision). The IVF
bounds/CSR and UMAP negative-sampling race portions are complete; VA3-027's two audited
flags are also fixed. Remaining concrete work includes neural `normalizeLatent` parity,
G hygiene/performance and the recorded residuals. Group C has no remaining
numbered findings. Let the owner select the next slice; do not infer approval to broaden this one.
