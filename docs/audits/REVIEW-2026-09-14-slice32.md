# /audit report — slice 32 (commit `39bb404`, five IVF placeholders) — light tier

**Reviewed:** 2026-09-14 · **Scope:** `git diff f685660..39bb404` (`IVFValidationTests.swift`
+190/−24; `AUDIT-3-shaders.md` +58; `HARDENING-HANDOFF.md` +45/−9) plus the full post-change
units it exercises (`AcceleratedVectorIndex`, `IVFStructure`, `IVFSearchPipeline`,
`IVFListSearch.metal`, `HandleAllocator`, `GPUVectorStorage`). Reviewer made no production,
test, or shader edits. Independent verification run: the five tests 5/0 in debug, and the three
clustering-dependent tests green in five consecutive invocations
(`scratchpad/ivf-review-run.log`, HEAD `39bb404`, 18:30 PDT).

## Verdict

Approve. No P1 or P2 findings. The five tests have real teeth: each has a nameable one-token
production regression that turns it red, three of those were fault-injected by the agent
(`mutations.py`: duplicate append, off-by-one CSR offset, threshold-zero rerouted to flat,
shifted read slot, doubled score) and the remaining reverts are derived below. Tolerances are
honest, all ledger claims reconcile with the evidence logs and the code, and the
"no production change" claim is verified from the commit range. Three P3 items are worth a
small follow-up slice, and one new masking pattern should be appended to `REVIEW-PATTERNS.md`.

## Findings

### P1 — none

### P2 — none

### P3

**P3-1 · Unseeded k-means++ under a cluster-shape assertion** · LIVE (flake risk only)
- `Tests/VectorAccelerateTests/IVFValidationTests.swift:1886` (`testRoutingThresholdZeroForcesIVFSearch`)
  asserts `0 < results.count < 32` for a one-list probe. That holds only if training yields
  at least two non-empty lists and the query's list is a strict subset.
- Mechanism: `KMeansPlusPlusKernel.swift:131,171` seed with `Int.random`/`Double.random`;
  nothing in the test pins initial centroids. The fixture (four groups 64 apart, spread ≤ 7)
  makes a degenerate single-list outcome practically unreachable under D²-weighted seeding,
  and five repeat runs were green, but the property is probabilistic, not proven.
- Class: pre-existing. `testNoEmptyClustersAfterTraining` and `testClusterBalanceIsReasonable`
  already depend on the same unseeded initializer.
- Proposed closure: a `seed:` on `KMeansConfiguration` (or an injectable RNG) so every IVF
  test that asserts on cluster shape is reproducible. Owner decision; not a slice-32 defect.

**P3-2 · Adjacent: CSR serializer silently skips out-of-range slots** · LATENT (production,
not in the diff; the new CSR test inspects this serializer but not this branch)
- `Sources/VectorAccelerate/Index/Internal/IVFStructure.swift:445`
  `if slotIndex < 0 || slotIndex >= storage.allocatedSlots { continue }` under a comment
  "shouldn't happen in normal flow". When it does happen, `vectorIndices[i]` still names the
  slot while `listVectors` row `i` keeps whatever `makeBuffer(length:)` returned, so the kernel
  scores garbage and can return that slot as a hit (pattern 5, stale-bytes readback, plus
  pattern 7, silent fallback). Sibling: `compact(using:)` line 487 drops unmapped entries via
  `compactMap` with no signal.
- Reachability today: none. Removal and the compaction remap keep list slots in range. The
  arming scenario is any future path that populates lists without `compact(using:)` (e.g. a
  recovery path restoring lists from persisted slots).
- Proposed fix (separate slice): throw `IndexError.bufferError` instead of `continue`, and
  add a list-slot validity invariant test (every slot in every list is `< allocatedSlots` and
  not deleted) after remove/compact sequences. The "shouldn't happen" comment is an
  unverified claim under the CLAUDE.md claim discipline.

**P3-3 · "Exact tie" leg discriminates less than the ledger wording implies** · note
- `testEndToEndWorkflow` query `dataset[7]` ties handles 3 and 11 at 1.5625 (verified,
  binary-exact). The kernel breaks ties on `.index`, which is the gathered slot
  (`IVFListSearch.metal:52,258`). Slot order, handle order, and CSR position order all agree
  for this fixture because no deletion or compaction precedes the search, so the leg cannot
  distinguish "tie by slot" from "tie by handle" or "tie by CSR position". It is a valid
  regression guard for the current policy, not a policy discriminator. If tie policy after
  compaction matters, add a leg where slot order ≠ handle order.

**P3-4 · Shader-validation memory-limit claim: symptom verified, mechanism derivable, likely
an instrumentation artifact** · recorded debt, not a slice-32 defect
- The ledger and handoff record
  `Functon(fused_l2_topk): total used threadgroupMemoryLength(45056) must be <= 32768`
  under GPU validation, reproduced on unchanged baseline (`baseline-validation.log`).
- Static threadgroup arrays in `AdvancedTopK.metal` `fused_l2_topk`:
  `query_cached[768]` floats = 3072 B, `shared_candidates[2048]` × 8 B = 16384 B,
  `scratch[256]` × 12 B = 3072 B, total **22528 B**, under the 32 KiB hardware limit (the
  kernel dispatches fine unvalidated). The reported 45056 B is exactly **2 × 22528**,
  consistent with the GPU validation layer shadowing each threadgroup allocation.
  **Unverified beyond the arithmetic**: confirm by reading
  `MTLComputePipelineState.staticThreadgroupMemoryLength` (expect 22528).
- Consequence if confirmed: any kernel with more than 16 KiB of static threadgroup memory is
  structurally invisible to GPU-validated legs. `ivf_list_search` has
  `query_cached[2048]` (8192 B) + 16384 B + 3072 B = 27648 B, so it is excluded too. The
  "shader instrumentation coverage remains limited" caveat is therefore structural, not
  slice-specific.
- Options for the owner: a validation build variant with smaller `MAX_SHARED_CANDIDATES_POT`
  and `IVF_MAX_D_CACHED`, or dynamically sized candidate storage via
  `setThreadgroupMemoryLength` so validated legs can run with a reduced footprint.

### Process notes (not findings)

- Commit `39bb404` is pushed to `origin/gifton/metal-hardening-checkpoint` and the tree is
  clean. Handoff §3.6 reserves checkpoint authorization to the owner; the session memory
  records owner approval of the five tests but no explicit push authorization. Owner to confirm.
- The ledger's "no production changes" claim is verified: the range diff touches only the
  test file and two docs. Production mtimes were bumped at 18:18:53 by the mutation script's
  write/restore cycle; content matches HEAD and the script asserts SHA-256 restoration.
- Gate claims verified from `/private/tmp/va-ivf-coverage/2026-09-14/`: debug 1764/0/6
  (260.595 s), release 1764/0/6 (55.628 s); `IVFValidationTests` 33 passed / 6 skipped, the
  six being exactly the remaining placeholders (`testCommonEmbeddingDimensions`,
  `testDifferentDistanceMetrics`, `testRecallStableAfterRepeatedInserts`,
  `testThroughputScalesWithNprobe`, `testThroughputStability`, `testVectorsCloserToOwnCentroid`).
- The handoff's earlier "missing retrieval API" statement was stale; the agent recorded the
  correction rather than rewriting history (rule 6). `vector(for:)`, `vectors(for:) -> [[Float]?]`
  and the dictionary overload at `AcceleratedVectorIndex.swift:87` all exist at HEAD.
- `@preconcurrency import Metal` in the test file is required by the `any MTLBuffer` helper
  parameter under strict concurrency. Harmless.

## Six Iron Questions — summary

- **Q1 Liveness.** All five tests reach the IVF GPU legs: `routingThreshold: 0` bypasses the
  flat route (`AcceleratedVectorIndex.swift:914-925, 976-987`), `IVFSearchPipeline` has no CPU
  fallback, `decisionEngine` defaults to nil so the flat control leg also runs the fused
  kernel via the k > 8 two-pass path. `isExhaustive == false` is set only by the IVF legs
  (lines 1429, 1476), so it is a real discriminator.
- **Q2 Claim coverage.** See table below.
- **Q3 Revert tests.** Membership: drop `gpuStructureDirty = true` in `assignToCluster`
  (`IVFStructure.swift:284`) → cached 16-vector structure reused after insert → count 16 ≠ 17.
  CSR: `listOffsets.append` moved or `+1` (fault-injected) → literal mismatch. Routing:
  threshold-zero treated as default (fault-injected) → `isExhaustive` true and count 32.
  Retrieval: drop `ivfStructure?.compact(using: slotMapping)`
  (`AcceleratedVectorIndex.swift:865`) → lists keep old slots, three fall outside the
  compacted range and are skipped, result set has 26 ≠ 29 handles (derived; the agent's
  mutation covered the read-slot offset instead). End-to-end: rooted L2 (observed red during
  development), doubled score (fault-injected), or expansion-formula distance (see Q4).
- **Q4 Tolerance honesty.** Top-7 squared distances are ≤ 71.3, where FP32 ulp ≈ 7.6e-6.
  Direct-difference accumulation error observed ≤ 3e-6 (bound ≈ 3e-5) against the 1e-4
  tolerance. An expansion-formula regression (‖q‖²+‖v‖²−2q·v) produces 2.9e-5 to 3.1e-3
  error on the same fixture, so five of seven top results for the third query would fail:
  the tolerance discriminates the regression class. Minimum adjacent gap in the top-8 is
  0.1375 excluding the exact tie; the 7|8 boundary gap is ≥ 12.2, so FP32 rounding cannot
  reorder results.
- **Q5 Masking patterns.** No vacuous or cumulative-state assertions. Sentinel skipping in
  `IVFSearchResult.results(for:)` is what makes the nprobe=1 count < 32; `DeletionMask.isDeleted`
  is bounds-checked, so sentinel indices cannot trap. Assertions are on exact sets, orders and
  literals. The one masking pattern found is P3-2 (adjacent production, not the tests).
- **Q6 Class closure.** Two families with ≥ 2 instances: unseeded k-means in shape-asserting
  tests (P3-1) and silent drop/skip of list slots in `IVFStructure` (P3-2). Closing tests
  proposed above.

## Claim-coverage table

| Claim (where) | Covering leg | Status |
|---|---|---|
| "Both insertion spellings must assign once and invalidate that cached structure" (test:1827) | `testAllVectorsAssignedToExactlyOneCluster` phases 0→1 (single) and 1→2 (batch), each followed by single and batch searches; mutation `duplicate-membership` (8) | COVERED; invalidation revert derived |
| "Returning raw CSR/slot indices as handles must fail retrieval checks" (test:815) | `testReturnedHandlesRetrieveCorrectVectors` set equality after non-tail removals + compaction; `HandleAllocator.applyCompaction` remaps `stableIDToSlot` | COVERED; remap revert derived (26 ≠ 29) |
| "CSR order differs from storage order, with an empty interior list" (test:1836) | literals `[0,2,2,3,4]`, `[6,9,1,3]`, removed `[0,2,2,2,3]`, cleared zeros | COVERED |
| "A one-list probe returns fewer than N; flat returns all N" (test:1887) | `testRoutingThresholdZeroForcesIVFSearch`; mutation `zero-routes-flat` (8) | COVERED, conditional on non-degenerate unseeded k-means (P3-1) |
| "independent Double squared-L2 oracle ... including an exact tie" (ledger) | tie 3/11 at 1.5625 verified; kernel tie key = slot | COVERED for slot == handle only (P3-3) |
| "Full probing deliberately isolates correctness from approximate recall" (ledger) | nprobe = nlist = 4; `maxCandidatesPerQuery = totalVectors` | COVERED by derivation |
| "without changing production code, public APIs, shaders or adding test hooks" (ledger) | `git diff f685660..39bb404 -- Sources/ Package.*` is empty | VERIFIED |
| "Full debug 1764/0/6, release 1764/0/6, IVF suite 39/0/6" (ledger, handoff, memory) | `debug-full.log`, `release-full.log` | VERIFIED |
| "fused_l2_topk 45056 > 32768 under validation, reproduced on baseline" (ledger, handoff) | `baseline-validation.log`, `validation-debug.log` | VERIFIED symptom; mechanism derived, UNVERIFIED (P3-4) |
| "The retrieval API already existed" (handoff) | `AcceleratedVectorIndex.swift:87, 711, 720` | VERIFIED |
| "shouldn't happen in normal flow" (`IVFStructure.swift:444`, adjacent) | none | UNCOVERED (P3-2) |

## Class-closure proposals

1. **Reproducible k-means in tests.** `seed:` on `KMeansConfiguration` (default random),
   then pin seeds in every IVF test that asserts on cluster shape or list membership under
   partial probing.
2. **List-slot validity invariant.** A test that, after arbitrary insert/remove/compact
   sequences on `AcceleratedVectorIndex`, reads the serialized CSR and asserts every
   `vectorIndices` entry is `< allocatedSlots` and not deleted; pair it with the serializer
   throwing instead of `continue`.

## Pattern-file update (proposed, not applied)

Append to `docs/audits/REVIEW-PATTERNS.md` once P3-4's mechanism is confirmed:

> **26. Validation-layer threadgroup shadowing hides large-static-threadgroup kernels**
> - Mechanism: GPU shader validation appears to reserve a shadow copy of each static
>   threadgroup allocation, so a kernel using more than 16 KiB of `threadgroup` arrays fails
>   `validateComputeFunctionArguments` under instrumentation while dispatching legally on
>   hardware. Every instrumented leg silently excludes such kernels.
> - Tell: `total used threadgroupMemoryLength(N) must be <= 32768` where N is exactly twice
>   the static sum; kernels declaring `threadgroup T buf[2048]`-scale arrays; "API-only
>   validation passes" caveats.
> - Incident: slice 32 — `fused_l2_topk` (22528 B static, reported 45056) and by the same
>   arithmetic `ivf_list_search` (27648 B).

## Offer

Remediation slice, awaiting scope decision: (a) seed injection for k-means and pinned seeds
in shape-asserting IVF tests; (b) make `prepareGPUStructure` throw on out-of-range slots and
add the list-slot invariant test; (c) confirm `staticThreadgroupMemoryLength` for
`fused_l2_topk`/`ivf_list_search`, then either a validation-sized build variant or dynamic
candidate storage so GPU-validated legs can cover both kernels.
