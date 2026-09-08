# REVIEW-PATTERNS — VectorAccelerate

How this codebase gets fooled. Loaded by `/audit` as mandatory checklist items; reviews that
discover a new masking pattern append it here (name / mechanism / tell / incident). Seeded
2026-08-21 from the AUDIT-2/AUDIT-3 epics and their slice-4 meta-review.

## 1. Dispatch-geometry shielding
- **Mechanism:** a kernel is correct only under the host's exact dispatch shape (one
  threadgroup, `min(256, dim)` width, pow2 size); nothing enforces the shape.
- **Tell:** `tgSize/2`-style strides, `id % 256`, guards before barriers, shared arrays
  indexed by raw thread id; hosts computing "safe" geometries with comments.
- **Incident:** VA3-002/-013/-014 — six kernels wrong at every geometry the hosts happened
  never to use.

## 2. Dual-compile divergence
- **Mechanism:** debug builds load the per-file-compiled metallib; release compiles all
  shaders as ONE concatenated source behind `KernelContext`'s hand-mirrored preamble.
  Symbol visibility, macro shadowing, and header edits behave differently per path.
- **Tell:** new shader helpers/macros used cross-file; edits to `Metal4Common.h` alone
  (plugin does not track header deps — touch `*.metal` after, VA2-013); any `#define`
  matching a preamble constant (`EPSILON`).
- **Incident:** VA2-001 (release suite red 37/1540 at pristine HEAD), VA3-012 (EPSILON
  1e-7/1e-8 drift, still open).

## 3. Routing-gate masking (and gate-change arming)
- **Mechanism:** GPU legs invisible to tests because thresholds route small test inputs to
  CPU; conversely, a gate RELAXED for a good reason arms every dormant defect it shielded.
- **Tell:** `dimension > 16`-style fallbacks; decision-engine gates; tests whose inputs sit
  below routing thresholds; any diff touching `shouldUseGPU`/threshold logic.
- **Incident:** VA3-002 (engine dotProduct GPU leg untested — all tests dim ≤ 16);
  VA3-032 (VA2-003's k-gate exemption armed a phantom-kernel throw on a public API).

## 4. Phantom names / resolution hijack
- **Mechanism:** runtime-resolved identifiers that match no kernel — or are REWRITTEN by a
  derivation to the wrong real kernel (`"dotProduct"` → `dot_product_kernel`).
- **Tell:** `getPipeline(functionName:)` / `makeFunction(name:)` literals; operation-string
  derivations; any new rewriting case in `PipelineCacheKey.functionName`.
- **Incident:** five instances (batchCosineDistance, tiledTransposeInPlace, vectorMultiply,
  batchDotProduct, batchManhattanDistance) + the VA3-031 hijack. Mechanically closed by
  `ShaderLibraryCompletenessTests` name round-trip + Swift-literal closure (shrink-only
  allowlists) — new instances should be impossible; if one appears, the closure test has a
  hole.

## 5. Pool stale-bytes readback
- **Mechanism:** a kernel that exits without writing its output leaves pooled-buffer bytes
  to be read back as the answer — wrong values with no error.
- **Tell:** any early `return` before output writes; capability caps (`k > MAX`) that no-op;
  result buffers acquired from pools without poisoning in tests.
- **Incident:** VA3-031 (stale dot products), VA3-008/-034 (warp-select tails and over-cap k).

## 6. Pow2-only test data
- **Mechanism:** reduction/tree defects invisible because every tested size is a power of
  two (or a multiple of the block size); comments may claim non-pow2 robustness the guard
  does not provide.
- **Tell:** test dims all in {16, 32, …, 256, 512}; "robustness check" comments on
  `size/2`-halving loops; hosts forcing pow2 widths "for correctness".
- **Incident:** VA3-014 — orphaning trees under a literal "Robustness check for
  non-power-of-2 tgSize" comment; hosts knew and compensated.

## 7. Silent CPU fallback
- **Mechanism:** provider rescues convert GPU breakage into correct CPU results — tests
  green while kernels rot (or never existed).
- **Tell:** `fallbackToCPU` defaults, `catch` → CPU result, `try?` on dispatch paths;
  differential tests that only ever run through the provider.
- **Incident:** AUDIT-2 foundational finding — `batchCosineDistance` was requested at three
  engine sites and NEVER EXISTED; every cosine batch dispatch silently ran CPU. Counter:
  RoutingProvenanceTests telemetry assertions + kernel-direct legs.

## 8. Fast-math idiom fragility
- **Mechanism:** `-ffast-math` reassociates `sqrt(a)*sqrt(b)` into `sqrt(a·b)` (overflow),
  rewrites `(x/a)/b` into subnormal-reciprocal multiplies (flush to zero), folds
  `isinf`/`isfinite` per compile path.
- **Tell:** those exact shapes; `== INFINITY` compares; new normalization/denominator math
  not using the established idioms (two-stage `precise::divide`, `> FLT_MAX` compares,
  pre-scaled rescues in `Metal4Common.h`).
- **Incident:** VA2-008 (measured corruptions), VA3-015 (stragglers, open).

## 9. Cumulative shared-state assertions
- **Mechanism:** process-shared pools/counters make `>= N` assertions vacuously true in
  full-suite runs (and order-dependently false in isolation — flake or theater, depending
  on direction).
- **Tell:** assertions on statistics without a pre-call snapshot; anything asserting on a
  shared singleton's totals.
- **Incident:** Priority2 buffer-pool test — flaked on `allocationCount`, then the fix was
  vacuous on cumulative hits+misses; final form asserts the delta.

## 10. Value-policy divergence across legs of one API
- **Mechanism:** one public operation, multiple implementations (GPU / SIMD / scalar CPU,
  or size-tiered legs) with different NaN/overflow/clamp/tie policies — answers flip at
  routing boundaries.
- **Tell:** size-threshold routing (`>= 100`); inline math in one leg where siblings call a
  shared core; NaN-swallowing guards (`norm > 0`); input-validation guards that check only
  element [0] of a collection (everything past index 0 reaches the legs unvalidated).
- **Incident:** VA3-033 (cosine answers flipped at the batch-size-100 boundary), VA3-016
  (NaN/tie policy inconsistencies, open), VA3-035 (ragged candidate past index 0: one
  input, four policies — +Inf / NaN / zip-truncated value / debug-assert-or-release-OOB —
  closed by a shared pre-routing all-candidates guard, 2026-08-23).

## 11. Score sentinels confused with candidate validity
- **Mechanism:** using infinity alone to mark empty/consumed candidates works on ordinary
  finite inputs but drops real NaNs or repeatedly selects a real infinity. A NaN-aware
  comparator also needs invalid-index precedence; value ordering alone is insufficient.
- **Tell:** heap initialization to infinity, winner removal that overwrites only distance,
  or different comparisons at admission, heap maintenance, and output sorting.
- **Incident:** VA3-016 slice 12 — real NaNs lost to padding in warp/streaming/fused/IVF
  selection; fused infinity winners reselected because their indices remained valid.
  `TopKNaNPolicyTests` checks real count, uniqueness/order, and poisoned padding across
  both compilation paths. Numeric ordering uses FP32 integer keys because floating
  comparisons also collapsed subnormal scores into zero ties in both libraries.

## 12. Intermediate range mistaken for final-result range
- **Mechanism:** clamping a power or exponent changes the formula; alternatively, a root
  factor can overflow before a small scale makes the final result representable. Moving
  everything into log space can lose the scale's significand at a representability boundary.
- **Tell:** fixed exponent clamps, absolute small-base cutoffs, `scale * pow(sum, 1/p)`,
  and `exp2(log2(scale) + rootExponent)` asserted to be universally stable.
- **Incident:** VA3-022 slice 13 — stable fractional root-factor overflow and ratio
  underflow; the first logarithmic rescale then failed finite FLT_MAX fixtures at p=1/64
  and 1/128. `MinkowskiRangePolicyTests` distinguishes intentionally limited fast-mode
  intermediates from stable-mode rescue and checks exponent-split rescaling endpoints.

## 13. Wide pointers hide narrow address arithmetic
- **Mechanism:** `pointer + row * stride` evaluates a pair of `uint` operands in 32 bits.
  Casting the completed product or assigning it to `ulong` does not recover lost bits.
  A wide result can also narrow again at a local assignment or helper argument. In
  `wideBase + row * stride`, the independent multiplication remains narrow.
- **Tell:** device-buffer offsets declared `uint`, casts outside whole products, compound
  batched addresses, and helpers taking `uint` indices from wide callers. Ordinary small
  allocations and CPU/GPU differentials do not cross the overflow boundary.
- **Incident:** VA3-018 slice 15 — row/stride expressions across 21 shader files, including
  normalization's downstream bit-copy index. `IndexWidthTests` compiles expressions and
  declarations extracted from production sources into small GPU probes, then checks
  UInt64 references below/at/above 2^32. These establish arithmetic behavior; real kernel
  regressions and both library compilation paths separately check integration. They do
  not establish that large buffers can be allocated or that count/ID limits were widened.


- **Follow-up:** VA3-026 slice 18 — ADC's `M * K` table size could wrap to zero before
  a shared-memory load bound was checked. Validate K first, then compare M to the
  capacity divided by K before multiplying. The shader rejection must be uniform before
  the cooperative load/barrier. Tests include the 2^32 product and ragged threadgroups.

## 14. Correct values do not prove legal memory accesses
- **Mechanism:** hardware can tolerate a misaligned cast, and an optimizer can remove
  invalid loads in unused vector lanes. Value parity and clean validation output may
  therefore coexist with a source-level alignment or bounds violation.
- **Tell:** scalar storage reinterpreted as aligned vectors, four-component loads guarded
  only by the number of components eventually stored, or a warning-only check that
  ignores explicit C++ reinterpret casts.
- **Incident:** VA3-017 slice 16 — packed storage views remove alignment-increasing casts.
  Compiler diagnostics need equivalent C-style spellings in temporary test copies to
  check explicit reinterpret casts too; a bad-cast canary verifies the diagnostic. The
  generic transposed neural decoder still read four weights for a 1-3-output tail;
  test-only load-footprint instrumentation found 20 invalid requests even though value
  and shader-validation tests passed. Bounded scalar tail reads close that hole.
  The sweep also found three VA3-018 specialized vector-store offsets missed by the
  previous inventory: address scans must include expressions inside vector pointer casts.

- **Follow-up:** VA3-025 slice 17 — sparse c-TF-IDF's final vector group gathered unused
  term IDs and overwrote tail padding. Checked gathers and output canaries reproduce
  both failures. Keep API validity separate from shader footprints: vector-typed buffer
  arguments require at least 16 bound bytes even when scalar tail code accesses less.
  Exact-size fixtures below that minimum fail API validation before the shader runs.


## 15. A bounded write needs a bounded consumer count
- **Mechanism:** clamping `buffer[index]` writes alone leaves an atomic append counter
  free to exceed capacity or wrap. A later CPU/GPU consumer can still read beyond the
  buffer, or interpret the valid prefix as a complete result. An algorithmic capacity
  proof also fails when a fusion caller skips the state transition it assumes.
- **Incident:** VA3-028 slice 19 — Borůvka keeps the proven 2N allocation for completely
  merged rounds, but repeated unmerged rounds saturate a bounded reservation counter
  at capacity+1. Checked readback throws before consuming an incomplete round. Tests
  cover logical output guards, already-overflowed counters, UInt32.max saturation,
  repeated fusion calls, and the actual geometric bound with CPU merging.


## 16. Reservation order is not logical order; requested size is not storage
- **Mechanism:** an atomic allocator returns disjoint segments in execution order.
  Treating per-owner starts as CSR boundaries silently gives consumers another owner's
  records or reversed ranges. Separately, an allocator request is not proof of returned
  storage capacity when a pool clamps bucket sizes.
- **Incident:** VA3-019 slice 20 — fused IVF assumed both query-ordered allocation and
  sufficient estimated/pooled storage. Tests force an explicit permutation independent
  of scheduler behavior, exercise underestimated hints, and request outputs above the
  pool's largest bucket. Validate actual storage and completed segment coverage before
  publishing CSR; do not infer either property from allocation intent.


## 17. Unique writers do not imply race-free reads
- **Mechanism:** assigning each point one writer is insufficient when those writers read
  other points' mutable coordinates. Even apparently stable output can depend on GPU
  scheduling. A threadgroup barrier cannot order all threadgroups in a dispatch.
- **Incident:** VA3-019 slice 21 — UMAP negative sampling now reads frozen target
  coordinates and writes distinct output, preserving sequential updates of each source.
  Publication waits for the complete sampling dispatch. Analytic fixtures distinguish
  both the original race and an accidental change to frozen-source accumulation; tests
  also exercise preceding producers, subsequent consumers and scratch reuse in a
  concurrent encoder. Allocation convenience requires retained command-buffer references;
  unretained command buffers need caller-owned scratch retained through completion.
