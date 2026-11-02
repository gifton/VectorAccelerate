# Numerical stability: current contracts and remaining investigations

**Reconciled:** 2026-09-13 against the source and hardening slices 1–25.
This document replaces the January guidance. The audit ledger and linked contracts
record implemented behavior; unchecked tasks below are investigations, not approved
numerical-policy changes. No production arithmetic changed during this reconciliation.

## Completed work and retained limits

| Surface | Current status | Reference and coverage |
|---|---|---|
| Minkowski | Artificial power clamps/cutoffs and approximate integer/large-p substitutions removed. Explicit fast mode retains FP32 intermediate limits; automatic routing selects stable computation for p > 10. Public p must be finite and positive. | [Audit VA3-022](../audits/AUDIT-3-shaders.md); `MinkowskiRangePolicyTests`, `MinkowskiLargePPolicyTests` |
| Direct rooted L2 | Exceptional accumulator range failures trigger recomputation; a truly unrepresentable final distance still overflows. This does not extend to roots taken after squared-score selection. | [Distance range contract](DISTANCE-RANGE-CONTRACT.md); `EuclideanRangePolicyTests` |
| Squared L2 and dot | FP32 product/accumulator limits intentionally retained and documented. No cancellation rescue or universal accuracy guarantee. | [Distance range contract](DISTANCE-RANGE-CONTRACT.md); `EuclideanRangePolicyTests` |
| Correlation, histogram, LSE and softmax finite-limit gates | VA3-015 repaired correlation finalization and aligned finite-limit checks with fast-math behavior. | [Audit slice 9](../audits/AUDIT-3-shaders.md); `FastMathPolicyTests` |
| LSE and basic-statistics NaNs | VA3-016 establishes NaN propagation through the covered reductions and public APIs. | [Audit slice 10](../audits/AUDIT-3-shaders.md); `NaNReductionPolicyTests` |
| UMAP negative sampling | Target reads are frozen for each pass, with separate output and synchronized publication. Whole-epoch determinism and numerical range are separate concerns. | [UMAP contract](UMAP-NEGATIVE-SAMPLING-CONTRACT.md); `UMAPNegativeSamplingTests` |
| Neural quantization | Activation/normalization flags and per-row reconstruction scales are preserved under the documented generic numerical policy. | [Normalization](NEURAL-LATENT-NORMALIZATION-CONTRACT.md), [result scales](NEURAL-ENCODING-SCALES-CONTRACT.md) |
| Elementwise math flag | Selects ordinary versus explicit fast intrinsics within the stock fast-math library. False does not select a precise pipeline. | [Elementwise contract](ELEMENTWISE-MATH-CONTRACT.md) |

Do not restore the earlier Minkowski base clamp, exponent clamp, small-value cutoff,
or epsilon-based integer-p substitution. They change the requested metric and conflict
with the owner's retained-range policy. `MINKOWSKI_HANDOFF_PHASE2.md` and
`MINKOWSKI_KICKOFF_PROMPT.md` are historical records, not implementation instructions.

## UMAP gradient range and parameter validation — investigation open

`UMAPGradient.metal` still computes squared differences, fractional powers, products
and coefficient divisions in Float32. Clipping the final gradient coefficient does not
establish that earlier expressions stayed finite. Negative-sampling bounds validation
also does not validate all epoch inputs, graph structures or floating-point parameters.

Existing `UMAPGradientKernelTests` cover CPU-reference agreement, gradient direction,
target accumulation and high-level epoch APIs. `UMAPNegativeSamplingTests` additionally
cover the race fix, ownership and sequencing. The remaining task is targeted range and
parameter coverage, not creation of the first UMAP test suite.

- [ ] Reproduce behavior with large finite coordinates, very small nonzero distances,
  and boundary/invalid curve parameters, epsilon and learning rate on actual GPU paths.
- [ ] Use a computable Double reference and inspect intermediate failure mechanisms;
  distinguish squared-distance overflow from power, denominator or final-product failure.
- [ ] Establish the supported parameter/range contract before selecting rejection,
  algebraic reformulation or another owner-approved behavior.
- [ ] Keep attractive-gradient atomic accumulation policy under VA3-019; numerical
  tolerances and seed control do not establish bitwise whole-epoch determinism.

The former suggestion to clamp squared distance to 1000 is withdrawn: that changes the
optimization curve and has not been approved. The example `(1e6)^0.7915 ≈ 5.6e4` is far
below Float32 overflow and does not demonstrate a defect.

## Attention sigmoid and scaling — investigation open

`AttentionSimilarity.metal::attn_sigmoid` currently uses `1 / (1 + exp(-x))`.
The existing `AttentionSimilarityKernelTests.testNormalizedSimilarities` checks output
bounds with random weights; it does not pin accuracy at extreme logits.

At x = -20, sigmoid is approximately 2.06e-9, a normal representable Float32 value.
Forcing every x < -20 to zero discards valid results. At large positive x, rounding to
one can be the appropriate Float32 result; that alone is not an instability.

- [ ] Add controlled-weight/raw GPU fixtures for negative and positive logits, including
  -20 and -40 (normal positive results), plus ordinary/zero controls. Separately
  investigate -89: exp(89) exceeds Float32 range, while sigmoid(-89) is approximately
  2.23e-39, a representable subnormal that the GPU may flush under the retained policy.
- [ ] Evaluate the sign-split identity: for x < 0, use exp(x) / (1 + exp(x)); otherwise
  use 1 / (1 + exp(-x)). This is a candidate, not an implemented or verified GPU fix.
- [ ] Verify both shader compilation paths: fast math may transform expressions, so
  source algebra alone is insufficient evidence of preserved behavior.
- [ ] Inspect temperature validation and projection/dot-product range separately;
  stabilizing sigmoid cannot recover a score that overflowed upstream.

The previous hard cutoffs at ±20 are withdrawn. Nonfinite-input and subnormal policies
must be specified with any implementation; no new guarantees are inferred here.

## LSE partial-max invariant — targeted verification open

For valid finite partial reductions, globalMax is the maximum of participating
partialMax values, so partialMax <= globalMax and exp(partialMax - globalMax) <= 1
in exact arithmetic. The earlier hypothetical partialMax >> globalMax is not a
reproduced failure of a valid reduction.

`NaNReductionPolicyTests` already exercises two-pass NaN propagation, strides and
infinite partials. Any new invariant test should add evidence beyond those fixtures.

- [ ] Trace first-pass output, active-lane masking and second-pass maximum participation
  at partial-group boundaries; add a test only for an uncovered case or reproduced defect.
- [ ] Treat NaN, positive infinity and empty/negative-infinity partials through their
  existing policy branches rather than applying the finite inequality indiscriminately.

## Corrected numerical examples

- `10000^5 = 1e20` fits Float32's finite range; `10000^10 = 1e40` does not.
  A finite rooted result can still follow an unrepresentable intermediate power, which
  is why explicit fast mode and stable computation have distinct contracts.
- Losing a tiny positive term when adding it to a large positive accumulator is
  absorption/rounding, not cancellation. Compensated summation cannot represent a final
  correction smaller than the output format permits and does not prevent overflow.
- `1536 * (1e18)^2 = 1.536e39` exceeds Float32 range. This is covered by the documented
  dot-product range limitation, not an outstanding promise to widen accumulation.

| Float32 property | Approximate value | Interpretation |
|---|---|---|
| Gap above 1 | 1.19e-7 | Spacing at 1, not an absolute error bound for every operation |
| Smallest positive normal | 1.175e-38 | Smaller positive values are subnormal |
| Smallest positive subnormal | 1.401e-45 | Representable when the execution path preserves subnormals |
| Largest finite value | 3.403e38 | Final results above the representable range overflow |
| ln(largest finite) | 88.72 | Approximate positive-exponential range boundary |
| ln(smallest normal) | -87.34 | Entry into the subnormal exponential range, not universal rounding to zero |
| ln(smallest subnormal) | -103.28 | Approximate magnitude boundary for the smallest subnormal, not the exact rounding-to-zero threshold |

GPU flush-to-zero and intrinsic/compiler behavior can change observed small-value
boundaries. Avoid universal exp-underflow thresholds or denormal slowdown claims based
on another architecture. Stock compilation behavior is described in the elementwise
contract; per-operation contracts control any stronger guarantees.

## Remaining validation work

- [ ] Design sustained-operation stress runs that check resource growth, numerical
  drift and completion behavior with explicit workloads, seeds and iteration counts.
- [ ] Measure subnormal behavior and throughput on supported Apple Silicon with GPU
  routing evidence. Thermal conditions matter for performance; do not infer that
  throttling changes the floating-point precision format.
- [ ] Add durable release-runtime test coverage to CI when the owner resumes that
  previously deferred work; a release build alone does not execute runtime shader compilation.

The latest recorded full gates are the 2026-09-08 slice-25 runs: debug and release each
1712 executed, 1701 passed, 11 explicit IVF placeholder skips, zero failures. Those are
historical verification results, not new runs from this documentation reconciliation.
See [the handoff](../audits/HARDENING-HANDOFF.md) for allocation, optional-bias,
performance, hygiene and coverage work outside this numerical backlog.
