# Fused Top-K shader instrumentation

Slice 34, 2026-09-14. Applies to `AdvancedTopK.metal`'s `fused_l2_topk`.

## Observed failure

On Apple M3 Max (32,768-byte threadgroup limit), Swift 6.3.3 and Metal toolchain
32023.883, the original pipeline reported 22,528 bytes of static threadgroup memory
without shader validation and 45,056 bytes with it. `MTL_DEBUG_LAYER=1
MTL_SHADER_VALIDATION=1` caused a dispatch-time assertion before execution.

A standalone runtime-compiled reproduction removed IVF routing from the experiment.
Minimal single-array kernels also showed static footprints of 1,024→2,048,
4,096→8,192 and 16,384→32,768 bytes under instrumentation. Disabling only the
threadgroup-memory or global-memory validation check did not remove the original
fused pipeline's excess. These are measured compiler/instrumentation effects on this
toolchain, not proof of a particular proprietary implementation or an Apple defect.

## Remediation and preserved behavior

The shader no longer copies every thread's eight retained candidates into a 2,048-entry
shared array. Each thread keeps its sorted private heap and publishes its current head
through the existing minimum reduction. Thread zero emits the winner and publishes its
owner; only that owner advances its private cursor. The next global winner must be a
head of one of the sorted heaps, so the retained candidate multiset is preserved.

The reduction barrier publishes SIMD minima. The subsequent barrier publishes the winner
owner and completes all scratch reads. Every thread reads the owner before entering the
next reduction, whose barrier precedes the next owner write. Exhausted threads continue
participating, supplying sentinels; no private heap read occurs after exhaustion.

Existing ordering remains numeric distance, then original index, with NaNs after numeric
values and padding after every real entry. Real infinities and NaNs are consumed once.
Distance arithmetic, optional distance-output behavior, query caching, argument layout,
and public routing are unchanged. Public fused selection retains K≤8 and D≤768;
`execute()` retains its exact fallback for K>8. Raw larger-K dispatch still selects only
from eight retained candidates per thread and gains no exact-global-top-K guarantee.
The fixed public geometry remains 256 threads; tests additionally cover complete SIMD
widths 32, 64, 96 and 128. Partial-SIMD support is not established by this change.

Measured static footprint is now **6,160 bytes normally and 12,308 bytes instrumented**,
for both plugin and runtime compilation in debug. Compiler alignment means reported
rounded values need not have an exact 2:1 ratio. These measurements are not a universal
future-toolchain guarantee; the regression test checks the actual pipeline/device budget.

## Verification

`Hardening/FusedTopKInstrumentationTests.swift` adds three tests: compiled memory budget;
retained-pool ordering with concentrated winners, ties, lane exhaustion and D=767/768;
and finite/infinity/NaN ordering with empty-input and exhausted-pool padding. Both output
buffers have suffix canaries. Widths 32/64/96/128/256 and K=1/4/8/33/129 are covered as
applicable. The old budget test failed on both compilation paths at 45,056 bytes; all
three new tests passed normally on the old implementation before remediation.

Scoped full API and shader validation runs use:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test \
  --filter 'FusedTopKInstrumentationTests|FusedL2TopKKernelTests|TopKNaNPolicyTests.testFusedSelectionConsumesWinnersIncludingInfinityAndNaN|CapabilityCapPolicyTests.testFusedL2TopKOverDimSentinelFills|IVFTests.testIVFBasicSearch' \
  --skip 'FusedL2TopKKernelTests.testWithoutDistances'
```

Repeat with `-c release` for runtime compilation. The skip expression excludes the two
without-distance methods from this diagnostic run; it does not change or disable tests
in the normal suite. Full normal debug/release gates include those methods. Exact final
counts and evidence are recorded in AUDIT-3 slice 34.

## Performance boundary

A paired standalone benchmark compiled both old and new pipelines in one process with
MSL 4.0 and fast math, without validation. It alternated execution order, discarded 50
warm-up pairs, then measured 100 pairs per shape. Medians below are GPU command time in
microseconds, excluding compilation and CPU end-to-end overhead:

| Q | N | D | K | Original | Heap-head merge |
|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 128 | 4 | 65.13 | 49.90 |
| 32 | 4096 | 8 | 1 | 36.15 | 32.23 |
| 32 | 4096 | 128 | 8 | 117.75 | 109.71 |
| 16 | 4096 | 768 | 8 | 482.94 | 471.83 |
| 32 | 64 | 8 | 128 (raw) | 65.38 | 118.96 |

No regression was observed in these supported fused-range samples. One device,
synthetic repeated queries and medians without a variance estimate do not establish a
general speedup. Raw K=128 is approximately 82% slower: repeated reductions replace the
old large-K bitonic network. Public K>8 fallback code is unchanged.

## Remaining validation blockers

**Trained IVF list search is still blocked under shader instrumentation.** Once fused
coarse selection succeeds, the unchanged `ivf_list_search` pipeline asserts at 55,296
bytes against the 32,768-byte limit. This was observed in
`IVFValidationTests.testAllVectorsAssignedToExactlyOneCluster`. Passing the basic IVF
test does not establish trained-list-kernel coverage; routing may use a flat path.

A separate IVF remediation must preserve its public large-K performance: both IVF search
APIs accept K>8 without the fused wrapper's fallback, and filtering can triple K before
list search. Simply copying the new all-K reduction would put the measured large-K
tradeoff on ordinary public calls. Next investigation: measure the IVF footprint in both
compilation modes, compare strategies that preserve efficient large-K selection, and
benchmark representative K, nprobe, dimension and filtered searches before choosing.

The subsequent [IVF list investigation](IVF-LIST-INSTRUMENTATION-INVESTIGATION.md)
reproduces the failure and compares three throwaway remedies. It recommends shared
workspace reuse with the existing small/large-K split; production remediation remains open.

**Indices-only fused dispatch has a separate pre-existing binding failure.**
`includeDistances: false` leaves buffer(3) unbound. API validation rejects it despite the
shader's null check. A standalone reproduction using the original shader, with API
validation alone, reported the missing `result_distances` binding and exited with
SIGABRT. The existing `testWithoutDistances` public regression reaches the same failure.
This slice does not change its ABI or optional-output policy. The previously recorded
unary elementwise `input_b` binding failure also remains open.

Evidence: `/private/tmp/va-fused-instrumentation/2026-09-14/` contains baseline/fixed
probes, minimal-array matrix, red/green tests, paired benchmark, independent reviews,
scoped validation logs and normal final gates. Temporary evidence is local; this contract,
the audit record and permanent regression tests are retained in the repository.
