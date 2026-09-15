# IVF list search: bounded instrumentation fix

Slice 35, 2026-09-14. Applies to `IVFListSearch.metal`'s `ivf_list_search`.
The preceding [investigation](IVF-LIST-INSTRUMENTATION-INVESTIGATION.md) records
the original reproductions, alternatives, and prototype benchmarks.

## Memory and synchronization

On Apple M3 Max / Metal toolchain 32023.883, the original pipeline uses 27,648
bytes normally and 55,296 bytes under shader instrumentation, exceeding the
32,768-byte device limit. The fixed pipeline uses a single workspace union:

- During scanning: the 2,048-float query cache.
- During K≤32 selection: eight SIMD minima and the winning thread index.
- During K>32 selection: the existing 2,048 distance/index candidates.

An explicit threadgroup barrier after scanning finishes every query-cache read
before either selection path writes into the same storage. Small-K selection
reduces the heads of the sorted private heaps, writes the winner, and publishes
its owner. Only that owner advances. Exhausted lanes keep participating with
sentinels and never read beyond their private heap.

The reduction barrier publishes SIMD minima. The owner-publication barrier also
finishes scratch reads. Every lane consumes the owner before entering the next
reduction; that next reduction's barrier prevents another owner write from racing
those reads. The scratch array needs one entry per SIMD group, not per thread.

Measured fixed storage is **16,384 bytes normally and 32,768 bytes instrumented**.
This fits exactly on the tested device/toolchain, with **no instrumented headroom**.
The budget regression checks actual compiled pipeline storage against the device
limit in plugin and runtime builds. It detects future overhead changes; it is not
a production fallback. Extra headroom is deferred until the owner's broader
benchmarking and guardrail suite can evaluate a replacement selection algorithm.

## Preserved behavior

- K≤32 uses private heap-head selection; K>32 retains the same shared-candidate
  layout, power-of-two padding, bitonic sorting network, and output writes.
- Each lane retains eight candidates across its CSR list scans. Large K selects
  from that retained pool; no new global-exactness guarantee is introduced.
- Numeric/NaN/original-index/sentinel ordering and distance arithmetic are unchanged.
- Query caching still applies through D=2048, with device reads above that boundary.
  This does not expand the public coarse-selection dimension limits.
- Public dispatch width remains 256. Regressions also cover full-SIMD widths
  32, 64, 96 and 128; partial-SIMD support is not established here.
- No buffer ABI, public API, host allocation, dispatch count, or routing change.

## Permanent coverage

`Hardening/IVFListInstrumentationTests.swift` adds five tests:

1. Compiled memory budget, including safe failure before an over-budget dispatch.
2. Retained-pool selection at K=1/8/31/32/33/128/513/2051 and widths
   32/64/96/128/256, with reversed original IDs, ties, nine concentrated winners
   (only eight retained), invalid/sentinel selected lists, and output canaries.
3. Finite/infinity/NaN ordering and padding through uneven, empty, and invalid lists.
4. Repeated workspace reuse with three queries, unequal lane workloads, empty
   inputs, D=1/767/768/2047/2048/2049 and both sides of the selection split.
5. Trained public IVF routing, K=32/33, and actual filtering with requested K=11
   over-fetched to 33, using nprobe=4/16 to exercise both coarse-selection routes.

CPU expectations use literals, independent sorting of retained rows, or exactly
representable scalar-vector distances. The original implementation passes all five
tests normally; the instrumented budget test fails at 55,296 bytes on both paths.

## Validation scope

The instrumented integration command is:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test \
  --filter 'IVFListInstrumentationTests|IVFSearchPipelineTests|FusedTopKInstrumentationTests|TopKNaNPolicyTests.testIVFSelectionOrdersComputedNaNsByOriginalIndex|IVFValidationTests.test(AllVectorsAssignedToExactlyOneCluster|InvertedListOffsetsAreCorrect|RoutingThresholdZeroForcesIVFSearch|ReturnedHandlesRetrieveCorrectVectors|EndToEndWorkflow|RecallStableAfterRepeatedInserts|VectorsCloserToOwnCentroid|CommonEmbeddingDimensions|DifferentDistanceMetrics)'
```

Repeat with `-c release` for runtime compilation. This selects five new tests,
six IVF pipeline tests, nine revived IVF correctness tests, the existing IVF NaN
regression, and three fused instrumentation regressions. No selected test is skipped.
Scoped debug and release each pass **24 tests with zero failures**, exit 0. Normal
full debug/release gates each pass **1772/0/2** (1770 passed, two existing skips),
exit 0. Tested source hashes match. Timings and evidence are in AUDIT-3 slice 35.

This closes the trained-list memory blocker; it does not claim full-suite shader
validation. The separate indices-only fused `result_distances` binding and unary
elementwise `input_b` binding failures remain open. The two throughput placeholders
remain skipped. No benchmark harness or new performance claim is added in this
slice; the owner's broader measurement suite owns that work.

Local evidence: `/private/tmp/va-ivf-bounded/2026-09-14/` contains red/baseline/green
logs, scoped instrumented gates, normal full gates, review notes, and source hashes.
