# IVF list search: instrumented memory investigation

2026-09-14. Investigated against `b69d4e9` on
`gifton/metal-hardening-checkpoint`. **Investigation only: the production shader
is unchanged and its validation failure remains open.**

## Finding

`ivf_list_search` allocates 27,648 bytes of static threadgroup memory normally and
55,296 bytes under shader instrumentation on the tested M3 Max. The device limit
is 32,768 bytes. Both the existing plugin-built library and standalone runtime
compilation reproduce the same dispatch-time assertion with API and shader
validation enabled. API validation alone completes the same dispatch correctly.

This isolates the failure from IVF training, coarse routing, buffer pools, and
query size. Even Q=1, N=32, D=8, K=4, nprobe=1 fails before GPU execution.

| Allocation | Source capacity | Bytes normally |
|---|---:|---:|
| Query cache | 2,048 floats | 8,192 |
| Shared candidates | 2,048 distance/index pairs | 16,384 |
| Reduction scratch | 256 distance/index/position triples | 3,072 |
| Total | | **27,648** |

The static allocations remain present regardless of runtime D or K. Only eight
scratch entries are needed for the eight SIMD groups at width 256, but reducing
that allocation alone would still leave the instrumented pipeline over budget.

Apple documents that static and dynamically allocated threadgroup storage must
fit the device limit; see
[threadgroup memory allocation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)).
The measured expansion is a compiler/instrumentation effect on this toolchain.
This investigation does not establish its proprietary implementation or an Apple
defect, and does not propose disabling validation.

## Public behavior that constrains the remedy

`IVFListSearchKernel.search` and `IVFSearchPipeline.search` accept positive K
without the fused coarse kernel's K≤8 routing restriction. The index's filtered
`searchIVF` triples K before list search: requesting 11 neighbors can reach K=33.
The list shader currently uses repeated reduction for K≤32 and bitonic sorting
for K>32. Large-K performance therefore matters for ordinary public calls.

Each lane retains eight candidates across all selected lists. A lane's CSR scan
restarts at each list's start plus its thread index. The retained pool, original
index tie-breaking, NaNs after numeric distances, and sentinels after real
entries must remain unchanged. This work must not silently claim exact global
top-K for arbitrary large K: candidates can already be discarded by private heaps.

The list shader caches queries through D=2048 and reads queries directly from
device memory above that boundary. This is distinct from restrictions imposed
by the public coarse-selection path. Preserving raw D>2048 behavior does not
expand the public index's dimension guarantees.

## Throwaway experiments

Three shader variants were built outside the repository. All retain distance
arithmetic, candidate capacity, buffer arguments, dispatch width, and ordering.

| Variant | Selection | Normal bytes | Instrumented bytes |
|---|---|---:|---:|
| Original | Existing reduction / bitonic split | 27,648 | 55,296 — abort |
| Shared workspace, sort all K | Bitonic for every K | 16,384 | 32,768 |
| Shared workspace, hybrid | Private heap heads for K≤32; bitonic for K>32 | 16,384 | 32,768 |
| Separate cache, all heap heads | Repeated head selection for every K | 8,304 | 16,596 |

The shared workspace is a union of the query cache, candidate array, and small
head-selection scratch (eight SIMD minima and one winner owner). Their lifetimes
do not overlap. An explicit threadgroup barrier after scanning completes every
query-cache read before any thread writes selection data into the same storage.

For small K, each thread supplies the head of its sorted private heap. Thread
zero writes the winning candidate and publishes its owner; only that owner
advances its cursor. Exhausted lanes still participate using sentinels. Existing
reduction and owner-publication barriers separate each iteration. For large K,
the same candidate layout, power-of-two padding, and bitonic network are retained.

The hybrid prototype also compiled offline using the package's Metal compiler
flags and completed instrumented checks at 32,768 bytes. This is a standalone
offline build, not a rebuilt package/plugin integration test.

## Correctness evidence

The standalone harness checks an independent CPU reference: group CSR rows by
their actual scan lane, sort and retain eight per lane, then globally sort the
retained pool. It checks both indices and distances, plus suffix canaries in both
output buffers. Fixtures include three distinct queries, uneven and empty lists,
reversed original IDs, ties, finite/infinite/NaN distances, and empty input.

Two fixture grids cover:

- Widths 32, 64, 96, 128, 256 with K=1, 8, 31, 32, 33, 128, 513, 2051;
  N=2051 and D=8, including nonfinite values and exhausted-pool padding.
- D=1, 767, 768, 2047, 2048, 2049 with N=0/65 and K=8/33/128 at width 256.

Results:

- Normal runtime compilation: **304 dispatches passed**, covering the original
  and all three prototypes (76 each).
- API plus shader validation, runtime compilation: **228 dispatches passed**,
  covering all three prototypes.
- API plus shader validation, offline hybrid compilation: **76 dispatches passed**.

The initial high-dimensional fixture demanded exact distance equality despite
Float accumulation rounding, and failed on the original shader too. The final
fixture uses smaller binary-fraction values with exactly representable sums;
exact comparisons then pass on the original and prototypes. This was a harness
correction, not a production numerical fix.

These experiments do not replace permanent package regressions, trained-index
integration coverage, malformed-CSR tests, or full debug/release gates.

## Performance evidence

All four pipelines were compiled in one process using MSL 4.0 and fast math.
Validation was disabled. The harness rotates execution order, uses the same
buffers, discards 30 warm-up rounds, and measures 70 rounds per shape. Output
indices agree with the original throughout. Values below are median GPU command
times in microseconds, excluding compilation and CPU/end-to-end costs.

| Q | N | D | K | nprobe | Original | Sort all K | Hybrid | All heads |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 8 | 8 | 4 | 90.85 | 285.33 | 68.29 | 65.19 |
| 32 | 64 | 8 | 33 | 4 | 126.85 | 128.33 | 128.96 | 78.27 |
| 32 | 64 | 8 | 128 | 4 | 83.75 | 82.44 | 86.81 | 132.81 |
| 32 | 4096 | 128 | 8 | 4 | 133.44 | 203.27 | 123.27 | 124.75 |
| 32 | 4096 | 128 | 32 | 4 | 181.38 | 197.00 | 149.60 | 147.31 |
| 32 | 4096 | 128 | 33 | 4 | 202.56 | 198.35 | 200.94 | 149.63 |
| 32 | 4096 | 128 | 128 | 4 | 201.67 | 197.52 | 200.04 | 262.75 |
| 16 | 4096 | 768 | 33 | 4 | 587.92 | 585.77 | 592.98 | 540.69 |
| 16 | 4096 | 2048 | 128 | 4 | 1585.54 | 1585.19 | 1587.79 | 1654.75 |
| 32 | 4096 | 128 | 33 | 1 | 89.21 | 89.85 | 91.08 | 62.63 |
| 32 | 4096 | 128 | 96 | 4 | 202.71 | 197.79 | 200.37 | 224.77 |
| 32 | 4096 | 128 | 384 | 4 | 201.75 | 197.73 | 200.27 | 570.75 |

The large-K trend rejects all-head selection as a blanket replacement: the
K=128, D=128 case is about 30% slower, and K=384 is about 183% slower. Sorting
everything incurs substantial small-K overhead. Hybrid measurements favor
keeping the existing threshold while improving the small-K merge.

These are synthetic microbenchmarks on one M3 Max, with only nprobe=1/4. Some
early cases have wide timing distributions; for example the N=64, K=33 hybrid
interquartile interval is approximately 112–293 µs. Later K=128 and K=384,
D=128 hybrid intervals are approximately 199–201 µs. Small differences do not
establish speedups or performance equivalence. K=33/96/384 represent values
reachable through filter over-fetch; the actual filter/index pipeline was not
benchmarked. Broader nprobe and end-to-end measurements remain implementation
validation work.

## Recommended implementation

Use the shared-workspace hybrid, retaining the K=32 split, private heap capacity,
distance policy, query-cache boundary, and public API. This is a bounded shader
change; no new host allocation, buffer binding, or dispatch is required.

The material tradeoff is **zero instrumented memory headroom on the measured
toolchain**. Equality satisfies the current device budget, but future compiler
overhead is not guaranteed. Add an actual pipeline/device budget assertion for
both package plugin and runtime compilation, run it under shader validation,
and guard regression dispatches against process-aborting over-budget pipelines.
The budget test catches future changes; it is not a production fallback.

If spare instrumented capacity is a requirement, choose a more involved selection
design before implementation (for example, a distributed sort or multiple passes).
Reducing threadgroup width/private heap capacity changes candidate retention;
all-head selection has the measured large-K cost. Neither is a transparent fix.

Implementation validation should add permanent coverage for the fixtures above,
invalid/sentinel selected-list entries, explicitly concentrated winners, and
repeated executions at the cache boundary. Run trained IVF correctness tests
under API/shader validation in debug and release, then normal full suites.
Verify filtered searches that cross K=32 and nprobe above eight without using
timing assertions under instrumentation. Keep unrelated optional-buffer binding
failures isolated and visible. No broad instrumentation clearance is claimed here.

Expected scope: roughly 60–110 shader lines touched, 130–220 lines of permanent
regression tests, a short wrapper-comment update, and audit/contract updates.
Complexity is moderate: synchronization and coverage dominate, with no intended
public API changes.

## Reproduction and retained evidence

Environment: Apple M3 Max (40 GPU cores), Metal toolchain 32023.883, xctrace
16.0 (17F113), full Xcode at `/Applications/Xcode.app/Contents/Developer`.

Local throwaway evidence is under:
`/private/tmp/va-ivf-instrumentation/2026-09-14/`.

- `baseline-matrix.json`: runtime/plugin API-only success and instrumented
  reflection/dispatch failures, including SIGABRT exit codes.
- `prototype-matrix.json`: compiled memory measurements and small dispatches.
- `probe.swift`, `experiment.swift`, `experiment-offline.swift`: reproduction,
  independent CPU checks, and rotating-order benchmark harnesses.
- `baseline/`, `union-sort/`, `union-hybrid/`, `all-heads/`: exact source variants.
- `check-normal.log`, `check-instrumented.log`, `check-offline-instrumented.log`:
  passing correctness counts.
- `benchmark.log`: all samples summarized by median and quartiles.
- `check-debug.log`: initial fixture-rounding diagnostic on the original shader.

The baseline is recoverable from commit `b69d4e9`; temporary probes/prototypes
are experimental and are not part of the package. To reproduce the production
failure through the package:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test \
  --filter IVFValidationTests.testAllVectorsAssignedToExactlyOneCluster
```

That package command was established in the preceding fused investigation; this
investigation independently reproduced the list-kernel assertion with standalone
runtime and existing plugin-built pipelines. It did not rerun the full suite.
