# IVF benchmark integration requirements

2026-09-14. Prepared after bounded instrumentation fix `2c57337`.
Status: workload and acceptance requirements; shared-suite integration location is
awaiting coordination with its owner. No benchmark implementation is claimed here.

## Purpose and ownership

Make an IVF implementation change reviewable through reproducible measurements of
correctness, search quality, execution time, and compiled memory usage. Start with
one workload that can be captured, compared against a stored baseline, and shown
to detect an injected regression; then expand to the matrix below.

Use the shared runner, schema, environment fingerprint, sampling, baseline storage,
and comparison policy from the observability program. Do not create a competing
IVF-specific framework or duplicate RNG/statistics infrastructure. The research
brief records non-product `VectorTestKit` and `VectorAccelerateBench` targets, but
neither exists in this checkout at the start of this integration task. Their
implementation location and ownership must be established before wiring adapters.

The recorded owner policy is authoritative for this integration: performance
changes alert in either direction, without blocking; the per-case boundary is
`max(2.5%, 2 × baseline RSD)`, with widening visible in reports. Correctness failures,
invalid artifacts and exceeded resource budgets remain explicit failures. A speed
regression must be detectable in the comparison result; it need not produce a
nonzero exit status under the default alert-only policy.

## Existing reusable coverage

`IVFListInstrumentationTests` covers compiled budget, retained-pool ordering,
K=31/32/33 and larger K, full-SIMD widths, invalid/empty lists, nonfinite ordering,
cache-boundary reuse, and trained/filter/coarse routing. Slice 35's instrumented
integration selection passes 24 tests in each build configuration. Those tests
remain correctness gates; their XCTest durations are not benchmark samples.

The existing `IndexBenchmarkHarness` is not sufficient as the new adapter without
changes: its RNG advances across cases, its IVF configuration leaves routing at
the default, batch samples duplicate batch-average time for each query, and paths
without measured recall report 1.0. The new integration must avoid inheriting these
behaviors. This task does not authorize unrelated cleanup of that older framework.

## First complete workload

Use an isolated `ivf_list_search` dispatch with preallocated buffers and a fixed
CSR fixture: N=4096, D=128, Q=16, nlist=16, nprobe=4, K=33, width=256. Freeze the
selected-list IDs as part of the fixture so coarse selection is outside this
measurement. Generate/reset fixture data per case from the shared seeded generator,
and record a content digest covering vectors, queries, offsets, original IDs, and
selected lists. The generator version and seed belong to case identity.

Measure a single isolated command buffer containing the dispatch, and label its
GPU timestamps **command-buffer GPU duration**, not pure shader execution time.
Record completed output outside the timed interval; independently check retained
candidate membership/order, distances and padding. Capture the actual compiled
threadgroup footprint separately. Runs with validation enabled must never enter
the performance baseline population.

Before widening, demonstrate:

1. Repeated generation yields identical fixture digests and reference results,
   independent of case ordering or filtering.
2. One command emits an artifact accepted by the shared schema with provenance,
   raw timing samples, sample/warm-up counts, correctness and memory fields.
3. Baseline capture preserves the approved original artifact rather than silently
   replacing it during a comparison.
4. Comparing a changed timing fixture produces the expected performance alert;
   changed hardware/mode/fixture identity produces an incomparable result.
5. An actual controlled GPU slowdown on an experimental variant produces a measured
   alert. Keep its fixture, output and resource contracts unchanged; do not simulate
   the measured experiment by editing recorded timing numbers.
6. An injected wrong output or invalid memory-budget record fails the appropriate
   correctness/resource guard. Restore all injected faults afterward.

## Matrix expansion

Use named cases and controlled sweeps, not the full Cartesian product. All public
cases must use configurations supported by their actual routing path.

| Family | Cases to include | What must be distinguishable |
|---|---|---|
| Selection transition | K=31, 32, 33 with all other inputs fixed | Cost of switching selection algorithms |
| Large selection | K=64, 128, 384 | Growth in selection cost with K |
| Embedding dimensions | D=128, 384, 768 on supported public routes | Distance-work sensitivity |
| Raw cache boundary | D=2047, 2048, 2049 in isolated list dispatch | Cache/device-read transition without implying new public support |
| Probe transition | nprobe=1, 4, 8, 9, 16, bounded by nlist | Coarse-routing transition and actual candidate work |
| List balance | Balanced lists; one dominant list; empty selected lists | Unequal work and candidate concentration |
| Batch size | Q=1, 8, 32 with the same query corpus | Per-call latency versus batch throughput |
| Filtering | Requested K=10/11 (internal K=30/33), larger K, multiple acceptance rates | Filter-driven selection transition and underfilled results |

For list-balance cases, record actual per-query selected candidate counts and list
sizes. When assessing list-distribution effects independently, match total selected
candidate counts; equal nprobe alone is not an equal-work comparison.

For batching, report measured batch latency and completed queries per second.
Amortized time per query may be a separate derived metric; do not present replicated
batch averages as independent single-query latency samples.

For filtering, record requested K, internal over-fetch K, acceptance predicate/version,
eligible corpus size, candidates tested and result count where observed. Fixed 3×
over-fetch can underfill results at low acceptance rates. Report that outcome and
recall against eligible ground truth; do not silently increase over-fetch or credit
fewer returned neighbors as a speed improvement.

## Three measurement levels

1. **List kernel:** explicit CSR/selected lists, known actual work, retained-pool
   correctness oracle, isolated command-buffer GPU duration.
2. **Coarse plus list pipeline:** fixed centroids/CSR, actual list selection,
   end-to-end call latency; separate per-submission GPU measurements only when the
   shared timing substrate can collect every submission.
3. **Public trained index:** forced IVF routing, training outside the search timing,
   batches and filtering, independently measured recall and returned-result counts.

The current `Metal4Context.lastGPUTiming` holds only the most recent submission. It
must not be reported as total search GPU time when nprobe>8 uses multiple command
buffers. Unsupported timing detail is absent and explicitly labeled unavailable.
Do not change Metal4Context's submission architecture as a side effect of this task.

A seeded dataset alone does not make training reproducible. Use fixed prepared
structures for kernel/pipeline comparisons. For public-index measurements, freeze
and identify a trained fixture when the shared substrate supports it; otherwise
record layout/quality variation and keep those results out of strict implementation
comparisons that require identical work. The small repeated-center regression
fixture remains useful for routing correctness, not broad quality claims.

## Correctness, quality, and resources

- Preserve the raw kernel's existing eight-candidates-per-lane retained pool.
  Compare its selection against an independent CPU ordering of that pool. Report
  global recall separately; never equate retained-pool correctness with exact global K.
- Use the shared numerical comparator and explicit tolerance metadata. Nonfinite
  correctness fixtures remain validation cases rather than normal performance data.
- Compute public-index ground truth independently over the eligible dataset, with
  the suite's declared tie/epsilon policy. Missing recall is missing data, never 1.0.
- Verify the actual route; an IVF-labeled configuration that executes flat search
  is ineligible for an IVF baseline.
- Check compiled static memory against device capacity in both library paths,
  and run instrumentation separately. Current bounded list storage is 16,384 bytes
  normal / 32,768 instrumented on the measured M3 Max/toolchain. This is a limit
  check, not a promise that future compilers preserve those exact numbers.

## Integration sequence and completion criteria

1. Coordinate the shared suite checkout and the ownership of runner/schema/timing.
2. Implement one IVF adapter against those interfaces, plus deterministic fixture
   identity and independent correctness checks. Keep Metal dependencies out of the
   generic kit as required by the recorded architecture decision.
3. Connect capture/compare and demonstrate the first complete workload's acceptance
   checks, including actual slowdown detection.
4. Expand the named matrix, adding pipeline/public-index adapters and honest quality,
   candidate-work and timing fields as the shared interfaces permit.
5. Document one-command reproduction and preserve artifacts with the shared storage
   policy. Mark unsupported combinations explicitly; do not silently skip them.

Completion means reproducible IVF cases participate in the shared suite's artifacts,
baselines and comparison reports, with correctness/resource failures and performance
alerts demonstrated. Printing timings, passing existing tests, or adding this document
alone does not complete the integration.
