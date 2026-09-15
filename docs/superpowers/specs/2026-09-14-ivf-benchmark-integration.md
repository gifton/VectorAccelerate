# IVF benchmark integration requirements

2026-09-14. Prepared after bounded instrumentation fix `2c57337`.
Status: ownership and interfaces coordinated; test-target fixture/oracle preparation
is in progress. Shared-suite adapters and measured benchmark integration remain pending.

## Purpose and ownership

Make an IVF implementation change reviewable through reproducible measurements of
correctness, search quality, execution time, and compiled memory usage. Start with
one workload that can be captured, compared against a stored baseline, and shown
to detect an injected regression; then expand to the matrix below.

Use the shared runner, schema, environment fingerprint, sampling, baseline storage,
and comparison policy from the observability program. Do not create a competing
IVF-specific framework or duplicate RNG/statistics infrastructure. The owner has
confirmed two future non-product targets in this package: `Sources/VectorTestKit/`
(generic kit; imports only Foundation, Darwin, CryptoKit) and
`Sources/VectorAccelerateBench/` (runner, store, comparison and CLI). The IVF family
owns `Sources/VectorAccelerateBench/Cases/IVF/` and one registration line in
`Cases/Registry.swift`. The shared owner builds the targets and infrastructure.

The [harness interface contract](2026-09-14-harness-interface-contract.md) defines
the adapter seam. `BenchCase` prepares outside timing; `PreparedCase.runOnce()`
executes one iteration, with verification outside timing and optional resource
facts. The harness owns wall time. The isolated list dispatch will report
`.commandBuffer` GPU timing; future context submission records can replace its
collection without changing the adapter interface. No GPU duration is fabricated.

Before the shared owner's M0, implementation is limited to test-target pure fixture
builders, content identity, independent CPU oracles and correctness checks using
the existing `TestRNG` and `TestDataGenerator`. No RNG, statistics, schema, timing
loop, artifact writer, baseline store, comparison or environment capture is added.

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

1. Ownership and the shared adapter seam are coordinated. Finish the test-target
   fixtures and oracles described below while the shared owner delivers M0.
2. After M0, implement one IVF adapter against those interfaces, plus deterministic fixture
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

## Preparatory fixture layer

The test-target implementation lives in `Tests/VectorAccelerateTests/Fixtures/IVF/`,
with CPU tests in `IVFBenchmarkFixtureTests.swift` and GPU correctness checks in
`IVFBenchmarkFixtureGPUTests.swift`. Its prepared data is CPU-owned CSR, original
IDs, centroids, queries and frozen selected lists; it contains no Metal buffers or
benchmark run schema.

- Each factory call resets the existing seeded utilities. Streams `s` through
  `s+3` generate centroids, vector noise, queries and shuffled original IDs.
  Query-count sweeps preserve the prefix of the query corpus and selected lists.
- Controlled binary-grid clusters provide exactly representable squared distances
  for the tested dimensions. They measure controlled kernel workloads; they are
  not representative embedding-quality datasets.
- Literal Float bit patterns and ID/list arrays pin a small prepared fixture for
  generator version one. Promotion to `VectorGenerator` must retain those outputs;
  the shared `VectorGenerator.version` becomes part of case identity at integration.
- The exact oracle independently sorts Double squared distances across the eligible
  corpus. The retained-pool oracle independently sorts candidates per lane, retains
  eight per lane, then sorts their union. A literal adversarial test proves that
  retained-pool correctness can differ from global exact K.
- GPU checks compare IDs, distances, padding and output canaries for the first
  workload, K=31/32/33/128/384, uneven and empty lists, widths 32/96/256 and
  D=2047/2048/2049. They exercise available plugin/runtime library paths and check
  compiled memory capacity. These dispatches collect no timing samples.

One digest contract gap remains: the shared canonical format says count-prefixed
little-endian elements but has not fixed the count-prefix width or scalar encoding.
The proposed encoding is UInt64 little-endian counts, Float bit patterns and
UInt32 arrays in little-endian form, then count-prefixed UInt64 shape scalars.
Fixture field order and a layout version will be pinned once the owner confirms
that encoding. Do not publish a temporary incompatible digest.

This layer does not register benchmark cases, emit artifacts, measure recall or
performance, or establish baselines. Filtered over-fetch and broader probe/batch
sweeps still need the pipeline/public adapters after M0, even though existing
instrumentation tests already provide routing correctness coverage.

### Preparatory verification (2026-09-14)

The eight new tests pass in normal debug. The combined selection of
`IVFBenchmarkFixtureTests|IVFBenchmarkFixtureGPUTests|IVFListInstrumentationTests`
passes 13 tests with zero failures or skips in normal release and with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` in both debug and release. Instrumented
pipelines report 32,768 bytes against the measured device's 32,768-byte capacity;
the normal release pipeline reports 16,384 bytes. Independent review found no
important issues in this preparation layer; digest review is still pending.

Reproduce with `swift test --filter 'IVFBenchmarkFixtureTests|IVFBenchmarkFixtureGPUTests|IVFListInstrumentationTests'`,
adding `-c release` for release and the environment variables above for validation.
Local logs are under `/private/tmp/va-ivf-fixtures/2026-09-14/`. These are focused
correctness checks, not a rerun of the entire package suite or benchmark evidence.
