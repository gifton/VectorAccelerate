# SP0 — Harness Foundation — Design Spec

**Date:** 2026-09-14
**Status:** Approved design (sections 1–4 approved by the owner in conversation on 2026-09-14); awaiting owner review of this written form before the implementation plan.
**Program:** Observability, guardrails, and reproducible benchmarks for VectorAccelerate. Research brief: `docs/superpowers/research/2026-09-14-observability-harness-research.md` (decisions in §6a). Adapter seam: `docs/superpowers/specs/2026-09-14-harness-interface-contract.md` (this spec is the authority where the two overlap; the contract is updated to match on approval). Library wish list: `docs/superpowers/research/2026-09-14-metal-improvements-wishlist-for-harness.md`. First consumer: `docs/superpowers/specs/2026-09-14-ivf-benchmark-integration.md`.
**Version impact:** VectorAccelerate **0.7.0** (breaking: removes the `Benchmarking/` public surface, the `VectorAccelerateBenchmarks` product, and `IndexAccelerationConfiguration.logDecisions`).

---

## 1. Goal and non-goals

**Goal.** Build the substrate every later sub-project writes into, and close one minimal measurement loop on it: one command produces a schema-valid artifact with full provenance for two cases, a baseline can be captured from it, and a comparison against that baseline reports a status per case. Nothing in SP0 changes what the library computes or how it routes.

**Milestone M0 (acceptance in §11).** `swift run -c release VectorAccelerateBench run --mode quick` on the M3 Max writes an artifact under `.bench/runs/`; `baseline capture` stores it; `compare` reports `ok` for an unchanged tree and `alert(slower)` for a deliberately slowed probe; every test in §10 is green in debug and release.

**Non-goals (later sub-projects).** Encoder-level kernel timing and per-submission records (SP1, blocked on wish-list HW-01..03). Validation-layer script and CI job, determinism bound, soak mode, skip audit (SP2). Signposts and decision records (SP3). The benchmark matrix, crossover sweep, recall–QPS curves, README regeneration, the two IVF throughput placeholders, VA3-024 measurement (SP4). Automation, history, viewers (SP5). Adopting ordo-one/package-benchmark (deferred to SP3 by Q9).

## 2. Decisions this spec inherits

| Q | Decision (owner, 2026-09-14) |
|---|---|
| Q1 | VectorAccelerate-only; the kit is generic by discipline so it can be lifted to VectorCore later. |
| Q2 | Non-product targets inside this package: `VectorTestKit` (library), `VectorAccelerateBench` (executable). Triggers: first bench-only external dependency moves the harness to a `Benchmarks/` sub-package; VectorCore adopting the kit moves the kit out. No app. |
| Q3 | M3 Max is the only trusted runner now; M4 Pro and M2 Pro are future environment keys. |
| Q4 | Determinism: accept and bound (SP2). |
| Q5 | Performance alerts are non-blocking; boundary `max(2.5%, 2 × baseline RSD)`; improvements alert. |
| Q6 | README performance claims retired now. |
| Q7 | Dead code deleted aggressively, with a mechanical orphan guard. |
| Q8 | Claim only hardware with a run record. |
| Q9 | ordo-one/package-benchmark deferred to SP3. |
| §2 approval | CryptoKit is on the kit's import allowlist (system framework, no third-party dependency). |

## 3. Package layout and ownership

| Path | Target / product | Owner | Notes |
|---|---|---|---|
| `Sources/VectorTestKit/` | library target `VectorTestKit`; **not a product** | harness owner | imports only `Foundation`, `Darwin`, `CryptoKit` |
| `Sources/VectorAccelerateBench/` | executable target `VectorAccelerateBench`; **not a declared product** (root packages get an implicit executable product, so `swift run VectorAccelerateBench` works) | harness owner for `Runner/`, `Store/`, `Compare/`, `Fingerprint/`, `Probe/`, `main.swift`; family owners for `Cases/<Family>/` | depends on `VectorAccelerate`, `VectorCore`, `VectorTestKit`, `Metal`, `IOKit` |
| `Sources/VectorAccelerateBench/Cases/Registry.swift` | — | shared | one line per family |
| `Tests/VectorAccelerateTests/` | test target | — | gains a dependency on `VectorTestKit` |
| `.bench/` | — | — | gitignored in SP0 (`.bench/` added to `.gitignore`); history strategy is SP5 |

Manifest changes: add the two targets; add `VectorTestKit` to the test target's dependencies; remove the `VectorAccelerateBenchmarks` executable target and product; add a `0.7.0` entry to the version-history header; SwiftLint scope gains `Sources/VectorTestKit` and `Sources/VectorAccelerateBench` (the `no_print_in_production` custom rule is scoped to exclude the bench executable, which legitimately prints reports).

## 4. `VectorTestKit`

### 4.1 Import rule

Every Swift file under `Sources/VectorTestKit/` may import only `Foundation`, `Darwin`, `CryptoKit`. No `@testable`. Enforced by `KitImportLintTests` (§10).

### 4.2 `SeededRNG` (promotion of `TestRNG`)

`Tests/VectorAccelerateTests/Utilities/TestRNG.swift` moves to `Sources/VectorTestKit/SeededRNG.swift` with the type renamed `SeededRNG`. Every public member keeps its name and semantics: `init(seed:stream:warmup:)` (warm-up default 10, stream mixing `seed ^ (stream << 32)`), `next()`, `nextFloat01()`, `nextFloat(in:)`, `nextDouble()`, `nextGaussian(mean:stdDev:)`, `nextInt(bound:)`, `nextInt(in:)` (both range forms), `nextBool(probability:)`, `shuffle`, `shuffled`, `sample(_:k:)`, and `RandomNumberGenerator` conformance. The LCG constants and the 24-bit float construction are unchanged. `public typealias TestRNG = SeededRNG` is provided, marked `@available(*, deprecated, renamed: "SeededRNG")`, and removed in 0.8.0.

**Output-stream freeze.** `SeededRNGGoldenTests` pins the first 64 values of `next()`, `nextFloat01()`, and `nextGaussian()` for seeds {0, 1, 42} × streams {0, 1} as literal arrays in the test file, generated once from the pre-move `TestRNG` and never regenerated. A failure means the sequence changed and every fixture digest in the suite is invalid.

### 4.3 `VectorGenerator` (promotion of `TestDataGenerator`)

`TestDataGenerator.swift` moves to `Sources/VectorTestKit/VectorGenerator.swift`, renamed `VectorGenerator`, same public members (`init(seed:stream:)`, `uniformVectors` both forms, `gaussianClusters`, `separatedGaussianClusters`, `unitVectors`, `clusteredUnitVectors`, `sparseVectors`, `skewedClusters`, `perturbedQueries`, `clusterQueries`, `randomQueries`, `shuffle`, `shuffled`, `sample`, `trainTestSplit`, `statistics(for:)`, `randomGenerator`, `generatedCount`) and `DatasetStatistics`. Adds `public static let version: String = "1"`. Deprecated typealias `TestDataGenerator`, removed in 0.8.0.

**Freeze.** `VectorGeneratorGoldenTests` pins, per generator method, the SHA-256 digest (§4.4) of a small fixed call (for example `uniformVectors(count: 16, dimension: 8)` at seed 42, stream 0). Any change bumps `version` deliberately, which changes every dependent case identity.

### 4.4 `ContentDigest`

`public enum ContentDigest { public static func sha256(_ parts: [DigestPart]) -> String }` where `DigestPart` is an enum over `[Float]`, `[Double]`, `[Int32]`, `[UInt32]`, `[Int64]`, `[UInt64]`, `Int`, `Double`, `String`. Canonical bytes: for arrays, a `UInt64` little-endian element count followed by each element little-endian; scalars as their fixed-width little-endian representation; strings as UTF-8 with a `UInt64` length prefix. Parts are hashed in the order given. Output is lowercase hex. `ContentDigestGoldenTests` pins the digest of one fixed part list.

### 4.5 `Tolerance` and `Comparator`

```swift
public struct Tolerance: Codable, Sendable, Equatable {
    public var absolute: Double      // default 0
    public var relative: Double      // default 0
    public var ulps: UInt64?         // optional fifth tier
    public static func forSum(terms: Int, reduction: Reduction, magnitude: Double) -> Tolerance
    public enum Reduction: String, Codable { case sequential, tree }
}
public enum Verdict: Equatable { case equal, withinTolerance, classMismatch, outOfTolerance(absError: Double, relError: Double) }
public enum Comparator {
    public static func compare(_ a: Double, _ b: Double, tolerance: Tolerance) -> Verdict
    public static func compare(_ a: Float, _ b: Float, tolerance: Tolerance) -> Verdict
}
```

Tiers, in order: both NaN → `.equal`; bitwise equal (covers ±0 and same-signed infinity) → `.equal`; class mismatch (exactly one NaN, exactly one infinite, or infinities of opposite sign) → `.classMismatch`; `|a−b| ≤ max(absolute, relative · max(|a|,|b|))` → `.withinTolerance`; if `ulps` is set and the ULP distance between `a` and `b` is ≤ `ulps` → `.withinTolerance`; else `.outOfTolerance`.

`forSum(terms: D, reduction:, magnitude: M)` returns `relative = 0` and `absolute = k · ε · D · M` for `.sequential` and `absolute = k · ε · ⌈log₂ D⌉ · M` for `.tree`, with `ε = 2⁻²⁴` (Float32) and `k = 2` as a safety factor. Rationale (documented in the doc comment): the standard forward-error bound for a sum of D terms is `(D−1)·ε·Σ|xᵢ|` sequentially and `⌈log₂D⌉·ε·Σ|xᵢ|` for a balanced tree; `magnitude` is the caller's bound on `Σ|xᵢ|`. This is an explicit contract with its own test (`ToleranceDerivationTests`), not a claim about any kernel.

### 4.6 `RecallPolicy` and `Recall`

`public struct RecallPolicy: Codable, Sendable { public var k: Int; public var epsilon: Double }`. `Recall.atK(returned: [(id: Int, distance: Double)], groundTruth: [(id: Int, distance: Double)], policy:) -> Double`: a returned neighbor counts if its distance ≤ `(1 + epsilon) × groundTruth[k−1].distance`; result is count / k, clamped to [0, 1]. Ground truth must have at least `k` entries; fewer throws. Distances are compared as `Double`.

### 4.7 `Stats`

`Stats.summarize(_ values: [Double], bootstrapSeed: UInt64) -> Summary` with `Summary { median, p90, p99, min, max, mean, rsd, ci95: (lo, hi), count }`. Percentiles use linear interpolation between order statistics (the method already in `IndexBenchmarkHarness.LatencyStats`, moved here). `rsd = sampleStdDev / mean` (0 when count < 2). `ci95` is a percentile bootstrap of the median: 1,000 resamples with replacement drawn from `SeededRNG(seed: bootstrapSeed, stream: 0)`, 2.5th and 97.5th percentiles of the resampled medians. `blackHole(_:)` (the `@inline(never)` DCE barrier from `BenchmarkFramework.swift`) also lives here.

### 4.8 Artifact schema v1 (`BenchArtifact`)

All types are `Codable`, `Sendable`, `Equatable`. Field names below are the JSON keys.

```
BenchArtifact { schemaVersion: 1, package: "VectorAccelerate", metadata: Metadata, cases: [CaseRecord] }

Metadata {
  packageVersion: String, gitSHA: String?, gitDirty: Bool, date: String (ISO-8601 UTC),
  os: String ("os26.5.2"), osBuild: String, arch: String, cpuCores: Int, performanceCores: Int?, efficiencyCores: Int?,
  memoryBytes: UInt64, deviceModel: String ("Apple M3 Max"), modelIdentifier: String ("Mac15,9"), hostname: String,
  swiftVersion: String, buildConfiguration: String ("debug"|"release"), deviceTag: String,
  environmentKey: String, gpu: GPUInfo?, power: PowerInfo, thermal: ThermalInfo, preflight: Preflight,
  validationLayers: Bool, runSeed: UInt64, runLabel: String?, mode: String, filters: Filters?, flags: [String: String]
}
GPUInfo { name: String, coreCount: Int?, unifiedMemory: Bool, maxThreadgroupBytes: Int, metal4: Bool, metalToolchain: String? }
PowerInfo { source: "ac"|"battery"|"unknown", lowPowerMode: Bool }
ThermalInfo { start: String, end: String }   // "nominal"|"fair"|"serious"|"critical"
Preflight { cpuCopyGBps: StartEnd, gpuCopyGBps: StartEnd?, driftPercent: Double, quiet: Bool, reasons: [String] }
StartEnd { start: Double, end: Double }
Filters { mode: "glob", include: [String]?, exclude: [String]? }

CaseRecord {
  id: String, identity: CaseIdentity, status: "measured"|"unavailable"|"failed", failureReason: String?,
  samples: Int, iterations: Int, warmupUsed: [Int], unitCount: Int,
  timing: Timing, correctness: Correctness?, resources: Resources?, notes: [String]
}
CaseIdentity { id: String, fixtureDigest: String, generatorVersion: String, seed: UInt64, stream: UInt64, route: String, variant: String?, dataDescriptor: String }
Timing { wall: Series?, commandBuffer: Series?, encoder: Series? }     // absent = unavailable
Series { unit: "ns", raw: [[Double]], summary: Summary, bias: String? } // raw[sample][iteration]
Summary { median, p90, p99, min, max, mean, rsd, ci95: [Double], count: Int }
Correctness { status: "pass"|"fail"|"notMeasured", comparator: String, tolerance: Tolerance?, maxAbsError: Double?, maxRelError: Double?, recall: Double?, recallPolicy: RecallPolicy?, notes: [String] }
Resources { staticThreadgroupBytes: Int?, maxThreadgroupBytes: Int?, poolHighWaterBytes: Int?, notes: [String] }
```

VectorCore compatibility: `packageVersion`, `gitSHA`, `date`, `os`, `arch`, `cpuCores`, `deviceModel`, `swiftVersion`, `buildConfiguration`, `deviceTag`, `runSeed`, `runLabel`, `filters` keep VectorCore's names and meanings; `deviceTag` is computed the same way (`<arch>-<os>-<hostname>` unless `VA_DEVICE_TAG` is set) so VectorBench can group runs. `BenchArtifact.validate()` checks: `schemaVersion == 1`; every required field present; each `Series.raw` has `samples` rows of `iterations` values; `warmupUsed.count == samples`; `status == "measured"` implies `timing.wall != nil`; `correctness.recall`, when present, is in [0, 1]; `validationLayers == true` is allowed. Decoding an artifact with an unknown `schemaVersion` fails with a typed error.

### 4.9 `EnvironmentFingerprint` (kit part) and the environment key

Gathered by the kit: `machdep.cpu.brand_string`, `hw.model`, `hw.ncpu`, `hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu`, `hw.memsize` (sysctl); OS version and build (`ProcessInfo` + `kern.osversion`); Swift version (compile-time `#if swift(>=…)` is insufficient, so the executable passes the output of `swift --version` captured at run time; the kit stores what it is given); build configuration (`#if DEBUG`); hostname; power source and Low Power Mode by running `pmset -g batt` and `pmset -g` and parsing (`"unknown"` if `pmset` is unavailable); thermal state from `ProcessInfo.thermalState` at start and end. Git SHA and dirty flag by running `git rev-parse HEAD` and `git status --porcelain` in the package directory (`nil`/`false` if not a repository).

Environment key: `"\(chipSlug)-g\(gpuCores ?? "?")-\(osBuild)-mtl\(metalToolchain ?? "unknown")-swift\(swiftVersion)-\(buildConfiguration)-\(validationLayers ? "val" : "noval")"`, where `chipSlug` is the brand string lowercased with non-alphanumerics removed (`applem3max`). Hostname is excluded by design so identical hardware on different machines compares.

### 4.10 `Preflight`

Kit part: `Preflight.cpuCopyGBps()` performs `memcpy` of a 256 MiB buffer three times and reports the best as `2 × bytes / seconds / 1e9` (read plus write). The executable supplies `gpuCopyGBps` (§5.8). `Preflight.evaluate(start:end:thermalStart:thermalEnd:) -> (quiet: Bool, driftPercent: Double, reasons: [String])`: drift is the larger of the CPU and GPU relative changes between start and end, in percent; `quiet` is false when drift > 5%, when either thermal state is not `nominal`, or when the GPU probe is unavailable while a Metal device exists. Reasons are human-readable strings recorded in the artifact.

### 4.11 `RunStore`

- Runs: `.bench/runs/<environmentKey>/<yyyyMMdd'T'HHmmss'Z'>_<short sha or "nogit">[_dirty].json`. Write is create-only; an existing path is an error.
- Baselines: `.bench/baselines/<environmentKey>/<caseID>/<n>.json` where `n` starts at 1 and increments; `approved` is a text file containing one integer. `capture(artifact:)` writes every measured case of the artifact as the next `n`; it never overwrites and never moves `approved` unless `approve: true`. Capture refuses artifacts with `validationLayers == true`, `preflight.quiet == false`, or `gitDirty == true` unless `allowDirty: true`; refusal is a typed error naming the reason.
- `approvedBaseline(environmentKey:caseID:) -> CaseRecord?` and `baseline(environmentKey:caseID:version:)`.
- The store never deletes.

## 5. `VectorAccelerateBench`

### 5.1 Adapter interface

As in the interface contract §4, restated here as the authority:

```swift
public enum TimingKind: String, Codable, Sendable { case wall, commandBuffer, encoder }
public struct GPUSample: Sendable { public var kind: TimingKind; public var startNs: Double; public var endNs: Double; public var label: String }
public struct IterationOutput: Sendable { public var gpu: [GPUSample]; public var payload: (any Sendable)? }

public protocol BenchCase: Sendable {
    var id: CaseID { get }
    var identity: CaseIdentity { get }
    var providedTiming: Set<TimingKind> { get }
    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase
}
public protocol PreparedCase: AnyObject {
    func runOnce() async throws -> IterationOutput
    func verify(_ output: IterationOutput) throws -> Correctness
    func resources() -> Resources?
    func resetForSample() async throws
}
```

`BenchContext` (executable) holds: the `MetalDevice`/`Metal4Context` if a device exists, `makeGenerator(stream:) -> VectorGenerator` seeded from the run seed, the `EnvironmentFingerprint`, and `RunOptions { mode, samples, iterations, validationLayers, requireGPU }`. `CaseID` is a validated `String` (§5.4 grammar).

### 5.2 Sampling loop

For each case: `prepare` once. Then for `sample in 0..<samples`: `resetForSample()`; warm-up: run `runOnce()` repeatedly, keeping the wall time of each, until at least 3 have run and the coefficient of variation of the last 8 (or all, if fewer) is below 0.05, or 50 have run; record the count in `warmupUsed[sample]`; then `iterations` measured calls, each wrapped in `ContinuousClock` for `.wall`, appending every `GPUSample` returned to the series of its kind; after the last iteration of the sample, `verify(lastOutput)` outside any timed region. `resources()` is called once after all samples. Any thrown error marks the case `failed` with `failureReason` and continues to the next case.

Mode defaults: `smoke` 1 × 3, `quick` 3 × 20, `full` 5 × 100 (samples × iterations); `--samples`/`--iterations` override.

### 5.3 Timing rules

The harness measures `.wall` for every case. It never synthesizes `.commandBuffer` or `.encoder` values; a kind not present in `providedTiming` is absent from the artifact. A `Series.bias` string is set to `"command-buffer duration includes submission and completion overhead (+2–10 µs measured on M3 Max, see research brief Appendix D)"` for `.commandBuffer` series. When wish-list HW-03 lands, the harness collects GPU kinds from the context's per-submission records and ignores adapter-supplied samples for those kinds; that change is SP1 and does not alter this protocol.

### 5.4 Case identity and registry

Grammar: `<family>.<op>(.<param>=<value>)*(.<variant>)?`, lowercase, `[a-z0-9_=.]` only, parameters in the order the family documents, never containing seeds, digests, or timestamps. `CaseID.init(validating:)` rejects anything else. Each family exposes `enum <Family>Cases { static var all: [any BenchCase] }` and one line in `Cases/Registry.swift`. Duplicate IDs across families are a registry error at startup.

### 5.5 CLI

```
VectorAccelerateBench list [--filter <glob>] [--exclude <glob>]
VectorAccelerateBench run  [--filter <glob>] [--exclude <glob>] [--mode smoke|quick|full] [--samples N] [--iterations N]
                           [--seed S] [--out <path>] [--label <text>] [--require-gpu] [--validation-layers] [--allow-dirty]
VectorAccelerateBench baseline capture --from <artifact> [--approve] [--allow-dirty]
VectorAccelerateBench compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
VectorAccelerateBench validate-artifact <path>
```

Argument parsing is hand-written (no `swift-argument-parser`, per the Q2 dependency trigger). Globs use `fnmatch`. `--out` defaults to the run-store path. `--validation-layers` only flags the artifact; the environment variables themselves are set by the caller (SP2 adds the script).

### 5.6 Compare

Input: a run artifact and, per case, the approved (or `--against <n>`) baseline `CaseRecord` under the same `environmentKey`. Per case, in this order:

1. `invalidArtifact` if either record fails `validate()`.
2. `incomparable` if `identity != baseline.identity` or the environment keys differ or the baseline is missing.
3. `correctnessFailed` if `correctness.status == "fail"`.
4. `resourceFailed` if `resources.staticThreadgroupBytes > resources.maxThreadgroupBytes` (both present) or any note in `resources.notes` begins with `LIMIT_EXCEEDED:`.
5. For each timing kind present in both: `delta = (new.median − base.median) / base.median`; `boundary = max(0.025, 2 × base.rsd)`; `alert(slower)` if `delta > boundary`, `alert(faster)` if `delta < −boundary`. A case with any alerting kind is reported as that alert; kinds are listed individually in the report with `delta`, `boundary`, and whether the boundary was widened beyond 2.5%.
6. Otherwise `ok`.

Exit status: 1 if any case is `invalidArtifact`, `correctnessFailed`, or `resourceFailed`; 1 for alerts only under `--strict-perf`; 2 for usage errors; 0 otherwise. Output formats: `text` (table), `markdown` (table plus a per-alert line), `json` (`[CompareResult]`).

### 5.7 First cases (M0)

**`probe.gpu_copy.bytes=134217728`** (`Cases/Probe/`): the spike's `float4` copy kernel compiled from an MSL string at runtime; one legacy `MTLCommandBuffer` per `runOnce()`, waited on synchronously; `.commandBuffer` sample from `gpuStartTime`/`gpuEndTime`. Correctness: `memcmp` of output against input once per sample in `verify`. Resources: the pipeline's `staticThreadgroupMemoryLength` and the device limit. Identity: `route = "gpu.probe_copy"`, `dataDescriptor = "sequential_u32_pattern"`, fixture digest over the pattern parameters. Purpose: known cost (expected ≈0.36 ms, ≈370 GB/s on M3 Max), validates the timing path end to end, and is the deliberate-slowdown vehicle for M0 (a `--probe-slowdown N` hidden option makes the kernel loop N extra times; used only by the M0 acceptance test, recorded in `notes`).

**`distance.l2.batch.n=10000.d=768`** (`Cases/Distance/`): candidates from `VectorGenerator.uniformVectors(count: 10000, dimension: 768)` and one query, seeded per case; a `Metal4Context` created by the case and a `MetalComputeProvider(context:configuration: .init(preferGPU: true, fallbackToCPU: false))` over it; `runOnce()` calls `batchDistance(query:candidates:metric: .euclidean)` and, immediately after, reads `context.lastGPUTiming` as the `.commandBuffer` sample. Route verification: `routingTelemetry().gpuKernel` must increase by exactly one per call; otherwise `verify` returns `fail` with note `route mismatch`. Correctness: the returned distances are compared with a Double-precision CPU oracle over the same inputs using `Tolerance.forSum(terms: 768, reduction: .tree, magnitude: Σ|xᵢ| bound from the generator range)`; `maxAbsError`/`maxRelError` recorded. Identity: `route = "gpu.l2.batch"`, `dataDescriptor = "uniform[-1,1]"`, digest over candidates and query.

### 5.8 Metal fingerprint and GPU probe (executable)

`MetalFingerprint.gather() -> GPUInfo?`: `MTLCreateSystemDefaultDevice()` name, `hasUnifiedMemory`, `maxThreadgroupMemoryLength`, `supportsFamily(.metal4)`; `coreCount` from the IORegistry property `gpu-core-count` on the first `AGXAccelerator` service (via IOKit), `nil` if absent; `metalToolchain` from `xcrun -sdk macosx metal --version` (first line, version token), `nil` on failure. `GPUProbe.copyGBps()` runs the probe kernel 5 warm-up + 10 measured times and returns `128 MiB / median command-buffer seconds / 1e9`.

## 6. Deletions and migrations

| Item | Action | Breaking? |
|---|---|---|
| `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`, `IndexBenchmarkHarness.swift` | delete; `blackHole` and percentile interpolation move to `Stats` | **yes** (public types removed) |
| `Tests/.../IndexBenchmarkHarnessTests.swift`, `IndexBenchmarkRecallValidationTests.swift` | the recall assertions (flat recall exactly 1.0; IVF recall > 0.9) are ported to a new `Tests/.../IndexRecallTests.swift` using `VectorGenerator` and the public index API **before** the harness files are deleted; the two old files are then deleted | no |
| `Sources/VectorAccelerateBenchmarks/` (5 files) and the `VectorAccelerateBenchmarks` product | delete | yes (product removed) |
| `Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift` | delete | yes (public, unused) |
| `IndexAccelerationConfiguration.logDecisions` | delete field and its init parameter | yes |
| `Tests/.../PerformanceBenchmarks.swift` | delete | no |
| private `SeededRNG` in `GPUCandidateBuilderValidationTests.swift` | replace with kit `SeededRNG`; the test must still pass (it asserts properties, not a specific sequence). The other private copy, in `IndexBenchmarkRecallValidationTests.swift`, disappears with that file (row above). | no |
| `Tests/.../Utilities/TestRNG.swift`, `TestDataGenerator.swift` | moved to the kit (§4.2, §4.3) | no |
| `README.md` line 5 "up to 100x speedups", lines 582–586 table, line 626 command | line 5 loses the number; the table becomes "Performance is measured by the benchmark harness; the latest run records live under `.bench/runs/` and are not committed. See `docs/superpowers/research/2026-09-14-observability-harness-research.md`."; line 626 becomes `swift run -c release VectorAccelerateBench run --mode quick` | no |
| `.gitignore` | add `.bench/` | no |
| `CHANGELOG.md`, `Package.swift` header | `[0.7.0] — Unreleased` entry listing the removals and the two new targets | — |

Not touched in SP0: `MLIntegrationBenchmarkTests.swift`, `SwiftTopicsBenchmarks.swift`, the `PerfGate`-gated assertions, the XCTest `measure {}` uses (SP4 re-homes them); `Logger.swift`; any kernel or `Metal4Context` code.

## 7. Error handling

| Condition | Behaviour |
|---|---|
| No Metal device | Metal cases recorded as `status: "unavailable"` with reason; run exits 0 unless `--require-gpu` (then 1). The CPU preflight still runs; `gpu` metadata is `nil`. |
| Case throws in `prepare`/`runOnce`/`verify` | case `status: "failed"`, `failureReason` set, run continues; exit 1 at the end. |
| Dirty tree | artifact name suffixed `_dirty`, `gitDirty: true`; capture refuses without `--allow-dirty`. |
| Preflight not quiet | artifact written with `quiet: false` and reasons; capture refuses; compare proceeds and prints a warning. |
| Schema validation fails on write | run exits 1 with the validation error; nothing is written (this is a harness bug). |
| Registry duplicate ID | startup error, exit 2. |
| Baseline missing | compare status `incomparable` for that case; exit 0. |
| Run-store path exists | error, exit 1 (never overwrite). |

## 8. Performance and cost of the harness itself

The harness allocates fixture data in `prepare`, never inside `runOnce`. The two M0 cases hold their buffers for the whole run. The bootstrap costs 1,000 median computations per series and is negligible next to any case. Preflight adds roughly one second per run (two copies of 256 MiB and 2 × 15 GPU dispatches). Nothing in SP0 is on a library hot path.

## 9. Security and hygiene

The fingerprint runs `git`, `pmset`, `swift`, and `xcrun` as subprocesses with fixed arguments and no user-controlled input; each has a 5-second timeout and degrades to `nil`/`"unknown"`. Artifacts contain the hostname; that is intentional for VectorBench compatibility and documented. No network access anywhere.

## 10. Tests (all in `Tests/VectorAccelerateTests/`, XCTest, run in debug and release)

| Test | Closes |
|---|---|
| `Harness/KitImportLintTests` — scans `Sources/VectorTestKit/**/*.swift` for `import` lines; fails on anything outside {Foundation, Darwin, CryptoKit} or any `@testable` | genericism rule |
| `Harness/OrphanPublicTypeTests` — regex-collects `public (struct|class|final class|enum|actor|protocol|typealias) Name` in `Sources/VectorAccelerate`; fails if `\bName\b` occurs nowhere in `Sources/` + `Tests/` outside the declaring file **and** `Name` is not in the in-test allowlist. The allowlist is seeded at implementation time with every currently-unreferenced intended-API type, each with a one-line justification; the test is a ratchet against new orphans. | Q7 class |
| `Harness/SeededRNGGoldenTests`, `VectorGeneratorGoldenTests`, `ContentDigestGoldenTests` | stream and digest freeze |
| `Harness/ComparatorTests` — the ten adversarial value classes from `DifferentialKernelVsCPUTests` (normal, zeros, duplicates, 1e19, 1e-20, subnormal, mixed scale, NaN, +Inf, −Inf) crossed pairwise; asserts tier outcomes; plus `ToleranceDerivationTests` for `forSum` | comparator contract |
| `Harness/RecallTests` — epsilon form vs ID-set form on a constructed tie case | recall policy |
| `Harness/StatsTests` — percentiles against hand-computed values; bootstrap CI reproducible for a fixed seed and contains the median | statistics |
| `Harness/ArtifactSchemaTests` — encode/decode round-trip of a full synthetic artifact; `validate()` fails on each required field removed in turn; unknown `schemaVersion` rejected | schema |
| `Harness/FingerprintTests` — kit fingerprint populates chip, cores, memory, OS build; environment key excludes hostname; key changes when `validationLayers` flips | fingerprint |
| `Harness/RunStoreTests` — in a temporary directory: run write is create-only; capture appends `n+1` and leaves `approved` alone; `--approve` moves it; capture refuses dirty, non-quiet, and validation artifacts with the right error | store |
| `Harness/CompareTests` — synthetic artifact pairs produce each status: `ok`, `alert(slower)`, `alert(faster)`, widened boundary when baseline RSD > 1.25%, `incomparable` (identity, key, missing), `correctnessFailed`, `resourceFailed`, `invalidArtifact`; exit-code mapping with and without `--strict-perf` | decision rule |
| `Harness/CaseIDTests` — grammar accepts the two M0 IDs and rejects seeds/uppercase/spaces | identity |
| `Harness/RegistryTests` — duplicate ID detection | registry |
| `IndexRecallTests` (ported) — flat recall exactly 1.0; IVF recall > 0.9 on a seeded clustered set | keeps existing coverage |

The M0 end-to-end run (§11) is a script step, not a unit test, because it needs the release build and a real GPU.

## 11. Milestone M0 acceptance

Run on the M3 Max, AC power, in this order, recording the commands and outputs in the implementation ledger:

1. `swift test` and `swift test -c release`: all green; expected count = previous gate (1764) − deleted (`PerformanceBenchmarks`, the two old index-harness suites) + new (§10) + ported (`IndexRecallTests`); the two remaining IVF throughput skips are unchanged.
2. `swift run -c release VectorAccelerateBench run --mode quick` writes one artifact; `validate-artifact` on it exits 0; the artifact has both M0 cases `measured`, `.wall` and `.commandBuffer` series for both, non-nil `gpu`, `preflight.quiet == true`, and correctness `pass` for both.
3. `baseline capture --from <artifact> --approve` creates `.bench/baselines/<key>/<id>/1.json` and `approved` for both cases.
4. A second `run` followed by `compare --against approved` reports `ok` for both cases (or an `alert` with a visibly widened boundary if the probe's own RSD exceeded 1.25%, which is itself recorded).
5. `run --probe-slowdown 4` then `compare` reports `alert(slower)` for the probe case and exit 0; with `--strict-perf` exit 1.
6. `run` on a dirty tree produces `_dirty` and `baseline capture` refuses it.
7. `MTL_DEBUG_LAYER=1 swift run -c release VectorAccelerateBench run --mode smoke --validation-layers` exits 0 (both cases are validation-clean) and `baseline capture` refuses the artifact.

## 12. Open items carried to later sub-projects

- Ground-truth artifact caching under `.bench/fixtures/` (SP4, when a real dataset arrives).
- Warm-up window and CoV threshold retuned from SP1's variance-profiling run; the values here (8, 0.05, cap 50) are the initial contract and are recorded in every artifact.
- `.bench/` history and a `bench-history` branch (SP5).
- The interface contract's §10 is superseded by this section.
