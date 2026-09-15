# Harness interface contract — DRAFT

**Date:** 2026-09-14
**Status:** Approved seam (owner approval 2026-09-14). The authoritative definitions now live in `2026-09-14-harness-sp0-foundation-design.md` (§4.5 tolerance, §4.8 schema, §5.1 adapter interface, §5.4 identity, §5.5 CLI, §5.6 compare, §4.11 store); this file is the adapter author's summary and defers to the spec wherever they differ.
**Who this is for:** anyone writing benchmark cases against the shared suite, first of all the IVF integration (`2026-09-14-ivf-benchmark-integration.md`).
**What it settles:** where the suite lives, who owns which directories, what an adapter conforms to, what the artifact contains, and what may be built before the suite skeleton exists.

---

## 1. Where the suite lives and who owns it

Both targets live in this package. Neither is a product. Neither exists yet; SP0 creates them.

| Path | Target | Owner | Contents |
|---|---|---|---|
| `Sources/VectorTestKit/` | `VectorTestKit` (library) | harness owner | seeded RNG, generators, digest, comparator, oracles, statistics, schema types, environment fingerprint, preflight probe, run store. **Imports only Foundation, Darwin, CryptoKit** (enforced by an import-lint test). No VectorAccelerate, VectorCore, Metal, XCTest. |
| `Sources/VectorAccelerateBench/Runner/`, `Store/`, `Compare/`, `Fingerprint/`, `main.swift` | `VectorAccelerateBench` (executable) | harness owner | sampling loop, warm-up, capture, baseline, compare, CLI, Metal part of the fingerprint |
| `Sources/VectorAccelerateBench/Cases/<Family>/` | same executable | family owner (IVF: the IVF agent) | adapters conforming to §4, fixture builders, family-specific oracles |
| `Sources/VectorAccelerateBench/Cases/Registry.swift` | same | shared, one line per family | the only file two owners both edit |
| `Tests/VectorAccelerateTests/Utilities/` | test target | harness owner for `PerfGate`, `requireMetalDevice()`; kit files are removed from here once promoted | |

## 2. What may be built before the skeleton exists

**Allowed now, in the test target or under `Cases/IVF/` once the directory exists:**
- Fixture builders as pure functions from seeded inputs to prepared structures (CSR lists, selected-list IDs, queries), plus their content digest.
- Independent CPU oracles (retained-pool ordering, exact ground truth in Double).
- Correctness checks that use those oracles.
- XCTest coverage of all of the above, using the existing `TestRNG` and `TestDataGenerator` under the compatibility commitment in §3.

**Not allowed before SP0's first milestone (M0):** any RNG, statistics, schema, timing loop, artifact writer, baseline store, comparison, or environment capture. If a case needs one of these, it is a gap in this contract; raise it rather than building a local version.

## 3. Kit API compatibility commitment

SP0 promotes the two test utilities into the kit **without changing their output streams**:

| Today (`Tests/.../Utilities/`) | In the kit | Commitment |
|---|---|---|
| `TestRNG` — `init(seed:stream:warmup:)`, `next()`, `nextFloat01()`, `nextFloat(in:)`, `nextDouble()`, `nextGaussian(mean:stdDev:)`, `nextInt(bound:)`, `nextInt(in:)`, `nextBool(probability:)`, `shuffle`, `shuffled`, `sample(_:k:)` | `SeededRNG`, same members; `TestRNG` kept as a deprecated typealias for one release | a golden test pins the first 64 outputs of `next()`, `nextFloat01()`, and `nextGaussian()` for seeds 0, 1, 42 and streams 0, 1. Any change to the sequence fails that test. |
| `TestDataGenerator` — `init(seed:stream:)`, `uniformVectors`, `gaussianClusters`, `separatedGaussianClusters`, `unitVectors`, `clusteredUnitVectors`, `sparseVectors`, `skewedClusters`, `perturbedQueries`, `clusterQueries`, `randomQueries`, `trainTestSplit`, `statistics(for:)` | `VectorGenerator`, same members | same golden mechanism on a small fixture per method; a `public static let version = "1"` is added and is part of case identity (§5). |

Consequence for adapters: a fixture digest computed today against `TestRNG` will be identical after promotion. If the golden test ever has to change, `VectorGenerator.version` increments and every dependent case identity changes with it, which is the intended behaviour.

**Digest:** `ContentDigest.sha256(of:)` over canonical little-endian bytes of, in order, every `[Float]` array, every `[Int32]`/`[UInt32]` array, and every scalar the fixture declares, each prefixed by its element count. Rendered as lowercase hex. CryptoKit is on the kit allowlist for this.

## 4. Adapter interface

```swift
public enum TimingKind: String, Codable, Sendable { case wall, commandBuffer, encoder }

public protocol BenchCase: Sendable {
    /// Stable ID per the grammar in §5. Never contains seeds, digests, or timestamps.
    var id: CaseID { get }
    /// Everything that determines the work. Two runs are comparable only if identities are equal.
    var identity: CaseIdentity { get }
    /// Clocks this case can supply itself (today: `.commandBuffer` for an isolated dispatch).
    /// `.wall` is always measured by the harness. Missing kinds are reported as unavailable, never 0.
    var providedTiming: Set<TimingKind> { get }
    /// Allocate buffers, build fixtures, compile pipelines. Outside every timed region.
    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase
}

public protocol PreparedCase: AnyObject {
    /// Exactly one measured iteration of the work. Must not allocate fixture data.
    /// Returns any GPU timing the case collected itself, labelled by kind.
    func runOnce() async throws -> IterationOutput
    /// Called on the last iteration of each sample, outside the timed region.
    func verify(_ output: IterationOutput) throws -> Correctness
    /// Optional resource facts (compiled static threadgroup bytes, pool high-water mark).
    func resources() -> Resources?
    /// Called between samples (outer loop). Reset state; do not regenerate fixtures.
    func resetForSample() async throws
}

public struct IterationOutput: Sendable {
    public var gpu: [GPUSample]          // may be empty; kind + start/end in nanoseconds + label
    public var payload: (any Sendable)?  // whatever `verify` needs; not serialized
}
```

`BenchContext` carries the Metal device/context, the seeded generator factory, the environment fingerprint, and the run options (`validationLayers: Bool`). The harness wraps `runOnce()` in `ContinuousClock` for `.wall`; it never invents a GPU number. When the library gains per-submission timing records (wish-list HW-03), the harness collects `.commandBuffer` and `.encoder` samples from the context instead of from `IterationOutput`; adapters do not change.

Sampling is two-level: `samples` outer × `iterations` inner, with adaptive warm-up per sample (coefficient of variation of a sliding window below a threshold, bounded by a maximum; the count actually used is recorded). All raw per-iteration values are kept in the artifact.

`Correctness` is `{ status: pass|fail|notMeasured, comparator: String, tolerance: Tolerance?, maxAbsError, maxRelError, recall: Double?, recallPolicy: {k, epsilon}?, notes: [String] }`. Missing recall is `nil`, never `1.0`. A `fail` status fails the run regardless of the performance policy.

`Resources` is `{ staticThreadgroupBytes: Int?, maxThreadgroupBytes: Int?, poolHighWaterBytes: Int?, notes }`. A recorded value above its limit fails the run.

## 5. Case identity and ID grammar

`CaseID` is dotted, lowercase, parameters in a fixed order per family, never containing seeds or digests:

```
<family>.<op>.<param>=<value>[.<param>=<value>...][.<variant>]
ivf.list_search.n=4096.d=128.q=16.nlist=16.nprobe=4.k=33.w=256.isolated
```

`CaseIdentity` = `{ id, fixtureDigest, generatorVersion, seed, stream, route, variant, dataDescriptor }` where `route` names the path actually verified to run (for example `gpu.ivf_list_search`) and `dataDescriptor` names the distribution (`uniform`, `gaussianClusters(k=8)`, `oneDominantList`). Comparison requires equal `CaseIdentity` **and** equal environment key; anything else is `incomparable`, which is a distinct report status, not an error.

## 6. Artifact schema v1 (JSON)

Top level: `schemaVersion: 1`, `package: "VectorAccelerate"`, `metadata`, `cases: []`.

`metadata` keeps VectorCore's `BenchMetadata` field names where the meaning is identical (`packageVersion`, `gitSHA`, `date`, `os`, `arch`, `cpuCores`, `deviceModel`, `swiftVersion`, `buildConfiguration`, `deviceTag`, `runSeed`, `runLabel`, `filters`) and adds: `gitDirty: Bool`, `environmentKey: String` (chip, GPU cores, OS build, Metal toolchain, Swift, build config, validation on/off), `gpu: { name, coreCount, unifiedMemory, maxThreadgroupBytes, metalToolchain }`, `power: { source, lowPowerMode }`, `thermal: { start, end }`, `preflight: { cpuCopyGBps, gpuCopyGBps, driftPercent, quiet: Bool }`, `validationLayers: Bool`, `flags: [String: String]`.

Each case: `id`, `identity`, `samples`, `iterations`, `warmupUsed: [Int]`, `timing: { wall: Series, commandBuffer: Series?, encoder: Series? }`, `correctness`, `resources`, `unitCount`, `notes`. A `Series` is `{ raw: [[Double]], summary: { median, p90, p99, min, max, mean, rsd, ci95: [lo, hi] } }` in nanoseconds, raw kept per sample. Fields absent from `timing` mean unavailable.

Artifacts with `validationLayers: true` are accepted by the schema and are **never** eligible as baselines.

## 7. Store, baseline, compare

- Runs: `.bench/runs/<environmentKey>/<UTC timestamp>_<short sha>[_dirty].json`. Append-only.
- Baselines: `.bench/baselines/<environmentKey>/<caseID>/<n>.json` plus `approved` containing the number of the approved version. Capture never overwrites; it appends `<n+1>` and does not move `approved` unless `--approve` is passed.
- Compare statuses per case: `ok`, `alert(slower)`, `alert(faster)`, `incomparable`, `correctnessFailed`, `resourceFailed`, `invalidArtifact`.
- Performance boundary: `max(2.5%, 2 × baseline RSD)` on the median, per case and per timing kind, widening shown in the report.
- Exit status: nonzero for `correctnessFailed`, `resourceFailed`, `invalidArtifact`; zero for performance alerts unless `--strict-perf` is passed. The report always lists alerts.

## 8. CLI surface (initial)

```
VectorAccelerateBench list [--filter <glob>]
VectorAccelerateBench run  [--filter <glob>] [--mode smoke|quick|full] [--samples N] [--iterations N] [--seed S] [--out <path>] [--label <text>]
VectorAccelerateBench baseline capture --from <artifact> [--approve]
VectorAccelerateBench compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
VectorAccelerateBench validate-artifact <path>
```

Run with `swift run -c release VectorAccelerateBench ...`. Validation-layer runs are started by `Scripts/validate.sh`, which sets `MTL_DEBUG_LAYER`/`MTL_SHADER_VALIDATION` before the process starts and passes `--validation-layers` so the artifact is flagged.

## 9. Registration

Each family exposes `enum <Family>Cases { static var all: [any BenchCase] }` in its own directory and adds one line to `Cases/Registry.swift`. Families never import each other.

## 10. Open points (superseded by spec §12; kept for history)

- Whether ground-truth artifacts (query IDs, k neighbour IDs, k distances, digest-named) are cached under `.bench/fixtures/` or committed. Default: cached, not committed, until a real embedding dataset is added.
- The exact adaptive warm-up window and CoV threshold are set from SP1's variance-profiling run, not chosen here; the schema already records what was used.
- The kit's `BenchContext` cannot reference Metal types (import rule). The executable defines `BenchContext`; the kit defines only the data types. Adapters live in the executable, so this is invisible to them.
