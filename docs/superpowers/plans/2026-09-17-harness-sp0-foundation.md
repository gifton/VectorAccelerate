# Harness SP0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the benchmark-harness substrate (generic `VectorTestKit` library, `VectorAccelerateBench` executable) and close one measurement loop: `run` writes a schema-valid artifact with full provenance for two cases, `baseline capture` stores it, `compare` reports a status per case, and a deliberately slowed probe is caught.

**Architecture:** Two non-product SwiftPM targets. `VectorTestKit` is pure Swift (Foundation, Darwin, CryptoKit only) holding the seeded RNG, generators, digest, comparator, recall, statistics, artifact schema, host fingerprint, preflight, run store, warm-up policy, and compare logic — everything testable without a GPU. `VectorAccelerateBench` is a thin executable holding the CLI, the sampling loop, the Metal part of the fingerprint, the GPU probe, and the two M0 cases. Deletions of dead benchmark code land at the end as the 0.7.0 breaking change.

**Tech Stack:** Swift 6.2 tools, XCTest, Metal (executable only), IOKit (executable only), CryptoKit (kit, SHA-256), SwiftPM. No third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md` (authority). Adapter seam: `docs/superpowers/specs/2026-09-14-harness-interface-contract.md`. Program brief: `docs/superpowers/research/2026-09-14-observability-harness-research.md`.

## Global Constraints

- Platform: macOS 26.0+ / iOS 26.0+ / Metal 4 (Package.swift `platforms`, unchanged). Development and trusted runs: Apple M3 Max, macOS 26.5.2, Swift 6.3.3, Xcode 26.6.
- `swift-tools-version: 6.2`; library target has `StrictConcurrency` and `ExistentialAny` enabled; new targets enable the same two settings.
- `Sources/VectorTestKit/**` may import only `Foundation`, `Darwin`, `CryptoKit`. No `@testable`. (spec §4.1, enforced by `KitImportLintTests`)
- Neither new target is a declared product (spec §3). `swift run VectorAccelerateBench` works through the root package's implicit executable product.
- No new clocks: wall time is `ContinuousClock`; GPU command-buffer time is `gpuStartTime/gpuEndTime`. `CFAbsoluteTimeGetCurrent` and `CACurrentMediaTime` do not appear in new code. (brief P2)
- Version: this is **0.7.0**, breaking (removes `Sources/VectorAccelerate/Benchmarking/` public types, the `VectorAccelerateBenchmarks` product, `IndexAccelerationConfiguration.logDecisions`).
- Performance boundary rule: `max(0.025, 2 × baseline RSD)` on the median per timing kind; alerts in both directions; alerts never fail the run unless `--strict-perf`. (spec §5.6)
- Missing recall is `nil`, never `1.0`. Runs with `validationLayers == true`, `preflight.quiet == false`, or `gitDirty == true` (without `--allow-dirty`) are never captured as baselines. (spec §4.6, §4.11)
- Warm-up: window 8, CoV < 0.05, minimum 3, maximum 50; the count used is recorded per sample. (spec §5.2)
- Mode defaults: `smoke` 1×3, `quick` 3×20, `full` 5×100 (samples × iterations).
- Case ID grammar: `^[a-z0-9_]+(\.[a-z0-9_]+)+(\.[a-z0-9_]+=[a-z0-9_]+)*(\.[a-z0-9_]+)?$` — a family, one or more bare op segments, zero or more `param=value` segments, an optional bare variant; never contains seeds, digests, timestamps. (spec §5.4, amended in Task 17 because the spec's one-op form rejects `distance.l2.batch.n=10000.d=768`)
- Commits end with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Commit after every task. Do not commit unrelated working-tree changes from the parallel IVF session (`git add` only the paths each task names).
- Test gates: every task ends with `swift test --filter <suite>` green; Task 17 runs the full suite in debug and release. Current full-suite expectation before this plan: 1764 passed / 0 failed / 2 skipped.
- Naming deviation from the spec, applied consistently: the artifact's preflight struct is `PreflightRecord` and the probing helper is `PreflightProbe` (the spec uses `Preflight` for both; JSON key stays `preflight`).

---

## File Structure

**Created — `Sources/VectorTestKit/` (library target, no product)**
- `VectorTestKit.swift` — module marker: `public enum VectorTestKit { public static let schemaVersion = 1 }`.
- `SeededRNG.swift` — moved from `Tests/VectorAccelerateTests/Utilities/TestRNG.swift`, type renamed; deprecated `TestRNG` typealias.
- `VectorGenerator.swift` — moved from `Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift`, type renamed; `static let version = "1"`; deprecated `TestDataGenerator` typealias.
- `ContentDigest.swift` — `DigestPart`, `ContentDigest.sha256(_:)`.
- `Tolerance.swift` — `Tolerance`, `Verdict`, `Comparator`.
- `Recall.swift` — `RecallPolicy`, `Recall`, `RecallError`.
- `Stats.swift` — `Summary`, `Stats`, `blackHole`.
- `Schema.swift` — `BenchArtifact`, `Metadata`, `GPUInfo`, `PowerInfo`, `ThermalInfo`, `PreflightRecord`, `StartEnd`, `Filters`, `CaseRecord`, `CaseIdentity`, `CaseStatus`, `Timing`, `TimingKind`, `Series`, `Correctness`, `CorrectnessStatus`, `Resources`, `ArtifactValidationError`.
- `CaseID.swift` — `CaseID`, `CaseIDError`, `CaseRegistryCheck`.
- `WarmupPolicy.swift` — `WarmupPolicy`.
- `HostFingerprint.swift` — `HostFingerprint`, `PowerState`, `Subprocess`, `EnvironmentKey`.
- `PreflightProbe.swift` — `PreflightProbe`.
- `RunStore.swift` — `RunStore`, `RunStoreError`.
- `Compare.swift` — `CompareStatus`, `KindDelta`, `CompareResult`, `Compare`.

**Created — `Sources/VectorAccelerateBench/` (executable target, no declared product)**
- `main.swift` — argument parsing and command dispatch.
- `CLI/Arguments.swift` — `Arguments` (flags and values).
- `Runner/BenchCase.swift` — `GPUSample`, `IterationOutput`, `BenchCase`, `PreparedCase`, `BenchContext`, `RunOptions`, `BenchMode`.
- `Runner/Runner.swift` — `Runner.run(case:context:options:)` → `CaseRecord`.
- `Fingerprint/MetalFingerprint.swift` — `MetalFingerprint.gather()` → `GPUInfo?`.
- `Probe/GPUProbe.swift` — `GPUProbe` (copy kernel, legacy command queue, `copyGBps()`, per-dispatch timing).
- `Cases/Registry.swift` — `Registry.allCases`.
- `Cases/Probe/ProbeCases.swift` — `probe.gpu_copy.bytes=134217728`.
- `Cases/Distance/DistanceCases.swift` — `distance.l2.batch.n=10000.d=768`.
- `Commands/ListCommand.swift`, `Commands/RunCommand.swift`, `Commands/BaselineCommand.swift`, `Commands/CompareCommand.swift`, `Commands/ValidateCommand.swift`.

**Created — `Tests/VectorAccelerateTests/Harness/`**
- `RepoPaths.swift` (helper), `KitImportLintTests.swift`, `OrphanPublicTypeTests.swift`, `SeededRNGGoldenTests.swift` (generated), `VectorGeneratorGoldenTests.swift` (generated), `ContentDigestGoldenTests.swift`, `ComparatorTests.swift`, `RecallTests.swift`, `StatsTests.swift`, `ArtifactSchemaTests.swift`, `CaseIDTests.swift`, `WarmupPolicyTests.swift`, `FingerprintTests.swift`, `RunStoreTests.swift`, `CompareTests.swift`.
- `Tests/VectorAccelerateTests/Utilities/RequireMetal.swift` — `requireMetalDevice()`.

**Modified**
- `Package.swift` — targets, test dependency, version-history header entry.
- `.gitignore` (`.bench/`), `.swiftlint.yml`, `README.md`, `CHANGELOG.md`.
- 15 test files gain `import VectorTestKit` and use the renamed types (listed in Task 3 and Task 4).
- `Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift` — harness-only test removed, private RNG replaced.
- `Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift` — private RNG replaced.
- `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift` — `logDecisions` removed.

**Deleted**
- `Sources/VectorAccelerateBenchmarks/` (5 files) and its product/target (Task 0).
- `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`, `IndexBenchmarkHarness.swift`; `Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift`; `Tests/VectorAccelerateTests/PerformanceBenchmarks.swift`; `Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift` (Task 16).
- `Tests/VectorAccelerateTests/Utilities/TestRNG.swift`, `TestDataGenerator.swift` (moved, Tasks 3–4).

**Spec deviation recorded (Task 16 updates the spec):** `IndexBenchmarkRecallValidationTests.swift` does not depend on the deleted harness except for one test (`testBenchmarkHarnessRealisticIVF`), so the file is kept with that test removed instead of being ported wholesale; its flat-recall-exactly-1.0 and IVF-recall->0.95 assertions are the retained coverage the spec wanted in `IndexRecallTests`. `IndexBenchmarkHarnessTests.swift` is deleted as planned.

---

### Task 0: Package targets, directories, lint and ignore rules

**Files:**
- Modify: `Package.swift` (targets array; version-history header)
- Create: `Sources/VectorTestKit/VectorTestKit.swift`
- Create: `Sources/VectorAccelerateBench/main.swift`
- Delete: `Sources/VectorAccelerateBenchmarks/` (all five files)
- Modify: `.gitignore`, `.swiftlint.yml`

**Interfaces:**
- Produces: targets `VectorTestKit` (library) and `VectorAccelerateBench` (executable); test target depends on `VectorTestKit`.

- [ ] **Step 1: Replace the targets array in `Package.swift`**

Replace everything from `targets: [` to the closing `]` of the package with:

```swift
    targets: [
        // MARK: - Core GPU Acceleration
        // Includes GPU-first vector index (AcceleratedVectorIndex) and all Metal kernels
        .target(
            name: "VectorAccelerate",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
            ],
            exclude: [
                "metal-compiler-plugin.json"  // Build-time config for MetalCompilerPlugin, not a runtime resource
            ],
            resources: [
                .process("Metal/Shaders")  // Metal shader files
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ],
            plugins: [
                // Compiles Metal shaders with debug symbols for Xcode Metal Debugger
                // Creates debug.metallib alongside SPM's default.metallib
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),

        // MARK: - Test kit (generic; not a product; Foundation/Darwin/CryptoKit only)
        .target(
            name: "VectorTestKit",
            dependencies: [],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),

        // MARK: - Benchmark harness executable (not a declared product; `swift run VectorAccelerateBench`)
        .executableTarget(
            name: "VectorAccelerateBench",
            dependencies: [
                "VectorAccelerate",
                "VectorTestKit",
                .product(name: "VectorCore", package: "VectorCore")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("IOKit")
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "VectorAccelerateTests",
            dependencies: [
                "VectorAccelerate",
                "VectorTestKit",
                .product(name: "VectorCore", package: "VectorCore")
            ]
        )
    ]
```

Also delete the `.executable(name: "VectorAccelerateBenchmarks", targets: ["VectorAccelerateBenchmarks"]),` line from `products:`.

- [ ] **Step 2: Add the 0.7.0 entry to the version-history header in `Package.swift`**

Insert directly above the line `// 0.6.0 — Zero-copy SoA scoring + normalization parity + the promised removal:`:

```swift
// 0.7.0 — Benchmark harness foundation (SP0) + dead-code removal (BREAKING):
//   - Added: VectorTestKit (generic seeded RNG, generators, digest, comparator, recall, stats,
//     artifact schema v1, host fingerprint, preflight, run store, compare) — not a product
//   - Added: VectorAccelerateBench executable (run / baseline capture / compare / validate-artifact)
//   - Removed: Sources/VectorAccelerate/Benchmarking (BenchmarkFramework, IndexBenchmarkHarness and
//     all their public types), the VectorAccelerateBenchmarks product, PerformanceMonitor,
//     IndexAccelerationConfiguration.logDecisions
//   - Changed: README performance claims retired pending measured runs under .bench/runs/
//   See docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md
```

- [ ] **Step 3: Create the kit marker and the executable placeholder**

`Sources/VectorTestKit/VectorTestKit.swift`:

```swift
import Foundation

/// Module marker. The artifact schema version lives on `BenchArtifact.schemaVersion`; this
/// constant exists so the kit compiles before any other file is added.
public enum VectorTestKit {
    public static let schemaVersion: Int = 1
}
```

`Sources/VectorAccelerateBench/main.swift`:

```swift
import Foundation

print("VectorAccelerateBench: no commands wired yet (SP0 Task 0 placeholder)")
exit(2)
```

- [ ] **Step 4: Delete the old benchmark executable sources**

```bash
git rm -r -q Sources/VectorAccelerateBenchmarks
```

- [ ] **Step 5: Ignore `.bench/` and update SwiftLint scope**

Append to `.gitignore`:

```
# Benchmark harness runs and baselines (SP0); history strategy is SP5
.bench/
```

In `.swiftlint.yml`: remove the line `  - Sources/VectorAccelerateBenchmarks  # Benchmarks can use print()` from `excluded:`, and change the custom rule's `excluded: ".*Tests.*"` to:

```yaml
    excluded: ".*(Tests|VectorAccelerateBench)/.*"
```

- [ ] **Step 6: Build**

Run: `swift build 2>&1 | tail -5`
Expected: `Build complete!` with no errors (warnings allowed).

- [ ] **Step 7: Commit**

```bash
git add Package.swift Sources/VectorTestKit/VectorTestKit.swift Sources/VectorAccelerateBench/main.swift .gitignore .swiftlint.yml
git commit -q -F - <<'MSG'
build: add VectorTestKit and VectorAccelerateBench targets; drop VectorAccelerateBenchmarks product

SP0 Task 0. Two non-product targets, test target depends on the kit,
.bench/ ignored, lint scope updated. Old benchmark executable removed
(its crossover sweep is rebuilt as a harness mode in SP4).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

---

### Task 1: Repo-path helper and kit import-lint test

**Files:**
- Create: `Tests/VectorAccelerateTests/Harness/RepoPaths.swift`
- Create: `Tests/VectorAccelerateTests/Harness/KitImportLintTests.swift`

**Interfaces:**
- Produces: `RepoPaths.root: URL` (package root, derived from `#filePath`); `RepoPaths.swiftFiles(under:) -> [URL]`.

- [ ] **Step 1: Write the helper**

```swift
import Foundation

/// Locates the package root from this file's path so mechanical tests can scan sources.
/// Layout assumed: <root>/Tests/VectorAccelerateTests/Harness/RepoPaths.swift
enum RepoPaths {
    static let root: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url = url.deletingLastPathComponent() }  // RepoPaths.swift, Harness, VectorAccelerateTests, Tests
        return url
    }()

    static func swiftFiles(under relative: String) -> [URL] {
        let base = root.appendingPathComponent(relative)
        guard let e = FileManager.default.enumerator(at: base, includingPropertiesForKeys: nil) else { return [] }
        return e.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }.sorted { $0.path < $1.path }
    }
}
```

- [ ] **Step 2: Write the lint test**

```swift
import XCTest

/// Genericism rule (spec §4.1): the kit imports only Foundation, Darwin, CryptoKit and never uses @testable.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    func testKitImportsAreWithinAllowlist() throws {
        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        let importLine = try NSRegularExpression(pattern: #"^\s*(@testable\s+)?import\s+([A-Za-z0-9_]+)"#, options: [.anchorsMatchLines])
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            let range = NSRange(text.startIndex..., in: text)
            for m in importLine.matches(in: text, range: range) {
                let testable = m.range(at: 1).location != NSNotFound
                let module = String(text[Range(m.range(at: 2), in: text)!])
                if testable { violations.append("\(file.lastPathComponent): @testable import is forbidden") }
                if !Self.allowed.contains(module) { violations.append("\(file.lastPathComponent): import \(module) is outside the allowlist \(Self.allowed.sorted())") }
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }
}
```

- [ ] **Step 3: Run it; expect PASS (only the marker file exists)**

Run: `swift test --filter KitImportLintTests 2>&1 | tail -3`
Expected: `Executed 1 test, with 0 failures`

- [ ] **Step 4: Prove the guard detects a violation, then remove the probe**

```bash
printf 'import XCTest\npublic enum _LintProbe {}\n' > Sources/VectorTestKit/_LintProbe.swift
swift test --filter KitImportLintTests 2>&1 | grep -c "outside the allowlist"   # expected: 1
rm Sources/VectorTestKit/_LintProbe.swift
swift test --filter KitImportLintTests 2>&1 | tail -1                           # expected: 0 failures
```

- [ ] **Step 5: Commit**

```bash
git add Tests/VectorAccelerateTests/Harness/RepoPaths.swift Tests/VectorAccelerateTests/Harness/KitImportLintTests.swift
git commit -q -m "test: enforce VectorTestKit import allowlist

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `ContentDigest`

**Files:**
- Create: `Sources/VectorTestKit/ContentDigest.swift`
- Test: `Tests/VectorAccelerateTests/Harness/ContentDigestGoldenTests.swift`

**Interfaces:**
- Produces: `public enum DigestPart { case floats([Float]), doubles([Double]), int32s([Int32]), uint32s([UInt32]), int64s([Int64]), uint64s([UInt64]), int(Int), double(Double), string(String) }`; `public enum ContentDigest { public static func sha256(_ parts: [DigestPart]) -> String }` (lowercase hex, 64 chars).

- [ ] **Step 1: Write the failing golden test**

```swift
import XCTest
import VectorTestKit

/// Pins the canonical byte encoding (spec §4.4). The golden was computed independently in Python:
/// LE u64 count + LE element bytes per array; LE Int64 for `.int`; Double bit pattern for `.double`;
/// LE u64 byte-length + UTF-8 for `.string`; parts hashed in order.
final class ContentDigestGoldenTests: XCTestCase {
    func testGoldenDigest() {
        let parts: [DigestPart] = [
            .floats([1.0, 2.5, -0.5]),
            .uint32s([7, 8]),
            .int(42),
            .double(0.25),
            .string("vector"),
        ]
        XCTAssertEqual(ContentDigest.sha256(parts), "82ea485579eebefdcbe8edff10d654835d0639807b801a2bf527b8b40c9070cb")
    }

    func testOrderMatters() {
        XCTAssertNotEqual(ContentDigest.sha256([.int(1), .int(2)]), ContentDigest.sha256([.int(2), .int(1)]))
    }

    func testEmptyArrayIsNotEmptyDigest() {
        XCTAssertNotEqual(ContentDigest.sha256([.floats([])]), ContentDigest.sha256([]))
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter ContentDigestGoldenTests 2>&1 | grep -E "error:|Executed" | head -3`
Expected: compile error `cannot find 'DigestPart' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation
import CryptoKit

/// One component of a fixture's identity. Arrays are prefixed by their element count.
public enum DigestPart: Sendable {
    case floats([Float])
    case doubles([Double])
    case int32s([Int32])
    case uint32s([UInt32])
    case int64s([Int64])
    case uint64s([UInt64])
    case int(Int)
    case double(Double)
    case string(String)
}

/// SHA-256 over a canonical little-endian encoding (spec §4.4). Rendered as lowercase hex.
public enum ContentDigest {
    public static func sha256(_ parts: [DigestPart]) -> String {
        var hasher = SHA256()
        for part in parts { append(part, to: &hasher) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func append(_ part: DigestPart, to hasher: inout SHA256) {
        switch part {
        case .floats(let a):  appendCount(a.count, &hasher); for x in a { appendLE(x.bitPattern, &hasher) }
        case .doubles(let a): appendCount(a.count, &hasher); for x in a { appendLE(x.bitPattern, &hasher) }
        case .int32s(let a):  appendCount(a.count, &hasher); for x in a { appendLE(UInt32(bitPattern: x), &hasher) }
        case .uint32s(let a): appendCount(a.count, &hasher); for x in a { appendLE(x, &hasher) }
        case .int64s(let a):  appendCount(a.count, &hasher); for x in a { appendLE(UInt64(bitPattern: x), &hasher) }
        case .uint64s(let a): appendCount(a.count, &hasher); for x in a { appendLE(x, &hasher) }
        case .int(let v):     appendLE(UInt64(bitPattern: Int64(v)), &hasher)
        case .double(let v):  appendLE(v.bitPattern, &hasher)
        case .string(let s):
            let bytes = Array(s.utf8)
            appendCount(bytes.count, &hasher)
            hasher.update(data: Data(bytes))
        }
    }

    private static func appendCount(_ n: Int, _ hasher: inout SHA256) {
        appendLE(UInt64(n), &hasher)
    }

    private static func appendLE<T: FixedWidthInteger>(_ value: T, _ hasher: inout SHA256) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { hasher.update(bufferPointer: $0) }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter "ContentDigestGoldenTests|KitImportLintTests" 2>&1 | tail -1`
Expected: `Executed 4 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/ContentDigest.swift Tests/VectorAccelerateTests/Harness/ContentDigestGoldenTests.swift
git commit -q -m "feat(kit): ContentDigest with canonical LE encoding and golden

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Promote `TestRNG` to `SeededRNG` with a frozen output stream

**Files:**
- Move: `Tests/VectorAccelerateTests/Utilities/TestRNG.swift` → `Sources/VectorTestKit/SeededRNG.swift`
- Create (generated): `Tests/VectorAccelerateTests/Harness/SeededRNGGoldenTests.swift`
- Modify (add `import VectorTestKit`, rename uses): `Tests/VectorAccelerateTests/BatchDistanceEngineTests.swift`, `Fixtures/IVF/IVFBenchmarkFixture.swift`, `Hardening/DifferentialKernelVsCPUTests.swift`, `Hardening/LatentReductionGeometryTests.swift`, `Hardening/RoutingProvenanceTests.swift`, `Hardening/SpecializedKernelStrideGuardTests.swift`, `Hardening/TreeReductionDispatchRobustnessTests.swift`, `IVFQualityAssessmentTests.swift`, `IVFValidationTests.swift`, `LogSumExpKernelTests.swift`, `NormalizationParityTests.swift`, `OperationsDispatchTests.swift`, `SoAFusedTopKTests.swift`, `TopKSelectionAdoptionTests.swift`, `Utilities/TestDataGenerator.swift`

**Interfaces:**
- Produces: `public struct SeededRNG: Sendable, RandomNumberGenerator` with `init(seed: UInt64, stream: UInt64 = 0, warmup: Int = 10)`, `next() -> UInt64`, `nextFloat01() -> Float`, `nextFloat(in:) -> Float`, `nextDouble() -> Double`, `nextGaussian(mean:stdDev:) -> Float`, `nextInt(bound:)`, `nextInt(in: Range<Int>)`, `nextInt(in: ClosedRange<Int>)`, `nextBool(probability:)`, `shuffle(_:)`, `shuffled(_:)`, `sample(_:k:)`; `public typealias TestRNG = SeededRNG` (deprecated).

- [ ] **Step 1: Generate the golden test from the PRE-MOVE generator (before touching the file)**

This compiles the current test utility as a tiny program and writes the whole golden test file. The literal arrays it emits are the frozen contract; never regenerate them.

```bash
mkdir -p /tmp/rng-gold && cat > /tmp/rng-gold/main.swift <<'GEN'
import Foundation
func lit<T>(_ a: [T]) -> String { "[" + a.map { "\($0)" }.joined(separator: ", ") + "]" }
var out = """
import XCTest
import VectorTestKit

/// FROZEN output streams of SeededRNG (spec §4.2). Generated once from the pre-promotion TestRNG on 2026-09-17.
/// A failure here means the sequence changed and every fixture digest in the suite is invalid. Do not regenerate.
final class SeededRNGGoldenTests: XCTestCase {
    private func first64<T>(_ f: (inout SeededRNG) -> T, seed: UInt64, stream: UInt64) -> [T] {
        var rng = SeededRNG(seed: seed, stream: stream)
        return (0..<64).map { _ in f(&rng) }
    }

"""
for seed: UInt64 in [0, 1, 42] {
    for stream: UInt64 in [0, 1] {
        var r1 = TestRNG(seed: seed, stream: stream); let a = (0..<64).map { _ in r1.next() }
        var r2 = TestRNG(seed: seed, stream: stream); let b = (0..<64).map { _ in r2.nextFloat01() }
        var r3 = TestRNG(seed: seed, stream: stream); let c = (0..<64).map { _ in r3.nextGaussian() }
        out += """
    func testNext_seed\(seed)_stream\(stream)() {
        XCTAssertEqual(first64({ $0.next() }, seed: \(seed), stream: \(stream)), \(lit(a)) as [UInt64])
    }
    func testFloat01_seed\(seed)_stream\(stream)() {
        XCTAssertEqual(first64({ $0.nextFloat01() }, seed: \(seed), stream: \(stream)), \(lit(b.map { $0.bitPattern })).map { Float(bitPattern: $0) })
    }
    func testGaussian_seed\(seed)_stream\(stream)() {
        XCTAssertEqual(first64({ $0.nextGaussian() }, seed: \(seed), stream: \(stream)), \(lit(c.map { $0.bitPattern })).map { Float(bitPattern: $0) })
    }

"""
    }
}
out += "}\n"
try! out.write(toFile: CommandLine.arguments[1], atomically: true, encoding: .utf8)
GEN
swiftc -O -o /tmp/rng-gold/gen Tests/VectorAccelerateTests/Utilities/TestRNG.swift /tmp/rng-gold/main.swift \
  && /tmp/rng-gold/gen Tests/VectorAccelerateTests/Harness/SeededRNGGoldenTests.swift \
  && grep -c "func test" Tests/VectorAccelerateTests/Harness/SeededRNGGoldenTests.swift   # expected: 18
```

Floats are pinned by bit pattern so the file round-trips exactly.

- [ ] **Step 2: Run to verify it fails (the kit has no `SeededRNG` yet)**

Run: `swift test --filter SeededRNGGoldenTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'SeededRNG' in scope`.

- [ ] **Step 3: Move and rename**

```bash
git mv Tests/VectorAccelerateTests/Utilities/TestRNG.swift Sources/VectorTestKit/SeededRNG.swift
sed -i '' -e 's/\bTestRNG\b/SeededRNG/g' -e 's|//  TestRNG.swift|//  SeededRNG.swift|' Sources/VectorTestKit/SeededRNG.swift
cat >> Sources/VectorTestKit/SeededRNG.swift <<'SWIFT'

/// Pre-0.7.0 name. Removed in 0.8.0.
@available(*, deprecated, renamed: "SeededRNG")
public typealias TestRNG = SeededRNG
SWIFT
grep -n "public struct SeededRNG\|extension SeededRNG: RandomNumberGenerator\|typealias TestRNG" Sources/VectorTestKit/SeededRNG.swift
```

Expected: three matching lines.

- [ ] **Step 4: Point the test target at the kit**

For each file listed under **Modify** above, add `import VectorTestKit` on the line after its last `import` statement and rename uses:

```bash
FILES="Tests/VectorAccelerateTests/BatchDistanceEngineTests.swift Tests/VectorAccelerateTests/Fixtures/IVF/IVFBenchmarkFixture.swift Tests/VectorAccelerateTests/Hardening/DifferentialKernelVsCPUTests.swift Tests/VectorAccelerateTests/Hardening/LatentReductionGeometryTests.swift Tests/VectorAccelerateTests/Hardening/RoutingProvenanceTests.swift Tests/VectorAccelerateTests/Hardening/SpecializedKernelStrideGuardTests.swift Tests/VectorAccelerateTests/Hardening/TreeReductionDispatchRobustnessTests.swift Tests/VectorAccelerateTests/IVFQualityAssessmentTests.swift Tests/VectorAccelerateTests/IVFValidationTests.swift Tests/VectorAccelerateTests/LogSumExpKernelTests.swift Tests/VectorAccelerateTests/NormalizationParityTests.swift Tests/VectorAccelerateTests/OperationsDispatchTests.swift Tests/VectorAccelerateTests/SoAFusedTopKTests.swift Tests/VectorAccelerateTests/TopKSelectionAdoptionTests.swift Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift"
for f in $FILES; do
  python3 - "$f" <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
if "import VectorTestKit" not in s:
    lines=s.split("\n"); idx=max(i for i,l in enumerate(lines) if l.startswith("import ") or l.startswith("@testable import ") or l.startswith("@preconcurrency import "))
    lines.insert(idx+1,"import VectorTestKit"); s="\n".join(lines)
s=re.sub(r"\bTestRNG\b","SeededRNG",s)
open(p,"w").write(s)
PY
done
grep -L "import VectorTestKit" $FILES   # expected: no output
```

- [ ] **Step 5: Build and run the goldens plus the touched suites**

Run: `swift test --filter "SeededRNGGoldenTests|DifferentialKernelVsCPUTests|RoutingProvenanceTests|KitImportLintTests" 2>&1 | tail -1`
Expected: `0 failures` (test count = 18 goldens + the two suites' existing counts + 1).

- [ ] **Step 6: Commit**

```bash
git add -A Sources/VectorTestKit/SeededRNG.swift Tests/VectorAccelerateTests
git commit -q -m "refactor(kit): promote TestRNG to SeededRNG with frozen golden streams

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Promote `TestDataGenerator` to `VectorGenerator` with frozen digests

**Files:**
- Move: `Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift` → `Sources/VectorTestKit/VectorGenerator.swift`
- Create (generated): `Tests/VectorAccelerateTests/Harness/VectorGeneratorGoldenTests.swift`
- Modify (rename uses): `Tests/VectorAccelerateTests/Fixtures/IVF/IVFBenchmarkFixture.swift`, `IVFQualityAssessmentTests.swift`, `OperationsDispatchTests.swift`, `IVFValidationTests.swift`

**Interfaces:**
- Consumes: `SeededRNG` (Task 3), `ContentDigest` (Task 2).
- Produces: `public struct VectorGenerator` with the same members as `TestDataGenerator` plus `public static let version: String = "1"`; `public typealias TestDataGenerator = VectorGenerator` (deprecated); `public struct DatasetStatistics` unchanged.

- [ ] **Step 1: Generate the golden test from the PRE-MOVE generator**

```bash
mkdir -p /tmp/gen-gold && cat > /tmp/gen-gold/main.swift <<'GEN'
import Foundation
func d(_ v: [[Float]]) -> String { ContentDigest.sha256([.int(v.count), .int(v.first?.count ?? 0), .floats(v.flatMap { $0 })]) }
var g = TestDataGenerator(seed: 42, stream: 0)
let uniform = d(g.uniformVectors(count: 16, dimension: 8))
let ranged  = d(g.uniformVectors(count: 16, dimension: 8, range: -3...3))
let gauss   = d(g.gaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8))
let sep     = d(g.separatedGaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8).data)
let unit    = d(g.unitVectors(count: 16, dimension: 8))
let cunit   = d(g.clusteredUnitVectors(numClusters: 4, pointsPerCluster: 8, dimension: 8))
let sparse  = d(g.sparseVectors(count: 16, dimension: 8))
let skew    = d(g.skewedClusters(numClusters: 4, totalPoints: 32, dimension: 8))
let base    = g.uniformVectors(count: 8, dimension: 4)
let pert    = d(g.perturbedQueries(from: base, count: 4))
let cq      = d(g.clusterQueries(centers: base, countPerCenter: 1))
let rq      = d(g.randomQueries(count: 4, dimension: 4))
let out = """
import XCTest
import VectorTestKit

/// FROZEN per-method digests of VectorGenerator at seed 42 / stream 0 (spec §4.3). Generated once from the
/// pre-promotion TestDataGenerator on 2026-09-17. If a method's output must change, bump VectorGenerator.version
/// and update the single affected digest in the same commit with a ledger entry. Do not regenerate wholesale.
final class VectorGeneratorGoldenTests: XCTestCase {
    private func d(_ v: [[Float]]) -> String { ContentDigest.sha256([.int(v.count), .int(v.first?.count ?? 0), .floats(v.flatMap { $0 })]) }

    func testVersionIsOne() { XCTAssertEqual(VectorGenerator.version, "1") }

    func testFrozenDigests() {
        var g = VectorGenerator(seed: 42, stream: 0)
        XCTAssertEqual(d(g.uniformVectors(count: 16, dimension: 8)), "\(uniform)", "uniformVectors")
        XCTAssertEqual(d(g.uniformVectors(count: 16, dimension: 8, range: -3...3)), "\(ranged)", "uniformVectors(range:)")
        XCTAssertEqual(d(g.gaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8)), "\(gauss)", "gaussianClusters")
        XCTAssertEqual(d(g.separatedGaussianClusters(numClusters: 4, pointsPerCluster: 8, dimension: 8).data), "\(sep)", "separatedGaussianClusters")
        XCTAssertEqual(d(g.unitVectors(count: 16, dimension: 8)), "\(unit)", "unitVectors")
        XCTAssertEqual(d(g.clusteredUnitVectors(numClusters: 4, pointsPerCluster: 8, dimension: 8)), "\(cunit)", "clusteredUnitVectors")
        XCTAssertEqual(d(g.sparseVectors(count: 16, dimension: 8)), "\(sparse)", "sparseVectors")
        XCTAssertEqual(d(g.skewedClusters(numClusters: 4, totalPoints: 32, dimension: 8)), "\(skew)", "skewedClusters")
        let base = g.uniformVectors(count: 8, dimension: 4)
        XCTAssertEqual(d(g.perturbedQueries(from: base, count: 4)), "\(pert)", "perturbedQueries")
        XCTAssertEqual(d(g.clusterQueries(centers: base, countPerCenter: 1)), "\(cq)", "clusterQueries")
        XCTAssertEqual(d(g.randomQueries(count: 4, dimension: 4)), "\(rq)", "randomQueries")
    }
}

"""
try! out.write(toFile: CommandLine.arguments[1], atomically: true, encoding: .utf8)
GEN
swiftc -O -o /tmp/gen-gold/gen Sources/VectorTestKit/SeededRNG.swift Sources/VectorTestKit/ContentDigest.swift Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift /tmp/gen-gold/main.swift \
  && /tmp/gen-gold/gen Tests/VectorAccelerateTests/Harness/VectorGeneratorGoldenTests.swift \
  && grep -c '"[0-9a-f]\{64\}"' Tests/VectorAccelerateTests/Harness/VectorGeneratorGoldenTests.swift   # expected: 11
```

If a method's parameter labels differ from the calls above, the compile step fails naming the label; correct the call in the script to the label that exists in `TestDataGenerator.swift` and re-run (the golden test uses the same calls, so both stay consistent).

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter VectorGeneratorGoldenTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'VectorGenerator' in scope`.

- [ ] **Step 3: Move, rename, add version and typealias**

```bash
git mv Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift Sources/VectorTestKit/VectorGenerator.swift
sed -i '' -e 's/\bTestDataGenerator\b/VectorGenerator/g' -e 's|//  TestDataGenerator.swift|//  VectorGenerator.swift|' Sources/VectorTestKit/VectorGenerator.swift
python3 - <<'PY'
p="Sources/VectorTestKit/VectorGenerator.swift"; s=open(p).read()
old="public struct VectorGenerator {\n    /// Internal RNG\n    var rng: SeededRNG\n"
assert s.count(old)==1, "anchor not found; inspect the struct header"
s=s.replace(old, "public struct VectorGenerator {\n    /// Bumped whenever any method's output stream changes; part of every case identity (spec §4.3).\n    public static let version: String = \"1\"\n\n    /// Internal RNG\n    var rng: SeededRNG\n")
s+="\n/// Pre-0.7.0 name. Removed in 0.8.0.\n@available(*, deprecated, renamed: \"VectorGenerator\")\npublic typealias TestDataGenerator = VectorGenerator\n"
open(p,"w").write(s)
PY
grep -n "public static let version\|typealias TestDataGenerator" Sources/VectorTestKit/VectorGenerator.swift
```

- [ ] **Step 4: Rename uses in tests**

```bash
sed -i '' 's/\bTestDataGenerator\b/VectorGenerator/g' Tests/VectorAccelerateTests/Fixtures/IVF/IVFBenchmarkFixture.swift Tests/VectorAccelerateTests/IVFQualityAssessmentTests.swift Tests/VectorAccelerateTests/OperationsDispatchTests.swift Tests/VectorAccelerateTests/IVFValidationTests.swift
grep -rn "TestDataGenerator" Tests/VectorAccelerateTests | grep -v "typealias"   # expected: no output
```

- [ ] **Step 5: Run**

Run: `swift test --filter "VectorGeneratorGoldenTests|SeededRNGGoldenTests|IVFQualityAssessmentTests|OperationsDispatchTests|KitImportLintTests" 2>&1 | tail -1`
Expected: `0 failures`.

- [ ] **Step 6: Commit**

```bash
git add -A Sources/VectorTestKit/VectorGenerator.swift Tests/VectorAccelerateTests
git commit -q -m "refactor(kit): promote TestDataGenerator to VectorGenerator with frozen digests and version

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `Tolerance`, `Verdict`, `Comparator`

**Files:**
- Create: `Sources/VectorTestKit/Tolerance.swift`
- Test: `Tests/VectorAccelerateTests/Harness/ComparatorTests.swift`

**Interfaces:**
- Produces: `public struct Tolerance: Codable, Sendable, Equatable { public var absolute: Double; public var relative: Double; public var ulps: UInt64?; public init(absolute: Double = 0, relative: Double = 0, ulps: UInt64? = nil); public static func forSum(terms: Int, reduction: Reduction, magnitude: Double) -> Tolerance; public enum Reduction: String, Codable, Sendable { case sequential, tree } }`; `public enum Verdict: Equatable, Sendable { case equal, withinTolerance, classMismatch, outOfTolerance(absError: Double, relError: Double) }`; `public enum Comparator { public static func compare(_ a: Double, _ b: Double, tolerance: Tolerance) -> Verdict; public static func compare(_ a: Float, _ b: Float, tolerance: Tolerance) -> Verdict; public static func ulpDistance(_ a: Double, _ b: Double) -> UInt64; public static func ulpDistance(_ a: Float, _ b: Float) -> UInt64 }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

/// Tiers (spec §4.5): both NaN → equal; bitwise equal → equal; class mismatch → hard fail;
/// abs/rel envelope; optional ULP tier. The ten adversarial value classes come from DifferentialKernelVsCPUTests.
final class ComparatorTests: XCTestCase {
    private let classes: [(String, Double)] = [
        ("normal", 1.234), ("zero", 0.0), ("negZero", -0.0), ("huge", 1e19), ("tiny", 1e-20),
        ("subnormal", 1e-310), ("mixed", 12345.678e-9), ("nan", .nan), ("posInf", .infinity), ("negInf", -.infinity),
    ]

    func testBothNaNAreEqual() {
        XCTAssertEqual(Comparator.compare(Double.nan, Double.nan, tolerance: Tolerance()), .equal)
    }

    func testBitwiseEqualIsEqualIncludingSignedZeroAndInfinity() {
        XCTAssertEqual(Comparator.compare(0.0, 0.0, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(-0.0, -0.0, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(Double.infinity, Double.infinity, tolerance: Tolerance()), .equal)
    }

    func testSignedZeroWithinToleranceNotBitwiseEqual() {
        // +0 and -0 differ bitwise but |a-b| = 0 ≤ any envelope.
        XCTAssertEqual(Comparator.compare(0.0, -0.0, tolerance: Tolerance()), .withinTolerance)
    }

    func testClassMismatchIsHardFail() {
        let t = Tolerance(absolute: 1e300, relative: 1e300)
        XCTAssertEqual(Comparator.compare(Double.nan, 1.0, tolerance: t), .classMismatch)
        XCTAssertEqual(Comparator.compare(1.0, Double.infinity, tolerance: t), .classMismatch)
        XCTAssertEqual(Comparator.compare(Double.infinity, -Double.infinity, tolerance: t), .classMismatch)
    }

    func testAbsRelEnvelope() {
        XCTAssertEqual(Comparator.compare(100.0, 100.5, tolerance: Tolerance(absolute: 1.0)), .withinTolerance)
        XCTAssertEqual(Comparator.compare(100.0, 100.5, tolerance: Tolerance(relative: 0.01)), .withinTolerance)
        if case .outOfTolerance(let abs, let rel) = Comparator.compare(100.0, 102.0, tolerance: Tolerance(absolute: 1.0, relative: 0.01)) {
            XCTAssertEqual(abs, 2.0, accuracy: 1e-12); XCTAssertEqual(rel, 2.0 / 102.0, accuracy: 1e-12)
        } else { XCTFail("expected outOfTolerance") }
    }

    func testULPTier() {
        XCTAssertEqual(Comparator.ulpDistance(1.0, 1.0.nextUp), 1)
        XCTAssertEqual(Comparator.ulpDistance(Float(1.0), Float(1.0).nextUp), 1)
        XCTAssertEqual(Comparator.ulpDistance(-1.0, 1.0) > 1_000_000, true)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp, tolerance: Tolerance(ulps: 1)), .withinTolerance)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp.nextUp, tolerance: Tolerance(ulps: 1)), .outOfTolerance(absError: 1.0.nextUp.nextUp - 1.0, relError: (1.0.nextUp.nextUp - 1.0) / 1.0.nextUp.nextUp))
    }

    func testAdversarialClassesPairwiseNeverCrash() {
        let t = Tolerance(absolute: 1e-5, relative: 2e-4)
        for (na, a) in classes { for (nb, b) in classes {
            let v = Comparator.compare(a, b, tolerance: t)
            let sameClass = (a.isNaN && b.isNaN) || (a.isInfinite && b.isInfinite && a.sign == b.sign) || (a.isFinite && b.isFinite)
            if !sameClass { XCTAssertEqual(v, .classMismatch, "\(na) vs \(nb)") }
            if a.isNaN && b.isNaN { XCTAssertEqual(v, .equal, "\(na) vs \(nb)") }
        } }
    }

    func testForSumSequentialAndTree() {
        let eps = 5.960464477539063e-08  // 2^-24
        let seq = Tolerance.forSum(terms: 768, reduction: .sequential, magnitude: 3072)
        XCTAssertEqual(seq.absolute, 2 * eps * 768 * 3072, accuracy: 1e-15)
        XCTAssertEqual(seq.relative, 0)
        let tree = Tolerance.forSum(terms: 768, reduction: .tree, magnitude: 3072)
        XCTAssertEqual(tree.absolute, 2 * eps * 10 * 3072, accuracy: 1e-15)   // ceil(log2 768) = 10
        let one = Tolerance.forSum(terms: 1, reduction: .tree, magnitude: 1)
        XCTAssertEqual(one.absolute, 2 * eps * 1 * 1, accuracy: 1e-20)           // depth floor of 1
    }

    func testToleranceRoundTrips() throws {
        let t = Tolerance(absolute: 1e-6, relative: 1e-4, ulps: 4)
        let data = try JSONEncoder().encode(t)
        XCTAssertEqual(try JSONDecoder().decode(Tolerance.self, from: data), t)
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter ComparatorTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'Comparator' in scope` (or `Tolerance`).

- [ ] **Step 3: Implement**

```swift
import Foundation

/// Numerical tolerance (spec §4.5). `absolute`/`relative` form the envelope
/// `|a−b| ≤ max(absolute, relative · max(|a|,|b|))`; `ulps` is an optional fifth tier.
public struct Tolerance: Codable, Sendable, Equatable {
    public var absolute: Double
    public var relative: Double
    public var ulps: UInt64?

    public init(absolute: Double = 0, relative: Double = 0, ulps: UInt64? = nil) {
        self.absolute = absolute
        self.relative = relative
        self.ulps = ulps
    }

    public enum Reduction: String, Codable, Sendable { case sequential, tree }

    /// Explicit, unverified-per-kernel forward-error contract for a Float32 sum of `terms` values whose
    /// absolute values sum to at most `magnitude`: sequential ≈ (D−1)·ε·Σ|xᵢ|, balanced tree ≈ ⌈log₂D⌉·ε·Σ|xᵢ|,
    /// with ε = 2⁻²⁴ and a safety factor of 2. Tested by `ComparatorTests.testForSumSequentialAndTree`.
    public static func forSum(terms: Int, reduction: Reduction, magnitude: Double) -> Tolerance {
        let eps = 5.960464477539063e-08
        let depth: Double
        switch reduction {
        case .sequential: depth = Double(max(terms, 1))
        case .tree: depth = max(1, (log2(Double(max(terms, 1)))).rounded(.up))
        }
        return Tolerance(absolute: 2 * eps * depth * magnitude, relative: 0, ulps: nil)
    }
}

public enum Verdict: Equatable, Sendable {
    case equal
    case withinTolerance
    case classMismatch
    case outOfTolerance(absError: Double, relError: Double)
}

public enum Comparator {
    public static func compare(_ a: Double, _ b: Double, tolerance: Tolerance) -> Verdict {
        if a.isNaN && b.isNaN { return .equal }
        if a.bitPattern == b.bitPattern { return .equal }
        if a.isNaN != b.isNaN { return .classMismatch }
        if a.isInfinite || b.isInfinite {
            // Both infinite with equal sign would have been bitwise equal above.
            return .classMismatch
        }
        let absError = abs(a - b)
        let scale = max(abs(a), abs(b))
        let relError = scale > 0 ? absError / scale : 0
        if absError <= max(tolerance.absolute, tolerance.relative * scale) { return .withinTolerance }
        if let u = tolerance.ulps, ulpDistance(a, b) <= u { return .withinTolerance }
        return .outOfTolerance(absError: absError, relError: relError)
    }

    public static func compare(_ a: Float, _ b: Float, tolerance: Tolerance) -> Verdict {
        if a.isNaN && b.isNaN { return .equal }
        if a.bitPattern == b.bitPattern { return .equal }
        if a.isNaN != b.isNaN { return .classMismatch }
        if a.isInfinite || b.isInfinite { return .classMismatch }
        let da = Double(a), db = Double(b)
        let absError = abs(da - db)
        let scale = max(abs(da), abs(db))
        let relError = scale > 0 ? absError / scale : 0
        if absError <= max(tolerance.absolute, tolerance.relative * scale) { return .withinTolerance }
        if let u = tolerance.ulps, ulpDistance(a, b) <= u { return .withinTolerance }
        return .outOfTolerance(absError: absError, relError: relError)
    }

    /// Distance in representable steps between two finite doubles (sign-aware ordering of bit patterns).
    public static func ulpDistance(_ a: Double, _ b: Double) -> UInt64 {
        func ordered(_ x: Double) -> Int64 {
            let i = Int64(bitPattern: x.bitPattern)
            return i >= 0 ? i : Int64.min - i
        }
        let (x, y) = (ordered(a), ordered(b))
        return x >= y ? UInt64(x - y) : UInt64(y - x)
    }

    public static func ulpDistance(_ a: Float, _ b: Float) -> UInt64 {
        func ordered(_ x: Float) -> Int32 {
            let i = Int32(bitPattern: x.bitPattern)
            return i >= 0 ? i : Int32.min - i
        }
        let (x, y) = (Int64(ordered(a)), Int64(ordered(b)))
        return x >= y ? UInt64(x - y) : UInt64(y - x)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter ComparatorTests 2>&1 | tail -1`
Expected: `Executed 9 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Tolerance.swift Tests/VectorAccelerateTests/Harness/ComparatorTests.swift
git commit -q -m "feat(kit): Tolerance, Verdict, Comparator with derived sum tolerances

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `RecallPolicy` and `Recall`

**Files:**
- Create: `Sources/VectorTestKit/Recall.swift`
- Test: `Tests/VectorAccelerateTests/Harness/RecallTests.swift`

**Interfaces:**
- Produces: `public struct RecallPolicy: Codable, Sendable, Equatable { public var k: Int; public var epsilon: Double; public init(k:epsilon:) }`; `public struct Neighbor: Sendable, Equatable { public var id: Int; public var distance: Double; public init(id:distance:) }`; `public enum RecallError: Error, Equatable { case insufficientGroundTruth(have: Int, need: Int), invalidK }`; `public enum Recall { public static func atK(returned: [Neighbor], groundTruth: [Neighbor], policy: RecallPolicy) throws -> Double }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

/// Distance-epsilon recall (spec §4.6 as amended): first K results only, each ID credited once,
/// a result counts if its distance ≤ (1+ε) × ground-truth k-th distance.
final class RecallTests: XCTestCase {
    private let gt = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 3, distance: 3.0), Neighbor(id: 4, distance: 4.0)]

    func testEpsilonFormCreditsTiesThatIDSetFormWouldNot() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 9, distance: 3.0)]
        let r = try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0))
        XCTAssertEqual(r, 1.0)
        let idSet = Set(returned.map(\.id)).intersection(gt.prefix(3).map(\.id)).count
        XCTAssertEqual(idSet, 2, "ID-set recall would be 2/3; epsilon form credits the tie at distance 3.0")
    }

    func testEpsilonWidensThreshold() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 9, distance: 3.2)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0.1)), 1.0)
    }

    func testDuplicateIDsCannotInflateCredit() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 1, distance: 1.0), Neighbor(id: 1, distance: 1.0)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 1.0 / 3.0, accuracy: 1e-12)
    }

    func testOnlyFirstKResultsAreConsidered() throws {
        let returned = [Neighbor(id: 7, distance: 100), Neighbor(id: 8, distance: 100), Neighbor(id: 9, distance: 100), Neighbor(id: 1, distance: 1.0)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 0.0)
    }

    func testEmptyReturnedIsZero() throws {
        XCTAssertEqual(try Recall.atK(returned: [], groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 0.0)
    }

    func testInsufficientGroundTruthThrows() {
        XCTAssertThrowsError(try Recall.atK(returned: [], groundTruth: Array(gt.prefix(2)), policy: RecallPolicy(k: 3, epsilon: 0))) { error in
            XCTAssertEqual(error as? RecallError, .insufficientGroundTruth(have: 2, need: 3))
        }
        XCTAssertThrowsError(try Recall.atK(returned: [], groundTruth: gt, policy: RecallPolicy(k: 0, epsilon: 0))) { error in
            XCTAssertEqual(error as? RecallError, .invalidK)
        }
    }

    func testClampedToUnitInterval() throws {
        // Every returned neighbour within threshold and k distinct IDs → exactly 1.0, never above.
        let returned = [Neighbor(id: 1, distance: 0.5), Neighbor(id: 2, distance: 0.5), Neighbor(id: 3, distance: 0.5)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 1.0)
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter RecallTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'Neighbor' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

public struct RecallPolicy: Codable, Sendable, Equatable {
    public var k: Int
    public var epsilon: Double
    public init(k: Int, epsilon: Double) { self.k = k; self.epsilon = epsilon }
}

public struct Neighbor: Sendable, Equatable {
    public var id: Int
    public var distance: Double
    public init(id: Int, distance: Double) { self.id = id; self.distance = distance }
}

public enum RecallError: Error, Equatable {
    case insufficientGroundTruth(have: Int, need: Int)
    case invalidK
}

/// Distance-threshold recall@k (spec §4.6). The caller supplies independently recomputed distances for
/// valid, eligible IDs and, for small corpora, passes `k = min(requestedK, eligibleCount)` (never 0).
public enum Recall {
    public static func atK(returned: [Neighbor], groundTruth: [Neighbor], policy: RecallPolicy) throws -> Double {
        guard policy.k >= 1 else { throw RecallError.invalidK }
        guard groundTruth.count >= policy.k else {
            throw RecallError.insufficientGroundTruth(have: groundTruth.count, need: policy.k)
        }
        let threshold = (1.0 + policy.epsilon) * groundTruth[policy.k - 1].distance
        var seen = Set<Int>()
        var credited = 0
        for n in returned.prefix(policy.k) {
            guard seen.insert(n.id).inserted else { continue }
            if n.distance <= threshold { credited += 1 }
        }
        return min(1.0, max(0.0, Double(credited) / Double(policy.k)))
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter RecallTests 2>&1 | tail -1`
Expected: `Executed 7 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Recall.swift Tests/VectorAccelerateTests/Harness/RecallTests.swift
git commit -q -m "feat(kit): distance-epsilon recall with first-K and single-credit rules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `Stats`, `Summary`, `blackHole`

**Files:**
- Create: `Sources/VectorTestKit/Stats.swift`
- Test: `Tests/VectorAccelerateTests/Harness/StatsTests.swift`

**Interfaces:**
- Produces: `public struct Summary: Codable, Sendable, Equatable { median, p90, p99, min, max, mean, rsd: Double; ci95: [Double]; count: Int }`; `public enum Stats { static func percentile(sorted: [Double], p: Double) -> Double; static func summarize(_ values: [Double], bootstrapSeed: UInt64) -> Summary }`; `@inline(never) public func blackHole<T>(_ value: T)`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

/// Percentiles use linear interpolation between order statistics (rank = p·(n−1)); ci95 is a seeded
/// percentile bootstrap of the median with 1,000 resamples (spec §4.7).
final class StatsTests: XCTestCase {
    func testPercentilesLinearInterpolation() {
        let v = (1...10).map(Double.init)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.5), 5.5, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.9), 9.1, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.99), 9.91, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0), 1)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 1), 10)
        XCTAssertEqual(Stats.percentile(sorted: [7], p: 0.9), 7)
        XCTAssertEqual(Stats.percentile(sorted: [], p: 0.5), 0)
    }

    func testSummaryFields() {
        let s = Stats.summarize([90, 100, 110], bootstrapSeed: 1)
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s.min, 90); XCTAssertEqual(s.max, 110); XCTAssertEqual(s.mean, 100)
        XCTAssertEqual(s.median, 100)
        XCTAssertEqual(s.rsd, 0.1, accuracy: 1e-12)   // sample stddev 10 / mean 100
        XCTAssertEqual(s.ci95.count, 2)
        XCTAssertLessThanOrEqual(s.ci95[0], s.median); XCTAssertGreaterThanOrEqual(s.ci95[1], s.median)
    }

    func testBootstrapIsReproducibleForSeed() {
        let v = (0..<50).map { Double($0 * 7 % 13) }
        XCTAssertEqual(Stats.summarize(v, bootstrapSeed: 42).ci95, Stats.summarize(v, bootstrapSeed: 42).ci95)
        XCTAssertNotEqual(Stats.summarize(v, bootstrapSeed: 42).ci95, Stats.summarize(v, bootstrapSeed: 43).ci95)
    }

    func testSingleAndEmptyInputs() {
        let one = Stats.summarize([5], bootstrapSeed: 0)
        XCTAssertEqual(one.rsd, 0); XCTAssertEqual(one.ci95, [5, 5])
        let none = Stats.summarize([], bootstrapSeed: 0)
        XCTAssertEqual(none.count, 0); XCTAssertEqual(none.median, 0); XCTAssertEqual(none.ci95, [0, 0])
    }

    func testBlackHoleAcceptsAnything() {
        blackHole([1, 2, 3]); blackHole("x"); blackHole(())
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter StatsTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'Stats' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

/// Descriptive statistics of one timing series (nanoseconds, or any positive quantity).
public struct Summary: Codable, Sendable, Equatable {
    public var median: Double
    public var p90: Double
    public var p99: Double
    public var min: Double
    public var max: Double
    public var mean: Double
    /// Sample standard deviation divided by the mean; 0 when count < 2 or mean == 0.
    public var rsd: Double
    /// 95% percentile-bootstrap interval of the median, `[lo, hi]`.
    public var ci95: [Double]
    public var count: Int

    public init(median: Double, p90: Double, p99: Double, min: Double, max: Double, mean: Double, rsd: Double, ci95: [Double], count: Int) {
        self.median = median; self.p90 = p90; self.p99 = p99; self.min = min; self.max = max
        self.mean = mean; self.rsd = rsd; self.ci95 = ci95; self.count = count
    }
}

public enum Stats {
    /// Linear interpolation between order statistics: rank = p·(n−1). Replaces the floor-index method
    /// the old IndexBenchmarkHarness.LatencyStats used.
    public static func percentile(sorted: [Double], p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        guard sorted.count > 1 else { return sorted[0] }
        let rank = Swift.min(Swift.max(p, 0), 1) * Double(sorted.count - 1)
        let lo = Int(rank.rounded(.down))
        let hi = Int(rank.rounded(.up))
        if lo == hi { return sorted[lo] }
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - Double(lo))
    }

    public static func summarize(_ values: [Double], bootstrapSeed: UInt64, resamples: Int = 1_000) -> Summary {
        let n = values.count
        guard n > 0 else { return Summary(median: 0, p90: 0, p99: 0, min: 0, max: 0, mean: 0, rsd: 0, ci95: [0, 0], count: 0) }
        let sorted = values.sorted()
        let mean = values.reduce(0, +) / Double(n)
        let rsd: Double
        if n >= 2, mean != 0 {
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n - 1)
            rsd = variance.squareRoot() / mean
        } else { rsd = 0 }
        let median = percentile(sorted: sorted, p: 0.5)
        var rng = SeededRNG(seed: bootstrapSeed, stream: 0)
        var medians: [Double] = []
        medians.reserveCapacity(resamples)
        var scratch = [Double](repeating: 0, count: n)
        for _ in 0..<resamples {
            for i in 0..<n { scratch[i] = values[rng.nextInt(bound: n)] }
            medians.append(percentile(sorted: scratch.sorted(), p: 0.5))
        }
        let ms = medians.sorted()
        return Summary(
            median: median, p90: percentile(sorted: sorted, p: 0.9), p99: percentile(sorted: sorted, p: 0.99),
            min: sorted[0], max: sorted[n - 1], mean: mean, rsd: rsd,
            ci95: [percentile(sorted: ms, p: 0.025), percentile(sorted: ms, p: 0.975)], count: n)
    }
}

/// Dead-code-elimination barrier for benchmarked expressions whose results are otherwise unused.
@inline(never)
public func blackHole<T>(_ value: T) {
    withExtendedLifetime(value) {}
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter StatsTests 2>&1 | tail -1`
Expected: `Executed 5 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Stats.swift Tests/VectorAccelerateTests/Harness/StatsTests.swift
git commit -q -m "feat(kit): Stats summary with interpolated percentiles and seeded bootstrap CI; blackHole

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Artifact schema v1

**Files:**
- Create: `Sources/VectorTestKit/Schema.swift`
- Test: `Tests/VectorAccelerateTests/Harness/ArtifactSchemaTests.swift`

**Interfaces:**
- Produces (all `Codable, Sendable, Equatable`, JSON keys = property names): `BenchArtifact { schemaVersion: Int; package: String; metadata: Metadata; cases: [CaseRecord]; func validate() throws }`, `Metadata`, `GPUInfo`, `PowerInfo`, `ThermalInfo`, `PreflightRecord`, `StartEnd`, `Filters`, `CaseRecord`, `CaseIdentity`, `CaseStatus` (`measured|unavailable|failed`), `Timing { wall, commandBuffer, encoder: Series? }`, `TimingKind` (`wall|commandBuffer|encoder`), `Series { unit: String; raw: [[Double]]; summary: Summary; bias: String? }`, `Correctness`, `CorrectnessStatus` (`pass|fail|notMeasured`), `Resources`, `ArtifactValidationError: Error, Equatable { case unsupportedSchemaVersion(Int), missing(String), invalid(String) }`. Also `public extension Timing { subscript(kind: TimingKind) -> Series? }` and `public extension CaseRecord { func validate() throws }` (per-case checks, reused by `Compare`).

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

final class ArtifactSchemaTests: XCTestCase {
    static func sampleArtifact(validation: Bool = false) -> BenchArtifact {
        let series = Series(unit: "ns", raw: [[1, 2, 3], [2, 3, 4]], summary: Stats.summarize([1, 2, 3, 2, 3, 4], bootstrapSeed: 1), bias: nil)
        let identity = CaseIdentity(id: "probe.gpu_copy.bytes=8", fixtureDigest: "abc", generatorVersion: "1", seed: 42, stream: 0,
                                    route: "gpu.probe_copy", variant: nil, dataDescriptor: "pattern")
        let c = CaseRecord(id: "probe.gpu_copy.bytes=8", identity: identity, status: .measured, failureReason: nil,
                           samples: 2, iterations: 3, warmupUsed: [3, 3], unitCount: 1,
                           timing: Timing(wall: series, commandBuffer: nil, encoder: nil),
                           correctness: Correctness(status: .pass, comparator: "memcmp", tolerance: nil, maxAbsError: nil, maxRelError: nil, recall: nil, recallPolicy: nil, notes: []),
                           resources: nil, notes: [])
        let md = Metadata(packageVersion: "0.7.0", gitSHA: "deadbeef", gitDirty: false, date: "2026-09-17T00:00:00Z",
                          os: "os26.5.2", osBuild: "25F84", arch: "arm64", cpuCores: 16, performanceCores: 12, efficiencyCores: 4,
                          memoryBytes: 1, deviceModel: "Apple M3 Max", modelIdentifier: "Mac15,9", hostname: "h",
                          swiftVersion: "6.3.3", buildConfiguration: "release", deviceTag: "arm64-os26.5.2-h",
                          environmentKey: "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval",
                          gpu: GPUInfo(name: "Apple M3 Max", coreCount: 40, unifiedMemory: true, maxThreadgroupBytes: 32768, metal4: true, metalToolchain: "32023.883"),
                          power: PowerInfo(source: "ac", lowPowerMode: false), thermal: ThermalInfo(start: "nominal", end: "nominal"),
                          preflight: PreflightRecord(cpuCopyGBps: StartEnd(start: 50, end: 50), gpuCopyGBps: StartEnd(start: 370, end: 370), driftPercent: 0, quiet: true, reasons: []),
                          validationLayers: validation, runSeed: 42, runLabel: nil, mode: "quick", filters: nil, flags: [:])
        return BenchArtifact(schemaVersion: 1, package: "VectorAccelerate", metadata: md, cases: [c])
    }

    func testRoundTrip() throws {
        let a = Self.sampleArtifact()
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(a)
        let back = try JSONDecoder().decode(BenchArtifact.self, from: data)
        XCTAssertEqual(back, a)
        try back.validate()
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(json.contains("\"preflight\"") && json.contains("\"environmentKey\"") && json.contains("\"commandBuffer\":null") == false)
    }

    func testValidationLayersArtifactIsValid() throws {
        try Self.sampleArtifact(validation: true).validate()
    }

    func testUnknownSchemaVersionRejected() {
        var a = Self.sampleArtifact(); a.schemaVersion = 2
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .unsupportedSchemaVersion(2)) }
    }

    func testRawShapeMustMatchSamplesAndIterations() {
        var a = Self.sampleArtifact(); a.cases[0].iterations = 4
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: timing.wall raw shape must be samples×iterations (2×4)")) }
    }

    func testWarmupUsedCountMustEqualSamples() {
        var a = Self.sampleArtifact(); a.cases[0].warmupUsed = [3]
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: warmupUsed.count must equal samples")) }
    }

    func testMeasuredRequiresWall() {
        var a = Self.sampleArtifact(); a.cases[0].timing.wall = nil
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("case probe.gpu_copy.bytes=8: timing.wall")) }
    }

    func testRecallRange() {
        var a = Self.sampleArtifact(); a.cases[0].correctness?.recall = 1.5
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: correctness.recall must be in [0, 1]")) }
    }

    func testMissingRequiredMetadataFieldFailsDecoding() throws {
        let data = try JSONEncoder().encode(Self.sampleArtifact())
        var obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var md = obj["metadata"] as! [String: Any]; md.removeValue(forKey: "environmentKey"); obj["metadata"] = md
        let stripped = try JSONSerialization.data(withJSONObject: obj)
        XCTAssertThrowsError(try JSONDecoder().decode(BenchArtifact.self, from: stripped))
    }

    func testTimingSubscript() {
        let a = Self.sampleArtifact()
        XCTAssertNotNil(a.cases[0].timing[.wall]); XCTAssertNil(a.cases[0].timing[.encoder])
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter ArtifactSchemaTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'Series' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

// Artifact schema v1 (spec §4.8). Property names are the JSON keys.

public enum TimingKind: String, Codable, Sendable, CaseIterable { case wall, commandBuffer, encoder }
public enum CaseStatus: String, Codable, Sendable { case measured, unavailable, failed }
public enum CorrectnessStatus: String, Codable, Sendable { case pass, fail, notMeasured }

public enum ArtifactValidationError: Error, Equatable {
    case unsupportedSchemaVersion(Int)
    case missing(String)
    case invalid(String)
}

public struct StartEnd: Codable, Sendable, Equatable {
    public var start: Double
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
}

public struct GPUInfo: Codable, Sendable, Equatable {
    public var name: String
    public var coreCount: Int?
    public var unifiedMemory: Bool
    public var maxThreadgroupBytes: Int
    public var metal4: Bool
    public var metalToolchain: String?
    public init(name: String, coreCount: Int?, unifiedMemory: Bool, maxThreadgroupBytes: Int, metal4: Bool, metalToolchain: String?) {
        self.name = name; self.coreCount = coreCount; self.unifiedMemory = unifiedMemory
        self.maxThreadgroupBytes = maxThreadgroupBytes; self.metal4 = metal4; self.metalToolchain = metalToolchain
    }
}

public struct PowerInfo: Codable, Sendable, Equatable {
    /// "ac" | "battery" | "unknown"
    public var source: String
    public var lowPowerMode: Bool
    public init(source: String, lowPowerMode: Bool) { self.source = source; self.lowPowerMode = lowPowerMode }
}

public struct ThermalInfo: Codable, Sendable, Equatable {
    /// "nominal" | "fair" | "serious" | "critical" | "unknown"
    public var start: String
    public var end: String
    public init(start: String, end: String) { self.start = start; self.end = end }
}

public struct PreflightRecord: Codable, Sendable, Equatable {
    public var cpuCopyGBps: StartEnd
    public var gpuCopyGBps: StartEnd?
    public var driftPercent: Double
    public var quiet: Bool
    public var reasons: [String]
    public init(cpuCopyGBps: StartEnd, gpuCopyGBps: StartEnd?, driftPercent: Double, quiet: Bool, reasons: [String]) {
        self.cpuCopyGBps = cpuCopyGBps; self.gpuCopyGBps = gpuCopyGBps; self.driftPercent = driftPercent; self.quiet = quiet; self.reasons = reasons
    }
}

public struct Filters: Codable, Sendable, Equatable {
    public var mode: String
    public var include: [String]?
    public var exclude: [String]?
    public init(mode: String = "glob", include: [String]?, exclude: [String]?) { self.mode = mode; self.include = include; self.exclude = exclude }
}

public struct Metadata: Codable, Sendable, Equatable {
    public var packageVersion: String
    public var gitSHA: String?
    public var gitDirty: Bool
    public var date: String
    public var os: String
    public var osBuild: String
    public var arch: String
    public var cpuCores: Int
    public var performanceCores: Int?
    public var efficiencyCores: Int?
    public var memoryBytes: UInt64
    public var deviceModel: String
    public var modelIdentifier: String
    public var hostname: String
    public var swiftVersion: String
    public var buildConfiguration: String
    public var deviceTag: String
    public var environmentKey: String
    public var gpu: GPUInfo?
    public var power: PowerInfo
    public var thermal: ThermalInfo
    public var preflight: PreflightRecord
    public var validationLayers: Bool
    public var runSeed: UInt64
    public var runLabel: String?
    public var mode: String
    public var filters: Filters?
    public var flags: [String: String]

    public init(packageVersion: String, gitSHA: String?, gitDirty: Bool, date: String, os: String, osBuild: String, arch: String,
                cpuCores: Int, performanceCores: Int?, efficiencyCores: Int?, memoryBytes: UInt64, deviceModel: String,
                modelIdentifier: String, hostname: String, swiftVersion: String, buildConfiguration: String, deviceTag: String,
                environmentKey: String, gpu: GPUInfo?, power: PowerInfo, thermal: ThermalInfo, preflight: PreflightRecord,
                validationLayers: Bool, runSeed: UInt64, runLabel: String?, mode: String, filters: Filters?, flags: [String: String]) {
        self.packageVersion = packageVersion; self.gitSHA = gitSHA; self.gitDirty = gitDirty; self.date = date; self.os = os
        self.osBuild = osBuild; self.arch = arch; self.cpuCores = cpuCores; self.performanceCores = performanceCores
        self.efficiencyCores = efficiencyCores; self.memoryBytes = memoryBytes; self.deviceModel = deviceModel
        self.modelIdentifier = modelIdentifier; self.hostname = hostname; self.swiftVersion = swiftVersion
        self.buildConfiguration = buildConfiguration; self.deviceTag = deviceTag; self.environmentKey = environmentKey
        self.gpu = gpu; self.power = power; self.thermal = thermal; self.preflight = preflight
        self.validationLayers = validationLayers; self.runSeed = runSeed; self.runLabel = runLabel; self.mode = mode
        self.filters = filters; self.flags = flags
    }
}

public struct CaseIdentity: Codable, Sendable, Equatable {
    public var id: String
    public var fixtureDigest: String
    public var generatorVersion: String
    public var seed: UInt64
    public var stream: UInt64
    public var route: String
    public var variant: String?
    public var dataDescriptor: String
    public init(id: String, fixtureDigest: String, generatorVersion: String, seed: UInt64, stream: UInt64, route: String, variant: String?, dataDescriptor: String) {
        self.id = id; self.fixtureDigest = fixtureDigest; self.generatorVersion = generatorVersion; self.seed = seed
        self.stream = stream; self.route = route; self.variant = variant; self.dataDescriptor = dataDescriptor
    }
}

public struct Series: Codable, Sendable, Equatable {
    public var unit: String
    /// raw[sample][iteration]
    public var raw: [[Double]]
    public var summary: Summary
    public var bias: String?
    public init(unit: String = "ns", raw: [[Double]], summary: Summary, bias: String?) { self.unit = unit; self.raw = raw; self.summary = summary; self.bias = bias }
}

public struct Timing: Codable, Sendable, Equatable {
    public var wall: Series?
    public var commandBuffer: Series?
    public var encoder: Series?
    public init(wall: Series?, commandBuffer: Series?, encoder: Series?) { self.wall = wall; self.commandBuffer = commandBuffer; self.encoder = encoder }

    public subscript(kind: TimingKind) -> Series? {
        switch kind { case .wall: return wall; case .commandBuffer: return commandBuffer; case .encoder: return encoder }
    }
}

public struct Correctness: Codable, Sendable, Equatable {
    public var status: CorrectnessStatus
    public var comparator: String
    public var tolerance: Tolerance?
    public var maxAbsError: Double?
    public var maxRelError: Double?
    public var recall: Double?
    public var recallPolicy: RecallPolicy?
    public var notes: [String]
    public init(status: CorrectnessStatus, comparator: String, tolerance: Tolerance?, maxAbsError: Double?, maxRelError: Double?, recall: Double?, recallPolicy: RecallPolicy?, notes: [String]) {
        self.status = status; self.comparator = comparator; self.tolerance = tolerance; self.maxAbsError = maxAbsError
        self.maxRelError = maxRelError; self.recall = recall; self.recallPolicy = recallPolicy; self.notes = notes
    }
}

public struct Resources: Codable, Sendable, Equatable {
    public var staticThreadgroupBytes: Int?
    public var maxThreadgroupBytes: Int?
    public var poolHighWaterBytes: Int?
    public var notes: [String]
    public init(staticThreadgroupBytes: Int?, maxThreadgroupBytes: Int?, poolHighWaterBytes: Int?, notes: [String]) {
        self.staticThreadgroupBytes = staticThreadgroupBytes; self.maxThreadgroupBytes = maxThreadgroupBytes
        self.poolHighWaterBytes = poolHighWaterBytes; self.notes = notes
    }
}

public struct CaseRecord: Codable, Sendable, Equatable {
    public var id: String
    public var identity: CaseIdentity
    public var status: CaseStatus
    public var failureReason: String?
    public var samples: Int
    public var iterations: Int
    public var warmupUsed: [Int]
    public var unitCount: Int
    public var timing: Timing
    public var correctness: Correctness?
    public var resources: Resources?
    public var notes: [String]
    public init(id: String, identity: CaseIdentity, status: CaseStatus, failureReason: String?, samples: Int, iterations: Int, warmupUsed: [Int],
                unitCount: Int, timing: Timing, correctness: Correctness?, resources: Resources?, notes: [String]) {
        self.id = id; self.identity = identity; self.status = status; self.failureReason = failureReason; self.samples = samples
        self.iterations = iterations; self.warmupUsed = warmupUsed; self.unitCount = unitCount; self.timing = timing
        self.correctness = correctness; self.resources = resources; self.notes = notes
    }
}

public struct BenchArtifact: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var package: String
    public var metadata: Metadata
    public var cases: [CaseRecord]
    public init(schemaVersion: Int = 1, package: String, metadata: Metadata, cases: [CaseRecord]) {
        self.schemaVersion = schemaVersion; self.package = package; self.metadata = metadata; self.cases = cases
    }

    /// Structural validation beyond what `Codable` enforces (spec §4.8).
    public func validate() throws {
        guard schemaVersion == 1 else { throw ArtifactValidationError.unsupportedSchemaVersion(schemaVersion) }
        if metadata.environmentKey.isEmpty { throw ArtifactValidationError.missing("metadata.environmentKey") }
        if metadata.deviceModel.isEmpty { throw ArtifactValidationError.missing("metadata.deviceModel") }
        for c in cases { try c.validate() }
    }
}

public extension CaseRecord {
    /// Per-case structural checks; also used by `Compare` on baseline records.
    func validate() throws {
        if warmupUsed.count != samples { throw ArtifactValidationError.invalid("case \(id): warmupUsed.count must equal samples") }
        if status == .measured && timing.wall == nil { throw ArtifactValidationError.missing("case \(id): timing.wall") }
        for kind in TimingKind.allCases {
            guard let s = timing[kind] else { continue }
            let shapeOK = s.raw.count == samples && s.raw.allSatisfy { $0.count == iterations }
            if !shapeOK { throw ArtifactValidationError.invalid("case \(id): timing.\(kind.rawValue) raw shape must be samples×iterations (\(samples)×\(iterations))") }
        }
        if let r = correctness?.recall, !(0...1).contains(r) { throw ArtifactValidationError.invalid("case \(id): correctness.recall must be in [0, 1]") }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter ArtifactSchemaTests 2>&1 | tail -1`
Expected: `Executed 9 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Schema.swift Tests/VectorAccelerateTests/Harness/ArtifactSchemaTests.swift
git commit -q -m "feat(kit): artifact schema v1 with structural validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `CaseID`, registry duplicate check, `WarmupPolicy`

**Files:**
- Create: `Sources/VectorTestKit/CaseID.swift`, `Sources/VectorTestKit/WarmupPolicy.swift`
- Test: `Tests/VectorAccelerateTests/Harness/CaseIDTests.swift`, `Tests/VectorAccelerateTests/Harness/WarmupPolicyTests.swift`

**Interfaces:**
- Produces: `public struct CaseID: Codable, Hashable, Sendable, CustomStringConvertible { public let rawValue: String; public init(validating: String) throws; public var family: String }`; `public enum CaseIDError: Error, Equatable { case malformed(String) }`; `public enum CaseRegistryCheck { public static func duplicates(_ ids: [CaseID]) -> [CaseID] }`; `public struct WarmupPolicy: Sendable { public var window: Int; public var cov: Double; public var minimum: Int; public var maximum: Int; public static let standard: WarmupPolicy; public func shouldStop(after wallNs: [Double]) -> Bool }`.

- [ ] **Step 1: Write the failing tests**

`CaseIDTests.swift`:

```swift
import XCTest
import VectorTestKit

/// Grammar (spec §5.4 as amended): `<family>(.<op>)+(.<param>=<value>)*(.<variant>)?`, lowercase, `[a-z0-9_=.]` only.
final class CaseIDTests: XCTestCase {
    func testAcceptsTheTwoM0IDs() throws {
        XCTAssertEqual(try CaseID(validating: "probe.gpu_copy.bytes=134217728").family, "probe")
        XCTAssertEqual(try CaseID(validating: "distance.l2.batch.n=10000.d=768").family, "distance")
    }

    func testAcceptsVariantAfterParams() throws {
        XCTAssertNoThrow(try CaseID(validating: "ivf.list_search.n=4096.k=33.isolated"))
        XCTAssertNoThrow(try CaseID(validating: "probe.gpu_copy.bytes=1.k"))   // single-letter variant is legal
        XCTAssertNoThrow(try CaseID(validating: "distance.l2.batch"))            // several bare op segments, no params
    }

    func testRejects() {
        for bad in ["Probe.gpu_copy", "probe", "probe.gpu copy", "probe.gpu_copy.seed=42x!", "probe..x", ".probe.x", "probe.x."] {
            XCTAssertThrowsError(try CaseID(validating: bad), bad)
        }
    }

    func testDuplicatesDetected() throws {
        let a = try CaseID(validating: "a.b"), b = try CaseID(validating: "a.c")
        XCTAssertEqual(CaseRegistryCheck.duplicates([a, b, a]), [a])
        XCTAssertEqual(CaseRegistryCheck.duplicates([a, b]), [])
    }
}
```

`WarmupPolicyTests.swift`:

```swift
import XCTest
import VectorTestKit

/// Spec §5.2: stop when ≥ minimum ran and CoV of the last `window` wall times < cov, or at maximum.
final class WarmupPolicyTests: XCTestCase {
    func testStableSeriesStopsAtMinimum() {
        let p = WarmupPolicy.standard
        XCTAssertFalse(p.shouldStop(after: [100, 100]))
        XCTAssertTrue(p.shouldStop(after: [100, 100, 100]))
    }

    func testNoisySeriesRunsToMaximum() {
        let p = WarmupPolicy.standard
        let noisy = (0..<49).map { Double(100 + ($0 % 2) * 50) }   // CoV ≈ 0.2
        XCTAssertFalse(p.shouldStop(after: noisy))
        XCTAssertTrue(p.shouldStop(after: noisy + [100]))            // 50 == maximum
    }

    func testWindowOnlyLooksAtTail() {
        let p = WarmupPolicy.standard
        let settled = [1000, 10, 1000, 10] + Array(repeating: 100.0, count: 8)
        XCTAssertTrue(p.shouldStop(after: settled))
    }

    func testStandardValues() {
        XCTAssertEqual(WarmupPolicy.standard.window, 8); XCTAssertEqual(WarmupPolicy.standard.cov, 0.05)
        XCTAssertEqual(WarmupPolicy.standard.minimum, 3); XCTAssertEqual(WarmupPolicy.standard.maximum, 50)
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `swift test --filter "CaseIDTests|WarmupPolicyTests" 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'CaseID' in scope`.

- [ ] **Step 3: Implement**

`CaseID.swift`:

```swift
import Foundation

public enum CaseIDError: Error, Equatable { case malformed(String) }

/// Stable case identity per the grammar in spec §5.4 (amended: one or more bare op segments). Never contains seeds, digests, or timestamps.
public struct CaseID: Codable, Hashable, Sendable, CustomStringConvertible {
    public let rawValue: String

    private static let grammar = try! NSRegularExpression(pattern: #"^[a-z0-9_]+(\.[a-z0-9_]+)+(\.[a-z0-9_]+=[a-z0-9_]+)*(\.[a-z0-9_]+)?$"#)

    public init(validating raw: String) throws {
        let range = NSRange(raw.startIndex..., in: raw)
        guard Self.grammar.firstMatch(in: raw, range: range) != nil else { throw CaseIDError.malformed(raw) }
        self.rawValue = raw
    }

    public var family: String { String(rawValue.split(separator: ".")[0]) }
    public var description: String { rawValue }

    public init(from decoder: Decoder) throws {
        try self.init(validating: try decoder.singleValueContainer().decode(String.self))
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(rawValue)
    }
}

public enum CaseRegistryCheck {
    /// IDs that appear more than once, in first-seen order, each listed once.
    public static func duplicates(_ ids: [CaseID]) -> [CaseID] {
        var seen = Set<CaseID>(), dups: [CaseID] = [], reported = Set<CaseID>()
        for id in ids where !seen.insert(id).inserted && reported.insert(id).inserted { dups.append(id) }
        return dups
    }
}
```

`WarmupPolicy.swift`:

```swift
import Foundation

/// Adaptive warm-up rule (spec §5.2). Values are the initial contract; SP1 retunes them from data.
public struct WarmupPolicy: Sendable {
    public var window: Int
    public var cov: Double
    public var minimum: Int
    public var maximum: Int

    public init(window: Int, cov: Double, minimum: Int, maximum: Int) {
        self.window = window; self.cov = cov; self.minimum = minimum; self.maximum = maximum
    }

    public static let standard = WarmupPolicy(window: 8, cov: 0.05, minimum: 3, maximum: 50)

    public func shouldStop(after wallNs: [Double]) -> Bool {
        let n = wallNs.count
        if n >= maximum { return true }
        if n < minimum { return false }
        let tail = Array(wallNs.suffix(window))
        let mean = tail.reduce(0, +) / Double(tail.count)
        guard mean > 0, tail.count >= 2 else { return true }
        let variance = tail.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(tail.count - 1)
        return variance.squareRoot() / mean < cov
    }
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `swift test --filter "CaseIDTests|WarmupPolicyTests" 2>&1 | tail -1`
Expected: `Executed 8 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/CaseID.swift Sources/VectorTestKit/WarmupPolicy.swift Tests/VectorAccelerateTests/Harness/CaseIDTests.swift Tests/VectorAccelerateTests/Harness/WarmupPolicyTests.swift
git commit -q -m "feat(kit): CaseID grammar, registry duplicate check, warm-up policy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Host fingerprint, environment key, preflight probe (kit)

**Files:**
- Create: `Sources/VectorTestKit/HostFingerprint.swift`, `Sources/VectorTestKit/PreflightProbe.swift`
- Test: `Tests/VectorAccelerateTests/Harness/FingerprintTests.swift`

**Interfaces:**
- Produces: `public struct HostFingerprint: Sendable, Equatable` (fields below) with `static func gather(packageRoot: URL?) -> HostFingerprint`, `static func thermalStateName() -> String`, `static func detectSwiftVersion() -> String`, `static var buildConfiguration: String`, `static func deviceTag(arch:osVersion:hostname:) -> String`; `public enum Subprocess { static func run(_ executable: String, _ args: [String], cwd: URL? = nil, timeout: TimeInterval = 5) -> String? }`; `public enum PowerState { static func read() -> PowerInfo }`; `public enum EnvironmentKey { static func slug(_:) -> String; static func make(chip:gpuCores:osBuild:metalToolchain:swiftVersion:buildConfiguration:validationLayers:) -> String }`; `public enum PreflightProbe { static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 3) -> Double; static func evaluate(cpu: StartEnd, gpu: StartEnd?, thermalStart: String, thermalEnd: String, gpuDeviceExists: Bool, driftLimitPercent: Double = 5) -> PreflightRecord }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

final class FingerprintTests: XCTestCase {
    func testGatherPopulatesHostFacts() {
        let f = HostFingerprint.gather(packageRoot: RepoPaths.root)
        XCTAssertFalse(f.chip.isEmpty); XCTAssertFalse(f.modelIdentifier.isEmpty)
        XCTAssertGreaterThan(f.cpuCores, 0); XCTAssertGreaterThan(f.memoryBytes, 0)
        XCTAssertFalse(f.osBuild.isEmpty); XCTAssertTrue(f.osVersion.hasPrefix("os"))
        XCTAssertEqual(f.arch, "arm64")
        XCTAssertFalse(f.hostname.isEmpty)
        XCTAssertNotNil(f.gitSHA, "tests run inside the repository; git rev-parse must succeed")
        XCTAssertTrue(["debug", "release"].contains(f.buildConfiguration))
        XCTAssertTrue(["ac", "battery", "unknown"].contains(f.power.source))
    }

    func testEnvironmentKeyExcludesHostnameAndTracksValidationFlag() {
        let a = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: 40, osBuild: "25F84", metalToolchain: "32023.883", swiftVersion: "6.3.3", buildConfiguration: "release", validationLayers: false)
        XCTAssertEqual(a, "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval")
        let b = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: 40, osBuild: "25F84", metalToolchain: "32023.883", swiftVersion: "6.3.3", buildConfiguration: "release", validationLayers: true)
        XCTAssertTrue(b.hasSuffix("-val")); XCTAssertNotEqual(a, b)
        let c = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: nil, osBuild: "25F84", metalToolchain: nil, swiftVersion: "6.3.3", buildConfiguration: "debug", validationLayers: false)
        XCTAssertEqual(c, "applem3max-g?-25F84-mtlunknown-swift6.3.3-debug-noval")
        XCTAssertFalse(a.contains(ProcessInfo.processInfo.hostName))
    }

    func testDeviceTagMatchesVectorCoreShape() {
        XCTAssertEqual(HostFingerprint.deviceTag(arch: "arm64", osVersion: "os26.5.2", hostname: "my mac"), "arm64-os26.5.2-my-mac")
    }

    func testSwiftVersionDetectionParses() {
        let v = HostFingerprint.detectSwiftVersion()
        XCTAssertTrue(v.range(of: #"^\d+\.\d+(\.\d+)?$"#, options: .regularExpression) != nil, "got \(v)")
    }

    func testThermalStateNameIsKnown() {
        XCTAssertTrue(["nominal", "fair", "serious", "critical", "unknown"].contains(HostFingerprint.thermalStateName()))
    }

    func testCPUCopyProbeIsPositive() {
        XCTAssertGreaterThan(PreflightProbe.cpuCopyGBps(bytes: 16 << 20, repeats: 2), 0)
    }

    func testPreflightEvaluate() {
        let quiet = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 49), gpu: StartEnd(start: 370, end: 372), thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: true)
        XCTAssertTrue(quiet.quiet); XCTAssertEqual(quiet.driftPercent, 2.0, accuracy: 1e-9); XCTAssertEqual(quiet.reasons, [])
        let drift = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 40), gpu: nil, thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: false)
        XCTAssertFalse(drift.quiet); XCTAssertEqual(drift.driftPercent, 20.0, accuracy: 1e-9)
        XCTAssertEqual(drift.reasons, ["drift 20.0% exceeds 5.0%"])
        let hot = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 50), gpu: StartEnd(start: 1, end: 1), thermalStart: "nominal", thermalEnd: "fair", gpuDeviceExists: true)
        XCTAssertEqual(hot.reasons, ["thermal state end=fair (not nominal)"])
        let noGPU = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 50), gpu: nil, thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: true)
        XCTAssertEqual(noGPU.reasons, ["GPU probe unavailable although a Metal device exists"])
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter FingerprintTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'HostFingerprint' in scope`.

- [ ] **Step 3: Implement `HostFingerprint.swift`**

```swift
import Foundation
import Darwin

enum Sysctl {
    static func string(_ key: String) -> String? {
        var size = 0
        guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(key, &buf, &size, nil, 0) == 0 else { return nil }
        let s = String(decoding: buf.prefix(max(size - 1, 0)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return s.isEmpty ? nil : s
    }
    static func int(_ key: String) -> Int? {
        var v: Int64 = 0; var size = MemoryLayout<Int64>.size
        guard sysctlbyname(key, &v, &size, nil, 0) == 0 else { return nil }
        return Int(v)
    }
}

/// Runs a fixed executable with fixed arguments, no shell, bounded by `timeout`; nil on any failure.
public enum Subprocess {
    public static func run(_ executable: String, _ args: [String], cwd: URL? = nil, timeout: TimeInterval = 5) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: executable)
        p.arguments = args
        if let cwd { p.currentDirectoryURL = cwd }
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = FileHandle.nullDevice
        let done = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in done.signal() }
        do { try p.run() } catch { return nil }
        // Drain on a background queue so a child that never exits cannot block the timeout below.
        final class Box: @unchecked Sendable { var data = Data() }
        let box = Box(); let drained = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async { box.data = pipe.fileHandleForReading.readDataToEndOfFile(); drained.signal() }
        if done.wait(timeout: .now() + timeout) == .timedOut { p.terminate(); return nil }
        drained.wait()
        guard p.terminationStatus == 0 else { return nil }
        return String(decoding: box.data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum PowerState {
    /// Parses `pmset -g batt` (power source) and `pmset -g` (lowpowermode). "unknown" if pmset is unavailable.
    public static func read() -> PowerInfo {
        var source = "unknown"
        if let batt = Subprocess.run("/usr/bin/pmset", ["-g", "batt"]) {
            if batt.contains("AC Power") { source = "ac" } else if batt.contains("Battery Power") { source = "battery" }
        }
        var low = false
        if let all = Subprocess.run("/usr/bin/pmset", ["-g"]) {
            for line in all.split(separator: "\n") where line.contains("lowpowermode") {
                low = line.trimmingCharacters(in: .whitespaces).hasSuffix("1")
            }
        }
        return PowerInfo(source: source, lowPowerMode: low)
    }
}

public enum EnvironmentKey {
    /// Lowercase alphanumerics only: "Apple M3 Max" → "applem3max".
    public static func slug(_ s: String) -> String {
        String(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
    }

    /// Hostname is deliberately excluded so identical hardware on different machines compares (spec §4.9).
    public static func make(chip: String, gpuCores: Int?, osBuild: String, metalToolchain: String?, swiftVersion: String,
                            buildConfiguration: String, validationLayers: Bool) -> String {
        let cores = gpuCores.map(String.init) ?? "?"
        return "\(slug(chip))-g\(cores)-\(osBuild)-mtl\(metalToolchain ?? "unknown")-swift\(swiftVersion)-\(buildConfiguration)-\(validationLayers ? "val" : "noval")"
    }
}

/// The Metal-free part of the environment fingerprint (spec §4.9). The executable adds `GPUInfo`.
public struct HostFingerprint: Sendable, Equatable {
    public var chip: String
    public var modelIdentifier: String
    public var cpuCores: Int
    public var performanceCores: Int?
    public var efficiencyCores: Int?
    public var memoryBytes: UInt64
    public var osVersion: String
    public var osBuild: String
    public var arch: String
    public var hostname: String
    public var swiftVersion: String
    public var buildConfiguration: String
    public var gitSHA: String?
    public var gitDirty: Bool
    public var power: PowerInfo

    public static var buildConfiguration: String {
        #if DEBUG
        return "debug"
        #else
        return "release"
        #endif
    }

    public static func detectSwiftVersion() -> String {
        guard let out = Subprocess.run("/usr/bin/swift", ["--version"]),
              let m = out.range(of: #"Swift version (\d+\.\d+(\.\d+)?)"#, options: .regularExpression) else { return "unknown" }
        return String(out[m]).replacingOccurrences(of: "Swift version ", with: "")
    }

    public static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// VectorCore-compatible tag: `<arch>-<os>-<hostname>` (spaces → dashes) unless `VA_DEVICE_TAG` is set.
    public static func deviceTag(arch: String, osVersion: String, hostname: String) -> String {
        if let tag = ProcessInfo.processInfo.environment["VA_DEVICE_TAG"], !tag.isEmpty { return tag }
        return "\(arch)-\(osVersion)-\(hostname.replacingOccurrences(of: " ", with: "-"))"
    }

    public static func gather(packageRoot: URL?) -> HostFingerprint {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        #if arch(arm64)
        let arch = "arm64"
        #else
        let arch = "x86_64"
        #endif
        let sha = packageRoot.flatMap { Subprocess.run("/usr/bin/git", ["rev-parse", "HEAD"], cwd: $0) }
        let dirty = packageRoot.flatMap { Subprocess.run("/usr/bin/git", ["status", "--porcelain"], cwd: $0) }.map { !$0.isEmpty } ?? false
        return HostFingerprint(
            chip: Sysctl.string("machdep.cpu.brand_string") ?? "unknown",
            modelIdentifier: Sysctl.string("hw.model") ?? "unknown",
            cpuCores: Sysctl.int("hw.ncpu") ?? ProcessInfo.processInfo.activeProcessorCount,
            performanceCores: Sysctl.int("hw.perflevel0.physicalcpu"),
            efficiencyCores: Sysctl.int("hw.perflevel1.physicalcpu"),
            memoryBytes: UInt64(Sysctl.int("hw.memsize") ?? 0),
            osVersion: "os\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
            osBuild: Sysctl.string("kern.osversion") ?? "unknown",
            arch: arch,
            hostname: ProcessInfo.processInfo.hostName,
            swiftVersion: detectSwiftVersion(),
            buildConfiguration: buildConfiguration,
            gitSHA: sha, gitDirty: dirty,
            power: PowerState.read())
    }
}
```

- [ ] **Step 4: Implement `PreflightProbe.swift`**

```swift
import Foundation

/// Calibration probes run before and after every run (spec §4.10). CPU copy here; the GPU copy lives in
/// the executable because it needs Metal. GB/s counts read plus write: 2 × bytes / seconds / 1e9.
public enum PreflightProbe {
    public static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 3) -> Double {
        let src = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        let dst = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        defer { src.deallocate(); dst.deallocate() }
        memset(src, 0x5A, bytes); memset(dst, 0, bytes)
        var best = Double.infinity
        let clock = ContinuousClock()
        for _ in 0..<max(repeats, 1) {
            let d = clock.measure { memcpy(dst, src, bytes) }
            let seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            best = min(best, seconds)
        }
        blackHole(dst.load(as: UInt8.self))
        return best > 0 ? 2.0 * Double(bytes) / best / 1e9 : 0
    }

    public static func evaluate(cpu: StartEnd, gpu: StartEnd?, thermalStart: String, thermalEnd: String,
                                gpuDeviceExists: Bool, driftLimitPercent: Double = 5) -> PreflightRecord {
        func drift(_ se: StartEnd) -> Double { se.start > 0 ? abs(se.end - se.start) / se.start * 100 : 0 }
        var reasons: [String] = []
        let d = max(drift(cpu), gpu.map(drift) ?? 0)
        if d > driftLimitPercent { reasons.append("drift \(String(format: "%.1f", d))% exceeds \(String(format: "%.1f", driftLimitPercent))%") }
        if thermalStart != "nominal" { reasons.append("thermal state start=\(thermalStart) (not nominal)") }
        if thermalEnd != "nominal" { reasons.append("thermal state end=\(thermalEnd) (not nominal)") }
        if gpu == nil && gpuDeviceExists { reasons.append("GPU probe unavailable although a Metal device exists") }
        return PreflightRecord(cpuCopyGBps: cpu, gpuCopyGBps: gpu, driftPercent: d, quiet: reasons.isEmpty, reasons: reasons)
    }
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `swift test --filter "FingerprintTests|KitImportLintTests" 2>&1 | tail -1`
Expected: `Executed 8 tests, with 0 failures`

- [ ] **Step 6: Commit**

```bash
git add Sources/VectorTestKit/HostFingerprint.swift Sources/VectorTestKit/PreflightProbe.swift Tests/VectorAccelerateTests/Harness/FingerprintTests.swift
git commit -q -m "feat(kit): host fingerprint, environment key, power/thermal capture, CPU preflight probe

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: `RunStore`

**Files:**
- Create: `Sources/VectorTestKit/RunStore.swift`
- Test: `Tests/VectorAccelerateTests/Harness/RunStoreTests.swift`

**Interfaces:**
- Produces: `public enum RunStoreError: Error, Equatable { case runExists(String), refusedValidationLayers, refusedNotQuiet([String]), refusedDirty, noMeasuredCases, badApprovedPointer(String) }`; `public struct RunStore { public let root: URL; public init(root: URL); public func runURL(for: BenchArtifact, at: Date) -> URL; public func writeRun(_: BenchArtifact, at: Date = Date()) throws -> URL; public func captureBaseline(from: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int]; public func baseline(environmentKey:caseID:version:) throws -> CaseRecord?; public func approvedVersion(environmentKey:caseID:) throws -> Int?; public func approvedBaseline(environmentKey:caseID:) throws -> CaseRecord?; public static func load(_ url: URL) throws -> BenchArtifact; public static func encode(_: some Encodable) throws -> Data }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

final class RunStoreTests: XCTestCase {
    private var root: URL!
    override func setUp() { root = FileManager.default.temporaryDirectory.appendingPathComponent("runstore-\(UUID().uuidString)") }
    override func tearDown() { try? FileManager.default.removeItem(at: root) }

    private func artifact(dirty: Bool = false, quiet: Bool = true, validation: Bool = false) -> BenchArtifact {
        var a = ArtifactSchemaTests.sampleArtifact(validation: validation)
        a.metadata.gitDirty = dirty
        a.metadata.preflight.quiet = quiet
        if !quiet { a.metadata.preflight.reasons = ["drift 9.0% exceeds 5.0%"] }
        return a
    }

    func testRunPathLayoutAndCreateOnly() throws {
        let store = RunStore(root: root)
        let date = Date(timeIntervalSince1970: 1_789_000_000)   // 2026-09-10T00:26:40Z
        let url = try store.writeRun(artifact(), at: date)
        XCTAssertEqual(url.deletingLastPathComponent().lastPathComponent, "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval")
        XCTAssertEqual(url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent, "runs")
        XCTAssertEqual(url.lastPathComponent, "20260910T002640Z_deadbee.json")
        XCTAssertThrowsError(try store.writeRun(artifact(), at: date)) { XCTAssertEqual($0 as? RunStoreError, .runExists(url.path)) }
        let dirtyURL = store.runURL(for: artifact(dirty: true), at: date)
        XCTAssertEqual(dirtyURL.lastPathComponent, "20260910T002640Z_deadbee_dirty.json")
        XCTAssertEqual(try RunStore.load(url), artifact())
    }

    func testCaptureAppendsAndApprovePointer() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 1])
        XCTAssertNil(try store.approvedVersion(environmentKey: key, caseID: id))
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: true, allowDirty: false), [id: 2])
        XCTAssertEqual(try store.approvedVersion(environmentKey: key, caseID: id), 2)
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 3])
        XCTAssertEqual(try store.approvedVersion(environmentKey: key, caseID: id), 2, "capture without --approve never moves the pointer")
        XCTAssertNotNil(try store.baseline(environmentKey: key, caseID: id, version: 1))
        XCTAssertNil(try store.baseline(environmentKey: key, caseID: id, version: 9))
        XCTAssertEqual(try store.approvedBaseline(environmentKey: key, caseID: id), artifact().cases[0])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("baselines/\(key)/\(id)/1.json").path))
    }

    func testCaptureRefusals() {
        let store = RunStore(root: root)
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(validation: true), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedValidationLayers) }
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(quiet: false), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedNotQuiet(["drift 9.0% exceeds 5.0%"])) }
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedDirty) }
        XCTAssertNoThrow(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: true))
        var empty = artifact(); empty.cases[0].status = .unavailable
        XCTAssertThrowsError(try store.captureBaseline(from: empty, approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .noMeasuredCases) }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter RunStoreTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'RunStore' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

public enum RunStoreError: Error, Equatable {
    case runExists(String)
    case refusedValidationLayers
    case refusedNotQuiet([String])
    case refusedDirty
    case noMeasuredCases
    case badApprovedPointer(String)
}

/// Append-only artifact and baseline storage (spec §4.11).
/// Runs: `<root>/runs/<environmentKey>/<yyyyMMdd'T'HHmmss'Z'>_<sha7|nogit>[_dirty].json`.
/// Baselines: `<root>/baselines/<environmentKey>/<caseID>/<n>.json` plus an `approved` pointer file.
public struct RunStore: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    private static let stamp: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"; return f
    }()

    public static func encode(_ value: some Encodable) throws -> Data {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]; return try enc.encode(value)
    }

    public static func load(_ url: URL) throws -> BenchArtifact {
        try JSONDecoder().decode(BenchArtifact.self, from: Data(contentsOf: url))
    }

    public func runURL(for artifact: BenchArtifact, at date: Date) -> URL {
        let sha = artifact.metadata.gitSHA.map { String($0.prefix(7)) } ?? "nogit"
        let name = "\(Self.stamp.string(from: date))_\(sha)\(artifact.metadata.gitDirty ? "_dirty" : "").json"
        return root.appendingPathComponent("runs").appendingPathComponent(artifact.metadata.environmentKey).appendingPathComponent(name)
    }

    /// Create-only: an existing path is an error, never overwritten.
    @discardableResult
    public func writeRun(_ artifact: BenchArtifact, at date: Date = Date()) throws -> URL {
        let url = runURL(for: artifact, at: date)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: url.path) else { throw RunStoreError.runExists(url.path) }
        try Self.encode(artifact).write(to: url, options: [.withoutOverwriting])
        return url
    }

    private func caseDir(_ key: String, _ id: String) -> URL {
        root.appendingPathComponent("baselines").appendingPathComponent(key).appendingPathComponent(id)
    }

    private func versions(in dir: URL) -> [Int] {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .compactMap { $0.hasSuffix(".json") ? Int($0.dropLast(5)) : nil }.sorted()
    }

    /// Writes every measured case as the next version; never overwrites; moves `approved` only when asked.
    public func captureBaseline(from artifact: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int] {
        if artifact.metadata.validationLayers { throw RunStoreError.refusedValidationLayers }
        if !artifact.metadata.preflight.quiet { throw RunStoreError.refusedNotQuiet(artifact.metadata.preflight.reasons) }
        if artifact.metadata.gitDirty && !allowDirty { throw RunStoreError.refusedDirty }
        let measured = artifact.cases.filter { $0.status == .measured }
        guard !measured.isEmpty else { throw RunStoreError.noMeasuredCases }
        var written: [String: Int] = [:]
        for c in measured {
            let dir = caseDir(artifact.metadata.environmentKey, c.id)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let n = (versions(in: dir).last ?? 0) + 1
            try Self.encode(c).write(to: dir.appendingPathComponent("\(n).json"), options: [.withoutOverwriting])
            if approve { try "\(n)\n".write(to: dir.appendingPathComponent("approved"), atomically: true, encoding: .utf8) }
            written[c.id] = n
        }
        return written
    }

    public func baseline(environmentKey: String, caseID: String, version: Int) throws -> CaseRecord? {
        let url = caseDir(environmentKey, caseID).appendingPathComponent("\(version).json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(CaseRecord.self, from: Data(contentsOf: url))
    }

    public func approvedVersion(environmentKey: String, caseID: String) throws -> Int? {
        let url = caseDir(environmentKey, caseID).appendingPathComponent("approved")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let text = try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let n = Int(text) else { throw RunStoreError.badApprovedPointer(text) }
        return n
    }

    public func approvedBaseline(environmentKey: String, caseID: String) throws -> CaseRecord? {
        guard let n = try approvedVersion(environmentKey: environmentKey, caseID: caseID) else { return nil }
        return try baseline(environmentKey: environmentKey, caseID: caseID, version: n)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter RunStoreTests 2>&1 | tail -1`
Expected: `Executed 3 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/RunStore.swift Tests/VectorAccelerateTests/Harness/RunStoreTests.swift
git commit -q -m "feat(kit): append-only run store with versioned baselines and approve pointer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: `Compare`

**Files:**
- Create: `Sources/VectorTestKit/Compare.swift`
- Test: `Tests/VectorAccelerateTests/Harness/CompareTests.swift`

**Interfaces:**
- Produces: `public enum CompareStatus: String, Codable, Sendable { case ok, alertSlower = "alert(slower)", alertFaster = "alert(faster)", incomparable, correctnessFailed, resourceFailed, invalidArtifact }`; `public struct KindDelta: Codable, Sendable, Equatable { kind: TimingKind; newMedian: Double; baseMedian: Double; delta: Double; boundary: Double; widened: Bool; alerting: Bool }`; `public struct CompareResult: Codable, Sendable, Equatable { id: String; status: CompareStatus; deltas: [KindDelta]; reasons: [String] }`; `public enum Compare { static func compare(new: CaseRecord, baseline: CaseRecord?, newKey: String, baselineKey: String?) -> CompareResult; static func exitCode(_ results: [CompareResult], strictPerf: Bool) -> Int32; static func render(_ results: [CompareResult], format: String) -> String }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

/// Decision rule (spec §5.6 as amended): invalidArtifact → correctnessFailed → resourceFailed → incomparable →
/// per-kind deltas with boundary max(2.5%, 2×baseline RSD) → ok.
final class CompareTests: XCTestCase {
    private let key = "k1"

    private func record(_ id: String = "probe.gpu_copy.bytes=8", wall: [Double], correctness: CorrectnessStatus = .pass,
                        resources: Resources? = nil, route: String = "gpu.probe_copy") -> CaseRecord {
        var c = ArtifactSchemaTests.sampleArtifact().cases[0]
        c.id = id; c.identity.id = id; c.identity.route = route
        c.samples = 1; c.iterations = wall.count; c.warmupUsed = [3]
        c.timing = Timing(wall: Series(raw: [wall], summary: Stats.summarize(wall, bootstrapSeed: 1), bias: nil), commandBuffer: nil, encoder: nil)
        c.correctness?.status = correctness
        c.resources = resources
        return c
    }

    func testOk() {
        let r = Compare.compare(new: record(wall: [101, 101, 101]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .ok); XCTAssertEqual(r.deltas.count, 1)
        XCTAssertEqual(r.deltas[0].delta, 0.01, accuracy: 1e-12); XCTAssertEqual(r.deltas[0].boundary, 0.025); XCTAssertFalse(r.deltas[0].widened)
    }

    func testAlertSlowerAndFaster() {
        XCTAssertEqual(Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key).status, .alertSlower)
        XCTAssertEqual(Compare.compare(new: record(wall: [96, 96, 96]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key).status, .alertFaster)
    }

    func testBoundaryWidensWithBaselineRSD() {
        // baseline [90,100,110]: rsd 0.1 → boundary 0.2; +15% is ok and visibly widened.
        let r = Compare.compare(new: record(wall: [115, 115, 115]), baseline: record(wall: [90, 100, 110]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .ok); XCTAssertTrue(r.deltas[0].widened); XCTAssertEqual(r.deltas[0].boundary, 0.2, accuracy: 1e-12)
    }

    func testIncomparableIdentityKeyAndMissing() {
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], route: "cpu"), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .incomparable)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: "k2").status, .incomparable)
        let missing = Compare.compare(new: record(wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        XCTAssertEqual(missing.status, .incomparable); XCTAssertEqual(missing.reasons, ["no baseline for probe.gpu_copy.bytes=8 under k1"])
    }

    func testCorrectnessAndResourceFailuresWinEvenWithoutBaseline() {
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail), baseline: nil, newKey: key, baselineKey: nil).status, .correctnessFailed)
        let over = Resources(staticThreadgroupBytes: 40_000, maxThreadgroupBytes: 32_768, poolHighWaterBytes: nil, notes: [])
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], resources: over), baseline: nil, newKey: key, baselineKey: nil).status, .resourceFailed)
        let noted = Resources(staticThreadgroupBytes: nil, maxThreadgroupBytes: nil, poolHighWaterBytes: nil, notes: ["LIMIT_EXCEEDED: pool"])
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], resources: noted), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .resourceFailed)
        // A correctness failure with a mismatched baseline still reports correctnessFailed and keeps the mismatch reason.
        let both = Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail, route: "cpu"), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
        XCTAssertEqual(both.status, .correctnessFailed); XCTAssertTrue(both.reasons.contains { $0.hasPrefix("identity differs") })
    }

    func testInvalidArtifact() {
        var bad = record(wall: [1, 1, 1]); bad.warmupUsed = []
        XCTAssertEqual(Compare.compare(new: bad, baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .invalidArtifact)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: bad, newKey: key, baselineKey: key).status, .invalidArtifact)
    }

    func testExitCodes() {
        let ok = CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])
        let alert = CompareResult(id: "a.b", status: .alertSlower, deltas: [], reasons: [])
        let fail = CompareResult(id: "a.b", status: .correctnessFailed, deltas: [], reasons: [])
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, fail], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .incomparable, deltas: [], reasons: [])], strictPerf: true), 0)
    }

    func testRenderFormats() {
        let r = Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertTrue(Compare.render([r], format: "text").contains("alert(slower)"))
        XCTAssertTrue(Compare.render([r], format: "markdown").contains("| probe.gpu_copy.bytes=8 |"))
        XCTAssertNoThrow(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([r], format: "json").utf8)))
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter CompareTests 2>&1 | grep -m1 "error:"`
Expected: `cannot find 'Compare' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

public enum CompareStatus: String, Codable, Sendable, Equatable {
    case ok
    case alertSlower = "alert(slower)"
    case alertFaster = "alert(faster)"
    case incomparable
    case correctnessFailed
    case resourceFailed
    case invalidArtifact
}

public struct KindDelta: Codable, Sendable, Equatable {
    public var kind: TimingKind
    public var newMedian: Double
    public var baseMedian: Double
    public var delta: Double
    public var boundary: Double
    public var widened: Bool
    public var alerting: Bool
}

public struct CompareResult: Codable, Sendable, Equatable {
    public var id: String
    public var status: CompareStatus
    public var deltas: [KindDelta]
    public var reasons: [String]
    public init(id: String, status: CompareStatus, deltas: [KindDelta], reasons: [String]) {
        self.id = id; self.status = status; self.deltas = deltas; self.reasons = reasons
    }
}

/// Owner policy (Q5, spec §5.6): alerts in both directions, boundary max(2.5%, 2 × baseline RSD), non-blocking
/// unless strict; correctness, resource, and artifact failures block regardless of baseline availability.
public enum Compare {
    public static let floorBoundary = 0.025

    public static func compare(new: CaseRecord, baseline: CaseRecord?, newKey: String, baselineKey: String?) -> CompareResult {
        var reasons: [String] = []
        var status: CompareStatus?

        // 1. invalid
        do { try new.validate() } catch { reasons.append("new record invalid: \(error)"); status = .invalidArtifact }
        if let b = baseline { do { try b.validate() } catch { reasons.append("baseline record invalid: \(error)"); status = status ?? .invalidArtifact } }
        if status == .invalidArtifact { return CompareResult(id: new.id, status: .invalidArtifact, deltas: [], reasons: reasons) }

        // 2. correctness, 3. resources — recorded regardless of baseline
        if new.correctness?.status == .fail { reasons.append("correctness failed: \(new.correctness?.notes.joined(separator: "; ") ?? "")"); status = status ?? .correctnessFailed }
        if let r = new.resources {
            if let s = r.staticThreadgroupBytes, let m = r.maxThreadgroupBytes, s > m { reasons.append("resource limit exceeded: staticThreadgroupBytes \(s) > \(m)"); status = status ?? .resourceFailed }
            for n in r.notes where n.hasPrefix("LIMIT_EXCEEDED:") { reasons.append("resource limit exceeded: \(n)"); status = status ?? .resourceFailed }
        }

        // 4. comparability
        var comparable = true
        guard let base = baseline, let baseKey = baselineKey else {
            reasons.append("no baseline for \(new.id) under \(newKey)")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        if new.identity != base.identity { reasons.append("identity differs from baseline"); comparable = false }
        if newKey != baseKey { reasons.append("environment key differs: \(newKey) vs \(baseKey)"); comparable = false }
        if !comparable { return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons) }

        // 5. deltas
        var deltas: [KindDelta] = []
        var alert: CompareStatus?
        for kind in TimingKind.allCases {
            guard let n = new.timing[kind], let b = base.timing[kind], b.summary.median > 0 else { continue }
            let delta = (n.summary.median - b.summary.median) / b.summary.median
            let boundary = max(floorBoundary, 2 * b.summary.rsd)
            let alerting = delta > boundary || delta < -boundary
            deltas.append(KindDelta(kind: kind, newMedian: n.summary.median, baseMedian: b.summary.median, delta: delta,
                                    boundary: boundary, widened: boundary > floorBoundary, alerting: alerting))
            if alerting, alert == nil { alert = delta > 0 ? .alertSlower : .alertFaster }
        }
        return CompareResult(id: new.id, status: status ?? alert ?? .ok, deltas: deltas, reasons: reasons)
    }

    public static func exitCode(_ results: [CompareResult], strictPerf: Bool) -> Int32 {
        let hard: Set<CompareStatus> = [.invalidArtifact, .correctnessFailed, .resourceFailed]
        if results.contains(where: { hard.contains($0.status) }) { return 1 }
        if strictPerf, results.contains(where: { $0.status == .alertSlower || $0.status == .alertFaster }) { return 1 }
        return 0
    }

    public static func render(_ results: [CompareResult], format: String) -> String {
        switch format {
        case "json":
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            return String(decoding: (try? enc.encode(results)) ?? Data("[]".utf8), as: UTF8.self)
        case "markdown":
            var out = "| case | status | kind | new median (ns) | base median (ns) | delta | boundary |\n|---|---|---|---|---|---|---|\n"
            for r in results {
                if r.deltas.isEmpty { out += "| \(r.id) | \(r.status.rawValue) | — | — | — | — | — |\n" }
                for d in r.deltas {
                    out += "| \(r.id) | \(r.status.rawValue) | \(d.kind.rawValue) | \(String(format: "%.0f", d.newMedian)) | \(String(format: "%.0f", d.baseMedian)) | \(String(format: "%+.2f%%", d.delta * 100)) | \(String(format: "%.2f%%", d.boundary * 100))\(d.widened ? " (widened)" : "") |\n"
                }
                for reason in r.reasons { out += "\n- \(r.id): \(reason)" }
            }
            return out
        default:
            var out = ""
            for r in results {
                out += "\(r.id.padding(toLength: 44, withPad: " ", startingAt: 0)) \(r.status.rawValue)\n"
                for d in r.deltas {
                    out += "    \(d.kind.rawValue.padding(toLength: 14, withPad: " ", startingAt: 0)) \(String(format: "%+.2f%%", d.delta * 100)) (boundary \(String(format: "%.2f%%", d.boundary * 100))\(d.widened ? ", widened" : ""))\(d.alerting ? "  <-- alert" : "")\n"
                }
                for reason in r.reasons { out += "    reason: \(reason)\n" }
            }
            return out
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter CompareTests 2>&1 | tail -1`
Expected: `Executed 8 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Compare.swift Tests/VectorAccelerateTests/Harness/CompareTests.swift
git commit -q -m "feat(kit): compare with owner boundary rule, statuses, exit codes, renderers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Executable skeleton — arguments, adapter types, sampling loop, `list`, `validate-artifact`

**Files:**
- Create: `Sources/VectorAccelerateBench/CLI/Arguments.swift`, `Runner/BenchCase.swift`, `Runner/Runner.swift`, `Cases/Registry.swift`, `Commands/ListCommand.swift`, `Commands/ValidateCommand.swift`
- Modify: `Sources/VectorAccelerateBench/main.swift`

**Interfaces:**
- Consumes: `CaseID`, `CaseIdentity`, `CaseRecord`, `Timing`, `Series`, `Stats`, `Correctness`, `Resources`, `WarmupPolicy`, `VectorGenerator`, `HostFingerprint`, `GPUInfo`, `RunStore`, `BenchArtifact`.
- Produces (executable-internal): `Arguments { command, subcommand, options: [String: String], flags: Set<String>; func value(_:) -> String?; func has(_:) -> Bool }`; `BenchMode` (`smoke|quick|full` with `samples`, `iterations`); `RunOptions { mode, samples, iterations, seed, validationLayers, requireGPU, probeSlowdown, label }`; `GPUSample { kind: TimingKind; startNs, endNs: Double; label: String }`; `IterationOutput { gpu: [GPUSample]; payload: (any Sendable)? }`; `protocol BenchCase { id: CaseID; identity: CaseIdentity; providedTiming: Set<TimingKind>; unitCount: Int; func prepare(_: BenchContext) async throws -> any PreparedCase }`; `protocol PreparedCase: AnyObject { runOnce() async throws -> IterationOutput; verify(_:) throws -> Correctness; resources() -> Resources?; resetForSample() async throws }`; `final class BenchContext { device: (any MTLDevice)?; host: HostFingerprint; gpu: GPUInfo?; options: RunOptions; func makeGenerator(stream:) -> VectorGenerator }`; `enum Runner { static func run(_ c: any BenchCase, context:, policy:) async -> CaseRecord }`; `enum Registry { static var all: [any BenchCase] }`.

- [ ] **Step 1: `CLI/Arguments.swift`**

```swift
import Foundation

/// Hand-written parser: `<command> [<subcommand>] [--key value | --flag] ...`. No third-party dependency (spec §5.5).
struct Arguments {
    static let booleanFlags: Set<String> = ["--approve", "--allow-dirty", "--strict-perf", "--require-gpu", "--validation-layers", "--help", "-h"]

    let command: String
    let subcommand: String?
    let options: [String: String]
    let flags: Set<String>

    init(_ argv: [String]) {
        var positionals: [String] = [], options: [String: String] = [:], flags: Set<String> = []
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") || a == "-h" {
                if Self.booleanFlags.contains(a) { flags.insert(a); i += 1; continue }
                if i + 1 < argv.count { options[a] = argv[i + 1]; i += 2 } else { options[a] = ""; i += 1 }
            } else { positionals.append(a); i += 1 }
        }
        command = positionals.first ?? "help"
        subcommand = positionals.count > 1 ? positionals[1] : nil
        self.options = options; self.flags = flags
    }

    func value(_ key: String) -> String? { options[key] }
    func has(_ flag: String) -> Bool { flags.contains(flag) }
    func int(_ key: String) -> Int? { options[key].flatMap(Int.init) }
}
```

- [ ] **Step 2: `Runner/BenchCase.swift`**

```swift
import Foundation
import Metal
import VectorTestKit

enum BenchMode: String {
    case smoke, quick, full
    var samples: Int { switch self { case .smoke: return 1; case .quick: return 3; case .full: return 5 } }
    var iterations: Int { switch self { case .smoke: return 3; case .quick: return 20; case .full: return 100 } }
}

struct RunOptions: Sendable {
    var mode: BenchMode
    var samples: Int
    var iterations: Int
    var seed: UInt64
    var validationLayers: Bool
    var requireGPU: Bool
    var probeSlowdown: Int
    var label: String?
}

struct GPUSample: Sendable {
    var kind: TimingKind
    var startNs: Double
    var endNs: Double
    var label: String
    var durationNs: Double { endNs - startNs }
}

struct IterationOutput: @unchecked Sendable {
    var gpu: [GPUSample]
    var payload: Any?
}

/// Adapter seam (spec §5.1 / interface contract §4).
protocol BenchCase: Sendable {
    var id: CaseID { get }
    var identity: CaseIdentity { get }
    var providedTiming: Set<TimingKind> { get }
    var unitCount: Int { get }
    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase
}

protocol PreparedCase: AnyObject {
    func runOnce() async throws -> IterationOutput
    func verify(_ output: IterationOutput) throws -> Correctness
    func resources() -> Resources?
    func resetForSample() async throws
}

final class BenchContext: @unchecked Sendable {
    let device: (any MTLDevice)?
    let host: HostFingerprint
    let gpu: GPUInfo?
    let options: RunOptions

    init(device: (any MTLDevice)?, host: HostFingerprint, gpu: GPUInfo?, options: RunOptions) {
        self.device = device; self.host = host; self.gpu = gpu; self.options = options
    }

    func makeGenerator(stream: UInt64) -> VectorGenerator { VectorGenerator(seed: options.seed, stream: stream) }
}

enum BenchError: Error, CustomStringConvertible {
    case metalUnavailable
    case routeMismatch(expected: String, telemetry: String)
    case usage(String)
    var description: String {
        switch self {
        case .metalUnavailable: return "Metal device unavailable"
        case .routeMismatch(let e, let t): return "route mismatch: expected \(e); telemetry \(t)"
        case .usage(let m): return m
        }
    }
}
```

- [ ] **Step 3: `Runner/Runner.swift` (the sampling loop, spec §5.2–§5.3)**

```swift
import Foundation
import VectorTestKit

enum Runner {
    static let commandBufferBias = "command-buffer duration includes submission and completion overhead (+2–10 µs measured on M3 Max, see research brief Appendix D)"

    static func run(_ c: any BenchCase, context: BenchContext, policy: WarmupPolicy = .standard) async -> CaseRecord {
        let o = context.options
        var record = CaseRecord(id: c.id.rawValue, identity: c.identity, status: .measured, failureReason: nil,
                                samples: o.samples, iterations: o.iterations, warmupUsed: [], unitCount: c.unitCount,
                                timing: Timing(wall: nil, commandBuffer: nil, encoder: nil), correctness: nil, resources: nil, notes: [])
        if o.probeSlowdown > 0 { record.notes.append("probe-slowdown=\(o.probeSlowdown)") }
        let clock = ContinuousClock()
        func ns(_ d: Duration) -> Double { Double(d.components.seconds) * 1e9 + Double(d.components.attoseconds) / 1e9 }

        do {
            let prepared = try await c.prepare(context)
            record.identity = c.identity   // cases whose fixture depends on the run seed finalize identity in prepare()
            var wall: [[Double]] = [], cb: [[Double]] = [], enc: [[Double]] = []
            var lastOutput: IterationOutput?
            for _ in 0..<o.samples {
                try await prepared.resetForSample()
                var warm: [Double] = []
                repeat {
                    let start = clock.now
                    _ = try await prepared.runOnce()
                    warm.append(ns(clock.now - start))
                } while !policy.shouldStop(after: warm)
                record.warmupUsed.append(warm.count)

                var w: [Double] = [], b: [Double] = [], e: [Double] = []
                for _ in 0..<o.iterations {
                    let start = clock.now
                    let out = try await prepared.runOnce()
                    w.append(ns(clock.now - start))
                    for s in out.gpu {
                        switch s.kind { case .commandBuffer: b.append(s.durationNs); case .encoder: e.append(s.durationNs); case .wall: break }
                    }
                    lastOutput = out
                }
                wall.append(w); if !b.isEmpty { cb.append(b) }; if !e.isEmpty { enc.append(e) }
                if let last = lastOutput { record.correctness = try prepared.verify(last) }
            }
            func series(_ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                guard raw.count == o.samples, raw.allSatisfy({ $0.count == o.iterations }) else { return nil }
                return Series(raw: raw, summary: Stats.summarize(raw.flatMap { $0 }, bootstrapSeed: o.seed &+ stream), bias: bias)
            }
            record.timing.wall = series(wall, bias: nil, stream: 1)
            if c.providedTiming.contains(.commandBuffer) { record.timing.commandBuffer = series(cb, bias: commandBufferBias, stream: 2) }
            if c.providedTiming.contains(.encoder) { record.timing.encoder = series(enc, bias: nil, stream: 3) }
            record.resources = prepared.resources()
        } catch BenchError.metalUnavailable {
            record.status = .unavailable; record.failureReason = "Metal device unavailable"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
        } catch {
            record.status = .failed; record.failureReason = "\(error)"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        }
        return record
    }
}
```

- [ ] **Step 4: `Cases/Registry.swift`, `Commands/ListCommand.swift`, `Commands/ValidateCommand.swift`**

```swift
// Cases/Registry.swift
import Foundation
import VectorTestKit

/// One line per family (spec §5.4). Families never import each other.
enum Registry {
    static var all: [any BenchCase] {
        var cases: [any BenchCase] = []
        // families are appended in Tasks 14 and 15
        return cases
    }

    /// Glob filtering with fnmatch; empty include means everything.
    static func select(include: [String], exclude: [String]) throws -> [any BenchCase] {
        let all = Self.all
        let dups = CaseRegistryCheck.duplicates(all.map(\.id))
        guard dups.isEmpty else { throw BenchError.usage("duplicate case IDs in registry: \(dups.map(\.rawValue))") }
        func matches(_ id: String, _ globs: [String]) -> Bool { globs.contains { fnmatch($0, id, 0) == 0 } }
        return all.filter { (include.isEmpty || matches($0.id.rawValue, include)) && !matches($0.id.rawValue, exclude) }
    }
}
```

```swift
// Commands/ListCommand.swift
import Foundation

enum ListCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        let cases = try Registry.select(include: args.value("--filter").map { [$0] } ?? [], exclude: args.value("--exclude").map { [$0] } ?? [])
        for c in cases { print("\(c.id.rawValue)  route=\(c.identity.route)  timing=\(c.providedTiming.map(\.rawValue).sorted().joined(separator: ","))") }
        print("\(cases.count) case(s)")
        return 0
    }
}
```

```swift
// Commands/ValidateCommand.swift
import Foundation
import VectorTestKit

enum ValidateCommand {
    static func run(_ args: Arguments) -> Int32 {
        guard let path = args.subcommand else { print("usage: validate-artifact <path>"); return 2 }
        do {
            let a = try RunStore.load(URL(fileURLWithPath: path))
            try a.validate()
            print("valid: schemaVersion \(a.schemaVersion), \(a.cases.count) case(s), key \(a.metadata.environmentKey)")
            return 0
        } catch {
            print("invalid: \(error)")
            return 1
        }
    }
}
```

- [ ] **Step 5: `main.swift` dispatch**

```swift
import Foundation

let usage = """
VectorAccelerateBench <command>
  list [--filter <glob>] [--exclude <glob>]
  run  [--filter <glob>] [--exclude <glob>] [--mode smoke|quick|full] [--samples N] [--iterations N] [--seed S]
       [--out <path>] [--label <text>] [--require-gpu] [--validation-layers] [--allow-dirty] [--probe-slowdown N]
  baseline capture --from <artifact> [--approve] [--allow-dirty]
  compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
  validate-artifact <path>
"""

let args = Arguments(Array(CommandLine.arguments.dropFirst()))
var status: Int32 = 2
do {
    switch args.command {
    case "list": status = try ListCommand.run(args)
    case "validate-artifact": status = ValidateCommand.run(args)
    case "run": status = try await RunCommand.run(args)              // Task 15
    case "baseline": status = try BaselineCommand.run(args)           // Task 16
    case "compare": status = try CompareCommand.run(args)             // Task 16
    default: print(usage); status = args.command == "help" ? 0 : 2
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    status = 1
}
exit(status)
```

Until Tasks 15–16 exist, add these three stubs at the bottom of `main.swift` so the executable compiles; each is deleted when its real file lands:

```swift
enum RunCommand { static func run(_ args: Arguments) async throws -> Int32 { print("run: not implemented until Task 15"); return 2 } }
enum BaselineCommand { static func run(_ args: Arguments) throws -> Int32 { print("baseline: not implemented until Task 16"); return 2 } }
enum CompareCommand { static func run(_ args: Arguments) throws -> Int32 { print("compare: not implemented until Task 16"); return 2 } }
```

- [ ] **Step 6: Build and exercise**

```bash
swift build 2>&1 | tail -2                                              # Build complete!
swift run -q VectorAccelerateBench list; echo "exit=$?"                  # "0 case(s)", exit=0
swift run -q VectorAccelerateBench nonsense; echo "exit=$?"              # usage, exit=2
swift run -q VectorAccelerateBench validate-artifact /nonexistent; echo "exit=$?"   # "invalid: ...", exit=1
```

- [ ] **Step 7: Commit**

```bash
git add Sources/VectorAccelerateBench
git commit -q -m "feat(bench): CLI skeleton, adapter protocols, sampling loop, list and validate-artifact

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Metal fingerprint, GPU probe, and the probe case

**Files:**
- Create: `Sources/VectorAccelerateBench/Fingerprint/MetalFingerprint.swift`, `Probe/GPUProbe.swift`, `Cases/Probe/ProbeCases.swift`
- Modify: `Sources/VectorAccelerateBench/Cases/Registry.swift`

**Interfaces:**
- Consumes: `GPUInfo`, `Subprocess`, `BenchCase`, `PreparedCase`, `BenchContext`, `GPUSample`, `Correctness`, `Resources`, `ContentDigest`, `VectorGenerator.version`.
- Produces: `enum MetalFingerprint { static func gather(device: (any MTLDevice)?) -> GPUInfo? ; static func gpuCoreCount() -> Int?; static func metalToolchainVersion() -> String? }`; `final class GPUProbe { init(device:) throws; static let bytes = 134_217_728; func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double); func copyGBps(warmup: Int = 5, measured: Int = 10) throws -> Double; func outputMatchesInput() -> Bool; var staticThreadgroupBytes: Int; var maxThreadgroupBytes: Int }`; `enum ProbeCases { static var all: [any BenchCase] }`.

- [ ] **Step 1: `Fingerprint/MetalFingerprint.swift`**

```swift
import Foundation
import Metal
import IOKit
import VectorTestKit

enum MetalFingerprint {
    static func gather(device: (any MTLDevice)?) -> GPUInfo? {
        guard let device else { return nil }
        return GPUInfo(name: device.name, coreCount: gpuCoreCount(), unifiedMemory: device.hasUnifiedMemory,
                       maxThreadgroupBytes: device.maxThreadgroupMemoryLength, metal4: device.supportsFamily(.metal4),
                       metalToolchain: metalToolchainVersion())
    }

    /// `gpu-core-count` on the first AGXAccelerator registry entry; nil when absent (never guessed).
    static func gpuCoreCount() -> Int? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }
        guard let value = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() else { return nil }
        return (value as? NSNumber)?.intValue
    }

    /// First line of `xcrun -sdk macosx metal --version`, e.g. "Apple metal version 32023.883 (metalfe-32023.883)".
    static func metalToolchainVersion() -> String? {
        guard let out = Subprocess.run("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "--version"], timeout: 10),
              let line = out.split(separator: "\n").first,
              let m = line.range(of: #"version (\S+)"#, options: .regularExpression) else { return nil }
        return String(line[m]).replacingOccurrences(of: "version ", with: "")
    }
}
```

- [ ] **Step 2: `Probe/GPUProbe.swift`** (the spike's kernel, with a `loops` parameter that adds extra reads without changing the output, so a slowed run still verifies)

```swift
import Foundation
import Metal
import VectorTestKit

/// 128 MiB float4 copy on a legacy command queue; `.commandBuffer` timing from gpuStartTime/gpuEndTime.
/// Also the preflight GPU calibration and the M0 deliberate-slowdown vehicle (spec §5.7, §5.8).
final class GPUProbe: @unchecked Sendable {
    static let bytes = 134_217_728
    static let floatCount = bytes / 4
    static let float4Count = bytes / 16

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct ProbeParams { uint n4; uint loops; float zero; };
    kernel void copy_f32x4(device const float4* src [[buffer(0)]],
                           device float4* dst [[buffer(1)]],
                           constant ProbeParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= p.n4) { return; }
        float4 v = src[gid];
        float4 sink = float4(0.0f);
        for (uint i = 1; i < p.loops; ++i) { sink += src[(gid + i * 4099u) % p.n4]; }
        dst[gid] = v + sink * p.zero;   // p.zero is 0.0f from the CPU; the compiler cannot fold it
    }
    """

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    private let src: any MTLBuffer
    private let dst: any MTLBuffer
    private let params: any MTLBuffer

    var staticThreadgroupBytes: Int { pipeline.staticThreadgroupMemoryLength }
    var maxThreadgroupBytes: Int { device.maxThreadgroupMemoryLength }

    init(device: any MTLDevice) throws {
        self.device = device
        let library = try device.makeLibrary(source: Self.source, options: MTLCompileOptions())
        guard let function = library.makeFunction(name: "copy_f32x4") else { throw BenchError.usage("probe kernel missing") }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let src = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let dst = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let params = device.makeBuffer(length: 16, options: .storageModeShared) else { throw BenchError.metalUnavailable }
        self.queue = queue; self.src = src; self.dst = dst; self.params = params
        src.label = "probe.src"; dst.label = "probe.dst"; params.label = "probe.params"
        let p = src.contents().bindMemory(to: Float32.self, capacity: Self.floatCount)
        for i in 0..<Self.floatCount { p[i] = Float32(i & 0xFFFF) * 0.5 + 1.0 }
        memset(dst.contents(), 0, Self.bytes)
    }

    /// One command buffer, one dispatch, synchronous wait. Returns the buffer's GPU start/end in host seconds.
    func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double) {
        struct ProbeParams { var n4: UInt32; var loops: UInt32; var zero: Float }
        var pp = ProbeParams(n4: UInt32(Self.float4Count), loops: UInt32(max(loops, 1)), zero: 0)
        memcpy(params.contents(), &pp, MemoryLayout<ProbeParams>.size)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw BenchError.metalUnavailable }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1); enc.setBuffer(params, offset: 0, index: 2)
        let tpg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: Self.float4Count, height: 1, depth: 1), threadsPerThreadgroup: tpg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        return (cb.gpuStartTime, cb.gpuEndTime)
    }

    func copyGBps(warmup: Int = 5, measured: Int = 10) throws -> Double {
        for _ in 0..<warmup { _ = try dispatch(loops: 1) }
        var secs: [Double] = []
        for _ in 0..<measured { let t = try dispatch(loops: 1); secs.append(t.endSeconds - t.startSeconds) }
        let median = Stats.percentile(sorted: secs.sorted(), p: 0.5)
        return median > 0 ? Double(Self.bytes) / median / 1e9 : 0
    }

    func outputMatchesInput() -> Bool { memcmp(src.contents(), dst.contents(), Self.bytes) == 0 }
}
```

- [ ] **Step 3: `Cases/Probe/ProbeCases.swift`**

```swift
import Foundation
import Metal
import VectorTestKit

struct GPUCopyProbeCase: BenchCase {
    let id = try! CaseID(validating: "probe.gpu_copy.bytes=\(GPUProbe.bytes)")
    var identity: CaseIdentity {
        CaseIdentity(id: id.rawValue,
                     fixtureDigest: ContentDigest.sha256([.int(GPUProbe.bytes), .string("pattern:(i&0xFFFF)*0.5+1")]),
                     generatorVersion: VectorGenerator.version, seed: 0, stream: 0,
                     route: "gpu.probe_copy", variant: nil, dataDescriptor: "sequential_u32_pattern")
    }
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = GPUProbe.bytes

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        guard let device = ctx.device else { throw BenchError.metalUnavailable }
        return PreparedProbe(probe: try GPUProbe(device: device), loops: max(1, ctx.options.probeSlowdown))
    }

    final class PreparedProbe: PreparedCase {
        let probe: GPUProbe; let loops: Int
        init(probe: GPUProbe, loops: Int) { self.probe = probe; self.loops = loops }
        func runOnce() async throws -> IterationOutput {
            let t = try probe.dispatch(loops: loops)
            return IterationOutput(gpu: [GPUSample(kind: .commandBuffer, startNs: t.startSeconds * 1e9, endNs: t.endSeconds * 1e9, label: "copy_f32x4")], payload: nil)
        }
        func verify(_ output: IterationOutput) throws -> Correctness {
            let ok = probe.outputMatchesInput()
            return Correctness(status: ok ? .pass : .fail, comparator: "memcmp", tolerance: nil, maxAbsError: nil, maxRelError: nil,
                               recall: nil, recallPolicy: nil, notes: ok ? [] : ["dst != src after copy"])
        }
        func resources() -> Resources? {
            Resources(staticThreadgroupBytes: probe.staticThreadgroupBytes, maxThreadgroupBytes: probe.maxThreadgroupBytes, poolHighWaterBytes: nil, notes: [])
        }
        func resetForSample() async throws {}
    }
}

enum ProbeCases {
    static var all: [any BenchCase] { [GPUCopyProbeCase()] }
}
```

- [ ] **Step 4: Register**

In `Cases/Registry.swift`, replace the comment line `// families are appended in Tasks 14 and 15` with:

```swift
        cases += ProbeCases.all
```

- [ ] **Step 5: Build and list**

Run: `swift build 2>&1 | tail -1 && swift run -q VectorAccelerateBench list`
Expected: one line `probe.gpu_copy.bytes=134217728  route=gpu.probe_copy  timing=commandBuffer,wall` then `1 case(s)`.

- [ ] **Step 6: Commit**

```bash
git add Sources/VectorAccelerateBench
git commit -q -m "feat(bench): Metal fingerprint, GPU copy probe, probe case

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: The distance case and the `run` command

**Files:**
- Create: `Sources/VectorAccelerateBench/Cases/Distance/DistanceCases.swift`, `Commands/RunCommand.swift`
- Modify: `Cases/Registry.swift`, `main.swift` (remove the `RunCommand` stub)

**Interfaces:**
- Consumes: `MetalComputeProvider(context:configuration:)`, `MetalComputeProvider.Configuration(preferGPU:fallbackToCPU:)`, `batchDistance(query:candidates:metric: .euclidean)`, `routingTelemetry().gpuKernel`, `Metal4Context()`, `Metal4Context.lastGPUTiming` (`GPUTimingInfo.gpuStartTime/gpuEndTime`, seconds), `DynamicVector([Float])` from VectorCore, `Tolerance.forSum`, `Comparator`, `PreflightProbe`, `HostFingerprint`, `MetalFingerprint`, `GPUProbe`, `RunStore`, `Runner`, `Registry`.
- Produces: `enum DistanceCases { static var all: [any BenchCase] }`; `enum RunCommand { static func run(_ args: Arguments) async throws -> Int32 }`; `enum BenchVersion { static let package = "0.7.0" }`.

- [ ] **Step 1: `Cases/Distance/DistanceCases.swift`**

```swift
import Foundation
import Metal
import VectorCore
import VectorAccelerate
import VectorTestKit

/// Batch L2 distance through the public façade at N=10,000, D=768 (spec §5.7). Route is verified per call via
/// routing telemetry; `.commandBuffer` timing is the context's last GPU timing read immediately after the single submission.
final class L2BatchDistanceCase: BenchCase, @unchecked Sendable {
    static let n = 10_000, d = 768
    static let candidateStream: UInt64 = 10, queryStream: UInt64 = 11

    let id = try! CaseID(validating: "distance.l2.batch.n=\(L2BatchDistanceCase.n).d=\(L2BatchDistanceCase.d)")
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = L2BatchDistanceCase.n

    /// Finalized in `prepare` from the run seed; before that it describes the default seed 42 fixture.
    private var cachedIdentity: CaseIdentity?
    var identity: CaseIdentity { cachedIdentity ?? Self.makeIdentity(seed: 42, fixture: Self.fixture(seed: 42)) }

    static func fixture(seed: UInt64) -> (candidates: [[Float]], query: [Float]) {
        var g = VectorGenerator(seed: seed, stream: candidateStream)
        let candidates = g.uniformVectors(count: n, dimension: d)
        var q = VectorGenerator(seed: seed, stream: queryStream)
        let query = q.uniformVectors(count: 1, dimension: d)[0]
        return (candidates, query)
    }

    static func makeIdentity(seed: UInt64, fixture f: (candidates: [[Float]], query: [Float])) -> CaseIdentity {
        CaseIdentity(id: "distance.l2.batch.n=\(n).d=\(d)",
                     fixtureDigest: ContentDigest.sha256([.int(n), .int(d), .floats(f.candidates.flatMap { $0 }), .floats(f.query)]),
                     generatorVersion: VectorGenerator.version, seed: seed, stream: candidateStream,
                     route: "gpu.l2.batch", variant: nil, dataDescriptor: "uniform[-1,1]")
    }

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        guard ctx.device != nil else { throw BenchError.metalUnavailable }
        let f = Self.fixture(seed: ctx.options.seed)
        cachedIdentity = Self.makeIdentity(seed: ctx.options.seed, fixture: f)
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(preferGPU: true, fallbackToCPU: false))
        return Prepared(context: context, provider: provider, query: DynamicVector(f.query), candidates: f.candidates.map { DynamicVector($0) },
                        oracleSquared: f.candidates.map { c in
                            var s = 0.0; for i in 0..<Self.d { let x = Double(f.query[i]) - Double(c[i]); s += x * x }; return s })
    }

    final class Prepared: PreparedCase {
        let context: Metal4Context; let provider: MetalComputeProvider
        let query: DynamicVector; let candidates: [DynamicVector]; let oracleSquared: [Double]
        // Reduction order inside the kernel is not documented, so the conservative sequential bound applies until SP2 derives
        // per-kernel tolerances. Components are in [-1,1] ⇒ each squared difference ≤ 4 ⇒ Σ|xᵢ| ≤ 4·D.
        let tolerance = Tolerance.forSum(terms: L2BatchDistanceCase.d, reduction: .sequential, magnitude: 4 * Double(L2BatchDistanceCase.d))

        init(context: Metal4Context, provider: MetalComputeProvider, query: DynamicVector, candidates: [DynamicVector], oracleSquared: [Double]) {
            self.context = context; self.provider = provider; self.query = query; self.candidates = candidates; self.oracleSquared = oracleSquared
        }

        func runOnce() async throws -> IterationOutput {
            let before = await provider.routingTelemetry().gpuKernel
            let distances = try await provider.batchDistance(query: query, candidates: candidates, metric: .euclidean)
            let telemetry = await provider.routingTelemetry()
            guard telemetry.gpuKernel == before + 1 else { throw BenchError.routeMismatch(expected: "gpu.l2.batch", telemetry: "\(telemetry)") }
            var gpu: [GPUSample] = []
            if let t = await context.lastGPUTiming {
                gpu.append(GPUSample(kind: .commandBuffer, startNs: t.gpuStartTime * 1e9, endNs: t.gpuEndTime * 1e9, label: "l2_batch"))
            }
            blackHole(distances)
            return IterationOutput(gpu: gpu, payload: distances)
        }

        func verify(_ output: IterationOutput) throws -> Correctness {
            guard let distances = output.payload as? [Float], distances.count == oracleSquared.count else {
                return Correctness(status: .fail, comparator: "squared-distance vs Double oracle", tolerance: tolerance, maxAbsError: nil, maxRelError: nil, recall: nil, recallPolicy: nil, notes: ["result count mismatch"])
            }
            var maxAbs = 0.0, maxRel = 0.0, failures = 0
            for (i, dist) in distances.enumerated() {
                let got = Double(dist) * Double(dist)
                switch Comparator.compare(got, oracleSquared[i], tolerance: tolerance) {
                case .equal, .withinTolerance:
                    let abs = Swift.abs(got - oracleSquared[i]); maxAbs = max(maxAbs, abs); maxRel = max(maxRel, oracleSquared[i] > 0 ? abs / oracleSquared[i] : 0)
                case .classMismatch: failures += 1
                case .outOfTolerance(let a, let r): failures += 1; maxAbs = max(maxAbs, a); maxRel = max(maxRel, r)
                }
            }
            return Correctness(status: failures == 0 ? .pass : .fail, comparator: "squared-distance vs Double oracle", tolerance: tolerance,
                               maxAbsError: maxAbs, maxRelError: maxRel, recall: nil, recallPolicy: nil,
                               notes: failures == 0 ? ["sequential fp32 sum bound, D=768, |x|≤1"] : ["\(failures) of \(distances.count) distances outside tolerance"])
        }

        func resources() -> Resources? { nil }
        func resetForSample() async throws { await provider.resetRoutingTelemetry() }
    }
}

enum DistanceCases {
    static var all: [any BenchCase] { [L2BatchDistanceCase()] }
}
```

- [ ] **Step 2: Register**

In `Cases/Registry.swift`, after `cases += ProbeCases.all` add:

```swift
        cases += DistanceCases.all
```

- [ ] **Step 3: `Commands/RunCommand.swift`**

```swift
import Foundation
import Metal
import VectorTestKit

enum BenchVersion { static let package = "0.7.0" }   // bump with Package.swift's version-history header

enum RunCommand {
    static func run(_ args: Arguments) async throws -> Int32 {
        let mode = BenchMode(rawValue: args.value("--mode") ?? "quick") ?? .quick
        let options = RunOptions(
            mode: mode, samples: args.int("--samples") ?? mode.samples, iterations: args.int("--iterations") ?? mode.iterations,
            seed: args.value("--seed").flatMap(UInt64.init) ?? 42, validationLayers: args.has("--validation-layers"),
            requireGPU: args.has("--require-gpu"), probeSlowdown: args.int("--probe-slowdown") ?? 0, label: args.value("--label"))
        let include = args.value("--filter").map { [$0] } ?? [], exclude = args.value("--exclude").map { [$0] } ?? []
        let cases = try Registry.select(include: include, exclude: exclude)
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let device = MTLCreateSystemDefaultDevice()
        let host = HostFingerprint.gather(packageRoot: root)
        let gpu = MetalFingerprint.gather(device: device)
        let context = BenchContext(device: device, host: host, gpu: gpu, options: options)

        let thermalStart = HostFingerprint.thermalStateName()
        let cpuStart = PreflightProbe.cpuCopyGBps()
        let probe = try device.map { try GPUProbe(device: $0) }
        let gpuStart = try probe?.copyGBps()
        print("preflight: cpu \(String(format: "%.1f", cpuStart)) GB/s" + (gpuStart.map { String(format: ", gpu %.1f GB/s", $0) } ?? ", gpu n/a"))

        var records: [CaseRecord] = []
        for c in cases {
            print("running \(c.id.rawValue) (\(options.samples)×\(options.iterations)) ...")
            let r = await Runner.run(c, context: context)
            print("  \(r.status.rawValue)" + (r.timing.wall.map { String(format: "  wall median %.0f ns", $0.summary.median) } ?? "") + (r.failureReason.map { "  \($0)" } ?? ""))
            records.append(r)
        }

        let cpuEnd = PreflightProbe.cpuCopyGBps()
        let gpuEnd = try probe?.copyGBps()
        let thermalEnd = HostFingerprint.thermalStateName()
        let preflight = PreflightProbe.evaluate(
            cpu: StartEnd(start: cpuStart, end: cpuEnd),
            gpu: gpuStart.flatMap { s in gpuEnd.map { StartEnd(start: s, end: $0) } },
            thermalStart: thermalStart, thermalEnd: thermalEnd, gpuDeviceExists: device != nil)

        let iso = ISO8601DateFormatter()
        let metadata = Metadata(
            packageVersion: BenchVersion.package, gitSHA: host.gitSHA, gitDirty: host.gitDirty, date: iso.string(from: Date()),
            os: host.osVersion, osBuild: host.osBuild, arch: host.arch, cpuCores: host.cpuCores, performanceCores: host.performanceCores,
            efficiencyCores: host.efficiencyCores, memoryBytes: host.memoryBytes, deviceModel: host.chip, modelIdentifier: host.modelIdentifier,
            hostname: host.hostname, swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
            deviceTag: HostFingerprint.deviceTag(arch: host.arch, osVersion: host.osVersion, hostname: host.hostname),
            environmentKey: EnvironmentKey.make(chip: host.chip, gpuCores: gpu?.coreCount, osBuild: host.osBuild, metalToolchain: gpu?.metalToolchain,
                                                swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration, validationLayers: options.validationLayers),
            gpu: gpu, power: host.power, thermal: ThermalInfo(start: thermalStart, end: thermalEnd), preflight: preflight,
            validationLayers: options.validationLayers, runSeed: options.seed, runLabel: options.label, mode: mode.rawValue,
            filters: (include.isEmpty && exclude.isEmpty) ? nil : Filters(include: include.isEmpty ? nil : include, exclude: exclude.isEmpty ? nil : exclude),
            flags: [:])
        let artifact = BenchArtifact(package: "VectorAccelerate", metadata: metadata, cases: records)
        try artifact.validate()

        let url: URL
        if let out = args.value("--out") {
            url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try RunStore.encode(artifact).write(to: url, options: [.withoutOverwriting])
        } else {
            url = try RunStore(root: root.appendingPathComponent(".bench")).writeRun(artifact)
        }
        print("artifact: \(url.path)")
        print("key: \(metadata.environmentKey)  quiet: \(preflight.quiet)" + (preflight.reasons.isEmpty ? "" : "  reasons: \(preflight.reasons)"))

        if records.contains(where: { $0.status == .failed }) { return 1 }
        if options.requireGPU, records.contains(where: { $0.status == .unavailable }) { return 1 }
        return 0
    }
}
```

Remove the `enum RunCommand { ... }` stub from `main.swift`.

- [ ] **Step 4: Smoke run and validate**

```bash
swift run -c release VectorAccelerateBench run --mode smoke 2>&1 | tail -6
```
Expected: both cases `measured`, an `artifact: .../.bench/runs/<key>/<stamp>_<sha>[_dirty].json` line, `quiet: true` on an idle machine.

```bash
A=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench validate-artifact "$A"; echo "exit=$?"
python3 -c "import json,sys; a=json.load(open('$A')); print([ (c['id'], c['status'], sorted(k for k,v in c['timing'].items() if v)) for c in a['cases']])"
```
Expected: `valid: schemaVersion 1, 2 case(s), ...`, exit=0, and both cases listing `['commandBuffer', 'wall']`.

If the distance case reports `failed` with `route mismatch`, the decision engine voted CPU at this shape; that is a real finding to record in the ledger, not something to paper over by changing the tolerance or the case.

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorAccelerateBench
git commit -q -m "feat(bench): batch L2 distance case and run command producing schema-valid artifacts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: `baseline capture` and `compare` commands

**Files:**
- Create: `Sources/VectorAccelerateBench/Commands/BaselineCommand.swift`, `Commands/CompareCommand.swift`
- Modify: `main.swift` (remove the two stubs)

**Interfaces:**
- Consumes: `RunStore`, `Compare`, `CompareResult`.
- Produces: `enum BaselineCommand { static func run(_:) throws -> Int32 }`, `enum CompareCommand { static func run(_:) throws -> Int32 }`.

- [ ] **Step 1: `Commands/BaselineCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum BaselineCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        guard args.subcommand == "capture", let from = args.value("--from") else { print("usage: baseline capture --from <artifact> [--approve] [--allow-dirty]"); return 2 }
        let store = RunStore(root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".bench"))
        let artifact = try RunStore.load(URL(fileURLWithPath: from))
        try artifact.validate()
        do {
            let written = try store.captureBaseline(from: artifact, approve: args.has("--approve"), allowDirty: args.has("--allow-dirty"))
            for (id, n) in written.sorted(by: { $0.key < $1.key }) { print("captured \(id) -> version \(n)\(args.has("--approve") ? " (approved)" : "")") }
            return 0
        } catch let e as RunStoreError {
            print("refused: \(e)")
            return 1
        }
    }
}
```

- [ ] **Step 2: `Commands/CompareCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum CompareCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        guard let path = args.value("--artifact") else { print("usage: compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]"); return 2 }
        let store = RunStore(root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".bench"))
        let artifact = try RunStore.load(URL(fileURLWithPath: path))
        let against = args.value("--against") ?? "approved"
        let key = artifact.metadata.environmentKey
        var results: [CompareResult] = []
        for c in artifact.cases {
            let baseline: CaseRecord?
            if against == "approved" { baseline = try store.approvedBaseline(environmentKey: key, caseID: c.id) }
            else if let n = Int(against) { baseline = try store.baseline(environmentKey: key, caseID: c.id, version: n) }
            else { throw BenchError.usage("--against must be 'approved' or an integer version") }
            results.append(Compare.compare(new: c, baseline: baseline, newKey: key, baselineKey: baseline == nil ? nil : key))
        }
        print(Compare.render(results, format: args.value("--format") ?? "text"))
        if !artifact.metadata.preflight.quiet { print("warning: artifact was not quiet: \(artifact.metadata.preflight.reasons)") }
        return Compare.exitCode(results, strictPerf: args.has("--strict-perf"))
    }
}
```

Remove the `BaselineCommand` and `CompareCommand` stubs from `main.swift`.

- [ ] **Step 3: Exercise the loop on the smoke artifact**

```bash
A=$(ls -t .bench/runs/*/*.json | head -1)
swift run -q -c release VectorAccelerateBench compare --artifact "$A"; echo "exit=$?"      # both incomparable (no baseline), exit=0
swift run -q -c release VectorAccelerateBench baseline capture --from "$A" --approve --allow-dirty; echo "exit=$?"   # two "captured ... version 1 (approved)", exit=0
swift run -q -c release VectorAccelerateBench compare --artifact "$A"; echo "exit=$?"      # both ok (identical artifact), exit=0
```

- [ ] **Step 4: Commit**

```bash
git add Sources/VectorAccelerateBench
git commit -q -m "feat(bench): baseline capture and compare commands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: Deletions, orphan guard, Metal-skip helper, README, CHANGELOG, spec amendment

**Files:**
- Delete: `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`, `Sources/VectorAccelerate/Benchmarking/IndexBenchmarkHarness.swift`, `Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift`, `Tests/VectorAccelerateTests/PerformanceBenchmarks.swift`, `Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift`
- Modify: `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift`, `Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift`, `Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift`, `README.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md`
- Create: `Tests/VectorAccelerateTests/Utilities/RequireMetal.swift`, `Tests/VectorAccelerateTests/Harness/OrphanPublicTypeTests.swift`, `Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt`

- [ ] **Step 1: Delete the dead code**

```bash
git rm -q Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift Sources/VectorAccelerate/Benchmarking/IndexBenchmarkHarness.swift \
          Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift \
          Tests/VectorAccelerateTests/PerformanceBenchmarks.swift Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift
rmdir Sources/VectorAccelerate/Benchmarking 2>/dev/null; true
```

- [ ] **Step 2: Remove `logDecisions`**

In `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift` delete these three fragments: the property (`/// Log acceleration decisions for debugging.` + `public var logDecisions: Bool`), the init parameter line `logDecisions: Bool = false` (and the trailing comma on the preceding `enableProfiling: Bool = false` line), and the assignment `self.logDecisions = logDecisions`. Then:

```bash
grep -rn "logDecisions" Sources Tests   # expected: no output
```

- [ ] **Step 3: Trim the recall validation test and replace the private RNGs**

In `Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift`: delete the whole `func testBenchmarkHarnessRealisticIVF()` (from its doc comment through its closing brace; it is the only harness user), delete the trailing `// MARK: - Seeded RNG for Reproducibility` section with `private struct SeededRNG`, and add `import VectorTestKit` after `import VectorCore`. The remaining `SeededRNG(seed: 42)` calls now resolve to the kit type (LCG instead of xorshift; the tests assert recall properties, not sequences).

In `Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift`: delete the `private struct SeededRNG` block and add `import VectorTestKit` after its last import.

```bash
grep -rn "private struct SeededRNG\|IndexBenchmarkHarness\|IndexBenchmarkConfiguration" Tests Sources   # expected: no output
```

- [ ] **Step 4: `Utilities/RequireMetal.swift`**

```swift
import XCTest
import Metal

/// Single skip point for Metal-dependent tests (spec §3). Adopt file by file; the 149 legacy XCTSkip sites
/// migrate opportunistically, not in SP0.
func requireMetalDevice(file: StaticString = #filePath, line: UInt = #line) throws -> any MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal device not available", file: file, line: line) }
    return device
}
```

- [ ] **Step 5: Orphan guard with a seeded allowlist**

`Tests/VectorAccelerateTests/Harness/OrphanPublicTypeTests.swift`:

```swift
import XCTest

/// Ratchet against unreferenced public types (spec §10). A public type declared in Sources/VectorAccelerate that is named
/// nowhere outside its own file (Sources + Tests) fails unless listed in OrphanAllowlist.txt with a justification.
final class OrphanPublicTypeTests: XCTestCase {
    func testNoNewOrphanPublicTypes() throws {
        let decl = try NSRegularExpression(pattern: #"^\s*(?:@[\w()., ]+\s+)*public\s+(?:final\s+)?(?:struct|class|enum|actor|protocol|typealias)\s+([A-Z][A-Za-z0-9_]*)"#, options: [.anchorsMatchLines])
        let sources = RepoPaths.swiftFiles(under: "Sources/VectorAccelerate")
        let all = sources + RepoPaths.swiftFiles(under: "Sources/VectorAccelerateBench") + RepoPaths.swiftFiles(under: "Tests/VectorAccelerateTests")
        let texts = try all.map { ($0, try String(contentsOf: $0, encoding: .utf8)) }
        let allowlistURL = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt")
        let allowed = Set(((try? String(contentsOf: allowlistURL, encoding: .utf8)) ?? "").split(separator: "\n")
            .map { $0.split(separator: "#")[0].trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })

        var orphans: [String] = []
        for (file, text) in texts where file.path.contains("/Sources/VectorAccelerate/") {
            let range = NSRange(text.startIndex..., in: text)
            for m in decl.matches(in: text, range: range) {
                let name = String(text[Range(m.range(at: 1), in: text)!])
                if allowed.contains(name) { continue }
                let pattern = try NSRegularExpression(pattern: "\\b\(NSRegularExpression.escapedPattern(for: name))\\b")
                let referenced = texts.contains { (f, t) in f != file && pattern.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)) != nil }
                if !referenced { orphans.append("\(name) (\(file.lastPathComponent))") }
            }
        }
        XCTAssertTrue(orphans.isEmpty, "Unreferenced public types (add to OrphanAllowlist.txt with a justification if they are intended API):\n" + orphans.sorted().joined(separator: "\n"))
    }
}
```

Seed the allowlist: run the test once, and for every name it reports decide whether it is intended public API. Each intended one goes into `Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt` as `Name  # one-line justification`; anything that is not intended API is deleted from `Sources` in this task instead. The ratchet then fails only on new orphans.

```bash
swift test --filter OrphanPublicTypeTests 2>&1 | grep -A200 "Unreferenced public types" | head -80   # review this list
# ... edit OrphanAllowlist.txt / delete genuine dead types ...
swift test --filter OrphanPublicTypeTests 2>&1 | tail -1   # expected: 0 failures
```

- [ ] **Step 6: README and CHANGELOG**

`README.md`:
- Line 5: replace `VectorAccelerate delivers up to 100x speedups for large-scale vector operations.` with `VectorAccelerate accelerates large-scale vector operations on Apple silicon; performance is measured, not asserted (see Performance).`
- Replace the `## 📊 Performance` section body (the sentence `Typical speedups over CPU implementations:` and the five-row table) with:

```markdown
Performance is measured by the benchmark harness (`swift run -c release VectorAccelerateBench run --mode quick`), which records hardware, OS, toolchain, power and thermal state, and a preflight calibration with every run. Run records live under `.bench/runs/` and are not committed; measured claims will be published from those records once the SP4 matrix lands. Method and program: `docs/superpowers/research/2026-09-14-observability-harness-research.md`.
```

- Line 626 area: replace `swift run VectorAccelerateBenchmarks` with `swift run -c release VectorAccelerateBench run --mode quick`.
- Line 7 version banner: `Version 0.6.0` → `Version 0.7.0`.

`CHANGELOG.md`: insert after the intro paragraph, before `## [0.6.0]`:

```markdown
## [0.7.0] - Unreleased

Benchmark harness foundation (SP0 of the observability program) and the removal of dead benchmarking code. The minor bump reflects the removed public `Benchmarking/` surface.

### Added
- **`VectorTestKit`** (non-product library target): `SeededRNG` (frozen streams), `VectorGenerator` (versioned, frozen digests), `ContentDigest`, `Tolerance`/`Comparator` with derived sum bounds, distance-epsilon `Recall`, `Stats` with bootstrap intervals, artifact schema v1, `HostFingerprint`/`EnvironmentKey`, `PreflightProbe`, append-only `RunStore`, `Compare` with the owner's boundary rule.
- **`VectorAccelerateBench`** (executable): `list`, `run`, `baseline capture`, `compare`, `validate-artifact`; two M0 cases (GPU copy probe, batch L2 distance at N=10,000 D=768).
- Mechanical guards: kit import allowlist, orphan public-type ratchet, golden streams/digests.

### Removed (breaking)
- `Sources/VectorAccelerate/Benchmarking/` and all its public types (`BenchmarkSuite`, `BenchmarkResult`, `BenchmarkConfiguration`, `BenchmarkReport`, `PerformanceComparison`, `IndexBenchmarkHarness`, `IndexBenchmarkConfiguration`, `IndexBenchmarkResult`, `LatencyStats`, `BenchmarkWorkload`, `SeededRandomNumberGenerator`, `blackHole` — the last now lives in `VectorTestKit`).
- The `VectorAccelerateBenchmarks` executable product (crossover sweep returns as a harness mode in SP4).
- `PerformanceMonitor` (unused), `IndexAccelerationConfiguration.logDecisions` (never read).

### Changed
- README performance claims retired pending measured run records.
```

- [ ] **Step 7: Amend the spec for the recorded deviations**

In `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md`:
- §4.7: replace `Percentiles use linear interpolation between order statistics (the method already in \`IndexBenchmarkHarness.LatencyStats\`, moved here).` with `Percentiles use linear interpolation between order statistics (rank = p·(n−1)); this replaces the floor-index method the old \`IndexBenchmarkHarness.LatencyStats\` used.`
- §6 row for the two old test files: replace with `| \`Tests/.../IndexBenchmarkHarnessTests.swift\` | delete (every test depends on the harness; its recall assertions duplicate the validation suite's) | no |` and a new row `| \`Tests/.../IndexBenchmarkRecallValidationTests.swift\` | keep; remove the one harness-only test (\`testBenchmarkHarnessRealisticIVF\`) and its private RNG; this file already asserts flat recall = 1.0 and IVF recall > 0.95 through the public index API | no |`.
- §10: replace the `IndexRecallTests (ported)` row with `| \`IndexBenchmarkRecallValidationTests\` (retained, trimmed) — flat recall exactly 1.0; IVF recall > 0.95 at nprobe = nlist on a seeded set | keeps existing coverage |`.
- §11 item 1: replace `+ ported (\`IndexRecallTests\`)` with `− one trimmed test`.
- §5.4: replace the grammar line with `<family>(.<op>)+(.<param>=<value>)*(.<variant>)?` and note that the M0 ID `distance.l2.batch.n=10000.d=768` requires more than one bare op segment.

- [ ] **Step 8: Full gates**

```bash
swift test 2>&1 | tail -3
swift test -c release 2>&1 | tail -3
swiftlint lint --quiet 2>&1 | grep -c "error:"    # expected: 0
```
Expected: 0 failures in both configurations; 2 skips (the IVF throughput placeholders) unchanged. Record the exact counts for Task 18.

- [ ] **Step 9: Commit**

```bash
git add -A Sources/VectorAccelerate Tests/VectorAccelerateTests README.md CHANGELOG.md docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md
git commit -q -F - <<'MSG'
refactor!: remove Benchmarking surface, PerformanceMonitor, logDecisions; retire README claims (0.7.0)

Adds the orphan public-type ratchet and requireMetalDevice(). Recall coverage
retained in IndexBenchmarkRecallValidationTests (harness-only test removed).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

---

### Task 18: Milestone M0 acceptance and evidence

**Files:**
- Create: `docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md`

- [ ] **Step 1: Run the acceptance sequence (spec §11) on the M3 Max, AC power, idle desktop; paste every command and its tail output into the evidence file**

```bash
swift test 2>&1 | tail -1;  swift test -c release 2>&1 | tail -1                                   # 1. counts
swift run -c release VectorAccelerateBench run --mode quick 2>&1 | tail -4                          # 2. artifact, quiet: true
A=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench validate-artifact "$A"
swift run -q -c release VectorAccelerateBench baseline capture --from "$A" --approve                # 3. (fails with refusedDirty if tree is dirty → commit first or add --allow-dirty and say so)
swift run -c release VectorAccelerateBench run --mode quick 2>&1 | tail -2                          # 4. second run
B=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench compare --artifact "$B"
swift run -c release VectorAccelerateBench run --mode quick --filter 'probe.*' --probe-slowdown 4 2>&1 | tail -2   # 5. slowdown
C=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench compare --artifact "$C"; echo "exit=$?"           # alert(slower), exit=0
swift run -q -c release VectorAccelerateBench compare --artifact "$C" --strict-perf; echo "exit=$?"                                      # exit=1
touch README.md.tmp && swift run -c release VectorAccelerateBench run --mode smoke 2>&1 | grep artifact; rm README.md.tmp               # 6. _dirty suffix
D=$(ls -t .bench/runs/*/*_dirty.json | head -1); swift run -q -c release VectorAccelerateBench baseline capture --from "$D"; echo "exit=$?"   # refused, exit=1
MTL_DEBUG_LAYER=1 swift run -c release VectorAccelerateBench run --mode smoke --validation-layers 2>&1 | tail -2; echo "exit=$?"         # 7. exit=0
E=$(ls -t .bench/runs/*-val/*.json | head -1); swift run -q -c release VectorAccelerateBench baseline capture --from "$E"; echo "exit=$?"    # refused, exit=1
```

- [ ] **Step 2: Write the evidence file**

`docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md` contains: date, machine fingerprint (chip, OS build, Swift, Xcode, power, thermal), the exact test counts from step 1 with the reconciliation against the previous 1764/0/2 gate (deleted suites, added suites, trimmed test), the artifact paths, the compare outputs for steps 4 and 5 (verbatim), the two refusals, and one paragraph of observations (probe GB/s, distance-case command-buffer median, whether the boundary widened and why). No number is rounded away from what the tool printed.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md
git commit -q -m "docs: SP0 milestone M0 acceptance evidence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes (run by the plan author before hand-off)

- **Spec coverage.** §3 layout → Task 0. §4.1 → Task 1. §4.2 → Task 3. §4.3 → Task 4. §4.4 → Task 2. §4.5 → Task 5. §4.6 (amended) → Task 6. §4.7 → Task 7. §4.8 → Task 8. §4.9–4.10 → Task 10. §4.11 → Task 11. §5.1–5.3 → Task 13. §5.4 → Tasks 9, 13. §5.5 → Tasks 13, 15, 16. §5.6 (amended order) → Task 12. §5.7 → Tasks 14, 15. §5.8 → Task 14. §6 → Tasks 0, 17. §7 → Tasks 13, 15, 16. §10 → Tasks 1–12, 17. §11 → Task 18. §13 is SP2 scope by the spec's own §1 non-goals and has no task here.
- **Placeholder scan.** No TBD/TODO; the only data-driven steps are the two golden generators (Tasks 3–4, deterministic scripts) and the orphan allowlist seeding (Task 17, explicit procedure).
- **Type consistency.** `PreflightRecord`/`PreflightProbe` used consistently (spec's `Preflight` split, noted in Global Constraints). `Correctness.maxAbsError`/`maxRelError` (Task 8) match the contract's amended names. `CaseRecord.validate()` (Task 8) is what `Compare` (Task 12) calls. `Runner` writes `Series` via `Stats.summarize` (Task 7). `BenchError` is defined once (Task 13) and thrown by Tasks 14–15.
