# Harness SP0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the benchmark-harness substrate (generic `VectorTestKit` library, `VectorAccelerateBench` executable) and close one measurement loop: `run` writes a schema-valid artifact with full provenance for two cases, `baseline capture` stores it, `compare` reports a status per case, and a deliberately slowed probe is caught.

**Architecture:** Two non-product SwiftPM targets. `VectorTestKit` is pure Swift (Foundation, Darwin, CryptoKit only) holding the seeded RNG, generators, digest, comparator, recall, statistics, artifact schema, host fingerprint, preflight, run store, warm-up policy, and compare logic — everything testable without a GPU. `VectorAccelerateBench` is a thin executable holding the CLI, the sampling loop, the Metal part of the fingerprint, the GPU probe, and the two M0 cases. Deletions of dead benchmark code land at the end as a breaking change recorded under CHANGELOG `[Unreleased]`.

**Tech Stack:** Swift 6.2 tools, XCTest, Metal (executable only), IOKit (executable only), CryptoKit (kit, SHA-256), SwiftPM. No third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md` (authority). Adapter seam: `docs/superpowers/specs/2026-09-14-harness-interface-contract.md`. Program brief: `docs/superpowers/research/2026-09-14-observability-harness-research.md`.

## Global Constraints

- Platform: macOS 26.0+ / iOS 26.0+ / Metal 4 (Package.swift `platforms`, unchanged). Development and trusted runs: Apple M3 Max, macOS 26.5.2, Swift 6.3.3, Xcode 26.6.
- `swift-tools-version: 6.2`; library target has `StrictConcurrency` and `ExistentialAny` enabled; new targets enable the same two settings.
- `Sources/VectorTestKit/**` may import only `Foundation`, `Darwin`, `CryptoKit`. No `@testable`. (spec §4.1, enforced by `KitImportLintTests`)
- Neither new target is a declared product (spec §3). `swift run VectorAccelerateBench` works through the root package's implicit executable product.
- No new clocks: wall time is `ContinuousClock`; GPU command-buffer time is `gpuStartTime/gpuEndTime`. `CFAbsoluteTimeGetCurrent` and `CACurrentMediaTime` do not appear in new code. (brief P2)
- Version: **none is written anywhere** (owner decision 2026-09-22: break early, no release strategy yet; everything lands on main and is bundled into the next tagged breaking release). The breaking changes (removal of `Sources/VectorAccelerate/Benchmarking/` public types, the `VectorAccelerateBenchmarks` product, `IndexAccelerationConfiguration.logDecisions`, two kernel `benchmark()` methods, the `TestRNG`/`TestDataGenerator` names) accumulate under CHANGELOG `## [Unreleased]` and an `// Unreleased —` entry in Package.swift's version-history header. `BenchVersion.package` is `"unreleased"`; `gitSHA` identifies a build. No task, commit message, comment or doc written from here on carries a version number; the README's `Version 0.6.0` banner names the last tagged release and is left alone.
- Performance boundary rule: `max(0.025, 2 × baseline RSD)` on the median per timing kind; alerts in both directions; alerts never fail the run unless `--strict-perf`. (spec §5.6)
- Missing recall is `nil`, never `1.0`. Runs with `validationLayers == true`, `preflight.quiet == false`, or `gitDirty == true` (without `--allow-dirty`) are never captured as baselines. (spec §4.6, §4.11)
- Warm-up: window 8, CoV < 0.05, minimum 3, maximum 50; the count used is recorded per sample. (spec §5.2)
- Mode defaults: `smoke` 1×3, `quick` 3×20, `full` 5×100 (samples × iterations).
- Case ID grammar: `^[a-z0-9_]+(\.[a-z0-9_]+)+(\.[a-z0-9_]+=[a-z0-9_]+)*(\.[a-z0-9_]+)?$` — a family, one or more bare op segments, zero or more `param=value` segments, an optional bare variant; never contains seeds, digests, timestamps. (spec §5.4, amended in Task 18 because the spec's one-op form rejects `distance.l2.batch.n=10000.d=768`)
- Commits end with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Commit after every task. Do not commit unrelated working-tree changes from the parallel IVF session (`git add` only the paths each task names).
- Test gates: every task ends with `swift test --filter <suite>` green; Task 18 runs the full suite in debug and release. Current full-suite expectation before this plan: 1764 passed / 0 failed / 2 skipped.
- Lint gate (from Task 17 on): `swiftlint lint --quiet` reports **no error-level finding in any file under `Sources/VectorTestKit/`, `Sources/VectorAccelerateBench/` or `Tests/VectorAccelerateTests/Harness/`**; the two generated golden files are excluded from lint by `.swiftlint.yml`. The repository's other error-level findings (20 at 853052d, all in files this plan never created) are not this plan's to fix — record the total, do not raise it.
- Machine state for measurements: every timing measurement in Tasks 15–16 ran while another session built and tested in the shared checkout (load average 3–8), so no quiet-rate claim from those tasks stands. Task 19 requires an idle machine and re-measures it. Never retry a run until it looks quiet; report every run.
- Parallel session: another session owns IVF hardening in the main checkout. Do not touch `Cases/IVF/`, IVF sources, or anything with uncommitted changes there; expect a merge at branch finish.
- Naming deviation from the spec, applied consistently: the artifact's preflight struct is `PreflightRecord` and the probing helper is `PreflightProbe` (the spec uses `Preflight` for both; JSON key stays `preflight`).

---

## File Structure

**Created — `Sources/VectorTestKit/` (library target, no product)**
- `VectorTestKit.swift` — module marker: `public enum VectorTestKit { public static let schemaVersion = 1 }`. Deleted in Task 7 once real kit files exist.
- `SeededRNG.swift` — moved from `Tests/VectorAccelerateTests/Utilities/TestRNG.swift`, type renamed. (A deprecated `TestRNG` typealias existed from Task 3 until Task 17 deleted it: no user, and the owner breaks early.)
- `VectorGenerator.swift` — moved from `Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift`, type renamed; `static let version = "1"`. (Deprecated `TestDataGenerator` typealias likewise deleted in Task 17.)
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
- `CLI/Arguments.swift` — `Arguments` (flags and values; `validate` rejects what a command does not declare).
- `CLI/CommandSupport.swift` — `Workspace` (working directory, `.bench` store, artifact loading that names the file), `ContextError`, `Diagnostics` (stderr lines).
- `Runner/BenchCase.swift` — `GPUSample`, `IterationOutput`, `BenchCase`, `PreparedCase`, `BenchContext`, `RunOptions`, `BenchMode`.
- `Runner/Runner.swift` — `Runner.run(case:context:options:)` → `CaseRecord`.
- `Fingerprint/MetalFingerprint.swift` — `MetalFingerprint.gather()` → `GPUInfo?`.
- `Probe/GPUProbe.swift` — `GPUProbe` (copy kernel, legacy command queue, `copyGBps()`, per-dispatch timing).
- `Cases/Registry.swift` — `Registry.allCases`.
- `Cases/Probe/ProbeCases.swift` — `probe.gpu_copy.bytes=134217728`.
- `Cases/Distance/DistanceCases.swift` — `distance.l2.batch.n=10000.d=768`.
- `Commands/ListCommand.swift`, `Commands/RunCommand.swift`, `Commands/BaselineCommand.swift`, `Commands/CompareCommand.swift`, `Commands/ValidateCommand.swift`.

**Created — `Tests/VectorAccelerateTests/Harness/`**
- `RepoPaths.swift` (helper), `KitImportLintTests.swift`, `ErrorDescriptionLintTests.swift` (Task 13), `CollectionInterpolationLintTests.swift` (Task 13 Step 6b; added by Task 17's fix round to close the collection-debug-literal class), `FixtureCitations.swift` (helper shared by the two scanner lints plus its own `FixtureCitationsTests`; Task 13 Step 6c, added by Task 17's fix round 1 so a lint's contract cannot cite a fixture row that does not exist), `OrphanPublicTypeTests.swift`, `SeededRNGGoldenTests.swift` (generated), `VectorGeneratorGoldenTests.swift` (generated), `ContentDigestGoldenTests.swift`, `ComparatorTests.swift`, `RecallTests.swift`, `StatsTests.swift`, `ArtifactSchemaTests.swift`, `CaseIDTests.swift`, `WarmupPolicyTests.swift`, `FingerprintTests.swift`, `PreflightInstrumentCheckTests.swift` (Task 15; opt-in via `VA_INSTRUMENT_CHECK=1`, skipped otherwise), `RunStoreTests.swift`, `CompareTests.swift`.
- `Tests/VectorAccelerateTests/Utilities/RequireMetal.swift` — `requireMetalDevice()`.
- `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md` — tracked evidence for both preflight probes (Task 15; accuracy fixes in Task 17).

**Modified**
- `Package.swift` — targets, test dependency, version-history header entry.
- `.gitignore` (`.bench/`), `.swiftlint.yml`, `README.md`, `CHANGELOG.md`.
- 15 test files gain `import VectorTestKit` and use the renamed types (listed in Task 3 and Task 4).
- `Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift` — harness-only test removed, private RNG replaced.
- `Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift` — private RNG replaced.
- `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift` — `logDecisions` removed.
- `Sources/VectorAccelerate/Kernels/Metal4/QuantizationStatisticsKernel.swift`, `WarpOptimizedSelectionKernel.swift` — their public `benchmark()` methods (and the nested `BenchmarkResult` the first returns) removed in Task 18: no caller anywhere.
- `Package.swift` version-history header entry reads `// Unreleased —` (Task 17).

**Deleted**
- `TestRNG` and `TestDataGenerator` typealiases in `Sources/VectorTestKit/` (Task 17).
- `Sources/VectorAccelerateBenchmarks/` (5 files) and its product/target (Task 0).
- `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`, `IndexBenchmarkHarness.swift`; `Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift`; `Tests/VectorAccelerateTests/PerformanceBenchmarks.swift`; `Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift` (Task 16).
- `Tests/VectorAccelerateTests/Utilities/TestRNG.swift`, `TestDataGenerator.swift` (moved, Tasks 3–4).

**Spec deviation recorded (Task 18 updates the spec):** `IndexBenchmarkRecallValidationTests.swift` does not depend on the deleted harness except for one test (`testBenchmarkHarnessRealisticIVF`), so the file is kept with that test removed instead of being ported wholesale; its flat-recall-exactly-1.0 and IVF-recall->0.95 assertions are the retained coverage the spec wanted in `IndexRecallTests`. `IndexBenchmarkHarnessTests.swift` is deleted as planned.

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

- [ ] **Step 2: Add the Unreleased entry to the version-history header in `Package.swift`**

Insert directly above the line `// 0.6.0 — Zero-copy SoA scoring + normalization parity + the promised removal:`:

```swift
// Unreleased — Benchmark harness foundation (SP0) + dead-code removal (BREAKING; lands in the next tagged release, unnumbered until then):
//   - Added: VectorTestKit (generic seeded RNG, generators, digest, comparator, recall, stats,
//     artifact schema v1, host fingerprint, preflight, run store, compare) — not a product
//   - Added: VectorAccelerateBench executable (run / baseline capture / compare / validate-artifact)
//   - Removed: Sources/VectorAccelerate/Benchmarking (BenchmarkFramework, IndexBenchmarkHarness and
//     all their public types), the VectorAccelerateBenchmarks product, PerformanceMonitor,
//     IndexAccelerationConfiguration.logDecisions, QuantizationStatisticsKernel.benchmark() and
//     WarpOptimizedSelectionKernel.benchmark() (with their nested result types), and the
//     ExecutionPath / OperationType enums whose only user was PerformanceMonitor
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

Add to the top-level `excluded:` list, after `  - Package.swift` (the two golden files that Tasks 3 and 4 generate are data, one literal array per line, up to 1,454 columns; Task 17 Step 9 added these):

```yaml
  - Tests/VectorAccelerateTests/Harness/SeededRNGGoldenTests.swift       # generated golden data: one literal array per line
  - Tests/VectorAccelerateTests/Harness/VectorGeneratorGoldenTests.swift  # generated golden data
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
///
/// The scan is line-anchored and covers attribute-prefixed imports (`@testable`, `@preconcurrency`, `@_exported`,
/// `@_implementationOnly`, `@_spi(...)`, in any order and count, including an attribute on the line before `import`),
/// SE-0409 access-level modifiers (`private import Metal`, `@_exported public import Metal`), and qualified imports
/// (`import struct Foundation.Data` reports module `Foundation`). A commented-out line such as `// import Metal`
/// does not match because `^\s*` must be followed directly by an attribute, an access-level modifier, or `import`.
/// `testClassifierFixtures` pins each of these behaviours.
///
/// Build-configuration invariance (Task 3 finding): kit code never calls `sin`/`cos`/`tan` directly; see
/// `testKitHasNoBareTrigCalls`.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    /// One import statement as seen by the lint.
    struct ImportStatement: Equatable {
        var module: String
        var testable: Bool
    }

    // Group 1: zero or more attributes (each with optional parenthesized arguments) preceding `import`, which may itself
    //          carry one SE-0409 access-level modifier (`public`, `package`, `internal`, `fileprivate`, `private`).
    // Group 2: the top-level module name, after an optional import-kind keyword and before any `.` qualifier.
    private static let importLine = try! NSRegularExpression(
        pattern: #"^\s*((?:@\w+(?:\([^)]*\))?\s+)*)(?:(?:public|package|internal|fileprivate|private)\s+)?import\s+(?:(?:typealias|struct|class|enum|protocol|actor|let|var|func)\s+)?([A-Za-z0-9_]+)"#,
        options: [.anchorsMatchLines])

    /// Every import statement in `text`, in source order. Non-import lines (code, comments) contribute nothing.
    static func imports(in text: String) -> [ImportStatement] {
        let range = NSRange(text.startIndex..., in: text)
        return importLine.matches(in: text, range: range).map { m in
            let attributes = m.range(at: 1).location == NSNotFound ? "" : String(text[Range(m.range(at: 1), in: text)!])
            let module = String(text[Range(m.range(at: 2), in: text)!])
            return ImportStatement(module: module,
                                   testable: attributes.split(whereSeparator: \.isWhitespace).contains("@testable"))
        }
    }

    /// Classifies a single source line; nil when it is not an import statement.
    static func classify(_ line: String) -> ImportStatement? { imports(in: line).first }

    func testKitImportsAreWithinAllowlist() throws {
        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for statement in Self.imports(in: text) {
                if statement.testable { violations.append("\(file.lastPathComponent): @testable import is forbidden") }
                if !Self.allowed.contains(statement.module) { violations.append("\(file.lastPathComponent): import \(statement.module) is outside the allowlist \(Self.allowed.sorted())") }
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }

    /// Pins the classifier on fixture lines so a pattern change that reopens a bypass fails here, not in review.
    func testClassifierFixtures() {
        typealias Stmt = ImportStatement
        let cases: [(line: String, expected: Stmt?)] = [
            ("import Foundation",                      Stmt(module: "Foundation", testable: false)),
            ("    import Darwin",                      Stmt(module: "Darwin", testable: false)),
            ("\timport CryptoKit",                     Stmt(module: "CryptoKit", testable: false)),
            ("// import Metal",                        nil),
            ("/* import Metal */",                     nil),
            ("@testable import Foundation",            Stmt(module: "Foundation", testable: true)),
            ("@preconcurrency import XCTest",          Stmt(module: "XCTest", testable: false)),
            ("@_exported import Metal",                Stmt(module: "Metal", testable: false)),
            ("@_implementationOnly import Metal",      Stmt(module: "Metal", testable: false)),
            ("@_spi(Internal) import Metal",           Stmt(module: "Metal", testable: false)),
            ("@preconcurrency @testable import Metal", Stmt(module: "Metal", testable: true)),
            ("@testable @preconcurrency import Metal", Stmt(module: "Metal", testable: true)),
            ("private import Metal",                   Stmt(module: "Metal", testable: false)),
            ("package import Metal",                   Stmt(module: "Metal", testable: false)),
            ("@_exported public import Metal",         Stmt(module: "Metal", testable: false)),
            ("import struct Foundation.Data",          Stmt(module: "Foundation", testable: false)),
            ("import func Darwin.sqrt",                Stmt(module: "Darwin", testable: false)),
            ("import Foundation.NSData",               Stmt(module: "Foundation", testable: false)),
            ("importantValue = 1",                     nil),
        ]
        for (line, expected) in cases {
            XCTAssertEqual(Self.classify(line), expected, "bare line: \(line)")
            // Mid-file placement pins the `.anchorsMatchLines` dependency: `^` must match at every line start.
            XCTAssertEqual(Self.imports(in: "// header\n\(line)\nlet x = 0\n"), expected.map { [$0] } ?? [], "embedded: \(line)")
        }
        // An attribute on its own line still belongs to the import that follows it.
        XCTAssertEqual(Self.imports(in: "@testable\nimport Foundation\n"), [Stmt(module: "Foundation", testable: true)])
    }

    // MARK: - Build-configuration-invariant transcendentals

    // Group 1: the trig function name. `\b` rejects `asin(` and, case-sensitively, `opaqueSin(`; `\s*\(` rejects `sinh(`/`cosine(`.
    private static let bareTrigCall = try! NSRegularExpression(pattern: #"\b(sin|cos|sinf|cosf|tan|tanf)\s*\("#)

    /// True when `line` contains a direct trig call. The `//` comment, if any, is removed first, so the wrapper's own doc
    /// comment naming `sin(x)` does not match and a comment cannot exempt a call that precedes it on the same line. The
    /// only exempt code is a wrapper definition: a line whose code part contains `func opaqueSin(` or `func opaqueCos(`,
    /// which is why each wrapper's body must stay on its `func` line. Known limits, both fail closed or are absent from the
    /// kit: `//` inside a string literal truncates that line's scan (no kit file has one); `/* */` blocks are scanned as code.
    static func hasBareTrigCall(_ line: String) -> Bool {
        var code = Substring(line)
        if let comment = code.range(of: "//") { code = code[..<comment.lowerBound] }
        if code.contains("func opaqueSin(") || code.contains("func opaqueCos(") { return false }
        let s = String(code)
        return bareTrigCall.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    /// Build-configuration invariance (Task 3 finding; see "Build-configuration-invariant transcendentals" in
    /// `SeededRNG.swift`): under `-O`, LLVM fuses an adjacent `sinf`/`cosf` pair inside one function into Darwin's
    /// `__sincosf_stret`, which differs from standalone `sinf`/`cosf` by 1-2 ulp for some inputs, so a stream frozen at
    /// `-Onone` would not reproduce at `-O` (6 of 384 frozen Gaussian values diverged before the fix). Rule: kit code never
    /// calls `sin`/`cos`/`sinf`/`cosf`/`tan`/`tanf` directly; it goes through the `@inline(never)` `opaqueSin`/`opaqueCos`
    /// wrappers in `SeededRNG.swift`, and any new transcendental that could fuse gets a sibling wrapper there. The debug
    /// (`-Onone`) stream is the contract by ruling; `SeededRNGGoldenTests` and `VectorGeneratorGoldenTests` prove both
    /// configurations match. Fixture lines pin the classifier first so a pattern change that reopens a bypass fails here.
    func testKitHasNoBareTrigCalls() throws {
        let cases: [(line: String, violation: Bool)] = [
            ("let z = sin(x)",                                                 true),
            ("let z = cos (x)",                                                true),
            ("        let z0 = r * cos(theta)",                                true),
            ("let z = Foundation.sin(x)",                                      true),
            ("let z = sinf(x) + cosf(x)",                                      true),
            ("let z = tan(x)",                                                 true),
            ("let z = tanf(x)",                                                true),
            ("let z = sin(x) // func opaqueSin( in a comment does not exempt", true),
            ("let z = opaqueSin(x) + opaqueCos(x)",                            false),
            ("let z = asin(x) + sinh(x) + cosine(x)",                          false),
            ("let n = sqrt(x) + log(x) + pow(x, 2)",                           false),
            ("/// `sin(x)` behind an inlining barrier",                        false),
            ("// Box-Muller calls `sin` and `cos` on the same argument.",      false),
            ("let z0 = r * opaqueCos(theta) // was cos(theta)",                false),
            ("internal func opaqueSin(_ x: Float) -> Float { sin(x) }",        false),
            ("internal func opaqueCos(_ x: Float) -> Float { cos(x) }",        false),
        ]
        for (line, violation) in cases {
            XCTAssertEqual(Self.hasBareTrigCall(line), violation, "line: \(line)")
        }

        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated()
            where Self.hasBareTrigCall(String(line)) {
                violations.append("\(file.lastPathComponent):\(index + 1): bare trig call; route it through opaqueSin/opaqueCos: \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }
}
```

- [ ] **Step 3: Run it; expect PASS (only the marker file exists)**

Run: `swift test --filter KitImportLintTests 2>&1 | tail -3`
Expected: `Executed 2 tests, with 0 failures`

- [ ] **Step 4: Prove the guard detects both violation kinds, then remove the probe**

The lint is a text scan, so the probe lines sit inside `#if false … #endif`: Swift parses but does not type-check
inactive branches, whereas an active `@testable import Foundation` is a hard compile error
("module 'Foundation' was not compiled for testing") that stops `swift test` before the lint ever runs.

```bash
printf '#if false\n@preconcurrency import XCTest\n@testable import Foundation\n#endif\npublic enum _LintProbe {}\n' > Sources/VectorTestKit/_LintProbe.swift
swift test --filter KitImportLintTests 2>&1 | grep -c "outside the allowlist"          # expected: 1
swift test --filter KitImportLintTests 2>&1 | grep -c "@testable import is forbidden"  # expected: 1
rm Sources/VectorTestKit/_LintProbe.swift
swift test --filter KitImportLintTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1   # expected: 0 failures
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

Run: `swift test --filter "ContentDigestGoldenTests|KitImportLintTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
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
- Produces: `public struct SeededRNG: Sendable, RandomNumberGenerator` with `init(seed: UInt64, stream: UInt64 = 0, warmup: Int = 10)`, `next() -> UInt64`, `nextFloat01() -> Float`, `nextFloat(in:) -> Float`, `nextDouble() -> Double`, `nextGaussian(mean:stdDev:) -> Float`, `nextInt(bound:)`, `nextInt(in: Range<Int>)`, `nextInt(in: ClosedRange<Int>)`, `nextBool(probability:)`, `shuffle(_:)`, `shuffled(_:)`, `sample(_:k:)`. (A deprecated `TestRNG` typealias existed from this task until Task 17 deleted it: no user.)

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

Run: `swift test --filter SeededRNGGoldenTests 2>&1 | grep -m1 "cannot find"`
Expected: `cannot find 'SeededRNG' in scope`.

- [ ] **Step 3: Move and rename**

```bash
git mv Tests/VectorAccelerateTests/Utilities/TestRNG.swift Sources/VectorTestKit/SeededRNG.swift
sed -i '' -e 's/\bTestRNG\b/SeededRNG/g' -e 's|//  TestRNG.swift|//  SeededRNG.swift|' Sources/VectorTestKit/SeededRNG.swift
grep -n "public struct SeededRNG\|extension SeededRNG: RandomNumberGenerator" Sources/VectorTestKit/SeededRNG.swift
```

Expected: two matching lines. No compatibility alias for the old name: a deprecated `TestRNG` typealias existed briefly and Task 17 deleted it (no user; the owner breaks early).

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

Run: `swift test --filter "SeededRNGGoldenTests|DifferentialKernelVsCPUTests|RoutingProvenanceTests|KitImportLintTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
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
- Produces: `public struct VectorGenerator` with the same members as `TestDataGenerator` plus `public static let version: String = "1"`; `public struct DatasetStatistics` unchanged. (A deprecated `TestDataGenerator` typealias existed from this task until Task 17 deleted it: no user.)

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

Run: `swift test --filter VectorGeneratorGoldenTests 2>&1 | grep -m1 "cannot find"`
Expected: `cannot find 'VectorGenerator' in scope`.

- [ ] **Step 3: Move, rename, add version**

```bash
git mv Tests/VectorAccelerateTests/Utilities/TestDataGenerator.swift Sources/VectorTestKit/VectorGenerator.swift
sed -i '' -e 's/\bTestDataGenerator\b/VectorGenerator/g' -e 's|//  TestDataGenerator.swift|//  VectorGenerator.swift|' -e '/^import VectorTestKit$/d' Sources/VectorTestKit/VectorGenerator.swift   # the file now lives inside the kit: drop the self-import Task 3 added
python3 - <<'PY'
p="Sources/VectorTestKit/VectorGenerator.swift"; s=open(p).read()
old="public struct VectorGenerator {\n    /// Internal RNG\n    var rng: SeededRNG\n"
assert s.count(old)==1, "anchor not found; inspect the struct header"
s=s.replace(old, "public struct VectorGenerator {\n    /// Bumped whenever any method's output stream changes; part of every case identity (spec §4.3).\n    public static let version: String = \"1\"\n\n    /// Internal RNG\n    var rng: SeededRNG\n")
open(p,"w").write(s)
PY
grep -n "public static let version" Sources/VectorTestKit/VectorGenerator.swift
```

Expected: one matching line. No compatibility alias for the old name: a deprecated `TestDataGenerator` typealias existed briefly and Task 17 deleted it (no user; the owner breaks early).

- [ ] **Step 4: Rename uses in tests**

```bash
sed -i '' 's/\bTestDataGenerator\b/VectorGenerator/g' Tests/VectorAccelerateTests/Fixtures/IVF/IVFBenchmarkFixture.swift Tests/VectorAccelerateTests/IVFQualityAssessmentTests.swift Tests/VectorAccelerateTests/OperationsDispatchTests.swift Tests/VectorAccelerateTests/IVFValidationTests.swift
grep -rn "TestDataGenerator" Tests/VectorAccelerateTests | grep -v "typealias"   # expected: no output
```

- [ ] **Step 5: Run**

Run: `swift test --filter "VectorGeneratorGoldenTests|SeededRNGGoldenTests|IVFQualityAssessmentTests|OperationsDispatchTests|KitImportLintTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
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
/// abs/rel envelope; optional ULP tier. The ten adversarial value classes are adapted from DifferentialKernelVsCPUTests.
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
        // Extreme finite range: the true distance exceeds Int64.max, so a checked Int64 subtraction would trap.
        XCTAssertEqual(Comparator.ulpDistance(-Double.greatestFiniteMagnitude, Double.greatestFiniteMagnitude), 18_437_736_874_454_810_622 as UInt64)
        XCTAssertEqual(Comparator.ulpDistance(-Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude), 4_278_190_078 as UInt64)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp, tolerance: Tolerance(ulps: 1)), .withinTolerance)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp.nextUp, tolerance: Tolerance(ulps: 1)),
                       .outOfTolerance(absError: 1.0.nextUp.nextUp - 1.0, relError: (1.0.nextUp.nextUp - 1.0) / 1.0.nextUp.nextUp))
    }

    func testFloatOverloadSharesTiersAndAppliesFloatULPs() {
        // Float arguments select the Float overload (Swift never promotes Float to Double implicitly).
        XCTAssertEqual(Comparator.compare(Float.nan, Float.nan, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(Float.nan, Float(1), tolerance: Tolerance(absolute: 1e300, relative: 1e300)), .classMismatch)
        XCTAssertEqual(Comparator.compare(Float(1), Float.infinity, tolerance: Tolerance(absolute: 1e300, relative: 1e300)), .classMismatch)
        XCTAssertEqual(Comparator.compare(Float(0), -Float(0), tolerance: Tolerance()), .withinTolerance)
        XCTAssertEqual(Comparator.compare(Float(1), Float(1).nextUp, tolerance: Tolerance(ulps: 1)), .withinTolerance)
        // Beyond the Float ULP tier: the payload is exactly what the Double overload computes on the widened values.
        let b = Double(Float(1).nextUp.nextUp)
        XCTAssertEqual(Comparator.compare(Float(1), Float(1).nextUp.nextUp, tolerance: Tolerance(ulps: 1)), .outOfTolerance(absError: b - 1.0, relError: (b - 1.0) / b))
    }

    func testAdversarialClassesPairwiseNeverCrash() {
        let t = Tolerance(absolute: 1e-5, relative: 2e-4)
        for (na, a) in classes { for (nb, b) in classes {
            let v = Comparator.compare(a, b, tolerance: t)
            let sameClass = (a.isNaN && b.isNaN) || (a.isInfinite && b.isInfinite && a.sign == b.sign) || (a.isFinite && b.isFinite)
            if !sameClass { XCTAssertEqual(v, .classMismatch, "\(na) vs \(nb)") }
            else { XCTAssertNotEqual(v, .classMismatch, "\(na) vs \(nb)") }
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

Run: `swift test --filter ComparatorTests 2>&1 | grep -m1 "cannot find"`
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
    /// absolute values sum to at most `magnitude`. Motivation: the textbook bounds sequential ≈ (D−1)·ε·Σ|xᵢ|
    /// and balanced tree ≈ ⌈log₂D⌉·ε·Σ|xᵢ|, ε = 2⁻²⁴. Implemented as `2·ε·depth·magnitude` with
    /// depth = D = max(terms, 1) for `.sequential` (dominates D−1) and depth = max(1, ⌈log₂D⌉) for `.tree`;
    /// the factor 2 is a safety margin. Tested by `ComparatorTests.testForSumSequentialAndTree`.
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

    /// Tiers 1–4 delegate to the `Double` overload on the widened values: widening is exact and injective and keeps
    /// NaN-ness, infinities and zero sign, so `Double(a).bitPattern == Double(b).bitPattern ⇔ a.bitPattern == b.bitPattern`
    /// for non-NaN pairs (NaN/NaN is settled by tier 1 first), and the envelope and the `.outOfTolerance` payload are
    /// evaluated in `Double` on the widened values. Only the ULP tier is Float-specific, applied when the envelope
    /// fails. Pinned by `ComparatorTests.testFloatOverloadSharesTiersAndAppliesFloatULPs`.
    public static func compare(_ a: Float, _ b: Float, tolerance: Tolerance) -> Verdict {
        let base = compare(Double(a), Double(b), tolerance: Tolerance(absolute: tolerance.absolute, relative: tolerance.relative, ulps: nil))
        guard case .outOfTolerance = base, let u = tolerance.ulps, ulpDistance(a, b) <= u else { return base }
        return .withinTolerance
    }

    /// Distance in representable steps between two finite doubles (sign-aware ordering of bit patterns).
    public static func ulpDistance(_ a: Double, _ b: Double) -> UInt64 {
        func ordered(_ x: Double) -> Int64 {
            let i = Int64(bitPattern: x.bitPattern)
            return i >= 0 ? i : Int64.min - i
        }
        let (x, y) = (ordered(a), ordered(b))
        // The true distance is < 2⁶⁴ but can exceed Int64.max (e.g. ±greatestFiniteMagnitude), so a checked
        // `x - y` traps; wrapping subtraction plus UInt64(bitPattern:) recovers the exact value. Pinned by testULPTier.
        return x >= y ? UInt64(bitPattern: x &- y) : UInt64(bitPattern: y &- x)
    }

    /// Float counterpart of the ordered-key scheme above. The keys lie in `[Int32.min + 1, Int32.max]`, so after
    /// widening to `Int64` the checked `-` cannot overflow; the checked form (vs the `Double` overload's `&-`) is
    /// deliberate. Pinned by `ComparatorTests.testULPTier` (`±Float.greatestFiniteMagnitude`).
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

Run: `swift test --filter ComparatorTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 10 tests, with 0 failures`

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
- Produces: `public struct RecallPolicy: Codable, Sendable, Equatable { public var k: Int; public var epsilon: Double; public init(k:epsilon:) }`; `public struct Neighbor: Sendable, Equatable { public var id: Int; public var distance: Double; public init(id:distance:) }`; `public enum RecallError: Error, Equatable { case insufficientGroundTruth(have: Int, need: Int), invalidK }` with `extension RecallError: CustomStringConvertible` (each kit error enum is `CustomStringConvertible`: the bench CLI prints errors as they are, and `ErrorDescriptionLintTests` in Task 13 fails for an `Error` type without that conformance); `public enum Recall { public static func atK(returned: [Neighbor], groundTruth: [Neighbor], policy: RecallPolicy) throws -> Double }`.

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

    /// Checked through the interpolation the CLI applies to a caught error; without a description these are the reflected cases.
    func testErrorDescriptionsStateTheRequirementAndTheRemedy() {
        let table: [(error: RecallError, caseName: String, mustContain: [String])] = [
            (.insufficientGroundTruth(have: 2, need: 3), "insufficientGroundTruth", ["at least 3", "got 2", "min(requestedK, eligibleCount)"]),
            (.invalidK, "invalidK", ["k >= 1", "min(requestedK, eligibleCount)"])
        ]
        for row in table {
            let message = "\(row.error as any Error)"
            XCTAssertFalse(message.hasPrefix(row.caseName), "\(row.caseName) renders as the raw case: \(message)")
            for piece in row.mustContain { XCTAssertTrue(message.contains(piece), "\(row.caseName): '\(piece)' missing from: \(message)") }
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

Run: `swift test --filter RecallTests 2>&1 | grep -m1 "cannot find"`
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

/// The payloads and key phrases are checked by `RecallTests.testErrorDescriptionsStateTheRequirementAndTheRemedy`.
extension RecallError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .insufficientGroundTruth(let have, let need):
            return "recall@k needs at least \(need) ground-truth neighbours (k = \(need)), got \(have); "
                + "for a small corpus pass k = min(requestedK, eligibleCount)"
        case .invalidK:
            return "recall@k needs k >= 1; for a small corpus pass k = min(requestedK, eligibleCount), never 0"
        }
    }
}

/// Distance-threshold recall@k (spec §4.6). The caller supplies independently recomputed distances for
/// valid, eligible IDs and, for small corpora, passes `k = min(requestedK, eligibleCount)` (never 0).
/// Precondition (unverified here): `groundTruth` is sorted ascending by distance — the k-th distance is
/// read at index k−1. Callers that build ground truth from an exact oracle must sort before calling.
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

Run: `swift test --filter RecallTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 8 tests, with 0 failures`

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
- Delete: `Sources/VectorTestKit/VectorTestKit.swift` (Task 0 module marker; a type named after its module shadows the module for qualified lookup, and nothing references it)

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
        // Interval bounds are discrete-ish, so two particular seeds can legitimately coincide; the seed must
        // move the interval across a handful of seeds (seeds 0–7 yield ≥ 2 distinct intervals on this data).
        let intervals = Set((0..<8).map { Stats.summarize(v, bootstrapSeed: UInt64($0)).ci95 })
        XCTAssertGreaterThan(intervals.count, 1)
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

Run: `swift test --filter StatsTests 2>&1 | grep -m1 "cannot find"`
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

Run: `swift test --filter StatsTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 5 tests, with 0 failures`

- [ ] **Step 5: Commit**

```bash
git rm -q Sources/VectorTestKit/VectorTestKit.swift   # module marker (see Files)
git add Sources/VectorTestKit/Stats.swift Tests/VectorAccelerateTests/Harness/StatsTests.swift docs/superpowers/plans/2026-09-17-harness-sp0-foundation.md
git commit -q -m "feat(kit): Stats summary with interpolated percentiles and seeded bootstrap CI; blackHole

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Artifact schema v1

**Files:**
- Create: `Sources/VectorTestKit/Schema.swift`
- Test: `Tests/VectorAccelerateTests/Harness/ArtifactSchemaTests.swift`

**Interfaces:**
- Produces: `public enum ArtifactValidationError: Error, Equatable { case unsupportedSchemaVersion(Int), missing(String), invalid(String) }` with `extension ArtifactValidationError: CustomStringConvertible` (each kit error enum is `CustomStringConvertible`: the bench CLI prints errors as they are, and `ErrorDescriptionLintTests` in Task 13 fails for an `Error` type without that conformance).
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
        let md = Metadata(packageVersion: "unreleased", gitSHA: "deadbeef", gitDirty: false, date: "2026-09-17T00:00:00Z",
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
        XCTAssertThrowsError(try JSONDecoder().decode(BenchArtifact.self, from: stripped)) { error in
            guard let decodingError = error as? DecodingError, case .keyNotFound(let key, _) = decodingError else {
                return XCTFail("expected DecodingError.keyNotFound, got \(error)")
            }
            XCTAssertEqual(key.stringValue, "environmentKey")
        }
    }

    func testTimingSubscript() {
        let a = Self.sampleArtifact()
        XCTAssertNotNil(a.cases[0].timing[.wall]); XCTAssertNil(a.cases[0].timing[.encoder])
    }

    func testMeasuredRequiresPositiveCounts() {
        var a = Self.sampleArtifact()
        a.cases[0].samples = 0; a.cases[0].iterations = 0; a.cases[0].warmupUsed = []
        a.cases[0].timing.wall = Series(raw: [], summary: Stats.summarize([], bootstrapSeed: 0), bias: nil)
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: samples/iterations/unitCount must be >= 1 for measured cases")) }
    }

    func testUnknownSchemaVersionRejectedAtDecode() throws {
        var a = Self.sampleArtifact(); a.schemaVersion = 2
        let data = try JSONEncoder().encode(a)
        XCTAssertThrowsError(try JSONDecoder().decode(BenchArtifact.self, from: data)) { XCTAssertEqual($0 as? ArtifactValidationError, .unsupportedSchemaVersion(2)) }
    }

    func testCaseIdsMustBeUniqueAndMatchIdentity() {
        var dup = Self.sampleArtifact(); dup.cases.append(dup.cases[0])
        XCTAssertThrowsError(try dup.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("duplicate case id: probe.gpu_copy.bytes=8")) }
        var mismatch = Self.sampleArtifact(); mismatch.cases[0].identity.id = "other"
        XCTAssertThrowsError(try mismatch.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: identity.id differs (other)")) }
    }

    func testEmptyRequiredMetadataFieldsRejected() {
        var a = Self.sampleArtifact(); a.metadata.environmentKey = ""
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("metadata.environmentKey")) }
        var b = Self.sampleArtifact(); b.metadata.deviceModel = ""
        XCTAssertThrowsError(try b.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("metadata.deviceModel")) }
    }

    /// Checked through the interpolation the CLI applies to a caught error; without a description these are the reflected
    /// cases (`unsupportedSchemaVersion(7)`, `invalid("…")`).
    func testValidationErrorDescriptionsCarryThePayloadAndSayWhatToDo() {
        let table: [(error: ArtifactValidationError, caseName: String, mustContain: [String])] = [
            (.unsupportedSchemaVersion(7), "unsupportedSchemaVersion", ["schemaVersion 7", "reads schemaVersion 1", "run"]),
            (.missing("metadata.environmentKey"), "missing", ["artifact is missing", "metadata.environmentKey"]),
            (.invalid("case a.b: warmupUsed.count must equal samples"), "invalid",
             ["artifact check failed", "case a.b: warmupUsed.count must equal samples"])
        ]
        for row in table {
            let message = "\(row.error as any Error)"
            XCTAssertFalse(message.hasPrefix(row.caseName), "\(row.caseName) renders as the raw case: \(message)")
            for piece in row.mustContain { XCTAssertTrue(message.contains(piece), "\(row.caseName): '\(piece)' missing from: \(message)") }
        }
    }

    func testUInt64FieldsRoundTripAtFullRange() throws {
        var a = Self.sampleArtifact()
        a.cases[0].identity.seed = 0x9E3779B97F4A7C15
        a.metadata.runSeed = UInt64.max
        let back = try JSONDecoder().decode(BenchArtifact.self, from: JSONEncoder().encode(a))
        XCTAssertEqual(back, a)
        XCTAssertEqual(back.cases[0].identity.seed, 0x9E3779B97F4A7C15)
        XCTAssertEqual(back.metadata.runSeed, UInt64.max)
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter ArtifactSchemaTests 2>&1 | grep -m1 "cannot find"`
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

/// The payloads and key phrases are checked by `ArtifactSchemaTests.testValidationErrorDescriptionsCarryThePayloadAndSayWhatToDo`.
extension ArtifactValidationError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unsupportedSchemaVersion(let version):
            return "unsupported artifact schemaVersion \(version): this build reads schemaVersion 1 only; "
                + "produce the artifact again with this build's `run`, or read it with the build that wrote it"
        case .missing(let what):
            return "artifact is missing a required value: \(what)"
        case .invalid(let what):
            return "artifact check failed: \(what)"
        }
    }
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
    /// "ac" | "battery" | "unknown" (producer convention, not validated)
    public var source: String
    public var lowPowerMode: Bool
    public init(source: String, lowPowerMode: Bool) { self.source = source; self.lowPowerMode = lowPowerMode }
}

public struct ThermalInfo: Codable, Sendable, Equatable {
    /// "nominal" | "fair" | "serious" | "critical" | "unknown" (producer convention, not validated)
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

    private enum CodingKeys: String, CodingKey { case schemaVersion, package, metadata, cases }

    /// Decoding an unknown `schemaVersion` fails with a typed error before the payload is read (spec §4.8);
    /// `encode(to:)` stays synthesized. Pinned by `ArtifactSchemaTests.testUnknownSchemaVersionRejectedAtDecode`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decode(Int.self, forKey: .schemaVersion)
        guard version == 1 else { throw ArtifactValidationError.unsupportedSchemaVersion(version) }
        schemaVersion = version
        package = try container.decode(String.self, forKey: .package)
        metadata = try container.decode(Metadata.self, forKey: .metadata)
        cases = try container.decode([CaseRecord].self, forKey: .cases)
    }

    /// Structural validation beyond what `Codable` enforces (spec §4.8). The version check is repeated here
    /// because in-memory records can be mutated after decoding.
    public func validate() throws {
        guard schemaVersion == 1 else { throw ArtifactValidationError.unsupportedSchemaVersion(schemaVersion) }
        if metadata.environmentKey.isEmpty { throw ArtifactValidationError.missing("metadata.environmentKey") }
        if metadata.deviceModel.isEmpty { throw ArtifactValidationError.missing("metadata.deviceModel") }
        var seen = Set<String>()
        for c in cases {
            guard seen.insert(c.id).inserted else { throw ArtifactValidationError.invalid("duplicate case id: \(c.id)") }
            if c.id != c.identity.id { throw ArtifactValidationError.invalid("case \(c.id): identity.id differs (\(c.identity.id))") }
        }
        for c in cases { try c.validate() }
    }
}

public extension CaseRecord {
    /// Per-case structural checks; intended for reuse by the compare step on baseline records.
    func validate() throws {
        // Non-measured cases may legitimately carry zero counts.
        if status == .measured && (samples < 1 || iterations < 1 || unitCount < 1) {
            throw ArtifactValidationError.invalid("case \(id): samples/iterations/unitCount must be >= 1 for measured cases")
        }
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

Run: `swift test --filter ArtifactSchemaTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 15 tests, with 0 failures`

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
- Produces: `public struct CaseID: Codable, Hashable, Sendable, CustomStringConvertible { public let rawValue: String; public init(validating: String) throws; public var family: String }`; `public enum CaseIDError: Error, Equatable { case malformed(String) }` with `extension CaseIDError: CustomStringConvertible` (each kit error enum is `CustomStringConvertible`: the bench CLI prints errors as they are, and `ErrorDescriptionLintTests` in Task 13 fails for an `Error` type without that conformance); `public enum CaseRegistryCheck { public static func duplicates(_ ids: [CaseID]) -> [CaseID] }`; `public struct WarmupPolicy: Sendable { public var window: Int; public var cov: Double; public var minimum: Int; public var maximum: Int; public static let standard: WarmupPolicy; public func shouldStop(after wallNs: [Double]) -> Bool }`.

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
        // "probe.x\n": ICU `$` matches before a final line terminator, so the pattern must anchor with `\z`.
        for bad in ["Probe.gpu_copy", "probe", "probe.gpu copy", "probe.gpu_copy.seed=42x!", "probe..x", ".probe.x", "probe.x.", "probe.x\n"] {
            XCTAssertThrowsError(try CaseID(validating: bad), bad)
        }
    }

    /// Checked through the interpolation the CLI applies to a caught error; without a description it is `malformed("…")`.
    func testMalformedDescriptionQuotesTheIDAndStatesTheGrammar() {
        let message = "\(CaseIDError.malformed("Probe.gpu copy") as any Error)"
        XCTAssertFalse(message.hasPrefix("malformed("), message)
        for piece in ["'Probe.gpu copy'", "<family>.<op>", "lowercase"] {
            XCTAssertTrue(message.contains(piece), "'\(piece)' missing from: \(message)")
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

Run: `swift test --filter "CaseIDTests|WarmupPolicyTests" 2>&1 | grep -m1 "cannot find"`
Expected: `cannot find 'CaseID' in scope`.

- [ ] **Step 3: Implement**

`CaseID.swift`:

```swift
import Foundation

public enum CaseIDError: Error, Equatable { case malformed(String) }

/// The quoted ID and the key phrases are checked by `CaseIDTests.testMalformedDescriptionQuotesTheIDAndStatesTheGrammar`.
extension CaseIDError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .malformed(let raw):
            return "malformed case ID '\(raw)': expected <family>.<op>[.<op>…][.<param>=<value>…][.<variant>], "
                + "each segment made of lowercase letters, digits and `_` only"
        }
    }
}

/// Stable case identity per the grammar in spec §5.4 (amended: one or more bare op segments). By contract, not enforced by
/// the grammar, a registered ID never contains seeds, digests, or timestamps.
public struct CaseID: Codable, Hashable, Sendable, CustomStringConvertible {
    public let rawValue: String

    /// Spec grammar `^…$`, anchored here with `\z` because ICU `$` also matches before a trailing newline (pinned by `CaseIDTests.testRejects`).
    private static let grammar = try! NSRegularExpression(pattern: #"^[a-z0-9_]+(\.[a-z0-9_]+)+(\.[a-z0-9_]+=[a-z0-9_]+)*(\.[a-z0-9_]+)?\z"#)

    public init(validating raw: String) throws {
        let range = NSRange(raw.startIndex..., in: raw)
        guard Self.grammar.firstMatch(in: raw, range: range) != nil else { throw CaseIDError.malformed(raw) }
        self.rawValue = raw
    }

    public var family: String { String(rawValue.split(separator: ".")[0]) }
    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        try self.init(validating: try decoder.singleValueContainer().decode(String.self))
    }
    public func encode(to encoder: any Encoder) throws {
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

Run: `swift test --filter "CaseIDTests|WarmupPolicyTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 9 tests, with 0 failures`   (5 CaseIDTests + 4 WarmupPolicyTests)

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
- Test: `Tests/VectorAccelerateTests/Harness/FingerprintTests.swift`, `Tests/VectorAccelerateTests/Harness/PreflightInstrumentCheckTests.swift` (opt-in; added by Task 15's review round 2)

**Interfaces:**
- Produces: `public struct HostFingerprint: Sendable, Equatable` (fields below) with `static func gather(packageRoot: URL?) -> HostFingerprint`, `static func thermalStateName() -> String`, `static func detectSwiftVersion() -> String`, `static var buildConfiguration: String`, `static func deviceTag(arch:osVersion:hostname:) -> String`; `public enum Subprocess { static func run(_ executable: String, _ args: [String], cwd: URL? = nil, timeout: TimeInterval = 5) -> String? }`; `public enum PowerState { static func read() -> PowerInfo }`; `public enum EnvironmentKey { static func slug(_:) -> String; static func make(chip:gpuCores:osBuild:metalToolchain:swiftVersion:buildConfiguration:validationLayers:) -> String }`; `public enum PreflightProbe { static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 100, warmupRepeats: Int = 20, warmupSeconds: Double = 0.2) -> Double; static func evaluate(cpu: StartEnd, gpu: StartEnd?, thermalStart: String, thermalEnd: String, gpuDeviceExists: Bool, driftLimitPercent: Double = 5) -> PreflightRecord }`.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import VectorTestKit

/// Spec §4.9 (host fingerprint, environment key) and §4.10 (preflight probes): host facts populate on this machine,
/// the environment key excludes the hostname and tracks the validation flag, and `evaluate` names every quiet-breaker.
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

    /// Unknown tree state must never read as clean: outside any repository both git calls fail, so `gitSHA` is nil
    /// and `gitDirty` is true (the fallback is dirty, not clean).
    func testGatherOutsideRepositoryReportsUnknownAsDirty() {
        let f = HostFingerprint.gather(packageRoot: FileManager.default.temporaryDirectory)
        XCTAssertNil(f.gitSHA); XCTAssertTrue(f.gitDirty)
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

Run: `swift test --filter FingerprintTests 2>&1 | grep -m1 "cannot find"`
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

/// Runs a fixed executable with fixed arguments, no shell; nil on any failure. Two waits are each bounded by
/// `timeout`: the child's exit, then the drain of its stdout, so a grandchild that inherits the write end cannot hold
/// the caller past the second bound. A child that ignores SIGTERM after the exit bound is SIGKILLed one second later.
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
        if done.wait(timeout: .now() + timeout) == .timedOut {
            p.terminate()  // SIGTERM, then a bounded grace period before escalating.
            if done.wait(timeout: .now() + 1) == .timedOut { _ = kill(p.processIdentifier, SIGKILL) }
            return nil
        }
        guard drained.wait(timeout: .now() + timeout) != .timedOut else { return nil }
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
    /// True when the tree has uncommitted changes, and also whenever the tree state could not be determined (no
    /// `packageRoot`, git unavailable, not a repository, `git status` failing or exceeding its timeout): unknown never
    /// reads as clean. When git itself is unavailable `gitSHA` is nil as well (the tell); a `status`-only failure is
    /// indistinguishable from a dirty tree by design. Pinned by `testGatherOutsideRepositoryReportsUnknownAsDirty`.
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

    /// VectorCore-shaped tag `<arch>-<os>-<hostname>` (spaces → dashes) unless `VA_DEVICE_TAG` is set. The hostname
    /// source differs from VectorCore (`ProcessInfo.hostName` here), so tags are not guaranteed to join across suites —
    /// unverified contract.
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
        let dirty = packageRoot.flatMap { Subprocess.run("/usr/bin/git", ["status", "--porcelain"], cwd: $0) }.map { !$0.isEmpty } ?? true
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
    /// Copy bandwidth from the fast 5th percentile of `repeats` timed copies, taken after a warm-up that ends only when BOTH
    /// `warmupRepeats` copies AND `warmupSeconds` of wall time have passed. GB/s counts read plus write.
    ///
    /// Why, from docs/superpowers/research/2026-09-18-preflight-probe-characterization.md (one M3 Max, never fully idle):
    /// a thread that was idle, blocked or waiting on subprocesses copies at as little as 0.52× its settled rate for up to 17
    /// copies (111.5 ms), so the previous best-of-3 with no warm-up read 56–117 GB/s across ten readings 0.3 s apart. Settled
    /// copy times are bimodal and alternate, which leaves the median in the valley between the modes. On identical data — ten
    /// trials of ten readings with no other process above 60 % CPU — the range of the ten readings was at most 1.76 % of their
    /// median for the 5th percentile of 100 copies (typically 1.08 %), against 1.89 % for the minimum, 2.11 % for the 10th
    /// percentile and 3.57 % for the median. With another session compiling, every estimator spread 2–12 %: that is the
    /// machine not being quiet, which the gate exists to report. Explicit, unverified contract on any other machine;
    /// exercised only by the opt-in `PreflightInstrumentCheckTests`.
    public static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 100, warmupRepeats: Int = 20, warmupSeconds: Double = 0.2) -> Double {
        precondition(warmupSeconds.isFinite && warmupSeconds >= 0, "warmupSeconds must be finite and non-negative")
        precondition(bytes > 0, "cpuCopyGBps needs at least one byte")
        let src = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        let dst = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        defer { src.deallocate(); dst.deallocate() }
        memset(src, 0x5A, bytes); memset(dst, 0, bytes)
        let clock = ContinuousClock()
        let warmupStart = clock.now
        var warmed = 0
        while warmed < warmupRepeats || clock.now - warmupStart < .seconds(warmupSeconds) { memcpy(dst, src, bytes); warmed += 1 }
        var seconds: [Double] = []
        for _ in 0..<max(repeats, 1) {
            let d = clock.measure { memcpy(dst, src, bytes) }
            seconds.append(Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
        }
        blackHole(dst.load(as: UInt8.self))
        let fast = Stats.percentile(sorted: seconds.sorted(), p: 0.05)
        return fast > 0 ? 2.0 * Double(bytes) / fast / 1e9 : 0
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

`Tests/VectorAccelerateTests/Harness/PreflightInstrumentCheckTests.swift` — the opt-in instrument check (Task 15 review round 2; skipped unless `VA_INSTRUMENT_CHECK=1`, so it does not change the default counts below):

```swift
import XCTest
import VectorTestKit

/// Opt-in instrument check: skipped unless the environment has `VA_INSTRUMENT_CHECK=1`, so the default suite cannot flake on a
/// loaded machine. It closes, for the CPU probe, the class "a preflight probe whose own repeatability is comparable to the 5 %
/// quiet gate": ten readings separated by 0.3 s idle sleeps (the gaps that used to land the reading on the post-idle ramp) must
/// span less than 2.5 % of their median, half the gate. A failure means either the instrument or the machine moved by that
/// much during the ten readings; it does not say which.
///
/// It does NOT cover the GPU probe: `GPUProbe` lives in the executable target, which has no test target. That probe's
/// repeatability rests only on the recorded runs in docs/superpowers/research/2026-09-18-preflight-probe-characterization.md.
final class PreflightInstrumentCheckTests: XCTestCase {
    func testCPUProbeReadingsSpanLessThanHalfTheQuietGate() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VA_INSTRUMENT_CHECK"] == "1",
                          "instrument check is opt-in: set VA_INSTRUMENT_CHECK=1")
        var readings: [Double] = []
        for _ in 0..<10 {
            Thread.sleep(forTimeInterval: 0.3)
            readings.append(PreflightProbe.cpuCopyGBps())
        }
        let sorted = readings.sorted()
        let median = Stats.percentile(sorted: sorted, p: 0.5)
        let spread = (sorted[sorted.count - 1] - sorted[0]) / median
        print("VA_INSTRUMENT_CHECK cpuCopyGBps GB/s: " + readings.map { String(format: "%.1f", $0) }.joined(separator: " ")
              + String(format: " | (max-min)/median = %.2f %%", spread * 100))
        XCTAssertGreaterThan(median, 0)
        XCTAssertLessThan(spread, 0.025, "ten CPU-probe readings spanned \(spread * 100) % of their median; the quiet gate is 5 %")
    }
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `swift test --filter "FingerprintTests|KitImportLintTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 11 tests, with 0 failures`   (8 FingerprintTests + 3 KitImportLintTests)

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
- Produces: `public enum RunStoreError: Error, Equatable { case runExists(String), refusedValidationLayers, refusedNotQuiet([String]), refusedDirty, noMeasuredCases, badApprovedPointer(String), unsafePathComponent(String) }`; `extension RunStoreError: CustomStringConvertible` (added by Task 16: each message says what was refused or failed and what resolves it, because the bench CLI prints these errors as they are); `public struct RunStore { public let root: URL; public init(root: URL); public func runURL(for: BenchArtifact, at: Date) -> URL; public func writeRun(_: BenchArtifact, at: Date = Date()) throws -> URL; public func captureBaseline(from: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int]; public func baseline(environmentKey:caseID:version:) throws -> CaseRecord?; public func approvedVersion(environmentKey:caseID:) throws -> Int?; public func approvedBaseline(environmentKey:caseID:) throws -> CaseRecord?; public static func load(_ url: URL) throws -> BenchArtifact; public static func encode(_: some Encodable) throws -> Data }`.

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
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(quiet: false), approve: false, allowDirty: false)) {
            XCTAssertEqual($0 as? RunStoreError, .refusedNotQuiet(["drift 9.0% exceeds 5.0%"]))
        }
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedDirty) }
        XCTAssertNoThrow(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: true))
        var empty = artifact(); empty.cases[0].status = .unavailable
        XCTAssertThrowsError(try store.captureBaseline(from: empty, approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .noMeasuredCases) }
    }

    func testUnsafePathComponentsAreRefusedBeforeAnyWrite() {
        let store = RunStore(root: root)
        var escape = artifact(); escape.cases[0].id = "../../escape"; escape.cases[0].identity.id = "../../escape"
        XCTAssertThrowsError(try store.captureBaseline(from: escape, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("caseID: ../../escape")) }
        var dotdot = artifact(); dotdot.metadata.environmentKey = ".."
        XCTAssertThrowsError(try store.writeRun(dotdot)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..")) }
        var sha = artifact(); sha.metadata.gitSHA = "../../x"
        XCTAssertThrowsError(try store.writeRun(sha)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("gitSHA: ../../x")) }
        var nulKey = artifact(); nulKey.metadata.environmentKey = "..\u{0}"; nulKey.cases[0].id = "..\u{0}"; nulKey.cases[0].identity.id = "..\u{0}"
        XCTAssertThrowsError(try store.captureBaseline(from: nulKey, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..\u{0}")) }
        XCTAssertThrowsError(try store.writeRun(nulKey)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..\u{0}")) }
        var nulID = artifact(); nulID.cases[0].id = "a\u{0}b"; nulID.cases[0].identity.id = "a\u{0}b"
        XCTAssertThrowsError(try store.captureBaseline(from: nulID, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("caseID: a\u{0}b")) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "an unsafe artifact must never create the store root")
    }

    func testBadApprovedPointerIsSurfaced() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        let dir = root.appendingPathComponent("baselines/\(key)/\(id)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let pointer = dir.appendingPathComponent("approved")
        try "abc\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedVersion(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("abc")) }
        try "0\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedVersion(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("0")) }
        try "9\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedBaseline(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("version 9 missing")) }
    }

    /// A file named `Int.max`.json in a baseline directory once made `last + 1` trap on overflow. It is not a version now, and
    /// a directory that is genuinely full ends in a thrown create-only write error rather than a trap.
    func testVersionNumberingNeverOverflows() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        let dir = root.appendingPathComponent("baselines/\(key)/\(id)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("\(Int.max).json"))
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 1])
        try Data("{}".utf8).write(to: dir.appendingPathComponent("\(Int.max - 1).json"))
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false))
    }

    /// `main.swift` prints a caught error by interpolating it as `any Error`; the same interpolation is checked here. A type
    /// without a description interpolates as its reflected case (`refusedDirty`, `runExists("…")`), which starts with the case
    /// name, hence the `hasPrefix` check. Payloads aside (the reflected rendering carries those too), every expected phrase
    /// contains a space or punctuation, which a case name cannot. The table is maintained by hand: a new `RunStoreError` case
    /// needs a row here.
    func testErrorDescriptionsSayWhatFailedAndCarryThePayload() {
        let table: [(error: RunStoreError, caseName: String, mustContain: [String])] = [
            (.runExists("/tmp/store/runs/key/run.json"), "runExists", ["/tmp/store/runs/key/run.json", "never overwritten"]),
            (.refusedValidationLayers, "refusedValidationLayers", ["baseline capture refused:", "validation layers"]),
            (.refusedNotQuiet(["drift 9.0% exceeds 5.0%", "thermal state serious"]), "refusedNotQuiet",
             ["baseline capture refused:", "not quiet", "drift 9.0% exceeds 5.0%; thermal state serious", "quiet runs"]),
            (.refusedNotQuiet([]), "refusedNotQuiet", ["baseline capture refused:", "not quiet", "no reason recorded"]),
            (.refusedDirty, "refusedDirty", ["baseline capture refused:", "clean tree", "--allow-dirty"]),
            (.noMeasuredCases, "noMeasuredCases", ["baseline capture refused:", "no measured case"]),
            (.badApprovedPointer("version 9 missing"), "badApprovedPointer", ["version 9 missing", "`approved` file"]),
            (.unsafePathComponent("caseID: ../../escape"), "unsafePathComponent", ["caseID: ../../escape", "path component"])
        ]
        for row in table {
            let message = "\(row.error as any Error)"
            XCTAssertFalse(message.isEmpty, row.caseName)
            XCTAssertFalse(message.hasPrefix(row.caseName), "\(row.caseName) renders as the raw case: \(message)")
            for piece in row.mustContain { XCTAssertTrue(message.contains(piece), "\(row.caseName): '\(piece)' missing from: \(message)") }
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter RunStoreTests 2>&1 | grep -m1 "cannot find"`
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
    case unsafePathComponent(String)
}

/// Each message says what was refused or failed and, where the caller can act on it, what resolves it, so the error can
/// be printed as it is. The `switch` is exhaustive, so the compiler requires a message for a new case; the payloads and key
/// phrases are checked by `RunStoreTests.testErrorDescriptionsSayWhatFailedAndCarryThePayload`, whose table is maintained by hand.
extension RunStoreError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .runExists(let path):
            return "run artifact path already exists and is never overwritten: \(path); run again to write under a new timestamp"
        case .refusedValidationLayers:
            return "baseline capture refused: the artifact was produced with Metal validation layers, and baselines are never "
                + "captured from validation-layer runs; run again without --validation-layers"
        case .refusedNotQuiet(let reasons):
            let why = reasons.isEmpty ? "no reason recorded" : reasons.joined(separator: "; ")
            return "baseline capture refused: the artifact's preflight was not quiet (\(why)); baselines are only captured from "
                + "quiet runs, so run again once that no longer holds (drift and thermal reasons need an idle, cool machine)"
        case .refusedDirty:
            return "baseline capture refused: the artifact was produced from a dirty git tree; run again on a clean tree or pass --allow-dirty"
        case .noMeasuredCases:
            return "baseline capture refused: the artifact has no measured case, so there is nothing to store as a baseline"
        case .badApprovedPointer(let detail):
            return "approved baseline pointer is unusable (\(detail)): the `approved` file must hold a positive version number "
                + "whose <n>.json exists beside it; capture again with --approve to rewrite it"
        case .unsafePathComponent(let detail):
            return "unsafe path component (\(detail)): the value names a directory or file under the store root, so it must be one "
                + "path component that a file URL reproduces unchanged (not empty, `.` or `..`, and without `/` or NUL)"
        }
    }
}

/// Append-only artifact and baseline storage (spec §4.11).
/// Runs: `<root>/runs/<environmentKey>/<yyyyMMdd'T'HHmmss'Z'>_<sha7|nogit>[_dirty].json`.
/// Baselines: `<root>/baselines/<environmentKey>/<caseID>/<n>.json` plus an `approved` pointer file.
public struct RunStore: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    private static let stamp: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = .gmt
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"; return f
    }()

    /// Artifact strings become single path components under `root`. Two rules, both applied before any filesystem
    /// access: (1) empty, `.`, `..`, anything containing `/`, and anything containing a NUL byte are refused; (2) the
    /// value must come back unchanged from `URL.appendingPathComponent` / `lastPathComponent`, so a component the joiner
    /// would not reproduce exactly (NUL truncation, an undecoded escape, any future normalization) is refused rather
    /// than silently rewritten. Pinned by `RunStoreTests.testUnsafePathComponentsAreRefusedBeforeAnyWrite`.
    private static func checkComponent(_ value: String, what: String) throws {
        if value.isEmpty || value.contains("/") || value == "." || value == ".." || value.utf8.contains(0) {
            throw RunStoreError.unsafePathComponent("\(what): \(value)")
        }
        guard URL(fileURLWithPath: "/").appendingPathComponent(value).lastPathComponent == value else {
            throw RunStoreError.unsafePathComponent("\(what): \(value)")
        }
    }

    private static func shaTag(_ metadata: Metadata) -> String {
        metadata.gitSHA.map { String($0.prefix(7)) } ?? "nogit"
    }

    public static func encode(_ value: some Encodable) throws -> Data {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]; return try enc.encode(value)
    }

    public static func load(_ url: URL) throws -> BenchArtifact {
        try JSONDecoder().decode(BenchArtifact.self, from: Data(contentsOf: url))
    }

    /// Pure path builder, never throws; `writeRun` validates the components before touching the filesystem.
    public func runURL(for artifact: BenchArtifact, at date: Date) -> URL {
        let name = "\(Self.stamp.string(from: date))_\(Self.shaTag(artifact.metadata))\(artifact.metadata.gitDirty ? "_dirty" : "").json"
        return root.appendingPathComponent("runs").appendingPathComponent(artifact.metadata.environmentKey).appendingPathComponent(name)
    }

    /// Create-only: an existing path is an error, never overwritten (pinned by `RunStoreTests.testRunPathLayoutAndCreateOnly`).
    @discardableResult
    public func writeRun(_ artifact: BenchArtifact, at date: Date = Date()) throws -> URL {
        try Self.checkComponent(artifact.metadata.environmentKey, what: "environmentKey")
        try Self.checkComponent(Self.shaTag(artifact.metadata), what: "gitSHA")
        let url = runURL(for: artifact, at: date)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: url.path) else { throw RunStoreError.runExists(url.path) }
        try Self.encode(artifact).write(to: url, options: [.withoutOverwriting])
        return url
    }

    private func caseDir(_ key: String, _ id: String) throws -> URL {
        try Self.checkComponent(key, what: "environmentKey")
        try Self.checkComponent(id, what: "caseID")
        return root.appendingPathComponent("baselines").appendingPathComponent(key).appendingPathComponent(id)
    }

    /// Existing version numbers, ascending. A listing error propagates (no `try?`), so a failed listing is never
    /// mistaken for an empty directory; non-positive names are not versions. Neither is `Int.max`: every value returned is
    /// below it, so the caller's `last + 1` cannot overflow (`RunStoreTests.testVersionNumberingNeverOverflows`).
    private func versions(in dir: URL) throws -> [Int] {
        try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .compactMap { $0.hasSuffix(".json") ? Int($0.dropLast(5)) : nil }.filter { $0 > 0 && $0 < Int.max }.sorted()
    }

    /// Writes every measured case as the next version; never overwrites (`.withoutOverwriting`); moves `approved` only
    /// when asked (pinned by `RunStoreTests.testCaptureAppendsAndApprovePointer`; refusals by `RunStoreTests.testCaptureRefusals`).
    /// Refusals, path checks, and encoding all complete before the first filesystem write.
    public func captureBaseline(from artifact: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int] {
        if artifact.metadata.validationLayers { throw RunStoreError.refusedValidationLayers }
        if !artifact.metadata.preflight.quiet { throw RunStoreError.refusedNotQuiet(artifact.metadata.preflight.reasons) }
        if artifact.metadata.gitDirty && !allowDirty { throw RunStoreError.refusedDirty }
        let measured = artifact.cases.filter { $0.status == .measured }
        guard !measured.isEmpty else { throw RunStoreError.noMeasuredCases }
        let prepared: [(id: String, dir: URL, data: Data)] = try measured.map {
            (id: $0.id, dir: try caseDir(artifact.metadata.environmentKey, $0.id), data: try Self.encode($0))
        }
        var written: [String: Int] = [:]
        for p in prepared {
            try FileManager.default.createDirectory(at: p.dir, withIntermediateDirectories: true)
            let n = try (versions(in: p.dir).last ?? 0) + 1
            try p.data.write(to: p.dir.appendingPathComponent("\(n).json"), options: [.withoutOverwriting])
            if approve { try "\(n)\n".write(to: p.dir.appendingPathComponent("approved"), atomically: true, encoding: .utf8) }
            written[p.id] = n
        }
        return written
    }

    public func baseline(environmentKey: String, caseID: String, version: Int) throws -> CaseRecord? {
        let url = try caseDir(environmentKey, caseID).appendingPathComponent("\(version).json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(CaseRecord.self, from: Data(contentsOf: url))
    }

    /// The pointer must hold a positive integer; anything else is `badApprovedPointer` (pinned by `RunStoreTests.testBadApprovedPointerIsSurfaced`).
    public func approvedVersion(environmentKey: String, caseID: String) throws -> Int? {
        let url = try caseDir(environmentKey, caseID).appendingPathComponent("approved")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let text = try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let n = Int(text), n >= 1 else { throw RunStoreError.badApprovedPointer(text) }
        return n
    }

    /// A pointer whose version file is missing is an error, not "no pointer" (pinned by `RunStoreTests.testBadApprovedPointerIsSurfaced`).
    public func approvedBaseline(environmentKey: String, caseID: String) throws -> CaseRecord? {
        guard let n = try approvedVersion(environmentKey: environmentKey, caseID: caseID) else { return nil }
        guard let record = try baseline(environmentKey: environmentKey, caseID: caseID, version: n) else {
            throw RunStoreError.badApprovedPointer("version \(n) missing")
        }
        return record
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter RunStoreTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 7 tests, with 0 failures`   (5 store tests + `testErrorDescriptionsSayWhatFailedAndCarryThePayload` and `testVersionNumberingNeverOverflows`, added by Task 16 and its review)

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

    /// A run that produced no timings: non-measured records may carry zero counts and no series (`CaseRecord.validate()`).
    private func nonMeasured(_ status: CaseStatus, correctness: CorrectnessStatus = .pass) -> CaseRecord {
        var c = record(wall: [1, 1, 1], correctness: correctness)
        c.status = status; c.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        c.warmupUsed = []; c.samples = 0; c.iterations = 0
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
        // A present baseline whose environment key is unknown is reported as such, not as "no baseline".
        let nilKey = Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: nil)
        XCTAssertEqual(nilKey.status, .incomparable); XCTAssertEqual(nilKey.reasons, ["baseline environment key missing"])
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
        // Correctness precedes resources and both reasons are kept; with no correctness notes the reason is bare.
        let ordered = Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail, resources: over), baseline: nil, newKey: key, baselineKey: nil)
        XCTAssertEqual(ordered.status, .correctnessFailed)
        XCTAssertTrue(ordered.reasons.contains("correctness failed"), "\(ordered.reasons)")
        XCTAssertTrue(ordered.reasons.contains("resource limit exceeded: staticThreadgroupBytes 40000 > 32768"), "\(ordered.reasons)")
        var withNotes = record(wall: [1, 1, 1], correctness: .fail); withNotes.correctness?.notes = ["row 3 differs", "row 7 differs"]
        XCTAssertEqual(Compare.compare(new: withNotes, baseline: nil, newKey: key, baselineKey: nil).reasons.first, "correctness failed: row 3 differs; row 7 differs")
    }

    func testInvalidArtifact() {
        var bad = record(wall: [1, 1, 1]); bad.warmupUsed = []
        XCTAssertEqual(Compare.compare(new: bad, baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .invalidArtifact)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: bad, newKey: key, baselineKey: key).status, .invalidArtifact)
    }

    func testNonMeasuredNewRecordIsIncomparable() {
        // A non-measured record carries no series, so every kind the baseline carries is present on the baseline side only; the
        // status line follows those skip reasons and names the cause, so `compare` output still says the new run failed.
        let oneSided = "timing.wall not compared: present on the baseline side only"
        for status in [CaseStatus.failed, .unavailable] {
            let r = Compare.compare(new: nonMeasured(status), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
            XCTAssertEqual(r.status, .incomparable, status.rawValue); XCTAssertTrue(r.deltas.isEmpty, status.rawValue)
            XCTAssertEqual(r.reasons, [oneSided, "no comparable timing kinds (new status \(status.rawValue), baseline status measured)"], status.rawValue)
        }
        // The same on the baseline side, and with no kind on either side (nothing dropped: the statuses are the whole explanation).
        let baseSide = Compare.compare(new: record(wall: [1, 1, 1]), baseline: nonMeasured(.failed), newKey: key, baselineKey: key)
        XCTAssertEqual(baseSide.status, .incomparable); XCTAssertTrue(baseSide.deltas.isEmpty)
        XCTAssertEqual(baseSide.reasons, ["timing.wall not compared: present on the new side only",
                                          "no comparable timing kinds (new status measured, baseline status failed)"])
        let neither = Compare.compare(new: nonMeasured(.failed), baseline: nonMeasured(.unavailable), newKey: key, baselineKey: key)
        XCTAssertEqual(neither.status, .incomparable); XCTAssertTrue(neither.deltas.isEmpty)
        XCTAssertEqual(neither.reasons, ["no comparable timing kinds (new status failed, baseline status unavailable)"])
        // A correctness failure on such a record still wins over incomparable; the skip reason and the status line are both kept.
        let failed = Compare.compare(new: nonMeasured(.failed, correctness: .fail), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertEqual(failed.reasons, ["correctness failed", oneSided, "no comparable timing kinds (new status failed, baseline status measured)"])
    }

    func testZeroMedianBaselineIsIncomparable() {
        let r = Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [0, 0, 0]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .incomparable); XCTAssertTrue(r.deltas.isEmpty)
        XCTAssertEqual(r.reasons, ["timing.wall not compared: baseline median 0.0 is not a positive finite number"])
    }

    /// Usable medians can still give a non-finite delta or boundary: (1e308 − 1e-300) / 1e-300 overflows, and so does 2 × rsd for
    /// rsd = 1e308. A NaN median cannot come from a JSON file but can be built in memory. When such a kind is the only one, the
    /// case is incomparable, never an alert, and the JSON rendering decodes (it used to trap on `inf`).
    func testNonFiniteDeltaOrBoundaryIsIncomparable() throws {
        func withWall(median: Double? = nil, rsd: Double? = nil) -> CaseRecord {
            var edited = record(wall: [100, 100, 100])
            if let median { edited.timing.wall?.summary.median = median }
            if let rsd { edited.timing.wall?.summary.rsd = rsd }
            return edited
        }
        let overflow = Compare.compare(new: withWall(median: 1e308), baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(overflow.status, .incomparable); XCTAssertTrue(overflow.deltas.isEmpty)
        XCTAssertEqual(overflow.reasons, ["timing.wall not compared: delta inf is not finite (new median 1e+308, baseline median 1e-300)"])

        let nan = Compare.compare(new: withWall(median: .nan), baseline: withWall(), newKey: key, baselineKey: key)
        XCTAssertEqual(nan.status, .incomparable); XCTAssertTrue(nan.deltas.isEmpty)
        let infiniteBase = Compare.compare(new: withWall(), baseline: withWall(median: .infinity), newKey: key, baselineKey: key)
        XCTAssertEqual(infiniteBase.status, .incomparable); XCTAssertTrue(infiniteBase.deltas.isEmpty)
        let boundary = Compare.compare(new: withWall(), baseline: withWall(rsd: 1e308), newKey: key, baselineKey: key)
        XCTAssertEqual(boundary.status, .incomparable); XCTAssertTrue(boundary.deltas.isEmpty)
        XCTAssertEqual(boundary.reasons, ["timing.wall not compared: boundary inf is not finite (baseline rsd 1e+308)"])

        // A correctness failure still wins, and keeps the comparison reason.
        var failing = withWall(median: 1e308); failing.correctness?.status = .fail
        let failed = Compare.compare(new: failing, baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertTrue(failed.reasons.contains { $0.contains("not finite") }, "\(failed.reasons)")

        let all = [overflow, nan, infiniteBase, boundary, failed]
        XCTAssertEqual(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render(all, format: "json").utf8)), all)
        // No delta line is rendered for these results (a reason may still quote an infinite median).
        let text = Compare.render(all, format: "text")
        XCTAssertFalse(text.contains("inf%") || text.contains("nan%") || text.contains("(boundary"), text)
    }

    /// A median that is not a positive finite number cannot anchor a ratio. On the NEW side that used to be finite garbage:
    /// 0 against 100 gave delta −1 and `alert(faster)`. A GPU timing pair can come back 0 from a real run. When such a kind is
    /// the only one, the case is incomparable; both sides use one wording that names the kind, the side and the value.
    func testNonPositiveMedianOnEitherSideIsIncomparable() {
        func withWall(median: Double) -> CaseRecord {
            var edited = record(wall: [100, 100, 100]); edited.timing.wall?.summary.median = median
            return edited
        }
        let good = record(wall: [100, 100, 100])
        for value in [0.0, -5.0, -Double.infinity, Double.nan] {
            let newSide = Compare.compare(new: withWall(median: value), baseline: good, newKey: key, baselineKey: key)
            XCTAssertEqual(newSide.status, .incomparable, "new median \(value)"); XCTAssertTrue(newSide.deltas.isEmpty, "new median \(value)")
            XCTAssertEqual(newSide.reasons, ["timing.wall not compared: new median \(value) is not a positive finite number"])
            XCTAssertEqual(Compare.exitCode([newSide], strictPerf: true), 0)
            let baseSide = Compare.compare(new: good, baseline: withWall(median: value), newKey: key, baselineKey: key)
            XCTAssertEqual(baseSide.status, .incomparable, "baseline median \(value)")
            XCTAssertTrue(baseSide.deltas.isEmpty, "baseline median \(value)")
            XCTAssertEqual(baseSide.reasons, ["timing.wall not compared: baseline median \(value) is not a positive finite number"])
        }
        // A measured run whose timings are all zero takes the same path (no hand-edited summary needed).
        let measuredZero = Compare.compare(new: record(wall: [0, 0, 0]), baseline: good, newKey: key, baselineKey: key)
        XCTAssertEqual(measuredZero.status, .incomparable)
        XCTAssertEqual(measuredZero.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number"])
        // Both sides unusable: one reason per side, new first.
        let both = Compare.compare(new: withWall(median: 0), baseline: withWall(median: -1), newKey: key, baselineKey: key)
        XCTAssertEqual(both.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number",
                                      "timing.wall not compared: baseline median -1.0 is not a positive finite number"])
    }

    /// An unusable kind must not mask the others: a broken GPU timer (commandBuffer median 0) beside a genuine wall regression
    /// is still `alert(slower)`, and `--strict-perf` still fails on it. The unusable kind contributes no delta and no verdict,
    /// only a reason; the case is incomparable only when no kind is left. The same holds for all three unusable conditions and
    /// for either side, and the reason is rendered in every format whatever the status.
    func testUnusableKindIsSkippedWithAReasonAndTheRestDecide() throws {
        /// `wall` from the given samples; `commandBuffer` from `gpu`, or with its median or rsd overwritten.
        func pair(wall: [Double], gpu: [Double] = [100, 100, 100], gpuMedian: Double? = nil, gpuRSD: Double? = nil) -> CaseRecord {
            var edited = record(wall: wall)
            edited.timing.commandBuffer = record(wall: gpu).timing.wall
            if let gpuMedian { edited.timing.commandBuffer?.summary.median = gpuMedian }
            if let gpuRSD { edited.timing.commandBuffer?.summary.rsd = gpuRSD }
            return edited
        }
        let base = pair(wall: [100, 100, 100])
        let newZero = "timing.commandBuffer not compared: new median 0.0 is not a positive finite number"
        let baseZero = "timing.commandBuffer not compared: baseline median 0.0 is not a positive finite number"
        func expect(_ result: CompareResult, _ status: CompareStatus, reasons: [String], strictExit: Int32, _ what: String) {
            XCTAssertEqual(result.status, status, what); XCTAssertEqual(result.reasons, reasons, what)
            XCTAssertEqual(result.deltas.map(\.kind), status == .incomparable ? [] : [.wall], what)
            XCTAssertEqual(Compare.exitCode([result], strictPerf: true), strictExit, what)
            XCTAssertEqual(Compare.exitCode([result], strictPerf: false), 0, what)
        }

        // wall regresses beyond the boundary; commandBuffer is unusable on the new side, then on the baseline side.
        let slowerNewSide = Compare.compare(new: pair(wall: [200, 200, 200], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(slowerNewSide, .alertSlower, reasons: [newZero], strictExit: 1, "regression, unusable on the new side")
        let slowerBaseSide = Compare.compare(new: pair(wall: [200, 200, 200]), baseline: pair(wall: [100, 100, 100], gpu: [0, 0, 0]),
                                             newKey: key, baselineKey: key)
        expect(slowerBaseSide, .alertSlower, reasons: [baseZero], strictExit: 1, "regression, unusable on the baseline side")
        // the usable kind is within the boundary: ok, and the reason is still recorded.
        let okWithReason = Compare.compare(new: pair(wall: [101, 101, 101], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(okWithReason, .ok, reasons: [newZero], strictExit: 0, "within the boundary")
        // every kind unusable: incomparable, and the per-kind reasons are the explanation.
        let none = Compare.compare(new: pair(wall: [0, 0, 0], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(none, .incomparable, reasons: ["timing.wall not compared: new median 0.0 is not a positive finite number", newZero],
               strictExit: 0, "no kind left")

        // The overflow pair and the boundary overflow beside a healthy kind: judged on the healthy kind.
        let overflowReason = "timing.commandBuffer not compared: delta inf is not finite (new median 1e+308, baseline median 1e-300)"
        let tinyBase = pair(wall: [100, 100, 100], gpuMedian: 1e-300)
        let overflowSlower = Compare.compare(new: pair(wall: [200, 200, 200], gpuMedian: 1e308), baseline: tinyBase, newKey: key, baselineKey: key)
        expect(overflowSlower, .alertSlower, reasons: [overflowReason], strictExit: 1, "overflow beside a regression")
        let overflowOK = Compare.compare(new: pair(wall: [101, 101, 101], gpuMedian: 1e308), baseline: tinyBase, newKey: key, baselineKey: key)
        expect(overflowOK, .ok, reasons: [overflowReason], strictExit: 0, "overflow beside a healthy kind")
        let wideBase = pair(wall: [100, 100, 100], gpuRSD: 1e308)
        let boundaryFaster = Compare.compare(new: pair(wall: [50, 50, 50]), baseline: wideBase, newKey: key, baselineKey: key)
        expect(boundaryFaster, .alertFaster, reasons: ["timing.commandBuffer not compared: boundary inf is not finite (baseline rsd 1e+308)"],
               strictExit: 1, "boundary overflow beside an improvement")

        // The other direction: wall unusable, commandBuffer decides.
        var wallZero = pair(wall: [0, 0, 0], gpu: [200, 200, 200])
        let gpuDecides = Compare.compare(new: wallZero, baseline: base, newKey: key, baselineKey: key)
        XCTAssertEqual(gpuDecides.status, .alertSlower); XCTAssertEqual(gpuDecides.deltas.map(\.kind), [.commandBuffer])
        XCTAssertEqual(gpuDecides.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number"])
        // A correctness failure still wins; the comparable kind's delta and the skip reason are both kept.
        wallZero.correctness?.status = .fail
        let failed = Compare.compare(new: wallZero, baseline: base, newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed); XCTAssertEqual(failed.deltas.map(\.kind), [.commandBuffer])
        XCTAssertEqual(failed.reasons, ["correctness failed", "timing.wall not compared: new median 0.0 is not a positive finite number"])
        // A kind present on one side only is the fourth skip condition: `Runner` omits a declared GPU kind whose samples came
        // back in the wrong shape, so a broken GPU timer can appear as an absent series, not only as a zero median.
        let oneSided = Compare.compare(new: pair(wall: [200, 200, 200], gpu: [0, 0, 0]), baseline: record(wall: [100, 100, 100]),
                                       newKey: key, baselineKey: key)
        XCTAssertEqual(oneSided.status, .alertSlower); XCTAssertEqual(oneSided.deltas.map(\.kind), [.wall])
        XCTAssertEqual(oneSided.reasons, ["timing.commandBuffer not compared: present on the new side only"])
        let otherSide = Compare.compare(new: record(wall: [200, 200, 200]), baseline: pair(wall: [100, 100, 100]),
                                        newKey: key, baselineKey: key)
        XCTAssertEqual(otherSide.status, .alertSlower); XCTAssertEqual(otherSide.deltas.map(\.kind), [.wall])
        XCTAssertEqual(otherSide.reasons, ["timing.commandBuffer not compared: present on the baseline side only"])
        // Absent on both sides is nothing to compare, not a drop: no reason.
        let neither = Compare.compare(new: record(wall: [100, 100, 100]), baseline: record(wall: [100, 100, 100]),
                                      newKey: key, baselineKey: key)
        XCTAssertEqual(neither.status, .ok); XCTAssertEqual(neither.reasons, [])

        // The reason is visible in every format when the status is ok or an alert, not only when it is incomparable.
        for result in [okWithReason, slowerNewSide] {
            let text = Compare.render([result], format: "text")
            XCTAssertTrue(text.contains("    wall "), text); XCTAssertTrue(text.contains("    reason: \(newZero)\n"), text)
            let markdown = Compare.render([result], format: "markdown")
            XCTAssertTrue(markdown.contains("| \(result.id) | \(result.status.rawValue) | wall |"), markdown)
            XCTAssertTrue(markdown.contains("\n\n- \(result.id): \(newZero)\n"), markdown)
            let decoded = try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([result], format: "json").utf8))
            XCTAssertEqual(decoded, [result]); XCTAssertEqual(decoded.first?.reasons, [newZero])
        }
        let everything = [slowerNewSide, slowerBaseSide, okWithReason, none, overflowSlower, overflowOK, boundaryFaster, gpuDecides, failed]
        XCTAssertEqual(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render(everything, format: "json").utf8)), everything)
    }

    func testExitCodes() {
        let ok = CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])
        let alert = CompareResult(id: "a.b", status: .alertSlower, deltas: [], reasons: [])
        let faster = CompareResult(id: "a.b", status: .alertFaster, deltas: [], reasons: [])
        let fail = CompareResult(id: "a.b", status: .correctnessFailed, deltas: [], reasons: [])
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, faster], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, faster], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, fail], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .invalidArtifact, deltas: [], reasons: [])], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .resourceFailed, deltas: [], reasons: [])], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .incomparable, deltas: [], reasons: [])], strictPerf: true), 0)
    }

    func testRenderFormats() throws {
        let r = Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertTrue(Compare.render([r], format: "text").contains("alert(slower)"))
        XCTAssertTrue(Compare.render([r], format: "markdown").contains("| probe.gpu_copy.bytes=8 |"))
        let decoded = try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([r], format: "json").utf8))
        XCTAssertEqual(decoded, [r])
    }

    func testMarkdownWithMultipleResultsKeepsTableContiguous() throws {
        let a = Compare.compare(new: record("probe.a", wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        let b = Compare.compare(new: record("probe.b", wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        let out = Compare.render([a, b], format: "markdown")
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false)
        // Header, separator, and one placeholder row per result form one contiguous block.
        let table = lines.indices.filter { lines[$0].hasPrefix("|") }
        XCTAssertEqual(table.count, 4, out)
        XCTAssertEqual(table, Array(try XCTUnwrap(table.first)...(try XCTUnwrap(table.last))), "table rows must be contiguous:\n\(out)")
        XCTAssertTrue(lines.contains("| probe.a | incomparable | — | — | — | — | — |"), out)
        XCTAssertTrue(lines.contains("| probe.b | incomparable | — | — | — | — | — |"), out)
        // Reasons come after the table, separated by one blank line, one newline-terminated bullet each.
        let lastRow = try XCTUnwrap(table.last)
        XCTAssertEqual(lines[lastRow + 1], "", out)
        let bullets = lines.indices.filter { lines[$0].hasPrefix("- ") }
        XCTAssertTrue(bullets.allSatisfy { $0 > lastRow }, out)
        XCTAssertEqual(bullets.map { String(lines[$0]) }, ["- probe.a: no baseline for probe.a under k1", "- probe.b: no baseline for probe.b under k1"])
        XCTAssertTrue(out.hasSuffix("\n"), out)
    }

    func testTextRenderDoesNotTruncateLongIDs() {
        let long = "distance.l2.batch.n=10000.d=768.k=100.nprobe=32.isolated_variant"
        XCTAssertGreaterThan(long.count, 44)
        let out = Compare.render([CompareResult(id: long, status: .ok, deltas: [], reasons: [])], format: "text")
        XCTAssertTrue(out.contains(long), out)
        XCTAssertTrue(out.hasPrefix("\(long) ok\n"), "a long id is printed in full followed by one space:\n\(out)")
        let short = Compare.render([CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])], format: "text")
        XCTAssertTrue(short.hasPrefix("a.b" + String(repeating: " ", count: 41) + " ok\n"), "a short id is still padded to 44 columns:\n\(short)")
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `swift test --filter CompareTests 2>&1 | grep -m1 "cannot find"`
Expected: `cannot find 'Compare' in scope`.

- [ ] **Step 3: Implement**

```swift
import Foundation

/// Outcome of comparing one new case against its baseline. Raw values are the labels the renderers print.
public enum CompareStatus: String, Codable, Sendable, Equatable {
    case ok
    case alertSlower = "alert(slower)"
    case alertFaster = "alert(faster)"
    case incomparable
    case correctnessFailed
    case resourceFailed
    case invalidArtifact
}

/// One timing kind's comparison. `delta` is `(newMedian - baseMedian) / baseMedian`; `boundary` is the symmetric
/// alert threshold; `widened` is true when 2 × baseline RSD exceeded the 2.5% floor (`CompareTests.testBoundaryWidensWithBaselineRSD`).
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
/// Precedence invalidArtifact → correctnessFailed → resourceFailed → incomparable → alert → ok is pinned by
/// `CompareTests.testInvalidArtifact`, `testCorrectnessAndResourceFailuresWinEvenWithoutBaseline`,
/// `testIncomparableIdentityKeyAndMissing`, `testNonMeasuredNewRecordIsIncomparable`, `testZeroMedianBaselineIsIncomparable`,
/// and `testAlertSlowerAndFaster`; blocking rules by `testExitCodes`. A timing kind that cannot be compared is skipped with
/// a reason and the remaining kinds decide (`testUnusableKindIsSkippedWithAReasonAndTheRestDecide`).
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
        if new.correctness?.status == .fail {
            let notes = new.correctness?.notes ?? []
            reasons.append("correctness failed" + (notes.isEmpty ? "" : ": " + notes.joined(separator: "; ")))
            status = status ?? .correctnessFailed
        }
        if let r = new.resources {
            if let s = r.staticThreadgroupBytes, let m = r.maxThreadgroupBytes, s > m {
                reasons.append("resource limit exceeded: staticThreadgroupBytes \(s) > \(m)"); status = status ?? .resourceFailed
            }
            for n in r.notes where n.hasPrefix("LIMIT_EXCEEDED:") { reasons.append("resource limit exceeded: \(n)"); status = status ?? .resourceFailed }
        }

        // 4. comparability
        guard let base = baseline else {
            reasons.append("no baseline for \(new.id) under \(newKey)")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        guard let baseKey = baselineKey else {
            reasons.append("baseline environment key missing")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        var comparable = true
        if new.identity != base.identity { reasons.append("identity differs from baseline"); comparable = false }
        if newKey != baseKey { reasons.append("environment key differs: \(newKey) vs \(baseKey)"); comparable = false }
        if !comparable { return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons) }

        // 5. deltas
        var deltas: [KindDelta] = []
        var alert: CompareStatus?
        var skipped: [String] = []
        for kind in TimingKind.allCases {
            // A kind that cannot be compared is skipped with a reason and the remaining kinds decide the case, so a broken GPU
            // timer cannot hide a wall regression (`CompareTests.testUnusableKindIsSkippedWithAReasonAndTheRestDecide`). Four
            // conditions, one wording — `timing.<kind> not compared: …` — on either side.
            let notCompared = "timing.\(kind.rawValue) not compared"
            // 0. Present on one side only. `Runner` omits a declared GPU kind whose samples came back in the wrong shape, so a
            // broken GPU timer can appear as an absent series, not only as a zero median. Absent on both sides is nothing to
            // compare, not a drop.
            let sides = (new: new.timing[kind], base: base.timing[kind])
            if sides.new == nil, sides.base == nil { continue }
            guard let n = sides.new, let b = sides.base else {
                skipped.append("\(notCompared): present on the \(sides.new == nil ? "baseline" : "new") side only")
                continue
            }
            // 1–2. `validate()` does not bound the summary fields. A median that is not a positive finite number cannot anchor a
            // ratio: as the baseline it divides by zero or flips the sign, and as the new value it is finite garbage that would
            // read as an alert (0 against 100 is delta −1).
            let medians = [(side: "new", value: n.summary.median), (side: "baseline", value: b.summary.median)]
            let rejected = medians.filter { !($0.value > 0 && $0.value.isFinite) }
            guard rejected.isEmpty else {
                skipped += rejected.map { "\(notCompared): \($0.side) median \($0.value) is not a positive finite number" }
                continue
            }
            let delta = (n.summary.median - b.summary.median) / b.summary.median
            let boundary = max(floorBoundary, 2 * b.summary.rsd)
            // 3. Usable medians can still overflow here (a huge median over a tiny one; 2 × a huge rsd).
            guard delta.isFinite, boundary.isFinite else {
                if !delta.isFinite {
                    skipped.append("\(notCompared): delta \(delta) is not finite "
                                   + "(new median \(n.summary.median), baseline median \(b.summary.median))")
                }
                if !boundary.isFinite { skipped.append("\(notCompared): boundary \(boundary) is not finite (baseline rsd \(b.summary.rsd))") }
                continue
            }
            let alerting = delta > boundary || delta < -boundary
            deltas.append(KindDelta(kind: kind, newMedian: n.summary.median, baseMedian: b.summary.median, delta: delta,
                                    boundary: boundary, widened: boundary > floorBoundary, alerting: alerting))
            // With several timing kinds the status reflects the first alerting kind in `TimingKind.allCases` order; each kind's own `alerting` is rendered.
            if alerting, alert == nil { alert = delta > 0 ? .alertSlower : .alertFaster }
        }
        // Skip reasons stay on the result whatever its status; every format of `render` prints a result's reasons. A `KindDelta`
        // is appended only after both guards above, which check all four of its Double fields, so every one this function
        // returns holds finite values (`CompareTests.testNonFiniteDeltaOrBoundaryIsIncomparable`).
        reasons += skipped
        // Nothing left to judge by (non-measured record, no timing kind present on both sides, or every such kind skipped) is
        // incomparable, never ok (`CompareTests.testNonMeasuredNewRecordIsIncomparable`, `testZeroMedianBaselineIsIncomparable`).
        // The status line is appended, after the skip reasons, when there are no skip reasons (nothing was dropped, so the two
        // statuses are the whole explanation) and whenever either record is not measured (its status is the cause of the
        // one-sided skips above it). A measured pair whose every kind was skipped is explained by the skip reasons alone.
        if deltas.isEmpty {
            if skipped.isEmpty || new.status != .measured || base.status != .measured {
                reasons.append("no comparable timing kinds (new status \(new.status.rawValue), baseline status \(base.status.rawValue))")
            }
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        return CompareResult(id: new.id, status: status ?? alert ?? .ok, deltas: deltas, reasons: reasons)
    }

    public static func exitCode(_ results: [CompareResult], strictPerf: Bool) -> Int32 {
        let hard: Set<CompareStatus> = [.invalidArtifact, .correctnessFailed, .resourceFailed]
        if results.contains(where: { hard.contains($0.status) }) { return 1 }
        if strictPerf, results.contains(where: { $0.status == .alertSlower || $0.status == .alertFaster }) { return 1 }
        return 0
    }

    /// Right-pads to `width` columns and never truncates, unlike `String.padding(toLength:)`
    /// (`CompareTests.testTextRenderDoesNotTruncateLongIDs`).
    private static func pad(_ s: String, to width: Int) -> String { s + String(repeating: " ", count: max(0, width - s.count)) }

    /// A ratio rendered as a percentage with two decimals, signed on request (`+4.00%`, `2.50%`).
    private static func pct(_ value: Double, signed: Bool) -> String { String(format: signed ? "%+.2f%%" : "%.2f%%", value * 100) }

    public static func render(_ results: [CompareResult], format: String) -> String {
        switch format {
        case "json":
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            // try!, as a contract: `JSONEncoder` throws here only for a non-finite Double. Results produced by `Compare.compare`
            // carry finite values only (`CompareTests.testNonFiniteDeltaOrBoundaryIsIncomparable` for skipped kinds;
            // `testUnusableKindIsSkippedWithAReasonAndTheRestDecide` round-trips results that carry deltas through this encoder).
            // A hand-built `CompareResult` holding NaN or an infinity is the caller's responsibility and traps here.
            return String(decoding: try! enc.encode(results), as: UTF8.self)
        case "markdown":
            var out = "| case | status | kind | new median (ns) | base median (ns) | delta | boundary |\n|---|---|---|---|---|---|---|\n"
            for r in results {
                if r.deltas.isEmpty { out += "| \(r.id) | \(r.status.rawValue) | — | — | — | — | — |\n" }
                for d in r.deltas {
                    out += "| \(r.id) | \(r.status.rawValue) | \(d.kind.rawValue) | \(String(format: "%.0f", d.newMedian)) | \(String(format: "%.0f", d.baseMedian)) "
                        + "| \(pct(d.delta, signed: true)) | \(pct(d.boundary, signed: false))\(d.widened ? " (widened)" : "") |\n"
                }
            }
            // Reasons follow the whole table, after one blank line, so rows stay contiguous across results
            // (`CompareTests.testMarkdownWithMultipleResultsKeepsTableContiguous`).
            let bullets = results.flatMap { r in r.reasons.map { "- \(r.id): \($0)\n" } }
            if !bullets.isEmpty { out += "\n" + bullets.joined() }
            return out
        default:
            var out = ""
            for r in results {
                out += "\(pad(r.id, to: 44)) \(r.status.rawValue)\n"
                for d in r.deltas {
                    out += "    \(pad(d.kind.rawValue, to: 14)) \(pct(d.delta, signed: true)) "
                        + "(boundary \(pct(d.boundary, signed: false))\(d.widened ? ", widened" : ""))\(d.alerting ? "  <-- alert" : "")\n"
                }
                for reason in r.reasons { out += "    reason: \(reason)\n" }
            }
            return out
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `swift test --filter CompareTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 15 tests, with 0 failures`

A timing kind present on both sides that cannot be compared is skipped with a reason, and the remaining kinds decide the case; the case is `incomparable` only when no kind is left (`CompareTests.testUnusableKindIsSkippedWithAReasonAndTheRestDecide`). Whole-case `incomparable` on one unusable kind would be a masking path: a broken GPU timer (`commandBuffer` median 0, which a real run can produce) would hide a genuine `wall` regression and `--strict-perf` would exit 0 on it; skipping silently would leave no trace. Four conditions make a kind unusable, identically on both sides and in one wording (`timing.<kind> not compared: …`): a kind present on one side only (Task 17: `Runner` omits a declared GPU kind whose samples came back in the wrong shape, so a broken GPU timer can appear as an absent series; absent on both sides is nothing to compare and carries no reason), a new median or a baseline median that is not a positive finite number (0 against 100 would otherwise read `alert(faster)`), and a `delta` or `boundary` that is not finite (usable medians can overflow the ratio; 2 × rsd can overflow; `validate()` does not bound summary fields). The skipped kind contributes no delta and no verdict; the reason stays on the result whatever its status, and every format of `render` prints a result's reasons (text and markdown already did so for every status, so `render` needed no change; the test pins it). The boundary rule and `Compare.exitCode` are untouched. This also closes the earlier minor that partially-comparable pairs emitted no per-kind skip reason. A kind present on one side only carried no reason until Task 17 made it the fourth condition. The `no comparable timing kinds (new status …, baseline status …)` line follows the skip reasons when there are none (nothing was dropped) and whenever either record is not measured (its status is the cause of the one-sided skips), so a failed run's cause stays in `compare` output; a measured pair whose every kind was skipped is explained by the skip reasons alone (`testNonMeasuredNewRecordIsIncomparable`, `testZeroMedianBaselineIsIncomparable`).

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorTestKit/Compare.swift Tests/VectorAccelerateTests/Harness/CompareTests.swift
git commit -q -m "feat(kit): compare with owner boundary rule, statuses, exit codes, renderers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Executable skeleton — arguments, adapter types, sampling loop, `list`, `validate-artifact`

**Files:**
- Create: `Sources/VectorAccelerateBench/CLI/Arguments.swift`, `CLI/CommandSupport.swift`, `Runner/BenchCase.swift`, `Runner/Runner.swift`, `Cases/Registry.swift`, `Commands/ListCommand.swift`, `Commands/ValidateCommand.swift`
- Modify: `Sources/VectorAccelerateBench/main.swift`
- Test: `Tests/VectorAccelerateTests/Harness/ErrorDescriptionLintTests.swift` (owned here: this is the first task at which every type the lint names exists — the four kit error enums from Tasks 6, 8, 9, 11 and the executable's `BenchError` and `ContextError` — and it is the task that introduces the `error: \(error)` printing the lint protects); `Tests/VectorAccelerateTests/Harness/CollectionInterpolationLintTests.swift` (Step 6b, added by Task 17's fix round); `Tests/VectorAccelerateTests/Harness/FixtureCitations.swift` (Step 6c, added by Task 17's fix round 1)

**Interfaces:**
- Consumes: `CaseID`, `CaseIdentity`, `CaseRecord`, `Timing`, `Series`, `Stats`, `Correctness`, `Resources`, `WarmupPolicy`, `VectorGenerator`, `HostFingerprint`, `GPUInfo`, `RunStore`, `BenchArtifact`.
- Produces (executable-internal): `Arguments { command, subcommand, positionals: [String], options: [String: String], flags: Set<String>; func validate(options:flags:operands:) throws; func value(_:) -> String?; func has(_:) -> Bool; func parsed<T>(_:expecting:_:) throws -> T? }` — every command calls `validate` first with the options and flags it reads and its operand count, so an undeclared option, flag or operand, or a declared option without a value, is a usage error (exit 2) before any work; `enum Workspace { static var root: URL; static var store: RunStore; static func loadArtifact(_:validating:) throws -> BenchArtifact }`, `struct ContextError: Error, CustomStringConvertible`, `enum Diagnostics { static func line(_:) }`; `BenchMode` (`smoke|quick|full` with `samples`, `iterations`); `RunOptions { mode, samples, iterations, seed, validationLayers, requireGPU, probeSlowdown, label }`; `GPUSample { kind: TimingKind; startNs, endNs: Double; label: String }`; `IterationOutput { gpu: [GPUSample]; payload: (any Sendable)? }`; `protocol BenchCase { id: CaseID; identity: CaseIdentity; providedTiming: Set<TimingKind>; unitCount: Int; func prepare(_: BenchContext) async throws -> any PreparedCase }`; `protocol PreparedCase: AnyObject { runOnce() async throws -> IterationOutput; verify(_:) throws -> Correctness; resources() -> Resources?; resetForSample() async throws }`; `final class BenchContext { device: (any MTLDevice)?; host: HostFingerprint; gpu: GPUInfo?; options: RunOptions; func makeGenerator(stream:) -> VectorGenerator }`; `enum Runner { static func run(_ c: any BenchCase, context:, policy:) async -> CaseRecord }`; `enum Registry { static var all: [any BenchCase] }`.

- [ ] **Step 1: `CLI/Arguments.swift` and `CLI/CommandSupport.swift`**

```swift
import Foundation

/// Hand-written parser: `<command> [<subcommand>] [--key value | --flag] ...`. No third-party dependency (spec §5.5).
struct Arguments {
    static let booleanFlags: Set<String> = ["--approve", "--allow-dirty", "--strict-perf", "--require-gpu", "--validation-layers", "--help", "-h"]

    let command: String
    let subcommand: String?
    /// Every positional in order, the command name first when one was given.
    let positionals: [String]
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
        self.positionals = positionals; self.options = options; self.flags = flags
    }

    /// Contract for commands (not enforced here): call this before anything else, with the options and flags the command reads
    /// and the number of operands (positionals after the command name) it takes. Anything else given on the command line is a
    /// `BenchError.usage` naming the offender and the command. Unknown names are reported before missing values: the parser
    /// takes an undeclared `--name` for an option, so a mistyped flag at the end of the line would otherwise be reported as
    /// "requires a value".
    func validate(options allowedOptions: Set<String>, flags allowedFlags: Set<String>, operands: Int) throws {
        let given = Set(options.keys).union(flags), allowed = allowedOptions.union(allowedFlags)
        if let unknown = given.subtracting(allowed).min() {
            let takes = allowed.isEmpty ? "it takes no options" : "it takes: \(allowed.sorted().joined(separator: ", "))"
            throw BenchError.usage("unknown option \(unknown) for `\(command)`; \(takes)")
        }
        if let key = options.filter({ $0.value.isEmpty }).keys.min() { throw BenchError.usage("option \(key) requires a value") }
        if let extra = positionals.dropFirst(1 + operands).first {
            throw BenchError.usage("unexpected argument '\(extra)' for `\(command)`; see --help")
        }
    }

    func value(_ key: String) -> String? { options[key] }
    func has(_ flag: String) -> Bool { flags.contains(flag) }

    /// An absent option is `nil` (the caller supplies the default); a present value that `parse` rejects throws `BenchError.usage`
    /// naming the option and the value.
    func parsed<T>(_ key: String, expecting what: String, _ parse: (String) -> T?) throws -> T? {
        guard let raw = options[key] else { return nil }
        guard let parsedValue = parse(raw) else { throw BenchError.usage("\(key) expects \(what), got '\(raw)'") }
        return parsedValue
    }
}
```

**`CLI/CommandSupport.swift`**

```swift
import Foundation
import VectorTestKit

/// The one place the bench spells its working directory and the `.bench` store under it. `run` writes there and
/// `baseline capture` / `compare` read there; two spellings that drifted apart would show up as "incomparable, exit 0",
/// not as an error.
enum Workspace {
    static var root: URL { URL(fileURLWithPath: FileManager.default.currentDirectoryPath) }
    static var store: RunStore { RunStore(root: root.appendingPathComponent(".bench")) }

    /// A failure names the file, which a bare `DecodingError` or `ArtifactValidationError` does not.
    static func loadArtifact(_ path: String, validating: Bool) throws -> BenchArtifact {
        do {
            let artifact = try RunStore.load(URL(fileURLWithPath: path))
            if validating { try artifact.validate() }
            return artifact
        } catch { throw ContextError("artifact \(path)", error) }
    }
}

/// An underlying error plus where it happened. It is not a `BenchError`, so `main.swift`'s general `catch` arm prints it and
/// exits 1: wrap only errors that would have taken that arm anyway.
struct ContextError: Error, CustomStringConvertible {
    let context: String
    let underlying: any Error
    init(_ context: String, _ underlying: any Error) { self.context = context; self.underlying = underlying }
    var description: String { "\(context): \(underlying)" }
}

/// Warnings, notes and errors go to stderr, so stdout carries only what a command was asked to produce.
enum Diagnostics {
    static func line(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
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
    /// The seed a run uses when `--seed` is absent; also the seed a seed-dependent case describes before `prepare` has run.
    static let defaultSeed: UInt64 = 42

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

struct IterationOutput: Sendable {
    var gpu: [GPUSample]
    var payload: (any Sendable)?
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

/// `@unchecked Sendable` is sound because every stored property is `let`, `MTLDevice` is documented thread-safe, and the remaining fields are `Sendable` value types.
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
    case registry(String)
    case probeSetup(String)
    case outputExists(String)
    var description: String {
        switch self {
        case .metalUnavailable: return "Metal device unavailable"
        case .routeMismatch(let e, let t): return "route mismatch: expected \(e); telemetry \(t)"
        case .usage(let m): return m
        case .registry(let m): return "registry error: \(m)"
        case .probeSetup(let m): return "GPU probe setup failed: \(m)"
        case .outputExists(let path): return "output path already exists and is never overwritten: \(path)"
        }
    }

    /// Spec §7: caller-input errors and startup errors (duplicate registry IDs) exit 2; every other failure exits 1.
    var exitCode: Int32 {
        switch self {
        case .usage, .registry: return 2
        case .metalUnavailable, .routeMismatch, .probeSetup, .outputExists: return 1
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
        // `prepare` runs before the record exists so `identity` is read exactly once, afterwards. Contract on adapters (unverified,
        // no test): one whose identity depends on the run seed publishes the seeded identity before anything in its `prepare` can
        // throw (`L2BatchDistanceCase` does), so failed and unavailable records carry the run's seed and digest too; a struct
        // adapter returns its registration-time identity either way.
        let preparation: Result<any PreparedCase, any Error>
        do { preparation = .success(try await c.prepare(context)) } catch { preparation = .failure(error) }
        var record = CaseRecord(id: c.id.rawValue, identity: c.identity, status: .measured, failureReason: nil,
                                samples: o.samples, iterations: o.iterations, warmupUsed: [], unitCount: c.unitCount,
                                timing: Timing(wall: nil, commandBuffer: nil, encoder: nil), correctness: nil, resources: nil, notes: [])
        if o.probeSlowdown > 0 { record.notes.append("probe-slowdown=\(o.probeSlowdown)") }
        let clock = ContinuousClock()
        func ns(_ d: Duration) -> Double { Double(d.components.seconds) * 1e9 + Double(d.components.attoseconds) / 1e9 }

        do {
            let prepared = try preparation.get()
            var wall: [[Double]] = [], cb: [[Double]] = [], enc: [[Double]] = []
            var lastOutput: IterationOutput?
            var anyCorrectnessFailed = false
            for k in 0..<o.samples {
                try await prepared.resetForSample()
                var warm: [Double] = []
                repeat {
                    let start = clock.now
                    _ = try await prepared.runOnce()
                    warm.append(ns(clock.now - start))
                } while !policy.shouldStop(after: warm)
                record.warmupUsed.append(warm.count)
                // Visibility only: the same policy with its maximum lifted says whether the CoV rule was met or the cap ended the warm-up.
                var uncapped = policy
                uncapped.maximum = .max
                if !uncapped.shouldStop(after: warm) {
                    record.notes.append("sample \(k + 1): warm-up ended at the policy maximum (\(warm.count)) with CoV ≥ \(policy.cov) over the last \(policy.window) iterations")
                }

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
                wall.append(w); cb.append(b); enc.append(e)   // every sample contributes a row, so `declared` below sees the true per-sample counts
                if let last = lastOutput {
                    // Worst status wins across samples: the latest details are kept, but a `.fail` in any sample is sticky.
                    var verified = try prepared.verify(last)
                    if verified.status == .fail { anyCorrectnessFailed = true; record.notes.append("correctness failed in sample \(k + 1)") }
                    if anyCorrectnessFailed { verified.status = .fail }
                    record.correctness = verified
                }
            }
            func series(_ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                guard raw.count == o.samples, raw.allSatisfy({ $0.count == o.iterations }) else { return nil }
                return Series(raw: raw, summary: Stats.summarize(raw.flatMap { $0 }, bootstrapSeed: o.seed &+ stream), bias: bias)
            }
            /// A declared kind that arrived in the wrong shape is dropped whole (no partial series) and the per-sample counts are recorded in `notes`.
            func declared(_ kind: TimingKind, _ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                if let s = series(raw, bias: bias, stream: stream) { return s }
                record.notes.append("\(kind.rawValue): declared but produced \(raw.map(\.count)) samples per sample against \(o.iterations) iterations; series omitted")
                return nil
            }
            record.timing.wall = series(wall, bias: nil, stream: 1)
            if c.providedTiming.contains(.commandBuffer) { record.timing.commandBuffer = declared(.commandBuffer, cb, bias: commandBufferBias, stream: 2) }
            if c.providedTiming.contains(.encoder) { record.timing.encoder = declared(.encoder, enc, bias: nil, stream: 3) }
            record.resources = prepared.resources()
        } catch BenchError.metalUnavailable {
            record.status = .unavailable; record.failureReason = "Metal device unavailable"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.correctness = nil
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        } catch {
            record.status = .failed; record.failureReason = "\(error)"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.correctness = nil
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        }
        return record
    }
}
```

- [ ] **Step 4: `Cases/Registry.swift`, `Commands/ListCommand.swift`, `Commands/ValidateCommand.swift`**

**`Cases/Registry.swift`**

```swift
import Foundation
import VectorTestKit

/// One line per family (spec §5.4). Families never import each other.
enum Registry {
    static var all: [any BenchCase] {
        let cases: [any BenchCase] = []   // Task 14 changes this to `var` when the first family is appended
        // families are appended in Tasks 14 and 15
        return cases
    }

    /// Glob filtering with fnmatch; empty include means everything.
    static func select(include: [String], exclude: [String]) throws -> [any BenchCase] {
        let all = Self.all
        let dups = CaseRegistryCheck.duplicates(all.map(\.id))
        guard dups.isEmpty else { throw BenchError.registry("duplicate case IDs: \(dups.map(\.rawValue))") }
        func matches(_ id: String, _ globs: [String]) -> Bool { globs.contains { fnmatch($0, id, 0) == 0 } }
        return all.filter { (include.isEmpty || matches($0.id.rawValue, include)) && !matches($0.id.rawValue, exclude) }
    }
}
```

**`Commands/ListCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum ListCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: ["--filter", "--exclude"], flags: [], operands: 0)
        let cases = try Registry.select(include: args.value("--filter").map { [$0] } ?? [], exclude: args.value("--exclude").map { [$0] } ?? [])
        for c in cases { print("\(c.id.rawValue)  route=\(c.identity.route)  timing=\(c.providedTiming.map(\.rawValue).sorted().joined(separator: ","))") }
        print("\(cases.count) case(s)")
        return 0
    }
}
```

**`Commands/ValidateCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum ValidateCommand {
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: [], flags: [], operands: 1)
        guard let path = args.subcommand else { throw BenchError.usage("usage: validate-artifact <path>") }
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
       [--out <path>] [--label <text>] [--require-gpu] [--validation-layers] [--probe-slowdown N]
  baseline capture --from <artifact> [--approve] [--allow-dirty]
  compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]
  validate-artifact <path>
"""

let args = Arguments(Array(CommandLine.arguments.dropFirst()))
if args.has("--help") || args.has("-h") { print(usage); exit(0) }
var status: Int32 = 2
do {
    switch args.command {
    case "list": status = try ListCommand.run(args)
    case "validate-artifact": status = try ValidateCommand.run(args)
    case "run": status = try await RunCommand.run(args)
    case "baseline": status = try BaselineCommand.run(args)
    case "compare": status = try CompareCommand.run(args)
    default: print(usage); status = args.command == "help" ? 0 : 2
    }
} catch let error as BenchError {
    Diagnostics.line("error: \(error)")
    status = error.exitCode
} catch {
    Diagnostics.line("error: \(error)")
    status = 1
}
exit(status)
```

Until Tasks 15–16 exist, add these three stubs at the bottom of `main.swift` so the executable compiles; each is deleted when its real file lands:

```swift
// Temporary stubs so the executable links before Tasks 15–16 land; each is deleted when its real file arrives.
enum RunCommand { static func run(_ args: Arguments) async throws -> Int32 { print("run: not implemented until Task 15"); return 2 } }
enum BaselineCommand { static func run(_ args: Arguments) throws -> Int32 { print("baseline: not implemented until Task 16"); return 2 } }
enum CompareCommand { static func run(_ args: Arguments) throws -> Int32 { print("compare: not implemented until Task 16"); return 2 } }
```

- [ ] **Step 6: `Tests/VectorAccelerateTests/Harness/ErrorDescriptionLintTests.swift`**

```swift
import XCTest

/// Error-rendering rule: the bench CLI prints a caught error by interpolating it (`"error: \(error)"`), so an `Error` type
/// without a description reaches the user as its reflected case (`unsupportedSchemaVersion(7)`). Every type declared as
/// conforming to `Error` in `Sources/VectorTestKit`, and likewise in `Sources/VectorAccelerateBench`, must be declared
/// `CustomStringConvertible` somewhere in the same directory.
///
/// `LocalizedError` is deliberately not accepted as a description: interpolation does not consult `errorDescription`
/// (`testInterpolationIgnoresLocalizedErrorDescription` pins that), so such a type would still print as its reflected case.
///
/// Contract — the declaration shapes the scanner recognises; anything outside this list is not seen. Each bullet cites the rows
/// of `testScannerFixtures` that pin it, and `testContractCitesExactlyTheFixtureRows` checks that every cited label is a row
/// and every row is cited (`FixtureCitations`):
/// - a nominal head `enum|struct|class|actor Name[<…>]: A, B, … {` whose inheritance clause lists `Error`, `Swift.Error` or
///   `LocalizedError` [pins: enum-error-flagged, struct-inline-description, class-swift-error-flagged, actor-flagged,
///   actor-described, localized-error-flagged]; the clause may span lines, ends at the first `{` or at a `where`, and drops
///   attributes such as `@unchecked` [pins: multiline-clause-flagged, generic-where-flagged, unchecked-attribute-flagged];
/// - an extension head `extension [Outer.]Name: A, B, … {` that adds one of those, or that adds `CustomStringConvertible`, in
///   the same file or another [pins: extension-adds-error-flagged, extension-adds-description, nested-extension-described,
///   description-in-other-file];
/// - a type is keyed by the last component of its name, so two types sharing a simple name are not told apart [pins:
///   nested-extension-described, shared-simple-name-merged];
/// - `//` comments are removed before matching [pins: comments-unseen]; `/* */` blocks are scanned as code
///   [pins: block-comment-scanned]; string literals are scanned as code and a `//` inside one truncates that line, hiding what
///   follows it [pins: string-slash-truncates];
/// - not recognised: `Error` inherited through another protocol (`protocol KitError: Error`) or through a typealias [pins:
///   protocol-inherited-unseen, typealias-unseen]; a type declared in another module [pins: none — all of a row's sources scan
///   as one directory, i.e. one module, so another module cannot appear in a fixture]; a `description` property without the
///   `CustomStringConvertible` conformance, because interpolation ignores it [pins: description-without-conformance]; and, as
///   non-matches, a name that merely contains `Error`, a `CustomStringConvertible` that is not an `Error`, and the `class`
///   keyword on members [pins: error-in-name-unseen, described-not-error, class-keyword-members-unseen].
final class ErrorDescriptionLintTests: XCTestCase {
    struct Scan: Equatable {
        var errorTypes: Set<String> = []
        var describedTypes: Set<String> = []
        var undescribed: [String] { errorTypes.subtracting(describedTypes).sorted() }
    }

    // Group 1: the type's name, possibly qualified (`Outer.Inner`). Group 2: the inheritance clause, up to the first `{`.
    private static let head = try! NSRegularExpression(
        pattern: #"\b(?:enum|struct|class|actor|extension)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*(?:<[^>{]*>)?\s*:\s*([^{]*)\{"#)

    /// The simple names listed in an inheritance clause: `where` clause cut off, attributes (`@unchecked`) and module
    /// qualifiers (`Swift.`) dropped.
    static func conformances(in clause: String) -> Set<String> {
        let list = clause.range(of: #"\bwhere\b"#, options: .regularExpression).map { String(clause[..<$0.lowerBound]) } ?? clause
        return Set(list.split(separator: ",").compactMap { entry in
            entry.split(whereSeparator: \.isWhitespace).last { !$0.hasPrefix("@") }?.split(separator: ".").last.map(String.init)
        })
    }

    static func scan(_ sources: [String]) -> Scan {
        var result = Scan()
        for source in sources {
            let code = source.split(separator: "\n", omittingEmptySubsequences: false)
                .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }
                .joined(separator: "\n")
            for match in head.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                // Both groups are mandatory in `head`, so a match always has both ranges and this never skips one.
                guard let nameRange = Range(match.range(at: 1), in: code), let clauseRange = Range(match.range(at: 2), in: code) else { continue }
                let qualified = String(code[nameRange])
                let name = qualified.split(separator: ".").last.map(String.init) ?? qualified
                let listed = conformances(in: String(code[clauseRange]))
                if !listed.isDisjoint(with: ["Error", "LocalizedError"]) { result.errorTypes.insert(name) }
                if listed.contains("CustomStringConvertible") { result.describedTypes.insert(name) }
            }
        }
        return result
    }

    /// The fixture rows, keyed by the labels the doc comment's `[pins: …]` citations name.
    static let fixtures: [(label: String, sources: [String], errorTypes: Set<String>, undescribed: [String])] = [
        ("enum-error-flagged", ["enum A: Error { case x }"], ["A"], ["A"]),
        ("extension-adds-description", ["public enum B: Error, Equatable {\n    case x\n}\n\nextension B: CustomStringConvertible {\n}"], ["B"], []),
        ("struct-inline-description", ["struct C: Error, CustomStringConvertible { var description: String { \"c\" } }"], ["C"], []),
        ("class-swift-error-flagged", ["final class D: Swift.Error {}"], ["D"], ["D"]),
        ("extension-adds-error-flagged", ["struct E {}\nextension E: Error {}"], ["E"], ["E"]),
        ("localized-error-flagged", ["enum F: LocalizedError { case x }"], ["F"], ["F"]),
        ("multiline-clause-flagged", ["public enum G:\n    Error,\n    Equatable {\n    case x\n}"], ["G"], ["G"]),
        ("description-in-other-file", ["enum H: Error { case x }", "extension H: CustomStringConvertible { var description: String { \"h\" } }"], ["H"], []),
        ("nested-extension-described", ["enum Outer { enum I: Error { case x } }\nextension Outer.I: CustomStringConvertible { }"], ["I"], []),
        ("generic-where-flagged", ["struct J<T>: Error where T: Sendable { }"], ["J"], ["J"]),
        ("unchecked-attribute-flagged", ["final class K: @unchecked Sendable, Error { }"], ["K"], ["K"]),
        ("description-without-conformance", ["enum L: Error { case x; var description: String { \"l\" } }"], ["L"], ["L"]),
        ("actor-flagged", ["actor Boom: Error {}"], ["Boom"], ["Boom"]),
        ("actor-described", ["actor Calm: Error, CustomStringConvertible { var description: String { \"calm\" } }"], ["Calm"], []),
        ("shared-simple-name-merged", ["enum Outer { enum Dup: Error { case x } }\nstruct Dup: CustomStringConvertible { var description: String { \"\" } }"], ["Dup"], []),
        ("comments-unseen", ["// enum M: Error { case x }\n/// enum N: Error { case x }"], [], []),
        ("block-comment-scanned", ["/* enum Z: Error { case x } */"], ["Z"], ["Z"]),
        ("string-slash-truncates", ["let url = \"http://x\"; enum W: Error { case x }"], [], []),
        ("protocol-inherited-unseen", ["protocol KitError: Error {}\nenum X: KitError { case x }"], [], []),
        ("typealias-unseen", ["typealias Failure = Error\nenum Y: Failure { case x }"], [], []),
        ("error-in-name-unseen", ["enum ErrorKind: String, Codable { case error }"], [], []),
        ("described-not-error", ["struct P: CustomStringConvertible { var description: String { \"p\" } }"], [], []),
        ("class-keyword-members-unseen", ["final class Q { class var shared: Q { Q() }\n    class func make() -> Q { Q() } }"], [], [])
    ]

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation fails here.
    func testScannerFixtures() {
        for row in Self.fixtures {
            let scan = Self.scan(row.sources)
            XCTAssertEqual(scan.errorTypes, row.errorTypes, "\(row.label): error types in: \(row.sources)")
            XCTAssertEqual(scan.undescribed, row.undescribed, "\(row.label): undescribed in: \(row.sources)")
        }
    }

    /// The doc comment above cannot claim a pin that does not exist: its `[pins: …]` labels are exactly the fixture rows.
    func testContractCitesExactlyTheFixtureRows() throws {
        try FixtureCitations.assertContractCitesExactlyTheFixtureRows(ofHarnessFile: "ErrorDescriptionLintTests.swift",
                                                                       fixtureLabels: Self.fixtures.map(\.label))
    }

    private enum LocalizedOnly: LocalizedError {
        case broken
        var errorDescription: String? { "a readable message" }
    }

    /// Why `LocalizedError` is not accepted: the interpolation the CLI uses prints the reflected case, not `errorDescription`.
    func testInterpolationIgnoresLocalizedErrorDescription() {
        let message = "\(LocalizedOnly.broken as any Error)"
        XCTAssertEqual(message, "broken")
        XCTAssertFalse(message.contains("a readable message"))
    }

    func testEveryErrorTypeHasADescription() throws {
        // The known types prove the scan is not vacuous: a scanner change that stops seeing one of these real declarations
        // fails here. Renaming or removing one of them means updating this table.
        let directories: [(path: String, known: Set<String>)] = [
            ("Sources/VectorTestKit", ["ArtifactValidationError", "CaseIDError", "RecallError", "RunStoreError"]),
            ("Sources/VectorAccelerateBench", ["BenchError", "ContextError"])
        ]
        for (path, known) in directories {
            let files = RepoPaths.swiftFiles(under: path)
            XCTAssertFalse(files.isEmpty, "\(path) must contain Swift files")
            let scan = Self.scan(try files.map { try String(contentsOf: $0, encoding: .utf8) })
            XCTAssertTrue(scan.errorTypes.isSuperset(of: known), "\(path): the scanner no longer sees \(known.subtracting(scan.errorTypes).sorted())")
            XCTAssertEqual(scan.undescribed, [], "\(path): Error types without a CustomStringConvertible conformance print as raw cases in the CLI")
        }
    }
}
```

Run: `swift test --filter ErrorDescriptionLintTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1`
Expected: `Executed 4 tests, with 0 failures`. To see it bite, delete one kit `extension …Error: CustomStringConvertible` and rerun: `testEveryErrorTypeHasADescription` names the type.

- [ ] **Step 6b: `Tests/VectorAccelerateTests/Harness/CollectionInterpolationLintTests.swift`** (added by Task 17's fix round)

Closes the class whose second occurrence Task 17 fixed in `RunCommand` (the first was `CompareCommand`, Task 16 review): a collection interpolated whole prints its debug literal. Same shape as the error-description lint: a mechanical scan of both source directories with an explicit contract, a fixture table that pins every recognised and every deliberately unrecognised shape, and a real-source test. A bare `\(name)` resolves against the collection names declared in its own file (so a `String` named `raw` in `CaseID.swift` is not `Series.raw`); a dotted `\(a.name)` resolves against every `let`/`var` collection name in the scanned set.

```swift
import XCTest

/// Rendering rule: interpolating an array or dictionary whole (`"reasons: \(preflight.reasons)"`) prints Swift's debug
/// literal — `["drift 6.5% exceeds 5.0%", "thermal state end=fair (not nominal)"]` — which is not a message. `CompareCommand`
/// (Task 16 review) and `RunCommand` (Task 17) each did it once; this closes the class. In `Sources/VectorTestKit` and
/// `Sources/VectorAccelerateBench`, no string-interpolation segment may consist of a bare or dotted reference to a name
/// declared with an explicit array or dictionary type. Render the elements instead (`.joined(separator: "; ")`).
///
/// Contract — what the scanner recognises; anything outside this list is not seen. Each bullet cites the rows of
/// `testScannerFixtures` that pin it, and `testContractCitesExactlyTheFixtureRows` checks that every cited label is a row and
/// every row is cited (`FixtureCitations`):
/// - a collection name is the `name` in `let name: [` or `var name: [` (stored, computed and local) [pins: let-array-flagged,
///   var-dict-flagged, var-computed-flagged, self-dotted-flagged], or in a parameter `[label ]name: [` of a `func` or `init`
///   declaration's parameter clause — the balanced parenthesised text after the name, so multi-line clauses and defaults with
///   parentheses are read [pins: func-params-flagged, init-multiline-default-flagged]; a call-site `.init(` is not a
///   declaration and contributes nothing [pins: call-site-init-unseen];
/// - the type must START with `[`: `[T]`, `[[T]]`, `[K: V]` and `[T]?` count [pins: let-array-flagged, nested-optional-flagged,
///   var-dict-flagged], while `Set<T>`, `Array<T>`, `Dictionary<K, V>`, `inout [T]`, every inferred type and every scalar do
///   not [pins: set-array-generic-unseen, dict-generic-unseen, inout-unseen, inferred-unseen, scalar-unflagged];
/// - a violation is a segment `\(chain)` whose whole content is an identifier chain — `reasons`, `preflight.reasons`,
///   `self.deltas`, `o.i.zs` — with a collection name as its LAST component; whitespace inside the parentheses is allowed
///   [pins: let-array-flagged, prefix-runcommand-flagged, self-dotted-flagged, dotted-three-flagged, padded-segment-flagged].
///   A chain that ends in a call, a subscript or another member (`.joined(…)`, `.map(…)`, `[0]`, `.count`, `.first`,
///   `.isEmpty`) is not one, nor is an inline literal or a call result [pins: operations-unflagged, literal-call-unflagged];
/// - a bare chain resolves against the names declared in the SAME file; a dotted chain's last component resolves against every
///   `let`/`var` collection name in all scanned files, since a member may be declared anywhere [pins: bare-same-file-only,
///   dotted-any-file-flagged]. Within a file two declarations sharing a name are not told apart, so a scalar that shares its
///   name with a collection declared in the same file is flagged too (rename one of them) [pins: shared-name-both-flagged];
/// - not seen: closure parameters, subscript parameters, enum associated values and tuple-type elements [pins:
///   closure-param-unseen, subscript-param-unseen, enum-case-unseen, tuple-element-unseen]; the parameters of operator
///   functions (`func == (`) and of functions whose generic clause nests angle brackets (`func g<T: P<U>>(`) [pins:
///   operator-func-unseen, nested-generic-unseen];
/// - `//` comments are removed before matching, for declarations and for uses [pins: comments-unseen]; `/* */` blocks are
///   scanned as code [pins: block-comment-scanned]; string literals are scanned as code and a `//` inside one truncates that
///   line, hiding what follows it [pins: string-slash-truncates].
final class CollectionInterpolationLintTests: XCTestCase {
    /// Group 1: the name, in `let name: [` / `var name: [`.
    private static let declared = try! NSRegularExpression(pattern: #"\b(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\["#)
    /// Opens a `func` or `init` declaration's parameter clause (an `init` preceded by `.` is a call); the clause ends at the
    /// parenthesis that balances this one.
    private static let clauseHead = try! NSRegularExpression(pattern: #"\b(?:func\s+[A-Za-z_][A-Za-z0-9_]*|(?<!\.)init[?!]?)\s*(?:<[^>]*>)?\s*\("#)
    /// Group 1: the name, in `[label ]name: [` at the start of a clause or after a comma.
    private static let parameter = try! NSRegularExpression(pattern: #"(?:^|,)\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\["#)
    /// Group 1: the identifier chain in `\(chain)`, with optional whitespace inside the parentheses.
    private static let segment = try! NSRegularExpression(pattern: #"\\\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\)"#)

    /// `//` comments removed, line structure kept (so line numbers below are source line numbers).
    static func stripped(_ source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }
            .joined(separator: "\n")
    }

    private static func captures(_ regex: NSRegularExpression, in code: String) -> [String] {
        regex.matches(in: code, range: NSRange(code.startIndex..., in: code))
            .compactMap { Range($0.range(at: 1), in: code).map { String(code[$0]) } }
    }

    /// Names declared with an explicit collection type in one stripped file: `let`/`var` declarations and `func`/`init`
    /// parameters.
    static func collectionNames(in code: String) -> Set<String> {
        var names = Set(captures(declared, in: code))
        for head in clauseHead.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
            guard let open = Range(head.range, in: code) else { continue }
            var depth = 1, index = open.upperBound
            while index < code.endIndex, depth > 0 {
                if code[index] == "(" { depth += 1 } else if code[index] == ")" { depth -= 1 }
                index = code.index(after: index)
            }
            let clause = String(code[open.upperBound..<(depth == 0 ? code.index(before: index) : index)])
            names.formUnion(captures(parameter, in: clause))
        }
        return names
    }

    /// One line per violation: `<file>:<line>: \(chain)`, in file order then source order.
    static func findings(in files: [(name: String, code: String)]) -> [String] {
        let perFile = files.map { collectionNames(in: $0.code) }
        let declaredAnywhere = Set(files.flatMap { captures(declared, in: $0.code) })
        var out: [String] = []
        for (file, local) in zip(files, perFile) {
            for match in segment.matches(in: file.code, range: NSRange(file.code.startIndex..., in: file.code)) {
                guard let whole = Range(match.range, in: file.code), let chainRange = Range(match.range(at: 1), in: file.code) else { continue }
                let chain = String(file.code[chainRange])
                let components = chain.split(separator: ".")
                guard let last = components.last.map(String.init) else { continue }
                let hit = components.count == 1 ? local.contains(last) : declaredAnywhere.contains(last)
                if hit {
                    let line = file.code[..<whole.lowerBound].filter { $0 == "\n" }.count + 1
                    out.append("\(file.name):\(line): \\(\(chain))")
                }
            }
        }
        return out
    }

    /// Fixture sources are named `f0.swift`, `f1.swift`, … in order.
    private static func scan(_ sources: [String]) -> (names: Set<String>, findings: [String]) {
        let files = sources.enumerated().map { (name: "f\($0.offset).swift", code: stripped($0.element)) }
        return (Set(files.flatMap { collectionNames(in: $0.code) }), findings(in: files))
    }

    /// The fixture rows, keyed by the labels the doc comment's `[pins: …]` citations name. `names` is the union of the
    /// collection names found in the row's files; `findings` the violations, `f<n>.swift:<line>: \(chain)`.
    static let fixtures: [(label: String, sources: [String], names: Set<String>, findings: [String])] = [
        ("let-array-flagged", ["let reasons: [String] = []\nprint(\"reasons: \\(reasons)\")"], ["reasons"], ["f0.swift:2: \\(reasons)"]),
        ("var-dict-flagged", ["var flags: [String: String] = [:]\n_ = \"\\(flags)\""], ["flags"], ["f0.swift:2: \\(flags)"]),
        ("var-computed-flagged", ["struct S {\n    var xs: [Int] { [1] }\n    var text: String { \"\\(xs)\" }\n}"], ["xs"], ["f0.swift:3: \\(xs)"]),
        ("nested-optional-flagged", ["let raw: [[Double]] = []\nvar extra: [Int]? = nil\n_ = \"\\(raw) \\(extra)\""], ["raw", "extra"],
         ["f0.swift:3: \\(raw)", "f0.swift:3: \\(extra)"]),
        ("func-params-flagged", ["func f(_ items: [Int], with list: [Int], n: Int) -> String { \"\\(items) \\(list) \\(n)\" }"], ["items", "list"],
         ["f0.swift:1: \\(items)", "f0.swift:1: \\(list)"]),
        ("init-multiline-default-flagged",
         ["struct S {\n    init(\n        a: [Int] = Array(repeating: 0, count: 2),\n        b: [String: Int]\n    ) { _ = \"\\(a) \\(b)\" }\n}"],
         ["a", "b"], ["f0.swift:5: \\(a)", "f0.swift:5: \\(b)"]),
        ("padded-segment-flagged", ["let ids: [Int] = []\n_ = \"\\( ids )\""], ["ids"], ["f0.swift:2: \\(ids)"]),
        ("self-dotted-flagged", ["struct S { var deltas: [Int] = []; var text: String { \"\\(self.deltas)\" } }"], ["deltas"], ["f0.swift:1: \\(self.deltas)"]),
        ("dotted-three-flagged", ["struct O { var i: I }\nstruct I { var zs: [Int] }", "func show(_ o: O) -> String { \"\\(o.i.zs)\" }"], ["zs"],
         ["f1.swift:1: \\(o.i.zs)"]),
        // The pre-fix `RunCommand` line verbatim, with the declaration in another file as in the real tree.
        ("prefix-runcommand-flagged",
         ["public struct PreflightRecord { public var reasons: [String] }",
          "print(\"key: \\(metadata.environmentKey)  quiet: \\(preflight.quiet)\" + (preflight.reasons.isEmpty ? \"\" : \"  reasons: \\(preflight.reasons)\"))"],
         ["reasons"], ["f1.swift:1: \\(preflight.reasons)"]),
        ("operations-unflagged",
         ["var notes: [String] = []\n_ = \"\\(notes.joined(separator: \"; \")) \\(notes.count) \\(notes[0]) \\(notes.first) \\(notes.map(\\.count)) \\(notes.isEmpty)\""],
         ["notes"], []),
        ("literal-call-unflagged", ["let xs: [Int] = []\n_ = \"\\([1, 2]) \\(f()) \\(xs.map { $0 })\""], ["xs"], []),
        ("bare-same-file-only", ["struct Series { var raw: [[Double]] }", "func parse(_ raw: String) -> String { \"bad: \\(raw)\" }"], ["raw"], []),
        ("dotted-any-file-flagged", ["struct Series { var raw: [[Double]] }", "func show(_ s: Series) -> String { \"\\(s.raw)\" }"], ["raw"],
         ["f1.swift:1: \\(s.raw)"]),
        ("shared-name-both-flagged", ["func a(_ values: [Int]) -> String { \"\\(values)\" }\nfunc b(_ values: Int) -> String { \"\\(values)\" }"], ["values"],
         ["f0.swift:1: \\(values)", "f0.swift:2: \\(values)"]),
        ("set-array-generic-unseen", ["let s: Set<String> = []\nlet g: Array<Int> = []\n_ = \"\\(s) \\(g)\""], [], []),
        ("dict-generic-unseen", ["let d: Dictionary<String, Int> = [:]\n_ = \"\\(d)\""], [], []),
        ("inferred-unseen", ["let inferred = [1, 2]\n_ = \"\\(inferred)\""], [], []),
        ("scalar-unflagged", ["let none: Int = 0\n_ = \"\\(none)\""], [], []),
        ("inout-unseen", ["func k(_ ys: inout [Int]) -> String { \"\\(ys)\" }"], [], []),
        ("closure-param-unseen", ["let c: ([Int]) -> String = { (zs: [Int]) in \"\\(zs)\" }"], [], []),
        ("subscript-param-unseen", ["struct Table { subscript(keys: [String]) -> String { \"\\(keys)\" } }"], [], []),
        ("enum-case-unseen", ["enum E { case list(items: [Int]) }\n_ = \"\\(items)\""], [], []),
        ("tuple-element-unseen", ["let t: (xs: [Int], n: Int) = ([], 0)\n_ = \"\\(t.xs)\""], [], []),
        ("operator-func-unseen", ["func == (lhs: [Int], rhs: [Int]) -> Bool { \"\\(lhs)\" == \"\\(rhs)\" }"], [], []),
        ("nested-generic-unseen", ["func g<T: P<U>>(xs: [T]) -> String { \"\\(xs)\" }"], [], []),
        ("call-site-init-unseen", ["let made = Maker.init(items: [1, 2])\nself.init(items: [])\n_ = \"\\(items)\""], [], []),
        ("comments-unseen", ["// let hidden: [Int] = []\n/// \\(hidden)\nlet shown: [Int] = [] // trailing \\(shown)"], ["shown"], []),
        ("block-comment-scanned", ["/* let hidden: [Int] = [] */\n_ = \"\\(hidden)\""], ["hidden"], ["f0.swift:2: \\(hidden)"]),
        ("string-slash-truncates", ["let urls: [String] = [\"http://x\"]; _ = \"\\(urls)\""], ["urls"], [])
    ]

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation, or starts seeing a false one,
    /// fails here.
    func testScannerFixtures() {
        for row in Self.fixtures {
            let scan = Self.scan(row.sources)
            XCTAssertEqual(scan.names, row.names, "\(row.label): collection names in: \(row.sources)")
            XCTAssertEqual(scan.findings, row.findings, "\(row.label): findings in: \(row.sources)")
        }
    }

    /// The doc comment above cannot claim a pin that does not exist: its `[pins: …]` labels are exactly the fixture rows.
    func testContractCitesExactlyTheFixtureRows() throws {
        try FixtureCitations.assertContractCitesExactlyTheFixtureRows(ofHarnessFile: "CollectionInterpolationLintTests.swift",
                                                                       fixtureLabels: Self.fixtures.map(\.label))
    }

    func testNoCollectionIsInterpolatedWhole() throws {
        let files = try ["Sources/VectorTestKit", "Sources/VectorAccelerateBench"].flatMap { directory -> [(name: String, code: String)] in
            let urls = RepoPaths.swiftFiles(under: directory)
            XCTAssertFalse(urls.isEmpty, "\(directory) must contain Swift files")
            return try urls.map { (name: $0.lastPathComponent, code: Self.stripped(try String(contentsOf: $0, encoding: .utf8))) }
        }
        // The known names prove the scan is not vacuous: a scanner change that stops seeing one of these real declarations
        // fails here. Renaming or removing one of them means updating this list.
        let declaredAnywhere = Set(files.flatMap { Self.collectionNames(in: $0.code) })
        let known: Set<String> = ["reasons", "notes", "warmupUsed", "flags", "raw", "argv"]
        XCTAssertTrue(declaredAnywhere.isSuperset(of: known), "the scanner no longer sees \(known.subtracting(declaredAnywhere).sorted())")
        let violations = Self.findings(in: files)
        XCTAssertEqual(violations, [], "a collection interpolated whole prints its debug literal; render its elements:\n" + violations.joined(separator: "\n"))
    }
}
```

- [ ] **Step 6c: `Tests/VectorAccelerateTests/Harness/FixtureCitations.swift`** (added by Task 17's fix round 1)

Shared by the two scanner lints above. Their doc comments are contracts, and a contract bullet that claims a fixture pin which does not exist was found three times (the `actor` shape in the error-description lint; then two shapes and then two more in the collection-interpolation lint). This closes the class: every `- ` bullet of a lint's leading doc comment ends in `[pins: label-a, label-b]` naming the fixture rows that exercise it (or exactly `[pins: none — <why>]`), every fixture row carries a unique label, and each lint's `testContractCitesExactlyTheFixtureRows` reads its own source and asserts the cited set equals the row set. The parsing lives here once; `FixtureCitationsTests` pins the `none` form and the bullet joining.

```swift
import Foundation
import XCTest

/// The pins contract of the scanner lints (`ErrorDescriptionLintTests`, `CollectionInterpolationLintTests`): every `- ` bullet
/// of a lint's leading doc comment ends in `[pins: label-a, label-b]`, naming the rows of its fixture table that exercise the
/// bullet, or in exactly `[pins: none — <why>]` when no fixture can. `assertContractCitesExactlyTheFixtureRows` reads the
/// lint's own source and fails when a cited label has no row, a row is never cited, a bullet ends without a citation, the
/// fixture table is empty, or a citation is malformed — so a doc comment cannot claim a pin that does not exist.
/// `FixtureCitationsTests` pins the parser and each of those conditions.
enum FixtureCitations {
    struct Malformed: Error, CustomStringConvertible { var description: String }

    /// A label: a lowercase letter, then lowercase letters and digits, groups joined by single hyphens (`actor-flagged`).
    static func isLabel(_ text: String) -> Bool {
        text.range(of: #"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"#, options: .regularExpression) != nil
    }

    /// The first run of `///` lines in `source`, with the `///` removed and nothing else changed (indentation is what tells a
    /// bullet's continuation line from a new paragraph).
    static func leadingDocComment(inSource source: String) -> [String] {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .drop { !$0.hasPrefix("///") }.prefix { $0.hasPrefix("///") }
            .map { String($0.dropFirst(3)) }
    }

    /// The `- ` bullets of a leading doc comment, each joined with its continuation lines (indented by three or more spaces);
    /// any other line ends the bullet.
    static func contractBullets(inLeadingDocComment lines: [String]) -> [String] {
        var bullets: [String] = []
        var current: String?
        for line in lines {
            if line.hasPrefix(" - ") {
                if let open = current { bullets.append(open) }
                current = String(line.dropFirst(3))
            } else if let open = current, line.hasPrefix("   ") {
                current = open + " " + line.trimmingCharacters(in: .whitespaces)
            } else {
                if let open = current { bullets.append(open) }
                current = nil
            }
        }
        if let open = current { bullets.append(open) }
        return bullets
    }

    /// The labels cited by every `[pins: …]` in `text`. `[pins: none — <why>]` with a non-empty why cites nothing; any other
    /// content must be a comma-separated list of labels. Anything else throws `Malformed`.
    static func citedLabels(in text: String) throws -> Set<String> {
        let citation = try NSRegularExpression(pattern: #"\[pins: ([^\]]*)\]"#)
        var labels: Set<String> = []
        for match in citation.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range(at: 1), in: text) else { continue }
            let body = String(text[range])
            if body.hasPrefix("none — "), !body.dropFirst("none — ".count).trimmingCharacters(in: .whitespaces).isEmpty { continue }
            if body.split(separator: " ").first == "none" || body.hasPrefix("none—") || body.hasPrefix("none-") {
                throw Malformed(description: "the no-fixture form is exactly `[pins: none — <why>]`, got `[pins: \(body)]`")
            }
            for part in body.split(separator: ",", omittingEmptySubsequences: false) {
                let label = part.trimmingCharacters(in: .whitespaces)
                guard isLabel(label) else { throw Malformed(description: "`\(label)` is not a label, in `[pins: \(body)]`") }
                labels.insert(label)
            }
        }
        return labels
    }

    /// The problems with one lint's contract, given the lint's source and its fixture labels; empty when the contract is sound.
    /// Reported, in this order: no `- ` bullet at all; a bullet that does not end in a citation; a fixture label that is not a
    /// label; duplicate labels; an empty table (under all-`none` citations an empty table would otherwise pass as ∅ == ∅); a
    /// cited label with no row; a row the contract never cites. A malformed citation throws `Malformed`. Each condition is
    /// pinned by `FixtureCitationsTests.testContractProblemsNameEachCondition`.
    static func contractProblems(fileName: String, source: String, fixtureLabels: [String]) throws -> [String] {
        let doc = leadingDocComment(inSource: source)
        let bullets = contractBullets(inLeadingDocComment: doc)
        var problems: [String] = []
        if bullets.isEmpty { problems.append("\(fileName): no `- ` bullet in the leading doc comment") }
        for bullet in bullets where bullet.range(of: #"\[pins: [^\]]*\][;.]?$"#, options: .regularExpression) == nil {
            problems.append("\(fileName): a contract bullet does not end with a [pins: …] citation: \(bullet.prefix(90))…")
        }
        let cited = try citedLabels(in: doc.joined(separator: " "))
        for label in fixtureLabels where !isLabel(label) { problems.append("\(fileName): fixture label `\(label)` is not a label") }
        let rows = Set(fixtureLabels)
        if rows.count != fixtureLabels.count { problems.append("\(fileName): duplicate fixture labels") }
        if rows.isEmpty { problems.append("\(fileName): no fixture rows") }
        let unrowed = cited.subtracting(rows).sorted(), uncited = rows.subtracting(cited).sorted()
        if !unrowed.isEmpty { problems.append("\(fileName): cited in the contract but no fixture row: \(unrowed.joined(separator: ", "))") }
        if !uncited.isEmpty { problems.append("\(fileName): fixture rows the contract never cites: \(uncited.joined(separator: ", "))") }
        return problems
    }

    /// Reads `Tests/VectorAccelerateTests/Harness/<fileName>` and fails on every problem `contractProblems` reports for it.
    static func assertContractCitesExactlyTheFixtureRows(ofHarnessFile fileName: String, fixtureLabels: [String],
                                                         file: StaticString = #filePath, line: UInt = #line) throws {
        let url = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/\(fileName)")
        let problems = try contractProblems(fileName: fileName, source: try String(contentsOf: url, encoding: .utf8), fixtureLabels: fixtureLabels)
        XCTAssertEqual(problems, [], problems.joined(separator: "\n"), file: file, line: line)
    }
}

/// Pins the parser above on strings, so a change that loosens the `none` form or the bullet joining fails here.
final class FixtureCitationsTests: XCTestCase {
    func testCitedLabelsAcceptsListsAndTheNoneFormOnly() throws {
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "x [pins: actor-flagged, dict2-unseen] y [pins: a1]"), ["actor-flagged", "dict2-unseen", "a1"])
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "[pins: none — a fixture is one source string]"), [])
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "no citation here"), [])
        for malformed in ["[pins: none]", "[pins: none —]", "[pins: none — ]", "[pins: none - why]", "[pins: None — why]", "[pins: none—why]",
                          "[pins: Actor-Flagged]", "[pins: a, ]", "[pins: ]", "[pins: a b]", "[pins: a--b]", "[pins: -a]", "[pins: 1a]"] {
            XCTAssertThrowsError(try FixtureCitations.citedLabels(in: malformed), malformed)
        }
    }

    func testContractBulletsJoinContinuationLinesAndStopAtParagraphs() throws {
        let source = "import XCTest\n\n/// Intro line.\n///\n/// Contract:\n/// - first bullet [pins: a]\n///   continued [pins: b];\n"
            + "/// - second\n///   [pins: none — why].\n/// Trailing paragraph [pins: c].\nfinal class T {}\n/// - after the class [pins: d]\n"
        let doc = FixtureCitations.leadingDocComment(inSource: source)
        XCTAssertEqual(FixtureCitations.contractBullets(inLeadingDocComment: doc), ["first bullet [pins: a] continued [pins: b];", "second [pins: none — why]."])
        // Citations anywhere in the leading block count; the block ends at the first non-`///` line.
        XCTAssertEqual(try FixtureCitations.citedLabels(in: doc.joined(separator: " ")), ["a", "b", "c"])
    }

    /// Each condition `contractProblems` reports, on synthetic sources — in particular an empty fixture table under all-`none`
    /// citations is a problem, not a vacuous ∅ == ∅ match.
    func testContractProblemsNameEachCondition() throws {
        func problems(_ source: String, _ labels: [String]) throws -> [String] {
            try FixtureCitations.contractProblems(fileName: "L.swift", source: source, fixtureLabels: labels)
        }
        let sound = "/// Rule.\n/// - one [pins: a, b];\n/// - two [pins: none — why].\nfinal class L {}\n"
        XCTAssertEqual(try problems(sound, ["a", "b"]), [])
        XCTAssertEqual(try problems("/// Rule.\n/// - two [pins: none — why].\nfinal class L {}\n", []), ["L.swift: no fixture rows"])
        XCTAssertEqual(try problems("/// Rule only.\nfinal class L {}\n", ["a"]),
                       ["L.swift: no `- ` bullet in the leading doc comment", "L.swift: fixture rows the contract never cites: a"])
        XCTAssertEqual(try problems("/// - one without a citation\nfinal class L {}\n", ["a"]),
                       ["L.swift: a contract bullet does not end with a [pins: …] citation: one without a citation…",
                        "L.swift: fixture rows the contract never cites: a"])
        XCTAssertEqual(try problems(sound, ["a", "b", "Bad_Label"]),
                       ["L.swift: fixture label `Bad_Label` is not a label", "L.swift: fixture rows the contract never cites: Bad_Label"])
        XCTAssertEqual(try problems(sound, ["a", "b", "a"]), ["L.swift: duplicate fixture labels"])
        XCTAssertEqual(try problems(sound, ["a"]), ["L.swift: cited in the contract but no fixture row: b"])
        XCTAssertEqual(try problems(sound, ["a", "b", "c"]), ["L.swift: fixture rows the contract never cites: c"])
        XCTAssertThrowsError(try problems("/// - one [pins: none]\nfinal class L {}\n", ["a"]))
    }
}
```

- [ ] **Step 7: Build and exercise**

```bash
swift build 2>&1 | tail -2                                              # Build complete!
swift run -q VectorAccelerateBench list; echo "exit=$?"                  # "0 case(s)", exit=0
swift run -q VectorAccelerateBench nonsense; echo "exit=$?"              # usage, exit=2
swift run -q VectorAccelerateBench validate-artifact /nonexistent; echo "exit=$?"   # "invalid: ...", exit=1
swift run -q VectorAccelerateBench list --help; echo "exit=$?"           # usage, exit=0
swift run -q VectorAccelerateBench list --filter; echo "exit=$?"         # "error: option --filter requires a value" on stderr, exit=2
swift run -q VectorAccelerateBench list --fliter 'probe.*'; echo "exit=$?"   # "error: unknown option --fliter for `list`; it takes: --exclude, --filter" on stderr, exit=2
swift run -q VectorAccelerateBench list extra; echo "exit=$?"            # "error: unexpected argument 'extra' for `list`; see --help" on stderr, exit=2
```

- [ ] **Step 8: Commit**

```bash
git add Sources/VectorAccelerateBench Tests/VectorAccelerateTests/Harness/ErrorDescriptionLintTests.swift
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
- Produces: `enum MetalFingerprint { static func gather(device: (any MTLDevice)?) -> GPUInfo? ; static func gpuCoreCount() -> Int?; static func metalToolchainVersion() -> String? }`; `final class GPUProbe { init(device:) throws; static let bytes = 134_217_728; func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double); func copyGBps(warmup: Int = 5, measured: Int = 10) throws -> Double; func outputMatchesInput() -> Bool; func resetDestination(); var staticThreadgroupBytes: Int; var maxThreadgroupBytes: Int }`; `enum ProbeCases { static var all: [any BenchCase] }`.

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

    /// The parsed version token from the first line of `xcrun -sdk macosx metal --version`: "32023.883" out of
    /// "Apple metal version 32023.883 (metalfe-32023.883)"; nil when xcrun fails or the line has no `version` token.
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
/// `@unchecked Sendable`: every stored property is `let` and `MTLDevice`/`MTLCommandQueue`/`MTLComputePipelineState` are documented thread-safe; the buffer contents are unguarded, so this is an unverified contract (no test) that one probe is driven sequentially — `Runner.run` awaits each `runOnce` before the next.
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
        dst[gid] = v + sink * p.zero;   // p.zero is 0.0f from the CPU; the compiler cannot fold it (exercised by the M0 slowdown run,
                                        // evidence in docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md)
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
        guard let function = library.makeFunction(name: "copy_f32x4") else { throw BenchError.probeSetup("kernel function copy_f32x4 not found in the compiled probe library") }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let src = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let dst = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let params = device.makeBuffer(length: 16, options: .storageModeShared) else { throw BenchError.metalUnavailable }
        self.queue = queue; self.src = src; self.dst = dst; self.params = params
        src.label = "probe.src"; dst.label = "probe.dst"; params.label = "probe.params"
        let p = src.contents().bindMemory(to: Float32.self, capacity: Self.floatCount)
        // This fill pattern is the case identity: `GPUCopyProbeCase.identity` (ProbeCases.swift) digests the string "pattern:(i&0xFFFF)*0.5+1" and labels the data "sawtooth_f32_(i&0xFFFF)*0.5+1"; change all three together.
        for i in 0..<Self.floatCount { p[i] = Float32(i & 0xFFFF) * 0.5 + 1.0 }
        resetDestination()
    }

    /// Zeroes `dst` so a later sample's `outputMatchesInput()` proves that sample's dispatch wrote it, not an earlier one.
    func resetDestination() { memset(dst.contents(), 0, Self.bytes) }

    /// One command buffer, one dispatch, synchronous wait. Returns the buffer's GPU start/end in host seconds.
    func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double) {
        struct ProbeParams { var n4: UInt32; var loops: UInt32; var zero: Float }
        var pp = ProbeParams(n4: UInt32(Self.float4Count), loops: UInt32(max(loops, 1)), zero: 0)
        precondition(MemoryLayout<ProbeParams>.size == 12, "ProbeParams must match the MSL struct")
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

    /// GB/s counting read plus write, like `PreflightProbe.cpuCopyGBps`: 2 × bytes / median seconds / 1e9.
    /// Warm-up ends only when BOTH `warmup` dispatches AND `warmupSeconds` of wall time have passed; the reading is the median of
    /// `measured` dispatches, which stays on the settled level as long as fewer than half of them (25 of 50) are slow.
    ///
    /// Why, from docs/superpowers/research/2026-09-18-preflight-probe-characterization.md (81 series, 13,500 dispatches, one
    /// M3 Max): after GPU idle — seen at process start and after gaps from 0.1 s to 5 s, in 56 of the 81 series — the first
    /// dispatches run slow (median 1.88×, range 1.08–2.40× the settled time) and reach the settled level after at most 41
    /// dispatches, or 55.5 ms of wall time in the 36 wall-stamped series. `WarmupPolicy` cannot find that point: the slow plateau is itself steady (CoV < 5 %), so the
    /// policy stopped on it at n = 3 in 3 of 45 series. A median of 10 settled dispatches strayed up to 4.66 % from its series'
    /// level, a median of 50 up to 1.85 %, against a 5 % quiet gate on the difference of two readings. Whether the ramp is bound
    /// by dispatch count or by wall time is not established (no series varied the dispatch duration): the time floor is a hedge
    /// that covers a time-bound ramp on a GPU that finishes 50 dispatches much sooner, and costs at most ≈ 0.1 s per call if
    /// the ramp is count-bound. Explicit, unverified contract on any other machine: the executable has no test target, so
    /// nothing exercises this.
    func copyGBps(warmup: Int = 50, warmupSeconds: Double = 0.1, measured: Int = 50) throws -> Double {
        precondition(warmupSeconds.isFinite && warmupSeconds >= 0, "warmupSeconds must be finite and non-negative")
        precondition(measured >= 1, "copyGBps needs at least one measured dispatch")
        let clock = ContinuousClock()
        let warmupStart = clock.now
        var warmed = 0
        while warmed < warmup || clock.now - warmupStart < .seconds(warmupSeconds) { _ = try dispatch(loops: 1); warmed += 1 }
        var secs: [Double] = []
        for _ in 0..<measured { let t = try dispatch(loops: 1); secs.append(t.endSeconds - t.startSeconds) }
        let median = Stats.percentile(sorted: secs.sorted(), p: 0.5)
        return median > 0 ? 2.0 * Double(Self.bytes) / median / 1e9 : 0
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
                     route: "gpu.probe_copy", variant: nil, dataDescriptor: "sawtooth_f32_(i&0xFFFF)*0.5+1")
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
        func resetForSample() async throws { probe.resetDestination() }
    }
}

enum ProbeCases {
    static var all: [any BenchCase] { [GPUCopyProbeCase()] }
}
```

- [ ] **Step 4: Register**

In `Cases/Registry.swift`, change `let cases: [any BenchCase] = []` to `var cases: [any BenchCase] = []` (drop the trailing comment) and replace the comment line `// families are appended in Tasks 14 and 15` with:

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
- Modify: `Cases/Registry.swift`, `main.swift` (remove the `RunCommand` stub). Folded back into the earlier tasks' blocks by this task's review rounds, so those blocks already carry them: `CLI/Arguments.swift` (`parsed` replaces `int`), `main.swift` (`BenchError.exitCode`: usage and registry errors exit 2), `Runner/BenchCase.swift` (`BenchError.probeSetup`, `.outputExists`, `exitCode`; `RunOptions.defaultSeed`), `Runner/Runner.swift` (identity re-read after `prepare` whatever its outcome; note when a warm-up ends at the policy maximum without the CoV rule met), `Probe/GPUProbe.swift` (`probeSetup` throw; `copyGBps` warm-up ≥ 50 dispatches AND ≥ 0.1 s, median of 50), `Sources/VectorTestKit/PreflightProbe.swift` (Task 10: warm-up ≥ 20 copies AND ≥ 0.2 s, fast 5th percentile of 100), and the opt-in `PreflightInstrumentCheckTests`. Evidence: `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md`

**Interfaces:**
- Consumes: `MetalComputeProvider(context:configuration:)`, `MetalComputeProvider.Configuration(preferGPU:fallbackToCPU:)`, `batchDistance(query:candidates:metric: .euclidean)`, `routingTelemetry().gpuKernel`, `Metal4Context()`, `Metal4Context.lastGPUTiming` (`GPUTimingInfo.gpuStartTime/gpuEndTime`, seconds), `DynamicVector([Float])` from VectorCore, `Tolerance.forSum`, `Comparator`, `PreflightProbe`, `HostFingerprint`, `MetalFingerprint`, `GPUProbe`, `RunStore`, `Runner`, `Registry`.
- Produces: `enum DistanceCases { static var all: [any BenchCase] }`; `enum RunCommand { static func run(_ args: Arguments) async throws -> Int32 }`; `enum BenchVersion { static let package = "unreleased" }` (no version number is committed until the owner's release discussion concludes; `packageVersion` is an unvalidated free string that nothing compares on). Consumes from Task 13 as amended: `Arguments.parsed<T>(_ key: String, expecting: String, _ parse: (String) -> T?) throws -> T?` (absent → `nil`, present-but-unparseable → `BenchError.usage` naming the option and the value) and `main.swift` exiting with `BenchError.exitCode` (2 for usage and registry errors, 1 otherwise; 1 for any non-`BenchError`).

- [ ] **Step 1: `Cases/Distance/DistanceCases.swift`**

```swift
import Foundation
import Metal
import Synchronization
import VectorCore
import VectorAccelerate
import VectorTestKit

/// Batch L2 distance through the public façade at N=10,000, D=768 (spec §5.7). Route is verified per call via routing telemetry.
///
/// `.commandBuffer` timing is `Metal4Context.lastGPUTiming` read right after the call. Contract, not derivable from this file:
/// that value covers the whole call only while `L2KernelDistanceProvider.batchDistance` makes exactly one `executeAndWait`
/// submission (Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:135) and `Metal4Context` sets or clears
/// `_lastGPUTiming` once per submission (Core/Metal4Context.swift:435-441). If that path ever chunks into several submissions,
/// this series silently reports only the last one. No test pins it.
final class L2BatchDistanceCase: BenchCase {
    static let n = 10_000, d = 768
    static let candidateStream: UInt64 = 10, queryStream: UInt64 = 11
    static let caseID = "distance.l2.batch.n=\(n).d=\(d)"

    let id = try! CaseID(validating: L2BatchDistanceCase.caseID)
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = L2BatchDistanceCase.n

    /// Published by `prepare` from the run seed before anything there can throw; until then `identity` describes the default-seed fixture.
    private let publishedIdentity = Mutex<CaseIdentity?>(nil)
    var identity: CaseIdentity {
        publishedIdentity.withLock { $0 } ?? Self.makeIdentity(seed: RunOptions.defaultSeed, fixture: Self.fixture(seed: RunOptions.defaultSeed))
    }

    static func fixture(seed: UInt64) -> (candidates: [[Float]], query: [Float]) {
        var g = VectorGenerator(seed: seed, stream: candidateStream)
        let candidates = g.uniformVectors(count: n, dimension: d)
        var q = VectorGenerator(seed: seed, stream: queryStream)
        let query = q.uniformVectors(count: 1, dimension: d)[0]
        return (candidates, query)
    }

    static func makeIdentity(seed: UInt64, fixture f: (candidates: [[Float]], query: [Float])) -> CaseIdentity {
        CaseIdentity(id: caseID,
                     fixtureDigest: ContentDigest.sha256([.int(n), .int(d), .floats(f.candidates.flatMap { $0 }), .floats(f.query)]),
                     generatorVersion: VectorGenerator.version, seed: seed, stream: candidateStream,
                     route: "gpu.l2.batch", variant: nil, dataDescriptor: "uniform[-1,1]")
    }

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        let f = Self.fixture(seed: ctx.options.seed)
        let seeded = Self.makeIdentity(seed: ctx.options.seed, fixture: f)
        publishedIdentity.withLock { $0 = seeded }   // first: `Runner.run` re-reads `identity` after `prepare` even when the lines below throw
        guard ctx.device != nil else { throw BenchError.metalUnavailable }
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(preferGPU: true, fallbackToCPU: false))
        return Prepared(context: context, provider: provider, query: DynamicVector(f.query), candidates: f.candidates.map { DynamicVector($0) },
                        oracleSquared: f.candidates.map { c in
                            var s = 0.0; for i in 0..<Self.d { let x = Double(f.query[i]) - Double(c[i]); s += x * x }; return s })
    }

    final class Prepared: PreparedCase {
        static let comparator = "squared-distance vs Double oracle"

        let context: Metal4Context; let provider: MetalComputeProvider
        let query: DynamicVector; let candidates: [DynamicVector]; let oracleSquared: [Double]
        // `verify` compares (Float distance)² in Double with the Double oracle Σ(qᵢ−cᵢ)². Components lie in [-1,1], so each term is ≤ 4
        // and Σ ≤ 4·D = 3072. `forSum(.sequential)` gives 2·u·D·3072 = 0.28125 with u = 2⁻²⁴. Error budget it has to cover, assuming
        // round-to-nearest fp32 operations in the kernel (unverified: the kernel's reduction order and fast-math setting are not
        // documented, which is also why the sequential bound is used until SP2 derives per-kernel tolerances):
        //   summation of D non-negative terms, any order   ≤ (D−1)·u·3072 = 767·u·3072
        //   per-term rounding (one subtract, one square)   ≤ D·3u·4       =   3·u·3072
        //   fp32 sqrt, then squaring that result in Double ≤ 2u·3072      =   2·u·3072
        // Total ≤ 772·u·3072 ≈ 0.1414, about half of 0.28125. The Double oracle's own error (≈ 768·2⁻⁵³·3072 ≈ 2.6e-10) is negligible.
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
                return Correctness(status: .fail, comparator: Self.comparator, tolerance: tolerance, maxAbsError: nil, maxRelError: nil,
                                   recall: nil, recallPolicy: nil, notes: ["result count mismatch"])
            }
            // maxRelError uses the comparator's definition everywhere: |got − oracle| / max(|got|, |oracle|), 0 when both are 0.
            var maxAbs = 0.0, maxRel = 0.0, failures = 0, invalid = 0
            for (i, dist) in distances.enumerated() {
                // Squaring would hide a wrong sign, and a distance is never negative, NaN or infinite.
                guard dist.isFinite, dist >= 0 else { failures += 1; invalid += 1; continue }
                let got = Double(dist) * Double(dist), oracle = oracleSquared[i]
                let absError = Swift.abs(got - oracle), scale = max(Swift.abs(got), Swift.abs(oracle))
                maxAbs = max(maxAbs, absError); maxRel = max(maxRel, scale > 0 ? absError / scale : 0)
                switch Comparator.compare(got, oracle, tolerance: tolerance) {
                case .equal, .withinTolerance: break
                case .classMismatch, .outOfTolerance: failures += 1
                }
            }
            let notes = failures == 0 ? ["sequential fp32 sum bound, D=768, |x|≤1"]
                : ["\(failures) of \(distances.count) distances outside tolerance" + (invalid > 0 ? " (\(invalid) negative or non-finite)" : "")]
            return Correctness(status: failures == 0 ? .pass : .fail, comparator: Self.comparator, tolerance: tolerance,
                               maxAbsError: maxAbs, maxRelError: maxRel, recall: nil, recallPolicy: nil, notes: notes)
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

enum BenchVersion { static let package = "unreleased" }   // set when a release is cut; until then gitSHA identifies the build

enum RunCommand {
    /// What `run` reads: `--filter`, `--exclude` and `--out` in `run`, the rest in `parseOptions`. `--allow-dirty` is not here:
    /// `run` never refuses a dirty tree (it records `gitDirty` and the store suffixes the file `_dirty`), so the flag would do
    /// nothing; it belongs to `baseline capture`.
    static let allowedOptions: Set<String> = ["--filter", "--exclude", "--mode", "--samples", "--iterations", "--seed", "--out", "--label",
                                              "--probe-slowdown"]
    static let allowedFlags: Set<String> = ["--require-gpu", "--validation-layers"]

    static func run(_ args: Arguments) async throws -> Int32 {
        try args.validate(options: allowedOptions, flags: allowedFlags, operands: 0)
        let options = try parseOptions(args)
        let include = args.value("--filter").map { [$0] } ?? [], exclude = args.value("--exclude").map { [$0] } ?? []
        let cases = try Registry.select(include: include, exclude: exclude)
        guard !cases.isEmpty else {
            throw BenchError.usage("no case selected: --filter \(include.first.map { "'\($0)'" } ?? "(none)") --exclude \(exclude.first.map { "'\($0)'" } ?? "(none)") "
                                   + "matched 0 of \(Registry.all.count) registered case(s); see `list`")
        }
        let root = Workspace.root
        let out = args.value("--out").map { URL(fileURLWithPath: $0) }
        // Spec §7: an existing path is an error and is never overwritten. Checked before the run so the measurements are not lost;
        // `.withoutOverwriting` in `write` remains the backstop for a file that appears during the run.
        if let out, FileManager.default.fileExists(atPath: out.path) { throw BenchError.outputExists(out.path) }

        let device = MTLCreateSystemDefaultDevice()
        let host = HostFingerprint.gather(packageRoot: root)
        let gpu = MetalFingerprint.gather(device: device)
        let context = BenchContext(device: device, host: host, gpu: gpu, options: options)

        // Spec §4.10: a preflight GPU probe that cannot be set up or read makes the probe "unavailable" (quiet: false with the
        // reason recorded); it never aborts the run, so the cases still run and the artifact is still written.
        var gpuProbeFailures: [String] = []
        func attempt<T>(_ step: String, _ body: () throws -> T) -> T? {
            do { return try body() } catch {
                let message = "preflight GPU probe (\(step)): \(error)"
                gpuProbeFailures.append(message)
                Diagnostics.line("warning: \(message)")
                return nil
            }
        }

        let thermalStart = HostFingerprint.thermalStateName()
        let cpuStart = PreflightProbe.cpuCopyGBps()
        let probe = device.flatMap { d in attempt("setup") { try GPUProbe(device: d) } }
        let gpuStart = probe.flatMap { p in attempt("start reading") { try p.copyGBps() } }
        print("preflight: cpu \(String(format: "%.1f", cpuStart)) GB/s" + (gpuStart.map { String(format: ", gpu %.1f GB/s", $0) } ?? ", gpu n/a"))

        var records: [CaseRecord] = []
        for c in cases {
            print("running \(c.id.rawValue) (\(options.samples)×\(options.iterations)) ...")
            let r = await Runner.run(c, context: context)
            print("  \(r.status.rawValue)" + (r.timing.wall.map { String(format: "  wall median %.0f ns", $0.summary.median) } ?? "") + (r.failureReason.map { "  \($0)" } ?? ""))
            records.append(r)
        }

        let cpuEnd = PreflightProbe.cpuCopyGBps()
        let gpuEnd = gpuStart.flatMap { _ in probe.flatMap { p in attempt("end reading") { try p.copyGBps() } } }
        let thermalEnd = HostFingerprint.thermalStateName()
        var preflight = PreflightProbe.evaluate(
            cpu: StartEnd(start: cpuStart, end: cpuEnd),
            gpu: gpuStart.flatMap { s in gpuEnd.map { StartEnd(start: s, end: $0) } },
            thermalStart: thermalStart, thermalEnd: thermalEnd, gpuDeviceExists: device != nil)
        preflight.reasons += gpuProbeFailures   // `evaluate` has already made this not quiet ("GPU probe unavailable ..."); these say why

        let filters = (include.isEmpty && exclude.isEmpty) ? nil : Filters(include: include.isEmpty ? nil : include, exclude: exclude.isEmpty ? nil : exclude)
        let metadata = makeMetadata(host: host, gpu: gpu, options: options, thermal: ThermalInfo(start: thermalStart, end: thermalEnd),
                                    preflight: preflight, filters: filters)
        let artifact = BenchArtifact(package: "VectorAccelerate", metadata: metadata, cases: records)
        try artifact.validate()
        let url = try write(artifact, to: out)
        print("artifact: \(url.path)")
        print("key: \(metadata.environmentKey)  quiet: \(preflight.quiet)"
              + (preflight.reasons.isEmpty ? "" : "  reasons: \(preflight.reasons.joined(separator: "; "))"))

        if records.contains(where: { $0.status == .failed }) { return 1 }
        if options.requireGPU, records.contains(where: { $0.status == .unavailable }) { return 1 }
        return 0
    }

    /// The five typed options go through `Arguments.parsed`: absent takes the default, present-but-rejected throws `BenchError.usage`
    /// naming the option and the value.
    private static func parseOptions(_ args: Arguments) throws -> RunOptions {
        let positive: (String) -> Int? = { Int($0).flatMap { $0 >= 1 ? $0 : nil } }
        let mode = try args.parsed("--mode", expecting: "smoke, quick or full", { BenchMode(rawValue: $0) }) ?? .quick
        return try RunOptions(
            mode: mode,
            samples: args.parsed("--samples", expecting: "a positive integer", positive) ?? mode.samples,
            iterations: args.parsed("--iterations", expecting: "a positive integer", positive) ?? mode.iterations,
            seed: args.parsed("--seed", expecting: "an unsigned 64-bit integer", { UInt64($0) }) ?? RunOptions.defaultSeed,
            validationLayers: args.has("--validation-layers"), requireGPU: args.has("--require-gpu"),
            probeSlowdown: args.parsed("--probe-slowdown", expecting: "a non-negative integer", { Int($0).flatMap { $0 >= 0 ? $0 : nil } }) ?? 0,
            label: args.value("--label"))
    }

    private static func makeMetadata(host: HostFingerprint, gpu: GPUInfo?, options: RunOptions, thermal: ThermalInfo,
                                     preflight: PreflightRecord, filters: Filters?) -> Metadata {
        Metadata(
            packageVersion: BenchVersion.package, gitSHA: host.gitSHA, gitDirty: host.gitDirty, date: ISO8601DateFormatter().string(from: Date()),
            os: host.osVersion, osBuild: host.osBuild, arch: host.arch, cpuCores: host.cpuCores, performanceCores: host.performanceCores,
            efficiencyCores: host.efficiencyCores, memoryBytes: host.memoryBytes, deviceModel: host.chip, modelIdentifier: host.modelIdentifier,
            hostname: host.hostname, swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
            deviceTag: HostFingerprint.deviceTag(arch: host.arch, osVersion: host.osVersion, hostname: host.hostname),
            environmentKey: EnvironmentKey.make(chip: host.chip, gpuCores: gpu?.coreCount, osBuild: host.osBuild, metalToolchain: gpu?.metalToolchain,
                                                swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
                                                validationLayers: options.validationLayers),
            gpu: gpu, power: host.power, thermal: thermal, preflight: preflight,
            validationLayers: options.validationLayers, runSeed: options.seed, runLabel: options.label, mode: options.mode.rawValue,
            filters: filters, flags: [:])
    }

    private static func write(_ artifact: BenchArtifact, to out: URL?) throws -> URL {
        guard let out else { return try Workspace.store.writeRun(artifact) }
        try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        try RunStore.encode(artifact).write(to: out, options: [.withoutOverwriting])
        return out
    }
}
```

Remove the `enum RunCommand { ... }` stub from `main.swift`.

`RunCommand` closes the class "CLI input accepted without validation" for its five typed options (`--mode`, `--seed`, `--samples`, `--iterations`, `--probe-slowdown`) through `Arguments.parsed` (Task 13 Step 1; it replaced the silent-default `int(_:)`, which had no other caller); `--samples` and `--iterations` reject anything below 1 in their parse closures. An empty selection is a usage error raised before any probe runs, so a run that measures nothing cannot write an artifact or exit 0. After this task the only `throw BenchError.usage` sites are `Arguments.validate`, `Arguments.parsed` and the empty-selection guard, all caller input. `RunCommand.allowedOptions` / `allowedFlags` are what `validate` accepts for `run`. `--allow-dirty` is not among them although spec §5.5's synopsis lists it on `run`: §7 gives the flag meaning only for `baseline capture` (`run` never refuses a dirty tree, it records `gitDirty` and the file is suffixed `_dirty`), and a flag that does nothing misleads, so `run --allow-dirty` is a usage error (Task 17 Step 7 amends the synopsis). `BenchError.exitCode` (Task 13 Step 2) maps usage and registry errors to exit 2 and everything else to 1.

Spec §4.10 / §7 behaviour that the block above implements: an existing `--out` path is refused before the run (exit 1), with `.withoutOverwriting` as the backstop; any preflight GPU probe failure (`GPUProbe.init`, the start reading, the end reading) degrades to "probe unavailable" — `gpu: nil` with `gpuDeviceExists: true`, which `PreflightProbe.evaluate` turns into `quiet: false`, plus the error text appended to `preflight.reasons` and written to stderr — and the cases still run and the artifact is still written. The probe case still fails in its own `prepare` (case `failed`, run continues, exit 1).

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
- Modify: `main.swift` (remove the two stubs); `Sources/VectorTestKit/RunStore.swift` (`RunStoreError: CustomStringConvertible`) and `Tests/VectorAccelerateTests/Harness/RunStoreTests.swift` (`testErrorDescriptionsSayWhatFailedAndCarryThePayload`, written first and seen failing; one over-long assertion wrapped). Task 11's blocks carry both files as committed.

**Interfaces:**
- Consumes: `RunStore`, `RunStoreError` (its `description`), `Compare`, `CompareResult`; from Task 13 as amended, `Arguments.validate` and `Arguments.parsed`, `Workspace` (store location; artifact loading that names the file), `ContextError`, `Diagnostics`, and the single error exit in `main.swift` (`BenchError.usage` → stderr, exit 2; any other error → stderr, exit 1).
- Produces: `enum BaselineCommand { static func run(_:) throws -> Int32 }`, `enum CompareCommand { static func run(_:) throws -> Int32 }` with `CompareCommand.Against` (`.approved` / `.version(Int)`, positive versions only) and `CompareCommand.Format` (`text`, `json`, `markdown`).
- Boundary rules: usage failures throw `BenchError.usage`; every option is validated before any file is read, and an undeclared option, flag or operand is a usage error (`Arguments.validate`); warnings and notes go to stderr so stdout is only the rendered report; refusals are not caught locally. The two catches that exist add what the underlying error cannot say: the artifact's path (`Workspace.loadArtifact`), and the case and baseline being read when the store is corrupt (`ContextError` in `compare`), which still aborts the comparison, exit 1.
- Things said on stderr rather than left silent: `baseline capture` without `--approve` notes that `compare` reads the approved pointer by default; a partial capture prints `skipped <id> (<status>)` per non-measured case; `compare` warns when the artifact has no cases and when it was not quiet (reasons joined with `; `).

- [ ] **Step 1: `Commands/BaselineCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum BaselineCommand {
    static let usage = "usage: baseline capture --from <artifact> [--approve] [--allow-dirty]"

    /// A missing subcommand or `--from` is thrown as `BenchError.usage`. A load or validation failure is rethrown with the file
    /// named (`Workspace.loadArtifact`); capture errors (`RunStoreError` refusals included) propagate unchanged. Printing and
    /// exit codes belong to the `catch` arms in `main.swift`.
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: ["--from"], flags: ["--approve", "--allow-dirty"], operands: 1)
        guard args.subcommand == "capture", let from = args.value("--from") else { throw BenchError.usage(usage) }
        let approve = args.has("--approve")
        let artifact = try Workspace.loadArtifact(from, validating: true)
        let written = try Workspace.store.captureBaseline(from: artifact, approve: approve, allowDirty: args.has("--allow-dirty"))
        for (id, version) in written.sorted(by: { $0.key < $1.key }) {
            print("captured \(id) -> version \(version)\(approve ? " (approved)" : "")")
        }
        // `captureBaseline` stores measured cases only; say which ones it left out instead of leaving a partial capture silent.
        for record in artifact.cases where record.status != .measured {
            Diagnostics.line("skipped \(record.id) (\(record.status.rawValue)): only measured cases are captured"
                             + (approve ? "; its approved pointer, if any, is unchanged" : ""))
        }
        if !approve {
            Diagnostics.line("note: captured without --approve, so no approved pointer moved. `compare` reads the approved pointer by "
                             + "default: capture with --approve, or compare with --against <n> using a version printed above")
        }
        return 0
    }
}
```

- [ ] **Step 2: `Commands/CompareCommand.swift`**

```swift
import Foundation
import VectorTestKit

enum CompareCommand {
    static let usage = "usage: compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]"

    /// Which stored baseline every case is compared against.
    enum Against: CustomStringConvertible {
        case approved
        case version(Int)

        /// `approved` or a positive integer; anything else is `nil`, which `Arguments.parsed` reports as a usage error.
        init?(_ raw: String) {
            if raw == "approved" { self = .approved; return }
            guard let n = Int(raw), n >= 1 else { return nil }
            self = .version(n)
        }

        var description: String {
            switch self {
            case .approved: return "approved"
            case .version(let n): return "version \(n)"
            }
        }

        func baseline(in store: RunStore, environmentKey: String, caseID: String) throws -> CaseRecord? {
            switch self {
            case .approved: return try store.approvedBaseline(environmentKey: environmentKey, caseID: caseID)
            case .version(let n): return try store.baseline(environmentKey: environmentKey, caseID: caseID, version: n)
            }
        }
    }

    /// Unverified contract: each raw value must equal a format name `Compare.render` switches on; that function renders a name
    /// it does not know as text instead of failing, so a mismatch here would go unreported.
    enum Format: String { case text, json, markdown }

    static func run(_ args: Arguments) throws -> Int32 {
        // Every option is checked before the first file is read, so a usage error never depends on the artifact or the store.
        try args.validate(options: ["--artifact", "--against", "--format"], flags: ["--strict-perf"], operands: 0)
        guard let path = args.value("--artifact") else { throw BenchError.usage(usage) }
        let against = try args.parsed("--against", expecting: "'approved' or a positive integer version", Against.init) ?? .approved
        let format = try args.parsed("--format", expecting: "text, json or markdown", { Format(rawValue: $0) }) ?? .text
        let strictPerf = args.has("--strict-perf")

        let artifact = try Workspace.loadArtifact(path, validating: false)   // each record is validated by `Compare.compare`
        let store = Workspace.store
        let key = artifact.metadata.environmentKey
        let results: [CompareResult] = try artifact.cases.map { record in
            let baseline: CaseRecord?
            // A corrupt store aborts the comparison; the error says which case and which baseline, which the store's error cannot.
            do { baseline = try against.baseline(in: store, environmentKey: key, caseID: record.id) } catch {
                throw ContextError("case \(record.id): reading its \(against) baseline failed (store \(store.root.path), key \(key))", error)
            }
            return Compare.compare(new: record, baseline: baseline, newKey: key, baselineKey: baseline == nil ? nil : key)
        }
        // One trailing newline on stdout whether or not the rendering already ends with one.
        let report = Compare.render(results, format: format.rawValue)
        print(report, terminator: report.hasSuffix("\n") ? "" : "\n")
        // Warnings go to stderr so stdout stays the rendered report. Spec §7: a not-quiet artifact is still compared, with a warning.
        if artifact.cases.isEmpty { Diagnostics.line("warning: the artifact has no cases, so nothing was compared") }
        if !artifact.metadata.preflight.quiet {
            let reasons = artifact.metadata.preflight.reasons
            Diagnostics.line("warning: artifact was not quiet: \(reasons.isEmpty ? "no reason recorded" : reasons.joined(separator: "; "))")
        }
        return Compare.exitCode(results, strictPerf: strictPerf)
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
swift run -q -c release VectorAccelerateBench compare --artifact /nonexistent.json --against 0; echo "exit=$?"    # "error: --against expects ..." on stderr, exit=2 (checked before the file is read)
swift run -q -c release VectorAccelerateBench compare --artifact "$A" --format json | python3 -c "import json,sys; json.load(sys.stdin)"; echo "exit=$?"   # exit=0, also for a not-quiet artifact (its warning is on stderr)
swift run -q -c release VectorAccelerateBench compare --artifact "$A" --agianst 1; echo "exit=$?"   # "error: unknown option --agianst for `compare`; it takes: ..." on stderr, exit=2
```

`baseline capture` needs a quiet artifact: on a busy machine the capture is refused with `error: baseline capture refused: the artifact's preflight was not quiet (...)` on stderr, exit=1 — run the smoke again rather than working around the refusal.

- [ ] **Step 4: Commit**

```bash
git add Sources/VectorAccelerateBench Sources/VectorTestKit/RunStore.swift Tests/VectorAccelerateTests/Harness/RunStoreTests.swift
git commit -q -m "feat(bench): baseline capture and compare commands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: Close the carried review findings, retire version strings, lint the harness files

Every item below is a finding a task review or re-review carried in the SDD ledger as "fix before merge", a class-closing edit of a few lines, or a consequence of the owner's 2026-09-22 decision (no version numbers; break early). None needs design. They are one task because each is small, they touch overlapping files, and together they leave the tree ready for the deletions and the full gates in Task 18. The plan's code blocks for every file changed here are resynced in Step 11 so the plan stays the record.

**Files:**
- Modify (kit): `Sources/VectorTestKit/Compare.swift`, `Sources/VectorTestKit/PreflightProbe.swift`, `Sources/VectorTestKit/SeededRNG.swift`, `Sources/VectorTestKit/VectorGenerator.swift`
- Modify (bench): `Sources/VectorAccelerateBench/Runner/Runner.swift`, `Sources/VectorAccelerateBench/Probe/GPUProbe.swift`, `Sources/VectorAccelerateBench/Commands/RunCommand.swift`, `Sources/VectorAccelerateBench/Commands/ValidateCommand.swift`, `Sources/VectorAccelerateBench/main.swift` (three comment labels)
- Modify (tests): `Tests/VectorAccelerateTests/Harness/CompareTests.swift`, `Tests/VectorAccelerateTests/Harness/ErrorDescriptionLintTests.swift`, `Tests/VectorAccelerateTests/Harness/ArtifactSchemaTests.swift`, `Tests/VectorAccelerateTests/Harness/ComparatorTests.swift`, `Tests/VectorAccelerateTests/Harness/KitImportLintTests.swift` (only if lint names it)
- Create (tests, by the fix rounds): `Tests/VectorAccelerateTests/Harness/CollectionInterpolationLintTests.swift` (Task 13 Step 6b), `Tests/VectorAccelerateTests/Harness/FixtureCitations.swift` (Task 13 Step 6c; the helper and its `FixtureCitationsTests` share this file)
- Modify (config, docs): `.swiftlint.yml`, `Package.swift` (one header comment line), `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md`, this plan (the Architecture paragraph, and blocks of Tasks 0, 3, 4, 8, 10, 12, 13, 14, 15 as named in Step 11)

**Interfaces:**
- Consumes: `Compare.compare(new:baseline:newKey:baselineKey:)`, `CompareResult.reasons`, `Runner.run(_:context:policy:)`, `BenchCase.identity` / `prepare(_:)`, `Arguments.validate(options:flags:operands:)`, `BenchError.usage`, `RepoPaths`.
- Produces: nothing new. `TestRNG` and `TestDataGenerator` cease to exist; `Compare` records a fourth skip reason (`timing.<kind> not compared: present on the <side> side only`).

- [ ] **Step 1: One-sided timing kind gets a reason — write the failing test**

In `Tests/VectorAccelerateTests/Harness/CompareTests.swift`, inside `testUnusableKindIsSkippedWithAReasonAndTheRestDecide`, replace the `oneSided` block (the comment `// A kind present on one side only is still simply not compared, without a reason.`, the `let oneSided = …` statement and its two assertions) with:

```swift
        // A kind present on one side only is the fourth skip condition: `Runner` omits a declared GPU kind whose samples came
        // back in the wrong shape, so a broken GPU timer can appear as an absent series, not only as a zero median.
        let oneSided = Compare.compare(new: pair(wall: [200, 200, 200], gpu: [0, 0, 0]), baseline: record(wall: [100, 100, 100]),
                                       newKey: key, baselineKey: key)
        XCTAssertEqual(oneSided.status, .alertSlower); XCTAssertEqual(oneSided.deltas.map(\.kind), [.wall])
        XCTAssertEqual(oneSided.reasons, ["timing.commandBuffer not compared: present on the new side only"])
        let otherSide = Compare.compare(new: record(wall: [200, 200, 200]), baseline: pair(wall: [100, 100, 100]),
                                        newKey: key, baselineKey: key)
        XCTAssertEqual(otherSide.status, .alertSlower); XCTAssertEqual(otherSide.deltas.map(\.kind), [.wall])
        XCTAssertEqual(otherSide.reasons, ["timing.commandBuffer not compared: present on the baseline side only"])
        // Absent on both sides is nothing to compare, not a drop: no reason.
        let neither = Compare.compare(new: record(wall: [100, 100, 100]), baseline: record(wall: [100, 100, 100]),
                                      newKey: key, baselineKey: key)
        XCTAssertEqual(neither.status, .ok); XCTAssertEqual(neither.reasons, [])
```

`record(wall:)` builds a wall-only record and `pair(wall:gpu:)` a wall + commandBuffer record (both helpers already exist in this file; confirm what `record` produces before relying on it — if the sample artifact it copies carries a `commandBuffer` series, strip it in the helper so the fixture is genuinely one-sided).

- [ ] **Step 2: Run it to see it fail**

```bash
swift test --filter CompareTests 2>&1 | grep -E "error:|Executed [0-9]+ test" | tail -6
```
Expected: `oneSided.reasons` is `[]` and `otherSide.reasons` is `[]` — failures, no traps. Any OTHER test in this class that fails here has a one-sided fixture and expects no reason; list each in the report and update its expectation in Step 3 (the reason wording is fixed; the verdicts must not change).

- [ ] **Step 3: Make it pass**

In `Sources/VectorTestKit/Compare.swift`, replace the head of the per-kind loop —

```swift
        for kind in TimingKind.allCases {
            guard let n = new.timing[kind], let b = base.timing[kind] else { continue }
            // A kind that cannot be compared is skipped with a reason and the remaining kinds decide the case, so a broken GPU
            // timer cannot hide a wall regression (`CompareTests.testUnusableKindIsSkippedWithAReasonAndTheRestDecide`). Three
            // conditions, one wording — `timing.<kind> not compared: <quantity> <value> is not …` — on either side.
            let notCompared = "timing.\(kind.rawValue) not compared"
```

— with:

```swift
        for kind in TimingKind.allCases {
            // A kind that cannot be compared is skipped with a reason and the remaining kinds decide the case, so a broken GPU
            // timer cannot hide a wall regression (`CompareTests.testUnusableKindIsSkippedWithAReasonAndTheRestDecide`). Four
            // conditions, one wording — `timing.<kind> not compared: …` — on either side.
            let notCompared = "timing.\(kind.rawValue) not compared"
            // 0. Present on one side only. `Runner` omits a declared GPU kind whose samples came back in the wrong shape, so a
            // broken GPU timer can appear as an absent series, not only as a zero median. Absent on both sides is nothing to
            // compare, not a drop.
            let sides = (new: new.timing[kind], base: base.timing[kind])
            if sides.new == nil, sides.base == nil { continue }
            guard let n = sides.new, let b = sides.base else {
                skipped.append("\(notCompared): present on the \(sides.new == nil ? "baseline" : "new") side only")
                continue
            }
```

The rest of the loop (conditions 1–3, the delta, the boundary, the alert) is unchanged. In the same file, replace the `try!` contract comment in `render` with one that cites both pinning tests:

```swift
            // try!, as a contract: `JSONEncoder` throws here only for a non-finite Double. Results produced by `Compare.compare`
            // carry finite values only (`CompareTests.testNonFiniteDeltaOrBoundaryIsIncomparable` for skipped kinds;
            // `testUnusableKindIsSkippedWithAReasonAndTheRestDecide` round-trips results that carry deltas through this encoder).
            // A hand-built `CompareResult` holding NaN or an infinity is the caller's responsibility and traps here.
```

Update any test expectations found in Step 2. Then:

```bash
swift test --filter CompareTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
```
Expected: `Executed 15 tests, with 0 failures`.

- [ ] **Step 4: `Runner` reads `identity` once, after `prepare`**

`Runner.run` currently reads `c.identity` to build the record BEFORE `prepare`, then overwrites it with a second read afterwards; for `L2BatchDistanceCase` the first read generates and SHA-256s the default-seed fixture (≈30 MB) and is thrown away. In `Sources/VectorAccelerateBench/Runner/Runner.swift` replace the function head through the `do {` / `let prepared = …` line —

```swift
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
```

— with:

```swift
    static func run(_ c: any BenchCase, context: BenchContext, policy: WarmupPolicy = .standard) async -> CaseRecord {
        let o = context.options
        // `prepare` runs before the record exists so `identity` is read exactly once, afterwards. Contract on adapters (unverified,
        // no test): one whose identity depends on the run seed publishes the seeded identity before anything in its `prepare` can
        // throw (`L2BatchDistanceCase` does), so failed and unavailable records carry the run's seed and digest too; a struct
        // adapter returns its registration-time identity either way.
        let preparation: Result<any PreparedCase, any Error>
        do { preparation = .success(try await c.prepare(context)) } catch { preparation = .failure(error) }
        var record = CaseRecord(id: c.id.rawValue, identity: c.identity, status: .measured, failureReason: nil,
                                samples: o.samples, iterations: o.iterations, warmupUsed: [], unitCount: c.unitCount,
                                timing: Timing(wall: nil, commandBuffer: nil, encoder: nil), correctness: nil, resources: nil, notes: [])
        if o.probeSlowdown > 0 { record.notes.append("probe-slowdown=\(o.probeSlowdown)") }
        let clock = ContinuousClock()
        func ns(_ d: Duration) -> Double { Double(d.components.seconds) * 1e9 + Double(d.components.attoseconds) / 1e9 }

        do {
            let prepared = try preparation.get()
```

and delete the trailing block before `return record` (the three-line `// Re-read after \`prepare\` whatever its outcome …` comment and `record.identity = c.identity`). `preparation.get()` rethrows the stored error, so both `catch` arms match exactly as before.

```bash
swift build 2>&1 | grep -E "warning:|error:|Build complete" | tail -3
swift run -q VectorAccelerateBench run --mode smoke --seed 7 --filter 'distance.*' --out /tmp/va-seed7.json >/dev/null; echo "exit=$?"
python3 -c "import json; c=json.load(open('/tmp/va-seed7.json'))['cases'][0]; print(c['status'], c['identity']['seed'], c['identity']['fixtureDigest'][:16])"; rm -f /tmp/va-seed7.json
```
Expected: build clean; `measured 7 <digest>` where the digest differs from the seed-42 digest `list` shows. (The failed-`prepare` path carries the seed by the same single read — derivable from the code above; Task 15's report pinned it with fault injection.)

- [ ] **Step 5: The lint test's doc comment claims an `actor` shape its fixtures never pin**

`Tests/VectorAccelerateTests/Harness/ErrorDescriptionLintTests.swift` line 12 lists `actor` among the recognised heads and line 20 says `testScannerFixtures` pins each shape named. No fixture has an `actor` head. Add two fixtures to `testScannerFixtures` in the same table form as the existing sixteen, one flagged and one described:

```swift
            ("actor Boom: Error {}", Scan(errorTypes: ["Boom"], describedTypes: [])),
            ("actor Calm: Error, CustomStringConvertible { var description: String { \"calm\" } }",
             Scan(errorTypes: ["Calm"], describedTypes: ["Calm"])),
```

(adapt the tuple shape to the table's actual shape — the expected `Scan` values are what matter). Run:

```bash
swift test --filter ErrorDescriptionLintTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
```
Expected: `Executed 4 tests, with 0 failures` (the class has four tests since the fix round added `testContractCitesExactlyTheFixtureRows`). If the scanner does NOT recognise `actor`, fix the scanner (the doc comment is the contract), not the comment.

- [ ] **Step 6: Finite, non-negative warm-up floors**

Both probes take a public `warmupSeconds: Double`; `.infinity` spins forever and NaN never ends the loop. Add, as the first statement of `GPUProbe.copyGBps(warmup:warmupSeconds:measured:)` in `Sources/VectorAccelerateBench/Probe/GPUProbe.swift` and of `PreflightProbe.cpuCopyGBps(...)` in `Sources/VectorTestKit/PreflightProbe.swift` (whatever its warm-up parameters are named — read the signature):

```swift
        precondition(warmupSeconds.isFinite && warmupSeconds >= 0, "warmupSeconds must be finite and non-negative")
```

```bash
swift test --filter FingerprintTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
```
Expected: `Executed 8 tests, with 0 failures`.

- [ ] **Step 7: Two stdout leaks in the bench commands**

`Sources/VectorAccelerateBench/Commands/ValidateCommand.swift`: the missing-path guard prints usage on stdout and returns 2 while every other usage error goes through `BenchError.usage` to stderr. Read `Arguments.validate(options:flags:operands:)` first: if `operands: 1` already rejects a MISSING operand, delete the guard; otherwise make it `guard let path = args.subcommand else { throw BenchError.usage("usage: validate-artifact <path>") }`. The `invalid: …` line stays on stdout — it is the command's verdict, not an error.

`Sources/VectorAccelerateBench/Commands/RunCommand.swift`: the end-of-run summary prints `reasons: \(preflight.reasons)` — a Swift array debug literal (second occurrence of the class the Task 16 review closed in `CompareCommand`). Use the same form: `reasons: \(preflight.reasons.joined(separator: "; "))`.

```bash
swift build 2>&1 | grep -E "warning:|error:|Build complete" | tail -3
.build/debug/VectorAccelerateBench validate-artifact; echo "exit=$?"
```
Expected: one `error: …` line on stderr (nothing on stdout), `exit=2`. The reasons format is derivable from the line itself; if any smoke run in this task comes out not quiet, paste its `key:` line as the demonstration.

- [ ] **Step 8: Retire every version string; delete the compatibility typealiases**

1. `Package.swift` line 16: `// 0.7.0 — Benchmark harness foundation (SP0) + dead-code removal (BREAKING):` → `// Unreleased — Benchmark harness foundation (SP0) + dead-code removal (BREAKING; lands in the next tagged release, unnumbered until then):`. Nothing else in the file changes.
2. Delete the three-line typealias blocks (doc comment, `@available`, `public typealias`) at the end of `Sources/VectorTestKit/SeededRNG.swift` (`TestRNG`) and `Sources/VectorTestKit/VectorGenerator.swift` (`TestDataGenerator`). First prove they have no user:
   ```bash
   grep -rn "TestRNG\b\|TestDataGenerator\b" Sources Tests | grep -v "^Sources/VectorTestKit/SeededRNG.swift\|^Sources/VectorTestKit/VectorGenerator.swift"
   ```
   Expected: only doc-comment lines in the two golden test files that describe where the goldens came from (history; they stay).
3. `Tests/VectorAccelerateTests/Harness/ArtifactSchemaTests.swift`: the sample artifact's `packageVersion: "0.7.0"` → `packageVersion: "unreleased"` (what `BenchVersion.package` writes).
4. Confirm no version number remains: `grep -rn "0\.7\.0\|0\.8\.0" Package.swift Sources Tests CHANGELOG.md` → no output (README's `0.6.0` banner and CHANGELOG's `## [0.6.0]` history are the last tagged release and stay).

```bash
swift test --filter "SeededRNGGoldenTests|VectorGeneratorGoldenTests|ArtifactSchemaTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
```
Expected: 0 failures (the goldens do not mention the aliases).

- [ ] **Step 9: Lint the harness paths to zero error-level findings**

The branch added 21 error-level swiftlint findings, 19 of them `line_length` in the two GENERATED golden files (one literal array per line, up to 1,454 columns) — data, not code. Exclude those two files from lint in `.swiftlint.yml` (add to the top-level `excluded:` list, after `- Package.swift`):

```yaml
  - Tests/VectorAccelerateTests/Harness/SeededRNGGoldenTests.swift       # generated golden data: one literal array per line
  - Tests/VectorAccelerateTests/Harness/VectorGeneratorGoldenTests.swift  # generated golden data
```

Then list what remains in the harness paths and fix each with the smallest change that leaves every assertion untouched (wrap the 206-column line at `ComparatorTests.swift:50`; for a finding in `KitImportLintTests.swift` whose subject is a lint-probe fixture, a `// swiftlint:disable:next <rule>` with a one-line reason is acceptable if renaming the fixture would change what the probe tests):

```bash
swiftlint lint --quiet 2>&1 | grep "error:" | grep -E "Sources/VectorTestKit/|Sources/VectorAccelerateBench/|Tests/VectorAccelerateTests/Harness/"   # fix until empty
swiftlint lint --quiet 2>&1 | grep -c "error:"    # record; expected 20 = the count at 853052d (kernels, IVF, hardening tests — not this plan's)
```

- [ ] **Step 10: Accuracy of the tracked evidence (claim discipline)**

`docs/superpowers/research/2026-09-18-preflight-probe-characterization.md` — the Task 15 re-review found these; fix each, quoting the report file `.superpowers/sdd/2026-09-17-harness-sp0-foundation/task-15-report.md` where a number is needed:
1. "the ramp is a wall-time effect" (§"WarmupPolicy refuted; candidates", the **Chosen** paragraph) — not established by any table (no series varies dispatch duration) and contradicted by the note's own "What is not established". Reword: the floor is a hedge that covers a time-bound ramp on a faster GPU and costs ≈0.1 s if the ramp is count-bound.
2. §"Estimator comparison": "the fast 5th percentile of 100 is the most repeatable in the typical calm trial" — the table's own p05-of-150 row reads 0.92 against 1.08, and the archived analysis (`task-15-probe-characterization/` beside the report; copy the two numbers into the note) has p05 of 200 at 0.84 and p10 of 200 at 0.90. State that 100 is the most repeatable AMONG the 100-copy estimators and give the cost reason for stopping at 100 (each copy's measured time is in the note's own CPU tables; state the per-call cost each extra 50 copies would add).
3. §"Before and after" table, round-2 set A background: "`xctest` in five" — the archived `bg_03`–`bg_08` samples and the report's table show six (runs 03–08).
4. Same section, the sentence "with the count-only warm-up and no such process sampled it had stayed under 1.30 %": round 1's background was never sampled and 1.30 % was the maximum observed, not a bound. Reword so it does not suggest that load, or the time floor, explains the 4.4–4.8 % round-2 drifts — the data does not separate them.
5. Same section, "five invocations" of the instrument check — the report shows six; list all six values (the omitted one passed).

Then the same wall-time claim in code: `Sources/VectorAccelerateBench/Probe/GPUProbe.swift`, the `copyGBps` doc comment's sentence "The ramp is a wall-time effect, so the count alone would not carry to a GPU that finishes 50 dispatches much sooner; hence the time floor." → the hedge wording of item 1. And the MSL comment on the kernel's `dst[gid] = …` line, which names "Task 18" (a plan-internal number inside shipped source): repoint it at the evidence file the acceptance task writes, `docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md`, keeping every line of the MSL string ≤ 200 columns (split the comment across two lines if needed — Step 9's gate covers it).

The same class — a pointer from shipped source into this plan's task numbering, which the 2026-09-22 renumbering already staled once — also appears as the `// Task 15` / `// Task 16` labels on the three dispatch lines of `Sources/VectorAccelerateBench/main.swift`: delete the labels (the called type names the command). The `Task 3 finding` / `Task 3 freeze` notes in `SeededRNG.swift`, `KitImportLintTests.swift` and the two golden files record where a decision came from and stay. (The two `blackHole` qualification comments that name the old Task 17 are deleted by Task 18 Step 1b.)

- [ ] **Step 11: Plan sync, gates, commit**

Resync the plan's code blocks to the committed files, byte-identical (`cmp` on extracted blocks; say so in the report): Task 3 (`SeededRNG.swift` — remove the typealias lines from the block AND from the Step grep that names `typealias TestRNG`), Task 4 (the generator script appends the `TestDataGenerator` typealias — remove that append line and the grep expectation, so a re-run of the recipe reproduces the committed file), Task 8 (`ArtifactSchemaTests` fixture string), Task 10 (`PreflightProbe.swift`), Task 12 (`Compare.swift`, `CompareTests.swift`), Task 13 (`Runner.swift`, `ValidateCommand.swift`, `ErrorDescriptionLintTests.swift`), Task 14 (`GPUProbe.swift`), Task 15 (`RunCommand.swift`), whichever task's block last defines `main.swift` (the labels), and Task 0 — its `.swiftlint.yml` step text gains the two exclusions, its `Package.swift` header block becomes the committed `// Unreleased —` entry, and its Step 2 heading becomes `Add the Unreleased entry to the version-history header in Package.swift` (the Step's anchor line `// 0.6.0 — …` is the last tagged release and stays). The plan's **Architecture** paragraph ends `Deletions of dead benchmark code land at the end as a breaking change recorded under CHANGELOG \`[Unreleased]\`.` instead of naming a version. Afterwards `grep -n "0\.7\.0\|0\.8\.0" docs/superpowers/plans/2026-09-17-harness-sp0-foundation.md` prints only lines that quote text being replaced (this task's Step 8, Task 18 Step 7, the self-review notes); list them in the report.

```bash
swift test --filter "ArtifactSchemaTests|CaseIDTests|CollectionInterpolationLintTests|ComparatorTests|CompareTests|ContentDigestGoldenTests|ErrorDescriptionLintTests|FingerprintTests|FixtureCitationsTests|KitImportLintTests|PreflightInstrumentCheckTests|RecallTests|RunStoreTests|SeededRNGGoldenTests|StatsTests|VectorGeneratorGoldenTests|WarmupPolicyTests" 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
# the 17 harness classes: 113 tests, 1 skipped (the opt-in instrument check), 0 failures — the 107 at a46acb1 plus the six this task's fix rounds
# added (CollectionInterpolationLintTests 3, FixtureCitationsTests 2, ErrorDescriptionLintTests.testContractCitesExactlyTheFixtureRows 1).
# Task 18 Step 0 adds FixtureCitationsTests.testContractProblemsNameEachCondition, so the same filter reads 114 from that task on.
# (A bare `--filter "Harness"` matches none of them: it selects the legacy IndexBenchmarkHarnessTests only.)
swift build -c release 2>&1 | grep -cE "warning:|error:"                          # expected: 0
git add .swiftlint.yml Package.swift Sources/VectorTestKit Sources/VectorAccelerateBench Tests/VectorAccelerateTests/Harness \
        docs/superpowers/research/2026-09-18-preflight-probe-characterization.md docs/superpowers/plans/2026-09-17-harness-sp0-foundation.md
git status --short     # nothing outside those paths
git commit -q -F - <<'MSG'
fix(harness): one-sided timing kinds carry a reason; identity read once; version strings retired; evidence note corrected

Closes the findings the Task 14–16 reviews carried: a timing kind present on one side of a
comparison is skipped with a reason like the other three conditions; Runner reads identity
once after prepare; the lint test pins the actor shape it documents; warm-up floors must be
finite; validate-artifact usage errors go to stderr; run prints preflight reasons joined.
Deletes the TestRNG/TestDataGenerator aliases (no users). No version number remains in
code or fixtures. The characterization note's four counting/wording slips are fixed and
the time-bound-ramp claim is a hedge, not a fact. Golden files are excluded from lint.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

---

### Task 18: Deletions, orphan guard, Metal-skip helper, README, CHANGELOG, spec amendment

**Files:**
- Delete: `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`, `Sources/VectorAccelerate/Benchmarking/IndexBenchmarkHarness.swift`, `Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift`, `Tests/VectorAccelerateTests/PerformanceBenchmarks.swift`, `Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift`
- Modify: `Package.swift` (header comment only), `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift`, `Sources/VectorAccelerate/Kernels/Metal4/QuantizationStatisticsKernel.swift`, `Sources/VectorAccelerate/Kernels/Metal4/WarpOptimizedSelectionKernel.swift`, `Sources/VectorAccelerate/Configuration/AccelerationConfiguration.swift` (Step 5 rule 3: `ExecutionPath`, `OperationType`), `Sources/VectorAccelerateBench/Cases/Distance/DistanceCases.swift`, `Tests/VectorAccelerateTests/Harness/StatsTests.swift`, `Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift`, `Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift`, `README.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md`, this plan (Task 7 and Task 15 blocks)
- Create: `Tests/VectorAccelerateTests/Utilities/RequireMetal.swift`, `Tests/VectorAccelerateTests/Harness/OrphanPublicTypeTests.swift`, `Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt`

- [ ] **Step 1: Delete the dead code**

```bash
# The parallel IVF session works in the MAIN checkout, not this worktree: check there, read-only (no index refresh, no lock).
MAIN=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
T18_PATHS="Sources/VectorAccelerate/Benchmarking Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift Sources/VectorAccelerate/Kernels/Metal4/QuantizationStatisticsKernel.swift Sources/VectorAccelerate/Kernels/Metal4/WarpOptimizedSelectionKernel.swift Tests/VectorAccelerateTests/PerformanceBenchmarks.swift Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift Tests/VectorAccelerateTests/IndexBenchmarkRecallValidationTests.swift Tests/VectorAccelerateTests/GPUCandidateBuilderValidationTests.swift README.md CHANGELOG.md Package.swift"
git --no-optional-locks -C "$MAIN" status --short -- $T18_PATHS             # expected: empty
git log --oneline 853052d..gifton/metal-hardening-checkpoint -- $T18_PATHS   # expected: empty — either non-empty: stop and report
git rm -q Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift Sources/VectorAccelerate/Benchmarking/IndexBenchmarkHarness.swift \
          Sources/VectorAccelerate/Configuration/PerformanceMonitor.swift \
          Tests/VectorAccelerateTests/PerformanceBenchmarks.swift Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift
rmdir Sources/VectorAccelerate/Benchmarking 2>/dev/null; true
grep -rn "\.benchmark(" Sources Tests    # expected: no output — the two kernel benchmark() methods have no caller (Task 9 pre-read; re-verify)
```

Then delete `public func benchmark(` in `Sources/VectorAccelerate/Kernels/Metal4/WarpOptimizedSelectionKernel.swift` (the whole method, from its doc comment through its closing brace) and in `Sources/VectorAccelerate/Kernels/Metal4/QuantizationStatisticsKernel.swift` the same plus the nested `public struct BenchmarkResult: Sendable` it returns. Owner Q7: dead code is deleted, not allow-listed. `swift build` must succeed before Step 2. (Done 2026-09-22: the `WarpOptimizedSelectionKernel.benchmark()` return type, the nested `public struct PerformanceMetrics` with no other producer, went with it under the same rule; the ratchet could not have reported it because two unrelated types share the simple name.)

- [ ] **Step 1b: Drop the `blackHole` qualification comments**

With the library's `Benchmarking/` gone, `blackHole` resolves to the kit's unambiguously. In `Sources/VectorAccelerateBench/Cases/Distance/DistanceCases.swift` change `VectorTestKit.blackHole(distances)   // qualified: … until Task 17 deletes Benchmarking/` to `blackHole(distances)` (no comment), and in `Tests/VectorAccelerateTests/Harness/StatsTests.swift` delete the comment line `// Qualified: VectorAccelerate.blackHole still exists until Task 17 deletes Benchmarking/.` and unqualify the call below it. `swift build` and `swift test --filter StatsTests` prove the resolution. Resync the plan's Task 7 and Task 15 blocks.

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
        let declPattern = #"^\s*(?:@[\w()., ]+\s+)*public\s+(?:final\s+)?(?:struct|class|enum|actor|protocol|typealias)\s+([A-Z][A-Za-z0-9_]*)"#
        let decl = try NSRegularExpression(pattern: declPattern, options: [.anchorsMatchLines])
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

Seed the allowlist: run the test once, and for every name it reports decide whether it is intended public API. Each intended one goes into `Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt` as `Name  # one-line justification`; anything that is not intended API is deleted from `Sources` in this task instead. The ratchet then fails only on new orphans. (Done 2026-09-22 under the controller's triage order — used by a sibling VSK package; named in README/docs as API; a leftover of code this task deletes; otherwise kept pending an owner decision: 150 names reported; 5 allow-listed as used by EmbedKit, 44 as named in `docs/`, 99 kept pending an owner decision, 2 deleted as leftovers of `PerformanceMonitor` — `ExecutionPath` and `OperationType` in `AccelerationConfiguration.swift`. The per-name table is in the Task 18 report.)

```bash
swift test --filter OrphanPublicTypeTests 2>&1 | grep -A200 "Unreferenced public types" | head -80   # review this list
# ... edit OrphanAllowlist.txt / delete genuine dead types ...
swift test --filter OrphanPublicTypeTests 2>&1 | grep -E "Executed [0-9]+ test" | tail -1   # expected: 0 failures
```

- [ ] **Step 6: README and CHANGELOG**

`README.md`:
- Line 5: replace `VectorAccelerate delivers up to 100x speedups for large-scale vector operations.` with `VectorAccelerate accelerates large-scale vector operations on Apple silicon; performance is measured, not asserted (see Performance).`
- Replace the `## 📊 Performance` section body (the sentence `Typical speedups over CPU implementations:` and the five-row table) with:

```markdown
Performance is measured by the benchmark harness (`swift run -c release VectorAccelerateBench run --mode quick`), which records hardware, OS, toolchain, power and thermal state, and a preflight calibration with every run. Run records live under `.bench/runs/` and are not committed; measured claims will be published from those records once the SP4 matrix lands. Method and program: `docs/superpowers/research/2026-09-14-observability-harness-research.md`.
```

- Line 626 area: replace `swift run VectorAccelerateBenchmarks` with `swift run -c release VectorAccelerateBench run --mode quick`.
- The line-7 `Version 0.6.0` banner is NOT changed: it names the last tagged release (Global Constraints, version).

`Package.swift`: the version-history header's `// Unreleased —` entry (Task 17) lists removals; its `Removed:` bullet gains `QuantizationStatisticsKernel.benchmark() and WarpOptimizedSelectionKernel.benchmark()` (wrapped to the header's existing width). Nothing else in the file changes. (Done 2026-09-22: the bullet also names the nested result types and the `ExecutionPath` / `OperationType` deletion from Step 5; the Task 0 Step 2 block holds the committed header.)

`CHANGELOG.md`: insert after the intro paragraph, before `## [0.6.0]` (Keep-a-Changelog's `[Unreleased]` section; it accumulates everything that lands before the next tagged release and is renamed when that release is cut):

```markdown
## [Unreleased]

Benchmark harness foundation (SP0 of the observability program) and the removal of dead benchmarking code. Breaking: the public `Benchmarking/` surface and two kernel `benchmark()` methods are removed. No version is assigned yet.

### Added
- **`VectorTestKit`** (non-product library target): `SeededRNG` (frozen streams), `VectorGenerator` (versioned, frozen digests), `ContentDigest`, `Tolerance`/`Comparator` with derived sum bounds, distance-epsilon `Recall`, `Stats` with bootstrap intervals, artifact schema v1, `HostFingerprint`/`EnvironmentKey`, `PreflightProbe`, append-only `RunStore`, `Compare` with the owner's boundary rule.
- **`VectorAccelerateBench`** (executable): `list`, `run`, `baseline capture`, `compare`, `validate-artifact`; two M0 cases (GPU copy probe, batch L2 distance at N=10,000 D=768).
- Mechanical guards: kit import allowlist, orphan public-type ratchet, golden streams/digests.

### Removed (breaking)
- `Sources/VectorAccelerate/Benchmarking/` and all its public types (`BenchmarkSuite`, `BenchmarkResult`, `BenchmarkConfiguration`, `BenchmarkReport`, `PerformanceComparison`, `IndexBenchmarkHarness`, `IndexBenchmarkConfiguration`, `IndexBenchmarkResult`, `LatencyStats`, `BenchmarkWorkload`, `SeededRandomNumberGenerator`, `blackHole` — the last now lives in `VectorTestKit`).
- The `VectorAccelerateBenchmarks` executable product (crossover sweep returns as a harness mode in SP4).
- `PerformanceMonitor` (unused), `IndexAccelerationConfiguration.logDecisions` (never read).
- `QuantizationStatisticsKernel.benchmark()` with its nested `BenchmarkResult`, and `WarpOptimizedSelectionKernel.benchmark()` with its nested `PerformanceMetrics` (no callers).
- `ExecutionPath` and `OperationType` (`AccelerationConfiguration.swift`): the orphan ratchet reported them; `PerformanceMonitor` was their only user.

### Changed
- README performance claims retired pending measured run records.
- The test utilities `TestRNG` and `TestDataGenerator` became `VectorTestKit.SeededRNG` and `VectorTestKit.VectorGenerator` with frozen output streams; no compatibility aliases are kept.
- Preflight calibration costs about 1.7 s per run (it was about 0.1 s and was not repeatable; see `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md`).
```

- [ ] **Step 7: Amend the spec for the recorded deviations**

In `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md`:
- §4.7: replace `Percentiles use linear interpolation between order statistics (the method already in \`IndexBenchmarkHarness.LatencyStats\`, moved here).` with `Percentiles use linear interpolation between order statistics (rank = p·(n−1)); this replaces the floor-index method the old \`IndexBenchmarkHarness.LatencyStats\` used.`
- §6 row for the two old test files: replace with `| \`Tests/.../IndexBenchmarkHarnessTests.swift\` | delete (every test depends on the harness; its recall assertions duplicate the validation suite's) | no |` and a new row `| \`Tests/.../IndexBenchmarkRecallValidationTests.swift\` | keep; remove the one harness-only test (\`testBenchmarkHarnessRealisticIVF\`) and its private RNG; this file already asserts flat recall = 1.0 and IVF recall > 0.95 through the public index API | no |`.
- §10: replace the `IndexRecallTests (ported)` row with `| \`IndexBenchmarkRecallValidationTests\` (retained, trimmed) — flat recall exactly 1.0; IVF recall > 0.95 at nprobe = nlist on a seeded set | keeps existing coverage |`.
- §11 item 1: replace `+ ported (\`IndexRecallTests\`)` with `− one trimmed test`.
- §5.4: replace the grammar line with `<family>(.<op>)+(.<param>=<value>)*(.<variant>)?` and note that the M0 ID `distance.l2.batch.n=10000.d=768` requires more than one bare op segment.
- §5.3 (sampling): state that an adapter returns exactly one `GPUSample` per declared timing kind per iteration; a kind produced in any other shape is omitted from the artifact and noted on the record (Task 13 runner behaviour).
- §4.8: the thermal-state list gains `"unknown"` (what `HostFingerprint.thermalStateName()` returns for `ProcessInfo.thermalState`'s `@unknown default`), matching the schema doc comment.
- §5.7 (probe case): `dataDescriptor = "sequential_u32_pattern"` becomes `"sawtooth_f32_(i&0xFFFF)*0.5+1"` (the fill is a float32 sawtooth; the old label misdescribed it), and `resetForSample` re-zeroes the destination so a passing `memcmp` proves that at least one of that sample's dispatches wrote the whole destination (zeroing is once per sample, before warm-up; `verify` runs after the last measured iteration). The expectation `≈0.36 ms, ≈370 GB/s` did not reproduce: the Task 15 characterization measured a settled dispatch of ≈0.765 ms on the M3 Max, which reads ≈350 GB/s under the read-plus-write convention (the spec's figure also used the 1× convention: 134217728 B / 0.36 ms). Replace it with the figures the M0 acceptance run measures.
- §5.8: `GPUProbe.copyGBps()` returns `2 × 128 MiB / median command-buffer seconds / 1e9` (read plus write), the same convention as §4.10's CPU probe, so the two preflight figures are comparable.
- §5.8: `5 warm-up + 10 measured` becomes "warm-up until both 50 dispatches and 0.1 s of wall time have passed, then the median of 50", with a fixed floor rather than `WarmupPolicy`. Measured on the M3 Max over 81 recorded series (13,500 dispatches): after GPU idle 56 of the 81 series started slow (median 1.88×, range 1.08–2.40× the settled ≈0.765 ms) and settled only after up to 41 dispatches (median 10, p90 25), at most 55.5 ms of wall time; whether the ramp is bounded by time or by dispatch count is not established (the note says so), and the time floor is the cheap hedge — it covers a GPU that finishes 50 dispatches sooner if the ramp is time-bound and costs ≈0.1 s if it is not; `WarmupPolicy.standard` stopped at n = 3 on that slow plateau in 3 of 45 series because the plateau's own CoV is under 5 % (one reading then came out +41 %); a median of 10 settled dispatches strays up to 4.66 % from its series' level (p99 2.38 %) against 1.85 % (p99 1.01 %) for a median of 50, with a 5 % gate on the difference of two readings. One `copyGBps()` call costs ≈0.16 s instead of ≈0.02–0.04 s. The 5 % gate of §4.10 is unchanged. Evidence and the before/after quiet tables: `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md`.
- §4.10 (CPU probe method): "best of 3 copies of 256 MiB" becomes "warm-up until both 20 copies and 0.2 s of wall time have passed, then the fast 5th percentile of 100 timed copies" (read plus write unchanged). Measured on the same machine: a thread that was idle, blocked or waiting on subprocesses copies at as little as 0.52× its settled rate for up to 17 copies (111.5 ms), and ten best-of-3 readings 0.3 s apart spanned 56–117 GB/s; settled copy times are bimodal and alternate, so on identical data (ten trials with no other process above 60 % CPU) ten readings spanned at most 1.76 % of their median for the 5th percentile of 100 against 1.89 % for the minimum and 3.57 % for the median. One call costs ≈0.71 s instead of ≈0.05 s. The opt-in `PreflightInstrumentCheckTests` (`VA_INSTRUMENT_CHECK=1`) asserts ten readings span under 2.5 %; there is no equivalent for the GPU probe because the executable has no test target. Same research note.
- §7: add "empty selection (no case matches `--filter`/`--exclude`) → usage error before any probe runs, exit 2", and state that a preflight GPU probe failure at setup, at the start reading or at the end reading is the §4.10 "probe unavailable" state (`quiet: false`, the error text recorded in `preflight.reasons` and on stderr) and never aborts the run: the cases run and the artifact is written. `BenchError.registry` exits 2 as §7 already says (it exited 1 until Task 15's review).
- §5.5, §5.6 step 4 and §7 (Task 16 review): every command rejects an option, flag or operand it does not declare — a usage error naming it, exit 2, before the command does any work (new §7 row: "unknown option, flag or operand → usage error, exit 2"). §5.6 step 5 gains a rule for a timing kind present in both that cannot be compared — its new or baseline median is not a positive finite number (a GPU timing pair can come back 0), or its `delta` or `boundary` is not finite (usable medians can overflow the ratio, and `2 × base.rsd` can overflow; `validate()` does not bound summary fields): the kind is skipped, contributing no delta and no verdict, and a reason naming the kind, the quantity and the value is kept on the result (`timing.<kind> not compared: …`, the same wording for all three conditions and for either side); the case's status comes from the remaining kinds exactly as before, so a broken GPU timer cannot hide a wall regression; and §5.6 step 4 gains the cause "no timing kind is comparable" (new §7 row: "timing kind not comparable → skipped with a reason, the case judged on the rest; `incomparable` only if none is left; exit status as §5.6"). Reasons are part of every output format whatever the status.
- §5.5: the `run` synopsis drops `[--allow-dirty]`. `run` never refuses a dirty tree (§7: it suffixes the artifact `_dirty` and records `gitDirty: true`), so the flag had no effect there; it stays on `baseline capture`, the one command §7 gives it meaning for, and `run --allow-dirty` is now a usage error like any other undeclared flag.
- §5.3 (sampling): when a sample's warm-up ends at the policy maximum without the CoV rule being met, the runner appends a note to the record with the count. The policy itself is unchanged; its weakness on a steady slow plateau (it can stop at n = 3) is recorded in the research note and deferred to the owner.
- §5.7 and §11 item 5: `--probe-slowdown 4` becomes `--probe-slowdown 64`. The Task 14 review measured the probe's slowdown against `loops` on the M3 Max as ≈1.0× for loops ≤ 6, 1.5× at 8, 2.8× at 32 and ≈8× at 64 (bandwidth headroom absorbs the first extra passes), so 4 would not cross the alert boundary; the acceptance evidence records the ratio actually measured.
- Header "**Version impact:** VectorAccelerate **0.7.0** (breaking: …)" → "**Version impact:** unreleased. Breaking (removes the `Benchmarking/` public surface, the `VectorAccelerateBenchmarks` product, `IndexAccelerationConfiguration.logDecisions`, and two kernel `benchmark()` methods); no version number is assigned — the owner bundles everything that lands on main into the next tagged breaking release (decision 2026-09-22)." §3's manifest sentence "add a `0.7.0` entry to the version-history header" → "add an `Unreleased` entry"; §6's row "`[0.7.0] — Unreleased` entry" → "`[Unreleased]` entry".
- §4.2 and §4.3: replace each alias sentence — §4.2's "`public typealias TestRNG = SeededRNG` is provided, marked `@available(*, deprecated, renamed: "SeededRNG")`, and removed in 0.8.0." and §4.3's "Deprecated typealias `TestDataGenerator`, removed in 0.8.0." — with "No compatibility alias for the old name is kept (deprecated `TestRNG`/`TestDataGenerator` aliases existed briefly and were deleted: no user; the owner breaks early)."
- Then `grep -n "0\.7\.0\|0\.8\.0" docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md` → no output (Global Constraints, version).
- §6 (deletions table): add a row `| \`QuantizationStatisticsKernel.benchmark()\` + nested \`BenchmarkResult\`, \`WarpOptimizedSelectionKernel.benchmark()\` | delete (no caller anywhere; the orphan ratchet would have flagged the struct) | no |`.
- §8: "Preflight adds roughly one second per run (two copies of 256 MiB and 2 × 15 GPU dispatches)" → "Preflight adds roughly 1.7 s per run: two CPU probe calls of ≈0.7 s each (warm-up of at least 20 copies and 0.2 s, then 100 timed copies of 256 MiB) and two GPU probe calls of ≈0.16 s each (warm-up of at least 50 dispatches and 0.1 s, then 50 timed dispatches). The original 0.1 s instruments were not repeatable within the 5 % gate (§4.10); the cost bought repeatability. Evidence: `docs/superpowers/research/2026-09-18-preflight-probe-characterization.md`."
- §5.2 (warm-up policy) and §12 (open items): add to §12 "**WarmupPolicy is refuted as a warm-up detector for GPU timing** (Task 15 characterization): on the post-idle slow plateau its CoV rule is satisfied at n = 3, and under compositor-cadence spikes it often never converges, so the runner reaches the maximum of 50 on every smoke sample (each such sample is noted on the record since Task 15). M0 baselines may carry ramp contamination, worst in smoke mode; they are versioned and re-capturable. Redesign — a time-floored warm-up is the candidate the data supports — is deferred to SP1 (timing), by the owner's scope." and a one-line pointer to it in §5.2.
- §12: add "Executable test target: three reviewers asked for automated pins of the CLI (option validation, exit codes, stderr routing) — the executable has no test target; splitting it into a library + thin `main` is SP1/SP3 work." and "`deviceTag` hostname source: `ProcessInfo.hostName` here vs `Host.current().localizedName` in VectorCore; cross-suite joins by `deviceTag` are SP4's concern (Task 10 review)." and "Absurd-but-finite baseline `rsd` (JSON can carry 1e300) widens the boundary without limit so any regression reads `ok` with `widened: true`; bounding summary fields in `validate()` is SP2's per-kernel tolerance work (Task 16 review)."

- [ ] **Step 8: Full gates**

```bash
swift test 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
swift test -c release 2>&1 | grep -E "Executed [0-9]+ test" | tail -1
swiftlint lint --quiet 2>&1 | grep "error:" | grep -E "Sources/VectorTestKit/|Sources/VectorAccelerateBench/|Tests/VectorAccelerateTests/Harness/" | wc -l   # expected: 0 (Task 17 Step 9 made it so; the deletions must not reintroduce one)
swiftlint lint --quiet 2>&1 | grep -c "error:"    # record it; expected ≤ 20, the count at 853052d — every one in a file this plan never created (kernels, IVF, hardening tests); not this plan's to fix
```
Expected: 0 failures in both configurations; 3 skips — the two IVF throughput placeholders, unchanged, plus `PreflightInstrumentCheckTests`, which skips unless `VA_INSTRUMENT_CHECK=1`. Record the exact counts for Task 19 and the commit whose tree they ran over (the one Step 9 creates); a review fix round that later changes `Sources`, `Tests` or `Package.swift` re-runs both suites and records the new commit. Each full run takes 20+ minutes and holds the SwiftPM build lock — run them sequentially, never alongside another build.

- [ ] **Step 9: Commit**

```bash
git add -A Package.swift Sources/VectorAccelerate Tests/VectorAccelerateTests README.md CHANGELOG.md docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md \
        Sources/VectorAccelerateBench/Cases/Distance/DistanceCases.swift docs/superpowers/plans/2026-09-17-harness-sp0-foundation.md
git status --short     # nothing outside those paths; nothing from the parallel session
git commit -q -F - <<'MSG'
refactor!: remove Benchmarking surface, kernel benchmark() methods, PerformanceMonitor, logDecisions; retire README claims

Adds the orphan public-type ratchet and requireMetalDevice(). Recall coverage
retained in IndexBenchmarkRecallValidationTests (harness-only test removed).
CHANGELOG accumulates under [Unreleased]; no version is assigned. Spec amended
for every deviation the ledger recorded (Step 7).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

---

### Task 19: Milestone M0 acceptance and evidence

**Files:**
- Create: `docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md`

**Precondition — an idle machine.** Every timing measurement in Tasks 15–16 ran while another session built and tested in the shared checkout (load average 3–8), so the preflight quiet rate has never been measured under the conditions the spec assumes. Before step 0 the owner pauses the other session; if `uptime`'s 1-minute load average is above 1.5 with nothing of ours running, stop and report rather than measure. Never retry a run until it looks quiet; every run is recorded.

**Test counts (spec §11 item 1) come from Task 18's full gates** when `git diff --quiet $T18 HEAD -- Sources Tests Package.swift` succeeds, `$T18` being the commit Task 18's report names for its last full-gate run: the counts then describe byte-identical code, and a second 45-minute pass would lengthen the owner's pause and heat the machine just before the idle measurements. If the diff is non-empty, run both suites BEFORE the idle window (they need no idle machine) and record them. Spec §11's order holds either way: item 1 precedes items 2–7.

- [ ] **Step 1: Run the acceptance sequence (spec §11) on the M3 Max, AC power, idle desktop; paste every command and its tail output into the evidence file**

```bash
uptime                                                                                              # 0. the machine must be idle: no other session building or testing (every Task 15/16 measurement ran under load 3–8 from a parallel session; record the load averages)
[ -d .bench/baselines ] && mv .bench/baselines ".bench/baselines.pre-m0.$(date +%s)"                # 0b. start from an empty baseline store (Task 16's exercise left version 1 + approved from a smoke, dirty-tree artifact); moved aside, not deleted
git diff --quiet "$T18" HEAD -- Sources Tests Package.swift && echo "counts: Task 18 gates at $T18 apply"   # 1. counts (see above); non-empty diff → both suites were run before step 0
VA_INSTRUMENT_CHECK=1 swift test --filter PreflightInstrumentCheckTests 2>&1 | grep -E "VA_INSTRUMENT_CHECK|Executed [0-9]+ test" | sort -u   # 1b. opt-in CPU-probe instrument check: ten readings span < 2.5 % (record the readings; a failure means the instrument or the machine moved)
swift build -c release 2>&1 | tail -1                                                               # 1c. warm the release binary so the loop below measures runs, not builds
for i in 1 2 3 4 5 6 7 8 9 10; do .build/release/VectorAccelerateBench run --mode smoke >/dev/null 2>&1; A=$(ls -t .bench/runs/*/*.json | head -1); python3 -c "import json; p=json.load(open('$A'))['metadata']['preflight']; print($i, p['cpuCopyGBps'], p.get('gpuCopyGBps'), p['driftPercent'], p['quiet'], p['reasons'])"; done   # 1d. quiet rate on the idle machine: ten consecutive smoke runs, every row kept
swift run -c release VectorAccelerateBench run --mode quick 2>&1 | tail -4                          # 2. artifact, quiet: true
A=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench validate-artifact "$A"
swift run -q -c release VectorAccelerateBench baseline capture --from "$A" --approve                # 3. (refused with exit 1 and a "dirty tree" message on stderr if the tree is dirty → commit first or add --allow-dirty and say so)
swift run -c release VectorAccelerateBench run --mode quick 2>&1 | tail -2                          # 4. second run
B=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench compare --artifact "$B"
swift run -c release VectorAccelerateBench run --mode quick --filter 'probe.*' --probe-slowdown 64 2>&1 | tail -2   # 5. slowdown (64, not the spec's 4: measured ≈1.0× at 4 on the M3 Max — see Task 18 Step 7)
C=$(ls -t .bench/runs/*/*.json | head -1); swift run -q -c release VectorAccelerateBench compare --artifact "$C"; echo "exit=$?"           # alert(slower), exit=0
swift run -q -c release VectorAccelerateBench compare --artifact "$C" --strict-perf; echo "exit=$?"                                      # exit=1
touch README.md.tmp && swift run -c release VectorAccelerateBench run --mode smoke 2>&1 | grep artifact; rm README.md.tmp               # 6. _dirty suffix
D=$(ls -t .bench/runs/*/*_dirty.json | head -1); swift run -q -c release VectorAccelerateBench baseline capture --from "$D"; echo "exit=$?"   # refused, exit=1
MTL_DEBUG_LAYER=1 swift run -c release VectorAccelerateBench run --mode smoke --validation-layers 2>&1 | tail -2; echo "exit=$?"         # 7. exit=0
E=$(ls -t .bench/runs/*-val/*.json | head -1); swift run -q -c release VectorAccelerateBench baseline capture --from "$E"; echo "exit=$?"    # refused, exit=1
```

- [ ] **Step 2: Write the evidence file**

`docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md` contains: date; machine fingerprint (chip, OS build, Swift, Xcode, power, thermal) and the `uptime` load averages from step 0; the exact test counts from step 1 with the reconciliation against the previous 1764/0/2 gate (deleted suites, added suites, trimmed test, the opt-in skip); the instrument-check readings from 1b; the ten-row quiet-rate table from 1d (cpu start/end, gpu start/end, drift, quiet, reasons) and the rate as a fraction — this is the first idle-machine measurement of the gate, and it stands whatever it shows; the artifact paths; the compare outputs for steps 4 and 5 (verbatim); the probe's measured slowdown ratio at `--probe-slowdown 64` (step 5 wall and command-buffer medians over step 4's); the two refusals; one paragraph on the runner's warm-up — how many samples of each case ended at the policy maximum (the record's notes and `warmupUsed`), that the policy's weakness is characterized in the probe note and its redesign deferred to SP1, and that these baselines are therefore provisional (versioned, re-capturable); and one paragraph of observations (probe GB/s, distance-case command-buffer median, whether the boundary widened and why, any `not compared` reason — none is expected). No number is rounded away from what the tool printed.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md
git commit -q -m "docs: SP0 milestone M0 acceptance evidence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## After Task 19: branch finish (controller steps, not a task)

1. **Final whole-branch review.** `scripts/review-package <plan> 853052d HEAD`; dispatch superpowers:requesting-code-review's `code-reviewer.md` on the most capable model, pointing it at the ledger's `minor (deferred)`, `parked`, `CORRECTION`, `LOAD-BEARING` and `Ruling:` lines so it can triage what must be fixed before merge. ONE fix dispatch for its findings, ONE scoped re-review, residuals adjudicated in the ledger.
2. **Reword one commit trailer.** `db622c0` (Task 0) carries `Co-Authored-By: Claude Haiku 4.5` instead of the mandated Fable trailer. The branch has never been pushed, so its history is local: with a clean tree, `git rebase --exec 'git log -1 --format=%B | sed "s/Claude Haiku 4.5/Claude Fable 5.1/" | git commit --amend -q -F -' db622c0~1`. Every SHA after `db622c0` changes; the ledger's SHAs become historical from this point, which is why this is the last step before the merge. Confirm with `git log --format=%B db622c0~1..HEAD | grep -c "Claude Haiku"` → 0.
3. **Merge.** superpowers:finishing-a-development-branch: the target is `gifton/metal-hardening-checkpoint` in the main checkout (the parallel IVF session commits there; resolve conflicts, never discard either side). Pushing is the owner's call.
4. **Clean up.** After the merge, `rm -rf .superpowers/sdd/2026-09-17-harness-sp0-foundation/` (git-ignored; the plan, the research notes and git history are the record). `.bench/` is local, git-ignored measurement state; leave it.

## Self-review notes (run by the plan author before hand-off)

- **Spec coverage.** §3 layout → Task 0. §4.1 → Task 1. §4.2 → Task 3 (alias deleted in Task 17). §4.3 → Task 4 (same). §4.4 → Task 2. §4.5 → Task 5. §4.6 (amended) → Task 6. §4.7 → Task 7. §4.8 → Task 8. §4.9–4.10 → Tasks 10, 15 (CPU probe method amended). §4.11 → Task 11. §5.1–5.3 → Task 13 (warm-up visibility note: Task 15). §5.4 → Tasks 9, 13. §5.5 → Tasks 13, 15, 16. §5.6 (amended order; per-kind skips) → Tasks 12, 16, 17. §5.7 → Tasks 14, 15. §5.8 → Tasks 14, 15. §6 → Tasks 0, 17, 18. §7 → Tasks 13, 15, 16. §8 (amended cost) → Task 18 Step 7. §10 → Tasks 1–12, 16 (error-description lint), 17, 18. §11 → Task 19. §12 gains the deferred items in Task 18 Step 7. §13 is SP2 scope by the spec's own §1 non-goals and has no task here. The spec header's "Version impact: 0.7.0" is superseded by the owner's 2026-09-22 decision and amended in Task 18 Step 7.
- **Placeholder scan.** No TBD/TODO; the only data-driven steps are the two golden generators (Tasks 3–4, deterministic scripts), the orphan allowlist seeding (Task 18, explicit procedure) and Task 17 Step 10's evidence corrections (each names its source number).
- **Type consistency.** `PreflightRecord`/`PreflightProbe` used consistently (spec's `Preflight` split, noted in Global Constraints). `Correctness.maxAbsError`/`maxRelError` (Task 8) match the contract's amended names. `CaseRecord.validate()` (Task 8) is what `Compare` (Task 12) calls. `Runner` writes `Series` via `Stats.summarize` (Task 7). `BenchError` is defined once (Task 13) and thrown by Tasks 14–16. The skip-reason wording `timing.<kind> not compared: …` is shared by Tasks 16 and 17. `Runner.run`'s single `identity` read (Task 17) rests on the Task 15 adapter contract (seeded identity published before `prepare` can throw).
- **Owner decisions folded in (2026-09-22).** No version number anywhere; break early (typealiases and kernel `benchmark()` methods deleted rather than deprecated); `WarmupPolicy` redesign, an executable test target, the `deviceTag` source and `validate()` bounds on summary fields are recorded in spec §12 as deferred, not silently dropped.
- **Renumbering (2026-09-22).** The task that was "Task 17: Deletions…" is now Task 18 and "Task 18: Milestone M0…" is now Task 19; the new Task 17 closes the reviews' carried findings first so Task 18's full gates run over the final code. Ledger entries written before this date use the old numbers.
