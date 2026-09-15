# Hardening Audit — Phase 0 + 1.1 + 1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert VectorAccelerate's green-but-uninstrumented test baseline into an honest one: prove which silicon actually serves each call (Phase 0), differential-test every GPU kernel entry surface against its CPU fallback with adversarial inputs (Phase 1.1), and close the dual shader-compilation drift class with generated guards (Phase 1.2).

**Architecture:** Three probe layers, no product-behavior changes: (a) actor-isolated routing telemetry on `MetalComputeProvider` + an env-gated GPU-submit trace in `Metal4Context`; (b) new `Tests/VectorAccelerateTests/Hardening/` test files that drive the internal kernel providers directly (forced-GPU leg) against `AccelerateFallback` (CPU leg) and a Double-accumulation oracle; (c) generated completeness/parity guards over `KernelContext.makeLibraryFromBundleSources` — the release-primary shader load path.

**Tech Stack:** Swift 6.2 strict concurrency, XCTest (`@testable import VectorAccelerate`), Metal 4, VectorCore 0.3.2.

## Global Constraints

- Swift 6.2 `StrictConcurrency` + `ExistentialAny` are enabled for the target — all new code must compile clean under them.
- Do NOT change product routing/fallback semantics in this phase. Telemetry observes; it must not alter which path runs. Routing defects are *findings for the ledger*, not fixes here (exception: the runtime-compile `shaderFiles` list omission, which is a straight defect with a failing test first).
- No commits unless the user asks. Untracked junk (`.antigravitycli/`, `GEMINI.md`, `consolidated_library.md`, `generate_consolidation.py`) is never staged.
- Every new test must `XCTSkip` cleanly when `MTLCreateSystemDefaultDevice()` is nil.
- Logs and measurement artifacts go to the session scratchpad, not the repo.
- Known facts this plan is built on (verified 2026-08-16):
  - `KernelContext.makeLibraryFromBundleSources` (`KernelContext.swift:249`) compiles a 27-entry `shaderFiles` list; the Shaders directory has 30 `.metal` files. Missing: `SoADistance.metal` (load-bearing — `MetalComputeProvider.init` throws if its kernels are absent), `ManhattanDistance.metal`, `ChebyshevDistance.metal` (both dead: zero Swift references).
  - `Metal4ShaderCompiler` gets all functions from `KernelContext.getSharedLibrary` (`Metal4ShaderCompiler.swift:452,468`) — one shared library serves every kernel.
  - `GPUDecisionEngine.shouldUseGPU` gates on `k >= minKForGPU` (default 10) and `q*n*k >= minOperationsForGPU` (default 50 000) (`GPUDecisionEngine.swift:347,351`); `batchDistance` passes `k: 0`, so the provider's batch-distance GPU path can never fire under default config, and `evaluateOperationSpecificCriteria` (`:415`) additionally hard-floors `candidateCount * dimension >= 100_000` regardless of thresholds.
  - `MetalComputeProvider.findNearest` silently falls through to CPU when the GPU fused path returns an empty result (`MetalComputeProvider.swift:202`).

---

### Task 1: RoutingTelemetry on MetalComputeProvider

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider+SoA.swift`
- Test: `Tests/VectorAccelerateTests/Hardening/RoutingProvenanceTests.swift`

**Interfaces:**
- Produces: `MetalComputeProvider.RoutingTelemetry` (public struct: `gpuKernel`, `cpuDecisionEngine`, `cpuPolicy`, `cpuFallbackAfterGPUError`, `cpuFallbackEmptyGPUResult: Int`, `lastGPUErrorDescription: String?`); actor methods `routingTelemetry() -> RoutingTelemetry`, `resetRoutingTelemetry()`; internal `recordSoAGPUDispatch()` for the extension file. Task 5's harness asserts on these.

- [ ] **Step 1: Write the failing test**

```swift
// Tests/VectorAccelerateTests/Hardening/RoutingProvenanceTests.swift
import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

/// Phase-0 provenance: prove which silicon served each MetalComputeProvider call.
/// These tests PIN currently-observed routing behavior — including behavior that is
/// itself a finding (batchDistance never GPUs under defaults). If a pin breaks
/// because routing was deliberately fixed, update the pin and the audit ledger.
final class RoutingProvenanceTests: XCTestCase {
    private func makeVectors(_ n: Int, _ dim: Int, seed: UInt64 = 42) -> [DynamicVector] {
        var rng = TestRNG(seed: seed)
        return (0..<n).map { _ in DynamicVector((0..<dim).map { _ in rng.nextFloat(in: -1...1) }) }
    }

    func testDefaultConfigBatchDistanceNeverUsesGPU() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
        let provider = try await MetalComputeProvider()
        let vecs = makeVectors(2000, 128)   // above minVectorsForGPU/minCandidatesForGPU
        _ = try await provider.batchDistance(query: vecs[0], candidates: Array(vecs.dropFirst()), metric: .euclidean)
        let t = await provider.routingTelemetry()
        // FINDING VA2-003: k=0 fails the minKForGPU/minOperationsForGPU gates, so this
        // is cpuDecisionEngine — the documented "no-copy GPU kernel path" is unreachable.
        XCTAssertEqual(t.gpuKernel, 0)
        XCTAssertEqual(t.cpuDecisionEngine, 1)
    }

    func testFindNearestLargeNRoutesToGPU() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
        let provider = try await MetalComputeProvider()
        let vecs = makeVectors(5001, 64)
        let r = try await provider.findNearest(query: vecs[0], in: Array(vecs.dropFirst()), k: 10, metric: .euclidean)
        XCTAssertEqual(r.count, 10)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1, "N=5000,k=10 passes every gate; fused GPU path must have served this, telemetry=\(t)")
    }

    func testPreferGPUFalseIsAllCPU() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
        let provider = try await MetalComputeProvider(configuration: .init(preferGPU: false))
        let vecs = makeVectors(5001, 64)
        _ = try await provider.findNearest(query: vecs[0], in: Array(vecs.dropFirst()), k: 10, metric: .euclidean)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0)
        XCTAssertGreaterThanOrEqual(t.cpuDecisionEngine, 1)
    }

    func testSoAPathAlwaysGPU() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
        let provider = try await MetalComputeProvider()
        let vecs = makeVectors(64, 32)
        let set = try SoACandidateSet(candidates: vecs, device: provider.context.device)
        _ = try await provider.batchDistance(query: vecs[0], against: set, metric: .euclidean)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1, "SoA scoring bypasses the decision engine by design")
    }
}
```

- [ ] **Step 2: Run to verify it fails** — `swift test --filter RoutingProvenanceTests` → FAIL: `routingTelemetry` undefined.

- [ ] **Step 3: Implement telemetry.** In `MetalComputeProvider.swift` add inside the actor:

```swift
/// Where provider calls were actually served (Phase-0 audit provenance; cheap, always on).
public struct RoutingTelemetry: Sendable, Equatable {
    /// Calls whose result was produced by a GPU kernel dispatch.
    public var gpuKernel: Int = 0
    /// Calls the decision engine (or `preferGPU == false`) routed to CPU up front.
    public var cpuDecisionEngine: Int = 0
    /// Calls served on CPU by policy: metrics with no GPU kernel, single-pair `distance`.
    public var cpuPolicy: Int = 0
    /// GPU kernel threw and `fallbackToCPU` silently rescued the call.
    public var cpuFallbackAfterGPUError: Int = 0
    /// GPU fused top-K returned an empty result and the call silently fell through to CPU.
    public var cpuFallbackEmptyGPUResult: Int = 0
    /// `String(describing:)` of the most recent GPU error that was swallowed by fallback.
    public var lastGPUErrorDescription: String? = nil
    public init() {}
}
var _telemetry = RoutingTelemetry()        // internal: +SoA extension increments it
public func routingTelemetry() -> RoutingTelemetry { _telemetry }
public func resetRoutingTelemetry() { _telemetry = RoutingTelemetry() }
```

Wire increments (no behavior change): in `batchDistance(_:SupportedDistanceMetric)` euclidean/cosine arms — `cpuDecisionEngine += 1` in the `guard routeToGPU else` leg, `gpuKernel += 1` on kernel success, `cpuFallbackAfterGPUError += 1` + `lastGPUErrorDescription` in the catch-with-fallback leg; `cpuPolicy += 1` in dotProduct/manhattan/chebyshev arms and in `distance(_:_:metric:)`; in `findNearest` fused branch — `gpuKernel += 1` before returning a non-empty GPU result, `cpuFallbackEmptyGPUResult += 1` when it falls through empty, `cpuFallbackAfterGPUError += 1` in its catch-with-fallback, and `cpuDecisionEngine += 1` when `routeToGPU` votes CPU for a euclidean/cosine call (note: the subsequent CPU `batchDistance` self-call adds its own count; document that `findNearest`-on-CPU contributes two events). In `+SoA.swift` add `_telemetry.gpuKernel += 1` after each successful `executeAndWait` (extension is same module → internal access OK; it is an actor method so isolation is already correct).

- [ ] **Step 4: Run to verify pass** — `swift test --filter RoutingProvenanceTests` → PASS (if `testDefaultConfigBatchDistanceNeverUsesGPU` fails with `gpuKernel == 1`, the routing finding is wrong — update the ledger, not the code).

### Task 2: VA_AUDIT_TRACE GPU-submit trace

**Files:**
- Modify: `Sources/VectorAccelerate/Core/Metal4Context.swift` (methods at ~:301 `execute`, ~:342 `executeAndWait`, ~:418 `executeBlitAndWait`)

**Interfaces:**
- Produces: one `"[VA_AUDIT] gpu-submit"` stdout line per command-buffer submission when env `VA_AUDIT_TRACE=1`. Consumed by Task 6's awk attribution.

- [ ] **Step 1: Implement.** Add to `Metal4Context`:

```swift
/// Phase-0 audit trace: when the environment variable `VA_AUDIT_TRACE` is "1", every
/// command-buffer submission prints one line, so a test-suite run can be attributed
/// per test case. Cached once; zero cost when disabled.
static let auditTraceGPUSubmits: Bool = ProcessInfo.processInfo.environment["VA_AUDIT_TRACE"] == "1"
```

and at the top of each of the three execute methods: `if Self.auditTraceGPUSubmits { print("[VA_AUDIT] gpu-submit") }`.

- [ ] **Step 2: Verify** — `VA_AUDIT_TRACE=1 swift test --filter RoutingProvenanceTests 2>&1 | grep -c "gpu-submit"` → nonzero; without the env var → zero occurrences.

### Task 3: Runtime-library completeness guard (expected RED, then fix)

**Files:**
- Test: `Tests/VectorAccelerateTests/Hardening/ShaderLibraryCompletenessTests.swift`
- Modify: `Sources/VectorAccelerate/Core/KernelContext.swift` (`shaderFiles` list → internal `runtimeCompileShaderFiles` + explicit `runtimeCompileExcludedShaderFiles`)

**Interfaces:**
- Consumes: `KernelContext.findVectorAccelerateBundle()`, `KernelContext.makeLibraryFromBundleSources(device:bundle:)` (both already internal for tests; NormalizationParityTests is precedent — copy its bundle-acquisition mechanism if it differs).
- Produces: `KernelContext.runtimeCompileShaderFiles: [String]`, `KernelContext.runtimeCompileExcludedShaderFiles: [String: String]` (file → reason), consumed by Task 4's parity test.

- [ ] **Step 1: Write the test**

```swift
final class ShaderLibraryCompletenessTests: XCTestCase {
    /// Repo Shaders dir, reachable from the test file's compile-time path.
    private static let shadersDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders", isDirectory: true)

    private static func kernelNames(inMetalFile url: URL) throws -> [String] {
        let source = try String(contentsOf: url, encoding: .utf8)
        let regex = try NSRegularExpression(pattern: #"kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)"#)
        let range = NSRange(source.startIndex..., in: source)
        return regex.matches(in: source, range: range).map {
            String(source[Range($0.range(at: 1), in: source)!])
        }
    }

    /// Every kernel in every non-excluded .metal file must exist in the runtime-compiled
    /// library — the release-primary load path. One missing file silently strands its
    /// kernels (SoADistance.metal was missing at audit time: VA2-001).
    func testRuntimeCompiledLibraryContainsEveryKernel() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal device") }
        guard let bundle = KernelContext.findVectorAccelerateBundle() else {
            throw XCTSkip("No VectorAccelerate resource bundle with .metal sources")
        }
        let library = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        let available = Set(library.functionNames)
        let files = try FileManager.default.contentsOfDirectory(at: Self.shadersDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
        XCTAssertGreaterThanOrEqual(files.count, 30, "expected the full shader corpus at \(Self.shadersDir.path)")
        var missing: [String] = []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let base = file.deletingPathExtension().lastPathComponent
            if KernelContext.runtimeCompileExcludedShaderFiles[base] != nil { continue }
            for name in try Self.kernelNames(inMetalFile: file) where !available.contains(name) {
                missing.append("\(base).metal: \(name)")
            }
        }
        XCTAssertTrue(missing.isEmpty, "runtime-compiled library is missing \(missing.count) kernel(s):\n"
            + missing.joined(separator: "\n"))
    }

    /// The exclusion list may only name files that exist and must carry a reason.
    func testExclusionListIsCurrent() throws {
        let files = Set(try FileManager.default.contentsOfDirectory(at: Self.shadersDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
            .map { $0.deletingPathExtension().lastPathComponent })
        for (excluded, reason) in KernelContext.runtimeCompileExcludedShaderFiles {
            XCTAssertTrue(files.contains(excluded), "excluded file \(excluded).metal no longer exists")
            XCTAssertFalse(reason.isEmpty)
        }
        for listed in KernelContext.runtimeCompileShaderFiles {
            XCTAssertTrue(files.contains(listed), "listed file \(listed).metal no longer exists")
        }
        // Directory = listed ∪ excluded, no overlap.
        let listed = Set(KernelContext.runtimeCompileShaderFiles)
        let excluded = Set(KernelContext.runtimeCompileExcludedShaderFiles.keys)
        XCTAssertTrue(listed.isDisjoint(with: excluded))
        XCTAssertEqual(listed.union(excluded), files, "every .metal file must be either compiled or explicitly excluded with a reason")
    }
}
```

- [ ] **Step 2: Refactor the list.** In `KernelContext.swift`, hoist the local `shaderFiles` array to `internal static let runtimeCompileShaderFiles: [String]` (same contents), add `internal static let runtimeCompileExcludedShaderFiles: [String: String] = [:]`, and make `makeLibraryFromBundleSources` iterate `runtimeCompileShaderFiles`.

- [ ] **Step 3: Run to verify RED** — expected missing: `soa_l2_distance`, `soa_cosine_distance`, 4 manhattan, 5 chebyshev kernels, and `testExclusionListIsCurrent` fails on the 3 unlisted files. This failure output is audit evidence — capture it into the ledger before fixing.

- [ ] **Step 4: Fix.** Add `"SoADistance"`, `"ManhattanDistance"`, `"ChebyshevDistance"` to `runtimeCompileShaderFiles`. If the combined compile then fails from symbol conflicts, fall back to: keep `SoADistance` listed (it must compile — resolve any conflict it has), move the conflicting dead file(s) to `runtimeCompileExcludedShaderFiles` with reason `"dead shader (no Swift call sites); conflicts with combined-source compile — candidate for deletion, see AUDIT-2 VA2-002"`.

- [ ] **Step 5: Run to verify GREEN** — `swift test --filter ShaderLibraryCompletenessTests`, and re-run `--filter NormalizationParityTests` (same compile path) to confirm no regression.

### Task 4: Preamble ↔ Metal4Common.h parity guard

**Files:**
- Modify: `Sources/VectorAccelerate/Core/KernelContext.swift` (hoist preamble string to `internal static let runtimeCompilePreamble: String`)
- Test: `Tests/VectorAccelerateTests/Hardening/PreambleParityTests.swift`

**Interfaces:**
- Consumes: `KernelContext.runtimeCompilePreamble`, `KernelContext.runtimeCompileShaderFiles`, repo `Metal4Common.h` via `#filePath`.

- [ ] **Step 1: Hoist the preamble** — move the `combinedSource` initial string literal verbatim to `internal static let runtimeCompilePreamble`, and start `makeLibraryFromBundleSources` with `var combinedSource = runtimeCompilePreamble`.

- [ ] **Step 2: Write the test**

```swift
final class PreambleParityTests: XCTestCase {
    private static let shadersDir = /* same #filePath derivation as Task 3 */

    /// name → numeric-literal token, from `constant float|uint NAME = VALUE;` and `#define NAME VALUE`.
    private static func constants(in source: String) throws -> [String: String] {
        var out: [String: String] = [:]
        for pattern in [#"constant\s+(?:float|uint)\s+(VA_[A-Z0-9_]+)\s*=\s*([^;]+);"#,
                        #"#define\s+(VA_[A-Z0-9_]+|EPSILON)\s+(\S+)"#] {
            let regex = try NSRegularExpression(pattern: pattern)
            for m in regex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                out[String(source[Range(m.range(at: 1), in: source)!])] =
                    String(source[Range(m.range(at: 2), in: source)!]).trimmingCharacters(in: .whitespaces)
            }
        }
        return out
    }

    private static func numeric(_ token: String) -> Double? {
        var t = token.trimmingCharacters(in: .whitespaces)
        if t.hasSuffix("f") || t.hasSuffix("h") { t.removeLast() }
        if t == "INFINITY" { return .infinity }
        if t.hasPrefix("0x"), t.contains("p") { return Double(t) }           // hex float
        if t.hasPrefix("0x") { return UInt64(t.dropFirst(2), radix: 16).map(Double.init) }
        return Double(t)
    }

    /// Every constant the preamble redefines must be numerically identical to Metal4Common.h.
    func testPreambleConstantsMatchHeader() throws {
        let header = try String(contentsOf: Self.shadersDir.appendingPathComponent("Metal4Common.h"), encoding: .utf8)
        let headerConsts = try Self.constants(in: header)
        var preambleConsts = try Self.constants(in: KernelContext.runtimeCompilePreamble)
        // The preamble expresses two header constants via textual replacement, not #define:
        preambleConsts["VA_EPSILON"] = preambleConsts["EPSILON"]         // VA_EPSILON → EPSILON
        preambleConsts["VA_INVALID_INDEX"] = "0xFFFFFFFF"                // literal substitution
        var mismatches: [String] = []
        for (name, headerToken) in headerConsts {
            guard let preambleToken = preambleConsts[name] else { continue }  // not redefined → header-only
            guard let h = Self.numeric(headerToken), let p = Self.numeric(preambleToken) else {
                mismatches.append("\(name): unparseable (header=\(headerToken) preamble=\(preambleToken))"); continue
            }
            if h != p { mismatches.append("\(name): header=\(headerToken) preamble=\(preambleToken)") }
        }
        XCTAssertTrue(mismatches.isEmpty, "preamble/header numeric drift:\n" + mismatches.joined(separator: "\n"))
    }

    /// Any VA_* symbol a compiled file *uses* must be provided by the preamble, its textual
    /// replacements, or a local definition in that same file — otherwise the ONE combined
    /// compile fails and every kernel in the package disappears at runtime.
    func testCompiledFilesUseOnlyCoveredSymbols() throws {
        let provided = Set(try Self.constants(in: KernelContext.runtimeCompilePreamble).keys)
            .union(["VA_EPSILON", "VA_INVALID_INDEX", "EPSILON"])        // replacement-covered
        let useRegex = try NSRegularExpression(pattern: #"\bVA_[A-Z][A-Z0-9_]*\b"#)
        var uncovered: [String] = []
        for base in KernelContext.runtimeCompileShaderFiles {
            let url = Self.shadersDir.appendingPathComponent(base + ".metal")
            var source = try String(contentsOf: url, encoding: .utf8)
            source = source.split(separator: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")                                  // strip line comments
            let local = Set(try Self.constants(in: source).keys)
            let localMacroRegex = try NSRegularExpression(pattern: #"#(?:define|ifndef|ifdef|if|undef)\s+\S*|#include\s+\S+"#)
            let stripped = localMacroRegex.stringByReplacingMatches(in: source, range: NSRange(source.startIndex..., in: source), withTemplate: "")
            for m in useRegex.matches(in: stripped, range: NSRange(stripped.startIndex..., in: stripped)) {
                let sym = String(stripped[Range(m.range, in: stripped)!])
                if !provided.contains(sym), !local.contains(sym), sym != "VA_ATOMIC_TYPES_DEFINED" {
                    uncovered.append("\(base).metal: \(sym)")
                }
            }
        }
        XCTAssertTrue(Set(uncovered).isEmpty, "symbols not covered by preamble/replacements:\n"
            + Set(uncovered).sorted().joined(separator: "\n"))
    }
}
```

- [ ] **Step 3: Run** — `swift test --filter PreambleParityTests`. Constants test expected GREEN today (drift guard for the future). Coverage test may legitimately surface symbols like `VA_SENTINEL_INDEX`/`va_*` helper uses in newly-added files (SoADistance from Task 3 — it includes `Metal4Common.h`); resolve by extending the preamble minimally (mirroring the header value verbatim) — never by loosening the test. Iterate until GREEN with the combined compile (Task 3's test) also GREEN.

### Task 5: Differential GPU↔CPU harness

**Files:**
- Test: `Tests/VectorAccelerateTests/Hardening/DifferentialKernelVsCPUTests.swift`

**Interfaces:**
- Consumes: `L2KernelDistanceProvider(context:)` / `CosineKernelDistanceProvider(context:)` `.batchDistance(from:to:metric:)` (public, kernel-direct = forced GPU); `Metal4ComputeEngine(context:decisionEngine:)` `.fusedDistanceTopK(query:database:k:metric:)`; `MetalComputeProvider` SoA extension (public); `AccelerateFallback.batchEuclideanDistance` / `.batchCosineSimilarity` (@testable); Task 1's telemetry for SoA provenance.

- [ ] **Step 1: Fixture generator + oracle + comparator (one file, real code):**

```swift
enum ValueClass: String, CaseIterable {
    case normal, zeros, duplicates, huge, tiny, subnormal, mixedScale, nanPoisoned, posInfPoisoned, negInfPoisoned
}

func makeMatrix(_ cls: ValueClass, n: Int, dim: Int, rng: inout TestRNG) -> [[Float]] {
    func randRow() -> [Float] { (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
    switch cls {
    case .normal:     return (0..<n).map { _ in randRow() }
    case .zeros:      return Array(repeating: [Float](repeating: 0, count: dim), count: n)
    case .duplicates: let r = randRow(); return Array(repeating: r, count: n)
    case .huge:       return (0..<n).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) * 1e19 } }
    case .tiny:       return (0..<n).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) * 1e-20 } }
    case .subnormal:  return (0..<n).map { _ in [Float](repeating: 1e-40, count: dim) }
    case .mixedScale: return (0..<n).map { i in (0..<dim).map { j in (i + j).isMultiple(of: 2) ? rng.nextFloat(in: -1...1) * 1e18 : rng.nextFloat(in: -1...1) * 1e-18 } }
    case .nanPoisoned:    var m = (0..<n).map { _ in randRow() }; m[n/2][dim/2] = .nan; return m
    case .posInfPoisoned: var m = (0..<n).map { _ in randRow() }; m[n/2][dim/2] = .infinity; return m
    case .negInfPoisoned: var m = (0..<n).map { _ in randRow() }; m[n/2][dim/2] = -.infinity; return m
    }
}

/// Double-accumulation oracle (independent of both legs under test).
func oracleL2(_ q: [Float], _ c: [Float]) -> Float {
    var s = 0.0; for i in 0..<q.count { let d = Double(q[i]) - Double(c[i]); s += d * d }
    return Float(s.squareRoot())
}
func oracleCosineDistance(_ q: [Float], _ c: [Float]) -> Float {
    var dot = 0.0, nq = 0.0, nc = 0.0
    for i in 0..<q.count { dot += Double(q[i]) * Double(c[i]); nq += Double(q[i]) * Double(q[i]); nc += Double(c[i]) * Double(c[i]) }
    return Float(1.0 - dot / (nq.squareRoot() * nc.squareRoot()))   // NaN for zero norms: flagged, not asserted
}

/// Semantic float agreement: both NaN, or bitwise-equal (covers ±Inf), or within mixed tolerance.
func agree(_ a: Float, _ b: Float, relTol: Float = 2e-4, absTol: Float = 1e-5) -> Bool {
    if a.isNaN && b.isNaN { return true }
    if a == b { return true }
    if a.isNaN != b.isNaN || a.isInfinite != b.isInfinite { return false }
    return abs(a - b) <= max(absTol, relTol * max(abs(a), abs(b)))
}
```

- [ ] **Step 2: Batch-distance differential (per metric, aggregated diagnostics):**

```swift
func testL2BatchGPUMatchesCPUFallback() async throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
    let context = try await Metal4Context()
    let gpu = try await L2KernelDistanceProvider(context: context)
    var failures: [String] = []
    var rng = TestRNG(seed: 0xD1FF)
    for dim in [1, 3, 7, 16, 33, 128, 384, 768] {
        for n in [1, 2, 33, 257, 1000] {
            for cls in ValueClass.allCases {
                let cands = makeMatrix(cls, n: n, dim: dim, rng: &rng)
                let query = makeMatrix(cls == .zeros ? .normal : cls, n: 1, dim: dim, rng: &rng)[0]
                let qv = DynamicVector(query); let cv = cands.map(DynamicVector.init)
                let g = try await gpu.batchDistance(from: qv, to: cv, metric: .euclidean)
                let c = AccelerateFallback.batchEuclideanDistance(query: query, candidates: cands)
                for j in 0..<n where !agree(g[j], c[j]) {
                    failures.append("dim=\(dim) n=\(n) \(cls) j=\(j): gpu=\(g[j]) cpu=\(c[j]) oracle=\(oracleL2(query, cands[j]))")
                    break   // one row per cell keeps output readable
                }
            }
        }
    }
    XCTAssertTrue(failures.isEmpty, "\(failures.count) diverging cells:\n" + failures.joined(separator: "\n"))
}
// testCosineBatchGPUMatchesCPUFallback: same loop; gpu = CosineKernelDistanceProvider,
// metric .cosine, cpu = AccelerateFallback.batchCosineSimilarity(...).map { 1 - $0 },
// oracle = oracleCosineDistance.
```

- [ ] **Step 3: Fused top-K differential + determinism + empty-result probe:**

```swift
func testFusedTopKMatchesCPUSelection() async throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device") }
    let context = try await Metal4Context()
    let engine = try await Metal4ComputeEngine(context: context, decisionEngine: GPUDecisionEngine())
    var failures: [String] = []
    var rng = TestRNG(seed: 0x70CC)
    for (n, dim) in [(33, 16), (257, 16), (1000, 768), (5000, 64)] {
        for k in [1, 8, 32, 33, 128, n, n + 7] {
            for cls in [ValueClass.normal, .duplicates, .nanPoisoned] {
                let db = makeMatrix(cls, n: n, dim: dim, rng: &rng)
                let q  = makeMatrix(.normal, n: 1, dim: dim, rng: &rng)[0]
                let g1 = try await engine.fusedDistanceTopK(query: q, database: db, k: k, metric: .euclidean)
                let g2 = try await engine.fusedDistanceTopK(query: q, database: db, k: k, metric: .euclidean)
                let expected = min(k, n)
                if g1.isEmpty { failures.append("EMPTY GPU RESULT n=\(n) k=\(k) \(cls) — provider would silently fall back"); continue }
                if g1.count != expected { failures.append("count n=\(n) k=\(k) \(cls): got \(g1.count) expected \(expected)") }
                if !(g1.map(\.index) == g2.map(\.index) && zip(g1, g2).allSatisfy { $0.distance.bitPattern == $1.distance.bitPattern }) {
                    failures.append("NONDETERMINISTIC n=\(n) k=\(k) \(cls)")
                }
                // CPU reference: full distances via fallback, VectorCore selection, smaller-index ties.
                let dists = AccelerateFallback.batchEuclideanDistance(query: q, candidates: db)
                let ref = MetalComputeProvider.selectTopK(dists, k: expected, largerIsCloser: false)
                for (i, (gr, rr)) in zip(g1, ref).enumerated() where !(agree(gr.distance, rr.distance) && (gr.index == rr.index || agree(dists[gr.index], rr.distance))) {
                    failures.append("n=\(n) k=\(k) \(cls) rank=\(i): gpu=(\(gr.index),\(gr.distance)) ref=(\(rr.index),\(rr.distance))")
                    break
                }
            }
        }
    }
    XCTAssertTrue(failures.isEmpty, "\(failures.count) fused-topK findings:\n" + failures.joined(separator: "\n"))
}
```

- [ ] **Step 4: SoA differential** — same value classes over `provider.batchDistance(query:against:metric:)` and `provider.findNearest(query:in:k:metric:)` for both metrics, N ∈ {1, 33, 1000}, dim ∈ {16, 768}, compared against `AccelerateFallback` + `selectTopK` reference; assert telemetry `gpuKernel` grew by exactly the number of SoA calls (provenance).

- [ ] **Step 5: Run, iterate** — `swift test --filter DifferentialKernelVsCPUTests`. Divergences are FINDINGS: record each in the ledger with the failing cell line verbatim. Where a divergence is real and severe, keep the failing assertion (it is the regression test); where a divergence is a semantics *decision* (e.g., zero-norm cosine convention), document the observed contract in the ledger and encode the observed behavior in the test with a comment linking the ledger entry.

### Task 6: Baseline measurement runs

**Files:** none (scratchpad logs only)

- [ ] **Step 1:** Pristine release-config suite — already running in worktree (`release-baseline.log`); collect exit code + failure lines when it lands.
- [ ] **Step 2:** In-repo, with all new code: `swift test -c release --filter Hardening 2>&1 | tee <scratchpad>/hardening-release.log` — the new guards must behave identically in release.
- [ ] **Step 3:** Trace measurement: `VA_AUDIT_TRACE=1 swift test 2>&1 | tee <scratchpad>/audit-trace.log`, then attribute:

```bash
awk '/Test Case .* started\./ {tc=$4} /\[VA_AUDIT\] gpu-submit/ {c[tc]++; total++}
     END {print "TOTAL gpu submissions:", total; for (t in c) print c[t], t | "sort -rn"}' audit-trace.log
```

and cross-list test classes whose names contain `Metal|GPU|Kernel` but attributed zero submissions.
- [ ] **Step 4:** Shader-validation leg: `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 swift test --filter "Hardening" 2>&1 | tee <scratchpad>/shader-validation.log`; grep for validation faults (OOB reads/writes surface here even when outputs look plausible).

### Task 7: Findings ledger

**Files:**
- Create: `docs/audits/AUDIT-2.md`

- [ ] **Step 1:** Write the ledger: one entry per confirmed finding — ID (VA2-NNN), severity, one-line claim, evidence (file:line + failing-test output), blast radius, recommended fix, status (`confirmed` / `fixed-in-tree` / `decision-needed`). Seed entries: VA2-001 SoADistance missing from runtime compile (fixed-in-tree, Task 3); VA2-002 dead Manhattan/Chebyshev shaders (decision-needed); VA2-003 batchDistance GPU path unreachable under defaults (decision-needed — routing fix changes behavior); VA2-004 provider tests compare CPU to CPU (confirmed via telemetry); VA2-005 empty-GPU-result silent fallthrough (decision-needed); plus everything Tasks 5–6 surface.
- [ ] **Step 2:** Update the session memory file with results; deliver the report.

## Self-Review

- Spec coverage: Phase 0 (provenance=T1, trace+runs=T2/T6, release leg=T6, skip inventory folded into T6 report), 1.1 (T5), 1.2 (T3+T4, mathMode decision deferred to ledger as decision-needed) — covered.
- Placeholders: none; all code blocks are concrete (comparator/oracle/regexes/commands written out).
- Type consistency: `RoutingTelemetry` field names used in T1 tests match the T1 implementation; `runtimeCompileShaderFiles`/`runtimeCompileExcludedShaderFiles`/`runtimeCompilePreamble` names consistent across T3/T4.
