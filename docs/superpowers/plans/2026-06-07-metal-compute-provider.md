# MetalComputeProvider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `MetalComputeProvider` — one canonical GPU compute façade for VectorAccelerate that unifies the scattered GPU distance/search surface behind batch distance, k-NN, distance-matrix, and single-distance methods, routes GPU-vs-CPU via `GPUDecisionEngine`, conforms to VectorCore's `ComputeProvider` as an honest capability shim, aggressively deprecates the superseded surface, and files the VectorCore R4 transparent-dispatch request.

**Architecture:** A `public actor MetalComputeProvider` composes the existing proven pieces — `Metal4Context`, `Metal4ComputeEngine`, `GPUDecisionEngine`, the no-copy `L2KernelDistanceProvider`/`CosineKernelDistanceProvider` (GPU euclidean/cosine), and `AccelerateFallback` (CPU). Each public method validates input, asks the decision engine whether to use GPU, dispatches to the no-copy kernel path on a GPU vote or the Accelerate path on a CPU vote (or on GPU error when `fallbackToCPU`), and returns results whose per-metric semantics match the existing engine (euclidean → L2 distance; cosine → 1−similarity; dotProduct → raw dot; manhattan → L1; chebyshev → L∞). The `ComputeProvider` conformance is a thin nonisolated shim (`device=.gpu`, real `deviceInfo`, `execute`=run-the-closure, `parallel*`=inherited TaskGroup defaults) documented as scheduling-only.

**Tech Stack:** Swift 6.2 actors, Metal 4 (`@preconcurrency import Metal`), Accelerate (vDSP via `AccelerateFallback`), VectorCore 0.2.2 (`ComputeProvider`, `ComputeDevice`, `ComputeDeviceInfo`, `SupportedDistanceMetric`, `VectorProtocol`, `DynamicVector`, `VectorError`), XCTest.

**Branch:** `gifton/metal-compute-provider` (already checked out; stacked on `gifton/vectorcore-integration-groundwork` / PR #31).

**Reference spec:** `docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md`

---

## Notes for the implementer (read once)

- **Commits:** This repo's standing rule is "commit only when the user asks." The commit steps below are the intended rhythm, but at execution time get the user's go-ahead before committing (e.g. ask once "commit per task, or batch at the end?"). Do not push.
- **Metal gating:** Everything is `@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)`. Tests `throw XCTSkip(...)` when `MTLCreateSystemDefaultDevice() == nil`. The dev machine is Apple Silicon with Metal, so tests run here.
- **`ExistentialAny` is enabled** — spell every existential `any` (`any DistanceMetric`, `any MTLDevice`). Provider code uses concrete types, so this mostly matters in the R4 doc snippet.
- **"Verify it fails":** for a not-yet-written Swift method, the failure is a compile error in the test target (`value of type 'MetalComputeProvider' has no member 'batchDistance'`). That is the red state. Run the single test with `swift test --filter`.
- **Build is slow** (large Metal package). Expect each `swift build`/`swift test` to take a while; that's normal.
- **Per-metric return semantics** (match existing engine, do not redefine): euclidean = L2 distance (with sqrt); cosine = `1 − similarity`; dotProduct = raw dot product (a similarity, larger = nearer); manhattan = L1; chebyshev = L∞.

---

## File Structure

| File | Responsibility |
|---|---|
| `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift` | **New.** The provider actor: ComputeProvider shim + `batchDistance`/`findNearest`/`distanceMatrix`/`distance` + private routing/CPU helpers. |
| `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift` | **New.** Conformance, GPU/CPU parity, routing, k-NN, matrix, single-distance, deprecation-integrity tests + naive CPU reference helpers. |
| `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift` | **Modify.** Deprecate + rewire the 3 `BatchOperations.*GPU` statics to delegate; deprecate-annotate `AcceleratedDistanceProvider` and the distance convenience extensions. |
| `docs/VECTORCORE_INTEGRATION_REQUESTS.md` | **Modify.** Append **R4** (transparent-dispatch hook request). |
| `CHANGELOG.md` | **Modify.** Add `Added` (provider) + `Deprecated` (scattered surface) entries. |

---

## Task 1: Provider skeleton + `ComputeProvider` shim

**Files:**
- Create: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Test: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

- [ ] **Step 1: Write the failing test**

Create `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`:

```swift
//
//  MetalComputeProviderTests.swift
//  VectorAccelerateTests
//
//  Tests for the unified GPU compute façade. See
//  docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md
//

import XCTest
@testable import VectorAccelerate
import VectorCore
@preconcurrency import Metal

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MetalComputeProviderTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    func testConformsToComputeProviderWithGPUIdentity() async throws {
        let provider = try await MetalComputeProvider()
        // ComputeProvider shim reports GPU identity with real capabilities.
        XCTAssertEqual(provider.device, .gpu(index: 0))
        XCTAssertGreaterThan(provider.maxConcurrency, 0)
        XCTAssertFalse(provider.deviceInfo.name.isEmpty)
        XCTAssertGreaterThan(provider.deviceInfo.maxThreads, 0)
        // It is usable as an existential ComputeProvider, and `execute` runs the closure.
        let p: any ComputeProvider = provider
        let answer = try await p.execute { 41 + 1 }
        XCTAssertEqual(answer, 42)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter MetalComputeProviderTests/testConformsToComputeProviderWithGPUIdentity`
Expected: FAIL — compile error `cannot find 'MetalComputeProvider' in scope`.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`:

```swift
//
//  MetalComputeProvider.swift
//  VectorAccelerate
//
//  The single canonical GPU compute façade. Composes Metal4Context + Metal4ComputeEngine +
//  GPUDecisionEngine + the no-copy kernel distance providers + AccelerateFallback, routing
//  GPU-vs-CPU per call. Conforms to VectorCore's ComputeProvider as an honest capability shim.
//
//  See docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md
//

import Foundation
import VectorCore
@preconcurrency import Metal

/// GPU compute façade for VectorAccelerate.
///
/// Use the dedicated methods (`batchDistance`, `findNearest`, `distanceMatrix`, `distance`) for
/// GPU-accelerated work. The `ComputeProvider` conformance is a capability/identity shim: it reports
/// `device == .gpu` with real device info, but its `execute`/`parallel*` methods run their (CPU)
/// closures via Swift concurrency — installing this as `Operations.computeProvider` does **not** move
/// VectorCore's Operations onto the GPU (those closures are CPU kernels; it will not be slower than
/// `CPUComputeProvider`, but it will not accelerate). For transparent GPU dispatch through VectorCore's
/// own API, see VectorCore request R4 in `docs/VECTORCORE_INTEGRATION_REQUESTS.md`.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor MetalComputeProvider: ComputeProvider {

    /// Routing/fallback configuration.
    public struct Configuration: Sendable {
        /// When false, every call takes the CPU path (the decision engine is not consulted).
        public var preferGPU: Bool
        /// When true, a GPU kernel error falls back to the CPU path instead of throwing.
        public var fallbackToCPU: Bool
        public init(preferGPU: Bool = true, fallbackToCPU: Bool = true) {
            self.preferGPU = preferGPU
            self.fallbackToCPU = fallbackToCPU
        }
    }

    // Composed collaborators.
    private let context: Metal4Context
    private let engine: Metal4ComputeEngine
    private let decisionEngine: GPUDecisionEngine
    private let l2Provider: L2KernelDistanceProvider
    private let cosineProvider: CosineKernelDistanceProvider
    private let configuration: Configuration

    // Nonisolated ComputeProvider shim state, captured at init (no actor hop on access).
    private nonisolated let _device: ComputeDevice
    private nonisolated let _maxConcurrency: Int
    private nonisolated let _deviceInfo: ComputeDeviceInfo

    // MARK: - ComputeProvider conformance (scheduling-only shim)

    public nonisolated var device: ComputeDevice { _device }
    public nonisolated var maxConcurrency: Int { _maxConcurrency }
    public nonisolated var deviceInfo: ComputeDeviceInfo { _deviceInfo }

    public func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
    // parallelExecute / parallelForEach / parallelReduce: inherited from the ComputeProvider
    // protocol's default TaskGroup implementations (correct, never slower than CPUComputeProvider).

    // MARK: - Initialization

    /// Primary initializer over an existing Metal4Context.
    public init(
        context: Metal4Context,
        configuration: Configuration = .init(),
        decisionEngine: GPUDecisionEngine? = nil
    ) async throws {
        self.context = context
        self.configuration = configuration
        let resolvedDecision = decisionEngine ?? GPUDecisionEngine(context: context)
        self.decisionEngine = resolvedDecision
        self.engine = try await Metal4ComputeEngine(context: context, decisionEngine: resolvedDecision)
        self.l2Provider = try await L2KernelDistanceProvider(context: context)
        self.cosineProvider = try await CosineKernelDistanceProvider(context: context)

        // `context.device` is nonisolated; `rawDevice` is a nonisolated `any MTLDevice`.
        let raw = context.device.rawDevice
        self._device = .gpu(index: 0)
        self._maxConcurrency = raw.maxThreadsPerThreadgroup.width
        self._deviceInfo = ComputeDeviceInfo(
            name: raw.name,
            availableMemory: Int(raw.recommendedMaxWorkingSetSize),
            maxThreads: raw.maxThreadsPerThreadgroup.width,
            preferredChunkSize: 1024
        )
    }

    /// Convenience initializer that builds a default Metal4Context.
    public init(configuration: Configuration = .init()) async throws {
        guard ComputeDevice.gpu().isAvailable else { throw VectorError.metalNotAvailable() }
        let ctx = try await Metal4Context()
        try await self.init(context: ctx, configuration: configuration)
    }

    /// Build a provider with all defaults.
    public static func makeDefault() async throws -> MetalComputeProvider {
        try await MetalComputeProvider()
    }
}
```

Note: `GPUDecisionEngine(context:)` is an `async` initializer (no `throws`), so `decisionEngine ?? GPUDecisionEngine(context: context)` must be written with `await`. If the compiler rejects the `??` with `await`, use:
```swift
        let resolvedDecision: GPUDecisionEngine
        if let decisionEngine { resolvedDecision = decisionEngine }
        else { resolvedDecision = await GPUDecisionEngine(context: context) }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter MetalComputeProviderTests/testConformsToComputeProviderWithGPUIdentity`
Expected: PASS (or SKIPPED if the runner has no Metal device — passes on Apple Silicon).

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "Add MetalComputeProvider skeleton + ComputeProvider shim"
```

---

## Task 2: `batchDistance` — routing + GPU (no-copy) + CPU fallback

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Modify: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

- [ ] **Step 1: Write the failing tests** (append to the test file, inside the class)

First add the reference helpers and a vector factory (place these as `private` methods in the test class):

```swift
    // MARK: - CPU reference helpers (independent ground truth)

    private func makeVectors(count: Int, dim: Int, seed: UInt64) -> (query: DynamicVector, candidates: [DynamicVector]) {
        // Deterministic LCG (Date/Math.random are intentionally avoided).
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1   // [-1, 1)
        }
        let query = DynamicVector((0..<dim).map { _ in next() })
        let candidates = (0..<count).map { _ in DynamicVector((0..<dim).map { _ in next() }) }
        return (query, candidates)
    }

    private func refEuclidean(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { let d = a[i] - b[i]; s += d * d }; return s.squareRoot()
    }
    private func refCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom < .leastNormalMagnitude ? 1.0 : 1.0 - (dot / denom)
    }
    private func refManhattan(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { s += abs(a[i] - b[i]) }; return s
    }

    private func assertClose(_ a: [Float], _ b: [Float], tol: Float = 1e-2, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "count mismatch", file: file, line: line)
        for i in 0..<min(a.count, b.count) {
            let scale = max(1, abs(b[i]))
            XCTAssertEqual(a[i], b[i], accuracy: tol * scale, "index \(i)", file: file, line: line)
        }
    }
```

Now the tests:

```swift
    func testBatchDistanceEuclideanParity_CPUandGPU() async throws {
        for dim in [64, 768, 1536] {
            let (q, cands) = makeVectors(count: 1100, dim: dim, seed: UInt64(dim))
            let reference = cands.map { refEuclidean(q.toArray(), $0.toArray()) }

            // CPU path (preferGPU: false bypasses the decision engine).
            let cpu = try await MetalComputeProvider(configuration: .init(preferGPU: false))
            let cpuOut = try await cpu.batchDistance(query: q, candidates: cands, metric: .euclidean)
            assertClose(cpuOut, reference)

            // GPU path (default config; 1100 candidates clears the routing minimums).
            let gpu = try await MetalComputeProvider()
            let gpuOut = try await gpu.batchDistance(query: q, candidates: cands, metric: .euclidean)
            assertClose(gpuOut, reference)
        }
    }

    func testBatchDistanceCosineParity_CPUandGPU() async throws {
        for dim in [64, 768] {
            let (q, cands) = makeVectors(count: 1100, dim: dim, seed: UInt64(dim) &+ 7)
            let reference = cands.map { refCosineDistance(q.toArray(), $0.toArray()) }

            let cpu = try await MetalComputeProvider(configuration: .init(preferGPU: false))
            assertClose(try await cpu.batchDistance(query: q, candidates: cands, metric: .cosine), reference)

            let gpu = try await MetalComputeProvider()
            assertClose(try await gpu.batchDistance(query: q, candidates: cands, metric: .cosine), reference)
        }
    }

    func testBatchDistanceManhattanFallsBackToCPU() async throws {
        let (q, cands) = makeVectors(count: 300, dim: 128, seed: 99)
        let reference = cands.map { refManhattan(q.toArray(), $0.toArray()) }
        let provider = try await MetalComputeProvider()   // manhattan has no GPU batch kernel → CPU
        assertClose(try await provider.batchDistance(query: q, candidates: cands, metric: .manhattan), reference)
    }

    func testBatchDistanceRejectsRaggedCandidates() async throws {
        let provider = try await MetalComputeProvider()
        let q = DynamicVector([1, 2, 3, 4])
        let cands = [DynamicVector([1, 2, 3, 4]), DynamicVector([1, 2, 3])]   // ragged
        do {
            _ = try await provider.batchDistance(query: q, candidates: cands, metric: .euclidean)
            XCTFail("expected dimension-mismatch rejection")
        } catch { /* expected */ }
    }

    func testBatchDistanceEmptyCandidatesReturnsEmpty() async throws {
        let provider = try await MetalComputeProvider()
        let out = try await provider.batchDistance(query: DynamicVector([1, 2, 3]), candidates: [], metric: .euclidean)
        XCTAssertTrue(out.isEmpty)
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter MetalComputeProviderTests/testBatchDistanceEuclideanParity_CPUandGPU`
Expected: FAIL — compile error `value of type 'MetalComputeProvider' has no member 'batchDistance'`.

- [ ] **Step 3: Write the implementation** (add to `MetalComputeProvider`, inside the actor)

```swift
    // MARK: - Batch distance

    /// Distance from `query` to each candidate. Per-metric semantics match the existing engine
    /// (euclidean → L2 distance; cosine → 1−similarity; dotProduct → raw dot; manhattan → L1;
    /// chebyshev → L∞). Euclidean/cosine use the no-copy GPU kernel path on a GPU vote; everything
    /// else (and the CPU vote / GPU error with `fallbackToCPU`) uses Accelerate.
    public func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: SupportedDistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        guard !candidates.isEmpty else { return [] }
        let dim = query.withUnsafeBufferPointer { $0.count }
        guard candidates.allSatisfy({ $0.withUnsafeBufferPointer { $0.count } == dim }) else {
            throw VectorError.invalidInput("All candidates must match the query dimension (\(dim))")
        }

        switch metric {
        case .euclidean:
            let cpu = { AccelerateFallback.batchEuclideanDistance(query: query.toArray(), candidates: candidates.map { $0.toArray() }) }
            guard await routeToGPU(.l2Distance, candidateCount: candidates.count, k: 0, dimension: dim) else { return cpu() }
            do { return try await l2Provider.batchDistance(from: query, to: candidates, metric: .euclidean) }
            catch { if configuration.fallbackToCPU { return cpu() } else { throw error } }

        case .cosine:
            let cpu = { AccelerateFallback.batchCosineSimilarity(query: query.toArray(), candidates: candidates.map { $0.toArray() }).map { 1.0 - $0 } }
            guard await routeToGPU(.cosineSimilarity, candidateCount: candidates.count, k: 0, dimension: dim) else { return cpu() }
            do { return try await cosineProvider.batchDistance(from: query, to: candidates, metric: .cosine) }
            catch { if configuration.fallbackToCPU { return cpu() } else { throw error } }

        case .dotProduct:
            return AccelerateFallback.batchDotProduct(query: query.toArray(), candidates: candidates.map { $0.toArray() })

        case .manhattan:
            let q = query.toArray()
            return try candidates.map { try AccelerateFallback.manhattanDistance(q, $0.toArray()) }

        case .chebyshev:
            let q = query.toArray()
            return candidates.map { Self.chebyshev(q, $0.toArray()) }
        }
    }

    // MARK: - Private routing / CPU helpers

    /// Ask the decision engine whether to use the GPU for an operation (honoring `preferGPU`).
    private func routeToGPU(_ op: GPUOperation, candidateCount: Int, k: Int, dimension: Int) async -> Bool {
        guard configuration.preferGPU else { return false }
        return await decisionEngine.shouldUseGPU(
            operation: op,
            vectorCount: candidateCount,
            candidateCount: candidateCount,
            k: k,
            dimension: dimension
        )
    }

    /// Chebyshev (L∞) distance — no Accelerate primitive; computed inline.
    static func chebyshev(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
        return m
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter MetalComputeProviderTests/testBatchDistance`
Expected: PASS for all five `testBatchDistance*` tests.

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "MetalComputeProvider: batchDistance with GPU/CPU routing + fallback"
```

---

## Task 3: `findNearest` (top-K)

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Modify: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

- [ ] **Step 1: Write the failing test** (append to the test class)

```swift
    func testFindNearestMatchesCPUReference() async throws {
        let dim = 256
        let (q, cands) = makeVectors(count: 1200, dim: dim, seed: 1234)
        let k = 10

        // Reference: brute-force euclidean top-k (ascending distance).
        let qa = q.toArray()
        let refPairs = cands.enumerated()
            .map { (index: $0.offset, distance: refEuclidean(qa, $0.element.toArray())) }
            .sorted { $0.distance < $1.distance }
            .prefix(k)
        let refIndices = refPairs.map { $0.index }

        let provider = try await MetalComputeProvider()
        let out = try await provider.findNearest(query: q, in: cands, k: k, metric: .euclidean)

        XCTAssertEqual(out.count, k)
        XCTAssertEqual(out.map { $0.index }, refIndices, "top-k indices should match brute force")
        for (got, ref) in zip(out, refPairs) {
            XCTAssertEqual(got.distance, ref.distance, accuracy: max(1e-2, abs(ref.distance) * 1e-3))
        }
    }

    func testFindNearestEmptyAndKClamp() async throws {
        let provider = try await MetalComputeProvider()
        let (q, cands) = makeVectors(count: 3, dim: 8, seed: 5)
        XCTAssertTrue(try await provider.findNearest(query: q, in: [], k: 5, metric: .euclidean).isEmpty)
        // k larger than candidate count clamps to candidate count.
        let out = try await provider.findNearest(query: q, in: cands, k: 100, metric: .euclidean)
        XCTAssertEqual(out.count, 3)
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter MetalComputeProviderTests/testFindNearestMatchesCPUReference`
Expected: FAIL — `has no member 'findNearest'`.

- [ ] **Step 3: Write the implementation** (add to the actor)

```swift
    // MARK: - k-Nearest Neighbors (top-K)

    /// Return the `k` nearest candidates as (index, distance), ordered nearest-first. Euclidean/cosine
    /// use the fused GPU distance+top-K kernel on a GPU vote; other metrics (and CPU vote / GPU error
    /// with `fallbackToCPU`) compute `batchDistance` then select top-K on the CPU.
    public func findNearest<V: VectorProtocol>(
        query: V, in candidates: [V], k: Int, metric: SupportedDistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, !candidates.isEmpty else { return [] }
        let dim = query.withUnsafeBufferPointer { $0.count }
        guard candidates.allSatisfy({ $0.withUnsafeBufferPointer { $0.count } == dim }) else {
            throw VectorError.invalidInput("All candidates must match the query dimension (\(dim))")
        }
        let effectiveK = min(k, candidates.count)

        // GPU fused path (euclidean/cosine only).
        if metric == .euclidean || metric == .cosine,
           await routeToGPU(.topKSelection, candidateCount: candidates.count, k: effectiveK, dimension: dim) {
            let m: Metal4DistanceMetric = (metric == .euclidean) ? .euclidean : .cosine
            do {
                let result = try await engine.fusedDistanceTopK(
                    query: query.toArray(), database: candidates.map { $0.toArray() },
                    k: effectiveK, metric: m
                )
                if !result.isEmpty { return result }
            } catch {
                if !configuration.fallbackToCPU { throw error }
            }
        }

        // CPU path: batch distance then select top-K.
        let distances = try await batchDistance(query: query, candidates: candidates, metric: metric)
        return Self.selectTopK(distances, k: effectiveK, largerIsCloser: metric == .dotProduct)
    }

    /// Select the k nearest (index, distance) pairs. For similarity metrics (dotProduct) larger is
    /// nearer; for distance metrics smaller is nearer.
    static func selectTopK(_ distances: [Float], k: Int, largerIsCloser: Bool) -> [(index: Int, distance: Float)] {
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }
        let sorted = largerIsCloser ? pairs.sorted { $0.distance > $1.distance }
                                    : pairs.sorted { $0.distance < $1.distance }
        return Array(sorted.prefix(k))
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter MetalComputeProviderTests/testFindNearest`
Expected: PASS for both `testFindNearest*` tests.

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "MetalComputeProvider: findNearest (fused GPU top-K + CPU fallback)"
```

---

## Task 4: `distanceMatrix`

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Modify: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
    func testDistanceMatrixMatchesReference() async throws {
        let dim = 64
        let (_, vectors) = makeVectors(count: 40, dim: dim, seed: 314)   // reuse candidates as both sets
        let arrays = vectors.map { $0.toArray() }

        let provider = try await MetalComputeProvider(configuration: .init(preferGPU: false))   // deterministic CPU
        let matrix = try await provider.distanceMatrix(queries: vectors, candidates: vectors, metric: .euclidean)

        XCTAssertEqual(matrix.count, vectors.count)
        for i in 0..<vectors.count {
            XCTAssertEqual(matrix[i].count, vectors.count)
            XCTAssertEqual(matrix[i][i], 0, accuracy: 1e-3, "self-distance is zero")
            for j in 0..<vectors.count {
                XCTAssertEqual(matrix[i][j], refEuclidean(arrays[i], arrays[j]), accuracy: max(1e-2, abs(refEuclidean(arrays[i], arrays[j])) * 1e-3))
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter MetalComputeProviderTests/testDistanceMatrixMatchesReference`
Expected: FAIL — `has no member 'distanceMatrix'`.

- [ ] **Step 3: Write the implementation** (add to the actor)

```swift
    // MARK: - Distance matrix

    /// Full `queries × candidates` distance matrix; row `i` is `batchDistance(queries[i], candidates)`.
    public func distanceMatrix<V: VectorProtocol>(
        queries: [V], candidates: [V], metric: SupportedDistanceMetric
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty, !candidates.isEmpty else { return [] }
        var matrix: [[Float]] = []
        matrix.reserveCapacity(queries.count)
        for q in queries {
            matrix.append(try await batchDistance(query: q, candidates: candidates, metric: metric))
        }
        return matrix
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter MetalComputeProviderTests/testDistanceMatrixMatchesReference`
Expected: PASS.

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "MetalComputeProvider: distanceMatrix"
```

---

## Task 5: `distance` (single pair)

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- Modify: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
    func testSingleDistancePerMetric() async throws {
        let provider = try await MetalComputeProvider()
        let a = DynamicVector([1, 2, 3, 4, 5, 6, 7, 8])
        let b = DynamicVector([8, 7, 6, 5, 4, 3, 2, 1])
        let av = a.toArray(); let bv = b.toArray()

        XCTAssertEqual(try await provider.distance(a, b, metric: .euclidean), refEuclidean(av, bv), accuracy: 1e-4)
        XCTAssertEqual(try await provider.distance(a, b, metric: .cosine), refCosineDistance(av, bv), accuracy: 1e-4)
        XCTAssertEqual(try await provider.distance(a, b, metric: .manhattan), refManhattan(av, bv), accuracy: 1e-4)
        // dotProduct returns the raw dot (a similarity).
        let refDot = zip(av, bv).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(try await provider.distance(a, b, metric: .dotProduct), refDot, accuracy: 1e-3)
        // chebyshev = max abs diff.
        let refCheby = zip(av, bv).map { abs($0 - $1) }.max() ?? 0
        XCTAssertEqual(try await provider.distance(a, b, metric: .chebyshev), refCheby, accuracy: 1e-4)
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter MetalComputeProviderTests/testSingleDistancePerMetric`
Expected: FAIL — `has no member 'distance'`.

- [ ] **Step 3: Write the implementation** (add to the actor)

```swift
    // MARK: - Single distance

    /// Distance between two vectors. Computed on CPU (single-pair GPU dispatch never wins); semantics
    /// match `batchDistance`.
    public func distance<V: VectorProtocol>(
        _ a: V, _ b: V, metric: SupportedDistanceMetric
    ) async throws -> Float where V.Scalar == Float {
        let av = a.toArray(); let bv = b.toArray()
        guard av.count == bv.count else {
            throw VectorError.dimensionMismatch(expected: av.count, actual: bv.count)
        }
        switch metric {
        case .euclidean:  return try AccelerateFallback.euclideanDistance(av, bv)
        case .cosine:     return 1.0 - (try AccelerateFallback.cosineSimilarity(av, bv))
        case .dotProduct: return try AccelerateFallback.dotProduct(av, bv)
        case .manhattan:  return try AccelerateFallback.manhattanDistance(av, bv)
        case .chebyshev:  return Self.chebyshev(av, bv)
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter MetalComputeProviderTests/testSingleDistancePerMetric`
Expected: PASS.

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "MetalComputeProvider: single distance()"
```

---

## Task 6: Aggressive deprecation + delegation of the scattered surface

**Files:**
- Modify: `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift`
- Modify: `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

Context: the three `BatchOperations.*GPU` statics (lines ~403, ~468, ~604) get **rewired to delegate** to `MetalComputeProvider` **and** deprecated. `AcceleratedDistanceProvider` (line ~29) and the distance convenience extensions (`acceleratedDistance` ~265, `acceleratedDistanceOptimized` ~305) get **deprecation annotations** (implementations left intact — they already return correct results). `AcceleratedVectorOperations` and `acceleratedNormalize*` are **untouched** (no replacement).

- [ ] **Step 1: Write the failing test** (append to the test class)

The test method is itself marked `@available(*, deprecated)` so the deprecated-symbol calls inside don't emit warnings.

```swift
    @available(*, deprecated, message: "Exercises deprecated delegation on purpose")
    func testDeprecatedBatchOperationsStillDelegateCorrectly() async throws {
        let dim = 128
        let (q, cands) = makeVectors(count: 600, dim: dim, seed: 4242)
        let provider = try await MetalComputeProvider()

        // batchDistancesGPU delegates to provider.batchDistance.
        let legacyDistances = try await BatchOperations.batchDistancesGPU(from: q, to: cands, metric: .euclidean)
        let providerDistances = try await provider.batchDistance(query: q, candidates: cands, metric: .euclidean)
        assertClose(legacyDistances, providerDistances)

        // findNearestGPU delegates to provider.findNearest (compare index sets; distances within eps).
        let legacyKnn = try await BatchOperations.findNearestGPU(to: q, in: cands, k: 8, metric: .euclidean)
        let providerKnn = try await provider.findNearest(query: q, in: cands, k: 8, metric: .euclidean)
        XCTAssertEqual(legacyKnn.map { $0.index }, providerKnn.map { $0.index })
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter MetalComputeProviderTests/testDeprecatedBatchOperationsStillDelegateCorrectly`
Expected: PASS already at the API level (the legacy methods exist), but the point of this task is the rewire+deprecation. If it passes pre-change, that is fine — proceed to Step 3 and re-run to confirm no regression. (If you prefer a true red: temporarily assert `legacyDistances == providerDistances` exactly; it will fail on fp differences until both share one implementation, then relax to `assertClose`.)

- [ ] **Step 3: Rewire + deprecate** — edit `VectorCoreIntegration.swift`.

3a. Replace the body of `findNearestGPU` (the `public extension BatchOperations` method, currently building its own context/engine) with a delegating, deprecated version. Find:

```swift
    static func findNearestGPU<V: VectorProtocol & Sendable>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
```
Insert immediately **above** that line:
```swift
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    @available(*, deprecated, message: "Use MetalComputeProvider.findNearest(query:in:k:metric:). Removed in 0.6.0.")
```
and replace the method's body (everything between its `{` and matching `}`) with:
```swift
        guard ComputeDevice.gpu().isAvailable else { throw VectorError.metalNotAvailable() }
        guard k > 0, !vectors.isEmpty else { return [] }
        return try await MetalComputeProvider().findNearest(query: query, in: vectors, k: k, metric: metric)
```

3b. Same treatment for `batchDistancesGPU`. Above its declaration add the two `@available` lines (deprecation message: `"Use MetalComputeProvider.batchDistance(query:candidates:metric:). Removed in 0.6.0."`) and replace its body with:
```swift
        guard ComputeDevice.gpu().isAvailable else { throw VectorError.metalNotAvailable() }
        guard !candidates.isEmpty else { return [] }
        return try await MetalComputeProvider().batchDistance(query: query, candidates: candidates, metric: metric)
```

3c. Same for `pairwiseDistancesGPU`. Above its declaration add the two `@available` lines (message: `"Use MetalComputeProvider.distanceMatrix(queries:candidates:metric:). Removed in 0.6.0."`) and replace its body with:
```swift
        guard !vectors.isEmpty else { return [] }
        return try await MetalComputeProvider().distanceMatrix(queries: vectors, candidates: vectors, metric: metric)
```

3d. Deprecate the `AcceleratedDistanceProvider` actor (leave its body). Find `public actor AcceleratedDistanceProvider: DistanceProvider {` and insert immediately above it:
```swift
@available(*, deprecated, message: "Use MetalComputeProvider (distance/batchDistance). Removed in 0.6.0.")
```

3e. Deprecate the two distance convenience extension methods. Above `func acceleratedDistance(` insert:
```swift
    @available(*, deprecated, message: "Use MetalComputeProvider.distance(_:_:metric:). Removed in 0.6.0.")
```
Above `func acceleratedDistanceOptimized(` insert:
```swift
    @available(*, deprecated, message: "Use MetalComputeProvider.distance(_:_:metric:). Removed in 0.6.0.")
```

Note: `AcceleratedVectorFactory.createDefaultProviders()`/`createProviders()` return the now-deprecated `AcceleratedDistanceProvider`, so they will emit deprecation warnings. That is acceptable (no `-warnings-as-errors`). If you want them silent, also add `@available(*, deprecated, message: "Use MetalComputeProvider. Removed in 0.6.0.")` above those two factory methods (recommended for the "move fast" goal) — referencing a deprecated type from a deprecated method does not warn.

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter MetalComputeProviderTests/testDeprecatedBatchOperationsStillDelegateCorrectly`
Expected: PASS. Then a broad build to confirm deprecation warnings didn't become errors and nothing else broke:
Run: `swift build 2>&1 | tail -20`
Expected: build succeeds (deprecation warnings are fine).

- [ ] **Step 5: Commit** (on user go-ahead)

```bash
git add Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift Tests/VectorAccelerateTests/MetalComputeProviderTests.swift
git commit -m "Deprecate + delegate the scattered GPU surface to MetalComputeProvider"
```

---

## Task 7: File VectorCore R4 (transparent-dispatch hook)

**Files:**
- Modify: `docs/VECTORCORE_INTEGRATION_REQUESTS.md`

- [ ] **Step 1: Append the R4 section.** Add, immediately after the `### R3 …` block and before `## Expected payoff`:

```markdown
### R4 — A kernel-injection hook so `ComputeProvider` can supply GPU kernels (P2, unlocks transparent dispatch)

`ComputeProvider` is a *work-scheduler*: `execute` / `parallelExecute` / `parallelReduce` take opaque
`@Sendable` closures, and the closures VectorCore passes already call its own CPU kernels
(`BatchKernels.range_euclid_512`, `TopKSelectionKernels.range_topk_euclid2_512`). A provider can only
choose *how to schedule* that closure — it cannot substitute a Metal kernel. So a `MetalComputeProvider`
installed as `Operations.computeProvider` accelerates nothing, and `findNearestGPU(...)` is a private
stub that throws and is unreachable from the public API.

**Ask:** add a sub-protocol VectorCore's `Operations` downcasts to, so a provider can offer real batch
kernels. VectorAccelerate already conforms its provider to this shape behind the same routing logic:

```swift
public protocol BatchKernelProvider: ComputeProvider {
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
```

Then in `Operations.findNearest` / the batch-distance path:

```swift
if let gpu = Operations.computeProvider as? BatchKernelProvider {
    return try await gpu.findNearest(query: query, candidates: vectors, k: k, metric: metric)
}
// …existing CPU path…
```

**Payoff:** GPU acceleration becomes transparent through VectorCore's own `findNearest`/`batchDistance`
— no separate VectorAccelerate entry point needed — completing the §11.2 "ComputeProvider/ComputeDevice(.gpu)"
hook. VectorAccelerate ships the conformance now (see `MetalComputeProvider`); only the downcast is owed.
```

- [ ] **Step 2: Verify the doc renders / is internally consistent**

Run: `grep -n "R4\|BatchKernelProvider" docs/VECTORCORE_INTEGRATION_REQUESTS.md`
Expected: shows the new R4 heading and the protocol name.

- [ ] **Step 3: Commit** (on user go-ahead)

```bash
git add docs/VECTORCORE_INTEGRATION_REQUESTS.md
git commit -m "Integration requests: add R4 (ComputeProvider kernel-injection hook)"
```

---

## Task 8: CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entries under the unreleased / next section.** Open `CHANGELOG.md`, and under the top-most version's `### Added` add:
```markdown
- `MetalComputeProvider` — unified GPU compute façade (batch distance, k-NN/top-K, distance matrix,
  single distance) with `GPUDecisionEngine` routing, CPU fallback, and a VectorCore `ComputeProvider`
  capability shim. Filed VectorCore request R4 for transparent dispatch.
```
and add a `### Deprecated` subsection (create it if absent):
```markdown
### Deprecated
- `BatchOperations.findNearestGPU/batchDistancesGPU/pairwiseDistancesGPU`, `AcceleratedDistanceProvider`,
  and `VectorProtocol.acceleratedDistance(to:metric:)` / `IndexableVector.acceleratedDistanceOptimized(...)`
  — superseded by `MetalComputeProvider`. **Removed in 0.6.0.**
```

- [ ] **Step 2: Commit** (on user go-ahead)

```bash
git add CHANGELOG.md
git commit -m "CHANGELOG: MetalComputeProvider + deprecations"
```

---

## Task 9: Full build + suite green

- [ ] **Step 1: Build the whole package**

Run: `swift build 2>&1 | tail -30`
Expected: `Build complete!` (deprecation warnings allowed).

- [ ] **Step 2: Run the new test class in full**

Run: `swift test --filter MetalComputeProviderTests 2>&1 | tail -30`
Expected: all tests pass (or skip cleanly if no Metal device).

- [ ] **Step 3: Run the full suite to confirm no regressions** (the deprecation rewire touches shared files)

Run: `swift test 2>&1 | tail -40`
Expected: full suite passes (it was 1484 tests / 0 failures before this work; the new tests add to that). Investigate any newly-failing test before proceeding.

- [ ] **Step 4: Final commit if anything was adjusted** (on user go-ahead)

```bash
git add -A -- Sources/VectorAccelerate Tests/VectorAccelerateTests docs CHANGELOG.md
git commit -m "MetalComputeProvider: build + full suite green"
```

> Do **not** stage `Tests/VectorAccelerateTests/ClusteringKernelTests.swift` (a pre-existing,
> unrelated working-tree edit not authored by this work) or the untracked junk
> (`.antigravitycli/`, `GEMINI.md`, `consolidated_library.md`, `generate_consolidation.py`).
> Use explicit pathspecs as shown, never `git add -A` without a pathspec.

---

## Self-review (completed by plan author)

**Spec coverage:** §Architecture → Task 1; §API surface (batchDistance/findNearest/distanceMatrix/distance + ComputeProvider shim) → Tasks 1–5; §Data flow/decisioning/fallback → Task 2 (`routeToGPU`) reused by 3–5; §Metric coverage (euclidean/cosine GPU, others CPU) → Tasks 2/3/5; §ComputeProvider honest shim → Task 1 (doc comment + inherited defaults); §Consolidation/aggressive deprecation → Task 6; §R4 → Task 7; §Error handling (Configuration, fallback, guards) → Tasks 1–2; §Testing (conformance/parity/k-NN/matrix/routing/fallback/deprecation) → Tasks 1–6; CHANGELOG → Task 8; build/suite → Task 9. No gaps.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output.

**Type consistency:** `MetalComputeProvider`, `Configuration(preferGPU:fallbackToCPU:)`, `batchDistance(query:candidates:metric:)`, `findNearest(query:in:k:metric:)`, `distanceMatrix(queries:candidates:metric:)`, `distance(_:_:metric:)`, `routeToGPU(_:candidateCount:k:dimension:)`, `chebyshev(_:_:)`, `selectTopK(_:k:largerIsCloser:)` are used consistently across tasks. Engine/provider call shapes match the verified source: `Metal4ComputeEngine(context:decisionEngine:)`, `GPUDecisionEngine(context:)` (async), `shouldUseGPU(operation:vectorCount:candidateCount:k:queryCount:dimension:)`, `L2/CosineKernelDistanceProvider(context:)` + `batchDistance(from:to:metric:)`, `fusedDistanceTopK(query:database:k:metric:)`, `AccelerateFallback.batchEuclideanDistance/batchCosineSimilarity/batchDotProduct/manhattanDistance/euclideanDistance/cosineSimilarity/dotProduct`.
