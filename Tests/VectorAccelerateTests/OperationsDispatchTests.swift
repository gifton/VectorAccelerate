//
//  OperationsDispatchTests.swift
//  VectorAccelerateTests
//
//  End-to-end proof that VectorCore's `Operations.findNearest` / `findNearestBatch`
//  actually downcast the task-local `Operations.computeProvider` to `any BatchKernelProvider`
//  and dispatch to it (see `.build/checkouts/VectorCore/Sources/VectorCore/Operations/Operations.swift`,
//  the `computeProvider as? any BatchKernelProvider` checks). Existing tests
//  (MetalComputeProviderTests) bind `any BatchKernelProvider` directly and call it — that proves
//  conformance, not VectorCore's routing. This file installs a counting spy via the task-local
//  scoped binding and drives everything through `Operations`' own entry points.
//

import XCTest
@testable import VectorAccelerate
import VectorCore
@preconcurrency import Metal

// MARK: - Spy BatchKernelProvider

/// Wraps a real `MetalComputeProvider` and delegates every `ComputeProvider` /
/// `BatchKernelProvider` requirement to it, while counting invocations of the three
/// `BatchKernelProvider` dispatch entry points. An `actor` is the natural isolation choice here:
/// `MetalComputeProvider` itself is an actor, so wrapping it in another actor keeps the delegation
/// forwarding trivially `Sendable` (actors are implicitly `Sendable`) without any manual locking
/// around the invocation counters — actor-isolated `var` state is exactly what we want for
/// mutable counters read back from the test via `await`.
///
/// The `ComputeProvider` scheduling shim members (`device`, `maxConcurrency`, `deviceInfo`) are
/// `nonisolated` on `MetalComputeProvider` (backed by `nonisolated let` storage), so forwarding
/// them `nonisolated` here requires no actor hop — this mirrors `MetalComputeProvider`'s own shim
/// and is required because `ComputeProvider` declares them as synchronous (non-async) requirements
/// that must be satisfiable without `await` from an arbitrary isolation context.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
actor SpyBatchKernelProvider: BatchKernelProvider {
    // `nonisolated let`, matching `MetalComputeProvider`'s own `_device`/`_maxConcurrency`/
    // `_deviceInfo` storage: a plain (isolated) `let` on an actor still requires actor isolation
    // to read synchronously, so the `nonisolated` `device`/`maxConcurrency`/`deviceInfo` forwards
    // below need `inner` itself to be `nonisolated` to read it without an actor hop.
    private nonisolated let inner: MetalComputeProvider

    private(set) var batchDistanceCallCount = 0
    private(set) var findNearestCallCount = 0
    private(set) var findNearestBatchCallCount = 0

    init(wrapping inner: MetalComputeProvider) {
        self.inner = inner
    }

    // MARK: ComputeProvider (pure delegation, not counted — these are the scheduling shim, not
    // the kernel dispatch surface this suite is verifying)

    nonisolated var device: ComputeDevice { inner.device }
    nonisolated var maxConcurrency: Int { inner.maxConcurrency }
    nonisolated var deviceInfo: ComputeDeviceInfo { inner.deviceInfo }

    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await inner.execute(work)
    }

    // MARK: BatchKernelProvider (delegated + counted — the VectorCore dispatch hook, R4)

    func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        batchDistanceCallCount += 1
        return try await inner.batchDistance(query: query, candidates: candidates, metric: metric)
    }

    func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        findNearestCallCount += 1
        return try await inner.findNearest(query: query, candidates: candidates, k: k, metric: metric)
    }

    func findNearestBatch<V: VectorProtocol>(
        queries: [V], candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        findNearestBatchCallCount += 1
        return try await inner.findNearestBatch(queries: queries, candidates: candidates, k: k, metric: metric)
    }
}

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class OperationsDispatchTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    // MARK: - CPU reference helpers (independent ground truth; mirrors
    // MetalComputeProviderTests.swift's pattern — ascending order, k smallest, per-pair
    // distance consistency, ~1e-3 relative tolerance)

    /// Deterministic dataset via `TestRNG` (through `TestDataGenerator`) — no `Date`, no
    /// unseeded `Random`.
    private func makeVectors(count: Int, dim: Int, seed: UInt64) -> (query: DynamicVector, candidates: [DynamicVector]) {
        var gen = TestDataGenerator(seed: seed)
        let rows = gen.uniformVectors(count: count + 1, dimension: dim)
        let query = DynamicVector(rows[0])
        let candidates = rows.dropFirst().map { DynamicVector($0) }
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

    /// Asserts each result's distance is self-consistent against the CPU reference for its index,
    /// and that the sequence is ascending — i.e. a real top-k, not just "some k values."
    private func assertIsValidTopK(
        _ results: [NearestNeighborResult], query: [Float], candidates: [[Float]],
        k: Int, reference: (([Float], [Float]) -> Float),
        tol: Float = 1e-3, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(results.count, k, "expected \(k) results", file: file, line: line)
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(
                results[i - 1].distance, results[i].distance + 1e-3,
                "results must be ascending by distance", file: file, line: line
            )
        }
        for r in results {
            let expected = reference(query, candidates[r.index])
            let scale = max(1, abs(expected))
            XCTAssertEqual(
                r.distance, expected, accuracy: tol * scale,
                "index \(r.index) distance must match the CPU reference for that pair",
                file: file, line: line
            )
        }
        // The k returned distances must be exactly the k smallest reference distances (order-robust
        // to near-tie index swaps, unlike comparing index lists directly).
        let refSmallestK = candidates.map { reference(query, $0) }.sorted().prefix(k)
        for (got, ref) in zip(results.map({ $0.distance }), refSmallestK) {
            let scale = max(1, abs(ref))
            XCTAssertEqual(got, ref, accuracy: tol * scale, "k-th smallest distance mismatch", file: file, line: line)
        }
    }

    // MARK: - Requirement A.2: dispatch tests (the core)

    func testFindNearestRoutesThroughSpy_euclidean() async throws {
        let real = try await MetalComputeProvider()
        let spy = SpyBatchKernelProvider(wrapping: real)
        let dim = 256
        let (q, cands) = makeVectors(count: 500, dim: dim, seed: 1001)
        let k = 10

        let results = try await Operations.$computeProvider.withValue(spy) {
            try await Operations.findNearest(to: q, in: cands, k: k, metric: EuclideanDistance())
        }

        let count = await spy.findNearestCallCount
        XCTAssertEqual(count, 1, "Operations.findNearest must route through the installed BatchKernelProvider exactly once")

        assertIsValidTopK(
            results, query: q.toArray(), candidates: cands.map { $0.toArray() },
            k: k, reference: refEuclidean
        )
    }

    func testFindNearestRoutesThroughSpy_cosine() async throws {
        let real = try await MetalComputeProvider()
        let spy = SpyBatchKernelProvider(wrapping: real)
        let dim = 256
        let (q, cands) = makeVectors(count: 500, dim: dim, seed: 2002)
        let k = 10

        let results = try await Operations.$computeProvider.withValue(spy) {
            try await Operations.findNearest(to: q, in: cands, k: k, metric: CosineDistance())
        }

        let count = await spy.findNearestCallCount
        XCTAssertEqual(count, 1, "Operations.findNearest must route through the installed BatchKernelProvider exactly once")

        assertIsValidTopK(
            results, query: q.toArray(), candidates: cands.map { $0.toArray() },
            k: k, reference: refCosineDistance
        )
    }

    func testFindNearestBatchRoutesThroughSpy() async throws {
        let real = try await MetalComputeProvider()
        let spy = SpyBatchKernelProvider(wrapping: real)
        let dim = 128
        let numQueries = 5
        let k = 6

        // Build one shared candidate pool and `numQueries` independent query vectors.
        var gen = TestDataGenerator(seed: 3003)
        let candidateRows = gen.uniformVectors(count: 400, dimension: dim)
        let queryRows = gen.uniformVectors(count: numQueries, dimension: dim)
        let cands = candidateRows.map { DynamicVector($0) }
        let queries = queryRows.map { DynamicVector($0) }

        let batched = try await Operations.$computeProvider.withValue(spy) {
            try await Operations.findNearestBatch(queries: queries, in: cands, k: k, metric: EuclideanDistance())
        }

        let count = await spy.findNearestBatchCallCount
        XCTAssertEqual(count, 1, "Operations.findNearestBatch must route through the installed BatchKernelProvider exactly once")
        XCTAssertEqual(batched.count, numQueries)

        let candArrays = cands.map { $0.toArray() }
        for (i, results) in batched.enumerated() {
            assertIsValidTopK(results, query: queryRows[i], candidates: candArrays, k: k, reference: refEuclidean)
        }
    }

    /// Control: without the spy installed (default task-local `computeProvider`, i.e.
    /// `CPUComputeProvider.automatic`, which does NOT conform to `BatchKernelProvider`), calling
    /// `Operations.findNearest` must leave the spy's counters at zero. Trivial, but documents the
    /// `@TaskLocal` scoping semantics the whole suite depends on: the spy only observes calls made
    /// *inside* `Operations.$computeProvider.withValue(spy) { ... }`.
    func testControlWithoutSpyInstalledLeavesCountersAtZero() async throws {
        let real = try await MetalComputeProvider()
        let spy = SpyBatchKernelProvider(wrapping: real)
        let (q, cands) = makeVectors(count: 50, dim: 32, seed: 4004)

        // No `$computeProvider.withValue` scope — spy is constructed but never installed.
        _ = try await Operations.findNearest(to: q, in: cands, k: 5, metric: EuclideanDistance())

        let batchCount = await spy.batchDistanceCallCount
        let findCount = await spy.findNearestCallCount
        let findBatchCount = await spy.findNearestBatchCallCount
        XCTAssertEqual(batchCount, 0)
        XCTAssertEqual(findCount, 0)
        XCTAssertEqual(findBatchCount, 0)
    }

    // MARK: - Requirement A.3: unsupported-metric (manhattan) end-to-end

    /// Manhattan has no GPU kernel in `MetalComputeProvider` — `Self.gpuMetric(for:)` returns `nil`
    /// for it, so the provider's `findNearest`/`batchDistance` (the `any DistanceMetric` overloads,
    /// i.e. the `BatchKernelProvider` conformance) fall through to
    /// `floatMetric.batchDistance(query:candidates:)`, VectorCore's own `ManhattanDistance`
    /// implementation — never a VA-internal manhattan path. That CPU-authoritative computation is
    /// what end-to-end results are checked against.
    ///
    /// Routing finding (see task-4-report.md for the full writeup): VectorCore's
    /// `Operations.findNearest` decides whether to dispatch to the provider *before* it knows
    /// anything about the metric — `if let gpu = computeProvider as? any BatchKernelProvider` is
    /// the very first branch, unconditional on `M`. So the spy's `findNearest` counter increments
    /// for manhattan exactly the same as for euclidean/cosine: VectorCore always hands the call to
    /// an installed `BatchKernelProvider`, and it is *that provider's own* responsibility (not
    /// VectorCore's) to decide, per metric, whether to GPU-accelerate or defer to the metric's
    /// authoritative CPU `batchDistance`. "Unsupported metric" is an internal `MetalComputeProvider`
    /// concept, not a VectorCore routing concept.
    func testFindNearestRoutesThroughSpy_manhattan_unsupportedMetric() async throws {
        let real = try await MetalComputeProvider()
        let spy = SpyBatchKernelProvider(wrapping: real)
        let dim = 200
        let (q, cands) = makeVectors(count: 300, dim: dim, seed: 5005)
        let k = 8

        let results = try await Operations.$computeProvider.withValue(spy) {
            try await Operations.findNearest(to: q, in: cands, k: k, metric: ManhattanDistance())
        }

        // Documented above: routing to the provider is unconditional on metric support, so this
        // still increments exactly once, identically to the euclidean/cosine tests.
        let count = await spy.findNearestCallCount
        XCTAssertEqual(count, 1, "Operations.findNearest routes to the provider regardless of metric support")

        assertIsValidTopK(
            results, query: q.toArray(), candidates: cands.map { $0.toArray() },
            k: k, reference: refManhattan
        )
    }
}
