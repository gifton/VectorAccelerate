//
//  SoAFusedTopKTests.swift
//  VectorAccelerateTests
//
//  Fused GPU top-K for the SoA `findNearest` path: the distance kernel and
//  `TopKSelectionKernel` are encoded into a single command buffer (separated by
//  `encoder.memoryBarrier(scope: .buffers)`) for k ≤ `TopKParameters.maxK` (128); above that,
//  the distance kernel runs alone and VectorCore's zero-copy pointer `TopKSelection.select`
//  selects top-k directly off the shared-storage buffer's contents (no `copyData` materialization).
//
//  Coverage: CPU-reference parity at dims {512, 768, 1536} × tail counts {1000, 1027},
//  k ∈ {1, 8, 128} (fused path) and k = 129 (pointer-select fallback path), both metrics,
//  nearest-first ordering, index validity, and build-once/query-many buffer reuse.
//
//  Distances are compared **by value** (sorted list), not by strict index-array equality: the
//  GPU distance kernel (SIMD4 lane-major partial sums) and the scalar CPU reference sum in a
//  different order, so distances that are extremely close can occupy either side of the k-th
//  boundary — the existing `SoAScoringParityTests.testBuildOnceReuseAcrossQueries` documents the
//  same tradeoff. Each returned (index, distance) pair is additionally checked for **self-
//  consistency** against the CPU reference recomputed for that specific index, which independently
//  validates the kernel didn't mismatch an index with the wrong distance.
//
//  Zero-copy vs. staged-fallback: `SoACandidateSet`'s designated init always builds its `SoA` with
//  `pageAligned: true`; per `VectorCore.SoA.build`/`allocateBuffer` (pinned 0.3.2), the page-aligned
//  allocation only degrades to non-page-aligned (and thus `isZeroCopy == false`) when `count == 0`
//  — and both `batchDistance`/`findNearest` early-return before ever touching `set.buffer` in that
//  case. So a staged-fallback GPU dispatch through this path is not constructible from the public
//  API today; the assertion below documents the zero-copy invariant that *is* reachable, matching
//  `SoACandidateSetTests`/`MetalComputeProviderSoATests`' existing `XCTAssertTrue(set.isZeroCopy)`.
//
//  See .superpowers/sdd/soa-060-release-hardening/task-3-brief.md
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoAFusedTopKTests: XCTestCase {
    var context: Metal4Context!
    var provider: MetalComputeProvider!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        context = try await Metal4Context()
        provider = try await MetalComputeProvider(context: context)
    }
    override func tearDown() async throws { context = nil; provider = nil }

    // MARK: - CPU reference (independent of VectorCore's TopKSelection and the GPU kernels)

    private func refL2(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { let d = a[i] - b[i]; s += d * d }; return s.squareRoot()
    }
    private func refDistance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric) -> Float {
        metric == .euclidean ? refL2(a, b) : SoAKernelGoldenTests.refCosineDistance(a, b)
    }
    private func tolerance(_ ref: Float, metric: SupportedDistanceMetric) -> Float {
        metric == .euclidean ? max(1e-2, abs(ref) * 1e-3) : max(1e-3, abs(ref) * 1e-3)
    }
    /// Brute-force top-k: full sort ascending by distance, ties broken by ascending index —
    /// ground truth independent of both `TopKSelectionKernel` (MSL register-heap) and
    /// `TopKSelection.select` (VectorCore, the k > 128 fallback).
    private func referenceTopK(_ distances: [Float], k: Int) -> [(index: Int, distance: Float)] {
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }
        let sorted = pairs.sorted { a, b in
            a.distance != b.distance ? a.distance < b.distance : a.index < b.index
        }
        return Array(sorted.prefix(min(k, distances.count)))
    }

    /// Validate one `findNearest` result against the CPU reference: count, per-position distance
    /// value (sorted-list parity, not index parity — see file header), nearest-first ordering,
    /// index bounds + uniqueness, and per-pair self-consistency against a fresh CPU recomputation.
    private func assertResult(
        _ got: [(index: Int, distance: Float)], candArrays: [[Float]], query: [Float],
        k: Int, metric: SupportedDistanceMetric, context ctx: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let refDistances = candArrays.map { refDistance(query, $0, metric: metric) }
        let want = referenceTopK(refDistances, k: k)

        XCTAssertEqual(got.count, want.count, "count \(ctx)", file: file, line: line)
        for (g, w) in zip(got, want) {
            XCTAssertEqual(g.distance, w.distance, accuracy: tolerance(w.distance, metric: metric),
                            "sorted-distance parity \(ctx)", file: file, line: line)
        }
        if got.count > 1 {
            for i in 1..<got.count {
                XCTAssertLessThanOrEqual(got[i - 1].distance, got[i].distance + 1e-3,
                                          "nearest-first \(ctx)", file: file, line: line)
            }
        }
        XCTAssertTrue(got.allSatisfy { $0.index >= 0 && $0.index < candArrays.count },
                       "index validity \(ctx)", file: file, line: line)
        XCTAssertEqual(Set(got.map { $0.index }).count, got.count, "index uniqueness \(ctx)", file: file, line: line)
        for r in got {
            let ref = refDistance(query, candArrays[r.index], metric: metric)
            XCTAssertEqual(r.distance, ref, accuracy: tolerance(ref, metric: metric),
                            "self-consistency idx=\(r.index) \(ctx)", file: file, line: line)
        }
    }

    // MARK: - Fused (k ≤ 128) parity across dims × tail counts × metrics

    private func assertFusedParity<V: SoACompatible & VectorProtocol>(
        _ type: V.Type, dim: Int, count: Int, seed: UInt64,
        file: StaticString = #filePath, line: UInt = #line
    ) async throws where V.Scalar == Float {
        var rng = TestRNG(seed: seed)
        let candArrays = (0..<count).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
        let candidates = try candArrays.map { try V($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        XCTAssertTrue(set.isZeroCopy, "candidate set expected zero-copy on this platform (see file header)",
                       file: file, line: line)

        let qArray = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let query = try V(qArray)

        for metric: SupportedDistanceMetric in [.euclidean, .cosine] {
            for k in [1, 8, 128] {
                let got = try await provider.findNearest(query: query, in: set, k: k, metric: metric)
                assertResult(got, candArrays: candArrays, query: qArray, k: k, metric: metric,
                             context: "dim=\(dim) count=\(count) k=\(k) metric=\(metric)",
                             file: file, line: line)
            }
        }
    }

    func testFusedParity512_count1000() async throws {
        try await assertFusedParity(Vector512Optimized.self, dim: 512, count: 1000, seed: 0x5A0_1000)
    }
    func testFusedParity512_count1027() async throws {
        try await assertFusedParity(Vector512Optimized.self, dim: 512, count: 1027, seed: 0x5A0_1027)
    }
    func testFusedParity768_count1000() async throws {
        try await assertFusedParity(Vector768Optimized.self, dim: 768, count: 1000, seed: 0x768_1000)
    }
    func testFusedParity768_count1027() async throws {
        try await assertFusedParity(Vector768Optimized.self, dim: 768, count: 1027, seed: 0x768_1027)
    }
    func testFusedParity1536_count1000() async throws {
        try await assertFusedParity(Vector1536Optimized.self, dim: 1536, count: 1000, seed: 0x1536_1000)
    }
    func testFusedParity1536_count1027() async throws {
        try await assertFusedParity(Vector1536Optimized.self, dim: 1536, count: 1027, seed: 0x1536_1027)
    }

    // MARK: - k = 129: CPU pointer-select fallback (TopKSelection.select over the raw buffer contents)

    func testK129UsesPointerSelectFallbackParity() async throws {
        let dim = 512, count = 300
        var rng = TestRNG(seed: 0xF411_BACC)
        let candArrays = (0..<count).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
        let candidates = try candArrays.map { try Vector512Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let qArray = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let query = try Vector512Optimized(qArray)

        XCTAssertGreaterThan(129, TopKParameters.maxK, "sanity: k=129 must exceed the fused kernel's cap")

        for metric: SupportedDistanceMetric in [.euclidean, .cosine] {
            let got = try await provider.findNearest(query: query, in: set, k: 129, metric: metric)
            assertResult(got, candArrays: candArrays, query: qArray, k: 129, metric: metric,
                         context: "k129 metric=\(metric)")
        }
    }

    // MARK: - Build-once / query-many: same buffer allocation reused across two distinct queries

    func testBuildOnceQueryManyFusedPath() async throws {
        let dim = 768, count = 200
        var rng = TestRNG(seed: 0xB0DE_0001)
        let candArrays = (0..<count).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
        let candidates = try candArrays.map { try Vector768Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let bufferPtr = set.buffer.contents()   // the aliased SoA allocation

        for qSeed: UInt64 in [0xA1, 0xB2] {
            var qrng = TestRNG(seed: qSeed)
            let qArray = (0..<dim).map { _ in qrng.nextFloat(in: -1...1) }
            let query = try Vector768Optimized(qArray)

            let got = try await provider.findNearest(query: query, in: set, k: 5, metric: .euclidean)
            assertResult(got, candArrays: candArrays, query: qArray, k: 5, metric: .euclidean,
                         context: "reuse seed=\(qSeed)")
            XCTAssertEqual(set.buffer.contents(), bufferPtr,
                            "candidate buffer must be reused, not rebuilt (seed \(qSeed))")
        }
    }
}
