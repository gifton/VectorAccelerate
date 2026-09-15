//
//  SoAScoringParityTests.swift
//  VectorAccelerateTests
//
//  End-to-end correctness for the zero-copy SoA scoring path: CPU-reference parity for L2 + cosine
//  at 512/768/1536, plus build-once / query-many reuse vs brute-force top-K.
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoAScoringParityTests: XCTestCase {
    var context: Metal4Context!
    var provider: MetalComputeProvider!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        context = try await Metal4Context()
        provider = try await MetalComputeProvider(context: context)
    }
    override func tearDown() async throws { context = nil; provider = nil }

    // Deterministic values in [-1, 1) (no Date/Math.random).
    private func lcg(_ n: Int, seed: UInt64) -> [Float] {
        var s = seed &+ 0x9E3779B97F4A7C15
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float((s >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1
        }
    }
    private func refL2(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { let d = a[i]-b[i]; s += d*d }; return s.squareRoot()
    }

    private func assertParity<V: SoACompatible & VectorProtocol>(_ type: V.Type, dim: Int) async throws
        where V.Scalar == Float {
        let query = try V(lcg(dim, seed: UInt64(dim)))
        let candArrays = (0..<64).map { lcg(dim, seed: UInt64(dim) &+ UInt64($0) &+ 1) }
        let candidates = try candArrays.map { try V($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let qa = query.toArray()

        let l2 = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
        for (i, c) in candArrays.enumerated() {
            let ref = refL2(qa, c)
            XCTAssertEqual(l2[i], ref, accuracy: max(1e-2, abs(ref) * 1e-3), "L2 dim \(dim) idx \(i)")
        }
        let cos = try await provider.batchDistance(query: query, against: set, metric: .cosine)
        for (i, c) in candArrays.enumerated() {
            let ref = SoAKernelGoldenTests.refCosineDistance(qa, c)
            XCTAssertEqual(cos[i], ref, accuracy: max(1e-3, abs(ref) * 1e-3), "cosine dim \(dim) idx \(i)")
        }
    }

    func testParity512() async throws { try await assertParity(Vector512Optimized.self,  dim: 512) }
    func testParity768() async throws { try await assertParity(Vector768Optimized.self,  dim: 768) }
    func testParity1536() async throws { try await assertParity(Vector1536Optimized.self, dim: 1536) }

    /// Counts that exercise the kernel's `j >= count` tail guard: 1 (single thread), 7 (sub-warp),
    /// 257 (two 256-thread groups, with a partial trailing group).
    func testParityTailCounts() async throws {
        let dim = 512
        for count in [1, 7, 257] {
            let query = try Vector512Optimized(lcg(dim, seed: UInt64(count)))
            let candArrays = (0..<count).map { lcg(dim, seed: UInt64(count) &+ UInt64($0) &+ 1) }
            let candidates = try candArrays.map { try Vector512Optimized($0) }
            let set = try SoACandidateSet(candidates: candidates, device: context.device)
            let qa = query.toArray()
            let l2 = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
            XCTAssertEqual(l2.count, count)
            for (i, c) in candArrays.enumerated() {
                let ref = refL2(qa, c)
                XCTAssertEqual(l2[i], ref, accuracy: max(1e-2, abs(ref) * 1e-3), "count \(count) idx \(i)")
            }
        }
    }

    /// Build the set once, query it twice — each query's k nearest DISTANCES match brute force.
    /// Compared by value (not index), so genuine/near ties between candidates don't flake the assertion.
    func testBuildOnceReuseAcrossQueries() async throws {
        let dim = 768
        let candArrays = (0..<200).map { lcg(dim, seed: 1000 &+ UInt64($0)) }
        let candidates = try candArrays.map { try Vector768Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)

        for qSeed in [UInt64(7), 99] {
            let qa = lcg(dim, seed: qSeed)
            let query = try Vector768Optimized(qa)
            let refDists = candArrays.map { refL2(qa, $0) }.sorted().prefix(5)
            let got = try await provider.findNearest(query: query, in: set, k: 5, metric: .euclidean)
            XCTAssertEqual(got.count, 5, "reuse seed \(qSeed)")
            for (g, r) in zip(got.map { $0.distance }, refDists) {
                XCTAssertEqual(g, r, accuracy: max(1e-2, abs(r) * 1e-3), "reuse seed \(qSeed)")
            }
        }
    }
}
