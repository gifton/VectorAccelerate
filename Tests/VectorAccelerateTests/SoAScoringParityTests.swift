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

    /// Build the set once, query it twice — proves the bridged buffer is reused across queries and
    /// each query matches a brute-force CPU top-K.
    func testBuildOnceReuseAcrossQueries() async throws {
        let dim = 768
        let candArrays = (0..<200).map { lcg(dim, seed: 1000 &+ UInt64($0)) }
        let candidates = try candArrays.map { try Vector768Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)

        for qSeed in [UInt64(7), 99] {
            let qa = lcg(dim, seed: qSeed)
            let query = try Vector768Optimized(qa)
            let refTop = candArrays.enumerated()
                .map { (index: $0.offset, d: refL2(qa, $0.element)) }
                .sorted { $0.d < $1.d }.prefix(5).map { $0.index }
            let got = try await provider.findNearest(query: query, in: set, k: 5, metric: .euclidean)
            XCTAssertEqual(got.map { $0.index }, refTop, "reuse query seed \(qSeed)")
        }
    }
}
