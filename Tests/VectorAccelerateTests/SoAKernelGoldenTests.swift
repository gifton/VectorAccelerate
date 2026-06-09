//
//  SoAKernelGoldenTests.swift
//  VectorAccelerateTests
//
//  Lane-major SoA distance kernels validated against VectorCore's frozen golden fixture
//  and a CPU cosine reference. See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoAKernelGoldenTests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testL2SquaredGoldenFixture() async throws {
        let context = try await Metal4Context()
        let kernel = try await SoADistanceKernel(context: context)
        let query = Vector512Optimized(repeating: 1.0)
        let candidates = (0..<5).map { Vector512Optimized(repeating: Float(1 + $0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)

        let qToken = try await context.getBuffer(for: query.toArray())
        let outToken = try await context.getBuffer(size: set.layout.count * MemoryLayout<Float>.stride)
        try await context.executeAndWait { cb, enc in
            kernel.encode(into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                          distancesBuffer: outToken.buffer, count: set.layout.count,
                          lanes: set.layout.lanes, metric: .euclidean, computeSqrt: false)
            qToken.keepAlive(until: cb); outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }                // pin SoA through GPU completion
        }
        let got = outToken.copyData(as: Float.self, count: 5)
        let golden: [Float] = [0, 512, 2048, 4608, 8192]
        for (a, b) in zip(got, golden) { XCTAssertEqual(a, b, accuracy: 1e-3) }
    }

    func testCosineKernelMatchesCPU() async throws {
        let context = try await Metal4Context()
        let kernel = try await SoADistanceKernel(context: context)
        // Distinct, non-degenerate vectors so denom >> FLT_MIN.
        let q = try Vector512Optimized((0..<512).map { Float(($0 % 7) - 3) + 0.5 })
        let cs = try (0..<6).map { k in try Vector512Optimized((0..<512).map { Float(($0 % 5) - 2) + Float(k) * 0.25 }) }
        let set = try SoACandidateSet(candidates: cs, device: context.device)

        let qa = q.toArray()
        let qNormSq = qa.reduce(Float(0)) { $0 + $1 * $1 }
        let qToken = try await context.getBuffer(for: qa)
        let outToken = try await context.getBuffer(size: cs.count * MemoryLayout<Float>.stride)
        try await context.executeAndWait { cb, enc in
            kernel.encode(into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                          distancesBuffer: outToken.buffer, count: set.layout.count,
                          lanes: set.layout.lanes, metric: .cosine, queryNormSq: qNormSq)
            qToken.keepAlive(until: cb); outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }
        }
        let got = outToken.copyData(as: Float.self, count: cs.count)
        for (i, c) in cs.enumerated() {
            let ref = SoAKernelGoldenTests.refCosineDistance(qa, c.toArray())
            XCTAssertEqual(got[i], ref, accuracy: max(1e-3, abs(ref) * 1e-3))
        }
    }

    // CPU reference matching the kernel exactly: sqrt(a)*sqrt(b), FLT_MIN floor, 1 − clamp(sim).
    static func refCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        let denom = na.squareRoot() * nb.squareRoot()
        guard denom >= .leastNormalMagnitude else { return 1.0 }
        return 1.0 - min(max(dot / denom, -1), 1)
    }
}
