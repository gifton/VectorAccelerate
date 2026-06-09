//
//  MetalComputeProviderSoATests.swift
//  VectorAccelerateTests
//
//  Provider-level zero-copy SoA scoring (batchDistance / findNearest against a prebuilt set).
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MetalComputeProviderSoATests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testProviderScoresAgainstSoASet() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        let candidates = (0..<10).map { Vector512Optimized(repeating: Float($0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let query = Vector512Optimized(repeating: 0.0)

        let distances = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
        XCTAssertEqual(distances.count, 10)
        // c_k = [k…] vs q=[0…] ⇒ ‖·‖ = sqrt(512)*k ; nearest is k=0.
        XCTAssertEqual(distances[0], 0, accuracy: 1e-3)

        let nearest = try await provider.findNearest(query: query, in: set, k: 3, metric: .euclidean)
        XCTAssertEqual(nearest.count, 3)
        XCTAssertEqual(nearest.map { $0.index }, [0, 1, 2])
    }
}
