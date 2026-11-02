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

    /// The candidate buffer must be the SAME allocation across queries — proves build-once / query-many
    /// (this fails if the path ever re-stages/rebuilds per query).
    func testBufferReusedAcrossQueries() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        let candidates = (0..<32).map { Vector512Optimized(repeating: Float($0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        XCTAssertTrue(set.isZeroCopy)
        let bufferPtr = set.buffer.contents()      // the aliased SoA allocation

        for s in 0..<3 {
            let q = Vector512Optimized(repeating: Float(s))
            _ = try await provider.batchDistance(query: q, against: set, metric: .euclidean)
            XCTAssertEqual(set.buffer.contents(), bufferPtr, "candidate buffer must be reused, not rebuilt")
            XCTAssertTrue(set.isZeroCopy)
        }
    }

    func testUnsupportedMetricThrows() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        let set = try SoACandidateSet(candidates: (0..<4).map { Vector512Optimized(repeating: Float($0)) },
                                      device: context.device)
        let q = Vector512Optimized(repeating: 1.0)
        for metric in [SupportedDistanceMetric.dotProduct, .manhattan, .chebyshev] {
            do {
                _ = try await provider.batchDistance(query: q, against: set, metric: metric)
                XCTFail("expected a throw for unsupported metric \(metric)")
            } catch {
                XCTAssertTrue(error is VectorError, "\(metric) should throw VectorError, got \(error)")
            }
        }
    }

    func testKGreaterThanCountClamps() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        let candidates = (0..<4).map { Vector512Optimized(repeating: Float($0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let q = Vector512Optimized(repeating: 0.0)
        let nearest = try await provider.findNearest(query: q, in: set, k: 100, metric: .euclidean)
        XCTAssertEqual(nearest.count, 4)                          // clamped to count
        XCTAssertEqual(nearest.map { $0.index }, [0, 1, 2, 3])   // well-separated, deterministic
    }

    func testCosineZeroVectorIsDistanceOne() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        var arrays: [[Float]] = [[Float](repeating: 0, count: 512)]            // zero candidate
        arrays += (1..<4).map { k in [Float](repeating: Float(k), count: 512) }
        let candidates = try arrays.map { try Vector512Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let q = try Vector512Optimized((0..<512).map { Float(($0 % 5) + 1) })  // non-zero query
        let cos = try await provider.batchDistance(query: q, against: set, metric: .cosine)
        XCTAssertEqual(cos[0], 1.0, accuracy: 1e-5, "zero candidate ⇒ cosine distance 1.0")
    }
}
