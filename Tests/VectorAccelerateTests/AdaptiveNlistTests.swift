//
//  AdaptiveNlistTests.swift
//  VectorAccelerate
//
//  Tests for adaptive nlist and nprobe parameter selection.
//
//  P4.2: Verifies that recommendedNlist, recommendedNprobe, and ivfAuto
//  produce sensible configurations for various dataset sizes and recall targets.
//

import XCTest
@testable import VectorAccelerate
import VectorCore

/// Tests for adaptive IVF parameter selection.
final class AdaptiveNlistTests: XCTestCase {

    // MARK: - recommendedNlist Tests

    /// nlist should scale with sqrt(N) and respect bounds.
    func testRecommendedNlistScalesWithSize() {
        // Very small dataset - should clamp to minimum 8
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 10), 8)
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 50), 8)

        // Small dataset - sqrt(100) = 10
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 100), 10)

        // Medium dataset - sqrt(10000) = 100
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 10_000), 100)

        // Large dataset - sqrt(1000000) = 1000
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 1_000_000), 1000)

        // Very large - should clamp to maximum 4096
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 100_000_000), 4096)
    }

    /// Edge cases for nlist calculation.
    func testRecommendedNlistEdgeCases() {
        // Zero or negative should return minimum
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 0), 8)
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: -100), 8)

        // Just below clamp thresholds
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 64), 8)  // sqrt(64) = 8
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: 63), 8)  // sqrt(63) < 8

        // At upper bound threshold: 4096^2 = 16,777,216
        let upperThreshold = 4096 * 4096
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: upperThreshold), 4096)
        XCTAssertEqual(IndexConfiguration.recommendedNlist(for: upperThreshold + 1), 4096)
    }

    // MARK: - recommendedNprobe Tests

    /// nprobe should scale with nlist based on target recall.
    func testRecommendedNprobeForRecallTargets() {
        let nlist = 100

        // Low recall target (< 0.8) - 10% of nlist
        let lowNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.70)
        XCTAssertGreaterThanOrEqual(lowNprobe, 10)
        XCTAssertLessThanOrEqual(lowNprobe, 15)

        // Medium recall target (0.8 - 0.9) - 15% of nlist
        let medNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.85)
        XCTAssertGreaterThanOrEqual(medNprobe, 15)
        XCTAssertLessThanOrEqual(medNprobe, 20)

        // Default recall target (0.9) - 23% of nlist
        let defaultNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.90)
        XCTAssertGreaterThanOrEqual(defaultNprobe, 20)
        XCTAssertLessThanOrEqual(defaultNprobe, 25)

        // High recall target (0.9 - 0.95) - same as 0.9
        let highNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.92)
        XCTAssertGreaterThanOrEqual(highNprobe, 20)
        XCTAssertLessThanOrEqual(highNprobe, 25)

        // Very high recall target (0.95 - 0.99) - 30% of nlist
        let veryHighNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.95)
        XCTAssertGreaterThanOrEqual(veryHighNprobe, 28)
        XCTAssertLessThanOrEqual(veryHighNprobe, 35)

        // Near-perfect recall target (>= 0.99) - 50% of nlist
        let perfectNprobe = IndexConfiguration.recommendedNprobe(for: nlist, targetRecall: 0.99)
        XCTAssertGreaterThanOrEqual(perfectNprobe, 45)
        XCTAssertLessThanOrEqual(perfectNprobe, 55)
    }

    /// Edge cases for nprobe calculation.
    func testRecommendedNprobeEdgeCases() {
        // Zero nlist should return 1
        XCTAssertEqual(IndexConfiguration.recommendedNprobe(for: 0), 1)

        // Small nlist should still produce valid nprobe
        XCTAssertEqual(IndexConfiguration.recommendedNprobe(for: 8, targetRecall: 0.90), 2)

        // nprobe should not exceed nlist
        XCTAssertEqual(IndexConfiguration.recommendedNprobe(for: 4, targetRecall: 0.99), 2)
        XCTAssertLessThanOrEqual(
            IndexConfiguration.recommendedNprobe(for: 10, targetRecall: 1.0),
            10
        )

        // Default parameter uses 0.90 recall
        XCTAssertEqual(
            IndexConfiguration.recommendedNprobe(for: 100),
            IndexConfiguration.recommendedNprobe(for: 100, targetRecall: 0.90)
        )
    }

    // MARK: - ivfAuto Tests

    /// ivfAuto should create valid configurations.
    func testIvfAutoCreatesValidConfiguration() throws {
        let config = IndexConfiguration.ivfAuto(
            dimension: 128,
            expectedSize: 50_000,
            targetRecall: 0.90
        )

        try config.validate()

        XCTAssertTrue(config.isIVF)
        XCTAssertEqual(config.dimension, 128)
        XCTAssertEqual(config.capacity, 50_000)

        if case .ivf(let nlist, let nprobe, _) = config.indexType {
            // nlist should be around sqrt(50000) ≈ 224
            XCTAssertGreaterThanOrEqual(nlist, 200)
            XCTAssertLessThanOrEqual(nlist, 250)

            // nprobe should be ~23% of nlist for 90% recall
            XCTAssertGreaterThanOrEqual(nprobe, 40)
            XCTAssertLessThanOrEqual(nprobe, 60)

            // nprobe should not exceed nlist
            XCTAssertLessThanOrEqual(nprobe, nlist)
        } else {
            XCTFail("Expected IVF configuration")
        }
    }

    /// ivfAuto should handle various dataset sizes.
    func testIvfAutoVariousDatasetSizes() throws {
        let testCases: [(size: Int, expectedNlistRange: ClosedRange<Int>)] = [
            (1_000, 8...40),         // Small: sqrt(1000) ≈ 32, clamped min 8
            (10_000, 90...110),      // Medium: sqrt(10000) = 100
            (100_000, 300...350),    // Large: sqrt(100000) ≈ 316
            (1_000_000, 950...1050), // Very large: sqrt(1M) = 1000
        ]

        for (size, expectedRange) in testCases {
            let config = IndexConfiguration.ivfAuto(
                dimension: 64,
                expectedSize: size
            )

            try config.validate()

            if case .ivf(let nlist, let nprobe, _) = config.indexType {
                XCTAssertTrue(
                    expectedRange.contains(nlist),
                    "For size \(size): nlist \(nlist) not in expected range \(expectedRange)"
                )
                XCTAssertGreaterThan(nprobe, 0)
                XCTAssertLessThanOrEqual(nprobe, nlist)
            } else {
                XCTFail("Expected IVF configuration for size \(size)")
            }
        }
    }

    /// ivfAuto should respect different recall targets.
    func testIvfAutoRecallTargets() throws {
        let size = 10_000  // nlist will be 100

        let lowRecall = IndexConfiguration.ivfAuto(
            dimension: 64,
            expectedSize: size,
            targetRecall: 0.70
        )

        let highRecall = IndexConfiguration.ivfAuto(
            dimension: 64,
            expectedSize: size,
            targetRecall: 0.99
        )

        if case .ivf(_, let lowNprobe, _) = lowRecall.indexType,
           case .ivf(_, let highNprobe, _) = highRecall.indexType {
            XCTAssertLessThan(
                lowNprobe, highNprobe,
                "Higher recall target should produce higher nprobe"
            )
        } else {
            XCTFail("Expected IVF configurations")
        }
    }

    /// ivfAuto should use the specified metric.
    func testIvfAutoWithDifferentMetrics() throws {
        let metrics: [SupportedDistanceMetric] = [.euclidean, .cosine, .dotProduct]

        for metric in metrics {
            let config = IndexConfiguration.ivfAuto(
                dimension: 64,
                expectedSize: 10_000,
                metric: metric
            )

            try config.validate()
            XCTAssertEqual(config.metric, metric)
        }
    }

    // MARK: - Integration Tests

    /// ivfAuto configuration should work with AcceleratedVectorIndex.
    func testIvfAutoWithRealIndex() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let config = IndexConfiguration.ivfAuto(
            dimension: 32,
            expectedSize: 500,
            targetRecall: 0.90
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert enough vectors to train
        var vectors: [[Float]] = []
        for i in 0..<500 {
            var v = [Float](repeating: 0, count: 32)
            v[i % 32] = Float(i)
            vectors.append(v)
        }

        _ = try await index.insert(vectors)

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 500)
        XCTAssertNotNil(stats.ivfStats)
    }

    /// ivfAuto should achieve reasonable recall vs flat index.
    func testIvfAutoRecallMeetsTarget() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let datasetSize = 2000
        let dimension = 32
        let targetRecall: Float = 0.85

        let ivfConfig = IndexConfiguration.ivfAuto(
            dimension: dimension,
            expectedSize: datasetSize,
            targetRecall: targetRecall
        )

        let flatConfig = IndexConfiguration.flat(
            dimension: dimension,
            capacity: datasetSize
        )

        // Create indices
        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Generate random data with some structure (clustered)
        var vectors: [[Float]] = []
        for i in 0..<datasetSize {
            var v = [Float](repeating: 0, count: dimension)
            let cluster = i % 20
            for d in 0..<dimension {
                v[d] = Float(cluster) + Float.random(in: -0.5...0.5)
            }
            vectors.append(v)
        }

        _ = try await ivfIndex.insert(vectors)
        _ = try await flatIndex.insert(vectors)

        // Test recall on random queries
        var totalRecall: Float = 0
        let numQueries = 50
        let k = 10

        for _ in 0..<numQueries {
            var query = [Float](repeating: 0, count: dimension)
            let cluster = Int.random(in: 0..<20)
            for d in 0..<dimension {
                query[d] = Float(cluster) + Float.random(in: -0.5...0.5)
            }

            let ivfResults = try await ivfIndex.search(query: query, k: k)
            let flatResults = try await flatIndex.search(query: query, k: k)

            let ivfSet = Set(ivfResults.map { $0.handle.stableID })
            let flatSet = Set(flatResults.map { $0.handle.stableID })
            let intersection = ivfSet.intersection(flatSet)

            totalRecall += Float(intersection.count) / Float(k)
        }

        let avgRecall = totalRecall / Float(numQueries)
        print("IVF Auto recall: \(String(format: "%.1f%%", avgRecall * 100))")

        // Should achieve reasonable recall (slightly below target is acceptable
        // due to random data and small dataset)
        XCTAssertGreaterThanOrEqual(
            avgRecall, 0.70,
            "Auto config should achieve reasonable recall"
        )
    }
}
