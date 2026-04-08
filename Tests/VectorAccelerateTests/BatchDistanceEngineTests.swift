//
//  BatchDistanceEngineTests.swift
//  VectorAccelerateTests
//
//  Tests for BatchDistanceEngine: batch distance computations with
//  GPU/SIMD/CPU routing, cosine similarity, dot product, Manhattan
//  distance, and k-nearest neighbor search.
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

final class BatchDistanceEngineTests: XCTestCase {

    var context: Metal4Context!
    var engine: BatchDistanceEngine!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        engine = try await BatchDistanceEngine(context: context)
    }

    override func tearDown() async throws {
        engine = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - CPU Reference Helpers

    /// CPU Euclidean distance reference: ||a - b||_2
    private func cpuEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(zip(a, b).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) })
    }

    /// CPU cosine similarity reference: (a . b) / (||a|| * ||b||)
    private func cpuCosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0) { $0 + $1 * $1 })
        if normA > 0 && normB > 0 {
            return dot / (normA * normB)
        }
        return 0
    }

    /// CPU dot product reference: a . b
    private func cpuDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// CPU Manhattan distance reference: sum |a_i - b_i|
    private func cpuManhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + abs($1.0 - $1.1) }
    }

    // MARK: - Batch Euclidean Distance Tests

    /// Empty candidates should return empty results
    func test_batchEuclideanDistance_emptyCandidates() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchEuclideanDistance(
            query: query, candidates: candidates
        )

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    /// Dimension mismatch between query and candidates should throw
    func test_batchEuclideanDistance_dimensionMismatch() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = [
            [1.0, 2.0],  // dim=2 vs query dim=3
        ]

        do {
            _ = try await engine.batchEuclideanDistance(
                query: query, candidates: candidates
            )
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected: dimension mismatch was thrown
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    /// Small batch (below simdThreshold=100) uses CPU path; verify against reference
    func test_batchEuclideanDistance_cpuPath_knownValues() async throws {
        let query: [Float] = [0.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [3.0, 4.0, 0.0],  // distance = 5.0
            [1.0, 0.0, 0.0],  // distance = 1.0
            [0.0, 0.0, 5.0],  // distance = 5.0
            [1.0, 1.0, 1.0],  // distance = sqrt(3)
            [2.0, 0.0, 0.0],  // distance = 2.0
        ]

        let results = try await engine.batchEuclideanDistance(
            query: query, candidates: candidates
        )

        XCTAssertEqual(results.count, 5)
        for (i, candidate) in candidates.enumerated() {
            let expected = cpuEuclideanDistance(query, candidate)
            XCTAssertEqual(results[i], expected, accuracy: 1e-4,
                           "Euclidean distance at index \(i) should match CPU reference")
        }
    }

    /// Explicitly forcing CPU path (useGPU: false) should produce correct results
    func test_batchEuclideanDistance_forceCPU() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],  // distance = 0.0
            [0.0, 1.0, 0.0],  // distance = sqrt(2)
            [2.0, 0.0, 0.0],  // distance = 1.0
        ]

        let results = try await engine.batchEuclideanDistance(
            query: query, candidates: candidates, useGPU: false
        )

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(results[1], sqrt(2), accuracy: 1e-4)
        XCTAssertEqual(results[2], 1.0, accuracy: 1e-4)
    }

    /// Explicitly forcing GPU path (useGPU: true) on a small batch should still work
    func test_batchEuclideanDistance_forceGPU() async throws {
        let query: [Float] = [0.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [3.0, 4.0, 0.0],  // distance = 5.0
            [1.0, 0.0, 0.0],  // distance = 1.0
        ]

        let results = try await engine.batchEuclideanDistance(
            query: query, candidates: candidates, useGPU: true
        )

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0], 5.0, accuracy: 1e-2,
                       "GPU path Euclidean distance should be approximately correct")
        XCTAssertEqual(results[1], 1.0, accuracy: 1e-2,
                       "GPU path Euclidean distance should be approximately correct")
    }

    // MARK: - Batch Cosine Similarity Tests

    /// Cosine similarity with known values: parallel ~ 1.0, orthogonal ~ 0.0
    func test_batchCosineSimilarity_knownValues() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [2.0, 0.0, 0.0],  // parallel -> similarity ~ 1.0
            [0.0, 1.0, 0.0],  // orthogonal -> similarity ~ 0.0
            [-1.0, 0.0, 0.0], // anti-parallel -> similarity ~ -1.0
        ]

        let results = try await engine.batchCosineSimilarity(
            query: query, candidates: candidates
        )

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 1.0, accuracy: 1e-4,
                       "Parallel vectors should have cosine similarity near 1.0")
        XCTAssertEqual(results[1], 0.0, accuracy: 1e-4,
                       "Orthogonal vectors should have cosine similarity near 0.0")
        XCTAssertEqual(results[2], -1.0, accuracy: 1e-4,
                       "Anti-parallel vectors should have cosine similarity near -1.0")
    }

    /// Empty candidates for cosine similarity should return []
    func test_batchCosineSimilarity_emptyCandidates() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchCosineSimilarity(
            query: query, candidates: candidates
        )

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    /// Cosine similarity with dimension mismatch should throw
    func test_batchCosineSimilarity_dimensionMismatch() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = [
            [1.0, 2.0],  // dim=2 vs query dim=3
        ]

        do {
            _ = try await engine.batchCosineSimilarity(
                query: query, candidates: candidates
            )
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected: dimension mismatch was thrown
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Batch Dot Product Tests

    /// Dot product with known values, verified against manual computation
    func test_batchDotProduct_knownValues() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = [
            [4.0, 5.0, 6.0],  // dot = 1*4 + 2*5 + 3*6 = 32
            [1.0, 1.0, 1.0],  // dot = 1 + 2 + 3 = 6
            [0.0, 0.0, 0.0],  // dot = 0
        ]

        let results = try await engine.batchDotProduct(
            query: query, candidates: candidates
        )

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 32.0, accuracy: 1e-4)
        XCTAssertEqual(results[1], 6.0, accuracy: 1e-4)
        XCTAssertEqual(results[2], 0.0, accuracy: 1e-4)
    }

    /// Empty candidates for dot product should return []
    func test_batchDotProduct_emptyCandidates() async throws {
        let query: [Float] = [1.0, 2.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchDotProduct(
            query: query, candidates: candidates
        )

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    // MARK: - Batch Manhattan Distance Tests

    /// Manhattan distance with known values, verified against manual L1 computation
    func test_batchManhattanDistance_knownValues() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = [
            [4.0, 1.0, 5.0],  // L1 = |3| + |1| + |2| = 6
            [1.0, 2.0, 3.0],  // L1 = 0 (identical)
            [0.0, 0.0, 0.0],  // L1 = 1 + 2 + 3 = 6
        ]

        let results = try await engine.batchManhattanDistance(
            query: query, candidates: candidates
        )

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 6.0, accuracy: 1e-2,
                       "Manhattan distance of [1,2,3] and [4,1,5] should be 6.0")
        XCTAssertEqual(results[1], 0.0, accuracy: 1e-2,
                       "Manhattan distance of identical vectors should be 0.0")
        XCTAssertEqual(results[2], 6.0, accuracy: 1e-2,
                       "Manhattan distance of [1,2,3] and [0,0,0] should be 6.0")
    }

    /// Empty candidates for Manhattan distance should return []
    func test_batchManhattanDistance_emptyCandidates() async throws {
        let query: [Float] = [1.0, 2.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchManhattanDistance(
            query: query, candidates: candidates
        )

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    // MARK: - K-Nearest Neighbors Tests

    /// KNN with Euclidean metric: verify correct k nearest are returned
    func test_kNearestNeighbors_euclidean() async throws {
        let query: [Float] = [0.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [10.0, 0.0, 0.0],  // distance = 10.0
            [1.0, 0.0, 0.0],   // distance = 1.0  (1st nearest)
            [5.0, 0.0, 0.0],   // distance = 5.0
            [2.0, 0.0, 0.0],   // distance = 2.0  (2nd nearest)
            [3.0, 0.0, 0.0],   // distance = 3.0  (3rd nearest)
        ]

        let results = try await engine.kNearestNeighbors(
            query: query, candidates: candidates, k: 3, metric: .euclidean
        )

        XCTAssertEqual(results.count, 3, "Should return exactly k=3 results")

        // Verify sorted ascending by distance
        XCTAssertEqual(results[0].index, 1)
        XCTAssertEqual(results[0].distance, 1.0, accuracy: 1e-4)
        XCTAssertEqual(results[1].index, 3)
        XCTAssertEqual(results[1].distance, 2.0, accuracy: 1e-4)
        XCTAssertEqual(results[2].index, 4)
        XCTAssertEqual(results[2].distance, 3.0, accuracy: 1e-4)
    }

    /// KNN with cosine metric: verify cosine distance ordering
    /// Note: kNearestNeighbors converts similarity to distance via (1 - similarity)
    func test_kNearestNeighbors_cosine() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [0.0, 1.0, 0.0],   // similarity ~ 0.0, distance ~ 1.0
            [1.0, 0.0, 0.0],   // similarity ~ 1.0, distance ~ 0.0 (nearest)
            [-1.0, 0.0, 0.0],  // similarity ~ -1.0, distance ~ 2.0
            [1.0, 1.0, 0.0],   // similarity ~ 0.707, distance ~ 0.293 (2nd nearest)
        ]

        let results = try await engine.kNearestNeighbors(
            query: query, candidates: candidates, k: 2, metric: .cosine
        )

        XCTAssertEqual(results.count, 2, "Should return exactly k=2 results")

        // Nearest should be the parallel vector (index 1)
        XCTAssertEqual(results[0].index, 1,
                       "Nearest cosine neighbor should be the parallel vector")
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 1e-4)

        // Second nearest should be [1,1,0] (index 3)
        XCTAssertEqual(results[1].index, 3,
                       "Second nearest should be [1,1,0]")
        let expectedDistance: Float = 1.0 - cpuCosineSimilarity(query, [1.0, 1.0, 0.0])
        XCTAssertEqual(results[1].distance, expectedDistance, accuracy: 1e-2)
    }

    /// KNN when k is larger than candidate count should return all candidates
    func test_kNearestNeighbors_kLargerThanCandidates() async throws {
        let query: [Float] = [0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]

        let results = try await engine.kNearestNeighbors(
            query: query, candidates: candidates, k: 20, metric: .euclidean
        )

        XCTAssertEqual(results.count, 5,
                       "When k > candidate count, should return all candidates")

        // Verify sorted ascending by distance
        for i in 0..<(results.count - 1) {
            XCTAssertLessThanOrEqual(results[i].distance, results[i + 1].distance,
                                     "Results should be sorted by distance ascending")
        }
    }
}
