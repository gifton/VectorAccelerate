//
//  FallbackProviderTests.swift
//  VectorAccelerateTests
//
//  Tests for CPU fallback provider
//

import XCTest
@testable import VectorAccelerate
import VectorCore

final class FallbackProviderTests: XCTestCase {

    var fallback: FallbackProvider!

    override func setUp() {
        super.setUp()
        fallback = FallbackProvider()
    }

    override func tearDown() {
        fallback = nil
        super.tearDown()
    }

    // MARK: - L2 Distance Tests

    func testL2DistanceSimple() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.l2Distance(from: a, to: b)

        // sqrt(1 + 1) = sqrt(2) ≈ 1.414
        XCTAssertEqual(result, sqrt(2), accuracy: 0.001)
    }

    func testL2DistanceIdentical() {
        let a: [Float] = [1, 2, 3, 4]

        let result = fallback.l2Distance(from: a, to: a)

        XCTAssertEqual(result, 0, accuracy: 0.001)
    }

    func testL2DistanceOrthogonal() {
        let a: [Float] = [3, 0]
        let b: [Float] = [0, 4]

        let result = fallback.l2Distance(from: a, to: b)

        // sqrt(9 + 16) = 5
        XCTAssertEqual(result, 5, accuracy: 0.001)
    }

    func testL2DistanceMismatchedDimensions() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2]

        let result = fallback.l2Distance(from: a, to: b)

        XCTAssertEqual(result, .infinity)
    }

    func testL2DistanceSquared() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.l2DistanceSquared(from: a, to: b)

        XCTAssertEqual(result, 2.0, accuracy: 0.001)
    }

    func testBatchL2Distance() {
        let query: [Float] = [1, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],  // Distance 0
            [0, 1, 0],  // Distance sqrt(2)
            [2, 0, 0],  // Distance 1
        ]

        let results = fallback.batchL2Distance(from: query, to: candidates)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 0, accuracy: 0.001)
        XCTAssertEqual(results[1], sqrt(2), accuracy: 0.001)
        XCTAssertEqual(results[2], 1, accuracy: 0.001)
    }

    // MARK: - Cosine Similarity Tests

    func testCosineSimilarityIdentical() {
        let a: [Float] = [1, 2, 3]

        let result = fallback.cosineSimilarity(from: a, to: a)

        XCTAssertEqual(result, 1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.cosineSimilarity(from: a, to: b)

        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }

    func testCosineSimilarityOpposite() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [-1, 0, 0]

        let result = fallback.cosineSimilarity(from: a, to: b)

        XCTAssertEqual(result, -1.0, accuracy: 0.001)
    }

    func testCosineSimilarityScaleInvariant() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [2, 4, 6]  // Same direction, different magnitude

        let result = fallback.cosineSimilarity(from: a, to: b)

        XCTAssertEqual(result, 1.0, accuracy: 0.001)
    }

    func testCosineDistance() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.cosineDistance(from: a, to: b)

        // 1 - 0 = 1
        XCTAssertEqual(result, 1.0, accuracy: 0.001)
    }

    func testBatchCosineSimilarity() {
        let query: [Float] = [1, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],   // Similarity 1
            [0, 1, 0],   // Similarity 0
            [-1, 0, 0],  // Similarity -1
        ]

        let results = fallback.batchCosineSimilarity(from: query, to: candidates)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(results[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(results[2], -1.0, accuracy: 0.001)
    }

    // MARK: - Dot Product Tests

    func testDotProductSimple() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]

        let result = fallback.dotProduct(from: a, to: b)

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        XCTAssertEqual(result, 32.0, accuracy: 0.001)
    }

    func testDotProductOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.dotProduct(from: a, to: b)

        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }

    func testBatchDotProduct() {
        let query: [Float] = [1, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],  // Dot product 1
            [2, 0, 0],  // Dot product 2
            [0, 1, 0],  // Dot product 0
        ]

        let results = fallback.batchDotProduct(from: query, to: candidates)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(results[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(results[2], 0.0, accuracy: 0.001)
    }

    // MARK: - Manhattan Distance Tests

    func testManhattanDistanceSimple() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = fallback.manhattanDistance(from: a, to: b)

        // |1-0| + |0-1| + |0-0| = 2
        XCTAssertEqual(result, 2.0, accuracy: 0.001)
    }

    func testManhattanDistanceIdentical() {
        let a: [Float] = [1, 2, 3]

        let result = fallback.manhattanDistance(from: a, to: a)

        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }

    // MARK: - Chebyshev Distance Tests

    func testChebyshevDistanceSimple() {
        let a: [Float] = [1, 5, 3]
        let b: [Float] = [4, 1, 2]

        let result = fallback.chebyshevDistance(from: a, to: b)

        // max(|1-4|, |5-1|, |3-2|) = max(3, 4, 1) = 4
        XCTAssertEqual(result, 4.0, accuracy: 0.001)
    }

    // MARK: - Normalization Tests

    func testNormalizeUnitVector() {
        let vector: [Float] = [1, 0, 0]

        let result = fallback.normalize(vector)

        XCTAssertEqual(result, [1, 0, 0])
    }

    func testNormalizeNonUnitVector() {
        let vector: [Float] = [3, 0, 0]

        let result = fallback.normalize(vector)

        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 0.0, accuracy: 0.001)
    }

    func testNormalizeComplexVector() {
        let vector: [Float] = [3, 4, 0]

        let result = fallback.normalize(vector)

        // norm = sqrt(9 + 16) = 5
        XCTAssertEqual(result[0], 0.6, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.8, accuracy: 0.001)
        XCTAssertEqual(result[2], 0.0, accuracy: 0.001)

        // Verify unit length
        let norm = fallback.l2Norm(result)
        XCTAssertEqual(norm, 1.0, accuracy: 0.001)
    }

    func testNormalizeZeroVector() {
        let vector: [Float] = [0, 0, 0]

        let result = fallback.normalize(vector)

        // Zero vector should be unchanged
        XCTAssertEqual(result, [0, 0, 0])
    }

    func testNormalizeBatch() {
        let vectors: [[Float]] = [
            [3, 0, 0],
            [0, 4, 0],
            [3, 4, 0],
        ]

        let results = fallback.normalizeBatch(vectors)

        XCTAssertEqual(results.count, 3)

        // All should have unit length
        for result in results {
            let norm = fallback.l2Norm(result)
            XCTAssertEqual(norm, 1.0, accuracy: 0.001)
        }
    }

    func testL2Norm() {
        let vector: [Float] = [3, 4, 0]

        let result = fallback.l2Norm(vector)

        XCTAssertEqual(result, 5.0, accuracy: 0.001)
    }

    // MARK: - Top-K Tests

    func testTopKByDistance() {
        let distances: [Float] = [0.5, 0.1, 0.3, 0.9, 0.2]

        let results = fallback.topKByDistance(distances, k: 3)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].index, 1)  // 0.1
        XCTAssertEqual(results[1].index, 4)  // 0.2
        XCTAssertEqual(results[2].index, 2)  // 0.3
    }

    func testTopKBySimilarity() {
        let similarities: [Float] = [0.5, 0.9, 0.3, 0.1, 0.8]

        let results = fallback.topKBySimilarity(similarities, k: 3)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].index, 1)  // 0.9
        XCTAssertEqual(results[1].index, 4)  // 0.8
        XCTAssertEqual(results[2].index, 0)  // 0.5
    }

    func testTopKZero() {
        let distances: [Float] = [0.5, 0.1, 0.3]

        let results = fallback.topKByDistance(distances, k: 0)

        XCTAssertTrue(results.isEmpty)
    }

    func testTopKLargerThanArray() {
        let distances: [Float] = [0.5, 0.1]

        let results = fallback.topKByDistance(distances, k: 10)

        XCTAssertEqual(results.count, 2)
    }

    // MARK: - Combined Search Tests

    func testSearchWithEuclidean() {
        let query: [Float] = [0, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],   // Distance 1
            [0, 2, 0],   // Distance 2
            [0, 0, 0.5], // Distance 0.5
        ]

        let results = fallback.search(query: query, in: candidates, k: 2, metric: .euclidean)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].index, 2)  // Closest
        XCTAssertEqual(results[1].index, 0)
    }

    // MARK: - Generic Metric Tests

    func testDistanceWithEuclidean() {
        let a: [Float] = [0, 0]
        let b: [Float] = [3, 4]

        let result = fallback.distance(from: a, to: b, metric: .euclidean)

        XCTAssertEqual(result, 5.0, accuracy: 0.001)
    }

    func testDistanceWithCosine() {
        let a: [Float] = [1, 0]
        let b: [Float] = [0, 1]

        let result = fallback.distance(from: a, to: b, metric: .cosine)

        // 1 - 0 = 1
        XCTAssertEqual(result, 1.0, accuracy: 0.001)
    }

    func testDistanceWithDotProduct() {
        let a: [Float] = [1, 2]
        let b: [Float] = [3, 4]

        let result = fallback.distance(from: a, to: b, metric: .dotProduct)

        // -(1*3 + 2*4) = -11
        XCTAssertEqual(result, -11.0, accuracy: 0.001)
    }

    func testDistanceWithManhattan() {
        let a: [Float] = [1, 2]
        let b: [Float] = [4, 6]

        let result = fallback.distance(from: a, to: b, metric: .manhattan)

        // |1-4| + |2-6| = 3 + 4 = 7
        XCTAssertEqual(result, 7.0, accuracy: 0.001)
    }

    func testDistanceWithChebyshev() {
        let a: [Float] = [1, 2]
        let b: [Float] = [4, 6]

        let result = fallback.distance(from: a, to: b, metric: .chebyshev)

        // max(|1-4|, |2-6|) = max(3, 4) = 4
        XCTAssertEqual(result, 4.0, accuracy: 0.001)
    }

    func testBatchDistanceGeneric() {
        let query: [Float] = [0, 0]
        let candidates: [[Float]] = [
            [1, 0],
            [0, 2],
            [3, 4],
        ]

        let results = fallback.batchDistance(from: query, to: candidates, metric: .euclidean)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(results[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(results[2], 5.0, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testEmptyVector() {
        let a: [Float] = []
        let b: [Float] = []

        XCTAssertEqual(fallback.l2Distance(from: a, to: b), .infinity)
        XCTAssertEqual(fallback.cosineSimilarity(from: a, to: b), 0)
        XCTAssertEqual(fallback.dotProduct(from: a, to: b), 0)
    }

    func testSingleElement() {
        let a: [Float] = [5]
        let b: [Float] = [3]

        XCTAssertEqual(fallback.l2Distance(from: a, to: b), 2.0, accuracy: 0.001)
        XCTAssertEqual(fallback.dotProduct(from: a, to: b), 15.0, accuracy: 0.001)
    }

    func testLargeVector() {
        let dimension = 1000
        let a = [Float](repeating: 1.0, count: dimension)
        let b = [Float](repeating: 2.0, count: dimension)

        let distance = fallback.l2Distance(from: a, to: b)

        // sqrt(1000 * 1^2) = sqrt(1000)
        XCTAssertEqual(distance, sqrt(Float(dimension)), accuracy: 0.01)
    }
}
