//
//  AccelerateFallbackTests.swift
//  VectorAccelerateTests
//
//  Tests for AccelerateFallback CPU operations using Apple's Accelerate framework
//

import XCTest
@testable import VectorAccelerate
import VectorCore

final class AccelerateFallbackTests: XCTestCase {

    private let tolerance: Float = 1e-4

    // MARK: - Euclidean Distance Tests

    func test_euclideanDistance_knownValues() throws {
        // dist([3,0], [0,4]) = sqrt(9 + 16) = 5.0
        let result = try AccelerateFallback.euclideanDistance([3, 0], [0, 4])
        XCTAssertEqual(result, 5.0, accuracy: tolerance)
    }

    func test_euclideanDistance_identicalVectors() throws {
        let v: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = try AccelerateFallback.euclideanDistance(v, v)
        XCTAssertEqual(result, 0.0, accuracy: tolerance)
    }

    func test_euclideanDistance_dimensionMismatch() {
        XCTAssertThrowsError(try AccelerateFallback.euclideanDistance([1, 2, 3], [1, 2]))
    }

    func test_euclideanDistance_emptyArrays() throws {
        // Two empty vectors have zero elements; vDSP_distancesq on length 0 leaves result at 0
        let result = try AccelerateFallback.euclideanDistance([], [])
        XCTAssertEqual(result, 0.0, accuracy: tolerance)
    }

    // MARK: - Cosine Similarity Tests

    func test_cosineSimilarity_parallel() throws {
        // [1,2] and [2,4] are parallel -> cosine similarity = 1.0
        let result = try AccelerateFallback.cosineSimilarity([1, 2], [2, 4])
        XCTAssertEqual(result, 1.0, accuracy: tolerance)
    }

    func test_cosineSimilarity_orthogonal() throws {
        // [1,0] and [0,1] are orthogonal -> cosine similarity = 0.0
        let result = try AccelerateFallback.cosineSimilarity([1, 0], [0, 1])
        XCTAssertEqual(result, 0.0, accuracy: tolerance)
    }

    func test_cosineSimilarity_antiParallel() throws {
        // [1,0] and [-1,0] are anti-parallel -> cosine similarity = -1.0
        let result = try AccelerateFallback.cosineSimilarity([1, 0], [-1, 0])
        XCTAssertEqual(result, -1.0, accuracy: tolerance)
    }

    func test_cosineSimilarity_zeroVector() throws {
        // Zero vector against non-zero returns 0 per documented behavior
        let result = try AccelerateFallback.cosineSimilarity([0, 0], [1, 2])
        XCTAssertEqual(result, 0.0, accuracy: tolerance)
    }

    func test_cosineSimilarity_dimensionMismatch() {
        XCTAssertThrowsError(try AccelerateFallback.cosineSimilarity([1, 2], [1, 2, 3]))
    }

    // MARK: - Dot Product Tests

    func test_dotProduct_knownValues() throws {
        // [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32.0
        let result = try AccelerateFallback.dotProduct([1, 2, 3], [4, 5, 6])
        XCTAssertEqual(result, 32.0, accuracy: tolerance)
    }

    // MARK: - Manhattan Distance Tests

    func test_manhattanDistance_knownValues() throws {
        // |1-4| + |2-1| + |3-5| = 3 + 1 + 2 = 6.0
        let result = try AccelerateFallback.manhattanDistance([1, 2, 3], [4, 1, 5])
        XCTAssertEqual(result, 6.0, accuracy: tolerance)
    }

    // MARK: - Normalize Tests

    func test_normalize_unitVector() throws {
        let v: [Float] = [3, 4, 0] // norm = 5
        let normalized = AccelerateFallback.normalize(v)

        // Verify unit length: dot(normalized, normalized) should be 1.0
        let selfDot = try AccelerateFallback.dotProduct(normalized, normalized)
        XCTAssertEqual(selfDot, 1.0, accuracy: tolerance)
    }

    func test_normalize_zeroVector() {
        // Zero vector should be returned unchanged
        let v: [Float] = [0, 0, 0]
        let result = AccelerateFallback.normalize(v)
        XCTAssertEqual(result, v)
    }

    // MARK: - Add / Subtract / Scale Tests

    func test_add_knownValues() {
        let result = AccelerateFallback.add([1, 2, 3], [4, 5, 6])
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 5.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 7.0, accuracy: tolerance)
        XCTAssertEqual(result[2], 9.0, accuracy: tolerance)
    }

    func test_subtract_knownValues() {
        let result = AccelerateFallback.subtract([4, 5, 6], [1, 2, 3])
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 3.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 3.0, accuracy: tolerance)
        XCTAssertEqual(result[2], 3.0, accuracy: tolerance)
    }

    func test_scale_knownValues() {
        let result = AccelerateFallback.scale([1, 2, 3], by: 2.0)
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 2.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 4.0, accuracy: tolerance)
        XCTAssertEqual(result[2], 6.0, accuracy: tolerance)
    }

    // MARK: - Matrix Operations Tests

    func test_matrixVectorMultiply_identity() {
        // 3x3 identity matrix * [1,2,3] = [1,2,3]
        let identity: [Float] = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ]
        let v: [Float] = [1, 2, 3]
        let result = AccelerateFallback.matrixVectorMultiply(
            matrix: identity, vector: v, rows: 3, columns: 3
        )
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 1.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 2.0, accuracy: tolerance)
        XCTAssertEqual(result[2], 3.0, accuracy: tolerance)
    }

    func test_matrixVectorMultiply_dimensionMismatch() {
        // matrix is 2x3 but vector has 2 elements (expects 3)
        let matrix: [Float] = [1, 2, 3, 4, 5, 6]
        let vector: [Float] = [1, 2]
        let result = AccelerateFallback.matrixVectorMultiply(
            matrix: matrix, vector: vector, rows: 2, columns: 3
        )
        XCTAssertTrue(result.isEmpty, "Expected empty array for dimension mismatch")
    }

    // MARK: - Statistical Operations Tests

    func test_mean_knownValues() {
        // mean([2, 4, 6]) = 12 / 3 = 4.0
        let result = AccelerateFallback.mean([2, 4, 6])
        XCTAssertEqual(result, 4.0, accuracy: tolerance)
    }

    func test_variance_knownValues() {
        // variance([2, 4, 6]):
        //   mean = 4, deviations = [-2, 0, 2], squares = [4, 0, 4]
        //   sum = 8, variance = 8/3 = 2.6667
        let result = AccelerateFallback.variance([2, 4, 6])
        let expected: Float = 8.0 / 3.0
        XCTAssertEqual(result, expected, accuracy: tolerance)
    }
}
