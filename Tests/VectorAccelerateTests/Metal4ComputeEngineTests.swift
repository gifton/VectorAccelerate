//
//  Metal4ComputeEngineTests.swift
//  VectorAccelerateTests
//
//  Tests for Metal4ComputeEngine: distance operations, batch operations,
//  CPU fallback paths, and fused pipeline functionality.
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

final class Metal4ComputeEngineTests: XCTestCase {

    var context: Metal4Context!
    var engine: Metal4ComputeEngine!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        engine = try await Metal4ComputeEngine(context: context, configuration: .default)
    }

    override func tearDown() async throws {
        engine = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - CPU Reference Helpers

    /// CPU Euclidean distance: ||a - b||_2
    private func cpuEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(zip(a, b).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) })
    }

    /// CPU cosine distance: 1 - (a . b) / (||a|| * ||b||)
    private func cpuCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0) { $0 + $1 * $1 })
        let denom = normA * normB
        if denom < 1e-8 { return 1.0 }
        return 1.0 - (dot / denom)
    }

    /// CPU dot product: a . b
    private func cpuDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// CPU Manhattan distance: sum |a_i - b_i|
    private func cpuManhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + abs($1.0 - $1.1) }
    }

    /// CPU Chebyshev distance: max |a_i - b_i|
    private func cpuChebyshevDistance(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    // MARK: - Euclidean Distance Tests

    /// Euclidean distance with known values: [3,0] vs [0,4] = 5.0 (3-4-5 triangle)
    func test_euclideanDistance_knownValues() async throws {
        let vectorA: [Float] = [3.0, 0.0]
        let vectorB: [Float] = [0.0, 4.0]

        let result = try await engine.euclideanDistance(vectorA, vectorB)

        XCTAssertEqual(result, 5.0, accuracy: 1e-4,
                       "Euclidean distance of [3,0] and [0,4] should be 5.0")
    }

    /// Euclidean distance of identical vectors should be 0.0
    func test_euclideanDistance_identicalVectors() async throws {
        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]

        let result = try await engine.euclideanDistance(vector, vector)

        XCTAssertEqual(result, 0.0, accuracy: 1e-4,
                       "Euclidean distance of identical vectors should be 0.0")
    }

    /// Euclidean distance with mismatched dimensions should throw dimensionMismatch
    func test_euclideanDistance_dimensionMismatch() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [1.0, 2.0]

        do {
            _ = try await engine.euclideanDistance(vectorA, vectorB)
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected: dimension mismatch was thrown
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Cosine Distance Tests

    /// Orthogonal vectors should have cosine distance of 1.0
    func test_cosineDistance_orthogonal() async throws {
        let vectorA: [Float] = [1.0, 0.0]
        let vectorB: [Float] = [0.0, 1.0]

        let result = try await engine.cosineDistance(vectorA, vectorB)

        XCTAssertEqual(result, 1.0, accuracy: 1e-4,
                       "Cosine distance of orthogonal vectors should be 1.0")
    }

    /// Parallel (same direction) vectors should have cosine distance near 0.0
    func test_cosineDistance_parallel() async throws {
        let vectorA: [Float] = [1.0, 2.0]
        let vectorB: [Float] = [2.0, 4.0]

        let result = try await engine.cosineDistance(vectorA, vectorB)

        XCTAssertEqual(result, 0.0, accuracy: 1e-4,
                       "Cosine distance of parallel vectors should be approximately 0.0")
    }

    /// Anti-parallel (opposite direction) vectors should have cosine distance near 2.0
    func test_cosineDistance_antiParallel() async throws {
        let vectorA: [Float] = [1.0, 0.0]
        let vectorB: [Float] = [-1.0, 0.0]

        let result = try await engine.cosineDistance(vectorA, vectorB)

        XCTAssertEqual(result, 2.0, accuracy: 1e-4,
                       "Cosine distance of anti-parallel vectors should be approximately 2.0")
    }

    // MARK: - Dot Product Tests

    /// Dot product with known values: [1,2,3] . [4,5,6] = 32.0
    func test_dotProduct_knownValues() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [4.0, 5.0, 6.0]

        let result = try await engine.dotProduct(vectorA, vectorB)

        // dim=3 <= 16 triggers CPU fallback, but result should be correct
        XCTAssertEqual(result, 32.0, accuracy: 1e-4,
                       "Dot product of [1,2,3] and [4,5,6] should be 32.0")
    }

    /// Dot product falls back to CPU when dimension <= 16; verify correctness at dim=8
    func test_dotProduct_cpuFallback_smallDimension() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let vectorB: [Float] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

        let result = try await engine.dotProduct(vectorA, vectorB)
        let expected = cpuDotProduct(vectorA, vectorB)

        // dim=8 is below the 16-element threshold, so CPU path is used
        XCTAssertEqual(result, expected, accuracy: 1e-4,
                       "CPU fallback dot product should match reference value")
    }

    /// Dot product with mismatched dimensions should throw dimensionMismatch
    func test_dotProduct_dimensionMismatch() async throws {
        let vectorA: [Float] = [1.0, 2.0]
        let vectorB: [Float] = [1.0, 2.0, 3.0]

        do {
            _ = try await engine.dotProduct(vectorA, vectorB)
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected: dimension mismatch was thrown
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Manhattan Distance Tests

    /// Manhattan distance with known values: [1,2,3] vs [4,1,5] = |3|+|1|+|2| = 6.0
    func test_manhattanDistance_knownValues() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [4.0, 1.0, 5.0]

        let result = try await engine.manhattanDistance(vectorA, vectorB)

        // dim=3 <= 64, so CPU fallback is used
        XCTAssertEqual(result, 6.0, accuracy: 1e-4,
                       "Manhattan distance of [1,2,3] and [4,1,5] should be 6.0")
    }

    /// Manhattan distance falls back to CPU when dimension <= 64; verify at dim=32
    func test_manhattanDistance_cpuFallback_smallDimension() async throws {
        let vectorA: [Float] = (0..<32).map { Float($0) }
        let vectorB: [Float] = (0..<32).map { Float($0) * 2.0 }

        let result = try await engine.manhattanDistance(vectorA, vectorB)
        let expected = cpuManhattanDistance(vectorA, vectorB)

        // dim=32 is below the 64-element threshold, so CPU path is used
        XCTAssertEqual(result, expected, accuracy: 1e-4,
                       "CPU fallback Manhattan distance should match reference value")
    }

    // MARK: - Chebyshev Distance Tests

    /// Chebyshev distance with known values: [1,5,3] vs [2,1,4] = max(1,4,1) = 4.0
    func test_chebyshevDistance_knownValues() async throws {
        let vectorA: [Float] = [1.0, 5.0, 3.0]
        let vectorB: [Float] = [2.0, 1.0, 4.0]

        let result = try await engine.chebyshevDistance(vectorA, vectorB)

        // dim=3 <= 64, so CPU fallback is used
        XCTAssertEqual(result, 4.0, accuracy: 1e-4,
                       "Chebyshev distance of [1,5,3] and [2,1,4] should be 4.0")
    }

    // MARK: - Batch Euclidean Distance Tests

    /// Batch Euclidean distance with known query and candidates
    func test_batchEuclideanDistance_knownValues() async throws {
        let query: [Float] = [0.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [3.0, 4.0, 0.0],  // distance = 5.0
            [1.0, 0.0, 0.0],  // distance = 1.0
            [0.0, 0.0, 2.0],  // distance = 2.0
        ]

        let results = try await engine.batchEuclideanDistance(query: query, candidates: candidates)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 5.0, accuracy: 1e-4)
        XCTAssertEqual(results[1], 1.0, accuracy: 1e-4)
        XCTAssertEqual(results[2], 2.0, accuracy: 1e-4)
    }

    /// Empty candidates should return empty results
    func test_batchEuclideanDistance_empty() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchEuclideanDistance(query: query, candidates: candidates)

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    /// Batch with dimension mismatch should throw
    func test_batchEuclideanDistance_dimensionMismatch() async throws {
        let query: [Float] = [1.0, 2.0, 3.0]
        let candidates: [[Float]] = [
            [1.0, 2.0],  // Wrong dimension
        ]

        do {
            _ = try await engine.batchEuclideanDistance(query: query, candidates: candidates)
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected: dimension mismatch was thrown
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    /// Batch Euclidean distance falls back to CPU when candidateCount <= 10 and dim <= 16
    func test_batchEuclideanDistance_cpuFallback() async throws {
        let dim = 8
        let query: [Float] = (0..<dim).map { Float($0) * 0.1 }
        let candidates: [[Float]] = (0..<5).map { i in
            (0..<dim).map { j in Float(i + j) * 0.5 }
        }

        let results = try await engine.batchEuclideanDistance(query: query, candidates: candidates)

        // Verify against CPU reference
        XCTAssertEqual(results.count, candidates.count)
        for (i, candidate) in candidates.enumerated() {
            let expected = cpuEuclideanDistance(query, candidate)
            XCTAssertEqual(results[i], expected, accuracy: 1e-4,
                           "CPU fallback batch distance at index \(i) should match reference")
        }
    }

    // MARK: - Batch Cosine Distance Tests

    /// Batch cosine distance with known values
    func test_batchCosineDistance_knownValues() async throws {
        let query: [Float] = [1.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0],   // parallel -> distance ~ 0.0
            [0.0, 1.0],   // orthogonal -> distance ~ 1.0
            [-1.0, 0.0],  // anti-parallel -> distance ~ 2.0
        ]

        let results = try await engine.batchCosineDistance(query: query, candidates: candidates)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0], 0.0, accuracy: 1e-4,
                       "Cosine distance for parallel vectors should be ~0.0")
        XCTAssertEqual(results[1], 1.0, accuracy: 1e-4,
                       "Cosine distance for orthogonal vectors should be ~1.0")
        XCTAssertEqual(results[2], 2.0, accuracy: 1e-4,
                       "Cosine distance for anti-parallel vectors should be ~2.0")
    }

    /// Empty candidates for batch cosine distance should return []
    func test_batchCosineDistance_empty() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = []

        let results = try await engine.batchCosineDistance(query: query, candidates: candidates)

        XCTAssertTrue(results.isEmpty, "Empty candidates should return empty results")
    }

    // MARK: - Fused Distance + Top-K Tests

    /// Fused distance top-K with known vectors, verify correct top 3
    func test_fusedDistanceTopK_basic() async throws {
        let query: [Float] = [0.0, 0.0, 0.0]
        let database: [[Float]] = [
            [10.0, 0.0, 0.0],  // distance = 10.0
            [1.0, 0.0, 0.0],   // distance = 1.0  (nearest)
            [5.0, 0.0, 0.0],   // distance = 5.0
            [3.0, 0.0, 0.0],   // distance = 3.0
            [2.0, 0.0, 0.0],   // distance = 2.0  (2nd nearest)
            [8.0, 0.0, 0.0],   // distance = 8.0
            [7.0, 0.0, 0.0],   // distance = 7.0
            [4.0, 0.0, 0.0],   // distance = 4.0
            [6.0, 0.0, 0.0],   // distance = 6.0
            [9.0, 0.0, 0.0],   // distance = 9.0
        ]

        let topK = try await engine.fusedDistanceTopK(
            query: query,
            database: database,
            k: 3,
            metric: .euclidean
        )

        XCTAssertEqual(topK.count, 3, "Should return exactly k=3 results")

        // Top 3 should be indices 1 (dist=1), 4 (dist=2), 3 (dist=3) in sorted order
        XCTAssertEqual(topK[0].index, 1)
        XCTAssertEqual(topK[0].distance, 1.0, accuracy: 1e-4)
        XCTAssertEqual(topK[1].index, 4)
        XCTAssertEqual(topK[1].distance, 2.0, accuracy: 1e-4)
        XCTAssertEqual(topK[2].index, 3)
        XCTAssertEqual(topK[2].distance, 3.0, accuracy: 1e-4)
    }

    /// When k is larger than the database size, return all candidates
    func test_fusedDistanceTopK_kLargerThanDB() async throws {
        let query: [Float] = [0.0, 0.0]
        let database: [[Float]] = [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]

        let topK = try await engine.fusedDistanceTopK(
            query: query,
            database: database,
            k: 20,
            metric: .euclidean
        )

        XCTAssertEqual(topK.count, 5,
                       "When k > database size, should return all candidates")

        // Verify sorted ascending by distance
        for i in 0..<(topK.count - 1) {
            XCTAssertLessThanOrEqual(topK[i].distance, topK[i + 1].distance,
                                     "Results should be sorted by distance ascending")
        }

        // Verify distances are correct
        XCTAssertEqual(topK[0].distance, 1.0, accuracy: 1e-4)
        XCTAssertEqual(topK[4].distance, 5.0, accuracy: 1e-4)
    }
}
