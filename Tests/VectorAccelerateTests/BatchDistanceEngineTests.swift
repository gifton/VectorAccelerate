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

    // MARK: - Phantom GPU branches (AUDIT-3 meta-review)

    /// The dot-product GPU branch requested a kernel named "batchDotProduct" that has never
    /// existed in any library. The branch arms at gpuThreshold (1000) candidates with no
    /// decision engine, and — post the VA2-003 k-gate exemption — for large engine-gated
    /// batches too, throwing shaderNotFound where callers previously got CPU results. The
    /// operation must return correct values on every routing, GPU-requested or not.
    func test_batchDotProduct_largeBatchAndExplicitGPU_matchesCPU() async throws {
        var rng = TestRNG(seed: 0x3A32_0001)
        let dim = 64
        let query = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }

        // Threshold-routed default (1000 ≥ gpuThreshold) and the explicit-GPU small batch.
        for (count, useGPU) in [(1000, nil), (8, true)] as [(Int, Bool?)] {
            let candidates = (0..<count).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
            let results = try await engine.batchDotProduct(
                query: query, candidates: candidates, useGPU: useGPU)
            XCTAssertEqual(results.count, count)
            for i in stride(from: 0, to: count, by: max(1, count / 16)) {
                XCTAssertEqual(results[i], cpuDotProduct(query, candidates[i]), accuracy: 1e-3,
                               "count=\(count) useGPU=\(String(describing: useGPU)) idx=\(i)")
            }
        }
    }

    /// Same defect for Manhattan: "batchManhattanDistance" exists in no library.
    func test_batchManhattanDistance_largeBatchAndExplicitGPU_matchesCPU() async throws {
        var rng = TestRNG(seed: 0x3A32_0002)
        let dim = 64
        let query = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }

        for (count, useGPU) in [(1000, nil), (8, true)] as [(Int, Bool?)] {
            let candidates = (0..<count).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
            let results = try await engine.batchManhattanDistance(
                query: query, candidates: candidates, useGPU: useGPU)
            XCTAssertEqual(results.count, count)
            for i in stride(from: 0, to: count, by: max(1, count / 16)) {
                XCTAssertEqual(results[i], cpuManhattanDistance(query, candidates[i]), accuracy: 1e-3,
                               "count=\(count) useGPU=\(String(describing: useGPU)) idx=\(i)")
            }
        }
    }

    /// `batchCosineSimilarity`'s below-simdThreshold CPU leg kept the pre-VA2-008 naive
    /// formula (NaN swallowed to 0 via `queryNorm > 0`, product denominator overflowing to
    /// Inf → similarity 0, no clamp) while the SIMD (≥ 100 candidates) and GPU legs run the
    /// shared rescue — one public API, different answers across the batch-size-100 boundary.
    /// Every CPU routing must agree with AccelerateFallback.
    func test_batchCosineSimilarity_cpuLegsAgreeAcrossSimdBoundary() async throws {
        var rng = TestRNG(seed: 0x3A32_0003)
        let dim = 32

        for (label, poison, scale) in
            [("nanPoisoned", Float.nan, Float(1)), ("huge", nil, Float(1e19))] as [(String, Float?, Float)] {
            for n in [99, 100] {
                let query = (0..<dim).map { _ in rng.nextFloat(in: -1...1) * scale }
                var candidates = (0..<n).map { _ in
                    (0..<dim).map { _ in rng.nextFloat(in: -1...1) * scale }
                }
                if let p = poison { candidates[n / 2][dim / 2] = p }

                let got = try await engine.batchCosineSimilarity(
                    query: query, candidates: candidates, useGPU: false)
                let expected = AccelerateFallback.batchCosineSimilarity(query: query, candidates: candidates)
                XCTAssertEqual(got.count, n)
                for i in 0..<n {
                    let ok = (got[i].isNaN && expected[i].isNaN) || abs(got[i] - expected[i]) <= 1e-4
                    XCTAssertTrue(ok, "\(label) n=\(n) idx=\(i): got=\(got[i]) expected=\(expected[i])")
                }
            }
        }
    }

    // MARK: - Ragged Candidate Rejection

    /// Audit finding (2026-08-21 /audit review of this file): the four batch entries
    /// validated only `candidates[0]`, so a ragged candidate at any later index sailed
    /// past the guard and each backend answered with a different policy. This test pins
    /// the closure: every candidate must match the query dimension, and a violation
    /// throws `dimensionMismatch` from the public entry before routing — so one
    /// small-batch leg per operation covers every routing outcome.
    ///
    /// Pre-guard behaviors this replaces (observed red 2026-08-23, via a stderr
    /// probe because the manhattan crash killed XCTest before its failure output
    /// flushed): euclidean swallowed the ragged slot to +inf, cosine to NaN, dot
    /// returned a zip-truncated partial product on the n < simdThreshold leg
    /// ([7] = 118.0 = the query's first-31-element sum; the vDSP leg answers 0),
    /// and manhattan's n < simdThreshold leg tripped VectorCore's debug-only
    /// dimension assert — in release that leg walks the candidate buffer with the
    /// query's count, an out-of-bounds read for shorter candidates.
    func test_raggedCandidateRejectedUniformlyAcrossOperations() async throws {
        let dim = 32
        let query: [Float] = (0..<dim).map { Float($0 % 7) + 1 }
        let base: [[Float]] = (0..<12).map { i in
            (0..<dim).map { Float(($0 + i) % 5) + 1 }
        }

        func ragged(at index: Int, count: Int) -> [[Float]] {
            var copy = base
            copy[index] = [Float](repeating: 1, count: count)
            return copy
        }

        func expectMismatch(_ label: String, _ body: () async throws -> [Float]) async {
            do {
                let values = try await body()
                let raggedSlot = values.indices.contains(7) ? "\(values[7])" : "missing"
                XCTFail("\(label): expected dimensionMismatch for ragged candidate, "
                    + "got \(values.count) values; value at ragged index 7 = \(raggedSlot)")
            } catch let error as VectorError where error.kind == .dimensionMismatch {
                // Expected: one policy for the whole operation family.
            } catch {
                XCTFail("\(label): expected dimensionMismatch, got \(error)")
            }
        }

        // Index 7 sits past the old candidates[0]-only guard. Sweep a shorter and a
        // longer ragged candidate so the pinned predicate is !=, not <. Manhattan runs
        // last: pre-guard, its small-batch leg died on VectorCore's debug assert, and
        // the crash suppresses XCTest failure output — last place let the three
        // value-returning defects execute (and be probed) before the process died.
        for count in [dim - 1, dim + 3] {
            let candidates = ragged(at: 7, count: count)
            await expectMismatch("euclidean(raggedCount: \(count))") {
                try await self.engine.batchEuclideanDistance(query: query, candidates: candidates)
            }
            await expectMismatch("cosine(raggedCount: \(count))") {
                try await self.engine.batchCosineSimilarity(query: query, candidates: candidates)
            }
            await expectMismatch("dotProduct(raggedCount: \(count))") {
                try await self.engine.batchDotProduct(query: query, candidates: candidates)
            }
            await expectMismatch("manhattan(raggedCount: \(count))") {
                try await self.engine.batchManhattanDistance(query: query, candidates: candidates)
            }
        }

        // Control: uniform candidates flow through the same entries untouched — the
        // guard must reject ragged input without overfiring on valid input.
        let controls: [(String, [Float])] = [
            ("euclidean", try await engine.batchEuclideanDistance(query: query, candidates: base)),
            ("cosine", try await engine.batchCosineSimilarity(query: query, candidates: base)),
            ("dotProduct", try await engine.batchDotProduct(query: query, candidates: base)),
            ("manhattan", try await engine.batchManhattanDistance(query: query, candidates: base)),
        ]
        for (label, values) in controls {
            XCTAssertEqual(values.count, base.count, "\(label) control: wrong result count")
            XCTAssertTrue(values.allSatisfy(\.isFinite),
                "\(label) control: non-finite value in a uniform batch")
        }
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
