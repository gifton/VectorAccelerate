// CosineSimilarityKernel Tests
// Comprehensive testing for GPU-accelerated cosine similarity computation

import XCTest
import Metal
@testable import VectorAccelerate

final class CosineSimilarityKernelTests: XCTestCase {

    var device: (any MTLDevice)?
    var kernel: CosineSimilarityKernel?

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        kernel = try CosineSimilarityKernel(device: device!)
    }

    // MARK: - Helper Methods

    /// Normalize a vector to unit length
    private func normalize(_ v: [Float]) -> [Float] {
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        guard norm > 1e-8 else { return v }
        return v.map { $0 / norm }
    }

    /// Generate random unit vector
    private func randomUnitVector(dimension: Int) -> [Float] {
        let v = (0..<dimension).map { _ in Float.random(in: -1...1) }
        return normalize(v)
    }

    /// Generate random vector (not normalized)
    private func randomVector(dimension: Int, range: ClosedRange<Float> = -1.0...1.0) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: range) }
    }

    // MARK: - Initialization Tests

    func testInitialization() throws {
        XCTAssertNotNil(kernel, "Kernel should initialize successfully")
    }

    // MARK: - Basic Similarity Tests

    func testIdenticalVectors() async throws {
        let v: [Float] = [1.0, 2.0, 3.0, 4.0]
        let query: [[Float]] = [v]
        let database: [[Float]] = [v]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 4,
            outputDistance: false
        )

        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001,
                      "Identical vectors should have cosine similarity of 1.0")
    }

    func testOrthogonalVectors() async throws {
        // Two orthogonal unit vectors
        let query: [[Float]] = [[1.0, 0.0, 0.0]]
        let database: [[Float]] = [[0.0, 1.0, 0.0]]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3,
            outputDistance: false
        )

        XCTAssertEqual(results[0][0], 0.0, accuracy: 0.001,
                      "Orthogonal vectors should have cosine similarity of 0.0")
    }

    func testOppositeVectors() async throws {
        let v: [Float] = [1.0, 2.0, 3.0]
        let opposite: [Float] = [-1.0, -2.0, -3.0]

        let query: [[Float]] = [v]
        let database: [[Float]] = [opposite]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3,
            outputDistance: false
        )

        XCTAssertEqual(results[0][0], -1.0, accuracy: 0.001,
                      "Opposite vectors should have cosine similarity of -1.0")
    }

    func testSimilarVectors() async throws {
        // Two vectors pointing in similar direction
        let query: [[Float]] = [[1.0, 1.0, 0.0]]
        let database: [[Float]] = [[1.0, 0.9, 0.1]]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3,
            outputDistance: false
        )

        XCTAssertGreaterThan(results[0][0], 0.9,
                            "Similar vectors should have high cosine similarity")
    }

    // MARK: - Distance Mode Tests

    func testDistanceMode() async throws {
        let v1: [Float] = [1.0, 0.0, 0.0]
        let v2: [Float] = [0.707, 0.707, 0.0]  // 45 degrees apart

        // Get similarity
        let similarity = try await kernel!.compute(
            queries: [v1],
            database: [v2],
            dimension: 3,
            outputDistance: false
        )

        // Get distance
        let distance = try await kernel!.compute(
            queries: [v1],
            database: [v2],
            dimension: 3,
            outputDistance: true
        )

        // Distance should be 1 - similarity
        XCTAssertEqual(distance[0][0], 1.0 - similarity[0][0], accuracy: 0.01,
                      "Distance should equal 1 - similarity")
    }

    func testDistanceIdenticalVectors() async throws {
        let v: [Float] = [1.0, 2.0, 3.0]

        let results = try await kernel!.compute(
            queries: [v],
            database: [v],
            dimension: 3,
            outputDistance: true
        )

        XCTAssertEqual(results[0][0], 0.0, accuracy: 0.001,
                      "Identical vectors should have distance of 0.0")
    }

    func testDistanceOrthogonalVectors() async throws {
        let query: [[Float]] = [[1.0, 0.0]]
        let database: [[Float]] = [[0.0, 1.0]]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 2,
            outputDistance: true
        )

        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001,
                      "Orthogonal vectors should have distance of 1.0")
    }

    // MARK: - Pre-normalized Input Tests

    func testPreNormalizedInputs() async throws {
        // Create normalized vectors
        let query = [normalize([1.0, 2.0, 3.0])]
        let database = [normalize([4.0, 5.0, 6.0])]

        // Compute with inputsNormalized=true
        let resultsNormalized = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3,
            outputDistance: false,
            inputsNormalized: true
        )

        // Compute with inputsNormalized=false (should give same result)
        let resultsGeneral = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3,
            outputDistance: false,
            inputsNormalized: false
        )

        XCTAssertEqual(resultsNormalized[0][0], resultsGeneral[0][0], accuracy: 0.01,
                      "Pre-normalized path should give same result as general path")
    }

    // MARK: - Correctness Tests

    func testCorrectnessAgainstCPU() async throws {
        let dimension = 64
        let query = randomVector(dimension: dimension)
        let database = randomVector(dimension: dimension)

        // GPU result
        let gpuResults = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension,
            outputDistance: false
        )

        // CPU reference
        let cpuResult = cpuCosineSimilarity(a: query, b: database, outputDistance: false)

        XCTAssertEqual(gpuResults[0][0], cpuResult, accuracy: 0.01,
                      "GPU result should match CPU reference")
    }

    func testCorrectnessAgainstCPUDistance() async throws {
        let dimension = 64
        let query = randomVector(dimension: dimension)
        let database = randomVector(dimension: dimension)

        // GPU distance
        let gpuResults = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension,
            outputDistance: true
        )

        // CPU reference distance
        let cpuResult = cpuCosineSimilarity(a: query, b: database, outputDistance: true)

        XCTAssertEqual(gpuResults[0][0], cpuResult, accuracy: 0.01,
                      "GPU distance should match CPU reference")
    }

    func testMultipleQueries() async throws {
        let dimension = 16
        let queries: [[Float]] = [
            randomVector(dimension: dimension),
            randomVector(dimension: dimension),
            randomVector(dimension: dimension)
        ]
        let database: [[Float]] = [
            randomVector(dimension: dimension),
            randomVector(dimension: dimension)
        ]

        let results = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: dimension
        )

        // Should have 3 rows (queries) x 2 columns (database)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].count, 2)

        // Verify each result against CPU reference
        for (i, query) in queries.enumerated() {
            for (j, dbVec) in database.enumerated() {
                let cpuResult = cpuCosineSimilarity(a: query, b: dbVec)
                XCTAssertEqual(results[i][j], cpuResult, accuracy: 0.02,
                              "Result[\(i)][\(j)] should match CPU reference")
            }
        }
    }

    func testSymmetry() async throws {
        let dimension = 32
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        // cos(v1, v2)
        let result1 = try await kernel!.compute(
            queries: [v1],
            database: [v2],
            dimension: dimension
        )

        // cos(v2, v1)
        let result2 = try await kernel!.compute(
            queries: [v2],
            database: [v1],
            dimension: dimension
        )

        XCTAssertEqual(result1[0][0], result2[0][0], accuracy: 0.001,
                      "Cosine similarity should be symmetric: cos(a,b) = cos(b,a)")
    }

    // MARK: - Edge Case Tests

    func testZeroVector() async throws {
        let zero: [Float] = [0.0, 0.0, 0.0]
        let nonZero: [Float] = [1.0, 2.0, 3.0]

        let results = try await kernel!.compute(
            queries: [zero],
            database: [nonZero],
            dimension: 3,
            outputDistance: false
        )

        // Zero vector should return 0 similarity (as per CPU reference)
        XCTAssertEqual(results[0][0], 0.0, accuracy: 0.001,
                      "Zero vector should have similarity 0")
    }

    func testZeroVectorDistance() async throws {
        let zero: [Float] = [0.0, 0.0, 0.0]
        let nonZero: [Float] = [1.0, 2.0, 3.0]

        let results = try await kernel!.compute(
            queries: [zero],
            database: [nonZero],
            dimension: 3,
            outputDistance: true
        )

        // Zero vector should return distance 1 (as per CPU reference)
        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001,
                      "Zero vector should have distance 1")
    }

    func testNearZeroVector() async throws {
        let nearZero: [Float] = [1e-10, 1e-10, 1e-10]
        let normal: [Float] = [1.0, 0.0, 0.0]

        let results = try await kernel!.compute(
            queries: [nearZero],
            database: [normal],
            dimension: 3
        )

        // Should handle gracefully without NaN/Inf
        XCTAssertFalse(results[0][0].isNaN, "Result should not be NaN")
        XCTAssertFalse(results[0][0].isInfinite, "Result should not be infinite")
    }

    func testSingleDimension() async throws {
        let query: [[Float]] = [[5.0]]
        let database: [[Float]] = [[3.0]]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 1
        )

        // Same direction in 1D = similarity 1.0
        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001,
                      "Same sign 1D vectors should have similarity 1.0")

        // Opposite direction
        let resultsOpposite = try await kernel!.compute(
            queries: [[5.0]],
            database: [[-3.0]],
            dimension: 1
        )

        XCTAssertEqual(resultsOpposite[0][0], -1.0, accuracy: 0.001,
                      "Opposite sign 1D vectors should have similarity -1.0")
    }

    // MARK: - Optimized Dimension Tests

    /// Test 384-dimension optimized kernel (MiniLM/Sentence-BERT - VectorCore 0.1.5)
    func testDimension384() async throws {
        let dimension = 384
        let query = randomUnitVector(dimension: dimension)
        let database = randomUnitVector(dimension: dimension)

        let gpuResult = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension
        )

        let cpuResult = cpuCosineSimilarity(a: query, b: database)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 0.02,
                      "384D optimized kernel should match CPU reference")
    }

    func testDimension512() async throws {
        let dimension = 512
        let query = randomUnitVector(dimension: dimension)
        let database = randomUnitVector(dimension: dimension)

        let gpuResult = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension
        )

        let cpuResult = cpuCosineSimilarity(a: query, b: database)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 0.02,
                      "512D optimized kernel should match CPU reference")
    }

    func testDimension768() async throws {
        let dimension = 768
        let query = randomUnitVector(dimension: dimension)
        let database = randomUnitVector(dimension: dimension)

        let gpuResult = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension
        )

        let cpuResult = cpuCosineSimilarity(a: query, b: database)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 0.02,
                      "768D optimized kernel should match CPU reference")
    }

    func testDimension1536() async throws {
        let dimension = 1536
        let query = randomUnitVector(dimension: dimension)
        let database = randomUnitVector(dimension: dimension)

        let gpuResult = try await kernel!.compute(
            queries: [query],
            database: [database],
            dimension: dimension
        )

        let cpuResult = cpuCosineSimilarity(a: query, b: database)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 0.02,
                      "1536D optimized kernel should match CPU reference")
    }

    // MARK: - Error Handling Tests

    func testEmptyQueriesThrows() async throws {
        let queries: [[Float]] = []
        let database: [[Float]] = [[1.0, 2.0, 3.0]]

        do {
            _ = try await kernel!.compute(
                queries: queries,
                database: database,
                dimension: 3
            )
            XCTFail("Should throw for empty queries")
        } catch {
            // Expected
        }
    }

    func testEmptyDatabaseThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let database: [[Float]] = []

        do {
            _ = try await kernel!.compute(
                queries: queries,
                database: database,
                dimension: 3
            )
            XCTFail("Should throw for empty database")
        } catch {
            // Expected
        }
    }

    func testDimensionMismatchThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let database: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]  // Different dimension

        do {
            _ = try await kernel!.compute(
                queries: queries,
                database: database,
                dimension: 3
            )
            XCTFail("Should throw for dimension mismatch")
        } catch {
            // Expected
        }
    }

    // MARK: - Similarity Range Tests

    func testSimilarityInValidRange() async throws {
        // Random vectors should produce similarity in [-1, 1]
        let dimension = 128

        for _ in 0..<10 {
            let query = randomVector(dimension: dimension)
            let database = randomVector(dimension: dimension)

            let results = try await kernel!.compute(
                queries: [query],
                database: [database],
                dimension: dimension
            )

            XCTAssertGreaterThanOrEqual(results[0][0], -1.0,
                                       "Similarity should be >= -1")
            XCTAssertLessThanOrEqual(results[0][0], 1.0,
                                    "Similarity should be <= 1")
        }
    }

    func testDistanceInValidRange() async throws {
        // Distance should be in [0, 2]
        let dimension = 128

        for _ in 0..<10 {
            let query = randomVector(dimension: dimension)
            let database = randomVector(dimension: dimension)

            let results = try await kernel!.compute(
                queries: [query],
                database: [database],
                dimension: dimension,
                outputDistance: true
            )

            XCTAssertGreaterThanOrEqual(results[0][0], 0.0,
                                       "Distance should be >= 0")
            XCTAssertLessThanOrEqual(results[0][0], 2.0,
                                    "Distance should be <= 2")
        }
    }

    // MARK: - Performance Tests

    func testPerformance() async throws {
        let queryCount = 10
        let databaseSize = 1000
        let dimension = 128

        let queries = (0..<queryCount).map { _ in randomVector(dimension: dimension) }
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        // Warm-up
        _ = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: dimension
        )

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel!.compute(
                queries: queries,
                database: database,
                dimension: dimension
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("CosineSimilarity performance: \(elapsed / 5.0)s per run (\(queryCount) queries x \(databaseSize) database, dim=\(dimension))")

        XCTAssertLessThan(elapsed / 5.0, 1.0, "Should complete in reasonable time")
    }

    // MARK: - Numerical Robustness Tests (NaN/Inf)

    func testNaNInQuery() async throws {
        // Query containing NaN should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[1] = Float.nan

        let database: [[Float]] = [[1.0, 0.0, 0.0]]

        let results = try await kernel!.compute(
            queries: [query],
            database: database,
            dimension: 3
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Result may be NaN, but should not crash
    }

    func testNaNInDatabase() async throws {
        // Database containing NaN should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        var dbVec: [Float] = [1.0, 0.0, 0.0]
        dbVec[0] = Float.nan

        let results = try await kernel!.compute(
            queries: query,
            database: [dbVec],
            dimension: 3
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
    }

    func testInfinityInQuery() async throws {
        // Query containing Infinity should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[0] = Float.infinity

        let database: [[Float]] = [[1.0, 0.0, 0.0]]

        let results = try await kernel!.compute(
            queries: [query],
            database: database,
            dimension: 3
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
    }

    func testNegativeInfinityInDatabase() async throws {
        // Database containing -Infinity should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        var dbVec: [Float] = [1.0, 0.0, 0.0]
        dbVec[1] = -Float.infinity

        let results = try await kernel!.compute(
            queries: query,
            database: [dbVec],
            dimension: 3
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
    }

    func testMixedNaNAndInfinity() async throws {
        // Mixed special values should not crash
        let query: [[Float]] = [[Float.nan, 1.0, 2.0]]
        let database: [[Float]] = [
            [Float.infinity, 0.0, 0.0],
            [-Float.infinity, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3
        )

        // Should return results without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)
    }

    func testLargeFiniteValues() async throws {
        // Very large but finite values should work correctly
        let largeValue = Float.greatestFiniteMagnitude / 1000.0
        let query: [[Float]] = [[largeValue, largeValue, largeValue]]
        let database: [[Float]] = [[largeValue, largeValue, largeValue]]

        let results = try await kernel!.compute(
            queries: query,
            database: database,
            dimension: 3
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Identical vectors should have similarity close to 1.0
        // (may have numerical issues with very large values)
        XCTAssertFalse(results[0][0].isNaN, "Result should not be NaN")
        XCTAssertFalse(results[0][0].isInfinite, "Result should not be infinite")
    }
}
