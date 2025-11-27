// DotProductKernel Tests
// Comprehensive testing for GPU-accelerated dot product computation

import XCTest
import Metal
@testable import VectorAccelerate

final class DotProductKernelTests: XCTestCase {

    var device: (any MTLDevice)?
    var kernel: DotProductKernel?

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        kernel = try DotProductKernel(device: device!)
    }

    // MARK: - Helper Methods

    /// Generate random vector
    private func randomVector(dimension: Int, range: ClosedRange<Float> = -1.0...1.0) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: range) }
    }

    /// Normalize a vector to unit length
    private func normalize(_ v: [Float]) -> [Float] {
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        guard norm > 1e-8 else { return v }
        return v.map { $0 / norm }
    }

    // MARK: - Initialization Tests

    func testInitialization() throws {
        XCTAssertNotNil(kernel, "Kernel should initialize successfully")
    }

    // MARK: - Basic Dot Product Tests

    func testOrthogonalVectors() async throws {
        // Orthogonal unit vectors
        let v1: [Float] = [1.0, 0.0, 0.0]
        let v2: [Float] = [0.0, 1.0, 0.0]

        let results = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        XCTAssertEqual(results[0][0], 0.0, accuracy: 0.001,
                      "Orthogonal vectors should have dot product of 0")
    }

    func testParallelVectors() async throws {
        // Same unit vector
        let v: [Float] = normalize([1.0, 1.0, 1.0])

        let results = try await kernel!.computeBatch(
            queries: [v],
            database: [v]
        )

        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001,
                      "Parallel unit vectors should have dot product of 1")
    }

    func testAntiParallelVectors() async throws {
        // Opposite unit vectors
        let v: [Float] = normalize([1.0, 1.0, 1.0])
        let opposite: [Float] = v.map { -$0 }

        let results = try await kernel!.computeBatch(
            queries: [v],
            database: [opposite]
        )

        XCTAssertEqual(results[0][0], -1.0, accuracy: 0.001,
                      "Anti-parallel unit vectors should have dot product of -1")
    }

    func testKnownDotProduct() async throws {
        // [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let v1: [Float] = [1.0, 2.0, 3.0]
        let v2: [Float] = [4.0, 5.0, 6.0]

        let results = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        XCTAssertEqual(results[0][0], 32.0, accuracy: 0.01,
                      "Dot product of [1,2,3]·[4,5,6] should be 32")
    }

    func testScaledVectors() async throws {
        // (2*[1,0,0]) · (3*[1,0,0]) = 6
        let v1: [Float] = [2.0, 0.0, 0.0]
        let v2: [Float] = [3.0, 0.0, 0.0]

        let results = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        XCTAssertEqual(results[0][0], 6.0, accuracy: 0.001,
                      "Dot product of scaled parallel vectors should be product of magnitudes")
    }

    // MARK: - GEMV Tests (Single Query)

    func testSingleQueryGEMV() async throws {
        let query: [Float] = [1.0, 2.0, 3.0, 4.0]
        let database: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],  // dot = 1
            [0.0, 1.0, 0.0, 0.0],  // dot = 2
            [0.0, 0.0, 1.0, 0.0],  // dot = 3
            [0.0, 0.0, 0.0, 1.0]   // dot = 4
        ]

        let results = try await kernel!.computeSingle(
            query: query,
            database: database
        )

        XCTAssertEqual(results.count, 4)
        XCTAssertEqual(results[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(results[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(results[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(results[3], 4.0, accuracy: 0.001)
    }

    func testSingleQueryMultipleDatabase() async throws {
        let dimension = 64
        let databaseSize = 100
        let query = randomVector(dimension: dimension)
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        let results = try await kernel!.computeSingle(
            query: query,
            database: database
        )

        XCTAssertEqual(results.count, databaseSize)

        // Verify against CPU reference
        for (i, dbVec) in database.enumerated() {
            let cpuResult = cpuDotProduct(a: query, b: dbVec)
            XCTAssertEqual(results[i], cpuResult, accuracy: 0.1,
                          "GEMV result[\(i)] should match CPU reference")
        }
    }

    // MARK: - GEMM Tests (Batch Queries)

    func testBatchQueriesGEMM() async throws {
        let queries: [[Float]] = [
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        let database: [[Float]] = [
            [1.0, 1.0],  // dots: 1, 1
            [2.0, 0.0]   // dots: 2, 0
        ]

        let results = try await kernel!.computeBatch(
            queries: queries,
            database: database
        )

        // Results should be [queries x database]
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 2)

        // query[0]=[1,0] · db[0]=[1,1] = 1
        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.001)
        // query[0]=[1,0] · db[1]=[2,0] = 2
        XCTAssertEqual(results[0][1], 2.0, accuracy: 0.001)
        // query[1]=[0,1] · db[0]=[1,1] = 1
        XCTAssertEqual(results[1][0], 1.0, accuracy: 0.001)
        // query[1]=[0,1] · db[1]=[2,0] = 0
        XCTAssertEqual(results[1][1], 0.0, accuracy: 0.001)
    }

    func testMultipleQueriesMultipleDatabase() async throws {
        let dimension = 32
        let queryCount = 5
        let databaseSize = 10

        let queries = (0..<queryCount).map { _ in randomVector(dimension: dimension) }
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        let results = try await kernel!.computeBatch(
            queries: queries,
            database: database
        )

        XCTAssertEqual(results.count, queryCount)
        XCTAssertEqual(results[0].count, databaseSize)

        // Verify against CPU reference
        for (i, query) in queries.enumerated() {
            for (j, dbVec) in database.enumerated() {
                let cpuResult = cpuDotProduct(a: query, b: dbVec)
                XCTAssertEqual(results[i][j], cpuResult, accuracy: 0.1,
                              "GEMM result[\(i)][\(j)] should match CPU reference")
            }
        }
    }

    // MARK: - Absolute Value Mode Tests

    func testAbsoluteValueMode() async throws {
        let v1: [Float] = [1.0, 2.0, 3.0]
        let v2: [Float] = [1.0, 1.0, 1.0]  // dot = 6

        let resultsNormal = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2],
            absoluteValue: false
        )

        let resultsAbsolute = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2],
            absoluteValue: true
        )

        // Positive result should be same
        XCTAssertEqual(resultsNormal[0][0], 6.0, accuracy: 0.01)
        XCTAssertEqual(resultsAbsolute[0][0], 6.0, accuracy: 0.01)
    }

    func testAbsoluteValueWithNegativeResult() async throws {
        let v1: [Float] = [1.0, 2.0, 3.0]
        let v2: [Float] = [-1.0, -1.0, -1.0]  // dot = -6

        let resultsNormal = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2],
            absoluteValue: false
        )

        let resultsAbsolute = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2],
            absoluteValue: true
        )

        XCTAssertEqual(resultsNormal[0][0], -6.0, accuracy: 0.01)
        XCTAssertEqual(resultsAbsolute[0][0], 6.0, accuracy: 0.01,
                      "Absolute value mode should return positive value")
    }

    // MARK: - Correctness Tests

    func testCorrectnessAgainstCPU() async throws {
        let dimension = 128
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        let gpuResults = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        let cpuResult = cpuDotProduct(a: v1, b: v2)

        XCTAssertEqual(gpuResults[0][0], cpuResult, accuracy: 0.1,
                      "GPU result should match CPU reference")
    }

    func testSymmetry() async throws {
        let dimension = 64
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        // v1 · v2
        let result1 = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        // v2 · v1
        let result2 = try await kernel!.computeBatch(
            queries: [v2],
            database: [v1]
        )

        XCTAssertEqual(result1[0][0], result2[0][0], accuracy: 0.01,
                      "Dot product should be symmetric: a·b = b·a")
    }

    func testDistributiveProperty() async throws {
        // (a + b) · c = a·c + b·c
        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 5.0, 6.0]
        let c: [Float] = [7.0, 8.0, 9.0]
        let aPlusB: [Float] = zip(a, b).map { $0 + $1 }

        let result_aPlusB_c = try await kernel!.computeBatch(queries: [aPlusB], database: [c])
        let result_a_c = try await kernel!.computeBatch(queries: [a], database: [c])
        let result_b_c = try await kernel!.computeBatch(queries: [b], database: [c])

        let leftSide = result_aPlusB_c[0][0]
        let rightSide = result_a_c[0][0] + result_b_c[0][0]

        XCTAssertEqual(leftSide, rightSide, accuracy: 0.1,
                      "Dot product should satisfy distributive property")
    }

    // MARK: - Edge Case Tests

    func testZeroVector() async throws {
        let zero: [Float] = [0.0, 0.0, 0.0]
        let nonZero: [Float] = [1.0, 2.0, 3.0]

        let results = try await kernel!.computeBatch(
            queries: [zero],
            database: [nonZero]
        )

        XCTAssertEqual(results[0][0], 0.0, accuracy: 0.001,
                      "Dot product with zero vector should be 0")
    }

    func testSingleDimension() async throws {
        let v1: [Float] = [5.0]
        let v2: [Float] = [3.0]

        let results = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        XCTAssertEqual(results[0][0], 15.0, accuracy: 0.001,
                      "1D dot product of 5 and 3 should be 15")
    }

    func testNegativeValues() async throws {
        let v1: [Float] = [-1.0, -2.0, -3.0]
        let v2: [Float] = [1.0, 2.0, 3.0]
        // dot = -1 + -4 + -9 = -14

        let results = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        XCTAssertEqual(results[0][0], -14.0, accuracy: 0.01,
                      "Dot product with negative values should be correct")
    }

    // MARK: - Optimized Dimension Tests

    /// Test 384-dimension optimized kernel (MiniLM/Sentence-BERT - VectorCore 0.1.5)
    func testDimension384() async throws {
        let dimension = 384
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        let gpuResult = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        let cpuResult = cpuDotProduct(a: v1, b: v2)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 1.0,
                      "384D optimized kernel should match CPU reference")
    }

    func testDimension512() async throws {
        let dimension = 512
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        let gpuResult = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        let cpuResult = cpuDotProduct(a: v1, b: v2)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 1.0,
                      "512D optimized kernel should match CPU reference")
    }

    func testDimension768() async throws {
        let dimension = 768
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        let gpuResult = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        let cpuResult = cpuDotProduct(a: v1, b: v2)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 1.0,
                      "768D optimized kernel should match CPU reference")
    }

    func testDimension1536() async throws {
        let dimension = 1536
        let v1 = randomVector(dimension: dimension)
        let v2 = randomVector(dimension: dimension)

        let gpuResult = try await kernel!.computeBatch(
            queries: [v1],
            database: [v2]
        )

        let cpuResult = cpuDotProduct(a: v1, b: v2)

        XCTAssertEqual(gpuResult[0][0], cpuResult, accuracy: 2.0,
                      "1536D optimized kernel should match CPU reference")
    }

    // MARK: - Error Handling Tests

    func testEmptyQueriesThrows() async throws {
        let queries: [[Float]] = []
        let database: [[Float]] = [[1.0, 2.0, 3.0]]

        do {
            _ = try await kernel!.computeBatch(
                queries: queries,
                database: database
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
            _ = try await kernel!.computeBatch(
                queries: queries,
                database: database
            )
            XCTFail("Should throw for empty database")
        } catch {
            // Expected
        }
    }

    func testDimensionMismatchThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let database: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]

        do {
            _ = try await kernel!.computeBatch(
                queries: queries,
                database: database
            )
            XCTFail("Should throw for dimension mismatch")
        } catch {
            // Expected
        }
    }

    // MARK: - Performance Tests

    func testPerformanceGEMV() async throws {
        let dimension = 128
        let databaseSize = 10000

        let query = randomVector(dimension: dimension)
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        // Warm-up
        _ = try await kernel!.computeSingle(query: query, database: database)

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel!.computeSingle(query: query, database: database)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("DotProduct GEMV performance: \(elapsed / 5.0)s per run (1 query x \(databaseSize) database, dim=\(dimension))")

        XCTAssertLessThan(elapsed / 5.0, 1.0, "GEMV should complete in reasonable time")
    }

    func testPerformanceGEMM() async throws {
        let dimension = 128
        let queryCount = 10
        let databaseSize = 1000

        let queries = (0..<queryCount).map { _ in randomVector(dimension: dimension) }
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        // Warm-up
        _ = try await kernel!.computeBatch(queries: queries, database: database)

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel!.computeBatch(queries: queries, database: database)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("DotProduct GEMM performance: \(elapsed / 5.0)s per run (\(queryCount) queries x \(databaseSize) database, dim=\(dimension))")

        XCTAssertLessThan(elapsed / 5.0, 1.0, "GEMM should complete in reasonable time")
    }

    // MARK: - Large Scale Tests

    func testLargeDatabase() async throws {
        let dimension = 64
        let databaseSize = 5000

        let query = randomVector(dimension: dimension)
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        let results = try await kernel!.computeSingle(query: query, database: database)

        XCTAssertEqual(results.count, databaseSize)

        // Spot check a few results
        for i in stride(from: 0, to: databaseSize, by: 500) {
            let cpuResult = cpuDotProduct(a: query, b: database[i])
            XCTAssertEqual(results[i], cpuResult, accuracy: 0.5,
                          "Large database result[\(i)] should match CPU reference")
        }
    }

    func testManyQueries() async throws {
        let dimension = 32
        let queryCount = 100
        let databaseSize = 50

        let queries = (0..<queryCount).map { _ in randomVector(dimension: dimension) }
        let database = (0..<databaseSize).map { _ in randomVector(dimension: dimension) }

        let results = try await kernel!.computeBatch(queries: queries, database: database)

        XCTAssertEqual(results.count, queryCount)
        XCTAssertEqual(results[0].count, databaseSize)
    }

    // MARK: - Numerical Robustness Tests (NaN/Inf)

    func testNaNInQuery() async throws {
        // Query containing NaN should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[1] = Float.nan

        let database: [[Float]] = [[1.0, 0.0, 0.0]]

        let results = try await kernel!.computeBatch(
            queries: [query],
            database: database
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Result will be NaN since input contains NaN
        XCTAssertTrue(results[0][0].isNaN, "Result should be NaN when input contains NaN")
    }

    func testNaNInDatabase() async throws {
        // Database containing NaN should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        var dbVec: [Float] = [1.0, 0.0, 0.0]
        dbVec[0] = Float.nan

        let results = try await kernel!.computeBatch(
            queries: query,
            database: [dbVec]
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Result will be NaN since input contains NaN
        XCTAssertTrue(results[0][0].isNaN, "Result should be NaN when database contains NaN")
    }

    func testInfinityInQuery() async throws {
        // Query containing Infinity should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[0] = Float.infinity

        let database: [[Float]] = [[1.0, 0.0, 0.0]]

        let results = try await kernel!.computeBatch(
            queries: [query],
            database: database
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Result should be infinity (inf * 1.0 = inf)
        XCTAssertTrue(results[0][0].isInfinite, "Result should be infinite when input contains infinity")
    }

    func testNegativeInfinityInDatabase() async throws {
        // Database containing -Infinity should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        var dbVec: [Float] = [1.0, 0.0, 0.0]
        dbVec[0] = -Float.infinity

        let results = try await kernel!.computeBatch(
            queries: query,
            database: [dbVec]
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // 1.0 * -inf + 2.0 * 0.0 + 3.0 * 0.0 = -inf
        XCTAssertTrue(results[0][0].isInfinite, "Result should be infinite")
        XCTAssertLessThan(results[0][0], 0, "Result should be negative infinity")
    }

    func testMixedNaNAndInfinity() async throws {
        // Mixed special values should not crash
        let query: [[Float]] = [[Float.nan, 1.0, 2.0]]
        let database: [[Float]] = [
            [Float.infinity, 0.0, 0.0],
            [-Float.infinity, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ]

        let results = try await kernel!.computeBatch(
            queries: query,
            database: database
        )

        // Should return results without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)
    }

    func testInfinityTimesZero() async throws {
        // Infinity * 0 = NaN (IEEE 754)
        let query: [[Float]] = [[Float.infinity, 0.0, 0.0]]
        let database: [[Float]] = [[0.0, 1.0, 1.0]]  // inf * 0 + 0 * 1 + 0 * 1 = NaN

        let results = try await kernel!.computeBatch(
            queries: query,
            database: database
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // inf * 0 = NaN per IEEE 754
        XCTAssertTrue(results[0][0].isNaN, "Infinity * 0 should produce NaN")
    }

    func testLargeFiniteValues() async throws {
        // Very large but finite values that might overflow
        let largeValue = Float.greatestFiniteMagnitude / 10.0
        let query: [[Float]] = [[largeValue, largeValue, largeValue]]
        let database: [[Float]] = [[1.0, 1.0, 1.0]]

        let results = try await kernel!.computeBatch(
            queries: query,
            database: database
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // largeValue * 3 might overflow to infinity
        // This is expected behavior for floating point
        XCTAssertFalse(results[0][0].isNaN, "Result should not be NaN")
    }

    func testAbsoluteValueWithNaN() async throws {
        // Absolute value mode with NaN
        var query: [Float] = [1.0, 2.0, 3.0]
        query[1] = Float.nan

        let database: [[Float]] = [[1.0, 0.0, 0.0]]

        let results = try await kernel!.computeBatch(
            queries: [query],
            database: database,
            absoluteValue: true
        )

        // Should return a result without crashing
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // abs(NaN) = NaN
        XCTAssertTrue(results[0][0].isNaN, "abs(NaN) should still be NaN")
    }
}
