// FusedL2TopKKernel Tests
// Comprehensive testing for GPU-accelerated fused L2 distance + top-k selection

import XCTest
import Metal
@testable import VectorAccelerate

final class FusedL2TopKKernelTests: XCTestCase {

    var device: (any MTLDevice)?
    var kernel: FusedL2TopKKernel?

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        kernel = try FusedL2TopKKernel(device: device!)
    }

    // MARK: - Helper Methods

    /// Compute squared L2 distance between two vectors (CPU reference)
    /// Note: The fused kernel returns squared distances for efficiency
    private func l2DistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
    }

    /// Compute L2 distance between two vectors (CPU reference)
    private func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
        return sqrt(l2DistanceSquared(a, b))
    }

    /// Brute-force k-NN for reference (CPU) - uses squared distances to match kernel
    private func bruteForceKNN(query: [Float], dataset: [[Float]], k: Int) -> [(index: Int, distance: Float)] {
        var distances: [(index: Int, distance: Float)] = []
        for (idx, vec) in dataset.enumerated() {
            distances.append((index: idx, distance: l2DistanceSquared(query, vec)))
        }
        distances.sort { $0.distance < $1.distance }
        return Array(distances.prefix(k))
    }

    /// Generate random vector of given dimension
    private func randomVector(dimension: Int, range: ClosedRange<Float> = -1.0...1.0) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: range) }
    }

    /// Generate random dataset
    private func randomDataset(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in randomVector(dimension: dimension) }
    }

    // MARK: - Initialization Tests

    func testInitialization() throws {
        XCTAssertNotNil(kernel, "Kernel should initialize successfully")
    }

    // MARK: - Basic Functionality Tests

    func testFindSingleNearestNeighbor() async throws {
        // Query: origin, Dataset: vectors at various distances
        // Note: Kernel returns SQUARED L2 distances for efficiency
        let query: [[Float]] = [[0.0, 0.0, 0.0]]
        let dataset: [[Float]] = [
            [3.0, 4.0, 0.0],  // L2=5, squared=25
            [1.0, 0.0, 0.0],  // L2=1, squared=1 (closest)
            [2.0, 0.0, 0.0],  // L2=2, squared=4
            [0.0, 10.0, 0.0]  // L2=10, squared=100
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 1, "Should have results for one query")
        XCTAssertEqual(results[0].count, 1, "Should return exactly 1 neighbor")
        XCTAssertEqual(results[0][0].index, 1, "Closest vector should be at index 1")
        XCTAssertEqual(results[0][0].distance, 1.0, accuracy: 0.01, "Distance should be 1.0")
    }

    func testFindTopKNeighbors() async throws {
        // Note: Kernel returns SQUARED L2 distances for efficiency
        let query: [[Float]] = [[0.0, 0.0, 0.0]]
        let dataset: [[Float]] = [
            [5.0, 0.0, 0.0],  // L2=5, squared=25
            [1.0, 0.0, 0.0],  // L2=1, squared=1
            [3.0, 0.0, 0.0],  // L2=3, squared=9
            [2.0, 0.0, 0.0],  // L2=2, squared=4
            [4.0, 0.0, 0.0]   // L2=4, squared=16
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 3
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3, "Should return exactly 3 neighbors")

        // Verify the 3 closest are returned (indices 1, 3, 2 with squared distances 1, 4, 9)
        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(1), "Should include index 1 (squared dist 1)")
        XCTAssertTrue(indices.contains(3), "Should include index 3 (squared dist 4)")
        XCTAssertTrue(indices.contains(2), "Should include index 2 (squared dist 9)")
    }

    func testMultipleQueries() async throws {
        let queries: [[Float]] = [
            [0.0, 0.0],
            [10.0, 10.0]
        ]
        let dataset: [[Float]] = [
            [0.0, 0.0],   // Close to query 0
            [10.0, 10.0], // Close to query 1
            [5.0, 5.0]    // Equidistant
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 2, "Should have results for two queries")
        XCTAssertEqual(results[0][0].index, 0, "Query 0's nearest should be index 0")
        XCTAssertEqual(results[1][0].index, 1, "Query 1's nearest should be index 1")
    }

    // MARK: - Correctness Tests

    func testCorrectnessAgainstBruteForce() async throws {
        let dimension = 64
        let datasetSize = 100
        let k = 8  // Note: kernel has K_PRIVATE=8 limit

        let query = randomVector(dimension: dimension)
        let dataset = randomDataset(count: datasetSize, dimension: dimension)

        // GPU result
        let gpuResults = try await kernel!.findNearestNeighbors(
            queries: [query],
            dataset: dataset,
            k: k
        )

        // CPU reference (uses squared distances to match kernel)
        let cpuResults = bruteForceKNN(query: query, dataset: dataset, k: k)

        XCTAssertEqual(gpuResults[0].count, k, "Should return k results")

        // Compare indices - they should match
        let gpuIndices = Set(gpuResults[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU and CPU should find the same k nearest neighbors")
    }

    func testDistanceAccuracy() async throws {
        // Use vectors with known distances
        // Note: kernel returns SQUARED L2 distances for efficiency
        let query: [[Float]] = [[0.0, 0.0, 0.0, 0.0]]
        let dataset: [[Float]] = [
            [3.0, 4.0, 0.0, 0.0],  // squared distance = 9+16 = 25
            [0.0, 0.0, 5.0, 0.0],  // squared distance = 25
            [1.0, 1.0, 1.0, 1.0]   // squared distance = 1+1+1+1 = 4
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 3
        )

        // Verify squared distances
        for result in results[0] {
            let expectedSquaredDistance = l2DistanceSquared(query[0], dataset[result.index])
            XCTAssertEqual(result.distance, expectedSquaredDistance, accuracy: 0.1,
                          "Squared distance for index \(result.index) should match expected")
        }
    }

    func testResultsAreSortedByDistance() async throws {
        let dimension = 32
        let datasetSize = 50
        let k = 8  // Note: kernel has K_PRIVATE=8 limit

        let query = randomVector(dimension: dimension)
        let dataset = randomDataset(count: datasetSize, dimension: dimension)

        let results = try await kernel!.findNearestNeighbors(
            queries: [query],
            dataset: dataset,
            k: k
        )

        // Verify results are sorted by ascending distance (squared)
        for i in 0..<(results[0].count - 1) {
            XCTAssertLessThanOrEqual(
                results[0][i].distance,
                results[0][i + 1].distance,
                "Results should be sorted by ascending distance"
            )
        }
    }

    // MARK: - Configuration Tests

    func testIncludeDistances() async throws {
        // Note: kernel returns SQUARED L2 distances
        let query: [[Float]] = [[0.0, 0.0]]
        let dataset: [[Float]] = [[1.0, 0.0], [2.0, 0.0]]  // squared distances: 1, 4

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 2,
            includeDistances: true
        )

        // Distances should be populated (squared)
        XCTAssertGreaterThan(results[0][0].distance, 0, "Distance should be populated when includeDistances=true")
        XCTAssertEqual(results[0][0].distance, 1.0, accuracy: 0.01, "Squared distance to [1,0] should be 1")
        XCTAssertEqual(results[0][1].distance, 4.0, accuracy: 0.01, "Squared distance to [2,0] should be 4")
    }

    func testExcludeDistances() async throws {
        let query: [[Float]] = [[0.0, 0.0]]
        let dataset: [[Float]] = [[1.0, 0.0], [2.0, 0.0]]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 2,
            includeDistances: false
        )

        // Indices should still be correct
        XCTAssertEqual(results[0].count, 2)
        // Distances will be 0 when not included
        XCTAssertEqual(results[0][0].distance, 0.0, "Distance should be 0 when includeDistances=false")
    }

    // MARK: - Edge Case Tests

    func testKEqualsDatasetSize() async throws {
        let query: [[Float]] = [[0.0, 0.0, 0.0]]
        let dataset: [[Float]] = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 3  // k equals dataset size
        )

        XCTAssertEqual(results[0].count, 3, "Should return all dataset vectors")

        // All indices should be present
        let indices = Set(results[0].map { $0.index })
        XCTAssertEqual(indices, Set([0, 1, 2]))
    }

    func testSingleVectorInDataset() async throws {
        let query: [[Float]] = [[0.0, 0.0, 0.0]]
        let dataset: [[Float]] = [[1.0, 2.0, 3.0]]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results[0].count, 1)
        XCTAssertEqual(results[0][0].index, 0)
    }

    func testIdenticalVectors() async throws {
        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]
        let query: [[Float]] = [vector]
        let dataset: [[Float]] = [vector, vector, vector]  // All identical

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 3
        )

        XCTAssertEqual(results[0].count, 3)
        // All distances should be 0 (or very close to 0)
        for result in results[0] {
            XCTAssertEqual(result.distance, 0.0, accuracy: 0.001,
                          "Distance to identical vector should be 0")
        }
    }

    func testQueryMatchesDatasetVector() async throws {
        let query: [[Float]] = [[5.0, 5.0, 5.0]]
        let dataset: [[Float]] = [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],  // Same as query
            [10.0, 10.0, 10.0]
        ]

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results[0][0].index, 1, "Nearest neighbor should be the matching vector")
        XCTAssertEqual(results[0][0].distance, 0.0, accuracy: 0.001, "Distance should be 0")
    }

    // MARK: - Error Handling Tests

    func testEmptyQueriesThrows() async throws {
        let queries: [[Float]] = []
        let dataset: [[Float]] = [[1.0, 2.0, 3.0]]

        do {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: 1
            )
            XCTFail("Should throw error for empty queries")
        } catch {
            // Expected
        }
    }

    func testEmptyDatasetThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let dataset: [[Float]] = []

        do {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: 1
            )
            XCTFail("Should throw error for empty dataset")
        } catch {
            // Expected
        }
    }

    func testInvalidKZeroThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let dataset: [[Float]] = [[4.0, 5.0, 6.0]]

        do {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: 0
            )
            XCTFail("Should throw error for k=0")
        } catch {
            // Expected
        }
    }

    func testInvalidKExceedsDatasetThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]
        let dataset: [[Float]] = [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        do {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: 10  // Exceeds dataset size of 2
            )
            XCTFail("Should throw error for k > dataset size")
        } catch {
            // Expected
        }
    }

    func testDimensionMismatchThrows() async throws {
        let queries: [[Float]] = [[1.0, 2.0, 3.0]]        // 3D
        let dataset: [[Float]] = [[4.0, 5.0, 6.0, 7.0]]  // 4D

        do {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: 1
            )
            XCTFail("Should throw error for dimension mismatch")
        } catch {
            // Expected
        }
    }

    // MARK: - Performance Tests

    func testPerformance() async throws {
        let queryCount = 10
        let datasetSize = 1000
        let dimension = 128
        let k = 8  // Note: kernel has K_PRIVATE=8 limit

        let queries = randomDataset(count: queryCount, dimension: dimension)
        let dataset = randomDataset(count: datasetSize, dimension: dimension)

        // Warm-up run
        _ = try await kernel!.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: k
        )

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel!.findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: k
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("FusedL2TopK performance: \(elapsed / 5.0)s per run (\(queryCount) queries, \(datasetSize) dataset, dim=\(dimension), k=\(k))")

        // Should complete in reasonable time
        XCTAssertLessThan(elapsed / 5.0, 1.0, "Each run should complete in under 1 second")
    }

    // MARK: - Higher Dimension Tests

    func testHighDimensionVectors() async throws {
        let dimension = 256  // Common embedding dimension
        let query = [randomVector(dimension: dimension)]
        let dataset = randomDataset(count: 100, dimension: dimension)

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 5
        )

        XCTAssertEqual(results[0].count, 5)

        // Verify against brute force
        let cpuResults = bruteForceKNN(query: query[0], dataset: dataset, k: 5)
        let gpuIndices = Set(results[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "High-dimension results should match CPU reference")
    }

    func testMaxSupportedDimension() async throws {
        let dimension = 512  // MAX_D from kernel
        let query = [randomVector(dimension: dimension)]
        let dataset = randomDataset(count: 50, dimension: dimension)

        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 3
        )

        XCTAssertEqual(results[0].count, 3, "Should work at maximum supported dimension")
    }

    // MARK: - Numerical Robustness Tests
    // These tests verify the kernel doesn't crash with invalid floating-point inputs.
    // The exact behavior with NaN/Inf is implementation-defined, but should never crash.

    func testNaNInQueryDoesNotCrash() async throws {
        // Query containing NaN should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[1] = Float.nan

        let dataset: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]

        // Main assertion: this should not crash
        let results = try await kernel!.findNearestNeighbors(
            queries: [query],
            dataset: dataset,
            k: 1
        )

        // Should return some results (may be empty or contain any values)
        XCTAssertEqual(results.count, 1, "Should return results array for query")
    }

    func testNaNInDatasetDoesNotCrash() async throws {
        // Dataset containing NaN should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        var datasetVec: [Float] = [1.0, 0.0, 0.0]
        datasetVec[0] = Float.nan

        let dataset: [[Float]] = [
            datasetVec,
            [0.0, 1.0, 0.0]  // Valid vector
        ]

        // Main assertion: this should not crash
        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 1, "Should return results array")
    }

    func testInfinityInQueryDoesNotCrash() async throws {
        // Query containing Infinity should not crash
        var query: [Float] = [1.0, 2.0, 3.0]
        query[0] = Float.infinity

        let dataset: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]

        // Main assertion: this should not crash
        let results = try await kernel!.findNearestNeighbors(
            queries: [query],
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 1, "Should return results array")
    }

    func testInfinityInDatasetDoesNotCrash() async throws {
        // Dataset containing Infinity should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]
        let dataset: [[Float]] = [
            [Float.infinity, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]

        // Main assertion: this should not crash
        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 1, "Should return results array")
    }

    func testMixedSpecialValuesDoesNotCrash() async throws {
        // Mixed NaN and Infinity should not crash
        let query: [[Float]] = [[1.0, 2.0, 3.0]]  // Valid query
        let dataset: [[Float]] = [
            [Float.infinity, 0.0, 0.0],
            [-Float.infinity, 1.0, 0.0],
            [1.0, 1.0, 1.0]  // Valid vector
        ]

        // Main assertion: this should not crash
        let results = try await kernel!.findNearestNeighbors(
            queries: query,
            dataset: dataset,
            k: 1
        )

        XCTAssertEqual(results.count, 1, "Should return results array")
    }
}
