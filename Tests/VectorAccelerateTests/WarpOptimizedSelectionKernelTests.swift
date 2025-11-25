// WarpOptimizedSelectionKernel Tests
// Comprehensive testing for GPU-accelerated top-k selection using warp optimization

import XCTest
import Metal
@testable import VectorAccelerate

final class WarpOptimizedSelectionKernelTests: XCTestCase {

    var device: (any MTLDevice)?
    var kernel: WarpOptimizedSelectionKernel?

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        kernel = try WarpOptimizedSelectionKernel(device: device!)
    }

    // MARK: - Helper Methods

    /// CPU reference: select k smallest indices
    private func selectKSmallest(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        let indexed = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { $0.value < $1.value }
        return Array(sorted.prefix(k))
    }

    /// CPU reference: select k largest indices
    private func selectKLargest(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        let indexed = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { $0.value > $1.value }
        return Array(sorted.prefix(k))
    }

    /// Generate random values array
    private func randomValues(count: Int, range: ClosedRange<Float> = 0.0...100.0) -> [Float] {
        (0..<count).map { _ in Float.random(in: range) }
    }

    /// Generate random 2D values array
    private func randomValues2D(queries: Int, candidates: Int) -> [[Float]] {
        (0..<queries).map { _ in randomValues(count: candidates) }
    }

    // MARK: - Initialization Tests

    func testInitialization() throws {
        XCTAssertNotNil(kernel, "Kernel should initialize successfully")
    }

    // MARK: - Warp Selection Tests (k ≤ 32)

    func testWarpSelectSmallestK() async throws {
        // Select 5 smallest from known values
        let values: [[Float]] = [[10.0, 5.0, 15.0, 2.0, 8.0, 1.0, 20.0, 3.0]]
        // Sorted ascending: 1.0(5), 2.0(3), 3.0(7), 5.0(1), 8.0(4), 10.0(0), 15.0(2), 20.0(6)

        let results = try await kernel!.selectTopK(
            from: values,
            k: 3,
            mode: .ascending
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        // The 3 smallest should be 1.0, 2.0, 3.0 (indices 5, 3, 7)
        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(5), "Should include index 5 (value 1.0)")
        XCTAssertTrue(indices.contains(3), "Should include index 3 (value 2.0)")
        XCTAssertTrue(indices.contains(7), "Should include index 7 (value 3.0)")
    }

    func testWarpSelectLargestK() async throws {
        // Select 3 largest from known values
        let values: [[Float]] = [[10.0, 5.0, 15.0, 2.0, 8.0, 1.0, 20.0, 3.0]]
        // Sorted descending: 20.0(6), 15.0(2), 10.0(0), 8.0(4), 5.0(1), 3.0(7), 2.0(3), 1.0(5)

        let results = try await kernel!.selectTopK(
            from: values,
            k: 3,
            mode: .descending
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        // The 3 largest should be 20.0, 15.0, 10.0 (indices 6, 2, 0)
        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(6), "Should include index 6 (value 20.0)")
        XCTAssertTrue(indices.contains(2), "Should include index 2 (value 15.0)")
        XCTAssertTrue(indices.contains(0), "Should include index 0 (value 10.0)")
    }

    func testWarpSelectSingleElement() async throws {
        let values: [[Float]] = [[5.0, 3.0, 8.0, 1.0, 9.0]]

        // Select single smallest
        let resultsAsc = try await kernel!.selectTopK(from: values, k: 1, mode: .ascending)
        XCTAssertEqual(resultsAsc[0].count, 1)
        XCTAssertEqual(resultsAsc[0][0].index, 3, "Smallest value is at index 3")
        XCTAssertEqual(resultsAsc[0][0].value, 1.0, accuracy: 0.01)

        // Select single largest
        let resultsDesc = try await kernel!.selectTopK(from: values, k: 1, mode: .descending)
        XCTAssertEqual(resultsDesc[0].count, 1)
        XCTAssertEqual(resultsDesc[0][0].index, 4, "Largest value is at index 4")
        XCTAssertEqual(resultsDesc[0][0].value, 9.0, accuracy: 0.01)
    }

    func testWarpSelectMaxSmallK() async throws {
        // Test with k=32 (MAX_SMALL_K)
        let candidateCount = 100
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel!.selectTopK(from: values, k: 32, mode: .ascending)

        XCTAssertEqual(results[0].count, 32, "Should return exactly 32 results")

        // Verify correctness against CPU reference
        let cpuResults = selectKSmallest(values[0], k: 32)
        let gpuIndices = Set(results[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU should match CPU reference")
    }

    // MARK: - Correctness Tests

    func testCorrectnessAscending() async throws {
        let candidateCount = 50
        let k = 10
        let values = [randomValues(count: candidateCount)]

        let gpuResults = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)
        let cpuResults = selectKSmallest(values[0], k: k)

        // Compare indices
        let gpuIndices = Set(gpuResults[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU ascending selection should match CPU")
    }

    func testCorrectnessDescending() async throws {
        let candidateCount = 50
        let k = 10
        let values = [randomValues(count: candidateCount)]

        let gpuResults = try await kernel!.selectTopK(from: values, k: k, mode: .descending)
        let cpuResults = selectKLargest(values[0], k: k)

        // Compare indices
        let gpuIndices = Set(gpuResults[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU descending selection should match CPU")
    }

    func testValueAccuracy() async throws {
        let values: [[Float]] = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        let results = try await kernel!.selectTopK(from: values, k: 3, mode: .ascending)

        // Verify values are correct
        for result in results[0] {
            XCTAssertEqual(result.value, values[0][result.index], accuracy: 0.001,
                          "Returned value should match original at index")
        }
    }

    func testMultipleQueries() async throws {
        let values: [[Float]] = [
            [10.0, 1.0, 5.0],  // Min at index 1
            [3.0, 8.0, 2.0],  // Min at index 2
            [7.0, 4.0, 9.0]   // Min at index 1
        ]

        let results = try await kernel!.selectTopK(from: values, k: 1, mode: .ascending)

        XCTAssertEqual(results.count, 3, "Should have results for 3 queries")
        XCTAssertEqual(results[0][0].index, 1, "Query 0 min at index 1")
        XCTAssertEqual(results[1][0].index, 2, "Query 1 min at index 2")
        XCTAssertEqual(results[2][0].index, 1, "Query 2 min at index 1")
    }

    func testResultsAreSorted() async throws {
        let candidateCount = 100
        let k = 20
        let values = [randomValues(count: candidateCount)]

        let resultsAsc = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)

        // Results should be sorted by ascending value
        for i in 0..<(resultsAsc[0].count - 1) {
            XCTAssertLessThanOrEqual(
                resultsAsc[0][i].value,
                resultsAsc[0][i + 1].value,
                "Ascending results should be sorted"
            )
        }

        let resultsDesc = try await kernel!.selectTopK(from: values, k: k, mode: .descending)

        // Results should be sorted by descending value
        for i in 0..<(resultsDesc[0].count - 1) {
            XCTAssertGreaterThanOrEqual(
                resultsDesc[0][i].value,
                resultsDesc[0][i + 1].value,
                "Descending results should be sorted"
            )
        }
    }

    // MARK: - Edge Case Tests

    func testKEqualsArraySize() async throws {
        let values: [[Float]] = [[5.0, 3.0, 8.0, 1.0, 9.0]]

        let results = try await kernel!.selectTopK(from: values, k: 5, mode: .ascending)

        XCTAssertEqual(results[0].count, 5, "Should return all elements")

        // All indices should be present
        let indices = Set(results[0].map { $0.index })
        XCTAssertEqual(indices, Set([0, 1, 2, 3, 4]))
    }

    func testSingleCandidate() async throws {
        let values: [[Float]] = [[42.0]]

        let results = try await kernel!.selectTopK(from: values, k: 1, mode: .ascending)

        XCTAssertEqual(results[0].count, 1)
        XCTAssertEqual(results[0][0].index, 0)
        XCTAssertEqual(results[0][0].value, 42.0, accuracy: 0.01)
    }

    func testIdenticalValues() async throws {
        let values: [[Float]] = [[5.0, 5.0, 5.0, 5.0, 5.0]]

        let results = try await kernel!.selectTopK(from: values, k: 3, mode: .ascending)

        XCTAssertEqual(results[0].count, 3)
        // All values should be 5.0
        for result in results[0] {
            XCTAssertEqual(result.value, 5.0, accuracy: 0.001)
        }
    }

    func testSingleQuery() async throws {
        let candidateCount = 100
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel!.selectTopK(from: values, k: 5, mode: .ascending)

        XCTAssertEqual(results.count, 1, "Should have results for single query")
        XCTAssertEqual(results[0].count, 5, "Should return 5 results")
    }

    // MARK: - Error Handling Tests

    func testEmptyInputThrows() async throws {
        let values: [[Float]] = []

        do {
            _ = try await kernel!.selectTopK(from: values, k: 1, mode: .ascending)
            XCTFail("Should throw for empty input")
        } catch {
            // Expected
        }
    }

    func testKExceedsSmallKLimit() async throws {
        // Test that k > 32 falls back to batch kernel (doesn't throw)
        let values = [randomValues(count: 100)]

        // Should work - will use batch kernel instead of warp kernel
        let results = try await kernel!.selectTopK(from: values, k: 50, mode: .ascending)
        XCTAssertEqual(results[0].count, 50)
    }

    func testKExceedsBatchLimit() async throws {
        // Test k > 128 (MAX_BATCH_K)
        let values = [randomValues(count: 200)]

        do {
            _ = try await kernel!.selectTopK(from: values, k: 150, mode: .ascending)
            XCTFail("Should throw for k > 128")
        } catch {
            // Expected
        }
    }

    // MARK: - Batch Processing Tests

    func testBatchProcessMultipleBatches() async throws {
        let batches: [[[Float]]] = [
            [[10.0, 1.0, 5.0], [3.0, 8.0, 2.0]],  // Batch 0: 2 queries
            [[7.0, 4.0, 9.0], [1.0, 1.0, 1.0]]   // Batch 1: 2 queries
        ]

        let results = try await kernel!.batchProcessSelections(
            batches: batches,
            k: 1,
            mode: .ascending
        )

        XCTAssertEqual(results.count, 2, "Should have 2 batches")
        XCTAssertEqual(results[0].count, 2, "Batch 0 should have 2 queries")
        XCTAssertEqual(results[1].count, 2, "Batch 1 should have 2 queries")

        // Verify results
        XCTAssertEqual(results[0][0][0].index, 1, "Batch 0, Query 0: min at index 1")
        XCTAssertEqual(results[0][1][0].index, 2, "Batch 0, Query 1: min at index 2")
        XCTAssertEqual(results[1][0][0].index, 1, "Batch 1, Query 0: min at index 1")
    }

    // MARK: - Large K Tests (Using Batch Kernel)

    func testLargeK() async throws {
        let candidateCount = 200
        let k = 64  // > 32, will use batch kernel
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)

        XCTAssertEqual(results[0].count, k, "Should return exactly k results")

        // Verify correctness
        let cpuResults = selectKSmallest(values[0], k: k)
        let gpuIndices = Set(results[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "Large k should match CPU reference")
    }

    func testMaxBatchK() async throws {
        let candidateCount = 200
        let k = 128  // MAX_BATCH_K
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)

        XCTAssertEqual(results[0].count, k, "Should return exactly 128 results")
    }

    // MARK: - Performance Tests

    func testPerformance() async throws {
        let queryCount = 100
        let candidateCount = 1000
        let k = 10

        let values = randomValues2D(queries: queryCount, candidates: candidateCount)

        // Warm-up
        _ = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel!.selectTopK(from: values, k: k, mode: .ascending)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("WarpOptimized performance: \(elapsed / 5.0)s per run (\(queryCount) queries, \(candidateCount) candidates, k=\(k))")

        XCTAssertLessThan(elapsed / 5.0, 1.0, "Should complete in reasonable time")
    }

    // MARK: - Numerical Edge Cases

    func testNegativeValues() async throws {
        let values: [[Float]] = [[-5.0, -10.0, -1.0, -8.0, -3.0]]
        // Ascending: -10, -8, -5, -3, -1 (indices 1, 3, 0, 4, 2)

        let results = try await kernel!.selectTopK(from: values, k: 3, mode: .ascending)

        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(1), "Should include index 1 (-10.0)")
        XCTAssertTrue(indices.contains(3), "Should include index 3 (-8.0)")
        XCTAssertTrue(indices.contains(0), "Should include index 0 (-5.0)")
    }

    func testMixedSignValues() async throws {
        let values: [[Float]] = [[-5.0, 0.0, 5.0, -10.0, 10.0]]

        let resultsAsc = try await kernel!.selectTopK(from: values, k: 2, mode: .ascending)
        let indicesAsc = Set(resultsAsc[0].map { $0.index })
        XCTAssertTrue(indicesAsc.contains(3), "Should include -10.0")
        XCTAssertTrue(indicesAsc.contains(0), "Should include -5.0")

        let resultsDesc = try await kernel!.selectTopK(from: values, k: 2, mode: .descending)
        let indicesDesc = Set(resultsDesc[0].map { $0.index })
        XCTAssertTrue(indicesDesc.contains(4), "Should include 10.0")
        XCTAssertTrue(indicesDesc.contains(2), "Should include 5.0")
    }

    func testVerySmallValues() async throws {
        let values: [[Float]] = [[1e-10, 1e-8, 1e-12, 1e-6, 1e-14]]
        // Ascending: 1e-14(4), 1e-12(2), 1e-10(0), 1e-8(1), 1e-6(3)

        let results = try await kernel!.selectTopK(from: values, k: 3, mode: .ascending)

        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(4), "Should include smallest (1e-14)")
        XCTAssertTrue(indices.contains(2), "Should include second smallest (1e-12)")
        XCTAssertTrue(indices.contains(0), "Should include third smallest (1e-10)")
    }

    // MARK: - Numerical Robustness Tests (NaN/Inf)
    // These tests verify the kernel doesn't crash with special floating-point values.
    // The exact behavior with NaN/Inf is implementation-defined, but should never crash.

    func testNaNInValuesDoesNotCrash() async throws {
        // Values containing NaN should not crash
        let values: [[Float]] = [[1.0, Float.nan, 3.0, 2.0, 5.0]]

        // Main assertion: should not crash
        let results = try await kernel!.selectTopK(from: values, k: 2, mode: .ascending)

        // Should return results array
        XCTAssertEqual(results.count, 1, "Should return results array")
    }

    func testInfinityInValuesDoesNotCrash() async throws {
        // Values containing Infinity should not crash
        let values: [[Float]] = [[1.0, Float.infinity, 3.0, 2.0, 5.0]]

        // Main assertion: should not crash
        let results = try await kernel!.selectTopK(from: values, k: 3, mode: .ascending)

        XCTAssertEqual(results.count, 1, "Should return results array")
        XCTAssertEqual(results[0].count, 3, "Should return k results")
    }

    func testNegativeInfinityInValues() async throws {
        // Values containing -Infinity should handle correctly
        let values: [[Float]] = [[1.0, -Float.infinity, 3.0, 2.0, 5.0]]

        // Ascending: -Infinity is smallest
        let results = try await kernel!.selectTopK(from: values, k: 1, mode: .ascending)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // -Infinity should be selected as smallest
        XCTAssertEqual(results[0][0].index, 1, "-Infinity should be the smallest")
    }

    func testInfinityDescending() async throws {
        // Descending: Infinity should be largest
        let values: [[Float]] = [[1.0, Float.infinity, 3.0, 2.0, 5.0]]

        let results = try await kernel!.selectTopK(from: values, k: 1, mode: .descending)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
        // Infinity should be selected as largest
        XCTAssertEqual(results[0][0].index, 1, "Infinity should be the largest")
    }

    func testMixedSpecialValuesDoesNotCrash() async throws {
        // Mixed special values should not crash
        let values: [[Float]] = [[Float.nan, Float.infinity, -Float.infinity, 1.0, 2.0]]

        // Main assertion: should not crash
        let results = try await kernel!.selectTopK(from: values, k: 2, mode: .ascending)

        XCTAssertEqual(results.count, 1, "Should return results array")
    }

    func testAllNaNValuesDoesNotCrash() async throws {
        // All NaN values should not crash
        let values: [[Float]] = [[Float.nan, Float.nan, Float.nan]]

        // Main assertion: should not crash
        // Behavior with all NaN is implementation-defined
        let results = try await kernel!.selectTopK(from: values, k: 2, mode: .ascending)

        XCTAssertEqual(results.count, 1, "Should return results array")
    }
}
