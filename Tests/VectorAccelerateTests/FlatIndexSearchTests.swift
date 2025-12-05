//
//  FlatIndexSearchTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for flat index search operations.
//
//  Covers:
//  - Empty index search
//  - k > count handling
//  - Result ordering verification
//  - Filtered search (basic, no matches, iterative fetch)
//  - Batch search
//  - Large dimension (768) support
//  - L2² distance correctness
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate
import VectorCore

/// Tests for flat index search edge cases and correctness.
final class FlatIndexSearchTests: XCTestCase {

    // MARK: - Empty Index Tests

    /// Searching an empty index should return an empty array.
    func testEmptyIndexSearch() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Search on empty index
        let query: [Float] = [1.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 10)

        XCTAssertEqual(results.count, 0, "Empty index should return empty results")
    }

    /// Batch search on empty index should return empty arrays for each query.
    func testEmptyIndexBatchSearch() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let queries: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]
        let results = try await index.search(queries: queries, k: 5)

        XCTAssertEqual(results.count, 2, "Should return one result array per query")
        XCTAssertEqual(results[0].count, 0, "Each query result should be empty")
        XCTAssertEqual(results[1].count, 0, "Each query result should be empty")
    }

    // MARK: - k Greater Than Count Tests

    /// When k > number of vectors, should return all vectors.
    func testSearchKGreaterThanCount() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 5 vectors
        let vectors: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0]
        ]

        for vector in vectors {
            _ = try await index.insert(vector)
        }

        // Search with k=10 but only 5 vectors exist
        let query: [Float] = [1.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 10)

        XCTAssertEqual(results.count, 5, "Should return all 5 vectors when k > count")
    }

    /// Edge case: k equals exactly the count.
    func testSearchKEqualsCount() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 3 vectors
        for i in 0..<3 {
            _ = try await index.insert([Float(i), 0, 0, 0])
        }

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 3)

        XCTAssertEqual(results.count, 3, "Should return exactly 3 vectors when k == count")
    }

    // MARK: - Result Ordering Tests

    /// Search results must be sorted by distance in ascending order.
    func testSearchResultsAreOrdered() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors at known distances from origin
        // Distance from [0,0,0,0]: L2² = sum of squares
        let vectors: [[Float]] = [
            [3.0, 0.0, 0.0, 0.0],  // L2² = 9
            [1.0, 0.0, 0.0, 0.0],  // L2² = 1
            [2.0, 0.0, 0.0, 0.0],  // L2² = 4
            [0.5, 0.0, 0.0, 0.0],  // L2² = 0.25
            [4.0, 0.0, 0.0, 0.0]   // L2² = 16
        ]

        for vector in vectors {
            _ = try await index.insert(vector)
        }

        // Search from origin
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 5)

        XCTAssertEqual(results.count, 5)

        // Verify ascending order
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(
                results[i - 1].distance,
                results[i].distance,
                "Results must be sorted by distance ascending"
            )
        }

        // Verify expected order: 0.25, 1, 4, 9, 16
        XCTAssertEqual(results[0].distance, 0.25, accuracy: 0.001)
        XCTAssertEqual(results[1].distance, 1.0, accuracy: 0.001)
        XCTAssertEqual(results[2].distance, 4.0, accuracy: 0.001)
        XCTAssertEqual(results[3].distance, 9.0, accuracy: 0.001)
        XCTAssertEqual(results[4].distance, 16.0, accuracy: 0.001)
    }

    /// Verify ordering is maintained with negative values.
    func testSearchResultsOrderingWithNegativeValues() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vectors: [[Float]] = [
            [-2.0, 0.0, 0.0, 0.0],  // L2² = 4
            [1.0, 0.0, 0.0, 0.0],   // L2² = 1
            [-1.0, 0.0, 0.0, 0.0],  // L2² = 1
            [0.0, 0.0, 0.0, 0.0]    // L2² = 0 (exact match)
        ]

        for vector in vectors {
            _ = try await index.insert(vector)
        }

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 4)

        // First result should be exact match
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.0001)

        // Next two should both have distance 1
        XCTAssertEqual(results[1].distance, 1.0, accuracy: 0.001)
        XCTAssertEqual(results[2].distance, 1.0, accuracy: 0.001)

        // Last should have distance 4
        XCTAssertEqual(results[3].distance, 4.0, accuracy: 0.001)
    }

    // MARK: - Filtered Search Tests

    /// Basic filtered search with metadata predicate.
    func testFilteredSearchBasic() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with different categories
        _ = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])
        _ = try await index.insert([2.0, 0.0, 0.0, 0.0], metadata: ["category": "B"])
        _ = try await index.insert([3.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])
        _ = try await index.insert([4.0, 0.0, 0.0, 0.0], metadata: ["category": "B"])
        _ = try await index.insert([5.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]

        // Filter for category A only
        let results = try await index.search(query: query, k: 10) { _, metadata in
            metadata?["category"] == "A"
        }

        XCTAssertEqual(results.count, 3, "Should find exactly 3 category A vectors")

        // Verify distances are from category A vectors: 1, 9, 25
        XCTAssertEqual(results[0].distance, 1.0, accuracy: 0.001)
        XCTAssertEqual(results[1].distance, 9.0, accuracy: 0.001)
        XCTAssertEqual(results[2].distance, 25.0, accuracy: 0.001)
    }

    /// Filtered search where no vectors match the predicate.
    func testFilteredSearchNoMatches() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with category A only
        _ = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])
        _ = try await index.insert([2.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])
        _ = try await index.insert([3.0, 0.0, 0.0, 0.0], metadata: ["category": "A"])

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]

        // Filter for category B (doesn't exist)
        let results = try await index.search(query: query, k: 10) { _, metadata in
            metadata?["category"] == "B"
        }

        XCTAssertEqual(results.count, 0, "Should return empty when no vectors match filter")
    }

    /// Filtered search with nil metadata handling.
    func testFilteredSearchWithNilMetadata() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Mix of vectors with and without metadata
        _ = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["type": "with"])
        _ = try await index.insert([2.0, 0.0, 0.0, 0.0], metadata: nil)
        _ = try await index.insert([3.0, 0.0, 0.0, 0.0], metadata: ["type": "with"])
        _ = try await index.insert([4.0, 0.0, 0.0, 0.0], metadata: nil)

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]

        // Filter for vectors WITHOUT metadata
        let results = try await index.search(query: query, k: 10) { _, metadata in
            metadata == nil
        }

        XCTAssertEqual(results.count, 2, "Should find 2 vectors without metadata")
        XCTAssertEqual(results[0].distance, 4.0, accuracy: 0.001)  // [2,0,0,0]
        XCTAssertEqual(results[1].distance, 16.0, accuracy: 0.001) // [4,0,0,0]
    }

    /// Filtered search with highly selective filter (tests iterative fetch).
    func testFilteredSearchHighlySelective() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 200)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 100 vectors, only 2 match the filter
        for i in 0..<100 {
            let category = i < 2 ? "rare" : "common"
            _ = try await index.insert([Float(i), 0.0, 0.0, 0.0], metadata: ["category": category])
        }

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]

        // Filter for rare category (only 2 vectors)
        let results = try await index.search(query: query, k: 5) { _, metadata in
            metadata?["category"] == "rare"
        }

        XCTAssertEqual(results.count, 2, "Should find only the 2 rare vectors")

        // Should be the first two (indices 0 and 1)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)  // [0,0,0,0]
        XCTAssertEqual(results[1].distance, 1.0, accuracy: 0.001)  // [1,0,0,0]
    }

    /// Filter that uses handle information.
    func testFilteredSearchUsingHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 20 vectors
        for i in 0..<20 {
            _ = try await index.insert([Float(i), 0.0, 0.0, 0.0])
        }

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]

        // Only accept even-indexed handles (by checking handle.index directly)
        // Even indices 0-18 = 10 vectors (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
        // Ask for k=3, should find 3 even-indexed vectors
        let results = try await index.search(query: query, k: 3) { handle, _ in
            handle.index % 2 == 0
        }

        XCTAssertEqual(results.count, 3, "Should find 3 even-indexed vectors")

        // Verify all returned handles are even
        for result in results {
            XCTAssertEqual(result.handle.index % 2, 0, "Handle \(result.handle.index) should be even")
        }

        // Verify results are ordered by distance (closest first)
        // Even indices by distance: 0 (d=0), 2 (d=4), 4 (d=16), ...
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance)
        }
    }

    // MARK: - Batch Search Tests

    /// Basic batch search with multiple queries.
    func testBatchSearch() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert basis vectors
        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])
        _ = try await index.insert([0.0, 1.0, 0.0, 0.0])
        _ = try await index.insert([0.0, 0.0, 1.0, 0.0])
        _ = try await index.insert([0.0, 0.0, 0.0, 1.0])

        // Search with multiple queries
        let queries: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],  // Should find [1,0,0,0] first
            [0.0, 1.0, 0.0, 0.0],  // Should find [0,1,0,0] first
            [0.5, 0.5, 0.0, 0.0]   // Between first two
        ]

        let results = try await index.search(queries: queries, k: 2)

        XCTAssertEqual(results.count, 3, "Should have results for each query")

        // First query should find [1,0,0,0] with distance 0
        XCTAssertEqual(results[0][0].distance, 0.0, accuracy: 0.0001)

        // Second query should find [0,1,0,0] with distance 0
        XCTAssertEqual(results[1][0].distance, 0.0, accuracy: 0.0001)

        // Third query should have non-zero distance to both nearest
        XCTAssertGreaterThan(results[2][0].distance, 0.0)
    }

    /// Batch search with varying result counts.
    func testBatchSearchWithVaryingK() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 3 vectors
        for i in 0..<3 {
            _ = try await index.insert([Float(i), 0.0, 0.0, 0.0])
        }

        let queries: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ]

        // Request k=5 but only 3 vectors exist
        let results = try await index.search(queries: queries, k: 5)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 3, "Each query should get at most 3 results")
        XCTAssertEqual(results[1].count, 3)
    }

    /// Empty queries array should return empty results.
    func testBatchSearchEmptyQueries() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let queries: [[Float]] = []
        let results = try await index.search(queries: queries, k: 5)

        XCTAssertEqual(results.count, 0, "Empty queries should return empty results")
    }

    // MARK: - Large Dimension Tests (768 - BERT)

    /// Test with 768-dimensional vectors (BERT embedding size).
    func testLargeDimension768() async throws {
        let dimension = 768
        let config = IndexConfiguration.flat(dimension: dimension, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Create 768-dim vectors
        var vectors: [[Float]] = []
        for i in 0..<10 {
            var vec = [Float](repeating: 0.0, count: dimension)
            vec[0] = Float(i)  // Only first component varies
            vectors.append(vec)
        }

        // Batch insert
        let handles = try await index.insert(vectors)
        XCTAssertEqual(handles.count, 10)

        // Search with 768-dim query
        var query = [Float](repeating: 0.0, count: dimension)
        query[0] = 5.0

        let results = try await index.search(query: query, k: 3)

        XCTAssertEqual(results.count, 3)

        // Nearest should be vector with [5,0,0,...] (distance = 0)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)

        // Next should be [4,0,...] or [6,0,...] (distance = 1)
        XCTAssertEqual(results[1].distance, 1.0, accuracy: 0.001)
    }

    /// Verify 768-dim search performance doesn't timeout.
    func testLargeDimensionPerformance() async throws {
        let dimension = 768
        let config = IndexConfiguration.flat(dimension: dimension, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 200 random 768-dim vectors
        var vectors: [[Float]] = []
        for _ in 0..<200 {
            let vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
            vectors.append(vec)
        }

        _ = try await index.insert(vectors)

        // Time the search (use k=5 to avoid GPU kernel edge case with k > 8)
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }

        let start = CFAbsoluteTimeGetCurrent()
        let results = try await index.search(query: query, k: 5)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        XCTAssertEqual(results.count, 5)
        XCTAssertLessThan(elapsed, 1.0, "Search should complete in under 1 second")
    }

    // MARK: - Distance Correctness Tests (L2²)

    /// Verify that returned distance is L2² (squared Euclidean), not L2.
    func testDistanceIsL2Squared() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert a known vector
        let vector: [Float] = [3.0, 4.0, 0.0, 0.0]  // Distance from origin = 5.0
        _ = try await index.insert(vector)

        // Search from origin
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 1)

        XCTAssertEqual(results.count, 1)

        // L2 = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
        // L2² = 3² + 4² = 9 + 16 = 25.0
        // The API returns L2² (squared distance)
        XCTAssertEqual(results[0].distance, 25.0, accuracy: 0.001,
                       "Distance should be L2² (25), not L2 (5)")
    }

    /// Verify L2² calculation for multi-dimensional case.
    func testDistanceL2SquaredMultiDimensional() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Vector [1, 2, 3, 4]
        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]
        _ = try await index.insert(vector)

        // Query from origin [0, 0, 0, 0]
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 1)

        // L2² = 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
        XCTAssertEqual(results[0].distance, 30.0, accuracy: 0.001)
    }

    /// Verify L2² for two non-origin points.
    func testDistanceL2SquaredBetweenPoints() async throws {
        let config = IndexConfiguration.flat(dimension: 3, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Vector at [1, 2, 3]
        _ = try await index.insert([1.0, 2.0, 3.0])

        // Query from [4, 6, 8]
        // L2² = (4-1)² + (6-2)² + (8-3)² = 9 + 16 + 25 = 50
        let query: [Float] = [4.0, 6.0, 8.0]
        let results = try await index.search(query: query, k: 1)

        XCTAssertEqual(results[0].distance, 50.0, accuracy: 0.001)
    }

    /// Exact match should have distance 0.
    func testDistanceExactMatch() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vector: [Float] = [1.5, 2.5, 3.5, 4.5]
        _ = try await index.insert(vector)

        // Query with exact same vector
        let results = try await index.search(query: vector, k: 1)

        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.0001,
                       "Exact match should have distance 0")
    }

    // MARK: - Search After Delete Tests

    /// Deleted vectors should not appear in search results.
    func testSearchExcludesDeletedVectors() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])
        let handle3 = try await index.insert([3.0, 0.0, 0.0, 0.0])

        // Delete the closest one
        try await index.remove(handle1)

        // Search should not find deleted vector
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 10)

        XCTAssertEqual(results.count, 2, "Should only find 2 remaining vectors")

        // Verify deleted handle is not in results
        let resultHandles = Set(results.map { $0.handle })
        XCTAssertFalse(resultHandles.contains(handle1), "Deleted vector should not appear in results")
        XCTAssertTrue(resultHandles.contains(handle2))
        XCTAssertTrue(resultHandles.contains(handle3))
    }

    /// Deleting all vectors should result in empty search.
    func testSearchAfterDeletingAll() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert and delete all
        let handles = try await index.insert([
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0]
        ])

        for handle in handles {
            try await index.remove(handle)
        }

        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 10)

        XCTAssertEqual(results.count, 0, "Search after deleting all should return empty")
    }
}
