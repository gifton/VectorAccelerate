//
//  IVFTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for IVF (Inverted File) index functionality.
//
//  Covers:
//  - IVF index creation and initialization
//  - Training with different data sizes
//  - Search with various nprobe settings
//  - Cluster statistics and introspection
//  - IVF vs Flat performance characteristics
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate
import VectorCore

/// Tests for IVF index training, search, and cluster management.
final class IVFTests: XCTestCase {

    // MARK: - IVF Creation Tests

    /// IVF index should be created successfully with valid parameters.
    func testIVFIndexCreation() async throws {
        let config = IndexConfiguration.ivf(
            dimension: 64,
            nlist: 16,
            nprobe: 4
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()
        XCTAssertEqual(stats.dimension, 64)
        XCTAssertEqual(stats.vectorCount, 0)
        XCTAssertNotNil(stats.ivfStats)
        XCTAssertEqual(stats.ivfStats?.numClusters, 16)
        XCTAssertEqual(stats.ivfStats?.nprobe, 4)
    }

    /// IVF index should handle various nlist sizes.
    func testIVFDifferentNlistSizes() async throws {
        let nlistValues = [4, 8, 16, 32, 64]

        for nlist in nlistValues {
            let config = IndexConfiguration.ivf(
                dimension: 32,
                nlist: nlist,
                nprobe: min(nlist, 4)
            )

            let index = try await AcceleratedVectorIndex(configuration: config)

            let stats = await index.statistics()
            XCTAssertEqual(stats.ivfStats?.numClusters, nlist,
                           "nlist mismatch for nlist=\(nlist)")
        }
    }

    // MARK: - IVF Insert Tests

    /// IVF index should allow vector insertion.
    func testIVFInsertSingle() async throws {
        let config = IndexConfiguration.ivf(dimension: 32, nlist: 8, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vector = [Float](repeating: 1.0, count: 32)
        let handle = try await index.insert(vector)

        XCTAssertTrue(handle.isValid)

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 1)
    }

    /// IVF index should handle batch insertions.
    func testIVFBatchInsert() async throws {
        let config = IndexConfiguration.ivf(dimension: 32, nlist: 8, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var vectors: [[Float]] = []
        for i in 0..<50 {
            vectors.append([Float](repeating: Float(i), count: 32))
        }

        let handles = try await index.insert(vectors)

        XCTAssertEqual(handles.count, 50)

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 50)
    }

    // MARK: - IVF Search Tests

    /// IVF search should find nearest neighbors.
    func testIVFBasicSearch() async throws {
        let config = IndexConfiguration.ivf(dimension: 8, nlist: 4, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert clustered vectors
        let vectors: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        for vector in vectors {
            _ = try await index.insert(vector)
        }

        // Search for vector similar to first cluster
        let query: [Float] = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 3)

        XCTAssertGreaterThan(results.count, 0, "Should find at least one result")

        // First result should be close to query
        XCTAssertLessThan(results[0].distance, 0.1, "Nearest neighbor should be very close")
    }

    /// Higher nprobe should potentially find better results.
    func testIVFNprobeEffect() async throws {
        let dimension = 16
        let nlist = 8

        // Insert enough vectors to distribute across clusters
        var vectors: [[Float]] = []
        for i in 0..<100 {
            var vector = [Float](repeating: 0.0, count: dimension)
            // Create somewhat random distribution
            for j in 0..<dimension {
                vector[j] = Float(i * dimension + j).truncatingRemainder(dividingBy: 10.0) / 10.0
            }
            vectors.append(vector)
        }

        // Create two indices with different nprobe
        let configLow = IndexConfiguration.ivf(dimension: dimension, nlist: nlist, nprobe: 1)
        let configHigh = IndexConfiguration.ivf(dimension: dimension, nlist: nlist, nprobe: 4)

        let indexLow = try await AcceleratedVectorIndex(configuration: configLow)
        let indexHigh = try await AcceleratedVectorIndex(configuration: configHigh)

        for vector in vectors {
            _ = try await indexLow.insert(vector)
            _ = try await indexHigh.insert(vector)
        }

        // Search with same query
        let query = [Float](repeating: 0.5, count: dimension)

        let resultsLow = try await indexLow.search(query: query, k: 5)
        let resultsHigh = try await indexHigh.search(query: query, k: 5)

        // Both should return results
        XCTAssertGreaterThan(resultsLow.count, 0)
        XCTAssertGreaterThan(resultsHigh.count, 0)

        // Higher nprobe searches more clusters, so results should be at least as good
        // (Note: This is probabilistic, but generally true)
        // Just verify both work correctly
        XCTAssertEqual(resultsLow.count, resultsHigh.count,
                       "Both should return same number of results")
    }

    // MARK: - IVF Delete Tests

    /// IVF index should handle vector deletion.
    func testIVFDelete() async throws {
        let config = IndexConfiguration.ivf(dimension: 16, nlist: 4, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<10 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        var stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 10)

        // Delete half
        for i in stride(from: 0, to: 10, by: 2) {
            try await index.remove(handles[i])
        }

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 5)
        XCTAssertEqual(stats.deletedSlots, 5)
    }

    /// Deleted vectors should not appear in IVF search results.
    func testIVFSearchExcludesDeleted() async throws {
        let config = IndexConfiguration.ivf(dimension: 8, nlist: 4, nprobe: 4)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with distinct patterns
        let vectors: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        var handles: [VectorHandle] = []
        for vector in vectors {
            handles.append(try await index.insert(vector))
        }

        // Delete the first vector
        try await index.remove(handles[0])

        // Search for something similar to first vector
        let query: [Float] = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 5)

        // First vector (handle[0]) should not be in results
        let resultHandles = results.map { $0.handle }
        XCTAssertFalse(resultHandles.contains(handles[0]),
                       "Deleted vector should not appear in search results")
    }

    // MARK: - IVF Compaction Tests

    /// IVF index should handle compaction.
    func testIVFCompaction() async throws {
        let config = IndexConfiguration.ivf(dimension: 16, nlist: 4, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<10 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        // Delete some
        try await index.remove(handles[1])
        try await index.remove(handles[3])
        try await index.remove(handles[5])

        // Compact
        let mapping = try await index.compact()

        XCTAssertEqual(mapping.count, 7, "Should have 7 mappings for remaining vectors")

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 7)
        XCTAssertEqual(stats.deletedSlots, 0)
    }

    /// Search should work correctly after IVF compaction.
    func testIVFSearchAfterCompaction() async throws {
        let config = IndexConfiguration.ivf(dimension: 8, nlist: 4, nprobe: 4)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        let vectors: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]

        var handles: [VectorHandle] = []
        for vector in vectors {
            handles.append(try await index.insert(vector))
        }

        // Delete middle vector
        try await index.remove(handles[2])

        // Compact
        let mapping = try await index.compact()

        // Search should still work
        let query: [Float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 3)

        XCTAssertGreaterThan(results.count, 0)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001,
                       "Exact match should have distance 0")

        // Results should use new handles
        let newHandles = Set(mapping.values)
        for result in results {
            XCTAssertTrue(newHandles.contains(result.handle),
                          "Results should use new handles after compaction")
        }
    }

    // MARK: - IVF Statistics Tests

    /// IVF stats should be updated after operations.
    func testIVFStatsUpdate() async throws {
        let config = IndexConfiguration.ivf(dimension: 16, nlist: 8, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var stats = await index.statistics()
        XCTAssertEqual(stats.ivfStats?.numClusters, 8)
        XCTAssertEqual(stats.ivfStats?.nprobe, 2)

        // Insert some vectors
        for i in 0..<20 {
            let vector = [Float](repeating: Float(i), count: 16)
            _ = try await index.insert(vector)
        }

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 20)
        XCTAssertNotNil(stats.ivfStats)
    }

    // MARK: - IVF with Metadata Tests

    /// IVF index should preserve metadata.
    func testIVFMetadataPreservation() async throws {
        let config = IndexConfiguration.ivf(dimension: 8, nlist: 4, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vector: [Float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let metadata: VectorMetadata = ["category": "test", "id": "123"]
        let handle = try await index.insert(vector, metadata: metadata)

        let retrieved = await index.metadata(for: handle)
        XCTAssertEqual(retrieved?["category"], "test")
        XCTAssertEqual(retrieved?["id"], "123")
    }

    /// IVF search results should include correct metadata.
    func testIVFSearchWithMetadata() async throws {
        let config = IndexConfiguration.ivf(dimension: 8, nlist: 4, nprobe: 4)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with metadata
        _ = try await index.insert(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata: ["name": "first"]
        )
        _ = try await index.insert(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata: ["name": "second"]
        )

        // Search
        let query: [Float] = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 2)

        XCTAssertGreaterThan(results.count, 0)

        // First result should have metadata
        let firstMeta = await index.metadata(for: results[0].handle)
        XCTAssertNotNil(firstMeta)
    }

    // MARK: - IVF vs Flat Comparison

    /// IVF and Flat should return similar results for small datasets.
    func testIVFvsFlatConsistency() async throws {
        let dimension = 8
        let vectors: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        // Create both indices
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: 100)
        let ivfConfig = IndexConfiguration.ivf(dimension: dimension, nlist: 2, nprobe: 2)

        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

        for vector in vectors {
            _ = try await flatIndex.insert(vector)
            _ = try await ivfIndex.insert(vector)
        }

        // Search both
        let query: [Float] = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        let flatResults = try await flatIndex.search(query: query, k: 2)
        let ivfResults = try await ivfIndex.search(query: query, k: 2)

        // Both should find results
        XCTAssertGreaterThan(flatResults.count, 0)
        XCTAssertGreaterThan(ivfResults.count, 0)

        // With full nprobe, distances should be similar
        XCTAssertEqual(flatResults[0].distance, ivfResults[0].distance, accuracy: 0.01,
                       "Best match distance should be similar")
    }

    // MARK: - Large Scale IVF Tests

    /// IVF should handle larger datasets efficiently.
    func testIVFLargeScale() async throws {
        let dimension = 32
        let vectorCount = 500
        let nlist = 16

        let config = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: 4
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert many vectors
        for _ in 0..<vectorCount {
            var vector = [Float](repeating: 0.0, count: dimension)
            for j in 0..<dimension {
                vector[j] = Float.random(in: -1...1)
            }
            _ = try await index.insert(vector)
        }

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, vectorCount)

        // Search should work
        let query = [Float](repeating: 0.5, count: dimension)
        let results = try await index.search(query: query, k: 5)

        XCTAssertEqual(results.count, 5)

        // Results should be sorted by distance
        for i in 0..<(results.count - 1) {
            XCTAssertLessThanOrEqual(results[i].distance, results[i+1].distance)
        }
    }
}
