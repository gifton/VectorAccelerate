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

        // Compact (returns Void with P0.8 stable handles)
        try await index.compact()

        // P0.8: Remaining handles should still be valid
        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 7)
        XCTAssertEqual(stats.deletedSlots, 0)

        // Verify remaining handles are still valid (P0.8)
        let contains0 = await index.contains(handles[0])
        let contains2 = await index.contains(handles[2])
        let contains4 = await index.contains(handles[4])
        XCTAssertTrue(contains0)
        XCTAssertTrue(contains2)
        XCTAssertTrue(contains4)
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

        // Compact (returns Void with P0.8)
        try await index.compact()

        // Search should still work
        let query: [Float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 3)

        XCTAssertGreaterThan(results.count, 0)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001,
                       "Exact match should have distance 0")

        // P0.8: Results should use original stable handles
        let validHandles = Set([handles[0], handles[1], handles[3], handles[4]])
        for result in results {
            XCTAssertTrue(validHandles.contains(result.handle),
                          "Results should use original stable handles (P0.8)")
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

    // MARK: - P0.4: Configurable IVF Training Tests

    /// IVF training should trigger when minTrainingVectors is reached.
    func testIVFTrainingActuallyTriggers() async throws {
        // Use low minTrainingVectors for testing
        let config = IndexConfiguration.ivf(
            dimension: 32,
            nlist: 4,
            nprobe: 2,
            minTrainingVectors: 50
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Initially not trained
        var stats = await index.statistics()
        XCTAssertFalse(stats.ivfStats?.isTrained ?? true, "IVF should not be trained initially")

        // Insert enough vectors to trigger training
        for i in 0..<60 {
            var vector = [Float](repeating: 0.0, count: 32)
            vector[0] = Float(i)
            vector[1] = Float(i % 10)
            _ = try await index.insert(vector)
        }

        stats = await index.statistics()
        XCTAssertTrue(stats.ivfStats?.isTrained == true,
                      "IVF should be trained after 60 inserts with minTrainingVectors=50")
        XCTAssertEqual(stats.vectorCount, 60)
    }

    /// IVF search should work correctly after training.
    func testIVFSearchAfterTraining() async throws {
        let dimension = 16
        let config = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 4,
            nprobe: 4,  // Search all clusters for accuracy
            minTrainingVectors: 40
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with distinct patterns
        var expectedHandles: [VectorHandle] = []
        for i in 0..<50 {
            var vector = [Float](repeating: 0.0, count: dimension)
            // Create 4 clusters by setting first dimension
            let cluster = i % 4
            vector[0] = Float(cluster) * 10.0
            vector[1] = Float(i)
            let h = try await index.insert(vector)
            expectedHandles.append(h)
        }

        // Verify trained
        let stats = await index.statistics()
        XCTAssertTrue(stats.ivfStats?.isTrained == true)

        // Search for vector near cluster 0
        var query = [Float](repeating: 0.0, count: dimension)
        query[0] = 0.5  // Close to cluster 0 (vectors 0, 4, 8, 12, ...)
        query[1] = 0.0

        let results = try await index.search(query: query, k: 5)

        XCTAssertGreaterThan(results.count, 0, "Should find at least one result after training")

        // First result should be very close to query
        XCTAssertLessThan(results[0].distance, 5.0, "Nearest neighbor should be close")
    }

    /// IVF search recall compared to brute force.
    ///
    /// Tests that IVF search achieves reasonable recall compared to brute force.
    /// With P0.1 fix (per-query CSR candidate lists with GPU indirection), recall
    /// should be at least 50% with nprobe=4 (searching half of 8 clusters).
    func testIVFRecallVsBruteForce() async throws {
        let dimension = 16
        let vectorCount = 100
        let nlist = 8
        let nprobe = 4  // Search half the clusters
        let k = 5

        // Create both IVF and flat indexes with same data
        let ivfConfig = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            minTrainingVectors: 50  // Low threshold for testing
        )
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: vectorCount + 10)

        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Insert same vectors into both
        var vectors: [[Float]] = []
        for _ in 0..<vectorCount {
            let vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
            vectors.append(vec)
        }

        let ivfHandles = try await ivfIndex.insert(vectors)
        let flatHandles = try await flatIndex.insert(vectors)

        // Create stableID to vector index mapping for recall calculation
        var ivfHandleToIdx: [UInt32: Int] = [:]
        for (i, h) in ivfHandles.enumerated() {
            ivfHandleToIdx[h.stableID] = i
        }
        var flatHandleToIdx: [UInt32: Int] = [:]
        for (i, h) in flatHandles.enumerated() {
            flatHandleToIdx[h.stableID] = i
        }

        // Verify IVF is trained
        let ivfStats = await ivfIndex.statistics()
        XCTAssertTrue(ivfStats.ivfStats?.isTrained == true)

        // Run multiple queries and compute recall
        var totalRecall: Float = 0.0
        let numQueries = 10

        for _ in 0..<numQueries {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1) }

            // Get ground truth from flat index (exact)
            let flatResults = try await flatIndex.search(query: query, k: k)
            let groundTruth = Set(flatResults.map { flatHandleToIdx[$0.handle.stableID]! })

            // Get IVF results (approximate)
            let ivfResults = try await ivfIndex.search(query: query, k: k)
            let ivfSet = Set(ivfResults.map { ivfHandleToIdx[$0.handle.stableID]! })

            // Compute recall: how many of ground truth did IVF find?
            let intersection = groundTruth.intersection(ivfSet)
            let recall = Float(intersection.count) / Float(groundTruth.count)
            totalRecall += recall
        }

        let avgRecall = totalRecall / Float(numQueries)

        // With P0.1 fix, IVF should achieve reasonable recall.
        // Searching 50% of clusters (nprobe=4/nlist=8) should yield ~50%+ recall.
        XCTAssertGreaterThanOrEqual(avgRecall, 0.5,
                                    "IVF recall should be at least 50% with nprobe=4/nlist=8")

        print("IVF Recall: \(avgRecall * 100)% (nprobe=\(nprobe)/nlist=\(nlist))")
    }

    /// Verify IVF correctness: nprobe = nlist should give near 100% recall.
    func testIVFRecallWithFullNprobe() async throws {
        let dimension = 16
        let vectorCount = 100
        let nlist = 8
        let nprobe = 8  // Search ALL clusters - should match brute force
        let k = 5

        let ivfConfig = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            minTrainingVectors: 50
        )
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: vectorCount + 10)

        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Insert same vectors into both
        var vectors: [[Float]] = []
        for _ in 0..<vectorCount {
            let vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
            vectors.append(vec)
        }

        let ivfHandles = try await ivfIndex.insert(vectors)
        let flatHandles = try await flatIndex.insert(vectors)

        var ivfHandleToIdx: [UInt32: Int] = [:]
        for (i, h) in ivfHandles.enumerated() {
            ivfHandleToIdx[h.stableID] = i
        }
        var flatHandleToIdx: [UInt32: Int] = [:]
        for (i, h) in flatHandles.enumerated() {
            flatHandleToIdx[h.stableID] = i
        }

        // Run multiple queries and compute recall
        var totalRecall: Float = 0.0
        let numQueries = 10

        for _ in 0..<numQueries {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1) }

            let flatResults = try await flatIndex.search(query: query, k: k)
            let groundTruth = Set(flatResults.map { flatHandleToIdx[$0.handle.stableID]! })

            let ivfResults = try await ivfIndex.search(query: query, k: k)
            let ivfSet = Set(ivfResults.map { ivfHandleToIdx[$0.handle.stableID]! })

            let intersection = groundTruth.intersection(ivfSet)
            let recall = Float(intersection.count) / Float(groundTruth.count)
            totalRecall += recall
        }

        let avgRecall = totalRecall / Float(numQueries)

        // With nprobe = nlist (searching all clusters), recall should be very high (95%+)
        // Small variations due to tie-breaking and floating point differences
        XCTAssertGreaterThanOrEqual(avgRecall, 0.95,
                                    "IVF with nprobe=nlist should achieve 95%+ recall")

        print("IVF Full Recall: \(avgRecall * 100)% (nprobe=\(nprobe)/nlist=\(nlist))")
    }

    /// Default minTrainingVectors should work when not specified.
    func testIVFDefaultMinTrainingVectors() async throws {
        // Default is max(nlist * 10, 1000), so for nlist=4, it's 1000
        let config = IndexConfiguration.ivf(
            dimension: 16,
            nlist: 4,
            nprobe: 2
            // Note: minTrainingVectors not specified, uses default
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert fewer than 1000 vectors
        for i in 0..<50 {
            _ = try await index.insert([Float](repeating: Float(i), count: 16))
        }

        // Should NOT be trained (need 1000 vectors by default)
        let stats = await index.statistics()
        XCTAssertFalse(stats.ivfStats?.isTrained ?? true,
                       "IVF should not be trained with only 50 vectors (default needs 1000)")
    }

    /// Manual training should work.
    func testIVFManualTraining() async throws {
        let config = IndexConfiguration.ivf(
            dimension: 16,
            nlist: 4,
            nprobe: 2
            // No minTrainingVectors, use default (1000)
        )

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Disable auto-training
        await index.setAutoTraining(false)

        // Insert enough vectors
        for i in 0..<50 {
            _ = try await index.insert([Float](repeating: Float(i), count: 16))
        }

        // Still not trained (auto-training disabled)
        var stats = await index.statistics()
        XCTAssertFalse(stats.ivfStats?.isTrained ?? true)

        // Manually trigger training
        try await index.train()

        // Now should be trained
        stats = await index.statistics()
        XCTAssertTrue(stats.ivfStats?.isTrained == true,
                      "IVF should be trained after manual train() call")
    }

    // MARK: - P0.3 Regression Test

    /// Regression test: batch IVF search should not have cross-query contamination.
    ///
    /// Before P0.1 fix, batch search built a union of candidates across ALL queries,
    /// causing queries to find results from other queries' probed clusters.
    /// P0.1's CSR-based per-query candidate lists fix this.
    func testIVFBatchSearchNoCrossQueryContamination() async throws {
        let dimension = 32
        let config = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 8,
            nprobe: 1,  // Only probe 1 cluster - makes contamination obvious
            minTrainingVectors: 50
        )
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors that cluster into distinct groups
        // Group A (slots 0-49): vectors near [0,0,0,...]
        // Group B (slots 50-99): vectors near [10,10,10,...]
        for i in 0..<100 {
            let value: Float = i < 50 ? 0.0 : 10.0
            // Add small noise to create realistic clusters
            let noise = Float(i % 10) * 0.01
            var vector = [Float](repeating: value + noise, count: dimension)
            vector[0] = value  // First dimension is the main discriminator
            _ = try await index.insert(vector)
        }

        // Verify trained
        let stats = await index.statistics()
        XCTAssertTrue(stats.ivfStats?.isTrained == true, "IVF should be trained")

        // Batch query: Q1 near group A, Q2 near group B
        let queries: [[Float]] = [
            [Float](repeating: 0.0, count: dimension),   // Should find group A (slots 0-49)
            [Float](repeating: 10.0, count: dimension)   // Should find group B (slots 50-99)
        ]

        let results = try await index.search(queries: queries, k: 5)

        XCTAssertEqual(results.count, 2, "Should have results for both queries")

        // Q1 results should be from group A (slots 0-49)
        for result in results[0] {
            guard let slot = await index.slot(for: result.handle) else {
                XCTFail("Invalid handle in Q1 results")
                continue
            }
            XCTAssertLessThan(slot, 50,
                "Q1 (near [0,0,...]) should only find group A vectors (slots 0-49), got slot \(slot)")
        }

        // Q2 results should be from group B (slots 50-99)
        for result in results[1] {
            guard let slot = await index.slot(for: result.handle) else {
                XCTFail("Invalid handle in Q2 results")
                continue
            }
            XCTAssertGreaterThanOrEqual(slot, 50,
                "Q2 (near [10,10,...]) should only find group B vectors (slots 50-99), got slot \(slot)")
        }
    }
}
