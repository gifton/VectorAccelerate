//
//  IntrospectionTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for index introspection and statistics.
//
//  Covers:
//  - GPUIndexStats accuracy
//  - Capacity and utilization tracking
//  - Memory reporting
//  - Index type introspection
//  - shouldCompact recommendations
//  - Configuration introspection
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate
import VectorCore

/// Tests for index introspection, statistics, and configuration.
final class IntrospectionTests: XCTestCase {

    // MARK: - Basic Statistics Tests

    /// Empty index should report zero counts.
    func testEmptyIndexStatistics() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()

        XCTAssertEqual(stats.vectorCount, 0)
        XCTAssertEqual(stats.allocatedSlots, 0)
        XCTAssertEqual(stats.deletedSlots, 0)
        XCTAssertEqual(stats.dimension, 128)
        XCTAssertEqual(stats.capacity, 1000)
    }

    /// Statistics should accurately reflect inserted vectors.
    func testStatisticsAfterInserts() async throws {
        let config = IndexConfiguration.flat(dimension: 64, capacity: 500)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 10 vectors
        for i in 0..<10 {
            let vector = [Float](repeating: Float(i), count: 64)
            _ = try await index.insert(vector)
        }

        let stats = await index.statistics()

        XCTAssertEqual(stats.vectorCount, 10)
        XCTAssertEqual(stats.allocatedSlots, 10)
        XCTAssertEqual(stats.deletedSlots, 0)
    }

    /// Statistics should accurately reflect deletions.
    func testStatisticsAfterDeletions() async throws {
        let config = IndexConfiguration.flat(dimension: 32, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<10 {
            let vector = [Float](repeating: Float(i), count: 32)
            handles.append(try await index.insert(vector))
        }

        // Delete 3 vectors
        try await index.remove(handles[0])
        try await index.remove(handles[5])
        try await index.remove(handles[9])

        let stats = await index.statistics()

        XCTAssertEqual(stats.vectorCount, 7, "Should have 7 active vectors")
        XCTAssertEqual(stats.allocatedSlots, 10, "Should still have 10 allocated slots")
        XCTAssertEqual(stats.deletedSlots, 3, "Should have 3 deleted slots")
    }

    /// Statistics should reset after compaction.
    func testStatisticsAfterCompaction() async throws {
        let config = IndexConfiguration.flat(dimension: 32, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<10 {
            let vector = [Float](repeating: Float(i), count: 32)
            handles.append(try await index.insert(vector))
        }

        // Delete half
        for i in stride(from: 0, to: 10, by: 2) {
            try await index.remove(handles[i])
        }

        // Compact
        _ = try await index.compact()

        let stats = await index.statistics()

        XCTAssertEqual(stats.vectorCount, 5, "Should have 5 active vectors")
        XCTAssertEqual(stats.allocatedSlots, 5, "Allocated slots should equal vector count after compact")
        XCTAssertEqual(stats.deletedSlots, 0, "Should have no deleted slots after compact")
    }

    // MARK: - Fragmentation and Utilization Tests

    /// Fragmentation ratio should be calculated correctly.
    func testFragmentationRatio() async throws {
        let config = IndexConfiguration.flat(dimension: 16, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<20 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        // Delete 5 vectors (25% of allocated)
        for i in 0..<5 {
            try await index.remove(handles[i])
        }

        let stats = await index.statistics()

        // fragmentation = deleted / allocated = 5 / 20 = 0.25
        XCTAssertEqual(stats.fragmentationRatio, 0.25, accuracy: 0.001)
    }

    /// Utilization ratio should be calculated correctly.
    func testUtilizationRatio() async throws {
        let config = IndexConfiguration.flat(dimension: 16, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 50 vectors into a 100-capacity index
        for i in 0..<50 {
            let vector = [Float](repeating: Float(i), count: 16)
            _ = try await index.insert(vector)
        }

        let stats = await index.statistics()

        // utilization = vectorCount / capacity = 50 / 100 = 0.5
        XCTAssertEqual(stats.utilizationRatio, 0.5, accuracy: 0.001)
    }

    /// shouldCompact should return true when fragmentation is high.
    func testShouldCompactHighFragmentation() async throws {
        let config = IndexConfiguration.flat(dimension: 16, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<20 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        // Delete 6 vectors (30% > 25% threshold)
        for i in 0..<6 {
            try await index.remove(handles[i])
        }

        let stats = await index.statistics()

        XCTAssertTrue(stats.shouldCompact, "Should recommend compaction at 30% fragmentation")
    }

    /// shouldCompact should return false when fragmentation is low.
    func testShouldCompactLowFragmentation() async throws {
        let config = IndexConfiguration.flat(dimension: 16, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<20 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        // Delete only 2 vectors (10% < 25% threshold)
        try await index.remove(handles[0])
        try await index.remove(handles[1])

        let stats = await index.statistics()

        XCTAssertFalse(stats.shouldCompact, "Should not recommend compaction at 10% fragmentation")
    }

    // MARK: - Index Type Tests

    /// Flat index should report correct index type.
    func testFlatIndexType() async throws {
        let config = IndexConfiguration.flat(dimension: 64, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()

        if case .flat = stats.indexType {
            // Expected
        } else {
            XCTFail("Expected flat index type, got \(stats.indexType)")
        }

        XCTAssertNil(stats.ivfStats, "Flat index should have no IVF stats")
    }

    /// IVF index should report correct index type and IVF stats.
    func testIVFIndexType() async throws {
        let config = IndexConfiguration.ivf(dimension: 64, nlist: 16, nprobe: 4)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()

        if case .ivf(let nlist, let nprobe) = stats.indexType {
            XCTAssertEqual(nlist, 16)
            XCTAssertEqual(nprobe, 4)
        } else {
            XCTFail("Expected IVF index type, got \(stats.indexType)")
        }

        XCTAssertNotNil(stats.ivfStats, "IVF index should have IVF stats")
        XCTAssertEqual(stats.ivfStats?.numClusters, 16)
    }

    // MARK: - Metric Tests

    /// Index should report correct distance metric.
    func testMetricReporting() async throws {
        // Default (euclidean)
        let euclideanConfig = IndexConfiguration.flat(dimension: 64, capacity: 100)
        let euclideanIndex = try await AcceleratedVectorIndex(configuration: euclideanConfig)

        let euclideanStats = await euclideanIndex.statistics()
        XCTAssertEqual(euclideanStats.metric, .euclidean)
    }

    // MARK: - Dimension Tests

    /// Statistics should report correct dimension.
    func testDimensionReporting() async throws {
        let dimensions = [32, 128, 512, 768, 1536]

        for dim in dimensions {
            let config = IndexConfiguration.flat(dimension: dim, capacity: 100)
            let index = try await AcceleratedVectorIndex(configuration: config)

            let stats = await index.statistics()
            XCTAssertEqual(stats.dimension, dim, "Dimension mismatch for dim=\(dim)")
        }
    }

    // MARK: - Capacity Tests

    /// Statistics should report correct capacity.
    func testCapacityReporting() async throws {
        let capacities = [100, 1000, 10_000]

        for cap in capacities {
            let config = IndexConfiguration.flat(dimension: 64, capacity: cap)
            let index = try await AcceleratedVectorIndex(configuration: config)

            let stats = await index.statistics()
            XCTAssertEqual(stats.capacity, cap, "Capacity mismatch for cap=\(cap)")
        }
    }

    // MARK: - Memory Reporting Tests

    /// GPU vector memory should scale with vector count and dimension.
    func testGPUVectorMemoryScaling() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 100 vectors
        for i in 0..<100 {
            let vector = [Float](repeating: Float(i), count: 128)
            _ = try await index.insert(vector)
        }

        let stats = await index.statistics()

        // Expected memory: 100 vectors * 128 dimensions * 4 bytes = 51,200 bytes
        // Allow some overhead for alignment
        XCTAssertGreaterThan(stats.gpuVectorMemoryBytes, 0, "GPU memory should be reported")

        // Add more vectors and check memory increases
        for i in 100..<200 {
            let vector = [Float](repeating: Float(i), count: 128)
            _ = try await index.insert(vector)
        }

        let stats2 = await index.statistics()
        XCTAssertGreaterThan(stats2.gpuVectorMemoryBytes, stats.gpuVectorMemoryBytes,
                             "Memory should increase with more vectors")
    }

    // MARK: - IVF Stats Tests

    /// IVF index should report cluster statistics.
    func testIVFClusterStats() async throws {
        let config = IndexConfiguration.ivf(dimension: 32, nlist: 8, nprobe: 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()

        XCTAssertNotNil(stats.ivfStats)
        XCTAssertEqual(stats.ivfStats?.numClusters, 8)
        XCTAssertEqual(stats.ivfStats?.nprobe, 2)
    }

    // MARK: - Index Count Tests

    /// Index should track vector count accurately across operations.
    func testVectorCountAccuracy() async throws {
        let config = IndexConfiguration.flat(dimension: 16, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 0)

        // Insert 5
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 5)

        // Delete 2
        try await index.remove(handles[0])
        try await index.remove(handles[1])

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 3)

        // Insert 3 more
        for i in 5..<8 {
            let vector = [Float](repeating: Float(i), count: 16)
            handles.append(try await index.insert(vector))
        }

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 6)

        // Compact
        _ = try await index.compact()

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 6, "Vector count should be preserved after compact")
        XCTAssertEqual(stats.deletedSlots, 0, "No deleted slots after compact")
    }

    // MARK: - Configuration Introspection

    /// Default configuration should have expected values.
    func testDefaultConfigurationValues() {
        let config = IndexAccelerationConfiguration.default

        XCTAssertEqual(config.minimumCandidatesForGPU, 500)
        XCTAssertEqual(config.minimumOperationsForGPU, 50_000)
        XCTAssertTrue(config.useFusedKernels)
        XCTAssertFalse(config.forceGPU)
        XCTAssertFalse(config.enableProfiling)
    }

    /// Aggressive configuration should have lower thresholds.
    func testAggressiveConfigurationValues() {
        let config = IndexAccelerationConfiguration.aggressive

        XCTAssertEqual(config.minimumCandidatesForGPU, 100)
        XCTAssertTrue(config.preallocateBuffers)
    }

    /// Conservative configuration should have higher thresholds.
    func testConservativeConfigurationValues() {
        let config = IndexAccelerationConfiguration.conservative

        XCTAssertEqual(config.minimumCandidatesForGPU, 5_000)
        XCTAssertFalse(config.preallocateBuffers)
    }

    /// Benchmarking configuration should force GPU and enable profiling.
    func testBenchmarkingConfigurationValues() {
        let config = IndexAccelerationConfiguration.benchmarking

        XCTAssertTrue(config.forceGPU)
        XCTAssertTrue(config.enableProfiling)
    }

}
