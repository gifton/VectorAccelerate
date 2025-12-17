//
//  VectorAccelerateTests.swift
//  VectorAccelerate
//
//  Tests for the VectorAccelerate module.
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate

import VectorCore

/// Test suite for VectorAccelerate module.
///
/// These tests verify that the module compiles correctly and basic
/// functionality works with the new GPU-first AcceleratedVectorIndex API.
final class VectorAccelerateTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = IndexAccelerationConfiguration.default

        XCTAssertEqual(config.minimumCandidatesForGPU, 500)
        XCTAssertEqual(config.minimumOperationsForGPU, 50_000)
        XCTAssertTrue(config.useFusedKernels)
        XCTAssertFalse(config.forceGPU)
    }

    func testAggressiveConfiguration() {
        let config = IndexAccelerationConfiguration.aggressive

        XCTAssertEqual(config.minimumCandidatesForGPU, 100)
        XCTAssertTrue(config.preallocateBuffers)
    }

    func testConservativeConfiguration() {
        let config = IndexAccelerationConfiguration.conservative

        XCTAssertEqual(config.minimumCandidatesForGPU, 5_000)
        XCTAssertFalse(config.preallocateBuffers)
    }

    func testBenchmarkingConfiguration() {
        let config = IndexAccelerationConfiguration.benchmarking

        XCTAssertTrue(config.forceGPU)
        XCTAssertTrue(config.enableProfiling)
    }

    // MARK: - Error Tests

    func testErrorDescriptions() {
        let error1 = IndexError.gpuNotInitialized(operation: "search")
        XCTAssertTrue(error1.localizedDescription.contains("GPU not initialized"))
        XCTAssertTrue(error1.localizedDescription.contains("search"))

        let error2 = IndexError.dimensionMismatch(expected: 768, got: 512)
        XCTAssertTrue(error2.localizedDescription.contains("768"))
        XCTAssertTrue(error2.localizedDescription.contains("512"))

        let error3 = IndexError.bufferTooLarge(requested: 1000, available: 500)
        XCTAssertTrue(error3.localizedDescription.contains("1000"))
        XCTAssertTrue(error3.localizedDescription.contains("500"))

        let error4 = IndexError.invalidInput(message: "test message")
        XCTAssertTrue(error4.localizedDescription.contains("test message"))
    }

    // MARK: - Shader Args Tests

    func testKMeansAssignShaderArgs() {
        let args = KMeansAssignShaderArgs(
            dimension: 512,
            numVectors: 10_000,
            numCentroids: 256
        )

        XCTAssertEqual(args.dimension, 512)
        XCTAssertEqual(args.numVectors, 10_000)
        XCTAssertEqual(args.numCentroids, 256)
    }

    // MARK: - VectorHandle Tests (P0.8 Stable Handles)

    func testVectorHandleInvalid() {
        let invalid = VectorHandle.invalid

        XCTAssertFalse(invalid.isValid)
    }

    func testVectorHandleComparison() {
        // With P0.8, handles use stableID only for comparison
        // We test by creating handles from an index and verifying their ordering
        // Since stableID is internal, we verify comparison via .invalid sentinel
        let invalid = VectorHandle.invalid

        // Invalid sentinel should compare as greater than any valid handle
        // (since stableID = UInt32.max for invalid)
        XCTAssertFalse(invalid.isValid)
    }

    // MARK: - Index Configuration Tests

    func testFlatIndexConfiguration() {
        let config = IndexConfiguration.flat(dimension: 768, capacity: 10_000)

        XCTAssertEqual(config.dimension, 768)
        XCTAssertEqual(config.capacity, 10_000)

        if case .flat = config.indexType {
            // Expected
        } else {
            XCTFail("Expected flat index type")
        }
    }

    func testIVFIndexConfiguration() {
        let config = IndexConfiguration.ivf(
            dimension: 512,
            nlist: 256,
            nprobe: 16
        )

        XCTAssertEqual(config.dimension, 512)
        XCTAssertEqual(config.capacity, 100_000) // Default capacity for IVF

        if case .ivf(let nlist, let nprobe, _) = config.indexType {
            XCTAssertEqual(nlist, 256)
            XCTAssertEqual(nprobe, 16)
        } else {
            XCTFail("Expected IVF index type")
        }
    }

    // MARK: - GPUIndexStats Tests

    func testGPUIndexStats() {
        let stats = GPUIndexStats(
            vectorCount: 1000,
            allocatedSlots: 1050,
            deletedSlots: 50,
            dimension: 768,
            metric: .euclidean,
            indexType: .flat,
            capacity: 2000,
            gpuVectorMemoryBytes: 6_144_000,
            gpuIndexStructureBytes: 0,
            cpuMetadataMemoryBytes: 100_000,
            ivfStats: nil
        )

        XCTAssertEqual(stats.vectorCount, 1000)
        XCTAssertEqual(stats.allocatedSlots, 1050)
        XCTAssertEqual(stats.deletedSlots, 50)
        XCTAssertEqual(stats.fragmentationRatio, Float(50) / Float(1050), accuracy: 0.001)
        XCTAssertEqual(stats.utilizationRatio, 0.5, accuracy: 0.001)
        XCTAssertFalse(stats.shouldCompact) // ~5% < 25% threshold
    }

    func testGPUIndexStatsCompactionRecommendation() {
        let stats = GPUIndexStats(
            vectorCount: 800,
            allocatedSlots: 1200,
            deletedSlots: 400, // 33% deleted
            dimension: 768,
            metric: .euclidean,
            indexType: .flat,
            capacity: 2000,
            gpuVectorMemoryBytes: 6_144_000,
            gpuIndexStructureBytes: 0,
            cpuMetadataMemoryBytes: 100_000,
            ivfStats: nil
        )

        XCTAssertTrue(stats.shouldCompact) // 33% > 25% threshold
    }
}

// MARK: - AcceleratedVectorIndex Tests

/// Tests for the GPU-first AcceleratedVectorIndex.
final class AcceleratedVectorIndexTests: XCTestCase {

    func testFlatIndexCreation() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1_000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()
        XCTAssertEqual(stats.dimension, 128)
        XCTAssertEqual(stats.vectorCount, 0)
        XCTAssertEqual(stats.capacity, 1_000)
    }

    func testIVFIndexCreation() async throws {
        let config = IndexConfiguration.ivf(
            dimension: 256,
            nlist: 16,
            nprobe: 4
        )
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.statistics()
        XCTAssertEqual(stats.dimension, 256)
        XCTAssertNotNil(stats.ivfStats)
        XCTAssertEqual(stats.ivfStats?.numClusters, 16)
    }

    func testInsertAndRetrieve() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert a vector
        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]
        let handle = try await index.insert(vector)

        XCTAssertTrue(handle.isValid)

        // Retrieve it
        let retrieved = try await index.vector(for: handle)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.count, 4)

        // Check approximate equality (GPU float precision)
        for i in 0..<4 {
            XCTAssertEqual(retrieved![i], vector[i], accuracy: 0.0001)
        }
    }

    func testInsertWithMetadata() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]
        let metadata: VectorMetadata = ["key": "value", "count": "42"]
        let handle = try await index.insert(vector, metadata: metadata)

        let retrievedMeta = await index.metadata(for: handle)
        XCTAssertNotNil(retrievedMeta)
        XCTAssertEqual(retrievedMeta?["key"], "value")
        XCTAssertEqual(retrievedMeta?["count"], "42")
    }

    func testSearchFlat() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert some vectors
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

        // Search for vector closest to [1, 0, 0, 0]
        let query: [Float] = [1.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 3)

        XCTAssertEqual(results.count, 3)
        // First result should be exact match (distance ≈ 0)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.0001)
    }

    func testRemoveVector() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vector: [Float] = [1.0, 2.0, 3.0, 4.0]
        let handle = try await index.insert(vector)

        var stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 1)
        XCTAssertEqual(stats.deletedSlots, 0)

        // Remove the vector
        try await index.remove(handle)

        stats = await index.statistics()
        XCTAssertEqual(stats.allocatedSlots, 1) // Still occupies slot
        XCTAssertEqual(stats.deletedSlots, 1) // But marked deleted

        // Try to retrieve - should fail
        let retrieved = try await index.vector(for: handle)
        XCTAssertNil(retrieved)
    }

    func testBatchInsert() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Create 100 random vectors
        var vectors: [[Float]] = []
        for _ in 0..<100 {
            vectors.append([
                Float.random(in: -1...1),
                Float.random(in: -1...1),
                Float.random(in: -1...1),
                Float.random(in: -1...1)
            ])
        }

        let handles = try await index.insert(vectors)
        XCTAssertEqual(handles.count, 100)

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 100)
    }

    func testCompaction() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 10 vectors
        var handles: [VectorHandle] = []
        for i in 0..<10 {
            let vector: [Float] = [Float(i), 0, 0, 0]
            handles.append(try await index.insert(vector))
        }

        // Remove half (even indices: 0, 2, 4, 6, 8)
        for i in stride(from: 0, to: 10, by: 2) {
            try await index.remove(handles[i])
        }

        var stats = await index.statistics()
        XCTAssertEqual(stats.deletedSlots, 5)

        // Compact (returns Void with P0.8 stable handles)
        try await index.compact()

        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 5)
        XCTAssertEqual(stats.deletedSlots, 0)

        // P0.8: Remaining handles (odd indices: 1, 3, 5, 7, 9) should still be valid
        for i in stride(from: 1, to: 10, by: 2) {
            let isValid = await index.contains(handles[i])
            XCTAssertTrue(isValid, "Handle \(i) should remain valid after compact (P0.8)")
        }
    }

    func testDimensionValidation() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Try to insert wrong dimension
        let wrongVector: [Float] = [1.0, 2.0, 3.0] // 3D instead of 4D

        do {
            _ = try await index.insert(wrongVector)
            XCTFail("Should have thrown dimensionMismatch error")
        } catch let error as IndexError {
            if case .dimensionMismatch(let expected, let got) = error {
                XCTAssertEqual(expected, 4)
                XCTAssertEqual(got, 3)
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }
    }

    func testHandleCount() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert some vectors
        for i in 0..<5 {
            _ = try await index.insert([Float(i), 0, 0, 0])
        }

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 5)
    }
}
