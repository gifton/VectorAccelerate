//
//  AcceleratedIndexIntegrationTests.swift
//  VectorIndexAccelerationTests
//
//  Integration tests for GPU-accelerated index wrappers.
//  Tests end-to-end functionality and correctness vs CPU baselines.
//

import XCTest
import VectorIndex
import VectorCore
@testable import VectorAccelerate
@testable import VectorIndexAcceleration

/// Helper to generate deterministic test vectors
func generateTestVectors(count: Int, dimension: Int, seed: Int = 42) -> [[Float]] {
    var vectors: [[Float]] = []
    vectors.reserveCapacity(count)
    for i in 0..<count {
        var vec: [Float] = []
        vec.reserveCapacity(dimension)
        for d in 0..<dimension {
            // Deterministic pseudo-random values
            let value = Float(((i * 17 + d * 31 + seed) % 1000)) / 500.0 - 1.0
            vec.append(value)
        }
        vectors.append(vec)
    }
    return vectors
}

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class FlatIndexAcceleratedIntegrationTests: XCTestCase {

    // MARK: - Test Data

    let dimension = 32
    let vectorCount = 100
    let k = 5

    // MARK: - Basic Functionality Tests

    func testBasicSearchGPU() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        // Insert vectors
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        // Create accelerated wrapper
        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )

        // Prepare for GPU
        try await accelerated.prepareForGPU()

        // Search
        let queryVector: [Float] = vectors[0]
        let results = try await accelerated.searchGPU(query: queryVector, k: k)

        // Verify results
        XCTAssertEqual(results.count, k, "Should return k results")
        XCTAssertEqual(results[0].id, "vec_0", "First result should be the query vector itself")
        XCTAssertEqual(results[0].score, 0, accuracy: 0.001, "Distance to self should be ~0")
    }

    func testGPUMatchesCPU() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        // Insert vectors
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        // Create accelerated wrapper with forceGPU
        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Query with a different vector
        let queryVector: [Float] = generateTestVectors(count: 1, dimension: dimension, seed: 99)[0]

        // Get CPU results
        let cpuResults = try await baseIndex.search(query: queryVector, k: k, filter: nil)

        // Get GPU results
        let gpuResults = try await accelerated.searchGPU(query: queryVector, k: k)

        // Compare results - IDs should match
        XCTAssertEqual(cpuResults.count, gpuResults.count, "Should have same number of results")
        for i in 0..<min(cpuResults.count, gpuResults.count) {
            XCTAssertEqual(cpuResults[i].id, gpuResults[i].id, "Result \(i) ID should match")
            XCTAssertEqual(cpuResults[i].score, gpuResults[i].score, accuracy: 0.01, "Result \(i) score should be close")
        }
    }

    func testBatchSearchGPU() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        // Insert vectors
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Batch search
        let queryVectors = generateTestVectors(count: 5, dimension: dimension, seed: 123)
        let results = try await accelerated.batchSearchGPU(queries: queryVectors, k: k)

        XCTAssertEqual(results.count, 5, "Should return results for each query")
        for queryResults in results {
            XCTAssertEqual(queryResults.count, k, "Each query should return k results")
        }
    }

    func testInsertInvalidatesCache() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        // Initial vectors
        let vectors = generateTestVectors(count: 10, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Insert new vector through accelerated wrapper
        let newVec: [Float] = generateTestVectors(count: 1, dimension: dimension, seed: 999)[0]
        try await accelerated.insert(id: "new_vec", vector: newVec)

        // Search should find the new vector
        let results = try await accelerated.searchGPU(query: newVec, k: 1)
        XCTAssertEqual(results[0].id, "new_vec", "New vector should be found")
    }

    func testFilterSupport() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        // Insert vectors with metadata
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let metadata = ["category": i % 2 == 0 ? "even" : "odd"]
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: metadata)
        }

        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Search with filter for "even" category
        let queryVector: [Float] = vectors[0]
        let filter: @Sendable ([String: String]?) -> Bool = { meta in
            meta?["category"] == "even"
        }

        let results = try await accelerated.searchGPU(query: queryVector, k: k, filter: filter)

        // All results should have "even" category
        for result in results {
            // The ID should be an even-indexed vector
            let idNum = Int(result.id.replacingOccurrences(of: "vec_", with: ""))!
            XCTAssertEqual(idNum % 2, 0, "Filtered results should only include even vectors")
        }
    }

    func testEmptyIndex() async throws {
        let baseIndex = FlatIndex(dimension: dimension, metric: .euclidean)

        let accelerated = FlatIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        let queryVector: [Float] = generateTestVectors(count: 1, dimension: dimension)[0]
        let results = try await accelerated.searchGPU(query: queryVector, k: k)

        XCTAssertTrue(results.isEmpty, "Empty index should return empty results")
    }
}

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class HNSWIndexAcceleratedIntegrationTests: XCTestCase {

    let dimension = 32
    let vectorCount = 100
    let k = 5

    func testBasicSearchGPU() async throws {
        let baseIndex = HNSWIndex(dimension: dimension, metric: .euclidean)

        // Insert vectors
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        let accelerated = HNSWIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Search
        let queryVector: [Float] = vectors[0]
        let results = try await accelerated.searchGPU(query: queryVector, k: k)

        XCTAssertEqual(results.count, k, "Should return k results")
        // Note: HNSW is approximate, so first result might not always be exact match
        XCTAssertTrue(results.contains { $0.id == "vec_0" }, "Results should contain the query vector")
    }

    func testBatchSearch() async throws {
        let baseIndex = HNSWIndex(dimension: dimension, metric: .euclidean)

        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        let accelerated = HNSWIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Batch search
        let queryVectors = generateTestVectors(count: 5, dimension: dimension, seed: 123)
        let results = try await accelerated.batchSearch(queries: queryVectors, k: k)

        XCTAssertEqual(results.count, 5, "Should return results for each query")
        for queryResults in results {
            XCTAssertLessThanOrEqual(queryResults.count, k, "Each query should return at most k results")
        }
    }

    func testDelegatedOperations() async throws {
        let baseIndex = HNSWIndex(dimension: dimension, metric: .euclidean)

        let accelerated = HNSWIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .default
        )

        // Insert through accelerated
        let vec: [Float] = generateTestVectors(count: 1, dimension: dimension)[0]
        try await accelerated.insert(id: "test", vector: vec)

        // Verify in base index
        let contains = await accelerated.contains(id: "test")
        XCTAssertTrue(contains)

        let count = await accelerated.count
        XCTAssertEqual(count, 1)

        // Remove through accelerated
        try await accelerated.remove(id: "test")
        let stillContains = await accelerated.contains(id: "test")
        XCTAssertFalse(stillContains)
    }
}

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFIndexAcceleratedIntegrationTests: XCTestCase {

    let dimension = 32
    let vectorCount = 100
    let k = 5

    func testBasicSearchGPU() async throws {
        let config = IVFIndex.Configuration(nlist: 8, nprobe: 4)
        let baseIndex = IVFIndex(dimension: dimension, metric: .euclidean, config: config)

        // Insert vectors
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }

        // Optimize to build centroids
        try await baseIndex.optimize()

        let accelerated = IVFIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Search
        let queryVector: [Float] = vectors[0]
        let results = try await accelerated.searchGPU(query: queryVector, k: k)

        XCTAssertGreaterThan(results.count, 0, "Should return results")
        XCTAssertLessThanOrEqual(results.count, k, "Should return at most k results")
    }

    func testGPUMatchesCPU() async throws {
        let config = IVFIndex.Configuration(nlist: 8, nprobe: 4)
        let baseIndex = IVFIndex(dimension: dimension, metric: .euclidean, config: config)

        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }
        try await baseIndex.optimize()

        let accelerated = IVFIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        let queryVector: [Float] = generateTestVectors(count: 1, dimension: dimension, seed: 99)[0]

        // Get CPU and GPU results
        let cpuResults = try await baseIndex.search(query: queryVector, k: k, filter: nil)
        let gpuResults = try await accelerated.searchGPU(query: queryVector, k: k)

        // Both should return results
        XCTAssertGreaterThan(cpuResults.count, 0, "CPU should return results")
        XCTAssertGreaterThan(gpuResults.count, 0, "GPU should return results")

        // Check that top result is similar (IVF is approximate)
        if !cpuResults.isEmpty && !gpuResults.isEmpty {
            // At least verify distances are in similar range
            let cpuTopDist = cpuResults[0].score
            let gpuTopDist = gpuResults[0].score
            XCTAssertEqual(cpuTopDist, gpuTopDist, accuracy: cpuTopDist * 0.5, "Top distances should be similar")
        }
    }

    func testDelegatedOperations() async throws {
        let baseIndex = IVFIndex(dimension: dimension, metric: .euclidean)

        let accelerated = IVFIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .default
        )

        // Insert through accelerated
        let vec: [Float] = generateTestVectors(count: 1, dimension: dimension)[0]
        try await accelerated.insert(id: "test", vector: vec)

        // Verify
        let contains = await accelerated.contains(id: "test")
        XCTAssertTrue(contains)

        let count = await accelerated.count
        XCTAssertEqual(count, 1)
    }

    func testBatchSearchGPU() async throws {
        let config = IVFIndex.Configuration(nlist: 8, nprobe: 4)
        let baseIndex = IVFIndex(dimension: dimension, metric: .euclidean, config: config)

        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await baseIndex.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }
        try await baseIndex.optimize()

        let accelerated = IVFIndexAccelerated(
            baseIndex: baseIndex,
            configuration: .benchmarking
        )
        try await accelerated.prepareForGPU()

        // Batch search
        let queryVectors = generateTestVectors(count: 5, dimension: dimension, seed: 123)
        let results = try await accelerated.batchSearchGPU(queries: queryVectors, k: k)

        XCTAssertEqual(results.count, 5, "Should return results for each query")
    }
}

// MARK: - Cross-Index Consistency Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class CrossIndexConsistencyTests: XCTestCase {

    let dimension = 16
    let vectorCount = 50
    let k = 3

    func testAllIndexTypesReturnSameTopResult() async throws {
        // Create the same dataset for all index types
        let vectors = generateTestVectors(count: vectorCount, dimension: dimension)
        let queryVector: [Float] = vectors[0]  // Query with first vector

        // Flat Index
        let flatBase = FlatIndex(dimension: dimension, metric: .euclidean)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await flatBase.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }
        let flatAccel = FlatIndexAccelerated(baseIndex: flatBase, configuration: .benchmarking)
        try await flatAccel.prepareForGPU()
        let flatResults = try await flatAccel.searchGPU(query: queryVector, k: k)

        // HNSW Index
        let hnswBase = HNSWIndex(dimension: dimension, metric: .euclidean)
        for (i, vec) in vectors.enumerated() {
            let vecArray: [Float] = vec
            try await hnswBase.insert(id: "vec_\(i)", vector: vecArray, metadata: nil)
        }
        let hnswAccel = HNSWIndexAccelerated(baseIndex: hnswBase, configuration: .benchmarking)
        try await hnswAccel.prepareForGPU()
        let hnswResults = try await hnswAccel.searchGPU(query: queryVector, k: k)

        // Verify Flat finds exact match (it's exact search)
        XCTAssertEqual(flatResults[0].id, "vec_0", "Flat should find exact match")
        XCTAssertEqual(flatResults[0].score, 0, accuracy: 0.001, "Flat distance to self should be 0")

        // HNSW should also find the vector (it's approximate but should be accurate for self)
        XCTAssertTrue(hnswResults.contains { $0.id == "vec_0" }, "HNSW should find the query vector")
    }
}
