//
//  EmbeddingEngineTests.swift
//  VectorAccelerateTests
//
//  Comprehensive tests for EmbeddingEngine ML operations
//

import XCTest
@testable import VectorAccelerate
import Foundation
import VectorCore

final class EmbeddingEngineTests: XCTestCase {
    
    var engine: EmbeddingEngine!
    var testEmbeddings: [[Float]]!
    var testMetadata: [[String: String]]!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Generate test embeddings (3D for simplicity but extensible)
        testEmbeddings = generateTestEmbeddings(count: 100, dimension: 128)
        testMetadata = generateTestMetadata(count: 100)
        
        let config = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .cosine,
            useGPU: MetalDevice.isAvailable,
            batchSize: 32,
            normalizeEmbeddings: true
        )
        
        engine = try await EmbeddingEngine(configuration: config)
    }
    
    override func tearDown() async throws {
        await engine?.clearEmbeddings()
        engine = nil
        testEmbeddings = nil
        testMetadata = nil
        try await super.tearDown()
    }
    
    // MARK: - Test Utilities
    
    private func generateTestEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        return (0..<count).map { i in
            return (0..<dimension).map { j in
                // Create some structure in the data for meaningful tests
                let base = Float(i) / Float(count)
                let variation = Float.random(in: -0.1...0.1)
                return sin(Float(j) * base * Float.pi) + variation
            }
        }
    }
    
    private func generateTestMetadata(count: Int) -> [[String: String]] {
        return (0..<count).map { i in
            return [
                "id": "embedding_\(i)",
                "category": "test_\(i % 5)",
                "index": "\(i)"
            ]
        }
    }
    
    private func createIdenticalEmbedding(_ original: [Float]) -> [Float] {
        return original.map { $0 }
    }
    
    private func createSimilarEmbedding(_ original: [Float], noise: Float = 0.01) -> [Float] {
        return original.map { $0 + Float.random(in: -noise...noise) }
    }
    
    // MARK: - Initialization Tests
    
    func testInitializationWithDefaultConfig() async throws {
        let config = EmbeddingConfiguration(dimension: 256)
        let testEngine = try await EmbeddingEngine(configuration: config)
        
        let metrics = await testEngine.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 0)
        XCTAssertEqual(metrics.searches, 0)
        XCTAssertEqual(metrics.averageSearchTime, 0)
    }
    
    func testInitializationWithCustomConfig() async throws {
        let config = EmbeddingConfiguration(
            dimension: 512,
            distanceMetric: .euclidean,
            useGPU: false,
            batchSize: 64,
            normalizeEmbeddings: false
        )
        
        let testEngine = try await EmbeddingEngine(configuration: config)
        XCTAssertNotNil(testEngine)
    }
    
    func testInitializationWithGPUContext() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await MetalContext()
        let config = EmbeddingConfiguration(dimension: 256, useGPU: true)
        let testEngine = try await EmbeddingEngine(configuration: config, context: context)
        
        XCTAssertNotNil(testEngine)
    }
    
    // MARK: - Embedding Management Tests
    
    func testAddEmbeddings() async throws {
        await engine!.clearEmbeddings()
        
        let initialMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(initialMetrics.embeddings, 0)
        
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let afterMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(afterMetrics.embeddings, 10)
    }
    
    func testAddEmbeddingsWithMetadata() async throws {
        await engine!.clearEmbeddings()
        
        try await engine!.addEmbeddings(
            Array(testEmbeddings[0..<5]),
            metadata: Array(testMetadata[0..<5])
        )
        
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 5)
        
        // Test search includes metadata
        let results = try await engine!.search(query: testEmbeddings[0], k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertNotNil(results[0].metadata)
        XCTAssertEqual(results[0].metadata?["id"], "embedding_0")
    }
    
    func testAddEmbeddingsWithDimensionMismatch() async throws {
        let wrongDimensionEmbedding = [[Float]](repeating: [1.0, 2.0], count: 1)
        
        do {
            try await engine!.addEmbeddings(wrongDimensionEmbedding)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
        }
    }
    
    func testClearEmbeddings() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        var metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 10)
        
        await engine!.clearEmbeddings()
        
        metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 0)
    }
    
    func testMultipleAddOperations() async throws {
        await engine!.clearEmbeddings()
        
        // Add in chunks
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<30]))
        try await engine!.addEmbeddings(Array(testEmbeddings[30..<60]))
        try await engine!.addEmbeddings(Array(testEmbeddings[60..<100]))
        
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 100)
    }
    
    // MARK: - Similarity Search Tests
    
    func testBasicSearch() async throws {
        try await engine!.addEmbeddings(testEmbeddings, metadata: testMetadata)
        
        let query = testEmbeddings[0]  // Search for first embedding
        let results = try await engine!.search(query: query, k: 5)
        
        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].index, 0)  // Should find itself first
        XCTAssertLessThan(results[0].distance, 0.01)  // Very close to zero
        
        // Results should be sorted by distance
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance)
        }
    }
    
    func testSearchWithThreshold() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let query = testEmbeddings[0]
        let results = try await engine!.search(query: query, k: 50, threshold: 0.5)
        
        // All results should be within threshold
        for result in results {
            XCTAssertLessThanOrEqual(result.distance, 0.5)
        }
        
        // Should have fewer results than k due to threshold
        XCTAssertLessThanOrEqual(results.count, 50)
    }
    
    func testSearchWithDimensionMismatch() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let wrongQuery = [Float](repeating: 1.0, count: 64)  // Wrong dimension
        
        do {
            _ = try await engine!.search(query: wrongQuery, k: 5)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
        }
    }
    
    func testSearchEmptyDatabase() async throws {
        await engine!.clearEmbeddings()
        
        let query = Array(repeating: Float(1.0), count: 128)
        let results = try await engine!.search(query: query, k: 5)
        
        XCTAssertEqual(results.count, 0)
    }
    
    func testSearchExactMatch() async throws {
        // Create test data with an exact duplicate
        let exactMatch = testEmbeddings[0]
        var modifiedEmbeddings = testEmbeddings!
        modifiedEmbeddings.append(createIdenticalEmbedding(exactMatch))
        
        try await engine!.addEmbeddings(modifiedEmbeddings)
        
        let results = try await engine!.search(query: exactMatch, k: 3)
        
        XCTAssertGreaterThan(results.count, 1)
        // First result should be exact match (distance ~0)
        XCTAssertLessThan(results[0].distance, 1e-6)
        // Should find the duplicate too
        XCTAssertLessThan(results[1].distance, 1e-6)
    }
    
    func testSearchWithNormalizedVectors() async throws {
        // Test with vectors that need normalization
        let unnormalizedEmbeddings = testEmbeddings.map { embedding in
            embedding.map { $0 * 10.0 }  // Scale up
        }
        
        try await engine!.addEmbeddings(unnormalizedEmbeddings)
        
        let query = unnormalizedEmbeddings[0]
        let results = try await engine!.search(query: query, k: 1)
        
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].index, 0)
        XCTAssertLessThan(results[0].distance, 0.01)  // Should still find exact match
    }
    
    // MARK: - Batch Search Tests
    
    func testBatchSearch() async throws {
        try await engine!.addEmbeddings(testEmbeddings, metadata: testMetadata)
        
        let queries = Array(testEmbeddings[0..<5])
        let batchResults = try await engine!.batchSearch(queries: queries, k: 3)
        
        XCTAssertEqual(batchResults.count, 5)
        
        for (i, results) in batchResults.enumerated() {
            XCTAssertEqual(results.count, 3)
            // First result should be the query itself
            XCTAssertEqual(results[0].index, i)
            XCTAssertLessThan(results[0].distance, 0.01)
        }
    }
    
    func testBatchSearchWithThreshold() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let queries = Array(testEmbeddings[0..<3])
        let batchResults = try await engine!.batchSearch(queries: queries, k: 10, threshold: 0.3)
        
        XCTAssertEqual(batchResults.count, 3)
        
        for results in batchResults {
            // All results should be within threshold
            for result in results {
                XCTAssertLessThanOrEqual(result.distance, 0.3)
            }
        }
    }
    
    func testEmptyBatchSearch() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let batchResults = try await engine!.batchSearch(queries: [], k: 5)
        XCTAssertEqual(batchResults.count, 0)
    }
    
    // MARK: - K-Means Clustering Tests
    
    func testKMeansClustering() async throws {
        // Create more structured data for clustering
        let clusterData = createClusteredEmbeddings(clusters: 3, pointsPerCluster: 20, dimension: 128)
        try await engine!.addEmbeddings(clusterData)
        
        let result = try await engine!.kMeansClustering(k: 3, maxIterations: 50)
        
        XCTAssertEqual(result.clusterAssignments.count, 60)
        XCTAssertEqual(result.centroids.count, 3)
        XCTAssertEqual(result.centroids[0].count, 128)
        XCTAssertGreaterThan(result.inertia, 0)
        XCTAssertLessThanOrEqual(result.iterations, 50)
        
        // Check that each point is assigned to a valid cluster
        for assignment in result.clusterAssignments {
            XCTAssertGreaterThanOrEqual(assignment, 0)
            XCTAssertLessThan(assignment, 3)
        }
    }
    
    func testKMeansWithSingleCluster() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let result = try await engine!.kMeansClustering(k: 1, maxIterations: 10)
        
        XCTAssertEqual(result.centroids.count, 1)
        XCTAssertEqual(result.clusterAssignments.count, 10)
        
        // All points should be assigned to cluster 0
        for assignment in result.clusterAssignments {
            XCTAssertEqual(assignment, 0)
        }
    }
    
    func testKMeansConvergence() async throws {
        // Create data that should converge quickly
        let simpleData = [[Float]](repeating: Array(repeating: 1.0, count: 128), count: 10)
        try await engine!.addEmbeddings(simpleData)
        
        let result = try await engine!.kMeansClustering(k: 1, maxIterations: 100, tolerance: 1e-6)
        
        // Should converge quickly for identical points
        XCTAssertLessThan(result.iterations, 10)
        XCTAssertLessThan(result.inertia, 1e-3)
    }
    
    func testKMeansWithEmptyDatabase() async throws {
        await engine!.clearEmbeddings()
        
        do {
            _ = try await engine!.kMeansClustering(k: 3)
            XCTFail("Should have thrown error for empty database")
        } catch let error as VectorError where error.kind == .invalidOperation || error.kind == .unsupportedOperation {
            // Expected
        }
    }
    
    func testKMeansWithMoreClustersThanPoints() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<3]))
        
        // This should still work, but some centroids may be duplicated
        let result = try await engine!.kMeansClustering(k: 5)
        
        XCTAssertEqual(result.centroids.count, 5)
        XCTAssertEqual(result.clusterAssignments.count, 3)
    }
    
    // MARK: - Performance and Metrics Tests
    
    func testPerformanceMetrics() async throws {
        let initialMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(initialMetrics.embeddings, 0)
        XCTAssertEqual(initialMetrics.searches, 0)
        XCTAssertEqual(initialMetrics.averageSearchTime, 0)
        
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<20]))
        
        let afterAddMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(afterAddMetrics.embeddings, 20)
        XCTAssertEqual(afterAddMetrics.searches, 0)
        
        // Perform some searches
        let query = testEmbeddings[0]
        _ = try await engine!.search(query: query, k: 5)
        _ = try await engine!.search(query: query, k: 3)
        
        let finalMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(finalMetrics.embeddings, 20)
        XCTAssertEqual(finalMetrics.searches, 2)
        XCTAssertGreaterThan(finalMetrics.averageSearchTime, 0)
    }
    
    func testSearchPerformanceScaling() async throws {
        // Test with different dataset sizes
        let sizes = [100, 500, 1000]
        var searchTimes: [TimeInterval] = []
        
        for size in sizes {
            await engine!.clearEmbeddings()
            
            let data = generateTestEmbeddings(count: size, dimension: 128)
            try await engine!.addEmbeddings(data)
            
            let query = data[0]
            let startTime = CFAbsoluteTimeGetCurrent()
            
            _ = try await engine!.search(query: query, k: 10)
            
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            searchTimes.append(elapsed)
            
            print("Search time for \(size) embeddings: \(elapsed)s")
        }
        
        // Search times should be reasonable (under 1 second for these sizes)
        for time in searchTimes {
            XCTAssertLessThan(time, 1.0)
        }
    }
    
    // MARK: - Different Distance Metrics Tests
    
    func testEuclideanDistanceMetric() async throws {
        let config = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .euclidean,
            useGPU: false,
            normalizeEmbeddings: false
        )
        
        let euclideanEngine = try await EmbeddingEngine(configuration: config)
        try await euclideanEngine.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let query = testEmbeddings[0]
        let results = try await euclideanEngine.search(query: query, k: 3)
        
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].index, 0)
        XCTAssertLessThan(results[0].distance, 0.01)
    }
    
    func testManhattanDistanceMetric() async throws {
        let config = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .manhattan,
            useGPU: false,
            normalizeEmbeddings: false
        )
        
        let manhattanEngine = try await EmbeddingEngine(configuration: config)
        try await manhattanEngine.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let query = testEmbeddings[0]
        let results = try await manhattanEngine.search(query: query, k: 3)
        
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].index, 0)
        XCTAssertLessThan(results[0].distance, 0.01)
    }
    
    // MARK: - GPU vs CPU Tests
    
    func testGPUvsCPUConsistency() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let testData = Array(testEmbeddings[0..<50])
        let query = testEmbeddings[0]
        
        // CPU engine
        let cpuConfig = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .cosine,
            useGPU: false
        )
        let cpuEngine = try await EmbeddingEngine(configuration: cpuConfig)
        try await cpuEngine.addEmbeddings(testData)
        
        // GPU engine
        let gpuConfig = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .cosine,
            useGPU: true
        )
        let gpuEngine = try await EmbeddingEngine(configuration: gpuConfig)
        try await gpuEngine.addEmbeddings(testData)
        
        let cpuResults = try await cpuEngine.search(query: query, k: 10)
        let gpuResults = try await gpuEngine.search(query: query, k: 10)
        
        XCTAssertEqual(cpuResults.count, gpuResults.count)
        
        // Results should be very similar (allowing for floating point differences)
        for (cpuResult, gpuResult) in zip(cpuResults, gpuResults) {
            XCTAssertEqual(cpuResult.index, gpuResult.index)
            XCTAssertEqual(cpuResult.distance, gpuResult.distance, accuracy: 1e-4)
        }
    }
    
    // MARK: - Concurrent Access Tests
    
    func testConcurrentSearches() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let queries = Array(testEmbeddings[0..<10])
        
        // Perform concurrent searches
        await withTaskGroup(of: Void.self) { group in
            for query in queries {
                group.addTask { [engine] in
                    do {
                        let results = try await engine!.search(query: query, k: 5)
                        XCTAssertGreaterThan(results.count, 0)
                    } catch {
                        XCTFail("Concurrent search failed: \(error)")
                    }
                }
            }
        }
        
        // Engine should still be functional
        let finalMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(finalMetrics.embeddings, 100)
        XCTAssertGreaterThanOrEqual(finalMetrics.searches, 10)
    }
    
    func testConcurrentAddAndSearch() async throws {
        await engine!.clearEmbeddings()
        
        // Add embeddings concurrently with searches
        await withTaskGroup(of: Void.self) { group in
            // Add task
            group.addTask { [engine, testEmbeddings] in
                do {
                    for chunk in testEmbeddings!.chunked(into: 10) {
                        try await engine!.addEmbeddings(chunk)
                        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                    }
                } catch {
                    print("Add failed: \(error)")
                }
            }
            
            // Search tasks
            for i in 0..<5 {
                group.addTask { [engine] in
                    do {
                        try await Task.sleep(nanoseconds: 50_000_000) // Wait for some data
                        let query = Array(repeating: Float.random(in: -1...1), count: 128)
                        let results = try await engine!.search(query: query, k: 3)
                        print("Search \(i): \(results.count) results")
                    } catch {
                        print("Search \(i) failed: \(error)")
                    }
                }
            }
        }
    }
    
    // MARK: - Edge Case Tests
    
    func testSearchWithZeroK() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let query = testEmbeddings[0]
        let results = try await engine!.search(query: query, k: 0)
        
        XCTAssertEqual(results.count, 0)
    }
    
    func testSearchWithVeryHighK() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        let query = testEmbeddings[0]
        let results = try await engine!.search(query: query, k: 100)
        
        // Should return at most the number of embeddings available
        XCTAssertEqual(results.count, 10)
    }
    
    func testSearchWithZeroVectors() async throws {
        let zeroVector = Array(repeating: Float(0.0), count: 128)
        try await engine!.addEmbeddings([zeroVector])
        
        let results = try await engine!.search(query: zeroVector, k: 1)
        
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].index, 0)
        // For cosine distance with zero vectors that can't be normalized,
        // the distance is undefined but typically returns 1.0 (max distance)
        // or 0.0 if special-cased. We just verify it doesn't crash.
        XCTAssertTrue(results[0].distance >= 0.0 && results[0].distance <= 1.0,
                     "Distance should be valid for zero vectors")
    }
    
    func testNormalizationEdgeCases() async throws {
        // Test with very small vectors that might cause normalization issues
        let smallVectors = [
            Array(repeating: Float(1e-8), count: 128),
            Array(repeating: Float(0.0), count: 128)
        ]
        
        let config = EmbeddingConfiguration(
            dimension: 128,
            useGPU: false,
            normalizeEmbeddings: true
        )
        
        let testEngine = try await EmbeddingEngine(configuration: config)
        try await testEngine.addEmbeddings(smallVectors)
        
        let query = Array(repeating: Float(1e-8), count: 128)
        let results = try await testEngine.search(query: query, k: 2)
        
        XCTAssertEqual(results.count, 2)
        // Should handle normalization gracefully without crashing
    }
    
    // MARK: - Helper Functions for Testing
    
    private func createClusteredEmbeddings(clusters: Int, pointsPerCluster: Int, dimension: Int) -> [[Float]] {
        var embeddings: [[Float]] = []
        
        for _ in 0..<clusters {
            // Create a base centroid for this cluster
            let centroid = (0..<dimension).map { _ in Float.random(in: -1...1) }
            
            // Generate points around this centroid
            for _ in 0..<pointsPerCluster {
                let point = centroid.map { base in
                    base + Float.random(in: -0.2...0.2)
                }
                embeddings.append(point)
            }
        }
        
        return embeddings
    }
}

// MARK: - Array Extension for Testing

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}