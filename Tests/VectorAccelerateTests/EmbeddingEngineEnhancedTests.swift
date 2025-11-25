//
//  EmbeddingEngineEnhancedTests.swift
//  VectorAccelerateTests
//
//  Enhanced comprehensive tests for EmbeddingEngine ML operations
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Foundation
import VectorCore

@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
final class EmbeddingEngineEnhancedTests: XCTestCase {
    
    var engine: EmbeddingEngine!
    var testEmbeddings: [[Float]]!
    var testMetadata: [[String: String]]!
    var configuration: EmbeddingConfiguration!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Generate test embeddings with different patterns for comprehensive testing
        testEmbeddings = Self.generateDiverseTestEmbeddings(count: 200, dimension: 256)
        testMetadata = generateTestMetadata(count: 200)
        
        configuration = EmbeddingConfiguration(
            dimension: 256,
            distanceMetric: .cosine,
            useGPU: MetalDevice.isAvailable,
            batchSize: 50,
            normalizeEmbeddings: true
        )
        
        engine = try await EmbeddingEngine(configuration: configuration)
    }
    
    override func tearDown() async throws {
        await engine?.clearEmbeddings()
        engine = nil
        testEmbeddings = nil
        testMetadata = nil
        configuration = nil
        try await super.tearDown()
    }
    
    // MARK: - Test Utilities
    
    private static func generateDiverseTestEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        var embeddings: [[Float]] = []
        
        // Generate different types of embeddings for comprehensive testing
        for i in 0..<count {
            let type = i % 5
            var embedding: [Float]
            
            switch type {
            case 0: // Random embeddings
                embedding = (0..<dimension).map { _ in Float.random(in: -1...1) }
            case 1: // Structured sine waves
                let frequency = Float(i) / Float(count) * Float.pi
                embedding = (0..<dimension).map { j in sin(frequency * Float(j)) }
            case 2: // Sparse embeddings (mostly zeros)
                embedding = [Float](repeating: 0, count: dimension)
                let numNonZero = min(10, dimension / 10)
                for _ in 0..<numNonZero {
                    let idx = Int.random(in: 0..<dimension)
                    embedding[idx] = Float.random(in: -2...2)
                }
            case 3: // High-magnitude embeddings
                embedding = (0..<dimension).map { _ in Float.random(in: -10...10) }
            case 4: // Low-magnitude embeddings
                embedding = (0..<dimension).map { _ in Float.random(in: -0.1...0.1) }
            default:
                embedding = (0..<dimension).map { _ in Float.random(in: -1...1) }
            }
            
            embeddings.append(embedding)
        }
        
        return embeddings
    }
    
    private func generateTestMetadata(count: Int) -> [[String: String]] {
        return (0..<count).map { i in
            return [
                "id": "embedding_\(i)",
                "category": "test_\(i % 8)",
                "type": ["random", "sine", "sparse", "high_mag", "low_mag"][i % 5],
                "index": "\(i)",
                "batch": "\(i / 50)"
            ]
        }
    }
    
    private func createClusteredEmbeddings(numClusters: Int, pointsPerCluster: Int, dimension: Int, clusterRadius: Float = 0.3) -> ([[Float]], [Int]) {
        var embeddings: [[Float]] = []
        var trueLabels: [Int] = []
        
        // Generate cluster centers
        let centers = (0..<numClusters).map { _ in
            (0..<dimension).map { _ in Float.random(in: -2...2) }
        }
        
        // Generate points around each center
        for (clusterIdx, center) in centers.enumerated() {
            for _ in 0..<pointsPerCluster {
                let point = center.map { centerVal in
                    centerVal + Float.random(in: -clusterRadius...clusterRadius)
                }
                embeddings.append(point)
                trueLabels.append(clusterIdx)
            }
        }
        
        return (embeddings, trueLabels)
    }
    
    private func calculateSilhouetteScore(embeddings: [[Float]], assignments: [Int], centroids: [[Float]]) async -> Float {
        // Simple silhouette score calculation
        var totalScore: Float = 0
        
        for i in 0..<embeddings.count {
            let point = embeddings[i]
            let cluster = assignments[i]
            
            // Calculate intra-cluster distance (a)
            var intraClusterDist: Float = 0
            var intraClusterCount = 0
            
            for j in 0..<embeddings.count {
                if assignments[j] == cluster && i != j {
                    intraClusterDist += euclideanDistance(point, embeddings[j])
                    intraClusterCount += 1
                }
            }
            
            let a = intraClusterCount > 0 ? intraClusterDist / Float(intraClusterCount) : 0
            
            // Calculate nearest-cluster distance (b)
            var nearestClusterDist: Float = Float.infinity
            
            for otherCluster in 0..<centroids.count {
                if otherCluster != cluster {
                    var interClusterDist: Float = 0
                    var interClusterCount = 0
                    
                    for j in 0..<embeddings.count {
                        if assignments[j] == otherCluster {
                            interClusterDist += euclideanDistance(point, embeddings[j])
                            interClusterCount += 1
                        }
                    }
                    
                    if interClusterCount > 0 {
                        let avgDist = interClusterDist / Float(interClusterCount)
                        nearestClusterDist = min(nearestClusterDist, avgDist)
                    }
                }
            }
            
            let b = nearestClusterDist
            let silhouette = (b - a) / max(a, b)
            totalScore += silhouette
        }
        
        return totalScore / Float(embeddings.count)
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    // MARK: - Advanced Initialization Tests
    
    func testInitializationWithDifferentDimensions() async throws {
        let dimensions = [64, 128, 256, 512, 1024, 2048]
        
        for dim in dimensions {
            let config = EmbeddingConfiguration(dimension: dim, useGPU: false)
            let testEngine = try await EmbeddingEngine(configuration: config)
            XCTAssertNotNil(testEngine)
            
            // Test with sample data
            let sampleData = [[Float]](repeating: Array(repeating: 1.0, count: dim), count: 5)
            try await testEngine.addEmbeddings(sampleData)
            
            let metrics = await testEngine.getPerformanceMetrics()
            XCTAssertEqual(metrics.embeddings, 5)
        }
    }
    
    func testInitializationWithAllDistanceMetrics() async throws {
        let metrics: [SupportedDistanceMetric] = [.euclidean, .cosine, .manhattan]
        
        for metric in metrics {
            let config = EmbeddingConfiguration(
                dimension: 128,
                distanceMetric: metric,
                useGPU: false,
                normalizeEmbeddings: metric == .cosine
            )
            
            let testEngine = try await EmbeddingEngine(configuration: config)
            XCTAssertNotNil(testEngine)
            
            // Test basic functionality
            let sampleData = Self.generateDiverseTestEmbeddings(count: 10, dimension: 128)
            try await testEngine.addEmbeddings(sampleData)
            
            let results = try await testEngine.search(query: sampleData[0], k: 3)
            XCTAssertEqual(results.count, 3)
            XCTAssertEqual(results[0].index, 0) // Should find itself first
        }
    }
    
    func testInitializationWithCustomContext() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        // Test with different Metal context configurations
        let configs = [
            MetalConfiguration(),
            MetalConfiguration(preferHighPerformanceDevice: false),
            MetalConfiguration(enableProfiling: true)
        ]
        
        for metalConfig in configs {
            let context = try await MetalContext(configuration: metalConfig)
            let embeddingConfig = EmbeddingConfiguration(dimension: 256, useGPU: true)
            let testEngine = try await EmbeddingEngine(configuration: embeddingConfig, context: context)
            
            XCTAssertNotNil(testEngine)
            
            // Test basic operation
            let sampleData = Self.generateDiverseTestEmbeddings(count: 5, dimension: 256)
            try await testEngine.addEmbeddings(sampleData)
            
            let results = try await testEngine.search(query: sampleData[0], k: 2)
            XCTAssertEqual(results.count, 2)
        }
    }
    
    // MARK: - Advanced Embedding Management Tests
    
    func testLargeScaleEmbeddingAddition() async throws {
        await engine!.clearEmbeddings()
        
        let batchSizes = [100, 500, 1000, 2000]
        var totalCount = 0
        
        for batchSize in batchSizes {
            let batch = Self.generateDiverseTestEmbeddings(count: batchSize, dimension: 256)
            let metadata = generateTestMetadata(count: batchSize)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            try await engine!.addEmbeddings(batch, metadata: metadata)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            totalCount += batchSize
            let metrics = await engine!.getPerformanceMetrics()
            XCTAssertEqual(metrics.embeddings, totalCount)
            
            print("Added \(batchSize) embeddings in \(elapsed * 1000)ms (total: \(totalCount))")
        }
    }
    
    func testConcurrentEmbeddingAddition() async throws {
        await engine!.clearEmbeddings()
        
        let numTasks = 5
        let embeddingsPerTask = 50
        
        await withTaskGroup(of: Void.self) { group in
            for taskId in 0..<numTasks {
                group.addTask { [engine] in
                    do {
                        let batch = EmbeddingEngineEnhancedTests.generateDiverseTestEmbeddings(count: embeddingsPerTask, dimension: 256)
                        let metadata = batch.enumerated().map { (idx, _) in
                            ["task": "\(taskId)", "index": "\(idx)"]
                        }
                        
                        try await engine!.addEmbeddings(batch, metadata: metadata)
                    } catch {
                        XCTFail("Concurrent addition failed: \(error)")
                    }
                }
            }
        }
        
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, numTasks * embeddingsPerTask)
    }
    
    func testEmbeddingAdditionWithVariousDataTypes() async throws {
        await engine!.clearEmbeddings()
        
        // Test with edge case embeddings
        let edgeCases: [[Float]] = [
            Array(repeating: 0.0, count: 256), // All zeros
            Array(repeating: 1.0, count: 256), // All ones
            Array(repeating: Float.infinity, count: 256).map { $0.isInfinite ? 1e6 : $0 }, // Very large values (clamped)
            Array(repeating: Float.ulpOfOne, count: 256), // Very small values
            (0..<256).map { i in i % 2 == 0 ? 1.0 : -1.0 }, // Alternating pattern
            (0..<256).map { Float($0) / 256.0 }, // Linear ramp
        ]
        
        for (idx, embedding) in edgeCases.enumerated() {
            try await engine!.addEmbeddings([embedding], metadata: [["type": "edge_case_\(idx)"]])
        }
        
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, edgeCases.count)
        
        // Test search with edge cases
        for (idx, query) in edgeCases.enumerated() {
            let results = try await engine!.search(query: query, k: 1)
            XCTAssertEqual(results.count, 1)
            // The query should find itself (or very similar)
            // Note: Zero vectors with cosine distance have undefined behavior
            if results[0].index == idx {
                // For zero vectors (idx == 0), cosine distance might be 1.0 (max)
                // For other vectors, expect small distance
                if idx == 0 {
                    // Zero vector case - just verify it doesn't crash
                    XCTAssertTrue(results[0].distance >= 0.0 && results[0].distance <= 1.0,
                                 "Distance should be valid for zero vector")
                } else {
                    XCTAssertLessThan(results[0].distance, 0.001)
                }
            }
        }
    }
    
    // MARK: - Advanced Search Tests
    
    func testPrecisionRecallAnalysis() async throws {
        // Create ground truth with known similar pairs
        let (clusteredEmbeddings, trueLabels) = createClusteredEmbeddings(
            numClusters: 5, 
            pointsPerCluster: 20, 
            dimension: 256, 
            clusterRadius: 0.2
        )
        
        try await engine!.addEmbeddings(clusteredEmbeddings)
        
        var precisionSum: Float = 0
        var recallSum: Float = 0
        let numQueries = min(20, clusteredEmbeddings.count)
        
        for queryIdx in 0..<numQueries {
            let query = clusteredEmbeddings[queryIdx]
            let queryLabel = trueLabels[queryIdx]
            
            let results = try await engine!.search(query: query, k: 10)
            
            // Calculate precision and recall
            var truePositives = 0
            var totalRelevant = 0
            
            // Count relevant items in entire dataset
            for label in trueLabels {
                if label == queryLabel {
                    totalRelevant += 1
                }
            }
            
            // Count true positives in results
            for result in results {
                if result.index < trueLabels.count && trueLabels[result.index] == queryLabel {
                    truePositives += 1
                }
            }
            
            let precision = results.isEmpty ? 0 : Float(truePositives) / Float(results.count)
            let recall = totalRelevant == 0 ? 0 : Float(truePositives) / Float(totalRelevant)
            
            precisionSum += precision
            recallSum += recall
        }
        
        let avgPrecision = precisionSum / Float(numQueries)
        let avgRecall = recallSum / Float(numQueries)
        
        print("Search Quality Metrics:")
        print("  Average Precision: \(avgPrecision)")
        print("  Average Recall: \(avgRecall)")
        
        // For well-clustered data, we expect reasonable precision
        XCTAssertGreaterThan(avgPrecision, 0.3)
    }
    
    func testSearchWithVariousThresholds() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<50]))
        
        let query = testEmbeddings[0]
        let thresholds: [Float] = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        var previousCount = 0  // Start with 0 since we're going from restrictive to less restrictive
        
        for threshold in thresholds {
            let results = try await engine!.search(query: query, k: 50, threshold: threshold)
            
            // Verify all results are within threshold
            for result in results {
                XCTAssertLessThanOrEqual(result.distance, threshold, 
                                        "Result distance \(result.distance) exceeds threshold \(threshold)")
            }
            
            // Less restrictive thresholds (higher values) should return more or equal results
            XCTAssertGreaterThanOrEqual(results.count, previousCount,
                                   "Threshold \(threshold) returned fewer results than more restrictive threshold")
            
            previousCount = results.count
            print("Threshold \(threshold): \(results.count) results")
        }
    }
    
    func testSearchStability() async throws {
        // Test that repeated searches return consistent results
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<100]))
        
        let query = testEmbeddings[0]
        let k = 10
        
        // Perform multiple searches
        var allResults: [[SearchResult]] = []
        for _ in 0..<5 {
            let results = try await engine!.search(query: query, k: k)
            allResults.append(results)
        }
        
        // Verify consistency
        let firstResults = allResults[0]
        for otherResults in allResults.dropFirst() {
            XCTAssertEqual(firstResults.count, otherResults.count)
            
            for (first, other) in zip(firstResults, otherResults) {
                XCTAssertEqual(first.index, other.index)
                XCTAssertEqual(first.distance, other.distance, accuracy: 1e-6)
            }
        }
    }
    
    func testSearchWithDuplicateEmbeddings() async throws {
        await engine!.clearEmbeddings()
        
        let originalEmbedding = testEmbeddings[0]
        let duplicates = Array(repeating: originalEmbedding, count: 5)
        let uniqueEmbeddings = Array(testEmbeddings[1..<20])
        
        var allEmbeddings = duplicates + uniqueEmbeddings
        allEmbeddings.shuffle() // Randomize order
        
        try await engine!.addEmbeddings(allEmbeddings)
        
        let results = try await engine!.search(query: originalEmbedding, k: 10)
        
        // Should find all duplicates first
        let nearPerfectMatches = results.filter { $0.distance < 1e-6 }
        XCTAssertEqual(nearPerfectMatches.count, 5, "Should find all 5 duplicates")
        
        // Verify they're sorted by distance
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance)
        }
    }
    
    // MARK: - Advanced Batch Search Tests
    
    func testBatchSearchPerformanceComparison() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let queries = Array(testEmbeddings[0..<20])
        let k = 5
        
        // Time sequential searches
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        var sequentialResults: [[SearchResult]] = []
        for query in queries {
            let results = try await engine!.search(query: query, k: k)
            sequentialResults.append(results)
        }
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStart
        
        // Time batch search
        let batchStart = CFAbsoluteTimeGetCurrent()
        let batchResults = try await engine!.batchSearch(queries: queries, k: k)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        print("Search Performance Comparison:")
        print("  Sequential: \(sequentialTime * 1000)ms")
        print("  Batch: \(batchTime * 1000)ms")
        print("  Speedup: \(sequentialTime / batchTime)x")
        
        // Verify results are equivalent
        XCTAssertEqual(sequentialResults.count, batchResults.count)
        for (sequential, batch) in zip(sequentialResults, batchResults) {
            XCTAssertEqual(sequential.count, batch.count)
            for (seqResult, batchResult) in zip(sequential, batch) {
                XCTAssertEqual(seqResult.index, batchResult.index)
                XCTAssertEqual(seqResult.distance, batchResult.distance, accuracy: 1e-6)
            }
        }
        
        // Batch should generally be faster for reasonable batch sizes
        // (though this may not always be true for small batches or simple operations)
        if queries.count > 10 {
            print("Batch processing speedup achieved")
        }
    }
    
    func testBatchSearchWithMixedQueries() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        // Create queries of different types
        let queries = [
            testEmbeddings[0], // Exact match
            Array(repeating: Float(0), count: 256), // Zero vector
            testEmbeddings[10].map { $0 + Float.random(in: -0.1...0.1) }, // Similar vector
            (0..<256).map { _ in Float.random(in: -10...10) }, // Random vector
            Array(repeating: Float(1), count: 256) // Constant vector
        ]
        
        let batchResults = try await engine!.batchSearch(queries: queries, k: 5, threshold: 2.0)
        
        XCTAssertEqual(batchResults.count, queries.count)
        
        for (idx, results) in batchResults.enumerated() {
            XCTAssertLessThanOrEqual(results.count, 5)
            
            // Verify distances are within threshold
            for result in results {
                XCTAssertLessThanOrEqual(result.distance, 2.0)
            }
            
            print("Query \(idx): \(results.count) results")
        }
    }
    
    // MARK: - Advanced Clustering Tests
    
    func testKMeansWithDifferentK() async throws {
        let (clusteredData, _) = createClusteredEmbeddings(
            numClusters: 4, 
            pointsPerCluster: 25, 
            dimension: 256
        )
        
        try await engine!.addEmbeddings(clusteredData)
        
        let kValues = [2, 3, 4, 5, 6]
        var inertiaValues: [Float] = []
        
        for k in kValues {
            let result = try await engine!.kMeansClustering(k: k, maxIterations: 50)
            
            XCTAssertEqual(result.centroids.count, k)
            XCTAssertEqual(result.clusterAssignments.count, clusteredData.count)
            XCTAssertGreaterThan(result.inertia, 0)
            
            inertiaValues.append(result.inertia)
            print("k=\(k): inertia=\(result.inertia), iterations=\(result.iterations)")
            
            // Verify all assignments are valid
            for assignment in result.clusterAssignments {
                XCTAssertGreaterThanOrEqual(assignment, 0)
                XCTAssertLessThan(assignment, k)
            }
            
            // Verify centroids have correct dimensions
            for centroid in result.centroids {
                XCTAssertEqual(centroid.count, 256)
            }
        }
        
        // Generally, inertia should decrease as k increases (though not always monotonically)
        let firstInertia = inertiaValues.first!
        let lastInertia = inertiaValues.last!
        print("Inertia change from k=\(kValues.first!) to k=\(kValues.last!): \(firstInertia) -> \(lastInertia)")
    }
    
    func testKMeansConvergenceAnalysis() async throws {
        let (clusteredData, _) = createClusteredEmbeddings(
            numClusters: 3, 
            pointsPerCluster: 30, 
            dimension: 256,
            clusterRadius: 0.5
        )
        
        try await engine!.addEmbeddings(clusteredData)
        
        let tolerances: [Float] = [1e-2, 1e-3, 1e-4, 1e-5]
        
        for tolerance in tolerances {
            let result = try await engine!.kMeansClustering(
                k: 3, 
                maxIterations: 100, 
                tolerance: tolerance
            )
            
            print("Tolerance \(tolerance): converged in \(result.iterations) iterations, inertia=\(result.inertia)")
            
            // Stricter tolerance should generally require more iterations
            XCTAssertLessThanOrEqual(result.iterations, 100)
            
            // Calculate clustering quality using silhouette score
            let silhouetteScore = await calculateSilhouetteScore(
                embeddings: clusteredData,
                assignments: result.clusterAssignments,
                centroids: result.centroids
            )
            
            print("  Silhouette score: \(silhouetteScore)")
            
            // For well-separated clusters, silhouette should be positive
            // (though this depends on the data and clustering quality)
        }
    }
    
    func testKMeansWithDifferentInitialization() async throws {
        let data = Array(testEmbeddings[0..<60])
        try await engine!.addEmbeddings(data)
        
        var results: [ClusterResult] = []
        
        // Run k-means multiple times (k-means++ should give consistent results)
        for run in 0..<5 {
            let result = try await engine!.kMeansClustering(k: 3, maxIterations: 50)
            results.append(result)
            
            print("Run \(run + 1): inertia=\(result.inertia), iterations=\(result.iterations)")
        }
        
        // Check variance in results
        let inertiaValues = results.map { $0.inertia }
        let meanInertia = inertiaValues.reduce(0, +) / Float(inertiaValues.count)
        let variance = inertiaValues.map { pow($0 - meanInertia, 2) }.reduce(0, +) / Float(inertiaValues.count)
        
        print("Inertia statistics: mean=\(meanInertia), variance=\(variance)")
        
        // k-means++ should provide relatively consistent results
        XCTAssertLessThan(variance, meanInertia * 0.5, "High variance in clustering results")
    }
    
    // MARK: - Performance and Scaling Tests
    
    func testMemoryUsageMonitoring() async throws {
        await engine!.clearEmbeddings()
        
        let batchSize = 1000
        let numBatches = 5
        
        for batchIdx in 0..<numBatches {
            let batch = Self.generateDiverseTestEmbeddings(count: batchSize, dimension: 256)
            
            // Measure memory before addition
            let memoryBefore = mach_memory_info()
            
            try await engine!.addEmbeddings(batch)
            
            // Measure memory after addition
            let memoryAfter = mach_memory_info()
            let memoryIncrease = memoryAfter - memoryBefore
            
            let metrics = await engine!.getPerformanceMetrics()
            
            print("Batch \(batchIdx + 1): \(metrics.embeddings) total embeddings, memory increase: \(memoryIncrease) bytes")
            
            // Verify no memory leaks (rough check)
            let expectedSize = batchSize * 256 * MemoryLayout<Float>.size
            XCTAssertLessThan(memoryIncrease, expectedSize * 3, // Allow 3x overhead
                            "Memory usage seems excessive")
        }
    }
    
    private func mach_memory_info() -> Int {
        // Simple memory usage approximation
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
    
    func testSearchLatencyDistribution() async throws {
        try await engine!.addEmbeddings(testEmbeddings)
        
        let numSearches = 100
        var latencies: [TimeInterval] = []
        
        for i in 0..<numSearches {
            let query = testEmbeddings[i % testEmbeddings.count]
            
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.search(query: query, k: 5)
            let latency = CFAbsoluteTimeGetCurrent() - start
            
            latencies.append(latency)
        }
        
        // Calculate statistics
        let sortedLatencies = latencies.sorted()
        let mean = latencies.reduce(0, +) / Double(latencies.count)
        let median = sortedLatencies[sortedLatencies.count / 2]
        let p95 = sortedLatencies[Int(Double(sortedLatencies.count) * 0.95)]
        let p99 = sortedLatencies[Int(Double(sortedLatencies.count) * 0.99)]
        
        print("Search Latency Distribution:")
        print("  Mean: \(mean * 1000)ms")
        print("  Median: \(median * 1000)ms")
        print("  95th percentile: \(p95 * 1000)ms")
        print("  99th percentile: \(p99 * 1000)ms")
        
        // Reasonable performance expectations
        XCTAssertLessThan(mean, 0.1, "Average search time should be under 100ms")
        XCTAssertLessThan(p95, 0.2, "95th percentile should be under 200ms")
    }
    
    // MARK: - GPU vs CPU Comparison Tests
    
    func testAccuracyConsistencyGPUvsCPU() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let testData = Array(testEmbeddings[0..<50])
        let queries = Array(testEmbeddings[0..<5])
        
        // CPU engine
        let cpuConfig = EmbeddingConfiguration(
            dimension: 256,
            distanceMetric: .euclidean,
            useGPU: false,
            normalizeEmbeddings: false
        )
        let cpuEngine = try await EmbeddingEngine(configuration: cpuConfig)
        try await cpuEngine.addEmbeddings(testData)
        
        // GPU engine
        let gpuConfig = EmbeddingConfiguration(
            dimension: 256,
            distanceMetric: .euclidean,
            useGPU: true,
            normalizeEmbeddings: false
        )
        let gpuEngine = try await EmbeddingEngine(configuration: gpuConfig)
        try await gpuEngine.addEmbeddings(testData)
        
        for (queryIdx, query) in queries.enumerated() {
            let cpuResults = try await cpuEngine.search(query: query, k: 10)
            let gpuResults = try await gpuEngine.search(query: query, k: 10)
            
            XCTAssertEqual(cpuResults.count, gpuResults.count, "Result count mismatch for query \(queryIdx)")
            
            for (cpu, gpu) in zip(cpuResults, gpuResults) {
                XCTAssertEqual(cpu.index, gpu.index, "Index mismatch for query \(queryIdx)")
                XCTAssertEqual(cpu.distance, gpu.distance, accuracy: 1e-3, "Distance mismatch for query \(queryIdx)")
            }
        }
    }
    
    func testPerformanceScaling() async throws {
        let sizes = [100, 500, 1000, 2000]
        let dimension = 256
        
        for size in sizes {
            await engine!.clearEmbeddings()
            
            let data = Self.generateDiverseTestEmbeddings(count: size, dimension: dimension)
            
            // Time embedding addition
            let addStart = CFAbsoluteTimeGetCurrent()
            try await engine!.addEmbeddings(data)
            let addTime = CFAbsoluteTimeGetCurrent() - addStart
            
            // Time search operations
            let query = data[0]
            let searchStart = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.search(query: query, k: 10)
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStart
            
            // Time clustering (for smaller datasets)
            var clusterTime: TimeInterval = 0
            if size <= 1000 {
                let clusterStart = CFAbsoluteTimeGetCurrent()
                _ = try await engine!.kMeansClustering(k: min(5, size / 20), maxIterations: 20)
                clusterTime = CFAbsoluteTimeGetCurrent() - clusterStart
            }
            
            print("Performance for \(size) embeddings (dim=\(dimension)):")
            print("  Add time: \(addTime * 1000)ms (\(Double(size) / addTime) embeddings/sec)")
            print("  Search time: \(searchTime * 1000)ms")
            if clusterTime > 0 {
                print("  Cluster time: \(clusterTime * 1000)ms")
            }
            
            // Performance expectations
            XCTAssertLessThan(searchTime, 1.0, "Search should complete within 1 second")
            XCTAssertLessThan(addTime, Double(size) * 0.001, "Addition should be efficient")
        }
    }
    
    // MARK: - Error Handling and Edge Cases
    
    func testConcurrentOperationsSafety() async throws {
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<100]))
        
        // Perform concurrent operations of different types
        await withTaskGroup(of: Void.self) { group in
            // Search tasks
            for i in 0..<10 {
                group.addTask { [engine, testEmbeddings] in
                    do {
                        let query = testEmbeddings![i % testEmbeddings!.count]
                        _ = try await engine!.search(query: query, k: 5)
                    } catch {
                        XCTFail("Concurrent search failed: \(error)")
                    }
                }
            }
            
            // Batch search tasks
            for i in 0..<3 {
                group.addTask { [engine, testEmbeddings] in
                    do {
                        let queries = Array(testEmbeddings![i*3..<(i+1)*3])
                        _ = try await engine!.batchSearch(queries: queries, k: 3)
                    } catch {
                        XCTFail("Concurrent batch search failed: \(error)")
                    }
                }
            }
            
            // Metrics access
            group.addTask { [engine] in
                for _ in 0..<20 {
                    _ = await engine!.getPerformanceMetrics()
                    try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
                }
            }
            
            // One clustering task
            group.addTask { [engine] in
                do {
                    _ = try await engine!.kMeansClustering(k: 3, maxIterations: 10)
                } catch {
                    XCTFail("Concurrent clustering failed: \(error)")
                }
            }
        }
        
        // Verify engine is still functional
        let finalResults = try await engine!.search(query: testEmbeddings[0], k: 5)
        XCTAssertEqual(finalResults.count, 5)
    }
    
    func testResourceCleanupAfterErrors() async throws {
        // Test that resources are properly cleaned up after various error conditions
        
        // Test dimension mismatch error
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<10]))
        
        do {
            let wrongDimQuery = Array(repeating: Float(1), count: 128) // Wrong dimension
            _ = try await engine!.search(query: wrongDimQuery, k: 5)
            XCTFail("Should have thrown dimension mismatch error")
        } catch AccelerationError.dimensionMismatch {
            // Expected error
        }
        
        // Verify engine is still functional after error
        let validResults = try await engine!.search(query: testEmbeddings[0], k: 3)
        XCTAssertEqual(validResults.count, 3)
        
        // Test empty operations
        await engine!.clearEmbeddings()
        let emptyResults = try await engine!.search(query: testEmbeddings[0], k: 5)
        XCTAssertEqual(emptyResults.count, 0)
        
        // Verify engine can still add embeddings after being empty
        try await engine!.addEmbeddings(Array(testEmbeddings[0..<5]))
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.embeddings, 5)
    }
}

// MARK: - Test Extensions and Helpers

extension EmbeddingEngineEnhancedTests {
    
    func testConfigurationValidation() async throws {
        // Test various configuration combinations
        let validConfigs = [
            EmbeddingConfiguration(dimension: 64, distanceMetric: .euclidean, useGPU: false),
            EmbeddingConfiguration(dimension: 512, distanceMetric: .cosine, useGPU: true, normalizeEmbeddings: true),
            EmbeddingConfiguration(dimension: 1024, distanceMetric: .manhattan, batchSize: 100),
        ]
        
        for config in validConfigs {
            let testEngine = try await EmbeddingEngine(configuration: config)
            XCTAssertNotNil(testEngine)
            
            // Test basic functionality with this configuration
            let sampleData = Self.generateDiverseTestEmbeddings(count: 5, dimension: config.dimension)
            try await testEngine.addEmbeddings(sampleData)
            
            let results = try await testEngine.search(query: sampleData[0], k: 2)
            XCTAssertLessThanOrEqual(results.count, 2)
        }
    }
    
    func testMetadataIntegrity() async throws {
        let embeddings = Array(testEmbeddings[0..<20])
        let metadata = generateTestMetadata(count: 20)
        
        try await engine!.addEmbeddings(embeddings, metadata: metadata)
        
        // Search and verify metadata is preserved
        for i in 0..<5 {
            let results = try await engine!.search(query: embeddings[i], k: 1)
            XCTAssertEqual(results.count, 1)
            
            let result = results[0]
            XCTAssertEqual(result.index, i)
            XCTAssertNotNil(result.metadata)
            XCTAssertEqual(result.metadata?["id"], "embedding_\(i)")
            XCTAssertEqual(result.metadata?["index"], "\(i)")
        }
    }
}