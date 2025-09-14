//
//  Phase3Tests.swift
//  VectorAccelerateTests
//
//  Tests for Phase 3: Machine Learning Operations & Integration
//

import XCTest
@testable import VectorAccelerate
import CoreML
import Foundation

final class Phase3Tests: XCTestCase {
    var context: MetalContext!
    
    override func setUp() async throws {
        try await super.setUp()
        context = try await MetalContext()
    }
    
    override func tearDown() async throws {
        await context?.cleanup()
        context = nil
        try await super.tearDown()
    }
    
    // MARK: - EmbeddingEngine Tests
    
    func testEmbeddingEngineInitialization() async throws {
        let config = EmbeddingConfiguration(
            dimension: 128,
            distanceMetric: .cosine,
            useGPU: true
        )
        
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        let metrics = await engine.getPerformanceMetrics()
        
        XCTAssertEqual(metrics.embeddings, 0)
        XCTAssertEqual(metrics.searches, 0)
    }
    
    func testSimilaritySearch() async throws {
        let config = EmbeddingConfiguration(dimension: 3, normalizeEmbeddings: true)
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        
        // Add test embeddings
        let embeddings: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ]
        
        try await engine.addEmbeddings(embeddings)
        
        // Search for similar vectors
        let query: [Float] = [1, 0, 0]
        let results = try await engine.search(query: query, k: 2)
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].index, 0)  // Exact match
        XCTAssertLessThan(results[0].distance, 0.01)  // Very close to 0
    }
    
    func testBatchSearch() async throws {
        let config = EmbeddingConfiguration(
            dimension: 4, 
            distanceMetric: .euclidean,
            normalizeEmbeddings: false
        )
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        
        // Add embeddings
        var embeddings: [[Float]] = []
        for i in 0..<100 {
            embeddings.append([Float(i), Float(i+1), Float(i+2), Float(i+3)])
        }
        try await engine.addEmbeddings(embeddings)
        
        // Batch search
        let queries: [[Float]] = [
            [0, 1, 2, 3],
            [50, 51, 52, 53]
        ]
        
        let results = try await engine.batchSearch(queries: queries, k: 1)
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0][0].index, 0)
        XCTAssertEqual(results[1][0].index, 50)
    }
    
    func testKMeansClustering() async throws {
        let config = EmbeddingConfiguration(
            dimension: 2,
            distanceMetric: .euclidean,
            normalizeEmbeddings: false
        )
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        
        // Create clustered data
        var embeddings: [[Float]] = []
        
        // Cluster 1 around (0, 0)
        for _ in 0..<20 {
            embeddings.append([
                Float.random(in: -0.5...0.5),
                Float.random(in: -0.5...0.5)
            ])
        }
        
        // Cluster 2 around (5, 5)
        for _ in 0..<20 {
            embeddings.append([
                Float.random(in: 4.5...5.5),
                Float.random(in: 4.5...5.5)
            ])
        }
        
        try await engine.addEmbeddings(embeddings)
        
        // Perform clustering
        let result = try await engine.kMeansClustering(k: 2, maxIterations: 50)
        
        XCTAssertEqual(result.clusterAssignments.count, 40)
        XCTAssertEqual(result.centroids.count, 2)
        
        // Check that points are correctly clustered
        // Due to random initialization, we check that points are reasonably separated
        let cluster0InFirst20 = result.clusterAssignments[0..<20].filter { $0 == 0 }.count
        let cluster1InFirst20 = 20 - cluster0InFirst20
        let cluster0InSecond20 = result.clusterAssignments[20..<40].filter { $0 == 0 }.count
        let cluster1InSecond20 = 20 - cluster0InSecond20
        
        // Either cluster 0 dominates first group and cluster 1 dominates second, or vice versa
        let correctClustering = (cluster0InFirst20 >= 15 && cluster1InSecond20 >= 15) ||
                                (cluster1InFirst20 >= 15 && cluster0InSecond20 >= 15)
        
        // More lenient check - at least some separation should exist
        XCTAssertTrue(correctClustering || abs(cluster0InFirst20 - cluster0InSecond20) >= 10,
                     "Clustering failed to separate groups: cluster0InFirst20=\(cluster0InFirst20), cluster0InSecond20=\(cluster0InSecond20)")
    }
    
    // MARK: - QuantizationEngine Tests
    
    func testScalarQuantization8Bit() async throws {
        let config = QuantizationConfiguration(method: .scalar(bits: 8))
        let engine = await QuantizationEngine(configuration: config)
        
        let vector: [Float] = [0.1, 0.5, -0.3, 0.8, -0.9]
        let quantized = try await engine.scalarQuantize(vector: vector, bits: 8)
        
        // Check compression
        let originalSize = vector.count * MemoryLayout<Float>.size
        let quantizedSize = quantized.data.count
        XCTAssertLessThan(quantizedSize, originalSize)
        XCTAssertEqual(quantized.compressionRatio, 4.0, accuracy: 0.1)
        
        // Check reconstruction
        let reconstructed = try await engine.scalarDequantize(quantized: quantized)
        XCTAssertEqual(reconstructed.count, vector.count)
        
        // Check reconstruction error
        for i in 0..<vector.count {
            XCTAssertEqual(reconstructed[i], vector[i], accuracy: 0.02)
        }
    }
    
    func testScalarQuantization4Bit() async throws {
        let config = QuantizationConfiguration(method: .scalar(bits: 4))
        let engine = await QuantizationEngine(configuration: config)
        
        let vector: [Float] = Array(repeating: 0, count: 100).map { _ in Float.random(in: -1...1) }
        let quantized = try await engine.scalarQuantize(vector: vector, bits: 4)
        
        // 4-bit quantization should achieve ~8x compression
        XCTAssertEqual(quantized.compressionRatio, 8.0, accuracy: 0.5)
        
        _ = try await engine.scalarDequantize(quantized: quantized)
        
        // Calculate RMSE
        let error = try await engine.measureError(original: vector, quantized: quantized)
        XCTAssertLessThan(error, 0.2)  // Reasonable error for 4-bit
    }
    
    func testBinaryQuantization() async throws {
        let engine = await QuantizationEngine(configuration: QuantizationConfiguration())
        
        let vector: [Float] = [0.1, 0.8, -0.5, 0.3, -0.2, 0.9, -0.7, 0.4]
        let quantized = try await engine.binaryQuantize(vector: vector)
        
        // Binary quantization: 1 bit per value
        XCTAssertEqual(quantized.data.count, 1)  // 8 bits = 1 byte
        XCTAssertEqual(quantized.compressionRatio, 32.0, accuracy: 1.0)
        
        let reconstructed = try await engine.binaryDequantize(quantized: quantized)
        XCTAssertEqual(reconstructed.count, vector.count)
    }
    
    func testProductQuantizationTraining() async throws {
        let engine = await QuantizationEngine(configuration: QuantizationConfiguration())
        
        // Generate training data
        let trainingData: [[Float]] = (0..<100).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }
        
        // Train product quantization
        try await engine.trainProductQuantization(
            trainingData: trainingData,
            numCodebooks: 8,
            centroids: 16
        )
        
        // Quantize a vector
        let testVector = (0..<64).map { _ in Float.random(in: -1...1) }
        let quantized = try await engine.productQuantize(vector: testVector)
        
        // Check compression (8 codebooks * 1 byte per code = 8 bytes for 256 bytes original)
        XCTAssertEqual(quantized.data.count, 8)
        XCTAssertEqual(quantized.compressionRatio, 32.0, accuracy: 1.0)
        
        // Check reconstruction
        let reconstructed = try await engine.productDequantize(quantized: quantized)
        XCTAssertEqual(reconstructed.count, testVector.count)
    }
    
    // MARK: - CoreMLBridge Tests
    
    func testCoreMLBridgeInitialization() async throws {
        let config = CoreMLConfiguration(
            computeUnits: .cpuAndGPU,
            batchSize: 16
        )
        
        let bridge = await CoreMLBridge(configuration: config, context: context)
        let metrics = await bridge.getPerformanceMetrics()
        
        XCTAssertEqual(metrics.modelsLoaded, 0)
        XCTAssertEqual(metrics.inferenceCount, 0)
    }
    
    func testMultiArrayConversion() async throws {
        let bridge = await CoreMLBridge(context: context)
        
        // Create a test MLMultiArray
        let shape = [1, 128] as [NSNumber]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Fill with test data
        for i in 0..<128 {
            multiArray[i] = NSNumber(value: Float(i))
        }
        
        // Convert to Metal buffer
        let buffer = try await bridge.multiArrayToMetalBuffer(multiArray)
        
        if let buffer = buffer {
            let data = buffer.copyData(as: Float.self, count: 128)
            XCTAssertEqual(data.count, 128)
            XCTAssertEqual(data[0], 0.0, accuracy: 0.001)
            XCTAssertEqual(data[127], 127.0, accuracy: 0.001)
        }
    }
    
    func testEmbeddingProcessing() async throws {
        let bridge = await CoreMLBridge(context: context)
        
        // Create test embeddings as MLMultiArray
        let multiArray = try MLMultiArray(shape: [128], dataType: .float32)
        for i in 0..<128 {
            multiArray[i] = NSNumber(value: Float.random(in: -1...1))
        }
        
        // Process with normalization
        let normalized = try await bridge.processEmbeddings(multiArray, operation: .normalize)
        
        // Check normalization
        var norm: Float = 0
        for value in normalized {
            norm += value * value
        }
        norm = sqrt(norm)
        
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndEmbeddingPipeline() async throws {
        // Create embedding engine
        let embeddingConfig = EmbeddingConfiguration(dimension: 64)
        let embeddingEngine = try await EmbeddingEngine(configuration: embeddingConfig, context: context)
        
        // Create quantization engine
        let quantizationEngine = await QuantizationEngine(configuration: QuantizationConfiguration())
        
        // Generate synthetic embeddings
        let embeddings: [[Float]] = (0..<1000).map { i in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }
        
        // Add to engine
        try await embeddingEngine.addEmbeddings(embeddings)
        
        // Train quantization
        try await quantizationEngine.trainProductQuantization(
            trainingData: Array(embeddings[0..<100]),
            numCodebooks: 8,
            centroids: 16
        )
        
        // Quantize embeddings
        var quantizedEmbeddings: [QuantizedVector] = []
        for embedding in embeddings {
            let quantized = try await quantizationEngine.productQuantize(vector: embedding)
            quantizedEmbeddings.append(quantized)
        }
        
        // Verify compression
        let metrics = await quantizationEngine.getPerformanceMetrics()
        XCTAssertGreaterThan(metrics.averageCompressionRatio, 20.0)
        
        // Search on original embeddings
        let query = embeddings[500]
        let results = try await embeddingEngine.search(query: query, k: 5)
        
        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].index, 500)  // Should find itself
    }
    
    func testLargeScalePerformance() async throws {
        let config = EmbeddingConfiguration(dimension: 128, useGPU: true)
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        
        // Generate large dataset
        let embeddings: [[Float]] = (0..<10000).map { _ in
            (0..<128).map { _ in Float.random(in: -1...1) }
        }
        
        try await engine.addEmbeddings(embeddings)
        
        // Measure search performance
        let query = embeddings[5000]
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let results = try await engine.search(query: query, k: 100)
        let searchTime = CFAbsoluteTimeGetCurrent() - startTime
        
        XCTAssertEqual(results.count, 100)
        XCTAssertLessThan(searchTime, 0.1)  // Should be fast
        
        print("Search time for 10K embeddings: \(searchTime * 1000)ms")
    }
    
    // MARK: - Performance Tests
    
    func testEmbeddingSearchPerformance() async throws {
        let config = EmbeddingConfiguration(dimension: 512, useGPU: true)
        let engine = try await EmbeddingEngine(configuration: config, context: context)
        
        // Add embeddings
        let embeddings: [[Float]] = (0..<1000).map { _ in
            (0..<512).map { _ in Float.random(in: -1...1) }
        }
        try await engine.addEmbeddings(embeddings)
        
        measure {
            Task {
                let query = embeddings[0]
                _ = try? await engine.search(query: query, k: 10)
            }
        }
    }
    
    func testQuantizationPerformance() async throws {
        let engine = await QuantizationEngine(configuration: QuantizationConfiguration())
        
        let vector = (0..<1024).map { _ in Float.random(in: -1...1) }
        
        measure {
            Task {
                _ = try? await engine.scalarQuantize(vector: vector, bits: 8)
            }
        }
    }
}