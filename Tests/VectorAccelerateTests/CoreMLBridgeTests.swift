//
//  CoreMLBridgeTests.swift
//  VectorAccelerateTests
//
//  Tests for CoreMLBridge
//

import XCTest
@testable import VectorAccelerate
import CoreML

final class CoreMLBridgeTests: XCTestCase {
    var context: MetalContext!
    var bridge: CoreMLBridge!
    
    override func setUp() async throws {
        try await super.setUp()
        context = try await MetalContext()
        bridge = await CoreMLBridge(context: context)
    }
    
    override func tearDown() async throws {
        await context?.cleanup()
        context = nil
        bridge = nil
        try await super.tearDown()
    }
    
    // MARK: - Configuration Tests
    
    func testDefaultConfiguration() {
        let config = CoreMLConfiguration()
        XCTAssertEqual(config.computeUnits, .all)
        XCTAssertEqual(config.batchSize, 32)
        XCTAssertTrue(config.shareMemory)
        XCTAssertFalse(config.enableProfiling)
    }
    
    func testCustomConfiguration() async throws {
        let config = CoreMLConfiguration(
            computeUnits: .cpuAndGPU,
            batchSize: 64,
            shareMemory: false,
            enableProfiling: true
        )
        
        let customBridge = await CoreMLBridge(configuration: config, context: context)
        let metrics = await customBridge.getPerformanceMetrics()
        
        XCTAssertEqual(metrics.modelsLoaded, 0)
        XCTAssertEqual(metrics.inferenceCount, 0)
        XCTAssertEqual(metrics.averageInferenceTime, 0)
    }
    
    // MARK: - MLMultiArray Tests
    
    func testMultiArrayToMetalBufferFloat32() async throws {
        // Create test MLMultiArray
        let shape = [1, 64] as [NSNumber]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Fill with test data
        for i in 0..<64 {
            multiArray[i] = NSNumber(value: Float(i) * 0.5)
        }
        
        // Convert to Metal buffer
        let buffer = try await bridge.multiArrayToMetalBuffer(multiArray)
        
        if let buffer = buffer {
            let data = buffer.copyData(as: Float.self, count: 64)
            XCTAssertEqual(data.count, 64)
            XCTAssertEqual(data[0], 0.0, accuracy: 0.001)
            XCTAssertEqual(data[10], 5.0, accuracy: 0.001)
            XCTAssertEqual(data[63], 31.5, accuracy: 0.001)
        }
    }
    
    func testMultiArrayToMetalBufferNoShareMemory() async throws {
        let config = CoreMLConfiguration(shareMemory: false)
        let customBridge = await CoreMLBridge(configuration: config, context: context)
        
        let multiArray = try MLMultiArray(shape: [32], dataType: .float32)
        for i in 0..<32 {
            multiArray[i] = NSNumber(value: Float(i))
        }
        
        let buffer = try await customBridge.multiArrayToMetalBuffer(multiArray)
        XCTAssertNil(buffer, "Should return nil when shareMemory is false")
    }
    
    func testMultiArrayToMetalBufferNonFloat32() async throws {
        let multiArray = try MLMultiArray(shape: [16], dataType: .double)
        let buffer = try await bridge.multiArrayToMetalBuffer(multiArray)
        XCTAssertNil(buffer, "Should return nil for non-float32 data types")
    }
    
    // MARK: - Embedding Processing Tests
    
    func testProcessEmbeddingsNormalize() async throws {
        // Create embeddings as MLMultiArray
        let dimension = 128
        let multiArray = try MLMultiArray(shape: [dimension as NSNumber], dataType: .float32)
        
        // Fill with random values
        for i in 0..<dimension {
            multiArray[i] = NSNumber(value: Float.random(in: -1...1))
        }
        
        // Process with normalization
        let normalized = try await bridge.processEmbeddings(
            multiArray,
            operation: .normalize
        )
        
        XCTAssertEqual(normalized.count, dimension)
        
        // Check that result is normalized (L2 norm = 1)
        var norm: Float = 0
        for value in normalized {
            norm += value * value
        }
        norm = sqrt(norm)
        
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)
    }
    
    func testProcessEmbeddingsReduceMean() async throws {
        let multiArray = try MLMultiArray(shape: [5], dataType: .float32)
        for i in 0..<5 {
            multiArray[i] = NSNumber(value: Float(i + 1))  // 1, 2, 3, 4, 5
        }
        
        let result = try await bridge.processEmbeddings(
            multiArray,
            operation: .reduce(.mean)
        )
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 3.0, accuracy: 0.001)  // Mean of 1,2,3,4,5
    }
    
    func testProcessEmbeddingsReduceMax() async throws {
        let multiArray = try MLMultiArray(shape: [6], dataType: .float32)
        let values: [Float] = [3, 1, 4, 1, 5, 9]
        for i in 0..<6 {
            multiArray[i] = NSNumber(value: values[i])
        }
        
        let result = try await bridge.processEmbeddings(
            multiArray,
            operation: .reduce(.max)
        )
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 9.0, accuracy: 0.001)
    }
    
    func testProcessEmbeddingsReduceSum() async throws {
        let multiArray = try MLMultiArray(shape: [4], dataType: .float32)
        for i in 0..<4 {
            multiArray[i] = NSNumber(value: Float(i + 1))  // 1, 2, 3, 4
        }
        
        let result = try await bridge.processEmbeddings(
            multiArray,
            operation: .reduce(.sum)
        )
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 10.0, accuracy: 0.001)  // Sum of 1,2,3,4
    }
    
    // MARK: - Performance Metrics Tests
    
    func testInitialPerformanceMetrics() async {
        let metrics = await bridge.getPerformanceMetrics()
        
        XCTAssertEqual(metrics.modelsLoaded, 0)
        XCTAssertEqual(metrics.inferenceCount, 0)
        XCTAssertEqual(metrics.averageInferenceTime, 0.0)
    }
    
    // MARK: - Edge Case Tests
    
    func testProcessEmptyEmbeddings() async throws {
        let multiArray = try MLMultiArray(shape: [0], dataType: .float32)
        
        let normalized = try await bridge.processEmbeddings(
            multiArray,
            operation: .normalize
        )
        
        XCTAssertEqual(normalized.count, 0)
    }
    
    func testProcessSingleEmbedding() async throws {
        let multiArray = try MLMultiArray(shape: [1], dataType: .float32)
        multiArray[0] = NSNumber(value: Float(5.0))
        
        let normalized = try await bridge.processEmbeddings(
            multiArray,
            operation: .normalize
        )
        
        XCTAssertEqual(normalized.count, 1)
        XCTAssertEqual(abs(normalized[0]), 1.0, accuracy: 0.001)
    }
    
    func testProcessLargeEmbeddings() async throws {
        let dimension = 1024
        let multiArray = try MLMultiArray(shape: [dimension as NSNumber], dataType: .float32)
        
        for i in 0..<dimension {
            multiArray[i] = NSNumber(value: Float.random(in: -10...10))
        }
        
        let normalized = try await bridge.processEmbeddings(
            multiArray,
            operation: .normalize
        )
        
        XCTAssertEqual(normalized.count, dimension)
        
        // Verify normalization
        var norm: Float = 0
        for value in normalized {
            norm += value * value
        }
        norm = sqrt(norm)
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)
    }
    
    // MARK: - Configuration Profiling Tests
    
    func testProfilingConfiguration() async {
        let config = CoreMLConfiguration(enableProfiling: true)
        let profilingBridge = await CoreMLBridge(configuration: config, context: context)
        
        let metrics = await profilingBridge.getPerformanceMetrics()
        XCTAssertEqual(metrics.inferenceCount, 0)
        
        // After processing, metrics should be updated (if we had models loaded)
    }
    
    // MARK: - Multi-threaded Tests
    
    func testConcurrentProcessing() async throws {
        let bridge = self.bridge!
        
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<5 {
                group.addTask {
                    do {
                        let multiArray = try MLMultiArray(shape: [32], dataType: .float32)
                        for j in 0..<32 {
                            multiArray[j] = NSNumber(value: Float(i * 32 + j))
                        }
                        
                        _ = try await bridge.processEmbeddings(
                            multiArray,
                            operation: .normalize
                        )
                    } catch {
                        XCTFail("Concurrent processing failed: \(error)")
                    }
                }
            }
        }
    }
}