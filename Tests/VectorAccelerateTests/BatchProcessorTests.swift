//
//  BatchProcessorTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for BatchProcessor
//

@preconcurrency import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

// Define test operations that conform to BatchOperation
struct MultiplyOperation: BatchOperation {
    typealias Input = Float
    typealias Output = Float
    
    let multiplier: Float
    
    func process(_ batch: [Float]) async throws -> [Float] {
        return batch.map { $0 * multiplier }
    }
    
    var estimatedMemoryPerItem: Int { 4 }
}

struct TestSquareOperation: BatchOperation {
    typealias Input = Float
    typealias Output = Float
    
    func process(_ batch: [Float]) async throws -> [Float] {
        return batch.map { $0 * $0 }
    }
    
    var estimatedMemoryPerItem: Int { 4 }
}

struct IncrementOperation: BatchOperation {
    typealias Input = Float
    typealias Output = Float
    
    func process(_ batch: [Float]) async throws -> [Float] {
        return batch.map { $0 + 1 }
    }
    
    var estimatedMemoryPerItem: Int { 4 }
}

struct SqrtOperation: BatchOperation {
    typealias Input = Float
    typealias Output = Float
    
    func process(_ batch: [Float]) async throws -> [Float] {
        return batch.map { sqrt($0) }
    }
    
    var estimatedMemoryPerItem: Int { 4 }
}

struct ErrorOperation: BatchOperation {
    typealias Input = Float
    typealias Output = Float
    
    let errorValue: Float
    
    func process(_ batch: [Float]) async throws -> [Float] {
        if batch.contains(errorValue) {
            throw VectorError.computeFailed(reason: "Test error: found \(errorValue)")
        }
        return batch
    }
    
    var estimatedMemoryPerItem: Int { 4 }
}

final class BatchProcessorTests: XCTestCase {
    var context: Metal4Context!
    var processor: BatchProcessor!
    
    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
        processor = await BatchProcessor(context: context)
    }
    
    override func tearDown() async throws {
        processor = nil
        context = nil
        try await super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testDefaultInitialization() async {
        XCTAssertNotNil(processor)
        // Configuration is private, we can only test that processor exists
    }
    
    func testCustomConfiguration() async {
        let config = BatchConfiguration(
            strategy: .parallel(maxConcurrency: 8),
            maxBatchSize: 512,
            memoryLimit: 1024 * 1024 * 100 // 100MB
        )
        
        let customProcessor = await BatchProcessor(context: context, configuration: config)
        XCTAssertNotNil(customProcessor)
    }
    
    // MARK: - Batch Processing Tests
    
    func testSequentialBatchProcessing() async throws {
        let data = Array(0..<1000).map { Float($0) }
        let operation = MultiplyOperation(multiplier: 2)
        
        let result = try await processor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 1000)
        for i in 0..<1000 {
            XCTAssertEqual(result.results[i], Float(i * 2), accuracy: 1e-5)
        }
        XCTAssertGreaterThan(result.processingTime, 0)
    }
    
    func testParallelBatchProcessing() async throws {
        let data = Array(0..<10000).map { Float($0) }
        let operation = TestSquareOperation()
        
        let config = BatchConfiguration(strategy: .parallel(maxConcurrency: 4))
        let parallelProcessor = await BatchProcessor(context: context, configuration: config)
        let result = try await parallelProcessor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 10000)
        for i in 0..<10000 {
            XCTAssertEqual(result.results[i], Float(i * i), accuracy: 1e-5)
        }
    }
    
    func testStreamingBatchProcessing() async throws {
        let data = Array(0..<5000).map { Float($0) }
        let operation = IncrementOperation()
        
        let config = BatchConfiguration(strategy: .streaming(chunkSize: 500))
        let streamingProcessor = await BatchProcessor(context: context, configuration: config)
        let result = try await streamingProcessor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 5000)
        XCTAssertGreaterThan(result.chunksProcessed, 1)
    }
    
    func testAdaptiveBatchProcessing() async throws {
        let data = Array(0..<10000).map { Float($0) }
        let operation = SqrtOperation()
        
        let config = BatchConfiguration(strategy: .adaptive)
        let adaptiveProcessor = await BatchProcessor(context: context, configuration: config)
        let result = try await adaptiveProcessor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 10000)
        for i in 0..<10000 {
            XCTAssertEqual(result.results[i], sqrt(Float(i)), accuracy: 1e-5)
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testBatchProcessingWithErrors() async throws {
        let data = Array(0..<100).map { Float($0) }
        let operation = ErrorOperation(errorValue: 50)
        
        do {
            _ = try await processor.processBatch(data, operation: operation)
            XCTFail("Should have thrown error")
        } catch {
            // Expected error
            XCTAssertTrue(error is VectorError)
        }
    }
    
    func testEmptyBatchHandling() async throws {
        let data: [Float] = []
        let operation = MultiplyOperation(multiplier: 2)
        
        let result = try await processor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 0)
    }
    
    // MARK: - Performance Tests
    
    func testLargeBatchPerformance() async throws {
        let data = Array(0..<100000).map { Float($0) }
        let operation = TestSquareOperation()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let result = try await processor.processBatch(data, operation: operation)
        
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        
        XCTAssertEqual(result.results.count, 100000)
        XCTAssertLessThan(elapsedTime, 5.0, "Large batch should process within 5 seconds")
    }
    
    func testMemoryEfficientProcessing() async throws {
        // Test with memory-constrained configuration
        let config = BatchConfiguration(
            strategy: .streaming(chunkSize: 1000),
            memoryLimit: 10 * 1024 * 1024 // 10MB limit
        )
        
        let memoryProcessor = await BatchProcessor(context: context, configuration: config)
        let data = Array(0..<50000).map { Float($0) }
        let operation = MultiplyOperation(multiplier: 1.5)
        
        let result = try await memoryProcessor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 50000)
        XCTAssertGreaterThan(result.chunksProcessed, 1, "Should process in multiple chunks")
    }
    
    
    // MARK: - Concurrent Operations Tests
    
    func testConcurrentBatchOperations() async throws {
        let operations = [
            MultiplyOperation(multiplier: 2),
            MultiplyOperation(multiplier: 3),
            MultiplyOperation(multiplier: 4)
        ]
        
        let data = Array(0..<1000).map { Float($0) }
        
        // Run multiple batch operations concurrently
        let results = try await withThrowingTaskGroup(of: [Float].self) { group in
            for operation in operations {
                group.addTask { [processor] in
                    let result = try await processor!.processBatch(data, operation: operation)
                    return result.results
                }
            }
            
            var allResults: [[Float]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }
        
        XCTAssertEqual(results.count, 3)
        for result in results {
            XCTAssertEqual(result.count, 1000)
        }
    }
    
    // MARK: - Edge Cases
    
    func testSingleElementBatch() async throws {
        let data = [Float(42.0)]
        let operation = TestSquareOperation()
        
        let result = try await processor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 1)
        XCTAssertEqual(result.results[0], 42.0 * 42.0, accuracy: 1e-5)
    }
    
    func testVeryLargeValueHandling() async throws {
        let data = [Float.greatestFiniteMagnitude / 2, Float.leastNormalMagnitude * 2]
        let operation = MultiplyOperation(multiplier: 0.5)
        
        let result = try await processor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 2)
        XCTAssertFalse(result.results[0].isNaN)
        XCTAssertFalse(result.results[0].isInfinite)
    }
}