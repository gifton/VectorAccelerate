//
//  MemoryMapManagerTests.swift
//  VectorAccelerateTests
//
//  Tests for MemoryMapManager
//

import XCTest
@testable import VectorAccelerate
import Foundation

final class MemoryMapManagerTests: XCTestCase {
    var manager: MemoryMapManager!
    var testFileURL: URL!
    
    override func setUp() async throws {
        try await super.setUp()
        manager = MemoryMapManager()
        
        // Create temp file URL
        let tempDir = FileManager.default.temporaryDirectory
        testFileURL = tempDir.appendingPathComponent("test_vectors_\(UUID().uuidString).dat")
    }
    
    override func tearDown() async throws {
        // Clean up test file
        if let url = testFileURL {
            try? FileManager.default.removeItem(at: url)
        }
        manager = nil
        testFileURL = nil
        try await super.tearDown()
    }
    
    // MARK: - Configuration Tests
    
    func testDefaultConfiguration() {
        let config = MemoryMapConfiguration.default
        XCTAssertEqual(config.pageSize, 4096 * 1024)
        XCTAssertEqual(config.maxCacheSize, 256 * 1024 * 1024)
        XCTAssertEqual(config.prefetchDistance, 2)
        XCTAssertTrue(config.enableAsyncIO)
        XCTAssertFalse(config.compressionEnabled)
    }
    
    func testLargeDatasetConfiguration() {
        let config = MemoryMapConfiguration.largeDataset
        XCTAssertEqual(config.pageSize, 16 * 1024 * 1024)
        XCTAssertEqual(config.maxCacheSize, 1024 * 1024 * 1024)
        XCTAssertEqual(config.prefetchDistance, 4)
    }
    
    // MARK: - Dataset Creation Tests
    
    func testCreateDatasetFile() async throws {
        let vectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL,
            dataType: .float32
        )
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: testFileURL.path))
        
        let fileSize = try FileManager.default.attributesOfItem(atPath: testFileURL.path)[.size] as? Int
        let expectedSize = 16 + (3 * 3 * 4)  // header + data
        XCTAssertEqual(fileSize, expectedSize)
    }
    
    func testCreateDatasetWithFloat16() async throws {
        let vectors: [[Float]] = [
            [0.5, -0.5, 1.0],
            [0.25, -0.25, 0.0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL,
            dataType: .float16
        )
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: testFileURL.path))
        
        let fileSize = try FileManager.default.attributesOfItem(atPath: testFileURL.path)[.size] as? Int
        let expectedSize = 16 + (2 * 3 * 2)  // header + data (half precision)
        XCTAssertEqual(fileSize, expectedSize)
    }
    
    func testCreateDatasetWithInt8() async throws {
        let vectors: [[Float]] = [
            [0.5, -0.5, 1.0, -1.0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL,
            dataType: .int8
        )
        
        let fileSize = try FileManager.default.attributesOfItem(atPath: testFileURL.path)[.size] as? Int
        let expectedSize = 16 + (1 * 4 * 1)  // header + data (int8)
        XCTAssertEqual(fileSize, expectedSize)
    }
    
    // MARK: - Dataset Mapping Tests
    
    func testMapDataset() async throws {
        // Create test dataset
        let vectors: [[Float]] = (0..<100).map { i in
            [Float(i), Float(i+1), Float(i+2)]
        }
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        // Map the dataset
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        XCTAssertEqual(dataset.fileURL, testFileURL)
        XCTAssertEqual(dataset.vectorCount, 100)
        XCTAssertEqual(dataset.dimension, 3)
        XCTAssertEqual(dataset.dataType, .float32)
        XCTAssertEqual(dataset.vectorByteSize, 12)
        XCTAssertEqual(dataset.totalByteSize, 1200)
    }
    
    func testMapNonExistentFile() async throws {
        let nonExistentURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent.dat")
        
        do {
            _ = try await manager.mapDataset(at: nonExistentURL)
            XCTFail("Should throw file not found error")
        } catch {
            // Expected error
            XCTAssertTrue(error.localizedDescription.contains("not found"))
        }
    }
    
    // MARK: - Vector Reading Tests
    
    func testReadVectors() async throws {
        // Create test dataset
        let vectors: [[Float]] = (0..<10).map { i in
            [Float(i), Float(i*2), Float(i*3)]
        }
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        // Read vectors
        let readVectors = try await manager.readVectors(
            from: dataset,
            range: 2..<5
        )
        
        XCTAssertEqual(readVectors.count, 3)
        XCTAssertEqual(readVectors[0], [2.0, 4.0, 6.0])
        XCTAssertEqual(readVectors[1], [3.0, 6.0, 9.0])
        XCTAssertEqual(readVectors[2], [4.0, 8.0, 12.0])
    }
    
    func testReadAllVectors() async throws {
        let vectors: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        let readVectors = try await manager.readVectors(
            from: dataset,
            range: 0..<3
        )
        
        XCTAssertEqual(readVectors.count, 3)
        for i in 0..<3 {
            XCTAssertEqual(readVectors[i], vectors[i])
        }
    }
    
    // MARK: - Distance Computation Tests
    
    func testComputeDistancesEuclidean() async throws {
        let vectors: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        let query: [Float] = [1, 0, 0]
        let results = try await manager.computeDistances(
            dataset: dataset,
            query: query,
            metric: .euclidean,
            topK: 2
        )
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].index, 0)  // Exact match
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)
        XCTAssertEqual(results[1].index, 3)  // [1,1,0] is next closest
        XCTAssertEqual(results[1].distance, 1.0, accuracy: 0.001)
    }
    
    func testComputeDistancesCosine() async throws {
        let vectors: [[Float]] = [
            [1, 0],
            [0, 1],
            [0.707, 0.707],  // 45 degrees
            [-1, 0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        let query: [Float] = [1, 0]
        let results = try await manager.computeDistances(
            dataset: dataset,
            query: query,
            metric: .cosine,
            topK: 4
        )
        
        XCTAssertEqual(results.count, 4)
        XCTAssertEqual(results[0].index, 0)  // Exact match
        XCTAssertLessThan(results[0].distance, 0.001)
    }
    
    func testComputeDistancesManhattan() async throws {
        let vectors: [[Float]] = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        let query: [Float] = [1, 2]
        let results = try await manager.computeDistances(
            dataset: dataset,
            query: query,
            metric: .manhattan
        )
        
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].index, 0)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)
        XCTAssertEqual(results[1].distance, 4.0, accuracy: 0.001)  // |3-1| + |4-2| = 4
    }
    
    // MARK: - Cache Tests
    
    func testCacheStatistics() async throws {
        let vectors: [[Float]] = (0..<100).map { i in
            [Float(i), Float(i+1)]
        }
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        // Initial stats
        var stats = await manager.getCacheStatistics()
        XCTAssertEqual(stats.hitRate, 0.0)
        XCTAssertEqual(stats.pagesCached, 0)
        XCTAssertEqual(stats.totalReads, 0)
        
        // Read some vectors to populate cache
        _ = try await manager.readVectors(from: dataset, range: 0..<10)
        
        stats = await manager.getCacheStatistics()
        XCTAssertGreaterThan(stats.totalReads, 0)
        
        // Read same vectors again - should hit cache
        _ = try await manager.readVectors(from: dataset, range: 0..<10)
        
        stats = await manager.getCacheStatistics()
        XCTAssertGreaterThan(stats.hitRate, 0.0)
    }
    
    func testClearCache() async throws {
        let vectors: [[Float]] = [[1, 2], [3, 4]]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        _ = try await manager.readVectors(from: dataset, range: 0..<2)
        
        var stats = await manager.getCacheStatistics()
        XCTAssertGreaterThan(stats.pagesCached, 0)
        
        await manager.clearCache()
        
        stats = await manager.getCacheStatistics()
        XCTAssertEqual(stats.pagesCached, 0)
        XCTAssertEqual(stats.cacheSize, 0)
    }
    
    // MARK: - Stream Processing Tests
    
    func testStreamProcessing() async throws {
        let vectors: [[Float]] = (0..<100).map { i in
            [Float(i), Float(i+1), Float(i+2)]
        }
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        
        let stream = try await manager.streamProcess(
            dataset: dataset,
            batchSize: 10
        ) { batch in
            return batch.count  // Return batch size
        }
        
        var totalProcessed = 0
        for await batchSize in stream {
            totalProcessed += batchSize
        }
        
        XCTAssertEqual(totalProcessed, 100)
    }
    
    // MARK: - Cleanup Tests
    
    func testCloseDataset() async throws {
        let vectors: [[Float]] = [[1, 2], [3, 4]]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        _ = try await manager.readVectors(from: dataset, range: 0..<2)
        
        try await manager.closeDataset(dataset)
        
        let stats = await manager.getCacheStatistics()
        XCTAssertEqual(stats.pagesCached, 0)
    }
    
    // MARK: - Data Type Tests
    
    func testDataTypeByteSizes() {
        XCTAssertEqual(MappedVectorDataset.DataType.float32.byteSize, 4)
        XCTAssertEqual(MappedVectorDataset.DataType.float16.byteSize, 2)
        XCTAssertEqual(MappedVectorDataset.DataType.int8.byteSize, 1)
        XCTAssertEqual(MappedVectorDataset.DataType.uint8.byteSize, 1)
    }
    
    func testUInt8Normalization() async throws {
        let vectors: [[Float]] = [
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.0]
        ]
        
        try MemoryMapManager.createDatasetFile(
            vectors: vectors,
            at: testFileURL,
            dataType: .uint8
        )
        
        let dataset = try await manager.mapDataset(at: testFileURL)
        let readVectors = try await manager.readVectors(
            from: dataset,
            range: 0..<2
        )
        
        // Values should be normalized to [0, 1]
        XCTAssertEqual(readVectors[0][0], 0.0, accuracy: 0.01)
        XCTAssertEqual(readVectors[0][1], 0.5, accuracy: 0.01) 
        XCTAssertEqual(readVectors[0][2], 1.0, accuracy: 0.01)
    }
}