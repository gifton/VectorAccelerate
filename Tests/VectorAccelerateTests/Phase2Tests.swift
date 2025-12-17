//
//  Phase2Tests.swift
//  VectorAccelerateTests
//
//  Tests for Phase 2: Advanced Operations & Optimizations
//

import XCTest
@testable import VectorAccelerate
import Foundation

final class Phase2Tests: XCTestCase {
    var context: Metal4Context!
    
    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
    }
    
    override func tearDown() async throws {
        await context?.cleanup()
        context = nil
        try await super.tearDown()
    }
    
    // MARK: - MatrixEngine Tests
    
    func testMatrixMultiplication() async throws {
        let engine = await MatrixEngine(context: context)
        
        // Test 2x3 * 3x2 = 2x2
        let matrixA: [Float] = [
            1, 2, 3,
            4, 5, 6
        ]
        let descriptorA = MatrixDescriptor(rows: 2, columns: 3)
        
        let matrixB: [Float] = [
            7, 8,
            9, 10,
            11, 12
        ]
        let descriptorB = MatrixDescriptor(rows: 3, columns: 2)
        
        let result = try await engine.multiply(
            matrixA, descriptorA: descriptorA,
            matrixB, descriptorB: descriptorB
        )
        
        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12,
        //            4*7+5*9+6*11, 4*8+5*10+6*12]
        //         = [58, 64, 139, 154]
        let expected: [Float] = [58, 64, 139, 154]
        
        XCTAssertEqual(result.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.001)
        }
    }
    
    func testMatrixTranspose() async throws {
        let engine = await MatrixEngine(context: context)
        
        let matrix: [Float] = [
            1, 2, 3,
            4, 5, 6
        ]
        let descriptor = MatrixDescriptor(rows: 2, columns: 3)
        
        let transposed = try await engine.transpose(matrix, descriptor: descriptor)
        
        // Expected: [1, 4,
        //           2, 5,
        //           3, 6]
        let expected: [Float] = [1, 4, 2, 5, 3, 6]
        
        XCTAssertEqual(transposed.count, 6)
        for i in 0..<6 {
            XCTAssertEqual(transposed[i], expected[i], accuracy: 0.001)
        }
    }
    
    func testBatchMatrixVectorMultiply() async throws {
        let engine = await MatrixEngine(context: context)
        
        let matrices: [[Float]] = [
            [1, 2, 3, 4],  // 2x2 matrix
            [5, 6, 7, 8]   // 2x2 matrix
        ]
        
        let vectors: [[Float]] = [
            [1, 2],
            [3, 4]
        ]
        
        let descriptor = MatrixDescriptor(rows: 2, columns: 2)
        
        let results = try await engine.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        
        XCTAssertEqual(results.count, 2)
        
        // First result: [1*1+2*2, 3*1+4*2] = [5, 11]
        XCTAssertEqual(results[0].count, 2)
        XCTAssertEqual(results[0][0], 5, accuracy: 0.001)
        XCTAssertEqual(results[0][1], 11, accuracy: 0.001)
        
        // Second result: [5*3+6*4, 7*3+8*4] = [39, 53]
        XCTAssertEqual(results[1].count, 2)
        XCTAssertEqual(results[1][0], 39, accuracy: 0.001)
        XCTAssertEqual(results[1][1], 53, accuracy: 0.001)
    }
    
    func testLargeMatrixMultiplication() async throws {
        let engine = await MatrixEngine(context: context)
        
        // Test 100x100 matrix multiplication
        let size = 100
        let matrixA = [Float](repeating: 1.0, count: size * size)
        let matrixB = [Float](repeating: 2.0, count: size * size)
        
        let descriptorA = MatrixDescriptor(rows: size, columns: size)
        let descriptorB = MatrixDescriptor(rows: size, columns: size)
        
        let result = try await engine.multiply(
            matrixA, descriptorA: descriptorA,
            matrixB, descriptorB: descriptorB
        )
        
        // Each element should be size * 1 * 2 = 200
        XCTAssertEqual(result.count, size * size)
        XCTAssertEqual(result[0], Float(size * 2), accuracy: 0.001)
        XCTAssertEqual(result[size * size - 1], Float(size * 2), accuracy: 0.001)
    }
    
    // MARK: - SIMDFallback Tests
    
    func testSIMDDotProduct() async throws {
        let simd = SIMDFallback(configuration: .performance)
        
        let a: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let b: [Float] = [8, 7, 6, 5, 4, 3, 2, 1]
        
        let result = try await simd.dotProduct(a, b)
        
        // Expected: 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
        //         = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
        XCTAssertEqual(result, 120, accuracy: 0.001)
    }
    
    func testSIMDEuclideanDistance() async throws {
        let simd = SIMDFallback()
        
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [5, 6, 7, 8]
        
        let result = try await simd.euclideanDistance(a, b)
        
        // Expected: sqrt((5-1)² + (6-2)² + (7-3)² + (8-4)²)
        //         = sqrt(16 + 16 + 16 + 16) = sqrt(64) = 8
        XCTAssertEqual(result, 8.0, accuracy: 0.001)
    }
    
    func testSIMDNormalize() async throws {
        let simd = SIMDFallback()
        
        let vector: [Float] = [3, 4, 0, 0]
        let normalized = await simd.normalize(vector)
        
        // Norm = sqrt(9 + 16) = 5
        // Expected: [3/5, 4/5, 0, 0] = [0.6, 0.8, 0, 0]
        XCTAssertEqual(normalized.count, 4)
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.001)
        XCTAssertEqual(normalized[2], 0.0, accuracy: 0.001)
        XCTAssertEqual(normalized[3], 0.0, accuracy: 0.001)
    }
    
    func testSIMDMatrixVectorMultiply() async throws {
        let simd = SIMDFallback()
        
        let matrix: [Float] = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]
        let vector: [Float] = [1, 2, 3]
        
        let result = try await simd.matrixVectorMultiply(
            matrix: matrix,
            rows: 3,
            columns: 3,
            vector: vector
        )
        
        // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3]
        //         = [14, 32, 50]
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 14, accuracy: 0.001)
        XCTAssertEqual(result[1], 32, accuracy: 0.001)
        XCTAssertEqual(result[2], 50, accuracy: 0.001)
    }
    
    func testSIMDBatchDistance() async throws {
        let simd = SIMDFallback()
        
        let vectors: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        let reference: [Float] = [1, 1, 1]
        
        let results = try await simd.batchEuclideanDistance(
            vectors: vectors,
            reference: reference
        )
        
        XCTAssertEqual(results.count, 3)
        // All unit vectors are sqrt(2) away from [1,1,1]
        for result in results {
            XCTAssertEqual(result, sqrt(2), accuracy: 0.001)
        }
    }
    
    // MARK: - BatchProcessor Tests
    
    func testBatchProcessorSequential() async throws {
        let processor = await BatchProcessor(context: context)
        
        let data = Array(0..<10).map { Float($0) }
        let operation = SquareOperation()
        
        let result = try await processor.processBatch([data], operation: operation)
        
        XCTAssertEqual(result.results.count, 1)
        XCTAssertEqual(result.results[0].count, 10)
        
        for i in 0..<10 {
            XCTAssertEqual(result.results[0][i], Float(i * i), accuracy: 0.001)
        }
    }
    
    func testBatchProcessorParallel() async throws {
        let processor = await BatchProcessor(
            context: context,
            configuration: .init(strategy: .parallel(maxConcurrency: 4))
        )
        
        let vectors = (0..<100).map { i in
            [Float](repeating: Float(i), count: 10)
        }
        
        let operation = SumOperation()
        let result = try await processor.processBatch(vectors, operation: operation)
        
        XCTAssertEqual(result.results.count, 100)
        for i in 0..<100 {
            XCTAssertEqual(result.results[i], Float(i * 10), accuracy: 0.001)
        }
    }
    
    func testBatchProcessorStreaming() async throws {
        let processor = await BatchProcessor(
            context: context,
            configuration: .init(
                strategy: .streaming(chunkSize: 10),
                memoryLimit: 1024
            )
        )
        
        let data = (0..<100).map { i in [Float(i)] }
        let operation = IdentityOperation()
        
        let result = try await processor.processBatch(data, operation: operation)
        
        XCTAssertEqual(result.results.count, 100)
        XCTAssertGreaterThan(result.chunksProcessed, 1)  // Should process in chunks
        
        for i in 0..<100 {
            XCTAssertEqual(result.results[i][0], Float(i), accuracy: 0.001)
        }
    }
    
    func testBatchSimilarity() async throws {
        let processor = await BatchProcessor(context: context)
        
        let vectors: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        
        let references: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0]
        ]
        
        let results = try await processor.batchSimilarity(
            vectors: vectors,
            references: references,
            metric: .euclidean
        )
        
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].count, 2)
        
        // First vector [1,0,0] distance to [1,0,0] should be 0
        XCTAssertEqual(results[0][0], 0, accuracy: 0.001)
        // First vector [1,0,0] distance to [0,1,0] should be sqrt(2)
        XCTAssertEqual(results[0][1], sqrt(2), accuracy: 0.001)
    }
    
    // MARK: - Performance Tests
    
    func testMatrixPerformance() async throws {
        let engine = await MatrixEngine(context: context)
        
        measure {
            Task {
                let size = 512
                let matrixA = [Float](repeating: 1.0, count: size * size)
                let matrixB = [Float](repeating: 2.0, count: size * size)
                
                let descriptorA = MatrixDescriptor(rows: size, columns: size)
                let descriptorB = MatrixDescriptor(rows: size, columns: size)
                
                _ = try? await engine.multiply(
                    matrixA, descriptorA: descriptorA,
                    matrixB, descriptorB: descriptorB
                )
            }
        }
    }
    
    func testSIMDPerformance() async throws {
        let simd = SIMDFallback(configuration: .performance)
        
        measure {
            Task {
                let size = 10000
                let a = [Float](repeating: 1.0, count: size)
                let b = [Float](repeating: 2.0, count: size)
                
                _ = try? await simd.dotProduct(a, b)
            }
        }
    }
}

// MARK: - Test Helper Operations

struct SquareOperation: BatchOperation {
    typealias Input = [Float]
    typealias Output = [Float]
    
    var estimatedMemoryPerItem: Int { 100 }
    
    func process(_ batch: [[Float]]) async throws -> [[Float]] {
        batch.map { vector in
            vector.map { $0 * $0 }
        }
    }
}

struct SumOperation: BatchOperation {
    typealias Input = [Float]
    typealias Output = Float
    
    var estimatedMemoryPerItem: Int { 100 }
    
    func process(_ batch: [[Float]]) async throws -> [Float] {
        batch.map { vector in
            vector.reduce(0, +)
        }
    }
}

struct IdentityOperation: BatchOperation {
    typealias Input = [Float]
    typealias Output = [Float]
    
    var estimatedMemoryPerItem: Int { 100 }
    
    func process(_ batch: [[Float]]) async throws -> [[Float]] {
        batch
    }
}