//
//  ComputeEngineTests.swift
//  VectorAccelerateTests
//
//  Tests for GPU-accelerated compute operations
//

import XCTest
@testable import VectorAccelerate
import VectorCore

final class ComputeEngineTests: XCTestCase {
    
    var context: VectorAccelerate.MetalContext!
    var engine: ComputeEngine!
    
    override func setUp() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        context = try await VectorAccelerate.MetalContext()
        engine = try await ComputeEngine(context: context)
    }
    
    // MARK: - Distance Tests
    
    func testEuclideanDistance() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0, 4.0]
        let vectorB: [Float] = [5.0, 6.0, 7.0, 8.0]
        
        // GPU computation
        let gpuDistance = try await engine.euclideanDistance(vectorA, vectorB)
        
        // CPU verification
        var sum: Float = 0
        for i in 0..<vectorA.count {
            let diff = vectorA[i] - vectorB[i]
            sum += diff * diff
        }
        let cpuDistance = sqrt(sum)
        
        XCTAssertEqual(gpuDistance, cpuDistance, accuracy: 1e-5)
        XCTAssertEqual(gpuDistance, 8.0, accuracy: 1e-5)
    }
    
    func testCosineDistance() async throws {
        let vectorA: [Float] = [1.0, 0.0, 0.0]
        let vectorB: [Float] = [0.0, 1.0, 0.0]
        
        // Orthogonal vectors should have cosine distance of 1
        let distance = try await engine.cosineDistance(vectorA, vectorB)
        XCTAssertEqual(distance, 1.0, accuracy: 1e-5)
        
        // Same vector should have cosine distance of 0
        let sameDistance = try await engine.cosineDistance(vectorA, vectorA)
        XCTAssertEqual(sameDistance, 0.0, accuracy: 1e-5)
        
        // Opposite vectors should have cosine distance of 2
        let opposite: [Float] = [-1.0, 0.0, 0.0]
        let oppositeDistance = try await engine.cosineDistance(vectorA, opposite)
        XCTAssertEqual(oppositeDistance, 2.0, accuracy: 1e-5)
    }
    
    func testDotProduct() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [4.0, 5.0, 6.0]
        
        // GPU computation
        let gpuDot = try await engine.dotProduct(vectorA, vectorB)
        
        // CPU verification: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let cpuDot: Float = 32.0
        
        XCTAssertEqual(gpuDot, cpuDot, accuracy: 1e-5)
    }
    
    // MARK: - Batch Operations
    
    func testBatchEuclideanDistance() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [0.0, 0.0, 0.0],  // Distance = 1
            [1.0, 1.0, 0.0],  // Distance = 1
            [1.0, 0.0, 1.0],  // Distance = 1
            [2.0, 0.0, 0.0],  // Distance = 1
        ]
        
        let distances = try await engine.batchEuclideanDistance(
            query: query,
            candidates: candidates
        )
        
        XCTAssertEqual(distances.count, candidates.count)
        
        // Verify distances
        XCTAssertEqual(distances[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(distances[2], 1.0, accuracy: 1e-5)
        XCTAssertEqual(distances[3], 1.0, accuracy: 1e-5)
    }
    
    func testBatchWithLargeDataset() async throws {
        let dimension = 128
        let candidateCount = 1000
        
        // Generate random data
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
        let candidates = (0..<candidateCount).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        // Measure performance
        let startTime = CFAbsoluteTimeGetCurrent()
        let distances = try await engine.batchEuclideanDistance(
            query: query,
            candidates: candidates
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        print("Batch distance computation for \(candidateCount) vectors of dimension \(dimension):")
        print("  Time: \(elapsed * 1000)ms")
        print("  Throughput: \(Double(candidateCount) / elapsed) vectors/sec")
        
        XCTAssertEqual(distances.count, candidateCount)
        
        // Verify all distances are positive
        for distance in distances {
            XCTAssertGreaterThanOrEqual(distance, 0)
        }
    }
    
    // MARK: - Vector Operations
    
    func testNormalize() async throws {
        let vector: [Float] = [3.0, 4.0, 0.0]
        
        let normalized = try await engine.normalize(vector)
        
        // Magnitude should be 5, so normalized = [3/5, 4/5, 0]
        XCTAssertEqual(normalized[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 1e-5)
        XCTAssertEqual(normalized[2], 0.0, accuracy: 1e-5)
        
        // Verify unit length
        let magnitude = sqrt(normalized.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-5)
    }
    
    func testScale() async throws {
        let vector: [Float] = [1.0, 2.0, 3.0]
        let scalar: Float = 2.5
        
        let scaled = try await engine.scale(vector, by: scalar)
        
        XCTAssertEqual(scaled[0], 2.5, accuracy: 1e-5)
        XCTAssertEqual(scaled[1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(scaled[2], 7.5, accuracy: 1e-5)
    }
    
    // MARK: - Matrix Operations
    
    func testMatrixVectorMultiply() async throws {
        let matrix: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        let vector: [Float] = [1.0, 2.0, 3.0]
        
        let result = try await engine.matrixVectorMultiply(
            matrix: matrix,
            vector: vector
        )
        
        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3, 7*1 + 8*2 + 9*3]
        //         = [1+4+9, 4+10+18, 7+16+27] = [14, 32, 50]
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 14.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 32.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 50.0, accuracy: 1e-5)
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceSmallVectors() async throws {
        let vectorA = Array(repeating: Float(1.0), count: 128)
        let vectorB = Array(repeating: Float(2.0), count: 128)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = try await engine.euclideanDistance(vectorA, vectorB)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        print("Small vector performance: \(elapsed * 1000)ms for 100 operations")
        print("Throughput: \(100.0 / elapsed) ops/sec")
    }
    
    func testPerformanceLargeVectors() async throws {
        let vectorA = Array(repeating: Float(1.0), count: 4096)
        let vectorB = Array(repeating: Float(2.0), count: 4096)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try await engine.euclideanDistance(vectorA, vectorB)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        print("Large vector performance: \(elapsed * 1000)ms for 10 operations")
        print("Throughput: \(10.0 / elapsed) ops/sec")
    }
    
    // MARK: - Edge Cases
    
    func testZeroVector() async throws {
        let zero: [Float] = [0.0, 0.0, 0.0]
        let vector: [Float] = [1.0, 2.0, 3.0]
        
        // Distance from zero
        let distance = try await engine.euclideanDistance(zero, vector)
        XCTAssertEqual(distance, sqrt(14.0), accuracy: 1e-5)
        
        // Dot product with zero
        let dot = try await engine.dotProduct(zero, vector)
        XCTAssertEqual(dot, 0.0, accuracy: 1e-5)
    }
    
    func testSingleElementVectors() async throws {
        let a: [Float] = [3.0]
        let b: [Float] = [4.0]
        
        let distance = try await engine.euclideanDistance(a, b)
        XCTAssertEqual(distance, 1.0, accuracy: 1e-5)
        
        let dot = try await engine.dotProduct(a, b)
        XCTAssertEqual(dot, 12.0, accuracy: 1e-5)
    }
    
    func testDimensionMismatch() async throws {
        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [4.0, 5.0]
        
        do {
            _ = try await engine.euclideanDistance(vectorA, vectorB)
            XCTFail("Should throw dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected
        }
    }
    
    // MARK: - Statistics
    
    func testEngineStatistics() async throws {
        // Perform some operations
        let v1 = [Float](repeating: 1.0, count: 100)
        let v2 = [Float](repeating: 2.0, count: 100)
        
        for _ in 0..<10 {
            _ = try await engine.euclideanDistance(v1, v2)
            _ = try await engine.dotProduct(v1, v2)
        }
        
        let stats = await engine.getStatistics()
        
        XCTAssertGreaterThan(stats.operationCount, 0)
        XCTAssertGreaterThan(stats.computeTime, 0)
        XCTAssertGreaterThanOrEqual(stats.bufferHitRate, 0)
        XCTAssertLessThanOrEqual(stats.bufferHitRate, 1.0)
        
        print("Engine Statistics:")
        print("  Operations: \(stats.operationCount)")
        print("  Compute time: \(stats.computeTime)s")
        print("  Buffer hit rate: \(stats.bufferHitRate * 100)%")
        print("  Memory utilization: \(stats.memoryUtilization * 100)%")
    }
}