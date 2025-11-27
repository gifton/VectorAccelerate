//
//  DistanceMetricsTests.swift
//  VectorAccelerateTests
//
//  Tests for DistanceMetrics (ComputeEngine extensions)
//

import XCTest
@testable import VectorAccelerate
import VectorCore
import Metal

final class DistanceMetricsTests: XCTestCase {
    var context: VectorAccelerate.MetalContext!
    var engine: ComputeEngine!
    
    override func setUp() async throws {
        try await super.setUp()
        context = try await VectorAccelerate.MetalContext()
        engine = try await ComputeEngine(context: context)
    }
    
    override func tearDown() async throws {
        await context?.cleanup()
        context = nil
        engine = nil
        try await super.tearDown()
    }
    
    // MARK: - Manhattan Distance Tests
    
    func testManhattanDistanceIdenticalVectors() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2, 3]
        
        let distance = try await engine.manhattanDistance(a, b)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    func testManhattanDistanceKnownValues() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        let distance = try await engine.manhattanDistance(a, b)
        
        // Distance = |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        XCTAssertEqual(distance, 12.0, accuracy: 0.001)
    }
    
    func testManhattanDistanceNegativeValues() async throws {
        let a: [Float] = [-1, -2, -3]
        let b: [Float] = [1, 2, 3]
        
        let distance = try await engine.manhattanDistance(a, b)
        
        // Distance = |1-(-1)| + |2-(-2)| + |3-(-3)| = 2 + 4 + 6 = 12
        XCTAssertEqual(distance, 12.0, accuracy: 0.001)
    }
    
    func testManhattanDistanceMixedValues() async throws {
        let a: [Float] = [0, 5, -3]
        let b: [Float] = [3, -2, 4]
        
        let distance = try await engine.manhattanDistance(a, b)
        
        // Distance = |3-0| + |-2-5| + |4-(-3)| = 3 + 7 + 7 = 17
        XCTAssertEqual(distance, 17.0, accuracy: 0.001)
    }
    
    func testManhattanDistanceDimensionMismatch() async {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2]
        
        do {
            _ = try await engine.manhattanDistance(a, b)
            XCTFail("Should throw dimension mismatch error")
        } catch {
            // Expected error
            XCTAssertTrue(error is VectorError)
        }
    }
    
    // MARK: - Chebyshev Distance Tests
    
    func testChebyshevDistanceIdenticalVectors() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2, 3]
        
        let distance = try await engine.chebyshevDistance(a, b)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    func testChebyshevDistanceKnownValues() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        let distance = try await engine.chebyshevDistance(a, b)
        
        // Distance = max(|4-1|, |6-2|, |8-3|) = max(3, 4, 5) = 5
        XCTAssertEqual(distance, 5.0, accuracy: 0.001)
    }
    
    func testChebyshevDistanceNegativeValues() async throws {
        let a: [Float] = [-5, 0, 5]
        let b: [Float] = [5, 0, -5]
        
        let distance = try await engine.chebyshevDistance(a, b)
        
        // Distance = max(|5-(-5)|, |0-0|, |-5-5|) = max(10, 0, 10) = 10
        XCTAssertEqual(distance, 10.0, accuracy: 0.001)
    }
    
    func testChebyshevDistanceSingleDimension() async throws {
        let a: [Float] = [7]
        let b: [Float] = [3]
        
        let distance = try await engine.chebyshevDistance(a, b)
        XCTAssertEqual(distance, 4.0, accuracy: 0.001)
    }
    
    // MARK: - Minkowski Distance Tests
    
    func testMinkowskiDistanceP1Manhattan() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        let distance = try await engine.minkowskiDistance(a, b, p: 1.0)
        
        // p=1 is Manhattan distance: |4-1| + |6-2| + |8-3| = 12
        XCTAssertEqual(distance, 12.0, accuracy: 0.001)
    }
    
    func testMinkowskiDistanceP2Euclidean() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        let distance = try await engine.minkowskiDistance(a, b, p: 2.0)
        
        // p=2 is Euclidean distance: sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(50)
        XCTAssertEqual(distance, sqrt(50.0), accuracy: 0.001)
    }
    
    func testMinkowskiDistancePInfinity() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        // Large p approximates Chebyshev (L∞) distance
        let distance = try await engine.minkowskiDistance(a, b, p: 100.0)
        
        // Should be close to max(|4-1|, |6-2|, |8-3|) = 5
        XCTAssertEqual(distance, 5.0, accuracy: 0.1)
    }
    
    func testMinkowskiDistanceFractionalP() async throws {
        let a: [Float] = [0, 0]
        let b: [Float] = [3, 4]
        
        let distance = try await engine.minkowskiDistance(a, b, p: 0.5)
        
        // For p < 1, this is not a proper metric but calculation should still work
        // Distance = (|3-0|^0.5 + |4-0|^0.5)^2 = (√3 + 2)^2
        let expected = pow(sqrt(3) + 2, 2)
        XCTAssertEqual(distance, Float(expected), accuracy: 0.01)
    }
    
    func testMinkowskiDistanceInvalidP() async {
        let a: [Float] = [1, 2]
        let b: [Float] = [3, 4]
        
        do {
            _ = try await engine.minkowskiDistance(a, b, p: -1.0)
            XCTFail("Should throw error for negative p")
        } catch {
            // Expected error
            XCTAssertTrue(error is VectorError)
        }
    }
    
    func testMinkowskiDistanceZeroP() async {
        let a: [Float] = [1, 2]
        let b: [Float] = [3, 4]
        
        do {
            _ = try await engine.minkowskiDistance(a, b, p: 0.0)
            XCTFail("Should throw error for p=0")
        } catch {
            // Expected error
            XCTAssertTrue(error is VectorError)
        }
    }
    
    // MARK: - Hamming Distance Tests
    
    func testHammingDistanceIdenticalBinary() async throws {
        let a: [Float] = [1, 0, 1, 0]
        let b: [Float] = [1, 0, 1, 0]
        
        let distance = try await engine.hammingDistance(a, b)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    func testHammingDistanceOppositeBinary() async throws {
        let a: [Float] = [1, 0, 1, 0]
        let b: [Float] = [0, 1, 0, 1]
        
        let distance = try await engine.hammingDistance(a, b)
        XCTAssertEqual(distance, 4.0, accuracy: 0.001)  // All bits different
    }
    
    func testHammingDistancePartialMatch() async throws {
        let a: [Float] = [1, 1, 0, 0, 1]
        let b: [Float] = [1, 0, 0, 1, 1]
        
        let distance = try await engine.hammingDistance(a, b)
        XCTAssertEqual(distance, 2.0, accuracy: 0.001)  // 2 bits different
    }
    
    func testHammingDistanceNonBinary() async throws {
        // Non-zero values are treated as 1
        let a: [Float] = [5, 0, -3, 0]
        let b: [Float] = [2, 0, 7, 1]
        
        let distance = try await engine.hammingDistance(a, b)
        
        // Binary interpretation: [1,0,1,0] vs [1,0,1,1] = 1 bit different
        XCTAssertEqual(distance, 1.0, accuracy: 0.001)
    }
    
    func testHammingDistanceAllZeros() async throws {
        let a: [Float] = [0, 0, 0, 0]
        let b: [Float] = [0, 0, 0, 0]
        
        let distance = try await engine.hammingDistance(a, b)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    // MARK: - Performance Tests
    
    func testManhattanDistancePerformance() async throws {
        let dimension = 1000
        let a = [Float](repeating: 1.0, count: dimension)
        let b = [Float](repeating: 2.0, count: dimension)
        
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = try await engine.manhattanDistance(a, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Should be fast with GPU acceleration
        XCTAssertLessThan(elapsed, 2.0)
    }
    
    func testChebyshevDistancePerformance() async throws {
        let dimension = 1000
        let a = (0..<dimension).map { Float($0) }
        let b = (0..<dimension).map { Float($0 + 1) }
        
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = try await engine.chebyshevDistance(a, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        XCTAssertLessThan(elapsed, 2.0)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyVectors() async {
        let a: [Float] = []
        let b: [Float] = []
        
        do {
            _ = try await engine.manhattanDistance(a, b)
            // Some implementations might handle empty vectors
        } catch {
            // Or throw an error - both are acceptable
            XCTAssertTrue(error is VectorError)
        }
    }
    
    func testSingleElementVectors() async throws {
        let a: [Float] = [5]
        let b: [Float] = [3]
        
        let manhattan = try await engine.manhattanDistance(a, b)
        XCTAssertEqual(manhattan, 2.0, accuracy: 0.001)
        
        let chebyshev = try await engine.chebyshevDistance(a, b)
        XCTAssertEqual(chebyshev, 2.0, accuracy: 0.001)
        
        let minkowski = try await engine.minkowskiDistance(a, b, p: 3.0)
        XCTAssertEqual(minkowski, 2.0, accuracy: 0.001)
    }
    
    func testLargeVectors() async throws {
        let dimension = 10000
        let a = [Float](repeating: 0.0, count: dimension)
        let b = [Float](repeating: 1.0, count: dimension)
        
        let manhattan = try await engine.manhattanDistance(a, b)
        XCTAssertEqual(manhattan, Float(dimension), accuracy: 1.0)
        
        let chebyshev = try await engine.chebyshevDistance(a, b)
        XCTAssertEqual(chebyshev, 1.0, accuracy: 0.001)
    }
    
    // MARK: - Numerical Stability Tests
    
    func testVerySmallDifferences() async throws {
        let a: [Float] = [1.0, 1.0]
        let b: [Float] = [1.0 + 1e-7, 1.0 + 1e-7]
        
        let manhattan = try await engine.manhattanDistance(a, b)
        XCTAssertGreaterThan(manhattan, 0)
        XCTAssertLessThan(manhattan, 1e-5)
    }
    
    func testVeryLargeDifferences() async throws {
        let a: [Float] = [1e10, -1e10]
        let b: [Float] = [-1e10, 1e10]
        
        let manhattan = try await engine.manhattanDistance(a, b)
        XCTAssertEqual(manhattan, 4e10, accuracy: 1e8)
        
        let chebyshev = try await engine.chebyshevDistance(a, b)
        XCTAssertEqual(chebyshev, 2e10, accuracy: 1e8)
    }
}