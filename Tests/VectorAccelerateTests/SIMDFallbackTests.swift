//
//  SIMDFallbackTests.swift
//  VectorAccelerateTests
//
//  Tests for SIMDFallback
//

import XCTest
@testable import VectorAccelerate

final class SIMDFallbackTests: XCTestCase {
    var fallback: SIMDFallback!
    
    override func setUp() {
        super.setUp()
        fallback = SIMDFallback()
    }
    
    override func tearDown() {
        fallback = nil
        super.tearDown()
    }
    
    // MARK: - Configuration Tests
    
    func testDefaultConfiguration() {
        let config = SIMDConfiguration.default
        XCTAssertEqual(config.vectorWidth, 8)
        XCTAssertTrue(config.useAccelerate)
        XCTAssertEqual(config.parallelThreshold, 1024)
        XCTAssertTrue(config.enablePrefetch)
    }
    
    func testPerformanceConfiguration() {
        let config = SIMDConfiguration.performance
        XCTAssertEqual(config.vectorWidth, 16)
        XCTAssertEqual(config.parallelThreshold, 512)
        XCTAssertTrue(config.enablePrefetch)
    }
    
    func testCustomConfiguration() {
        let config = SIMDConfiguration(
            vectorWidth: 4,
            useAccelerate: false,
            parallelThreshold: 2048,
            enablePrefetch: false
        )
        
        let customFallback = SIMDFallback(configuration: config)
        XCTAssertNotNil(customFallback)
    }
    
    // MARK: - Dot Product Tests
    
    func testDotProductCorrectness() async throws {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [2, 3, 4, 5]
        
        let result = try await fallback.dotProduct(a, b)
        
        // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        XCTAssertEqual(result, 40.0, accuracy: 0.001)
    }
    
    func testDotProductWithZeros() async throws {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [1, 2, 3]
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }
    
    func testDotProductWithNegatives() async throws {
        let a: [Float] = [1, -2, 3]
        let b: [Float] = [4, 5, -6]
        
        let result = try await fallback.dotProduct(a, b)
        
        // Expected: 1*4 + (-2)*5 + 3*(-6) = 4 - 10 - 18 = -24
        XCTAssertEqual(result, -24.0, accuracy: 0.001)
    }
    
    func testDotProductLargeVectors() async throws {
        let size = 1000
        let a = [Float](repeating: 1.0, count: size)
        let b = [Float](repeating: 2.0, count: size)
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertEqual(result, 2000.0, accuracy: 0.001)
    }
    
    func testDotProductDimensionMismatch() async {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2]
        
        do {
            _ = try await fallback.dotProduct(a, b)
            XCTFail("Should throw dimension mismatch error")
        } catch AccelerationError.dimensionMismatch(let expected, let actual) {
            XCTAssertEqual(expected, 3)
            XCTAssertEqual(actual, 2)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    // MARK: - Euclidean Distance Tests
    
    func testEuclideanDistanceIdenticalVectors() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [1, 2, 3]
        
        let result = try await fallback.euclideanDistance(a, b)
        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }
    
    func testEuclideanDistanceOrthogonal() async throws {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        
        let result = try await fallback.euclideanDistance(a, b)
        XCTAssertEqual(result, sqrt(2.0), accuracy: 0.001)
    }
    
    func testEuclideanDistanceKnownValues() async throws {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 6, 8]
        
        let result = try await fallback.euclideanDistance(a, b)
        
        // Distance = sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(9 + 16 + 25) = sqrt(50)
        XCTAssertEqual(result, sqrt(50.0), accuracy: 0.001)
    }
    
    // MARK: - Normalize Tests
    
    func testNormalizeVector() async {
        let vector: [Float] = [3, 4, 0]  // Length = 5
        
        let normalized = await fallback.normalize(vector)
        
        XCTAssertEqual(normalized.count, 3)
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.001)
        XCTAssertEqual(normalized[2], 0.0, accuracy: 0.001)
        
        // Check that length is 1
        let length = sqrt(normalized.map { $0 * $0 }.reduce(0, +))
        XCTAssertEqual(length, 1.0, accuracy: 0.001)
    }
    
    func testNormalizeZeroVector() async {
        let vector: [Float] = [0, 0, 0]
        
        let normalized = await fallback.normalize(vector)
        
        // Zero vector should remain zero
        XCTAssertEqual(normalized, [0, 0, 0])
    }
    
    func testNormalizeSingleElement() async {
        let vector: [Float] = [5.0]
        
        let normalized = await fallback.normalize(vector)
        
        XCTAssertEqual(normalized.count, 1)
        XCTAssertEqual(abs(normalized[0]), 1.0, accuracy: 0.001)
    }
    
    // MARK: - Configuration-based Tests
    
    func testWithAccelerateDisabled() async throws {
        let config = SIMDConfiguration(useAccelerate: false)
        let manualFallback = SIMDFallback(configuration: config)
        
        let a: [Float] = [1, 2, 3, 4, 5]
        let b: [Float] = [5, 4, 3, 2, 1]
        
        let result = try await manualFallback.dotProduct(a, b)
        
        // Expected: 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        XCTAssertEqual(result, 35.0, accuracy: 0.001)
    }
    
    func testVectorWidthVariations() async throws {
        for width in [4, 8, 16] {
            let config = SIMDConfiguration(
                vectorWidth: width,
                useAccelerate: false
            )
            let customFallback = SIMDFallback(configuration: config)
            
            let a = [Float](repeating: 1.0, count: 100)
            let b = [Float](repeating: 2.0, count: 100)
            
            let result = try await customFallback.dotProduct(a, b)
            XCTAssertEqual(result, 200.0, accuracy: 0.001,
                          "Failed for vector width \(width)")
        }
    }
    
    // MARK: - Performance Tests
    
    func testDotProductPerformanceSmall() async throws {
        let a = [Float](repeating: 1.0, count: 128)
        let b = [Float](repeating: 2.0, count: 128)
        
        // Warm up
        _ = try await fallback.dotProduct(a, b)
        
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<1000 {
            _ = try await fallback.dotProduct(a, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Should be very fast for small vectors
        XCTAssertLessThan(elapsed, 1.0)
    }
    
    func testDotProductPerformanceLarge() async throws {
        let a = [Float](repeating: 1.0, count: 10000)
        let b = [Float](repeating: 2.0, count: 10000)
        
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = try await fallback.dotProduct(a, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Should still be reasonably fast
        XCTAssertLessThan(elapsed, 2.0)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyVectors() async throws {
        let a: [Float] = []
        let b: [Float] = []
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertEqual(result, 0.0)
    }
    
    func testSingleElementVectors() async throws {
        let a: [Float] = [3.0]
        let b: [Float] = [4.0]
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertEqual(result, 12.0, accuracy: 0.001)
    }
    
    func testVerySmallValues() async throws {
        let a: [Float] = [1e-10, 1e-10]
        let b: [Float] = [1e-10, 1e-10]
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertGreaterThan(result, 0)
        XCTAssertLessThan(result, 1e-15)
    }
    
    func testVeryLargeValues() async throws {
        let a: [Float] = [1e10, 1e10]
        let b: [Float] = [1e10, 1e10]
        
        let result = try await fallback.dotProduct(a, b)
        XCTAssertEqual(result, 2e20, accuracy: 1e18)
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentOperations() async throws {
        let fallback = self.fallback!
        
        await withTaskGroup(of: Float.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let a = [Float](repeating: Float(i), count: 100)
                    let b = [Float](repeating: 1.0, count: 100)
                    
                    do {
                        return try await fallback.dotProduct(a, b)
                    } catch {
                        XCTFail("Concurrent operation failed: \(error)")
                        return 0
                    }
                }
            }
            
            var results: [Float] = []
            for await result in group {
                results.append(result)
            }
            
            XCTAssertEqual(results.count, 10)
        }
    }
}