//
//  SIMDOptimizerTests.swift
//  VectorAccelerateTests
//
//  Tests for SIMDOptimizer
//

import XCTest
@testable import VectorAccelerate

final class SIMDOptimizerTests: XCTestCase {
    var optimizer: SIMDOptimizer!
    
    override func setUp() {
        super.setUp()
        optimizer = SIMDOptimizer()
    }
    
    override func tearDown() {
        optimizer = nil
        super.tearDown()
    }
    
    // MARK: - Capability Detection Tests
    
    func testCapabilityDetection() {
        let capabilities = optimizer.capabilities
        
        // Basic validation
        XCTAssertGreaterThan(capabilities.maxVectorWidth, 0)
        XCTAssertGreaterThan(capabilities.optimalWidth, 0)
        XCTAssertLessThanOrEqual(capabilities.optimalWidth, capabilities.maxVectorWidth)
        
        // Cache sizes should be reasonable
        XCTAssertGreaterThan(capabilities.l1CacheSize, 0)
        XCTAssertGreaterThan(capabilities.l2CacheSize, 0)
        XCTAssertGreaterThan(capabilities.l2CacheSize, capabilities.l1CacheSize)
        
        // Platform-specific checks
        #if arch(arm64)
        XCTAssertTrue(capabilities.supportsNEON)
        XCTAssertFalse(capabilities.supportsAVX)
        XCTAssertFalse(capabilities.supportsAVX512)
        XCTAssertEqual(capabilities.maxVectorWidth, 4)  // NEON is 128-bit
        #elseif arch(x86_64)
        XCTAssertFalse(capabilities.supportsNEON)
        // AVX support depends on CPU
        #endif
    }
    
    func testStaticDetection() {
        let capabilities = SIMDCapabilities.detect()
        XCTAssertGreaterThan(capabilities.maxVectorWidth, 0)
        XCTAssertGreaterThan(capabilities.optimalWidth, 0)
    }
    
    // MARK: - Width Selection Tests
    
    func testSelectOptimalWidthSmallData() {
        // For small data (< 64 elements), should use small width
        let width = optimizer.selectOptimalWidth(for: 32)
        XCTAssertLessThanOrEqual(width, 4)
    }
    
    func testSelectOptimalWidthMediumData() {
        // For medium data (< 10000 elements), should use optimal width
        let width = optimizer.selectOptimalWidth(for: 5000)
        XCTAssertEqual(width, optimizer.capabilities.optimalWidth)
    }
    
    func testSelectOptimalWidthLargeData() {
        // For very large data, considers cache effects
        let largeSize = (optimizer.capabilities.l2CacheSize / MemoryLayout<Float>.stride) + 1000
        let width = optimizer.selectOptimalWidth(for: largeSize)
        XCTAssertLessThanOrEqual(width, 8)
    }
    
    func testSelectOptimalWidthEdgeCases() {
        XCTAssertGreaterThan(optimizer.selectOptimalWidth(for: 0), 0)
        XCTAssertGreaterThan(optimizer.selectOptimalWidth(for: 1), 0)
        XCTAssertGreaterThan(optimizer.selectOptimalWidth(for: 63), 0)
        XCTAssertGreaterThan(optimizer.selectOptimalWidth(for: 64), 0)
    }
    
    // MARK: - Dot Product Tests
    
    func testAdaptiveDotProductCorrectness() {
        let a: [Float] = [1, 2, 3, 4, 5]
        let b: [Float] = [2, 3, 4, 5, 6]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        
        // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        XCTAssertEqual(result, 70.0, accuracy: 0.001)
    }
    
    func testAdaptiveDotProductZeroVectors() {
        let a: [Float] = [0, 0, 0, 0]
        let b: [Float] = [1, 2, 3, 4]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        XCTAssertEqual(result, 0.0, accuracy: 0.001)
    }
    
    func testAdaptiveDotProductNegativeValues() {
        let a: [Float] = [1, -2, 3, -4]
        let b: [Float] = [2, 3, -4, 5]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        
        // Expected: 1*2 + (-2)*3 + 3*(-4) + (-4)*5 = 2 - 6 - 12 - 20 = -36
        XCTAssertEqual(result, -36.0, accuracy: 0.001)
    }
    
    func testAdaptiveDotProductLargeVectors() {
        let size = 1000
        let a = [Float](repeating: 1.0, count: size)
        let b = [Float](repeating: 2.0, count: size)
        
        let result = optimizer.adaptiveDotProduct(a, b)
        
        // Expected: 1000 * (1 * 2) = 2000
        XCTAssertEqual(result, 2000.0, accuracy: 0.001)
    }
    
    func testAdaptiveDotProductVaryingSizes() {
        // Test with sizes that aren't multiples of vector width
        for size in [3, 7, 13, 17, 31, 63, 127, 255] {
            let a = [Float](repeating: 1.0, count: size)
            let b = [Float](repeating: 1.0, count: size)
            
            let result = optimizer.adaptiveDotProduct(a, b)
            XCTAssertEqual(result, Float(size), accuracy: 0.001,
                          "Failed for size \(size)")
        }
    }
    
    // MARK: - Benchmark Tests
    
    func testBenchmarkAndOptimize() {
        let optimalWidth = optimizer.benchmarkAndOptimize(dataSize: 1000)
        
        // Should return a valid width
        XCTAssertTrue([4, 8, 16].contains(optimalWidth))
        // Note: benchmarkAndOptimize may test widths beyond maxVectorWidth for comparison
        // So we just verify it returns a reasonable value
        XCTAssertLessThanOrEqual(optimalWidth, 16)
    }
    
    func testBenchmarkSmallData() {
        let optimalWidth = optimizer.benchmarkAndOptimize(dataSize: 64)
        XCTAssertGreaterThan(optimalWidth, 0)
        // The benchmark tests [4, 8, optimalWidth] so may return 8 even on ARM
        XCTAssertLessThanOrEqual(optimalWidth, 16)
    }
    
    func testBenchmarkLargeData() {
        let optimalWidth = optimizer.benchmarkAndOptimize(dataSize: 10000)
        XCTAssertGreaterThan(optimalWidth, 0)
        // The benchmark tests [4, 8, optimalWidth] so may return 8 even on ARM
        XCTAssertLessThanOrEqual(optimalWidth, 16)
    }
    
    // MARK: - Performance Stats Tests
    
    func testGetPerformanceStats() {
        let stats = optimizer.getPerformanceStats()
        
        // Should contain expected information
        XCTAssertTrue(stats.contains("SIMD Capabilities"))
        XCTAssertTrue(stats.contains("Max Vector Width"))
        XCTAssertTrue(stats.contains("Optimal Width"))
        XCTAssertTrue(stats.contains("L1 Cache"))
        XCTAssertTrue(stats.contains("L2 Cache"))
        
        // Platform-specific checks
        #if arch(arm64)
        XCTAssertTrue(stats.contains("NEON: true"))
        XCTAssertTrue(stats.contains("AVX: false"))
        #endif
    }
    
    // MARK: - Performance Tests
    
    func testDotProductPerformanceSmall() {
        let a = [Float](repeating: 1.0, count: 100)
        let b = [Float](repeating: 2.0, count: 100)
        
        measure {
            for _ in 0..<1000 {
                _ = optimizer.adaptiveDotProduct(a, b)
            }
        }
    }
    
    func testDotProductPerformanceMedium() {
        let a = [Float](repeating: 1.0, count: 1000)
        let b = [Float](repeating: 2.0, count: 1000)
        
        measure {
            for _ in 0..<100 {
                _ = optimizer.adaptiveDotProduct(a, b)
            }
        }
    }
    
    func testDotProductPerformanceLarge() {
        let a = [Float](repeating: 1.0, count: 10000)
        let b = [Float](repeating: 2.0, count: 10000)
        
        measure {
            for _ in 0..<10 {
                _ = optimizer.adaptiveDotProduct(a, b)
            }
        }
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptyVectors() {
        let a: [Float] = []
        let b: [Float] = []
        
        let result = optimizer.adaptiveDotProduct(a, b)
        XCTAssertEqual(result, 0.0)
    }
    
    func testSingleElementVectors() {
        let a: [Float] = [3.0]
        let b: [Float] = [4.0]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        XCTAssertEqual(result, 12.0, accuracy: 0.001)
    }
    
    func testVerySmallValues() {
        let a: [Float] = [1e-10, 1e-10, 1e-10]
        let b: [Float] = [1e-10, 1e-10, 1e-10]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        XCTAssertGreaterThan(result, 0)
        XCTAssertLessThan(result, 1e-15)
    }
    
    func testVeryLargeValues() {
        let a: [Float] = [1e10, 1e10]
        let b: [Float] = [1e10, 1e10]
        
        let result = optimizer.adaptiveDotProduct(a, b)
        XCTAssertEqual(result, 2e20, accuracy: 1e18)
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentBenchmarking() {
        let expectation = XCTestExpectation(description: "Concurrent benchmarking")
        expectation.expectedFulfillmentCount = 5
        
        for i in 0..<5 {
            DispatchQueue.global().async {
                let width = self.optimizer.benchmarkAndOptimize(dataSize: 100 * (i + 1))
                XCTAssertGreaterThan(width, 0)
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
}