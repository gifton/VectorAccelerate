//
//  QuantizationEngineTests.swift
//  VectorAccelerateTests
//
//  Comprehensive tests for QuantizationEngine vector compression operations
//

import XCTest
@testable import VectorAccelerate
import Foundation

final class QuantizationEngineTests: XCTestCase {
    
    var engine: QuantizationEngine!
    var testVectors: [[Float]]!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Generate test vectors with various characteristics
        testVectors = generateTestVectors()
        
        let config = QuantizationConfiguration(
            method: .scalar(bits: 8),
            preserveNorm: false,
            useSymmetric: true,
            useGPU: MetalDevice.isAvailable
        )
        
        engine = await QuantizationEngine(configuration: config)
    }
    
    override func tearDown() async throws {
        engine = nil
        testVectors = nil
        try await super.tearDown()
    }
    
    // MARK: - Test Data Generation
    
    private func generateTestVectors() -> [[Float]] {
        var vectors: [[Float]] = []
        
        // Various test vector types for comprehensive testing
        let vectorTypes: [(name: String, generator: () -> [Float])] = [
            ("uniform", { Array(repeating: 1.0, count: 128) }),
            ("random_normal", { (0..<128).map { _ in Float.random(in: -1...1) } }),
            ("large_values", { (0..<128).map { _ in Float.random(in: -100...100) } }),
            ("small_values", { (0..<128).map { _ in Float.random(in: -0.01...0.01) } }),
            ("sparse", { (0..<128).map { i in i % 10 == 0 ? Float.random(in: -1...1) : 0.0 } }),
            ("sequential", { (0..<128).map { Float($0) / 128.0 } }),
            ("sinusoidal", { (0..<128).map { sin(Float($0) * Float.pi / 64.0) } })
        ]
        
        for (_, generator) in vectorTypes {
            for _ in 0..<10 {  // Multiple instances of each type
                vectors.append(generator())
            }
        }
        
        return vectors
    }
    
    private func generateTrainingData(count: Int, dimension: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }
    
    // MARK: - Initialization Tests
    
    func testInitializationWithDefaultConfig() async throws {
        let config = QuantizationConfiguration()
        let testEngine = await QuantizationEngine(configuration: config)
        
        let metrics = await testEngine.getPerformanceMetrics()
        XCTAssertEqual(metrics.vectorsQuantized, 0)
        XCTAssertEqual(metrics.averageCompressionRatio, 0)
    }
    
    func testInitializationWithCustomConfig() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 4, bitsPerCode: 8),
            preserveNorm: true,
            useSymmetric: false,
            useGPU: false
        )
        
        let testEngine = await QuantizationEngine(configuration: config)
        XCTAssertNotNil(testEngine)
    }
    
    func testInitializationWithGPUContext() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await MetalContext()
        let config = QuantizationConfiguration(useGPU: true)
        let testEngine = await QuantizationEngine(configuration: config, context: context)
        
        XCTAssertNotNil(testEngine)
    }
    
    // MARK: - Scalar Quantization Tests
    
    func testScalarQuantization8Bit() async throws {
        let testVector = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: testVector, bits: 8)
        
        // Verify quantized vector properties
        XCTAssertEqual(quantized.dimensions, testVector.count)
        XCTAssertEqual(quantized.scheme, .scalar(bits: 8))
        XCTAssertNotNil(quantized.metadata)
        
        // Check compression ratio
        let expectedSize = testVector.count  // 1 byte per element for 8-bit
        XCTAssertEqual(quantized.data.count, expectedSize)
        
        // Verify compression ratio calculation
        let ratio = quantized.compressionRatio
        XCTAssertEqual(ratio, Float(testVector.count * 4) / Float(quantized.data.count))
    }
    
    func testScalarQuantization4Bit() async throws {
        let testVector = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: testVector, bits: 4)
        
        XCTAssertEqual(quantized.dimensions, testVector.count)
        XCTAssertEqual(quantized.scheme, .scalar(bits: 4))
        
        // 4-bit should use half the bytes (plus padding)
        let expectedSize = (testVector.count + 1) / 2
        XCTAssertEqual(quantized.data.count, expectedSize)
        
        // Better compression ratio than 8-bit
        XCTAssertGreaterThan(quantized.compressionRatio, 4.0)
    }
    
    func testScalarQuantization16Bit() async throws {
        let testVector = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: testVector, bits: 16)
        
        XCTAssertEqual(quantized.dimensions, testVector.count)
        XCTAssertEqual(quantized.scheme, .scalar(bits: 16))
        
        // 16-bit should use 2 bytes per element
        let expectedSize = testVector.count * 2
        XCTAssertEqual(quantized.data.count, expectedSize)
    }
    
    func testScalarQuantizationInvalidBits() async throws {
        let testVector = testVectors[0]
        
        do {
            _ = try await engine!.scalarQuantize(vector: testVector, bits: 7)
            XCTFail("Should have thrown error for invalid bit width")
        } catch AccelerationError.unsupportedOperation(let message) {
            XCTAssertTrue(message.contains("4, 8, or 16 bits"))
        }
    }
    
    func testScalarDequantization8Bit() async throws {
        let original = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: original, bits: 8)
        let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, original.count)
        
        // Check reconstruction quality (should be reasonably close)
        let error = try await engine!.measureError(original: original, quantized: quantized)
        XCTAssertLessThan(error, 0.1)  // RMSE should be reasonably small
    }
    
    func testScalarDequantization4Bit() async throws {
        let original = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: original, bits: 4)
        let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, original.count)
        
        // 4-bit quantization has higher error but should still be reasonable
        let error = try await engine!.measureError(original: original, quantized: quantized)
        XCTAssertLessThan(error, 0.3)
    }
    
    func testSymmetricVsAsymmetricQuantization() async throws {
        let testVector = testVectors[0]
        
        // Symmetric quantization
        let symmetricConfig = QuantizationConfiguration(
            method: .scalar(bits: 8),
            useSymmetric: true,
            useGPU: false
        )
        let symmetricEngine = await QuantizationEngine(configuration: symmetricConfig)
        let symmetricQuantized = try await symmetricEngine.scalarQuantize(vector: testVector, bits: 8)
        let symmetricReconstructed = try await symmetricEngine.scalarDequantize(quantized: symmetricQuantized)
        
        // Asymmetric quantization
        let asymmetricConfig = QuantizationConfiguration(
            method: .scalar(bits: 8),
            useSymmetric: false,
            useGPU: false
        )
        let asymmetricEngine = await QuantizationEngine(configuration: asymmetricConfig)
        let asymmetricQuantized = try await asymmetricEngine.scalarQuantize(vector: testVector, bits: 8)
        let asymmetricReconstructed = try await asymmetricEngine.scalarDequantize(quantized: asymmetricQuantized)
        
        // Both should reconstruct reasonably well
        XCTAssertEqual(symmetricReconstructed.count, testVector.count)
        XCTAssertEqual(asymmetricReconstructed.count, testVector.count)
        
        // Verify different metadata
        XCTAssertNotNil(symmetricQuantized.metadata)
        XCTAssertNotNil(asymmetricQuantized.metadata)
    }
    
    // MARK: - Product Quantization Tests
    
    func testProductQuantizationTraining() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 4, bitsPerCode: 8),
            useGPU: false
        )
        let pqEngine = await QuantizationEngine(configuration: config)
        
        let trainingData = generateTrainingData(count: 500, dimension: 128)
        
        try await pqEngine.trainProductQuantization(
            trainingData: trainingData,
            numCodebooks: 4,
            centroids: 16
        )
        
        // Should be able to quantize after training
        let testVector = trainingData[0]
        let quantized = try await pqEngine.productQuantize(vector: testVector)
        
        XCTAssertEqual(quantized.dimensions, testVector.count)
        XCTAssertEqual(quantized.scheme, .product(codebooks: 4, bitsPerCode: 4)) // log2(16) = 4 bits
        XCTAssertNotNil(quantized.metadata)
    }
    
    func testProductQuantizationWithoutTraining() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 4, bitsPerCode: 8),
            useGPU: false
        )
        let pqEngine = await QuantizationEngine(configuration: config)
        
        let testVector = testVectors[0]
        
        do {
            _ = try await pqEngine.productQuantize(vector: testVector)
            XCTFail("Should have thrown error for untrained product quantization")
        } catch AccelerationError.unsupportedOperation(let message) {
            XCTAssertTrue(message.contains("not trained"))
        }
    }
    
    func testProductQuantizationDimensionMismatch() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 4, bitsPerCode: 8),
            useGPU: false
        )
        let pqEngine = await QuantizationEngine(configuration: config)
        
        // Dimension not divisible by number of codebooks
        let trainingData = generateTrainingData(count: 100, dimension: 127)  // 127 % 4 != 0
        
        do {
            try await pqEngine.trainProductQuantization(
                trainingData: trainingData,
                numCodebooks: 4,
                centroids: 16
            )
            XCTFail("Should have thrown error for dimension mismatch")
        } catch AccelerationError.unsupportedOperation(let message) {
            XCTAssertTrue(message.contains("divisible"))
        }
    }
    
    func testProductQuantizationRoundtrip() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 8, bitsPerCode: 8),
            useGPU: false
        )
        let pqEngine = await QuantizationEngine(configuration: config)
        
        let trainingData = generateTrainingData(count: 200, dimension: 128)
        
        try await pqEngine.trainProductQuantization(
            trainingData: trainingData,
            numCodebooks: 8,
            centroids: 256
        )
        
        let testVector = trainingData[0]
        let quantized = try await pqEngine.productQuantize(vector: testVector)
        let reconstructed = try await pqEngine.productDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, testVector.count)
        
        // Product quantization typically has higher error than scalar
        let error = try await pqEngine.measureError(original: testVector, quantized: quantized)
        XCTAssertLessThan(error, 1.0)  // Should still be reasonable
    }
    
    func testProductQuantizationEmptyTrainingData() async throws {
        let config = QuantizationConfiguration(
            method: .product(codebooks: 4, bitsPerCode: 8),
            useGPU: false
        )
        let pqEngine = await QuantizationEngine(configuration: config)
        
        do {
            try await pqEngine.trainProductQuantization(
                trainingData: [],
                numCodebooks: 4,
                centroids: 16
            )
            XCTFail("Should have thrown error for empty training data")
        } catch AccelerationError.unsupportedOperation(let message) {
            XCTAssertTrue(message.contains("empty"))
        }
    }
    
    // MARK: - Binary Quantization Tests
    
    func testBinaryQuantization() async throws {
        let config = QuantizationConfiguration(
            method: .binary,
            useGPU: false
        )
        let binaryEngine = await QuantizationEngine(configuration: config)
        
        let testVector = testVectors[0]
        let quantized = try await binaryEngine.binaryQuantize(vector: testVector)
        
        XCTAssertEqual(quantized.dimensions, testVector.count)
        XCTAssertEqual(quantized.scheme, .binary)
        XCTAssertNotNil(quantized.metadata)
        
        // Binary quantization should use 1 bit per element
        let expectedBytes = (testVector.count + 7) / 8
        XCTAssertEqual(quantized.data.count, expectedBytes)
        
        // Should have very high compression ratio
        XCTAssertGreaterThan(quantized.compressionRatio, 16.0)  // At least 16x compression
    }
    
    func testBinaryDequantization() async throws {
        let config = QuantizationConfiguration(
            method: .binary,
            useGPU: false
        )
        let binaryEngine = await QuantizationEngine(configuration: config)
        
        let testVector = testVectors[0]
        let quantized = try await binaryEngine.binaryQuantize(vector: testVector)
        let reconstructed = try await binaryEngine.binaryDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, testVector.count)
        
        // Binary quantization is very lossy but should reconstruct something reasonable
        // All values should be close to threshold +/- 0.5
        guard let thresholdStr = quantized.metadata?["threshold"],
              let threshold = Float(thresholdStr) else {
            XCTFail("Missing threshold metadata")
            return
        }
        
        for value in reconstructed {
            let distance = abs(value - threshold)
            XCTAssertLessThanOrEqual(distance, 0.6)  // Should be threshold ± 0.5
        }
    }
    
    func testBinaryQuantizationWithZeroVector() async throws {
        let config = QuantizationConfiguration(
            method: .binary,
            useGPU: false
        )
        let binaryEngine = await QuantizationEngine(configuration: config)
        
        let zeroVector = Array(repeating: Float(0.0), count: 128)
        let quantized = try await binaryEngine.binaryQuantize(vector: zeroVector)
        let reconstructed = try await binaryEngine.binaryDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, zeroVector.count)
        
        // With zero vector, threshold should be 0, and all values should be -0.5
        for value in reconstructed {
            XCTAssertEqual(value, -0.5, accuracy: 0.01)
        }
    }
    
    // MARK: - General Dequantization Tests
    
    func testGeneralDequantizationScalar() async throws {
        let testVector = testVectors[0]
        let quantized = try await engine!.scalarQuantize(vector: testVector, bits: 8)
        let reconstructed = try await engine!.dequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, testVector.count)
    }
    
    func testGeneralDequantizationBinary() async throws {
        let config = QuantizationConfiguration(
            method: .binary,
            useGPU: false
        )
        let binaryEngine = await QuantizationEngine(configuration: config)
        
        let testVector = testVectors[0]
        let quantized = try await binaryEngine.binaryQuantize(vector: testVector)
        let reconstructed = try await binaryEngine.dequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, testVector.count)
    }
    
    func testGeneralDequantizationCustomScheme() async throws {
        // Create a quantized vector with custom scheme
        let customQuantized = QuantizedVector(
            data: Data([1, 2, 3, 4]),
            dimensions: 4,
            scheme: .custom(name: "test", parameters: [:]),
            metadata: ["test": "value"]
        )
        
        do {
            _ = try await engine!.dequantize(quantized: customQuantized)
            XCTFail("Should have thrown error for custom dequantization")
        } catch AccelerationError.unsupportedOperation(let message) {
            XCTAssertTrue(message.contains("Custom dequantization"))
        }
    }
    
    // MARK: - Error Measurement Tests
    
    func testErrorMeasurement() async throws {
        let original = testVectors[0]
        let quantized8 = try await engine!.scalarQuantize(vector: original, bits: 8)
        let quantized4 = try await engine!.scalarQuantize(vector: original, bits: 4)
        
        let error8 = try await engine!.measureError(original: original, quantized: quantized8)
        let error4 = try await engine!.measureError(original: original, quantized: quantized4)
        
        // 4-bit should have higher error than 8-bit
        XCTAssertGreaterThan(error4, error8)
        
        // Both should be finite and non-negative
        XCTAssertTrue(error8.isFinite && error8 >= 0)
        XCTAssertTrue(error4.isFinite && error4 >= 0)
    }
    
    func testErrorMeasurementPerfectReconstruction() async throws {
        // Test with uniform vector that should quantize perfectly
        let uniform = Array(repeating: Float(1.0), count: 128)
        let quantized = try await engine!.scalarQuantize(vector: uniform, bits: 8)
        let error = try await engine!.measureError(original: uniform, quantized: quantized)
        
        // Should have very low error for uniform data
        XCTAssertLessThan(error, 0.01)
    }
    
    // MARK: - Performance Metrics Tests
    
    func testPerformanceMetrics() async throws {
        let initialMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(initialMetrics.vectorsQuantized, 0)
        XCTAssertEqual(initialMetrics.averageCompressionRatio, 0)
        
        // Quantize several vectors
        for i in 0..<5 {
            _ = try await engine!.scalarQuantize(vector: testVectors[i], bits: 8)
        }
        
        let finalMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(finalMetrics.vectorsQuantized, 5)
        XCTAssertGreaterThan(finalMetrics.averageCompressionRatio, 0)
        XCTAssertLessThan(finalMetrics.averageCompressionRatio, 10)  // Reasonable range
    }
    
    func testCompressionRatioCalculation() async throws {
        let testVector = Array(repeating: Float(1.0), count: 100)
        
        let quantized8 = try await engine!.scalarQuantize(vector: testVector, bits: 8)
        let quantized4 = try await engine!.scalarQuantize(vector: testVector, bits: 4)
        
        // 8-bit should have 4x compression (4 bytes -> 1 byte)
        XCTAssertEqual(quantized8.compressionRatio, 4.0, accuracy: 0.01)
        
        // 4-bit should have ~8x compression (4 bytes -> 0.5 bytes)
        XCTAssertGreaterThan(quantized4.compressionRatio, 7.0)
    }
    
    // MARK: - Edge Cases and Error Handling Tests
    
    func testQuantizationWithExtremeValues() async throws {
        let extremeVector = [Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude] + 
                           Array(repeating: Float(0.0), count: 126)
        
        let quantized = try await engine!.scalarQuantize(vector: extremeVector, bits: 8)
        let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
        
        XCTAssertEqual(reconstructed.count, extremeVector.count)
        
        // Should handle extreme values without crashing
        for value in reconstructed {
            XCTAssertTrue(value.isFinite)
        }
    }
    
    func testQuantizationWithNaN() async throws {
        let nanVector = [Float.nan] + Array(repeating: Float(1.0), count: 127)
        
        // Quantization should handle NaN gracefully (either error or replace)
        do {
            let quantized = try await engine!.scalarQuantize(vector: nanVector, bits: 8)
            let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
            
            // If it succeeds, reconstructed values should be finite
            for value in reconstructed {
                XCTAssertTrue(value.isFinite || value.isNaN)
            }
        } catch {
            // Alternatively, it might throw an error, which is also acceptable
            print("NaN handling threw error (acceptable): \(error)")
        }
    }
    
    func testQuantizationWithInfinity() async throws {
        let infVector = [Float.infinity, -Float.infinity] + Array(repeating: Float(1.0), count: 126)
        
        do {
            let quantized = try await engine!.scalarQuantize(vector: infVector, bits: 8)
            let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
            
            // Should handle infinity without crashing
            XCTAssertEqual(reconstructed.count, infVector.count)
        } catch {
            // Alternatively, it might throw an error for infinity
            print("Infinity handling threw error (acceptable): \(error)")
        }
    }
    
    func testEmptyVectorQuantization() async throws {
        let emptyVector: [Float] = []
        
        let quantized = try await engine!.scalarQuantize(vector: emptyVector, bits: 8)
        let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
        
        XCTAssertEqual(quantized.dimensions, 0)
        XCTAssertEqual(quantized.data.count, 0)
        XCTAssertEqual(reconstructed.count, 0)
    }
    
    func testSingleElementVector() async throws {
        let singleVector = [Float(42.0)]
        
        let quantized = try await engine!.scalarQuantize(vector: singleVector, bits: 8)
        let reconstructed = try await engine!.scalarDequantize(quantized: quantized)
        
        XCTAssertEqual(quantized.dimensions, 1)
        XCTAssertEqual(reconstructed.count, 1)
        XCTAssertEqual(reconstructed[0], 42.0, accuracy: 0.5)  // Should be close
    }
    
    // MARK: - Concurrent Access Tests
    
    func testConcurrentQuantization() async throws {
        // Test multiple concurrent quantization operations
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask { [engine, testVectors] in
                    do {
                        let testVector = testVectors![i % testVectors!.count]
                        _ = try await engine!.scalarQuantize(vector: testVector, bits: 8)
                    } catch {
                        print("Concurrent quantization \(i) failed: \(error)")
                    }
                }
            }
        }
        
        // Engine should still be functional
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertGreaterThan(metrics.vectorsQuantized, 0)
    }
    
    func testConcurrentQuantizationAndDequantization() async throws {
        // Pre-quantize some vectors
        var quantizedVectors: [QuantizedVector] = []
        for i in 0..<5 {
            let quantized = try await engine!.scalarQuantize(vector: testVectors[i], bits: 8)
            quantizedVectors.append(quantized)
        }
        
        // Concurrent quantization and dequantization
        await withTaskGroup(of: Void.self) { group in
            // Quantization tasks
            for i in 5..<10 {
                group.addTask { [engine, testVectors] in
                    do {
                        _ = try await engine!.scalarQuantize(vector: testVectors![i], bits: 8)
                    } catch {
                        print("Concurrent quantization failed: \(error)")
                    }
                }
            }
            
            // Dequantization tasks
            for quantized in quantizedVectors {
                group.addTask { [engine] in
                    do {
                        _ = try await engine!.scalarDequantize(quantized: quantized)
                    } catch {
                        print("Concurrent dequantization failed: \(error)")
                    }
                }
            }
        }
    }
    
    // MARK: - Integration Tests
    
    func testQuantizationPipelineIntegration() async throws {
        // Test a complete pipeline: quantize -> dequantize -> measure error
        let testVector = testVectors[0]
        
        let quantized = try await engine!.scalarQuantize(vector: testVector, bits: 8)
        let reconstructed = try await engine!.dequantize(quantized: quantized)
        let error = try await engine!.measureError(original: testVector, quantized: quantized)
        
        // All operations should succeed and produce reasonable results
        XCTAssertEqual(reconstructed.count, testVector.count)
        XCTAssertTrue(error.isFinite && error >= 0)
        XCTAssertLessThan(error, 1.0)
        
        // Verify the metrics are updated
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.vectorsQuantized, 1)
        XCTAssertGreaterThan(metrics.averageCompressionRatio, 0)
    }
    
    func testMultipleQuantizationMethods() async throws {
        let testVector = testVectors[0]
        
        // Test scalar quantization
        let scalarEngine = await QuantizationEngine(
            configuration: QuantizationConfiguration(method: .scalar(bits: 8), useGPU: false)
        )
        let scalarQuantized = try await scalarEngine.scalarQuantize(vector: testVector, bits: 8)
        
        // Test binary quantization
        let binaryEngine = await QuantizationEngine(
            configuration: QuantizationConfiguration(method: .binary, useGPU: false)
        )
        let binaryQuantized = try await binaryEngine.binaryQuantize(vector: testVector)
        
        // Compare compression ratios
        XCTAssertGreaterThan(binaryQuantized.compressionRatio, scalarQuantized.compressionRatio)
        
        // Both should reconstruct without error
        let scalarReconstructed = try await scalarEngine.dequantize(quantized: scalarQuantized)
        let binaryReconstructed = try await binaryEngine.dequantize(quantized: binaryQuantized)
        
        XCTAssertEqual(scalarReconstructed.count, testVector.count)
        XCTAssertEqual(binaryReconstructed.count, testVector.count)
    }
    
    // MARK: - Performance Tests
    
    func testQuantizationPerformance() async throws {
        let largeVector = Array(repeating: Float(1.0), count: 10000)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        _ = try await engine!.scalarQuantize(vector: largeVector, bits: 8)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        // Quantization should be reasonably fast (under 1 second for 10k elements)
        XCTAssertLessThan(elapsed, 1.0)
        print("Quantization time for 10k elements: \(elapsed)s")
    }
    
    func testBatchQuantizationPerformance() async throws {
        let batchSize = 100
        let vectors = (0..<batchSize).map { _ in testVectors[0] }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for vector in vectors {
            _ = try await engine!.scalarQuantize(vector: vector, bits: 8)
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let perVector = elapsed / Double(batchSize)
        
        print("Average quantization time per vector: \(perVector)s")
        XCTAssertLessThan(perVector, 0.01)  // Should be very fast per vector
    }
}