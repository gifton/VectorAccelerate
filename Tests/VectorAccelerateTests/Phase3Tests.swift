//
//  Phase3Tests.swift
//  VectorAccelerateTests
//
//  Tests for Phase 3: QuantizationEngine operations
//

import XCTest
@testable import VectorAccelerate
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

    // MARK: - Performance Tests

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
