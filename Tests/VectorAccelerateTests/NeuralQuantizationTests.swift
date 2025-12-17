// VectorAccelerate: Metal 4 Neural Quantization Tests
//
// Tests for Phase 4 ML Integration - Neural Quantization kernel.
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

// MARK: - Neural Quantization Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NeuralQuantizationKernelTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testKernelInitialization() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        XCTAssertEqual(kernel.name, "NeuralQuantizationKernel")
        XCTAssertFalse(kernel.hasWeights)
    }

    func testRandomWeightCreation() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 128,
            latentDimension: 32,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        XCTAssertTrue(kernel.hasWeights)

        let stats = await kernel.getTensorStatistics()
        XCTAssertEqual(stats.loadedTensors, 3) // encoder + decoder + transposed decoder
    }

    // MARK: - Encoding Tests

    func testBasicEncoding() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 128)

        try await kernel.createRandomWeights(config: config)

        // Create test vectors
        let vectors = (0..<10).map { _ in
            (0..<128).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.encode(vectors)

        XCTAssertEqual(result.numVectors, 10)
        XCTAssertEqual(result.latentDimension, 128)  // balanced config uses latentDim=128
        XCTAssertEqual(result.latentCodes.count, 10 * 128)
        XCTAssertGreaterThan(result.encodingTime, 0)
    }

    func testHighCompressionEncoding() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.highCompression(inputDim: 256)

        try await kernel.createRandomWeights(config: config)

        let vectors = (0..<5).map { _ in
            (0..<256).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.encode(vectors)

        XCTAssertEqual(result.numVectors, 5)
        XCTAssertEqual(result.latentDimension, 64)  // highCompression uses 64
        XCTAssertEqual(result.latentCodes.count, 5 * 64)

        // Verify compression ratio
        let originalSize = 5 * 256 * MemoryLayout<Float>.size
        let compressedSize = result.compressedSize
        let ratio = Float(originalSize) / Float(compressedSize)
        XCTAssertGreaterThan(ratio, 10.0)  // 256*4 bytes -> 64*1 bytes = 16x
    }

    // MARK: - Decoding Tests

    func testEncodeDecode() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 64,
            latentDimension: 32,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        let vectors = (0..<5).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }

        let encoded = try await kernel.encode(vectors)
        let decoded = try await kernel.decode(encoded)

        XCTAssertEqual(decoded.count, 5)
        XCTAssertEqual(decoded[0].count, 64)

        // With random weights, reconstruction won't be perfect
        // but values should be finite and in reasonable range
        for (i, vec) in decoded.enumerated() {
            for (j, val) in vec.enumerated() {
                XCTAssertFalse(val.isNaN, "NaN at vector \(i), element \(j)")
                XCTAssertFalse(val.isInfinite, "Infinite at vector \(i), element \(j)")
            }
        }
    }

    // MARK: - Metrics Tests

    func testEncodeDecodeWithMetrics() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 64,
            latentDimension: 32,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        let vectors = (0..<10).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }

        let (reconstructed, metrics) = try await kernel.encodeDecodeWithMetrics(vectors)

        XCTAssertEqual(reconstructed.count, 10)

        // Verify metrics are computed
        XCTAssertGreaterThan(metrics.compressionRatio, 1.0)
        XCTAssertGreaterThan(metrics.encodingThroughput, 0)
        XCTAssertGreaterThan(metrics.decodingThroughput, 0)

        // MSE should be finite
        XCTAssertFalse(metrics.mse.isNaN)
        XCTAssertFalse(metrics.mse.isInfinite)

        // Cosine similarity should be in valid range
        XCTAssertGreaterThanOrEqual(metrics.cosineSimilarity, -1.0)
        XCTAssertLessThanOrEqual(metrics.cosineSimilarity, 1.0)
    }

    // MARK: - Configuration Tests

    func testMiniLMConfiguration() async throws {
        let config = Metal4NeuralQuantizationConfig.miniLM()

        XCTAssertEqual(config.inputDimension, 384)
        XCTAssertEqual(config.latentDimension, 64)
        XCTAssertTrue(config.useActivation)
        XCTAssertFalse(config.normalizeLatent)

        // Compression ratio: 384 floats (1536 bytes) -> 64 INT8s (64 bytes) = 24x
        XCTAssertGreaterThan(config.compressionRatio, 20.0)
    }

    func testBalancedConfiguration() async throws {
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        XCTAssertEqual(config.inputDimension, 768)
        XCTAssertEqual(config.latentDimension, 128)
        XCTAssertTrue(config.useActivation)
        XCTAssertFalse(config.normalizeLatent)

        // Compression ratio: 768 floats (3072 bytes) -> 128 INT8s (128 bytes) = 24x
        XCTAssertGreaterThan(config.compressionRatio, 20.0)
    }

    // MARK: - Weight Management Tests

    func testWeightUnloading() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 64,
            latentDimension: 16,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)
        XCTAssertTrue(kernel.hasWeights)

        var stats = await kernel.getTensorStatistics()
        XCTAssertEqual(stats.loadedTensors, 3) // encoder + decoder + transposed decoder

        await kernel.unloadWeights()
        XCTAssertFalse(kernel.hasWeights)

        stats = await kernel.getTensorStatistics()
        XCTAssertEqual(stats.loadedTensors, 0)
    }

    func testLoadWeightsFromArrays() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 8,
            latentDimension: 4,
            useActivation: false,
            normalizeLatent: false
        )

        // Create identity-like encoder (for testability)
        var encoderWeights = [Float](repeating: 0, count: 4 * 8)
        for i in 0..<4 {
            encoderWeights[i * 8 + i] = 1.0  // Diagonal
        }

        // Create identity-like decoder
        var decoderWeights = [Float](repeating: 0, count: 8 * 4)
        for i in 0..<4 {
            decoderWeights[i * 4 + i] = 1.0  // Diagonal
        }

        try await kernel.loadWeights(
            encoderWeights: encoderWeights,
            decoderWeights: decoderWeights,
            config: config
        )

        XCTAssertTrue(kernel.hasWeights)
    }

    // MARK: - Error Handling Tests

    func testEncodeWithoutWeights() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        let vectors = [[Float](repeating: 0.5, count: 64)]

        do {
            _ = try await kernel.encode(vectors)
            XCTFail("Expected error for encoding without weights")
        } catch {
            // Expected - weights not loaded
        }
    }

    func testDimensionMismatch() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 64,
            latentDimension: 16,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Try to encode vectors with wrong dimension
        let wrongDimVectors = [[Float](repeating: 0.5, count: 128)]  // Expected 64

        do {
            _ = try await kernel.encode(wrongDimVectors)
            XCTFail("Expected error for dimension mismatch")
        } catch {
            // Expected - dimension mismatch
        }
    }

    func testEmptyInput() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 64,
            latentDimension: 16,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        let emptyVectors: [[Float]] = []

        do {
            _ = try await kernel.encode(emptyVectors)
            XCTFail("Expected error for empty input")
        } catch {
            // Expected - empty input
        }
    }

    // MARK: - Performance Tests

    func testEncodingPerformance() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        // Generate 1000 vectors
        let vectors = (0..<1000).map { _ in
            (0..<768).map { _ in Float.random(in: -1...1) }
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await kernel.encode(vectors)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        let throughput = Double(vectors.count) / elapsed

        print("Neural Quantization Performance:")
        print("  Vectors: \(vectors.count)")
        print("  Input dimension: 768")
        print("  Latent dimension: 128")
        print("  Encoding time: \(String(format: "%.3f", elapsed * 1000)) ms")
        print("  Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
        print("  Compression ratio: \(String(format: "%.1f", config.compressionRatio))x")
        print("  Output size: \(result.compressedSize) bytes")

        // Should process at reasonable speed (at least 100 vectors/sec)
        XCTAssertGreaterThan(throughput, 100.0)
    }
}
