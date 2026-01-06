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

    // MARK: - Tiled Encode Tests

    /// Test that tiled encode kernel produces identical output to reference kernel.
    func testTiledEncodeCorrectness() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        // Skip if tiled encode is not available
        guard kernel.isTiledEncodeAvailable else {
            throw XCTSkip("Tiled encode kernel not available")
        }

        // Test with common configuration: 768 -> 128
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Test batch sizes: 1 (edge case), 32 (one threadgroup), 100 (multiple threadgroups)
        for numVectors in [1, 32, 100] {
            // Generate test vectors
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            // Get reference output using the standard 2D encode kernel
            let referenceOutput = try await encodeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Get tiled output
            let tiledOutput = try await encodeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Compare outputs
            XCTAssertEqual(referenceOutput.count, tiledOutput.count,
                          "Output count mismatch for \(numVectors) vectors")

            let tolerance: Float = 1e-5
            for i in 0..<referenceOutput.count {
                let diff = abs(referenceOutput[i] - tiledOutput[i])
                XCTAssertLessThan(diff, tolerance,
                    "Mismatch at index \(i) for \(numVectors) vectors: ref=\(referenceOutput[i]), tiled=\(tiledOutput[i]), diff=\(diff)")
            }
        }
    }

    /// Test tiled encode with different configurations (768->64, 384->64).
    func testTiledEncodeConfigurations() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeAvailable else {
            throw XCTSkip("Tiled encode kernel not available")
        }

        let configs: [(inputDim: Int, latentDim: Int, name: String)] = [
            (768, 128, "768→128"),
            (768, 64, "768→64"),
            (384, 64, "384→64"),
        ]

        for (inputDim, latentDim, name) in configs {
            let config = Metal4NeuralQuantizationConfig(
                inputDimension: inputDim,
                latentDimension: latentDim,
                useActivation: true,
                normalizeLatent: false
            )

            try await kernel.createRandomWeights(config: config)

            let numVectors = 50
            let vectors = (0..<numVectors).map { _ in
                (0..<inputDim).map { _ in Float.random(in: -1...1) }
            }

            let referenceOutput = try await encodeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let tiledOutput = try await encodeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let tolerance: Float = 1e-5
            var maxDiff: Float = 0
            for i in 0..<referenceOutput.count {
                let diff = abs(referenceOutput[i] - tiledOutput[i])
                maxDiff = max(maxDiff, diff)
            }

            XCTAssertLessThan(maxDiff, tolerance,
                "\(name): Max diff \(maxDiff) exceeds tolerance \(tolerance)")
        }
    }

    /// Test tiled encode with non-divisible batch sizes.
    func testTiledEncodeEdgeCases() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeAvailable else {
            throw XCTSkip("Tiled encode kernel not available")
        }

        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Edge cases: not divisible by 32 (threadgroup size), prime number, etc.
        for numVectors in [1, 7, 31, 33, 63, 97] {
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            let referenceOutput = try await encodeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let tiledOutput = try await encodeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let tolerance: Float = 1e-5
            var maxDiff: Float = 0
            for i in 0..<referenceOutput.count {
                let diff = abs(referenceOutput[i] - tiledOutput[i])
                maxDiff = max(maxDiff, diff)
            }

            XCTAssertLessThan(maxDiff, tolerance,
                "Batch size \(numVectors): Max diff \(maxDiff) exceeds tolerance \(tolerance)")
        }
    }

    // MARK: - Tiled Encode + Quantize Tests (Phase 2)

    /// Test that tiled encode+quantize kernel produces identical output to reference kernel.
    func testTiledEncodeQuantizeCorrectness() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        // Skip if tiled encode+quantize is not available
        guard kernel.isTiledEncodeQuantizeAvailable else {
            throw XCTSkip("Tiled encode+quantize kernel not available")
        }

        // Test with common configuration: 768 -> 128
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Test batch sizes: 1 (edge case), 32 (one threadgroup), 100 (multiple threadgroups)
        for numVectors in [1, 32, 100] {
            // Generate test vectors
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            // Get reference output using the standard encode+quantize kernel
            let (refCodes, refScales) = try await encodeQuantizeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Get tiled output
            let (tiledCodes, tiledScales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Compare INT8 codes (should be byte-for-byte identical)
            XCTAssertEqual(refCodes.count, tiledCodes.count,
                          "Code count mismatch for \(numVectors) vectors")

            for i in 0..<refCodes.count {
                XCTAssertEqual(refCodes[i], tiledCodes[i],
                    "INT8 mismatch at index \(i) for \(numVectors) vectors: ref=\(refCodes[i]), tiled=\(tiledCodes[i])")
            }

            // Compare scales (should match within tolerance)
            XCTAssertEqual(refScales.count, tiledScales.count,
                          "Scale count mismatch for \(numVectors) vectors")

            let scaleTolerance: Float = 1e-6
            for i in 0..<refScales.count {
                let diff = abs(refScales[i] - tiledScales[i])
                XCTAssertLessThan(diff, scaleTolerance,
                    "Scale mismatch at vector \(i) for \(numVectors) vectors: ref=\(refScales[i]), tiled=\(tiledScales[i]), diff=\(diff)")
            }
        }
    }

    /// Test tiled encode+quantize with different configurations.
    func testTiledEncodeQuantizeConfigurations() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeQuantizeAvailable else {
            throw XCTSkip("Tiled encode+quantize kernel not available")
        }

        let configs: [(inputDim: Int, latentDim: Int, name: String)] = [
            (768, 128, "768→128"),
            (768, 64, "768→64"),
            (384, 64, "384→64"),
        ]

        for (inputDim, latentDim, name) in configs {
            let config = Metal4NeuralQuantizationConfig(
                inputDimension: inputDim,
                latentDimension: latentDim,
                useActivation: true,
                normalizeLatent: false
            )

            try await kernel.createRandomWeights(config: config)

            let numVectors = 50
            let vectors = (0..<numVectors).map { _ in
                (0..<inputDim).map { _ in Float.random(in: -1...1) }
            }

            let (refCodes, refScales) = try await encodeQuantizeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let (tiledCodes, tiledScales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Check INT8 codes match byte-for-byte
            var mismatches = 0
            for i in 0..<refCodes.count {
                if refCodes[i] != tiledCodes[i] {
                    mismatches += 1
                }
            }
            XCTAssertEqual(mismatches, 0,
                "\(name): \(mismatches) INT8 code mismatches out of \(refCodes.count)")

            // Check scales match
            var maxScaleDiff: Float = 0
            for i in 0..<refScales.count {
                maxScaleDiff = max(maxScaleDiff, abs(refScales[i] - tiledScales[i]))
            }
            XCTAssertLessThan(maxScaleDiff, 1e-6,
                "\(name): Max scale diff \(maxScaleDiff) exceeds tolerance")
        }
    }

    /// Test tiled encode+quantize with edge case batch sizes.
    func testTiledEncodeQuantizeEdgeCases() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeQuantizeAvailable else {
            throw XCTSkip("Tiled encode+quantize kernel not available")
        }

        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Edge cases: not divisible by 32, prime number, etc.
        for numVectors in [1, 7, 31, 33, 63, 97] {
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            let (refCodes, refScales) = try await encodeQuantizeWithReferenceKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let (tiledCodes, tiledScales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Verify byte-for-byte match
            var mismatches = 0
            for i in 0..<refCodes.count {
                if refCodes[i] != tiledCodes[i] {
                    mismatches += 1
                }
            }
            XCTAssertEqual(mismatches, 0,
                "Batch size \(numVectors): \(mismatches) INT8 code mismatches")

            // Verify scales match
            var maxScaleDiff: Float = 0
            for i in 0..<refScales.count {
                maxScaleDiff = max(maxScaleDiff, abs(refScales[i] - tiledScales[i]))
            }
            XCTAssertLessThan(maxScaleDiff, 1e-6,
                "Batch size \(numVectors): Max scale diff \(maxScaleDiff) exceeds tolerance")
        }
    }

    // MARK: - Helper Methods for Tiled Encode Tests

    /// Encode using reference (2D) kernel - returns flat float array.
    private func encodeWithReferenceKernel(
        kernel: NeuralQuantizationKernel,
        vectors: [[Float]],
        config: Metal4NeuralQuantizationConfig
    ) async throws -> [Float] {
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        let flatInput = vectors.flatMap { $0 }
        let inputToken = try await context.getBuffer(for: flatInput)
        inputToken.buffer.label = "TiledTest.reference.input"

        let outputSize = numVectors * latentDim * MemoryLayout<Float>.size
        let outputToken = try await context.getBuffer(size: outputSize)
        outputToken.buffer.label = "TiledTest.reference.output"

        let parameters = NeuralQuantizationParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { _, encoder in
            try kernel.encodeEncode(
                into: encoder,
                input: inputToken.buffer,
                output: outputToken.buffer,
                parameters: parameters
            )
        }

        let outputPtr = outputToken.buffer.contents().bindMemory(to: Float.self, capacity: numVectors * latentDim)
        return Array(UnsafeBufferPointer(start: outputPtr, count: numVectors * latentDim))
    }

    /// Encode using tiled GEMM kernel - returns flat float array.
    private func encodeWithTiledKernel(
        kernel: NeuralQuantizationKernel,
        vectors: [[Float]],
        config: Metal4NeuralQuantizationConfig
    ) async throws -> [Float] {
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        let flatInput = vectors.flatMap { $0 }
        let inputToken = try await context.getBuffer(for: flatInput)
        inputToken.buffer.label = "TiledTest.tiled.input"

        let outputSize = numVectors * latentDim * MemoryLayout<Float>.size
        let outputToken = try await context.getBuffer(size: outputSize)
        outputToken.buffer.label = "TiledTest.tiled.output"

        let parameters = TiledEncodeParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { _, encoder in
            try kernel.encodeTiledEncode(
                into: encoder,
                input: inputToken.buffer,
                output: outputToken.buffer,
                parameters: parameters
            )
        }

        let outputPtr = outputToken.buffer.contents().bindMemory(to: Float.self, capacity: numVectors * latentDim)
        return Array(UnsafeBufferPointer(start: outputPtr, count: numVectors * latentDim))
    }

    // MARK: - Tiled Encode + Quantize V2 Tests (Phase 3 - Dual Accumulators)

    /// Test that tiled encode+quantize V2 kernel (dual accumulators) produces identical output to V1.
    func testTiledEncodeQuantizeV2Correctness() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        // Skip if V2 is not available
        guard kernel.isTiledEncodeQuantizeV2Available else {
            throw XCTSkip("Tiled encode+quantize V2 kernel not available")
        }

        // Test with common configuration: 768 -> 128
        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Test batch sizes: 1 (edge case), 32 (one threadgroup), 100 (multiple threadgroups)
        for numVectors in [1, 32, 100] {
            // Generate test vectors
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            // Get V1 output (reference for V2)
            let (v1Codes, v1Scales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Get V2 output
            let (v2Codes, v2Scales) = try await encodeQuantizeWithTiledV2Kernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Compare INT8 codes (should be byte-for-byte identical)
            XCTAssertEqual(v1Codes.count, v2Codes.count,
                          "Code count mismatch for \(numVectors) vectors")

            for i in 0..<v1Codes.count {
                XCTAssertEqual(v1Codes[i], v2Codes[i],
                    "INT8 mismatch at index \(i) for \(numVectors) vectors: V1=\(v1Codes[i]), V2=\(v2Codes[i])")
            }

            // Compare scales (should match within tolerance)
            XCTAssertEqual(v1Scales.count, v2Scales.count,
                          "Scale count mismatch for \(numVectors) vectors")

            let scaleTolerance: Float = 1e-6
            for i in 0..<v1Scales.count {
                let diff = abs(v1Scales[i] - v2Scales[i])
                XCTAssertLessThan(diff, scaleTolerance,
                    "Scale mismatch at vector \(i) for \(numVectors) vectors: V1=\(v1Scales[i]), V2=\(v2Scales[i]), diff=\(diff)")
            }
        }
    }

    /// Test tiled encode+quantize V2 with different configurations.
    func testTiledEncodeQuantizeV2Configurations() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeQuantizeV2Available else {
            throw XCTSkip("Tiled encode+quantize V2 kernel not available")
        }

        let configs: [(inputDim: Int, latentDim: Int, name: String)] = [
            (768, 128, "768→128"),
            (768, 64, "768→64"),
            (384, 64, "384→64"),
        ]

        for (inputDim, latentDim, name) in configs {
            let config = Metal4NeuralQuantizationConfig(
                inputDimension: inputDim,
                latentDimension: latentDim,
                useActivation: true,
                normalizeLatent: false
            )

            try await kernel.createRandomWeights(config: config)

            let numVectors = 50
            let vectors = (0..<numVectors).map { _ in
                (0..<inputDim).map { _ in Float.random(in: -1...1) }
            }

            let (v1Codes, v1Scales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let (v2Codes, v2Scales) = try await encodeQuantizeWithTiledV2Kernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Check INT8 codes match byte-for-byte
            var mismatches = 0
            for i in 0..<v1Codes.count {
                if v1Codes[i] != v2Codes[i] {
                    mismatches += 1
                }
            }
            XCTAssertEqual(mismatches, 0,
                "\(name): \(mismatches) INT8 code mismatches out of \(v1Codes.count)")

            // Check scales match
            var maxScaleDiff: Float = 0
            for i in 0..<v1Scales.count {
                maxScaleDiff = max(maxScaleDiff, abs(v1Scales[i] - v2Scales[i]))
            }
            XCTAssertLessThan(maxScaleDiff, 1e-6,
                "\(name): Max scale diff \(maxScaleDiff) exceeds tolerance")
        }
    }

    /// Test tiled encode+quantize V2 with edge case batch sizes.
    func testTiledEncodeQuantizeV2EdgeCases() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        guard kernel.isTiledEncodeQuantizeV2Available else {
            throw XCTSkip("Tiled encode+quantize V2 kernel not available")
        }

        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        // Edge cases: not divisible by 32, prime number, etc.
        for numVectors in [1, 7, 31, 33, 63, 97] {
            let vectors = (0..<numVectors).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            let (v1Codes, v1Scales) = try await encodeQuantizeWithTiledKernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            let (v2Codes, v2Scales) = try await encodeQuantizeWithTiledV2Kernel(
                kernel: kernel,
                vectors: vectors,
                config: config
            )

            // Verify byte-for-byte match
            var mismatches = 0
            for i in 0..<v1Codes.count {
                if v1Codes[i] != v2Codes[i] {
                    mismatches += 1
                }
            }
            XCTAssertEqual(mismatches, 0,
                "Batch size \(numVectors): \(mismatches) INT8 code mismatches")

            // Verify scales match
            var maxScaleDiff: Float = 0
            for i in 0..<v1Scales.count {
                maxScaleDiff = max(maxScaleDiff, abs(v1Scales[i] - v2Scales[i]))
            }
            XCTAssertLessThan(maxScaleDiff, 1e-6,
                "Batch size \(numVectors): Max scale diff \(maxScaleDiff) exceeds tolerance")
        }
    }

    /// Performance benchmark comparing reference vs V1 vs V2 kernels.
    func testTiledEncodeQuantizePerformance() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)

        // Require all kernels for this benchmark
        guard kernel.isTiledEncodeQuantizeAvailable else {
            throw XCTSkip("Tiled encode+quantize kernel not available")
        }

        let config = Metal4NeuralQuantizationConfig(
            inputDimension: 768,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )

        try await kernel.createRandomWeights(config: config)

        let batchSizes = [100, 1000]
        let warmupIterations = 3
        let measureIterations = 5

        print("\n=== Neural Encoder Performance Benchmark ===")
        print("Config: 768 → 128 (balanced)")
        print("Iterations: \(warmupIterations) warmup + \(measureIterations) measured")
        print("")

        for batchSize in batchSizes {
            let vectors = (0..<batchSize).map { _ in
                (0..<config.inputDimension).map { _ in Float.random(in: -1...1) }
            }

            // Benchmark reference kernel (1D dispatch)
            var refTimes: [Double] = []
            for i in 0..<(warmupIterations + measureIterations) {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await encodeQuantizeWithReferenceKernel(
                    kernel: kernel,
                    vectors: vectors,
                    config: config
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                if i >= warmupIterations {
                    refTimes.append(elapsed)
                }
            }
            let refMedian = refTimes.sorted()[refTimes.count / 2]

            // Benchmark V1 kernel (tiled GEMM)
            var v1Times: [Double] = []
            for i in 0..<(warmupIterations + measureIterations) {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await encodeQuantizeWithTiledKernel(
                    kernel: kernel,
                    vectors: vectors,
                    config: config
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                if i >= warmupIterations {
                    v1Times.append(elapsed)
                }
            }
            let v1Median = v1Times.sorted()[v1Times.count / 2]

            // Benchmark V2 kernel (dual accumulators) if available
            var v2Median: Double? = nil
            if kernel.isTiledEncodeQuantizeV2Available {
                var v2Times: [Double] = []
                for i in 0..<(warmupIterations + measureIterations) {
                    let start = CFAbsoluteTimeGetCurrent()
                    _ = try await encodeQuantizeWithTiledV2Kernel(
                        kernel: kernel,
                        vectors: vectors,
                        config: config
                    )
                    let elapsed = CFAbsoluteTimeGetCurrent() - start
                    if i >= warmupIterations {
                        v2Times.append(elapsed)
                    }
                }
                v2Median = v2Times.sorted()[v2Times.count / 2]
            }

            // Calculate throughput and speedups
            let refThroughput = Double(batchSize) / refMedian
            let v1Throughput = Double(batchSize) / v1Median
            let v1Speedup = refMedian / v1Median

            print("Batch \(batchSize):")
            print("  Reference: \(String(format: "%.3f", refMedian * 1000)) ms (\(String(format: "%.0f", refThroughput)) vec/s)")
            print("  V1 (tiled): \(String(format: "%.3f", v1Median * 1000)) ms (\(String(format: "%.0f", v1Throughput)) vec/s) [\(String(format: "%.2f", v1Speedup))x]")

            if let v2Med = v2Median {
                let v2Throughput = Double(batchSize) / v2Med
                let v2SpeedupVsRef = refMedian / v2Med
                let v2SpeedupVsV1 = v1Median / v2Med
                print("  V2 (dual acc): \(String(format: "%.3f", v2Med * 1000)) ms (\(String(format: "%.0f", v2Throughput)) vec/s) [\(String(format: "%.2f", v2SpeedupVsRef))x vs ref, \(String(format: "%.2f", v2SpeedupVsV1))x vs V1]")
            }
            print("")

            // Performance assertions (generous thresholds for CI variability)
            // V1 should be faster than reference
            XCTAssertGreaterThan(v1Speedup, 1.0, "V1 should be faster than reference for batch \(batchSize)")

            // V2 should be at least as fast as V1 (ideally faster)
            if let v2Med = v2Median {
                let v2VsV1 = v1Median / v2Med
                XCTAssertGreaterThan(v2VsV1, 0.9, "V2 should not be significantly slower than V1 for batch \(batchSize)")
            }
        }
    }

    // MARK: - Helper Methods for V2 Tests

    /// Encode+quantize using tiled V2 kernel (dual accumulators) - returns (INT8 codes, scales).
    private func encodeQuantizeWithTiledV2Kernel(
        kernel: NeuralQuantizationKernel,
        vectors: [[Float]],
        config: Metal4NeuralQuantizationConfig
    ) async throws -> ([Int8], [Float]) {
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        let flatInput = vectors.flatMap { $0 }
        let inputToken = try await context.getBuffer(for: flatInput)
        inputToken.buffer.label = "TiledV2Test.input"

        // Output buffer (INT8)
        let outputSize = numVectors * latentDim
        let outputToken = try await context.getBuffer(size: outputSize)
        outputToken.buffer.label = "TiledV2Test.output"

        // Scale buffer
        let scaleToken = try await context.getBuffer(size: numVectors * MemoryLayout<Float>.size)
        scaleToken.buffer.label = "TiledV2Test.scale"

        let parameters = TiledEncodeParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { _, encoder in
            try kernel.encodeTiledEncodeQuantizeV2(
                into: encoder,
                input: inputToken.buffer,
                output: outputToken.buffer,
                scale: scaleToken.buffer,
                parameters: parameters
            )
        }

        // Extract INT8 codes
        let codesPtr = outputToken.buffer.contents().bindMemory(to: Int8.self, capacity: outputSize)
        let codes = Array(UnsafeBufferPointer(start: codesPtr, count: outputSize))

        // Extract scales
        let scalesPtr = scaleToken.buffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        let scales = Array(UnsafeBufferPointer(start: scalesPtr, count: numVectors))

        return (codes, scales)
    }

    // MARK: - Helper Methods for Tiled Encode+Quantize Tests

    /// Encode+quantize using reference kernel - returns (INT8 codes, scales).
    private func encodeQuantizeWithReferenceKernel(
        kernel: NeuralQuantizationKernel,
        vectors: [[Float]],
        config: Metal4NeuralQuantizationConfig
    ) async throws -> ([Int8], [Float]) {
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        let flatInput = vectors.flatMap { $0 }
        let inputToken = try await context.getBuffer(for: flatInput)
        inputToken.buffer.label = "TiledQuantizeTest.reference.input"

        // Output buffer (INT8)
        let outputSize = numVectors * latentDim
        let outputToken = try await context.getBuffer(size: outputSize)
        outputToken.buffer.label = "TiledQuantizeTest.reference.output"

        // Scale buffer
        let scaleToken = try await context.getBuffer(size: numVectors * MemoryLayout<Float>.size)
        scaleToken.buffer.label = "TiledQuantizeTest.reference.scale"

        let parameters = NeuralQuantizationParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { _, encoder in
            try kernel.encodeEncodeQuantize(
                into: encoder,
                input: inputToken.buffer,
                output: outputToken.buffer,
                scale: scaleToken.buffer,
                parameters: parameters
            )
        }

        // Extract INT8 codes
        let codesPtr = outputToken.buffer.contents().bindMemory(to: Int8.self, capacity: outputSize)
        let codes = Array(UnsafeBufferPointer(start: codesPtr, count: outputSize))

        // Extract scales
        let scalesPtr = scaleToken.buffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        let scales = Array(UnsafeBufferPointer(start: scalesPtr, count: numVectors))

        return (codes, scales)
    }

    /// Encode+quantize using tiled kernel - returns (INT8 codes, scales).
    private func encodeQuantizeWithTiledKernel(
        kernel: NeuralQuantizationKernel,
        vectors: [[Float]],
        config: Metal4NeuralQuantizationConfig
    ) async throws -> ([Int8], [Float]) {
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        let flatInput = vectors.flatMap { $0 }
        let inputToken = try await context.getBuffer(for: flatInput)
        inputToken.buffer.label = "TiledQuantizeTest.tiled.input"

        // Output buffer (INT8)
        let outputSize = numVectors * latentDim
        let outputToken = try await context.getBuffer(size: outputSize)
        outputToken.buffer.label = "TiledQuantizeTest.tiled.output"

        // Scale buffer
        let scaleToken = try await context.getBuffer(size: numVectors * MemoryLayout<Float>.size)
        scaleToken.buffer.label = "TiledQuantizeTest.tiled.scale"

        let parameters = TiledEncodeParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { _, encoder in
            try kernel.encodeTiledEncodeQuantize(
                into: encoder,
                input: inputToken.buffer,
                output: outputToken.buffer,
                scale: scaleToken.buffer,
                parameters: parameters
            )
        }

        // Extract INT8 codes
        let codesPtr = outputToken.buffer.contents().bindMemory(to: Int8.self, capacity: outputSize)
        let codes = Array(UnsafeBufferPointer(start: codesPtr, count: outputSize))

        // Extract scales
        let scalesPtr = scaleToken.buffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        let scales = Array(UnsafeBufferPointer(start: scalesPtr, count: numVectors))

        return (codes, scales)
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
