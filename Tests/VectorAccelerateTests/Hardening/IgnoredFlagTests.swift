import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IgnoredFlagTests: XCTestCase {
    private func buffer<T>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                       options: .storageModeShared))
    }
    private func floats(_ buffer: any MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    // Negative affine outputs must survive when activation is disabled. Bias is
    // applied before the optional ReLU, including partial vector/latent/input tiles.
    func testTiledPassOneHonorsActivationAndBias() throws {
        try withLibraries { device, library, path in
            let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "neural_encode_pass1")))
            for (n, d, l) in [(1, 5, 5), (33, 32, 33), (33, 35, 33)] {
                var input = [Float](repeating: 0, count: n * d)
                var weights = [Float](repeating: 0, count: l * d)
                for i in 0..<n { input[i * d] = -2; input[i * d + 1] = 3 }
                for j in 0..<l { weights[j * d + j % 2] = 1 }
                for activation: UInt32 in [0, 1] {
                    for hasBias: UInt32 in [0, 1] {
                        let output = try buffer([Float](repeating: 12345, count: n * l + 4), device)
                        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(try buffer(input, device), offset: 0, index: 0)
                        encoder.setBuffer(try buffer(weights, device), offset: 0, index: 1)
                        encoder.setBuffer(try buffer([Float](repeating: -1, count: l), device), offset: 0, index: 2)
                        encoder.setBuffer(output, offset: 0, index: 3)
                        for (index, value) in [UInt32(n), UInt32(d), UInt32(l), hasBias, activation].enumerated() {
                            var word = value
                            encoder.setBytes(&word, length: 4, index: index + 4)
                        }
                        encoder.dispatchThreadgroups(MTLSize(width: (n + 31) / 32 + 1, height: (l + 31) / 32 + 1, depth: 1),
                                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                        XCTAssertEqual(command.status, .completed, String(describing: command.error))
                        let actual = floats(output, n * l + 4)
                        for i in 0..<(n * l) {
                            let affine: Float = (i % l) % 2 == 0 ? -2 - Float(hasBias) : 3 - Float(hasBias)
                            XCTAssertEqual(actual[i], activation == 0 ? affine : max(0, affine), "\(path), activation=\(activation)")
                        }
                        XCTAssertEqual(Array(actual.suffix(4)), [12345, 12345, 12345, 12345])
                    }
                }
            }
        }
    }

    func testPublicTiledEncoderPreservesSignedCodesWithoutActivation() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        let n = 33, d = 35, l = 33
        var input = [Float](repeating: 0, count: n * d)
        var weights = [Float](repeating: 0, count: l * d)
        for i in 0..<n { input[i * d] = -2; input[i * d + 1] = 3 }
        for j in 0..<l { weights[j * d + j % 2] = 1 }
        // Deliberately load with activation enabled; dispatch parameters must control it.
        try await kernel.loadWeights(encoderWeights: weights, decoderWeights: [Float](repeating: 0, count: d * l),
            config: Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l))
        for activation in [false, true] {
            let config = Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l, useActivation: activation)
            let output = try buffer([Int8](repeating: 42, count: n * l + 4), device)
            let scales = try buffer([Float](repeating: 12345, count: n + 1), device)
            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
            try await kernel.encodeTiledV3(commandBuffer: command, input: buffer(input, device), output: output,
                scale: scales, parameters: NeuralQuantizationParameters(numVectors: n, config: config))
            command.commit(); await command.completed()
            XCTAssertEqual(command.status, .completed, String(describing: command.error))
            let codes = output.contents().assumingMemoryBound(to: Int8.self)
            for i in 0..<(n * l) {
                XCTAssertEqual(codes[i], (i % l) % 2 == 0 ? (activation ? 0 : -85) : 127)
            }
            for i in (n * l)..<(n * l + 4) { XCTAssertEqual(codes[i], 42) }
            for scale in floats(scales, n) { XCTAssertEqual(scale, 3 / 127, accuracy: 1e-7) }
            XCTAssertEqual(floats(scales, n + 1).last, 12345)
        }
    }

    private func projection(_ d: Int, _ l: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: d * l)
        weights[0] = 1; weights[d + 1] = 1
        return weights
    }
    private func rows(_ coordinates: [[Float]], dimension: Int) -> [Float] {
        coordinates.flatMap { $0 + [Float](repeating: 0, count: dimension - $0.count) }
    }

    func testSpecializedLearnedDistancesHonorNormalizationAndSqrt() throws {
        try withLibraries { device, library, path in
            for (d, l, name) in [(384, 64, "learned_l2_384_to_64_kernel"), (768, 128, "learned_l2_768_to_128_kernel")] {
                for function in [name, "learned_l2_distance_kernel"] {
                    let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: function)))
                    let queries = try buffer(rows([[3, 4], [0, 0]], dimension: d), device)
                    let database = try buffer(rows([[0, 10], [0, 0], [6, 8], [-3, -4], [1e-10, 0]], dimension: d), device)
                    let weights = try buffer(projection(d, l), device)
                    for normalize in [false, true] {
                        for root in [false, true] {
                            let output = try buffer([Float](repeating: 12345, count: 14), device)
                            // Seven UInt32 fields followed by two flag bytes and two padding bytes.
                            let flags: UInt32 = (root ? 1 : 0) | (normalize ? 256 : 0)
                            let params: [UInt32] = [2, 5, UInt32(d), UInt32(l), UInt32(d), UInt32(d), 7, flags]
                            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                            encoder.setComputePipelineState(pipeline)
                            for (i, b) in [queries, database, weights, output].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
                            encoder.setBuffer(try buffer(params, device), offset: 0, index: 4)
                            encoder.dispatchThreadgroups(MTLSize(width: 2, height: 2, depth: 1),
                                                         threadsPerThreadgroup: MTLSize(width: 2, height: 4, depth: 1))
                            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                            XCTAssertEqual(command.status, .completed, String(describing: command.error))
                            let expected: [[Float]] = normalize ? [[0.4, 1, 0, 4, 1], [1, 0, 1, 1, 0]]
                                : [[45, 25, 25, 100, 25], [100, 0, 100, 25, 1e-20]]
                            let result = floats(output, 14)
                            for q in 0..<2 {
                                for j in 0..<5 {
                                    let want = root ? sqrt(expected[q][j]) : expected[q][j]
                                    XCTAssertEqual(result[q * 7 + j], want, accuracy: 2e-5, "\(path), \(function), norm=\(normalize)")
                                }
                                XCTAssertEqual(Array(result[(q * 7 + 5)..<(q * 7 + 7)]), [12345, 12345])
                            }
                        }
                    }
                }
            }
        }
    }

    func testPublicLearnedDistanceNormalizationAtSpecializedDimensions() async throws {
        let context = try await Metal4Context()
        let kernel = try await LearnedDistanceKernel(context: context)
        for (d, l) in [(384, 64), (768, 128)] {
            let tensor = try await kernel.createProjection(from: projection(d, l), inputDim: d, outputDim: l)
            let queries = [rows([[3, 4]], dimension: d)]
            let database = [rows([[0, 10]], dimension: d)]
            for normalize in [false, true] {
                for root in [false, true] {
                    let result = try await kernel.compute(queries: queries, database: database, projection: tensor,
                                                         computeSqrt: root, normalizeProjected: normalize)
                    let expected: Float = normalize ? 0.4 : 45
                    XCTAssertEqual(result[0][0], root ? sqrt(expected) : expected, accuracy: 2e-5)
                }
            }
        }
    }
}
