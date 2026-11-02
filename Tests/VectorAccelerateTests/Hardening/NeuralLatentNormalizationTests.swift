import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NeuralLatentNormalizationTests: XCTestCase {
    private let epsilon: Float = 1e-7
    private func buffer<T>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
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
    private func rows(_ coordinates: [[Float]], width: Int) -> [Float] {
        coordinates.flatMap { $0 + [Float](repeating: 0, count: width - $0.count) }
    }
    private func weights(_ d: Int, _ l: Int) -> [Float] {
        var w = [Float](repeating: 0, count: d * l)
        w[0] = 1; w[d + 1] = 1
        return w
    }
    // Independent Double model of the generic contract; ordinary inputs are well
    // away from quantization halfway cases and the computed-norm epsilon boundary.
    private func expected(_ row: [Float], activation: Bool, normalized: Bool, tiled: Bool = false) -> (codes: [Int8], scale: Float) {
        var values = row.map { activation ? max(0, Double($0)) : Double($0) }
        if normalized {
            let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
            values = norm > Double(epsilon) ? values.map { $0 / norm } : values.map { _ in 0 }
        }
        let maximum = values.map(abs).max() ?? 0
        let scale = tiled && !normalized ? maximum / 127 : max(maximum / 127, Double(epsilon))
        let invScale = tiled && !normalized && scale <= 1e-8 ? 0 : 1 / scale
        return (values.map { Int8(max(-127, min(127, ($0 * invScale).rounded()))) }, Float(scale))
    }

    func testSpecializedQuantizersNormalizeAfterBiasAndActivation() throws {
        try withLibraries { device, library, path in
            for (d, l, function) in [(768, 128, "neural_encode_768_to_128_kernel"),
                                      (768, 64, "neural_encode_768_to_64_kernel"),
                                      (384, 64, "neural_encode_384_to_64_kernel")] {
                for name in [function, "neural_encode_quantize_kernel"] {
                    let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
                    // A nonzero bias produces [-3,4] from [-4,5]. It must precede normalization.
                    let input = [[Float(-4), 5], [-1, 1], [-1, -4]]
                    let bias: [Float] = [1, -1] + [Float](repeating: 0, count: l - 2)
                    let inputBuffer = try buffer(rows(input, width: d), device)
                    let weightBuffer = try buffer(weights(d, l), device)
                    let biasBuffer = try buffer(bias, device)
                    for normalized in [false, true] {
                        for activation in [false, true] {
                            let output = try buffer([Int8](repeating: 42, count: input.count * l + 4), device)
                            let scales = try buffer([Float](repeating: 12345, count: input.count + 1), device)
                            var params = NeuralQuantizationParameters(numVectors: input.count,
                                config: Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l,
                                    useActivation: activation, normalizeLatent: normalized))
                            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                            encoder.setComputePipelineState(pipeline)
                            for (i, b) in [inputBuffer, weightBuffer, output, scales, biasBuffer].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
                            encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.size, index: 5)
                            encoder.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1),
                                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                            XCTAssertEqual(command.status, .completed, String(describing: command.error))
                            let codes = output.contents().assumingMemoryBound(to: Int8.self)
                            let actualScales = floats(scales, input.count + 1)
                            for i in input.indices {
                                let affine = [input[i][0] + 1, input[i][1] - 1] + [Float](repeating: 0, count: l - 2)
                                let want = expected(affine, activation: activation, normalized: normalized)
                                XCTAssertEqual(actualScales[i], want.scale, accuracy: 1e-8, "\(path), \(name), norm=\(normalized)")
                                XCTAssertEqual(Array(UnsafeBufferPointer(start: codes + i * l, count: l)), want.codes)
                            }
                            XCTAssertEqual(actualScales.last, 12345)
                            for i in (input.count * l)..<(input.count * l + 4) { XCTAssertEqual(codes[i], 42) }
                        }
                    }
                }
            }
        }
    }

    func testTiledQuantizationNormalizesWithoutChangingIntermediates() throws {
        try withLibraries { device, library, path in
            let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "neural_quantize_pass2")))
            for l in [1, 3, 33, 128, 257] {
                let first: [Float] = l == 1 ? [-3] : [-3, 4] + [Float](repeating: 0, count: l - 2)
                let vectors = [first, [Float](repeating: 0, count: l), [Float](repeating: 2, count: l),
                               [5e-8] + [Float](repeating: 0, count: l - 1), [2e-7] + [Float](repeating: 0, count: l - 1)]
                let original: [Float] = vectors.flatMap { $0 } + [12345]
                let input = try buffer(original, device)
                for width in [32, 96, 256] {
                    for normalized: UInt32 in [0, 1] {
                        let output = try buffer([Int8](repeating: 42, count: vectors.count * l + 4), device)
                        let scales = try buffer([Float](repeating: 12345, count: vectors.count + 1), device)
                        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                        encoder.setComputePipelineState(pipeline)
                        for (i, b) in [input, output, scales].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
                        for (i, value) in [UInt32(vectors.count), UInt32(l), normalized].enumerated() {
                            var word = value
                            encoder.setBytes(&word, length: 4, index: i + 3)
                        }
                        encoder.dispatchThreadgroups(MTLSize(width: vectors.count + 1, height: 1, depth: 1),
                                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                        XCTAssertEqual(command.status, .completed, String(describing: command.error))
                        XCTAssertEqual(floats(input, original.count), original)
                        let codes = output.contents().assumingMemoryBound(to: Int8.self)
                        for i in vectors.indices {
                            let want = expected(vectors[i], activation: false, normalized: normalized != 0, tiled: true)
                            XCTAssertEqual(floats(scales, vectors.count)[i], want.scale, accuracy: 1e-8, "\(path), L=\(l), norm=\(normalized)")
                            XCTAssertEqual(Array(UnsafeBufferPointer(start: codes + i * l, count: l)), want.codes)
                        }
                        for i in (vectors.count * l)..<(vectors.count * l + 4) { XCTAssertEqual(codes[i], 42) }
                        XCTAssertEqual(floats(scales, vectors.count + 1).last, 12345)
                    }
                }
            }
        }
    }

    func testPublicTiledNormalizationMatchesGenericQuantization() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        for (d, l) in [(5, 3), (35, 33), (384, 64), (768, 128)] {
            let config = Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l,
                                                       useActivation: false, normalizeLatent: false)
            try await kernel.loadWeights(encoderWeights: weights(d, l), decoderWeights: [Float](repeating: 0, count: d * l), config: config)
            let coordinates: [[Float]] = (0..<33).map { i in
                switch i % 5 {
                case 0: return [-3, 4]
                case 1: return [0, 0]
                case 2: return [5e-8, 0]
                case 3: return [2e-7, 0]
                default: return [-3, -4]
                }
            }
            let input = try buffer(rows(coordinates, width: d), device)
            for activation in [false, true] {
                var parameters = NeuralQuantizationParameters(numVectors: coordinates.count, config: config)
                parameters.normalizeLatent = 1; parameters.useActivation = activation ? 1 : 0
                let immutableParameters = parameters
                let tiledCodes = try buffer([Int8](repeating: 42, count: coordinates.count * l + 4), device)
                let genericCodes = try buffer([Int8](repeating: 42, count: coordinates.count * l + 4), device)
                let tiledScales = try buffer([Float](repeating: 12345, count: coordinates.count + 1), device)
                let genericScales = try buffer([Float](repeating: 12345, count: coordinates.count + 1), device)
                let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                try await kernel.encodeTiledV3(commandBuffer: command, input: input, output: tiledCodes,
                                               scale: tiledScales, parameters: parameters)
                command.commit(); await command.completed()
                XCTAssertEqual(command.status, .completed, String(describing: command.error))
                try await context.executeAndWait { _, encoder in
                    try kernel.encodeEncodeQuantize(into: encoder, input: input, output: genericCodes,
                                                    scale: genericScales, parameters: immutableParameters)
                }
                for (codes, scales) in [(tiledCodes, tiledScales), (genericCodes, genericScales)] {
                    let actualCodes = codes.contents().assumingMemoryBound(to: Int8.self)
                    for i in coordinates.indices {
                        let want = expected(coordinates[i] + [Float](repeating: 0, count: l - 2), activation: activation, normalized: true)
                        XCTAssertEqual(floats(scales, coordinates.count)[i], want.scale, accuracy: 1e-8)
                        XCTAssertEqual(Array(UnsafeBufferPointer(start: actualCodes + i * l, count: l)), want.codes)
                    }
                    for i in (coordinates.count * l)..<(coordinates.count * l + 4) { XCTAssertEqual(actualCodes[i], 42) }
                    XCTAssertEqual(floats(scales, coordinates.count + 1).last, 12345)
                }
            }
        }
    }
}
