import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class OptionalBiasValidationTests: XCTestCase {
    private func buffer(_ values: [Float], device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
    }

    private func loadSignedProjection(_ kernel: NeuralQuantizationKernel, d: Int, l: Int) async throws {
        var weights = [Float](repeating: 0, count: d * l)
        // Each latent coordinate is +/- the last input coordinate, including the scalar tail.
        for j in 0..<l { weights[j * d + d - 1] = j.isMultiple(of: 2) ? 1 : -2 }
        try await kernel.loadWeights(encoderWeights: weights,
            decoderWeights: [Float](repeating: 0, count: d * l),
            config: Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l,
                useActivation: false, normalizeLatent: false))
    }

    // Missing buffer(3), incorrect fallback contents, row stride, or premature activation
    // breaks these exact signed projections. Both ragged shapes and the latent cap execute.
    func testFloatEncoderNoBiasPreservesSignedProjectionAndCanary() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        for (d, l) in [(5, 3), (129, 128)] {
            try await loadSignedProjection(kernel, d: d, l: l)
            var values = [Float](repeating: 0, count: 2 * d)
            values[d - 1] = -3
            values[2 * d - 1] = 4
            let input = try buffer(values, device: device)
            for activation in [false, true] {
                let output = try buffer([Float](repeating: 12345, count: 2 * l + 1), device: device)
                let params = NeuralQuantizationParameters(numVectors: 2,
                    config: Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l,
                        useActivation: activation, normalizeLatent: false))
                try await context.executeAndWait { _, encoder in
                    try kernel.encodeEncode(into: encoder, input: input, output: output, parameters: params)
                }
                let actual = output.contents().assumingMemoryBound(to: Float.self)
                for j in 0..<l {
                    let first: Float = j.isMultiple(of: 2) ? -3 : 6
                    let second: Float = j.isMultiple(of: 2) ? 4 : -8
                    XCTAssertEqual(actual[j], activation ? max(0, first) : first)
                    XCTAssertEqual(actual[l + j], activation ? max(0, second) : second)
                }
                XCTAssertEqual(actual[2 * l], 12345)
            }
        }
    }

    // Never submit this command buffer: the baseline accepts L=129 despite finite fallback capacity.
    func testFloatEncoderRejectsOverCapBeforeEncoderMutation() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        try await loadSignedProjection(kernel, d: 5, l: 3)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        defer { encoder.endEncoding() }
        let input = try buffer([0, 0, 0, 0, 1], device: device)
        let output = try buffer([Float](repeating: 12345, count: 129), device: device)
        encoder.label = "before rejected neural call"
        let params = NeuralQuantizationParameters(numVectors: 1,
            config: Metal4NeuralQuantizationConfig(inputDimension: 5, latentDimension: 129))
        XCTAssertThrowsError(try kernel.encodeEncode(into: encoder, input: input,
            output: output, parameters: params)) { error in
            XCTAssertEqual((error as? VectorError)?.kind, .invalidData)
        }
        XCTAssertEqual(encoder.label, "before rejected neural call")
    }

    func testFloatEncoderFallbackSurvivesUnloadAndReload() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        let input = try buffer([0, 0, 0, 0, -3], device: device)
        let params = NeuralQuantizationParameters(numVectors: 1,
            config: Metal4NeuralQuantizationConfig(inputDimension: 5, latentDimension: 3, useActivation: false))
        for _ in 0..<2 {
            try await loadSignedProjection(kernel, d: 5, l: 3)
            let output = try buffer([12345, 12345, 12345, 12345], device: device)
            try await context.executeAndWait { _, encoder in
                try kernel.encodeEncode(into: encoder, input: input, output: output, parameters: params)
            }
            XCTAssertEqual(Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: 4)), [-3, 6, -3, 12345])
            await kernel.unloadWeights()
            let queue = try XCTUnwrap(device.makeCommandQueue())
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            XCTAssertThrowsError(try kernel.encodeEncode(into: encoder, input: input, output: output, parameters: params))
            encoder.endEncoding()
        }
    }

    // Raw shader control, separate from wrapper no-bias coverage: bias precedes ReLU.
    func testRawFloatEncoderAddsExplicitBiasBeforeActivation() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let library = try await context.shaderCompiler.getDefaultLibrary()
        let function = try XCTUnwrap(library.makeFunction(name: "neural_encode_kernel"))
        let pipeline = try await device.makeComputePipelineState(function: function)
        let input = try buffer([-3, 4, -5], device: device)
        let weights = try buffer([1, 0, 0, 0, 1, 0, 0, 0, 1], device: device)
        let bias = try buffer([5, -6, 1], device: device)
        for activation in [false, true] {
            let output = try buffer([12345, 12345, 12345, 12345], device: device)
            let parameters = NeuralQuantizationParameters(numVectors: 1,
                config: Metal4NeuralQuantizationConfig(inputDimension: 3, latentDimension: 3, useActivation: activation))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                for (index, value) in [input, weights, output, bias].enumerated() {
                    encoder.setBuffer(value, offset: 0, index: index)
                }
                var params = parameters
                encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.stride, index: 4)
                encoder.dispatchThreads(MTLSize(width: 1, height: 3, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            XCTAssertEqual(Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: 4)),
                activation ? [2, 0, 0, 12345] : [2, -2, -4, 12345])
        }
    }
}
