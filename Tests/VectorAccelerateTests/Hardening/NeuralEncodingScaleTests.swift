import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NeuralEncodingScaleTests: XCTestCase {
    private func loadIdentity(_ kernel: NeuralQuantizationKernel, d: Int, l: Int, normalize: Bool = false) async throws {
        var encoder = [Float](repeating: 0, count: d * l)
        var decoder = encoder
        for i in 0..<l { encoder[i * d + i] = 1; decoder[i * l + i] = 1 }
        try await kernel.loadWeights(encoderWeights: encoder, decoderWeights: decoder,
            config: Metal4NeuralQuantizationConfig(inputDimension: d, latentDimension: l,
                                                  useActivation: false, normalizeLatent: normalize))
    }

    // Exact INT8 coordinates at different magnitudes isolate metadata loss from
    // quantization error. This fails if either decoder broadcasts an average scale.
    func testBothDecodersPreserveEachRowsMagnitude() async throws {
        let context = try await Metal4Context()
        let kernel = try await NeuralQuantizationKernel(context: context)
        for (d, l) in [(4, 4), (35, 33), (384, 64), (768, 64), (768, 128)] {
            try await loadIdentity(kernel, d: d, l: l)
            var input = [[Float]](repeating: [Float](repeating: 0, count: d), count: 4)
            for (i, factor): (Int, Float) in [Float(1), 10, 0.01].enumerated() {
                input[i][0] = 127 * factor
                input[i][l - 1] = -63 * factor
            }
            let encoded = try await kernel.encode(input)
            XCTAssertEqual(encoded.scales.count, 4)
            for (actual, want) in zip(encoded.scales, [Float(1), 10, 0.01, 1e-7]) {
                XCTAssertEqual(actual, want, accuracy: max(1e-8, want * 2e-6))
            }
            let nested = try await kernel.decode(encoded)
            let flat = try await kernel.decodeFlat(encoded)
            for i in input.indices {
                for j in 0..<d {
                    let tolerance = max(1e-6, abs(input[i][j]) * 2e-6)
                    XCTAssertEqual(nested[i][j], input[i][j], accuracy: tolerance, "D=\(d), L=\(l), row=\(i)")
                    XCTAssertEqual(flat[i * d + j], input[i][j], accuracy: tolerance)
                }
            }
        }
    }

    func testNormalizedRowsRetainDifferentScales() async throws {
        let context = try await Metal4Context()
        let kernel = try await NeuralQuantizationKernel(context: context)
        try await loadIdentity(kernel, d: 4, l: 4, normalize: true)
        let encoded = try await kernel.encode([[-3, 4, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]])
        // [-.6,.8] quantizes to [-95,127] with scale .8/127; [0,1] uses 1/127.
        let expected: [Float] = [-95 * (0.8 / 127), 0.8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        let flat = try await kernel.decodeFlat(encoded)
        for (actual, want) in zip(flat, expected) { XCTAssertEqual(actual, want, accuracy: 2e-6) }
    }

    func testResultSurvivesPoolReuseAndWeightReload() async throws {
        let context = try await Metal4Context()
        let kernel = try await NeuralQuantizationKernel(context: context)
        try await loadIdentity(kernel, d: 4, l: 4)
        let original: [[Float]] = [[127, 0, 0, 0], [1270, 0, 0, 0]]
        let encoded = try await kernel.encode(original)
        await kernel.unloadWeights()
        try await loadIdentity(kernel, d: 4, l: 4)
        for _ in 0..<4 { _ = try await kernel.encode([[12700, 0, 0, 0], [127000, 0, 0, 0]]) }
        let decoded = try await kernel.decode(encoded)
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(encoded.scales, [1, 10])
    }
    func testMalformedResultShapesAreRejectedByBothDecoders() async throws {
        let context = try await Metal4Context()
        let kernel = try await NeuralQuantizationKernel(context: context)
        try await loadIdentity(kernel, d: 4, l: 4)
        for (n, l, codeCount, scales): (Int, Int, Int, [Float]) in [
            (2, 4, 8, [1]), (2, 4, 8, [1, 2, 3]), (2, 4, 7, [1, 2]),
            (2, 4, 9, [1, 2]), (2, 3, 6, [1, 2]), (0, 4, 0, []),
            (-1, 4, 0, []), (Int.max, 4, 0, []), (1, Int.max, 0, [1])
        ] {
            let encoded = Metal4NeuralEncodingResult(latentCodes: Data(repeating: 0, count: codeCount),
                numVectors: n, latentDimension: l, scales: scales, encodingTime: 0)
            do {
                _ = try await kernel.decode(encoded)
                XCTFail("Malformed result must be rejected")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
            do {
                _ = try await kernel.decodeFlat(encoded)
                XCTFail("Malformed flat result must be rejected")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
    }

    func testRaggedInputIsRejectedBeforeEncoding() async throws {
        let context = try await Metal4Context()
        let kernel = try await NeuralQuantizationKernel(context: context)
        try await loadIdentity(kernel, d: 4, l: 4)
        do {
            _ = try await kernel.encode([[1, 2, 3, 4], [1, 2, 3]])
            XCTFail("Ragged input must be rejected")
        } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
    }
    func testDirectDecoderFallbacksConsumePerVectorScales() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await NeuralQuantizationKernel(context: context)
        for l in [3, 4] {
            try await loadIdentity(kernel, d: 5, l: l)
            let original: [[Float]] = [[127, 0, -63, 0, 0], [1270, 0, -630, 0, 0], [0, 0, 0, 0, 0]]
            let encoded = try await kernel.encode(original)
            let codes = try XCTUnwrap(device.makeBuffer(bytes: [UInt8](encoded.latentCodes),
                length: encoded.latentCodes.count, options: .storageModeShared))
            let scales = try XCTUnwrap(device.makeBuffer(bytes: encoded.scales, length: 12, options: .storageModeShared))
            let params = NeuralQuantizationParameters(numVectors: 3,
                config: Metal4NeuralQuantizationConfig(inputDimension: 5, latentDimension: l, useActivation: false))
            for width in [16, 32, 64, 128, 256] {
                let poison = [Float](repeating: 12345, count: 16)
                let output = try XCTUnwrap(device.makeBuffer(bytes: poison, length: 64, options: .storageModeShared))
                try await context.executeAndWait { _, encoder in
                    // Width 16 and ragged L=3 select the original scalar fallback;
                    // the other widths exercise threadgroup variants when L=4.
                    try kernel.encodeDequantizeDecodeWithThreadgroupSize(into: encoder, input: codes,
                        scale: scales, output: output, parameters: params, threadgroupSize: width)
                }
                let actual = output.contents().assumingMemoryBound(to: Float.self)
                for (i, want) in original.flatMap({ $0 }).enumerated() { XCTAssertEqual(actual[i], want, accuracy: 1e-5, "L=\(l), width=\(width)") }
                XCTAssertEqual(actual[15], 12345)
            }
        }
    }
}
