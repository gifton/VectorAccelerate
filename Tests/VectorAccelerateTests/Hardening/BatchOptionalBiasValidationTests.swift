import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class BatchOptionalBiasValidationTests: XCTestCase {
    private let batchSize = 2
    private let rows = 2
    private let inner = 2
    private let columns = 3

    private func buffer(_ values: [Float], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
    }

    private func floats(_ buffer: any MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: count
        ))
    }

    private func inputs(_ device: any MTLDevice) throws -> (any MTLBuffer, any MTLBuffer) {
        // Two identity matrices make the hand-derived expected products equal B.
        let a: [Float] = [
            1, 0, 0, 1,
            1, 0, 0, 1,
        ]
        let b: [Float] = [
             1, -2,  3,
             4,  5, -6,
            -7,  8,  9,
            10, -11, 12,
        ]
        return (try buffer(a, device), try buffer(b, device))
    }

    private func parameters(
        batchSize: Int = 2,
        rows: Int = 2,
        columns: Int = 3,
        alpha: Float = 1,
        activation: Metal4ActivationType = .none,
        hasBias: Bool = false
    ) -> BatchFusedParameters {
        BatchFusedParameters(
            batchSize: batchSize,
            M: rows,
            K: inner,
            N: columns,
            config: Metal4BatchFusedConfig(alpha: alpha, hasBias: hasBias, activation: activation)
        )
    }

    private func encodeAndWait(
        kernel: BatchMatrixKernel,
        device: any MTLDevice,
        a: any MTLBuffer,
        b: any MTLBuffer,
        output: any MTLBuffer,
        bias: (any MTLBuffer)?,
        layout: BatchBiasLayout,
        parameters: BatchFusedParameters
    ) async throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        try kernel.encodeFused(
            into: encoder,
            batchA: a,
            batchB: b,
            output: output,
            bias: bias,
            biasLayout: layout,
            parameters: parameters
        )
        encoder.endEncoding()
        command.commit()
        await command.completed()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
    }

    /// Removing either active layout, requiring exact raw length, or consulting
    /// config.hasBias changes one of these independently derived results.
    func testActiveLayoutsAcceptMinimumAndOversizedBuffersIndependentOfHasBias() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await BatchMatrixKernel(context: context)
        let (a, b) = try inputs(device)

        let cases: [(String, [Float], BatchBiasLayout, Bool, [Float])] = [
            ("shared exact", [0.5, 2, -1], .sharedColumns, false,
             [1.5, 0, 2, 4.5, 7, -7, -6.5, 10, 8, 10.5, -9, 11]),
            ("shared oversized", [0.5, 2, -1, 99, 98], .sharedColumns, true,
             [1.5, 0, 2, 4.5, 7, -7, -6.5, 10, 8, 10.5, -9, 11]),
            ("per-batch exact", [0.5, 2, -1, -3, 4, 0.25], .perBatchColumns, false,
             [1.5, 0, 2, 4.5, 7, -7, -10, 12, 9.25, 7, -7, 12.25]),
            ("per-batch oversized", [0.5, 2, -1, -3, 4, 0.25, 99], .perBatchColumns, true,
             [1.5, 0, 2, 4.5, 7, -7, -10, 12, 9.25, 7, -7, 12.25]),
        ]

        for (name, biasValues, layout, hasBias, expected) in cases {
            let output = try buffer([Float](repeating: -12345, count: 14), device)
            try await encodeAndWait(
                kernel: kernel, device: device, a: a, b: b, output: output,
                bias: try buffer(biasValues, device), layout: layout,
                parameters: parameters(hasBias: hasBias)
            )
            let actual = floats(output, count: 14)
            for i in 0..<12 {
                XCTAssertEqual(actual[i], expected[i], accuracy: 1e-6, "\(name), element \(i)")
            }
            XCTAssertEqual(Array(actual.suffix(2)), [-12345, -12345], name)
        }

        #if os(macOS)
        // A second device is uncommon; when present, exercise the hardware branch here
        // after the unconditional active-layout assertions above.
        if let other = MTLCopyAllDevices().first(where: { $0.registryID != device.registryID }) {
            let output = try buffer([Float](repeating: -12345, count: 12), device)
            let foreignBias = try buffer([0.5, 2, -1], other)
            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            do {
                try kernel.encodeFused(
                    into: encoder, batchA: a, batchB: b, output: output,
                    bias: foreignBias, biasLayout: .sharedColumns, parameters: parameters()
                )
                XCTFail("active bias from another device must throw")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            } catch {
                XCTFail("expected VectorError.invalidData, got \(error)")
            }
            encoder.endEncoding()
        }
        #endif
    }

    /// Nil and explicit .none both disable bias, even when the legacy flag says
    /// otherwise or storage containing values was supplied.
    func testDisabledBiasModesIgnoreLayoutStorageAndHasBias() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await BatchMatrixKernel(context: context)
        let (a, b) = try inputs(device)
        let expected: [Float] = [1, -2, 3, 4, 5, -6, -7, 8, 9, 10, -11, 12]

        for (name, bias, layout): (String, (any MTLBuffer)?, BatchBiasLayout) in [
            ("nil shared-default", nil, .sharedColumns),
            ("nil per-batch", nil, .perBatchColumns),
            ("supplied none", try buffer([99, 98, 97], device), .none),
        ] {
            let output = try buffer([Float](repeating: -12345, count: 14), device)
            try await encodeAndWait(
                kernel: kernel, device: device, a: a, b: b, output: output,
                bias: bias, layout: layout, parameters: parameters(hasBias: true)
            )
            XCTAssertEqual(Array(floats(output, count: 14).prefix(12)), expected, name)
            XCTAssertEqual(Array(floats(output, count: 14).suffix(2)), [-12345, -12345], name)
        }
    }

    /// The missing guard used to encode a dispatch whose shader could read beyond
    /// the caller's active bias. Rejection must leave that output poisoned, while a
    /// subsequent valid operation on the same encoder still executes.
    func testShortActiveBiasThrowsBeforeEncodingAndEncoderRemainsUsable() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await BatchMatrixKernel(context: context)
        let (a, b) = try inputs(device)
        for (name, values, layout): (String, [Float], BatchBiasLayout) in [
            ("shared", [0.5, 2], .sharedColumns),
            ("per-batch", [0.5, 2, -1, -3, 4], .perBatchColumns),
        ] {
            let rejectedOutput = try buffer([Float](repeating: -12345, count: 12), device)
            let validOutput = try buffer([Float](repeating: -12345, count: 12), device)
            let queue = try XCTUnwrap(device.makeCommandQueue())
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            var didReject = false
            do {
                try kernel.encodeFused(
                    into: encoder, batchA: a, batchB: b, output: rejectedOutput,
                    bias: try buffer(values, device), biasLayout: layout,
                    parameters: parameters()
                )
                XCTFail("one-element-short \(name) bias must throw")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
                didReject = true
            } catch {
                XCTFail("expected VectorError.invalidData, got \(error)")
                didReject = true
            }
            guard didReject else {
                encoder.endEncoding() // Never commit an unsafe baseline dispatch.
                continue
            }

            try kernel.encodeFused(
                into: encoder, batchA: a, batchB: b, output: validOutput,
                bias: nil, biasLayout: .none, parameters: parameters()
            )
            encoder.endEncoding()
            command.commit()
            await command.completed()
            XCTAssertEqual(command.status, .completed, String(describing: command.error))
            XCTAssertEqual(floats(rejectedOutput, count: 12), [Float](repeating: -12345, count: 12), name)
            XCTAssertEqual(floats(validOutput, count: 12), [1, -2, 3, 4, 5, -6, -7, 8, 9, 10, -11, 12], name)
        }
    }

    /// Checked byte-count arithmetic must reject impossible products without
    /// dispatching or attempting to allocate their advertised size.
    func testActiveBiasByteCountOverflowThrowsBeforeEncoding() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await BatchMatrixKernel(context: context)
        let dummy = try buffer([Float(1)], device)

        for (layout, parameters): (BatchBiasLayout, BatchFusedParameters) in [
            (.sharedColumns, self.parameters(columns: Int(UInt32.max))),
            (.perBatchColumns, self.parameters(batchSize: Int(UInt32.max), columns: Int(UInt32.max))),
            (.perBatchColumns, self.parameters(batchSize: 1 << 30, columns: Int(UInt32.max))),
        ] {
            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            do {
                try kernel.encodeFused(
                    into: encoder, batchA: dummy, batchB: dummy, output: dummy,
                    bias: dummy, biasLayout: layout, parameters: parameters
                )
                XCTFail("unrepresentable active bias must throw")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            } catch {
                XCTFail("expected VectorError.invalidData, got \(error)")
            }
            encoder.endEncoding() // Deliberately discard: an unsafe dispatch must never reach the GPU.
        }
    }

    /// Disabled zero-work calls retain their empty dispatch geometry without making
    /// a bias buffer mandatory.
    func testDisabledBiasPreservesZeroWorkGeometry() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await BatchMatrixKernel(context: context)
        let dummy = try buffer([Float(1)], device)

        for params in [parameters(batchSize: 0), parameters(rows: 0), parameters(columns: 0)] {
            let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            let result = try kernel.encodeFused(
                into: encoder, batchA: dummy, batchB: dummy, output: dummy,
                bias: nil, biasLayout: .sharedColumns, parameters: params
            )
            encoder.endEncoding()
            XCTAssertTrue(
                result.threadgroups.width == 0 || result.threadgroups.height == 0 || result.threadgroups.depth == 0
            )
        }
    }

    /// Bias must be applied before activation; this also covers the high-level
    /// allocation path with requested bias present.
    func testHighLevelBiasContentAndActivationRemainEffective() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let kernel = try await BatchMatrixKernel(context: context)
        let a = Matrix(rows: 1, columns: 2, values: [1, 0])
        let b = Matrix(rows: 2, columns: 3, values: [-2, 1, 3, 0, 0, 0])
        let result = try await kernel.multiplyFused(
            batchA: [a], batchB: [b], bias: [1, -3, 0.5],
            config: Metal4BatchFusedConfig(hasBias: false, activation: .relu)
        )
        XCTAssertEqual(result.flattenedData(), [0, 0, 3.5])
    }
}
