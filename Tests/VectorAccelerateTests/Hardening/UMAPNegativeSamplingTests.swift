import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class UMAPNegativeSamplingTests: XCTestCase {
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

    // A distinct output is essential: input coordinates must remain immutable even
    // with one threadgroup or one sample. Padding catches missing tid/dimension bounds.
    func testRawKernelWritesSeparateOutputAndPreservesInput() throws {
        try withLibraries { device, library, path in
            let pipeline = try device.makeComputePipelineState(function:
                XCTUnwrap(library.makeFunction(name: "umap_negative_sample_kernel")))
            for d in [1, 2, 3, 50, 65] {
                let input: [Float] = [Float](repeating: 0, count: d) + [1] + [Float](repeating: 0, count: d - 1)
                let embedding = try buffer(input + [12345], device)
                let output = try buffer([Float](repeating: 12345, count: 2 * d + 1), device)
                let targets = try buffer([UInt32(1), 1, 0, 0], device)
                var params = UMAPParamsGPU(a: 0, b: 1, learningRate: 0.1, epsilon: 1,
                                          n: 2, d: UInt32(d), edgeCount: 0, negSampleRate: 2)
                let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(embedding, offset: 0, index: 0)
                encoder.setBuffer(targets, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<UMAPParamsGPU>.stride, index: 2)
                encoder.setBuffer(output, offset: 0, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed, String(describing: command.error))
                XCTAssertEqual(floats(embedding, 2 * d + 1), input + [12345], path)
                let result = floats(output, 2 * d + 1)
                let displacement: Float = 0.1 + 0.22 / 2.21
                for k in 0..<(2 * d) {
                    let expected: Float = k == 0 ? -displacement : (k == d ? 1 + displacement : 0)
                    XCTAssertEqual(result[k], expected, accuracy: 2e-6, "\(path), D=\(d), k=\(k)")
                }
                XCTAssertEqual(result.last, 12345)
            }
        }
    }

    // a=0, b=1, epsilon=1, lr=0.1 gives coefficient 0.2/(1+distance²).
    // Against a frozen target at 1, x=0 becomes -0.1 then -0.1-0.22/2.21.
    // This catches both racing target reads and accidentally freezing the source
    // across samples. All points target point 0 or 1, across threadgroup boundaries.
    func testStandaloneUsesFrozenTargetsAndSequentialSourceUpdates() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let n = 257
        let original: [Float] = (0..<n).flatMap { [Float($0 % 2), 0] }
        let targets: [UInt32] = (0..<n).flatMap { i -> [UInt32] in
            let target = UInt32(1 - (i % 2))
            return [target, target]
        }
        let targetBuffer = try buffer(targets, device)
        let params = UMAPParameters(a: 0, b: 1, learningRate: 0.1, negativeSampleRate: 2, epsilon: 1)
        let displacement: Float = 0.1 + 0.22 / 2.21
        for _ in 0..<3 {
            let embedding = try buffer(original + [12345, 12345], device)
            try await kernel.applyNegativeSampling(embedding: embedding, randomTargets: targetBuffer,
                                                   n: n, d: 2, params: params)
            let result = floats(embedding, n * 2 + 2)
            for i in 0..<n {
                XCTAssertEqual(result[i * 2], i % 2 == 0 ? -displacement : 1 + displacement, accuracy: 2e-6)
                XCTAssertEqual(result[i * 2 + 1], 0)
            }
            XCTAssertEqual(Array(result.suffix(2)), [12345, 12345])
        }
    }
    func testRawSelfInvalidTargetsAndZeroWork() throws {
        try withLibraries { device, library, path in
            let pipeline = try device.makeComputePipelineState(function:
                XCTUnwrap(library.makeFunction(name: "umap_negative_sample_kernel")))
            for (n, d, rate): (UInt32, UInt32, UInt32) in [(2, 2, 4), (2, 2, 0), (0, 2, 4), (2, 0, 4)] {
                let embedding = try buffer([Float(0), 0, 1, 0, 12345], device)
                let output = try buffer([Float](repeating: 12345, count: 5), device)
                // Each row contains self, one valid opposite point, n, and UInt32.max.
                let targets = try buffer([UInt32(0), 1, 2, .max, 1, 0, 2, .max], device)
                var params = UMAPParamsGPU(a: 0, b: 1, learningRate: 0.1, epsilon: 1,
                                          n: n, d: d, edgeCount: 0, negSampleRate: rate)
                let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
                let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(embedding, offset: 0, index: 0)
                encoder.setBuffer(targets, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<UMAPParamsGPU>.stride, index: 2)
                encoder.setBuffer(output, offset: 0, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed, String(describing: command.error))
                XCTAssertEqual(floats(embedding, 5), [0, 0, 1, 0, 12345], path)
                let expected: [Float] = n == 0 || d == 0 ? [Float](repeating: 12345, count: 5)
                    : (rate == 0 ? [0, 0, 1, 0, 12345] : [-0.1, 0, 1.1, 0, 12345])
                for (actual, want) in zip(floats(output, 5), expected) {
                    XCTAssertEqual(actual, want, accuracy: 2e-6, path)
                }
            }
        }
    }

    func testFusedOrderingScratchReuseAndUnretainedCommandBuffers() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let n = 257, d = 3
        let original: [Float] = (0..<n).flatMap { [Float($0 % 2), 0, 0] }
        let targets: [UInt32] = (0..<n).flatMap { i -> [UInt32] in
            let target = UInt32(1 - (i % 2))
            return [target, target]
        }
        let params = UMAPParameters(a: 0, b: 1, learningRate: 0.1, negativeSampleRate: 2, epsilon: 1)
        let displacement: Float = 0.1 + 0.22 / 2.21
        for useScratch in [false, true] {
            let first = try buffer(original + [12345], device)
            let second = try buffer(original + [12345], device)
            let targetBuffer = try buffer(targets, device)
            let gradients = try buffer([Float](repeating: 0.5, count: n * d), device)
            let scratch = try buffer([Float](repeating: 12345, count: n * d + 1), device)
            let descriptor = MTLCommandBufferDescriptor()
            descriptor.retainedReferences = !useScratch
            let queue = try XCTUnwrap(device.makeCommandQueue())
            let command = try XCTUnwrap(queue.makeCommandBuffer(descriptor: descriptor))
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            // Predecessor writes must be visible to the frozen target pass.
            kernel.encodeApplyGradients(into: encoder, embedding: first, gradients: gradients, n: n, d: d)
            if useScratch {
                try kernel.encodeNegativeSampling(into: encoder, embedding: first, randomTargets: targetBuffer,
                                                  scratch: scratch, n: n, d: d, params: params)
            } else {
                try kernel.encodeNegativeSampling(into: encoder, embedding: first, randomTargets: targetBuffer,
                                                  n: n, d: d, params: params)
            }
            // Consumer must see completed copy-back, with no caller barrier needed.
            kernel.encodeApplyGradients(into: encoder, embedding: first, gradients: gradients, n: n, d: d)
            if useScratch {
                try kernel.encodeNegativeSampling(into: encoder, embedding: second, randomTargets: targetBuffer,
                                                  scratch: scratch, n: n, d: d, params: params)
            } else {
                try kernel.encodeNegativeSampling(into: encoder, embedding: second, randomTargets: targetBuffer,
                                                  n: n, d: d, params: params)
            }
            encoder.endEncoding(); command.commit(); await command.completed()
            XCTAssertEqual(command.status, .completed, String(describing: command.error))
            for (output, shift): (any MTLBuffer, Float) in [(first, 1), (second, 0)] {
                let result = floats(output, n * d + 1)
                for i in 0..<n {
                    XCTAssertEqual(result[i * d], (i % 2 == 0 ? -displacement : 1 + displacement) + shift, accuracy: 3e-6)
                    XCTAssertEqual(result[i * d + 1], shift, accuracy: 2e-6)
                    XCTAssertEqual(result[i * d + 2], shift, accuracy: 2e-6)
                }
                XCTAssertEqual(result.last, 12345)
            }
            XCTAssertEqual(floats(scratch, n * d + 1).last, 12345)
            // Explicit retention is part of the unretained-command-buffer contract.
            withExtendedLifetime((first, second, targetBuffer, gradients, scratch)) {}
        }
    }

    func testEpochFreezesTargetsAfterAttractiveUpdates() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let embedding = try buffer([Float(0), 0, 1, 0], device)
        let params = UMAPParameters(a: 1, b: 1, learningRate: 0.1, negativeSampleRate: 2, epsilon: 1)
        try await kernel.executeEpoch(embedding: embedding,
            edges: buffer([UMAPEdge(source: 0, target: 1, weight: 1)], device),
            segmentStarts: buffer([UInt32(0), 1], device), segmentCounts: buffer([UInt32(1), 0], device),
            randomTargets: buffer([UInt32(1), 1, 0, 0], device), n: 2, d: 2, edgeCount: 1, params: params)
        // Attraction changes x to 0.1 and 0.9. epsilon clamps pow(distance², b)
        // to 1 here, giving repulsive coefficient 0.1/(1+distance²).
        let firstDistance = 0.8
        let x = 0.1 - 0.1 * firstDistance / (1 + firstDistance * firstDistance)
        let secondDistance = 0.9 - x
        let expected = x - 0.1 * secondDistance / (1 + secondDistance * secondDistance)
        let result = floats(embedding, 4)
        XCTAssertEqual(result[0], Float(expected), accuracy: 2e-6)
        XCTAssertEqual(result[2], Float(1 - expected), accuracy: 2e-6)
        XCTAssertEqual(result[1], 0); XCTAssertEqual(result[3], 0)
    }

    func testDefaultCurveMatchesDoubleSnapshotReference() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let n = 513, d = 7, rate = 5
        let initial: [Float] = (0..<(n * d)).map { Float(($0 * 13) % 101) / 37 - 1 }
        let targets: [UInt32] = (0..<(n * rate)).map { UInt32(($0 * 31 + 17) % n) }
        let params = UMAPParameters(learningRate: 0.17, negativeSampleRate: rate)
        let frozen = initial.map(Double.init)
        var expected = frozen
        for i in 0..<n {
            for sample in 0..<rate {
                let target = Int(targets[i * rate + sample])
                if target == i { continue }
                let delta = (0..<d).map { expected[i * d + $0] - frozen[target * d + $0] }
                let distanceSquared = delta.reduce(0) { $0 + $1 * $1 }
                let coefficient = min(4, max(-4, 2 * Double(params.b) * Double(params.learningRate)
                    / ((Double(params.epsilon) + distanceSquared)
                       * (1 + Double(params.a) * pow(max(distanceSquared, Double(params.epsilon)), Double(params.b))))))
                for k in 0..<d { expected[i * d + k] += coefficient * delta[k] }
            }
        }
        let embedding = try buffer(initial, device)
        try await kernel.applyNegativeSampling(embedding: embedding, randomTargets: buffer(targets, device),
                                               n: n, d: d, params: params)
        for (actual, want) in zip(floats(embedding, n * d), expected) {
            XCTAssertEqual(Double(actual), want, accuracy: 2e-5)
        }
    }

    func testEmptyInputsAreNoOpsOnBothPublicPaths() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let embedding = try buffer([Float(12345)], device)
        let targets = try buffer([UInt32.max], device)
        for (n, d, rate) in [(0, 2, 5), (2, 0, 5), (2, 2, 0)] {
            let params = UMAPParameters(negativeSampleRate: rate)
            try await kernel.applyNegativeSampling(embedding: embedding, randomTargets: targets, n: n, d: d, params: params)
            try await context.executeAndWait { _, encoder in
                let result = try kernel.encodeNegativeSampling(into: encoder, embedding: embedding,
                    randomTargets: targets, n: n, d: d, params: params)
                XCTAssertEqual(result.totalThreads, 0)
                let scratchResult = try kernel.encodeNegativeSampling(into: encoder, embedding: embedding,
                    randomTargets: targets, scratch: embedding, n: n, d: d, params: params)
                XCTAssertEqual(scratchResult.totalThreads, 0)
            }
            XCTAssertEqual(floats(embedding, 1), [12345])
        }
    }

    func testInvalidCountsAndShortInputsThrowBeforeEncoding() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let embedding = try buffer([Float](repeating: 12345, count: 4), device)
        let targets = try buffer([UInt32](repeating: 0, count: 4), device)
        for (n, d, rate) in [(-1, 2, 1), (2, -1, 1), (2, 2, -1), (Int.max, 2, 1),
                              (1, Int.max, 1), (1, 1, Int.max), (Int(UInt32.max), Int(UInt32.max), 1),
                              (2, 3, 1), (2, 2, 3)] {
            let params = UMAPParameters(negativeSampleRate: rate)
            do {
                try await kernel.applyNegativeSampling(embedding: embedding, randomTargets: targets,
                                                       n: n, d: d, params: params)
                XCTFail("Invalid shape must throw")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
            do {
                try await context.executeAndWait { _, encoder in
                    try kernel.encodeNegativeSampling(into: encoder, embedding: embedding,
                        randomTargets: targets, n: n, d: d, params: params)
                }
                XCTFail("Invalid fused shape must throw")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
        XCTAssertEqual(floats(embedding, 4), [Float](repeating: 12345, count: 4))
    }

    func testAliasingAndShortScratchAreRejected() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await UMAPGradientKernel(context: context)
        let embedding = try buffer([Float](repeating: 12345, count: 4), device)
        let targets = try buffer([UInt32](repeating: 0, count: 4), device)
        let short = try buffer([Float(12345)], device)
        for scratch in [embedding, targets, short] {
            do {
                try await context.executeAndWait { _, encoder in
                    try kernel.encodeNegativeSampling(into: encoder, embedding: embedding,
                        randomTargets: targets, scratch: scratch, n: 2, d: 2,
                        params: UMAPParameters(negativeSampleRate: 2))
                }
                XCTFail("Aliasing or undersized scratch must throw")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
        do {
            try await kernel.applyNegativeSampling(embedding: embedding, randomTargets: embedding,
                n: 2, d: 2, params: UMAPParameters(negativeSampleRate: 2))
            XCTFail("Aliasing inputs must throw")
        } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        XCTAssertEqual(floats(embedding, 4), [Float](repeating: 12345, count: 4))
    }
}
