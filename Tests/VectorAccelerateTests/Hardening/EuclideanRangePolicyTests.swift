import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-030: rooted L2 rescues accumulator range failures; squared L2/dot keep FP32 limits.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class EuclideanRangePolicyTests: XCTestCase {
    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func buffer<T: BitwiseCopyable>(_ data: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: data, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared))
    }

    private func run(_ name: String, _ device: any MTLDevice, _ library: any MTLLibrary,
                     _ buffers: [any MTLBuffer], width: Int = 1) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(name): \(String(describing: command.error))")
    }

    private let names = ["euclideanDistance", "batchEuclideanDistance", "l2_distance", "l2_distance_kernel", "soa_l2_distance"]

    private func gpu(_ name: String, a: [Float], b: [Float], sqrt: Bool = true, width: Int = 32,
                     device: any MTLDevice, library: any MTLLibrary) throws -> Float {
        let n = a.count
        let inputA = try buffer(a, device)
        let output = try buffer([Float](repeating: 12345, count: 2), device)
        // SoA has two candidates per lane: interleaving poison distinguishes the lane stride.
        let storedB: [Float] = name == "soa_l2_distance"
            ? stride(from: 0, to: n, by: 4).flatMap { Array(b[$0..<($0+4)]) + [Float](repeating: 999, count: 4) }
            : b
        let inputB = try buffer(storedB, device)
        var args: [any MTLBuffer] = [inputA, inputB, output]
        switch name {
        case "euclideanDistance": args += [try buffer([UInt32(n)], device)]
        case "batchEuclideanDistance": args += [try buffer([UInt32(n)], device), try buffer([UInt32(1)], device)]
        case "l2_distance": args += [try buffer([UInt32(n)], device), try buffer([UInt32(sqrt ? 1 : 0)], device)]
        case "l2_distance_kernel":
            args += [try buffer([UInt32(1), 1, UInt32(n), UInt32(n), UInt32(n), 2, sqrt ? 1 : 0], device)]
        default: args += [try buffer([UInt32(2), UInt32(n / 4), sqrt ? 1 : 0, 0], device)]
        }
        // Only reduction kernels use a full group; scalar kernels execute one candidate.
        try run(name, device, library, args, width: ["euclideanDistance", "l2_distance"].contains(name) ? width : 1)
        XCTAssertEqual(output.contents().advanced(by: 4).load(as: Float.self), 12345, "\(name): output guard")
        return output.contents().load(as: Float.self)
    }

    private func close(_ got: Float, _ expected: Double, _ label: String) {
        XCTAssertTrue(got.isFinite, label)
        XCTAssertEqual(Double(got), expected, accuracy: abs(expected) * 2e-5, label)
    }

    func testRootedGPUPathsRescueHugeAndTinyDifferences() throws {
        try withLibraries { device, library, path in
            for name in names {
                for n in [4, 384] {
                    for scale: Float in [1e20, 1e-20, 1e-30] {
                        var b = [Float](repeating: 0, count: n)
                        b[n-2] = 3 * scale; b[n-1] = 4 * scale
                        let expected = hypot(Double(b[n-2]), Double(b[n-1]))
                        close(try gpu(name, a: [Float](repeating: 0, count: n), b: b, device: device, library: library),
                              expected, "\(path) \(name) n=\(n) scale=\(scale)")
                    }
                }
            }
        }
    }

    func testRootedScalarTailsAndRaggedReductionWidths() throws {
        try withLibraries { device, library, path in
            for name in names where name != "soa_l2_distance" {
                for n in [1, 17, 65] {
                    var b = [Float](repeating: 0, count: n); b[n-1] = 1e20
                    for width in (name == "euclideanDistance" ? [1, 17, 100, 512] : [32]) {
                        close(try gpu(name, a: [Float](repeating: 0, count: n), b: b, width: width,
                                      device: device, library: library), Double(b[n-1]), "\(path) \(name) n=\(n) width=\(width)")
                    }
                }
            }
        }
    }

    func testTrueOverflowAndNonfiniteInputsRemainNonfinite() throws {
        try withLibraries { device, library, path in
            for name in names {
                let z = [Float](repeating: 0, count: 4)
                XCTAssertEqual(try gpu(name, a: z, b: [.greatestFiniteMagnitude, .greatestFiniteMagnitude, 0, 0], device: device, library: library), .infinity, "\(path) \(name)")
                XCTAssertEqual(try gpu(name, a: z, b: [.infinity, 0, 0, 0], device: device, library: library), .infinity, "\(path) \(name)")
                XCTAssertTrue(try gpu(name, a: z, b: [.infinity, .nan, 0, 0], device: device, library: library).isNaN, "\(path) \(name)")
                XCTAssertTrue(try gpu(name, a: [.infinity, 0, 0, 0], b: [.infinity, 0, 0, 0], device: device, library: library).isNaN, "\(path) \(name)")
            }
        }
    }

    func testSquaredL2RetainsItsOutputRangeAndOrdinaryControls() throws {
        try withLibraries { device, library, path in
            for name in names {
                let z = [Float](repeating: 0, count: 4)
                XCTAssertEqual(try gpu(name, a: z, b: [3, 4, 0, 0], device: device, library: library), 5, "\(path) \(name)")
                close(try gpu(name, a: z, b: [.greatestFiniteMagnitude, 0, 0, 0], device: device, library: library),
                      Double(Float.greatestFiniteMagnitude), "\(path) \(name) finite endpoint")
                let offset: Float = 1e30
                close(try gpu(name, a: [offset, 0, 0, 0], b: [offset.nextUp, 0, 0, 0], device: device, library: library),
                      Double(offset.nextUp) - Double(offset), "\(path) \(name) common offset")
                XCTAssertEqual(try gpu(name, a: z, b: z, device: device, library: library), 0, "\(path) \(name)")
                if ["l2_distance", "l2_distance_kernel", "soa_l2_distance"].contains(name) {
                    XCTAssertEqual(try gpu(name, a: z, b: [3, 4, 0, 0], sqrt: false, device: device, library: library), 25, path)
                    XCTAssertEqual(try gpu(name, a: z, b: [1e20, 0, 0, 0], sqrt: false, device: device, library: library), .infinity, path)
                }
            }
        }
    }

    func testCPUFallbacksRescueRangeFailures() async throws {
        for scale: Float in [1e20, 1e-20, 1e-30] {
            let a = [Float](repeating: 0, count: 17)
            var b = a; b[15] = 3 * scale; b[16] = 4 * scale
            let expected = hypot(Double(b[15]), Double(b[16]))
            close(try AccelerateFallback.euclideanDistance(a, b), expected, "AccelerateFallback")
            close(try FallbackProvider().l2Distance(from: a, to: b), expected, "FallbackProvider")
            for accelerate in [false, true] {
                let simd = SIMDFallback(configuration: SIMDConfiguration(useAccelerate: accelerate))
                close(try await simd.euclideanDistance(a, b), expected, "SIMDFallback accelerate=\(accelerate)")
            }
        }
        XCTAssertEqual(try FallbackProvider().l2DistanceSquared(from: [0], to: [1e20]), .infinity)
    }

    func testEngineAndBatchRoutingAgreeOnRescue() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)
        let batch = try await BatchDistanceEngine(context: context)
        for n in [4, 17] {
            let a = [Float](repeating: 0, count: n)
            var b = a; b[n-1] = 1e20
            close(try await engine.euclideanDistance(a, b), Double(b[n-1]), "engine n=\(n)")
            for count in [1, 100] {
                let result = try await engine.batchEuclideanDistance(query: a, candidates: Array(repeating: b, count: count))
                for value in result { close(value, Double(b[n-1]), "engine batch n=\(n) count=\(count)") }
            }
            for gpu in [false, true] {
                let result = try await batch.batchEuclideanDistance(query: a, candidates: [b], useGPU: gpu)
                close(try XCTUnwrap(result.first), Double(b[n-1]), "batch forced GPU=\(gpu)")
            }
        }
    }

    func testMappedDatasetEuclideanSearchRescuesRange() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("va3-030-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try MemoryMapManager.createDatasetFile(vectors: [[Float(1e20), 0, 0, 0]], at: url)
        let manager = MemoryMapManager()
        let dataset = try await manager.mapDataset(at: url)
        let results = try await manager.computeDistances(dataset: dataset, query: [0, 0, 0, 0], metric: .euclidean)
        close(try XCTUnwrap(results.first).distance, Double(Float(1e20)), "mapped dataset")
    }

    func testDotProductRangeLimitIsSeparateFromCancellationAccuracy() throws {
        // Exact real dot is zero; separate FP32 products overflow with opposing signs.
        // Out-of-range behavior is documented, not specified as one NaN/Inf bit pattern.
        let a: [Float] = [1e20, 1e20, 0, 0], b: [Float] = [1e20, -1e20, 0, 0]
        XCTAssertFalse(try AccelerateFallback.dotProduct(a, b).isFinite)
        try withLibraries { device, library, path in
            let output = try buffer([Float(12345)], device)
            try run("dotProduct", device, library, [buffer(a, device), buffer(b, device), output, buffer([UInt32(4)], device)], width: 32)
            XCTAssertFalse(output.contents().load(as: Float.self).isFinite, path)
        }
    }
}
