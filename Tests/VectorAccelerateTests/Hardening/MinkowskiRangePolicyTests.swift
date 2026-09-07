import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-022: remove silent mathematical substitutions, retain explicit fast-path range limits.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MinkowskiRangePolicyTests: XCTestCase {
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

    private func distance(_ data: [Float], p: Float, stable: Bool, device: any MTLDevice,
                          library: any MTLLibrary, chebyshev: Bool = false, query: [Float]? = nil) throws -> Float {
        let name = stable ? "minkowski_distance_stable" : "minkowski_distance_batch"
        let function = try XCTUnwrap(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)
        let output = try buffer([Float(12345)], device)
        let buffers = try [buffer(query ?? [Float](repeating: 0, count: data.count), device), buffer(data, device), output,
                           buffer([p], device), buffer([UInt32(1)], device), buffer([UInt32(1)], device),
                           buffer([UInt32(data.count)], device), buffer([UInt32(chebyshev ? 1 : 0)], device)]
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(name): \(String(describing: command.error))")
        return output.contents().load(as: Float.self)
    }

    private func relative(_ got: Float, _ expected: Double, _ label: String, tolerance: Double = 3e-5) {
        XCTAssertTrue(got.isFinite, label)
        XCTAssertEqual(Double(got), expected, accuracy: abs(expected) * tolerance, label)
    }

    func testSmallFractionalDistancesSurviveScalarAndVectorPaths() throws {
        try withLibraries { device, library, path in
            for stable in [false, true] {
                for p: Float in [0.5, 0.25, 1.5] {
                    for dimension in [1, 4, 5, 65] {
                        let value: Float = 1e-12
                        let expected = Double(value) * pow(Double(dimension), 1 / Double(p))
                        relative(try distance([Float](repeating: value, count: dimension), p: p, stable: stable,
                                              device: device, library: library), expected,
                                 "\(path) stable=\(stable) p=\(p) D=\(dimension)")
                    }
                }
            }
        }
    }

    func testNearIntegerPIsNotSilentlyRoundedToAnotherMetric() throws {
        try withLibraries { device, library, path in
            for stable in [false, true] {
                for p: Float in [0.9995, 1.0005, 1.9995, 2.0005] {
                    let data = [Float](repeating: 1, count: 64)
                    relative(try distance(data, p: p, stable: stable, device: device, library: library),
                             pow(64, 1 / Double(p)), "\(path) stable=\(stable) p=\(p)", tolerance: 1e-5)
                }
            }
        }
    }

    func testFastPathReportsOverflowWithoutArtificialCapOrRescue() throws {
        try withLibraries { device, library, path in
            for p: Float in [3, 5, 11] {
                let got = try distance([1e20], p: p, stable: false, device: device, library: library)
                XCTAssertEqual(got, .infinity, "\(path) p=\(p): explicit fast path may overflow the power before rooting")
                relative(try distance([1e20], p: p, stable: true, device: device, library: library),
                         Double(Float(1e20)), "\(path) stable p=\(p)")
            }
            // Powered intermediate fits FP32 but exceeds the former exp(87) ceiling.
            relative(try distance([4e7], p: 5, stable: false, device: device, library: library),
                     4e7, "\(path) finite power above old cap")
            // Underflow remains a documented fast-path limitation; stable mode rescues it.
            XCTAssertEqual(try distance([1e-20], p: 3, stable: false, device: device, library: library), 0, path)
            relative(try distance([1e-20], p: 3, stable: true, device: device, library: library),
                     Double(Float(1e-20)), "\(path) stable underflow rescue")
            // A genuinely overflowing final distance must also report infinity.
            for stable in [false, true] {
                XCTAssertEqual(try distance([Float.greatestFiniteMagnitude], p: 3, stable: stable,
                                            device: device, library: library, query: [-Float.greatestFiniteMagnitude]),
                               .infinity, "\(path) finite subtraction overflow")
                XCTAssertEqual(try distance([1e30, 1e30, 1e30, 1e30], p: 0.05, stable: stable,
                                            device: device, library: library), .infinity, path)
            }
        }
    }

    func testStableNormalizationAvoidsReciprocalAndRootRangeFailures() throws {
        try withLibraries { device, library, path in
            for p: Float in [1, 2, 12] {
                let value = Float.greatestFiniteMagnitude / 4
                relative(try distance([value, value, value, value], p: p, stable: true,
                                      device: device, library: library),
                         Double(value) * pow(4, 1 / Double(p)), "\(path) large scale p=\(p)")
            }
            for exponent in [2, 4, 8, 16, 32, 64, 128] {
                let endpoint = Float(Double(Float.greatestFiniteMagnitude) / pow(2, Double(exponent)))
                relative(try distance([endpoint, endpoint], p: 1 / Float(exponent), stable: true,
                                      device: device, library: library),
                         Double(Float.greatestFiniteMagnitude), "\(path) fractional finite endpoint exponent=\(exponent)")
            }
            // The root factor overflows Float32, but the rescaled final result fits.
            let value: Float = 1e-30, p: Float = 0.01
            relative(try distance([value, value, value, value], p: p, stable: true, device: device, library: library),
                     Double(value) * pow(4, 1 / Double(p)), "\(path) final rescaling", tolerance: 1e-4)
            // A ratio of 1e-60 underflows before its fractional power, losing a meaningful term.
            let mixed: [Float] = [1e20, 1e-30, 0, 0]
            let fractional: Float = 0.1
            let expected = pow(pow(Double(mixed[0]), Double(fractional)) + pow(Double(mixed[1]), Double(fractional)), 1 / Double(fractional))
            relative(try distance(mixed, p: fractional, stable: true, device: device, library: library),
                     expected, "\(path) fractional ratio", tolerance: 2e-5)
        }
    }

    func testZeroAndExactMetricControls() throws {
        try withLibraries { device, library, path in
            for stable in [false, true] {
                for p: Float in [0.5, 1, 2, 3, 4, 12] {
                    XCTAssertEqual(try distance([0, 0, 0, 0, 0], p: p, stable: stable, device: device, library: library), 0, path)
                    relative(try distance([1, 1, 1, 1], p: p, stable: stable, device: device, library: library),
                             pow(4, 1 / Double(p)), "\(path) control p=\(p)")
                }
            }
            XCTAssertEqual(try distance([3, 7, 2, 0], p: 12, stable: false, device: device, library: library, chebyshev: true), 7, path)
        }
    }

    func testPublicParameterValidationAndExplicitRouting() async throws {
        let context = try await Metal4Context()
        let kernel = try await MinkowskiDistanceKernel(context: context)
        for p: Float in [.infinity, -.infinity, .nan, 0, -1] {
            do {
                _ = try await kernel.computeDistances(queries: [[0]], dataset: [[1]], config: Metal4MinkowskiConfig(p: p))
                XCTFail("Accepted invalid p=\(p)")
            } catch { /* Expected public validation. */ }
        }
        for stable in [false, true] {
            let result = try await kernel.computeDistances(queries: [[0]], dataset: [[1e20]],
                config: Metal4MinkowskiConfig(p: 11, useStableComputation: stable))
            if stable { relative(result.distance(row: 0, col: 0), Double(Float(1e20)), "public stable") }
            else { XCTAssertEqual(result.distance(row: 0, col: 0), .infinity) }
        }
        // Finite positive exponent extremes are accepted; a single nonzero component
        // has the same Lp value for every p, without needing a representable reciprocal.
        for p: Float in [Float(bitPattern: 1), Float.greatestFiniteMagnitude] {
            let result = try await kernel.computeDistances(queries: [[0]], dataset: [[2]],
                config: Metal4MinkowskiConfig(p: p, useStableComputation: true))
            XCTAssertEqual(result.distance(row: 0, col: 0), 2)
        }
        XCTAssertFalse(Metal4MinkowskiConfig(p: 1.0005).isManhattan)
        XCTAssertFalse(Metal4MinkowskiConfig(p: 2.0005).isEuclidean)
    }
}
