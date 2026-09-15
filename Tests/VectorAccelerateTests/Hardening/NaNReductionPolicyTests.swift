import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-016: owner-approved NaN propagation for LSE and basic statistics.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NaNReductionPolicyTests: XCTestCase {
    private func withLibraries(
        _ body: (any MTLDevice, any MTLLibrary, String) throws -> Void
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        let url = try XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"),
                                "Debug gate must exercise the plugin metallib")
        try body(device, device.makeLibrary(URL: url), "plugin")
        #endif
        let runtime = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        try body(device, runtime, "runtime")
    }

    private func buffer<T: BitwiseCopyable>(_ values: [T], device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                       options: .storageModeShared))
    }

    private func floats(_ buffer: any MTLBuffer) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self),
                                  count: buffer.length / MemoryLayout<Float>.stride))
    }

    private func run(
        _ name: String, device: any MTLDevice, library: any MTLLibrary,
        buffers: [any MTLBuffer], groups: MTLSize = MTLSize(width: 1, height: 1, depth: 1),
        threads: MTLSize = MTLSize(width: 1, height: 1, depth: 1)
    ) throws {
        let function = try XCTUnwrap(library.makeFunction(name: name), "Missing kernel: \(name)")
        let pipeline = try device.makeComputePipelineState(function: function)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(name): \(String(describing: command.error))")
        XCTAssertNil(command.error)
    }

    private let nanValues = [Float(bitPattern: 0x7fc00001), Float(bitPattern: 0xffc00123),
                             Float(bitPattern: 0x7fa00001)]

    func testRowLSEPropagatesNaNBeforeInfinityAtEveryPosition() throws {
        try withLibraries { device, library, path in
            for width in [1, 16, 17] {
                var rows: [[Float]] = []
                for background: Float in [0, .infinity, -.infinity] {
                    for position in 0..<width {
                        var row = [Float](repeating: background, count: width)
                        row[position] = nanValues[position % nanValues.count]
                        rows.append(row)
                    }
                }
                let input = try buffer(rows.flatMap { $0 }, device: device)
                let n = try buffer([UInt32(rows.count)], device: device)
                let names = width % 4 == 0
                    ? ["logsumexp_row_kernel", "logsumexp_row_vectorized_kernel"] : ["logsumexp_row_kernel"]
                for name in names {
                    let d = try buffer([UInt32(name.contains("vectorized") ? width / 4 : width)], device: device)
                    let output = try buffer([Float](repeating: 12345, count: rows.count), device: device)
                    try run(name, device: device, library: library, buffers: [input, output, n, d],
                            threads: MTLSize(width: rows.count, height: 1, depth: 1))
                    let missing = floats(output).enumerated().filter { !$0.element.isNaN }.map { $0.offset }
                    XCTAssertTrue(missing.isEmpty, "\(path) \(name) width=\(width), non-NaN rows: \(missing)")
                }
            }
        }
    }

    private func reduce(_ data: [Float], width: Int, groups: Int,
                        device: any MTLDevice, library: any MTLLibrary) throws -> (Float, [Float], [Float]) {
        let input = try buffer(data, device: device)
        let maxima = try buffer([Float](repeating: 12345, count: groups), device: device)
        let sums = try buffer([Float](repeating: 12345, count: groups), device: device)
        let count = try buffer([UInt32(data.count)], device: device)
        let groupCount = try buffer([UInt32(groups)], device: device)
        let output = try buffer([Float(12345)], device: device)
        try run("logsumexp_reduce_pass1_kernel", device: device, library: library,
                buffers: [input, maxima, sums, count, groupCount],
                groups: MTLSize(width: groups, height: 1, depth: 1),
                threads: MTLSize(width: width, height: 1, depth: 1))
        try run("logsumexp_reduce_pass2_kernel", device: device, library: library,
                buffers: [maxima, sums, output, groupCount],
                threads: MTLSize(width: width, height: 1, depth: 1))
        return (floats(output)[0], floats(maxima), floats(sums))
    }

    func testTwoPassLSEPropagatesNaNAcrossLanesGroupsAndStrides() throws {
        try withLibraries { device, library, path in
            for (width, groups, count) in [(17, 3, 1), (17, 3, 513), (256, 4, 2051), (512, 3, 1025)] {
                let lanes = min(width, 256)
                let positions = Set([0, min(lanes - 1, count - 1), min(lanes, count - 1),
                                     min(lanes * groups, count - 1), count - 1]).sorted()
                for background: Float in [0, .infinity, -.infinity] {
                    for position in positions {
                        var data = [Float](repeating: background, count: count)
                        data[position] = nanValues[position % nanValues.count]
                        let (result, maxima, sums) = try reduce(data, width: width, groups: groups,
                                                               device: device, library: library)
                        let owner = (position % (lanes * groups)) / lanes
                        XCTAssertTrue(result.isNaN, "\(path) width=\(width) count=\(count) NaN=\(position) background=\(background)")
                        XCTAssertTrue(maxima[owner].isNaN, "\(path) pass1 must preserve the NaN marker")
                        XCTAssertTrue(sums[owner].isNaN, "\(path) pass1 NaN sum")
                    }
                }
            }
        }
    }

    func testTwoPassLSEControlsIncludingPositiveInfinityPartials() throws {
        try withLibraries { device, library, path in
            for (width, groups, count) in [(17, 3, 1), (17, 3, 513), (256, 4, 2051), (512, 3, 1025)] {
                for value: Float in [0, .infinity, -.infinity,
                                     .greatestFiniteMagnitude, -.greatestFiniteMagnitude] {
                    let data = [Float](repeating: value, count: count)
                    let (result, _, sums) = try reduce(data, width: width, groups: groups,
                                                      device: device, library: library)
                    if value != 0 {
                        // At finite extrema, adding log(count) rounds back to value.
                        XCTAssertEqual(result, value, path)
                    } else {
                        XCTAssertEqual(result, Float(log(Double(count))), accuracy: 2e-5, path)
                    }
                    // Valid infinity partials must not manufacture a NaN through Inf-Inf;
                    // otherwise pass2 cannot distinguish them from input NaN poison.
                    XCTAssertTrue(sums.allSatisfy { !$0.isNaN }, "\(path) valid input produced NaN partial sums")
                }
            }
        }
    }

    func testPass2NaNInEitherPartialFieldDominatesPositiveInfinity() throws {
        try withLibraries { device, library, path in
            for width in [17, 256, 512] {
                for poisonMaximum in [false, true] {
                    var maxima = [Float](repeating: 0, count: 300)
                    var sums = [Float](repeating: 1, count: 300)
                    maxima[0] = .infinity
                    if poisonMaximum { maxima[299] = nanValues[1] } else { sums[299] = nanValues[2] }
                    let maxBuffer = try buffer(maxima, device: device)
                    let sumBuffer = try buffer(sums, device: device)
                    let output = try buffer([Float(12345)], device: device)
                    let groups = try buffer([UInt32(300)], device: device)
                    try run("logsumexp_reduce_pass2_kernel", device: device, library: library,
                            buffers: [maxBuffer, sumBuffer, output, groups],
                            threads: MTLSize(width: width, height: 1, depth: 1))
                    XCTAssertTrue(floats(output)[0].isNaN, "\(path) width=\(width) poisonMaximum=\(poisonMaximum)")
                }
            }
        }
    }

    func testStatisticsNaNPropagatesThroughEveryAggregateButCount() throws {
        try withLibraries { device, library, path in
            for (width, count) in [(1, 1), (17, 7), (17, 513), (256, 2051), (512, 1025), (1024, 2051)] {
                for position in Set([0, min(width - 1, count - 1), min(width, count - 1), count - 1]).sorted() {
                    for background: Float in [2, .infinity, -.infinity] {
                        var data = [Float](repeating: background, count: count)
                        data[position] = nanValues[position % nanValues.count]
                        let input = try buffer(data, device: device)
                        let output = try buffer([Float](repeating: 12345, count: 6), device: device)
                        let params = try buffer([UInt32(count), 0], device: device)
                        try run("computeBasicStatistics", device: device, library: library,
                                buffers: [input, output, params],
                                threads: MTLSize(width: width, height: 1, depth: 1))
                        let values = floats(output)
                        XCTAssertTrue(values.prefix(5).allSatisfy(\.isNaN),
                                      "\(path) width=\(width) count=\(count) NaN=\(position): \(values)")
                        XCTAssertEqual(values[5], Float(count), "\(path) count includes NaNs")
                    }
                }
            }
        }
    }

    func testStatisticsFiniteAndEmptyKernelControls() throws {
        try withLibraries { device, library, path in
            for width in [1, 17, 256, 512, 1024] {
                for count in [0, 1, 513] {
                    let data = (0..<max(count, 1)).map { Float($0 % 5 - 2) }
                    let input = try buffer(data, device: device)
                    let output = try buffer([Float](repeating: 12345, count: 6), device: device)
                    let params = try buffer([UInt32(count), 0], device: device)
                    try run("computeBasicStatistics", device: device, library: library,
                            buffers: [input, output, params],
                            threads: MTLSize(width: width, height: 1, depth: 1))
                    var expected = [Float](repeating: 0, count: 6)
                    if count > 0 {
                        let values = data.prefix(count).map(Double.init)
                        let sum = values.reduce(0, +)
                        let mean = sum / Double(count)
                        let m2 = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
                        expected = [Float(mean), Float(m2), Float(values.min()!), Float(values.max()!), Float(sum), Float(count)]
                    }
                    for (actual, reference) in zip(floats(output), expected) {
                        XCTAssertEqual(actual, reference, accuracy: max(abs(reference) * 2e-5, 2e-5), path)
                    }
                }
            }
        }
    }

    func testPublicBasicStatisticsPropagatesNaNIncludingSingleton() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let kernel = try await StatisticsKernel(context: context)
        for data: [Float] in [[.nan], [1, .nan, 3], [.nan, .infinity], [-.infinity, .nan]] {
            let basic = try await kernel.computeBasicStatistics(data)
            let population = try await kernel.computeStatistics(data, config: Metal4StatisticsConfig(
                computeHigherMoments: false, computeQuantiles: false, biasCorrection: false))
            for result in [basic, population.basic] {
                XCTAssertEqual(result.count, data.count)
                XCTAssertTrue([result.mean, result.variance, result.standardDeviation, result.minimum,
                               result.maximum, result.range, result.sum].allSatisfy(\.isNaN))
            }
        }
        let singleton = try await kernel.computeBasicStatistics([3])
        XCTAssertEqual(singleton.variance, 0)
        XCTAssertEqual(singleton.standardDeviation, 0)
        XCTAssertEqual(singleton.range, 0)
    }

    func testPublicLSEUsesSameNaNPolicyForRowAndReduction() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let kernel = try await LogSumExpKernel(context: context)
        for width in [1, 16, 17, 513] {
            var row = [Float](repeating: .infinity, count: width)
            row[width - 1] = nanValues[1]
            let rows = try await kernel.rowwiseArray(input: [row])
            let reduced = try await kernel.reduceValue(input: row)
            XCTAssertTrue(rows[0].isNaN, "row width=\(width)")
            XCTAssertTrue(reduced.isNaN, "reduction width=\(width)")
        }
    }

    func testOtherStatisticsValidationRemainsFiniteOnly() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let context = try await Metal4Context()
        let kernel = try await StatisticsKernel(context: context)
        for data: [Float] in [[], [.infinity], [-.infinity], [1, .infinity]] {
            do {
                _ = try await kernel.computeBasicStatistics(data)
                XCTFail("Expected rejection of empty or infinity-only-invalid basic input")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            }
        }
        for config in [Metal4StatisticsConfig.default, .full,
                       Metal4StatisticsConfig(computeHigherMoments: false, computeQuantiles: true)] {
            do {
                _ = try await kernel.computeStatistics([1, .nan, 3], config: config)
                XCTFail("Moments/quantiles retain finite-input validation")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            }
        }
    }
}
