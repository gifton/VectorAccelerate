import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-015: execute real kernels from both compilation paths; CPU fallback cannot mask them.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class FastMathPolicyTests: XCTestCase {
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

    private func checkCorrelation(scales: [Float]) throws {
        try withLibraries { device, library, path in
            for scale in scales {
                // Zero means, signed correlation, and an orthogonal control. All M2/covariance
                // accumulators remain representable; only the denominator arithmetic is stressed.
                let rows: [[Float]] = [[scale, -scale, scale, -scale],
                                      [-scale, scale, -scale, scale],
                                      [scale, scale, -scale, -scale]]
                let input = try buffer(rows.flatMap { $0 }, device: device)
                let output = try buffer([Float](repeating: 12345, count: 18), device: device)
                let params = try buffer([UInt32(4), UInt32(3)], device: device)
                try run("computeCorrelation", device: device, library: library,
                        buffers: [input, output, params], threads: MTLSize(width: 3, height: 3, depth: 1))
                let result = floats(output)
                for i in 0..<3 {
                    for j in 0..<3 {
                        let a = rows[i].map(Double.init)
                        let b = rows[j].map(Double.init)
                        let aa = a.reduce(0) { $0 + $1 * $1 }
                        let bb = b.reduce(0) { $0 + $1 * $1 }
                        let ab = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
                        let reference = Float(ab / (sqrt(aa) * sqrt(bb)))
                        XCTAssertEqual(result[i * 3 + j], reference, accuracy: 2e-5,
                                       "\(path) scale=\(scale) correlation[\(i),\(j)]")
                        let covariance = Float(ab / 3)
                        // Cancellation can leave an FMA rounding residual around zero;
                        // bound absolute error by the input product scale, not |covariance|.
                        let covarianceTolerance = Float(sqrt(aa) * sqrt(bb) / 3) * 2e-5
                        XCTAssertEqual(result[9 + i * 3 + j], covariance,
                                       accuracy: max(covarianceTolerance, Float.leastNormalMagnitude),
                                       "\(path) scale=\(scale) covariance[\(i),\(j)]")
                    }
                }
            }
        }
    }

    func testCorrelationOrdinaryControl() throws {
        try checkCorrelation(scales: [1, 16])
    }

    func testCorrelationFiniteAccumulatorsAtExtremeScales() throws {
        // Large product overflow, large reciprocal FTZ, and small product underflow.
        try checkCorrelation(scales: [1e10, 1e18, 6e18, 1e-18])
    }

    func testCorrelationConstantDataRemainsZero() throws {
        try withLibraries { device, library, path in
            let input = try buffer([Float](repeating: 3, count: 8), device: device)
            let output = try buffer([Float](repeating: 12345, count: 8), device: device)
            let params = try buffer([UInt32(4), UInt32(2)], device: device)
            try run("computeCorrelation", device: device, library: library,
                    buffers: [input, output, params], threads: MTLSize(width: 2, height: 2, depth: 1))
            XCTAssertEqual(floats(output), [Float](repeating: 0, count: 8), path)
        }
    }

    func testCorrelationUnequalVariancesAndOneConstantDataset() throws {
        try withLibraries { device, library, path in
            let input = try buffer([Float(1e18), -1e18, 1e18, -1e18,
                                    1e-18, -1e-18, 1e-18, -1e-18, 3, 3, 3, 3], device: device)
            let output = try buffer([Float](repeating: 12345, count: 18), device: device)
            let params = try buffer([UInt32(4), UInt32(3)], device: device)
            try run("computeCorrelation", device: device, library: library,
                    buffers: [input, output, params], threads: MTLSize(width: 3, height: 3, depth: 1))
            let expected: [Float] = [1, 1, 0, 1, 1, 0, 0, 0, 0]
            for (actual, reference) in zip(floats(output).prefix(9), expected) {
                XCTAssertEqual(actual, reference, accuracy: 2e-5, path)
            }
        }
    }

    func testCorrelationTinyResultWithUnequalVariancesInBothOrders() throws {
        try withLibraries { device, library, path in
            let large: [Float] = [1e18, -1e18, 1e-10, -1e-10]
            let small: [Float] = [0, 0, 1e-18, -1e-18]
            let aa = large.map(Double.init).reduce(0) { $0 + $1 * $1 }
            let bb = small.map(Double.init).reduce(0) { $0 + $1 * $1 }
            let ab = zip(large, small).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
            let reference = Float(ab / (sqrt(aa) * sqrt(bb)))
            XCTAssertGreaterThan(reference, Float.leastNormalMagnitude)
            for rows in [[large, small], [small, large]] {
                let input = try buffer(rows.flatMap { $0 }, device: device)
                let output = try buffer([Float](repeating: 12345, count: 8), device: device)
                let params = try buffer([UInt32(4), UInt32(2)], device: device)
                try run("computeCorrelation", device: device, library: library,
                        buffers: [input, output, params], threads: MTLSize(width: 2, height: 2, depth: 1))
                // About 1e-28: an ordinary absolute correlation tolerance would accept zero.
                for index in [1, 2] {
                    XCTAssertEqual(floats(output)[index], reference, accuracy: reference * 2e-5,
                                   "\(path) first dataset begins with \(rows[0][0])")
                }
            }
        }
    }

    private func checkHistograms(includeNonFinite: Bool) throws {
        try withLibraries { device, library, path in
            for kernel in ["uniformHistogram", "adaptiveHistogram", "logarithmicHistogram"] {
                for degenerate in (kernel == "uniformHistogram" ? [false, true] : [false]) {
                    for outliers in [false, true] {
                        let edges: [Float] = degenerate ? [1, 1, 1] : [1, 2.5, 4]
                        let finite: [Float] = [0.5, 1, 1.5, 2.5, 4, 8,
                                               .greatestFiniteMagnitude, -.greatestFiniteMagnitude]
                        let data = finite + (includeNonFinite ? [.nan, .infinity, -.infinity] : [])
                        let input = try buffer(data, device: device)
                        let edgeBuffer = try buffer(edges, device: device)
                        // Atomic histograms require initialized counts; nonzero seeds also check
                        // accumulation semantics rather than accidentally accepting a zero fill.
                        let output = try buffer([UInt32(7), UInt32(11)], device: device)
                        let params = try buffer([UInt32(data.count), 2, outliers ? 1 : 0, 0], device: device)
                        try run(kernel, device: device, library: library,
                                buffers: [input, edgeBuffer, output, params],
                                threads: MTLSize(width: 32, height: 1, depth: 1))
                        let result = Array(UnsafeBufferPointer(
                            start: output.contents().assumingMemoryBound(to: UInt32.self), count: 2))
                        let outlierCounts: [UInt32] = kernel == "logarithmicHistogram" ? [10, 15] : [11, 15]
                        let expected: [UInt32] = degenerate
                            ? (outliers ? [15, 11] : [8, 11])
                            : (outliers ? outlierCounts : [9, 13])
                        XCTAssertEqual(result, expected,
                                       "\(path) \(kernel) degenerate=\(degenerate) outliers=\(outliers)")
                    }
                }
            }
        }
    }

    func testHistogramsFiniteControl() throws {
        try checkHistograms(includeNonFinite: false)
    }

    func testHistogramsExcludeNaNAndInfinities() throws {
        try checkHistograms(includeNonFinite: true)
    }

    func testLogSumExpAndSoftmaxInfinityBranches() throws {
        try withLibraries { device, library, path in
            let rows: [[Float]] = [[-.infinity, -.infinity, -.infinity, -.infinity],
                                  [.infinity, 0, .infinity, -.infinity],
                                  [0, 0, 0, 0],
                                  [-.infinity, 0, 0, -.infinity]]
            let input = try buffer(rows.flatMap { $0 }, device: device)
            let n = try buffer([UInt32(rows.count)], device: device)
            let d = try buffer([UInt32(4)], device: device)
            for name in ["logsumexp_row_kernel", "logsumexp_row_vectorized_kernel"] {
                let dimension = name.contains("vectorized") ? try buffer([UInt32(1)], device: device) : d
                let output = try buffer([Float](repeating: 12345, count: rows.count), device: device)
                try run(name, device: device, library: library, buffers: [input, output, n, dimension],
                        threads: MTLSize(width: rows.count, height: 1, depth: 1))
                let result = floats(output)
                XCTAssertEqual(result[0], -.infinity, "\(path) \(name)")
                XCTAssertEqual(result[1], .infinity, "\(path) \(name)")
                XCTAssertEqual(result[2], Float(log(4.0)), accuracy: 1e-6, "\(path) \(name)")
                XCTAssertEqual(result[3], Float(log(2.0)), accuracy: 1e-6, "\(path) \(name)")
            }
            for name in ["softmax_row_kernel", "softmax_row_efficient_kernel"] {
                let output = try buffer([Float](repeating: 12345, count: rows.count * 4), device: device)
                let threads = name == "softmax_row_kernel"
                    ? MTLSize(width: 4, height: rows.count, depth: 1)
                    : MTLSize(width: rows.count, height: 1, depth: 1)
                try run(name, device: device, library: library, buffers: [input, output, n, d], threads: threads)
                let expected: [Float] = [0, 0, 0, 0, 0.5, 0, 0.5, 0,
                                         0.25, 0.25, 0.25, 0.25, 0, 0.5, 0.5, 0]
                for (actual, reference) in zip(floats(output), expected) {
                    XCTAssertEqual(actual, reference, accuracy: 1e-6, "\(path) \(name)")
                }
            }
            // Exercise both reduction passes, including an empty partial group and a group
            // containing +Inf. Pass-1's +Inf partial sum is not a public result contract.
            for (row, expected) in zip(rows, [-Float.infinity, Float.infinity, Float(log(4.0)), Float(log(2.0))]) {
                let data = try buffer(row, device: device)
                let maxima = try buffer([Float](repeating: 12345, count: 2), device: device)
                let sums = try buffer([Float](repeating: 12345, count: 2), device: device)
                let groups = try buffer([UInt32(2)], device: device)
                let output = try buffer([Float(12345)], device: device)
                try run("logsumexp_reduce_pass1_kernel", device: device, library: library,
                        buffers: [data, maxima, sums, d, groups],
                        groups: MTLSize(width: 2, height: 1, depth: 1),
                        threads: MTLSize(width: 32, height: 1, depth: 1))
                XCTAssertEqual(floats(maxima)[1], -.infinity, "\(path) empty partial maximum")
                XCTAssertEqual(floats(sums)[1], 0, "\(path) empty partial sum")
                try run("logsumexp_reduce_pass2_kernel", device: device, library: library,
                        buffers: [maxima, sums, output, groups],
                        threads: MTLSize(width: 32, height: 1, depth: 1))
                if expected.isInfinite {
                    XCTAssertEqual(floats(output)[0], expected, "\(path) two-pass")
                } else {
                    XCTAssertEqual(floats(output)[0], expected, accuracy: 1e-6, "\(path) two-pass")
                }
            }
        }
    }

    func testLogSumExpFiniteExtremaAreNotInfinity() throws {
        try withLibraries { device, library, path in
            let limit = Float.greatestFiniteMagnitude
            let input = try buffer([Float](repeating: limit, count: 4)
                                   + [Float](repeating: -limit, count: 4), device: device)
            let n = try buffer([UInt32(2)], device: device)
            for name in ["logsumexp_row_kernel", "logsumexp_row_vectorized_kernel"] {
                let d = try buffer([UInt32(name.contains("vectorized") ? 1 : 4)], device: device)
                let output = try buffer([Float](repeating: 12345, count: 2), device: device)
                try run(name, device: device, library: library, buffers: [input, output, n, d],
                        threads: MTLSize(width: 2, height: 1, depth: 1))
                // Adding log(4) rounds back to these finite endpoints in Float.
                XCTAssertEqual(floats(output), [limit, -limit], "\(path) \(name)")
            }
        }
    }
}
