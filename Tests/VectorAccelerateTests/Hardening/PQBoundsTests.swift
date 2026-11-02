import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class PQBoundsTests: XCTestCase {
    private func buffer<T: BitwiseCopyable>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
    }

    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func run(_ name: String, device: any MTLDevice, library: any MTLLibrary,
                     buffers: [any MTLBuffer], groups: MTLSize, width: Int = 1, sharedBytes: Int = 0) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
        if sharedBytes > 0 { encoder.setThreadgroupMemoryLength(sharedBytes, index: 0) }
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
    }

    private func shaderSource() throws -> String {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders")
        let source = try String(contentsOf: folder.appendingPathComponent("ProductQuantization.metal"), encoding: .utf8)
        let header = try String(contentsOf: folder.appendingPathComponent("Metal4Common.h"), encoding: .utf8)
        return source.replacingOccurrences(of: "#include \"Metal4Common.h\"", with: header)
    }

    private func instrumentedADC(_ device: any MTLDevice, transform: (inout String) throws -> Void) throws -> any MTLLibrary {
        var source = try shaderSource()
        let start = try XCTUnwrap(source.range(of: "kernel void pq_compute_distances_adc("))
        var function = String(source[start.lowerBound...])
        function = function.replacingOccurrences(of: "uint tid [[thread_position_in_grid]]",
            with: "device atomic_uint* violations [[buffer(4)]], uint tid [[thread_position_in_grid]]")
        try transform(&function)
        source.replaceSubrange(start.lowerBound..., with: function)
        let options = MTLCompileOptions(); options.languageVersion = .version4_0; options.mathMode = .fast
        return try device.makeLibrary(source: source, options: options)
    }

    /// K=257 would choose centroid 256 and wrap to byte zero. K=256 must still
    /// represent centroid 255; unsupported configurations publish 0xff instead.
    func testAssignmentCentroidByteBoundaryAndUnsupportedK() throws {
        try withLibraries { device, library, path in
            for k in [0, 1, 255, 256, 257] {
                let books = (0..<max(1, k)).map { Float($0) }
                let output = try buffer([UInt8](repeating: 123, count: 5), device)
                try run("pq_assignment_or_encoding", device: device, library: library,
                        buffers: [buffer([Float(max(0, k - 1))], device), buffer(books, device), output,
                                  buffer([UInt32(1), 1, 1, UInt32(k), 1], device)], groups: MTLSize(width: 3, height: 2, depth: 1))
                let bytes = output.contents().bindMemory(to: UInt8.self, capacity: 5)
                XCTAssertEqual(bytes[0], k == 0 || k > 256 ? 255 : UInt8(k - 1), "\(path), K=\(k)")
                for i in 1..<5 { XCTAssertEqual(bytes[i], 123, path) }
            }
        }
    }

    /// Intercept entry into the cooperative table load so an unsupported shape
    /// cannot fault the GPU during the pre-fix reproduction. Every group must
    /// take the uniform rejection before that load and before the barrier.
    func testADCCapGuardPrecedesSharedTableAccess() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let library = try instrumentedADC(device) { function in
            let firstLoad = try XCTUnwrap(function.range(of: "for(uint i = local_id; i < MK; i += threadgroup_size)"))
            function.insert(contentsOf: """
            if (config.K == 0 || config.K > 256 || config.M == 0 || (ulong)config.M * config.K > 8192) {
                if (local_id == 0) atomic_fetch_add_explicit(violations, 1u, memory_order_relaxed);
                return;
            }
            """, at: firstLoad.lowerBound)
        }
        for (m, k) in [(33, 256), (8193, 1), (0, 1), (1, 0), (1, 257), (1 << 24, 256)] {
            let violations = try buffer([UInt32(0)], device)
            let output = try buffer([Float](repeating: 12345, count: 8), device)
            let dummy = try buffer([UInt32(0)], device)
            try run("pq_compute_distances_adc", device: device, library: library,
                    buffers: [dummy, dummy, output, buffer([UInt32(5), UInt32(m), UInt32(m), UInt32(k), 1], device), violations],
                    groups: MTLSize(width: 3, height: 1, depth: 1), width: 3, sharedBytes: 16)
            XCTAssertEqual(violations.contents().load(as: UInt32.self), 0, "M=\(m), K=\(k) reached shared loads")
            let result = output.contents().bindMemory(to: Float.self, capacity: 8)
            for i in 0..<5 { XCTAssertTrue(result[i].isNaN, "Unsupported ADC must replace poison with NaN") }
            for i in 5..<8 { XCTAssertEqual(result[i], 12345) }
        }
    }

    /// A byte code can exceed K even though it fits UInt8. Detect the attempted
    /// lookup before reading uninitialized/out-of-range shared memory; crossing
    /// into another valid subspace is also invalid and must produce NaN.
    func testADCRejectsCodesOutsideTheirSubspace() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let library = try instrumentedADC(device) { function in
            let lookup = try XCTUnwrap(function.range(of: "approx_dist_sq += shared_dist_table[(ulong)m * K + code];"))
            function.replaceSubrange(lookup, with: """
            if (code >= K) {
                atomic_fetch_add_explicit(violations, 1u, memory_order_relaxed);
            } else {
                approx_dist_sq += shared_dist_table[(ulong)m * K + code];
            }
            """)
        }
        for k in [1, 3, 255] {
            let violations = try buffer([UInt32(0)], device)
            let output = try buffer([Float](repeating: 12345, count: 4), device)
            try run("pq_compute_distances_adc", device: device, library: library,
                    buffers: [buffer([UInt8(0), 0, UInt8(k), 0, 0, 255], device),
                              buffer([Float](repeating: 2, count: 2 * k), device), output,
                              buffer([UInt32(3), 2, 2, UInt32(k), 1], device), violations],
                    groups: MTLSize(width: 2, height: 1, depth: 1), width: 3, sharedBytes: ((2 * k * 4 + 15) / 16) * 16)
            XCTAssertEqual(violations.contents().load(as: UInt32.self), 0, "K=\(k)")
            let result = output.contents().bindMemory(to: Float.self, capacity: 4)
            XCTAssertEqual(result[0], 4)
            XCTAssertTrue(result[1].isNaN); XCTAssertTrue(result[2].isNaN)
            XCTAssertEqual(result[3], 12345)
        }
    }

    /// Invalid assignments must not update another subspace or the output guard.
    /// Physical guard capacity keeps the original bad writes inside the allocation.
    func testTrainingSkipsOutOfRangeAssignments() throws {
        try withLibraries { device, library, path in
            let capacity = 262
            let sums = try buffer([Float](repeating: 0, count: capacity), device)
            let counts = try buffer([UInt32](repeating: 0, count: capacity), device)
            try run("pq_train_update_accumulate", device: device, library: library,
                    buffers: [buffer([Float(1), 2, 4, 8, 16, 32], device),
                              buffer([UInt8(0), 1, 3, 255, 2, 0], device), sums, counts,
                              buffer([UInt32(3), 2, 2, 3, 1], device)], groups: MTLSize(width: 5, height: 3, depth: 1))
            let actualSums = sums.contents().bindMemory(to: Float.self, capacity: capacity)
            let actualCounts = counts.contents().bindMemory(to: UInt32.self, capacity: capacity)
            let expectedSums: [Float] = [1, 0, 16, 32, 2, 0]
            let expectedCounts: [UInt32] = [1, 0, 1, 1, 1, 0]
            for i in 0..<capacity {
                XCTAssertEqual(actualSums[i], i < 6 ? expectedSums[i] : 0, "\(path), sum \(i)")
                XCTAssertEqual(actualCounts[i], i < 6 ? expectedCounts[i] : 0, "\(path), count \(i)")
            }
            // K=256/code255 is valid. The same byte must not reach accumulators
            // when the complete configuration is unsupported.
            for k in [0, 256, 257] {
                let sums = try buffer([Float](repeating: 0, count: 260), device)
                let counts = try buffer([UInt32](repeating: 0, count: 260), device)
                try run("pq_train_update_accumulate", device: device, library: library,
                        buffers: [buffer([Float(7)], device), buffer([UInt8(255)], device), sums, counts,
                                  buffer([UInt32(1), 1, 1, UInt32(k), 1], device)],
                        groups: MTLSize(width: 3, height: 2, depth: 1))
                let actualSums = sums.contents().bindMemory(to: Float.self, capacity: 260)
                let actualCounts = counts.contents().bindMemory(to: UInt32.self, capacity: 260)
                for i in 0..<260 {
                    XCTAssertEqual(actualSums[i], k == 256 && i == 255 ? 7 : 0, "\(path), K=\(k), sum \(i)")
                    XCTAssertEqual(actualCounts[i], k == 256 && i == 255 ? 1 : 0, "\(path), K=\(k), count \(i)")
                }
            }
        }
    }

    func testADCAtAndBelowTableCapWithRaggedDispatch() throws {
        try withLibraries { device, library, path in
            for (m, k) in [(1, 1), (3, 3), (32, 255), (32, 256), (33, 248)] {
                let table = (0..<(m * k)).map { Float($0 / k + 1 + $0 % k % 3) }
                let codes = (0..<(5 * m)).map { UInt8(($0 / m + $0 % m) % k) }
                for width in [1, 3, 32, 64] {
                    let output = try buffer([Float](repeating: 12345, count: 8), device)
                    try run("pq_compute_distances_adc", device: device, library: library,
                            buffers: [buffer(codes, device), buffer(table, device), output,
                                      buffer([UInt32(5), UInt32(m), UInt32(m), UInt32(k), 1], device)],
                            groups: MTLSize(width: (5 + width - 1) / width + 1, height: 1, depth: 1), width: width,
                            sharedBytes: (m * k * 4 + 15) & ~15)
                    let result = output.contents().bindMemory(to: Float.self, capacity: 8)
                    for r in 0..<5 {
                        let expected = (0..<m).reduce(0) { $0 + ($1 + 1) + ((r + $1) % k) % 3 }
                        XCTAssertEqual(result[r], Float(expected), "\(path), M=\(m), K=\(k), width=\(width)")
                    }
                    for i in 5..<8 { XCTAssertEqual(result[i], 12345, path) }
                }
            }
        }
    }

    /// Actual libraries reject the same unsupported shapes as the instrumented
    /// red-first probe, and reject bad codes after the shared-table barrier.
    func testRawADCUnsupportedShapesAndCodesPublishNaN() throws {
        try withLibraries { device, library, path in
            for (m, k) in [(33, 256), (8193, 1), (0, 1), (1, 0), (1, 257), (1 << 24, 256)] {
                let output = try buffer([Float](repeating: 12345, count: 8), device)
                let dummy = try buffer([UInt32(0)], device)
                try run("pq_compute_distances_adc", device: device, library: library,
                        buffers: [dummy, dummy, output, buffer([UInt32(5), UInt32(m), UInt32(m), UInt32(k), 1], device)],
                        groups: MTLSize(width: 3, height: 1, depth: 1), width: 3, sharedBytes: 16)
                let result = output.contents().bindMemory(to: Float.self, capacity: 8)
                for i in 0..<5 { XCTAssertTrue(result[i].isNaN, path) }
                for i in 5..<8 { XCTAssertEqual(result[i], 12345, path) }
            }
            let output = try buffer([Float](repeating: 12345, count: 4), device)
            try run("pq_compute_distances_adc", device: device, library: library,
                    buffers: [buffer([UInt8(0), 2, 3, 0, 0, 255], device),
                              buffer([Float](repeating: 2, count: 6), device), output,
                              buffer([UInt32(3), 2, 2, 3, 1], device)],
                    groups: MTLSize(width: 2, height: 1, depth: 1), width: 3, sharedBytes: 32)
            let result = output.contents().bindMemory(to: Float.self, capacity: 4)
            XCTAssertEqual(result[0], 4, path)
            XCTAssertTrue(result[1].isNaN, path); XCTAssertTrue(result[2].isNaN, path)
            XCTAssertEqual(result[3], 12345, path)
        }
    }

    func testHostADCRejectsOversizedTablesBeforeBufferUse() async throws {
        let context = try await Metal4Context()
        let kernel = try await ProductQuantizationKernel(context: context)
        let device = context.device.rawDevice
        let dummy = try buffer([UInt32(0)], device)
        for (m, k) in [(33, 256), (8193, 1), (Int.max, 256)] {
            let config = Metal4PQConfig(dimension: m, M: m, K: k)
            let model = Metal4PQModel(codebooks: dummy, config: config, device: device)
            let encoded = Metal4EncodedVectors(codes: dummy, count: 1, config: config)
            do {
                _ = try await kernel.computeDistances(query: dummy, encodedVectors: encoded, model: model)
                XCTFail("Oversized ADC table must throw before allocation/UInt32 conversion/encoding")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
    }

    func testHostADCAtCapAndSmallUnalignedTables() async throws {
        let context = try await Metal4Context()
        let kernel = try await ProductQuantizationKernel(context: context)
        let device = context.device.rawDevice
        for (m, k) in [(1, 1), (3, 3), (32, 255), (32, 256), (33, 248)] {
            let config = Metal4PQConfig(dimension: m, M: m, K: k)
            let books = (0..<(m * k)).map { Float($0 % k) }
            let model = Metal4PQModel(codebooks: try buffer(books, device), config: config, device: device)
            let encoded = Metal4EncodedVectors(codes: try buffer([UInt8](repeating: UInt8(k - 1), count: 3 * m), device),
                                              count: 3, config: config)
            let output = try await kernel.computeDistances(query: buffer([Float](repeating: 0, count: m), device),
                                                           encodedVectors: encoded, model: model)
            let result = output.contents().bindMemory(to: Float.self, capacity: 3)
            for i in 0..<3 { XCTAssertEqual(result[i], Float(m * (k - 1) * (k - 1)), "M=\(m), K=\(k)") }
        }
    }

    func testTrainingAndEncodingRemainAvailableAboveADCTableCap() async throws {
        let context = try await Metal4Context()
        let kernel = try await ProductQuantizationKernel(context: context)
        let config = Metal4PQConfig(dimension: 33, M: 33, K: 256, trainIterations: 1)
        // Identical vectors make random centroid initialization deterministic.
        let (model, encoded) = try await kernel.trainAndEncode(data: [[Float](repeating: 2, count: 33)], config: config)
        XCTAssertEqual(encoded.count, 1)
        XCTAssertEqual(encoded.decode(index: 0, using: model), [Float](repeating: 2, count: 33))
        let codes = encoded.codes.contents().bindMemory(to: UInt8.self, capacity: 33)
        for i in 0..<33 { XCTAssertEqual(codes[i], 0) }
    }

    func testHostOperationsRejectNonpositiveK() async throws {
        let context = try await Metal4Context()
        let kernel = try await ProductQuantizationKernel(context: context)
        let device = context.device.rawDevice
        let dummy = try buffer([Float(0)], device)
        for k in [-1, 0] {
            let config = Metal4PQConfig(dimension: 1, M: 1, K: k)
            let model = Metal4PQModel(codebooks: dummy, config: config, device: device)
            let encoded = Metal4EncodedVectors(codes: dummy, count: 1, config: config)
            do {
                _ = try await kernel.train(data: dummy, count: 1, config: config)
                XCTFail("Training must reject K=\(k)")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
            do {
                _ = try await kernel.encode(vectors: dummy, count: 1, model: model)
                XCTFail("Encoding must reject K=\(k)")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
            do {
                _ = try await kernel.computeDistances(query: dummy, encodedVectors: encoded, model: model)
                XCTFail("ADC must reject K=\(k)")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
            do {
                _ = try await kernel.trainAndEncode(data: [], config: config)
                XCTFail("Convenience API must reject K before empty data allocation")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
    }
}
