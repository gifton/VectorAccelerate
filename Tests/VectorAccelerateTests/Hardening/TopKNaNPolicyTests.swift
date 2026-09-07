import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-016: real GPU readbacks, with literal contract fixtures and VectorCore 0.3.3 parity.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class TopKNaNPolicyTests: XCTestCase {
    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        let url = try XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))
        try body(device, device.makeLibrary(URL: url), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func buffer<T: BitwiseCopyable>(_ data: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        if data.isEmpty { return try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared)) }
        return try XCTUnwrap(device.makeBuffer(bytes: data, length: data.count * MemoryLayout<T>.stride,
                                              options: .storageModeShared))
    }

    private func read<T>(_ b: any MTLBuffer, as: T.Type) -> [T] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: T.self),
                                  count: b.length / MemoryLayout<T>.stride))
    }

    private func run(_ name: String, _ device: any MTLDevice, _ library: any MTLLibrary,
                     _ buffers: [Int: any MTLBuffer], groups: MTLSize = MTLSize(width: 1, height: 1, depth: 1),
                     width: Int = 1) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers { encoder.setBuffer(b, offset: 0, index: i) }
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(name): \(String(describing: command.error))")
        XCTAssertNil(command.error)
    }

    private enum Selector: CaseIterable { case batch, warp, general }

    private func select(_ rows: [[Float]], k: Int, maximum: Bool, selector: Selector, sorted: Bool = true,
                        device: any MTLDevice, library: any MTLLibrary) throws -> ([UInt32], [Float]) {
        let n = rows[0].count
        let strided = selector == .batch
        let inputStride = n + (strided ? 3 : 0)
        let outputStride = k + (strided ? 2 : 0)
        let input = try buffer(rows.flatMap { $0 + (strided ? [Float](repeating: -999, count: 3) : []) }, device)
        let values = try buffer([Float](repeating: 12345, count: rows.count * outputStride), device)
        let indices = try buffer([UInt32](repeating: 0xDEADBEEF, count: rows.count * outputStride), device)
        if selector == .batch {
            let params: [UInt32] = [UInt32(rows.count), UInt32(n), UInt32(k), UInt32(inputStride),
                                    UInt32(outputStride), (maximum ? 1 : 0) | (sorted ? 256 : 0)]
            try run("topk_select_batch_kernel", device, library,
                    [0: input, 1: values, 2: indices, 3: buffer(params, device)], width: rows.count)
        } else {
            let direction = maximum ? "descending" : "ascending"
            var bindings: [Int: any MTLBuffer] = [0: input, 1: indices, 2: values]
            let args = selector == .warp ? [rows.count, n, k] : [1, rows.count, n, k]
            for (i, arg) in args.enumerated() { bindings[i + 3] = try buffer([UInt32(arg)], device) }
            let name = selector == .warp ? "warp_select_small_k_" : "batch_select_k_nearest_"
            try run(name + direction, device, library, bindings,
                    groups: MTLSize(width: 1, height: rows.count, depth: 1), width: 32)
        }
        let ids = read(indices, as: UInt32.self)
        let vals = read(values, as: Float.self)
        if strided {
            for q in rows.indices {
                XCTAssertEqual(Array(ids[(q * outputStride + k)..<((q + 1) * outputStride)]), [0xDEADBEEF, 0xDEADBEEF])
                XCTAssertEqual(Array(vals[(q * outputStride + k)..<((q + 1) * outputStride)]), [12345, 12345])
            }
        }
        return (rows.indices.flatMap { Array(ids[($0 * outputStride)..<($0 * outputStride + k)]) },
                rows.indices.flatMap { Array(vals[($0 * outputStride)..<($0 * outputStride + k)]) })
    }

    private func check(_ result: ([UInt32], [Float]), scores: [Float], expected: [Int], k: Int,
                       maximum: Bool, label: String, sorted: Bool = true) {
        let real = min(k, scores.count)
        let want = expected.prefix(real).map(UInt32.init)
        if sorted { XCTAssertEqual(Array(result.0.prefix(real)), want, label) }
        else { XCTAssertEqual(Set(result.0.prefix(real)), Set(want), label) }
        for slot in 0..<real {
            let idx = Int(result.0[slot])
            guard scores.indices.contains(idx) else { XCTFail("\(label): invalid real index \(idx)"); continue }
            if scores[idx].isNaN { XCTAssertTrue(result.1[slot].isNaN, label) }
            else { XCTAssertEqual(result.1[slot].bitPattern, scores[idx].bitPattern, label) }
        }
        for slot in real..<k {
            XCTAssertEqual(result.0[slot], .max, label)
            XCTAssertEqual(result.1[slot], maximum ? -.infinity : .infinity, label)
        }
    }

    func testSelectorsOrderNaNsInfinitiesAndSignedZero() throws {
        let scores: [Float] = [Float(bitPattern: 0x7fc00001), 2, -.infinity, 2, .infinity,
                               -0.0, 0.0, Float(bitPattern: 0xffa00123), -3]
        try withLibraries { device, library, path in
            for selector in Selector.allCases {
                for maximum in [false, true] {
                    let expected = maximum ? [4, 1, 3, 5, 6, 8, 2, 0, 7] : [2, 8, 5, 6, 1, 3, 4, 0, 7]
                    for k in [1, 5, 9, 12] {
                        let result = try select([scores, scores], k: k, maximum: maximum, selector: selector,
                                                device: device, library: library)
                        for q in 0..<2 {
                            check((Array(result.0[(q*k)..<(q*k+k)]), Array(result.1[(q*k)..<(q*k+k)])),
                                  scores: scores, expected: expected, k: k, maximum: maximum,
                                  label: "\(path) \(selector) max=\(maximum) k=\(k) q=\(q)")
                        }
                    }
                }
            }
        }
    }

    func testHeapEvictionRetainsEarlierEqualCandidates() throws {
        try withLibraries { device, library, path in
            for selector in Selector.allCases {
                for maximum in [false, true] {
                    // Evict index 2, the worse tie, when the last candidate arrives.
                    let scores: [Float] = maximum ? [-2, -1, -2, 0] : [2, 1, 2, 0]
                    for sorted in (selector == .batch ? [true, false] : [true]) {
                        let result = try select([scores], k: 3, maximum: maximum, selector: selector, sorted: sorted,
                                                device: device, library: library)
                        check(result, scores: scores, expected: [3, 1, 0], k: 3, maximum: maximum,
                              label: "\(path) \(selector) max=\(maximum) sorted=\(sorted)", sorted: sorted)
                    }
                }
            }
        }
    }

    func testNaNHeapAdmissionAcrossLanesAndLargeKMatchesVectorCore() throws {
        try withLibraries { device, library, path in
            for selector in Selector.allCases {
                for maximum in [false, true] {
                    var scores = [Float](repeating: .nan, count: 513)
                    for i in [33, 127, 256, 512] { scores[i] = maximum ? -1 : 1 }
                    for k in (selector == .warp ? [3, 32] : [3, 32, 33, 128]) {
                        let reference = MetalComputeProvider.selectTopK(scores, k: k, largerIsCloser: maximum)
                        let literal = Array(([33, 127, 256, 512] + Array(0..<32)).prefix(k))
                        if k <= 32 { XCTAssertEqual(reference.map(\.index), literal) }
                        check(try select([scores], k: k, maximum: maximum, selector: selector, device: device, library: library),
                              scores: scores, expected: reference.map(\.index), k: k, maximum: maximum,
                              label: "\(path) \(selector) max=\(maximum) k=\(k)")
                    }
                }
            }
        }
    }

    func testAllNaNsAndEmptyRowsPreserveCountAndPadding() throws {
        try withLibraries { device, library, path in
            for selector in Selector.allCases {
                for scores: [Float] in [[], [.nan], [.nan, Float(bitPattern: 0xffc12345), Float(bitPattern: 0x7fa00001)]] {
                    for maximum in [false, true] {
                        check(try select([scores], k: 8, maximum: maximum, selector: selector, device: device, library: library),
                              scores: scores, expected: Array(scores.indices), k: 8, maximum: maximum,
                              label: "\(path) \(selector) n=\(scores.count) max=\(maximum)")
                    }
                }
            }
        }
    }

    func testFiniteControlsAndSubnormalOrdering() throws {
        try withLibraries { device, library, path in
            for selector in Selector.allCases {
                for scores: [Float] in [[3, 1, 2], [0, Float(bitPattern: 1), Float(bitPattern: 0x80000001), -0.0]] {
                    for maximum in [false, true] {
                        let expected = scores.count == 3 ? (maximum ? [0, 2, 1] : [1, 2, 0])
                            : (maximum ? [1, 0, 3, 2] : [2, 0, 3, 1])
                        check(try select([scores], k: 8, maximum: maximum, selector: selector, device: device, library: library),
                              scores: scores, expected: expected, k: 8, maximum: maximum, label: "\(path) \(selector)")
                    }
                }
            }
        }
    }

    func testStreamingChunksRetainNaNsAndGlobalIndexTies() throws {
        struct Config { var q: UInt32; var size: UInt32; var k: UInt32; var base: UInt64 }
        let scores: [Float] = [.nan, 2, 1, .infinity, 1, .nan]
        try withLibraries { device, library, path in
            for k in [3, 8] {
                let values = try buffer([Float](repeating: 12345, count: k), device)
                let ids = try buffer([UInt32](repeating: 0xDEADBEEF, count: k), device)
                let kb = try buffer([UInt32(k)], device), q = try buffer([UInt32(1)], device)
                try run("streaming_topk_init", device, library, [0: values, 1: ids, 2: kb, 3: q], width: k)
                // Reverse chunk submission to discriminate tie admission by global index.
                for base in [3, 0] {
                    let chunk = try buffer(Array(scores[base..<(base+3)]), device)
                    let config = try buffer([Config(q: 1, size: 3, k: UInt32(k), base: UInt64(base))], device)
                    try run("streaming_topk_process_chunk", device, library, [0: chunk, 2: values, 3: ids, 5: config])
                }
                try run("streaming_topk_finalize", device, library, [0: values, 1: ids, 2: kb, 3: q])
                check((read(ids, as: UInt32.self), read(values, as: Float.self)), scores: scores,
                      expected: [2, 4, 1, 3, 0, 5], k: k, maximum: false, label: "\(path) streaming k=\(k)")
            }
        }
    }

    func testSortedMergePlacesRealNaNsBeforePadding() throws {
        try withLibraries { device, library, path in
            let ri = try buffer([UInt32(2), 0, .max, .max, .max, .max], device)
            let rv = try buffer([Float(1), .nan, .infinity, .infinity, .infinity, .infinity], device)
            let ci = try buffer([UInt32(0), 1, 2, .max], device)
            let cv = try buffer([Float(1), .infinity, .nan, .infinity], device)
            let oi = try buffer([UInt32](repeating: 0xDEADBEEF, count: 6), device)
            let ov = try buffer([Float](repeating: 12345, count: 6), device)
            let params = try buffer([UInt32(1), 6, 4, 3], device)
            try run("merge_topk_sorted_kernel", device, library, [0: ri, 1: rv, 2: ci, 3: cv, 4: oi, 5: ov, 6: params])
            XCTAssertEqual(read(oi, as: UInt32.self), [2, 3, 4, 0, 5, .max], path)
            let out = read(ov, as: Float.self)
            XCTAssertEqual(Array(out.prefix(3)), [1, 1, .infinity], path)
            XCTAssertTrue(out[3].isNaN && out[4].isNaN, path)
            XCTAssertEqual(out[5], .infinity, path)
        }
    }

    func testFusedSelectionConsumesWinnersIncludingInfinityAndNaN() throws {
        let scores: [Float] = [.nan, 2, 1, .infinity, 1, .nan]
        try withLibraries { device, library, path in
            for (width, k) in [(32, 8), (256, 8), (32, 33)] {
                let input = try buffer(scores, device), query = try buffer([Float(0)], device)
                let ids = try buffer([UInt32](repeating: 0xDEADBEEF, count: k), device)
                let vals = try buffer([Float](repeating: 12345, count: k), device)
                try run("fused_l2_topk", device, library,
                        [0: query, 1: input, 2: ids, 3: vals, 4: buffer([UInt32(1)], device),
                         5: buffer([UInt32(6)], device), 6: buffer([UInt32(1)], device), 7: buffer([UInt32(k)], device)], width: width)
                check((read(ids, as: UInt32.self), read(vals, as: Float.self)),
                      scores: [.nan, 4, 1, .infinity, 1, .nan], expected: [2, 4, 1, 3, 0, 5], k: k,
                      maximum: false, label: "\(path) fused width=\(width) k=\(k)")
            }
        }
    }

    func testIVFSelectionOrdersComputedNaNsByOriginalIndex() throws {
        try withLibraries { device, library, path in
            for k in [8, 33] {
                let ids = try buffer([UInt32](repeating: 0xDEADBEEF, count: k), device)
                let vals = try buffer([Float](repeating: 12345, count: k), device)
                // CSR order differs from original vector order; ties use original indices.
                try run("ivf_list_search", device, library,
                        [0: buffer([Float(0)], device),
                         1: buffer([Float.nan, 2, 1, .infinity, 1, .nan], device),
                         2: buffer([UInt32(0), 6], device),
                         3: buffer([UInt32(5), 1, 4, 3, 2, 0], device),
                         4: buffer([UInt32(0)], device), 5: ids, 6: vals,
                         7: buffer([UInt32(1), 1, 1, 1, UInt32(k), 6], device)], width: 32)
                check((read(ids, as: UInt32.self), read(vals, as: Float.self)),
                      scores: [.nan, 4, 1, .infinity, 1, .nan], expected: [2, 4, 1, 3, 0, 5],
                      k: k, maximum: false, label: "\(path) IVF k=\(k)")
            }
        }
    }

    func testChunkedPublicAPIKeepsCPUAndGPUMergeInAgreement() async throws {
        let context = try await Metal4Context()
        let kernel = try await FusedL2TopKKernel(context: context)
        let device = context.device.rawDevice
        let scores: [Float] = [.nan, 2, 1, .infinity, 1, .nan]
        let query = try buffer([Float(0)], device), dataset = try buffer(scores, device)
        for gpuMerge in [false, true] {
            let result = try await kernel.execute(
                queries: query, dataset: dataset,
                parameters: FusedL2TopKParameters(numQueries: 1, numDataset: 6, dimension: 1, k: 9),
                config: Metal4FusedL2Config(maxDistanceMatrixBytes: 12,
                                            preferGPUMergeInChunkedFallback: gpuMerge))
            XCTAssertEqual(result.results(for: 0).map(\.index), [2, 4, 1, 3, 0, 5], "GPU merge=\(gpuMerge)")
            check((read(result.indices, as: UInt32.self), read(try XCTUnwrap(result.distances), as: Float.self)),
                  scores: [.nan, 4, 1, .infinity, 1, .nan], expected: [2, 4, 1, 3, 0, 5],
                  k: 9, maximum: false, label: "public GPU merge=\(gpuMerge)")
        }
    }

}
