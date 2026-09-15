import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

final class IVFListInstrumentationTests: XCTestCase {
    private func withPipelines(_ body: (any MTLDevice, any MTLComputePipelineState, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        func check(_ library: any MTLLibrary, _ path: String) throws {
            let function = try XCTUnwrap(library.makeFunction(name: "ivf_list_search"))
            try body(device, device.makeComputePipelineState(function: function), path)
        }
        #if DEBUG
        try check(device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try check(KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    // Check actual compiled storage before dispatch: the old instrumented kernel
    // aborts during encoding instead of reporting a recoverable command failure.
    func testPipelineFitsInstrumentedThreadgroupMemoryBudget() throws {
        try withPipelines { device, pipeline, path in
            print("IVF list \(path): static=\(pipeline.staticThreadgroupMemoryLength), limit=\(device.maxThreadgroupMemoryLength)")
            XCTAssertLessThanOrEqual(pipeline.staticThreadgroupMemoryLength, device.maxThreadgroupMemoryLength, path)
        }
    }

    private func buffer<T: BitwiseCopyable>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        if values.isEmpty { return try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared)) }
        return try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
    }

    private func run(_ device: any MTLDevice, _ pipeline: any MTLComputePipelineState,
                     values: [Float], dimension: Int, offsets: [UInt32], ids: [UInt32],
                     queries: [Float], selected: [[UInt32]], width: Int = 256, k: Int)
        throws -> ([UInt32], [Float]) {
        guard pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
            XCTFail("IVF list pipeline exceeds the device memory budget")
            return ([], [])
        }
        let outputCount = queries.count * k
        let buffers = try [
            buffer(queries.flatMap { [Float](repeating: $0, count: dimension) }, device),
            buffer(values.flatMap { [Float](repeating: $0, count: dimension) }, device),
            buffer(offsets, device), buffer(ids, device), buffer(selected.flatMap { $0 }, device),
            buffer([UInt32](repeating: 0xDEADBEEF, count: outputCount + 2), device),
            buffer([Float](repeating: 12345, count: outputCount + 2), device)
        ]
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        var params = IVFListSearchParameters(numQueries: queries.count, numCentroids: offsets.count - 1,
            dimension: dimension, nprobe: selected[0].count, k: k, maxCandidatesPerQuery: values.count)
        encoder.setBytes(&params, length: MemoryLayout<IVFListSearchParameters>.stride, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: queries.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
        let outputIDs = Array(UnsafeBufferPointer(start: buffers[5].contents().assumingMemoryBound(to: UInt32.self), count: outputCount + 2))
        let distances = Array(UnsafeBufferPointer(start: buffers[6].contents().assumingMemoryBound(to: Float.self), count: outputCount + 2))
        XCTAssertEqual(Array(outputIDs.suffix(2)), [0xDEADBEEF, 0xDEADBEEF])
        XCTAssertEqual(Array(distances.suffix(2)), [12345, 12345])
        return (Array(outputIDs.prefix(outputCount)), Array(distances.prefix(outputCount)))
    }

    func testSelectionPreservesRetainedPoolAcrossSplitAndWidths() throws {
        try withPipelines { device, pipeline, path in
            var values = (0..<2051).map { Float(1000 + $0) }
            for i in 0..<9 { values[i * 256] = Float(i) }
            for i in [31, 63, 127, 255] { values[i] = 2 }
            let ids = (0..<values.count).map { UInt32(values.count - 1 - $0) }
            // The ninth concentrated winner is discarded by the existing per-lane
            // retention contract. Select the same pool, without claiming global exactness.
            let offsets: [UInt32] = [0, 0, 2051]
            for width in [32, 64, 96, 128, 256] {
                func precedes(_ a: Int, _ b: Int) -> Bool {
                    values[a] == values[b] ? ids[a] < ids[b] : values[a] < values[b]
                }
                let retained = (0..<width).flatMap { lane in
                    Array(stride(from: lane, to: values.count, by: width)).sorted(by: precedes).prefix(8)
                }.sorted(by: precedes)
                for k in [1, 8, 31, 32, 33, 128, 513, 2051] {
                    let result = try run(device, pipeline, values: values, dimension: 1,
                        offsets: offsets, ids: ids, queries: [0], selected: [[1, .max, 0, 9]], width: width, k: k)
                    let expected = Array(retained.prefix(k))
                    let padding = max(0, k - expected.count)
                    XCTAssertEqual(result.0, expected.map { ids[$0] } + [UInt32](repeating: .max, count: padding), "\(path) width=\(width) k=\(k)")
                    XCTAssertEqual(result.1, expected.map { values[$0] * values[$0] } + [Float](repeating: .infinity, count: padding), path)
                }
            }
        }
    }

    func testNonfiniteOrderingAndInvalidListsPreservePadding() throws {
        try withPipelines { device, pipeline, path in
            let values: [Float] = [.nan, 2, 1, .infinity, 1, .nan]
            let ids: [UInt32] = [5, 4, 3, 2, 1, 0]
            for width in [32, 96, 256] {
                for k in [8, 32, 33] {
                    let result = try run(device, pipeline, values: values, dimension: 1,
                        offsets: [0, 3, 3, 6], ids: ids, queries: [0, 0],
                        selected: [[2, .max, 1, 0, 17], [.max, 3, 1, 8, 1]], width: width, k: k)
                    XCTAssertEqual(result.0, [1, 3, 4, 2, 0, 5] + [UInt32](repeating: .max, count: 2 * k - 6), path)
                    guard result.1.count == 2 * k else { continue }
                    XCTAssertEqual(Array(result.1.prefix(4)), [1, 1, 4, .infinity], path)
                    XCTAssertTrue(result.1[4].isNaN && result.1[5].isNaN, path)
                    XCTAssertEqual(Array(result.1.dropFirst(6)), [Float](repeating: .infinity, count: 2 * k - 6), path)
                }
            }
        }
    }

    func testWorkspaceReuseAtQueryCacheBoundary() throws {
        try withPipelines { device, pipeline, path in
            for dimension in [1, 767, 768, 2047, 2048, 2049] {
                for n in [0, 65] {
                    // Small binary fractions keep CPU and GPU sums exactly representable.
                    let values = (0..<n).map { Float($0 % 17) * 0.25 }
                    let ids = (0..<n).map { UInt32(n - 1 - $0) }
                    let queries: [Float] = [0, 0.25, 0.5]
                    for k in [8, 32, 33, 128] {
                        var expectedIDs: [UInt32] = [], expectedDistances: [Float] = []
                        for query in queries {
                            let ordered = (0..<n).sorted { a, b in
                                let da = abs(values[a] - query), db = abs(values[b] - query)
                                return da == db ? ids[a] < ids[b] : da < db
                            }
                            for row in ordered.prefix(k) {
                                let delta = values[row] - query
                                expectedIDs.append(ids[row]); expectedDistances.append(Float(dimension) * delta * delta)
                            }
                            expectedIDs += [UInt32](repeating: .max, count: max(0, k - n))
                            expectedDistances += [Float](repeating: .infinity, count: max(0, k - n))
                        }
                        // Unequal lane workloads expose query-cache reuse before scanning finishes.
                        for _ in 0..<(dimension == 2048 ? 8 : 1) {
                            let result = try run(device, pipeline, values: values, dimension: dimension,
                                offsets: [0, UInt32(n / 7), UInt32(n / 7), UInt32(n)], ids: ids,
                                queries: queries, selected: [[2, 1, 0], [0, 2, 1], [1, 0, 2]], k: k)
                            XCTAssertEqual(result.0, expectedIDs, "\(path) D=\(dimension) n=\(n) k=\(k)")
                            XCTAssertEqual(result.1, expectedDistances, path)
                        }
                    }
                }
            }
        }
    }

    func testTrainedFilteredSearchAcrossSelectionAndCoarseSplits() async throws {
        // Keep an over-budget regression from aborting the process through the public API.
        var canDispatch = true
        try withPipelines { device, pipeline, path in
            if pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength {
                XCTFail("\(path): IVF list pipeline exceeds the device memory budget")
                canDispatch = false
            }
        }
        guard canDispatch else { return }
        // Sixteen distinct centers make training independent of seed numbering.
        // Eight rows per center preserve all nearest 33 rows in the retained pool.
        let values = (0..<128).map { Float($0 / 8) }
        let vectors = values.map { [Float](repeating: $0, count: 8) }
        for nprobe in [4, 16] {
            let index = try await AcceleratedVectorIndex(configuration: .ivf(
                dimension: 8, nlist: 16, nprobe: nprobe, capacity: 128, routingThreshold: 0))
            await index.setAutoTraining(false)
            let handles = try await index.insert(vectors)
            try await index.train()
            let trained = await index.isTrained
            XCTAssertTrue(trained)
            let query = [Float](repeating: -1, count: 8)
            for k in [32, 33] {
                let result = try await index.search(query: query, k: k)
                XCTAssertFalse(result.isExhaustive) // Ensure the trained IVF route was used.
                XCTAssertEqual(result.map { $0.id }, Array(handles.prefix(min(k, nprobe * 8))))
                XCTAssertEqual(result.map { $0.distance }, values.prefix(min(k, nprobe * 8)).map { 8 * ($0 + 1) * ($0 + 1) })
            }
            let allowed = Set(stride(from: 0, to: handles.count, by: 2).map { handles[$0] })
            // Requested 11 becomes K=33 in list search; filtering rejects actual results.
            let filtered = try await index.search(query: query, k: 11, filter: { handle, _ in allowed.contains(handle) })
            XCTAssertFalse(filtered.isExhaustive)
            XCTAssertEqual(filtered.map { $0.id }, stride(from: 0, to: 22, by: 2).map { handles[$0] })
            let expectedDistances: [Float] = stride(from: 0, to: 22, by: 2).map { row in
                let delta = values[row] + 1
                return 8 * delta * delta
            }
            XCTAssertEqual(filtered.map { $0.distance }, expectedDistances)
        }
    }
}
