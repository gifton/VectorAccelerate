import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class IVFBenchmarkFixtureGPUTests: XCTestCase {
    private func withPipelines(_ body: (any MTLDevice, any MTLComputePipelineState, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        func check(_ library: any MTLLibrary, _ path: String) throws {
            let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "ivf_list_search")))
            guard pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
                XCTFail("\(path): compiled list-search memory exceeds device capacity")
                return
            }
            try body(device, pipeline, path)
        }
        #if DEBUG
        try check(device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try check(KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func check(_ fixture: IVFBenchmarkFixture, k: Int, width: Int,
                       device: any MTLDevice, pipeline: any MTLComputePipelineState, path: String) throws {
        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            if values.isEmpty { return try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared)) }
            return try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
        }
        let count = fixture.queries.count * k
        let buffers = try [buffer(fixture.queries.flatMap { $0 }), buffer(fixture.vectors.flatMap { $0 }),
            buffer(fixture.listOffsets), buffer(fixture.originalIDs), buffer(fixture.selectedLists.flatMap { $0 }),
            buffer([UInt32](repeating: 0xDEADBEEF, count: count + 2)), buffer([Float](repeating: 12345, count: count + 2))]
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        var params = IVFListSearchParameters(numQueries: fixture.queries.count, numCentroids: fixture.centroids.count,
            dimension: fixture.dimension, nprobe: fixture.selectedLists[0].count, k: k, maxCandidatesPerQuery: fixture.vectors.count)
        encoder.setBytes(&params, length: MemoryLayout<IVFListSearchParameters>.stride, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: fixture.queries.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
        let ids = Array(UnsafeBufferPointer(start: buffers[5].contents().assumingMemoryBound(to: UInt32.self), count: count + 2))
        let distances = Array(UnsafeBufferPointer(start: buffers[6].contents().assumingMemoryBound(to: Float.self), count: count + 2))
        XCTAssertEqual(Array(ids.suffix(2)), [0xDEADBEEF, 0xDEADBEEF])
        XCTAssertEqual(Array(distances.suffix(2)), [12345, 12345])
        for q in fixture.queries.indices {
            let expected = try IVFBenchmarkOracle.retained(fixture, query: q, k: k, width: width)
            let padding = k - expected.count
            XCTAssertEqual(Array(ids[(q * k)..<((q + 1) * k)]), expected.map(\.index) + [UInt32](repeating: .max, count: padding), "\(path) query=\(q) K=\(k)")
            // Binary-grid fixtures keep these sums exact; this does not introduce a
            // general-purpose numerical comparator or tolerance policy for the kit.
            XCTAssertEqual(Array(distances[(q * k)..<((q + 1) * k)]).map(Double.init), expected.map(\.distance) + [Double](repeating: .infinity, count: padding), path)
        }
    }

    func testFirstBenchmarkFixtureAcrossSelectionBoundary() throws {
        let fixture = try IVFBenchmarkFixtures.make(seed: 42, dimension: 128,
            listSizes: [Int](repeating: 256, count: 16), queryCount: 16, nprobe: 4)
        XCTAssertEqual(fixture.selectedCandidateCounts, [Int](repeating: 1024, count: 16))
        try withPipelines { device, pipeline, path in
            for k in [31, 32, 33, 128, 384] {
                try check(fixture, k: k, width: 256, device: device, pipeline: pipeline, path: path)
            }
        }
    }

    func testUnevenAndEmptyListFixturesAcrossWidthsAndCacheBoundary() throws {
        try withPipelines { device, pipeline, path in
            let skewed = try IVFBenchmarkFixtures.make(seed: 19, stream: 4, dimension: 8,
                listSizes: [260, 0, 257, 3], queryCount: 3, nprobe: 4)
            for width in [32, 96, 256] {
                for k in [32, 33, 128] {
                    try check(skewed, k: k, width: width, device: device, pipeline: pipeline, path: path)
                }
            }
            for dimension in [2047, 2048, 2049] {
                let fixture = try IVFBenchmarkFixtures.make(seed: 7, dimension: dimension,
                    listSizes: [9, 0, 56], queryCount: 3, nprobe: 3)
                try check(fixture, k: 128, width: 256, device: device, pipeline: pipeline, path: path)
            }
            let empty = try IVFBenchmarkFixtures.make(seed: 42, dimension: 8,
                listSizes: [0, 0], queryCount: 3, nprobe: 2)
            try check(empty, k: 33, width: 256, device: device, pipeline: pipeline, path: path)
        }
    }
}
