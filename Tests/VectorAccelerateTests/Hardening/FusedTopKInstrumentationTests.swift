import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class FusedTopKInstrumentationTests: XCTestCase {
    private func withPipelines(_ body: (any MTLDevice, any MTLComputePipelineState, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        func check(_ library: any MTLLibrary, _ label: String) throws {
            let function = try XCTUnwrap(library.makeFunction(name: "fused_l2_topk"))
            try body(device, device.makeComputePipelineState(function: function), label)
        }
        #if DEBUG
        try check(device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try check(KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    // With MTL_SHADER_VALIDATION=1 the old shared candidate array exceeds the
    // device limit. Check before encoding so regression produces an assertion, not SIGABRT.
    func testFusedPipelineFitsThreadgroupMemoryBudget() throws {
        try withPipelines { device, pipeline, path in
            print("Fused Top-K \(path): static=\(pipeline.staticThreadgroupMemoryLength), limit=\(device.maxThreadgroupMemoryLength)")
            XCTAssertLessThanOrEqual(pipeline.staticThreadgroupMemoryLength, device.maxThreadgroupMemoryLength, path)
        }
    }

    private func buffer<T: BitwiseCopyable>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        if values.isEmpty { return try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared)) }
        return try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
    }

    private func run(_ device: any MTLDevice, _ pipeline: any MTLComputePipelineState,
                     rows: [[Float]], dimension: Int, width: Int, k: Int) throws -> ([UInt32], [Float]) {
        guard pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
            XCTFail("Instrumented pipeline exceeds the device memory budget")
            return ([], [])
        }
        let query = try buffer([Float](repeating: 0, count: dimension), device)
        let data = try buffer(rows.flatMap { $0 }, device)
        let ids = try buffer([UInt32](repeating: 0xDEADBEEF, count: k + 2), device)
        let scores = try buffer([Float](repeating: 12345, count: k + 2), device)
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in [query, data, ids, scores].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        for (i, value) in [1, rows.count, dimension, k].enumerated() {
            var value = UInt32(value)
            encoder.setBytes(&value, length: 4, index: i + 4)
        }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
        let indices = Array(UnsafeBufferPointer(start: ids.contents().assumingMemoryBound(to: UInt32.self), count: k + 2))
        let distances = Array(UnsafeBufferPointer(start: scores.contents().assumingMemoryBound(to: Float.self), count: k + 2))
        XCTAssertEqual(Array(indices.suffix(2)), [0xDEADBEEF, 0xDEADBEEF])
        XCTAssertEqual(Array(distances.suffix(2)), [12345, 12345])
        return (Array(indices.prefix(k)), Array(distances.prefix(k)))
    }

    func testHeapHeadsPreserveRetainedCandidatesAndConcentratedWinners() throws {
        try withPipelines { device, pipeline, path in
            var values = (0..<2051).map { Float(1000 + $0) }
            for i in 0..<8 { values[i * 256] = Float(i) }
            for i in [31, 63, 127, 255] { values[i] = 2 }
            // Eight best values in one lane exercise repeated ownership and exhaustion.
            // For raw K > 8, preserve selection over eight retained candidates per lane,
            // rather than claiming an exact global top-K beyond the public fused limit.
            for width in [32, 64, 96, 128, 256] {
                func precedes(_ a: Int, _ b: Int) -> Bool {
                    values[a] == values[b] ? a < b : values[a] < values[b]
                }
                var retained: [Int] = []
                for lane in 0..<width {
                    let ordered = Array(stride(from: lane, to: values.count, by: width)).sorted(by: precedes)
                    retained.append(contentsOf: ordered.prefix(8))
                }
                retained.sort(by: precedes)
                for k in [1, 4, 8, 33, 129] {
                    let result = try run(device, pipeline, rows: values.map { [$0] }, dimension: 1, width: width, k: k)
                    XCTAssertEqual(result.0, retained.prefix(k).map(UInt32.init), "\(path) width=\(width) k=\(k)")
                    XCTAssertEqual(result.1, retained.prefix(k).map { values[$0] * values[$0] }, path)
                }
            }
            for dimension in [767, 768] {
                let rows = (0..<17).map { i in [Float](repeating: Float(i) / 16, count: dimension) }
                let result = try run(device, pipeline, rows: rows, dimension: dimension, width: 256, k: 8)
                XCTAssertEqual(result.0, Array(0..<UInt32(8)), path)
                XCTAssertEqual(result.1, (0..<8).map { Float(dimension * $0 * $0) / 256 }, path)
            }
        }
    }

    func testExhaustedHeapsPreserveNonfiniteOrderingAndPadding() throws {
        try withPipelines { device, pipeline, path in
            for width in [32, 96, 256] {
                for k in [8, 33] {
                    let result = try run(device, pipeline, rows: [[.nan], [2], [1], [.infinity], [1], [.nan]], dimension: 1, width: width, k: k)
                    XCTAssertEqual(result.0, [2, 4, 1, 3, 0, 5] + [UInt32](repeating: .max, count: k - 6), path)
                    XCTAssertEqual(Array(result.1.prefix(4)), [1, 1, 4, .infinity], path)
                    guard result.1.count == k else { continue }
                    XCTAssertTrue(result.1[4].isNaN && result.1[5].isNaN, path)
                    XCTAssertEqual(Array(result.1.dropFirst(6)), [Float](repeating: .infinity, count: k - 6), path)
                }
                let empty = try run(device, pipeline, rows: [], dimension: 1, width: width, k: 8)
                XCTAssertEqual(empty.0, [UInt32](repeating: .max, count: 8), path)
                XCTAssertEqual(empty.1, [Float](repeating: .infinity, count: 8), path)
            }
        }
    }
}
