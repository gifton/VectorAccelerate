import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFCandidateBoundsTests: XCTestCase {
    private func buffer(_ values: [UInt32], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
    }

    private func words(_ buffer: any MTLBuffer, _ count: Int) -> [UInt32] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
    }

    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func run(_ device: any MTLDevice, _ library: any MTLLibrary, _ buffers: [any MTLBuffer],
                     name: String = "ivf_build_candidates_fused", groups: Int = 2, width: Int = 32) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
    }

    // Baseline writes remain within physical guard storage; logical capacity is smaller.
    // Removing bounded reservations corrupts the guard and publishes unusable ranges.
    func testFusedReservationsRespectCapacityAndPublishOnlyCompleteQueries() throws {
        try withLibraries { device, library, path in
            for capacity in [0, 1, 3, 7, 8, 9] {
                let indices = try buffer([UInt32](repeating: 12345, count: 32), device)
                let ids = try buffer([UInt32](repeating: 12345, count: 32), device)
                let count = try buffer([0], device)
                let offsets = try buffer([UInt32](repeating: 12345, count: 6), device)
                let counts = try buffer([UInt32](repeating: 12345, count: 6), device)
                // Four queries, each emits entries [0, 1]. Duplicate and invalid probes ignored.
                try run(device, library, [buffer([0, 0, .max, 0, 0, .max, 0, 0, .max, 0, 0, .max], device),
                    buffer([0, 2], device), indices, ids, count, offsets, counts,
                    buffer([4, 3, 1, UInt32(capacity)], device)])
                XCTAssertEqual(words(count, 1)[0], UInt32(capacity < 8 ? capacity + 1 : 8), path)
                let starts = words(offsets, 6), sizes = words(counts, 6)
                var occupied = Set<Int>()
                for q in 0..<4 {
                    if starts[q] == .max { XCTAssertEqual(sizes[q], 0, path); continue }
                    XCTAssertEqual(sizes[q], 2, path)
                    let start = Int(starts[q])
                    XCTAssertLessThanOrEqual(start + 2, capacity, path)
                    guard start + 2 <= capacity else { continue }
                    for j in 0..<2 {
                        XCTAssertTrue(occupied.insert(start + j).inserted, path)
                        XCTAssertEqual(words(indices, 32)[start + j], UInt32(j), path)
                        XCTAssertEqual(words(ids, 32)[start + j], UInt32(q), path)
                    }
                }
                for i in 0..<32 where !occupied.contains(i) {
                    XCTAssertEqual(words(indices, 32)[i], 12345, "\(path), cap=\(capacity), slot=\(i)")
                    XCTAssertEqual(words(ids, 32)[i], 12345, path)
                }
                XCTAssertEqual(Array(starts[4...]), [12345, 12345]); XCTAssertEqual(Array(sizes[4...]), [12345, 12345])
            }
        }
    }

    // Unequal per-query work makes atomic allocation order differ from query order.
    // Correct CSR must still contain exactly each query's own list entries.
    func testPublicFusedResultHasCorrectCSRForUnevenQueries() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        var probes = [UInt32]()
        for q in 0..<256 { probes += q % 2 == 0 ? Array(0..<UInt32(64)) : [UInt32](repeating: 63, count: 64) }
        let result = try await kernel.buildCandidates(nearestCentroids: buffer(probes, device),
            listOffsets: buffer(Array(0...UInt32(64)), device), numQueries: 256, nprobe: 64, numLists: 64,
            maxCandidatesPerQuery: 64)
        XCTAssertEqual(result.totalCandidates, 8320)
        let offsets = words(result.candidateOffsets, 257)
        let entries = words(result.candidateIVFIndices, result.totalCandidates)
        let ids = words(result.candidateQueryIds, result.totalCandidates)
        var start = 0
        for q in 0..<256 {
            let expected: [UInt32] = q % 2 == 0 ? Array(0..<UInt32(64)) : [63]
            XCTAssertEqual(offsets[q], UInt32(start), "query \(q)")
            XCTAssertEqual(Array(entries[start..<(start + expected.count)]), expected, "query \(q)")
            XCTAssertEqual(Array(ids[start..<(start + expected.count)]), [UInt32](repeating: UInt32(q), count: expected.count))
            start += expected.count
        }
        XCTAssertEqual(offsets[256], 8320)
    }

    func testFusedCounterCannotWrapAndEmptyQueriesWriteNoCandidates() throws {
        try withLibraries { device, library, path in
            for capacity: UInt32 in [0, 7, .max] {
                let indices = try buffer([12345], device), ids = try buffer([12345], device)
                let count = try buffer([.max], device)
                let offsets = try buffer([UInt32](repeating: 12345, count: 257), device)
                let counts = try buffer([UInt32](repeating: 12345, count: 257), device)
                try run(device, library, [buffer([UInt32](repeating: 0, count: 257), device), buffer([0, 2], device),
                    indices, ids, count, offsets, counts, buffer([257, 1, 1, capacity], device)], groups: 9)
                XCTAssertEqual(words(count, 1), [.max], path)
                XCTAssertEqual(words(indices, 1), [12345]); XCTAssertEqual(words(ids, 1), [12345])
                XCTAssertEqual(words(offsets, 257), [UInt32](repeating: .max, count: 257))
                XCTAssertEqual(words(counts, 257), [UInt32](repeating: 0, count: 257))
            }
            let indices = try buffer([12345], device), ids = try buffer([12345], device)
            let count = try buffer([0], device), offsets = try buffer([12345], device), counts = try buffer([12345], device)
            try run(device, library, [buffer([0], device), buffer([0, 0], device), indices, ids,
                count, offsets, counts, buffer([1, 1, 1, 0], device)])
            XCTAssertEqual(words(count, 1), [0]); XCTAssertEqual(words(offsets, 1), [0]); XCTAssertEqual(words(counts, 1), [0])
            XCTAssertEqual(words(indices, 1), [12345]); XCTAssertEqual(words(ids, 1), [12345])
        }
    }

    func testPrefixSumSignalsUnrepresentableTotalWithoutWrapping() throws {
        try withLibraries { device, library, path in
            let offsets = try buffer([12345, 12345, 12345, 12345, 12345], device)
            try run(device, library, [buffer([UInt32.max - 1, 3, 1], device), offsets, buffer([3, 0, 0, 0], device)],
                    name: "ivf_prefix_sum_sequential")
            XCTAssertEqual(words(offsets, 5), [0, .max - 1, .max, .max, 12345], path)
        }
    }

    func testPublicHintsNeverTruncateSkewedLists() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        // 15 empty lists and one 129-entry list: average-based estimate is too small.
        let lists: [UInt32] = [0] + [UInt32](repeating: 129, count: 16)
        for q in [1, 256, 257] {
            for hint: Int? in [nil, 0, 1, 129, Int.max] {
                let result = try await kernel.buildCandidates(nearestCentroids: buffer([UInt32](repeating: 0, count: q), device),
                    listOffsets: buffer(lists, device), numQueries: q, nprobe: 1, numLists: 16, maxCandidatesPerQuery: hint)
                XCTAssertEqual(result.totalCandidates, q * 129)
                XCTAssertEqual(words(result.candidateOffsets, q + 1), (0...q).map { UInt32($0 * 129) })
                let entries = words(result.candidateIVFIndices, q * 129), ids = words(result.candidateQueryIds, q * 129)
                for query in 0..<q {
                    XCTAssertEqual(Array(entries[(query * 129)..<((query + 1) * 129)]), Array(0..<UInt32(129)))
                    XCTAssertEqual(Array(ids[(query * 129)..<((query + 1) * 129)]), [UInt32](repeating: UInt32(query), count: 129))
                }
            }
        }
    }

    func testCSRConversionReordersExplicitSegmentsAndRejectsOverlap() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        let indices = try await context.getBuffer(size: 20), ids = try await context.getBuffer(size: 20)
        // Atomic allocation order q3, q2, q0; q1 is empty. Force the blit path.
        for (i, v): (Int, UInt32) in [40, 41, 30, 10, 11].enumerated() { indices.buffer.contents().storeBytes(of: v, toByteOffset: i * 4, as: UInt32.self) }
        for (i, v): (Int, UInt32) in [3, 3, 2, 0, 0].enumerated() { ids.buffer.contents().storeBytes(of: v, toByteOffset: i * 4, as: UInt32.self) }
        let result = try await kernel.makeFusedCSR(indices: indices, queryIds: ids,
            offsets: buffer([3, 0, 2, 0], device), counts: buffer([2, 0, 1, 2], device), total: 5, numQueries: 4)
        XCTAssertEqual(words(result.candidateOffsets, 5), [0, 2, 2, 3, 5])
        XCTAssertEqual(words(result.candidateIVFIndices, 5), [10, 11, 30, 40, 41])
        XCTAssertEqual(words(result.candidateQueryIds, 5), [0, 0, 2, 3, 3])
        XCTAssertEqual(result.offsetRange(for: 1), 2..<2)
        for (starts, sizes): ([UInt32], [UInt32]) in [([0, 0], [2, 3]), ([0, 4], [2, 3]), ([0, 2], [2, 2]), ([.max, 0], [0, 5])] {
            do {
                _ = try await kernel.makeFusedCSR(indices: indices, queryIds: ids, offsets: buffer(starts, device),
                    counts: buffer(sizes, device), total: 5, numQueries: 2)
                XCTFail("Invalid fused segments must be rejected")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .operationFailed) }
        }
    }

    func testPublicEmptyInputsAndInvalidCounts() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        for q in [0, 1, 257] {
            let result = try await kernel.buildCandidates(nearestCentroids: buffer([0], device),
                listOffsets: buffer([0], device), numQueries: q, nprobe: 0, numLists: 0)
            XCTAssertEqual(result.totalCandidates, 0)
            XCTAssertEqual(words(result.candidateOffsets, q + 1), [UInt32](repeating: 0, count: q + 1))
        }
        for (q, probe, lists, hint) in [(-1, 1, 0, 0), (1, -1, 0, 0), (1, 1, -1, 0), (1, 1, 0, -1), (Int.max, 1, 0, 0), (1, 1, Int.max, 0), (2, 1, 0, 0)] {
            do {
                _ = try await kernel.buildCandidates(nearestCentroids: buffer([0], device), listOffsets: buffer([0], device),
                    numQueries: q, nprobe: probe, numLists: lists, maxCandidatesPerQuery: hint)
                XCTFail("Invalid count/storage must throw before encoding")
            } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
        }
        // Q > 256 selects exact counting; a valid CSR can still exceed uint total.
        do {
            _ = try await kernel.buildCandidates(nearestCentroids: buffer([UInt32](repeating: 0, count: 257), device),
                listOffsets: buffer([0, UInt32.max / 2], device), numQueries: 257, nprobe: 1, numLists: 1)
            XCTFail("Unrepresentable total must throw before candidate allocation/build")
        } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
    }


    // The pool currently caps buckets at 64 MiB. Valid counts above that storage
    // must fail before building or receive sufficient storage if the pool changes.
    func testLargeOutputsCannotPublishCountsBeyondPhysicalStorage() async throws {
        let context = try await Metal4Context()
        let device = context.device.rawDevice
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        for (q, entries) in [(1, UInt32(16_777_217)), (257, UInt32(65_537))] {
            do {
                let result = try await kernel.buildCandidates(nearestCentroids: buffer([UInt32](repeating: 0, count: q), device),
                    listOffsets: buffer([0, entries], device), numQueries: q, nprobe: 1, numLists: 1)
                XCTAssertEqual(result.totalCandidates, q * Int(entries))
                XCTAssertGreaterThanOrEqual(result.candidateIVFIndices.length / 4, result.totalCandidates)
                XCTAssertGreaterThanOrEqual(result.candidateQueryIds.length / 4, result.totalCandidates)
                XCTAssertEqual(words(result.candidateOffsets, q + 1), (0...q).map { UInt32($0) * entries })
            } catch let error as VectorError { XCTAssertEqual(error.kind, .allocationFailed) }
        }
    }

}
