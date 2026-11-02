import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class BoruvkaBoundsTests: XCTestCase {
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
                     buffers: [any MTLBuffer], groups: Int, width: Int = 1) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
    }

    func testEveryFindKernelSeparatesInfiniteEdgesFromNoEdge() throws {
        try withLibraries { device, library, path in
            for d in [5, 384, 512, 768, 1536] {
                let name = d == 5 ? "boruvka_find_min_edge_kernel" : "boruvka_find_min_edge_\(d)_kernel"
                var embeddings = [Float](repeating: 0, count: 3 * d)
                embeddings[d] = 1; embeddings[2 * d] = 3
                for fixture in 0..<4 {
                    let weights = try buffer([Float](repeating: -12345, count: 5), device)
                    let targets = try buffer([UInt32](repeating: 12345, count: 5), device)
                    let components: [UInt32] = fixture == 2 ? [0, 0, 0] : [0, 1, 2]
                    let cores: [Float] = fixture == 0 ? [0, 0, 0] : fixture == 3 ? [.infinity, 0, 0] : [.infinity, .infinity, .infinity]
                    try run(name, device: device, library: library,
                            buffers: [buffer(embeddings, device), buffer(cores, device), buffer(components, device),
                                      weights, targets, buffer([UInt32(3), UInt32(d), 0, 6], device)], groups: 2, width: 3)
                    let w = weights.contents().bindMemory(to: Float.self, capacity: 5)
                    let t = targets.contents().bindMemory(to: UInt32.self, capacity: 5)
                    let expectedTargets: [UInt32] = fixture == 0 ? [1, 0, 1] : fixture == 1 ? [1, 0, 0] : fixture == 3 ? [1, 2, 1] : [.max, .max, .max]
                    let expectedWeights: [Float] = fixture == 0 ? [1, 1, 2] : fixture == 3 ? [.infinity, 2, 2] : [.infinity, .infinity, .infinity]
                    for i in 0..<3 {
                        XCTAssertEqual(t[i], expectedTargets[i], "\(path), d=\(d), fixture=\(fixture), point=\(i)")
                        XCTAssertEqual(w[i], expectedWeights[i])
                    }
                    for i in 3..<5 { XCTAssertEqual(t[i], 12345); XCTAssertEqual(w[i], -12345) }
                }
            }
        }
    }

    func testReductionPublishesEndpointValidityAndClearsNonrepresentatives() throws {
        try withLibraries { device, library, path in
            for finite in [false, true] {
                let weights = try buffer([Float](repeating: -12345, count: 6), device)
                let sources = try buffer([UInt32](repeating: 12345, count: 6), device)
                let targets = try buffer([UInt32](repeating: 12345, count: 6), device)
                let pointWeights: [Float] = finite ? [.infinity, 2, .nan, .infinity] : [.infinity, .infinity, .nan, .infinity]
                try run("boruvka_component_reduce_kernel", device: device, library: library,
                        buffers: [buffer([UInt32(0), 0, 2, 2], device), buffer(pointWeights, device),
                                  buffer([UInt32(2), 2, 0, 1], device), weights, sources, targets,
                                  buffer([UInt32(4), 1, 0, 8], device)], groups: 2, width: 3)
                let w = weights.contents().bindMemory(to: Float.self, capacity: 6)
                let s = sources.contents().bindMemory(to: UInt32.self, capacity: 6)
                let t = targets.contents().bindMemory(to: UInt32.self, capacity: 6)
                XCTAssertEqual(s[0], finite ? 1 : 0, path); XCTAssertEqual(t[0], 2, path)
                XCTAssertEqual(w[0], finite ? 2 : .infinity, path)
                XCTAssertEqual(s[2], 3, path); XCTAssertEqual(t[2], 1, path); XCTAssertEqual(w[2], .infinity, path)
                for i in [1, 3] { XCTAssertEqual(s[i], .max, path); XCTAssertEqual(t[i], .max, path); XCTAssertEqual(w[i], .infinity, path) }
                for i in 4..<6 { XCTAssertEqual(s[i], 12345); XCTAssertEqual(t[i], 12345); XCTAssertEqual(w[i], -12345) }
            }
        }
    }

    /// Physical guard capacity keeps baseline writes inside the allocation while
    /// proving that the shader respects the logical candidate capacity.
    func testCollectorBoundsReservationsAndSignalsOverflow() throws {
        try withLibraries { device, library, path in
            let n = 8
            for (capacity, initial) in [(0, 0), (1, 0), (7, 0), (8, 0), (7, 6), (1, 2)] {
                let edges = try buffer([MSTEdgeGPU](repeating: MSTEdgeGPU(source: 12345, target: 12345, weight: -12345), count: 32), device)
                let count = try buffer([UInt32(initial)], device)
                try run("boruvka_merge_kernel", device: device, library: library,
                        buffers: [buffer((0..<n).map(UInt32.init), device), buffer([Float](repeating: 1, count: n), device),
                                  buffer((0..<n).map(UInt32.init), device), buffer((0..<n).map { UInt32(($0 + 1) % n) }, device),
                                  edges, count, buffer([UInt32(n), 1, 0, UInt32(capacity)], device)], groups: 3, width: 3)
                XCTAssertEqual(count.contents().load(as: UInt32.self), UInt32(min(initial + n, capacity + 1)), "\(path), capacity=\(capacity), initial=\(initial)")
                let result = edges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: 32)
                var seen = Set<UInt32>()
                for i in 0..<32 {
                    if i >= initial && i < capacity {
                        XCTAssertLessThan(result[i].source, UInt32(n), path)
                        XCTAssertEqual(result[i].target, (result[i].source + 1) % UInt32(n), path)
                        XCTAssertEqual(result[i].weight, 1, path)
                        XCTAssertTrue(seen.insert(result[i].source).inserted, path)
                    } else {
                        XCTAssertEqual(result[i].source, 12345, "\(path), capacity=\(capacity), slot=\(i)")
                        XCTAssertEqual(result[i].target, 12345, path); XCTAssertEqual(result[i].weight, -12345, path)
                    }
                }
            }
        }
    }

    func testCollectorRetainsInfiniteWeightsAndRejectsNoEdgeEndpoints() throws {
        try withLibraries { device, library, path in
            let edges = try buffer([MSTEdgeGPU](repeating: MSTEdgeGPU(source: 12345, target: 12345, weight: -12345), count: 8), device)
            let count = try buffer([UInt32(0)], device)
            try run("boruvka_merge_kernel", device: device, library: library,
                    buffers: [buffer([UInt32(0), 0, 2, 3], device), buffer([Float.infinity, .infinity, 1, .infinity], device),
                              buffer([UInt32(0), .max, 2, 3], device), buffer([UInt32(2), .max, 3, 0], device),
                              edges, count, buffer([UInt32(4), 1, 0, 8], device)], groups: 2, width: 3)
            let actualCount = count.contents().load(as: UInt32.self)
            XCTAssertEqual(actualCount, 3, path)
            let result = edges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: 8)
            let live = (0..<min(Int(actualCount), 8)).map { result[$0] }
            XCTAssertEqual(Set(live.map(\.source)), Set([UInt32(0), 2, 3]), path)
            XCTAssertEqual(live.filter { $0.weight == .infinity }.count, 2, path)
        }
    }

    func testPublicMSTConnectsInfiniteWeightEdges() async throws {
        let context = try await Metal4Context()
        let kernel = try await BoruvkaMSTKernel(context: context)
        for cores: [Float] in [[.infinity, .infinity, .infinity, .infinity], [0, 0, .infinity, .infinity]] {
            let result = try await kernel.computeMST(embeddings: [[0], [1], [10], [11]], coreDistances: cores)
            XCTAssertEqual(result.edges.count, 3)
            XCTAssertEqual(result.totalWeight, .infinity)
            XCTAssertTrue(Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: 4))
        }
        // Derived roots retain FP32 squared-accumulator limits. An overflowed
        // weight is still an edge, rather than silently disconnecting the graph.
        let huge = try await kernel.computeMST(embeddings: [[-Float.greatestFiniteMagnitude], [0], [Float.greatestFiniteMagnitude]],
                                              coreDistances: [0, 0, 0])
        XCTAssertEqual(huge.edges.count, 2); XCTAssertEqual(huge.totalWeight, .infinity)
        XCTAssertTrue(Metal4KernelTestHelpers.verifyConnected(edges: huge.edges, n: 3))
    }

    func testSaturatedCounterCannotWrapIntoCandidateStorage() throws {
        try withLibraries { device, library, path in
            let n = 257
            for capacity: UInt32 in [0, 7, .max] {
                let edges = try buffer([MSTEdgeGPU(source: 12345, target: 12345, weight: -12345)], device)
                let count = try buffer([UInt32.max], device)
                try run("boruvka_merge_kernel", device: device, library: library,
                        buffers: [buffer((0..<n).map(UInt32.init), device), buffer([Float](repeating: 1, count: n), device),
                                  buffer((0..<n).map(UInt32.init), device), buffer((0..<n).map { UInt32(($0 + 1) % n) }, device),
                                  edges, count, buffer([UInt32(n), 1, 0, capacity], device)], groups: 9, width: 32)
                XCTAssertEqual(count.contents().load(as: UInt32.self), .max, path)
                let edge = edges.contents().load(as: MSTEdgeGPU.self)
                XCTAssertEqual(edge.source, 12345, path); XCTAssertEqual(edge.target, 12345, path)
                XCTAssertEqual(edge.weight, -12345, path)
            }
        }
    }

    func testInvalidEndpointsAreRejectedBeforeComponentLookup() throws {
        try withLibraries { device, library, path in
            let components = try buffer([UInt32(0), 1, 2], device)
            let weights = try buffer([Float](repeating: -12345, count: 3), device)
            let sources = try buffer([UInt32](repeating: 12345, count: 3), device)
            let targets = try buffer([UInt32](repeating: 12345, count: 3), device)
            let params = try buffer([UInt32(3), 1, 0, 6], device)
            try run("boruvka_component_reduce_kernel", device: device, library: library,
                    buffers: [components, buffer([Float(1), 1, 1], device), buffer([UInt32.max, 3, 2], device),
                              weights, sources, targets, params], groups: 2, width: 3)
            let s = sources.contents().bindMemory(to: UInt32.self, capacity: 3)
            let t = targets.contents().bindMemory(to: UInt32.self, capacity: 3)
            for i in 0..<3 { XCTAssertEqual(s[i], .max, path); XCTAssertEqual(t[i], .max, path) }
            let edges = try buffer([MSTEdgeGPU(source: 12345, target: 12345, weight: -12345)], device)
            let count = try buffer([UInt32(0)], device)
            try run("boruvka_merge_kernel", device: device, library: library,
                    buffers: [components, buffer([Float(1), 1, 1], device), buffer([UInt32.max, 0, 2], device),
                              buffer([UInt32(0), 3, 2], device), edges, count, params], groups: 2, width: 3)
            XCTAssertEqual(count.contents().load(as: UInt32.self), 0, path)
            XCTAssertEqual(edges.contents().load(as: MSTEdgeGPU.self).source, 12345, path)
        }
    }

    private func initialize(_ work: BoruvkaWorkBuffers, n: Int) {
        work.edgeCount.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let components = work.componentIds.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n { components[i] = UInt32(i) }
    }

    func testRepeatedUnmergedFusionRoundsReportOverflow() async throws {
        let context = try await Metal4Context()
        let kernel = try await BoruvkaMSTKernel(context: context)
        let device = context.device.rawDevice
        for n in [3, 257] {
            let work = try kernel.createWorkBuffers(n: n)
            initialize(work, n: n)
            let embeddings = try buffer([Float](repeating: 0, count: n), device)
            let cores = try buffer([Float](repeating: .infinity, count: n), device)
            for iteration in 0..<5 {
                try await context.executeAndWait { _, encoder in
                    kernel.encodeIteration(into: encoder, embeddings: embeddings, coreDistances: cores,
                                           workBuffers: work, n: n, d: 1, iteration: iteration)
                }
                if iteration < 2 {
                    XCTAssertEqual(try work.readCandidateCount(), n * (iteration + 1))
                } else {
                    XCTAssertThrowsError(try work.readCandidateCount()) { error in
                        XCTAssertEqual((error as? VectorError)?.kind, .operationFailed)
                    }
                    XCTAssertEqual(work.edgeCount.contents().load(as: UInt32.self), UInt32(work.candidateCapacity + 1))
                }
            }
            let edges = work.candidateEdges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: work.candidateCapacity)
            for i in 0..<work.candidateCapacity {
                XCTAssertLessThan(edges[i].source, UInt32(n)); XCTAssertLessThan(edges[i].target, UInt32(n))
                XCTAssertNotEqual(edges[i].source, edges[i].target); XCTAssertEqual(edges[i].weight, .infinity)
            }
        }
    }

    func testMergedRoundsRespectGeometricCandidateBound() async throws {
        let context = try await Metal4Context()
        let kernel = try await BoruvkaMSTKernel(context: context)
        let device = context.device.rawDevice
        let n = 32
        // Nested well-separated pairs force multiple rounds. A 1-D MST has
        // total weight max(position) - min(position), independently of edge ties.
        let positions: [Float] = (0..<n).map { value in
            var x = value, place: Float = 1, position: Float = 0
            while x > 0 { position += Float(x % 2) * place; place *= 10; x /= 2 }
            return position
        }
        let embeddings = try buffer(positions, device)
        let cores = try buffer([Float](repeating: 0, count: n), device)
        let work = try kernel.createWorkBuffers(n: n)
        initialize(work, n: n)
        var roots = Array(0..<n), componentCount = n, previousCount = 0, rounds = 0, mstWeight: Float = 0, accepted = 0
        func root(_ v: Int) -> Int {
            var result = v
            while roots[result] != result { result = roots[result] }
            return result
        }
        while componentCount > 1 && rounds < 10 {
            let iteration = rounds
            try await context.executeAndWait { _, encoder in
                kernel.encodeIteration(into: encoder, embeddings: embeddings, coreDistances: cores,
                                       workBuffers: work, n: n, d: 1, iteration: iteration)
            }
            let count = try work.readCandidateCount()
            XCTAssertEqual(count - previousCount, componentCount, "One candidate per active component")
            XCTAssertLessThan(count, 2 * n)
            let edges = work.candidateEdges.contents().bindMemory(to: MSTEdgeGPU.self, capacity: count)
            for i in previousCount..<count {
                let a = Int(edges[i].source), b = Int(edges[i].target)
                guard a < n && b < n else { XCTFail("Invalid candidate endpoints"); return }
                let ra = root(a), rb = root(b)
                if ra != rb { roots[rb] = ra; mstWeight += edges[i].weight; accepted += 1 }
            }
            let ids = (0..<n).map { root($0) }
            let nextCount = Set(ids).count
            XCTAssertLessThanOrEqual(nextCount, componentCount / 2)
            let gpuIDs = work.componentIds.contents().bindMemory(to: UInt32.self, capacity: n)
            for i in 0..<n { gpuIDs[i] = UInt32(ids[i]) }
            componentCount = nextCount; previousCount = count; rounds += 1
        }
        XCTAssertGreaterThan(rounds, 1); XCTAssertEqual(componentCount, 1); XCTAssertEqual(accepted, n - 1)
        XCTAssertEqual(mstWeight, positions.last!)
        let standalone = try await kernel.computeMST(embeddings: positions.map { [$0] }, coreDistances: [Float](repeating: 0, count: n))
        XCTAssertEqual(standalone.edges.count, n - 1); XCTAssertEqual(standalone.totalWeight, positions.last!)
    }

    func testCandidateAllocationRejectsUnrepresentableCounts() async throws {
        let context = try await Metal4Context()
        let kernel = try await BoruvkaMSTKernel(context: context)
        for n in [0, -1, Int(UInt32.max) / 2 + 1, Int.max] {
            XCTAssertThrowsError(try kernel.createWorkBuffers(n: n)) { error in
                XCTAssertEqual((error as? VectorError)?.kind, .invalidData)
            }
        }
        let dummy = try buffer([Float(0)], context.device.rawDevice)
        do {
            _ = try await kernel.computeMST(embeddings: dummy, coreDistances: dummy, n: Int.max, d: 1)
            XCTFail("Reject 2N overflow before allocation or GPU reads")
        } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidData) }
    }
}
