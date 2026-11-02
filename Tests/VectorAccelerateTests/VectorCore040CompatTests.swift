//
//  VectorCore040CompatTests.swift
//  VectorAccelerateTests
//
//  Downstream evidence for the unreleased VectorCore 0.4.0 candidate
//  (379e610f4365e820454c8c5328c4f87dfaa2031b; tested implementation 73d6a5bcd4fa1063e39d467284638b2812a84e7d).
//  Each test names the row(s) of VectorCore's downstream upgrade list
//  (Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/DOWNSTREAM_UPGRADES.md) it exercises, so
//  docs/migrations/VECTORCORE-0.4.0-CANDIDATE.md can cite it.
//
//  Core's CPU tests cannot certify VectorAccelerate's GPU paths (VectorCore Docs/Package_Boundaries.md), so
//  every GPU assertion here is paired with `MetalComputeProvider.routingTelemetry()` evidence that the GPU
//  kernel served the call, and every GPU test has a `preferGPU: false` twin that pins the CPU fallback
//  separately. Providers are built with `fallbackToCPU: false` so a GPU failure surfaces as a thrown error
//  instead of a CPU result masquerading as GPU evidence.
//
//  Only API that exists in VectorCore 0.3.3 as well is referenced, so this file compiles against either
//  pin; the assertions that encode 0.4.0-only behaviour are marked "(0.4.0 behaviour)".
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class VectorCore040CompatTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    // MARK: - Fixtures

    /// Deterministic values in [-1, 1) (the SoAScoringParityTests generator; no Date / Math.random).
    private func lcg(_ n: Int, seed: UInt64) -> [Float] {
        var s = seed &+ 0x9E3779B97F4A7C15
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float((s >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1
        }
    }

    private func dynamicVectors(count: Int, dim: Int, seed: UInt64) -> [DynamicVector] {
        (0..<count).map { DynamicVector(lcg(dim, seed: seed &+ UInt64($0))) }
    }

    private func makeProvider(gpu: Bool) async throws -> MetalComputeProvider {
        try await MetalComputeProvider(configuration: .init(preferGPU: gpu, fallbackToCPU: false))
    }

    /// Clears every default distance gate of `GPUDecisionEngine.shouldUseGPU`: candidates >= 500 and
    /// candidates * dimension >= 50_000 (pinned by RoutingProvenanceTests).
    private let distanceGPUScale = (count: 2000, dim: 128)
    /// Clears every default selection gate: candidates >= 500, 10 <= k <= 1000, candidates * k >= 50_000.
    private let selectionGPUScale = (count: 5000, dim: 64, k: 10)
    /// Below `minCandidatesForGPU` (500): the CPU path under default configuration.
    private let cpuScale = (count: 300, dim: 64, k: 10)

    /// FP32 accumulation bound. Each of `dim` products carries about one ulp and the summation order
    /// differs between VectorCore's SIMD4 kernels and VectorAccelerate's GPU / vDSP paths, so cross-package
    /// agreement is asserted to `dim * 2^-23` relative (1.5e-5 at 128, 6.1e-5 at 512) and never bit-exact.
    /// Cosine needs three such sums (dot and both norms), so cosine assertions use twice this bound.
    private func crossPackageTolerance(dim: Int) -> Float { max(1e-6, Float(dim) * Float.ulpOfOne) }

    private func assertServedBy(gpu: Bool, _ t: MetalComputeProvider.RoutingTelemetry,
                                file: StaticString = #filePath, line: UInt = #line) {
        if gpu {
            XCTAssertGreaterThanOrEqual(t.gpuKernel, 1, "GPU kernel must have served this call. telemetry=\(t)",
                                        file: file, line: line)
            XCTAssertEqual(t.cpuDecisionEngine, 0, "decision engine must not have refused the GPU. telemetry=\(t)",
                           file: file, line: line)
        } else {
            XCTAssertEqual(t.gpuKernel, 0, "CPU twin must not dispatch a GPU kernel. telemetry=\(t)",
                           file: file, line: line)
            XCTAssertGreaterThanOrEqual(t.cpuDecisionEngine + t.cpuPolicy, 1, "telemetry=\(t)", file: file, line: line)
        }
        XCTAssertEqual(t.cpuFallbackAfterGPUError, 0, "a GPU error was rescued silently: \(t.lastGPUErrorDescription ?? "?")",
                       file: file, line: line)
        XCTAssertEqual(t.cpuFallbackEmptyGPUResult, 0, "an empty GPU result was rescued silently", file: file, line: line)
    }

    // Double-accumulated references, independent of both packages' FP32 arithmetic.
    private func refL2(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { let d = Double(a[i]) - Double(b[i]); s += d * d }
        return Float(s.squareRoot())
    }
    private func refDot(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { s += Double(a[i]) * Double(b[i]) }
        return Float(s)
    }

    // MARK: - Priority 1: API and dispatch (rows CB-02, SG-01, SG-05, BH-52, BH-55)

    /// Row CB-02: `ComputeDevice.gpu().isAvailable` is removed. VectorAccelerate's guards now share one
    /// runtime Metal check; on a Metal host every guard reports available and the convenience
    /// initializer that used the removed member constructs a provider.
    func testGPUAvailabilityGuardsUseRuntimeMetalCheck() async throws {
        XCTAssertTrue(MetalDevice.isGPUAvailable())
        XCTAssertEqual(MetalDevice.isGPUAvailable(), MetalDevice.isAvailable)
        XCTAssertTrue(AcceleratedVectorFactory.isAccelerationAvailable)
        let provider = try await MetalComputeProvider()
        XCTAssertEqual(provider.device, .gpu(index: 0))
    }

    /// Rows SG-05 / BH-52 (context) and BH-55: VectorCore still declares `BatchKernelProvider` with
    /// `any DistanceMetric` parameters at the candidate (R01's narrowing to `any DistanceMetric<Float>`
    /// applies to the search facades and `VectorCollection.search`, which VectorAccelerate does not
    /// implement), and `Operations.findNearest` still dispatches to the installed provider first. This
    /// proves the witnesses bind through the existential and that the GPU actually executed the search,
    /// then checks the result against Core's own CPU route with no provider installed.
    func testOperationsDispatchReachesProviderWitnessesAndExecutesGPU() async throws {
        let (n, dim, k) = selectionGPUScale
        let cands = dynamicVectors(count: n, dim: dim, seed: 0x0401)
        let query = DynamicVector(lcg(dim, seed: 0x0400))
        let reference = try await Operations.findNearest(to: query, in: cands, k: k, metric: EuclideanDistance())
        let qa = query.toArray()
        let tol = crossPackageTolerance(dim: dim)
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let results = try await Operations.$computeProvider.withValue(provider) {
                try await Operations.findNearest(to: query, in: cands, k: k, metric: EuclideanDistance())
            }
            let t = await provider.routingTelemetry()
            assertServedBy(gpu: gpu, t)
            XCTAssertEqual(results.count, k)
            // The k nearest distances, in order (robust to near-tie index swaps), each self-consistent.
            for (got, ref) in zip(results, reference) {
                XCTAssertEqual(got.distance, ref.distance, accuracy: tol * max(1, abs(ref.distance)), "gpu=\(gpu)")
            }
            for r in results {
                XCTAssertEqual(r.distance, refL2(qa, cands[r.index].toArray()),
                               accuracy: tol * max(1, abs(r.distance)), "gpu=\(gpu) index \(r.index)")
            }
        }
    }

    /// Row BH-12 and the dot-product convention: VectorCore's `DotProductDistance` scores `-dot`
    /// (ascending = most similar first); VectorAccelerate's own `.dotProduct` is the raw dot with
    /// larger-is-closer. Through the `BatchKernelProvider` hook the provider must defer to Core's witness,
    /// so the distances are Core's bits exactly and the ranking is Core's ranking.
    func testDotProductThroughProviderKeepsCoreNegatedConvention() async throws {
        let provider = try await makeProvider(gpu: true)
        let (n, dim, k) = (600, 32, 5)
        let cands = dynamicVectors(count: n, dim: dim, seed: 0x0D07)
        let query = DynamicVector(lcg(dim, seed: 0x0D00))
        let metric = DotProductDistance()
        let got = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: query, in: cands, k: k, metric: metric)
        }
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t, .init(),
                       "dot product is never GPU-routed through the Core hook (the raw-dot kernel is not Core's -dot); telemetry=\(t)")
        let coreDistances = metric.batchDistance(query: query, candidates: cands)
        let coreTopK = TopKSelection.select(k: k, from: coreDistances, tieBreaker: .smallerIndex)
        XCTAssertEqual(got.map(\.index), coreTopK.indices)
        XCTAssertEqual(got.map(\.distance.bitPattern), coreTopK.distances.map(\.bitPattern), "distances must be Core's bits")
        let qa = query.toArray()
        for r in got {
            XCTAssertEqual(r.distance, -refDot(qa, cands[r.index].toArray()),
                           accuracy: crossPackageTolerance(dim: dim) * max(1, abs(r.distance)), "Core's convention is -dot")
        }
        // The provider's own dot-product API is the raw similarity: same ranking, opposite sign.
        let raw = try await provider.findNearest(query: query, in: cands, k: k, metric: .dotProduct)
        XCTAssertEqual(raw.map(\.index), got.map(\.index))
        for (a, b) in zip(raw, got) {
            XCTAssertEqual(a.distance, -b.distance, accuracy: crossPackageTolerance(dim: dim) * max(1, abs(a.distance)))
        }
    }

    /// Row BH-02: with a provider installed, Core hands every metric to it (dispatch is unconditional on
    /// the metric); for a metric VectorAccelerate does not accelerate, the provider defers to the metric's
    /// own `batchDistance` and Core's shared Top-K policy, so the result is bit-identical to Core's CPU
    /// route and no VectorAccelerate kernel or fallback is involved.
    func testUnsupportedMetricThroughProviderIsBitIdenticalToCoreCPURoute() async throws {
        let provider = try await makeProvider(gpu: true)
        let (n, dim, k) = (300, 200, 8)   // k < n / 10: Core's generic small-k streaming route
        let cands = dynamicVectors(count: n, dim: dim, seed: 0x5005)
        let query = DynamicVector(lcg(dim, seed: 0x5000))
        let viaProvider = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: query, in: cands, k: k, metric: ManhattanDistance())
        }
        let coreOnly = try await Operations.findNearest(to: query, in: cands, k: k, metric: ManhattanDistance())
        XCTAssertEqual(viaProvider.map(\.index), coreOnly.map(\.index))
        XCTAssertEqual(viaProvider.map(\.distance.bitPattern), coreOnly.map(\.distance.bitPattern))
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t, .init(), "the Core-metric path bypasses every VectorAccelerate kernel and fallback. telemetry=\(t)")
    }

    /// Row SG-01 / BH-01: `Operations.findNearestBatch(queries:in:k:metric:)` compiles unchanged (the new
    /// `cpuStrategy:` parameter defaults) and, with a provider installed, routes through the provider's
    /// inherited `findNearestBatch` (per-query `findNearest`), so every row equals the single-query result
    /// bit for bit and the GPU runs one fused dispatch per query.
    func testFindNearestBatchThroughProviderEqualsPerQuery() async throws {
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let (n, dim, k) = selectionGPUScale
            let cands = dynamicVectors(count: n, dim: dim, seed: 0xB47C)
            let queries = dynamicVectors(count: 3, dim: dim, seed: 0xB400)
            let batched = try await Operations.$computeProvider.withValue(provider) {
                try await Operations.findNearestBatch(queries: queries, in: cands, k: k, metric: EuclideanDistance())
            }
            let t = await provider.routingTelemetry()
            assertServedBy(gpu: gpu, t)
            if gpu { XCTAssertEqual(t.gpuKernel, queries.count, "one fused GPU dispatch per query. telemetry=\(t)") }
            XCTAssertEqual(batched.count, queries.count)
            for (i, q) in queries.enumerated() {
                let single = try await Operations.$computeProvider.withValue(provider) {
                    try await Operations.findNearest(to: q, in: cands, k: k, metric: EuclideanDistance())
                }
                XCTAssertEqual(batched[i].map(\.index), single.map(\.index), "gpu=\(gpu) query \(i)")
                XCTAssertEqual(batched[i].map(\.distance.bitPattern), single.map(\.distance.bitPattern),
                               "gpu=\(gpu) query \(i): same path, same bits")
            }
        }
    }

    // MARK: - Priority 2: ownership and GPU lifetimes (SoA_Layout_Contract.md; rows CB-43, BH-62, AB-09, BH-47)

    /// The frozen SoA layout contract, checked against the candidate's descriptor and against what the GPU
    /// kernel reads: the golden case of SoA_Layout_Contract.md (five constant 512-d vectors, zero query,
    /// squared distances 512 * j^2) scored through the zero-copy buffer.
    func testSoALayoutContractGoldenThroughZeroCopyGPUPath() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(fallbackToCPU: false))
        let vectors = (0..<5).map { Vector512Optimized(repeating: Float($0)) }
        let set = try SoACandidateSet(candidates: vectors, device: context.device)
        XCTAssertEqual(set.layout, SoALayout.forType(Vector512Optimized.self, count: 5, pageAligned: true))
        XCTAssertEqual(set.layout.lanes, 128)
        XCTAssertEqual(set.layout.count, 5)
        XCTAssertEqual(set.layout.laneStrideBytes, 80)
        XCTAssertEqual(set.layout.logicalByteCount, 10_240)
        XCTAssertEqual(set.layout.elementIndex(lane: 127, candidate: 4), 639)
        XCTAssertEqual(set.layout.allocatedByteCount % MetalDevice.pageSize, 0)
        XCTAssertGreaterThanOrEqual(set.layout.allocatedByteCount, set.layout.logicalByteCount)
        XCTAssertTrue(set.isZeroCopy)
        let zero = Vector512Optimized(repeating: 0)
        let l2 = try await provider.batchDistance(query: zero, against: set, metric: .euclidean)
        XCTAssertEqual(l2.count, 5)
        for j in 0..<5 {
            // sqrt(512 * j^2) = j * sqrt(512): 512 equal squares, so only the root rounds.
            let expected = Float(j) * Float(512).squareRoot()
            XCTAssertEqual(l2[j], expected, accuracy: 4 * Float.ulpOfOne * max(1, expected), "candidate \(j)")
        }
        let nearest = try await provider.findNearest(query: zero, in: set, k: 5, metric: .euclidean)
        XCTAssertEqual(nearest.map(\.index), [0, 1, 2, 3, 4])
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 2, "both SoA calls run the lane-major GPU kernel. telemetry=\(t)")
    }

    /// Borrow mode (SoA contract, "Lifetime and transfer"): VectorAccelerate never calls
    /// `consumeAllocation()`; the SoA keeps ownership, the `MTLBuffer` aliases its page-aligned allocation
    /// for every query, and the set is released once the last GPU completion handler has fired.
    func testSoABorrowModeKeepsOwnershipAndReleasesAfterGPUCompletion() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(fallbackToCPU: false))
        weak var weakSet: SoACandidateSet<Vector768Optimized>?
        do {
            let candidates = try (0..<64).map { try Vector768Optimized(lcg(768, seed: 0x50A0 &+ UInt64($0))) }
            let set = try SoACandidateSet(candidates: candidates, device: context.device)
            weakSet = set
            let query = try Vector768Optimized(lcg(768, seed: 0x50AF))
            for _ in 0..<3 {
                _ = try await provider.batchDistance(query: query, against: set, metric: .cosine)
                let (base, byteCount) = try XCTUnwrap(set.soa.pageAlignedBytes,
                    "ownership must not have been transferred (consumeAllocation() would make this nil)")
                XCTAssertEqual(UnsafeMutableRawPointer(mutating: base), set.buffer.contents(), "the buffer must alias the SoA allocation")
                XCTAssertEqual(byteCount, set.layout.allocatedByteCount)
                XCTAssertEqual(set.buffer.length, byteCount)
            }
        }
        // `executeAndWait` returned after completion, so the pin (`cb.addCompletedHandler { _ in _ = set }`)
        // has fired; Metal may release the command buffer, and with it the handler's capture, a little later.
        for _ in 0..<200 where weakSet != nil { try await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertNil(weakSet, "the candidate set (and its SoA allocation) must be released after the GPU completes")
    }

    /// SoA contract: empty storage has no page-aligned allocation, and a single candidate is still a whole
    /// page. The bridge must handle both ends without an unsafe import.
    func testSoACandidateSetBoundaryCounts() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(fallbackToCPU: false))
        let zero = Vector512Optimized(repeating: 0)

        let one = try SoACandidateSet(candidates: [Vector512Optimized(repeating: 3)], device: context.device)
        XCTAssertEqual(one.layout.count, 1)
        XCTAssertEqual(one.layout.allocatedByteCount, MetalDevice.pageAlignedLength(one.layout.logicalByteCount))
        XCTAssertTrue(one.isZeroCopy)
        let d1 = try await provider.batchDistance(query: zero, against: one, metric: .euclidean)
        XCTAssertEqual(d1.count, 1)
        XCTAssertEqual(d1[0], 3 * Float(512).squareRoot(), accuracy: 4 * Float.ulpOfOne * 68)

        let empty = try SoACandidateSet(candidates: [Vector512Optimized](), device: context.device)
        XCTAssertEqual(empty.layout.count, 0)
        XCTAssertNil(empty.soa.pageAlignedBytes, "empty SoA storage provides no page-aligned allocation (contract)")
        XCTAssertFalse(empty.isZeroCopy)
        let d0 = try await provider.batchDistance(query: zero, against: empty, metric: .euclidean)
        XCTAssertTrue(d0.isEmpty)
        let n0 = try await provider.findNearest(query: zero, in: empty, k: 3, metric: .euclidean)
        XCTAssertTrue(n0.isEmpty)
    }

    /// Rows CB-43 / BH-62 / AB-09: dynamic storage is now a copy-on-write struct. A copy taken before a
    /// mutation must stay independent all the way through the provider's staging into GPU buffers.
    func testDynamicVectorCopyOnWriteIsolationThroughProvider() async throws {
        let provider = try await makeProvider(gpu: true)
        let original = lcg(128, seed: 0xC0)
        var a = DynamicVector(original)
        let b = a
        a[0] = 0.25                                      // exactly representable: the difference is exactly original[0] - 0.25
        XCTAssertEqual(b[0].bitPattern, original[0].bitPattern, "the copy must not observe the write")
        XCTAssertEqual(a[0], 0.25)
        XCTAssertEqual(b.toArray().map(\.bitPattern), original.map(\.bitPattern))
        let expected = abs(original[0] - 0.25)
        let d = try await provider.batchDistance(query: b, candidates: [a], metric: .euclidean)   // CPU path (1 candidate)
        XCTAssertEqual(d.count, 1)
        XCTAssertEqual(d[0], expected, accuracy: 4 * Float.ulpOfOne * max(1, expected), "only coordinate 0 differs")
        // At GPU scale (distanceGPUScale) the staged candidates are copies too: a and b keep their own values.
        let cands = Array(repeating: a, count: distanceGPUScale.count - 1) + [b]
        let d2 = try await provider.batchDistance(query: b, candidates: cands, metric: .euclidean)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1, "\(distanceGPUScale.count) x 128 clears the distance gates. telemetry=\(t)")
        XCTAssertEqual(d2.last, 0, "b against itself")
        XCTAssertEqual(d2[0], expected, accuracy: 4 * Float.ulpOfOne * max(1, expected))
    }

    // MARK: - Priority 3: numerical and search contracts (rows BH-12, BH-13, BH-14, BH-03, BH-06)

    /// Row BH-12: Euclidean through the Core hook is the rooted distance and matches Core's witness within
    /// the FP32 bound, on the GPU and on the CPU fallback separately; the hook and the typed `.euclidean`
    /// API take the same path (same bits).
    func testEuclideanMatchesCoreWitnessOnGPUAndCPU() async throws {
        let (n, dim) = distanceGPUScale
        let cands = dynamicVectors(count: n, dim: dim, seed: 0xE0C1)
        let query = DynamicVector(lcg(dim, seed: 0xE000))
        let core = EuclideanDistance().batchDistance(query: query, candidates: cands)
        let qa = query.toArray()
        let tol = crossPackageTolerance(dim: dim)
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let hook: any BatchKernelProvider = provider
            let viaHook = try await hook.batchDistance(query: query, candidates: cands, metric: EuclideanDistance())
            let direct = try await provider.batchDistance(query: query, candidates: cands, metric: .euclidean)
            let t = await provider.routingTelemetry()
            assertServedBy(gpu: gpu, t)
            if gpu { XCTAssertEqual(t.gpuKernel, 2, "telemetry=\(t)") }
            XCTAssertEqual(viaHook.map(\.bitPattern), direct.map(\.bitPattern), "gpu=\(gpu): the hook takes the typed API's path")
            var worstCore: Float = 0, worstDouble: Float = 0
            for i in 0..<n {
                let scale = max(1, abs(core[i]))
                worstCore = max(worstCore, abs(viaHook[i] - core[i]) / scale)
                worstDouble = max(worstDouble, abs(viaHook[i] - refL2(qa, cands[i].toArray())) / scale)
            }
            XCTAssertLessThanOrEqual(worstCore, tol, "gpu=\(gpu): worst relative deviation from Core's EuclideanDistance witness")
            XCTAssertLessThanOrEqual(worstDouble, tol, "gpu=\(gpu): worst relative deviation from the Double reference")
        }
    }

    /// Row BH-14 (0.4.0 behaviour): Core's `CosineDistance` now returns NaN for any non-finite coordinate
    /// and 1 for a zero vector on either side. The provider GPU-routes `CosineDistance` (one of the two
    /// metrics whose kernel semantics are claimed to match Core's), so every edge class must agree with the
    /// candidate's witness on the GPU and on the CPU fallback.
    func testCosineEdgePolicyMatchesCoreOnGPUAndCPU() async throws {
        let (n, dim) = distanceGPUScale
        var arrays = (0..<n).map { lcg(dim, seed: 0xC05 &+ UInt64($0)) }
        let qa = lcg(dim, seed: 0xC00)
        arrays[0] = [Float](repeating: 0, count: dim)        // zero candidate            -> 1
        arrays[1] = qa                                        // identical                 -> 0
        arrays[2] = qa.map { -$0 }                            // antipodal                 -> 2
        arrays[3] = qa.map { $0 * 1e-3 }                      // scaled                    -> 0 (scale invariance)
        arrays[4][7] = .nan                                   // NaN coordinate            -> NaN
        arrays[5][11] = .infinity                             // +Inf coordinate           -> NaN
        arrays[6][13] = -.infinity                            // -Inf coordinate           -> NaN
        arrays[7] = [Float](repeating: .infinity, count: dim) // all +Inf                  -> NaN
        arrays[8] = qa.map { $0 * 1e-15 }                     // tiny, norm still normal   -> 0
        let cands = arrays.map { DynamicVector($0) }
        let query = DynamicVector(qa)
        let metric = CosineDistance()
        let core = metric.batchDistance(query: query, candidates: cands)
        XCTAssertEqual(core[0], 1)
        XCTAssertEqual(core[1], 0, accuracy: 1e-6)
        XCTAssertEqual(core[2], 2, accuracy: 1e-6)
        XCTAssertTrue(core[4].isNaN && core[5].isNaN && core[6].isNaN && core[7].isNaN,
                      "precondition: the candidate's cosine policy (BH-14) returns NaN for non-finite coordinates")
        let tol = crossPackageTolerance(dim: dim) * 2
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let hook: any BatchKernelProvider = provider
            let got = try await hook.batchDistance(query: query, candidates: cands, metric: metric)
            assertServedBy(gpu: gpu, await provider.routingTelemetry())
            for i in 0..<n {
                if core[i].isNaN {
                    XCTAssertTrue(got[i].isNaN, "gpu=\(gpu) idx \(i): Core returns NaN, provider returned \(got[i])")
                } else {
                    XCTAssertFalse(got[i].isNaN, "gpu=\(gpu) idx \(i): Core returns \(core[i]), provider returned NaN")
                    XCTAssertEqual(got[i], core[i], accuracy: tol, "gpu=\(gpu) idx \(i)")   // cosine distance is in [0, 2]
                }
            }
        }
        // Zero query: 1 against every candidate, including a zero candidate (two zero vectors -> 1).
        let zeroQuery = DynamicVector([Float](repeating: 0, count: dim))
        let big = Array(cands[0..<4]) + Array(repeating: cands[9], count: n - 4)   // keeps the GPU vote
        XCTAssertEqual(Array(metric.batchDistance(query: zeroQuery, candidates: big)[0..<4]), [1, 1, 1, 1])
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let hook: any BatchKernelProvider = provider
            let got = try await hook.batchDistance(query: zeroQuery, candidates: big, metric: metric)
            assertServedBy(gpu: gpu, await provider.routingTelemetry())
            XCTAssertEqual(Array(got[0..<4]), [1, 1, 1, 1], "gpu=\(gpu): zero query -> distance 1 (two zero vectors included)")
        }
    }

    /// Row BH-12 (non-finite Euclidean): a NaN coordinate gives NaN, an infinite difference gives +Inf, and
    /// the same infinity on both sides gives NaN (Inf - Inf), exactly as docs/stability/DISTANCE-RANGE-CONTRACT.md
    /// states for VectorAccelerate. Huge *finite* coordinates are excluded on purpose: the rooted L2 keeps a
    /// Double rescue above the FP32 sum range where Core's FP32 witness overflows to +Inf.
    func testEuclideanNonFiniteCoordinatesMatchCoreOnGPUAndCPU() async throws {
        let (n, dim) = distanceGPUScale
        var arrays = (0..<n).map { lcg(dim, seed: 0xE1 &+ UInt64($0)) }
        var qa = lcg(dim, seed: 0xE100)
        qa[3] = .infinity                 // every ordinary candidate -> +Inf
        arrays[0][5] = .nan               // NaN coordinate            -> NaN
        arrays[1][9] = .infinity          // a second infinite diff    -> +Inf
        arrays[2][3] = .infinity          // same +Inf as the query    -> Inf - Inf = NaN
        arrays[3][3] = -.infinity         // opposite infinity         -> +Inf
        let cands = arrays.map { DynamicVector($0) }
        let query = DynamicVector(qa)
        let core = EuclideanDistance().batchDistance(query: query, candidates: cands)
        XCTAssertTrue(core[0].isNaN && core[2].isNaN && core[1] == .infinity && core[3] == .infinity && core[4] == .infinity,
                      "precondition: Core's Euclidean witness policy on non-finite coordinates")
        for gpu in [true, false] {
            let provider = try await makeProvider(gpu: gpu)
            let got = try await provider.batchDistance(query: query, candidates: cands, metric: .euclidean)
            assertServedBy(gpu: gpu, await provider.routingTelemetry())
            for i in 0..<n {
                XCTAssertEqual(got[i].isNaN, core[i].isNaN, "gpu=\(gpu) idx \(i): NaN parity (core \(core[i]), got \(got[i]))")
                XCTAssertEqual(got[i].isInfinite, core[i].isInfinite, "gpu=\(gpu) idx \(i): infinity parity (core \(core[i]), got \(got[i]))")
            }
        }
    }

    /// Rows BH-03 / BH-06 (the shared Top-K policy): exact ties keep the smaller original index and NaN
    /// ranks after every number, through the GPU fused distance + top-K path (k = 10, N = 5000) and through
    /// the CPU path (`preferGPU: false`, and k = N which no selection gate admits). Planted duplicates make
    /// the ties exact on the GPU too: identical inputs produce identical bits.
    func testTopKTiesAndNaNOrderingThroughGPUFusedAndCPUPaths() async throws {
        let (n, dim, k) = selectionGPUScale
        let qa = lcg(dim, seed: 0x70B0)
        // Far field: every coordinate offset by 10 (distance about 80); near field planted at the tail.
        var arrays = (0..<n).map { i in lcg(dim, seed: 0x70B1 &+ UInt64(i)).map { $0 + 10 } }
        func planted(_ delta: Float) -> [Float] { var v = qa; v[0] += delta; return v }
        arrays[n - 10] = planted(0.1); arrays[n - 9] = planted(0.1)   // exact tie -> n-10 before n-9
        arrays[n - 8] = planted(0.2); arrays[n - 7] = planted(0.2)
        arrays[n - 6] = planted(0.3); arrays[n - 5] = planted(0.3)
        arrays[n - 4] = planted(0.4); arrays[n - 3] = planted(0.4)
        arrays[n - 2] = planted(0.5); arrays[n - 1] = planted(0.6)
        let query = DynamicVector(qa)
        let cands = arrays.map { DynamicVector($0) }
        let expectedTop10 = Array((n - 10)..<n)

        let gpu = try await makeProvider(gpu: true)
        let fused = try await gpu.findNearest(query: query, in: cands, k: k, metric: .euclidean)
        assertServedBy(gpu: true, await gpu.routingTelemetry())
        XCTAssertEqual(fused.map(\.index), expectedTop10, "GPU fused path: ascending distance, exact ties by smaller index")
        XCTAssertEqual(fused[0].distance.bitPattern, fused[1].distance.bitPattern, "identical inputs must give identical GPU bits")

        // NaN variant: poison the first member of the nearest tie pair.
        var poisoned = arrays
        poisoned[n - 10][1] = .nan
        let pcands = poisoned.map { DynamicVector($0) }
        await gpu.resetRoutingTelemetry()
        let fusedNaN = try await gpu.findNearest(query: query, in: pcands, k: k, metric: .euclidean)
        assertServedBy(gpu: true, await gpu.routingTelemetry())
        XCTAssertFalse(fusedNaN.map(\.index).contains(n - 10), "a NaN distance ranks after every numeric distance")
        XCTAssertTrue(fusedNaN.allSatisfy { !$0.distance.isNaN })
        XCTAssertEqual(Array(fusedNaN.map(\.index).prefix(9)), Array((n - 9)..<n))

        // CPU twin, and k = N so the NaN tail is observable.
        let cpu = try await makeProvider(gpu: false)
        let top = try await cpu.findNearest(query: query, in: pcands, k: k, metric: .euclidean)
        XCTAssertEqual(Array(top.map(\.index).prefix(9)), Array((n - 9)..<n))
        let all = try await cpu.findNearest(query: query, in: pcands, k: n, metric: .euclidean)
        let t = await cpu.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0, "telemetry=\(t)")
        XCTAssertEqual(all.count, n)
        XCTAssertEqual(all.last?.index, n - 10, "NaN last")
        XCTAssertEqual(all.last?.distance.isNaN, true)
        XCTAssertEqual(Array(all.map(\.index).prefix(9)), Array((n - 9)..<n))
        XCTAssertEqual(Set(all.map(\.index)), Set(0..<n))
        // Core's witness agrees on the planted prefix and the NaN tail.
        let coreRef = TopKSelection.select(k: n, from: EuclideanDistance().batchDistance(query: query, candidates: pcands),
                                           tieBreaker: .smallerIndex)
        XCTAssertEqual(Array(coreRef.indices.prefix(9)), Array((n - 9)..<n))
        XCTAssertEqual(coreRef.indices.last, n - 10)
    }

    /// SoA path (pointer `TopKSelection.select` over the GPU distance buffer): duplicates tie to the smaller index.
    func testSoAFindNearestTiesResolveToSmallerIndex() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(fallbackToCPU: false))
        let base = try Vector512Optimized(lcg(512, seed: 0x7E))
        var candidates = try (0..<40).map { try Vector512Optimized(lcg(512, seed: 0x7E1 &+ UInt64($0)).map { $0 + 5 }) }
        candidates[13] = base; candidates[29] = base; candidates[7] = base   // three exact duplicates of the query
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let nearest = try await provider.findNearest(query: base, in: set, k: 3, metric: .euclidean)
        XCTAssertEqual(nearest.map(\.index), [7, 13, 29])
        XCTAssertTrue(nearest.allSatisfy { $0.distance == 0 })
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1, "telemetry=\(t)")
    }

    // MARK: - Priority 4: routing and fallback

    /// Provider installation is task-local and the decision engine still gates by scale: the same installed
    /// provider serves a small search on the CPU and a large one on the GPU, and nothing reaches it outside
    /// the `withValue` scope.
    func testInstalledProviderRoutesByScaleAndOnlyInsideScope() async throws {
        let provider = try await makeProvider(gpu: true)
        let small = dynamicVectors(count: cpuScale.count, dim: cpuScale.dim, seed: 0x5A11)
        let q = DynamicVector(lcg(cpuScale.dim, seed: 0x5A00))
        _ = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: q, in: small, k: cpuScale.k, metric: EuclideanDistance())
        }
        var t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0, "300 candidates < minCandidatesForGPU. telemetry=\(t)")
        XCTAssertGreaterThanOrEqual(t.cpuDecisionEngine, 1)
        await provider.resetRoutingTelemetry()
        _ = try await Operations.findNearest(to: q, in: small, k: cpuScale.k, metric: EuclideanDistance())   // not installed
        t = await provider.routingTelemetry()
        XCTAssertEqual(t, .init(), "nothing may reach a provider that is not installed")
        let (n, dim, k) = selectionGPUScale
        let big = dynamicVectors(count: n, dim: dim, seed: 0xB16)
        let qb = DynamicVector(lcg(dim, seed: 0xB00))
        _ = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: qb, in: big, k: k, metric: EuclideanDistance())
        }
        t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1, "N=5000, k=10 clears every selection gate. telemetry=\(t)")
        XCTAssertEqual(t.cpuDecisionEngine, 0, "telemetry=\(t)")
    }

    /// Unsupported shapes: a ragged candidate set is rejected (by Core's validation or the provider's guard)
    /// before any routing event, and a dimension that is not a multiple of four at GPU scale is still
    /// served by the GPU with results within the FP32 bound of Core's route.
    func testRaggedInputRejectedAndOddDimensionServedOnGPU() async throws {
        let provider = try await makeProvider(gpu: true)
        let ragged = [DynamicVector([1, 2, 3, 4]), DynamicVector([1, 2, 3])]
        do {
            _ = try await Operations.$computeProvider.withValue(provider) {
                try await Operations.findNearest(to: DynamicVector([0, 0, 0, 0]), in: ragged, k: 1, metric: EuclideanDistance())
            }
            XCTFail("ragged candidates must be rejected")
        } catch {
            XCTAssertTrue(error is VectorError, "expected a VectorError, got \(error)")
        }
        var t = await provider.routingTelemetry()
        XCTAssertEqual(t, .init(), "rejection must happen before any routing event. telemetry=\(t)")

        let (n, k) = (selectionGPUScale.count, selectionGPUScale.k)
        let dim = 13
        let cands = dynamicVectors(count: n, dim: dim, seed: 0x0DD)
        let q = DynamicVector(lcg(dim, seed: 0x0D0))
        let got = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: q, in: cands, k: k, metric: EuclideanDistance())
        }
        t = await provider.routingTelemetry()
        assertServedBy(gpu: true, t)
        let ref = try await Operations.findNearest(to: q, in: cands, k: k, metric: EuclideanDistance())
        let tol = crossPackageTolerance(dim: dim)
        for (g, r) in zip(got, ref) {
            XCTAssertEqual(g.distance, r.distance, accuracy: tol * max(1, abs(r.distance)))
        }
    }

    /// `preferGPU: false` is the documented CPU fallback: at GPU scale, through the Core hook, nothing
    /// reaches a kernel and the cosine results still satisfy Core's witness.
    func testPreferGPUFalseServesOperationsDispatchOnCPU() async throws {
        let provider = try await makeProvider(gpu: false)
        let (n, dim, k) = selectionGPUScale
        let cands = dynamicVectors(count: n, dim: dim, seed: 0xF00)
        let q = DynamicVector(lcg(dim, seed: 0xF0F))
        let got = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: q, in: cands, k: k, metric: CosineDistance())
        }
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0, "telemetry=\(t)")
        XCTAssertGreaterThanOrEqual(t.cpuDecisionEngine, 1, "telemetry=\(t)")
        let ref = TopKSelection.select(k: k, from: CosineDistance().batchDistance(query: q, candidates: cands), tieBreaker: .smallerIndex)
        for (g, r) in zip(got.map(\.distance), ref.distances) {
            XCTAssertEqual(g, r, accuracy: crossPackageTolerance(dim: dim) * 2)
        }
    }

    // MARK: - Row BH-19: `Vector<D>.isNormalized` is a computed witness at the candidate

    /// `KernelDistanceProviders` switch cosine to the dot-product kernel when both sides report
    /// `isNormalized`. At the candidate `Vector<Dim512>` answers with its computed check (0.3.3 answered the
    /// protocol default `false`), so normalized `Vector<Dim512>` inputs now take the dot path (0.4.0
    /// behaviour). Both paths must stay within tolerance of Core's `CosineDistance`; `DynamicVector` keeps
    /// the default and the full kernel.
    func testVectorDNormalizedInputsTakeDotPathWithinCoreTolerance() async throws {
        let context = try await Metal4Context()
        let cosineProvider = try await CosineKernelDistanceProvider(context: context)
        let dim = 512
        let unit = try (0..<64).map { try Vector<Dim512>(lcg(dim, seed: 0x19 &+ UInt64($0))).normalized().get() }
        let q = try Vector<Dim512>(lcg(dim, seed: 0x1900)).normalized().get()

        // (0.4.0 behaviour, BH-19) Read through the existential, exactly as `KernelDistanceProviders` do
        // (`(vector as? any IndexableVector)?.isNormalized`): at the candidate the `IndexableVector` witness
        // for `Vector<D>` is the documented computed check; at 0.3.3 the same read answered the protocol
        // default `false` (the concrete member existed, but the witness did not use it). The check reads an
        // FP32 `magnitudeSquared`, so a freshly normalized 512-d vector can legitimately report false when its
        // accumulated squared norm misses 1 by more than 1e-6; what must hold is that the witness *is* that check.
        func witness(_ v: Vector<Dim512>) -> Bool { (v as any IndexableVector).isNormalized }
        for v in unit + [q] {
            let deviation = abs(v.magnitudeSquared - 1)
            XCTAssertEqual(witness(v), deviation < 1e-6,
                           "the IndexableVector witness of Vector<D> must be the computed check (|‖v‖²−1| = \(deviation))")
            XCTAssertEqual(witness(v), v.isNormalized, "witness and concrete member must agree")
        }
        XCTAssertFalse((DynamicVector(q.toArray()) as any IndexableVector).isNormalized, "DynamicVector keeps the protocol default")
        let reporting = unit.filter(witness)
        let worstDeviation = unit.map { abs($0.magnitudeSquared - 1) }.max() ?? 0
        XCTAssertTrue(witness(q) && reporting.count >= unit.count / 2,
                      "precondition: enough unit vectors must report normalized to exercise the dot path "
                      + "(q: \(witness(q)), \(reporting.count)/\(unit.count) candidates, worst |‖v‖²−1| = \(worstDeviation))")
        // Against 0.3.3 nothing reports normalized; the assertion above has already failed, so stop here
        // instead of indexing an empty array (a crash would take the whole test process down).
        guard witness(q), let firstReporting = reporting.first else { return }

        // Dot path: query and every candidate report normalized -> `KernelDistanceProviders` use the dot kernel.
        let core = CosineDistance().batchDistance(query: q, candidates: reporting)
        let dotPath = try await cosineProvider.batchDistance(from: q, to: reporting, metric: .cosine)
        // Full path: one non-normalized candidate (scaled by 3) forces the cosine kernel for the whole batch.
        let mixed = try reporting + [Vector<Dim512>(firstReporting.toArray().map { $0 * 3 })]
        let coreMixed = CosineDistance().batchDistance(query: q, candidates: mixed)
        let fullPath = try await cosineProvider.batchDistance(from: q, to: mixed, metric: .cosine)
        // |‖v‖² − 1| < 1e-6 on both sides ⇒ 1 − q·c is within 1e-6 of the true cosine distance, plus accumulation.
        let tol = crossPackageTolerance(dim: dim) * 2
        for i in 0..<reporting.count {
            XCTAssertEqual(dotPath[i], core[i], accuracy: tol + 1e-6, "dot path idx \(i)")
        }
        for i in 0..<mixed.count {
            XCTAssertEqual(fullPath[i], coreMixed[i], accuracy: tol, "full path idx \(i)")
        }
        XCTAssertEqual(fullPath.last ?? -1, core[0], accuracy: tol, "cosine is scale-invariant: the x3 copy scores like its unit original")
    }
}
