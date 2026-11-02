//
//  CapabilityCapPolicyTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-011: silent capability caps. Kernels clamped over-cap requests to their
//  hardware limits and produced plausible-but-wrong output (attention/learned/neural
//  truncated projections; IVF builders dropped probes >= 64) or silently returned leaving
//  stale pool bytes in the output (topk_select_batch_kernel K > 128, fused_l2_topk
//  D > 768 / tgs > 256). TopKParameters.init even clamped k = min(k, 128) itself — a
//  top-200 request silently became top-128.
//
//  Policy (Group C amortized fix): capability caps become THROWN ERRORS at the host
//  choke points and SENTINEL-FILLS kernel-side — never silent truncation, never stale
//  bytes. Sentinel convention (established by WarpSelectionPaddingTests / VA3-034):
//  index 0xFFFFFFFF, value +INF for min-selection / -INF for max-selection.
//
import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class CapabilityCapPolicyTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        try await super.setUp()
        context = try await Metal4Context()
    }

    override func tearDown() async throws {
        context = nil
        try await super.tearDown()
    }

    // MARK: - Attention: headDim caps (single-head 256, multi-head 64)

    /// Single-head attention silently projected only the first 256 of headDim dimensions
    /// (`effectiveHeadDim = min(headDim, 256)`). Over-cap requests must throw.
    func testAttentionSingleHeadDimOverCapThrows() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(inputDimension: 32, headDimension: 300, numHeads: 1)
        try await kernel.createRandomWeights(config: config)

        let v = [Float](repeating: 0.5, count: 32)
        do {
            _ = try await kernel.computeSimilarities(queries: [v], keys: [v])
            XCTFail("headDim 300 > 256 must throw, not silently truncate the projection")
        } catch { /* expected */ }
    }

    /// The multi-head kernel's cap is only 64 — trivially exceeded by real configs.
    func testAttentionMultiHeadDimOverCapThrows() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(inputDimension: 32, headDimension: 96, numHeads: 2)
        try await kernel.createRandomWeights(config: config)

        let v = [Float](repeating: 0.5, count: 32)
        do {
            _ = try await kernel.computeSimilarities(queries: [v], keys: [v])
            XCTFail("multi-head headDim 96 > 64 must throw, not silently truncate")
        } catch { /* expected */ }
    }

    /// Control (green before AND after): at-cap configurations keep working.
    func testAttentionAtCapStillWorks() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(inputDimension: 16, headDimension: 256, numHeads: 1)
        try await kernel.createRandomWeights(config: config)
        let v = [Float](repeating: 0.5, count: 16)
        let result = try await kernel.computeSimilarities(queries: [v], keys: [v])
        XCTAssertEqual(result.similarities.count, 1)
        XCTAssertTrue(result.similarities[0][0].isFinite, "at-cap attention must produce a finite similarity")
    }

    // MARK: - LearnedDistance: projectedDimension cap (256)

    /// The learned L2/cosine kernels hold the projected vectors in `float[256]` stack
    /// arrays and clamp `outputDim = min(projectedDimension, 256)`. Over-cap must throw.
    func testLearnedProjectedDimOverCapThrows() async throws {
        let kernel = try await LearnedDistanceKernel(context: context)
        let device = context.device.rawDevice
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())

        let inputDim = 8, projDim = 300
        let q = try XCTUnwrap(device.makeBuffer(length: inputDim * 4, options: .storageModeShared))
        let d = try XCTUnwrap(device.makeBuffer(length: inputDim * 4, options: .storageModeShared))
        let w = try XCTUnwrap(device.makeBuffer(length: projDim * inputDim * 4, options: .storageModeShared))
        let out = try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared))

        let params = LearnedDistanceKernel.Parameters(
            numQueries: 1, numDatabase: 1,
            inputDimension: inputDim, projectedDimension: projDim)
        do {
            try kernel.computeL2(
                queryVectors: q, databaseVectors: d, projectionWeights: w,
                distances: out, parameters: params, commandBuffer: commandBuffer)
            XCTFail("projectedDimension 300 > 256 must throw, not silently truncate")
        } catch { /* expected */ }
    }

    /// Control: at-cap projectedDimension encodes without throwing.
    func testLearnedAtCapStillEncodes() async throws {
        let kernel = try await LearnedDistanceKernel(context: context)
        let device = context.device.rawDevice
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())

        let inputDim = 8, projDim = 256
        let q = try XCTUnwrap(device.makeBuffer(length: inputDim * 4, options: .storageModeShared))
        let d = try XCTUnwrap(device.makeBuffer(length: inputDim * 4, options: .storageModeShared))
        let w = try XCTUnwrap(device.makeBuffer(length: projDim * inputDim * 4, options: .storageModeShared))
        let out = try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared))

        let params = LearnedDistanceKernel.Parameters(
            numQueries: 1, numDatabase: 1,
            inputDimension: inputDim, projectedDimension: projDim)
        try kernel.computeL2(
            queryVectors: q, databaseVectors: d, projectionWeights: w,
            distances: out, parameters: params, commandBuffer: commandBuffer)
    }

    // MARK: - NeuralQuantization: latentDimension cap (128)

    /// Worse than truncation: the truncated latentDim is then used as the codes/scales
    /// row stride, so latentDimension > 128 also scrambles the output layout. Every
    /// config entry point (loadWeights x2, createRandomWeights) must throw.
    func testNeuralLatentDimOverCapThrows() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig(inputDimension: 16, latentDimension: 200)
        do {
            try await kernel.createRandomWeights(config: config)
            XCTFail("latentDimension 200 > 128 must throw — the kernels truncate AND scramble the row stride")
        } catch { /* expected */ }
    }

    // MARK: - IVF candidate builder: nprobe cap (64)

    /// All the builder kernels bound their probe loops with `p < nprobe && p < 64` —
    /// probes >= 64 were silently dropped (recall silently degrades). Must throw.
    func testIVFNprobeOverCapThrows() async throws {
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        let device = context.device.rawDevice
        let centroids = try XCTUnwrap(device.makeBuffer(length: 65 * 4, options: .storageModeShared))
        memset(centroids.contents(), 0, 65 * 4)
        let offsets = try XCTUnwrap(device.makeBuffer(length: 2 * 4, options: .storageModeShared))
        memset(offsets.contents(), 0, 2 * 4)

        do {
            _ = try await kernel.buildCandidates(
                nearestCentroids: centroids, listOffsets: offsets,
                numQueries: 1, nprobe: 65, numLists: 1)
            XCTFail("nprobe 65 > 64 must throw — the kernels silently drop probes >= 64")
        } catch { /* expected */ }
    }

    /// Control: nprobe at the cap still runs.
    func testIVFNprobeAtCapStillRuns() async throws {
        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        let device = context.device.rawDevice
        let centroids = try XCTUnwrap(device.makeBuffer(length: 64 * 4, options: .storageModeShared))
        memset(centroids.contents(), 0, 64 * 4)
        let offsets = try XCTUnwrap(device.makeBuffer(length: 2 * 4, options: .storageModeShared))
        memset(offsets.contents(), 0, 2 * 4)

        let result = try await kernel.buildCandidates(
            nearestCentroids: centroids, listOffsets: offsets,
            numQueries: 1, nprobe: 64, numLists: 1)
        XCTAssertEqual(result.numQueries, 1)
    }

    // MARK: - TopKParameters: the init must not silently clamp k

    /// `TopKParameters.init` clamped `k = min(k, 128)` — a top-200 request silently
    /// became top-128 with no signal to the caller. The init must preserve the request
    /// (the kernel sentinel-fills over-cap rows; `select()` still throws).
    func testTopKParametersPreserveRequestedK() {
        let dense = TopKParameters(batchSize: 1, numElements: 4, k: 200)
        XCTAssertEqual(dense.k, 200, "dense init silently clamped k")
        XCTAssertEqual(dense.outputStride, 200, "dense init silently clamped outputStride")

        let strided = TopKParameters(batchSize: 1, numElements: 4, k: 200,
                                     inputStride: 4, outputStride: 200)
        XCTAssertEqual(strided.k, 200, "custom-stride init silently clamped k")
    }

    // MARK: - topk_select_batch_kernel: over-cap K must sentinel-fill, not no-op

    /// K > 128 hit `if (K > MAX_K) return;` — the output buffer kept whatever bytes the
    /// pool left there (the VA3-008/VA3-034 stale-bytes symptom). Over-cap rows must read
    /// back all-sentinel through the raw `encode()` API (which bypasses `select()`'s guard).
    func testTopKBatchEncodeOverCapSentinelFills() async throws {
        let kernel = try await TopKSelectionKernel(context: context)
        let device = context.device.rawDevice

        let batch = 2, n = 10, k = 200
        var input: [Float] = []
        for q in 0..<batch { for i in 0..<n { input.append(Float(i + q)) } }
        let inputBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: input, length: input.count * 4, options: .storageModeShared))

        let outCount = batch * k
        let poisonValues = [Float](repeating: 12345.0, count: outCount)
        let poisonIndices = [UInt32](repeating: 0xDEAD_BEEF, count: outCount)
        let valuesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonValues, length: outCount * 4, options: .storageModeShared))
        let indicesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonIndices, length: outCount * 4, options: .storageModeShared))

        let params = TopKParameters(batchSize: batch, numElements: n, k: k, mode: .minimum)
        try await context.executeAndWait { _, encoder in
            kernel.encode(into: encoder, input: inputBuffer,
                          outputValues: valuesBuffer, outputIndices: indicesBuffer,
                          parameters: params)
        }

        let vals = Array(UnsafeBufferPointer(
            start: valuesBuffer.contents().bindMemory(to: Float.self, capacity: outCount), count: outCount))
        let idxs = Array(UnsafeBufferPointer(
            start: indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: outCount), count: outCount))
        for slot in 0..<outCount {
            XCTAssertEqual(vals[slot], .infinity, "over-cap K: value slot \(slot) must be sentinel, not stale/partial data")
            XCTAssertEqual(idxs[slot], 0xFFFF_FFFF, "over-cap K: index slot \(slot) must be sentinel")
        }
    }

    // MARK: - fused_l2_topk: over-cap D must sentinel-fill, not no-op

    /// `if (D > MAX_D || tgs > MAX_TGS || K == 0) return;` left the entire output
    /// untouched. Driven kernel-direct (the params type guards D <= 768, so the host
    /// path can't reach it — the kernel must still never publish stale bytes).
    func testFusedL2TopKOverDimSentinelFills() async throws {
        let pipeline = try await context.getPipeline(functionName: "fused_l2_topk")
        let device = context.device.rawDevice

        let q = 2, n = 4, d = 1000, k = 4   // d > MAX_D (768)
        let queries = try XCTUnwrap(device.makeBuffer(length: q * d * 4, options: .storageModeShared))
        memset(queries.contents(), 0, q * d * 4)
        let dataset = try XCTUnwrap(device.makeBuffer(length: n * d * 4, options: .storageModeShared))
        memset(dataset.contents(), 0, n * d * 4)

        let outCount = q * k
        let poisonIndices = [UInt32](repeating: 0xDEAD_BEEF, count: outCount)
        let poisonDistances = [Float](repeating: 12345.0, count: outCount)
        let indicesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonIndices, length: outCount * 4, options: .storageModeShared))
        let distancesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonDistances, length: outCount * 4, options: .storageModeShared))

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queries, offset: 0, index: 0)
            encoder.setBuffer(dataset, offset: 0, index: 1)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(distancesBuffer, offset: 0, index: 3)
            var pQ = UInt32(q), pN = UInt32(n), pD = UInt32(d), pK = UInt32(k)
            encoder.setBytes(&pQ, length: 4, index: 4)
            encoder.setBytes(&pN, length: 4, index: 5)
            encoder.setBytes(&pD, length: 4, index: 6)
            encoder.setBytes(&pK, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: q, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        let idxs = Array(UnsafeBufferPointer(
            start: indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: outCount), count: outCount))
        let dists = Array(UnsafeBufferPointer(
            start: distancesBuffer.contents().bindMemory(to: Float.self, capacity: outCount), count: outCount))
        for slot in 0..<outCount {
            XCTAssertEqual(idxs[slot], 0xFFFF_FFFF, "over-cap D: index slot \(slot) holds stale bytes")
            XCTAssertEqual(dists[slot], .infinity, "over-cap D: distance slot \(slot) holds stale bytes")
        }
    }

    // MARK: - Throw-mid-encode: armed by these guards, fixed in Metal4Context

    /// The VA3-011 guards throw from inside `executeAndWait`'s open encoder — which armed
    /// the AUDIT-2 "throw-mid-encode" anchor: Metal4Context dropped the encoder without
    /// `endEncoding()`, and Metal API validation ABORTED the process ("Command encoder
    /// released without endEncoding" — observed as a signal-6 crash of this very suite
    /// pre-fix). A throwing encode closure must propagate its error; nothing may commit.
    func testThrowingEncodeClosurePropagatesWithoutCrash() async throws {
        struct Probe: Error {}
        do {
            try await context.executeAndWait { _, _ in throw Probe() }
            XCTFail("error must propagate out of executeAndWait")
        } catch is Probe { /* expected — and no process abort */ }

        do {
            try await context.executeBlitAndWait { _, _ in throw Probe() }
            XCTFail("error must propagate out of executeBlitAndWait")
        } catch is Probe { /* expected */ }
    }
}
