//
//  TreeReductionDispatchRobustnessTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-002: the single-pair kernels in BasicOperations.metal (`euclideanDistance`,
//  `cosineDistance`, `dotProduct`) reduced with `for (stride = tgSize/2; ...; stride /= 2)`,
//  which silently orphans lanes whenever an odd intermediate width is halved (tgSize 100 →
//  50 → 25 → 12 drops lane 24, and so on): the orphaned partial sums never merge, so the
//  kernel returns an under-count — wrong results, not a crash. Live path: `Metal4ComputeEngine`
//  dispatches `cosineDistance`/`dotProduct` with `threadsPerGroup = min(256, dimension)` in a
//  single threadgroup, and its `decisionEngine` defaults to nil, where the hardcoded fallback
//  routes any dimension > 16 to the GPU — so every non-power-of-two dimension in 17...255
//  returned a wrong value to direct-engine users. `euclideanDistance` is dispatched with
//  width 1 (serially correct by accident — the VA3-024 perf item), so its leg here drives the
//  kernel directly with non-pow2 threadgroup widths the engine cannot produce.
//
//  The same kernels also wrote their `threadgroup float …[256]` arrays at `[tid]` with no
//  clamp, an OOB threadgroup write for any dispatch wider than 256 — covered by the >256
//  widths in the kernel-direct sweep.
//
//  The fix is the file's own `va_tg_reduce_add` contract: lanes clamped to 256, lanes-guarded
//  accumulation, fixed power-of-two starting stride with the `lane + stride < lanes`
//  ragged-tail guard, uniform barriers. Post-fix every width in these sweeps must agree with
//  `AccelerateFallback`; the pow2 widths are controls the broken tree already handled.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class TreeReductionDispatchRobustnessTests: XCTestCase {

    /// Non-pow2 dims in 17...255 hit the live engine defect (GPU-routed, tgSize = dimension).
    /// Pow2 dims are controls the broken tree handled; dims > 256 clamp the threadgroup to
    /// 256 lanes (pow2) and were also accidentally correct.
    private static let sweepDims = [17, 20, 100, 200, 255, 32, 64, 128, 256, 300, 1000]

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Semantic float agreement (same contract as DifferentialKernelVsCPUTests): both NaN,
    /// bitwise equal, or within mixed absolute/relative tolerance.
    private static func agree(_ a: Float, _ b: Float, relTol: Float = 2e-4, absTol: Float = 1e-5) -> Bool {
        if a.isNaN && b.isNaN { return true }
        if a == b { return true }
        if a.isNaN != b.isNaN || a.isInfinite != b.isInfinite { return false }
        return abs(a - b) <= max(absTol, relTol * max(abs(a), abs(b)))
    }

    // MARK: - Engine-level sweeps (the live path: decisionEngine nil → GPU for dimension > 16)

    func testEngineDotProductAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)   // decisionEngine: nil
        var rng = TestRNG(seed: 0x3A02_0001)
        var failures: [String] = []

        for dim in Self.sweepDims {
            // Structural leg: Σ 1·1 = dim exactly (each partial sum is a small integer, so an
            // orphaned lane shows up as an exact integer under-count).
            let ones = [Float](repeating: 1, count: dim)
            let structural = try await engine.dotProduct(ones, ones)
            if structural != Float(dim) {
                failures.append("dim=\(dim) ones: gpu=\(structural) expected=\(Float(dim))")
            }

            // Differential leg vs the provider's CPU fallback.
            let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let gpu = try await engine.dotProduct(a, b)
            let cpu = try AccelerateFallback.dotProduct(a, b)
            if !Self.agree(gpu, cpu) {
                failures.append("dim=\(dim) random: gpu=\(gpu) cpu=\(cpu)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging dotProduct dims:\n" + failures.joined(separator: "\n"))
    }

    func testEngineCosineDistanceAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)   // decisionEngine: nil
        var rng = TestRNG(seed: 0x3A02_0002)
        var failures: [String] = []

        for dim in Self.sweepDims {
            // No structural all-ones leg here: orphaning truncates dot/aa/bb through the same
            // tree, and cosine of identical vectors is scale-invariant — only data whose
            // coordinates disagree can expose the under-count.
            let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let gpu = try await engine.cosineDistance(a, b)
            let cpu = 1.0 - (try AccelerateFallback.cosineSimilarity(a, b))
            if !Self.agree(gpu, cpu) {
                failures.append("dim=\(dim): gpu=\(gpu) cpu=\(cpu)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging cosineDistance dims:\n" + failures.joined(separator: "\n"))
    }

    // MARK: - Kernel-direct sweeps (the engine's dispatch cannot produce these geometries:
    // euclidean is dispatched width-1, and cosine/dot are capped at min(256, dimension), so
    // the >256 lanes clamp of all three kernels is reachable only from here)

    /// 17/100/255/384: non-pow2 halvings orphaned lanes pre-fix; 384/512 also exceed the
    /// 256-float shared arrays (OOB write pre-fix, lanes-clamp post-fix); 64/256 are pow2
    /// controls.
    private static let sweepWidths = [17, 100, 255, 384, 512, 64, 256]

    /// The `where` clamp must never silently drop the >256 legs — they are the only
    /// coverage the lanes clamp has.
    private func allowedWidths(_ pipeline: any MTLComputePipelineState, _ label: String) -> [Int] {
        let allowed = Self.sweepWidths.filter { $0 <= pipeline.maxTotalThreadsPerThreadgroup }
        XCTAssertTrue(allowed.contains { $0 > 256 },
            "\(label): width sweep lost its >256 clamp legs (maxTotalThreadsPerThreadgroup = \(pipeline.maxTotalThreadsPerThreadgroup))")
        return allowed
    }

    func testEuclideanDistanceKernelAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let pipeline = try await context.getPipeline(functionName: "euclideanDistance")
        var rng = TestRNG(seed: 0x3A02_0003)
        var failures: [String] = []
        let dim = 1000

        let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let cpu = try AccelerateFallback.euclideanDistance(a, b)
        let ones = [Float](repeating: 1, count: dim)
        let zeros = [Float](repeating: 0, count: dim)
        let expectedStructural = sqrt(Float(dim))   // ‖1 − 0‖₂: any orphaned lane under-counts

        for tgWidth in allowedWidths(pipeline, "euclideanDistance") {
            let structural = try await runPairKernel(
                pipeline, ones, zeros, tgWidth: tgWidth, context: context)
            if !Self.agree(structural, expectedStructural, relTol: 1e-4) {
                failures.append("tg=\(tgWidth) ones: gpu=\(structural) expected=\(expectedStructural)")
            }

            let gpu = try await runPairKernel(pipeline, a, b, tgWidth: tgWidth, context: context)
            if !Self.agree(gpu, cpu) {
                failures.append("tg=\(tgWidth) random: gpu=\(gpu) cpu=\(cpu)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging euclideanDistance widths:\n" + failures.joined(separator: "\n"))
    }

    /// AUDIT-3 meta-review: the cosine/dot halves of VA3-002 were previously guarded only
    /// through the engine's routing (GPU iff dimension > 16 with a nil decision engine) —
    /// raising that threshold, the ledger's own recommended VA3-024 follow-up, would have
    /// silently turned those sweeps into CPU-vs-CPU comparisons. These legs bind the same
    /// kernels routing-independently, and are the only coverage of their >256 lanes clamp.
    func testCosineAndDotProductKernelsAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        var rng = TestRNG(seed: 0x3A02_0004)
        var failures: [String] = []
        let dim = 1000

        let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let ones = [Float](repeating: 1, count: dim)
        let cpuDot = try AccelerateFallback.dotProduct(a, b)
        let cpuCos = 1.0 - (try AccelerateFallback.cosineSimilarity(a, b))

        let dotPipeline = try await context.getPipeline(functionName: "dotProduct")
        for tgWidth in allowedWidths(dotPipeline, "dotProduct") {
            // Structural: Σ 1·1 = dim exactly (orphaned or unclamped lanes break the integer).
            let structural = try await runPairKernel(
                dotPipeline, ones, ones, tgWidth: tgWidth, context: context)
            if structural != Float(dim) {
                failures.append("dot tg=\(tgWidth) ones: gpu=\(structural) expected=\(Float(dim))")
            }
            let gpu = try await runPairKernel(dotPipeline, a, b, tgWidth: tgWidth, context: context)
            if !Self.agree(gpu, cpuDot) {
                failures.append("dot tg=\(tgWidth) random: gpu=\(gpu) cpu=\(cpuDot)")
            }
        }

        let cosPipeline = try await context.getPipeline(functionName: "cosineDistance")
        for tgWidth in allowedWidths(cosPipeline, "cosineDistance") {
            // Identical vectors: similarity 1 within finalize rounding → distance ≈ 0.
            let structural = try await runPairKernel(
                cosPipeline, ones, ones, tgWidth: tgWidth, context: context)
            if abs(structural) > 1e-5 {
                failures.append("cos tg=\(tgWidth) ones: gpu=\(structural) expected=0")
            }
            let gpu = try await runPairKernel(cosPipeline, a, b, tgWidth: tgWidth, context: context)
            if !Self.agree(gpu, cpuCos) {
                failures.append("cos tg=\(tgWidth) random: gpu=\(gpu) cpu=\(cpuCos)")
            }
        }

        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging cosine/dot widths:\n" + failures.joined(separator: "\n"))
    }

    /// Dispatch a single-pair kernel (buffers 0/1/2 + dimension at 3) with an explicit
    /// threadgroup width — same buffers/indices as the engine's dispatch, geometry under
    /// test control. The pooled result buffer is poisoned before the dispatch so a kernel
    /// that stops writing result[0] cannot replay a previous dispatch's value (the VA3-031
    /// stale-bytes failure shape).
    private func runPairKernel(
        _ pipeline: any MTLComputePipelineState, _ a: [Float], _ b: [Float], tgWidth: Int,
        context: Metal4Context
    ) async throws -> Float {
        let bufferA = try await context.getBuffer(for: a)
        let bufferB = try await context.getBuffer(for: b)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        resultBuffer.buffer.contents().storeBytes(of: Float(-98765.0), as: Float.self)

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            var dim = UInt32(a.count)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        return resultBuffer.readScalar(as: Float.self)
    }
}
