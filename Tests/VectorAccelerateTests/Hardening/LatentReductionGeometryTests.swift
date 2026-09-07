//
//  LatentReductionGeometryTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-013 + VA3-014: reduction kernels whose correctness was an accident of the
//  host's exact dispatch geometry.
//
//  VA3-013 — `manhattanDistance`/`chebyshevDistance` (DistanceShaders.metal): guard-before-
//  barrier early return, `uint tid = id % 256` with a hardcoded 256 stride, per-threadgroup
//  racing `atomic_store` to result[0], and a thread-0 serial sum over `min(256, dimension)`
//  shared slots regardless of how many threads actually wrote them. All four are wrong in
//  general; `Metal4ComputeEngine`'s dispatch (`min(256, dimension)` threads, exactly one
//  group) happens to make every one of them unobservable — the same accident that hid
//  VA3-001 in `jaccardDistance` one page below.
//
//  VA3-014 — `logsumexp_reduce_pass1/2_kernel` (LogSumExp.metal) and
//  `computeBasicStatistics`/`computeHigherMoments` (StatisticsShaders.metal): tgSize/2-start
//  reduction trees whose `lid + s < tsize` guards prevent the out-of-bounds read but ORPHAN
//  lanes on every odd halving — under a comment literally reading "Robustness check for
//  non-power-of-2 tgSize". The hosts know better: StatisticsKernel forces a power-of-two
//  width ("Use power-of-2 threadgroup size for correct parallel reduction") and
//  LogSumExpKernel dispatches fixed 256/256. Pass 2 additionally starts its tree at a fixed
//  128 with NO tail guard (dispatching fewer than 256 threads reads uninitialized threadgroup
//  memory into a max-reduce) and silently ignores partials beyond lane 255.
//
//  These tests drive the kernels directly with the geometries the hosts never use (red
//  pre-fix), plus the hosts' own geometries and the live engine paths as must-stay-green
//  controls. Post-fix contract: correct for ANY threadgroup width (lanes clamped to the
//  shared-array capacity) and, for pass 2, any numGroups.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class LatentReductionGeometryTests: XCTestCase {

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private static func agree(_ a: Float, _ b: Float, relTol: Float = 2e-4, absTol: Float = 1e-5) -> Bool {
        if a.isNaN && b.isNaN { return true }
        if a == b { return true }
        if a.isNaN != b.isNaN || a.isInfinite != b.isInfinite { return false }
        return abs(a - b) <= max(absTol, relTol * max(abs(a), abs(b)))
    }

    /// Dispatch a single-pair distance kernel (buffers 0/1/2 + dimension at 3) with explicit
    /// geometry. Mirrors the engine's binding layout; geometry is under test control.
    private func runPairKernel(
        _ functionName: String, _ a: [Float], _ b: [Float],
        tgWidth: Int, groups: Int, context: Metal4Context
    ) async throws -> Float {
        let pipeline = try await context.getPipeline(functionName: functionName)
        let bufferA = try await context.getBuffer(for: a)
        let bufferB = try await context.getBuffer(for: b)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            var dim = UInt32(a.count)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        return resultBuffer.readScalar(as: Float.self)
    }

    // MARK: - VA3-013: manhattan/chebyshev kernel-direct geometry sweep

    func testManhattanChebyshevKernelsAcrossGeometries() async throws {
        let context = try await Metal4Context()
        var rng = TestRNG(seed: 0x3A13_0001)
        var failures: [String] = []

        // dim=1000 with tg=100: the hardcoded 256 stride leaves every element with
        // i % 256 >= 100 uncovered AND thread 0 sums min(256, dim) = 256 shared slots of
        // which 156 were never written. dim=1024 with 4×256: every group's thread 0 races
        // its own (mostly wrong) total into result[0]. tg=512: two threads alias each
        // shared slot through `id % 256`. tg=256×1 is the engine's geometry (control).
        for (dim, tgWidth, groups) in [(1000, 100, 1), (1000, 17, 1), (1024, 256, 4),
                                       (1000, 512, 1), (1000, 256, 1)] {
            let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            var manhattanRef: Double = 0
            var chebyshevRef: Double = 0
            for i in 0..<dim {
                let d = abs(Double(a[i]) - Double(b[i]))
                manhattanRef += d
                chebyshevRef = max(chebyshevRef, d)
            }

            let mGPU = try await runPairKernel(
                "manhattanDistance", a, b, tgWidth: tgWidth, groups: groups, context: context)
            if !Self.agree(mGPU, Float(manhattanRef)) {
                failures.append("manhattan dim=\(dim) tg=\(tgWidth)x\(groups): gpu=\(mGPU) ref=\(Float(manhattanRef))")
            }

            // Chebyshev spike at index 255 (255 % 256 >= any sub-256 width → uncovered
            // pre-fix) on an otherwise small-magnitude pair.
            var aSpike = a.map { $0 * 0.125 }
            aSpike[255] = 7.5
            var bSpike = b.map { $0 * 0.125 }
            bSpike[255] = 0
            var spikeRef: Double = 0
            for i in 0..<dim { spikeRef = max(spikeRef, abs(Double(aSpike[i]) - Double(bSpike[i]))) }

            let cGPU = try await runPairKernel(
                "chebyshevDistance", aSpike, bSpike, tgWidth: tgWidth, groups: groups, context: context)
            if !Self.agree(cGPU, Float(spikeRef)) {
                failures.append("chebyshev dim=\(dim) tg=\(tgWidth)x\(groups): gpu=\(cGPU) ref=\(Float(spikeRef))")
            }

            // Jaccard shares the identical one-group contract and buffer layout (rewritten
            // in slice 1) but had no kernel-direct geometry coverage of its own —
            // meta-review symmetry leg. Non-negative data keeps the metric meaningful.
            let aPos = a.map { abs($0) }
            let bPos = b.map { abs($0) }
            var minSum = 0.0
            var maxSum = 0.0
            for i in 0..<dim {
                minSum += Double(min(aPos[i], bPos[i]))
                maxSum += Double(max(aPos[i], bPos[i]))
            }
            let jaccardRef: Float = maxSum > 0 ? Float(1.0 - minSum / maxSum) : 1.0
            let jGPU = try await runPairKernel(
                "jaccardDistance", aPos, bPos, tgWidth: tgWidth, groups: groups, context: context)
            if !Self.agree(jGPU, jaccardRef) {
                failures.append("jaccard dim=\(dim) tg=\(tgWidth)x\(groups): gpu=\(jGPU) ref=\(jaccardRef)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging manhattan/chebyshev geometries:\n" + failures.joined(separator: "\n"))
    }

    // MARK: - VA3-013: live engine paths (shielded today — must stay green through the rewrite)

    func testEngineManhattanChebyshevParityAcrossDimensions() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)   // decisionEngine: nil
        var rng = TestRNG(seed: 0x3A13_0002)
        var failures: [String] = []

        // GPU-routed for dimension > 64 (nil decision engine, fallbackThreshold 64).
        for dim in [65, 100, 128, 200, 255, 256, 300, 1000] {
            let a = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            let b = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
            var manhattanRef: Double = 0
            var chebyshevRef: Double = 0
            for i in 0..<dim {
                let d = abs(Double(a[i]) - Double(b[i]))
                manhattanRef += d
                chebyshevRef = max(chebyshevRef, d)
            }

            let m = try await engine.manhattanDistance(a, b)
            if !Self.agree(m, Float(manhattanRef)) {
                failures.append("manhattan dim=\(dim): gpu=\(m) ref=\(Float(manhattanRef))")
            }
            let c = try await engine.chebyshevDistance(a, b)
            if !Self.agree(c, Float(chebyshevRef)) {
                failures.append("chebyshev dim=\(dim): gpu=\(c) ref=\(Float(chebyshevRef))")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging engine dims:\n" + failures.joined(separator: "\n"))
    }

    // MARK: - VA3-014: logsumexp pass 1 (non-pow2 threadgroup orphans the max/sum trees)

    func testLogSumExpPass1KernelAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let pipeline = try await context.getPipeline(functionName: "logsumexp_reduce_pass1_kernel")
        let device = context.device.rawDevice
        var failures: [String] = []

        // input[i] = i · 0.1 — the maximum (9.9) lives at lane 99, whose merged subtree the
        // 100 → 50 → 25 → 12 halving orphans at lane 24. Max-reduce of exact values must be
        // exact, so partialMax is compared bitwise.
        let count = 100
        let input = (0..<count).map { Float($0) * 0.1 }
        let trueMax = input.max()!
        var sumExpRef: Double = 0   // Σ exp(x − max), the pass-1 contract
        for x in input { sumExpRef += exp(Double(x) - Double(trueMax)) }

        let inputBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: input, length: count * MemoryLayout<Float>.size, options: .storageModeShared))

        // 100/17: non-pow2 orphaning. 300: exceeds the 256-float shared arrays (unclamped
        // writes pre-fix). 128/256: pow2 controls (256 is the host's fixed width).
        for tsize in [100, 17, 300, 128, 256] where tsize <= pipeline.maxTotalThreadsPerThreadgroup {
            let partialMaxBuffer = try XCTUnwrap(device.makeBuffer(
                length: MemoryLayout<Float>.size, options: .storageModeShared))
            let partialSumBuffer = try XCTUnwrap(device.makeBuffer(
                length: MemoryLayout<Float>.size, options: .storageModeShared))

            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(partialMaxBuffer, offset: 0, index: 1)
                encoder.setBuffer(partialSumBuffer, offset: 0, index: 2)
                var countU32 = UInt32(count)
                var numGroupsU32 = UInt32(1)
                encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tsize, height: 1, depth: 1))
            }

            let gotMax = partialMaxBuffer.contents().load(as: Float.self)
            let gotSum = partialSumBuffer.contents().load(as: Float.self)
            if gotMax != trueMax {
                failures.append("tsize=\(tsize): partialMax=\(gotMax) expected=\(trueMax)")
            }
            if !Self.agree(gotSum, Float(sumExpRef), relTol: 1e-3) {
                failures.append("tsize=\(tsize): partialSumExp=\(gotSum) expected=\(Float(sumExpRef))")
            }

            // Structural all-equal leg (meta-review): exp(1 − 1) is exactly 1, so
            // partialSumExp must be exactly Float(count) — an orphaned lane is an integer
            // deficit that the random leg's 1e-3 tolerance could have absorbed for a
            // small-valued element.
            let ones = [Float](repeating: 1, count: count)
            let onesBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: ones, length: count * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(onesBuffer, offset: 0, index: 0)
                encoder.setBuffer(partialMaxBuffer, offset: 0, index: 1)
                encoder.setBuffer(partialSumBuffer, offset: 0, index: 2)
                var countU32 = UInt32(count)
                var numGroupsU32 = UInt32(1)
                encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tsize, height: 1, depth: 1))
            }
            let onesMax = partialMaxBuffer.contents().load(as: Float.self)
            let onesSum = partialSumBuffer.contents().load(as: Float.self)
            if onesMax != 1.0 {
                failures.append("tsize=\(tsize) ones: partialMax=\(onesMax) expected=1.0")
            }
            if onesSum != Float(count) {
                failures.append("tsize=\(tsize) ones: partialSumExp=\(onesSum) expected=\(Float(count))")
            }
        }

        // Multi-group leg (meta-review Critical): the slice-3 rewrite re-indexed the
        // grid-stride to `base = tgid·lanes + lid` stepping `lanes·numThreadgroups`, and no
        // test asserted values at numThreadgroups > 1 — let alone at a non-256 width where
        // the re-indexing actually differs from the original. Combine the per-group partials
        // exactly as pass 2 does and compare against a Double reference: any element missed
        // or double-counted by the cross-group partition shows up here.
        let bigCount = 1000
        var bigRng = TestRNG(seed: 0x3A14_0002)
        let bigInput = (0..<bigCount).map { _ in bigRng.nextFloat(in: -8...8) }
        var bigRef: Double = 0
        let bigTrueMax = bigInput.max()!
        for x in bigInput { bigRef += exp(Double(x) - Double(bigTrueMax)) }
        let bigLSERef = Float(log(bigRef) + Double(bigTrueMax))

        let bigBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: bigInput, length: bigCount * MemoryLayout<Float>.size, options: .storageModeShared))
        for (tsize, groups) in [(100, 3), (256, 4), (64, 5)] {
            let pmBuffer = try XCTUnwrap(device.makeBuffer(
                length: groups * MemoryLayout<Float>.size, options: .storageModeShared))
            let psBuffer = try XCTUnwrap(device.makeBuffer(
                length: groups * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bigBuffer, offset: 0, index: 0)
                encoder.setBuffer(pmBuffer, offset: 0, index: 1)
                encoder.setBuffer(psBuffer, offset: 0, index: 2)
                var countU32 = UInt32(bigCount)
                var numGroupsU32 = UInt32(groups)
                encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tsize, height: 1, depth: 1))
            }
            let pm = Array(UnsafeBufferPointer(
                start: pmBuffer.contents().bindMemory(to: Float.self, capacity: groups), count: groups))
            let ps = Array(UnsafeBufferPointer(
                start: psBuffer.contents().bindMemory(to: Float.self, capacity: groups), count: groups))
            let globalMax = pm.max()!
            var combined: Double = 0
            for g in 0..<groups where pm[g] > -.infinity {
                combined += Double(ps[g]) * exp(Double(pm[g]) - Double(globalMax))
            }
            let lse = Float(log(combined) + Double(globalMax))
            if !Self.agree(lse, bigLSERef, relTol: 1e-3) {
                failures.append("multi-group tsize=\(tsize)x\(groups): lse=\(lse) expected=\(bigLSERef) partials=\(pm)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging pass-1 widths:\n" + failures.joined(separator: "\n"))
    }

    // MARK: - VA3-014: logsumexp pass 2 (guardless fixed-128 tree; partials beyond lane 255 dropped)

    func testLogSumExpPass2KernelSmallThreadgroupAndManyGroups() async throws {
        let context = try await Metal4Context()
        let pipeline = try await context.getPipeline(functionName: "logsumexp_reduce_pass2_kernel")
        let device = context.device.rawDevice
        var failures: [String] = []

        func runPass2(partialMax: [Float], partialSum: [Float], tsize: Int) async throws -> Float {
            let n = partialMax.count
            let maxBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: partialMax, length: n * MemoryLayout<Float>.size, options: .storageModeShared))
            let sumBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: partialSum, length: n * MemoryLayout<Float>.size, options: .storageModeShared))
            let outBuffer = try XCTUnwrap(device.makeBuffer(
                length: MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(maxBuffer, offset: 0, index: 0)
                encoder.setBuffer(sumBuffer, offset: 0, index: 1)
                encoder.setBuffer(outBuffer, offset: 0, index: 2)
                var numGroups = UInt32(n)
                encoder.setBytes(&numGroups, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tsize, height: 1, depth: 1))
            }
            return outBuffer.contents().load(as: Float.self)
        }

        func referenceLSE(partialMax: [Float], partialSum: [Float]) -> Float {
            let gm = partialMax.max()!
            var s: Double = 0
            for i in 0..<partialMax.count where partialMax[i] > -.infinity {
                s += Double(partialSum[i]) * exp(Double(partialMax[i]) - Double(gm))
            }
            return Float(log(s) + Double(gm))
        }

        // Leg A — sub-256 threadgroups, 4 groups, all partials near −100: the pre-fix
        // fixed-128 tree read sharedMax[tsize..191], which no thread wrote; any garbage
        // ≳ −87 makes every exp(partialMax − globalMax) underflow to 0 → output −inf
        // instead of ≈ −99.3. Widths 17/100 are non-pow2 (meta-review: the ragged-tail
        // guard in the pass-2 trees never fired at the original 64/256-only widths).
        let maxA = (0..<4).map { Float(-100 - $0) }
        let sumA = [Float](repeating: 1.0, count: 4)
        let refA = referenceLSE(partialMax: maxA, partialSum: sumA)
        for tsize in [64, 17, 100] {
            let gotA = try await runPass2(partialMax: maxA, partialSum: sumA, tsize: tsize)
            if !Self.agree(gotA, refA, relTol: 1e-3) {
                failures.append("tsize=\(tsize) numGroups=4: got=\(gotA) expected=\(refA)")
            }
        }

        // Leg B — 300 partials, 256 threads: the dominant partial sits at index 299, beyond
        // the highest lane, and is silently dropped pre-fix.
        var maxB = [Float](repeating: -100, count: 300)
        maxB[299] = 50
        let sumB = [Float](repeating: 1.0, count: 300)
        let refB = referenceLSE(partialMax: maxB, partialSum: sumB)
        let gotB = try await runPass2(partialMax: maxB, partialSum: sumB, tsize: 256)
        if !Self.agree(gotB, refB, relTol: 1e-3) {
            failures.append("tsize=256 numGroups=300: got=\(gotB) expected=\(refB)")
        }

        // Control — the host's exact geometry (256 threads, numGroups ≤ 256).
        let refC = referenceLSE(partialMax: maxA, partialSum: sumA)
        let gotC = try await runPass2(partialMax: maxA, partialSum: sumA, tsize: 256)
        if !Self.agree(gotC, refC, relTol: 1e-3) {
            failures.append("control tsize=256 numGroups=4: got=\(gotC) expected=\(refC)")
        }

        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging pass-2 legs:\n" + failures.joined(separator: "\n"))
    }

    // MARK: - VA3-014: statistics kernels (orphaning tree under a comment claiming robustness)

    func testBasicStatisticsKernelAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let pipeline = try await context.getPipeline(functionName: "computeBasicStatistics")
        let device = context.device.rawDevice
        var failures: [String] = []

        // input[i] = i: with tgSize = dimension = 100 every lane holds exactly one element,
        // so orphaned lanes show up as an exact count deficit (and 99 — lane 99's subtree —
        // vanishes from max).
        let dim = 100
        let input = (0..<dim).map { Float($0) }
        let trueSum = Float((0..<dim).reduce(0, +))
        let trueMean = trueSum / Float(dim)

        let inputBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: input, length: dim * MemoryLayout<Float>.size, options: .storageModeShared))

        for tgSize in [100, 17, 128, 256] {
            let outputBuffer = try XCTUnwrap(device.makeBuffer(
                length: 6 * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                var params = SIMD2<UInt32>(UInt32(dim), 0)
                encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            let out = Array(UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: 6), count: 6))
            // [mean, M2, min, max, sum, count]
            if !Self.agree(out[0], trueMean, relTol: 1e-3) {
                failures.append("tg=\(tgSize) mean: got=\(out[0]) expected=\(trueMean)")
            }
            if out[2] != 0 { failures.append("tg=\(tgSize) min: got=\(out[2]) expected=0") }
            if out[3] != Float(dim - 1) {
                failures.append("tg=\(tgSize) max: got=\(out[3]) expected=\(Float(dim - 1))")
            }
            if !Self.agree(out[4], trueSum, relTol: 1e-4) {
                failures.append("tg=\(tgSize) sum: got=\(out[4]) expected=\(trueSum)")
            }
            if out[5] != Float(dim) {
                failures.append("tg=\(tgSize) count: got=\(out[5]) expected=\(Float(dim))")
            }
        }

        // Widths beyond 256 (meta-review): the kernel claims "correct for ANY tgSize up to
        // MAX_TG_SIZE", but the fixed-512/256 starting strides of the rewritten tree only
        // execute for tgSize > 256 — untested by the sweep above. dim = 2048 puts real data
        // in every lane; the sum is an exact FP32 integer (< 2²⁴) at every merge, so an
        // orphaned or garbage-folded lane cannot hide in tolerance.
        let bigDim = 2048
        let bigInput = (0..<bigDim).map { Float($0) }
        let bigSum = Float(bigDim * (bigDim - 1) / 2)   // 2 096 128 < 2²⁴, exact
        let bigBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: bigInput, length: bigDim * MemoryLayout<Float>.size, options: .storageModeShared))

        let maxWidth = pipeline.maxTotalThreadsPerThreadgroup
        let bigWidths = [300, 1000, 1024].filter { $0 <= maxWidth }
        XCTAssertTrue(bigWidths.contains { $0 > 512 },
            "statistics width sweep cannot reach the 512-stride tree (maxTotalThreadsPerThreadgroup = \(maxWidth))")
        for tgSize in bigWidths {
            let outputBuffer = try XCTUnwrap(device.makeBuffer(
                length: 6 * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bigBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                var params = SIMD2<UInt32>(UInt32(bigDim), 0)
                encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            let out = Array(UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: 6), count: 6))
            if !Self.agree(out[0], bigSum / Float(bigDim), relTol: 1e-3) {
                failures.append("tg=\(tgSize) big mean: got=\(out[0]) expected=\(bigSum / Float(bigDim))")
            }
            if out[2] != 0 { failures.append("tg=\(tgSize) big min: got=\(out[2]) expected=0") }
            if out[3] != Float(bigDim - 1) {
                failures.append("tg=\(tgSize) big max: got=\(out[3]) expected=\(Float(bigDim - 1))")
            }
            if out[4] != bigSum {
                failures.append("tg=\(tgSize) big sum: got=\(out[4]) expected=\(bigSum)")
            }
            if out[5] != Float(bigDim) {
                failures.append("tg=\(tgSize) big count: got=\(out[5]) expected=\(Float(bigDim))")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging basic-statistics cells:\n" + failures.joined(separator: "\n"))
    }

    func testHigherMomentsKernelAcrossThreadgroupWidths() async throws {
        let context = try await Metal4Context()
        let pipeline = try await context.getPipeline(functionName: "computeHigherMoments")
        let device = context.device.rawDevice
        var failures: [String] = []

        // mean = 0 with input[i] = i: M3 = Σ i³ = 24 502 500 < 2²⁵ — every term and every
        // partial sum is an exact FP32 integer, so M3 admits exact comparison; an orphaned
        // lane's missing cube is enormous. M4 terms exceed 2²⁵ and get a tolerance.
        struct MomentParams { var n: UInt32; var mean: Float }
        let dim = 100
        let input = (0..<dim).map { Float($0) }
        var m3Ref: Double = 0
        var m4Ref: Double = 0
        for i in 0..<dim {
            let d = Double(i)
            m3Ref += d * d * d
            m4Ref += d * d * d * d
        }

        let inputBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: input, length: dim * MemoryLayout<Float>.size, options: .storageModeShared))

        for tgSize in [100, 17, 128, 256] {
            let outputBuffer = try XCTUnwrap(device.makeBuffer(
                length: 2 * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                var params = MomentParams(n: UInt32(dim), mean: 0)
                encoder.setBytes(&params, length: MemoryLayout<MomentParams>.size, index: 2)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            let m3 = outputBuffer.contents().load(as: Float.self)
            let m4 = outputBuffer.contents().load(fromByteOffset: 4, as: Float.self)
            if m3 != Float(m3Ref) {
                failures.append("tg=\(tgSize) M3: got=\(m3) expected=\(Float(m3Ref))")
            }
            if !Self.agree(m4, Float(m4Ref), relTol: 1e-4) {
                failures.append("tg=\(tgSize) M4: got=\(m4) expected=\(Float(m4Ref))")
            }
        }

        // Widths beyond 256 (meta-review): exercise the fixed-512/256 starting strides with
        // real data in every lane. FP32 rounds the big cubes/quartics, so these legs use a
        // relative tolerance — a dropped 512-lane block is a ~25-50% error, far outside it.
        let bigDim = 2048
        let bigInput = (0..<bigDim).map { Float($0) }
        var bigM3: Double = 0
        var bigM4: Double = 0
        for i in 0..<bigDim {
            let d = Double(i)
            bigM3 += d * d * d
            bigM4 += d * d * d * d
        }
        let bigBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: bigInput, length: bigDim * MemoryLayout<Float>.size, options: .storageModeShared))

        let maxWidth = pipeline.maxTotalThreadsPerThreadgroup
        let bigWidths = [300, 1000, 1024].filter { $0 <= maxWidth }
        XCTAssertTrue(bigWidths.contains { $0 > 512 },
            "moments width sweep cannot reach the 512-stride tree (maxTotalThreadsPerThreadgroup = \(maxWidth))")
        for tgSize in bigWidths {
            let outputBuffer = try XCTUnwrap(device.makeBuffer(
                length: 2 * MemoryLayout<Float>.size, options: .storageModeShared))
            try await context.executeAndWait { _, encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bigBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                var params = MomentParams(n: UInt32(bigDim), mean: 0)
                encoder.setBytes(&params, length: MemoryLayout<MomentParams>.size, index: 2)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            let m3 = outputBuffer.contents().load(as: Float.self)
            let m4 = outputBuffer.contents().load(fromByteOffset: 4, as: Float.self)
            if !Self.agree(m3, Float(bigM3), relTol: 1e-3) {
                failures.append("tg=\(tgSize) big M3: got=\(m3) expected=\(Float(bigM3))")
            }
            if !Self.agree(m4, Float(bigM4), relTol: 1e-3) {
                failures.append("tg=\(tgSize) big M4: got=\(m4) expected=\(Float(bigM4))")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging higher-moments cells:\n" + failures.joined(separator: "\n"))
    }
}
