//
//  DifferentialKernelVsCPUTests.swift
//  VectorAccelerateTests
//
//  Phase-1.1 hardening audit: differential GPU↔CPU testing with adversarial inputs.
//
//  The provider's silent-fallback design means a GPU kernel and its CPU fallback MUST be
//  observably interchangeable — any divergence is a correctness bug that fallback masks rather
//  than surfaces. These tests drive the GPU legs kernel-direct (no decision engine, no fallback:
//  errors throw), the CPU legs through the same `AccelerateFallback` routines the provider uses,
//  and carry an independent Double-accumulation oracle in every failure message so blame is
//  attributable (GPU wrong vs CPU wrong vs both).
//
//  Comparison semantics ("agree"): both NaN, or bitwise equal (covers ±Inf with matching sign),
//  or within mixed absolute/relative tolerance. Top-K comparisons allow index differences only
//  between distance-equivalent candidates (tie-order is intentionally NOT pinned here — the two
//  findNearest paths use different tie-break rules; see AUDIT-2 VA2-007).
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class DifferentialKernelVsCPUTests: XCTestCase {

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    // MARK: - Fixtures

    enum ValueClass: String, CaseIterable {
        case normal        // U(-1, 1)
        case zeros         // all zero (degenerate norms)
        case duplicates    // every row identical (tie stress)
        case huge          // ±1e19 (squared terms near/over FLT_MAX)
        case tiny          // ±1e-20 (squared terms subnormal/underflowing)
        case subnormal     // 1e-40 everywhere (below Float.leastNormalMagnitude)
        case mixedScale    // alternating ±1e18 / ±1e-18 (cancellation + dynamic range)
        case nanPoisoned   // one NaN component in one row
        case posInfPoisoned
        case negInfPoisoned
    }

    static func makeMatrix(_ cls: ValueClass, n: Int, dim: Int, rng: inout TestRNG) -> [[Float]] {
        func randRow() -> [Float] { (0..<dim).map { _ in rng.nextFloat(in: -1...1) } }
        switch cls {
        case .normal:     return (0..<n).map { _ in randRow() }
        case .zeros:      return Array(repeating: [Float](repeating: 0, count: dim), count: n)
        case .duplicates: let row = randRow(); return Array(repeating: row, count: n)
        case .huge:       return (0..<n).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) * 1e19 } }
        case .tiny:       return (0..<n).map { _ in (0..<dim).map { _ in rng.nextFloat(in: -1...1) * 1e-20 } }
        case .subnormal:  return Array(repeating: [Float](repeating: 1e-40, count: dim), count: n)
        case .mixedScale:
            return (0..<n).map { i in
                (0..<dim).map { j in
                    (i + j).isMultiple(of: 2) ? rng.nextFloat(in: -1...1) * 1e18
                                              : rng.nextFloat(in: -1...1) * 1e-18
                }
            }
        case .nanPoisoned:
            var m = (0..<n).map { _ in randRow() }; m[n / 2][dim / 2] = .nan; return m
        case .posInfPoisoned:
            var m = (0..<n).map { _ in randRow() }; m[n / 2][dim / 2] = .infinity; return m
        case .negInfPoisoned:
            var m = (0..<n).map { _ in randRow() }; m[n / 2][dim / 2] = -.infinity; return m
        }
    }

    /// Query row for a class: same distribution as the candidates, except classes that would
    /// make the comparison degenerate (all-zero query against all-zero candidates).
    static func makeQuery(_ cls: ValueClass, dim: Int, rng: inout TestRNG) -> [Float] {
        let effective: ValueClass = (cls == .zeros || cls == .duplicates) ? .normal : cls
        var row = makeMatrix(effective, n: 1, dim: dim, rng: &rng)[0]
        // Poison classes: the poison lives in the candidates; keep the query finite.
        if cls == .nanPoisoned || cls == .posInfPoisoned || cls == .negInfPoisoned {
            row = makeMatrix(.normal, n: 1, dim: dim, rng: &rng)[0]
        }
        return row
    }

    // MARK: - Oracle + comparator

    /// Double-accumulation Euclidean distance (independent of both legs under test).
    static func oracleL2(_ q: [Float], _ c: [Float]) -> Float {
        var s = 0.0
        for i in 0..<q.count { let d = Double(q[i]) - Double(c[i]); s += d * d }
        return Float(s.squareRoot())
    }

    static func oracleCosineDistance(_ q: [Float], _ c: [Float]) -> Float {
        var dot = 0.0, nq = 0.0, nc = 0.0
        for i in 0..<q.count {
            dot += Double(q[i]) * Double(c[i])
            nq += Double(q[i]) * Double(q[i])
            nc += Double(c[i]) * Double(c[i])
        }
        return Float(1.0 - dot / (nq.squareRoot() * nc.squareRoot()))
    }

    /// Semantic float agreement: both NaN, bitwise equal (covers ±Inf same sign), or within
    /// mixed absolute/relative tolerance.
    static func agree(_ a: Float, _ b: Float, relTol: Float = 2e-4, absTol: Float = 1e-5) -> Bool {
        if a.isNaN && b.isNaN { return true }
        if a == b { return true }
        if a.isNaN != b.isNaN || a.isInfinite != b.isInfinite { return false }
        return abs(a - b) <= max(absTol, relTol * max(abs(a), abs(b)))
    }

    private static let batchDims = [1, 3, 7, 16, 33, 128, 384, 768]
    private static let batchNs = [1, 2, 33, 257, 1000]

    // MARK: - Batch distance differentials

    func testL2BatchGPUMatchesCPUFallback() async throws {
        let context = try await Metal4Context()
        let gpu = try await L2KernelDistanceProvider(context: context)
        var rng = TestRNG(seed: 0xD1FF_0001)
        var failures: [String] = []

        for dim in Self.batchDims {
            for n in Self.batchNs {
                for cls in ValueClass.allCases {
                    let cands = Self.makeMatrix(cls, n: n, dim: dim, rng: &rng)
                    let query = Self.makeQuery(cls, dim: dim, rng: &rng)
                    let g = try await gpu.batchDistance(
                        from: DynamicVector(query), to: cands.map(DynamicVector.init), metric: .euclidean)
                    let c = AccelerateFallback.batchEuclideanDistance(query: query, candidates: cands)
                    XCTAssertEqual(g.count, n)
                    for j in 0..<n where !Self.agree(g[j], c[j]) {
                        failures.append("dim=\(dim) n=\(n) \(cls.rawValue) j=\(j): gpu=\(g[j]) cpu=\(c[j]) oracle=\(Self.oracleL2(query, cands[j]))")
                        break   // one diagnostic row per cell keeps output readable
                    }
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) diverging L2 cells:\n" + failures.prefix(40).joined(separator: "\n"))
    }

    /// VA2-008/VA2-009 are fixed: the GPU cosine kernels carry the pre-scaled overflow/underflow
    /// rescue (`va_cosine_rescaled_terms`, Metal4Common.h) and `AccelerateFallback` mirrors it in
    /// Double with the same NaN-propagation policy — every value class is now a hard assertion.
    func testCosineBatchGPUMatchesCPUFallback() async throws {
        let context = try await Metal4Context()
        let gpu = try await CosineKernelDistanceProvider(context: context)
        var rng = TestRNG(seed: 0xD1FF_0002)
        var failures: [String] = []

        for dim in Self.batchDims {
            for n in Self.batchNs {
                for cls in ValueClass.allCases {
                    let cands = Self.makeMatrix(cls, n: n, dim: dim, rng: &rng)
                    let query = Self.makeQuery(cls, dim: dim, rng: &rng)
                    let g = try await gpu.batchDistance(
                        from: DynamicVector(query), to: cands.map(DynamicVector.init), metric: .cosine)
                    let c = AccelerateFallback.batchCosineSimilarity(query: query, candidates: cands)
                        .map { 1.0 - $0 }
                    XCTAssertEqual(g.count, n)
                    for j in 0..<n where !Self.agree(g[j], c[j]) {
                        failures.append("dim=\(dim) n=\(n) \(cls.rawValue) j=\(j): gpu=\(g[j]) cpu=\(c[j]) oracle=\(Self.oracleCosineDistance(query, cands[j]))")
                        break
                    }
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) diverging cosine cells:\n" + failures.prefix(40).joined(separator: "\n"))
    }

    /// AUDIT-3 VA3-006: `BatchDistanceOperations.batchCosineSimilarity` dispatches the
    /// `batchCosineSimilarity` kernel (BasicOperations.metal) — a live cosine entry point the
    /// VA2-008 remediation slice missed. Pre-fix it kept naive accumulators and the
    /// reassociation-prone `sqrt(aa)·sqrt(bb)` product denominator, so the huge/tiny/mixed
    /// classes below collapsed to 0 on the GPU leg while the CPU leg answered correctly.
    /// The kernel now finishes through the shared rescue trio like every other cosine kernel.
    func testBatchCosineSimilarityOperationsGPUMatchesCPU() async throws {
        let context = try await Metal4Context()
        let ops = try await BatchDistanceEngine(context: context)
        var rng = TestRNG(seed: 0xD1FF_0006)
        var failures: [String] = []

        for dim in Self.batchDims {
            for n in Self.batchNs {
                for cls in ValueClass.allCases {
                    let cands = Self.makeMatrix(cls, n: n, dim: dim, rng: &rng)
                    let query = Self.makeQuery(cls, dim: dim, rng: &rng)
                    let g = try await ops.batchCosineSimilarity(
                        query: query, candidates: cands, useGPU: true)
                    let c = AccelerateFallback.batchCosineSimilarity(query: query, candidates: cands)
                    XCTAssertEqual(g.count, n)
                    for j in 0..<n where !Self.agree(g[j], c[j]) {
                        failures.append("dim=\(dim) n=\(n) \(cls.rawValue) j=\(j): gpu=\(g[j]) cpu=\(c[j]) oracle=\(1.0 - Self.oracleCosineDistance(query, cands[j]))")
                        break
                    }
                }
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "\(failures.count) diverging batchCosineSimilarity cells:\n" + failures.prefix(40).joined(separator: "\n"))
    }

    // MARK: - Fused top-K differential

    /// The provider's GPU-vote findNearest path (`Metal4ComputeEngine.fusedDistanceTopK`) vs the
    /// CPU-vote path (`AccelerateFallback` + VectorCore `TopKSelection`). Also probes:
    /// * empty GPU results (the provider silently falls back on them — VA2-005),
    /// * k > N handling,
    /// * run-to-run determinism (bitwise).
    func testFusedTopKMatchesCPUSelectionAndIsDeterministic() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context, decisionEngine: GPUDecisionEngine())
        var rng = TestRNG(seed: 0xD1FF_0003)
        var failures: [String] = []

        for (n, dim) in [(33, 16), (257, 16), (1000, 768), (5000, 64)] {
            for k in [1, 8, 32, 33, 128, n, n + 7] {
                for cls in [ValueClass.normal, .duplicates, .nanPoisoned] {
                    for metric in [Metal4DistanceMetric.euclidean, .cosine] {
                        let db = Self.makeMatrix(cls, n: n, dim: dim, rng: &rng)
                        let q = Self.makeQuery(cls, dim: dim, rng: &rng)
                        let expectedCount = min(k, n)

                        let g1 = try await engine.fusedDistanceTopK(query: q, database: db, k: k, metric: metric)
                        let g2 = try await engine.fusedDistanceTopK(query: q, database: db, k: k, metric: metric)

                        if g1.isEmpty {
                            failures.append("EMPTY GPU RESULT \(metric) n=\(n) k=\(k) \(cls.rawValue) — provider would silently fall back (VA2-005)")
                            continue
                        }
                        if g1.count != expectedCount {
                            failures.append("count \(metric) n=\(n) k=\(k) \(cls.rawValue): got \(g1.count), expected \(expectedCount)")
                        }
                        let deterministic = g1.map(\.index) == g2.map(\.index)
                            && zip(g1, g2).allSatisfy { $0.distance.bitPattern == $1.distance.bitPattern }
                        if !deterministic {
                            failures.append("NONDETERMINISTIC \(metric) n=\(n) k=\(k) \(cls.rawValue)")
                        }

                        let dists: [Float] = metric == .euclidean
                            ? AccelerateFallback.batchEuclideanDistance(query: q, candidates: db)
                            : AccelerateFallback.batchCosineSimilarity(query: q, candidates: db).map { 1.0 - $0 }
                        let ref = MetalComputeProvider.selectTopK(dists, k: expectedCount, largerIsCloser: false)
                        for (rank, (gpuHit, refHit)) in zip(g1, ref).enumerated() {
                            let distanceOK = Self.agree(gpuHit.distance, refHit.distance)
                            // Index may differ only between distance-equivalent candidates (tie).
                            let indexOK = gpuHit.index == refHit.index || Self.agree(dists[gpuHit.index], refHit.distance)
                            if !(distanceOK && indexOK) {
                                let oracle = metric == .euclidean
                                    ? Self.oracleL2(q, db[gpuHit.index])
                                    : Self.oracleCosineDistance(q, db[gpuHit.index])
                                failures.append("\(metric) n=\(n) k=\(k) \(cls.rawValue) rank=\(rank): gpu=(\(gpuHit.index), \(gpuHit.distance)) ref=(\(refHit.index), \(refHit.distance)) oracle=\(oracle)")
                                break
                            }
                        }
                    }
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) fused-topK findings:\n" + failures.prefix(40).joined(separator: "\n"))
    }

    // MARK: - Zero-copy SoA differential (with provenance)

    /// SoA L2 and cosine are both hard assertions: `soa_cosine_distance` carries the VA2-008
    /// rescue and `AccelerateFallback` the matching Double rescue + NaN policy (VA2-009).
    func testSoAScoringMatchesCPUWithProvenance() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        var rng = TestRNG(seed: 0xD1FF_0004)
        var failures: [String] = []
        var soaCalls = 0
        await provider.resetRoutingTelemetry()

        let dim = 512   // SoACompatible fixed-dimension type
        for n in [1, 33, 1000] {
            // .tiny and .mixedScale added by the AUDIT-3 meta-review: soa_cosine_distance
            // carries a hand-rolled lane-major copy of va_cosine_rescaled_terms, and the
            // underflow-collapse trigger of that copy (normSq flushing to 0 on ~1e-20
            // components, rescue reconstructing a finite similarity) was never driven —
            // only the Inf trigger (.huge) and the trivial zero case were.
            for cls in [ValueClass.normal, .zeros, .duplicates, .huge, .nanPoisoned, .tiny, .mixedScale] {
                let candArrays = Self.makeMatrix(cls, n: n, dim: dim, rng: &rng)
                let queryArray = Self.makeQuery(cls, dim: dim, rng: &rng)
                let candidates = try candArrays.map { try Vector512Optimized($0) }
                let query = try Vector512Optimized(queryArray)
                let set = try SoACandidateSet(candidates: candidates, device: context.device)

                for metric in [SupportedDistanceMetric.euclidean, .cosine] {
                    let g = try await provider.batchDistance(query: query, against: set, metric: metric)
                    soaCalls += 1
                    let c: [Float]
                    switch metric {
                    case .euclidean:
                        c = AccelerateFallback.batchEuclideanDistance(query: queryArray, candidates: candArrays)
                    default:
                        c = AccelerateFallback.batchCosineSimilarity(query: queryArray, candidates: candArrays)
                            .map { 1.0 - $0 }
                    }
                    for j in 0..<n where !Self.agree(g[j], c[j]) {
                        let oracle = metric == .euclidean
                            ? Self.oracleL2(queryArray, candArrays[j])
                            : Self.oracleCosineDistance(queryArray, candArrays[j])
                        failures.append("soa \(metric) n=\(n) \(cls.rawValue) j=\(j): gpu=\(g[j]) cpu=\(c[j]) oracle=\(oracle)")
                        break
                    }

                    let k = min(5, n)
                    let nearest = try await provider.findNearest(query: query, in: set, k: k, metric: metric)
                    soaCalls += 1
                    XCTAssertEqual(nearest.count, k)
                    let ref = MetalComputeProvider.selectTopK(c, k: k, largerIsCloser: false)
                    for (rank, (gpuHit, refHit)) in zip(nearest, ref).enumerated() {
                        let indexOK = gpuHit.index == refHit.index || Self.agree(c[gpuHit.index], refHit.distance)
                        if !(Self.agree(gpuHit.distance, refHit.distance) && indexOK) {
                            failures.append("soa-topk \(metric) n=\(n) \(cls.rawValue) rank=\(rank): gpu=(\(gpuHit.index), \(gpuHit.distance)) ref=(\(refHit.index), \(refHit.distance))")
                            break
                        }
                    }
                }
            }
        }

        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, soaCalls,
            "provenance: every SoA call must have been served by the GPU kernel. telemetry=\(t)")
        XCTAssertTrue(failures.isEmpty, "\(failures.count) diverging SoA cells:\n" + failures.prefix(40).joined(separator: "\n"))
    }
}
