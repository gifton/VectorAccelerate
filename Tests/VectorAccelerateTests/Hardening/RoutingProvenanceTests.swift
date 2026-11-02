//
//  RoutingProvenanceTests.swift
//  VectorAccelerateTests
//
//  Phase-0 hardening audit: prove which silicon actually served each MetalComputeProvider call.
//
//  These tests PIN currently-observed routing behavior — including behavior that is itself an
//  audit finding (batchDistance can never take its GPU path under default configuration, because
//  it consults the decision engine with k = 0 while the engine gates on k >= minKForGPU and on
//  queryCount·candidateCount·k >= minOperationsForGPU; see AUDIT-2 VA2-003). If a pin breaks
//  because routing was deliberately fixed, update the pin AND the audit ledger together.
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class RoutingProvenanceTests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private func makeVectors(_ n: Int, _ dim: Int, seed: UInt64 = 42) -> [DynamicVector] {
        var rng = TestRNG(seed: seed)
        return (0..<n).map { _ in DynamicVector((0..<dim).map { _ in rng.nextFloat(in: -1...1) }) }
    }

    /// VA2-003 (FIXED): distance-shaped operations no longer consult the selection k-gates, so
    /// `batchDistance` at documented GPU scale (2000 candidates × 128 dims) actually runs on the
    /// GPU under default configuration — while small workloads still route to CPU.
    func testDefaultConfigBatchDistanceRoutesByScale() async throws {
        let provider = try await MetalComputeProvider()

        let big = makeVectors(2001, 128)
        let distances = try await provider.batchDistance(
            query: big[0], candidates: Array(big.dropFirst()), metric: .euclidean)
        XCTAssertEqual(distances.count, 2000)
        var t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1,
            "2000×128 clears every distance gate; the GPU kernel must have served this. telemetry=\(t)")
        XCTAssertEqual(t.cpuFallbackAfterGPUError, 0,
            "GPU errored and was silently rescued: \(t.lastGPUErrorDescription ?? "?")")

        await provider.resetRoutingTelemetry()
        let small = makeVectors(301, 128)   // 300 candidates < minCandidatesForGPU (500)
        _ = try await provider.batchDistance(
            query: small[0], candidates: Array(small.dropFirst()), metric: .euclidean)
        t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0, "small workloads still belong on CPU. telemetry=\(t)")
        XCTAssertEqual(t.cpuDecisionEngine, 1, "telemetry=\(t)")
    }

    /// The fused GPU findNearest path is reachable at N=5000, k=10 (5000·10 = the exact
    /// minOperationsForGPU boundary). This is the one default-config front-door GPU path.
    func testFindNearestLargeNRoutesToGPU() async throws {
        let provider = try await MetalComputeProvider()
        let vecs = makeVectors(5001, 64)
        let result = try await provider.findNearest(
            query: vecs[0], in: Array(vecs.dropFirst()), k: 10, metric: .euclidean)
        XCTAssertEqual(result.count, 10)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 1,
            "N=5000, k=10 passes every decision-engine gate; the fused GPU path must have served this. telemetry=\(t)")
        XCTAssertEqual(t.cpuFallbackAfterGPUError, 0,
            "GPU path errored and was silently rescued: \(t.lastGPUErrorDescription ?? "?")")
    }

    /// `preferGPU: false` must mean zero GPU dispatches, visibly.
    func testPreferGPUFalseIsAllCPU() async throws {
        let provider = try await MetalComputeProvider(configuration: .init(preferGPU: false))
        let vecs = makeVectors(5001, 64)
        _ = try await provider.findNearest(
            query: vecs[0], in: Array(vecs.dropFirst()), k: 10, metric: .euclidean)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 0)
        XCTAssertGreaterThanOrEqual(t.cpuDecisionEngine, 1, "telemetry=\(t)")
    }

    /// Non-GPU metrics are CPU by policy and must be counted as such.
    func testPolicyMetricsCountAsCPUPolicy() async throws {
        let provider = try await MetalComputeProvider()
        let vecs = makeVectors(64, 32)
        _ = try await provider.batchDistance(query: vecs[0], candidates: Array(vecs.dropFirst()), metric: .dotProduct)
        _ = try await provider.batchDistance(query: vecs[0], candidates: Array(vecs.dropFirst()), metric: .manhattan)
        _ = try await provider.distance(vecs[0], vecs[1], metric: .euclidean)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.cpuPolicy, 3, "telemetry=\(t)")
        XCTAssertEqual(t.gpuKernel, 0)
    }

    /// The SoA scoring extension bypasses the decision engine by design — always GPU.
    func testSoAPathAlwaysGPU() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        var rng = TestRNG(seed: 7)
        let candidates = try (0..<64).map { _ in
            try Vector512Optimized((0..<512).map { _ in rng.nextFloat(in: -1...1) })
        }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let query = candidates[0]
        _ = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
        _ = try await provider.findNearest(query: query, in: set, k: 5, metric: .cosine)
        let t = await provider.routingTelemetry()
        XCTAssertEqual(t.gpuKernel, 2, "SoA scoring never consults the decision engine. telemetry=\(t)")
    }
}
