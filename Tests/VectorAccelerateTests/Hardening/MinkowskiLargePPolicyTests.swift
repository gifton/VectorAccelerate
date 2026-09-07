//
//  MinkowskiLargePPolicyTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-007: `minkowski_distance_batch` silently substituted the Chebyshev max-norm
//  for every p > 10 (`is_large_p` branch) — no flag, no warning. True Lp exceeds L∞ by up
//  to D^(1/p): with every |x_i − y_i| equal (the worst case, driven below), the kernel
//  returned 1.0 where the honest answer is 768^(1/50) ≈ 1.142 (p = 50) or
//  768^(1/11) ≈ 1.829 (p = 11 with an explicit `useStableComputation: false` — a flag that
//  read as a numerics choice but silently changed the *metric*). Reachable two ways under
//  the old routing: p > 30 auto-selected the batch kernel, and explicit
//  `useStableComputation: false` did so for any p.
//
//  Policy after the fix (owner decision, option B): the max-norm path survives but is
//  explicit opt-in via `Metal4MinkowskiConfig.chebyshevApproximation`. Without the flag,
//  every p computes true Lp (auto-routing to the stable kernel for all p > 10, with no
//  upper cap; explicit non-stable large p uses the requested formula within FP32
//  intermediate range limits; see MinkowskiRangePolicyTests).
//  The `.chebyshev` preset and the providers' `.chebyshev` metric carry the flag, so their
//  exact-L∞ semantics are unchanged — pinned by the must-stay-green controls below.
//
//  The uniform-difference geometry (query ≡ 0, dataset row ≡ 1) makes the expected value
//  the closed form D^(1/p) — no CPU reference kernel needed, and the substitution error is
//  fully realized.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MinkowskiLargePPolicyTests: XCTestCase {

    var context: Metal4Context!
    var kernel: MinkowskiDistanceKernel!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        try await super.setUp()
        context = try await Metal4Context()
        kernel = try await MinkowskiDistanceKernel(context: context)
    }

    override func tearDown() async throws {
        kernel = nil
        context = nil
        try await super.tearDown()
    }

    private static let dimension = 768

    /// query ≡ 0, dataset row ≡ 1 ⇒ every |diff| = 1 ⇒ true Lp = D^(1/p), L∞ = 1.
    private var uniformPair: (query: [Float], dataset: [Float]) {
        (Array(repeating: 0, count: Self.dimension),
         Array(repeating: 1, count: Self.dimension))
    }

    private static func trueLp(_ p: Double) -> Float {
        Float(pow(Double(dimension), 1.0 / p))
    }

    // MARK: - Red pre-fix: silent substitution paths

    /// p = 50 with all defaults: auto-routing must deliver true L50 = 768^(1/50) ≈ 1.1424,
    /// not the max-norm 1.0 the old `is_large_p` branch returned (a 14% error, silently).
    func testAutoRoutedPAbove30ComputesTrueLp() async throws {
        let (q, d) = uniformPair
        let result = try await kernel.computeDistances(
            queries: [q], dataset: [d],
            config: Metal4MinkowskiConfig(p: 50.0)
        )
        XCTAssertEqual(
            result.distance(row: 0, col: 0), Self.trueLp(50), accuracy: 0.01,
            "p=50 (default config) must compute true L50, not a silent Chebyshev substitute"
        )
    }

    /// Explicit `useStableComputation: false` at p = 11 previously flipped the *metric* to
    /// L∞ (worst case of the class: 768^(1/11) ≈ 1.829 vs 1.0 — an 83% error). Opting out
    /// of the stable kernel must still use the requested Lp formula on these in-range inputs.
    func testExplicitNonStableLargePComputesTrueLp() async throws {
        let (q, d) = uniformPair
        let result = try await kernel.computeDistances(
            queries: [q], dataset: [d],
            config: Metal4MinkowskiConfig(p: 11.0, useStableComputation: false)
        )
        XCTAssertEqual(
            result.distance(row: 0, col: 0), Self.trueLp(11), accuracy: 0.01,
            "useStableComputation:false is a numerics choice, not license to change the metric"
        )
    }

    // MARK: - Must-stay-green controls: opted-in Chebyshev semantics preserved

    /// The `.chebyshev` preset explicitly requests the max-norm and must return exact L∞ —
    /// green before AND after the fix. Mixed magnitudes with the max at an interior,
    /// non-multiple-of-4 index exercise both the float4 and remainder legs.
    func testChebyshevPresetStillExactMaxNorm() async throws {
        var d: [Float] = (0..<Self.dimension).map { Float($0 % 7) * 0.25 }
        d[137] = 5.0
        let q = [Float](repeating: 0, count: Self.dimension)

        let result = try await kernel.chebyshevDistances(queries: [q], dataset: [d])
        XCTAssertEqual(result.distance(row: 0, col: 0), 5.0, accuracy: 1e-4,
                       ".chebyshev preset must remain exact L∞ (max |diff|)")

        // Discriminating leg: with every |diff| equal, exact L∞ = 1.0 while true L100 would
        // be 768^(1/100) ≈ 1.069 — this pins the preset to the max-norm path specifically.
        let (uq, ud) = uniformPair
        let uniform = try await kernel.chebyshevDistances(queries: [uq], dataset: [ud])
        XCTAssertEqual(uniform.distance(row: 0, col: 0), 1.0, accuracy: 1e-4,
                       ".chebyshev preset must be exact L∞, not true L100")
    }

    /// The providers' `.chebyshev` metric is the other opted-in surface — exact L∞ for both
    /// the single-pair and batch entry points, green before AND after.
    func testChebyshevProviderMetricStillExactMaxNorm() async throws {
        let provider = try await MinkowskiKernelDistanceProvider(context: context)

        var raw: [Float] = (0..<Self.dimension).map { Float($0 % 5) * 0.5 }
        raw[301] = 9.0
        let v1 = DynamicVector([Float](repeating: 0, count: Self.dimension))
        let v2 = DynamicVector(raw)

        let single = try await provider.distance(from: v1, to: v2, metric: .chebyshev)
        XCTAssertEqual(single, 9.0, accuracy: 1e-4, "provider .chebyshev single-pair must be exact L∞")

        let batch = try await provider.batchDistance(from: v1, to: [v2, v1], metric: .chebyshev)
        XCTAssertEqual(batch[0], 9.0, accuracy: 1e-4, "provider .chebyshev batch must be exact L∞")
        XCTAssertEqual(batch[1], 0.0, accuracy: 1e-5, "self-distance must be 0")

        // Discriminating leg (see preset control): uniform diffs separate exact L∞ (1.0)
        // from true L100 (≈ 1.069), so a provider that forgot to opt in goes red here.
        let ones = DynamicVector([Float](repeating: 1, count: Self.dimension))
        let uniformSingle = try await provider.distance(from: v1, to: ones, metric: .chebyshev)
        XCTAssertEqual(uniformSingle, 1.0, accuracy: 1e-4, "provider .chebyshev must be exact L∞, not true L100")
        let uniformBatch = try await provider.batchDistance(from: v1, to: [ones], metric: .chebyshev)
        XCTAssertEqual(uniformBatch[0], 1.0, accuracy: 1e-4, "provider .chebyshev batch must be exact L∞, not true L100")
    }

    /// The already-correct auto-stable range (10, 30] must keep computing true Lp while the
    /// routing around it changes — green before AND after.
    func testStableAutoRangeControl() async throws {
        let (q, d) = uniformPair
        let result = try await kernel.computeDistances(
            queries: [q], dataset: [d],
            config: Metal4MinkowskiConfig(p: 15.0)
        )
        XCTAssertEqual(result.distance(row: 0, col: 0), Self.trueLp(15), accuracy: 0.01,
                       "p=15 auto-routes to the stable kernel and must stay exact")
    }
}
