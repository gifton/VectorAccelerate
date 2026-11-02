//
//  ChebyshevNaNPolicyTests.swift
//  VectorAccelerateTests
//
//  Owner decision 2026-10-31: every VectorAccelerate Chebyshev (L∞) path propagates NaN exactly as VectorCore
//  0.4.0's `ChebyshevDistance` does (downstream upgrade row BH-15): a NaN coordinate, or the same infinity on
//  both sides (Inf − Inf), makes the distance NaN; an infinite difference stays +Inf; finite input is the exact
//  maximum absolute difference. Before this every path used `max`, which drops NaN. Paths exercised here:
//  `MetalComputeProvider` (single and batch), `FallbackProvider`, `Metal4ComputeEngine.chebyshevDistance` on
//  its CPU branch and through the `chebyshevDistance` kernel, and `MinkowskiDistanceKernel.chebyshevDistances`
//  (the `minkowski_distance_batch` kernel's opted-in L∞ branch). `BatchProcessor` and `MemoryMapManager`
//  carry the same four-line loop and are not separately exercised.
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ChebyshevNaNPolicyTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    private struct Fixture {
        let name: String
        let a: [Float]
        let b: [Float]
    }

    /// Finite control plus every non-finite class of Core's policy, at the given dimension.
    private static func fixtures(dim: Int) -> [Fixture] {
        var rng = TestRNG(seed: 0xC4EB)
        let base = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        let other = (0..<dim).map { _ in rng.nextFloat(in: -1...1) }
        func with(_ v: [Float], _ i: Int, _ x: Float) -> [Float] { var c = v; c[i] = x; return c }
        return [
            Fixture(name: "finite", a: base, b: other),
            Fixture(name: "nan-in-a", a: with(base, 3, .nan), b: other),
            Fixture(name: "nan-in-b-last", a: base, b: with(other, dim - 1, .nan)),
            Fixture(name: "inf-in-a-only", a: with(base, 5, .infinity), b: other),
            Fixture(name: "same-inf-both", a: with(base, 7, .infinity), b: with(other, 7, .infinity)),
            Fixture(name: "opposite-inf", a: with(base, 2, .infinity), b: with(other, 2, -.infinity)),
        ]
    }

    /// Core's witness is the oracle: NaN parity, infinity parity, and exact equality for finite input (both
    /// sides take the maximum of the same exactly rounded FP32 differences).
    private func assertMatchesCore(_ got: Float, _ f: Fixture, _ path: String,
                                   file: StaticString = #filePath, line: UInt = #line) {
        let core = ChebyshevDistance().distance(DynamicVector(f.a), DynamicVector(f.b))
        XCTAssertEqual(got.isNaN, core.isNaN, "\(path)/\(f.name): NaN parity (core \(core), got \(got))", file: file, line: line)
        XCTAssertEqual(got.isInfinite, core.isInfinite, "\(path)/\(f.name): infinity parity (core \(core), got \(got))", file: file, line: line)
        if core.isFinite {
            XCTAssertEqual(got, core, "\(path)/\(f.name): finite Chebyshev must be exact", file: file, line: line)
        }
    }

    func testCoreWitnessPreconditions() {
        // The policy this file pins, stated against the oracle itself (0.4.0 behaviour).
        let f = Self.fixtures(dim: 16)
        let core = f.map { ChebyshevDistance().distance(DynamicVector($0.a), DynamicVector($0.b)) }
        XCTAssertTrue(core[0].isFinite)
        XCTAssertTrue(core[1].isNaN && core[2].isNaN, "NaN coordinate -> NaN")
        XCTAssertEqual(core[3], .infinity, "infinite difference -> +Inf")
        XCTAssertTrue(core[4].isNaN, "same infinity on both sides -> NaN")
        XCTAssertEqual(core[5], .infinity, "opposite infinities -> +Inf")
    }

    func testMetalComputeProviderSingleAndBatchMatchCore() async throws {
        let provider = try await MetalComputeProvider(configuration: .init(fallbackToCPU: false))
        for dim in [8, 259] {
            let fixtures = Self.fixtures(dim: dim)
            for f in fixtures {
                let single = try await provider.distance(DynamicVector(f.a), DynamicVector(f.b), metric: .chebyshev)
                assertMatchesCore(single, f, "provider.distance(dim \(dim))")
            }
            // One query against every fixture's `b`, as a batch.
            let query = fixtures[0].a
            let batch = try await provider.batchDistance(query: DynamicVector(query),
                                                         candidates: fixtures.map { DynamicVector($0.b) }, metric: .chebyshev)
            for (i, f) in fixtures.enumerated() {
                assertMatchesCore(batch[i], Fixture(name: f.name, a: query, b: f.b), "provider.batchDistance(dim \(dim))")
            }
        }
    }

    func testFallbackProviderMatchesCore() throws {
        let fallback = FallbackProvider()
        for f in Self.fixtures(dim: 37) {
            assertMatchesCore(try fallback.distance(from: f.a, to: f.b, metric: .chebyshev), f, "FallbackProvider")
            assertMatchesCore(try fallback.chebyshevDistance(from: f.a, to: f.b), f, "FallbackProvider.chebyshevDistance")
        }
    }

    /// `Metal4ComputeEngine.chebyshevDistance` takes its CPU branch at or below 64 dimensions and the
    /// `chebyshevDistance` kernel above it (no decision engine installed), so both paths are exercised.
    func testEngineCPUBranchAndKernelMatchCore() async throws {
        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)
        for dim in [16, 300] {
            for f in Self.fixtures(dim: dim) {
                let got = try await engine.chebyshevDistance(f.a, f.b)
                assertMatchesCore(got, f, "engine.chebyshevDistance(dim \(dim), \(dim > 64 ? "kernel" : "cpu"))")
            }
        }
    }

    /// The tiled `minkowski_distance_batch` kernel's opted-in L∞ branch, including the float4 lanes and the
    /// scalar remainder (259 = 64 * 4 + 3).
    func testMinkowskiKernelChebyshevMatchesCore() async throws {
        let context = try await Metal4Context()
        let kernel = try await MinkowskiDistanceKernel(context: context)
        for dim in [8, 259] {
            let fixtures = Self.fixtures(dim: dim)
            let result = try await kernel.chebyshevDistances(queries: fixtures.map(\.a), dataset: fixtures.map(\.b))
            for (i, f) in fixtures.enumerated() {
                assertMatchesCore(result.distance(row: i, col: i), f, "MinkowskiDistanceKernel.chebyshevDistances(dim \(dim))")
            }
        }
    }
}
