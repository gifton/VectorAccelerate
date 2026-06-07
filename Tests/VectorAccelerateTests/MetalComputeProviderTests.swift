//
//  MetalComputeProviderTests.swift
//  VectorAccelerateTests
//
//  Tests for the unified GPU compute façade. See
//  docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md
//

import XCTest
@testable import VectorAccelerate
import VectorCore
@preconcurrency import Metal

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MetalComputeProviderTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    func testConformsToComputeProviderWithGPUIdentity() async throws {
        let provider = try await MetalComputeProvider()
        // ComputeProvider shim reports GPU identity with real capabilities.
        XCTAssertEqual(provider.device, .gpu(index: 0))
        XCTAssertGreaterThan(provider.maxConcurrency, 0)
        XCTAssertFalse(provider.deviceInfo.name.isEmpty)
        XCTAssertGreaterThan(provider.deviceInfo.maxThreads, 0)
        // It is usable as an existential ComputeProvider, and `execute` runs the closure.
        let p: any ComputeProvider = provider
        let answer = try await p.execute { 41 + 1 }
        XCTAssertEqual(answer, 42)
    }

    // MARK: - CPU reference helpers (independent ground truth)

    private func makeVectors(count: Int, dim: Int, seed: UInt64) -> (query: DynamicVector, candidates: [DynamicVector]) {
        // Deterministic LCG (Date/Math.random are intentionally avoided).
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1   // [-1, 1)
        }
        let query = DynamicVector((0..<dim).map { _ in next() })
        let candidates = (0..<count).map { _ in DynamicVector((0..<dim).map { _ in next() }) }
        return (query, candidates)
    }

    private func refEuclidean(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { let d = a[i] - b[i]; s += d * d }; return s.squareRoot()
    }
    private func refCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom < .leastNormalMagnitude ? 1.0 : 1.0 - (dot / denom)
    }
    private func refManhattan(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { s += abs(a[i] - b[i]) }; return s
    }

    private func assertClose(_ a: [Float], _ b: [Float], tol: Float = 1e-2, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "count mismatch", file: file, line: line)
        for i in 0..<min(a.count, b.count) {
            let scale = max(1, abs(b[i]))
            XCTAssertEqual(a[i], b[i], accuracy: tol * scale, "index \(i)", file: file, line: line)
        }
    }

    // MARK: - batchDistance

    func testBatchDistanceEuclideanParity_CPUandGPU() async throws {
        for dim in [64, 768, 1536] {
            let (q, cands) = makeVectors(count: 1100, dim: dim, seed: UInt64(dim))
            let reference = cands.map { refEuclidean(q.toArray(), $0.toArray()) }

            // CPU path (preferGPU: false bypasses the decision engine).
            let cpu = try await MetalComputeProvider(configuration: .init(preferGPU: false))
            let cpuOut = try await cpu.batchDistance(query: q, candidates: cands, metric: .euclidean)
            assertClose(cpuOut, reference)

            // GPU path (default config; 1100 candidates clears the routing minimums).
            let gpu = try await MetalComputeProvider()
            let gpuOut = try await gpu.batchDistance(query: q, candidates: cands, metric: .euclidean)
            assertClose(gpuOut, reference)
        }
    }

    func testBatchDistanceCosineParity_CPUandGPU() async throws {
        for dim in [64, 768] {
            let (q, cands) = makeVectors(count: 1100, dim: dim, seed: UInt64(dim) &+ 7)
            let reference = cands.map { refCosineDistance(q.toArray(), $0.toArray()) }

            let cpu = try await MetalComputeProvider(configuration: .init(preferGPU: false))
            assertClose(try await cpu.batchDistance(query: q, candidates: cands, metric: .cosine), reference)

            let gpu = try await MetalComputeProvider()
            assertClose(try await gpu.batchDistance(query: q, candidates: cands, metric: .cosine), reference)
        }
    }

    func testBatchDistanceManhattanFallsBackToCPU() async throws {
        let (q, cands) = makeVectors(count: 300, dim: 128, seed: 99)
        let reference = cands.map { refManhattan(q.toArray(), $0.toArray()) }
        let provider = try await MetalComputeProvider()   // manhattan has no GPU batch kernel → CPU
        assertClose(try await provider.batchDistance(query: q, candidates: cands, metric: .manhattan), reference)
    }

    func testBatchDistanceRejectsRaggedCandidates() async throws {
        let provider = try await MetalComputeProvider()
        let q = DynamicVector([1, 2, 3, 4])
        let cands = [DynamicVector([1, 2, 3, 4]), DynamicVector([1, 2, 3])]   // ragged
        do {
            _ = try await provider.batchDistance(query: q, candidates: cands, metric: .euclidean)
            XCTFail("expected dimension-mismatch rejection")
        } catch { /* expected */ }
    }

    func testBatchDistanceEmptyCandidatesReturnsEmpty() async throws {
        let provider = try await MetalComputeProvider()
        let out = try await provider.batchDistance(query: DynamicVector([1, 2, 3]), candidates: [], metric: .euclidean)
        XCTAssertTrue(out.isEmpty)
    }
}
