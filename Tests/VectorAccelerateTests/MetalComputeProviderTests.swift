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

    // MARK: - findNearest (top-K)

    func testFindNearestMatchesCPUReference() async throws {
        let dim = 256
        let (q, cands) = makeVectors(count: 1200, dim: dim, seed: 1234)
        let k = 10

        // Reference: brute-force euclidean top-k (ascending distance).
        let qa = q.toArray()
        let refPairs = cands.enumerated()
            .map { (index: $0.offset, distance: refEuclidean(qa, $0.element.toArray())) }
            .sorted { $0.distance < $1.distance }
            .prefix(k)
        let refIndices = refPairs.map { $0.index }

        let provider = try await MetalComputeProvider()
        let out = try await provider.findNearest(query: q, in: cands, k: k, metric: .euclidean)

        XCTAssertEqual(out.count, k)
        XCTAssertEqual(out.map { $0.index }, refIndices, "top-k indices should match brute force")
        for (got, ref) in zip(out, refPairs) {
            XCTAssertEqual(got.distance, ref.distance, accuracy: max(1e-2, abs(ref.distance) * 1e-3))
        }
    }

    func testFindNearestEmptyAndKClamp() async throws {
        let provider = try await MetalComputeProvider()
        let (q, cands) = makeVectors(count: 3, dim: 8, seed: 5)
        let empty = try await provider.findNearest(query: q, in: [], k: 5, metric: .euclidean)
        XCTAssertTrue(empty.isEmpty)
        // k larger than candidate count clamps to candidate count.
        let out = try await provider.findNearest(query: q, in: cands, k: 100, metric: .euclidean)
        XCTAssertEqual(out.count, 3)
    }

    // MARK: - distanceMatrix

    func testDistanceMatrixMatchesReference() async throws {
        let dim = 64
        let (_, vectors) = makeVectors(count: 40, dim: dim, seed: 314)   // reuse candidates as both sets
        let arrays = vectors.map { $0.toArray() }

        let provider = try await MetalComputeProvider(configuration: .init(preferGPU: false))   // deterministic CPU
        let matrix = try await provider.distanceMatrix(queries: vectors, candidates: vectors, metric: .euclidean)

        XCTAssertEqual(matrix.count, vectors.count)
        for i in 0..<vectors.count {
            XCTAssertEqual(matrix[i].count, vectors.count)
            XCTAssertEqual(matrix[i][i], 0, accuracy: 1e-3, "self-distance is zero")
            for j in 0..<vectors.count {
                XCTAssertEqual(matrix[i][j], refEuclidean(arrays[i], arrays[j]), accuracy: max(1e-2, abs(refEuclidean(arrays[i], arrays[j])) * 1e-3))
            }
        }
    }

    // MARK: - single distance()

    func testSingleDistancePerMetric() async throws {
        let provider = try await MetalComputeProvider()
        let a = DynamicVector([1, 2, 3, 4, 5, 6, 7, 8])
        let b = DynamicVector([8, 7, 6, 5, 4, 3, 2, 1])
        let av = a.toArray(); let bv = b.toArray()

        let eu = try await provider.distance(a, b, metric: .euclidean)
        XCTAssertEqual(eu, refEuclidean(av, bv), accuracy: 1e-4)
        let co = try await provider.distance(a, b, metric: .cosine)
        XCTAssertEqual(co, refCosineDistance(av, bv), accuracy: 1e-4)
        let ma = try await provider.distance(a, b, metric: .manhattan)
        XCTAssertEqual(ma, refManhattan(av, bv), accuracy: 1e-4)
        // dotProduct returns the raw dot (a similarity).
        let dp = try await provider.distance(a, b, metric: .dotProduct)
        let refDot = zip(av, bv).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(dp, refDot, accuracy: 1e-3)
        // chebyshev = max abs diff.
        let ch = try await provider.distance(a, b, metric: .chebyshev)
        let refCheby = zip(av, bv).map { abs($0 - $1) }.max() ?? 0
        XCTAssertEqual(ch, refCheby, accuracy: 1e-4)
    }

    // MARK: - Deprecation integrity

    @available(*, deprecated, message: "Exercises deprecated delegation on purpose")
    func testDeprecatedBatchOperationsStillDelegateCorrectly() async throws {
        let dim = 128
        let (q, cands) = makeVectors(count: 600, dim: dim, seed: 4242)
        let provider = try await MetalComputeProvider()

        // batchDistancesGPU delegates to provider.batchDistance.
        let legacyDistances = try await BatchOperations.batchDistancesGPU(from: q, to: cands, metric: .euclidean)
        let providerDistances = try await provider.batchDistance(query: q, candidates: cands, metric: .euclidean)
        assertClose(legacyDistances, providerDistances)

        // findNearestGPU delegates to provider.findNearest (indices must match).
        let legacyKnn = try await BatchOperations.findNearestGPU(to: q, in: cands, k: 8, metric: .euclidean)
        let providerKnn = try await provider.findNearest(query: q, in: cands, k: 8, metric: .euclidean)
        XCTAssertEqual(legacyKnn.map { $0.index }, providerKnn.map { $0.index })
    }

    // MARK: - BatchKernelProvider conformance (R4: transparent VectorCore dispatch)

    func testUsableAsBatchKernelProvider_euclideanTopKMatchesReference() async throws {
        let provider = try await MetalComputeProvider()
        let bk: any BatchKernelProvider = provider   // the VectorCore dispatch hook (R4)

        let (q, cands) = makeVectors(count: 500, dim: 768, seed: 0xB47C)
        let k = 10
        let result = try await bk.findNearest(query: q, candidates: cands, k: k, metric: EuclideanDistance())

        XCTAssertEqual(result.count, k)
        // Sorted ascending by distance.
        for i in 1..<result.count { XCTAssertLessThanOrEqual(result[i - 1].distance, result[i].distance + 1e-3) }
        // The k returned distances are the k smallest euclidean distances (robust to near-tie index swaps).
        let refSmallestK = cands.map { refEuclidean(q.toArray(), $0.toArray()) }.sorted().prefix(k)
        assertClose(result.map { $0.distance }, Array(refSmallestK), tol: 1e-2)
        // Each (index, distance) pair is self-consistent against the CPU reference.
        for r in result {
            XCTAssertEqual(r.distance, refEuclidean(q.toArray(), cands[r.index].toArray()), accuracy: 1e-2)
        }
    }

    func testBatchKernelProvider_unsupportedMetricDefersToVectorCore() async throws {
        let provider = try await MetalComputeProvider()
        let bk: any BatchKernelProvider = provider

        let (q, cands) = makeVectors(count: 200, dim: 256, seed: 0x3A11)
        // Manhattan is not a GPU-routed metric → the bridge must defer to VectorCore's own
        // ManhattanDistance.batchDistance (never the raw VA path), so results match exactly.
        let got = try await bk.batchDistance(query: q, candidates: cands, metric: ManhattanDistance())
        let ref = cands.map { refManhattan(q.toArray(), $0.toArray()) }
        assertClose(got, ref, tol: 1e-3)
    }
}
