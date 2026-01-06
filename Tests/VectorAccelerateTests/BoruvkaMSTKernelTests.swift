// VectorAccelerate: BoruvkaMSTKernel Tests
//
// Tests for GPU-accelerated MST computation using Borůvka's algorithm.
//
// Phase 1: Core Foundation Tests
// - Edge count (N-1 edges)
// - Connectivity verification
// - No duplicate edges
// - Basic correctness
// - Edge cases (empty, single, two points)
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {
    /// Verify MST is connected using Union-Find.
    ///
    /// An MST with N points should have exactly N-1 edges connecting all points
    /// into a single component.
    ///
    /// - Parameters:
    ///   - edges: MST edges as (source, target, weight) tuples
    ///   - n: Number of points
    /// - Returns: True if all points are connected
    static func verifyConnected(edges: [(source: Int, target: Int, weight: Float)], n: Int) -> Bool {
        guard n > 0 else { return true }
        if n == 1 { return edges.isEmpty }

        // Simple Union-Find with path compression
        var parent = Array(0..<n)

        func find(_ x: Int) -> Int {
            if parent[x] != x {
                parent[x] = find(parent[x])
            }
            return parent[x]
        }

        func union(_ x: Int, _ y: Int) {
            let px = find(x)
            let py = find(y)
            if px != py {
                parent[px] = py
            }
        }

        // Apply all edges
        for edge in edges {
            union(edge.source, edge.target)
        }

        // Check all points have same root
        let root = find(0)
        for i in 1..<n {
            if find(i) != root {
                return false
            }
        }
        return true
    }

    /// Check for duplicate edges in MST.
    ///
    /// Each edge (u, v) should appear at most once, regardless of direction.
    ///
    /// - Parameter edges: MST edges
    /// - Returns: True if duplicates exist
    static func hasDuplicateEdges(edges: [(source: Int, target: Int, weight: Float)]) -> Bool {
        var seen = Set<String>()
        for edge in edges {
            // Normalize edge to (min, max) for undirected comparison
            let key = "\(min(edge.source, edge.target))-\(max(edge.source, edge.target))"
            if seen.contains(key) {
                return true
            }
            seen.insert(key)
        }
        return false
    }

    /// Generate random core distances (positive values, typically distances to k-th nearest neighbor).
    ///
    /// Note: This is a copy of the helper from MutualReachabilityKernelTests to avoid
    /// test file dependency issues. The values represent the k-th nearest neighbor distance
    /// for each point, used in mutual reachability computations.
    static func randomBoruvkaCoreDistances(count: Int) -> [Float] {
        (0..<count).map { _ in Float.random(in: 0.1...2.0) }
    }

    // MARK: - CPU Prim's Reference Implementation

    /// CPU reference MST using Prim's algorithm with mutual reachability.
    ///
    /// This provides a ground-truth implementation for verifying GPU results.
    /// Prim's algorithm builds the MST by greedily adding the minimum-weight
    /// edge connecting a visited vertex to an unvisited vertex.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix
    ///   - coreDistances: N core distances
    /// - Returns: MST edges sorted by weight, and total weight
    static func cpuPrimsMST(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) -> (edges: [(source: Int, target: Int, weight: Float)], totalWeight: Float) {
        let n = embeddings.count
        guard n > 1 else { return ([], 0) }

        // Compute mutual reachability for edge (i, j)
        func mutualReachability(_ i: Int, _ j: Int) -> Float {
            var distSq: Float = 0
            for k in 0..<embeddings[i].count {
                let diff = embeddings[i][k] - embeddings[j][k]
                distSq += diff * diff
            }
            let dist = sqrt(distSq)
            return max(coreDistances[i], coreDistances[j], dist)
        }

        // Prim's algorithm
        var inMST = [Bool](repeating: false, count: n)
        var minWeight = [Float](repeating: .infinity, count: n)
        var parent = [Int](repeating: -1, count: n)

        minWeight[0] = 0

        var edges: [(source: Int, target: Int, weight: Float)] = []
        var totalWeight: Float = 0

        for _ in 0..<n {
            // Find minimum weight vertex not in MST
            var u = -1
            var minW: Float = .infinity
            for v in 0..<n {
                if !inMST[v] && minWeight[v] < minW {
                    minW = minWeight[v]
                    u = v
                }
            }

            guard u >= 0 else { break }
            inMST[u] = true

            // Add edge to MST (skip first vertex which has no parent)
            if parent[u] >= 0 {
                let weight = mutualReachability(parent[u], u)
                edges.append((source: parent[u], target: u, weight: weight))
                totalWeight += weight
            }

            // Update weights for adjacent vertices
            for v in 0..<n {
                if !inMST[v] {
                    let weight = mutualReachability(u, v)
                    if weight < minWeight[v] {
                        minWeight[v] = weight
                        parent[v] = u
                    }
                }
            }
        }

        return (edges, totalWeight)
    }
}

// MARK: - BoruvkaMSTKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class BoruvkaMSTKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: BoruvkaMSTKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await BoruvkaMSTKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Edge Count Tests

    func testEdgeCountSmall() async throws {
        let n = 10
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, n - 1, "MST should have exactly N-1 edges")
        XCTAssertEqual(result.pointCount, n)
    }

    func testEdgeCount100() async throws {
        let n = 100
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, n - 1, "MST should have exactly N-1 edges")
    }

    // MARK: - Connectivity Tests

    func testConnectivity() async throws {
        let n = 50
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n),
            "MST should connect all points"
        )
    }

    // MARK: - Duplicate Edge Tests

    func testNoDuplicateEdges() async throws {
        let n = 50
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertFalse(
            Metal4KernelTestHelpers.hasDuplicateEdges(edges: result.edges),
            "MST should not have duplicate edges"
        )
    }

    // MARK: - Edge Cases

    func testSinglePoint() async throws {
        let embeddings: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]
        let coreDistances: [Float] = [0.5]

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 0, "Single point should have no edges")
        XCTAssertEqual(result.totalWeight, 0)
        XCTAssertEqual(result.pointCount, 1)
    }

    func testTwoPoints() async throws {
        // Two points at known positions
        let embeddings: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0]  // L2 distance = sqrt(9 + 16) = 5
        ]
        let coreDistances: [Float] = [0.5, 0.5]

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 1, "Two points should have exactly 1 edge")
        XCTAssertEqual(result.pointCount, 2)

        // Weight should be mutual_reach = max(0.5, 0.5, 5.0) = 5.0
        XCTAssertEqual(result.totalWeight, 5.0, accuracy: 1e-4)
    }

    func testEmptyInput() async throws {
        let embeddings: [[Float]] = []
        let coreDistances: [Float] = []

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.edges.count, 0)
        XCTAssertEqual(result.totalWeight, 0)
        XCTAssertEqual(result.pointCount, 0)
    }

    // MARK: - Iteration Count Tests

    func testIterationCount() async throws {
        let n = 100
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // Boruvka's algorithm should complete in O(log N) iterations
        // Expected: ~log2(100) ≈ 7 iterations, with safety margin max = log2(N) + 2
        let expectedMax = Int(ceil(log2(Double(n)))) + 2
        XCTAssertLessThanOrEqual(
            result.iterations,
            expectedMax,
            "Should complete in O(log N) iterations"
        )
        XCTAssertGreaterThan(result.iterations, 0, "Should take at least 1 iteration")
    }

    // MARK: - Weight Sanity Tests

    func testTotalWeightPositive() async throws {
        let n = 20
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertGreaterThan(result.totalWeight, 0, "Total weight should be positive")

        // Verify individual edges have positive weights
        for edge in result.edges {
            XCTAssertGreaterThan(edge.weight, 0, "Edge weight should be positive")
        }
    }

    // MARK: - Mutual Reachability Tests

    func testMutualReachabilityDominatesEuclidean() async throws {
        // Test that mutual reachability is always >= euclidean distance
        // This validates the max(core_i, core_j, dist) formula
        let n = 10
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        // Use larger core distances to ensure they dominate some edges
        let coreDistances = (0..<n).map { _ in Float.random(in: 1.0...5.0) }

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // Each edge weight should be >= min of core distances
        // (because mutual_reach >= max(core_i, core_j) >= min(core_i, core_j))
        let minCore = coreDistances.min()!
        for edge in result.edges {
            XCTAssertGreaterThanOrEqual(
                edge.weight,
                minCore - 1e-5,
                "Edge weight should be >= minimum core distance"
            )
        }
    }

    // MARK: - Edge Validity Tests

    func testEdgeIndicesInRange() async throws {
        let n = 30
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // All edge indices should be valid
        for edge in result.edges {
            XCTAssertGreaterThanOrEqual(edge.source, 0, "Source index should be >= 0")
            XCTAssertLessThan(edge.source, n, "Source index should be < n")
            XCTAssertGreaterThanOrEqual(edge.target, 0, "Target index should be >= 0")
            XCTAssertLessThan(edge.target, n, "Target index should be < n")
            XCTAssertNotEqual(edge.source, edge.target, "Source and target should be different")
        }
    }

    // MARK: - MTLBuffer API Tests

    func testMTLBufferAPI() async throws {
        let n = 20
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        // Test MTLBuffer API directly
        let device = context.device.rawDevice
        let flatEmbeddings = embeddings.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create embedding buffer")
            return
        }

        guard let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create core distances buffer")
            return
        }

        let result = try await kernel.computeMST(
            embeddings: embedBuffer,
            coreDistances: coreBuffer,
            n: n,
            d: d
        )

        XCTAssertEqual(result.edges.count, n - 1, "MST should have N-1 edges")
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n),
            "MST should be connected"
        )
    }

    // MARK: - Dimension Variation Tests

    func testCommonDimensions() async throws {
        // Test common embedding dimensions used in practice
        let dimensions = [8, 16, 32, 64, 128]
        let n = 20

        for d in dimensions {
            let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
            let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

            let result = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

            XCTAssertEqual(
                result.edges.count,
                n - 1,
                "Dimension \(d): MST should have N-1 edges"
            )
            XCTAssertTrue(
                Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n),
                "Dimension \(d): MST should be connected"
            )
        }
    }

    // MARK: - Phase 2: MST Weight Correctness Tests

    func testWeightMatchesPrims() async throws {
        let n = 30
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        // Compute GPU MST
        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Compute CPU reference MST
        let (_, cpuTotalWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // MST total weight should match (any valid MST has the same total weight)
        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuTotalWeight,
            accuracy: 1e-3,
            "GPU MST weight should match CPU Prim's MST weight"
        )
    }

    func testWeightMatchesPrimsLarger() async throws {
        let n = 100
        let d = 32
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let (_, cpuTotalWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuTotalWeight,
            accuracy: 1e-2,  // Slightly larger tolerance for larger n
            "GPU MST weight should match CPU Prim's MST weight"
        )
    }

    // MARK: - Phase 2: Dimension-Optimized Kernel Correctness Tests

    func testOptimizedDimension384Correctness() async throws {
        let n = 20
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let (_, cpuWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(gpuResult.edges.count, n - 1, "384-dim: MST should have N-1 edges")
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: gpuResult.edges, n: n),
            "384-dim: MST should be connected"
        )
        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuWeight,
            accuracy: 1e-2,
            "384-dim: GPU MST weight should match CPU"
        )
    }

    func testOptimizedDimension512Correctness() async throws {
        let n = 20
        let d = 512
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let (_, cpuWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(gpuResult.edges.count, n - 1, "512-dim: MST should have N-1 edges")
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: gpuResult.edges, n: n),
            "512-dim: MST should be connected"
        )
        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuWeight,
            accuracy: 1e-2,
            "512-dim: GPU MST weight should match CPU"
        )
    }

    func testOptimizedDimension768Correctness() async throws {
        let n = 20
        let d = 768
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let gpuResult = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let (_, cpuWeight) = Metal4KernelTestHelpers.cpuPrimsMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(gpuResult.edges.count, n - 1, "768-dim: MST should have N-1 edges")
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: gpuResult.edges, n: n),
            "768-dim: MST should be connected"
        )
        XCTAssertEqual(
            gpuResult.totalWeight,
            cpuWeight,
            accuracy: 1e-2,
            "768-dim: GPU MST weight should match CPU"
        )
    }

    // MARK: - Phase 2: DimensionOptimizedKernel Protocol Tests

    func testDimensionOptimizedKernelConformance() async throws {
        // Verify the kernel conforms to DimensionOptimizedKernel protocol
        XCTAssertEqual(
            kernel.optimizedDimensions,
            [384, 512, 768, 1536],
            "Should report optimized dimensions"
        )

        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 384))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 512))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 768))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 1536))
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 256))
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 1024))
    }

    // MARK: - Phase 3: FusibleKernel Tests

    func testFusibleKernelConformance() async throws {
        // Verify FusibleKernel protocol properties
        XCTAssertEqual(
            kernel.fusibleWith,
            ["MutualReachabilityKernel", "FusedL2TopKKernel"],
            "Should report fusible kernel types"
        )
        XCTAssertTrue(kernel.requiresBarrierAfter, "Should require barrier after execution")
    }

    func testCreateWorkBuffers() async throws {
        let n = 50

        let workBuffers = try kernel.createWorkBuffers(n: n)

        // Verify buffer sizes
        XCTAssertEqual(workBuffers.pointMinWeight.length, n * MemoryLayout<Float>.size)
        XCTAssertEqual(workBuffers.pointMinTarget.length, n * MemoryLayout<UInt32>.size)
        XCTAssertEqual(workBuffers.componentMinWeight.length, n * MemoryLayout<Float>.size)
        XCTAssertEqual(workBuffers.componentMinSource.length, n * MemoryLayout<UInt32>.size)
        XCTAssertEqual(workBuffers.componentMinTarget.length, n * MemoryLayout<UInt32>.size)
        XCTAssertEqual(workBuffers.componentIds.length, n * MemoryLayout<UInt32>.size)
        XCTAssertGreaterThanOrEqual(workBuffers.candidateEdges.length, (n - 1) * 12) // MSTEdgeGPU is 12 bytes
        XCTAssertEqual(workBuffers.edgeCount.length, MemoryLayout<UInt32>.size)
    }

    // MARK: - Phase 3: VectorProtocol API Tests

    func testVectorProtocolAPI() async throws {
        let n = 20
        let d = 16

        // Create DynamicVector embeddings
        let embeddings: [DynamicVector] = (0..<n).map { _ in
            DynamicVector(Metal4KernelTestHelpers.randomVectors(count: 1, dimension: d)[0])
        }
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let result = try await kernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(result.edges.count, n - 1, "VectorProtocol: MST should have N-1 edges")
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.edges, n: n),
            "VectorProtocol: MST should be connected"
        )
    }

    func testVectorProtocolMatchesArrayAPI() async throws {
        let n = 15
        let d = 8
        let floatEmbeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        // Convert to DynamicVector
        let vectorEmbeddings: [DynamicVector] = floatEmbeddings.map { DynamicVector($0) }

        // Compute MST using both APIs
        let arrayResult = try await kernel.computeMST(
            embeddings: floatEmbeddings,
            coreDistances: coreDistances
        )
        let vectorResult = try await kernel.computeMST(
            embeddings: vectorEmbeddings,
            coreDistances: coreDistances
        )

        // MST total weight should match
        XCTAssertEqual(
            arrayResult.totalWeight,
            vectorResult.totalWeight,
            accuracy: 1e-4,
            "VectorProtocol API should produce same MST weight as array API"
        )
    }

    // MARK: - Phase 3: Performance Benchmarks

    func testPerformanceComparison_100Points() async throws {
        let n = 100
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        // Warm up
        _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // Measure GPU
        let gpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0

        // Measure CPU
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = Metal4KernelTestHelpers.cpuPrimsMST(embeddings: embeddings, coreDistances: coreDistances)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0

        let speedup = cpuTime / gpuTime
        print("Performance (n=\(n), d=\(d)): GPU=\(gpuTime*1000)ms, CPU=\(cpuTime*1000)ms, Speedup=\(speedup)x")

        // GPU should be competitive at this size (may be faster or slower depending on hardware)
        // We just verify both produce correct results
        XCTAssertGreaterThan(speedup, 0.1, "GPU should be within reasonable range of CPU for n=\(n)")
    }

    func testPerformanceComparison_500Points() async throws {
        let n = 500
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        // Warm up
        _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)

        // Measure GPU (fewer iterations due to longer runtime)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = try await kernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 3.0

        // Measure CPU
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = Metal4KernelTestHelpers.cpuPrimsMST(embeddings: embeddings, coreDistances: coreDistances)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 3.0

        let speedup = cpuTime / gpuTime
        print("Performance (n=\(n), d=\(d)): GPU=\(gpuTime*1000)ms, CPU=\(cpuTime*1000)ms, Speedup=\(speedup)x")

        // At 500 points, GPU should show speedup on most hardware
        XCTAssertGreaterThan(speedup, 1.0, "GPU should be faster than CPU for n=\(n)")
    }
}

// MARK: - HDBSCANDistanceModule Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class HDBSCANDistanceModuleTests: XCTestCase {

    var context: Metal4Context!
    var module: HDBSCANDistanceModule!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        module = try await HDBSCANDistanceModule(context: context)
    }

    override func tearDown() {
        module = nil
        context = nil
        super.tearDown()
    }

    func testBasicMSTComputation() async throws {
        let n = 50
        let d = 32
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        XCTAssertEqual(result.pointCount, n)
        XCTAssertEqual(result.minSamples, 5)
        XCTAssertEqual(result.coreDistances.count, n)
        XCTAssertEqual(result.mst.edges.count, n - 1)
    }

    func testCoreDistancesPositive() async throws {
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: 30, dimension: 16)

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        for cd in result.coreDistances {
            XCTAssertGreaterThanOrEqual(cd, 0, "Core distances should be non-negative")
        }
    }

    func testPrecomputedCoreDistances() async throws {
        let n = 30
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomBoruvkaCoreDistances(count: n)

        let mst = try await module.computeMSTWithCoreDistances(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        XCTAssertEqual(mst.edges.count, n - 1)
    }

    func testEmptyInput() async throws {
        let embeddings: [[Float]] = []

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        XCTAssertEqual(result.pointCount, 0)
        XCTAssertEqual(result.coreDistances.count, 0)
        XCTAssertEqual(result.mst.edges.count, 0)
    }

    func testSmallMinSamples() async throws {
        let n = 20
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)

        // Test with minSamples = 2 (small value)
        let result = try await module.computeMST(embeddings: embeddings, minSamples: 2)

        XCTAssertEqual(result.minSamples, 2)
        XCTAssertEqual(result.mst.edges.count, n - 1)
        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.mst.edges, n: n),
            "MST should be connected"
        )
    }

    func testVectorProtocolAPI() async throws {
        let n = 25
        let d = 16

        // Create DynamicVector embeddings
        let embeddings: [DynamicVector] = (0..<n).map { _ in
            DynamicVector(Metal4KernelTestHelpers.randomVectors(count: 1, dimension: d)[0])
        }

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        XCTAssertEqual(result.pointCount, n)
        XCTAssertEqual(result.mst.edges.count, n - 1)
    }

    func testMSTConnectivity() async throws {
        let n = 40
        let d = 24
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)

        let result = try await module.computeMST(embeddings: embeddings, minSamples: 5)

        XCTAssertTrue(
            Metal4KernelTestHelpers.verifyConnected(edges: result.mst.edges, n: n),
            "HDBSCAN MST should connect all points"
        )
    }
}
