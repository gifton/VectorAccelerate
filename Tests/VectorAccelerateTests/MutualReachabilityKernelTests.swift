// VectorAccelerate: MutualReachabilityKernel Tests
//
// Tests for mutual reachability distance computation (HDBSCAN).
//
// Phase 1: Core Foundation Tests
// - Dense mode correctness
// - Sparse mode correctness
// - Diagonal is zero
// - Symmetry property
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {
    /// CPU mutual reachability reference implementation.
    ///
    /// mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
    static func cpuMutualReachability(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) -> [[Float]] {
        let n = embeddings.count
        var result = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    result[i][j] = 0.0
                } else {
                    let euclidean = cpuL2Distance(embeddings[i], embeddings[j])
                    let mutualReach = max(max(coreDistances[i], coreDistances[j]), euclidean)
                    result[i][j] = mutualReach
                }
            }
        }

        return result
    }

    /// Generate random core distances (positive values, typically distances to k-th nearest neighbor)
    static func randomCoreDistances(count: Int) -> [Float] {
        (0..<count).map { _ in Float.random(in: 0.1...2.0) }
    }
}

// MARK: - MutualReachabilityKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MutualReachabilityKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: MutualReachabilityKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await MutualReachabilityKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Dense Mode Tests

    func testDenseCorrectness() async throws {
        // Small test case for exact verification
        let n = 10
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let gpuResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
        let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Verify dimensions
        XCTAssertEqual(gpuResult.count, n)
        XCTAssertEqual(gpuResult[0].count, n)

        // Verify values match CPU reference
        for i in 0..<n {
            for j in 0..<n {
                XCTAssertEqual(
                    gpuResult[i][j],
                    cpuResult[i][j],
                    accuracy: 1e-4,
                    "Mismatch at [\(i)][\(j)]: GPU=\(gpuResult[i][j]), CPU=\(cpuResult[i][j])"
                )
            }
        }
    }

    func testDiagonalIsZero() async throws {
        let n = 20
        let d = 32
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // Diagonal should always be zero
        for i in 0..<n {
            XCTAssertEqual(result[i][i], 0.0, "Diagonal at [\(i)][\(i)] should be 0")
        }
    }

    func testSymmetry() async throws {
        let n = 15
        let d = 24
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // Matrix should be symmetric: result[i][j] == result[j][i]
        for i in 0..<n {
            for j in i+1..<n {
                XCTAssertEqual(
                    result[i][j],
                    result[j][i],
                    accuracy: 1e-6,
                    "Not symmetric at [\(i)][\(j)]"
                )
            }
        }
    }

    func testMutualReachLowerBound() async throws {
        // mutual_reach(a, b) >= euclidean_dist(a, b) always
        let n = 10
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        for i in 0..<n {
            for j in 0..<n {
                if i != j {
                    let euclidean = Metal4KernelTestHelpers.cpuL2Distance(embeddings[i], embeddings[j])
                    XCTAssertGreaterThanOrEqual(
                        result[i][j],
                        euclidean - 1e-5,
                        "mutual_reach should be >= euclidean at [\(i)][\(j)]"
                    )
                }
            }
        }
    }

    func testDenseDimension384() async throws {
        // Test common embedding dimension
        let n = 50
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let gpuResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(gpuResult.count, n)
        XCTAssertEqual(gpuResult[0].count, n)

        // Spot check a few values against CPU reference
        let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Check corners and a few random positions
        XCTAssertEqual(gpuResult[0][1], cpuResult[0][1], accuracy: 1e-3)
        XCTAssertEqual(gpuResult[n-1][0], cpuResult[n-1][0], accuracy: 1e-3)
        XCTAssertEqual(gpuResult[n/2][n/2+1], cpuResult[n/2][n/2+1], accuracy: 1e-3)
    }

    // MARK: - Sparse Mode Tests

    func testSparseCorrectness() async throws {
        let n = 20
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Generate some random pairs to compute
        var pairs: [(Int, Int)] = []
        for _ in 0..<50 {
            let i = Int.random(in: 0..<n)
            let j = Int.random(in: 0..<n)
            pairs.append((i, j))
        }

        let sparseResult = try await kernel.computeSparse(
            embeddings: embeddings,
            coreDistances: coreDistances,
            pairs: pairs
        )

        // Compute dense for reference
        let denseResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // Verify sparse results match corresponding dense entries
        XCTAssertEqual(sparseResult.count, pairs.count)
        for (idx, (i, j)) in pairs.enumerated() {
            XCTAssertEqual(
                sparseResult[idx],
                denseResult[i][j],
                accuracy: 1e-5,
                "Sparse[\(idx)] for pair (\(i),\(j)) doesn't match dense"
            )
        }
    }

    func testSparseSamePointIsZero() async throws {
        let n = 10
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Pairs where i == j
        let pairs: [(Int, Int)] = [(0, 0), (5, 5), (9, 9)]

        let result = try await kernel.computeSparse(
            embeddings: embeddings,
            coreDistances: coreDistances,
            pairs: pairs
        )

        for (idx, (i, j)) in pairs.enumerated() {
            XCTAssertEqual(result[idx], 0.0, "Same-point pair (\(i),\(j)) should be 0")
        }
    }

    // MARK: - Edge Cases

    func testSinglePoint() async throws {
        let embeddings: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]
        let coreDistances: [Float] = [0.5]

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // 1x1 matrix with single zero
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].count, 1)
        XCTAssertEqual(result[0][0], 0.0)
    }

    func testTwoIdenticalPoints() async throws {
        let vec: [Float] = [1.0, 2.0, 3.0, 4.0]
        let embeddings: [[Float]] = [vec, vec]
        let coreDistances: [Float] = [0.3, 0.5]

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // euclidean distance is 0, so mutual_reach = max(0.3, 0.5, 0) = 0.5
        XCTAssertEqual(result[0][1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(result[1][0], 0.5, accuracy: 1e-6)
    }

    func testZeroCoreDistances() async throws {
        // When core distances are 0, mutual_reach falls back to euclidean distance
        let n = 5
        let d = 8
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = [Float](repeating: 0.0, count: n)

        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        // With zero core distances, mutual_reach == euclidean
        for i in 0..<n {
            for j in 0..<n {
                if i != j {
                    let expected = Metal4KernelTestHelpers.cpuL2Distance(embeddings[i], embeddings[j])
                    XCTAssertEqual(result[i][j], expected, accuracy: 1e-5)
                }
            }
        }
    }

    // MARK: - Dimension-Optimized Kernel Tests (Phase 2)

    func testDimensionOptimizedMatchesGeneric() async throws {
        // Test each optimized dimension produces same results as CPU reference
        for d in [384, 512, 768, 1536] {
            let n = 30  // Smaller n for faster tests with large dimensions
            let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
            let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

            let gpuResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
            let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
                embeddings: embeddings,
                coreDistances: coreDistances
            )

            // Verify against CPU reference
            for i in 0..<n {
                for j in 0..<n {
                    XCTAssertEqual(
                        gpuResult[i][j],
                        cpuResult[i][j],
                        accuracy: 1e-3,
                        "Dimension \(d) mismatch at [\(i)][\(j)]: GPU=\(gpuResult[i][j]), CPU=\(cpuResult[i][j])"
                    )
                }
            }
        }
    }

    func testOptimizedPipelineSelection() async throws {
        // Verify kernel reports correct optimized dimensions
        XCTAssertEqual(kernel.optimizedDimensions.sorted(), [384, 512, 768, 1536])
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 384))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 768))
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 256))  // Non-optimized
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 1024)) // Non-optimized
    }

    func testGenericFallbackForNonOptimizedDimension() async throws {
        // Test that non-optimized dimensions still work via generic kernel
        let n = 20
        let d = 256  // Not an optimized dimension
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let gpuResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
        let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Verify results match
        for i in 0..<n {
            for j in 0..<n {
                XCTAssertEqual(
                    gpuResult[i][j],
                    cpuResult[i][j],
                    accuracy: 1e-4,
                    "Generic fallback mismatch at [\(i)][\(j)]"
                )
            }
        }
    }

    func testPerformanceOptimized384() async throws {
        // Compare performance for optimized 384-dim
        let n = 200
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let start = CFAbsoluteTimeGetCurrent()
        _ = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // 200x200 @ 384-dim should complete quickly with optimized kernel
        XCTAssertLessThan(elapsed, 0.5, "384-dim optimized kernel too slow: \(elapsed)s")
    }

    // MARK: - Performance Tests

    func testPerformanceDense100() async throws {
        let n = 100
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Simple timing test (measure block has Sendable issues)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Should complete in reasonable time (< 1 second for 100x100 = 10K pairs)
        XCTAssertLessThan(elapsed, 1.0, "Dense 100x100 took too long: \(elapsed)s")
    }

    func testPerformanceSparse50K() async throws {
        let n = 500
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Generate 50K random pairs
        var pairs: [(Int, Int)] = []
        pairs.reserveCapacity(50000)
        for _ in 0..<50000 {
            pairs.append((Int.random(in: 0..<n), Int.random(in: 0..<n)))
        }

        // Simple timing test
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await kernel.computeSparse(
            embeddings: embeddings,
            coreDistances: coreDistances,
            pairs: pairs
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Should complete in reasonable time (< 2 seconds for 50K pairs)
        XCTAssertLessThan(elapsed, 2.0, "Sparse 50K took too long: \(elapsed)s")
    }

    // MARK: - Phase 3: Fusion & VectorProtocol Tests

    func testEncodeAPIProducesCorrectResults() async throws {
        let n = 30
        let d = 384
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Flatten and create buffers
        let device = context.device.rawDevice
        let flatEmbeddings = embeddings.flatMap { $0 }
        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create embeddings buffer")
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

        guard let outputBuffer = device.makeBuffer(
            length: n * n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create output buffer")
            return
        }

        // Use encode() API directly
        try await context.executeAndWait { [kernel] _, encoder in
            kernel!.encode(
                into: encoder,
                embeddings: embedBuffer,
                coreDistances: coreBuffer,
                output: outputBuffer,
                n: n,
                d: d
            )
        }

        // Verify against CPU reference
        let cpuResult = Metal4KernelTestHelpers.cpuMutualReachability(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: n * n)
        for i in 0..<n {
            for j in 0..<n {
                XCTAssertEqual(
                    ptr[i * n + j],
                    cpuResult[i][j],
                    accuracy: 1e-3,
                    "encode() mismatch at [\(i)][\(j)]"
                )
            }
        }
    }

    func testEncodeSparseAPIProducesCorrectResults() async throws {
        let n = 20
        let d = 128
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Generate some pairs
        let pairs: [(Int, Int)] = [(0, 1), (2, 3), (5, 5), (n-1, 0)]
        var packedPairs: [UInt32] = []
        for (i, j) in pairs {
            packedPairs.append(UInt32(i))
            packedPairs.append(UInt32(j))
        }

        let device = context.device.rawDevice
        let flatEmbeddings = embeddings.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let pairsBuffer = device.makeBuffer(
            bytes: packedPairs,
            length: packedPairs.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: pairs.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Use encodeSparse() API directly
        try await context.executeAndWait { [kernel] _, encoder in
            kernel!.encodeSparse(
                into: encoder,
                embeddings: embedBuffer,
                coreDistances: coreBuffer,
                pairs: pairsBuffer,
                output: outputBuffer,
                pairCount: pairs.count,
                d: d
            )
        }

        // Get dense result for comparison
        let denseResult = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: pairs.count)
        for (idx, (i, j)) in pairs.enumerated() {
            XCTAssertEqual(
                ptr[idx],
                denseResult[i][j],
                accuracy: 1e-5,
                "encodeSparse() mismatch for pair (\(i),\(j))"
            )
        }
    }

    func testFusibleKernelProperties() async throws {
        XCTAssertTrue(kernel.requiresBarrierAfter, "requiresBarrierAfter should be true")
        XCTAssertFalse(kernel.fusibleWith.isEmpty, "fusibleWith should not be empty")
        XCTAssertTrue(kernel.fusibleWith.contains("BoruvkaMST"), "Should be fusible with BoruvkaMST")
    }

    func testVectorProtocolSupport() async throws {
        // Test with DynamicVector from VectorCore
        let n = 20
        let d = 128

        // Create DynamicVector embeddings
        var embeddings: [DynamicVector] = []
        for _ in 0..<n {
            let values = (0..<d).map { _ in Float.random(in: -1...1) }
            embeddings.append(DynamicVector(values))
        }
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        // Should work with VectorProtocol
        let result = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)

        XCTAssertEqual(result.count, n, "Result should have n rows")
        XCTAssertEqual(result[0].count, n, "Result should have n columns")

        // Diagonal should be zero
        for i in 0..<n {
            XCTAssertEqual(result[i][i], 0.0, "Diagonal at [\(i)][\(i)] should be 0")
        }

        // Verify symmetry
        for i in 0..<n {
            for j in i+1..<n {
                XCTAssertEqual(
                    result[i][j],
                    result[j][i],
                    accuracy: 1e-6,
                    "VectorProtocol result not symmetric at [\(i)][\(j)]"
                )
            }
        }
    }

    func testMetal4EncodingResultReturned() async throws {
        let n = 10
        let d = 16
        let embeddings = Metal4KernelTestHelpers.randomVectors(count: n, dimension: d)
        let coreDistances = Metal4KernelTestHelpers.randomCoreDistances(count: n)

        let device = context.device.rawDevice
        let flatEmbeddings = embeddings.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbeddings,
            length: flatEmbeddings.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let coreBuffer = device.makeBuffer(
            bytes: coreDistances,
            length: coreDistances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: n * n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Use nonisolated(unsafe) to capture the result from the closure
        nonisolated(unsafe) var encodingResult: Metal4EncodingResult?

        try await context.executeAndWait { [kernel] _, encoder in
            encodingResult = kernel!.encode(
                into: encoder,
                embeddings: embedBuffer,
                coreDistances: coreBuffer,
                output: outputBuffer,
                n: n,
                d: d
            )
        }

        // Verify encoding result is returned with valid data
        XCTAssertNotNil(encodingResult)
        XCTAssertFalse(encodingResult!.pipelineName.isEmpty, "Pipeline name should not be empty")
        XCTAssertGreaterThan(encodingResult!.totalThreads, 0, "Total threads should be > 0")
    }
}
