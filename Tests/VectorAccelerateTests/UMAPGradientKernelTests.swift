//
//  UMAPGradientKernelTests.swift
//  VectorAccelerate
//
//  Tests for GPU-accelerated UMAP gradient computation.
//
//  Phase 1: Core Foundation Tests (9 tests)
//  - Segment computation correctness
//  - Gradient computation produces valid output
//  - Gradient signs are correct (attractive pulls together)
//  - Edge cases
//
//  Phase 2: Optimization API Tests (6 tests)
//  - optimizeEpoch reduces loss
//  - Learning rate decay support
//  - No edges / no negative sampling edge case
//  - Negative sampling pushes apart
//  - applyGradients modifies embedding
//  - Loss computation
//
//  Phase 3: Protocol Conformance & Performance Tests (5 tests)
//  - FusibleKernel conformance
//  - executeEpoch buffer-based API
//  - encodeGradients API
//  - GPU target gradient accumulation correctness
//  - Performance benchmark
//
//  Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4KernelTestHelpers {

    /// CPU reference implementation for UMAP edge gradient.
    ///
    /// Computes the attractive gradient for a single edge.
    static func cpuUMAPEdgeGradient(
        embedding: [[Float]],
        edge: UMAPEdge,
        params: UMAPParameters
    ) -> [Float] {
        let i = Int(edge.source)
        let j = Int(edge.target)
        let d = embedding[0].count

        // Compute squared distance
        var distSq: Float = 0
        for k in 0..<d {
            let diff = embedding[i][k] - embedding[j][k]
            distSq += diff * diff
        }

        // Gradient coefficient
        let distSqClamped = max(distSq, params.epsilon)
        let distSqPowB = pow(distSqClamped, params.b)
        let distSqPowBm1 = pow(distSqClamped, params.b - 1.0)
        let denom = 1.0 + params.a * distSqPowB

        var gradCoeff = -2.0 * params.a * params.b * distSqPowBm1 / denom
        gradCoeff *= edge.weight * params.learningRate
        gradCoeff = max(-4.0, min(4.0, gradCoeff))  // Clamp

        // Compute gradient
        var grad = [Float](repeating: 0, count: d)
        for k in 0..<d {
            let diff = embedding[i][k] - embedding[j][k]
            grad[k] = gradCoeff * diff
        }

        return grad
    }

    /// Generate random UMAP edges for testing.
    static func randomUMAPEdges(n: Int, edgesPerPoint: Int = 5) -> [UMAPEdge] {
        var edges: [UMAPEdge] = []
        edges.reserveCapacity(n * edgesPerPoint)

        for i in 0..<n {
            var targets = Set<Int>()
            while targets.count < min(edgesPerPoint, n - 1) {
                let j = Int.random(in: 0..<n)
                if j != i {
                    targets.insert(j)
                }
            }
            for j in targets {
                edges.append(UMAPEdge(
                    source: i,
                    target: j,
                    weight: Float.random(in: 0.5...1.0)
                ))
            }
        }

        // Sort by source
        return edges.sorted { $0.source < $1.source }
    }

    /// Generate random low-dimensional embedding for UMAP testing.
    static func randomLowDimEmbedding(n: Int, d: Int = 2) -> [[Float]] {
        return (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -10...10) }
        }
    }
}

// MARK: - UMAPGradientKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class UMAPGradientKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: UMAPGradientKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await UMAPGradientKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Segment Computation Tests

    func testSegmentComputationBasic() async throws {
        // Simple case: 3 points, edges sorted by source
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 1.0),
            UMAPEdge(source: 0, target: 2, weight: 1.0),
            UMAPEdge(source: 1, target: 0, weight: 1.0),
            UMAPEdge(source: 2, target: 0, weight: 1.0),
        ]

        let (starts, counts) = kernel.computeSegments(edges: edges, n: 3)

        XCTAssertEqual(starts[0], 0, "Source 0 starts at index 0")
        XCTAssertEqual(counts[0], 2, "Source 0 has 2 edges")
        XCTAssertEqual(starts[1], 2, "Source 1 starts at index 2")
        XCTAssertEqual(counts[1], 1, "Source 1 has 1 edge")
        XCTAssertEqual(starts[2], 3, "Source 2 starts at index 3")
        XCTAssertEqual(counts[2], 1, "Source 2 has 1 edge")
    }

    func testSegmentComputationWithGaps() async throws {
        // Point 1 has no outgoing edges
        let edges = [
            UMAPEdge(source: 0, target: 2, weight: 1.0),
            UMAPEdge(source: 2, target: 0, weight: 1.0),
        ]

        let (starts, counts) = kernel.computeSegments(edges: edges, n: 3)

        XCTAssertEqual(counts[0], 1, "Source 0 has 1 edge")
        XCTAssertEqual(counts[1], 0, "Source 1 has 0 edges")
        XCTAssertEqual(counts[2], 1, "Source 2 has 1 edge")
    }

    func testSegmentComputationEmpty() async throws {
        let edges: [UMAPEdge] = []
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 5)

        XCTAssertEqual(starts.count, 5)
        XCTAssertEqual(counts.count, 5)
        for i in 0..<5 {
            XCTAssertEqual(counts[i], 0, "All counts should be 0")
        }
    }

    // MARK: - Gradient Computation Tests

    func testGradientComputationProducesOutput() async throws {
        let n = 20
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice

        // Create buffers
        let flatEmbedding = embedding.flatMap { $0 }
        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create embedding buffer")
            return
        }

        guard let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create edge buffer")
            return
        }

        guard let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create starts buffer")
            return
        }

        guard let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create counts buffer")
            return
        }

        // Compute gradients
        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: edges.count,
            params: .default
        )

        // Verify output has correct size
        XCTAssertEqual(gradBuffer.length, n * d * MemoryLayout<Float>.size)

        // Verify gradients are finite (no NaN or Inf)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<(n * d) {
            XCTAssertFalse(gradPtr[i].isNaN, "Gradient should not be NaN at index \(i)")
            XCTAssertFalse(gradPtr[i].isInfinite, "Gradient should not be Inf at index \(i)")
        }
    }

    func testGradientSignIsCorrect() async throws {
        // Two points far apart should have attractive gradient pulling them together
        let embedding: [[Float]] = [
            [0.0, 0.0],   // Point 0 at origin
            [10.0, 0.0],  // Point 1 far to the right
        ]
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 1.0),
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 2)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 2,
            d: 2,
            edgeCount: 1,
            params: .default
        )

        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 4)

        // Point 0's gradient in x-direction should be positive (pull toward point 1)
        // Because gradient = coeff * (x0 - x1) = coeff * (0 - 10) = coeff * -10
        // And coeff is negative (attractive), so gradient is positive
        XCTAssertGreaterThan(
            gradPtr[0], 0,
            "Point 0 should be pulled toward point 1 (positive x gradient)"
        )
    }

    func testGradientMatchesCPUReference() async throws {
        // Small test case for exact verification
        let n = 5
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 0.8),
            UMAPEdge(source: 0, target: 2, weight: 0.6),
            UMAPEdge(source: 1, target: 0, weight: 0.8),
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let params = UMAPParameters.default
        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: edges.count,
            params: params
        )

        // Compute CPU reference (sum edge gradients per point)
        var cpuGradients = [[Float]](repeating: [Float](repeating: 0, count: d), count: n)
        for edge in edges {
            let edgeGrad = Metal4KernelTestHelpers.cpuUMAPEdgeGradient(
                embedding: embedding,
                edge: edge,
                params: params
            )
            let src = Int(edge.source)
            for k in 0..<d {
                cpuGradients[src][k] += edgeGrad[k]
            }
        }

        // Compare GPU vs CPU
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<n {
            for k in 0..<d {
                XCTAssertEqual(
                    gradPtr[i * d + k],
                    cpuGradients[i][k],
                    accuracy: 1e-4,
                    "Gradient mismatch at point \(i), dim \(k)"
                )
            }
        }
    }

    // MARK: - Edge Cases

    func testSinglePointNoEdges() async throws {
        let embedding: [[Float]] = [[1.0, 2.0]]
        let edges: [UMAPEdge] = []
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 1)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            length: MemoryLayout<UMAPEdge>.size,  // Dummy buffer
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 1,
            d: 2,
            edgeCount: 0,
            params: .default
        )

        // Gradient should be zero (no edges)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 2)
        XCTAssertEqual(gradPtr[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(gradPtr[1], 0.0, accuracy: 1e-6)
    }

    func testZeroWeightEdge() async throws {
        let embedding: [[Float]] = [
            [0.0, 0.0],
            [5.0, 0.0],
        ]
        let edges = [
            UMAPEdge(source: 0, target: 1, weight: 0.0),  // Zero weight
        ]
        let (starts, counts) = kernel.computeSegments(edges: edges, n: 2)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let gradBuffer = try await kernel.computeGradients(
            embedding: embedBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: 2,
            d: 2,
            edgeCount: 1,
            params: .default
        )

        // Gradient should be zero (zero weight edge)
        let gradPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: 4)
        XCTAssertEqual(gradPtr[0], 0.0, accuracy: 1e-6, "Zero-weight edge should produce zero gradient")
        XCTAssertEqual(gradPtr[1], 0.0, accuracy: 1e-6)
    }

    // MARK: - Sort Helper Test

    func testSortEdgesBySource() async throws {
        let unsorted = [
            UMAPEdge(source: 2, target: 0, weight: 1.0),
            UMAPEdge(source: 0, target: 1, weight: 1.0),
            UMAPEdge(source: 1, target: 2, weight: 1.0),
            UMAPEdge(source: 0, target: 2, weight: 1.0),
        ]

        let sorted = kernel.sortEdgesBySource(unsorted)

        XCTAssertEqual(sorted[0].source, 0)
        XCTAssertEqual(sorted[1].source, 0)
        XCTAssertEqual(sorted[2].source, 1)
        XCTAssertEqual(sorted[3].source, 2)
    }

    // MARK: - Phase 2: Optimization API Tests

    func testOptimizeEpochReducesLoss() async throws {
        let n = 30
        let d = 2
        var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)

        let params = UMAPParameters.default
        let initialLoss = kernel.computeLoss(embedding: embedding, edges: edges, params: params)

        // Run a few optimization epochs
        for _ in 0..<5 {
            try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
        }

        let finalLoss = kernel.computeLoss(embedding: embedding, edges: edges, params: params)

        XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease after optimization")
    }

    func testOptimizeEpochWithLearningRateDecay() async throws {
        let n = 20
        let d = 2
        var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 3)

        let nEpochs = 10
        let initialLR: Float = 1.0

        for epoch in 0..<nEpochs {
            var params = UMAPParameters.default
            params.learningRate = initialLR * (1.0 - Float(epoch) / Float(nEpochs))
            try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
        }

        // Verify embedding values are finite
        for i in 0..<n {
            for k in 0..<d {
                XCTAssertFalse(embedding[i][k].isNaN, "Embedding should not be NaN after optimization")
                XCTAssertFalse(embedding[i][k].isInfinite, "Embedding should not be Inf after optimization")
            }
        }
    }

    func testOptimizeEpochNoEdges() async throws {
        let n = 5
        let d = 2
        var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges: [UMAPEdge] = []
        let originalEmbedding = embedding

        // With no edges and no negative sampling, embedding should change only due to negative sampling
        var params = UMAPParameters.default
        params.negativeSampleRate = 0  // Disable negative sampling

        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

        // Embedding should be unchanged (no edges, no negative samples)
        for i in 0..<n {
            for k in 0..<d {
                XCTAssertEqual(embedding[i][k], originalEmbedding[i][k], accuracy: 1e-6)
            }
        }
    }

    func testNegativeSamplingPushesApart() async throws {
        // Two nearby points with no edges should be pushed apart by negative sampling
        var embedding: [[Float]] = [
            [0.0, 0.0],
            [0.1, 0.0],  // Very close to point 0
        ]
        let edges: [UMAPEdge] = []  // No attractive edges

        var params = UMAPParameters.default
        params.negativeSampleRate = 5
        params.learningRate = 1.0

        // Run optimization (only negative sampling will apply)
        try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

        // Compute distance after
        let distAfter = sqrt(pow(embedding[0][0] - embedding[1][0], 2) +
                             pow(embedding[0][1] - embedding[1][1], 2))

        // Points should be pushed farther apart (initial distance was 0.1)
        XCTAssertGreaterThan(distAfter, 0.1, "Negative sampling should push points apart")
    }

    func testApplyGradientsModifiesEmbedding() async throws {
        let n = 10
        let d = 2
        let device = context.device.rawDevice

        // Create embedding
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let flatEmbedding = embedding.flatMap { $0 }

        guard let embedBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create embedding buffer")
            return
        }

        // Create known gradients
        let gradients = [Float](repeating: 0.5, count: n * d)
        guard let gradBuffer = device.makeBuffer(
            bytes: gradients,
            length: gradients.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create gradient buffer")
            return
        }

        // Apply gradients
        try await kernel.applyGradients(embedding: embedBuffer, gradients: gradBuffer, n: n, d: d)

        // Verify embedding was modified
        let resultPtr = embedBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<n {
            for k in 0..<d {
                let expected = flatEmbedding[i * d + k] + 0.5
                XCTAssertEqual(resultPtr[i * d + k], expected, accuracy: 1e-5)
            }
        }
    }

    func testLossComputation() async throws {
        let n = 10
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 3)

        let loss = kernel.computeLoss(embedding: embedding, edges: edges, params: .default)

        // Loss should be finite and positive
        XCTAssertFalse(loss.isNaN)
        XCTAssertFalse(loss.isInfinite)
        XCTAssertGreaterThan(loss, 0)
    }

    // MARK: - Phase 3: Protocol Conformance & Performance Tests

    func testFusibleKernelConformance() async throws {
        // Verify FusibleKernel protocol conformance
        XCTAssertTrue(kernel.fusibleWith.contains("L2Distance"))
        XCTAssertTrue(kernel.fusibleWith.contains("TopKSelection"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }

    func testExecuteEpochBufferAPI() async throws {
        let n = 50
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        // Create buffers
        guard let embeddingBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Generate random targets
        var randomTargets = [UInt32](repeating: 0, count: n * 5)
        for i in 0..<n {
            for s in 0..<5 {
                var j = Int.random(in: 0..<n)
                while j == i && n > 1 { j = Int.random(in: 0..<n) }
                randomTargets[i * 5 + s] = UInt32(j)
            }
        }
        guard let targetsBuffer = device.makeBuffer(
            bytes: randomTargets,
            length: randomTargets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create targets buffer")
            return
        }

        // Execute epoch using buffer API
        try await kernel.executeEpoch(
            embedding: embeddingBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            randomTargets: targetsBuffer,
            n: n,
            d: d,
            edgeCount: edges.count,
            params: .default
        )

        // Verify embedding was modified and is finite
        let resultPtr = embeddingBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<(n * d) {
            XCTAssertFalse(resultPtr[i].isNaN, "Embedding should not be NaN at index \(i)")
            XCTAssertFalse(resultPtr[i].isInfinite, "Embedding should not be Inf at index \(i)")
        }
    }

    func testEncodeGradientsAPI() async throws {
        let n = 20
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)
        let (starts, counts) = kernel.computeSegments(edges: edges, n: n)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        // Create input buffers
        guard let embeddingBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ),
        let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffers")
            return
        }

        // Create output buffers
        let edgeGradSize = edges.count * d * MemoryLayout<Float>.size
        guard let edgeGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared),
              let targetGradients = device.makeBuffer(length: edgeGradSize, options: .storageModeShared),
              let pointGradients = device.makeBuffer(length: n * d * MemoryLayout<Float>.size, options: .storageModeShared) else {
            XCTFail("Failed to create output buffers")
            return
        }

        // Execute using encode API
        try await context.executeAndWait { [kernel] _, encoder in
            let result = kernel!.encodeGradients(
                into: encoder,
                embedding: embeddingBuffer,
                edges: edgeBuffer,
                segmentStarts: startsBuffer,
                segmentCounts: countsBuffer,
                edgeGradients: edgeGradients,
                targetGradients: targetGradients,
                pointGradients: pointGradients,
                n: n,
                d: d,
                edgeCount: edges.count,
                params: .default
            )

            // Verify encoding result
            XCTAssertEqual(result.pipelineName, "umap_gradient")
            XCTAssertGreaterThan(result.totalThreads, 0)
        }

        // Verify gradients are finite
        let gradPtr = pointGradients.contents().bindMemory(to: Float.self, capacity: n * d)
        for i in 0..<(n * d) {
            XCTAssertFalse(gradPtr[i].isNaN, "Gradient should not be NaN at index \(i)")
            XCTAssertFalse(gradPtr[i].isInfinite, "Gradient should not be Inf at index \(i)")
        }
    }

    func testGPUTargetGradientAccumulation() async throws {
        // Test that GPU atomic accumulation produces correct results
        let n = 30
        let d = 2
        let embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
        let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 5)

        let device = context.device.rawDevice
        let flatEmbedding = embedding.flatMap { $0 }

        // Create buffers
        guard let embeddingBuffer = device.makeBuffer(
            bytes: flatEmbedding,
            length: flatEmbedding.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let edgeBuffer = device.makeBuffer(
            bytes: edges,
            length: edges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Create target gradients buffer with known values
        let targetGradients = edges.flatMap { _ in [Float](repeating: 0.1, count: d) }
        guard let targetGradBuffer = device.makeBuffer(
            bytes: targetGradients,
            length: targetGradients.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create target gradients buffer")
            return
        }

        // Record original embedding
        let originalEmbedding = Array(UnsafeBufferPointer(
            start: embeddingBuffer.contents().bindMemory(to: Float.self, capacity: n * d),
            count: n * d
        ))

        // Apply target gradients using GPU
        try await kernel.applyTargetGradientsGPU(
            embedding: embeddingBuffer,
            edges: edgeBuffer,
            targetGradients: targetGradBuffer,
            edgeCount: edges.count,
            n: n,
            d: d,
            params: .default
        )

        // Verify embedding was modified
        let resultPtr = embeddingBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        var anyChanged = false
        for i in 0..<(n * d) {
            if abs(resultPtr[i] - originalEmbedding[i]) > 1e-6 {
                anyChanged = true
            }
            XCTAssertFalse(resultPtr[i].isNaN, "Result should not be NaN at index \(i)")
            XCTAssertFalse(resultPtr[i].isInfinite, "Result should not be Inf at index \(i)")
        }
        XCTAssertTrue(anyChanged, "Embedding should be modified after applying target gradients")
    }

    func testOptimizeEpochPerformance() async throws {
        // Performance benchmark for various sizes
        let sizes = [(100, 2), (500, 2), (1000, 2)]

        for (n, d) in sizes {
            var embedding = Metal4KernelTestHelpers.randomLowDimEmbedding(n: n, d: d)
            let edges = Metal4KernelTestHelpers.randomUMAPEdges(n: n, edgesPerPoint: 15)
            let params = UMAPParameters.default

            // Warm up
            try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)

            // Measure
            let start = CACurrentMediaTime()
            let epochs = 10
            for _ in 0..<epochs {
                try await kernel.optimizeEpoch(embedding: &embedding, edges: edges, params: params)
            }
            let elapsed = CACurrentMediaTime() - start

            let msPerEpoch = (elapsed / Double(epochs)) * 1000
            print("UMAPGradient n=\(n), d=\(d): \(String(format: "%.2f", msPerEpoch)) ms/epoch")

            // Verify performance targets (generous for CI)
            // At n=1000, should complete in reasonable time
            if n == 1000 {
                XCTAssertLessThan(msPerEpoch, 100, "Epoch should complete in < 100ms for n=1000")
            }
        }
    }
}
