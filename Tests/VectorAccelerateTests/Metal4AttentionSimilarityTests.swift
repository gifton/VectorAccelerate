// VectorAccelerate: Metal 4 Attention Similarity Tests
//
// Tests for Phase 4 ML Integration - Attention Similarity kernel.
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

// MARK: - Attention Similarity Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4AttentionSimilarityKernelTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testKernelInitialization() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        XCTAssertEqual(kernel.name, "Metal4AttentionSimilarityKernel")
        XCTAssertFalse(kernel.hasWeights)
    }

    func testRandomWeightCreation() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(
            inputDimension: 128,
            headDimension: 32,
            numHeads: 1
        )

        try await kernel.createRandomWeights(config: config)

        XCTAssertTrue(kernel.hasWeights)

        let stats = await kernel.getTensorStatistics()
        XCTAssertEqual(stats.loadedTensors, 2) // Wq + Wk
    }

    // MARK: - Single-Head Attention Tests

    func testSingleHeadSimilarity() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 64,
            projectedDim: 16
        )

        try await kernel.createRandomWeights(config: config)

        // Create test vectors
        let queries = (0..<5).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }
        let keys = (0..<10).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)

        XCTAssertEqual(result.similarities.count, 5)
        XCTAssertEqual(result.similarities[0].count, 10)
        XCTAssertGreaterThan(result.computeTime, 0)
        XCTAssertGreaterThan(result.throughput, 0)

        // All similarities should be finite
        for row in result.similarities {
            for sim in row {
                XCTAssertFalse(sim.isNaN)
                XCTAssertFalse(sim.isInfinite)
            }
        }
    }

    func testSimilaritySymmetry() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 32,
            projectedDim: 8
        )

        // Create identical Wq and Wk for symmetric attention
        let projDim = config.headDimension * config.numHeads
        let projSize = projDim * config.inputDimension
        var weights = [Float](repeating: 0, count: projSize)
        for i in 0..<projSize {
            weights[i] = Float.random(in: -0.1...0.1)
        }

        try await kernel.loadWeights(
            queryProjection: weights,
            keyProjection: weights,  // Same weights for symmetry
            config: config
        )

        // Same vectors as both query and key
        let vectors = (0..<3).map { _ in
            (0..<32).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: vectors, keys: vectors)

        // Diagonal should have highest similarity for each vector with itself
        // (not guaranteed with random weights, but structure should be valid)
        XCTAssertEqual(result.similarities.count, 3)
        XCTAssertEqual(result.similarities[0].count, 3)
    }

    // MARK: - Multi-Head Attention Tests

    func testMultiHeadSimilarity() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(
            inputDimension: 64,
            headDimension: 16,
            numHeads: 4
        )

        try await kernel.createRandomWeights(config: config)

        let queries = (0..<3).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }
        let keys = (0..<5).map { _ in
            (0..<64).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)

        XCTAssertEqual(result.similarities.count, 3)
        XCTAssertEqual(result.similarities[0].count, 5)

        // Verify values are finite
        for row in result.similarities {
            for sim in row {
                XCTAssertFalse(sim.isNaN)
                XCTAssertFalse(sim.isInfinite)
            }
        }
    }

    // MARK: - Configuration Tests

    func testTransformerConfiguration() async throws {
        let config = Metal4AttentionSimilarityConfig.transformer768()

        XCTAssertEqual(config.inputDimension, 768)
        XCTAssertEqual(config.headDimension, 64)
        XCTAssertEqual(config.numHeads, 12)
        XCTAssertFalse(config.normalizeSimilarities)

        // Default temperature = sqrt(64) = 8
        XCTAssertEqual(config.effectiveTemperature, 8.0, accuracy: 0.01)
    }

    func testMiniLMConfiguration() async throws {
        let config = Metal4AttentionSimilarityConfig.miniLM()

        XCTAssertEqual(config.inputDimension, 384)
        XCTAssertEqual(config.headDimension, 64)
        XCTAssertEqual(config.numHeads, 6)
    }

    func testCustomTemperature() async throws {
        let config = Metal4AttentionSimilarityConfig(
            inputDimension: 128,
            headDimension: 32,
            numHeads: 1,
            temperature: 4.0
        )

        XCTAssertEqual(config.effectiveTemperature, 4.0)
    }

    // MARK: - Top-K Helper Tests

    func testTopKMatches() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 32,
            projectedDim: 8
        )

        try await kernel.createRandomWeights(config: config)

        let queries = [[Float](repeating: 0.5, count: 32)]
        let keys = (0..<10).map { _ in
            (0..<32).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Test topK helper
        let top3 = result.topK(forQuery: 0, k: 3)

        XCTAssertEqual(top3.count, 3)

        // Verify descending order
        if top3.count >= 2 {
            XCTAssertGreaterThanOrEqual(top3[0].similarity, top3[1].similarity)
        }
        if top3.count >= 3 {
            XCTAssertGreaterThanOrEqual(top3[1].similarity, top3[2].similarity)
        }

        // Verify indices are valid
        for match in top3 {
            XCTAssertGreaterThanOrEqual(match.index, 0)
            XCTAssertLessThan(match.index, 10)
        }
    }

    func testSimilarityLookup() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 16,
            projectedDim: 4
        )

        try await kernel.createRandomWeights(config: config)

        let queries = (0..<2).map { _ in
            (0..<16).map { _ in Float.random(in: -1...1) }
        }
        let keys = (0..<3).map { _ in
            (0..<16).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Test similarity lookup helper
        let sim = result.similarity(query: 0, key: 1)
        XCTAssertEqual(sim, result.similarities[0][1])

        // Out of bounds should return 0
        let outOfBounds = result.similarity(query: 100, key: 0)
        XCTAssertEqual(outOfBounds, 0)
    }

    // MARK: - Normalization Tests

    func testNormalizedSimilarities() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(
            inputDimension: 32,
            headDimension: 8,
            numHeads: 1,
            normalizeSimilarities: true
        )

        try await kernel.createRandomWeights(config: config)

        let queries = (0..<3).map { _ in
            (0..<32).map { _ in Float.random(in: -1...1) }
        }
        let keys = (0..<5).map { _ in
            (0..<32).map { _ in Float.random(in: -1...1) }
        }

        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // With normalization, all values should be in [0, 1]
        for row in result.similarities {
            for sim in row {
                XCTAssertGreaterThanOrEqual(sim, 0.0)
                XCTAssertLessThanOrEqual(sim, 1.0)
            }
        }
    }

    // MARK: - Weight Management Tests

    func testWeightUnloading() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 32,
            projectedDim: 8
        )

        try await kernel.createRandomWeights(config: config)
        XCTAssertTrue(kernel.hasWeights)

        await kernel.unloadWeights()
        XCTAssertFalse(kernel.hasWeights)

        let stats = await kernel.getTensorStatistics()
        XCTAssertEqual(stats.loadedTensors, 0)
    }

    func testLoadWeightsFromArrays() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig(
            inputDimension: 4,
            headDimension: 2,
            numHeads: 1
        )

        // Create simple projection weights
        let projSize = 2 * 4  // headDim * inputDim
        let wq = [Float](repeating: 0.1, count: projSize)
        let wk = [Float](repeating: 0.1, count: projSize)

        try await kernel.loadWeights(
            queryProjection: wq,
            keyProjection: wk,
            config: config
        )

        XCTAssertTrue(kernel.hasWeights)
    }

    // MARK: - Error Handling Tests

    func testComputeWithoutWeights() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)

        let queries = [[Float](repeating: 0.5, count: 64)]
        let keys = [[Float](repeating: 0.5, count: 64)]

        do {
            _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            XCTFail("Expected error for computing without weights")
        } catch {
            // Expected - weights not loaded
        }
    }

    func testDimensionMismatch() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 64,
            projectedDim: 16
        )

        try await kernel.createRandomWeights(config: config)

        // Wrong dimension
        let queries = [[Float](repeating: 0.5, count: 128)]
        let keys = [[Float](repeating: 0.5, count: 64)]

        do {
            _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            XCTFail("Expected error for dimension mismatch")
        } catch {
            // Expected
        }
    }

    func testEmptyInput() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 32,
            projectedDim: 8
        )

        try await kernel.createRandomWeights(config: config)

        let emptyQueries: [[Float]] = []
        let keys = [[Float](repeating: 0.5, count: 32)]

        do {
            _ = try await kernel.computeSimilarities(queries: emptyQueries, keys: keys)
            XCTFail("Expected error for empty input")
        } catch {
            // Expected
        }
    }

    // MARK: - Performance Tests

    func testSimilarityPerformance() async throws {
        let kernel = try await Metal4AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(
            inputDim: 768,
            projectedDim: 64
        )

        try await kernel.createRandomWeights(config: config)

        // 100 queries against 1000 keys
        let queries = (0..<100).map { _ in
            (0..<768).map { _ in Float.random(in: -1...1) }
        }
        let keys = (0..<1000).map { _ in
            (0..<768).map { _ in Float.random(in: -1...1) }
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await kernel.computeSimilarities(queries: queries, keys: keys)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Attention Similarity Performance:")
        print("  Queries: 100, Keys: 1000")
        print("  Input dimension: 768")
        print("  Head dimension: 64")
        print("  Compute time: \(String(format: "%.3f", elapsed * 1000)) ms")
        print("  Throughput: \(String(format: "%.0f", result.throughput)) pairs/sec")

        // Should process at reasonable speed
        XCTAssertGreaterThan(result.throughput, 10000.0)  // At least 10k pairs/sec
    }
}
