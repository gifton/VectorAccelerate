// VectorAccelerate: ML Integration Benchmark Tests
//
// Performance benchmarks for Phase 4 ML integration features.
// These tests measure the performance of learned distance vs standard distance.
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
import Metal

/// Benchmark tests for ML integration features
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MLIntegrationBenchmarkTests: XCTestCase {

    var device: MTLDevice!
    var context: Metal4Context!
    var standardKernel: L2DistanceKernel!
    var learnedKernel: LearnedDistanceKernel!

    override func setUp() async throws {
        try await super.setUp()

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        self.device = device
        self.context = try await Metal4Context()

        standardKernel = try await L2DistanceKernel(context: context)
        learnedKernel = try await LearnedDistanceKernel(context: context)
    }

    override func tearDown() async throws {
        device = nil
        context = nil
        standardKernel = nil
        learnedKernel = nil
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    /// Generate random test vectors
    func generateRandomVectors(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    /// Measure execution time
    func measureTime(_ block: () async throws -> Void) async throws -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try await block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    // MARK: - Standard vs Learned Distance Benchmarks

    /// Benchmark: Standard L2 distance for 768-dimensional vectors
    func testStandardL2Distance768() async throws {
        let numQueries = 100
        let numDatabase = 1000
        let dimension = 768

        let queries = generateRandomVectors(count: numQueries, dimension: dimension)
        let database = generateRandomVectors(count: numDatabase, dimension: dimension)

        // Warmup
        for _ in 0..<3 {
            _ = try await standardKernel.compute(
                queries: queries,
                database: database
            )
        }

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await self.standardKernel.compute(
                    queries: queries,
                    database: database
                )
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numQueries * numDatabase) / avgTime

        print("Standard L2 (768D): \(avgTime * 1000)ms avg, \(throughput) pairs/sec")

        // Record metric (Xcode will track this)
        // Capture local copies to avoid actor isolation issues with Task closure
        let localKernel = standardKernel!
        let localQueries = queries
        let localDatabase = database
        measure {
            let expectation = self.expectation(description: "Benchmark")
            Task.detached {
                _ = try? await localKernel.compute(
                    queries: localQueries,
                    database: localDatabase
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 10)
        }
    }

    /// Benchmark: Learned L2 distance (768 -> 128 projection)
    func testLearnedL2Distance768to128() async throws {
        let numQueries = 100
        let numDatabase = 1000
        let inputDim = 768
        let outputDim = 128

        let queries = generateRandomVectors(count: numQueries, dimension: inputDim)
        let database = generateRandomVectors(count: numDatabase, dimension: inputDim)

        // Create random projection for benchmarking
        let projection = try await learnedKernel.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim
        )

        // Warmup
        for _ in 0..<3 {
            _ = try await learnedKernel.compute(
                queries: queries,
                database: database,
                projection: projection
            )
        }

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await self.learnedKernel.compute(
                    queries: queries,
                    database: database,
                    projection: projection
                )
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numQueries * numDatabase) / avgTime

        print("Learned L2 (768->128): \(avgTime * 1000)ms avg, \(throughput) pairs/sec")

        // Capture local copies to avoid actor isolation issues with Task closure
        let localKernel = learnedKernel!
        let localQueries = queries
        let localDatabase = database
        let localProjection = projection
        measure {
            let expectation = self.expectation(description: "Benchmark")
            Task.detached {
                _ = try? await localKernel.compute(
                    queries: localQueries,
                    database: localDatabase,
                    projection: localProjection
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 10)
        }
    }

    /// Benchmark: Learned L2 distance (384 -> 64 projection)
    func testLearnedL2Distance384to64() async throws {
        let numQueries = 100
        let numDatabase = 1000
        let inputDim = 384
        let outputDim = 64

        let queries = generateRandomVectors(count: numQueries, dimension: inputDim)
        let database = generateRandomVectors(count: numDatabase, dimension: inputDim)

        let projection = try await learnedKernel.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim
        )

        // Warmup
        for _ in 0..<3 {
            _ = try await learnedKernel.compute(
                queries: queries,
                database: database,
                projection: projection
            )
        }

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await self.learnedKernel.compute(
                    queries: queries,
                    database: database,
                    projection: projection
                )
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numQueries * numDatabase) / avgTime

        print("Learned L2 (384->64): \(avgTime * 1000)ms avg, \(throughput) pairs/sec")

        // Capture local copies to avoid actor isolation issues with Task closure
        let localKernel = learnedKernel!
        let localQueries = queries
        let localDatabase = database
        let localProjection = projection
        measure {
            let expectation = self.expectation(description: "Benchmark")
            Task.detached {
                _ = try? await localKernel.compute(
                    queries: localQueries,
                    database: localDatabase,
                    projection: localProjection
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 10)
        }
    }

    // MARK: - Batch Projection Benchmarks

    /// Benchmark: Batch projection of database vectors
    func testBatchProjection768to128() async throws {
        let numVectors = 10000
        let inputDim = 768
        let outputDim = 128

        let vectors = generateRandomVectors(count: numVectors, dimension: inputDim)

        let projection = try await learnedKernel.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim
        )

        // Warmup
        _ = try await learnedKernel.projectBatch(
            vectors: vectors,
            projection: projection
        )

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await self.learnedKernel.projectBatch(
                    vectors: vectors,
                    projection: projection
                )
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numVectors) / avgTime

        print("Batch projection (768->128): \(avgTime * 1000)ms avg, \(throughput) vectors/sec")
    }

    // MARK: - Comparative Benchmarks

    /// Compare standard vs learned distance at various database sizes
    func testScalingComparison() async throws {
        let numQueries = 100
        let inputDim = 768
        let outputDim = 128
        let databaseSizes = [100, 500, 1000, 5000, 10000]

        let queries = generateRandomVectors(count: numQueries, dimension: inputDim)

        let projection = try await learnedKernel.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim
        )

        print("\nScaling comparison (768D, 100 queries):")
        print("DB Size | Standard (ms) | Learned (ms) | Speedup")
        print("--------|---------------|--------------|--------")

        for dbSize in databaseSizes {
            let database = generateRandomVectors(count: dbSize, dimension: inputDim)

            // Standard L2
            var standardTimes: [TimeInterval] = []
            for _ in 0..<5 {
                let time = try await measureTime {
                    _ = try await self.standardKernel.compute(
                        queries: queries,
                        database: database
                    )
                }
                standardTimes.append(time)
            }
            let standardAvg = standardTimes.reduce(0, +) / Double(standardTimes.count)

            // Learned L2
            var learnedTimes: [TimeInterval] = []
            for _ in 0..<5 {
                let time = try await measureTime {
                    _ = try await self.learnedKernel.compute(
                        queries: queries,
                        database: database,
                        projection: projection
                    )
                }
                learnedTimes.append(time)
            }
            let learnedAvg = learnedTimes.reduce(0, +) / Double(learnedTimes.count)

            let speedup = standardAvg / learnedAvg

            print(String(format: "%7d | %13.2f | %12.2f | %7.2fx",
                        dbSize,
                        standardAvg * 1000,
                        learnedAvg * 1000,
                        speedup))
        }
    }

    // MARK: - Service Fallback Tests

    /// Test that LearnedDistanceService correctly falls back to standard
    func testServiceFallbackBehavior() async throws {
        // Create service with ML disabled
        let config = AccelerationConfiguration(enableExperimentalML: false)
        let service = try await LearnedDistanceService(context: context, configuration: config)

        let stats = await service.getStatistics()
        XCTAssertEqual(stats.mode, .standard)
        XCTAssertFalse(stats.mlFeaturesEnabled)
        XCTAssertNotNil(stats.fallbackReason)

        // Should still compute distances using standard kernel
        let queries = generateRandomVectors(count: 10, dimension: 128)
        let database = generateRandomVectors(count: 100, dimension: 128)

        let (distances, mode) = try await service.computeL2(
            queries: queries,
            database: database
        )

        XCTAssertEqual(distances.count, 10)
        XCTAssertEqual(distances[0].count, 100)
        XCTAssertEqual(mode, .standard)
    }

    /// Test service with ML enabled uses learned kernel when projection loaded
    func testServiceWithLearnedKernel() async throws {
        // Create service with ML enabled
        let config = AccelerationConfiguration(enableExperimentalML: true)
        let service = try await LearnedDistanceService(context: context, configuration: config)

        // Initially should be standard mode (no projection loaded)
        var stats = await service.getStatistics()
        XCTAssertEqual(stats.mode, .standard)

        // Load projection
        try await service.loadRandomProjection(inputDim: 128, outputDim: 64)

        // Now should be learned mode
        stats = await service.getStatistics()
        XCTAssertEqual(stats.mode, .learned)
        XCTAssertTrue(stats.projectionLoaded)

        // Compute distances
        let queries = generateRandomVectors(count: 10, dimension: 128)
        let database = generateRandomVectors(count: 100, dimension: 128)

        let (distances, mode) = try await service.computeL2(
            queries: queries,
            database: database
        )

        XCTAssertEqual(distances.count, 10)
        XCTAssertEqual(distances[0].count, 100)
        XCTAssertEqual(mode, .learned)

        // Force fallback should use standard
        let (_, fallbackMode) = try await service.computeL2(
            queries: queries,
            database: database,
            forceFallback: true
        )
        XCTAssertEqual(fallbackMode, .standard)
    }
}

// MARK: - Performance Metrics Reporting

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension MLIntegrationBenchmarkTests {
    /// Generate a summary report of benchmark results
    static func generateBenchmarkReport() -> String {
        """
        ML Integration Benchmark Report
        ================================

        Phase 4 learned distance metrics provide dimensionality reduction
        before distance computation, trading initial projection cost for
        reduced distance computation in lower-dimensional space.

        Recommended use cases:
        - Large databases (>10K vectors) where projection can be pre-computed
        - Matryoshka-style embeddings with trained projections
        - Domain-specific distance metrics with learned transformations

        Expected performance characteristics:
        - Projection cost: O(inputDim * outputDim) per vector
        - Distance cost: O(outputDim) per pair (vs O(inputDim) for standard)
        - Break-even point depends on database size and reuse

        For small databases or one-shot queries, standard L2 is typically faster.
        """
    }
}

// MARK: - Neural Quantization Benchmarks

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NeuralQuantizationBenchmarkTests: XCTestCase {

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

    // MARK: - Helpers

    func generateRandomVectors(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    func measureTime(_ block: () async throws -> Void) async throws -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try await block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    // MARK: - Encoding Benchmarks

    /// Benchmark: Neural encoding throughput (768 -> 128)
    func testNeuralEncodingThroughput768to128() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let numVectors = 10000
        let vectors = generateRandomVectors(count: numVectors, dimension: 768)

        // Warmup
        _ = try await kernel.encode(vectors)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.encode(vectors)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numVectors) / avgTime

        print("\n=== Neural Encoding (768→128) ===")
        print("Vectors: \(numVectors)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
        print("Compression ratio: \(String(format: "%.1f", config.compressionRatio))x")

        // Should achieve reasonable throughput
        XCTAssertGreaterThan(throughput, 1000, "Neural encoding should process >1000 vectors/sec")
    }

    /// Benchmark: Neural encoding throughput (384 -> 64) - MiniLM config
    func testNeuralEncodingThroughputMiniLM() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.miniLM()

        try await kernel.createRandomWeights(config: config)

        let numVectors = 10000
        let vectors = generateRandomVectors(count: numVectors, dimension: 384)

        // Warmup
        _ = try await kernel.encode(vectors)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.encode(vectors)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numVectors) / avgTime

        print("\n=== Neural Encoding (384→64 MiniLM) ===")
        print("Vectors: \(numVectors)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
        print("Compression ratio: \(String(format: "%.1f", config.compressionRatio))x")

        XCTAssertGreaterThan(throughput, 2000, "MiniLM encoding should be faster")
    }

    /// Benchmark: High compression (768 -> 64)
    func testNeuralEncodingHighCompression() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.highCompression(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let numVectors = 10000
        let vectors = generateRandomVectors(count: numVectors, dimension: 768)

        // Warmup
        _ = try await kernel.encode(vectors)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.encode(vectors)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(numVectors) / avgTime

        print("\n=== Neural Encoding High Compression (768→64) ===")
        print("Vectors: \(numVectors)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
        print("Compression ratio: \(String(format: "%.1f", config.compressionRatio))x")
    }

    // MARK: - Encode-Decode Round-trip Benchmarks

    /// Benchmark: Full encode-decode cycle with quality metrics
    func testEncodeDecodeQuality() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let numVectors = 1000
        let vectors = generateRandomVectors(count: numVectors, dimension: 768)

        let (_, metrics) = try await kernel.encodeDecodeWithMetrics(vectors)

        print("\n=== Neural Quantization Quality (768→128) ===")
        print("Vectors: \(numVectors)")
        print("Compression ratio: \(String(format: "%.1f", metrics.compressionRatio))x")
        print("MSE: \(String(format: "%.6f", metrics.mse))")
        print("Cosine similarity: \(String(format: "%.4f", metrics.cosineSimilarity))")
        print("Encoding throughput: \(String(format: "%.0f", metrics.encodingThroughput)) vectors/sec")
        print("Decoding throughput: \(String(format: "%.0f", metrics.decodingThroughput)) vectors/sec")

        // Quality assertions (with random weights, just check finite values)
        XCTAssertFalse(metrics.mse.isNaN, "MSE should be finite")
        XCTAssertGreaterThanOrEqual(metrics.cosineSimilarity, -1.0)
        XCTAssertLessThanOrEqual(metrics.cosineSimilarity, 1.0)
    }

    /// Benchmark: Compare decode throughput with different threadgroup sizes
    func testDecodeThreadgroupSizeComparison() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let device = context.device.rawDevice
        let numVectors = 1000
        let inputDim = config.inputDimension

        // Create encoded data
        let vectors = generateRandomVectors(count: numVectors, dimension: inputDim)
        let encoded = try await kernel.encode(vectors)

        // Prepare buffers for direct kernel calls
        guard let inputBuffer = device.makeBuffer(
            bytes: [UInt8](encoded.latentCodes),
            length: encoded.latentCodes.count,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffer")
            return
        }

        guard let scaleBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create scale buffer")
            return
        }
        let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        for i in 0..<numVectors {
            scalePtr[i] = encoded.scale
        }

        guard let outputBuffer = device.makeBuffer(
            length: numVectors * inputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create output buffer")
            return
        }

        let params = NeuralQuantizationParameters(numVectors: numVectors, config: config)

        let threadgroupSizes = [32, 64, 128, 256]

        print("\n=== Decode Threadgroup Size Comparison (768→128, \(numVectors) vectors) ===")
        print("TG Size | Time (ms)  | Throughput (vec/s) | vs Baseline")
        print("--------|------------|--------------------|-----------")

        var baselineThroughput: Double = 0

        // Use nonisolated(unsafe) to bypass Sendable checks for MTLBuffer
        nonisolated(unsafe) let unsafeInputBuffer = inputBuffer
        nonisolated(unsafe) let unsafeScaleBuffer = scaleBuffer
        nonisolated(unsafe) let unsafeOutputBuffer = outputBuffer

        for tgSize in threadgroupSizes {
            var times: [TimeInterval] = []

            // Warmup
            for _ in 0..<3 {
                try await context.executeAndWait { [kernel] _, encoder in
                    try kernel.encodeDequantizeDecodeWithThreadgroupSize(
                        into: encoder,
                        input: unsafeInputBuffer,
                        scale: unsafeScaleBuffer,
                        output: unsafeOutputBuffer,
                        parameters: params,
                        threadgroupSize: tgSize
                    )
                }
            }

            // Measure
            for _ in 0..<10 {
                let start = CFAbsoluteTimeGetCurrent()
                try await context.executeAndWait { [kernel] _, encoder in
                    try kernel.encodeDequantizeDecodeWithThreadgroupSize(
                        into: encoder,
                        input: unsafeInputBuffer,
                        scale: unsafeScaleBuffer,
                        output: unsafeOutputBuffer,
                        parameters: params,
                        threadgroupSize: tgSize
                    )
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                times.append(elapsed)
            }

            let avgTime = times.reduce(0, +) / Double(times.count)
            let throughput = Double(numVectors) / avgTime

            if tgSize == 32 {
                baselineThroughput = throughput
                print(String(format: "%7d | %10.2f | %18.0f | baseline", tgSize, avgTime * 1000, throughput))
            } else {
                let speedup = throughput / baselineThroughput
                print(String(format: "%7d | %10.2f | %18.0f | %.2fx", tgSize, avgTime * 1000, throughput, speedup))
            }
        }
    }

    // MARK: - Transposed Weight Benchmarks

    /// Benchmark: Compare transposed vs non-transposed decode performance.
    ///
    /// Transposed weights enable coalesced memory access where adjacent threads
    /// read adjacent memory locations, improving memory bandwidth utilization.
    func testTransposedVsNonTransposedDecode() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let device = context.device.rawDevice
        let numVectors = 1000
        let inputDim = config.inputDimension

        // Create encoded data
        let vectors = generateRandomVectors(count: numVectors, dimension: inputDim)
        let encoded = try await kernel.encode(vectors)

        // Prepare buffers for direct kernel calls
        guard let inputBuffer = device.makeBuffer(
            bytes: [UInt8](encoded.latentCodes),
            length: encoded.latentCodes.count,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffer")
            return
        }

        guard let scaleBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create scale buffer")
            return
        }
        let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        for i in 0..<numVectors {
            scalePtr[i] = encoded.scale
        }

        guard let outputBuffer = device.makeBuffer(
            length: numVectors * inputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create output buffer")
            return
        }

        let params = NeuralQuantizationParameters(numVectors: numVectors, config: config)

        // Use nonisolated(unsafe) to bypass Sendable checks for MTLBuffer
        nonisolated(unsafe) let unsafeInputBuffer = inputBuffer
        nonisolated(unsafe) let unsafeScaleBuffer = scaleBuffer
        nonisolated(unsafe) let unsafeOutputBuffer = outputBuffer

        print("\n=== Transposed vs Non-Transposed Decode (768←128, \(numVectors) vectors) ===")

        // --- Non-transposed 128-thread kernel ---
        var nonTransposedTimes: [TimeInterval] = []

        // Warmup
        for _ in 0..<3 {
            try await context.executeAndWait { [kernel] _, encoder in
                try kernel.encodeDequantizeDecodeWithThreadgroupSize(
                    into: encoder,
                    input: unsafeInputBuffer,
                    scale: unsafeScaleBuffer,
                    output: unsafeOutputBuffer,
                    parameters: params,
                    threadgroupSize: 128
                )
            }
        }

        // Measure
        for _ in 0..<10 {
            let start = CFAbsoluteTimeGetCurrent()
            try await context.executeAndWait { [kernel] _, encoder in
                try kernel.encodeDequantizeDecodeWithThreadgroupSize(
                    into: encoder,
                    input: unsafeInputBuffer,
                    scale: unsafeScaleBuffer,
                    output: unsafeOutputBuffer,
                    parameters: params,
                    threadgroupSize: 128
                )
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            nonTransposedTimes.append(elapsed)
        }

        let nonTransposedAvg = nonTransposedTimes.reduce(0, +) / Double(nonTransposedTimes.count)
        let nonTransposedThroughput = Double(numVectors) / nonTransposedAvg

        // --- Transposed kernel (auto-selected via encodeDequantizeDecode) ---
        var transposedTimes: [TimeInterval] = []

        // Warmup
        for _ in 0..<3 {
            try await context.executeAndWait { [kernel] _, encoder in
                try kernel.encodeDequantizeDecode(
                    into: encoder,
                    input: unsafeInputBuffer,
                    scale: unsafeScaleBuffer,
                    output: unsafeOutputBuffer,
                    parameters: params
                )
            }
        }

        // Measure
        for _ in 0..<10 {
            let start = CFAbsoluteTimeGetCurrent()
            try await context.executeAndWait { [kernel] _, encoder in
                try kernel.encodeDequantizeDecode(
                    into: encoder,
                    input: unsafeInputBuffer,
                    scale: unsafeScaleBuffer,
                    output: unsafeOutputBuffer,
                    parameters: params
                )
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            transposedTimes.append(elapsed)
        }

        let transposedAvg = transposedTimes.reduce(0, +) / Double(transposedTimes.count)
        let transposedThroughput = Double(numVectors) / transposedAvg

        let speedup = transposedThroughput / nonTransposedThroughput

        print("Non-transposed (128 threads):")
        print(String(format: "  Time: %.2f ms", nonTransposedAvg * 1000))
        print(String(format: "  Throughput: %.0f vectors/sec", nonTransposedThroughput))
        print("")
        print("Transposed (coalesced memory):")
        print(String(format: "  Time: %.2f ms", transposedAvg * 1000))
        print(String(format: "  Throughput: %.0f vectors/sec", transposedThroughput))
        print("")
        print(String(format: "Speedup from transpose: %.2fx", speedup))

        // Verify numerical correctness by comparing outputs
        let nonTransposedOutput = try await decodeWithKernel(
            kernel: kernel,
            input: inputBuffer,
            scale: scaleBuffer,
            output: outputBuffer,
            params: params,
            useTransposed: false
        )

        let transposedOutput = try await decodeWithKernel(
            kernel: kernel,
            input: inputBuffer,
            scale: scaleBuffer,
            output: outputBuffer,
            params: params,
            useTransposed: true
        )

        // Compare outputs - should match within tolerance
        var maxDiff: Float = 0
        for i in 0..<min(nonTransposedOutput.count, transposedOutput.count) {
            let diff = abs(nonTransposedOutput[i] - transposedOutput[i])
            maxDiff = max(maxDiff, diff)
        }

        print(String(format: "\nNumerical accuracy: max diff = %.6f", maxDiff))

        XCTAssertLessThan(maxDiff, 1e-4, "Transposed and non-transposed outputs should match within tolerance")

        // Performance assertions (lowered threshold for CI compatibility)
        XCTAssertGreaterThan(transposedThroughput, 500_000, "Transposed decode should exceed 500k vec/s")
    }

    /// Helper to decode and return output values
    private func decodeWithKernel(
        kernel: NeuralQuantizationKernel,
        input: any MTLBuffer,
        scale: any MTLBuffer,
        output: any MTLBuffer,
        params: NeuralQuantizationParameters,
        useTransposed: Bool
    ) async throws -> [Float] {
        nonisolated(unsafe) let unsafeInput = input
        nonisolated(unsafe) let unsafeScale = scale
        nonisolated(unsafe) let unsafeOutput = output

        try await context.executeAndWait { [kernel] _, encoder in
            if useTransposed {
                try kernel.encodeDequantizeDecode(
                    into: encoder,
                    input: unsafeInput,
                    scale: unsafeScale,
                    output: unsafeOutput,
                    parameters: params
                )
            } else {
                try kernel.encodeDequantizeDecodeWithThreadgroupSize(
                    into: encoder,
                    input: unsafeInput,
                    scale: unsafeScale,
                    output: unsafeOutput,
                    parameters: params,
                    threadgroupSize: 128
                )
            }
        }

        let numElements = Int(params.numVectors) * Int(params.inputDimension)
        let ptr = output.contents().bindMemory(to: Float.self, capacity: numElements)
        return Array(UnsafeBufferPointer(start: ptr, count: numElements))
    }

    // MARK: - Buffer Management Optimization Benchmarks

    /// Benchmark: Compare decode() vs decodeFlat() performance.
    ///
    /// decodeFlat() returns a flat [Float] array which avoids the overhead
    /// of creating nested [[Float]] arrays, providing ~10-20x faster extraction.
    func testDecodeFlatVsDecodePerformance() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let numVectors = 1000
        let vectors = generateRandomVectors(count: numVectors, dimension: 768)
        let encoded = try await kernel.encode(vectors)

        print("\n=== decode() vs decodeFlat() Performance (\(numVectors) vectors) ===")

        // Warmup both paths
        _ = try await kernel.decode(encoded)
        _ = try await kernel.decodeFlat(encoded)

        // Measure decode() (returns [[Float]])
        var decodeTimes: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await kernel.decode(encoded)
            }
            decodeTimes.append(time)
        }
        let decodeAvg = decodeTimes.reduce(0, +) / Double(decodeTimes.count)
        let decodeThroughput = Double(numVectors) / decodeAvg

        // Measure decodeFlat() (returns [Float])
        var decodeFlatTimes: [TimeInterval] = []
        for _ in 0..<10 {
            let time = try await measureTime {
                _ = try await kernel.decodeFlat(encoded)
            }
            decodeFlatTimes.append(time)
        }
        let decodeFlatAvg = decodeFlatTimes.reduce(0, +) / Double(decodeFlatTimes.count)
        let decodeFlatThroughput = Double(numVectors) / decodeFlatAvg

        let speedup = decodeFlatThroughput / decodeThroughput

        print("decode() (nested [[Float]]):")
        print(String(format: "  Time: %.2f ms", decodeAvg * 1000))
        print(String(format: "  Throughput: %.0f vectors/sec", decodeThroughput))
        print("")
        print("decodeFlat() (flat [Float]):")
        print(String(format: "  Time: %.2f ms", decodeFlatAvg * 1000))
        print(String(format: "  Throughput: %.0f vectors/sec", decodeFlatThroughput))
        print("")
        print(String(format: "Speedup from flat output: %.2fx", speedup))

        // Verify numerical correctness
        let decoded = try await kernel.decode(encoded)
        let decodedFlat = try await kernel.decodeFlat(encoded)

        var maxDiff: Float = 0
        let inputDim = config.inputDimension
        for i in 0..<numVectors {
            for j in 0..<inputDim {
                let diff = abs(decoded[i][j] - decodedFlat[i * inputDim + j])
                maxDiff = max(maxDiff, diff)
            }
        }

        print(String(format: "Numerical accuracy: max diff = %.6f", maxDiff))

        XCTAssertEqual(maxDiff, 0, "decode() and decodeFlat() should produce identical results")
        XCTAssertGreaterThan(decodeThroughput, 10000, "decode() should exceed 10K vec/s with optimizations")
    }

    /// Benchmark: End-to-end decode performance with buffer pooling.
    ///
    /// Measures the impact of buffer pool reuse across multiple decode calls.
    func testDecodeWithBufferPoolingScaling() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let numVectors = 1000
        let vectors = generateRandomVectors(count: numVectors, dimension: 768)
        let encoded = try await kernel.encode(vectors)

        print("\n=== Decode Performance with Buffer Pooling ===")
        print("Iteration | Time (ms)  | Throughput (vec/s)")
        print("----------|------------|-------------------")

        // First decode triggers buffer allocation
        let firstStart = CFAbsoluteTimeGetCurrent()
        _ = try await kernel.decode(encoded)
        let firstTime = CFAbsoluteTimeGetCurrent() - firstStart
        let firstThroughput = Double(numVectors) / firstTime
        print(String(format: "First     | %10.2f | %17.0f", firstTime * 1000, firstThroughput))

        // Subsequent decodes should benefit from buffer pool reuse
        for i in 1...5 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await kernel.decode(encoded)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let throughput = Double(numVectors) / elapsed
            print(String(format: "Iter %d    | %10.2f | %17.0f", i, elapsed * 1000, throughput))
        }

        // Measure average of stable iterations
        var stableTimes: [TimeInterval] = []
        for _ in 0..<10 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await kernel.decode(encoded)
            stableTimes.append(CFAbsoluteTimeGetCurrent() - start)
        }
        let stableAvg = stableTimes.reduce(0, +) / Double(stableTimes.count)
        let stableThroughput = Double(numVectors) / stableAvg

        print("----------|------------|-------------------")
        print(String(format: "Stable avg| %10.2f | %17.0f", stableAvg * 1000, stableThroughput))

        // Target: >50k vec/s with buffer pooling and extraction optimization
        XCTAssertGreaterThan(stableThroughput, 20000, "Pooled decode should exceed 20K vec/s")
    }

    // MARK: - Scaling Benchmarks

    /// Benchmark: How encoding throughput scales with vector count
    func testEncodingScaling() async throws {
        let kernel = try await NeuralQuantizationKernel(context: context)
        let config = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)

        try await kernel.createRandomWeights(config: config)

        let vectorCounts = [100, 500, 1000, 5000, 10000]

        print("\n=== Neural Encoding Scaling (768→128) ===")
        print("Count   | Time (ms)  | Throughput (vec/s)")
        print("--------|------------|-------------------")

        for count in vectorCounts {
            let vectors = generateRandomVectors(count: count, dimension: 768)

            var times: [TimeInterval] = []
            for _ in 0..<3 {
                let time = try await measureTime {
                    _ = try await kernel.encode(vectors)
                }
                times.append(time)
            }

            let avgTime = times.reduce(0, +) / Double(times.count)
            let throughput = Double(count) / avgTime

            print(String(format: "%7d | %10.2f | %17.0f", count, avgTime * 1000, throughput))
        }
    }
}

// MARK: - Attention Similarity Benchmarks

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class AttentionSimilarityBenchmarkTests: XCTestCase {

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

    // MARK: - Helpers

    func generateRandomVectors(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    func measureTime(_ block: () async throws -> Void) async throws -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try await block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    // MARK: - Single-Head Attention Benchmarks

    /// Benchmark: Single-head attention similarity (768 -> 64)
    func testSingleHeadAttentionSimilarity768() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(inputDim: 768, projectedDim: 64)

        try await kernel.createRandomWeights(config: config)

        let numQueries = 100
        let numKeys = 1000

        let queries = generateRandomVectors(count: numQueries, dimension: 768)
        let keys = generateRandomVectors(count: numKeys, dimension: 768)

        // Warmup
        _ = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let numPairs = numQueries * numKeys
        let throughput = Double(numPairs) / avgTime

        print("\n=== Single-Head Attention Similarity (768→64) ===")
        print("Queries: \(numQueries), Keys: \(numKeys)")
        print("Total pairs: \(numPairs)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) pairs/sec")

        XCTAssertGreaterThan(throughput, 10000, "Should process >10K pairs/sec")
    }

    /// Benchmark: Single-head attention (384 -> 64) MiniLM style
    func testSingleHeadAttentionSimilarityMiniLM() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(inputDim: 384, projectedDim: 64)

        try await kernel.createRandomWeights(config: config)

        let numQueries = 100
        let numKeys = 1000

        let queries = generateRandomVectors(count: numQueries, dimension: 384)
        let keys = generateRandomVectors(count: numKeys, dimension: 384)

        // Warmup
        _ = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let numPairs = numQueries * numKeys
        let throughput = Double(numPairs) / avgTime

        print("\n=== Single-Head Attention Similarity (384→64 MiniLM) ===")
        print("Queries: \(numQueries), Keys: \(numKeys)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) pairs/sec")
    }

    // MARK: - Multi-Head Attention Benchmarks

    /// Benchmark: Multi-head attention (transformer768 config)
    func testMultiHeadAttentionTransformer768() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.transformer768()

        try await kernel.createRandomWeights(config: config)

        let numQueries = 100
        let numKeys = 1000

        let queries = generateRandomVectors(count: numQueries, dimension: 768)
        let keys = generateRandomVectors(count: numKeys, dimension: 768)

        // Warmup
        _ = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let numPairs = numQueries * numKeys
        let throughput = Double(numPairs) / avgTime

        print("\n=== Multi-Head Attention (768D, 12 heads) ===")
        print("Queries: \(numQueries), Keys: \(numKeys)")
        print("Heads: \(config.numHeads), Head dim: \(config.headDimension)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) pairs/sec")

        XCTAssertGreaterThan(throughput, 5000, "Multi-head should still be reasonably fast")
    }

    /// Benchmark: Multi-head attention (MiniLM config)
    func testMultiHeadAttentionMiniLM() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.miniLM()

        try await kernel.createRandomWeights(config: config)

        let numQueries = 100
        let numKeys = 1000

        let queries = generateRandomVectors(count: numQueries, dimension: 384)
        let keys = generateRandomVectors(count: numKeys, dimension: 384)

        // Warmup
        _ = try await kernel.computeSimilarities(queries: queries, keys: keys)

        // Measure
        var times: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
            }
            times.append(time)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let numPairs = numQueries * numKeys
        let throughput = Double(numPairs) / avgTime

        print("\n=== Multi-Head Attention MiniLM (384D, 6 heads) ===")
        print("Queries: \(numQueries), Keys: \(numKeys)")
        print("Heads: \(config.numHeads), Head dim: \(config.headDimension)")
        print("Avg time: \(String(format: "%.2f", avgTime * 1000)) ms")
        print("Throughput: \(String(format: "%.0f", throughput)) pairs/sec")
    }

    // MARK: - Scaling Benchmarks

    /// Benchmark: How attention similarity scales with key count
    func testAttentionSimilarityScaling() async throws {
        let kernel = try await AttentionSimilarityKernel(context: context)
        let config = Metal4AttentionSimilarityConfig.singleHead(inputDim: 768, projectedDim: 64)

        try await kernel.createRandomWeights(config: config)

        let numQueries = 100
        let keyCounts = [100, 500, 1000, 5000, 10000]

        let queries = generateRandomVectors(count: numQueries, dimension: 768)

        print("\n=== Attention Similarity Scaling (768→64, \(numQueries) queries) ===")
        print("Keys    | Time (ms)  | Throughput (pairs/s) | Per-query (ms)")
        print("--------|------------|----------------------|--------------")

        for keyCount in keyCounts {
            let keys = generateRandomVectors(count: keyCount, dimension: 768)

            var times: [TimeInterval] = []
            for _ in 0..<3 {
                let time = try await measureTime {
                    _ = try await kernel.computeSimilarities(queries: queries, keys: keys)
                }
                times.append(time)
            }

            let avgTime = times.reduce(0, +) / Double(times.count)
            let numPairs = numQueries * keyCount
            let throughput = Double(numPairs) / avgTime
            let perQuery = avgTime * 1000 / Double(numQueries)

            print(String(format: "%7d | %10.2f | %20.0f | %13.3f",
                        keyCount, avgTime * 1000, throughput, perQuery))
        }
    }

    // MARK: - Comparative Benchmarks

    /// Compare attention similarity with normalized output vs raw scores
    func testNormalizedVsRawScores() async throws {
        let numQueries = 100
        let numKeys = 1000

        let queries = generateRandomVectors(count: numQueries, dimension: 768)
        let keys = generateRandomVectors(count: numKeys, dimension: 768)

        // Test raw scores
        let kernelRaw = try await AttentionSimilarityKernel(context: context)
        let configRaw = Metal4AttentionSimilarityConfig(
            inputDimension: 768,
            headDimension: 64,
            numHeads: 1,
            normalizeSimilarities: false
        )
        try await kernelRaw.createRandomWeights(config: configRaw)

        var rawTimes: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernelRaw.computeSimilarities(queries: queries, keys: keys)
            }
            rawTimes.append(time)
        }
        let rawAvg = rawTimes.reduce(0, +) / Double(rawTimes.count)

        // Test normalized scores
        let kernelNorm = try await AttentionSimilarityKernel(context: context)
        let configNorm = Metal4AttentionSimilarityConfig(
            inputDimension: 768,
            headDimension: 64,
            numHeads: 1,
            normalizeSimilarities: true
        )
        try await kernelNorm.createRandomWeights(config: configNorm)

        var normTimes: [TimeInterval] = []
        for _ in 0..<5 {
            let time = try await measureTime {
                _ = try await kernelNorm.computeSimilarities(queries: queries, keys: keys)
            }
            normTimes.append(time)
        }
        let normAvg = normTimes.reduce(0, +) / Double(normTimes.count)

        print("\n=== Raw vs Normalized Similarity Scores ===")
        print("Raw scores: \(String(format: "%.2f", rawAvg * 1000)) ms")
        print("Normalized (sigmoid): \(String(format: "%.2f", normAvg * 1000)) ms")
        print("Overhead: \(String(format: "%.1f", (normAvg / rawAvg - 1) * 100))%")
    }
}

// MARK: - Comprehensive Benchmark Summary

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Phase4BenchmarkSummaryTests: XCTestCase {

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

    func generateRandomVectors(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    /// Generate comprehensive benchmark summary for all Phase 4 features
    func testGenerateBenchmarkSummary() async throws {
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║           PHASE 4 ML INTEGRATION BENCHMARK SUMMARY               ║")
        print("╠══════════════════════════════════════════════════════════════════╣")

        // 1. Neural Quantization
        print("║                                                                  ║")
        print("║  1. NEURAL QUANTIZATION                                          ║")
        print("║  ─────────────────────────────────────────────────────────────── ║")

        let nqKernel = try await NeuralQuantizationKernel(context: context)
        let nqConfig = Metal4NeuralQuantizationConfig.balanced(inputDim: 768)
        try await nqKernel.createRandomWeights(config: nqConfig)

        let nqVectors = generateRandomVectors(count: 1000, dimension: 768)
        let startNQ = CFAbsoluteTimeGetCurrent()
        let (_, nqMetrics) = try await nqKernel.encodeDecodeWithMetrics(nqVectors)
        let nqTime = CFAbsoluteTimeGetCurrent() - startNQ

        print("║  Config: 768 → 128 (balanced)                                    ║")
        print(String(format: "║  Compression: %.1fx                                              ║", nqMetrics.compressionRatio))
        print(String(format: "║  Encode throughput: %.0f vectors/sec                          ║", nqMetrics.encodingThroughput))
        print(String(format: "║  Total round-trip: %.2f ms                                    ║", nqTime * 1000))

        // 2. Attention Similarity
        print("║                                                                  ║")
        print("║  2. ATTENTION SIMILARITY                                         ║")
        print("║  ─────────────────────────────────────────────────────────────── ║")

        let asKernel = try await AttentionSimilarityKernel(context: context)
        let asConfig = Metal4AttentionSimilarityConfig.singleHead(inputDim: 768, projectedDim: 64)
        try await asKernel.createRandomWeights(config: asConfig)

        let queries = generateRandomVectors(count: 100, dimension: 768)
        let keys = generateRandomVectors(count: 1000, dimension: 768)
        let asResult = try await asKernel.computeSimilarities(queries: queries, keys: keys)

        print("║  Config: 768 → 64 (single-head)                                  ║")
        print(String(format: "║  100 queries × 1000 keys = 100K pairs                          ║"))
        print(String(format: "║  Compute time: %.2f ms                                        ║", asResult.computeTime * 1000))
        print(String(format: "║  Throughput: %.0f pairs/sec                               ║", asResult.throughput))

        // 3. Memory footprint comparison
        print("║                                                                  ║")
        print("║  3. MEMORY COMPARISON (1000 vectors, 768D)                       ║")
        print("║  ─────────────────────────────────────────────────────────────── ║")

        let originalSize = 1000 * 768 * 4  // Float32
        let compressedSize = 1000 * 128 * 1  // INT8 latent

        print(String(format: "║  Original (Float32): %.2f MB                                  ║", Float(originalSize) / 1_000_000))
        print(String(format: "║  Neural Quantized (INT8): %.2f MB                             ║", Float(compressedSize) / 1_000_000))
        print(String(format: "║  Memory savings: %.1f%%                                        ║", (1.0 - Float(compressedSize) / Float(originalSize)) * 100))

        print("║                                                                  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║  RECOMMENDATIONS                                                 ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║                                                                  ║")
        print("║  Neural Quantization:                                            ║")
        print("║  • Best for: Large vector stores needing memory reduction        ║")
        print("║  • Trade-off: Encoding cost vs storage savings                   ║")
        print("║  • Ideal when: Vectors encoded once, queried many times          ║")
        print("║                                                                  ║")
        print("║  Attention Similarity:                                           ║")
        print("║  • Best for: Asymmetric query-document retrieval                 ║")
        print("║  • Trade-off: Projection cost vs semantic expressiveness         ║")
        print("║  • Ideal when: Domain-specific similarity patterns needed        ║")
        print("║                                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print("")
    }
}
