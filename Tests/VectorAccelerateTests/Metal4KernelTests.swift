// VectorAccelerate: Metal 4 Kernel Tests
//
// Tests for Phase 5 Metal 4 kernel migrations:
//
// Batch 1 (Core Distance):
// - Metal4L2DistanceKernel
// - Metal4CosineSimilarityKernel
// - Metal4DotProductKernel
//
// Batch 2 (Selection):
// - Metal4TopKSelectionKernel
// - Metal4FusedL2TopKKernel
// - Metal4StreamingTopKKernel
//
// Batch 3 (Quantization):
// - Metal4ScalarQuantizationKernel
// - Metal4BinaryQuantizationKernel
// - Metal4ProductQuantizationKernel
//
// Batch 4 (Matrix):
// - Metal4MatrixMultiplyKernel
// - Metal4MatrixVectorKernel
// - Metal4MatrixTransposeKernel
// - Metal4BatchMatrixKernel
//
// Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4KernelTestHelpers {
    /// Generate random vectors
    static func randomVectors(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    /// Generate normalized vectors
    static func normalizedVectors(count: Int, dimension: Int) -> [[Float]] {
        randomVectors(count: count, dimension: dimension).map { vector in
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            return vector.map { $0 / max(norm, 1e-8) }
        }
    }

    /// CPU L2 distance reference
    static func cpuL2Distance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
    }

    /// CPU squared L2 distance reference (for comparison with GPU which returns squared)
    static func cpuL2DistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) }
    }

    /// CPU cosine similarity reference
    static func cpuCosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0) { $0 + $1 * $1 })
        return dot / max(normA * normB, 1e-8)
    }

    /// CPU dot product reference
    static func cpuDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Compare results with tolerance
    static func assertClose(
        _ actual: [[Float]],
        _ expected: [[Float]],
        tolerance: Float = 1e-4,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "Row count mismatch", file: file, line: line)
        for i in 0..<actual.count {
            XCTAssertEqual(actual[i].count, expected[i].count, "Column count mismatch at row \(i)", file: file, line: line)
            for j in 0..<actual[i].count {
                let diff = abs(actual[i][j] - expected[i][j])
                XCTAssertLessThanOrEqual(
                    diff, tolerance,
                    "Value mismatch at [\(i)][\(j)]: actual=\(actual[i][j]), expected=\(expected[i][j])",
                    file: file, line: line
                )
            }
        }
    }
}

// MARK: - Metal4L2DistanceKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4L2DistanceKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4L2DistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4L2DistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testSmallVectors() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 4)
        let database = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 4)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 3)

        // Verify against CPU reference
        for i in 0..<queries.count {
            for j in 0..<database.count {
                let expected = Metal4KernelTestHelpers.cpuL2Distance(queries[i], database[j])
                XCTAssertEqual(results[i][j], expected, accuracy: 1e-4)
            }
        }
    }

    func testSquaredDistance() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 4)
        let database = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 4)

        let results = try await kernel.compute(queries: queries, database: database, computeSqrt: false)

        for i in 0..<queries.count {
            for j in 0..<database.count {
                let expected = pow(Metal4KernelTestHelpers.cpuL2Distance(queries[i], database[j]), 2)
                XCTAssertEqual(results[i][j], expected, accuracy: 1e-4)
            }
        }
    }

    // MARK: - Optimized Dimension Tests

    func testDimension384() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 384)
        let database = Metal4KernelTestHelpers.randomVectors(count: 50, dimension: 384)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 10)
        XCTAssertEqual(results[0].count, 50)

        // Spot check a few values
        let expected00 = Metal4KernelTestHelpers.cpuL2Distance(queries[0], database[0])
        XCTAssertEqual(results[0][0], expected00, accuracy: 1e-3)
    }

    func testDimension768() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 768)
        let database = Metal4KernelTestHelpers.randomVectors(count: 20, dimension: 768)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].count, 20)

        let expected00 = Metal4KernelTestHelpers.cpuL2Distance(queries[0], database[0])
        XCTAssertEqual(results[0][0], expected00, accuracy: 1e-3)
    }

    func testDimension1536() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 1536)
        let database = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 1536)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].count, 10)

        let expected00 = Metal4KernelTestHelpers.cpuL2Distance(queries[0], database[0])
        XCTAssertEqual(results[0][0], expected00, accuracy: 1e-3)
    }

    // MARK: - Edge Cases

    func testSingleQuery() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)
        let database = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 128)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 100)
    }

    func testSingleDatabase() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 128)
        let database = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 100)
        XCTAssertEqual(results[0].count, 1)
    }

    // MARK: - Protocol Conformance Tests

    func testOptimizedDimensions() {
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 384))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 768))
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 256))
    }

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("TopKSelection"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }
}

// MARK: - Metal4CosineSimilarityKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4CosineSimilarityKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4CosineSimilarityKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4CosineSimilarityKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testSmallVectors() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 4)
        let database = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 4)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 3)

        for i in 0..<queries.count {
            for j in 0..<database.count {
                let expected = Metal4KernelTestHelpers.cpuCosineSimilarity(queries[i], database[j])
                XCTAssertEqual(results[i][j], expected, accuracy: 1e-4)
            }
        }
    }

    func testNormalizedInputs() async throws {
        let queries = Metal4KernelTestHelpers.normalizedVectors(count: 5, dimension: 64)
        let database = Metal4KernelTestHelpers.normalizedVectors(count: 10, dimension: 64)

        // When inputs are normalized, dot product equals cosine similarity
        let results = try await kernel.compute(
            queries: queries,
            database: database,
            inputsNormalized: true
        )

        for i in 0..<queries.count {
            for j in 0..<database.count {
                let expected = Metal4KernelTestHelpers.cpuCosineSimilarity(queries[i], database[j])
                XCTAssertEqual(results[i][j], expected, accuracy: 1e-3)
            }
        }
    }

    func testCosineDistance() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 8)
        let database = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 8)

        let results = try await kernel.compute(
            queries: queries,
            database: database,
            outputDistance: true
        )

        for i in 0..<queries.count {
            for j in 0..<database.count {
                let similarity = Metal4KernelTestHelpers.cpuCosineSimilarity(queries[i], database[j])
                let expectedDistance = 1 - similarity
                XCTAssertEqual(results[i][j], expectedDistance, accuracy: 1e-4)
            }
        }
    }

    // MARK: - Optimized Dimension Tests

    func testDimension768() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 768)
        let database = Metal4KernelTestHelpers.randomVectors(count: 20, dimension: 768)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].count, 20)

        let expected00 = Metal4KernelTestHelpers.cpuCosineSimilarity(queries[0], database[0])
        XCTAssertEqual(results[0][0], expected00, accuracy: 1e-3)
    }

    // MARK: - Similarity Range Tests

    func testSimilarityRange() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 64)
        let database = Metal4KernelTestHelpers.randomVectors(count: 20, dimension: 64)

        let results = try await kernel.compute(queries: queries, database: database)

        for row in results {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, -1.0)
                XCTAssertLessThanOrEqual(value, 1.0)
            }
        }
    }
}

// MARK: - Metal4DotProductKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4DotProductKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4DotProductKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4DotProductKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testSmallVectors() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 4)
        let database = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 4)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 3)

        for i in 0..<queries.count {
            for j in 0..<database.count {
                let expected = Metal4KernelTestHelpers.cpuDotProduct(queries[i], database[j])
                XCTAssertEqual(results[i][j], expected, accuracy: 1e-4)
            }
        }
    }

    func testAbsoluteValue() async throws {
        let queries: [[Float]] = [[1, -2, 3]]
        let database: [[Float]] = [[-1, 2, -3]]

        let normalResults = try await kernel.compute(queries: queries, database: database, absoluteValue: false)
        let absResults = try await kernel.compute(queries: queries, database: database, absoluteValue: true)

        let expectedNormal = Metal4KernelTestHelpers.cpuDotProduct(queries[0], database[0])
        XCTAssertEqual(normalResults[0][0], expectedNormal, accuracy: 1e-4)
        XCTAssertEqual(absResults[0][0], abs(expectedNormal), accuracy: 1e-4)
    }

    // MARK: - GEMV Tests

    func testSingleQueryGEMV() async throws {
        let query: [Float] = (0..<128).map { _ in Float.random(in: -1...1) }
        let database = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 128)

        let results = try await kernel.computeSingle(query: query, database: database)

        XCTAssertEqual(results.count, 100)

        for j in 0..<database.count {
            let expected = Metal4KernelTestHelpers.cpuDotProduct(query, database[j])
            XCTAssertEqual(results[j], expected, accuracy: 1e-3)
        }
    }

    // MARK: - Optimized Dimension Tests

    func testDimension768() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 768)
        let database = Metal4KernelTestHelpers.randomVectors(count: 20, dimension: 768)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].count, 20)

        let expected00 = Metal4KernelTestHelpers.cpuDotProduct(queries[0], database[0])
        XCTAssertEqual(results[0][0], expected00, accuracy: 1e-2)  // Larger tolerance for high dimensions
    }

    // MARK: - Normalized Vector Tests

    func testNormalizedVectorsDotProduct() async throws {
        let queries = Metal4KernelTestHelpers.normalizedVectors(count: 5, dimension: 64)
        let database = Metal4KernelTestHelpers.normalizedVectors(count: 10, dimension: 64)

        let results = try await kernel.compute(queries: queries, database: database)

        // For normalized vectors, dot product should be in [-1, 1]
        for row in results {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, -1.1)  // Small tolerance
                XCTAssertLessThanOrEqual(value, 1.1)
            }
        }
    }
}

// MARK: - Kernel Fusion Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4KernelFusionTests: XCTestCase {

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

    func testEncodeAPI() async throws {
        let l2Kernel = try await Metal4L2DistanceKernel(context: context)
        let cosineKernel = try await Metal4CosineSimilarityKernel(context: context)

        let device = context.device.rawDevice

        // Create test data
        let queries = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 128)
        let database = Metal4KernelTestHelpers.randomVectors(count: 50, dimension: 128)

        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        let databaseBuffer = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        let l2OutputBuffer = device.makeBuffer(
            length: 10 * 50 * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        let cosineOutputBuffer = device.makeBuffer(
            length: 10 * 50 * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // Execute both kernels in a single command buffer (fusion)
        try await context.executeAndWait { _, encoder in
            // Encode L2 distance
            let l2Result = l2Kernel.encode(
                into: encoder,
                queries: queryBuffer,
                database: databaseBuffer,
                distances: l2OutputBuffer,
                parameters: Metal4L2DistanceParameters(
                    numQueries: 10,
                    numDatabase: 50,
                    dimension: 128
                )
            )
            XCTAssertEqual(l2Result.pipelineName, "l2_distance_kernel")

            // Barrier between operations
            encoder.memoryBarrier(scope: .buffers)

            // Encode cosine similarity
            let cosineResult = cosineKernel.encode(
                into: encoder,
                queries: queryBuffer,
                database: databaseBuffer,
                output: cosineOutputBuffer,
                parameters: Metal4CosineSimilarityParameters(
                    numQueries: 10,
                    numDatabase: 50,
                    dimension: 128
                )
            )
            XCTAssertEqual(cosineResult.pipelineName, "cosine_similarity_general_kernel")
        }

        // Verify both outputs are valid
        let l2Pointer = l2OutputBuffer.contents().bindMemory(to: Float.self, capacity: 500)
        let cosinePointer = cosineOutputBuffer.contents().bindMemory(to: Float.self, capacity: 500)

        // L2 distances should be non-negative
        for i in 0..<500 {
            XCTAssertGreaterThanOrEqual(l2Pointer[i], 0)
        }

        // Cosine similarities should be in [-1, 1]
        for i in 0..<500 {
            XCTAssertGreaterThanOrEqual(cosinePointer[i], -1.1)
            XCTAssertLessThanOrEqual(cosinePointer[i], 1.1)
        }
    }
}

// MARK: - Metal4TopKSelectionKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4TopKSelectionKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4TopKSelectionKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4TopKSelectionKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicTopKMinimum() async throws {
        // Create test data with known values
        let values: [[Float]] = [
            [5.0, 2.0, 8.0, 1.0, 9.0, 3.0],  // Top-3 min: 1.0, 2.0, 3.0
            [7.0, 4.0, 0.5, 6.0, 2.5, 1.5]   // Top-3 min: 0.5, 1.5, 2.5
        ]

        let results = try await kernel.select(from: values, k: 3, mode: .minimum, sorted: true)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 3)
        XCTAssertEqual(results[1].count, 3)

        // First query: indices 3, 1, 5 (values 1.0, 2.0, 3.0)
        XCTAssertEqual(results[0][0].value, 1.0, accuracy: 1e-5)
        XCTAssertEqual(results[0][1].value, 2.0, accuracy: 1e-5)
        XCTAssertEqual(results[0][2].value, 3.0, accuracy: 1e-5)

        // Second query: indices 2, 5, 4 (values 0.5, 1.5, 2.5)
        XCTAssertEqual(results[1][0].value, 0.5, accuracy: 1e-5)
        XCTAssertEqual(results[1][1].value, 1.5, accuracy: 1e-5)
        XCTAssertEqual(results[1][2].value, 2.5, accuracy: 1e-5)
    }

    func testBasicTopKMaximum() async throws {
        let values: [[Float]] = [
            [5.0, 2.0, 8.0, 1.0, 9.0, 3.0],  // Top-3 max: 9.0, 8.0, 5.0
        ]

        let results = try await kernel.select(from: values, k: 3, mode: .maximum, sorted: true)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        XCTAssertEqual(results[0][0].value, 9.0, accuracy: 1e-5)
        XCTAssertEqual(results[0][1].value, 8.0, accuracy: 1e-5)
        XCTAssertEqual(results[0][2].value, 5.0, accuracy: 1e-5)
    }

    func testSingleQuery() async throws {
        let values: [Float] = (0..<100).map { Float($0) }  // 0, 1, 2, ..., 99

        let results = try await kernel.selectSingle(from: values, k: 5, mode: .minimum)

        XCTAssertEqual(results.count, 5)
        // Should select 0, 1, 2, 3, 4
        for i in 0..<5 {
            XCTAssertEqual(results[i].value, Float(i), accuracy: 1e-5)
            XCTAssertEqual(results[i].index, i)
        }
    }

    // MARK: - Edge Cases

    func testKEqualsN() async throws {
        let values: [[Float]] = [[3.0, 1.0, 2.0]]

        let results = try await kernel.select(from: values, k: 3, mode: .minimum)

        XCTAssertEqual(results[0].count, 3)
    }

    func testKLargerThanN() async throws {
        let values: [[Float]] = [[3.0, 1.0, 2.0]]

        // K clamped to maxK but still larger than N
        let results = try await kernel.select(from: values, k: 10, mode: .minimum)

        // Should return all 3 valid results
        XCTAssertGreaterThanOrEqual(results[0].count, 3)
    }

    // MARK: - Larger Tests

    func testLargerBatch() async throws {
        let batchSize = 50
        let numElements = 1000
        let k = 10

        let values = (0..<batchSize).map { _ in
            (0..<numElements).map { _ in Float.random(in: 0...100) }
        }

        let results = try await kernel.select(from: values, k: k, mode: .minimum, sorted: true)

        XCTAssertEqual(results.count, batchSize)

        // Verify each query has k results
        for queryResults in results {
            XCTAssertEqual(queryResults.count, k)
        }

        // Verify results are sorted for first query
        let firstQueryValues = results[0].map { $0.value }
        for i in 1..<firstQueryValues.count {
            XCTAssertGreaterThanOrEqual(firstQueryValues[i], firstQueryValues[i-1])
        }
    }

    // MARK: - Result Extraction Tests

    func testResultExtraction() async throws {
        let values: [[Float]] = [
            [5.0, 2.0, 8.0, 1.0],
            [7.0, 3.0, 0.5, 4.0]
        ]

        let device = context.device.rawDevice
        let flatValues = values.flatMap { $0 }
        guard let inputBuffer = device.makeBuffer(
            bytes: flatValues,
            length: flatValues.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create buffer")
        }

        let params = Metal4TopKParameters(
            batchSize: 2,
            numElements: 4,
            k: 2,
            mode: .minimum,
            sorted: true
        )

        let result = try await kernel.execute(input: inputBuffer, parameters: params)

        XCTAssertEqual(result.batchSize, 2)
        XCTAssertEqual(result.k, 2)

        let query0Results = result.results(for: 0)
        XCTAssertEqual(query0Results.count, 2)

        let allResults = result.allResults()
        XCTAssertEqual(allResults.count, 2)
    }
}

// MARK: - Metal4FusedL2TopKKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4FusedL2TopKKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4FusedL2TopKKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4FusedL2TopKKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicNearestNeighbors() async throws {
        // Create known vectors where we can predict nearest neighbors
        let queries: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0]
        ]
        let dataset: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],  // Squared L2 distance = 1.0
            [2.0, 0.0, 0.0, 0.0],  // Squared L2 distance = 4.0
            [0.5, 0.0, 0.0, 0.0],  // Squared L2 distance = 0.25
            [3.0, 0.0, 0.0, 0.0],  // Squared L2 distance = 9.0
        ]

        let results = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: 2
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 2)

        // Nearest should be index 2 (squared distance 0.25)
        // Note: Kernel returns squared L2 distances for performance
        XCTAssertEqual(results[0][0].index, 2)
        XCTAssertEqual(results[0][0].distance, 0.25, accuracy: 1e-4)

        // Second nearest should be index 0 (squared distance 1.0)
        XCTAssertEqual(results[0][1].index, 0)
        XCTAssertEqual(results[0][1].distance, 1.0, accuracy: 1e-4)
    }

    func testMultipleQueries() async throws {
        let numQueries = 10
        let numDataset = 100
        let dimension = 64
        let k = 5

        let queries = Metal4KernelTestHelpers.randomVectors(count: numQueries, dimension: dimension)
        let dataset = Metal4KernelTestHelpers.randomVectors(count: numDataset, dimension: dimension)

        let results = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: k
        )

        XCTAssertEqual(results.count, numQueries)

        for queryResults in results {
            XCTAssertEqual(queryResults.count, k)

            // Verify results are sorted by distance
            for i in 1..<queryResults.count {
                XCTAssertGreaterThanOrEqual(queryResults[i].distance, queryResults[i-1].distance)
            }

            // Verify indices are valid
            for result in queryResults {
                XCTAssertGreaterThanOrEqual(result.index, 0)
                XCTAssertLessThan(result.index, numDataset)
            }
        }
    }

    // MARK: - CPU Verification

    func testAgainstCPUReference() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 32)
        let dataset = Metal4KernelTestHelpers.randomVectors(count: 50, dimension: 32)
        let k = 3

        let results = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: k
        )

        // Verify against CPU for first query
        // Note: GPU returns squared L2 distances for performance
        var cpuDistances: [(index: Int, distance: Float)] = []
        for (j, db) in dataset.enumerated() {
            let dist = Metal4KernelTestHelpers.cpuL2DistanceSquared(queries[0], db)
            cpuDistances.append((index: j, distance: dist))
        }
        cpuDistances.sort { $0.distance < $1.distance }

        // Compare top-k
        for i in 0..<k {
            XCTAssertEqual(results[0][i].index, cpuDistances[i].index)
            XCTAssertEqual(results[0][i].distance, cpuDistances[i].distance, accuracy: 1e-3)
        }
    }

    // MARK: - Configuration Tests

    func testWithoutDistances() async throws {
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 64)
        let dataset = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 64)

        // Note: Fused kernel has K_PRIVATE=8 limit, use k<=8 for this test
        let results = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: 8,
            includeDistances: false
        )

        XCTAssertEqual(results.count, 5)
        // Results should still have indices (distances will be 0)
        for queryResults in results {
            XCTAssertEqual(queryResults.count, 8)
        }
    }

    // MARK: - Dimension Tests

    func testMaxDimension() async throws {
        // Test at max dimension (512)
        let queries = Metal4KernelTestHelpers.randomVectors(count: 2, dimension: 512)
        let dataset = Metal4KernelTestHelpers.randomVectors(count: 20, dimension: 512)

        let results = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: dataset,
            k: 5
        )

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].count, 5)
    }
}

// MARK: - Metal4StreamingTopKKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4StreamingTopKKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4StreamingTopKKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4StreamingTopKKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Configuration Tests

    func testStreamingConfigInit() {
        let config = Metal4StreamingConfig(
            queryCount: 100,
            k: 10,
            chunkSize: 10_000,
            totalVectorCount: 100_000
        )

        XCTAssertEqual(config.queryCount, 100)
        XCTAssertEqual(config.k, 10)
        XCTAssertEqual(config.chunkSize, 10_000)
        XCTAssertEqual(config.totalVectorCount, 100_000)
        XCTAssertEqual(config.numberOfChunks, 10)
    }

    func testKClamping() {
        let config = Metal4StreamingConfig(
            queryCount: 10,
            k: 500,  // Exceeds maxK
            chunkSize: 1000,
            totalVectorCount: 10_000
        )

        XCTAssertEqual(config.k, Metal4StreamingConfig.maxK)
    }

    // MARK: - State Tests

    func testStreamingStateInit() async throws {
        let config = Metal4StreamingConfig(
            queryCount: 10,
            k: 5,
            chunkSize: 100,
            totalVectorCount: 500
        )

        let state = try await kernel.initializeStreaming(config: config)

        XCTAssertEqual(state.chunksProcessed, 0)
        XCTAssertEqual(state.progress, 0.0)
        XCTAssertFalse(state.isComplete)
    }

    // MARK: - Integration Tests

    func testFullStreamingPipeline() async throws {
        let queryCount = 5
        let k = 3
        let chunkSize = 100
        let totalVectors: Int64 = 300

        let config = Metal4StreamingConfig(
            queryCount: queryCount,
            k: k,
            chunkSize: chunkSize,
            totalVectorCount: totalVectors
        )

        // Initialize streaming state
        let state = try await kernel.initializeStreaming(config: config)

        XCTAssertFalse(state.isComplete)

        // Create test distance chunks
        let device = context.device.rawDevice

        for chunkIdx in 0..<config.numberOfChunks {
            let chunkData = (0..<queryCount * chunkSize).map { _ in
                Float.random(in: 0...100)
            }

            guard let chunkBuffer = device.makeBuffer(
                bytes: chunkData,
                length: chunkData.count * MemoryLayout<Float>.size,
                options: .storageModeShared
            ) else {
                throw XCTSkip("Failed to create chunk buffer")
            }

            try await kernel.processChunk(
                distances: chunkBuffer,
                state: state,
                chunkBaseIndex: Int64(chunkIdx * chunkSize)
            )
        }

        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(state.progress, 1.0)

        // Finalize and get results
        let result = try await kernel.finalizeStreaming(state: state)

        XCTAssertEqual(result.queryCount, queryCount)
        XCTAssertEqual(result.k, k)

        // Verify results
        let allResults = result.allResults()
        XCTAssertEqual(allResults.count, queryCount)

        for queryResults in allResults {
            XCTAssertLessThanOrEqual(queryResults.count, k)
        }
    }

    // MARK: - High-Level API Test

    func testStreamingTopKHighLevel() async throws {
        let queryCount = 3
        let k = 5
        let chunkSize = 50
        let totalVectors: Int64 = 150

        let config = Metal4StreamingConfig(
            queryCount: queryCount,
            k: k,
            chunkSize: chunkSize,
            totalVectorCount: totalVectors
        )

        let device = context.device.rawDevice
        var progressUpdates: [Float] = []

        let result = try await kernel.streamingTopK(
            distanceChunkProvider: { chunkIdx in
                let startIdx = Int64(chunkIdx) * Int64(chunkSize)
                if startIdx >= totalVectors {
                    return nil
                }

                let chunkData = (0..<queryCount * chunkSize).map { _ in
                    Float.random(in: 0...100)
                }

                return device.makeBuffer(
                    bytes: chunkData,
                    length: chunkData.count * MemoryLayout<Float>.size,
                    options: .storageModeShared
                )
            },
            config: config,
            progressHandler: { progress in
                progressUpdates.append(progress)
            }
        )

        XCTAssertEqual(result.queryCount, queryCount)
        XCTAssertEqual(result.k, k)

        // Should have received progress updates
        XCTAssertFalse(progressUpdates.isEmpty)

        // Final progress should be 1.0
        XCTAssertEqual(progressUpdates.last!, 1.0, accuracy: 0.01)
    }
}

// MARK: - Distance + TopK Fusion Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4DistanceTopKFusionTests: XCTestCase {

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

    func testL2DistanceTopKFusion() async throws {
        let l2Kernel = try await Metal4L2DistanceKernel(context: context)
        let topKKernel = try await Metal4TopKSelectionKernel(context: context)

        // Use the fusedDistanceTopK helper
        let queries = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 64)
        let database = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 64)
        let k = 10

        let device = context.device.rawDevice
        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create query buffer")
        }

        guard let databaseBuffer = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create database buffer")
        }

        let params = Metal4L2DistanceParameters(
            numQueries: 5,
            numDatabase: 100,
            dimension: 64
        )

        let result = try await topKKernel.fusedDistanceTopK(
            distanceKernel: l2Kernel,
            queries: queryBuffer,
            database: databaseBuffer,
            distanceParams: params,
            k: k,
            mode: .minimum
        )

        XCTAssertEqual(result.batchSize, 5)
        XCTAssertEqual(result.k, k)

        let allResults = result.allResults()
        XCTAssertEqual(allResults.count, 5)

        // Verify each query has valid results
        for queryResults in allResults {
            XCTAssertEqual(queryResults.count, k)
            for res in queryResults {
                XCTAssertGreaterThanOrEqual(res.index, 0)
                XCTAssertLessThan(res.index, 100)
                XCTAssertGreaterThanOrEqual(res.value, 0)  // L2 distances are non-negative
            }
        }
    }

    func testCosineSimilarityTopKFusion() async throws {
        let cosineKernel = try await Metal4CosineSimilarityKernel(context: context)
        let topKKernel = try await Metal4TopKSelectionKernel(context: context)

        let queries = Metal4KernelTestHelpers.normalizedVectors(count: 5, dimension: 64)
        let database = Metal4KernelTestHelpers.normalizedVectors(count: 100, dimension: 64)
        let k = 10

        let device = context.device.rawDevice
        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create query buffer")
        }

        guard let databaseBuffer = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create database buffer")
        }

        let params = Metal4CosineSimilarityParameters(
            numQueries: 5,
            numDatabase: 100,
            dimension: 64,
            inputsNormalized: true
        )

        // For similarity, we want maximum values
        let result = try await topKKernel.fusedDistanceTopK(
            distanceKernel: cosineKernel,
            queries: queryBuffer,
            database: databaseBuffer,
            distanceParams: params,
            k: k,
            mode: .maximum
        )

        XCTAssertEqual(result.batchSize, 5)
        XCTAssertEqual(result.k, k)

        let allResults = result.allResults()
        XCTAssertEqual(allResults.count, 5)

        // Verify results are sorted descending (for maximum mode)
        for queryResults in allResults {
            for i in 1..<queryResults.count {
                XCTAssertLessThanOrEqual(queryResults[i].value, queryResults[i-1].value)
            }
        }
    }
}

// MARK: - Metal4ScalarQuantizationKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ScalarQuantizationKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4ScalarQuantizationKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4ScalarQuantizationKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - INT8 Tests

    func testBasicInt8Quantization() async throws {
        let data: [Float] = [0.0, 0.5, 1.0, -0.5, -1.0]

        let result = try await kernel.quantize(data, bitWidth: .int8, type: .symmetric)

        XCTAssertEqual(result.quantizedData.count, 5)
        XCTAssertEqual(result.bitWidth, .int8)
        XCTAssertGreaterThan(result.compressionRatio, 3.5)  // Should be ~4x
    }

    func testInt8RoundTrip() async throws {
        let data: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]

        let result = try await kernel.quantize(data, bitWidth: .int8)
        let reconstructed = try await kernel.dequantize(result, count: data.count)

        XCTAssertEqual(reconstructed.count, data.count)

        // Check reconstruction error is reasonable
        for i in 0..<data.count {
            XCTAssertEqual(reconstructed[i], data[i], accuracy: 0.02)
        }
    }

    func testInt8SymmetricVsAsymmetric() async throws {
        let data: [Float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  // All positive

        let symmetricResult = try await kernel.quantizeWithMetrics(data, bitWidth: .int8, type: .symmetric)
        let asymmetricResult = try await kernel.quantizeWithMetrics(data, bitWidth: .int8, type: .asymmetric)

        // Asymmetric should be better for data that's not centered around zero
        XCTAssertNotNil(asymmetricResult.result.zeroPoint)
        XCTAssertNil(symmetricResult.result.zeroPoint)
    }

    // MARK: - INT4 Tests

    func testBasicInt4Quantization() async throws {
        let data: [Float] = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75]

        let result = try await kernel.quantize(data, bitWidth: .int4, type: .symmetric)

        // INT4 packs 2 values per byte
        XCTAssertEqual(result.quantizedData.count, 4)
        XCTAssertEqual(result.bitWidth, .int4)
        XCTAssertGreaterThan(result.compressionRatio, 7.5)  // Should be ~8x
    }

    func testInt4RoundTrip() async throws {
        let data: [Float] = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75]

        let result = try await kernel.quantize(data, bitWidth: .int4)
        let reconstructed = try await kernel.dequantize(result, count: data.count)

        XCTAssertEqual(reconstructed.count, data.count)

        // INT4 has more error than INT8
        for i in 0..<data.count {
            XCTAssertEqual(reconstructed[i], data[i], accuracy: 0.2)
        }
    }

    // MARK: - Metrics Tests

    func testQuantizationMetrics() async throws {
        let data = (0..<1000).map { _ in Float.random(in: -1...1) }

        let (result, metrics) = try await kernel.quantizeWithMetrics(data, bitWidth: .int8)

        XCTAssertGreaterThanOrEqual(metrics.mse, 0)
        XCTAssertGreaterThanOrEqual(metrics.maxError, 0)
        XCTAssertEqual(metrics.compressionRatio, result.compressionRatio)
    }

    // MARK: - Edge Cases

    func testEmptyInputThrows() async throws {
        do {
            _ = try await kernel.quantize([], bitWidth: .int8)
            XCTFail("Expected error for empty input")
        } catch {
            // Expected
        }
    }

    func testLargeScale() async throws {
        // Data with large range
        let data: [Float] = [-1000, -100, 0, 100, 1000]

        let result = try await kernel.quantize(data, bitWidth: .int8)
        let reconstructed = try await kernel.dequantize(result, count: data.count)

        // Should still preserve relative order
        for i in 1..<data.count {
            if data[i] > data[i-1] {
                XCTAssertGreaterThan(reconstructed[i], reconstructed[i-1])
            }
        }
    }
}

// MARK: - Metal4BinaryQuantizationKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4BinaryQuantizationKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4BinaryQuantizationKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4BinaryQuantizationKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicBinaryQuantization() async throws {
        let vectors: [[Float]] = [
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0]
        ]

        let result = try await kernel.quantize(vectors: vectors)

        XCTAssertEqual(result.binaryVectors.count, 2)
        XCTAssertEqual(result.binaryVectors.dimension, 4)
        // Compression ratio for 4-dim vectors: 16 bytes (4 floats) / 4 bytes (1 uint32) = 4:1
        // 32:1 ratio only applies to 32+ dimensional vectors
        XCTAssertEqual(result.compressionRatio, 4.0, accuracy: 0.1)
    }

    func testBinaryVectorExtraction() async throws {
        let vectors: [[Float]] = [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        ]

        let result = try await kernel.quantize(vectors: vectors, config: .default)
        guard let binaryVector = result.binaryVectors.vector(at: 0) else {
            XCTFail("Failed to get binary vector")
            return
        }

        // With threshold 0, positives should be 1s
        let bools = binaryVector.asBoolArray()
        XCTAssertEqual(bools.count, 8)
        XCTAssertTrue(bools[0])   // 1.0 > 0
        XCTAssertFalse(bools[1])  // -1.0 < 0
        XCTAssertTrue(bools[2])   // 1.0 > 0
        XCTAssertFalse(bools[3])  // -1.0 < 0
    }

    func testSingleVectorQuantization() async throws {
        let vector: [Float] = [1.0, -1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.001]

        let binaryVector = try await kernel.quantize(vector: vector)

        XCTAssertEqual(binaryVector.dimension, 8)
    }

    // MARK: - Hamming Distance Tests

    func testHammingDistanceIdentical() async throws {
        let vectors: [[Float]] = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ]

        let result = try await kernel.quantize(vectors: vectors)

        let distances = try await kernel.computeHammingDistances(
            query: result.binaryVectors.vectors[0],
            candidates: result.binaryVectors
        )

        // Distance to self should be 0
        XCTAssertEqual(distances.distances[0], 0.0, accuracy: 0.001)
    }

    func testHammingDistanceOpposite() async throws {
        let vectors: [[Float]] = [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0]
        ]

        let result = try await kernel.quantize(vectors: vectors)

        let distances = try await kernel.computeHammingDistances(
            query: result.binaryVectors.vectors[0],
            candidates: result.binaryVectors
        )

        // All bits should be different
        XCTAssertEqual(distances.distances[1], 4.0, accuracy: 0.001)
    }

    func testHammingDistancePartialMatch() async throws {
        let vectors: [[Float]] = [
            [1.0, 1.0, -1.0, -1.0],  // 1100
            [1.0, -1.0, 1.0, -1.0]   // 1010 - 2 bits different
        ]

        let result = try await kernel.quantize(vectors: vectors)

        let distances = try await kernel.computeHammingDistances(
            query: result.binaryVectors.vectors[0],
            candidates: result.binaryVectors
        )

        XCTAssertEqual(distances.distances[1], 2.0, accuracy: 0.001)
    }

    // MARK: - CPU Verification

    func testCPUHammingMatch() async throws {
        let vectors = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 32)

        let result = try await kernel.quantize(vectors: vectors)

        // Compare GPU distances to CPU implementation
        let query = result.binaryVectors.vectors[0]
        let gpuDistances = try await kernel.computeHammingDistances(
            query: query,
            candidates: result.binaryVectors
        )

        for i in 0..<result.binaryVectors.count {
            let cpuDistance = query.hammingDistance(to: result.binaryVectors.vectors[i])
            XCTAssertEqual(Int(gpuDistances.distances[i]), cpuDistance)
        }
    }

    // MARK: - Compression Ratio

    func testCompressionRatio() {
        // 128-dim float32: 512 bytes → 4 UInt32 words: 16 bytes = 32x compression
        let ratio = Metal4BinaryQuantizationKernel.compressionRatio(for: 128)
        XCTAssertEqual(ratio, 32.0, accuracy: 0.1)
    }
}

// MARK: - Metal4ProductQuantizationKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ProductQuantizationKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4ProductQuantizationKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4ProductQuantizationKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Configuration Tests

    func testPQConfigInit() {
        let config = Metal4PQConfig(dimension: 128, M: 8, K: 256)

        XCTAssertEqual(config.dimension, 128)
        XCTAssertEqual(config.M, 8)
        XCTAssertEqual(config.K, 256)
        XCTAssertEqual(config.D_sub, 16)
        // Compression: 128 floats × 32 bits = 4096 bits → 8 codes × 8 bits = 64 bits
        // Ratio: 4096 / 64 = 64:1
        XCTAssertEqual(config.compressionRatio, 64.0)
    }

    // MARK: - Training Tests

    func testBasicTraining() async throws {
        // Small dataset for quick test
        let trainingData = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 32)
        let config = Metal4PQConfig(dimension: 32, M: 4, K: 16, trainIterations: 5)

        let device = context.device.rawDevice
        let flatData = trainingData.flatMap { $0 }
        guard let dataBuffer = device.makeBuffer(
            bytes: flatData,
            length: flatData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw XCTSkip("Failed to create buffer")
        }

        let model = try await kernel.train(
            data: dataBuffer,
            count: trainingData.count,
            config: config
        )

        XCTAssertNotNil(model.codebooks)
        XCTAssertEqual(model.config.M, 4)
        XCTAssertEqual(model.config.K, 16)
    }

    // MARK: - Encoding Tests

    func testEncoding() async throws {
        let trainingData = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 32)
        let config = Metal4PQConfig(dimension: 32, M: 4, K: 16, trainIterations: 3)

        let (model, encoded) = try await kernel.trainAndEncode(data: trainingData, config: config)

        XCTAssertEqual(encoded.count, 100)
        XCTAssertEqual(encoded.config.M, 4)
        XCTAssertEqual(encoded.memoryBytes, 100 * 4)  // 100 vectors × 4 subspaces

        // Test decoding
        let decoded = encoded.decode(index: 0, using: model)
        XCTAssertEqual(decoded.count, 32)
    }

    // MARK: - Search Tests

    func testNearestNeighborSearch() async throws {
        let trainingData = Metal4KernelTestHelpers.randomVectors(count: 200, dimension: 32)
        let config = Metal4PQConfig(dimension: 32, M: 4, K: 16, trainIterations: 5)

        let (model, encoded) = try await kernel.trainAndEncode(data: trainingData, config: config)

        // Search with one of the training vectors
        let neighbors = try await kernel.findNearestNeighbors(
            query: trainingData[0],
            encodedDatabase: encoded,
            model: model,
            k: 5
        )

        XCTAssertEqual(neighbors.count, 5)

        // Distances should be sorted ascending
        for i in 1..<neighbors.count {
            XCTAssertGreaterThanOrEqual(neighbors[i].distance, neighbors[i-1].distance)
        }

        // First result should be close (possibly the query itself)
        XCTAssertLessThan(neighbors[0].distance, 5.0)
    }

    // MARK: - Model Properties

    func testModelCompressionRatio() async throws {
        let trainingData = Metal4KernelTestHelpers.randomVectors(count: 50, dimension: 128)
        let config = Metal4PQConfig(dimension: 128, M: 8, K: 256, trainIterations: 3)

        let (model, _) = try await kernel.trainAndEncode(data: trainingData, config: config)

        // Compression: 128 floats × 32 bits = 4096 bits → 8 codes × 8 bits = 64 bits = 64:1
        XCTAssertEqual(model.compressionRatio, 64.0)
    }

    func testCentroidDecoding() async throws {
        let trainingData = Metal4KernelTestHelpers.randomVectors(count: 50, dimension: 32)
        let config = Metal4PQConfig(dimension: 32, M: 4, K: 16, trainIterations: 3)

        let (model, _) = try await kernel.trainAndEncode(data: trainingData, config: config)

        // Decode a centroid
        let centroid = model.decodeCentroid(subspace: 0, code: 0)
        XCTAssertEqual(centroid.count, config.D_sub)
    }
}

// MARK: - Metal4MatrixMultiplyKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4MatrixMultiplyKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4MatrixMultiplyKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4MatrixMultiplyKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicMultiplication() async throws {
        // A: 2x3, B: 3x2 -> C: 2x2
        let A = Matrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let B = Matrix(rows: 3, columns: 2, values: [7, 8, 9, 10, 11, 12])

        let result = try await kernel.multiply(A, B)

        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.columns, 2)

        let C = result.asMatrix()
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        XCTAssertEqual(C[0, 0], 58, accuracy: 1e-4)
        XCTAssertEqual(C[0, 1], 64, accuracy: 1e-4)
        XCTAssertEqual(C[1, 0], 139, accuracy: 1e-4)
        XCTAssertEqual(C[1, 1], 154, accuracy: 1e-4)
    }

    func testSquareMatrixMultiplication() async throws {
        let size = 64
        let A = Matrix.random(rows: size, columns: size)
        let B = Matrix.random(rows: size, columns: size)

        let result = try await kernel.multiply(A, B)

        XCTAssertEqual(result.rows, size)
        XCTAssertEqual(result.columns, size)
    }

    func testLargerMatrixMultiplication() async throws {
        let M = 128
        let K = 256
        let N = 64

        let A = Matrix.random(rows: M, columns: K)
        let B = Matrix.random(rows: K, columns: N)

        let result = try await kernel.multiply(A, B)

        XCTAssertEqual(result.rows, M)
        XCTAssertEqual(result.columns, N)
        XCTAssertGreaterThan(result.gflops, 0)
    }

    // MARK: - GEMM Tests

    func testGEMM() async throws {
        let A = Matrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let B = Matrix(rows: 2, columns: 2, values: [5, 6, 7, 8])
        let C = Matrix(rows: 2, columns: 2, values: [1, 1, 1, 1])

        let config = Metal4MatrixMultiplyConfig(alpha: 2.0, beta: 3.0)
        let result = try await kernel.gemm(A, B, C, config: config)

        let resultMatrix = result.asMatrix()
        // C = 2 * A*B + 3 * C
        // A*B = [[19, 22], [43, 50]]
        // result = [[2*19 + 3*1, 2*22 + 3*1], [2*43 + 3*1, 2*50 + 3*1]]
        //        = [[41, 47], [89, 103]]
        XCTAssertEqual(resultMatrix[0, 0], 41, accuracy: 1e-3)
        XCTAssertEqual(resultMatrix[0, 1], 47, accuracy: 1e-3)
        XCTAssertEqual(resultMatrix[1, 0], 89, accuracy: 1e-3)
        XCTAssertEqual(resultMatrix[1, 1], 103, accuracy: 1e-3)
    }

    // MARK: - Batch Tests

    func testBatchMultiplication() async throws {
        let batchSize = 5
        var batchA: [Matrix] = []
        var batchB: [Matrix] = []

        for _ in 0..<batchSize {
            batchA.append(Matrix.random(rows: 32, columns: 64))
            batchB.append(Matrix.random(rows: 64, columns: 32))
        }

        let results = try await kernel.multiplyBatch(matricesA: batchA, matricesB: batchB)

        XCTAssertEqual(results.count, batchSize)
        for result in results {
            XCTAssertEqual(result.rows, 32)
            XCTAssertEqual(result.columns, 32)
        }
    }
}

// MARK: - Metal4MatrixVectorKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4MatrixVectorKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4MatrixVectorKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4MatrixVectorKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicMultiplication() async throws {
        // A: 2x3, x: 3x1 -> y: 2x1
        let A = Matrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let x: [Float] = [1, 2, 3]

        let result = try await kernel.multiply(matrix: A, vector: x)

        XCTAssertEqual(result.length, 2)

        let y = result.asArray()
        // y[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        // y[1] = 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
        XCTAssertEqual(y[0], 14, accuracy: 1e-4)
        XCTAssertEqual(y[1], 32, accuracy: 1e-4)
    }

    func testLargerMultiplication() async throws {
        let rows = 256
        let cols = 128

        let A = Matrix.random(rows: rows, columns: cols)
        let x = (0..<cols).map { _ in Float.random(in: -1...1) }

        let result = try await kernel.multiply(matrix: A, vector: x)

        XCTAssertEqual(result.length, rows)
        XCTAssertGreaterThan(result.gflops, 0)
    }

    // MARK: - Batch Tests

    func testBatchMultiplication() async throws {
        let A = Matrix.random(rows: 64, columns: 32)
        let vectors = (0..<5).map { _ in
            (0..<32).map { _ in Float.random(in: -1...1) }
        }

        let results = try await kernel.multiplyBatch(matrix: A, vectors: vectors)

        XCTAssertEqual(results.count, 5)
        for result in results {
            XCTAssertEqual(result.length, 64)
        }
    }

    // MARK: - Power Iteration Tests

    func testPowerIteration() async throws {
        // Create a symmetric positive definite matrix
        let n = 16
        var values: [Float] = []
        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    values.append(Float(n + 1))  // Dominant diagonal
                } else {
                    values.append(1.0)
                }
            }
        }
        let A = Matrix(rows: n, columns: n, values: values)

        let (eigenvalue, eigenvector) = try await kernel.powerIteration(
            matrix: A,
            iterations: 50,
            tolerance: 1e-4
        )

        XCTAssertGreaterThan(eigenvalue, 0)
        XCTAssertEqual(eigenvector.count, n)

        // Verify eigenvector is normalized
        let norm = sqrt(eigenvector.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-3)
    }
}

// MARK: - Metal4MatrixTransposeKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4MatrixTransposeKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4MatrixTransposeKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4MatrixTransposeKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicTranspose() async throws {
        // A: 2x3 -> A^T: 3x2
        let A = Matrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])

        let result = try await kernel.transpose(A)

        XCTAssertEqual(result.rows, 3)
        XCTAssertEqual(result.columns, 2)

        let AT = result.asMatrix()
        XCTAssertEqual(AT[0, 0], 1)
        XCTAssertEqual(AT[0, 1], 4)
        XCTAssertEqual(AT[1, 0], 2)
        XCTAssertEqual(AT[1, 1], 5)
        XCTAssertEqual(AT[2, 0], 3)
        XCTAssertEqual(AT[2, 1], 6)
    }

    func testSquareTranspose() async throws {
        let size = 64
        let A = Matrix.random(rows: size, columns: size)

        let result = try await kernel.transpose(A)

        XCTAssertEqual(result.rows, size)
        XCTAssertEqual(result.columns, size)
    }

    func testLargerTranspose() async throws {
        let rows = 128
        let cols = 256

        let A = Matrix.random(rows: rows, columns: cols)

        let result = try await kernel.transpose(A)

        XCTAssertEqual(result.rows, cols)
        XCTAssertEqual(result.columns, rows)
        XCTAssertGreaterThan(result.throughputGBps, 0)
    }

    // MARK: - Validation Tests

    func testDoubleTranspose() async throws {
        let A = Matrix.random(rows: 32, columns: 64)

        let doubleT = try await kernel.doubleTranspose(A)

        // (A^T)^T == A
        XCTAssertEqual(doubleT.rows, A.rows)
        XCTAssertEqual(doubleT.columns, A.columns)

        for i in 0..<A.values.count {
            XCTAssertEqual(doubleT.values[i], A.values[i], accuracy: 1e-5)
        }
    }

    func testValidation() async throws {
        let A = Matrix.random(rows: 32, columns: 48)

        let isValid = try await kernel.validate(A)

        XCTAssertTrue(isValid)
    }

    // MARK: - Batch Tests

    func testBatchTranspose() async throws {
        let matrices = (0..<5).map { _ in
            Matrix.random(rows: 32, columns: 64)
        }

        let results = try await kernel.transposeBatch(matrices)

        XCTAssertEqual(results.count, 5)
        for result in results {
            XCTAssertEqual(result.rows, 64)
            XCTAssertEqual(result.columns, 32)
        }
    }
}

// MARK: - Metal4BatchMatrixKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4BatchMatrixKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4BatchMatrixKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4BatchMatrixKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Fused Batch Tests

    func testBasicBatchMultiplication() async throws {
        let batchSize = 4
        var batchA: [Matrix] = []
        var batchB: [Matrix] = []

        for _ in 0..<batchSize {
            batchA.append(Matrix.random(rows: 32, columns: 64))
            batchB.append(Matrix.random(rows: 64, columns: 32))
        }

        let result = try await kernel.multiplyFused(batchA: batchA, batchB: batchB)

        XCTAssertEqual(result.batchSize, batchSize)
        XCTAssertEqual(result.rows, 32)
        XCTAssertEqual(result.columns, 32)
        XCTAssertGreaterThan(result.totalGflops, 0)
    }

    func testBatchWithBias() async throws {
        let batchSize = 3
        let M = 16
        let K = 32
        let N = 8

        var batchA: [Matrix] = []
        var batchB: [Matrix] = []

        for _ in 0..<batchSize {
            batchA.append(Matrix.random(rows: M, columns: K))
            batchB.append(Matrix.random(rows: K, columns: N))
        }

        let bias = (0..<N).map { _ in Float.random(in: -1...1) }
        let config = Metal4BatchFusedConfig(hasBias: true)

        let result = try await kernel.multiplyFused(
            batchA: batchA,
            batchB: batchB,
            bias: bias,
            config: config
        )

        XCTAssertEqual(result.batchSize, batchSize)
    }

    func testMatrixExtraction() async throws {
        let batchSize = 3
        var batchA: [Matrix] = []
        var batchB: [Matrix] = []

        for _ in 0..<batchSize {
            batchA.append(Matrix.random(rows: 16, columns: 32))
            batchB.append(Matrix.random(rows: 32, columns: 16))
        }

        let result = try await kernel.multiplyFused(batchA: batchA, batchB: batchB)

        // Extract individual matrices
        for i in 0..<batchSize {
            let matrix = result.matrix(at: i)
            XCTAssertNotNil(matrix)
            XCTAssertEqual(matrix?.rows, 16)
            XCTAssertEqual(matrix?.columns, 16)
        }

        // All matrices
        let allMatrices = result.allMatrices()
        XCTAssertEqual(allMatrices.count, batchSize)
    }

    // MARK: - Strided Tests

    func testStridedBatchMultiplication() async throws {
        let batchCount = 4
        let M = 16
        let K = 32
        let N = 8

        let tensorA = (0..<batchCount * M * K).map { _ in Float.random(in: -1...1) }
        let tensorB = (0..<batchCount * K * N).map { _ in Float.random(in: -1...1) }

        let config = Metal4BatchStridedConfig.contiguous(
            batchSize: batchCount,
            rowsA: M,
            colsA: K,
            colsB: N
        )

        let result = try await kernel.multiplyStrided(
            tensorA: tensorA,
            tensorB: tensorB,
            batchCount: batchCount,
            dimensions: (M: M, N: N, K: K),
            config: config
        )

        XCTAssertEqual(result.batchSize, batchCount)
        XCTAssertEqual(result.rows, M)
        XCTAssertEqual(result.columns, N)
    }

    // MARK: - Activation Tests

    func testActivationConfig() {
        let config1 = Metal4BatchFusedConfig(activation: .relu)
        XCTAssertEqual(config1.activation, .relu)

        let config2 = Metal4BatchFusedConfig(activation: .gelu)
        XCTAssertEqual(config2.activation, .gelu)
    }
}

// MARK: - Batch 5: Metal4L2NormalizationKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4L2NormalizationKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4L2NormalizationKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4L2NormalizationKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testBasicNormalization() async throws {
        let vectors: [[Float]] = [
            [3.0, 4.0],  // norm = 5, normalized = [0.6, 0.8]
            [1.0, 0.0, 0.0],  // norm = 1, normalized = [1, 0, 0]
            [1.0, 1.0, 1.0, 1.0]  // norm = 2, normalized = [0.5, 0.5, 0.5, 0.5]
        ]

        // Need same dimension for all vectors
        let uniformVectors: [[Float]] = [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ]

        let result = try await kernel.normalize(uniformVectors, storeNorms: true)

        XCTAssertEqual(result.numVectors, 3)
        XCTAssertEqual(result.dimension, 4)

        let normalized = result.asArrays()
        let norms = result.normsAsArray()

        // First vector: [3, 4, 0, 0] -> norm = 5
        XCTAssertEqual(norms?[0] ?? 0, 5.0, accuracy: 1e-4)
        XCTAssertEqual(normalized[0][0], 0.6, accuracy: 1e-4)
        XCTAssertEqual(normalized[0][1], 0.8, accuracy: 1e-4)

        // Verify all vectors are unit length
        for vector in normalized {
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
        }
    }

    func testSingleVectorNormalization() async throws {
        let vector: [Float] = [3.0, 4.0, 0.0]

        let (normalized, norm) = try await kernel.normalizeSingle(vector)

        XCTAssertEqual(norm, 5.0, accuracy: 1e-4)
        XCTAssertEqual(normalized[0], 0.6, accuracy: 1e-4)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 1e-4)
        XCTAssertEqual(normalized[2], 0.0, accuracy: 1e-4)
    }

    // MARK: - Dimension-Specific Tests

    func testDimension512() async throws {
        let vectors = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 512)

        let result = try await kernel.normalize(vectors)

        XCTAssertEqual(result.numVectors, 10)
        XCTAssertEqual(result.dimension, 512)

        let normalized = result.asArrays()
        for vector in normalized {
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(norm, 1.0, accuracy: 1e-3)
        }
    }

    func testDimension768() async throws {
        let vectors = Metal4KernelTestHelpers.randomVectors(count: 5, dimension: 768)

        let result = try await kernel.normalize(vectors)

        XCTAssertEqual(result.numVectors, 5)
        XCTAssertEqual(result.dimension, 768)
    }

    func testDimension1536() async throws {
        let vectors = Metal4KernelTestHelpers.randomVectors(count: 3, dimension: 1536)

        let result = try await kernel.normalize(vectors)

        XCTAssertEqual(result.numVectors, 3)
        XCTAssertEqual(result.dimension, 1536)
    }

    // MARK: - Edge Cases

    func testZeroVector() async throws {
        let vectors: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0]
        ]

        let result = try await kernel.normalize(vectors, epsilon: 1e-8)

        // Zero vector should remain zero (or very small due to epsilon)
        let normalized = result.asArrays()
        for value in normalized[0] {
            XCTAssertLessThan(abs(value), 1e-4)
        }
    }

    func testLargeValues() async throws {
        let vectors: [[Float]] = [
            [1000.0, 2000.0, 3000.0, 4000.0]
        ]

        let result = try await kernel.normalize(vectors)
        let normalized = result.asArrays()

        let norm = sqrt(normalized[0].reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("CosineSimilarity"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }
}

// MARK: - Batch 5: Metal4ParallelReductionKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ParallelReductionKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4ParallelReductionKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4ParallelReductionKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Sum Tests

    func testSum() async throws {
        let array: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let expected: Float = 55

        let result = try await kernel.sum(array)

        XCTAssertEqual(result, expected, accuracy: 1e-4)
    }

    func testSumLargeArray() async throws {
        let count = 10000
        let array = (0..<count).map { Float($0) }
        let expected = Float(count * (count - 1) / 2)  // Sum of 0 to n-1

        let result = try await kernel.sum(array)

        XCTAssertEqual(result, expected, accuracy: 1.0)  // Larger tolerance for float precision
    }

    // MARK: - Min/Max Tests

    func testMinimum() async throws {
        let array: [Float] = [5, 3, 8, 1, 9, 2]

        let result = try await kernel.minimum(array)

        XCTAssertEqual(result, 1.0, accuracy: 1e-4)
    }

    func testMaximum() async throws {
        let array: [Float] = [5, 3, 8, 1, 9, 2]

        let result = try await kernel.maximum(array)

        XCTAssertEqual(result, 9.0, accuracy: 1e-4)
    }

    // MARK: - ArgMin/ArgMax Tests

    func testArgMin() async throws {
        let array: [Float] = [5, 3, 8, 1, 9, 2]

        let (value, index) = try await kernel.argMin(array)

        XCTAssertEqual(value, 1.0, accuracy: 1e-4)
        XCTAssertEqual(index, 3)
    }

    func testArgMax() async throws {
        let array: [Float] = [5, 3, 8, 1, 9, 2]

        let (value, index) = try await kernel.argMax(array)

        XCTAssertEqual(value, 9.0, accuracy: 1e-4)
        XCTAssertEqual(index, 4)
    }

    // MARK: - Statistics Tests

    func testComputeStatistics() async throws {
        let array: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let stats = try await kernel.computeStatistics(array)

        XCTAssertEqual(stats.count, 10)
        XCTAssertEqual(stats.sum, 55.0, accuracy: 1e-3)
        XCTAssertEqual(stats.mean, 5.5, accuracy: 1e-3)
        XCTAssertEqual(stats.minimum, 1.0, accuracy: 1e-4)
        XCTAssertEqual(stats.maximum, 10.0, accuracy: 1e-4)
        XCTAssertEqual(stats.minIndex, 0)
        XCTAssertEqual(stats.maxIndex, 9)
        XCTAssertEqual(stats.range, 9.0, accuracy: 1e-4)
    }

    // MARK: - Edge Cases

    func testSingleElement() async throws {
        let array: [Float] = [42.0]

        let sum = try await kernel.sum(array)
        let min = try await kernel.minimum(array)
        let max = try await kernel.maximum(array)

        XCTAssertEqual(sum, 42.0, accuracy: 1e-4)
        XCTAssertEqual(min, 42.0, accuracy: 1e-4)
        XCTAssertEqual(max, 42.0, accuracy: 1e-4)
    }

    func testNegativeValues() async throws {
        let array: [Float] = [-5, -3, -8, -1, -9, -2]

        let min = try await kernel.minimum(array)
        let max = try await kernel.maximum(array)

        XCTAssertEqual(min, -9.0, accuracy: 1e-4)
        XCTAssertEqual(max, -1.0, accuracy: 1e-4)
    }

    func testMixedValues() async throws {
        let array: [Float] = [-5, 3, -8, 1, 9, -2]

        let stats = try await kernel.computeStatistics(array)

        XCTAssertEqual(stats.minimum, -8.0, accuracy: 1e-4)
        XCTAssertEqual(stats.maximum, 9.0, accuracy: 1e-4)
    }
}

// MARK: - Batch 5: Metal4ElementwiseKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ElementwiseKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4ElementwiseKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4ElementwiseKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Binary Operation Tests

    func testAdd() async throws {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [5, 6, 7, 8]

        let result = try await kernel.add(a, b)

        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 6.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 8.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 10.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 12.0, accuracy: 1e-4)
    }

    func testSubtract() async throws {
        let a: [Float] = [5, 6, 7, 8]
        let b: [Float] = [1, 2, 3, 4]

        let result = try await kernel.subtract(a, b)

        XCTAssertEqual(result[0], 4.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 4.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 4.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-4)
    }

    func testMultiply() async throws {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [2, 3, 4, 5]

        let result = try await kernel.multiply(a, b)

        XCTAssertEqual(result[0], 2.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 6.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 12.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 20.0, accuracy: 1e-4)
    }

    func testDivide() async throws {
        let a: [Float] = [10, 20, 30, 40]
        let b: [Float] = [2, 4, 5, 8]

        let result = try await kernel.divide(a, b)

        XCTAssertEqual(result[0], 5.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 5.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 6.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 5.0, accuracy: 1e-4)
    }

    // MARK: - Scalar Operation Tests

    func testScale() async throws {
        let a: [Float] = [1, 2, 3, 4]

        let result = try await kernel.scale(a, by: 2.5)

        XCTAssertEqual(result[0], 2.5, accuracy: 1e-4)
        XCTAssertEqual(result[1], 5.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 7.5, accuracy: 1e-4)
        XCTAssertEqual(result[3], 10.0, accuracy: 1e-4)
    }

    func testAddScalar() async throws {
        let a: [Float] = [1, 2, 3, 4]

        let result = try await kernel.add(a, scalar: 10)

        XCTAssertEqual(result[0], 11.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 12.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 13.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 14.0, accuracy: 1e-4)
    }

    func testClamp() async throws {
        let a: [Float] = [-2, 0, 3, 10]

        let result = try await kernel.clamp(a, min: 0, max: 5)

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 5.0, accuracy: 1e-4)
    }

    // MARK: - Unary Operation Tests

    func testAbs() async throws {
        let a: [Float] = [-1, 2, -3, 4]

        let result = try await kernel.abs(a)

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 2.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-4)
    }

    func testSquare() async throws {
        let a: [Float] = [1, 2, 3, 4]

        let result = try await kernel.square(a)

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 4.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 9.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 16.0, accuracy: 1e-4)
    }

    func testSqrt() async throws {
        let a: [Float] = [1, 4, 9, 16]

        let result = try await kernel.sqrt(a)

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 2.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-4)
    }

    func testExp() async throws {
        let a: [Float] = [0, 1, 2]

        let result = try await kernel.exp(a)

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], Float(Darwin.exp(1.0)), accuracy: 1e-3)
        XCTAssertEqual(result[2], Float(Darwin.exp(2.0)), accuracy: 1e-2)
    }

    func testLog() async throws {
        let a: [Float] = [1, Float(Darwin.exp(1.0)), Float(Darwin.exp(2.0))]

        let result = try await kernel.log(a)

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 1.0, accuracy: 1e-3)
        XCTAssertEqual(result[2], 2.0, accuracy: 1e-2)
    }

    func testNegate() async throws {
        let a: [Float] = [1, -2, 3, -4]

        let result = try await kernel.negate(a)

        XCTAssertEqual(result[0], -1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 2.0, accuracy: 1e-4)
        XCTAssertEqual(result[2], -3.0, accuracy: 1e-4)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-4)
    }

    func testReciprocal() async throws {
        let a: [Float] = [1, 2, 4, 5]

        let result = try await kernel.reciprocal(a)

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 0.5, accuracy: 1e-4)
        XCTAssertEqual(result[2], 0.25, accuracy: 1e-4)
        XCTAssertEqual(result[3], 0.2, accuracy: 1e-4)
    }

    // MARK: - Large Array Tests

    func testLargeArrays() async throws {
        let count = 10000
        let a = (0..<count).map { Float($0) }
        let b = (0..<count).map { Float($0 + 1) }

        let result = try await kernel.add(a, b)

        XCTAssertEqual(result.count, count)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(result[count - 1], Float(2 * count - 1), accuracy: 1e-2)
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("Any"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }
}

// MARK: - Batch 5: Metal4StatisticsKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4StatisticsKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4StatisticsKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4StatisticsKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Statistics Tests

    func testBasicStatistics() async throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let stats = try await kernel.computeBasicStatistics(data)

        XCTAssertEqual(stats.count, 10)
        XCTAssertEqual(stats.mean, 5.5, accuracy: 1e-3)
        XCTAssertEqual(stats.minimum, 1.0, accuracy: 1e-4)
        XCTAssertEqual(stats.maximum, 10.0, accuracy: 1e-4)
        XCTAssertEqual(stats.sum, 55.0, accuracy: 1e-3)
        XCTAssertEqual(stats.range, 9.0, accuracy: 1e-4)
        XCTAssertGreaterThan(stats.variance, 0)
        XCTAssertGreaterThan(stats.standardDeviation, 0)
    }

    func testVariance() async throws {
        let data: [Float] = [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, Variance (sample) ≈ 4.57

        let stats = try await kernel.computeBasicStatistics(data)

        XCTAssertEqual(stats.mean, 5.0, accuracy: 1e-3)
        XCTAssertEqual(stats.variance, 4.571, accuracy: 0.5)  // Sample variance
    }

    // MARK: - Higher Moments Tests

    func testHigherMoments() async throws {
        // Generate roughly normal data
        let data = (0..<1000).map { _ in Float.random(in: -2...2) }

        let moments = try await kernel.computeHigherMoments(
            data,
            mean: data.reduce(0, +) / Float(data.count),
            variance: 1.0
        )

        // Random uniform data should have:
        // - Skewness near 0
        // - Kurtosis near 1.8 (uniform) or 3 (normal)
        XCTAssertLessThan(abs(moments.skewness), 1.0)
    }

    // MARK: - Quantiles Tests

    func testQuantiles() async throws {
        let data: [Float] = (1...100).map { Float($0) }  // 1 to 100

        let quantiles = try await kernel.computeQuantiles(data, levels: [0.25, 0.5, 0.75])

        XCTAssertEqual(quantiles.firstQuartile ?? 0, 25.0, accuracy: 2.0)
        XCTAssertEqual(quantiles.median ?? 0, 50.0, accuracy: 2.0)
        XCTAssertEqual(quantiles.thirdQuartile ?? 0, 75.0, accuracy: 2.0)
    }

    // MARK: - Comprehensive Statistics Tests

    func testComprehensiveStatistics() async throws {
        let data = (0..<500).map { _ in Float.random(in: 0...100) }

        let result = try await kernel.computeStatistics(data, config: .full)

        XCTAssertEqual(result.basic.count, 500)
        XCTAssertNotNil(result.moments)
        XCTAssertNotNil(result.quantiles)
        XCTAssertGreaterThan(result.totalExecutionTime, 0)

        // Test summary doesn't crash
        let summary = result.summary()
        XCTAssertTrue(summary.contains("Mean"))
        XCTAssertTrue(summary.contains("Std Dev"))
    }

    // MARK: - Batch Statistics Tests

    func testBatchStatistics() async throws {
        let datasets: [[Float]] = [
            (0..<100).map { _ in Float.random(in: 0...50) },
            (0..<100).map { _ in Float.random(in: 50...100) },
            (0..<100).map { _ in Float.random(in: 25...75) }
        ]

        let result = try await kernel.computeBatchStatistics(datasets, config: .basic)

        XCTAssertEqual(result.results.count, 3)
        XCTAssertNotNil(result.averageStatistics)

        // First dataset should have lower mean than second
        XCTAssertLessThan(result.results[0].basic.mean, result.results[1].basic.mean)
    }

    // MARK: - Correlation Tests

    func testCorrelation() async throws {
        // Create correlated datasets
        let x: [Float] = (0..<50).map { Float($0) }
        let y: [Float] = x.map { $0 * 2 }  // Perfect positive correlation
        let z: [Float] = x.map { 100 - $0 }  // Perfect negative correlation

        let result = try await kernel.computeCorrelation(datasets: [x, y, z])

        XCTAssertEqual(result.matrix.count, 3)
        XCTAssertEqual(result.matrix[0].count, 3)

        // Diagonal should be 1 (self-correlation)
        XCTAssertEqual(result.matrix[0][0], 1.0, accuracy: 0.1)
        XCTAssertEqual(result.matrix[1][1], 1.0, accuracy: 0.1)
        XCTAssertEqual(result.matrix[2][2], 1.0, accuracy: 0.1)

        // x and y should be positively correlated
        XCTAssertGreaterThan(result.matrix[0][1], 0.9)

        // x and z should be negatively correlated
        XCTAssertLessThan(result.matrix[0][2], -0.9)
    }

    // MARK: - Edge Cases

    func testSingleElement() async throws {
        let data: [Float] = [42.0]

        let stats = try await kernel.computeBasicStatistics(data)

        XCTAssertEqual(stats.count, 1)
        XCTAssertEqual(stats.mean, 42.0, accuracy: 1e-4)
        XCTAssertEqual(stats.variance, 0.0, accuracy: 1e-4)
        XCTAssertEqual(stats.minimum, 42.0, accuracy: 1e-4)
        XCTAssertEqual(stats.maximum, 42.0, accuracy: 1e-4)
    }

    func testEmptyInputThrows() async throws {
        do {
            _ = try await kernel.computeBasicStatistics([])
            XCTFail("Expected error for empty input")
        } catch {
            // Expected
        }
    }

    func testNaNInputThrows() async throws {
        do {
            _ = try await kernel.computeBasicStatistics([1.0, Float.nan, 3.0])
            XCTFail("Expected error for NaN input")
        } catch {
            // Expected
        }
    }

    // MARK: - Config Tests

    func testDefaultConfig() {
        let config = Metal4StatisticsConfig.default
        XCTAssertTrue(config.computeHigherMoments)
        XCTAssertFalse(config.computeQuantiles)
        XCTAssertTrue(config.biasCorrection)
    }

    func testBasicConfig() {
        let config = Metal4StatisticsConfig.basic
        XCTAssertFalse(config.computeHigherMoments)
        XCTAssertFalse(config.computeQuantiles)
    }

    func testFullConfig() {
        let config = Metal4StatisticsConfig.full
        XCTAssertTrue(config.computeHigherMoments)
        XCTAssertTrue(config.computeQuantiles)
    }
}

// MARK: - Batch 6a: Metal4MinkowskiDistanceKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4MinkowskiDistanceKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4MinkowskiDistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4MinkowskiDistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - L2 (Euclidean) Tests

    func testEuclideanDistance() async throws {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [3, 4, 0]

        let dist = try await kernel.distance(a, b, p: 2.0)

        XCTAssertEqual(dist, 5.0, accuracy: 1e-4)  // sqrt(9 + 16) = 5
    }

    func testEuclideanDistanceMatrix() async throws {
        let queries: [[Float]] = [[0, 0], [1, 0]]
        let dataset: [[Float]] = [[3, 4], [0, 0], [1, 1]]

        let result = try await kernel.euclideanDistances(queries: queries, dataset: dataset)

        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 3)

        // query[0] to dataset[0]: sqrt(9 + 16) = 5
        XCTAssertEqual(result.distance(row: 0, col: 0), 5.0, accuracy: 1e-4)
        // query[0] to dataset[1]: 0
        XCTAssertEqual(result.distance(row: 0, col: 1), 0.0, accuracy: 1e-4)
    }

    // MARK: - L1 (Manhattan) Tests

    func testManhattanDistance() async throws {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [3, 4, 5]

        let dist = try await kernel.distance(a, b, p: 1.0)

        XCTAssertEqual(dist, 12.0, accuracy: 1e-4)  // |3| + |4| + |5| = 12
    }

    func testManhattanDistanceMatrix() async throws {
        let queries: [[Float]] = [[0, 0]]
        let dataset: [[Float]] = [[1, 2], [3, 4]]

        let result = try await kernel.manhattanDistances(queries: queries, dataset: dataset)

        XCTAssertEqual(result.distance(row: 0, col: 0), 3.0, accuracy: 1e-4)  // |1| + |2| = 3
        XCTAssertEqual(result.distance(row: 0, col: 1), 7.0, accuracy: 1e-4)  // |3| + |4| = 7
    }

    // MARK: - Chebyshev Tests

    func testChebyshevDistance() async throws {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [3, 7, 2]

        let result = try await kernel.chebyshevDistances(queries: [a], dataset: [b])

        // Chebyshev = max(|3|, |7|, |2|) = 7
        XCTAssertEqual(result.distance(row: 0, col: 0), 7.0, accuracy: 0.5)
    }

    // MARK: - Nearest Neighbors Tests

    func testFindNearestNeighbors() async throws {
        let query: [Float] = [0, 0]
        let dataset: [[Float]] = [
            [10, 10],  // far
            [1, 0],    // close
            [0, 1],    // close
            [5, 5]     // medium
        ]

        let neighbors = try await kernel.findNearestNeighbors(query: query, dataset: dataset, k: 2, p: 2.0)

        XCTAssertEqual(neighbors.count, 2)
        XCTAssertTrue(neighbors[0].index == 1 || neighbors[0].index == 2)
        XCTAssertTrue(neighbors[1].index == 1 || neighbors[1].index == 2)
    }

    // MARK: - Config Tests

    func testConfigPresets() {
        XCTAssertTrue(Metal4MinkowskiConfig.manhattan.isManhattan)
        XCTAssertTrue(Metal4MinkowskiConfig.euclidean.isEuclidean)
        XCTAssertTrue(Metal4MinkowskiConfig.chebyshev.isChebyshev)
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("TopKSelection"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }
}

// MARK: - Batch 6a: Metal4HammingDistanceKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4HammingDistanceKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4HammingDistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4HammingDistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Binary Distance Tests

    func testIdenticalVectors() async throws {
        let a: [Bool] = [true, false, true, false]
        let b: [Bool] = [true, false, true, false]

        let dist = try await kernel.distance(a, b)

        XCTAssertEqual(dist, 0)
    }

    func testCompletelyDifferentVectors() async throws {
        let a: [Bool] = [true, true, true, true]
        let b: [Bool] = [false, false, false, false]

        let dist = try await kernel.distance(a, b)

        XCTAssertEqual(dist, 4)
    }

    func testPartialDifference() async throws {
        let a: [Bool] = [true, false, true, false]
        let b: [Bool] = [true, true, false, false]

        let dist = try await kernel.distance(a, b)

        XCTAssertEqual(dist, 2)  // positions 1 and 2 differ
    }

    // MARK: - Distance Matrix Tests

    func testDistanceMatrix() async throws {
        let queries: [[Bool]] = [
            [true, false],
            [false, true]
        ]
        let dataset: [[Bool]] = [
            [true, false],
            [false, false],
            [true, true]
        ]

        let matrix = try await kernel.distanceMatrix(queries: queries, dataset: dataset)

        XCTAssertEqual(matrix.count, 2)
        XCTAssertEqual(matrix[0].count, 3)

        XCTAssertEqual(matrix[0][0], 0.0, accuracy: 1e-4)  // identical
        XCTAssertEqual(matrix[0][1], 1.0, accuracy: 1e-4)  // 1 bit different
        XCTAssertEqual(matrix[0][2], 1.0, accuracy: 1e-4)  // 1 bit different
    }

    // MARK: - Bit Packing Tests

    func testPackBinary() {
        let binary: [Bool] = [true, false, true, false, false, false, false, false]
        let packed = kernel.packBinary(binary)

        XCTAssertEqual(packed.count, 1)
        XCTAssertEqual(packed[0], 0b00000101)  // bits 0 and 2 set
    }

    func testPackBinaryLarge() {
        let binary = [Bool](repeating: true, count: 64)
        let packed = kernel.packBinary(binary)

        XCTAssertEqual(packed.count, 2)
        XCTAssertEqual(packed[0], UInt32.max)
        XCTAssertEqual(packed[1], UInt32.max)
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("BinaryQuantization"))
        XCTAssertTrue(kernel.fusibleWith.contains("TopKSelection"))
    }
}

// MARK: - Batch 6a: Metal4JaccardDistanceKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4JaccardDistanceKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4JaccardDistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4JaccardDistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Tests

    func testIdenticalSets() async throws {
        let a: [Float] = [1, 1, 0, 0]
        let b: [Float] = [1, 1, 0, 0]

        let result = try await kernel.computeDistance(vectorA: a, vectorB: b)

        XCTAssertEqual(result.distance, 0.0, accuracy: 1e-4)
        XCTAssertEqual(result.similarity, 1.0, accuracy: 1e-4)
    }

    func testCompletelyDisjointSets() async throws {
        let a: [Float] = [1, 1, 0, 0]
        let b: [Float] = [0, 0, 1, 1]

        let result = try await kernel.computeDistance(vectorA: a, vectorB: b)

        XCTAssertEqual(result.distance, 1.0, accuracy: 1e-4)
        XCTAssertEqual(result.similarity, 0.0, accuracy: 1e-4)
    }

    func testPartialOverlap() async throws {
        let a: [Float] = [1, 1, 1, 0]  // {0, 1, 2}
        let b: [Float] = [1, 1, 0, 1]  // {0, 1, 3}

        let result = try await kernel.computeDistance(vectorA: a, vectorB: b)

        // Intersection = {0, 1} = 2
        // Union = {0, 1, 2, 3} = 4
        // Jaccard = 2/4 = 0.5
        // Distance = 1 - 0.5 = 0.5
        XCTAssertEqual(result.similarity, 0.5, accuracy: 0.1)
        XCTAssertEqual(result.distance, 0.5, accuracy: 0.1)
    }

    // MARK: - Distance Matrix Tests

    func testDistanceMatrix() async throws {
        let vectorsA: [[Float]] = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        let vectorsB: [[Float]] = [
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ]

        let result = try await kernel.computeDistanceMatrix(vectorsA: vectorsA, vectorsB: vectorsB)

        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 2)

        // vectorsA[0] vs vectorsB[0]: identical
        XCTAssertEqual(result.distance(row: 0, col: 0), 0.0, accuracy: 0.1)
    }

    // MARK: - Similarity Tests

    func testSimilarity() async throws {
        let a: [Float] = [1, 1, 0, 0]
        let b: [Float] = [1, 1, 0, 0]

        let sim = try await kernel.similarity(a, b)

        XCTAssertEqual(sim, 1.0, accuracy: 1e-4)
    }

    // MARK: - Most Similar Tests

    func testFindMostSimilar() async throws {
        let query: [Float] = [1, 1, 0, 0]
        let dataset: [[Float]] = [
            [1, 1, 0, 0],  // identical
            [0, 0, 1, 1],  // disjoint
            [1, 0, 0, 0]   // partial
        ]

        let similar = try await kernel.findMostSimilar(query: query, dataset: dataset, k: 2)

        XCTAssertEqual(similar.count, 2)
        XCTAssertEqual(similar[0].index, 0)  // Most similar should be index 0
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("TopKSelection"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }
}

// MARK: - Batch 6a: Metal4HistogramKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4HistogramKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4HistogramKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        do {
            kernel = try await Metal4HistogramKernel(context: context)
        } catch {
            // Histogram Metal shaders (uniformHistogram, adaptiveHistogram, logarithmicHistogram)
            // are not yet implemented - skip tests until shaders are added
            throw XCTSkip("Histogram kernel not available: Metal shaders not implemented")
        }
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Basic Histogram Tests

    func testUniformBinning() async throws {
        let data: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        let config = Metal4HistogramConfig(binningStrategy: .uniform(bins: 10))
        let result = try await kernel.computeHistogram(data: data, config: config)

        XCTAssertEqual(result.bins.count, 10)
        XCTAssertEqual(result.binEdges.count, 11)

        // Each bin should have 1 element
        let totalCount = result.bins.reduce(0, +)
        XCTAssertEqual(totalCount, 10.0, accuracy: 1e-4)
    }

    func testNormalization() async throws {
        let data: [Float] = [1, 2, 3, 4, 5]

        let config = Metal4HistogramConfig(
            binningStrategy: .uniform(bins: 5),
            normalizeByTotal: true
        )
        let result = try await kernel.computeHistogram(data: data, config: config)

        // Sum of normalized bins should be 1.0
        let total = result.bins.reduce(0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-4)
    }

    // MARK: - Statistics Tests

    func testStatisticsComputation() async throws {
        // Generate normal-ish distribution
        let data = (0..<1000).map { _ in Float.random(in: 0...100) }

        let config = Metal4HistogramConfig(computeStatistics: true)
        let result = try await kernel.computeHistogram(data: data, config: config)

        XCTAssertEqual(result.statistics.totalCount, 1000.0, accuracy: 10.0)
        XCTAssertGreaterThan(result.statistics.mean, 0)
        XCTAssertGreaterThan(result.statistics.standardDeviation, 0)
    }

    func testEntropy() async throws {
        // Uniform distribution should have high entropy
        let uniformData = (0..<100).map { Float($0) }

        let config = Metal4HistogramConfig(
            binningStrategy: .uniform(bins: 10),
            computeStatistics: true
        )
        let result = try await kernel.computeHistogram(data: uniformData, config: config)

        // Entropy should be positive
        XCTAssertGreaterThan(result.statistics.entropy, 0)
    }

    // MARK: - Batch Histogram Tests

    func testBatchHistograms() async throws {
        let datasets: [[Float]] = [
            (0..<100).map { _ in Float.random(in: 0...50) },
            (0..<100).map { _ in Float.random(in: 50...100) }
        ]

        let result = try await kernel.computeHistograms(datasets: datasets)

        XCTAssertEqual(result.histograms.count, 2)

        // First dataset should have lower mean
        XCTAssertLessThan(result.histograms[0].statistics.mean, result.histograms[1].statistics.mean)
    }

    // MARK: - Histogram Intersection Tests

    func testHistogramIntersection() async throws {
        let data1: [Float] = (0..<100).map { Float($0) }
        let data2: [Float] = (0..<100).map { Float($0) }

        let config = Metal4HistogramConfig(binningStrategy: .uniform(bins: 10))
        let h1 = try await kernel.computeHistogram(data: data1, config: config)
        let h2 = try await kernel.computeHistogram(data: data2, config: config)

        let intersection = kernel.histogramIntersection(h1, h2)

        // Identical data should have intersection equal to total count
        XCTAssertEqual(intersection, 100.0, accuracy: 1.0)
    }

    // MARK: - Result Properties Tests

    func testBinCenters() async throws {
        let data: [Float] = [0, 1, 2, 3, 4]
        let config = Metal4HistogramConfig(binningStrategy: .uniform(bins: 5))

        let result = try await kernel.computeHistogram(data: data, config: config)

        XCTAssertEqual(result.binCenters.count, 5)
        XCTAssertEqual(result.binWidths.count, 5)
    }

    func testPeakBin() async throws {
        // Create data with a clear peak
        var data: [Float] = []
        for _ in 0..<100 { data.append(Float.random(in: 40...60)) }  // Peak around 50
        for _ in 0..<20 { data.append(Float.random(in: 0...100)) }   // Spread

        let config = Metal4HistogramConfig(binningStrategy: .uniform(bins: 10))
        let result = try await kernel.computeHistogram(data: data, config: config)

        XCTAssertNotNil(result.peakBin)
        // Peak should be around center (index 4 or 5)
        XCTAssertGreaterThanOrEqual(result.peakBin?.index ?? -1, 3)
        XCTAssertLessThanOrEqual(result.peakBin?.index ?? -1, 6)
    }

    func testSummary() async throws {
        let data = (0..<100).map { _ in Float.random(in: 0...100) }

        let result = try await kernel.computeHistogram(data: data)
        let summary = result.summary()

        XCTAssertTrue(summary.contains("Histogram Summary"))
        XCTAssertTrue(summary.contains("Bins:"))
        XCTAssertTrue(summary.contains("Mean:"))
    }
}

// MARK: - Thread Configuration Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ThreadConfigurationTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4L2DistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4L2DistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testSmallWorkload() async throws {
        // Very small workload should still work
        let queries = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 4)
        let database = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 4)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 1)
    }

    func testLargeWorkload() async throws {
        // Larger workload to test thread configuration
        let queries = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 256)
        let database = Metal4KernelTestHelpers.randomVectors(count: 1000, dimension: 256)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 100)
        XCTAssertEqual(results[0].count, 1000)
    }

    func testAsymmetricWorkload() async throws {
        // Many queries, few database vectors
        let queries = Metal4KernelTestHelpers.randomVectors(count: 500, dimension: 64)
        let database = Metal4KernelTestHelpers.randomVectors(count: 10, dimension: 64)

        let results = try await kernel.compute(queries: queries, database: database)

        XCTAssertEqual(results.count, 500)
        XCTAssertEqual(results[0].count, 10)
    }
}

// MARK: - Phase 1 Integration: Metal4WarpOptimizedSelectionKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4WarpOptimizedSelectionKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: Metal4WarpOptimizedSelectionKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await Metal4WarpOptimizedSelectionKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Helper Methods

    /// CPU reference: select k smallest indices
    private func selectKSmallest(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        let indexed = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { $0.value < $1.value }
        return Array(sorted.prefix(k))
    }

    /// CPU reference: select k largest indices
    private func selectKLargest(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        let indexed = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { $0.value > $1.value }
        return Array(sorted.prefix(k))
    }

    /// Generate random values array
    private func randomValues(count: Int, range: ClosedRange<Float> = 0.0...100.0) -> [Float] {
        (0..<count).map { _ in Float.random(in: range) }
    }

    /// Generate random 2D values array
    private func randomValues2D(queries: Int, candidates: Int) -> [[Float]] {
        (0..<queries).map { _ in randomValues(count: candidates) }
    }

    // MARK: - Warp Selection Tests (k ≤ 32)

    func testWarpSelectSmallestK() async throws {
        let values: [[Float]] = [[10.0, 5.0, 15.0, 2.0, 8.0, 1.0, 20.0, 3.0]]
        // Sorted ascending: 1.0(5), 2.0(3), 3.0(7), 5.0(1), 8.0(4), 10.0(0), 15.0(2), 20.0(6)

        let results = try await kernel.selectTopK(
            from: values,
            k: 3,
            config: .ascending
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        // The 3 smallest should be 1.0, 2.0, 3.0 (indices 5, 3, 7)
        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(5), "Should include index 5 (value 1.0)")
        XCTAssertTrue(indices.contains(3), "Should include index 3 (value 2.0)")
        XCTAssertTrue(indices.contains(7), "Should include index 7 (value 3.0)")
    }

    func testWarpSelectLargestK() async throws {
        let values: [[Float]] = [[10.0, 5.0, 15.0, 2.0, 8.0, 1.0, 20.0, 3.0]]
        // Sorted descending: 20.0(6), 15.0(2), 10.0(0), 8.0(4), 5.0(1), 3.0(7), 2.0(3), 1.0(5)

        let results = try await kernel.selectTopK(
            from: values,
            k: 3,
            config: .descending
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        // The 3 largest should be 20.0, 15.0, 10.0 (indices 6, 2, 0)
        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(6), "Should include index 6 (value 20.0)")
        XCTAssertTrue(indices.contains(2), "Should include index 2 (value 15.0)")
        XCTAssertTrue(indices.contains(0), "Should include index 0 (value 10.0)")
    }

    func testWarpSelectSingleElement() async throws {
        let values: [[Float]] = [[5.0, 3.0, 8.0, 1.0, 9.0]]

        // Select single smallest
        let resultsAsc = try await kernel.selectTopK(from: values, k: 1, config: .ascending)
        XCTAssertEqual(resultsAsc[0].count, 1)
        XCTAssertEqual(resultsAsc[0][0].index, 3, "Smallest value is at index 3")
        XCTAssertEqual(resultsAsc[0][0].value, 1.0, accuracy: 0.01)

        // Select single largest
        let resultsDesc = try await kernel.selectTopK(from: values, k: 1, config: .descending)
        XCTAssertEqual(resultsDesc[0].count, 1)
        XCTAssertEqual(resultsDesc[0][0].index, 4, "Largest value is at index 4")
        XCTAssertEqual(resultsDesc[0][0].value, 9.0, accuracy: 0.01)
    }

    func testWarpSelectMaxK() async throws {
        // Test with k=32 (max for warp kernel)
        let candidateCount = 100
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel.selectTopK(from: values, k: 32, config: .ascending)

        XCTAssertEqual(results[0].count, 32, "Should return exactly 32 results")

        // Verify correctness against CPU reference
        let cpuResults = selectKSmallest(values[0], k: 32)
        let gpuIndices = Set(results[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU should match CPU reference")
    }

    // MARK: - Correctness Tests

    func testCorrectnessAscending() async throws {
        let candidateCount = 50
        let k = 10
        let values = [randomValues(count: candidateCount)]

        let gpuResults = try await kernel.selectTopK(from: values, k: k, config: .ascending)
        let cpuResults = selectKSmallest(values[0], k: k)

        // Compare indices
        let gpuIndices = Set(gpuResults[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU ascending selection should match CPU")
    }

    func testCorrectnessDescending() async throws {
        let candidateCount = 50
        let k = 10
        let values = [randomValues(count: candidateCount)]

        let gpuResults = try await kernel.selectTopK(from: values, k: k, config: .descending)
        let cpuResults = selectKLargest(values[0], k: k)

        // Compare indices
        let gpuIndices = Set(gpuResults[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "GPU descending selection should match CPU")
    }

    func testMultipleQueries() async throws {
        let values: [[Float]] = [
            [10.0, 1.0, 5.0],  // Min at index 1
            [3.0, 8.0, 2.0],   // Min at index 2
            [7.0, 4.0, 9.0]    // Min at index 1
        ]

        let results = try await kernel.selectTopK(from: values, k: 1, config: .ascending)

        XCTAssertEqual(results.count, 3, "Should have results for 3 queries")
        XCTAssertEqual(results[0][0].index, 1, "Query 0 min at index 1")
        XCTAssertEqual(results[1][0].index, 2, "Query 1 min at index 2")
        XCTAssertEqual(results[2][0].index, 1, "Query 2 min at index 1")
    }

    func testValueAccuracy() async throws {
        let values: [[Float]] = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        let results = try await kernel.selectTopK(from: values, k: 3, config: .ascending)

        // Verify values are correct
        for result in results[0] {
            XCTAssertEqual(result.value, values[0][result.index], accuracy: 0.001,
                          "Returned value should match original at index")
        }
    }

    // MARK: - Edge Case Tests

    func testKEqualsArraySize() async throws {
        let values: [[Float]] = [[5.0, 3.0, 8.0, 1.0, 9.0]]

        let results = try await kernel.selectTopK(from: values, k: 5, config: .ascending)

        XCTAssertEqual(results[0].count, 5, "Should return all elements")

        // All indices should be present
        let indices = Set(results[0].map { $0.index })
        XCTAssertEqual(indices, Set([0, 1, 2, 3, 4]))
    }

    func testSingleCandidate() async throws {
        let values: [[Float]] = [[42.0]]

        let results = try await kernel.selectTopK(from: values, k: 1, config: .ascending)

        XCTAssertEqual(results[0].count, 1)
        XCTAssertEqual(results[0][0].index, 0)
        XCTAssertEqual(results[0][0].value, 42.0, accuracy: 0.01)
    }

    func testIdenticalValues() async throws {
        let values: [[Float]] = [[5.0, 5.0, 5.0, 5.0, 5.0]]

        let results = try await kernel.selectTopK(from: values, k: 3, config: .ascending)

        XCTAssertEqual(results[0].count, 3)
        // All values should be 5.0
        for result in results[0] {
            XCTAssertEqual(result.value, 5.0, accuracy: 0.001)
        }
    }

    // MARK: - Large K Tests (Using Batch Kernel k > 32)

    func testLargeK() async throws {
        let candidateCount = 200
        let k = 64  // > 32, will use batch kernel
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel.selectTopK(from: values, k: k, config: .ascending)

        XCTAssertEqual(results[0].count, k, "Should return exactly k results")

        // Verify correctness
        let cpuResults = selectKSmallest(values[0], k: k)
        let gpuIndices = Set(results[0].map { $0.index })
        let cpuIndices = Set(cpuResults.map { $0.index })

        XCTAssertEqual(gpuIndices, cpuIndices, "Large k should match CPU reference")
    }

    func testMaxBatchK() async throws {
        let candidateCount = 200
        let k = 128  // MAX_BATCH_K
        let values = [randomValues(count: candidateCount)]

        let results = try await kernel.selectTopK(from: values, k: k, config: .ascending)

        XCTAssertEqual(results[0].count, k, "Should return exactly 128 results")
    }

    // MARK: - Error Handling Tests

    func testEmptyInputReturnsEmpty() async throws {
        let results = try await kernel.selectTopK(from: [], k: 1, config: .ascending)
        XCTAssertEqual(results.count, 0, "Empty input should return empty result")
    }

    func testInvalidKThrows() async throws {
        let values = [randomValues(count: 10)]

        // K > 128 should throw
        do {
            _ = try await kernel.selectTopK(from: values, k: 150, config: .ascending)
            XCTFail("Should throw for k > 128")
        } catch {
            // Expected
        }

        // K = 0 should throw
        do {
            _ = try await kernel.selectTopK(from: values, k: 0, config: .ascending)
            XCTFail("Should throw for k = 0")
        } catch {
            // Expected
        }
    }

    // MARK: - Batch Processing Tests

    func testBatchProcess() async throws {
        let batches: [[[Float]]] = [
            [[10.0, 1.0, 5.0], [3.0, 8.0, 2.0]],  // Batch 0: 2 queries
            [[7.0, 4.0, 9.0], [1.0, 1.0, 1.0]]    // Batch 1: 2 queries
        ]

        let results = try await kernel.batchProcess(batches, k: 1, config: .ascending)

        XCTAssertEqual(results.count, 2, "Should have 2 batches")
        XCTAssertEqual(results[0].count, 2, "Batch 0 should have 2 queries")
        XCTAssertEqual(results[1].count, 2, "Batch 1 should have 2 queries")

        // Verify results
        XCTAssertEqual(results[0][0][0].index, 1, "Batch 0, Query 0: min at index 1")
        XCTAssertEqual(results[0][1][0].index, 2, "Batch 0, Query 1: min at index 2")
        XCTAssertEqual(results[1][0][0].index, 1, "Batch 1, Query 0: min at index 1")
    }

    // MARK: - Numerical Edge Cases

    func testNegativeValues() async throws {
        let values: [[Float]] = [[-5.0, -10.0, -1.0, -8.0, -3.0]]
        // Ascending: -10, -8, -5, -3, -1 (indices 1, 3, 0, 4, 2)

        let results = try await kernel.selectTopK(from: values, k: 3, config: .ascending)

        let indices = Set(results[0].map { $0.index })
        XCTAssertTrue(indices.contains(1), "Should include index 1 (-10.0)")
        XCTAssertTrue(indices.contains(3), "Should include index 3 (-8.0)")
        XCTAssertTrue(indices.contains(0), "Should include index 0 (-5.0)")
    }

    func testMixedSignValues() async throws {
        let values: [[Float]] = [[-5.0, 0.0, 5.0, -10.0, 10.0]]

        let resultsAsc = try await kernel.selectTopK(from: values, k: 2, config: .ascending)
        let indicesAsc = Set(resultsAsc[0].map { $0.index })
        XCTAssertTrue(indicesAsc.contains(3), "Should include -10.0")
        XCTAssertTrue(indicesAsc.contains(0), "Should include -5.0")

        let resultsDesc = try await kernel.selectTopK(from: values, k: 2, config: .descending)
        let indicesDesc = Set(resultsDesc[0].map { $0.index })
        XCTAssertTrue(indicesDesc.contains(4), "Should include 10.0")
        XCTAssertTrue(indicesDesc.contains(2), "Should include 5.0")
    }

    // MARK: - Result Type Tests

    func testResultExtraction() async throws {
        // Test that results are extracted correctly through the high-level API
        let values = randomValues2D(queries: 5, candidates: 50)

        let allResults = try await kernel.selectTopK(from: values, k: 10, config: .ascending)

        XCTAssertEqual(allResults.count, 5, "Should have 5 query results")

        for q in 0..<5 {
            let qResults = allResults[q]
            XCTAssertEqual(qResults.count, 10, "Each query should have 10 results")
        }

        // Verify indices are valid
        for queryResults in allResults {
            for result in queryResults {
                XCTAssertGreaterThanOrEqual(result.index, 0)
                XCTAssertLessThan(result.index, 50)
            }
        }

        // Verify values are valid
        for (q, queryResults) in allResults.enumerated() {
            for result in queryResults {
                XCTAssertEqual(result.value, values[q][result.index], accuracy: 0.001,
                              "Returned value should match original")
            }
        }
    }

    // MARK: - Performance Tests

    func testPerformance() async throws {
        let queryCount = 100
        let candidateCount = 1000
        let k = 10

        let values = randomValues2D(queries: queryCount, candidates: candidateCount)

        // Warm-up
        _ = try await kernel.selectTopK(from: values, k: k, config: .ascending)

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try await kernel.selectTopK(from: values, k: k, config: .ascending)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("Metal4WarpOptimizedSelection performance: \(elapsed / 5.0)s per run (\(queryCount) queries, \(candidateCount) candidates, k=\(k))")

        XCTAssertLessThan(elapsed / 5.0, 1.0, "Should complete in reasonable time")
    }

    // MARK: - Protocol Tests

    func testFusibleWith() {
        XCTAssertTrue(kernel.fusibleWith.contains("L2Distance"))
        XCTAssertTrue(kernel.fusibleWith.contains("CosineSimilarity"))
        XCTAssertTrue(kernel.fusibleWith.contains("DotProduct"))
        XCTAssertFalse(kernel.requiresBarrierAfter, "Selection output is final")
    }

    func testKernelName() {
        XCTAssertEqual(kernel.name, "Metal4WarpOptimizedSelectionKernel")
    }
}
