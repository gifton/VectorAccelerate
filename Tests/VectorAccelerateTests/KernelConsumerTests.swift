//
//  KernelConsumerTests.swift
//  VectorAccelerateTests
//
//  Tests verifying the public API exposure of GPU kernel primitives.
//
//  These tests ensure that consumers can:
//  1. Import and instantiate all kernels directly
//  2. Use type aliases for common kernels
//  3. Access kernel-specific features
//  4. Compose kernels in pipelines
//

import XCTest
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KernelConsumerTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
    }

    override func tearDown() async throws {
        context = nil
        try await super.tearDown()
    }

    // MARK: - Kernel Instantiation Tests

    func testDistanceKernelInstantiation() async throws {
        // All distance kernels should be publicly instantiable
        _ = try await L2DistanceKernel(context: context)
        _ = try await CosineSimilarityKernel(context: context)
        _ = try await DotProductKernel(context: context)
        _ = try await MinkowskiDistanceKernel(context: context)
        _ = try await HammingDistanceKernel(context: context)
        _ = try await JaccardDistanceKernel(context: context)
        _ = try await AttentionSimilarityKernel(context: context)
    }

    func testSelectionKernelInstantiation() async throws {
        // All selection kernels should be publicly instantiable
        _ = try await TopKSelectionKernel(context: context)
        _ = try await FusedL2TopKKernel(context: context)
        _ = try await StreamingTopKKernel(context: context)
        _ = try await WarpOptimizedSelectionKernel(context: context)
    }

    func testQuantizationKernelInstantiation() async throws {
        // All quantization kernels should be publicly instantiable
        _ = try await ScalarQuantizationKernel(context: context)
        _ = try await BinaryQuantizationKernel(context: context)
        _ = try await ProductQuantizationKernel(context: context)
        _ = try await NeuralQuantizationKernel(context: context)
    }

    func testMatrixKernelInstantiation() async throws {
        // All matrix kernels should be publicly instantiable
        _ = try await MatrixMultiplyKernel(context: context)
        _ = try await MatrixVectorKernel(context: context)
        _ = try await MatrixTransposeKernel(context: context)
        _ = try await BatchMatrixKernel(context: context)
    }

    func testUtilityKernelInstantiation() async throws {
        // All utility kernels should be publicly instantiable
        _ = try await StatisticsKernel(context: context)
        _ = try await HistogramKernel(context: context)
        _ = try await ElementwiseKernel(context: context)
        _ = try await ParallelReductionKernel(context: context)
        _ = try await L2NormalizationKernel(context: context)
    }

    // MARK: - Type Alias Tests

    func testContextTypeAliases() async throws {
        // GPUContext should be an alias for Metal4Context
        let gpuContext: GPUContext = try await Metal4Context()
        XCTAssertNotNil(gpuContext)

        // GPUConfiguration should be an alias for Metal4Configuration
        let config: GPUConfiguration = Metal4Configuration()
        XCTAssertTrue(config.preferHighPerformanceDevice)
    }

    func testKernelTypeAliases() async throws {
        // Distance kernel aliases
        _ = try await L2Kernel(context: context)
        _ = try await CosineKernel(context: context)
        _ = try await DotKernel(context: context)
        _ = try await MinkowskiKernel(context: context)
        _ = try await HammingKernel(context: context)
        _ = try await JaccardKernel(context: context)

        // Selection kernel aliases
        _ = try await TopKKernel(context: context)
        _ = try await FusedTopKKernel(context: context)
        _ = try await StreamingKernel(context: context)

        // Quantization kernel aliases
        _ = try await ScalarQuantKernel(context: context)
        _ = try await BinaryQuantKernel(context: context)
        _ = try await PQKernel(context: context)
    }

    // MARK: - Basic Kernel Functionality Tests

    func testL2DistanceComputation() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        let queries: [[Float]] = [[1.0, 0.0, 0.0]]
        let database: [[Float]] = [
            [1.0, 0.0, 0.0],  // Distance = 0
            [0.0, 1.0, 0.0],  // Distance = sqrt(2)
            [0.0, 0.0, 1.0]   // Distance = sqrt(2)
        ]

        let distances = try await kernel.compute(
            queries: queries,
            database: database,
            computeSqrt: true
        )

        XCTAssertEqual(distances.count, 1)
        XCTAssertEqual(distances[0].count, 3)
        XCTAssertEqual(distances[0][0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(distances[0][1], sqrt(2), accuracy: 1e-4)
        XCTAssertEqual(distances[0][2], sqrt(2), accuracy: 1e-4)
    }

    func testCosineSimilarityComputation() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)

        let queries: [[Float]] = [[1.0, 0.0, 0.0]]
        let database: [[Float]] = [
            [1.0, 0.0, 0.0],   // Similarity = 1.0 (same direction)
            [0.0, 1.0, 0.0],   // Similarity = 0.0 (orthogonal)
            [-1.0, 0.0, 0.0]   // Similarity = -1.0 (opposite)
        ]

        let similarities = try await kernel.compute(
            queries: queries,
            database: database,
            outputDistance: false,
            inputsNormalized: false
        )

        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(similarities[0][1], 0.0, accuracy: 1e-4)
        XCTAssertEqual(similarities[0][2], -1.0, accuracy: 1e-4)
    }

    func testTopKSelection() async throws {
        let kernel = try await TopKSelectionKernel(context: context)

        let values: [[Float]] = [
            [5.0, 2.0, 8.0, 1.0, 9.0, 3.0]
        ]

        let results = try await kernel.select(
            from: values,
            k: 3,
            mode: .minimum,
            sorted: true
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 3)

        // Should return indices of 3 smallest: 1.0, 2.0, 3.0
        let topValues = results[0].map { $0.1 }
        XCTAssertTrue(topValues.contains { abs($0 - 1.0) < 0.01 })
        XCTAssertTrue(topValues.contains { abs($0 - 2.0) < 0.01 })
        XCTAssertTrue(topValues.contains { abs($0 - 3.0) < 0.01 })
    }

    func testBinaryQuantization() async throws {
        let kernel = try await BinaryQuantizationKernel(context: context)

        // Use 64 elements to properly demonstrate compression
        // 64 floats = 256 bytes -> 64 bits = 8 bytes = 32x compression
        let vectors: [[Float]] = [
            (0..<64).map { i in Float(i % 2 == 0 ? 1.0 : -1.0) }
        ]

        let result = try await kernel.quantize(
            vectors: vectors,
            config: .init(useSignBit: true)
        )

        // Binary quantization achieves ~32x compression (Float32 -> 1 bit)
        XCTAssertGreaterThan(result.compressionRatio, 20.0)
        XCTAssertEqual(result.binaryVectors.count, 1)
    }

    func testStatisticsComputation() async throws {
        let kernel = try await StatisticsKernel(context: context)

        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let stats = try await kernel.computeStatistics(data, config: .basic)

        XCTAssertEqual(stats.basic.count, 10)
        XCTAssertEqual(stats.basic.mean, 5.5, accuracy: 0.01)
        XCTAssertEqual(stats.basic.minimum, 1.0, accuracy: 0.01)
        XCTAssertEqual(stats.basic.maximum, 10.0, accuracy: 0.01)
    }

    func testElementwiseOperations() async throws {
        let kernel = try await ElementwiseKernel(context: context)

        let a: [Float] = [1, 2, 3, 4, 5]
        let b: [Float] = [5, 4, 3, 2, 1]

        let sum = try await kernel.add(a, b)
        XCTAssertEqual(sum, [6, 6, 6, 6, 6])

        let scaled = try await kernel.scale(a, by: 2.0)
        XCTAssertEqual(scaled, [2, 4, 6, 8, 10])
    }

    func testL2Normalization() async throws {
        let kernel = try await L2NormalizationKernel(context: context)

        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0]  // norm = 5
        ]

        let result = try await kernel.normalize(vectors, storeNorms: true)
        let normalized = result.asArrays()
        let norms = result.normsAsArray()

        XCTAssertEqual(norms?[0] ?? 0, 5.0, accuracy: 0.01)
        XCTAssertEqual(normalized[0][0], 0.6, accuracy: 0.01)  // 3/5
        XCTAssertEqual(normalized[0][1], 0.8, accuracy: 0.01)  // 4/5
    }

    // MARK: - VectorCore Integration Tests

    func testKernelWithDynamicVector() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let database = [
            DynamicVector([1.0, 0.0, 0.0]),
            DynamicVector([0.0, 1.0, 0.0])
        ]

        let distances = try await kernel.compute(queries: [query], database: database)

        XCTAssertEqual(distances[0][0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(distances[0][1], sqrt(2), accuracy: 1e-4)
    }

    // MARK: - Distance Metric Dispatch Tests

    func testDistanceKernelDispatch() async throws {
        // Euclidean -> L2DistanceKernel
        let euclideanKernel = try await context.distanceKernel(for: .euclidean)
        XCTAssertEqual(euclideanKernel.name, "L2DistanceKernel")

        // Cosine -> CosineSimilarityKernel
        let cosineKernel = try await context.distanceKernel(for: .cosine)
        XCTAssertEqual(cosineKernel.name, "CosineSimilarityKernel")

        // DotProduct -> DotProductKernel
        let dotKernel = try await context.distanceKernel(for: .dotProduct)
        XCTAssertEqual(dotKernel.name, "DotProductKernel")

        // Manhattan -> MinkowskiDistanceKernel
        let manhattanKernel = try await context.distanceKernel(for: .manhattan)
        XCTAssertEqual(manhattanKernel.name, "MinkowskiDistanceKernel")
    }

    // MARK: - Pipeline Composition Tests

    func testFusedL2TopK() async throws {
        let kernel = try await FusedL2TopKKernel(context: context)

        let queries: [[Float]] = [[0.5, 0.5, 0.5, 0.5]]
        let database: [[Float]] = (0..<10).map { i in
            let offset = Float(i) * 0.1
            return [0.5 + offset, 0.5, 0.5, 0.5]
        }

        let neighbors = try await kernel.findNearestNeighbors(
            queries: queries,
            dataset: database,
            k: 3
        )

        XCTAssertEqual(neighbors.count, 1)
        XCTAssertEqual(neighbors[0].count, 3)

        // First neighbor should be database[0] (exact match or closest)
        XCTAssertEqual(neighbors[0][0].0, 0)  // Index 0 should be closest
    }

    // MARK: - Dimension Optimization Tests

    func testL2KernelDimensionOptimizations() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        // L2DistanceKernel has optimized paths for these dimensions
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 384))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 512))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 768))
        XCTAssertTrue(kernel.hasOptimizedPipeline(for: 1536))

        // Non-optimized dimension should still work
        XCTAssertFalse(kernel.hasOptimizedPipeline(for: 100))
    }
}
