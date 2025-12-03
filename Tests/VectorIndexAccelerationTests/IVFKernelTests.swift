//
//  IVFKernelTests.swift
//  VectorIndexAcceleration
//
//  Tests for IVF (Inverted File) GPU kernels.
//
//  Tests cover:
//  - IVFSearchConfiguration validation and presets
//  - IVFCoarseQuantizerKernel functionality
//  - IVFSearchPipeline end-to-end search
//  - IVFGPUIndexStructure construction
//

import XCTest
@testable import VectorIndexAcceleration
@testable import VectorAccelerate
import VectorIndex
import VectorCore
@preconcurrency import Metal

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFTestHelpers {
    /// Generate random vectors
    static func randomVectors(count: Int, dimension: Int, seed: UInt64 = 42) -> [[Float]] {
        srand48(Int(seed))
        return (0..<count).map { _ in
            (0..<dimension).map { _ in Float(drand48() * 2 - 1) }
        }
    }

    /// Generate clustered vectors around centroids
    static func clusteredVectors(
        centroids: [[Float]],
        pointsPerCluster: Int,
        noise: Float = 0.1,
        seed: UInt64 = 42
    ) -> (vectors: [[Float]], labels: [Int]) {
        srand48(Int(seed))
        var vectors: [[Float]] = []
        var labels: [Int] = []
        let dimension = centroids[0].count

        for (clusterIdx, centroid) in centroids.enumerated() {
            for _ in 0..<pointsPerCluster {
                var vec = centroid.map { $0 + Float(drand48() * 2 - 1) * noise }
                // Ensure dimension matches
                while vec.count < dimension { vec.append(0) }
                vectors.append(vec)
                labels.append(clusterIdx)
            }
        }

        return (vectors, labels)
    }

    /// CPU L2 distance (squared)
    static func cpuL2DistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) }
    }

    /// CPU L2 distance
    static func cpuL2Distance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(cpuL2DistanceSquared(a, b))
    }

    /// Find k nearest neighbors using brute force CPU search
    static func cpuKNN(query: [Float], candidates: [[Float]], k: Int) -> [(index: Int, distance: Float)] {
        var distances: [(index: Int, distance: Float)] = candidates.enumerated().map { (idx, vec) in
            (idx, cpuL2DistanceSquared(query, vec))
        }
        distances.sort { $0.distance < $1.distance }
        return Array(distances.prefix(k))
    }
}

// MARK: - IVFSearchConfiguration Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFSearchConfigurationTests: XCTestCase {

    func testDefaultInitialization() {
        let config = IVFSearchConfiguration(
            numCentroids: 256,
            nprobe: 8,
            dimension: 128
        )

        XCTAssertEqual(config.numCentroids, 256)
        XCTAssertEqual(config.nprobe, 8)
        XCTAssertEqual(config.dimension, 128)
        XCTAssertEqual(config.metric, .euclidean)
        XCTAssertFalse(config.enableProfiling)
    }

    func testNprobeClampedToNumCentroids() {
        let config = IVFSearchConfiguration(
            numCentroids: 10,
            nprobe: 100,  // Should be clamped to 10
            dimension: 64
        )

        XCTAssertEqual(config.nprobe, 10)
    }

    func testValidationSuccess() throws {
        let config = IVFSearchConfiguration(
            numCentroids: 256,
            nprobe: 8,
            dimension: 128
        )

        XCTAssertNoThrow(try config.validate())
    }

    func testValidationFailsForZeroCentroids() {
        let config = IVFSearchConfiguration(
            numCentroids: 0,
            nprobe: 1,
            dimension: 128
        )

        XCTAssertThrowsError(try config.validate())
    }

    func testValidationFailsForZeroDimension() {
        let config = IVFSearchConfiguration(
            numCentroids: 256,
            nprobe: 8,
            dimension: 0
        )

        XCTAssertThrowsError(try config.validate())
    }

    func testValidationFailsForExcessiveDimension() {
        let config = IVFSearchConfiguration(
            numCentroids: 256,
            nprobe: 8,
            dimension: 5000  // > 4096
        )

        XCTAssertThrowsError(try config.validate())
    }

    func testSmallPreset() {
        let config = IVFSearchConfiguration.small(dimension: 128)

        XCTAssertEqual(config.numCentroids, 256)
        XCTAssertEqual(config.nprobe, 8)
        XCTAssertEqual(config.dimension, 128)
    }

    func testStandardPreset() {
        let config = IVFSearchConfiguration.standard(dimension: 256)

        XCTAssertEqual(config.numCentroids, 1024)
        XCTAssertEqual(config.nprobe, 16)
        XCTAssertEqual(config.dimension, 256)
    }

    func testLargePreset() {
        let config = IVFSearchConfiguration.large(dimension: 512)

        XCTAssertEqual(config.numCentroids, 4096)
        XCTAssertEqual(config.nprobe, 32)
        XCTAssertEqual(config.dimension, 512)
    }

    func testHighRecallPreset() {
        let config = IVFSearchConfiguration.highRecall(
            numCentroids: 256,
            dimension: 128
        )

        // nprobe should be numCentroids / 8 = 32
        XCTAssertEqual(config.nprobe, 32)
    }
}

// MARK: - IVFCoarseQuantizerKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFCoarseQuantizerKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: IVFCoarseQuantizerKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await IVFCoarseQuantizerKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testFindNearestCentroidsSingleQuery() async throws {
        // Create simple centroids at known positions
        let centroids: [[Float]] = [
            [0, 0, 0, 0],   // Centroid 0 at origin
            [10, 0, 0, 0],  // Centroid 1
            [0, 10, 0, 0],  // Centroid 2
            [0, 0, 10, 0]   // Centroid 3
        ]

        // Query near centroid 0
        let queries: [[Float]] = [
            [0.1, 0.1, 0.1, 0.1]
        ]

        let results = try await kernel.findNearestCentroids(
            queries: queries,
            centroids: centroids,
            nprobe: 2
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].count, 2)

        // Nearest should be centroid 0
        XCTAssertEqual(results[0][0], 0)
    }

    func testFindNearestCentroidsMultipleQueries() async throws {
        let dimension = 8
        let numCentroids = 16
        let numQueries = 4
        let nprobe = 4

        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let queries = IVFTestHelpers.randomVectors(count: numQueries, dimension: dimension, seed: 123)

        let results = try await kernel.findNearestCentroids(
            queries: queries,
            centroids: centroids,
            nprobe: nprobe
        )

        XCTAssertEqual(results.count, numQueries)
        for queryResult in results {
            XCTAssertEqual(queryResult.count, nprobe)

            // Verify indices are valid
            for idx in queryResult {
                XCTAssertGreaterThanOrEqual(idx, 0)
                XCTAssertLessThan(idx, numCentroids)
            }

            // Verify no duplicates
            XCTAssertEqual(Set(queryResult).count, nprobe)
        }
    }

    func testFindNearestCentroidsMatchesCPU() async throws {
        let dimension = 4
        let numCentroids = 8
        let nprobe = 3

        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let queries: [[Float]] = [[0.5, 0.5, 0.5, 0.5]]

        let gpuResults = try await kernel.findNearestCentroids(
            queries: queries,
            centroids: centroids,
            nprobe: nprobe
        )

        // CPU reference
        let cpuResults = IVFTestHelpers.cpuKNN(query: queries[0], candidates: centroids, k: nprobe)
        let cpuIndices = cpuResults.map { $0.index }

        // GPU should find the same nearest centroids
        XCTAssertEqual(Set(gpuResults[0]), Set(cpuIndices))
    }

    func testFindNearestCentroidsWithNprobeEqualsCentroids() async throws {
        let dimension = 4
        let numCentroids = 4

        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let queries = IVFTestHelpers.randomVectors(count: 2, dimension: dimension)

        // nprobe = numCentroids means return all centroids
        let results = try await kernel.findNearestCentroids(
            queries: queries,
            centroids: centroids,
            nprobe: numCentroids
        )

        for queryResult in results {
            XCTAssertEqual(queryResult.count, numCentroids)
            // Should contain all centroid indices
            XCTAssertEqual(Set(queryResult), Set(0..<numCentroids))
        }
    }
}

// MARK: - IVFSearchPipeline Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFSearchPipelineTests: XCTestCase {

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

    func testPrepareIndexStructure() async throws {
        let dimension = 4
        let numCentroids = 3

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 2,
            dimension: dimension
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        // Create simple centroids and lists
        let centroids: [[Float]] = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]

        let lists: [[[Float]]] = [
            [[0.1, 0, 0, 0], [0.2, 0, 0, 0]],  // List 0: 2 vectors
            [[1.1, 0, 0, 0]],                   // List 1: 1 vector
            [[0, 1.1, 0, 0], [0, 1.2, 0, 0], [0, 1.3, 0, 0]]  // List 2: 3 vectors
        ]

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        XCTAssertEqual(structure.numCentroids, numCentroids)
        XCTAssertEqual(structure.dimension, dimension)
        XCTAssertEqual(structure.totalVectors, 6)  // 2 + 1 + 3

        // Verify list counts
        XCTAssertEqual(structure.listCount(at: 0), 2)
        XCTAssertEqual(structure.listCount(at: 1), 1)
        XCTAssertEqual(structure.listCount(at: 2), 3)
    }

    func testPrepareIndexStructureWithEmptyLists() async throws {
        let dimension = 4
        let numCentroids = 3

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 2,
            dimension: dimension
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        let centroids: [[Float]] = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]

        // Some lists are empty
        let lists: [[[Float]]] = [
            [[0.1, 0, 0, 0]],  // List 0: 1 vector
            [],                // List 1: empty
            [[0, 1.1, 0, 0]]   // List 2: 1 vector
        ]

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        XCTAssertEqual(structure.totalVectors, 2)
        XCTAssertEqual(structure.listCount(at: 0), 1)
        XCTAssertEqual(structure.listCount(at: 1), 0)
        XCTAssertEqual(structure.listCount(at: 2), 1)
    }

    func testSearchSimple() async throws {
        let dimension = 4
        let numCentroids = 2

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 2,  // Search all lists
            dimension: dimension
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        // Two clusters: one at origin, one at (10,0,0,0)
        let centroids: [[Float]] = [
            [0, 0, 0, 0],
            [10, 0, 0, 0]
        ]

        // Vectors close to each centroid
        let lists: [[[Float]]] = [
            [[0.1, 0.1, 0, 0], [0.2, 0, 0.1, 0], [-0.1, 0.1, 0, 0]],  // Near origin
            [[10.1, 0, 0, 0], [9.9, 0.1, 0, 0]]  // Near (10,0,0,0)
        ]

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        // Query near origin
        let queries: [[Float]] = [[0, 0, 0, 0]]
        let k = 2

        let result = try await pipeline.search(
            queries: queries,
            structure: structure,
            k: k
        )

        XCTAssertEqual(result.numQueries, 1)
        XCTAssertEqual(result.k, k)
        XCTAssertGreaterThan(result.executionTime, 0)

        // Should find vectors from list 0 (near origin)
        let queryResults = result.results(for: 0)
        XCTAssertEqual(queryResults.count, k)

        // Distances should be small (near origin)
        for (_, distance) in queryResults {
            XCTAssertLessThan(distance, 1.0)
        }
    }

    func testSearchBatch() async throws {
        let dimension = 4
        let numCentroids = 4
        let k = 3

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 2,
            dimension: dimension
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        // Generate clustered data
        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let (vectors, _) = IVFTestHelpers.clusteredVectors(
            centroids: centroids,
            pointsPerCluster: 10,
            noise: 0.2
        )

        // Build lists from clustered vectors
        var lists: [[[Float]]] = Array(repeating: [], count: numCentroids)
        for (i, vec) in vectors.enumerated() {
            let listIdx = i / 10  // 10 points per cluster
            lists[listIdx].append(vec)
        }

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        // Multiple queries
        let queries = IVFTestHelpers.randomVectors(count: 5, dimension: dimension, seed: 999)

        let result = try await pipeline.search(
            queries: queries,
            structure: structure,
            k: k
        )

        XCTAssertEqual(result.numQueries, 5)

        // Each query should have k results
        let allResults = result.allResults()
        for queryResults in allResults {
            XCTAssertLessThanOrEqual(queryResults.count, k)
        }
    }

    func testSearchWithProfiling() async throws {
        let dimension = 4
        let numCentroids = 4

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 2,
            dimension: dimension,
            enableProfiling: true
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let lists: [[[Float]]] = (0..<numCentroids).map { _ in
            IVFTestHelpers.randomVectors(count: 5, dimension: dimension)
        }

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        let queries = IVFTestHelpers.randomVectors(count: 2, dimension: dimension)

        let result = try await pipeline.search(
            queries: queries,
            structure: structure,
            k: 3
        )

        // Profiling should be available
        XCTAssertNotNil(result.phaseTimings)

        if let timings = result.phaseTimings {
            XCTAssertGreaterThanOrEqual(timings.coarseQuantization, 0)
            XCTAssertGreaterThanOrEqual(timings.listSearch, 0)
            XCTAssertGreaterThanOrEqual(timings.totalGPUTime, 0)
        }
    }

    func testEstimatedMemoryBytes() async throws {
        let dimension = 128
        let numCentroids = 256
        let totalVectors = 10000

        let config = IVFSearchConfiguration(
            numCentroids: numCentroids,
            nprobe: 8,
            dimension: dimension
        )

        let pipeline = try await IVFSearchPipeline(context: context, configuration: config)

        // Create structure with many vectors
        let centroids = IVFTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let vectorsPerList = totalVectors / numCentroids
        let lists: [[[Float]]] = (0..<numCentroids).map { _ in
            IVFTestHelpers.randomVectors(count: vectorsPerList, dimension: dimension)
        }

        let structure = try await pipeline.prepareIndexStructure(
            centroids: centroids,
            lists: lists,
            dimension: dimension
        )

        let estimatedBytes = structure.estimatedMemoryBytes

        // Should be approximately:
        // centroids: 256 * 128 * 4 = 131,072 bytes
        // vectors: 10000 * 128 * 4 = 5,120,000 bytes
        // indices: 10000 * 4 = 40,000 bytes
        // offsets: 257 * 4 = 1,028 bytes
        // Total ~5.3 MB

        XCTAssertGreaterThan(estimatedBytes, 5_000_000)
        XCTAssertLessThan(estimatedBytes, 10_000_000)
    }
}

// MARK: - IVF Result Types Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFResultTypesTests: XCTestCase {

    func testIVFPhaseTimings() {
        let timings = IVFPhaseTimings(
            coarseQuantization: 0.001,
            listSearch: 0.005,
            candidateMerge: 0.002,
            bufferOperations: 0.0005
        )

        XCTAssertEqual(timings.coarseQuantization, 0.001)
        XCTAssertEqual(timings.listSearch, 0.005)
        XCTAssertEqual(timings.candidateMerge, 0.002)
        XCTAssertEqual(timings.totalGPUTime, 0.008, accuracy: 0.0001)
    }

    func testIVFListMetadata() {
        let metadata = IVFListMetadata(
            listIndex: 5,
            count: 100,
            vectorOffset: 500,
            idOffset: 500
        )

        XCTAssertEqual(metadata.listIndex, 5)
        XCTAssertEqual(metadata.count, 100)
        XCTAssertEqual(metadata.vectorOffset, 500)
        XCTAssertEqual(metadata.idOffset, 500)
    }
}

// MARK: - IVF Shader Args Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IVFShaderArgsTests: XCTestCase {

    func testIVFCoarseQuantizerShaderArgs() {
        let args = IVFCoarseQuantizerShaderArgs(
            dimension: 128,
            numQueries: 10,
            numCentroids: 256,
            nprobe: 8
        )

        XCTAssertEqual(args.dimension, 128)
        XCTAssertEqual(args.numQueries, 10)
        XCTAssertEqual(args.numCentroids, 256)
        XCTAssertEqual(args.nprobe, 8)
    }

    func testIVFListSearchShaderArgs() {
        let args = IVFListSearchShaderArgs(
            dimension: 128,
            k: 10,
            numQueries: 5,
            numCandidates: 1000
        )

        XCTAssertEqual(args.dimension, 128)
        XCTAssertEqual(args.k, 10)
        XCTAssertEqual(args.numQueries, 5)
        XCTAssertEqual(args.numCandidates, 1000)
    }

    func testIVFMergeShaderArgs() {
        let args = IVFMergeShaderArgs(
            numQueries: 10,
            numSets: 8,
            candidatesPerSet: 20,
            finalK: 10
        )

        XCTAssertEqual(args.numQueries, 10)
        XCTAssertEqual(args.numSets, 8)
        XCTAssertEqual(args.candidatesPerSet, 20)
        XCTAssertEqual(args.finalK, 10)
    }
}
