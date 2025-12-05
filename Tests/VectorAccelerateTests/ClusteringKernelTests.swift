//
//  ClusteringKernelTests.swift
//  VectorAccelerate
//
//  Tests for K-Means clustering GPU kernels.
//
//  Tests cover:
//  - KMeansConfiguration validation and presets
//  - KMeansAssignKernel functionality
//  - KMeansUpdateKernel functionality
//  - KMeansConvergenceKernel functionality
//  - KMeansPipeline end-to-end clustering
//

import XCTest
@testable import VectorAccelerate
@testable import VectorAccelerate

import VectorCore
@preconcurrency import Metal

// MARK: - Test Helpers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ClusteringTestHelpers {
    /// Generate random vectors
    static func randomVectors(count: Int, dimension: Int, seed: UInt64 = 42) -> [[Float]] {
        srand48(Int(seed))
        return (0..<count).map { _ in
            (0..<dimension).map { _ in Float(drand48() * 2 - 1) }
        }
    }

    /// Generate clustered vectors around known centroids
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
                let vec = centroid.map { $0 + Float(drand48() * 2 - 1) * noise }
                vectors.append(vec)
                labels.append(clusterIdx)
            }
        }

        return (vectors, labels)
    }

    /// CPU L2 distance squared
    static func cpuL2DistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) }
    }

    /// Find nearest centroid for a vector
    static func cpuFindNearestCentroid(_ vector: [Float], centroids: [[Float]]) -> Int {
        var minDist = Float.infinity
        var minIdx = 0
        for (i, centroid) in centroids.enumerated() {
            let dist = cpuL2DistanceSquared(vector, centroid)
            if dist < minDist {
                minDist = dist
                minIdx = i
            }
        }
        return minIdx
    }

    /// Compute clustering accuracy (percentage of correct assignments)
    static func clusteringAccuracy(
        assignments: [Int],
        trueLabels: [Int],
        numClusters: Int
    ) -> Float {
        guard !assignments.isEmpty && assignments.count == trueLabels.count else { return 0 }

        // Find best mapping from assignment to true label
        // For each assignment cluster, find the most common true label
        var clusterToLabel: [Int: Int] = [:]
        for k in 0..<numClusters {
            var labelCounts: [Int: Int] = [:]
            for (i, assign) in assignments.enumerated() where assign == k {
                let label = trueLabels[i]
                labelCounts[label, default: 0] += 1
            }
            if let bestLabel = labelCounts.max(by: { $0.value < $1.value })?.key {
                clusterToLabel[k] = bestLabel
            }
        }

        // Count correct assignments
        var correct = 0
        for (i, assign) in assignments.enumerated() {
            if clusterToLabel[assign] == trueLabels[i] {
                correct += 1
            }
        }

        return Float(correct) / Float(assignments.count)
    }
}

// MARK: - KMeansConfiguration Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KMeansConfigurationTests: XCTestCase {

    func testDefaultInitialization() {
        let config = KMeansConfiguration(
            numClusters: 16,
            dimension: 64
        )

        XCTAssertEqual(config.numClusters, 16)
        XCTAssertEqual(config.dimension, 64)
        XCTAssertEqual(config.maxIterations, 100)
        XCTAssertEqual(config.convergenceThreshold, 1e-4)
        XCTAssertEqual(config.metric, .euclidean)
        XCTAssertFalse(config.enableProfiling)
    }

    func testValidationSuccess() throws {
        let config = KMeansConfiguration(
            numClusters: 256,
            dimension: 128
        )
        XCTAssertNoThrow(try config.validate())
    }

    func testValidationFailsForZeroClusters() {
        let config = KMeansConfiguration(numClusters: 0, dimension: 64)
        XCTAssertThrowsError(try config.validate())
    }

    func testValidationFailsForExcessiveClusters() {
        let config = KMeansConfiguration(numClusters: 100000, dimension: 64)
        XCTAssertThrowsError(try config.validate())
    }

    func testValidationFailsForZeroDimension() {
        let config = KMeansConfiguration(numClusters: 16, dimension: 0)
        XCTAssertThrowsError(try config.validate())
    }

    func testValidationFailsForZeroIterations() {
        let config = KMeansConfiguration(
            numClusters: 16,
            dimension: 64,
            maxIterations: 0
        )
        XCTAssertThrowsError(try config.validate())
    }

    func testSmallPreset() {
        let config = KMeansConfiguration.small(dimension: 64)
        XCTAssertEqual(config.numClusters, 256)
        XCTAssertEqual(config.dimension, 64)
        XCTAssertEqual(config.maxIterations, 50)
    }

    func testStandardPreset() {
        let config = KMeansConfiguration.standard(dimension: 128)
        XCTAssertEqual(config.numClusters, 1024)
        XCTAssertEqual(config.dimension, 128)
    }

    func testFastPreset() {
        let config = KMeansConfiguration.fast(numClusters: 64, dimension: 32)
        XCTAssertEqual(config.maxIterations, 20)
        XCTAssertEqual(config.convergenceThreshold, 1e-3)
    }

    func testHighQualityPreset() {
        let config = KMeansConfiguration.highQuality(numClusters: 64, dimension: 32)
        XCTAssertEqual(config.maxIterations, 200)
        XCTAssertEqual(config.convergenceThreshold, 1e-5)
    }
}

// MARK: - KMeansAssignKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KMeansAssignKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: KMeansAssignKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await KMeansAssignKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testAssignSimple() async throws {
        // Simple centroids
        let centroids: [[Float]] = [
            [0, 0, 0, 0],
            [10, 0, 0, 0],
            [0, 10, 0, 0]
        ]

        // Vectors close to each centroid
        let vectors: [[Float]] = [
            [0.1, 0.1, 0, 0],   // Near centroid 0
            [9.9, 0.1, 0, 0],  // Near centroid 1
            [0.1, 9.9, 0, 0],  // Near centroid 2
            [0, 0.1, 0, 0]     // Near centroid 0
        ]

        let (assignments, distances) = try await kernel.assign(
            vectors: vectors,
            centroids: centroids
        )

        XCTAssertEqual(assignments, [0, 1, 2, 0])
        XCTAssertEqual(distances.count, 4)

        // Distances should be small (points are close to centroids)
        for dist in distances {
            XCTAssertLessThan(dist, 1.0)
        }
    }

    func testAssignMatchesCPU() async throws {
        let dimension = 8
        let numCentroids = 4
        let numVectors = 20

        let centroids = ClusteringTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let vectors = ClusteringTestHelpers.randomVectors(count: numVectors, dimension: dimension, seed: 123)

        let (gpuAssignments, _) = try await kernel.assign(
            vectors: vectors,
            centroids: centroids
        )

        // CPU reference
        let cpuAssignments = vectors.map { vec in
            ClusteringTestHelpers.cpuFindNearestCentroid(vec, centroids: centroids)
        }

        XCTAssertEqual(gpuAssignments, cpuAssignments)
    }

    func testComputeClusterCounts() async throws {
        let dimension = 4
        let numCentroids = 3
        let numVectors = 100

        let centroids = ClusteringTestHelpers.randomVectors(count: numCentroids, dimension: dimension)
        let vectors = ClusteringTestHelpers.randomVectors(count: numVectors, dimension: dimension, seed: 456)

        let device = context.device.rawDevice

        // Create buffers
        let flatVectors = vectors.flatMap { $0 }
        let vectorBuffer = device.makeBuffer(bytes: flatVectors, length: flatVectors.count * MemoryLayout<Float>.size, options: .storageModeShared)!

        let flatCentroids = centroids.flatMap { $0 }
        let centroidBuffer = device.makeBuffer(bytes: flatCentroids, length: flatCentroids.count * MemoryLayout<Float>.size, options: .storageModeShared)!

        // Assign and count
        let assignResult = try await kernel.assign(
            vectors: vectorBuffer,
            centroids: centroidBuffer,
            numVectors: numVectors,
            numCentroids: numCentroids,
            dimension: dimension
        )

        let countsBuffer = try await kernel.computeClusterCounts(
            assignments: assignResult.assignments,
            numVectors: numVectors,
            numCentroids: numCentroids
        )

        // Verify counts sum to numVectors
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        let totalCount = (0..<numCentroids).reduce(0) { $0 + Int(countsPtr[$1]) }
        XCTAssertEqual(totalCount, numVectors)
    }
}

// MARK: - KMeansUpdateKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KMeansUpdateKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: KMeansUpdateKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await KMeansUpdateKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testUpdateSimple() async throws {
        // Two clusters with known assignments
        let vectors: [[Float]] = [
            [0, 0],  // Cluster 0
            [2, 0],  // Cluster 0
            [10, 0], // Cluster 1
            [12, 0]  // Cluster 1
        ]
        let assignments = [0, 0, 1, 1]
        let currentCentroids: [[Float]] = [
            [1, 0],   // Initial centroid 0
            [11, 0]   // Initial centroid 1
        ]

        let (newCentroids, counts, emptyClusters) = try await kernel.update(
            vectors: vectors,
            assignments: assignments,
            currentCentroids: currentCentroids
        )

        XCTAssertEqual(emptyClusters, 0)
        XCTAssertEqual(counts, [2, 2])

        // New centroids should be mean of assigned vectors
        // Cluster 0: mean([0,0], [2,0]) = [1, 0]
        // Cluster 1: mean([10,0], [12,0]) = [11, 0]
        XCTAssertEqual(newCentroids[0], [1, 0])
        XCTAssertEqual(newCentroids[1], [11, 0])
    }

    func testUpdateWithEmptyCluster() async throws {
        // All vectors assigned to cluster 0, cluster 1 is empty
        let vectors: [[Float]] = [
            [0, 0],
            [1, 0],
            [2, 0]
        ]
        let assignments = [0, 0, 0]  // All in cluster 0
        let currentCentroids: [[Float]] = [
            [1, 0],
            [10, 0]  // This cluster will be empty
        ]

        let (newCentroids, counts, emptyClusters) = try await kernel.update(
            vectors: vectors,
            assignments: assignments,
            currentCentroids: currentCentroids
        )

        XCTAssertEqual(emptyClusters, 1)
        XCTAssertEqual(counts, [3, 0])

        // Empty cluster should keep its current centroid
        XCTAssertEqual(newCentroids[1], currentCentroids[1])
    }
}

// MARK: - KMeansConvergenceKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KMeansConvergenceKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: KMeansConvergenceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await KMeansConvergenceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testConvergenceWhenIdentical() async throws {
        let centroids: [[Float]] = [
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]

        let result = try await kernel.checkConvergence(
            oldCentroids: centroids,
            newCentroids: centroids,
            threshold: 1e-4
        )

        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.maxMovement, 0)
        XCTAssertEqual(result.meanMovement, 0)
        XCTAssertEqual(result.centroidsMoved, 0)
    }

    func testNoConvergenceWhenMoved() async throws {
        let oldCentroids: [[Float]] = [
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]
        let newCentroids: [[Float]] = [
            [1, 0, 0, 0],  // Moved by 1
            [1, 1, 1, 1]   // Not moved
        ]

        let result = try await kernel.checkConvergence(
            oldCentroids: oldCentroids,
            newCentroids: newCentroids,
            threshold: 0.5
        )

        XCTAssertFalse(result.converged)
        XCTAssertEqual(result.maxMovement, 1.0, accuracy: 1e-6)
        XCTAssertEqual(result.centroidsMoved, 1)
    }

    func testConvergenceWithSmallMovement() async throws {
        let oldCentroids: [[Float]] = [
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]
        let newCentroids: [[Float]] = [
            [0.0001, 0, 0, 0],  // Moved by 0.0001
            [1.0001, 1, 1, 1]   // Moved by 0.0001
        ]

        let result = try await kernel.checkConvergence(
            oldCentroids: oldCentroids,
            newCentroids: newCentroids,
            threshold: 0.001
        )

        XCTAssertTrue(result.converged)
        XCTAssertLessThan(result.maxMovement, 0.001)
    }
}

// MARK: - KMeansPipeline Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KMeansPipelineTests: XCTestCase {

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

    func testFitSimpleClusters() async throws {
        let dimension = 4
        let numClusters = 3
        let pointsPerCluster = 50

        // Create well-separated clusters
        let trueCentroids: [[Float]] = [
            [0, 0, 0, 0],
            [10, 0, 0, 0],
            [0, 10, 0, 0]
        ]

        let (vectors, trueLabels) = ClusteringTestHelpers.clusteredVectors(
            centroids: trueCentroids,
            pointsPerCluster: pointsPerCluster,
            noise: 0.5
        )

        let config = KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 20,
            convergenceThreshold: 1e-4
        )

        let pipeline = try await KMeansPipeline(context: context, configuration: config)

        // Fit with known initial centroids (for reproducibility)
        let result = try await pipeline.fit(
            vectors: vectors,
            initialCentroids: trueCentroids
        )

        XCTAssertEqual(result.numClusters, numClusters)
        XCTAssertEqual(result.numVectors, vectors.count)
        XCTAssertEqual(result.dimension, dimension)
        XCTAssertGreaterThan(result.iterations, 0)
        XCTAssertLessThanOrEqual(result.iterations, 20)

        // Check clustering accuracy
        let assignments = result.extractAssignments()
        let accuracy = ClusteringTestHelpers.clusteringAccuracy(
            assignments: assignments,
            trueLabels: trueLabels,
            numClusters: numClusters
        )

        // Should have high accuracy for well-separated clusters
        XCTAssertGreaterThan(accuracy, 0.9)
    }

    func testFitWithProfiling() async throws {
        let dimension = 4
        let numClusters = 2

        let config = KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 5,
            enableProfiling: true
        )

        let pipeline = try await KMeansPipeline(context: context, configuration: config)

        let vectors = ClusteringTestHelpers.randomVectors(count: 100, dimension: dimension)

        let result = try await pipeline.fit(vectors: vectors)

        // Profiling should be available
        XCTAssertNotNil(result.iterationTimings)

        if let timings = result.iterationTimings {
            XCTAssertEqual(timings.count, result.iterations)

            for timing in timings {
                XCTAssertGreaterThanOrEqual(timing.assignmentTime, 0)
                XCTAssertGreaterThanOrEqual(timing.updateTime, 0)
                XCTAssertGreaterThanOrEqual(timing.convergenceTime, 0)
            }
        }
    }

    func testResultExtraction() async throws {
        let dimension = 4
        let numClusters = 3
        let numVectors = 60

        let config = KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 10
        )

        let pipeline = try await KMeansPipeline(context: context, configuration: config)

        let vectors = ClusteringTestHelpers.randomVectors(count: numVectors, dimension: dimension)

        let result = try await pipeline.fit(vectors: vectors)

        // Extract and verify centroids
        let centroids = result.extractCentroids()
        XCTAssertEqual(centroids.count, numClusters)
        for centroid in centroids {
            XCTAssertEqual(centroid.count, dimension)
        }

        // Extract and verify assignments
        let assignments = result.extractAssignments()
        XCTAssertEqual(assignments.count, numVectors)
        for assign in assignments {
            XCTAssertGreaterThanOrEqual(assign, 0)
            XCTAssertLessThan(assign, numClusters)
        }

        // Extract and verify counts
        let counts = result.extractClusterCounts()
        XCTAssertEqual(counts.count, numClusters)
        XCTAssertEqual(counts.reduce(0, +), numVectors)
    }
}

// MARK: - Shader Args Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ClusteringShaderArgsTests: XCTestCase {

    func testKMeansAssignShaderArgs() {
        let args = KMeansAssignShaderArgs(
            dimension: 128,
            numVectors: 10000,
            numCentroids: 256
        )

        XCTAssertEqual(args.dimension, 128)
        XCTAssertEqual(args.numVectors, 10000)
        XCTAssertEqual(args.numCentroids, 256)
    }

    func testKMeansUpdateShaderArgs() {
        let args = KMeansUpdateShaderArgs(
            dimension: 128,
            numCentroids: 256,
            numVectors: 10000
        )

        XCTAssertEqual(args.dimension, 128)
        XCTAssertEqual(args.numCentroids, 256)
        XCTAssertEqual(args.numVectors, 10000)
    }

    func testKMeansConvergenceShaderArgs() {
        let args = KMeansConvergenceShaderArgs(
            numCentroids: 256,
            dimension: 128,
            threshold: 0.001
        )

        XCTAssertEqual(args.numCentroids, 256)
        XCTAssertEqual(args.dimension, 128)
        XCTAssertEqual(args.thresholdSquared, 0.000001, accuracy: 1e-10)
    }
}
