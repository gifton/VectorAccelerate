//
//  ClusteringKernels.swift
//  VectorAccelerate
//
//  Metal 4 kernels for clustering operations (primarily K-Means).
//
//  ## Architecture
//
//  K-Means is implemented as an iterative pipeline:
//  1. Assignment: Assign each vector to nearest centroid (GPU-parallel)
//  2. Update: Compute new centroids as mean of assigned vectors (GPU reduction)
//  3. Convergence: Check if centroids moved less than threshold
//  4. Repeat until convergence or max iterations
//
//  ## Integration with VectorIndex
//
//  These GPU kernels complement the CPU-based implementations in VectorIndex:
//  - Kernel #11 (KMeansSeeding): K-means++ initialization
//  - Kernel #12 (KMeansMiniBatch): Mini-batch training
//
//  GPU acceleration provides speedup for large-scale clustering.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - K-Means Configuration

/// Configuration for K-Means GPU clustering.
public struct KMeansConfiguration: Sendable {
    /// Number of clusters (K)
    public let numClusters: Int

    /// Vector dimension
    public let dimension: Int

    /// Maximum iterations
    public let maxIterations: Int

    /// Convergence threshold (centroid movement)
    public let convergenceThreshold: Float

    /// Distance metric
    public let metric: SupportedDistanceMetric

    /// Enable profiling for performance analysis
    public let enableProfiling: Bool

    /// Batch size for mini-batch updates (0 = full batch)
    public let batchSize: Int

    public init(
        numClusters: Int,
        dimension: Int,
        maxIterations: Int = 100,
        convergenceThreshold: Float = 1e-4,
        metric: SupportedDistanceMetric = .euclidean,
        enableProfiling: Bool = false,
        batchSize: Int = 0
    ) {
        self.numClusters = numClusters
        self.dimension = dimension
        self.maxIterations = maxIterations
        self.convergenceThreshold = convergenceThreshold
        self.metric = metric
        self.enableProfiling = enableProfiling
        self.batchSize = batchSize
    }

    /// Validate configuration parameters.
    public func validate() throws {
        guard numClusters >= 1 else {
            throw IndexError.invalidInput(message: "numClusters must be at least 1, got \(numClusters)")
        }
        guard numClusters <= 65536 else {
            throw IndexError.invalidInput(message: "numClusters must be at most 65536, got \(numClusters)")
        }
        guard dimension >= 1 && dimension <= 4096 else {
            throw IndexError.invalidInput(message: "dimension must be 1-4096, got \(dimension)")
        }
        guard maxIterations >= 1 else {
            throw IndexError.invalidInput(message: "maxIterations must be at least 1, got \(maxIterations)")
        }
        guard convergenceThreshold > 0 else {
            throw IndexError.invalidInput(message: "convergenceThreshold must be positive, got \(convergenceThreshold)")
        }
    }

    // MARK: - Presets

    /// Small clustering preset (256 clusters)
    public static func small(dimension: Int) -> KMeansConfiguration {
        KMeansConfiguration(numClusters: 256, dimension: dimension, maxIterations: 50)
    }

    /// Standard clustering preset (1024 clusters)
    public static func standard(dimension: Int) -> KMeansConfiguration {
        KMeansConfiguration(numClusters: 1024, dimension: dimension, maxIterations: 100)
    }

    /// Large clustering preset (4096 clusters)
    public static func large(dimension: Int) -> KMeansConfiguration {
        KMeansConfiguration(numClusters: 4096, dimension: dimension, maxIterations: 100)
    }

    /// Fast convergence preset (lower threshold)
    public static func fast(numClusters: Int, dimension: Int) -> KMeansConfiguration {
        KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 20,
            convergenceThreshold: 1e-3
        )
    }

    /// High quality preset (more iterations, tighter threshold)
    public static func highQuality(numClusters: Int, dimension: Int) -> KMeansConfiguration {
        KMeansConfiguration(
            numClusters: numClusters,
            dimension: dimension,
            maxIterations: 200,
            convergenceThreshold: 1e-5
        )
    }
}

// MARK: - K-Means Result

/// Result from K-Means clustering operation.
public struct KMeansResult: Sendable {
    /// Final centroids [numClusters × dimension]
    public let centroids: any MTLBuffer

    /// Cluster assignments for each vector [numVectors]
    public let assignments: any MTLBuffer

    /// Number of vectors assigned to each cluster [numClusters]
    public let clusterCounts: any MTLBuffer

    /// Number of clusters
    public let numClusters: Int

    /// Number of vectors
    public let numVectors: Int

    /// Vector dimension
    public let dimension: Int

    /// Number of iterations performed
    public let iterations: Int

    /// Whether convergence was achieved
    public let converged: Bool

    /// Final inertia (sum of squared distances to centroids)
    public let inertia: Float

    /// Total execution time
    public let executionTime: TimeInterval

    /// Per-iteration timings (if profiling enabled)
    public let iterationTimings: [KMeansIterationTiming]?

    /// Extract centroids as 2D Float array.
    public func extractCentroids() -> [[Float]] {
        let ptr = centroids.contents().bindMemory(to: Float.self, capacity: numClusters * dimension)
        var result: [[Float]] = []
        result.reserveCapacity(numClusters)

        for k in 0..<numClusters {
            var centroid: [Float] = []
            centroid.reserveCapacity(dimension)
            for d in 0..<dimension {
                centroid.append(ptr[k * dimension + d])
            }
            result.append(centroid)
        }
        return result
    }

    /// Extract assignments as array.
    public func extractAssignments() -> [Int] {
        let ptr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        return (0..<numVectors).map { Int(ptr[$0]) }
    }

    /// Extract cluster counts as array.
    public func extractClusterCounts() -> [Int] {
        let ptr = clusterCounts.contents().bindMemory(to: UInt32.self, capacity: numClusters)
        return (0..<numClusters).map { Int(ptr[$0]) }
    }
}

/// Timing for a single K-Means iteration.
public struct KMeansIterationTiming: Sendable {
    /// Iteration number
    public let iteration: Int

    /// Time for assignment phase
    public let assignmentTime: TimeInterval

    /// Time for update phase
    public let updateTime: TimeInterval

    /// Time for convergence check
    public let convergenceTime: TimeInterval

    /// Centroid movement in this iteration
    public let centroidMovement: Float

    /// Total iteration time
    public var totalTime: TimeInterval {
        assignmentTime + updateTime + convergenceTime
    }
}

// MARK: - Assignment Result

/// Result from K-Means assignment phase.
public struct KMeansAssignmentResult: Sendable {
    /// Cluster assignments for each vector [numVectors]
    public let assignments: any MTLBuffer

    /// Distances to assigned centroids [numVectors]
    public let distances: any MTLBuffer

    /// Number of vectors
    public let numVectors: Int

    /// Execution time
    public let executionTime: TimeInterval

    /// Total inertia (sum of squared distances)
    public var inertia: Float {
        let ptr = distances.contents().bindMemory(to: Float.self, capacity: numVectors)
        return (0..<numVectors).reduce(Float(0)) { $0 + ptr[$1] }
    }
}

// MARK: - Convergence Result

/// Result from K-Means convergence check.
public struct KMeansConvergenceResult: Sendable {
    /// Whether convergence was achieved
    public let converged: Bool

    /// Maximum centroid movement
    public let maxMovement: Float

    /// Mean centroid movement
    public let meanMovement: Float

    /// Number of centroids that moved
    public let centroidsMoved: Int

    /// Execution time
    public let executionTime: TimeInterval
}

// MARK: - Shader Arguments

/// Shader arguments for KMeans point assignment.
/// Memory layout matches `KMeansAssignArgs` in ClusteringShaders.metal.
public struct KMeansAssignShaderArgs: Sendable {
    /// Vector dimension
    public let dimension: UInt32

    /// Number of vectors to assign
    public let numVectors: UInt32

    /// Number of centroids (K)
    public let numCentroids: UInt32

    /// Padding for alignment
    private let _padding: UInt32 = 0

    public init(dimension: Int, numVectors: Int, numCentroids: Int) {
        self.dimension = UInt32(dimension)
        self.numVectors = UInt32(numVectors)
        self.numCentroids = UInt32(numCentroids)
    }
}

/// Shader arguments for KMeans centroid update.
/// Memory layout matches `KMeansUpdateArgs` in ClusteringShaders.metal.
public struct KMeansUpdateShaderArgs: Sendable {
    /// Vector dimension
    public let dimension: UInt32

    /// Number of centroids (K)
    public let numCentroids: UInt32

    /// Number of vectors
    public let numVectors: UInt32

    /// Padding for alignment
    private let _padding: UInt32 = 0

    public init(dimension: Int, numCentroids: Int, numVectors: Int) {
        self.dimension = UInt32(dimension)
        self.numCentroids = UInt32(numCentroids)
        self.numVectors = UInt32(numVectors)
    }
}

/// Shader arguments for KMeans convergence check.
/// Memory layout matches `KMeansConvergenceArgs` in ClusteringShaders.metal.
public struct KMeansConvergenceShaderArgs: Sendable {
    /// Number of centroids (K)
    public let numCentroids: UInt32

    /// Vector dimension
    public let dimension: UInt32

    /// Convergence threshold (squared)
    public let thresholdSquared: Float

    /// Padding for alignment
    private let _padding: UInt32 = 0

    public init(numCentroids: Int, dimension: Int, threshold: Float) {
        self.numCentroids = UInt32(numCentroids)
        self.dimension = UInt32(dimension)
        self.thresholdSquared = threshold * threshold
    }
}
