//
//  TestDataGenerator.swift
//  VectorAccelerateTests
//
//  Comprehensive test data generation utilities for vector index testing.
//
//  Provides multiple data distributions to stress-test different aspects
//  of vector search algorithms:
//
//  - Uniform random: Worst case for clustering algorithms
//  - Gaussian clusters: Natural for K-means, tests recall at boundaries
//  - Unit vectors: Realistic for normalized embeddings
//  - Sparse vectors: Typical for certain embedding models
//  - Skewed distributions: Tests robustness to imbalanced clusters
//
//  Usage:
//    var gen = TestDataGenerator(seed: 42)
//    let uniform = gen.uniformVectors(count: 1000, dimension: 128)
//    let clusters = gen.gaussianClusters(numClusters: 20, pointsPerCluster: 50, dimension: 128)
//    let queries = gen.perturbedQueries(from: uniform, count: 100, noiseStdDev: 0.1)
//

import Foundation

/// Comprehensive test data generator for vector index testing.
///
/// Generates various data distributions with deterministic seeding
/// for reproducible test results.
public struct TestDataGenerator {
    /// Internal RNG
    var rng: TestRNG

    /// Track data generation for diagnostics
    public private(set) var generatedCount: Int = 0

    // MARK: - Initialization

    /// Initialize generator with seed.
    ///
    /// - Parameters:
    ///   - seed: Random seed for reproducibility
    ///   - stream: Stream ID for parallel generation (default: 0)
    public init(seed: UInt64, stream: UInt64 = 0) {
        self.rng = TestRNG(seed: seed, stream: stream)
    }

    // MARK: - Uniform Random Vectors

    /// Generate uniformly random vectors in hypercube [-1, 1]^D.
    ///
    /// This is the **hardest** distribution for IVF:
    /// - No natural cluster structure
    /// - K-means produces roughly equal-sized clusters
    /// - Recall depends entirely on nprobe coverage
    ///
    /// Expected IVF recall: ~(nprobe / nlist) for large datasets.
    ///
    /// - Parameters:
    ///   - count: Number of vectors to generate
    ///   - dimension: Vector dimension
    /// - Returns: Array of uniformly random vectors
    public mutating func uniformVectors(count: Int, dimension: Int) -> [[Float]] {
        generatedCount += count
        return (0..<count).map { _ in
            (0..<dimension).map { _ in rng.nextFloat(in: -1...1) }
        }
    }

    /// Generate uniformly random vectors in specified range.
    ///
    /// - Parameters:
    ///   - count: Number of vectors
    ///   - dimension: Vector dimension
    ///   - range: Range for each component
    /// - Returns: Array of random vectors
    public mutating func uniformVectors(
        count: Int,
        dimension: Int,
        range: ClosedRange<Float>
    ) -> [[Float]] {
        generatedCount += count
        return (0..<count).map { _ in
            (0..<dimension).map { _ in rng.nextFloat(in: range) }
        }
    }

    // MARK: - Gaussian Cluster Vectors

    /// Generate vectors from Gaussian clusters with controlled separation.
    ///
    /// Cluster centers are placed on a grid or randomly with minimum separation.
    /// Points are drawn from N(center, stdDev^2) independently per dimension.
    ///
    /// This tests IVF's ability to:
    /// - Learn natural cluster structure
    /// - Route queries to correct clusters
    /// - Handle boundary cases between clusters
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters
    ///   - pointsPerCluster: Points per cluster
    ///   - dimension: Vector dimension
    ///   - clusterSpread: Range for cluster center placement (default: 3.0)
    ///   - clusterStdDev: Standard deviation within clusters (default: 0.5)
    /// - Returns: Array of clustered vectors
    public mutating func gaussianClusters(
        numClusters: Int,
        pointsPerCluster: Int,
        dimension: Int,
        clusterSpread: Float = 3.0,
        clusterStdDev: Float = 0.5
    ) -> [[Float]] {
        // Generate cluster centers uniformly in [-spread, spread]^D
        var centers: [[Float]] = []
        centers.reserveCapacity(numClusters)
        for _ in 0..<numClusters {
            let center = (0..<dimension).map { _ in
                rng.nextFloat(in: -clusterSpread...clusterSpread)
            }
            centers.append(center)
        }

        // Generate points around each center with Gaussian noise
        var data: [[Float]] = []
        data.reserveCapacity(numClusters * pointsPerCluster)

        for center in centers {
            for _ in 0..<pointsPerCluster {
                let point = (0..<dimension).map { d in
                    center[d] + rng.nextGaussian(stdDev: clusterStdDev)
                }
                data.append(point)
            }
        }

        generatedCount += data.count
        return data
    }

    /// Generate Gaussian clusters with well-separated centers.
    ///
    /// Uses a grid layout to ensure minimum separation between clusters.
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters
    ///   - pointsPerCluster: Points per cluster
    ///   - dimension: Vector dimension
    ///   - minSeparation: Minimum distance between cluster centers
    ///   - clusterStdDev: Standard deviation within clusters
    /// - Returns: Tuple of (data, centers) for verification
    public mutating func separatedGaussianClusters(
        numClusters: Int,
        pointsPerCluster: Int,
        dimension: Int,
        minSeparation: Float = 4.0,
        clusterStdDev: Float = 0.5
    ) -> (data: [[Float]], centers: [[Float]]) {
        // Place centers on a high-dimensional grid
        // Grid size in each dimension
        let gridSize = Int(ceil(pow(Float(numClusters), 1.0 / Float(min(dimension, 3)))))
        let spacing = minSeparation

        var centers: [[Float]] = []
        centers.reserveCapacity(numClusters)

        // Generate centers on grid (using first 3 dims for structure)
        for i in 0..<numClusters {
            var center = [Float](repeating: 0, count: dimension)
            var idx = i
            for d in 0..<min(dimension, 3) {
                center[d] = Float(idx % gridSize) * spacing
                idx /= gridSize
            }
            // Random offset in remaining dimensions
            for d in 3..<dimension {
                center[d] = rng.nextFloat(in: -spacing...spacing)
            }
            centers.append(center)
        }

        // Generate points
        var data: [[Float]] = []
        data.reserveCapacity(numClusters * pointsPerCluster)

        for center in centers {
            for _ in 0..<pointsPerCluster {
                let point = (0..<dimension).map { d in
                    center[d] + rng.nextGaussian(stdDev: clusterStdDev)
                }
                data.append(point)
            }
        }

        generatedCount += data.count
        return (data, centers)
    }

    // MARK: - Unit Vectors (Normalized)

    /// Generate random unit vectors (L2 normalized).
    ///
    /// Realistic for embedding models that output normalized vectors.
    /// All vectors lie on the unit hypersphere.
    ///
    /// - Parameters:
    ///   - count: Number of vectors
    ///   - dimension: Vector dimension
    /// - Returns: Array of unit vectors
    public mutating func unitVectors(count: Int, dimension: Int) -> [[Float]] {
        generatedCount += count
        return (0..<count).map { _ in
            // Generate Gaussian random vector, then normalize
            // This produces uniform distribution on hypersphere
            var vec = (0..<dimension).map { _ in rng.nextGaussian() }
            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
            if norm > Float.ulpOfOne {
                vec = vec.map { $0 / norm }
            }
            return vec
        }
    }

    /// Generate clustered unit vectors.
    ///
    /// Creates clusters on the unit hypersphere with controlled angular spread.
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters
    ///   - pointsPerCluster: Points per cluster
    ///   - dimension: Vector dimension
    ///   - angularSpread: Angular spread within cluster (radians, default: 0.3)
    /// - Returns: Array of clustered unit vectors
    public mutating func clusteredUnitVectors(
        numClusters: Int,
        pointsPerCluster: Int,
        dimension: Int,
        angularSpread: Float = 0.3
    ) -> [[Float]] {
        // Generate random unit vector centers
        var centers: [[Float]] = []
        for _ in 0..<numClusters {
            var center = (0..<dimension).map { _ in rng.nextGaussian() }
            let norm = sqrt(center.reduce(0) { $0 + $1 * $1 })
            center = center.map { $0 / norm }
            centers.append(center)
        }

        var data: [[Float]] = []
        data.reserveCapacity(numClusters * pointsPerCluster)

        for center in centers {
            for _ in 0..<pointsPerCluster {
                // Add Gaussian noise and renormalize
                var vec = center.map { $0 + rng.nextGaussian(stdDev: angularSpread) }
                let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
                vec = vec.map { $0 / norm }
                data.append(vec)
            }
        }

        generatedCount += data.count
        return data
    }

    // MARK: - Sparse Vectors

    /// Generate sparse vectors with controlled sparsity.
    ///
    /// Typical for certain embedding models or document vectors.
    ///
    /// - Parameters:
    ///   - count: Number of vectors
    ///   - dimension: Vector dimension
    ///   - sparsity: Fraction of zeros (default: 0.9 = 90% zeros)
    /// - Returns: Array of sparse vectors
    public mutating func sparseVectors(
        count: Int,
        dimension: Int,
        sparsity: Float = 0.9
    ) -> [[Float]] {
        generatedCount += count
        return (0..<count).map { _ in
            (0..<dimension).map { _ in
                if rng.nextFloat01() < sparsity {
                    return 0.0
                } else {
                    return rng.nextGaussian()
                }
            }
        }
    }

    // MARK: - Skewed Distributions

    /// Generate clusters with skewed sizes (power-law distribution).
    ///
    /// Tests IVF robustness to imbalanced cluster sizes.
    /// Some clusters will be much larger than others.
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters
    ///   - totalPoints: Total number of points
    ///   - dimension: Vector dimension
    ///   - skewExponent: Power-law exponent (default: 1.5, higher = more skew)
    ///   - clusterStdDev: Standard deviation within clusters
    /// - Returns: Array of vectors with skewed cluster sizes
    public mutating func skewedClusters(
        numClusters: Int,
        totalPoints: Int,
        dimension: Int,
        skewExponent: Float = 1.5,
        clusterStdDev: Float = 0.5
    ) -> [[Float]] {
        // Generate cluster sizes with power-law distribution
        var weights: [Float] = (0..<numClusters).map { i in
            pow(Float(numClusters - i), skewExponent)
        }
        let totalWeight = weights.reduce(0, +)
        weights = weights.map { $0 / totalWeight }

        var sizes = weights.map { Int(Float(totalPoints) * $0) }
        // Ensure we have exactly totalPoints
        let currentTotal = sizes.reduce(0, +)
        sizes[0] += totalPoints - currentTotal

        // Generate cluster centers
        var centers: [[Float]] = []
        for _ in 0..<numClusters {
            let center = (0..<dimension).map { _ in rng.nextFloat(in: -3...3) }
            centers.append(center)
        }

        // Generate points
        var data: [[Float]] = []
        data.reserveCapacity(totalPoints)

        for (center, size) in zip(centers, sizes) {
            for _ in 0..<size {
                let point = (0..<dimension).map { d in
                    center[d] + rng.nextGaussian(stdDev: clusterStdDev)
                }
                data.append(point)
            }
        }

        generatedCount += data.count
        return data
    }

    // MARK: - Query Generation

    /// Generate queries as perturbed versions of dataset points.
    ///
    /// This simulates realistic queries that are similar to indexed vectors.
    /// Controls the "hardness" of queries via noise level.
    ///
    /// - Parameters:
    ///   - dataset: Source dataset to sample from
    ///   - count: Number of queries to generate
    ///   - noiseStdDev: Standard deviation of Gaussian noise (default: 0.1)
    /// - Returns: Array of query vectors
    public mutating func perturbedQueries(
        from dataset: [[Float]],
        count: Int,
        noiseStdDev: Float = 0.1
    ) -> [[Float]] {
        precondition(!dataset.isEmpty, "Dataset cannot be empty")
        let dimension = dataset[0].count

        return (0..<count).map { _ in
            let baseIdx = rng.nextInt(bound: dataset.count)
            var query = dataset[baseIdx]
            for d in 0..<dimension {
                query[d] += rng.nextGaussian(stdDev: noiseStdDev)
            }
            return query
        }
    }

    /// Generate queries from specific cluster regions.
    ///
    /// Useful for testing recall within specific clusters.
    ///
    /// - Parameters:
    ///   - centers: Cluster centers
    ///   - count: Number of queries per center
    ///   - noiseStdDev: Standard deviation of noise
    /// - Returns: Array of query vectors
    public mutating func clusterQueries(
        centers: [[Float]],
        countPerCenter: Int,
        noiseStdDev: Float = 0.1
    ) -> [[Float]] {
        var queries: [[Float]] = []
        queries.reserveCapacity(centers.count * countPerCenter)

        for center in centers {
            for _ in 0..<countPerCenter {
                let query = center.enumerated().map { (d, val) in
                    val + rng.nextGaussian(stdDev: noiseStdDev)
                }
                queries.append(query)
            }
        }

        return queries
    }

    /// Generate random queries independent of dataset.
    ///
    /// Harder than perturbed queries - nearest neighbors may be far.
    ///
    /// - Parameters:
    ///   - count: Number of queries
    ///   - dimension: Vector dimension
    ///   - range: Range for query components
    /// - Returns: Array of query vectors
    public mutating func randomQueries(
        count: Int,
        dimension: Int,
        range: ClosedRange<Float> = -1...1
    ) -> [[Float]] {
        uniformVectors(count: count, dimension: dimension, range: range)
    }

    // MARK: - Utility Methods

    /// Get a copy of the internal RNG for external use.
    public var randomGenerator: TestRNG {
        rng
    }

    /// Shuffle the dataset in place.
    public mutating func shuffle(_ data: inout [[Float]]) {
        rng.shuffle(&data)
    }

    /// Return shuffled copy of dataset.
    public mutating func shuffled(_ data: [[Float]]) -> [[Float]] {
        rng.shuffled(data)
    }

    /// Sample k vectors from dataset without replacement.
    public mutating func sample(_ data: [[Float]], k: Int) -> [[Float]] {
        rng.sample(data, k: k)
    }

    /// Split dataset into train/test sets.
    public mutating func trainTestSplit(
        _ data: [[Float]],
        testRatio: Float = 0.2
    ) -> (train: [[Float]], test: [[Float]]) {
        let shuffled = self.shuffled(data)
        let testCount = Int(Float(data.count) * testRatio)
        let test = Array(shuffled.prefix(testCount))
        let train = Array(shuffled.dropFirst(testCount))
        return (train, test)
    }
}

// MARK: - Data Distribution Statistics

extension TestDataGenerator {
    /// Compute basic statistics for a dataset.
    public static func statistics(for data: [[Float]]) -> DatasetStatistics {
        guard !data.isEmpty else {
            return DatasetStatistics(
                count: 0, dimension: 0,
                meanNorm: 0, stdDevNorm: 0,
                minNorm: 0, maxNorm: 0,
                meanMean: 0, meanStdDev: 0
            )
        }

        let dimension = data[0].count
        var norms: [Float] = []
        var means: [Float] = []
        var stdDevs: [Float] = []

        for vec in data {
            // L2 norm
            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
            norms.append(norm)

            // Per-vector statistics
            let mean = vec.reduce(0, +) / Float(dimension)
            let variance = vec.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(dimension)
            means.append(mean)
            stdDevs.append(sqrt(variance))
        }

        let meanNorm = norms.reduce(0, +) / Float(norms.count)
        let normVariance = norms.reduce(0) { $0 + ($1 - meanNorm) * ($1 - meanNorm) } / Float(norms.count)

        return DatasetStatistics(
            count: data.count,
            dimension: dimension,
            meanNorm: meanNorm,
            stdDevNorm: sqrt(normVariance),
            minNorm: norms.min() ?? 0,
            maxNorm: norms.max() ?? 0,
            meanMean: means.reduce(0, +) / Float(means.count),
            meanStdDev: stdDevs.reduce(0, +) / Float(stdDevs.count)
        )
    }
}

/// Statistics for a vector dataset.
public struct DatasetStatistics {
    public let count: Int
    public let dimension: Int
    public let meanNorm: Float
    public let stdDevNorm: Float
    public let minNorm: Float
    public let maxNorm: Float
    public let meanMean: Float
    public let meanStdDev: Float

    public var description: String {
        """
        Dataset: \(count) vectors, D=\(dimension)
        Norms: mean=\(String(format: "%.3f", meanNorm)), std=\(String(format: "%.3f", stdDevNorm)), range=[\(String(format: "%.3f", minNorm)), \(String(format: "%.3f", maxNorm))]
        Values: mean=\(String(format: "%.3f", meanMean)), std=\(String(format: "%.3f", meanStdDev))
        """
    }
}
