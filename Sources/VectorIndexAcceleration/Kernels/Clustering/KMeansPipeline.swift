//
//  KMeansPipeline.swift
//  VectorIndexAcceleration
//
//  Metal 4 pipeline for complete K-Means clustering operations.
//
//  Orchestrates the iterative K-Means algorithm:
//  1. Initialize centroids (K-means++ or random)
//  2. Assignment: Assign each vector to nearest centroid
//  3. Update: Compute new centroids as mean of assigned vectors
//  4. Convergence: Check if centroids have stabilized
//  5. Repeat until convergence or max iterations
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - K-Means Pipeline

/// Metal 4 pipeline for GPU-accelerated K-Means clustering.
///
/// Implements the Lloyd's algorithm with GPU acceleration for the
/// assignment phase. Supports mini-batch updates for large datasets.
///
/// ## Usage
/// ```swift
/// let pipeline = try await KMeansPipeline(context: context, configuration: config)
///
/// // Train on data
/// let result = try await pipeline.fit(vectors: vectors)
///
/// // Extract centroids
/// let centroids = result.extractCentroids()
/// ```
///
/// ## Features
/// - GPU-accelerated point assignment
/// - K-means++ initialization (via VectorIndex)
/// - Empty cluster handling
/// - Convergence monitoring
/// - Optional per-iteration profiling
public final class KMeansPipeline: @unchecked Sendable {

    // MARK: - Properties

    /// The Metal 4 context
    public let context: Metal4Context

    /// The clustering configuration
    public let configuration: KMeansConfiguration

    // MARK: - Private Properties

    private let assignKernel: KMeansAssignKernel
    private let updateKernel: KMeansUpdateKernel
    private let convergenceKernel: KMeansConvergenceKernel

    // MARK: - Initialization

    /// Create a K-Means pipeline.
    ///
    /// - Parameters:
    ///   - context: The Metal 4 context
    ///   - configuration: Clustering configuration
    public init(
        context: Metal4Context,
        configuration: KMeansConfiguration
    ) async throws {
        try configuration.validate()

        self.context = context
        self.configuration = configuration

        // Initialize component kernels
        self.assignKernel = try await KMeansAssignKernel(context: context)
        self.updateKernel = try await KMeansUpdateKernel(context: context)
        self.convergenceKernel = try await KMeansConvergenceKernel(context: context)
    }

    // MARK: - Warm Up

    /// Warm up all pipeline components.
    public func warmUp() async throws {
        try await assignKernel.warmUp()
        try await updateKernel.warmUp()
        try await convergenceKernel.warmUp()
    }

    // MARK: - Fit

    /// Fit K-Means to the given vectors.
    ///
    /// - Parameters:
    ///   - vectors: Vector data as 2D Float array [numVectors × dimension]
    ///   - initialCentroids: Optional initial centroids (uses K-means++ if nil)
    /// - Returns: K-Means result with centroids and assignments
    public func fit(
        vectors: [[Float]],
        initialCentroids: [[Float]]? = nil
    ) async throws -> KMeansResult {
        guard !vectors.isEmpty else {
            throw IndexAccelerationError.invalidInput(message: "Vectors cannot be empty")
        }

        let numVectors = vectors.count
        let dimension = vectors[0].count

        guard dimension == configuration.dimension else {
            throw IndexAccelerationError.dimensionMismatch(expected: configuration.dimension, got: dimension)
        }
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw IndexAccelerationError.invalidInput(message: "All vectors must have same dimension")
        }

        let startTime = CACurrentMediaTime()
        let device = context.device.rawDevice

        // Create vector buffer
        let flatVectors = vectors.flatMap { $0 }
        guard let vectorBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: flatVectors.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatVectors.count * MemoryLayout<Float>.size)
        }
        vectorBuffer.label = "KMeansPipeline.vectors"

        // Initialize centroids
        let centroids: [[Float]]
        if let initial = initialCentroids {
            guard initial.count == configuration.numClusters else {
                throw IndexAccelerationError.invalidInput(
                    message: "Initial centroids count (\(initial.count)) must match numClusters (\(configuration.numClusters))"
                )
            }
            centroids = initial
        } else {
            // Use K-means++ initialization from VectorIndex
            centroids = try initializeCentroidsKMeansPlusPlus(
                vectors: vectors,
                k: configuration.numClusters
            )
        }

        // Create centroid buffer
        let currentCentroids = centroids.flatMap { $0 }
        guard var centroidBuffer = device.makeBuffer(
            bytes: currentCentroids,
            length: currentCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: currentCentroids.count * MemoryLayout<Float>.size)
        }
        centroidBuffer.label = "KMeansPipeline.centroids"

        // Iteration tracking
        var iterationTimings: [KMeansIterationTiming] = []
        var converged = false
        var iteration = 0
        var assignmentResult: KMeansAssignmentResult?
        var countsBuffer: (any MTLBuffer)?

        // Main iteration loop
        while iteration < configuration.maxIterations && !converged {
            // Step 1: Assignment
            let assignStart = CACurrentMediaTime()
            assignmentResult = try await assignKernel.assign(
                vectors: vectorBuffer,
                centroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: configuration.numClusters,
                dimension: dimension
            )
            let assignTime = CACurrentMediaTime() - assignStart

            // Step 2: Update centroids
            let updateStart = CACurrentMediaTime()
            let (newCentroidBuffer, newCountsBuffer, emptyClusters) = try await updateKernel.update(
                vectors: vectorBuffer,
                assignments: assignmentResult!.assignments,
                currentCentroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: configuration.numClusters,
                dimension: dimension
            )
            let updateTime = CACurrentMediaTime() - updateStart

            // Handle empty clusters
            if emptyClusters > 0 {
                _ = updateKernel.handleEmptyClusters(
                    centroids: newCentroidBuffer,
                    vectors: vectorBuffer,
                    assignments: assignmentResult!.assignments,
                    distances: assignmentResult!.distances,
                    counts: newCountsBuffer,
                    numVectors: numVectors,
                    numCentroids: configuration.numClusters,
                    dimension: dimension
                )
            }

            // Step 3: Check convergence
            let convStart = CACurrentMediaTime()
            let convResult = try await convergenceKernel.checkConvergence(
                oldCentroids: centroidBuffer,
                newCentroids: newCentroidBuffer,
                numCentroids: configuration.numClusters,
                dimension: dimension,
                threshold: configuration.convergenceThreshold
            )
            let convTime = CACurrentMediaTime() - convStart

            converged = convResult.converged

            // Record timing if profiling enabled
            if configuration.enableProfiling {
                iterationTimings.append(KMeansIterationTiming(
                    iteration: iteration,
                    assignmentTime: assignTime,
                    updateTime: updateTime,
                    convergenceTime: convTime,
                    centroidMovement: convResult.maxMovement
                ))
            }

            // Update for next iteration
            centroidBuffer = newCentroidBuffer
            countsBuffer = newCountsBuffer
            iteration += 1
        }

        // Final assignment if needed
        if assignmentResult == nil {
            assignmentResult = try await assignKernel.assign(
                vectors: vectorBuffer,
                centroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: configuration.numClusters,
                dimension: dimension
            )
        }

        // Ensure we have counts
        if countsBuffer == nil {
            countsBuffer = try await assignKernel.computeClusterCounts(
                assignments: assignmentResult!.assignments,
                numVectors: numVectors,
                numCentroids: configuration.numClusters
            )
        }

        let totalTime = CACurrentMediaTime() - startTime
        let inertia = convergenceKernel.computeInertia(
            distances: assignmentResult!.distances,
            numVectors: numVectors
        )

        return KMeansResult(
            centroids: centroidBuffer,
            assignments: assignmentResult!.assignments,
            clusterCounts: countsBuffer!,
            numClusters: configuration.numClusters,
            numVectors: numVectors,
            dimension: dimension,
            iterations: iteration,
            converged: converged,
            inertia: inertia,
            executionTime: totalTime,
            iterationTimings: configuration.enableProfiling ? iterationTimings : nil
        )
    }

    /// Fit K-Means to vectors in a buffer.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - numVectors: Number of vectors
    ///   - initialCentroids: Optional initial centroid buffer
    /// - Returns: K-Means result
    public func fit(
        vectors: any MTLBuffer,
        numVectors: Int,
        initialCentroids: (any MTLBuffer)? = nil
    ) async throws -> KMeansResult {
        let startTime = CACurrentMediaTime()
        let device = context.device.rawDevice
        let dimension = configuration.dimension
        let numClusters = configuration.numClusters

        // Get or create initial centroids
        var centroidBuffer: any MTLBuffer
        if let initial = initialCentroids {
            centroidBuffer = initial
        } else {
            // Extract vectors and use K-means++
            let vectorPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
            var vectorsArray: [[Float]] = []
            vectorsArray.reserveCapacity(numVectors)
            for i in 0..<numVectors {
                var vec: [Float] = []
                vec.reserveCapacity(dimension)
                for d in 0..<dimension {
                    vec.append(vectorPtr[i * dimension + d])
                }
                vectorsArray.append(vec)
            }

            let centroids = try initializeCentroidsKMeansPlusPlus(vectors: vectorsArray, k: numClusters)
            let flatCentroids = centroids.flatMap { $0 }

            guard let buffer = device.makeBuffer(
                bytes: flatCentroids,
                length: flatCentroids.count * MemoryLayout<Float>.size,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
            }
            buffer.label = "KMeansPipeline.centroids"
            centroidBuffer = buffer
        }

        // Iteration loop
        var iterationTimings: [KMeansIterationTiming] = []
        var converged = false
        var iteration = 0
        var assignmentResult: KMeansAssignmentResult?
        var countsBuffer: (any MTLBuffer)?

        while iteration < configuration.maxIterations && !converged {
            // Assignment
            let assignStart = CACurrentMediaTime()
            assignmentResult = try await assignKernel.assign(
                vectors: vectors,
                centroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: numClusters,
                dimension: dimension
            )
            let assignTime = CACurrentMediaTime() - assignStart

            // Update
            let updateStart = CACurrentMediaTime()
            let (newCentroidBuffer, newCountsBuffer, emptyClusters) = try await updateKernel.update(
                vectors: vectors,
                assignments: assignmentResult!.assignments,
                currentCentroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: numClusters,
                dimension: dimension
            )
            let updateTime = CACurrentMediaTime() - updateStart

            if emptyClusters > 0 {
                _ = updateKernel.handleEmptyClusters(
                    centroids: newCentroidBuffer,
                    vectors: vectors,
                    assignments: assignmentResult!.assignments,
                    distances: assignmentResult!.distances,
                    counts: newCountsBuffer,
                    numVectors: numVectors,
                    numCentroids: numClusters,
                    dimension: dimension
                )
            }

            // Convergence
            let convStart = CACurrentMediaTime()
            let convResult = try await convergenceKernel.checkConvergence(
                oldCentroids: centroidBuffer,
                newCentroids: newCentroidBuffer,
                numCentroids: numClusters,
                dimension: dimension,
                threshold: configuration.convergenceThreshold
            )
            let convTime = CACurrentMediaTime() - convStart

            converged = convResult.converged

            if configuration.enableProfiling {
                iterationTimings.append(KMeansIterationTiming(
                    iteration: iteration,
                    assignmentTime: assignTime,
                    updateTime: updateTime,
                    convergenceTime: convTime,
                    centroidMovement: convResult.maxMovement
                ))
            }

            centroidBuffer = newCentroidBuffer
            countsBuffer = newCountsBuffer
            iteration += 1
        }

        if assignmentResult == nil {
            assignmentResult = try await assignKernel.assign(
                vectors: vectors,
                centroids: centroidBuffer,
                numVectors: numVectors,
                numCentroids: numClusters,
                dimension: dimension
            )
        }

        if countsBuffer == nil {
            countsBuffer = try await assignKernel.computeClusterCounts(
                assignments: assignmentResult!.assignments,
                numVectors: numVectors,
                numCentroids: numClusters
            )
        }

        let totalTime = CACurrentMediaTime() - startTime
        let inertia = convergenceKernel.computeInertia(
            distances: assignmentResult!.distances,
            numVectors: numVectors
        )

        return KMeansResult(
            centroids: centroidBuffer,
            assignments: assignmentResult!.assignments,
            clusterCounts: countsBuffer!,
            numClusters: numClusters,
            numVectors: numVectors,
            dimension: dimension,
            iterations: iteration,
            converged: converged,
            inertia: inertia,
            executionTime: totalTime,
            iterationTimings: configuration.enableProfiling ? iterationTimings : nil
        )
    }

    // MARK: - Private Helpers

    /// Initialize centroids using K-means++ algorithm.
    private func initializeCentroidsKMeansPlusPlus(
        vectors: [[Float]],
        k: Int
    ) throws -> [[Float]] {
        let n = vectors.count
        let d = vectors[0].count

        guard n >= k else {
            throw IndexAccelerationError.invalidInput(
                message: "Not enough vectors (\(n)) for \(k) clusters"
            )
        }

        // Flatten vectors for seeding kernel
        let flatData = vectors.flatMap { $0 }
        var flatCentroids = [Float](repeating: 0, count: k * d)

        let cfg = KMeansSeedConfig(
            algorithm: .plusPlus,
            k: k,
            sampleSize: 0,
            rngSeed: UInt64.random(in: 0..<UInt64.max),
            rngStreamID: 0,
            strictFP: false,
            prefetchDistance: 2,
            oversamplingFactor: 2,
            rounds: 5
        )

        _ = try kmeansPlusPlusSeed(
            data: flatData,
            count: n,
            dimension: d,
            k: k,
            config: cfg,
            centroidsOut: &flatCentroids,
            chosenIndicesOut: nil
        )

        // Reshape to [[Float]]
        var centroids: [[Float]] = []
        centroids.reserveCapacity(k)
        for i in 0..<k {
            let start = i * d
            centroids.append(Array(flatCentroids[start..<(start + d)]))
        }

        return centroids
    }
}
