//
//  KMeansAssignKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for K-Means point assignment phase.
//
//  Assigns each vector to its nearest centroid using GPU-accelerated
//  distance computation. Uses FusedL2TopKKernel with k=1.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means Assign Kernel

/// Metal 4 kernel for K-Means point assignment.
///
/// Assigns each vector to its nearest centroid using GPU-accelerated
/// distance computation. This is the E-step of the EM algorithm.
///
/// ## Usage
/// ```swift
/// let kernel = try await KMeansAssignKernel(context: context)
/// let result = try await kernel.assign(
///     vectors: vectorBuffer,
///     centroids: centroidBuffer,
///     numVectors: 10000,
///     numCentroids: 256,
///     dimension: 128
/// )
/// ```
public final class KMeansAssignKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansAssignKernel"

    // MARK: - Private Properties

    private let fusedL2TopK: FusedL2TopKKernel

    // MARK: - Initialization

    /// Create a K-Means assignment kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    public init(context: Metal4Context) async throws {
        self.context = context
        self.fusedL2TopK = try await FusedL2TopKKernel(context: context)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        try await fusedL2TopK.warmUp()
    }

    // MARK: - Assignment

    /// Assign vectors to nearest centroids.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - centroids: Centroid buffer [numCentroids × dimension]
    ///   - numVectors: Number of vectors to assign
    ///   - numCentroids: Number of centroids (K)
    ///   - dimension: Vector dimension
    /// - Returns: Assignment result with cluster assignments and distances
    public func assign(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws -> KMeansAssignmentResult {
        let startTime = CACurrentMediaTime()

        guard numVectors > 0 else {
            throw IndexError.invalidInput(message: "numVectors must be positive")
        }
        guard numCentroids > 0 else {
            throw IndexError.invalidInput(message: "numCentroids must be positive")
        }

        // Use fused L2 + Top-K with k=1 to find nearest centroid
        let params = FusedL2TopKParameters(
            numQueries: numVectors,
            numDataset: numCentroids,
            dimension: dimension,
            k: 1  // We only need the nearest centroid
        )

        let result = try await fusedL2TopK.execute(
            queries: vectors,
            dataset: centroids,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansAssignmentResult(
            assignments: result.indices,
            distances: result.distances!,
            numVectors: numVectors,
            executionTime: executionTime
        )
    }

    /// Assign vectors from Float arrays.
    ///
    /// - Parameters:
    ///   - vectors: Vector data as 2D Float array
    ///   - centroids: Centroid data as 2D Float array
    /// - Returns: Tuple of (assignments, distances)
    public func assign(
        vectors: [[Float]],
        centroids: [[Float]]
    ) async throws -> (assignments: [Int], distances: [Float]) {
        guard !vectors.isEmpty else { return ([], []) }
        guard !centroids.isEmpty else {
            throw IndexError.invalidInput(message: "Centroids cannot be empty")
        }

        let dimension = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dimension }),
              centroids.allSatisfy({ $0.count == dimension }) else {
            throw IndexError.invalidInput(message: "All vectors must have same dimension")
        }

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
        vectorBuffer.label = "KMeansAssign.vectors"

        // Create centroid buffer
        let flatCentroids = centroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: flatCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
        }
        centroidBuffer.label = "KMeansAssign.centroids"

        // Execute
        let result = try await assign(
            vectors: vectorBuffer,
            centroids: centroidBuffer,
            numVectors: vectors.count,
            numCentroids: centroids.count,
            dimension: dimension
        )

        // Extract results
        let assignPtr = result.assignments.contents().bindMemory(to: UInt32.self, capacity: vectors.count)
        let distPtr = result.distances.contents().bindMemory(to: Float.self, capacity: vectors.count)

        let assignments = (0..<vectors.count).map { Int(assignPtr[$0]) }
        let distances = (0..<vectors.count).map { distPtr[$0] }

        return (assignments, distances)
    }

    /// Compute cluster counts from assignments.
    ///
    /// - Parameters:
    ///   - assignments: Assignment buffer [numVectors]
    ///   - numVectors: Number of vectors
    ///   - numCentroids: Number of centroids
    /// - Returns: Buffer with count per cluster [numCentroids]
    public func computeClusterCounts(
        assignments: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice

        // Allocate counts buffer
        guard let countsBuffer = device.makeBuffer(
            length: numCentroids * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numCentroids * MemoryLayout<UInt32>.size)
        }
        countsBuffer.label = "KMeansAssign.counts"

        // Initialize to zero
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        for i in 0..<numCentroids {
            countsPtr[i] = 0
        }

        // Count assignments (CPU for now - could be GPU with atomics)
        let assignPtr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        for i in 0..<numVectors {
            let cluster = Int(assignPtr[i])
            if cluster < numCentroids {
                countsPtr[cluster] += 1
            }
        }

        return countsBuffer
    }
}
