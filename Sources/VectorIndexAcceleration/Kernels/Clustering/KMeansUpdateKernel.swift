//
//  KMeansUpdateKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for K-Means centroid update phase.
//
//  Updates centroids by computing the mean of all vectors assigned
//  to each cluster. This is the M-step of the EM algorithm.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorCore

// MARK: - K-Means Update Kernel

/// Metal 4 kernel for K-Means centroid update.
///
/// Computes new centroids as the mean of all vectors assigned to each cluster.
/// This is the M-step of the EM algorithm.
///
/// ## Algorithm
/// For each centroid k:
///   centroid[k] = sum(vectors where assignment == k) / count[k]
///
/// ## Empty Cluster Handling
/// If a cluster becomes empty, the centroid is reinitialized to the
/// vector furthest from its current centroid (split strategy).
///
/// ## Usage
/// ```swift
/// let kernel = try await KMeansUpdateKernel(context: context)
/// let newCentroids = try await kernel.update(
///     vectors: vectorBuffer,
///     assignments: assignmentBuffer,
///     currentCentroids: centroidBuffer,
///     numVectors: 10000,
///     numCentroids: 256,
///     dimension: 128
/// )
/// ```
public final class KMeansUpdateKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansUpdateKernel"

    // MARK: - Initialization

    /// Create a K-Means update kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    public init(context: Metal4Context) async throws {
        self.context = context
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // No GPU pipeline to warm up - using CPU accumulation
    }

    // MARK: - Update

    /// Update centroids based on vector assignments.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - assignments: Assignment buffer [numVectors]
    ///   - currentCentroids: Current centroid buffer [numCentroids × dimension]
    ///   - numVectors: Number of vectors
    ///   - numCentroids: Number of centroids (K)
    ///   - dimension: Vector dimension
    /// - Returns: New centroid buffer [numCentroids × dimension]
    public func update(
        vectors: any MTLBuffer,
        assignments: any MTLBuffer,
        currentCentroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws -> (centroids: any MTLBuffer, counts: any MTLBuffer, emptyClusters: Int) {
        let device = context.device.rawDevice

        // Allocate output buffers
        guard let newCentroidsBuffer = device.makeBuffer(
            length: numCentroids * dimension * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numCentroids * dimension * MemoryLayout<Float>.size)
        }
        newCentroidsBuffer.label = "KMeansUpdate.newCentroids"

        guard let countsBuffer = device.makeBuffer(
            length: numCentroids * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numCentroids * MemoryLayout<UInt32>.size)
        }
        countsBuffer.label = "KMeansUpdate.counts"

        // Get pointers
        let vectorsPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        let assignPtr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        let currentPtr = currentCentroids.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let newPtr = newCentroidsBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)

        // Initialize accumulators to zero
        for i in 0..<(numCentroids * dimension) {
            newPtr[i] = 0
        }
        for i in 0..<numCentroids {
            countsPtr[i] = 0
        }

        // Accumulate vectors per cluster
        for v in 0..<numVectors {
            let cluster = Int(assignPtr[v])
            guard cluster < numCentroids else { continue }

            countsPtr[cluster] += 1

            let vecOffset = v * dimension
            let centOffset = cluster * dimension
            for d in 0..<dimension {
                newPtr[centOffset + d] += vectorsPtr[vecOffset + d]
            }
        }

        // Divide by count to get mean, track empty clusters
        var emptyClusters = 0
        for k in 0..<numCentroids {
            let count = countsPtr[k]
            let offset = k * dimension

            if count > 0 {
                let invCount = 1.0 / Float(count)
                for d in 0..<dimension {
                    newPtr[offset + d] *= invCount
                }
            } else {
                // Empty cluster - keep current centroid
                emptyClusters += 1
                for d in 0..<dimension {
                    newPtr[offset + d] = currentPtr[offset + d]
                }
            }
        }

        return (newCentroidsBuffer, countsBuffer, emptyClusters)
    }

    /// Update centroids from Float arrays.
    ///
    /// - Parameters:
    ///   - vectors: Vector data as 2D Float array
    ///   - assignments: Cluster assignment for each vector
    ///   - currentCentroids: Current centroid positions
    /// - Returns: New centroids and counts per cluster
    public func update(
        vectors: [[Float]],
        assignments: [Int],
        currentCentroids: [[Float]]
    ) async throws -> (centroids: [[Float]], counts: [Int], emptyClusters: Int) {
        guard !vectors.isEmpty else {
            return (currentCentroids, Array(repeating: 0, count: currentCentroids.count), currentCentroids.count)
        }

        let dimension = vectors[0].count
        let numCentroids = currentCentroids.count
        let numVectors = vectors.count

        let device = context.device.rawDevice

        // Create buffers
        let flatVectors = vectors.flatMap { $0 }
        guard let vectorBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: flatVectors.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatVectors.count * MemoryLayout<Float>.size)
        }

        let assignUInt32 = assignments.map { UInt32($0) }
        guard let assignBuffer = device.makeBuffer(
            bytes: assignUInt32,
            length: assignUInt32.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: assignUInt32.count * MemoryLayout<UInt32>.size)
        }

        let flatCentroids = currentCentroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: flatCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
        }

        // Execute update
        let (newCentroidsBuffer, countsBuffer, emptyClusters) = try await update(
            vectors: vectorBuffer,
            assignments: assignBuffer,
            currentCentroids: centroidBuffer,
            numVectors: numVectors,
            numCentroids: numCentroids,
            dimension: dimension
        )

        // Extract results
        let newPtr = newCentroidsBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)

        var newCentroids: [[Float]] = []
        newCentroids.reserveCapacity(numCentroids)
        for k in 0..<numCentroids {
            var centroid: [Float] = []
            centroid.reserveCapacity(dimension)
            for d in 0..<dimension {
                centroid.append(newPtr[k * dimension + d])
            }
            newCentroids.append(centroid)
        }

        let counts = (0..<numCentroids).map { Int(countsPtr[$0]) }

        return (newCentroids, counts, emptyClusters)
    }

    /// Handle empty clusters by reinitializing them.
    ///
    /// Uses the "split largest cluster" strategy: the vector furthest
    /// from its centroid in the largest cluster becomes a new centroid.
    ///
    /// - Parameters:
    ///   - centroids: Current centroids buffer
    ///   - vectors: Vector buffer
    ///   - assignments: Assignment buffer
    ///   - distances: Distance buffer (to assigned centroids)
    ///   - counts: Cluster counts
    ///   - numVectors: Number of vectors
    ///   - numCentroids: Number of centroids
    ///   - dimension: Vector dimension
    /// - Returns: Number of clusters reinitialized
    public func handleEmptyClusters(
        centroids: any MTLBuffer,
        vectors: any MTLBuffer,
        assignments: any MTLBuffer,
        distances: any MTLBuffer,
        counts: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) -> Int {
        let centroidsPtr = centroids.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let vectorsPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        let assignPtr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: numVectors)
        let countsPtr = counts.contents().bindMemory(to: UInt32.self, capacity: numCentroids)

        // Find empty clusters
        var emptyIndices: [Int] = []
        for k in 0..<numCentroids {
            if countsPtr[k] == 0 {
                emptyIndices.append(k)
            }
        }

        if emptyIndices.isEmpty { return 0 }

        // Find the largest cluster
        var largestCluster = 0
        var largestCount: UInt32 = 0
        for k in 0..<numCentroids {
            if countsPtr[k] > largestCount {
                largestCount = countsPtr[k]
                largestCluster = k
            }
        }

        // Find the vector furthest from its centroid in the largest cluster
        var furthestIdx = -1
        var furthestDist: Float = -1
        for v in 0..<numVectors {
            if assignPtr[v] == UInt32(largestCluster) {
                if distPtr[v] > furthestDist {
                    furthestDist = distPtr[v]
                    furthestIdx = v
                }
            }
        }

        // Reinitialize empty clusters
        var reinitialized = 0
        for emptyIdx in emptyIndices {
            if furthestIdx >= 0 {
                // Copy the furthest vector as new centroid
                let srcOffset = furthestIdx * dimension
                let dstOffset = emptyIdx * dimension
                for d in 0..<dimension {
                    centroidsPtr[dstOffset + d] = vectorsPtr[srcOffset + d]
                }
                reinitialized += 1

                // Update counts
                countsPtr[emptyIdx] = 1
                countsPtr[largestCluster] -= 1
                assignPtr[furthestIdx] = UInt32(emptyIdx)

                // Find next furthest in largest cluster
                furthestDist = -1
                furthestIdx = -1
                for v in 0..<numVectors {
                    if assignPtr[v] == UInt32(largestCluster) && distPtr[v] > furthestDist {
                        furthestDist = distPtr[v]
                        furthestIdx = v
                    }
                }
            }
        }

        return reinitialized
    }
}
