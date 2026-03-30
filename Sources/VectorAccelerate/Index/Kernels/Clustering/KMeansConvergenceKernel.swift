//
//  KMeansConvergenceKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for K-Means convergence checking.
//
//  Computes the movement of centroids between iterations to determine
//  if the algorithm has converged.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means Convergence Kernel

/// Metal 4 kernel for K-Means convergence checking.
///
/// Computes the distance each centroid moved from the previous iteration
/// and determines if the algorithm has converged based on a threshold.
///
/// ## Convergence Criteria
/// The algorithm is considered converged when:
/// - max(centroid_movement) < threshold, OR
/// - all centroids moved less than threshold
///
/// ## Usage
/// ```swift
/// let kernel = try await KMeansConvergenceKernel(context: context)
/// let result = try await kernel.checkConvergence(
///     oldCentroids: oldBuffer,
///     newCentroids: newBuffer,
///     numCentroids: 256,
///     dimension: 128,
///     threshold: 1e-4
/// )
/// ```
public final class KMeansConvergenceKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansConvergenceKernel"

    // MARK: - Initialization

    /// Create a K-Means convergence kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    public init(context: Metal4Context) async throws {
        self.context = context
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // No GPU pipeline to warm up
    }

    // MARK: - Convergence Check

    /// Check if centroids have converged.
    ///
    /// - Parameters:
    ///   - oldCentroids: Previous centroid buffer [numCentroids × dimension]
    ///   - newCentroids: New centroid buffer [numCentroids × dimension]
    ///   - numCentroids: Number of centroids
    ///   - dimension: Vector dimension
    ///   - threshold: Convergence threshold (movement distance)
    /// - Returns: Convergence result
    public func checkConvergence(
        oldCentroids: any MTLBuffer,
        newCentroids: any MTLBuffer,
        numCentroids: Int,
        dimension: Int,
        threshold: Float
    ) async throws -> KMeansConvergenceResult {
        let startTime = CACurrentMediaTime()

        let oldPtr = oldCentroids.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let newPtr = newCentroids.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)

        var maxMovement: Float = 0
        var totalMovement: Float = 0
        var centroidsMoved = 0
        let thresholdSquared = threshold * threshold

        for k in 0..<numCentroids {
            let offset = k * dimension

            // Compute squared L2 distance between old and new centroid
            var distSquared: Float = 0
            for d in 0..<dimension {
                let diff = newPtr[offset + d] - oldPtr[offset + d]
                distSquared += diff * diff
            }

            let dist = sqrt(distSquared)
            totalMovement += dist
            maxMovement = max(maxMovement, dist)

            if distSquared > thresholdSquared {
                centroidsMoved += 1
            }
        }

        let meanMovement = numCentroids > 0 ? totalMovement / Float(numCentroids) : 0
        let converged = maxMovement < threshold

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansConvergenceResult(
            converged: converged,
            maxMovement: maxMovement,
            meanMovement: meanMovement,
            centroidsMoved: centroidsMoved,
            executionTime: executionTime
        )
    }

    /// Check convergence from Float arrays.
    ///
    /// - Parameters:
    ///   - oldCentroids: Previous centroids
    ///   - newCentroids: New centroids
    ///   - threshold: Convergence threshold
    /// - Returns: Convergence result
    public func checkConvergence(
        oldCentroids: [[Float]],
        newCentroids: [[Float]],
        threshold: Float
    ) async throws -> KMeansConvergenceResult {
        guard oldCentroids.count == newCentroids.count else {
            throw IndexError.invalidInput(message: "Centroid counts must match")
        }
        guard !oldCentroids.isEmpty else {
            return KMeansConvergenceResult(
                converged: true,
                maxMovement: 0,
                meanMovement: 0,
                centroidsMoved: 0,
                executionTime: 0
            )
        }

        let dimension = oldCentroids[0].count
        let numCentroids = oldCentroids.count

        // Create buffers
        let flatOld = oldCentroids.flatMap { $0 }
        let oldToken = try await context.getBuffer(for: flatOld)
        let oldBuffer = oldToken.buffer

        let flatNew = newCentroids.flatMap { $0 }
        let newToken = try await context.getBuffer(for: flatNew)
        let newBuffer = newToken.buffer

        let result = try await checkConvergence(
            oldCentroids: oldBuffer,
            newCentroids: newBuffer,
            numCentroids: numCentroids,
            dimension: dimension,
            threshold: threshold
        )
        
        return result
    }

    /// Compute inertia (sum of squared distances to centroids).
    ///
    /// - Parameters:
    ///   - distances: Distance buffer [numVectors]
    ///   - numVectors: Number of vectors
    /// - Returns: Total inertia
    public func computeInertia(
        distances: any MTLBuffer,
        numVectors: Int
    ) -> Float {
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: numVectors)
        return (0..<numVectors).reduce(Float(0)) { $0 + distPtr[$1] }
    }
}
