//
//  KMeansPlusPlusKernel.swift
//  VectorAccelerate
//
//  GPU-accelerated K-means++ initialization.
//
//  Uses the compute_min_distances Metal kernel to accelerate the distance
//  computation phase of K-means++, reducing complexity from O(N × K² × D)
//  on CPU to O(N × K × D) on GPU.
//
//  Algorithm:
//  1. Select first centroid uniformly at random
//  2. For each remaining centroid:
//     a. GPU: Compute min distance from each point to any selected centroid
//     b. CPU: Sample next centroid with probability proportional to D²
//  3. Return K selected centroids
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means++ Kernel

/// GPU-accelerated K-means++ centroid initialization.
///
/// K-means++ selects initial centroids with probability proportional to
/// squared distance from existing centroids, providing:
/// - 2-approximation to optimal clustering in expectation
/// - Faster convergence than random initialization
/// - More stable results across runs
///
/// ## Performance
/// - GPU-accelerated distance computation: O(N × K × D)
/// - CPU probability sampling: O(N × K)
/// - Total: ~10-20x faster than CPU K-means++ for typical workloads
///
/// ## Usage
/// ```swift
/// let kernel = try await KMeansPlusPlusKernel(context: context)
/// let centroids = try await kernel.selectCentroids(
///     from: vectorBuffer,
///     numVectors: 20000,
///     dimension: 128,
///     k: 141
/// )
/// ```
public final class KMeansPlusPlusKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansPlusPlusKernel"

    // MARK: - Private Properties

    private let computeMinDistancesPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a K-means++ initialization kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "compute_min_distances") else {
            throw VectorError.shaderNotFound(name: "compute_min_distances")
        }

        self.computeMinDistancesPipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // No warmup needed - kernel is simple
    }

    // MARK: - Centroid Selection

    /// Select K centroids using K-means++ algorithm with GPU acceleration.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - numVectors: Number of vectors
    ///   - dimension: Vector dimension
    ///   - k: Number of centroids to select
    /// - Returns: Selected centroid indices
    public func selectCentroids(
        from vectors: any MTLBuffer,
        numVectors: Int,
        dimension: Int,
        k: Int
    ) async throws -> [Int] {
        guard numVectors >= k else {
            throw IndexError.invalidInput(
                message: "Not enough vectors (\(numVectors)) for \(k) clusters"
            )
        }
        guard k > 0 else {
            throw IndexError.invalidInput(message: "k must be positive")
        }

        // Track selected centroid indices (ordered list returned to the caller).
        var selectedIndices: [Int] = []
        selectedIndices.reserveCapacity(k)
        // Parallel O(1) membership test. The sampling loops below previously used
        // `selectedIndices.contains(_:)` — an O(K) linear scan — inside O(N) loops nested
        // in the O(K) centroid loop, i.e. O(N·K²). A boolean mask makes membership O(1),
        // reducing the whole selection to O(N·K).
        var isSelected = [Bool](repeating: false, count: numVectors)

        // Buffer for selected centroids [k × dimension]
        let centroidBufferSize = k * dimension * MemoryLayout<Float>.size
        let centroidToken = try await context.getBuffer(size: centroidBufferSize)
        let centroidBuffer = centroidToken.buffer
        centroidBuffer.label = "KMeansPlusPlus.selectedCentroids"

        // Buffer for min distances [numVectors]
        let distanceBufferSize = numVectors * MemoryLayout<Float>.size
        let distanceToken = try await context.getBuffer(size: distanceBufferSize)
        let distanceBuffer = distanceToken.buffer
        distanceBuffer.label = "KMeansPlusPlus.minDistances"

        // Step 1: Select first centroid uniformly at random
        let firstIdx = Int.random(in: 0..<numVectors)
        selectedIndices.append(firstIdx)
        isSelected[firstIdx] = true

        // Copy first centroid to centroid buffer
        let vectorsPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        let centroidsPtr = centroidBuffer.contents().bindMemory(to: Float.self, capacity: k * dimension)
        for d in 0..<dimension {
            centroidsPtr[d] = vectorsPtr[firstIdx * dimension + d]
        }

        // Step 2: Select remaining k-1 centroids
        for centroidIdx in 1..<k {
            // Run GPU kernel to compute min distances
            try await computeMinDistances(
                vectors: vectors,
                centroids: centroidBuffer,
                minDistances: distanceBuffer,
                numVectors: numVectors,
                numCentroids: centroidIdx,
                dimension: dimension
            )

            // Sample next centroid with probability proportional to D²
            let distancesPtr = distanceBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)

            // Compute total weight (sum of squared distances).
            // Accumulate in Double: a Float32 running sum saturates its 24-bit mantissa
            // when summing millions of positive squared distances (catastrophic absorption),
            // which silently drives tail points' selection probability toward zero.
            var totalWeight: Double = 0
            for i in 0..<numVectors {
                if !isSelected[i] {
                    totalWeight += Double(distancesPtr[i])  // Already squared in kernel
                }
            }

            // Sample next centroid
            var nextIdx = 0
            if totalWeight > 0 {
                let threshold = Double.random(in: 0..<totalWeight)
                var cumulative: Double = 0

                for i in 0..<numVectors {
                    if isSelected[i] { continue }
                    cumulative += Double(distancesPtr[i])
                    if cumulative >= threshold {
                        nextIdx = i
                        break
                    }
                }
            } else {
                // All remaining points are at centroids, pick any unselected
                for i in 0..<numVectors {
                    if !isSelected[i] {
                        nextIdx = i
                        break
                    }
                }
            }

            selectedIndices.append(nextIdx)
            isSelected[nextIdx] = true

            // Copy selected centroid to buffer
            let offset = centroidIdx * dimension
            for d in 0..<dimension {
                centroidsPtr[offset + d] = vectorsPtr[nextIdx * dimension + d]
            }
        }

        return selectedIndices
    }

    /// Select K centroids and return them as a buffer.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - numVectors: Number of vectors
    ///   - dimension: Vector dimension
    ///   - k: Number of centroids to select
    /// - Returns: Buffer containing selected centroids [k × dimension]
    public func selectCentroidsBuffer(
        from vectors: any MTLBuffer,
        numVectors: Int,
        dimension: Int,
        k: Int
    ) async throws -> any MTLBuffer {
        let indices = try await selectCentroids(
            from: vectors,
            numVectors: numVectors,
            dimension: dimension,
            k: k
        )

        let bufferSize = k * dimension * MemoryLayout<Float>.size

        let centroidToken = try await context.getBuffer(size: bufferSize)
        let centroidBuffer = centroidToken.buffer
        centroidBuffer.label = "KMeansPlusPlus.centroids"

        let vectorsPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        let centroidsPtr = centroidBuffer.contents().bindMemory(to: Float.self, capacity: k * dimension)

        for (i, idx) in indices.enumerated() {
            let srcOffset = idx * dimension
            let dstOffset = i * dimension
            for d in 0..<dimension {
                centroidsPtr[dstOffset + d] = vectorsPtr[srcOffset + d]
            }
        }

        return centroidBuffer
    }

    // MARK: - Private Methods

    /// Run GPU kernel to compute minimum distance from each point to any selected centroid.
    private func computeMinDistances(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        minDistances: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(computeMinDistancesPipeline)

            encoder.setBuffer(vectors, offset: 0, index: 0)
            encoder.setBuffer(centroids, offset: 0, index: 1)
            encoder.setBuffer(minDistances, offset: 0, index: 2)

            var nVec = UInt32(numVectors)
            var nCent = UInt32(numCentroids)
            var dim = UInt32(dimension)

            encoder.setBytes(&nVec, length: 4, index: 3)
            encoder.setBytes(&nCent, length: 4, index: 4)
            encoder.setBytes(&dim, length: 4, index: 5)

            let threadsPerThreadgroup = min(256, computeMinDistancesPipeline.maxTotalThreadsPerThreadgroup)
            let threadgroups = (numVectors + threadsPerThreadgroup - 1) / threadsPerThreadgroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
            )
        }
    }
}
