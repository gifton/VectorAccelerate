//
//  IVFCoarseQuantizerKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for IVF coarse quantization (finding nearest centroids).
//
//  This kernel wraps the existing FusedL2TopKKernel to find the nprobe
//  nearest centroids for each query vector.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - IVF Coarse Quantizer Kernel

/// Metal 4 kernel for IVF coarse quantization.
///
/// Finds the nprobe nearest centroids for each query vector using
/// fused distance computation and top-k selection.
///
/// ## Usage
/// ```swift
/// let kernel = try await IVFCoarseQuantizerKernel(context: context)
/// let result = try await kernel.findNearestCentroids(
///     queries: queryBuffer,
///     centroids: centroidBuffer,
///     numQueries: 100,
///     numCentroids: 256,
///     dimension: 128,
///     nprobe: 8
/// )
/// ```
public final class IVFCoarseQuantizerKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "IVFCoarseQuantizerKernel"

    // MARK: - Private Properties

    private let fusedL2TopK: FusedL2TopKKernel

    // MARK: - Initialization

    /// Create an IVF coarse quantizer kernel.
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

    // MARK: - Coarse Quantization

    /// Find the nprobe nearest centroids for each query.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer [numQueries × dimension]
    ///   - centroids: Centroid vectors buffer [numCentroids × dimension]
    ///   - numQueries: Number of query vectors
    ///   - numCentroids: Number of centroids
    ///   - dimension: Vector dimension
    ///   - nprobe: Number of nearest centroids to find per query
    /// - Returns: Coarse quantization result with list indices and distances
    public func findNearestCentroids(
        queries: any MTLBuffer,
        centroids: any MTLBuffer,
        numQueries: Int,
        numCentroids: Int,
        dimension: Int,
        nprobe: Int
    ) async throws -> IVFCoarseResult {
        // Validate inputs
        guard numQueries > 0 else {
            throw IndexError.invalidInput(message: "numQueries must be positive")
        }
        guard numCentroids > 0 else {
            throw IndexError.invalidInput(message: "numCentroids must be positive")
        }
        guard nprobe > 0 && nprobe <= numCentroids else {
            throw IndexError.invalidInput(message: "nprobe must be between 1 and numCentroids")
        }

        // Use fused L2 + Top-K to find nearest centroids
        let params = try FusedL2TopKParameters(
            numQueries: numQueries,
            numDataset: numCentroids,
            dimension: dimension,
            k: nprobe
        )

        let result = try await fusedL2TopK.execute(
            queries: queries,
            dataset: centroids,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        // Wrap result in IVF-specific type
        return IVFCoarseResult(
            listIndices: result.indices,
            listDistances: result.distances!,
            numQueries: numQueries,
            nprobe: nprobe
        )
    }

    /// Find nearest centroids from Float arrays.
    ///
    /// - Parameters:
    ///   - queries: Query vectors as 2D Float array
    ///   - centroids: Centroid vectors as 2D Float array
    ///   - nprobe: Number of nearest centroids to find
    /// - Returns: Array of selected list indices per query
    public func findNearestCentroids(
        queries: [[Float]],
        centroids: [[Float]],
        nprobe: Int
    ) async throws -> [[Int]] {
        guard !queries.isEmpty else { return [] }
        guard !centroids.isEmpty else {
            throw IndexError.invalidInput(message: "Centroids cannot be empty")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }),
              centroids.allSatisfy({ $0.count == dimension }) else {
            throw IndexError.invalidInput(message: "All vectors must have same dimension")
        }

        let device = context.device.rawDevice

        // Create query buffer
        let flatQueries = queries.flatMap { $0 }
        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "IVFCoarseQuantizer.queries"

        // Create centroid buffer
        let flatCentroids = centroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: flatCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
        }
        centroidBuffer.label = "IVFCoarseQuantizer.centroids"

        // Execute
        let result = try await findNearestCentroids(
            queries: queryBuffer,
            centroids: centroidBuffer,
            numQueries: queries.count,
            numCentroids: centroids.count,
            dimension: dimension,
            nprobe: nprobe
        )

        // Extract results
        return (0..<queries.count).map { result.selectedLists(for: $0) }
    }
}
