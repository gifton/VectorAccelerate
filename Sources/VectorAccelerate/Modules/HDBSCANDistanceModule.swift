//
//  HDBSCANDistanceModule.swift
//  VectorAccelerate
//
//  High-level module for HDBSCAN distance computations.
//  Combines core distance computation with MST construction.
//
//  Phase 3: BoruvkaMSTKernel Integration
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Result Type

/// Result of HDBSCAN distance computation.
public struct HDBSCANDistanceResult: Sendable {
    /// Core distances for each point (k-th nearest neighbor distance)
    public let coreDistances: [Float]

    /// Minimum Spanning Tree over mutual reachability distances
    public let mst: MSTResult

    /// Number of points processed
    public let pointCount: Int

    /// k value used for core distance computation
    public let minSamples: Int

    public init(
        coreDistances: [Float],
        mst: MSTResult,
        pointCount: Int,
        minSamples: Int
    ) {
        self.coreDistances = coreDistances
        self.mst = mst
        self.pointCount = pointCount
        self.minSamples = minSamples
    }
}

// MARK: - Module Implementation

/// High-level module for HDBSCAN clustering distance computations.
///
/// This module provides a simplified API for computing the mutual reachability
/// MST required by HDBSCAN clustering. It combines:
/// 1. Core distance computation (k-th nearest neighbor distance)
/// 2. MST construction over mutual reachability distances
///
/// ## Usage
///
/// ```swift
/// let module = try await HDBSCANDistanceModule(context: context)
/// let result = try await module.computeMST(
///     embeddings: documentEmbeddings,
///     minSamples: 5
/// )
/// // result.mst contains the MST for cluster extraction
/// ```
///
/// ## Performance
///
/// | Corpus Size | Expected Time |
/// |-------------|---------------|
/// | 500 docs | ~50ms |
/// | 1,000 docs | ~150ms |
/// | 5,000 docs | ~2s |
///
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class HDBSCANDistanceModule: @unchecked Sendable {

    // MARK: - Properties

    public let context: Metal4Context

    private let topKKernel: FusedL2TopKKernel
    private let mstKernel: BoruvkaMSTKernel

    // MARK: - Initialization

    /// Create an HDBSCAN distance module.
    ///
    /// - Parameter context: Metal 4 context
    public init(context: Metal4Context) async throws {
        self.context = context
        self.topKKernel = try await FusedL2TopKKernel(context: context)
        self.mstKernel = try await BoruvkaMSTKernel(context: context)
    }

    // MARK: - Public API

    /// Compute MST over mutual reachability distances.
    ///
    /// This is the primary entry point for HDBSCAN distance computation.
    /// It computes core distances using k-NN search, then builds the MST.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix (row-major)
    ///   - minSamples: k value for core distance (default: 5)
    /// - Returns: Core distances and MST result
    public func computeMST(
        embeddings: [[Float]],
        minSamples: Int = 5
    ) async throws -> HDBSCANDistanceResult {
        let n = embeddings.count
        guard n > 0 else {
            return HDBSCANDistanceResult(
                coreDistances: [],
                mst: MSTResult(edges: [], totalWeight: 0, iterations: 0, pointCount: 0),
                pointCount: 0,
                minSamples: minSamples
            )
        }

        // Step 1: Compute core distances (k-th nearest neighbor distance)
        let coreDistances = try await computeCoreDistances(
            embeddings: embeddings,
            k: minSamples
        )

        // Step 2: Compute MST over mutual reachability
        let mst = try await mstKernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        return HDBSCANDistanceResult(
            coreDistances: coreDistances,
            mst: mst,
            pointCount: n,
            minSamples: minSamples
        )
    }

    /// Compute MST from VectorProtocol embeddings.
    public func computeMST<V: VectorProtocol>(
        embeddings: [V],
        minSamples: Int = 5
    ) async throws -> HDBSCANDistanceResult where V.Scalar == Float {
        // Convert to [[Float]] for now; optimize later with direct buffer API
        let floatEmbeddings = embeddings.map { vector -> [Float] in
            var result = [Float](repeating: 0, count: vector.count)
            vector.withUnsafeBufferPointer { ptr in
                for i in 0..<ptr.count {
                    result[i] = ptr[i]
                }
            }
            return result
        }
        return try await computeMST(embeddings: floatEmbeddings, minSamples: minSamples)
    }

    /// Compute MST with pre-computed core distances.
    ///
    /// Use this when core distances are already available (e.g., from a previous
    /// FusedL2TopKKernel call).
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix
    ///   - coreDistances: Pre-computed core distances
    /// - Returns: MST result
    public func computeMSTWithCoreDistances(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> MSTResult {
        return try await mstKernel.computeMST(
            embeddings: embeddings,
            coreDistances: coreDistances
        )
    }

    // MARK: - Private Helpers

    /// Compute core distances using k-NN search.
    ///
    /// The core distance for point p is the distance to its k-th nearest neighbor.
    private func computeCoreDistances(
        embeddings: [[Float]],
        k: Int
    ) async throws -> [Float] {
        let n = embeddings.count
        guard n > 1 else {
            return [Float](repeating: 0, count: n)
        }

        // Use k+1 because the nearest neighbor of a point is itself (distance 0)
        let effectiveK = min(k + 1, n)

        // Compute all-pairs k-NN (query = database = embeddings)
        let results = try await topKKernel.findNearestNeighbors(
            queries: embeddings,
            dataset: embeddings,
            k: effectiveK,
            includeDistances: true
        )

        // Extract k-th nearest neighbor distance (index k, since index 0 is self)
        var coreDistances = [Float](repeating: 0, count: n)
        for i in 0..<n {
            if results[i].count >= effectiveK {
                // The k-th index (0-indexed) after self is at position min(k, count-1)
                let kIndex = min(k, results[i].count - 1)
                // FusedL2TopK returns squared distances, need sqrt for core distance
                coreDistances[i] = sqrt(results[i][kIndex].distance)
            }
        }

        return coreDistances
    }
}
