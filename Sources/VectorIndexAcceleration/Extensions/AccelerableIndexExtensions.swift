//
//  AccelerableIndexExtensions.swift
//  VectorIndexAcceleration
//
//  Protocol extensions for AccelerableIndex that provide automatic GPU acceleration.
//
//  Note: This is a Phase 1 placeholder. Full implementation in Phase 4.
//

import Metal
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - AccelerableIndex GPU Extensions

/// Extension providing GPU-accelerated search methods for AccelerableIndex.
///
/// These methods will be fully implemented in Phase 4 when the kernel
/// integration is complete. For Phase 1, they delegate to CPU implementations.
public extension AccelerableIndex {

    /// GPU-accelerated search using Metal 4 kernels.
    ///
    /// This extension automatically accelerates any index conforming to AccelerableIndex
    /// using VectorAccelerate's optimized GPU kernels.
    ///
    /// - Parameters:
    ///   - query: Query vector (must match index dimension)
    ///   - k: Number of nearest neighbors to return
    ///   - filter: Optional metadata filter
    ///   - context: Optional pre-created Metal4Context (created if nil)
    /// - Returns: Search results sorted by distance
    /// - Throws: `VectorError` if search fails
    ///
    /// - Note: Phase 1 placeholder - delegates to CPU implementation.
    ///   Full GPU implementation in Phase 4.
    func searchAccelerated(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil,
        context: Metal4Context? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        // Phase 1: Delegate to CPU implementation
        // Phase 4 will implement:
        // 1. Check if acceleration is beneficial via shouldAccelerate()
        // 2. Get candidates from index
        // 3. Use FusedL2TopKKernel for GPU-accelerated search
        // 4. Finalize results
        return try await search(query: query, k: k, filter: filter)
    }

    /// Batch GPU-accelerated search.
    ///
    /// - Note: Phase 1 placeholder - delegates to CPU implementation.
    func batchSearchAccelerated(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil,
        context: Metal4Context? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        // Phase 1: Delegate to CPU implementation
        return try await batchSearch(queries: queries, k: k, filter: filter)
    }

    /// Search with automatic acceleration decision.
    ///
    /// Automatically decides whether to use GPU or CPU based on workload size.
    func searchAuto(
        query: [Float],
        k: Int
    ) async throws -> [VectorIndex.SearchResult] {
        try await searchAccelerated(query: query, k: k, filter: nil, context: nil)
    }

    /// Batch search with automatic acceleration decision.
    func batchSearchAuto(
        queries: [[Float]],
        k: Int
    ) async throws -> [[VectorIndex.SearchResult]] {
        try await batchSearchAccelerated(queries: queries, k: k, filter: nil, context: nil)
    }
}
