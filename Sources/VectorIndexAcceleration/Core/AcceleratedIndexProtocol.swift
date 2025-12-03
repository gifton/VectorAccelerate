//
//  AcceleratedIndexProtocol.swift
//  VectorIndexAcceleration
//
//  Protocol defining requirements for GPU-accelerated index implementations.
//
//  This protocol extends VectorIndex's AccelerableIndex with Metal 4 specific
//  acceleration capabilities, providing a bridge between index structures and
//  GPU kernels.
//

import Metal
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - GPU Accelerated Index Protocol

/// Protocol for indices that support Metal 4 GPU acceleration.
///
/// This protocol bridges VectorIndex's `AccelerableIndex` with VectorAccelerate's
/// Metal 4 infrastructure, enabling high-performance GPU-accelerated search operations.
///
/// ## Conforming Types
/// - `HNSWIndexAccelerated`
/// - `IVFIndexAccelerated`
/// - `FlatIndexAccelerated`
///
/// ## Usage
/// ```swift
/// let index = try await HNSWIndexAccelerated(baseIndex: hnswIndex)
/// let results = try await index.searchGPU(query: queryVector, k: 10)
/// ```
public protocol GPUAcceleratedIndex: AccelerableIndex {

    /// The Metal 4 context used for GPU operations.
    /// May be nil if GPU acceleration is not yet initialized.
    var gpuContext: Metal4Context? { get }

    /// Whether GPU acceleration is currently active.
    var isGPUActive: Bool { get }

    /// Initialize GPU resources for acceleration.
    ///
    /// Call this before performing GPU-accelerated operations. If not called explicitly,
    /// GPU resources will be lazily initialized on first accelerated operation.
    ///
    /// - Parameter context: Optional pre-created Metal4Context. If nil, creates a new one.
    /// - Throws: `VectorError` if GPU initialization fails.
    func prepareForGPU(context: Metal4Context?) async throws

    /// Release GPU resources.
    ///
    /// Call this when GPU acceleration is no longer needed to free GPU memory.
    /// The index remains usable with CPU-only operations after this call.
    func releaseGPUResources() async

    /// Perform GPU-accelerated k-NN search.
    ///
    /// - Parameters:
    ///   - query: Query vector (must match index dimension)
    ///   - k: Number of nearest neighbors to return
    ///   - filter: Optional metadata filter
    /// - Returns: Array of search results sorted by distance
    /// - Throws: `VectorError` if search fails
    func searchGPU(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [VectorIndex.SearchResult]

    /// Perform GPU-accelerated batch k-NN search.
    ///
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of nearest neighbors per query
    ///   - filter: Optional metadata filter
    /// - Returns: Array of search result arrays, one per query
    /// - Throws: `VectorError` if search fails
    func batchSearchGPU(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [[VectorIndex.SearchResult]]
}

// MARK: - Default Implementations

public extension GPUAcceleratedIndex {

    /// Default implementation checks if gpuContext is non-nil
    var isGPUActive: Bool {
        gpuContext != nil
    }
}

// MARK: - Acceleration Decision

/// Factors used to decide whether to use GPU acceleration.
public struct AccelerationDecision: Sendable {
    /// Whether GPU acceleration is recommended
    public let useGPU: Bool

    /// Reason for the decision
    public let reason: Reason

    /// Estimated speedup factor (1.0 = no speedup, 2.0 = 2x faster)
    public let estimatedSpeedup: Float

    public enum Reason: Sendable {
        case datasetTooSmall
        case queryCountTooLow
        case gpuNotAvailable
        case gpuRecommended
        case forcedByConfiguration
    }

    public init(useGPU: Bool, reason: Reason, estimatedSpeedup: Float = 1.0) {
        self.useGPU = useGPU
        self.reason = reason
        self.estimatedSpeedup = estimatedSpeedup
    }
}
