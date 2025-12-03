//
//  IndexConfiguration.swift
//  VectorIndexAcceleration
//
//  Configuration for creating an AcceleratedVectorIndex.
//

import Foundation
import VectorCore

// MARK: - Index Configuration

/// Configuration for creating an AcceleratedVectorIndex.
///
/// Specifies the vector dimension, distance metric, storage capacity,
/// and index type (flat or IVF).
///
/// ## Usage
/// ```swift
/// // Using factory methods (recommended)
/// let flatConfig = IndexConfiguration.flat(dimension: 768, capacity: 10_000)
/// let ivfConfig = IndexConfiguration.ivf(dimension: 768, nlist: 100, nprobe: 10)
///
/// // Using initializer
/// let config = IndexConfiguration(
///     dimension: 768,
///     metric: .euclidean,
///     capacity: 50_000,
///     indexType: .ivf(nlist: 256, nprobe: 16)
/// )
/// ```
public struct IndexConfiguration: Sendable, Equatable {

    // MARK: - Properties

    /// Vector dimension. All vectors must have this exact length.
    public let dimension: Int

    /// Distance metric for similarity computation.
    ///
    /// - `.euclidean`: Returns L2² (squared euclidean distance)
    /// - `.cosine`: Returns 1 - cosine_similarity
    /// - `.dotProduct`: Returns -dot_product (negated for min-heap ordering)
    public let metric: SupportedDistanceMetric

    /// Initial capacity (number of vectors).
    ///
    /// The GPU buffer is pre-allocated to this size.
    /// If exceeded, the buffer grows automatically (2x strategy).
    public let capacity: Int

    /// Index type and parameters.
    public let indexType: IndexType

    // MARK: - Index Type

    /// Type of index structure.
    public enum IndexType: Sendable, Equatable {
        /// Flat (brute-force) index.
        /// Best for small datasets (< 10K vectors) or when exact results are required.
        case flat

        /// Inverted File index with clustering.
        /// - Parameters:
        ///   - nlist: Number of clusters (centroids)
        ///   - nprobe: Number of clusters to search at query time
        ///
        /// Recommended settings:
        /// - nlist: sqrt(n) to 4*sqrt(n) where n = expected vector count
        /// - nprobe: 1-20% of nlist for good recall/speed tradeoff
        case ivf(nlist: Int, nprobe: Int)
    }

    // MARK: - Initialization

    /// Create an index configuration.
    /// - Parameters:
    ///   - dimension: Vector dimension (must be > 0)
    ///   - metric: Distance metric (default: euclidean)
    ///   - capacity: Initial capacity (default: 10,000)
    ///   - indexType: Index type (default: flat)
    public init(
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        capacity: Int = 10_000,
        indexType: IndexType = .flat
    ) {
        self.dimension = dimension
        self.metric = metric
        self.capacity = capacity
        self.indexType = indexType
    }

    // MARK: - Factory Methods

    /// Create a flat index configuration.
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - metric: Distance metric (default: euclidean)
    ///   - capacity: Initial capacity (default: 10,000)
    /// - Returns: Configuration for a flat index
    public static func flat(
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        capacity: Int = 10_000
    ) -> IndexConfiguration {
        IndexConfiguration(
            dimension: dimension,
            metric: metric,
            capacity: capacity,
            indexType: .flat
        )
    }

    /// Create an IVF index configuration.
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - nlist: Number of clusters
    ///   - nprobe: Number of clusters to search
    ///   - metric: Distance metric (default: euclidean)
    ///   - capacity: Initial capacity (default: 100,000)
    /// - Returns: Configuration for an IVF index
    public static func ivf(
        dimension: Int,
        nlist: Int,
        nprobe: Int,
        metric: SupportedDistanceMetric = .euclidean,
        capacity: Int = 100_000
    ) -> IndexConfiguration {
        IndexConfiguration(
            dimension: dimension,
            metric: metric,
            capacity: capacity,
            indexType: .ivf(nlist: nlist, nprobe: nprobe)
        )
    }

    // MARK: - Validation

    /// Validate the configuration.
    /// - Throws: `IndexAccelerationError.invalidConfiguration` if invalid
    public func validate() throws {
        guard dimension > 0 else {
            throw IndexAccelerationError.invalidConfiguration(
                parameter: "dimension",
                reason: "must be positive, got \(dimension)"
            )
        }

        guard capacity > 0 else {
            throw IndexAccelerationError.invalidConfiguration(
                parameter: "capacity",
                reason: "must be positive, got \(capacity)"
            )
        }

        switch indexType {
        case .flat:
            break // No additional validation needed

        case .ivf(let nlist, let nprobe):
            guard nlist > 0 else {
                throw IndexAccelerationError.invalidConfiguration(
                    parameter: "nlist",
                    reason: "must be positive, got \(nlist)"
                )
            }
            guard nprobe > 0 else {
                throw IndexAccelerationError.invalidConfiguration(
                    parameter: "nprobe",
                    reason: "must be positive, got \(nprobe)"
                )
            }
            guard nprobe <= nlist else {
                throw IndexAccelerationError.invalidConfiguration(
                    parameter: "nprobe",
                    reason: "must be <= nlist (\(nlist)), got \(nprobe)"
                )
            }
        }
    }

    // MARK: - Computed Properties

    /// Whether this is a flat index.
    public var isFlat: Bool {
        if case .flat = indexType { return true }
        return false
    }

    /// Whether this is an IVF index.
    public var isIVF: Bool {
        if case .ivf = indexType { return true }
        return false
    }

    /// Estimated memory usage per vector in bytes.
    public var bytesPerVector: Int {
        dimension * MemoryLayout<Float>.size
    }

    /// Estimated total memory usage for vectors at capacity.
    public var estimatedVectorMemoryBytes: Int {
        capacity * bytesPerVector
    }
}

// MARK: - CustomStringConvertible

extension IndexConfiguration: CustomStringConvertible {
    public var description: String {
        switch indexType {
        case .flat:
            return "IndexConfiguration(flat, dim=\(dimension), metric=\(metric), capacity=\(capacity))"
        case .ivf(let nlist, let nprobe):
            return "IndexConfiguration(ivf, dim=\(dimension), metric=\(metric), capacity=\(capacity), nlist=\(nlist), nprobe=\(nprobe))"
        }
    }
}
