//
//  IndexConfiguration.swift
//  VectorAccelerate
//
//  Configuration for creating an AcceleratedVectorIndex.
//

import Foundation
import VectorCore

// MARK: - Vector Quantization

/// Vector quantization method for memory reduction in IVF indexes.
///
/// Quantization reduces memory usage by storing vectors in lower precision formats.
/// Trade-off: Lower precision means some recall loss but significant memory savings.
///
/// ## Compression Ratios
/// - `.none`: 1:1 (full float32 precision)
/// - `.sq8`: 4:1 (float32 → int8, symmetric)
/// - `.sq8Asymmetric`: 4:1 (float32 → int8, with zero-point offset)
/// - `.sq4`: 8:1 (float32 → int4, packed)
///
/// ## Typical Recall Impact
/// - `.sq8`: < 5% recall loss
/// - `.sq8Asymmetric`: < 3% recall loss (better for non-centered data)
/// - `.sq4`: < 10% recall loss
public enum VectorQuantization: Sendable, Equatable {
    /// No quantization - full float32 precision (default)
    case none

    /// INT8 symmetric quantization (4:1 compression)
    /// Uses scale only: q = round(x / scale)
    case sq8

    /// INT8 asymmetric quantization (4:1 compression)
    /// Uses scale and zero-point: q = round(x / scale) + zero_point
    /// Better for data not centered around zero.
    case sq8Asymmetric

    /// INT4 quantization (8:1 compression)
    /// Maximum compression, higher recall loss.
    case sq4

    /// Compression ratio achieved by this quantization method.
    public var compressionRatio: Float {
        switch self {
        case .none: return 1.0
        case .sq8, .sq8Asymmetric: return 4.0
        case .sq4: return 8.0
        }
    }

    /// Bytes per element for this quantization method.
    public var bytesPerElement: Float {
        switch self {
        case .none: return 4.0
        case .sq8, .sq8Asymmetric: return 1.0
        case .sq4: return 0.5
        }
    }
}

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

    /// Vector quantization method for memory reduction.
    /// Only applicable to IVF indexes. Ignored for flat indexes.
    public let quantization: VectorQuantization

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
        ///   - minTrainingVectors: Minimum vectors before auto-training triggers
        ///     (nil uses default: max(nlist * 10, 1000))
        ///
        /// Recommended settings:
        /// - nlist: sqrt(n) to 4*sqrt(n) where n = expected vector count
        /// - nprobe: 1-20% of nlist for good recall/speed tradeoff
        case ivf(nlist: Int, nprobe: Int, minTrainingVectors: Int? = nil)
    }

    // MARK: - Initialization

    /// Create an index configuration.
    /// - Parameters:
    ///   - dimension: Vector dimension (must be > 0)
    ///   - metric: Distance metric (default: euclidean)
    ///   - capacity: Initial capacity (default: 10,000)
    ///   - indexType: Index type (default: flat)
    ///   - quantization: Vector quantization method (default: none)
    public init(
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        capacity: Int = 10_000,
        indexType: IndexType = .flat,
        quantization: VectorQuantization = .none
    ) {
        self.dimension = dimension
        self.metric = metric
        self.capacity = capacity
        self.indexType = indexType
        self.quantization = quantization
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
    ///   - minTrainingVectors: Minimum vectors before auto-training triggers
    ///     (nil uses default: max(nlist * 10, 1000))
    ///   - quantization: Vector quantization method (default: none)
    /// - Returns: Configuration for an IVF index
    public static func ivf(
        dimension: Int,
        nlist: Int,
        nprobe: Int,
        metric: SupportedDistanceMetric = .euclidean,
        capacity: Int = 100_000,
        minTrainingVectors: Int? = nil,
        quantization: VectorQuantization = .none
    ) -> IndexConfiguration {
        IndexConfiguration(
            dimension: dimension,
            metric: metric,
            capacity: capacity,
            indexType: .ivf(nlist: nlist, nprobe: nprobe, minTrainingVectors: minTrainingVectors),
            quantization: quantization
        )
    }

    // MARK: - Validation

    /// Validate the configuration.
    /// - Throws: `IndexError.invalidConfiguration` if invalid
    public func validate() throws {
        guard dimension > 0 else {
            throw IndexError.invalidConfiguration(
                parameter: "dimension",
                reason: "must be positive, got \(dimension)"
            )
        }

        guard capacity > 0 else {
            throw IndexError.invalidConfiguration(
                parameter: "capacity",
                reason: "must be positive, got \(capacity)"
            )
        }

        switch indexType {
        case .flat:
            break // No additional validation needed

        case .ivf(let nlist, let nprobe, let minTrainingVectors):
            guard nlist > 0 else {
                throw IndexError.invalidConfiguration(
                    parameter: "nlist",
                    reason: "must be positive, got \(nlist)"
                )
            }
            guard nprobe > 0 else {
                throw IndexError.invalidConfiguration(
                    parameter: "nprobe",
                    reason: "must be positive, got \(nprobe)"
                )
            }
            guard nprobe <= nlist else {
                throw IndexError.invalidConfiguration(
                    parameter: "nprobe",
                    reason: "must be <= nlist (\(nlist)), got \(nprobe)"
                )
            }
            if let minVecs = minTrainingVectors, minVecs < nlist {
                throw IndexError.invalidConfiguration(
                    parameter: "minTrainingVectors",
                    reason: "must be >= nlist (\(nlist)), got \(minVecs)"
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
        if case .ivf(_, _, _) = indexType { return true }
        return false
    }

    /// Estimated memory usage per vector in bytes.
    /// Accounts for quantization when enabled.
    public var bytesPerVector: Int {
        if isIVF && quantization != .none {
            return Int(ceil(Float(dimension) * quantization.bytesPerElement))
        }
        return dimension * MemoryLayout<Float>.size
    }

    /// Estimated total memory usage for vectors at capacity.
    public var estimatedVectorMemoryBytes: Int {
        capacity * bytesPerVector
    }

    /// Whether quantization is enabled for this index.
    public var isQuantized: Bool {
        quantization != .none
    }
}

// MARK: - CustomStringConvertible

extension IndexConfiguration: CustomStringConvertible {
    public var description: String {
        switch indexType {
        case .flat:
            return "IndexConfiguration(flat, dim=\(dimension), metric=\(metric), capacity=\(capacity))"
        case .ivf(let nlist, let nprobe, let minTrainingVectors):
            var desc = "IndexConfiguration(ivf, dim=\(dimension), metric=\(metric), capacity=\(capacity), nlist=\(nlist), nprobe=\(nprobe)"
            if let minVecs = minTrainingVectors {
                desc += ", minTrainingVectors=\(minVecs)"
            }
            if quantization != .none {
                desc += ", quantization=\(quantization)"
            }
            desc += ")"
            return desc
        }
    }
}

// MARK: - Adaptive Parameter Selection

extension IndexConfiguration {

    /// Recommend optimal nlist based on expected dataset size.
    ///
    /// Uses the square root heuristic: nlist ≈ sqrt(N)
    /// - Clamped to [8, 4096] for practical limits
    /// - sqrt(N) provides good balance between:
    ///   - Too few clusters → large lists, slow search
    ///   - Too many clusters → poor recall, training instability
    ///
    /// - Parameter datasetSize: Expected number of vectors
    /// - Returns: Recommended nlist value
    public static func recommendedNlist(for datasetSize: Int) -> Int {
        guard datasetSize > 0 else { return 8 }
        let sqrtN = Int(sqrt(Double(datasetSize)))
        // Clamp to reasonable range
        // Minimum 8 for small datasets
        // Maximum 4096 to limit centroid computation
        return max(8, min(sqrtN, 4096))
    }

    /// Recommend optimal nprobe based on nlist and target recall.
    ///
    /// - Parameters:
    ///   - nlist: Number of clusters
    ///   - targetRecall: Target recall percentage (default 0.90 for 90%)
    /// - Returns: Recommended nprobe value
    ///
    /// Guidelines:
    /// - nprobe = 1: ~30% recall (very fast)
    /// - nprobe = nlist/10: ~80% recall (good tradeoff)
    /// - nprobe = nlist/4: ~95% recall (high recall)
    /// - nprobe = nlist: 100% recall (exhaustive, slow)
    public static func recommendedNprobe(
        for nlist: Int,
        targetRecall: Float = 0.90
    ) -> Int {
        guard nlist > 0 else { return 1 }

        // Approximate relationship: recall ≈ 1 - e^(-k * nprobe/nlist)
        // For 90% recall, nprobe ≈ nlist * 0.23
        // For 95% recall, nprobe ≈ nlist * 0.30
        // For 99% recall, nprobe ≈ nlist * 0.46

        let fraction: Float
        switch targetRecall {
        case ..<0.8:  fraction = 0.10
        case ..<0.9:  fraction = 0.15
        case ..<0.95: fraction = 0.23
        case ..<0.99: fraction = 0.30
        default:      fraction = 0.50
        }

        let recommended = Int(ceil(Float(nlist) * fraction))
        return max(1, min(recommended, nlist))
    }

    /// Create an IVF index configuration with automatically tuned parameters.
    ///
    /// This is the **recommended** way to create IVF indexes when you don't
    /// have specific requirements. Parameters are chosen based on established
    /// heuristics from the ANN-benchmarks community.
    ///
    /// ## Example
    /// ```swift
    /// // For a 100K vector dataset with 90% target recall
    /// let config = IndexConfiguration.ivfAuto(
    ///     dimension: 768,
    ///     expectedSize: 100_000,
    ///     targetRecall: 0.90
    /// )
    /// // Results: nlist=316, nprobe=73
    /// ```
    ///
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - expectedSize: Expected dataset size
    ///   - targetRecall: Target recall (default 0.90)
    ///   - metric: Distance metric
    ///   - quantization: Vector quantization method (default: none)
    /// - Returns: Optimally configured IVF index
    public static func ivfAuto(
        dimension: Int,
        expectedSize: Int,
        targetRecall: Float = 0.90,
        metric: SupportedDistanceMetric = .euclidean,
        quantization: VectorQuantization = .none
    ) -> IndexConfiguration {
        let nlist = recommendedNlist(for: expectedSize)
        let nprobe = recommendedNprobe(for: nlist, targetRecall: targetRecall)

        return IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            metric: metric,
            capacity: expectedSize,
            quantization: quantization
        )
    }
}
