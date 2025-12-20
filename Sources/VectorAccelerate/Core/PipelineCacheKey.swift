//
//  PipelineCacheKey.swift
//  VectorAccelerate
//
//  Unique identifiers for compiled Metal pipelines
//

import Foundation
@preconcurrency import Metal

// MARK: - Pipeline Cache Key

/// Unique identifier for a compiled compute pipeline
///
/// Cache keys encode all parameters that affect compilation:
/// - Operation type (distance metric, quantization, etc.)
/// - Dimension (optimized kernels per dimension)
/// - Data type
/// - Feature flags
///
/// Example:
/// ```swift
/// let key = PipelineCacheKey.distance("l2Distance", dimension: 384)
/// let pipeline = try await cache.getPipeline(for: key)
/// ```
public struct PipelineCacheKey: Hashable, Codable, Sendable {
    /// Operation type (l2Distance, cosine, topK, etc.)
    public let operation: String

    /// Target dimension (384, 512, 768, 1536, or 0 for generic)
    public let dimension: Int

    /// Data type for computation
    public let dataType: DataType

    /// Quantization mode if applicable
    public let quantizationMode: QuantizationMode?

    /// Feature flags for specialized variants
    public let features: FeatureFlags

    // MARK: - Data Types

    /// Supported data types for pipeline compilation
    public enum DataType: String, Codable, Sendable {
        case float32
        case float16
        case int8
        case uint8
        case bfloat16
    }

    /// Quantization modes for compressed vectors
    public enum QuantizationMode: String, Codable, Sendable {
        case scalar4
        case scalar8
        case binary
        case productQuantization
    }

    /// Feature flags for specialized pipeline variants
    public struct FeatureFlags: OptionSet, Hashable, Codable, Sendable {
        public let rawValue: UInt32

        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        /// Enable fused L2 normalization before distance
        public static let fusedNormalize = FeatureFlags(rawValue: 1 << 0)

        /// Enable fused top-K selection after distance
        public static let fusedTopK = FeatureFlags(rawValue: 1 << 1)

        /// Use SIMD group matrix operations (Apple GPU family 7+)
        public static let simdgroupMatrix = FeatureFlags(rawValue: 1 << 2)

        /// Use ML tensor operations (Metal 4)
        public static let mlTensor = FeatureFlags(rawValue: 1 << 3)

        /// Enable in-place computation
        public static let inPlace = FeatureFlags(rawValue: 1 << 4)

        /// Use half-precision intermediate values
        public static let halfPrecisionIntermediate = FeatureFlags(rawValue: 1 << 5)

        /// No features
        public static let none: FeatureFlags = []
    }

    // MARK: - Initialization

    public init(
        operation: String,
        dimension: Int = 0,
        dataType: DataType = .float32,
        quantizationMode: QuantizationMode? = nil,
        features: FeatureFlags = []
    ) {
        self.operation = operation
        self.dimension = dimension
        self.dataType = dataType
        self.quantizationMode = quantizationMode
        self.features = features
    }

    // MARK: - Factory Methods

    /// Create key for standard distance kernel
    public static func distance(_ operation: String, dimension: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: operation,
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for L2 distance with specific dimension
    public static func l2Distance(dimension: Int) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "l2Distance",
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for cosine similarity
    public static func cosineSimilarity(dimension: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "cosineSimilarity",
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for dot product
    public static func dotProduct(dimension: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "dotProduct",
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for top-K selection
    public static func topK(k: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "topK",
            dimension: k,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for quantized operation
    public static func quantized(_ operation: String, mode: QuantizationMode) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: operation,
            dimension: 0,
            dataType: .uint8,
            quantizationMode: mode,
            features: []
        )
    }

    /// Create key for fused distance + top-K pipeline
    public static func fusedDistanceTopK(metric: String, dimension: Int, k: Int) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "fused_\(metric)_topk",
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: [.fusedTopK]
        )
    }

    /// Create key for batch operation
    public static func batch(_ operation: String, dimension: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "batch_\(operation)",
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    // MARK: - Function Name Resolution

    /// Get the Metal function name for this cache key
    public var functionName: String {
        // Dimension-specific variants
        let dimensionSuffix: String
        switch dimension {
        case 384:
            dimensionSuffix = "_384"
        case 512:
            dimensionSuffix = "_512"
        case 768:
            dimensionSuffix = "_768"
        case 1536:
            dimensionSuffix = "_1536"
        case 0, _:
            dimensionSuffix = ""
        }

        // Operation mapping
        let baseName: String
        switch operation {
        case "l2Distance":
            baseName = "l2_distance\(dimensionSuffix)_kernel"
        case "cosineSimilarity":
            baseName = "cosine_similarity\(dimensionSuffix)_kernel"
        case "dotProduct":
            baseName = "dot_product\(dimensionSuffix)_kernel"
        case "topK":
            baseName = "top_k_selection"
        case "euclideanDistance":
            baseName = "euclideanDistance"
        case "cosineDistance":
            baseName = "cosineDistance"
        default:
            // Handle fused operations
            if operation.hasPrefix("fused_") {
                baseName = operation.replacingOccurrences(of: "_", with: "")
            } else if operation.hasPrefix("batch_") && !operation.hasSuffix("_kernel") {
                // Only transform abbreviated batch operation names (e.g., "batch_l2")
                // Complete function names (e.g., "batch_projection_kernel") pass through unchanged
                baseName = "batch" + operation.dropFirst(6).capitalized
            } else {
                baseName = operation
            }
        }

        // Quantization suffix
        if let qMode = quantizationMode {
            switch qMode {
            case .scalar4:
                return baseName + "_q4"
            case .scalar8:
                return baseName + "_q8"
            case .binary:
                return baseName + "_binary"
            case .productQuantization:
                return baseName + "_pq"
            }
        }

        return baseName
    }

    /// Unique string representation for caching
    public var cacheString: String {
        var components = [operation]

        if dimension > 0 {
            components.append("d\(dimension)")
        }

        components.append(dataType.rawValue)

        if let qMode = quantizationMode {
            components.append(qMode.rawValue)
        }

        if !features.isEmpty {
            components.append("f\(features.rawValue)")
        }

        return components.joined(separator: "-")
    }
}

// MARK: - Common Cache Keys

public extension PipelineCacheKey {
    /// Common keys for pre-warming cache
    static var commonKeys: [PipelineCacheKey] {
        [
            // L2 Distance variants
            .l2Distance(dimension: 384),
            .l2Distance(dimension: 512),
            .l2Distance(dimension: 768),
            .l2Distance(dimension: 1536),
            .l2Distance(dimension: 0),  // Generic

            // Cosine similarity
            .cosineSimilarity(dimension: 0),
            .cosineSimilarity(dimension: 384),
            .cosineSimilarity(dimension: 768),

            // Dot product
            .dotProduct(dimension: 0),

            // Top-K
            .topK(k: 0),

            // Batch operations
            .batch("euclideanDistance", dimension: 0),

            // Basic operations (from embedded shaders)
            .distance("euclideanDistance", dimension: 0),
            .distance("cosineDistance", dimension: 0),
            .distance("dotProduct", dimension: 0),
        ]
    }

    /// Keys for embedding model dimensions (MiniLM, BERT, GPT)
    static var embeddingModelKeys: [PipelineCacheKey] {
        [
            // MiniLM / Sentence-BERT (384)
            .l2Distance(dimension: 384),
            .cosineSimilarity(dimension: 384),

            // BERT / DistilBERT (768)
            .l2Distance(dimension: 768),
            .cosineSimilarity(dimension: 768),

            // OpenAI ada-002 (1536)
            .l2Distance(dimension: 1536),
            .cosineSimilarity(dimension: 1536),

            // All-MPNet (768)
            .l2Distance(dimension: 768),
        ]
    }
}

// MARK: - Cache Key Set

/// Set of cache keys for batch operations
public struct PipelineCacheKeySet: Codable, Sendable {
    public let keys: [PipelineCacheKey]
    public let version: String
    public let createdAt: Date

    public init(keys: [PipelineCacheKey], version: String = "1.0.0") {
        self.keys = keys
        self.version = version
        self.createdAt = Date()
    }

    /// Standard key set for VectorAccelerate
    public static var standard: PipelineCacheKeySet {
        PipelineCacheKeySet(keys: PipelineCacheKey.commonKeys)
    }

    /// Key set for embedding workloads
    public static var embeddings: PipelineCacheKeySet {
        PipelineCacheKeySet(keys: PipelineCacheKey.embeddingModelKeys)
    }
}
