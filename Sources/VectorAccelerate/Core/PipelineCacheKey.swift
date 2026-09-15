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

    // NOTE: the `.cosineSimilarity(dimension:)` constructor was removed with the
    // CosineSimilarity.metal specialized family (AUDIT-3 Group F): no dispatch path ever used
    // those kernels, so the constructor could only mint keys that resolve to nothing. Live
    // cosine work goes through `cosineDistance` / `batchCosineDistance` / `cosine_similarity`
    // (the pair kernel) / `soa_cosine_distance`.

    /// Create key for the batch dot-product family (`dot_product{_384,_512,_768,_1536}_kernel`).
    ///
    /// The operation string is deliberately NOT "dotProduct": that exact spelling is the
    /// single-pair kernel's literal function name, and `getPipeline(functionName:)` funnels
    /// literal names through this same key type — sharing the string would (and did, AUDIT-3
    /// VA3-031) reroute the engine's single-pair dispatch to the batch kernel.
    public static func dotProduct(dimension: Int = 0) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: "dot_product",
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

        // Operation mapping. Every branch must produce a function name that actually exists in
        // the shader corpus — `ShaderLibraryCompletenessTests.testCommonPipelineKeysResolveToRealFunctions`
        // enforces this for the pre-warm key sets. Before the 2026-08 audit four derivations
        // produced phantom names that existed in no library (AUDIT-2 VA2-006): the generic
        // cosine key ("cosine_similarity_kernel"), "topK" ("top_k_selection"), fused keys
        // (underscore-stripping turned the real `fused_l2_topk` into "fusedl2topk"), and batch
        // keys (Swift's `.capitalized` lowercases the rest of the word:
        // "batchEuclideandistance").
        //
        // A second trap (AUDIT-3 VA3-031): `getPipeline(functionName:)` funnels *literal*
        // kernel names through this switch via `PipelineCacheKey(operation: functionName)`,
        // so any case that rewrites its operation string hijacks the identically-spelled
        // function name. "dotProduct" → "dot_product_kernel" silently sent the engine's
        // single-pair dot product to the batch kernel (whose params read zeros past the
        // 4-byte dimension constant, so nothing was written and callers got whatever bytes
        // the pooled result buffer already held). The keyed batch family therefore uses
        // operation "dot_product", and the three literal single-pair kernel names are
        // identity-mapped below. Do not add a rewriting case whose operation string equals
        // an existing kernel name.
        let baseName: String
        switch operation {
        case "l2Distance":
            // The dimension-specialized l2_distance_{384,512,768,1536}_kernel variants were
            // deleted in AUDIT-3 Group F (no dispatch path ever selected them); every
            // l2Distance key resolves to the general kernel regardless of dimension.
            baseName = "l2_distance_kernel"
        case "dot_product":
            baseName = "dot_product\(dimensionSuffix)_kernel"
        case "topK":
            baseName = "topk_select_batch_kernel"
        case "euclideanDistance", "cosineDistance", "dotProduct":
            // Literal single-pair kernel names (BasicOperations.metal), dispatched by
            // Metal4ComputeEngine through getPipeline(functionName:) — never rewritten.
            baseName = operation
        default:
            if operation.hasPrefix("fused_") {
                // fused_* operations ARE literal kernel names (e.g. `fused_l2_topk`).
                baseName = operation
            } else if operation.hasPrefix("batch_") && !operation.hasSuffix("_kernel") {
                // Abbreviated batch operation names (e.g. "batch_euclideanDistance" →
                // "batchEuclideanDistance"): uppercase ONLY the first character of the tail.
                let tail = operation.dropFirst(6)
                baseName = "batch" + tail.prefix(1).uppercased() + tail.dropFirst()
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
            // L2 Distance (all dimensions resolve to the general kernel; the specialized
            // variants were deleted in AUDIT-3 Group F)
            .l2Distance(dimension: 0),

            // Dot product (generic + the live dimension-specialized variants)
            .dotProduct(dimension: 0),
            .dotProduct(dimension: 384),
            .dotProduct(dimension: 512),
            .dotProduct(dimension: 768),
            .dotProduct(dimension: 1536),

            // Top-K
            .topK(k: 0),

            // Batch operations
            .batch("euclideanDistance", dimension: 0),

            // Basic operations (from embedded shaders)
            .distance("euclideanDistance", dimension: 0),
            .distance("cosineDistance", dimension: 0),
            .distance("dotProduct", dimension: 0),
            
            // KMeans and IVF
            PipelineCacheKey(operation: "fused_l2_topk"),
            PipelineCacheKey(operation: "kmeans_assign_points"),
            PipelineCacheKey(operation: "ivf_distance_with_indirection"),
            PipelineCacheKey(operation: "ivf_build_candidates"),
            PipelineCacheKey(operation: "ivf_list_search")
        ]
    }

    /// Keys for embedding model dimensions (MiniLM, BERT, GPT)
    static var embeddingModelKeys: [PipelineCacheKey] {
        [
            // The dimension-specialized L2/cosine matrix kernels these keys used to warm were
            // deleted in AUDIT-3 Group F (no dispatch path ever selected them). What remains
            // dimension-specialized AND live is the dot-product family, plus the general
            // L2 kernel that now backs every l2Distance key.
            .l2Distance(dimension: 0),
            .dotProduct(dimension: 384),   // MiniLM / Sentence-BERT
            .dotProduct(dimension: 768),   // BERT / DistilBERT / MPNet
            .dotProduct(dimension: 1536),  // OpenAI ada-002
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
