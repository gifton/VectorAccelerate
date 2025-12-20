//
//  PipelineRegistry.swift
//  VectorAccelerate
//
//  Pipeline tier categorization for warmup prioritization
//

import Foundation

/// Tier classification for pipeline warmup priority.
///
/// Pipelines are categorized by how frequently they're used in a typical session.
/// This allows the warmup system to prioritize critical operations while deferring
/// rarely-used pipelines to idle time or on-demand compilation.
public enum PipelineTier: String, Sendable, CaseIterable, Codable {
    /// Must be ready before first user operation (e.g., save).
    /// Warmed in Phase B with high priority.
    case critical

    /// Used for secondary features, can be lazily loaded.
    /// Warmed in Phase C during idle time.
    case occasional

    /// May never run in most sessions.
    /// Only compiled on-demand when requested.
    case rare
}

/// Registry of pipelines organized by usage tier.
///
/// `PipelineRegistry` provides tier-based classification of compute pipelines
/// to enable intelligent warmup scheduling. Critical pipelines are warmed first
/// to ensure the app's primary operations (like saving journal entries) are
/// never blocked on pipeline compilation.
///
/// ## Usage
///
/// ```swift
/// let registry = PipelineRegistry.journalingApp
///
/// // Get critical pipelines for Phase B warmup
/// let criticalKeys = registry.criticalKeys
///
/// // Check tier for a specific key
/// let tier = registry.tier(for: .l2Distance(dimension: 384))
/// assert(tier == .critical)
/// ```
///
/// ## Custom Registry
///
/// ```swift
/// let customRegistry = PipelineRegistry(entries: [
///     .critical: [
///         .l2Distance(dimension: 512),
///         .cosineSimilarity(dimension: 512),
///     ],
///     .occasional: [
///         .dotProduct(dimension: 0),
///     ],
///     .rare: []
/// ])
/// ```
public struct PipelineRegistry: Sendable {

    /// Entries organized by tier
    private let entries: [PipelineTier: [PipelineCacheKey]]

    /// Reverse lookup cache for tier queries
    private let tierLookup: [PipelineCacheKey: PipelineTier]

    /// Create a registry with explicit tier assignments.
    ///
    /// - Parameter entries: Dictionary mapping tiers to their pipeline keys
    public init(entries: [PipelineTier: [PipelineCacheKey]]) {
        self.entries = entries

        // Build reverse lookup
        var lookup: [PipelineCacheKey: PipelineTier] = [:]
        for (tier, keys) in entries {
            for key in keys {
                lookup[key] = tier
            }
        }
        self.tierLookup = lookup
    }

    /// Get all keys for a specific tier.
    ///
    /// - Parameter tier: The tier to query
    /// - Returns: Array of pipeline cache keys for that tier (may be empty)
    public func keys(for tier: PipelineTier) -> [PipelineCacheKey] {
        entries[tier] ?? []
    }

    /// Get tier for a specific key.
    ///
    /// - Parameter key: The pipeline cache key to look up
    /// - Returns: The tier this key belongs to, or `.rare` if not registered
    public func tier(for key: PipelineCacheKey) -> PipelineTier {
        tierLookup[key] ?? .rare
    }

    /// All critical pipeline keys.
    ///
    /// These pipelines should be warmed during Phase B (post-first-frame)
    /// to ensure primary operations are never blocked on compilation.
    public var criticalKeys: [PipelineCacheKey] {
        keys(for: .critical)
    }

    /// All occasional pipeline keys.
    ///
    /// These pipelines can be warmed during Phase C (idle time)
    /// when the user is not actively interacting with the app.
    public var occasionalKeys: [PipelineCacheKey] {
        keys(for: .occasional)
    }

    /// All rare pipeline keys.
    ///
    /// These pipelines are typically compiled on-demand only.
    /// They may never be used in most sessions.
    public var rareKeys: [PipelineCacheKey] {
        keys(for: .rare)
    }

    /// Total number of registered pipelines across all tiers.
    public var totalCount: Int {
        entries.values.reduce(0) { $0 + $1.count }
    }

    /// Number of pipelines in a specific tier.
    public func count(for tier: PipelineTier) -> Int {
        entries[tier]?.count ?? 0
    }
}

// MARK: - Default Registries

extension PipelineRegistry {

    /// Default registry using common keys as critical.
    ///
    /// Suitable for development and testing where all common
    /// operations should be immediately available.
    public static let `default` = PipelineRegistry(entries: [
        .critical: PipelineCacheKey.commonKeys,
        .occasional: [],
        .rare: []
    ])

    /// Registry optimized for journaling app workloads.
    ///
    /// **Critical:** Operations that run on every journal entry save
    /// - L2 distance (384) for embedding similarity
    /// - Cosine similarity (384) for semantic matching
    /// - Top-K selection for finding similar entries
    /// - L2 normalization for preprocessing
    ///
    /// **Occasional:** Secondary features
    /// - Higher dimension variants (768, 1536) for advanced models
    /// - Statistics kernels for analytics
    /// - Dot product for alternative similarity
    ///
    /// **Rare:** Specialized operations
    /// - Quantization (int8, binary) for compression
    /// - Matrix operations for ML inference
    /// - Advanced ML kernels (attention, neural quantization)
    public static let journalingApp = PipelineRegistry(entries: [
        .critical: [
            // Core embedding operations (384-dim MiniLM)
            .l2Distance(dimension: 384),
            .cosineSimilarity(dimension: 384),
            .topK(k: 0),  // Generic top-K
            PipelineCacheKey(operation: "l2_normalize"),
        ],
        .occasional: [
            // Higher dimension models (BERT 768, OpenAI 1536)
            .l2Distance(dimension: 768),
            .l2Distance(dimension: 1536),
            .cosineSimilarity(dimension: 768),
            .cosineSimilarity(dimension: 1536),
            // Generic fallbacks
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
            // Alternative distance metrics
            .dotProduct(dimension: 0),
            .dotProduct(dimension: 384),
            // Statistics
            PipelineCacheKey(operation: "compute_statistics"),
        ],
        .rare: [
            // Quantization
            PipelineCacheKey(operation: "scalar_quantize_int8"),
            PipelineCacheKey(operation: "scalar_quantize_int4"),
            PipelineCacheKey(operation: "binary_quantize"),
            // Matrix operations
            PipelineCacheKey(operation: "matrix_multiply"),
            PipelineCacheKey(operation: "matrix_transpose"),
            // ML kernels
            PipelineCacheKey(operation: "attention_similarity"),
            PipelineCacheKey(operation: "neural_quantization"),
            // Batch operations
            .batch("euclideanDistance", dimension: 0),
        ]
    ])

    /// Registry for embedding-focused workloads.
    ///
    /// Prioritizes common embedding model dimensions for
    /// applications that primarily do semantic search.
    public static let embeddingFocused = PipelineRegistry(entries: [
        .critical: PipelineCacheKey.embeddingModelKeys + [
            .topK(k: 0),
            PipelineCacheKey(operation: "l2_normalize"),
        ],
        .occasional: [
            .dotProduct(dimension: 0),
            PipelineCacheKey(operation: "compute_statistics"),
        ],
        .rare: []
    ])

    /// Minimal registry for testing.
    ///
    /// Only includes the absolute minimum pipelines needed
    /// for basic functionality testing.
    public static let minimal = PipelineRegistry(entries: [
        .critical: [
            .l2Distance(dimension: 0),
        ],
        .occasional: [],
        .rare: []
    ])
}

// MARK: - Codable Support

extension PipelineRegistry: Codable {
    enum CodingKeys: String, CodingKey {
        case entries
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let entries = try container.decode([PipelineTier: [PipelineCacheKey]].self, forKey: .entries)
        self.init(entries: entries)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(entries, forKey: .entries)
    }
}
