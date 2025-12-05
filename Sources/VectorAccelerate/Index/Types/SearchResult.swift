//
//  SearchResult.swift
//  VectorAccelerate
//
//  GPU-native search result containing a vector handle and distance.
//

import Foundation

// MARK: - Search Result

/// Result from a vector search operation.
///
/// Contains the handle of a matching vector and its distance to the query.
///
/// ## Distance Values
///
/// The `distance` field contains the native GPU-computed distance:
///
/// | Metric | Distance Value | Interpretation |
/// |--------|---------------|----------------|
/// | `.euclidean` | L2² (squared) | Lower is closer. Use `sqrt(distance)` for actual euclidean distance. |
/// | `.cosine` | 1 - similarity | Range [0, 2]. 0 = identical, 2 = opposite. |
/// | `.dotProduct` | -dot | Lower (more negative) means higher similarity. |
///
/// ## Performance Note
///
/// For euclidean metric, we return L2² (squared distance) because:
/// 1. GPU computes squared distance natively
/// 2. sqrt() is expensive and unnecessary for ranking
/// 3. Relative ordering is preserved
///
/// If you need actual euclidean distance, use:
/// ```swift
/// let actualDistance = sqrt(result.distance)
/// ```
///
/// ## Usage
/// ```swift
/// let results = try await index.search(query: query, k: 10)
/// for result in results {
///     print("Handle: \(result.handle), Distance: \(result.distance)")
/// }
/// ```
public struct IndexSearchResult: Sendable, Equatable {

    // MARK: - Properties

    /// Handle to the matched vector.
    public let handle: VectorHandle

    /// Distance from query to this vector.
    ///
    /// For euclidean metric, this is L2² (squared distance).
    /// Call `sqrt(distance)` if you need actual euclidean distance.
    public let distance: Float

    // MARK: - Initialization

    /// Create a search result.
    /// - Parameters:
    ///   - handle: Handle to the matched vector
    ///   - distance: Distance from query (interpretation depends on metric)
    @inlinable
    public init(handle: VectorHandle, distance: Float) {
        self.handle = handle
        self.distance = distance
    }

    // MARK: - Convenience

    /// The actual euclidean distance (sqrt of L2²).
    ///
    /// Only meaningful when the index uses `.euclidean` metric.
    /// For other metrics, this value has no semantic meaning.
    @inlinable
    public var euclideanDistance: Float {
        sqrt(distance)
    }

    /// Whether this result is valid (not a padding result).
    @inlinable
    public var isValid: Bool {
        handle.isValid && distance.isFinite
    }
}

// MARK: - CustomStringConvertible

extension IndexSearchResult: CustomStringConvertible {
    public var description: String {
        "IndexSearchResult(handle: \(handle), distance: \(distance))"
    }
}

// MARK: - Comparable by Distance

extension IndexSearchResult: Comparable {
    /// Compare by distance (lower distance = "less than" = closer/better).
    @inlinable
    public static func < (lhs: IndexSearchResult, rhs: IndexSearchResult) -> Bool {
        lhs.distance < rhs.distance
    }
}
