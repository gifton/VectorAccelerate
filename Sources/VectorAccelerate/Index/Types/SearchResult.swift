//
//  SearchResult.swift
//  VectorAccelerate
//
//  GPU-native search result containing a vector handle and distance.
//

import Foundation
import VectorCore

// MARK: - Search Result Typealias

/// Result from a vector search operation.
///
/// Contains the handle of a matching vector and its distance to the query.
///
/// This is a typealias for `SearchResult<VectorHandle>` from VectorCore.
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
/// ## Usage
/// ```swift
/// let results = try await index.search(query: query, k: 10)
/// for result in results {
///     print("Handle: \(result.id), Distance: \(result.distance)")
/// }
/// ```
public typealias IndexSearchResult = SearchResult<VectorHandle>

// MARK: - Convenience Extensions

public extension SearchResult where ID == VectorHandle {
    /// Alias for `id` to maintain compatibility with older versions.
    @available(*, deprecated, renamed: "id")
    var handle: VectorHandle { id }

    /// The actual euclidean distance (sqrt of L2²).
    ///
    /// Only meaningful when the search uses `.euclidean` metric.
    @inlinable
    var euclideanDistance: Float {
        sqrt(distance)
    }

    /// Whether this result is valid (not a padding result).
    @inlinable
    var isValid: Bool {
        id.isValid && distance.isFinite
    }
}
