//
//  FallbackProvider.swift
//  VectorAccelerate
//
//  CPU fallback implementations for when Metal is unavailable
//

import Foundation
import Accelerate
import VectorCore

/// Provides CPU fallback implementations for Metal operations.
///
/// When Metal is unavailable (initialization failed, thermal throttling, etc.),
/// `FallbackProvider` offers CPU-based alternatives using Apple's Accelerate framework.
///
/// ## Performance Expectations
///
/// CPU fallbacks are significantly slower than GPU for large batches:
/// - Distance computation: ~10-50x slower for 1000+ vectors
/// - Normalization: ~5-10x slower for batch operations
/// - Top-K selection: ~2-5x slower depending on K
///
/// For single operations or small batches (<100 vectors), CPU performance
/// is often acceptable and avoids GPU overhead.
///
/// ## Thread Safety
///
/// `FallbackProvider` is fully `Sendable` and all methods are thread-safe.
/// Operations are stateless and can be called concurrently from any thread.
///
/// ## Example Usage
///
/// ```swift
/// let fallback = metalSubsystem.fallback
///
/// // Single distance
/// let dist = fallback.l2Distance(from: queryVector, to: targetVector)
///
/// // Batch distances
/// let distances = fallback.batchL2Distance(from: query, to: candidates)
///
/// // With metric selection
/// let result = fallback.distance(from: a, to: b, metric: .cosine)
/// ```
public struct FallbackProvider: Sendable {

    /// Create a new fallback provider.
    public init() {}

    // MARK: - L2 Distance Operations

    /// Compute L2 (Euclidean) distance between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Euclidean distance, or `.infinity` if dimensions don't match
    ///
    /// - Complexity: O(n) where n is vector dimension
    public func l2Distance(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }

        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return sqrt(result)
    }

    /// Compute squared L2 distance (avoids sqrt for comparison purposes).
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Squared Euclidean distance
    public func l2DistanceSquared(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }

        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Compute batch L2 distances from query to all candidates.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors to compare against
    /// - Returns: Array of distances, same length as candidates
    ///
    /// - Complexity: O(n * m) where n is candidate count and m is dimension
    public func batchL2Distance(from query: [Float], to candidates: [[Float]]) -> [Float] {
        candidates.map { l2Distance(from: query, to: $0) }
    }

    // MARK: - Cosine Similarity Operations

    /// Compute cosine similarity between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Cosine similarity in range [-1, 1], or 0 if invalid
    ///
    /// - Complexity: O(n) where n is vector dimension
    public func cosineSimilarity(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))

        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dotProduct / denom : 0
    }

    /// Compute cosine distance (1 - similarity) between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Cosine distance in range [0, 2]
    public func cosineDistance(from a: [Float], to b: [Float]) -> Float {
        1.0 - cosineSimilarity(from: a, to: b)
    }

    /// Compute batch cosine similarities from query to all candidates.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors
    /// - Returns: Array of similarities
    public func batchCosineSimilarity(from query: [Float], to candidates: [[Float]]) -> [Float] {
        candidates.map { cosineSimilarity(from: query, to: $0) }
    }

    // MARK: - Dot Product Operations

    /// Compute dot product between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Dot product value
    public func dotProduct(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(min(a.count, b.count)))
        return result
    }

    /// Compute batch dot products from query to all candidates.
    public func batchDotProduct(from query: [Float], to candidates: [[Float]]) -> [Float] {
        candidates.map { dotProduct(from: query, to: $0) }
    }

    // MARK: - Manhattan Distance

    /// Compute Manhattan (L1) distance between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Sum of absolute differences
    public func manhattanDistance(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).reduce(0) { $0 + abs($1.0 - $1.1) }
    }

    // MARK: - Vector Operations

    /// Normalize a vector to unit length.
    ///
    /// - Parameter vector: Input vector
    /// - Returns: Unit vector, or original if zero-length
    ///
    /// - Complexity: O(n) where n is vector dimension
    public func normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return vector }

        var norm: Float = 0
        vDSP_dotpr(vector, 1, vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)

        guard norm > 0 else { return vector }

        var result = [Float](repeating: 0, count: vector.count)
        var divisor = norm
        vDSP_vsdiv(vector, 1, &divisor, &result, 1, vDSP_Length(vector.count))
        return result
    }

    /// Normalize a batch of vectors.
    ///
    /// - Parameter vectors: Array of vectors to normalize
    /// - Returns: Array of normalized vectors
    ///
    /// - Complexity: O(n * m) where n is vector count and m is dimension
    public func normalizeBatch(_ vectors: [[Float]]) -> [[Float]] {
        vectors.map { normalize($0) }
    }

    /// Compute the L2 norm of a vector.
    ///
    /// - Parameter vector: Input vector
    /// - Returns: L2 norm (magnitude)
    public func l2Norm(_ vector: [Float]) -> Float {
        guard !vector.isEmpty else { return 0 }

        var norm: Float = 0
        vDSP_dotpr(vector, 1, vector, 1, &norm, vDSP_Length(vector.count))
        return sqrt(norm)
    }

    // MARK: - Top-K Selection

    /// Select top K indices by smallest distance (nearest neighbors).
    ///
    /// - Parameters:
    ///   - distances: Array of distances
    ///   - k: Number of results to return
    /// - Returns: Array of (index, distance) tuples, sorted by ascending distance
    ///
    /// - Complexity: O(n log n) for full sort, could be O(n log k) with partial sort
    public func topKByDistance(_ distances: [Float], k: Int) -> [(index: Int, distance: Float)] {
        guard k > 0 else { return [] }

        let indexed = distances.enumerated().map { ($0.offset, $0.element) }
        return Array(indexed.sorted { $0.1 < $1.1 }.prefix(k))
    }

    /// Select top K indices by largest similarity.
    ///
    /// - Parameters:
    ///   - similarities: Array of similarity values
    ///   - k: Number of results to return
    /// - Returns: Array of (index, similarity) tuples, sorted by descending similarity
    public func topKBySimilarity(_ similarities: [Float], k: Int) -> [(index: Int, similarity: Float)] {
        guard k > 0 else { return [] }

        let indexed = similarities.enumerated().map { ($0.offset, $0.element) }
        return Array(indexed.sorted { $0.1 > $1.1 }.prefix(k))
    }

    /// Combined search: compute distances and return top K.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Candidate vectors to search
    ///   - k: Number of results
    ///   - metric: Distance metric to use
    /// - Returns: Top K results with indices and distances
    public func search(
        query: [Float],
        in candidates: [[Float]],
        k: Int,
        metric: SupportedDistanceMetric = .euclidean
    ) -> [(index: Int, distance: Float)] {
        let distances = batchDistance(from: query, to: candidates, metric: metric)
        return topKByDistance(distances, k: k)
    }

    // MARK: - Generic Metric Support

    /// Compute Chebyshev (L-infinity) distance between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Maximum absolute difference
    public func chebyshevDistance(from a: [Float], to b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    /// Distance computation using specified metric.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    ///   - metric: Distance metric from VectorCore
    /// - Returns: Distance value (interpretation depends on metric)
    public func distance(
        from a: [Float],
        to b: [Float],
        metric: SupportedDistanceMetric
    ) -> Float {
        switch metric {
        case .euclidean:
            return l2Distance(from: a, to: b)
        case .cosine:
            return 1.0 - cosineSimilarity(from: a, to: b)
        case .dotProduct:
            // Negate for distance semantics (higher dot product = closer)
            return -dotProduct(from: a, to: b)
        case .manhattan:
            return manhattanDistance(from: a, to: b)
        case .chebyshev:
            return chebyshevDistance(from: a, to: b)
        }
    }

    /// Batch distance computation using specified metric.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Candidate vectors
    ///   - metric: Distance metric
    /// - Returns: Array of distances
    public func batchDistance(
        from query: [Float],
        to candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) -> [Float] {
        candidates.map { distance(from: query, to: $0, metric: metric) }
    }
}

// MARK: - VectorProtocol Support

extension FallbackProvider {

    /// Compute distance between VectorProtocol-conforming types.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    ///   - metric: Distance metric
    /// - Returns: Distance value
    public func distance<V: VectorProtocol>(
        from a: V,
        to b: V,
        metric: SupportedDistanceMetric = .euclidean
    ) -> Float where V.Scalar == Float {
        distance(from: a.toArray(), to: b.toArray(), metric: metric)
    }

    /// Compute batch distances from VectorProtocol query.
    public func batchDistance<V: VectorProtocol>(
        from query: V,
        to candidates: [V],
        metric: SupportedDistanceMetric = .euclidean
    ) -> [Float] where V.Scalar == Float {
        let queryArray = query.toArray()
        return candidates.map { distance(from: queryArray, to: $0.toArray(), metric: metric) }
    }
}
