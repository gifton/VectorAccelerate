import Foundation

public struct RecallPolicy: Codable, Sendable, Equatable {
    public var k: Int
    public var epsilon: Double
    public init(k: Int, epsilon: Double) { self.k = k; self.epsilon = epsilon }
}

public struct Neighbor: Sendable, Equatable {
    public var id: Int
    public var distance: Double
    public init(id: Int, distance: Double) { self.id = id; self.distance = distance }
}

public enum RecallError: Error, Equatable {
    case insufficientGroundTruth(have: Int, need: Int)
    case invalidK
}

/// Distance-threshold recall@k (spec §4.6). The caller supplies independently recomputed distances for
/// valid, eligible IDs and, for small corpora, passes `k = min(requestedK, eligibleCount)` (never 0).
/// Precondition (unverified here): `groundTruth` is sorted ascending by distance — the k-th distance is
/// read at index k−1. Callers that build ground truth from an exact oracle must sort before calling.
public enum Recall {
    public static func atK(returned: [Neighbor], groundTruth: [Neighbor], policy: RecallPolicy) throws -> Double {
        guard policy.k >= 1 else { throw RecallError.invalidK }
        guard groundTruth.count >= policy.k else {
            throw RecallError.insufficientGroundTruth(have: groundTruth.count, need: policy.k)
        }
        let threshold = (1.0 + policy.epsilon) * groundTruth[policy.k - 1].distance
        var seen = Set<Int>()
        var credited = 0
        for n in returned.prefix(policy.k) {
            guard seen.insert(n.id).inserted else { continue }
            if n.distance <= threshold { credited += 1 }
        }
        return min(1.0, max(0.0, Double(credited) / Double(policy.k)))
    }
}
