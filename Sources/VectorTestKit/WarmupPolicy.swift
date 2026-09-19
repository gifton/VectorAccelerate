import Foundation

/// Adaptive warm-up rule (spec §5.2). Values are the initial contract; SP1 retunes them from data.
public struct WarmupPolicy: Sendable {
    public var window: Int
    public var cov: Double
    public var minimum: Int
    public var maximum: Int

    public init(window: Int, cov: Double, minimum: Int, maximum: Int) {
        self.window = window; self.cov = cov; self.minimum = minimum; self.maximum = maximum
    }

    public static let standard = WarmupPolicy(window: 8, cov: 0.05, minimum: 3, maximum: 50)

    public func shouldStop(after wallNs: [Double]) -> Bool {
        let n = wallNs.count
        if n >= maximum { return true }
        if n < minimum { return false }
        let tail = Array(wallNs.suffix(window))
        let mean = tail.reduce(0, +) / Double(tail.count)
        guard mean > 0, tail.count >= 2 else { return true }
        let variance = tail.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(tail.count - 1)
        return variance.squareRoot() / mean < cov
    }
}
