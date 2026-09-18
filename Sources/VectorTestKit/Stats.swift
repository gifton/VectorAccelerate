import Foundation

/// Descriptive statistics of one timing series (nanoseconds, or any positive quantity).
public struct Summary: Codable, Sendable, Equatable {
    public var median: Double
    public var p90: Double
    public var p99: Double
    public var min: Double
    public var max: Double
    public var mean: Double
    /// Sample standard deviation divided by the mean; 0 when count < 2 or mean == 0.
    public var rsd: Double
    /// 95% percentile-bootstrap interval of the median, `[lo, hi]`.
    public var ci95: [Double]
    public var count: Int

    public init(median: Double, p90: Double, p99: Double, min: Double, max: Double, mean: Double, rsd: Double, ci95: [Double], count: Int) {
        self.median = median; self.p90 = p90; self.p99 = p99; self.min = min; self.max = max
        self.mean = mean; self.rsd = rsd; self.ci95 = ci95; self.count = count
    }
}

public enum Stats {
    /// Linear interpolation between order statistics: rank = p·(n−1). Replaces the floor-index method
    /// the old IndexBenchmarkHarness.LatencyStats used.
    public static func percentile(sorted: [Double], p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        guard sorted.count > 1 else { return sorted[0] }
        let rank = Swift.min(Swift.max(p, 0), 1) * Double(sorted.count - 1)
        let lo = Int(rank.rounded(.down))
        let hi = Int(rank.rounded(.up))
        if lo == hi { return sorted[lo] }
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - Double(lo))
    }

    public static func summarize(_ values: [Double], bootstrapSeed: UInt64, resamples: Int = 1_000) -> Summary {
        let n = values.count
        guard n > 0 else { return Summary(median: 0, p90: 0, p99: 0, min: 0, max: 0, mean: 0, rsd: 0, ci95: [0, 0], count: 0) }
        let sorted = values.sorted()
        let mean = values.reduce(0, +) / Double(n)
        let rsd: Double
        if n >= 2, mean != 0 {
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n - 1)
            rsd = variance.squareRoot() / mean
        } else { rsd = 0 }
        let median = percentile(sorted: sorted, p: 0.5)
        var rng = SeededRNG(seed: bootstrapSeed, stream: 0)
        var medians: [Double] = []
        medians.reserveCapacity(resamples)
        var scratch = [Double](repeating: 0, count: n)
        for _ in 0..<resamples {
            for i in 0..<n { scratch[i] = values[rng.nextInt(bound: n)] }
            medians.append(percentile(sorted: scratch.sorted(), p: 0.5))
        }
        let ms = medians.sorted()
        return Summary(
            median: median, p90: percentile(sorted: sorted, p: 0.9), p99: percentile(sorted: sorted, p: 0.99),
            min: sorted[0], max: sorted[n - 1], mean: mean, rsd: rsd,
            ci95: [percentile(sorted: ms, p: 0.025), percentile(sorted: ms, p: 0.975)], count: n)
    }
}

/// Dead-code-elimination barrier for benchmarked expressions whose results are otherwise unused.
@inline(never)
public func blackHole<T>(_ value: T) {
    withExtendedLifetime(value) {}
}
