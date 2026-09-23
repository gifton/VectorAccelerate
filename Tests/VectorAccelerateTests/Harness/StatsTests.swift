import XCTest
import VectorTestKit

/// Percentiles use linear interpolation between order statistics (rank = p·(n−1)); ci95 is a seeded
/// percentile bootstrap of the median with 1,000 resamples (spec §4.7).
final class StatsTests: XCTestCase {
    func testPercentilesLinearInterpolation() {
        let v = (1...10).map(Double.init)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.5), 5.5, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.9), 9.1, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0.99), 9.91, accuracy: 1e-12)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 0), 1)
        XCTAssertEqual(Stats.percentile(sorted: v, p: 1), 10)
        XCTAssertEqual(Stats.percentile(sorted: [7], p: 0.9), 7)
        XCTAssertEqual(Stats.percentile(sorted: [], p: 0.5), 0)
    }

    func testSummaryFields() {
        let s = Stats.summarize([90, 100, 110], bootstrapSeed: 1)
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s.min, 90); XCTAssertEqual(s.max, 110); XCTAssertEqual(s.mean, 100)
        XCTAssertEqual(s.median, 100)
        XCTAssertEqual(s.rsd, 0.1, accuracy: 1e-12)   // sample stddev 10 / mean 100
        XCTAssertEqual(s.ci95.count, 2)
        XCTAssertLessThanOrEqual(s.ci95[0], s.median); XCTAssertGreaterThanOrEqual(s.ci95[1], s.median)
    }

    func testBootstrapIsReproducibleForSeed() {
        let v = (0..<50).map { Double($0 * 7 % 13) }
        XCTAssertEqual(Stats.summarize(v, bootstrapSeed: 42).ci95, Stats.summarize(v, bootstrapSeed: 42).ci95)
        // Interval bounds are discrete-ish, so two particular seeds can legitimately coincide; the seed must
        // move the interval across a handful of seeds (seeds 0–7 yield ≥ 2 distinct intervals on this data).
        let intervals = Set((0..<8).map { Stats.summarize(v, bootstrapSeed: UInt64($0)).ci95 })
        XCTAssertGreaterThan(intervals.count, 1)
    }

    func testSingleAndEmptyInputs() {
        let one = Stats.summarize([5], bootstrapSeed: 0)
        XCTAssertEqual(one.rsd, 0); XCTAssertEqual(one.ci95, [5, 5])
        let none = Stats.summarize([], bootstrapSeed: 0)
        XCTAssertEqual(none.count, 0); XCTAssertEqual(none.median, 0); XCTAssertEqual(none.ci95, [0, 0])
    }

    func testBlackHoleAcceptsAnything() {
        blackHole([1, 2, 3]); blackHole("x"); blackHole(())
    }
}
