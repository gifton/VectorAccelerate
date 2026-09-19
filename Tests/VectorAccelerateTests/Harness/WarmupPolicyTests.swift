import XCTest
import VectorTestKit

/// Spec §5.2: stop when ≥ minimum ran and CoV of the last `window` wall times < cov, or at maximum.
final class WarmupPolicyTests: XCTestCase {
    func testStableSeriesStopsAtMinimum() {
        let p = WarmupPolicy.standard
        XCTAssertFalse(p.shouldStop(after: [100, 100]))
        XCTAssertTrue(p.shouldStop(after: [100, 100, 100]))
    }

    func testNoisySeriesRunsToMaximum() {
        let p = WarmupPolicy.standard
        let noisy = (0..<49).map { Double(100 + ($0 % 2) * 50) }   // CoV ≈ 0.2
        XCTAssertFalse(p.shouldStop(after: noisy))
        XCTAssertTrue(p.shouldStop(after: noisy + [100]))            // 50 == maximum
    }

    func testWindowOnlyLooksAtTail() {
        let p = WarmupPolicy.standard
        let settled = [1000, 10, 1000, 10] + Array(repeating: 100.0, count: 8)
        XCTAssertTrue(p.shouldStop(after: settled))
    }

    func testStandardValues() {
        XCTAssertEqual(WarmupPolicy.standard.window, 8); XCTAssertEqual(WarmupPolicy.standard.cov, 0.05)
        XCTAssertEqual(WarmupPolicy.standard.minimum, 3); XCTAssertEqual(WarmupPolicy.standard.maximum, 50)
    }
}
