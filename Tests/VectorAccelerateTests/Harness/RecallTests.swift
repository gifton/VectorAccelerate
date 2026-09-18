import XCTest
import VectorTestKit

/// Distance-epsilon recall (spec §4.6 as amended): first K results only, each ID credited once,
/// a result counts if its distance ≤ (1+ε) × ground-truth k-th distance.
final class RecallTests: XCTestCase {
    private let gt = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 3, distance: 3.0), Neighbor(id: 4, distance: 4.0)]

    func testEpsilonFormCreditsTiesThatIDSetFormWouldNot() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 9, distance: 3.0)]
        let r = try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0))
        XCTAssertEqual(r, 1.0)
        let idSet = Set(returned.map(\.id)).intersection(gt.prefix(3).map(\.id)).count
        XCTAssertEqual(idSet, 2, "ID-set recall would be 2/3; epsilon form credits the tie at distance 3.0")
    }

    func testEpsilonWidensThreshold() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 2, distance: 2.0), Neighbor(id: 9, distance: 3.2)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0.1)), 1.0)
    }

    func testDuplicateIDsCannotInflateCredit() throws {
        let returned = [Neighbor(id: 1, distance: 1.0), Neighbor(id: 1, distance: 1.0), Neighbor(id: 1, distance: 1.0)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 1.0 / 3.0, accuracy: 1e-12)
    }

    func testOnlyFirstKResultsAreConsidered() throws {
        let returned = [Neighbor(id: 7, distance: 100), Neighbor(id: 8, distance: 100), Neighbor(id: 9, distance: 100), Neighbor(id: 1, distance: 1.0)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 0.0)
    }

    func testEmptyReturnedIsZero() throws {
        XCTAssertEqual(try Recall.atK(returned: [], groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 0.0)
    }

    func testInsufficientGroundTruthThrows() {
        XCTAssertThrowsError(try Recall.atK(returned: [], groundTruth: Array(gt.prefix(2)), policy: RecallPolicy(k: 3, epsilon: 0))) { error in
            XCTAssertEqual(error as? RecallError, .insufficientGroundTruth(have: 2, need: 3))
        }
        XCTAssertThrowsError(try Recall.atK(returned: [], groundTruth: gt, policy: RecallPolicy(k: 0, epsilon: 0))) { error in
            XCTAssertEqual(error as? RecallError, .invalidK)
        }
    }

    func testClampedToUnitInterval() throws {
        // Every returned neighbour within threshold and k distinct IDs → exactly 1.0, never above.
        let returned = [Neighbor(id: 1, distance: 0.5), Neighbor(id: 2, distance: 0.5), Neighbor(id: 3, distance: 0.5)]
        XCTAssertEqual(try Recall.atK(returned: returned, groundTruth: gt, policy: RecallPolicy(k: 3, epsilon: 0)), 1.0)
    }
}
