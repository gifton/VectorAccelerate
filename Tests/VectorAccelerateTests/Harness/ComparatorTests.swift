import XCTest
import VectorTestKit

/// Tiers (spec §4.5): both NaN → equal; bitwise equal → equal; class mismatch → hard fail;
/// abs/rel envelope; optional ULP tier. The ten adversarial value classes are adapted from DifferentialKernelVsCPUTests.
final class ComparatorTests: XCTestCase {
    private let classes: [(String, Double)] = [
        ("normal", 1.234), ("zero", 0.0), ("negZero", -0.0), ("huge", 1e19), ("tiny", 1e-20),
        ("subnormal", 1e-310), ("mixed", 12345.678e-9), ("nan", .nan), ("posInf", .infinity), ("negInf", -.infinity),
    ]

    func testBothNaNAreEqual() {
        XCTAssertEqual(Comparator.compare(Double.nan, Double.nan, tolerance: Tolerance()), .equal)
    }

    func testBitwiseEqualIsEqualIncludingSignedZeroAndInfinity() {
        XCTAssertEqual(Comparator.compare(0.0, 0.0, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(-0.0, -0.0, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(Double.infinity, Double.infinity, tolerance: Tolerance()), .equal)
    }

    func testSignedZeroWithinToleranceNotBitwiseEqual() {
        // +0 and -0 differ bitwise but |a-b| = 0 ≤ any envelope.
        XCTAssertEqual(Comparator.compare(0.0, -0.0, tolerance: Tolerance()), .withinTolerance)
    }

    func testClassMismatchIsHardFail() {
        let t = Tolerance(absolute: 1e300, relative: 1e300)
        XCTAssertEqual(Comparator.compare(Double.nan, 1.0, tolerance: t), .classMismatch)
        XCTAssertEqual(Comparator.compare(1.0, Double.infinity, tolerance: t), .classMismatch)
        XCTAssertEqual(Comparator.compare(Double.infinity, -Double.infinity, tolerance: t), .classMismatch)
    }

    func testAbsRelEnvelope() {
        XCTAssertEqual(Comparator.compare(100.0, 100.5, tolerance: Tolerance(absolute: 1.0)), .withinTolerance)
        XCTAssertEqual(Comparator.compare(100.0, 100.5, tolerance: Tolerance(relative: 0.01)), .withinTolerance)
        if case .outOfTolerance(let abs, let rel) = Comparator.compare(100.0, 102.0, tolerance: Tolerance(absolute: 1.0, relative: 0.01)) {
            XCTAssertEqual(abs, 2.0, accuracy: 1e-12); XCTAssertEqual(rel, 2.0 / 102.0, accuracy: 1e-12)
        } else { XCTFail("expected outOfTolerance") }
    }

    func testULPTier() {
        XCTAssertEqual(Comparator.ulpDistance(1.0, 1.0.nextUp), 1)
        XCTAssertEqual(Comparator.ulpDistance(Float(1.0), Float(1.0).nextUp), 1)
        XCTAssertEqual(Comparator.ulpDistance(-1.0, 1.0) > 1_000_000, true)
        // Extreme finite range: the true distance exceeds Int64.max, so a checked Int64 subtraction would trap.
        XCTAssertEqual(Comparator.ulpDistance(-Double.greatestFiniteMagnitude, Double.greatestFiniteMagnitude), 18_437_736_874_454_810_622 as UInt64)
        XCTAssertEqual(Comparator.ulpDistance(-Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude), 4_278_190_078 as UInt64)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp, tolerance: Tolerance(ulps: 1)), .withinTolerance)
        XCTAssertEqual(Comparator.compare(1.0, 1.0.nextUp.nextUp, tolerance: Tolerance(ulps: 1)), .outOfTolerance(absError: 1.0.nextUp.nextUp - 1.0, relError: (1.0.nextUp.nextUp - 1.0) / 1.0.nextUp.nextUp))
    }

    func testFloatOverloadSharesTiersAndAppliesFloatULPs() {
        // Float arguments select the Float overload (Swift never promotes Float to Double implicitly).
        XCTAssertEqual(Comparator.compare(Float.nan, Float.nan, tolerance: Tolerance()), .equal)
        XCTAssertEqual(Comparator.compare(Float.nan, Float(1), tolerance: Tolerance(absolute: 1e300, relative: 1e300)), .classMismatch)
        XCTAssertEqual(Comparator.compare(Float(1), Float.infinity, tolerance: Tolerance(absolute: 1e300, relative: 1e300)), .classMismatch)
        XCTAssertEqual(Comparator.compare(Float(0), -Float(0), tolerance: Tolerance()), .withinTolerance)
        XCTAssertEqual(Comparator.compare(Float(1), Float(1).nextUp, tolerance: Tolerance(ulps: 1)), .withinTolerance)
        // Beyond the Float ULP tier: the payload is exactly what the Double overload computes on the widened values.
        let b = Double(Float(1).nextUp.nextUp)
        XCTAssertEqual(Comparator.compare(Float(1), Float(1).nextUp.nextUp, tolerance: Tolerance(ulps: 1)), .outOfTolerance(absError: b - 1.0, relError: (b - 1.0) / b))
    }

    func testAdversarialClassesPairwiseNeverCrash() {
        let t = Tolerance(absolute: 1e-5, relative: 2e-4)
        for (na, a) in classes { for (nb, b) in classes {
            let v = Comparator.compare(a, b, tolerance: t)
            let sameClass = (a.isNaN && b.isNaN) || (a.isInfinite && b.isInfinite && a.sign == b.sign) || (a.isFinite && b.isFinite)
            if !sameClass { XCTAssertEqual(v, .classMismatch, "\(na) vs \(nb)") }
            else { XCTAssertNotEqual(v, .classMismatch, "\(na) vs \(nb)") }
            if a.isNaN && b.isNaN { XCTAssertEqual(v, .equal, "\(na) vs \(nb)") }
        } }
    }

    func testForSumSequentialAndTree() {
        let eps = 5.960464477539063e-08  // 2^-24
        let seq = Tolerance.forSum(terms: 768, reduction: .sequential, magnitude: 3072)
        XCTAssertEqual(seq.absolute, 2 * eps * 768 * 3072, accuracy: 1e-15)
        XCTAssertEqual(seq.relative, 0)
        let tree = Tolerance.forSum(terms: 768, reduction: .tree, magnitude: 3072)
        XCTAssertEqual(tree.absolute, 2 * eps * 10 * 3072, accuracy: 1e-15)   // ceil(log2 768) = 10
        let one = Tolerance.forSum(terms: 1, reduction: .tree, magnitude: 1)
        XCTAssertEqual(one.absolute, 2 * eps * 1 * 1, accuracy: 1e-20)           // depth floor of 1
    }

    func testToleranceRoundTrips() throws {
        let t = Tolerance(absolute: 1e-6, relative: 1e-4, ulps: 4)
        let data = try JSONEncoder().encode(t)
        XCTAssertEqual(try JSONDecoder().decode(Tolerance.self, from: data), t)
    }
}
