import Foundation

/// Numerical tolerance (spec §4.5). `absolute`/`relative` form the envelope
/// `|a−b| ≤ max(absolute, relative · max(|a|,|b|))`; `ulps` is an optional fifth tier.
public struct Tolerance: Codable, Sendable, Equatable {
    public var absolute: Double
    public var relative: Double
    public var ulps: UInt64?

    public init(absolute: Double = 0, relative: Double = 0, ulps: UInt64? = nil) {
        self.absolute = absolute
        self.relative = relative
        self.ulps = ulps
    }

    public enum Reduction: String, Codable, Sendable { case sequential, tree }

    /// Explicit, unverified-per-kernel forward-error contract for a Float32 sum of `terms` values whose
    /// absolute values sum to at most `magnitude`: sequential ≈ (D−1)·ε·Σ|xᵢ|, balanced tree ≈ ⌈log₂D⌉·ε·Σ|xᵢ|,
    /// with ε = 2⁻²⁴ and a safety factor of 2. Tested by `ComparatorTests.testForSumSequentialAndTree`.
    public static func forSum(terms: Int, reduction: Reduction, magnitude: Double) -> Tolerance {
        let eps = 5.960464477539063e-08
        let depth: Double
        switch reduction {
        case .sequential: depth = Double(max(terms, 1))
        case .tree: depth = max(1, (log2(Double(max(terms, 1)))).rounded(.up))
        }
        return Tolerance(absolute: 2 * eps * depth * magnitude, relative: 0, ulps: nil)
    }
}

public enum Verdict: Equatable, Sendable {
    case equal
    case withinTolerance
    case classMismatch
    case outOfTolerance(absError: Double, relError: Double)
}

public enum Comparator {
    public static func compare(_ a: Double, _ b: Double, tolerance: Tolerance) -> Verdict {
        if a.isNaN && b.isNaN { return .equal }
        if a.bitPattern == b.bitPattern { return .equal }
        if a.isNaN != b.isNaN { return .classMismatch }
        if a.isInfinite || b.isInfinite {
            // Both infinite with equal sign would have been bitwise equal above.
            return .classMismatch
        }
        let absError = abs(a - b)
        let scale = max(abs(a), abs(b))
        let relError = scale > 0 ? absError / scale : 0
        if absError <= max(tolerance.absolute, tolerance.relative * scale) { return .withinTolerance }
        if let u = tolerance.ulps, ulpDistance(a, b) <= u { return .withinTolerance }
        return .outOfTolerance(absError: absError, relError: relError)
    }

    public static func compare(_ a: Float, _ b: Float, tolerance: Tolerance) -> Verdict {
        if a.isNaN && b.isNaN { return .equal }
        if a.bitPattern == b.bitPattern { return .equal }
        if a.isNaN != b.isNaN { return .classMismatch }
        if a.isInfinite || b.isInfinite { return .classMismatch }
        let da = Double(a), db = Double(b)
        let absError = abs(da - db)
        let scale = max(abs(da), abs(db))
        let relError = scale > 0 ? absError / scale : 0
        if absError <= max(tolerance.absolute, tolerance.relative * scale) { return .withinTolerance }
        if let u = tolerance.ulps, ulpDistance(a, b) <= u { return .withinTolerance }
        return .outOfTolerance(absError: absError, relError: relError)
    }

    /// Distance in representable steps between two finite doubles (sign-aware ordering of bit patterns).
    public static func ulpDistance(_ a: Double, _ b: Double) -> UInt64 {
        func ordered(_ x: Double) -> Int64 {
            let i = Int64(bitPattern: x.bitPattern)
            return i >= 0 ? i : Int64.min - i
        }
        let (x, y) = (ordered(a), ordered(b))
        // The true distance is < 2⁶⁴ but can exceed Int64.max (e.g. ±greatestFiniteMagnitude), so a checked
        // `x - y` traps; wrapping subtraction plus UInt64(bitPattern:) recovers the exact value. Pinned by testULPTier.
        return x >= y ? UInt64(bitPattern: x &- y) : UInt64(bitPattern: y &- x)
    }

    public static func ulpDistance(_ a: Float, _ b: Float) -> UInt64 {
        func ordered(_ x: Float) -> Int32 {
            let i = Int32(bitPattern: x.bitPattern)
            return i >= 0 ? i : Int32.min - i
        }
        let (x, y) = (Int64(ordered(a)), Int64(ordered(b)))
        return x >= y ? UInt64(x - y) : UInt64(y - x)
    }
}
