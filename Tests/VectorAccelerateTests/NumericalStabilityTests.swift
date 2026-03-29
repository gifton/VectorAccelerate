// VectorAccelerate: Numerical Stability Test Suite
//
// Comprehensive tests for edge cases that can cause numerical instability:
// - Zero/near-zero vectors (division by zero, normalization)
// - Extreme values (overflow, underflow)
// - Denormal numbers (subnormal floats)
// - NaN/Inf propagation
// - Catastrophic cancellation (mixed-scale accumulation)
// - Kernel-specific edge cases
//
// These tests verify correctness at the boundaries of floating-point representation
// and ensure the library degrades gracefully under extreme conditions.
//
// Reference: IEEE 754-2019 floating-point standard
// Float32 properties:
//   - Epsilon (smallest x where 1+x != 1): ~1.19e-7
//   - Min normal: ~1.18e-38
//   - Min subnormal: ~1.4e-45
//   - Max finite: ~3.4e38
//   - Overflow threshold for exp(): ~88.7
//   - Underflow threshold for exp(): ~-87.3

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

// MARK: - Numerical Constants

/// IEEE 754 Float32 boundary values for testing
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
enum Float32Boundaries {
    /// Machine epsilon: smallest x where 1 + x != 1
    static let epsilon: Float = Float.ulpOfOne  // ~1.19e-7

    /// Smallest positive normal number
    static let minNormal: Float = Float.leastNormalMagnitude  // ~1.18e-38

    /// Smallest positive subnormal (denormal) number
    static let minSubnormal: Float = Float.leastNonzeroMagnitude  // ~1.4e-45

    /// Largest finite number
    static let maxFinite: Float = Float.greatestFiniteMagnitude  // ~3.4e38

    /// Threshold where exp(x) overflows to Inf
    static let expOverflow: Float = 88.7

    /// Threshold where exp(x) underflows to 0
    static let expUnderflow: Float = -87.3

    /// Threshold where sqrt underflows
    static let sqrtUnderflow: Float = minSubnormal * minSubnormal

    /// A "large but safe" value that won't overflow in most operations
    static let largeSafe: Float = 1e18

    /// A "small but safe" value above denormal range
    static let smallSafe: Float = 1e-18
}

// MARK: - Test Helpers for Numerical Stability

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NumericalStabilityHelpers {

    // MARK: - Edge Case Vector Generation

    /// Generate a zero vector
    static func zeroVector(dimension: Int) -> [Float] {
        [Float](repeating: 0.0, count: dimension)
    }

    /// Generate a near-zero vector (all components ~epsilon)
    static func nearZeroVector(dimension: Int, scale: Float = Float32Boundaries.epsilon) -> [Float] {
        [Float](repeating: scale, count: dimension)
    }

    /// Generate a denormal vector (subnormal floats)
    static func denormalVector(dimension: Int) -> [Float] {
        [Float](repeating: Float32Boundaries.minSubnormal * 100, count: dimension)
    }

    /// Generate a vector with one extreme value and rest zeros
    static func spikeVector(dimension: Int, spikeIndex: Int, spikeValue: Float) -> [Float] {
        var v = zeroVector(dimension: dimension)
        v[spikeIndex] = spikeValue
        return v
    }

    /// Generate a vector with mixed scales (some large, some small)
    static func mixedScaleVector(dimension: Int, largeValue: Float = 1e6, smallValue: Float = 1e-6) -> [Float] {
        (0..<dimension).map { i in
            i % 2 == 0 ? largeValue : smallValue
        }
    }

    /// Generate a vector that would overflow if squared
    static func overflowRiskVector(dimension: Int) -> [Float] {
        // sqrt(Float.greatestFiniteMagnitude) ≈ 1.84e19
        let sqrtMax = sqrt(Float32Boundaries.maxFinite)
        return [Float](repeating: sqrtMax / Float(dimension), count: dimension)
    }

    /// Generate a vector with values that cause exp() overflow
    static func expOverflowVector(dimension: Int) -> [Float] {
        [Float](repeating: 100.0, count: dimension)  // exp(100) = Inf
    }

    /// Generate a vector with values that cause exp() underflow
    static func expUnderflowVector(dimension: Int) -> [Float] {
        [Float](repeating: -100.0, count: dimension)  // exp(-100) = 0
    }

    /// Generate opposing vectors (for cancellation tests)
    static func cancellingPair(dimension: Int, value: Float) -> ([Float], [Float]) {
        let a = [Float](repeating: value, count: dimension)
        let b = [Float](repeating: -value, count: dimension)
        return (a, b)
    }

    /// Generate NaN-containing vector
    static func nanVector(dimension: Int, nanIndex: Int = 0) -> [Float] {
        var v = [Float](repeating: 1.0, count: dimension)
        v[nanIndex] = Float.nan
        return v
    }

    /// Generate Inf-containing vector
    static func infVector(dimension: Int, infIndex: Int = 0, negative: Bool = false) -> [Float] {
        var v = [Float](repeating: 1.0, count: dimension)
        v[infIndex] = negative ? -Float.infinity : Float.infinity
        return v
    }

    // MARK: - CPU Reference Implementations

    /// Stable L2 normalization with explicit epsilon handling
    static func cpuL2Normalize(_ v: [Float], epsilon: Float = 1e-8) -> [Float] {
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if norm <= epsilon {
            return [Float](repeating: 0, count: v.count)
        }
        return v.map { $0 / norm }
    }

    /// Stable cosine similarity with epsilon guard
    static func cpuCosineSimilarity(_ a: [Float], _ b: [Float], epsilon: Float = 1e-8) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0) { $0 + $1 * $1 })
        let denom = normA * normB
        if denom <= epsilon {
            return 0.0
        }
        // Clamp to [-1, 1] for numerical stability
        return min(max(dot / denom, -1.0), 1.0)
    }

    /// Stable Minkowski distance with overflow protection
    static func cpuMinkowski(_ a: [Float], _ b: [Float], p: Float) -> Float {
        // Handle special cases first
        if p == 1.0 {
            return zip(a, b).reduce(0) { $0 + abs($1.0 - $1.1) }
        }
        if p == 2.0 {
            return sqrt(zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
        }

        // For general p, use log-space to prevent overflow
        let diffs = zip(a, b).map { abs($0 - $1) }
        let maxDiff = diffs.max() ?? 0

        if maxDiff < 1e-10 {
            return 0.0
        }

        // Normalize to prevent overflow: sum((x/max)^p) * max
        let normalized = diffs.map { $0 / maxDiff }
        let sumPowers = normalized.reduce(0) { $0 + pow($1, p) }
        return maxDiff * pow(sumPowers, 1.0 / p)
    }

    /// Stable LogSumExp using max-shift trick
    static func cpuLogSumExp(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return -Float.infinity }
        let maxVal = values.max()!
        if maxVal == -Float.infinity { return -Float.infinity }
        if maxVal == Float.infinity { return Float.infinity }
        let sumExp = values.reduce(0) { $0 + exp($1 - maxVal) }
        return log(sumExp) + maxVal
    }

    /// Stable softmax using LogSumExp
    static func cpuSoftmax(_ values: [Float]) -> [Float] {
        let lse = cpuLogSumExp(values)
        if lse == -Float.infinity || lse == Float.infinity || lse.isNaN {
            return [Float](repeating: 0, count: values.count)
        }
        return values.map { exp($0 - lse) }
    }

    /// Stable sigmoid with clipping for extreme inputs
    static func cpuSigmoid(_ x: Float) -> Float {
        if x > 20 { return 1.0 }
        if x < -20 { return 0.0 }
        return 1.0 / (1.0 + exp(-x))
    }

    // MARK: - Validation Helpers

    /// Check if result is finite (not NaN or Inf)
    static func isFinite(_ value: Float) -> Bool {
        !value.isNaN && !value.isInfinite
    }

    /// Check if all values in array are finite
    static func allFinite(_ values: [Float]) -> Bool {
        values.allSatisfy { isFinite($0) }
    }

    /// Check if all values in 2D array are finite
    static func allFinite(_ values: [[Float]]) -> Bool {
        values.allSatisfy { allFinite($0) }
    }

    /// Count NaN values
    static func countNaN(_ values: [Float]) -> Int {
        values.filter { $0.isNaN }.count
    }

    /// Count Inf values
    static func countInf(_ values: [Float]) -> Int {
        values.filter { $0.isInfinite }.count
    }
}

// MARK: - Zero Vector Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ZeroVectorStabilityTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - L2 Distance with Zero Vectors

    func testL2DistanceZeroVsZero() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]

        let results = try await kernel.compute(queries: zero, database: zero)

        // Distance from zero to zero should be exactly 0
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-10)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "L2(zero, zero) should be finite, got: \(results[0][0])")
    }

    func testL2DistanceZeroVsNonZero() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]
        let unit = [[Float](repeating: 1.0, count: 128)]

        let results = try await kernel.compute(queries: zero, database: unit)

        // L2(zero, ones) = sqrt(128) ≈ 11.31
        let expected = sqrt(Float(128))
        XCTAssertEqual(results[0][0], expected, accuracy: 1e-4)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
    }

    // MARK: - Cosine Similarity with Zero Vectors

    func testCosineSimilarityZeroVsZero() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]

        let results = try await kernel.compute(queries: zero, database: zero)

        // Cosine of zero vectors should be 0 (not NaN from 0/0)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Cosine(zero, zero) must be finite, got: \(results[0][0])")
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-6,
                       "Cosine(zero, zero) should be 0 by convention")
    }

    func testCosineSimilarityZeroVsNonZero() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]
        let unit = [NumericalStabilityHelpers.cpuL2Normalize([Float](repeating: 1.0, count: 128))]

        let results = try await kernel.compute(queries: zero, database: unit)

        // Cosine(zero, any) should be 0 (not NaN)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-6)
    }

    // MARK: - Dot Product with Zero Vectors

    func testDotProductZeroVector() async throws {
        let kernel = try await DotProductKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]
        let random = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)[0]]

        let results = try await kernel.compute(queries: zero, database: random)

        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-10)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
    }
}

// MARK: - Near-Zero and Denormal Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class DenormalStabilityTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    func testL2DistanceNearZeroVectors() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let nearZero = [NumericalStabilityHelpers.nearZeroVector(dimension: 128)]

        let results = try await kernel.compute(queries: nearZero, database: nearZero)

        // Should compute correctly, not underflow to garbage
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Near-zero L2 distance should be finite")
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-4,
                       "L2(x, x) should be 0 even for near-zero x")
    }

    func testCosineSimilarityDenormalVectors() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)
        let denormal = [NumericalStabilityHelpers.denormalVector(dimension: 128)]

        let results = try await kernel.compute(queries: denormal, database: denormal)

        // Identical vectors should have cosine = 1, even if denormal
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Denormal cosine should be finite, got: \(results[0][0])")
        // Note: Due to precision loss in denormal range, we accept wider tolerance
        XCTAssertEqual(results[0][0], 1.0, accuracy: 0.01,
                       "Cosine(x, x) should be ~1 for denormal x")
    }

    func testDotProductDenormalVectors() async throws {
        let kernel = try await DotProductKernel(context: context)
        let denormal = [NumericalStabilityHelpers.denormalVector(dimension: 128)]

        let results = try await kernel.compute(queries: denormal, database: denormal)

        // Result may underflow to 0, but should not be NaN
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Denormal dot product should be finite")
    }
}

// MARK: - Overflow/Underflow Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class OverflowUnderflowStabilityTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - L2 Distance Overflow

    func testL2DistanceLargeValues() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        // Values that could overflow when squared
        let largeA = [[Float](repeating: 1e19, count: 128)]
        let largeB = [[Float](repeating: 0.0, count: 128)]

        let results = try await kernel.compute(queries: largeA, database: largeB)

        // May overflow, but should handle gracefully
        // Accept either Inf (overflow) or correct large value
        if results[0][0].isInfinite {
            // Overflow detected - this is acceptable but we should document it
            XCTAssertFalse(results[0][0].isNaN, "Should overflow to Inf, not NaN")
        } else {
            XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
        }
    }

    func testL2DistanceOverflowSafeValues() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        // Large but safe values that shouldn't overflow
        let safeA = [[Float](repeating: Float32Boundaries.largeSafe, count: 128)]
        let safeB = [[Float](repeating: 0.0, count: 128)]

        let results = try await kernel.compute(queries: safeA, database: safeB)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Safe large values should not overflow")
    }

    // MARK: - Dot Product Overflow

    func testDotProductLargeValues() async throws {
        let kernel = try await DotProductKernel(context: context)

        // 128 dimensions * (1e18)^2 = 1.28e38, close to max float
        let large = [[Float](repeating: 1e18, count: 128)]

        let results = try await kernel.compute(queries: large, database: large)

        // Should handle gracefully - either correct value or Inf
        XCTAssertFalse(results[0][0].isNaN, "Should not produce NaN")
    }

    func testDotProductHighDimension() async throws {
        let kernel = try await DotProductKernel(context: context)

        // High dimension with moderate values: 1536 * 100^2 = 15.36M (safe)
        let moderate = [[Float](repeating: 100.0, count: 1536)]

        let results = try await kernel.compute(queries: moderate, database: moderate)

        let expected: Float = Float(1536) * 100.0 * 100.0  // 15,360,000
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 1e-4)
    }
}

// MARK: - Minkowski Distance Stability Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MinkowskiStabilityTests: XCTestCase {

    var context: Metal4Context!
    var kernel: MinkowskiDistanceKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await MinkowskiDistanceKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Special p Values

    func testMinkowskiP1_Manhattan() async throws {
        let queries = [[Float](repeating: 1.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 1.0)
        )
        let results = result.asMatrix()

        let expected: Float = 64.0  // Sum of |1 - 0| for 64 dimensions
        XCTAssertEqual(results[0][0], expected, accuracy: 1e-4)
    }

    func testMinkowskiP2_Euclidean() async throws {
        let queries = [[Float](repeating: 1.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: .euclidean
        )
        let results = result.asMatrix()

        let expected = sqrt(Float(64))  // sqrt(sum(1^2)) = 8
        XCTAssertEqual(results[0][0], expected, accuracy: 1e-4)
    }

    // MARK: - Overflow Risk with Large p

    func testMinkowskiLargeP_OverflowRisk() async throws {
        // With p=3 and large differences, x^3 can overflow
        let largeA = [[Float](repeating: 1000.0, count: 64)]
        let largeB = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: largeA,
            dataset: largeB,
            config: Metal4MinkowskiConfig(p: 3.0)
        )
        let results = result.asMatrix()

        // 1000^3 = 1e9, sum = 64e9, root = ~4000
        // Should not overflow since 1e9 fits in float
        let expected = NumericalStabilityHelpers.cpuMinkowski(largeA[0], largeB[0], p: 3.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=3 should be finite, got: \(results[0][0])")
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.01)
    }

    func testMinkowskiP5_ExtremeOverflowRisk() async throws {
        // With p=5: 100^5 = 1e10, but still within float range
        let queries = [[Float](repeating: 100.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 5.0)
        )
        let results = result.asMatrix()

        // Verify result is finite
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=5 should handle moderate values")
    }

    // MARK: - Extreme p Values (Phase 2 Stability Tests)

    func testMinkowskiP10_StableKernel() async throws {
        // p=10 with moderate values - this triggers the stable kernel
        // 10^10 = 1e10 per dimension, 64 * 1e10 = 6.4e11 (within float range)
        // But with p=10, accumulation overflow becomes a real risk
        let queries = [[Float](repeating: 10.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        // useStableComputation defaults to true for p > 10, so force it on
        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 10.0, useStableComputation: true)
        )
        let results = result.asMatrix()

        // CPU reference using stable algorithm
        let expected = NumericalStabilityHelpers.cpuMinkowski(queries[0], database[0], p: 10.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=10 stable should be finite, got: \(results[0][0])")
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.01,
                       "Minkowski p=10 stable should match CPU within 1%")
    }

    func testMinkowskiP20_StableKernel() async throws {
        // p=20 with moderate values - extreme accumulation pressure
        // Without normalization: 5^20 = 9.5e13, sum = 64 * 9.5e13 = 6e15
        // With normalization: all values are in [0, 1] before power
        let queries = [[Float](repeating: 5.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 20.0, useStableComputation: true)
        )
        let results = result.asMatrix()

        // CPU reference
        let expected = NumericalStabilityHelpers.cpuMinkowski(queries[0], database[0], p: 20.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=20 stable should be finite, got: \(results[0][0])")
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.01,
                       "Minkowski p=20 stable should match CPU within 1%")
    }

    func testMinkowskiP10_HighDimension() async throws {
        // p=10 with 256 dimensions - stress test for accumulation
        // Even with normalization, sum of 256 terms approaches limit
        let queries = [[Float](repeating: 10.0, count: 256)]
        let database = [[Float](repeating: 0.0, count: 256)]

        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 10.0, useStableComputation: true)
        )
        let results = result.asMatrix()

        let expected = NumericalStabilityHelpers.cpuMinkowski(queries[0], database[0], p: 10.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=10 (256D) should be finite, got: \(results[0][0])")
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.02,
                       "Minkowski p=10 (256D) should match CPU within 2%")
    }

    func testMinkowskiP10_VsNonStable() async throws {
        // Compare stable vs non-stable for p=10 with small values
        // Both should produce similar results for well-behaved inputs
        let queries = [[Float](repeating: 2.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        // Stable kernel
        let stableResult = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 10.0, useStableComputation: true)
        )
        let stableDistance = stableResult.asMatrix()[0][0]

        // Non-stable kernel (force off)
        let batchResult = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 10.0, useStableComputation: false)
        )
        let batchDistance = batchResult.asMatrix()[0][0]

        // CPU reference
        let expected = NumericalStabilityHelpers.cpuMinkowski(queries[0], database[0], p: 10.0)

        // Both should be finite
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(stableDistance),
                      "Stable kernel should produce finite result")
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(batchDistance),
                      "Batch kernel should produce finite result for small inputs")

        // Both should match CPU
        XCTAssertEqual(stableDistance, expected, accuracy: expected * 0.02)
        XCTAssertEqual(batchDistance, expected, accuracy: expected * 0.02)
    }

    func testMinkowskiP15_MixedValues() async throws {
        // p=15 with mixed magnitude inputs - stable kernel should handle correctly
        // Some differences are large, some small
        var query = [Float](repeating: 0, count: 64)
        for i in 0..<64 {
            query[i] = Float(i + 1)  // 1, 2, 3, ..., 64
        }
        let database = [[Float](repeating: 0.0, count: 64)]

        let result = try await kernel.computeDistances(
            queries: [query],
            dataset: database,
            config: Metal4MinkowskiConfig(p: 15.0, useStableComputation: true)
        )
        let results = result.asMatrix()

        let expected = NumericalStabilityHelpers.cpuMinkowski(query, database[0], p: 15.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Minkowski p=15 mixed should be finite")
        // Wider tolerance for mixed values (numerical precision varies)
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.05,
                       "Minkowski p=15 mixed should match CPU within 5%")
    }

    // MARK: - Mixed Scale (Catastrophic Cancellation Risk)

    func testMinkowskiMixedScales() async throws {
        // Mixed scales: some differences are 1e6, others are 1e-6
        // When raised to power p, these become vastly different magnitudes
        let queryA = NumericalStabilityHelpers.mixedScaleVector(dimension: 64, largeValue: 1e6, smallValue: 1e-6)
        let queryB = NumericalStabilityHelpers.zeroVector(dimension: 64)

        let result = try await kernel.computeDistances(
            queries: [queryA],
            dataset: [queryB],
            config: Metal4MinkowskiConfig(p: 3.0)
        )
        let results = result.asMatrix()

        let expected = NumericalStabilityHelpers.cpuMinkowski(queryA, queryB, p: 3.0)

        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
        // Accept wider tolerance due to accumulation precision loss
        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.1,
                       "Mixed scale Minkowski should be approximately correct")
    }

    // MARK: - Zero Differences

    func testMinkowskiIdenticalVectors() async throws {
        let vector = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 64)[0]

        let result = try await kernel.computeDistances(
            queries: [vector],
            dataset: [vector],
            config: Metal4MinkowskiConfig(p: 3.0)
        )
        let results = result.asMatrix()

        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-6,
                       "Minkowski(x, x) should be 0")
    }

    // MARK: - Fractional p (0 < p < 1)

    func testMinkowskiFractionalP() async throws {
        let queries = [[Float](repeating: 1.0, count: 64)]
        let database = [[Float](repeating: 0.0, count: 64)]

        // p = 0.5 means we sum sqrt(|diff|) then square the result
        let result = try await kernel.computeDistances(
            queries: queries,
            dataset: database,
            config: Metal4MinkowskiConfig(p: 0.5)
        )
        let results = result.asMatrix()

        // sum(1^0.5) = 64, then 64^(1/0.5) = 64^2 = 4096
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]),
                      "Fractional p should produce finite result")
    }
}

// MARK: - LogSumExp Stability Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class LogSumExpStabilityTests: XCTestCase {

    var context: Metal4Context!
    var kernel: LogSumExpKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await LogSumExpKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    // MARK: - Extreme Values

    func testLogSumExpLargePositiveValues() async throws {
        // Values where exp(x) would overflow without max-shift
        let values: [Float] = [100, 101, 102, 99, 100]

        let result = try await kernel.reduceValue(input: values)

        // LSE should be ~102.4 (dominated by max + log(sum of small terms))
        let expected = NumericalStabilityHelpers.cpuLogSumExp(values)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(result),
                      "LSE of large values should be finite, got: \(result)")
        XCTAssertEqual(result, expected, accuracy: 0.1)
    }

    func testLogSumExpLargeNegativeValues() async throws {
        // Values where exp(x) would all underflow to 0
        let values: [Float] = [-100, -101, -102, -99, -100]

        let result = try await kernel.reduceValue(input: values)

        // Should still compute correctly via max-shift
        let expected = NumericalStabilityHelpers.cpuLogSumExp(values)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(result))
        XCTAssertEqual(result, expected, accuracy: 0.1)
    }

    func testLogSumExpMixedExtremes() async throws {
        // Mix of values that could cause issues
        let values: [Float] = [100, -100, 0, 50, -50]

        let result = try await kernel.reduceValue(input: values)

        // Dominated by max (100), result should be ~100
        let expected = NumericalStabilityHelpers.cpuLogSumExp(values)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(result))
        XCTAssertEqual(result, expected, accuracy: 0.1)
    }

    // MARK: - Edge Cases

    func testLogSumExpAllNegativeInfinity() async throws {
        let values: [Float] = [-Float.infinity, -Float.infinity, -Float.infinity]

        let result = try await kernel.reduceValue(input: values)

        // LSE of all -inf should be -inf
        XCTAssertEqual(result, -Float.infinity)
    }

    func testLogSumExpContainsPositiveInfinity() async throws {
        let values: [Float] = [1.0, Float.infinity, 2.0]

        let result = try await kernel.reduceValue(input: values)

        // LSE with +inf should be +inf
        XCTAssertEqual(result, Float.infinity)
    }

    func testLogSumExpSingleValue() async throws {
        let values: [Float] = [42.0]

        let result = try await kernel.reduceValue(input: values)

        // LSE of single value x = log(exp(x)) = x
        XCTAssertEqual(result, 42.0, accuracy: 1e-4)
    }

    func testLogSumExpIdenticalValues() async throws {
        let values: [Float] = [5.0, 5.0, 5.0, 5.0]

        let result = try await kernel.reduceValue(input: values)

        // LSE of n identical values x = x + log(n)
        let expected: Float = 5.0 + log(4.0)
        XCTAssertEqual(result, expected, accuracy: 1e-4)
    }
}

// MARK: - NaN/Inf Propagation Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class NaNInfPropagationTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - L2 Distance with NaN/Inf

    func testL2DistanceWithNaN() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let nanVec = [NumericalStabilityHelpers.nanVector(dimension: 128)]
        let normal = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)[0]]

        let results = try await kernel.compute(queries: nanVec, database: normal)

        // NaN should propagate - distance with NaN input should be NaN
        XCTAssertTrue(results[0][0].isNaN,
                      "L2 distance with NaN input should produce NaN, got: \(results[0][0])")
    }

    func testL2DistanceWithInf() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let infVec = [NumericalStabilityHelpers.infVector(dimension: 128)]
        let normal = [[Float](repeating: 1.0, count: 128)]

        let results = try await kernel.compute(queries: infVec, database: normal)

        // Inf - finite = Inf, sqrt(Inf) = Inf
        XCTAssertTrue(results[0][0].isInfinite,
                      "L2 distance with Inf input should produce Inf")
    }

    // MARK: - Cosine Similarity with NaN/Inf

    func testCosineSimilarityWithNaN() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)
        let nanVec = [NumericalStabilityHelpers.nanVector(dimension: 128)]
        let normal = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)[0]]

        let results = try await kernel.compute(queries: nanVec, database: normal)

        // NaN should propagate through cosine computation
        XCTAssertTrue(results[0][0].isNaN,
                      "Cosine with NaN input should produce NaN")
    }

    // MARK: - Dot Product with NaN/Inf

    func testDotProductWithNaN() async throws {
        let kernel = try await DotProductKernel(context: context)
        let nanVec = [NumericalStabilityHelpers.nanVector(dimension: 128)]
        let normal = [[Float](repeating: 1.0, count: 128)]

        let results = try await kernel.compute(queries: nanVec, database: normal)

        XCTAssertTrue(results[0][0].isNaN,
                      "Dot product with NaN should produce NaN")
    }

    func testDotProductInfTimesZero() async throws {
        let kernel = try await DotProductKernel(context: context)

        // Inf * 0 = NaN in IEEE 754
        let infVec = [NumericalStabilityHelpers.infVector(dimension: 128)]
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]

        let results = try await kernel.compute(queries: infVec, database: zero)

        // Should propagate the Inf*0 = NaN or handle it gracefully
        // Either NaN or 0 is acceptable depending on implementation
        // We just verify it doesn't crash and produces some output
        XCTAssertEqual(results.count, 1, "Kernel handled Inf*0 without crashing")
        XCTAssertEqual(results[0].count, 1)
    }
}

// MARK: - Catastrophic Cancellation Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class CatastrophicCancellationTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - L2 Distance Cancellation

    func testL2DistanceNearlyIdentical() async throws {
        let kernel = try await L2DistanceKernel(context: context)

        // Two vectors that differ by only epsilon in each dimension
        let base = [Float](repeating: 1e6, count: 128)
        var perturbed = base
        for i in 0..<128 {
            perturbed[i] += Float32Boundaries.epsilon * base[i]
        }

        let results = try await kernel.compute(queries: [base], database: [perturbed])

        // Should detect the small difference without catastrophic cancellation
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
        XCTAssertGreaterThan(results[0][0], 0,
                             "Should detect small difference")
    }

    // MARK: - Dot Product Cancellation

    func testDotProductCancellation() async throws {
        let kernel = try await DotProductKernel(context: context)

        // Vectors designed to cause cancellation: alternating +large, -large
        var a = [Float](repeating: 0, count: 128)
        var b = [Float](repeating: 0, count: 128)

        for i in 0..<128 {
            a[i] = i % 2 == 0 ? 1e6 : -1e6
            b[i] = 1.0
        }

        let results = try await kernel.compute(queries: [a], database: [b])

        // Sum should be 0 (64 * 1e6 - 64 * 1e6)
        XCTAssertTrue(NumericalStabilityHelpers.isFinite(results[0][0]))
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-2,
                       "Cancelling dot product should be ~0")
    }
}

// MARK: - Dimension Edge Cases

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class DimensionEdgeCaseTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - Single Dimension

    func testL2DistanceDim1() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let a: [[Float]] = [[5.0]]
        let b: [[Float]] = [[3.0]]

        let results = try await kernel.compute(queries: a, database: b)

        XCTAssertEqual(results[0][0], 2.0, accuracy: 1e-6)
    }

    func testCosineSimilarityDim1() async throws {
        let kernel = try await CosineSimilarityKernel(context: context)
        let a: [[Float]] = [[5.0]]
        let b: [[Float]] = [[3.0]]

        let results = try await kernel.compute(queries: a, database: b)

        // Same sign => cosine = 1
        XCTAssertEqual(results[0][0], 1.0, accuracy: 1e-6)
    }

    // MARK: - Non-Multiple of 4 (SIMD remainder handling)

    func testL2DistanceDim7() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let a = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 7)[0]]
        let b = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 7)[0]]

        let results = try await kernel.compute(queries: a, database: b)
        let expected = Metal4KernelTestHelpers.cpuL2Distance(a[0], b[0])

        XCTAssertEqual(results[0][0], expected, accuracy: 1e-4)
    }

    func testDotProductDim13() async throws {
        let kernel = try await DotProductKernel(context: context)
        let a = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 13)[0]]
        let b = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 13)[0]]

        let results = try await kernel.compute(queries: a, database: b)
        let expected = Metal4KernelTestHelpers.cpuDotProduct(a[0], b[0])

        XCTAssertEqual(results[0][0], expected, accuracy: 1e-4)
    }

    // MARK: - Large Dimensions (Accumulation Stress)

    func testL2DistanceDim1536() async throws {
        let kernel = try await L2DistanceKernel(context: context)
        let a = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 1536)[0]]
        let b = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 1536)[0]]

        let results = try await kernel.compute(queries: a, database: b)
        let expected = Metal4KernelTestHelpers.cpuL2Distance(a[0], b[0])

        XCTAssertEqual(results[0][0], expected, accuracy: expected * 0.01,
                       "1536D L2 should match CPU within 1%")
    }

    func testDotProductDim1536() async throws {
        let kernel = try await DotProductKernel(context: context)
        let a = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 1536)[0]]
        let b = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 1536)[0]]

        let results = try await kernel.compute(queries: a, database: b)
        let expected = Metal4KernelTestHelpers.cpuDotProduct(a[0], b[0])

        XCTAssertEqual(results[0][0], expected, accuracy: abs(expected) * 0.01,
                       "1536D dot product should match CPU within 1%")
    }
}

// MARK: - Cosine Similarity Boundary Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class CosineBoundaryTests: XCTestCase {

    var context: Metal4Context!
    var kernel: CosineSimilarityKernel!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
        kernel = try await CosineSimilarityKernel(context: context)
    }

    override func tearDown() {
        kernel = nil
        context = nil
        super.tearDown()
    }

    func testCosineIdenticalVectors() async throws {
        let vec = [Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)[0]]

        let results = try await kernel.compute(queries: vec, database: vec)

        // cos(0) = 1
        XCTAssertEqual(results[0][0], 1.0, accuracy: 1e-5,
                       "Cosine of identical vectors should be 1")
    }

    func testCosineOppositeVectors() async throws {
        let vec = Metal4KernelTestHelpers.randomVectors(count: 1, dimension: 128)[0]
        let opposite = vec.map { -$0 }

        let results = try await kernel.compute(queries: [vec], database: [opposite])

        // cos(π) = -1
        XCTAssertEqual(results[0][0], -1.0, accuracy: 1e-5,
                       "Cosine of opposite vectors should be -1")
    }

    func testCosineOrthogonalVectors() async throws {
        // Create orthogonal vectors
        var a = [Float](repeating: 0, count: 128)
        var b = [Float](repeating: 0, count: 128)

        // First half non-zero in a, second half non-zero in b
        for i in 0..<64 {
            a[i] = Float.random(in: -1...1)
            b[64 + i] = Float.random(in: -1...1)
        }

        let results = try await kernel.compute(queries: [a], database: [b])

        // cos(π/2) = 0
        XCTAssertEqual(results[0][0], 0.0, accuracy: 1e-5,
                       "Cosine of orthogonal vectors should be 0")
    }

    func testCosineOutputRange() async throws {
        // Generate many random pairs and verify all outputs in [-1, 1]
        let queries = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 128)
        let database = Metal4KernelTestHelpers.randomVectors(count: 100, dimension: 128)

        let results = try await kernel.compute(queries: queries, database: database)

        for row in results {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, -1.0,
                                            "Cosine should be >= -1, got \(value)")
                XCTAssertLessThanOrEqual(value, 1.0,
                                          "Cosine should be <= 1, got \(value)")
            }
        }
    }
}

// MARK: - Quantization Stability Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class QuantizationStabilityTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        context = try await Metal4Context()
    }

    override func tearDown() {
        context = nil
        super.tearDown()
    }

    // MARK: - Scalar Quantization Edge Cases

    func testScalarQuantizationZeroVector() async throws {
        let kernel = try await ScalarQuantizationKernel(context: context)
        let zero = NumericalStabilityHelpers.zeroVector(dimension: 128)

        // Quantizing zero vector should not crash
        let result = try await kernel.quantize(zero)

        // Should produce valid output
        XCTAssertFalse(result.quantizedData.isEmpty)
        XCTAssertTrue(result.scale.isFinite || result.scale == 0,
                      "Scale should be finite for zero vector")
    }

    func testScalarQuantizationConstantVector() async throws {
        let kernel = try await ScalarQuantizationKernel(context: context)
        let constant = [Float](repeating: 5.0, count: 128)

        // All same value means min == max, potential divide by zero in normalization
        let result = try await kernel.quantize(constant)

        XCTAssertFalse(result.quantizedData.isEmpty)
        // Scale may be 0 or a small value for constant input
        XCTAssertTrue(result.scale.isFinite || result.scale == 0)
    }

    func testScalarQuantizationExtremeRange() async throws {
        let kernel = try await ScalarQuantizationKernel(context: context)

        // Mix of very large and very small values
        var extreme = [Float](repeating: 0, count: 128)
        for i in 0..<64 {
            extreme[i] = 1e6
        }
        for i in 64..<128 {
            extreme[i] = 1e-6
        }

        let result = try await kernel.quantize(extreme)

        XCTAssertFalse(result.quantizedData.isEmpty)
        XCTAssertTrue(result.scale.isFinite)
    }

    // MARK: - Binary Quantization Edge Cases

    func testBinaryQuantizationZeroVector() async throws {
        let kernel = try await BinaryQuantizationKernel(context: context)
        let zero = [NumericalStabilityHelpers.zeroVector(dimension: 128)]

        let result = try await kernel.quantize(vectors: zero)

        // Zero vector quantized should produce valid output
        XCTAssertEqual(result.binaryVectors.count, 1)
    }

    func testBinaryQuantizationBoundaryValues() async throws {
        let kernel = try await BinaryQuantizationKernel(context: context)

        // Values exactly at 0 (the threshold)
        let boundary = [[Float](repeating: 0.0, count: 128)]

        let result = try await kernel.quantize(vectors: boundary)

        XCTAssertEqual(result.binaryVectors.count, 1)
    }

    func testBinaryQuantizationMixedSigns() async throws {
        let kernel = try await BinaryQuantizationKernel(context: context)

        // Half positive, half negative
        var mixed = [Float](repeating: 0, count: 128)
        for i in 0..<64 {
            mixed[i] = 1.0
        }
        for i in 64..<128 {
            mixed[i] = -1.0
        }

        let result = try await kernel.quantize(vectors: [mixed])

        XCTAssertEqual(result.binaryVectors.count, 1)
    }
}
