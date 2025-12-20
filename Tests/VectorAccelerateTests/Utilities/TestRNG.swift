//
//  TestRNG.swift
//  VectorAccelerateTests
//
//  High-quality deterministic pseudorandom number generator for reproducible tests.
//
//  Based on VectorIndex RNGState with improvements:
//  - Automatic warmup to discard low-entropy initial outputs
//  - Stream mixing for parallel independent sequences
//  - Box-Muller Gaussian generation
//  - 24-bit float precision (vs 16-bit original)
//
//  Mathematical Properties:
//    - Period: 2^64 (full cycle on UInt64)
//    - Multiplier: 2862933555777941757 (prime, passes spectral test)
//    - Increment: 3037000493 (odd, ensures full period)
//
//  Usage:
//    var rng = TestRNG(seed: 42)
//    let f = rng.nextFloat(in: -1...1)
//    let g = rng.nextGaussian(mean: 0, stdDev: 1)
//

import Foundation

/// Deterministic pseudorandom number generator for reproducible tests.
///
/// Thread-safe when not shared (value semantics).
/// For parallel RNG: use different stream IDs per thread/task.
///
/// Algorithm: 64-bit Linear Congruential Generator (LCG)
///   s_next = a * s + c  (mod 2^64)
///
/// Quality: Suitable for sampling, shuffling, and test data generation.
/// NOT suitable for cryptography or high-quality Monte Carlo.
@frozen
public struct TestRNG: Sendable {
    /// Internal state (64-bit)
    @usableFromInline
    internal var state: UInt64

    /// Cached value for Box-Muller (generates pairs)
    @usableFromInline
    internal var cachedGaussian: Float?

    // MARK: - Initialization

    /// Initialize RNG with seed and optional stream ID.
    ///
    /// - Parameters:
    ///   - seed: Primary seed value (0 will be treated as 1)
    ///   - stream: Stream identifier for parallel RNG (default: 0)
    ///   - warmup: Number of initial values to discard (default: 10)
    ///
    /// Stream mixing: `seed ^ (stream << 32)` ensures different sequences
    /// for different streams even with the same base seed.
    @inlinable
    public init(seed: UInt64, stream: UInt64 = 0, warmup: Int = 10) {
        // Ensure non-zero seed (LCG with s=0 produces trivial sequence)
        let baseSeed = (seed == 0) ? 1 : seed
        // Mix stream ID into high bits to generate independent sequences
        self.state = baseSeed ^ (stream << 32)
        self.cachedGaussian = nil

        // Warmup: discard initial outputs for better state mixing
        for _ in 0..<warmup {
            _ = rawNext()
        }
    }

    // MARK: - Core Generation

    /// Generate next 64-bit random integer (internal).
    @inline(__always)
    @usableFromInline
    internal mutating func rawNext() -> UInt64 {
        // LCG recurrence: s = a * s + c (wrapping modulo 2^64)
        // Constants from VectorIndex (passes spectral test)
        state = 2862933555777941757 &* state &+ 3037000493
        return state
    }

    /// Generate next 64-bit random integer.
    ///
    /// - Returns: Uniformly distributed UInt64 in [0, 2^64)
    @inlinable
    public mutating func next() -> UInt64 {
        rawNext()
    }

    // MARK: - Float Generation

    /// Generate random float in [0, 1).
    ///
    /// Uses top 24 bits for float mantissa precision.
    /// High bits have better statistical quality in LCG.
    ///
    /// - Returns: Float uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextFloat01() -> Float {
        // Use top 24 bits for mantissa (float has 23-bit mantissa + implicit 1)
        let u = rawNext() >> 40
        return Float(u) / Float(1 << 24)
    }

    /// Generate random float in specified range.
    ///
    /// - Parameter range: Closed range for output
    /// - Returns: Float uniformly distributed in [range.lowerBound, range.upperBound]
    @inlinable
    public mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let t = nextFloat01()
        return range.lowerBound + (range.upperBound - range.lowerBound) * t
    }

    /// Generate random double in [0, 1) with full precision.
    ///
    /// Uses 53 high bits for IEEE 754 double precision.
    ///
    /// - Returns: Double uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextDouble() -> Double {
        let u = rawNext() >> 11
        return Double(u) / Double(1 << 53)
    }

    // MARK: - Gaussian Generation

    /// Generate random value from standard normal distribution.
    ///
    /// Uses Box-Muller transform for high-quality Gaussian samples.
    /// Generates pairs internally for efficiency.
    ///
    /// - Parameters:
    ///   - mean: Mean of the distribution (default: 0)
    ///   - stdDev: Standard deviation (default: 1)
    /// - Returns: Float from N(mean, stdDev^2)
    @inlinable
    public mutating func nextGaussian(mean: Float = 0, stdDev: Float = 1) -> Float {
        // Check for cached value from previous Box-Muller pair
        if let cached = cachedGaussian {
            cachedGaussian = nil
            return mean + stdDev * cached
        }

        // Box-Muller transform: generate two independent standard normals
        // from two uniform randoms
        let u1 = max(nextFloat01(), Float.ulpOfOne)  // avoid log(0)
        let u2 = nextFloat01()

        let r = sqrt(-2 * log(u1))
        let theta = 2 * Float.pi * u2

        let z0 = r * cos(theta)
        let z1 = r * sin(theta)

        // Cache second value for next call
        cachedGaussian = z1

        return mean + stdDev * z0
    }

    // MARK: - Integer Generation

    /// Generate random integer in range [0, bound).
    ///
    /// Uses modulo reduction. Slight bias for non-power-of-2 bounds,
    /// acceptable for test data generation.
    ///
    /// - Parameter bound: Upper bound (exclusive), must be positive
    /// - Returns: Random integer in [0, bound)
    @inlinable
    public mutating func nextInt(bound: Int) -> Int {
        precondition(bound > 0, "Bound must be positive")
        return Int(rawNext() % UInt64(bound))
    }

    /// Generate random integer in specified range.
    ///
    /// - Parameter range: Range for output
    /// - Returns: Random integer in the range
    @inlinable
    public mutating func nextInt(in range: Range<Int>) -> Int {
        range.lowerBound + nextInt(bound: range.count)
    }

    /// Generate random integer in specified closed range.
    ///
    /// - Parameter range: Closed range for output
    /// - Returns: Random integer in the range
    @inlinable
    public mutating func nextInt(in range: ClosedRange<Int>) -> Int {
        range.lowerBound + nextInt(bound: range.count)
    }

    // MARK: - Boolean Generation

    /// Generate random boolean with specified probability of true.
    ///
    /// - Parameter probability: Probability of returning true (default: 0.5)
    /// - Returns: Random boolean
    @inlinable
    public mutating func nextBool(probability: Float = 0.5) -> Bool {
        nextFloat01() < probability
    }

    // MARK: - Array Operations

    /// Shuffle array in place using Fisher-Yates algorithm.
    ///
    /// - Parameter array: Array to shuffle
    @inlinable
    public mutating func shuffle<T>(_ array: inout [T]) {
        guard array.count > 1 else { return }
        for i in stride(from: array.count - 1, through: 1, by: -1) {
            let j = nextInt(bound: i + 1)
            array.swapAt(i, j)
        }
    }

    /// Return a shuffled copy of the array.
    ///
    /// - Parameter array: Array to shuffle
    /// - Returns: New shuffled array
    @inlinable
    public mutating func shuffled<T>(_ array: [T]) -> [T] {
        var copy = array
        shuffle(&copy)
        return copy
    }

    /// Sample k elements from array without replacement.
    ///
    /// - Parameters:
    ///   - array: Source array
    ///   - k: Number of elements to sample
    /// - Returns: Array of k sampled elements
    @inlinable
    public mutating func sample<T>(_ array: [T], k: Int) -> [T] {
        precondition(k >= 0 && k <= array.count, "k must be in [0, array.count]")
        guard k > 0 else { return [] }

        // For small k, use selection sampling
        // For large k, shuffle and take prefix
        if k < array.count / 2 {
            var result: [T] = []
            result.reserveCapacity(k)
            var indices = Set<Int>()
            while indices.count < k {
                let idx = nextInt(bound: array.count)
                if indices.insert(idx).inserted {
                    result.append(array[idx])
                }
            }
            return result
        } else {
            return Array(shuffled(array).prefix(k))
        }
    }
}

// MARK: - RandomNumberGenerator Conformance

// TestRNG conforms to RandomNumberGenerator via the next() -> UInt64 method
// defined in the main struct. This allows use with Swift's standard random APIs
// like Float.random(in:using:) and Array.shuffled(using:).
extension TestRNG: RandomNumberGenerator {}
