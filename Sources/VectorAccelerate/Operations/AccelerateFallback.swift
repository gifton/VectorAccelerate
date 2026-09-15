// VectorAccelerate: Accelerate Framework Fallback
//
// CPU-optimized operations using Apple's Accelerate framework
//

import Foundation
import Accelerate
import VectorCore

/// High-performance CPU operations using Accelerate framework
public struct AccelerateFallback {
    
    // MARK: - Distance Operations
    
    /// Compute Euclidean distance using Accelerate, with exceptional range rescue.
    /// See EuclideanRangePolicyTests and docs/stability/DISTANCE-RANGE-CONTRACT.md.
    public static func euclideanDistance(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }

        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return finalizeEuclidean(result, a, b)
    }

    /// Shared range rescue for rooted L2; inputs must have matching dimensions.
    /// EuclideanRangePolicyTests covers CPU callers and equivalent GPU results.
    static func finalizeEuclidean(_ squaredSum: Float, _ a: [Float], _ b: [Float]) -> Float {
        if squaredSum.isFinite && squaredSum >= Float.leastNormalMagnitude {
            return sqrt(squaredSum)
        }
        // Convert BEFORE subtraction. Float input differences/products fit Double's range.
        var sum: Double = 0
        for i in a.indices {
            let diff = Double(a[i]) - Double(b[i])
            sum += diff * diff
        }
        return Float(sqrt(sum))
    }

    /// Compute cosine similarity using Accelerate
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        return cosineSimilarityCore(a, b)
    }

    /// Shared cosine-similarity core (AUDIT-2 VA2-008/VA2-009), matching the GPU cosine
    /// kernels' semantics so the silent-fallback legs stay interchangeable:
    /// * primary path — single-precision vDSP accumulation, taken whenever the dot product is
    ///   finite and `√‖a‖²·√‖b‖²` is a *normal* float (finite, nonzero, not subnormal);
    /// * rescue path — Double accumulation for everything else: overflowed accumulators
    ///   (components ≳1e19), zero/subnormal norms, and non-finite inputs. Mirrors the GPU
    ///   kernels' pre-scaled rescue (`va_cosine_rescaled_terms`); Double has the range to make
    ///   the same cases exact here;
    /// * NaN inputs propagate as NaN — the pre-audit `norm > 0` guard silently collapsed
    ///   NaN-bearing pairs to similarity 0 while the GPU propagated NaN (VA2-009);
    /// * zero vectors → similarity 0; result clamped to [-1, 1], NaN-preserving.
    static func cosineSimilarityCore(_ a: [Float], _ b: [Float], queryNormSq: Float? = nil) -> Float {
        guard a.count == b.count else { return .nan }   // ragged pair (previously an OOB vDSP read)
        guard !a.isEmpty else { return 0 }              // degenerate-by-construction, matches GPU

        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        var aa: Float
        if let precomputed = queryNormSq {
            aa = precomputed
        } else {
            aa = 0
            vDSP_svesq(a, 1, &aa, vDSP_Length(a.count))
        }
        var bb: Float = 0
        vDSP_svesq(b, 1, &bb, vDSP_Length(b.count))

        let denominator = aa.squareRoot() * bb.squareRoot()
        if dot.isFinite && aa.isNormal && bb.isNormal && denominator.isNormal {
            // Primary path: every operand is well-scaled and full-precision; raw cannot be NaN.
            // A *subnormal* accumulator (not just zero) also routes to the rescue: it carries as
            // few as ~10 mantissa bits, which visibly degrades the similarity — and it mirrors
            // the GPU, where flush-to-zero turns the same accumulator into the rescue trigger.
            let raw = dot / denominator
            return min(max(raw, -1), 1)
        }

        // Rescue: Double accumulation.
        var dotD = 0.0
        var aaD = 0.0
        var bbD = 0.0
        var aMaxAbs: Float = 0
        var bMaxAbs: Float = 0
        for i in 0..<a.count {
            let x = Double(a[i])
            let y = Double(b[i])
            dotD += x * y
            aaD += x * x
            bbD += y * y
            aMaxAbs = max(aMaxAbs, abs(a[i]))
            bMaxAbs = max(bMaxAbs, abs(b[i]))
        }
        if dotD.isNaN || aaD.isNaN || bbD.isNaN { return .nan }
        // Library-wide degenerate policy (matches the normalize family and the GPU kernels):
        // a vector whose largest magnitude is subnormal is not a representable operand on the
        // GPU — flush-to-zero erases it before any arithmetic — so BOTH legs classify it as
        // degenerate (similarity 0) rather than diverging. Double could compute a value here;
        // interchangeability of the silent-fallback legs wins (AUDIT-2 VA2-008).
        guard aMaxAbs >= Float.leastNormalMagnitude, bMaxAbs >= Float.leastNormalMagnitude else {
            return 0
        }
        let denominatorD = aaD.squareRoot() * bbD.squareRoot()
        guard denominatorD > 0 else { return 0 }
        let raw = dotD / denominatorD
        return Float(min(max(raw, -1), 1))
    }
    
    /// Compute dot product using Accelerate. FP32 product/accumulation range and
    /// cancellation limits apply; see docs/stability/DISTANCE-RANGE-CONTRACT.md.
    public static func dotProduct(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }

        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Compute Manhattan distance using Accelerate
    public static func manhattanDistance(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        guard !a.isEmpty else { return 0 }  // withUnsafeTemporaryAllocation(capacity: 0) base may be nil
        
        // Compute absolute differences in a scratch buffer instead of a heap [Float] per
        // call — under concurrent batch use that per-call malloc was pure lock contention.
        var result: Float = 0
        withUnsafeTemporaryAllocation(of: Float.self, capacity: a.count) { diff in
            let base = diff.baseAddress!
            vDSP_vsub(b, 1, a, 1, base, 1, vDSP_Length(a.count))
            vDSP_vabs(base, 1, base, 1, vDSP_Length(a.count))
            vDSP_sve(base, 1, &result, vDSP_Length(a.count))
        }
        return result
    }
    
    // MARK: - Vector Operations
    
    /// Normalize a vector using Accelerate
    ///
    /// Uses the Kahan pre-scaled algorithm (see ``StableNormalization``), so the
    /// result matches VectorCore's CPU normalization — and VectorAccelerate's
    /// Metal kernels — for subnormal and huge-magnitude inputs alike.
    ///
    /// - Returns: `v / ‖v‖₂`, or `v` unchanged when the vector cannot be normalized:
    ///   the zero vector, `‖v‖₂ ≤ 2^-127` (deep subnormal), or a non-finite component.
    public static func normalize(_ vector: [Float]) -> [Float] {
        StableNormalization.normalizedAccelerate(vector)
    }
    
    /// Add two vectors using Accelerate
    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Subtract two vectors using Accelerate
    public static func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Scale a vector using Accelerate
    public static func scale(_ vector: [Float], by scalar: Float) -> [Float] {
        var result = [Float](repeating: 0, count: vector.count)
        var s = scalar
        vDSP_vsmul(vector, 1, &s, &result, 1, vDSP_Length(vector.count))
        return result
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix-vector multiplication using Accelerate
    public static func matrixVectorMultiply(
        matrix: [Float],
        vector: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        guard matrix.count == rows * columns && vector.count == columns else { return [] }
        
        var result = [Float](repeating: 0, count: rows)

        // Use vDSP for matrix-vector multiplication (treating vector as Nx1 matrix)
        // vDSP_mmul performs C = A * B where:
        // A is matrix (rows x columns), B is vector (columns x 1), C is result (rows x 1)
        vDSP_mmul(
            matrix,           // Input matrix A
            1,                // Stride for A
            vector,           // Input vector B (as column matrix)
            1,                // Stride for B
            &result,          // Output C
            1,                // Stride for C
            vDSP_Length(rows),     // M: rows in A (and C)
            vDSP_Length(1),        // N: columns in B (and C) = 1
            vDSP_Length(columns)   // P: columns in A = rows in B
        )

        return result
    }
    
    /// Matrix multiplication using Accelerate
    public static func matrixMultiply(
        a: [Float],
        b: [Float],
        rowsA: Int,
        colsA: Int,
        colsB: Int
    ) -> [Float] {
        let rowsB = colsA  // Must match for multiplication
        guard a.count == rowsA * colsA && b.count == rowsB * colsB else { return [] }
        
        var result = [Float](repeating: 0, count: rowsA * colsB)

        // Use vDSP for matrix multiplication
        // vDSP_mmul performs C = A * B where:
        // A is (rowsA x colsA), B is (colsA x colsB), C is (rowsA x colsB)
        vDSP_mmul(
            a,                      // Input matrix A
            1,                      // Stride for A
            b,                      // Input matrix B
            1,                      // Stride for B
            &result,                // Output matrix C
            1,                      // Stride for C
            vDSP_Length(rowsA),     // M: rows in A (and C)
            vDSP_Length(colsB),     // N: columns in B (and C)
            vDSP_Length(colsA)      // P: columns in A = rows in B
        )

        return result
    }
    
    /// Matrix transpose using Accelerate
    public static func transpose(
        matrix: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        guard matrix.count == rows * columns else { return [] }
        
        var result = [Float](repeating: 0, count: rows * columns)
        vDSP_mtrans(matrix, 1, &result, 1, vDSP_Length(columns), vDSP_Length(rows))
        return result
    }
    
    // MARK: - Batch Operations
    
    /// Batch Euclidean distance using Accelerate
    public static func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { (try? euclideanDistance(query, $0)) ?? .infinity }
    }
    
    /// Batch cosine similarity using Accelerate.
    ///
    /// Delegates to ``cosineSimilarityCore(_:_:queryNormSq:)`` per candidate with the query's
    /// squared norm precomputed once. The pre-audit implementation short-circuited on a "zero"
    /// query norm — which was also true for NaN/overflowed query norms, silently returning 0
    /// for every candidate (AUDIT-2 VA2-009) — and read out of bounds for ragged candidates.
    public static func batchCosineSimilarity(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        var queryNormSq: Float = 0
        vDSP_svesq(query, 1, &queryNormSq, vDSP_Length(query.count))
        return candidates.map { cosineSimilarityCore(query, $0, queryNormSq: queryNormSq) }
    }
    
    /// Batch dot product using Accelerate
    public static func batchDotProduct(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { (try? dotProduct(query, $0)) ?? 0 }
    }
    
    // MARK: - Statistical Operations
    
    /// Compute mean using Accelerate
    public static func mean(_ vector: [Float]) -> Float {
        var result: Float = 0
        vDSP_meanv(vector, 1, &result, vDSP_Length(vector.count))
        return result
    }
    
    /// Compute variance using Accelerate
    public static func variance(_ vector: [Float]) -> Float {
        let meanValue = mean(vector)
        
        // Subtract mean from each element
        var centered = [Float](repeating: 0, count: vector.count)
        var negativeMean = -meanValue
        vDSP_vsadd(vector, 1, &negativeMean, &centered, 1, vDSP_Length(vector.count))
        
        // Square the differences
        var squared = [Float](repeating: 0, count: vector.count)
        vDSP_vsq(centered, 1, &squared, 1, vDSP_Length(vector.count))
        
        // Sum and divide by count
        var sum: Float = 0
        vDSP_sve(squared, 1, &sum, vDSP_Length(vector.count))
        
        return sum / Float(vector.count)
    }
    
    /// Compute standard deviation using Accelerate
    public static func standardDeviation(_ vector: [Float]) -> Float {
        sqrt(variance(vector))
    }
    
    // MARK: - Element-wise Operations
    
    /// Element-wise multiplication using Accelerate
    public static func elementwiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vmul(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Element-wise division using Accelerate
    public static func elementwiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vdiv(b, 1, a, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Apply function element-wise using Accelerate
    public static func applyFunction(_ vector: [Float], function: (Float) -> Float) -> [Float] {
        vector.map(function)
    }
    
    // MARK: - Performance Utilities
    
    /// Check if Accelerate should be used based on vector size
    public static func shouldUseAccelerate(for size: Int) -> Bool {
        // Accelerate is efficient for vectors larger than 32 elements
        size >= 32
    }
    
    /// Optimal chunk size for batch processing
    public static func optimalChunkSize(for totalSize: Int) -> Int {
        // Balance between cache efficiency and parallelism
        let cacheLineSize = 64  // bytes
        let floatSize = MemoryLayout<Float>.size
        let elementsPerCacheLine = cacheLineSize / floatSize
        
        // Use multiples of cache line size
        let baseChunkSize = elementsPerCacheLine * 64  // 1024 elements
        
        if totalSize <= baseChunkSize {
            return totalSize
        }
        
        // Divide into reasonable chunks
        let numChunks = (totalSize + baseChunkSize - 1) / baseChunkSize
        return totalSize / numChunks
    }
}

// MARK: - Accelerate Extensions

extension Array where Element == Float {
    /// Normalize this vector using Accelerate
    public func normalizedAccelerate() -> [Float] {
        AccelerateFallback.normalize(self)
    }
    
    /// Compute dot product with another vector using Accelerate
    public func dotAccelerate(with other: [Float]) throws -> Float {
        try AccelerateFallback.dotProduct(self, other)
    }

    /// Compute Euclidean distance to another vector using Accelerate
    public func euclideanDistanceAccelerate(to other: [Float]) throws -> Float {
        try AccelerateFallback.euclideanDistance(self, other)
    }

    /// Compute cosine similarity with another vector using Accelerate
    public func cosineSimilarityAccelerate(with other: [Float]) throws -> Float {
        try AccelerateFallback.cosineSimilarity(self, other)
    }
}