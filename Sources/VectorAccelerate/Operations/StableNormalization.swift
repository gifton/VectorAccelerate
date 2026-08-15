//
//  StableNormalization.swift
//  VectorAccelerate
//
//  Single CPU implementation of VectorAccelerate's L2 normalization policy.
//  Shared by every CPU fallback (`AccelerateFallback`, `SIMDFallback`,
//  `FallbackProvider`) so the CPU paths cannot drift from each other or from the
//  Metal kernels in `BasicOperations.metal` / `L2Normalization.metal`.
//

import Foundation
import Accelerate
import simd

/// L2 normalization with Kahan pre-scaling (BE3 §4.4 parity with VectorCore).
///
/// Mirrors `VectorCore.NormalizeKernels` and the GPU kernels exactly:
///
/// 1. `maxAbs = max |v_i|`
/// 2. `den   = max(maxAbs, Float.leastNormalMagnitude)` — the *normal* `FLT_MIN`
///    (`0x1p-126`), deliberately not `leastNonzeroMagnitude`: it is the clamp that
///    keeps `1/den` finite (`≤ 2^126`) for subnormal-dominated vectors.
/// 3. `scale = 1 / den`
/// 4. `sumSq = Σ (v_i · scale)²` — every term is `≤ 1`, so the accumulation is
///    finite for any finite input (a naive `Σ v²` overflows to `+∞` for `|v| > 1e19`
///    and underflows to `0` for subnormal `v`).
/// 5. `sNorm = √sumSq = ‖v‖ / den`
/// 6. `norm  = den · sNorm = ‖v‖` — exact reconstruction, cannot overflow.
/// 7. `out   = v · (scale / sNorm)` — algebraically `v / ‖v‖` — taken only when
///    `sNorm > 0.5`.
///
/// Because `den ≥ 2^-126`, `1/‖v‖` is representable exactly when `‖v‖ > 2^-128`,
/// i.e. `sNorm > 0.25`; the guard uses `0.5` (one binade stricter) so the
/// reciprocal `scale / sNorm ≤ 2^127` keeps a factor of two below
/// `greatestFiniteMagnitude` and no subnormal intermediate is ever formed — the
/// same expression the Metal kernels use, where a subnormal intermediate would be
/// flushed to zero and then inverted to `+Inf`. When `den == maxAbs` the largest
/// scaled component is `±1`, so `sNorm ≥ 1` and the guard always passes; it can only
/// fail for the zero vector and for vectors whose largest magnitude is subnormal.
///
/// - Note: Degenerate inputs (guard failed) are returned **unchanged**, matching
///   `VectorCore.NormalizeKernels.normalizeUnchecked`, which leaves the buffer
///   untouched rather than scaling every element by `Inf`/`NaN`. For the zero
///   vector "unchanged" and "zeroed" coincide.
/// - Complexity: `O(n)` in three passes (max, scaled squares, scale).
@usableFromInline
internal enum StableNormalization {

    /// Pre-scale clamp: the smallest positive *normal* float, `0x1p-126`.
    @usableFromInline
    internal static let minDenominator: Float = .leastNormalMagnitude

    /// Guard on `‖v‖ / den`; see the type documentation for the derivation.
    /// Mirrors `VA_NORM_MIN_SCALED` in `Metal4Common.h`.
    @usableFromInline
    internal static let minScaledNorm: Float = 0.5

    /// Accelerate (vDSP) implementation of the policy above.
    @usableFromInline
    internal static func normalizedAccelerate(_ vector: [Float]) -> [Float] {
        let count = vector.count
        guard count > 0 else { return vector }
        let n = vDSP_Length(count)

        // Pass 1: maxAbs = max |v_i|
        var maxAbs: Float = 0
        vDSP_maxmgv(vector, 1, &maxAbs, n)

        let den = Swift.max(maxAbs, minDenominator)
        var scale = 1.0 / den

        // Pass 2: sumSq = Σ (v_i · scale)²  (scratch doubles as the output buffer)
        var scratch = [Float](repeating: 0, count: count)
        vDSP_vsmul(vector, 1, &scale, &scratch, 1, n)

        var sumSquares: Float = 0
        vDSP_svesq(scratch, 1, &sumSquares, n)

        let scaledNorm = sqrt(sumSquares)
        guard scaledNorm > minScaledNorm else { return vector }

        // Pass 3: out = v · (1 / ‖v‖), formed as scale / sNorm (no subnormal
        // intermediate — see the type documentation).
        var invNorm = scale / scaledNorm
        vDSP_vsmul(vector, 1, &invNorm, &scratch, 1, n)
        return scratch
    }

    /// Pure-SIMD implementation of the policy above (no Accelerate dependency).
    @usableFromInline
    internal static func normalizedSIMD(_ vector: [Float]) -> [Float] {
        let count = vector.count
        guard count > 0 else { return vector }

        let width = 8
        let full = count / width
        let tailStart = full * width

        var maxAbs: Float = 0
        var sumSquares: Float = 0
        var scale: Float = 0
        var den: Float = 0

        vector.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }

            // Pass 1: maxAbs
            var maxVec = SIMD8<Float>.zero
            for i in 0..<full {
                let v = UnsafeRawPointer(base + i * width).loadUnaligned(as: SIMD8<Float>.self)
                // |v| elementwise: max(v, -v) (avoids an ambiguous `abs` overload)
                maxVec = pointwiseMax(maxVec, pointwiseMax(v, -v))
            }
            maxAbs = maxVec.max()
            for i in tailStart..<count {
                maxAbs = Swift.max(maxAbs, Swift.abs(base[i]))
            }

            den = Swift.max(maxAbs, minDenominator)
            scale = 1.0 / den

            // Pass 2: sumSq of the pre-scaled vector
            let scaleVec = SIMD8<Float>(repeating: scale)
            var sumVec = SIMD8<Float>.zero
            for i in 0..<full {
                let s = UnsafeRawPointer(base + i * width).loadUnaligned(as: SIMD8<Float>.self) * scaleVec
                sumVec += s * s
            }
            sumSquares = sumVec.sum()
            for i in tailStart..<count {
                let s = base[i] * scale
                sumSquares += s * s
            }
        }

        let scaledNorm = sqrt(sumSquares)
        guard scaledNorm > minScaledNorm else { return vector }

        // Pass 3: out = v · (1 / ‖v‖), formed as scale / sNorm
        let invNorm = scale / scaledNorm
        let invVec = SIMD8<Float>(repeating: invNorm)

        var result = [Float](repeating: 0, count: count)
        vector.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                for i in 0..<full {
                    let offset = i * width
                    let v = UnsafeRawPointer(s + offset).loadUnaligned(as: SIMD8<Float>.self)
                    let scaled = v * invVec
                    for j in 0..<width { d[offset + j] = scaled[j] }
                }
                for i in tailStart..<count {
                    d[i] = s[i] * invNorm
                }
            }
        }
        return result
    }
}
