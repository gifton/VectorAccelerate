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
/// Numerically identical to VectorAccelerate's Metal kernels (`BasicOperations.metal`,
/// `L2Normalization.metal`) — the constants and the expression order are shared:
///
/// 1. `maxAbs = max |v_i|`
/// 2. `den   = min(max(maxAbs, 0x1p-126), 0x1p126)` — clamped on both sides so
///    `scale` is always a *normal* float. The lower bound is `FLT_MIN`
///    (`Float.leastNormalMagnitude`), deliberately not `leastNonzeroMagnitude`:
///    below it `1/den` would overflow. Above `2^126`, `1/den` would be subnormal —
///    harmless on the CPU, but flushed to zero by the GPU's default math mode, so
///    both sides clamp identically to keep the results bit-comparable.
/// 3. `scale = 1 / den`
/// 4. `sumSq = Σ (v_i · scale)²` — every term is `≤ 16`, so the accumulation is
///    finite for any finite input (a naive `Σ v²` overflows to `+∞` for `|v| > 1e19`
///    and underflows to `0` for subnormal `v`).
/// 5. `sNorm = √sumSq = ‖v‖ · scale`
/// 6. `out   = (v · scale) / sNorm` — algebraically `v / ‖v‖` — taken only when
///    `sNorm > 0.5`. A *division*, matching the kernels' `precise::divide`: the
///    equivalent multiply by `1/sNorm` is reassociable by the Metal compiler back
///    into the subnormal factor `scale/sNorm`, so both sides divide.
///
/// `1/‖v‖` is never formed: it is subnormal for every `‖v‖ > 2^126` and does not
/// exist at all for `‖v‖ > greatestFiniteMagnitude`, yet such vectors are perfectly
/// normalizable in the pre-scaled domain (`1/sNorm ∈ (0, 2]` and `|v·scale| ≤ 4`
/// are both normal). Representability of `1/‖v‖` is still the *low-end* criterion:
/// it holds exactly when `‖v‖ > 2^-128`, i.e. `sNorm > 0.25`; the guard uses `0.5`,
/// one binade stricter, to keep `1/sNorm ≤ 2`. When `den == maxAbs` the largest
/// scaled component is `±1`, so `sNorm ≥ 1` and the guard always passes; it can only
/// fail for the zero vector and for vectors whose largest magnitude is subnormal.
///
/// - Note: Degenerate inputs (guard failed) are returned **unchanged**, matching
///   `VectorCore.NormalizeKernels.normalizeUnchecked`, which leaves the buffer
///   untouched rather than scaling every element by `Inf`/`NaN`. For the zero
///   vector "unchanged" and "zeroed" coincide.
/// - Note: Two *deliberate* deviations from `NormalizeKernels.normalizeUnchecked`,
///   both cases where it is demonstrably wrong: it reconstructs `‖v‖` with the
///   *unclamped* `maxAbs` (yielding a result of norm 1.1755 when `maxAbs` is
///   subnormal), and for `‖v‖ > greatestFiniteMagnitude` its `mag` overflows to
///   `+∞`, whose reciprocal `0` passes its `isFinite` guard and zeroes the whole
///   vector. This implementation returns the correct unit vector in both cases.
/// - Complexity: `O(n)` in three passes (max, scaled squares, scale).
@usableFromInline
internal enum StableNormalization {

    /// Lower pre-scale clamp: the smallest positive *normal* float, `0x1p-126`.
    /// Mirrors `VA_NORM_MIN_DENOM` in `Metal4Common.h`.
    @usableFromInline
    internal static let minDenominator: Float = .leastNormalMagnitude

    /// Upper pre-scale clamp, `0x1p126` — keeps `1/den` normal (see the type doc).
    /// Mirrors `VA_NORM_MAX_DENOM` in `Metal4Common.h`.
    @usableFromInline
    internal static let maxDenominator: Float = 0x1p126

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

        let den = Swift.min(Swift.max(maxAbs, minDenominator), maxDenominator)
        var scale = 1.0 / den

        // Pass 2: sumSq = Σ (v_i · scale)²  (scratch holds the pre-scaled vector)
        var scratch = [Float](repeating: 0, count: count)
        vDSP_vsmul(vector, 1, &scale, &scratch, 1, n)

        var sumSquares: Float = 0
        vDSP_svesq(scratch, 1, &sumSquares, n)

        let scaledNorm = sqrt(sumSquares)
        guard scaledNorm > minScaledNorm else { return vector }

        // Pass 3: out = (v · scale) / sNorm ≡ v / ‖v‖. Dividing the already
        // pre-scaled buffer keeps every operand in the normal range (‖v‖ itself is
        // never inverted, and may not even be representable).
        var divisor = scaledNorm
        vDSP_vsdiv(scratch, 1, &divisor, &scratch, 1, n)
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

            den = Swift.min(Swift.max(maxAbs, minDenominator), maxDenominator)
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

        // Pass 3: out = (v · scale) / sNorm ≡ v / ‖v‖ (see the type doc)
        let scaleVec = SIMD8<Float>(repeating: scale)
        let divisorVec = SIMD8<Float>(repeating: scaledNorm)

        var result = [Float](repeating: 0, count: count)
        vector.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                for i in 0..<full {
                    let offset = i * width
                    let v = UnsafeRawPointer(s + offset).loadUnaligned(as: SIMD8<Float>.self)
                    let scaled = (v * scaleVec) / divisorVec
                    for j in 0..<width { d[offset + j] = scaled[j] }
                }
                for i in tailStart..<count {
                    d[i] = (s[i] * scale) / scaledNorm
                }
            }
        }
        return result
    }
}
