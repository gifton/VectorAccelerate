# MinkowskiDistance Kernel Stability Fix - Agent Kickoff

## Context

VectorAccelerate is a GPU-accelerated vector operations library for Swift using Metal. During a numerical stability audit, the MinkowskiDistance kernel was identified as having **CRITICAL** overflow and precision issues that need to be fixed before the 1.0 release.

## Problem Summary

The MinkowskiDistance kernel computes Lp-norm distances: `distance = (Σ|x_i - y_i|^p)^(1/p)`

Three numerical stability issues were identified:

### Issue 1: Power Operation Overflow
**Location**: `Sources/VectorAccelerate/Metal/Shaders/MinkowskiDistance.metal`, lines 25-36

```metal
inline float fast_pow_positive(float base, float p) {
    return exp(log(base) * p);  // Can overflow for large base and p
}
```

For `base=10000, p=5`: `exp(log(10000)*5) = exp(46) ≈ 1e20` - overflows for larger values.

### Issue 2: Accumulation Without Compensation
**Location**: Lines 212-213

```metal
float4 powered = pow4(diff, p);
accumulator += powered.x + powered.y + powered.z + powered.w;
```

When differences have mixed scales (e.g., `1e6` and `1e-6`), the small terms are lost entirely due to floating-point precision limits. No Kahan summation or log-space accumulation is used.

### Issue 3: Final Root Computation
**Location**: Lines 243-245

If the accumulator overflowed or underflowed during summation, the final `safe_pow(accumulator, inv_p)` produces incorrect results.

## Relevant Files

| File | Purpose |
|------|---------|
| `Sources/VectorAccelerate/Metal/Shaders/MinkowskiDistance.metal` | Metal shader with the bug |
| `Sources/VectorAccelerate/Kernels/Metal4/MinkowskiDistanceKernel.swift` | Swift kernel wrapper |
| `Tests/VectorAccelerateTests/NumericalStabilityTests.swift` | Test suite with `MinkowskiStabilityTests` class |
| `docs/stability/NUMERICAL_STABILITY_FINDINGS.md` | Full findings document |

## Recommended Approach

The stable approach uses **normalization** to prevent overflow:

```
// Instead of: sum(|diff|^p)^(1/p)
// Use: max_diff * (sum((|diff|/max_diff)^p))^(1/p)
```

This is mathematically equivalent but keeps all intermediate values in [0, 1] range.

**Note**: The shader already has a `minkowski_distance_stable` kernel (lines 257-336) that attempts this approach, but it appears incomplete (tile loading code is stubbed). Investigate whether to:
1. Complete the `minkowski_distance_stable` implementation
2. Fix the main `minkowski_distance_batch` kernel
3. Both (stable as default, fast as opt-in)

## Proposed Fix Pattern

```metal
inline float safe_pow_bounded(float x, float p) {
    x = abs(x);
    if (x < 1e-10f) return 0.0f;
    if (x > 1e4f) x = 1e4f;  // Clamp to prevent overflow
    if (p == 1.0f) return x;
    if (p == 2.0f) return x * x;
    return exp(clamp(p * log(x), -100.0f, 100.0f));
}
```

Or use the two-pass normalized approach:
1. Pass 1: Find max difference across all dimensions
2. Pass 2: Compute sum of normalized powered differences
3. Denormalize final result

## Validation

Run the existing stability tests to validate fixes:

```bash
swift test --filter MinkowskiStabilityTests
```

Key test cases to pass:
- `testMinkowskiLargeP_OverflowRisk` - p=3 with values of 1000
- `testMinkowskiP5_ExtremeOverflowRisk` - p=5 with values of 100
- `testMinkowskiMixedScales` - mixed 1e6 and 1e-6 differences
- `testMinkowskiFractionalP` - p=0.5 (fractional powers)

## Deliverables

1. **Analysis**: Document which approach (clamping vs normalization vs both) is best
2. **Metal Fix**: Update `MinkowskiDistance.metal` with stable implementation
3. **Swift Integration**: Ensure `MinkowskiDistanceKernel.swift` exposes stability options if needed
4. **Test Validation**: All `MinkowskiStabilityTests` pass
5. **Additional Tests**: Add any edge cases discovered during implementation

## Constraints

- Maintain backwards compatibility with existing API
- Keep performance impact minimal for the common case (p=1, p=2 are already optimized)
- Follow existing code patterns in the codebase
- Use Metal 4 / MSL 4.0 features as appropriate (target macOS 26+)

## Getting Started

1. Read the Metal shader: `Sources/VectorAccelerate/Metal/Shaders/MinkowskiDistance.metal`
2. Review the existing `minkowski_distance_stable` stub
3. Run current tests to see baseline: `swift test --filter MinkowskiStabilityTests`
4. Implement fix
5. Validate with tests
6. Update `docs/stability/NUMERICAL_STABILITY_FINDINGS.md` with completion status

---

*Reference: docs/stability/NUMERICAL_STABILITY_FINDINGS.md*
