# Numerical Stability Findings

This document tracks critical numerical stability issues identified in VectorAccelerate kernels, along with remediation tasks for the 1.0 release.

## Status Legend

- [ ] Not started
- [x] Completed
- [~] In progress

---

## Critical Findings

### 1. MinkowskiDistance Kernel - CRITICAL

**File**: `Sources/VectorAccelerate/Metal/Shaders/MinkowskiDistance.metal`

**Risk Level**: HIGH

**Issues Identified**:

1. **Power Operation Overflow** (Lines 25-36)
   - `safe_pow` uses `exp(log(x) * p)` which can overflow for large bases
   - No bounds checking on intermediate result
   - Example: `x=1000, p=5` → `exp(log(1000)*5) = exp(34.5) ≈ 1e15` (fits, but `x=10000, p=5` overflows)

2. **Accumulation Without Compensation** (Lines 212-213)
   - `pow4()` can produce huge values (e.g., `100^3 = 1,000,000`)
   - Summing into accumulator with no Kahan summation or similar correction
   - Catastrophic cancellation when mixing scales (e.g., `1e6` and `1e-6` differences)

3. **Final Root Computation** (Lines 243-245)
   - `safe_pow(accumulator, inv_p)` - if accumulator overflowed, final result is wrong
   - If accumulator underflowed, result too small

**Failure Case Example**:
```
Query: [1e6, 1e-6, 1e6, 1e-6, ...]
Database: [0, 0, 0, 0, ...]
p = 3

Differences: [1e6, 1e-6, 1e6, 1e-6, ...]
Powers: [1e18, 1e-18, 1e18, 1e-18, ...]

When summing: 1e18 + 1e-18 = 1e18 (small terms lost entirely)
```

**Recommended Fix**:
```metal
// Use log-space computation for stability
// log(sum(x^p)) = log(x_max^p) + log(sum((x_i/x_max)^p))
//               = p*log(x_max) + log(sum(exp(p*log(x_i/x_max))))

inline float safe_pow_bounded(float x, float p) {
    x = abs(x);
    if (x < 1e-10f) return 0.0f;
    if (x > 1e4f) x = 1e4f;  // Clamp to prevent overflow
    if (p == 1.0f) return x;
    if (p == 2.0f) return x * x;
    return exp(clamp(p * log(x), -100.0f, 100.0f));
}
```

**Task**: [ ] Implement stable power function with clamping or log-space computation

---

### 2. UMAPGradient Kernel - HIGH RISK

**File**: `Sources/VectorAccelerate/Metal/Shaders/UMAPGradient.metal`

**Risk Level**: HIGH

**Issues Identified**:

1. **Fractional Power Overflow** (Lines 81-82)
   - `pow(max(distSq, epsilon), params.b)` where `b ≈ 0.7915`
   - For `distSq=1e6, b=0.7915`: `(1e6)^0.7915 ≈ 56000` - borderline
   - No explicit overflow check

2. **Denominator Computation** (Line 83)
   - `denom = 1.0f + params.a * distSqPowB`
   - If `distSqPowB` is very large, product could approach float limits

3. **Gradient Coefficient Division** (Line 85)
   - `gradCoeff = -2ab * distSqPowBm1 / denom`
   - `distSqPowBm1` can also be very large
   - Division by `denom` could underflow to 0

**Existing Mitigation**:
- Line 85: `gradCoeff = clamp(gradCoeff, -4.0f, 4.0f)` - helps but doesn't prevent intermediate overflow

**Recommended Fix**:
```metal
// Clamp distSq before power computation
float distSqClamped = clamp(distSq, epsilon, 1000.0f);
float distSqPowB = pow(distSqClamped, params.b);
```

**Task**: [ ] Add input validation for distance squared values before power computation

---

### 3. AttentionSimilarity Kernel - MODERATE RISK

**File**: `Sources/VectorAccelerate/Metal/Shaders/AttentionSimilarity.metal`

**Risk Level**: MODERATE

**Issues Identified**:

1. **Unstable Sigmoid** (Line 141)
   - Current: `1.0f / (1.0f + exp(-x))`
   - For `x > 20`: `exp(-20) ≈ 2e-9`, result = 1.0 exactly (precision loss)
   - For `x < -20`: `exp(20) ≈ 5e8`, result underflows toward 0

2. **Temperature Scaling** (Line 137)
   - `similarity /= params.temperature`
   - Large dot products before scaling could cause issues

**Recommended Fix**:
```metal
inline float attn_sigmoid_stable(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + exp(-x));
}
```

**Task**: [ ] Implement stable sigmoid with early exit for extreme values

---

### 4. LogSumExp Second Pass - LOW RISK (Well Protected)

**File**: `Sources/VectorAccelerate/Metal/Shaders/LogSumExp.metal`

**Risk Level**: LOW (existing guards are good)

**Current Protection**:
- Max-shift trick implemented correctly
- `-INFINITY` and `+INFINITY` checks present
- Finite value filtering before `exp()`

**Minor Concern**:
- Line 274: `localSum = partialSumExp[lid] * exp(partialMax[lid] - globalMax)`
- Could overflow if `partialMax >> globalMax` (shouldn't happen in practice)

**Task**: [ ] Verify partialMax ≤ globalMax invariant holds in all code paths

---

### 5. DotProduct High-Dimensional Accumulation - MODERATE RISK

**File**: `Sources/VectorAccelerate/Metal/Shaders/DotProduct.metal`

**Risk Level**: MODERATE (for extreme inputs)

**Issues Identified**:

- 1536D dot product with large values could overflow
- `1536 * (1e18)^2 = 1.536e39` exceeds Float32 max (~3.4e38)

**Current Mitigation**:
- FMA operations reduce intermediate rounding errors
- Multiple independent accumulators help

**Task**: [ ] Document input value range requirements for high-dimensional operations

---

## Follow-up Tasks

### Testing Gaps

- [ ] **Add tests for UMAPGradient kernel**
  - Requires checking if high-level Swift API exists
  - Test large distance squared values
  - Test fractional power edge cases

- [ ] **Add tests for AttentionSimilarity kernel**
  - Test extreme similarity scores (>20, <-20)
  - Test temperature scaling edge cases

### Stress Testing

- [ ] **Create stress test variant with longer iterations**
  - Run stability tests with 10,000+ iterations
  - Monitor for memory leaks during repeated operations
  - Check for thermal throttling effects on precision
  - Verify consistency across runs (determinism)

### Performance Validation

- [ ] **Add benchmarking for denormal edge case performance**
  - Denormal (subnormal) floats can be 10-100x slower on some hardware
  - Benchmark `DenormalStabilityTests` scenarios
  - Consider flush-to-zero mode for performance-critical paths
  - Document performance implications

---

## Test Coverage Summary

**Created**: `Tests/VectorAccelerateTests/NumericalStabilityTests.swift`

| Test Class | Kernel Coverage | Status |
|------------|-----------------|--------|
| `ZeroVectorStabilityTests` | L2, Cosine, DotProduct | [x] Created |
| `DenormalStabilityTests` | L2, Cosine, DotProduct | [x] Created |
| `OverflowUnderflowStabilityTests` | L2, DotProduct | [x] Created |
| `MinkowskiStabilityTests` | Minkowski | [x] Created |
| `LogSumExpStabilityTests` | LogSumExp | [x] Created |
| `NaNInfPropagationTests` | L2, Cosine, DotProduct | [x] Created |
| `CatastrophicCancellationTests` | L2, DotProduct | [x] Created |
| `DimensionEdgeCaseTests` | L2, Cosine, DotProduct | [x] Created |
| `CosineBoundaryTests` | Cosine | [x] Created |
| `QuantizationStabilityTests` | Scalar, Binary Quantization | [x] Created |

---

## Kernel Risk Summary

| Kernel | Risk | Issue | Fix Complexity |
|--------|------|-------|----------------|
| MinkowskiDistance | CRITICAL | Power overflow, accumulation | Medium |
| UMAPGradient | HIGH | Fractional power overflow | Low |
| AttentionSimilarity | MODERATE | Unstable sigmoid | Low |
| LogSumExp | LOW | Well protected | N/A |
| DotProduct (1536D) | MODERATE | Accumulation overflow | Documentation |
| CosineSimilarity | LOW | Well protected | N/A |
| L2Distance | LOW | Well protected | N/A |

---

## IEEE 754 Reference

Float32 properties relevant to these fixes:

| Property | Value | Relevance |
|----------|-------|-----------|
| Epsilon | ~1.19e-7 | Smallest distinguishable from 1 |
| Min Normal | ~1.18e-38 | Below this is subnormal |
| Min Subnormal | ~1.4e-45 | Absolute minimum positive |
| Max Finite | ~3.4e38 | Overflow threshold |
| exp() Overflow | ~88.7 | exp(x) = Inf for x > 88.7 |
| exp() Underflow | ~-87.3 | exp(x) = 0 for x < -87.3 |

---

## Completion Checklist

### Phase 1: Critical Fixes
- [x] MinkowskiDistance stable power function *(Completed 2026-01-09)*
  - Fixed `safe_pow` with proper exp() clamping
  - Added epsilon comparison for integer p detection
  - Added fast paths for p=3, p=4
  - All 18 Minkowski tests passing
- [x] MinkowskiDistance stable accumulation *(Completed 2026-01-09 in Phase 1b)*
- [ ] UMAPGradient input clamping
- [ ] AttentionSimilarity stable sigmoid

### Phase 1b: MinkowskiDistance Remaining Work *(Completed 2026-01-09)*
- [x] Complete `minkowski_distance_stable` kernel (tile loading is stubbed)
- [x] Wire up Swift `useStableComputation` flag to use stable kernel
- [x] Add extreme p-value tests (p=10, p=20) for accumulation overflow

### Phase 2: Test Coverage
- [ ] UMAPGradient test coverage
- [ ] AttentionSimilarity test coverage
- [ ] Stress test suite

### Phase 3: Documentation & Validation
- [ ] Input range documentation for all kernels
- [ ] Denormal performance benchmarks
- [ ] Run full test suite on target hardware

---

*Last Updated: 2026-01-09*
*Analysis performed with: NumericalStabilityTests.swift*
