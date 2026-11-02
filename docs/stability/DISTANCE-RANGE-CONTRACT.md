# Distance range contract

VA3-030 policy, approved 2026-09-07. Outputs remain Float32. Direct rooted L2 rescues
exceptional accumulator range failures; squared L2 and dot products retain their FP32
range and precision limits. This policy does not change Minkowski's explicit fast mode.

## Direct Euclidean distance

The ordinary path accumulates squared differences in Float32 and takes the square root.
When the accumulated value is nonfinite, subnormal, or zero, these entry points recompute:

| Surface | Rescue |
|---|---|
| Metal `euclideanDistance`, `batchEuclideanDistance` | Scaled differences on GPU |
| Metal `l2_distance`, `l2_distance_kernel`, `soa_l2_distance` with square root enabled | Same GPU helper; squared mode bypasses it |
| `AccelerateFallback`, `FallbackProvider`, both `SIMDFallback` configurations | Double recomputation, converted to Float32 after the root |
| `Metal4ComputeEngine` single/batch and `BatchDistanceEngine` CPU/GPU routes | The corresponding shared CPU/GPU rescue |
| `L2KernelDistanceProvider`, direct SoA scoring | Root-enabled kernels above |
| `MemoryMapManager` Euclidean search and `QuantizationEngine` CPU centroid-distance helper | Shared CPU finalizer |

GPU rescue scans the maximum absolute difference, accumulates squared normalized
differences with precise division, and rescales using a significand/exponent split.
CPU rescue converts operands to Double before subtraction. No extra buffers or dispatches
are needed; reduction kernels perform the exceptional scan after their existing barriers.
The normal finite accumulator path retains its existing arithmetic.

For example, `[0, 0]` versus `[3e20, 4e20]` returns approximately `5e20` rather than an
infinity caused by squaring. The analogous `3e-30, 4e-30` inputs retain their tiny distance
rather than losing it when their squares underflow. A truly unrepresentable final distance
still returns infinity. NaN differences propagate, including when another difference is
infinite; a real infinite difference without NaN returns infinity.

This is range rescue, not an all-input accuracy guarantee. Normal finite totals retain
Float32 accumulation error. GPU flush-to-zero can still erase subnormal operands or
differences; the Double CPU rescue can preserve cases the GPU cannot. Ill-conditioned
inputs, device/compiler math behavior, and precision lost upstream remain limitations.
No new performance bound or bitwise CPU/GPU equivalence is promised.

## Squared L2

Squared outputs deliberately retain their current computation and representation.
A difference of `1e20` has a squared distance near `1e40`, beyond Float32's maximum, even
though its rooted distance fits. Squared-mode outputs can therefore be infinity.

For dimension D and maximum absolute component difference M, `D * M² < FLT_MAX` is a
conservative upper-range condition in exact arithmetic; callers should leave rounding
margin. Small squared terms can also underflow or flush to zero. This is a documented
range condition, not runtime validation or rejection of out-of-range vectors.

Fused L2 Top-K, IVF/index search, and other APIs that compute or select squared scores
retain this contract. Taking the square root of an already overflowed squared score does
not recover the distance or restore ordering lost before selection. This includes rooted
presentations in clustering, `ProductQuantizationKernel`, `HDBSCANDistanceModule`, and
`SearchResult`. No wider score representation or search-algorithm change is part of this
slice.

## Dot product

Products and partial sums remain Float32, with the existing SIMD/FMA/reduction order.
No cancellation rescue or higher-precision GPU accumulator is added. A conservative
upper-range condition is `sum(abs(a[i] * b[i])) < FLT_MAX` in exact arithmetic, with
rounding margin; `D * max(abs(a)) * max(abs(b)) < FLT_MAX` is a simpler sufficient bound.

Opposing terms can each overflow even when the exact final dot product fits. For example,
`[1e20, 1e20] · [1e20, -1e20]` is mathematically zero but can produce a nonfinite FP32
result. Outside the range condition, no particular NaN/infinity result is promised.
Even within range, cancellation can cause substantial relative error. Rescaling alone
would not establish a strong cancellation-accuracy contract.

## Other numerical pipelines

Minkowski retains the separate VA3-022 policy: explicit `useStableComputation: false`
keeps intermediate range limits; automatic routing remains p > 10 to stable. Learned
projection distances (`LearnedDistance.metal`) retain their projection and squared-sum
limits, including their optional rooted outputs. They are not covered by the direct-L2
rescue above. Metric-specific transforms, weighted distances, normalization, and RMSE
are not made range-safe by this change.

## Verification

`EuclideanRangePolicyTests` exercises actual GPU buffers in both shader compilation paths,
CPU fallback variants, forced batch routes, and mapped search. Cases include huge/tiny
normal differences, subnormal squared accumulators, finite endpoints, common offsets,
scalar tails, ragged reduction widths, SoA lane strides, zero/ordinary controls, NaN and
infinity, unchanged squared-mode overflow, and dot-product cancellation outside range.
Full-suite regression gates supplement these boundary tests.
