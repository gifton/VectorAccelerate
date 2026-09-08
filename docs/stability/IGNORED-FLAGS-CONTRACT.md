# Neural activation and learned-distance normalization

VA3-027 covers two existing flags that optimized GPU paths silently ignored.
This slice makes those paths honor the flags without changing Swift signatures.

`NeuralQuantizationKernel.encodeTiledV3` now forwards the dispatch parameters'
`useActivation` value to `neural_encode_pass1`. Pass 1 computes the affine projection,
including any bias, then applies ReLU only when the flag is nonzero. Disabled activation
preserves negative intermediate coordinates and allows negative INT8 codes from pass 2.
The configuration used to load weights does not override the dispatch flag.

**Raw shader migration:** `neural_encode_pass1` requires a new UInt32 constant at
**buffer(8)**: zero disables activation, nonzero enables ReLU. Buffers 0–7 retain their
meanings, and `NeuralQuantizationParameters` keeps its existing layout. Raw callers must
supply the new binding even when requesting the previous always-ReLU behavior. Pass 2 additionally requires the normalization flag at buffer(5), added in slice 23;
see [the normalization contract](NEURAL-LATENT-NORMALIZATION-CONTRACT.md). The pass-1 threadgroup remains exactly
256 threads; dense layouts, valid dimensions, sufficient buffers and synchronization
remain caller requirements.

The specialized `learned_l2_768_to_128_kernel` and `learned_l2_384_to_64_kernel` now honor
`normalizeProjected`. When enabled, each kernel materializes both complete projections,
normalizes them with the same helper as the general learned-L2 kernel, then accumulates
squared component differences. `computeSqrt` still independently selects rooted or
squared distance. Disabled normalization keeps the existing fused projection/distance
loop. Learned-distance buffer bindings and parameter layouts are unchanged.

Normalization retains the existing general-path epsilon and FP32 range policy: vectors
whose computed norm is at or below VA_EPSILON are scaled to zero. This is not the robust
full-range normalization used by some other package surfaces; projection/norm overflow,
underflow, nonfinite inputs and fast-math limitations are not changed here. See the
[distance range contract](DISTANCE-RANGE-CONTRACT.md). Specialized learned kernels retain
their fixed, densely packed input dimensions; this slice does not broaden stride support.
Normalized paths need complete temporary projections; no performance improvement is
claimed.

`Hardening/IgnoredFlagTests.swift` exercises both bundled and runtime shader libraries.
Analytic fixtures cover activation on/off, bias on/off, ragged input/latent/batch tiles,
over-dispatch and output canaries. Public tiled encoding checks signed codes, scales and
per-dispatch flag forwarding. Learned-distance tests cover both specialized shapes and
the general kernel, normalization on/off, rooted/squared output, zero and tiny projected
vectors, parallel/opposite directions, and padded output rows. Public compute calls
exercise the specialized selection paths. This adds the previously missing dedicated
`encodeTiledV3` test coverage.

The adjacent `normalizeLatent` neural-quantization omission was fixed in slice 23 for
specialized and tiled quantized encoders. See the normalization contract for the new
raw pass-2 flag, enabled-path code/scale semantics and retained disabled behavior.
High-level preservation of per-vector scales remains separate reconstruction debt.
VA3-019's atomic accumulation policy also remains open.
