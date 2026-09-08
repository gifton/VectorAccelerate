# Neural latent normalization before quantization

`normalizeLatent` applies to quantized neural encoding. The generic quantizing shader,
all three exported dimension-specialized quantizers (768→128, 768→64, 384→64), and the
tiled two-pass encoder now honor the flag. The float-only projection and decoding APIs
retain their existing behavior. Current `encodeEncodeQuantize` and high-level `encode`
use the generic quantizing pipeline; direct raw tests cover the exported specializations.

When enabled, the order is:

1. Project input through encoder weights.
2. Add encoder bias, if present.
3. Apply ReLU, if `useActivation` is enabled.
4. Compute the FP32 L2 norm and normalize the latent coordinates.
5. Compute a per-vector scale and quantize to signed INT8.

The existing generic path defines the numerical contract. The computed norm must exceed
`VA_EPSILON` (currently 1e-7) to use its reciprocal; otherwise the normalization multiplier
is zero. Quantization uses `max(maxAbs(normalizedLatent) / 127, VA_EPSILON)` and rounds
scaled values before clamping to [-127, 127]. Finite zero or below-cutoff vectors therefore
produce zero codes and the epsilon scale. Normalization may change the scale without
changing codes, so consumers must preserve each vector's scale when reconstructing it.
The existing high-level `encode()` result stores only their average, and `decode()`
reuses it for all rows. That source-confirmed loss of per-vector scale is a separate
reconstruction defect awaiting a result/API fix; the buffer APIs preserve the full
scale array.

The three specialized quantizers call the generic normalization helper after their
existing affine/ReLU calculations. Tiled pass 2 keeps its input intermediates immutable:
one lane computes the norm with the generic sequential FMA order, then publishes its
reciprocal through threadgroup memory and a uniform barrier. Threads reduce the maximum
of normalized values and quantize those same normalized values. There is no extra device
buffer or dispatch. Enabled tiled normalization adds a serial O(L) norm loop and a barrier;
no performance improvement is claimed.

**Disabled normalization retains prior behavior.** Generic and specialized quantizers
keep the epsilon scale floor. Tiled pass 2 keeps its legacy `maxAbs / 127` scale, including
zero for an all-zero vector, and its inverse-scale cutoff at 1e-8. This slice does not
silently unify disabled-path quantization conventions.

The no-bias generic quantizing wrapper also binds a persistent 128-float zero buffer.
Metal API validation rejected its previous nil buffer(4) binding even though the shader
checks for a null bias pointer. A 512-byte fallback allocated at kernel initialization
preserves the no-bias quantized output; a real bias takes precedence. Allocation failure
throws through the existing initializer. Other optional-bias entry points are outside
this local correction and remain candidates for broader validation.

## Raw migration and retained limits

`neural_quantize_pass2` now requires a UInt32 constant at **buffer(5)**: zero disables
normalization, nonzero enables it. Bindings 0–4 remain intermediates, codes, scales, N,
and L. `encodeTiledV3` supplies this value from the dispatch parameters, independently of
the configuration used to load weights. Swift signatures and the standard neural
parameter struct are unchanged. Pass 1 still requires the activation flag at buffer(8)
introduced in slice 22. Existing raw callers must bind both relevant constants.

No public dimension limits, raw layouts, buffer-capacity rules or lifetime requirements
are broadened. Public neural weight configurations retain the existing latent cap of
128. Raw tiled pass 2 remains a per-vector threadgroup operation; callers supply valid
positive dimensions, supported threadgroup geometry, enough nonaliasing storage and
ordering after pass 1. Tests exercise larger raw latent rows without changing the public
cap. The input intermediate buffer is read-only, and over-dispatched vector groups return
uniformly before barriers.

FP32 projection/norm overflow, underflow, nonfinite behavior and fast-math limits remain.
This is the generic neural normalization policy, not a new robust full-range norm.
Projection/reduction rounding can vary between pipelines; no bitwise parity or exact
quantization-boundary guarantee is added.

`Hardening/NeuralLatentNormalizationTests.swift` checks both shader compilation paths,
the three raw specializations and generic quantizer, bias/ReLU/normalization combinations,
public tiled and generic output, per-dispatch flag forwarding, zero and tiny vectors,
ragged rows, multiple SIMD groups, input immutability and output guards. Assertions check
codes **and scales** against independent expectations. The previous ignored-flag tests
also continue to protect activation and learned-distance normalization.
