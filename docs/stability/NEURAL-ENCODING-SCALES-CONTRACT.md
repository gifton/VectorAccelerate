# Neural encoding result and reconstruction scales

High-level `NeuralQuantizationKernel.encode()` returns an owned `scales: [Float]` in
`Metal4NeuralEncodingResult`. There is exactly one Float32 quantization scale per vector,
in the same row order as the INT8 `latentCodes`. Both `decode()` and `decodeFlat()` pass
these individual scales to the GPU. Previously, the result retained only their average
and both decoders broadcast it, corrupting reconstruction whenever row scales differed.

Codes and scales are copied out after GPU completion. Results remain valid after pooled
buffers are reused or weights are unloaded. Decoding still requires loaded, compatible
weights: matching latent dimensions does not identify a model or guarantee that different
weights reproduce the original vectors. Concurrent weight replacement is not covered.

## Compatibility and storage

The existing `.scale` read remains as a deprecated computed average for diagnostics;
it must not be used for reconstruction. Buffer-level consumers should copy
`encoded.scales[i]` into scale slot `i`. The two benchmark consumers have been migrated.
Encode/decode method signatures and raw shader bindings are unchanged by this slice.
The result previously had no public memberwise initializer; its internal initializer
now takes `scales` instead of `scale`.

`bytesPerVector`, `compressedSize`, and configuration `compressionRatio` continue to
measure the INT8 code payload only. Scale values add four bytes per vector, excluding
Swift container overhead. Numerical projection, quantization, normalization and FP32
range limits remain as documented in [the normalization contract](NEURAL-LATENT-NORMALIZATION-CONTRACT.md).
Preserving metadata does not make quantization lossless or add nonfinite-input guarantees.

## Shape, storage and decoder routing

Both high-level decoders reject nonpositive or unsupported dimensions, latent dimensions
that differ from the loaded configuration, code lengths other than N×L, and scale counts
other than N. Output products must fit the device buffer limit before allocation and
UInt32 conversion. High-level encoding rejects ragged input before flattening it. Each
pooled allocation is checked for sufficient actual storage before a CPU copy or dispatch.
The global pool's oversized-request behavior remains separate debt; these wrappers reject
an undersized allocation rather than writing past it.

Weight loading prepares a persistent zero decoder-bias buffer sized to the output row.
All decoder wrappers bind it when no real bias exists. Metal API validation rejected the
previous unbound transposed-decoder bias argument despite its shader null check. Real
bias still takes precedence, and unloading weights releases the fallback. This local
correction does not establish validation coverage for every optional-bias API.

The non-transposed optimized decoders process complete float4 latent blocks. Their Swift
wrapper selects them only when L is divisible by four, otherwise using the existing
scalar decoder. Raw callers of those optimized shaders must still satisfy L % 4 == 0.
The public latent cap remains 128. Transposed decoders retain their existing routing.
No performance improvement or new raw buffer-validation guarantee is claimed.

## Regression coverage

`Hardening/NeuralEncodingScaleTests.swift` uses identity weights and exact INT8 coordinates
to separate scale loss from quantization error. Six tests cover different row magnitudes,
zero rows, normalized rows, both high-level decoders, all transposed specializations,
owned metadata across pool reuse/reload, malformed results and ragged input rejection.
Direct calls cover the scalar fallback and four non-transposed threadgroup widths with
L=3 and L=4, checking output values and a trailing guard. Metal API and shader validation
exercise these paths; full debug/release gates cover the two shader compilation modes.
