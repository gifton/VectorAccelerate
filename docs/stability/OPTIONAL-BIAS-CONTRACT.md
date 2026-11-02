# Optional bias: bindings, capacity and failure

This contract covers the float-only `NeuralQuantizationKernel.encodeEncode` wrapper,
`BatchMatrixKernel.encodeFused`, and the bias allocation in `multiplyFused`. It retains
the existing arithmetic, activation order and raw shader bindings.

## Float neural encoding

When model encoder bias is absent, float encoding binds the kernel's persistent,
128-Float zero buffer at buffer(3). Model bias takes precedence when present. Before
mutating the encoder, the wrapper rejects a raw latent width above 128 and bias storage
shorter than the requested latent width. An unloaded model still throws.

The fallback is allocated during kernel initialization and survives weight unload/reload;
there is no per-encode allocation or GPU initialization pass. Current weight-loading APIs
do not populate encoder bias. Explicit-bias shader tests cover the raw shader independently
of the public no-bias wrapper; this change does not add a bias-loading API.

Input, weight and output storage, shape agreement with the loaded model, device ownership,
and synchronization remain raw caller requirements. The new bias guard is not a general
tensor validator. The latent cap remains 128. No new bitwise or signed-zero guarantee is
made for adding zero bias under the library's existing floating-point policy.

## Fused batch encoding

The existing raw `encodeFused` method is now **throwing**. Update source callers:

```swift
try kernel.encodeFused(
    into: encoder,
    batchA: a,
    batchB: b,
    output: c,
    bias: bias,
    biasLayout: .sharedColumns,
    parameters: parameters
)
```

Propagate the error or catch it before proceeding with other work. A rejected active bias
must not change bindings, pipeline state or dispatch work; callers can encode a valid
operation after catching the error on the same encoder. The high-level `multiplyFused`
method was already throwing and retains its source signature.

Bias mode is selected by the buffer and `biasLayout`:

| Input | Meaning | Minimum active bias storage |
|---|---|---|
| nil bias, any layout | Disabled | None supplied by the caller |
| Any supplied bias, `.none` | Disabled; supplied buffer ignored | None |
| Supplied bias, `.sharedColumns` | One bias per output column, shared across rows/batches | N Float32 values |
| Supplied bias, `.perBatchColumns` | One column-bias vector per batch, shared across rows | batchSize × N Float32 values |

The legacy `config.hasBias`/`parameters.hasBias` field does not override these rules.
Active bias byte counts use checked arithmetic. Short storage, an unrepresentable required
byte count, or an active bias from another device throws before encoder mutation. Larger
buffers are accepted; padding is not read. Disabled mode uses a persistent one-Float zero
binding at buffer(4), with mode zero at buffer(7); it does not read the supplied bias.
Bias addition precedes activation. The shader ABI and dispatch geometry are unchanged.

Raw A/B/output shape, storage, aliasing, strides, resource residency and lifetime remain
caller-owned. In particular, this does not validate all matrix buffers or make overlapping
inputs/outputs safe. Bias and other resources must remain valid through GPU completion;
callers must synchronize any writes. Existing zero-work geometry is retained.

## High-level arrays and allocation failure

`multiplyFused` continues to require an exact bias count of N or batchSize × N; nil
selects no bias. When allocation for a supplied bias fails, it throws
`VectorError.bufferAllocationFailed` before entering command execution. It cannot silently
switch the requested biased calculation to an unbiased one.

The allocation-failure branch is source-reviewed: no safe allocator-failure injection
seam is available, and tests do not exhaust GPU memory to force failure. Valid high-level
bias contents and activation are exercised by numerical regression tests. Wrong-device
rejection is hardware-tested only when a second Metal device is available; otherwise that
branch is source-reviewed. No broader allocator or model-concurrency policy is introduced.

## Verification

`OptionalBiasValidationTests` covers neural signed projections, activation, ragged input
and latent widths, the 128 cap, output canaries, unload/reload, and explicit raw shader bias.
`BatchOptionalBiasValidationTests` covers layouts, exact/oversized/short storage, disabled
modes, legacy flag disagreement, required-size arithmetic, encoder recovery, zero work,
and high-level bias results. Audit slice 27 records final test counts and validation
coverage, including the limitations above.
