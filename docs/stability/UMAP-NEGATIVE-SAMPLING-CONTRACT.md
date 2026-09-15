# UMAP negative-sampling contract

VA3-019's negative-sampling race is fixed by keeping target coordinates immutable
throughout each pass. Each point starts at its input coordinates and evolves in the
supplied sample order. Every target is read from the embedding at the **start of that
negative-sampling pass**, after any preceding attractive updates. Repeated targets
remain repeated samples; self and out-of-range IDs (including UInt32.max) are skipped.
The coefficient formula, epsilon placement, coefficient clipping and FP32 fast-math
behavior are unchanged. This is not a change to fully simultaneous source gradients:
later samples see their own point's earlier updates.

The sampling shader writes a separate N × D Float32 output buffer, initializing each
row itself. A second GPU dispatch copies completed output back to the embedding.
Each dispatch assigns an entire row to one thread, using wide row-address arithmetic;
there is no cross-thread read of output during sampling. Temporary storage costs
4 × N × D bytes and publication adds a copy dispatch. No speedup is claimed.

## Swift API and migration

`applyNegativeSampling`, `optimizeEpoch`, and `executeEpoch` retain their existing
throwing async signatures and publish results in place. `encodeNegativeSampling`
now **throws**: existing fused call sites must add `try`. The owner approved this
source compatibility change so allocation and input failures can be reported safely.

The original parameter list allocates private temporary output for each call. It
requires a command buffer that retains bound resources. An overload adds
`scratch: any MTLBuffer` after `randomTargets`, allowing allocation-free encoding
and reuse across ordered passes:

```swift
try kernel.encodeNegativeSampling(
    into: encoder, embedding: embedding, randomTargets: targets,
    scratch: scratch, n: n, d: d, params: params
)
```

Scratch must hold at least N × D Float32 values; its initial contents are ignored.
All buffers must be on the context's device and have sufficient physical lengths.
The embedding, target-ID array, and scratch must not overlap. The host rejects
overlapping GPU-address ranges, including identical buffers. Callers must also avoid
physical storage aliasing through distinct mappings or aliasable heaps, which address
range checks cannot establish. Keep caller-owned resources alive through GPU completion,
especially for command buffers with unretained references. Do not reuse scratch or
modify inputs concurrently with the pass; synchronization across queues is caller-owned.

The encode method inserts buffer barriers before sampling, between sampling and
copy-back, and after copy-back. These order earlier producers, scratch publication,
later consumers and scratch reuse within the same encoder, including concurrent compute
encoders. See Apple's [buffer barrier documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(scope:)).
The returned `Metal4EncodingResult` describes the sampling dispatch geometry; the
copy-back dispatch is additional work.

The negative-sampling entry points require nonnegative N, D and sample rate fitting
UInt32. Required byte counts must fit the device's maximum buffer length; validation
precedes multiplication, allocation and encoding. Zero N, D or rate performs no
allocation or dispatch after validating these counts; buffers are unused in that case.
Negative-sampling input errors throw `VectorError.invalidInput`; scratch allocation
failure throws `VectorError.bufferAllocationFailed`. No work from an invalid call is
encoded. These checks are local to negative sampling, not a general validator for all
UMAP epoch inputs, graph endpoints or floating-point parameters.

## Raw shader ABI

`UMAPParams` remains 32 bytes. `umap_negative_sample_kernel` now binds:

| Buffer | Meaning |
| --- | --- |
| 0 | Immutable Float32 embedding [N, D] |
| 1 | UInt32 random target IDs [N, negativeSampleRate] |
| 2 | UMAPParams |
| **3 (new, required)** | Distinct Float32 output [N, D] |

Raw callers must bind output at buffer(3); the shader no longer updates buffer(0).
`umap_copy_embedding_kernel` binds source at 0, destination at 1 and UMAPParams at 2.
Synchronize the whole sampling dispatch before publishing output or modifying input.
A threadgroup-local barrier alone cannot order all participating threadgroups.
Both kernels tolerate over-dispatch, zero N and zero D. With rate zero, the raw
sampling kernel copies each input row to output; the Swift API skips the whole pass.
Raw callers remain responsible for buffer lengths, nonaliasing, lifetime and ordering.

## Scope and verification

`Hardening/UMAPNegativeSamplingTests.swift` exercises actual GPU output on both shader
compilation paths. Hand-derived fixtures distinguish frozen target reads from the old
race and from accidentally freezing source updates. Coverage includes ragged dimensions,
over-dispatch canaries, skipped IDs, empty/invalid inputs, scratch reuse, concurrent
encoders, unretained command buffers with explicit ownership, epoch ordering, and a
Double reference for the default curve parameters.

Removing this race does **not** make complete UMAP epochs bitwise reproducible.
Attractive target gradients still use floating-point atomics whose accumulation order
can vary, and high-level optimization generates random targets. No cross-device or
cross-compiler bitwise guarantee is added. VA3-019 remains open for the broader atomic
accumulation policy; numerical-range and input-clamping backlog work is separate.
