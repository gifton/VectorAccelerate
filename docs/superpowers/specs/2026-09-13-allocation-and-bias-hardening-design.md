# Allocation and optional-bias hardening: design discussion

**Status:** proposed, awaiting owner decisions. Planning was requested on 2026-09-13;
production implementation has not been authorized in this planning session.
**Baseline:** `1d2928f`, branch `gifton/metal-hardening-checkpoint`.

These are two independently deliverable changes. Numerical-backlog reconciliation is
an authorized documentation change alongside this discussion, not a dependency on either
implementation. Maintain separate checkpoints on the existing branch.

## Decisions to settle

| Decision | Recommended starting point | Alternative and consequence |
|---|---|---|
| A: Requests larger than 64 MiB | Enforce the existing pool limit with a recoverable error. Preserve direct non-bucketed allocation APIs for explicitly larger workloads. | Support larger buffers within device and configured memory limits. Requires a design for their accounting, return and reuse; expands this slice. |
| B: Raw batch `encodeFused` compatibility | Make the existing method throwing; validate before mutating the encoder. Migrate internal callers with `try`. | Add a separate checked API. Existing callers remain unchecked, so this does not close safety on the original public entry point. |

Neither recommendation is an owner decision yet. The linked plans are provisional for
the recommended choices; revise them before execution if alternatives are selected.

## A. Allocation contract

### Evidence and impact

`MetalBufferFactory.selectBucketSize` returns the largest bucket for any oversized
request. `BufferPool.getBuffer(size:)` checks the selected bucket against that same
maximum, so its intended rejection cannot trigger. `createBucketedBuffer(size:)` also
uses the capped size. This violates the documented allocation promise of at least the
requested bytes. Typed uploads subsequently hit `BufferToken.write`'s precondition;
raw consumers can receive less storage than their declared shape requires.

`BufferPoolEnhancedTests.testVeryLargeBuffer` explicitly expects the cap, and
`testLargeSizeHandling` accepts either a capped allocation or almost any VectorError.
Those tests must be replaced with discriminating contract checks, not simply retained.
IVF and high-level neural wrappers already reject insufficient actual storage locally.
Keep those defensive checks after the central fix. The new cap rejection uses
`invalidBufferSize`, whose VectorError kind is `.invalidData`. Update
`IVFCandidateBoundsTests.testLargeOutputsCannotPublishCountsBeyondPhysicalStorage`, which
currently expects only `.allocationFailed`, to pin the new early cap rejection and its
size details without weakening its physical-capacity assertions.

### Recommended bounded design (decision A)

- Successful allocations expose at least the requested byte count. An unsupported
  request fails before allocation, handle registration, copying or GPU submission.
- Keep the standard bucket sizes and reuse behavior for supported requests.
- Preserve current zero-byte requests as minimum-bucket leases, including empty typed
  uploads. Negative byte sizes and counts fail; empty uploads perform no pointer copy.
- Pool requests above 67,108,864 bytes throw `VectorError.invalidBufferSize` with the
  original request and supported maximum. Do not return a partial allocation or truncate.
- The optional-returning factory bucketed allocator returns nil for unsupported sizes.
  Direct `createBuffer(length:)` retains its existing device-limited allocation behavior.
- Retain the public `selectBucketSize(for:) -> Int` signature. Its lookup can retain its
  documented capped result for compatibility; allocation entry points must validate the
  original request before lookup and must not treat a capped lookup as sufficient storage.
  Document this distinction explicitly. Do not use a trap/precondition to reject input.
- Pool typed-count multiplication and `BufferPool.getAlignedBuffer` rounding use checked
  arithmetic. Alignment
  must be positive and a power of two; reject overflow before rounding can wrap. Preserve
  existing supported valid calls. Distinguish byte-length rounding from a promise about
  physical GPU-address alignment; this slice does not add the latter.
- Compare selected allocation length with device limits and the current memory budget.
  Preserve existing memory-pressure cleanup/error behavior and successful lease return.
  Use overflow-safe comparisons where these checks are touched.
- Check actual buffer capacity before returning a lease or copying initialized data.
  Failed input must not create a tracked handle or increase retained allocation usage.
  Draining already-pending returns may legitimately change statistics on a call.

### Scope boundaries

The factory portion covers `createBucketedBuffer` only. Direct `createAlignedBuffer`
and VectorProtocol upload conveniences are excluded from this bounded plan. Source review
found separate unchecked arithmetic in those methods and a padded-source read in
`createAlignedBuffer(from:)` and `createBuffer(fromVector:)`: they pass the rounded
allocation length to `makeBuffer(bytes:)` even when the source contains fewer bytes.
Record a separate high-priority reproduction/fix for allocating destination capacity
independently of the exact source copy length. Do not claim the full factory is hardened.
The factory's multi-vector path also needs a separate shape/storage-mode review.

No new large-buffer cache, custom bucket configuration, eviction redesign, general
BufferToken API redesign, residency rewrite or concurrency-policy change. Existing
pool-reset/return and accounting concerns discovered during implementation are separate
unless they prevent the required allocation tests from passing; report such a dependency.
`BufferToken.write` remains a low-level caller-preconditioned method. The throwing
allocation/upload conveniences must avoid reaching that precondition for rejected input.

### Alternative larger-buffer design

If the owner chooses larger-buffer support, establish whether >64 MiB buffers are
uncached tracked leases or dynamically pooled sizes. Uncached tracked leases are the
smaller extension: account their full size until the lease returns, enforce the device
and pool budget, then release rather than retain them. A token with `pool: nil` alone
would bypass pool accounting and is not sufficient. Specify pressure handling, explicit
return/deinit, reset with outstanding leases, statistics and reuse before implementation.
This alternative requires a revised plan and broader lifecycle tests.

### Acceptance

For the pool and bucketed factory, test 0, 1, every bucket boundary, 64 MiB minus one/exact/plus one, negative values,
Int.max, overflowing typed counts and invalid/overflowing alignment. Test direct factory,
pool, initialized typed upload and compatibility handle entry points. Valid at-cap
allocation must actually meet capacity; invalid tests must not allocate gigantic arrays
or write outside a buffer. A real just-over-cap typed upload can use a bounded ~64 MiB
fixture after direct rejection is proven. Verify normal reuse/returns and no new tracked
allocation on rejected requests. Run existing IVF/neural guards as consumers.

## B. Optional-bias contract

### Evidence and impact

`NeuralQuantizationKernel.encodeEncode` still binds `encoderBias?.buffer` at buffer(3).
The generic quantizing wrapper already uses persistent `zeroEncoderBias` with capacity
128 Float32 values; decoders use their output-sized fallback. Float encoding has not yet
been reproduced under Metal validation in this slice. Treat the missing-binding failure
as a candidate until observed, then retain its log as evidence.

`BatchMatrixKernel.encodeFused` binds caller bias at buffer(4) without validating length.
The shader's buffer(7) bias mode selects no bias, N shared column values, or batchSize*N
per-batch column values. The high-level array API validates element counts, but a failed
optional bias allocation currently becomes nil and silently selects no bias. Both the
raw bounds path and that high-level allocation-failure path belong in this local fix.

### Recommended bounded design (decision B)

- Reuse the persistent neural zero-bias buffer in the float-only encoder. Ensure the
  declared latent width fits the fallback before binding; do not widen the public cap
  of 128. Perform rejection before encoder mutation. Valid raw input/weight/output
  layouts remain caller requirements; this is not a full neural raw-buffer validator.
- Preserve real-bias precedence if present. Current neural loading APIs do not expose
  encoder-bias loading; do not add a model/bias-loading API merely to construct a test.
  Raw shader fixtures can exercise explicit real bias independently.
- Make `encodeFused` throwing if approved. Derive active bias mode from the existing
  buffer/layout rules: nil means no bias regardless of the default layout; `.none`
  means no bias even if a buffer was supplied. Preserve the existing behavior in which
  supplied bias and layout, not the legacy `config.hasBias`, control addition.
- For active shared bias require at least N Float32 elements. For active per-batch bias
  require at least batchSize*N Float32 elements. Compute byte counts without overflow;
  compare against actual buffer length before binding or dispatch. Permit larger buffers.
  Reject a bias on the wrong device. Other raw matrix storage/layout requirements remain
  unchanged and caller-owned; do not advertise universal GEMM input validation.
- Bind valid dummy storage when bias is disabled if API validation requires a binding.
  Preferred bounded solution: a persistent one-Float zero buffer prepared in the throwing
  initializer, with mode zero guaranteeing no shader reads. Prove this on both compile
  paths; if validation requires a wider footprint, adjust based on evidence. Do not allocate
  a full batch bias or add a dispatch per call.
- A supplied high-level bias that cannot be allocated throws `bufferAllocationFailed`;
  it must never degrade to the no-bias calculation. Keep high-level exact-count validation.
- Complete all validation before calling `setComputePipelineState`, binding or dispatch.
  Invalid calls encode no work; normal fusion can still proceed after a caught error.
- No raw shader binding/layout change is expected. Existing valid output values, activation
  order and bias broadcasting remain unchanged. Zero-work behavior should be preserved;
  no unrelated shape-domain policy should be introduced to simplify this change.

### Compatibility and alternatives

In-repository search found one production `encodeFused` call, inside `multiplyFused`.
External callers are unknown. The source-breaking change therefore needs a migration
note and owner decision even though the internal migration is small. A separate checked
method preserves source compatibility but leaves the old unsafe surface until migrated;
a precondition or silent no-op is not an acceptable substitute for recoverable failure.
No strict signed-zero/bitwise guarantee is added by the neural zero-bias addition.

### Acceptance

Use analytic matrix/identity fixtures with multiple rows and batches; distinguish shared
from per-batch bias, and check activation after bias. Exercise nil, `.none` with supplied
storage, exact/oversized/one-element-short buffers, disabled bias and `hasBias` mismatch
controls. Validate the no-bias binding under Metal API and shader validation. Neural
fixtures cover ragged input/latent widths and the 128 cap, signed outputs with activation
off/on, loaded-weight lifecycle and output canaries. Reject over-cap dispatch before
using the finite zero buffer. No private test setter or unrelated public API is needed.

## Execution and validation

Implement A and B as separate checkpoints on the same existing branch, preferably A
first. Neither needs a new dependency or a VectorCore update. Preserve Metal 4/platform
requirements and VectorCore 0.3.3. Use mechanism-specific failing tests before each fix,
then targeted tests, Metal validation for bias work, and full debug and release gates.
Run GPU workloads serially; release must execute runtime shader compilation.

Documentation-only reconciliation gets source/reference/math checks, not new runtime
coverage claims. Latest historical full gates remain 1701 passed + 11 skips in each
configuration. Implementation counts are recorded only after fresh runs.

Provisional task plans:
- [Allocation plan](../plans/2026-09-13-buffer-allocation-bounds.md)
- [Optional-bias plan](../plans/2026-09-13-optional-bias-validation.md)
