# Factory upload capacity and source bounds

The aligned-array and single-VectorProtocol uploads in `MetalBufferFactory` previously
passed the padded destination length to `makeBuffer(bytes:length:)`. The source only
promised the payload length. A three-Float payload could therefore upload a fourth
Float from outside its exposed storage. Regression fixtures reproduced this for both
an array with a removed trailing element and a vector exposing a bounded storage window.

## Covered entry points

- `createAlignedBuffer(length:alignment:options:)` checks byte-length rounding.
- `createAlignedBuffer(from:alignment:options:)` copies the exact array payload.
- `createBuffer(fromVector:alignment:options:)` copies the exact exposed vector payload.
- `createBuffer(fromVectors:alignment:options:)` requires rectangular, nonempty rows.
- `createBufferPair` applies the same batch rules independently to each side. The two
  sides may have different dimensions; each must be internally rectangular.

Initialized uploads allocate destination capacity first, copy only payload bytes, then
zero the tail between payload and rounded length. They do not read source padding or
retain a borrow of the source storage. Every vector's exposed pointer count must match
its declared count; a mismatch returns nil, including one discovered during batch copying.
No partially populated buffer is returned. Sources must remain stable during the call.

## Sizes and storage modes

Alignment is a positive power-of-two byte-length rounding multiple. It does not establish
arbitrary CPU/GPU base-address alignment. Nonpositive payload/length requests, negative or
overflowing counts, invalid alignment, rounding overflow, and rounded sizes beyond
`device.maxBufferLength` return nil. These direct upload APIs return nil for empty inputs;
the pool's separate minimum-bucket behavior for zero-byte requests is unchanged.

Array elements must be safe to copy as raw bytes. The generic API does not serialize
reference-containing values or change their ownership semantics.

Initialized uploads require CPU-accessible storage: shared, or managed on macOS when
supported by the device. Private and memoryless modes return nil before allocation/copy.
Other resource-option flags are forwarded. Managed writes notify the initialized range
with `didModifyRange`; this branch was source-reviewed, while current hardware regression
coverage exercises shared uploads. Device/platform restrictions on resource options still
apply. Uninitialized aligned allocations may use private storage because no CPU upload is
performed; their contents are not covered by the zero-padding promise.

All existing public method signatures remain intact, including the inlinable generic
methods. Invalid calls that previously truncated, copied outside the payload or trapped
now return nil on the covered paths. Rebuild consumers to pick up changes in methods that
may have been inlined. No throughput improvement or arbitrary alignment guarantee is made.

## Boundaries and verification

This does not change the pool's 64 MiB policy or fix its oversized bucket selection.
It does not harden every direct allocation or utility: non-aligned raw-pointer methods,
`isBufferAligned`, and low-level caller-owned pointer validity retain their existing
requirements. No new bias, residency or model-concurrency policy is introduced.

`FactoryUploadBoundsTests` checks controlled suffix leakage for both affected APIs,
exact payload and zero padding, scalar byte payloads, ordinary/aligned controls, invalid
sizes/alignment, overflow, ragged rows, inconsistent declared counts, storage options,
and GPU blit readback after the source's lifetime ends. Initial tests reproduced the
extra source value, ragged acceptance, and invalid-alignment acceptance; the Int.max
rounding probe terminated the original test process. Final API/shader validation and
full debug/release gate evidence is recorded in audit slice 26.
