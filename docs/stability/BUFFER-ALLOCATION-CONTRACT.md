# Buffer allocation bounds contract

`BufferPool` supports requests from 0 through 67,108,864 bytes (64 MiB), subject to
Metal device and pool memory limits. Every successful lease has actual storage at least
as large as its requested byte count. Unsupported requests fail before a new allocation,
compatibility-handle registration, initialized-data copy or GPU submission.

## Sizes and errors

- Zero bytes requests the minimum 1,024-byte bucket; it is not a nil/empty allocation.
- Positive supported requests round up to the existing standard bucket sizes.
- Requests above 64 MiB throw `VectorError.invalidBufferSize`, whose kind is `.invalidData`.
  Error details retain `requested_size` and `maximum_size` in bytes.
- Negative sizes/counts and unrepresentable derived byte counts throw `invalidInput`
  (also `.invalidData`). They do not wrap or trap during allocation arithmetic.
- A selected allocation must fit the Metal device's buffer limit and the pool budget.
  Allocation failure and memory-pressure errors remain recoverable.

The same allocator backs `getBuffer(size:)`, typed conveniences, `acquire(size:)` and
`acquireBuffer(byteSize:)`. Invalid requests register no handle and retain no new
allocation. Calls may drain previously pending returns; statistics are not promised to
remain byte-for-byte identical when other leases have been returned.

## Typed uploads and byte rounding

Both `getBuffer(for: data)` and `getBuffer(with: data)` check count times element stride
before allocating and copying. Empty uploads retain a zero element count, so
`copyData(as:)` returns an empty array; they do not dereference a source pointer. A typed
count-only request with count zero retains the ordinary minimum-bucket lease semantics.

`getAlignedBuffer(size:alignment:)` requires a positive power-of-two alignment and checks
rounding overflow before applying the pool bounds to the rounded request. Alignment rounds
the requested byte length; it does not promise arbitrary base-address alignment. Bucket
rounding can enlarge capacity further. Padding in pooled buffers is not automatically zeroed.

Elements must remain suitable for the raw memory-copy interface. Low-level
`BufferToken.write`, explicit-count reads and typed pointers retain their existing caller
requirements; this change does not make arbitrary token reads/writes recoverable or safe.
Callers must retain leases through GPU completion and synchronize access as before.

The nonthrowing `preallocateCommonSizes` convenience treats nonpositive counts as a
no-op. It checks each buffer against the device length limit and skips size groups whose
aggregate byte count overflows or exceeds the remaining pool budget. Supported groups
still populate the ordinary cache.

## Factory policy and larger workloads

`MetalBufferFactory.createBucketedBuffer(size:)` applies the same size range and zero-byte
minimum, returning nil for invalid/unsupported requests or allocation failure. It checks
the selected size against the factory device and checks actual returned capacity.

`selectBucketSize(for:)` remains a capped lookup for source compatibility. Its result
alone does not validate an allocation request: requests above the maximum still return
the maximum bucket from this lookup.

Direct non-bucketed factory allocation remains available for larger workloads within
Metal limits; it does not participate in pool budgeting or automatic lease return. Its
caller requirements are unchanged. Direct aligned/vector uploads have their separate
[factory upload contract](FACTORY-UPLOAD-BOUNDS-CONTRACT.md), including zeroed padding
and rejection of empty initialized input.

## Consumers and scope

[IVF candidate construction](IVF-CANDIDATE-BOUNDS-CONTRACT.md) and
[neural encoding](NEURAL-ENCODING-SCALES-CONTRACT.md) retain their local actual-capacity
checks. Oversize requests now fail centrally before an undersized lease can reach those
checks. High-level GPU APIs backed by the pool can therefore reject outputs above 64 MiB;
the existence of a larger direct-allocation API does not automatically reroute them.

This contract does not redesign cache eviction, reset with outstanding leases, residency,
concurrent access or general pool accounting. The subsequent
[return/reset lifecycle fix](BUFFER-POOL-LIFECYCLE-CONTRACT.md) removes global actor-address
routing and retires old queues on reset. Reset budgets exclude outstanding retired leases.
The subsequent [cache-clearing contract](BUFFER-CACHE-CLEARING-CONTRACT.md) specifies budget
restoration for cleared available buffers. No large-buffer cache is added, and these changes make no
performance or universal allocation-success guarantee.

Regression coverage: `BufferAllocationBoundsTests`, updated `BufferPoolEnhancedTests`,
and the IVF/neural consumer suites. See AUDIT-3 slice 28 for final verification evidence.
