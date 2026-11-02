# Buffer pool cache clearing

`BufferPool.clearCache()` first drains the returns already in its pending-return queue,
then releases all available cached buffers and subtracts their bucket bytes from
`currentMemoryUsage`. The VectorCore `BufferProvider.clear()` spelling uses the same path.

Clearing restores the allocation budget charged to the removed cache. For example, a
1 KiB pool holding one returned 1 KiB buffer reports zero current usage after clearing
and can allocate a fresh 1 KiB buffer. Allocation/hit/miss counters remain cumulative;
clearing does not reset the pool's statistics or start a new generation.

Outstanding leases and compatibility handles remain tracked and backed by their buffers.
Their bytes remain charged. A live 1 KiB lease alongside 4 KiB of returned cache therefore
leaves 1 KiB of accounted usage after clearing. Those live leases can return and be reused
normally afterward, using the same return queue.

Draining may already discard some returns because the bucket's available-cache limit is
full. Those discarded bytes are subtracted during return processing. Cache clearing then
subtracts only the accepted available buffers, exactly once. Repeated clearing of an empty
cache leaves live accounting and cumulative statistics unchanged.

## Concurrency and lifetime

The pool actor serializes bucket/accounting changes. Token return enqueues synchronously
under the queue lock and can race clearing: the call drains once, so returns enqueued after
that drain may be cached by a later acquisition/statistics read. Clearing does not promise
a globally empty cache while other callers continue returning leases.

Explicit return ends the caller's lease, even if a retained token still owns the underlying
Metal object. Current usage describes pool-accounted storage, not every external reference
or GPU allocation on the device. Callers retain responsibility for GPU completion and must
not return buffers prematurely.

`reset()` remains distinct: it retires a generation and resets its statistics. Physically
live retired leases are excluded from new-generation budgets, and their late returns do
not enter the current queue. Clearing after reset never subtracts those retired leases.
See the [pool lifecycle contract](BUFFER-POOL-LIFECYCLE-CONTRACT.md) and
[allocation bounds contract](BUFFER-ALLOCATION-CONTRACT.md).

## Verification scope

`BufferPoolCacheAccountingTests` covers restored budget, queued explicit/deinit returns,
live-lease content and later reuse, multi-bucket/idempotent clearing, compatibility pointer
lifetime, retired-generation isolation and drain-time cache-limit discards. Existing
allocation/lifecycle/consumer tests remain in place. See AUDIT-3 slice 31 for gate results.

No public signatures, allocation limits, shader behavior, reset policy or general eviction
strategy changes. This does not change `ArgumentTablePool.clearAvailable()` timing or its
separate descriptor/batch-acquisition behavior.
