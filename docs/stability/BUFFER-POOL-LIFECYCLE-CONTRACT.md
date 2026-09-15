# Buffer pool return and reset lifecycle

Each `BufferPool` owns a lock-protected pending-return queue. A `BufferToken` owns its
Metal buffer and weakly references the queue that was current when it was acquired.
There is no global buffer-return registry or routing by actor address.

## Returning a lease

Explicit `returnToPool()` and token deinitialization enqueue at most one return. The pool
drains its queue synchronously on acquisition and statistics reads; normal reuse needs no
sleep, detached task or eventual actor callback. Only a buffer tracked as in use may enter
the cache or reduce current allocation accounting when discarded. Unknown/duplicate
returns do not modify the cache or accounting.

Explicit return ends the caller's lease. A returned buffer may be reused immediately;
retaining the token afterward does not grant permission to keep reading or writing it.
Callers must still coordinate CPU/GPU access and keep leases until GPU work completes.
`keepAlive(until:)` prevents token deinitialization before command completion; it does not
make explicit premature return safe.

## Pool destruction

A token does not retain the pool. Its buffer remains valid for its lifetime even after the
pool is destroyed. An attempted late return then has no destination and is discarded.
Queued returned buffers are released with the pool's queue; they cannot accumulate in a
global queue or enter a different pool whose actor occupies a reused memory address.

The pool owns compatibility tokens in `activeHandles`. Those tokens no longer point back
to the pool, so unreleased handles do not create a pool retain cycle.

- `BufferToken` and `MetalBuffer` own their Metal storage independently of pool lifetime.
- A VectorCore `BufferHandle` borrows a pointer. Retain its `BufferPool` provider until
  release, and never use the pointer after release or provider destruction. A handle alone
  does not own its backing Metal buffer.

## Reset

`reset()` replaces the return queue, drops current cache/tracking and starts fresh
statistics. Returns already queued in the old generation are discarded. Tokens acquired
before reset still own their buffers, but later explicit/deinit returns cannot enter the
new generation. Enqueue racing reset either reaches the retired queue or sees no queue;
it cannot switch to the new destination.

Compatibility handles remain backed by their retained tokens across reset while the pool
is alive. Releasing those handles still removes them from `activeHandles`, without
returning their retired allocations to the new cache. Newly acquired leases reuse normally.

Reset does not wait for GPU work, invalidate memory still owned by a live token/handle,
or enforce a memory budget across old and new generations. Its statistics and allocation
budget cover the new generation; older outstanding storage can remain physically live.
The [allocation bounds contract](BUFFER-ALLOCATION-CONTRACT.md), including the 64 MiB
per-request cap, remains in effect.

## Scope and evidence

`BufferPoolLifecycleTests` covers pool and queued-buffer destruction, independent token
lifetime, compatibility-cycle removal, explicit/deinit returns after reset, compatibility
pointer lifetime, foreign/duplicate returns, concurrent exactly-once returns, and reset
races. A GPU blit verifies valid storage after pool destruction with completion anchoring;
Metal command buffers also retain resources, so this is not proof that anchoring alone
is necessary for that particular blit. Previous bounds tests no longer reset fresh pools
to avoid stale global returns. See AUDIT-3 slice 29 for execution evidence.

The subsequent [cache-clearing fix](BUFFER-CACHE-CLEARING-CONTRACT.md) drains pending
returns and restores the budget charged to cleared cached storage while preserving live
leases and this generation policy. General eviction and raw token data-access synchronization
remain separate. ArgumentTablePool's analogous return ownership was subsequently
fixed separately; see the [argument table lifecycle contract](ARGUMENT-TABLE-LIFECYCLE-CONTRACT.md).
