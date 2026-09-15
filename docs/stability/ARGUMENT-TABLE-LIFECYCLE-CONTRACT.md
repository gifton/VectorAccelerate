# Argument table pool lifecycle

Each `ArgumentTablePool` owns a lock-protected pending-return queue. An
`ArgumentTableToken` owns its table and weakly references that queue. The queue has no
global registry, actor-address routing or reference back to the pool.

Explicit token `release()` and token deinitialization enqueue at most one return, capturing
the weak destination once. Acquisition and statistics reads drain the queue synchronously.
The existing pool release guard accepts only tracked in-use tables, clears their bindings,
and makes them available for reuse. There is no detached return task or required sleep. The engine's twelve direct
queue-return sites now acquire tokens and defer their release over the same operations,
preserving default/batch/matrix descriptors and existing GPU completion waits.

Pool destruction releases undrained queued tables and their retained buffer bindings.
Outstanding tokens do not retain the pool: their tables remain owned and usable until
released or destroyed, but a late return to a dead pool has no destination. Both default
and descriptor-specific token acquisition follow this policy.

`setBuffer` retains the referenced Metal buffer through the table's binding storage until
bindings are cleared or the table is destroyed. A numeric address supplied to `setAddress`
does not itself own the resource at that address. Callers remain responsible for resource
lifetime and synchronization through GPU consumption.

Explicit release ends a lease; callers must not continue using a returned table because
it may be reset/reused immediately. Use the token's release/deinit path for token leases,
and the pool's direct release API for directly acquired tables. Mixing release paths or
using a stale raw reference after release is not made safe by the membership guard.

`clearAvailable()` still drops only the available cache. It does not replace the queue,
retire outstanding tokens, or gain reset semantics. Its existing pending-return timing is
unchanged: queued returns may be drained by a later acquisition/statistics read. Descriptor
reuse, batch-acquisition rollback, count validation and native binding integration are
outside this ownership fix. Public signatures and table limits remain unchanged.

`ArgumentTableLifecycleTests` covers queued table/buffer destruction, live-token ownership,
both acquisition spellings, explicit/deinit immediate reuse with binding cleanup, concurrent
exactly-once returns, cross-pool isolation and outstanding returns after cache clearing.
Tests use the real Swift table implementation and Metal buffers; they do not claim new
native argument-table dispatch coverage. See AUDIT-3 slice 30 for gate results.
