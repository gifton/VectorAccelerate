# Metal 4 Command Model Adoption — Design Spec

**Date:** 2026-09-17
**Status:** Approved design (sections 1–6 approved by the owner in conversation on 2026-09-17, plus decisions D1–D3); awaiting owner review of this written form, then Metal-agent feasibility review, then the stage-0 implementation plan.
**Program context:** research brief `docs/superpowers/research/2026-09-14-observability-harness-research.md`; harness foundation `docs/superpowers/specs/2026-09-14-harness-sp0-foundation-design.md`; adapter seam `docs/superpowers/specs/2026-09-14-harness-interface-contract.md`; wish list `docs/superpowers/research/2026-09-14-metal-improvements-wishlist-for-harness.md` (items HW-01, HW-02, HW-03, HW-04, HW-05, HW-09 are delivered by this spec and no longer stand alone); IVF integration `docs/superpowers/specs/2026-09-14-ivf-benchmark-integration.md`; timing spike `docs/superpowers/research/spikes/2026-09-14-mtl4-timestamps/`.
**Version impact:** VectorAccelerate **0.8.0** (breaking). SP0 ships separately as 0.7.0 first.
**Implementer:** the Metal hardening agent (stages 0, 1, 2, 4, 5); the IVF agent (stage 3, its own kernels). The harness owner writes this spec, the acceptance tests, and the parity harness.

---

## 1. Goal, non-goals, decisions

**Goal.** Every GPU submission in `Sources/VectorAccelerate` goes through `MTL4CommandQueue`, `MTL4CommandBuffer`, and `MTL4ComputeCommandEncoder`. No legacy command type (`MTLCommandQueue`, `MTLCommandBuffer`, `MTLComputeCommandEncoder`, `MTLBlitCommandEncoder`) remains in `Sources`. Residency is explicit. Per-submission timing records and opt-in counter-heap encoder spans exist from the first stage. Results do not change; every kernel proves that with a bitwise parity gate before its legacy path is removed.

**Non-goals.**
- `MTL4Compiler`. Legacy `MTLComputePipelineState` objects run on Metal 4 encoders (verified in the spike: pipeline built with `device.makeComputePipelineState(function:)`, dispatched from an `MTL4ComputeCommandEncoder`). `PipelineCache`, `Metal4ShaderCompiler`, and `ArchivePipelineCache` are untouched.
- Consolidating the six per-provider `Metal4Context` instances into one shared context. The migration keeps one queue per context. Consolidation is a follow-up.
- Any shader change. Binding indices and `[[buffer(n)]]` attributes are unchanged.
- ML command encoders, tensors, indirect command buffers, acceleration structures.
- Async (non-waiting) submission pipelining. `submit` waits; a non-waiting variant is a later addition on the same record type.

**Decisions inherited from the owner (2026-09-17).**

| ID | Decision |
|---|---|
| Drop legacy | The legacy command path is removed, not deprecated: the platform floor (macOS 26, iOS 26, Metal 4) makes it dead weight. |
| Staging | New path added inside `Metal4Context` beside the old one; per-kernel migration; legacy deleted at the end. |
| Versioning | SP0 = 0.7.0, this = 0.8.0. |
| Ownership | Metal agent implements; harness owner specs and gates; IVF agent migrates IVF and clustering kernels (D2). |
| D1 | Kernel-level `encode` entry points stay **public**, taking `KernelEncoder`. |
| D3 | Hardening tests that build their own legacy queue to exercise shaders directly are **left alone**; the class-closing "no legacy command types" test is scoped to `Sources/` only. |
| Closure model | The `submit` closure is synchronous; no suspension while an encoder is open (section 4). |
| `BarrierHelper` | Deleted; replaced by `KernelEncoder.barrier()` (section 5). |

## 2. Current state (facts the design is built on)

- **Five submission paths, all legacy.** `Metal4Context.execute` (`Core/Metal4Context.swift:318`, 0 callers), `executeAndWait` (`:373`, ~111 call sites in 42 source files, 54 in 18 test files), `executeBlitAndWait` (`:462`, 2 callers), `makeCommandBufferUnsafe` (`:298`, used by `LearnedDistanceKernel.swift:669` and `QuantizationStatisticsKernel.swift:346`), and direct `context.commandQueue.makeCommandBuffer()` (`KMeansAssignKernel.swift:130`, `KMeansUpdateKernel.swift:93`). Only the first three update timing and the audit trace. Two queue factories are dead (`MetalDevice.getCommandQueue/makeCommandQueue`, `KernelContext.commandQueue`).
- **No Metal 4 command API is in use.** `grep MTL4` over `Sources` finds error strings, capability flag names, and comments. `ArgumentTablePool.Metal4ArgumentTable` is a dictionary replayed through `encoder.setBuffer` (address bindings written by `setAddress` are never read; `setAddress` has zero callers). `ResidencyManager` declares its own `MTLResidencySet`, `MTLResidencySetDescriptor`, and `MTLDevice.makeResidencySet` **shadowing the real Metal symbols**; its `commit()` is a no-op; nothing is attached to a queue.
- **Binding shapes.** Maximum buffer index anywhere is 8 (`NeuralQuantizationKernel.swift:790`); typical dispatch binds 3–4 buffers plus 1–3 scalars (~5.2 bindings per dispatch over 116 dispatch sites). All 176 `setBytes` payloads are POD structs or scalars; the largest is 40 bytes (`AttentionSimilarityParameters`). No kernel binds the same buffer at two indices. Nine sites pass a non-zero `offset:` (`TopKSelectionKernel.swift:300-302`, `FusedL2TopKKernel.swift:1129-1131`, `JaccardDistanceKernel.swift:386-393`). Six `setThreadgroupMemoryLength` sites in five kernels. `ParameterBinding.apply` (`Kernels/Metal4/KernelProtocol.swift:299`) is the generic `setBytes` path.
- **Barriers and blits.** All 53 `memoryBarrier(scope:)` calls use `.buffers`. `BarrierHelper.swift` emits that same call from every branch and ignores its resource lists. Three blit users: `KMeansUpdateKernel` (two `fill`s then two compute encoders in one buffer), `StreamingTopKKernel` (ends the context's encoder, blits two copies, opens a second compute encoder, after which the context calls `endEncoding()` a second time), `ParallelReductionKernel` (two readback copies via `executeBlitAndWait`).
- **Lifetime.** `BufferToken.keepAlive(until:)` is `commandBuffer.addCompletedHandler { [self] _ in _ = self }` (`BufferPool.swift:179`), 43 call sites. A token dropped without it returns its buffer to the pool immediately on `deinit`, while the GPU may still be reading it.
- **Concurrency.** `execute*` closures are `async`; `VectorCoreIntegration.swift:68` and `:98` suspend between `setBytes` and `dispatchThreadgroups`. The actor is reentrant; multiple command buffers can be open. Contexts are created per provider (`KernelDistanceProviders.swift` builds six), per index, per engine.
- **Wait.** Shared event; completed handler assigns `signaledValue = ticket` (absolute, not monotonic); `MTLSharedEventListener` created inline and unretained; `Task.checkCancellation()` at `:424` runs after `commit()`, so a cancelled call returns while the GPU is still executing.
- **Pipelines.** ~130 `makeComputePipelineState(function:)` sites; the only descriptor path is `ArchivePipelineCache.swift:361-388` (`binaryArchives`, `label`). No function constants are used.
- **Metal 4 API facts verified** (spike source and Apple/objc2 documentation): `device.makeMTL4CommandQueue()`, `device.makeCommandAllocator()`, `device.makeCommandBuffer()` (Metal 4 buffer), `commandBuffer.beginCommandBuffer(allocator:)`, `endCommandBuffer()`, `makeComputeCommandEncoder()`, `encoder.setArgumentTable(_:)`, `MTL4ArgumentTableDescriptor.maxBufferBindCount`, `device.makeArgumentTable(descriptor:)`, `argumentTable.setAddress(_:index:)`, `encoder.dispatchThreads(threadsPerGrid:threadsPerThreadgroup:)`, `dispatchThreadgroups(_:threadsPerThreadgroup:)`, `setThreadgroupMemoryLength(_:index:)`, copy and fill on the compute encoder (`copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`, `fillBuffer:range:value:`), intra-encoder barrier `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` with `MTLStages` `.dispatch`, `encoder.writeTimestamp(granularity:counterHeap:index:)`, `MTL4CommitOptions.addFeedbackHandler`, `queue.commit([cb], options:)`, `queue.signalEvent(_:value:)`, `MTLResidencySetDescriptor`, `device.makeResidencySet(descriptor:)`, `addAllocation`/`removeAllocation`/`commit()`, `queue.addResidencySet(_:)`, `commandBuffer.useResidencySet(_:)`, `MTL4CommitFeedback.gpuStartTime`/`gpuEndTime`/`error` (host seconds), `MTL4CounterHeapDescriptor`, `device.makeCounterHeap(descriptor:)`, `heap.resolveCounterRange(_:)` (throws, returns `Data?` of packed `UInt64`), `device.queryTimestampFrequency()` (24 MHz on M3 Max). Argument-table entries are collected at dispatch time, so a table may be rebound between dispatches in one encoder. Metal 4 command buffers do not retain resources. The exact `MTL4VisibilityOptions` case for device-scope buffer visibility is confirmed against the SDK header in stage 0 (section 14).

## 3. Architecture

```
Metal4Context (actor)
├─ queue: MTL4CommandQueue                      one per context
├─ allocators: [Slot]  (maxInFlight, default 4)  Slot = { allocator, commandBuffer, constantRing, heapSlots }
├─ event: MTLSharedEvent + ticket: UInt64        monotonic; one retained MTLSharedEventListener
├─ residency: ResidencyRegistry                  one MTLResidencySet attached to the queue + membership set
├─ argumentTables: ArgumentTablePool             real MTL4ArgumentTable wrappers, bind count 16
├─ timing: TimingRegistry                        counter heap ring (opt-in), SubmissionRecord ring
└─ submit(label:) { KernelEncoder in ... } -> SubmissionRecord
```

`KernelEncoder` (struct, public) is the only thing kernels see. `BufferPool` learns about submissions (in-flight lease tracking) but not about Metal 4 types beyond `MTLBuffer`.

## 4. Command model

### 4.1 Context members

| Member | Type | Notes |
|---|---|---|
| `queue` | `any MTL4CommandQueue` | created in `init` from `device.rawDevice.makeMTL4CommandQueue()`; label from configuration |
| `slots` | `[SubmissionSlot]` | `maxInFlight` entries (configuration, default 4, minimum 1). Each: `allocator: MTL4CommandAllocator`, `commandBuffer: MTL4CommandBuffer` (reused; re-begun per submission), `constantRing: ConstantRing`, `heapBase: Int` (2 counter-heap slots), `inUse: Bool` |
| `event`, `ticket` | `MTLSharedEvent`, `UInt64` | ticket increments per submission; `signalEvent` in commit order keeps the event monotonic |
| `listener` | `MTLSharedEventListener` | created once in `init` on a dedicated serial `DispatchQueue`; retained for the context's lifetime |
| `residency` | `ResidencyRegistry` | section 6 |
| `argumentTablePool` | `ArgumentTablePool` | existing actor, now vending real tables (section 5.2); `Metal4ComputeEngine`'s second pool is removed |
| `timing` | `TimingRegistry` | section 7 |
| `instrumentation` | `InstrumentationConfiguration` | section 7, read once |

### 4.2 The submission API

```swift
public struct SubmissionRecord: Sendable { ... }   // section 7.1

extension Metal4Context {
    /// Encode one command buffer with one compute encoder, commit it, and wait for completion.
    /// The closure runs synchronously on the actor; it must not suspend.
    public func submit(
        label: String,
        transientResidency: [any MTLBuffer] = [],
        _ encode: (inout KernelEncoder) throws -> Void
    ) async throws -> SubmissionRecord
}
```

`execute`, `executeAndWait`, `executeBlitAndWait`, `makeCommandBuffer`, and `makeCommandBufferUnsafe` remain until stage 5 and are deleted there. New code never calls them.

### 4.3 Submission steps (normative)

1. **Acquire a slot.** If every slot is in flight, wait on the context's `slotAvailable` continuation queue (FIFO) with a timeout of `configuration.submitTimeout` (default 30 s); on timeout throw `VectorError.submissionTimeout(label:)`. Mark the slot in use.
2. `slot.allocator.reset()`; `slot.constantRing.reset()`.
3. `slot.commandBuffer.beginCommandBuffer(allocator: slot.allocator)`; `commandBuffer.label = label`.
4. **Residency.** If `residency.isDirty`, `residency.set.commit()` and count it. If `transientResidency` is non-empty, build a per-submission `MTLResidencySet` from it, commit it, and attach it with `commandBuffer.useResidencySet(_:)` before the encoder is created (section 6.3). Residency is fully decided before any command is encoded.
5. `encoder = commandBuffer.makeComputeCommandEncoder()`; `encoder.label = label`; acquire an argument table from the pool; `encoder.setArgumentTable(table)`.
6. If `instrumentation.timestamps`: `encoder.writeTimestamp(granularity: .relaxed, counterHeap: heap, index: slot.heapBase)`.
7. **Run the closure** with `KernelEncoder(slot:table:encoder:context:)`. `cpuEncodeNs` is the `ContinuousClock` span of this step. Any throw: `encoder.endEncoding()`, `commandBuffer.endCommandBuffer()`, release the table and slot, rethrow. Nothing is committed (preserves `CapabilityCapPolicyTests.testThrowingEncodeClosurePropagatesWithoutCrash` semantics).
8. If timestamps: `writeTimestamp(... index: slot.heapBase + 1)`.
9. `encoder.endEncoding()`; `commandBuffer.endCommandBuffer()`.
10. `try Task.checkCancellation()` — the **only** cancellation check, before commit.
11. `ticket += 1`; build `MTL4CommitOptions` with a feedback handler that stores `gpuStartTime`, `gpuEndTime`, `error` into the slot's pending record; `queue.commit([slot.commandBuffer], options: options)`; `queue.signalEvent(event, value: ticket)`. No suspension point between steps 10 and 11.
12. **Wait.** `await withCheckedContinuation { event.notify(listener, atValue: ticket) { _, _ in resume } }`. `cpuWaitNs` is the span of this step. The call never returns before the GPU has completed the buffer; cancellation after commit is ignored (documented).
13. If timestamps: `heap.resolveCounterRange(slot.heapBase..<slot.heapBase+2)`; convert with the cached frequency.
14. Release retained objects (section 6.4), return the argument table to the pool, mark the slot free, resume one waiter from `slotAvailable`.
15. Assemble the `SubmissionRecord`, push it to the ring, print the `VA_AUDIT` line if enabled, return it.

### 4.4 Concurrency

Reentrancy is allowed and bounded by `maxInFlight`. Two submissions may be open (steps 5–9) simultaneously because the closure is synchronous but the actor is reentrant across the awaits in steps 1 and 12. Tickets are issued inside step 11 with no suspension between issue and commit, and Metal signals events in queue order, so `event.signaledValue` is monotonic and every waiter wakes at its own ticket. Each slot owns its allocator, command buffer, constant ring, and heap slots, so concurrent submissions never share encoder-side state. The four kernels that own command buffers today move onto `submit` in stages 3 and 4.

### 4.5 Errors

| Condition | Behaviour |
|---|---|
| Closure throws | encoder and buffer ended, not committed; slot and table released; error rethrown |
| Constant ring exhausted | `KernelEncoder.setBytes` throws `VectorError.constantRingExhausted(label:requested:capacity:)` (ring is 1 MiB per slot by default) |
| Argument index ≥ bind count | `VectorError.argumentIndexOutOfRange` (bind count 16, observed maximum 8) |
| Slot timeout | `VectorError.submissionTimeout` |
| Feedback reports an error | `SubmissionRecord.error` set; `submit` throws `VectorError.gpuExecutionFailed(label:underlying:)` after releasing resources |
| Cancellation before commit | `CancellationError`, nothing committed |
| Cancellation after commit | ignored; the call completes normally |

## 5. `KernelEncoder`

### 5.1 API (public struct, non-copyable in spirit: `inout` through the closure)

| Today (`any MTLComputeCommandEncoder`) | `KernelEncoder` | Implementation |
|---|---|---|
| `setComputePipelineState(p)` | `setPipeline(_ p: any MTLComputePipelineState)` | pass-through |
| `setBuffer(b, offset: o, index: i)` | `setBuffer(_ b: any MTLBuffer, offset: Int = 0, index: Int)` | `residency.ensure(b)`; `table.setAddress(b.gpuAddress + UInt64(o), index: i)`; retain `b` for the submission |
| — | `setBuffer(_ token: BufferToken, offset: Int = 0, index: Int)` | as above plus `token.holdForSubmission(id)` (section 6.4) |
| `setBytes(&v, length:, index: i)` | `setBytes<T>(_ v: T, index: Int) throws` | copy `v` into the slot's constant ring at a 16-byte-aligned cursor; bind the ring buffer's address + cursor; ring buffer is registered resident once at init |
| `setThreadgroupMemoryLength(n, index: i)` | same name | pass-through |
| `dispatchThreadgroups(g, threadsPerThreadgroup: t)` | same name | pass-through; increments `dispatchCount` on the record |
| `dispatchThreads(g, threadsPerThreadgroup: t)` | same name | pass-through to `dispatchThreads(threadsPerGrid:threadsPerThreadgroup:)` |
| `memoryBarrier(scope: .buffers)` | `barrier()` | `barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: <device-scope>)` |
| blit `fill(buffer:range:value:)` | `fill(_ b: any MTLBuffer, range: Range<Int>, value: UInt8)` | compute-encoder fill; residency ensured |
| blit `copy(from:sourceOffset:to:destinationOffset:size:)` | `copy(from:sourceOffset:to:destinationOffset:size:)` | compute-encoder copy; residency ensured |
| `label` / `pushDebugGroup` | `pushDebugGroup(_:)`, `popDebugGroup()`, `insertDebugSignpost(_:)` | pass-through |
| `ParameterBinding.apply(&v, to: encoder, at: i)` | `ParameterBinding.apply(&v, to: inout KernelEncoder, at: i) throws` | calls `setBytes` |

`KernelEncoder` also exposes read-only `deviceLimits` (max threadgroup memory, thread execution width of the current pipeline) so kernels stop reaching for `context`.

### 5.2 Argument tables

The existing `ArgumentTablePool` actor and `ArgumentTableToken` lifecycle stay. `Metal4ArgumentTable` becomes a wrapper around a real `any MTL4ArgumentTable` created from `MTL4ArgumentTableDescriptor` with `maxBufferBindCount = 16`; the dictionary bookkeeping, `apply(to:)`, and the `setBuffer(_:offset:index:)` protocol member are deleted; the `ArgumentTable` protocol keeps `setAddress(_:index:)`, `reset()`, `maxBufferBindCount`. One table per submission, set on the encoder once (step 5), rebound between dispatches as kernels call `setBuffer`/`setBytes`; `reset()` on return to the pool clears entries (table state is per-submission, never assumed from a previous one). The presets `.distance/.matrix/.quantization/.batch` are removed. `Metal4ComputeEngine`'s private second pool is removed; it uses the context's.

### 5.3 Constant ring

Per slot: one `MTLBuffer` (`.storageModeShared`, default 1 MiB, configurable) registered in the residency set at init. `setBytes` bumps a cursor aligned to 16 bytes, copies the value, and binds `ring.gpuAddress + cursor`. The ring is reset in step 2. The observed maximum payload is 40 bytes and a dispatch binds at most three constants, so the default holds several thousand dispatches per submission; exhaustion throws rather than wraps.

### 5.4 What the rewrite looks like

For a typical kernel the mechanical change is: the `encode` signature (`into encoder: inout KernelEncoder`), `try` on `setBytes`, `memoryBarrier(scope: .buffers)` → `barrier()`, and deletion of `keepAlive` calls and `commandBuffer:` parameters. The nine `offset:` sites are unchanged in form. Multi-encoder kernels collapse to one encoder: `KMeansUpdateKernel` becomes fill, fill, barrier, accumulate, barrier, normalize; `StreamingTopKKernel` becomes dispatch, barrier, copy, copy, barrier, finalize (it is deprecated per the handoff and may instead be deleted in stage 4 by owner choice); `ParallelReductionKernel`'s readback copies follow its dispatch in the same encoder. `LearnedDistanceKernel` and `QuantizationStatisticsKernel` replace their owned command buffers with `submit`. The two `VectorCoreIntegration` sites compute `calculateThreadGroups` before the closure.

## 6. Residency and lifetime

### 6.1 `ResidencyRegistry` (context-owned, actor-isolated)

- One `MTLResidencySet` created in `init` (`MTLResidencySetDescriptor.initialCapacity = configuration.residencySetCapacity`, existing default 256) and attached with `queue.addResidencySet(set)`.
- `members: Set<ObjectIdentifier>` mirrors the set's contents; `isDirty` is true after any add or remove until `commit()`.
- `register(_ buffer:)` / `unregister(_ buffer:)`: add or remove and mark dirty. Idempotent.
- `isResident(_ buffer:, in submission:) -> Bool` (called by `KernelEncoder.setBuffer`): true if the buffer is a member of the queue-level set or of the current submission's transient set; false otherwise, in which case the encoder throws (6.3).
- Commit happens in step 4 only, so a burst of registrations costs one commit. The record counts commits (`residencyCommits`).

### 6.2 Who registers what

| Buffer source | Registration |
|---|---|
| `BufferPool` allocations | the pool calls `registry.register` when it allocates a new `MTLBuffer` (not per lease) and `unregister` when it frees one (`clearCache`, `reset`, eviction). The pool receives the registry at init. |
| Constant rings, counter heaps' resolve buffers (if any) | context init |
| Long-lived non-pool buffers: `IVFGPUIndexStructure`, `IVFListStructure`, `IVFStructure` buffers, clustering state, `TensorManager` tensors, kernel-owned scratch (`BoruvkaMSTKernel`, `UMAPGradientKernel`, `StreamingTopKKernel`, `BatchMatrixKernel`, `NeuralQuantizationKernel` bias buffers) | `context.makeResident(_:)` at creation and `context.releaseResidency(_:)` on teardown; each owning type gets a one-line addition in its migration stage |
| User-owned buffers (`SoACandidateSet.buffer`, `MetalBuffer.buffer`) and anything unregistered | transient (6.3), automatically |

### 6.3 Transient residency

Metal 4 requires residency to be attached to the command buffer before commands are encoded, and binding happens inside the closure after the encoder is open, so residency cannot be discovered lazily during encoding. The rule is therefore explicit: buffers that are not queue-level members must be passed in `submit(transientResidency:)`; the context builds a per-submission `MTLResidencySet` from that list in step 4. `KernelEncoder.setBuffer` on a buffer that is in neither set **throws `VectorError.bufferNotResident(label:index:)`** at bind time. Callers that bind user-owned buffers (the SoA provider paths, `MetalBuffer` users) list them; the error names the label and index so an omission is diagnosed in one run and never reaches the GPU as a fault.

### 6.4 Lifetime replaces `keepAlive`

- `submit` keeps a `retained: [AnyObject]` list on the slot; `KernelEncoder.setBuffer` appends the `MTLBuffer` (and the `BufferToken` when the token overload is used). The list is cleared in step 14, after completion. Metal 4 buffers do not retain resources; this list is what keeps them alive.
- `BufferToken` gains `holdForSubmission(_ id:)` / `releaseFromSubmission(_ id:)` and an `inFlightCount`. `returnToPool()` and `deinit` enqueue the return only when `inFlightCount == 0`; otherwise the return is deferred and performed by the last `releaseFromSubmission`. A token dropped while a submission holds it therefore returns exactly once, after the GPU is done.
- `keepAlive(until:)` and all 43 call sites are deleted in the stage that migrates each caller.
- `BufferPool.PoolStatistics` gains `highWaterMarkBytes`, `liveLeaseCount`, `peakLiveLeaseCount`, `inFlightLeaseCount`, `evictionCount` (wish-list HW-09), with the existing lifecycle contract doc updated.

### 6.5 Deletions

`Core/ResidencyManager.swift` (the shim, its `ResidencyMode`, and the three shadow declarations), `Metal4Context.getBuffer(size:)`/`getBuffer(for:)` registration wrappers and `commitResidency()`, `MetalDevice.makeResidencyManager`, `SharedMetalConfiguration`'s residency-manager references (retargeted to the registry).

## 7. Timing and records

### 7.1 `SubmissionRecord`

```
SubmissionRecord {
  id: UInt64 (== ticket), label: String, kernelLabels: [String],   // labels pushed via pushDebugGroup or setPipeline labels
  cpuEncodeNs: UInt64, cpuWaitNs: UInt64,
  gpuStartHostSeconds: Double?, gpuEndHostSeconds: Double?,       // from commit feedback; nil if feedback did not arrive
  encoderSpan: (startTicks: UInt64, endTicks: UInt64, nanoseconds: Double)?,  // nil when timestamps are off
  dispatchCount: Int, tableBinds: Int, constantBytes: Int, residencyCommits: Int, transientBuffers: Int,
  error: (any Error)?
}
```

Kept in a bounded ring on the context (`configuration.recordRingCapacity`, default 64) and returned from `submit`. `Metal4Context.recentSubmissions(limit:)` reads the ring. The harness's `.commandBuffer` series comes from `gpuEnd − gpuStart`; `.encoder` from `encoderSpan.nanoseconds`. Rule enforced in code: `encoderSpan` is computed only from the two encoder stamps; the record never subtracts a buffer-level time from an encoder-level one.

### 7.2 `InstrumentationConfiguration`

`{ timestamps: Bool = false, records: Bool = true, auditTrace: Bool = env VA_AUDIT_TRACE, recordRingCapacity: Int = 64 }` on `Metal4Configuration`. Read once into stored properties. With `timestamps == false` no counter heap is created and steps 6, 8, 13 are skipped. The five `enableProfiling` flags (`Metal4Configuration`, `Metal4ComputeEngineConfiguration`, `ClusteringConfiguration`, `IVFConfiguration`, `IndexAccelerationConfiguration`) are deprecated with a message naming this type and removed in 0.9.0; while present they map to `records`.

### 7.3 Counter heap

One `MTL4CounterHeap` (`.timestamp`) per context when enabled, `entryCount = 2 × maxInFlight`; slot `k` uses entries `2k` and `2k+1`; `.relaxed` granularity (the spike measured no difference from `.precise` and the header warns `.precise` may split encoders). Resolved on the CPU in step 13; frequency from `queryTimestampFrequency()` cached at init.

### 7.4 Deprecations

`lastGPUTiming`, `GPUTimingInfo`, `totalComputeTime`, `computeOperationCount`, `getPerformanceStats()`, `resetPerformanceStats()`, and `Metal4ComputeEngine`'s mirror (`Metal4EngineStatistics.contextComputeTime`) are deprecated in 0.8.0 in favour of records and removed in 0.9.0. During 0.8.0 they are derived from the ring.

## 8. Public API changes (0.8.0)

**Removed:** `Metal4Context.execute`, `executeAndWait`, `executeBlitAndWait`, `makeCommandBuffer`, `makeCommandBufferUnsafe`; `MetalDevice.getCommandQueue`, `makeCommandQueue(label:)`; `KernelContext.commandQueue`; `ConcurrencyShims` extensions on `MTLCommandQueue`/`MTLCommandBuffer`; all of `BarrierHelper.swift` (`PipelineStage`, `BarrierType`, `BarrierDescriptor`, `HazardTracker`, `BarrierHelper`, `FusedPipelineBuilder`, the `MTLComputeCommandEncoder` extension); `BufferToken.keepAlive(until:)`; `ResidencyManager`, `ResidencyMode`, the shadow `MTLResidencySet`/`MTLResidencySetDescriptor`/`MTLDevice.makeResidencySet`; `ArgumentTable.setBuffer(_:offset:index:)` and the `ArgumentTableDescriptor` presets; `Metal4ComputeEngine`'s second pool.

**Changed:** every public kernel `encode*` entry point takes `into encoder: inout KernelEncoder` and drops `commandBuffer:` parameters (the 42 methods listed in the inventory across `Kernels/Metal4/` and `Index/Kernels/IVF/`); `ParameterBinding.apply` signature.

**Added:** `Metal4Context.submit(label:transientResidency:_:)`, `SubmissionRecord`, `recentSubmissions(limit:)`, `makeResident(_:)`, `releaseResidency(_:)`, `KernelEncoder`, `InstrumentationConfiguration`, `Metal4Configuration.maxInFlight`, `.submitTimeout`, `.constantRingBytes`; new `VectorError` cases named in section 4.5 and 6.3; pool statistics fields.

**Deprecated (removed 0.9.0):** section 7.2 and 7.4 items.

## 9. Migration stages

Every stage ends with: `swift test` and `swift test -c release` green (1764-equivalent, minus deletions, plus additions); `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test` green for the migrated suites (Metal 4 validation reports non-resident resources, which is the point); the harness comparison for the migrated kernels' cases showing no `alert(slower)` on `.encoder` or `.commandBuffer` beyond the boundary, with the wall-time change recorded. Each stage is one or more commits with the standard trailer and a ledger entry.

| Stage | Owner | Scope | Gate additions |
|---|---|---|---|
| **0 — core** | Metal agent | `Metal4Context` gains queue, slots, event/listener, `submit`, `ResidencyRegistry`, `TimingRegistry`, counter heap, `KernelEncoder`, real `Metal4ArgumentTable`, constant ring; `BufferPool` gains registry hookup, in-flight lease tracking, new statistics; new `VectorError` cases; `InstrumentationConfiguration`. **No kernel changes; legacy path untouched.** | all section 11.1 unit tests; a stage-0 smoke test that runs the spike's copy kernel through `submit` and asserts the record's fields |
| **1 — first kernel** | Metal agent | `L2DistanceKernel` + `L2KernelDistanceProvider` path in `KernelDistanceProviders.swift` (the harness M0 case) | parity gate for L2 (section 10); harness before/after for `distance.l2.batch.n=10000.d=768` |
| **2 — search path** | Metal agent | `CosineSimilarityKernel`, `DotProductKernel`, `SoADistanceKernel`, `FusedL2TopKKernel`, `TopKSelectionKernel`, `WarpOptimizedSelectionKernel`, the remaining `KernelDistanceProviders`, `MetalComputeProvider` + `+SoA`, `Metal4ComputeEngine` (15 execute sites), `VectorCoreIntegration` (fix the two await sites) | parity gate per kernel; `RoutingProvenanceTests`, `SoAKernelGoldenTests`, `DifferentialKernelVsCPUTests` green unchanged |
| **3 — IVF and clustering** | **IVF agent** | `IVFCoarseQuantizerKernel`, `IVFIndirectionDistanceKernel`, `IVFListSearchKernel`, `IVFGPUCandidateBuilderKernel`, `IVFSearchPipeline`, `KMeansAssignKernel`, `KMeansUpdateKernel`, `KMeansPlusPlusKernel`, `KMeansPipeline`, `ClusteringKernels`, `AcceleratedVectorIndex` IVF paths; `makeResident` for the IVF structures | parity gate per kernel; `IVFListInstrumentationTests` and the 33 hardening suites green; the IVF adapter's fixtures unchanged (their digests are input-side) |
| **4 — remainder** | Metal agent | `LearnedDistanceKernel`, `QuantizationStatisticsKernel`, `NeuralQuantizationKernel`, `ProductQuantizationKernel`, `ScalarQuantizationKernel`, `BinaryQuantizationKernel`, `UMAPGradientKernel`, `BoruvkaMSTKernel`, `MutualReachabilityKernel`, `SparseLogTFIDFKernel`, `LogSumExpKernel`, `MatrixMultiplyKernel`, `MatrixTransposeKernel`, `MatrixVectorKernel`, `BatchMatrixKernel`, `BatchMaxKernel`, `ElementwiseKernel`, `L2NormalizationKernel`, `MinkowskiDistanceKernel`, `JaccardDistanceKernel`, `HammingDistanceKernel`, `AttentionSimilarityKernel`, `ParallelReductionKernel`, `StreamingTopKKernel` (or delete it, owner's call), `LearnedDistanceService`, `MatrixEngine`, `BatchProcessor`, `QuantizationEngine`, `TensorManager` residency | parity gate per kernel |
| **5 — delete legacy** | Metal agent | remove section 8's "Removed" list; delete legacy branches in `Metal4Context`; retarget the 14 test call sites that bind the command buffer; freeze parity goldens (section 10.3); README, CHANGELOG `[0.8.0]`, `Package.swift` header entry; capability flags that claimed Metal 4 usage corrected | class-closing grep test (11.3) green; full gates; harness baselines recaptured as the 0.8.0 reference |

Stage 3 starts only after stage 2 lands, and the IVF agent's in-flight hardening slices land before stage 3 begins. Stages 4 and 3 may overlap because they touch disjoint files.

## 10. Parity harness

### 10.1 Mechanism

`Tests/VectorAccelerateTests/Migration/MigrationParityTests.swift` (harness owner) plus one small `<Kernel>ParityCase` per migrated kernel (the migrating agent). A parity case provides: seeded inputs from `VectorGenerator` across the ten adversarial value classes used by `DifferentialKernelVsCPUTests` (normal, zeros, duplicates, 1e19, 1e-20, subnormal, mixed scale, NaN, +Inf, −Inf), a `runLegacy(context) -> [Output]` and a `runNew(context) -> [Output]` closure, and an output digest function. The test asserts **bitwise-identical** output buffers between the two paths for every class and every shape in the kernel's sweep. Same shader, same pipeline, same inputs: any difference is a binding, offset, constant, barrier, or residency defect. Tolerance is never used here.

### 10.2 Shapes

Each case sweeps the dimensions and counts its kernel documents as supported plus the boundary values the hardening suites already use (D ∈ {1, 3, 7, 16, 33, 128, 384, 768}; N ∈ {1, 2, 33, 257, 1000}; K ∈ {1, 8, 9, 31, 32, 33, 64}), truncated to what the kernel accepts.

### 10.3 Golden freeze at stage 5

Before the legacy path is deleted, `MigrationParityTests` runs in `--record` mode once and writes `Tests/VectorAccelerateTests/Migration/Goldens/<kernel>.json` containing, per class and shape, the SHA-256 digest of the legacy output. After deletion the same test compares the new path against the stored digests. The goldens are committed with the stage-5 change and never regenerated silently: a mismatch is a defect or a deliberate, documented behaviour change with a new golden and a ledger entry.

## 11. Tests

### 11.1 Stage-0 unit tests (`Tests/VectorAccelerateTests/Metal4/`)

| Test | Asserts |
|---|---|
| `SubmitSlotTests` | with `maxInFlight = 2`, three concurrent submissions serialize the third; slot timeout throws; slots are released on closure throw |
| `SubmitTicketTests` | 100 concurrent submissions: records have strictly increasing ids; `event.signaledValue` never decreases (sampled from a completion observer) |
| `SubmitCancellationTests` | cancel before commit → `CancellationError`, no feedback; cancel after commit → completes, feedback present, output written |
| `SubmitRecordTests` | `gpuEnd ≥ gpuStart`; `cpuEncodeNs`, `cpuWaitNs` > 0; `dispatchCount`, `tableBinds`, `constantBytes` match the closure; `error == nil` |
| `TimestampTests` | with timestamps on: encoder span non-zero, `≤ gpuEnd − gpuStart`; off: no heap allocated (introspected), record field nil |
| `ResidencyRegistryTests` | pool allocation registers; free unregisters; exactly one commit per burst; unregistered bind without `transientResidency` throws `bufferNotResident`; with it, the submission succeeds under `MTL_DEBUG_LAYER=1` |
| `TokenLifetimeTests` | token dropped inside the closure returns to the pool only after completion (poll pool statistics during a long copy); `inFlightLeaseCount` and `peakLiveLeaseCount` exact for a scripted sequence |
| `ConstantRingTests` | alignment 16; exhaustion throws with the label; reset per submission |
| `ArgumentTableRebindTests` | two dispatches in one encoder binding different buffers at index 0 produce two correct outputs |
| `BarrierTests` | write-then-read across two dispatches with `barrier()` is correct for 10⁶ elements; the same without a barrier is not asserted (undefined), only that `barrier()` emits |
| `CopyFillTests` | `fill` then `copy` on the compute encoder match CPU expectations |
| `KernelEncoderAPITests` | every member in section 5.1 is exercised once against the spike copy kernel |

### 11.2 Per-kernel

The parity case (section 10) and the kernel's existing hardening and golden suites, unchanged in expectation.

### 11.3 Class-closing tests (harness owner, land at stage 0, red-until-stage-5 ones marked `XCTSkip` with the stage number until then)

- `NoLegacyCommandTypesTests`: greps `Sources/VectorAccelerate/**/*.swift` for `MTLCommandQueue`, `MTLCommandBuffer`, `MTLComputeCommandEncoder`, `MTLBlitCommandEncoder`, `makeCommandQueue(`, `addCompletedHandler`, `keepAlive(until` and fails on any hit (skipped until stage 5). Scoped to `Sources` per D3.
- `NoShadowMetalSymbolsTests`: fails if `Sources` declares a type named `MTLResidencySet`, `MTLResidencySetDescriptor`, or an extension `MTLDevice.makeResidencySet` (active from stage 0, since the shim is deleted then).
- `SingleSubmissionPathTests`: fails if any file outside `Metal4Context.swift` creates a command buffer or encoder (grep for `makeCommandBuffer(`, `makeComputeCommandEncoder(`); skipped until stage 5.
- `MetalCapabilityHonestyTests`: `Metal4Capabilities.supportsMTL4Compiler` and similar flags must be derived from a real query or removed (stage 5).

## 12. Error handling and failure modes

Covered by section 4.5 and 6.3. Two additional rules: a feedback handler that never fires (driver fault) is bounded by `submitTimeout` on the event wait, after which `submit` throws `VectorError.submissionTimeout` and the slot is quarantined (marked unusable, counted in statistics) rather than reused; a GPU error reported by feedback marks the slot for allocator reset and the record carries the error.

## 13. Performance expectations (to be measured, not claimed)

The harness records, per migrated kernel and per stage: `.encoder` span (expected unchanged, same shader), `.commandBuffer` span (expected within a few microseconds of the encoder span with timestamps off), and `.wall` (expected to fall, because argument-table binding replaces per-encode `setBuffer` calls and residency is no longer implicit; the spike measured roughly 0.2 ms of wall overhead above the GPU span on the legacy path for one dispatch). Any `alert(slower)` on `.encoder` blocks the stage; a wall-time increase is investigated but does not block. Counters in the record (`tableBinds`, `constantBytes`, `residencyCommits`) make the old master plan's targets (<10 residency commits per batch, >80% argument-table reuse) measurable for the first time.

## 14. Risks and unknowns

- **`MTL4VisibilityOptions` case name** for device-scope buffer visibility is confirmed against the SDK header in stage 0 (`BarrierTests` is the proof).
- **Argument-table entry count** is fixed at 16; a future kernel exceeding it throws at bind time (section 4.5), never silently.
- **Constant-ring sizing**: 1 MiB per slot is generous for observed use; the exhaustion error names the label so a new heavy kernel is diagnosed in one run.
- **`transientResidency` discipline** (section 6.3): callers binding user-owned buffers must list them. The error is loud and immediate; the SoA provider paths are the known sites and migrate in stage 2.
- **IVF hardening overlap**: stage 3 waits for in-flight IVF slices; the IVF agent owns its parity cases.
- **Slot exhaustion under heavy concurrency**: `maxInFlight` is configurable; the timeout error is explicit.
- **`StreamingTopKKernel`**: deprecated, contains the double-`endEncoding` defect; the owner decides in stage 4 between migrating and deleting.
- **Behaviour change in `reset()`/`clearCache()` semantics** of the pool when leases are in flight: the new deferral rule is stricter (a buffer cannot be freed while a submission holds it); the lifecycle contract doc is updated and `BufferPoolLifecycleTests` extended.

## 15. Acceptance for 0.8.0

1. Section 11.3 tests all green with no skips.
2. `swift test` and `swift test -c release` green; count reconciled in the ledger.
3. `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test` green in both configurations.
4. `MigrationParityTests` green against the frozen goldens for every migrated kernel.
5. Harness: every SP0/SP1 case reports `.encoder` spans with timestamps on and identical `.encoder` medians within the boundary against the pre-migration baseline; the 0.8.0 baseline is captured and approved.
6. README states the Metal 4 command model accurately; `Package.swift` header and CHANGELOG `[0.8.0]` list the removals and additions in section 8.

## 16. Coordination

- **Metal agent review (before any plan):** feasibility of sections 4–7 against `Metal4Context.swift`, `BufferPool.swift`, `ArgumentTablePool.swift`; the `transientResidency` rule; slot and ring defaults; anything in section 2 that is stale relative to their in-flight slices. Findings are folded into this spec before the stage-0 plan is written.
- **IVF agent:** stage 3 scope and the parity-case shape; their adapter reads `SubmissionRecord` for `.commandBuffer` and `.encoder` timing once stage 0 lands (the interface contract's timing kinds are unchanged).
- **Harness owner:** stage-0 tests in 11.1 and 11.3, the parity harness in 10.1, and the SP0/SP1 cases that gate each stage.
