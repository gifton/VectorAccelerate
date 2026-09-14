# Buffer allocation bounds implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task after owner approval. Steps use checkbox syntax for tracking.

**Status:** PROVISIONAL — planning only. Do not execute production changes until the
owner resolves the design decisions and approves the resulting plan.

**Goal:** Every successful pool or bucketed-factory allocation supplies at least the
requested bytes; unsupported sizes fail before copying or registering a handle.

**Architecture:** This provisional plan enforces the existing 64 MiB pool cap. Validate
original sizes at allocation entry points, protect derived byte counts, and retain
standard buckets, zero-size leases and existing valid reuse behavior.

**Tech Stack:** Swift 6.2, Metal 4, XCTest, existing VectorError types.

**Spec:** [Allocation and optional-bias design](../specs/2026-09-13-allocation-and-bias-hardening-design.md), section A.

## Global constraints

- Decision A is unresolved. If larger-buffer support is selected, revise this plan first.
- Work on `gifton/metal-hardening-checkpoint`; no new dependency or VectorCore update.
- Preserve Metal 4/platform requirements and VectorCore 0.3.3.
- Preserve zero-byte requests as minimum-bucket leases; reject negative sizes/counts.
- Keep successful requests at/below 67,108,864 bytes compatible, subject to device/budget limits.
- Factory scope is `createBucketedBuffer` only. Direct aligned/vector upload bounds were
  addressed separately in slice 26; retain that shared copy/rounding logic and its tests.
  Pool-cap and pool-derived-size work in this plan remain unimplemented.
- No production changes have been made by the planning session.
- Do not remove IVF/neural actual-capacity checks after fixing the shared pool.
- Run GPU tests serially; record current results instead of copying historical counts.

## Task 1: Correct allocation entry-point bounds

**Files:**
- Modify `Sources/VectorAccelerate/Core/MetalBufferFactory.swift` — bucketed allocation validation/documentation.
- Modify `Sources/VectorAccelerate/Core/BufferPool.swift` — original-request bounds and capacity checks.
- Modify `Tests/VectorAccelerateTests/BufferPoolEnhancedTests.swift` — replace capped-success expectations.
- Create `Tests/VectorAccelerateTests/Hardening/BufferAllocationBoundsTests.swift` — precise boundary and bookkeeping assertions.

- [ ] Re-read the approved design and current source; confirm no intervening edits.
- [ ] Replace `testVeryLargeBuffer` with an exact invalid-size error expectation; strengthen
  `testLargeSizeHandling` so an undersized successful allocation cannot pass.
- [ ] Add tests for 64 MiB-1, 64 MiB and 64 MiB+1; negative input, Int.max, zero and ordinary
  bucket boundaries. Test `createBucketedBuffer` returns nil on unsupported input.
- [ ] Add at least one compatibility-handle entry-point check (`acquire`/`acquireBuffer`)
  and verify rejection does not retain a new handle/allocation. Use an isolated pool and
  stable stats baseline, accounting for pending returns.
- [ ] Run `swift test --filter 'BufferPoolEnhancedTests|BufferAllocationBoundsTests'` and
  preserve failure output proving old oversized success. Invalid fixtures must not write
  into undersized buffers or attempt enormous host allocations.
- [ ] Validate the original requested size before bucket lookup. Preserve the public
  lookup signature/capped lookup semantics, but explicitly separate lookup from allocation.
  The factory returns nil; the throwing pool uses the approved recoverable error contract.
- [ ] Check selected length against device and budget limits, and actual returned length
  before returning/registering a lease. Preserve cleanup/reuse and avoid overflow in touched
  budget comparisons. Do not use traps to reject malformed allocation requests.
- [ ] Rerun the targeted filter; inspect exact failures and counts. Review the allocation
  contract with an independent reviewer before broadening to typed conveniences.

## Task 2: Protect derived requests and initialized uploads

**Files:** `BufferPool.swift`, `BufferAllocationBoundsTests.swift`, existing enhanced tests.

- [ ] Add red tests for typed negative/overflowing counts and invalid alignment values
  (zero, negative, non-power-of-two), including Int.max rounding overflow.
- [ ] Preserve and exercise zero-count and empty-data success without dereferencing an
  empty source pointer. Verify valid alignment rounding and normal typed content round trips.
- [ ] Test initialized just-over-cap input through both typed convenience spellings after
  direct rejection is established. Keep the fixture near the cap and release it promptly;
  no giant allocation is needed for arithmetic-overflow tests.
- [ ] Implement checked count multiplication and checked alignment rounding before calling
  the allocator. Use `invalidInput` for unrepresentable derived sizes rather than inventing
  a wrapped requested byte count. Ensure failed uploads never call `BufferToken.write`.
- [ ] Verify the two typed convenience paths and compatibility interfaces share the guarded
  allocator. Keep the low-level token's existing caller-preconditioned write API.
- [ ] Update `IVFCandidateBoundsTests.testLargeOutputsCannotPublishCountsBeyondPhysicalStorage`
  for the new early `invalidBufferSize` error (kind `.invalidData`), asserting the requested
  and maximum size details. Its old `.allocationFailed` expectation describes the local
  post-allocation capacity guard. Preserve its successful-result physical-storage assertions;
  do not replace the catch with acceptance of any VectorError.
- [ ] Run targeted pool tests and existing `IVFCandidateBoundsTests` and
  `NeuralEncodingScaleTests` serially; inspect any behavior changes before proceeding.

## Task 3: Final verification, contract and checkpoint

- [ ] Run full `swift test`, then `swift test -c release`, preserving exit codes and logs
  under a dated `/private/tmp/va-buffer-bounds/` directory. Confirm GPU-backed tests ran;
  never count skips as passes. Run once per final production/test state.
- [ ] Add `docs/stability/BUFFER-ALLOCATION-CONTRACT.md`: supported sizes/errors, zero and
  typed behavior, factory-vs-pool policy, byte rounding, and retained caller requirements.
- [ ] Update IVF/neural contracts to cross-link the central correction while retaining their
  local guards. Update handoff, audit evidence and the regression-pattern record.
- [ ] Obtain read-only review of the final diff and resolve material findings. No general
  pool lifecycle correctness claim should be inferred from this bounded size fix.
- [ ] Check diff/branch/source hashes, then commit and push this checkpoint on the existing
  branch under the standing workflow; verify remote HEAD and update session memory.

**Acceptance:** No successful undersized allocation, no malformed-request arithmetic trap,
no typed-copy precondition for an allocation request that should have thrown, valid boundary
allocations/reuse remain correct, and complete debug/release gates pass.
