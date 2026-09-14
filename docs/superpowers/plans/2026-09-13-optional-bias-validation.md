# Optional-bias validation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task after owner approval. Steps use checkbox syntax for tracking.

**Status:** COMPLETE — owner approved the source-breaking throwing `encodeFused` change
on 2026-09-13; implemented and verified as audit slice 27.

**Goal:** Preserve no-bias computation with valid Metal bindings and reject invalid or
unallocated requested bias before GPU encoding.

**Architecture:** Reuse neural zero storage for float encoding; validate batch bias layout
and capacity on a throwing raw entry point; supply a persistent disabled-bias binding if
required by Metal validation. Retain existing arithmetic and raw shader ABI.

**Tech Stack:** Swift 6.2, Metal 4, XCTest, Metal API/shader validation.

**Spec:** [Allocation and optional-bias design](../specs/2026-09-13-allocation-and-bias-hardening-design.md), section B.

## Global constraints

- Decision B is settled: make the existing raw `encodeFused` throwing and migrate callers.
- Work on `gifton/metal-hardening-checkpoint`; no new dependency or VectorCore update.
- Preserve Metal 4/platform requirements and VectorCore 0.3.3.
- Retain neural latent cap 128 and existing raw shader buffer bindings.
- Preserve batch bias/layout semantics, including nil and explicit `.none`; do not
  repurpose the legacy `config.hasBias` field during this fix.
- General raw matrix and neural input storage/layout remain caller-owned.
- The planning session made no production changes; this approved execution is separate.
- All rejected calls must finish validation before encoder mutation or dispatch.

## Task 1: Reproduce and correct float-only neural no-bias binding

**Files:**
- Modify `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`.
- Create `Tests/VectorAccelerateTests/Hardening/OptionalBiasValidationTests.swift`.
- Reuse existing neural scale/normalization and capability test fixtures where practical.

- [x] Read `encodeEncode`, the float shader, weight loading and zeroEncoderBias ownership.
- [x] Add analytic float-encoder fixtures with no bias, signed weights/output, activation
  on/off, ragged dimensions and L=128. Assert expected values and output canaries.
- [x] Run the focused fixture under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test
  --filter OptionalBiasValidationTests`. Capture any validation abort in a separate process.
  If the predicted missing-binding failure does not reproduce, record that accurately and
  inspect validation coverage before claiming a reproduced defect.
- [x] Before using the fixed-size fallback, add a red rejection test for an over-cap raw
  dispatch parameter; avoid actually dispatching an unsafe shape to prove bounds.
- [x] Reuse `zeroEncoderBias` at float encoder buffer(3), retaining real bias precedence;
  validate fallback capacity before encoder mutation. Keep allocation outside the hot path.
- [x] Verify loaded/unloaded/reloaded weight behavior and preserve the existing throwing
  unloaded-model behavior. Do not expose private model state just to inject test bias.
- [x] Raw shader explicit-bias controls may independently establish bias/ReLU ordering;
  label these separately from the public no-bias wrapper coverage.
- [x] Rerun focused Metal validation and existing neural guard suites.

## Task 2: Validate batch bias before encoding

**Files:** `BatchMatrixKernel.swift`, `BatchOptionalBiasValidationTests.swift`, existing batch tests.
The batch regressions use a separate suite so the neural and batch work can be reviewed independently.

- [x] Add raw fixtures for shared N-element bias and per-batch batchSize*N bias using
  distinct biases and multiple output rows/batches. Assert analytic values and canaries.
- [x] Add exact/oversized/one-element-short bias cases, nil bias, and `.none` with supplied
  storage. Include config.hasBias disagreement controls to preserve actual existing semantics.
- [x] Run valid controls against the baseline. For rejection tests, use the checked
  signature when introduced; document compile-only migration separately from behavioral red.
  Do not claim a compiler error is a reproduced out-of-bounds GPU failure.
- [x] Apply the approved throwing signature and migrate the internal `multiplyFused` call
  with `try`. Search all Sources/Tests/examples for callers; external migration goes in docs.
- [x] With the throwing signature in place but before bounds guards, record a short-bias
  call into an encoder that will be ended and discarded without committing its command
  buffer. Assert that it throws; observe failure on the unguarded implementation without
  executing an out-of-bounds GPU dispatch. Then add the pre-encoding bounds guards.
  Compute N*4 and batchSize*N*4 with overflow checks, require actual length >= required bytes,
  and reject wrong-device active bias. Do not require exact length on the raw API.
- [x] Derive no-bias mode from nil/`.none`. Validate under the Metal layers; if a bound argument
  is required, initialize one persistent Float32 dummy in the throwing initializer and bind
  it with mode zero. Verify the shader does not access disabled bias on both compile paths.
- [x] For rejected calls, catch the error inside a valid encoder, then submit a valid control
  operation and inspect poison outputs to prove the rejected call encoded no work.
- [x] Keep zero-work and activation behavior unchanged. Any unrelated A/B/output validation
  defects are recorded separately unless essential to the agreed bias contract.

## Task 3: Preserve requested high-level bias on allocation failure

**Files:** `BatchMatrixKernel.swift`, bias regression tests if a failure seam exists.

- [x] Confirm `multiplyFused` currently turns `device.makeBuffer` failure for nonnil bias
  into a nil buffer and no-bias mode. Preserve that source evidence.
- [x] Replace the optional assignment with a guard that throws `bufferAllocationFailed`
  for the requested bias size before entering executeAndWait; preserve existing shape checks.
- [x] Use an existing allocator-failure seam if available to exercise the failure branch.
  Do not exhaust GPU memory or add a broad allocator abstraction merely to force nil.
  If no safe seam exists, explicitly record the branch as source-reviewed rather than
  claiming a fault-injection test. Keep normal bias-content tests as behavioral coverage.

## Task 4: Verification, migration and checkpoint

- [x] Run focused bias tests under Metal API/shader validation for debug and release paths,
  serially and separately from any performance measurements. Preserve original and final logs.
- [x] Run existing batch, neural, capability and exception-cleanup guards, then full
  `swift test` and `swift test -c release` on final production/test sources.
- [x] Add an optional-bias contract and source migration note for `try encodeFused` if approved.
  Describe raw minimum lengths, disabled mode, allocation failure, lifetime and retained
  caller-owned requirements. Link from the earlier neural contracts and handoff.
- [x] Record exact gate counts and reproduction limits in the audit; obtain independent
  review of code, contract, source compatibility and validation evidence.
- [x] Commit/push the separate checkpoint on the existing branch, verify remote HEAD and
  update session memory. Do not bundle unrelated optional-buffer families into this slice.

**Acceptance:** Valid optional-bias calls pass Metal validation, active short bias is rejected
before encoding, requested high-level bias cannot silently disappear on allocation failure,
normal outputs/activation remain correct, and complete debug/release gates pass.


## Execution notes — 2026-09-13

- Owner approved decision B: existing raw `encodeFused` is throwing. Decision A remains open.
- Neural validation reproduced missing buffer(3); the raw L=129 test reproduced acceptance
  and encoder-label mutation. Persistent fallback plus pre-encoding guards fixed both.
- Batch short-bias rejection reproduced on a signature-only migration; the failing test
  ended/discarded the command buffer without GPU submission. Disabled bias reproduced a
  missing buffer(4) validation abort. No baseline valid-control run is claimed; final
  analytic controls verify preserved results.
- Zero-sized batch grids return without encoder mutation or dispatch, retaining their
  result geometry. Positive grid geometry is unchanged.
- The batch-size product widens to Int before checking multiplication and then byte size;
  no new UInt32 product cap is introduced. Both multiplication stages have boundary tests.
- No safe allocation-failure seam exists, so the high-level nil-allocation branch is
  source-reviewed. Wrong-device coverage is conditional within the active-layout test;
  it is source-reviewed on the current single-device host. No empty standalone test or
  additional skip is used to imply that branch was hardware-tested.
- Local evidence is under `/private/tmp/va-optional-bias/`; final gates and review are
  recorded in AUDIT-3 slice 27. All work stays on `gifton/metal-hardening-checkpoint`.
