# Optional-bias validation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task after owner approval. Steps use checkbox syntax for tracking.

**Status:** PROVISIONAL — planning only. Do not execute production changes until the
owner resolves the design decisions and approves the resulting plan.

**Goal:** Preserve no-bias computation with valid Metal bindings and reject invalid or
unallocated requested bias before GPU encoding.

**Architecture:** Reuse neural zero storage for float encoding; validate batch bias layout
and capacity on a throwing raw entry point; supply a persistent disabled-bias binding if
required by Metal validation. Retain existing arithmetic and raw shader ABI.

**Tech Stack:** Swift 6.2, Metal 4, XCTest, Metal API/shader validation.

**Spec:** [Allocation and optional-bias design](../specs/2026-09-13-allocation-and-bias-hardening-design.md), section B.

## Global constraints

- Decision B is unresolved. If signature preservation is chosen, revise migration and
  closure claims: a new checked API alone leaves the old raw entry point unchecked.
- Work on `gifton/metal-hardening-checkpoint`; no new dependency or VectorCore update.
- Preserve Metal 4/platform requirements and VectorCore 0.3.3.
- Retain neural latent cap 128 and existing raw shader buffer bindings.
- Preserve batch bias/layout semantics, including nil and explicit `.none`; do not
  repurpose the legacy `config.hasBias` field during this fix.
- General raw matrix and neural input storage/layout remain caller-owned.
- No production changes have been made by the planning session.
- All rejected calls must finish validation before encoder mutation or dispatch.

## Task 1: Reproduce and correct float-only neural no-bias binding

**Files:**
- Modify `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`.
- Create `Tests/VectorAccelerateTests/Hardening/OptionalBiasValidationTests.swift`.
- Reuse existing neural scale/normalization and capability test fixtures where practical.

- [ ] Read `encodeEncode`, the float shader, weight loading and zeroEncoderBias ownership.
- [ ] Add analytic float-encoder fixtures with no bias, signed weights/output, activation
  on/off, ragged dimensions and L=128. Assert expected values and output canaries.
- [ ] Run the focused fixture under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift test
  --filter OptionalBiasValidationTests`. Capture any validation abort in a separate process.
  If the predicted missing-binding failure does not reproduce, record that accurately and
  inspect validation coverage before claiming a reproduced defect.
- [ ] Before using the fixed-size fallback, add a red rejection test for an over-cap raw
  dispatch parameter; avoid actually dispatching an unsafe shape to prove bounds.
- [ ] Reuse `zeroEncoderBias` at float encoder buffer(3), retaining real bias precedence;
  validate fallback capacity before encoder mutation. Keep allocation outside the hot path.
- [ ] Verify loaded/unloaded/reloaded weight behavior and preserve the existing throwing
  unloaded-model behavior. Do not expose private model state just to inject test bias.
- [ ] Raw shader explicit-bias controls may independently establish bias/ReLU ordering;
  label these separately from the public no-bias wrapper coverage.
- [ ] Rerun focused Metal validation and existing neural guard suites.

## Task 2: Validate batch bias before encoding

**Files:** `BatchMatrixKernel.swift`, `OptionalBiasValidationTests.swift`, existing batch tests.

- [ ] Add raw fixtures for shared N-element bias and per-batch batchSize*N bias using
  distinct biases and multiple output rows/batches. Assert analytic values and canaries.
- [ ] Add exact/oversized/one-element-short bias cases, nil bias, and `.none` with supplied
  storage. Include config.hasBias disagreement controls to preserve actual existing semantics.
- [ ] Run valid controls against the baseline. For rejection tests, use the checked
  signature when introduced; document compile-only migration separately from behavioral red.
  Do not claim a compiler error is a reproduced out-of-bounds GPU failure.
- [ ] Apply the approved throwing signature and migrate the internal `multiplyFused` call
  with `try`. Search all Sources/Tests/examples for callers; external migration goes in docs.
- [ ] With the throwing signature in place but before bounds guards, record a short-bias
  call into an encoder that will be ended and discarded without committing its command
  buffer. Assert that it throws; observe failure on the unguarded implementation without
  executing an out-of-bounds GPU dispatch. Then add the pre-encoding bounds guards.
  Compute N*4 and batchSize*N*4 with overflow checks, require actual length >= required bytes,
  and reject wrong-device active bias. Do not require exact length on the raw API.
- [ ] Derive no-bias mode from nil/`.none`. Validate under the Metal layers; if a bound argument
  is required, initialize one persistent Float32 dummy in the throwing initializer and bind
  it with mode zero. Verify the shader does not access disabled bias on both compile paths.
- [ ] For rejected calls, catch the error inside a valid encoder, then submit a valid control
  operation and inspect poison outputs to prove the rejected call encoded no work.
- [ ] Keep zero-work and activation behavior unchanged. Any unrelated A/B/output validation
  defects are recorded separately unless essential to the agreed bias contract.

## Task 3: Preserve requested high-level bias on allocation failure

**Files:** `BatchMatrixKernel.swift`, bias regression tests if a failure seam exists.

- [ ] Confirm `multiplyFused` currently turns `device.makeBuffer` failure for nonnil bias
  into a nil buffer and no-bias mode. Preserve that source evidence.
- [ ] Replace the optional assignment with a guard that throws `bufferAllocationFailed`
  for the requested bias size before entering executeAndWait; preserve existing shape checks.
- [ ] Use an existing allocator-failure seam if available to exercise the failure branch.
  Do not exhaust GPU memory or add a broad allocator abstraction merely to force nil.
  If no safe seam exists, explicitly record the branch as source-reviewed rather than
  claiming a fault-injection test. Keep normal bias-content tests as behavioral coverage.

## Task 4: Verification, migration and checkpoint

- [ ] Run focused bias tests under Metal API/shader validation for debug and release paths,
  serially and separately from any performance measurements. Preserve original and final logs.
- [ ] Run existing batch, neural, capability and exception-cleanup guards, then full
  `swift test` and `swift test -c release` on final production/test sources.
- [ ] Add an optional-bias contract and source migration note for `try encodeFused` if approved.
  Describe raw minimum lengths, disabled mode, allocation failure, lifetime and retained
  caller-owned requirements. Link from the earlier neural contracts and handoff.
- [ ] Record exact gate counts and reproduction limits in the audit; obtain independent
  review of code, contract, source compatibility and validation evidence.
- [ ] Commit/push the separate checkpoint on the existing branch, verify remote HEAD and
  update session memory. Do not bundle unrelated optional-buffer families into this slice.

**Acceptance:** Valid optional-bias calls pass Metal validation, active short bias is rejected
before encoding, requested high-level bias cannot silently disappear on allocation failure,
normal outputs/activation remain correct, and complete debug/release gates pass.
