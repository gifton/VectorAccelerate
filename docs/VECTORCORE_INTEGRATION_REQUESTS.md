# VectorCore Integration Requests (from VectorAccelerate)

Tracks changes VectorAccelerate needs from VectorCore to advance the CPU↔GPU integration,
mirroring VectorCore's own cross-package-request convention (see its `ROADMAP.md` §11.2 /
"Downstream Package Requests"). Filed against **VectorCore ≥ 0.2.2** (BE3).

## Context

VectorCore's BE3 audit (`Docs/beta-evolution-3`) culminates in **Phase 4 — "Metal Compute
Pipeline Prep"**: page-aligning CPU storage so a GPU layer can map it into Metal with
`MTLDevice.makeBuffer(bytesNoCopy:)` (Apple-Silicon UMA zero-copy) instead of paying a blocking
`memcpy` across the memory bus. VectorAccelerate *is* that GPU layer. The roadmap reserves the
hooks (`ComputeProvider`/`ComputeDevice(.gpu)` §11.2, batch `DistanceMetric` §2.2, pointer Top-K
§9.3 "zero-copy from GPU buffers", "Metal acceleration for large matrices" §9.2).

On our side we've landed the consuming primitive: `MetalDevice.makeNoCopyBuffer(bytes:length:…)`
wraps a page-aligned region as a shared `MTLBuffer` with no copy, falling back to `nil` (→ staged
copy) when alignment isn't met. It's tested and ready to consume aligned VectorCore storage.

## Requests

### R1 — Page-align the SoA batch buffer (P1, blocks zero-copy batch search)

> **✅ RESOLVED — VectorCore 0.3.0.** `SoA.build(from:pageAligned: true)` ships the page-aligned batch
> buffer; consumed by `SoACandidateSet` (zero-copy via `makeNoCopyBuffer`, borrow mode) and scored by
> the lane-major `SoADistanceKernel`. Validated on Apple Silicon (golden fixture + 512/768/1536 parity).

`SoA<Vector>` (the batch candidate-database layout) allocates via
`UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity:)` ≈ **16-byte** alignment. The BE3 Phase-4
page alignment landed on `AlignedMemory`/`AlignedDynamicArrayStorage` (now `getpagesize()` ≈ 16 KB),
but **not** on the `SoA` buffer — which is exactly the high-value object to bridge zero-copy for
large-batch GPU search.

**Ask:** allocate the `SoA` buffer through `AlignedMemory` (page-aligned), or expose an opt-in
page-aligned variant. Without it, `makeNoCopyBuffer` rejects the SoA pointer and we fall back to a
full candidate-database copy.

### R2 — Publicly expose the SoA buffer pointer + byte length (P1)

> **✅ RESOLVED — VectorCore 0.3.0.** `SoA.pageAlignedBytes` (base + page-rounded length),
> `consumeAllocation()`, and the `SoALayout` descriptor are public/stable. Lifetime contract honored:
> `SoACandidateSet` holds the `SoA` strongly (borrow mode) for the `MTLBuffer`'s life.

`SoA.buffer` is `@usableFromInline internal`. We need a public, stable accessor —
`withUnsafeRawBuffer { (ptr, byteCount) in … }` or `var rawBufferBytes: (UnsafeRawPointer, Int)` —
so we can hand the base pointer to `makeNoCopyBuffer`. Document the **lifetime contract**: the
`SoA` must outlive the wrapping `MTLBuffer` (we'll hold a strong reference for the buffer's life, or
honor a deallocator handshake).

### R3 — Confirm the release that ships AlignedMemory page alignment

> **✅ RESOLVED.** Shipped in the **VectorCore 0.3.0** tag (pinned `from: "0.3.0"`); the page-size
> handshake (16 KB) is validated by the SoA bridge smoke-test and golden-fixture parity.

The 0.2.2 release notes cover BE3 Phases 1–3 only; Phase-4 alignment appears in the working copy
(ahead of the 0.2.1 tag) but we could **not confirm it's in the `v0.2.2` tag** we pin. Please
confirm the version (and bump our floor if it's a later release).

### R4 — A kernel-injection hook so `ComputeProvider` can supply GPU kernels (P2, unlocks transparent dispatch)

> **✅ DELIVERED (VectorCore 0.3.0) + ADOPTED.** VectorCore 0.3.0 ships `BatchKernelProvider` exactly as
> proposed below, and `Operations.findNearest` / `findNearestBatch` downcast to it. `MetalComputeProvider`
> now conforms — euclidean/cosine route to the fused GPU kernel; other metrics defer to the metric's own
> `batchDistance`. The original ask is retained below for context.

`ComputeProvider` is a *work-scheduler*: `execute` / `parallelExecute` / `parallelReduce` take opaque
`@Sendable` closures, and the closures VectorCore passes already call its own CPU kernels
(`BatchKernels.range_euclid_512`, `TopKSelectionKernels.range_topk_euclid2_512`). A provider can only
choose *how to schedule* that closure — it cannot substitute a Metal kernel. So a `MetalComputeProvider`
installed as `Operations.computeProvider` accelerates nothing, and `findNearestGPU(...)` is a private
stub that throws and is unreachable from the public API.

**Ask:** add a sub-protocol VectorCore's `Operations` downcasts to, so a provider can offer real batch
kernels:

```swift
public protocol BatchKernelProvider: ComputeProvider {
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
```

Then in `Operations.findNearest` / the batch-distance path:

```swift
if let gpu = Operations.computeProvider as? BatchKernelProvider {
    return try await gpu.findNearest(query: query, candidates: vectors, k: k, metric: metric)
}
// …existing CPU path…
```

VectorAccelerate's `MetalComputeProvider` **now conforms to `BatchKernelProvider`** (VectorCore 0.3.0): a
thin adapter over its existing `SupportedDistanceMetric` kernels that maps `any DistanceMetric` to the
euclidean/cosine GPU paths and defers every other metric to that metric's own `batchDistance` (so
semantics never diverge — VectorCore maps `dotProduct` to −dot, which the raw-dot GPU kernel does not).

**Payoff (realized):** GPU acceleration is now transparent through VectorCore's own
`findNearest`/`findNearestBatch` — no separate VectorAccelerate entry point needed — completing the §11.2
"ComputeProvider/ComputeDevice(.gpu)" hook.

### R5 — Freeze the SoAFP16 layout contract (P2, blocks the 0.7 FP16 bridge)

> **OPEN — requested 2026-08-14.**

`SoA<Vector>` in VectorCore has a documented layout contract (`Docs/SoA_Layout_Contract.md` upstream), with a stable descriptor type (`SoALayout`) and page-aligned allocation (`pageAlignedBytes`). VectorAccelerate plans to bridge `SoAFP16` (half-precision) zero-copy in version 0.7, but the FP32 layout contract does not cover the FP16 case: the new layout groups elements in fours (SIMD4<Float16>) and stores dimension-first, per the zero-copy planning brief §7. Without an upstream contract, we cannot reliably consume the buffer via `makeNoCopyBuffer`.

**Ask:** (a) a frozen, documented layout formula for `SoAFP16` (groups-of-4, dimension-first, with the rounding logic for ragged vectors) and a descriptor surface equivalent to `SoALayout` (dimensions, groups, page alignment, byte offsets); (b) a page-aligned build option for `SoA<Vector16>` parallel to the current `SoA.build(from:pageAligned: true)` for FP32; (c) a golden fixture equivalent to the FP32 one, so VA can validate the same way.

## Expected payoff

For large-batch GPU search the candidate-database transfer dominates latency. R1+R2 turn that
copy into a pointer hand-off — the single biggest win on the GPU path, and the natural completion
of VectorAccelerate's zero-copy staging work (PR #30, T2a).

## Upstream defect reports from the 0.6.0 normalization parity work (BE3 §4.4)

Measured during the normalization parity audit (Task 4, fix-round discussions); all three reflect the BE3 §4.4 fix landing in `NormalizeKernels` but not in the generic `VectorProtocol` paths or the CPU reference implementation. Status for all three: **OPEN — reported 2026-08-15.**

**D1 — Generic `Vector<D>.normalized()` returns +Inf-poisoned output for all-subnormal input.** File/line: VectorCore `Operations/VectorNormalization.swift`. Measured: with all-subnormal input (1e-40, ‖v‖₂ = 2.26e-39), the generic `Vector<D>.normalized()` returns every element as `+Inf` (Table §1.2, Task 4 report). VectorAccelerate's `StableNormalization` handles all-subnormal vectors by passing the input unchanged, matching the CPU reference path `normalizeUnchecked`.

**D2 — `NormalizeKernels.normalizeUnchecked` reconstructs the magnitude with the unclamped `maxAbs` while dividing by the clamped one.** File/line: VectorCore `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` (steps 4 & 6). Measured: for [1e-38]×512 (all-subnormal, genuinely normalizable to unit), step 4 divides by `max(maxAbs, FLT_MIN)` but step 6 reconstructs as `mag = maxAbs · sqrt(sumSquares)`, yielding output norm ≈ 1.1755 instead of 1 (Task 4, §1.4). VectorAccelerate reconstructs as `‖v‖ = den · sNorm` (clamped denominator), which is exact for every input class and yields unit norm precisely.

**D3 — `NormalizeKernels.normalizeUnchecked` returns all-zeros for any vector with ‖v‖₂ > FLT_MAX.** File/line: VectorCore `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` (steps 6–8: magnitude overflow and finiteness guard). Measured: with `FLT_MAX` elements at dimension 512, `mag` overflows to `+Inf`; step 8's `guard invMag.isFinite` passes (since 1/Inf = 0.0 is finite), scaling the vector by 0.0 → all-zeros (elements 1e38, 2^126, `FLT_MAX` all measured in Task 4, §9.3). VectorAccelerate never forms 1/‖v‖, instead dividing in the pre-scaled domain (v·scale)/sNorm, avoiding overflow and yielding the exact unit vector.

## Adjacent (not blocking, tracked)

- **ComputeProvider(.gpu) backend** (their §11.2): ✅ `MetalComputeProvider` shipped — a GPU façade
  (batch distance / k-NN / distance matrix with `GPUDecisionEngine` routing + CPU fallback) that now
  conforms to **`BatchKernelProvider`** (R4), so transparent *dispatch* through VectorCore's own
  Operations is live (euclidean/cosine on the GPU; other metrics fall back to the metric's `batchDistance`).
- **Pointer Top-K** (their §9.3, shipped): feed our GPU distance buffers into VectorCore's
  `select(k:from:UnsafePointer<Float>,count:,ids:)` for the hybrid path (no distance copy-back).
- **Numerical parity** (BE3 Phase 1): cosine zero-vector floor aligned to
  `leastNormalMagnitude`; normalization subnormal handling (BE3 4.4) audited and fixed in the
  0.6.0 branch — all five VA normalize implementations unified on VectorCore's algorithm, and
  three upstream defects filed above (D1–D3).
