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

`SoA<Vector>` (the batch candidate-database layout) allocates via
`UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity:)` ≈ **16-byte** alignment. The BE3 Phase-4
page alignment landed on `AlignedMemory`/`AlignedDynamicArrayStorage` (now `getpagesize()` ≈ 16 KB),
but **not** on the `SoA` buffer — which is exactly the high-value object to bridge zero-copy for
large-batch GPU search.

**Ask:** allocate the `SoA` buffer through `AlignedMemory` (page-aligned), or expose an opt-in
page-aligned variant. Without it, `makeNoCopyBuffer` rejects the SoA pointer and we fall back to a
full candidate-database copy.

### R2 — Publicly expose the SoA buffer pointer + byte length (P1)

`SoA.buffer` is `@usableFromInline internal`. We need a public, stable accessor —
`withUnsafeRawBuffer { (ptr, byteCount) in … }` or `var rawBufferBytes: (UnsafeRawPointer, Int)` —
so we can hand the base pointer to `makeNoCopyBuffer`. Document the **lifetime contract**: the
`SoA` must outlive the wrapping `MTLBuffer` (we'll hold a strong reference for the buffer's life, or
honor a deallocator handshake).

### R3 — Confirm the release that ships AlignedMemory page alignment

The 0.2.2 release notes cover BE3 Phases 1–3 only; Phase-4 alignment appears in the working copy
(ahead of the 0.2.1 tag) but we could **not confirm it's in the `v0.2.2` tag** we pin. Please
confirm the version (and bump our floor if it's a later release).

### R4 — A kernel-injection hook so `ComputeProvider` can supply GPU kernels (P2, unlocks transparent dispatch)

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

VectorAccelerate's `MetalComputeProvider` already implements matching `batchDistance`/`findNearest`
methods (today over `SupportedDistanceMetric`); conforming to `BatchKernelProvider` once VectorCore
defines it is a thin adapter — the routing/kernels are done.

**Payoff:** GPU acceleration becomes transparent through VectorCore's own `findNearest`/`batchDistance`
— no separate VectorAccelerate entry point needed — completing the §11.2 "ComputeProvider/ComputeDevice(.gpu)"
hook. We have the implementation ready (see `MetalComputeProvider`); only the protocol + downcast are owed.

## Expected payoff

For large-batch GPU search the candidate-database transfer dominates latency. R1+R2 turn that
copy into a pointer hand-off — the single biggest win on the GPU path, and the natural completion
of VectorAccelerate's zero-copy staging work (PR #30, T2a).

## Adjacent (not blocking, tracked)

- **ComputeProvider(.gpu) backend** (their §11.2): ✅ `MetalComputeProvider` shipped — a GPU façade
  (batch distance / k-NN / distance matrix with `GPUDecisionEngine` routing + CPU fallback) that
  conforms to `ComputeProvider` as a capability shim. Transparent *dispatch* (VectorCore routing work
  to us through its own Operations) now depends on **R4** above.
- **Pointer Top-K** (their §9.3, shipped): feed our GPU distance buffers into VectorCore's
  `select(k:from:UnsafePointer<Float>,count:,ids:)` for the hybrid path (no distance copy-back).
- **Numerical parity** (BE3 Phase 1): ongoing — cosine zero-vector floor aligned to
  `leastNormalMagnitude` (this branch); normalization subnormal handling (BE3 4.4) still to audit.
