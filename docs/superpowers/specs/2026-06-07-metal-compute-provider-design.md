# Design — `MetalComputeProvider`: VectorAccelerate's GPU compute façade

**Date:** 2026-06-07
**Status:** Approved (design), pending implementation plan
**Branch:** `gifton/metal-compute-provider` (stacked on `gifton/vectorcore-integration-groundwork` / PR #31)
**Supersedes planning in:** `docs/ADD/VECTORCORE_ALIGNMENT_ROADMAP.md` §1.1 (whose naïve `ComputeProvider` sketch is now known to be a no-op — see Finding below)

---

## Context & the core finding

VectorCore 0.2.2 reserves a GPU integration hook — its docs literally write
`Operations.computeProvider = MetalComputeProvider()` and comment *"GPU providers now live in
VectorAccelerate package."* The queued backlog item ("ComputeProvider(.gpu) backend") asks us to
fill it.

**The finding that reframes the task:** VectorCore's `ComputeProvider` is a **work-scheduler**, not a
kernel dispatcher. Its entire surface is `execute` / `parallelExecute` / `parallelReduce` over
opaque `@Sendable` closures — and the closures VectorCore passes already call *its own CPU SIMD
kernels* (`BatchKernels.range_euclid_512`, `TopKSelectionKernels.range_topk_euclid2_512`). A
provider only chooses *how to schedule* that closure across threads; it **cannot substitute a Metal
kernel** for the CPU math inside it. Consequently:

- A `MetalComputeProvider: ComputeProvider` installed as the task-local accelerates **nothing** — the
  math stays on CPU. The old `VECTORCORE_ALIGNMENT_ROADMAP.md` §1.1 sketch (a serial
  `for index in items { await work(index) }`) would in fact be **slower** than `CPUComputeProvider`'s
  TaskGroup.
- VectorCore's `findNearestGPU(...)` is a `private static` stub that just `throw`s
  `unsupportedDevice` and is unreachable from the public API (the task-local provider is always
  `.cpu`). It cannot be filled from a downstream package.

So real GPU acceleration of VectorCore-level operations requires a **new VectorCore hook** (filed
here as **R4**). What we *can* and *should* build now is a coherent, self-contained GPU provider that
unifies VectorAccelerate's already-working-but-scattered GPU surface behind one clean API — usable
today by callers who invoke it directly, and shaped so it slots into R4 transparently once VectorCore
lands the downcast.

### What already exists (and works) — we compose, not rebuild

- `Metal4ComputeEngine` (actor): real kernel engine — single + batch distance, `fusedDistanceTopK`,
  normalize/scale, matrix. Built `(context:configuration:decisionEngine:)`.
- `GPUDecisionEngine` (actor): `shouldUseGPU(operation:vectorCount:candidateCount:k:queryCount:dimension:) -> Bool`
  with adaptive thresholds + `recordPerformance(...)`.
- `KernelDistanceProviders`: the `stageScalars` zero-copy-staging seam (T2a) — fills Metal buffers
  via `withUnsafeBufferPointer`, no `.toArray()`.
- `AccelerateFallback`: CPU `batchEuclideanDistance` / `batchCosineSimilarity` + single-distance fns.
- `SupportedDistanceMetric` (VectorCore) is already the canonical metric enum.

### What's scattered (and will be deprecated)

GPU entry points are spread across **static** `BatchOperations.findNearestGPU / batchDistancesGPU /
pairwiseDistancesGPU` **and** the actor `AcceleratedDistanceProvider` **and** the
`VectorProtocol.acceleratedDistance` convenience extensions — all paying `.toArray()` copies, none
consulting `GPUDecisionEngine`, none behind one named provider.

---

## Goals / Non-goals

**Goals**
1. One canonical GPU entry point — `MetalComputeProvider` — for batch distance, k-NN (top-K),
   distance matrix, and single distance, with size-based GPU/CPU routing and CPU fallback.
2. Conform to VectorCore's `ComputeProvider` as an honest capability/identity shim.
3. Aggressively deprecate the superseded scattered surface to move users forward fast.
4. File VectorCore **R4** (transparent-dispatch hook) and conform our provider to the proposed
   protocol so our half is ready.

**Non-goals (this PR)**
- Zero-copy candidate-database wiring (`makeNoCopyBuffer`) — blocked on VectorCore R1/R2; we only
  leave the *seam*.
- New Metal batch kernels for non-euclidean/cosine metrics — those route to CPU batch.
- Folding vector ops (add/multiply/normalize/scale) into the provider — `AcceleratedVectorOperations`
  stays; the provider does not replace it, so it is **not** deprecated.

---

## Architecture

New single-purpose file `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`:

```swift
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor MetalComputeProvider: ComputeProvider {
    public struct Configuration: Sendable {
        public var preferGPU: Bool = true
        public var fallbackToCPU: Bool = true
    }

    private let context: Metal4Context
    private let engine: Metal4ComputeEngine
    private let decisionEngine: GPUDecisionEngine
    private let configuration: Configuration

    // Captured at init for the nonisolated ComputeProvider shim:
    private nonisolated let _device: ComputeDevice          // .gpu(index: 0)
    private nonisolated let _maxConcurrency: Int             // from Metal caps
    private nonisolated let _deviceInfo: ComputeDeviceInfo   // name/memory/threads from caps
}
```

It **owns nothing new** — it composes `Metal4Context` + `Metal4ComputeEngine` (kernels) +
`GPUDecisionEngine` (routing), and reuses the `stageScalars` seam + `AccelerateFallback`.

**Construction**
- `init(context:configuration:decisionEngine:) async throws` — primary.
- `init(configuration:) async throws` — builds a default context/engine/decision engine; guarded by
  `ComputeDevice.gpu().isAvailable` (throws `VectorError.metalNotAvailable()` otherwise).
- `static func makeDefault() async throws -> MetalComputeProvider`.

---

## Public API surface

### The real GPU value (dedicated methods)
Generic over `V: VectorProtocol where V.Scalar == Float`:

```swift
func batchDistance(query: V, candidates: [V], metric: SupportedDistanceMetric) async throws -> [Float]
func findNearest(query: V, in candidates: [V], k: Int, metric: SupportedDistanceMetric)
    async throws -> [(index: Int, distance: Float)]            // top-K
func distanceMatrix(queries: [V], candidates: [V], metric: SupportedDistanceMetric)
    async throws -> [[Float]]
func distance(_ a: V, _ b: V, metric: SupportedDistanceMetric) async throws -> Float   // completeness
```

### `ComputeProvider` conformance (the honest shim)
- `nonisolated var device` → `.gpu(index: 0)`
- `nonisolated var maxConcurrency`, `nonisolated var deviceInfo` → real Metal capabilities.
- `execute` → `try await work()`. `parallelExecute` / `parallelForEach` / `parallelReduce` → rely on
  VectorCore's **default TaskGroup implementations** (the protocol provides them).
- **Doc comment states plainly:** *"Installing this as `Operations.computeProvider` does not move
  VectorCore's Operations onto the GPU — those closures run CPU kernels (it will not be slower than
  CPUComputeProvider, but it will not accelerate). For GPU acceleration call the dedicated
  `batchDistance` / `findNearest` / `distanceMatrix` methods, or await VectorCore R4 (transparent
  dispatch)."*

---

## Data flow, decisioning, fallback (per call)

1. **Validate** — non-empty; uniform dimension via `candidates.allSatisfy { $0.withUnsafeBufferPointer { $0.count } == dim }`
   (mirrors the KernelDistanceProviders M1 guard).
2. **Route** — `decisionEngine.shouldUseGPU(operation:<mapped>, vectorCount:, candidateCount:, k:, dimension:)`.
   Metric → `GPUOperation` mapping: euclidean→`.l2Distance`, cosine→`.cosineSimilarity`,
   dotProduct→`.dotProduct`, manhattan→`.manhattanDistance`, chebyshev→`.chebyshevDistance`;
   `findNearest`→`.topKSelection`; `distanceMatrix`→`.distanceMatrix`. Honor `configuration.preferGPU`.
3. **GPU vote** → stage via `stageScalars` into pooled `Metal4Context` buffers → dispatch through
   `Metal4ComputeEngine` (`batchEuclideanDistance` / `batchCosineDistance` / `fusedDistanceTopK`).
   **Zero-copy seam:** the staging copy is the one line that flips to `makeNoCopyBuffer` once R1/R2 land.
4. **CPU vote / Metal unavailable / GPU error with `fallbackToCPU`** → `AccelerateFallback`.
5. **Adaptive feedback (optional)** → `decisionEngine.recordPerformance(...)`.

### Metric coverage (embedded call ①, accepted)
GPU batch kernels exist today for **euclidean + cosine** only. **dotProduct / manhattan / chebyshev**
route to the CPU batch path (a single-distance map over the CPU fallbacks — `AccelerateFallback` where
it has the metric, else the VectorCore `DistanceMetric` witness). Documented; **no new Metal kernels
this PR.** `findNearest` top-K uses `fusedDistanceTopK` for euclidean/cosine and a (batch distance →
CPU top-K heap) path otherwise.

---

## Consolidation — aggressive deprecation

All superseded **distance/search** GPU entry points are deprecated **now**, each with a migration
message and a removal target of **0.6.0** (the next minor), to create a clear forcing function.
Bodies are re-pointed to delegate to a shared `MetalComputeProvider` so behavior is preserved while
deprecated.

| Deprecated symbol | Replacement | Mechanism |
|---|---|---|
| `BatchOperations.findNearestGPU(to:in:k:metric:)` | `MetalComputeProvider.findNearest(query:in:k:metric:)` | delegate + `@available(*, deprecated, message:)` |
| `BatchOperations.batchDistancesGPU(from:to:metric:)` | `MetalComputeProvider.batchDistance(query:candidates:metric:)` | delegate + deprecate |
| `BatchOperations.pairwiseDistancesGPU(_:metric:)` | `MetalComputeProvider.distanceMatrix(queries:candidates:metric:)` | delegate + deprecate |
| `AcceleratedDistanceProvider` (whole actor) | `MetalComputeProvider` | type-level `@available(*, deprecated)`; methods delegate |
| `VectorProtocol.acceleratedDistance(to:metric:)` | `MetalComputeProvider.distance(_:_:metric:)` | deprecate (distance only) |
| `IndexableVector.acceleratedDistanceOptimized(...)` | `MetalComputeProvider.distance(...)` | deprecate |

**Not deprecated** (no replacement in provider scope): `AcceleratedVectorOperations` (add / multiply /
scale / normalize / dotProduct vector ops) and `VectorProtocol.acceleratedNormalize*`.

Deprecation message form: `"Deprecated; use MetalComputeProvider.<method>. Removed in 0.6.0."`

---

## VectorCore R4 — transparent dispatch hook

Append **R4** to `docs/VECTORCORE_INTEGRATION_REQUESTS.md`. Proposes a sub-protocol VectorCore's
`Operations.findNearest` / batch-distance path downcasts to (`if let gpu = computeProvider as? BatchKernelProvider { … }`),
so a GPU provider can supply real kernels — the piece `ComputeProvider` structurally lacks. Proposed
shape (to be refined in the doc):

```swift
public protocol BatchKernelProvider: ComputeProvider {
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
```

`MetalComputeProvider` will conform to this (behind the same routing logic) so our half is ready the
moment VectorCore adds the downcast. No VectorCore dependency ships in this PR — R4 is a documented
request, like R1/R2.

---

## Error handling

- `Configuration { preferGPU, fallbackToCPU }`.
- Dimension-mismatch and empty-input guards throw `VectorError`.
- GPU kernel/buffer failure → fall back to CPU when `fallbackToCPU`, else rethrow.
- `metalNotAvailable()` at construction when no GPU.

---

## Testing

New `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`, all `@available`-gated with
`XCTSkip` when no Metal device:

- **Conformance** — `device == .gpu(index: 0)`; `deviceInfo` name/threads populated.
- **Parity** — `batchDistance` and `distance` vs a CPU reference (`AccelerateFallback` /
  VectorCore metric) within eps, euclidean + cosine, at dims {64, 768, 1536}.
- **k-NN** — `findNearest` returns the same indices + distances as a CPU top-K reference.
- **Distance matrix** — symmetric, diagonal handling correct vs CPU reference.
- **Routing** — inject `GPUDecisionEngine` thresholds forcing CPU (assert CPU path result) and forcing
  GPU (assert GPU path runs / skip if no Metal).
- **Fallback metric** — manhattan batch returns correct values via the CPU map.
- **Deprecation integrity** — old `BatchOperations.*GPU` still compile and return identical results to
  the new provider (delegation correctness).

---

## File manifest

**New**
- `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- `Tests/VectorAccelerateTests/MetalComputeProviderTests.swift`

**Modified**
- `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift` — deprecate + delegate the
  scattered statics, `AcceleratedDistanceProvider`, and the distance convenience extensions.
- `docs/VECTORCORE_INTEGRATION_REQUESTS.md` — add R4.
- `CHANGELOG.md` — new "Added/Deprecated" entries (at version time).

**Out of scope (seam only)**
- Zero-copy `makeNoCopyBuffer` wiring (R1/R2); non-euclidean/cosine batch kernels.

---

## Risks / open items

- **Naming (embedded call ②, accepted):** `MetalComputeProvider` (matches VectorCore docs), not
  `GPUComputeProvider`.
- **Shim could mislead** — mitigated by the explicit doc comment + R4 as the real transparent path.
- **`nonisolated` shim properties** must be set from `Sendable` constants captured at init (no actor
  hop when VectorCore reads `provider.device`).
- **Stacked-PR hygiene** — branch carries #31's commits until #31 merges; final PR targets `main`.
