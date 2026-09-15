# Planning Brief — Zero-Copy Lane-Major SoA Scoring for VectorAccelerate

**Prepared for an external planning/reasoning model that does NOT have direct access to the codebase.**
Everything needed to produce the implementation plan is quoted inline. Where you'd normally read a
file, the relevant code/contract is reproduced verbatim below.

**Date:** 2026-06-08 · **Target package:** VectorAccelerate (Swift 6.2, Metal 4, Apple-Silicon-only) ·
**Dependency:** VectorCore 0.3.0 (already pinned).

---

## 0. What we want from you (the desired outcome)

Produce a **detailed, test-driven implementation plan** for adding a *zero-copy, lane-major SoA
scoring path* to VectorAccelerate. The plan is consumed by an engineer (or coding agent) who knows
Swift/Metal but **not this codebase**, so it must be concrete: exact file paths, exact code for each
step, exact test code, and exact shell/test commands with expected output.

**Plan format (follow this):**
- Bite-sized tasks, each ~2–5 minutes, in dependency order.
- Each task lists the **files** it creates/modifies, then numbered **steps**:
  1. write a failing test (show the test code),
  2. run it, show the command + expected failure,
  3. write the minimal implementation (show the code),
  4. run the test, show the command + expected pass,
  5. commit (show the message).
- **No placeholders** ("TODO", "add error handling", "similar to above"): every code step shows the
  actual code. If two tasks need the same code, repeat it.
- Resolve the **open design decisions in §5** explicitly — state the choice and why — before or within
  the tasks that depend on them. We give our analysis and lean for each; you may override with
  reasoning.
- Call out where the **lane-major Metal kernel** indexing must match the frozen SoA layout (§2)
  byte-for-byte, and include a **parity test against the golden fixture** (§2.5).

**Definition of done** is in §8. **Scope boundaries** (what's explicitly out) are in §7.

---

## 1. Context — why this exists

VectorAccelerate (VA) is a GPU (Metal 4) acceleration layer for VectorCore, a dependency-free Swift
vector-math library. Today VA computes batch distances by **staging** the candidate database into a
Metal buffer on **every query** (`withUnsafeBufferPointer` copy of each candidate's floats into a
pooled `MTLBuffer`). For a candidate database queried many times, that per-query copy across the
memory bus is the dominant cost.

VectorCore 0.3.0 ships a **page-aligned Structure-of-Arrays (SoA) candidate layout** specifically so a
GPU layer can map it into Metal with `makeBuffer(bytesNoCopy:)` — true Apple-Silicon unified-memory
**zero-copy** — and **froze that memory layout** as a published contract so a GPU kernel can index it
directly. VA already has the consuming primitive (`makeNoCopyBuffer`, §3) and has **validated the full
bridge end-to-end on real hardware** (§3.1).

The remaining work — **this project** — is to actually consume it: build a page-aligned `SoA` once,
bridge it zero-copy, and score it with **new lane-major Metal kernels** that read the SoA layout
directly (which is *also* the GPU-optimal, memory-coalesced layout — see §5.3). This is the
"ideal path" VA committed to: zero-copy **and** coalesced **and** identical to VectorCore's own CPU
batch layout (so CPU and GPU consume the same bytes).

**Why not the easy path:** VA's current Metal kernels are **row-major** (candidate `c`, dim `d` at
`buffer[c*dim + d]`). The SoA buffer is **lane-major / transposed** (`buffer[lane*count + j]`). So the
SoA buffer is *not* consumable by the existing kernels — new kernels are required. A row-major
page-aligned slab would fit existing kernels but is GPU-coalescing-suboptimal and is not the layout
VectorCore froze for cross-CPU/GPU parity. We deliberately chose the lane-major SoA path.

---

## 2. The frozen SoA layout contract (this is your kernel ABI)

VectorCore published `Docs/SoA_Layout_Contract.md`, **🔒 frozen as of 0.3.0** (any change is a major
version bump). Your Metal kernel's indexing must match it exactly. The essentials:

### 2.1 Layout

```
element(lane ℓ, candidate j)  ==  buffer[ℓ * count + j]
byteOffset(lane ℓ, candidate j)  ==  (ℓ * count + j) * 16
```

- **Element type:** `SIMD4<Float>` — 4 contiguous dimensions, **16 bytes** (`MemoryLayout<SIMD4<Float>>.stride == 16`).
- **Lanes per vector:** `lanes = dimension / 4`. Every supported dimension is divisible by 4 → **no tail/partial lanes**; every lane is a full `SIMD4<Float>`.
- **Lane stride:** consecutive lanes are `count` elements apart → `laneStrideBytes = count * 16`.
- **`count` (= N, candidate count) is NEVER padded.** It is exactly the inter-lane stride. There is no candidate-axis block padding.
- Read the candidate buffer in a shader as `device const float4*`; candidate `j`'s lane `ℓ` is `buffer[ℓ * count + j]`.

So for a fixed lane ℓ, the `count` candidates are **contiguous** in memory (`buffer[ℓ*count + 0 .. ℓ*count + count-1]`). This is the property that makes the lane-major kernel coalesced (§5.3).

### 2.2 The descriptor (use it; do not hardcode)

VectorCore exposes `SoALayout` (a `Sendable, Equatable` struct), reachable as `soa.layoutDescriptor`
(live) or `SoALayout.forType(_:count:pageAligned:)` (no allocation, to precompute shader constants).
Fields:

| Member | Meaning | Formula |
|---|---|---|
| `lanes: Int` | SIMD4 lanes per vector | `dimension / 4` |
| `count: Int` | candidate count **N** (the true stride) | — |
| `elementStrideBytes` (static) | one element | `16` |
| `laneStrideBytes: Int` | bytes between consecutive lanes | `count * 16` |
| `logicalByteCount: Int` | bytes of real data | `lanes * count * 16` |
| `allocatedByteCount: Int` | bytes the allocation occupies | page-rounded (see §2.3) |
| `elementCount: Int` | logical elements | `lanes * count` |
| `elementIndex(lane:candidate:) -> Int` | the frozen index formula | `lane * count + candidate` |

`forType` signature: `static func forType<V: SoACompatible>(_ type: V.Type, count: Int, pageAligned: Bool = false) -> SoALayout`.

### 2.3 The one sharp edge — allocated vs logical bytes

For a **page-aligned** SoA, `makeBuffer(bytesNoCopy:)` requires the *length* to be a page multiple, so
the **whole buffer** is rounded up: `allocatedByteCount == roundUpToPage(logicalByteCount) ≥ logicalByteCount`.
The trailing `allocatedByteCount − logicalByteCount` bytes are **zero-filled slack**.

- Pass **`allocatedByteCount`** as the `length:` to the bridge.
- Bound the kernel by **`lanes` and `count`**; the valid indexed region is exactly `[0, lanes*count)` elements.
- **Never** derive `count` from the byte length — it's page-inflated (e.g. 512-dim, N=5: `logicalByteCount=10240` but `allocatedByteCount=16384` on a 16 KB page; `16384/(128*16)=8 ≠ 5`).

On Apple Silicon the page size is **16384** (16 KB). `getpagesize()` returns 16384. VectorCore's
page-rounding and VA's `makeNoCopyBuffer` agree (validated, §3.1).

### 2.4 Supported types (`SoACompatible`)

SoA needs a **static** dimension/lane count, so it applies to VectorCore's fixed-dimension optimized
vector types — **not** `DynamicVector`:

| Type | `dimension` | `lanes` | SoACompatible |
|---|---|---|---|
| `Vector384Optimized` | 384 | 96 | ✅ |
| `Vector512Optimized` | 512 | 128 | ✅ |
| `Vector768Optimized` | 768 | 192 | ✅ |
| `Vector1536Optimized` | 1536 | 384 | ✅ |
| `DynamicVector` | runtime | — | ❌ (no static `lanes`) |

`DynamicVector` stays on the existing **staged** path (out of scope here — §7). `SoACompatible` (a
VectorCore protocol) requires `static var dimension: Int`, `static var lanes: Int`, and
`var storage: ContiguousArray<SIMD4<Float>>`.

Relevant VectorCore APIs you'll call:
- `SoA<V>.build(from candidates: [V], pageAligned: Bool = false) -> SoA<V>` where `V: SoACompatible`.
- `SoA<V>.init(vectors: [V], pageAligned: Bool = false)`.
- `var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)?` — page-aligned base + page-rounded length, `nil` if not page-aligned.
- `func consumeAllocation() -> (base: UnsafeMutableRawPointer, byteCount: Int)?` — ownership transfer (we are NOT using this; see §2.6).
- `var layoutDescriptor: SoALayout`.
- Constructors for the optimized vectors include `init(_ array: [Float]) throws` and `init(repeating value: Float)`.

### 2.5 Golden parity fixture (regression-locked in VectorCore)

Use this to validate the kernel: build a page-aligned `SoA<Vector512Optimized>` of N=5 candidates.

```
query     q   = [1, 1, …, 1]              (512 dims)
candidate c_k = [1+k, 1+k, …, 1+k]        (512 dims), k = 0 … 4
```

`q − c_k = [−k, …]` over 512 dims ⇒ **Euclidean squared** distance `512·k²`:

| k | candidate value | ‖q − c_k‖² |
|---|---|---|
| 0 | 1.0 | 0 |
| 1 | 2.0 | 512 |
| 2 | 3.0 | 2048 |
| 3 | 4.0 | 4608 |
| 4 | 5.0 | 8192 |

Descriptor for this fixture: `lanes=128, count=5, laneStrideBytes=80, logicalByteCount=10240,
allocatedByteCount=16384 (16 KB page), elementIndex(1,0)=5, elementIndex(127,4)=639`. Buffer
spot-check: for every lane ℓ, `buffer[ℓ*5 + j] == SIMD4<Float>(repeating: Float(1+j))`.

### 2.6 Lifetime / deallocation contract — we use **Borrow mode**

We committed to **borrow mode** (the simpler, no-manual-free model):
- Hold a strong reference to the `SoA` and read `pageAlignedBytes` **without** consuming.
- Bridge with `deallocator: nil`. The `SoA` frees its memory on `deinit`, so **the `SoA` must outlive
  the `MTLBuffer`** — object lifetime is the sole validity guarantee for the pointer.
- Do **not** call `consumeAllocation()` in borrow mode (mixing modes double-frees).

Implication for the design: whatever object owns the bridged `MTLBuffer` must also hold the strong
`SoA` reference for at least as long. (Transfer mode — `consumeAllocation()` + free via
`AlignedMemory.deallocate(base)` ≡ `free(base)` from a Metal deallocator — exists but we are not using
it here.)

---

## 3. The validated bridge primitive (`makeNoCopyBuffer`)

VA already has the consuming half, on `MetalDevice` (a `public actor` whose `rawDevice` is a
nonisolated `any MTLDevice`). Verbatim:

```swift
public actor MetalDevice {
    public nonisolated let rawDevice: any MTLDevice
    // …
    /// Host memory page size — base alignment + length granularity makeNoCopyBuffer requires.
    public static var pageSize: Int { Int(getpagesize()) }                 // 16384 on Apple Silicon

    /// Round `byteLength` up to a whole number of pages.
    public static func pageAlignedLength(_ byteLength: Int) -> Int {
        let p = pageSize
        return (byteLength + p - 1) & ~(p - 1)
    }

    /// Wrap an externally-owned, page-aligned region as an MTLBuffer with NO copy (UMA zero-copy).
    /// Returns nil if base isn't page-aligned or length isn't a page multiple (→ caller stages a copy).
    public nonisolated func makeNoCopyBuffer(
        bytes: UnsafeMutableRawPointer,
        length: Int,
        options: MTLResourceOptions = .storageModeShared,
        deallocator: (@Sendable (UnsafeMutableRawPointer, Int) -> Void)? = nil
    ) -> (any MTLBuffer)? {
        let p = Self.pageSize
        guard length > 0, Int(bitPattern: bytes) % p == 0, length % p == 0 else { return nil }
        return rawDevice.makeBuffer(bytesNoCopy: bytes, length: length, options: options, deallocator: deallocator)
    }
}
```

The bridge call (borrow mode) is therefore:
```swift
guard let (base, byteCount) = soa.pageAlignedBytes,
      let mtlBuffer = metalDevice.makeNoCopyBuffer(
          bytes: UnsafeMutableRawPointer(mutating: base), length: byteCount, deallocator: nil)
else { /* fall back to staged copy */ }
// hold `soa` strongly alongside `mtlBuffer`
```

### 3.1 Smoke-test evidence (already passing on this hardware)

A standalone validator built `SoA<Vector512Optimized>.build(from: candidates, pageAligned: true)`,
bridged it via `makeNoCopyBuffer` (borrow mode), and asserted — **all green on Apple Silicon (16 KB
pages)**:
- descriptor exactly matches §2.5 (`allocatedByteCount == 16384`);
- `mtlBuffer.contents()` **aliases** the SoA allocation (proven true zero-copy, no deallocator);
- §2.5 buffer spot-check `buffer[ℓ*5+j] == SIMD4(1+j)`;
- reading the bridged buffer in `buffer[ℓ*count+j]` layout reproduces the golden squared distances `[0, 512, 2048, 4608, 8192]` — i.e. the exact arithmetic the kernel will do, validated on CPU;
- trailing slack `[logicalByteCount, allocatedByteCount)` is zero-filled.

So the bridge + indexing are proven; the remaining work is the **GPU kernel + the VA API/plumbing**.

---

## 4. Existing VA code the plan builds on (verbatim patterns)

### 4.1 `MetalComputeProvider` — the GPU façade (the integration point)

A `public actor` (`Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`) that already
unifies GPU scoring and conforms to VectorCore's `BatchKernelProvider` (transparent dispatch from
`Operations.findNearest`). Its current batch path **stages** candidates via no-copy kernel providers
(GPU) or `AccelerateFallback` (CPU), routed by `GPUDecisionEngine`. Shape (abridged):

```swift
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor MetalComputeProvider: BatchKernelProvider {
    public struct Configuration: Sendable { public var preferGPU = true; public var fallbackToCPU = true }
    private let context: Metal4Context
    private let engine: Metal4ComputeEngine
    private let decisionEngine: GPUDecisionEngine
    private let l2Provider: L2KernelDistanceProvider          // staged GPU euclid
    private let cosineProvider: CosineKernelDistanceProvider  // staged GPU cosine
    private let configuration: Configuration

    public init(context: Metal4Context, configuration: Configuration = .init(),
                decisionEngine: GPUDecisionEngine? = nil) async throws { … }

    // Generic over any VectorProtocol whose Scalar == Float. Per-metric semantics:
    // euclidean → L2 distance (sqrt); cosine → 1−similarity; dotProduct → raw dot; manhattan → L1; chebyshev → L∞.
    public func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: SupportedDistanceMetric)
        async throws -> [Float] where V.Scalar == Float { … }   // stages per call today

    public func findNearest<V: VectorProtocol>(query: V, in candidates: [V], k: Int, metric: SupportedDistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float { … }
    // … distanceMatrix, single distance, + BatchKernelProvider `any DistanceMetric` overloads …
}
```

The new zero-copy path should integrate with this façade (see §5.1/§5.5).

### 4.2 The Metal4 kernel-wrapper Swift pattern (mirror this exactly)

`Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift` — the template for a new
`SoAL2DistanceKernel`:

```swift
public final class L2DistanceKernel: @unchecked Sendable, Metal4Kernel {
    public let name = "L2DistanceKernel"
    private let pipelineState: any MTLComputePipelineState
    public let context: Metal4Context

    public init(context: Metal4Context) async throws {
        self.context = context
        let device = context.device.rawDevice
        let library = try await context.shaderCompiler.getDefaultLibrary()
        guard let function = library.makeFunction(name: "l2_distance") else {
            throw VectorError.shaderNotFound(name: "VectorAccelerate: 'l2_distance' kernel not found.")
        }
        self.pipelineState = try await device.makeComputePipelineState(function: function)
    }

    public func encode(into encoder: any MTLComputeCommandEncoder, commandBuffer: any MTLCommandBuffer,
                       queriesToken: BufferToken, targetsToken: BufferToken, distancesToken: BufferToken,
                       numQueries: Int, dimension: Int, computeSqrt: Bool = true) {
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(queriesToken.buffer, offset: 0, index: 0)
        encoder.setBuffer(targetsToken.buffer, offset: 0, index: 1)
        encoder.setBuffer(distancesToken.buffer, offset: 0, index: 2)
        var dim = UInt32(dimension); encoder.setBytes(&dim, length: 4, index: 3)
        var sqrtFlag = UInt32(computeSqrt ? 1 : 0); encoder.setBytes(&sqrtFlag, length: 4, index: 4)
        let w = pipelineState.threadExecutionWidth
        let threadsPerGroup = (pipelineState.maxTotalThreadsPerThreadgroup / w) * w
        encoder.dispatchThreadgroups(MTLSizeMake(numQueries, 1, 1),
                                     threadsPerThreadgroup: MTLSizeMake(threadsPerGroup, 1, 1))
        queriesToken.keepAlive(until: commandBuffer)      // pin buffers for GPU lifetime
        targetsToken.keepAlive(until: commandBuffer)
        distancesToken.keepAlive(until: commandBuffer)
    }
}
```

Notes: pipelines come from `context.shaderCompiler.getDefaultLibrary()` → `library.makeFunction(name:)`
→ `device.makeComputePipelineState(function:)`. Buffers are bound at indices 0…, scalars via
`setBytes` at higher indices. `BufferToken` is VA's pooled-buffer handle; `keepAlive(until:)` pins it.
A bridged no-copy `MTLBuffer` is a raw `any MTLBuffer` (not a `BufferToken`) — the new kernel's encode
should accept raw `any MTLBuffer` for the candidate buffer (and keep the owning object alive itself).

### 4.3 The existing row-major batch kernel (the pattern to INVERT)

`Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal` (`l2_distance` / `l2_distance_kernel`),
abridged — note the **row-major** indexing you must replace with lane-major:

```metal
kernel void l2_distance_kernel(
    device const float* queryVectors    [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],   // ROW-major: candidate c at [c*stride ..]
    device float*       distances       [[buffer(2)]],
    constant L2DistanceParams& params   [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]])
{
    // …bounds check…
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);  // ROW-major
    float4 sum4 = float4(0);
    for (…) { float4 d = q4 - db4; sum4 = fma(d, d, sum4); }     // accumulate squared diffs
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    distances[…] = params.computeSqrt ? sqrt(sum) : sum;
}
```

The SoA kernel keeps the squared-diff accumulation but reads candidates **lane-major**
(`buffer[ℓ*count + j]`, `float4` per lane) instead of row-major. See §5.3 for the coalescing rationale.

### 4.4 Persistent GPU buffer pattern (the index already does build-once-reuse)

`AcceleratedVectorIndex` owns a **lifetime-persistent** dataset buffer via `GPUVectorStorage`
(`Sources/VectorAccelerate/Index/Internal/GPUVectorStorage.swift`), reused across all queries:

```swift
final class GPUVectorStorage: @unchecked Sendable {
    private(set) var buffer: (any MTLBuffer)?      // persistent, reused
    let dimension: Int
    private(set) var capacity: Int
    private(set) var allocatedSlots: Int = 0
    init(device: any MTLDevice, dimension: Int, capacity: Int) throws { … allocateBuffer() }
    private func allocateBuffer() throws {
        guard let b = device.makeBuffer(length: capacity*bytesPerSlot, options: .storageModeShared) else { throw … }
        buffer = b
    }
    // writeVector(_, at:) on insert; grows 2× on capacity exhaustion (ROW-major slots).
}
```
Search reads `storage.buffer` directly each query (no per-query restage). **But** this storage is
**mutable/incremental** (insert/delete, 2× growth, row-major slots) — it does **not** match SoA's
**immutable bulk-build** model (`SoA.build(from: wholeSet)`). That mismatch drives the §5.1 decision.

### 4.5 `Metal4Context` — buffers, pipelines, execution

```swift
public actor Metal4Context {
    public var device: MetalDevice { get }                       // nonisolated access in practice; .rawDevice is nonisolated
    public func getBuffer(size: Int) async throws -> BufferToken
    public func getBuffer<T: Sendable>(for data: [T]) async throws -> BufferToken
    public func getPipeline(functionName: String) async throws -> any MTLComputePipelineState
    public func executeAndWait(_ op: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> Void) async throws
    public func execute<T: Sendable>(_ op: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T) async throws -> T
    var shaderCompiler: Metal4ShaderCompiler { get }             // .getDefaultLibrary()
}
```
`BufferToken` wraps a pooled `MTLBuffer` (`.buffer`), has `copyData(as:count:)`, `readScalar(as:)`,
`keepAlive(until:)`.

### 4.6 Shader files & registration

`.metal` files live in `Sources/VectorAccelerate/Metal/Shaders/` and are compiled into a default
library; functions are discovered by name (`library.makeFunction(name: "…")`) — **no explicit
registration**. A new kernel = a new `.metal` file there (e.g. `SoADistance.metal` with
`kernel void soa_l2_distance(…)`) + a Swift wrapper in `Kernels/Metal4/`. The package processes the
shaders directory as a resource (`.process("Metal/Shaders")` in `Package.swift`) and uses the
`MetalCompilerPlugin`.

### 4.7 Supporting types

- `SupportedDistanceMetric` (VectorCore enum): `.euclidean, .cosine, .dotProduct, .manhattan, .chebyshev`.
- `Metal4DistanceMetric` (VA enum, same cases) used by `Metal4ComputeEngine.fusedDistanceTopK(query:database:k:metric:)`.
- `GPUDecisionEngine` (actor): `shouldUseGPU(operation: GPUOperation, vectorCount:, candidateCount:, k:, queryCount:, dimension:) async -> Bool`, adaptive thresholds. `GPUOperation` cases include `.l2Distance, .cosineSimilarity, .topKSelection, .distanceMatrix`.
- `AccelerateFallback` (CPU, vDSP): `batchEuclideanDistance(query:candidates:) -> [Float]`, `batchCosineSimilarity(query:candidates:) -> [Float]` (returns *similarity*; distance = `1 − sim`), single `euclideanDistance/cosineSimilarity(_:_:) throws -> Float`.
- `VectorError`: `.invalidInput(String)`, `.dimensionMismatch(expected:actual:)`, `.metalNotAvailable()`, `.shaderNotFound(name:)`.

---

## 5. Open design decisions to resolve (with trade-offs and our lean)

The cross-package design is settled (frozen layout, borrow mode, golden fixture, validated bridge).
These VA-internal decisions are **what the plan must resolve.** Our lean is given; override with reasoning.

### 5.1 Where the bridged SoA buffer lives — **new "prepared candidate set" type (lean)**

The zero-copy win requires **build-once, query-many** (build `SoA(pageAligned:true)` + bridge once,
reuse the `MTLBuffer` across queries). Options:
- **(A, lean) A new immutable value/reference type** — e.g. `SoACandidateSet<V: SoACompatible>` (an
  actor or final class) that owns the `SoA` (held strongly for borrow mode) and the bridged
  `MTLBuffer` + the `SoALayout`, built once from `[V]`. This is the "prepared candidate set"
  abstraction VA currently **lacks** (§4.4 confirms none exists). It cleanly matches SoA's bulk-build
  immutability and the borrow-mode lifetime (one object holds both `SoA` and `MTLBuffer`). The
  provider scores a query against it.
- **(B) Reuse the index's `GPUVectorStorage`.** Rejected as primary: that storage is mutable/incremental/
  row-major (§4.4) and doesn't match SoA's immutable bulk build; retrofitting it is invasive and
  fights the model.

Decide: type name, value vs reference (lean: `final class`/actor for the held buffer + Sendable),
whether it falls back to a staged buffer when `pageAlignedBytes` is `nil`, and how it exposes the
`MTLBuffer` + descriptor to the kernel.

### 5.2 Entry-point shape (`SoACompatible` constraint vs the generic provider)

The provider's existing `batchDistance<V: VectorProtocol>` can't be constrained to `SoACompatible`
(which needs static dims). Options:
- **(A, lean)** The new prepared-set type is generic over `V: SoACompatible` (so it's only constructible
  for `Vector512/768/1536Optimized`), and the provider gains methods that take a prepared set:
  `func batchDistance(query: V, against set: SoACandidateSet<V>, metric:) async throws -> [Float]` and a
  matching `findNearest(...)`. The existing generic `[V]` methods are unchanged (still stage; serve
  `DynamicVector` and ad-hoc calls).
- **(B)** Add a `where V: SoACompatible` overload that builds+bridges a SoA *per call* (no cross-call
  reuse). Marginal vs staging (still one bulk copy per call, but coalesced kernel + one contiguous
  transfer). Possibly worth offering as a convenience, but the prepared-set is the real win.

Decide the public surface. Lean: prepared-set is primary (A); a per-call convenience (B) is optional/
follow-up.

### 5.3 Lane-major kernel design (the core) — coalesced one-thread-per-candidate (lean)

Because for a fixed lane ℓ the candidates are contiguous (`buffer[ℓ*count + 0..count-1]`), the
coalesced mapping is **one GPU thread per candidate `j`**, looping lanes `ℓ = 0..lanes-1`:
```
thread j:
  acc = 0
  for ℓ in 0..<lanes:
     float4 c = candidateBuffer[ℓ * count + j]   // adjacent threads j,j+1 → adjacent addresses ⇒ COALESCED
     float4 d = queryLane[ℓ] - c                 // queryLane[ℓ] broadcast to all threads
     acc = fma(d, d, acc)                        // (euclidean)
  distances[j] = computeSqrt ? sqrt(horizontal_sum(acc)) : horizontal_sum(acc)
```
Grid = `count` threads (1-D), bound by `count` from the descriptor. The **query** is `lanes` `float4`s
(the dim floats) bound as a small separate buffer; each thread reads the same `queryLane[ℓ]`
(broadcast — ideally via constant/threadgroup memory, but a plain `device const float4*` read is fine
for a first cut). Pass `lanes` and `count` as scalar params (`setBytes`). **Cosine** variant
additionally accumulates `dot`, `‖q‖²` (can be precomputed on host), `‖c‖²` per thread, then
`distance = 1 − dot/(sqrt(qNorm²)·sqrt(cNorm²))` with the BE3 `leastNormalMagnitude` floor (match VA's
existing cosine parity).

Decide: threadgroup size; whether query lanes go in constant/threadgroup memory; whether to precompute
query norm on host; whether to also write a cosine kernel now or L2-first. Lean: L2 first
(golden-fixture-validated), cosine second, both with the one-thread-per-candidate mapping above.

### 5.4 Top-K integration

The lane-major kernel produces a `distances` array (one per candidate). For `findNearest`:
- **(A, lean)** distance kernel → existing top-K (CPU heap for small K, or VA's existing GPU top-K on
  the distances buffer). Simple, reuses validated top-K.
- **(B)** a fused lane-major distance+top-K kernel (like the existing `FusedL2TopKKernel`). Faster,
  more kernel work. Defer as an optimization.

Lean: (A) for this project; fused is a follow-up.

### 5.5 How it plugs into `findNearest` / `BatchKernelProvider`

The transparent dispatch (`BatchKernelProvider.findNearest`, called by VectorCore `Operations`) gets
`[V]` per call (no prepared set). Options:
- The prepared-set path is a **new explicit API** on the provider (caller opts in by building a
  `SoACandidateSet`), separate from the transparent `[V]` dispatch. (Lean — keeps the zero-copy win for
  the reuse case without changing transparent semantics.)
- Optionally, `BatchKernelProvider.findNearest` could detect `V: SoACompatible` and use the §5.2(B)
  per-call SoA build. Decide whether that's in scope.

Lean: ship the explicit prepared-set API; leave transparent-dispatch auto-detection as a follow-up.

### 5.6 Borrow-mode lifetime wrapper

The owning object (§5.1's type) must hold the strong `SoA` ref for ≥ the `MTLBuffer`'s life. Decide how
this is enforced structurally (e.g. both stored on the same object; `withExtendedLifetime` not needed if
both are stored properties released together — but document that the `MTLBuffer` must not outlive the
object). Include a test that proves `mtlBuffer.contents()` aliases the SoA bytes (zero-copy) and that
results are correct.

---

## 6. Constraints & conventions (must follow)

- **Swift 6.2, strict concurrency.** Actors for GPU-owning types. Top-level/global mutable state is
  MainActor-isolated; helper funcs that mutate it need `@MainActor`. `@Sendable` closures for Metal
  execute blocks.
- **`ExistentialAny` upcoming feature is ON** — spell every existential `any` (`any MTLBuffer`,
  `any DistanceMetric`, `any MTLComputePipelineState`).
- **Availability:** everything Metal-touching is `@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)`.
- **No `-warnings-as-errors`** (deprecation warnings are fine).
- **Tests:** XCTest; gate with `guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip(...) }`
  in `setUp`. The dev machine is Apple Silicon with Metal, so tests run. **Do not** put `await` inside
  `XCTAssert*` autoclosures (Swift rejects it) — hoist to a `let` first.
- **Parity tolerances:** GPU fp32 vs CPU reference — use relative tolerance (`accuracy: max(1e-2, abs(ref)*1e-3)`); the golden fixture (§2.5) values are exact integers and can use tight tolerance.
- **Commits:** small, per-task; the repo's rule is "commit only when the user asks," so the plan's
  commit steps are the intended rhythm but execution should confirm cadence. Commit messages end with a
  trailer line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Build is slow** (large Metal package); each `swift build`/`swift test` takes a while — that's normal.
  Use `swift test --filter <ClassName>/<testName>` for single tests.
- **Files focused/small**, one responsibility each; follow existing folder conventions
  (`Kernels/Metal4/`, `Metal/Shaders/`, `Integration/`).

---

## 7. Scope

**In scope (this plan):**
- A `.metal` lane-major **L2** distance kernel (`soa_l2_distance`) indexing `buffer[ℓ*count + j]`, plus
  its Swift wrapper (mirroring §4.2), validated against the golden fixture (§2.5) **on GPU**.
- A lane-major **cosine** kernel (or a clear note if deferred to a fast-follow within the same plan).
- The **prepared candidate set** type (§5.1) that builds `SoA(pageAligned:true)`, bridges via
  `makeNoCopyBuffer` (borrow mode), holds both, and exposes buffer + descriptor; with a **staged-buffer
  fallback** when `pageAlignedBytes` is `nil` (e.g. unsupported alignment).
- Provider methods to score a query against a prepared set (batch distance + findNearest), §5.2/§5.5.
- Tests: golden-fixture GPU parity, zero-copy aliasing proof, CPU-reference parity at dims {512, 768,
  1536}, fallback path, lifetime/borrow correctness.
- **Doc hygiene:** mark R1–R3 **resolved (delivered in VectorCore 0.3.0)** in
  `docs/VECTORCORE_INTEGRATION_REQUESTS.md` (R4 is already marked delivered+adopted).

**Out of scope (explicit follow-ups, do NOT plan here):**
- FP16 `SoAFP16` bridge (different layout: groups-of-4, dimension-first; not part of the frozen FP32
  contract).
- True batched `findNearestBatch` single-GPU-dispatch over many queries (current code has a default
  that loops per query; the fast-follow is one dispatch for the whole query set).
- Fused lane-major distance+top-K kernel (§5.4 B).
- `DynamicVector` (stays on the staged path — no static lanes).
- Auto-detecting `SoACompatible` inside transparent `BatchKernelProvider` dispatch (§5.5).
- Pointer Top-K hybrid (feeding GPU distance buffers into VectorCore's pointer `select`).

---

## 8. Definition of done

- A page-aligned `SoA<VNNNOptimized>` is built once, bridged zero-copy (proven: `MTLBuffer.contents()`
  aliases the SoA bytes), and **reused across queries** with no per-query candidate restage.
- The lane-major **GPU** kernel reproduces the golden fixture (§2.5) squared/Euclidean distances within
  tolerance, and matches a CPU reference at dims {512, 768, 1536} for L2 (and cosine if included).
- The prepared-set API is usable from `MetalComputeProvider`; the staged fallback works when alignment
  isn't available.
- Borrow-mode lifetime is correct (no use-after-free; SoA outlives the buffer).
- Full suite green (currently 1497 tests, 0 failures); new tests added for all the above.
- R1–R3 marked resolved in the integration-requests doc.

---

## 9. Things to get exactly right (common-mistake guardrails)

- **Length passed to the bridge is `allocatedByteCount`** (page-rounded), but the **kernel is bounded by
  `count`** (logical). Never infer `count` from byte length (§2.3).
- **Borrow mode**: hold the `SoA` strongly; bridge with `deallocator: nil`; never `consumeAllocation()`
  in this path (§2.6).
- **Lane-major indexing** in the shader is `buffer[ℓ*count + j]` as `float4` — *not* row-major. Pull
  `lanes`/`count` from the descriptor, pass as scalars; don't hardcode per-dimension.
- **Coalescing**: thread-per-candidate, loop lanes — adjacent threads read adjacent addresses (§5.3).
- **Cosine** must match VA's existing parity (BE3 `leastNormalMagnitude` floor; `1 − similarity`).
- **`@available` + `XCTSkip`** on every Metal test; **`any`** on every existential; **no `await` in
  `XCTAssert*`**.
- The query buffer is small (`lanes` `float4`s); stage it per query (that copy is negligible — the win
  is not restaging the *candidates*).

---

*End of brief. Produce the implementation plan per §0 and §9, resolving §5 explicitly, scoped by §7,
meeting §8.*
