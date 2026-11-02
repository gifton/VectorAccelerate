# Zero-Copy Lane-Major SoA Scoring — Implementation Plan (corrected)

> **Provenance:** drafted by an external model from `docs/zero-copy-soa-planning-brief.md`, then
> **corrected against the actual codebase** (5 blocking API errors fixed, cosine floor aligned to the
> BE3 convention, kernel caching added, missing parity/reuse tests added, doc task pointed at the real
> R1–R3). Every API below is verified against source. Execute with `superpowers:executing-plans`
> (inline, TDD, commit per task).

**Goal:** Add a zero-copy, lane-major SoA scoring path: build `SoA(pageAligned: true)` once from a
fixed-dim candidate set, bridge it via `makeNoCopyBuffer` (borrow mode), and score it with new
lane-major Metal kernels that index `buffer[ℓ*count + j]` per VectorCore's frozen SoA Layout Contract
(0.3.0). Reuse the bridged buffer across queries.

**Architecture:** A new immutable `SoACandidateSet<V: SoACompatible>` owns the `SoA` (strong, borrow
mode) + bridged `MTLBuffer` + `SoALayout`. New `SoADistanceKernel` (L2 + cosine) reads the SoA layout
coalesced (one thread per candidate, looping lanes). `MetalComputeProvider` gains
`batchDistance(query:against:metric:)` / `findNearest(query:in:k:metric:)` taking a `SoACandidateSet`,
caching one `SoADistanceKernel`; top-K reuses the provider's existing `selectTopK`.

**Tech Stack:** Swift 6.2 actors, Metal 4 (`@preconcurrency import Metal`), VectorCore 0.3.0
(`SoA`, `SoALayout`, `SoACompatible`, `Vector512/768/1536Optimized`), Accelerate (CPU ref), XCTest.

**Contract & context:** see `docs/zero-copy-soa-planning-brief.md` (frozen layout §2, golden fixture
§2.5, borrow mode §2.6, `makeNoCopyBuffer` §3, patterns §4). This plan does not re-quote it.

**Branch:** `gifton/metal-compute-provider` (current). Resolves brief §7 in-scope items; out-of-scope
items stay follow-ups.

---

## Resolved design decisions (brief §5)

- **§5.1** New `SoACandidateSet<V: SoACompatible>` `final class` (`@unchecked Sendable`), borrow mode,
  with a staged-copy fallback when `pageAlignedBytes == nil`.
- **§5.2/§5.5** Explicit API: new `MetalComputeProvider` methods take `SoACandidateSet<V>`. The generic
  `[V]` staging paths and the transparent `BatchKernelProvider` dispatch are untouched.
- **§5.3** Coalesced one-thread-per-candidate, loop-lanes. Both L2 and cosine kernels.
- **§5.4** Distance kernel → CPU top-K via the provider's existing `selectTopK`.
- **§5.6** Borrow lifetime: `SoACandidateSet` holds `SoA` strongly; the provider pins `set` through GPU
  completion (prevents ARC freeing the SoA mid-flight → use-after-free).

## Verified API notes (the corrections — do not regress)

- `SoACompatible` is **`Sendable` only** (no `Scalar`/`toArray`). The optimized vectors are
  `VectorProtocol` separately ⇒ provider methods constrain **`V: SoACompatible & VectorProtocol where V.Scalar == Float`**.
- `MetalComputeProvider.context` is **`private`** ⇒ widen to `internal` so a same-module extension can
  use it; the new `soaKernel` is also `internal`. `selectTopK` is already internal-reachable.
- SoA raw bytes accessor is **`withUnsafeRawBuffer`** (not `withUnsafeBytes`).
- Use **`VectorError.invalidInput(_:)`** for unsupported metric (no `unsupportedGPUOperation` case);
  `VectorError.bufferAllocationFailed(size:)` **does** exist.
- Top-K: **`Self.selectTopK(distances, k:, largerIsCloser: false)`** (no `TopKSelection.select`).
- `MetalDevice` init is **`throws`, not `async`** — `try MetalDevice()` or reuse `context.device`
  (a `MetalDevice`); no `await`. `Vector512Optimized(repeating:)` is **non-throwing** — no `try`.
- Cosine floor: **`FLT_MIN`** + `sqrt(a)*sqrt(b)` + NaN-preserving `clamp` (BE3 parity), not `1e-8`.
- `Metal4Common.h` exists; `uint32_t`/`float4`/`fma`/`FLT_MIN` are already used in shipping shaders.

---

## Task 1: `SoACandidateSet` (borrow-mode wrapper)

**Files:** Create `Sources/VectorAccelerate/Integration/SoACandidateSet.swift`,
`Tests/VectorAccelerateTests/SoACandidateSetTests.swift`.

- [ ] **Step 1 — failing test**

```swift
// Tests/VectorAccelerateTests/SoACandidateSetTests.swift
import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoACandidateSetTests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testZeroCopyBridging() async throws {
        let device = try MetalDevice()                              // throws, not async
        let candidates = (0..<5).map { Vector512Optimized(repeating: Float(1 + $0)) }  // non-throwing
        let set = try SoACandidateSet(candidates: candidates, device: device)

        XCTAssertEqual(set.layout.count, 5)
        XCTAssertEqual(set.layout.lanes, 128)
        XCTAssertEqual(set.layout.allocatedByteCount, 16384)        // 16 KB page (Apple Silicon)
        XCTAssertTrue(set.isZeroCopy)
        // Borrow mode: the MTLBuffer aliases the SoA allocation (no copy).
        let (base, _) = try XCTUnwrap(set.soa.pageAlignedBytes)
        XCTAssertEqual(set.buffer.contents(), UnsafeMutableRawPointer(mutating: base))
    }
}
```

- [ ] **Step 2 — run, expect fail**
Run: `swift test --filter SoACandidateSetTests/testZeroCopyBridging`
Expected: `cannot find 'SoACandidateSet' in scope`.

- [ ] **Step 3 — implement**

```swift
// Sources/VectorAccelerate/Integration/SoACandidateSet.swift
import Foundation
@preconcurrency import Metal
import VectorCore

/// An immutable, GPU-ready candidate set in VectorCore's page-aligned SoA layout, bridged into a
/// zero-copy `MTLBuffer` (borrow mode). Build once, score many times.
///
/// Borrow mode (SoA Layout Contract §5): the `SoA` is held strongly and frees its allocation on
/// `deinit`, so it must outlive `buffer`. Holding both on this one object guarantees that.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class SoACandidateSet<V: SoACompatible>: @unchecked Sendable {
    /// Owns the SoA allocation; pinned for the buffer's lifetime (borrow mode).
    public let soa: SoA<V>
    /// The candidate buffer the kernel reads (zero-copy alias, or a staged copy — see `isZeroCopy`).
    public let buffer: any MTLBuffer
    /// Frozen layout descriptor (lanes, count, strides) — the kernel's source of truth.
    public let layout: SoALayout
    /// True when `buffer` is a zero-copy alias of the SoA allocation; false when a staged copy was made.
    public let isZeroCopy: Bool

    public init(candidates: [V], device: MetalDevice) throws {
        let built = SoA<V>.build(from: candidates, pageAligned: true)
        self.soa = built
        self.layout = built.layoutDescriptor

        if let (base, byteCount) = built.pageAlignedBytes,
           let noCopy = device.makeNoCopyBuffer(
               bytes: UnsafeMutableRawPointer(mutating: base),
               length: byteCount,                       // page-rounded allocatedByteCount
               options: .storageModeShared,
               deallocator: nil) {                      // borrow mode: SoA frees on deinit
            self.buffer = noCopy
            self.isZeroCopy = true
        } else {
            // Staged fallback (e.g. not page-aligned): copy the logical bytes into a fresh buffer.
            let logical = built.layoutDescriptor.logicalByteCount
            guard let staged = device.rawDevice.makeBuffer(length: logical, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: logical)
            }
            built.withUnsafeRawBuffer { raw in
                if let src = raw.baseAddress { memcpy(staged.contents(), src, logical) }
            }
            self.buffer = staged
            self.isZeroCopy = false
        }
        self.buffer.label = "SoACandidateSet<\(V.self)>"
    }
}
```

- [ ] **Step 4 — run, expect pass**
Run: `swift test --filter SoACandidateSetTests/testZeroCopyBridging` → `Executed 1 test, with 0 failures`.

- [ ] **Step 5 — commit**
```bash
git add Sources/VectorAccelerate/Integration/SoACandidateSet.swift Tests/VectorAccelerateTests/SoACandidateSetTests.swift
git commit -m "Add SoACandidateSet: zero-copy borrow-mode bridge for SoA candidates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Lane-major Metal kernels (L2 + cosine) + `SoADistanceKernel`

**Files:** Create `Sources/VectorAccelerate/Metal/Shaders/SoADistance.metal`,
`Sources/VectorAccelerate/Kernels/Metal4/SoADistanceKernel.swift`,
`Tests/VectorAccelerateTests/SoAKernelGoldenTests.swift`.

- [ ] **Step 1 — failing tests** (L2 golden squared + cosine parity, direct kernel)

```swift
// Tests/VectorAccelerateTests/SoAKernelGoldenTests.swift
import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoAKernelGoldenTests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testL2SquaredGoldenFixture() async throws {
        let context = try await Metal4Context()
        let kernel = try await SoADistanceKernel(context: context)
        let query = Vector512Optimized(repeating: 1.0)
        let candidates = (0..<5).map { Vector512Optimized(repeating: Float(1 + $0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)

        let qToken = try await context.getBuffer(for: query.toArray())
        let outToken = try await context.getBuffer(size: set.layout.count * MemoryLayout<Float>.stride)
        try await context.executeAndWait { cb, enc in
            kernel.encode(into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                          distancesBuffer: outToken.buffer, count: set.layout.count,
                          lanes: set.layout.lanes, metric: .euclidean, computeSqrt: false)
            qToken.keepAlive(until: cb); outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }                // pin SoA through GPU completion
        }
        let got = outToken.copyData(as: Float.self, count: 5)
        let golden: [Float] = [0, 512, 2048, 4608, 8192]
        for (a, b) in zip(got, golden) { XCTAssertEqual(a, b, accuracy: 1e-3) }
    }

    func testCosineKernelMatchesCPU() async throws {
        let context = try await Metal4Context()
        let kernel = try await SoADistanceKernel(context: context)
        // Distinct, non-degenerate vectors so denom >> FLT_MIN.
        let q = Vector512Optimized((0..<512).map { Float(($0 % 7) - 3) + 0.5 })
        let cs = (0..<6).map { k in Vector512Optimized((0..<512).map { Float(($0 % 5) - 2) + Float(k) * 0.25 }) }
        let set = try SoACandidateSet(candidates: cs, device: context.device)

        let qa = q.toArray()
        let qNormSq = qa.reduce(Float(0)) { $0 + $1 * $1 }
        let qToken = try await context.getBuffer(for: qa)
        let outToken = try await context.getBuffer(size: cs.count * MemoryLayout<Float>.stride)
        try await context.executeAndWait { cb, enc in
            kernel.encode(into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                          distancesBuffer: outToken.buffer, count: set.layout.count,
                          lanes: set.layout.lanes, metric: .cosine, queryNormSq: qNormSq)
            qToken.keepAlive(until: cb); outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }
        }
        let got = outToken.copyData(as: Float.self, count: cs.count)
        for (i, c) in cs.enumerated() {
            let ref = SoAKernelGoldenTests.refCosineDistance(qa, c.toArray())
            XCTAssertEqual(got[i], ref, accuracy: max(1e-3, abs(ref) * 1e-3))
        }
    }

    // CPU reference matching the kernel exactly: sqrt(a)*sqrt(b), FLT_MIN floor, 1 − clamp(sim).
    static func refCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        let denom = na.squareRoot() * nb.squareRoot()
        guard denom >= .leastNormalMagnitude else { return 1.0 }
        return 1.0 - min(max(dot / denom, -1), 1)
    }
}
```

- [ ] **Step 2 — run, expect fail**
Run: `swift test --filter SoAKernelGoldenTests` → `cannot find 'SoADistanceKernel' in scope`.

- [ ] **Step 3 — implement the shader**

```cpp
// Sources/VectorAccelerate/Metal/Shaders/SoADistance.metal
#include "Metal4Common.h"

// Lane-major SoA layout (VectorCore 0.3.0, frozen): element(lane ℓ, candidate j) == buffer[ℓ*count + j],
// each a float4 (4 dims). One thread per candidate; loop lanes. Adjacent threads (j, j+1) read adjacent
// float4s within a lane block ⇒ coalesced.

struct SoAL2Params { uint count; uint lanes; uint computeSqrt; uint _pad; };
struct SoACosineParams { uint count; uint lanes; float queryNormSq; uint _pad; };

kernel void soa_l2_distance(
    device const float4* query      [[buffer(0)]],   // `lanes` elements
    device const float4* candidates [[buffer(1)]],   // lanes*count, lane-major
    device float*        distances  [[buffer(2)]],
    constant SoAL2Params& p         [[buffer(3)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= p.count) return;
    float4 acc = float4(0.0f);
    for (uint l = 0; l < p.lanes; ++l) {
        float4 d = query[l] - candidates[l * p.count + j];
        acc = fma(d, d, acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    distances[j] = p.computeSqrt ? sqrt(sum) : sum;
}

kernel void soa_cosine_distance(
    device const float4* query      [[buffer(0)]],
    device const float4* candidates [[buffer(1)]],
    device float*        distances  [[buffer(2)]],
    constant SoACosineParams& p     [[buffer(3)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= p.count) return;
    float4 dotAcc = float4(0.0f);
    float4 cNormAcc = float4(0.0f);
    for (uint l = 0; l < p.lanes; ++l) {
        float4 q = query[l];
        float4 c = candidates[l * p.count + j];
        dotAcc = fma(q, c, dotAcc);
        cNormAcc = fma(c, c, cNormAcc);
    }
    float dot = dotAcc.x + dotAcc.y + dotAcc.z + dotAcc.w;
    float cNormSq = cNormAcc.x + cNormAcc.y + cNormAcc.z + cNormAcc.w;
    // BE3 parity: sqrt(a)*sqrt(b) (overflow-safe), FLT_MIN floor, NaN-preserving clamp.
    float denom = sqrt(p.queryNormSq) * sqrt(cNormSq);
    float raw = (denom < FLT_MIN) ? 0.0f : (dot / denom);
    float sim = isnan(raw) ? raw : clamp(raw, -1.0f, 1.0f);
    distances[j] = 1.0f - sim;
}
```

- [ ] **Step 3b — implement the Swift wrapper**

```swift
// Sources/VectorAccelerate/Kernels/Metal4/SoADistanceKernel.swift
import Foundation
@preconcurrency import Metal
import VectorCore

/// Param structs mirroring the MSL layouts (16 bytes each).
public struct SoAL2Params: Sendable {
    public var count: UInt32; public var lanes: UInt32; public var computeSqrt: UInt32; public var _pad: UInt32 = 0
    public init(count: Int, lanes: Int, computeSqrt: Bool) {
        self.count = UInt32(count); self.lanes = UInt32(lanes); self.computeSqrt = computeSqrt ? 1 : 0
    }
}
public struct SoACosineParams: Sendable {
    public var count: UInt32; public var lanes: UInt32; public var queryNormSq: Float; public var _pad: UInt32 = 0
    public init(count: Int, lanes: Int, queryNormSq: Float) {
        self.count = UInt32(count); self.lanes = UInt32(lanes); self.queryNormSq = queryNormSq
    }
}

/// Lane-major SoA distance kernel (L2 + cosine). Reads a candidate buffer in VectorCore's frozen SoA
/// layout (`buffer[ℓ*count + j]`), one thread per candidate.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class SoADistanceKernel: @unchecked Sendable, Metal4Kernel {
    public let name = "SoADistanceKernel"
    public let context: Metal4Context
    private let l2Pipeline: any MTLComputePipelineState
    private let cosinePipeline: any MTLComputePipelineState

    public init(context: Metal4Context) async throws {
        self.context = context
        let library = try await context.shaderCompiler.getDefaultLibrary()
        guard let l2 = library.makeFunction(name: "soa_l2_distance"),
              let cos = library.makeFunction(name: "soa_cosine_distance") else {
            throw VectorError.shaderNotFound(name: "VectorAccelerate: soa_l2_distance/soa_cosine_distance not found")
        }
        let dev = context.device.rawDevice
        self.l2Pipeline = try await dev.makeComputePipelineState(function: l2)
        self.cosinePipeline = try await dev.makeComputePipelineState(function: cos)
    }

    public func warmUp() async throws {}

    /// `count`/`lanes` come from the candidate set's `SoALayout`. `queryNormSq` only used for cosine.
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queryBuffer: any MTLBuffer, candidateBuffer: any MTLBuffer, distancesBuffer: any MTLBuffer,
        count: Int, lanes: Int, metric: SupportedDistanceMetric,
        computeSqrt: Bool = true, queryNormSq: Float = 0)
    {
        let pipeline = (metric == .cosine) ? cosinePipeline : l2Pipeline
        encoder.setComputePipelineState(pipeline)
        encoder.label = "SoADistance(\(metric))"
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidateBuffer, offset: 0, index: 1)
        encoder.setBuffer(distancesBuffer, offset: 0, index: 2)
        if metric == .cosine {
            var p = SoACosineParams(count: count, lanes: lanes, queryNormSq: queryNormSq)
            encoder.setBytes(&p, length: MemoryLayout<SoACosineParams>.stride, index: 3)
        } else {
            var p = SoAL2Params(count: count, lanes: lanes, computeSqrt: computeSqrt)
            encoder.setBytes(&p, length: MemoryLayout<SoAL2Params>.stride, index: 3)
        }
        let w = pipeline.threadExecutionWidth
        let per = (min(pipeline.maxTotalThreadsPerThreadgroup, 256) / w) * w
        let groups = MTLSizeMake((count + per - 1) / per, 1, 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSizeMake(per, 1, 1))
    }
}
```

- [ ] **Step 4 — run, expect pass**
Run: `swift test --filter SoAKernelGoldenTests` → `Executed 2 tests, with 0 failures`.
> If `soa_l2_distance` isn't found at runtime, the `.metal` file isn't in the compiled library —
> confirm it's under `Sources/VectorAccelerate/Metal/Shaders/` (the `.process("Metal/Shaders")`
> resource); no manual registration is needed.

- [ ] **Step 5 — commit**
```bash
git add Sources/VectorAccelerate/Metal/Shaders/SoADistance.metal Sources/VectorAccelerate/Kernels/Metal4/SoADistanceKernel.swift Tests/VectorAccelerateTests/SoAKernelGoldenTests.swift
git commit -m "Add lane-major SoA L2 + cosine Metal kernels (golden + cosine parity)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire into `MetalComputeProvider` (cached kernel, explicit API)

**Files:** Modify `Sources/VectorAccelerate/Integration/MetalComputeProvider.swift` (widen `context` to
`internal`; add a cached `soaKernel`); Create
`Sources/VectorAccelerate/Integration/MetalComputeProvider+SoA.swift`,
`Tests/VectorAccelerateTests/MetalComputeProviderSoATests.swift`.

- [ ] **Step 1 — failing test**

```swift
// Tests/VectorAccelerateTests/MetalComputeProviderSoATests.swift
import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MetalComputeProviderSoATests: XCTestCase {
    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testProviderScoresAgainstSoASet() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context)
        let candidates = (0..<10).map { Vector512Optimized(repeating: Float($0)) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let query = Vector512Optimized(repeating: 0.0)

        let distances = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
        XCTAssertEqual(distances.count, 10)
        // c_k = [k…] vs q=[0…] ⇒ ‖·‖ = sqrt(512)*k ; nearest is k=0.
        XCTAssertEqual(distances[0], 0, accuracy: 1e-3)

        let nearest = try await provider.findNearest(query: query, in: set, k: 3, metric: .euclidean)
        XCTAssertEqual(nearest.count, 3)
        XCTAssertEqual(nearest.map { $0.index }, [0, 1, 2])
    }
}
```

- [ ] **Step 2 — run, expect fail**
Run: `swift test --filter MetalComputeProviderSoATests/testProviderScoresAgainstSoASet`
Expected: `value of type 'MetalComputeProvider' has no member 'batchDistance' …(against:…)`.

- [ ] **Step 3a — modify `MetalComputeProvider.swift`** (three edits)

1. Widen `context` so the extension file can use it. Change:
```swift
    private let context: Metal4Context
```
to:
```swift
    let context: Metal4Context          // internal: used by MetalComputeProvider+SoA
```

2. Add a cached SoA kernel property, next to the other kernels:
```swift
    let soaKernel: SoADistanceKernel    // internal: lane-major SoA scoring (built once)
```

3. In the **primary** `init(context:configuration:decisionEngine:)`, after
`self.cosineProvider = try await CosineKernelDistanceProvider(context: context)`, add:
```swift
        self.soaKernel = try await SoADistanceKernel(context: context)
```

- [ ] **Step 3b — implement the extension**

```swift
// Sources/VectorAccelerate/Integration/MetalComputeProvider+SoA.swift
import Foundation
@preconcurrency import Metal
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension MetalComputeProvider {

    /// Zero-copy batch distance from `query` to every candidate in a prebuilt SoA set.
    /// Euclidean → L2 distance; cosine → 1 − similarity. Other metrics are unsupported on this path.
    func batchDistance<V: SoACompatible & VectorProtocol>(
        query: V, against set: SoACandidateSet<V>, metric: SupportedDistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        guard metric == .euclidean || metric == .cosine else {
            throw VectorError.invalidInput("SoA scoring supports euclidean and cosine only")
        }
        let count = set.layout.count
        guard count > 0 else { return [] }

        let qa = query.toArray()
        let qToken = try await context.getBuffer(for: qa)
        let outToken = try await context.getBuffer(size: count * MemoryLayout<Float>.stride)
        let qNormSq: Float = metric == .cosine ? qa.reduce(0) { $0 + $1 * $1 } : 0

        try await context.executeAndWait { cb, enc in
            self.soaKernel.encode(
                into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                distancesBuffer: outToken.buffer, count: count, lanes: set.layout.lanes,
                metric: metric, computeSqrt: true, queryNormSq: qNormSq)
            qToken.keepAlive(until: cb)
            outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin SoA until GPU completes
        }
        return outToken.copyData(as: Float.self, count: count)
    }

    /// k nearest candidates in a prebuilt SoA set, nearest-first. Distance kernel → CPU top-K.
    func findNearest<V: SoACompatible & VectorProtocol>(
        query: V, in set: SoACandidateSet<V>, k: Int, metric: SupportedDistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, set.layout.count > 0 else { return [] }
        let distances = try await batchDistance(query: query, against: set, metric: metric)
        return Self.selectTopK(distances, k: min(k, distances.count), largerIsCloser: false)
    }
}
```

- [ ] **Step 4 — run, expect pass**
Run: `swift test --filter MetalComputeProviderSoATests/testProviderScoresAgainstSoASet` → pass.

- [ ] **Step 5 — commit**
```bash
git add Sources/VectorAccelerate/Integration/MetalComputeProvider.swift Sources/VectorAccelerate/Integration/MetalComputeProvider+SoA.swift Tests/VectorAccelerateTests/MetalComputeProviderSoATests.swift
git commit -m "MetalComputeProvider: zero-copy SoA batchDistance + findNearest (cached kernel)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Correctness — CPU-reference parity at {512, 768, 1536} + build-once-reuse

Closes the brief's done-criteria (the external draft tested only counts). Pure-test task: written
tests should pass against Tasks 1–3; a failure means a real kernel bug to fix before proceeding.

**Files:** Create `Tests/VectorAccelerateTests/SoAScoringParityTests.swift`.

- [ ] **Step 1 — write the tests**

```swift
// Tests/VectorAccelerateTests/SoAScoringParityTests.swift
import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoAScoringParityTests: XCTestCase {
    var context: Metal4Context!
    var provider: MetalComputeProvider!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        context = try await Metal4Context()
        provider = try await MetalComputeProvider(context: context)
    }
    override func tearDown() async throws { context = nil; provider = nil }

    // Deterministic values in [-1, 1) (no Date/Math.random).
    private func lcg(_ n: Int, seed: UInt64) -> [Float] {
        var s = seed &+ 0x9E3779B97F4A7C15
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float((s >> 33) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1
        }
    }
    private func refL2(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<a.count { let d = a[i]-b[i]; s += d*d }; return s.squareRoot()
    }

    private func assertParity<V: SoACompatible & VectorProtocol>(_ type: V.Type, dim: Int) async throws
        where V.Scalar == Float {
        let query = try V(lcg(dim, seed: UInt64(dim)))
        let candArrays = (0..<64).map { lcg(dim, seed: UInt64(dim) &+ UInt64($0) &+ 1) }
        let candidates = try candArrays.map { try V($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)
        let qa = query.toArray()

        let l2 = try await provider.batchDistance(query: query, against: set, metric: .euclidean)
        for (i, c) in candArrays.enumerated() {
            let ref = refL2(qa, c)
            XCTAssertEqual(l2[i], ref, accuracy: max(1e-2, abs(ref) * 1e-3), "L2 dim \(dim) idx \(i)")
        }
        let cos = try await provider.batchDistance(query: query, against: set, metric: .cosine)
        for (i, c) in candArrays.enumerated() {
            let ref = SoAKernelGoldenTests.refCosineDistance(qa, c)
            XCTAssertEqual(cos[i], ref, accuracy: max(1e-3, abs(ref) * 1e-3), "cosine dim \(dim) idx \(i)")
        }
    }

    func testParity512() async throws { try await assertParity(Vector512Optimized.self,  dim: 512) }
    func testParity768() async throws { try await assertParity(Vector768Optimized.self,  dim: 768) }
    func testParity1536() async throws { try await assertParity(Vector1536Optimized.self, dim: 1536) }

    /// Build the set once, query it twice — proves the bridged buffer is reused across queries and
    /// each query matches a brute-force CPU top-K.
    func testBuildOnceReuseAcrossQueries() async throws {
        let dim = 768
        let candArrays = (0..<200).map { lcg(dim, seed: 1000 &+ UInt64($0)) }
        let candidates = try candArrays.map { try Vector768Optimized($0) }
        let set = try SoACandidateSet(candidates: candidates, device: context.device)

        for qSeed in [UInt64(7), 99] {
            let qa = lcg(dim, seed: qSeed)
            let query = try Vector768Optimized(qa)
            let refTop = candArrays.enumerated()
                .map { (index: $0.offset, d: refL2(qa, $0.element)) }
                .sorted { $0.d < $1.d }.prefix(5).map { $0.index }
            let got = try await provider.findNearest(query: query, in: set, k: 5, metric: .euclidean)
            XCTAssertEqual(got.map { $0.index }, refTop, "reuse query seed \(qSeed)")
        }
    }
}
```

- [ ] **Step 2 — run**
Run: `swift test --filter SoAScoringParityTests`
Expected: `Executed 4 tests, with 0 failures`. (If a parity test fails, fix the kernel — likely a
lane/count indexing or cosine-floor mismatch — before continuing.)

- [ ] **Step 3 — commit**
```bash
git add Tests/VectorAccelerateTests/SoAScoringParityTests.swift
git commit -m "Tests: SoA scoring CPU parity at 512/768/1536 + build-once-reuse

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Doc hygiene — mark the real R1–R3 resolved

**Files:** Modify `docs/VECTORCORE_INTEGRATION_REQUESTS.md`. (Edit the ACTUAL R1/R2/R3 entries — *not*
the external draft's invented labels. R4 is already marked delivered+adopted.)

- [ ] **Step 1 — edit.** Under each of the three real headers, insert a resolved blockquote
immediately after the header line:

Under `### R1 — Page-align the SoA batch buffer (P1, blocks zero-copy batch search)`:
```markdown
> **✅ RESOLVED — VectorCore 0.3.0.** `SoA.build(from:pageAligned: true)` ships the page-aligned batch
> buffer; consumed by `SoACandidateSet` (zero-copy via `makeNoCopyBuffer`). Validated on Apple Silicon.
```
Under `### R2 — Publicly expose the SoA buffer pointer + byte length (P1)`:
```markdown
> **✅ RESOLVED — VectorCore 0.3.0.** `SoA.pageAlignedBytes` (base + page-rounded length) +
> `consumeAllocation()` + the `SoALayout` descriptor are public/stable; consumed in borrow mode.
```
Under `### R3 — Confirm the release that ships AlignedMemory page alignment`:
```markdown
> **✅ RESOLVED.** Shipped in the **VectorCore 0.3.0** tag (pinned `from: "0.3.0"`); page-size handshake
> (16 KB) validated by the SoA bridge smoke-test + golden-fixture parity.
```

- [ ] **Step 2 — verify**
Run: `grep -n "✅ RESOLVED" docs/VECTORCORE_INTEGRATION_REQUESTS.md` → shows three new lines (R1–R3) plus R4.

- [ ] **Step 3 — commit**
```bash
git add docs/VECTORCORE_INTEGRATION_REQUESTS.md
git commit -m "docs: mark R1-R3 resolved (delivered in VectorCore 0.3.0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Full suite green

- [ ] `swift build` → `Build complete!`
- [ ] `swift test --filter SoA` → all SoA* tests pass (or skip cleanly without Metal).
- [ ] `swift test` → full suite green (was 1497 tests / 0 failures; this adds ~9). Investigate any new
  failure before declaring done.
- [ ] Do **not** stage unrelated working-tree files; use explicit pathspecs as above.

---

## Self-review (plan author)

- **Spec coverage (brief §7/§8):** SoACandidateSet + borrow mode (T1); lane-major L2+cosine kernels,
  FLT_MIN floor, golden fixture (T2); provider explicit API, cached kernel, internal `context` (T3);
  CPU parity {512,768,1536} + cosine + build-once-reuse (T4); R1–R3 doc (T5); suite (T6). Out-of-scope
  items (FP16, batched single-dispatch, fused top-K, DynamicVector, transparent auto-detect) untouched.
- **Blocking fixes applied:** private→internal `context` + same-module extension (B1); `SoACompatible &
  VectorProtocol where Scalar==Float` (B2); `withUnsafeRawBuffer` (B3); `.invalidInput` (B4);
  `Self.selectTopK` (B5).
- **Non-blocking fixes:** cosine `FLT_MIN`/`sqrt(a)*sqrt(b)`/NaN-clamp (not 1e-8); cached `soaKernel`;
  deterministic LCG (no `Float.random`); `try MetalDevice()` / `context.device` (no bogus `await`);
  real R1–R3 doc edit with explicit pathspec (no `git commit -am`).
- **Type consistency:** `SoACandidateSet<V: SoACompatible>`; provider methods `<V: SoACompatible &
  VectorProtocol> … where V.Scalar == Float`; kernel encode signature identical across call sites;
  `SoAL2Params`/`SoACosineParams` (16 B) match the MSL structs.
- **Lifetime:** every GPU call pins `set` via `addCompletedHandler { _ in _ = set }` (borrow mode).
