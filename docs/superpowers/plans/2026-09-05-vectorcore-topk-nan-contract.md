# VectorCore Top-K NaN Ordering Contract — Implementation Plan and Agent Handoff

> **For agentic workers:** Use `superpowers:executing-plans`, if available, to implement this plan task by task. Steps use checkboxes for tracking. The owner will handle the release; this document does not authorize staging, committing, tagging, or publishing.

**Goal:** Establish an explicit NaN-aware ordering contract in VectorCore so CPU and Metal Top-K implementations can agree on candidate membership and result order.

**Architecture:** Use one internal candidate-ordering primitive for ascending and descending selection. Derive heap eviction and final sorting from it, and route existing CPU selection wrappers through the same contract. Preserve public signatures and existing tie-policy choices.

**Tech stack:** Swift, Swift Testing, existing VectorCore CPU/SIMD kernels. No new dependency or Metal implementation required.

**Spec:** The owner-approved Top-K contract is specified in §1 of this self-contained document. Origin: VectorAccelerate Group B, VA3-016, in `docs/audits/HARDENING-HANDOFF.md` and `docs/audits/AUDIT-3-shaders.md` in the VectorAccelerate repository. Those documents are background, not prerequisites for executing this handoff.

**Prepared:** 2026-09-05. Source inspected at `/Users/goftin/dev/gsuite/VSK/VectorCore`, HEAD `ad39ab6` (`Merge pull request #39 from gifton/release/0.3.2`). Recheck the target checkout before editing; the owner may already be working there. The VectorCore source tree was clean at inspection apart from untracked `.antigravitycli/` and `GEMINI.md`.

## Global constraints

- Work in the owner's VectorCore checkout, **not** VectorAccelerate's `.build/checkouts/VectorCore` dependency cache.
- Read applicable repository instructions and preserve unrelated work. The owner controls commits and releases.
- Keep Swift tools version 6.0 and the existing deployment targets: macOS 14, iOS 17, tvOS 17, watchOS 10, visionOS 1.
- Preserve public signatures, default `.smallerIndex`, existing invalid-input behavior, and existing score conventions.
- Keep heap selection O(n log k), with O(k) heap storage; preserve the pointer heap path's O(k) auxiliary memory. Do not replace it with a full sort or materialize n candidate pairs there.
- Observe failing mechanism-specific regressions before fixing code; run targeted tests and full debug/release gates afterward.
- This is a selection contract. It does not authorize changes to distance formulas, overflow rescue, cosine degeneracy policy, LSE/statistics policy, GPU routing, or concurrency architecture.
- No new public comparator API is required. VectorAccelerate can implement the same semantics in MSL using shared fixtures as the contract.

## 1. Required behavior

### 1.1 Candidate ordering

A candidate consists of a Float score and its original input index. Ordering is best-first:

| Case | Minimize / nearest | Maximize / largest similarity |
|---|---|---|
| Two unequal non-NaN scores | Smaller score first | Larger score first |
| One NaN and one non-NaN | Non-NaN first | Non-NaN first |
| Equal non-NaN scores | Resolve using `TieBreaker` | Resolve using `TieBreaker` |
| Two NaNs | Resolve using `TieBreaker` | Resolve using `TieBreaker` |

Additional requirements:

- Infinities are non-NaN numeric scores: ascending order is `-Inf < finite < +Inf < NaN`; descending is `+Inf > finite > -Inf`, then NaN.
- `+0` and `-0` compare equal. They use the tie policy; do not order them by bit pattern.
- `.smallerIndex`: smaller original input index wins, including NaN-versus-NaN ties.
- `.insertionOrder`: preserve its current equivalence to `.smallerIndex` for public array/pointer APIs, which scan original positions in order. Do not introduce an encounter counter into the internal heap or reinterpret parallel merge order as input order.
- `.smallerValue`: retain value-only tie semantics. Equal numeric values and two NaNs are equivalent; do not promise deterministic identities/order within those equivalence classes. NaNs still rank last.
- With unique indices, `.smallerIndex` defines a strict total ordering of candidates. `.smallerValue` defines a strict weak ordering with equivalence classes. Do not describe all policies as total orders.
- Do not implement descending order by reversing an ascending result or negating a comparison result: either approach can reverse NaN placement or index ties.
- Tie equality is exact Float equality, plus the explicit two-NaN case. No epsilon or approximate-equality ties.

### 1.2 Membership, cardinality, and values

The same candidate ordering must govern admission, eviction, heap repair, partial-result merging, and final output sorting. A final sort alone cannot recover the correct candidates after incorrect eviction.

For selection APIs that currently return `min(k, n)` entries when `k > 0`, keep that count. NaNs are candidates, not missing rows: they fill remaining positions only when fewer than k non-NaN candidates exist. All-NaN inputs under `.smallerIndex` select indices `0..<min(k, n)`.

Retain the selected scores. Do not replace NaN with infinity, zero, or a sentinel. Precomputed selection should copy original values, preserving signed zero. No new signaling-NaN, NaN-payload-ordering, or payload-preservation guarantee is required.

Keep existing API-specific input validation: `TopKSelection.select(k: 0, ...)` returns empty, while `Operations.findNearest` currently throws for nonpositive k. This slice does not unify validation behavior.

### 1.3 Index identity and scope of determinism

Pointer API `ids` are output labels, mapped **after** selection. `.smallerIndex` refers to original buffer position, not the external ID. Test nonmonotonic and duplicate IDs to prevent accidental reinterpretation.

Parallel partial heaps must carry global original indices, not chunk-local positions. For the default policy, changing chunk boundaries or merge order must not change membership or result order for identical candidate scores. Candidate indices are unique across disjoint chunks; deduplicating overlapping candidate sets is outside scope.

The guarantee applies to identical scores presented to the comparator. CPU/GPU distance accumulation can still produce different scores through rounding. Existing optimized Euclidean paths select using squared distance and convert to rooted distance afterward: preserve that convention. Distinct squared scores that round to the same rooted Float are not newly required to tie at the selection boundary.

## 2. Confirmed implementation entry points

Paths below are relative to VectorCore. Symbol names are authoritative; line numbers can move.

| File | Current issue / required integration |
|---|---|
| `Sources/VectorCore/Operations/TopKSelection.swift` | `orderedAscending` checks `!=` before `<`, so NaNs bypass index ties. `extractSortedResultDescending` duplicates the same defect. Array, pointer, generic element, and optimized entry points converge on these helpers and `TopKBuffer`. |
| `Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift` | `TopKBuffer.isWorse` lacks NaN classification. It drives admission and heap repair. `mergeTopK` constructs a new buffer without propagating `out.tieBreaker`. |
| `Sources/VectorCore/Operations/Operations.swift` | Generic `findNearest` uses a separate value-only sort. Parallel optimized paths merge `TopKBuffer`s, then `toResults` uses another value-only sort. `gemmBatchRun` converts a selected Euclidean score with `raw > 0 ? raw.squareRoot() : 0`, which converts NaN to zero. |
| `Sources/VectorCore/Operations/BatchOperations.swift` | Private `heapSelect` maintains a separate value-only small-k implementation and value-only large-k sort. Public `findNearest` reaches it via serial/parallel helpers. |
| `Tests/ComprehensiveTests/TopKTieBreakingTests.swift` | Existing Swift Testing coverage for finite ties, array/pointer parity, and optimized selection. Retain it. |
| `Tests/ComprehensiveTests/TopKSelectionTests.swift` | Existing basic selection, optimized-vector, and cardinality tests. Retain it. |
| `Tests/ComprehensiveTests/ExecutionOperationsTests.swift` | Existing Operations integration tests; useful patterns for async wrapper tests. |
| `Tests/ComprehensiveTests/TopKNaNContractTests.swift` | Create focused contract and comparator-law regressions here. |
| `CHANGELOG.md` and public API doc comments | Document this observable behavioral fix and its limits. |

These are source observations, not test-confirmed failures in this handoff. No builds or tests were run while preparing it.

## 3. Task 1 — Implement the shared ordering and core selection contract

**Files:** Modify the two TopKSelection source files; create `TopKNaNContractTests.swift`.

**Interface:** Recommended internal helper, colocated in `TopKSelection`:

```swift
@usableFromInline @inline(__always)
internal static func orderedBefore(
    _ a: (Int, Float), _ b: (Int, Float),
    descending: Bool, tieBreaker: TieBreaker
) -> Bool {
    let aNaN = a.1.isNaN
    let bNaN = b.1.isNaN
    if aNaN != bNaN { return !aNaN }
    if !aNaN && a.1 != b.1 {
        return descending ? a.1 > b.1 : a.1 < b.1
    }
    switch tieBreaker {
    case .smallerIndex, .insertionOrder: return a.0 < b.0
    case .smallerValue: return false
    }
}
```

Preserve the existing `@usableFromInline` / `@inlinable` visibility chain; public inlinable optimized entry points must still compile for downstream consumers. This is suggested implementation code, not a compiled patch.

- [ ] Add the following minimal membership regression and run it before implementation. On the inspected code, the NaN initially at the root cannot be evicted by a numeric candidate; confirm the observed failure rather than relying on that prediction.

```swift
import Testing
@testable import VectorCore

@Suite("Top-K NaN ordering contract")
struct TopKNaNContractTests {
    @Test func heapEvictsInitialNaN() {
        for maximize in [false, true] {
            var heap = TopKBuffer(k: 1, isMinHeap: maximize)
            heap.pushIfBetter(val: .nan, idx: 0)
            heap.pushIfBetter(val: 4, idx: 1)
            #expect(heap.size == 1)
            #expect(heap.idxs[0] == 1)
            #expect(heap.vals[0] == 4)
        }
    }

    @Test func publicSelectionKeepsNaNsLastAndPreservesCount() {
        let scores: [Float] = [.nan, 2, -.infinity, 2,
                               .infinity, -0.0, 0.0, .nan, -3]
        let result = TopKSelection.select(k: 99, from: scores)
        #expect(result.indices == [2, 8, 5, 6, 1, 3, 4, 0, 7])
        #expect(result.count == scores.count)
        for position in result.indices.indices {
            let expected = scores[result.indices[position]]
            let actual = result.distances[position]
            if expected.isNaN {
                #expect(actual.isNaN)
            } else {
                #expect(actual.bitPattern == expected.bitPattern)
            }
        }
    }
}
```

- [ ] Run `swift test --filter TopKNaNContractTests` and record the failing assertions. Existing finite-tie controls should remain green: `swift test --filter TopKTieBreakingTests`.
- [ ] Add `orderedBefore`; keep `orderedAscending` as a thin wrapper passing `descending: false` so existing callers share it. Replace the descending extraction closure with `orderedBefore(..., descending: true, tieBreaker: buffer.tieBreaker)`.
- [ ] Implement heap eviction by reversing candidate operands, **not** reversing score direction or negating the predicate:

```swift
return TopKSelection.orderedBefore(
    (i2, v2), (i1, v1),
    descending: isMinHeap, tieBreaker: tieBreaker
)
```

Here `isMinHeap == true` means retain the largest scores. The root is always the worst retained candidate under the complete policy; therefore a retained NaN belongs ahead of numeric candidates for eviction in either heap mode.

- [ ] Preserve the destination tie policy when rebuilding merged buffers:

```swift
var merged = TopKBuffer(k: out.k, isMinHeap: out.isMinHeap,
                       tieBreaker: out.tieBreaker)
```

All partial heaps participating in one selection must use the same tie policy. Confirm existing callers do so; keep the internal invariant explicit. Do not silently change `.smallerValue` to `.smallerIndex` during merge.

- [ ] Extend the new suite with the exact fixtures and laws in §5, covering both membership and order. Run `swift test --filter 'TopKNaNContractTests|TopKTieBreakingTests|TopKSelectionSuite'` and confirm tests were actually discovered.

## 4. Task 2 — Close CPU wrapper paths that bypass or erase the contract

**Files:** `Operations.swift`, `BatchOperations.swift`, and the new contract tests. Keep public APIs and routing unchanged.

**Interfaces:** Consume `TopKSelection.orderedAscending`, the corrected `TopKBuffer`, and existing `heapSelectSmallK` / `sortSelectLargeK` helpers. Produce the same public result types as before.

- [ ] Add failing public-wrapper regressions before each integration fix. Use finite ties to isolate output-sort failures; use computed NaN scores to test retention and formatting. Run with a CPU provider so external GPU delegation cannot mask the path.
- [ ] Replace `Operations.findNearest`'s generic value-only result sort with canonical Top-K selection over its already-computed distance array:

```swift
let selection = TopKSelection.select(k: k, from: distances)
return selection.toTuples().map {
    NearestNeighborResult(index: $0.index, distance: $0.distance)
}
```

- [ ] In `Operations.toResults`, sort the existing transformed `(index, value)` pairs using `TopKSelection.orderedAscending(..., buf.tieBreaker)`. Keep the current output conventions: Euclidean square root; dot similarity negated into distance. NaNs remain last after either transformation.
- [ ] In `gemmBatchRun`, preserve a selected NaN through Euclidean output conversion while retaining the existing nonpositive finite clamp:

```swift
let dist = euclid
    ? (raw.isNaN ? raw : (raw > 0 ? raw.squareRoot() : 0))
    : raw
```

Add a regression exercising a GEMM-sized batch (currently at least 8 queries and 256 candidates, using an eligible optimized vector type). Select enough entries to include NaNs and assert they remain NaNs. If upstream metric computation already changes them before selection, isolate the formatting conversion in a small internal helper and test it directly; document the separate metric behavior instead of claiming this comparator slice fixes it.

- [ ] Replace `BatchOperations.heapSelect`'s duplicated value-only algorithms with existing canonical pair helpers, preserving candidate indices rather than re-enumerating a reordered list:

```swift
let actualK = min(k, elements.count)
guard actualK > 0 else { return [] }
if actualK < elements.count / 10 {
    return TopKSelection.heapSelectSmallK(elements, k: actualK,
                                         tieBreaker: .smallerIndex)
}
return TopKSelection.sortSelectLargeK(elements, k: actualK,
                                      tieBreaker: .smallerIndex)
```

- [ ] Verify global indices survive parallel chunk construction and partial merging. Test disjoint chunks merged in forward and reverse order with explicit expected indices. Do not infer coverage solely from an async API name; directly test merge behavior and exercise the actual public CPU route.
- [ ] Run the contract suite and `swift test --filter ExecutionOperationsTests`; inspect discovery output and include any additional test suite names used for wrapper regressions.

Do not rewrite metric kernels to manufacture NaNs for these tests. Primitive tests can inject exact scores, and public tests should use data whose metric behavior is established by a control assertion. Third-party `BatchKernelProvider` implementations remain responsible for their own results; retain delegation behavior.

## 5. Acceptance fixtures and test matrix

### Exact expected results

Canonical scores, indexed in input order:

```text
index:     0    1     2    3     4    5    6    7    8
score:   NaN    2  -Inf    2  +Inf   -0   +0  NaN   -3

minimize, k=9: [2, 8, 5, 6, 1, 3, 4, 0, 7]
maximize, k=9: [4, 1, 3, 5, 6, 8, 2, 0, 7]
minimize, k=5: [2, 8, 5, 6, 1]
maximize, k=5: [4, 1, 3, 5, 6]
```

Use prefixes of these literal lists for smaller k. For maximization, test `TopKBuffer` plus descending extraction and the public `nearestDotProduct512` path. There is no public descending precomputed `select` overload today; do not add one just for testing.

| Required case | Fixture / assertion |
|---|---|
| Actual public heap path | n=100, k=3, initial scores `[NaN, NaN, NaN, 2, 1, 1, 1]`, remaining scores 100. Expected indices `[4, 5, 6]`. Exercise array, pointer, and generic-element overloads. |
| Actual public sort path | Canonical n=9 fixture above, k=5 and k=9. |
| Algorithm crossover | n=100: k=9 takes heap, k=10 takes sort under the current strict integer condition `actualK < count / 10`. Use a literal ordered prefix for each and common-prefix agreement. |
| NaNs must fill heap results | n=100, all NaN except score[99]=1, k=3. Expected `[99, 0, 1]`. Confirm returned tail values are NaN. |
| All NaN | n=100, k=3 -> `[0,1,2]`; small-n full-sort path also selects the smallest original positions. |
| Numeric infinity outranks NaN | Minimize `[NaN,+Inf]` with k=1 -> `[1]`; maximize `[NaN,-Inf]` with k=1 -> `[1]`. |
| Numeric ties crossing k | n=100 all scores 1, k=3 -> `[0,1,2]`; also insert candidates into an internal buffer with global indices in reverse order. |
| Signed zero | `[+0,-0,+0,-0]`, k=3 -> `[0,1,2]`; selected original bit patterns preserved by precomputed APIs. |
| Pointer external IDs | Scores `[1,1,1]`, IDs `[90,10,50]`, k=2 -> output IDs `[90,10]`. Repeat with duplicate IDs and with NaNs; labels do not affect membership. |
| Generic element API | Elements carry a separate label and Float score; selected labels correspond to literal expected original positions. Exercise both k/n paths. |
| Explicit tie policies | Default equals `.smallerIndex`; `.insertionOrder` matches for input-ordered public scans. `.smallerValue` keeps numeric-before-NaN and valid cardinality without asserting an index preference for equal scores. |
| Merge | Split canonical candidates into disjoint chunks carrying original indices, use identical k/policy/direction, merge in multiple orders. Final indices equal the relevant literal prefix. Repeat with all-NaN and cutoff ties. |
| Public maximization | Query Vector512Optimized of ones; constant candidates with scores induced by values `[NaN,2,2,-1,NaN]`, k=4 -> indices `[1,2,3,0]`, NaN last. Verify direct dot scores first. |
| Public CPU wrappers | `Operations.findNearest` generic and optimized branches, `BatchOperations.findNearest`, and eligible GEMM batch formatting: literal expected indices, cardinality, and selected NaN values. |
| Existing edges | Empty input, k=0, k=1, k=n, k>n retain each API's existing behavior. Do not pass invalid zero-capacity internal heaps unless separately hardening that contract. |

Check comparator laws directly over a small candidate set containing both zeros, finite ties, both infinities, and multiple quiet NaNs with distinct indices. For each direction and tie policy assert:

```swift
// before is a test closure invoking the production orderedBefore helper.
for a in candidates {
    #expect(!before(a, a))
    for b in candidates {
        if before(a, b) { #expect(!before(b, a)) }
        for c in candidates {
            if before(a, b) && before(b, c) { #expect(before(a, c)) }
            let abEquivalent = !before(a, b) && !before(b, a)
            let bcEquivalent = !before(b, c) && !before(c, b)
            if abEquivalent && bcEquivalent {
                #expect(!before(a, c) && !before(c, a))
            }
        }
    }
}
```

Comparator-law tests supplement the literal expected-result tests; they do not replace them. Do not generate all expected outputs by sorting with the production helper. Do not compare NaN-containing Float arrays or `TopKResult` values with plain `==`; compare indices exactly and values using explicit NaN-aware assertions. Leave `TopKResult.Equatable` and Codable behavior unchanged.

## 6. Task 3 — Documentation, verification, and release handoff

**Files:** Public doc comments in affected APIs; `CHANGELOG.md`; the repository's usual verification record if required by local instructions.

- [ ] Document NaN-last ordering, min/max direction, exact ties, zero handling, count preservation, and pointer index-versus-ID semantics. Correct `TopKResult`'s blanket ascending-distance description to acknowledge the existing descending similarity API.
- [ ] Explain `.smallerValue`'s weaker guarantee and the scope of `.insertionOrder`. Do not call this IEEE `totalOrder`: the chosen NaN and signed-zero behavior is an application contract.
- [ ] Add a changelog entry such as:

> Top-K selection now ranks NaN scores after all numeric scores in both minimization and maximization. The default smaller-index policy resolves equal numeric scores and NaN ties consistently for selection and output ordering. NaNs remain in results when needed to satisfy the requested count. Pointer IDs remain output labels; ties use original input positions.

- [ ] Run targeted regressions after the final edits, then full debug and release tests from VectorCore:

```sh
swift test --filter 'TopKNaNContractTests|TopKTieBreakingTests|TopKSelectionSuite'
swift test
swift test -c release
git diff --check
```

Record actual discovered/executed counts, failures, and skips. Separate pre-existing failures from regressions with evidence. Do not copy VectorAccelerate's 1590-test baseline into the VectorCore report.

- [ ] Measure finite-input selection in release mode against the pre-change baseline for n=100,000 with k=10 (heap) and k=20,000 (sort), covering array and pointer entry points. Use fixed-seed mixed values and duplicate-heavy values, identical inputs, warmup, repeated samples, and a consumed output checksum. Record medians and environment; do not invent a hard performance threshold or claim no regression without measurements. Do not add a new benchmark framework for this slice.
- [ ] Review all selection comparisons in the touched call graph for leftover value-only sorting/admission. Record any separate selection APIs found outside that graph rather than claiming a repository-wide guarantee from these tests.
- [ ] Give the owner a release-ready summary: exact changed APIs/paths, red-before-green evidence, debug/release results, performance measurements, and any remaining limitations. The owner chooses the version and performs publication.

## 7. Information to return to the VectorAccelerate agent

After the owner publishes, provide:

1. Released version/tag and commit SHA containing the contract.
2. Confirmation that §1 was implemented, identifying any deviations explicitly.
3. Paths to the contract tests and documentation, including the canonical fixtures above.
4. Debug/release test results and finite-input benchmark comparison.
5. Any wrapper or metric limitations that remain outside the comparator guarantee.

VectorAccelerate will then consume the released dependency, implement the corresponding MSL ordering, and test GPU membership/output against both literal expectations and VectorCore. Its debug plugin metallib and release runtime-compiled shaders require separate verification. This handoff does not mark VA3-016 fixed and does not settle the other Group B policies.
