//
//  TopKSelectionAdoptionTests.swift
//  VectorAccelerateTests
//
//  Coverage for MetalComputeProvider.selectTopK's adoption of VectorCore's TopKSelection:
//  parity against a naive brute-force reference (both orderings), deterministic tie-breaking,
//  and empty/k=0 guards.
//
//  Pure CPU: `selectTopK` is a `static func` on the `MetalComputeProvider` actor, and static
//  members of an actor are nonisolated by default — so it is called directly and synchronously
//  here without ever constructing a `MetalComputeProvider` instance or touching a GPU device. The
//  class still carries the actor's `@available(macOS 26.0, ...)` attribute because referencing
//  any member of the type (even a nonisolated static one) requires it — this is a compile-time
//  availability annotation, unrelated to runtime `MTLCreateSystemDefaultDevice()` presence, so no
//  `XCTSkip` is needed.
//
//  See .superpowers/sdd/soa-060-release-hardening/task-2-brief.md
//

import XCTest
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class TopKSelectionAdoptionTests: XCTestCase {

    // MARK: - Independent reference

    /// Naive brute-force top-k: full sort by (`largerIsCloser` ? descending : ascending) distance,
    /// ties broken by ascending original index — independent ground truth for `selectTopK`, built
    /// without any of VectorCore's `TopKSelection` machinery.
    private func referenceTopK(
        _ distances: [Float], k: Int, largerIsCloser: Bool
    ) -> [(index: Int, distance: Float)] {
        guard k > 0, !distances.isEmpty else { return [] }
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }
        let sorted = pairs.sorted { a, b in
            if a.distance != b.distance {
                return largerIsCloser ? (a.distance > b.distance) : (a.distance < b.distance)
            }
            return a.index < b.index
        }
        return Array(sorted.prefix(min(k, distances.count)))
    }

    /// Deterministic pseudo-random distances in a bounded range — exercises both the heap-select
    /// path (k < n/10) and the sort-select path (k >= n/10) of `TopKSelection.select`.
    private func randomDistances(n: Int, seed: UInt64) -> [Float] {
        var rng = TestRNG(seed: seed)
        return (0..<n).map { _ in rng.nextFloat(in: -1000...1000) }
    }

    private func kValues(for n: Int) -> [Int] {
        Array(Set([1, 3, n / 2, n, n + 5])).sorted()
    }

    // MARK: - Parity: distance metrics (largerIsCloser == false)

    func testParityAgainstNaiveReference_smallerIsCloser() {
        for n in [10, 1000, 5000] {
            let distances = randomDistances(n: n, seed: 0xC0FFEE_0000 &+ UInt64(n))
            for k in kValues(for: n) {
                let got = MetalComputeProvider.selectTopK(distances, k: k, largerIsCloser: false)
                let want = referenceTopK(distances, k: k, largerIsCloser: false)

                XCTAssertEqual(got.count, min(k, n), "count n=\(n) k=\(k)")
                XCTAssertEqual(got.map { $0.index }, want.map { $0.index }, "indices n=\(n) k=\(k)")
                XCTAssertEqual(got.map { $0.distance }, want.map { $0.distance }, "distances n=\(n) k=\(k)")
            }
        }
    }

    // MARK: - Parity: dotProduct (largerIsCloser == true)

    func testParityAgainstNaiveReference_dotProductLargerIsCloser() {
        for n in [10, 1000, 5000] {
            let distances = randomDistances(n: n, seed: 0xBEEF_0000 &+ UInt64(n))
            for k in kValues(for: n) {
                let got = MetalComputeProvider.selectTopK(distances, k: k, largerIsCloser: true)
                let want = referenceTopK(distances, k: k, largerIsCloser: true)

                XCTAssertEqual(got.count, min(k, n), "count n=\(n) k=\(k)")
                XCTAssertEqual(got.map { $0.index }, want.map { $0.index }, "indices n=\(n) k=\(k)")
                XCTAssertEqual(got.map { $0.distance }, want.map { $0.distance }, "distances n=\(n) k=\(k)")
            }
        }
    }

    // MARK: - Tie determinism (strict improvement over the previous unstable full-sort)

    func testTiesResolveByAscendingIndex_smallerIsCloser() {
        let distances = [Float](repeating: 1.0, count: 20)
        let got = MetalComputeProvider.selectTopK(distances, k: 8, largerIsCloser: false)
        XCTAssertEqual(got.map { $0.index }, Array(0..<8))
        XCTAssertTrue(got.allSatisfy { $0.distance == 1.0 })
    }

    func testTiesResolveByAscendingIndex_dotProduct() {
        let distances = [Float](repeating: 0.5, count: 20)
        let got = MetalComputeProvider.selectTopK(distances, k: 8, largerIsCloser: true)
        XCTAssertEqual(got.map { $0.index }, Array(0..<8))
        XCTAssertTrue(got.allSatisfy { $0.distance == 0.5 })
    }

    /// `0.0` and `-0.0` must be indifferent for tie purposes on the negated (dotProduct) path —
    /// ordering only ever compares magnitudes via `<`/`>`, never bit patterns.
    func testNegativeZeroIndifference_dotProduct() {
        let distances: [Float] = [0.0, -0.0, 1.0, -1.0]
        let got = MetalComputeProvider.selectTopK(distances, k: 4, largerIsCloser: true)
        // Descending by value: 1.0 (idx 2), then the {0.0, -0.0} tie (idx 0 before idx 1), then -1.0 (idx 3).
        XCTAssertEqual(got.map { $0.index }, [2, 0, 1, 3])
    }

    // MARK: - Empty / k=0 guards

    func testEmptyDistancesReturnsEmpty() {
        XCTAssertTrue(MetalComputeProvider.selectTopK([], k: 5, largerIsCloser: false).isEmpty)
        XCTAssertTrue(MetalComputeProvider.selectTopK([], k: 5, largerIsCloser: true).isEmpty)
    }

    func testZeroKReturnsEmpty() {
        let distances: [Float] = [1, 2, 3]
        XCTAssertTrue(MetalComputeProvider.selectTopK(distances, k: 0, largerIsCloser: false).isEmpty)
        XCTAssertTrue(MetalComputeProvider.selectTopK(distances, k: 0, largerIsCloser: true).isEmpty)
    }

    func testEmptyAndZeroKTogetherReturnsEmpty() {
        XCTAssertTrue(MetalComputeProvider.selectTopK([], k: 0, largerIsCloser: false).isEmpty)
    }
}
