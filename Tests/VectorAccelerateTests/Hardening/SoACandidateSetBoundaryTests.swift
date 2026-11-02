//
//  SoACandidateSetBoundaryTests.swift
//  VectorAccelerateTests
//
//  SoA_Layout_Contract.md: empty storage has no page-aligned allocation, and a single candidate is
//  still a whole page. The bridge must handle both ends. Metal refuses a zero-length buffer, so an
//  empty set used to throw `bufferAllocationFailed(size: 0)`.
//

import XCTest
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SoACandidateSetBoundaryTests: XCTestCase {

    func testEmptyAndSingleCandidateSets() async throws {
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(fallbackToCPU: false))
        let zero = Vector512Optimized(repeating: 0)

        let one = try SoACandidateSet(candidates: [Vector512Optimized(repeating: 3)], device: context.device)
        XCTAssertEqual(one.layout.count, 1)
        XCTAssertEqual(one.layout.allocatedByteCount, MetalDevice.pageAlignedLength(one.layout.logicalByteCount))
        XCTAssertTrue(one.isZeroCopy)
        let distances = try await provider.batchDistance(query: zero, against: one, metric: .euclidean)
        XCTAssertEqual(distances.count, 1)
        XCTAssertEqual(distances[0], 3 * Float(512).squareRoot(), accuracy: 4 * Float.ulpOfOne * 68)

        let empty = try SoACandidateSet(candidates: [Vector512Optimized](), device: context.device)
        XCTAssertEqual(empty.layout.count, 0)
        XCTAssertNil(empty.soa.pageAlignedBytes, "empty SoA storage provides no page-aligned allocation (contract)")
        XCTAssertFalse(empty.isZeroCopy)
        let noDistances = try await provider.batchDistance(query: zero, against: empty, metric: .euclidean)
        XCTAssertTrue(noDistances.isEmpty)
        let noNeighbors = try await provider.findNearest(query: zero, in: empty, k: 3, metric: .euclidean)
        XCTAssertTrue(noNeighbors.isEmpty)
    }
}
