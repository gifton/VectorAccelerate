//
//  SoACandidateSetTests.swift
//  VectorAccelerateTests
//
//  Zero-copy borrow-mode bridge of VectorCore's page-aligned SoA into Metal.
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

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
        // Page-rounded length, derived (host-agnostic) rather than hardcoding the 16 KB page size.
        XCTAssertEqual(set.layout.allocatedByteCount, MetalDevice.pageAlignedLength(set.layout.logicalByteCount))
        XCTAssertEqual(set.layout.allocatedByteCount % MetalDevice.pageSize, 0)
        XCTAssertTrue(set.isZeroCopy)
        // Borrow mode: the MTLBuffer aliases the SoA allocation (no copy).
        let (base, _) = try XCTUnwrap(set.soa.pageAlignedBytes)
        XCTAssertEqual(set.buffer.contents(), UnsafeMutableRawPointer(mutating: base))
    }
}
