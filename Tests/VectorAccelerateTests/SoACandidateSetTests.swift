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
        XCTAssertEqual(set.layout.allocatedByteCount, 16384)        // 16 KB page (Apple Silicon)
        XCTAssertTrue(set.isZeroCopy)
        // Borrow mode: the MTLBuffer aliases the SoA allocation (no copy).
        let (base, _) = try XCTUnwrap(set.soa.pageAlignedBytes)
        XCTAssertEqual(set.buffer.contents(), UnsafeMutableRawPointer(mutating: base))
    }
}
