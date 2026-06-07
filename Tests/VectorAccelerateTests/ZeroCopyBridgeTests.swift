//
//  ZeroCopyBridgeTests.swift
//  VectorAccelerateTests
//
//  Groundwork for bridging VectorCore's page-aligned storage into Metal with no copy
//  (Apple-Silicon UMA `makeBuffer(bytesNoCopy:)`). See docs/VECTORCORE_INTEGRATION_REQUESTS.md.
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ZeroCopyBridgeTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
        context = try await Metal4Context()
    }

    override func tearDown() async throws {
        context = nil
        try await super.tearDown()
    }

    /// A page-aligned region wraps as an `MTLBuffer` with NO copy — the buffer's `contents()`
    /// aliases the original allocation (proof nothing was copied across the memory bus).
    func testPageAlignedRegionBridgesWithoutCopy() async throws {
        let device = context.device
        let count = 4096
        let byteCount = MetalDevice.pageAlignedLength(count * MemoryLayout<Float>.stride)
        let bytes = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: MetalDevice.pageSize)
        let floats = bytes.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { floats[i] = Float(i) }

        guard let buffer = device.makeNoCopyBuffer(
            bytes: bytes,
            length: byteCount,
            deallocator: { ptr, _ in ptr.deallocate() }
        ) else {
            bytes.deallocate()
            return XCTFail("page-aligned region should bridge without copying")
        }

        // Same backing memory => true zero-copy.
        XCTAssertEqual(buffer.contents(), bytes)
        let readBack = buffer.contents().bindMemory(to: Float.self, capacity: count)
        XCTAssertEqual(readBack[0], 0)
        XCTAssertEqual(readBack[count - 1], Float(count - 1))
        // `buffer` releasing at scope exit runs the deallocator, freeing `bytes`.
    }

    /// Misaligned base or non-page-multiple length is rejected (→ nil), signalling the caller to
    /// fall back to a staged copy.
    func testUnalignedRegionIsRejected() async throws {
        let device = context.device
        let page = MetalDevice.pageSize
        let bytes = UnsafeMutableRawPointer.allocate(byteCount: page * 2, alignment: page)
        defer { bytes.deallocate() }

        XCTAssertNil(device.makeNoCopyBuffer(bytes: bytes.advanced(by: 16), length: page),
                     "non-page-aligned base must be rejected")
        XCTAssertNil(device.makeNoCopyBuffer(bytes: bytes, length: page - 1),
                     "non-page-multiple length must be rejected")
    }
}
