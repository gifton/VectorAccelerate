//
//  MetalComputeProviderTests.swift
//  VectorAccelerateTests
//
//  Tests for the unified GPU compute façade. See
//  docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md
//

import XCTest
@testable import VectorAccelerate
import VectorCore
@preconcurrency import Metal

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class MetalComputeProviderTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device not available") }
    }

    func testConformsToComputeProviderWithGPUIdentity() async throws {
        let provider = try await MetalComputeProvider()
        // ComputeProvider shim reports GPU identity with real capabilities.
        XCTAssertEqual(provider.device, .gpu(index: 0))
        XCTAssertGreaterThan(provider.maxConcurrency, 0)
        XCTAssertFalse(provider.deviceInfo.name.isEmpty)
        XCTAssertGreaterThan(provider.deviceInfo.maxThreads, 0)
        // It is usable as an existential ComputeProvider, and `execute` runs the closure.
        let p: any ComputeProvider = provider
        let answer = try await p.execute { 41 + 1 }
        XCTAssertEqual(answer, 42)
    }
}
