//
//  GPUAvailabilityGuardTests.swift
//  VectorAccelerateTests
//
//  VectorAccelerate's GPU guards share one runtime Metal check instead of VectorCore's
//  `ComputeDevice.gpu().isAvailable`, which is a compile-time constant in 0.3.3 and removed in the
//  unreleased 0.4.0. Only the available side can be tested: a host without a Metal device can't run
//  this suite.
//

import XCTest
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class GPUAvailabilityGuardTests: XCTestCase {

    func testGuardsShareTheRuntimeMetalCheck() async throws {
        XCTAssertEqual(MetalDevice.isGPUAvailable(), MetalDevice.isAvailable)
        XCTAssertTrue(MetalDevice.isGPUAvailable())
        XCTAssertTrue(AcceleratedVectorFactory.isAccelerationAvailable)
        let provider = try await MetalComputeProvider()
        XCTAssertEqual(provider.device, .gpu(index: 0))
    }
}
