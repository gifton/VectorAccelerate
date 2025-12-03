//
//  VectorIndexAccelerationTests.swift
//  VectorAccelerate
//
//  Tests for the VectorIndexAcceleration module.
//

import XCTest
@testable import VectorIndexAcceleration
import VectorAccelerate
import VectorIndex
import VectorCore

/// Test suite for VectorIndexAcceleration module.
///
/// These tests verify that the module compiles correctly and basic
/// functionality works. More comprehensive tests will be added as
/// kernels are migrated in subsequent phases.
final class VectorIndexAccelerationTests: XCTestCase {

    // MARK: - Module Import Tests

    func testModuleImports() {
        // Verify all modules are accessible
        // This test passing means the dependency graph is correct
        XCTAssertEqual(VectorIndexAccelerationVersion.current, "0.1.0-alpha")
        XCTAssertEqual(VectorIndexAccelerationVersion.minimumVAVersion, "0.2.0")
        XCTAssertEqual(VectorIndexAccelerationVersion.minimumVIVersion, "0.1.3")
    }

    // MARK: - Configuration Tests

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testDefaultConfiguration() {
        let config = IndexAccelerationConfiguration.default

        XCTAssertEqual(config.minimumCandidatesForGPU, 500)
        XCTAssertEqual(config.minimumOperationsForGPU, 50_000)
        XCTAssertTrue(config.useFusedKernels)
        XCTAssertFalse(config.forceGPU)
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testAggressiveConfiguration() {
        let config = IndexAccelerationConfiguration.aggressive

        XCTAssertEqual(config.minimumCandidatesForGPU, 100)
        XCTAssertTrue(config.preallocateBuffers)
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testConservativeConfiguration() {
        let config = IndexAccelerationConfiguration.conservative

        XCTAssertEqual(config.minimumCandidatesForGPU, 5_000)
        XCTAssertFalse(config.preallocateBuffers)
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testBenchmarkingConfiguration() {
        let config = IndexAccelerationConfiguration.benchmarking

        XCTAssertTrue(config.forceGPU)
        XCTAssertTrue(config.enableProfiling)
    }

    // MARK: - Error Tests

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testErrorDescriptions() {
        let error1 = IndexAccelerationError.gpuNotInitialized(operation: "search")
        XCTAssertTrue(error1.localizedDescription.contains("GPU not initialized"))
        XCTAssertTrue(error1.localizedDescription.contains("search"))

        let error2 = IndexAccelerationError.dimensionMismatch(expected: 768, got: 512)
        XCTAssertTrue(error2.localizedDescription.contains("768"))
        XCTAssertTrue(error2.localizedDescription.contains("512"))

        let error3 = IndexAccelerationError.emptyIndex(indexType: "HNSW")
        XCTAssertTrue(error3.localizedDescription.contains("empty"))
    }

    // MARK: - Shader Args Tests

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testHNSWSearchShaderArgs() {
        let args = HNSWSearchShaderArgs(
            dimension: 768,
            efSearch: 64,
            k: 10,
            layer: 2,
            maxConnections: 16
        )

        XCTAssertEqual(args.dimension, 768)
        XCTAssertEqual(args.efSearch, 64)
        XCTAssertEqual(args.k, 10)
        XCTAssertEqual(args.layer, 2)
        XCTAssertEqual(args.maxConnections, 16)
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testKMeansAssignShaderArgs() {
        let args = KMeansAssignShaderArgs(
            dimension: 512,
            numVectors: 10_000,
            numCentroids: 256
        )

        XCTAssertEqual(args.dimension, 512)
        XCTAssertEqual(args.numVectors, 10_000)
        XCTAssertEqual(args.numCentroids, 256)
    }

    // MARK: - Acceleration Decision Tests

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    func testAccelerationDecision() {
        let gpuRecommended = AccelerationDecision(
            useGPU: true,
            reason: .gpuRecommended,
            estimatedSpeedup: 5.0
        )

        XCTAssertTrue(gpuRecommended.useGPU)
        XCTAssertEqual(gpuRecommended.estimatedSpeedup, 5.0)

        let tooSmall = AccelerationDecision(
            useGPU: false,
            reason: .datasetTooSmall,
            estimatedSpeedup: 0.5
        )

        XCTAssertFalse(tooSmall.useGPU)
        XCTAssertEqual(tooSmall.reason, .datasetTooSmall)
    }
}

// MARK: - Index Wrapper Tests

/// Tests for the accelerated index wrappers.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class VectorIndexAccelerationWrapperTests: XCTestCase {

    func testHNSWIndexAcceleratedCreation() async throws {
        let baseIndex = HNSWIndex(dimension: 128, metric: .euclidean)
        let accelerated = HNSWIndexAccelerated(baseIndex: baseIndex)

        let dimension = await accelerated.dimension
        XCTAssertEqual(dimension, 128)

        let isActive = await accelerated.isGPUActive
        XCTAssertFalse(isActive) // Not initialized yet
    }

    func testFlatIndexAcceleratedCreation() async throws {
        let baseIndex = FlatIndex(dimension: 256, metric: .cosine)
        let accelerated = FlatIndexAccelerated(baseIndex: baseIndex)

        let dimension = await accelerated.dimension
        XCTAssertEqual(dimension, 256)
    }

    func testIVFIndexAcceleratedCreation() async throws {
        let baseIndex = IVFIndex(dimension: 512, metric: .euclidean)
        let accelerated = IVFIndexAccelerated(baseIndex: baseIndex)

        let dimension = await accelerated.dimension
        XCTAssertEqual(dimension, 512)
    }

    func testDelegationToBaseIndex() async throws {
        let baseIndex = FlatIndex(dimension: 4, metric: .euclidean)
        let accelerated = FlatIndexAccelerated(baseIndex: baseIndex)

        // Insert via accelerated wrapper
        let id = VectorID()
        try await accelerated.insert(id: id, vector: [1.0, 2.0, 3.0, 4.0], metadata: ["key": "value"])

        // Verify in base index
        let contains = await accelerated.contains(id: id)
        XCTAssertTrue(contains)

        let count = await accelerated.count
        XCTAssertEqual(count, 1)
    }
}
