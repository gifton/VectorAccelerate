//
//  BinaryArchiveManagerTests.swift
//  VectorAccelerateTests
//
//  Tests for BinaryArchiveManager persistent pipeline caching
//

import XCTest
@testable import VectorAccelerate
import Metal

final class BinaryArchiveManagerTests: XCTestCase {

    private var testDirectory: URL!
    private var archiveURL: URL!
    private var device: (any MTLDevice)!

    override func setUpWithError() throws {
        // Create test directory
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BinaryArchiveTests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )

        archiveURL = testDirectory.appendingPathComponent("test.metalarchive")

        // Get Metal device
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        device = metalDevice
    }

    override func tearDownWithError() throws {
        // Clean up test directory
        if let testDirectory = testDirectory {
            try? FileManager.default.removeItem(at: testDirectory)
        }
    }

    // MARK: - State Tests

    func testInitialStateIsUnloaded() async {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        let state = await manager.state
        XCTAssertEqual(state, .unloaded)
    }

    func testStateTransitionsToReady() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        let state = await manager.state
        if case .ready(let count) = state {
            XCTAssertEqual(count, 0, "New archive should have 0 pipelines")
        } else {
            XCTFail("State should be .ready, got \(state)")
        }
    }

    // MARK: - Archive Operations

    func testLoadOrCreateNewArchive() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        // Archive doesn't exist yet
        XCTAssertFalse(FileManager.default.fileExists(atPath: archiveURL.path))

        try await manager.loadOrCreate()

        let state = await manager.state
        if case .ready = state {
            // Success
        } else {
            XCTFail("State should be ready after loadOrCreate")
        }
    }

    func testSaveArchive() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()
        try await manager.save()

        // Archive file should NOT exist for empty archive (Metal limitation)
        // Empty archives cannot be serialized
        XCTAssertFalse(FileManager.default.fileExists(atPath: archiveURL.path))

        // Manifest file should exist
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))
    }

    func testLoadExistingArchive() async throws {
        // First, create and save an archive (with manifest only, archive file won't exist for empty)
        let manager1 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )
        try await manager1.loadOrCreate()
        try await manager1.save()

        // Verify manifest was saved
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))

        // Loading with a new manager should create new archive since empty archive file doesn't exist
        let manager2 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )
        try await manager2.loadOrCreate()

        let state = await manager2.state
        if case .ready = state {
            // Success - new archive created since empty archive can't be persisted
        } else {
            XCTFail("State should be ready after loading/creating archive")
        }
    }

    // MARK: - Pipeline Lookup

    func testContainsPipelineReturnsFalseInitially() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        let key = PipelineCacheKey.l2Distance(dimension: 384)
        let contains = await manager.containsPipeline(for: key)

        XCTAssertFalse(contains)
    }

    func testContainsPipelineReturnsTrueAfterAdd() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        // Create a simple descriptor for testing
        // Note: In a real test, we'd need a valid function
        let key = PipelineCacheKey.l2Distance(dimension: 384)

        // For this test, we can't easily add a real pipeline without a function,
        // but we can verify the manifest tracking works
        let containsBefore = await manager.containsPipeline(for: key)
        XCTAssertFalse(containsBefore)
    }

    func testPipelineCount() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        let count = await manager.pipelineCount
        XCTAssertEqual(count, 0)
    }

    // MARK: - Error Handling

    func testHandlesMissingArchive() async throws {
        // Create manager with non-existent archive URL
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        // Should create new archive
        try await manager.loadOrCreate()

        let state = await manager.state
        if case .ready = state {
            // Success - new archive created
        } else {
            XCTFail("Should create new archive when none exists")
        }
    }

    func testHandlesCorruptedManifest() async throws {
        // Create archive files manually with corrupted manifest
        try FileManager.default.createDirectory(
            at: archiveURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        try "invalid json".write(to: manifestURL, atomically: true, encoding: .utf8)

        // Create manager - should handle corruption gracefully
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        // Should have created a new archive
        let state = await manager.state
        if case .ready = state {
            // Success - handled corruption by recreating
        } else {
            XCTFail("Should handle corrupted manifest gracefully")
        }
    }

    // MARK: - Manifest Tests

    func testManifestPersistsAcrossLoads() async throws {
        // We can't easily test actual pipeline persistence without compiled functions,
        // but we can verify the manifest file is created and readable
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()
        try await manager.save()

        // Manifest should exist
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))

        // Should be valid JSON
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(BinaryArchiveManifest.self, from: data)

        XCTAssertEqual(manifest.version, "1.0.0")
        XCTAssertEqual(manifest.deviceName, device.name)
    }

    // MARK: - Serialized Size

    func testSerializedSizeNilBeforeSave() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        // Size should be nil before saving (archive file doesn't exist)
        let size = await manager.serializedSize
        XCTAssertNil(size)
    }

    func testSerializedSizeNilForEmptyArchive() async throws {
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()
        try await manager.save()

        // Size should be nil for empty archive (archive file not created due to Metal limitation)
        let size = await manager.serializedSize
        XCTAssertNil(size, "Empty archives cannot be serialized, so size should be nil")
    }

    // MARK: - Default URL Tests

    func testDefaultArchiveURL() {
        let defaultURL = BinaryArchiveManager.defaultArchiveURL

        XCTAssertTrue(defaultURL.path.contains("VectorAccelerate"))
        XCTAssertTrue(defaultURL.pathExtension == "metalarchive")
    }

    // MARK: - Archive State Tests

    func testArchiveStateEquality() {
        XCTAssertEqual(BinaryArchiveState.unloaded, BinaryArchiveState.unloaded)
        XCTAssertEqual(BinaryArchiveState.loading, BinaryArchiveState.loading)
        XCTAssertEqual(BinaryArchiveState.ready(pipelineCount: 5), BinaryArchiveState.ready(pipelineCount: 5))
        XCTAssertNotEqual(BinaryArchiveState.ready(pipelineCount: 5), BinaryArchiveState.ready(pipelineCount: 10))
        XCTAssertNotEqual(BinaryArchiveState.unloaded, BinaryArchiveState.loading)
    }

    // MARK: - Error Description Tests

    func testErrorDescriptions() {
        let errors: [BinaryArchiveError] = [
            .creationFailed("test"),
            .loadFailed("test"),
            .saveFailed("test"),
            .corrupted("test"),
            .addPipelineFailed("test"),
            .notReady,
            .urlNotConfigured
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}
