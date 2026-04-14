//
//  FactoryWALTests.swift
//  VectorAccelerate
//
//  Tests that .flat(...) and .ivf(...) factory methods correctly forward
//  walConfiguration to the main IndexConfiguration init.
//

import XCTest
@testable import VectorAccelerate

final class FactoryWALTests: XCTestCase {
    var tempDirectory: URL!

    override func setUp() async throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FactoryWALTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func randomVector(_ dimension: Int) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }

    // MARK: - Configuration Unit Tests

    func testFlatFactoryDefaultsToWALDisabled() {
        let config = IndexConfiguration.flat(dimension: 128)
        XCTAssertEqual(config.walConfiguration, .disabled)
    }

    func testIVFFactoryDefaultsToWALDisabled() {
        let config = IndexConfiguration.ivf(dimension: 128, nlist: 4, nprobe: 2)
        XCTAssertEqual(config.walConfiguration, .disabled)
    }

    func testFlatFactoryForwardsWALConfiguration() {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = IndexConfiguration.flat(
            dimension: 128,
            walConfiguration: .enabled(directory: walDir)
        )
        XCTAssertTrue(config.walConfiguration.enabled)
        XCTAssertEqual(config.walConfiguration.directory, walDir)
    }

    func testIVFFactoryForwardsWALConfiguration() {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = IndexConfiguration.ivf(
            dimension: 128,
            nlist: 4,
            nprobe: 2,
            walConfiguration: .enabled(directory: walDir)
        )
        XCTAssertTrue(config.walConfiguration.enabled)
        XCTAssertEqual(config.walConfiguration.directory, walDir)
    }

    // MARK: - Integration: Factory-Configured WAL Recovery

    func testFlatFactoryWALRecovery() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal-flat")
        let config = IndexConfiguration.flat(
            dimension: 128,
            capacity: 1000,
            walConfiguration: .enabled(directory: walDir)
        )

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let handle1 = try await index1.insert(randomVector(128))
        let handle2 = try await index1.insert(randomVector(128))
        _ = try await index1.checkpoint()

        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        let count = await index2.count
        XCTAssertEqual(count, 2)
        let contains1 = await index2.contains(handle1)
        let contains2 = await index2.contains(handle2)
        XCTAssertTrue(contains1)
        XCTAssertTrue(contains2)
    }

    func testIVFFactoryWALRecovery() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal-ivf")
        let config = IndexConfiguration.ivf(
            dimension: 128,
            nlist: 4,
            nprobe: 2,
            capacity: 1000,
            walConfiguration: .enabled(directory: walDir)
        )

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let handle1 = try await index1.insert(randomVector(128))
        let handle2 = try await index1.insert(randomVector(128))
        let handle3 = try await index1.insert(randomVector(128))
        _ = try await index1.checkpoint()

        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.ivf(
                dimension: 128,
                nlist: 4,
                nprobe: 2,
                capacity: 1000
            ),
            walDirectory: walDir
        )

        let count = await index2.count
        XCTAssertEqual(count, 3)
        let contains1 = await index2.contains(handle1)
        let contains2 = await index2.contains(handle2)
        let contains3 = await index2.contains(handle3)
        XCTAssertTrue(contains1)
        XCTAssertTrue(contains2)
        XCTAssertTrue(contains3)
    }
}
