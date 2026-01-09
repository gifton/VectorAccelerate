//
//  WALRecoveryTests.swift
//  VectorAccelerate
//
//  Tests for WAL recovery functionality.
//

import XCTest
@testable import VectorAccelerate

final class WALRecoveryTests: XCTestCase {
    var tempDirectory: URL!

    override func setUp() async throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("WALRecoveryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    // MARK: - Helper

    func randomVector(_ dimension: Int) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }

    func makeConfigWithWAL(walDir: URL) -> IndexConfiguration {
        IndexConfiguration(
            dimension: 128,
            metric: .euclidean,
            capacity: 1000,
            indexType: .flat,
            routingThreshold: 0,
            quantization: .none,
            walConfiguration: .enabled(directory: walDir)
        )
    }

    // MARK: - Tests

    func testRecoveryAfterInserts() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        // 1. Create index with WAL, insert vectors
        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let handle1 = try await index1.insert(randomVector(128))
        let handle2 = try await index1.insert(randomVector(128))

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // 2. Simulate crash - create new index pointing to same WAL
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        // 3. Verify vectors exist
        let count = await index2.count
        XCTAssertEqual(count, 2)

        let contains1 = await index2.contains(handle1)
        let contains2 = await index2.contains(handle2)
        XCTAssertTrue(contains1)
        XCTAssertTrue(contains2)
    }

    func testRecoveryAfterRemoves() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        // 1. Create index, insert and remove
        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let handle1 = try await index1.insert(randomVector(128))
        let handle2 = try await index1.insert(randomVector(128))
        let handle3 = try await index1.insert(randomVector(128))
        try await index1.remove(handle2)

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // 2. Recover
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        // 3. Verify state
        let count = await index2.count
        XCTAssertEqual(count, 2)

        let contains1 = await index2.contains(handle1)
        let contains2 = await index2.contains(handle2)
        let contains3 = await index2.contains(handle3)
        XCTAssertTrue(contains1)
        XCTAssertFalse(contains2)  // Removed
        XCTAssertTrue(contains3)
    }

    func testIdempotentReplay() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        _ = try await index1.insert(randomVector(128))
        _ = try await index1.insert(randomVector(128))

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // Recover twice - should be idempotent
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )
        let count2 = await index2.count
        XCTAssertEqual(count2, 2)

        // Recover again (without closing - simulates double replay)
        let recovered = try await index2.recover()
        XCTAssertEqual(recovered, 0)  // Nothing new to replay
        let count2After = await index2.count
        XCTAssertEqual(count2After, 2)
    }

    func testRecoveryWithNoWAL() async throws {
        // Index without WAL should return 0 from recover
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let recovered = try await index.recover()
        XCTAssertEqual(recovered, 0)
    }

    func testRecoveryWithBatchInsert() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        // 1. Create index with WAL, batch insert vectors
        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let vectors = (0..<5).map { _ in randomVector(128) }
        let handles = try await index1.insert(vectors)
        XCTAssertEqual(handles.count, 5)

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // 2. Recover
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        // 3. Verify all vectors exist
        let count = await index2.count
        XCTAssertEqual(count, 5)

        for handle in handles {
            let contains = await index2.contains(handle)
            XCTAssertTrue(contains)
        }
    }

    func testRecoveryWithBatchRemove() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        // 1. Create index, insert and batch remove
        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let vectors = (0..<5).map { _ in randomVector(128) }
        let handles = try await index1.insert(vectors)

        // Remove first 3
        let handlesToRemove = Array(handles.prefix(3))
        _ = try await index1.remove(handlesToRemove)

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // 2. Recover
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        // 3. Verify state
        let count = await index2.count
        XCTAssertEqual(count, 2)

        // First 3 should be removed
        for handle in handles.prefix(3) {
            let contains = await index2.contains(handle)
            XCTAssertFalse(contains)
        }

        // Last 2 should exist
        for handle in handles.suffix(2) {
            let contains = await index2.contains(handle)
            XCTAssertTrue(contains)
        }
    }

    func testRecoveryPreservesHandleStability() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")

        // 1. Create index with WAL
        let config = makeConfigWithWAL(walDir: walDir)

        let index1 = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors and remember their handles
        let vector1 = randomVector(128)
        let vector2 = randomVector(128)
        let handle1 = try await index1.insert(vector1)
        let handle2 = try await index1.insert(vector2)

        // Checkpoint to flush WAL entries to disk before "crash"
        _ = try await index1.checkpoint()

        // 2. Recover
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        // 3. Verify the same handles work and return correct data
        let recovered1 = try await index2.vector(for: handle1)
        let recovered2 = try await index2.vector(for: handle2)

        XCTAssertNotNil(recovered1)
        XCTAssertNotNil(recovered2)

        // Verify vector data matches
        XCTAssertEqual(recovered1?.count, 128)
        XCTAssertEqual(recovered2?.count, 128)

        // Compare vectors (they should be identical)
        for i in 0..<128 {
            XCTAssertEqual(recovered1![i], vector1[i], accuracy: 1e-6)
            XCTAssertEqual(recovered2![i], vector2[i], accuracy: 1e-6)
        }
    }
}
