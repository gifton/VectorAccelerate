//
//  WALMaintenanceTests.swift
//  VectorAccelerate
//
//  Tests for WAL maintenance functionality (statistics, compaction, auto-checkpoint).
//

import XCTest
@testable import VectorAccelerate

final class WALMaintenanceTests: XCTestCase {
    var tempDirectory: URL!

    override func setUp() async throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("WALMaintenanceTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    // MARK: - Helpers

    func randomVector(_ dimension: Int) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }

    func makeConfigWithWAL(walDir: URL, autoCheckpoint: Int = 0, autoCompact: Int = 0) -> IndexConfiguration {
        IndexConfiguration(
            dimension: 128,
            metric: .euclidean,
            capacity: 1000,
            indexType: .flat,
            routingThreshold: 0,
            quantization: .none,
            walConfiguration: .enabled(
                directory: walDir,
                syncMode: .periodic,
                autoCheckpointThreshold: autoCheckpoint,
                autoCompactThreshold: autoCompact
            )
        )
    }

    // MARK: - Statistics Tests

    func testWALStatisticsReturnsNilWhenDisabled() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let stats = await index.walStatistics()
        XCTAssertNil(stats)
    }

    func testWALStatisticsAfterOperations() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        let index = try await AcceleratedVectorIndex(configuration: config)
        _ = try await index.insert(randomVector(128))
        _ = try await index.insert(randomVector(128))
        _ = try await index.checkpoint()

        let stats = await index.walStatistics()
        XCTAssertNotNil(stats)
        XCTAssertGreaterThan(stats!.currentSequence, 0)
        XCTAssertEqual(stats!.lastCheckpointSequence, stats!.currentSequence)
        XCTAssertGreaterThan(stats!.segmentCount, 0)
    }

    func testWALStatisticsTracksDirtyState() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        let index = try await AcceleratedVectorIndex(configuration: config)
        _ = try await index.insert(randomVector(128))

        // Flush to clear dirty state
        try await index.flushWAL()

        let stats = await index.walStatistics()
        XCTAssertNotNil(stats)
        XCTAssertFalse(stats!.isDirty)
    }

    func testWALStatisticsEntriesSinceCheckpoint() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Checkpoint to establish baseline
        _ = try await index.checkpoint()

        // Insert vectors after checkpoint
        _ = try await index.insert(randomVector(128))
        _ = try await index.insert(randomVector(128))
        _ = try await index.insert(randomVector(128))
        try await index.flushWAL()

        let stats = await index.walStatistics()
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats!.entriesSinceCheckpoint, 3)
    }

    // MARK: - Compaction Tests

    func testWALCompactionWithNoWAL() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Should not throw and return 0
        let reclaimed = try await index.compactWAL()
        XCTAssertEqual(reclaimed, 0)
    }

    func testWALCompaction() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert many vectors to create WAL entries
        for _ in 0..<100 {
            _ = try await index.insert(randomVector(128))
        }
        _ = try await index.checkpoint()

        let statsBefore = await index.walStatistics()
        XCTAssertNotNil(statsBefore)

        // Compact WAL
        let reclaimed = try await index.compactWAL()

        // Should have reclaimed some space (or at least not crashed)
        XCTAssertGreaterThanOrEqual(reclaimed, 0)
    }

    // MARK: - Auto-Checkpoint Tests

    func testAutoCheckpointTriggersAtThreshold() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir, autoCheckpoint: 5)

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 4 vectors - should not trigger checkpoint
        for _ in 0..<4 {
            _ = try await index.insert(randomVector(128))
        }
        try await index.flushWAL()

        let statsBeforeThreshold = await index.walStatistics()
        let checkpointBefore = statsBeforeThreshold?.lastCheckpointSequence ?? 0

        // Insert 5th vector - should trigger auto-checkpoint
        _ = try await index.insert(randomVector(128))
        try await index.flushWAL()

        let statsAfterThreshold = await index.walStatistics()
        XCTAssertNotNil(statsAfterThreshold)
        XCTAssertGreaterThan(statsAfterThreshold!.lastCheckpointSequence, checkpointBefore)
    }

    func testAutoCheckpointDisabledWhenThresholdIsZero() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir, autoCheckpoint: 0)  // Disabled

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert many vectors - no auto-checkpoint should trigger
        for _ in 0..<20 {
            _ = try await index.insert(randomVector(128))
        }
        try await index.flushWAL()

        let stats = await index.walStatistics()
        XCTAssertNotNil(stats)
        // Should have no checkpoint (sequence 0)
        XCTAssertEqual(stats!.lastCheckpointSequence, 0)
    }

    // MARK: - Flush Tests

    func testFlushWAL() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        let index = try await AcceleratedVectorIndex(configuration: config)
        _ = try await index.insert(randomVector(128))

        // Flush should not throw
        try await index.flushWAL()

        // After flush, isDirty should be false
        let stats = await index.walStatistics()
        XCTAssertFalse(stats?.isDirty ?? true)
    }

    func testFlushWALWithNoWAL() async throws {
        let config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Should not throw when WAL is disabled
        try await index.flushWAL()
    }

    // MARK: - Combined Tests

    func testAutoCheckpointWithAutoCompaction() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        // Set auto-checkpoint at 10 operations and auto-compact at 100 bytes (will trigger)
        let config = makeConfigWithWAL(walDir: walDir, autoCheckpoint: 10, autoCompact: 100)

        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert enough to trigger checkpoint and compaction
        for _ in 0..<15 {
            _ = try await index.insert(randomVector(128))
        }
        try await index.flushWAL()

        let stats = await index.walStatistics()
        XCTAssertNotNil(stats)
        // Checkpoint should have been created
        XCTAssertGreaterThan(stats!.lastCheckpointSequence, 0)
    }

    func testWALStatisticsAfterRecovery() async throws {
        let walDir = tempDirectory.appendingPathComponent("wal")
        let config = makeConfigWithWAL(walDir: walDir)

        // Create index, insert, and checkpoint
        let index1 = try await AcceleratedVectorIndex(configuration: config)
        for _ in 0..<5 {
            _ = try await index1.insert(randomVector(128))
        }
        _ = try await index1.checkpoint()

        let statsBefore = await index1.walStatistics()
        XCTAssertNotNil(statsBefore)

        // Recover into new index
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: IndexConfiguration.flat(dimension: 128, capacity: 1000),
            walDirectory: walDir
        )

        let statsAfter = await index2.walStatistics()
        XCTAssertNotNil(statsAfter)
        // Should have same sequence number
        XCTAssertEqual(statsAfter!.currentSequence, statsBefore!.currentSequence)
    }
}
