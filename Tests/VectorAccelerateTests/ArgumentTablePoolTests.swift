//
//  ArgumentTablePoolTests.swift
//  VectorAccelerateTests
//
//  Tests for ArgumentTablePool actor: acquisition, release, pooling, statistics,
//  descriptor presets, warm-up, and RAII token lifecycle.
//

@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore
import XCTest

final class ArgumentTablePoolTests: XCTestCase {

    private var device: (any MTLDevice)!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        device = MTLCreateSystemDefaultDevice()!
    }

    override func tearDown() async throws {
        device = nil
        try await super.tearDown()
    }

    // MARK: - Basic Acquisition

    /// Acquiring a table from a fresh pool should succeed and return a usable table.
    func test_acquire_returnsTable() async throws {
        let pool = ArgumentTablePool(device: device)

        let table = try await pool.acquire()

        XCTAssertGreaterThan(table.maxBufferBindCount, 0,
                             "Acquired table should have a positive maxBufferBindCount")
    }

    // MARK: - Acquire and Release Cycle

    /// A single acquire-release cycle should be reflected correctly in statistics.
    func test_acquire_andRelease_cycle() async throws {
        let pool = ArgumentTablePool(device: device)

        let table = try await pool.acquire()
        await pool.release(table)

        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.acquisitionCount, 1, "Should record exactly 1 acquisition")
        XCTAssertEqual(stats.releaseCount, 1, "Should record exactly 1 release")
        XCTAssertEqual(stats.inUseTables, 0, "No tables should be in use after release")
        XCTAssertEqual(stats.availableTables, 1, "Released table should be available for reuse")
    }

    // MARK: - Reuse After Release

    /// After releasing a table back to the pool, the next acquire should reuse it,
    /// bringing the available count back to zero.
    func test_acquire_reuseAfterRelease() async throws {
        let pool = ArgumentTablePool(device: device)

        let table1 = try await pool.acquire()
        await pool.release(table1)

        let availableBeforeReacquire = await pool.availableCount
        XCTAssertEqual(availableBeforeReacquire, 1, "Pool should have 1 available table after release")

        _ = try await pool.acquire()

        let availableAfterReacquire = await pool.availableCount
        XCTAssertEqual(availableAfterReacquire, 0,
                       "Available count should be 0 after reuse -- the released table was reused")
    }

    // MARK: - Pool Exhaustion

    /// When the pool reaches maxTables, further acquisitions should throw
    /// VectorError with resourceExhausted kind.
    func test_acquire_exhausted() async throws {
        let pool = ArgumentTablePool(device: device, maxTables: 2)

        // Hold references so deinit doesn't return them to the pool
        let table1 = try await pool.acquire()
        let table2 = try await pool.acquire()

        do {
            _ = try await pool.acquire()
            XCTFail("Third acquire should throw when maxTables is 2")
        } catch let error as VectorError where error.kind == .resourceExhausted {
            // Expected: argumentTablePoolExhausted
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        // Keep tables alive past the assertion
        _ = (table1, table2)
    }

    // MARK: - Release Enables Re-acquire

    /// Releasing a table from an exhausted pool should allow a new acquisition.
    func test_release_enablesReacquire() async throws {
        let pool = ArgumentTablePool(device: device, maxTables: 2)

        let table1 = try await pool.acquire()
        let table2 = try await pool.acquire()

        // Pool is now at capacity
        await pool.release(table1)

        // Should now succeed because one slot is free
        let table3 = try await pool.acquire()
        XCTAssertGreaterThan(table3.maxBufferBindCount, 0,
                             "Acquire after release on exhausted pool should succeed")
        _ = table2 // keep alive
    }

    // MARK: - Double Release No-Op

    /// Releasing the same table twice should not crash and should only count
    /// one release in statistics.
    func test_doubleRelease_noop() async throws {
        let pool = ArgumentTablePool(device: device)

        let table = try await pool.acquire()
        await pool.release(table)
        await pool.release(table)

        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.releaseCount, 1,
                       "Double release should only count as 1 release in statistics")
    }

    // MARK: - Release Unknown Table No-Op

    /// Releasing a table that was NOT acquired from the pool should be a safe no-op
    /// with no effect on pool statistics.
    func test_release_unknownTable_noop() async throws {
        let pool = ArgumentTablePool(device: device)

        // Create a table directly, bypassing the pool
        let manualTable = Metal4ArgumentTable(descriptor: .distance)
        await pool.release(manualTable)

        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.releaseCount, 0,
                       "Releasing an unknown table should not increment release count")
        XCTAssertEqual(stats.totalTables, 0,
                       "Releasing an unknown table should not add it to the pool")
    }

    // MARK: - Warm-Up

    /// warmUp should pre-create the requested number of tables in the available pool.
    func test_warmUp_createsAvailableTables() async throws {
        let pool = ArgumentTablePool(device: device)

        await pool.warmUp(count: 5)

        let available = await pool.availableCount
        XCTAssertEqual(available, 5, "warmUp(count: 5) should create 5 available tables")
    }

    /// warmUp should respect maxTables and never exceed pool capacity.
    func test_warmUp_respectsMaxTables() async throws {
        let pool = ArgumentTablePool(device: device, maxTables: 3)

        await pool.warmUp(count: 10)

        let available = await pool.availableCount
        XCTAssertLessThanOrEqual(available, 3,
                                 "warmUp should not exceed maxTables capacity")
    }

    // MARK: - Batch Operations

    /// acquireMultiple should return the exact number of tables requested.
    func test_acquireMultiple_count() async throws {
        let pool = ArgumentTablePool(device: device)

        let tables = try await pool.acquireMultiple(count: 3)

        XCTAssertEqual(tables.count, 3, "acquireMultiple(count: 3) should return exactly 3 tables")

        let inUse = await pool.inUseCount
        XCTAssertEqual(inUse, 3, "All 3 acquired tables should be tracked as in-use")
    }

    /// releaseMultiple should return all tables to the pool.
    func test_releaseMultiple_returnsAll() async throws {
        let pool = ArgumentTablePool(device: device)

        let tables = try await pool.acquireMultiple(count: 3)
        await pool.releaseMultiple(tables)

        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.inUseTables, 0, "All tables should be returned after releaseMultiple")
        XCTAssertEqual(stats.availableTables, 3,
                       "All 3 tables should be available after releaseMultiple")
        XCTAssertEqual(stats.releaseCount, 3, "releaseMultiple should record 3 releases")
    }

    // MARK: - Clear Available

    /// clearAvailable should remove all available (not in-use) tables from the pool.
    func test_clearAvailable_clearsPool() async throws {
        let pool = ArgumentTablePool(device: device)

        await pool.warmUp(count: 5)
        let availableBefore = await pool.availableCount
        XCTAssertEqual(availableBefore, 5, "Precondition: 5 tables warmed up")

        await pool.clearAvailable()

        let availableAfter = await pool.availableCount
        XCTAssertEqual(availableAfter, 0, "clearAvailable should remove all available tables")
    }

    // MARK: - Statistics: Peak In-Use

    /// peakInUse should track the high-water mark of simultaneously in-use tables.
    func test_getStatistics_peakInUse() async throws {
        let pool = ArgumentTablePool(device: device, maxTables: 8)

        // Acquire 3 tables
        let tables1 = try await pool.acquireMultiple(count: 3)
        // Release 1, bringing in-use to 2
        await pool.release(tables1[0])
        // Acquire 2 more, bringing in-use to 4 (peak)
        let tables2 = try await pool.acquireMultiple(count: 2)

        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.peakInUse, 4,
                       "Peak in-use should be 4 (3 acquired, 1 released, 2 more acquired)")
        XCTAssertEqual(stats.inUseTables, 4,
                       "Currently 4 tables should be in use")

        // Clean up
        await pool.releaseMultiple(Array(tables1.dropFirst()))
        await pool.releaseMultiple(tables2)
    }

    // MARK: - Descriptor Presets

    /// Verify that descriptor presets have the documented maxBufferBindCount values.
    func test_descriptorPresets_bufferCounts() {
        XCTAssertEqual(ArgumentTableDescriptor.distance.maxBufferBindCount, 8,
                       ".distance should have maxBufferBindCount of 8")
        XCTAssertEqual(ArgumentTableDescriptor.batch.maxBufferBindCount, 16,
                       ".batch should have maxBufferBindCount of 16")
        XCTAssertEqual(ArgumentTableDescriptor.matrix.maxBufferBindCount, 8,
                       ".matrix should have maxBufferBindCount of 8")
        XCTAssertEqual(ArgumentTableDescriptor.quantization.maxBufferBindCount, 12,
                       ".quantization should have maxBufferBindCount of 12")
    }

    // MARK: - Table Operations

    /// Acquired table should support setBuffer and reset without crashing.
    /// After reset, binding the same index again should succeed.
    func test_table_setBuffer_andReset() async throws {
        let pool = ArgumentTablePool(device: device)
        let table = try await pool.acquire()

        // Create a small buffer for testing
        guard let buffer = device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw XCTSkip("Failed to create MTLBuffer for test")
        }

        // Bind at index 0
        table.setBuffer(buffer, offset: 0, index: 0)

        // Reset clears all bindings
        table.reset()

        // Bind at the same index again after reset -- should not crash
        table.setBuffer(buffer, offset: 0, index: 0)

        await pool.release(table)
    }

    // MARK: - Token Auto-Release

    /// ArgumentTableToken should automatically release its table back to the pool
    /// when the token is deallocated. The deinit enqueues into PendingTableReturns,
    /// which is drained by getStatistics().
    func test_acquireToken_autoRelease() async throws {
        let pool = ArgumentTablePool(device: device)

        // Acquire a token in a confined scope so it deallocates
        do {
            let token = try await pool.acquireToken()
            XCTAssertGreaterThan(token.table.maxBufferBindCount, 0,
                                 "Token should contain a valid table")
            // token goes out of scope here, triggering deinit -> PendingTableReturns.enqueue
        }

        // getStatistics drains pending returns automatically
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.inUseTables, 0,
                       "Table should be auto-released when token is deallocated")
        XCTAssertEqual(stats.availableTables, 1,
                       "Auto-released table should be back in the available pool")
    }
}
