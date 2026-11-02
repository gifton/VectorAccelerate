import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class BufferPoolCacheAccountingTests: XCTestCase {
    private func makePool(limit: Int) throws -> BufferPool {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        return BufferPool(device: try MetalDevice(), maxTotalMemory: limit)
    }

    // Dropping cache storage must restore budget, without resetting cumulative statistics.
    func testClearingCachedStorageRestoresAllocationBudget() async throws {
        let pool = try makePool(limit: 1024)
        let old = try await pool.getBuffer(size: 1024)
        old.returnToPool()
        let cached = await pool.getStatistics()
        XCTAssertEqual(cached.currentMemoryUsage, 1024)
        await pool.clearCache()
        let cleared = await pool.getStatistics()
        XCTAssertEqual(cleared.currentMemoryUsage, 0)
        XCTAssertEqual(cleared.totalBuffers, 0)
        XCTAssertEqual(cleared.allocationCount, 1)
        let next = try await pool.getBuffer(size: 1024)
        XCTAssertNotEqual(ObjectIdentifier(next.buffer), ObjectIdentifier(old.buffer))
        let allocated = await pool.getStatistics()
        XCTAssertEqual(allocated.currentMemoryUsage, 1024)
        XCTAssertEqual(allocated.allocationCount, 2)
    }

    // Clear must drain already-queued explicit/deinit returns, preserving outstanding leases.
    func testQueuedReturnsAreClearedAndLiveLeaseCanReturnLater() async throws {
        for explicit in [false, true] {
            let pool = try makePool(limit: 8192)
            let live = try await pool.getBuffer(with: [UInt32(12345)])
            var returned: BufferToken? = try await pool.getBuffer(size: 4096)
            if explicit { returned!.returnToPool() }
            returned = nil
            // No statistics/acquisition here: the return must still be queued at clear.
            await pool.clearCache()
            let stats = await pool.getStatistics()
            XCTAssertEqual(stats.currentMemoryUsage, 1024)
            XCTAssertEqual(stats.totalBuffers, 1)
            XCTAssertEqual(stats.availableBuffers, 0)
            XCTAssertEqual(stats.allocationCount, 2)
            XCTAssertEqual(live.copyData(as: UInt32.self), [12345])
            let inUse = await pool.isBufferInUse(live.buffer)
            XCTAssertTrue(inUse)
            live.returnToPool()
            let reused = try await pool.getBuffer(size: 1024)
            XCTAssertEqual(ObjectIdentifier(reused.buffer), ObjectIdentifier(live.buffer))
            let reusedStats = await pool.getStatistics()
            XCTAssertEqual(reusedStats.currentMemoryUsage, 1024)
            XCTAssertEqual(reusedStats.allocationCount, 2)
            XCTAssertEqual(reusedStats.hitCount, 1)
        }
    }

    // Sum the actual cached buckets only; repeated clearing must not subtract live storage.
    func testMultiBucketClearIsIdempotentAndPreservesLiveAccounting() async throws {
        let pool = try makePool(limit: 32768)
        await pool.preallocateCommonSizes(buffersPerSize: 1, maxSizes: 2)
        let live = try await pool.getBuffer(size: 1024)
        let before = await pool.getStatistics()
        XCTAssertEqual(before.currentMemoryUsage, 21504) // 4096 + 16384 + 1024
        for _ in 0..<3 {
            await pool.clearCache()
            let stats = await pool.getStatistics()
            XCTAssertEqual(stats.currentMemoryUsage, 1024)
            XCTAssertEqual(stats.availableBuffers, 0)
            XCTAssertEqual(stats.totalBuffers, 1)
            XCTAssertEqual(stats.allocationCount, 3)
            XCTAssertEqual(stats.hitCount, before.hitCount)
            XCTAssertEqual(stats.missCount, before.missCount)
        }
        let inUse = await pool.isBufferInUse(live.buffer)
        XCTAssertTrue(inUse)
    }

    // The BufferProvider clear spelling must share accounting while retaining handle storage.
    func testProviderClearPreservesHandleUntilRelease() async throws {
        let pool = try makePool(limit: 8192)
        let handle = try await pool.acquire(size: 1024)
        handle.pointer.storeBytes(of: UInt32(12345), as: UInt32.self)
        let cached = try await pool.getBuffer(size: 4096)
        cached.returnToPool()
        await pool.clear()
        let liveStats = await pool.getStatistics()
        XCTAssertEqual(liveStats.currentMemoryUsage, 1024)
        XCTAssertEqual(liveStats.availableBuffers, 0)
        XCTAssertEqual(liveStats.totalBuffers, 1)
        XCTAssertEqual(handle.pointer.load(as: UInt32.self), 12345)
        await pool.release(handle)
        await pool.clear()
        let empty = await pool.getStatistics()
        XCTAssertEqual(empty.currentMemoryUsage, 0)
        XCTAssertEqual(empty.totalBuffers, 0)
        XCTAssertEqual(empty.allocationCount, 2)
    }

    // Retired-generation returns must neither reappear nor reduce the current budget twice.
    func testClearAfterResetExcludesRetiredLeases() async throws {
        let pool = try makePool(limit: 8192)
        let retired = try await pool.getBuffer(size: 1024)
        await pool.reset()
        let cached = try await pool.getBuffer(size: 4096)
        cached.returnToPool()
        await pool.clearCache()
        retired.returnToPool()
        await pool.clearCache()
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.currentMemoryUsage, 0)
        XCTAssertEqual(stats.totalBuffers, 0)
        XCTAssertEqual(stats.allocationCount, 1)
        let next = try await pool.getBuffer(size: 4096)
        let allocated = await pool.getStatistics()
        XCTAssertEqual(next.size, 4096)
        XCTAssertEqual(allocated.currentMemoryUsage, 4096)
        XCTAssertEqual(allocated.allocationCount, 2)
    }

    // Draining may itself discard returns beyond cache capacity; subtract each buffer once.
    func testClearDoesNotDoubleSubtractReturnsDiscardedDuringDrain() async throws {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        let pool = BufferPool(device: try MetalDevice(), maxBuffersPerBucket: 1, maxTotalMemory: 2048)
        let first = try await pool.getBuffer(size: 1024)
        let second = try await pool.getBuffer(size: 1024)
        first.returnToPool()
        second.returnToPool()
        await pool.clearCache()
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.currentMemoryUsage, 0)
        XCTAssertEqual(stats.availableBuffers, 0)
        XCTAssertEqual(stats.totalBuffers, 0)
        XCTAssertEqual(stats.allocationCount, 2)
        let next = try await pool.getBuffer(size: 1024)
        let allocated = await pool.getStatistics()
        XCTAssertEqual(next.size, 1024)
        XCTAssertEqual(allocated.currentMemoryUsage, 1024)
    }

}
