import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class ArgumentTableLifecycleTests: XCTestCase {
    private final class WeakResources {
        weak var pool: ArgumentTablePool?
        weak var table: (any ArgumentTable)?
        weak var buffer: (any MTLBuffer)?

        init(pool: ArgumentTablePool, table: any ArgumentTable, buffer: any MTLBuffer) {
            self.pool = pool
            self.table = table
            self.buffer = buffer
        }
    }

    private func makeDevice() throws -> any MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        return device
    }

    private func abandonedReturn(explicit: Bool) async throws -> WeakResources {
        let device = try makeDevice()
        let pool = ArgumentTablePool(device: device)
        let token = try await pool.acquireToken()
        let buffer = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
        token.table.setBuffer(buffer, offset: 0, index: 0)
        let refs = WeakResources(pool: pool, table: token.table, buffer: buffer)
        if explicit { token.release() }
        return refs
    }

    // A global orphan queue must not retain a returned table or its bound buffers.
    func testPoolDestructionReleasesQueuedTablesAndBindings() async throws {
        for explicit in [false, true] {
            let refs = try await abandonedReturn(explicit: explicit)
            XCTAssertNil(refs.pool)
            XCTAssertNil(refs.table)
            XCTAssertNil(refs.buffer)
        }
    }

    private func liveTokenWithoutPool(custom: Bool) async throws -> (ArgumentTableToken, WeakResources) {
        let device = try makeDevice()
        let pool = ArgumentTablePool(device: device)
        let token = try await custom
            ? pool.acquireToken(descriptor: ArgumentTableDescriptor(maxBufferBindCount: 3))
            : pool.acquireToken()
        let buffer = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
        token.table.setBuffer(buffer, offset: 0, index: 0)
        return (token, WeakResources(pool: pool, table: token.table, buffer: buffer))
    }

    // Both token acquisition spellings own the table independently of the pool.
    func testLiveTokenKeepsBindingsButDoesNotRetainPool() async throws {
        for custom in [false, true] {
            let (token, refs) = try await liveTokenWithoutPool(custom: custom)
            XCTAssertNil(refs.pool)
            XCTAssertNotNil(refs.table)
            XCTAssertNotNil(refs.buffer)
            XCTAssertEqual(token.table.maxBufferBindCount, custom ? 3 : 16)
            token.table.reset()
            XCTAssertNil(refs.buffer)
            token.release()
            token.release()
        }
    }

    // Return must be synchronous and clear binding ownership before reuse.
    func testExplicitAndDeinitReturnsReuseImmediatelyAndClearBindings() async throws {
        let device = try makeDevice()
        for explicit in [false, true] {
            let pool = ArgumentTablePool(device: device, maxTables: 1)
            var token: ArgumentTableToken? = try await pool.acquireToken()
            let table = token!.table
            weak var bound: (any MTLBuffer)?
            do {
                let buffer = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
                bound = buffer
                table.setBuffer(buffer, offset: 0, index: 0)
            }
            XCTAssertNotNil(bound)
            if explicit { token!.release() }
            token = nil
            let next = try await pool.acquireToken()
            XCTAssertEqual(ObjectIdentifier(next.table), ObjectIdentifier(table))
            XCTAssertNil(bound)
            let stats = await pool.getStatistics()
            XCTAssertEqual(stats.totalTables, 1)
            XCTAssertEqual(stats.inUseTables, 1)
            XCTAssertEqual(stats.releaseCount, 1)
        }
    }

    // Concurrent explicit releases enqueue once and cannot affect another pool.
    func testConcurrentReleaseIsExactlyOnceAndPoolLocal() async throws {
        let device = try makeDevice()
        let pool = ArgumentTablePool(device: device, maxTables: 1)
        let other = ArgumentTablePool(device: device, maxTables: 1)
        let token = try await pool.acquireToken()
        let otherToken = try await other.acquireToken()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<64 { group.addTask { token.release() } }
        }
        let stats = await pool.getStatistics()
        let otherStats = await other.getStatistics()
        XCTAssertEqual(stats.releaseCount, 1)
        XCTAssertEqual(stats.availableTables, 1)
        XCTAssertEqual(otherStats.releaseCount, 0)
        XCTAssertEqual(otherStats.inUseTables, 1)
        let next = try await pool.acquireToken()
        XCTAssertEqual(ObjectIdentifier(next.table), ObjectIdentifier(token.table))
        XCTAssertNotEqual(ObjectIdentifier(next.table), ObjectIdentifier(otherToken.table))
    }

    // Clearing available tables is not a pool reset: outstanding token returns remain valid.
    func testClearAvailablePreservesOutstandingTokenReturn() async throws {
        let pool = ArgumentTablePool(device: try makeDevice(), maxTables: 2)
        let cached = try await pool.acquireToken()
        let live = try await pool.acquireToken()
        cached.release()
        _ = await pool.getStatistics()
        await pool.clearAvailable()
        let cleared = await pool.getStatistics()
        XCTAssertEqual(cleared.availableTables, 0)
        XCTAssertEqual(cleared.inUseTables, 1)
        live.release()
        let next = try await pool.acquireToken()
        XCTAssertEqual(ObjectIdentifier(next.table), ObjectIdentifier(live.table))
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.totalTables, 1)
        XCTAssertEqual(stats.releaseCount, 2)
    }

    // Unknown raw returns remain rejected and must not reset a foreign table's bindings.
    func testForeignReleaseCannotClearBindingsOrChangeStatistics() async throws {
        let device = try makeDevice()
        let owner = ArgumentTablePool(device: device)
        let other = ArgumentTablePool(device: device)
        let token = try await owner.acquireToken()
        weak var bound: (any MTLBuffer)?
        do {
            let buffer = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
            bound = buffer
            token.table.setBuffer(buffer, offset: 0, index: 0)
        }
        await other.release(token.table)
        XCTAssertNotNil(bound)
        let stats = await other.getStatistics()
        XCTAssertEqual(stats.totalTables, 0)
        XCTAssertEqual(stats.releaseCount, 0)
        token.release()
        _ = await owner.getStatistics()
        XCTAssertNil(bound)
    }
}
