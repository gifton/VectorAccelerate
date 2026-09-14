import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class BufferPoolLifecycleTests: XCTestCase {
    private final class WeakResources {
        weak var pool: BufferPool?
        weak var buffer: (any MTLBuffer)?

        init(pool: BufferPool, buffer: any MTLBuffer) {
            self.pool = pool
            self.buffer = buffer
        }
    }

    private func makePool() throws -> BufferPool {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        return BufferPool(device: try MetalDevice(), maxTotalMemory: 4096)
    }

    private func abandonReturnedLease(explicitReturn: Bool) async throws -> WeakResources {
        let pool = try makePool()
        let token = try await pool.getBuffer(size: 1024)
        let references = WeakResources(pool: pool, buffer: token.buffer)
        if explicitReturn { token.returnToPool() }
        return references
    }

    // A dead pool must not leave its returned buffer owned by a global queue.
    func testPoolDestructionReleasesUndrainedReturn() async throws {
        for explicitReturn in [false, true] {
            let references = try await abandonReturnedLease(explicitReturn: explicitReturn)
            XCTAssertNil(references.pool)
            XCTAssertNil(references.buffer)
        }
    }

    private func abandonPoolWithLiveToken() async throws -> (BufferToken, WeakResources) {
        let pool = try makePool()
        let token = try await pool.getBuffer(with: [UInt32(12345)])
        return (token, WeakResources(pool: pool, buffer: token.buffer))
    }

    // The token owns its storage, not the pool; returning after pool destruction is safe.
    func testLiveTokenSurvivesPoolDestructionWithoutRetainingPool() async throws {
        let (token, references) = try await abandonPoolWithLiveToken()
        XCTAssertNil(references.pool)
        XCTAssertNotNil(references.buffer)
        XCTAssertEqual(token.copyData(as: UInt32.self), [12345])
        token.returnToPool()
        token.returnToPool()
    }

    private func abandonCompatibilityHandle() async throws -> WeakResources {
        let pool = try makePool()
        let buffer = try await pool.acquireBuffer(byteSize: 1024)
        return WeakResources(pool: pool, buffer: buffer.buffer)
    }

    // Pool -> handle token -> pool must not form a retain cycle.
    func testUnreleasedCompatibilityHandleDoesNotKeepAbandonedPoolAlive() async throws {
        let references = try await abandonCompatibilityHandle()
        XCTAssertNil(references.pool)
        XCTAssertNil(references.buffer)
    }

    // Reset retires outstanding leases; their later returns cannot reappear unaccounted.
    func testLateReturnAfterResetCannotEnterNewCache() async throws {
        for explicitReturn in [false, true] {
            let pool = try makePool()
            var old: BufferToken? = try await pool.getBuffer(with: [UInt32(12345)])
            let oldBuffer = old!.buffer
            await pool.reset()
            XCTAssertEqual(old!.copyData(as: UInt32.self), [12345])
            if explicitReturn { old!.returnToPool() }
            old = nil
            let empty = await pool.getStatistics()
            XCTAssertEqual(empty.totalBuffers, 0)
            XCTAssertEqual(empty.currentMemoryUsage, 0)
            let current = try await pool.getBuffer(size: 1024)
            XCTAssertNotEqual(ObjectIdentifier(current.buffer), ObjectIdentifier(oldBuffer))
            let allocated = await pool.getStatistics()
            XCTAssertEqual(allocated.allocationCount, 1)
            XCTAssertEqual(allocated.currentMemoryUsage, 1024)
            current.returnToPool()
            let reused = try await pool.getBuffer(size: 1024)
            XCTAssertEqual(ObjectIdentifier(reused.buffer), ObjectIdentifier(current.buffer))
        }
    }

    // Compatibility leases remain releasable after reset, without changing new accounting.
    func testCompatibilityReleaseAfterResetCannotContaminateNewLease() async throws {
        let pool = try makePool()
        let handle = try await pool.acquire(size: 1024)
        let metalBuffer = try await pool.acquireBuffer(byteSize: 1024)
        handle.pointer.storeBytes(of: UInt32(12345), as: UInt32.self)
        await pool.reset()
        XCTAssertEqual(handle.pointer.load(as: UInt32.self), 12345)
        let current = try await pool.getBuffer(size: 1024)
        await pool.release(handle)
        await pool.releaseBuffer(metalBuffer)
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.totalBuffers, 1)
        XCTAssertEqual(stats.availableBuffers, 0)
        XCTAssertEqual(stats.currentMemoryUsage, 1024)
        let currentIsInUse = await pool.isBufferInUse(current.buffer)
        XCTAssertTrue(currentIsInUse)
    }

    // A foreign/duplicate return must not create available storage or subtract accounting.
    func testUntrackedAndDuplicateReturnsCannotChangePoolAccounting() async throws {
        let pool = try makePool()
        let other = try makePool()
        let foreign = try await other.getBuffer(size: 1024)
        await pool.returnBuffer(foreign.buffer, size: 1024)
        let empty = await pool.getStatistics()
        XCTAssertEqual(empty.totalBuffers, 0)
        XCTAssertEqual(empty.currentMemoryUsage, 0)
        let token = try await pool.getBuffer(size: 1024)
        token.returnToPool()
        _ = await pool.getStatistics()
        await pool.returnBuffer(token.buffer, size: 1024)
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.availableBuffers, 1)
        XCTAssertEqual(stats.totalBuffers, 1)
        XCTAssertEqual(stats.currentMemoryUsage, 1024)
    }

    // Structured concurrent callers may all return one token, but only one return is queued.
    func testConcurrentExplicitReturnRemainsExactlyOnceAndPoolLocal() async throws {
        let pool = try makePool()
        let other = try makePool()
        let token = try await pool.getBuffer(size: 1024)
        let otherToken = try await other.getBuffer(size: 1024)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<64 { group.addTask { token.returnToPool() } }
        }
        let stats = await pool.getStatistics()
        let otherStats = await other.getStatistics()
        XCTAssertEqual(stats.availableBuffers, 1)
        XCTAssertEqual(stats.currentMemoryUsage, 1024)
        XCTAssertEqual(otherStats.availableBuffers, 0)
        XCTAssertEqual(otherStats.currentMemoryUsage, 1024)
        let reused = try await pool.getBuffer(size: 1024)
        XCTAssertEqual(ObjectIdentifier(reused.buffer), ObjectIdentifier(token.buffer))
        XCTAssertNotEqual(ObjectIdentifier(reused.buffer), ObjectIdentifier(otherToken.buffer))
    }

    func testRacingResetAndReturnsLeaveNewGenerationEmpty() async throws {
        let pool = try makePool()
        for _ in 0..<32 {
            let token = try await pool.getBuffer(size: 1024)
            await withTaskGroup(of: Void.self) { group in
                group.addTask { await pool.reset() }
                for _ in 0..<16 { group.addTask { token.returnToPool() } }
            }
            let stats = await pool.getStatistics()
            XCTAssertEqual(stats.totalBuffers, 0)
            XCTAssertEqual(stats.currentMemoryUsage, 0)
            XCTAssertEqual(stats.allocationCount, 0)
        }
    }

    // GPU completion owns the token independently of pool lifetime.
    func testCompletionAnchoredTokenWorksAfterPoolDestruction() async throws {
        var pool: BufferPool? = try makePool()
        let device = try XCTUnwrap(pool).bufferFactory.device
        var token: BufferToken? = try await pool!.getBuffer(with: [UInt32(12345)])
        let references = WeakResources(pool: pool!, buffer: token!.buffer)
        let output = try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared))
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeBlitCommandEncoder())
        encoder.copy(from: token!.buffer, sourceOffset: 0, to: output, destinationOffset: 0, size: 4)
        encoder.endEncoding()
        token!.keepAlive(until: command)
        token = nil
        pool = nil
        XCTAssertNil(references.pool)
        XCTAssertNotNil(references.buffer)
        command.commit()
        await command.completed()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
        XCTAssertEqual(output.contents().load(as: UInt32.self), 12345)
    }

}
