import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

final class BufferAllocationBoundsTests: XCTestCase {
    private let maximum = 64 * 1024 * 1024

    private let bucketCases: [(requested: Int, expected: Int)] = [
        (0, 1024), (1, 1024), (1023, 1024), (1024, 1024), (1025, 4096),
        (4095, 4096), (4096, 4096), (4097, 16384),
        (16383, 16384), (16384, 16384), (16385, 65536),
        (65535, 65536), (65536, 65536), (65537, 262144),
        (262143, 262144), (262144, 262144), (262145, 1048576),
        (1048575, 1048576), (1048576, 1048576), (1048577, 4194304),
        (4194303, 4194304), (4194304, 4194304), (4194305, 16777216),
        (16777215, 16777216), (16777216, 16777216), (16777217, 67108864),
        (67108863, 67108864), (67108864, 67108864),
    ]

    private func deviceAndPool() throws -> (MetalDevice, BufferPool) {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        let device = try MetalDevice()
        return (device, BufferPool(device: device, maxTotalMemory: 2 * maximum))
    }

    private func assertInvalidSize(
        _ operation: () async throws -> Void,
        requested: Int,
        expectsBoundsDetails: Bool = true,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            try await operation()
            XCTFail("request \(requested) must throw", file: file, line: line)
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .invalidData, file: file, line: line)
            if expectsBoundsDetails {
                XCTAssertEqual(error.context.additionalInfo["requested_size"], String(requested), file: file, line: line)
                XCTAssertEqual(error.context.additionalInfo["maximum_size"], String(maximum), file: file, line: line)
            }
        } catch {
            XCTFail("expected VectorError.invalidData, got \(error)", file: file, line: line)
        }
    }

    /// The allocation entry point must distinguish a capped bucket lookup from a
    /// supported request and never return less storage than requested.
    func testPoolBoundariesNeverReturnUndersizedStorage() async throws {
        let (_, pool) = try deviceAndPool()
        for (requested, expected) in bucketCases {
            let token = try await pool.getBuffer(size: requested)
            XCTAssertGreaterThanOrEqual(token.buffer.length, requested, "request \(requested)")
            XCTAssertEqual(token.buffer.length, expected, "request \(requested)")
            XCTAssertGreaterThan(token.buffer.length, 0, "zero remains a minimum-bucket lease")
            token.returnToPool()
        }
    }

    /// Invalid requests must fail before allocation or lease registration. Statistics
    /// are sampled on an isolated pool so draining unrelated pending returns cannot mask it.
    func testRejectedPoolRequestsLeaveBookkeepingUnchanged() async throws {
        let (_, pool) = try deviceAndPool()
        let baseline = await pool.getStatistics()
        await assertInvalidSize(
            { _ = try await pool.getBuffer(size: -1) }, requested: -1, expectsBoundsDetails: false
        )
        for requested in [maximum + 1, Int.max] {
            await assertInvalidSize({ _ = try await pool.getBuffer(size: requested) }, requested: requested)
        }
        let final = await pool.getStatistics()
        XCTAssertEqual(final.allocationCount, baseline.allocationCount)
        XCTAssertEqual(final.currentMemoryUsage, baseline.currentMemoryUsage)
        XCTAssertEqual(final.totalBuffers, baseline.totalBuffers)
        XCTAssertEqual(final.availableBuffers, baseline.availableBuffers)
        XCTAssertEqual(final.hitCount, baseline.hitCount)
        XCTAssertEqual(final.missCount, baseline.missCount)
    }

    /// Compatibility handles share the guarded pool entry point; rejection must not
    /// retain a handle or affect the following valid acquisition.
    func testCompatibilityAcquireRejectsBeforeRegisteringHandle() async throws {
        let (_, pool) = try deviceAndPool()
        let baseline = await pool.getStatistics()
        await assertInvalidSize({ _ = try await pool.acquire(size: maximum + 1) }, requested: maximum + 1)
        let afterReject = await pool.getStatistics()
        XCTAssertEqual(afterReject.allocationCount, baseline.allocationCount)
        XCTAssertEqual(afterReject.totalBuffers, baseline.totalBuffers)

        let handle = try await pool.acquire(size: 1024)
        XCTAssertGreaterThanOrEqual(handle.size, 1024)
        await pool.release(handle)

        await assertInvalidSize(
            { _ = try await pool.acquireBuffer(byteSize: maximum + 1) }, requested: maximum + 1
        )
        let metalBuffer = try await pool.acquireBuffer(byteSize: 1024)
        XCTAssertGreaterThanOrEqual(metalBuffer.buffer.length, 1024)
        await pool.releaseBuffer(metalBuffer)

        let reused = try await pool.acquire(size: 1024)
        let stats = await pool.getStatistics()
        XCTAssertGreaterThan(stats.hitCount, 0)
        await pool.release(reused)
    }

    /// Bucket selection remains a capped lookup for compatibility, while allocation
    /// rejects malformed or unsupported original requests before calling Metal.
    func testBucketedFactoryRejectsUnsupportedOriginalSizes() throws {
        guard let rawDevice = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let factory = MetalBufferFactory(device: rawDevice)
        XCTAssertEqual(MetalBufferFactory.selectBucketSize(for: maximum + 1), maximum)
        XCTAssertEqual(MetalBufferFactory.selectBucketSize(for: Int.max), maximum)
        for requested in [-1, maximum + 1, Int.max] {
            XCTAssertNil(factory.createBucketedBuffer(size: requested), "request \(requested)")
        }
        for (requested, expected) in bucketCases {
            let buffer = try XCTUnwrap(factory.createBucketedBuffer(size: requested), "request \(requested)")
            XCTAssertGreaterThanOrEqual(buffer.length, requested)
            XCTAssertEqual(buffer.length, expected, "request \(requested)")
            XCTAssertGreaterThan(buffer.length, 0)
        }
    }

    func testTypedCountsRejectNegativeAndOverflowBeforeAllocation() async throws {
        let (_, pool) = try deviceAndPool()
        let baseline = await pool.getStatistics()
        for count in [-1, Int.max] {
            do {
                _ = try await pool.getBuffer(for: Float.self, count: count)
                XCTFail("invalid typed count \(count) must throw")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            }
        }
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.allocationCount, baseline.allocationCount)
        XCTAssertEqual(stats.totalBuffers, baseline.totalBuffers)
        let zero = try await pool.getBuffer(for: Float.self, count: 0)
        XCTAssertGreaterThan(zero.buffer.length, 0)
    }

    func testAlignmentRejectsInvalidAndOverflowingRequests() async throws {
        let (_, pool) = try deviceAndPool()
        let baseline = await pool.getStatistics()
        for alignment in [0, -1, 3, Int.max] {
            do {
                _ = try await pool.getAlignedBuffer(size: 17, alignment: alignment)
                XCTFail("invalid alignment \(alignment) must throw")
            } catch let error as VectorError {
                XCTAssertEqual(error.kind, .invalidData)
            }
        }
        do {
            _ = try await pool.getAlignedBuffer(size: Int.max, alignment: 16)
            XCTFail("overflowing aligned size must throw")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .invalidData)
        }
        do {
            _ = try await pool.getAlignedBuffer(size: -1, alignment: 16)
            XCTFail("negative aligned size must throw")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .invalidData)
        }
        await assertInvalidSize(
            { _ = try await pool.getAlignedBuffer(size: 1, alignment: 128 * 1024 * 1024) },
            requested: 128 * 1024 * 1024
        )
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.allocationCount, baseline.allocationCount)

        let zero = try await pool.getAlignedBuffer(size: 0, alignment: 16)
        XCTAssertGreaterThan(zero.buffer.length, 0)
        let rounded = try await pool.getAlignedBuffer(size: 1025, alignment: 8192)
        XCTAssertEqual(rounded.size, 16384)
        XCTAssertGreaterThanOrEqual(rounded.buffer.length, 8192)
    }

    func testEmptyAndOrdinaryTypedUploadsRoundTrip() async throws {
        let (_, pool) = try deviceAndPool()
        let emptyFor = try await pool.getBuffer(for: [UInt32]())
        XCTAssertEqual(emptyFor.copyData(as: UInt32.self), [])
        emptyFor.returnToPool()
        let emptyWith = try await pool.getBuffer(with: [UInt32]())
        XCTAssertEqual(emptyWith.copyData(as: UInt32.self), [])
        emptyWith.returnToPool()

        let values: [UInt32] = [0, 1, 7, UInt32.max]
        let initialized = try await pool.getBuffer(with: values)
        XCTAssertEqual(initialized.copyData(as: UInt32.self), values)
        let aligned = try await pool.getAlignedBuffer(size: 17, alignment: 16)
        XCTAssertEqual(aligned.size, 1024)
        XCTAssertGreaterThanOrEqual(aligned.buffer.length, 32)
    }

    func testInitializedJustOverCapRejectsThroughBothConveniences() async throws {
        let (_, pool) = try deviceAndPool()
        let baseline = await pool.getStatistics()
        let bytes = [UInt8](repeating: 0xA5, count: maximum + 1)
        await assertInvalidSize({ _ = try await pool.getBuffer(for: bytes) }, requested: maximum + 1)
        await assertInvalidSize({ _ = try await pool.getBuffer(with: bytes) }, requested: maximum + 1)
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.allocationCount, baseline.allocationCount)
        XCTAssertEqual(stats.totalBuffers, baseline.totalBuffers)
    }

    func testSmallBudgetRejectsThenReusesReturnedBuffer() async throws {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        let pool = BufferPool(device: try MetalDevice(), maxTotalMemory: 1024)
        let first = try await pool.getBuffer(size: 1000)
        let firstBuffer = first.buffer
        let allocatedStats = await pool.getStatistics()
        do {
            _ = try await pool.getBuffer(size: 1000)
            XCTFail("second live bucket must exceed the deterministic budget")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .resourceExhausted)
        }
        let rejectedStats = await pool.getStatistics()
        XCTAssertEqual(rejectedStats.allocationCount, 1)
        first.returnToPool()
        let reused = try await pool.getBuffer(size: 1000)
        let reusedStats = await pool.getStatistics()
        XCTAssertEqual(ObjectIdentifier(reused.buffer), ObjectIdentifier(firstBuffer))
        XCTAssertEqual(reusedStats.hitCount, 1)
        XCTAssertEqual(reusedStats.allocationCount, allocatedStats.allocationCount)
        XCTAssertEqual(reusedStats.currentMemoryUsage, allocatedStats.currentMemoryUsage)
        reused.returnToPool()
    }

    func testBudgetCleanupFreesCachedBucketBeforeLargerAllocation() async throws {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        let pool = BufferPool(device: try MetalDevice(), maxTotalMemory: 4096)
        let small = try await pool.getBuffer(size: 1024)
        small.returnToPool()
        _ = await pool.getStatistics()
        let cachedStats = await pool.getStatistics()
        XCTAssertEqual(cachedStats.currentMemoryUsage, 1024)
        let large = try await pool.getBuffer(size: 4096)
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.currentMemoryUsage, 4096)
        XCTAssertEqual(stats.allocationCount, 2)
        large.returnToPool()
    }

    func testPreallocationInvalidCountsNoOpAndValidBudgetIsBounded() async throws {
        guard MetalDevice.isAvailable else { throw XCTSkip("Metal unavailable") }
        let pool = BufferPool(device: try MetalDevice(), maxTotalMemory: 8192)
        let baseline = await pool.getStatistics()
        for (count, sizes) in [(-1, 1), (1, -1), (0, Int.max), (Int.max, 1), (3, 1)] {
            await pool.preallocateCommonSizes(buffersPerSize: count, maxSizes: sizes)
        }
        let invalidStats = await pool.getStatistics()
        XCTAssertEqual(invalidStats.allocationCount, baseline.allocationCount)
        XCTAssertEqual(invalidStats.currentMemoryUsage, baseline.currentMemoryUsage)
        await pool.preallocateCommonSizes(buffersPerSize: 2, maxSizes: 1)
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.allocationCount - baseline.allocationCount, 2)
        XCTAssertEqual(stats.currentMemoryUsage - baseline.currentMemoryUsage, 8192)
        XCTAssertEqual(stats.availableBuffers - baseline.availableBuffers, 2)
    }
}
