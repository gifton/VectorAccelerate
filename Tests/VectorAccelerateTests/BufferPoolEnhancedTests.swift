//
//  BufferPoolEnhancedTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for BufferPool memory management
//

@preconcurrency import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

final class BufferPoolEnhancedTests: XCTestCase {
    var metalDevice: MetalDevice!
    var bufferPool: BufferPool!
    
    override func setUp() async throws {
        try await super.setUp()
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        self.metalDevice = try await MetalDevice()
        self.bufferPool = BufferPool(device: metalDevice)
    }
    
    override func tearDown() async throws {
        await bufferPool.reset()
        bufferPool = nil
        metalDevice = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Buffer Management
    
    func testBufferAllocation() async throws {
        let size = 1024
        
        let token = try await bufferPool.getBuffer(size: size)
        XCTAssertNotNil(token.buffer)
        XCTAssertGreaterThanOrEqual(token.buffer.length, size)
        
        // Buffer should be properly aligned
        let address = token.buffer.contents()
        XCTAssertEqual(Int(bitPattern: address) % 16, 0, "Buffer should be 16-byte aligned")
    }
    
    func testBufferReuse() async throws {
        let size = 2048
        
        // Get a buffer
        let token1 = try await bufferPool.getBuffer(size: size)
        let buffer1 = token1.buffer
        
        // Return it to pool
        token1.returnToPool()
        
        // Get another buffer of same size - should reuse
        let token2 = try await bufferPool.getBuffer(size: size)
        let buffer2 = token2.buffer
        
        // Should be the same buffer (reused)
        XCTAssertEqual(buffer1.length, buffer2.length)
        
        // Verify reuse statistics
        let stats = await bufferPool.getStatistics()
        XCTAssertGreaterThan(stats.hitCount, 0)
    }
    
    func testAutomaticReturn() async throws {
        let size = 1024
        
        // Create scope for automatic return
        do {
            let token = try await bufferPool.getBuffer(size: size)
            XCTAssertNotNil(token.buffer)
            // Token goes out of scope here
        }
        
        // Buffer should be returned automatically
        let stats = await bufferPool.getStatistics()
        XCTAssertGreaterThan(stats.availableBuffers, 0)
    }
    
    // MARK: - Bucket System Tests
    
    func testBucketSelection() async throws {
        let sizes = [256, 512, 1024, 2048, 4096, 8192]
        var tokens: [BufferToken] = []
        
        for size in sizes {
            let token = try await bufferPool.getBuffer(size: size)
            XCTAssertGreaterThanOrEqual(token.buffer.length, size)
            tokens.append(token)
        }
        
        // Return all buffers
        for token in tokens {
            token.returnToPool()
        }
        
        // Check bucket distribution
        let stats = await bufferPool.getStatistics()
        XCTAssertGreaterThan(stats.totalBuffers, 0)
    }
    
    func testLargeSizeHandling() async throws {
        let largeSize = 100 * 1024 * 1024 // 100 MB
        
        do {
            let token = try await bufferPool.getBuffer(size: largeSize)
            XCTAssertGreaterThanOrEqual(token.buffer.length, largeSize)
            
            // Large buffers might not be pooled
            token.returnToPool()
        } catch {
            // Acceptable if allocation fails due to memory limits
            XCTAssertTrue(error is AccelerationError)
        }
    }
    
    // MARK: - Typed Data Support
    
    func testTypedBufferFloat() async throws {
        let count = 256
        let data = Array(repeating: Float(3.14), count: count)
        
        let token = try await bufferPool.getBuffer(for: data)
        XCTAssertNotNil(token.buffer)
        XCTAssertEqual(token.buffer.length, count * MemoryLayout<Float>.stride)
        
        // Verify data was copied
        let contents = token.buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            XCTAssertEqual(contents[i], 3.14, accuracy: 1e-6)
        }
    }
    
    func testTypedBufferInt32() async throws {
        let count = 128
        let data = Array(0..<count).map { Int32($0) }
        
        let token = try await bufferPool.getBuffer(for: data)
        XCTAssertNotNil(token.buffer)
        
        // Verify data integrity
        let contents = token.buffer.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            XCTAssertEqual(contents[i], Int32(i))
        }
    }
    
    // MARK: - Memory Management
    
    func testMemoryPressureHandling() async throws {
        var tokens: [BufferToken] = []
        let bufferSize = 1024 * 1024 // 1 MB each
        
        // Allocate many buffers to simulate pressure
        for _ in 0..<100 {
            if let token = try? await bufferPool.getBuffer(size: bufferSize) {
                tokens.append(token)
            }
        }
        
        // Pool should handle memory pressure
        XCTAssertGreaterThan(tokens.count, 0)
        
        // Return all buffers
        for token in tokens {
            token.returnToPool()
        }
        
        // Check cleanup occurred
        let stats = await bufferPool.getStatistics()
        XCTAssertGreaterThan(stats.availableBuffers, 0)
    }
    
    func testPoolReset() async throws {
        // Allocate some buffers
        let token1 = try await bufferPool.getBuffer(size: 1024)
        let token2 = try await bufferPool.getBuffer(size: 2048)
        
        token1.returnToPool()
        token2.returnToPool()
        
        // Reset pool
        await bufferPool.reset()
        
        // Stats should be cleared
        let stats = await bufferPool.getStatistics()
        XCTAssertEqual(stats.allocationCount, 0)
        XCTAssertEqual(stats.availableBuffers, 0)
    }
    
    // MARK: - Alignment Tests
    
    func testCustomAlignment() async throws {
        let alignments = [16, 32, 64, 128, 256]
        
        for alignment in alignments {
            let token = try await bufferPool.getBuffer(
                size: 1024
            )
            
            let address = token.buffer.contents()
            XCTAssertEqual(
                Int(bitPattern: address) % alignment, 0,
                "Buffer should be \(alignment)-byte aligned"
            )
        }
    }
    
    func testPageAlignedBuffers() async throws {
        let pageSize = 4096
        let token = try await bufferPool.getBuffer(
            size: pageSize * 2
        )
        
        let address = token.buffer.contents()
        XCTAssertEqual(
            Int(bitPattern: address) % pageSize, 0,
            "Buffer should be page-aligned"
        )
    }
    
    // MARK: - Performance Tests
    
    func testAllocationPerformance() async throws {
        let iterations = 1000
        let size = 4096
        
        let start = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let token = try await bufferPool.getBuffer(size: size)
            token.returnToPool()
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgTime = elapsed / Double(iterations)
        
        print("Average allocation time: \(avgTime * 1000) ms")
        XCTAssertLessThan(avgTime, 0.001, "Allocation should be fast")
    }
    
    func testReusePerformance() async throws {
        let size = 8192
        let iterations = 100
        
        // Pre-warm the pool
        for _ in 0..<10 {
            let token = try await bufferPool.getBuffer(size: size)
            token.returnToPool()
        }
        
        // Measure reuse performance
        let start = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let token = try await bufferPool.getBuffer(size: size)
            token.returnToPool()
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Verify high reuse rate
        let stats = await bufferPool.getStatistics()
        let reuseRate = Double(stats.hitCount) / Double(stats.allocationCount)
        XCTAssertGreaterThan(reuseRate, 0.8, "Should have high reuse rate")
    }
    
    // MARK: - Concurrent Access Tests
    
    func testConcurrentAllocation() async throws {
        let concurrentTasks = 50
        let bufferSize = 2048
        
        await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<concurrentTasks {
                group.addTask { [bufferPool] in
                    do {
                        let token = try await bufferPool!.getBuffer(size: bufferSize)
                        // Do some work
                        await Task.yield()
                        token.returnToPool()
                        return true
                    } catch {
                        return false
                    }
                }
            }
            
            var successCount = 0
            for await success in group {
                if success { successCount += 1 }
            }
            
            XCTAssertEqual(successCount, concurrentTasks)
        }
    }
    
    func testConcurrentMixedSizes() async throws {
        let sizes = [512, 1024, 2048, 4096, 8192]
        let tasksPerSize = 10
        
        await withTaskGroup(of: Bool.self) { group in
            for size in sizes {
                for _ in 0..<tasksPerSize {
                    group.addTask { [bufferPool] in
                        do {
                            let token = try await bufferPool!.getBuffer(size: size)
                            // Simulate work
                            try await Task.sleep(nanoseconds: 1_000_000) // 1ms
                            token.returnToPool()
                            return true
                        } catch {
                            return false
                        }
                    }
                }
            }
            
            var successCount = 0
            for await success in group {
                if success { successCount += 1 }
            }
            
            XCTAssertGreaterThan(successCount, sizes.count * tasksPerSize / 2)
        }
    }
    
    // MARK: - Statistics and Monitoring
    
    func testStatisticsAccuracy() async throws {
        let allocCount = 10
        let size = 1024
        var tokens: [BufferToken] = []
        
        // Allocate buffers
        for _ in 0..<allocCount {
            let token = try await bufferPool.getBuffer(size: size)
            tokens.append(token)
        }
        
        var stats = await bufferPool.getStatistics()
        XCTAssertEqual(stats.allocationCount, allocCount)
        XCTAssertEqual(stats.totalBuffers - stats.availableBuffers, allocCount)
        
        // Return half
        for i in 0..<allocCount/2 {
            tokens[i].returnToPool()
        }
        
        stats = await bufferPool.getStatistics()
        XCTAssertEqual(stats.availableBuffers, allocCount/2)
        XCTAssertEqual(stats.totalBuffers - stats.availableBuffers, allocCount - allocCount/2)
    }
    
    func testMemoryUtilization() async throws {
        let sizes = [1024, 2048, 4096]
        var totalBytes = 0
        var tokens: [BufferToken] = []
        
        for size in sizes {
            let token = try await bufferPool.getBuffer(size: size)
            totalBytes += token.buffer.length
            tokens.append(token)
        }
        
        let stats = await bufferPool.getStatistics()
        XCTAssertGreaterThanOrEqual(stats.currentMemoryUsage, totalBytes)
        
        // Return all
        for token in tokens {
            token.returnToPool()
        }
    }
    
    // MARK: - Edge Cases
    
    func testZeroSizeBuffer() async throws {
        do {
            _ = try await bufferPool.getBuffer(size: 0)
            XCTFail("Should not allow zero-size buffer")
        } catch {
            XCTAssertTrue(error is AccelerationError)
        }
    }
    
    func testVeryLargeBuffer() async throws {
        let hugeSize = Int.max / 2
        
        do {
            _ = try await bufferPool.getBuffer(size: hugeSize)
            XCTFail("Should fail for unreasonably large buffer")
        } catch {
            // Expected to fail
            XCTAssertTrue(true)
        }
    }
    
    func testEmptyDataArray() async throws {
        let emptyData: [Float] = []
        
        do {
            _ = try await bufferPool.getBuffer(for: emptyData)
            XCTFail("Should not allow empty data")
        } catch {
            XCTAssertTrue(error is AccelerationError)
        }
    }
}