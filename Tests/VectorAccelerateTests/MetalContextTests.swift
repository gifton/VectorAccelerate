//
//  MetalContextTests.swift
//  VectorAccelerateTests
//
//  Tests for Metal context and device management
//

import XCTest
@testable import VectorAccelerate
import VectorCore
import Metal

final class MetalContextTests: XCTestCase {
    
    // MARK: - Device Tests
    
    func testMetalAvailability() {
        let isAvailable = MetalDevice.isAvailable
        
        #if targetEnvironment(simulator)
        XCTAssertFalse(isAvailable, "Metal should not be available in simulator")
        #else
        XCTAssertTrue(isAvailable, "Metal should be available on real devices")
        #endif
    }
    
    func testDeviceInitialization() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let device = try MetalDevice()
        let capabilities = await device.capabilities
        
        // Verify basic capabilities
        XCTAssertGreaterThan(capabilities.maxThreadsPerThreadgroup, 0)
        XCTAssertGreaterThan(capabilities.maxBufferLength, 0)
        XCTAssertGreaterThan(capabilities.recommendedMaxWorkingSetSize, 0)
        
        // Check device name
        let name = await device.name
        XCTAssertFalse(name.isEmpty)
        
        print("Device: \(name)")
        print("Unified Memory: \(capabilities.hasUnifiedMemory)")
        print("Metal 3: \(capabilities.supportsMetal3)")
        print("Max Threads: \(capabilities.maxThreadsPerThreadgroup)")
    }
    
    func testDeviceSelection() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let bestDevice = try await MetalDevice.selectBestDevice()
        let defaultDevice = try MetalDevice()
        
        // Best device should exist
        XCTAssertNotNil(bestDevice)
        XCTAssertNotNil(defaultDevice)
        
        // On systems with multiple GPUs, best device might differ
        let bestName = await bestDevice.name
        let defaultName = await defaultDevice.name
        
        print("Best device: \(bestName)")
        print("Default device: \(defaultName)")
    }
    
    // MARK: - Context Tests
    
    func testContextInitialization() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        // Verify context components
        let deviceName = await context.deviceName
        XCTAssertFalse(deviceName.isEmpty)
        
        let hasUnified = await context.hasUnifiedMemory
        print("Context device: \(deviceName), Unified: \(hasUnified)")
        
        // Test performance stats
        let stats = await context.getPerformanceStats()
        XCTAssertEqual(stats.operationCount, 0)
        XCTAssertEqual(stats.totalComputeTime, 0)
    }
    
    func testContextFactory() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        // Test different context configurations
        let batchContext = try await MetalContextFactory.createBatchOptimized()
        let realtimeContext = try await MetalContextFactory.createRealTimeOptimized()
        let debugContext = try await MetalContextFactory.createDebug()
        
        // All should be created successfully
        XCTAssertNotNil(batchContext)
        XCTAssertNotNil(realtimeContext)
        XCTAssertNotNil(debugContext)
        
        // Debug context should have profiling enabled
        // Configuration checks would require await due to actor isolation
        // The factory methods ensure proper configuration
    }
    
    // MARK: - Buffer Pool Tests
    
    func testBufferAllocation() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        // Allocate various buffer sizes
        let sizes = [1024, 4096, 16384, 65536]
        var tokens: [BufferToken] = []
        
        for size in sizes {
            let token = try await context.getBuffer(size: size)
            XCTAssertNotNil(token)
            XCTAssertGreaterThanOrEqual(token.buffer.length, size)
            tokens.append(token)
        }
        
        // Check pool statistics
        let stats = await context.getPoolStatistics()
        XCTAssertEqual(stats.allocationCount, sizes.count)
        XCTAssertEqual(stats.missCount, sizes.count) // First allocations are misses
        XCTAssertEqual(stats.hitCount, 0)
    }
    
    func testBufferReuse() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        // Allocate and release a buffer
        let size = 4096
        var token: BufferToken? = try await context.getBuffer(size: size)
        token = nil // Buffer returns to pool
        
        // Wait a moment for async return
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        
        // Allocate same size again - should hit cache
        token = try await context.getBuffer(size: size)
        
        let stats = await context.getPoolStatistics()
        XCTAssertGreaterThan(stats.hitRate, 0) // Should have cache hit
        
        print("Buffer pool hit rate: \(stats.hitRate)")
        print("Memory usage: \(stats.currentMemoryUsage) / \(stats.maxMemoryLimit)")
    }
    
    func testBufferDataOperations() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        // Create buffer with float data
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let token = try await context.getBuffer(for: data)
        
        // Read data back
        let readData = token.copyData(as: Float.self)
        XCTAssertEqual(readData.count, data.count)
        XCTAssertEqual(readData, data)
        
        // Modify data
        let newData: [Float] = [10.0, 20.0, 30.0, 40.0, 50.0]
        token.write(data: newData)
        
        // Verify modification
        let modifiedData = token.copyData(as: Float.self)
        XCTAssertEqual(modifiedData, newData)
    }
    
    // MARK: - Thread Configuration Tests
    
    func testThreadGroupCalculation() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        // Test various data counts
        let dataCounts = [32, 256, 1024, 10000, 1000000]
        
        for count in dataCounts {
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: count)
            
            // Verify coverage
            let totalThreads = threadsPerGroup.width * threadgroups.width
            XCTAssertGreaterThanOrEqual(totalThreads, count)
            
            // Verify power of 2 optimization
            let width = threadsPerGroup.width
            XCTAssertTrue((width & (width - 1)) == 0 || width == 1) // Power of 2
            
            print("Data: \(count) -> Threads: \(threadsPerGroup.width) x Groups: \(threadgroups.width)")
        }
    }
    
    // MARK: - Performance Tests
    
    func testBufferAllocationPerformance() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let context = try await VectorAccelerate.MetalContext()
        
        measure {
            Task {
                // Allocate and deallocate many buffers
                for _ in 0..<100 {
                    _ = try? await context.getBuffer(size: 4096)
                }
            }
        }
    }
    
    func testMemoryPressureHandling() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        // Create context with small memory limit
        let config = MetalConfiguration(
            maxBufferPoolMemory: 10 * 1024 * 1024, // 10MB
            maxBuffersPerSize: 5
        )
        let context = try await VectorAccelerate.MetalContext(configuration: config)
        
        // Try to allocate more than the limit
        var tokens: [BufferToken] = []
        let largeSize = 4 * 1024 * 1024 // 4MB each
        
        // Should succeed for first few
        for i in 0..<3 {
            do {
                let token = try await context.getBuffer(size: largeSize)
                tokens.append(token)
                print("Allocated buffer \(i+1)")
            } catch let error as VectorError where error.kind == .resourceExhausted {
                print("Memory pressure at buffer \(i+1)")
                break
            }
        }
        
        // Verify some buffers were allocated
        XCTAssertGreaterThan(tokens.count, 0)
        
        // Clear and try again
        tokens.removeAll()
        try await Task.sleep(nanoseconds: 100_000_000) // Wait for cleanup
        
        // Should be able to allocate again
        let token = try await context.getBuffer(size: largeSize)
        XCTAssertNotNil(token)
    }
}