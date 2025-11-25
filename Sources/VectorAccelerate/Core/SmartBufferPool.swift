// VectorAccelerate: Smart Buffer Pool
//
// Optimized buffer management with intelligent caching and reuse
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Smart buffer pool with automatic sizing and efficient memory management
public actor SmartBufferPool: BufferProvider {
    private let device: MetalDevice
    private var buffers: [Int: [any MTLBuffer]] = [:]
    private var currentMemory: Int = 0
    private let maxMemory: Int

    // BufferProvider conformance - track handles for VectorCore integration
    private var activeHandles: [UUID: MetalBuffer] = [:]
    private var peakMemory: Int = 0

    // Performance metrics
    private var hits: Int = 0
    private var misses: Int = 0
    private var allocations: Int = 0
    
    // Common sizes for pre-allocation
    private static let commonSizes: [Int] = [
        4096,       // 4KB - small vectors
        16384,      // 16KB - medium vectors  
        65536,      // 64KB - large vectors
        262144,     // 256KB - embeddings
        1048576,    // 1MB - matrices
        4194304,    // 4MB - large batches
    ]
    
    public init(device: MetalDevice, maxMemoryMB: Int = 512) {
        self.device = device
        self.maxMemory = maxMemoryMB * 1024 * 1024
        
        // Pre-allocate common sizes
        Task {
            await preallocateCommonSizes()
        }
    }
    
    // MARK: - Core Operations
    
    /// Get a buffer of at least the specified size
    public func acquire(byteSize: Int) async throws -> MetalBuffer {
        let alignedSize = alignToOptimal(byteSize)
        
        // Check cache first
        if let cached = buffers[alignedSize], !cached.isEmpty {
            var available = cached
            let buffer = available.removeLast()
            buffers[alignedSize] = available
            hits += 1
            return MetalBuffer(buffer: buffer, count: byteSize / MemoryLayout<Float>.stride)
        }
        
        // Cache miss - allocate new
        misses += 1
        
        // Check memory pressure
        if currentMemory + alignedSize > maxMemory {
            performCleanup()

            if currentMemory + alignedSize > maxMemory {
                throw AccelerationError.memoryPressure
            }
        }
        
        // Allocate new buffer via Sendable wrapper
        guard let metal = await device.makeMetalBuffer(length: alignedSize) else {
            throw AccelerationError.bufferAllocationFailed(size: alignedSize)
        }
        
        allocations += 1
        currentMemory += alignedSize
        peakMemory = max(peakMemory, currentMemory)

        return MetalBuffer(buffer: metal.buffer, count: byteSize / MemoryLayout<Float>.stride)
    }
    
    /// Release a buffer back to the pool
    public func release(_ metalBuffer: MetalBuffer) {
        let alignedSize = alignToOptimal(metalBuffer.byteLength)
        
        if buffers[alignedSize] == nil {
            buffers[alignedSize] = []
        }
        
        // Only keep if under limit for this size
        let maxPerSize = min(10, maxMemory / alignedSize / 4)
        if buffers[alignedSize]!.count < maxPerSize {
            buffers[alignedSize]?.append(metalBuffer.buffer)
        } else {
            // Release least recently used
            currentMemory -= alignedSize
        }
    }
    
    // MARK: - Typed Operations
    
    /// Get a buffer for a specific type and count
    public func acquire<T>(for type: T.Type, count: Int) async throws -> MetalBuffer {
        let byteSize = count * MemoryLayout<T>.stride
        return try await acquire(byteSize: byteSize)
    }
    
    /// Get a buffer initialized with data
    public func acquire<T>(with data: [T]) async throws -> MetalBuffer {
        let buffer = try await acquire(for: T.self, count: data.count)
        
        // Initialize buffer with data
        data.withUnsafeBytes { bytes in
            buffer.buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        
        return buffer
    }
    
    // MARK: - Memory Management
    
    private func alignToOptimal(_ size: Int) -> Int {
        // Align to page size for efficiency
        let pageSize = 4096
        
        // Find the smallest common size that fits
        for commonSize in Self.commonSizes {
            if size <= commonSize {
                return commonSize
            }
        }
        
        // Align to page boundary for large sizes
        return ((size + pageSize - 1) / pageSize) * pageSize
    }
    
    private func performCleanup() {
        // Release 25% of cached buffers
        for (size, var cached) in buffers {
            let toRelease = max(1, cached.count / 4)
            for _ in 0..<toRelease {
                if !cached.isEmpty {
                    _ = cached.removeLast()
                    currentMemory -= size
                }
            }
            buffers[size] = cached
        }
    }
    
    private func preallocateCommonSizes() async {
        // Pre-allocate 2 buffers of each common size
        for size in Self.commonSizes.prefix(3) {
            for _ in 0..<2 {
                if let metal = await device.makeMetalBuffer(length: size) {
                    if buffers[size] == nil {
                        buffers[size] = []
                    }
                    buffers[size]?.append(metal.buffer)
                    currentMemory += size
                    allocations += 1
                }
            }
        }
    }
    
    // MARK: - Statistics
    
    public struct Statistics: Sendable {
        public let hitRate: Double
        public let totalAllocations: Int
        public let currentMemoryMB: Double
        public let cacheEfficiency: Double
    }
    
    public func getStatistics() -> Statistics {
        let total = Double(hits + misses)
        let hitRate = total > 0 ? Double(hits) / total : 0
        let efficiency = allocations > 0 ? Double(hits) / Double(allocations) : 0
        
        return Statistics(
            hitRate: hitRate,
            totalAllocations: allocations,
            currentMemoryMB: Double(currentMemory) / 1024 / 1024,
            cacheEfficiency: efficiency
        )
    }
    
    /// Reset the pool and release all buffers
    public func reset() {
        buffers.removeAll()
        currentMemory = 0
        hits = 0
        misses = 0
        allocations = 0
        peakMemory = 0
        activeHandles.removeAll()
    }

    // MARK: - BufferProvider Protocol Conformance

    /// Preferred alignment for Metal buffers (256 bytes per Metal specification)
    public nonisolated var alignment: Int {
        256  // Metal buffer alignment requirement
    }

    /// Acquire a buffer of at least the specified size (VectorCore interface)
    public func acquire(size: Int) async throws -> BufferHandle {
        // Get buffer from pool using existing infrastructure
        let metalBuffer = try await acquire(byteSize: size)

        // Create BufferHandle from the MetalBuffer
        let handle = BufferHandle(
            id: UUID(),
            size: metalBuffer.byteLength,
            pointer: metalBuffer.buffer.contents()
        )

        // Track the handle → MetalBuffer mapping for later release
        activeHandles[handle.id] = metalBuffer

        return handle
    }

    /// Release a buffer back to the pool (VectorCore interface)
    public func release(_ handle: BufferHandle) async {
        // Find the corresponding MetalBuffer
        guard let metalBuffer = activeHandles.removeValue(forKey: handle.id) else {
            // Handle not found - may have already been released
            return
        }

        // Return the buffer to the pool
        release(metalBuffer)
    }

    /// Get current buffer statistics (VectorCore interface)
    public func statistics() async -> BufferStatistics {
        let stats = getStatistics()

        return BufferStatistics(
            totalAllocations: stats.totalAllocations,
            reusedBuffers: hits,
            currentUsageBytes: currentMemory,
            peakUsageBytes: peakMemory
        )
    }

    /// Clear all cached buffers (VectorCore interface)
    public func clear() async {
        // Clear cached buffers but keep active handles
        performCleanup()
    }
}

// MARK: - Buffer Reference

/// Simple reference-counted buffer wrapper
public class BufferReference {
    public let metalBuffer: MetalBuffer
    private let pool: SmartBufferPool
    
    init(buffer: MetalBuffer, pool: SmartBufferPool) {
        self.metalBuffer = buffer
        self.pool = pool
    }
    
    deinit {
        // Release is handled by the pool when needed
        // Automatic cleanup via ARC
    }
}
