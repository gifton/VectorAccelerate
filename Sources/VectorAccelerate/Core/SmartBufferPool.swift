// VectorAccelerate: Smart Buffer Pool
//
// Optimized buffer management with intelligent caching and reuse
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Smart buffer pool with automatic sizing and efficient memory management
public actor SmartBufferPool {
    private let device: MetalDevice
    private var buffers: [Int: [any MTLBuffer]] = [:]
    private var currentMemory: Int = 0
    private let maxMemory: Int
    
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
            await performCleanup()
            
            if currentMemory + alignedSize > maxMemory {
                throw AccelerationError.memoryPressure
            }
        }
        
        // Allocate new buffer
        guard let buffer = await device.makeBuffer(length: alignedSize) else {
            throw AccelerationError.bufferAllocationFailed(size: alignedSize)
        }
        
        allocations += 1
        currentMemory += alignedSize
        
        return MetalBuffer(buffer: buffer, count: byteSize / MemoryLayout<Float>.stride)
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
                if let buffer = await device.makeBuffer(length: size) {
                    if buffers[size] == nil {
                        buffers[size] = []
                    }
                    buffers[size]?.append(buffer)
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