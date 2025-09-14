//
//  BufferPool.swift
//  VectorAccelerate
//
//  Token-based buffer pool for efficient Metal memory management
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Configuration for buffer pool behavior
public struct BufferPoolConfiguration: Sendable {
    public let maxBufferCount: Int
    public let maxTotalSize: Int
    public let reuseThreshold: Int
    public let adaptiveSizing: Bool
    
    public init(
        maxBufferCount: Int = 100,
        maxTotalSize: Int = 1024 * 1024 * 1024, // 1GB
        reuseThreshold: Int = 4096,
        adaptiveSizing: Bool = true
    ) {
        self.maxBufferCount = maxBufferCount
        self.maxTotalSize = maxTotalSize
        self.reuseThreshold = reuseThreshold
        self.adaptiveSizing = adaptiveSizing
    }
    
    public static let `default` = BufferPoolConfiguration()
}

/// Token representing a borrowed buffer from the pool
/// Automatically returns buffer to pool when deallocated (RAII pattern)
public final class BufferToken: @unchecked Sendable {
    public let buffer: any MTLBuffer
    private let pool: BufferPool?
    public let size: Int
    private var dataCount: Int?  // Track actual data count for typed data
    private var isReturned: Bool = false
    private let lock = NSLock()
    
    init(buffer: any MTLBuffer, size: Int, pool: BufferPool?, dataCount: Int? = nil) {
        self.buffer = buffer
        self.size = size
        self.pool = pool
        self.dataCount = dataCount
    }
    
    deinit {
        lock.lock()
        defer { lock.unlock() }
        
        guard !isReturned else { return }
        
        // Capture what we need before self is deallocated
        let capturedBuffer = buffer
        let capturedSize = size
        let capturedPool = pool
        
        Task.detached {
            await capturedPool?.returnBuffer(capturedBuffer, size: capturedSize)
        }
    }
    
    /// Manually return buffer to pool (optional - happens automatically on deinit)
    public func returnToPool() {
        lock.lock()
        defer { lock.unlock() }
        
        guard !isReturned else { return }
        isReturned = true
        
        let capturedBuffer = buffer
        let capturedSize = size
        let capturedPool = pool
        
        Task.detached {
            await capturedPool?.returnBuffer(capturedBuffer, size: capturedSize)
        }
    }
    
    /// Get buffer contents as typed array
    public func contents<T>(as type: T.Type) -> UnsafeMutablePointer<T> {
        buffer.contents().bindMemory(to: T.self, capacity: size / MemoryLayout<T>.stride)
    }
    
    /// Copy data from buffer
    public func copyData<T>(as type: T.Type) -> [T] {
        lock.lock()
        defer { lock.unlock() }
        
        // If we have tracked data count, use it; otherwise use full buffer
        let count: Int
        if let dataCount = dataCount {
            count = dataCount
        } else {
            count = size / MemoryLayout<T>.stride
        }
        
        let pointer = contents(as: type)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    /// Copy specific amount of data from buffer
    public func copyData<T>(as type: T.Type, count: Int) -> [T] {
        precondition(count * MemoryLayout<T>.stride <= size, "Requested count exceeds buffer size")
        let pointer = contents(as: type)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    /// Write data to buffer
    public func write<T>(data: [T]) {
        let maxCount = size / MemoryLayout<T>.stride
        precondition(data.count <= maxCount, "Data exceeds buffer capacity")
        
        let pointer = contents(as: T.self)
        data.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: data.count)
        }
        
        // Track the actual data count
        lock.lock()
        dataCount = data.count
        lock.unlock()
    }
}

/// Size-based bucket for buffer organization
private struct BufferBucket {
    let size: Int
    var available: [any MTLBuffer] = []
    var inUse: Set<ObjectIdentifier> = []
    
    init(size: Int) {
        self.size = size
    }
}

/// High-performance buffer pool with automatic memory management
public actor BufferPool {
    private let device: MetalDevice
    private var buckets: [Int: BufferBucket] = [:]
    private let maxBuffersPerBucket: Int
    private let maxTotalMemory: Int
    private var currentMemoryUsage: Int = 0
    
    // Predefined bucket sizes for common operations
    private let bucketSizes: [Int] = [
        1024,           // 1 KB - Small metadata
        4096,           // 4 KB - Small vectors
        16384,          // 16 KB - Medium vectors
        65536,          // 64 KB - Large vectors
        262144,         // 256 KB - Batch operations
        1048576,        // 1 MB - Large batches
        4194304,        // 4 MB - Very large batches
        16777216,       // 16 MB - Massive operations
        67108864        // 64 MB - Maximum single buffer
    ]
    
    // Performance metrics
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var allocationCount: Int = 0
    
    // MARK: - Initialization
    
    public init(device: MetalDevice, maxBuffersPerBucket: Int = 10, maxTotalMemory: Int? = nil) {
        self.device = device
        self.maxBuffersPerBucket = maxBuffersPerBucket
        
        // Set max memory based on device capabilities or default to 1GB
        if let maxMemory = maxTotalMemory {
            self.maxTotalMemory = maxMemory
        } else {
            let capabilities = device.capabilities
            let recommendedSize = capabilities.recommendedMaxWorkingSetSize
            self.maxTotalMemory = min(recommendedSize / 2, 1024 * 1024 * 1024) // Max 1GB or half of recommended
        }
        
        // Initialize buckets
        for size in bucketSizes {
            buckets[size] = BufferBucket(size: size)
        }
    }
    
    // MARK: - Buffer Management
    
    /// Get a buffer of at least the specified size
    public func getBuffer(size: Int) async throws -> BufferToken {
        // Find appropriate bucket size
        let bucketSize = selectBucketSize(for: size)
        
        // Check if size exceeds maximum
        if bucketSize > bucketSizes.last! {
            throw AccelerationError.invalidBufferSize(requested: size, maximum: bucketSizes.last!)
        }
        
        // Get or create bucket
        if buckets[bucketSize] == nil {
            buckets[bucketSize] = BufferBucket(size: bucketSize)
        }
        
        // Try to get available buffer
        if var bucket = buckets[bucketSize], !bucket.available.isEmpty {
            let buffer = bucket.available.removeLast()
            bucket.inUse.insert(ObjectIdentifier(buffer))
            buckets[bucketSize] = bucket
            hitCount += 1
            return BufferToken(buffer: buffer, size: bucketSize, pool: self)
        }
        
        // Need to allocate new buffer
        missCount += 1
        
        // Check memory pressure
        if currentMemoryUsage + bucketSize > maxTotalMemory {
            try await performMemoryCleanup()
            
            // Check again after cleanup
            if currentMemoryUsage + bucketSize > maxTotalMemory {
                throw AccelerationError.memoryPressure
            }
        }
        
        // Check bucket limit
        if var bucket = buckets[bucketSize] {
            if bucket.available.count + bucket.inUse.count >= maxBuffersPerBucket {
                // Try to clean up this specific bucket
                bucket.available.removeAll()
                buckets[bucketSize] = bucket
            }
        }
        
        // Allocate new buffer
        guard let buffer = await device.makeBuffer(length: bucketSize) else {
            throw AccelerationError.bufferAllocationFailed(size: bucketSize)
        }
        
        buffer.label = "VectorAccelerate.BufferPool.\(bucketSize)"
        
        // Track allocation
        allocationCount += 1
        currentMemoryUsage += bucketSize
        
        // Update bucket
        var bucket = buckets[bucketSize]!
        bucket.inUse.insert(ObjectIdentifier(buffer))
        buckets[bucketSize] = bucket
        
        return BufferToken(buffer: buffer, size: bucketSize, pool: self)
    }
    
    /// Return a buffer to the pool
    func returnBuffer(_ buffer: any MTLBuffer, size: Int) {
        // Find the appropriate bucket
        guard var bucket = buckets[size] else { return }
        
        // Remove from in-use set
        bucket.inUse.remove(ObjectIdentifier(buffer))
        
        // Add to available if under limit
        if bucket.available.count < maxBuffersPerBucket {
            bucket.available.append(buffer)
        } else {
            // Buffer pool is full for this size, let it be deallocated
            currentMemoryUsage -= size
        }
        
        buckets[size] = bucket
    }
    
    /// Get buffer for typed data
    public func getBuffer<T>(for data: [T]) async throws -> BufferToken {
        let size = data.count * MemoryLayout<T>.stride
        let token = try await getBuffer(size: size)
        token.write(data: data)
        return token
    }
    
    /// Check if a buffer is currently in use
    public func isBufferInUse(_ buffer: any MTLBuffer) -> Bool {
        for bucket in buckets.values {
            if bucket.inUse.contains(ObjectIdentifier(buffer)) {
                return true
            }
        }
        return false
    }
    
    /// Create a buffer with specific alignment requirements
    public func getAlignedBuffer(size: Int, alignment: Int = 16) async throws -> BufferToken {
        let alignedSize = (size + alignment - 1) & ~(alignment - 1)
        return try await getBuffer(size: alignedSize)
    }
    
    // MARK: - Memory Management
    
    /// Perform memory cleanup under pressure
    private func performMemoryCleanup() async throws {
        var freedMemory = 0
        
        // Clear available buffers from largest buckets first
        for size in bucketSizes.reversed() {
            guard var bucket = buckets[size] else { continue }
            
            let buffersToFree = bucket.available.count
            if buffersToFree > 0 {
                freedMemory += size * buffersToFree
                bucket.available.removeAll()
                buckets[size] = bucket
            }
            
            // Stop if we've freed enough memory
            if freedMemory > maxTotalMemory / 4 {
                break
            }
        }
        
        currentMemoryUsage -= freedMemory
    }
    
    /// Clear all cached buffers
    public func clearCache() {
        for size in buckets.keys {
            buckets[size]?.available.removeAll()
        }
    }
    
    /// Reset the pool completely
    public func reset() {
        buckets.removeAll()
        for size in bucketSizes {
            buckets[size] = BufferBucket(size: size)
        }
        currentMemoryUsage = 0
        hitCount = 0
        missCount = 0
        allocationCount = 0
    }
    
    // MARK: - Utilities
    
    /// Select appropriate bucket size for requested size
    private func selectBucketSize(for requestedSize: Int) -> Int {
        for size in bucketSizes {
            if size >= requestedSize {
                return size
            }
        }
        return bucketSizes.last!
    }
    
    // MARK: - Performance Metrics
    
    /// Get pool performance statistics
    public func getStatistics() -> PoolStatistics {
        let totalRequests = hitCount + missCount
        let hitRate = totalRequests > 0 ? Double(hitCount) / Double(totalRequests) : 0.0
        
        var totalBuffers = 0
        var availableBuffers = 0
        
        for bucket in buckets.values {
            totalBuffers += bucket.available.count + bucket.inUse.count
            availableBuffers += bucket.available.count
        }
        
        return PoolStatistics(
            hitRate: hitRate,
            hitCount: hitCount,
            missCount: missCount,
            allocationCount: allocationCount,
            currentMemoryUsage: currentMemoryUsage,
            maxMemoryLimit: maxTotalMemory,
            totalBuffers: totalBuffers,
            availableBuffers: availableBuffers
        )
    }
}

/// Statistics for buffer pool performance
public struct PoolStatistics: Sendable {
    public let hitRate: Double
    public let hitCount: Int
    public let missCount: Int
    public let allocationCount: Int
    public let currentMemoryUsage: Int
    public let maxMemoryLimit: Int
    public let totalBuffers: Int
    public let availableBuffers: Int
    
    public var memoryUtilization: Double {
        Double(currentMemoryUsage) / Double(maxMemoryLimit)
    }
    
    public var bufferUtilization: Double {
        let inUse = totalBuffers - availableBuffers
        return totalBuffers > 0 ? Double(inUse) / Double(totalBuffers) : 0.0
    }
}

// MARK: - Buffer Extensions

extension BufferToken {
    /// Create a unified buffer for Apple Silicon (shared between CPU and GPU)
    public var isUnified: Bool {
        #if os(macOS)
        return buffer.storageMode == .shared
        #else
        return true // iOS/tvOS always use unified memory
        #endif
    }
    
    /// Synchronize buffer contents (only needed for managed buffers on Intel Macs)
    public func synchronize() {
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<buffer.length)
        }
        #endif
    }
}