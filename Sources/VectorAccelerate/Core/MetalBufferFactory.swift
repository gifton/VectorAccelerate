//
//  MetalBufferFactory.swift
//  VectorAccelerate
//
//  Synchronous, non-actor buffer factory for Metal buffers.
//  Enables zero-copy buffer creation without async boundaries.
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Synchronous buffer factory for Metal buffer creation.
///
/// This class provides non-actor buffer creation, enabling synchronous buffer operations
/// without the async overhead of going through the `BufferPool` actor. It's designed
/// for performance-critical paths where buffer pooling is not needed.
///
/// ## Thread Safety
/// MTLDevice's buffer creation methods are thread-safe, so this class can be safely
/// used from multiple threads concurrently. The `@unchecked Sendable` conformance
/// reflects this thread-safe design.
///
/// ## Usage
/// ```swift
/// let factory = MetalBufferFactory(device: metalDevice)
///
/// // Synchronous buffer creation
/// let buffer = factory.createBuffer(length: 4096)
///
/// // Zero-copy from VectorProtocol
/// let vectorBuffer = factory.createBuffer(from: vectors)
/// ```
///
/// ## Relationship to BufferPool
/// - Use `MetalBufferFactory` when you need direct buffer creation without pooling overhead
/// - Use `BufferPool` when you need buffer reuse and memory management
/// - `BufferPool` uses `MetalBufferFactory` internally for actual buffer allocation
public final class MetalBufferFactory: @unchecked Sendable {

    /// The underlying Metal device for buffer creation
    public let device: any MTLDevice

    /// Default resource options based on device capabilities
    public let defaultOptions: MTLResourceOptions

    /// Device capabilities for optimization decisions
    public let hasUnifiedMemory: Bool

    // MARK: - Initialization

    /// Create a buffer factory from an MTLDevice
    /// - Parameter device: The Metal device to use for buffer creation
    public init(device: any MTLDevice) {
        self.device = device
        self.hasUnifiedMemory = device.hasUnifiedMemory

        // Select optimal default options based on device architecture
        if device.hasUnifiedMemory {
            // Apple Silicon - shared memory is optimal
            self.defaultOptions = .storageModeShared
        } else {
            // Intel/AMD - managed memory for automatic CPU/GPU sync
            #if os(macOS)
            self.defaultOptions = .storageModeManaged
            #else
            self.defaultOptions = .storageModeShared
            #endif
        }
    }

    // MARK: - Basic Buffer Creation

    /// Create an empty buffer of the specified length
    /// - Parameters:
    ///   - length: Buffer size in bytes
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Created buffer or nil if allocation fails
    public func createBuffer(
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        device.makeBuffer(length: length, options: options ?? defaultOptions)
    }

    /// Create a buffer initialized with data
    /// - Parameters:
    ///   - bytes: Pointer to source data
    ///   - length: Size of data in bytes
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Created buffer with data copied, or nil if allocation fails
    public func createBuffer(
        bytes: UnsafeRawPointer,
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        device.makeBuffer(bytes: bytes, length: length, options: options ?? defaultOptions)
    }

    /// Create a buffer from an array of elements
    /// - Parameters:
    ///   - data: Array of elements to copy into buffer
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Created buffer with data copied, or nil if allocation fails
    public func createBuffer<T>(
        from data: [T],
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        let actualOptions = options ?? defaultOptions
        let size = data.count * MemoryLayout<T>.stride

        return data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: size, options: actualOptions)
        }
    }

    // MARK: - Aligned Buffer Creation

    /// Create a buffer with guaranteed alignment for SIMD operations
    /// - Parameters:
    ///   - length: Requested buffer size in bytes
    ///   - alignment: Required alignment in bytes (default 16 for float4)
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Aligned buffer or nil if allocation fails
    public func createAlignedBuffer(
        length: Int,
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        let alignedLength = (length + alignment - 1) & ~(alignment - 1)
        return createBuffer(length: alignedLength, options: options)
    }

    /// Create an aligned buffer from an array
    /// - Parameters:
    ///   - data: Array of elements to copy
    ///   - alignment: Required alignment in bytes (default 16)
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Aligned buffer with data copied, or nil if allocation fails
    public func createAlignedBuffer<T>(
        from data: [T],
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        let actualOptions = options ?? defaultOptions
        let size = data.count * MemoryLayout<T>.stride
        let alignedSize = (size + alignment - 1) & ~(alignment - 1)

        return data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: alignedSize, options: actualOptions)
        }
    }

    // MARK: - Zero-Copy VectorProtocol Buffer Creation

    /// Create an aligned buffer directly from VectorProtocol types without intermediate allocations.
    ///
    /// This method avoids the `.toArray()` anti-pattern by using `withUnsafeBufferPointer`
    /// to copy vector data directly into the Metal buffer.
    ///
    /// - Parameters:
    ///   - vectors: Array of VectorProtocol-conforming vectors
    ///   - alignment: Required alignment in bytes (default 16 for float4 SIMD)
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Buffer containing flattened vector data, or nil if creation fails
    ///
    /// - Complexity: O(n * d) where n is number of vectors and d is dimension
    @inlinable
    public func createBuffer<V: VectorProtocol>(
        fromVectors vectors: [V],
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        guard !vectors.isEmpty else { return nil }

        let actualOptions = options ?? defaultOptions
        let dimension = vectors[0].count
        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.stride
        let alignedSize = (byteSize + alignment - 1) & ~(alignment - 1)

        // Create buffer with aligned size
        guard let buffer = device.makeBuffer(length: alignedSize, options: actualOptions) else {
            return nil
        }

        // Get pointer to buffer contents
        let destination = buffer.contents().bindMemory(to: Float.self, capacity: totalCount)

        // Copy each vector directly using withUnsafeBufferPointer (zero intermediate allocation)
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            vector.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                let dst = destination.advanced(by: offset)
                dst.update(from: srcBase, count: min(srcPtr.count, dimension))
            }
        }

        return buffer
    }

    /// Create a buffer from a single VectorProtocol without intermediate allocation.
    ///
    /// - Parameters:
    ///   - vector: Single VectorProtocol-conforming vector
    ///   - alignment: Required alignment in bytes (default 16)
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Buffer containing vector data, or nil if creation fails
    @inlinable
    public func createBuffer<V: VectorProtocol>(
        fromVector vector: V,
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        let actualOptions = options ?? defaultOptions
        let count = vector.count
        let byteSize = count * MemoryLayout<Float>.stride
        let alignedSize = (byteSize + alignment - 1) & ~(alignment - 1)

        // Use withUnsafeBufferPointer to create buffer directly from vector storage
        return vector.withUnsafeBufferPointer { srcPtr in
            guard let srcBase = srcPtr.baseAddress else { return nil }
            return device.makeBuffer(bytes: srcBase, length: alignedSize, options: actualOptions)
        }
    }

    /// Create aligned buffers from two vector arrays efficiently.
    ///
    /// Optimized for the common case of query/database vector pairs.
    ///
    /// - Parameters:
    ///   - vectorsA: First array of vectors (e.g., queries)
    ///   - vectorsB: Second array of vectors (e.g., database)
    ///   - alignment: Required alignment in bytes (default 16)
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Tuple of buffers (A, B), or nil if creation fails
    @inlinable
    public func createBufferPair<V: VectorProtocol>(
        _ vectorsA: [V],
        _ vectorsB: [V],
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (bufferA: any MTLBuffer, bufferB: any MTLBuffer)? where V.Scalar == Float {
        guard let bufferA = createBuffer(fromVectors: vectorsA, alignment: alignment, options: options),
              let bufferB = createBuffer(fromVectors: vectorsB, alignment: alignment, options: options) else {
            return nil
        }
        return (bufferA, bufferB)
    }

    // MARK: - Bucket Size Helpers (for BufferPool integration)

    /// Standard bucket sizes for buffer pooling
    public static let standardBucketSizes: [Int] = [
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

    /// Select appropriate bucket size for a requested size
    /// - Parameter requestedSize: The size needed in bytes
    /// - Returns: The smallest bucket size >= requestedSize, or max bucket size
    public static func selectBucketSize(for requestedSize: Int) -> Int {
        for size in standardBucketSizes {
            if size >= requestedSize {
                return size
            }
        }
        return standardBucketSizes.last!
    }

    /// Create a buffer using bucket sizing (useful for pooled allocations)
    /// - Parameters:
    ///   - requestedSize: The minimum size needed
    ///   - options: Metal resource options (uses default if not specified)
    /// - Returns: Buffer with bucket-rounded size, or nil if allocation fails
    public func createBucketedBuffer(
        size requestedSize: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        let bucketSize = Self.selectBucketSize(for: requestedSize)
        return createBuffer(length: bucketSize, options: options)
    }

    // MARK: - Buffer Utilities

    /// Validate buffer alignment for SIMD operations
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - alignment: Required alignment in bytes (default 16)
    /// - Returns: True if buffer is properly aligned
    public static func isBufferAligned(_ buffer: any MTLBuffer, alignment: Int = 16) -> Bool {
        let address = buffer.contents()
        let addressInt = Int(bitPattern: address)
        return addressInt % alignment == 0
    }

    /// Get recommended alignment for the current device
    /// Metal buffers are always 256-byte aligned by Metal specification
    public var recommendedAlignment: Int {
        256 // Metal buffer alignment requirement
    }
}

// MARK: - MetalBuffer Wrapper Extension

extension MetalBufferFactory {

    /// Create a MetalBuffer (Sendable wrapper) from the factory
    /// - Parameters:
    ///   - length: Buffer size in bytes
    ///   - options: Metal resource options
    /// - Returns: MetalBuffer wrapper or nil if allocation fails
    public func createMetalBuffer(
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> MetalBuffer? {
        guard let buffer = createBuffer(length: length, options: options) else {
            return nil
        }
        return MetalBuffer(buffer: buffer, count: length / MemoryLayout<Float>.stride)
    }

    /// Create a MetalBuffer from an array
    /// - Parameters:
    ///   - data: Array of elements
    ///   - options: Metal resource options
    /// - Returns: MetalBuffer wrapper or nil if allocation fails
    public func createMetalBuffer<T>(
        from data: [T],
        options: MTLResourceOptions? = nil
    ) -> MetalBuffer? {
        guard let buffer = createBuffer(from: data, options: options) else {
            return nil
        }

        let elementType: MetalElementType
        switch T.self {
        case is Float.Type: elementType = .float32
        case is UInt8.Type: elementType = .uint8
        case is Int32.Type: elementType = .int32
        case is UInt32.Type: elementType = .uint32
        default: elementType = .float32
        }

        return MetalBuffer(buffer: buffer, count: data.count, elementType: elementType)
    }
}
