//
//  MetalBufferFactory.swift
//  VectorAccelerate
//
//  Synchronous, non-actor buffer factory for Metal buffers.
//  Creates buffers and copies payloads without async boundaries.
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
/// // Copy directly from VectorProtocol storage
/// let vectorBuffer = factory.createBuffer(fromVectors: vectors)
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

    /// Create a buffer whose byte length is rounded up to the requested alignment.
    ///
    /// This does not promise arbitrary base-address alignment. Invalid/nonpositive
    /// sizes, non-power-of-two alignment and unsupported rounded lengths return nil.
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
        guard let alignedLength = Self.checkedAlignedLength(
            length, alignment: alignment, maximum: device.maxBufferLength
        ) else { return nil }
        return createBuffer(length: alignedLength, options: options)
    }

    /// Copy an array into a buffer with a rounded byte length and zeroed tail padding.
    /// Empty payloads, invalid sizes/alignment, and CPU-inaccessible storage modes return nil.
    /// Source elements must be safe to copy as raw bytes, as with the unaligned array API.
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
        guard let byteCount = Self.checkedByteCount(data.count, stride: MemoryLayout<T>.stride) else {
            return nil
        }
        return createInitializedBuffer(byteCount: byteCount, alignment: alignment, options: options) { destination in
            data.withUnsafeBytes { bytes in
                guard bytes.count == byteCount, let base = bytes.baseAddress else { return false }
                destination.copyMemory(from: base, byteCount: byteCount)
                return true
            }
        }
    }

    // MARK: - VectorProtocol Buffer Creation

    /// Copy a rectangular batch directly into a buffer with zeroed tail padding.
    /// Empty/ragged rows, mismatched exposed counts, invalid sizes/alignment, and
    /// CPU-inaccessible storage modes return nil. Alignment rounds byte length only.
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
        guard let first = vectors.first else { return nil }
        let dimension = first.count
        guard dimension > 0, vectors.allSatisfy({ $0.count == dimension }) else { return nil }
        let (totalCount, overflow) = vectors.count.multipliedReportingOverflow(by: dimension)
        guard !overflow, let byteCount = Self.checkedByteCount(totalCount, stride: MemoryLayout<Float>.stride) else {
            return nil
        }

        return createInitializedBuffer(byteCount: byteCount, alignment: alignment, options: options) { destination in
            for (i, vector) in vectors.enumerated() {
                let copied = vector.withUnsafeBufferPointer { source in
                    guard source.count == dimension, let base = source.baseAddress else { return false }
                    // The complete product was checked before allocation; every row lies within it.
                    destination.advanced(by: i * dimension * MemoryLayout<Float>.stride)
                        .copyMemory(from: base, byteCount: dimension * MemoryLayout<Float>.stride)
                    return true
                }
                guard copied else { return false }
            }
            return true
        }
    }

    /// Copy a single VectorProtocol payload into a buffer with zeroed tail padding.
    /// Empty payloads, mismatched exposed counts, invalid sizes/alignment, and
    /// CPU-inaccessible storage modes return nil. Alignment rounds byte length only.
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
        let count = vector.count
        guard let byteCount = Self.checkedByteCount(count, stride: MemoryLayout<Float>.stride) else {
            return nil
        }
        return createInitializedBuffer(byteCount: byteCount, alignment: alignment, options: options) { destination in
            vector.withUnsafeBufferPointer { source in
                guard source.count == count, let base = source.baseAddress else { return false }
                destination.copyMemory(from: base, byteCount: byteCount)
                return true
            }
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

    // MARK: - Checked initialized storage

    @usableFromInline
    internal static func checkedByteCount(_ count: Int, stride: Int) -> Int? {
        guard count > 0, stride > 0 else { return nil }
        let (bytes, overflow) = count.multipliedReportingOverflow(by: stride)
        return overflow ? nil : bytes
    }

    @usableFromInline
    internal static func checkedAlignedLength(_ length: Int, alignment: Int, maximum: Int) -> Int? {
        guard length > 0, alignment > 0, alignment & (alignment - 1) == 0 else { return nil }
        let (sum, overflow) = length.addingReportingOverflow(alignment - 1)
        guard !overflow else { return nil }
        let rounded = sum & ~(alignment - 1)
        return rounded <= maximum ? rounded : nil
    }

    /// Allocate capacity independently from source payload length. The copy closure
    /// writes exactly byteCount bytes; only destination padding is initialized afterward.
    @usableFromInline
    internal func createInitializedBuffer(
        byteCount: Int,
        alignment: Int,
        options: MTLResourceOptions?,
        copy: (UnsafeMutableRawPointer) -> Bool
    ) -> (any MTLBuffer)? {
        guard let length = Self.checkedAlignedLength(
            byteCount, alignment: alignment, maximum: device.maxBufferLength
        ) else { return nil }
        let actualOptions = options ?? defaultOptions
        let storageMode = (actualOptions.rawValue & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift
        var cpuAccessible = storageMode == MTLStorageMode.shared.rawValue
        #if os(macOS)
        cpuAccessible = cpuAccessible || storageMode == MTLStorageMode.managed.rawValue
        #endif
        guard cpuAccessible,
              let buffer = device.makeBuffer(length: length, options: actualOptions),
              buffer.length >= length else { return nil }
        let destination = buffer.contents()
        guard copy(destination) else { return nil }
        destination.advanced(by: byteCount).initializeMemory(
            as: UInt8.self, repeating: 0, count: length - byteCount
        )
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<length)
        }
        #endif
        return buffer
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
