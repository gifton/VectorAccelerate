// KernelContext.swift
// Synchronous wrapper for kernel initialization

import Metal
import Foundation

/// Synchronous context for kernel initialization
/// This is a lightweight wrapper that provides the minimal Metal resources needed by kernels
public final class KernelContext: @unchecked Sendable {
    public let device: any MTLDevice
    public let commandQueue: any MTLCommandQueue

    // Cache for shared instances
    // Using nonisolated(unsafe) as this is protected by explicit lock synchronization
    nonisolated(unsafe) private static var sharedInstances: [ObjectIdentifier: KernelContext] = [:]
    private static let lock = NSLock()

    /// Create a kernel context with the given device
    public init(device: any MTLDevice) throws {
        self.device = device

        // Create command queue synchronously
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.commandQueueCreationFailed
        }
        self.commandQueue = queue
    }

    /// Get or create a shared context for the device
    public static func shared(for device: any MTLDevice) throws -> KernelContext {
        let deviceId = ObjectIdentifier(device)

        lock.lock()
        defer { lock.unlock() }

        if let existing = sharedInstances[deviceId] {
            return existing
        }

        let context = try KernelContext(device: device)
        sharedInstances[deviceId] = context
        return context
    }

    /// Create a buffer from data
    public func createBuffer<T>(from data: [T], options: MTLResourceOptions) -> (any MTLBuffer)? {
        let size = data.count * MemoryLayout<T>.stride
        return data.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: size, options: options)
        }
    }

    /// Create a buffer with guaranteed alignment for SIMD operations
    /// - Parameters:
    ///   - data: Input data array
    ///   - options: Metal resource options
    ///   - alignment: Required alignment in bytes (default 16 for SIMD)
    /// - Returns: Aligned buffer or nil if creation fails
    public func createAlignedBuffer<T>(
        from data: [T],
        options: MTLResourceOptions,
        alignment: Int = 16
    ) -> (any MTLBuffer)? {
        let size = data.count * MemoryLayout<T>.stride
        // Ensure size is aligned
        let alignedSize = (size + alignment - 1) & ~(alignment - 1)

        return data.withUnsafeBytes { bytes in
            // Metal buffers are already 256-byte aligned by default
            // But we add explicit alignment for documentation
            device.makeBuffer(bytes: bytes.baseAddress!, length: alignedSize, options: options)
        }
    }

    /// Validate buffer alignment for SIMD operations
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - requiredAlignment: Required alignment in bytes
    /// - Returns: True if buffer is properly aligned
    public static func isBufferAligned(_ buffer: any MTLBuffer, alignment: Int = 16) -> Bool {
        // Check if buffer address is aligned
        let address = buffer.contents()
        let addressInt = Int(bitPattern: address)
        return addressInt % alignment == 0
    }

    /// Validate that buffer size is suitable for SIMD operations
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - elementSize: Size of each element in bytes
    ///   - simdWidth: SIMD width (e.g., 4 for float4)
    /// - Returns: True if buffer can be processed with SIMD
    public static func isBufferSIMDCompatible(
        _ buffer: any MTLBuffer,
        elementSize: Int,
        simdWidth: Int = 4
    ) -> Bool {
        let elementCount = buffer.length / elementSize
        // Check if we have at least one full SIMD vector
        return elementCount >= simdWidth && isBufferAligned(buffer, alignment: elementSize * simdWidth)
    }
}