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
}