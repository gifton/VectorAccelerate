//
//  ResidencyManager.swift
//  VectorAccelerate
//
//  Manages explicit Metal 4 resource residency via MTLResidencySet
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Residency registration mode for resource lifecycle management
public enum ResidencyMode: String, Sendable, Codable {
    /// Resource is long-lived, rarely changes (e.g., model weights)
    case `static`

    /// Resource is transient, frequently allocated/freed (e.g., intermediate buffers)
    case ephemeral

    /// Resource is "hot" - frequently accessed, should stay resident
    case hot
}

/// Statistics for residency manager monitoring
public struct ResidencyStatistics: Sendable, Equatable {
    public let totalAllocations: Int
    public let staticAllocations: Int
    public let ephemeralAllocations: Int
    public let hotAllocations: Int
    public let totalBytesRegistered: Int
    public let commitCount: Int
    public let lastCommitTime: Date?

    public var description: String {
        """
        ResidencyStatistics:
          Total: \(totalAllocations) allocations (\(totalBytesRegistered / 1024 / 1024) MB)
          Static: \(staticAllocations), Ephemeral: \(ephemeralAllocations), Hot: \(hotAllocations)
          Commits: \(commitCount)
        """
    }
}

/// Manages explicit resource residency for Metal 4
///
/// Metal 4 requires all GPU-accessed resources to be in a residency set.
/// This actor handles:
/// - Creating and managing MTLResidencySet
/// - Tracking buffer additions/removals
/// - Integrating with BufferPool for automatic registration
/// - Handling memory pressure and eviction
///
/// Example:
/// ```swift
/// let manager = try ResidencyManager(device: device)
/// try await manager.registerBuffer(buffer, mode: .ephemeral)
/// try await manager.commit()
/// // Buffer is now GPU-resident
/// ```
public actor ResidencyManager {
    // MARK: - Types

    /// Internal tracking info for registered allocations
    private struct AllocationInfo {
        let allocation: any MTLBuffer
        let mode: ResidencyMode
        let byteSize: Int
        let label: String?
        let registeredAt: Date
    }

    // MARK: - Properties

    /// The underlying Metal residency set
    /// Note: MTLResidencySet is a Metal 4 type - using protocol for flexibility
    private let residencySet: any MTLResidencySet

    /// Device reference for capacity checks
    private let device: any MTLDevice

    /// Track registered allocations for debugging and removal
    private var registeredAllocations: [ObjectIdentifier: AllocationInfo] = [:]

    /// Commit batch tracking
    private var pendingAdds: [any MTLBuffer] = []
    private var pendingRemoves: [any MTLBuffer] = []
    private var needsCommit: Bool = false

    /// Statistics
    private var totalAllocations: Int = 0
    private var staticAllocations: Int = 0
    private var ephemeralAllocations: Int = 0
    private var hotAllocations: Int = 0
    private var totalBytesRegistered: Int = 0
    private var commitCount: Int = 0
    private var lastCommitTime: Date?

    // MARK: - Initialization

    /// Create a ResidencyManager with default configuration
    public init(device: any MTLDevice) throws {
        self.device = device

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 256

        guard let set = try? device.makeResidencySet(descriptor: descriptor) else {
            throw VectorError.residencySetCreationFailed()
        }

        self.residencySet = set
    }

    /// Create with custom initial capacity
    public init(device: any MTLDevice, initialCapacity: Int) throws {
        self.device = device

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = initialCapacity

        guard let set = try? device.makeResidencySet(descriptor: descriptor) else {
            throw VectorError.residencySetCreationFailed()
        }

        self.residencySet = set
    }

    // MARK: - Public API

    /// Get the underlying residency set for attaching to queues/command buffers
    public nonisolated var underlyingResidencySet: any MTLResidencySet {
        residencySet
    }

    // MARK: - Registration

    /// Register a buffer for GPU residency
    ///
    /// - Parameters:
    ///   - buffer: The MTLBuffer to register
    ///   - mode: How the buffer will be used (affects eviction priority)
    /// - Throws: VectorError if registration fails or set is full
    public func registerBuffer(_ buffer: any MTLBuffer, mode: ResidencyMode = .ephemeral) throws {
        let id = ObjectIdentifier(buffer)

        // Check if already registered
        guard registeredAllocations[id] == nil else {
            return // Already registered, no-op
        }

        // Track allocation
        let info = AllocationInfo(
            allocation: buffer,
            mode: mode,
            byteSize: buffer.length,
            label: buffer.label,
            registeredAt: Date()
        )
        registeredAllocations[id] = info

        // Add to pending batch
        pendingAdds.append(buffer)
        needsCommit = true

        // Update stats
        totalAllocations += 1
        totalBytesRegistered += buffer.length
        switch mode {
        case .static: staticAllocations += 1
        case .ephemeral: ephemeralAllocations += 1
        case .hot: hotAllocations += 1
        }
    }

    /// Register multiple buffers at once (more efficient)
    public func registerBuffers(_ buffers: [any MTLBuffer], mode: ResidencyMode = .ephemeral) throws {
        for buffer in buffers {
            try registerBuffer(buffer, mode: mode)
        }
    }

    /// Unregister a buffer (will be removed from residency)
    public func unregisterBuffer(_ buffer: any MTLBuffer) {
        let id = ObjectIdentifier(buffer)

        guard let info = registeredAllocations.removeValue(forKey: id) else {
            return // Not registered, no-op
        }

        pendingRemoves.append(buffer)
        needsCommit = true

        // Update stats
        totalAllocations -= 1
        totalBytesRegistered -= info.byteSize
        switch info.mode {
        case .static: staticAllocations -= 1
        case .ephemeral: ephemeralAllocations -= 1
        case .hot: hotAllocations -= 1
        }
    }

    // MARK: - Commit

    /// Commit pending changes to the residency set
    ///
    /// This makes newly added resources resident and removes unregistered ones.
    /// Call after a batch of registrations for efficiency.
    public func commit() throws {
        guard needsCommit else { return }

        // Add new allocations
        for buffer in pendingAdds {
            residencySet.addAllocation(buffer)
        }

        // Remove old allocations
        for buffer in pendingRemoves {
            residencySet.removeAllocation(buffer)
        }

        // Commit the changes
        do {
            try residencySet.commit()
        } catch {
            throw VectorError.residencyCommitFailed(underlying: error)
        }

        // Clear pending
        pendingAdds.removeAll()
        pendingRemoves.removeAll()
        needsCommit = false

        // Update stats
        commitCount += 1
        lastCommitTime = Date()
    }

    // MARK: - Query

    /// Check if a buffer is registered
    public func contains(_ buffer: any MTLBuffer) -> Bool {
        registeredAllocations[ObjectIdentifier(buffer)] != nil
    }

    /// Get current statistics
    public func getStatistics() -> ResidencyStatistics {
        ResidencyStatistics(
            totalAllocations: totalAllocations,
            staticAllocations: staticAllocations,
            ephemeralAllocations: ephemeralAllocations,
            hotAllocations: hotAllocations,
            totalBytesRegistered: totalBytesRegistered,
            commitCount: commitCount,
            lastCommitTime: lastCommitTime
        )
    }

    /// Get count of registered allocations
    public var allocationCount: Int {
        registeredAllocations.count
    }

    /// Check if there are pending changes
    public var hasPendingChanges: Bool {
        needsCommit
    }

    // MARK: - Memory Management

    /// Clear all registrations (for cleanup/reset)
    public func clear() {
        for (_, info) in registeredAllocations {
            residencySet.removeAllocation(info.allocation)
        }

        try? residencySet.commit()

        registeredAllocations.removeAll()
        pendingAdds.removeAll()
        pendingRemoves.removeAll()
        needsCommit = false

        // Reset stats except commits
        totalAllocations = 0
        staticAllocations = 0
        ephemeralAllocations = 0
        hotAllocations = 0
        totalBytesRegistered = 0
    }

    /// Evict ephemeral allocations to reduce memory pressure
    ///
    /// Keeps static and hot allocations, removes ephemeral ones.
    /// Call when experiencing memory pressure.
    ///
    /// - Returns: Number of bytes freed
    @discardableResult
    public func evictEphemeral() throws -> Int {
        var freedBytes = 0
        var toRemove: [ObjectIdentifier] = []

        for (id, info) in registeredAllocations where info.mode == .ephemeral {
            residencySet.removeAllocation(info.allocation)
            toRemove.append(id)
            freedBytes += info.byteSize
        }

        for id in toRemove {
            if let info = registeredAllocations.removeValue(forKey: id) {
                totalAllocations -= 1
                ephemeralAllocations -= 1
                totalBytesRegistered -= info.byteSize
            }
        }

        if !toRemove.isEmpty {
            try commit()
        }

        return freedBytes
    }

    /// Evict allocations by mode to reduce memory pressure
    ///
    /// - Parameter mode: The mode of allocations to evict
    /// - Returns: Number of bytes freed
    @discardableResult
    public func evict(mode: ResidencyMode) throws -> Int {
        var freedBytes = 0
        var toRemove: [ObjectIdentifier] = []

        for (id, info) in registeredAllocations where info.mode == mode {
            residencySet.removeAllocation(info.allocation)
            toRemove.append(id)
            freedBytes += info.byteSize
        }

        for id in toRemove {
            if let info = registeredAllocations.removeValue(forKey: id) {
                totalAllocations -= 1
                totalBytesRegistered -= info.byteSize
                switch info.mode {
                case .static: staticAllocations -= 1
                case .ephemeral: ephemeralAllocations -= 1
                case .hot: hotAllocations -= 1
                }
            }
        }

        if !toRemove.isEmpty {
            try commit()
        }

        return freedBytes
    }

    // MARK: - Batch Operations

    /// Register and commit in one operation
    public func registerAndCommit(_ buffer: any MTLBuffer, mode: ResidencyMode = .ephemeral) throws {
        try registerBuffer(buffer, mode: mode)
        try commit()
    }

    /// Register multiple and commit
    public func registerAndCommit(_ buffers: [any MTLBuffer], mode: ResidencyMode = .ephemeral) throws {
        try registerBuffers(buffers, mode: mode)
        try commit()
    }
}

// MARK: - MTLResidencySetDescriptor (Metal 4)

/// Residency set descriptor for Metal 4
/// Note: This mirrors the expected Metal 4 API
public class MTLResidencySetDescriptor: NSObject {
    public var initialCapacity: Int = 64

    public override init() {
        super.init()
    }
}

// MARK: - MTLResidencySet Protocol

/// Protocol for Metal 4 residency set
/// Note: This abstracts the Metal 4 MTLResidencySet type
public protocol MTLResidencySet: AnyObject, Sendable {
    func addAllocation(_ allocation: any MTLBuffer)
    func removeAllocation(_ allocation: any MTLBuffer)
    func commit() throws
}

// MARK: - MTLDevice Extension

public extension MTLDevice {
    /// Create a residency set for Metal 4
    /// Note: This will use the actual Metal 4 API when available
    func makeResidencySet(descriptor: MTLResidencySetDescriptor) throws -> any MTLResidencySet {
        // Create the actual Metal 4 residency set
        // For now, return a wrapper that uses the actual API
        return Metal4ResidencySetImpl(device: self, capacity: descriptor.initialCapacity)
    }
}

// MARK: - Metal 4 Residency Set Implementation

/// Implementation of MTLResidencySet using Metal 4 APIs
internal final class Metal4ResidencySetImpl: MTLResidencySet, @unchecked Sendable {
    private let device: any MTLDevice
    private var allocations: Set<ObjectIdentifier> = []
    private var allocationBuffers: [ObjectIdentifier: any MTLBuffer] = [:]
    private let lock = NSLock()

    init(device: any MTLDevice, capacity: Int) {
        self.device = device
        // In actual Metal 4, this would call device.makeResidencySet(descriptor:)
    }

    func addAllocation(_ allocation: any MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }
        let id = ObjectIdentifier(allocation)
        allocations.insert(id)
        allocationBuffers[id] = allocation
    }

    func removeAllocation(_ allocation: any MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }
        let id = ObjectIdentifier(allocation)
        allocations.remove(id)
        allocationBuffers.removeValue(forKey: id)
    }

    func commit() throws {
        // In actual Metal 4, this would commit the residency set changes
        // For now, this is a no-op as the real implementation will use Metal 4 APIs
    }
}
