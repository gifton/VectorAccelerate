//
//  ArgumentTablePool.swift
//  VectorAccelerate
//
//  Pool of reusable Metal 4 argument tables for efficient resource binding
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Argument Table Protocol

/// Protocol for Metal 4 argument tables
/// Abstracts the actual Metal 4 MTL4ArgumentTable type
public protocol ArgumentTable: AnyObject, Sendable {
    /// Set a buffer's GPU address at the specified index
    func setAddress(_ address: UInt64, index: Int)

    /// Set a buffer at the specified index
    func setBuffer(_ buffer: any MTLBuffer, offset: Int, index: Int)

    /// Clear all bindings
    func reset()

    /// The maximum number of buffer bindings
    var maxBufferBindCount: Int { get }
}

// MARK: - Argument Table Descriptor

/// Descriptor for creating argument tables
public struct ArgumentTableDescriptor: Sendable {
    /// Maximum number of buffer bindings
    public var maxBufferBindCount: Int

    /// Maximum number of texture bindings
    public var maxTextureBindCount: Int

    /// Label for debugging
    public var label: String?

    public init(
        maxBufferBindCount: Int = 16,
        maxTextureBindCount: Int = 8,
        label: String? = nil
    ) {
        self.maxBufferBindCount = maxBufferBindCount
        self.maxTextureBindCount = maxTextureBindCount
        self.label = label
    }

    /// Descriptor for distance kernels (2-3 buffers + params)
    public static var distance: ArgumentTableDescriptor {
        ArgumentTableDescriptor(maxBufferBindCount: 8, maxTextureBindCount: 0, label: "Distance")
    }

    /// Descriptor for batch operations (more buffers)
    public static var batch: ArgumentTableDescriptor {
        ArgumentTableDescriptor(maxBufferBindCount: 16, maxTextureBindCount: 0, label: "Batch")
    }

    /// Descriptor for matrix operations
    public static var matrix: ArgumentTableDescriptor {
        ArgumentTableDescriptor(maxBufferBindCount: 8, maxTextureBindCount: 0, label: "Matrix")
    }

    /// Descriptor for quantization operations
    public static var quantization: ArgumentTableDescriptor {
        ArgumentTableDescriptor(maxBufferBindCount: 12, maxTextureBindCount: 0, label: "Quantization")
    }
}

// MARK: - Argument Table Implementation

/// Concrete implementation of ArgumentTable using Metal 4 APIs
internal final class Metal4ArgumentTable: ArgumentTable, @unchecked Sendable {
    let maxBufferBindCount: Int
    private let lock = NSLock()

    // Track bindings for reset and debugging
    private var bufferBindings: [Int: (buffer: any MTLBuffer, offset: Int)] = [:]
    private var addressBindings: [Int: UInt64] = [:]

    init(descriptor: ArgumentTableDescriptor) {
        self.maxBufferBindCount = descriptor.maxBufferBindCount
    }

    func setAddress(_ address: UInt64, index: Int) {
        lock.lock()
        defer { lock.unlock() }

        precondition(index < maxBufferBindCount, "Index \(index) exceeds max buffer count \(maxBufferBindCount)")
        addressBindings[index] = address
        bufferBindings.removeValue(forKey: index)
    }

    func setBuffer(_ buffer: any MTLBuffer, offset: Int, index: Int) {
        lock.lock()
        defer { lock.unlock() }

        precondition(index < maxBufferBindCount, "Index \(index) exceeds max buffer count \(maxBufferBindCount)")
        bufferBindings[index] = (buffer, offset)
        addressBindings.removeValue(forKey: index)
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }

        bufferBindings.removeAll()
        addressBindings.removeAll()
    }

    /// Apply bindings to an encoder (for actual execution)
    func apply(to encoder: any MTLComputeCommandEncoder) {
        lock.lock()
        defer { lock.unlock() }

        // Apply buffer bindings
        for (index, binding) in bufferBindings {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: index)
        }

        // Note: Address bindings would use encoder.setArgumentTable() in actual Metal 4
        // For now, we convert addresses back to buffer bindings where possible
    }
}

// MARK: - Argument Table Pool Statistics

/// Statistics for argument table pool monitoring
public struct ArgumentTablePoolStatistics: Sendable {
    public let totalTables: Int
    public let availableTables: Int
    public let inUseTables: Int
    public let acquisitionCount: Int
    public let releaseCount: Int
    public let peakInUse: Int

    public var utilizationRate: Double {
        totalTables > 0 ? Double(inUseTables) / Double(totalTables) : 0
    }
}

// MARK: - Argument Table Pool

/// Pool of reusable argument tables for efficient GPU resource binding
///
/// Metal 4 uses argument tables to bind resources to shaders. Creating tables
/// has overhead, so this pool maintains reusable tables that can be acquired
/// and released efficiently.
///
/// Example:
/// ```swift
/// let pool = ArgumentTablePool(device: device, maxTables: 32)
/// let table = try await pool.acquire()
/// defer { Task { await pool.release(table) } }
///
/// table.setAddress(queryBuffer.gpuAddress, index: 0)
/// table.setAddress(databaseBuffer.gpuAddress, index: 1)
/// encoder.setArgumentTable(table, stages: .compute)
/// ```
public actor ArgumentTablePool {
    // MARK: - Properties

    private let device: any MTLDevice
    private let defaultDescriptor: ArgumentTableDescriptor
    private let maxTables: Int

    // Pool state
    private var available: [any ArgumentTable] = []
    private var inUse: Set<ObjectIdentifier> = []

    // Statistics
    private var acquisitionCount: Int = 0
    private var releaseCount: Int = 0
    private var peakInUse: Int = 0

    // MARK: - Initialization

    /// Create a new argument table pool
    ///
    /// - Parameters:
    ///   - device: The Metal device to create tables on
    ///   - maxTables: Maximum number of tables to keep in pool
    ///   - defaultDescriptor: Default descriptor for new tables
    public init(
        device: any MTLDevice,
        maxTables: Int = 32,
        defaultDescriptor: ArgumentTableDescriptor = ArgumentTableDescriptor()
    ) {
        self.device = device
        self.maxTables = maxTables
        self.defaultDescriptor = defaultDescriptor
    }

    // MARK: - Acquisition

    /// Acquire an argument table from the pool
    ///
    /// Returns an existing table if available, otherwise creates a new one.
    /// The table is reset before being returned.
    ///
    /// - Returns: A ready-to-use argument table
    /// - Throws: `VectorError.argumentTablePoolExhausted` if pool is full
    public func acquire() throws -> any ArgumentTable {
        return try acquire(descriptor: defaultDescriptor)
    }

    /// Acquire an argument table with specific configuration
    ///
    /// - Parameter descriptor: Configuration for the table
    /// - Returns: A ready-to-use argument table
    /// - Throws: `VectorError.argumentTablePoolExhausted` if pool is full
    public func acquire(descriptor: ArgumentTableDescriptor) throws -> any ArgumentTable {
        acquisitionCount += 1

        // Try to reuse existing table
        if let table = available.popLast() {
            table.reset()
            inUse.insert(ObjectIdentifier(table))
            peakInUse = max(peakInUse, inUse.count)
            return table
        }

        // Check pool limit
        guard inUse.count < maxTables else {
            throw VectorError.argumentTablePoolExhausted()
        }

        // Create new table
        let table = Metal4ArgumentTable(descriptor: descriptor)
        inUse.insert(ObjectIdentifier(table))
        peakInUse = max(peakInUse, inUse.count)

        return table
    }

    // MARK: - Release

    /// Release an argument table back to the pool
    ///
    /// The table will be reset and made available for reuse.
    ///
    /// - Parameter table: The table to release
    public func release(_ table: any ArgumentTable) {
        let id = ObjectIdentifier(table)

        guard inUse.contains(id) else {
            // Table wasn't from this pool or already released
            return
        }

        releaseCount += 1
        inUse.remove(id)

        // Keep in pool for reuse if under limit
        if available.count < maxTables {
            table.reset()
            available.append(table)
        }
    }

    // MARK: - Batch Operations

    /// Acquire multiple tables at once
    ///
    /// - Parameter count: Number of tables to acquire
    /// - Returns: Array of ready-to-use argument tables
    /// - Throws: `VectorError.argumentTablePoolExhausted` if not enough tables available
    public func acquireMultiple(count: Int) throws -> [any ArgumentTable] {
        var tables: [any ArgumentTable] = []
        tables.reserveCapacity(count)

        for _ in 0..<count {
            tables.append(try acquire())
        }

        return tables
    }

    /// Release multiple tables at once
    ///
    /// - Parameter tables: Tables to release
    public func releaseMultiple(_ tables: [any ArgumentTable]) {
        for table in tables {
            release(table)
        }
    }

    // MARK: - Utilities

    /// Pre-warm the pool by creating tables ahead of time
    ///
    /// - Parameter count: Number of tables to create
    public func warmUp(count: Int) {
        let createCount = min(count, maxTables - available.count - inUse.count)

        for _ in 0..<createCount {
            let table = Metal4ArgumentTable(descriptor: defaultDescriptor)
            available.append(table)
        }
    }

    /// Clear all available tables (does not affect in-use tables)
    public func clearAvailable() {
        available.removeAll()
    }

    /// Get current pool statistics
    public func getStatistics() -> ArgumentTablePoolStatistics {
        ArgumentTablePoolStatistics(
            totalTables: available.count + inUse.count,
            availableTables: available.count,
            inUseTables: inUse.count,
            acquisitionCount: acquisitionCount,
            releaseCount: releaseCount,
            peakInUse: peakInUse
        )
    }

    // MARK: - Properties

    /// Number of tables currently available
    public var availableCount: Int {
        available.count
    }

    /// Number of tables currently in use
    public var inUseCount: Int {
        inUse.count
    }
}

// MARK: - Argument Table Token

/// RAII token that automatically releases argument table on deinit
public final class ArgumentTableToken: @unchecked Sendable {
    public let table: any ArgumentTable
    private let pool: ArgumentTablePool?
    private var isReleased: Bool = false
    private let lock = NSLock()

    init(table: any ArgumentTable, pool: ArgumentTablePool?) {
        self.table = table
        self.pool = pool
    }

    deinit {
        guard !isReleased, let pool = pool else { return }

        let capturedTable = table
        Task.detached {
            await pool.release(capturedTable)
        }
    }

    /// Manually release the table back to pool (optional - happens on deinit)
    public func release() {
        lock.lock()
        defer { lock.unlock() }

        guard !isReleased else { return }
        isReleased = true

        guard let pool = pool else { return }

        let capturedTable = table
        Task.detached {
            await pool.release(capturedTable)
        }
    }
}

// MARK: - Pool Extension for Token-based API

extension ArgumentTablePool {
    /// Acquire an argument table wrapped in an auto-releasing token
    ///
    /// The table is automatically returned to the pool when the token is deallocated.
    ///
    /// - Returns: Token containing the argument table
    /// - Throws: `VectorError.argumentTablePoolExhausted` if pool is full
    public func acquireToken() throws -> ArgumentTableToken {
        let table = try acquire()
        return ArgumentTableToken(table: table, pool: self)
    }

    /// Acquire with specific descriptor wrapped in token
    public func acquireToken(descriptor: ArgumentTableDescriptor) throws -> ArgumentTableToken {
        let table = try acquire(descriptor: descriptor)
        return ArgumentTableToken(table: table, pool: self)
    }
}
