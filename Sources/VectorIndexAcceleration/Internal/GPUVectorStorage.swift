//
//  GPUVectorStorage.swift
//  VectorIndexAcceleration
//
//  GPU buffer management for vector storage.
//
//  Manages a contiguous MTLBuffer containing vector data with:
//  - Pre-allocated capacity with 2x growth strategy
//  - Direct GPU writes via contents()
//  - Efficient batch operations
//

import Foundation
@preconcurrency import Metal
import VectorCore
import VectorAccelerate

// MARK: - GPU Vector Storage

/// Manages GPU buffer for vector storage.
///
/// Provides efficient GPU memory management for vector data with:
/// - Pre-allocated capacity to minimize reallocations
/// - Automatic 2x growth when capacity is exceeded
/// - Direct memory access for fast writes
/// - Compaction support for removing deleted vectors
///
/// ## Memory Layout
/// Vectors are stored contiguously as `[slot0_dim0, slot0_dim1, ..., slot1_dim0, ...]`
/// Each slot occupies `dimension * sizeof(Float)` bytes.
///
/// ## Thread Safety
/// This class is not thread-safe. Access should be synchronized by the owning actor.
final class GPUVectorStorage: @unchecked Sendable {

    // MARK: - Properties

    /// The underlying Metal buffer.
    private(set) var buffer: (any MTLBuffer)?

    /// Metal device for buffer allocation.
    private let device: any MTLDevice

    /// Vector dimension.
    let dimension: Int

    /// Current capacity (maximum slots).
    private(set) var capacity: Int

    /// Number of slots currently written (may include deleted).
    private(set) var allocatedSlots: Int = 0

    /// Bytes per vector slot.
    var bytesPerSlot: Int {
        dimension * MemoryLayout<Float>.size
    }

    /// Total bytes allocated for vector buffer.
    var allocatedBytes: Int {
        capacity * bytesPerSlot
    }

    /// Bytes currently in use (allocated slots).
    var usedBytes: Int {
        allocatedSlots * bytesPerSlot
    }

    // MARK: - Initialization

    /// Create GPU vector storage.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation
    ///   - dimension: Vector dimension
    ///   - capacity: Initial capacity (number of vectors)
    /// - Throws: `IndexAccelerationError` if buffer allocation fails
    init(device: any MTLDevice, dimension: Int, capacity: Int) throws {
        self.device = device
        self.dimension = dimension
        self.capacity = capacity

        try allocateBuffer()
    }

    // MARK: - Buffer Management

    /// Allocate or reallocate the GPU buffer.
    private func allocateBuffer() throws {
        let bufferSize = capacity * bytesPerSlot
        guard bufferSize > 0 else {
            buffer = nil
            return
        }

        guard let newBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw IndexAccelerationError.gpuResourceCreationFailed(
                index: "GPUVectorStorage",
                reason: "Failed to allocate \(bufferSize) bytes for vector buffer"
            )
        }
        newBuffer.label = "VectorIndexAcceleration.VectorStorage"
        buffer = newBuffer
    }

    /// Grow the buffer to accommodate more vectors.
    ///
    /// Uses a 2x growth strategy to minimize reallocations.
    ///
    /// - Throws: `IndexAccelerationError` if reallocation fails
    func grow() throws {
        let newCapacity = max(capacity * 2, 1000)
        let newBufferSize = newCapacity * bytesPerSlot

        guard let newBuffer = device.makeBuffer(length: newBufferSize, options: .storageModeShared) else {
            throw IndexAccelerationError.bufferError(
                operation: "grow",
                reason: "Failed to allocate \(newBufferSize) bytes"
            )
        }
        newBuffer.label = "VectorIndexAcceleration.VectorStorage"

        // Copy existing data
        if let oldBuffer = buffer, allocatedSlots > 0 {
            let copySize = allocatedSlots * bytesPerSlot
            memcpy(newBuffer.contents(), oldBuffer.contents(), copySize)
        }

        buffer = newBuffer
        capacity = newCapacity
    }

    /// Ensure capacity for at least the specified number of slots.
    ///
    /// - Parameter requiredCapacity: Minimum required capacity
    /// - Throws: `IndexAccelerationError` if growth fails
    func ensureCapacity(_ requiredCapacity: Int) throws {
        while capacity < requiredCapacity {
            try grow()
        }
    }

    // MARK: - Vector Operations

    /// Write a vector to a specific slot.
    ///
    /// - Parameters:
    ///   - vector: Vector data to write
    ///   - slotIndex: Slot index to write to
    /// - Throws: `IndexAccelerationError` if buffer not initialized or index out of bounds
    func writeVector(_ vector: [Float], at slotIndex: Int) throws {
        guard let buffer = buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "writeVector")
        }

        guard vector.count == dimension else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: dimension,
                got: vector.count
            )
        }

        guard slotIndex >= 0 && slotIndex < capacity else {
            throw IndexAccelerationError.bufferError(
                operation: "writeVector",
                reason: "Slot index \(slotIndex) out of bounds (capacity: \(capacity))"
            )
        }

        let offset = slotIndex * bytesPerSlot
        let ptr = buffer.contents().advanced(by: offset)
        _ = vector.withUnsafeBytes { src in
            memcpy(ptr, src.baseAddress!, bytesPerSlot)
        }

        // Update allocated slots if this is a new slot
        if slotIndex >= allocatedSlots {
            allocatedSlots = slotIndex + 1
        }
    }

    /// Write multiple vectors starting at a specific slot.
    ///
    /// - Parameters:
    ///   - vectors: Array of vectors to write
    ///   - startSlot: Starting slot index
    /// - Throws: `IndexAccelerationError` if buffer not initialized or validation fails
    func writeVectors(_ vectors: [[Float]], startingAt startSlot: Int) throws {
        guard let buffer = buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "writeVectors")
        }

        guard startSlot + vectors.count <= capacity else {
            throw IndexAccelerationError.bufferError(
                operation: "writeVectors",
                reason: "Write would exceed capacity"
            )
        }

        for (i, vector) in vectors.enumerated() {
            guard vector.count == dimension else {
                throw IndexAccelerationError.dimensionMismatch(
                    expected: dimension,
                    got: vector.count
                )
            }

            let slotIndex = startSlot + i
            let offset = slotIndex * bytesPerSlot
            let ptr = buffer.contents().advanced(by: offset)
            _ = vector.withUnsafeBytes { src in
                memcpy(ptr, src.baseAddress!, bytesPerSlot)
            }
        }

        let endSlot = startSlot + vectors.count
        if endSlot > allocatedSlots {
            allocatedSlots = endSlot
        }
    }

    /// Read a vector from a specific slot.
    ///
    /// - Parameter slotIndex: Slot index to read from
    /// - Returns: Vector data
    /// - Throws: `IndexAccelerationError` if buffer not initialized or index out of bounds
    func readVector(at slotIndex: Int) throws -> [Float] {
        guard let buffer = buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "readVector")
        }

        guard slotIndex >= 0 && slotIndex < allocatedSlots else {
            throw IndexAccelerationError.bufferError(
                operation: "readVector",
                reason: "Slot index \(slotIndex) out of bounds (allocated: \(allocatedSlots))"
            )
        }

        let offset = slotIndex * bytesPerSlot
        let ptr = buffer.contents().advanced(by: offset)
            .bindMemory(to: Float.self, capacity: dimension)

        return Array(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    // MARK: - Compaction

    /// Compact the storage by removing deleted slots.
    ///
    /// - Parameter keepMask: Boolean array where `true` means keep the slot
    /// - Returns: Mapping from old slot indices to new slot indices
    /// - Throws: `IndexAccelerationError` if compaction fails
    @discardableResult
    func compact(keepMask: [Bool]) throws -> [Int: Int] {
        guard let oldBuffer = buffer else {
            return [:]
        }

        // Count vectors to keep
        var keptCount = 0
        for keep in keepMask.prefix(allocatedSlots) {
            if keep { keptCount += 1 }
        }

        guard keptCount > 0 else {
            // All deleted - reset to empty
            allocatedSlots = 0
            return [:]
        }

        // Allocate new buffer
        let newBufferSize = max(keptCount, capacity / 2) * bytesPerSlot
        guard let newBuffer = device.makeBuffer(length: newBufferSize, options: .storageModeShared) else {
            throw IndexAccelerationError.bufferError(
                operation: "compact",
                reason: "Failed to allocate compacted buffer"
            )
        }
        newBuffer.label = "VectorIndexAcceleration.VectorStorage.Compacted"

        // Copy kept vectors and build mapping
        var indexMapping: [Int: Int] = [:]
        var newIndex = 0

        for oldIndex in 0..<min(allocatedSlots, keepMask.count) {
            if keepMask[oldIndex] {
                // Copy vector
                let srcOffset = oldIndex * bytesPerSlot
                let dstOffset = newIndex * bytesPerSlot
                memcpy(
                    newBuffer.contents().advanced(by: dstOffset),
                    oldBuffer.contents().advanced(by: srcOffset),
                    bytesPerSlot
                )
                indexMapping[oldIndex] = newIndex
                newIndex += 1
            }
        }

        // Update state
        buffer = newBuffer
        allocatedSlots = newIndex
        capacity = max(keptCount, capacity / 2)

        return indexMapping
    }

    /// Create a compacted copy with only the specified slots.
    ///
    /// Does not modify this storage; returns new data.
    ///
    /// - Parameter keepMask: Boolean array where `true` means include the slot
    /// - Returns: Flattened Float array of kept vectors
    /// - Throws: `IndexAccelerationError` if read fails
    func extractVectors(keepMask: [Bool]) throws -> [Float] {
        guard buffer != nil else {
            return []
        }

        var result: [Float] = []
        let keptCount = keepMask.prefix(allocatedSlots).filter { $0 }.count
        result.reserveCapacity(keptCount * dimension)

        for i in 0..<min(allocatedSlots, keepMask.count) {
            if keepMask[i] {
                let vector = try readVector(at: i)
                result.append(contentsOf: vector)
            }
        }

        return result
    }

    // MARK: - Reset

    /// Reset storage to empty state.
    ///
    /// Keeps the allocated buffer but marks all slots as unused.
    func reset() {
        allocatedSlots = 0
    }

    /// Release the GPU buffer.
    func release() {
        buffer = nil
        allocatedSlots = 0
    }
}
