//
//  HandleAllocator.swift
//  VectorAccelerate
//
//  Stable handle allocator with indirection.
//
//  - User-facing handles are stable IDs (never change for the lifetime of a vector)
//  - Internally, vectors live in GPU storage slots which may be compacted/moved
//  - Two arrays provide O(1) mapping in both directions:
//      stableID -> currentSlot
//      currentSlot -> stableID
//

import Foundation

/// Manages `VectorHandle` creation, lookup, and deletion.
///
/// This allocator implements stable handles (P0.8): handles remain valid across
/// `compact()` by updating an internal indirection table.
///
/// Important invariants:
/// - Stable IDs are monotonically increasing and never reused.
/// - Slots are allocated sequentially and may be renumbered during compaction.
/// - Deleted handles are tombstoned and will never become valid again.
final class HandleAllocator: @unchecked Sendable {

    // MARK: - Sentinel

    /// Tombstone value used for both `stableIDToSlot` and `slotToStableID`.
    @usableFromInline
    static let tombstone: UInt32 = 0xFFFF_FFFF

    // MARK: - State

    /// Mapping: stableID -> current slot.
    ///
    /// Index into this array is the stable ID.
    @usableFromInline
    var stableIDToSlot: [UInt32]

    /// Mapping: current slot -> stableID.
    ///
    /// Index into this array is the current slot.
    @usableFromInline
    var slotToStableID: [UInt32]

    /// Next slot index to allocate (sequential append-only allocation).
    @usableFromInline
    var nextSlot: UInt32

    /// Number of active (non-deleted) handles.
    @usableFromInline
    private(set) var occupiedCount: Int

    // MARK: - Init

    init(initialSlotCapacity: Int = 1024) {
        self.stableIDToSlot = []
        self.slotToStableID = Array(repeating: Self.tombstone, count: max(0, initialSlotCapacity))
        self.nextSlot = 0
        self.occupiedCount = 0
    }

    // MARK: - Capacity

    /// Total slots that have been allocated historically (including deleted).
    ///
    /// This is primarily useful for diagnostics.
    var totalSlotsAllocated: Int {
        Int(nextSlot)
    }

    /// Current slot capacity of the reverse map.
    var slotCapacity: Int {
        slotToStableID.count
    }

    private func ensureSlotCapacity(_ required: Int) {
        guard required > slotToStableID.count else { return }
        slotToStableID.append(contentsOf: repeatElement(Self.tombstone, count: required - slotToStableID.count))
    }

    // MARK: - Allocation

    /// Allocate a new handle and slot.
    func allocate() -> VectorHandle {
        let slot = nextSlot
        nextSlot &+= 1
        ensureSlotCapacity(Int(nextSlot))

        let stableID = UInt32(stableIDToSlot.count)
        stableIDToSlot.append(slot)
        slotToStableID[Int(slot)] = stableID

        occupiedCount += 1
        return VectorHandle(stableID: stableID)
    }

    /// Allocate multiple handles in a contiguous slot range.
    func allocate(count: Int) -> [VectorHandle] {
        guard count > 0 else { return [] }

        let startSlot = nextSlot
        nextSlot &+= UInt32(count)
        ensureSlotCapacity(Int(nextSlot))

        let startStableID = UInt32(stableIDToSlot.count)
        stableIDToSlot.reserveCapacity(stableIDToSlot.count + count)

        var handles: [VectorHandle] = []
        handles.reserveCapacity(count)

        for i in 0..<count {
            let stableID = startStableID &+ UInt32(i)
            let slot = startSlot &+ UInt32(i)

            stableIDToSlot.append(slot)
            slotToStableID[Int(slot)] = stableID
            handles.append(VectorHandle(stableID: stableID))
        }

        occupiedCount += count
        return handles
    }

    // MARK: - Lookup

    /// Validate that the handle exists and has not been deleted.
    @inlinable
    func validate(_ handle: VectorHandle) -> Bool {
        guard handle.isValid else { return false }
        let id = handle.stableID
        guard id < stableIDToSlot.count else { return false }
        let slot = stableIDToSlot[Int(id)]
        if slot == Self.tombstone { return false }
        guard Int(slot) < slotToStableID.count else { return false }
        return slotToStableID[Int(slot)] == id
    }

    /// Resolve a handle to the current storage slot.
    ///
    /// - Returns: Slot index if the handle exists and is active.
    @inlinable
    func slot(for handle: VectorHandle) -> UInt32? {
        guard handle.isValid else { return nil }
        let id = handle.stableID
        guard id < stableIDToSlot.count else { return nil }
        let slot = stableIDToSlot[Int(id)]
        if slot == Self.tombstone { return nil }
        return slot
    }

    /// Get the handle corresponding to a storage slot.
    ///
    /// Used when mapping GPU search results (slot indices) back to user handles.
    @inlinable
    func handle(for slotIndex: UInt32) -> VectorHandle? {
        guard Int(slotIndex) < slotToStableID.count else { return nil }
        let id = slotToStableID[Int(slotIndex)]
        guard id != Self.tombstone else { return nil }
        return VectorHandle(stableID: id)
    }

    // MARK: - Deletion

    /// Mark a handle as deleted (tombstone it) and return its last-known slot.
    ///
    /// - Returns: The slot that was freed, or nil if the handle was already invalid.
    @discardableResult
    func markDeleted(_ handle: VectorHandle) -> UInt32? {
        guard handle.isValid else { return nil }
        let id = handle.stableID
        guard id < stableIDToSlot.count else { return nil }
        let slot = stableIDToSlot[Int(id)]
        guard slot != Self.tombstone else { return nil }

        stableIDToSlot[Int(id)] = Self.tombstone
        if Int(slot) < slotToStableID.count {
            slotToStableID[Int(slot)] = Self.tombstone
        }

        occupiedCount = max(0, occupiedCount - 1)
        return slot
    }

    // MARK: - Compaction

    /// Apply a slot renumbering produced by `GPUVectorStorage.compact`.
    ///
    /// Stable IDs remain unchanged; only the indirection tables are updated.
    func applyCompaction(slotMapping: [UInt32: UInt32], newSlotCount: Int) {
        // Build new reverse map
        var newSlotToStableID = Array(repeating: Self.tombstone, count: max(0, newSlotCount))

        for (oldSlot, newSlot) in slotMapping {
            guard Int(oldSlot) < slotToStableID.count else { continue }
            let id = slotToStableID[Int(oldSlot)]
            guard id != Self.tombstone else { continue }

            if Int(newSlot) < newSlotToStableID.count {
                newSlotToStableID[Int(newSlot)] = id
            }

            // Update forward map
            if Int(id) < stableIDToSlot.count {
                stableIDToSlot[Int(id)] = newSlot
            }
        }

        slotToStableID = newSlotToStableID
        nextSlot = UInt32(newSlotCount)
        occupiedCount = newSlotCount
    }

    // MARK: - Reset

    func reset() {
        stableIDToSlot.removeAll(keepingCapacity: true)
        slotToStableID.removeAll(keepingCapacity: true)
        nextSlot = 0
        occupiedCount = 0
    }
}
