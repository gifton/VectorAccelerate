//
//  HandleAllocator.swift
//  VectorIndexAcceleration
//
//  Handle allocation and lifecycle management with generation-based stale detection.
//
//  Manages VectorHandle creation and validation:
//  - Monotonic slot allocation
//  - Generation counters for stale handle detection
//  - Slot recycling after compaction
//

import Foundation

// MARK: - Slot Info

/// Internal state for a single slot.
struct SlotInfo: Sendable {
    /// Generation counter, incremented when slot is reused.
    var generation: UInt16 = 0

    /// Whether the slot is currently occupied by an active vector.
    var isOccupied: Bool = false
}

// MARK: - Handle Allocator

/// Manages handle allocation and validation.
///
/// Provides:
/// - Monotonic slot allocation for new vectors
/// - Generation-based stale handle detection
/// - Slot state tracking (occupied/deleted)
/// - Compaction with generation increment for reused slots
///
/// ## Generation-Based Stability
/// Each slot has a generation counter that increments when the slot is reused
/// after deletion and compaction. This allows detection of stale handles:
///
/// ```swift
/// let handle = allocator.allocate()  // generation = 0
/// allocator.markDeleted(handle)
/// allocator.compact(...)             // slot may be reused
/// allocator.validate(handle)         // false - generation mismatch
/// ```
///
/// ## Thread Safety
/// This class is not thread-safe. Access should be synchronized by the owning actor.
final class HandleAllocator: @unchecked Sendable {

    // MARK: - Properties

    /// Slot information array.
    private var slots: [SlotInfo] = []

    /// Next slot to allocate (monotonically increasing until compaction).
    private var nextSlot: UInt32 = 0

    /// Number of currently occupied slots.
    private(set) var occupiedCount: Int = 0

    /// Total slots ever allocated (including deleted).
    var totalSlots: Int {
        Int(nextSlot)
    }

    /// Number of slots currently tracked.
    var slotCapacity: Int {
        slots.count
    }

    // MARK: - Initialization

    /// Create a handle allocator.
    ///
    /// - Parameter initialCapacity: Initial slot array capacity (optimization hint)
    init(initialCapacity: Int = 1000) {
        slots.reserveCapacity(initialCapacity)
    }

    // MARK: - Allocation

    /// Allocate a new handle.
    ///
    /// - Returns: A new valid handle for the allocated slot
    func allocate() -> VectorHandle {
        let slotIndex = nextSlot
        nextSlot += 1

        // Ensure slot array is large enough
        ensureCapacity(for: Int(slotIndex))

        // Get or create slot info
        let generation = slots[Int(slotIndex)].generation
        slots[Int(slotIndex)].isOccupied = true
        occupiedCount += 1

        return VectorHandle(index: slotIndex, generation: generation)
    }

    /// Allocate multiple handles.
    ///
    /// - Parameter count: Number of handles to allocate
    /// - Returns: Array of new valid handles
    func allocate(count: Int) -> [VectorHandle] {
        guard count > 0 else { return [] }

        let startSlot = nextSlot
        nextSlot += UInt32(count)

        // Ensure capacity
        ensureCapacity(for: Int(nextSlot) - 1)

        var handles: [VectorHandle] = []
        handles.reserveCapacity(count)

        for i in 0..<count {
            let slotIndex = startSlot + UInt32(i)
            let generation = slots[Int(slotIndex)].generation
            slots[Int(slotIndex)].isOccupied = true
            handles.append(VectorHandle(index: slotIndex, generation: generation))
        }

        occupiedCount += count
        return handles
    }

    // MARK: - Validation

    /// Validate a handle.
    ///
    /// A handle is valid if:
    /// 1. It's not the invalid sentinel
    /// 2. Its slot index is within bounds
    /// 3. Its generation matches the current slot generation
    /// 4. The slot is currently occupied
    ///
    /// - Parameter handle: Handle to validate
    /// - Returns: true if handle is valid
    func validate(_ handle: VectorHandle) -> Bool {
        guard handle.isValid else { return false }
        guard handle.index < slots.count else { return false }

        let slot = slots[Int(handle.index)]
        return slot.isOccupied && slot.generation == handle.generation
    }

    /// Get the current generation for a slot.
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Current generation, or nil if slot doesn't exist
    func generation(for slotIndex: UInt32) -> UInt16? {
        guard slotIndex < slots.count else { return nil }
        return slots[Int(slotIndex)].generation
    }

    /// Check if a slot is occupied.
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: true if slot is occupied
    func isOccupied(_ slotIndex: UInt32) -> Bool {
        guard slotIndex < slots.count else { return false }
        return slots[Int(slotIndex)].isOccupied
    }

    // MARK: - Deletion

    /// Mark a handle as deleted.
    ///
    /// The slot remains allocated but is marked as unoccupied.
    /// The generation is NOT incremented until compaction reuses the slot.
    ///
    /// - Parameter handle: Handle to mark as deleted
    /// - Returns: true if successfully deleted, false if handle was invalid
    @discardableResult
    func markDeleted(_ handle: VectorHandle) -> Bool {
        guard validate(handle) else { return false }

        slots[Int(handle.index)].isOccupied = false
        occupiedCount -= 1
        return true
    }

    /// Mark a slot as deleted by index.
    ///
    /// - Parameter slotIndex: Slot index to mark as deleted
    /// - Returns: true if successfully deleted
    @discardableResult
    func markDeleted(slotIndex: UInt32) -> Bool {
        guard slotIndex < slots.count else { return false }
        guard slots[Int(slotIndex)].isOccupied else { return false }

        slots[Int(slotIndex)].isOccupied = false
        occupiedCount -= 1
        return true
    }

    // MARK: - Compaction

    /// Result of a compaction operation.
    struct CompactionResult: Sendable {
        /// Mapping from old slot indices to new slot indices.
        let indexMapping: [UInt32: UInt32]

        /// New handles for the compacted slots (in new index order).
        let newHandles: [VectorHandle]

        /// Number of slots after compaction.
        let newSlotCount: Int
    }

    /// Compact the allocator, removing unoccupied slots.
    ///
    /// This operation:
    /// 1. Creates a new slot array with only occupied slots
    /// 2. Increments generation for all reused slot positions
    /// 3. Returns mapping from old to new indices
    ///
    /// After compaction, old handles are invalid (generation mismatch).
    ///
    /// - Returns: Compaction result with index mapping and new handles
    func compact() -> CompactionResult {
        var indexMapping: [UInt32: UInt32] = [:]
        var newSlots: [SlotInfo] = []
        var newHandles: [VectorHandle] = []

        var newIndex: UInt32 = 0
        for oldIndex in 0..<Int(nextSlot) {
            guard oldIndex < slots.count else { break }

            if slots[oldIndex].isOccupied {
                // Increment generation for the new slot position
                let newGeneration = slots[oldIndex].generation &+ 1
                newSlots.append(SlotInfo(generation: newGeneration, isOccupied: true))
                newHandles.append(VectorHandle(index: newIndex, generation: newGeneration))
                indexMapping[UInt32(oldIndex)] = newIndex
                newIndex += 1
            }
        }

        // Update state
        slots = newSlots
        nextSlot = newIndex
        // occupiedCount stays the same (we only removed unoccupied)

        return CompactionResult(
            indexMapping: indexMapping,
            newHandles: newHandles,
            newSlotCount: Int(newIndex)
        )
    }

    /// Rebuild allocator state from a keep mask.
    ///
    /// Alternative to compact() when you need to specify which slots to keep.
    ///
    /// - Parameter keepMask: Boolean array where true means keep the slot
    /// - Returns: Compaction result
    func compact(keepMask: [Bool]) -> CompactionResult {
        var indexMapping: [UInt32: UInt32] = [:]
        var newSlots: [SlotInfo] = []
        var newHandles: [VectorHandle] = []

        var newIndex: UInt32 = 0
        for oldIndex in 0..<min(Int(nextSlot), keepMask.count) {
            if keepMask[oldIndex] && oldIndex < slots.count && slots[oldIndex].isOccupied {
                let newGeneration = slots[oldIndex].generation &+ 1
                newSlots.append(SlotInfo(generation: newGeneration, isOccupied: true))
                newHandles.append(VectorHandle(index: newIndex, generation: newGeneration))
                indexMapping[UInt32(oldIndex)] = newIndex
                newIndex += 1
            }
        }

        slots = newSlots
        nextSlot = newIndex
        occupiedCount = Int(newIndex)

        return CompactionResult(
            indexMapping: indexMapping,
            newHandles: newHandles,
            newSlotCount: Int(newIndex)
        )
    }

    // MARK: - Iteration

    /// Iterate over all occupied slot indices.
    ///
    /// - Parameter body: Closure called for each occupied slot index
    func forEachOccupied(_ body: (UInt32) -> Void) {
        for i in 0..<min(Int(nextSlot), slots.count) {
            if slots[i].isOccupied {
                body(UInt32(i))
            }
        }
    }

    /// Get all occupied slot indices.
    ///
    /// - Returns: Array of occupied slot indices
    func occupiedSlotIndices() -> [UInt32] {
        var indices: [UInt32] = []
        indices.reserveCapacity(occupiedCount)
        forEachOccupied { indices.append($0) }
        return indices
    }

    /// Create a handle for a slot (for internal use after compaction).
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Handle with current generation, or nil if slot doesn't exist
    func handle(for slotIndex: UInt32) -> VectorHandle? {
        guard slotIndex < slots.count else { return nil }
        guard slots[Int(slotIndex)].isOccupied else { return nil }
        return VectorHandle(index: slotIndex, generation: slots[Int(slotIndex)].generation)
    }

    // MARK: - Reset

    /// Reset to empty state.
    func reset() {
        slots.removeAll(keepingCapacity: true)
        nextSlot = 0
        occupiedCount = 0
    }

    // MARK: - Private Helpers

    private func ensureCapacity(for index: Int) {
        if index >= slots.count {
            let growth = max(1000, index - slots.count + 1)
            slots.append(contentsOf: repeatElement(SlotInfo(), count: growth))
        }
    }
}
