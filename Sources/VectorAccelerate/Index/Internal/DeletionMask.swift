//
//  DeletionMask.swift
//  VectorAccelerate
//
//  Efficient bitset-based deletion tracking.
//
//  Provides:
//  - O(1) deletion check and marking
//  - Compact memory representation (1 bit per slot)
//  - Fragmentation metrics for compaction decisions
//

import Foundation

// MARK: - Deletion Mask

/// Bitset-based deletion tracking for lazy deletion.
///
/// Uses a compact bitset representation where each bit represents
/// whether a slot is deleted (1) or active (0).
///
/// ## Memory Efficiency
/// Uses 1 bit per slot (vs 1 byte for Bool array).
/// For 1M vectors: ~125KB vs ~1MB.
///
/// ## Fragmentation Tracking
/// Tracks deletion count for fragmentation analysis:
/// ```swift
/// if mask.fragmentationRatio > 0.25 {
///     // Consider compaction
/// }
/// ```
///
/// ## Thread Safety
/// This struct is value-type safe but not designed for concurrent mutation.
struct DeletionMask: Sendable {

    // MARK: - Properties

    /// Bitset storage (each UInt64 holds 64 deletion flags).
    private var bits: [UInt64]

    /// Number of slots currently tracked.
    private(set) var capacity: Int

    /// Number of deleted slots.
    private(set) var deletedCount: Int = 0

    /// Bits per word.
    private static let bitsPerWord = 64

    // MARK: - Computed Properties

    /// Number of active (non-deleted) slots.
    var activeCount: Int {
        capacity - deletedCount
    }

    /// Fragmentation ratio (deleted / total).
    var fragmentationRatio: Float {
        guard capacity > 0 else { return 0 }
        return Float(deletedCount) / Float(capacity)
    }

    /// Whether compaction is recommended (fragmentation > 25%).
    var shouldCompact: Bool {
        fragmentationRatio > 0.25
    }

    // MARK: - Initialization

    /// Create a deletion mask with the specified capacity.
    ///
    /// - Parameter capacity: Initial capacity (number of slots)
    init(capacity: Int = 0) {
        self.capacity = capacity
        let wordCount = (capacity + Self.bitsPerWord - 1) / Self.bitsPerWord
        self.bits = Array(repeating: 0, count: wordCount)
    }

    // MARK: - Capacity Management

    /// Ensure capacity for at least the specified number of slots.
    ///
    /// - Parameter requiredCapacity: Minimum required capacity
    mutating func ensureCapacity(_ requiredCapacity: Int) {
        guard requiredCapacity > capacity else { return }

        let requiredWords = (requiredCapacity + Self.bitsPerWord - 1) / Self.bitsPerWord
        if requiredWords > bits.count {
            bits.append(contentsOf: repeatElement(0, count: requiredWords - bits.count))
        }
        capacity = requiredCapacity
    }

    // MARK: - Deletion Operations

    /// Check if a slot is deleted.
    ///
    /// - Parameter index: Slot index
    /// - Returns: true if deleted, false if active or out of bounds
    func isDeleted(_ index: Int) -> Bool {
        guard index >= 0 && index < capacity else { return false }

        let wordIndex = index / Self.bitsPerWord
        let bitIndex = index % Self.bitsPerWord
        return (bits[wordIndex] & (1 << bitIndex)) != 0
    }

    /// Check if a slot is active (not deleted).
    ///
    /// - Parameter index: Slot index
    /// - Returns: true if active, false if deleted or out of bounds
    func isActive(_ index: Int) -> Bool {
        guard index >= 0 && index < capacity else { return false }
        return !isDeleted(index)
    }

    /// Mark a slot as deleted.
    ///
    /// - Parameter index: Slot index to mark as deleted
    /// - Returns: true if newly deleted, false if already deleted or out of bounds
    @discardableResult
    mutating func markDeleted(_ index: Int) -> Bool {
        guard index >= 0 && index < capacity else { return false }

        let wordIndex = index / Self.bitsPerWord
        let bitIndex = index % Self.bitsPerWord
        let mask: UInt64 = 1 << bitIndex

        if (bits[wordIndex] & mask) == 0 {
            bits[wordIndex] |= mask
            deletedCount += 1
            return true
        }
        return false
    }

    /// Mark a slot as active (undelete).
    ///
    /// - Parameter index: Slot index to mark as active
    /// - Returns: true if newly activated, false if already active or out of bounds
    @discardableResult
    mutating func markActive(_ index: Int) -> Bool {
        guard index >= 0 && index < capacity else { return false }

        let wordIndex = index / Self.bitsPerWord
        let bitIndex = index % Self.bitsPerWord
        let mask: UInt64 = 1 << bitIndex

        if (bits[wordIndex] & mask) != 0 {
            bits[wordIndex] &= ~mask
            deletedCount -= 1
            return true
        }
        return false
    }

    // MARK: - Bulk Operations

    /// Get a keep mask (inverse of deletion mask) as Bool array.
    ///
    /// - Returns: Bool array where true = keep (active), false = delete
    func keepMask() -> [Bool] {
        var result = Array(repeating: true, count: capacity)
        for i in 0..<capacity {
            if isDeleted(i) {
                result[i] = false
            }
        }
        return result
    }

    /// Get indices of all active (non-deleted) slots.
    ///
    /// - Returns: Array of active slot indices
    func activeIndices() -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(activeCount)

        for i in 0..<capacity {
            if !isDeleted(i) {
                result.append(i)
            }
        }
        return result
    }

    /// Get indices of all deleted slots.
    ///
    /// - Returns: Array of deleted slot indices
    func deletedIndices() -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(deletedCount)

        for i in 0..<capacity {
            if isDeleted(i) {
                result.append(i)
            }
        }
        return result
    }

    // MARK: - Compaction Support

    /// Reset after compaction.
    ///
    /// Creates a fresh mask with the new size, all slots active.
    ///
    /// - Parameter newCapacity: Capacity after compaction
    mutating func resetAfterCompaction(newCapacity: Int) {
        self = DeletionMask(capacity: newCapacity)
    }

    // MARK: - Reset

    /// Reset to empty state.
    mutating func reset() {
        bits = []
        capacity = 0
        deletedCount = 0
    }

    /// Clear all deletion flags (mark all as active).
    mutating func clearAll() {
        for i in 0..<bits.count {
            bits[i] = 0
        }
        deletedCount = 0
    }
}

// MARK: - Sequence Conformance

extension DeletionMask: Sequence {
    /// Iterator over active (non-deleted) slot indices.
    struct ActiveIterator: IteratorProtocol {
        let mask: DeletionMask
        var currentIndex: Int = 0

        mutating func next() -> Int? {
            while currentIndex < mask.capacity {
                let index = currentIndex
                currentIndex += 1
                if !mask.isDeleted(index) {
                    return index
                }
            }
            return nil
        }
    }

    /// Returns an iterator over active slot indices.
    func makeIterator() -> ActiveIterator {
        ActiveIterator(mask: self)
    }
}

// MARK: - CustomStringConvertible

extension DeletionMask: CustomStringConvertible {
    var description: String {
        "DeletionMask(capacity: \(capacity), deleted: \(deletedCount), fragmentation: \(String(format: "%.1f%%", fragmentationRatio * 100)))"
    }
}
