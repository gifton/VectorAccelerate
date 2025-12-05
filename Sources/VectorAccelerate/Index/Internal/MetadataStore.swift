//
//  MetadataStore.swift
//  VectorAccelerate
//
//  CPU-side sparse metadata storage.
//
//  Provides:
//  - Sparse storage (only non-nil entries)
//  - O(1) lookup by slot index
//  - Compaction alongside vector buffer
//

import Foundation

// MARK: - Metadata Store

/// Sparse CPU-side metadata storage.
///
/// Uses a dictionary for memory-efficient storage of optional per-vector metadata.
/// Only stores entries where metadata is actually provided.
///
/// ## Memory Efficiency
/// For an index with 1M vectors where only 10% have metadata:
/// - Dense array: 1M entries (most nil/empty)
/// - Sparse dict: 100K entries
///
/// ## Thread Safety
/// This struct is value-type safe. For concurrent access from an actor,
/// use the owning actor's isolation.
struct MetadataStore: Sendable {

    // MARK: - Properties

    /// Sparse storage: slot index -> metadata.
    private var storage: [UInt32: VectorMetadata]

    /// Number of entries with metadata.
    var count: Int {
        storage.count
    }

    /// Whether the store is empty.
    var isEmpty: Bool {
        storage.isEmpty
    }

    /// Estimated memory usage in bytes.
    var estimatedMemoryBytes: Int {
        // Rough estimate: ~100 bytes per entry (keys, values, dict overhead)
        count * 100
    }

    // MARK: - Initialization

    /// Create an empty metadata store.
    init() {
        self.storage = [:]
    }

    /// Create a metadata store with initial capacity hint.
    ///
    /// - Parameter capacity: Expected number of entries (optimization hint)
    init(capacity: Int) {
        self.storage = Dictionary(minimumCapacity: capacity)
    }

    // MARK: - Access

    /// Get metadata for a slot.
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Metadata if set, nil otherwise
    func get(_ slotIndex: UInt32) -> VectorMetadata? {
        storage[slotIndex]
    }

    /// Get metadata for a handle.
    ///
    /// - Parameter handle: Vector handle
    /// - Returns: Metadata if set, nil otherwise
    func get(for handle: VectorHandle) -> VectorMetadata? {
        guard handle.isValid else { return nil }
        return storage[handle.index]
    }

    /// Set metadata for a slot.
    ///
    /// - Parameters:
    ///   - metadata: Metadata to set (nil to remove)
    ///   - slotIndex: Slot index
    mutating func set(_ metadata: VectorMetadata?, for slotIndex: UInt32) {
        if let metadata = metadata, !metadata.isEmpty {
            storage[slotIndex] = metadata
        } else {
            storage.removeValue(forKey: slotIndex)
        }
    }

    /// Set metadata for a handle.
    ///
    /// - Parameters:
    ///   - metadata: Metadata to set (nil to remove)
    ///   - handle: Vector handle
    mutating func set(_ metadata: VectorMetadata?, for handle: VectorHandle) {
        guard handle.isValid else { return }
        set(metadata, for: handle.index)
    }

    /// Check if a slot has metadata.
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: true if metadata exists for this slot
    func hasMetadata(_ slotIndex: UInt32) -> Bool {
        storage[slotIndex] != nil
    }

    /// Remove metadata for a slot.
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Removed metadata, or nil if none existed
    @discardableResult
    mutating func remove(_ slotIndex: UInt32) -> VectorMetadata? {
        storage.removeValue(forKey: slotIndex)
    }

    /// Remove metadata for a handle.
    ///
    /// - Parameter handle: Vector handle
    /// - Returns: Removed metadata, or nil if none existed
    @discardableResult
    mutating func remove(for handle: VectorHandle) -> VectorMetadata? {
        guard handle.isValid else { return nil }
        return storage.removeValue(forKey: handle.index)
    }

    // MARK: - Bulk Operations

    /// Set metadata for multiple slots.
    ///
    /// - Parameters:
    ///   - entries: Array of (slotIndex, metadata) pairs
    mutating func setMultiple(_ entries: [(UInt32, VectorMetadata?)]) {
        for (slotIndex, metadata) in entries {
            set(metadata, for: slotIndex)
        }
    }

    /// Get all slot indices that have metadata.
    ///
    /// - Returns: Array of slot indices with metadata
    func slotsWithMetadata() -> [UInt32] {
        Array(storage.keys)
    }

    // MARK: - Compaction

    /// Compact the store using an index mapping.
    ///
    /// Remaps all slot indices according to the provided mapping.
    /// Entries not in the mapping are discarded.
    ///
    /// - Parameter indexMapping: Mapping from old slot indices to new indices
    /// - Returns: New metadata store with remapped indices
    func compacted(using indexMapping: [UInt32: UInt32]) -> MetadataStore {
        var newStore = MetadataStore(capacity: indexMapping.count)

        for (oldIndex, metadata) in storage {
            if let newIndex = indexMapping[oldIndex] {
                newStore.storage[newIndex] = metadata
            }
        }

        return newStore
    }

    /// Compact in place using an index mapping.
    ///
    /// - Parameter indexMapping: Mapping from old slot indices to new indices
    mutating func compact(using indexMapping: [UInt32: UInt32]) {
        var newStorage: [UInt32: VectorMetadata] = Dictionary(minimumCapacity: indexMapping.count)

        for (oldIndex, metadata) in storage {
            if let newIndex = indexMapping[oldIndex] {
                newStorage[newIndex] = metadata
            }
        }

        storage = newStorage
    }

    /// Compact using a keep mask.
    ///
    /// - Parameter keepMask: Boolean array where true = keep the slot
    /// - Returns: Tuple of (new store, index mapping from old to new)
    func compacted(keepMask: [Bool]) -> (store: MetadataStore, indexMapping: [UInt32: UInt32]) {
        var indexMapping: [UInt32: UInt32] = [:]
        var newIndex: UInt32 = 0

        for oldIndex in 0..<keepMask.count {
            if keepMask[oldIndex] {
                indexMapping[UInt32(oldIndex)] = newIndex
                newIndex += 1
            }
        }

        let newStore = compacted(using: indexMapping)
        return (newStore, indexMapping)
    }

    // MARK: - Filtering

    /// Filter entries matching a predicate.
    ///
    /// - Parameter predicate: Filter function
    /// - Returns: Array of (slotIndex, metadata) pairs matching the predicate
    func filter(_ predicate: (UInt32, VectorMetadata) -> Bool) -> [(UInt32, VectorMetadata)] {
        storage.compactMap { slotIndex, metadata in
            predicate(slotIndex, metadata) ? (slotIndex, metadata) : nil
        }
    }

    /// Get all entries.
    ///
    /// - Returns: Array of (slotIndex, metadata) pairs
    func allEntries() -> [(UInt32, VectorMetadata)] {
        storage.map { ($0.key, $0.value) }
    }

    // MARK: - Reset

    /// Remove all entries.
    mutating func removeAll() {
        storage.removeAll(keepingCapacity: true)
    }

    /// Remove all entries and release memory.
    mutating func reset() {
        storage = [:]
    }
}

// MARK: - Subscript Access

extension MetadataStore {
    /// Subscript access for slot indices.
    subscript(slotIndex: UInt32) -> VectorMetadata? {
        get { get(slotIndex) }
        set { set(newValue, for: slotIndex) }
    }

    /// Subscript access for handles.
    subscript(handle: VectorHandle) -> VectorMetadata? {
        get { get(for: handle) }
        set { set(newValue, for: handle) }
    }
}

// MARK: - CustomStringConvertible

extension MetadataStore: CustomStringConvertible {
    var description: String {
        "MetadataStore(entries: \(count), ~\(estimatedMemoryBytes / 1024)KB)"
    }
}
