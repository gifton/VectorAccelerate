//
//  MetadataStore.swift
//  VectorAccelerate
//
//  Metadata storage keyed by stable handle IDs.
//
//  With stable handles (P0.8), metadata is keyed by `VectorHandle.stableID`
//  so it remains valid across compaction (slots may move, stable IDs do not).
//

import Foundation

// MARK: - Metadata Store

/// Sparse CPU-side metadata storage keyed by stable ID.
///
/// Uses a dictionary for memory-efficient storage of optional per-vector metadata.
/// Only stores entries where metadata is actually provided.
///
/// ## Stability Across Compaction
/// Since metadata is keyed by `stableID` (not slot index), no remapping is
/// needed during compaction. This simplifies lifecycle management.
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

    /// Sparse storage: stableID -> metadata.
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

    // MARK: - Access by stableID

    /// Get metadata for a stable ID.
    ///
    /// - Parameter stableID: Stable identifier
    /// - Returns: Metadata if set, nil otherwise
    func get(for stableID: UInt32) -> VectorMetadata? {
        storage[stableID]
    }

    /// Get metadata for a handle.
    ///
    /// - Parameter handle: Vector handle
    /// - Returns: Metadata if set, nil otherwise
    func get(for handle: VectorHandle) -> VectorMetadata? {
        guard handle.isValid else { return nil }
        return storage[handle.stableID]
    }

    /// Set metadata for a stable ID.
    ///
    /// - Parameters:
    ///   - metadata: Metadata to set (nil to remove)
    ///   - stableID: Stable identifier
    mutating func set(_ metadata: VectorMetadata?, for stableID: UInt32) {
        if let metadata = metadata, !metadata.isEmpty {
            storage[stableID] = metadata
        } else {
            storage.removeValue(forKey: stableID)
        }
    }

    /// Set metadata for a handle.
    ///
    /// - Parameters:
    ///   - metadata: Metadata to set (nil to remove)
    ///   - handle: Vector handle
    mutating func set(_ metadata: VectorMetadata?, for handle: VectorHandle) {
        guard handle.isValid else { return }
        set(metadata, for: handle.stableID)
    }

    /// Check if a stable ID has metadata.
    ///
    /// - Parameter stableID: Stable identifier
    /// - Returns: true if metadata exists
    func hasMetadata(_ stableID: UInt32) -> Bool {
        storage[stableID] != nil
    }

    /// Remove metadata for a stable ID.
    ///
    /// - Parameter stableID: Stable identifier
    /// - Returns: Removed metadata, or nil if none existed
    @discardableResult
    mutating func remove(_ stableID: UInt32) -> VectorMetadata? {
        storage.removeValue(forKey: stableID)
    }

    /// Remove metadata for a handle.
    ///
    /// - Parameter handle: Vector handle
    /// - Returns: Removed metadata, or nil if none existed
    @discardableResult
    mutating func remove(for handle: VectorHandle) -> VectorMetadata? {
        guard handle.isValid else { return nil }
        return storage.removeValue(forKey: handle.stableID)
    }

    // MARK: - Bulk Operations

    /// Set metadata for multiple entries.
    ///
    /// - Parameters:
    ///   - entries: Array of (stableID, metadata) pairs
    mutating func setMultiple(_ entries: [(UInt32, VectorMetadata?)]) {
        for (stableID, metadata) in entries {
            set(metadata, for: stableID)
        }
    }

    /// Get all stable IDs that have metadata.
    ///
    /// - Returns: Array of stable IDs with metadata
    func stableIDsWithMetadata() -> [UInt32] {
        Array(storage.keys)
    }

    // MARK: - Filtering

    /// Filter entries matching a predicate.
    ///
    /// - Parameter predicate: Filter function
    /// - Returns: Array of (stableID, metadata) pairs matching the predicate
    func filter(_ predicate: (UInt32, VectorMetadata) -> Bool) -> [(UInt32, VectorMetadata)] {
        storage.compactMap { stableID, metadata in
            predicate(stableID, metadata) ? (stableID, metadata) : nil
        }
    }

    /// Get all entries.
    ///
    /// - Returns: Array of (stableID, metadata) pairs
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
    /// Subscript access for stable IDs.
    subscript(stableID: UInt32) -> VectorMetadata? {
        get { get(for: stableID) }
        set { set(newValue, for: stableID) }
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
