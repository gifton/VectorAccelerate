//
//  VectorHintStore.swift
//  VectorAccelerate
//
//  Sparse storage for IndexableVector optimization hints (isNormalized, cachedMagnitude),
//  keyed by stable handle IDs. Mirrors MetadataStore's structure.
//
//  Hints are captured at insert time but are not yet consumed by the L2 search kernel.
//  They are reserved for future cosine-metric and normalized-L2 fast-path support.
//

import Foundation

// MARK: - Vector Hints

/// Optimization hints captured from an IndexableVector at insert time.
public struct VectorHints: Sendable, Equatable {
    /// Whether the vector was L2-normalized when inserted.
    public let isNormalized: Bool

    /// Cached L2 magnitude of the vector at insert time, if available.
    public let cachedMagnitude: Float?

    public init(isNormalized: Bool, cachedMagnitude: Float?) {
        self.isNormalized = isNormalized
        self.cachedMagnitude = cachedMagnitude
    }
}

// MARK: - Vector Hint Store

/// Sparse CPU-side hint storage keyed by stable ID.
///
/// Stores `VectorHints` captured from `IndexableVector` conformances at insert time.
/// Keyed by `VectorHandle.stableID` so hints remain valid across compaction.
///
/// ## Current Status
/// Hints are **captured but not consumed** by the current L2 search kernel.
/// They are reserved for future cosine-metric and normalized-L2 fast-path support,
/// where knowing that vectors are pre-normalized allows the kernel to skip
/// redundant normalization or switch to a dot-product-based distance path.
struct VectorHintStore: Sendable {

    // MARK: - Properties

    private var storage: [UInt32: VectorHints]

    var count: Int { storage.count }
    var isEmpty: Bool { storage.isEmpty }

    // MARK: - Initialization

    init() {
        self.storage = [:]
    }

    init(capacity: Int) {
        self.storage = Dictionary(minimumCapacity: capacity)
    }

    // MARK: - Access

    func get(for handle: VectorHandle) -> VectorHints? {
        guard handle.isValid else { return nil }
        return storage[handle.stableID]
    }

    mutating func set(_ hints: VectorHints, for handle: VectorHandle) {
        guard handle.isValid else { return }
        storage[handle.stableID] = hints
    }

    @discardableResult
    mutating func remove(for handle: VectorHandle) -> VectorHints? {
        guard handle.isValid else { return nil }
        return storage.removeValue(forKey: handle.stableID)
    }

    // MARK: - Bulk Operations

    mutating func setMultiple(_ entries: [(VectorHandle, VectorHints)]) {
        for (handle, hints) in entries {
            set(hints, for: handle)
        }
    }

    // MARK: - Reset

    mutating func removeAll() {
        storage.removeAll(keepingCapacity: true)
    }

    mutating func reset() {
        storage = [:]
    }
}

// MARK: - Subscript Access

extension VectorHintStore {
    subscript(handle: VectorHandle) -> VectorHints? {
        get { get(for: handle) }
        set {
            if let newValue = newValue {
                set(newValue, for: handle)
            } else {
                remove(for: handle)
            }
        }
    }
}
