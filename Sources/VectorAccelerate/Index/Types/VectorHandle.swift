//
//  VectorHandle.swift
//  VectorAccelerate
//
//  Opaque handle to a vector stored in an AcceleratedVectorIndex.
//
//  Handles are stable across compaction: a handle's identity never changes for the
//  lifetime of the vector. Internally the index uses an indirection table to map
//  handle IDs to the current storage slot.
//

import Foundation

// MARK: - Vector Handle

/// Opaque handle to a vector in an ``AcceleratedVectorIndex``.
///
/// Handles are returned from `insert()` and used to identify vectors in
/// search results, metadata operations, and removal.
///
/// ## Characteristics
/// - Lightweight: 4 bytes (`UInt32` stable ID)
/// - Hashable: Can be used as dictionary keys
/// - Comparable: Supports sorting
/// - Sendable: Safe for concurrent access
///
/// ## Stability
/// Handles are stable across `compact()`.
///
/// Internally, the index may move vectors to different GPU storage slots during
/// compaction. A handle's `stableID` is mapped to the current slot via an
/// indirection table, so user code never needs to remap handles.
public struct VectorHandle: Hashable, Sendable, Comparable {

    // MARK: - Internal Storage

    /// Stable identifier for this vector.
    ///
    /// This value never changes for the lifetime of the vector.
    @usableFromInline
    internal let stableID: UInt32

    // MARK: - Constants

    /// Invalid handle sentinel value.
    /// Used to indicate "no vector" in search results when `k` exceeds available vectors.
    public static let invalid = VectorHandle(stableID: .max)

    // MARK: - Initialization

    /// Create a handle with the given stable identifier.
    ///
    /// This is intentionally internal to keep handles opaque to consumers.
    @usableFromInline
    internal init(stableID: UInt32) {
        self.stableID = stableID
    }

    // MARK: - Validation

    /// Whether this handle is potentially valid (not the sentinel value).
    ///
    /// Note: This only checks if the handle is not the sentinel value.
    /// A handle may still not exist in a particular index instance.
    @inlinable
    public var isValid: Bool {
        stableID != .max
    }

    // MARK: - Comparable

    /// Compare handles by stable ID.
    @inlinable
    public static func < (lhs: VectorHandle, rhs: VectorHandle) -> Bool {
        lhs.stableID < rhs.stableID
    }
}

// MARK: - CustomStringConvertible

extension VectorHandle: CustomStringConvertible {
    public var description: String {
        isValid ? "VectorHandle(\(stableID))" : "VectorHandle.invalid"
    }
}

// MARK: - CustomDebugStringConvertible

extension VectorHandle: CustomDebugStringConvertible {
    public var debugDescription: String {
        "VectorHandle(stableID: \(stableID), valid: \(isValid))"
    }
}

// MARK: - Codable Support

extension VectorHandle: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.stableID = try container.decode(UInt32.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(stableID)
    }
}
