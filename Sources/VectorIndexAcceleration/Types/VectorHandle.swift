//
//  VectorHandle.swift
//  VectorIndexAcceleration
//
//  Opaque handle to a vector stored in an AcceleratedVectorIndex.
//
//  Handles are lightweight value types that uniquely identify vectors.
//  They use a generation counter for stale handle detection after compact().
//

import Foundation

// MARK: - Vector Handle

/// Opaque handle to a vector in an AcceleratedVectorIndex.
///
/// Handles are returned from `insert()` and used to identify vectors in
/// search results, metadata operations, and removal.
///
/// ## Characteristics
/// - Lightweight: 6 bytes (UInt32 index + UInt16 generation)
/// - Hashable: Can be used as dictionary keys
/// - Comparable: Supports sorting
/// - Sendable: Safe for concurrent access
///
/// ## Generation-Based Stability
/// Handles include a generation counter that enables detection of stale handles
/// after `compact()` operations. When a slot is reused, its generation increments,
/// making old handles to that slot invalid.
///
/// ```swift
/// let handle = try await index.insert(vector)
/// try await index.remove(handle)
/// try await index.compact()
/// // handle is now stale - the slot may be reused with a new generation
/// ```
///
/// ## Usage
/// ```swift
/// let handle = try await index.insert(vector)
/// let results = try await index.search(query: query, k: 10)
/// if results[0].handle == handle {
///     print("Found our vector!")
/// }
/// ```
public struct VectorHandle: Hashable, Sendable, Comparable {

    // MARK: - Internal Storage

    /// Internal index into the vector buffer.
    /// This is an implementation detail and should not be relied upon.
    @usableFromInline
    internal let index: UInt32

    /// Generation counter for stale handle detection.
    /// Incremented each time a slot is reused after deletion.
    @usableFromInline
    internal let generation: UInt16

    // MARK: - Constants

    /// Invalid handle sentinel value.
    /// Used to indicate "no vector" in search results when k exceeds available vectors.
    public static let invalid = VectorHandle(index: .max, generation: 0)

    // MARK: - Initialization

    /// Create a handle with the given internal index and generation.
    /// - Parameters:
    ///   - index: The internal buffer index
    ///   - generation: The generation counter for this slot
    @usableFromInline
    internal init(index: UInt32, generation: UInt16) {
        self.index = index
        self.generation = generation
    }

    // MARK: - Validation

    /// Whether this handle is potentially valid (not the sentinel value).
    ///
    /// Note: This only checks if the handle is not the sentinel value.
    /// A handle may still be stale if its generation doesn't match the
    /// current generation of its slot in the index.
    @inlinable
    public var isValid: Bool {
        self != .invalid
    }

    // MARK: - Comparable

    /// Compare handles by index, then by generation.
    @inlinable
    public static func < (lhs: VectorHandle, rhs: VectorHandle) -> Bool {
        if lhs.index != rhs.index {
            return lhs.index < rhs.index
        }
        return lhs.generation < rhs.generation
    }
}

// MARK: - CustomStringConvertible

extension VectorHandle: CustomStringConvertible {
    public var description: String {
        isValid ? "VectorHandle(\(index))" : "VectorHandle.invalid"
    }
}

// MARK: - CustomDebugStringConvertible

extension VectorHandle: CustomDebugStringConvertible {
    public var debugDescription: String {
        "VectorHandle(index: \(index), generation: \(generation), valid: \(isValid))"
    }
}

// MARK: - Codable Support

extension VectorHandle: Codable {
    public init(from decoder: any Decoder) throws {
        var container = try decoder.unkeyedContainer()
        self.index = try container.decode(UInt32.self)
        self.generation = try container.decode(UInt16.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(index)
        try container.encode(generation)
    }
}
