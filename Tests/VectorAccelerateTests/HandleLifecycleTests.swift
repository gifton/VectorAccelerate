//
//  HandleLifecycleTests.swift
//  VectorAccelerate
//
//  Comprehensive tests for handle lifecycle and compaction.
//
//  Covers:
//  - Handle validity checking (contains, isHandleValid)
//  - Stale handle detection after compact
//  - Handle mapping during compaction
//  - Handle reuse and generation tracking
//  - allHandles enumeration
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate
import VectorCore

/// Tests for handle lifecycle, validity, and compaction behavior.
final class HandleLifecycleTests: XCTestCase {

    // MARK: - Basic Contains Tests

    /// A valid, active handle should return true for contains().
    func testContainsActiveHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let containsActive = await index.contains(handle)
        XCTAssertTrue(containsActive, "Active handle should be contained")
    }

    /// A deleted handle should return false for contains().
    func testContainsDeletedHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])
        try await index.remove(handle)

        let containsDeleted = await index.contains(handle)
        XCTAssertFalse(containsDeleted, "Deleted handle should not be contained")
    }

    /// An invalid handle should return false for contains().
    func testContainsInvalidHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let invalidHandle = VectorHandle.invalid
        let containsInvalid = await index.contains(invalidHandle)
        XCTAssertFalse(containsInvalid, "Invalid handle should not be contained")
    }

    /// A handle with wrong generation should return false for contains().
    func testContainsWrongGenerationHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        // Create a handle with same index but different generation
        let wrongGenHandle = VectorHandle(index: handle.index, generation: handle.generation &+ 1)

        let containsWrong = await index.contains(wrongGenHandle)
        XCTAssertFalse(containsWrong, "Handle with wrong generation should not be contained")
    }

    // MARK: - isHandleValid Tests

    /// isHandleValid should return true for active handle.
    func testIsHandleValidForActiveHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let isValid = await index.isHandleValid(handle)
        XCTAssertTrue(isValid, "Active handle should be valid")
    }

    /// isHandleValid should return false for deleted handle.
    func testIsHandleValidForDeletedHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])
        try await index.remove(handle)

        let isValid = await index.isHandleValid(handle)
        XCTAssertFalse(isValid, "Deleted handle should not be valid")
    }

    // MARK: - allHandles Tests

    /// allHandles should return all active handles.
    func testAllHandles() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var insertedHandles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            insertedHandles.append(handle)
        }

        let allHandles = await index.allHandles()

        XCTAssertEqual(allHandles.count, 5, "Should have 5 handles")
        XCTAssertEqual(Set(allHandles), Set(insertedHandles), "Should contain all inserted handles")
    }

    /// allHandles should not include deleted handles.
    func testAllHandlesExcludesDeleted() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var insertedHandles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            insertedHandles.append(handle)
        }

        // Delete handle at index 2
        try await index.remove(insertedHandles[2])

        let allHandles = await index.allHandles()

        XCTAssertEqual(allHandles.count, 4, "Should have 4 handles after deletion")
        XCTAssertFalse(allHandles.contains(insertedHandles[2]), "Deleted handle should not be in allHandles")

        // Other handles should still be present
        XCTAssertTrue(allHandles.contains(insertedHandles[0]))
        XCTAssertTrue(allHandles.contains(insertedHandles[1]))
        XCTAssertTrue(allHandles.contains(insertedHandles[3]))
        XCTAssertTrue(allHandles.contains(insertedHandles[4]))
    }

    /// allHandles on empty index should return empty array.
    func testAllHandlesEmpty() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let allHandles = await index.allHandles()
        XCTAssertEqual(allHandles.count, 0, "Empty index should have no handles")
    }

    // MARK: - handle(for:) Tests

    /// handle(for:) should return the correct handle for a slot.
    func testHandleForSlotIndex() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let retrievedHandle = await index.handle(for: handle.index)
        XCTAssertEqual(retrievedHandle, handle, "handle(for:) should return the same handle")
    }

    /// handle(for:) should return nil for unoccupied slot.
    func testHandleForUnoccupiedSlot() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])

        // Slot 99 should not be occupied
        let retrievedHandle = await index.handle(for: 99)
        XCTAssertNil(retrievedHandle, "Unoccupied slot should return nil")
    }

    /// handle(for:) should return nil for deleted slot.
    func testHandleForDeletedSlot() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let slotIndex = handle.index

        try await index.remove(handle)

        // After deletion, the slot should still return the handle (but contains should be false)
        // The slot is marked deleted but not yet compacted
        let retrievedHandle = await index.handle(for: slotIndex)
        // Note: handle(for:) might return the handle even if deleted, but contains() should return false
        if let retrieved = retrievedHandle {
            let isContained = await index.contains(retrieved)
            XCTAssertFalse(isContained, "Handle from deleted slot should not be contained")
        }
    }

    // MARK: - Compaction Tests

    /// After compaction, old handles should become stale (return nil on vector retrieval).
    func testStaleHandleAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 5 vectors
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }

        // Delete vectors at indices 1 and 3
        try await index.remove(handles[1])
        try await index.remove(handles[3])

        // Compact
        let mapping = try await index.compact()

        // Old handles for kept vectors should be stale (not in contains)
        // The mapping provides new handles
        XCTAssertEqual(mapping.count, 3, "Should have 3 handle mappings for kept vectors")

        // Old handle for vector 0 should be stale
        let oldHandle0 = handles[0]
        let isStale = await index.contains(oldHandle0)
        XCTAssertFalse(isStale, "Old handle should be stale after compact")

        // Vector retrieval with old handle should return nil
        let vector = try await index.vector(for: oldHandle0)
        XCTAssertNil(vector, "Old handle should not retrieve vector after compact")
    }

    /// Compaction should return a valid mapping from old to new handles.
    func testHandleMappingAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 5 vectors
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }

        // Delete vectors at indices 1 and 3
        try await index.remove(handles[1])
        try await index.remove(handles[3])

        // Compact
        let mapping = try await index.compact()

        // Verify mapping contains the kept handles
        XCTAssertTrue(mapping.keys.contains(handles[0]), "Mapping should include handle 0")
        XCTAssertTrue(mapping.keys.contains(handles[2]), "Mapping should include handle 2")
        XCTAssertTrue(mapping.keys.contains(handles[4]), "Mapping should include handle 4")

        // Deleted handles should not be in mapping
        XCTAssertFalse(mapping.keys.contains(handles[1]), "Mapping should not include deleted handle 1")
        XCTAssertFalse(mapping.keys.contains(handles[3]), "Mapping should not include deleted handle 3")

        // New handles should be valid
        for (_, newHandle) in mapping {
            let isValid = await index.contains(newHandle)
            XCTAssertTrue(isValid, "New handle should be valid")
        }
    }

    /// New handles from mapping should retrieve the correct vectors.
    func testNewHandlesRetrieveCorrectVectors() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors with distinct values
        let vectors: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0]
        ]

        var handles: [VectorHandle] = []
        for vector in vectors {
            let handle = try await index.insert(vector)
            handles.append(handle)
        }

        // Delete middle vectors
        try await index.remove(handles[1])
        try await index.remove(handles[3])

        // Compact
        let mapping = try await index.compact()

        // Verify new handles retrieve correct vectors
        // Handle 0 → should retrieve [0,0,0,0]
        if let newHandle = mapping[handles[0]] {
            let retrieved = try await index.vector(for: newHandle)
            XCTAssertNotNil(retrieved)
            XCTAssertEqual(Double(retrieved?[0] ?? -1), 0.0, accuracy: 0.001)
        }

        // Handle 2 → should retrieve [2,0,0,0]
        if let newHandle = mapping[handles[2]] {
            let retrieved = try await index.vector(for: newHandle)
            XCTAssertNotNil(retrieved)
            XCTAssertEqual(Double(retrieved?[0] ?? -1), 2.0, accuracy: 0.001)
        }

        // Handle 4 → should retrieve [4,0,0,0]
        if let newHandle = mapping[handles[4]] {
            let retrieved = try await index.vector(for: newHandle)
            XCTAssertNotNil(retrieved)
            XCTAssertEqual(Double(retrieved?[0] ?? -1), 4.0, accuracy: 0.001)
        }
    }

    /// New inserts after compaction should get fresh handles.
    func testHandleReuseAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert and delete to create fragmentation
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        _ = try await index.insert([2.0, 0.0, 0.0, 0.0])
        try await index.remove(handle1)

        // Compact
        _ = try await index.compact()

        // Insert a new vector
        let newHandle = try await index.insert([3.0, 0.0, 0.0, 0.0])

        // New handle should be valid
        let isNewValid = await index.contains(newHandle)
        XCTAssertTrue(isNewValid, "New handle should be valid")

        // New handle should be different from the old handles
        XCTAssertNotEqual(newHandle.index, handle1.index,
                          "New handle may reuse slot but should have different generation or index")

        // Verify vector retrieval works
        let retrieved = try await index.vector(for: newHandle)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(Double(retrieved?[0] ?? -1), 3.0, accuracy: 0.001)
    }

    // MARK: - Generation Tests

    /// After compaction, old handles become stale when slots are reorganized.
    /// Note: When a slot is fully deleted and compacted, fresh inserts may reuse the slot
    /// at generation 0 since there's no active handle to conflict with.
    func testHandleValidityAfterCompactAndReinsert() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert two vectors
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])

        // Delete only one
        try await index.remove(handle1)

        // Compact - handle2 gets remapped
        let mapping = try await index.compact()

        // handle2 should be in the mapping and get a new handle
        XCTAssertTrue(mapping.keys.contains(handle2), "Kept vector should be in mapping")

        // New handle should be valid
        if let newHandle = mapping[handle2] {
            let isValid = await index.contains(newHandle)
            XCTAssertTrue(isValid, "New handle should be valid")

            // Old handle should be stale
            let oldValid = await index.contains(handle2)
            XCTAssertFalse(oldValid, "Old handle should be stale after compaction")
        }
    }

    /// Test that compaction properly tracks generation for reorganized handles.
    /// This tests the handle mapping mechanism rather than generation tracking
    /// after full deletion (which resets to generation 0).
    func testCompactPreservesGenerationTracking() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert 3 vectors
        let handle0 = try await index.insert([0.0, 0.0, 0.0, 0.0])
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])

        // Delete the middle one
        try await index.remove(handle1)

        // Compact
        let mapping = try await index.compact()

        // Old handles should be stale (mapping provides new ones)
        let contains0Old = await index.contains(handle0)
        let contains2Old = await index.contains(handle2)
        XCTAssertFalse(contains0Old, "Old handle 0 should be stale after compact")
        XCTAssertFalse(contains2Old, "Old handle 2 should be stale after compact")

        // New handles should be valid
        if let newHandle0 = mapping[handle0] {
            let containsNew = await index.contains(newHandle0)
            XCTAssertTrue(containsNew, "New handle should be valid")

            // Verify correct vector is retrieved
            let retrieved = try await index.vector(for: newHandle0)
            XCTAssertNotNil(retrieved)
            XCTAssertEqual(Double(retrieved?[0] ?? -1), 0.0, accuracy: 0.001)
        }
    }

    // MARK: - Compaction Edge Cases

    /// Compacting with no deletions should return empty mapping.
    func testCompactWithNoDeletions() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert without deleting
        for i in 0..<5 {
            _ = try await index.insert([Float(i), 0.0, 0.0, 0.0])
        }

        // Compact
        let mapping = try await index.compact()

        XCTAssertEqual(mapping.count, 0, "No deletions means no compaction needed")
    }

    /// Compacting empty index should work without error.
    func testCompactEmptyIndex() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let mapping = try await index.compact()
        XCTAssertEqual(mapping.count, 0, "Empty index compaction should return empty mapping")
    }

    /// After deleting all vectors and compacting, index should be empty.
    func testCompactAfterDeletingAll() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert and delete all
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }

        for handle in handles {
            try await index.remove(handle)
        }

        // Compact
        let mapping = try await index.compact()

        // Should have no mappings (all deleted)
        XCTAssertEqual(mapping.count, 0, "All vectors deleted means no mappings")

        // Index should be empty
        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 0)
        XCTAssertEqual(stats.deletedSlots, 0)
    }

    /// Multiple sequential compactions should work correctly.
    func testMultipleCompactions() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // First round: insert 5, delete 2, compact
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }
        try await index.remove(handles[1])
        try await index.remove(handles[3])

        let mapping1 = try await index.compact()
        XCTAssertEqual(mapping1.count, 3)

        // Get new handles
        var currentHandles = mapping1.values.map { $0 }

        // Second round: add more, delete some, compact
        for i in 5..<8 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            currentHandles.append(handle)
        }

        // Delete first current handle
        if let first = currentHandles.first {
            try await index.remove(first)
        }

        let mapping2 = try await index.compact()

        // Should have mappings for remaining vectors
        XCTAssertEqual(mapping2.count, 5, "Should have 5 vectors after second compaction")

        // All new handles should be valid
        for newHandle in mapping2.values {
            let isValid = await index.contains(newHandle)
            XCTAssertTrue(isValid)
        }
    }

    // MARK: - Metadata Preservation After Compact

    /// Metadata should be preserved after compaction.
    func testMetadataPreservedAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert with metadata
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["key": "value1"])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0], metadata: ["key": "value2"])
        let handle3 = try await index.insert([3.0, 0.0, 0.0, 0.0], metadata: ["key": "value3"])

        // Delete middle one
        try await index.remove(handle2)

        // Compact
        let mapping = try await index.compact()

        // Check metadata is preserved
        if let newHandle1 = mapping[handle1] {
            let meta = await index.metadata(for: newHandle1)
            XCTAssertEqual(meta?["key"], "value1")
        }

        if let newHandle3 = mapping[handle3] {
            let meta = await index.metadata(for: newHandle3)
            XCTAssertEqual(meta?["key"], "value3")
        }
    }

    // MARK: - Search After Compact

    /// Search should work correctly after compaction.
    func testSearchAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        let vectors: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0]
        ]

        var handles: [VectorHandle] = []
        for vector in vectors {
            let handle = try await index.insert(vector)
            handles.append(handle)
        }

        // Delete some
        try await index.remove(handles[1])
        try await index.remove(handles[3])

        // Compact
        let mapping = try await index.compact()

        // Search should find the remaining vectors
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 5)

        XCTAssertEqual(results.count, 3, "Should find 3 remaining vectors")

        // First result should be exact match with distance 0
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)

        // Results should use new handles from mapping
        let newHandleSet = Set(mapping.values)
        for result in results {
            XCTAssertTrue(newHandleSet.contains(result.handle),
                          "Search results should use new handles")
        }
    }
}
