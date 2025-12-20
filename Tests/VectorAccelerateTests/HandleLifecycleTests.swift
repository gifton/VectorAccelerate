//
//  HandleLifecycleTests.swift
//  VectorAccelerateTests
//
//  Tests for VectorHandle lifecycle - insertion, deletion, validation, and compaction.
//
//  P0.8 Stable Handles: Handles remain valid across compact() operations.
//  The index maintains an internal indirection table that maps stable IDs
//  to current storage slots.
//

import XCTest
@testable import VectorAccelerate

final class HandleLifecycleTests: XCTestCase {

    // MARK: - Handle Creation Tests

    /// Handles from insert should be valid.
    func testInsertReturnsValidHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let isValid = await index.contains(handle)
        XCTAssertTrue(isValid, "Newly inserted handle should be valid")
    }

    /// Multiple inserts should return distinct handles.
    func testMultipleInsertsReturnDistinctHandles() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])

        XCTAssertNotEqual(handle1, handle2, "Different inserts should return different handles")
    }

    // MARK: - contains() Tests

    /// contains() should return true for valid handle.
    func testContainsValidHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let containsResult = await index.contains(handle)
        XCTAssertTrue(containsResult, "Valid handle should be contained")
    }

    /// contains() should return false for .invalid handle.
    func testContainsInvalidHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let invalidHandle = VectorHandle.invalid
        let containsInvalid = await index.contains(invalidHandle)
        XCTAssertFalse(containsInvalid, "Invalid handle should not be contained")
    }

    /// contains() should return false for deleted handle.
    func testContainsDeletedHandle() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])
        try await index.remove(handle)

        let containsResult = await index.contains(handle)
        XCTAssertFalse(containsResult, "Deleted handle should not be contained")
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

        // Get the slot for this handle
        let slot = await index.slot(for: handle)
        XCTAssertNotNil(slot, "Should have a slot for the handle")

        if let slot = slot {
            let retrievedHandle = await index.handle(for: slot)
            XCTAssertEqual(retrievedHandle, handle, "handle(for:) should return the same handle")
        }
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
        let slot = await index.slot(for: handle)

        try await index.remove(handle)

        // After deletion, the slot should return nil
        if let slot = slot {
            let retrievedHandle = await index.handle(for: slot)
            XCTAssertNil(retrievedHandle, "Deleted slot should return nil for handle(for:)")
        }
    }

    // MARK: - P0.8 Stable Handles - Compaction Tests

    /// With P0.8 stable handles: handles remain valid after compaction.
    func testStableHandlesAfterCompact() async throws {
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

        // Compact (returns Void with P0.8)
        try await index.compact()

        // P0.8: Original handles for kept vectors should STILL BE VALID
        let contains0 = await index.contains(handles[0])
        let contains2 = await index.contains(handles[2])
        let contains4 = await index.contains(handles[4])

        XCTAssertTrue(contains0, "Handle 0 should still be valid after compact (P0.8)")
        XCTAssertTrue(contains2, "Handle 2 should still be valid after compact (P0.8)")
        XCTAssertTrue(contains4, "Handle 4 should still be valid after compact (P0.8)")

        // Deleted handles should still be invalid
        let contains1 = await index.contains(handles[1])
        let contains3 = await index.contains(handles[3])

        XCTAssertFalse(contains1, "Deleted handle 1 should remain invalid")
        XCTAssertFalse(contains3, "Deleted handle 3 should remain invalid")
    }

    /// P0.8: Original handles should retrieve correct vectors after compaction.
    func testVectorRetrievalWithStableHandles() async throws {
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
        try await index.compact()

        // P0.8: Original handles should retrieve correct vectors
        let retrieved0 = try await index.vector(for: handles[0])
        XCTAssertNotNil(retrieved0)
        XCTAssertEqual(Double(retrieved0?[0] ?? -1), 0.0, accuracy: 0.001)

        let retrieved2 = try await index.vector(for: handles[2])
        XCTAssertNotNil(retrieved2)
        XCTAssertEqual(Double(retrieved2?[0] ?? -1), 2.0, accuracy: 0.001)

        let retrieved4 = try await index.vector(for: handles[4])
        XCTAssertNotNil(retrieved4)
        XCTAssertEqual(Double(retrieved4?[0] ?? -1), 4.0, accuracy: 0.001)

        // Deleted handles should return nil
        let retrieved1 = try await index.vector(for: handles[1])
        XCTAssertNil(retrieved1)

        let retrieved3 = try await index.vector(for: handles[3])
        XCTAssertNil(retrieved3)
    }

    /// New inserts after compaction should get fresh handles.
    func testNewHandlesAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert and delete to create fragmentation
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])
        try await index.remove(handle1)

        // Compact
        try await index.compact()

        // Insert a new vector
        let newHandle = try await index.insert([3.0, 0.0, 0.0, 0.0])

        // New handle should be valid
        let isNewValid = await index.contains(newHandle)
        XCTAssertTrue(isNewValid, "New handle should be valid")

        // Old handle (handle2) should still be valid (P0.8)
        let isOldValid = await index.contains(handle2)
        XCTAssertTrue(isOldValid, "Old handle should still be valid (P0.8)")

        // Verify vector retrieval works for both
        let retrievedOld = try await index.vector(for: handle2)
        XCTAssertNotNil(retrievedOld)
        XCTAssertEqual(Double(retrievedOld?[0] ?? -1), 2.0, accuracy: 0.001)

        let retrievedNew = try await index.vector(for: newHandle)
        XCTAssertNotNil(retrievedNew)
        XCTAssertEqual(Double(retrievedNew?[0] ?? -1), 3.0, accuracy: 0.001)
    }

    /// P0.8: stableID never changes for a vector's lifetime.
    func testStableIDNeverChanges() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        let handle0 = try await index.insert([0.0, 0.0, 0.0, 0.0])
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0])

        // Record stableIDs (these should never change)
        let stableID0 = handle0.stableID
        let stableID2 = handle2.stableID

        // Delete middle vector
        try await index.remove(handle1)

        // Compact
        try await index.compact()

        // stableIDs should be unchanged
        XCTAssertEqual(handle0.stableID, stableID0, "stableID should never change")
        XCTAssertEqual(handle2.stableID, stableID2, "stableID should never change")
    }

    // MARK: - Edge Case Compaction Tests

    /// Compaction with no deletions should be a no-op.
    func testCompactWithNoDeletions() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }

        // No deletions - compact should be a no-op
        try await index.compact()

        // All handles should still be valid
        for handle in handles {
            let isValid = await index.contains(handle)
            XCTAssertTrue(isValid, "Handle should remain valid after no-op compact")
        }
    }

    /// Empty index compaction should be a no-op.
    func testCompactEmptyIndex() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        try await index.compact()

        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 0, "Empty index should remain empty")
    }

    /// All vectors deleted then compacted.
    func testCompactWithAllDeleted() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        var handles: [VectorHandle] = []
        for i in 0..<5 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            handles.append(handle)
        }

        for handle in handles {
            try await index.remove(handle)
        }

        // Compact
        try await index.compact()

        // Index should be empty
        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 0)
        XCTAssertEqual(stats.deletedSlots, 0)
    }

    /// Multiple sequential compactions should work correctly.
    /// With P0.8 stable handles, original handles remain valid across compaction.
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

        try await index.compact()

        // With P0.8 stable handles: remaining handles should still be valid
        var stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 3)

        // Original handles 0, 2, 4 should still be valid (P0.8 guarantee)
        var contains0 = await index.contains(handles[0])
        let contains1 = await index.contains(handles[1])
        var contains2 = await index.contains(handles[2])
        let contains3 = await index.contains(handles[3])
        var contains4 = await index.contains(handles[4])
        XCTAssertTrue(contains0)
        XCTAssertFalse(contains1) // Deleted
        XCTAssertTrue(contains2)
        XCTAssertFalse(contains3) // Deleted
        XCTAssertTrue(contains4)

        // Second round: add more, delete some, compact
        var newHandles: [VectorHandle] = []
        for i in 5..<8 {
            let handle = try await index.insert([Float(i), 0.0, 0.0, 0.0])
            newHandles.append(handle)
        }

        // Delete handle[0] (original first one)
        try await index.remove(handles[0])

        try await index.compact()

        // Should have 5 vectors (2 from first round + 3 from second)
        stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 5, "Should have 5 vectors after second compaction")

        // All original remaining handles should still be valid
        contains0 = await index.contains(handles[0])
        contains2 = await index.contains(handles[2])
        contains4 = await index.contains(handles[4])
        XCTAssertFalse(contains0) // Deleted in second round
        XCTAssertTrue(contains2)
        XCTAssertTrue(contains4)

        // All new handles should be valid
        for newHandle in newHandles {
            let isValid = await index.contains(newHandle)
            XCTAssertTrue(isValid)
        }
    }

    // MARK: - Metadata Preservation After Compact

    /// Metadata should be preserved after compaction.
    /// With P0.8 stable handles, original handles remain valid.
    func testMetadataPreservedAfterCompact() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert with metadata
        let handle1 = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["key": "value1"])
        let handle2 = try await index.insert([2.0, 0.0, 0.0, 0.0], metadata: ["key": "value2"])
        let handle3 = try await index.insert([3.0, 0.0, 0.0, 0.0], metadata: ["key": "value3"])

        // Delete middle one
        try await index.remove(handle2)

        // Compact (returns Void with P0.8)
        try await index.compact()

        // With P0.8 stable handles: original handles remain valid
        // Check metadata is preserved using original handles
        let meta1 = await index.metadata(for: handle1)
        XCTAssertEqual(meta1?["key"], "value1")

        let meta3 = await index.metadata(for: handle3)
        XCTAssertEqual(meta3?["key"], "value3")

        // Deleted handle's metadata should be gone
        let meta2 = await index.metadata(for: handle2)
        XCTAssertNil(meta2)
    }

    // MARK: - Search After Compact

    /// Search should work correctly after compaction.
    /// With P0.8 stable handles, original handles remain valid.
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

        // Compact (returns Void with P0.8)
        try await index.compact()

        // Search should find the remaining vectors
        let query: [Float] = [0.0, 0.0, 0.0, 0.0]
        let results = try await index.search(query: query, k: 5)

        XCTAssertEqual(results.count, 3, "Should find 3 remaining vectors")

        // First result should be exact match with distance 0
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 0.001)

        // With P0.8 stable handles: results should use original handles
        let validHandleSet = Set([handles[0], handles[2], handles[4]])
        for result in results {
            XCTAssertTrue(validHandleSet.contains(result.handle),
                          "Search results should use original stable handles")
        }
    }
}
