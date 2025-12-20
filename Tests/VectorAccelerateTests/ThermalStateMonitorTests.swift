//
//  ThermalStateMonitorTests.swift
//  VectorAccelerateTests
//
//  Tests for ThermalStateMonitor device thermal monitoring
//

import XCTest
@testable import VectorAccelerate

final class ThermalStateMonitorTests: XCTestCase {

    // MARK: - Basic State Tests

    func testInitialState() {
        let monitor = ThermalStateMonitor()

        // Initial state should match ProcessInfo
        let expected = ProcessInfo.processInfo.thermalState
        XCTAssertEqual(monitor.currentState, expected)
    }

    func testSharedSingleton() {
        let shared1 = ThermalStateMonitor.shared
        let shared2 = ThermalStateMonitor.shared

        // Should be the same instance
        XCTAssertTrue(shared1 === shared2)
    }

    func testCurrentStateIsReadable() {
        let monitor = ThermalStateMonitor()

        // Should be able to read state without crashing
        let state = monitor.currentState
        XCTAssertNotNil(state)
    }

    // MARK: - Throttle Tests

    func testShouldThrottleForNominalState() {
        // Note: We can't control the actual thermal state, but we can test the logic
        // by checking the relationship between currentState and shouldThrottle
        let monitor = ThermalStateMonitor()
        let state = monitor.currentState

        // Verify shouldThrottle matches expected behavior for current state
        let expectedThrottle = state == .serious || state == .critical
        XCTAssertEqual(monitor.shouldThrottle, expectedThrottle)
    }

    func testShouldThrottleThreadSafety() {
        let monitor = ThermalStateMonitor()
        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 100

        // Access shouldThrottle from multiple threads
        for _ in 0..<100 {
            DispatchQueue.global().async {
                _ = monitor.shouldThrottle
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }

    func testCurrentStateThreadSafety() {
        let monitor = ThermalStateMonitor()
        let expectation = XCTestExpectation(description: "Concurrent state access")
        expectation.expectedFulfillmentCount = 100

        // Access currentState from multiple threads
        for _ in 0..<100 {
            DispatchQueue.global().async {
                _ = monitor.currentState
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Observer Tests

    func testAddObserver() {
        let monitor = ThermalStateMonitor()

        let id = monitor.addObserver { _ in
            // Observer closure
        }

        XCTAssertNotNil(id)
    }

    func testRemoveObserver() {
        let monitor = ThermalStateMonitor()

        let id = monitor.addObserver { _ in }

        // Should not crash
        monitor.removeObserver(id)
    }

    func testRemoveObserverMultipleTimes() {
        let monitor = ThermalStateMonitor()

        let id = monitor.addObserver { _ in }

        // Should not crash when removing multiple times
        monitor.removeObserver(id)
        monitor.removeObserver(id)
    }

    func testRemoveNonexistentObserver() {
        let monitor = ThermalStateMonitor()

        // Should not crash when removing nonexistent observer
        monitor.removeObserver(UUID())
    }

    func testMultipleObservers() {
        let monitor = ThermalStateMonitor()

        let id1 = monitor.addObserver { _ in }
        let id2 = monitor.addObserver { _ in }
        let id3 = monitor.addObserver { _ in }

        XCTAssertNotEqual(id1, id2)
        XCTAssertNotEqual(id2, id3)
        XCTAssertNotEqual(id1, id3)

        // Remove all
        monitor.removeObserver(id1)
        monitor.removeObserver(id2)
        monitor.removeObserver(id3)
    }

    // MARK: - ThermalState Description Tests

    func testThermalStateDescriptionNominal() {
        XCTAssertEqual(ProcessInfo.ThermalState.nominal.description, "nominal")
    }

    func testThermalStateDescriptionFair() {
        XCTAssertEqual(ProcessInfo.ThermalState.fair.description, "fair")
    }

    func testThermalStateDescriptionSerious() {
        XCTAssertEqual(ProcessInfo.ThermalState.serious.description, "serious")
    }

    func testThermalStateDescriptionCritical() {
        XCTAssertEqual(ProcessInfo.ThermalState.critical.description, "critical")
    }

    // MARK: - Integration Tests

    func testMonitorDoesNotRetainObserverClosure() {
        let monitor = ThermalStateMonitor()

        let deallocTracker = DeallocTracker()
        do {
            let testObject = TestObject()
            testObject.tracker = deallocTracker
            _ = monitor.addObserver { [weak testObject] _ in
                // Weak capture should not retain
                _ = testObject
            }
        }

        // Object should be deallocated since observer uses weak capture
        // Wait a moment for deallocation
        XCTAssertTrue(deallocTracker.wasDeallocated)
    }

    func testObserverConcurrentAddRemove() {
        let monitor = ThermalStateMonitor()
        let expectation = XCTestExpectation(description: "Concurrent add/remove")
        expectation.expectedFulfillmentCount = 200

        // Add observers from multiple threads
        for _ in 0..<100 {
            DispatchQueue.global().async {
                let id = monitor.addObserver { _ in }
                monitor.removeObserver(id)
                expectation.fulfill()
            }
        }

        // Read state from multiple threads
        for _ in 0..<100 {
            DispatchQueue.global().async {
                _ = monitor.shouldThrottle
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }
}

// MARK: - Test Helpers

/// Sendable tracker for deallocation detection
private final class DeallocTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var _wasDeallocated = false

    var wasDeallocated: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _wasDeallocated
    }

    func markDeallocated() {
        lock.lock()
        defer { lock.unlock() }
        _wasDeallocated = true
    }
}

/// Test object that reports to a tracker when deallocated
private final class TestObject: @unchecked Sendable {
    var tracker: DeallocTracker?

    deinit {
        tracker?.markDeallocated()
    }
}
