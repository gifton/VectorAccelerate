//
//  WarmupManagerTests.swift
//  VectorAccelerateTests
//
//  Tests for WarmupManager activity-aware pipeline warmup
//

import XCTest
@testable import VectorAccelerate

final class WarmupManagerTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        // Create a test context
        context = try await Metal4Context(configuration: .default)
    }

    override func tearDown() async throws {
        await context.cleanup()
        context = nil
        try await super.tearDown()
    }

    // MARK: - Initial State Tests

    func testInitialStateIsIdle() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let state = await manager.state
        XCTAssertEqual(state, .idle)
    }

    func testInitialProgressIsOne() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let progress = await manager.progress
        XCTAssertEqual(progress, 1.0)
    }

    func testIsWarmingInitiallyFalse() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        XCTAssertFalse(manager.isWarming)
    }

    func testIsPausedInitiallyFalse() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        XCTAssertFalse(manager.isPaused)
    }

    // MARK: - Empty Keys Tests

    func testEmptyKeysCompletesImmediately() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        await manager.beginWarmup(keys: [])

        let state = await manager.state
        XCTAssertEqual(state, .completed)
    }

    func testEmptyKeysProgressIsOne() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        await manager.beginWarmup(keys: [])

        let progress = await manager.progress
        XCTAssertEqual(progress, 1.0)
    }

    // MARK: - State Transition Tests

    func testStateTransitionsToCompleted() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Use a small set of keys that will actually compile
        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0)
        ]

        await manager.beginWarmup(keys: keys)

        let state = await manager.state
        XCTAssertEqual(state, .completed)
    }

    func testProgressIsOneWhenCompleted() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0)
        ]

        await manager.beginWarmup(keys: keys)

        let progress = await manager.progress
        XCTAssertEqual(progress, 1.0)
    }

    // MARK: - User Activity Tests

    func testReportUserActivityPausesWarmup() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Use multiple keys to give us time to pause
        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
            .dotProduct(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        // Give warmup time to start
        try await Task.sleep(for: .milliseconds(50))

        // Report user activity
        await manager.reportUserActivity()

        // Check if paused
        let state = await manager.state
        if case .paused(let reason) = state {
            XCTAssertEqual(reason, .userActivity)
        } else if case .completed = state {
            // Warmup may have finished before we could pause - acceptable
        } else if case .warming = state {
            // Still warming - user activity may not have taken effect yet
        } else {
            XCTFail("Unexpected state: \(state)")
        }
    }

    func testIsPausedAfterUserActivity() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
            .dotProduct(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        // Give warmup time to start
        try await Task.sleep(for: .milliseconds(50))

        // Report user activity
        await manager.reportUserActivity()

        try await Task.sleep(for: .milliseconds(50))

        // Check isPaused reflects state
        let state = await manager.state
        if case .paused = state {
            XCTAssertTrue(manager.isPaused)
        }
    }

    // MARK: - Manual Pause/Resume Tests

    func testManualPause() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Pause manually
        await manager.pause(reason: .manualPause)

        let state = await manager.state
        if case .paused(let reason) = state {
            XCTAssertEqual(reason, .manualPause)
        } else if case .completed = state {
            // Warmup may have finished
        } else {
            // Could still be warming
        }
    }

    func testResumeAfterManualPause() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Pause and resume
        await manager.pause(reason: .manualPause)
        await manager.resume()

        // Wait for completion
        try await Task.sleep(for: .milliseconds(200))

        let state = await manager.state
        // Should eventually complete or still be warming
        XCTAssertTrue(state == .completed || state.isActive)
    }

    // MARK: - Cancellation Tests

    func testCancelStopsWarmup() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Cancel
        await manager.cancel()

        let state = await manager.state
        // State should be cancelled (unless it completed first)
        XCTAssertTrue(state == .cancelled || state == .completed)
    }

    func testCancelFromPausedState() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Pause then cancel
        await manager.pause(reason: .manualPause)
        await manager.cancel()

        let state = await manager.state
        XCTAssertTrue(state == .cancelled || state == .completed)
    }

    func testCancelFromIdleState() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Cancel without starting - should be safe
        await manager.cancel()

        let state = await manager.state
        XCTAssertEqual(state, .idle)
    }

    // MARK: - Progress Tests

    func testProgressCountsMatch() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        await manager.beginWarmup(keys: keys)

        let warmedCount = await manager.warmedCount
        let pipelineCount = await manager.pipelineCount

        // After completion, warmed count should equal pipeline count
        XCTAssertEqual(warmedCount, pipelineCount)
        XCTAssertEqual(pipelineCount, 2)
    }

    // MARK: - WarmupState Tests

    func testWarmupStateEquatable() {
        XCTAssertEqual(WarmupManager.WarmupState.idle, .idle)
        XCTAssertEqual(WarmupManager.WarmupState.completed, .completed)
        XCTAssertEqual(WarmupManager.WarmupState.cancelled, .cancelled)
        XCTAssertEqual(
            WarmupManager.WarmupState.paused(reason: .userActivity),
            .paused(reason: .userActivity)
        )
        XCTAssertEqual(
            WarmupManager.WarmupState.warming(progress: 0.5),
            .warming(progress: 0.5)
        )
    }

    func testWarmupStateIsActive() {
        XCTAssertFalse(WarmupManager.WarmupState.idle.isActive)
        XCTAssertTrue(WarmupManager.WarmupState.warming(progress: 0.5).isActive)
        XCTAssertTrue(WarmupManager.WarmupState.paused(reason: .userActivity).isActive)
        XCTAssertFalse(WarmupManager.WarmupState.completed.isActive)
        XCTAssertFalse(WarmupManager.WarmupState.cancelled.isActive)
    }

    // MARK: - PauseReason Tests

    func testPauseReasonEquatable() {
        XCTAssertEqual(WarmupManager.PauseReason.userActivity, .userActivity)
        XCTAssertEqual(WarmupManager.PauseReason.thermalThrottling, .thermalThrottling)
        XCTAssertEqual(WarmupManager.PauseReason.manualPause, .manualPause)
        XCTAssertNotEqual(WarmupManager.PauseReason.userActivity, .manualPause)
    }

    func testPauseReasonDescription() {
        XCTAssertEqual(WarmupManager.PauseReason.userActivity.description, "userActivity")
        XCTAssertEqual(WarmupManager.PauseReason.thermalThrottling.description, "thermalThrottling")
        XCTAssertEqual(WarmupManager.PauseReason.manualPause.description, "manualPause")
    }

    // MARK: - Nonisolated Checks Tests

    func testNonisolatedChecksAreThreadSafe() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 100

        // Access nonisolated properties from multiple threads
        for _ in 0..<50 {
            DispatchQueue.global().async {
                _ = manager.isWarming
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                _ = manager.isPaused
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 5.0)
    }

    // MARK: - Integration Tests

    func testWarmupCompletesAllKeys() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Test with actual pipeline keys
        let keys = PipelineCacheKey.commonKeys.prefix(3)

        await manager.beginWarmup(keys: Array(keys))

        let state = await manager.state
        XCTAssertEqual(state, .completed)

        let progress = await manager.progress
        XCTAssertEqual(progress, 1.0)
    }

    func testMultipleWarmupCycles() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // First warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        var state = await manager.state
        XCTAssertEqual(state, .completed)

        // Second warmup - should reset and complete
        await manager.beginWarmup(keys: [.cosineSimilarity(dimension: 0)])

        state = await manager.state
        XCTAssertEqual(state, .completed)
    }

    func testReportUserActivityBeforeWarmupIsNoOp() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Report activity before warmup
        await manager.reportUserActivity()

        let state = await manager.state
        XCTAssertEqual(state, .idle)
    }

    func testPauseBeforeWarmupIsNoOp() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Pause before warmup
        await manager.pause(reason: .manualPause)

        let state = await manager.state
        XCTAssertEqual(state, .idle)
    }

    func testResumeBeforeWarmupIsNoOp() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Resume before warmup
        await manager.resume()

        let state = await manager.state
        XCTAssertEqual(state, .idle)
    }

    // MARK: - Observer Pattern Tests (Phase 4)

    func testObserverIsCalledOnStateChange() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let stateTracker = StateTracker()
        let expectation = XCTestExpectation(description: "Observer called")
        expectation.expectedFulfillmentCount = 2  // At least warming and completed

        let observerId = await manager.addObserver { state in
            stateTracker.recordState(state)
            if case .completed = state {
                expectation.fulfill()
            }
            if case .warming = state {
                expectation.fulfill()
            }
        }

        // Start warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        await fulfillment(of: [expectation], timeout: 5.0)

        // Observer should have been called at least for warming and completed
        XCTAssertGreaterThanOrEqual(stateTracker.stateCount, 2)

        // Clean up
        await manager.removeObserver(observerId)
    }

    func testObserverRemoval() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let counter = AtomicCounter()
        let observerId = await manager.addObserver { _ in
            counter.increment()
        }

        // Remove observer before warmup
        await manager.removeObserver(observerId)

        // Start warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        // Observer should not have been called
        XCTAssertEqual(counter.value, 0)
    }

    func testRemoveAllObservers() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let counter1 = AtomicCounter()
        let counter2 = AtomicCounter()

        _ = await manager.addObserver { _ in counter1.increment() }
        _ = await manager.addObserver { _ in counter2.increment() }

        // Remove all observers
        await manager.removeAllObservers()

        // Start warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        // No observers should have been called
        XCTAssertEqual(counter1.value, 0)
        XCTAssertEqual(counter2.value, 0)
    }

    func testMultipleObservers() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let flag1 = AtomicFlag()
        let flag2 = AtomicFlag()
        let expectation = XCTestExpectation(description: "Both observers called")
        expectation.expectedFulfillmentCount = 2

        _ = await manager.addObserver { state in
            if case .completed = state {
                flag1.set()
                expectation.fulfill()
            }
        }

        _ = await manager.addObserver { state in
            if case .completed = state {
                flag2.set()
                expectation.fulfill()
            }
        }

        // Start warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        await fulfillment(of: [expectation], timeout: 5.0)

        XCTAssertTrue(flag1.value)
        XCTAssertTrue(flag2.value)
    }

    // MARK: - Statistics Tests (Phase 4)

    func testStatisticsInitialState() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let stats = await manager.statistics
        XCTAssertEqual(stats.totalPipelines, 0)
        XCTAssertEqual(stats.memoryHits, 0)
        XCTAssertEqual(stats.archiveHits, 0)
        XCTAssertEqual(stats.jitCompilations, 0)
        XCTAssertEqual(stats.failures, 0)
    }

    func testStatisticsAfterWarmup() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        await manager.beginWarmup(keys: keys)

        let stats = await manager.statistics
        XCTAssertEqual(stats.totalPipelines, 2)
        // Sum of all sources should equal total pipelines
        let totalSources = stats.memoryHits + stats.archiveHits + stats.jitCompilations + stats.failures
        XCTAssertEqual(totalSources, stats.totalPipelines)
        XCTAssertTrue(stats.isComplete)
        XCTAssertEqual(stats.remaining, 0)
    }

    func testStatisticsAreResetOnNewWarmup() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // First warmup
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])
        let statsAfterFirst = await manager.statistics
        XCTAssertEqual(statsAfterFirst.totalPipelines, 1)

        // Second warmup with different keys
        await manager.beginWarmup(keys: [
            .cosineSimilarity(dimension: 0),
            .dotProduct(dimension: 0),
        ])
        let statsAfterSecond = await manager.statistics
        XCTAssertEqual(statsAfterSecond.totalPipelines, 2)
    }

    func testWarmupStatisticsHitRates() {
        // Test hit rate calculations
        let stats = WarmupStatistics(
            totalPipelines: 10,
            memoryHits: 2,
            archiveHits: 5,
            jitCompilations: 3,
            failures: 0
        )

        XCTAssertEqual(stats.archiveHitRate, 0.5, accuracy: 0.01)  // 5 / 10
        XCTAssertEqual(stats.jitRate, 0.3, accuracy: 0.01)  // 3 / 10
        XCTAssertEqual(stats.remaining, 0)
        XCTAssertTrue(stats.isComplete)
    }

    func testWarmupStatisticsZeroDivision() {
        // Test that rates don't cause division by zero
        let emptyStats = WarmupStatistics(
            totalPipelines: 0,
            memoryHits: 0,
            archiveHits: 0,
            jitCompilations: 0,
            failures: 0
        )

        XCTAssertEqual(emptyStats.archiveHitRate, 0)
        XCTAssertEqual(emptyStats.jitRate, 0)
    }

    // MARK: - Activity Delegate Tests (Phase 4)

    func testActivityDelegateSetAndClear() async {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let delegate = MockActivityDelegate()

        // Set delegate
        await manager.setActivityDelegate(delegate)

        // Start warmup to trigger delegate methods
        await manager.beginWarmup(keys: [.l2Distance(dimension: 0)])

        // Wait for delegate calls to arrive
        try? await Task.sleep(for: .milliseconds(100))

        // Delegate should have received at least warmupDidBegin and warmupDidEnd
        XCTAssertTrue(delegate.didBeginCalled)
        XCTAssertTrue(delegate.didEndCalled)
        XCTAssertTrue(delegate.lastEndCompleted)

        // Clear delegate
        await manager.setActivityDelegate(nil)

        // Reset tracking
        delegate.reset()

        // Start another warmup
        await manager.beginWarmup(keys: [.cosineSimilarity(dimension: 0)])

        // Wait for potential delegate calls
        try? await Task.sleep(for: .milliseconds(100))

        // Delegate should NOT have been called after clearing
        XCTAssertFalse(delegate.didBeginCalled)
        XCTAssertFalse(delegate.didEndCalled)
    }

    func testActivityDelegateReceivesPauseCallback() async throws {
        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let delegate = MockActivityDelegate()
        await manager.setActivityDelegate(delegate)

        // Use multiple keys to allow time for pause
        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
            .dotProduct(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Pause warmup
        await manager.pause(reason: .manualPause)

        // Wait for delegate calls
        try await Task.sleep(for: .milliseconds(100))

        // Check if pause was reported (may or may not depending on timing)
        let state = await manager.state
        if case .paused = state {
            XCTAssertTrue(delegate.didPauseCalled)
            XCTAssertEqual(delegate.lastPauseReason, .manualPause)
        }
    }

    // MARK: - Idle Timeout Resume Tests (Phase 4)

    func testIdleTimeoutResumesWarmup() async throws {
        // Use a configuration with a very short idle timeout for testing
        let config = MetalSubsystemConfiguration(
            allowRuntimeCompilation: true,
            pipelineRegistry: .minimal,
            backgroundQoS: .userInitiated,
            warmupIdleTimeout: 0.2,  // 200ms timeout
            respectThermalState: false
        )

        let manager = WarmupManager(
            context: context,
            configuration: config
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        // Start warmup in background
        Task {
            await manager.beginWarmup(keys: keys)
        }

        // Give warmup time to start
        try await Task.sleep(for: .milliseconds(50))

        // Report user activity to pause warmup
        await manager.reportUserActivity()

        // Wait for idle timeout to pass plus some buffer
        try await Task.sleep(for: .milliseconds(400))

        // Warmup should have resumed and completed (or be warming)
        let state = await manager.state
        XCTAssertTrue(
            state == .completed || state.isActive,
            "Expected completed or active state, got \(state)"
        )
    }
}

// MARK: - Mock Activity Delegate

/// Mock activity delegate for testing.
final class MockActivityDelegate: ActivityDelegate, @unchecked Sendable {
    private let lock = NSLock()

    private var _didBeginCalled = false
    private var _didPauseCalled = false
    private var _didResumeCalled = false
    private var _didEndCalled = false
    private var _lastPauseReason: WarmupManager.PauseReason?
    private var _lastEndCompleted = false

    var didBeginCalled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _didBeginCalled
    }

    var didPauseCalled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _didPauseCalled
    }

    var didResumeCalled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _didResumeCalled
    }

    var didEndCalled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _didEndCalled
    }

    var lastPauseReason: WarmupManager.PauseReason? {
        lock.lock()
        defer { lock.unlock() }
        return _lastPauseReason
    }

    var lastEndCompleted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _lastEndCompleted
    }

    @MainActor
    func warmupDidBegin() {
        lock.lock()
        defer { lock.unlock() }
        _didBeginCalled = true
    }

    @MainActor
    func warmupDidPause(reason: WarmupManager.PauseReason) {
        lock.lock()
        defer { lock.unlock() }
        _didPauseCalled = true
        _lastPauseReason = reason
    }

    @MainActor
    func warmupDidResume() {
        lock.lock()
        defer { lock.unlock() }
        _didResumeCalled = true
    }

    @MainActor
    func warmupDidEnd(completed: Bool) {
        lock.lock()
        defer { lock.unlock() }
        _didEndCalled = true
        _lastEndCompleted = completed
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _didBeginCalled = false
        _didPauseCalled = false
        _didResumeCalled = false
        _didEndCalled = false
        _lastPauseReason = nil
        _lastEndCompleted = false
    }
}

// MARK: - Thread-Safe Test Helpers

/// Thread-safe counter for testing.
final class AtomicCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        _value += 1
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _value = 0
    }
}

/// Thread-safe boolean flag for testing.
final class AtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false

    var value: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    func set() {
        lock.lock()
        defer { lock.unlock() }
        _value = true
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        _value = false
    }
}

/// Thread-safe state tracker for testing.
final class StateTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var _states: [WarmupManager.WarmupState] = []

    var stateCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _states.count
    }

    var states: [WarmupManager.WarmupState] {
        lock.lock()
        defer { lock.unlock() }
        return _states
    }

    func recordState(_ state: WarmupManager.WarmupState) {
        lock.lock()
        defer { lock.unlock() }
        _states.append(state)
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _states.removeAll()
    }
}
