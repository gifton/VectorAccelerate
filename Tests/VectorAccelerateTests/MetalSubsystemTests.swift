//
//  MetalSubsystemTests.swift
//  VectorAccelerateTests
//
//  Tests for MetalSubsystem lifecycle management
//

import XCTest
@testable import VectorAccelerate

final class MetalSubsystemTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInitialStateIsDormant() async {
        let subsystem = MetalSubsystem()

        let state = await subsystem.state
        XCTAssertEqual(state.description, "dormant")
    }

    func testInitIsInstant() async {
        // Measure that init completes quickly (no Metal work)
        let start = CFAbsoluteTimeGetCurrent()
        let _ = MetalSubsystem()
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Init should complete in < 10ms (being generous for test overhead)
        XCTAssertLessThan(elapsed, 0.010, "MetalSubsystem.init() should be instant (< 10ms)")
    }

    func testInitWithConfiguration() async {
        let config = MetalSubsystemConfiguration(
            allowRuntimeCompilation: true,
            pipelineRegistry: .minimal
        )
        let subsystem = MetalSubsystem(configuration: config)

        let state = await subsystem.state
        XCTAssertEqual(state.description, "dormant")
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = MetalSubsystemConfiguration.default

        XCTAssertTrue(config.allowRuntimeCompilation)
        XCTAssertNil(config.metallibURL)
        XCTAssertEqual(config.backgroundQoS, .utility)
        XCTAssertEqual(config.criticalQoS, .userInitiated)
        XCTAssertEqual(config.warmupIdleTimeout, 2.0)
        XCTAssertTrue(config.respectThermalState)
    }

    func testDevelopmentConfiguration() {
        let config = MetalSubsystemConfiguration.development

        XCTAssertTrue(config.allowRuntimeCompilation)
        XCTAssertEqual(config.warmupIdleTimeout, 1.0)
        XCTAssertFalse(config.respectThermalState)
    }

    func testTestingConfiguration() {
        let config = MetalSubsystemConfiguration.testing

        XCTAssertTrue(config.allowRuntimeCompilation)
        XCTAssertEqual(config.backgroundQoS, .userInitiated)
        XCTAssertEqual(config.warmupIdleTimeout, 0.1)
        XCTAssertFalse(config.respectThermalState)
    }

    func testProductionConfigurationRequiresMetallib() {
        let fakeURL = URL(fileURLWithPath: "/fake/path/default.metallib")
        let config = MetalSubsystemConfiguration.production(metallibURL: fakeURL)

        XCTAssertFalse(config.allowRuntimeCompilation)
        XCTAssertNotNil(config.metallibURL)
    }

    func testConfigurationValidation() throws {
        // Valid production config
        let validURL = URL(fileURLWithPath: "/fake/path/default.metallib")
        let validConfig = MetalSubsystemConfiguration.production(metallibURL: validURL)
        XCTAssertNoThrow(try validConfig.validateForProduction())

        // Development config doesn't need validation
        let devConfig = MetalSubsystemConfiguration.development
        XCTAssertNoThrow(try devConfig.validateForProduction())

        // Invalid production config (no metallib)
        let invalidConfig = MetalSubsystemConfiguration(
            allowRuntimeCompilation: false,
            metallibURL: nil
        )
        XCTAssertThrowsError(try invalidConfig.validateForProduction())
    }

    func testConfigurationIsValid() {
        let validConfig = MetalSubsystemConfiguration.default
        XCTAssertTrue(validConfig.isValid)

        let invalidConfig = MetalSubsystemConfiguration(
            allowRuntimeCompilation: false,
            metallibURL: nil
        )
        XCTAssertFalse(invalidConfig.isValid)
    }

    // MARK: - State Machine Tests

    func testBeginBackgroundInitialization() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        // Start initialization
        await subsystem.beginBackgroundInitialization()

        // State should no longer be dormant
        let state = await subsystem.state
        XCTAssertNotEqual(state.description, "dormant")
    }

    func testBeginBackgroundInitializationIdempotent() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        // Call multiple times
        await subsystem.beginBackgroundInitialization()
        await subsystem.beginBackgroundInitialization()
        await subsystem.beginBackgroundInitialization()

        // Should still be in a valid state
        let state = await subsystem.state
        XCTAssertNotEqual(state.description, "dormant")
    }

    func testStateTransitionsToCriticalReady() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state
        // Should be at least criticalReady (could be fullyReady or failed)
        XCTAssertTrue(
            state.isAtLeast(.criticalReady) || state.isFailed,
            "State should be criticalReady or failed, got: \(state)"
        )
    }

    // MARK: - Observer Tests

    func testObserverCalledImmediately() async {
        let subsystem = MetalSubsystem()
        let stateCollector = StateCollector()

        let expectation = XCTestExpectation(description: "Observer called immediately")

        let _ = await subsystem.addReadinessObserver { state in
            Task { @MainActor in
                stateCollector.append(state.description)
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 1.0)

        let states = await stateCollector.states
        XCTAssertEqual(states.count, 1)
        XCTAssertEqual(states.first, "dormant")
    }

    func testObserverCalledOnStateChange() async {
        let subsystem = MetalSubsystem(configuration: .testing)
        let stateCollector = StateCollector()

        let expectation = XCTestExpectation(description: "Observer called on state change")
        expectation.expectedFulfillmentCount = 2  // Initial + at least one transition

        let _ = await subsystem.addReadinessObserver { state in
            Task { @MainActor in
                stateCollector.append(state.description)
                expectation.fulfill()
            }
        }

        await subsystem.beginBackgroundInitialization()

        await fulfillment(of: [expectation], timeout: 5.0)

        let states = await stateCollector.states
        XCTAssertGreaterThan(states.count, 1)
        XCTAssertEqual(states.first, "dormant")
    }

    func testObserverRemoval() async {
        let subsystem = MetalSubsystem()
        let counter = CallCounter()

        let id = await subsystem.addReadinessObserver { _ in
            Task { @MainActor in
                counter.increment()
            }
        }

        // Wait for initial callback
        try? await Task.sleep(for: .milliseconds(50))
        let initialCount = await counter.count
        XCTAssertEqual(initialCount, 1)

        // Remove observer
        await subsystem.removeReadinessObserver(id)

        // Start initialization - observer should not be called
        await subsystem.beginBackgroundInitialization()

        // Wait a bit
        try? await Task.sleep(for: .milliseconds(100))

        // Call count should still be 1 (only initial call)
        let finalCount = await counter.count
        XCTAssertEqual(finalCount, 1)
    }

    // MARK: - Nonisolated Checks Tests

    func testIsMetalAvailableInitially() async {
        let subsystem = MetalSubsystem()

        // Initially false
        XCTAssertFalse(subsystem.isMetalAvailable)
    }

    func testAreCriticalPipelinesReadyInitially() async {
        let subsystem = MetalSubsystem()

        // Initially false
        XCTAssertFalse(subsystem.areCriticalPipelinesReady)
    }

    func testNonisolatedChecksAfterInit() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state

        // If not failed, checks should reflect ready state
        if !state.isFailed {
            XCTAssertTrue(subsystem.isMetalAvailable)
            XCTAssertTrue(subsystem.areCriticalPipelinesReady)
        }
    }

    // MARK: - Context Access Tests

    func testContextNilBeforeInit() async {
        let subsystem = MetalSubsystem()

        let context = await subsystem.context
        XCTAssertNil(context)
    }

    func testContextAvailableAfterInit() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state
        let context = await subsystem.context

        if !state.isFailed {
            XCTAssertNotNil(context)
        }
    }

    // MARK: - Fallback Tests

    func testFallbackAlwaysAvailable() async {
        let subsystem = MetalSubsystem()

        // Fallback should be available immediately
        let fallback = await subsystem.fallback
        XCTAssertNotNil(fallback)

        // Should work even before initialization
        let result = fallback.l2Distance(from: [1, 0], to: [0, 1])
        XCTAssertEqual(result, sqrt(2), accuracy: 0.001)
    }

    // MARK: - MetalReadinessState Tests

    func testStateDescriptions() {
        XCTAssertEqual(MetalReadinessState.dormant.description, "dormant")
        XCTAssertEqual(MetalReadinessState.initializing.description, "initializing")
        XCTAssertEqual(MetalReadinessState.deviceReady.description, "deviceReady")
        XCTAssertEqual(MetalReadinessState.criticalReady.description, "criticalReady")
        XCTAssertEqual(MetalReadinessState.fullyReady.description, "fullyReady")
    }

    func testStateIsAtLeast() {
        XCTAssertTrue(MetalReadinessState.fullyReady.isAtLeast(.dormant))
        XCTAssertTrue(MetalReadinessState.fullyReady.isAtLeast(.criticalReady))
        XCTAssertTrue(MetalReadinessState.fullyReady.isAtLeast(.fullyReady))

        XCTAssertTrue(MetalReadinessState.criticalReady.isAtLeast(.dormant))
        XCTAssertTrue(MetalReadinessState.criticalReady.isAtLeast(.deviceReady))
        XCTAssertFalse(MetalReadinessState.criticalReady.isAtLeast(.fullyReady))

        XCTAssertTrue(MetalReadinessState.dormant.isAtLeast(.dormant))
        XCTAssertFalse(MetalReadinessState.dormant.isAtLeast(.initializing))
    }

    func testStateIsFailed() {
        XCTAssertFalse(MetalReadinessState.dormant.isFailed)
        XCTAssertFalse(MetalReadinessState.criticalReady.isFailed)

        let error = NSError(domain: "test", code: 0)
        XCTAssertTrue(MetalReadinessState.failed(error).isFailed)
    }

    func testStateError() {
        XCTAssertNil(MetalReadinessState.dormant.error)
        XCTAssertNil(MetalReadinessState.criticalReady.error)

        let error = NSError(domain: "test", code: 42)
        let failedState = MetalReadinessState.failed(error)
        XCTAssertNotNil(failedState.error)
    }

    // MARK: - Error Tests

    func testMetalSubsystemError() {
        let notInitialized = MetalSubsystemError.notInitialized
        XCTAssertNotNil(notInitialized.errorDescription)

        let metallibRequired = MetalSubsystemError.metallibRequired
        XCTAssertNotNil(metallibRequired.errorDescription)

        let archiveNotAvailable = MetalSubsystemError.archiveNotAvailable
        XCTAssertNotNil(archiveNotAvailable.errorDescription)

        let contextFailed = MetalSubsystemError.contextCreationFailed("test reason")
        XCTAssertNotNil(contextFailed.errorDescription)
        XCTAssertTrue(contextFailed.errorDescription!.contains("test reason"))
    }

    // MARK: - QoS Extension Tests

    func testQoSTaskPriority() {
        XCTAssertEqual(QualityOfService.userInteractive.taskPriority, .high)
        XCTAssertEqual(QualityOfService.userInitiated.taskPriority, .high)
        XCTAssertEqual(QualityOfService.utility.taskPriority, .medium)
        XCTAssertEqual(QualityOfService.background.taskPriority, .low)
        XCTAssertEqual(QualityOfService.default.taskPriority, .medium)
    }

    // MARK: - Integration Tests

    func testFullLifecycle() async {
        let subsystem = MetalSubsystem(configuration: .testing)
        let stateCollector = StateCollector()

        let _ = await subsystem.addReadinessObserver { state in
            Task { @MainActor in
                stateCollector.append(state.description)
            }
        }

        // Wait for initial callback
        try? await Task.sleep(for: .milliseconds(50))

        // Should start dormant
        let initialStates = await stateCollector.states
        XCTAssertEqual(initialStates.first, "dormant")

        // Request critical pipelines
        await subsystem.requestCriticalPipelines()

        // Wait for state transitions to complete
        try? await Task.sleep(for: .milliseconds(100))

        // Should have transitioned through states
        let finalStates = await stateCollector.states
        XCTAssertGreaterThan(finalStates.count, 1, "Should have multiple state transitions")

        // Final state should be ready or failed
        let finalState = await subsystem.state
        XCTAssertTrue(
            finalState.isAtLeast(.criticalReady) || finalState.isFailed,
            "Final state should be criticalReady+ or failed"
        )
    }

    func testWaitForDeviceReady() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await subsystem.waitForDeviceReady()

        let state = await subsystem.state

        XCTAssertTrue(
            state.isAtLeast(.deviceReady) || state.isFailed,
            "State should be at least deviceReady or failed"
        )
    }
}

// MARK: - Test Helpers

/// Thread-safe state collector for observer tests.
@MainActor
private final class StateCollector {
    private var _states: [String] = []

    var states: [String] {
        _states
    }

    func append(_ state: String) {
        _states.append(state)
    }
}

/// Thread-safe call counter for observer tests.
@MainActor
private final class CallCounter {
    private var _count: Int = 0

    var count: Int {
        _count
    }

    func increment() {
        _count += 1
    }
}
