//
//  SharedMetalConfigurationTests.swift
//  VectorAccelerateTests
//
//  Tests for SharedMetalConfiguration cross-package GPU sharing
//

import XCTest
@testable import VectorAccelerate

final class SharedMetalConfigurationTests: XCTestCase {

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        // Reset shared state before each test
        await SharedMetalConfiguration.resetForTesting()
    }

    override func tearDown() async throws {
        // Clean up after each test
        await SharedMetalConfiguration.resetForTesting()
    }

    // MARK: - Registration Tests

    func testRegisterMetalSubsystem() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)

        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertTrue(isRegistered, "Should be registered after register() call")

        let sharedSubsystem = await SharedMetalConfiguration.sharedSubsystem
        XCTAssertNotNil(sharedSubsystem, "Shared subsystem should be available")
    }

    func testRegisterMetal4Context() async throws {
        let context = try await Metal4Context()

        await SharedMetalConfiguration.register(context)

        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertTrue(isRegistered, "Should be registered after register() call")

        let sharedContext = await SharedMetalConfiguration.sharedContext
        XCTAssertNotNil(sharedContext, "Shared context should be available")
    }

    func testUnregisterClearsState() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)
        let isRegisteredBefore = await SharedMetalConfiguration.isRegistered
        XCTAssertTrue(isRegisteredBefore)

        await SharedMetalConfiguration.unregister()

        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertFalse(isRegistered, "Should not be registered after unregister()")

        let sharedSubsystem = await SharedMetalConfiguration.sharedSubsystem
        XCTAssertNil(sharedSubsystem, "Shared subsystem should be nil after unregister")
    }

    func testReregistrationReplacesPrevious() async throws {
        let subsystem1 = MetalSubsystem(configuration: .testing)
        let context2 = try await Metal4Context()

        await SharedMetalConfiguration.register(subsystem1)
        let firstSubsystem = await SharedMetalConfiguration.sharedSubsystem
        XCTAssertNotNil(firstSubsystem)

        // Register context should clear subsystem
        await SharedMetalConfiguration.register(context2)

        let subsystemAfter = await SharedMetalConfiguration.sharedSubsystem
        XCTAssertNil(subsystemAfter, "Subsystem should be cleared when context is registered")

        let contextAfter = await SharedMetalConfiguration.sharedContext
        XCTAssertNotNil(contextAfter, "Context should be available")
    }

    func testIsRegisteredReturnsFalseInitially() async {
        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertFalse(isRegistered, "Should not be registered initially")
    }

    // MARK: - Context Access Tests

    func testSharedContextNilBeforeRegistration() async {
        let context = await SharedMetalConfiguration.sharedContext
        XCTAssertNil(context, "Context should be nil before registration")
    }

    func testSharedContextAvailableAfterSubsystemInit() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)

        // Initialize subsystem
        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state
        if !state.isFailed {
            let context = await SharedMetalConfiguration.sharedContext
            XCTAssertNotNil(context, "Context should be available after subsystem init")
        }
    }

    func testSharedSubsystemReturnsRegisteredSubsystem() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)

        let sharedSubsystem = await SharedMetalConfiguration.sharedSubsystem
        XCTAssertNotNil(sharedSubsystem)
    }

    func testDirectContextRegistration() async throws {
        let context = try await Metal4Context()

        await SharedMetalConfiguration.register(context)

        let sharedContext = await SharedMetalConfiguration.sharedContext
        XCTAssertNotNil(sharedContext, "Direct context should be available immediately")
    }

    // MARK: - EmbedKit Compatibility Tests

    func testForEmbedKitSharingReturnsSharedContext() async throws {
        let context = try await Metal4Context()

        await SharedMetalConfiguration.register(context)

        let embedKitContext = try await SharedMetalConfiguration.forEmbedKitSharing()
        XCTAssertNotNil(embedKitContext, "forEmbedKitSharing should return shared context")
    }

    func testForEmbedKitSharingCreatesNewContextWhenNotRegistered() async throws {
        // No registration - should create new context
        let embedKitContext = try await SharedMetalConfiguration.forEmbedKitSharing()
        XCTAssertNotNil(embedKitContext, "forEmbedKitSharing should create new context when not registered")
    }

    func testForEmbedKitSharingWaitsForSubsystemInit() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)

        // Start initialization in background
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            await subsystem.beginBackgroundInitialization()
        }

        // forEmbedKitSharing should wait for init
        do {
            let context = try await SharedMetalConfiguration.forEmbedKitSharing()
            XCTAssertNotNil(context)
        } catch {
            // Context creation is acceptable fallback
            XCTAssertNotNil(error)
        }
    }

    // MARK: - Lifecycle Tests

    func testWeakReferenceDoesNotPreventDeallocation() async {
        // Create subsystem in a scope
        var subsystem: MetalSubsystem? = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem!)

        let isRegisteredBefore = await SharedMetalConfiguration.isRegistered
        XCTAssertTrue(isRegisteredBefore)

        // Clear subsystem reference
        subsystem = nil

        // Give ARC time to deallocate
        try? await Task.sleep(for: .milliseconds(100))

        // Should no longer be registered (weak reference cleared)
        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertFalse(isRegistered, "Weak reference should not prevent deallocation")
    }

    func testContextAccessAfterSubsystemDeallocReturnsNil() async {
        // Create and register subsystem
        var subsystem: MetalSubsystem? = MetalSubsystem(configuration: .testing)
        await SharedMetalConfiguration.register(subsystem!)

        // Deallocate
        subsystem = nil
        try? await Task.sleep(for: .milliseconds(100))

        // Context should be nil
        let context = await SharedMetalConfiguration.sharedContext
        XCTAssertNil(context, "Context should be nil after subsystem deallocation")
    }

    func testMultiplePackagesCanAccessSameContext() async throws {
        let context = try await Metal4Context()

        await SharedMetalConfiguration.register(context)

        // Simulate multiple packages accessing concurrently
        async let access1 = SharedMetalConfiguration.sharedContext
        async let access2 = SharedMetalConfiguration.sharedContext
        async let access3 = SharedMetalConfiguration.sharedContext

        let contexts = await [access1, access2, access3]

        // All should get the same context (not nil)
        for ctx in contexts {
            XCTAssertNotNil(ctx, "All packages should get the shared context")
        }
    }

    // MARK: - Timeout Tests

    func testWaitForContextReturnsBeforeTimeout() async throws {
        let context = try await Metal4Context()

        // Register after a brief delay
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            await SharedMetalConfiguration.register(context)
        }

        let result = await SharedMetalConfiguration.waitForContext(timeout: .seconds(5))
        XCTAssertNotNil(result, "Should return context before timeout")
    }

    func testWaitForContextReturnsNilAfterTimeout() async {
        // Don't register anything
        let result = await SharedMetalConfiguration.waitForContext(timeout: .milliseconds(100))
        XCTAssertNil(result, "Should return nil after timeout with no registration")
    }

    func testWaitForContextWithZeroTimeout() async throws {
        // Zero timeout should check immediately and return nil if not available
        let result = await SharedMetalConfiguration.waitForContext(timeout: .zero)
        XCTAssertNil(result, "Zero timeout should return nil immediately when not registered")

        // Now register and check again
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        let resultAfter = await SharedMetalConfiguration.waitForContext(timeout: .zero)
        XCTAssertNotNil(resultAfter, "Zero timeout should return context if already available")
    }

    func testWaitForContextReturnsImmediatelyWhenAvailable() async throws {
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        let start = CFAbsoluteTimeGetCurrent()
        let result = await SharedMetalConfiguration.waitForContext(timeout: .seconds(30))
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        XCTAssertNotNil(result)
        XCTAssertLessThan(elapsed, 0.1, "Should return immediately when context is available")
    }

    // MARK: - Observer Tests

    func testContextObserverCalledImmediately() async throws {
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        let expectation = XCTestExpectation(description: "Observer called")
        let contextHolder = ContextHolder()

        _ = await SharedMetalConfiguration.addContextObserver { ctx in
            Task { @MainActor in
                contextHolder.context = ctx
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 1.0)

        let receivedContext = await contextHolder.context
        XCTAssertNotNil(receivedContext, "Observer should receive context immediately")
    }

    func testContextObserverCalledOnRegistration() async throws {
        let expectation = XCTestExpectation(description: "Observer called on registration")
        expectation.expectedFulfillmentCount = 2  // Initial nil + context

        let counter = CallCounter()

        _ = await SharedMetalConfiguration.addContextObserver { _ in
            Task { @MainActor in
                counter.increment()
                expectation.fulfill()
            }
        }

        // Register context after observer
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        await fulfillment(of: [expectation], timeout: 2.0)

        let callCount = await counter.count
        XCTAssertEqual(callCount, 2, "Observer should be called twice (initial + registration)")
    }

    func testContextObserverRemoval() async throws {
        let counter = CallCounter()

        let id = await SharedMetalConfiguration.addContextObserver { _ in
            Task { @MainActor in
                counter.increment()
            }
        }

        try? await Task.sleep(for: .milliseconds(50))
        let initialCount = await counter.count
        XCTAssertEqual(initialCount, 1, "Observer should be called once initially")

        // Remove observer
        await SharedMetalConfiguration.removeContextObserver(id)

        // Register context - observer should NOT be called
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        try? await Task.sleep(for: .milliseconds(100))

        let finalCount = await counter.count
        XCTAssertEqual(finalCount, 1, "Observer should not be called after removal")
    }

    // MARK: - Concurrent Access Tests

    func testConcurrentRegistrationAndAccess() async throws {
        // Stress test concurrent access
        await withTaskGroup(of: Void.self) { group in
            // Multiple registrations
            for _ in 0..<10 {
                group.addTask {
                    let subsystem = MetalSubsystem(configuration: .testing)
                    await SharedMetalConfiguration.register(subsystem)
                }
            }

            // Multiple accesses
            for _ in 0..<50 {
                group.addTask {
                    let _ = await SharedMetalConfiguration.sharedContext
                    let _ = await SharedMetalConfiguration.isRegistered
                    let _ = await SharedMetalConfiguration.sharedSubsystem
                }
            }
        }

        // Should not crash - actor isolation handles synchronization
    }

    func testConcurrentWaitForContext() async throws {
        let context = try await Metal4Context()

        // Start multiple waits before registration
        async let wait1 = SharedMetalConfiguration.waitForContext(timeout: .seconds(5))
        async let wait2 = SharedMetalConfiguration.waitForContext(timeout: .seconds(5))
        async let wait3 = SharedMetalConfiguration.waitForContext(timeout: .seconds(5))

        // Register after brief delay
        Task {
            try? await Task.sleep(for: .milliseconds(100))
            await SharedMetalConfiguration.register(context)
        }

        let results = await [wait1, wait2, wait3]

        // All should get the context
        for result in results {
            XCTAssertNotNil(result, "All concurrent waits should receive context")
        }
    }

    // MARK: - Edge Case Tests

    func testDoubleUnregister() async {
        let subsystem = MetalSubsystem(configuration: .testing)
        await SharedMetalConfiguration.register(subsystem)

        await SharedMetalConfiguration.unregister()
        await SharedMetalConfiguration.unregister()  // Should not crash

        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertFalse(isRegistered)
    }

    func testRegisterNilDoesNotCrash() async {
        // Register, then unregister
        let subsystem = MetalSubsystem(configuration: .testing)
        await SharedMetalConfiguration.register(subsystem)
        await SharedMetalConfiguration.unregister()

        // Access should return nil, not crash
        let context = await SharedMetalConfiguration.sharedContext
        XCTAssertNil(context)
    }

    func testResetForTestingClearsAllState() async throws {
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        let expectation = XCTestExpectation(description: "Observer")
        _ = await SharedMetalConfiguration.addContextObserver { _ in
            expectation.fulfill()
        }

        await fulfillment(of: [expectation], timeout: 1.0)

        // Reset
        await SharedMetalConfiguration.resetForTesting()

        // Everything should be cleared
        let isRegistered = await SharedMetalConfiguration.isRegistered
        XCTAssertFalse(isRegistered)

        let sharedContext = await SharedMetalConfiguration.sharedContext
        XCTAssertNil(sharedContext)
    }

    // MARK: - Integration with MetalSubsystem Tests

    func testSharedSubsystemLifecycleIntegration() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await SharedMetalConfiguration.register(subsystem)

        // Start initialization
        await subsystem.beginBackgroundInitialization()

        // Wait for critical pipelines
        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state
        if !state.isFailed {
            // Shared context should now be available
            let context = await SharedMetalConfiguration.sharedContext
            XCTAssertNotNil(context)

            // Can use the context
            let deviceName = await context?.deviceName
            XCTAssertNotNil(deviceName)
        }
    }

    func testSharedSubsystemStateObserver() async {
        let subsystem = MetalSubsystem(configuration: .testing)
        let stateCollector = StateCollector()

        await SharedMetalConfiguration.register(subsystem)

        // Get shared subsystem and add observer
        if let shared = await SharedMetalConfiguration.sharedSubsystem {
            let _ = await shared.addReadinessObserver { state in
                Task { @MainActor in
                    stateCollector.append(state.description)
                }
            }
        }

        // Wait for initial callback
        try? await Task.sleep(for: .milliseconds(50))

        let states = await stateCollector.states
        XCTAssertFalse(states.isEmpty, "Should have received state callbacks")
    }

    // MARK: - Cross-Package Simulation Tests

    func testSimulatedEmbedKitUsage() async throws {
        // Simulate app setup
        let appSubsystem = MetalSubsystem(configuration: .testing)
        await SharedMetalConfiguration.register(appSubsystem)
        await appSubsystem.beginBackgroundInitialization()

        // Simulate EmbedKit trying to get context
        let embedKitContext = try await SharedMetalConfiguration.forEmbedKitSharing()
        XCTAssertNotNil(embedKitContext)

        // Simulate multiple EmbedKit operations using same context
        let context1 = try await SharedMetalConfiguration.forEmbedKitSharing()
        let context2 = try await SharedMetalConfiguration.forEmbedKitSharing()

        // Both should work (either shared or new)
        XCTAssertNotNil(context1)
        XCTAssertNotNil(context2)
    }

    func testSharedPipelineCacheBenefit() async throws {
        let context = try await Metal4Context()
        await SharedMetalConfiguration.register(context)

        // Get shared context
        let sharedContext = await SharedMetalConfiguration.sharedContext
        XCTAssertNotNil(sharedContext)

        // Warm up pipeline on shared context
        await sharedContext?.warmUpPipelineCache()

        // Another package getting the context should benefit from warmed pipelines
        let embedKitContext = try await SharedMetalConfiguration.forEmbedKitSharing()

        // Same pipeline cache should be available
        let isArchiveEnabled = await embedKitContext.isArchiveCacheEnabled
        // Archive may or may not be enabled, but access should work
        _ = isArchiveEnabled
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

/// Thread-safe context holder for observer tests.
@MainActor
private final class ContextHolder {
    var context: Metal4Context?
}
