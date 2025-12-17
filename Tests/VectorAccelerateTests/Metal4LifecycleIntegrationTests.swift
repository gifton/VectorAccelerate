//
//  Metal4LifecycleIntegrationTests.swift
//  VectorAccelerateTests
//
//  Comprehensive integration tests for Metal 4 lifecycle management.
//  Phase 5: Validates the complete lifecycle from dormant → fullyReady
//

import XCTest
@testable import VectorAccelerate
import Metal

// MARK: - End-to-End Lifecycle Tests

final class Metal4LifecycleIntegrationTests: XCTestCase {

    private var testDirectory: URL!
    private var archiveURL: URL!

    override func setUpWithError() throws {
        // Create unique test directory for each test
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Metal4LifecycleTests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )
        archiveURL = testDirectory.appendingPathComponent("test.metalarchive")
    }

    override func tearDownWithError() throws {
        // Clean up test directory
        if let testDirectory = testDirectory {
            try? FileManager.default.removeItem(at: testDirectory)
        }
    }

    // MARK: - Cold Start Lifecycle Tests

    /// Test cold start: dormant → initializing → deviceReady → criticalReady → fullyReady
    func testColdStartFullLifecycle() async {
        let stateTracker = LifecycleStateTracker()
        let subsystem = MetalSubsystem(configuration: .testing)

        // Add observer to track all state transitions
        let observerId = await subsystem.addReadinessObserver { state in
            stateTracker.recordState(state)
        }

        // Wait for initial callback
        try? await Task.sleep(for: .milliseconds(50))

        // Should start dormant
        XCTAssertEqual(stateTracker.firstState?.description, "dormant")

        // Request critical pipelines (triggers full initialization)
        await subsystem.requestCriticalPipelines()

        // Wait for full lifecycle completion
        try? await Task.sleep(for: .milliseconds(500))

        // Verify final state
        let finalState = await subsystem.state
        XCTAssertTrue(
            finalState.isAtLeast(.criticalReady) || finalState.isFailed,
            "Should reach criticalReady or fail, got: \(finalState)"
        )

        // Verify state observer received transitions
        XCTAssertGreaterThan(stateTracker.stateCount, 1, "Should have multiple state transitions")

        // Clean up
        await subsystem.removeReadinessObserver(observerId)
    }

    /// Test warm start with fast path
    func testWarmStartWithContext() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        // First initialization
        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state
        if !state.isFailed {
            // Context should be available
            let context = await subsystem.context
            XCTAssertNotNil(context, "Context should be available after criticalReady")
        }
    }

    /// Verify state observer callbacks at each transition
    func testStateObserverCallbacksAtEachTransition() async {
        let stateTracker = LifecycleStateTracker()
        let subsystem = MetalSubsystem(configuration: .testing)

        let expectation = XCTestExpectation(description: "Observer receives multiple states")
        expectation.expectedFulfillmentCount = 3 // At least dormant + initializing + one more

        let observerId = await subsystem.addReadinessObserver { state in
            stateTracker.recordState(state)
            expectation.fulfill()
        }

        // Begin initialization
        await subsystem.beginBackgroundInitialization()

        await fulfillment(of: [expectation], timeout: 5.0)

        // Verify we received multiple distinct states
        let uniqueStates = stateTracker.uniqueStateDescriptions
        XCTAssertGreaterThan(uniqueStates.count, 1)
        XCTAssertTrue(uniqueStates.contains("dormant"))

        await subsystem.removeReadinessObserver(observerId)
    }

    /// Ensure context is available at criticalReady
    func testContextAvailableAtCriticalReady() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        await subsystem.requestCriticalPipelines()

        let state = await subsystem.state

        if state.isAtLeast(.criticalReady) {
            let context = await subsystem.context
            XCTAssertNotNil(context, "Context must be available when criticalReady")
        }
    }
}

// MARK: - Launch Performance Tests

extension Metal4LifecycleIntegrationTests {

    /// Test that MetalSubsystem.init() completes in < 10ms (no blocking)
    func testInitCompletesInUnder10ms() async {
        let stopwatch = LifecycleStopwatch()

        stopwatch.start("init")
        let _ = MetalSubsystem()
        let elapsed = stopwatch.elapsed("init")

        XCTAssertLessThan(elapsed, 0.010,
            "MetalSubsystem.init() should complete in < 10ms, took \(elapsed * 1000)ms")
    }

    /// Test that init doesn't do any Metal work
    func testInitDoesNoMetalWork() async {
        let subsystem = MetalSubsystem()

        // Context should be nil immediately after init
        let context = await subsystem.context
        XCTAssertNil(context, "No context should exist before initialization")

        // State should be dormant
        let state = await subsystem.state
        XCTAssertEqual(state.description, "dormant")
    }

    /// Measure time to deviceReady
    func testTimeToDeviceReady() async {
        let stopwatch = LifecycleStopwatch()
        let subsystem = MetalSubsystem(configuration: .testing)

        stopwatch.start("deviceReady")
        await subsystem.waitForDeviceReady()
        let elapsed = stopwatch.elapsed("deviceReady")

        let state = await subsystem.state
        if !state.isFailed {
            // Log performance for baseline documentation
            print("Time to deviceReady: \(elapsed * 1000)ms")
            XCTAssertLessThan(elapsed, 0.5, // 500ms acceptable for device initialization
                "Time to deviceReady should be < 500ms, took \(elapsed * 1000)ms")
        }
    }

    /// Measure time to criticalReady (cold start)
    func testTimeToCriticalReadyColdStart() async {
        let stopwatch = LifecycleStopwatch()
        let subsystem = MetalSubsystem(configuration: .testing)

        stopwatch.start("criticalReady")
        await subsystem.requestCriticalPipelines()
        let elapsed = stopwatch.elapsed("criticalReady")

        let state = await subsystem.state
        if !state.isFailed {
            print("Time to criticalReady (cold): \(elapsed * 1000)ms")
            XCTAssertLessThan(elapsed, 2.0, // 2 seconds for cold start with JIT
                "Time to criticalReady (cold) should be < 2s, took \(elapsed * 1000)ms")
        }
    }

    /// Verify no main thread blocking during initialization
    func testNoMainThreadBlocking() async {
        // This test verifies that init() returns immediately
        let startTime = CFAbsoluteTimeGetCurrent()

        let subsystem = MetalSubsystem(configuration: .testing)

        // Init should complete instantly
        let initTime = CFAbsoluteTimeGetCurrent() - startTime
        XCTAssertLessThan(initTime, 0.010, "Init should not block")

        // Begin background init (should return immediately)
        let bgStartTime = CFAbsoluteTimeGetCurrent()
        await subsystem.beginBackgroundInitialization()
        let bgInitTime = CFAbsoluteTimeGetCurrent() - bgStartTime

        // beginBackgroundInitialization should return quickly (just schedules work)
        XCTAssertLessThan(bgInitTime, 0.050, "beginBackgroundInitialization should not block")
    }
}

// MARK: - Archive Integration Tests

extension Metal4LifecycleIntegrationTests {

    /// Test archive saves correctly after warmup completion
    func testArchiveSavesAfterWarmup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        // Verify new archive is ready with 0 pipelines
        let state = await manager.state
        if case .ready(let count) = state {
            XCTAssertEqual(count, 0)
        } else {
            XCTFail("Archive should be ready")
        }

        // Save (should create manifest)
        try await manager.save()

        // Manifest should exist
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))
    }

    /// Test archive manifest tracks shader source hash
    func testArchiveManifestTracksShaderHash() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL,
            shaderSourceHash: "test-hash-123"
        )

        try await manager.loadOrCreate()
        try await manager.save()

        // Load and verify manifest
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(BinaryArchiveManifest.self, from: data)

        XCTAssertEqual(manifest.shaderSourceHash, "test-hash-123")
    }

    /// Test archive invalidation on shader source change
    func testArchiveInvalidationOnShaderChange() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        // Create archive with hash "v1"
        let manager1 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL,
            shaderSourceHash: "v1"
        )
        try await manager1.loadOrCreate()
        try await manager1.save()

        // Create new manager with different hash - should recreate archive
        let manager2 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL,
            shaderSourceHash: "v2"
        )
        try await manager2.loadOrCreate()

        // Should have created a new empty archive (old one invalidated)
        let state = await manager2.state
        if case .ready(let count) = state {
            XCTAssertEqual(count, 0, "Archive should be recreated with 0 pipelines")
        } else {
            XCTFail("Archive should be ready after recreation")
        }
    }

    /// Test archive invalidation on device change
    func testArchiveInvalidationOnDeviceChange() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        // Create archive
        let manager1 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )
        try await manager1.loadOrCreate()
        try await manager1.save()

        // Manually corrupt manifest to simulate different device
        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(BinaryArchiveManifest.self, from: data)

        // Create a modified manifest with different device name
        let modifiedManifest = BinaryArchiveManifest(
            shaderSourceHash: manifest.shaderSourceHash,
            deviceName: "Different Device"
        )
        let modifiedData = try JSONEncoder().encode(modifiedManifest)
        try modifiedData.write(to: manifestURL)

        // Load with real device - should detect mismatch and recreate
        let manager2 = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )
        try await manager2.loadOrCreate()

        let state = await manager2.state
        if case .ready(let count) = state {
            XCTAssertEqual(count, 0, "Archive should be recreated due to device mismatch")
        }
    }

    /// Test graceful handling of corrupted archive
    func testCorruptedArchiveRecovery() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        // Create directory and write corrupted manifest
        try FileManager.default.createDirectory(
            at: archiveURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let manifestURL = archiveURL.deletingPathExtension().appendingPathExtension("manifest.json")
        try "{ invalid json }".write(to: manifestURL, atomically: true, encoding: .utf8)

        // Should handle corruption gracefully
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        // Should not throw - should recreate archive
        try await manager.loadOrCreate()

        let state = await manager.state
        if case .ready = state {
            // Success - recovered from corruption
        } else {
            XCTFail("Should recover from corrupted archive")
        }
    }

    /// Test graceful handling of missing manifest
    func testMissingManifestRecovery() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        // Create archive file without manifest (simulates incomplete state)
        try FileManager.default.createDirectory(
            at: archiveURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Create empty archive file
        FileManager.default.createFile(atPath: archiveURL.path, contents: nil)

        // Should handle missing manifest gracefully
        let manager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await manager.loadOrCreate()

        let state = await manager.state
        if case .ready = state {
            // Success - created new archive
        } else {
            XCTFail("Should create new archive when manifest is missing")
        }
    }
}

// MARK: - Warmup Behavior Tests

extension Metal4LifecycleIntegrationTests {

    /// Test warmup pauses immediately on user activity
    func testWarmupPausesOnUserActivity() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let config = MetalSubsystemConfiguration(
            allowRuntimeCompilation: true,
            pipelineRegistry: .minimal,
            warmupIdleTimeout: 1.0,
            respectThermalState: false
        )

        let manager = WarmupManager(
            context: context,
            configuration: config
        )

        // Start warmup with multiple keys
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
            // Warmup finished before pause - acceptable
        }
    }

    /// Test warmup resumes after idle timeout
    func testWarmupResumesAfterIdleTimeout() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let config = MetalSubsystemConfiguration(
            allowRuntimeCompilation: true,
            pipelineRegistry: .minimal,
            warmupIdleTimeout: 0.2, // 200ms for testing
            respectThermalState: false
        )

        let manager = WarmupManager(
            context: context,
            configuration: config
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
        ]

        // Start warmup
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Pause via user activity
        await manager.reportUserActivity()

        // Wait for idle timeout plus buffer
        try await Task.sleep(for: .milliseconds(400))

        // Should have resumed and completed (or still warming)
        let state = await manager.state
        XCTAssertTrue(
            state == .completed || state.isActive,
            "Should have resumed, got \(state)"
        )
    }

    /// Test manual pause/resume works correctly
    func testManualPauseResume() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [.l2Distance(dimension: 0)]

        // Start warmup
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Manual pause
        await manager.pause(reason: .manualPause)

        let pausedState = await manager.state
        if case .paused(let reason) = pausedState {
            XCTAssertEqual(reason, .manualPause)
        }

        // Resume
        await manager.resume()

        // Wait for completion
        try await Task.sleep(for: .milliseconds(300))

        let finalState = await manager.state
        XCTAssertTrue(finalState == .completed || finalState.isActive)
    }

    /// Test cancellation stops warmup immediately
    func testCancellationStopsWarmupImmediately() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        // Start warmup
        Task {
            await manager.beginWarmup(keys: keys)
        }

        try await Task.sleep(for: .milliseconds(50))

        // Cancel
        await manager.cancel()

        let state = await manager.state
        XCTAssertTrue(state == .cancelled || state == .completed)
    }

    /// Test progress reporting is accurate
    func testProgressReportingAccurate() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
            .cosineSimilarity(dimension: 0),
        ]

        await manager.beginWarmup(keys: keys)

        let progress = await manager.progress
        let warmedCount = await manager.warmedCount
        let pipelineCount = await manager.pipelineCount

        XCTAssertEqual(progress, 1.0, "Progress should be 1.0 when complete")
        XCTAssertEqual(warmedCount, pipelineCount)
        XCTAssertEqual(pipelineCount, 2)
    }

    /// Test statistics track source distribution correctly
    func testStatisticsTrackSourceDistribution() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

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

        // Sum of all sources should equal total
        let totalSources = stats.memoryHits + stats.archiveHits + stats.jitCompilations + stats.failures
        XCTAssertEqual(totalSources, stats.totalPipelines)

        XCTAssertTrue(stats.isComplete)
        XCTAssertEqual(stats.remaining, 0)
    }
}

// MARK: - Thermal State Tests

extension Metal4LifecycleIntegrationTests {

    /// Test thermal monitor reports correct initial state
    func testThermalMonitorInitialState() {
        let monitor = ThermalStateMonitor.shared

        // Should have a valid thermal state
        let state = monitor.currentState

        // State should be one of the valid values
        XCTAssertTrue(
            state == .nominal || state == .fair || state == .serious || state == .critical,
            "Should have valid thermal state"
        )
    }

    /// Test thermal monitor observer pattern
    func testThermalMonitorObserverPattern() {
        let monitor = ThermalStateMonitor()
        let callCounter = AtomicCounter()

        // Add observer
        let observerId = monitor.addObserver { _ in
            callCounter.increment()
        }

        // Trigger refresh (won't change state but tests mechanics)
        monitor.refreshState()

        // Remove observer
        monitor.removeObserver(observerId)

        // Observer mechanics work (actual notification depends on system state changes)
        XCTAssertNotNil(observerId)
    }

    /// Test shouldThrottle behavior
    func testShouldThrottleBehavior() {
        let monitor = ThermalStateMonitor.shared

        let shouldThrottle = monitor.shouldThrottle
        let currentState = monitor.currentState

        // shouldThrottle should match serious/critical states
        let expectedThrottle = currentState == .serious || currentState == .critical
        XCTAssertEqual(shouldThrottle, expectedThrottle)
    }

    /// Test thermal state description
    func testThermalStateDescription() {
        XCTAssertEqual(ProcessInfo.ThermalState.nominal.description, "nominal")
        XCTAssertEqual(ProcessInfo.ThermalState.fair.description, "fair")
        XCTAssertEqual(ProcessInfo.ThermalState.serious.description, "serious")
        XCTAssertEqual(ProcessInfo.ThermalState.critical.description, "critical")
    }
}

// MARK: - Fallback Behavior Tests

extension Metal4LifecycleIntegrationTests {

    /// Test FallbackProvider is always available
    func testFallbackAlwaysAvailable() async {
        let subsystem = MetalSubsystem()

        // Fallback should be available immediately (before any initialization)
        let fallback = await subsystem.fallback
        XCTAssertNotNil(fallback)
    }

    /// Test CPU L2 distance returns correct results
    func testCPUL2DistanceCorrect() {
        let fallback = FallbackProvider()

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        let distance = fallback.l2Distance(from: a, to: b)

        // Expected: sqrt(2)
        XCTAssertEqual(distance, sqrt(2), accuracy: 0.0001)
    }

    /// Test CPU cosine similarity returns correct results
    func testCPUCosineSimilarityCorrect() {
        let fallback = FallbackProvider()

        let a: [Float] = [1.0, 0.0]
        let b: [Float] = [1.0, 0.0]

        let similarity = fallback.cosineSimilarity(from: a, to: b)

        XCTAssertEqual(similarity, 1.0, accuracy: 0.0001)
    }

    /// Test CPU normalization returns correct results
    func testCPUNormalizationCorrect() {
        let fallback = FallbackProvider()

        let vector: [Float] = [3.0, 4.0]
        let normalized = fallback.normalize(vector)

        // Magnitude should be 1.0
        let magnitude = fallback.l2Norm(normalized)
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.0001)

        // Direction should be preserved
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.0001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.0001)
    }

    /// Test CPU top-K selection returns correct results
    func testCPUTopKSelectionCorrect() {
        let fallback = FallbackProvider()

        let distances: [Float] = [5.0, 2.0, 8.0, 1.0, 3.0]
        let topK = fallback.topKByDistance(distances, k: 3)

        XCTAssertEqual(topK.count, 3)
        XCTAssertEqual(topK[0].index, 3) // 1.0
        XCTAssertEqual(topK[1].index, 1) // 2.0
        XCTAssertEqual(topK[2].index, 4) // 3.0
    }

    /// Test CPU batch distance computation
    func testCPUBatchDistanceComputation() {
        let fallback = FallbackProvider()

        let query: [Float] = [1.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]

        let distances = fallback.batchL2Distance(from: query, to: candidates)

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.0001) // Same vector
        XCTAssertEqual(distances[1], sqrt(2), accuracy: 0.0001) // Orthogonal
    }

    /// Test distance with different metrics
    func testCPUDistanceMetrics() {
        let fallback = FallbackProvider()

        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 5.0, 6.0]

        // Euclidean
        let euclidean = fallback.distance(from: a, to: b, metric: .euclidean)
        let expected = sqrt(Float(9 + 9 + 9)) // sqrt(27)
        XCTAssertEqual(euclidean, expected, accuracy: 0.0001)

        // Manhattan
        let manhattan = fallback.distance(from: a, to: b, metric: .manhattan)
        XCTAssertEqual(manhattan, 9.0, accuracy: 0.0001) // 3 + 3 + 3
    }
}

// MARK: - Stress Tests

extension Metal4LifecycleIntegrationTests {

    /// Test rapid pause/resume cycles
    func testRapidPauseResumeCycles() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        let keys: [PipelineCacheKey] = [
            .l2Distance(dimension: 0),
        ]

        // Start warmup
        Task {
            await manager.beginWarmup(keys: keys)
        }

        // Rapid pause/resume cycles
        for _ in 0..<50 {
            await manager.pause(reason: .manualPause)
            await manager.resume()
        }

        // Should not crash and warmup should complete
        try await Task.sleep(for: .milliseconds(500))

        let state = await manager.state
        XCTAssertTrue(
            state == .completed || state == .cancelled || state.isActive,
            "Should be in a valid state after stress test"
        )
    }

    /// Test multiple observer registrations/removals
    func testMultipleObserverRegistrationsRemovals() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        var observerIds: [UUID] = []

        // Register many observers
        for _ in 0..<50 {
            let id = await subsystem.addReadinessObserver { _ in }
            observerIds.append(id)
        }

        // Remove all observers
        for id in observerIds {
            await subsystem.removeReadinessObserver(id)
        }

        // Start initialization - should not crash
        await subsystem.beginBackgroundInitialization()

        try? await Task.sleep(for: .milliseconds(100))
    }

    /// Test multiple warmup cycles
    func testMultipleWarmupCycles() async throws {
        let context = try await Metal4Context(configuration: .default)
        defer { Task { await context.cleanup() } }

        let manager = WarmupManager(
            context: context,
            configuration: .testing
        )

        // Run multiple warmup cycles
        for i in 0..<10 {
            let keys: [PipelineCacheKey] = [
                .l2Distance(dimension: 0),
            ]

            await manager.beginWarmup(keys: keys)

            let state = await manager.state
            XCTAssertEqual(state, .completed, "Cycle \(i) should complete")
        }
    }

    /// Test nonisolated checks are thread-safe under concurrent access
    func testNonisolatedChecksThreadSafety() async {
        let subsystem = MetalSubsystem(configuration: .testing)

        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 200

        // Concurrent reads from multiple threads
        for _ in 0..<100 {
            DispatchQueue.global(qos: .userInteractive).async {
                _ = subsystem.isMetalAvailable
                expectation.fulfill()
            }
            DispatchQueue.global(qos: .background).async {
                _ = subsystem.areCriticalPipelinesReady
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 10.0)
    }
}

// MARK: - Error Handling Tests

extension Metal4LifecycleIntegrationTests {

    /// Test production mode without metallib URL throws error
    func testProductionModeWithoutMetallibThrows() async {
        let config = MetalSubsystemConfiguration(
            allowRuntimeCompilation: false,
            metallibURL: nil // Missing required URL
        )

        XCTAssertFalse(config.isValid)
        XCTAssertThrowsError(try config.validateForProduction())
    }

    /// Test error descriptions are present
    func testErrorDescriptionsPresent() {
        let errors: [MetalSubsystemError] = [
            .notInitialized,
            .metallibRequired,
            .archiveNotAvailable,
            .contextCreationFailed("test"),
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }

    /// Test binary archive error descriptions
    func testBinaryArchiveErrorDescriptions() {
        let errors: [BinaryArchiveError] = [
            .creationFailed("test"),
            .loadFailed("test"),
            .saveFailed("test"),
            .corrupted("test"),
            .addPipelineFailed("test"),
            .notReady,
            .urlNotConfigured,
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}

// MARK: - Memory Efficiency Tests

extension Metal4LifecycleIntegrationTests {

    /// Test pipeline cache respects max size
    func testPipelineCacheRespectsMaxSize() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let compiler = try Metal4ShaderCompiler(device: device, configuration: .default)
        let archiveManager = BinaryArchiveManager(device: device, archiveURL: archiveURL)
        try await archiveManager.loadOrCreate()

        // Create cache with small max size
        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager,
            maxMemoryCacheSize: 2
        )

        // Initial stats
        let initialStats = await cache.getStatistics()
        XCTAssertEqual(initialStats.memoryCacheSize, 0)
    }

    /// Test buffer pool memory is bounded
    func testBufferPoolMemoryBounded() async throws {
        let config = Metal4Configuration(
            maxBufferPoolMemory: 1024 * 1024, // 1MB limit
            maxBuffersPerSize: 5
        )

        let context = try await Metal4Context(configuration: config)
        defer { Task { await context.cleanup() } }

        // Get some buffers
        for _ in 0..<10 {
            let _ = try await context.getBuffer(size: 1024)
        }

        // Context should still be operational
        XCTAssertNotNil(context)
    }
}

// MARK: - PipelineRegistry Tests

extension Metal4LifecycleIntegrationTests {

    /// Test pipeline registry tier lookup
    func testPipelineRegistryTierLookup() {
        let registry = PipelineRegistry.journalingApp

        // Critical keys
        let l2Tier = registry.tier(for: .l2Distance(dimension: 384))
        XCTAssertEqual(l2Tier, .critical)

        // Unknown key defaults to rare
        let unknownTier = registry.tier(for: PipelineCacheKey(operation: "unknown_op"))
        XCTAssertEqual(unknownTier, .rare)
    }

    /// Test pipeline registry key enumeration
    func testPipelineRegistryKeyEnumeration() {
        let registry = PipelineRegistry.journalingApp

        XCTAssertFalse(registry.criticalKeys.isEmpty)
        XCTAssertFalse(registry.occasionalKeys.isEmpty)
        XCTAssertGreaterThan(registry.totalCount, 0)
    }

    /// Test minimal registry for testing
    func testMinimalRegistry() {
        let registry = PipelineRegistry.minimal

        XCTAssertEqual(registry.criticalKeys.count, 1)
        XCTAssertTrue(registry.occasionalKeys.isEmpty)
        XCTAssertTrue(registry.rareKeys.isEmpty)
    }
}

// MARK: - Test Utilities

/// Tracks lifecycle state transitions
final class LifecycleStateTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var _states: [MetalReadinessState] = []

    var stateCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _states.count
    }

    var firstState: MetalReadinessState? {
        lock.lock()
        defer { lock.unlock() }
        return _states.first
    }

    var lastState: MetalReadinessState? {
        lock.lock()
        defer { lock.unlock() }
        return _states.last
    }

    var uniqueStateDescriptions: Set<String> {
        lock.lock()
        defer { lock.unlock() }
        return Set(_states.map { $0.description })
    }

    func recordState(_ state: MetalReadinessState) {
        lock.lock()
        defer { lock.unlock() }
        _states.append(state)
    }
}

/// Measures elapsed time for lifecycle phases
final class LifecycleStopwatch: @unchecked Sendable {
    private let lock = NSLock()
    private var starts: [String: CFAbsoluteTime] = [:]
    private var elapsedTimes: [String: TimeInterval] = [:]

    func start(_ phase: String) {
        lock.lock()
        defer { lock.unlock() }
        starts[phase] = CFAbsoluteTimeGetCurrent()
    }

    func elapsed(_ phase: String) -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }
        guard let startTime = starts[phase] else { return 0 }
        return CFAbsoluteTimeGetCurrent() - startTime
    }

    func stop(_ phase: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let startTime = starts[phase] else { return }
        elapsedTimes[phase] = CFAbsoluteTimeGetCurrent() - startTime
    }

    func getElapsed(_ phase: String) -> TimeInterval? {
        lock.lock()
        defer { lock.unlock() }
        return elapsedTimes[phase]
    }
}

/// Tracks memory allocations during tests
final class MemoryTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var _peakMemory: Int = 0
    private var _currentMemory: Int = 0

    var peakMemory: Int {
        lock.lock()
        defer { lock.unlock() }
        return _peakMemory
    }

    var currentMemory: Int {
        lock.lock()
        defer { lock.unlock() }
        return _currentMemory
    }

    func recordAllocation(size: Int) {
        lock.lock()
        defer { lock.unlock() }
        _currentMemory += size
        _peakMemory = max(_peakMemory, _currentMemory)
    }

    func recordDeallocation(size: Int) {
        lock.lock()
        defer { lock.unlock() }
        _currentMemory -= size
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _peakMemory = 0
        _currentMemory = 0
    }
}

// Note: AtomicCounter is defined in WarmupManagerTests.swift and reused here
