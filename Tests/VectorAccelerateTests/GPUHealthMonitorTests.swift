//
//  GPUHealthMonitorTests.swift
//  VectorAccelerateTests
//
//  Tests for GPUHealthMonitor GPU health tracking and fallback management
//

import XCTest
@testable import VectorAccelerate

final class GPUHealthMonitorTests: XCTestCase {

    // MARK: - Initialization Tests

    func testDefaultInitialization() async {
        let monitor = GPUHealthMonitor()

        let status = await monitor.getHealthStatus()

        XCTAssertEqual(status.totalFailureCount, 0)
        XCTAssertNil(status.lastFailure)
        XCTAssertEqual(status.recoveryAttempts, 0)
        XCTAssertTrue(status.isHealthy)
        XCTAssertEqual(status.degradationLevel, .none)
        XCTAssertTrue(status.disabledOperations.isEmpty)
    }

    func testCustomConfigurationInitialization() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 5,
            disableDurationSeconds: 600,
            recoveryCheckInterval: 120,
            successesForRecovery: 4
        )
        let monitor = GPUHealthMonitor(configuration: config)

        let returnedConfig = await monitor.getConfiguration()

        XCTAssertEqual(returnedConfig.maxFailuresBeforeDisable, 5)
        XCTAssertEqual(returnedConfig.disableDurationSeconds, 600)
        XCTAssertEqual(returnedConfig.recoveryCheckInterval, 120)
        XCTAssertEqual(returnedConfig.successesForRecovery, 4)
    }

    func testPredefinedConfigurations() {
        // Test that predefined configurations compile and have expected values
        let defaultConfig = GPUHealthMonitorConfiguration.default
        XCTAssertEqual(defaultConfig.maxFailuresBeforeDisable, 3)
        XCTAssertEqual(defaultConfig.disableDurationSeconds, 300)

        let aggressiveConfig = GPUHealthMonitorConfiguration.aggressive
        XCTAssertEqual(aggressiveConfig.maxFailuresBeforeDisable, 2)
        XCTAssertGreaterThan(aggressiveConfig.disableDurationSeconds, defaultConfig.disableDurationSeconds)

        let lenientConfig = GPUHealthMonitorConfiguration.lenient
        XCTAssertEqual(lenientConfig.maxFailuresBeforeDisable, 5)
        XCTAssertLessThan(lenientConfig.disableDurationSeconds, defaultConfig.disableDurationSeconds)
    }

    // MARK: - Degradation Level Tests

    func testDegradationLevelComparison() {
        XCTAssertLessThan(GPUDegradationLevel.none, GPUDegradationLevel.minor)
        XCTAssertLessThan(GPUDegradationLevel.minor, GPUDegradationLevel.moderate)
        XCTAssertLessThan(GPUDegradationLevel.moderate, GPUDegradationLevel.severe)
    }

    func testDegradationLevelDescription() {
        XCTAssertEqual(GPUDegradationLevel.none.description, "none")
        XCTAssertEqual(GPUDegradationLevel.minor.description, "minor")
        XCTAssertEqual(GPUDegradationLevel.moderate.description, "moderate")
        XCTAssertEqual(GPUDegradationLevel.severe.description, "severe")
    }

    func testDegradationLevelShouldFallback() {
        XCTAssertFalse(GPUDegradationLevel.none.shouldFallback)
        XCTAssertFalse(GPUDegradationLevel.minor.shouldFallback)
        XCTAssertFalse(GPUDegradationLevel.moderate.shouldFallback)
        XCTAssertTrue(GPUDegradationLevel.severe.shouldFallback)
    }

    func testDegradationLevelHasIssues() {
        XCTAssertFalse(GPUDegradationLevel.none.hasIssues)
        XCTAssertTrue(GPUDegradationLevel.minor.hasIssues)
        XCTAssertTrue(GPUDegradationLevel.moderate.hasIssues)
        XCTAssertTrue(GPUDegradationLevel.severe.hasIssues)
    }

    // MARK: - Failure Recording Tests

    func testSingleFailureRecording() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await monitor.recordFailure(operation: "test_op", error: error)

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.totalFailureCount, 1)
        XCTAssertNotNil(status.lastFailure)
        XCTAssertEqual(status.operationFailureCounts["test_op"], 1)
    }

    func testMultipleFailuresIncrementCount() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        for _ in 0..<5 {
            await monitor.recordFailure(operation: "test_op", error: error)
        }

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.totalFailureCount, 5)
        XCTAssertEqual(status.operationFailureCounts["test_op"], 5)
    }

    func testFailuresPerOperationAreIsolated() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op2", error: error)

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.totalFailureCount, 3)
        XCTAssertEqual(status.operationFailureCounts["op1"], 2)
        XCTAssertEqual(status.operationFailureCounts["op2"], 1)
    }

    // MARK: - Degradation Progression Tests

    func testDegradationProgressionWithFailures() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 6,
            degradationThresholds: .init(minor: 1, moderate: 3, severe: 5)
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Initial: none
        var level = await monitor.getDegradationLevel(for: "test_op")
        XCTAssertEqual(level, .none)

        // 1 failure: minor
        await monitor.recordFailure(operation: "test_op", error: error)
        level = await monitor.getDegradationLevel(for: "test_op")
        XCTAssertEqual(level, .minor)

        // 3 failures: moderate
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)
        level = await monitor.getDegradationLevel(for: "test_op")
        XCTAssertEqual(level, .moderate)

        // 5 failures: severe
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)
        level = await monitor.getDegradationLevel(for: "test_op")
        XCTAssertEqual(level, .severe)
    }

    // MARK: - Success Recording Tests

    func testSuccessResetsFailureCount() async {
        let config = GPUHealthMonitorConfiguration(
            successesForRecovery: 1
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Record some failures
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)

        var status = await monitor.getHealthStatus()
        XCTAssertEqual(status.operationFailureCounts["test_op"], 2)

        // Record success
        await monitor.recordSuccess(operation: "test_op")

        status = await monitor.getHealthStatus()
        // Failure count should be reduced by 1
        XCTAssertEqual(status.operationFailureCounts["test_op"], 1)
    }

    func testMultipleSuccessesRequiredForRecovery() async {
        let config = GPUHealthMonitorConfiguration(
            successesForRecovery: 3
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Record failures
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)

        var status = await monitor.getHealthStatus()
        XCTAssertEqual(status.operationFailureCounts["test_op"], 2)

        // Record 2 successes (not enough for recovery)
        await monitor.recordSuccess(operation: "test_op")
        await monitor.recordSuccess(operation: "test_op")

        status = await monitor.getHealthStatus()
        XCTAssertEqual(status.operationFailureCounts["test_op"], 2) // Still 2

        // Record 3rd success
        await monitor.recordSuccess(operation: "test_op")

        status = await monitor.getHealthStatus()
        XCTAssertEqual(status.operationFailureCounts["test_op"], 1) // Now reduced
    }

    func testFailureResetsSuccessCounter() async {
        let config = GPUHealthMonitorConfiguration(
            successesForRecovery: 3
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        await monitor.recordFailure(operation: "test_op", error: error)

        // Record 2 successes
        await monitor.recordSuccess(operation: "test_op")
        await monitor.recordSuccess(operation: "test_op")

        // Then fail - this should reset success counter
        await monitor.recordFailure(operation: "test_op", error: error)

        // Record 2 more successes - not enough because counter was reset
        await monitor.recordSuccess(operation: "test_op")
        await monitor.recordSuccess(operation: "test_op")

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.operationFailureCounts["test_op"], 2) // Still 2 failures
    }

    // MARK: - Operation Disabling Tests

    func testOperationDisabledAfterMaxFailures() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 3
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Should not be disabled initially
        var shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertFalse(shouldFallback)

        // Record 3 failures
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)

        // Should now be disabled
        shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertTrue(shouldFallback)

        let status = await monitor.getHealthStatus()
        XCTAssertTrue(status.disabledOperations.contains("test_op"))
    }

    func testDisabledOperationInStatus() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 2
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op2", error: error)
        await monitor.recordFailure(operation: "op2", error: error)

        let status = await monitor.getHealthStatus()
        XCTAssertTrue(status.disabledOperations.contains("op1"))
        XCTAssertTrue(status.disabledOperations.contains("op2"))
        XCTAssertEqual(status.disabledOperations.count, 2)
    }

    // MARK: - Fallback Recommendation Tests

    func testFallbackRecommendedAtModerateLevel() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 10,
            degradationThresholds: .init(minor: 1, moderate: 3, severe: 5)
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // 1-2 failures: not recommended
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)

        var recommended = await monitor.isFallbackRecommended(operation: "test_op")
        XCTAssertFalse(recommended)

        // 3+ failures: recommended
        await monitor.recordFailure(operation: "test_op", error: error)

        recommended = await monitor.isFallbackRecommended(operation: "test_op")
        XCTAssertTrue(recommended)
    }

    // MARK: - Reset Tests

    func testResetClearsAllState() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        // Record some failures
        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op2", error: error)

        // Reset
        await monitor.reset()

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.totalFailureCount, 0)
        XCTAssertNil(status.lastFailure)
        XCTAssertEqual(status.recoveryAttempts, 0)
        XCTAssertTrue(status.isHealthy)
        XCTAssertTrue(status.disabledOperations.isEmpty)
        XCTAssertTrue(status.operationFailureCounts.isEmpty)
    }

    func testResetOperationClearsSpecificOperation() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op2", error: error)

        // Reset only op1
        await monitor.resetOperation("op1")

        let status = await monitor.getHealthStatus()
        XCTAssertNil(status.operationFailureCounts["op1"])
        XCTAssertEqual(status.operationFailureCounts["op2"], 1)
    }

    // MARK: - Manual Enable/Disable Tests

    func testManualDisableOperation() async {
        let monitor = GPUHealthMonitor()

        await monitor.disableOperation("test_op")

        let shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertTrue(shouldFallback)
    }

    func testManualEnableOperation() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 2
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Disable via failures
        await monitor.recordFailure(operation: "test_op", error: error)
        await monitor.recordFailure(operation: "test_op", error: error)

        var shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertTrue(shouldFallback)

        // Manually enable
        await monitor.enableOperation("test_op")

        shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertFalse(shouldFallback)
    }

    // MARK: - Recent Errors Tests

    func testRecentErrorsTracking() async {
        let monitor = GPUHealthMonitor()

        await monitor.recordFailure(operation: "op1", error: TestError.first)
        await monitor.recordFailure(operation: "op2", error: TestError.second)

        let errors = await monitor.getRecentErrors()
        XCTAssertEqual(errors.count, 2)
        XCTAssertEqual(errors[0].operation, "op1")
        XCTAssertEqual(errors[1].operation, "op2")
    }

    func testRecentErrorsLimitedTo10() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        // Record 15 errors
        for i in 0..<15 {
            await monitor.recordFailure(operation: "op\(i)", error: error)
        }

        let errors = await monitor.getRecentErrors()
        XCTAssertEqual(errors.count, 10) // Limited to 10
    }

    // MARK: - Health Status Summary Tests

    func testHealthStatusSummaryFormat() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await monitor.recordFailure(operation: "test_op", error: error)

        let status = await monitor.getHealthStatus()
        let summary = status.summary

        XCTAssertTrue(summary.contains("GPU Health Status"))
        XCTAssertTrue(summary.contains("test_op"))
    }

    // MARK: - Overall Health Tests

    func testOverallDegradationLevelIsMaximum() async {
        let config = GPUHealthMonitorConfiguration(
            degradationThresholds: .init(minor: 1, moderate: 3, severe: 5)
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // op1: 1 failure (minor)
        await monitor.recordFailure(operation: "op1", error: error)

        // op2: 4 failures (moderate)
        for _ in 0..<4 {
            await monitor.recordFailure(operation: "op2", error: error)
        }

        let overallLevel = await monitor.getOverallDegradationLevel()
        XCTAssertEqual(overallLevel, .moderate) // Maximum of all operations
    }

    func testIsHealthyReturnsFalseWithAnyFailures() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        var healthy = await monitor.isHealthy()
        XCTAssertTrue(healthy)

        await monitor.recordFailure(operation: "test_op", error: error)

        healthy = await monitor.isHealthy()
        XCTAssertFalse(healthy)
    }

    // MARK: - Concurrent Access Tests

    func testConcurrentFailureRecording() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    await monitor.recordFailure(operation: "op\(i % 10)", error: error)
                }
            }
        }

        let status = await monitor.getHealthStatus()
        XCTAssertEqual(status.totalFailureCount, 100)
    }

    func testConcurrentSuccessAndFailureRecording() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    if i % 2 == 0 {
                        await monitor.recordFailure(operation: "test_op", error: error)
                    } else {
                        await monitor.recordSuccess(operation: "test_op")
                    }
                }
            }
        }

        // Should complete without crashing - exact state depends on ordering
        let status = await monitor.getHealthStatus()
        XCTAssertNotNil(status)
    }

    func testConcurrentStatusReads() async {
        let monitor = GPUHealthMonitor()
        let error = TestError.generic

        // Record some initial state
        await monitor.recordFailure(operation: "test_op", error: error)

        await withTaskGroup(of: GPUHealthStatus.self) { group in
            for _ in 0..<100 {
                group.addTask {
                    await monitor.getHealthStatus()
                }
            }

            for await status in group {
                XCTAssertNotNil(status)
            }
        }
    }

    // MARK: - Recovery After Disable Duration Tests

    func testRecoveryAttemptAfterDisableDuration() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 1,
            disableDurationSeconds: 0.1 // Very short for testing
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        // Disable via failure
        await monitor.recordFailure(operation: "test_op", error: error)

        var shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertTrue(shouldFallback)

        // Wait for disable duration to expire
        try? await Task.sleep(for: .milliseconds(150))

        // Should allow recovery attempt
        shouldFallback = await monitor.shouldFallbackToCPU(operation: "test_op")
        XCTAssertFalse(shouldFallback)
    }

    // MARK: - Disabled Operation Count Tests

    func testGetDisabledOperationCount() async {
        let config = GPUHealthMonitorConfiguration(
            maxFailuresBeforeDisable: 1
        )
        let monitor = GPUHealthMonitor(configuration: config)
        let error = TestError.generic

        var count = await monitor.getDisabledOperationCount()
        XCTAssertEqual(count, 0)

        await monitor.recordFailure(operation: "op1", error: error)
        await monitor.recordFailure(operation: "op2", error: error)

        count = await monitor.getDisabledOperationCount()
        XCTAssertEqual(count, 2)
    }
}

// MARK: - Test Helpers

private enum TestError: Error {
    case generic
    case first
    case second
}
