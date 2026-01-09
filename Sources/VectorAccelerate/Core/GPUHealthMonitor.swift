//
//  GPUHealthMonitor.swift
//  VectorAccelerate
//
//  GPU health tracking and automatic CPU fallback management.
//
//  Migrated from VectorIndexAccelerated with adaptations for Metal 4 architecture.
//

import Foundation

// MARK: - Degradation Level

/// Represents the current health degradation level of GPU operations.
///
/// The degradation level increases as consecutive failures accumulate and
/// decreases when operations succeed or after recovery timeouts.
public enum GPUDegradationLevel: Int, Sendable, Comparable, CustomStringConvertible {
    /// No degradation - GPU is operating normally
    case none = 0

    /// Minor degradation - 1-2 consecutive failures
    /// GPU is still preferred but may have occasional issues
    case minor = 1

    /// Moderate degradation - 3-4 consecutive failures
    /// Consider using CPU fallback for critical operations
    case moderate = 2

    /// Severe degradation - 5+ consecutive failures
    /// Force CPU fallback until recovery
    case severe = 3

    public static func < (lhs: GPUDegradationLevel, rhs: GPUDegradationLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String {
        switch self {
        case .none: return "none"
        case .minor: return "minor"
        case .moderate: return "moderate"
        case .severe: return "severe"
        }
    }

    /// Whether this degradation level should trigger CPU fallback
    public var shouldFallback: Bool {
        self >= .severe
    }

    /// Whether this degradation level indicates potential issues
    public var hasIssues: Bool {
        self >= .minor
    }
}

// MARK: - Health Status

/// A snapshot of the current GPU health status.
///
/// This struct provides a complete view of the GPU health state at a point in time,
/// including failure counts, degradation levels, and disabled operations.
public struct GPUHealthStatus: Sendable {
    /// Total number of failures recorded across all operations
    public let totalFailureCount: Int

    /// Timestamp of the most recent failure, if any
    public let lastFailure: Date?

    /// Number of recovery attempts made
    public let recoveryAttempts: Int

    /// Whether the GPU is considered healthy overall
    public let isHealthy: Bool

    /// Current degradation level
    public let degradationLevel: GPUDegradationLevel

    /// Set of operations currently disabled due to repeated failures
    public let disabledOperations: Set<String>

    /// Per-operation failure counts
    public let operationFailureCounts: [String: Int]

    /// Per-operation degradation levels
    public let operationDegradationLevels: [String: GPUDegradationLevel]

    /// Human-readable summary of health status
    public var summary: String {
        var result = "GPU Health Status:\n"
        result += "  Overall: \(isHealthy ? "Healthy" : "Degraded")\n"
        result += "  Degradation Level: \(degradationLevel)\n"
        result += "  Total Failures: \(totalFailureCount)\n"
        result += "  Recovery Attempts: \(recoveryAttempts)\n"

        if let lastFailure = lastFailure {
            let formatter = DateFormatter()
            formatter.dateStyle = .short
            formatter.timeStyle = .medium
            result += "  Last Failure: \(formatter.string(from: lastFailure))\n"
        }

        if !disabledOperations.isEmpty {
            result += "  Disabled Operations: \(disabledOperations.sorted().joined(separator: ", "))\n"
        }

        if !operationFailureCounts.isEmpty {
            result += "  Per-Operation Failures:\n"
            for (operation, count) in operationFailureCounts.sorted(by: { $0.key < $1.key }) {
                let level = operationDegradationLevels[operation] ?? .none
                result += "    \(operation): \(count) failures (\(level))\n"
            }
        }

        return result
    }
}

// MARK: - Configuration

/// Configuration for the GPU health monitor.
///
/// These settings control how the monitor tracks failures and determines
/// when to disable GPU operations and attempt recovery.
public struct GPUHealthMonitorConfiguration: Sendable {
    /// Maximum consecutive failures before disabling an operation
    public let maxFailuresBeforeDisable: Int

    /// Duration in seconds to keep an operation disabled before attempting recovery
    public let disableDurationSeconds: TimeInterval

    /// Interval in seconds between recovery check attempts
    public let recoveryCheckInterval: TimeInterval

    /// Number of consecutive successes required to reduce degradation level
    public let successesForRecovery: Int

    /// Failure count thresholds for each degradation level
    public let degradationThresholds: DegradationThresholds

    /// Thresholds for degradation level transitions
    public struct DegradationThresholds: Sendable {
        /// Failures to reach minor degradation
        public let minor: Int

        /// Failures to reach moderate degradation
        public let moderate: Int

        /// Failures to reach severe degradation
        public let severe: Int

        public init(minor: Int = 1, moderate: Int = 3, severe: Int = 5) {
            self.minor = minor
            self.moderate = moderate
            self.severe = severe
        }

        public static let `default` = DegradationThresholds()
    }

    public init(
        maxFailuresBeforeDisable: Int = 3,
        disableDurationSeconds: TimeInterval = 300, // 5 minutes
        recoveryCheckInterval: TimeInterval = 60,   // 1 minute
        successesForRecovery: Int = 3,
        degradationThresholds: DegradationThresholds = .default
    ) {
        self.maxFailuresBeforeDisable = maxFailuresBeforeDisable
        self.disableDurationSeconds = disableDurationSeconds
        self.recoveryCheckInterval = recoveryCheckInterval
        self.successesForRecovery = successesForRecovery
        self.degradationThresholds = degradationThresholds
    }

    /// Default configuration suitable for most workloads
    public static let `default` = GPUHealthMonitorConfiguration()

    /// Aggressive configuration that disables GPU quickly on failures
    public static let aggressive = GPUHealthMonitorConfiguration(
        maxFailuresBeforeDisable: 2,
        disableDurationSeconds: 600, // 10 minutes
        recoveryCheckInterval: 120,  // 2 minutes
        successesForRecovery: 5,
        degradationThresholds: DegradationThresholds(minor: 1, moderate: 2, severe: 3)
    )

    /// Lenient configuration that tolerates more failures before disabling
    public static let lenient = GPUHealthMonitorConfiguration(
        maxFailuresBeforeDisable: 5,
        disableDurationSeconds: 180, // 3 minutes
        recoveryCheckInterval: 30,   // 30 seconds
        successesForRecovery: 2,
        degradationThresholds: DegradationThresholds(minor: 2, moderate: 4, severe: 6)
    )
}

// MARK: - GPU Health Monitor

/// Actor responsible for tracking GPU health and managing automatic CPU fallbacks.
///
/// The health monitor tracks consecutive failures for each operation type and
/// automatically disables GPU acceleration when too many failures occur. After
/// a configurable timeout, it will attempt to recover by re-enabling GPU operations.
///
/// ## Usage
/// ```swift
/// let monitor = GPUHealthMonitor()
///
/// // Before executing a GPU operation
/// if await monitor.shouldFallbackToCPU(operation: "ivf_search") {
///     // Use CPU implementation
/// } else {
///     do {
///         // Execute GPU operation
///         await monitor.recordSuccess(operation: "ivf_search")
///     } catch {
///         await monitor.recordFailure(operation: "ivf_search", error: error)
///     }
/// }
///
/// // Check overall health
/// let status = await monitor.getHealthStatus()
/// print(status.summary)
/// ```
public actor GPUHealthMonitor {
    // MARK: - Internal State

    /// Configuration settings
    private let configuration: GPUHealthMonitorConfiguration

    /// Consecutive failure count per operation
    private var consecutiveFailures: [String: Int] = [:]

    /// Consecutive success count per operation (for recovery)
    private var consecutiveSuccesses: [String: Int] = [:]

    /// Timestamp when each operation was disabled
    private var disabledOperations: [String: Date] = [:]

    /// Total failure count across all operations
    private var totalFailureCount: Int = 0

    /// Timestamp of the most recent failure
    private var lastFailure: Date?

    /// Number of recovery attempts made
    private var recoveryAttempts: Int = 0

    /// Recent error history for debugging (last 10 errors)
    private var recentErrors: [(date: Date, operation: String, error: String)] = []

    /// Maximum recent errors to keep
    private let maxRecentErrors = 10

    // MARK: - Initialization

    /// Create a health monitor with default configuration
    public init() {
        self.configuration = .default
    }

    /// Create a health monitor with custom configuration
    ///
    /// - Parameter configuration: Custom configuration settings
    public init(configuration: GPUHealthMonitorConfiguration) {
        self.configuration = configuration
    }

    // MARK: - Success/Failure Recording

    /// Records a successful GPU operation.
    ///
    /// Success recording reduces the consecutive failure count for the operation
    /// and may trigger recovery from degraded states.
    ///
    /// - Parameter operation: The operation identifier (e.g., "ivf_search", "l2_distance")
    public func recordSuccess(operation: String) async {
        // Increment consecutive successes
        consecutiveSuccesses[operation, default: 0] += 1

        // Check if we should recover
        if consecutiveSuccesses[operation, default: 0] >= configuration.successesForRecovery {
            // Reduce consecutive failures (graduated recovery)
            if let failures = consecutiveFailures[operation], failures > 0 {
                consecutiveFailures[operation] = max(0, failures - 1)
            }

            // Reset success counter
            consecutiveSuccesses[operation] = 0

            // Check if operation should be re-enabled
            if disabledOperations[operation] != nil {
                recoveryAttempts += 1
                disabledOperations.removeValue(forKey: operation)
            }
        }
    }

    /// Records a failed GPU operation.
    ///
    /// Failure recording increments the consecutive failure count and may
    /// trigger operation disabling if the threshold is reached.
    ///
    /// - Parameters:
    ///   - operation: The operation identifier
    ///   - error: The error that occurred
    public func recordFailure(operation: String, error: any Error) async {
        // Reset consecutive successes
        consecutiveSuccesses[operation] = 0

        // Increment failure counts
        consecutiveFailures[operation, default: 0] += 1
        totalFailureCount += 1
        lastFailure = Date()

        // Record in error history
        let errorDescription = String(describing: error)
        recentErrors.append((date: Date(), operation: operation, error: errorDescription))
        if recentErrors.count > maxRecentErrors {
            recentErrors.removeFirst()
        }

        // Check if operation should be disabled
        let failures = consecutiveFailures[operation, default: 0]
        if failures >= configuration.maxFailuresBeforeDisable {
            disabledOperations[operation] = Date()
        }
    }

    // MARK: - Fallback Decision

    /// Determines whether CPU fallback should be used for an operation.
    ///
    /// Returns `true` if:
    /// - The operation is currently disabled due to repeated failures
    /// - The disable timeout has not yet expired
    /// - The operation has severe degradation
    ///
    /// After the disable duration expires, this method will return `false`
    /// to allow a recovery attempt.
    ///
    /// - Parameter operation: The operation identifier
    /// - Returns: `true` if CPU fallback should be used
    public func shouldFallbackToCPU(operation: String) async -> Bool {
        // Check if operation is disabled
        if let disabledTime = disabledOperations[operation] {
            let elapsed = Date().timeIntervalSince(disabledTime)

            // Check if disable duration has expired
            if elapsed >= configuration.disableDurationSeconds {
                // Allow recovery attempt
                disabledOperations.removeValue(forKey: operation)
                consecutiveFailures[operation] = max(0, (consecutiveFailures[operation] ?? 0) - 1)
                return false
            }

            return true
        }

        // Check degradation level
        let degradation = getDegradationLevel(for: operation)
        return degradation.shouldFallback
    }

    /// Checks whether any fallback is recommended (not forced).
    ///
    /// This method returns `true` if the operation has moderate or higher
    /// degradation, indicating potential issues even if not yet disabled.
    ///
    /// - Parameter operation: The operation identifier
    /// - Returns: `true` if fallback is recommended
    public func isFallbackRecommended(operation: String) async -> Bool {
        let degradation = getDegradationLevel(for: operation)
        return degradation >= .moderate
    }

    // MARK: - Status Queries

    /// Gets the current health status snapshot.
    ///
    /// - Returns: A complete health status including all tracked metrics
    public func getHealthStatus() async -> GPUHealthStatus {
        var operationDegradationLevels: [String: GPUDegradationLevel] = [:]
        for operation in consecutiveFailures.keys {
            operationDegradationLevels[operation] = getDegradationLevel(for: operation)
        }

        let overallDegradation = getOverallDegradationLevel()

        return GPUHealthStatus(
            totalFailureCount: totalFailureCount,
            lastFailure: lastFailure,
            recoveryAttempts: recoveryAttempts,
            isHealthy: overallDegradation == .none,
            degradationLevel: overallDegradation,
            disabledOperations: Set(disabledOperations.keys),
            operationFailureCounts: consecutiveFailures,
            operationDegradationLevels: operationDegradationLevels
        )
    }

    /// Gets the degradation level for a specific operation.
    ///
    /// - Parameter operation: The operation identifier
    /// - Returns: The current degradation level for the operation
    public func getDegradationLevel(for operation: String) -> GPUDegradationLevel {
        let failures = consecutiveFailures[operation] ?? 0
        return degradationLevel(for: failures)
    }

    /// Gets the overall degradation level across all operations.
    ///
    /// Returns the maximum degradation level among all tracked operations.
    ///
    /// - Returns: The overall degradation level
    public func getOverallDegradationLevel() -> GPUDegradationLevel {
        var maxLevel = GPUDegradationLevel.none
        for (_, failures) in consecutiveFailures {
            let level = degradationLevel(for: failures)
            if level > maxLevel {
                maxLevel = level
            }
        }
        return maxLevel
    }

    /// Checks if the GPU is considered healthy overall.
    ///
    /// - Returns: `true` if no operations are degraded
    public func isHealthy() async -> Bool {
        getOverallDegradationLevel() == .none
    }

    /// Gets the number of currently disabled operations.
    ///
    /// - Returns: Count of disabled operations
    public func getDisabledOperationCount() async -> Int {
        disabledOperations.count
    }

    /// Gets recent errors for debugging.
    ///
    /// - Returns: Array of recent errors with timestamps
    public func getRecentErrors() async -> [(date: Date, operation: String, error: String)] {
        recentErrors
    }

    // MARK: - Management

    /// Resets all health tracking state.
    ///
    /// This clears all failure counts, disabled operations, and error history.
    /// Use with caution as it will re-enable all previously disabled operations.
    public func reset() async {
        consecutiveFailures.removeAll()
        consecutiveSuccesses.removeAll()
        disabledOperations.removeAll()
        totalFailureCount = 0
        lastFailure = nil
        recoveryAttempts = 0
        recentErrors.removeAll()
    }

    /// Resets health tracking for a specific operation.
    ///
    /// - Parameter operation: The operation identifier to reset
    public func resetOperation(_ operation: String) async {
        consecutiveFailures.removeValue(forKey: operation)
        consecutiveSuccesses.removeValue(forKey: operation)
        disabledOperations.removeValue(forKey: operation)
    }

    /// Manually disables an operation.
    ///
    /// - Parameter operation: The operation identifier to disable
    public func disableOperation(_ operation: String) async {
        disabledOperations[operation] = Date()
    }

    /// Manually enables an operation.
    ///
    /// - Parameter operation: The operation identifier to enable
    public func enableOperation(_ operation: String) async {
        disabledOperations.removeValue(forKey: operation)
        consecutiveFailures.removeValue(forKey: operation)
    }

    /// Gets the current configuration.
    ///
    /// - Returns: The configuration used by this monitor
    public func getConfiguration() -> GPUHealthMonitorConfiguration {
        configuration
    }

    // MARK: - Private Helpers

    /// Calculates degradation level from failure count.
    private func degradationLevel(for failures: Int) -> GPUDegradationLevel {
        if failures >= configuration.degradationThresholds.severe {
            return .severe
        } else if failures >= configuration.degradationThresholds.moderate {
            return .moderate
        } else if failures >= configuration.degradationThresholds.minor {
            return .minor
        } else {
            return .none
        }
    }
}
