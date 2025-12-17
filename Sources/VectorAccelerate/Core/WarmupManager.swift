//
//  WarmupManager.swift
//  VectorAccelerate
//
//  Activity-aware pipeline warmup coordinator
//

import Foundation

// MARK: - Activity Delegate Protocol

/// Protocol for external activity detection injection.
///
/// Apps can implement this protocol to provide custom activity detection
/// from gesture recognizers, scroll events, keyboard input, or other sources.
///
/// ## Example Implementation
///
/// ```swift
/// class AppActivityDelegate: ActivityDelegate {
///     private weak var warmupManager: WarmupManager?
///
///     func setWarmupManager(_ manager: WarmupManager) {
///         warmupManager = manager
///     }
///
///     // Called from UIGestureRecognizer callbacks
///     func handleGesture(_ gesture: UIGestureRecognizer) {
///         if gesture.state == .began || gesture.state == .changed {
///             Task { await warmupManager?.reportUserActivity() }
///         }
///     }
/// }
/// ```
public protocol ActivityDelegate: AnyObject, Sendable {
    /// Called when warmup begins.
    /// Use this to start listening for activity events.
    @MainActor func warmupDidBegin()

    /// Called when warmup pauses.
    @MainActor func warmupDidPause(reason: WarmupManager.PauseReason)

    /// Called when warmup resumes.
    @MainActor func warmupDidResume()

    /// Called when warmup completes or is cancelled.
    @MainActor func warmupDidEnd(completed: Bool)
}

// MARK: - Warmup Statistics

/// Statistics about warmup source distribution.
public struct WarmupStatistics: Sendable {
    /// Total pipelines targeted for warmup
    public let totalPipelines: Int

    /// Pipelines that were already warm (memory cache hit)
    public let memoryHits: Int

    /// Pipelines loaded from binary archive
    public let archiveHits: Int

    /// Pipelines compiled via JIT
    public let jitCompilations: Int

    /// Pipelines that failed to compile
    public let failures: Int

    /// Pipelines remaining to warm
    public var remaining: Int {
        totalPipelines - memoryHits - archiveHits - jitCompilations - failures
    }

    /// Whether warmup is complete
    public var isComplete: Bool {
        remaining == 0
    }

    /// Percentage of pipelines from archive (0.0-1.0)
    public var archiveHitRate: Double {
        let completed = memoryHits + archiveHits + jitCompilations
        guard completed > 0 else { return 0 }
        return Double(archiveHits) / Double(completed)
    }

    /// Percentage of pipelines that needed JIT compilation (0.0-1.0)
    public var jitRate: Double {
        let completed = memoryHits + archiveHits + jitCompilations
        guard completed > 0 else { return 0 }
        return Double(jitCompilations) / Double(completed)
    }

    internal init(
        totalPipelines: Int = 0,
        memoryHits: Int = 0,
        archiveHits: Int = 0,
        jitCompilations: Int = 0,
        failures: Int = 0
    ) {
        self.totalPipelines = totalPipelines
        self.memoryHits = memoryHits
        self.archiveHits = archiveHits
        self.jitCompilations = jitCompilations
        self.failures = failures
    }
}

// MARK: - State Observer

/// Observer for warmup state changes.
public typealias WarmupStateObserver = @Sendable (WarmupManager.WarmupState) -> Void

/// Activity-aware pipeline warmup coordinator.
///
/// `WarmupManager` provides intelligent warmup of GPU pipelines that respects
/// user activity and system state. Key behaviors:
///
/// - **Pauses during user activity:** When `reportUserActivity()` is called,
///   warmup pauses to avoid competing with user operations.
/// - **Resumes after idle timeout:** Warmup resumes after `warmupIdleTimeout`
///   seconds of no user activity.
/// - **Respects thermal state:** When thermal throttling is detected (serious
///   or critical), warmup pauses to avoid contributing to heat.
/// - **Interruptible:** Warmup can be cancelled at any time; work is performed
///   one pipeline at a time for granular cancellation.
/// - **Observable:** State changes can be observed via the observer pattern.
///
/// ## Core Principle
///
/// GPU warmup should never cause UI jank or compete with user operations.
///
/// ## Usage
///
/// ```swift
/// let manager = WarmupManager(context: context, configuration: config)
///
/// // Add state observer
/// let observerId = await manager.addObserver { state in
///     print("Warmup state: \(state)")
/// }
///
/// // Set activity delegate for external activity detection
/// await manager.setActivityDelegate(appActivityDelegate)
///
/// // Start warmup
/// await manager.beginWarmup(keys: registry.occasionalKeys)
///
/// // Report user activity (e.g., from touch/keyboard events)
/// await manager.reportUserActivity()
///
/// // Check progress
/// let progress = await manager.progress  // 0.0 to 1.0
///
/// // Get statistics
/// let stats = await manager.statistics
/// print("Archive hit rate: \(stats.archiveHitRate)")
///
/// // Cancel if needed
/// manager.cancel()
///
/// // Remove observer
/// await manager.removeObserver(observerId)
/// ```
///
/// ## State Machine
///
/// ```
/// idle → warming ⇄ paused(reason) → completed
///         ↓
///       cancelled
/// ```
public actor WarmupManager {

    // MARK: - State Types

    /// Current state of the warmup process.
    public enum WarmupState: Sendable, Equatable {
        /// No warmup in progress
        case idle

        /// Actively warming pipelines
        case warming(progress: Float)

        /// Warmup paused for a specific reason
        case paused(reason: PauseReason)

        /// Warmup completed successfully
        case completed

        /// Warmup was cancelled
        case cancelled

        /// Whether this state represents active work
        public var isActive: Bool {
            switch self {
            case .warming, .paused:
                return true
            case .idle, .completed, .cancelled:
                return false
            }
        }
    }

    /// Reason for warmup pause.
    public enum PauseReason: Sendable, Equatable, CustomStringConvertible {
        /// User is actively interacting with the app
        case userActivity

        /// Device is thermally throttled
        case thermalThrottling

        /// Manually paused via `pause()` call
        case manualPause

        public var description: String {
            switch self {
            case .userActivity:
                return "userActivity"
            case .thermalThrottling:
                return "thermalThrottling"
            case .manualPause:
                return "manualPause"
            }
        }
    }

    // MARK: - Configuration

    /// The Metal context providing pipeline cache access
    private let context: Metal4Context

    /// Configuration controlling warmup behavior
    private let configuration: MetalSubsystemConfiguration

    /// Thermal state monitor for throttling checks
    private let thermalMonitor: ThermalStateMonitor

    // MARK: - State

    /// Current warmup state
    public private(set) var state: WarmupState = .idle {
        didSet {
            atomicState.update(from: state)
            notifyObservers(state)
            notifyDelegateOfStateChange(from: oldValue, to: state)
        }
    }

    /// Atomic state for nonisolated access
    private let atomicState = AtomicWarmupState()

    // MARK: - Observers

    /// Registered state observers
    private var observers: [UUID: WarmupStateObserver] = [:]

    // MARK: - Activity Delegate

    /// External activity delegate for app-specific activity detection
    private weak var activityDelegate: (any ActivityDelegate)?

    // MARK: - Warmup Tracking

    /// Keys to warm up
    private var pendingKeys: [PipelineCacheKey] = []

    /// Number of keys completed
    private var completedCount: Int = 0

    /// Total number of keys to warm
    private var totalCount: Int = 0

    /// Timestamp of last user activity
    private var lastActivityTime: Date?

    /// Active warmup task (for cancellation)
    private var warmupTask: Task<Void, Never>?

    /// Thermal observer ID
    private var thermalObserverId: UUID?

    // MARK: - Statistics Tracking

    /// Count of pipelines loaded from memory cache
    private var memoryHits: Int = 0

    /// Count of pipelines loaded from binary archive
    private var archiveHits: Int = 0

    /// Count of pipelines compiled via JIT
    private var jitCompilations: Int = 0

    /// Count of pipelines that failed to compile
    private var failures: Int = 0

    // MARK: - Initialization

    /// Create a warmup manager.
    ///
    /// - Parameters:
    ///   - context: Metal context with pipeline cache
    ///   - configuration: Subsystem configuration
    ///   - thermalMonitor: Thermal state monitor (defaults to shared instance)
    public init(
        context: Metal4Context,
        configuration: MetalSubsystemConfiguration,
        thermalMonitor: ThermalStateMonitor = .shared
    ) {
        self.context = context
        self.configuration = configuration
        self.thermalMonitor = thermalMonitor
    }

    // MARK: - Lifecycle

    /// Begin warming up a set of pipeline keys.
    ///
    /// Starts an interruptible warmup process that warms pipelines one-by-one.
    /// The warmup automatically pauses when:
    /// - `reportUserActivity()` is called
    /// - Thermal throttling is detected (if `respectThermalState` is enabled)
    /// - `pause()` is called
    ///
    /// - Parameter keys: Pipeline cache keys to warm up
    public func beginWarmup(keys: [PipelineCacheKey]) async {
        // Cancel any existing warmup
        cancel()

        // Reset state
        pendingKeys = keys
        completedCount = 0
        totalCount = keys.count
        lastActivityTime = nil

        // Reset statistics
        memoryHits = 0
        archiveHits = 0
        jitCompilations = 0
        failures = 0

        guard !keys.isEmpty else {
            state = .completed
            return
        }

        // Subscribe to thermal state changes
        setupThermalObserver()

        // Check initial thermal state
        if configuration.respectThermalState && thermalMonitor.shouldThrottle {
            state = .paused(reason: .thermalThrottling)
            return
        }

        state = .warming(progress: 0)

        // Start warmup task
        warmupTask = Task { [weak self] in
            await self?.runWarmupLoop()
        }

        // Wait for completion
        await warmupTask?.value
    }

    /// Pause warmup for a specified reason.
    ///
    /// Warmup can be resumed with `resume()`.
    ///
    /// - Parameter reason: Why warmup is being paused
    public func pause(reason: PauseReason) {
        guard case .warming = state else { return }
        state = .paused(reason: reason)
    }

    /// Resume paused warmup.
    ///
    /// If warmup was paused due to user activity, this method checks
    /// whether the idle timeout has elapsed. If not, warmup remains paused.
    public func resume() {
        guard case .paused(let reason) = state else { return }

        // Check if we should still be paused due to user activity
        if reason == .userActivity {
            if let lastActivity = lastActivityTime {
                let elapsed = Date().timeIntervalSince(lastActivity)
                if elapsed < configuration.warmupIdleTimeout {
                    // Not enough idle time has passed
                    return
                }
            }
        }

        // Check thermal state before resuming
        if configuration.respectThermalState && thermalMonitor.shouldThrottle {
            state = .paused(reason: .thermalThrottling)
            return
        }

        state = .warming(progress: progress)
    }

    /// Cancel warmup.
    ///
    /// Immediately stops warmup and transitions to cancelled state.
    /// The warmup task is cancelled cooperatively.
    public func cancel() {
        warmupTask?.cancel()
        warmupTask = nil
        removeThermalObserver()

        if state.isActive {
            state = .cancelled
        }
    }

    // MARK: - Activity Tracking

    /// Report user activity.
    ///
    /// Call this method when the user interacts with the app (e.g., touch,
    /// keyboard, scroll events). Warmup will pause and resume after
    /// `warmupIdleTimeout` seconds of no activity.
    public func reportUserActivity() {
        lastActivityTime = Date()

        if case .warming = state {
            state = .paused(reason: .userActivity)
            scheduleIdleResume()
        }
    }

    // MARK: - Nonisolated Checks

    /// Whether warmup is actively running.
    ///
    /// Thread-safe, non-blocking check.
    public nonisolated var isWarming: Bool {
        atomicState.isWarming
    }

    /// Whether warmup is currently paused.
    ///
    /// Thread-safe, non-blocking check.
    public nonisolated var isPaused: Bool {
        atomicState.isPaused
    }

    // MARK: - Progress

    /// Current warmup progress (0.0 to 1.0).
    public var progress: Float {
        guard totalCount > 0 else { return 1.0 }
        return Float(completedCount) / Float(totalCount)
    }

    /// Number of pipelines warmed so far.
    public var warmedCount: Int {
        completedCount
    }

    /// Total pipelines to warm.
    public var pipelineCount: Int {
        totalCount
    }

    // MARK: - Statistics

    /// Get current warmup statistics.
    ///
    /// Statistics track the source of each warmed pipeline (memory cache,
    /// binary archive, or JIT compilation) to help measure archive effectiveness.
    ///
    /// ```swift
    /// let stats = await warmupManager.statistics
    /// print("Archive hit rate: \(stats.archiveHitRate * 100)%")
    /// print("JIT compilations: \(stats.jitCompilations)")
    /// ```
    public var statistics: WarmupStatistics {
        WarmupStatistics(
            totalPipelines: totalCount,
            memoryHits: memoryHits,
            archiveHits: archiveHits,
            jitCompilations: jitCompilations,
            failures: failures
        )
    }

    // MARK: - Observer Pattern

    /// Add an observer for state changes.
    ///
    /// The observer is called on each state change. Unlike some observer patterns,
    /// this does NOT call the observer immediately with the current state.
    ///
    /// - Parameter observer: Closure called when state changes
    /// - Returns: Observer ID for removal
    ///
    /// ```swift
    /// let id = await warmupManager.addObserver { state in
    ///     switch state {
    ///     case .warming(let progress):
    ///         print("Progress: \(progress * 100)%")
    ///     case .paused(let reason):
    ///         print("Paused: \(reason)")
    ///     case .completed:
    ///         print("Warmup complete!")
    ///     default:
    ///         break
    ///     }
    /// }
    /// ```
    @discardableResult
    public func addObserver(_ observer: @escaping WarmupStateObserver) -> UUID {
        let id = UUID()
        observers[id] = observer
        return id
    }

    /// Remove a state observer.
    ///
    /// - Parameter id: Observer ID returned from `addObserver`
    public func removeObserver(_ id: UUID) {
        observers.removeValue(forKey: id)
    }

    /// Remove all observers.
    public func removeAllObservers() {
        observers.removeAll()
    }

    // MARK: - Activity Delegate

    /// Set the activity delegate for external activity detection.
    ///
    /// The delegate receives callbacks when warmup state changes, allowing
    /// apps to coordinate activity detection (e.g., gesture recognizers).
    ///
    /// - Parameter delegate: The activity delegate, or nil to remove
    ///
    /// ```swift
    /// await warmupManager.setActivityDelegate(appActivityDelegate)
    /// ```
    public func setActivityDelegate(_ delegate: (any ActivityDelegate)?) {
        activityDelegate = delegate
    }

    // MARK: - Private Implementation

    /// Main warmup loop
    private func runWarmupLoop() async {
        while !pendingKeys.isEmpty {
            // Check for cancellation
            if Task.isCancelled {
                state = .cancelled
                return
            }

            // Check if paused
            guard case .warming = state else {
                // Wait and retry
                try? await Task.sleep(for: .milliseconds(100))
                continue
            }

            // Check thermal state
            if configuration.respectThermalState && thermalMonitor.shouldThrottle {
                state = .paused(reason: .thermalThrottling)
                continue
            }

            // Get next key
            let key = pendingKeys.removeFirst()

            // Track statistics before warmup
            let statsBefore = await context.getArchiveCacheStatistics()

            // Warm this pipeline using context's getPipeline which prefers archive cache
            do {
                _ = try await context.getPipeline(for: key)

                // Track statistics after warmup to determine source
                let statsAfter = await context.getArchiveCacheStatistics()
                trackWarmupSource(before: statsBefore, after: statsAfter)
            } catch {
                // Log but continue - don't fail entire warmup for one pipeline
                failures += 1
            }

            completedCount += 1
            state = .warming(progress: progress)
        }

        // Completed all pipelines
        state = .completed
        removeThermalObserver()
    }

    /// Track the source of a warmed pipeline by comparing cache statistics.
    private func trackWarmupSource(
        before: ArchivePipelineCacheStatistics?,
        after: ArchivePipelineCacheStatistics?
    ) {
        guard let before = before, let after = after else {
            // If we can't get statistics, assume JIT
            jitCompilations += 1
            return
        }

        // Determine source by checking which counter increased
        if after.memoryHits > before.memoryHits {
            memoryHits += 1
        } else if after.archiveHits > before.archiveHits {
            archiveHits += 1
        } else if after.compilations > before.compilations {
            jitCompilations += 1
        } else {
            // Unknown source, assume JIT
            jitCompilations += 1
        }
    }

    /// Notify all registered observers of a state change.
    private func notifyObservers(_ state: WarmupState) {
        for observer in observers.values {
            observer(state)
        }
    }

    /// Notify the activity delegate of a state change.
    private func notifyDelegateOfStateChange(from oldState: WarmupState, to newState: WarmupState) {
        guard let delegate = activityDelegate else { return }

        Task { @MainActor in
            switch (oldState, newState) {
            case (_, .warming) where oldState == .idle || oldState == .cancelled:
                // Warmup began
                delegate.warmupDidBegin()

            case (.paused, .warming):
                // Warmup resumed
                delegate.warmupDidResume()

            case (.warming, .paused(let reason)):
                // Warmup paused
                delegate.warmupDidPause(reason: reason)

            case (_, .completed):
                // Warmup ended successfully
                delegate.warmupDidEnd(completed: true)

            case (_, .cancelled):
                // Warmup was cancelled
                delegate.warmupDidEnd(completed: false)

            default:
                break
            }
        }
    }

    /// Schedule a resume after idle timeout
    private func scheduleIdleResume() {
        // Capture timeout value before spawning task (configuration is Sendable)
        let timeout = configuration.warmupIdleTimeout

        Task { [weak self] in
            // Wait for idle timeout
            try? await Task.sleep(for: .seconds(timeout))

            // Check if still paused due to user activity
            await self?.checkIdleResume()
        }
    }

    /// Check if we should resume after idle timeout
    private func checkIdleResume() {
        guard case .paused(reason: .userActivity) = state else { return }

        if let lastActivity = lastActivityTime {
            let elapsed = Date().timeIntervalSince(lastActivity)
            if elapsed >= configuration.warmupIdleTimeout {
                resume()
            }
        }
    }

    /// Setup thermal state observer
    private func setupThermalObserver() {
        guard configuration.respectThermalState else { return }

        thermalObserverId = thermalMonitor.addObserver { [weak self] state in
            Task { [weak self] in
                await self?.handleThermalStateChange(state)
            }
        }
    }

    /// Handle thermal state change
    private func handleThermalStateChange(_ thermalState: ProcessInfo.ThermalState) {
        let shouldThrottle = thermalState == .serious || thermalState == .critical

        if shouldThrottle {
            if case .warming = state {
                state = .paused(reason: .thermalThrottling)
            }
        } else {
            if case .paused(reason: .thermalThrottling) = state {
                resume()
            }
        }
    }

    /// Remove thermal observer
    private func removeThermalObserver() {
        if let id = thermalObserverId {
            thermalMonitor.removeObserver(id)
            thermalObserverId = nil
        }
    }
}

// MARK: - Atomic State Helper

/// Thread-safe state for nonisolated access.
private final class AtomicWarmupState: @unchecked Sendable {
    private let lock = NSLock()
    private var _isWarming = false
    private var _isPaused = false

    var isWarming: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isWarming
    }

    var isPaused: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isPaused
    }

    func update(from state: WarmupManager.WarmupState) {
        lock.lock()
        defer { lock.unlock() }

        switch state {
        case .warming:
            _isWarming = true
            _isPaused = false
        case .paused:
            _isWarming = false
            _isPaused = true
        case .idle, .completed, .cancelled:
            _isWarming = false
            _isPaused = false
        }
    }
}
