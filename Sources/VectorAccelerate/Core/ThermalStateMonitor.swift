//
//  ThermalStateMonitor.swift
//  VectorAccelerate
//
//  Device thermal state monitoring for activity-aware warmup
//

import Foundation

/// Monitor device thermal state for warmup throttling.
///
/// `ThermalStateMonitor` observes the system's thermal state and provides
/// a centralized point for warmup systems to check whether GPU work should
/// be paused to avoid contributing to thermal pressure.
///
/// ## Throttling Behavior
///
/// Warmup should be paused when `shouldThrottle` returns `true`, which happens
/// when the device reports `.serious` or `.critical` thermal state. This helps:
/// - Prevent further heat generation from GPU compute
/// - Allow the device to cool down
/// - Avoid thermal throttling that could impact user-facing operations
///
/// ## Thread Safety
///
/// This class is thread-safe for concurrent access. State reads are protected
/// by a lock, and observers are notified on an unspecified queue.
///
/// ## Example Usage
///
/// ```swift
/// let monitor = ThermalStateMonitor.shared
///
/// if monitor.shouldThrottle {
///     // Pause warmup work
///     return
/// }
///
/// // Or observe changes:
/// let id = monitor.addObserver { state in
///     if state == .serious || state == .critical {
///         pauseWarmup()
///     }
/// }
/// ```
public final class ThermalStateMonitor: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared thermal state monitor instance.
    ///
    /// Use this singleton for app-wide thermal monitoring.
    public static let shared = ThermalStateMonitor()

    // MARK: - State

    /// Lock for thread-safe access to mutable state
    private let lock = NSLock()

    /// Current thermal state (protected by lock)
    private var _currentState: ProcessInfo.ThermalState

    /// Registered observers (protected by lock)
    private var observers: [UUID: @Sendable (ProcessInfo.ThermalState) -> Void] = [:]

    /// Notification observer token
    private var notificationObserver: (any NSObjectProtocol)?

    // MARK: - Initialization

    /// Create a thermal state monitor.
    ///
    /// Prefer using `ThermalStateMonitor.shared` for singleton access.
    public init() {
        // Read initial state
        self._currentState = ProcessInfo.processInfo.thermalState

        // Subscribe to thermal state changes
        notificationObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: nil  // Delivered on posting thread
        ) { [weak self] _ in
            self?.handleThermalStateChange()
        }
    }

    deinit {
        if let observer = notificationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - State Access

    /// Current thermal state of the device.
    ///
    /// Thread-safe read of the cached thermal state.
    public var currentState: ProcessInfo.ThermalState {
        lock.lock()
        defer { lock.unlock() }
        return _currentState
    }

    /// Whether warmup should be throttled due to thermal state.
    ///
    /// Returns `true` when thermal state is `.serious` or `.critical`,
    /// indicating the device is under thermal pressure and GPU work
    /// should be deferred.
    ///
    /// Thread-safe.
    public var shouldThrottle: Bool {
        let state = currentState
        return state == .serious || state == .critical
    }

    // MARK: - Observer Pattern

    /// Add an observer for thermal state changes.
    ///
    /// The observer is called on an unspecified queue when thermal state changes.
    /// It is NOT called immediately with the current state (unlike MetalSubsystem).
    ///
    /// - Parameter observer: Closure called when thermal state changes
    /// - Returns: Observer ID for removal
    @discardableResult
    public func addObserver(
        _ observer: @escaping @Sendable (ProcessInfo.ThermalState) -> Void
    ) -> UUID {
        lock.lock()
        defer { lock.unlock() }

        let id = UUID()
        observers[id] = observer
        return id
    }

    /// Remove a thermal state observer.
    ///
    /// - Parameter id: Observer ID returned from `addObserver`
    public func removeObserver(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        observers.removeValue(forKey: id)
    }

    // MARK: - Internal

    /// Handle thermal state change notification
    private func handleThermalStateChange() {
        let newState = ProcessInfo.processInfo.thermalState

        lock.lock()
        let previousState = _currentState
        _currentState = newState
        let currentObservers = observers
        lock.unlock()

        // Only notify if state actually changed
        guard newState != previousState else { return }

        // Notify all observers
        for observer in currentObservers.values {
            observer(newState)
        }
    }

    // MARK: - Testing Support

    /// Force a thermal state update (for testing).
    ///
    /// In tests, call this after simulating thermal state changes.
    /// Note: This only refreshes from ProcessInfo; it cannot override the actual state.
    internal func refreshState() {
        handleThermalStateChange()
    }
}

// MARK: - ThermalState Description Extension

extension ProcessInfo.ThermalState: @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .nominal:
            return "nominal"
        case .fair:
            return "fair"
        case .serious:
            return "serious"
        case .critical:
            return "critical"
        @unknown default:
            return "unknown"
        }
    }
}
