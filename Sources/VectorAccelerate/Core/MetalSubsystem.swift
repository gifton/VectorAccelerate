//
//  MetalSubsystem.swift
//  VectorAccelerate
//
//  Central coordinator for Metal 4 lifecycle management
//

import Foundation
@preconcurrency import Metal

/// Central coordinator for Metal 4 lifecycle management.
///
/// `MetalSubsystem` ensures Metal never blocks the UI by:
/// - Providing a synchronous, instant `init()` that does no Metal work
/// - Deferring all GPU initialization to background queues
/// - Exposing a state machine with observable transitions
/// - Providing CPU fallback when Metal is unavailable
///
/// ## Lifecycle Phases
///
/// **Phase A (App Launch):** No Metal work. UI is immediately responsive.
/// `init()` completes instantly with state `.dormant`.
///
/// **Phase B (Post-First-Frame):** Background Metal initialization via
/// `beginBackgroundInitialization()`:
/// 1. Create MTLDevice
/// 2. Load precompiled metallib (or compile if development mode)
/// 3. Create Metal4Context
/// 4. Warm critical pipelines
/// 5. Transition to `.criticalReady`
///
/// **Phase C (Idle):** Opportunistic warmup of occasional pipelines during
/// idle periods. Pauses automatically during user interaction.
///
/// ## Thread Safety
///
/// `MetalSubsystem` is an actor, providing full isolation. The `isMetalAvailable`
/// and `areCriticalPipelinesReady` properties are nonisolated and thread-safe,
/// using atomic state for lock-free reads.
///
/// ## Example Usage
///
/// ```swift
/// // In AppDelegate
/// let metalSubsystem = MetalSubsystem(configuration: .production(...))
///
/// func applicationDidFinishLaunching() {
///     // Schedule Phase B after first frame
///     DispatchQueue.main.async {
///         self.metalSubsystem.beginBackgroundInitialization()
///     }
/// }
///
/// // In SaveManager
/// func saveEntry(_ entry: Entry) async throws {
///     // Ensure critical pipelines before first save
///     await metalSubsystem.requestCriticalPipelines()
///
///     if let context = await metalSubsystem.context {
///         // GPU path
///     } else {
///         // CPU fallback via metalSubsystem.fallback
///     }
/// }
/// ```
public actor MetalSubsystem {

    // MARK: - State Machine

    /// Current readiness state of the Metal subsystem.
    public private(set) var state: MetalReadinessState = .dormant {
        didSet {
            atomicState.update(from: state)
            notifyObservers(state)
        }
    }

    // MARK: - Configuration

    /// Active configuration
    public let configuration: MetalSubsystemConfiguration

    // MARK: - Internal State

    /// The Metal4Context, nil until initialization completes
    private var _context: Metal4Context?

    /// Fallback provider for CPU operations
    private let fallbackProvider: FallbackProvider

    /// Warmup manager for activity-aware pipeline warming
    private var warmupManager: WarmupManager?

    /// Binary archive manager for persistent pipeline caching
    private var archiveManager: BinaryArchiveManager?

    /// Archive-aware pipeline cache
    private var archivePipelineCache: ArchivePipelineCache?

    // MARK: - Observers

    /// Registered state observers
    private var observers: [UUID: @Sendable (MetalReadinessState) -> Void] = [:]

    // MARK: - Synchronous State (nonisolated access)

    /// Atomic state for nonisolated access
    private let atomicState = AtomicMetalState()

    // MARK: - Initialization

    /// Create a MetalSubsystem with configuration.
    ///
    /// This initializer is synchronous and instant - no Metal work is performed.
    /// Call `beginBackgroundInitialization()` to start Phase B.
    ///
    /// - Parameter configuration: Subsystem configuration
    public init(configuration: MetalSubsystemConfiguration = .default) {
        self.configuration = configuration
        self.fallbackProvider = FallbackProvider()
    }

    /// Convenience initializer with default configuration.
    public init() {
        self.init(configuration: .default)
    }

    // MARK: - Lifecycle Control

    /// Begin background Metal initialization (Phase B).
    ///
    /// This method dispatches all Metal work to background queues and returns
    /// immediately. State transitions are observable via `addReadinessObserver`.
    ///
    /// Safe to call multiple times - subsequent calls are no-ops if already initializing.
    ///
    /// ## State Transitions
    /// `.dormant` → `.initializing` → `.deviceReady` → `.criticalReady` → `.fullyReady`
    ///
    /// If initialization fails, state transitions to `.failed(error)`.
    public func beginBackgroundInitialization() {
        guard case .dormant = state else { return }

        state = .initializing

        Task.detached(priority: configuration.backgroundQoS.taskPriority) { [weak self] in
            await self?.performInitialization()
        }
    }

    /// Request that critical pipelines be ready before continuing.
    ///
    /// If already in `.criticalReady` or `.fullyReady`, returns immediately.
    /// If in `.initializing` or `.deviceReady`, waits for critical warmup.
    /// If in `.dormant`, triggers initialization and waits.
    /// If in `.failed`, returns immediately (caller should check state).
    ///
    /// ## Usage
    ///
    /// Call this before any operation that requires GPU pipelines:
    /// ```swift
    /// await metalSubsystem.requestCriticalPipelines()
    /// // Now safe to use critical pipelines
    /// ```
    public func requestCriticalPipelines() async {
        switch state {
        case .criticalReady, .fullyReady:
            return  // Already ready

        case .failed:
            return  // Caller should check state and use fallback

        case .dormant:
            beginBackgroundInitialization()
            await waitForState(atLeast: .criticalReady)

        case .initializing, .deviceReady:
            await waitForState(atLeast: .criticalReady)
        }
    }

    /// Wait for Metal to be available (at least device ready).
    ///
    /// Unlike `requestCriticalPipelines()`, this only waits for the device
    /// to be ready, not for pipeline warmup. Useful when you need to check
    /// Metal availability without requiring specific pipelines.
    public func waitForDeviceReady() async {
        switch state {
        case .deviceReady, .criticalReady, .fullyReady:
            return

        case .failed:
            return

        case .dormant:
            beginBackgroundInitialization()
            await waitForState(atLeast: .deviceReady)

        case .initializing:
            await waitForState(atLeast: .deviceReady)
        }
    }

    // MARK: - Context Access

    /// The Metal4Context, if available.
    ///
    /// Returns `nil` if:
    /// - Initialization hasn't started (`.dormant`)
    /// - Initialization is in progress (`.initializing`)
    /// - Initialization failed (`.failed`)
    ///
    /// For guaranteed access, use `requestCriticalPipelines()` first.
    public var context: Metal4Context? {
        _context
    }

    /// CPU fallback provider for when Metal is unavailable.
    ///
    /// Always available, regardless of Metal state.
    public var fallback: FallbackProvider {
        fallbackProvider
    }

    // MARK: - Warmup Control

    /// Report user activity to pause warmup.
    ///
    /// Call this when the user interacts with the app (touch, keyboard, scroll).
    /// Warmup will pause and resume after `warmupIdleTimeout` seconds of no activity.
    ///
    /// Safe to call even if warmup hasn't started or has completed.
    public func reportUserActivity() async {
        await warmupManager?.reportUserActivity()
    }

    /// Pause warmup manually.
    ///
    /// Warmup will remain paused until `resumeWarmup()` is called.
    /// Has no effect if warmup isn't currently running.
    public func pauseWarmup() async {
        await warmupManager?.pause(reason: .manualPause)
    }

    /// Resume manually paused warmup.
    ///
    /// Resumes warmup that was paused via `pauseWarmup()`.
    /// Has no effect if warmup isn't paused or was paused for other reasons.
    public func resumeWarmup() async {
        await warmupManager?.resume()
    }

    /// Current warmup progress (0.0 to 1.0).
    ///
    /// Returns 1.0 if warmup hasn't started, is completed, or there are no
    /// occasional pipelines to warm.
    public var warmupProgress: Float {
        get async {
            await warmupManager?.progress ?? 1.0
        }
    }

    /// Whether warmup is currently active.
    ///
    /// Returns `true` if warmup is running or paused (but not completed/cancelled).
    public var isWarmupActive: Bool {
        get async {
            guard let manager = warmupManager else { return false }
            return await manager.state.isActive
        }
    }

    // MARK: - Binary Archive

    /// Save binary archive to disk.
    ///
    /// This method persists all compiled pipelines to the binary archive,
    /// enabling near-instant PSO creation on subsequent launches.
    ///
    /// - Throws: `BinaryArchiveError` if save fails
    public func saveArchive() async throws {
        try await archivePipelineCache?.saveToArchive()
        try await archiveManager?.save()
    }

    /// Archive pipeline cache statistics.
    ///
    /// Returns statistics about archive cache performance, including
    /// memory hits, archive hits, and JIT compilations.
    public var archiveStatistics: ArchivePipelineCacheStatistics? {
        get async {
            await archivePipelineCache?.getStatistics()
        }
    }

    /// Current binary archive state.
    ///
    /// Returns the state of the underlying binary archive manager.
    public var archiveState: BinaryArchiveState? {
        get async {
            await archiveManager?.state
        }
    }

    /// Number of pipelines stored in the binary archive.
    public var archivedPipelineCount: Int {
        get async {
            await archiveManager?.pipelineCount ?? 0
        }
    }

    // MARK: - Synchronous Checks (nonisolated)

    /// Whether Metal is available (device ready, context created).
    ///
    /// Thread-safe, non-blocking check. Does not wait for pipelines.
    /// Use this for quick availability checks in non-async code.
    ///
    /// ```swift
    /// if metalSubsystem.isMetalAvailable {
    ///     // Safe to schedule GPU work
    /// }
    /// ```
    public nonisolated var isMetalAvailable: Bool {
        atomicState.isDeviceReady
    }

    /// Whether critical pipelines are warmed and ready.
    ///
    /// Thread-safe, non-blocking check.
    /// Use this to check if primary operations can use GPU.
    ///
    /// ```swift
    /// if metalSubsystem.areCriticalPipelinesReady {
    ///     // Critical pipelines available for immediate use
    /// }
    /// ```
    public nonisolated var areCriticalPipelinesReady: Bool {
        atomicState.isCriticalReady
    }

    // MARK: - Observer Pattern

    /// Add an observer for state changes.
    ///
    /// The observer is called immediately with the current state, then
    /// on each subsequent state change. Callbacks occur on an unspecified queue.
    ///
    /// - Parameter observer: Closure called on state changes
    /// - Returns: Observer ID for removal
    ///
    /// ```swift
    /// let id = await metalSubsystem.addReadinessObserver { state in
    ///     switch state {
    ///     case .criticalReady:
    ///         print("GPU ready for use")
    ///     case .failed(let error):
    ///         print("GPU unavailable: \(error)")
    ///     default:
    ///         break
    ///     }
    /// }
    /// ```
    @discardableResult
    public func addReadinessObserver(
        _ observer: @escaping @Sendable (MetalReadinessState) -> Void
    ) -> UUID {
        let id = UUID()
        observers[id] = observer

        // Immediately notify of current state
        observer(state)

        return id
    }

    /// Remove a state observer.
    ///
    /// - Parameter id: Observer ID returned from `addReadinessObserver`
    public func removeReadinessObserver(_ id: UUID) {
        observers.removeValue(forKey: id)
    }

    // MARK: - Private Implementation

    private func performInitialization() async {
        do {
            // Step 1: Create Metal4Context
            let context = try await createContext()
            self._context = context

            // Transition to deviceReady
            state = .deviceReady

            // Step 2: Warm critical pipelines
            try await warmCriticalPipelines(context: context)

            // Transition to criticalReady
            state = .criticalReady

            // Step 3: Begin opportunistic warmup in background
            Task.detached(priority: .utility) { [weak self] in
                await self?.warmOccasionalPipelines()
            }

        } catch {
            state = .failed(error)
        }
    }

    private func createContext() async throws -> Metal4Context {
        // Validate production configuration
        if !configuration.allowRuntimeCompilation {
            guard configuration.metallibURL != nil else {
                throw MetalSubsystemError.metallibRequired
            }
        }

        // Create context with default configuration
        let context = try await Metal4Context(configuration: .default)

        // Setup binary archive if URL configured
        if let archiveURL = configuration.binaryArchiveURL {
            let manager = BinaryArchiveManager(
                device: context.device.rawDevice,
                archiveURL: archiveURL
            )

            do {
                try await manager.loadOrCreate()
                self.archiveManager = manager

                // Create archive-aware pipeline cache
                let archiveCache = ArchivePipelineCache(
                    compiler: context.shaderCompiler,
                    archiveManager: manager
                )
                self.archivePipelineCache = archiveCache

                // Inject into context for pipeline lookups
                await context.setArchivePipelineCache(archiveCache)
            } catch {
                // Archive setup failed - continue without archive
                // Log error in production
            }
        }

        return context
    }

    private func warmCriticalPipelines(context: Metal4Context) async throws {
        let criticalKeys = configuration.pipelineRegistry.criticalKeys

        guard !criticalKeys.isEmpty else { return }

        // Warm all critical pipelines using the context's pipeline cache
        await context.pipelineCache.warmUp(keys: criticalKeys)
    }

    private func warmOccasionalPipelines() async {
        guard let context = _context else { return }

        let occasionalKeys = configuration.pipelineRegistry.occasionalKeys

        guard !occasionalKeys.isEmpty else {
            // Save archive even if no occasional pipelines (critical pipelines may be new)
            try? await saveArchive()
            transitionToFullyReady()
            return
        }

        // Create warmup manager for activity-aware warming
        let manager = WarmupManager(context: context, configuration: configuration)
        self.warmupManager = manager

        // Begin interruptible warmup
        await manager.beginWarmup(keys: occasionalKeys)

        // Check if warmup completed successfully
        let finalState = await manager.state
        if case .completed = finalState {
            // Save archive after warmup completion
            try? await saveArchive()
            transitionToFullyReady()
        }
    }

    private func transitionToFullyReady() {
        guard case .criticalReady = state else { return }
        state = .fullyReady
    }

    private func notifyObservers(_ state: MetalReadinessState) {
        for observer in observers.values {
            observer(state)
        }
    }

    private func waitForState(atLeast target: MetalReadinessState) async {
        while !state.isAtLeast(target) && !state.isFailed {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }
}

// MARK: - Readiness State

/// State machine for Metal subsystem readiness.
///
/// States progress linearly from `.dormant` through `.fullyReady`,
/// or branch to `.failed` if an error occurs during any phase.
public enum MetalReadinessState: Sendable, CustomStringConvertible {
    /// No Metal work has started.
    /// This is the initial state after `init()`.
    case dormant

    /// Background initialization in progress.
    /// MTLDevice is being created, metallib being loaded.
    case initializing

    /// MTLDevice available, Metal4Context created.
    /// No pipelines have been warmed yet.
    case deviceReady

    /// Critical pipelines warmed and ready.
    /// Primary app operations can now use GPU.
    case criticalReady

    /// All pipelines (including occasional) warmed.
    /// Full GPU capability available.
    case fullyReady

    /// Initialization failed with error.
    /// GPU unavailable, use CPU fallback.
    case failed(any Error)

    public var description: String {
        switch self {
        case .dormant: return "dormant"
        case .initializing: return "initializing"
        case .deviceReady: return "deviceReady"
        case .criticalReady: return "criticalReady"
        case .fullyReady: return "fullyReady"
        case .failed(let error): return "failed(\(error.localizedDescription))"
        }
    }

    /// Whether this state represents at least the given target.
    ///
    /// Used for waitForState comparisons. States are ordered:
    /// dormant < initializing < deviceReady < criticalReady < fullyReady
    func isAtLeast(_ target: MetalReadinessState) -> Bool {
        switch (self, target) {
        case (.fullyReady, _): return true
        case (.criticalReady, .criticalReady), (.criticalReady, .deviceReady),
             (.criticalReady, .initializing), (.criticalReady, .dormant): return true
        case (.deviceReady, .deviceReady), (.deviceReady, .initializing),
             (.deviceReady, .dormant): return true
        case (.initializing, .initializing), (.initializing, .dormant): return true
        case (.dormant, .dormant): return true
        default: return false
        }
    }

    /// Whether this state represents a failure.
    var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }

    /// The error if this state is `.failed`, otherwise `nil`.
    public var error: (any Error)? {
        if case .failed(let error) = self { return error }
        return nil
    }
}

// MARK: - Errors

/// Errors from MetalSubsystem operations.
public enum MetalSubsystemError: Error, LocalizedError, Sendable {
    /// MetalSubsystem has not been initialized
    case notInitialized

    /// Precompiled metallib required but not provided
    case metallibRequired

    /// Binary archive operation failed
    case archiveNotAvailable

    /// Context creation failed
    case contextCreationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "MetalSubsystem not initialized"
        case .metallibRequired:
            return "Precompiled metallib required in production mode"
        case .archiveNotAvailable:
            return "Binary archive not available"
        case .contextCreationFailed(let reason):
            return "Failed to create Metal context: \(reason)"
        }
    }
}

// MARK: - Atomic State Helper

/// Thread-safe state for nonisolated access.
///
/// This class provides lock-protected atomic access to readiness booleans,
/// allowing nonisolated properties on MetalSubsystem to be read safely
/// from any thread without requiring actor isolation.
private final class AtomicMetalState: @unchecked Sendable {
    private let lock = NSLock()
    private var _isDeviceReady = false
    private var _isCriticalReady = false
    private var _isFailed = false

    var isDeviceReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isDeviceReady
    }

    var isCriticalReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isCriticalReady
    }

    var isFailed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isFailed
    }

    func update(from state: MetalReadinessState) {
        lock.lock()
        defer { lock.unlock() }

        switch state {
        case .dormant:
            _isDeviceReady = false
            _isCriticalReady = false
            _isFailed = false

        case .initializing:
            _isDeviceReady = false
            _isCriticalReady = false
            _isFailed = false

        case .deviceReady:
            _isDeviceReady = true
            _isCriticalReady = false
            _isFailed = false

        case .criticalReady:
            _isDeviceReady = true
            _isCriticalReady = true
            _isFailed = false

        case .fullyReady:
            _isDeviceReady = true
            _isCriticalReady = true
            _isFailed = false

        case .failed:
            _isFailed = true
        }
    }
}
