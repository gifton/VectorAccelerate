//
//  SharedMetalConfiguration.swift
//  VectorAccelerate
//
//  Cross-package Metal context sharing for unified GPU resource management
//

import Foundation

/// Configuration for sharing Metal resources across packages.
///
/// `SharedMetalConfiguration` enables a single `MetalSubsystem` instance
/// to be shared between VectorAccelerate and dependent packages like EmbedKit.
/// This avoids duplicate Metal initialization, enables shared pipeline caching,
/// and reduces memory overhead.
///
/// ## Cross-Package Sharing Pattern
///
/// 1. App creates `MetalSubsystem` with configuration
/// 2. App registers it with `SharedMetalConfiguration`
/// 3. EmbedKit retrieves shared context via `SharedMetalConfiguration`
/// 4. All packages share device, queue, and pipeline cache
///
/// ## Benefits
///
/// - **Single Metal Device:** One MTLDevice, one command queue
/// - **Unified Pipeline Cache:** Pipelines compiled by either package are shared
/// - **Single Warmup:** Critical pipelines warmed once, used everywhere
/// - **Coordinated Lifecycle:** App controls when Metal initializes
/// - **Memory Efficiency:** Single buffer pool, single residency manager
/// - **Consistent Fallback:** Same fallback behavior across packages
///
/// ## Example Usage
///
/// ```swift
/// // In App (owns the lifecycle)
/// let metalSubsystem = MetalSubsystem(configuration: .journalingApp(...))
/// await SharedMetalConfiguration.register(metalSubsystem)
/// await metalSubsystem.beginBackgroundInitialization()
///
/// // In EmbedKit (consumes shared context)
/// if let context = await SharedMetalConfiguration.sharedContext {
///     let manager = try await AccelerationManager(context: context)
/// }
/// ```
///
/// ## Thread Safety
///
/// `SharedMetalConfiguration` is an actor, providing full isolation.
/// All methods are safe to call from any context.
public actor SharedMetalConfiguration {

    // MARK: - Singleton

    /// The shared instance.
    private static let instance = SharedMetalConfiguration()

    // MARK: - State

    /// Registered MetalSubsystem (weak to avoid retain cycles).
    /// The app owns the subsystem; we just provide access to it.
    private weak var _registeredSubsystem: MetalSubsystem?

    /// Direct context for packages that don't use MetalSubsystem.
    /// Used when `register(_ context:)` is called directly.
    private var _directContext: Metal4Context?

    /// Continuations waiting for context availability.
    private var waitingContinuations: [UUID: CheckedContinuation<Metal4Context?, Never>] = [:]

    /// Whether a registration has occurred (even if subsystem was deallocated).
    private var hasBeenRegistered = false

    // MARK: - Registration

    /// Register a MetalSubsystem for cross-package sharing.
    ///
    /// Call this early in app lifecycle, before EmbedKit initialization.
    /// Only one subsystem can be registered at a time; subsequent calls
    /// replace the previous registration.
    ///
    /// The subsystem is held with a weak reference to avoid retain cycles.
    /// The app is responsible for keeping the subsystem alive.
    ///
    /// - Parameter subsystem: The MetalSubsystem to share
    ///
    /// ## Example
    /// ```swift
    /// let metalSubsystem = MetalSubsystem(configuration: .journalingApp(...))
    /// await SharedMetalConfiguration.register(metalSubsystem)
    /// ```
    public static func register(_ subsystem: MetalSubsystem) async {
        await instance.setSubsystem(subsystem)
    }

    /// Register a Metal4Context directly (without MetalSubsystem).
    ///
    /// Use this for simpler sharing scenarios where full lifecycle
    /// management isn't needed. The context is held with a strong reference.
    ///
    /// - Parameter context: The Metal4Context to share
    ///
    /// ## Example
    /// ```swift
    /// let context = try await Metal4Context()
    /// await SharedMetalConfiguration.register(context)
    /// ```
    public static func register(_ context: Metal4Context) async {
        await instance.setDirectContext(context)
    }

    /// Unregister the current shared configuration.
    ///
    /// Clears the registered subsystem or context. Packages that have
    /// already obtained a context reference will continue to use it.
    ///
    /// This is typically called during app termination or when
    /// releasing GPU resources.
    public static func unregister() async {
        await instance.clear()
    }

    // MARK: - Access

    /// Get the shared Metal4Context, if available.
    ///
    /// Returns `nil` if:
    /// - No subsystem/context has been registered
    /// - Registered subsystem hasn't completed initialization
    /// - Registered subsystem was deallocated
    /// - Initialization failed
    ///
    /// ## Usage
    /// ```swift
    /// if let context = await SharedMetalConfiguration.sharedContext {
    ///     // Use GPU
    /// } else {
    ///     // Use CPU fallback
    /// }
    /// ```
    public static var sharedContext: Metal4Context? {
        get async {
            await instance.getContext()
        }
    }

    /// Get the shared MetalSubsystem, if registered.
    ///
    /// Returns `nil` if no subsystem was registered or if it was deallocated.
    /// Use this when you need access to the full lifecycle management API.
    public static var sharedSubsystem: MetalSubsystem? {
        get async {
            await instance._registeredSubsystem
        }
    }

    /// Whether a shared configuration is currently registered.
    ///
    /// Returns `true` if either a MetalSubsystem or direct Metal4Context
    /// is registered and available.
    public static var isRegistered: Bool {
        get async {
            await instance.checkIsRegistered()
        }
    }

    /// Wait for shared context to become available.
    ///
    /// Blocks until a subsystem is registered and initialized, or until
    /// the timeout expires. Use this when you need to ensure the context
    /// is ready before proceeding.
    ///
    /// - Parameter timeout: Maximum time to wait (default: 30 seconds)
    /// - Returns: Shared context or `nil` if timeout expires or initialization fails
    ///
    /// ## Example
    /// ```swift
    /// if let context = await SharedMetalConfiguration.waitForContext(timeout: .seconds(5)) {
    ///     // Context is ready
    /// } else {
    ///     // Timeout or failure - use fallback
    /// }
    /// ```
    public static func waitForContext(timeout: Duration = .seconds(30)) async -> Metal4Context? {
        await instance.waitForContextInternal(timeout: timeout)
    }

    // MARK: - Factory Methods for EmbedKit

    /// Create a Metal4Context for EmbedKit sharing.
    ///
    /// If a shared subsystem is registered, waits for initialization and
    /// returns its context. Otherwise, creates a new context (legacy behavior).
    ///
    /// This method provides backwards compatibility for EmbedKit's
    /// `Metal4ContextManager` while enabling shared usage when available.
    ///
    /// ## Migration Pattern
    ///
    /// EmbedKit's `Metal4ContextManager.getContext()` should call this method:
    /// ```swift
    /// // In EmbedKit Metal4ContextManager
    /// private func getContext() async throws -> Metal4Context {
    ///     return try await SharedMetalConfiguration.forEmbedKitSharing()
    /// }
    /// ```
    ///
    /// - Returns: Shared context if available, otherwise a new context
    /// - Throws: If context creation fails
    public static func forEmbedKitSharing() async throws -> Metal4Context {
        // Check if registration has occurred
        if await instance.hasBeenRegistered {
            // Wait briefly for subsystem initialization
            if let context = await waitForContext(timeout: .seconds(10)) {
                return context
            }
        }

        // Check for direct context registration
        if let direct = await instance._directContext {
            return direct
        }

        // Check for subsystem context without waiting
        if let context = await instance.getContext() {
            return context
        }

        // Fall back to creating new context (legacy EmbedKit behavior)
        return try await Metal4Context()
    }

    // MARK: - Observer Pattern

    /// Observer callback type for context availability notifications.
    public typealias ContextObserver = @Sendable (Metal4Context?) -> Void

    /// Add an observer for context availability changes.
    ///
    /// The observer is called immediately with the current context (or nil),
    /// and again whenever the context becomes available or unavailable.
    ///
    /// - Parameter observer: Closure called on context changes
    /// - Returns: Observer ID for removal
    @discardableResult
    public static func addContextObserver(
        _ observer: @escaping ContextObserver
    ) async -> UUID {
        await instance.addObserver(observer)
    }

    /// Remove a context observer.
    ///
    /// - Parameter id: Observer ID returned from `addContextObserver`
    public static func removeContextObserver(_ id: UUID) async {
        await instance.removeObserver(id)
    }

    // MARK: - Private State

    /// Registered observers
    private var observers: [UUID: ContextObserver] = [:]

    // MARK: - Private Implementation

    private func setSubsystem(_ subsystem: MetalSubsystem) {
        _registeredSubsystem = subsystem
        _directContext = nil
        hasBeenRegistered = true
        notifyObservers()
        resumeWaitingContinuations()
    }

    private func setDirectContext(_ context: Metal4Context) {
        _directContext = context
        _registeredSubsystem = nil
        hasBeenRegistered = true
        notifyObservers()
        resumeWaitingContinuations()
    }

    private func clear() {
        _registeredSubsystem = nil
        _directContext = nil
        // Note: hasBeenRegistered is NOT reset, as packages may still be waiting
        notifyObservers()
    }

    private func checkIsRegistered() -> Bool {
        _registeredSubsystem != nil || _directContext != nil
    }

    private func getContext() async -> Metal4Context? {
        // Try direct context first
        if let direct = _directContext {
            return direct
        }

        // Try subsystem context
        guard let subsystem = _registeredSubsystem else {
            return nil
        }

        return await subsystem.context
    }

    private func waitForContextInternal(timeout: Duration) async -> Metal4Context? {
        // Fast path: context already available
        if let context = await getContext() {
            return context
        }

        // Zero timeout means immediate check only
        if timeout <= .zero {
            return nil
        }

        let id = UUID()

        // Use withTaskGroup for proper timeout handling
        return await withTaskGroup(of: Metal4Context?.self) { group in
            // Context waiting task
            group.addTask {
                await withCheckedContinuation { continuation in
                    Task {
                        await self.addWaitingContinuation(id: id, continuation: continuation)
                    }
                }
            }

            // Timeout task
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }

            // Return first result (either context or timeout)
            if let result = await group.next() {
                group.cancelAll()

                // Clean up continuation if timeout won
                Task {
                    self.removeWaitingContinuation(id: id)
                }

                return result
            }

            return nil
        }
    }

    private func addWaitingContinuation(
        id: UUID,
        continuation: CheckedContinuation<Metal4Context?, Never>
    ) async {
        // Check again if context is now available
        if let context = await getContext() {
            continuation.resume(returning: context)
            return
        }

        waitingContinuations[id] = continuation
    }

    private func removeWaitingContinuation(id: UUID) {
        if let continuation = waitingContinuations.removeValue(forKey: id) {
            continuation.resume(returning: nil)
        }
    }

    private func resumeWaitingContinuations() {
        Task {
            let context = await getContext()
            for (_, continuation) in waitingContinuations {
                continuation.resume(returning: context)
            }
            waitingContinuations.removeAll()
        }
    }

    private func addObserver(_ observer: @escaping ContextObserver) async -> UUID {
        let id = UUID()
        observers[id] = observer

        // Immediately notify of current state
        let context = await getContext()
        observer(context)

        return id
    }

    private func removeObserver(_ id: UUID) {
        observers.removeValue(forKey: id)
    }

    private func notifyObservers() {
        Task {
            let context = await getContext()
            for observer in observers.values {
                observer(context)
            }
        }
    }

    // MARK: - Testing Support

    /// Reset the shared configuration for testing.
    ///
    /// This method clears all state including the `hasBeenRegistered` flag.
    /// Only use this in test teardown.
    @available(*, deprecated, message: "For testing only")
    public static func resetForTesting() async {
        await instance.resetAllState()
    }

    private func resetAllState() {
        _registeredSubsystem = nil
        _directContext = nil
        hasBeenRegistered = false
        observers.removeAll()

        // Resume any waiting continuations with nil
        for (_, continuation) in waitingContinuations {
            continuation.resume(returning: nil)
        }
        waitingContinuations.removeAll()
    }
}

// MARK: - EmbedKit Migration Guide

/*
 ## EmbedKit Migration Guide

 To enable cross-package GPU sharing, update EmbedKit's `Metal4ContextManager`:

 ### Before (Current EmbedKit Implementation)

 ```swift
 private func getContext() async throws -> Metal4Context {
     if let context = cachedContext {
         return context
     }

     let context = try await Metal4Context()
     cachedContext = context
     return context
 }
 ```

 ### After (Shared Implementation)

 ```swift
 private func getContext() async throws -> Metal4Context {
     if let context = cachedContext {
         return context
     }

     // Use shared context when available, fall back to creating own
     let context = try await SharedMetalConfiguration.forEmbedKitSharing()
     cachedContext = context
     return context
 }
 ```

 ### App Integration

 The app should register its MetalSubsystem before EmbedKit initialization:

 ```swift
 // In AppDelegate or App struct
 @main
 struct JournalApp: App {
     let metalSubsystem = MetalSubsystem(configuration: .journalingApp(
         metallibURL: Bundle.main.url(forResource: "default", withExtension: "metallib")!,
         archiveURL: cacheDirectory.appending(path: "pipelines.metalarchive")
     ))

     init() {
         Task {
             // Register for cross-package sharing
             await SharedMetalConfiguration.register(metalSubsystem)

             // Begin background initialization
             metalSubsystem.beginBackgroundInitialization()
         }
     }
 }
 ```

 ### Benefits

 1. **Shared Pipelines:** Pipelines warmed by VectorAccelerate are available to EmbedKit
 2. **Single Device:** No duplicate MTLDevice creation
 3. **Unified Caching:** Binary archives benefit both packages
 4. **Coordinated Lifecycle:** App controls initialization timing
 5. **Memory Efficiency:** Single buffer pool, single residency manager
 */
