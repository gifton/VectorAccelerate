//
//  SharedMetalContext.swift
//  VectorAccelerate
//
//  Shared Metal context APIs for multi-component resource sharing.
//  Enables efficient GPU resource sharing across embedding pipelines
//  and other multi-stage processing scenarios.
//

import Foundation
@preconcurrency import Metal

// MARK: - Command Queue Priority

/// Priority level for Metal command queue operations.
///
/// Affects how command buffers are scheduled relative to other GPU work.
/// Higher priority operations may be scheduled more aggressively.
///
/// ## Usage
/// ```swift
/// let config = SharedConfiguration(
///     queuePriority: .high  // Time-sensitive embedding operations
/// )
/// ```
///
/// ## Note
/// Metal doesn't expose explicit queue priorities, but this setting
/// influences command buffer scheduling hints and can affect latency
/// in multi-queue scenarios.
public enum CommandQueuePriority: Int, Sendable, Comparable, CaseIterable, Codable {
    /// Low priority - background processing, can be preempted.
    /// Use for batch operations that aren't time-sensitive.
    case low = 0

    /// Normal priority - default for most operations.
    /// Balanced scheduling between throughput and latency.
    case normal = 1

    /// High priority - time-sensitive operations.
    /// Use for real-time inference or interactive workloads.
    case high = 2

    public static func < (lhs: CommandQueuePriority, rhs: CommandQueuePriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Human-readable description
    public var description: String {
        switch self {
        case .low: return "low"
        case .normal: return "normal"
        case .high: return "high"
        }
    }
}

// MARK: - Shared Configuration

/// Configuration for shared Metal context across multiple consumers.
///
/// Use `SharedConfiguration` when multiple components need to efficiently
/// share Metal resources, such as in embedding pipelines where tokenizer,
/// encoder, and pooling stages share GPU buffers and command queues.
///
/// ## Usage
/// ```swift
/// // Create shared configuration for an embedding pipeline
/// let config = SharedConfiguration(
///     baseConfiguration: .default,
///     queuePriority: .high,
///     enableBufferSharing: true,
///     identifier: "embedding-pipeline"
/// )
///
/// // Create or retrieve shared context
/// let context = try await MetalContext.create(sharedConfig: config)
///
/// // Get shared buffer factory for component use
/// let bufferFactory = context.sharedBufferFactory(configuration: config)
/// ```
///
/// ## Thread Safety
/// `SharedConfiguration` is `Sendable` and can be safely shared across
/// actor boundaries and concurrent contexts.
public struct SharedConfiguration: Sendable, Equatable, Codable {

    /// Base Metal configuration settings.
    /// Controls device selection, buffer pool sizing, and profiling.
    public let baseConfiguration: MetalConfiguration

    /// Priority for command queue operations.
    /// Affects scheduling of GPU work relative to other operations.
    public let queuePriority: CommandQueuePriority

    /// Enable buffer sharing across components.
    ///
    /// When enabled, buffers can be efficiently shared between different
    /// pipeline stages without copying. This is optimal for pipelines
    /// where data flows through multiple processing stages.
    public let enableBufferSharing: Bool

    /// Unique identifier for this shared context.
    ///
    /// Components using the same identifier will share the same Metal
    /// context and resources. Use distinct identifiers to isolate
    /// different pipelines or workloads.
    public let identifier: String

    // MARK: - Initialization

    /// Create a shared configuration.
    ///
    /// - Parameters:
    ///   - baseConfiguration: Base Metal configuration (default: `.default`)
    ///   - queuePriority: Command queue priority (default: `.normal`)
    ///   - enableBufferSharing: Enable cross-component buffer sharing (default: `true`)
    ///   - identifier: Unique identifier for context lookup (default: `"default"`)
    public init(
        baseConfiguration: MetalConfiguration = .default,
        queuePriority: CommandQueuePriority = .normal,
        enableBufferSharing: Bool = true,
        identifier: String = "default"
    ) {
        self.baseConfiguration = baseConfiguration
        self.queuePriority = queuePriority
        self.enableBufferSharing = enableBufferSharing
        self.identifier = identifier
    }

    /// Default shared configuration.
    /// Uses default Metal configuration with normal priority and buffer sharing enabled.
    public static let `default` = SharedConfiguration()

    /// High-performance configuration for real-time workloads.
    /// Prioritizes latency over throughput.
    public static let realtime = SharedConfiguration(
        baseConfiguration: MetalConfiguration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 512 * 1024 * 1024,  // 512MB
            maxBuffersPerSize: 8,
            enableProfiling: false
        ),
        queuePriority: .high,
        enableBufferSharing: true,
        identifier: "realtime"
    )

    /// Batch processing configuration.
    /// Optimized for throughput with larger buffer pools.
    public static let batch = SharedConfiguration(
        baseConfiguration: MetalConfiguration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 2 * 1024 * 1024 * 1024,  // 2GB
            maxBuffersPerSize: 20,
            enableProfiling: false
        ),
        queuePriority: .normal,
        enableBufferSharing: true,
        identifier: "batch"
    )
}

// MARK: - MetalConfiguration Codable Extension

extension MetalConfiguration: Codable {
    enum CodingKeys: String, CodingKey {
        case preferHighPerformanceDevice
        case maxBufferPoolMemory
        case maxBuffersPerSize
        case enableProfiling
        case commandQueueLabel
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            preferHighPerformanceDevice: try container.decode(Bool.self, forKey: .preferHighPerformanceDevice),
            maxBufferPoolMemory: try container.decodeIfPresent(Int.self, forKey: .maxBufferPoolMemory),
            maxBuffersPerSize: try container.decode(Int.self, forKey: .maxBuffersPerSize),
            enableProfiling: try container.decode(Bool.self, forKey: .enableProfiling),
            commandQueueLabel: try container.decode(String.self, forKey: .commandQueueLabel)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(preferHighPerformanceDevice, forKey: .preferHighPerformanceDevice)
        try container.encodeIfPresent(maxBufferPoolMemory, forKey: .maxBufferPoolMemory)
        try container.encode(maxBuffersPerSize, forKey: .maxBuffersPerSize)
        try container.encode(enableProfiling, forKey: .enableProfiling)
        try container.encode(commandQueueLabel, forKey: .commandQueueLabel)
    }
}

// MARK: - MetalConfiguration Equatable Extension

extension MetalConfiguration: Equatable {
    public static func == (lhs: MetalConfiguration, rhs: MetalConfiguration) -> Bool {
        lhs.preferHighPerformanceDevice == rhs.preferHighPerformanceDevice &&
        lhs.maxBufferPoolMemory == rhs.maxBufferPoolMemory &&
        lhs.maxBuffersPerSize == rhs.maxBuffersPerSize &&
        lhs.enableProfiling == rhs.enableProfiling &&
        lhs.commandQueueLabel == rhs.commandQueueLabel
    }
}

// MARK: - Shared Buffer Factory

/// Thread-safe buffer factory for shared Metal context scenarios.
///
/// Wraps `MetalBufferFactory` with configuration context for use in
/// shared scenarios where multiple components create buffers from
/// the same underlying device.
///
/// ## Usage
/// ```swift
/// let context = try await MetalContext.create(sharedConfig: config)
/// let factory = context.sharedBufferFactory(configuration: config)
///
/// // Create buffers using the shared factory
/// let buffer = factory.createBuffer(length: 4096)
/// let alignedBuffer = factory.createAlignedBuffer(length: 1024, alignment: 256)
/// ```
///
/// ## Thread Safety
/// `SharedBufferFactory` is thread-safe and can be used from multiple
/// threads concurrently. The underlying `MetalBufferFactory` uses
/// thread-safe MTLDevice buffer creation.
public final class SharedBufferFactory: @unchecked Sendable {

    /// The underlying buffer factory for actual buffer creation.
    public let factory: MetalBufferFactory

    /// The shared configuration this factory was created with.
    public let configuration: SharedConfiguration

    // MARK: - Initialization

    /// Create a shared buffer factory.
    ///
    /// - Parameters:
    ///   - factory: The underlying Metal buffer factory
    ///   - configuration: The shared configuration
    public init(factory: MetalBufferFactory, configuration: SharedConfiguration) {
        self.factory = factory
        self.configuration = configuration
    }

    // MARK: - Device Information

    /// The underlying Metal device.
    public var device: any MTLDevice {
        factory.device
    }

    /// Whether the device has unified memory (Apple Silicon).
    public var hasUnifiedMemory: Bool {
        factory.hasUnifiedMemory
    }

    /// Default resource options for buffer creation.
    public var defaultOptions: MTLResourceOptions {
        factory.defaultOptions
    }

    /// Recommended alignment for optimal performance.
    public var recommendedAlignment: Int {
        factory.recommendedAlignment
    }

    // MARK: - Basic Buffer Creation

    /// Create an empty buffer of the specified length.
    ///
    /// - Parameters:
    ///   - length: Buffer size in bytes
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Created buffer or nil if allocation fails
    public func createBuffer(
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createBuffer(length: length, options: options)
    }

    /// Create a buffer initialized with raw bytes.
    ///
    /// - Parameters:
    ///   - bytes: Pointer to source data
    ///   - length: Size of data in bytes
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Created buffer with data copied, or nil if allocation fails
    public func createBuffer(
        bytes: UnsafeRawPointer,
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createBuffer(bytes: bytes, length: length, options: options)
    }

    /// Create a buffer from an array of elements.
    ///
    /// - Parameters:
    ///   - data: Array of elements to copy into buffer
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Created buffer with data copied, or nil if allocation fails
    public func createBuffer<T>(
        from data: [T],
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createBuffer(from: data, options: options)
    }

    // MARK: - Aligned Buffer Creation

    /// Create a buffer with guaranteed alignment for SIMD operations.
    ///
    /// - Parameters:
    ///   - length: Requested buffer size in bytes
    ///   - alignment: Required alignment in bytes (default: 16 for float4)
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Aligned buffer or nil if allocation fails
    public func createAlignedBuffer(
        length: Int,
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createAlignedBuffer(length: length, alignment: alignment, options: options)
    }

    /// Create an aligned buffer from an array.
    ///
    /// - Parameters:
    ///   - data: Array of elements to copy
    ///   - alignment: Required alignment in bytes (default: 16)
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Aligned buffer with data copied, or nil if allocation fails
    public func createAlignedBuffer<T>(
        from data: [T],
        alignment: Int = 16,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createAlignedBuffer(from: data, alignment: alignment, options: options)
    }

    // MARK: - MetalBuffer Wrapper Creation

    /// Create a MetalBuffer (Sendable wrapper) with specified length.
    ///
    /// - Parameters:
    ///   - length: Buffer size in bytes
    ///   - options: Metal resource options
    /// - Returns: MetalBuffer wrapper or nil if allocation fails
    public func createMetalBuffer(
        length: Int,
        options: MTLResourceOptions? = nil
    ) -> MetalBuffer? {
        factory.createMetalBuffer(length: length, options: options)
    }

    /// Create a MetalBuffer from an array.
    ///
    /// - Parameters:
    ///   - data: Array of elements
    ///   - options: Metal resource options
    /// - Returns: MetalBuffer wrapper or nil if allocation fails
    public func createMetalBuffer<T>(
        from data: [T],
        options: MTLResourceOptions? = nil
    ) -> MetalBuffer? {
        factory.createMetalBuffer(from: data, options: options)
    }

    // MARK: - Bucketed Buffer Creation

    /// Create a buffer using bucket sizing for efficient pooling.
    ///
    /// - Parameters:
    ///   - requestedSize: The minimum size needed
    ///   - options: Metal resource options (uses default if nil)
    /// - Returns: Buffer with bucket-rounded size, or nil if allocation fails
    public func createBucketedBuffer(
        size requestedSize: Int,
        options: MTLResourceOptions? = nil
    ) -> (any MTLBuffer)? {
        factory.createBucketedBuffer(size: requestedSize, options: options)
    }

    // MARK: - Buffer Utilities

    /// Standard bucket sizes for buffer pooling.
    public static var standardBucketSizes: [Int] {
        MetalBufferFactory.standardBucketSizes
    }

    /// Select appropriate bucket size for a requested size.
    ///
    /// - Parameter requestedSize: The size needed in bytes
    /// - Returns: The smallest bucket size >= requestedSize
    public static func selectBucketSize(for requestedSize: Int) -> Int {
        MetalBufferFactory.selectBucketSize(for: requestedSize)
    }

    /// Validate buffer alignment for SIMD operations.
    ///
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - alignment: Required alignment in bytes (default: 16)
    /// - Returns: True if buffer is properly aligned
    public static func isBufferAligned(_ buffer: any MTLBuffer, alignment: Int = 16) -> Bool {
        MetalBufferFactory.isBufferAligned(buffer, alignment: alignment)
    }
}

// MARK: - Shared Context Registry (Actor)

/// Actor-based registry for managing shared Metal contexts.
///
/// This actor provides thread-safe storage and retrieval of shared contexts.
/// Using an actor instead of locks ensures proper async-safety in Swift 6.
public actor SharedContextRegistry {

    /// Singleton instance of the registry.
    public static let shared = SharedContextRegistry()

    /// Storage for shared contexts keyed by identifier.
    private var contexts: [String: MetalContext] = [:]

    /// Storage for shared configurations.
    private var configurations: [String: SharedConfiguration] = [:]

    private init() {}

    // MARK: - Context Management

    /// Get an existing context or nil if not found.
    func getContext(identifier: String) -> MetalContext? {
        contexts[identifier]
    }

    /// Get the configuration for a context.
    func getConfiguration(identifier: String) -> SharedConfiguration? {
        configurations[identifier]
    }

    /// Register a context with its configuration.
    func register(context: MetalContext, configuration: SharedConfiguration) {
        contexts[configuration.identifier] = context
        configurations[configuration.identifier] = configuration
    }

    /// Remove a context by identifier.
    @discardableResult
    func remove(identifier: String) -> MetalContext? {
        configurations.removeValue(forKey: identifier)
        return contexts.removeValue(forKey: identifier)
    }

    /// Clear all registered contexts.
    func clearAll() {
        contexts.removeAll()
        configurations.removeAll()
    }

    /// Get all registered identifiers.
    var identifiers: [String] {
        Array(contexts.keys)
    }

    /// Get the count of registered contexts.
    var count: Int {
        contexts.count
    }
}

// MARK: - MetalContext Shared Context Extension

extension MetalContext {

    // MARK: - Shared Context Creation

    /// Create or retrieve a shared Metal context.
    ///
    /// If a context with the same identifier already exists, it will be returned.
    /// Otherwise, a new context is created, registered, and returned.
    ///
    /// ## Usage
    /// ```swift
    /// let config = SharedConfiguration(
    ///     queuePriority: .high,
    ///     identifier: "embedding-pipeline"
    /// )
    ///
    /// // First call creates the context
    /// let context1 = try await MetalContext.create(sharedConfig: config)
    ///
    /// // Subsequent calls return the same context
    /// let context2 = try await MetalContext.create(sharedConfig: config)
    /// // context1 === context2
    /// ```
    ///
    /// - Parameter sharedConfig: Configuration for the shared context
    /// - Returns: A Metal context configured for sharing
    /// - Throws: `VectorError` if context creation fails
    public static func create(sharedConfig: SharedConfiguration) async throws -> MetalContext {
        let registry = SharedContextRegistry.shared

        // Check for existing shared context
        if let existing = await registry.getContext(identifier: sharedConfig.identifier) {
            return existing
        }

        // Create new context with base configuration
        let context = try await MetalContext(configuration: sharedConfig.baseConfiguration)

        // Double-check and register (another task might have created it)
        if let existing = await registry.getContext(identifier: sharedConfig.identifier) {
            return existing
        }

        await registry.register(context: context, configuration: sharedConfig)
        return context
    }

    // MARK: - Shared Buffer Factory

    /// Get a shared buffer factory for this context.
    ///
    /// The returned factory is configured with the provided shared configuration
    /// and can be used by multiple components to create buffers that share
    /// the same underlying device.
    ///
    /// - Parameter configuration: The shared configuration for the factory
    /// - Returns: A `SharedBufferFactory` for creating shared buffers
    public func sharedBufferFactory(configuration: SharedConfiguration) -> SharedBufferFactory {
        SharedBufferFactory(factory: bufferFactory, configuration: configuration)
    }

    /// Get a shared buffer factory using the default shared configuration.
    ///
    /// Convenience method that uses `SharedConfiguration.default`.
    ///
    /// - Returns: A `SharedBufferFactory` with default configuration
    public func sharedBufferFactory() -> SharedBufferFactory {
        SharedBufferFactory(factory: bufferFactory, configuration: .default)
    }

    // MARK: - Shared Context Management

    /// Get an existing shared context by identifier.
    ///
    /// - Parameter identifier: The identifier used when creating the context
    /// - Returns: The shared context, or nil if none exists with that identifier
    public static func getSharedContext(identifier: String) async -> MetalContext? {
        await SharedContextRegistry.shared.getContext(identifier: identifier)
    }

    /// Get the configuration for a shared context.
    ///
    /// - Parameter identifier: The identifier of the shared context
    /// - Returns: The shared configuration, or nil if not found
    public static func getSharedConfiguration(identifier: String) async -> SharedConfiguration? {
        await SharedContextRegistry.shared.getConfiguration(identifier: identifier)
    }

    /// Remove a shared context from the registry.
    ///
    /// After removal, subsequent calls to `create(sharedConfig:)` with
    /// the same identifier will create a new context.
    ///
    /// - Parameter identifier: The identifier of the context to remove
    /// - Returns: The removed context, or nil if none existed
    @discardableResult
    public static func removeSharedContext(identifier: String) async -> MetalContext? {
        await SharedContextRegistry.shared.remove(identifier: identifier)
    }

    /// Clear all shared contexts from the registry.
    ///
    /// Use with caution - this invalidates all shared context references.
    public static func clearSharedContexts() async {
        await SharedContextRegistry.shared.clearAll()
    }

    /// Get all registered shared context identifiers.
    ///
    /// - Returns: Array of identifier strings for all registered shared contexts
    public static func sharedContextIdentifiers() async -> [String] {
        await SharedContextRegistry.shared.identifiers
    }

    /// Get the count of registered shared contexts.
    ///
    /// - Returns: Number of shared contexts currently registered
    public static func sharedContextCount() async -> Int {
        await SharedContextRegistry.shared.count
    }
}
