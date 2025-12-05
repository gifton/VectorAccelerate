//
//  Metal4Context.swift
//  VectorAccelerate
//
//  Metal 4 context actor with explicit residency management and unified command encoding
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Configuration for Metal 4 compute context
public struct Metal4Configuration: Sendable {
    public let preferHighPerformanceDevice: Bool
    public let maxBufferPoolMemory: Int?
    public let maxBuffersPerSize: Int
    public let enableProfiling: Bool
    public let commandQueueLabel: String
    public let maxCommandBufferCount: Int
    public let residencySetCapacity: Int

    public init(
        preferHighPerformanceDevice: Bool = true,
        maxBufferPoolMemory: Int? = nil,
        maxBuffersPerSize: Int = 10,
        enableProfiling: Bool = false,
        commandQueueLabel: String = "VectorAccelerate.Metal4Context",
        maxCommandBufferCount: Int = 3,
        residencySetCapacity: Int = 256
    ) {
        self.preferHighPerformanceDevice = preferHighPerformanceDevice
        self.maxBufferPoolMemory = maxBufferPoolMemory
        self.maxBuffersPerSize = maxBuffersPerSize
        self.enableProfiling = enableProfiling
        self.commandQueueLabel = commandQueueLabel
        self.maxCommandBufferCount = maxCommandBufferCount
        self.residencySetCapacity = residencySetCapacity
    }

    public static let `default` = Metal4Configuration()
}

/// Metal 4 compute context with explicit residency management
///
/// This actor provides the primary interface for Metal 4 GPU compute operations.
/// Key differences from MetalContext (Metal 3):
/// - Explicit residency management via ResidencyManager
/// - Command buffers created from device (not queue)
/// - Event-based synchronization via MTLSharedEvent
/// - Unified command encoding
///
/// ## Concurrency Model
/// - One command buffer per `execute()` call
/// - Multiple overlapping operations are allowed
/// - Use `executeAndWait()` when ordering matters
///
/// Example:
/// ```swift
/// let context = try await Metal4Context()
/// try await context.executeAndWait { commandBuffer, encoder in
///     encoder.setComputePipelineState(pipeline)
///     encoder.setArgumentTable(argTable, stages: .compute)
///     encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
/// }
/// ```
public actor Metal4Context {
    // MARK: - Core Components

    /// The Metal device wrapper
    /// Nonisolated because MetalDevice is itself an actor with its own isolation
    public nonisolated let device: MetalDevice

    /// Metal 4 capabilities for this device
    public let capabilities: Metal4Capabilities

    /// Buffer pool for efficient buffer management
    public let bufferPool: BufferPool

    /// Synchronous buffer factory
    public nonisolated let bufferFactory: MetalBufferFactory

    /// Residency manager for explicit resource tracking
    public let residencyManager: ResidencyManager

    /// Argument table pool for efficient resource binding
    public let argumentTablePool: ArgumentTablePool

    /// Metal 4 shader compiler with QoS support
    public let shaderCompiler: Metal4ShaderCompiler

    /// Pipeline cache for efficient pipeline reuse
    public let pipelineCache: PipelineCache

    // MARK: - Command Infrastructure

    /// Metal 4 command queue
    internal let commandQueue: any MTLCommandQueue

    /// Shared event for synchronization
    private let completionEvent: any MTLSharedEvent

    /// Event counter for ordering
    private var eventCounter: UInt64 = 0

    // MARK: - Configuration

    /// Active configuration
    public let configuration: Metal4Configuration

    // MARK: - Performance Tracking

    private var totalComputeTime: TimeInterval = 0
    private var computeOperationCount: Int = 0

    // MARK: - Shader Management

    private var defaultLibrary: (any MTLLibrary)?
    private var legacyPipelineCache: [String: any MTLComputePipelineState] = [:]

    // MARK: - Initialization

    /// Create a Metal 4 context with default configuration
    public init() async throws {
        try await self.init(configuration: .default)
    }

    /// Create a Metal 4 context with custom configuration
    public init(configuration: Metal4Configuration) async throws {
        self.configuration = configuration

        // Select device
        if configuration.preferHighPerformanceDevice {
            self.device = try await MetalDevice.selectBestDevice()
        } else {
            self.device = try MetalDevice()
        }

        // Initialize capabilities
        self.capabilities = Metal4Capabilities(device: device.rawDevice)

        // Verify Metal 4 support
        guard capabilities.supportsMetal4Core else {
            throw VectorError.metal4NotSupported(reason: "Device does not support Metal 4")
        }

        // Create buffer factory
        self.bufferFactory = device.makeBufferFactory()

        // Create residency manager
        self.residencyManager = try ResidencyManager(
            device: device.rawDevice,
            initialCapacity: configuration.residencySetCapacity
        )

        // Create argument table pool
        self.argumentTablePool = ArgumentTablePool(
            device: device.rawDevice,
            maxTables: 32
        )

        // Create shader compiler
        self.shaderCompiler = try Metal4ShaderCompiler(
            device: device.rawDevice,
            configuration: .default
        )

        // Create pipeline cache
        self.pipelineCache = PipelineCacheFactory.createMemoryOnly(
            compiler: shaderCompiler,
            maxSize: 100
        )

        // Create command queue
        // Note: In actual Metal 4, this would use MTL4CommandQueueDescriptor
        guard let queue = device.rawDevice.makeCommandQueue() else {
            throw VectorError.metal4CommandQueueCreationFailed()
        }
        queue.label = configuration.commandQueueLabel
        self.commandQueue = queue

        // Attach residency set to queue
        // Note: In actual Metal 4, queue.addResidencySet(residencyManager.underlyingResidencySet)

        // Create completion event
        guard let event = device.rawDevice.makeSharedEvent() else {
            throw VectorError.sharedEventCreationFailed()
        }
        self.completionEvent = event

        // Initialize buffer pool
        self.bufferPool = BufferPool(
            device: device,
            factory: bufferFactory,
            maxBuffersPerBucket: configuration.maxBuffersPerSize,
            maxTotalMemory: configuration.maxBufferPoolMemory
        )

        // Load default library if available
        do {
            self.defaultLibrary = try await device.getDefaultLibrary()
        } catch {
            self.defaultLibrary = nil
        }
    }

    // MARK: - Command Buffer Creation

    /// Create a new command buffer from device
    ///
    /// In Metal 4, command buffers are created from the device, not the queue.
    /// This allows for better command buffer reuse and parallel encoding.
    public func makeCommandBuffer() -> (any MTLCommandBuffer)? {
        // In Metal 4: device.makeCommandBuffer() as? MTL4CommandBuffer
        commandQueue.makeCommandBuffer()
    }

    // MARK: - Execution

    /// Execute a compute operation
    ///
    /// Creates a new command buffer, executes the operation, and commits.
    /// The operation is submitted asynchronously - use `executeAndWait()` if
    /// you need to wait for completion.
    ///
    /// - Parameter operation: Closure receiving command buffer and encoder
    /// - Returns: Result from the operation
    public func execute<T: Sendable>(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        // Check for cancellation
        try Task.checkCancellation()

        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        // In Metal 4: commandBuffer.useResidencySet(residencyManager.underlyingResidencySet)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        let result = try await operation(commandBuffer, encoder)

        encoder.endEncoding()

        // Submit via queue (Metal 4 pattern)
        // In Metal 4: commandQueue.commit([commandBuffer])
        commandBuffer.commit()

        if configuration.enableProfiling {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalComputeTime += elapsed
            computeOperationCount += 1
        }

        return result
    }

    /// Execute and wait for GPU completion
    ///
    /// Creates a command buffer, executes the operation, commits, and waits
    /// for GPU completion using MTLSharedEvent.
    ///
    /// - Parameter operation: Closure receiving command buffer and encoder
    public func executeAndWait(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> Void
    ) async throws {
        // Check for cancellation
        try Task.checkCancellation()

        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        // In Metal 4: commandBuffer.useResidencySet(residencyManager.underlyingResidencySet)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        try await operation(commandBuffer, encoder)

        encoder.endEncoding()

        // Increment event counter
        eventCounter += 1
        let targetValue = eventCounter

        // Submit and signal
        // In Metal 4:
        // commandQueue.commit([commandBuffer])
        // commandQueue.signalEvent(completionEvent, value: targetValue)
        // Capture event outside closure to avoid actor isolation issues
        let event = completionEvent
        commandBuffer.addCompletedHandler { _ in
            event.signaledValue = targetValue
        }
        commandBuffer.commit()

        // Check for cancellation before waiting
        try Task.checkCancellation()

        // Wait for completion using shared event
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            completionEvent.notify(MTLSharedEventListener(dispatchQueue: .global()), atValue: targetValue) { _, _ in
                continuation.resume()
            }
        }

        if configuration.enableProfiling {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalComputeTime += elapsed
            computeOperationCount += 1
        }

        // Check for errors
        if let error = commandBuffer.error {
            throw VectorError.computeFailed(reason: error.localizedDescription)
        }
    }

    /// Execute a blit operation and wait for GPU completion
    ///
    /// Creates a command buffer with a blit encoder, executes the operation,
    /// commits, and waits for GPU completion.
    ///
    /// - Parameter operation: Closure receiving command buffer and blit encoder
    public func executeBlitAndWait(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLBlitCommandEncoder) throws -> Void
    ) async throws {
        try Task.checkCancellation()

        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        try operation(commandBuffer, blitEncoder)

        blitEncoder.endEncoding()

        // Increment event counter
        eventCounter += 1
        let targetValue = eventCounter

        // Capture event outside closure to avoid actor isolation issues
        let event = completionEvent
        commandBuffer.addCompletedHandler { _ in
            event.signaledValue = targetValue
        }
        commandBuffer.commit()

        try Task.checkCancellation()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            completionEvent.notify(MTLSharedEventListener(dispatchQueue: .global()), atValue: targetValue) { _, _ in
                continuation.resume()
            }
        }

        if let error = commandBuffer.error {
            throw VectorError.computeFailed(reason: error.localizedDescription)
        }
    }

    // MARK: - Buffer Management

    /// Get a buffer from the pool with automatic residency registration
    public func getBuffer(size: Int) async throws -> BufferToken {
        let token = try await bufferPool.getBuffer(size: size)

        // Register with residency manager
        try await residencyManager.registerBuffer(token.buffer, mode: .ephemeral)

        return token
    }

    /// Get a buffer for typed data with automatic residency
    public func getBuffer<T: Sendable>(for data: [T]) async throws -> BufferToken {
        let token = try await bufferPool.getBuffer(for: data)

        // Register with residency manager
        try await residencyManager.registerBuffer(token.buffer, mode: .ephemeral)

        return token
    }

    /// Commit any pending residency changes
    public func commitResidency() async throws {
        try await residencyManager.commit()
    }

    // MARK: - Shader Management

    /// Compile shader from source
    @preconcurrency
    public func compileShader(source: String, functionName: String) async throws -> any MTLComputePipelineState {
        let cacheKey = "\(functionName)_\(source.hashValue)"
        if let cached = legacyPipelineCache[cacheKey] {
            return cached
        }

        let library = try await device.makeLibrary(source: source)

        guard let function = library.makeFunction(name: functionName) else {
            throw VectorError.shaderNotFound(name: functionName)
        }

        let pipelineState = try await device.makeComputePipelineState(function: function)
        legacyPipelineCache[cacheKey] = pipelineState

        return pipelineState
    }

    /// Load shader from default library
    @preconcurrency
    public func loadShader(functionName: String) async throws -> any MTLComputePipelineState {
        if let cached = legacyPipelineCache[functionName] {
            return cached
        }

        if defaultLibrary == nil {
            defaultLibrary = try await device.getDefaultLibrary()
        }

        guard let library = defaultLibrary else {
            throw VectorError.shaderNotFound(name: "default library")
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw VectorError.shaderNotFound(name: functionName)
        }

        let pipelineState = try await device.makeComputePipelineState(function: function)
        legacyPipelineCache[functionName] = pipelineState

        return pipelineState
    }

    /// Get pipeline using Metal 4 cache key system
    public func getPipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        try await pipelineCache.getPipeline(for: key)
    }

    /// Get pipeline by function name using new cache
    public func getPipeline(functionName: String) async throws -> any MTLComputePipelineState {
        try await pipelineCache.getPipeline(functionName: functionName)
    }

    /// Warm up pipeline cache with common pipelines
    public func warmUpPipelineCache() async {
        await pipelineCache.warmUpStandard()
    }

    /// Warm up pipeline cache for embedding workloads
    public func warmUpEmbeddingPipelines() async {
        await pipelineCache.warmUpEmbeddings()
    }

    // MARK: - Thread Configuration

    /// Calculate optimal thread group sizes
    public func calculateThreadGroups(
        for dataCount: Int,
        maxThreadsPerGroup: Int? = nil
    ) async -> (threadsPerThreadgroup: MTLSize, threadgroupsPerGrid: MTLSize) {
        let deviceCaps = device.capabilities
        let maxThreads = maxThreadsPerGroup ?? min(deviceCaps.maxThreadsPerThreadgroup, 1024)

        var threadsPerGroup = 1
        while threadsPerGroup * 2 <= maxThreads && threadsPerGroup * 2 <= dataCount {
            threadsPerGroup *= 2
        }

        let threadgroups = (dataCount + threadsPerGroup - 1) / threadsPerGroup

        return (
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1),
            threadgroupsPerGrid: MTLSize(width: threadgroups, height: 1, depth: 1)
        )
    }

    // MARK: - Performance Metrics

    /// Get performance statistics
    public func getPerformanceStats() -> PerformanceStats {
        let averageTime = computeOperationCount > 0
            ? totalComputeTime / Double(computeOperationCount)
            : 0

        return PerformanceStats(
            totalComputeTime: totalComputeTime,
            operationCount: computeOperationCount,
            averageOperationTime: averageTime
        )
    }

    /// Reset performance counters
    public func resetPerformanceStats() {
        totalComputeTime = 0
        computeOperationCount = 0
    }

    // MARK: - Cleanup

    /// Clean up resources
    public func cleanup() async {
        await bufferPool.reset()
        await residencyManager.clear()
        await pipelineCache.clear()
        legacyPipelineCache.removeAll()
    }

    // MARK: - Device Information

    public var deviceName: String {
        get async { await device.name }
    }

    public var hasUnifiedMemory: Bool {
        get async { device.capabilities.hasUnifiedMemory }
    }

    // MARK: - Static Methods

    /// Check if Metal 4 acceleration is available
    public static var isAvailable: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }
        let caps = Metal4Capabilities(device: device)
        return caps.supportsMetal4Core
    }

    /// Create default context if Metal 4 is available
    public static func createDefault() async -> Metal4Context? {
        do {
            return try await Metal4Context()
        } catch {
            print("Failed to create Metal 4 context: \(error)")
            return nil
        }
    }
}

// MARK: - Context Factory

/// Factory for creating specialized Metal 4 contexts
public enum Metal4ContextFactory {

    /// Create context optimized for batch operations
    public static func createBatchOptimized() async throws -> Metal4Context {
        let config = Metal4Configuration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 2 * 1024 * 1024 * 1024, // 2GB
            maxBuffersPerSize: 20,
            enableProfiling: false,
            maxCommandBufferCount: 5,
            residencySetCapacity: 1024
        )
        return try await Metal4Context(configuration: config)
    }

    /// Create context optimized for real-time operations
    public static func createRealTimeOptimized() async throws -> Metal4Context {
        let config = Metal4Configuration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 512 * 1024 * 1024, // 512MB
            maxBuffersPerSize: 5,
            enableProfiling: false,
            maxCommandBufferCount: 2,
            residencySetCapacity: 256
        )
        return try await Metal4Context(configuration: config)
    }

    /// Create context for development/debugging
    public static func createDebug() async throws -> Metal4Context {
        let config = Metal4Configuration(
            preferHighPerformanceDevice: false,
            maxBufferPoolMemory: 256 * 1024 * 1024, // 256MB
            maxBuffersPerSize: 3,
            enableProfiling: true,
            maxCommandBufferCount: 2,
            residencySetCapacity: 128
        )
        return try await Metal4Context(configuration: config)
    }
}
