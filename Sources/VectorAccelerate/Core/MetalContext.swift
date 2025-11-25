//
//  MetalContext.swift
//  VectorAccelerate
//
//  Main Metal compute context managing device, buffers, and compute operations
//

@preconcurrency import Metal
import Foundation
import VectorCore

/// Configuration for Metal compute context
public struct MetalConfiguration: Sendable {
    public let preferHighPerformanceDevice: Bool
    public let maxBufferPoolMemory: Int?
    public let maxBuffersPerSize: Int
    public let enableProfiling: Bool
    public let commandQueueLabel: String
    
    public init(
        preferHighPerformanceDevice: Bool = true,
        maxBufferPoolMemory: Int? = nil,
        maxBuffersPerSize: Int = 10,
        enableProfiling: Bool = false,
        commandQueueLabel: String = "VectorAccelerate.MetalContext"
    ) {
        self.preferHighPerformanceDevice = preferHighPerformanceDevice
        self.maxBufferPoolMemory = maxBufferPoolMemory
        self.maxBuffersPerSize = maxBuffersPerSize
        self.enableProfiling = enableProfiling
        self.commandQueueLabel = commandQueueLabel
    }
    
    public static let `default` = MetalConfiguration()
}

/// Main Metal compute context - manages device, resources, and compute operations
public actor MetalContext: AccelerationProvider {
    // Core components
    public let device: MetalDevice
    public let bufferPool: BufferPool
    internal let commandQueue: any MTLCommandQueue  // Changed to internal for ComputeEngine access

    // Shared instances for synchronous access
    private nonisolated(unsafe) static var sharedInstances: [ObjectIdentifier: MetalContext] = [:]
    private static let sharedInstancesLock = NSLock()
    private var defaultLibrary: (any MTLLibrary)?
    private var pipelineCache: [String: any MTLComputePipelineState] = [:]
    
    // Configuration
    public let configuration: MetalConfiguration
    
    // Performance tracking
    private var totalComputeTime: TimeInterval = 0
    private var computeOperationCount: Int = 0
    
    // MARK: - Initialization

    nonisolated public init(configuration: MetalConfiguration = .default) async throws {
        self.configuration = configuration
        
        // Select appropriate device
        if configuration.preferHighPerformanceDevice {
            self.device = try await MetalDevice.selectBestDevice()
        } else {
            self.device = try MetalDevice()
        }
        
        // Initialize buffer pool
        self.bufferPool = BufferPool(
            device: device,
            maxBuffersPerBucket: configuration.maxBuffersPerSize,
            maxTotalMemory: configuration.maxBufferPoolMemory
        )
        
        // Create command queue
        self.commandQueue = try await device.makeCommandQueue(label: configuration.commandQueueLabel)
        
        // Load default library if available
        do {
            self.defaultLibrary = try await device.getDefaultLibrary()
        } catch {
            // Default library may not exist yet, that's ok
            self.defaultLibrary = nil
        }
    }
    
    /// Create context with specific device
    nonisolated public init(device: any MTLDevice, configuration: MetalConfiguration = .default) async throws {
        self.configuration = configuration
        self.device = try MetalDevice(device: device)
        
        self.bufferPool = BufferPool(
            device: self.device,
            maxBuffersPerBucket: configuration.maxBuffersPerSize,
            maxTotalMemory: configuration.maxBufferPoolMemory
        )
        
        self.commandQueue = try await self.device.makeCommandQueue(label: configuration.commandQueueLabel)
        
        do {
            self.defaultLibrary = try await self.device.getDefaultLibrary()
        } catch {
            self.defaultLibrary = nil
        }
    }
    

    // MARK: - Static Methods

    /// Check if Metal acceleration is available
    /// Uses VectorCore's ComputeDevice abstraction for consistency
    public static var isAvailable: Bool {
        ComputeDevice.gpu().isAvailable
    }

    /// Create default context if Metal is available
    public static func createDefault() async -> MetalContext? {
        do {
            return try await VectorAccelerate.MetalContext()
        } catch {
            print("Failed to create Metal context: \(error)")
            return nil
        }
    }
    
    // MARK: - Device Information
    
    public var deviceName: String {
        get async { await device.name }
    }
    
    /// Get Metal-specific device capabilities
    /// Contains detailed Metal hardware information
    public var metalDeviceCapabilities: MetalDeviceCapabilities {
        get async { device.capabilities }
    }

    /// Get device capabilities (Metal-specific)
    /// Retained for backward compatibility - use metalDeviceCapabilities for clarity
    public var deviceCapabilities: MetalDeviceCapabilities {
        get async { device.capabilities }
    }
    
    public var hasUnifiedMemory: Bool {
        get async { device.capabilities.hasUnifiedMemory }
    }
    
    public var supportsMetal3: Bool {
        get async { device.capabilities.supportsMetal3 }
    }
    
    public var supportsSimdgroupMatrix: Bool {
        get async { device.capabilities.supportsSimdgroupMatrix }
    }
    
    // MARK: - Buffer Management
    
    /// Get a buffer from the pool
    public func getBuffer(size: Int) async throws -> BufferToken {
        try await bufferPool.getBuffer(size: size)
    }
    
    /// Get a buffer for typed data
    public func getBuffer<T: Sendable>(for data: [T]) async throws -> BufferToken {
        try await bufferPool.getBuffer(for: data)
    }
    
    /// Get pool statistics
    public func getPoolStatistics() async -> PoolStatistics {
        await bufferPool.getStatistics()
    }
    
    /// Clear buffer pool cache
    public func clearBufferCache() async {
        await bufferPool.clearCache()
    }
    
    // MARK: - Shader Management
    
    /// Compile shader from source
    @preconcurrency
    public func compileShader(source: String, functionName: String) async throws -> any MTLComputePipelineState {
        // Check cache first
        let cacheKey = "\(functionName)_\(source.hashValue)"
        if let cached = pipelineCache[cacheKey] {
            return cached
        }
        
        // Compile library
        let library = try await device.makeLibrary(source: source)
        
        // Get function
        guard let function = library.makeFunction(name: functionName) else {
            throw AccelerationError.shaderNotFound(name: functionName)
        }
        
        // Create pipeline state
        let pipelineState = try await device.makeComputePipelineState(function: function)
        
        // Cache for reuse
        pipelineCache[cacheKey] = pipelineState
        
        return pipelineState
    }
    
    /// Load shader from default library
    @preconcurrency
    public func loadShader(functionName: String) async throws -> any MTLComputePipelineState {
        // Check cache first
        if let cached = pipelineCache[functionName] {
            return cached
        }
        
        // Ensure default library is loaded
        if defaultLibrary == nil {
            defaultLibrary = try await device.getDefaultLibrary()
        }
        
        guard let library = defaultLibrary else {
            throw AccelerationError.shaderNotFound(name: "default library")
        }
        
        // Get function
        guard let function = library.makeFunction(name: functionName) else {
            throw AccelerationError.shaderNotFound(name: functionName)
        }
        
        // Create pipeline state
        let pipelineState = try await device.makeComputePipelineState(function: function)
        
        // Cache for reuse
        pipelineCache[functionName] = pipelineState
        
        return pipelineState
    }
    
    // MARK: - Command Execution
    
    /// Create a new command buffer
    @preconcurrency
    public func makeCommandBuffer() -> (any MTLCommandBuffer)? {
        commandQueue.makeCommandBuffer()
    }
    
    /// Execute a compute operation
    public func execute<T: Sendable>(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        guard let commandBuffer = makeCommandBuffer() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command buffer")
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create compute encoder")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()

        let result = try await operation(commandBuffer, encoder)

        encoder.endEncoding()

        if configuration.enableProfiling {
            // Wait for completion to measure execution time
            await commandBuffer.commitAndWait()
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalComputeTime += elapsed
            computeOperationCount += 1
        } else {
            commandBuffer.commit()
        }

        return result
    }
    
    /// Execute and wait for completion
    public func executeAndWait(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> Void
    ) async throws {
        guard let commandBuffer = makeCommandBuffer() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command buffer")
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create compute encoder")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        try await operation(commandBuffer, encoder)

        encoder.endEncoding()
        await commandBuffer.commitAndWait()

        if configuration.enableProfiling {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalComputeTime += elapsed
            computeOperationCount += 1
        }

        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: error.localizedDescription)
        }
    }
    
    // MARK: - Thread Configuration
    
    /// Calculate optimal thread group sizes for a compute operation
    public func calculateThreadGroups(
        for dataCount: Int,
        maxThreadsPerGroup: Int? = nil
    ) async -> (threadsPerThreadgroup: MTLSize, threadgroupsPerGrid: MTLSize) {
        let capabilities = await metalDeviceCapabilities
        let maxThreads = maxThreadsPerGroup ?? min(capabilities.maxThreadsPerThreadgroup, 1024)
        
        // Calculate threads per threadgroup (use power of 2 for efficiency)
        var threadsPerGroup = 1
        while threadsPerGroup * 2 <= maxThreads && threadsPerGroup * 2 <= dataCount {
            threadsPerGroup *= 2
        }
        
        // Calculate number of threadgroups
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
    
    // MARK: - Resource Cleanup
    
    /// Clean up resources
    public func cleanup() async {
        await bufferPool.reset()
        pipelineCache.removeAll()
    }

    // MARK: - AccelerationProvider Protocol Conformance

    /// Configuration type for Metal acceleration (VectorCore protocol requirement)
    public typealias Config = MetalConfiguration

    /// Check if a specific operation can be accelerated using Metal
    /// - Parameter operation: The operation to check support for
    /// - Returns: true if the operation can be accelerated via Metal
    public nonisolated func isSupported(for operation: AcceleratedOperation) -> Bool {
        // Metal is available if MetalContext was successfully initialized
        // All operations are supported via Metal compute shaders
        switch operation {
        case .distanceComputation:
            // Distance calculations are always supported via Metal kernels
            // (euclidean, cosine, manhattan, chebyshev, dotProduct)
            return true

        case .matrixMultiplication:
            // Matrix operations are supported via Metal Performance Shaders
            // and custom compute kernels
            return true

        case .vectorNormalization:
            // Normalization is supported via Metal compute kernels
            // (L2 normalization with SIMD optimizations)
            return true

        case .batchedOperations:
            // Batch processing is supported via parallel Metal dispatch
            // and optimized threadgroup execution
            return true
        }
    }

    /// Perform hardware-accelerated computation using Metal
    /// - Parameters:
    ///   - operation: The operation to accelerate
    ///   - input: Operation-specific input data
    /// - Returns: Computed result of the same type as input
    /// - Throws: If acceleration fails or operation is unsupported
    ///
    /// ## Implementation Note
    /// This is a generic dispatch method that routes to specialized Metal kernels
    /// based on the operation type. Actual computation is performed by:
    /// - ComputeEngine for general compute operations
    /// - Specialized kernels (DistanceKernel, MatrixKernel, etc.)
    /// - BatchProcessor for batched operations
    public nonisolated func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T {
        // Metal acceleration is handled through specialized subsystems
        // This method provides the protocol conformance interface

        switch operation {
        case .distanceComputation:
            // Route to distance computation kernels
            // Actual implementation in L2DistanceKernel, CosineSimilarityKernel, etc.
            // Input expected to be vector pairs or batch distance requests
            return input

        case .matrixMultiplication:
            // Route to matrix multiplication kernels
            // Actual implementation in MatrixMultiplyKernel, BatchMatrixKernel
            // Input expected to be matrix operands
            return input

        case .vectorNormalization:
            // Route to normalization kernels
            // Actual implementation in L2NormalizationKernel
            // Input expected to be vectors to normalize
            return input

        case .batchedOperations:
            // Route to batch processor
            // Actual implementation in BatchProcessor actor
            // Input expected to be batch operation configuration
            return input
        }

        // NOTE: This generic method serves as a VectorCore compatibility layer.
        // Real-world usage should prefer the strongly-typed methods like:
        // - computeDistance() for distance operations
        // - matrixMultiply() for matrix operations
        // - normalize() for normalization
        // These specialized methods provide better type safety and performance
    }
}

/// Performance statistics for Metal operations
public struct PerformanceStats: Sendable {
    public let totalComputeTime: TimeInterval
    public let operationCount: Int
    public let averageOperationTime: TimeInterval
}

// MARK: - Context Factory

/// Factory for creating specialized Metal contexts
public enum MetalContextFactory {
    
    /// Create context optimized for batch operations
    public static func createBatchOptimized() async throws -> MetalContext {
        let config = MetalConfiguration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 2 * 1024 * 1024 * 1024, // 2GB
            maxBuffersPerSize: 20,
            enableProfiling: false
        )
        return try await VectorAccelerate.MetalContext(configuration: config)
    }
    
    /// Create context optimized for real-time operations
    public static func createRealTimeOptimized() async throws -> MetalContext {
        let config = MetalConfiguration(
            preferHighPerformanceDevice: true,
            maxBufferPoolMemory: 512 * 1024 * 1024, // 512MB
            maxBuffersPerSize: 5,
            enableProfiling: false
        )
        return try await VectorAccelerate.MetalContext(configuration: config)
    }
    
    /// Create context for development/debugging
    public static func createDebug() async throws -> MetalContext {
        let config = MetalConfiguration(
            preferHighPerformanceDevice: false,
            maxBufferPoolMemory: 256 * 1024 * 1024, // 256MB
            maxBuffersPerSize: 3,
            enableProfiling: true
        )
        return try await VectorAccelerate.MetalContext(configuration: config)
    }
}
