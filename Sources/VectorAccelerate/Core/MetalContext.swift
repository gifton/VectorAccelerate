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
public actor MetalContext {
    // Core components
    public let device: MetalDevice
    public let bufferPool: BufferPool
    private let commandQueue: any MTLCommandQueue
    private var defaultLibrary: (any MTLLibrary)?
    private var pipelineCache: [String: any MTLComputePipelineState] = [:]
    
    // Configuration
    public let configuration: MetalConfiguration
    
    // Performance tracking
    private var totalComputeTime: TimeInterval = 0
    private var computeOperationCount: Int = 0
    
    // MARK: - Initialization
    
    public init(configuration: MetalConfiguration = .default) async throws {
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
    public init(device: any MTLDevice, configuration: MetalConfiguration = .default) async throws {
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
    public static var isAvailable: Bool {
        MetalDevice.isAvailable
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
    
    public var deviceCapabilities: DeviceCapabilities {
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
        
        defer {
            encoder.endEncoding()
            commandBuffer.commit()
            
            if configuration.enableProfiling {
                commandBuffer.waitUntilCompleted()
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                totalComputeTime += elapsed
                computeOperationCount += 1
            }
        }
        
        return try await operation(commandBuffer, encoder)
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
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
        let capabilities = await deviceCapabilities
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
