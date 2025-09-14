//
//  MetalDevice.swift
//  VectorAccelerate
//
//  Enhanced Metal device management with capability detection
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Metal device capabilities detection
public struct DeviceCapabilities: Sendable {
    public let hasUnifiedMemory: Bool
    public let maxThreadsPerThreadgroup: Int
    public let maxThreadgroupMemoryLength: Int
    public let maxBufferLength: Int
    public let supportsFloat16: Bool
    public let supportsBFloat16: Bool
    public let supportsSimdgroupMatrix: Bool
    public let supportsRaytracing: Bool
    public let supportsMetal3: Bool
    public let recommendedMaxWorkingSetSize: Int
    public let registryID: UInt64
    
    init(device: any MTLDevice) {
        // Basic capabilities
        self.hasUnifiedMemory = device.hasUnifiedMemory
        self.registryID = device.registryID
        
        // Memory limits
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
        self.maxBufferLength = device.maxBufferLength
        self.recommendedMaxWorkingSetSize = Int(device.recommendedMaxWorkingSetSize)
        
        // Threadgroup memory
        #if os(iOS) || os(tvOS) || os(visionOS)
        self.maxThreadgroupMemoryLength = 32768 // 32KB typical for iOS devices
        #elseif os(macOS)
        self.maxThreadgroupMemoryLength = 65536 // 64KB for macOS
        #else
        self.maxThreadgroupMemoryLength = 16384 // Conservative default
        #endif
        
        // Feature detection
        #if os(macOS)
        if #available(macOS 13.0, *) {
            self.supportsMetal3 = device.supportsFamily(.apple7)
        } else {
            self.supportsMetal3 = false
        }
        
        // Float16 support
        self.supportsFloat16 = device.supports32BitFloatFiltering
        
        // BFloat16 (Metal 3+)
        if #available(macOS 14.0, *) {
            self.supportsBFloat16 = device.supportsFamily(.apple8)
        } else {
            self.supportsBFloat16 = false
        }
        
        // Simdgroup matrix (Apple Silicon)
        if #available(macOS 13.0, *) {
            self.supportsSimdgroupMatrix = device.supportsFamily(.apple7) || 
                                          device.supportsFamily(.apple8) ||
                                          device.supportsFamily(.apple9)
        } else {
            self.supportsSimdgroupMatrix = false
        }
        
        // Raytracing (M1+ and some AMD GPUs)
        if #available(macOS 11.0, *) {
            self.supportsRaytracing = device.supportsRaytracing
        } else {
            self.supportsRaytracing = false
        }
        
        #else // iOS, tvOS, visionOS
        
        self.supportsMetal3 = false
        self.supportsFloat16 = device.supports32BitFloatFiltering
        self.supportsBFloat16 = false
        
        if #available(iOS 16.0, tvOS 16.0, *) {
            self.supportsSimdgroupMatrix = device.supportsFamily(.apple7) ||
                                          device.supportsFamily(.apple8) ||
                                          device.supportsFamily(.apple9)
        } else {
            self.supportsSimdgroupMatrix = false
        }
        
        self.supportsRaytracing = false
        #endif
    }
}

/// Enhanced Metal device wrapper with comprehensive capability detection
public actor MetalDevice {
    private let device: any MTLDevice
    public let capabilities: DeviceCapabilities
    private var commandQueues: [any MTLCommandQueue] = []
    private let queueLock = NSLock()
    
    // Performance monitoring
    private var deviceUtilization: Double = 0.0
    private var lastSampleTime: Date = Date()
    
    // MARK: - Initialization
    
    public init(device: (any MTLDevice)? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw AccelerationError.metalNotAvailable
        }
        
        self.device = device
        self.capabilities = DeviceCapabilities(device: device)
        
        // Create initial command queue
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueues = [queue]
    }
    
    // MARK: - Static Properties
    
    /// Check if Metal is available on this system
    public static var isAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
    
    /// Get all available Metal devices
    public static func availableDevices() -> [any MTLDevice] {
        #if os(macOS)
        return MTLCopyAllDevices()
        #else
        if let device = MTLCreateSystemDefaultDevice() {
            return [device]
        }
        return []
        #endif
    }
    
    // MARK: - Device Information
    
    public var name: String {
        device.name
    }
    
    public var isLowPower: Bool {
        device.isLowPower
    }
    
    public var isRemovable: Bool {
        device.isRemovable
    }
    
    public var isHeadless: Bool {
        device.isHeadless
    }
    
    public var location: MTLDeviceLocation {
        device.location
    }
    
    public var locationNumber: UInt64 {
        UInt64(device.locationNumber)
    }
    
    public var maxThreadgroupMemoryLength: Int {
        capabilities.maxThreadgroupMemoryLength
    }
    
    public var maxBufferLength: Int {
        capabilities.maxBufferLength
    }
    
    // MARK: - Command Queue Management
    
    /// Get or create a command queue
    @preconcurrency
    public func getCommandQueue() -> any MTLCommandQueue {
        queueLock.lock()
        defer { queueLock.unlock() }
        
        // Return existing queue if available
        if let queue = commandQueues.first {
            return queue
        }
        
        // Create new queue if needed
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        commandQueues.append(queue)
        return queue
    }
    
    /// Create a new dedicated command queue
    @preconcurrency
    public func makeCommandQueue(label: String? = nil) throws -> any MTLCommandQueue {
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        
        if let label = label {
            queue.label = label
        }
        
        queueLock.lock()
        commandQueues.append(queue)
        queueLock.unlock()
        
        return queue
    }
    
    // MARK: - Buffer Creation
    
    /// Create a buffer with data
    public func makeBuffer<T>(data: [T], options: MTLResourceOptions = []) -> (any MTLBuffer)? {
        let actualOptions = options.isEmpty ? getDefaultResourceOptions() : options
        let size = data.count * MemoryLayout<T>.stride
        
        return data.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: size, options: actualOptions)
        }
    }
    
    /// Create an empty buffer
    public func makeBuffer(length: Int, options: MTLResourceOptions = []) -> (any MTLBuffer)? {
        let actualOptions = options.isEmpty ? getDefaultResourceOptions() : options
        return device.makeBuffer(length: length, options: actualOptions)
    }
    
    /// Get default resource options based on device capabilities
    private func getDefaultResourceOptions() -> MTLResourceOptions {
        if capabilities.hasUnifiedMemory {
            // Apple Silicon - use shared memory
            return .storageModeShared
        } else {
            // Intel/AMD - use managed memory for automatic syncing
            #if os(macOS)
            return .storageModeManaged
            #else
            return .storageModeShared
            #endif
        }
    }
    
    // MARK: - Library Management
    
    /// Create a Metal library from source
    @preconcurrency
    public func makeLibrary(source: String) async throws -> any MTLLibrary {
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version3_0
        
        do {
            return try await withCheckedThrowingContinuation { continuation in
                device.makeLibrary(source: source, options: options) { library, error in
                    if let library = library {
                        continuation.resume(returning: library)
                    } else {
                        let error = error ?? AccelerationError.shaderCompilationFailed("Unknown error")
                        continuation.resume(throwing: error)
                    }
                }
            }
        } catch {
            throw AccelerationError.shaderCompilationFailed(error.localizedDescription)
        }
    }
    
    /// Get the default Metal library
    @preconcurrency
    public func getDefaultLibrary() throws -> any MTLLibrary {
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.shaderCompilationFailed("Failed to load default library")
        }
        return library
    }
    
    // MARK: - Pipeline State Creation
    
    /// Create a compute pipeline state
    @preconcurrency
    public func makeComputePipelineState(function: any MTLFunction) async throws -> any MTLComputePipelineState {
        do {
            return try await withCheckedThrowingContinuation { continuation in
                device.makeComputePipelineState(function: function) { state, error in
                    if let state = state {
                        continuation.resume(returning: state)
                    } else {
                        let error = error ?? AccelerationError.pipelineCreationFailed("Unknown error")
                        continuation.resume(throwing: error)
                    }
                }
            }
        } catch {
            throw AccelerationError.pipelineCreationFailed(error.localizedDescription)
        }
    }
    
    // MARK: - Performance Monitoring
    
    /// Update device utilization metrics
    public func updateUtilization(_ utilization: Double) {
        self.deviceUtilization = utilization
        self.lastSampleTime = Date()
    }
    
    /// Get current device utilization
    public func getUtilization() -> (utilization: Double, lastUpdate: Date) {
        (deviceUtilization, lastSampleTime)
    }
    
    // MARK: - Device Selection
    
    /// Select best device for computation
    public static func selectBestDevice() async throws -> MetalDevice {
        let devices = availableDevices()
        
        guard !devices.isEmpty else {
            throw AccelerationError.metalNotAvailable
        }
        
        // Prefer high-performance discrete GPUs on macOS
        #if os(macOS)
        // Look for discrete GPU first
        if let discreteGPU = devices.first(where: { !$0.isLowPower && !$0.isHeadless }) {
            return try MetalDevice(device: discreteGPU)
        }
        #endif
        
        // Fall back to default device
        return try MetalDevice()
    }
    
    // MARK: - Internal Access
    
    /// Get the underlying MTLDevice (for internal use)
    public func getDevice() -> any MTLDevice {
        device
    }
}
