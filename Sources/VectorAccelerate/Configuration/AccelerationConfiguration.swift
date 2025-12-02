// VectorAccelerate: Acceleration Configuration
//
// Configuration types for hardware acceleration settings

import Foundation
@preconcurrency import Metal

/// Main configuration for acceleration engine
public struct AccelerationConfiguration: Sendable {
    public let bufferConfig: BufferPoolConfiguration
    public let cpuThreshold: Int
    public let gpuThreshold: Int
    public let hybridThreshold: Int
    public let adaptiveThresholds: Bool
    public let preferredDevice: PreferredDevice

    /// Enable experimental ML features (Phase 4)
    ///
    /// When enabled, allows use of MTLTensor-based operations like:
    /// - Learned distance metrics with projection matrices
    /// - Neural quantization encoders/decoders
    /// - Attention-based similarity
    ///
    /// - Note: Requires `Metal4Capabilities.supportsMLTensor == true` at runtime.
    ///   Operations will fall back to standard implementations if MLTensor is unavailable.
    /// - Warning: These features are experimental and may change in future releases.
    public let enableExperimentalML: Bool

    public enum PreferredDevice: Sendable {
        case auto
        case cpu
        case gpu
        case hybrid
    }

    public init(
        bufferConfig: BufferPoolConfiguration = .default,
        cpuThreshold: Int = 1000,
        gpuThreshold: Int = 10000,
        hybridThreshold: Int = 100000,
        adaptiveThresholds: Bool = true,
        preferredDevice: PreferredDevice = .auto,
        enableExperimentalML: Bool = false
    ) {
        self.bufferConfig = bufferConfig
        self.cpuThreshold = cpuThreshold
        self.gpuThreshold = gpuThreshold
        self.hybridThreshold = hybridThreshold
        self.adaptiveThresholds = adaptiveThresholds
        self.preferredDevice = preferredDevice
        self.enableExperimentalML = enableExperimentalML
    }
    
    public static let `default` = AccelerationConfiguration()
    
    public static let performance = AccelerationConfiguration(
        bufferConfig: .largeVectors,
        cpuThreshold: 500,
        gpuThreshold: 5000,
        hybridThreshold: 50000,
        adaptiveThresholds: true,
        preferredDevice: .auto
    )
    
    public static let balanced = AccelerationConfiguration(
        bufferConfig: .default,
        cpuThreshold: 2000,
        gpuThreshold: 20000,
        hybridThreshold: 200000,
        adaptiveThresholds: true,
        preferredDevice: .auto
    )
    
    public static let cpuOnly = AccelerationConfiguration(
        bufferConfig: .smallVectors,
        cpuThreshold: Int.max,
        gpuThreshold: Int.max,
        hybridThreshold: Int.max,
        adaptiveThresholds: false,
        preferredDevice: .cpu
    )
}

/// Execution path for operations
public enum ExecutionPath: String, Sendable {
    case cpu = "CPU"
    case gpu = "GPU"
    case hybrid = "Hybrid"
}

/// Operation types for performance tracking
public enum OperationType: String, Sendable {
    case distanceComputation = "DistanceComputation"
    case batchDistanceComputation = "BatchDistanceComputation"
    case matrixMultiplication = "MatrixMultiplication"
    case quantization = "Quantization"
    case clustering = "Clustering"
    case indexing = "Indexing"
    case search = "Search"
}

/// Buffer pool configuration extensions
public extension BufferPoolConfiguration {
    static let smallVectors = BufferPoolConfiguration(
        maxBufferCount: 50,
        maxTotalSize: 256 * 1024 * 1024,  // 256MB
        reuseThreshold: 1024,
        adaptiveSizing: true
    )
    
    static let largeVectors = BufferPoolConfiguration(
        maxBufferCount: 200,
        maxTotalSize: 2 * 1024 * 1024 * 1024,  // 2GB
        reuseThreshold: 4096,
        adaptiveSizing: true
    )
}