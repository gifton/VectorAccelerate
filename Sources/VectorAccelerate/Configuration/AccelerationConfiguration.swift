// VectorAccelerate: Acceleration Configuration
//
// Configuration types for hardware acceleration settings

import Foundation
@preconcurrency import Metal

/// Main configuration for acceleration engine
public struct AccelerationConfiguration: Sendable {
    public let bufferConfig: BufferPoolConfiguration
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
        adaptiveThresholds: Bool = true,
        preferredDevice: PreferredDevice = .auto,
        enableExperimentalML: Bool = false
    ) {
        self.bufferConfig = bufferConfig
        self.adaptiveThresholds = adaptiveThresholds
        self.preferredDevice = preferredDevice
        self.enableExperimentalML = enableExperimentalML
    }

    public static let `default` = AccelerationConfiguration()
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