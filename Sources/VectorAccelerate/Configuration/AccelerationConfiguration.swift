// VectorAccelerate: Acceleration Configuration
//
// Configuration types for hardware acceleration settings

import Foundation
@preconcurrency import Metal

/// Main configuration for acceleration engine
public struct AccelerationConfiguration: Sendable {
    public let bufferConfig: BufferPoolConfiguration

    /// Minimum operation count before considering CPU path.
    @available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing instead")
    public let cpuThreshold: Int

    /// Minimum operation count before considering GPU path.
    @available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing instead")
    public let gpuThreshold: Int

    /// Minimum operation count before considering hybrid CPU+GPU path.
    @available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing instead")
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

    // MARK: - Migration Helpers

    /// Converts this configuration to GPUActivationThresholds for use with GPUDecisionEngine.
    ///
    /// Use this method to migrate from the deprecated threshold-based configuration
    /// to the adaptive GPUDecisionEngine pattern.
    ///
    /// ```swift
    /// let oldConfig = AccelerationConfiguration.performance
    /// let thresholds = oldConfig.toGPUActivationThresholds()
    /// let engine = GPUDecisionEngine(thresholds: thresholds)
    /// ```
    public func toGPUActivationThresholds() -> GPUActivationThresholds {
        GPUActivationThresholds(
            minVectorsForGPU: cpuThreshold,
            minCandidatesForGPU: gpuThreshold / 10,  // Scale down for candidate-specific threshold
            minKForGPU: 10,
            maxKForGPU: 1000,
            minOperationsForGPU: gpuThreshold,
            maxGPUMemoryMB: bufferConfig.maxTotalSize / (1024 * 1024)
        )
    }

    /// Creates a GPUDecisionEngine configured with thresholds derived from this configuration.
    ///
    /// This is a convenience method for migrating to the new adaptive routing pattern.
    ///
    /// ```swift
    /// let config = AccelerationConfiguration.performance
    /// let engine = config.createDecisionEngine()
    /// ```
    public func createDecisionEngine() -> GPUDecisionEngine {
        GPUDecisionEngine(thresholds: toGPUActivationThresholds())
    }
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