//
//  IndexAccelerationConfiguration.swift
//  VectorAccelerate
//
//  Configuration options for index acceleration behavior.
//

import Foundation

// MARK: - Index Acceleration Configuration

/// Configuration for GPU-accelerated index operations.
///
/// Controls when and how GPU acceleration is applied to index searches.
///
/// ## Presets
/// - `.default` - Balanced settings for general use
/// - `.aggressive` - Maximize GPU usage
/// - `.conservative` - Only use GPU for large workloads
///
/// ## Custom Configuration
/// ```swift
/// let config = IndexAccelerationConfiguration(
///     minimumCandidatesForGPU: 1000,
///     useFusedKernels: true,
///     enableProfiling: true
/// )
/// ```
public struct IndexAccelerationConfiguration: Sendable {

    // MARK: - Threshold Settings

    /// Minimum number of candidate vectors to consider GPU acceleration.
    /// Below this threshold, CPU is typically faster due to GPU overhead.
    @available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing. See toGPUActivationThresholds().")
    public var minimumCandidatesForGPU: Int

    /// Minimum total operations (queries * candidates * dimension) for GPU.
    @available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing. See toGPUActivationThresholds().")
    public var minimumOperationsForGPU: Int

    /// Maximum batch size for GPU operations (memory constraint).
    public var maxBatchSize: Int

    // MARK: - Kernel Settings

    /// Whether to use fused kernels (e.g., L2+TopK) when available.
    /// Fused kernels reduce memory bandwidth but may have compatibility limits.
    public var useFusedKernels: Bool

    /// Whether to use dimension-optimized kernel variants.
    /// Optimized variants exist for common dimensions (384, 512, 768, 1536).
    public var useDimensionOptimizedKernels: Bool

    // MARK: - Resource Management

    /// Whether to pre-allocate GPU buffers for expected workload sizes.
    public var preallocateBuffers: Bool

    /// Size hint for buffer pre-allocation (number of vectors).
    public var bufferPreallocationSize: Int

    // MARK: - Behavioral Flags

    /// Force GPU acceleration regardless of heuristics.
    /// Use for benchmarking or when you know GPU is always beneficial.
    public var forceGPU: Bool

    /// Enable performance profiling and metrics collection.
    public var enableProfiling: Bool

    /// Log acceleration decisions for debugging.
    public var logDecisions: Bool

    // MARK: - Initialization

    public init(
        minimumCandidatesForGPU: Int = 500,
        minimumOperationsForGPU: Int = 50_000,
        maxBatchSize: Int = 10_000,
        useFusedKernels: Bool = true,
        useDimensionOptimizedKernels: Bool = true,
        preallocateBuffers: Bool = false,
        bufferPreallocationSize: Int = 10_000,
        forceGPU: Bool = false,
        enableProfiling: Bool = false,
        logDecisions: Bool = false
    ) {
        self.minimumCandidatesForGPU = minimumCandidatesForGPU
        self.minimumOperationsForGPU = minimumOperationsForGPU
        self.maxBatchSize = maxBatchSize
        self.useFusedKernels = useFusedKernels
        self.useDimensionOptimizedKernels = useDimensionOptimizedKernels
        self.preallocateBuffers = preallocateBuffers
        self.bufferPreallocationSize = bufferPreallocationSize
        self.forceGPU = forceGPU
        self.enableProfiling = enableProfiling
        self.logDecisions = logDecisions
    }

    // MARK: - Presets

    /// Default configuration with balanced settings.
    public static let `default` = IndexAccelerationConfiguration()

    /// Aggressive GPU usage - lower thresholds, always try GPU first.
    public static let aggressive = IndexAccelerationConfiguration(
        minimumCandidatesForGPU: 100,
        minimumOperationsForGPU: 10_000,
        maxBatchSize: 50_000,
        useFusedKernels: true,
        useDimensionOptimizedKernels: true,
        preallocateBuffers: true,
        bufferPreallocationSize: 50_000,
        forceGPU: false,
        enableProfiling: false,
        logDecisions: false
    )

    /// Conservative GPU usage - only for large workloads.
    public static let conservative = IndexAccelerationConfiguration(
        minimumCandidatesForGPU: 5_000,
        minimumOperationsForGPU: 500_000,
        maxBatchSize: 10_000,
        useFusedKernels: true,
        useDimensionOptimizedKernels: true,
        preallocateBuffers: false,
        bufferPreallocationSize: 10_000,
        forceGPU: false,
        enableProfiling: false,
        logDecisions: false
    )

    /// Benchmarking configuration - force GPU, enable profiling.
    public static let benchmarking = IndexAccelerationConfiguration(
        minimumCandidatesForGPU: 0,
        minimumOperationsForGPU: 0,
        maxBatchSize: 100_000,
        useFusedKernels: true,
        useDimensionOptimizedKernels: true,
        preallocateBuffers: true,
        bufferPreallocationSize: 100_000,
        forceGPU: true,
        enableProfiling: true,
        logDecisions: true
    )

    // MARK: - Migration Helpers

    /// Converts this configuration to GPUActivationThresholds for use with GPUDecisionEngine.
    ///
    /// Use this method to migrate from the deprecated threshold-based configuration
    /// to the adaptive GPUDecisionEngine pattern.
    ///
    /// ```swift
    /// let oldConfig = IndexAccelerationConfiguration.aggressive
    /// let thresholds = oldConfig.toGPUActivationThresholds()
    /// let engine = GPUDecisionEngine(thresholds: thresholds)
    /// ```
    public func toGPUActivationThresholds() -> GPUActivationThresholds {
        GPUActivationThresholds(
            minVectorsForGPU: minimumCandidatesForGPU,
            minCandidatesForGPU: minimumCandidatesForGPU,
            minKForGPU: 10,
            maxKForGPU: maxBatchSize / 10,
            minOperationsForGPU: minimumOperationsForGPU
        )
    }

    /// Creates a GPUDecisionEngine configured with thresholds derived from this configuration.
    ///
    /// This is a convenience method for migrating to the new adaptive routing pattern.
    ///
    /// ```swift
    /// let config = IndexAccelerationConfiguration.aggressive
    /// let engine = config.createDecisionEngine()
    /// ```
    public func createDecisionEngine() -> GPUDecisionEngine {
        GPUDecisionEngine(thresholds: toGPUActivationThresholds())
    }
}
