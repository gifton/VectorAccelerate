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
/// Controls how GPU acceleration is applied to index searches. For adaptive
/// routing decisions (when to prefer GPU over CPU), use ``GPUDecisionEngine``.
///
/// ## Custom Configuration
/// ```swift
/// let config = IndexAccelerationConfiguration(
///     useFusedKernels: true,
///     enableProfiling: true
/// )
/// ```
public struct IndexAccelerationConfiguration: Sendable {

    // MARK: - Batch Settings

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
        maxBatchSize: Int = 10_000,
        useFusedKernels: Bool = true,
        useDimensionOptimizedKernels: Bool = true,
        preallocateBuffers: Bool = false,
        bufferPreallocationSize: Int = 10_000,
        forceGPU: Bool = false,
        enableProfiling: Bool = false,
        logDecisions: Bool = false
    ) {
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
}
