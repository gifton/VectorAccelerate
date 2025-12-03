//
//  KernelProtocol.swift
//  VectorAccelerate
//
//  Protocol definitions for Metal 4 kernels with ArgumentTable support.
//
//  Phase 5: Kernel Migrations
//
//  Design Goals:
//  - Support both standalone execution and fusion into shared encoders
//  - Efficient ArgumentTable usage via pooling
//  - Automatic residency management
//  - Type-safe parameter passing
//  - Dimension-specific optimizations where applicable

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Metal 4 Kernel Protocol

/// Protocol for Metal 4 compute kernels.
///
/// Metal 4 kernels differ from Metal 3 kernels in several ways:
/// - Use ArgumentTable for efficient buffer binding
/// - Support encoder reuse for kernel fusion
/// - Integrate with ResidencyManager for explicit resource management
/// - Provide both encode-only and full execution APIs
///
/// ## Usage Patterns
///
/// ### Standalone Execution
/// ```swift
/// let kernel = try await L2DistanceKernel(context: context)
/// let distances = try await kernel.execute(
///     queries: queryBuffer,
///     database: databaseBuffer,
///     parameters: params
/// )
/// ```
///
/// ### Fused Execution (with other kernels)
/// ```swift
/// try await context.executeAndWait { commandBuffer, encoder in
///     // Encode distance computation
///     try distanceKernel.encode(
///         into: encoder,
///         queries: queryBuffer,
///         database: databaseBuffer,
///         distances: distanceBuffer,
///         parameters: params
///     )
///
///     // Add barrier before next operation
///     encoder.memoryBarrier(scope: .buffers)
///
///     // Encode top-k selection on same encoder
///     try topKKernel.encode(
///         into: encoder,
///         distances: distanceBuffer,
///         ...
///     )
/// }
/// ```
public protocol Metal4Kernel: Sendable {
    /// The Metal 4 context this kernel operates with
    var context: Metal4Context { get }

    /// Human-readable name for debugging and profiling
    var name: String { get }

    /// Precompile any pipelines needed by this kernel
    func warmUp() async throws
}

// MARK: - Encoding Result

/// Result of encoding a kernel operation.
///
/// Contains information about the dispatch for debugging and profiling.
public struct Metal4EncodingResult: Sendable {
    /// The pipeline that was used
    public let pipelineName: String

    /// Thread group configuration used
    public let threadgroups: MTLSize

    /// Threads per threadgroup
    public let threadsPerThreadgroup: MTLSize

    /// Total thread count dispatched
    public var totalThreads: Int {
        threadgroups.width * threadgroups.height * threadgroups.depth *
        threadsPerThreadgroup.width * threadsPerThreadgroup.height * threadsPerThreadgroup.depth
    }
}

// MARK: - Distance Kernel Protocol

/// Protocol for distance computation kernels.
///
/// Distance kernels compute pairwise distances between query and database vectors.
/// The output is a matrix of distances with shape [numQueries, numDatabase].
public protocol Metal4DistanceKernel: Metal4Kernel {
    /// Parameters type for this distance kernel
    associatedtype Parameters: Sendable

    /// Encode distance computation into an existing encoder.
    ///
    /// This method does NOT create or end the encoder - it only adds dispatch commands.
    /// Use this for fusing multiple operations into a single command buffer.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into
    ///   - queries: Buffer containing query vectors [N, D]
    ///   - database: Buffer containing database vectors [M, D]
    ///   - distances: Output buffer for distances [N, M]
    ///   - parameters: Kernel execution parameters
    /// - Returns: Information about the encoding for debugging
    /// - Throws: `VectorError` if encoding fails
    @discardableResult
    func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: Parameters
    ) throws -> Metal4EncodingResult

    /// Execute distance computation as a standalone operation.
    ///
    /// Creates a command buffer, encodes the operation, submits, and waits.
    ///
    /// - Parameters:
    ///   - queries: Buffer containing query vectors [N, D]
    ///   - database: Buffer containing database vectors [M, D]
    ///   - parameters: Kernel execution parameters
    /// - Returns: Buffer containing computed distances
    /// - Throws: `VectorError` if execution fails
    func execute(
        queries: any MTLBuffer,
        database: any MTLBuffer,
        parameters: Parameters
    ) async throws -> any MTLBuffer
}

// MARK: - Dimension-Optimized Kernel

/// Protocol for kernels with dimension-specific optimizations.
///
/// Some kernels (like L2 distance) have hand-tuned implementations for
/// common embedding dimensions (384, 512, 768, 1536). This protocol
/// provides introspection of available optimizations.
public protocol DimensionOptimizedKernel: Metal4Kernel {
    /// Dimensions that have specialized optimized pipelines
    var optimizedDimensions: [Int] { get }

    /// Check if a specific dimension has an optimized pipeline
    func hasOptimizedPipeline(for dimension: Int) -> Bool
}

extension DimensionOptimizedKernel {
    public func hasOptimizedPipeline(for dimension: Int) -> Bool {
        optimizedDimensions.contains(dimension)
    }
}

// MARK: - Fusible Kernel

/// Protocol for kernels that support fusion with other operations.
///
/// Fusible kernels can share a command encoder with other operations,
/// reducing submission overhead and enabling better GPU utilization.
public protocol FusibleKernel: Metal4Kernel {
    /// Types of kernels this can be fused with
    var fusibleWith: [String] { get }

    /// Whether a barrier is required before this kernel's output is read
    var requiresBarrierAfter: Bool { get }
}

// MARK: - Thread Configuration

/// Thread group configuration for Metal 4 kernels.
///
/// Provides optimal thread configuration based on pipeline characteristics
/// and workload size.
public struct Metal4ThreadConfiguration: Sendable {
    public let threadgroups: MTLSize
    public let threadsPerThreadgroup: MTLSize

    /// Create a 2D thread configuration optimized for distance kernels.
    ///
    /// Distance kernels dispatch over (numQueries, numDatabase) pairs.
    /// This configuration optimizes for:
    /// - Square-ish thread groups for cache locality
    /// - Respecting device limits
    /// - Minimizing wasted threads at boundaries
    ///
    /// - Parameters:
    ///   - numQueries: Number of query vectors (grid width)
    ///   - numDatabase: Number of database vectors (grid height)
    ///   - pipeline: The pipeline to dispatch with (for device limits)
    public static func forDistanceKernel(
        numQueries: Int,
        numDatabase: Int,
        pipeline: any MTLComputePipelineState
    ) -> Metal4ThreadConfiguration {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup

        // Aim for square-ish thread groups for better 2D cache behavior
        let threadsPerSide = Int(sqrt(Double(min(maxThreads, 256))))

        // Clamp to actual work size to avoid waste
        let threadWidth = min(threadsPerSide, numQueries)
        let threadHeight = min(threadsPerSide, numDatabase)

        let threadsPerThreadgroup = MTLSize(
            width: max(1, threadWidth),
            height: max(1, threadHeight),
            depth: 1
        )

        let threadgroups = MTLSize(
            width: (numQueries + threadWidth - 1) / max(1, threadWidth),
            height: (numDatabase + threadHeight - 1) / max(1, threadHeight),
            depth: 1
        )

        return Metal4ThreadConfiguration(
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    /// Create a 1D thread configuration.
    ///
    /// - Parameters:
    ///   - count: Total number of elements to process
    ///   - pipeline: The pipeline to dispatch with
    public static func linear(
        count: Int,
        pipeline: any MTLComputePipelineState
    ) -> Metal4ThreadConfiguration {
        let maxThreadsPerGroup = min(pipeline.maxTotalThreadsPerThreadgroup, 256)

        let threadsPerThreadgroup = MTLSize(
            width: min(maxThreadsPerGroup, count),
            height: 1,
            depth: 1
        )

        let threadgroups = MTLSize(
            width: (count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        return Metal4ThreadConfiguration(
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }
}

// MARK: - Kernel Statistics

/// Statistics for Metal 4 kernel execution.
public struct Metal4KernelStatistics: Sendable {
    public let encodeCount: Int
    public let executeCount: Int
    public let totalComputeTime: TimeInterval

    public var averageComputeTime: TimeInterval {
        executeCount > 0 ? totalComputeTime / Double(executeCount) : 0
    }

    public init(encodeCount: Int, executeCount: Int, totalComputeTime: TimeInterval) {
        self.encodeCount = encodeCount
        self.executeCount = executeCount
        self.totalComputeTime = totalComputeTime
    }
}

// MARK: - Parameter Buffer Helper

/// Helper for creating parameter buffers efficiently.
///
/// Small parameters (< 4KB) can use setBytes, but larger or frequently
/// reused parameters benefit from a dedicated buffer.
public enum ParameterBinding {
    /// Bind parameters inline using setBytes (for small, one-off parameters)
    case inline

    /// Bind parameters via buffer (for larger or reused parameters)
    case buffer(any MTLBuffer)

    /// Apply this binding to an encoder at the specified index.
    public func apply<T>(
        _ value: inout T,
        to encoder: any MTLComputeCommandEncoder,
        at index: Int
    ) {
        switch self {
        case .inline:
            encoder.setBytes(&value, length: MemoryLayout<T>.size, index: index)
        case .buffer(let buffer):
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
    }
}
