//
//  IndexAccelerationContext.swift
//  VectorIndexAcceleration
//
//  Shared context for index acceleration operations.
//  Manages Metal 4 resources and kernel instances for efficient reuse.
//

import Metal
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Acceleration Decision Types

/// Reason for acceleration decision.
public enum AccelerationReason: Sendable {
    /// GPU acceleration forced by configuration
    case forcedByConfiguration
    /// Dataset too small for GPU to provide benefit
    case datasetTooSmall
    /// Query count too low for GPU overhead to be worthwhile
    case queryCountTooLow
    /// GPU acceleration recommended based on workload analysis
    case gpuRecommended
}

/// Decision about whether to use GPU acceleration.
public struct AccelerationDecision: Sendable {
    /// Whether to use GPU for this operation
    public let useGPU: Bool
    /// Reason for the decision
    public let reason: AccelerationReason
    /// Estimated speedup factor (1.0 = same as CPU, >1.0 = faster on GPU)
    public let estimatedSpeedup: Float

    public init(useGPU: Bool, reason: AccelerationReason, estimatedSpeedup: Float) {
        self.useGPU = useGPU
        self.reason = reason
        self.estimatedSpeedup = estimatedSpeedup
    }
}

// MARK: - Index Acceleration Context

/// Shared context for GPU-accelerated index operations.
///
/// This actor manages Metal 4 resources and pre-compiled kernels for efficient
/// reuse across multiple index acceleration operations. It wraps `Metal4Context`
/// and adds index-specific kernel management.
///
/// ## Usage
/// ```swift
/// let context = try await IndexAccelerationContext()
///
/// // Use with multiple indices
/// let hnswAccel = HNSWIndexAccelerated(baseIndex: hnsw, context: context)
/// let ivfAccel = IVFIndexAccelerated(baseIndex: ivf, context: context)
/// ```
///
/// ## Thread Safety
/// This is an actor and all operations are thread-safe.
public actor IndexAccelerationContext {

    // MARK: - Properties

    /// Underlying Metal 4 context for GPU operations
    public let metal4Context: Metal4Context

    /// Configuration for acceleration decisions
    public let configuration: IndexAccelerationConfiguration

    // MARK: - Cached Kernels (lazily initialized)

    // Distance kernels (from VectorAccelerate)
    private var _l2DistanceKernel: L2DistanceKernel?
    private var _cosineKernel: CosineSimilarityKernel?
    private var _dotProductKernel: DotProductKernel?

    // Selection kernels (from VectorAccelerate)
    private var _topKKernel: TopKSelectionKernel?
    private var _fusedL2TopKKernel: FusedL2TopKKernel?

    // Index-specific kernels (will be added in later phases)
    // private var _hnswSearchKernel: HNSWSearchKernel?
    // private var _ivfSearchKernel: IVFSearchKernel?
    // private var _kmeansKernel: KMeansKernel?

    // MARK: - Initialization

    /// Create a new index acceleration context.
    ///
    /// - Parameters:
    ///   - metal4Context: Optional pre-created Metal4Context. If nil, creates a new one.
    ///   - configuration: Configuration for acceleration decisions.
    /// - Throws: `VectorError` if Metal 4 initialization fails.
    public init(
        metal4Context: Metal4Context? = nil,
        configuration: IndexAccelerationConfiguration = .default
    ) async throws {
        if let ctx = metal4Context {
            self.metal4Context = ctx
        } else {
            self.metal4Context = try await Metal4Context()
        }
        self.configuration = configuration
    }

    // MARK: - Kernel Access

    /// Get the L2 distance kernel, creating it if needed.
    public func l2DistanceKernel() async throws -> L2DistanceKernel {
        if let kernel = _l2DistanceKernel {
            return kernel
        }
        let kernel = try await L2DistanceKernel(context: metal4Context)
        _l2DistanceKernel = kernel
        return kernel
    }

    /// Get the cosine similarity kernel, creating it if needed.
    public func cosineSimilarityKernel() async throws -> CosineSimilarityKernel {
        if let kernel = _cosineKernel {
            return kernel
        }
        let kernel = try await CosineSimilarityKernel(context: metal4Context)
        _cosineKernel = kernel
        return kernel
    }

    /// Get the dot product kernel, creating it if needed.
    public func dotProductKernel() async throws -> DotProductKernel {
        if let kernel = _dotProductKernel {
            return kernel
        }
        let kernel = try await DotProductKernel(context: metal4Context)
        _dotProductKernel = kernel
        return kernel
    }

    /// Get the top-K selection kernel, creating it if needed.
    public func topKSelectionKernel() async throws -> TopKSelectionKernel {
        if let kernel = _topKKernel {
            return kernel
        }
        let kernel = try await TopKSelectionKernel(context: metal4Context)
        _topKKernel = kernel
        return kernel
    }

    /// Get the fused L2 + TopK kernel, creating it if needed.
    public func fusedL2TopKKernel() async throws -> FusedL2TopKKernel {
        if let kernel = _fusedL2TopKKernel {
            return kernel
        }
        let kernel = try await FusedL2TopKKernel(context: metal4Context)
        _fusedL2TopKKernel = kernel
        return kernel
    }

    // MARK: - Resource Management

    /// Warm up all commonly used kernels.
    ///
    /// Call this during initialization to avoid first-call latency.
    public func warmUp() async throws {
        _ = try await l2DistanceKernel()
        _ = try await topKSelectionKernel()

        if configuration.useFusedKernels {
            _ = try await fusedL2TopKKernel()
        }
    }

    /// Release cached kernel resources.
    ///
    /// Kernels will be recreated on next access.
    public func releaseKernels() {
        _l2DistanceKernel = nil
        _cosineKernel = nil
        _dotProductKernel = nil
        _topKKernel = nil
        _fusedL2TopKKernel = nil
    }

    // MARK: - Acceleration Decision

    /// Decide whether to use GPU acceleration for given parameters.
    ///
    /// - Parameters:
    ///   - queryCount: Number of queries to process
    ///   - candidateCount: Number of candidate vectors
    ///   - dimension: Vector dimension
    ///   - k: Number of results per query
    /// - Returns: Decision with recommendation and reasoning
    public func shouldAccelerate(
        queryCount: Int,
        candidateCount: Int,
        dimension: Int,
        k: Int
    ) -> AccelerationDecision {
        // Force GPU if configured
        if configuration.forceGPU {
            return AccelerationDecision(
                useGPU: true,
                reason: .forcedByConfiguration,
                estimatedSpeedup: 1.0
            )
        }

        // Check minimum thresholds
        let totalOperations = queryCount * candidateCount * dimension

        if candidateCount < configuration.minimumCandidatesForGPU {
            return AccelerationDecision(
                useGPU: false,
                reason: .datasetTooSmall,
                estimatedSpeedup: 0.5 // GPU would be slower
            )
        }

        if totalOperations < configuration.minimumOperationsForGPU {
            return AccelerationDecision(
                useGPU: false,
                reason: .queryCountTooLow,
                estimatedSpeedup: 0.8
            )
        }

        // Estimate speedup based on workload
        let estimatedSpeedup = estimateSpeedup(
            queryCount: queryCount,
            candidateCount: candidateCount,
            dimension: dimension
        )

        return AccelerationDecision(
            useGPU: true,
            reason: .gpuRecommended,
            estimatedSpeedup: estimatedSpeedup
        )
    }

    private func estimateSpeedup(
        queryCount: Int,
        candidateCount: Int,
        dimension: Int
    ) -> Float {
        // Simple heuristic - can be refined with actual benchmarks
        let workload = Float(queryCount * candidateCount * dimension)

        // GPU excels at larger workloads
        if workload > 100_000_000 {
            return 10.0
        } else if workload > 10_000_000 {
            return 5.0
        } else if workload > 1_000_000 {
            return 2.0
        } else {
            return 1.2
        }
    }
}

// MARK: - Convenience Extensions

public extension IndexAccelerationContext {

    /// Get appropriate distance kernel for the given metric.
    func distanceKernel(for metric: SupportedDistanceMetric) async throws -> any Metal4Kernel {
        switch metric {
        case .euclidean:
            return try await l2DistanceKernel()
        case .cosine:
            return try await cosineSimilarityKernel()
        case .dotProduct:
            return try await dotProductKernel()
        case .manhattan, .chebyshev:
            // These metrics will use L2 as fallback for now
            // Phase 4 will add dedicated kernels
            return try await l2DistanceKernel()
        }
    }
}
