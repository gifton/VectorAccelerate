// VectorAccelerate: Batch Distance Operations
//
// Batch distance computations with GPU/CPU routing. Euclidean and cosine run Metal 4
// kernels when routing selects the GPU; dot product and Manhattan have no batch kernel
// yet and run Accelerate paths on every routing outcome (AUDIT-3 VA3-032).
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Batch distance computation engine using Metal 4
public actor BatchDistanceEngine {
    private let context: Metal4Context
    private let bufferPool: BufferPool

    // Performance thresholds (fallback when decisionEngine is nil)
    private let gpuThreshold = 1000  // Use GPU for batches > 1000 vectors
    private let simdThreshold = 100  // Use SIMD for batches > 100 vectors

    /// Optional decision engine for adaptive GPU/CPU routing
    private let decisionEngine: GPUDecisionEngine?

    /// Create with a Metal4Context and optional decision engine.
    ///
    /// - Parameters:
    ///   - context: The Metal4Context for GPU operations
    ///   - decisionEngine: Optional decision engine for adaptive routing (recommended for production)
    public init(context: Metal4Context, decisionEngine: GPUDecisionEngine? = nil) async throws {
        self.context = context
        self.bufferPool = context.bufferPool
        self.decisionEngine = decisionEngine
    }

    /// Create with default context and optional decision engine.
    ///
    /// - Parameter decisionEngine: Optional decision engine for adaptive routing
    public init(decisionEngine: GPUDecisionEngine? = nil) async throws {
        self.context = try await Metal4Context()
        self.bufferPool = context.bufferPool
        self.decisionEngine = decisionEngine
    }

    // MARK: - Input Validation

    /// Every public batch entry runs this guard before routing: each candidate must match
    /// the query's dimension exactly. The pre-audit guard checked only `candidates[0]`, so
    /// a ragged candidate at any later index reached the backends, where each answered with
    /// a different policy — +inf (euclidean), NaN (cosine), a zip-truncated partial product
    /// (dot), a VectorCore debug assert or release out-of-bounds read (manhattan). Pinned by
    /// BatchDistanceEngineTests.test_raggedCandidateRejectedUniformlyAcrossOperations.
    private func validateCandidateDimensions(query: [Float], candidates: [[Float]]) throws {
        for candidate in candidates where candidate.count != query.count {
            throw VectorError.dimensionMismatch(expected: query.count, actual: candidate.count)
        }
    }

    // MARK: - Batch Euclidean Distance

    /// Compute Euclidean distances between a query and multiple candidates.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - candidates: Candidate vectors; every candidate must match the query's
    ///     dimension (`dimensionMismatch` otherwise).
    ///   - useGPU: `true` forces the Metal path and `false` forces CPU; `nil` (default)
    ///     routes via the decision engine when present, else the batch-size threshold.
    public func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        try validateCandidateDimensions(query: query, candidates: candidates)

        let shouldUseGPU: Bool
        if let explicit = useGPU {
            shouldUseGPU = explicit
        } else if let engine = decisionEngine {
            shouldUseGPU = await engine.shouldUseGPU(
                operation: .l2Distance,
                vectorCount: candidates.count,
                candidateCount: candidates.count,
                k: 1,
                dimension: query.count
            )
        } else {
            shouldUseGPU = candidates.count >= gpuThreshold
        }

        if shouldUseGPU {
            return try await batchEuclideanDistanceGPU(query: query, candidates: candidates)
        }
        return batchEuclideanDistanceCPU(query: query, candidates: candidates)
    }

    private func batchEuclideanDistanceGPU(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        let dimension = query.count
        let candidateCount = candidates.count

        // Flatten candidates array
        let flatCandidates = candidates.flatMap { $0 }

        // Allocate buffers (BufferTokens auto-release when they go out of scope)
        let queryToken = try await bufferPool.getBuffer(with: query)
        let candidatesToken = try await bufferPool.getBuffer(with: flatCandidates)
        let resultToken = try await bufferPool.getBuffer(for: Float.self, count: candidateCount)

        // Get pipeline using Metal 4 shader compiler
        let pipeline = try await context.getPipeline(functionName: "batchEuclideanDistance")

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryToken.buffer, offset: 0, index: 0)
            encoder.setBuffer(candidatesToken.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultToken.buffer, offset: 0, index: 2)

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            // Dispatch threads
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(
                width: (candidateCount + 255) / 256,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        // Read results (token keeps buffer alive until we're done reading)
        return resultToken.copyData(as: Float.self, count: candidateCount)
    }

    private func batchEuclideanDistanceCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        // Route through Accelerate (vDSP) rather than a hand-rolled scalar loop. Swift can't
        // auto-vectorize the aliased float loop, so the previous version left NEON/AMX on the
        // table even for the small batches that land on this path. The former separate SIMD
        // leg (n >= simdThreshold) made this identical call, so all non-GPU routing collapses
        // here.
        AccelerateFallback.batchEuclideanDistance(query: query, candidates: candidates)
    }

    // MARK: - Batch Cosine Similarity

    /// Compute cosine similarities between a query and multiple candidates.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - candidates: Candidate vectors; every candidate must match the query's
    ///     dimension (`dimensionMismatch` otherwise).
    ///   - useGPU: `true` forces the Metal path and `false` forces CPU; `nil` (default)
    ///     routes via the decision engine when present, else the batch-size threshold.
    public func batchCosineSimilarity(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        try validateCandidateDimensions(query: query, candidates: candidates)

        let shouldUseGPU: Bool
        if let explicit = useGPU {
            shouldUseGPU = explicit
        } else if let engine = decisionEngine {
            shouldUseGPU = await engine.shouldUseGPU(
                operation: .cosineSimilarity,
                vectorCount: candidates.count,
                candidateCount: candidates.count,
                k: 1,
                dimension: query.count
            )
        } else {
            shouldUseGPU = candidates.count >= gpuThreshold
        }

        if shouldUseGPU {
            return try await batchCosineSimilarityGPU(query: query, candidates: candidates)
        }
        return batchCosineSimilarityCPU(query: query, candidates: candidates)
    }

    private func batchCosineSimilarityGPU(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        let dimension = query.count
        let candidateCount = candidates.count

        // Prepare data
        let flatCandidates = candidates.flatMap { $0 }

        // Allocate buffers (BufferTokens auto-release when they go out of scope)
        let queryToken = try await bufferPool.getBuffer(with: query)
        let candidatesToken = try await bufferPool.getBuffer(with: flatCandidates)
        let resultToken = try await bufferPool.getBuffer(for: Float.self, count: candidateCount)

        // Get pipeline using Metal 4 shader compiler
        let pipeline = try await context.getPipeline(functionName: "batchCosineSimilarity")

        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryToken.buffer, offset: 0, index: 0)
            encoder.setBuffer(candidatesToken.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultToken.buffer, offset: 0, index: 2)

            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(
                width: (candidateCount + 255) / 256,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        // Read results (token keeps buffer alive until we're done reading)
        return resultToken.copyData(as: Float.self, count: candidateCount)
    }

    private func batchCosineSimilarityCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        // AUDIT-3 VA3-033: this leg carried the pre-VA2-008 naive formula (NaN swallowed to
        // 0 by the `queryNorm > 0` gate, the `queryNorm * candidateNorm` product overflowing
        // and collapsing the similarity to 0, no [-1, 1] clamp) while the SIMD and GPU legs
        // run the shared rescue — one public API, different answers across the simdThreshold
        // boundary. All CPU routings now share AccelerateFallback's rescued core; the former
        // separate SIMD leg made this identical call and is collapsed here
        // (BatchDistanceEngineTests.test_batchCosineSimilarity_cpuLegsAgreeAcrossSimdBoundary
        // pins the across-the-boundary agreement).
        AccelerateFallback.batchCosineSimilarity(query: query, candidates: candidates)
    }

    // MARK: - Batch Dot Product

    /// Compute dot products between a query and multiple candidates.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - candidates: Candidate vectors; every candidate must match the query's
    ///     dimension (`dimensionMismatch` otherwise).
    ///   - useGPU: Accepted for API symmetry, but no batch dot-product kernel exists yet
    ///     (AUDIT-3 VA3-032), so every routing outcome — including an explicit `true` —
    ///     runs an Accelerate path.
    public func batchDotProduct(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        try validateCandidateDimensions(query: query, candidates: candidates)

        let shouldUseGPU: Bool
        if let explicit = useGPU {
            shouldUseGPU = explicit
        } else if let engine = decisionEngine {
            shouldUseGPU = await engine.shouldUseGPU(
                operation: .dotProduct,
                vectorCount: candidates.count,
                candidateCount: candidates.count,
                k: 1,
                dimension: query.count
            )
        } else {
            shouldUseGPU = candidates.count >= gpuThreshold
        }

        // AUDIT-3 VA3-032: the GPU branch dispatched "batchDotProduct" — a kernel that has
        // never existed in any library — so a GPU-routed call threw shaderNotFound where
        // callers expected values. The branch armed at gpuThreshold candidates with no
        // decision engine, and through the VA2-003 k-gate exemption with one attached. Until
        // a real batch kernel is wired up (`dot_product_kernel` is the natural candidate),
        // GPU-routed requests run the vDSP path: same results, no throw.
        if shouldUseGPU || candidates.count >= simdThreshold {
            return AccelerateFallback.batchDotProduct(query: query, candidates: candidates)
        }
        return batchDotProductCPU(query: query, candidates: candidates)
    }

    private func batchDotProductCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { candidate in
            zip(query, candidate).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    // MARK: - Batch Manhattan Distance

    /// Compute Manhattan (L1) distances between a query and multiple candidates.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - candidates: Candidate vectors; every candidate must match the query's
    ///     dimension (`dimensionMismatch` otherwise).
    ///   - useGPU: Accepted for API symmetry, but no batch Manhattan kernel exists yet
    ///     (AUDIT-3 VA3-032), so every routing outcome — including an explicit `true` —
    ///     runs an Accelerate path.
    public func batchManhattanDistance(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        try validateCandidateDimensions(query: query, candidates: candidates)

        let shouldUseGPU: Bool
        if let explicit = useGPU {
            shouldUseGPU = explicit
        } else if let engine = decisionEngine {
            shouldUseGPU = await engine.shouldUseGPU(
                operation: .manhattanDistance,
                vectorCount: candidates.count,
                candidateCount: candidates.count,
                k: 1,
                dimension: query.count
            )
        } else {
            shouldUseGPU = candidates.count >= gpuThreshold
        }

        // AUDIT-3 VA3-032: like batchDotProduct, the GPU branch dispatched
        // "batchManhattanDistance" — a kernel that exists in no library — and threw
        // shaderNotFound whenever routing selected GPU. GPU-routed requests run the
        // SIMD path until a real batch kernel exists.
        if shouldUseGPU || candidates.count >= simdThreshold {
            return try await batchManhattanDistanceSIMD(query: query, candidates: candidates)
        }
        return batchManhattanDistanceCPU(query: query, candidates: candidates)
    }

    private func batchManhattanDistanceSIMD(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        try candidates.map { try AccelerateFallback.manhattanDistance(query, $0) }
    }

    private func batchManhattanDistanceCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        // Use VectorCore's SIMD4-optimized ManhattanDistance for CPU path too
        let queryVector = DynamicVector(query)
        return candidates.map { candidate in
            let candidateVector = DynamicVector(candidate)
            return ManhattanDistance().distance(queryVector, candidateVector)
        }
    }

    // MARK: - K-Nearest Neighbors

    /// Find k-nearest neighbors using specified distance metric
    public func kNearestNeighbors(
        query: [Float],
        candidates: [[Float]],
        k: Int,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [(index: Int, distance: Float)] {
        let distances: [Float]

        switch metric {
        case .euclidean:
            distances = try await batchEuclideanDistance(query: query, candidates: candidates)
        case .cosine:
            let similarities = try await batchCosineSimilarity(query: query, candidates: candidates)
            distances = similarities.map { 1.0 - $0 } // Convert similarity to distance
        case .dotProduct:
            let products = try await batchDotProduct(query: query, candidates: candidates)
            distances = products.map { -$0 } // Negate for distance (higher dot product = closer)
        case .manhattan:
            distances = try await batchManhattanDistance(query: query, candidates: candidates)
        case .chebyshev:
            throw VectorError.unsupportedGPUOperation("Metric \(metric) not yet implemented for batch operations")
        }

        // Find k smallest distances
        let indexed = distances.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 < $1.1 }
        let topK = Array(sorted.prefix(k))

        return topK
    }

    // MARK: - Performance Metrics

    public struct PerformanceMetrics: Sendable {
        public let averageGPUTime: TimeInterval
        public let averageCPUTime: TimeInterval
        public let gpuSpeedup: Double
    }

    private var gpuTimes: [TimeInterval] = []
    private var cpuTimes: [TimeInterval] = []

    public func getPerformanceMetrics() -> PerformanceMetrics {
        let avgGPU = gpuTimes.isEmpty ? 0 : gpuTimes.reduce(0, +) / Double(gpuTimes.count)
        let avgCPU = cpuTimes.isEmpty ? 0 : cpuTimes.reduce(0, +) / Double(cpuTimes.count)
        let speedup = avgCPU > 0 ? avgCPU / avgGPU : 1.0

        return PerformanceMetrics(
            averageGPUTime: avgGPU,
            averageCPUTime: avgCPU,
            gpuSpeedup: speedup
        )
    }
}
