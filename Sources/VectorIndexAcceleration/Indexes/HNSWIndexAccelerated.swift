//
//  HNSWIndexAccelerated.swift
//  VectorIndexAcceleration
//
//  GPU-accelerated wrapper for HNSW indices.
//
//  Provides Metal 4 GPU acceleration for HNSW search operations.
//  The graph traversal remains CPU-based (inherently sequential),
//  while distance computations can be offloaded to GPU for large batches.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - HNSW Index Accelerated

/// GPU-accelerated HNSW index wrapper.
///
/// Wraps a `HNSWIndex` and provides Metal 4 GPU acceleration for search operations.
/// The underlying graph structure remains on CPU (graph traversal is inherently sequential),
/// while large batch operations can benefit from GPU acceleration.
///
/// ## Acceleration Strategy
///
/// HNSW's graph traversal algorithm is sequential by nature - each step depends
/// on the previous step's results. However, GPU acceleration can help in these cases:
///
/// 1. **Large Batch Queries**: When processing many queries, GPU can parallelize
///    the candidate reranking step across all queries.
///
/// 2. **High efSearch**: When efSearch is very large (1000+), GPU can accelerate
///    the final distance recomputation for candidate selection.
///
/// 3. **Hybrid Search**: For scenarios where HNSW is used as a first stage
///    followed by exact reranking on GPU.
///
/// ## Implementation Notes
///
/// The base HNSWIndex already uses:
/// - SIMD-optimized distance computation (via HNSWTraversal kernel)
/// - CSR graph format for cache-friendly memory access
/// - Parallel batch search via Swift TaskGroup
///
/// GPU acceleration complements rather than replaces these optimizations.
///
/// ## Usage
/// ```swift
/// let baseIndex = HNSWIndex(dimension: 768, metric: .euclidean)
/// // ... add vectors to baseIndex ...
///
/// let accelerated = HNSWIndexAccelerated(baseIndex: baseIndex)
/// try await accelerated.prepareForGPU()
///
/// // GPU-accelerated batch search (most benefit)
/// let results = try await accelerated.batchSearchGPU(queries: queryVectors, k: 10)
/// ```
public actor HNSWIndexAccelerated {

    // MARK: - Properties

    /// The underlying HNSW index
    public let baseIndex: HNSWIndex

    /// GPU acceleration context
    public private(set) var gpuContext: Metal4Context?

    /// Shared acceleration context for kernel reuse
    private var accelerationContext: IndexAccelerationContext?

    /// Configuration for acceleration decisions
    private let configuration: IndexAccelerationConfiguration

    // MARK: - GPU Acceleration Settings

    /// Minimum number of queries to consider GPU batch search
    private let minBatchQueriesForGPU: Int = 10

    /// Minimum efSearch to consider GPU candidate reranking
    private let minEfSearchForGPU: Int = 500

    // MARK: - Computed Properties

    /// Vector dimension (delegated to base index)
    public var dimension: Int {
        get async { await baseIndex.dimension }
    }

    /// Number of vectors (delegated to base index)
    public var count: Int {
        get async { await baseIndex.count }
    }

    /// Distance metric (delegated to base index)
    public var metric: SupportedDistanceMetric {
        get async { await baseIndex.metric }
    }

    /// Whether GPU acceleration is currently active
    public var isGPUActive: Bool {
        gpuContext != nil && accelerationContext != nil
    }

    // MARK: - Initialization

    /// Create an accelerated wrapper for an existing HNSW index.
    ///
    /// - Parameters:
    ///   - baseIndex: The HNSW index to accelerate
    ///   - configuration: Configuration for acceleration behavior
    public init(
        baseIndex: HNSWIndex,
        configuration: IndexAccelerationConfiguration = .default
    ) {
        self.baseIndex = baseIndex
        self.configuration = configuration
    }

    /// Create an accelerated wrapper with a pre-created context.
    ///
    /// - Parameters:
    ///   - baseIndex: The HNSW index to accelerate
    ///   - context: Pre-created acceleration context for kernel sharing
    public init(
        baseIndex: HNSWIndex,
        context: IndexAccelerationContext
    ) async {
        self.baseIndex = baseIndex
        self.configuration = context.configuration
        self.accelerationContext = context
        self.gpuContext = context.metal4Context
    }

    // MARK: - GPU Lifecycle

    /// Initialize GPU resources for acceleration.
    public func prepareForGPU(context: Metal4Context? = nil) async throws {
        if let ctx = context {
            self.gpuContext = ctx
            self.accelerationContext = try await IndexAccelerationContext(
                metal4Context: ctx,
                configuration: configuration
            )
        } else if self.gpuContext == nil {
            let newContext = try await IndexAccelerationContext(configuration: configuration)
            self.gpuContext = newContext.metal4Context
            self.accelerationContext = newContext
        }

        try await accelerationContext?.warmUp()
    }

    /// Release GPU resources.
    public func releaseGPUResources() async {
        await accelerationContext?.releaseKernels()
        accelerationContext = nil
        gpuContext = nil
    }

    // MARK: - Search Operations

    /// GPU-accelerated k-NN search.
    ///
    /// For single queries, delegates to the highly-optimized CPU implementation
    /// since HNSW traversal is inherently sequential and the base index already
    /// uses SIMD-optimized distance computation.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of nearest neighbors
    ///   - filter: Optional metadata filter
    /// - Returns: Search results sorted by distance
    public func searchGPU(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        // Ensure GPU is initialized
        if gpuContext == nil || accelerationContext == nil {
            try await prepareForGPU()
        }

        guard accelerationContext != nil else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "searchGPU")
        }

        // For single queries, the CPU implementation is already highly optimized
        // with SIMD-based distance computation and CSR graph format.
        // GPU overhead would likely outweigh benefits for single queries.
        return try await baseIndex.search(query: query, k: k, filter: filter)
    }

    /// GPU-accelerated batch k-NN search.
    ///
    /// For large batches, this can provide speedups by parallelizing across queries.
    /// The base implementation already uses TaskGroup for parallelization, so GPU
    /// benefits are most significant for very large batches (100+ queries) or
    /// when the GPU is already warm from other operations.
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - k: Number of nearest neighbors per query
    ///   - filter: Optional metadata filter
    /// - Returns: Search results for each query
    public func batchSearchGPU(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        guard !queries.isEmpty else { return [] }

        if gpuContext == nil || accelerationContext == nil {
            try await prepareForGPU()
        }

        guard let context = accelerationContext else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "batchSearchGPU")
        }

        let dimension = await baseIndex.dimension
        let vectorCount = await baseIndex.count

        // Determine if GPU acceleration is worthwhile
        let shouldUseGPU = configuration.forceGPU ||
            (queries.count >= minBatchQueriesForGPU &&
             vectorCount >= configuration.minimumCandidatesForGPU)

        if !shouldUseGPU {
            // Fall back to CPU batch search (already parallelized via TaskGroup)
            return try await baseIndex.batchSearch(queries: queries, k: k, filter: filter)
        }

        // For GPU-accelerated batch search, we use a hybrid approach:
        // 1. Get candidates from HNSW graph traversal (CPU)
        // 2. Use GPU for final reranking if candidate set is large
        return try await executeBatchSearchWithGPUReranking(
            queries: queries,
            k: k,
            filter: filter,
            dimension: dimension,
            context: context
        )
    }

    // MARK: - Private GPU Execution

    private func executeBatchSearchWithGPUReranking(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?,
        dimension: Int,
        context: IndexAccelerationContext
    ) async throws -> [[VectorIndex.SearchResult]] {
        // For HNSW, the most practical approach is:
        // 1. Use the base index's optimized CPU search (which already parallelizes)
        // 2. GPU doesn't help much because graph traversal is sequential
        //
        // The main GPU benefit would be if we had a large candidate pool to rerank,
        // but HNSW's efSearch already limits this to a manageable size.
        //
        // For now, delegate to the CPU implementation which is already highly optimized.
        // Future enhancement: GPU-accelerated exhaustive search on candidate sets
        // when efSearch is very large (1000+).

        // Get candidates for each query using parallel graph traversal
        let allCandidates = try await baseIndex.getBatchCandidates(
            queries: queries,
            k: k,
            filter: filter
        )

        // Check if GPU reranking would be beneficial
        let totalCandidates = allCandidates.reduce(0) { $0 + $1.vectorCount }
        let avgCandidatesPerQuery = totalCandidates / max(queries.count, 1)

        // If candidate sets are small, CPU is faster due to GPU overhead
        if avgCandidatesPerQuery < 100 {
            // Just use CPU finalization
            return try await withThrowingTaskGroup(of: (Int, [VectorIndex.SearchResult]).self) { group in
                for (index, (query, candidates)) in zip(queries, allCandidates).enumerated() {
                    group.addTask {
                        let results = try await self.cpuRerank(
                            query: query,
                            candidates: candidates,
                            k: k,
                            filter: filter
                        )
                        return (index, results)
                    }
                }

                var results = [[VectorIndex.SearchResult]](repeating: [], count: queries.count)
                for try await (index, result) in group {
                    results[index] = result
                }
                return results
            }
        }

        // For larger candidate sets, use GPU
        return try await gpuRerankBatch(
            queries: queries,
            candidates: allCandidates,
            k: k,
            filter: filter,
            dimension: dimension,
            context: context
        )
    }

    private func cpuRerank(
        query: [Float],
        candidates: AccelerationCandidates,
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [VectorIndex.SearchResult] {
        // Compute distances for candidates and apply filter
        var results: [(VectorID, Float)] = []
        results.reserveCapacity(candidates.vectorCount)

        let d = candidates.dimension
        let metric = await baseIndex.metric

        // Compute distances using vDSP for efficiency
        for i in 0..<candidates.vectorCount {
            // Apply filter if provided
            if let filter = filter {
                if !filter(candidates.metadata[i]) { continue }
            }

            // Compute distance using SIMD
            let start = i * d
            var dist: Float = 0

            switch metric {
            case .euclidean:
                // L2 squared distance using vDSP
                var sum: Float = 0
                for j in 0..<d {
                    let diff = query[j] - candidates.vectorStorage[start + j]
                    sum += diff * diff
                }
                dist = sqrt(sum)

            case .dotProduct:
                // Dot product (negate for distance)
                var sum: Float = 0
                for j in 0..<d {
                    sum += query[j] * candidates.vectorStorage[start + j]
                }
                dist = -sum  // Negate so lower is better

            case .cosine:
                // Cosine distance = 1 - cosine similarity
                var dot: Float = 0
                var normQ: Float = 0
                var normV: Float = 0
                for j in 0..<d {
                    let q = query[j]
                    let v = candidates.vectorStorage[start + j]
                    dot += q * v
                    normQ += q * q
                    normV += v * v
                }
                let denom = sqrt(normQ * normV)
                let sim = denom > 0 ? dot / denom : 0
                dist = 1 - sim

            case .manhattan:
                var sum: Float = 0
                for j in 0..<d {
                    sum += abs(query[j] - candidates.vectorStorage[start + j])
                }
                dist = sum

            case .chebyshev:
                var maxDiff: Float = 0
                for j in 0..<d {
                    maxDiff = max(maxDiff, abs(query[j] - candidates.vectorStorage[start + j]))
                }
                dist = maxDiff
            }

            results.append((candidates.ids[i], dist))
        }

        // Sort and take top-k
        results.sort { $0.1 < $1.1 }
        return results.prefix(k).map { VectorIndex.SearchResult(id: $0.0, score: $0.1) }
    }

    private func gpuRerankBatch(
        queries: [[Float]],
        candidates: [AccelerationCandidates],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?,
        dimension: Int,
        context: IndexAccelerationContext
    ) async throws -> [[VectorIndex.SearchResult]] {
        guard let device = gpuContext?.device.rawDevice else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "gpuRerankBatch")
        }

        // For simplicity, fall back to CPU if filter is provided
        // (GPU can't efficiently evaluate arbitrary closures)
        if filter != nil {
            return try await withThrowingTaskGroup(of: (Int, [VectorIndex.SearchResult]).self) { group in
                for (index, (query, cand)) in zip(queries, candidates).enumerated() {
                    group.addTask {
                        let results = try await self.cpuRerank(
                            query: query,
                            candidates: cand,
                            k: k,
                            filter: filter
                        )
                        return (index, results)
                    }
                }

                var results = [[VectorIndex.SearchResult]](repeating: [], count: queries.count)
                for try await (index, result) in group {
                    results[index] = result
                }
                return results
            }
        }

        // GPU reranking for each query's candidate set
        let fusedKernel = try await context.fusedL2TopKKernel()

        return try await withThrowingTaskGroup(of: (Int, [VectorIndex.SearchResult]).self) { group in
            for (index, (query, cand)) in zip(queries, candidates).enumerated() {
                group.addTask {
                    guard cand.vectorCount > 0 else { return (index, []) }

                    // Create query buffer
                    let queryBufferSize = query.count * MemoryLayout<Float>.size
                    guard let queryBuffer = device.makeBuffer(
                        bytes: query,
                        length: queryBufferSize,
                        options: .storageModeShared
                    ) else {
                        throw VectorError.bufferAllocationFailed(size: queryBufferSize)
                    }

                    // Create candidate buffer
                    let candBufferSize = cand.vectorStorage.count * MemoryLayout<Float>.size
                    guard let candBuffer = cand.vectorStorage.withUnsafeBufferPointer({ ptr in
                        device.makeBuffer(
                            bytes: ptr.baseAddress!,
                            length: candBufferSize,
                            options: .storageModeShared
                        )
                    }) else {
                        throw VectorError.bufferAllocationFailed(size: candBufferSize)
                    }

                    let params = FusedL2TopKParameters(
                        numQueries: 1,
                        numDataset: cand.vectorCount,
                        dimension: dimension,
                        k: min(k, FusedL2TopKParameters.maxK)
                    )

                    let result = try await fusedKernel.execute(
                        queries: queryBuffer,
                        dataset: candBuffer,
                        parameters: params,
                        config: Metal4FusedL2Config(includeDistances: true)
                    )

                    // Map results back to VectorIDs
                    let gpuResults = result.results(for: 0)
                    let searchResults = gpuResults.compactMap { (idx, dist) -> VectorIndex.SearchResult? in
                        guard idx < cand.ids.count else { return nil }
                        return VectorIndex.SearchResult(id: cand.ids[idx], score: dist)
                    }

                    return (index, searchResults)
                }
            }

            var results = [[VectorIndex.SearchResult]](repeating: [], count: queries.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }

    // MARK: - Delegated Operations

    /// Insert a vector (delegated to base index)
    public func insert(id: VectorID, vector: [Float], metadata: [String: String]? = nil) async throws {
        try await baseIndex.insert(id: id, vector: vector, metadata: metadata)
    }

    /// Remove a vector (delegated to base index)
    public func remove(id: VectorID) async throws {
        try await baseIndex.remove(id: id)
    }

    /// Batch insert (delegated to base index)
    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws {
        try await baseIndex.batchInsert(items)
    }

    /// Search (auto-selects GPU or CPU based on workload)
    public func search(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        // For single queries, CPU is optimal due to sequential graph traversal
        return try await baseIndex.search(query: query, k: k, filter: filter)
    }

    /// Batch search (auto-selects GPU or CPU based on workload)
    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        // Use GPU path if GPU is active and batch is large enough
        if isGPUActive && queries.count >= minBatchQueriesForGPU {
            return try await batchSearchGPU(queries: queries, k: k, filter: filter)
        }
        return try await baseIndex.batchSearch(queries: queries, k: k, filter: filter)
    }

    /// Check if vector exists
    public func contains(id: VectorID) async -> Bool {
        await baseIndex.contains(id: id)
    }

    /// Clear the index
    public func clear() async {
        await baseIndex.clear()
    }

    /// Get index statistics
    public func statistics() async -> VectorIndex.IndexStats {
        await baseIndex.statistics()
    }

    /// Optimize the index
    public func optimize() async throws {
        try await baseIndex.optimize()
    }

    /// Compact the index
    public func compact() async throws {
        try await baseIndex.compact()
    }
}
