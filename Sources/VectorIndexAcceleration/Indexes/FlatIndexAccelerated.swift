//
//  FlatIndexAccelerated.swift
//  VectorIndexAcceleration
//
//  GPU-accelerated wrapper for Flat (brute-force) indices.
//
//  Provides Metal 4 GPU acceleration for exhaustive k-NN search operations.
//  Uses FusedL2TopKKernel for single-pass distance computation and selection.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Flat Index Accelerated

/// GPU-accelerated Flat (brute-force) index wrapper.
///
/// Wraps a `FlatIndex` and provides Metal 4 GPU acceleration for search operations.
/// Flat indices benefit greatly from GPU acceleration due to the embarrassingly
/// parallel nature of brute-force distance computation.
///
/// ## GPU Acceleration Strategy
/// Uses `FusedL2TopKKernel` for single-pass distance computation and top-k selection,
/// avoiding the memory overhead of materializing the full distance matrix.
///
/// ## Usage
/// ```swift
/// let baseIndex = FlatIndex(dimension: 768, metric: .euclidean)
/// // ... insert vectors ...
///
/// let accelerated = FlatIndexAccelerated(baseIndex: baseIndex)
/// try await accelerated.prepareForGPU()
///
/// // GPU-accelerated search
/// let results = try await accelerated.searchGPU(query: queryVector, k: 10)
/// ```
public actor FlatIndexAccelerated {

    // MARK: - Properties

    /// The underlying Flat index
    public let baseIndex: FlatIndex

    /// GPU acceleration context
    public private(set) var gpuContext: Metal4Context?

    /// Shared acceleration context for kernel reuse
    private var accelerationContext: IndexAccelerationContext?

    /// Configuration for acceleration decisions
    private let configuration: IndexAccelerationConfiguration

    // MARK: - Cached GPU Data

    /// Cached GPU buffer containing all vectors
    private var vectorBuffer: (any MTLBuffer)?

    /// Mapping from buffer index to VectorID
    private var indexToID: [VectorID] = []

    /// Mapping from buffer index to metadata
    private var indexToMetadata: [[String: String]?] = []

    /// Whether the GPU cache needs refresh
    private var cacheNeedsRefresh: Bool = true

    /// Cached vector count (for detecting changes)
    private var cachedVectorCount: Int = 0

    // MARK: - Computed Properties

    public var dimension: Int {
        get async { await baseIndex.dimension }
    }

    public var count: Int {
        get async { await baseIndex.count }
    }

    public var metric: SupportedDistanceMetric {
        get async { await baseIndex.metric }
    }

    public var isGPUActive: Bool {
        gpuContext != nil && accelerationContext != nil
    }

    // MARK: - Initialization

    public init(
        baseIndex: FlatIndex,
        configuration: IndexAccelerationConfiguration = .default
    ) {
        self.baseIndex = baseIndex
        self.configuration = configuration
    }

    public init(
        baseIndex: FlatIndex,
        context: IndexAccelerationContext
    ) async {
        self.baseIndex = baseIndex
        self.configuration = context.configuration
        self.accelerationContext = context
        self.gpuContext = context.metal4Context
    }

    // MARK: - GPU Lifecycle

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

        // Refresh GPU cache
        try await refreshGPUCache()
    }

    public func releaseGPUResources() async {
        await accelerationContext?.releaseKernels()
        accelerationContext = nil
        gpuContext = nil
        vectorBuffer = nil
        indexToID = []
        indexToMetadata = []
        cacheNeedsRefresh = true
    }

    /// Mark GPU cache as needing refresh (call after index modifications).
    public func invalidateGPUCache() {
        cacheNeedsRefresh = true
    }

    /// Refresh GPU cache from base index.
    private func refreshGPUCache() async throws {
        guard let context = gpuContext else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "refreshGPUCache")
        }

        let dimension = await baseIndex.dimension
        let candidates = try await baseIndex.getCandidates(
            query: Array(repeating: 0, count: dimension),
            k: Int.max,
            filter: nil
        )

        guard candidates.vectorCount > 0 else {
            // Empty index
            vectorBuffer = nil
            indexToID = []
            indexToMetadata = []
            cachedVectorCount = 0
            cacheNeedsRefresh = false
            return
        }

        // Create GPU buffer from contiguous vector storage
        let device = context.device.rawDevice
        let bufferSize = candidates.vectorStorage.count * MemoryLayout<Float>.size

        guard let buffer = candidates.vectorStorage.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: bufferSize,
                options: .storageModeShared
            )
        }) else {
            throw VectorError.bufferAllocationFailed(size: bufferSize)
        }
        buffer.label = "FlatIndexAccelerated.vectors"

        self.vectorBuffer = buffer
        self.indexToID = candidates.ids
        self.indexToMetadata = candidates.metadata
        self.cachedVectorCount = candidates.vectorCount
        self.cacheNeedsRefresh = false
    }

    // MARK: - Search Operations

    /// GPU-accelerated k-NN search.
    ///
    /// Uses FusedL2TopKKernel for single-pass distance computation and selection.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of nearest neighbors to find
    ///   - filter: Optional metadata filter (applied post-GPU)
    /// - Returns: Search results sorted by distance
    public func searchGPU(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        // Ensure GPU is ready
        if gpuContext == nil || accelerationContext == nil {
            try await prepareForGPU()
        }

        guard let context = accelerationContext else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "searchGPU")
        }

        // Check if cache needs refresh
        let currentCount = await baseIndex.count
        if cacheNeedsRefresh || currentCount != cachedVectorCount {
            try await refreshGPUCache()
        }

        // Handle empty index
        guard let datasetBuffer = vectorBuffer, cachedVectorCount > 0 else {
            return []
        }

        let dimension = await baseIndex.dimension
        let metric = await baseIndex.metric

        // Check if GPU is worthwhile
        let decision = await context.shouldAccelerate(
            queryCount: 1,
            candidateCount: cachedVectorCount,
            dimension: dimension,
            k: k
        )

        if !decision.useGPU && !configuration.forceGPU {
            // Fall back to CPU for small datasets
            return try await baseIndex.search(query: query, k: k, filter: filter)
        }

        // Execute GPU search
        let results = try await executeGPUSearch(
            queries: [query],
            dataset: datasetBuffer,
            numDataset: cachedVectorCount,
            dimension: dimension,
            k: filter != nil ? min(k * 2, cachedVectorCount) : k, // Over-fetch if filtering
            metric: metric,
            context: context
        )

        // Apply filter and map to VectorIDs
        return applyFilterAndMapResults(
            gpuResults: results[0],
            k: k,
            filter: filter,
            metric: metric
        )
    }

    /// GPU-accelerated batch k-NN search.
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - k: Number of nearest neighbors per query
    ///   - filter: Optional metadata filter (applied post-GPU)
    /// - Returns: Search results for each query
    public func batchSearchGPU(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        guard !queries.isEmpty else { return [] }

        // Ensure GPU is ready
        if gpuContext == nil || accelerationContext == nil {
            try await prepareForGPU()
        }

        guard let context = accelerationContext else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "batchSearchGPU")
        }

        // Check if cache needs refresh
        let currentCount = await baseIndex.count
        if cacheNeedsRefresh || currentCount != cachedVectorCount {
            try await refreshGPUCache()
        }

        // Handle empty index
        guard let datasetBuffer = vectorBuffer, cachedVectorCount > 0 else {
            return queries.map { _ in [] }
        }

        let dimension = await baseIndex.dimension
        let metric = await baseIndex.metric

        // Check if GPU is worthwhile
        let decision = await context.shouldAccelerate(
            queryCount: queries.count,
            candidateCount: cachedVectorCount,
            dimension: dimension,
            k: k
        )

        if !decision.useGPU && !configuration.forceGPU {
            // Fall back to CPU
            return try await baseIndex.batchSearch(queries: queries, k: k, filter: filter)
        }

        // Execute GPU search
        let gpuResults = try await executeGPUSearch(
            queries: queries,
            dataset: datasetBuffer,
            numDataset: cachedVectorCount,
            dimension: dimension,
            k: filter != nil ? min(k * 2, cachedVectorCount) : k,
            metric: metric,
            context: context
        )

        // Apply filter and map to VectorIDs for each query
        return gpuResults.map { queryResults in
            applyFilterAndMapResults(
                gpuResults: queryResults,
                k: k,
                filter: filter,
                metric: metric
            )
        }
    }

    // MARK: - Private GPU Execution

    private func executeGPUSearch(
        queries: [[Float]],
        dataset: any MTLBuffer,
        numDataset: Int,
        dimension: Int,
        k: Int,
        metric: SupportedDistanceMetric,
        context: IndexAccelerationContext
    ) async throws -> [[(index: Int, distance: Float)]] {
        let device = gpuContext!.device.rawDevice

        // Create query buffer
        let flatQueries = queries.flatMap { $0 }
        let queryBufferSize = flatQueries.count * MemoryLayout<Float>.size
        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: queryBufferSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: queryBufferSize)
        }
        queryBuffer.label = "FlatIndexAccelerated.queries"

        // Get fused kernel
        let fusedKernel = try await context.fusedL2TopKKernel()

        // Note: FusedL2TopKKernel is optimized for L2/Euclidean
        // For other metrics, we'd need separate kernels
        // For now, use fused kernel (which computes L2) and note the metric limitation
        // TODO: Add cosine/dot product fused kernels in future

        let params = FusedL2TopKParameters(
            numQueries: queries.count,
            numDataset: numDataset,
            dimension: dimension,
            k: min(k, FusedL2TopKParameters.maxK)
        )

        let result = try await fusedKernel.execute(
            queries: queryBuffer,
            dataset: dataset,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        // Extract results
        return result.allResults()
    }

    private func applyFilterAndMapResults(
        gpuResults: [(index: Int, distance: Float)],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?,
        metric: SupportedDistanceMetric
    ) -> [VectorIndex.SearchResult] {
        var results: [VectorIndex.SearchResult] = []
        results.reserveCapacity(k)

        for (index, gpuDistance) in gpuResults {
            guard index < indexToID.count else { continue }

            // Apply filter if provided
            if let filter = filter {
                let metadata = indexToMetadata[index]
                if !filter(metadata) { continue }
            }

            // Convert GPU squared L2 distance to actual distance for euclidean metric
            // VectorIndex returns sqrt(L2^2) for euclidean, but GPU returns L2^2
            let score: Float
            switch metric {
            case .euclidean:
                score = sqrt(gpuDistance)
            default:
                // Other metrics (cosine, dot product) don't need conversion
                score = gpuDistance
            }

            results.append(VectorIndex.SearchResult(
                id: indexToID[index],
                score: score
            ))

            if results.count >= k { break }
        }

        return results
    }

    // MARK: - Delegated Operations

    public func insert(id: VectorID, vector: [Float], metadata: [String: String]? = nil) async throws {
        try await baseIndex.insert(id: id, vector: vector, metadata: metadata)
        invalidateGPUCache()
    }

    public func remove(id: VectorID) async throws {
        try await baseIndex.remove(id: id)
        invalidateGPUCache()
    }

    /// CPU search (auto-delegates to base index).
    public func search(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        return try await baseIndex.search(query: query, k: k, filter: filter)
    }

    public func contains(id: VectorID) async -> Bool {
        await baseIndex.contains(id: id)
    }

    /// Batch insert with GPU cache invalidation.
    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws {
        try await baseIndex.batchInsert(items)
        invalidateGPUCache()
    }

    /// Clear index and GPU cache.
    public func clear() async {
        await baseIndex.clear()
        vectorBuffer = nil
        indexToID = []
        indexToMetadata = []
        cachedVectorCount = 0
        cacheNeedsRefresh = true
    }

    /// Get statistics about the index.
    public func statistics() async -> VectorIndex.IndexStats {
        await baseIndex.statistics()
    }
}
