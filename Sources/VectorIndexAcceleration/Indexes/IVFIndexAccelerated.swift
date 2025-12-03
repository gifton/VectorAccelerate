//
//  IVFIndexAccelerated.swift
//  VectorIndexAcceleration
//
//  GPU-accelerated wrapper for IVF indices.
//
//  Provides Metal 4 GPU acceleration for IVF search operations using
//  the three-phase IVF search pipeline.
//

import Foundation
@preconcurrency import Metal
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - IVF Index Accelerated

/// GPU-accelerated IVF index wrapper.
///
/// Wraps an `IVFIndex` and provides Metal 4 GPU acceleration for search operations.
/// GPU acceleration is particularly effective for IVF due to:
/// - Parallel centroid distance computation
/// - Batch processing of inverted list searches
/// - Efficient candidate merging
///
/// ## Usage
/// ```swift
/// let baseIndex = IVFIndex(dimension: 128, metric: .euclidean)
/// let accelerated = IVFIndexAccelerated(baseIndex: baseIndex)
///
/// // Prepare GPU (loads centroids and lists to GPU)
/// try await accelerated.prepareForGPU()
///
/// // GPU-accelerated search
/// let results = try await accelerated.searchGPU(query: queryVector, k: 10)
/// ```
public actor IVFIndexAccelerated {

    // MARK: - Properties

    /// The underlying IVF index
    public let baseIndex: IVFIndex

    /// GPU acceleration context
    public private(set) var gpuContext: Metal4Context?

    /// Shared acceleration context for kernel reuse
    private var accelerationContext: IndexAccelerationContext?

    /// Configuration for acceleration decisions
    private let configuration: IndexAccelerationConfiguration

    /// IVF search pipeline
    private var searchPipeline: IVFSearchPipeline?

    /// Cached GPU index structure
    private var gpuIndexStructure: IVFGPUIndexStructure?

    /// Cached ID mapping (global index → VectorID)
    private var cachedIDMapping: [VectorID] = []

    /// Cached metadata for filtering
    private var cachedMetadata: [[String: String]?] = []

    /// Whether the GPU structure needs refresh
    private var structureNeedsRefresh: Bool = true

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
        gpuContext != nil && searchPipeline != nil
    }

    // MARK: - Initialization

    public init(
        baseIndex: IVFIndex,
        configuration: IndexAccelerationConfiguration = .default
    ) {
        self.baseIndex = baseIndex
        self.configuration = configuration
    }

    public init(
        baseIndex: IVFIndex,
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

        guard let metal4Context = self.gpuContext else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "prepareForGPU")
        }

        // Get index structure
        let structure = await baseIndex.getIndexStructure()

        // Create search pipeline with appropriate configuration
        let searchConfig: IVFSearchConfiguration
        switch structure {
        case .ivf(let ivfStructure):
            let dim = await baseIndex.dimension
            let met = await baseIndex.metric
            searchConfig = IVFSearchConfiguration(
                numCentroids: ivfStructure.centroids.count,
                nprobe: ivfStructure.nprobe,
                dimension: dim,
                metric: met,
                enableProfiling: configuration.enableProfiling
            )
        default:
            // Fallback to default config
            let dim = await baseIndex.dimension
            let met = await baseIndex.metric
            let stats = await baseIndex.statistics()
            let nlist = Int(stats.details["trained_nlist"] ?? "256") ?? 256
            let nprobe = Int(stats.details["nprobe"] ?? "8") ?? 8
            searchConfig = IVFSearchConfiguration(
                numCentroids: max(nlist, 1),
                nprobe: nprobe,
                dimension: dim,
                metric: met,
                enableProfiling: configuration.enableProfiling
            )
        }

        self.searchPipeline = try await IVFSearchPipeline(
            context: metal4Context,
            configuration: searchConfig
        )

        try await accelerationContext?.warmUp()
        try await searchPipeline?.warmUp()

        // Prepare GPU index structure
        try await refreshGPUStructure()
    }

    public func releaseGPUResources() async {
        await accelerationContext?.releaseKernels()
        accelerationContext = nil
        gpuContext = nil
        searchPipeline = nil
        gpuIndexStructure = nil
    }

    /// Mark GPU structure as needing refresh (call after index modifications).
    public func invalidateGPUStructure() {
        structureNeedsRefresh = true
    }

    /// Refresh GPU structure from base index.
    private func refreshGPUStructure() async throws {
        guard let pipeline = searchPipeline else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "refreshGPUStructure")
        }

        let structure = await baseIndex.getIndexStructure()
        let dimension = await baseIndex.dimension

        switch structure {
        case .ivf(let ivfStructure):
            // Convert IVFStructure to the format expected by the pipeline
            // ivfStructure has centroids: [[Float]] and invertedLists: [[VectorID]]
            // We need to get the actual vectors from the base index

            // Use getCandidates to retrieve all vectors with their data
            // This is a query-independent way to get all vectors organized by list
            let allCandidates = try await baseIndex.getCandidates(
                query: Array(repeating: 0, count: dimension),
                k: Int.max,
                filter: nil
            )

            // Build a mapping from VectorID to vector data
            var idToVector: [VectorID: [Float]] = [:]
            idToVector.reserveCapacity(allCandidates.vectorCount)
            for i in 0..<allCandidates.vectorCount {
                let start = i * dimension
                let end = start + dimension
                let vec = Array(allCandidates.vectorStorage[start..<end])
                idToVector[allCandidates.ids[i]] = vec
            }

            // Build list vectors using the mapping
            var listVectors: [[[Float]]] = []
            listVectors.reserveCapacity(ivfStructure.invertedLists.count)

            for list in ivfStructure.invertedLists {
                var vectors: [[Float]] = []
                vectors.reserveCapacity(list.count)
                for vectorID in list {
                    if let vec = idToVector[vectorID] {
                        vectors.append(vec)
                    }
                    // Skip vectors not found (shouldn't happen with consistent data)
                }
                listVectors.append(vectors)
            }

            self.gpuIndexStructure = try await pipeline.prepareIndexStructure(
                centroids: ivfStructure.centroids,
                lists: listVectors,
                dimension: dimension
            )

            // Cache the ID mapping for result translation
            // Build mapping in same order as vectors were added to lists
            var idMapping: [VectorID] = []
            var metadataMapping: [[String: String]?] = []
            for list in ivfStructure.invertedLists {
                for vectorID in list {
                    if idToVector[vectorID] != nil {
                        idMapping.append(vectorID)
                        // Get metadata from candidates
                        if let idx = allCandidates.ids.firstIndex(of: vectorID) {
                            metadataMapping.append(allCandidates.metadata[idx])
                        } else {
                            metadataMapping.append(nil)
                        }
                    }
                }
            }
            self.cachedIDMapping = idMapping
            self.cachedMetadata = metadataMapping

        case .flat:
            // For flat structure, create a single-centroid IVF
            let candidates = try await baseIndex.getCandidates(
                query: Array(repeating: 0, count: dimension),
                k: Int.max,
                filter: nil
            )

            // Use the mean vector as centroid (or zero vector)
            let centroid = Array(repeating: Float(0), count: dimension)

            // Get all vectors
            var allVectors: [[Float]] = []
            allVectors.reserveCapacity(candidates.vectorCount)
            for i in 0..<candidates.vectorCount {
                let start = i * dimension
                let vec = Array(candidates.vectorStorage[start..<(start + dimension)])
                allVectors.append(vec)
            }

            self.gpuIndexStructure = try await pipeline.prepareIndexStructure(
                centroids: [centroid],
                lists: [allVectors],
                dimension: dimension
            )

            // Cache IDs and metadata for flat structure
            self.cachedIDMapping = candidates.ids
            self.cachedMetadata = candidates.metadata

        case .hnsw:
            throw IndexAccelerationError.invalidInput(message: "HNSW structure not supported for IVF acceleration")
        }

        structureNeedsRefresh = false
    }

    // MARK: - Search Operations

    public func searchGPU(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        if gpuContext == nil || searchPipeline == nil {
            try await prepareForGPU()
        }

        guard let pipeline = searchPipeline,
              let structure = gpuIndexStructure else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "searchGPU")
        }

        // Refresh structure if needed
        if structureNeedsRefresh {
            try await refreshGPUStructure()
        }

        // Check if GPU is worthwhile
        let decision = shouldUseGPU(candidateCount: structure.totalVectors, k: k)
        if !decision.useGPU && !configuration.forceGPU {
            return try await baseIndex.search(query: query, k: k, filter: filter)
        }

        let metric = await baseIndex.metric

        // Execute GPU search
        // Over-fetch if filtering to ensure we get enough results after filtering
        let gpuK = filter != nil ? min(k * 2, structure.totalVectors) : k
        let result = try await pipeline.search(
            queries: [query],
            structure: structure,
            k: gpuK
        )

        // Convert to SearchResult format using cached ID mapping
        let gpuResults = result.results(for: 0)

        // Map indices to VectorIDs and apply filter
        return mapGPUResultsToSearchResults(
            gpuResults: gpuResults,
            k: k,
            filter: filter,
            metric: metric
        )
    }

    /// Map GPU result indices to VectorIDs and apply optional filtering.
    private func mapGPUResultsToSearchResults(
        gpuResults: [(index: Int, distance: Float)],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?,
        metric: SupportedDistanceMetric
    ) -> [VectorIndex.SearchResult] {
        var results: [VectorIndex.SearchResult] = []
        results.reserveCapacity(k)

        for (idx, gpuDistance) in gpuResults {
            // Validate index is within cached mapping
            guard idx < cachedIDMapping.count else { continue }

            // Apply metadata filter if provided
            if let filter = filter {
                let metadata = idx < cachedMetadata.count ? cachedMetadata[idx] : nil
                if !filter(metadata) { continue }
            }

            // Convert GPU squared L2 distance to actual distance for euclidean metric
            // VectorIndex returns sqrt(L2^2) for euclidean, but GPU returns L2^2
            let score: Float
            switch metric {
            case .euclidean:
                score = sqrt(gpuDistance)
            default:
                score = gpuDistance
            }

            let vectorID = cachedIDMapping[idx]
            results.append(VectorIndex.SearchResult(id: vectorID, score: score))

            if results.count >= k { break }
        }

        return results
    }

    public func batchSearchGPU(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        if gpuContext == nil || searchPipeline == nil {
            try await prepareForGPU()
        }

        guard let pipeline = searchPipeline,
              let structure = gpuIndexStructure else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "batchSearchGPU")
        }

        // Refresh structure if needed
        if structureNeedsRefresh {
            try await refreshGPUStructure()
        }

        // Check if GPU is worthwhile
        let decision = shouldUseGPU(candidateCount: structure.totalVectors * queries.count, k: k)
        if !decision.useGPU && !configuration.forceGPU {
            return try await baseIndex.batchSearch(queries: queries, k: k, filter: filter)
        }

        let metric = await baseIndex.metric

        // Execute GPU search
        // Over-fetch if filtering to ensure we get enough results after filtering
        let gpuK = filter != nil ? min(k * 2, structure.totalVectors) : k
        let result = try await pipeline.search(
            queries: queries,
            structure: structure,
            k: gpuK
        )

        // Convert to SearchResult format using cached ID mapping
        return result.allResults().map { queryResults in
            mapGPUResultsToSearchResults(
                gpuResults: queryResults,
                k: k,
                filter: filter,
                metric: metric
            )
        }
    }

    // MARK: - Decision Logic

    private func shouldUseGPU(candidateCount: Int, k: Int) -> AccelerationDecision {
        // Simple heuristic based on configuration
        if candidateCount < configuration.minimumCandidatesForGPU {
            return AccelerationDecision(
                useGPU: false,
                reason: .datasetTooSmall,
                estimatedSpeedup: 0.5
            )
        }

        // Estimate speedup based on candidate count
        let estimatedSpeedup = Float(candidateCount) / Float(configuration.minimumCandidatesForGPU) * 2.0

        return AccelerationDecision(
            useGPU: true,
            reason: .gpuRecommended,
            estimatedSpeedup: estimatedSpeedup
        )
    }

    // MARK: - Delegated Operations

    public func insert(id: VectorID, vector: [Float], metadata: [String: String]? = nil) async throws {
        try await baseIndex.insert(id: id, vector: vector, metadata: metadata)
        invalidateGPUStructure()
    }

    public func remove(id: VectorID) async throws {
        try await baseIndex.remove(id: id)
        invalidateGPUStructure()
    }

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
}
