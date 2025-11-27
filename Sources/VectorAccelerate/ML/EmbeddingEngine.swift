//
//  EmbeddingEngine.swift
//  VectorAccelerate
//
//  High-performance embedding operations for ML workflows
//

import Foundation
@preconcurrency import Metal
import Accelerate
import VectorCore

// MARK: - Array Extension for Chunking
extension Array {
    fileprivate func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

/// Configuration for embedding operations
public struct EmbeddingConfiguration: Sendable {
    public let dimension: Int
    public let distanceMetric: SupportedDistanceMetric
    public let useGPU: Bool
    public let batchSize: Int
    public let normalizeEmbeddings: Bool

    public init(
        dimension: Int,
        distanceMetric: SupportedDistanceMetric = .cosine,
        useGPU: Bool = true,
        batchSize: Int = 1000,
        normalizeEmbeddings: Bool = true
    ) {
        self.dimension = dimension
        self.distanceMetric = distanceMetric
        self.useGPU = useGPU
        self.batchSize = batchSize
        self.normalizeEmbeddings = normalizeEmbeddings
    }
}

/// Result of a similarity search operation in the embedding engine
/// Named `EmbeddingSearchResult` to avoid collision with VectorCore's `SearchResult<ID>`
public struct EmbeddingSearchResult: Sendable {
    public let index: Int
    public let distance: Float
    public let metadata: [String: String]?  // Changed to String:String for Sendable

    public init(index: Int, distance: Float, metadata: [String: String]? = nil) {
        self.index = index
        self.distance = distance
        self.metadata = metadata
    }
}

/// Type alias for backwards compatibility
@available(*, deprecated, renamed: "EmbeddingSearchResult")
public typealias SearchResult = EmbeddingSearchResult

/// Clustering result
public struct ClusterResult: Sendable {
    public let clusterAssignments: [Int]
    public let centroids: [[Float]]
    public let inertia: Float
    public let iterations: Int
}

/// Main embedding operations engine
public actor EmbeddingEngine {
    private let context: MetalContext?
    private let configuration: EmbeddingConfiguration
    private let simdFallback: SIMDFallback
    private let logger: Logger
    
    // Cached embeddings and metadata
    private var embeddings: [[Float]] = []
    private var metadata: [[String: String]] = []
    private var normalizedEmbeddings: [[Float]]?
    
    // GPU buffers for cached data
    private var gpuEmbeddingBuffer: BufferToken?
    private var isDirty: Bool = true
    
    // Performance tracking
    private var searchOperations: Int = 0
    private var totalSearchTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    public init(
        configuration: EmbeddingConfiguration,
        context: MetalContext? = nil
    ) async throws {
        self.configuration = configuration
        
        // Handle async context creation separately
        if let ctx = context {
            self.context = ctx
        } else if configuration.useGPU {
            self.context = await MetalContext.createDefault()
        } else {
            self.context = nil
        }
        
        self.simdFallback = SIMDFallback(configuration: .performance)
        self.logger = Logger.shared
        
        await logger.info("Initialized EmbeddingEngine with dimension \(configuration.dimension)", 
                         category: "EmbeddingEngine")
    }
    
    // MARK: - Embedding Management
    
    /// Add embeddings to the engine
    public func addEmbeddings(_ newEmbeddings: [[Float]], metadata: [[String: String]]? = nil) async throws {
        guard newEmbeddings.allSatisfy({ $0.count == configuration.dimension }) else {
            throw VectorError.dimensionMismatch(
                expected: configuration.dimension,
                actual: newEmbeddings.first?.count ?? 0
            )
        }
        
        let measureToken = await logger.startMeasure("addEmbeddings")
        measureToken.addMetadata("count", value: "\(newEmbeddings.count)")
        defer { measureToken.end() }
        
        embeddings.append(contentsOf: newEmbeddings)
        
        if let meta = metadata {
            self.metadata.append(contentsOf: meta)
        } else {
            self.metadata.append(contentsOf: Array(repeating: [:], count: newEmbeddings.count))
        }
        
        // Invalidate caches
        isDirty = true
        normalizedEmbeddings = nil
        gpuEmbeddingBuffer = nil
        
        await logger.debug("Added \(newEmbeddings.count) embeddings, total: \(embeddings.count)",
                          category: "EmbeddingEngine")
    }
    
    /// Clear all embeddings
    public func clearEmbeddings() async {
        embeddings.removeAll()
        metadata.removeAll()
        normalizedEmbeddings = nil
        gpuEmbeddingBuffer = nil
        isDirty = true
    }
    
    /// Get normalized embeddings (cached)
    private func getNormalizedEmbeddings() async -> [[Float]] {
        if let cached = normalizedEmbeddings, !isDirty {
            return cached
        }
        
        if configuration.normalizeEmbeddings {
            normalizedEmbeddings = await normalizeVectors(embeddings)
        } else {
            normalizedEmbeddings = embeddings
        }
        
        isDirty = false
        return normalizedEmbeddings!
    }
    
    // MARK: - Similarity Search
    
    /// Find k nearest neighbors for a query embedding
    ///
    /// Uses optimized GPU kernels for large datasets, falls back to SIMD for small datasets.
    /// - Parameters:
    ///   - query: Query embedding vector
    ///   - k: Number of nearest neighbors to find
    ///   - threshold: Optional distance threshold
    /// - Returns: Array of search results sorted by distance
    public func search(
        query: [Float],
        k: Int = 10,
        threshold: Float? = nil
    ) async throws -> [EmbeddingSearchResult] {
        guard query.count == configuration.dimension else {
            throw VectorError.dimensionMismatch(
                expected: configuration.dimension,
                actual: query.count
            )
        }
        
        guard !embeddings.isEmpty else {
            return []
        }
        
        let measureToken = await logger.startMeasure("embeddingSearch")
        measureToken.addMetadata("k", value: "\(k)")
        defer { measureToken.end() }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Normalize query if needed
        let normalizedQuery = configuration.normalizeEmbeddings 
            ? await normalizeVector(query)
            : query
        
        // Compute distances
        let distances: [Float]
        if embeddings.count > 10000 && context != nil {
            distances = try await computeGPUDistances(query: normalizedQuery)
        } else {
            distances = try await computeCPUDistances(query: normalizedQuery)
        }
        
        // Find top-k with threshold
        var results: [(index: Int, distance: Float)] = []
        for (index, distance) in distances.enumerated() {
            if let thresh = threshold, distance <= thresh {
                results.append((index, distance))
            } else if threshold == nil {
                results.append((index, distance))
            }
        }
        
        // Sort and take top-k (stable sort with index as tiebreaker)
        results.sort { 
            if abs($0.distance - $1.distance) < Float.ulpOfOne * 10 {
                // If distances are essentially equal, prefer lower index
                return $0.index < $1.index
            }
            return $0.distance < $1.distance
        }
        results = Array(results.prefix(k))
        
        // Create search results with metadata
        let searchResults = results.map { result in
            EmbeddingSearchResult(
                index: result.index,
                distance: result.distance,
                metadata: metadata.indices.contains(result.index) ? metadata[result.index] : nil
            )
        }
        
        // Update metrics
        searchOperations += 1
        totalSearchTime += CFAbsoluteTimeGetCurrent() - startTime
        
        return searchResults
    }
    
    /// Batch similarity search
    public func batchSearch(
        queries: [[Float]],
        k: Int = 10,
        threshold: Float? = nil
    ) async throws -> [[EmbeddingSearchResult]] {
        let measureToken = await logger.startMeasure("batchSearch")
        measureToken.addMetadata("batchSize", value: "\(queries.count)")
        defer { measureToken.end() }
        
        // Process queries in parallel
        return try await withThrowingTaskGroup(of: (Int, [EmbeddingSearchResult]).self) { group in
            for (index, query) in queries.enumerated() {
                group.addTask {
                    let results = try await self.search(query: query, k: k, threshold: threshold)
                    return (index, results)
                }
            }

            // Collect results in order
            var orderedResults = [[EmbeddingSearchResult]](repeating: [], count: queries.count)
            for try await (index, results) in group {
                orderedResults[index] = results
            }

            return orderedResults
        }
    }
    
    // MARK: - Clustering
    
    /// Perform k-means clustering on embeddings
    ///
    /// Uses GPU-accelerated Lloyd's algorithm for large datasets
    /// - Parameters:
    ///   - k: Number of clusters
    ///   - maxIterations: Maximum iterations for convergence
    ///   - tolerance: Convergence tolerance
    /// - Returns: Clustering result with assignments and centroids
    public func kMeansClustering(
        k: Int,
        maxIterations: Int = 100,
        tolerance: Float = 1e-4
    ) async throws -> ClusterResult {
        guard !embeddings.isEmpty else {
            throw VectorError.invalidOperation("No embeddings available for clustering")
        }
        
        let measureToken = await logger.startMeasure("kMeansClustering")
        measureToken.addMetadata("k", value: "\(k)")
        measureToken.addMetadata("samples", value: "\(embeddings.count)")
        defer { measureToken.end() }
        
        let normalized = await getNormalizedEmbeddings()
        
        // Initialize centroids using k-means++
        var centroids = try await initializeCentroidsKMeansPlusPlus(data: normalized, k: k)
        var assignments = [Int](repeating: 0, count: normalized.count)
        var previousInertia: Float = Float.infinity
        var iterations = 0
        
        // Lloyd's algorithm
        while iterations < maxIterations {
            // Assignment step
            assignments = await assignToClusters(data: normalized, centroids: centroids)
            
            // Update step
            let newCentroids = await updateCentroids(data: normalized, assignments: assignments, k: k)
            
            // Check convergence
            let inertia = await computeInertia(data: normalized, assignments: assignments, centroids: newCentroids)
            
            if abs(previousInertia - inertia) < tolerance {
                await logger.debug("K-means converged at iteration \(iterations)", category: "EmbeddingEngine")
                centroids = newCentroids
                break
            }
            
            centroids = newCentroids
            previousInertia = inertia
            iterations += 1
        }
        
        let finalInertia = await computeInertia(data: normalized, assignments: assignments, centroids: centroids)
        
        return ClusterResult(
            clusterAssignments: assignments,
            centroids: centroids,
            inertia: finalInertia,
            iterations: iterations
        )
    }
    
    /// Initialize centroids using k-means++ algorithm
    private func initializeCentroidsKMeansPlusPlus(data: [[Float]], k: Int) async throws -> [[Float]] {
        var centroids: [[Float]] = []
        var distances = [Float](repeating: Float.infinity, count: data.count)
        
        // Handle case where k >= data.count
        let actualK = min(k, data.count)
        
        // Choose first centroid randomly
        let firstIdx = Int.random(in: 0..<data.count)
        centroids.append(data[firstIdx])
        
        // Choose remaining centroids
        for _ in 1..<actualK {
            // Update distances to nearest centroid
            for (idx, point) in data.enumerated() {
                let distToNew = try await simdFallback.euclideanDistance(point, centroids.last!)
                distances[idx] = min(distances[idx], distToNew)
            }
            
            // Choose next centroid with probability proportional to squared distance
            let squaredDistances = distances.map { $0 * $0 }
            let totalSquared = squaredDistances.reduce(0, +)
            
            // If all remaining points have zero distance (already selected), pick randomly
            if totalSquared == 0 {
                // Find points not yet selected as centroids
                var availableIndices = [Int]()
                for i in 0..<data.count {
                    if !centroids.contains(where: { $0 == data[i] }) {
                        availableIndices.append(i)
                    }
                }
                
                if !availableIndices.isEmpty {
                    let idx = availableIndices.randomElement()!
                    centroids.append(data[idx])
                } else {
                    // All points are centroids, duplicate last one
                    centroids.append(centroids.last!)
                }
            } else {
                let random = Float.random(in: 0..<totalSquared)
                var cumulative: Float = 0
                
                for (idx, dist) in squaredDistances.enumerated() {
                    cumulative += dist
                    if cumulative >= random {
                        centroids.append(data[idx])
                        distances[idx] = 0  // Mark as selected
                        break
                    }
                }
            }
        }
        
        // If k > data.count, duplicate centroids to reach k
        while centroids.count < k {
            centroids.append(centroids[centroids.count % actualK])
        }
        
        return centroids
    }
    
    /// Assign data points to nearest clusters
    private func assignToClusters(data: [[Float]], centroids: [[Float]]) async -> [Int] {
        var assignments = [Int](repeating: 0, count: data.count)
        
        // Parallel assignment
        await withTaskGroup(of: (Int, Int).self) { group in
            for (idx, point) in data.enumerated() {
                group.addTask { [self] in
                    var minDist = Float.infinity
                    var assignment = 0
                    
                    for (centroidIdx, centroid) in centroids.enumerated() {
                        let dist = try! await self.simdFallback.euclideanDistance(point, centroid)
                        if dist < minDist {
                            minDist = dist
                            assignment = centroidIdx
                        }
                    }
                    
                    return (idx, assignment)
                }
            }
            
            for await (idx, assignment) in group {
                assignments[idx] = assignment
            }
        }
        
        return assignments
    }
    
    /// Update centroids based on assignments
    private func updateCentroids(data: [[Float]], assignments: [Int], k: Int) async -> [[Float]] {
        var centroids = [[Float]](repeating: [Float](repeating: 0, count: configuration.dimension), count: k)
        var counts = [Int](repeating: 0, count: k)
        
        // Sum points in each cluster
        for (point, cluster) in zip(data, assignments) {
            for (idx, value) in point.enumerated() {
                centroids[cluster][idx] += value
            }
            counts[cluster] += 1
        }
        
        // Average to get new centroids
        for cluster in 0..<k {
            if counts[cluster] > 0 {
                for idx in 0..<configuration.dimension {
                    centroids[cluster][idx] /= Float(counts[cluster])
                }
            } else {
                // Empty cluster - reinitialize with a random point
                if !data.isEmpty {
                    let randomIdx = Int.random(in: 0..<data.count)
                    centroids[cluster] = data[randomIdx]
                }
            }
        }
        
        return centroids
    }
    
    /// Compute clustering inertia (sum of squared distances to nearest centroid)
    private func computeInertia(data: [[Float]], assignments: [Int], centroids: [[Float]]) async -> Float {
        var inertia: Float = 0
        
        for (point, cluster) in zip(data, assignments) {
            let dist = try! await simdFallback.euclideanDistance(point, centroids[cluster])
            inertia += dist * dist
        }
        
        return inertia
    }
    
    // MARK: - Distance Computation
    
    /// Compute distances using GPU
    private func computeGPUDistances(query: [Float]) async throws -> [Float] {
        guard let ctx = context else {
            return try await computeCPUDistances(query: query)
        }
        
        let normalized = await getNormalizedEmbeddings()
        
        // Flatten embeddings for GPU processing
        let flatEmbeddings = normalized.flatMap { $0 }
        
        // Get GPU buffers
        let embeddingBuffer = try await ctx.getBuffer(for: flatEmbeddings)
        let queryBuffer = try await ctx.getBuffer(for: query)
        let resultBuffer = try await ctx.getBuffer(size: embeddings.count * MemoryLayout<Float>.stride)
        
        // Load distance computation shader
        let shader = try await loadDistanceShader()
        
        // Execute on GPU
        try await ctx.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(shader)
            encoder.setBuffer(embeddingBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            let embeddingCount = await self.embeddings.count
            var params = (
                embeddingCount: UInt32(embeddingCount),
                dimension: UInt32(self.configuration.dimension),
                metric: UInt32(self.configuration.distanceMetric == .euclidean ? 0 : 1)
            )
            encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 3)
            
            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroupCount = MTLSize(
                width: (embeddingCount + 255) / 256,
                height: 1,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        }
        
        return resultBuffer.copyData(as: Float.self, count: embeddings.count)
    }
    
    /// Compute distances using CPU SIMD
    private func computeCPUDistances(query: [Float]) async throws -> [Float] {
        let normalized = await getNormalizedEmbeddings()
        var distances = [Float](repeating: Float.infinity, count: normalized.count)
        
        // Compute distances sequentially to ensure correct ordering and avoid race conditions
        for (idx, embedding) in normalized.enumerated() {
            let distance: Float
            switch configuration.distanceMetric {
            case .euclidean:
                distance = try await simdFallback.euclideanDistance(query, embedding)
            case .cosine:
                let dot = try await simdFallback.dotProduct(query, embedding)
                distance = 1.0 - dot  // Assuming normalized vectors
            case .manhattan:
                // Delegate to VectorCore's SIMD4-optimized Manhattan distance (0.1.5)
                distance = try await simdFallback.manhattanDistance(query, embedding)
            case .dotProduct:
                let dot = try await simdFallback.dotProduct(query, embedding)
                distance = -dot  // Negate for distance (higher dot product = closer)
            case .chebyshev:
                var maxDiff: Float = 0
                for i in 0..<query.count {
                    maxDiff = max(maxDiff, abs(query[i] - embedding[i]))
                }
                distance = maxDiff
            }
            distances[idx] = distance
        }
        
        return distances
    }
    
    /// Load distance computation shader
    private func loadDistanceShader() async throws -> any MTLComputePipelineState {
        guard let ctx = context else {
            throw VectorError.metalNotAvailable()
        }
        
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void computeDistances(
            device const float* embeddings [[buffer(0)]],
            device const float* query [[buffer(1)]],
            device float* distances [[buffer(2)]],
            constant uint3& params [[buffer(3)]],  // embeddingCount, dimension, metric
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= params.x) return;
            
            uint dimension = params.y;
            uint metric = params.z;
            uint offset = tid * dimension;
            
            float distance = 0.0f;
            
            if (metric == 0) {  // Euclidean
                for (uint i = 0; i < dimension; ++i) {
                    float diff = embeddings[offset + i] - query[i];
                    distance += diff * diff;
                }
                distance = sqrt(distance);
            } else {  // Cosine
                float dot = 0.0f;
                for (uint i = 0; i < dimension; ++i) {
                    dot += embeddings[offset + i] * query[i];
                }
                distance = 1.0f - dot;  // Assuming normalized vectors
            }
            
            distances[tid] = distance;
        }
        """
        
        return try await ctx.compileShader(source: source, functionName: "computeDistances")
    }
    
    // MARK: - Utility Functions
    
    /// Normalize a single vector
    private func normalizeVector(_ vector: [Float]) async -> [Float] {
        return await simdFallback.normalize(vector)
    }
    
    /// Normalize multiple vectors
    private func normalizeVectors(_ vectors: [[Float]]) async -> [[Float]] {
        // Process sequentially to ensure order is preserved
        var results = [[Float]]()
        results.reserveCapacity(vectors.count)
        
        for vector in vectors {
            let normalized = await simdFallback.normalize(vector)
            results.append(normalized)
        }
        
        return results
    }
    
    // MARK: - Performance Metrics
    
    public func getPerformanceMetrics() -> (
        embeddings: Int,
        searches: Int,
        averageSearchTime: TimeInterval
    ) {
        let avgTime = searchOperations > 0 ? totalSearchTime / Double(searchOperations) : 0
        return (embeddings.count, searchOperations, avgTime)
    }
}