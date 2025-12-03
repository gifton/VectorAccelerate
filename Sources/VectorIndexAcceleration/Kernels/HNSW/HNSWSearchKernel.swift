//
//  HNSWSearchKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW layer-wise search.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Search Result

/// Result from HNSW layer search.
public struct HNSWSearchResult: Sendable {
    /// Top-K node indices
    public let indices: [UInt32]

    /// Corresponding distances
    public let distances: [Float]

    /// Number of nodes explored
    public let nodesVisited: Int

    /// Execution time
    public let executionTime: TimeInterval

    /// Get results as tuples
    public var results: [(index: UInt32, distance: Float)] {
        zip(indices, distances).map { ($0, $1) }
    }

    /// Filter out sentinel values
    public var validResults: [(index: UInt32, distance: Float)] {
        results.filter { $0.index != 0xFFFFFFFF && $0.distance != Float.infinity }
    }
}

/// Result from batch HNSW search.
public struct BatchHNSWSearchResult: Sendable {
    /// Individual search results
    public let batchResults: [HNSWSearchResult]

    /// Total execution time
    public let totalExecutionTime: TimeInterval

    /// Average nodes visited per query
    public let averageNodesVisited: Double

    /// Get flattened indices
    public var flatIndices: [UInt32] {
        batchResults.flatMap { $0.indices }
    }

    /// Get flattened distances
    public var flatDistances: [Float] {
        batchResults.flatMap { $0.distances }
    }
}

// MARK: - HNSW Search Kernel

/// Metal 4 kernel for GPU-accelerated HNSW layer search.
///
/// Performs parallel beam search in HNSW graph layers using visited set
/// tracking and dynamic candidate list management.
///
/// ## Features
/// - Parallel query processing
/// - Bit-packed visited set management
/// - Configurable ef_search parameter
/// - Entry point support
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWSearchKernel(context: context, configuration: config)
/// let result = try await kernel.search(
///     query: queryVector,
///     graphLayer: layer,
///     entryPoints: [0],
///     k: 10
/// )
/// ```
public final class HNSWSearchKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWSearchKernel"

    public let fusibleWith: [String] = ["HNSWDistanceMatrix"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: HNSWSearchConfiguration

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: HNSWSearchConfiguration = HNSWSearchConfiguration()
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library and create pipeline
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "hnsw_search_layer_parallel") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_search_layer_parallel. Ensure HNSW shaders are compiled."
            )
        }

        self.pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline is created in init
    }

    // MARK: - Single Query Search

    /// Search for K nearest neighbors of a single query.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - graphLayer: The graph layer to search
    ///   - entryPoints: Entry point nodes
    ///   - k: Number of neighbors to return
    /// - Returns: Search result
    public func search(
        query: [Float],
        graphLayer: HNSWGraphLayer,
        entryPoints: [UInt32],
        k: Int
    ) async throws -> HNSWSearchResult {
        let batchResult = try await batchSearch(
            queries: [query],
            graphLayer: graphLayer,
            entryPoints: [entryPoints],
            k: k
        )

        return batchResult.batchResults[0]
    }

    // MARK: - Batch Search

    /// Search for K nearest neighbors of multiple queries.
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - graphLayer: The graph layer to search
    ///   - entryPoints: Entry points per query
    ///   - k: Number of neighbors per query
    /// - Returns: Batch search result
    public func batchSearch(
        queries: [[Float]],
        graphLayer: HNSWGraphLayer,
        entryPoints: [[UInt32]],
        k: Int
    ) async throws -> BatchHNSWSearchResult {
        let startTime = CACurrentMediaTime()

        let batchSize = queries.count
        guard batchSize > 0 else {
            throw IndexAccelerationError.invalidInput(message: "Empty query batch")
        }

        guard queries[0].count == graphLayer.dimension else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: graphLayer.dimension,
                got: queries[0].count
            )
        }

        guard entryPoints.count == batchSize else {
            throw IndexAccelerationError.invalidInput(
                message: "Entry points count (\(entryPoints.count)) doesn't match batch size (\(batchSize))"
            )
        }

        let device = context.device.rawDevice

        // Create query buffer
        let flatQueries = queries.flatMap { $0 }
        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "HNSWSearch.queries"

        // Create entry points buffer
        let maxEntryPoints = entryPoints.map { $0.count }.max() ?? 1
        var flatEntryPoints: [UInt32] = []
        let sentinel: UInt32 = 0xFFFFFFFF

        for points in entryPoints {
            flatEntryPoints.append(contentsOf: points)
            let padding = maxEntryPoints - points.count
            if padding > 0 {
                flatEntryPoints.append(contentsOf: Array(repeating: sentinel, count: padding))
            }
        }

        guard let entryPointsBuffer = device.makeBuffer(
            bytes: flatEntryPoints,
            length: flatEntryPoints.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatEntryPoints.count * MemoryLayout<UInt32>.size)
        }
        entryPointsBuffer.label = "HNSWSearch.entryPoints"

        // Create visited flags buffer
        let wordsPerBatch = (graphLayer.nodeCount + 31) / 32
        let totalWords = batchSize * wordsPerBatch
        guard let visitedFlagsBuffer = device.makeBuffer(
            length: totalWords * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalWords * MemoryLayout<UInt32>.size)
        }
        visitedFlagsBuffer.label = "HNSWSearch.visited"
        visitedFlagsBuffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: totalWords)

        // Create output buffers
        guard let resultIndicesBuffer = device.makeBuffer(
            length: batchSize * k * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: batchSize * k * MemoryLayout<UInt32>.size)
        }
        resultIndicesBuffer.label = "HNSWSearch.indices"

        guard let resultDistancesBuffer = device.makeBuffer(
            length: batchSize * k * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: batchSize * k * MemoryLayout<Float>.size)
        }
        resultDistancesBuffer.label = "HNSWSearch.distances"

        // Execute kernel
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "HNSW Layer Search"

            // Set buffers
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(graphLayer.vectors, offset: 0, index: 1)
            encoder.setBuffer(graphLayer.edges, offset: 0, index: 2)
            encoder.setBuffer(graphLayer.edgeOffsets, offset: 0, index: 3)
            encoder.setBuffer(entryPointsBuffer, offset: 0, index: 4)
            encoder.setBuffer(visitedFlagsBuffer, offset: 0, index: 6)
            encoder.setBuffer(resultIndicesBuffer, offset: 0, index: 7)
            encoder.setBuffer(resultDistancesBuffer, offset: 0, index: 8)

            // Set constants
            var b = UInt32(batchSize)
            var n = UInt32(graphLayer.nodeCount)
            var d = UInt32(graphLayer.dimension)
            var ef = UInt32(configuration.efSearch)
            var kOut = UInt32(k)
            var e = UInt32(maxEntryPoints)

            encoder.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 10)
            encoder.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 11)
            encoder.setBytes(&ef, length: MemoryLayout<UInt32>.size, index: 12)
            encoder.setBytes(&kOut, length: MemoryLayout<UInt32>.size, index: 13)
            encoder.setBytes(&e, length: MemoryLayout<UInt32>.size, index: 14)

            // Dispatch one threadgroup per query
            encoder.dispatchThreadgroups(
                MTLSize(width: batchSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: configuration.threadgroupSize, height: 1, depth: 1)
            )
        }

        // Extract results
        let results = extractResults(
            indicesBuffer: resultIndicesBuffer,
            distancesBuffer: resultDistancesBuffer,
            visitedFlagsBuffer: visitedFlagsBuffer,
            batchSize: batchSize,
            nodeCount: graphLayer.nodeCount,
            k: k
        )

        let executionTime = CACurrentMediaTime() - startTime
        let totalVisited = results.map { $0.nodesVisited }.reduce(0, +)

        return BatchHNSWSearchResult(
            batchResults: results,
            totalExecutionTime: executionTime,
            averageNodesVisited: Double(totalVisited) / Double(batchSize)
        )
    }

    // MARK: - Private Helpers

    private func extractResults(
        indicesBuffer: any MTLBuffer,
        distancesBuffer: any MTLBuffer,
        visitedFlagsBuffer: any MTLBuffer,
        batchSize: Int,
        nodeCount: Int,
        k: Int
    ) -> [HNSWSearchResult] {
        let indicesPtr = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: batchSize * k)
        let distancesPtr = distancesBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * k)
        let visitedPtr = visitedFlagsBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: batchSize * ((nodeCount + 31) / 32)
        )

        var results: [HNSWSearchResult] = []

        for b in 0..<batchSize {
            let offset = b * k
            let indices = Array(UnsafeBufferPointer(start: indicesPtr + offset, count: k))
            let distances = Array(UnsafeBufferPointer(start: distancesPtr + offset, count: k))

            // Count visited nodes
            let nodesVisited = countVisitedNodes(
                visitedPtr: visitedPtr,
                batchIndex: b,
                nodeCount: nodeCount
            )

            results.append(HNSWSearchResult(
                indices: indices,
                distances: distances,
                nodesVisited: nodesVisited,
                executionTime: 0
            ))
        }

        return results
    }

    private func countVisitedNodes(
        visitedPtr: UnsafeMutablePointer<UInt32>,
        batchIndex: Int,
        nodeCount: Int
    ) -> Int {
        let wordsPerBatch = (nodeCount + 31) / 32
        let batchOffset = batchIndex * wordsPerBatch

        var count = 0
        for i in 0..<wordsPerBatch {
            let word = visitedPtr[batchOffset + i]
            count += word.nonzeroBitCount
        }

        return count
    }
}
