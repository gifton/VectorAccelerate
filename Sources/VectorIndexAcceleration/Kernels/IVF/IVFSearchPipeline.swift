//
//  IVFSearchPipeline.swift
//  VectorIndexAcceleration
//
//  Metal 4 pipeline for complete IVF search operations.
//
//  Coordinates three phases:
//  1. Coarse quantization (find nprobe nearest centroids)
//  2. List search (search within selected lists)
//  3. Candidate merge (merge into final top-k)
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - IVF Search Pipeline

/// Metal 4 pipeline for GPU-accelerated IVF search.
///
/// Orchestrates the three-phase IVF search algorithm:
/// 1. **Coarse Quantization**: Find nprobe nearest centroids for each query
/// 2. **List Search**: Search within the selected inverted lists
/// 3. **Candidate Merge**: Merge candidates from all lists into final top-k
///
/// ## Usage
/// ```swift
/// let pipeline = try await IVFSearchPipeline(context: context, configuration: config)
///
/// // Prepare index structure (one-time setup)
/// let structure = try await pipeline.prepareIndexStructure(
///     centroids: centroids,
///     lists: invertedLists,
///     dimension: 128
/// )
///
/// // Execute searches
/// let result = try await pipeline.search(
///     queries: queryVectors,
///     structure: structure,
///     k: 10
/// )
/// ```
public final class IVFSearchPipeline: @unchecked Sendable {

    // MARK: - Properties

    /// The Metal 4 context
    public let context: Metal4Context

    /// The search configuration
    public let configuration: IVFSearchConfiguration

    // MARK: - Private Properties

    private let coarseQuantizer: IVFCoarseQuantizerKernel
    private let fusedL2TopK: FusedL2TopKKernel
    private let topKSelection: TopKSelectionKernel

    // MARK: - Initialization

    /// Create an IVF search pipeline.
    ///
    /// - Parameters:
    ///   - context: The Metal 4 context
    ///   - configuration: Search configuration
    public init(
        context: Metal4Context,
        configuration: IVFSearchConfiguration
    ) async throws {
        try configuration.validate()

        self.context = context
        self.configuration = configuration

        // Initialize component kernels
        self.coarseQuantizer = try await IVFCoarseQuantizerKernel(context: context)
        self.fusedL2TopK = try await FusedL2TopKKernel(context: context)
        self.topKSelection = try await TopKSelectionKernel(context: context)
    }

    // MARK: - Warm Up

    /// Warm up all pipeline components.
    public func warmUp() async throws {
        try await coarseQuantizer.warmUp()
        try await fusedL2TopK.warmUp()
        try await topKSelection.warmUp()
    }

    // MARK: - Index Structure Preparation

    /// Prepare GPU buffers for index structure.
    ///
    /// - Parameters:
    ///   - centroids: Centroid vectors [numCentroids × dimension]
    ///   - lists: Inverted lists - array of vector arrays per centroid
    ///   - dimension: Vector dimension
    /// - Returns: GPU-ready index structure
    public func prepareIndexStructure(
        centroids: [[Float]],
        lists: [[[Float]]],
        dimension: Int
    ) async throws -> IVFGPUIndexStructure {
        let device = context.device.rawDevice

        guard centroids.count == lists.count else {
            throw IndexAccelerationError.invalidInput(
                message: "Number of centroids (\(centroids.count)) must match number of lists (\(lists.count))"
            )
        }

        // Flatten centroids
        let flatCentroids = centroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: flatCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
        }
        centroidBuffer.label = "IVFSearchPipeline.centroids"

        // Build CSR structure for lists
        var listOffsets: [UInt32] = [0]
        var totalVectors = 0

        for list in lists {
            totalVectors += list.count
            listOffsets.append(UInt32(totalVectors))
        }

        // Flatten all list vectors
        var flatVectors: [Float] = []
        flatVectors.reserveCapacity(totalVectors * dimension)
        var vectorIndices: [UInt32] = []
        vectorIndices.reserveCapacity(totalVectors)

        // Track flat index as we append vectors
        var flatIndex: UInt32 = 0
        for list in lists {
            for vector in list {
                flatVectors.append(contentsOf: vector)
                // Store flat index into the concatenated vector buffer
                // This matches the order expected by IVFIndexAccelerated's cachedIDMapping
                vectorIndices.append(flatIndex)
                flatIndex += 1
            }
        }

        // Create GPU buffers
        let vectorBuffer: (any MTLBuffer)?
        if flatVectors.isEmpty {
            vectorBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        } else {
            vectorBuffer = device.makeBuffer(
                bytes: flatVectors,
                length: flatVectors.count * MemoryLayout<Float>.size,
                options: .storageModeShared
            )
        }
        guard let vectors = vectorBuffer else {
            throw VectorError.bufferAllocationFailed(size: flatVectors.count * MemoryLayout<Float>.size)
        }
        vectors.label = "IVFSearchPipeline.listVectors"

        let indexBuffer: (any MTLBuffer)?
        if vectorIndices.isEmpty {
            indexBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        } else {
            indexBuffer = device.makeBuffer(
                bytes: vectorIndices,
                length: vectorIndices.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            )
        }
        guard let indices = indexBuffer else {
            throw VectorError.bufferAllocationFailed(size: vectorIndices.count * MemoryLayout<UInt32>.size)
        }
        indices.label = "IVFSearchPipeline.vectorIndices"

        guard let offsetBuffer = device.makeBuffer(
            bytes: listOffsets,
            length: listOffsets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: listOffsets.count * MemoryLayout<UInt32>.size)
        }
        offsetBuffer.label = "IVFSearchPipeline.listOffsets"

        return IVFGPUIndexStructure(
            centroids: centroidBuffer,
            numCentroids: centroids.count,
            listVectors: vectors,
            vectorIndices: indices,
            listOffsets: offsetBuffer,
            totalVectors: totalVectors,
            dimension: dimension
        )
    }

    // MARK: - Search

    /// Execute IVF search.
    ///
    /// - Parameters:
    ///   - queries: Query vectors [numQueries × dimension]
    ///   - structure: Prepared GPU index structure
    ///   - k: Number of nearest neighbors to find
    /// - Returns: Search result with indices and distances
    public func search(
        queries: [[Float]],
        structure: IVFGPUIndexStructure,
        k: Int
    ) async throws -> IVFSearchResult {
        guard !queries.isEmpty else {
            throw IndexAccelerationError.invalidInput(message: "Queries cannot be empty")
        }
        guard k > 0 else {
            throw IndexAccelerationError.invalidInput(message: "k must be positive")
        }

        let startTime = CACurrentMediaTime()
        let device = context.device.rawDevice
        let numQueries = queries.count
        let dimension = structure.dimension
        let nprobe = configuration.nprobe

        // Validate dimension
        guard queries.allSatisfy({ $0.count == dimension }) else {
            throw IndexAccelerationError.dimensionMismatch(expected: dimension, got: queries[0].count)
        }

        // Create query buffer
        let flatQueries = queries.flatMap { $0 }
        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "IVFSearchPipeline.queries"

        var coarseTime: TimeInterval = 0
        var listSearchTime: TimeInterval = 0
        var mergeTime: TimeInterval = 0
        let bufferTime = CACurrentMediaTime() - startTime

        // Phase 1: Coarse Quantization
        let coarseStart = CACurrentMediaTime()
        let coarseResult = try await coarseQuantizer.findNearestCentroids(
            queries: queryBuffer,
            centroids: structure.centroids,
            numQueries: numQueries,
            numCentroids: structure.numCentroids,
            dimension: dimension,
            nprobe: nprobe
        )
        coarseTime = CACurrentMediaTime() - coarseStart

        // Phase 2: Gather candidates and search
        let listSearchStart = CACurrentMediaTime()

        // Gather candidates from selected lists for each query
        let gatheredResult = try await gatherAndSearch(
            queries: queryBuffer,
            structure: structure,
            coarseResult: coarseResult,
            numQueries: numQueries,
            dimension: dimension,
            k: k
        )
        listSearchTime = CACurrentMediaTime() - listSearchStart

        // Phase 3 is integrated into gatherAndSearch for efficiency
        mergeTime = 0  // Merge happens within the search phase

        let totalTime = CACurrentMediaTime() - startTime

        let phaseTimings = configuration.enableProfiling ? IVFPhaseTimings(
            coarseQuantization: coarseTime,
            listSearch: listSearchTime,
            candidateMerge: mergeTime,
            bufferOperations: bufferTime
        ) : nil

        return IVFSearchResult(
            indices: gatheredResult.indices,
            distances: gatheredResult.distances,
            numQueries: numQueries,
            k: k,
            executionTime: totalTime,
            phaseTimings: phaseTimings
        )
    }

    // MARK: - Private Helpers

    private func gatherAndSearch(
        queries: any MTLBuffer,
        structure: IVFGPUIndexStructure,
        coarseResult: IVFCoarseResult,
        numQueries: Int,
        dimension: Int,
        k: Int
    ) async throws -> (indices: any MTLBuffer, distances: any MTLBuffer) {
        let device = context.device.rawDevice
        let nprobe = coarseResult.nprobe

        // Read list offsets to CPU for gathering
        let offsetPtr = structure.listOffsets.contents().bindMemory(
            to: UInt32.self,
            capacity: structure.numCentroids + 1
        )
        let listIndicesPtr = coarseResult.listIndices.contents().bindMemory(
            to: UInt32.self,
            capacity: numQueries * nprobe
        )

        // For each query, gather vectors from selected lists
        // This is done per-query to handle variable list sizes

        // Allocate output buffers
        let outputSize = numQueries * k
        guard let outputIndices = device.makeBuffer(
            length: outputSize * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize * MemoryLayout<UInt32>.size)
        }
        outputIndices.label = "IVFSearchPipeline.outputIndices"

        guard let outputDistances = device.makeBuffer(
            length: outputSize * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize * MemoryLayout<Float>.size)
        }
        outputDistances.label = "IVFSearchPipeline.outputDistances"

        // Initialize with sentinel values
        let indicesPtr = outputIndices.contents().bindMemory(to: UInt32.self, capacity: outputSize)
        let distancesPtr = outputDistances.contents().bindMemory(to: Float.self, capacity: outputSize)
        for i in 0..<outputSize {
            indicesPtr[i] = 0xFFFFFFFF
            distancesPtr[i] = Float.infinity
        }

        // Process queries - gather candidates and search
        // For efficiency, we batch queries that select similar lists

        // Simple approach: process all queries together by gathering all unique candidates
        var allCandidateIndices = Set<Int>()
        var queryToCandidates: [[Int]] = Array(repeating: [], count: numQueries)

        for q in 0..<numQueries {
            var candidates: [Int] = []
            for p in 0..<nprobe {
                let listIdx = Int(listIndicesPtr[q * nprobe + p])
                guard listIdx < structure.numCentroids else { continue }

                let listStart = Int(offsetPtr[listIdx])
                let listEnd = Int(offsetPtr[listIdx + 1])

                for vecIdx in listStart..<listEnd {
                    candidates.append(vecIdx)
                    allCandidateIndices.insert(vecIdx)
                }
            }
            queryToCandidates[q] = candidates
        }

        // If no candidates, return empty results
        if allCandidateIndices.isEmpty {
            return (indices: outputIndices, distances: outputDistances)
        }

        // Gather unique candidate vectors
        let candidateList = Array(allCandidateIndices).sorted()

        // Read vectors from structure
        let vectorsPtr = structure.listVectors.contents().bindMemory(
            to: Float.self,
            capacity: structure.totalVectors * dimension
        )
        let originalIndicesPtr = structure.vectorIndices.contents().bindMemory(
            to: UInt32.self,
            capacity: structure.totalVectors
        )

        // Create gathered vectors buffer
        var gatheredVectors: [Float] = []
        gatheredVectors.reserveCapacity(candidateList.count * dimension)
        var gatheredOriginalIndices: [UInt32] = []
        gatheredOriginalIndices.reserveCapacity(candidateList.count)

        for candidateIdx in candidateList {
            let vecStart = candidateIdx * dimension
            for d in 0..<dimension {
                gatheredVectors.append(vectorsPtr[vecStart + d])
            }
            gatheredOriginalIndices.append(originalIndicesPtr[candidateIdx])
        }

        guard let gatheredBuffer = device.makeBuffer(
            bytes: gatheredVectors,
            length: gatheredVectors.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: gatheredVectors.count * MemoryLayout<Float>.size)
        }

        // For each query, search within its candidates
        // Use fused L2 + top-k on the gathered candidates
        let params = FusedL2TopKParameters(
            numQueries: numQueries,
            numDataset: candidateList.count,
            dimension: dimension,
            k: min(k, candidateList.count)
        )

        let searchResult = try await fusedL2TopK.execute(
            queries: queries,
            dataset: gatheredBuffer,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        // Map gathered indices back to original indices
        let resultIndicesPtr = searchResult.indices.contents().bindMemory(
            to: UInt32.self,
            capacity: numQueries * k
        )
        let resultDistancesPtr = searchResult.distances!.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * k
        )

        for q in 0..<numQueries {
            for i in 0..<k {
                let offset = q * k + i
                let gatheredIdx = resultIndicesPtr[offset]
                if gatheredIdx != 0xFFFFFFFF && Int(gatheredIdx) < gatheredOriginalIndices.count {
                    indicesPtr[offset] = gatheredOriginalIndices[Int(gatheredIdx)]
                    distancesPtr[offset] = resultDistancesPtr[offset]
                }
            }
        }

        return (indices: outputIndices, distances: outputDistances)
    }
}

// MARK: - GPU Index Structure

/// GPU-ready structure for IVF index.
public struct IVFGPUIndexStructure: Sendable {
    /// Centroid vectors [numCentroids × dimension]
    public let centroids: any MTLBuffer

    /// Number of centroids
    public let numCentroids: Int

    /// Flattened list vectors [totalVectors × dimension]
    public let listVectors: any MTLBuffer

    /// Original vector indices [totalVectors]
    public let vectorIndices: any MTLBuffer

    /// List offsets in CSR format [numCentroids + 1]
    public let listOffsets: any MTLBuffer

    /// Total vectors across all lists
    public let totalVectors: Int

    /// Vector dimension
    public let dimension: Int

    /// Get the number of vectors in a specific list.
    public func listCount(at listIndex: Int) -> Int {
        guard listIndex < numCentroids else { return 0 }
        let offsets = listOffsets.contents().bindMemory(to: UInt32.self, capacity: numCentroids + 1)
        return Int(offsets[listIndex + 1]) - Int(offsets[listIndex])
    }

    /// Estimated memory usage in bytes.
    public var estimatedMemoryBytes: Int {
        let centroidBytes = numCentroids * dimension * MemoryLayout<Float>.size
        let vectorBytes = totalVectors * dimension * MemoryLayout<Float>.size
        let indexBytes = totalVectors * MemoryLayout<UInt32>.size
        let offsetBytes = (numCentroids + 1) * MemoryLayout<UInt32>.size
        return centroidBytes + vectorBytes + indexBytes + offsetBytes
    }
}
