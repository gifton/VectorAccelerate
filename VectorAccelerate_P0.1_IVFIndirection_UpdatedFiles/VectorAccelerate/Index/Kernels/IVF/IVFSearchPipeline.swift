//
//  IVFSearchPipeline.swift
//  VectorAccelerate
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
    private let topKSelection: TopKSelectionKernel
    private let indirectionDistance: IVFIndirectionDistanceKernel

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
        self.topKSelection = try await TopKSelectionKernel(context: context)
        self.indirectionDistance = try await IVFIndirectionDistanceKernel(context: context)
    }

    // MARK: - Warm Up

    /// Warm up all pipeline components.
    public func warmUp() async throws {
        try await coarseQuantizer.warmUp()
        try await topKSelection.warmUp()
        try await indirectionDistance.warmUp()
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
            throw IndexError.invalidInput(
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
            throw IndexError.invalidInput(message: "Queries cannot be empty")
        }
        guard k > 0 else {
            throw IndexError.invalidInput(message: "k must be positive")
        }
        // NOTE: TopKSelectionKernel currently supports K up to 128.
        guard k <= TopKParameters.maxK else {
            throw IndexError.invalidInput(message: "k must be <= \(TopKParameters.maxK) for GPU IVF search")
        }

        let startTime = CACurrentMediaTime()
        let device = context.device.rawDevice
        let numQueries = queries.count
        let dimension = structure.dimension
        let nprobe = configuration.nprobe

        // Validate dimension
        guard queries.allSatisfy({ $0.count == dimension }) else {
            throw IndexError.dimensionMismatch(expected: dimension, got: queries[0].count)
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

        // Allocate final output buffers
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

        // Build per-query CSR candidate list (IVF entry indices) + query ID indirection.
        // Layout:
        // - candidateOffsets[q]..candidateOffsets[q+1] is the candidate range for query q
        // - candidateIVFIndices[c] is an index into structure.vectorIndices
        // - candidateQueryIds[c] is the originating query for candidate c
        var candidateOffsets: [UInt32] = Array(repeating: 0, count: numQueries + 1)
        var candidateIVFIndices: [UInt32] = []
        var candidateQueryIds: [UInt32] = []

        candidateOffsets[0] = 0

        for q in 0..<numQueries {
            // De-dupe lists within a query (coarse quantizer should not return duplicates,
            // but ties can cause repeats).
            var uniqueLists: [UInt32] = []
            uniqueLists.reserveCapacity(nprobe)
            for p in 0..<nprobe {
                let listIdx = listIndicesPtr[q * nprobe + p]
                if listIdx == 0xFFFFFFFF { continue }
                if uniqueLists.contains(listIdx) { continue }
                uniqueLists.append(listIdx)
            }

            for listIdxU32 in uniqueLists {
                let listIdx = Int(listIdxU32)
                guard listIdx < structure.numCentroids else { continue }

                let listStart = Int(offsetPtr[listIdx])
                let listEnd = Int(offsetPtr[listIdx + 1])
                if listEnd <= listStart { continue }

                // Append all IVF entries for this list
                for ivfEntry in listStart..<listEnd {
                    candidateIVFIndices.append(UInt32(ivfEntry))
                    candidateQueryIds.append(UInt32(q))
                }
            }

            candidateOffsets[q + 1] = UInt32(candidateIVFIndices.count)
        }

        let totalCandidates = candidateIVFIndices.count
        if totalCandidates == 0 {
            return (indices: outputIndices, distances: outputDistances)
        }

        // Create candidate buffers
        let candidateIVFBuffer: (any MTLBuffer)?
        if candidateIVFIndices.isEmpty {
            candidateIVFBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        } else {
            candidateIVFBuffer = device.makeBuffer(
                bytes: candidateIVFIndices,
                length: candidateIVFIndices.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            )
        }
        guard let candidateIVFIndicesBuffer = candidateIVFBuffer else {
            throw VectorError.bufferAllocationFailed(size: max(candidateIVFIndices.count * MemoryLayout<UInt32>.size, 4))
        }
        candidateIVFIndicesBuffer.label = "IVFSearchPipeline.candidateIVFIndices"

        let candidateQueryBuffer: (any MTLBuffer)?
        if candidateQueryIds.isEmpty {
            candidateQueryBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        } else {
            candidateQueryBuffer = device.makeBuffer(
                bytes: candidateQueryIds,
                length: candidateQueryIds.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            )
        }
        guard let candidateQueryIdsBuffer = candidateQueryBuffer else {
            throw VectorError.bufferAllocationFailed(size: max(candidateQueryIds.count * MemoryLayout<UInt32>.size, 4))
        }
        candidateQueryIdsBuffer.label = "IVFSearchPipeline.candidateQueryIds"

        // Allocate candidate outputs
        let distanceBytes = max(totalCandidates * MemoryLayout<Float>.size, 4)
        let slotBytes = max(totalCandidates * MemoryLayout<UInt32>.size, 4)
        guard let candidateDistances = device.makeBuffer(length: distanceBytes, options: .storageModeShared),
              let candidateSlots = device.makeBuffer(length: slotBytes, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distanceBytes + slotBytes)
        }
        candidateDistances.label = "IVFSearchPipeline.candidateDistances"
        candidateSlots.label = "IVFSearchPipeline.candidateSlots"

        // Allocate top-k outputs per query (indices are per-query candidate positions)
        let topKBytesValues = max(numQueries * k * MemoryLayout<Float>.size, 4)
        let topKBytesIndices = max(numQueries * k * MemoryLayout<UInt32>.size, 4)
        guard let topKValues = device.makeBuffer(length: topKBytesValues, options: .storageModeShared),
              let topKCandidateIndices = device.makeBuffer(length: topKBytesIndices, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: topKBytesValues + topKBytesIndices)
        }
        topKValues.label = "IVFSearchPipeline.topKValues"
        topKCandidateIndices.label = "IVFSearchPipeline.topKCandidateIndices"

        // Compute storage capacity in slots from the vector buffer length.
        let bytesPerVector = dimension * MemoryLayout<Float>.size
        let storageCapacity = bytesPerVector > 0 ? (structure.listVectors.length / bytesPerVector) : 0

        let distanceParams = IVFIndirectionDistanceParameters(
            dimension: dimension,
            totalCandidates: totalCandidates,
            numQueries: numQueries,
            totalIVFEntries: structure.totalVectors,
            storageCapacity: storageCapacity
        )

        // Run distance + per-query top-k selection in a single command buffer.
        try await context.executeAndWait { [self] _, encoder in
            _ = self.indirectionDistance.encode(
                into: encoder,
                queries: queries,
                vectors: structure.listVectors,
                vectorIndices: structure.vectorIndices,
                candidateIVFIndices: candidateIVFIndicesBuffer,
                candidateQueryIds: candidateQueryIdsBuffer,
                outDistances: candidateDistances,
                outSlots: candidateSlots,
                parameters: distanceParams
            )

            // Distances must be visible before selection.
            encoder.memoryBarrier(scope: .buffers)

            for q in 0..<numQueries {
                let start = Int(candidateOffsets[q])
                let end = Int(candidateOffsets[q + 1])
                let count = end - start
                if count <= 0 { continue }

                let topParams = TopKParameters(
                    batchSize: 1,
                    numElements: count,
                    k: k,
                    mode: .minimum,
                    sorted: true
                )

                let inputOffsetBytes = start * MemoryLayout<Float>.size
                let outValueOffsetBytes = (q * k) * MemoryLayout<Float>.size
                let outIndexOffsetBytes = (q * k) * MemoryLayout<UInt32>.size

                _ = self.topKSelection.encodeWithOffsets(
                    into: encoder,
                    input: candidateDistances,
                    inputOffset: inputOffsetBytes,
                    outputValues: topKValues,
                    outputValuesOffset: outValueOffsetBytes,
                    outputIndices: topKCandidateIndices,
                    outputIndicesOffset: outIndexOffsetBytes,
                    parameters: topParams
                )
            }
        }

        // Map per-query candidate positions to storage slots.
        let slotsPtr = candidateSlots.contents().bindMemory(to: UInt32.self, capacity: totalCandidates)
        let topValPtr = topKValues.contents().bindMemory(to: Float.self, capacity: numQueries * k)
        let topIdxPtr = topKCandidateIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)

        for q in 0..<numQueries {
            let start = Int(candidateOffsets[q])
            let end = Int(candidateOffsets[q + 1])
            let count = end - start
            if count <= 0 { continue }

            let base = q * k
            for i in 0..<k {
                let localPos = topIdxPtr[base + i]
                if localPos == 0xFFFFFFFF { continue }

                let globalCandidate = start + Int(localPos)
                if globalCandidate < 0 || globalCandidate >= totalCandidates { continue }

                let slot = slotsPtr[globalCandidate]
                if slot == 0xFFFFFFFF { continue }

                indicesPtr[base + i] = slot
                distancesPtr[base + i] = topValPtr[base + i]
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
