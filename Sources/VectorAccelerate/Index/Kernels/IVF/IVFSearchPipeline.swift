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
    private let fusedL2TopK: FusedL2TopKKernel
    private let topKSelection: TopKSelectionKernel

    // MARK: - Debugging

    /// Global toggle for lightweight debug prints during search.
    /// Tests can set this to true to get coarse/fine valid-count summaries.
    nonisolated(unsafe) public static var debugEnabled: Bool = false

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

        let startTime = CACurrentMediaTime()
        let device = context.device.rawDevice
        let numQueries = queries.count
        let dimension = structure.dimension
        let nprobe = configuration.nprobe

        // Validate dimension
        guard queries.allSatisfy({ $0.count == dimension }) else {
            throw IndexError.dimensionMismatch(expected: dimension, got: queries[0].count)
        }

        // === DEBUG: Log cluster utilization (list size distribution) ===
        if Self.debugEnabled {
            let offsetPtr = structure.listOffsets.contents().bindMemory(
                to: UInt32.self,
                capacity: structure.numCentroids + 1
            )
            var listSizes: [Int] = []
            var emptyLists = 0
            var minSize = Int.max
            var maxSize = 0
            var totalVecs = 0

            for c in 0..<structure.numCentroids {
                let start = Int(offsetPtr[c])
                let end = Int(offsetPtr[c + 1])
                let size = end - start
                listSizes.append(size)
                totalVecs += size
                if size == 0 { emptyLists += 1 }
                minSize = min(minSize, size)
                maxSize = max(maxSize, size)
            }

            let avgSize = Double(totalVecs) / Double(max(1, structure.numCentroids))
            let variance = listSizes.reduce(0.0) { sum, size in
                let diff = Double(size) - avgSize
                return sum + diff * diff
            } / Double(max(1, structure.numCentroids))
            let stdDev = sqrt(variance)

            print("[IVF Debug] === CLUSTER UTILIZATION ===")
            print("[IVF Debug] numCentroids=\(structure.numCentroids), totalVectors=\(totalVecs)")
            print("[IVF Debug] listSizes: min=\(minSize), max=\(maxSize), avg=\(String(format: "%.1f", avgSize)), stdDev=\(String(format: "%.1f", stdDev))")
            print("[IVF Debug] emptyLists=\(emptyLists)")

            // Show first 10 list sizes
            let sampleSizes = listSizes.prefix(10).map { String($0) }.joined(separator: ", ")
            print("[IVF Debug] first 10 list sizes: [\(sampleSizes)]")
        }

        // === DEBUG: Verify centroid computation (sample check) ===
        if Self.debugEnabled {
            let centroidPtr = structure.centroids.contents().bindMemory(
                to: Float.self,
                capacity: structure.numCentroids * dimension
            )
            let vectorsPtr = structure.listVectors.contents().bindMemory(
                to: Float.self,
                capacity: structure.totalVectors * dimension
            )
            let offsetPtr = structure.listOffsets.contents().bindMemory(
                to: UInt32.self,
                capacity: structure.numCentroids + 1
            )

            print("[IVF Debug] === CENTROID VERIFICATION (first 3 non-empty clusters) ===")
            var verified = 0
            for c in 0..<structure.numCentroids where verified < 3 {
                let listStart = Int(offsetPtr[c])
                let listEnd = Int(offsetPtr[c + 1])
                let listCount = listEnd - listStart
                guard listCount > 0 else { continue }

                // Compute actual mean of vectors in this cluster
                var actualMean = [Float](repeating: 0, count: dimension)
                for vecIdx in listStart..<listEnd {
                    for d in 0..<dimension {
                        actualMean[d] += vectorsPtr[vecIdx * dimension + d]
                    }
                }
                for d in 0..<dimension {
                    actualMean[d] /= Float(listCount)
                }

                // Get stored centroid
                var storedCentroid = [Float](repeating: 0, count: dimension)
                for d in 0..<dimension {
                    storedCentroid[d] = centroidPtr[c * dimension + d]
                }

                // Compute L2 distance between stored and actual
                var l2Dist: Float = 0
                for d in 0..<dimension {
                    let diff = storedCentroid[d] - actualMean[d]
                    l2Dist += diff * diff
                }
                l2Dist = sqrt(l2Dist)

                // Compute norms for reference
                var storedNorm: Float = 0
                var actualNorm: Float = 0
                for d in 0..<dimension {
                    storedNorm += storedCentroid[d] * storedCentroid[d]
                    actualNorm += actualMean[d] * actualMean[d]
                }
                storedNorm = sqrt(storedNorm)
                actualNorm = sqrt(actualNorm)

                print("[IVF Debug] cluster[\(c)]: count=\(listCount), storedNorm=\(String(format: "%.3f", storedNorm)), actualMeanNorm=\(String(format: "%.3f", actualNorm)), centroid-vs-mean L2=\(String(format: "%.4f", l2Dist))")

                // Show first 4 dims of each
                let storedSample = storedCentroid.prefix(4).map { String(format: "%.3f", $0) }.joined(separator: ", ")
                let actualSample = actualMean.prefix(4).map { String(format: "%.3f", $0) }.joined(separator: ", ")
                print("[IVF Debug]   stored[0:4]=[\(storedSample)]")
                print("[IVF Debug]   actual[0:4]=[\(actualSample)]")

                verified += 1
            }
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

        // Debug: summarize how many coarse lists were actually selected (non-sentinel)
        if Self.debugEnabled {
            let listPtr = coarseResult.listIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * nprobe)
            let distPtr = coarseResult.listDistances.contents().bindMemory(to: Float.self, capacity: numQueries * nprobe)
            let offsetPtr = structure.listOffsets.contents().bindMemory(to: UInt32.self, capacity: structure.numCentroids + 1)

            var minSel = Int.max
            var maxSel = Int.min
            var sumSel = 0

            print("[IVF Debug] === COARSE QUANTIZATION (cluster selection) ===")
            print("[IVF Debug] nprobe=\(nprobe), numQueries=\(numQueries)")

            // Show detailed selection for first 3 queries
            let detailQueries = min(3, numQueries)
            for q in 0..<detailQueries {
                var selectedClusters: [(cluster: Int, distance: Float, listSize: Int)] = []
                for p in 0..<nprobe {
                    let clusterIdx = listPtr[q * nprobe + p]
                    let dist = distPtr[q * nprobe + p]
                    if clusterIdx != 0xFFFFFFFF && Int(clusterIdx) < structure.numCentroids {
                        let listStart = Int(offsetPtr[Int(clusterIdx)])
                        let listEnd = Int(offsetPtr[Int(clusterIdx) + 1])
                        let listSize = listEnd - listStart
                        selectedClusters.append((Int(clusterIdx), dist, listSize))
                    }
                }
                let clusterStr = selectedClusters.map { "c\($0.cluster)(d=\(String(format: "%.2f", $0.distance)),n=\($0.listSize))" }.joined(separator: ", ")
                let totalCandidates = selectedClusters.reduce(0) { $0 + $1.listSize }
                print("[IVF Debug] query[\(q)]: selected \(selectedClusters.count) clusters, totalCandidates=\(totalCandidates)")
                print("[IVF Debug]   clusters: [\(clusterStr)]")
            }

            // Aggregate stats for all queries
            for q in 0..<numQueries {
                var count = 0
                for p in 0..<nprobe {
                    let val = listPtr[q * nprobe + p]
                    if val != 0xFFFFFFFF && Int(val) < structure.numCentroids { count += 1 }
                }
                minSel = min(minSel, count)
                maxSel = max(maxSel, count)
                sumSel += count
            }
            let avgSel = Double(sumSel) / Double(max(1, numQueries))
            print("[IVF Debug] Coarse summary: requested=\(nprobe), avg=\(String(format: "%.1f", avgSel)), min=\(minSel), max=\(maxSel)")
        }

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

        // Debug: summarize how many valid neighbors were produced per query (non-sentinel)
        if Self.debugEnabled {
            let idxPtr = gatheredResult.indices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)
            let distPtr = gatheredResult.distances.contents().bindMemory(to: Float.self, capacity: numQueries * k)

            var minValid = Int.max
            var maxValid = Int.min
            var sumValid = 0

            print("[IVF Debug] === FINE SEARCH RESULTS ===")

            // Show detailed results for first 3 queries
            let detailQueries = min(3, numQueries)
            for q in 0..<detailQueries {
                var results: [(index: UInt32, distance: Float)] = []
                let base = q * k
                for i in 0..<k {
                    let idx = idxPtr[base + i]
                    let dist = distPtr[base + i]
                    if idx != 0xFFFFFFFF {
                        results.append((idx, dist))
                    }
                }
                let resultStr = results.prefix(5).map { "idx\($0.index)(d=\(String(format: "%.3f", $0.distance)))" }.joined(separator: ", ")
                print("[IVF Debug] query[\(q)]: found \(results.count)/\(k) neighbors")
                print("[IVF Debug]   top-5: [\(resultStr)]")

                // Check if distances are sorted
                var isSorted = true
                for i in 1..<results.count {
                    if results[i].distance < results[i-1].distance {
                        isSorted = false
                        break
                    }
                }
                if !isSorted && results.count > 1 {
                    print("[IVF Debug]   WARNING: results NOT sorted by distance!")
                }
            }

            // Aggregate stats
            for q in 0..<numQueries {
                var cnt = 0
                let base = q * k
                for i in 0..<k {
                    if idxPtr[base + i] != 0xFFFFFFFF { cnt += 1 }
                }
                minValid = min(minValid, cnt)
                maxValid = max(maxValid, cnt)
                sumValid += cnt
            }
            let avgValid = Double(sumValid) / Double(max(1, numQueries))
            print("[IVF Debug] Fine summary: requested K=\(k), avg=\(String(format: "%.1f", avgValid)), min=\(minValid), max=\(maxValid)")
        }

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

        // Debug: log candidate gathering details
        if IVFSearchPipeline.debugEnabled {
            print("[IVF Debug] === CANDIDATE GATHERING ===")
            print("[IVF Debug] totalUniqueCandidates=\(allCandidateIndices.count), totalVectors=\(structure.totalVectors)")

            // Show details for first 3 queries
            let detailQueries = min(3, numQueries)
            for q in 0..<detailQueries {
                let candidates = queryToCandidates[q]
                let uniqueCandidates = Set(candidates)
                print("[IVF Debug] query[\(q)]: candidatesGathered=\(candidates.count), unique=\(uniqueCandidates.count)")
                if !candidates.isEmpty {
                    let sampleIndices = candidates.prefix(10).map { String($0) }.joined(separator: ", ")
                    print("[IVF Debug]   first 10 candidate indices: [\(sampleIndices)]")
                }
            }
        }

        // If no candidates, return empty results
        if allCandidateIndices.isEmpty {
            if IVFSearchPipeline.debugEnabled {
                print("[IVF Debug] WARNING: No candidates gathered! Returning empty results.")
            }
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

        // Debug: log fused L2 search results before mapping
        if IVFSearchPipeline.debugEnabled {
            print("[IVF Debug] === FUSED L2 SEARCH (before index mapping) ===")
            print("[IVF Debug] searchedDatasetSize=\(candidateList.count), effectiveK=\(params.k)")

            let fusedIdxPtr = searchResult.indices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)
            let fusedDistPtr = searchResult.distances!.contents().bindMemory(to: Float.self, capacity: numQueries * k)

            // Show details for first 3 queries
            let detailQueries = min(3, numQueries)
            for q in 0..<detailQueries {
                var results: [(gatheredIdx: UInt32, distance: Float)] = []
                for i in 0..<k {
                    let idx = fusedIdxPtr[q * k + i]
                    let dist = fusedDistPtr[q * k + i]
                    if idx != 0xFFFFFFFF {
                        results.append((idx, dist))
                    }
                }
                let resultStr = results.prefix(5).map { "g\($0.gatheredIdx)(d=\(String(format: "%.3f", $0.distance)))" }.joined(separator: ", ")
                print("[IVF Debug] query[\(q)] fused results: \(results.count) found")
                print("[IVF Debug]   top-5 (gatheredIdx): [\(resultStr)]")
            }
        }

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

        // Debug: log index mapping results
        if IVFSearchPipeline.debugEnabled {
            print("[IVF Debug] === INDEX MAPPING (gatheredIdx -> originalIdx) ===")
            print("[IVF Debug] gatheredOriginalIndices.count=\(gatheredOriginalIndices.count)")

            // Show first 10 mappings
            let sampleMappings = gatheredOriginalIndices.prefix(10).enumerated().map { "g\($0.offset)->o\($0.element)" }.joined(separator: ", ")
            print("[IVF Debug] first 10 mappings: [\(sampleMappings)]")

            // Show final output for first 3 queries
            let detailQueries = min(3, numQueries)
            for q in 0..<detailQueries {
                var finalResults: [(originalIdx: UInt32, distance: Float)] = []
                for i in 0..<k {
                    let idx = indicesPtr[q * k + i]
                    let dist = distancesPtr[q * k + i]
                    if idx != 0xFFFFFFFF {
                        finalResults.append((idx, dist))
                    }
                }
                let resultStr = finalResults.prefix(5).map { "o\($0.originalIdx)(d=\(String(format: "%.3f", $0.distance)))" }.joined(separator: ", ")
                print("[IVF Debug] query[\(q)] final: \(finalResults.count) results")
                print("[IVF Debug]   top-5 (originalIdx): [\(resultStr)]")
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
