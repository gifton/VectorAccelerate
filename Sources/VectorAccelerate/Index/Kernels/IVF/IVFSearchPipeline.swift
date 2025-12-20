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
    private let ivfListSearch: IVFListSearchKernel

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
        self.ivfListSearch = try await IVFListSearchKernel(context: context)
    }

    // MARK: - Warm Up

    /// Warm up all pipeline components.
    public func warmUp() async throws {
        try await coarseQuantizer.warmUp()
        try await fusedL2TopK.warmUp()
        try await topKSelection.warmUp()
        try await ivfListSearch.warmUp()
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

        var fusedSearchTime: TimeInterval = 0
        var mergeTime: TimeInterval = 0
        let bufferTime = CACurrentMediaTime() - startTime

        // === FUSED DISPATCH: Allocate all output buffers upfront ===

        // Coarse quantization output buffers
        let coarseIndicesSize = numQueries * nprobe * MemoryLayout<UInt32>.size
        let coarseDistancesSize = numQueries * nprobe * MemoryLayout<Float>.size

        guard let coarseListIndices = device.makeBuffer(length: coarseIndicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: coarseIndicesSize)
        }
        coarseListIndices.label = "IVFSearchPipeline.coarseListIndices"

        guard let coarseListDistances = device.makeBuffer(length: coarseDistancesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: coarseDistancesSize)
        }
        coarseListDistances.label = "IVFSearchPipeline.coarseListDistances"

        // IVF list search output buffers
        let outputCount = numQueries * k
        let outputIndicesSize = outputCount * MemoryLayout<UInt32>.size
        let outputDistancesSize = outputCount * MemoryLayout<Float>.size

        guard let outputIndices = device.makeBuffer(length: outputIndicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputIndicesSize)
        }
        outputIndices.label = "IVFListSearch.outputIndices"

        guard let outputDistances = device.makeBuffer(length: outputDistancesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputDistancesSize)
        }
        outputDistances.label = "IVFListSearch.outputDistances"

        // Build IVF list search parameters
        let listSearchParams = IVFListSearchParameters(
            numQueries: numQueries,
            numCentroids: structure.numCentroids,
            dimension: dimension,
            nprobe: nprobe,
            k: k,
            maxCandidatesPerQuery: structure.totalVectors
        )

        // === FUSED DISPATCH: Single command buffer for both phases ===
        let fusedSearchStart = CACurrentMediaTime()

        try await context.executeAndWait { [self] _, encoder in
            // Phase 1: Coarse Quantization (find nprobe nearest centroids)
            try self.coarseQuantizer.encode(
                into: encoder,
                queries: queryBuffer,
                centroids: structure.centroids,
                outputIndices: coarseListIndices,
                outputDistances: coarseListDistances,
                numQueries: numQueries,
                numCentroids: structure.numCentroids,
                dimension: dimension,
                nprobe: nprobe
            )

            // Memory barrier to ensure coarse results are visible to list search
            encoder.memoryBarrier(scope: .buffers)

            // Phase 2: IVF List Search (search within selected lists)
            self.ivfListSearch.encode(
                into: encoder,
                queries: queryBuffer,
                structure: structure,
                selectedLists: coarseListIndices,
                outputIndices: outputIndices,
                outputDistances: outputDistances,
                parameters: listSearchParams
            )
        }

        fusedSearchTime = CACurrentMediaTime() - fusedSearchStart

        // Debug: summarize how many coarse lists were actually selected (non-sentinel)
        if Self.debugEnabled {
            let listPtr = coarseListIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * nprobe)
            let distPtr = coarseListDistances.contents().bindMemory(to: Float.self, capacity: numQueries * nprobe)
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

        // Debug: summarize how many valid neighbors were produced per query (non-sentinel)
        if Self.debugEnabled {
            let idxPtr = outputIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)
            let distPtr = outputDistances.contents().bindMemory(to: Float.self, capacity: numQueries * k)

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

        // Phase 3 is integrated into the fused search
        mergeTime = 0  // Merge happens within the search kernel

        let totalTime = CACurrentMediaTime() - startTime

        // With fused dispatch, we report combined time in listSearch for backwards compatibility
        let phaseTimings = configuration.enableProfiling ? IVFPhaseTimings(
            coarseQuantization: 0,  // Fused with list search
            listSearch: fusedSearchTime,  // Combined coarse + list search
            candidateMerge: mergeTime,
            bufferOperations: bufferTime
        ) : nil

        return IVFSearchResult(
            indices: outputIndices,
            distances: outputDistances,
            numQueries: numQueries,
            k: k,
            executionTime: totalTime,
            phaseTimings: phaseTimings
        )
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
