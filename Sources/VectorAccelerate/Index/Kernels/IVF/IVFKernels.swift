//
//  IVFKernels.swift
//  VectorAccelerate
//
//  Metal 4 kernels for IVF (Inverted File) index operations.
//
//  ## Architecture
//
//  IVF search is implemented as a three-phase pipeline:
//  1. Coarse Quantization: Find nprobe nearest centroids (reuses FusedL2TopKKernel)
//  2. List Search: Search within selected inverted lists
//  3. Candidate Merge: Merge candidates into final top-k (reuses TopKSelectionKernel)
//
//  ## Design Notes
//
//  Configuration types (*Configuration) are developer-facing with validation and presets.
//  Shader argument types (*ShaderArgs) are GPU-bound with strict memory layout.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - IVF Search Configuration

/// Configuration for IVF search operations.
public struct IVFSearchConfiguration: Sendable {
    /// Number of centroids/clusters (nlist)
    public let numCentroids: Int

    /// Number of lists to probe during search (nprobe)
    public let nprobe: Int

    /// Vector dimension
    public let dimension: Int

    /// Distance metric
    public let metric: SupportedDistanceMetric

    /// Enable profiling for performance analysis
    public let enableProfiling: Bool

    /// Use GPU-accelerated candidate list building.
    ///
    /// When enabled (default), candidate lists are built entirely on GPU,
    /// eliminating CPU-GPU synchronization during the search pipeline.
    /// Falls back to CPU candidate building if GPU kernels are unavailable.
    public let useGPUCandidateBuilder: Bool

    public init(
        numCentroids: Int,
        nprobe: Int = 8,
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        enableProfiling: Bool = false,
        useGPUCandidateBuilder: Bool = true
    ) {
        self.numCentroids = numCentroids
        self.nprobe = min(nprobe, numCentroids)
        self.dimension = dimension
        self.metric = metric
        self.enableProfiling = enableProfiling
        self.useGPUCandidateBuilder = useGPUCandidateBuilder
    }

    /// Validate configuration parameters.
    public func validate() throws {
        guard numCentroids >= 1 else {
            throw IndexError.invalidInput(message: "numCentroids must be at least 1, got \(numCentroids)")
        }
        guard nprobe >= 1 else {
            throw IndexError.invalidInput(message: "nprobe must be at least 1, got \(nprobe)")
        }
        guard nprobe <= numCentroids else {
            throw IndexError.invalidInput(message: "nprobe (\(nprobe)) cannot exceed numCentroids (\(numCentroids))")
        }
        guard dimension >= 1 && dimension <= 4096 else {
            throw IndexError.invalidInput(message: "dimension must be 1-4096, got \(dimension)")
        }
    }

    // MARK: - Presets

    /// Small index preset (256 centroids, probe 8)
    public static func small(dimension: Int, metric: SupportedDistanceMetric = .euclidean, useGPUCandidateBuilder: Bool = true) -> IVFSearchConfiguration {
        IVFSearchConfiguration(numCentroids: 256, nprobe: 8, dimension: dimension, metric: metric, useGPUCandidateBuilder: useGPUCandidateBuilder)
    }

    /// Standard preset (1024 centroids, probe 16)
    public static func standard(dimension: Int, metric: SupportedDistanceMetric = .euclidean, useGPUCandidateBuilder: Bool = true) -> IVFSearchConfiguration {
        IVFSearchConfiguration(numCentroids: 1024, nprobe: 16, dimension: dimension, metric: metric, useGPUCandidateBuilder: useGPUCandidateBuilder)
    }

    /// Large index preset (4096 centroids, probe 32)
    public static func large(dimension: Int, metric: SupportedDistanceMetric = .euclidean, useGPUCandidateBuilder: Bool = true) -> IVFSearchConfiguration {
        IVFSearchConfiguration(numCentroids: 4096, nprobe: 32, dimension: dimension, metric: metric, useGPUCandidateBuilder: useGPUCandidateBuilder)
    }

    /// High recall preset - probes more lists
    public static func highRecall(numCentroids: Int, dimension: Int, metric: SupportedDistanceMetric = .euclidean, useGPUCandidateBuilder: Bool = true) -> IVFSearchConfiguration {
        IVFSearchConfiguration(numCentroids: numCentroids, nprobe: max(numCentroids / 8, 16), dimension: dimension, metric: metric, useGPUCandidateBuilder: useGPUCandidateBuilder)
    }
}

// MARK: - Inverted List Metadata

/// Metadata for a single inverted list.
public struct IVFListMetadata: Sendable {
    /// List index (centroid ID)
    public let listIndex: Int

    /// Number of vectors in this list
    public let count: Int

    /// Offset into the flattened vector storage
    public let vectorOffset: Int

    /// Offset into the ID storage
    public let idOffset: Int

    public init(listIndex: Int, count: Int, vectorOffset: Int, idOffset: Int) {
        self.listIndex = listIndex
        self.count = count
        self.vectorOffset = vectorOffset
        self.idOffset = idOffset
    }
}

/// GPU-friendly representation of inverted list structure.
/// Uses CSR (Compressed Sparse Row) format for efficient access.
public struct IVFListStructure: Sendable {
    /// Flattened vector storage [totalVectors × dimension]
    public let vectors: any MTLBuffer

    /// Vector IDs [totalVectors] - maps position to original ID
    public let vectorIDs: any MTLBuffer

    /// List offsets [numLists + 1] - CSR row pointers
    public let listOffsets: any MTLBuffer

    /// Total number of lists
    public let numLists: Int

    /// Total number of vectors across all lists
    public let totalVectors: Int

    /// Vector dimension
    public let dimension: Int

    public init(
        vectors: any MTLBuffer,
        vectorIDs: any MTLBuffer,
        listOffsets: any MTLBuffer,
        numLists: Int,
        totalVectors: Int,
        dimension: Int
    ) {
        self.vectors = vectors
        self.vectorIDs = vectorIDs
        self.listOffsets = listOffsets
        self.numLists = numLists
        self.totalVectors = totalVectors
        self.dimension = dimension
    }

    /// Get the count of vectors in a specific list.
    public func listCount(at index: Int) -> Int {
        guard index < numLists else { return 0 }
        let offsets = listOffsets.contents().bindMemory(to: UInt32.self, capacity: numLists + 1)
        return Int(offsets[index + 1]) - Int(offsets[index])
    }
}

// MARK: - Shader Arguments

/// Shader arguments for IVF coarse quantization.
/// Memory layout matches `IVFCoarseQuantizerArgs` in IVFShaders.metal.
public struct IVFCoarseQuantizerShaderArgs: Sendable {
    /// Query vector dimension
    public let dimension: UInt32

    /// Number of query vectors
    public let numQueries: UInt32

    /// Number of centroids
    public let numCentroids: UInt32

    /// Number of lists to probe
    public let nprobe: UInt32

    public init(dimension: Int, numQueries: Int, numCentroids: Int, nprobe: Int) {
        self.dimension = UInt32(dimension)
        self.numQueries = UInt32(numQueries)
        self.numCentroids = UInt32(numCentroids)
        self.nprobe = UInt32(nprobe)
    }
}

/// Shader arguments for IVF list search.
/// Memory layout matches `IVFListSearchArgs` in IVFShaders.metal.
public struct IVFListSearchShaderArgs: Sendable {
    /// Query vector dimension
    public let dimension: UInt32

    /// Number of results to return per query
    public let k: UInt32

    /// Number of queries
    public let numQueries: UInt32

    /// Total number of candidate vectors
    public let numCandidates: UInt32

    public init(dimension: Int, k: Int, numQueries: Int, numCandidates: Int) {
        self.dimension = UInt32(dimension)
        self.k = UInt32(k)
        self.numQueries = UInt32(numQueries)
        self.numCandidates = UInt32(numCandidates)
    }
}

/// Shader arguments for candidate merging.
/// Memory layout matches `IVFMergeArgs` in IVFShaders.metal.
public struct IVFMergeShaderArgs: Sendable {
    /// Number of queries
    public let numQueries: UInt32

    /// Number of candidate sets to merge (nprobe)
    public let numSets: UInt32

    /// Candidates per set (k from list search)
    public let candidatesPerSet: UInt32

    /// Final k to output
    public let finalK: UInt32

    public init(numQueries: Int, numSets: Int, candidatesPerSet: Int, finalK: Int) {
        self.numQueries = UInt32(numQueries)
        self.numSets = UInt32(numSets)
        self.candidatesPerSet = UInt32(candidatesPerSet)
        self.finalK = UInt32(finalK)
    }
}

// MARK: - IVF Search Result

/// Result from IVF search operation.
public struct IVFSearchResult: Sendable {
    /// Indices of nearest neighbors [numQueries × k]
    public let indices: any MTLBuffer

    /// Distances to nearest neighbors [numQueries × k]
    public let distances: any MTLBuffer

    /// Number of queries
    public let numQueries: Int

    /// K value used
    public let k: Int

    /// Execution time for the search
    public let executionTime: TimeInterval

    /// Breakdown of phase timings (if profiling enabled)
    public let phaseTimings: IVFPhaseTimings?

    /// Extract results for a specific query.
    public func results(for queryIndex: Int) -> [(index: Int, distance: Float)] {
        guard queryIndex < numQueries else { return [] }

        let offset = queryIndex * k
        let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: numQueries * k)
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: numQueries * k)

        var results: [(index: Int, distance: Float)] = []
        results.reserveCapacity(k)

        for i in 0..<k {
            let idx = indexPtr[offset + i]
            if idx != 0xFFFFFFFF {  // Skip sentinel values
                results.append((index: Int(idx), distance: distPtr[offset + i]))
            }
        }
        return results
    }

    /// Get all results as array of arrays.
    public func allResults() -> [[(index: Int, distance: Float)]] {
        (0..<numQueries).map { results(for: $0) }
    }
}

/// Timing breakdown for IVF search phases.
public struct IVFPhaseTimings: Sendable {
    /// Time for coarse quantization phase
    public let coarseQuantization: TimeInterval

    /// Time for list search phase
    public let listSearch: TimeInterval

    /// Time for candidate merge phase
    public let candidateMerge: TimeInterval

    /// Time for buffer operations (allocation, copy)
    public let bufferOperations: TimeInterval

    /// Total GPU time
    public var totalGPUTime: TimeInterval {
        coarseQuantization + listSearch + candidateMerge
    }
}

// MARK: - Coarse Quantization Result

/// Result from coarse quantization phase.
public struct IVFCoarseResult: Sendable {
    /// Selected list indices [numQueries × nprobe]
    public let listIndices: any MTLBuffer

    /// Distances to selected centroids [numQueries × nprobe]
    public let listDistances: any MTLBuffer

    /// Number of queries
    public let numQueries: Int

    /// Number of lists probed per query
    public let nprobe: Int

    /// Get selected lists for a query.
    public func selectedLists(for queryIndex: Int) -> [Int] {
        guard queryIndex < numQueries else { return [] }

        let offset = queryIndex * nprobe
        let indexPtr = listIndices.contents().bindMemory(to: UInt32.self, capacity: numQueries * nprobe)

        return (0..<nprobe).map { Int(indexPtr[offset + $0]) }
    }
}
