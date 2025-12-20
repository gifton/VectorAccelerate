//
//  IVFListSearchKernel.swift
//  VectorAccelerate
//
//  GPU-based IVF list search kernel wrapper.
//
//  This replaces the CPU candidate gathering path by searching directly over the
//  inverted list buffers (CSR-style) already resident in GPU memory.
//
//  Performance: Eliminates the 27x CPU overhead from gather-and-search path,
//  enabling IVF to be faster than flat search at scale.
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Parameters

/// Parameters for the IVF list search kernel.
///
/// Must match the `IVFSearchParams` struct in `IVFListSearch.metal`.
public struct IVFListSearchParameters: Sendable {
    public var numQueries: UInt32
    public var numCentroids: UInt32
    public var dimension: UInt32
    public var nprobe: UInt32
    public var k: UInt32
    public var maxCandidatesPerQuery: UInt32

    public init(
        numQueries: Int,
        numCentroids: Int,
        dimension: Int,
        nprobe: Int,
        k: Int,
        maxCandidatesPerQuery: Int
    ) {
        self.numQueries = UInt32(numQueries)
        self.numCentroids = UInt32(numCentroids)
        self.dimension = UInt32(dimension)
        self.nprobe = UInt32(nprobe)
        self.k = UInt32(k)
        self.maxCandidatesPerQuery = UInt32(maxCandidatesPerQuery)
    }
}

// MARK: - Kernel

/// Metal wrapper for `ivf_list_search`.
///
/// This kernel performs IVF search directly on GPU by:
/// 1. Reading selected cluster IDs from coarse quantization
/// 2. Iterating through vectors in those clusters using CSR offsets
/// 3. Computing L2 distances and maintaining per-thread top-K heaps
/// 4. Merging heaps across threads to produce final results
///
/// ## Performance Characteristics
/// - One threadgroup per query (256 threads)
/// - Query cached in threadgroup memory for dimension ≤ 2048
/// - Per-thread heap size: 8 candidates
/// - Two top-K paths: parallel min-reduce for K≤32, bitonic sort for K>32
public final class IVFListSearchKernel: @unchecked Sendable, Metal4Kernel {
    public let context: Metal4Context
    public let name: String = "IVFListSearchKernel"

    private let pipeline: any MTLComputePipelineState

    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()
        guard let function = library.makeFunction(name: "ivf_list_search") else {
            throw VectorError.shaderNotFound(
                name: "ivf_list_search - ensure IVFListSearch.metal is compiled"
            )
        }

        let device = context.device.rawDevice
        self.pipeline = try await device.makeComputePipelineState(function: function)
    }

    public func warmUp() async throws {
        // Pipeline state compilation happens in init.
        // Optionally run a small dispatch to warm GPU caches.
    }

    // MARK: - Encode

    /// Encode the IVF list search kernel into an existing encoder.
    ///
    /// - Parameters:
    ///   - encoder: Compute command encoder to encode into
    ///   - queries: Query vectors buffer [Q × D]
    ///   - structure: IVF GPU structure with list vectors, offsets, indices
    ///   - selectedLists: Coarse quantizer output [Q × nprobe]
    ///   - outputIndices: Output buffer for indices [Q × K]
    ///   - outputDistances: Output buffer for distances [Q × K]
    ///   - parameters: Search parameters
    /// - Returns: Encoding result for profiling
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        structure: IVFGPUIndexStructure,
        selectedLists: any MTLBuffer,
        outputIndices: any MTLBuffer,
        outputDistances: any MTLBuffer,
        parameters: IVFListSearchParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "IVFListSearch (nprobe=\(parameters.nprobe), K=\(parameters.k))"

        // Bind buffers (must match Metal argument indices)
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(structure.listVectors, offset: 0, index: 1)
        encoder.setBuffer(structure.listOffsets, offset: 0, index: 2)
        encoder.setBuffer(structure.vectorIndices, offset: 0, index: 3)
        encoder.setBuffer(selectedLists, offset: 0, index: 4)
        encoder.setBuffer(outputIndices, offset: 0, index: 5)
        encoder.setBuffer(outputDistances, offset: 0, index: 6)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<IVFListSearchParameters>.size, index: 7)

        // Dispatch: one threadgroup per query
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: Int(parameters.numQueries), height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "ivf_list_search",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    // MARK: - Execute

    /// Execute IVF list search.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer [Q × D]
    ///   - structure: IVF GPU structure containing list vectors and CSR offsets
    ///   - selectedLists: Coarse quantizer result [Q × nprobe] centroid IDs
    ///   - numQueries: Number of queries (Q)
    ///   - nprobe: Number of lists per query
    ///   - k: Top-K to return
    /// - Returns: Tuple of (indices, distances) buffers, each sized [Q × K]
    /// - Throws: `VectorError` on invalid input or execution failure
    public func search(
        queries: any MTLBuffer,
        structure: IVFGPUIndexStructure,
        selectedLists: any MTLBuffer,
        numQueries: Int,
        nprobe: Int,
        k: Int
    ) async throws -> (indices: any MTLBuffer, distances: any MTLBuffer) {
        let device = context.device.rawDevice

        // Validate inputs
        guard numQueries > 0 else {
            throw VectorError.invalidInput("numQueries must be > 0")
        }
        guard nprobe > 0 else {
            throw VectorError.invalidInput("nprobe must be > 0")
        }
        guard k > 0 else {
            throw VectorError.invalidInput("k must be > 0")
        }
        guard structure.dimension > 0 else {
            throw VectorError.invalidInput("structure.dimension must be > 0")
        }

        // Validate buffer sizes
        let expectedQueriesBytes = numQueries * structure.dimension * MemoryLayout<Float>.size
        guard queries.length >= expectedQueriesBytes else {
            throw VectorError.invalidInput(
                "queries buffer too small: expected >= \(expectedQueriesBytes) bytes, got \(queries.length)"
            )
        }

        let expectedSelectedListsBytes = numQueries * nprobe * MemoryLayout<UInt32>.size
        guard selectedLists.length >= expectedSelectedListsBytes else {
            throw VectorError.invalidInput(
                "selectedLists buffer too small: expected >= \(expectedSelectedListsBytes) bytes, got \(selectedLists.length)"
            )
        }

        let expectedOffsetsBytes = (structure.numCentroids + 1) * MemoryLayout<UInt32>.size
        guard structure.listOffsets.length >= expectedOffsetsBytes else {
            throw VectorError.invalidInput(
                "listOffsets buffer too small: expected >= \(expectedOffsetsBytes) bytes, got \(structure.listOffsets.length)"
            )
        }

        let expectedVectorIndicesBytes = structure.totalVectors * MemoryLayout<UInt32>.size
        guard structure.vectorIndices.length >= expectedVectorIndicesBytes else {
            throw VectorError.invalidInput(
                "vectorIndices buffer too small: expected >= \(expectedVectorIndicesBytes) bytes, got \(structure.vectorIndices.length)"
            )
        }

        let expectedListVectorsBytes = structure.totalVectors * structure.dimension * MemoryLayout<Float>.size
        guard structure.listVectors.length >= expectedListVectorsBytes else {
            throw VectorError.invalidInput(
                "listVectors buffer too small: expected >= \(expectedListVectorsBytes) bytes, got \(structure.listVectors.length)"
            )
        }

        // Allocate output buffers
        let outputCount = numQueries * k
        let indicesSize = outputCount * MemoryLayout<UInt32>.size
        let distancesSize = outputCount * MemoryLayout<Float>.size

        guard let outputIndices = device.makeBuffer(length: indicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: indicesSize)
        }
        outputIndices.label = "IVFListSearch.outputIndices"

        guard let outputDistances = device.makeBuffer(length: distancesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distancesSize)
        }
        outputDistances.label = "IVFListSearch.outputDistances"

        // Build parameters
        let params = IVFListSearchParameters(
            numQueries: numQueries,
            numCentroids: structure.numCentroids,
            dimension: structure.dimension,
            nprobe: nprobe,
            k: k,
            maxCandidatesPerQuery: structure.totalVectors  // Conservative upper bound
        )

        // Execute
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                queries: queries,
                structure: structure,
                selectedLists: selectedLists,
                outputIndices: outputIndices,
                outputDistances: outputDistances,
                parameters: params
            )
        }

        return (indices: outputIndices, distances: outputDistances)
    }
}
