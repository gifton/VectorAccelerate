//
//  IVFGPUCandidateBuilderKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for GPU-native IVF candidate list construction.
//
//  This kernel eliminates the CPU bottleneck in IVF search by building
//  candidate lists entirely on GPU. It replaces the CPU loop that previously
//  gathered candidate indices from selected IVF lists.
//
//  The kernel supports two modes:
//  1. Three-pass: count → prefix sum → build (deterministic ordering)
//  2. Fused: single pass with atomics (faster but non-deterministic ordering)
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Parameters

/// Parameters for IVF candidate counting kernel.
public struct IVFCandidateCountParameters: Sendable {
    public let numQueries: UInt32
    public let nprobe: UInt32
    public let numLists: UInt32
    private let padding: UInt32 = 0

    public init(numQueries: Int, nprobe: Int, numLists: Int) {
        self.numQueries = UInt32(numQueries)
        self.nprobe = UInt32(nprobe)
        self.numLists = UInt32(numLists)
    }
}

/// Parameters for IVF candidate build kernel.
public struct IVFCandidateBuildParameters: Sendable {
    public let numQueries: UInt32
    public let nprobe: UInt32
    public let numLists: UInt32
    public let totalCandidates: UInt32

    public init(numQueries: Int, nprobe: Int, numLists: Int, totalCandidates: Int) {
        self.numQueries = UInt32(numQueries)
        self.nprobe = UInt32(nprobe)
        self.numLists = UInt32(numLists)
        self.totalCandidates = UInt32(totalCandidates)
    }
}

/// Parameters for prefix sum kernel.
public struct PrefixSumParameters: Sendable {
    public let numElements: UInt32
    private let padding: (UInt32, UInt32, UInt32) = (0, 0, 0)

    public init(numElements: Int) {
        self.numElements = UInt32(numElements)
    }
}

// MARK: - Result

/// Result from GPU candidate builder.
public struct IVFGPUCandidateResult: Sendable {
    public let indicesToken: BufferToken?
    public let queryIdsToken: BufferToken?
    public let offsetsToken: BufferToken?
    
    /// IVF entry indices [totalCandidates]
    public var candidateIVFIndices: any MTLBuffer {
        if let token = indicesToken { return token.buffer }
        return rawIndices!
    }
    private let rawIndices: (any MTLBuffer)?

    /// Query ID for each candidate [totalCandidates]
    public var candidateQueryIds: any MTLBuffer {
        if let token = queryIdsToken { return token.buffer }
        return rawQueryIds!
    }
    private let rawQueryIds: (any MTLBuffer)?

    /// Per-query offsets [numQueries + 1] (CSR format)
    public var candidateOffsets: any MTLBuffer {
        if let token = offsetsToken { return token.buffer }
        return rawOffsets!
    }
    private let rawOffsets: (any MTLBuffer)?

    /// Total number of candidates
    public let totalCandidates: Int

    /// Number of queries
    public let numQueries: Int
    
    public init(indicesToken: BufferToken, queryIdsToken: BufferToken, offsetsToken: BufferToken, totalCandidates: Int, numQueries: Int) {
        self.indicesToken = indicesToken
        self.queryIdsToken = queryIdsToken
        self.offsetsToken = offsetsToken
        self.rawIndices = nil
        self.rawQueryIds = nil
        self.rawOffsets = nil
        self.totalCandidates = totalCandidates
        self.numQueries = numQueries
    }

    public init(candidateIVFIndices: any MTLBuffer, candidateQueryIds: any MTLBuffer, candidateOffsets: any MTLBuffer, totalCandidates: Int, numQueries: Int) {
        self.indicesToken = nil
        self.queryIdsToken = nil
        self.offsetsToken = nil
        self.rawIndices = candidateIVFIndices
        self.rawQueryIds = candidateQueryIds
        self.rawOffsets = candidateOffsets
        self.totalCandidates = totalCandidates
        self.numQueries = numQueries
    }

    /// Get offset range for a specific query.
    public func offsetRange(for queryIndex: Int) -> Range<Int> {
        guard queryIndex >= 0 && queryIndex < numQueries else {
            return 0..<0  // Return empty range for invalid query index
        }
        let ptr = candidateOffsets.contents().bindMemory(to: UInt32.self, capacity: numQueries + 1)
        let start = Int(ptr[queryIndex])
        let end = Int(ptr[queryIndex + 1])
        // Ensure valid range (end >= start)
        guard end >= start else {
            return start..<start  // Return empty range if data is corrupted
        }
        return start..<end
    }
}

// MARK: - Kernel

/// Metal 4 kernel for GPU-native IVF candidate list construction.
///
/// Builds candidate lists entirely on GPU, eliminating the CPU bottleneck
/// in IVF search. Takes coarse quantization results and IVF list structure
/// as input, produces flat candidate indices with per-query offsets.
///
/// ## Usage
/// ```swift
/// let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
/// let result = try await kernel.buildCandidates(
///     nearestCentroids: coarseResult.listIndices,
///     listOffsets: structure.listOffsets,
///     numQueries: 100,
///     nprobe: 8,
///     numLists: 256
/// )
/// ```
public final class IVFGPUCandidateBuilderKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "IVFGPUCandidateBuilderKernel"

    // MARK: - Pipelines

    private let countPipeline: any MTLComputePipelineState
    private let buildPipeline: any MTLComputePipelineState
    private let prefixSumSequentialPipeline: any MTLComputePipelineState
    private let fusedPipeline: (any MTLComputePipelineState)?

    // MARK: - Configuration

    /// Maximum number of queries for fused kernel (uses atomics)
    private static let fusedMaxQueries: Int = 256

    /// Threshold to switch from sequential to parallel prefix sum
    private static let parallelPrefixSumThreshold: Int = 64

    // MARK: - Initialization

    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let countFunc = library.makeFunction(name: "ivf_count_candidates") else {
            throw VectorError.shaderNotFound(
                name: "ivf_count_candidates. Ensure IVFCandidateBuilder.metal is compiled."
            )
        }

        guard let buildFunc = library.makeFunction(name: "ivf_build_candidates") else {
            throw VectorError.shaderNotFound(
                name: "ivf_build_candidates. Ensure IVFCandidateBuilder.metal is compiled."
            )
        }

        guard let prefixSumFunc = library.makeFunction(name: "ivf_prefix_sum_sequential") else {
            throw VectorError.shaderNotFound(
                name: "ivf_prefix_sum_sequential. Ensure IVFCandidateBuilder.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.countPipeline = try await device.makeComputePipelineState(function: countFunc)
        self.buildPipeline = try await device.makeComputePipelineState(function: buildFunc)
        self.prefixSumSequentialPipeline = try await device.makeComputePipelineState(function: prefixSumFunc)

        // Fused kernel is optional (may not exist in all metallib versions)
        if let fusedFunc = library.makeFunction(name: "ivf_build_candidates_fused") {
            self.fusedPipeline = try? await device.makeComputePipelineState(function: fusedFunc)
        } else {
            self.fusedPipeline = nil
        }
    }

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Build Candidates

    /// Build candidate lists from coarse quantization results.
    ///
    /// - Parameters:
    ///   - nearestCentroids: Buffer with nearest centroid indices [numQueries × nprobe]
    ///   - listOffsets: IVF list offsets in CSR format [numLists + 1]
    ///   - numQueries: Number of queries
    ///   - nprobe: Number of probes per query
    ///   - numLists: Total number of IVF lists (nlist)
    ///   - maxCandidatesPerQuery: Optional hint for pre-allocation
    /// - Returns: GPU candidate result with indices, query IDs, and offsets
    public func buildCandidates(
        nearestCentroids: any MTLBuffer,
        listOffsets: any MTLBuffer,
        numQueries: Int,
        nprobe: Int,
        numLists: Int,
        maxCandidatesPerQuery: Int? = nil
    ) async throws -> IVFGPUCandidateResult {
        // Use fused kernel for small query batches (faster due to single pass)
        if numQueries <= Self.fusedMaxQueries, let _ = fusedPipeline {
            return try await buildCandidatesFused(
                nearestCentroids: nearestCentroids,
                listOffsets: listOffsets,
                numQueries: numQueries,
                nprobe: nprobe,
                numLists: numLists,
                maxCandidatesPerQuery: maxCandidatesPerQuery
            )
        }

        // Use three-pass kernel for larger batches (deterministic ordering)
        return try await buildCandidatesThreePass(
            nearestCentroids: nearestCentroids,
            listOffsets: listOffsets,
            numQueries: numQueries,
            nprobe: nprobe,
            numLists: numLists
        )
    }

    // MARK: - Three-Pass Implementation

    private func buildCandidatesThreePass(
        nearestCentroids: any MTLBuffer,
        listOffsets: any MTLBuffer,
        numQueries: Int,
        nprobe: Int,
        numLists: Int
    ) async throws -> IVFGPUCandidateResult {
        // Step 1: Count candidates per query
        let countsBytes = numQueries * MemoryLayout<UInt32>.size
        let candidateCountsToken = try await context.getBuffer(size: countsBytes)
        let candidateCounts = candidateCountsToken.buffer
        candidateCounts.label = "IVFCandidateBuilder.counts"

        let countParams = IVFCandidateCountParameters(
            numQueries: numQueries,
            nprobe: nprobe,
            numLists: numLists
        )

        try await context.executeAndWait { [self] _, encoder in
            self.encodeCount(
                into: encoder,
                nearestCentroids: nearestCentroids,
                listOffsets: listOffsets,
                candidateCounts: candidateCounts,
                parameters: countParams
            )
        }

        // Step 2: Prefix sum to compute offsets
        let offsetsBytes = (numQueries + 1) * MemoryLayout<UInt32>.size
        let candidateOffsetsToken = try await context.getBuffer(size: offsetsBytes)
        let candidateOffsets = candidateOffsetsToken.buffer
        candidateOffsets.label = "IVFCandidateBuilder.offsets"

        let prefixParams = PrefixSumParameters(numElements: numQueries)

        try await context.executeAndWait { [self] _, encoder in
            self.encodePrefixSum(
                into: encoder,
                candidateCounts: candidateCounts,
                candidateOffsets: candidateOffsets,
                parameters: prefixParams
            )
        }

        // Read total candidates from offsets[numQueries]
        let offsetsPtr = candidateOffsets.contents().bindMemory(to: UInt32.self, capacity: numQueries + 1)
        let totalCandidates = Int(offsetsPtr[numQueries])

        if totalCandidates == 0 {
            // No candidates - return empty result
            let emptyIndicesToken = try await context.getBuffer(size: 4)
            let emptyQueryIdsToken = try await context.getBuffer(size: 4)
            emptyIndicesToken.buffer.label = "IVFCandidateBuilder.indices.empty"
            emptyQueryIdsToken.buffer.label = "IVFCandidateBuilder.queryIds.empty"

            return IVFGPUCandidateResult(
                indicesToken: emptyIndicesToken,
                queryIdsToken: emptyQueryIdsToken,
                offsetsToken: candidateOffsetsToken,
                totalCandidates: 0,
                numQueries: numQueries
            )
        }

        // Step 3: Build candidate lists
        let indicesBytes = totalCandidates * MemoryLayout<UInt32>.size
        let queryIdsBytes = totalCandidates * MemoryLayout<UInt32>.size
        let candidateIVFIndicesToken = try await context.getBuffer(size: indicesBytes)
        let candidateQueryIdsToken = try await context.getBuffer(size: queryIdsBytes)
        let candidateIVFIndices = candidateIVFIndicesToken.buffer
        let candidateQueryIds = candidateQueryIdsToken.buffer
        candidateIVFIndices.label = "IVFCandidateBuilder.indices"
        candidateQueryIds.label = "IVFCandidateBuilder.queryIds"

        let buildParams = IVFCandidateBuildParameters(
            numQueries: numQueries,
            nprobe: nprobe,
            numLists: numLists,
            totalCandidates: totalCandidates
        )

        try await context.executeAndWait { [self] _, encoder in
            self.encodeBuild(
                into: encoder,
                nearestCentroids: nearestCentroids,
                listOffsets: listOffsets,
                candidateOffsets: candidateOffsets,
                candidateIVFIndices: candidateIVFIndices,
                candidateQueryIds: candidateQueryIds,
                parameters: buildParams
            )
        }

        return IVFGPUCandidateResult(
            indicesToken: candidateIVFIndicesToken,
            queryIdsToken: candidateQueryIdsToken,
            offsetsToken: candidateOffsetsToken,
            totalCandidates: totalCandidates,
            numQueries: numQueries
        )
    }

    // MARK: - Fused Implementation

    private func buildCandidatesFused(
        nearestCentroids: any MTLBuffer,
        listOffsets: any MTLBuffer,
        numQueries: Int,
        nprobe: Int,
        numLists: Int,
        maxCandidatesPerQuery: Int?
    ) async throws -> IVFGPUCandidateResult {
        guard let fusedPipeline = fusedPipeline else {
            return try await buildCandidatesThreePass(
                nearestCentroids: nearestCentroids,
                listOffsets: listOffsets,
                numQueries: numQueries,
                nprobe: nprobe,
                numLists: numLists
            )
        }

        // Estimate max candidates (conservative)
        // Read listOffsets to get average list size
        let offsetsPtr = listOffsets.contents().bindMemory(to: UInt32.self, capacity: numLists + 1)
        let totalIVFEntries = Int(offsetsPtr[numLists])
        let avgListSize = totalIVFEntries / max(1, numLists)
        let estimatedMax = maxCandidatesPerQuery ?? (nprobe * avgListSize * 2)  // 2x for safety
        let maxTotalCandidates = numQueries * estimatedMax

        // Allocate output buffers
        let indicesBytes = maxTotalCandidates * MemoryLayout<UInt32>.size
        let queryIdsBytes = maxTotalCandidates * MemoryLayout<UInt32>.size
        let offsetsBytes = numQueries * MemoryLayout<UInt32>.size
        let countsBytes = numQueries * MemoryLayout<UInt32>.size
        let atomicBytes = MemoryLayout<UInt32>.size

        let candidateIVFIndicesToken = try await context.getBuffer(size: max(indicesBytes, 4))
        let candidateQueryIdsToken = try await context.getBuffer(size: max(queryIdsBytes, 4))
        let perQueryOffsetsToken = try await context.getBuffer(size: offsetsBytes)
        let perQueryCountsToken = try await context.getBuffer(size: countsBytes)
        let totalCounterToken = try await context.getBuffer(size: atomicBytes)

        let candidateIVFIndices = candidateIVFIndicesToken.buffer
        let candidateQueryIds = candidateQueryIdsToken.buffer
        let perQueryOffsets = perQueryOffsetsToken.buffer
        let perQueryCounts = perQueryCountsToken.buffer
        let totalCounter = totalCounterToken.buffer

        candidateIVFIndices.label = "IVFCandidateBuilder.fused.indices"
        candidateQueryIds.label = "IVFCandidateBuilder.fused.queryIds"
        perQueryOffsets.label = "IVFCandidateBuilder.fused.offsets"
        perQueryCounts.label = "IVFCandidateBuilder.fused.counts"
        totalCounter.label = "IVFCandidateBuilder.fused.counter"

        // Initialize atomic counter to 0
        let counterPtr = totalCounter.contents().bindMemory(to: UInt32.self, capacity: 1)
        counterPtr[0] = 0

        let countParams = IVFCandidateCountParameters(
            numQueries: numQueries,
            nprobe: nprobe,
            numLists: numLists
        )

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(fusedPipeline)
            encoder.label = "IVFCandidateBuilder.fused"

            encoder.setBuffer(nearestCentroids, offset: 0, index: 0)
            encoder.setBuffer(listOffsets, offset: 0, index: 1)
            encoder.setBuffer(candidateIVFIndices, offset: 0, index: 2)
            encoder.setBuffer(candidateQueryIds, offset: 0, index: 3)
            encoder.setBuffer(totalCounter, offset: 0, index: 4)
            encoder.setBuffer(perQueryOffsets, offset: 0, index: 5)
            encoder.setBuffer(perQueryCounts, offset: 0, index: 6)

            var params = countParams
            encoder.setBytes(&params, length: MemoryLayout<IVFCandidateCountParameters>.size, index: 7)

            let config = Metal4ThreadConfiguration.linear(count: numQueries, pipeline: fusedPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }

        // Read actual total from atomic counter
        let actualTotal = Int(counterPtr[0])

        // Build proper CSR offsets from perQueryOffsets and perQueryCounts
        let csrOffsetsBytes = (numQueries + 1) * MemoryLayout<UInt32>.size
        let candidateOffsetsToken = try await context.getBuffer(size: csrOffsetsBytes)
        let candidateOffsets = candidateOffsetsToken.buffer
        candidateOffsets.label = "IVFCandidateBuilder.fused.csrOffsets"

        // Build CSR offsets on CPU (small data)
        // The fused kernel provides per-query offsets and counts directly.
        // CSR format: offset[q] = start of query q's candidates, offset[q+1] = end
        let csrPtr = candidateOffsets.contents().bindMemory(to: UInt32.self, capacity: numQueries + 1)
        let fusedOffsetsPtr = perQueryOffsets.contents().bindMemory(to: UInt32.self, capacity: numQueries)

        // Build CSR offsets directly from fused kernel output
        // (existing code does this right after...)
        // offset[q] = start position for query q, offset[q+1] = start + count
        for q in 0..<numQueries {
            csrPtr[q] = fusedOffsetsPtr[q]
        }
        // Final offset is the total count
        csrPtr[numQueries] = UInt32(actualTotal)

        return IVFGPUCandidateResult(
            indicesToken: candidateIVFIndicesToken,
            queryIdsToken: candidateQueryIdsToken,
            offsetsToken: candidateOffsetsToken,
            totalCandidates: actualTotal,
            numQueries: numQueries
        )
    }

    // MARK: - Encode Methods

    @discardableResult
    private func encodeCount(
        into encoder: any MTLComputeCommandEncoder,
        nearestCentroids: any MTLBuffer,
        listOffsets: any MTLBuffer,
        candidateCounts: any MTLBuffer,
        parameters: IVFCandidateCountParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(countPipeline)
        encoder.label = "IVFCandidateBuilder.count"

        encoder.setBuffer(nearestCentroids, offset: 0, index: 0)
        encoder.setBuffer(listOffsets, offset: 0, index: 1)
        encoder.setBuffer(candidateCounts, offset: 0, index: 2)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<IVFCandidateCountParameters>.size, index: 3)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numQueries),
            pipeline: countPipeline
        )
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "ivf_count_candidates",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    @discardableResult
    private func encodePrefixSum(
        into encoder: any MTLComputeCommandEncoder,
        candidateCounts: any MTLBuffer,
        candidateOffsets: any MTLBuffer,
        parameters: PrefixSumParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(prefixSumSequentialPipeline)
        encoder.label = "IVFCandidateBuilder.prefixSum"

        encoder.setBuffer(candidateCounts, offset: 0, index: 0)
        encoder.setBuffer(candidateOffsets, offset: 0, index: 1)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<PrefixSumParameters>.size, index: 2)

        // Sequential prefix sum uses single thread
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "ivf_prefix_sum_sequential",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    @discardableResult
    private func encodeBuild(
        into encoder: any MTLComputeCommandEncoder,
        nearestCentroids: any MTLBuffer,
        listOffsets: any MTLBuffer,
        candidateOffsets: any MTLBuffer,
        candidateIVFIndices: any MTLBuffer,
        candidateQueryIds: any MTLBuffer,
        parameters: IVFCandidateBuildParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(buildPipeline)
        encoder.label = "IVFCandidateBuilder.build"

        encoder.setBuffer(nearestCentroids, offset: 0, index: 0)
        encoder.setBuffer(listOffsets, offset: 0, index: 1)
        encoder.setBuffer(candidateOffsets, offset: 0, index: 2)
        encoder.setBuffer(candidateIVFIndices, offset: 0, index: 3)
        encoder.setBuffer(candidateQueryIds, offset: 0, index: 4)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<IVFCandidateBuildParameters>.size, index: 5)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numQueries),
            pipeline: buildPipeline
        )
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "ivf_build_candidates",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }
}
