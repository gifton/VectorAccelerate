//
//  IVFIndirectionDistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for IVF candidate distance computation with indirection.
//
//  This kernel computes L2 squared distances for a per-query candidate set where
//  each candidate is represented as an IVF entry index. The IVF entry index must
//  be mapped through `vectorIndices` to obtain the underlying storage slot in the
//  global vector buffer.
//
//  The key goal is to eliminate CPU-side vector gathering and fix the IVF data
//  layout mismatch between IVFStructure.prepareGPUStructure() (global storage
//  buffer) and IVFSearchPipeline (previously assuming a flattened contiguous IVF
//  buffer).
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Parameters

/// Parameters for IVF indirection distance kernel.
///
/// Memory layout must match the Metal shader's `IVFIndirectionDistanceParams`.
public struct IVFIndirectionDistanceParameters: Sendable {
    /// Vector dimension (D)
    public let dimension: UInt32

    /// Total number of candidate entries across all queries
    public let totalCandidates: UInt32

    /// Number of queries (Q)
    public let numQueries: UInt32

    /// Total IVF entries (length of `vectorIndices`)
    public let totalIVFEntries: UInt32

    /// Storage capacity in slots for the global vector buffer
    public let storageCapacity: UInt32

    /// Padding for 32-byte alignment
    private let padding: (UInt32, UInt32, UInt32) = (0, 0, 0)

    public init(
        dimension: Int,
        totalCandidates: Int,
        numQueries: Int,
        totalIVFEntries: Int,
        storageCapacity: Int
    ) {
        self.dimension = UInt32(dimension)
        self.totalCandidates = UInt32(totalCandidates)
        self.numQueries = UInt32(numQueries)
        self.totalIVFEntries = UInt32(totalIVFEntries)
        self.storageCapacity = UInt32(storageCapacity)
    }
}

// MARK: - Result

public struct IVFIndirectionDistanceResult: Sendable {
    /// Distances for each candidate entry (flat)
    public let distances: any MTLBuffer

    /// Storage slot for each candidate entry (flat)
    public let slots: any MTLBuffer

    /// Total candidates
    public let totalCandidates: Int
}

// MARK: - Kernel

/// Metal 4 kernel for IVF candidate distance computation using indirection.
public final class IVFIndirectionDistanceKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "IVFIndirectionDistanceKernel"

    // MARK: - Pipeline

    private let pipeline: any MTLComputePipelineState

    // MARK: - Init

    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()
        guard let function = library.makeFunction(name: "ivf_distance_with_indirection") else {
            throw VectorError.shaderNotFound(
                name: "ivf_distance_with_indirection. Ensure SearchAndRetrieval.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipeline = try await device.makeComputePipelineState(function: function)
    }

    public func warmUp() async throws {
        // Pipeline created in init
    }

    // MARK: - Encode

    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        vectors: any MTLBuffer,
        vectorIndices: any MTLBuffer,
        candidateIVFIndices: any MTLBuffer,
        candidateQueryIds: any MTLBuffer,
        outDistances: any MTLBuffer,
        outSlots: any MTLBuffer,
        parameters: IVFIndirectionDistanceParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "IVFIndirectionDistance"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(vectors, offset: 0, index: 1)
        encoder.setBuffer(vectorIndices, offset: 0, index: 2)
        encoder.setBuffer(candidateIVFIndices, offset: 0, index: 3)
        encoder.setBuffer(candidateQueryIds, offset: 0, index: 4)
        encoder.setBuffer(outDistances, offset: 0, index: 5)
        encoder.setBuffer(outSlots, offset: 0, index: 6)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<IVFIndirectionDistanceParameters>.size, index: 7)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.totalCandidates),
            pipeline: pipeline
        )

        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )

        return Metal4EncodingResult(
            pipelineName: "ivf_distance_with_indirection",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute

    public func execute(
        queries: any MTLBuffer,
        vectors: any MTLBuffer,
        vectorIndices: any MTLBuffer,
        candidateIVFIndices: any MTLBuffer,
        candidateQueryIds: any MTLBuffer,
        parameters: IVFIndirectionDistanceParameters
    ) async throws -> IVFIndirectionDistanceResult {
        let device = context.device.rawDevice
        let total = Int(parameters.totalCandidates)

        // Allocate outputs (use a minimum size to avoid zero-length buffers)
        let distancesBytes = max(total * MemoryLayout<Float>.size, 4)
        let slotsBytes = max(total * MemoryLayout<UInt32>.size, 4)

        guard let distances = device.makeBuffer(length: distancesBytes, options: .storageModeShared),
              let slots = device.makeBuffer(length: slotsBytes, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distancesBytes + slotsBytes)
        }
        distances.label = "IVFIndirectionDistance.distances"
        slots.label = "IVFIndirectionDistance.slots"

        if total == 0 {
            return IVFIndirectionDistanceResult(distances: distances, slots: slots, totalCandidates: 0)
        }

        try await context.executeAndWait { [self] _, encoder in
            _ = self.encode(
                into: encoder,
                queries: queries,
                vectors: vectors,
                vectorIndices: vectorIndices,
                candidateIVFIndices: candidateIVFIndices,
                candidateQueryIds: candidateQueryIds,
                outDistances: distances,
                outSlots: slots,
                parameters: parameters
            )
        }

        return IVFIndirectionDistanceResult(
            distances: distances,
            slots: slots,
            totalCandidates: total
        )
    }
}
