//
//  L2DistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 L2 (Euclidean) distance kernel with hierarchical SIMD reduction.
//
//  Phase 3: SIMD-Group Reductions & Threadgroup Sizing
//
//  Features:
//  - Hierarchical SIMD-group reduction for maximum Apple Silicon throughput
//  - Dynamic threadgroup sizing based on pipeline characteristics
//  - Fully asynchronous execution via BufferPool tokens
//  - Strictly 1 Threadgroup per Query Evaluation (1:1 Batch)
//  - Optional sqrt: squared L2 (default) or Euclidean distance

import Metal
import Foundation

public final class L2DistanceKernel: @unchecked Sendable, Metal4Kernel {
    public let name: String = "L2DistanceKernel"
    private let pipelineState: any MTLComputePipelineState
    public let context: Metal4Context

    public init(context: Metal4Context) async throws {
        self.context = context
        let device = context.device.rawDevice
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "l2_distance") else {
            throw VectorError.shaderNotFound(name: "VectorAccelerate: 'l2_distance' kernel not found.")
        }
        self.pipelineState = try await device.makeComputePipelineState(function: function)
    }

    /// Encode L2 distance computation into an existing encoder.
    ///
    /// Computes `distance[i] = ||query[i] - target[i]||₂` (or squared) for each
    /// 1:1 query-target pair.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to dispatch into
    ///   - commandBuffer: The command buffer (used for token lifecycle management)
    ///   - queriesToken: Buffer containing query vectors [numQueries, dimension]
    ///   - targetsToken: Buffer containing target vectors [numQueries, dimension] (1:1 pairing)
    ///   - distancesToken: Output buffer [numQueries] — one float per pair
    ///   - numQueries: Number of query-target pairs
    ///   - dimension: Vector dimension
    ///   - computeSqrt: If true, return Euclidean distance. If false, squared distance. Default true.
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        commandBuffer: any MTLCommandBuffer,
        queriesToken: BufferToken,
        targetsToken: BufferToken,
        distancesToken: BufferToken,
        numQueries: Int,
        dimension: Int,
        computeSqrt: Bool = true
    ) {
        encoder.setComputePipelineState(pipelineState)

        encoder.setBuffer(queriesToken.buffer, offset: 0, index: 0)
        encoder.setBuffer(targetsToken.buffer, offset: 0, index: 1)
        encoder.setBuffer(distancesToken.buffer, offset: 0, index: 2)

        var dim = UInt32(dimension)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

        var sqrtFlag = UInt32(computeSqrt ? 1 : 0)
        encoder.setBytes(&sqrtFlag, length: MemoryLayout<UInt32>.size, index: 4)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadgroupsPerGrid = MTLSizeMake(numQueries, 1, 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        queriesToken.keepAlive(until: commandBuffer)
        targetsToken.keepAlive(until: commandBuffer)
        distancesToken.keepAlive(until: commandBuffer)
    }

    public func warmUp() async throws { }
}
