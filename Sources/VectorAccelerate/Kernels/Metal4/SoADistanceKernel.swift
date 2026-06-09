//
//  SoADistanceKernel.swift
//  VectorAccelerate
//
//  Lane-major SoA distance kernel (L2 + cosine). Reads a candidate buffer in VectorCore's frozen
//  SoA layout (buffer[ℓ*count + j]), one thread per candidate.
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Parameters for `soa_l2_distance` (16 bytes; mirrors the MSL `SoAL2Params`).
public struct SoAL2Params: Sendable {
    public var count: UInt32
    public var lanes: UInt32
    public var computeSqrt: UInt32
    public var _pad: UInt32 = 0
    public init(count: Int, lanes: Int, computeSqrt: Bool) {
        self.count = UInt32(count); self.lanes = UInt32(lanes); self.computeSqrt = computeSqrt ? 1 : 0
    }
}

/// Parameters for `soa_cosine_distance` (16 bytes; mirrors the MSL `SoACosineParams`).
public struct SoACosineParams: Sendable {
    public var count: UInt32
    public var lanes: UInt32
    public var queryNormSq: Float
    public var _pad: UInt32 = 0
    public init(count: Int, lanes: Int, queryNormSq: Float) {
        self.count = UInt32(count); self.lanes = UInt32(lanes); self.queryNormSq = queryNormSq
    }
}

/// Lane-major SoA distance kernel (L2 + cosine). One thread per candidate, looping lanes; reads the
/// candidate buffer in VectorCore's frozen SoA layout (`buffer[ℓ*count + j]`, `float4` per lane).
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class SoADistanceKernel: @unchecked Sendable, Metal4Kernel {
    public let name = "SoADistanceKernel"
    public let context: Metal4Context
    private let l2Pipeline: any MTLComputePipelineState
    private let cosinePipeline: any MTLComputePipelineState

    public init(context: Metal4Context) async throws {
        self.context = context
        let library = try await context.shaderCompiler.getDefaultLibrary()
        guard let l2 = library.makeFunction(name: "soa_l2_distance"),
              let cos = library.makeFunction(name: "soa_cosine_distance") else {
            throw VectorError.shaderNotFound(name: "VectorAccelerate: soa_l2_distance/soa_cosine_distance not found")
        }
        let dev = context.device.rawDevice
        self.l2Pipeline = try await dev.makeComputePipelineState(function: l2)
        self.cosinePipeline = try await dev.makeComputePipelineState(function: cos)
    }

    public func warmUp() async throws {}

    /// `count`/`lanes` come from the candidate set's `SoALayout`. `queryNormSq` is used for cosine only.
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queryBuffer: any MTLBuffer, candidateBuffer: any MTLBuffer, distancesBuffer: any MTLBuffer,
        count: Int, lanes: Int, metric: SupportedDistanceMetric,
        computeSqrt: Bool = true, queryNormSq: Float = 0)
    {
        let pipeline = (metric == .cosine) ? cosinePipeline : l2Pipeline
        encoder.setComputePipelineState(pipeline)
        encoder.label = "SoADistance(\(metric))"
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidateBuffer, offset: 0, index: 1)
        encoder.setBuffer(distancesBuffer, offset: 0, index: 2)
        if metric == .cosine {
            var p = SoACosineParams(count: count, lanes: lanes, queryNormSq: queryNormSq)
            encoder.setBytes(&p, length: MemoryLayout<SoACosineParams>.stride, index: 3)
        } else {
            var p = SoAL2Params(count: count, lanes: lanes, computeSqrt: computeSqrt)
            encoder.setBytes(&p, length: MemoryLayout<SoAL2Params>.stride, index: 3)
        }
        let w = pipeline.threadExecutionWidth
        let per = (min(pipeline.maxTotalThreadsPerThreadgroup, 256) / w) * w
        let groups = MTLSizeMake((count + per - 1) / per, 1, 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSizeMake(per, 1, 1))
    }
}
