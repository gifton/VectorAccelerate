//
//  TopKMergeKernel.swift
//  VectorAccelerate
//
//  GPU merge kernel for two sorted top-k lists.
//
//  Used by the chunked fallback path in FusedL2TopKKernel to avoid CPU merges.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

/// Parameters for merging two sorted top-k lists.
///
/// Memory layout must match the Metal shader's `TopKMergeParams`.
struct TopKMergeParameters: Sendable {
    /// Number of queries (Q)
    let numQueries: UInt32

    /// Output K (length of running list)
    let k: UInt32

    /// Input K for the current chunk
    let chunkK: UInt32

    /// Base offset to add to chunk-local indices
    let chunkBase: UInt32

    init(numQueries: Int, k: Int, chunkK: Int, chunkBase: Int) {
        self.numQueries = UInt32(numQueries)
        self.k = UInt32(k)
        self.chunkK = UInt32(chunkK)
        self.chunkBase = UInt32(chunkBase)
    }
}

/// Metal 4 kernel that merges two sorted (best-to-worst) top-k lists.
///
/// - `running*` buffers are [Q × K]
/// - `chunk*` buffers are [Q × chunkK]
/// - outputs are [Q × K]
///
/// The chunk indices are assumed to be local to the chunk (0..<chunkSize) and
/// will be converted to global indices by adding `chunkBase`.
final class TopKMergeKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    let context: Metal4Context
    let name: String = "TopKMergeKernel"

    // MARK: - Pipeline

    private let pipeline: any MTLComputePipelineState

    // MARK: - Init

    init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let fn = library.makeFunction(name: "merge_topk_sorted_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Top-K merge kernel. Ensure SearchAndRetrieval.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipeline = try await device.makeComputePipelineState(function: fn)
    }

    func warmUp() async throws {
        // Pipeline created in init
    }

    /// Encode merge into an existing encoder.
    ///
    /// - Parameters:
    ///   - runningIndices: [Q × K]
    ///   - runningDistances: [Q × K]
    ///   - chunkIndices: [Q × chunkK] (indices local to current chunk)
    ///   - chunkDistances: [Q × chunkK]
    ///   - outputIndices: [Q × K]
    ///   - outputDistances: [Q × K]
    @discardableResult
    func encode(
        into encoder: any MTLComputeCommandEncoder,
        runningIndices: any MTLBuffer,
        runningDistances: any MTLBuffer,
        chunkIndices: any MTLBuffer,
        chunkDistances: any MTLBuffer,
        outputIndices: any MTLBuffer,
        outputDistances: any MTLBuffer,
        parameters: TopKMergeParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "TopKMerge.sorted (K=\(parameters.k), chunkK=\(parameters.chunkK))"

        encoder.setBuffer(runningIndices, offset: 0, index: 0)
        encoder.setBuffer(runningDistances, offset: 0, index: 1)
        encoder.setBuffer(chunkIndices, offset: 0, index: 2)
        encoder.setBuffer(chunkDistances, offset: 0, index: 3)
        encoder.setBuffer(outputIndices, offset: 0, index: 4)
        encoder.setBuffer(outputDistances, offset: 0, index: 5)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<TopKMergeParameters>.size, index: 6)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numQueries),
            pipeline: pipeline
        )

        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )

        return Metal4EncodingResult(
            pipelineName: "merge_topk_sorted_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }
}
