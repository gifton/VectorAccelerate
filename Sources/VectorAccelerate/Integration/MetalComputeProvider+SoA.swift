//
//  MetalComputeProvider+SoA.swift
//  VectorAccelerate
//
//  Zero-copy, lane-major SoA scoring on MetalComputeProvider: score a query against a prebuilt
//  SoACandidateSet (build once, query many) using the cached SoADistanceKernel.
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md
//

import Foundation
@preconcurrency import Metal
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension MetalComputeProvider {

    /// Zero-copy batch distance from `query` to every candidate in a prebuilt SoA set.
    /// Euclidean → L2 distance; cosine → 1 − similarity. Other metrics throw `invalidInput`.
    ///
    /// This is an explicit, build-once / query-many API — it is **not** reached by the transparent
    /// `BatchKernelProvider` dispatch (which has no prebuilt `SoACandidateSet` to pass).
    func batchDistance<V: SoACompatible & VectorProtocol>(
        query: V, against set: SoACandidateSet<V>, metric: SupportedDistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        guard metric == .euclidean || metric == .cosine else {
            throw VectorError.invalidInput("SoA scoring supports euclidean and cosine only")
        }
        let count = set.layout.count
        guard count > 0 else { return [] }

        let qa = query.toArray()
        let qToken = try await context.getBuffer(for: qa)
        let outToken = try await context.getBuffer(size: count * MemoryLayout<Float>.stride)
        let lanes = set.layout.lanes
        let kernel = soaKernel    // capture the Sendable kernel reference for the @Sendable closure

        try await context.executeAndWait { cb, enc in
            kernel.encode(
                into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                distancesBuffer: outToken.buffer, count: count, lanes: lanes,
                metric: metric, computeSqrt: true)
            qToken.keepAlive(until: cb)
            outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin the SoA until the GPU completes
        }
        return outToken.copyData(as: Float.self, count: count)
    }

    /// k nearest candidates in a prebuilt SoA set, nearest-first.
    ///
    /// `k ≤ 128` (`TopKParameters.maxK`, the selection kernel's register-heap cap): the distance
    /// kernel and `TopKSelectionKernel` are encoded into the **same** command buffer, separated by
    /// `encoder.memoryBarrier(scope: .buffers)` — only the k selected (index, distance) pairs are
    /// read back; the full N-element distance vector never leaves the GPU.
    ///
    /// `k > 128`: the distance kernel runs alone (as before), then VectorCore's zero-copy pointer
    /// selection (`TopKSelection.select(k:from:UnsafePointer<Float>,count:...)`) selects the top-k
    /// directly off the shared-storage distances buffer's contents — no `[Float]` materialization
    /// via `copyData`, unlike `batchDistance`.
    func findNearest<V: SoACompatible & VectorProtocol>(
        query: V, in set: SoACandidateSet<V>, k: Int, metric: SupportedDistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, set.layout.count > 0 else { return [] }
        guard metric == .euclidean || metric == .cosine else {
            throw VectorError.invalidInput("SoA scoring supports euclidean and cosine only")
        }
        let count = set.layout.count
        let effectiveK = min(k, count)
        let lanes = set.layout.lanes

        let qa = query.toArray()
        let qToken = try await context.getBuffer(for: qa)
        let distToken = try await context.getBuffer(size: count * MemoryLayout<Float>.stride)
        let distanceKernel = soaKernel   // capture Sendable kernel refs for the @Sendable closures

        if effectiveK <= TopKParameters.maxK {
            let selectionKernel = soaTopKKernel
            let valuesToken = try await context.getBuffer(size: effectiveK * MemoryLayout<Float>.stride)
            let indicesToken = try await context.getBuffer(size: effectiveK * MemoryLayout<UInt32>.stride)
            let params = TopKParameters(
                batchSize: 1, numElements: count, k: effectiveK, mode: .minimum, sorted: true)

            try await context.executeAndWait { cb, enc in
                distanceKernel.encode(
                    into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                    distancesBuffer: distToken.buffer, count: count, lanes: lanes,
                    metric: metric, computeSqrt: true)
                enc.memoryBarrier(scope: .buffers)
                selectionKernel.encode(
                    into: enc, input: distToken.buffer, outputValues: valuesToken.buffer,
                    outputIndices: indicesToken.buffer, parameters: params)
                qToken.keepAlive(until: cb)
                distToken.keepAlive(until: cb)
                valuesToken.keepAlive(until: cb)
                indicesToken.keepAlive(until: cb)
                cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin the SoA until the GPU completes
            }

            let indices = indicesToken.copyData(as: UInt32.self, count: effectiveK)
            let values = valuesToken.copyData(as: Float.self, count: effectiveK)
            return zip(indices, values).map { index, distance in (index: Int(index), distance: distance) }
        }

        // k > 128: distance kernel alone, then CPU pointer-select straight off the shared-storage
        // buffer's contents (avoids the full N-element `[Float]` that `copyData` would allocate).
        try await context.executeAndWait { cb, enc in
            distanceKernel.encode(
                into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                distancesBuffer: distToken.buffer, count: count, lanes: lanes,
                metric: metric, computeSqrt: true)
            qToken.keepAlive(until: cb)
            distToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin the SoA until the GPU completes
        }

        let distancesPtr = distToken.contents(as: Float.self)
        let selected = TopKSelection.select(
            k: effectiveK, from: distancesPtr, count: count, tieBreaker: .smallerIndex)
        return zip(selected.indices, selected.distances).map { index, distance in
            (index: Int(index), distance: distance)
        }
    }
}
