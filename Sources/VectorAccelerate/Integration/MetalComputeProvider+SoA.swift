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
    /// Euclidean → L2 distance; cosine → 1 − similarity. Other metrics are unsupported on this path.
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
        let qNormSq: Float = metric == .cosine ? qa.reduce(0) { $0 + $1 * $1 } : 0
        let lanes = set.layout.lanes
        let kernel = soaKernel    // capture the Sendable kernel reference for the @Sendable closure

        try await context.executeAndWait { cb, enc in
            kernel.encode(
                into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                distancesBuffer: outToken.buffer, count: count, lanes: lanes,
                metric: metric, computeSqrt: true, queryNormSq: qNormSq)
            qToken.keepAlive(until: cb)
            outToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin the SoA until the GPU completes
        }
        return outToken.copyData(as: Float.self, count: count)
    }

    /// k nearest candidates in a prebuilt SoA set, nearest-first. Distance kernel → CPU top-K.
    func findNearest<V: SoACompatible & VectorProtocol>(
        query: V, in set: SoACandidateSet<V>, k: Int, metric: SupportedDistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, set.layout.count > 0 else { return [] }
        let distances = try await batchDistance(query: query, against: set, metric: metric)
        return Self.selectTopK(distances, k: min(k, distances.count), largerIsCloser: false)
    }
}
