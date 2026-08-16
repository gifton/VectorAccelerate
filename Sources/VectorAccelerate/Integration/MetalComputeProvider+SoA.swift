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

    /// k nearest candidates in a prebuilt SoA set, nearest-first. The distance kernel runs on the
    /// GPU, then VectorCore's zero-copy pointer selection (`TopKSelection.select(k:from:
    /// UnsafePointer<Float>,count:...)`) selects the top-k directly off the shared-storage
    /// distances buffer's contents — no `[Float]` materialization via `copyData`, unlike
    /// `batchDistance`.
    ///
    /// A GPU-fused single-command-buffer variant was prototyped and measured (distance kernel +
    /// `TopKSelectionKernel`, `encoder.memoryBarrier(scope: .buffers)` between passes, for
    /// `k ≤ TopKParameters.maxK`) and **rejected**: `topk_select_batch_kernel` dispatches one
    /// thread per query, so at `batchSize == 1` a single GPU thread serially scans all N
    /// candidates through a 128-slot thread-private heap (register-spilling for most of that
    /// range). Measured on an Apple M3 Max, dim 768, euclidean: 1.8×–6.7× **slower** than this
    /// CPU pointer-select path across N ∈ {1e3, 1e4, 1e5} × k ∈ {8, 128}, with the gap widening as
    /// N grows — see `.superpowers/sdd/soa-060-release-hardening/task-3-report.md` ("Performance
    /// evidence") for the full table.
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
        let distanceKernel = soaKernel   // capture the Sendable kernel reference for the @Sendable closure

        try await context.executeAndWait { cb, enc in
            distanceKernel.encode(
                into: enc, queryBuffer: qToken.buffer, candidateBuffer: set.buffer,
                distancesBuffer: distToken.buffer, count: count, lanes: lanes,
                metric: metric, computeSqrt: true)
            qToken.keepAlive(until: cb)
            distToken.keepAlive(until: cb)
            cb.addCompletedHandler { _ in _ = set }   // borrow mode: pin the SoA until the GPU completes
        }

        // ARC lifetime: `distToken`'s last syntactic use is `.contents(as:)`, so under -O the
        // optimizer is free to release it (→ `BufferToken.deinit` → `PendingBufferReturns.enqueue`,
        // making the buffer eligible for reuse by a concurrent task) before `TopKSelection.select`
        // finishes reading `count` floats through the raw pointer it returned. `keepAlive(until:)`
        // does not cover this window: its completion handler already fired before `executeAndWait`
        // returned, so past that point the local `let distToken` is the only strong reference.
        // `withExtendedLifetime` keeps it alive through the read and the construction of `selected`.
        let selected = withExtendedLifetime(distToken) { () -> (indices: [Int32], distances: [Float]) in
            let distancesPtr = distToken.contents(as: Float.self)
            return TopKSelection.select(
                k: effectiveK, from: distancesPtr, count: count, tieBreaker: .smallerIndex)
        }
        return zip(selected.indices, selected.distances).map { index, distance in
            (index: Int(index), distance: distance)
        }
    }
}
