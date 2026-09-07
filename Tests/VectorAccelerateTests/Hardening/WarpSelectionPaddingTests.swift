//
//  WarpSelectionPaddingTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-008: `warp_select_small_k_{ascending,descending}` clamp K to
//  min(k, candidateCount) but write the output at stride k — and the pre-fix kernels wrote
//  only non-sentinel lanes below K, leaving slots K..k-1 holding whatever bytes the output
//  buffer already contained. Reachable through FusedL2TopKKernel's chunked two-pass fallback
//  whenever the final chunk is shorter than k. The convention everywhere else in the corpus
//  (topk_select_batch_kernel, fused_l2_topk, ivf_list_search) is sentinel padding:
//  index 0xFFFFFFFF with +Inf (ascending) / −Inf (descending).
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class WarpSelectionPaddingTests: XCTestCase {

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testShortCandidateListPadsTailWithSentinels() async throws {
        let context = try await Metal4Context()
        let kernel = try await WarpOptimizedSelectionKernel(context: context)
        let device = context.device.rawDevice

        let queryCount = 2
        let candidateCount = 3
        let k = 8

        // Row q holds [3+q, 1+q, 2+q] → ascending order of candidates is (1, 2, 0).
        var distances: [Float] = []
        for q in 0..<queryCount {
            distances.append(Float(3 + q))
            distances.append(Float(1 + q))
            distances.append(Float(2 + q))
        }
        let distancesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: distances, length: distances.count * MemoryLayout<Float>.size,
            options: .storageModeShared))

        // Poison the output buffers: the pre-fix kernels left these bytes in the tail slots.
        let outCount = queryCount * k
        let poisonIndices = [UInt32](repeating: 0xDEAD_BEEF, count: outCount)
        let poisonValues = [Float](repeating: 12345.0, count: outCount)
        let indicesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonIndices, length: outCount * MemoryLayout<UInt32>.size,
            options: .storageModeShared))
        let valuesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonValues, length: outCount * MemoryLayout<Float>.size,
            options: .storageModeShared))

        for mode in [Metal4WarpSelectionMode.ascending, .descending] {
            indicesBuffer.contents().copyMemory(
                from: poisonIndices, byteCount: outCount * MemoryLayout<UInt32>.size)
            valuesBuffer.contents().copyMemory(
                from: poisonValues, byteCount: outCount * MemoryLayout<Float>.size)

            try await context.executeAndWait { _, encoder in
                kernel.encodeWarp(
                    into: encoder,
                    distances: distancesBuffer,
                    outputIndices: indicesBuffer,
                    outputValues: valuesBuffer,
                    queryCount: queryCount,
                    candidateCount: candidateCount,
                    k: k,
                    mode: mode)
            }

            let gpuIndices = Array(UnsafeBufferPointer(
                start: indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: outCount),
                count: outCount))
            let gpuValues = Array(UnsafeBufferPointer(
                start: valuesBuffer.contents().bindMemory(to: Float.self, capacity: outCount),
                count: outCount))

            let padValue: Float = mode == .ascending ? .infinity : -.infinity
            for q in 0..<queryCount {
                // Real results occupy the first candidateCount slots.
                let realIndices = Set(gpuIndices[(q * k)..<(q * k + candidateCount)])
                XCTAssertEqual(realIndices, Set<UInt32>([0, 1, 2]),
                               "\(mode): query \(q) should select all \(candidateCount) candidates")
                // The tail must be sentinel-padded, never leftover buffer contents.
                for slot in candidateCount..<k {
                    XCTAssertEqual(gpuIndices[q * k + slot], 0xFFFF_FFFF,
                                   "\(mode): tail index slot \(slot) of query \(q) not sentinel-padded")
                    XCTAssertEqual(gpuValues[q * k + slot], padValue,
                                   "\(mode): tail value slot \(slot) of query \(q) not padded with \(padValue)")
                }
            }
        }
    }

    /// AUDIT-3 meta-review: k beyond K4_MAX_K (32) hit the kernel's capability cap, which
    /// returned without writing ANY output slot — resurrecting the exact VA3-008 stale-bytes
    /// symptom on the public `encodeWarp` API (`selectTopK` routes such k elsewhere, but the
    /// encode API bypasses that routing). An over-cap request must read back as all-sentinel
    /// "no results", never as whatever the pool left in the buffers.
    func testOverCapKPadsAllSlotsWithSentinels() async throws {
        let context = try await Metal4Context()
        let kernel = try await WarpOptimizedSelectionKernel(context: context)
        let device = context.device.rawDevice

        let queryCount = 2
        let candidateCount = 5
        let k = 33   // K4_MAX_K + 1

        var distances: [Float] = []
        for q in 0..<queryCount {
            for c in 0..<candidateCount { distances.append(Float(c + q)) }
        }
        let distancesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: distances, length: distances.count * MemoryLayout<Float>.size,
            options: .storageModeShared))

        let outCount = queryCount * k
        let poisonIndices = [UInt32](repeating: 0xDEAD_BEEF, count: outCount)
        let poisonValues = [Float](repeating: 12345.0, count: outCount)
        let indicesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonIndices, length: outCount * MemoryLayout<UInt32>.size,
            options: .storageModeShared))
        let valuesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: poisonValues, length: outCount * MemoryLayout<Float>.size,
            options: .storageModeShared))

        for mode in [Metal4WarpSelectionMode.ascending, .descending] {
            indicesBuffer.contents().copyMemory(
                from: poisonIndices, byteCount: outCount * MemoryLayout<UInt32>.size)
            valuesBuffer.contents().copyMemory(
                from: poisonValues, byteCount: outCount * MemoryLayout<Float>.size)

            try await context.executeAndWait { _, encoder in
                kernel.encodeWarp(
                    into: encoder,
                    distances: distancesBuffer,
                    outputIndices: indicesBuffer,
                    outputValues: valuesBuffer,
                    queryCount: queryCount,
                    candidateCount: candidateCount,
                    k: k,
                    mode: mode)
            }

            let gpuIndices = Array(UnsafeBufferPointer(
                start: indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: outCount),
                count: outCount))
            let gpuValues = Array(UnsafeBufferPointer(
                start: valuesBuffer.contents().bindMemory(to: Float.self, capacity: outCount),
                count: outCount))

            let padValue: Float = mode == .ascending ? .infinity : -.infinity
            for slot in 0..<outCount {
                XCTAssertEqual(gpuIndices[slot], 0xFFFF_FFFF,
                               "\(mode): over-cap k slot \(slot) holds stale index bytes")
                XCTAssertEqual(gpuValues[slot], padValue,
                               "\(mode): over-cap k slot \(slot) holds stale value bytes")
            }
        }
    }
}
