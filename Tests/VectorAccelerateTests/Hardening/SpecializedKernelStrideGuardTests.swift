//
//  SpecializedKernelStrideGuardTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-010: dimension-specialized kernels hardcode dense packing (stride == dimension).
//  Selecting one for strided input silently reads the wrong elements — no error, plausible
//  numbers. `DotProductKernel` exposes explicit strides through its public
//  `DotProductParameters` init and `execute(parameters:)`, so its pipeline selection must fall
//  back to the general kernel whenever either stride differs from the dimension
//  (`L2NormalizationKernel.selectPipeline` is the reference pattern).
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOSApplicationExtension 3.0, *)
final class SpecializedKernelStrideGuardTests: XCTestCase {

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Strided input at a specialized dimension (384) must produce the same dot products as
    /// the dense general path. Pre-fix, `selectPipeline` chose `dot_product_384_kernel` on
    /// dimension alone; that kernel hardcodes row offsets of 384, so with a row stride of 400
    /// it read parts of neighboring rows — silently wrong results.
    func testDotProductStridedInputAtSpecializedDimension() async throws {
        let context = try await Metal4Context()
        let kernel = try await DotProductKernel(context: context)

        let dimension = 384
        let stride = 400          // 16 floats of padding per row
        let numQueries = 2        // > 1 so the GEMV path is not taken
        let numDatabase = 3

        var rng = TestRNG(seed: 0xA3_0010)
        // Padded rows: payload in [0, 384), poison in the padding so a dense-stride read
        // is guaranteed to change the answer.
        func paddedRows(_ count: Int) -> [Float] {
            var flat = [Float](repeating: 0, count: count * stride)
            for r in 0..<count {
                for i in 0..<dimension {
                    flat[r * stride + i] = rng.nextFloat(in: -1...1)
                }
                for i in dimension..<stride {
                    flat[r * stride + i] = 1e6   // poison
                }
            }
            return flat
        }
        let queriesFlat = paddedRows(numQueries)
        let databaseFlat = paddedRows(numDatabase)

        let device = context.device.rawDevice
        let queriesBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: queriesFlat, length: queriesFlat.count * MemoryLayout<Float>.size,
            options: .storageModeShared))
        let databaseBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: databaseFlat, length: databaseFlat.count * MemoryLayout<Float>.size,
            options: .storageModeShared))

        let parameters = DotProductParameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            strideQuery: stride,
            strideDatabase: stride,
            strideOutput: numDatabase)

        let output = try await kernel.execute(
            queries: queriesBuffer, database: databaseBuffer, parameters: parameters)
        let gpu = Array(UnsafeBufferPointer(
            start: output.contents().bindMemory(to: Float.self, capacity: numQueries * numDatabase),
            count: numQueries * numDatabase))

        for q in 0..<numQueries {
            for c in 0..<numDatabase {
                var expected: Double = 0
                for i in 0..<dimension {
                    expected += Double(queriesFlat[q * stride + i]) * Double(databaseFlat[c * stride + i])
                }
                XCTAssertEqual(gpu[q * numDatabase + c], Float(expected), accuracy: 1e-2,
                               "strided dot product diverged at (\(q), \(c)) — specialized dense kernel selected for strided input?")
            }
        }
    }

    /// Meta-review: the guard is a two-conjunct check — BOTH `strideQuery == dimension` and
    /// `strideDatabase == dimension` must hold before the dense kernel is selected. The leg
    /// above pads both strides at once, so a regression that kept only one conjunct would
    /// still pass it while single-side-padded inputs silently corrupted. These legs pad
    /// exactly one side at a time.
    func testDotProductMixedStrideCombinations() async throws {
        let context = try await Metal4Context()
        let kernel = try await DotProductKernel(context: context)
        let device = context.device.rawDevice

        let dimension = 384
        let padded = 400
        let numQueries = 2
        let numDatabase = 3
        var rng = TestRNG(seed: 0xA3_0011)

        func rows(_ count: Int, stride: Int) -> [Float] {
            var flat = [Float](repeating: 0, count: count * stride)
            for r in 0..<count {
                for i in 0..<dimension { flat[r * stride + i] = rng.nextFloat(in: -1...1) }
                for i in dimension..<stride { flat[r * stride + i] = 1e6 }   // poison padding
            }
            return flat
        }

        for (strideQuery, strideDatabase) in [(padded, dimension), (dimension, padded)] {
            let queriesFlat = rows(numQueries, stride: strideQuery)
            let databaseFlat = rows(numDatabase, stride: strideDatabase)
            let queriesBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: queriesFlat, length: queriesFlat.count * MemoryLayout<Float>.size,
                options: .storageModeShared))
            let databaseBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: databaseFlat, length: databaseFlat.count * MemoryLayout<Float>.size,
                options: .storageModeShared))

            let parameters = DotProductParameters(
                numQueries: numQueries,
                numDatabase: numDatabase,
                dimension: dimension,
                strideQuery: strideQuery,
                strideDatabase: strideDatabase,
                strideOutput: numDatabase)

            let output = try await kernel.execute(
                queries: queriesBuffer, database: databaseBuffer, parameters: parameters)
            let gpu = Array(UnsafeBufferPointer(
                start: output.contents().bindMemory(to: Float.self, capacity: numQueries * numDatabase),
                count: numQueries * numDatabase))

            for q in 0..<numQueries {
                for c in 0..<numDatabase {
                    var expected: Double = 0
                    for i in 0..<dimension {
                        expected += Double(queriesFlat[q * strideQuery + i])
                            * Double(databaseFlat[c * strideDatabase + i])
                    }
                    XCTAssertEqual(gpu[q * numDatabase + c], Float(expected), accuracy: 1e-2,
                                   "sq=\(strideQuery) sd=\(strideDatabase) (\(q), \(c)): mixed-stride input hit a dense kernel?")
                }
            }
        }
    }
}
