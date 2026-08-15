//
//  NormalizationParityTests.swift
//  VectorAccelerateTests
//
//  BE3 §4.4: CPU/GPU normalization parity across every degenerate input class.
//
//  Both GPU shader families (`vectorNormalize`/`normalizeVectors` in
//  BasicOperations.metal and the `l2_normalize_*` family in L2Normalization.metal)
//  and VectorAccelerate's CPU fallbacks implement the same Kahan pre-scaled
//  algorithm as VectorCore's `NormalizeKernels`:
//
//      den = max(maxAbs, FLT_MIN=0x1p-126) ; scale = 1/den
//      ‖v‖ = den · sqrt(Σ (v·scale)²)      ; out = v · (1/‖v‖)
//
//  Invariants asserted here, for every fixture and every path:
//    * no NaN/Inf ever reaches the output,
//    * ‖out‖₂ = 1 ± 1e-5 whenever 1/‖v‖₂ is representable in FP32,
//    * the input is returned unchanged otherwise (zero vector, deep subnormals) —
//      the observable behavior of `NormalizeKernels.normalizeUnchecked`,
//    * GPU and CPU agree elementwise within 1e-5.
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

final class NormalizationParityTests: XCTestCase {

    // MARK: - Fixtures

    private struct Fixture: Sendable {
        let name: String
        let input: [Float]
        /// True when `1/‖v‖₂` is representable in FP32, i.e. the normalizer must
        /// return a unit vector. False for the classes every implementation
        /// (CPU and GPU) passes through unchanged.
        let isNormalizable: Bool
    }

    /// 1e-40 is subnormal in FP32 (below `Float.leastNormalMagnitude` = 1.18e-38).
    private static let subnormal = Float(1e-40 as Double)

    private static func fixtures(dimension: Int) -> [Fixture] {
        precondition(dimension % 8 == 0)

        // Subnormal background with 8 unit spikes → maxAbs = 1 (normal) while most
        // components are subnormal: exercises the pre-scale on a mixed vector.
        var mixed = [Float](repeating: subnormal, count: dimension)
        for i in stride(from: 0, to: dimension, by: dimension / 8) { mixed[i] = 1.0 }

        var rng = TestRNG(seed: 0xBE30044)
        var random = [Float](repeating: 0, count: dimension)
        for i in 0..<dimension { random[i] = rng.nextFloat(in: -1...1) }

        return [
            // ‖v‖ = 2.26e-39 → 1/‖v‖ = 4.4e38 > FLT_MAX: not normalizable in FP32.
            Fixture(name: "all-subnormal(1e-40)",
                    input: [Float](repeating: subnormal, count: dimension),
                    isNormalizable: false),
            // Naive Σ v² = 5.1e-38 survives, but the old GPU epsilon (1e-7/1e-8)
            // rejected the 2.26e-19 norm and returned zeros / the input unchanged.
            Fixture(name: "micro(1e-20)",
                    input: [Float](repeating: 1e-20, count: dimension),
                    isNormalizable: true),
            // Naive Σ v² = 5.1e40 → +Inf on the old path → all-zero output.
            Fixture(name: "huge(1e19)",
                    input: [Float](repeating: 1e19, count: dimension),
                    isNormalizable: true),
            Fixture(name: "zero",
                    input: [Float](repeating: 0, count: dimension),
                    isNormalizable: false),
            Fixture(name: "mixed-subnormal-normal", input: mixed, isNormalizable: true),
            Fixture(name: "ordinary-random", input: random, isNormalizable: true)
        ]
    }

    // MARK: - Assertions

    private func l2Norm(_ v: [Float]) -> Double {
        var s = 0.0
        for x in v { s += Double(x) * Double(x) }
        return s.squareRoot()
    }

    private func assertNoNonFinite(
        _ v: [Float], _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        if let bad = v.enumerated().first(where: { !$0.element.isFinite }) {
            XCTFail("\(label): non-finite output at [\(bad.offset)] = \(bad.element)", file: file, line: line)
        }
    }

    /// The full contract for one (implementation, fixture) pair.
    private func assertPolicy(
        _ output: [Float], fixture: Fixture, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(output.count, fixture.input.count, "\(label): wrong length", file: file, line: line)
        guard output.count == fixture.input.count else { return }
        assertNoNonFinite(output, label, file: file, line: line)

        if fixture.isNormalizable {
            XCTAssertEqual(l2Norm(output), 1.0, accuracy: 1e-5,
                           "\(label): expected a unit vector", file: file, line: line)
        } else {
            XCTAssertEqual(output, fixture.input,
                           "\(label): a vector whose reciprocal norm is not representable must be returned unchanged",
                           file: file, line: line)
        }
    }

    /// Elementwise agreement between two implementations.
    private func assertParity(
        _ got: [Float], _ expected: [Float], label: String, accuracy: Float = 1e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, expected.count, "\(label): length mismatch", file: file, line: line)
        guard got.count == expected.count else { return }

        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<got.count {
            let d = Swift.abs(got[i] - expected[i])
            if d > worst || !d.isFinite { worst = d; worstIndex = i }
        }
        XCTAssertLessThanOrEqual(
            worst, accuracy,
            "\(label): max |Δ| = \(worst) at [\(worstIndex)] (got \(got[worstIndex]) vs \(expected[worstIndex]))",
            file: file, line: line
        )
    }

    // MARK: - Metal helpers

    private func makeContext() async throws -> Metal4Context {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        return try await Metal4Context()
    }

    /// Dispatch one of the single-vector normalize kernels in BasicOperations.metal
    /// by name, mirroring `Metal4ComputeEngine.normalize`'s dispatch geometry.
    private func runBasicNormalizeKernel(
        _ functionName: String,
        _ input: [Float],
        context: Metal4Context
    ) async throws -> [Float] {
        let dimension = input.count
        let pipeline = try await context.getPipeline(functionName: functionName)
        let inputBuffer = try await context.getBuffer(for: input)
        let outputBuffer = try await context.getBuffer(size: dimension * MemoryLayout<Float>.size)

        try await context.executeAndWait { _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 2)

            let threadsPerGroup = MTLSize(width: min(256, dimension), height: 1, depth: 1)
            let threadgroups = MTLSize(width: (dimension + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }

        return outputBuffer.copyData(as: Float.self, count: dimension)
    }

    // MARK: - CPU: VectorAccelerate fallbacks vs VectorCore

    /// Every VectorAccelerate CPU normalize path implements the same policy, and
    /// that policy is observationally identical to VectorCore's public
    /// `NormalizeKernels.normalizeUnchecked`.
    func testCPUFallbacksMatchVectorCore() async {
        let simdFallback = SIMDFallback()
        let provider = FallbackProvider()

        for fixture in Self.fixtures(dimension: 512) {
            let reference = StableNormalization.normalizedAccelerate(fixture.input)
            assertPolicy(reference, fixture: fixture, label: "StableNormalization.accelerate/\(fixture.name)")

            let simd = StableNormalization.normalizedSIMD(fixture.input)
            assertPolicy(simd, fixture: fixture, label: "StableNormalization.simd/\(fixture.name)")
            assertParity(simd, reference, label: "SIMD vs Accelerate/\(fixture.name)")

            assertParity(AccelerateFallback.normalize(fixture.input), reference,
                         label: "AccelerateFallback/\(fixture.name)")
            assertParity(provider.normalize(fixture.input), reference,
                         label: "FallbackProvider/\(fixture.name)")
            let async = await simdFallback.normalize(fixture.input)
            assertParity(async, reference, label: "SIMDFallback/\(fixture.name)")

            // VectorCore reference. The zero vector is skipped because
            // `normalizeUnchecked` carries `assert(maxAbs > 0)`, which traps in
            // debug builds; its documented release behavior (buffer untouched) is
            // what `isNormalizable == false` already asserts above.
            if fixture.name != "zero" {
                var buffer = fixture.input
                buffer.withUnsafeMutableBufferPointer { p in
                    NormalizeKernels.normalizeUnchecked(p.baseAddress!, dimension: p.count)
                }
                assertParity(reference, buffer, label: "VectorCore.normalizeUnchecked/\(fixture.name)")
            }
        }
    }

    // MARK: - GPU: vectorNormalize / normalizeVectors

    /// `Metal4ComputeEngine.normalize` (→ `vectorNormalize`) matches the CPU.
    func testVectorNormalizeMatchesCPU() async throws {
        let context = try await makeContext()
        let engine = try await Metal4ComputeEngine(context: context, configuration: .default)

        for fixture in Self.fixtures(dimension: 512) {
            let gpu = try await engine.normalize(fixture.input)
            assertPolicy(gpu, fixture: fixture, label: "vectorNormalize/\(fixture.name)")
            assertParity(gpu, StableNormalization.normalizedAccelerate(fixture.input),
                         label: "vectorNormalize vs CPU/\(fixture.name)")
        }
    }

    /// The `normalizeVectors` alias kernel must not drift from `vectorNormalize`.
    func testNormalizeVectorsAliasMatchesVectorNormalize() async throws {
        let context = try await makeContext()

        for fixture in Self.fixtures(dimension: 512) {
            let alias = try await runBasicNormalizeKernel("normalizeVectors", fixture.input, context: context)
            let primary = try await runBasicNormalizeKernel("vectorNormalize", fixture.input, context: context)

            assertPolicy(alias, fixture: fixture, label: "normalizeVectors/\(fixture.name)")
            assertParity(alias, primary, label: "normalizeVectors vs vectorNormalize/\(fixture.name)", accuracy: 0)
        }
    }

    /// Non-power-of-two dimensions exercise the ragged tail of the threadgroup
    /// reductions (the old `stride = tgSize/2` tree dropped lanes there).
    func testVectorNormalizeRaggedDimensions() async throws {
        let context = try await makeContext()
        let engine = try await Metal4ComputeEngine(context: context, configuration: .default)

        for dimension in [3, 5, 17, 100, 257, 768] {
            var rng = TestRNG(seed: UInt64(dimension) &+ 7)
            let input = (0..<dimension).map { _ in rng.nextFloat(in: -1...1) }

            let gpu = try await engine.normalize(input)
            assertNoNonFinite(gpu, "vectorNormalize/dim=\(dimension)")
            XCTAssertEqual(l2Norm(gpu), 1.0, accuracy: 1e-5, "vectorNormalize/dim=\(dimension)")
            assertParity(gpu, StableNormalization.normalizedAccelerate(input),
                         label: "vectorNormalize vs CPU/dim=\(dimension)")
        }
    }

    // MARK: - GPU: l2_normalize_* family

    /// Specialized (512) and general (384, non-specialized) kernels, default epsilon.
    func testL2NormalizationMatchesCPU_defaultEpsilon() async throws {
        let context = try await makeContext()
        let kernel = try await L2NormalizationKernel(context: context)

        for dimension in [512, 384] {
            let fixtures = Self.fixtures(dimension: dimension)
            let result = try await kernel.normalize(fixtures.map(\.input), storeNorms: true)
            let outputs = result.asArrays()
            let norms = try XCTUnwrap(result.normsAsArray())

            for (i, fixture) in fixtures.enumerated() {
                let label = "l2_normalize(dim=\(dimension))/\(fixture.name)"
                assertPolicy(outputs[i], fixture: fixture, label: label)
                assertParity(outputs[i], StableNormalization.normalizedAccelerate(fixture.input),
                             label: "\(label) vs CPU")

                // Stored norms are the true ‖v‖₂ — never +Inf for huge inputs.
                XCTAssertTrue(norms[i].isFinite, "\(label): stored norm \(norms[i]) is not finite")
                let expected = l2Norm(fixture.input)
                if expected > 1e-30 {
                    XCTAssertEqual(Double(norms[i]) / expected, 1.0, accuracy: 1e-4,
                                   "\(label): stored norm \(norms[i]) vs true \(expected)")
                }
            }
        }
    }

    /// `epsilon: 0` must never produce Inf/NaN — the reciprocal is only formed
    /// after the kernel has proved it representable.
    func testL2NormalizationMatchesCPU_epsilonZero() async throws {
        let context = try await makeContext()
        let kernel = try await L2NormalizationKernel(context: context)

        let fixtures = Self.fixtures(dimension: 512)
        let result = try await kernel.normalize(fixtures.map(\.input), storeNorms: true, epsilon: 0)
        let outputs = result.asArrays()

        for (i, fixture) in fixtures.enumerated() {
            let label = "l2_normalize(eps=0)/\(fixture.name)"
            assertPolicy(outputs[i], fixture: fixture, label: label)
            assertParity(outputs[i], StableNormalization.normalizedAccelerate(fixture.input),
                         label: "\(label) vs CPU")
        }
    }

    /// The in-place kernel follows the same policy as the out-of-place ones.
    func testL2NormalizationInPlaceMatchesCPU() async throws {
        let context = try await makeContext()
        let kernel = try await L2NormalizationKernel(context: context)

        let dimension = 512
        let fixtures = Self.fixtures(dimension: dimension)
        let flat = fixtures.flatMap(\.input)

        let buffer = try await context.getBuffer(for: flat)
        let vectors = buffer.buffer
        let parameters = L2NormalizationParameters(
            numVectors: fixtures.count,
            dimension: dimension,
            epsilon: 0,
            storeNorms: false
        )

        try await context.executeAndWait { _, encoder in
            kernel.encodeInPlace(into: encoder, vectors: vectors, norms: nil, parameters: parameters)
        }

        let output = buffer.copyData(as: Float.self, count: flat.count)
        for (i, fixture) in fixtures.enumerated() {
            let slice = Array(output[(i * dimension)..<((i + 1) * dimension)])
            let label = "l2_normalize_inplace/\(fixture.name)"
            assertPolicy(slice, fixture: fixture, label: label)
            assertParity(slice, StableNormalization.normalizedAccelerate(fixture.input),
                         label: "\(label) vs CPU")
        }
    }

    /// A positive `epsilon` keeps its documented meaning: vectors at or below the
    /// threshold are emitted as zeros (an explicit caller policy, no longer a
    /// numerical-stability crutch).
    func testL2NormalizationEpsilonThresholdZeroesSmallVectors() async throws {
        let context = try await makeContext()
        let kernel = try await L2NormalizationKernel(context: context)

        let dimension = 512
        let micro = [Float](repeating: 1e-20, count: dimension)   // ‖v‖ = 2.26e-19
        let ordinary = [Float](repeating: 1.0, count: dimension)  // ‖v‖ = 22.6

        let result = try await kernel.normalize([micro, ordinary], storeNorms: true, epsilon: 1e-8)
        let outputs = result.asArrays()

        XCTAssertEqual(outputs[0], [Float](repeating: 0, count: dimension),
                       "epsilon=1e-8 must zero a vector with ‖v‖ = 2.26e-19")
        XCTAssertEqual(l2Norm(outputs[1]), 1.0, accuracy: 1e-5)

        // With epsilon 0 the same micro vector normalizes, matching the CPU.
        let unthresholded = try await kernel.normalize([micro], epsilon: 0).asArrays()[0]
        XCTAssertEqual(l2Norm(unthresholded), 1.0, accuracy: 1e-5)
        assertParity(unthresholded, StableNormalization.normalizedAccelerate(micro),
                     label: "micro(eps=0) vs CPU")
    }
}
