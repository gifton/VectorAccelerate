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
//      den   = clamp(maxAbs, 0x1p-126, 0x1p126) ; scale = 1/den
//      sNorm = sqrt(Σ (v·scale)²) = ‖v‖ · scale
//      out   = (v · scale) / sNorm              iff 0.5 < sNorm < 0x1p100
//
//  The division is literal (precise::divide on GPU, vDSP/SIMD division on CPU) —
//  fast math would reassociate a (v·scale)·(1/sNorm) form back into the subnormal
//  multiplier v·(scale/sNorm); the two-sided guard excludes both unrepresentable
//  reciprocals (lower leg) and Inf-contaminated sNorm (upper leg).
//
//  Invariants asserted here, for every fixture and every path:
//    * no NaN/Inf ever reaches the output,
//    * ‖out‖₂ = 1 ± 1e-5 whenever the vector is normalizable — including vectors
//      whose ‖v‖₂ is not itself representable in FP32 (elements ≥ 1e38), since
//      1/‖v‖₂ is never formed,
//    * the input is returned BIT-EXACTLY unchanged otherwise (zero vector, deep
//      subnormals) — the observable behavior of `NormalizeKernels.normalizeUnchecked`,
//    * GPU and CPU agree elementwise within 1e-5,
//    * the runtime-compiled Metal library (the only library load path in release
//      builds) still contains every normalize kernel.
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
        /// True when the vector is normalizable, i.e. the implementation must
        /// return a unit vector. False for the classes every implementation
        /// (CPU and GPU) passes through unchanged.
        let isNormalizable: Bool
        /// False where VectorCore's own `normalizeUnchecked` is known to be wrong,
        /// so cross-checking VectorAccelerate against it would enshrine that bug.
        let crossCheckVectorCore: Bool
    }

    /// 1e-40 is subnormal in FP32 (below `Float.leastNormalMagnitude` = 1.18e-38).
    private static let subnormal = Float(1e-40 as Double)

    /// Ordinary values with one non-finite component planted at `index`.
    private static func infixed(_ dimension: Int, at index: Int, value: Float) -> [Float] {
        var v = [Float](repeating: 0, count: dimension)
        var rng = TestRNG(seed: 0x11F0000)
        for i in 0..<dimension { v[i] = rng.nextFloat(in: -1...1) }
        v[index] = value
        return v
    }

    private static func fixtures(dimension: Int) -> [Fixture] {
        precondition(dimension >= 8)

        // Subnormal background with unit spikes → maxAbs = 1 (normal) while most
        // components are subnormal: exercises the pre-scale on a mixed vector.
        var mixed = [Float](repeating: subnormal, count: dimension)
        for i in stride(from: 0, to: dimension, by: Swift.max(1, dimension / 8)) { mixed[i] = 1.0 }

        var rng = TestRNG(seed: 0xBE30044)
        var random = [Float](repeating: 0, count: dimension)
        for i in 0..<dimension { random[i] = rng.nextFloat(in: -1...1) }

        return [
            // ‖v‖ = 2.26e-39 → 1/‖v‖ = 4.4e38 > FLT_MAX: not normalizable in FP32.
            Fixture(name: "all-subnormal(1e-40)",
                    input: [Float](repeating: subnormal, count: dimension),
                    isNormalizable: false, crossCheckVectorCore: true),
            // Naive Σ v² = 5.1e-38 survives, but the old GPU epsilon (1e-7/1e-8)
            // rejected the 2.26e-19 norm and returned zeros / the input unchanged.
            Fixture(name: "micro(1e-20)",
                    input: [Float](repeating: 1e-20, count: dimension),
                    isNormalizable: true, crossCheckVectorCore: true),
            // Naive Σ v² = 5.1e40 → +Inf on the old path → all-zero output.
            Fixture(name: "huge(1e19)",
                    input: [Float](repeating: 1e19, count: dimension),
                    isNormalizable: true, crossCheckVectorCore: true),
            // ‖v‖ = 9.1e37 > 2^126, so 1/‖v‖ = 1.1e-38 is SUBNORMAL. Forming it on
            // the GPU (denormals-are-zero) flushed it to 0 and the vector was
            // misread as degenerate; the output is now built in the scaled domain.
            Fixture(name: "big(4e36)",
                    input: [Float](repeating: 4e36, count: dimension),
                    isNormalizable: true, crossCheckVectorCore: true),
            // maxAbs > 2^126, so the pre-scale `1/maxAbs` is itself subnormal — the
            // upper clamp keeps it normal. ‖v‖ = 2.3e39 exceeds FLT_MAX entirely,
            // yet the unit vector is still exactly representable.
            // VectorCore's normalizeUnchecked returns all-zeros here (its `mag`
            // overflows to +Inf and 1/(+Inf) = 0 passes its `isFinite` guard), so it
            // is deliberately not used as the reference for this fixture.
            Fixture(name: "overflow-norm(1e38)",
                    input: [Float](repeating: 1e38, count: dimension),
                    isNormalizable: true, crossCheckVectorCore: false),
            Fixture(name: "zero",
                    input: [Float](repeating: 0, count: dimension),
                    // `normalizeUnchecked` carries `assert(maxAbs > 0)`, which traps
                    // in debug builds; its release behavior (buffer untouched) is
                    // what `isNormalizable == false` already asserts.
                    isNormalizable: false, crossCheckVectorCore: false),
            Fixture(name: "mixed-subnormal-normal", input: mixed,
                    isNormalizable: true, crossCheckVectorCore: true),
            // Non-finite components. The upper `den` clamp keeps `scale` finite, so
            // a ±Inf component survives the pre-scale and makes sNorm = +Inf; the
            // guard's upper half sends it to the pass-through instead of computing
            // Inf/Inf = NaN. VectorCore agrees in release (its `mag` is NaN and
            // fails `guard mag > 0`) but carries `assert(mag > 0)`, which traps in
            // debug builds — hence no cross-check.
            Fixture(name: "single-+Inf", input: infixed(dimension, at: 3, value: .infinity),
                    isNormalizable: false, crossCheckVectorCore: false),
            Fixture(name: "single--Inf", input: infixed(dimension, at: dimension - 1, value: -.infinity),
                    isNormalizable: false, crossCheckVectorCore: false),
            Fixture(name: "all-Inf", input: [Float](repeating: .infinity, count: dimension),
                    isNormalizable: false, crossCheckVectorCore: false),
            // NaN fails the lower comparison (`NaN > 0.5` is false) and passes
            // through with its payload bits intact.
            Fixture(name: "single-NaN", input: infixed(dimension, at: 5, value: .nan),
                    isNormalizable: false, crossCheckVectorCore: false),
            Fixture(name: "ordinary-random", input: random,
                    isNormalizable: true, crossCheckVectorCore: true)
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

        // "No NaN/Inf" means none is *introduced*: a non-finite input is passed
        // through verbatim, so its own ±Inf/NaN lanes are expected in the output
        // (and are pinned by the bit-exact comparison below).
        if fixture.input.allSatisfy(\.isFinite) {
            assertNoNonFinite(output, label, file: file, line: line)
        }

        if fixture.isNormalizable {
            XCTAssertEqual(l2Norm(output), 1.0, accuracy: 1e-5,
                           "\(label): expected a unit vector", file: file, line: line)
        } else {
            // Bit patterns, not values: a subnormal that survived as a raw copy and
            // one that was flushed to +0 compare equal as Floats but are not the
            // same bits, and "unchanged" is a claim about the bits.
            XCTAssertEqual(output.map(\.bitPattern), fixture.input.map(\.bitPattern),
                           "\(label): a vector that is not normalizable must be returned bit-exactly unchanged",
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
            // Identical bits (including NaN payloads and ±Inf) are a perfect match;
            // subtracting them would yield NaN and look like a failure.
            if got[i].bitPattern == expected[i].bitPattern { continue }
            let d = Swift.abs(got[i] - expected[i])
            if d > worst || !d.isFinite { worst = d; worstIndex = i }
        }
        XCTAssertLessThanOrEqual(
            worst, accuracy,
            "\(label): max |Δ| = \(worst) at [\(worstIndex)] (got \(got[worstIndex]) vs \(expected[worstIndex]))",
            file: file, line: line
        )
    }

    /// `norms[]` must be the true ‖v‖₂ — or +Inf when ‖v‖₂ genuinely exceeds
    /// FLT_MAX, which is the honest answer (0.0 would be silently wrong).
    private func assertStoredNorm(
        _ stored: Float, input: [Float], label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let expected = l2Norm(input)
        if expected > Double(Float.greatestFiniteMagnitude) {
            XCTAssertTrue(stored.isInfinite && stored > 0,
                          "\(label): ‖v‖ = \(expected) exceeds FLT_MAX, expected +Inf, got \(stored)",
                          file: file, line: line)
        } else if expected > 1e-30 {
            XCTAssertTrue(stored.isFinite, "\(label): stored norm \(stored) is not finite", file: file, line: line)
            XCTAssertEqual(Double(stored) / expected, 1.0, accuracy: 1e-4,
                           "\(label): stored norm \(stored) vs true \(expected)", file: file, line: line)
        }
        // Subnormal norms may read 0 on a GPU that flushes denormals — not asserted.
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

    // MARK: - Release-configuration library guard

    /// The Metal library that `KernelContext` compiles from bundled `.metal`
    /// sources must contain every normalize kernel.
    ///
    /// This is the **only** library load path in release builds: `debug.metallib`
    /// is loaded under `#if DEBUG`, and no `default.metallib` ships in the bundle.
    /// That path strips `#include "Metal4Common.h"` and substitutes its own
    /// preamble, so a constant added to the header but not mirrored into
    /// `KernelContext`'s preamble makes the single combined source fail to compile
    /// — taking down *every* kernel in the package, in release only. This test runs
    /// in both configurations and fails loudly with the compiler diagnostic.
    func testRuntimeCompiledLibraryContainsNormalizeKernels() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle(),
                                   "VectorAccelerate resource bundle not found")

        let library = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)

        for name in [
            "vectorNormalize", "normalizeVectors", "batchNormalize",
            "l2_normalize_general_kernel", "l2_normalize_inplace_kernel",
            "l2_normalize_512_kernel", "l2_normalize_768_kernel", "l2_normalize_1536_kernel",
            // Sentinels used by KernelContext.isVectorAccelerateLibrary
            "l2_distance_kernel", "dot_product_kernel"
        ] {
            XCTAssertNotNil(library.makeFunction(name: name),
                            "runtime-compiled library is missing '\(name)'")
        }
    }

    // MARK: - CPU: VectorAccelerate fallbacks vs VectorCore

    /// Every VectorAccelerate CPU normalize path implements the same policy, and
    /// that policy is observationally identical to VectorCore's public
    /// `NormalizeKernels.normalizeUnchecked` wherever the latter is correct.
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

            guard fixture.crossCheckVectorCore else { continue }
            var buffer = fixture.input
            buffer.withUnsafeMutableBufferPointer { p in
                NormalizeKernels.normalizeUnchecked(p.baseAddress!, dimension: p.count)
            }
            assertParity(reference, buffer, label: "VectorCore.normalizeUnchecked/\(fixture.name)")
        }
    }

    // MARK: - GPU: vectorNormalize / normalizeVectors

    /// `Metal4ComputeEngine.normalize` (→ `vectorNormalize`) matches the CPU.
    /// Dimension 100 is deliberately ragged (non-power-of-two threadgroup).
    func testVectorNormalizeMatchesCPU() async throws {
        let context = try await makeContext()
        let engine = try await Metal4ComputeEngine(context: context, configuration: .default)

        for dimension in [512, 100] {
            for fixture in Self.fixtures(dimension: dimension) {
                let gpu = try await engine.normalize(fixture.input)
                assertPolicy(gpu, fixture: fixture, label: "vectorNormalize(dim=\(dimension))/\(fixture.name)")
                assertParity(gpu, StableNormalization.normalizedAccelerate(fixture.input),
                             label: "vectorNormalize(dim=\(dimension)) vs CPU/\(fixture.name)")
            }
        }
    }

    /// The `normalizeVectors` alias kernel must not drift from `vectorNormalize`.
    func testNormalizeVectorsAliasMatchesVectorNormalize() async throws {
        let context = try await makeContext()

        for fixture in Self.fixtures(dimension: 512) {
            let alias = try await runBasicNormalizeKernel("normalizeVectors", fixture.input, context: context)
            let primary = try await runBasicNormalizeKernel("vectorNormalize", fixture.input, context: context)

            assertPolicy(alias, fixture: fixture, label: "normalizeVectors/\(fixture.name)")
            XCTAssertEqual(alias.map(\.bitPattern), primary.map(\.bitPattern),
                           "normalizeVectors must be bit-identical to vectorNormalize/\(fixture.name)")
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

    /// The two magnitudes where a GPU that flushes denormals used to silently give
    /// up and pass the vector through: `1/‖v‖` subnormal (‖v‖ > 2^126) and the
    /// pre-scale `1/maxAbs` itself subnormal (maxAbs > 2^126).
    func testTopOfRangeMagnitudesAreNormalized() async throws {
        let context = try await makeContext()
        let engine = try await Metal4ComputeEngine(context: context, configuration: .default)
        let kernel = try await L2NormalizationKernel(context: context)

        let dimension = 512
        let magnitudes: [(String, Float)] = [
            ("4e36 (1/‖v‖ subnormal)", 4e36),
            ("2^126 (maxAbs at the clamp)", 0x1p126),
            ("1e38 (pre-scale subnormal, ‖v‖ > FLT_MAX)", 1e38),
            ("FLT_MAX", .greatestFiniteMagnitude)
        ]

        for (name, magnitude) in magnitudes {
            let input = [Float](repeating: magnitude, count: dimension)
            let cpu = StableNormalization.normalizedAccelerate(input)
            XCTAssertEqual(l2Norm(cpu), 1.0, accuracy: 1e-5, "CPU/\(name)")

            let basic = try await engine.normalize(input)
            assertNoNonFinite(basic, "vectorNormalize/\(name)")
            XCTAssertEqual(l2Norm(basic), 1.0, accuracy: 1e-5, "vectorNormalize/\(name)")
            XCTAssertNotEqual(basic.map(\.bitPattern), input.map(\.bitPattern),
                              "vectorNormalize/\(name): must not fall back to the pass-through")
            assertParity(basic, cpu, label: "vectorNormalize vs CPU/\(name)")

            let result = try await kernel.normalize([input], storeNorms: true)
            let l2 = result.asArrays()[0]
            assertNoNonFinite(l2, "l2_normalize/\(name)")
            XCTAssertEqual(l2Norm(l2), 1.0, accuracy: 1e-5, "l2_normalize/\(name)")
            assertParity(l2, cpu, label: "l2_normalize vs CPU/\(name)")
            assertStoredNorm(try XCTUnwrap(result.normsAsArray()).first ?? 0,
                             input: input, label: "l2_normalize/\(name)")
        }
    }

    // MARK: - GPU: l2_normalize_* family

    /// All three specialized kernels (512/768/1536), plus the general kernel at a
    /// non-specialized dimension (384) and a ragged one (100), at default epsilon.
    func testL2NormalizationMatchesCPU_defaultEpsilon() async throws {
        let context = try await makeContext()
        let kernel = try await L2NormalizationKernel(context: context)

        for dimension in [512, 768, 1536, 384, 100] {
            let fixtures = Self.fixtures(dimension: dimension)
            let result = try await kernel.normalize(fixtures.map(\.input), storeNorms: true)
            let outputs = result.asArrays()
            let norms = try XCTUnwrap(result.normsAsArray())

            for (i, fixture) in fixtures.enumerated() {
                let label = "l2_normalize(dim=\(dimension))/\(fixture.name)"
                assertPolicy(outputs[i], fixture: fixture, label: label)
                assertParity(outputs[i], StableNormalization.normalizedAccelerate(fixture.input),
                             label: "\(label) vs CPU")
                assertStoredNorm(norms[i], input: fixture.input, label: label)
            }
        }
    }

    /// `epsilon: 0` must never produce Inf/NaN — the reciprocal is only formed
    /// after the kernel has proved the vector normalizable.
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
