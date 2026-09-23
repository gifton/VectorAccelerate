import Foundation
import Metal
import Synchronization
import VectorCore
import VectorAccelerate
import VectorTestKit

/// Batch L2 distance through the public façade at N=10,000, D=768 (spec §5.7). Route is verified per call via routing telemetry.
///
/// `.commandBuffer` timing is `Metal4Context.lastGPUTiming` read right after the call. Contract, not derivable from this file:
/// that value covers the whole call only while `L2KernelDistanceProvider.batchDistance` makes exactly one `executeAndWait`
/// submission (Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:135) and `Metal4Context` sets or clears
/// `_lastGPUTiming` once per submission (Core/Metal4Context.swift:435-441). If that path ever chunks into several submissions,
/// this series silently reports only the last one. No test pins it.
final class L2BatchDistanceCase: BenchCase {
    static let n = 10_000, d = 768
    static let candidateStream: UInt64 = 10, queryStream: UInt64 = 11
    static let caseID = "distance.l2.batch.n=\(n).d=\(d)"

    let id = try! CaseID(validating: L2BatchDistanceCase.caseID)
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = L2BatchDistanceCase.n

    /// Published by `prepare` from the run seed before anything there can throw; until then `identity` describes the default-seed fixture.
    private let publishedIdentity = Mutex<CaseIdentity?>(nil)
    var identity: CaseIdentity {
        publishedIdentity.withLock { $0 } ?? Self.makeIdentity(seed: RunOptions.defaultSeed, fixture: Self.fixture(seed: RunOptions.defaultSeed))
    }

    static func fixture(seed: UInt64) -> (candidates: [[Float]], query: [Float]) {
        var g = VectorGenerator(seed: seed, stream: candidateStream)
        let candidates = g.uniformVectors(count: n, dimension: d)
        var q = VectorGenerator(seed: seed, stream: queryStream)
        let query = q.uniformVectors(count: 1, dimension: d)[0]
        return (candidates, query)
    }

    static func makeIdentity(seed: UInt64, fixture f: (candidates: [[Float]], query: [Float])) -> CaseIdentity {
        CaseIdentity(id: caseID,
                     fixtureDigest: ContentDigest.sha256([.int(n), .int(d), .floats(f.candidates.flatMap { $0 }), .floats(f.query)]),
                     generatorVersion: VectorGenerator.version, seed: seed, stream: candidateStream,
                     route: "gpu.l2.batch", variant: nil, dataDescriptor: "uniform[-1,1]")
    }

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        let f = Self.fixture(seed: ctx.options.seed)
        let seeded = Self.makeIdentity(seed: ctx.options.seed, fixture: f)
        publishedIdentity.withLock { $0 = seeded }   // first: `Runner.run` re-reads `identity` after `prepare` even when the lines below throw
        guard ctx.device != nil else { throw BenchError.metalUnavailable }
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(preferGPU: true, fallbackToCPU: false))
        return Prepared(context: context, provider: provider, query: DynamicVector(f.query), candidates: f.candidates.map { DynamicVector($0) },
                        oracleSquared: f.candidates.map { c in
                            var s = 0.0; for i in 0..<Self.d { let x = Double(f.query[i]) - Double(c[i]); s += x * x }; return s })
    }

    final class Prepared: PreparedCase {
        static let comparator = "squared-distance vs Double oracle"

        let context: Metal4Context; let provider: MetalComputeProvider
        let query: DynamicVector; let candidates: [DynamicVector]; let oracleSquared: [Double]
        // `verify` compares (Float distance)² in Double with the Double oracle Σ(qᵢ−cᵢ)². Components lie in [-1,1], so each term is ≤ 4
        // and Σ ≤ 4·D = 3072. `forSum(.sequential)` gives 2·u·D·3072 = 0.28125 with u = 2⁻²⁴. Error budget it has to cover, assuming
        // round-to-nearest fp32 operations in the kernel (unverified: the kernel's reduction order and fast-math setting are not
        // documented, which is also why the sequential bound is used until SP2 derives per-kernel tolerances):
        //   summation of D non-negative terms, any order   ≤ (D−1)·u·3072 = 767·u·3072
        //   per-term rounding (one subtract, one square)   ≤ D·3u·4       =   3·u·3072
        //   fp32 sqrt, then squaring that result in Double ≤ 2u·3072      =   2·u·3072
        // Total ≤ 772·u·3072 ≈ 0.1414, about half of 0.28125. The Double oracle's own error (≈ 768·2⁻⁵³·3072 ≈ 2.6e-10) is negligible.
        let tolerance = Tolerance.forSum(terms: L2BatchDistanceCase.d, reduction: .sequential, magnitude: 4 * Double(L2BatchDistanceCase.d))

        init(context: Metal4Context, provider: MetalComputeProvider, query: DynamicVector, candidates: [DynamicVector], oracleSquared: [Double]) {
            self.context = context; self.provider = provider; self.query = query; self.candidates = candidates; self.oracleSquared = oracleSquared
        }

        func runOnce() async throws -> IterationOutput {
            let before = await provider.routingTelemetry().gpuKernel
            let distances = try await provider.batchDistance(query: query, candidates: candidates, metric: .euclidean)
            let telemetry = await provider.routingTelemetry()
            guard telemetry.gpuKernel == before + 1 else { throw BenchError.routeMismatch(expected: "gpu.l2.batch", telemetry: "\(telemetry)") }
            var gpu: [GPUSample] = []
            if let t = await context.lastGPUTiming {
                gpu.append(GPUSample(kind: .commandBuffer, startNs: t.gpuStartTime * 1e9, endNs: t.gpuEndTime * 1e9, label: "l2_batch"))
            }
            blackHole(distances)
            return IterationOutput(gpu: gpu, payload: distances)
        }

        func verify(_ output: IterationOutput) throws -> Correctness {
            guard let distances = output.payload as? [Float], distances.count == oracleSquared.count else {
                return Correctness(status: .fail, comparator: Self.comparator, tolerance: tolerance, maxAbsError: nil, maxRelError: nil,
                                   recall: nil, recallPolicy: nil, notes: ["result count mismatch"])
            }
            // maxRelError uses the comparator's definition everywhere: |got − oracle| / max(|got|, |oracle|), 0 when both are 0.
            var maxAbs = 0.0, maxRel = 0.0, failures = 0, invalid = 0
            for (i, dist) in distances.enumerated() {
                // Squaring would hide a wrong sign, and a distance is never negative, NaN or infinite.
                guard dist.isFinite, dist >= 0 else { failures += 1; invalid += 1; continue }
                let got = Double(dist) * Double(dist), oracle = oracleSquared[i]
                let absError = Swift.abs(got - oracle), scale = max(Swift.abs(got), Swift.abs(oracle))
                maxAbs = max(maxAbs, absError); maxRel = max(maxRel, scale > 0 ? absError / scale : 0)
                switch Comparator.compare(got, oracle, tolerance: tolerance) {
                case .equal, .withinTolerance: break
                case .classMismatch, .outOfTolerance: failures += 1
                }
            }
            let notes = failures == 0 ? ["sequential fp32 sum bound, D=768, |x|≤1"]
                : ["\(failures) of \(distances.count) distances outside tolerance" + (invalid > 0 ? " (\(invalid) negative or non-finite)" : "")]
            return Correctness(status: failures == 0 ? .pass : .fail, comparator: Self.comparator, tolerance: tolerance,
                               maxAbsError: maxAbs, maxRelError: maxRel, recall: nil, recallPolicy: nil, notes: notes)
        }

        func resources() -> Resources? { nil }
        func resetForSample() async throws { await provider.resetRoutingTelemetry() }
    }
}

enum DistanceCases {
    static var all: [any BenchCase] { [L2BatchDistanceCase()] }
}
