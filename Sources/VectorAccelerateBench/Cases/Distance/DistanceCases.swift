import Foundation
import Metal
import VectorCore
import VectorAccelerate
import VectorTestKit

/// Batch L2 distance through the public façade at N=10,000, D=768 (spec §5.7). Route is verified per call via
/// routing telemetry; `.commandBuffer` timing is the context's last GPU timing read immediately after the single submission.
///
/// `@unchecked Sendable` is a contract on the caller, not a verified property (no test exercises it). `cachedIdentity` is the
/// only mutable stored property and it is unguarded: `prepare` writes it, `identity` reads it. `Runner.run` touches one case
/// strictly in sequence inside one task — it reads `c.identity` to build the `CaseRecord`, then `try await c.prepare(context)`,
/// then `record.identity = c.identity` — and `RunCommand` awaits each `Runner.run` before starting the next, on an instance that
/// `DistanceCases.all` created for that `Registry.select` call alone. A caller that shares one instance between concurrent
/// `Runner.run` calls, or reads `identity` while `prepare` is in flight, has a data race.
final class L2BatchDistanceCase: BenchCase, @unchecked Sendable {
    static let n = 10_000, d = 768
    static let candidateStream: UInt64 = 10, queryStream: UInt64 = 11

    let id = try! CaseID(validating: "distance.l2.batch.n=\(L2BatchDistanceCase.n).d=\(L2BatchDistanceCase.d)")
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = L2BatchDistanceCase.n

    /// Finalized in `prepare` from the run seed; before that it describes the default seed 42 fixture.
    private var cachedIdentity: CaseIdentity?
    var identity: CaseIdentity { cachedIdentity ?? Self.makeIdentity(seed: 42, fixture: Self.fixture(seed: 42)) }

    static func fixture(seed: UInt64) -> (candidates: [[Float]], query: [Float]) {
        var g = VectorGenerator(seed: seed, stream: candidateStream)
        let candidates = g.uniformVectors(count: n, dimension: d)
        var q = VectorGenerator(seed: seed, stream: queryStream)
        let query = q.uniformVectors(count: 1, dimension: d)[0]
        return (candidates, query)
    }

    static func makeIdentity(seed: UInt64, fixture f: (candidates: [[Float]], query: [Float])) -> CaseIdentity {
        CaseIdentity(id: "distance.l2.batch.n=\(n).d=\(d)",
                     fixtureDigest: ContentDigest.sha256([.int(n), .int(d), .floats(f.candidates.flatMap { $0 }), .floats(f.query)]),
                     generatorVersion: VectorGenerator.version, seed: seed, stream: candidateStream,
                     route: "gpu.l2.batch", variant: nil, dataDescriptor: "uniform[-1,1]")
    }

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        guard ctx.device != nil else { throw BenchError.metalUnavailable }
        let f = Self.fixture(seed: ctx.options.seed)
        cachedIdentity = Self.makeIdentity(seed: ctx.options.seed, fixture: f)
        let context = try await Metal4Context()
        let provider = try await MetalComputeProvider(context: context, configuration: .init(preferGPU: true, fallbackToCPU: false))
        return Prepared(context: context, provider: provider, query: DynamicVector(f.query), candidates: f.candidates.map { DynamicVector($0) },
                        oracleSquared: f.candidates.map { c in
                            var s = 0.0; for i in 0..<Self.d { let x = Double(f.query[i]) - Double(c[i]); s += x * x }; return s })
    }

    final class Prepared: PreparedCase {
        let context: Metal4Context; let provider: MetalComputeProvider
        let query: DynamicVector; let candidates: [DynamicVector]; let oracleSquared: [Double]
        // Reduction order inside the kernel is not documented, so the conservative sequential bound applies until SP2 derives
        // per-kernel tolerances. Components are in [-1,1] ⇒ each squared difference ≤ 4 ⇒ Σ|xᵢ| ≤ 4·D.
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
            VectorTestKit.blackHole(distances)   // qualified: VectorAccelerate exports an identical blackHole until Task 17 deletes Benchmarking/
            return IterationOutput(gpu: gpu, payload: distances)
        }

        func verify(_ output: IterationOutput) throws -> Correctness {
            guard let distances = output.payload as? [Float], distances.count == oracleSquared.count else {
                return Correctness(status: .fail, comparator: "squared-distance vs Double oracle", tolerance: tolerance, maxAbsError: nil, maxRelError: nil, recall: nil, recallPolicy: nil, notes: ["result count mismatch"])
            }
            var maxAbs = 0.0, maxRel = 0.0, failures = 0
            for (i, dist) in distances.enumerated() {
                let got = Double(dist) * Double(dist)
                switch Comparator.compare(got, oracleSquared[i], tolerance: tolerance) {
                case .equal, .withinTolerance:
                    let abs = Swift.abs(got - oracleSquared[i]); maxAbs = max(maxAbs, abs); maxRel = max(maxRel, oracleSquared[i] > 0 ? abs / oracleSquared[i] : 0)
                case .classMismatch: failures += 1
                case .outOfTolerance(let a, let r): failures += 1; maxAbs = max(maxAbs, a); maxRel = max(maxRel, r)
                }
            }
            return Correctness(status: failures == 0 ? .pass : .fail, comparator: "squared-distance vs Double oracle", tolerance: tolerance,
                               maxAbsError: maxAbs, maxRelError: maxRel, recall: nil, recallPolicy: nil,
                               notes: failures == 0 ? ["sequential fp32 sum bound, D=768, |x|≤1"] : ["\(failures) of \(distances.count) distances outside tolerance"])
        }

        func resources() -> Resources? { nil }
        func resetForSample() async throws { await provider.resetRoutingTelemetry() }
    }
}

enum DistanceCases {
    static var all: [any BenchCase] { [L2BatchDistanceCase()] }
}
