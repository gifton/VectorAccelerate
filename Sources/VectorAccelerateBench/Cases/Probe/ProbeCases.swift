import Foundation
import Metal
import VectorTestKit

struct GPUCopyProbeCase: BenchCase {
    let id = try! CaseID(validating: "probe.gpu_copy.bytes=\(GPUProbe.bytes)")
    var identity: CaseIdentity {
        CaseIdentity(id: id.rawValue,
                     fixtureDigest: ContentDigest.sha256([.int(GPUProbe.bytes), .string("pattern:(i&0xFFFF)*0.5+1")]),
                     generatorVersion: VectorGenerator.version, seed: 0, stream: 0,
                     route: "gpu.probe_copy", variant: nil, dataDescriptor: "sequential_u32_pattern")
    }
    let providedTiming: Set<TimingKind> = [.wall, .commandBuffer]
    let unitCount = GPUProbe.bytes

    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase {
        guard let device = ctx.device else { throw BenchError.metalUnavailable }
        return PreparedProbe(probe: try GPUProbe(device: device), loops: max(1, ctx.options.probeSlowdown))
    }

    final class PreparedProbe: PreparedCase {
        let probe: GPUProbe; let loops: Int
        init(probe: GPUProbe, loops: Int) { self.probe = probe; self.loops = loops }
        func runOnce() async throws -> IterationOutput {
            let t = try probe.dispatch(loops: loops)
            return IterationOutput(gpu: [GPUSample(kind: .commandBuffer, startNs: t.startSeconds * 1e9, endNs: t.endSeconds * 1e9, label: "copy_f32x4")], payload: nil)
        }
        func verify(_ output: IterationOutput) throws -> Correctness {
            let ok = probe.outputMatchesInput()
            return Correctness(status: ok ? .pass : .fail, comparator: "memcmp", tolerance: nil, maxAbsError: nil, maxRelError: nil,
                               recall: nil, recallPolicy: nil, notes: ok ? [] : ["dst != src after copy"])
        }
        func resources() -> Resources? {
            Resources(staticThreadgroupBytes: probe.staticThreadgroupBytes, maxThreadgroupBytes: probe.maxThreadgroupBytes, poolHighWaterBytes: nil, notes: [])
        }
        func resetForSample() async throws {}
    }
}

enum ProbeCases {
    static var all: [any BenchCase] { [GPUCopyProbeCase()] }
}
