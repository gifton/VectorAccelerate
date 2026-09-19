import Foundation
import Metal
import VectorTestKit

enum BenchMode: String {
    case smoke, quick, full
    var samples: Int { switch self { case .smoke: return 1; case .quick: return 3; case .full: return 5 } }
    var iterations: Int { switch self { case .smoke: return 3; case .quick: return 20; case .full: return 100 } }
}

struct RunOptions: Sendable {
    var mode: BenchMode
    var samples: Int
    var iterations: Int
    var seed: UInt64
    var validationLayers: Bool
    var requireGPU: Bool
    var probeSlowdown: Int
    var label: String?
}

struct GPUSample: Sendable {
    var kind: TimingKind
    var startNs: Double
    var endNs: Double
    var label: String
    var durationNs: Double { endNs - startNs }
}

struct IterationOutput: @unchecked Sendable {
    var gpu: [GPUSample]
    var payload: Any?
}

/// Adapter seam (spec §5.1 / interface contract §4).
protocol BenchCase: Sendable {
    var id: CaseID { get }
    var identity: CaseIdentity { get }
    var providedTiming: Set<TimingKind> { get }
    var unitCount: Int { get }
    func prepare(_ ctx: BenchContext) async throws -> any PreparedCase
}

protocol PreparedCase: AnyObject {
    func runOnce() async throws -> IterationOutput
    func verify(_ output: IterationOutput) throws -> Correctness
    func resources() -> Resources?
    func resetForSample() async throws
}

final class BenchContext: @unchecked Sendable {
    let device: (any MTLDevice)?
    let host: HostFingerprint
    let gpu: GPUInfo?
    let options: RunOptions

    init(device: (any MTLDevice)?, host: HostFingerprint, gpu: GPUInfo?, options: RunOptions) {
        self.device = device; self.host = host; self.gpu = gpu; self.options = options
    }

    func makeGenerator(stream: UInt64) -> VectorGenerator { VectorGenerator(seed: options.seed, stream: stream) }
}

enum BenchError: Error, CustomStringConvertible {
    case metalUnavailable
    case routeMismatch(expected: String, telemetry: String)
    case usage(String)
    var description: String {
        switch self {
        case .metalUnavailable: return "Metal device unavailable"
        case .routeMismatch(let e, let t): return "route mismatch: expected \(e); telemetry \(t)"
        case .usage(let m): return m
        }
    }
}
