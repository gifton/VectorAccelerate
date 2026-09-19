import Foundation
import VectorTestKit

enum Runner {
    static let commandBufferBias = "command-buffer duration includes submission and completion overhead (+2–10 µs measured on M3 Max, see research brief Appendix D)"

    static func run(_ c: any BenchCase, context: BenchContext, policy: WarmupPolicy = .standard) async -> CaseRecord {
        let o = context.options
        var record = CaseRecord(id: c.id.rawValue, identity: c.identity, status: .measured, failureReason: nil,
                                samples: o.samples, iterations: o.iterations, warmupUsed: [], unitCount: c.unitCount,
                                timing: Timing(wall: nil, commandBuffer: nil, encoder: nil), correctness: nil, resources: nil, notes: [])
        if o.probeSlowdown > 0 { record.notes.append("probe-slowdown=\(o.probeSlowdown)") }
        let clock = ContinuousClock()
        func ns(_ d: Duration) -> Double { Double(d.components.seconds) * 1e9 + Double(d.components.attoseconds) / 1e9 }

        do {
            let prepared = try await c.prepare(context)
            // Contract (unverified until adapters exist): an adapter whose identity depends on the run seed must be a class that
            // republishes `identity` after prepare() (the Task 15 distance case does this); struct adapters carry registration-time identity.
            record.identity = c.identity
            var wall: [[Double]] = [], cb: [[Double]] = [], enc: [[Double]] = []
            var lastOutput: IterationOutput?
            var anyCorrectnessFailed = false
            for k in 0..<o.samples {
                try await prepared.resetForSample()
                var warm: [Double] = []
                repeat {
                    let start = clock.now
                    _ = try await prepared.runOnce()
                    warm.append(ns(clock.now - start))
                } while !policy.shouldStop(after: warm)
                record.warmupUsed.append(warm.count)

                var w: [Double] = [], b: [Double] = [], e: [Double] = []
                for _ in 0..<o.iterations {
                    let start = clock.now
                    let out = try await prepared.runOnce()
                    w.append(ns(clock.now - start))
                    for s in out.gpu {
                        switch s.kind { case .commandBuffer: b.append(s.durationNs); case .encoder: e.append(s.durationNs); case .wall: break }
                    }
                    lastOutput = out
                }
                wall.append(w); cb.append(b); enc.append(e)   // every sample contributes a row, so `declared` below sees the true per-sample counts
                if let last = lastOutput {
                    // Worst status wins across samples: the latest details are kept, but a `.fail` in any sample is sticky.
                    var verified = try prepared.verify(last)
                    if verified.status == .fail { anyCorrectnessFailed = true; record.notes.append("correctness failed in sample \(k + 1)") }
                    if anyCorrectnessFailed { verified.status = .fail }
                    record.correctness = verified
                }
            }
            func series(_ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                guard raw.count == o.samples, raw.allSatisfy({ $0.count == o.iterations }) else { return nil }
                return Series(raw: raw, summary: Stats.summarize(raw.flatMap { $0 }, bootstrapSeed: o.seed &+ stream), bias: bias)
            }
            /// A declared kind that arrived in the wrong shape is dropped whole (no partial series) and the per-sample counts are recorded in `notes`.
            func declared(_ kind: TimingKind, _ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                if let s = series(raw, bias: bias, stream: stream) { return s }
                record.notes.append("\(kind.rawValue): declared but produced \(raw.map(\.count)) samples per sample against \(o.iterations) iterations; series omitted")
                return nil
            }
            record.timing.wall = series(wall, bias: nil, stream: 1)
            if c.providedTiming.contains(.commandBuffer) { record.timing.commandBuffer = declared(.commandBuffer, cb, bias: commandBufferBias, stream: 2) }
            if c.providedTiming.contains(.encoder) { record.timing.encoder = declared(.encoder, enc, bias: nil, stream: 3) }
            record.resources = prepared.resources()
        } catch BenchError.metalUnavailable {
            record.status = .unavailable; record.failureReason = "Metal device unavailable"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.correctness = nil
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        } catch {
            record.status = .failed; record.failureReason = "\(error)"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.correctness = nil
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        }
        return record
    }
}
