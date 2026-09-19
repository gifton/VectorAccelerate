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
            record.identity = c.identity   // cases whose fixture depends on the run seed finalize identity in prepare()
            var wall: [[Double]] = [], cb: [[Double]] = [], enc: [[Double]] = []
            var lastOutput: IterationOutput?
            for _ in 0..<o.samples {
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
                wall.append(w); if !b.isEmpty { cb.append(b) }; if !e.isEmpty { enc.append(e) }
                if let last = lastOutput { record.correctness = try prepared.verify(last) }
            }
            func series(_ raw: [[Double]], bias: String?, stream: UInt64) -> Series? {
                guard raw.count == o.samples, raw.allSatisfy({ $0.count == o.iterations }) else { return nil }
                return Series(raw: raw, summary: Stats.summarize(raw.flatMap { $0 }, bootstrapSeed: o.seed &+ stream), bias: bias)
            }
            record.timing.wall = series(wall, bias: nil, stream: 1)
            if c.providedTiming.contains(.commandBuffer) { record.timing.commandBuffer = series(cb, bias: commandBufferBias, stream: 2) }
            if c.providedTiming.contains(.encoder) { record.timing.encoder = series(enc, bias: nil, stream: 3) }
            record.resources = prepared.resources()
        } catch BenchError.metalUnavailable {
            record.status = .unavailable; record.failureReason = "Metal device unavailable"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
        } catch {
            record.status = .failed; record.failureReason = "\(error)"
            record.warmupUsed = Array(repeating: 0, count: o.samples)
            record.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        }
        return record
    }
}
