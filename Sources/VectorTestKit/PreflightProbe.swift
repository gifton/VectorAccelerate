import Foundation

/// Calibration probes run before and after every run (spec §4.10). CPU copy here; the GPU copy lives in
/// the executable because it needs Metal. GB/s counts read plus write: 2 × bytes / seconds / 1e9.
public enum PreflightProbe {
    /// Copy bandwidth from the fast 5th percentile of `repeats` timed copies, taken after a warm-up that ends only when BOTH
    /// `warmupRepeats` copies AND `warmupSeconds` of wall time have passed. GB/s counts read plus write.
    ///
    /// Why, from docs/superpowers/research/2026-09-18-preflight-probe-characterization.md (one M3 Max, never fully idle):
    /// a thread that was idle, blocked or waiting on subprocesses copies at as little as 0.52× its settled rate for up to 17
    /// copies (111.5 ms), so the previous best-of-3 with no warm-up read 56–117 GB/s across ten readings 0.3 s apart. Settled
    /// copy times are bimodal and alternate, which leaves the median in the valley between the modes. On identical data — ten
    /// trials of ten readings with no other process above 60 % CPU — the range of the ten readings was at most 1.76 % of their
    /// median for the 5th percentile of 100 copies (typically 1.08 %), against 1.89 % for the minimum, 2.11 % for the 10th
    /// percentile and 3.57 % for the median. With another session compiling, every estimator spread 2–11 %: that is the
    /// machine not being quiet, which the gate exists to report. Explicit, unverified contract on any other machine;
    /// exercised only by the opt-in `PreflightInstrumentCheckTests`.
    public static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 100, warmupRepeats: Int = 20, warmupSeconds: Double = 0.2) -> Double {
        precondition(bytes > 0, "cpuCopyGBps needs at least one byte")
        let src = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        let dst = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        defer { src.deallocate(); dst.deallocate() }
        memset(src, 0x5A, bytes); memset(dst, 0, bytes)
        let clock = ContinuousClock()
        let warmupStart = clock.now
        var warmed = 0
        while warmed < warmupRepeats || clock.now - warmupStart < .seconds(warmupSeconds) { memcpy(dst, src, bytes); warmed += 1 }
        var seconds: [Double] = []
        for _ in 0..<max(repeats, 1) {
            let d = clock.measure { memcpy(dst, src, bytes) }
            seconds.append(Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
        }
        blackHole(dst.load(as: UInt8.self))
        let fast = Stats.percentile(sorted: seconds.sorted(), p: 0.05)
        return fast > 0 ? 2.0 * Double(bytes) / fast / 1e9 : 0
    }

    public static func evaluate(cpu: StartEnd, gpu: StartEnd?, thermalStart: String, thermalEnd: String,
                                gpuDeviceExists: Bool, driftLimitPercent: Double = 5) -> PreflightRecord {
        func drift(_ se: StartEnd) -> Double { se.start > 0 ? abs(se.end - se.start) / se.start * 100 : 0 }
        var reasons: [String] = []
        let d = max(drift(cpu), gpu.map(drift) ?? 0)
        if d > driftLimitPercent { reasons.append("drift \(String(format: "%.1f", d))% exceeds \(String(format: "%.1f", driftLimitPercent))%") }
        if thermalStart != "nominal" { reasons.append("thermal state start=\(thermalStart) (not nominal)") }
        if thermalEnd != "nominal" { reasons.append("thermal state end=\(thermalEnd) (not nominal)") }
        if gpu == nil && gpuDeviceExists { reasons.append("GPU probe unavailable although a Metal device exists") }
        return PreflightRecord(cpuCopyGBps: cpu, gpuCopyGBps: gpu, driftPercent: d, quiet: reasons.isEmpty, reasons: reasons)
    }
}
