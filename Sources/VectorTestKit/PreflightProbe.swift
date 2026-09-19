import Foundation

/// Calibration probes run before and after every run (spec §4.10). CPU copy here; the GPU copy lives in
/// the executable because it needs Metal. GB/s counts read plus write: 2 × bytes / seconds / 1e9.
public enum PreflightProbe {
    public static func cpuCopyGBps(bytes: Int = 268_435_456, repeats: Int = 3) -> Double {
        let src = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        let dst = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        defer { src.deallocate(); dst.deallocate() }
        memset(src, 0x5A, bytes); memset(dst, 0, bytes)
        var best = Double.infinity
        let clock = ContinuousClock()
        for _ in 0..<max(repeats, 1) {
            let d = clock.measure { memcpy(dst, src, bytes) }
            let seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            best = min(best, seconds)
        }
        blackHole(dst.load(as: UInt8.self))
        return best > 0 ? 2.0 * Double(bytes) / best / 1e9 : 0
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
