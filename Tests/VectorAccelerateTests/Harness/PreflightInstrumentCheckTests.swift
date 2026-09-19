import XCTest
import VectorTestKit

/// Opt-in instrument check: skipped unless the environment has `VA_INSTRUMENT_CHECK=1`, so the default suite cannot flake on a
/// loaded machine. It closes, for the CPU probe, the class "a preflight probe whose own repeatability is comparable to the 5 %
/// quiet gate": ten readings separated by 0.3 s idle sleeps (the gaps that used to land the reading on the post-idle ramp) must
/// span less than 2.5 % of their median, half the gate. A failure means either the instrument or the machine moved by that
/// much during the ten readings; it does not say which.
///
/// It does NOT cover the GPU probe: `GPUProbe` lives in the executable target, which has no test target. That probe's
/// repeatability rests only on the recorded runs in docs/superpowers/research/2026-09-18-preflight-probe-characterization.md.
final class PreflightInstrumentCheckTests: XCTestCase {
    func testCPUProbeReadingsSpanLessThanHalfTheQuietGate() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VA_INSTRUMENT_CHECK"] == "1",
                          "instrument check is opt-in: set VA_INSTRUMENT_CHECK=1")
        var readings: [Double] = []
        for _ in 0..<10 {
            Thread.sleep(forTimeInterval: 0.3)
            readings.append(PreflightProbe.cpuCopyGBps())
        }
        let sorted = readings.sorted()
        let median = Stats.percentile(sorted: sorted, p: 0.5)
        let spread = (sorted[sorted.count - 1] - sorted[0]) / median
        print("VA_INSTRUMENT_CHECK cpuCopyGBps GB/s: " + readings.map { String(format: "%.1f", $0) }.joined(separator: " ")
              + String(format: " | (max-min)/median = %.2f %%", spread * 100))
        XCTAssertGreaterThan(median, 0)
        XCTAssertLessThan(spread, 0.025, "ten CPU-probe readings spanned \(spread * 100) % of their median; the quiet gate is 5 %")
    }
}
