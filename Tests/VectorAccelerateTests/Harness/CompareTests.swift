import XCTest
import VectorTestKit

/// Decision rule (spec §5.6 as amended): invalidArtifact → correctnessFailed → resourceFailed → incomparable →
/// per-kind deltas with boundary max(2.5%, 2×baseline RSD) → ok.
final class CompareTests: XCTestCase {
    private let key = "k1"

    private func record(_ id: String = "probe.gpu_copy.bytes=8", wall: [Double], correctness: CorrectnessStatus = .pass,
                        resources: Resources? = nil, route: String = "gpu.probe_copy") -> CaseRecord {
        var c = ArtifactSchemaTests.sampleArtifact().cases[0]
        c.id = id; c.identity.id = id; c.identity.route = route
        c.samples = 1; c.iterations = wall.count; c.warmupUsed = [3]
        c.timing = Timing(wall: Series(raw: [wall], summary: Stats.summarize(wall, bootstrapSeed: 1), bias: nil), commandBuffer: nil, encoder: nil)
        c.correctness?.status = correctness
        c.resources = resources
        return c
    }

    func testOk() {
        let r = Compare.compare(new: record(wall: [101, 101, 101]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .ok); XCTAssertEqual(r.deltas.count, 1)
        XCTAssertEqual(r.deltas[0].delta, 0.01, accuracy: 1e-12); XCTAssertEqual(r.deltas[0].boundary, 0.025); XCTAssertFalse(r.deltas[0].widened)
    }

    func testAlertSlowerAndFaster() {
        XCTAssertEqual(Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key).status, .alertSlower)
        XCTAssertEqual(Compare.compare(new: record(wall: [96, 96, 96]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key).status, .alertFaster)
    }

    func testBoundaryWidensWithBaselineRSD() {
        // baseline [90,100,110]: rsd 0.1 → boundary 0.2; +15% is ok and visibly widened.
        let r = Compare.compare(new: record(wall: [115, 115, 115]), baseline: record(wall: [90, 100, 110]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .ok); XCTAssertTrue(r.deltas[0].widened); XCTAssertEqual(r.deltas[0].boundary, 0.2, accuracy: 1e-12)
    }

    func testIncomparableIdentityKeyAndMissing() {
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], route: "cpu"), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .incomparable)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: "k2").status, .incomparable)
        let missing = Compare.compare(new: record(wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        XCTAssertEqual(missing.status, .incomparable); XCTAssertEqual(missing.reasons, ["no baseline for probe.gpu_copy.bytes=8 under k1"])
    }

    func testCorrectnessAndResourceFailuresWinEvenWithoutBaseline() {
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail), baseline: nil, newKey: key, baselineKey: nil).status, .correctnessFailed)
        let over = Resources(staticThreadgroupBytes: 40_000, maxThreadgroupBytes: 32_768, poolHighWaterBytes: nil, notes: [])
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], resources: over), baseline: nil, newKey: key, baselineKey: nil).status, .resourceFailed)
        let noted = Resources(staticThreadgroupBytes: nil, maxThreadgroupBytes: nil, poolHighWaterBytes: nil, notes: ["LIMIT_EXCEEDED: pool"])
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1], resources: noted), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .resourceFailed)
        // A correctness failure with a mismatched baseline still reports correctnessFailed and keeps the mismatch reason.
        let both = Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail, route: "cpu"), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
        XCTAssertEqual(both.status, .correctnessFailed); XCTAssertTrue(both.reasons.contains { $0.hasPrefix("identity differs") })
    }

    func testInvalidArtifact() {
        var bad = record(wall: [1, 1, 1]); bad.warmupUsed = []
        XCTAssertEqual(Compare.compare(new: bad, baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .invalidArtifact)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: bad, newKey: key, baselineKey: key).status, .invalidArtifact)
    }

    func testExitCodes() {
        let ok = CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])
        let alert = CompareResult(id: "a.b", status: .alertSlower, deltas: [], reasons: [])
        let fail = CompareResult(id: "a.b", status: .correctnessFailed, deltas: [], reasons: [])
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, fail], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .incomparable, deltas: [], reasons: [])], strictPerf: true), 0)
    }

    func testRenderFormats() {
        let r = Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertTrue(Compare.render([r], format: "text").contains("alert(slower)"))
        XCTAssertTrue(Compare.render([r], format: "markdown").contains("| probe.gpu_copy.bytes=8 |"))
        XCTAssertNoThrow(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([r], format: "json").utf8)))
    }
}
