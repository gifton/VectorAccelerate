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

    /// A run that produced no timings: non-measured records may carry zero counts and no series (`CaseRecord.validate()`).
    private func nonMeasured(_ status: CaseStatus, correctness: CorrectnessStatus = .pass) -> CaseRecord {
        var c = record(wall: [1, 1, 1], correctness: correctness)
        c.status = status; c.timing = Timing(wall: nil, commandBuffer: nil, encoder: nil)
        c.warmupUsed = []; c.samples = 0; c.iterations = 0
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
        // A present baseline whose environment key is unknown is reported as such, not as "no baseline".
        let nilKey = Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: nil)
        XCTAssertEqual(nilKey.status, .incomparable); XCTAssertEqual(nilKey.reasons, ["baseline environment key missing"])
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
        // Correctness precedes resources and both reasons are kept; with no correctness notes the reason is bare.
        let ordered = Compare.compare(new: record(wall: [1, 1, 1], correctness: .fail, resources: over), baseline: nil, newKey: key, baselineKey: nil)
        XCTAssertEqual(ordered.status, .correctnessFailed)
        XCTAssertTrue(ordered.reasons.contains("correctness failed"), "\(ordered.reasons)")
        XCTAssertTrue(ordered.reasons.contains("resource limit exceeded: staticThreadgroupBytes 40000 > 32768"), "\(ordered.reasons)")
        var withNotes = record(wall: [1, 1, 1], correctness: .fail); withNotes.correctness?.notes = ["row 3 differs", "row 7 differs"]
        XCTAssertEqual(Compare.compare(new: withNotes, baseline: nil, newKey: key, baselineKey: nil).reasons.first, "correctness failed: row 3 differs; row 7 differs")
    }

    func testInvalidArtifact() {
        var bad = record(wall: [1, 1, 1]); bad.warmupUsed = []
        XCTAssertEqual(Compare.compare(new: bad, baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key).status, .invalidArtifact)
        XCTAssertEqual(Compare.compare(new: record(wall: [1, 1, 1]), baseline: bad, newKey: key, baselineKey: key).status, .invalidArtifact)
    }

    func testNonMeasuredNewRecordIsIncomparable() {
        for status in [CaseStatus.failed, .unavailable] {
            let r = Compare.compare(new: nonMeasured(status), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
            XCTAssertEqual(r.status, .incomparable, status.rawValue); XCTAssertTrue(r.deltas.isEmpty, status.rawValue)
            XCTAssertEqual(r.reasons, ["no comparable timing kinds (new status \(status.rawValue), baseline status measured)"])
        }
        // A correctness failure on such a record still wins over incomparable.
        let failed = Compare.compare(new: nonMeasured(.failed, correctness: .fail), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertTrue(failed.reasons.contains { $0.hasPrefix("no comparable timing kinds") }, "\(failed.reasons)")
    }

    func testZeroMedianBaselineIsIncomparable() {
        let r = Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [0, 0, 0]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .incomparable); XCTAssertTrue(r.deltas.isEmpty)
        XCTAssertEqual(r.reasons, ["no comparable timing kinds (new status measured, baseline status measured)"])
    }

    func testExitCodes() {
        let ok = CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])
        let alert = CompareResult(id: "a.b", status: .alertSlower, deltas: [], reasons: [])
        let faster = CompareResult(id: "a.b", status: .alertFaster, deltas: [], reasons: [])
        let fail = CompareResult(id: "a.b", status: .correctnessFailed, deltas: [], reasons: [])
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, alert], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, faster], strictPerf: false), 0)
        XCTAssertEqual(Compare.exitCode([ok, faster], strictPerf: true), 1)
        XCTAssertEqual(Compare.exitCode([ok, fail], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .invalidArtifact, deltas: [], reasons: [])], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .resourceFailed, deltas: [], reasons: [])], strictPerf: false), 1)
        XCTAssertEqual(Compare.exitCode([CompareResult(id: "a.b", status: .incomparable, deltas: [], reasons: [])], strictPerf: true), 0)
    }

    func testRenderFormats() throws {
        let r = Compare.compare(new: record(wall: [104, 104, 104]), baseline: record(wall: [100, 100, 100]), newKey: key, baselineKey: key)
        XCTAssertTrue(Compare.render([r], format: "text").contains("alert(slower)"))
        XCTAssertTrue(Compare.render([r], format: "markdown").contains("| probe.gpu_copy.bytes=8 |"))
        let decoded = try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([r], format: "json").utf8))
        XCTAssertEqual(decoded, [r])
    }

    func testMarkdownWithMultipleResultsKeepsTableContiguous() throws {
        let a = Compare.compare(new: record("probe.a", wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        let b = Compare.compare(new: record("probe.b", wall: [1, 1, 1]), baseline: nil, newKey: key, baselineKey: nil)
        let out = Compare.render([a, b], format: "markdown")
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false)
        // Header, separator, and one placeholder row per result form one contiguous block.
        let table = lines.indices.filter { lines[$0].hasPrefix("|") }
        XCTAssertEqual(table.count, 4, out)
        XCTAssertEqual(table, Array(try XCTUnwrap(table.first)...(try XCTUnwrap(table.last))), "table rows must be contiguous:\n\(out)")
        XCTAssertTrue(lines.contains("| probe.a | incomparable | — | — | — | — | — |"), out)
        XCTAssertTrue(lines.contains("| probe.b | incomparable | — | — | — | — | — |"), out)
        // Reasons come after the table, separated by one blank line, one newline-terminated bullet each.
        let lastRow = try XCTUnwrap(table.last)
        XCTAssertEqual(lines[lastRow + 1], "", out)
        let bullets = lines.indices.filter { lines[$0].hasPrefix("- ") }
        XCTAssertTrue(bullets.allSatisfy { $0 > lastRow }, out)
        XCTAssertEqual(bullets.map { String(lines[$0]) }, ["- probe.a: no baseline for probe.a under k1", "- probe.b: no baseline for probe.b under k1"])
        XCTAssertTrue(out.hasSuffix("\n"), out)
    }

    func testTextRenderDoesNotTruncateLongIDs() {
        let long = "distance.l2.batch.n=10000.d=768.k=100.nprobe=32.isolated_variant"
        XCTAssertGreaterThan(long.count, 44)
        let out = Compare.render([CompareResult(id: long, status: .ok, deltas: [], reasons: [])], format: "text")
        XCTAssertTrue(out.contains(long), out)
        XCTAssertTrue(out.hasPrefix("\(long) ok\n"), "a long id is printed in full followed by one space:\n\(out)")
        let short = Compare.render([CompareResult(id: "a.b", status: .ok, deltas: [], reasons: [])], format: "text")
        XCTAssertTrue(short.hasPrefix("a.b" + String(repeating: " ", count: 41) + " ok\n"), "a short id is still padded to 44 columns:\n\(short)")
    }
}
