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
        XCTAssertEqual(r.reasons, ["timing.wall: baseline median 0.0 is not a positive finite number"])
    }

    /// Finite medians can still give a non-finite delta or boundary: (1e308 − 1e-300) / 1e-300 overflows, and so does 2 × rsd for
    /// rsd = 1e308. A NaN median cannot come from a JSON file but can be built in memory. Each is incomparable for the whole case,
    /// never an alert, and the JSON rendering of such results decodes (it used to trap on `inf`).
    func testNonFiniteDeltaOrBoundaryIsIncomparable() throws {
        func withWall(median: Double? = nil, rsd: Double? = nil) -> CaseRecord {
            var edited = record(wall: [100, 100, 100])
            if let median { edited.timing.wall?.summary.median = median }
            if let rsd { edited.timing.wall?.summary.rsd = rsd }
            return edited
        }
        let overflow = Compare.compare(new: withWall(median: 1e308), baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(overflow.status, .incomparable); XCTAssertTrue(overflow.deltas.isEmpty)
        XCTAssertEqual(overflow.reasons.count, 1, "\(overflow.reasons)")
        let overflowReason = overflow.reasons.first ?? ""
        for piece in ["timing.wall", "1e+308", "1e-300", "not finite"] {
            XCTAssertTrue(overflowReason.contains(piece), "'\(piece)' missing from: \(overflow.reasons)")
        }

        let nan = Compare.compare(new: withWall(median: .nan), baseline: withWall(), newKey: key, baselineKey: key)
        XCTAssertEqual(nan.status, .incomparable); XCTAssertTrue(nan.deltas.isEmpty)
        let infiniteBase = Compare.compare(new: withWall(), baseline: withWall(median: .infinity), newKey: key, baselineKey: key)
        XCTAssertEqual(infiniteBase.status, .incomparable); XCTAssertTrue(infiniteBase.deltas.isEmpty)
        let boundary = Compare.compare(new: withWall(), baseline: withWall(rsd: 1e308), newKey: key, baselineKey: key)
        XCTAssertEqual(boundary.status, .incomparable); XCTAssertTrue(boundary.deltas.isEmpty)
        XCTAssertTrue(boundary.reasons.contains { $0.contains("baseline rsd 1e+308") }, "\(boundary.reasons)")

        // One non-finite kind makes the whole case incomparable, even beside a kind that would alert on its own.
        var mixedNew = withWall(median: 1e308), mixedBase = withWall(median: 1e-300)
        mixedNew.timing.commandBuffer = record(wall: [200, 200, 200]).timing.wall
        mixedBase.timing.commandBuffer = record(wall: [100, 100, 100]).timing.wall
        let mixed = Compare.compare(new: mixedNew, baseline: mixedBase, newKey: key, baselineKey: key)
        XCTAssertEqual(mixed.status, .incomparable); XCTAssertTrue(mixed.deltas.isEmpty)
        XCTAssertEqual(Compare.exitCode([mixed], strictPerf: true), 0)

        // A correctness failure still wins, and keeps the comparison reason.
        var failing = withWall(median: 1e308); failing.correctness?.status = .fail
        let failed = Compare.compare(new: failing, baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertTrue(failed.reasons.contains { $0.contains("not finite") }, "\(failed.reasons)")

        let all = [overflow, nan, infiniteBase, boundary, mixed, failed]
        XCTAssertEqual(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render(all, format: "json").utf8)), all)
        // No delta line is rendered for these results (a reason may still quote an infinite median).
        let text = Compare.render(all, format: "text")
        XCTAssertFalse(text.contains("inf%") || text.contains("nan%") || text.contains("boundary"), text)
    }

    /// A median that is not a positive finite number cannot anchor a ratio. On the NEW side that used to be finite garbage:
    /// 0 against 100 gave delta −1 and `alert(faster)`. A GPU timing pair can come back 0 from a real run. Both sides now make
    /// the whole case incomparable, in parallel wording that names the kind, the side and the value.
    func testNonPositiveMedianOnEitherSideIsIncomparable() {
        func withWall(median: Double) -> CaseRecord {
            var edited = record(wall: [100, 100, 100]); edited.timing.wall?.summary.median = median
            return edited
        }
        let good = record(wall: [100, 100, 100])
        for value in [0.0, -5.0, -Double.infinity, Double.nan] {
            let newSide = Compare.compare(new: withWall(median: value), baseline: good, newKey: key, baselineKey: key)
            XCTAssertEqual(newSide.status, .incomparable, "new median \(value)"); XCTAssertTrue(newSide.deltas.isEmpty, "new median \(value)")
            XCTAssertEqual(newSide.reasons, ["timing.wall: new median \(value) is not a positive finite number"])
            XCTAssertEqual(Compare.exitCode([newSide], strictPerf: true), 0)
            let baseSide = Compare.compare(new: good, baseline: withWall(median: value), newKey: key, baselineKey: key)
            XCTAssertEqual(baseSide.status, .incomparable, "baseline median \(value)")
            XCTAssertTrue(baseSide.deltas.isEmpty, "baseline median \(value)")
            XCTAssertEqual(baseSide.reasons, ["timing.wall: baseline median \(value) is not a positive finite number"])
        }
        // A measured run whose timings are all zero takes the same path (no hand-edited summary needed).
        let measuredZero = Compare.compare(new: record(wall: [0, 0, 0]), baseline: good, newKey: key, baselineKey: key)
        XCTAssertEqual(measuredZero.status, .incomparable)
        XCTAssertEqual(measuredZero.reasons, ["timing.wall: new median 0.0 is not a positive finite number"])
        // Both sides unusable: one reason per side, new first.
        let both = Compare.compare(new: withWall(median: 0), baseline: withWall(median: -1), newKey: key, baselineKey: key)
        XCTAssertEqual(both.reasons, ["timing.wall: new median 0.0 is not a positive finite number",
                                      "timing.wall: baseline median -1.0 is not a positive finite number"])

        // One unusable kind makes the whole case incomparable on either side, even beside a kind that would alert on its own.
        let alertingWall = record(wall: [200, 200, 200])
        var zeroGPUNew = alertingWall, goodGPUBase = good
        zeroGPUNew.timing.commandBuffer = record(wall: [0, 0, 0]).timing.wall
        goodGPUBase.timing.commandBuffer = good.timing.wall
        let mixedNew = Compare.compare(new: zeroGPUNew, baseline: goodGPUBase, newKey: key, baselineKey: key)
        XCTAssertEqual(mixedNew.status, .incomparable); XCTAssertTrue(mixedNew.deltas.isEmpty)
        XCTAssertEqual(mixedNew.reasons, ["timing.commandBuffer: new median 0.0 is not a positive finite number"])
        var goodGPUNew = alertingWall, zeroGPUBase = good
        goodGPUNew.timing.commandBuffer = good.timing.wall
        zeroGPUBase.timing.commandBuffer = record(wall: [0, 0, 0]).timing.wall
        let mixedBase = Compare.compare(new: goodGPUNew, baseline: zeroGPUBase, newKey: key, baselineKey: key)
        XCTAssertEqual(mixedBase.status, .incomparable); XCTAssertTrue(mixedBase.deltas.isEmpty)
        XCTAssertEqual(mixedBase.reasons, ["timing.commandBuffer: baseline median 0.0 is not a positive finite number"])

        // A kind present on one side only is still just not compared; the boundary rule is untouched for usable medians.
        var wallOnlyBase = good; wallOnlyBase.timing.commandBuffer = nil
        XCTAssertEqual(Compare.compare(new: zeroGPUNew, baseline: wallOnlyBase, newKey: key, baselineKey: key).status, .alertSlower)
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
