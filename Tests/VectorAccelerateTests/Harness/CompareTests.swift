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
        // A non-measured record carries no series, so every kind the baseline carries is present on the baseline side only; the
        // status line follows those skip reasons and names the cause, so `compare` output still says the new run failed.
        let oneSided = "timing.wall not compared: present on the baseline side only"
        for status in [CaseStatus.failed, .unavailable] {
            let r = Compare.compare(new: nonMeasured(status), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
            XCTAssertEqual(r.status, .incomparable, status.rawValue); XCTAssertTrue(r.deltas.isEmpty, status.rawValue)
            XCTAssertEqual(r.reasons, [oneSided, "no comparable timing kinds (new status \(status.rawValue), baseline status measured)"], status.rawValue)
        }
        // The same on the baseline side, and with no kind on either side (nothing dropped: the statuses are the whole explanation).
        let baseSide = Compare.compare(new: record(wall: [1, 1, 1]), baseline: nonMeasured(.failed), newKey: key, baselineKey: key)
        XCTAssertEqual(baseSide.status, .incomparable); XCTAssertTrue(baseSide.deltas.isEmpty)
        XCTAssertEqual(baseSide.reasons, ["timing.wall not compared: present on the new side only",
                                          "no comparable timing kinds (new status measured, baseline status failed)"])
        let neither = Compare.compare(new: nonMeasured(.failed), baseline: nonMeasured(.unavailable), newKey: key, baselineKey: key)
        XCTAssertEqual(neither.status, .incomparable); XCTAssertTrue(neither.deltas.isEmpty)
        XCTAssertEqual(neither.reasons, ["no comparable timing kinds (new status failed, baseline status unavailable)"])
        // A correctness failure on such a record still wins over incomparable; the skip reason and the status line are both kept.
        let failed = Compare.compare(new: nonMeasured(.failed, correctness: .fail), baseline: record(wall: [1, 1, 1]), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertEqual(failed.reasons, ["correctness failed", oneSided, "no comparable timing kinds (new status failed, baseline status measured)"])
    }

    func testZeroMedianBaselineIsIncomparable() {
        let r = Compare.compare(new: record(wall: [1, 1, 1]), baseline: record(wall: [0, 0, 0]), newKey: key, baselineKey: key)
        XCTAssertEqual(r.status, .incomparable); XCTAssertTrue(r.deltas.isEmpty)
        XCTAssertEqual(r.reasons, ["timing.wall not compared: baseline median 0.0 is not a positive finite number"])
    }

    /// Usable medians can still give a non-finite delta or boundary: (1e308 − 1e-300) / 1e-300 overflows, and so does 2 × rsd for
    /// rsd = 1e308. A NaN median cannot come from a JSON file but can be built in memory. When such a kind is the only one, the
    /// case is incomparable, never an alert, and the JSON rendering decodes (it used to trap on `inf`).
    func testNonFiniteDeltaOrBoundaryIsIncomparable() throws {
        func withWall(median: Double? = nil, rsd: Double? = nil) -> CaseRecord {
            var edited = record(wall: [100, 100, 100])
            if let median { edited.timing.wall?.summary.median = median }
            if let rsd { edited.timing.wall?.summary.rsd = rsd }
            return edited
        }
        let overflow = Compare.compare(new: withWall(median: 1e308), baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(overflow.status, .incomparable); XCTAssertTrue(overflow.deltas.isEmpty)
        XCTAssertEqual(overflow.reasons, ["timing.wall not compared: delta inf is not finite (new median 1e+308, baseline median 1e-300)"])

        let nan = Compare.compare(new: withWall(median: .nan), baseline: withWall(), newKey: key, baselineKey: key)
        XCTAssertEqual(nan.status, .incomparable); XCTAssertTrue(nan.deltas.isEmpty)
        let infiniteBase = Compare.compare(new: withWall(), baseline: withWall(median: .infinity), newKey: key, baselineKey: key)
        XCTAssertEqual(infiniteBase.status, .incomparable); XCTAssertTrue(infiniteBase.deltas.isEmpty)
        let boundary = Compare.compare(new: withWall(), baseline: withWall(rsd: 1e308), newKey: key, baselineKey: key)
        XCTAssertEqual(boundary.status, .incomparable); XCTAssertTrue(boundary.deltas.isEmpty)
        XCTAssertEqual(boundary.reasons, ["timing.wall not compared: boundary inf is not finite (baseline rsd 1e+308)"])

        // A correctness failure still wins, and keeps the comparison reason.
        var failing = withWall(median: 1e308); failing.correctness?.status = .fail
        let failed = Compare.compare(new: failing, baseline: withWall(median: 1e-300), newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed)
        XCTAssertTrue(failed.reasons.contains { $0.contains("not finite") }, "\(failed.reasons)")

        let all = [overflow, nan, infiniteBase, boundary, failed]
        XCTAssertEqual(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render(all, format: "json").utf8)), all)
        // No delta line is rendered for these results (a reason may still quote an infinite median).
        let text = Compare.render(all, format: "text")
        XCTAssertFalse(text.contains("inf%") || text.contains("nan%") || text.contains("(boundary"), text)
    }

    /// A median that is not a positive finite number cannot anchor a ratio. On the NEW side that used to be finite garbage:
    /// 0 against 100 gave delta −1 and `alert(faster)`. A GPU timing pair can come back 0 from a real run. When such a kind is
    /// the only one, the case is incomparable; both sides use one wording that names the kind, the side and the value.
    func testNonPositiveMedianOnEitherSideIsIncomparable() {
        func withWall(median: Double) -> CaseRecord {
            var edited = record(wall: [100, 100, 100]); edited.timing.wall?.summary.median = median
            return edited
        }
        let good = record(wall: [100, 100, 100])
        for value in [0.0, -5.0, -Double.infinity, Double.nan] {
            let newSide = Compare.compare(new: withWall(median: value), baseline: good, newKey: key, baselineKey: key)
            XCTAssertEqual(newSide.status, .incomparable, "new median \(value)"); XCTAssertTrue(newSide.deltas.isEmpty, "new median \(value)")
            XCTAssertEqual(newSide.reasons, ["timing.wall not compared: new median \(value) is not a positive finite number"])
            XCTAssertEqual(Compare.exitCode([newSide], strictPerf: true), 0)
            let baseSide = Compare.compare(new: good, baseline: withWall(median: value), newKey: key, baselineKey: key)
            XCTAssertEqual(baseSide.status, .incomparable, "baseline median \(value)")
            XCTAssertTrue(baseSide.deltas.isEmpty, "baseline median \(value)")
            XCTAssertEqual(baseSide.reasons, ["timing.wall not compared: baseline median \(value) is not a positive finite number"])
        }
        // A measured run whose timings are all zero takes the same path (no hand-edited summary needed).
        let measuredZero = Compare.compare(new: record(wall: [0, 0, 0]), baseline: good, newKey: key, baselineKey: key)
        XCTAssertEqual(measuredZero.status, .incomparable)
        XCTAssertEqual(measuredZero.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number"])
        // Both sides unusable: one reason per side, new first.
        let both = Compare.compare(new: withWall(median: 0), baseline: withWall(median: -1), newKey: key, baselineKey: key)
        XCTAssertEqual(both.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number",
                                      "timing.wall not compared: baseline median -1.0 is not a positive finite number"])
    }

    /// An unusable kind must not mask the others: a broken GPU timer (commandBuffer median 0) beside a genuine wall regression
    /// is still `alert(slower)`, and `--strict-perf` still fails on it. The unusable kind contributes no delta and no verdict,
    /// only a reason; the case is incomparable only when no kind is left. The same holds for all three unusable conditions and
    /// for either side, and the reason is rendered in every format whatever the status.
    func testUnusableKindIsSkippedWithAReasonAndTheRestDecide() throws {
        /// `wall` from the given samples; `commandBuffer` from `gpu`, or with its median or rsd overwritten.
        func pair(wall: [Double], gpu: [Double] = [100, 100, 100], gpuMedian: Double? = nil, gpuRSD: Double? = nil) -> CaseRecord {
            var edited = record(wall: wall)
            edited.timing.commandBuffer = record(wall: gpu).timing.wall
            if let gpuMedian { edited.timing.commandBuffer?.summary.median = gpuMedian }
            if let gpuRSD { edited.timing.commandBuffer?.summary.rsd = gpuRSD }
            return edited
        }
        let base = pair(wall: [100, 100, 100])
        let newZero = "timing.commandBuffer not compared: new median 0.0 is not a positive finite number"
        let baseZero = "timing.commandBuffer not compared: baseline median 0.0 is not a positive finite number"
        func expect(_ result: CompareResult, _ status: CompareStatus, reasons: [String], strictExit: Int32, _ what: String) {
            XCTAssertEqual(result.status, status, what); XCTAssertEqual(result.reasons, reasons, what)
            XCTAssertEqual(result.deltas.map(\.kind), status == .incomparable ? [] : [.wall], what)
            XCTAssertEqual(Compare.exitCode([result], strictPerf: true), strictExit, what)
            XCTAssertEqual(Compare.exitCode([result], strictPerf: false), 0, what)
        }

        // wall regresses beyond the boundary; commandBuffer is unusable on the new side, then on the baseline side.
        let slowerNewSide = Compare.compare(new: pair(wall: [200, 200, 200], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(slowerNewSide, .alertSlower, reasons: [newZero], strictExit: 1, "regression, unusable on the new side")
        let slowerBaseSide = Compare.compare(new: pair(wall: [200, 200, 200]), baseline: pair(wall: [100, 100, 100], gpu: [0, 0, 0]),
                                             newKey: key, baselineKey: key)
        expect(slowerBaseSide, .alertSlower, reasons: [baseZero], strictExit: 1, "regression, unusable on the baseline side")
        // the usable kind is within the boundary: ok, and the reason is still recorded.
        let okWithReason = Compare.compare(new: pair(wall: [101, 101, 101], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(okWithReason, .ok, reasons: [newZero], strictExit: 0, "within the boundary")
        // every kind unusable: incomparable, and the per-kind reasons are the explanation.
        let none = Compare.compare(new: pair(wall: [0, 0, 0], gpu: [0, 0, 0]), baseline: base, newKey: key, baselineKey: key)
        expect(none, .incomparable, reasons: ["timing.wall not compared: new median 0.0 is not a positive finite number", newZero],
               strictExit: 0, "no kind left")

        // The overflow pair and the boundary overflow beside a healthy kind: judged on the healthy kind.
        let overflowReason = "timing.commandBuffer not compared: delta inf is not finite (new median 1e+308, baseline median 1e-300)"
        let tinyBase = pair(wall: [100, 100, 100], gpuMedian: 1e-300)
        let overflowSlower = Compare.compare(new: pair(wall: [200, 200, 200], gpuMedian: 1e308), baseline: tinyBase, newKey: key, baselineKey: key)
        expect(overflowSlower, .alertSlower, reasons: [overflowReason], strictExit: 1, "overflow beside a regression")
        let overflowOK = Compare.compare(new: pair(wall: [101, 101, 101], gpuMedian: 1e308), baseline: tinyBase, newKey: key, baselineKey: key)
        expect(overflowOK, .ok, reasons: [overflowReason], strictExit: 0, "overflow beside a healthy kind")
        let wideBase = pair(wall: [100, 100, 100], gpuRSD: 1e308)
        let boundaryFaster = Compare.compare(new: pair(wall: [50, 50, 50]), baseline: wideBase, newKey: key, baselineKey: key)
        expect(boundaryFaster, .alertFaster, reasons: ["timing.commandBuffer not compared: boundary inf is not finite (baseline rsd 1e+308)"],
               strictExit: 1, "boundary overflow beside an improvement")

        // The other direction: wall unusable, commandBuffer decides.
        var wallZero = pair(wall: [0, 0, 0], gpu: [200, 200, 200])
        let gpuDecides = Compare.compare(new: wallZero, baseline: base, newKey: key, baselineKey: key)
        XCTAssertEqual(gpuDecides.status, .alertSlower); XCTAssertEqual(gpuDecides.deltas.map(\.kind), [.commandBuffer])
        XCTAssertEqual(gpuDecides.reasons, ["timing.wall not compared: new median 0.0 is not a positive finite number"])
        // A correctness failure still wins; the comparable kind's delta and the skip reason are both kept.
        wallZero.correctness?.status = .fail
        let failed = Compare.compare(new: wallZero, baseline: base, newKey: key, baselineKey: key)
        XCTAssertEqual(failed.status, .correctnessFailed); XCTAssertEqual(failed.deltas.map(\.kind), [.commandBuffer])
        XCTAssertEqual(failed.reasons, ["correctness failed", "timing.wall not compared: new median 0.0 is not a positive finite number"])
        // A kind present on one side only is the fourth skip condition: `Runner` omits a declared GPU kind whose samples came
        // back in the wrong shape, so a broken GPU timer can appear as an absent series, not only as a zero median.
        let oneSided = Compare.compare(new: pair(wall: [200, 200, 200], gpu: [0, 0, 0]), baseline: record(wall: [100, 100, 100]),
                                       newKey: key, baselineKey: key)
        XCTAssertEqual(oneSided.status, .alertSlower); XCTAssertEqual(oneSided.deltas.map(\.kind), [.wall])
        XCTAssertEqual(oneSided.reasons, ["timing.commandBuffer not compared: present on the new side only"])
        let otherSide = Compare.compare(new: record(wall: [200, 200, 200]), baseline: pair(wall: [100, 100, 100]),
                                        newKey: key, baselineKey: key)
        XCTAssertEqual(otherSide.status, .alertSlower); XCTAssertEqual(otherSide.deltas.map(\.kind), [.wall])
        XCTAssertEqual(otherSide.reasons, ["timing.commandBuffer not compared: present on the baseline side only"])
        // Absent on both sides is nothing to compare, not a drop: no reason.
        let neither = Compare.compare(new: record(wall: [100, 100, 100]), baseline: record(wall: [100, 100, 100]),
                                      newKey: key, baselineKey: key)
        XCTAssertEqual(neither.status, .ok); XCTAssertEqual(neither.reasons, [])

        // The reason is visible in every format when the status is ok or an alert, not only when it is incomparable.
        for result in [okWithReason, slowerNewSide] {
            let text = Compare.render([result], format: "text")
            XCTAssertTrue(text.contains("    wall "), text); XCTAssertTrue(text.contains("    reason: \(newZero)\n"), text)
            let markdown = Compare.render([result], format: "markdown")
            XCTAssertTrue(markdown.contains("| \(result.id) | \(result.status.rawValue) | wall |"), markdown)
            XCTAssertTrue(markdown.contains("\n\n- \(result.id): \(newZero)\n"), markdown)
            let decoded = try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render([result], format: "json").utf8))
            XCTAssertEqual(decoded, [result]); XCTAssertEqual(decoded.first?.reasons, [newZero])
        }
        let everything = [slowerNewSide, slowerBaseSide, okWithReason, none, overflowSlower, overflowOK, boundaryFaster, gpuDecides, failed]
        XCTAssertEqual(try JSONDecoder().decode([CompareResult].self, from: Data(Compare.render(everything, format: "json").utf8)), everything)
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
