import Foundation

/// Outcome of comparing one new case against its baseline. Raw values are the labels the renderers print.
public enum CompareStatus: String, Codable, Sendable, Equatable {
    case ok
    case alertSlower = "alert(slower)"
    case alertFaster = "alert(faster)"
    case incomparable
    case correctnessFailed
    case resourceFailed
    case invalidArtifact
}

/// One timing kind's comparison. `delta` is `(newMedian - baseMedian) / baseMedian`; `boundary` is the symmetric
/// alert threshold; `widened` is true when 2 × baseline RSD exceeded the 2.5% floor (`CompareTests.testBoundaryWidensWithBaselineRSD`).
public struct KindDelta: Codable, Sendable, Equatable {
    public var kind: TimingKind
    public var newMedian: Double
    public var baseMedian: Double
    public var delta: Double
    public var boundary: Double
    public var widened: Bool
    public var alerting: Bool
}

public struct CompareResult: Codable, Sendable, Equatable {
    public var id: String
    public var status: CompareStatus
    public var deltas: [KindDelta]
    public var reasons: [String]
    public init(id: String, status: CompareStatus, deltas: [KindDelta], reasons: [String]) {
        self.id = id; self.status = status; self.deltas = deltas; self.reasons = reasons
    }
}

/// Owner policy (Q5, spec §5.6): alerts in both directions, boundary max(2.5%, 2 × baseline RSD), non-blocking
/// unless strict; correctness, resource, and artifact failures block regardless of baseline availability.
/// Precedence invalidArtifact → correctnessFailed → resourceFailed → incomparable → alert → ok is pinned by
/// `CompareTests.testInvalidArtifact`, `testCorrectnessAndResourceFailuresWinEvenWithoutBaseline`,
/// `testIncomparableIdentityKeyAndMissing`, `testNonMeasuredNewRecordIsIncomparable`, `testZeroMedianBaselineIsIncomparable`,
/// and `testAlertSlowerAndFaster`; blocking rules by `testExitCodes`. A timing kind that cannot be compared is skipped with
/// a reason and the remaining kinds decide (`testUnusableKindIsSkippedWithAReasonAndTheRestDecide`).
public enum Compare {
    public static let floorBoundary = 0.025

    public static func compare(new: CaseRecord, baseline: CaseRecord?, newKey: String, baselineKey: String?) -> CompareResult {
        var reasons: [String] = []
        var status: CompareStatus?

        // 1. invalid
        do { try new.validate() } catch { reasons.append("new record invalid: \(error)"); status = .invalidArtifact }
        if let b = baseline { do { try b.validate() } catch { reasons.append("baseline record invalid: \(error)"); status = status ?? .invalidArtifact } }
        if status == .invalidArtifact { return CompareResult(id: new.id, status: .invalidArtifact, deltas: [], reasons: reasons) }

        // 2. correctness, 3. resources — recorded regardless of baseline
        if new.correctness?.status == .fail {
            let notes = new.correctness?.notes ?? []
            reasons.append("correctness failed" + (notes.isEmpty ? "" : ": " + notes.joined(separator: "; ")))
            status = status ?? .correctnessFailed
        }
        if let r = new.resources {
            if let s = r.staticThreadgroupBytes, let m = r.maxThreadgroupBytes, s > m {
                reasons.append("resource limit exceeded: staticThreadgroupBytes \(s) > \(m)"); status = status ?? .resourceFailed
            }
            for n in r.notes where n.hasPrefix("LIMIT_EXCEEDED:") { reasons.append("resource limit exceeded: \(n)"); status = status ?? .resourceFailed }
        }

        // 4. comparability
        guard let base = baseline else {
            reasons.append("no baseline for \(new.id) under \(newKey)")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        guard let baseKey = baselineKey else {
            reasons.append("baseline environment key missing")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        var comparable = true
        if new.identity != base.identity { reasons.append("identity differs from baseline"); comparable = false }
        if newKey != baseKey { reasons.append("environment key differs: \(newKey) vs \(baseKey)"); comparable = false }
        if !comparable { return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons) }

        // 5. deltas
        var deltas: [KindDelta] = []
        var alert: CompareStatus?
        var skipped: [String] = []
        for kind in TimingKind.allCases {
            // A kind that cannot be compared is skipped with a reason and the remaining kinds decide the case, so a broken GPU
            // timer cannot hide a wall regression (`CompareTests.testUnusableKindIsSkippedWithAReasonAndTheRestDecide`). Four
            // conditions, one wording — `timing.<kind> not compared: …` — on either side.
            let notCompared = "timing.\(kind.rawValue) not compared"
            // 0. Present on one side only. `Runner` omits a declared GPU kind whose samples came back in the wrong shape, so a
            // broken GPU timer can appear as an absent series, not only as a zero median. Absent on both sides is nothing to
            // compare, not a drop.
            let sides = (new: new.timing[kind], base: base.timing[kind])
            if sides.new == nil, sides.base == nil { continue }
            guard let n = sides.new, let b = sides.base else {
                skipped.append("\(notCompared): present on the \(sides.new == nil ? "baseline" : "new") side only")
                continue
            }
            // 1–2. `validate()` does not bound the summary fields. A median that is not a positive finite number cannot anchor a
            // ratio: as the baseline it divides by zero or flips the sign, and as the new value it is finite garbage that would
            // read as an alert (0 against 100 is delta −1).
            let medians = [(side: "new", value: n.summary.median), (side: "baseline", value: b.summary.median)]
            let rejected = medians.filter { !($0.value > 0 && $0.value.isFinite) }
            guard rejected.isEmpty else {
                skipped += rejected.map { "\(notCompared): \($0.side) median \($0.value) is not a positive finite number" }
                continue
            }
            let delta = (n.summary.median - b.summary.median) / b.summary.median
            let boundary = max(floorBoundary, 2 * b.summary.rsd)
            // 3. Usable medians can still overflow here (a huge median over a tiny one; 2 × a huge rsd).
            guard delta.isFinite, boundary.isFinite else {
                if !delta.isFinite {
                    skipped.append("\(notCompared): delta \(delta) is not finite "
                                   + "(new median \(n.summary.median), baseline median \(b.summary.median))")
                }
                if !boundary.isFinite { skipped.append("\(notCompared): boundary \(boundary) is not finite (baseline rsd \(b.summary.rsd))") }
                continue
            }
            let alerting = delta > boundary || delta < -boundary
            deltas.append(KindDelta(kind: kind, newMedian: n.summary.median, baseMedian: b.summary.median, delta: delta,
                                    boundary: boundary, widened: boundary > floorBoundary, alerting: alerting))
            // With several timing kinds the status reflects the first alerting kind in `TimingKind.allCases` order; each kind's own `alerting` is rendered.
            if alerting, alert == nil { alert = delta > 0 ? .alertSlower : .alertFaster }
        }
        // Skip reasons stay on the result whatever its status; every format of `render` prints a result's reasons. A `KindDelta`
        // is appended only after both guards above, which check all four of its Double fields, so every one this function
        // returns holds finite values (`CompareTests.testNonFiniteDeltaOrBoundaryIsIncomparable`).
        reasons += skipped
        // Nothing left to judge by (non-measured record, no timing kind present on both sides, or every such kind skipped) is
        // incomparable, never ok (`CompareTests.testNonMeasuredNewRecordIsIncomparable`, `testZeroMedianBaselineIsIncomparable`).
        // The skip reasons, when there are any, are the explanation.
        if deltas.isEmpty {
            if skipped.isEmpty {
                reasons.append("no comparable timing kinds (new status \(new.status.rawValue), baseline status \(base.status.rawValue))")
            }
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        return CompareResult(id: new.id, status: status ?? alert ?? .ok, deltas: deltas, reasons: reasons)
    }

    public static func exitCode(_ results: [CompareResult], strictPerf: Bool) -> Int32 {
        let hard: Set<CompareStatus> = [.invalidArtifact, .correctnessFailed, .resourceFailed]
        if results.contains(where: { hard.contains($0.status) }) { return 1 }
        if strictPerf, results.contains(where: { $0.status == .alertSlower || $0.status == .alertFaster }) { return 1 }
        return 0
    }

    /// Right-pads to `width` columns and never truncates, unlike `String.padding(toLength:)`
    /// (`CompareTests.testTextRenderDoesNotTruncateLongIDs`).
    private static func pad(_ s: String, to width: Int) -> String { s + String(repeating: " ", count: max(0, width - s.count)) }

    /// A ratio rendered as a percentage with two decimals, signed on request (`+4.00%`, `2.50%`).
    private static func pct(_ value: Double, signed: Bool) -> String { String(format: signed ? "%+.2f%%" : "%.2f%%", value * 100) }

    public static func render(_ results: [CompareResult], format: String) -> String {
        switch format {
        case "json":
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            // try!, as a contract: `JSONEncoder` throws here only for a non-finite Double. Results produced by `Compare.compare`
            // carry finite values only (`CompareTests.testNonFiniteDeltaOrBoundaryIsIncomparable` for skipped kinds;
            // `testUnusableKindIsSkippedWithAReasonAndTheRestDecide` round-trips results that carry deltas through this encoder).
            // A hand-built `CompareResult` holding NaN or an infinity is the caller's responsibility and traps here.
            return String(decoding: try! enc.encode(results), as: UTF8.self)
        case "markdown":
            var out = "| case | status | kind | new median (ns) | base median (ns) | delta | boundary |\n|---|---|---|---|---|---|---|\n"
            for r in results {
                if r.deltas.isEmpty { out += "| \(r.id) | \(r.status.rawValue) | — | — | — | — | — |\n" }
                for d in r.deltas {
                    out += "| \(r.id) | \(r.status.rawValue) | \(d.kind.rawValue) | \(String(format: "%.0f", d.newMedian)) | \(String(format: "%.0f", d.baseMedian)) "
                        + "| \(pct(d.delta, signed: true)) | \(pct(d.boundary, signed: false))\(d.widened ? " (widened)" : "") |\n"
                }
            }
            // Reasons follow the whole table, after one blank line, so rows stay contiguous across results
            // (`CompareTests.testMarkdownWithMultipleResultsKeepsTableContiguous`).
            let bullets = results.flatMap { r in r.reasons.map { "- \(r.id): \($0)\n" } }
            if !bullets.isEmpty { out += "\n" + bullets.joined() }
            return out
        default:
            var out = ""
            for r in results {
                out += "\(pad(r.id, to: 44)) \(r.status.rawValue)\n"
                for d in r.deltas {
                    out += "    \(pad(d.kind.rawValue, to: 14)) \(pct(d.delta, signed: true)) "
                        + "(boundary \(pct(d.boundary, signed: false))\(d.widened ? ", widened" : ""))\(d.alerting ? "  <-- alert" : "")\n"
                }
                for reason in r.reasons { out += "    reason: \(reason)\n" }
            }
            return out
        }
    }
}
