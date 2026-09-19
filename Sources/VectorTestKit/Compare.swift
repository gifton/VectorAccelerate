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
/// `testIncomparableIdentityKeyAndMissing`, and `testAlertSlowerAndFaster`; blocking rules by `testExitCodes`.
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
        if new.correctness?.status == .fail { reasons.append("correctness failed: \(new.correctness?.notes.joined(separator: "; ") ?? "")"); status = status ?? .correctnessFailed }
        if let r = new.resources {
            if let s = r.staticThreadgroupBytes, let m = r.maxThreadgroupBytes, s > m { reasons.append("resource limit exceeded: staticThreadgroupBytes \(s) > \(m)"); status = status ?? .resourceFailed }
            for n in r.notes where n.hasPrefix("LIMIT_EXCEEDED:") { reasons.append("resource limit exceeded: \(n)"); status = status ?? .resourceFailed }
        }

        // 4. comparability
        var comparable = true
        guard let base = baseline, let baseKey = baselineKey else {
            reasons.append("no baseline for \(new.id) under \(newKey)")
            return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons)
        }
        if new.identity != base.identity { reasons.append("identity differs from baseline"); comparable = false }
        if newKey != baseKey { reasons.append("environment key differs: \(newKey) vs \(baseKey)"); comparable = false }
        if !comparable { return CompareResult(id: new.id, status: status ?? .incomparable, deltas: [], reasons: reasons) }

        // 5. deltas
        var deltas: [KindDelta] = []
        var alert: CompareStatus?
        for kind in TimingKind.allCases {
            guard let n = new.timing[kind], let b = base.timing[kind], b.summary.median > 0 else { continue }
            let delta = (n.summary.median - b.summary.median) / b.summary.median
            let boundary = max(floorBoundary, 2 * b.summary.rsd)
            let alerting = delta > boundary || delta < -boundary
            deltas.append(KindDelta(kind: kind, newMedian: n.summary.median, baseMedian: b.summary.median, delta: delta,
                                    boundary: boundary, widened: boundary > floorBoundary, alerting: alerting))
            if alerting, alert == nil { alert = delta > 0 ? .alertSlower : .alertFaster }
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

    public static func render(_ results: [CompareResult], format: String) -> String {
        switch format {
        case "json":
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            return String(decoding: (try? enc.encode(results)) ?? Data("[]".utf8), as: UTF8.self)
        case "markdown":
            var out = "| case | status | kind | new median (ns) | base median (ns) | delta | boundary |\n|---|---|---|---|---|---|---|\n"
            for r in results {
                if r.deltas.isEmpty { out += "| \(r.id) | \(r.status.rawValue) | — | — | — | — | — |\n" }
                for d in r.deltas {
                    out += "| \(r.id) | \(r.status.rawValue) | \(d.kind.rawValue) | \(String(format: "%.0f", d.newMedian)) | \(String(format: "%.0f", d.baseMedian)) | \(String(format: "%+.2f%%", d.delta * 100)) | \(String(format: "%.2f%%", d.boundary * 100))\(d.widened ? " (widened)" : "") |\n"
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
                    out += "    \(pad(d.kind.rawValue, to: 14)) \(String(format: "%+.2f%%", d.delta * 100)) (boundary \(String(format: "%.2f%%", d.boundary * 100))\(d.widened ? ", widened" : ""))\(d.alerting ? "  <-- alert" : "")\n"
                }
                for reason in r.reasons { out += "    reason: \(reason)\n" }
            }
            return out
        }
    }
}
