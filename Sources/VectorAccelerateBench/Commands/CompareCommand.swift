import Foundation
import VectorTestKit

enum CompareCommand {
    static let usage = "usage: compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]"

    /// Which stored baseline every case is compared against.
    enum Against: CustomStringConvertible {
        case approved
        case version(Int)

        /// `approved` or a positive integer; anything else is `nil`, which `Arguments.parsed` reports as a usage error.
        init?(_ raw: String) {
            if raw == "approved" { self = .approved; return }
            guard let n = Int(raw), n >= 1 else { return nil }
            self = .version(n)
        }

        var description: String {
            switch self {
            case .approved: return "approved"
            case .version(let n): return "version \(n)"
            }
        }

        func baseline(in store: RunStore, environmentKey: String, caseID: String) throws -> CaseRecord? {
            switch self {
            case .approved: return try store.approvedBaseline(environmentKey: environmentKey, caseID: caseID)
            case .version(let n): return try store.baseline(environmentKey: environmentKey, caseID: caseID, version: n)
            }
        }
    }

    /// Unverified contract: each raw value must equal a format name `Compare.render` switches on; that function renders a name
    /// it does not know as text instead of failing, so a mismatch here would go unreported.
    enum Format: String { case text, json, markdown }

    static func run(_ args: Arguments) throws -> Int32 {
        // Every option is checked before the first file is read, so a usage error never depends on the artifact or the store.
        try args.validate(options: ["--artifact", "--against", "--format"], flags: ["--strict-perf"], operands: 0)
        guard let path = args.value("--artifact") else { throw BenchError.usage(usage) }
        let against = try args.parsed("--against", expecting: "'approved' or a positive integer version", Against.init) ?? .approved
        let format = try args.parsed("--format", expecting: "text, json or markdown", { Format(rawValue: $0) }) ?? .text
        let strictPerf = args.has("--strict-perf")

        let artifact = try Workspace.loadArtifact(path, validating: false)   // each record is validated by `Compare.compare`
        let store = Workspace.store
        let key = artifact.metadata.environmentKey
        let results: [CompareResult] = try artifact.cases.map { record in
            let baseline: CaseRecord?
            // A corrupt store aborts the comparison; the error says which case and which baseline, which the store's error cannot.
            do { baseline = try against.baseline(in: store, environmentKey: key, caseID: record.id) } catch {
                throw ContextError("case \(record.id): reading its \(against) baseline failed (store \(store.root.path), key \(key))", error)
            }
            return Compare.compare(new: record, baseline: baseline, newKey: key, baselineKey: baseline == nil ? nil : key)
        }
        // One trailing newline on stdout whether or not the rendering already ends with one.
        let report = Compare.render(results, format: format.rawValue)
        print(report, terminator: report.hasSuffix("\n") ? "" : "\n")
        // Warnings go to stderr so stdout stays the rendered report. Spec §7: a not-quiet artifact is still compared, with a warning.
        if artifact.cases.isEmpty { Diagnostics.line("warning: the artifact has no cases, so nothing was compared") }
        if !artifact.metadata.preflight.quiet {
            let reasons = artifact.metadata.preflight.reasons
            Diagnostics.line("warning: artifact was not quiet: \(reasons.isEmpty ? "no reason recorded" : reasons.joined(separator: "; "))")
        }
        return Compare.exitCode(results, strictPerf: strictPerf)
    }
}
