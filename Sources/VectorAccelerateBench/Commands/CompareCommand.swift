import Foundation
import VectorTestKit

enum CompareCommand {
    static let usage = "usage: compare --artifact <path> [--against approved|<n>] [--strict-perf] [--format text|json|markdown]"

    /// Which stored baseline every case is compared against.
    enum Against {
        case approved
        case version(Int)

        /// `approved` or a positive integer; anything else is `nil`, which `Arguments.parsed` reports as a usage error.
        init?(_ raw: String) {
            if raw == "approved" { self = .approved; return }
            guard let n = Int(raw), n >= 1 else { return nil }
            self = .version(n)
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
        guard let path = args.value("--artifact") else { throw BenchError.usage(usage) }
        let against = try args.parsed("--against", expecting: "'approved' or a positive integer version", Against.init) ?? .approved
        let format = try args.parsed("--format", expecting: "text, json or markdown", { Format(rawValue: $0) }) ?? .text
        let strictPerf = args.has("--strict-perf")

        let artifact = try RunStore.load(URL(fileURLWithPath: path))
        let store = RunStore(root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".bench"))
        let key = artifact.metadata.environmentKey
        let results: [CompareResult] = try artifact.cases.map { record in
            let baseline = try against.baseline(in: store, environmentKey: key, caseID: record.id)
            return Compare.compare(new: record, baseline: baseline, newKey: key, baselineKey: baseline == nil ? nil : key)
        }
        // One trailing newline on stdout whether or not the rendering already ends with one.
        let report = Compare.render(results, format: format.rawValue)
        print(report, terminator: report.hasSuffix("\n") ? "" : "\n")
        // Spec §7: a not-quiet artifact is still compared, with a warning. The warning goes to stderr so stdout stays the rendered report.
        if !artifact.metadata.preflight.quiet {
            FileHandle.standardError.write(Data("warning: artifact was not quiet: \(artifact.metadata.preflight.reasons)\n".utf8))
        }
        return Compare.exitCode(results, strictPerf: strictPerf)
    }
}
