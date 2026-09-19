import Foundation
import VectorTestKit

enum BaselineCommand {
    static let usage = "usage: baseline capture --from <artifact> [--approve] [--allow-dirty]"

    /// A missing subcommand or `--from` is thrown as `BenchError.usage`. A load or validation failure is rethrown with the file
    /// named (`Workspace.loadArtifact`); capture errors (`RunStoreError` refusals included) propagate unchanged. Printing and
    /// exit codes belong to the `catch` arms in `main.swift`.
    static func run(_ args: Arguments) throws -> Int32 {
        try args.validate(options: ["--from"], flags: ["--approve", "--allow-dirty"], operands: 1)
        guard args.subcommand == "capture", let from = args.value("--from") else { throw BenchError.usage(usage) }
        let approve = args.has("--approve")
        let artifact = try Workspace.loadArtifact(from, validating: true)
        let written = try Workspace.store.captureBaseline(from: artifact, approve: approve, allowDirty: args.has("--allow-dirty"))
        for (id, version) in written.sorted(by: { $0.key < $1.key }) {
            print("captured \(id) -> version \(version)\(approve ? " (approved)" : "")")
        }
        // `captureBaseline` stores measured cases only; say which ones it left out instead of leaving a partial capture silent.
        for record in artifact.cases where record.status != .measured {
            Diagnostics.line("skipped \(record.id) (\(record.status.rawValue)): only measured cases are captured"
                             + (approve ? "; its approved pointer, if any, is unchanged" : ""))
        }
        if !approve {
            Diagnostics.line("note: captured without --approve, so no approved pointer moved. `compare` reads the approved pointer by "
                             + "default: capture with --approve, or compare with --against <n> using a version printed above")
        }
        return 0
    }
}
