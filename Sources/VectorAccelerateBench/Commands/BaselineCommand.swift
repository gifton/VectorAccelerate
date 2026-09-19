import Foundation
import VectorTestKit

enum BaselineCommand {
    static let usage = "usage: baseline capture --from <artifact> [--approve] [--allow-dirty]"

    /// Nothing is caught here. A missing subcommand or `--from` is thrown as `BenchError.usage`; load, validation and capture
    /// errors (`RunStoreError` refusals included) propagate unchanged. Printing and exit codes belong to the `catch` arms in `main.swift`.
    static func run(_ args: Arguments) throws -> Int32 {
        guard args.subcommand == "capture", let from = args.value("--from") else { throw BenchError.usage(usage) }
        let approve = args.has("--approve")
        let artifact = try RunStore.load(URL(fileURLWithPath: from))
        try artifact.validate()
        let store = RunStore(root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".bench"))
        let written = try store.captureBaseline(from: artifact, approve: approve, allowDirty: args.has("--allow-dirty"))
        for (id, version) in written.sorted(by: { $0.key < $1.key }) {
            print("captured \(id) -> version \(version)\(approve ? " (approved)" : "")")
        }
        return 0
    }
}
