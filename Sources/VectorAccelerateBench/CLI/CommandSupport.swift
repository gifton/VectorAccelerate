import Foundation
import VectorTestKit

/// The one place the bench spells its working directory and the `.bench` store under it. `run` writes there and
/// `baseline capture` / `compare` read there; two spellings that drifted apart would show up as "incomparable, exit 0",
/// not as an error.
enum Workspace {
    static var root: URL { URL(fileURLWithPath: FileManager.default.currentDirectoryPath) }
    static var store: RunStore { RunStore(root: root.appendingPathComponent(".bench")) }

    /// A failure names the file, which a bare `DecodingError` or `ArtifactValidationError` does not.
    static func loadArtifact(_ path: String, validating: Bool) throws -> BenchArtifact {
        do {
            let artifact = try RunStore.load(URL(fileURLWithPath: path))
            if validating { try artifact.validate() }
            return artifact
        } catch { throw ContextError("artifact \(path)", error) }
    }
}

/// An underlying error plus where it happened. It is not a `BenchError`, so `main.swift`'s general `catch` arm prints it and
/// exits 1: wrap only errors that would have taken that arm anyway.
struct ContextError: Error, CustomStringConvertible {
    let context: String
    let underlying: any Error
    init(_ context: String, _ underlying: any Error) { self.context = context; self.underlying = underlying }
    var description: String { "\(context): \(underlying)" }
}

/// Warnings, notes and errors go to stderr, so stdout carries only what a command was asked to produce.
enum Diagnostics {
    static func line(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
}
