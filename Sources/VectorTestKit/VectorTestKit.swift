import Foundation

/// Module marker. The artifact schema version lives on `BenchArtifact.schemaVersion`; this
/// constant exists so the kit compiles before any other file is added.
public enum VectorTestKit {
    public static let schemaVersion: Int = 1
}
