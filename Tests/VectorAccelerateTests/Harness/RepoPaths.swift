import Foundation

/// Locates the package root from this file's path so mechanical tests can scan sources.
/// Layout assumed: <root>/Tests/VectorAccelerateTests/Harness/RepoPaths.swift
enum RepoPaths {
    static let root: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url = url.deletingLastPathComponent() }  // RepoPaths.swift, Harness, VectorAccelerateTests, Tests
        return url
    }()

    static func swiftFiles(under relative: String) -> [URL] {
        let base = root.appendingPathComponent(relative)
        guard let e = FileManager.default.enumerator(at: base, includingPropertiesForKeys: nil) else { return [] }
        return e.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }.sorted { $0.path < $1.path }
    }
}
