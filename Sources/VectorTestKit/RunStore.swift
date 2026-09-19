import Foundation

public enum RunStoreError: Error, Equatable {
    case runExists(String)
    case refusedValidationLayers
    case refusedNotQuiet([String])
    case refusedDirty
    case noMeasuredCases
    case badApprovedPointer(String)
}

/// Append-only artifact and baseline storage (spec §4.11).
/// Runs: `<root>/runs/<environmentKey>/<yyyyMMdd'T'HHmmss'Z'>_<sha7|nogit>[_dirty].json`.
/// Baselines: `<root>/baselines/<environmentKey>/<caseID>/<n>.json` plus an `approved` pointer file.
public struct RunStore: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    private static let stamp: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"; return f
    }()

    public static func encode(_ value: some Encodable) throws -> Data {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]; return try enc.encode(value)
    }

    public static func load(_ url: URL) throws -> BenchArtifact {
        try JSONDecoder().decode(BenchArtifact.self, from: Data(contentsOf: url))
    }

    public func runURL(for artifact: BenchArtifact, at date: Date) -> URL {
        let sha = artifact.metadata.gitSHA.map { String($0.prefix(7)) } ?? "nogit"
        let name = "\(Self.stamp.string(from: date))_\(sha)\(artifact.metadata.gitDirty ? "_dirty" : "").json"
        return root.appendingPathComponent("runs").appendingPathComponent(artifact.metadata.environmentKey).appendingPathComponent(name)
    }

    /// Create-only: an existing path is an error, never overwritten (pinned by `RunStoreTests.testRunPathLayoutAndCreateOnly`).
    @discardableResult
    public func writeRun(_ artifact: BenchArtifact, at date: Date = Date()) throws -> URL {
        let url = runURL(for: artifact, at: date)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: url.path) else { throw RunStoreError.runExists(url.path) }
        try Self.encode(artifact).write(to: url, options: [.withoutOverwriting])
        return url
    }

    private func caseDir(_ key: String, _ id: String) -> URL {
        root.appendingPathComponent("baselines").appendingPathComponent(key).appendingPathComponent(id)
    }

    private func versions(in dir: URL) -> [Int] {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .compactMap { $0.hasSuffix(".json") ? Int($0.dropLast(5)) : nil }.sorted()
    }

    /// Writes every measured case as the next version; never overwrites; moves `approved` only when asked
    /// (pinned by `RunStoreTests.testCaptureAppendsAndApprovePointer`; refusals by `RunStoreTests.testCaptureRefusals`).
    public func captureBaseline(from artifact: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int] {
        if artifact.metadata.validationLayers { throw RunStoreError.refusedValidationLayers }
        if !artifact.metadata.preflight.quiet { throw RunStoreError.refusedNotQuiet(artifact.metadata.preflight.reasons) }
        if artifact.metadata.gitDirty && !allowDirty { throw RunStoreError.refusedDirty }
        let measured = artifact.cases.filter { $0.status == .measured }
        guard !measured.isEmpty else { throw RunStoreError.noMeasuredCases }
        var written: [String: Int] = [:]
        for c in measured {
            let dir = caseDir(artifact.metadata.environmentKey, c.id)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let n = (versions(in: dir).last ?? 0) + 1
            try Self.encode(c).write(to: dir.appendingPathComponent("\(n).json"), options: [.withoutOverwriting])
            if approve { try "\(n)\n".write(to: dir.appendingPathComponent("approved"), atomically: true, encoding: .utf8) }
            written[c.id] = n
        }
        return written
    }

    public func baseline(environmentKey: String, caseID: String, version: Int) throws -> CaseRecord? {
        let url = caseDir(environmentKey, caseID).appendingPathComponent("\(version).json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(CaseRecord.self, from: Data(contentsOf: url))
    }

    public func approvedVersion(environmentKey: String, caseID: String) throws -> Int? {
        let url = caseDir(environmentKey, caseID).appendingPathComponent("approved")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let text = try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let n = Int(text) else { throw RunStoreError.badApprovedPointer(text) }
        return n
    }

    public func approvedBaseline(environmentKey: String, caseID: String) throws -> CaseRecord? {
        guard let n = try approvedVersion(environmentKey: environmentKey, caseID: caseID) else { return nil }
        return try baseline(environmentKey: environmentKey, caseID: caseID, version: n)
    }
}
