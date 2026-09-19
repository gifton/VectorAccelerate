import Foundation

public enum RunStoreError: Error, Equatable {
    case runExists(String)
    case refusedValidationLayers
    case refusedNotQuiet([String])
    case refusedDirty
    case noMeasuredCases
    case badApprovedPointer(String)
    case unsafePathComponent(String)
}

/// Append-only artifact and baseline storage (spec §4.11).
/// Runs: `<root>/runs/<environmentKey>/<yyyyMMdd'T'HHmmss'Z'>_<sha7|nogit>[_dirty].json`.
/// Baselines: `<root>/baselines/<environmentKey>/<caseID>/<n>.json` plus an `approved` pointer file.
public struct RunStore: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    private static let stamp: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = .gmt
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"; return f
    }()

    /// Artifact strings become single path components under `root`. Empty, `.`, `..`, and anything containing `/` are
    /// refused before any filesystem access (pinned by `RunStoreTests.testUnsafePathComponentsAreRefusedBeforeAnyWrite`).
    private static func checkComponent(_ value: String, what: String) throws {
        if value.isEmpty || value.contains("/") || value == "." || value == ".." {
            throw RunStoreError.unsafePathComponent("\(what): \(value)")
        }
    }

    private static func shaTag(_ metadata: Metadata) -> String {
        metadata.gitSHA.map { String($0.prefix(7)) } ?? "nogit"
    }

    public static func encode(_ value: some Encodable) throws -> Data {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]; return try enc.encode(value)
    }

    public static func load(_ url: URL) throws -> BenchArtifact {
        try JSONDecoder().decode(BenchArtifact.self, from: Data(contentsOf: url))
    }

    /// Pure path builder, never throws; `writeRun` validates the components before touching the filesystem.
    public func runURL(for artifact: BenchArtifact, at date: Date) -> URL {
        let name = "\(Self.stamp.string(from: date))_\(Self.shaTag(artifact.metadata))\(artifact.metadata.gitDirty ? "_dirty" : "").json"
        return root.appendingPathComponent("runs").appendingPathComponent(artifact.metadata.environmentKey).appendingPathComponent(name)
    }

    /// Create-only: an existing path is an error, never overwritten (pinned by `RunStoreTests.testRunPathLayoutAndCreateOnly`).
    @discardableResult
    public func writeRun(_ artifact: BenchArtifact, at date: Date = Date()) throws -> URL {
        try Self.checkComponent(artifact.metadata.environmentKey, what: "environmentKey")
        try Self.checkComponent(Self.shaTag(artifact.metadata), what: "gitSHA")
        let url = runURL(for: artifact, at: date)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: url.path) else { throw RunStoreError.runExists(url.path) }
        try Self.encode(artifact).write(to: url, options: [.withoutOverwriting])
        return url
    }

    private func caseDir(_ key: String, _ id: String) throws -> URL {
        try Self.checkComponent(key, what: "environmentKey")
        try Self.checkComponent(id, what: "caseID")
        return root.appendingPathComponent("baselines").appendingPathComponent(key).appendingPathComponent(id)
    }

    /// Existing version numbers, ascending. A listing error propagates (no `try?`), so a failed listing is never
    /// mistaken for an empty directory; non-positive names are not versions.
    private func versions(in dir: URL) throws -> [Int] {
        try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .compactMap { $0.hasSuffix(".json") ? Int($0.dropLast(5)) : nil }.filter { $0 > 0 }.sorted()
    }

    /// Writes every measured case as the next version; never overwrites (`.withoutOverwriting`); moves `approved` only
    /// when asked (pinned by `RunStoreTests.testCaptureAppendsAndApprovePointer`; refusals by `RunStoreTests.testCaptureRefusals`).
    /// Refusals, path checks, and encoding all complete before the first filesystem write.
    public func captureBaseline(from artifact: BenchArtifact, approve: Bool, allowDirty: Bool) throws -> [String: Int] {
        if artifact.metadata.validationLayers { throw RunStoreError.refusedValidationLayers }
        if !artifact.metadata.preflight.quiet { throw RunStoreError.refusedNotQuiet(artifact.metadata.preflight.reasons) }
        if artifact.metadata.gitDirty && !allowDirty { throw RunStoreError.refusedDirty }
        let measured = artifact.cases.filter { $0.status == .measured }
        guard !measured.isEmpty else { throw RunStoreError.noMeasuredCases }
        let prepared: [(id: String, dir: URL, data: Data)] = try measured.map {
            (id: $0.id, dir: try caseDir(artifact.metadata.environmentKey, $0.id), data: try Self.encode($0))
        }
        var written: [String: Int] = [:]
        for p in prepared {
            try FileManager.default.createDirectory(at: p.dir, withIntermediateDirectories: true)
            let n = try (versions(in: p.dir).last ?? 0) + 1
            try p.data.write(to: p.dir.appendingPathComponent("\(n).json"), options: [.withoutOverwriting])
            if approve { try "\(n)\n".write(to: p.dir.appendingPathComponent("approved"), atomically: true, encoding: .utf8) }
            written[p.id] = n
        }
        return written
    }

    public func baseline(environmentKey: String, caseID: String, version: Int) throws -> CaseRecord? {
        let url = try caseDir(environmentKey, caseID).appendingPathComponent("\(version).json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(CaseRecord.self, from: Data(contentsOf: url))
    }

    /// The pointer must hold a positive integer; anything else is `badApprovedPointer` (pinned by `RunStoreTests.testBadApprovedPointerIsSurfaced`).
    public func approvedVersion(environmentKey: String, caseID: String) throws -> Int? {
        let url = try caseDir(environmentKey, caseID).appendingPathComponent("approved")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let text = try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let n = Int(text), n >= 1 else { throw RunStoreError.badApprovedPointer(text) }
        return n
    }

    /// A pointer whose version file is missing is an error, not "no pointer" (pinned by `RunStoreTests.testBadApprovedPointerIsSurfaced`).
    public func approvedBaseline(environmentKey: String, caseID: String) throws -> CaseRecord? {
        guard let n = try approvedVersion(environmentKey: environmentKey, caseID: caseID) else { return nil }
        guard let record = try baseline(environmentKey: environmentKey, caseID: caseID, version: n) else {
            throw RunStoreError.badApprovedPointer("version \(n) missing")
        }
        return record
    }
}
