import XCTest

/// Ratchet against unreferenced public types (spec §10). A public type declared in Sources/VectorAccelerate that is named
/// nowhere outside its own file (Sources + Tests) fails unless listed in OrphanAllowlist.txt with a justification, and an
/// allow-listed name that is no longer an orphan (named outside its file now, or no longer declared) fails too, so the list
/// cannot go stale. `testAllowlistParser` pins the file format; `testNoNewOrphanPublicTypes` runs both checks on the tree.
final class OrphanPublicTypeTests: XCTestCase {
    struct Malformed: Error, CustomStringConvertible { var description: String }

    /// One entry per non-comment line, `Name  # justification`: the text before the first `#` is the name, the text after it
    /// the justification, and both must be non-empty. A line whose first non-blank character is `#` is a comment, so
    /// commenting an entry out removes it and re-arms the ratchet for that type; blank lines are ignored. Anything else —
    /// a bare name, an empty justification, a non-type name, a name listed twice — throws `Malformed` naming the line.
    static func parseAllowlist(_ text: String) throws -> [String: String] {
        var entries: [String: String] = [:]
        for (offset, rawLine) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let name = line.prefix { $0 != "#" }.trimmingCharacters(in: .whitespaces)
            let justification = line.drop { $0 != "#" }.dropFirst().trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty, !justification.isEmpty else {
                throw Malformed(description: "OrphanAllowlist.txt line \(offset + 1): expected `Name  # justification`, got `\(line)`")
            }
            guard name.range(of: #"^[A-Z][A-Za-z0-9_]*$"#, options: .regularExpression) != nil else {
                throw Malformed(description: "OrphanAllowlist.txt line \(offset + 1): `\(name)` is not a type name")
            }
            guard entries[name] == nil else {
                throw Malformed(description: "OrphanAllowlist.txt line \(offset + 1): `\(name)` is listed twice")
            }
            entries[name] = justification
        }
        return entries
    }

    /// Pins the file format: comments and blank lines add nothing (so commenting an entry out removes it), a bare name, an
    /// empty justification, a non-type name and a duplicate are rejected naming the line.
    func testAllowlistParser() throws {
        XCTAssertEqual(try Self.parseAllowlist("# Foo"), [:])
        XCTAssertEqual(try Self.parseAllowlist("# Foo  # commented-out entry"), [:])
        XCTAssertEqual(try Self.parseAllowlist("   # indented comment\n\n"), [:])
        XCTAssertEqual(try Self.parseAllowlist("Foo  # why"), ["Foo": "why"])
        XCTAssertEqual(try Self.parseAllowlist("# header\nFoo  # why\n\nBar#because\n"), ["Foo": "why", "Bar": "because"])
        for (malformed, fragment) in [("Foo", "line 1"), ("Foo  #   ", "line 1"), ("# ok\nFoo  # why\nBar", "line 3"),
                                      ("foo  # lower-case", "not a type name"), ("Foo  # a\nFoo  # b", "listed twice")] {
            XCTAssertThrowsError(try Self.parseAllowlist(malformed), malformed) { error in
                XCTAssertTrue("\(error)".contains(fragment), "\(malformed): \(error)")
            }
        }
    }

    func testNoNewOrphanPublicTypes() throws {
        let declPattern = #"^\s*(?:@[\w()., ]+\s+)*public\s+(?:final\s+)?(?:struct|class|enum|actor|protocol|typealias)\s+([A-Z][A-Za-z0-9_]*)"#
        let decl = try NSRegularExpression(pattern: declPattern, options: [.anchorsMatchLines])
        let all = RepoPaths.swiftFiles(under: "Sources/VectorAccelerate") + RepoPaths.swiftFiles(under: "Sources/VectorAccelerateBench")
            + RepoPaths.swiftFiles(under: "Tests/VectorAccelerateTests")
        let texts = try all.map { (url: $0, text: try String(contentsOf: $0, encoding: .utf8)) }
        let allowlistURL = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt")
        let allowed = try Self.parseAllowlist(try String(contentsOf: allowlistURL, encoding: .utf8))

        // Every public type declared under Sources/VectorAccelerate; a declaration is an orphan when no other scanned file
        // names the type. `orphans` maps a name to the declaring files with no outside reference.
        var declared: Set<String> = []
        var orphans: [String: [String]] = [:]
        for (file, text) in texts where file.path.contains("/Sources/VectorAccelerate/") {
            for match in decl.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                guard let nameRange = Range(match.range(at: 1), in: text) else { continue }
                let name = String(text[nameRange])
                declared.insert(name)
                let pattern = try NSRegularExpression(pattern: "\\b\(NSRegularExpression.escapedPattern(for: name))\\b")
                let referenced = texts.contains { other in
                    other.url != file && pattern.firstMatch(in: other.text, range: NSRange(other.text.startIndex..., in: other.text)) != nil
                }
                if !referenced { orphans[name, default: []].append(file.lastPathComponent) }
            }
        }
        XCTAssertFalse(declared.isEmpty, "the scan found no public type under Sources/VectorAccelerate")

        let new = orphans.filter { allowed[$0.key] == nil }.map { "\($0.key) (\($0.value.joined(separator: ", ")))" }.sorted()
        XCTAssertTrue(new.isEmpty, "Unreferenced public types (add to OrphanAllowlist.txt with a justification if they are intended API):\n"
            + new.joined(separator: "\n"))

        let stale = allowed.keys.filter { orphans[$0] == nil }.sorted()
            .map { "\($0) (\(declared.contains($0) ? "named outside its file now" : "no longer declared"))" }
        XCTAssertTrue(stale.isEmpty, "Allow-listed names that are no longer orphans; remove them from OrphanAllowlist.txt:\n"
            + stale.joined(separator: "\n"))
    }
}
