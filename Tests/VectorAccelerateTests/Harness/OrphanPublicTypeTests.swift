import XCTest

/// Ratchet against unreferenced public types (spec §10). A public type declared in Sources/VectorAccelerate that is named
/// nowhere outside its own file (Sources + Tests) fails unless listed in OrphanAllowlist.txt with a justification.
final class OrphanPublicTypeTests: XCTestCase {
    func testNoNewOrphanPublicTypes() throws {
        let declPattern = #"^\s*(?:@[\w()., ]+\s+)*public\s+(?:final\s+)?(?:struct|class|enum|actor|protocol|typealias)\s+([A-Z][A-Za-z0-9_]*)"#
        let decl = try NSRegularExpression(pattern: declPattern, options: [.anchorsMatchLines])
        let sources = RepoPaths.swiftFiles(under: "Sources/VectorAccelerate")
        let all = sources + RepoPaths.swiftFiles(under: "Sources/VectorAccelerateBench") + RepoPaths.swiftFiles(under: "Tests/VectorAccelerateTests")
        let texts = try all.map { ($0, try String(contentsOf: $0, encoding: .utf8)) }
        let allowlistURL = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/OrphanAllowlist.txt")
        let allowed = Set(((try? String(contentsOf: allowlistURL, encoding: .utf8)) ?? "").split(separator: "\n")
            .map { $0.split(separator: "#")[0].trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })

        var orphans: [String] = []
        for (file, text) in texts where file.path.contains("/Sources/VectorAccelerate/") {
            let range = NSRange(text.startIndex..., in: text)
            for m in decl.matches(in: text, range: range) {
                let name = String(text[Range(m.range(at: 1), in: text)!])
                if allowed.contains(name) { continue }
                let pattern = try NSRegularExpression(pattern: "\\b\(NSRegularExpression.escapedPattern(for: name))\\b")
                let referenced = texts.contains { (f, t) in f != file && pattern.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)) != nil }
                if !referenced { orphans.append("\(name) (\(file.lastPathComponent))") }
            }
        }
        XCTAssertTrue(orphans.isEmpty, "Unreferenced public types (add to OrphanAllowlist.txt with a justification if they are intended API):\n" + orphans.sorted().joined(separator: "\n"))
    }
}
