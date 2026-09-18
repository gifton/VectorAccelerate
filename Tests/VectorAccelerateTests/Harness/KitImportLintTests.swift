import XCTest

/// Genericism rule (spec §4.1): the kit imports only Foundation, Darwin, CryptoKit and never uses @testable.
///
/// The scan is line-anchored and covers attribute-prefixed imports (`@testable`, `@preconcurrency`, `@_exported`,
/// `@_implementationOnly`, `@_spi(...)`, in any order and count, including an attribute on the line before `import`)
/// and qualified imports (`import struct Foundation.Data` reports module `Foundation`). A commented-out line such as
/// `// import Metal` does not match because `^\s*` must be followed directly by an attribute or the `import` keyword.
/// `testClassifierFixtures` pins each of these behaviours.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    /// One import statement as seen by the lint.
    struct ImportStatement: Equatable {
        var module: String
        var testable: Bool
    }

    // Group 1: zero or more attributes (each with optional parenthesized arguments) preceding `import`.
    // Group 2: the top-level module name, after an optional import-kind keyword and before any `.` qualifier.
    private static let importLine = try! NSRegularExpression(
        pattern: #"^\s*((?:@\w+(?:\([^)]*\))?\s+)*)import\s+(?:(?:typealias|struct|class|enum|protocol|actor|let|var|func)\s+)?([A-Za-z0-9_]+)"#,
        options: [.anchorsMatchLines])

    /// Every import statement in `text`, in source order. Non-import lines (code, comments) contribute nothing.
    static func imports(in text: String) -> [ImportStatement] {
        let range = NSRange(text.startIndex..., in: text)
        return importLine.matches(in: text, range: range).map { m in
            let attributes = m.range(at: 1).location == NSNotFound ? "" : String(text[Range(m.range(at: 1), in: text)!])
            let module = String(text[Range(m.range(at: 2), in: text)!])
            return ImportStatement(module: module,
                                   testable: attributes.split(whereSeparator: \.isWhitespace).contains("@testable"))
        }
    }

    /// Classifies a single source line; nil when it is not an import statement.
    static func classify(_ line: String) -> ImportStatement? { imports(in: line).first }

    func testKitImportsAreWithinAllowlist() throws {
        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for statement in Self.imports(in: text) {
                if statement.testable { violations.append("\(file.lastPathComponent): @testable import is forbidden") }
                if !Self.allowed.contains(statement.module) { violations.append("\(file.lastPathComponent): import \(statement.module) is outside the allowlist \(Self.allowed.sorted())") }
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }

    /// Pins the classifier on fixture lines so a pattern change that reopens a bypass fails here, not in review.
    func testClassifierFixtures() {
        typealias S = ImportStatement
        let cases: [(line: String, expected: S?)] = [
            ("import Foundation",                      S(module: "Foundation", testable: false)),
            ("    import Darwin",                      S(module: "Darwin", testable: false)),
            ("\timport CryptoKit",                     S(module: "CryptoKit", testable: false)),
            ("// import Metal",                        nil),
            ("/* import Metal */",                     nil),
            ("@testable import Foundation",            S(module: "Foundation", testable: true)),
            ("@preconcurrency import XCTest",          S(module: "XCTest", testable: false)),
            ("@_exported import Metal",                S(module: "Metal", testable: false)),
            ("@_implementationOnly import Metal",      S(module: "Metal", testable: false)),
            ("@_spi(Internal) import Metal",           S(module: "Metal", testable: false)),
            ("@preconcurrency @testable import Metal", S(module: "Metal", testable: true)),
            ("@testable @preconcurrency import Metal", S(module: "Metal", testable: true)),
            ("import struct Foundation.Data",          S(module: "Foundation", testable: false)),
            ("import func Darwin.sqrt",                S(module: "Darwin", testable: false)),
            ("import Foundation.NSData",               S(module: "Foundation", testable: false)),
            ("importantValue = 1",                     nil),
        ]
        for (line, expected) in cases {
            XCTAssertEqual(Self.classify(line), expected, "bare line: \(line)")
            // Mid-file placement pins the `.anchorsMatchLines` dependency: `^` must match at every line start.
            XCTAssertEqual(Self.imports(in: "// header\n\(line)\nlet x = 0\n"), expected.map { [$0] } ?? [], "embedded: \(line)")
        }
        // An attribute on its own line still belongs to the import that follows it.
        XCTAssertEqual(Self.imports(in: "@testable\nimport Foundation\n"), [S(module: "Foundation", testable: true)])
    }
}
