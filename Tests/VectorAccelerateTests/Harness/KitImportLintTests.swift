import XCTest

/// Genericism rule (spec §4.1): the kit imports only Foundation, Darwin, CryptoKit and never uses @testable.
///
/// The scan is line-based and covers attribute-prefixed imports (`@testable`, `@preconcurrency`, `@_exported`,
/// `@_implementationOnly`, `@_spi(...)`, in any order and count) and qualified imports
/// (`import struct Foundation.Data` reports module `Foundation`). A commented-out line such as `// import Metal`
/// does not match because `^\s*` must be followed directly by an attribute or the `import` keyword.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    func testKitImportsAreWithinAllowlist() throws {
        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        // Group 1: zero or more attributes (each with optional parenthesized arguments) preceding `import`.
        // Group 2: the top-level module name, after an optional import-kind keyword and before any `.` qualifier.
        let importLine = try NSRegularExpression(
            pattern: #"^\s*((?:@\w+(?:\([^)]*\))?\s+)*)import\s+(?:(?:typealias|struct|class|enum|protocol|actor|let|var|func)\s+)?([A-Za-z0-9_]+)"#,
            options: [.anchorsMatchLines])
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            let range = NSRange(text.startIndex..., in: text)
            for m in importLine.matches(in: text, range: range) {
                let attributes = m.range(at: 1).location == NSNotFound ? "" : String(text[Range(m.range(at: 1), in: text)!])
                let testable = attributes.split(whereSeparator: \.isWhitespace).contains("@testable")
                let module = String(text[Range(m.range(at: 2), in: text)!])
                if testable { violations.append("\(file.lastPathComponent): @testable import is forbidden") }
                if !Self.allowed.contains(module) { violations.append("\(file.lastPathComponent): import \(module) is outside the allowlist \(Self.allowed.sorted())") }
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }
}
