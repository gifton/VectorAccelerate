import XCTest

/// Genericism rule (spec §4.1): the kit imports only Foundation, Darwin, CryptoKit and never uses @testable.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    func testKitImportsAreWithinAllowlist() throws {
        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        let importLine = try NSRegularExpression(pattern: #"^\s*(@testable\s+)?import\s+([A-Za-z0-9_]+)"#, options: [.anchorsMatchLines])
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            let range = NSRange(text.startIndex..., in: text)
            for m in importLine.matches(in: text, range: range) {
                let testable = m.range(at: 1).location != NSNotFound
                let module = String(text[Range(m.range(at: 2), in: text)!])
                if testable { violations.append("\(file.lastPathComponent): @testable import is forbidden") }
                if !Self.allowed.contains(module) { violations.append("\(file.lastPathComponent): import \(module) is outside the allowlist \(Self.allowed.sorted())") }
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }
}
