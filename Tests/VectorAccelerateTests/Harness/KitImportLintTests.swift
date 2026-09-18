import XCTest

/// Genericism rule (spec §4.1): the kit imports only Foundation, Darwin, CryptoKit and never uses @testable.
///
/// The scan is line-anchored and covers attribute-prefixed imports (`@testable`, `@preconcurrency`, `@_exported`,
/// `@_implementationOnly`, `@_spi(...)`, in any order and count, including an attribute on the line before `import`),
/// SE-0409 access-level modifiers (`private import Metal`, `@_exported public import Metal`), and qualified imports
/// (`import struct Foundation.Data` reports module `Foundation`). A commented-out line such as `// import Metal`
/// does not match because `^\s*` must be followed directly by an attribute, an access-level modifier, or `import`.
/// `testClassifierFixtures` pins each of these behaviours.
///
/// Build-configuration invariance (Task 3 finding): kit code never calls `sin`/`cos`/`tan` directly; see
/// `testKitHasNoBareTrigCalls`.
final class KitImportLintTests: XCTestCase {
    private static let allowed: Set<String> = ["Foundation", "Darwin", "CryptoKit"]

    /// One import statement as seen by the lint.
    struct ImportStatement: Equatable {
        var module: String
        var testable: Bool
    }

    // Group 1: zero or more attributes (each with optional parenthesized arguments) preceding `import`, which may itself
    //          carry one SE-0409 access-level modifier (`public`, `package`, `internal`, `fileprivate`, `private`).
    // Group 2: the top-level module name, after an optional import-kind keyword and before any `.` qualifier.
    private static let importLine = try! NSRegularExpression(
        pattern: #"^\s*((?:@\w+(?:\([^)]*\))?\s+)*)(?:(?:public|package|internal|fileprivate|private)\s+)?import\s+(?:(?:typealias|struct|class|enum|protocol|actor|let|var|func)\s+)?([A-Za-z0-9_]+)"#,
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
            ("private import Metal",                   S(module: "Metal", testable: false)),
            ("package import Metal",                   S(module: "Metal", testable: false)),
            ("@_exported public import Metal",         S(module: "Metal", testable: false)),
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

    // MARK: - Build-configuration-invariant transcendentals

    // Group 1: the trig function name. `\b` rejects `asin(` and, case-sensitively, `opaqueSin(`; `\s*\(` rejects `sinh(`/`cosine(`.
    private static let bareTrigCall = try! NSRegularExpression(pattern: #"\b(sin|cos|sinf|cosf|tan|tanf)\s*\("#)

    /// True when `line` contains a direct trig call. The `//` comment, if any, is removed first, so the wrapper's own doc
    /// comment naming `sin(x)` does not match and a comment cannot exempt a call that precedes it on the same line. The
    /// only exempt code is a wrapper definition: a line whose code part contains `func opaqueSin(` or `func opaqueCos(`,
    /// which is why each wrapper's body must stay on its `func` line. Known limits, both fail closed or are absent from the
    /// kit: `//` inside a string literal truncates that line's scan (no kit file has one); `/* */` blocks are scanned as code.
    static func hasBareTrigCall(_ line: String) -> Bool {
        var code = Substring(line)
        if let comment = code.range(of: "//") { code = code[..<comment.lowerBound] }
        if code.contains("func opaqueSin(") || code.contains("func opaqueCos(") { return false }
        let s = String(code)
        return bareTrigCall.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    /// Build-configuration invariance (Task 3 finding; see "Build-configuration-invariant transcendentals" in
    /// `SeededRNG.swift`): under `-O`, LLVM fuses an adjacent `sinf`/`cosf` pair inside one function into Darwin's
    /// `__sincosf_stret`, which differs from standalone `sinf`/`cosf` by 1-2 ulp for some inputs, so a stream frozen at
    /// `-Onone` would not reproduce at `-O` (6 of 384 frozen Gaussian values diverged before the fix). Rule: kit code never
    /// calls `sin`/`cos`/`sinf`/`cosf`/`tan`/`tanf` directly; it goes through the `@inline(never)` `opaqueSin`/`opaqueCos`
    /// wrappers in `SeededRNG.swift`, and any new transcendental that could fuse gets a sibling wrapper there. The debug
    /// (`-Onone`) stream is the contract by ruling; `SeededRNGGoldenTests` and `VectorGeneratorGoldenTests` prove both
    /// configurations match. Fixture lines pin the classifier first so a pattern change that reopens a bypass fails here.
    func testKitHasNoBareTrigCalls() throws {
        let cases: [(line: String, violation: Bool)] = [
            ("let z = sin(x)",                                                 true),
            ("let z = cos (x)",                                                true),
            ("        let z0 = r * cos(theta)",                                true),
            ("let z = Foundation.sin(x)",                                      true),
            ("let z = sinf(x) + cosf(x)",                                      true),
            ("let z = tan(x)",                                                 true),
            ("let z = tanf(x)",                                                true),
            ("let z = sin(x) // func opaqueSin( in a comment does not exempt", true),
            ("let z = opaqueSin(x) + opaqueCos(x)",                            false),
            ("let z = asin(x) + sinh(x) + cosine(x)",                          false),
            ("let n = sqrt(x) + log(x) + pow(x, 2)",                           false),
            ("/// `sin(x)` behind an inlining barrier",                        false),
            ("// Box-Muller calls `sin` and `cos` on the same argument.",      false),
            ("let z0 = r * opaqueCos(theta) // was cos(theta)",                false),
            ("internal func opaqueSin(_ x: Float) -> Float { sin(x) }",        false),
            ("internal func opaqueCos(_ x: Float) -> Float { cos(x) }",        false),
        ]
        for (line, violation) in cases {
            XCTAssertEqual(Self.hasBareTrigCall(line), violation, "line: \(line)")
        }

        let files = RepoPaths.swiftFiles(under: "Sources/VectorTestKit")
        XCTAssertFalse(files.isEmpty, "Sources/VectorTestKit must contain Swift files")
        var violations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated()
            where Self.hasBareTrigCall(String(line)) {
                violations.append("\(file.lastPathComponent):\(index + 1): bare trig call; route it through opaqueSin/opaqueCos: \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertTrue(violations.isEmpty, violations.joined(separator: "\n"))
    }
}
