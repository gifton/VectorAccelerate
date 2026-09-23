import XCTest

/// Error-rendering rule: the bench CLI prints a caught error by interpolating it (`"error: \(error)"`), so an `Error` type
/// without a description reaches the user as its reflected case (`unsupportedSchemaVersion(7)`). Every type declared as
/// conforming to `Error` in `Sources/VectorTestKit`, and likewise in `Sources/VectorAccelerateBench`, must be declared
/// `CustomStringConvertible` somewhere in the same directory.
///
/// `LocalizedError` is deliberately not accepted as a description: interpolation does not consult `errorDescription`
/// (`testInterpolationIgnoresLocalizedErrorDescription` pins that), so such a type would still print as its reflected case.
///
/// Contract — the declaration shapes the scanner recognises; anything outside this list is not seen. Each bullet cites the rows
/// of `testScannerFixtures` that pin it, and `testContractCitesExactlyTheFixtureRows` checks that every cited label is a row
/// and every row is cited (`FixtureCitations`):
/// - a nominal head `enum|struct|class|actor Name[<…>]: A, B, … {` whose inheritance clause lists `Error`, `Swift.Error` or
///   `LocalizedError` [pins: enum-error-flagged, struct-inline-description, class-swift-error-flagged, actor-flagged,
///   actor-described, localized-error-flagged]; the clause may span lines, ends at the first `{` or at a `where`, and drops
///   attributes such as `@unchecked` [pins: multiline-clause-flagged, generic-where-flagged, unchecked-attribute-flagged];
/// - an extension head `extension [Outer.]Name: A, B, … {` that adds one of those, or that adds `CustomStringConvertible`, in
///   the same file or another [pins: extension-adds-error-flagged, extension-adds-description, nested-extension-described,
///   description-in-other-file];
/// - a type is keyed by the last component of its name, so two types sharing a simple name are not told apart [pins:
///   nested-extension-described, shared-simple-name-merged];
/// - `//` comments are removed before matching [pins: comments-unseen]; `/* */` blocks are scanned as code
///   [pins: block-comment-scanned]; string literals are scanned as code and a `//` inside one truncates that line, hiding what
///   follows it [pins: string-slash-truncates];
/// - not recognised: `Error` inherited through another protocol (`protocol KitError: Error`) or through a typealias [pins:
///   protocol-inherited-unseen, typealias-unseen]; a type declared in another module [pins: none — a fixture is one source
///   string, so another module cannot appear in a scan]; a `description` property without the `CustomStringConvertible`
///   conformance, because interpolation ignores it [pins: description-without-conformance]; and, as non-matches, a name that
///   merely contains `Error`, a `CustomStringConvertible` that is not an `Error`, and the `class` keyword on members [pins:
///   error-in-name-unseen, described-not-error, class-keyword-members-unseen].
final class ErrorDescriptionLintTests: XCTestCase {
    struct Scan: Equatable {
        var errorTypes: Set<String> = []
        var describedTypes: Set<String> = []
        var undescribed: [String] { errorTypes.subtracting(describedTypes).sorted() }
    }

    // Group 1: the type's name, possibly qualified (`Outer.Inner`). Group 2: the inheritance clause, up to the first `{`.
    private static let head = try! NSRegularExpression(
        pattern: #"\b(?:enum|struct|class|actor|extension)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*(?:<[^>{]*>)?\s*:\s*([^{]*)\{"#)

    /// The simple names listed in an inheritance clause: `where` clause cut off, attributes (`@unchecked`) and module
    /// qualifiers (`Swift.`) dropped.
    static func conformances(in clause: String) -> Set<String> {
        let list = clause.range(of: #"\bwhere\b"#, options: .regularExpression).map { String(clause[..<$0.lowerBound]) } ?? clause
        return Set(list.split(separator: ",").compactMap { entry in
            entry.split(whereSeparator: \.isWhitespace).last { !$0.hasPrefix("@") }?.split(separator: ".").last.map(String.init)
        })
    }

    static func scan(_ sources: [String]) -> Scan {
        var result = Scan()
        for source in sources {
            let code = source.split(separator: "\n", omittingEmptySubsequences: false)
                .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }
                .joined(separator: "\n")
            for match in head.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                // Both groups are mandatory in `head`, so a match always has both ranges and this never skips one.
                guard let nameRange = Range(match.range(at: 1), in: code), let clauseRange = Range(match.range(at: 2), in: code) else { continue }
                let qualified = String(code[nameRange])
                let name = qualified.split(separator: ".").last.map(String.init) ?? qualified
                let listed = conformances(in: String(code[clauseRange]))
                if !listed.isDisjoint(with: ["Error", "LocalizedError"]) { result.errorTypes.insert(name) }
                if listed.contains("CustomStringConvertible") { result.describedTypes.insert(name) }
            }
        }
        return result
    }

    /// The fixture rows, keyed by the labels the doc comment's `[pins: …]` citations name.
    static let fixtures: [(label: String, sources: [String], errorTypes: Set<String>, undescribed: [String])] = [
        ("enum-error-flagged", ["enum A: Error { case x }"], ["A"], ["A"]),
        ("extension-adds-description", ["public enum B: Error, Equatable {\n    case x\n}\n\nextension B: CustomStringConvertible {\n}"], ["B"], []),
        ("struct-inline-description", ["struct C: Error, CustomStringConvertible { var description: String { \"c\" } }"], ["C"], []),
        ("class-swift-error-flagged", ["final class D: Swift.Error {}"], ["D"], ["D"]),
        ("extension-adds-error-flagged", ["struct E {}\nextension E: Error {}"], ["E"], ["E"]),
        ("localized-error-flagged", ["enum F: LocalizedError { case x }"], ["F"], ["F"]),
        ("multiline-clause-flagged", ["public enum G:\n    Error,\n    Equatable {\n    case x\n}"], ["G"], ["G"]),
        ("description-in-other-file", ["enum H: Error { case x }", "extension H: CustomStringConvertible { var description: String { \"h\" } }"], ["H"], []),
        ("nested-extension-described", ["enum Outer { enum I: Error { case x } }\nextension Outer.I: CustomStringConvertible { }"], ["I"], []),
        ("generic-where-flagged", ["struct J<T>: Error where T: Sendable { }"], ["J"], ["J"]),
        ("unchecked-attribute-flagged", ["final class K: @unchecked Sendable, Error { }"], ["K"], ["K"]),
        ("description-without-conformance", ["enum L: Error { case x; var description: String { \"l\" } }"], ["L"], ["L"]),
        ("actor-flagged", ["actor Boom: Error {}"], ["Boom"], ["Boom"]),
        ("actor-described", ["actor Calm: Error, CustomStringConvertible { var description: String { \"calm\" } }"], ["Calm"], []),
        ("shared-simple-name-merged", ["enum Outer { enum Dup: Error { case x } }\nstruct Dup: CustomStringConvertible { var description: String { \"\" } }"], ["Dup"], []),
        ("comments-unseen", ["// enum M: Error { case x }\n/// enum N: Error { case x }"], [], []),
        ("block-comment-scanned", ["/* enum Z: Error { case x } */"], ["Z"], ["Z"]),
        ("string-slash-truncates", ["let url = \"http://x\"; enum W: Error { case x }"], [], []),
        ("protocol-inherited-unseen", ["protocol KitError: Error {}\nenum X: KitError { case x }"], [], []),
        ("typealias-unseen", ["typealias Failure = Error\nenum Y: Failure { case x }"], [], []),
        ("error-in-name-unseen", ["enum ErrorKind: String, Codable { case error }"], [], []),
        ("described-not-error", ["struct P: CustomStringConvertible { var description: String { \"p\" } }"], [], []),
        ("class-keyword-members-unseen", ["final class Q { class var shared: Q { Q() }\n    class func make() -> Q { Q() } }"], [], [])
    ]

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation fails here.
    func testScannerFixtures() {
        for row in Self.fixtures {
            let scan = Self.scan(row.sources)
            XCTAssertEqual(scan.errorTypes, row.errorTypes, "\(row.label): error types in: \(row.sources)")
            XCTAssertEqual(scan.undescribed, row.undescribed, "\(row.label): undescribed in: \(row.sources)")
        }
    }

    /// The doc comment above cannot claim a pin that does not exist: its `[pins: …]` labels are exactly the fixture rows.
    func testContractCitesExactlyTheFixtureRows() throws {
        try FixtureCitations.assertContractCitesExactlyTheFixtureRows(ofHarnessFile: "ErrorDescriptionLintTests.swift",
                                                                       fixtureLabels: Self.fixtures.map(\.label))
    }

    private enum LocalizedOnly: LocalizedError {
        case broken
        var errorDescription: String? { "a readable message" }
    }

    /// Why `LocalizedError` is not accepted: the interpolation the CLI uses prints the reflected case, not `errorDescription`.
    func testInterpolationIgnoresLocalizedErrorDescription() {
        let message = "\(LocalizedOnly.broken as any Error)"
        XCTAssertEqual(message, "broken")
        XCTAssertFalse(message.contains("a readable message"))
    }

    func testEveryErrorTypeHasADescription() throws {
        // The known types prove the scan is not vacuous: a scanner change that stops seeing one of these real declarations
        // fails here. Renaming or removing one of them means updating this table.
        let directories: [(path: String, known: Set<String>)] = [
            ("Sources/VectorTestKit", ["ArtifactValidationError", "CaseIDError", "RecallError", "RunStoreError"]),
            ("Sources/VectorAccelerateBench", ["BenchError", "ContextError"])
        ]
        for (path, known) in directories {
            let files = RepoPaths.swiftFiles(under: path)
            XCTAssertFalse(files.isEmpty, "\(path) must contain Swift files")
            let scan = Self.scan(try files.map { try String(contentsOf: $0, encoding: .utf8) })
            XCTAssertTrue(scan.errorTypes.isSuperset(of: known), "\(path): the scanner no longer sees \(known.subtracting(scan.errorTypes).sorted())")
            XCTAssertEqual(scan.undescribed, [], "\(path): Error types without a CustomStringConvertible conformance print as raw cases in the CLI")
        }
    }
}
