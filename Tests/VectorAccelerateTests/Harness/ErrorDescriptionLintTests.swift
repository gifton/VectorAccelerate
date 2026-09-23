import XCTest

/// Error-rendering rule: the bench CLI prints a caught error by interpolating it (`"error: \(error)"`), so an `Error` type
/// without a description reaches the user as its reflected case (`unsupportedSchemaVersion(7)`). Every type declared as
/// conforming to `Error` in `Sources/VectorTestKit`, and likewise in `Sources/VectorAccelerateBench`, must be declared
/// `CustomStringConvertible` somewhere in the same directory.
///
/// `LocalizedError` is deliberately not accepted as a description: interpolation does not consult `errorDescription`
/// (`testInterpolationIgnoresLocalizedErrorDescription` pins that), so such a type would still print as its reflected case.
///
/// Contract — the declaration shapes the scanner recognises; anything outside this list is not seen:
/// - a nominal head `enum|struct|class|actor Name[<…>]: A, B, … {` whose inheritance clause lists `Error`, `Swift.Error` or
///   `LocalizedError`; the clause may span lines and ends at the first `{` or at a `where`;
/// - an extension head `extension [Outer.]Name: A, B, … {` that adds one of those, or that adds `CustomStringConvertible`;
/// - a type is keyed by the last component of its name, so two types sharing a simple name are not told apart;
/// - `//` comments are removed before matching. `/* */` blocks and string literals are scanned as code, and a `//` inside a
///   string literal truncates that line.
/// Not recognised: `Error` inherited through another protocol (`protocol KitError: Error`), through a typealias, or declared in
/// another module. A `description` property without the `CustomStringConvertible` conformance is not accepted either, because
/// interpolation ignores it. `testScannerFixtures` pins each shape named above.
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

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation fails here.
    func testScannerFixtures() {
        let cases: [(sources: [String], errorTypes: Set<String>, undescribed: [String])] = [
            (["enum A: Error { case x }"], ["A"], ["A"]),
            (["public enum B: Error, Equatable {\n    case x\n}\n\nextension B: CustomStringConvertible {\n}"], ["B"], []),
            (["struct C: Error, CustomStringConvertible { var description: String { \"c\" } }"], ["C"], []),
            (["final class D: Swift.Error {}"], ["D"], ["D"]),
            (["struct E {}\nextension E: Error {}"], ["E"], ["E"]),
            (["enum F: LocalizedError { case x }"], ["F"], ["F"]),
            (["public enum G:\n    Error,\n    Equatable {\n    case x\n}"], ["G"], ["G"]),
            (["enum H: Error { case x }", "extension H: CustomStringConvertible { var description: String { \"h\" } }"], ["H"], []),
            (["enum Outer { enum I: Error { case x } }\nextension Outer.I: CustomStringConvertible { }"], ["I"], []),
            (["struct J<T>: Error where T: Sendable { }"], ["J"], ["J"]),
            (["final class K: @unchecked Sendable, Error { }"], ["K"], ["K"]),
            (["enum L: Error { case x; var description: String { \"l\" } }"], ["L"], ["L"]),
            (["actor Boom: Error {}"], ["Boom"], ["Boom"]),
            (["actor Calm: Error, CustomStringConvertible { var description: String { \"calm\" } }"], ["Calm"], []),
            (["// enum M: Error { case x }\n/// enum N: Error { case x }"], [], []),
            (["enum ErrorKind: String, Codable { case error }"], [], []),
            (["struct P: CustomStringConvertible { var description: String { \"p\" } }"], [], []),
            (["final class Q { class var shared: Q { Q() }\n    class func make() -> Q { Q() } }"], [], [])
        ]
        for (sources, errorTypes, undescribed) in cases {
            let scan = Self.scan(sources)
            XCTAssertEqual(scan.errorTypes, errorTypes, "error types in: \(sources)")
            XCTAssertEqual(scan.undescribed, undescribed, "undescribed in: \(sources)")
        }
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
