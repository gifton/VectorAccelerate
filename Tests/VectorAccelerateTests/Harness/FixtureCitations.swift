import Foundation
import XCTest

/// The pins contract of the scanner lints (`ErrorDescriptionLintTests`, `CollectionInterpolationLintTests`): every `- ` bullet
/// of a lint's leading doc comment ends in `[pins: label-a, label-b]`, naming the rows of its fixture table that exercise the
/// bullet, or in exactly `[pins: none — <why>]` when no fixture can. `assertContractCitesExactlyTheFixtureRows` reads the
/// lint's own source and fails when a cited label has no row, a row is never cited, a bullet ends without a citation, the
/// fixture table is empty, or a citation is malformed — so a doc comment cannot claim a pin that does not exist.
/// `FixtureCitationsTests` pins the parser and each of those conditions.
enum FixtureCitations {
    struct Malformed: Error, CustomStringConvertible { var description: String }

    /// A label: a lowercase letter, then lowercase letters and digits, groups joined by single hyphens (`actor-flagged`).
    static func isLabel(_ text: String) -> Bool {
        text.range(of: #"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"#, options: .regularExpression) != nil
    }

    /// The first run of `///` lines in `source`, with the `///` removed and nothing else changed (indentation is what tells a
    /// bullet's continuation line from a new paragraph).
    static func leadingDocComment(inSource source: String) -> [String] {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .drop { !$0.hasPrefix("///") }.prefix { $0.hasPrefix("///") }
            .map { String($0.dropFirst(3)) }
    }

    /// The `- ` bullets of a leading doc comment, each joined with its continuation lines (indented by three or more spaces);
    /// any other line ends the bullet.
    static func contractBullets(inLeadingDocComment lines: [String]) -> [String] {
        var bullets: [String] = []
        var current: String?
        for line in lines {
            if line.hasPrefix(" - ") {
                if let open = current { bullets.append(open) }
                current = String(line.dropFirst(3))
            } else if let open = current, line.hasPrefix("   ") {
                current = open + " " + line.trimmingCharacters(in: .whitespaces)
            } else {
                if let open = current { bullets.append(open) }
                current = nil
            }
        }
        if let open = current { bullets.append(open) }
        return bullets
    }

    /// The labels cited by every `[pins: …]` in `text`. `[pins: none — <why>]` with a non-empty why cites nothing; any other
    /// content must be a comma-separated list of labels. Anything else throws `Malformed`.
    static func citedLabels(in text: String) throws -> Set<String> {
        let citation = try NSRegularExpression(pattern: #"\[pins: ([^\]]*)\]"#)
        var labels: Set<String> = []
        for match in citation.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range(at: 1), in: text) else { continue }
            let body = String(text[range])
            if body.hasPrefix("none — "), !body.dropFirst("none — ".count).trimmingCharacters(in: .whitespaces).isEmpty { continue }
            if body.split(separator: " ").first == "none" || body.hasPrefix("none—") || body.hasPrefix("none-") {
                throw Malformed(description: "the no-fixture form is exactly `[pins: none — <why>]`, got `[pins: \(body)]`")
            }
            for part in body.split(separator: ",", omittingEmptySubsequences: false) {
                let label = part.trimmingCharacters(in: .whitespaces)
                guard isLabel(label) else { throw Malformed(description: "`\(label)` is not a label, in `[pins: \(body)]`") }
                labels.insert(label)
            }
        }
        return labels
    }

    /// The problems with one lint's contract, given the lint's source and its fixture labels; empty when the contract is sound.
    /// Reported, in this order: no `- ` bullet at all; a bullet that does not end in a citation; a fixture label that is not a
    /// label; duplicate labels; an empty table (under all-`none` citations an empty table would otherwise pass as ∅ == ∅); a
    /// cited label with no row; a row the contract never cites. A malformed citation throws `Malformed`. Each condition is
    /// pinned by `FixtureCitationsTests.testContractProblemsNameEachCondition`.
    static func contractProblems(fileName: String, source: String, fixtureLabels: [String]) throws -> [String] {
        let doc = leadingDocComment(inSource: source)
        let bullets = contractBullets(inLeadingDocComment: doc)
        var problems: [String] = []
        if bullets.isEmpty { problems.append("\(fileName): no `- ` bullet in the leading doc comment") }
        for bullet in bullets where bullet.range(of: #"\[pins: [^\]]*\][;.]?$"#, options: .regularExpression) == nil {
            problems.append("\(fileName): a contract bullet does not end with a [pins: …] citation: \(bullet.prefix(90))…")
        }
        let cited = try citedLabels(in: doc.joined(separator: " "))
        for label in fixtureLabels where !isLabel(label) { problems.append("\(fileName): fixture label `\(label)` is not a label") }
        let rows = Set(fixtureLabels)
        if rows.count != fixtureLabels.count { problems.append("\(fileName): duplicate fixture labels") }
        if rows.isEmpty { problems.append("\(fileName): no fixture rows") }
        let unrowed = cited.subtracting(rows).sorted(), uncited = rows.subtracting(cited).sorted()
        if !unrowed.isEmpty { problems.append("\(fileName): cited in the contract but no fixture row: \(unrowed.joined(separator: ", "))") }
        if !uncited.isEmpty { problems.append("\(fileName): fixture rows the contract never cites: \(uncited.joined(separator: ", "))") }
        return problems
    }

    /// Reads `Tests/VectorAccelerateTests/Harness/<fileName>` and fails on every problem `contractProblems` reports for it.
    static func assertContractCitesExactlyTheFixtureRows(ofHarnessFile fileName: String, fixtureLabels: [String],
                                                         file: StaticString = #filePath, line: UInt = #line) throws {
        let url = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/\(fileName)")
        let problems = try contractProblems(fileName: fileName, source: try String(contentsOf: url, encoding: .utf8), fixtureLabels: fixtureLabels)
        XCTAssertEqual(problems, [], problems.joined(separator: "\n"), file: file, line: line)
    }
}

/// Pins the parser above on strings, so a change that loosens the `none` form or the bullet joining fails here.
final class FixtureCitationsTests: XCTestCase {
    func testCitedLabelsAcceptsListsAndTheNoneFormOnly() throws {
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "x [pins: actor-flagged, dict2-unseen] y [pins: a1]"), ["actor-flagged", "dict2-unseen", "a1"])
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "[pins: none — a fixture is one source string]"), [])
        XCTAssertEqual(try FixtureCitations.citedLabels(in: "no citation here"), [])
        for malformed in ["[pins: none]", "[pins: none —]", "[pins: none — ]", "[pins: none - why]", "[pins: None — why]", "[pins: none—why]",
                          "[pins: Actor-Flagged]", "[pins: a, ]", "[pins: ]", "[pins: a b]", "[pins: a--b]", "[pins: -a]", "[pins: 1a]"] {
            XCTAssertThrowsError(try FixtureCitations.citedLabels(in: malformed), malformed)
        }
    }

    func testContractBulletsJoinContinuationLinesAndStopAtParagraphs() throws {
        let source = "import XCTest\n\n/// Intro line.\n///\n/// Contract:\n/// - first bullet [pins: a]\n///   continued [pins: b];\n"
            + "/// - second\n///   [pins: none — why].\n/// Trailing paragraph [pins: c].\nfinal class T {}\n/// - after the class [pins: d]\n"
        let doc = FixtureCitations.leadingDocComment(inSource: source)
        XCTAssertEqual(FixtureCitations.contractBullets(inLeadingDocComment: doc), ["first bullet [pins: a] continued [pins: b];", "second [pins: none — why]."])
        // Citations anywhere in the leading block count; the block ends at the first non-`///` line.
        XCTAssertEqual(try FixtureCitations.citedLabels(in: doc.joined(separator: " ")), ["a", "b", "c"])
    }

    /// Each condition `contractProblems` reports, on synthetic sources — in particular an empty fixture table under all-`none`
    /// citations is a problem, not a vacuous ∅ == ∅ match.
    func testContractProblemsNameEachCondition() throws {
        func problems(_ source: String, _ labels: [String]) throws -> [String] {
            try FixtureCitations.contractProblems(fileName: "L.swift", source: source, fixtureLabels: labels)
        }
        let sound = "/// Rule.\n/// - one [pins: a, b];\n/// - two [pins: none — why].\nfinal class L {}\n"
        XCTAssertEqual(try problems(sound, ["a", "b"]), [])
        XCTAssertEqual(try problems("/// Rule.\n/// - two [pins: none — why].\nfinal class L {}\n", []), ["L.swift: no fixture rows"])
        XCTAssertEqual(try problems("/// Rule only.\nfinal class L {}\n", ["a"]),
                       ["L.swift: no `- ` bullet in the leading doc comment", "L.swift: fixture rows the contract never cites: a"])
        XCTAssertEqual(try problems("/// - one without a citation\nfinal class L {}\n", ["a"]),
                       ["L.swift: a contract bullet does not end with a [pins: …] citation: one without a citation…",
                        "L.swift: fixture rows the contract never cites: a"])
        XCTAssertEqual(try problems(sound, ["a", "b", "Bad_Label"]),
                       ["L.swift: fixture label `Bad_Label` is not a label", "L.swift: fixture rows the contract never cites: Bad_Label"])
        XCTAssertEqual(try problems(sound, ["a", "b", "a"]), ["L.swift: duplicate fixture labels"])
        XCTAssertEqual(try problems(sound, ["a"]), ["L.swift: cited in the contract but no fixture row: b"])
        XCTAssertEqual(try problems(sound, ["a", "b", "c"]), ["L.swift: fixture rows the contract never cites: c"])
        XCTAssertThrowsError(try problems("/// - one [pins: none]\nfinal class L {}\n", ["a"]))
    }
}
