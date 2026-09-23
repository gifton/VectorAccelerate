import Foundation
import XCTest

/// The pins contract of the scanner lints (`ErrorDescriptionLintTests`, `CollectionInterpolationLintTests`): every `- ` bullet
/// of a lint's leading doc comment ends in `[pins: label-a, label-b]`, naming the rows of its fixture table that exercise the
/// bullet, or in exactly `[pins: none — <why>]` when no fixture can. `assertContractCitesExactlyTheFixtureRows` reads the
/// lint's own source and fails when a cited label has no row, a row is never cited, a bullet ends without a citation, or a
/// citation is malformed — so a doc comment cannot claim a pin that does not exist. `FixtureCitationsTests` pins the parser.
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

    /// Reads `Tests/VectorAccelerateTests/Harness/<fileName>`; every contract bullet must end with a citation, and the labels
    /// cited anywhere in the leading doc comment must equal `fixtureLabels` as sets (which must be well-formed and unique).
    static func assertContractCitesExactlyTheFixtureRows(ofHarnessFile fileName: String, fixtureLabels: [String],
                                                         file: StaticString = #filePath, line: UInt = #line) throws {
        let url = RepoPaths.root.appendingPathComponent("Tests/VectorAccelerateTests/Harness/\(fileName)")
        let doc = leadingDocComment(inSource: try String(contentsOf: url, encoding: .utf8))
        let bullets = contractBullets(inLeadingDocComment: doc)
        XCTAssertFalse(bullets.isEmpty, "\(fileName): no `- ` bullet in the leading doc comment", file: file, line: line)
        for bullet in bullets where bullet.range(of: #"\[pins: [^\]]*\][;.]?$"#, options: .regularExpression) == nil {
            XCTFail("\(fileName): a contract bullet does not end with a [pins: …] citation: \(bullet.prefix(90))…", file: file, line: line)
        }
        let cited = try citedLabels(in: doc.joined(separator: " "))
        for label in fixtureLabels where !isLabel(label) { XCTFail("\(fileName): fixture label `\(label)` is not a label", file: file, line: line) }
        let rows = Set(fixtureLabels)
        XCTAssertEqual(rows.count, fixtureLabels.count, "\(fileName): duplicate fixture labels", file: file, line: line)
        XCTAssertEqual(cited.subtracting(rows).sorted(), [], "\(fileName): cited in the contract but no fixture row", file: file, line: line)
        XCTAssertEqual(rows.subtracting(cited).sorted(), [], "\(fileName): fixture rows the contract never cites", file: file, line: line)
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
}
