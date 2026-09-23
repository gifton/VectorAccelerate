import XCTest

/// Rendering rule: interpolating an array or dictionary whole (`"reasons: \(preflight.reasons)"`) prints Swift's debug
/// literal — `["drift 6.5% exceeds 5.0%", "thermal state end=fair (not nominal)"]` — which is not a message. `CompareCommand`
/// (Task 16 review) and `RunCommand` (Task 17) each did it once; this closes the class. In `Sources/VectorTestKit` and
/// `Sources/VectorAccelerateBench`, no string-interpolation segment may consist of a bare or dotted reference to a name
/// declared with an explicit array or dictionary type. Render the elements instead (`.joined(separator: "; ")`).
///
/// Contract — what the scanner recognises; anything outside this list is not seen. Each bullet cites the rows of
/// `testScannerFixtures` that pin it, and `testContractCitesExactlyTheFixtureRows` checks that every cited label is a row and
/// every row is cited (`FixtureCitations`):
/// - a collection name is the `name` in `let name: [` or `var name: [` (stored, computed and local) [pins: let-array-flagged,
///   var-dict-flagged, self-dotted-flagged], or in a parameter `[label ]name: [` of a `func` or `init` declaration's parameter
///   clause — the balanced parenthesised text after the name, so multi-line clauses and defaults with parentheses are read
///   [pins: func-params-flagged, init-multiline-default-flagged]; a call-site `.init(` is not a declaration and contributes
///   nothing [pins: call-site-init-unseen];
/// - the type must START with `[`: `[T]`, `[[T]]`, `[K: V]` and `[T]?` count [pins: let-array-flagged, nested-optional-flagged,
///   var-dict-flagged], while `Set<T>`, `Array<T>`, `Dictionary<K, V>`, `inout [T]`, every inferred type and every scalar do
///   not [pins: set-array-generic-unseen, dict-generic-unseen, inout-unseen, inferred-unseen, scalar-unflagged];
/// - a violation is a segment `\(chain)` whose whole content is an identifier chain — `reasons`, `preflight.reasons`,
///   `self.deltas`, `o.i.zs` — with a collection name as its LAST component; whitespace inside the parentheses is allowed
///   [pins: let-array-flagged, prefix-runcommand-flagged, self-dotted-flagged, dotted-three-flagged, padded-segment-flagged].
///   A chain that ends in a call, a subscript or another member (`.joined(…)`, `.map(…)`, `[0]`, `.count`, `.first`,
///   `.isEmpty`) is not one, nor is an inline literal or a call result [pins: operations-unflagged, literal-call-unflagged];
/// - a bare chain resolves against the names declared in the SAME file; a dotted chain's last component resolves against every
///   `let`/`var` collection name in all scanned files, since a member may be declared anywhere [pins: bare-same-file-only,
///   dotted-any-file-flagged]. Within a file two declarations sharing a name are not told apart, so a scalar that shares its
///   name with a collection declared in the same file is flagged too (rename one of them) [pins: shared-name-both-flagged];
/// - not seen: closure parameters, subscript parameters, enum associated values and tuple-type elements [pins:
///   closure-param-unseen, subscript-param-unseen, enum-case-unseen, tuple-element-unseen]; the parameters of operator
///   functions (`func == (`) and of functions whose generic clause nests angle brackets (`func g<T: P<U>>(`) [pins:
///   operator-func-unseen, nested-generic-unseen];
/// - `//` comments are removed before matching, for declarations and for uses [pins: comments-unseen]; `/* */` blocks are
///   scanned as code [pins: block-comment-scanned]; string literals are scanned as code and a `//` inside one truncates that
///   line, hiding what follows it [pins: string-slash-truncates].
final class CollectionInterpolationLintTests: XCTestCase {
    /// Group 1: the name, in `let name: [` / `var name: [`.
    private static let declared = try! NSRegularExpression(pattern: #"\b(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\["#)
    /// Opens a `func` or `init` declaration's parameter clause (an `init` preceded by `.` is a call); the clause ends at the
    /// parenthesis that balances this one.
    private static let clauseHead = try! NSRegularExpression(pattern: #"\b(?:func\s+[A-Za-z_][A-Za-z0-9_]*|(?<!\.)init[?!]?)\s*(?:<[^>]*>)?\s*\("#)
    /// Group 1: the name, in `[label ]name: [` at the start of a clause or after a comma.
    private static let parameter = try! NSRegularExpression(pattern: #"(?:^|,)\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\["#)
    /// Group 1: the identifier chain in `\(chain)`, with optional whitespace inside the parentheses.
    private static let segment = try! NSRegularExpression(pattern: #"\\\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\)"#)

    /// `//` comments removed, line structure kept (so line numbers below are source line numbers).
    static func stripped(_ source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }
            .joined(separator: "\n")
    }

    private static func captures(_ regex: NSRegularExpression, in code: String) -> [String] {
        regex.matches(in: code, range: NSRange(code.startIndex..., in: code))
            .compactMap { Range($0.range(at: 1), in: code).map { String(code[$0]) } }
    }

    /// Names declared with an explicit collection type in one stripped file: `let`/`var` declarations and `func`/`init`
    /// parameters.
    static func collectionNames(in code: String) -> Set<String> {
        var names = Set(captures(declared, in: code))
        for head in clauseHead.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
            guard let open = Range(head.range, in: code) else { continue }
            var depth = 1, index = open.upperBound
            while index < code.endIndex, depth > 0 {
                if code[index] == "(" { depth += 1 } else if code[index] == ")" { depth -= 1 }
                index = code.index(after: index)
            }
            let clause = String(code[open.upperBound..<(depth == 0 ? code.index(before: index) : index)])
            names.formUnion(captures(parameter, in: clause))
        }
        return names
    }

    /// One line per violation: `<file>:<line>: \(chain)`, in file order then source order.
    static func findings(in files: [(name: String, code: String)]) -> [String] {
        let perFile = files.map { collectionNames(in: $0.code) }
        let declaredAnywhere = Set(files.flatMap { captures(declared, in: $0.code) })
        var out: [String] = []
        for (file, local) in zip(files, perFile) {
            for match in segment.matches(in: file.code, range: NSRange(file.code.startIndex..., in: file.code)) {
                guard let whole = Range(match.range, in: file.code), let chainRange = Range(match.range(at: 1), in: file.code) else { continue }
                let chain = String(file.code[chainRange])
                let components = chain.split(separator: ".")
                guard let last = components.last.map(String.init) else { continue }
                let hit = components.count == 1 ? local.contains(last) : declaredAnywhere.contains(last)
                if hit {
                    let line = file.code[..<whole.lowerBound].filter { $0 == "\n" }.count + 1
                    out.append("\(file.name):\(line): \\(\(chain))")
                }
            }
        }
        return out
    }

    /// Fixture sources are named `f0.swift`, `f1.swift`, … in order.
    private static func scan(_ sources: [String]) -> (names: Set<String>, findings: [String]) {
        let files = sources.enumerated().map { (name: "f\($0.offset).swift", code: stripped($0.element)) }
        return (Set(files.flatMap { collectionNames(in: $0.code) }), findings(in: files))
    }

    /// The fixture rows, keyed by the labels the doc comment's `[pins: …]` citations name. `names` is the union of the
    /// collection names found in the row's files; `findings` the violations, `f<n>.swift:<line>: \(chain)`.
    static let fixtures: [(label: String, sources: [String], names: Set<String>, findings: [String])] = [
        ("let-array-flagged", ["let reasons: [String] = []\nprint(\"reasons: \\(reasons)\")"], ["reasons"], ["f0.swift:2: \\(reasons)"]),
        ("var-dict-flagged", ["var flags: [String: String] = [:]\n_ = \"\\(flags)\""], ["flags"], ["f0.swift:2: \\(flags)"]),
        ("nested-optional-flagged", ["let raw: [[Double]] = []\nvar extra: [Int]? = nil\n_ = \"\\(raw) \\(extra)\""], ["raw", "extra"],
         ["f0.swift:3: \\(raw)", "f0.swift:3: \\(extra)"]),
        ("func-params-flagged", ["func f(_ items: [Int], with list: [Int], n: Int) -> String { \"\\(items) \\(list) \\(n)\" }"], ["items", "list"],
         ["f0.swift:1: \\(items)", "f0.swift:1: \\(list)"]),
        ("init-multiline-default-flagged",
         ["struct S {\n    init(\n        a: [Int] = Array(repeating: 0, count: 2),\n        b: [String: Int]\n    ) { _ = \"\\(a) \\(b)\" }\n}"],
         ["a", "b"], ["f0.swift:5: \\(a)", "f0.swift:5: \\(b)"]),
        ("padded-segment-flagged", ["let ids: [Int] = []\n_ = \"\\( ids )\""], ["ids"], ["f0.swift:2: \\(ids)"]),
        ("self-dotted-flagged", ["struct S { var deltas: [Int] = []; var text: String { \"\\(self.deltas)\" } }"], ["deltas"], ["f0.swift:1: \\(self.deltas)"]),
        ("dotted-three-flagged", ["struct O { var i: I }\nstruct I { var zs: [Int] }", "func show(_ o: O) -> String { \"\\(o.i.zs)\" }"], ["zs"],
         ["f1.swift:1: \\(o.i.zs)"]),
        // The pre-fix `RunCommand` line verbatim, with the declaration in another file as in the real tree.
        ("prefix-runcommand-flagged",
         ["public struct PreflightRecord { public var reasons: [String] }",
          "print(\"key: \\(metadata.environmentKey)  quiet: \\(preflight.quiet)\" + (preflight.reasons.isEmpty ? \"\" : \"  reasons: \\(preflight.reasons)\"))"],
         ["reasons"], ["f1.swift:1: \\(preflight.reasons)"]),
        ("operations-unflagged",
         ["var notes: [String] = []\n_ = \"\\(notes.joined(separator: \"; \")) \\(notes.count) \\(notes[0]) \\(notes.first) \\(notes.map(\\.count)) \\(notes.isEmpty)\""],
         ["notes"], []),
        ("literal-call-unflagged", ["let xs: [Int] = []\n_ = \"\\([1, 2]) \\(f()) \\(xs.map { $0 })\""], ["xs"], []),
        ("bare-same-file-only", ["struct Series { var raw: [[Double]] }", "func parse(_ raw: String) -> String { \"bad: \\(raw)\" }"], ["raw"], []),
        ("dotted-any-file-flagged", ["struct Series { var raw: [[Double]] }", "func show(_ s: Series) -> String { \"\\(s.raw)\" }"], ["raw"],
         ["f1.swift:1: \\(s.raw)"]),
        ("shared-name-both-flagged", ["func a(_ values: [Int]) -> String { \"\\(values)\" }\nfunc b(_ values: Int) -> String { \"\\(values)\" }"], ["values"],
         ["f0.swift:1: \\(values)", "f0.swift:2: \\(values)"]),
        ("set-array-generic-unseen", ["let s: Set<String> = []\nlet g: Array<Int> = []\n_ = \"\\(s) \\(g)\""], [], []),
        ("dict-generic-unseen", ["let d: Dictionary<String, Int> = [:]\n_ = \"\\(d)\""], [], []),
        ("inferred-unseen", ["let inferred = [1, 2]\n_ = \"\\(inferred)\""], [], []),
        ("scalar-unflagged", ["let none: Int = 0\n_ = \"\\(none)\""], [], []),
        ("inout-unseen", ["func k(_ ys: inout [Int]) -> String { \"\\(ys)\" }"], [], []),
        ("closure-param-unseen", ["let c: ([Int]) -> String = { (zs: [Int]) in \"\\(zs)\" }"], [], []),
        ("subscript-param-unseen", ["struct Table { subscript(keys: [String]) -> String { \"\\(keys)\" } }"], [], []),
        ("enum-case-unseen", ["enum E { case list(items: [Int]) }\n_ = \"\\(items)\""], [], []),
        ("tuple-element-unseen", ["let t: (xs: [Int], n: Int) = ([], 0)\n_ = \"\\(t.xs)\""], [], []),
        ("operator-func-unseen", ["func == (lhs: [Int], rhs: [Int]) -> Bool { \"\\(lhs)\" == \"\\(rhs)\" }"], [], []),
        ("nested-generic-unseen", ["func g<T: P<U>>(xs: [T]) -> String { \"\\(xs)\" }"], [], []),
        ("call-site-init-unseen", ["let made = Maker.init(items: [1, 2])\nself.init(items: [])\n_ = \"\\(items)\""], [], []),
        ("comments-unseen", ["// let hidden: [Int] = []\n/// \\(hidden)\nlet shown: [Int] = [] // trailing \\(shown)"], ["shown"], []),
        ("block-comment-scanned", ["/* let hidden: [Int] = [] */\n_ = \"\\(hidden)\""], ["hidden"], ["f0.swift:2: \\(hidden)"]),
        ("string-slash-truncates", ["let urls: [String] = [\"http://x\"]; _ = \"\\(urls)\""], ["urls"], [])
    ]

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation, or starts seeing a false one,
    /// fails here.
    func testScannerFixtures() {
        for row in Self.fixtures {
            let scan = Self.scan(row.sources)
            XCTAssertEqual(scan.names, row.names, "\(row.label): collection names in: \(row.sources)")
            XCTAssertEqual(scan.findings, row.findings, "\(row.label): findings in: \(row.sources)")
        }
    }

    /// The doc comment above cannot claim a pin that does not exist: its `[pins: …]` labels are exactly the fixture rows.
    func testContractCitesExactlyTheFixtureRows() throws {
        try FixtureCitations.assertContractCitesExactlyTheFixtureRows(ofHarnessFile: "CollectionInterpolationLintTests.swift",
                                                                       fixtureLabels: Self.fixtures.map(\.label))
    }

    func testNoCollectionIsInterpolatedWhole() throws {
        let files = try ["Sources/VectorTestKit", "Sources/VectorAccelerateBench"].flatMap { directory -> [(name: String, code: String)] in
            let urls = RepoPaths.swiftFiles(under: directory)
            XCTAssertFalse(urls.isEmpty, "\(directory) must contain Swift files")
            return try urls.map { (name: $0.lastPathComponent, code: Self.stripped(try String(contentsOf: $0, encoding: .utf8))) }
        }
        // The known names prove the scan is not vacuous: a scanner change that stops seeing one of these real declarations
        // fails here. Renaming or removing one of them means updating this list.
        let declaredAnywhere = Set(files.flatMap { Self.collectionNames(in: $0.code) })
        let known: Set<String> = ["reasons", "notes", "warmupUsed", "flags", "raw", "argv"]
        XCTAssertTrue(declaredAnywhere.isSuperset(of: known), "the scanner no longer sees \(known.subtracting(declaredAnywhere).sorted())")
        let violations = Self.findings(in: files)
        XCTAssertEqual(violations, [], "a collection interpolated whole prints its debug literal; render its elements:\n" + violations.joined(separator: "\n"))
    }
}
