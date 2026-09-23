import XCTest

/// Rendering rule: interpolating an array or dictionary whole (`"reasons: \(preflight.reasons)"`) prints Swift's debug
/// literal — `["drift 6.5% exceeds 5.0%", "thermal state end=fair (not nominal)"]` — which is not a message. `CompareCommand`
/// (Task 16 review) and `RunCommand` (Task 17) each did it once; this closes the class. In `Sources/VectorTestKit` and
/// `Sources/VectorAccelerateBench`, no string-interpolation segment may consist of a bare or dotted reference to a name
/// declared with an explicit array or dictionary type. Render the elements instead (`.joined(separator: "; ")`).
///
/// Contract — what the scanner recognises; anything outside this list is not seen:
/// - a collection name is the `name` in `let name: [` or `var name: [` (stored, computed and local), or in a parameter
///   `[label ]name: [` of a `func` or `init` parameter clause (the clause is the parenthesised text after the name, balanced,
///   so multi-line clauses and defaults with parentheses are read); the type must START with `[`, so `[T]`, `[[T]]`,
///   `[K: V]` and `[T]?` count, while `Set<T>`, `Array<T>`, `Dictionary<K, V>`, `inout [T]` and every inferred type do not;
/// - a violation is a segment `\(chain)` whose whole content is an identifier chain — `reasons`, `preflight.reasons`,
///   `self.a.b` — with a collection name as its LAST component. A chain that ends in a call, a subscript or another member
///   (`.joined(…)`, `.map(…)`, `[0]`, `.count`, `.first`) is not one, nor is an inline literal (`\([a, b])`);
/// - a bare chain resolves against the names declared in the SAME file (locals, parameters, properties of types declared
///   there); a dotted chain's last component resolves against every `let`/`var` collection name in all scanned files, since a
///   member may be declared anywhere. Within a file two declarations sharing a name are not told apart, so a scalar that
///   shares its name with a collection declared in the same file is flagged too (rename one of them);
/// - closure parameters, subscript parameters, enum associated values and tuple-type elements are not seen;
/// - `//` comments are removed before matching; `/* */` blocks and string literals are scanned as code, and a `//` inside a
///   string literal truncates that line.
/// `testScannerFixtures` pins each shape named above; `testNoCollectionIsInterpolatedWhole` runs the scan on the real sources.
final class CollectionInterpolationLintTests: XCTestCase {
    /// Group 1: the name, in `let name: [` / `var name: [`.
    private static let declared = try! NSRegularExpression(pattern: #"\b(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\["#)
    /// Opens a `func` or `init` parameter clause; the clause ends at the parenthesis that balances this one.
    private static let clauseHead = try! NSRegularExpression(pattern: #"\b(?:func\s+[A-Za-z_][A-Za-z0-9_]*|init[?!]?)\s*(?:<[^>]*>)?\s*\("#)
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

    /// Pins the scanner on fixture sources, so a pattern change that stops seeing a violation, or starts seeing a false one,
    /// fails here.
    func testScannerFixtures() {
        let cases: [(sources: [String], names: Set<String>, findings: [String])] = [
            // Recognised declaration shapes, bare use in the same file.
            (["let reasons: [String] = []\nprint(\"reasons: \\(reasons)\")"], ["reasons"], ["f0.swift:2: \\(reasons)"]),
            (["var flags: [String: String] = [:]\n_ = \"\\(flags)\""], ["flags"], ["f0.swift:2: \\(flags)"]),
            (["let raw: [[Double]] = []\nvar extra: [Int]? = nil\n_ = \"\\(raw) \\(extra)\""], ["raw", "extra"],
             ["f0.swift:3: \\(raw)", "f0.swift:3: \\(extra)"]),
            (["func f(_ items: [Int], with list: [Int], n: Int) -> String { \"\\(items) \\(list) \\(n)\" }"], ["items", "list"],
             ["f0.swift:1: \\(items)", "f0.swift:1: \\(list)"]),
            (["struct S {\n    init(\n        a: [Int] = Array(repeating: 0, count: 2),\n        b: [String: Int]\n    ) { _ = \"\\(a) \\(b)\" }\n}"],
             ["a", "b"], ["f0.swift:5: \\(a)", "f0.swift:5: \\(b)"]),
            (["let ids: [Int] = []\n_ = \"\\( ids )\""], ["ids"], ["f0.swift:2: \\(ids)"]),
            // Dotted use: the member may be declared in another file. The second is the pre-fix `RunCommand` line verbatim.
            (["struct S { var deltas: [Int] = []; var text: String { \"\\(self.deltas)\" } }"], ["deltas"], ["f0.swift:1: \\(self.deltas)"]),
            (["public struct PreflightRecord { public var reasons: [String] }",
              "print(\"key: \\(metadata.environmentKey)  quiet: \\(preflight.quiet)\" + (preflight.reasons.isEmpty ? \"\" : \"  reasons: \\(preflight.reasons)\"))"],
             ["reasons"], ["f1.swift:1: \\(preflight.reasons)"]),
            // Rendering the elements, or any other operation on the name, is not a violation.
            (["var notes: [String] = []\n_ = \"\\(notes.joined(separator: \"; \")) \\(notes.count) \\(notes[0]) \\(notes.first) \\(notes.map(\\.count)) \\(notes.isEmpty)\""],
             ["notes"], []),
            // A bare name resolves in its own file: a `String` named `raw` in one file is not the `[[Double]]` of another …
            (["struct Series { var raw: [[Double]] }", "func parse(_ raw: String) -> String { \"bad: \\(raw)\" }"], ["raw"], []),
            // … while a dotted `raw` still is.
            (["struct Series { var raw: [[Double]] }", "func show(_ s: Series) -> String { \"\\(s.raw)\" }"], ["raw"], ["f1.swift:1: \\(s.raw)"]),
            // Within one file, two declarations sharing a name are not told apart: both uses are flagged.
            (["func a(_ values: [Int]) -> String { \"\\(values)\" }\nfunc b(_ values: Int) -> String { \"\\(values)\" }"], ["values"],
             ["f0.swift:1: \\(values)", "f0.swift:2: \\(values)"]),
            // Not recognised: generic spellings, inferred types, inline literals, call results, closure parameters, tuple
            // elements, `inout`, subscript parameters, enum associated values.
            (["let s: Set<String> = []\nlet g: Array<Int> = []\nlet inferred = [1, 2]\nlet t: (xs: [Int], n: Int) = ([], 0)\n"
              + "func k(_ ys: inout [Int]) -> String { \"\\(ys)\" }\nlet c: ([Int]) -> String = { (zs: [Int]) in \"\\(zs)\" }\n"
              + "_ = \"\\(s) \\(g) \\(inferred) \\([1, 2]) \\(f()) \\(t.xs)\"",
              "struct Table { subscript(keys: [String]) -> String { \"\\(keys)\" } }\nenum E { case list(items: [Int]) }"], [], []),
            // Comments are not scanned, for declarations or for uses.
            (["// let hidden: [Int] = []\n/// \\(hidden)\nlet shown: [Int] = [] // trailing \\(shown)"], ["shown"], []),
            (["let none: Int = 0\n_ = \"\\(none)\""], [], [])
        ]
        for (sources, names, findings) in cases {
            let scan = Self.scan(sources)
            XCTAssertEqual(scan.names, names, "collection names in: \(sources)")
            XCTAssertEqual(scan.findings, findings, "findings in: \(sources)")
        }
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
