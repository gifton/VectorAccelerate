//
//  LostStoreShapeGuardTests.swift
//  VectorAccelerateTests
//
//  Lexical guard for the Swift optimizer miscompile VectorCore filed as VC-C001 (T15 lost-store class,
//  swiftlang/swift#92562; see CompilerLostStoreCanaryTests for the shape and the evidence). The compiler
//  forwards a known value across a mutable-pointer borrow whose closure hands the pointer to an out-of-line
//  callee `inout`; whether a given site is armed depends on inlining, so the guard flags the lexical shape
//  wherever it appears, armed or not. It is a port of the rules of VectorCore's
//  `Scripts/ci/tests/test_lost_store_reads.py` that VectorAccelerate needs, over `Sources/**/*.swift` and
//  `Tests/**/*.swift`, with comments and string literals blanked:
//
//  1. No parameter of type `inout Unsafe[Mutable][Raw][Buffer]Pointer` anywhere. VectorAccelerate passes
//     buffers by value; the only writer that takes one `inout` is the canary's, allowlisted below.
//  2. Buffer form: a local `var x = [ ... ]` or `var x: T = [ ... ]` whose whole initializer is an array
//     literal, then `x.withUnsafeMutableBufferPointer` (trailing closure or function argument), then, after
//     the call and inside the scope that declares `x`, a direct read: `x[...]` that is not a store, `x.first`,
//     `x.last`, `x.withUnsafeBufferPointer`, `x.withUnsafeBytes`, or a whole-array comparison.
//  3. Pointer form: `withUnsafeMutablePointer(to: &x)`, then, after the call and inside the scope that
//     declares `x`, any use of `x` that is not a plain assignment target, an `&x` argument or a label `x:`.
//
//  Accepted on purpose: non-literal initializers (`[T](repeating:count:)`, which is not forwardable
//  today), reading through the buffer inside the closure and returning it (VectorCore's W2 shape), and
//  by-value writers. Limits: lexical, not dataflow; stores between the declaration and the call are not
//  tracked; aliases are not followed; `#if` branches are all scanned; the captured-variable form VectorCore
//  flags by class is not enforced here (20 sites, listed in docs/migrations/VECTORCORE-0.4.0-CANDIDATE.md,
//  pinned by the canary file's accumulator control).
//

import XCTest

final class LostStoreShapeGuardTests: XCTestCase {

    // MARK: - Tree

    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Hardening
        .deletingLastPathComponent()   // VectorAccelerateTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // repo root

    /// `path::function::name` keys of reviewed sites that keep the shape, with the reason.
    private static let allowed: [String: String] = [
        "Tests/VectorAccelerateTests/Hardening/CompilerLostStoreCanaryTests.swift::lostStoreCanaryWriteOneInout::inout-parameter":
            "the canary's deliberate inout writer (CompilerLostStoreCanaryTests)",
        "Tests/VectorAccelerateTests/Hardening/CompilerLostStoreCanaryTests.swift::lostStoreCanaryObservedAfterBorrow::values":
            "the canary's deliberate miscompiled shape (CompilerLostStoreCanaryTests)",
        "Tests/VectorAccelerateTests/Hardening/CompilerLostStoreCanaryTests.swift::lostStoreControlByValueWriter::values":
            "the by-value writer control: VectorCore flags this shape by class because arming follows inlining; "
            + "it keeps its store (CompilerLostStoreCanaryTests.testByValueWriterControlKeepsTheStore)",
    ]

    private static func swiftFiles() throws -> [(path: String, source: String)] {
        var result: [(String, String)] = []
        for sub in ["Sources", "Tests"] {
            let dir = repoRoot.appendingPathComponent(sub)
            guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else { continue }
            for case let url as URL in e where url.pathExtension == "swift" {
                let rel = url.path.replacingOccurrences(of: repoRoot.path + "/", with: "")
                result.append((rel, try String(contentsOf: url, encoding: .utf8)))
            }
        }
        return result.sorted { $0.0 < $1.0 }
    }

    // MARK: - Scanner

    /// Replaces comments and string literals with spaces (newlines kept) so offsets and lines are preserved.
    static func mask(_ source: String) -> [Character] {
        var out = Array(source)
        let chars = out
        var i = 0
        func blank(_ from: Int, _ to: Int) { for k in from..<min(to, out.count) where out[k] != "\n" { out[k] = " " } }
        while i < chars.count {
            let c = chars[i]
            let next: Character? = i + 1 < chars.count ? chars[i + 1] : nil
            if c == "/" && next == "/" {
                var j = i
                while j < chars.count && chars[j] != "\n" { j += 1 }
                blank(i, j); i = j
            } else if c == "/" && next == "*" {
                var depth = 1; var j = i + 2
                while j < chars.count && depth > 0 {
                    if chars[j] == "/" && j + 1 < chars.count && chars[j + 1] == "*" { depth += 1; j += 2; continue }
                    if chars[j] == "*" && j + 1 < chars.count && chars[j + 1] == "/" { depth -= 1; j += 2; continue }
                    j += 1
                }
                blank(i, j); i = j
            } else if c == "\"" {
                let triple = i + 2 < chars.count && chars[i + 1] == "\"" && chars[i + 2] == "\""
                var j = i + (triple ? 3 : 1)
                while j < chars.count {
                    if chars[j] == "\\" { j += 2; continue }
                    if triple {
                        if j + 2 < chars.count && chars[j] == "\"" && chars[j + 1] == "\"" && chars[j + 2] == "\"" { j += 3; break }
                    } else if chars[j] == "\"" || chars[j] == "\n" { j += 1; break }
                    j += 1
                }
                blank(i + 1, j - 1 > i + 1 ? j - 1 : i + 1); i = j
            } else {
                i += 1
            }
        }
        return out
    }

    private static func line(of offset: Int, in text: [Character]) -> Int {
        text[0..<min(offset, text.count)].reduce(1) { $0 + ($1 == "\n" ? 1 : 0) }
    }

    /// Name of the nearest enclosing `func` before `offset` (or `<top>`).
    private static func enclosingFunction(_ text: String, before offset: Int) -> String {
        let head = String(text.prefix(offset))
        guard let m = try! NSRegularExpression(pattern: #"func\s+([A-Za-z_][A-Za-z0-9_]*)"#)
                .matches(in: head, range: NSRange(head.startIndex..., in: head)).last,
              let r = Range(m.range(at: 1), in: head) else { return "<top>" }
        return String(head[r])
    }

    /// Index just past a call whose receiver/name ends at `index`: `( ... )` and/or a trailing `{ ... }`.
    private static func callEnd(_ text: [Character], from index: Int) -> Int? {
        var i = index
        func skipSpace() { while i < text.count, text[i] == " " || text[i] == "\n" || text[i] == "\t" { i += 1 } }
        func matchBracket(_ open: Character, _ close: Character) -> Bool {
            guard i < text.count, text[i] == open else { return false }
            var depth = 0
            while i < text.count {
                if text[i] == open { depth += 1 } else if text[i] == close { depth -= 1; if depth == 0 { i += 1; return true } }
                i += 1
            }
            return false
        }
        skipSpace()
        let hadParens = matchBracket("(", ")")
        skipSpace()
        let hadBraces = matchBracket("{", "}")
        return (hadParens || hadBraces) ? i : nil
    }

    /// True when the `[` at `openBracket` starts an array literal that is the whole initializer: its matching
    /// `]` is followed by a newline, `;` or `}` (not by `(`, `.`, `+`, `as` or another operator).
    private static func isArrayLiteral(_ text: [Character], openBracket: Int) -> Bool {
        guard openBracket < text.count, text[openBracket] == "[" else { return false }
        var depth = 0; var i = openBracket
        while i < text.count {
            if text[i] == "[" { depth += 1 } else if text[i] == "]" { depth -= 1; if depth == 0 { break } }
            i += 1
        }
        guard i < text.count else { return false }
        i += 1
        while i < text.count, text[i] == " " || text[i] == "\t" { i += 1 }
        return i >= text.count || text[i] == "\n" || text[i] == ";" || text[i] == "}"
    }

    /// Index where the scope that is open at `offset` closes (first `}` that takes the depth below 0).
    private static func scopeEnd(_ text: [Character], from offset: Int) -> Int {
        var depth = 0; var i = offset
        while i < text.count {
            if text[i] == "{" { depth += 1 } else if text[i] == "}" { depth -= 1; if depth < 0 { return i } }
            i += 1
        }
        return text.count
    }

    static func scan(path: String, source: String) -> [String] {
        let masked = mask(source)
        let text = String(masked)
        var hits: [String] = []
        func ns(_ r: Range<String.Index>) -> Int { text.distance(from: text.startIndex, to: r.lowerBound) }

        // Rule 1: inout unsafe-pointer parameters.
        let inoutRe = try! NSRegularExpression(pattern: #"inout\s+Unsafe(?:Mutable)?(?:Raw)?(?:Buffer)?Pointer\b"#)
        for m in inoutRe.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let r = Range(m.range, in: text)!
            // The parameter belongs to the `func` whose signature it is in: the nearest `func` before it.
            let fn = enclosingFunction(text, before: ns(r) + 1)
            hits.append("\(path)::\(fn)::inout-parameter@\(line(of: ns(r), in: masked))")
        }

        // Rule 2: buffer form.
        let bufferRe = try! NSRegularExpression(pattern: #"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*withUnsafeMutableBufferPointer"#)
        for m in bufferRe.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let whole = Range(m.range, in: text)!, nameR = Range(m.range(at: 1), in: text)!
            let name = String(text[nameR]); let call = ns(whole)
            let matchLength = text.distance(from: whole.lowerBound, to: whole.upperBound)
            // Nearest preceding `var name` whose initializer starts with `[`, in the same function.
            let declRe = try! NSRegularExpression(pattern: #"\bvar\s+"# + NSRegularExpression.escapedPattern(for: name) + #"\s*(?::[^=\n]+)?=\s*\["#)
            let head = String(text.prefix(call))
            guard let decl = declRe.matches(in: head, range: NSRange(head.startIndex..., in: head)).last,
                  let declR = Range(decl.range, in: head) else { continue }
            let declOffset = head.distance(from: head.startIndex, to: declR.lowerBound)
            let declEnd = head.distance(from: head.startIndex, to: declR.upperBound)   // just past the `[`
            if enclosingFunction(text, before: declOffset) != enclosingFunction(text, before: call) { continue }
            // The whole initializer must be an array literal: `[Float](repeating:)` and `[1, 2].map` are not.
            guard isArrayLiteral(masked, openBracket: declEnd - 1) else { continue }
            guard let end = callEnd(masked, from: call + matchLength) else { continue }
            let scopeClose = scopeEnd(masked, from: declEnd)
            guard end < scopeClose else { continue }
            let region = String(masked[end..<scopeClose])
            let n = NSRegularExpression.escapedPattern(for: name)
            let readRe = try! NSRegularExpression(pattern:
                #"\b"# + n + #"\s*\[[^\]]*\](?!\s*=[^=])|\b"# + n + #"\.(?:first|last|withUnsafeBufferPointer|withUnsafeBytes)\b|\b"# + n + #"\s*[!=]=|[!=]=\s*"# + n + #"\b"#)
            if let read = readRe.firstMatch(in: region, range: NSRange(region.startIndex..., in: region)) {
                let readOffset = end + region.distance(from: region.startIndex, to: Range(read.range, in: region)!.lowerBound)
                hits.append("\(path)::\(enclosingFunction(text, before: call))::\(name)@\(line(of: readOffset, in: masked))")
            }
        }

        // Rule 3: pointer form.
        let pointerRe = try! NSRegularExpression(pattern: #"withUnsafeMutablePointer\(\s*to:\s*&\s*([A-Za-z_][A-Za-z0-9_.]*)"#)
        for m in pointerRe.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let whole = Range(m.range, in: text)!, nameR = Range(m.range(at: 1), in: text)!
            let name = String(text[nameR]); let call = ns(whole)
            // Call end: the `withUnsafeMutablePointer(` parens (and a trailing closure), starting at the `(`.
            let parenIndex = call + "withUnsafeMutablePointer".count
            guard let end = callEnd(masked, from: parenIndex) else { continue }
            let scopeClose = scopeEnd(masked, from: end)
            guard end < scopeClose else { continue }
            let region = String(masked[end..<scopeClose])
            let n = NSRegularExpression.escapedPattern(for: name)
            let useRe = try! NSRegularExpression(pattern: #"(?<![A-Za-z0-9_.&])"# + n + #"\b(?!\s*=[^=])(?!\s*:)"#)
            if let use = useRe.firstMatch(in: region, range: NSRange(region.startIndex..., in: region)) {
                let useOffset = end + region.distance(from: region.startIndex, to: Range(use.range, in: region)!.lowerBound)
                hits.append("\(path)::\(enclosingFunction(text, before: call))::\(name)@\(line(of: useOffset, in: masked))")
            }
        }
        return hits
    }

    private static func key(_ hit: String) -> String { String(hit[..<hit.lastIndex(of: "@")!]) }

    // MARK: - Fixture controls (the guard must be able to see the shapes it exists for)

    func testCanaryShapesAreFlagged() {
        let positive = """
        @inline(never) func w(_ out: inout UnsafeMutableBufferPointer<Float>) { out[0] = 1 }
        func observed() -> Float {
            var values: [Float] = [2]
            values.withUnsafeMutableBufferPointer(w)
            return values[0]
        }
        func trailing() -> Float {
            var v = [2, 3]
            v.withUnsafeMutableBufferPointer { b in w(&b) }
            return v.first ?? 0
        }
        func pointer() -> Int {
            var info = 5
            withUnsafeMutablePointer(to: &info) { p in p.pointee = 1 }
            return info
        }
        """
        let hits = Self.scan(path: "F.swift", source: positive).map(Self.key)
        XCTAssertEqual(hits, ["F.swift::w::inout-parameter", "F.swift::observed::values", "F.swift::trailing::v", "F.swift::pointer::info"])
    }

    func testAcceptedShapesAreNotFlagged() {
        let negative = """
        @inline(never) func w(_ out: UnsafeMutableBufferPointer<Float>) { out[0] = 1 }
        func byValueWriter() -> Float {
            var values: [Float] = [2]
            values.withUnsafeMutableBufferPointer { w($0) }
            return values[0]
        }
        func repeatingInitializer() -> Float {
            var result = [Float](repeating: 0, count: 4)
            result.withUnsafeMutableBufferPointer { $0[0] = 1 }
            return result[0]
        }
        func readInside() -> Float {
            var values: [Float] = [2]
            return values.withUnsafeMutableBufferPointer { b in b[0] = 1; return b[0] }
        }
        func storeOnly() {
            var values: [Float] = [2]
            values.withUnsafeMutableBufferPointer { $0[0] = 1 }
            values[0] = 3
        }
        func pointerReadInside() -> Float {
            var value: Float = 2
            return withUnsafeMutablePointer(to: &value) { p -> Float in p.pointee = 1; return p.pointee }
        }
        // var values: [Float] = [2] in a comment, and "values[0]" in a string, are not code.
        """
        let byValue = Self.scan(path: "F.swift", source: negative).filter { !$0.contains("::byValueWriter::") }
        XCTAssertEqual(byValue, [], "accepted shapes must not be flagged")
        // The by-value adapter still reads a literal-initialized local after the borrow; VectorCore flags that
        // shape too because arming follows inlining. VectorAccelerate has no such site; it is reported here so
        // a future one is reviewed rather than silently accepted.
        XCTAssertEqual(Self.scan(path: "F.swift", source: negative).map(Self.key), ["F.swift::byValueWriter::values"])
    }

    // MARK: - Real tree

    func testNoUnreviewedLostStoreShapeInSourcesOrTests() throws {
        let files = try Self.swiftFiles()
        XCTAssertGreaterThan(files.count, 200, "the scan must see the whole tree, not a stub")
        let hits = files.flatMap { Self.scan(path: $0.path, source: $0.source) }
        let keys = Set(hits.map(Self.key))
        let unreviewed = hits.filter { !Self.allowed.keys.contains(Self.key($0)) }
        XCTAssertEqual(unreviewed, [], "lost-store read shape (VC-C001, swiftlang/swift#92562); review, rewrite to the W2 "
                       + "inside-borrow read or a by-value writer, or allowlist with a reason")
        let stale = Self.allowed.keys.filter { !keys.contains($0) }.sorted()
        XCTAssertEqual(stale, [], "stale allowlist entries")
    }

    func testScanIsNotVacuous() throws {
        // The canary file must be seen, and its deliberate shape must be the one the allowlist names.
        let files = try Self.swiftFiles()
        let canary = try XCTUnwrap(files.first { $0.path.hasSuffix("CompilerLostStoreCanaryTests.swift") })
        let hits = Self.scan(path: canary.path, source: canary.source).map(Self.key)
        XCTAssertTrue(hits.contains("Tests/VectorAccelerateTests/Hardening/CompilerLostStoreCanaryTests.swift::lostStoreCanaryObservedAfterBorrow::values"), "\(hits)")
    }
}
