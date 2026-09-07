//
//  PreambleParityTests.swift
//  VectorAccelerateTests
//
//  Phase-1.2 hardening audit: `KernelContext.runtimeCompilePreamble` replaces the stripped
//  `#include "Metal4Common.h"` in the runtime combined-source compile and is documented as
//  "MUST STAY NUMERICALLY IDENTICAL" to the header. Before this audit that identity was
//  enforced only by a comment; these guards enforce it mechanically:
//    * every constant the preamble redefines is numerically identical to the header's value,
//    * every VA_* symbol a compiled file actually uses is provided by the preamble, by the
//      compiler's textual replacements, or by a definition local to that same file — because
//      one unresolved symbol fails the ONE combined compile and strands all ~200 kernels.
//

import XCTest
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class PreambleParityTests: XCTestCase {

    private static let shadersDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Hardening/
        .deletingLastPathComponent()   // VectorAccelerateTests/
        .deletingLastPathComponent()   // Tests/
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders", isDirectory: true)

    private func requireHeader() throws -> String {
        let url = Self.shadersDir.appendingPathComponent("Metal4Common.h")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("repo shader sources not present (running outside the repo)")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// name → literal token, from `constant float|uint NAME = VALUE;` and `#define NAME VALUE`.
    private static func constants(in source: String) throws -> [String: String] {
        var out: [String: String] = [:]
        for pattern in [#"constant\s+(?:float|uint)\s+(VA_[A-Z0-9_]+)\s*=\s*([^;]+);"#,
                        #"#define\s+(VA_[A-Z0-9_]+|EPSILON)[ \t]+(\S+)"#] {
            let regex = try NSRegularExpression(pattern: pattern)
            for match in regex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                let name = String(source[Range(match.range(at: 1), in: source)!])
                let token = String(source[Range(match.range(at: 2), in: source)!])
                    .trimmingCharacters(in: .whitespaces)
                out[name] = token
            }
        }
        return out
    }

    /// Parse a MSL numeric literal token (decimal, hex float `0x1p-126f`, hex int, INFINITY).
    private static func numeric(_ token: String) -> Double? {
        var t = token.trimmingCharacters(in: .whitespaces)
        if t.hasSuffix("f") || t.hasSuffix("F") || t.hasSuffix("h") { t.removeLast() }
        if t == "INFINITY" { return .infinity }
        if t.lowercased().hasPrefix("0x") {
            if t.lowercased().contains("p") { return Double(t) }             // hex float literal
            return UInt64(t.dropFirst(2), radix: 16).map(Double.init)        // hex integer
        }
        return Double(t)
    }

    /// Every constant defined in BOTH the preamble and Metal4Common.h must agree numerically.
    /// (VA_INVALID_INDEX is expressed via textual replacement, not #define; VA_EPSILON is a
    /// real preamble #define since the VA3-012 fix removed the EPSILON token rewriting.)
    func testPreambleConstantsMatchHeader() throws {
        let headerConsts = try Self.constants(in: requireHeader())
        var preambleConsts = try Self.constants(in: KernelContext.runtimeCompilePreamble)
        preambleConsts["VA_INVALID_INDEX"] = "0xFFFFFFFF"          // replacement: literal substitution
        XCTAssertNotNil(preambleConsts["VA_EPSILON"],
            "preamble must define VA_EPSILON directly (no token rewriting since VA3-012)")

        XCTAssertFalse(headerConsts.isEmpty, "failed to parse any constants out of Metal4Common.h")
        var mismatches: [String] = []
        for (name, headerToken) in headerConsts.sorted(by: { $0.key < $1.key }) {
            guard let preambleToken = preambleConsts[name] else { continue }   // header-only: fine
            guard let h = Self.numeric(headerToken), let p = Self.numeric(preambleToken) else {
                mismatches.append("\(name): unparseable (header=\(headerToken) preamble=\(preambleToken))")
                continue
            }
            // Exact identity — these are compile-time constants, not computed values.
            if h != p { mismatches.append("\(name): header=\(headerToken) preamble=\(preambleToken)") }
        }
        XCTAssertTrue(mismatches.isEmpty,
            "preamble/header numeric drift:\n" + mismatches.joined(separator: "\n"))
    }

    /// The cosine overflow/underflow rescue block (`#ifndef VA_COSINE_RESCUE_DEFINED` …
    /// `#endif // VA_COSINE_RESCUE_DEFINED`) exists twice by necessity — in `Metal4Common.h`
    /// (per-file metallib builds) and in `KernelContext.runtimeCompilePreamble` (the combined
    /// runtime compile, which strips the header include). The two copies MUST stay byte-identical
    /// modulo leading indentation, or the two libraries silently diverge numerically.
    func testCosineRescueBlockIdentical() throws {
        func rescueBlock(in source: String, origin: String) throws -> [String] {
            guard let start = source.range(of: "#ifndef VA_COSINE_RESCUE_DEFINED"),
                  let end = source.range(of: "#endif // VA_COSINE_RESCUE_DEFINED") else {
                XCTFail("\(origin): VA_COSINE_RESCUE_DEFINED block not found"); return []
            }
            return String(source[start.lowerBound..<end.upperBound])
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
        }
        let header = try requireHeader()
        let headerBlock = try rescueBlock(in: header, origin: "Metal4Common.h")
        let preambleBlock = try rescueBlock(in: KernelContext.runtimeCompilePreamble, origin: "runtimeCompilePreamble")
        XCTAssertEqual(headerBlock, preambleBlock,
            "cosine rescue block drifted between Metal4Common.h and the runtime preamble")
    }

    /// Every `VA_*` constant a compiled `.metal` file uses (outside comments and its own
    /// definitions) must be covered by the preamble or the compiler's textual replacements.
    /// An uncovered symbol = a compile error in the ONE combined source = zero kernels at runtime.
    func testCompiledFilesUseOnlyCoveredSymbols() throws {
        _ = try requireHeader()
        let provided = Set(try Self.constants(in: KernelContext.runtimeCompilePreamble).keys)
            .union(["VA_INVALID_INDEX",              // textual replacement, not a #define
                    "VA_ATOMIC_TYPES_DEFINED"])      // preamble's own include guard
        let useRegex = try NSRegularExpression(pattern: #"\bVA_[A-Z][A-Z0-9_]*\b"#)

        var uncovered: Set<String> = []
        for base in KernelContext.runtimeCompileShaderFiles {
            let url = Self.shadersDir.appendingPathComponent("\(base).metal")
            let raw = try String(contentsOf: url, encoding: .utf8)
            // Strip line comments so prose mentions of constants don't count as uses.
            let source = raw.split(separator: "\n", omittingEmptySubsequences: false)
                .map { line -> Substring in
                    if let idx = line.range(of: "//") { return line[line.startIndex..<idx.lowerBound] }
                    return line
                }
                .joined(separator: "\n")
            // Symbols this file defines for itself survive the include-strip. Include bare
            // `#define VA_X` include-guard macros (no value), which `constants(in:)` skips.
            var localDefs = Set(try Self.constants(in: source).keys)
            let guardRegex = try NSRegularExpression(pattern: #"#(?:define|ifndef|ifdef)\s+(VA_[A-Z0-9_]+)"#)
            for match in guardRegex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                localDefs.insert(String(source[Range(match.range(at: 1), in: source)!]))
            }
            let range = NSRange(source.startIndex..., in: source)
            for match in useRegex.matches(in: source, range: range) {
                let sym = String(source[Range(match.range, in: source)!])
                if !provided.contains(sym) && !localDefs.contains(sym) {
                    uncovered.insert("\(base).metal: \(sym)")
                }
            }
        }
        XCTAssertTrue(uncovered.isEmpty, """
            \(uncovered.count) VA_* use(s) not covered by the preamble, its replacements, or a local \
            definition — the combined runtime compile would fail and strand every kernel:
            \(uncovered.sorted().joined(separator: "\n"))
            """)
    }
}
