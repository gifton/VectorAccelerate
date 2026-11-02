//
//  PerfGateCoverageTests.swift
//  VectorAccelerateTests
//
//  `PerfGate` says wall-clock assertions run only under VECTORACCELERATE_STRICT_PERF, because shared
//  virtualized runners fail them nondeterministically. Sites that skipped the gate kept failing CI
//  (`MatrixEngineTests.testOperationPerformanceComparison`: 1.06 s against a 1.0 s limit), so this
//  scans the test sources to keep every such assertion gated.
//
//  Recognised: `XCTAssert{LessThan,GreaterThan}[OrEqual]` whose first argument is a variable or
//  property named like a measurement and whose second is a nonzero numeric literal. Not recognised:
//  a threshold held in a variable, a computed first argument other than a `name / literal` division,
//  and `XCTAssertTrue(elapsed < limit)`. Pinned by `testScannerFixtures`. "Gated" means the assertion
//  is inside an `if PerfGate.strict {` block.
//

import Foundation
import XCTest

final class PerfGateCoverageTests: XCTestCase {

    private static let testsRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Hardening/
        .deletingLastPathComponent()  // VectorAccelerateTests/

    private static let assertion = try! NSRegularExpression(  // swiftlint:disable:this force_try
        pattern: #"XCTAssert(?:LessThan|GreaterThan)(?:OrEqual)?\(\s*"#
            + #"[\w.]*(?:[tT]ime|[eE]lapsed|[dD]uration|[lL]atency|[sS]peedup|[tT]hroughput|qps|QPS)\w*"#
            + #"(?:\s*/\s*[\d.]+)?\s*,\s*(?!0(?:\.0*)?\s*[,)])[\d.]+"#
    )

    /// Line numbers of timing assertions not inside an `if PerfGate.strict {` block.
    private static func ungatedLines(in source: String) -> [Int] {
        var ungated: [Int] = []
        var depth = 0
        var gateDepths: [Int] = []

        for (index, line) in source.components(separatedBy: "\n").enumerated() {
            let range = NSRange(line.startIndex..., in: line)
            let opensGate = line.contains("if PerfGate.strict") && line.contains("{")
            if assertion.firstMatch(in: line, range: range) != nil, gateDepths.isEmpty, !opensGate {
                ungated.append(index + 1)
            }
            for character in line {
                if character == "{" {
                    depth += 1
                } else if character == "}" {
                    depth -= 1
                    if gateDepths.last == depth + 1 { gateDepths.removeLast() }
                }
            }
            if opensGate { gateDepths.append(depth) }
        }
        return ungated
    }

    func testScannerFixtures() {
        let flagged = [
            "XCTAssertLessThan(elapsed, 1.0)",
            "XCTAssertLessThan(elapsed / 5.0, 1.0, \"reasonable\")",
            "XCTAssertGreaterThan(speedup, 1.0, \"GPU faster\")",
            "XCTAssertGreaterThan(quantizeThroughput, 50, \"rate\")",
            "XCTAssertLessThanOrEqual(result.totalTime, 0.5)",
        ]
        for line in flagged {
            XCTAssertEqual(Self.ungatedLines(in: line), [1], line)
        }

        let ignored = [
            "XCTAssertGreaterThan(result.computeTime, 0)",
            "XCTAssertGreaterThanOrEqual(timing.updateTime, 0.0)",
            "XCTAssertLessThan(distance, 1.0)",
            "XCTAssertLessThan(elapsed, limit)",
            "if PerfGate.strict { XCTAssertLessThan(elapsed, 1.0) }",
            "if PerfGate.strict {\n    XCTAssertLessThan(elapsed, 1.0)\n}",
            "if PerfGate.strict {\n    if flag {\n        XCTAssertLessThan(elapsed, 1.0)\n    }\n}",
        ]
        for source in ignored {
            XCTAssertEqual(Self.ungatedLines(in: source), [], source)
        }

        // The gate ends with its block.
        XCTAssertEqual(
            Self.ungatedLines(in: "if PerfGate.strict {\n    work()\n}\nXCTAssertLessThan(elapsed, 1.0)"),
            [4]
        )
        // An unrelated block does not count as a gate.
        XCTAssertEqual(Self.ungatedLines(in: "if flag {\n    XCTAssertLessThan(elapsed, 1.0)\n}"), [2])
    }

    func testEveryTimingAssertionIsGated() throws {
        let files = try XCTUnwrap(
            FileManager.default.enumerator(at: Self.testsRoot, includingPropertiesForKeys: nil)
        )
        var scanned = 0
        var violations: [String] = []
        for case let url as URL in files where url.pathExtension == "swift" {
            guard url.lastPathComponent != "PerfGateCoverageTests.swift" else { continue }
            scanned += 1
            let source = try String(contentsOf: url, encoding: .utf8)
            violations += Self.ungatedLines(in: source).map { "\(url.lastPathComponent):\($0)" }
        }
        XCTAssertGreaterThan(scanned, 100, "Scanner found too few test files; the root path is wrong")
        XCTAssertEqual(violations, [], "Wrap these in `if PerfGate.strict { ... }`")
    }
}
