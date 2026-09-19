import XCTest
import VectorTestKit

/// Grammar (spec §5.4 as amended): `<family>(.<op>)+(.<param>=<value>)*(.<variant>)?`, lowercase, `[a-z0-9_=.]` only.
final class CaseIDTests: XCTestCase {
    func testAcceptsTheTwoM0IDs() throws {
        XCTAssertEqual(try CaseID(validating: "probe.gpu_copy.bytes=134217728").family, "probe")
        XCTAssertEqual(try CaseID(validating: "distance.l2.batch.n=10000.d=768").family, "distance")
    }

    func testAcceptsVariantAfterParams() throws {
        XCTAssertNoThrow(try CaseID(validating: "ivf.list_search.n=4096.k=33.isolated"))
        XCTAssertNoThrow(try CaseID(validating: "probe.gpu_copy.bytes=1.k"))   // single-letter variant is legal
        XCTAssertNoThrow(try CaseID(validating: "distance.l2.batch"))            // several bare op segments, no params
    }

    func testRejects() {
        // "probe.x\n": ICU `$` matches before a final line terminator, so the pattern must anchor with `\z`.
        for bad in ["Probe.gpu_copy", "probe", "probe.gpu copy", "probe.gpu_copy.seed=42x!", "probe..x", ".probe.x", "probe.x.", "probe.x\n"] {
            XCTAssertThrowsError(try CaseID(validating: bad), bad)
        }
    }

    /// Checked through the interpolation the CLI applies to a caught error; without a description it is `malformed("…")`.
    func testMalformedDescriptionQuotesTheIDAndStatesTheGrammar() {
        let message = "\(CaseIDError.malformed("Probe.gpu copy") as any Error)"
        XCTAssertFalse(message.hasPrefix("malformed("), message)
        for piece in ["'Probe.gpu copy'", "<family>.<op>", "lowercase"] {
            XCTAssertTrue(message.contains(piece), "'\(piece)' missing from: \(message)")
        }
    }

    func testDuplicatesDetected() throws {
        let a = try CaseID(validating: "a.b"), b = try CaseID(validating: "a.c")
        XCTAssertEqual(CaseRegistryCheck.duplicates([a, b, a]), [a])
        XCTAssertEqual(CaseRegistryCheck.duplicates([a, b]), [])
    }
}
