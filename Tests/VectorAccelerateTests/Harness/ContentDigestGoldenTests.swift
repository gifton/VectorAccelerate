import XCTest
import VectorTestKit

/// Pins the canonical byte encoding (spec §4.4). The golden was computed independently in Python:
/// LE u64 count + LE element bytes per array; LE Int64 for `.int`; Double bit pattern for `.double`;
/// LE u64 byte-length + UTF-8 for `.string`; parts hashed in order.
final class ContentDigestGoldenTests: XCTestCase {
    func testGoldenDigest() {
        let parts: [DigestPart] = [
            .floats([1.0, 2.5, -0.5]),
            .uint32s([7, 8]),
            .int(42),
            .double(0.25),
            .string("vector"),
        ]
        XCTAssertEqual(ContentDigest.sha256(parts), "82ea485579eebefdcbe8edff10d654835d0639807b801a2bf527b8b40c9070cb")
    }

    func testOrderMatters() {
        XCTAssertNotEqual(ContentDigest.sha256([.int(1), .int(2)]), ContentDigest.sha256([.int(2), .int(1)]))
    }

    func testEmptyArrayIsNotEmptyDigest() {
        XCTAssertNotEqual(ContentDigest.sha256([.floats([])]), ContentDigest.sha256([]))
    }
}
