import XCTest
@testable import VectorAccelerate

/// Guards the hand-maintained Swift/MSL parameter-struct layout parity that the kernels rely
/// on when copying configuration structs into constant buffers via `setBytes`.
final class StructLayoutTests: XCTestCase {

    /// `AttentionSimilarityParameters` must stay byte-compatible with the MSL `AttentionParams`
    /// struct: 8 × uint32 (32) + float (4) + uint8 (1) + uint8[3] padding (3) = 40 bytes.
    /// The padding is three explicit `UInt8` fields rather than a tuple precisely so this
    /// layout is deterministic.
    func testAttentionSimilarityParametersMatchesMSLLayout() {
        XCTAssertEqual(MemoryLayout<AttentionSimilarityParameters>.size, 40)
        XCTAssertEqual(MemoryLayout<AttentionSimilarityParameters>.stride, 40)
    }
}
