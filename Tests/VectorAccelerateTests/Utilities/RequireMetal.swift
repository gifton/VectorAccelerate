import XCTest
import Metal

/// Single skip point for Metal-dependent tests (spec §3). Adopt file by file; the legacy `XCTSkip` sites migrate
/// opportunistically, not in SP0.
func requireMetalDevice(file: StaticString = #filePath, line: UInt = #line) throws -> any MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal device not available", file: file, line: line) }
    return device
}
