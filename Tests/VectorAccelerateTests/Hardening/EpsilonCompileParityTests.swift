//
//  EpsilonCompileParityTests.swift
//  VectorAccelerateTests
//
//  AUDIT-3 VA3-012: EPSILON dual-compile drift. `DistanceShaders.metal` shadows the header's
//  `VA_EPSILON = 1e-7f` with a file-local `#define VA_EPSILON 1e-8f`, and the runtime
//  combined compile (`KernelContext.makeLibraryFromBundleSources` — the ONLY shader path in
//  release builds) rewrote every `VA_EPSILON` token to `EPSILON`. That turned the file-local
//  shadow into `#define EPSILON 1e-8f` in the ONE combined translation unit, silently
//  redefining the preamble's 1e-7 for EVERY downstream file: LearnedDistance's normalize
//  gates, NeuralQuantization's computeScale floor, and StatisticsShaders' histogram
//  degenerate-range gate all ran with 1e-8 in the release build and 1e-7 in the debug
//  metallib. Same source, two numerical behaviors — the exact masking class this audit
//  series hunts.
//
//  Post-fix contract (single epsilon authority):
//    * `Metal4Common.h` owns `VA_EPSILON = 1e-7f`; the runtime preamble defines the SAME
//      symbol with the SAME value (PreambleParityTests enforces numeric identity).
//    * No compiled `.metal` file may define `VA_EPSILON` or `EPSILON` — a file needing a
//      different floor names its own constant (DistanceShaders' jaccard union floor became
//      `VA_JACCARD_UNION_EPSILON`, value 1e-8 preserved on BOTH build paths).
//    * KernelContext performs no EPSILON token surgery.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class EpsilonCompileParityTests: XCTestCase {

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private static let shadersDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Hardening/
        .deletingLastPathComponent()   // VectorAccelerateTests/
        .deletingLastPathComponent()   // Tests/
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders", isDirectory: true)

    // MARK: - Helpers

    /// Run `uniformHistogram` from the given library: `dataCount` values, `edges` bin edges,
    /// returns the bin counts.
    private func runUniformHistogram(
        library: any MTLLibrary, device: any MTLDevice,
        data: [Float], edges: [Float]
    ) throws -> [UInt32] {
        let numBins = edges.count - 1
        guard let function = library.makeFunction(name: "uniformHistogram") else {
            throw XCTSkip("uniformHistogram not present in library")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let dataBuf = device.makeBuffer(bytes: data, length: data.count * 4, options: .storageModeShared),
              let edgesBuf = device.makeBuffer(bytes: edges, length: edges.count * 4, options: .storageModeShared),
              let histBuf = device.makeBuffer(length: numBins * 4, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: data.count * 4)
        }
        memset(histBuf.contents(), 0, numBins * 4)
        var params = SIMD4<UInt32>(UInt32(data.count), UInt32(numBins), 1 /* includeOutliers */, 0)

        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw VectorError.invalidInput("encoder creation failed")
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(edgesBuf, offset: 0, index: 1)
        enc.setBuffer(histBuf, offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: data.count, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        XCTAssertNil(cmd.error, "GPU execution error: \(String(describing: cmd.error))")

        let ptr = histBuf.contents().bindMemory(to: UInt32.self, capacity: numBins)
        return (0..<numBins).map { ptr[$0] }
    }

    private func bothLibraries() throws -> (metallib: any MTLLibrary, runtime: any MTLLibrary, device: any MTLDevice) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle(), "resource bundle not found")
        guard let metallibURL = bundle.url(forResource: "debug", withExtension: "metallib")
                ?? bundle.url(forResource: "default", withExtension: "metallib") else {
            throw XCTSkip("no prebuilt metallib in the resource bundle")
        }
        let metallib = try device.makeLibrary(URL: metallibURL)
        let runtime = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        return (metallib, runtime, device)
    }

    // MARK: - Behavioral: the drift, demonstrated on a live victim gate

    /// `uniformHistogram`'s degenerate-range gate is `range <= VA_EPSILON`
    /// (StatisticsShaders.metal). A range of 5e-8 sits exactly in the drift window
    /// (1e-8, 1e-7]: the debug metallib (header VA_EPSILON = 1e-7) takes the degenerate
    /// branch and puts every valid value in bin 0, while the pre-fix combined build (EPSILON
    /// silently redefined to 1e-8 by DistanceShaders upstream) binned normally. Same kernel,
    /// same input, two different histograms — red pre-fix. The header value is the contract,
    /// so both paths must produce the degenerate result.
    func testHistogramDegenerateGateAgreesAcrossBuildPaths() throws {
        let (metallib, runtime, device) = try bothLibraries()
        // range = 5e-8; half the values land in bin 0, half in bin 1 when binned normally.
        let edges: [Float] = [0.0, 2.5e-8, 5e-8]
        let data: [Float] = Array(repeating: 1e-8, count: 8) + Array(repeating: 4e-8, count: 8)

        let fromMetallib = try runUniformHistogram(library: metallib, device: device, data: data, edges: edges)
        let fromRuntime = try runUniformHistogram(library: runtime, device: device, data: data, edges: edges)

        XCTAssertEqual(fromMetallib, fromRuntime,
            "dual-compile EPSILON drift: the same kernel disagrees between the debug metallib and the runtime combined build")
        XCTAssertEqual(fromMetallib, [16, 0],
            "header contract (VA_EPSILON = 1e-7): range 5e-8 is degenerate — all valid values in bin 0")
    }

    /// Control (must stay green before AND after): a clearly non-degenerate range bins
    /// normally and identically on both build paths.
    func testHistogramNormalRangeAgreesAcrossBuildPaths() throws {
        let (metallib, runtime, device) = try bothLibraries()
        let edges: [Float] = [0.0, 0.5, 1.0]
        let data: [Float] = Array(repeating: 0.25, count: 8) + Array(repeating: 0.75, count: 8)

        let fromMetallib = try runUniformHistogram(library: metallib, device: device, data: data, edges: edges)
        let fromRuntime = try runUniformHistogram(library: runtime, device: device, data: data, edges: edges)

        XCTAssertEqual(fromMetallib, [8, 8], "non-degenerate range must bin normally")
        XCTAssertEqual(fromRuntime, [8, 8], "non-degenerate range must bin normally")
    }

    // MARK: - Structural: single epsilon authority (closes the class)

    /// No compiled `.metal` file may define `VA_EPSILON` or `EPSILON` — one file-local shadow
    /// plus token rewriting is exactly how VA3-012 poisoned every downstream file in the
    /// combined TU. A file needing a different floor must name its own constant
    /// (`VA_JACCARD_UNION_EPSILON` is the precedent). The runtime preamble must provide
    /// `VA_EPSILON` itself — with no `EPSILON` alias and no token surgery, a missing preamble
    /// definition would fail the combined compile loudly, but this assertion catches it at
    /// test time with a readable message.
    func testSingleEpsilonAuthority() throws {
        let defineRegex = try NSRegularExpression(
            pattern: #"^\s*#define\s+(VA_EPSILON|EPSILON)\b|^\s*constant\s+float\s+(VA_EPSILON|EPSILON)\s*="#,
            options: [.anchorsMatchLines])

        var offenders: [String] = []
        for base in KernelContext.runtimeCompileShaderFiles {
            let url = Self.shadersDir.appendingPathComponent("\(base).metal")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("repo shader sources not present (running outside the repo)")
            }
            let source = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(source.startIndex..., in: source)
            if defineRegex.firstMatch(in: source, range: range) != nil {
                offenders.append("\(base).metal")
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            \(offenders.count) file(s) define VA_EPSILON/EPSILON — in the combined runtime TU a \
            file-scope (re)definition changes the value for every file after it (VA3-012): \
            \(offenders.joined(separator: ", "))
            """)

        XCTAssertTrue(KernelContext.runtimeCompilePreamble.contains("#define VA_EPSILON 1e-7f"),
            "the runtime preamble must define VA_EPSILON itself (identical to Metal4Common.h; PreambleParityTests guards the value)")
    }
}
