import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class VectorAlignmentTests: XCTestCase {
    /// -Wcast-align deliberately ignores explicit C++ reinterpret_cast. Rewrite
    /// only the cast spelling in a temporary copy, retaining type and operand,
    /// so the compiler checks those conversions as well as C-style casts.
    private func diagnosticSource(_ source: String) throws -> String {
        var result = source
        let regex = try NSRegularExpression(pattern: #"\breinterpret_cast\s*<([^>\n]+)>\s*\("#)
        while let match = regex.firstMatch(in: result, range: NSRange(result.startIndex..., in: result)) {
            let whole = try XCTUnwrap(Range(match.range, in: result))
            let type = String(result[Range(match.range(at: 1), in: result)!])
            let open = result.index(before: whole.upperBound)
            var end = whole.upperBound, depth = 1
            while end < result.endIndex && depth > 0 {
                if result[end] == "(" { depth += 1 }
                if result[end] == ")" { depth -= 1 }
                end = result.index(after: end)
            }
            guard depth == 0 else { throw NSError(domain: "Unbalanced shader cast", code: 1) }
            let operand = result[open..<end]
            result.replaceSubrange(whole.lowerBound..<end, with: "((\(type))\(operand))")
        }
        XCTAssertFalse(result.contains("reinterpret_cast"), "Every reinterpret cast must be checked")
        return result
    }

    /// Current Apple GPUs may execute a misaligned vector access correctly, so
    /// value parity alone cannot establish the MSL alignment contract.
    func testShaderCorpusDoesNotIncreasePointerAlignment() throws {
        #if os(macOS)
        let shaders = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders")
        let files = try FileManager.default.contentsOfDirectory(at: shaders, includingPropertiesForKeys: nil)
            .filter { ["metal", "h"].contains($0.pathExtension) }.sorted { $0.path < $1.path }
        XCTAssertFalse(files.isEmpty)
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("va-alignment-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            try diagnosticSource(source).write(to: temporary.appendingPathComponent(file.lastPathComponent), atomically: true, encoding: .utf8)
        }
        func compile(_ paths: [String]) throws -> (Int32, String) {
            let log = temporary.appendingPathComponent("diagnostics.log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let output = try FileHandle(forWritingTo: log)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            process.arguments = ["-sdk", "macosx", "metal", "-std=metal4.0", "-fsyntax-only", "-Werror=cast-align"] + paths
            process.standardOutput = output; process.standardError = output
            try process.run(); process.waitUntilExit(); try output.close()
            return (process.terminationStatus, try String(contentsOf: log, encoding: .utf8))
        }
        // Fail closed if the compiler flag or reinterpret-cast transformation
        // stops detecting an actual scalar-to-vector alignment increase.
        let canary = temporary.appendingPathComponent("alignment_canary.metal")
        try diagnosticSource("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void alignment_canary(device const float* p [[buffer(0)]], device float4* out [[buffer(1)]]) {
            out[0] = reinterpret_cast <device const float4*>(p + 1)[0];
        }
        """).write(to: canary, atomically: true, encoding: .utf8)
        let (canaryStatus, canaryDiagnostics) = try compile([canary.path])
        XCTAssertNotEqual(canaryStatus, 0, "The alignment canary must be rejected")
        XCTAssertTrue(canaryDiagnostics.contains("increases required alignment"), canaryDiagnostics)
        let (status, diagnostics) = try compile(files.filter { $0.pathExtension == "metal" }.map { temporary.appendingPathComponent($0.lastPathComponent).path })
        XCTAssertEqual(status, 0, diagnostics)
        #else
        throw XCTSkip("Offline Metal alignment diagnostics require the macOS toolchain")
        #endif
    }

    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func buffer<T: BitwiseCopyable>(_ data: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: data, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared))
    }

    private func run(_ name: String, _ device: any MTLDevice, _ library: any MTLLibrary,
                     _ buffers: [any MTLBuffer], grid: MTLSize, group: MTLSize = MTLSize(width: 1, height: 1, depth: 1),
                     threadgroupBytes: Int = 0) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        if threadgroupBytes > 0 { encoder.setThreadgroupMemoryLength(threadgroupBytes, index: 0) }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(name): \(String(describing: command.error))")
    }

    func testStridedL2AndDotRowsIncludeEveryAlignmentResidue() throws {
        try withLibraries { device, library, path in
            for d in [4, 6, 17, 19] {
                for stride in d...(d + 3) {
                    let rows = 4, outputStride = 5
                    var a = [Float](repeating: 999, count: rows * stride)
                    var b = a
                    for r in 0..<rows { for j in 0..<d {
                        a[r * stride + j] = Float(r + j % 3)
                        b[r * stride + j] = Float(2 * r - j % 5)
                    } }
                    for name in ["l2_distance_kernel", "dot_product_kernel"] {
                        let output = try buffer([Float](repeating: -12345, count: rows * outputStride), device)
                        let params = try buffer([UInt32(rows), UInt32(rows), UInt32(d), UInt32(stride), UInt32(stride), UInt32(outputStride), 0], device)
                        try run(name, device, library, [buffer(a, device), buffer(b, device), output, params], grid: MTLSize(width: rows, height: rows, depth: 1))
                        let result = output.contents().bindMemory(to: Float.self, capacity: rows * outputStride)
                        for r in 0..<rows { for c in 0..<rows {
                            var expected: Float = 0
                            for j in 0..<d {
                                let x = a[r * stride + j], y = b[c * stride + j]
                                expected += name == "l2_distance_kernel" ? (x - y) * (x - y) : x * y
                            }
                            XCTAssertEqual(result[r * outputStride + c], expected, accuracy: 1e-4, "\(path) \(name) d=\(d) stride=\(stride)")
                        }
                        XCTAssertEqual(result[r * outputStride + rows], -12345, "output padding") }
                    }
                }
            }
        }
    }
    func testNormalizationPreservesOddStrideWritesAndDegenerateBits() throws {
        try withLibraries { device, library, path in
            let d = 7, inputStride = 9, outputStride = 11, rows = 4
            for degenerate in [false, true] {
                var values = [Float](repeating: 999, count: rows * inputStride)
                for r in 0..<rows { for j in 0..<d {
                    values[r * inputStride + j] = degenerate
                        ? Float(bitPattern: UInt32(j + 1) | (j % 2 == 0 ? 0x80000000 : 0))
                        : Float(r + j + 1)
                } }
                let input = try buffer(values, device)
                let output = try buffer([Float](repeating: -12345, count: rows * outputStride), device)
                let norms = try buffer([Float](repeating: -1, count: rows), device)
                let params = try buffer([UInt32(rows), UInt32(d), UInt32(inputStride), UInt32(outputStride), 0, 1], device)
                try run("l2_normalize_general_kernel", device, library, [input, output, norms, params], grid: MTLSize(width: rows, height: 1, depth: 1))
                let result = output.contents().bindMemory(to: Float.self, capacity: rows * outputStride)
                for r in 0..<rows {
                    let norm = sqrt((0..<d).reduce(Float(0)) { $0 + values[r * inputStride + $1] * values[r * inputStride + $1] })
                    for j in 0..<d {
                        if degenerate {
                            XCTAssertEqual(result[r * outputStride + j].bitPattern, values[r * inputStride + j].bitPattern, path)
                        } else {
                            XCTAssertEqual(result[r * outputStride + j], values[r * inputStride + j] / norm, accuracy: 1e-6, path)
                        }
                    }
                    for j in d..<outputStride { XCTAssertEqual(result[r * outputStride + j], -12345, "output padding") }
                }
            }
        }
    }

    func testPairedReductionsReadOddDimensionRows() throws {
        try withLibraries { device, library, path in
            for d in [5, 6, 7, 19] {
                let rows = 4
                let a = (0..<(rows * d)).map { Float($0 % 7 + 1) }
                let b = (0..<(rows * d)).map { Float($0 % 5 - 2) }
                for name in ["l2_distance", "cosine_similarity"] {
                    let output = try buffer([Float](repeating: -12345, count: rows + 1), device)
                    try run(name, device, library, [buffer(a, device), buffer(b, device), output, buffer([UInt32(d)], device), buffer([UInt32(0)], device)],
                            grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
                    let result = output.contents().bindMemory(to: Float.self, capacity: rows + 1)
                    for r in 0..<rows {
                        var sq: Float = 0, dot: Float = 0, aa: Float = 0, bb: Float = 0
                        for j in 0..<d {
                            let x = a[r * d + j], y = b[r * d + j]
                            sq += (x - y) * (x - y); dot += x * y; aa += x * x; bb += y * y
                        }
                        XCTAssertEqual(result[r], name == "l2_distance" ? sq : dot / sqrt(aa * bb), accuracy: 1e-5, path)
                    }
                    XCTAssertEqual(result[rows], -12345)
                }
            }
        }
    }

    func testPQSubvectorsWithOddDimensions() throws {
        try withLibraries { device, library, path in
            for dSub in [5, 6, 7] {
                let rows = 4, m = 3, k = 4, d = m * dSub
                var vectors: [Float] = [], codebooks: [Float] = []
                for r in 0..<rows { for sub in 0..<m {
                    vectors += [Float](repeating: Float((r + sub) % k) * 10 + 1, count: dSub)
                } }
                for _ in 0..<m { for centroid in 0..<k {
                    codebooks += [Float](repeating: Float(centroid * 10), count: dSub)
                } }
                let output = try buffer([UInt8](repeating: 255, count: rows * m + 1), device)
                let config = try buffer([UInt32(rows), UInt32(d), UInt32(m), UInt32(k), UInt32(dSub)], device)
                try run("pq_assignment_or_encoding", device, library, [buffer(vectors, device), buffer(codebooks, device), output, config], grid: MTLSize(width: rows, height: m, depth: 1))
                let result = output.contents().bindMemory(to: UInt8.self, capacity: rows * m + 1)
                for r in 0..<rows { for sub in 0..<m { XCTAssertEqual(result[r * m + sub], UInt8((r + sub) % k), path) } }
                XCTAssertEqual(result[rows * m], 255)
            }
        }
    }

    func testNeuralTransposedDecodeHandlesOddCodeAndWeightRows() throws {
        try withLibraries { device, library, path in
            for d in [5, 6, 7, 19, 20] {
                for latent in [5, 6, 7, 64, 128] {
                    let rows = 4
                    let codes = (0..<(rows * latent)).map { Int8($0 % 7 - 3) }
                    let scales: [Float] = [0.5, 1, 2, 4]
                    let weights = (0..<(latent * d)).map { Float($0 % 5 - 2) }
                    let bias = (0..<d).map { Float($0 % 3) }
                    let output = try buffer([Float](repeating: -12345, count: rows * d + 1), device)
                    let params = try buffer([UInt32(rows), UInt32(d), UInt32(latent), UInt32(d), 0], device)
                    try run("neural_dequantize_decode_2d_transposed_v2_kernel", device, library,
                            [buffer(codes, device), buffer(scales, device), buffer(weights, device), output, buffer(bias, device), params],
                            grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 1, height: 32, depth: 1))
                    let result = output.contents().bindMemory(to: Float.self, capacity: rows * d + 1)
                    for r in 0..<rows { for j in 0..<d {
                        var expected = bias[j]
                        for i in 0..<latent { expected += Float(codes[r * latent + i]) * scales[r] * weights[i * d + j] }
                        XCTAssertEqual(result[r * d + j], expected, accuracy: 1e-5, "\(path) d=\(d) latent=\(latent)")
                    } }
                    XCTAssertEqual(result[rows * d], -12345)
                }
            }
        }
    }

    /// Instrument the production full-block loads to report their requested
    /// footprint before dereferencing. Unused lanes can hide illegal requests
    /// from value tests and GPU validation after optimization.
    func testNeuralTailNeverRequestsWeightsPastBufferEnd() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        let url = try XCTUnwrap(bundle.url(forResource: "NeuralQuantization", withExtension: "metal"))
        let source = try String(contentsOf: url, encoding: .utf8)
        let structStart = try XCTUnwrap(source.range(of: "struct NeuralQuantParams {"))
        let structEnd = try XCTUnwrap(source.range(of: "};", range: structStart.upperBound..<source.endIndex))
        let kernelStart = try XCTUnwrap(source.range(of: "kernel void neural_dequantize_decode_2d_transposed_v2_kernel("))
        let kernelEnd = try XCTUnwrap(source.range(of: "// MARK: - Specialized Transposed Decode Variants", range: kernelStart.upperBound..<source.endIndex))
        var kernel = String(source[kernelStart.lowerBound..<kernelEnd.lowerBound])
        let loads = try NSRegularExpression(pattern: #"\*\(\(device const (?:packed_)?float4\*\)\((decoderWeightsT \+ [^\n]+?)\)\)"#)
        XCTAssertEqual(loads.numberOfMatches(in: kernel, range: NSRange(kernel.startIndex..., in: kernel)), 20, "Instrument every production full-block weight load")
        kernel = loads.stringByReplacingMatches(in: kernel, range: NSRange(kernel.startIndex..., in: kernel),
            withTemplate: "checked_weights($1, decoderWeightsT, ulong(latentDim) * inputDim, invalidLoads)")
        kernel = kernel.replacingOccurrences(of: #"params\s+\[\[buffer\(5\)\]\],"#, with: "params [[buffer(5)]], device atomic_uint* invalidLoads [[buffer(6)]],", options: .regularExpression)
        let instrumented = """
        #include <metal_stdlib>
        using namespace metal;
        \(source[structStart.lowerBound..<structEnd.upperBound])
        inline float4 checked_weights(device const float* p, device const float* start,
                                      ulong count, device atomic_uint* invalid) {
            ulong offset = p - start;
            if (offset > count || count - offset < 4) {
                atomic_fetch_add_explicit(invalid, 1u, memory_order_relaxed);
                return float4(0);
            }
            return *((device const packed_float4*)p);
        }
        \(kernel)
        """
        let options = MTLCompileOptions(); options.languageVersion = .version4_0; options.mathMode = .fast
        let library = try device.makeLibrary(source: instrumented, options: options)
        for d in [5, 6, 7, 19, 20, 129] { for latent in [5, 7, 64, 128] {
            let invalid = try buffer([UInt32(0)], device)
            try run("neural_dequantize_decode_2d_transposed_v2_kernel", device, library,
                    [buffer([Int8](repeating: 1, count: latent), device), buffer([Float(1)], device),
                     buffer([Float](repeating: 1, count: latent * d), device), buffer([Float](repeating: 0, count: d), device),
                     buffer([Float](repeating: 0, count: d), device), buffer([UInt32(1), UInt32(d), UInt32(latent), UInt32(d), 0], device), invalid],
                    grid: MTLSize(width: 1, height: (d + 127) / 128, depth: 1), group: MTLSize(width: 1, height: 32, depth: 1))
            XCTAssertEqual(invalid.contents().load(as: UInt32.self), 0, "d=\(d) latent=\(latent): invalid full-block weight read")
        } }
    }

    func testTiledIntegerAndFloatStorageHandlesRaggedRows() throws {
        try withLibraries { device, library, path in
            let q = 4, n = 3
            for d in [5, 6, 7, 65] {
                let wordsA = (0..<(q * d)).map { UInt32($0) &* 0x9E3779B9 }
                let wordsB = (0..<(n * d)).map { UInt32($0 + 7) &* 0x85EBCA6B }
                let bits = try buffer([UInt32](repeating: .max, count: q * n + 1), device)
                try run("hamming_distance_batch", device, library,
                        [buffer(wordsA, device), buffer(wordsB, device), bits, buffer([UInt32(q)], device), buffer([UInt32(n)], device), buffer([UInt32(d)], device)],
                        grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 16, height: 16, depth: 1))
                let counts = bits.contents().bindMemory(to: UInt32.self, capacity: q * n + 1)
                for r in 0..<q { for c in 0..<n {
                    let expected = (0..<d).reduce(0) { $0 + (wordsA[r * d + $1] ^ wordsB[c * d + $1]).nonzeroBitCount }
                    XCTAssertEqual(counts[r * n + c], UInt32(expected), "\(path) d_words=\(d)")
                } }
                XCTAssertEqual(counts[q * n], .max)

                let a = (0..<(q * d)).map { Float($0 % 7) }
                let b = (0..<(n * d)).map { Float($0 % 5 - 2) }
                for name in ["minkowski_distance_batch", "minkowski_distance_stable"] {
                    let output = try buffer([Float](repeating: -12345, count: q * n + 1), device)
                    try run(name, device, library,
                            [buffer(a, device), buffer(b, device), output, buffer([Float(1)], device), buffer([UInt32(q)], device), buffer([UInt32(n)], device), buffer([UInt32(d)], device), buffer([UInt32(0)], device)],
                            grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 16, height: 16, depth: 1))
                    let result = output.contents().bindMemory(to: Float.self, capacity: q * n + 1)
                    for r in 0..<q { for c in 0..<n {
                        let expected = (0..<d).reduce(Float(0)) { $0 + abs(a[r * d + $1] - b[c * d + $1]) }
                        XCTAssertEqual(result[r * n + c], expected, accuracy: expected * 1e-5, "\(path) \(name) d=\(d)")
                    } }
                    XCTAssertEqual(result[q * n], -12345)
                }
            }
        }
    }

}
