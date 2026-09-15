import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

/// VA3-018: execute address expressions extracted from the actual shader sources.
/// This tests the 2^32 boundary without allocating or dereferencing a 16+ GiB buffer.
/// Declarations keep their production types, so widening only the RHS cannot hide
/// a subsequent narrowing. Ordinary dispatch tests and both full library builds
/// complement these arithmetic probes; they are not large-allocation tests.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class IndexWidthTests: XCTestCase {
    private struct Probe {
        let file: String
        let pattern: String
        let substitutions: [String: String]
        let result: String?
        init(_ file: String, _ pattern: String, _ substitutions: [String: String], result: String? = nil) {
            self.file = file; self.pattern = pattern
            self.substitutions = substitutions; self.result = result
        }
    }

    private let shaders = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders")

    private func check(_ probes: [Probe], values cases: [[UInt32]] = [[3, 7, 2], [65535, 65536, 65535], [65536, 65536, 0],
                                  [65536, 65536, 1], [3_000_000, 1536, 0], [66_000, 66_000, 19], [.max, .max, 0]], expected: (UInt64, UInt64, UInt64) -> UInt64 = { $0 * $1 + $2 }) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var snippets: [(String, String)] = []
        for probe in probes {
            let source = try String(contentsOf: shaders.appendingPathComponent(probe.file + ".metal"), encoding: .utf8)
                .replacingOccurrences(of: #"//[^\n]*"#, with: "", options: .regularExpression)
            let regex = try NSRegularExpression(pattern: probe.pattern)
            let matches = regex.matches(in: source, range: NSRange(source.startIndex..., in: source))
            XCTAssertFalse(matches.isEmpty, "Missing production probe: \(probe.file) \(probe.pattern)")
            for match in matches {
                var code = String(source[Range(match.range(at: 1), in: source)!])
                for key in probe.substitutions.keys.sorted(by: { $0.count > $1.count }) {
                    code = code.replacingOccurrences(of: #"\b"# + NSRegularExpression.escapedPattern(for: key) + #"\b"#,
                                                    with: probe.substitutions[key]!, options: .regularExpression)
                }
                let label = "\(probe.file): \(code)"
                code = probe.result.map { "\(code) out[\(snippets.count)] = \($0);" }
                    ?? "out[\(snippets.count)] = \(code);"
                snippets.append((code, label))
            }
        }
        XCTAssertFalse(snippets.isEmpty)
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void offset_probe(device ulong* out [[buffer(0)]], constant uint* values [[buffer(1)]]) {
            const uint a = values[0], b = values[1], c = values[2];
            \(snippets.map { "{ \($0.0) }" }.joined(separator: "\n"))
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "offset_probe")))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let output = try XCTUnwrap(device.makeBuffer(length: snippets.count * 8, options: .storageModeShared))
        // Ordinary control; immediately below/at/above 2^32; realistic normalize
        // and dense-matrix shapes; full UInt32 product with a representable sum.
        for values in cases {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(output, offset: 0, index: 0)
            encoder.setBytes(values, length: 12, index: 1)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed, String(describing: command.error))
            let reference = expected(UInt64(values[0]), UInt64(values[1]), UInt64(values[2]))
            for (i, snippet) in snippets.enumerated() {
                XCTAssertEqual(output.contents().load(fromByteOffset: i * 8, as: UInt64.self), reference,
                               "\(snippet.1), inputs=\(values)")
            }
        }
    }

    func testStridedDistanceOutputsPreserveWideOffsets() throws {
        try check(["L2Distance", "DotProduct", "LearnedDistance"].map {
            Probe($0, #"(const (?:uint|ulong) outputIdx = [^;]+;)"#,
                  ["queryIdx": "a", "params.strideOutput": "b", "dbIdx": "c"], result: "outputIdx")
        } + [
            Probe("AttentionSimilarity", #"similarities\[([^\]]+)\] = similarity;"#,
                  ["queryIdx": "a", "params.strideOutput": "b", "keyIdx": "c"]),
            Probe("MutualReachability", #"output\[([^\]]*params\.n[^\]]+)\]"#,
                  ["i": "a", "params.n": "b", "j": "c"])
        ])
    }

    func testNormalizationAndBatchOffsetsPreserveWideLocals() throws {
        try check([
            Probe("L2Normalization", #"(const (?:uint|ulong) input_offset = [^;]+;)"#,
                  ["tid": "a", "params.input_stride": "b"], result: "input_offset"),
            Probe("L2Normalization", #"(const (?:uint|ulong) output_offset = [^;]+;)"#,
                  ["tid": "a", "params.output_stride": "b"], result: "output_offset"),
            Probe("L2Normalization", #"(const (?:uint|ulong) offset = [^;]+;)"#,
                  ["tid": "a", "params.input_stride": "b", "DIMENSION": "b"], result: "offset"),
            Probe("BasicOperations", #"(const (?:uint|ulong) vector_offset = [^;]+;)"#,
                  ["vector_idx": "a", "dimension": "b"], result: "vector_offset"),
            Probe("BasicOperations", #"((?:uint|ulong) dbOffset = [^;]+;)"#,
                  ["dbIdx": "a", "dimension": "b"], result: "dbOffset"),
            Probe("BasicOperations", #"((?:uint|ulong) rowOffset = [^;]+;)"#,
                  ["tid": "a", "cols": "b"], result: "rowOffset")
        ], expected: { a, b, _ in a * b })
    }

    func testRowPointerExpressions() throws {
        try check([
            Probe("L2Distance", #"= (?:queryVectors|databaseVectors) \+ ([^;]+);"#,
                  ["queryIdx": "a", "dbIdx": "a", "params.strideQuery": "b", "params.strideDatabase": "b"]),
            Probe("DotProduct", #"= (?:queryVectors|databaseVectors|matrix) \+ ([^;]+);"#,
                  ["queryIdx": "a", "dbIdx": "a", "params.strideQuery": "b", "params.strideDatabase": "b"]),
            Probe("LearnedDistance", #"= (?:queryVectors|databaseVectors|inputVectors|outputVectors|vectors|weights|projectionWeights) \+ ([^;]+);"#,
                  ["queryIdx": "a", "dbIdx": "a", "vectorIdx": "a", "tid": "a", "j": "a", "outputDimIdx": "a",
                   "params.strideQuery": "b", "params.strideDatabase": "b", "params.stride": "b", "params.inputDimension": "b", "params.outputDimension": "b", "INPUT_DIM": "b", "inputDim": "b"]),
            Probe("LogSumExp", #"= input \+ ([^;]+);"#,
                  ["tid": "a", "row": "a", "d": "b", "d4": "b"]),
            Probe("StatisticsShaders", #"= datasets \+ ([^;]+);"#,
                  ["i": "a", "j": "a", "dataSize": "b"])
        ], expected: { a, b, _ in a * b })
    }

    func testWideOffsetsAreNotNarrowedDownstream() throws {
        try check([
            Probe("NeuralQuantization", #"(const (?:uint|ulong) codeOffset = [^;]+;)"#,
                  ["vectorIdx": "a", "latentDim": "b", "i4": "0u", "tptg.y": "0u"], result: "codeOffset"),
            Probe("QuantizationShaders", #"((?:uint|ulong) offset = [^;]+;)"#,
                  ["vectorIdx": "a", "vectorDimension": "b"], result: "offset"),
            Probe("OptimizedMatrixOps", #"(const (?:uint|ulong) aOffset = (?:\(ulong\))?batch \* stridesA.z;)"#,
                  ["batch": "a", "stridesA.z": "b"], result: "aOffset")
        ], expected: { a, b, _ in a * b })
        try check([
            Probe("BasicOperations", #"(const (?:uint|ulong) idx = vector_offset \+ d;)"#,
                  ["vector_offset": "(ulong(a) * b)", "d": "c"], result: "idx"),
            // Preserve the helper parameter type as a local declaration: the same
            // implicit conversion must accept the caller's complete wide index.
            Probe("BasicOperations", #"inline void va_copy_bits\([^\n]+, ((?:uint|ulong) i)\)"#,
                  ["i": "offset = (ulong)a * b + c;"], result: "offset")
        ])
    }

    func testUMAPAndSoAAddressExpressions() throws {
        try check([
            Probe("UMAPGradient", #"(?:embedding|edgeGradients|targetGradients|pointGradients)\[([^\]]*params\.d[^\]]+)\]"#,
                  ["i": "a", "j": "a", "tid": "a", "edgeIdx": "a", "params.d": "b", "k": "c"]),
            Probe("UMAPGradient", #"randomTargets\[([^\]]+)\]"#,
                  ["tid": "a", "params.negSampleRate": "b", "s": "c"]),
            Probe("SoADistance", #"candidates\[([^\]]+)\]"#,
                  ["l": "a", "p.count": "b", "j": "c"])
        ])
    }

    func testMatrixAddressExpressions() throws {
        try check([
            Probe("OptimizedMatrixOps", #"(?:A|B|C|input|output|matrix|bias)\[((?:\(ulong\))?(?:aRow|bRow|globalRow|inRow|outRow|row|batch)\s*\*[^\]]+)\]"#,
                  ["aRow": "a", "bRow": "a", "globalRow": "a", "inRow": "a", "outRow": "a", "row": "a", "batch": "a",
                   "K": "b", "N": "b", "cols": "b", "rows": "b", "aCol": "c", "bCol": "c", "globalCol": "c", "inCol": "c", "outCol": "c", "i": "c", "col": "c",
                   "aOffset": "0ul", "bOffset": "0ul", "cOffset": "0ul", "k": "a"])
        ])
    }

    func testBatchAndStridedMatrixProducts() throws {
        try check([
            Probe("OptimizedMatrixOps", #"(const (?:uint|ulong) aOffset = [^;]*batch \* M \* K;)"#,
                  ["batch": "a", "M": "b", "K": "c"], result: "aOffset"),
            Probe("OptimizedMatrixOps", #"(const (?:uint|ulong) bOffset = [^;]*batch \* K \* N;)"#,
                  ["batch": "a", "K": "b", "N": "c"], result: "bOffset"),
            Probe("OptimizedMatrixOps", #"(const (?:uint|ulong) cOffset = [^;]*batch \* M \* N;)"#,
                  ["batch": "a", "M": "b", "N": "c"], result: "cOffset")
        ], values: [[3, 7, 2], [65536, 65536, 1], [4096, 4096, 512], [65535, 65535, 2]],
           expected: { $0 * $1 * $2 })
        // Exercise the nested unrolled expressions without overflowing the
        // within-row k+lane itself (the production loop guards that separately).
        for lane in 0..<4 {
            try check([
                Probe("OptimizedMatrixOps", "B\\[(bOffset \\+ [^\\]]*\\(k \\+ \(lane)\\)[^\\]]+)\\]",
                      ["bOffset": "0ul", "k": "a", "N": "b", "col": "c"])
            ], values: [[3, 7, 2], [65536, 65536, 1], [3_000_000, 1536, 0]],
               expected: { a, b, c in (a + UInt64(lane)) * b + c })
        }
        // A wide batch base does not widen either independent multiplication.
        for selectRow in [true, false] {
            try check([
                Probe("OptimizedMatrixOps", #"((?:uint|ulong) aIdx = [^;]+;)"#,
                      ["aOffset": "ulong(c)", "row": selectRow ? "a" : "0u", "k": selectRow ? "0u" : "a",
                       "stridesA.x": "b", "stridesA.y": "b"], result: "aIdx"),
                Probe("OptimizedMatrixOps", #"((?:uint|ulong) bIdx = [^;]+;)"#,
                      ["bOffset": "ulong(c)", "k": selectRow ? "a" : "0u", "col": selectRow ? "0u" : "a",
                       "stridesB.x": "b", "stridesB.y": "b"], result: "bIdx"),
                Probe("OptimizedMatrixOps", #"((?:uint|ulong) cIdx = [^;]+;)"#,
                      ["cOffset": "ulong(c)", "row": selectRow ? "a" : "0u", "col": selectRow ? "0u" : "a",
                       "stridesC.x": "b", "stridesC.y": "b"], result: "cIdx")
            ])
        }
    }

    func testSpecializedNeuralVectorStoreOffsets() throws {
        try check([
            Probe("NeuralQuantization", #"outputVectors \+ ([^;]*vectorIdx \* INPUT_DIM \+ outBase)\)"#,
                  ["vectorIdx": "a", "INPUT_DIM": "b", "outBase": "c"])
        ])
    }

    func testNeuralAndPQRowAddressExpressions() throws {
        try check([
            Probe("NeuralQuantization", #"(?:latentVectors|outputVectors)\[([^\]]+)\] = sum;"#,
                  ["vectorIdx": "a", "params.latentDimension": "b", "params.inputDimension": "b", "inputDim": "b", "latentDimIdx": "c", "outputDimIdx": "c", "outIdx": "c", "outputOffset": "(ulong(a) * b)", "j": "c"]),
            Probe("NeuralQuantization", #"(const (?:uint|ulong) outputOffset = [^;]+;)"#,
                  ["vectorIdx": "a", "inputDim": "b", "outBase": "c"], result: "outputOffset"),
            Probe("ProductQuantization", #"assignments(?:_or_codes)?\[([^\]]+)\]"#,
                  ["vec_id": "a", "config.M": "b", "m": "c"]),
            Probe("QuantizationShaders", #"(?:vectors|binaryVectors|candidateBinary)\[([^\]]*\*[^\]]+)\]"#,
                  ["vectorIdx": "a", "candidateIdx": "a", "vectorDimension": "b", "numWords": "b", "dimIdx": "c", "wordIdx": "c"])
        ])
    }

    func testSearchSparseAndBoruvkaRowOffsets() throws {
        try check([
            Probe("SearchAndRetrieval", #"= (?:topk_values|topk_indices|distances) \+ ([^;]+);"#,
                  ["tid": "a", "params.output_stride": "b", "params.input_stride": "b"]),
            Probe("SearchAndRetrieval", #"(const (?:uint|ulong) (?:running_row|chunk_row) = [^;]+;)"#,
                  ["tid": "a", "K": "b", "Kc": "b", "running_row": "offset", "chunk_row": "offset"], result: "offset"),
            Probe("SparseLogTFIDF", #"= (?:topKIndices|topKScores) \+ ([^;]+);"#,
                  ["cid": "a", "topK": "b"]),
            Probe("BoruvkaMST", #"= embeddings \+ ([^;]+);"#,
                  ["tid": "a", "j": "a", "params.d": "b", "simd_blocks": "0u"])
        ], expected: { a, b, _ in a * b })
    }

    func testClusteringStatisticsAndIVFOffsets() throws {
        try check([
            Probe("ClusteringShaders", #"((?:uint|ulong) v_offset = [^;]+;)"#,
                  ["i": "a", "dimension": "b", "base_dim": "c"], result: "v_offset"),
            Probe("ClusteringShaders", #"((?:uint|ulong) out_offset = [^;]+;)"#,
                  ["cluster_id": "a", "dimension": "b", "base_dim": "c"], result: "out_offset"),
            Probe("StatisticsShaders", #"((?:uint|ulong) idx_ij = [^;]+;)"#,
                  ["i": "a", "numDatasets": "b", "j": "c"], result: "idx_ij"),
            Probe("IVFCandidateBuilder", #"nearestCentroids\[([^\]]+)\]"#,
                  ["q": "a", "nprobe": "b", "p": "c"])
        ])
        try check([
            Probe("SearchAndRetrieval", #"(const (?:uint|ulong) [qv]Base = [^;]+;)"#,
                  ["q": "a", "slot": "a", "D": "b", "qBase": "offset", "vBase": "offset"], result: "offset"),
            Probe("DataTransformations", #"(const (?:uint|ulong) idx_(?:a|b|out) = [^;]+;)"#,
                  ["tid": "a", "params.stride_a": "b", "params.stride_b": "b", "params.stride_output": "b", "idx_a": "offset", "idx_b": "offset", "idx_out": "offset"], result: "offset")
        ], expected: { a, b, _ in a * b })
    }
}
