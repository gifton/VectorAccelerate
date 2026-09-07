import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SparseTFIDFBoundsTests: XCTestCase {
    private func buffer<T: BitwiseCopyable>(_ values: [T], _ device: any MTLDevice) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                        options: .storageModeShared))
    }

    private func withLibraries(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        #if DEBUG
        try body(device, device.makeLibrary(URL: XCTUnwrap(bundle.url(forResource: "debug", withExtension: "metallib"))), "plugin")
        #endif
        try body(device, KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle), "runtime")
    }

    private func run(_ name: String, device: any MTLDevice, library: any MTLLibrary,
                     buffers: [any MTLBuffer], threads: Int) throws {
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
        encoder.dispatchThreadgroups(MTLSize(width: threads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
    }

    private func shaderSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders/SparseLogTFIDF.metal"), encoding: .utf8)
    }

    /// Removing the scalar tail must corrupt the output canary even on GPUs that
    /// tolerate full-vector reads beyond the logical end of a padded allocation.
    func testVectorizedTailWritesOnlyLiveScores() throws {
        try withLibraries { device, library, path in
            for nnz in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19] {
                let capacity = ((nnz + 3) / 4 + 1) * 4
                let indices = try buffer([UInt32](repeating: 0, count: capacity), device)
                let freqs = try buffer((0..<capacity).map { Float($0 + 1) }, device)
                let scores = try buffer([Float](repeating: -12345, count: capacity), device)
                let params = try buffer([CTFIDFParamsGPU(avgClusterSize: 10, nnz: UInt32(nnz))], device)
                try run("sparse_ctfidf_vectorized_kernel", device: device, library: library,
                        buffers: [indices, freqs, buffer([Float(10)], device), scores, params],
                        threads: (nnz + 3) / 4 + 3)
                let result = scores.contents().bindMemory(to: Float.self, capacity: capacity)
                for i in 0..<nnz {
                    XCTAssertEqual(result[i], Float(Double(i + 1) * log(2.0)), accuracy: 2e-5, "\(path), nnz=\(nnz), i=\(i)")
                }
                for i in nnz..<capacity { XCTAssertEqual(result[i], -12345, "\(path), tail write nnz=\(nnz), i=\(i)") }
            }
        }
    }

    /// Checked corpus gathers turn poisoned unused term IDs into reported
    /// violations before dereferencing them. The production control flow is retained.
    func testVectorizedTailDoesNotGatherUnusedTermIDs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var source = try shaderSource()
        let start = try XCTUnwrap(source.range(of: "kernel void sparse_ctfidf_vectorized_kernel("))
        let end = try XCTUnwrap(source.range(of: "// MARK: - Top-K Extraction", range: start.upperBound..<source.endIndex))
        var function = String(source[start.lowerBound..<end.lowerBound])
        function = function.replacingOccurrences(of: "uint tid [[thread_position_in_grid]]",
            with: "device atomic_uint* violations [[buffer(5)]], uint tid [[thread_position_in_grid]]")
        let gathers = try NSRegularExpression(pattern: #"corpusFreqs\[([^\]]+)\]"#)
        let matches = gathers.matches(in: function, range: NSRange(function.startIndex..., in: function))
        XCTAssertGreaterThanOrEqual(matches.count, 4, "Every vector gather must be instrumented")
        for match in matches.reversed() {
            let expression = String(function[Range(match.range(at: 1), in: function)!])
            function.replaceSubrange(Range(match.range, in: function)!, with: "checkedCorpus(corpusFreqs, \(expression), violations)")
        }
        source.replaceSubrange(start.lowerBound..<end.lowerBound, with: function)
        source = source.replacingOccurrences(of: "using namespace metal;", with: """
        using namespace metal;
        inline float checkedCorpus(device const float* corpus, uint index, device atomic_uint* violations) {
            if (index >= 1) { atomic_fetch_add_explicit(violations, 1u, memory_order_relaxed); return 1.0f; }
            return corpus[index];
        }
        """)
        let options = MTLCompileOptions(); options.languageVersion = .version4_0; options.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: options)
        for nnz in [1, 2, 3, 5, 6, 7, 17, 18, 19] {
            let capacity = (nnz + 3) / 4 * 4
            let indices = [UInt32](repeating: 0, count: nnz) + [UInt32](repeating: .max, count: capacity - nnz)
            let violations = try buffer([UInt32(0)], device)
            try run("sparse_ctfidf_vectorized_kernel", device: device, library: library,
                    buffers: [buffer(indices, device), buffer([Float](repeating: 2, count: capacity), device),
                              buffer([Float(10)], device), buffer([Float](repeating: 0, count: capacity), device),
                              buffer([CTFIDFParamsGPU(avgClusterSize: 10, nnz: UInt32(nnz))], device), violations],
                    threads: (nnz + 3) / 4 + 2)
            XCTAssertEqual(violations.contents().load(as: UInt32.self), 0, "Unused term IDs gathered for nnz=\(nnz)")
        }
    }

    /// Count entry into the first buffer-reading statement and stop a bad zero-K
    /// dispatch before its unsigned underflow can trigger an actual GPU fault.
    func testZeroKShaderReturnsBeforeReadingClusterData() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var source = try shaderSource()
        let start = try XCTUnwrap(source.range(of: "kernel void ctfidf_topk_per_cluster_kernel("))
        var function = String(source[start.lowerBound...])
        function = function.replacingOccurrences(of: "uint cid [[thread_position_in_grid]]",
            with: "device atomic_uint* violations [[buffer(7)]], uint cid [[thread_position_in_grid]]")
        let firstRead = try XCTUnwrap(function.range(of: "uint start = clusterOffsets[cid];"))
        function.insert(contentsOf: "if (topK == 0) { atomic_fetch_add_explicit(violations, 1u, memory_order_relaxed); return; }\n", at: firstRead.lowerBound)
        source.replaceSubrange(start.lowerBound..., with: function)
        let options = MTLCompileOptions(); options.languageVersion = .version4_0; options.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: options)
        let dummy = try buffer([UInt32.max], device)
        let violations = try buffer([UInt32(0)], device)
        try run("ctfidf_topk_per_cluster_kernel", device: device, library: library,
                buffers: [dummy, dummy, dummy, dummy, dummy, buffer([UInt32(3)], device), buffer([UInt32(0)], device), violations], threads: 5)
        XCTAssertEqual(violations.contents().load(as: UInt32.self), 0, "Zero K must return before any buffer access")
    }

    func testZeroKEncoderReportsNoDispatch() async throws {
        let context = try await Metal4Context()
        let kernel = try await SparseLogTFIDFKernel(context: context)
        let device = context.device.rawDevice
        let command = try XCTUnwrap(try XCTUnwrap(device.makeCommandQueue()).makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        let dummy = try buffer([UInt32.max], device)
        let result = kernel.encodeTopK(into: encoder, scores: dummy, termIndices: dummy, clusterOffsets: dummy,
                                      topKIndices: dummy, topKScores: dummy, numClusters: 3, k: 0)
        encoder.endEncoding() // Do not commit a pre-fix zero-K dispatch with invalid dummy data.
        XCTAssertEqual(result.threadgroups.width, 0)
    }

    /// Exact logical allocations above one vector exercise the tail loads under
    /// Metal validation. The vector-typed ABI requires at least 16 bound bytes.
    func testVectorizedTailNeedsNoInputPadding() throws {
        try withLibraries { device, library, path in
            for nnz in [5, 6, 7, 17, 18, 19] {
                let indices = (0..<nnz).map { UInt32($0 % 3) }
                let corpus: [Float] = [0, 5, 20]
                let output = try buffer([Float](repeating: -12345, count: nnz), device)
                try run("sparse_ctfidf_vectorized_kernel", device: device, library: library,
                        buffers: [buffer(indices, device), buffer([Float](repeating: 2, count: nnz), device),
                                  buffer(corpus, device), output,
                                  buffer([CTFIDFParamsGPU(avgClusterSize: 10, nnz: UInt32(nnz))], device)],
                        threads: (nnz + 3) / 4 + 3)
                let result = output.contents().bindMemory(to: Float.self, capacity: nnz)
                let expected: [Float] = [Float(2 * log(11.0)), Float(2 * log(3.0)), Float(2 * log(1.5))]
                for i in 0..<nnz { XCTAssertEqual(result[i], expected[i % 3], accuracy: 1e-5, "\(path), nnz=\(nnz)") }
            }
        }
    }

    /// Real pipeline coverage for zero, one, and oversized K, empty clusters,
    /// output padding, and over-dispatched cluster IDs on both compile paths.
    func testTopKShaderBoundsAndSentinels() throws {
        try withLibraries { device, library, path in
            for k in [0, 1, 5] {
                let capacity = 3 * k + 4
                let outIndices = try buffer([UInt32](repeating: 12345, count: capacity), device)
                let outScores = try buffer([Float](repeating: -12345, count: capacity), device)
                try run("ctfidf_topk_per_cluster_kernel", device: device, library: library,
                        buffers: [buffer([Float(2), 5, 1, 7], device), buffer([UInt32(3), 9, 2, 4], device),
                                  buffer([UInt32(0), 3, 3, 4], device), outIndices, outScores,
                                  buffer([UInt32(3)], device), buffer([UInt32(k)], device)], threads: 6)
                let indices = outIndices.contents().bindMemory(to: UInt32.self, capacity: capacity)
                let scores = outScores.contents().bindMemory(to: Float.self, capacity: capacity)
                let expectedIDs: [[UInt32]] = [[9, 3, 2], [], [4]]
                let expectedScores: [[Float]] = [[5, 2, 1], [], [7]]
                for c in 0..<3 { for j in 0..<k {
                    let valid = j < expectedIDs[c].count
                    XCTAssertEqual(indices[c * k + j], valid ? expectedIDs[c][j] : .max, path)
                    XCTAssertEqual(scores[c * k + j], valid ? expectedScores[c][j] : -.infinity, path)
                } }
                for i in (3 * k)..<capacity {
                    XCTAssertEqual(indices[i], 12345, path)
                    XCTAssertEqual(scores[i], -12345, path)
                }
            }
        }
    }

    func testZeroKPublicAPIKeepsClusterShapeWithoutWork() async throws {
        let context = try await Metal4Context()
        let kernel = try await SparseLogTFIDFKernel(context: context)
        let empty = ClusterTermFrequencies(termIndices: [], frequencies: [])
        let populated = ClusterTermFrequencies(termIndices: [0, 1], frequencies: [2, 3])
        for clusters in [[], [empty, empty], [populated, empty, populated]] {
            let result = try await kernel.topKPerCluster(clusterTerms: clusters, corpusFrequencies: [10, 20],
                                                        avgClusterSize: 10, k: 0)
            XCTAssertEqual(result.clusterCount, clusters.count)
            XCTAssertEqual(result.topKPerCluster.count, clusters.count)
            XCTAssertTrue(result.topKPerCluster.allSatisfy { $0.isEmpty })
            XCTAssertEqual(result.k, 0)
            XCTAssertEqual(result.executionTime, 0)
            XCTAssertEqual(result.throughputGBps, 0)
        }
    }

    func testInvalidKThrowsEvenForEmptyInput() async throws {
        let context = try await Metal4Context()
        let kernel = try await SparseLogTFIDFKernel(context: context)
        for k in [-1, Int(UInt32.max) + 1] {
            do {
                _ = try await kernel.topKPerCluster(clusterTerms: [], corpusFrequencies: [], avgClusterSize: 10, k: k)
                XCTFail("Invalid K must be rejected consistently, including empty input: \(k)")
            } catch { XCTAssertTrue(error is VectorError) }
        }
    }
}
