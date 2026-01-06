//
//  SwiftTopicsBenchmarks.swift
//  VectorAccelerate
//
//  Performance benchmarks for SwiftTopics integration kernels.
//
//  Tests GPU vs CPU performance at scales from the implementation plan:
//  - MutualReachabilityKernel: Expected 25-125x speedup
//  - BoruvkaMSTKernel: Expected significant speedup for MST construction
//  - UMAPGradientKernel: Expected speedup for gradient computation
//  - SparseLogTFIDFKernel: Expected speedup for c-TF-IDF
//
//  Reference: docs/SWIFTTOPICS_IMPLEMENTATION_PLAN.md

import XCTest
@testable import VectorAccelerate
import VectorCore

// MARK: - CPU Baselines

/// CPU implementations for benchmark comparison.
private enum CPUBaselines {

    /// CPU mutual reachability: max(core_a, core_b, euclidean_dist(a, b))
    static func mutualReachability(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) -> [[Float]] {
        let n = embeddings.count
        let d = embeddings[0].count
        var result = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)

        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    result[i][j] = 0
                } else {
                    // Euclidean distance
                    var sum: Float = 0
                    for k in 0..<d {
                        let diff = embeddings[i][k] - embeddings[j][k]
                        sum += diff * diff
                    }
                    let euclidean = sqrt(sum)

                    // Mutual reachability
                    result[i][j] = max(coreDistances[i], coreDistances[j], euclidean)
                }
            }
        }
        return result
    }

    /// CPU UMAP gradient computation (simplified - attractive only)
    static func umapGradient(
        embedding: [[Float]],
        edges: [(source: Int, target: Int, weight: Float)],
        a: Float = 1.929,
        b: Float = 0.7915
    ) -> [[Float]] {
        let n = embedding.count
        let d = embedding[0].count
        var gradients = [[Float]](repeating: [Float](repeating: 0, count: d), count: n)

        for edge in edges {
            let i = edge.source
            let j = edge.target

            // Distance squared
            var distSq: Float = 0
            for k in 0..<d {
                let diff = embedding[i][k] - embedding[j][k]
                distSq += diff * diff
            }

            // UMAP gradient: w * 2ab * d^(2b-2) / (1 + a*d^2b) * (y_i - y_j)
            let dist2b = pow(distSq, b)
            let denom = 1.0 + a * dist2b
            let coeff = edge.weight * 2.0 * a * b * pow(distSq, b - 1.0) / denom

            for k in 0..<d {
                let delta = embedding[i][k] - embedding[j][k]
                gradients[i][k] += coeff * delta
            }
        }
        return gradients
    }

    /// CPU c-TF-IDF computation
    static func ctfidf(
        termFreqs: [[(termIndex: Int, freq: Float)]],
        corpusFreqs: [Float],
        avgClusterSize: Float
    ) -> [[Float]] {
        var result: [[Float]] = []
        for cluster in termFreqs {
            var scores: [Float] = []
            for (termIdx, tf) in cluster {
                let corpusTf = max(corpusFreqs[termIdx], 1.0)
                let idf = log(1.0 + avgClusterSize / corpusTf)
                scores.append(tf * idf)
            }
            result.append(scores)
        }
        return result
    }
}

// MARK: - Benchmark Results

/// Results from a single benchmark run
private struct BenchmarkRun {
    let name: String
    let scale: String
    let cpuTimeMs: Double
    let gpuTimeMs: Double

    var speedup: Double { cpuTimeMs / gpuTimeMs }
    var gpuIsFaster: Bool { speedup > 1.0 }
}

// MARK: - SwiftTopics Benchmarks

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SwiftTopicsBenchmarks: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        context = try await Metal4Context()
    }

    // MARK: - Benchmark Configuration

    /// Smaller corpus sizes for test environment (CPU baseline is O(n²×D), very slow)
    /// Full-scale benchmarks should use the standalone runner
    private let corpusSizes = [50, 100, 200]

    /// Embedding dimension (typical for MiniLM/Sentence-BERT)
    private let dimension = 128  // Reduced for faster CPU baseline

    /// Iterations for stable measurements
    private let gpuIterations = 10
    private let cpuIterations = 2

    /// Warmup iterations
    private let warmupIterations = 2

    // MARK: - Utility Functions

    private func generateEmbeddings(n: Int, d: Int) -> [[Float]] {
        (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -1...1) }
        }
    }

    private func generateCoreDistances(n: Int) -> [Float] {
        // Simulate core distances (typically k-th neighbor distance)
        (0..<n).map { _ in Float.random(in: 0.1...2.0) }
    }

    private func measureCPU(iterations: Int, operation: () -> Void) -> Double {
        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            operation()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        return times.sorted()[times.count / 2] * 1000 // median in ms
    }

    private func measureGPU(iterations: Int, warmup: Int, operation: () async throws -> Void) async throws -> Double {
        // Warmup
        for _ in 0..<warmup {
            try await operation()
        }

        // Measured runs
        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try await operation()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        return times.sorted()[times.count / 2] * 1000 // median in ms
    }

    private func printBenchmarkTable(_ results: [BenchmarkRun], title: String, expectedSpeedups: [String: Double]) {
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║  \(title.padding(toLength: 78, withPad: " ", startingAt: 0))  ║")
        print("╠═══════════════════╦══════════════╦═══════════════╦════════════╦════════════════════╣")
        print("║ Scale             ║ CPU (ms)     ║ GPU (ms)      ║ Speedup    ║ Expected           ║")
        print("╠═══════════════════╬══════════════╬═══════════════╬════════════╬════════════════════╣")

        for result in results {
            let expected = expectedSpeedups[result.scale] ?? 0
            let status = result.speedup >= expected * 0.5 ? "✓" : "⚠"
            let scale = result.scale.padding(toLength: 17, withPad: " ", startingAt: 0)
            print("║ \(scale) ║ \(String(format: "%10.2f", result.cpuTimeMs))   ║ \(String(format: "%11.3f", result.gpuTimeMs))   ║ \(String(format: "%8.1f", result.speedup))x  ║ \(String(format: "%5.0f", expected))x expected \(status) ║")
        }

        print("╚═══════════════════╩══════════════╩═══════════════╩════════════╩════════════════════╝")
    }

    // MARK: - MutualReachability Benchmark

    /// Benchmark MutualReachabilityKernel vs CPU baseline.
    ///
    /// Expected speedups from plan:
    /// - 500 docs: ~25x
    /// - 1,000 docs: ~33x
    /// - 5,000 docs: ~125x
    func testMutualReachabilityBenchmark() async throws {
        let kernel = try await MutualReachabilityKernel(context: context)
        var results: [BenchmarkRun] = []

        // Expected speedups scale with problem size (O(n²×D) computation)
        let expectedSpeedups: [String: Double] = [
            "50 docs": 5,
            "100 docs": 10,
            "200 docs": 15
        ]

        for n in corpusSizes {
            let embeddings = generateEmbeddings(n: n, d: dimension)
            let coreDistances = generateCoreDistances(n: n)

            // CPU benchmark
            let cpuTimeMs = measureCPU(iterations: cpuIterations) {
                _ = CPUBaselines.mutualReachability(embeddings: embeddings, coreDistances: coreDistances)
            }

            // GPU benchmark (uses convenience array API)
            let gpuTimeMs = try await measureGPU(iterations: gpuIterations, warmup: warmupIterations) {
                _ = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
            }

            results.append(BenchmarkRun(
                name: "MutualReachability",
                scale: "\(n) docs",
                cpuTimeMs: cpuTimeMs,
                gpuTimeMs: gpuTimeMs
            ))
        }

        printBenchmarkTable(results, title: "MutualReachabilityKernel Benchmark", expectedSpeedups: expectedSpeedups)

        // Verify we get meaningful speedups
        for result in results where result.scale != "5000 docs" {
            XCTAssertGreaterThan(result.speedup, 5.0, "Expected at least 5x speedup for \(result.scale)")
        }
    }

    // MARK: - UMAP Gradient Benchmark

    /// Benchmark UMAPGradientKernel vs CPU baseline.
    ///
    /// Expected: GPU faster at O(epochs × k × n) complexity
    func testUMAPGradientBenchmark() async throws {
        let kernel = try await UMAPGradientKernel(context: context)
        var results: [BenchmarkRun] = []

        let outputDimension = 2  // UMAP typically projects to 2D
        let kNeighbors = 15  // Typical k for UMAP

        let expectedSpeedups: [String: Double] = [
            "50 docs": 2,
            "100 docs": 3,
            "200 docs": 5
        ]

        for n in corpusSizes {
            // Generate 2D embedding (UMAP output space)
            let embedding = generateEmbeddings(n: n, d: outputDimension)

            // Generate edges (k neighbors per point)
            var edges: [UMAPEdge] = []
            for i in 0..<n {
                for _ in 0..<kNeighbors {
                    let j = Int.random(in: 0..<n)
                    if i != j {
                        edges.append(UMAPEdge(source: i, target: j, weight: Float.random(in: 0.1...1.0)))
                    }
                }
            }

            // Sort edges by source
            let sortedEdges = kernel.sortEdgesBySource(edges)

            // CPU edges format
            let cpuEdges = sortedEdges.map { (source: Int($0.source), target: Int($0.target), weight: $0.weight) }

            // CPU benchmark
            let cpuTimeMs = measureCPU(iterations: cpuIterations) {
                _ = CPUBaselines.umapGradient(embedding: embedding, edges: cpuEdges)
            }

            // GPU benchmark - uses convenience array API
            let gpuTimeMs = try await measureGPU(iterations: gpuIterations, warmup: warmupIterations) {
                var embeddingCopy = embedding  // Copy to avoid modifying original
                try await kernel.optimizeEpoch(
                    embedding: &embeddingCopy,
                    edges: sortedEdges,
                    params: .default
                )
            }

            results.append(BenchmarkRun(
                name: "UMAPGradient",
                scale: "\(n) docs",
                cpuTimeMs: cpuTimeMs,
                gpuTimeMs: gpuTimeMs
            ))
        }

        printBenchmarkTable(results, title: "UMAPGradientKernel Benchmark", expectedSpeedups: expectedSpeedups)
    }

    // MARK: - SparseLogTFIDF Benchmark

    /// Benchmark SparseLogTFIDFKernel vs CPU baseline.
    ///
    /// Expected: Modest speedup (c-TF-IDF is already fast on CPU)
    func testSparseLogTFIDFBenchmark() async throws {
        let kernel = try await SparseLogTFIDFKernel(context: context)
        var results: [BenchmarkRun] = []

        let vocabSize = 10_000
        let termsPerCluster = 100

        let expectedSpeedups: [String: Double] = [
            "50 clusters": 3,
            "100 clusters": 5,
            "200 clusters": 8
        ]

        for numClusters in [50, 100, 200] {
            // Generate cluster term frequencies
            var clusterTerms: [ClusterTermFrequencies] = []
            for _ in 0..<numClusters {
                var indices: [UInt32] = []
                var freqs: [Float] = []
                for _ in 0..<termsPerCluster {
                    indices.append(UInt32.random(in: 0..<UInt32(vocabSize)))
                    freqs.append(Float.random(in: 1...100))
                }
                clusterTerms.append(ClusterTermFrequencies(termIndices: indices, frequencies: freqs))
            }

            // Corpus frequencies
            let corpusFreqs = (0..<vocabSize).map { _ in Float.random(in: 1...1000) }
            let avgClusterSize = Float(500)

            // CPU format
            let cpuTermFreqs = clusterTerms.map { cluster in
                zip(cluster.termIndices, cluster.frequencies).map { (Int($0), $1) }
            }

            // CPU benchmark
            let cpuTimeMs = measureCPU(iterations: cpuIterations * 2) {
                _ = CPUBaselines.ctfidf(
                    termFreqs: cpuTermFreqs,
                    corpusFreqs: corpusFreqs,
                    avgClusterSize: avgClusterSize
                )
            }

            // GPU benchmark
            let gpuTimeMs = try await measureGPU(iterations: gpuIterations * 2, warmup: warmupIterations) {
                _ = try await kernel.compute(
                    clusterTerms: clusterTerms,
                    corpusFrequencies: corpusFreqs,
                    avgClusterSize: avgClusterSize
                )
            }

            results.append(BenchmarkRun(
                name: "SparseLogTFIDF",
                scale: "\(numClusters) clusters",
                cpuTimeMs: cpuTimeMs,
                gpuTimeMs: gpuTimeMs
            ))
        }

        printBenchmarkTable(results, title: "SparseLogTFIDFKernel Benchmark", expectedSpeedups: expectedSpeedups)
    }

    // MARK: - Full Pipeline Benchmark

    /// Benchmark the complete HDBSCAN distance pipeline.
    ///
    /// Tests BoruvkaMSTKernel which internally computes mutual reachability + MST.
    func testHDBSCANPipelineBenchmark() async throws {
        let mstKernel = try await BoruvkaMSTKernel(context: context)
        var results: [BenchmarkRun] = []

        let expectedSpeedups: [String: Double] = [
            "50 docs": 5,
            "100 docs": 10
        ]

        // Test smaller sizes for test environment
        for n in [50, 100] {
            let embeddings = generateEmbeddings(n: n, d: dimension)
            let coreDistances = generateCoreDistances(n: n)

            // CPU baseline (mutual reachability only - MST too slow to include)
            let cpuTimeMs = measureCPU(iterations: cpuIterations) {
                _ = CPUBaselines.mutualReachability(embeddings: embeddings, coreDistances: coreDistances)
            }

            // GPU pipeline: BoruvkaMST computes mutual reachability + MST together
            // Uses convenience array API
            let gpuTimeMs = try await measureGPU(iterations: gpuIterations, warmup: warmupIterations) {
                _ = try await mstKernel.computeMST(embeddings: embeddings, coreDistances: coreDistances)
            }

            results.append(BenchmarkRun(
                name: "HDBSCAN Pipeline",
                scale: "\(n) docs",
                cpuTimeMs: cpuTimeMs,
                gpuTimeMs: gpuTimeMs
            ))
        }

        printBenchmarkTable(results, title: "HDBSCAN Pipeline (MR + MST) Benchmark", expectedSpeedups: expectedSpeedups)
    }

    // MARK: - Summary Benchmark

    /// Run all benchmarks and print a comprehensive summary.
    func testFullBenchmarkSummary() async throws {
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                    SWIFTTOPICS KERNEL PERFORMANCE SUMMARY                        ║")
        print("║                                                                                  ║")
        print("║  Reference: docs/SWIFTTOPICS_IMPLEMENTATION_PLAN.md                              ║")
        print("║  Dimension: \(dimension) (typical MiniLM/Sentence-BERT)                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")

        // Run all benchmarks
        try await testMutualReachabilityBenchmark()
        try await testUMAPGradientBenchmark()
        try await testSparseLogTFIDFBenchmark()
        try await testHDBSCANPipelineBenchmark()

        print("\n")
        print("═══════════════════════════════════════════════════════════════════════════════════")
        print("  BENCHMARK COMPLETE")
        print("═══════════════════════════════════════════════════════════════════════════════════")
    }
}
