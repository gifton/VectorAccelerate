//
//  SwiftTopicsBenchmarkRunner.swift
//  VectorAccelerateBenchmarks
//
//  Standalone benchmark runner for SwiftTopics integration kernels.
//  Run with: swift run VectorAccelerateBenchmarks --swift-topics
//
//  Tests GPU vs CPU performance at scales from the implementation plan:
//  - MutualReachabilityKernel: Expected 25-125x speedup
//  - UMAPGradientKernel: Expected speedup for gradient computation
//  - SparseLogTFIDFKernel: Expected speedup for c-TF-IDF
//
//  Reference: docs/SWIFTTOPICS_IMPLEMENTATION_PLAN.md

import Foundation
@preconcurrency import Metal
import VectorCore
import VectorAccelerate

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
                    var sum: Float = 0
                    for k in 0..<d {
                        let diff = embeddings[i][k] - embeddings[j][k]
                        sum += diff * diff
                    }
                    let euclidean = sqrt(sum)
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

            var distSq: Float = 0
            for k in 0..<d {
                let diff = embedding[i][k] - embedding[j][k]
                distSq += diff * diff
            }

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

// MARK: - Benchmark Runner

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct SwiftTopicsBenchmarkRunner {

    // Configuration
    private let dimension = 384  // MiniLM/Sentence-BERT
    private let gpuIterations = 20
    private let cpuIterations = 5
    private let warmupIterations = 3

    public init() {}

    // MARK: - Utility Functions

    private func generateEmbeddings(n: Int, d: Int) -> [[Float]] {
        (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -1...1) }
        }
    }

    private func generateCoreDistances(n: Int) -> [Float] {
        (0..<n).map { _ in Float.random(in: 0.1...2.0) }
    }

    private func measureCPU(iterations: Int, operation: () -> Void) -> Double {
        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            operation()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        return times.sorted()[times.count / 2] * 1000
    }

    private func measureGPU(iterations: Int, warmup: Int, operation: () async throws -> Void) async throws -> Double {
        for _ in 0..<warmup {
            try await operation()
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try await operation()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        return times.sorted()[times.count / 2] * 1000
    }

    // MARK: - Print Results

    private func printHeader() {
        print("")
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                    SWIFTTOPICS KERNEL PERFORMANCE BENCHMARKS                     ║")
        print("║                                                                                  ║")
        print("║  Reference: docs/SWIFTTOPICS_IMPLEMENTATION_PLAN.md                              ║")
        print("║  Dimension: \(dimension) (typical MiniLM/Sentence-BERT)                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")
        print("")
    }

    private func printTableHeader(_ title: String) {
        print("")
        print("┌──────────────────────────────────────────────────────────────────────────────────┐")
        print("│ \(title.padding(toLength: 80, withPad: " ", startingAt: 0)) │")
        print("├───────────────────┬──────────────┬───────────────┬────────────┬──────────────────┤")
        print("│ Scale             │ CPU (ms)     │ GPU (ms)      │ Speedup    │ Expected         │")
        print("├───────────────────┼──────────────┼───────────────┼────────────┼──────────────────┤")
    }

    private func printRow(scale: String, cpuMs: Double, gpuMs: Double, speedup: Double, expected: Double) {
        let status = speedup >= expected * 0.5 ? "✓" : "⚠"
        let scalePadded = scale.padding(toLength: 17, withPad: " ", startingAt: 0)
        print("│ \(scalePadded) │ \(String(format: "%10.2f", cpuMs))   │ \(String(format: "%11.3f", gpuMs))   │ \(String(format: "%8.1f", speedup))x  │ \(String(format: "%5.0f", expected))x expected \(status) │")
    }

    private func printTableFooter() {
        print("└───────────────────┴──────────────┴───────────────┴────────────┴──────────────────┘")
    }

    // MARK: - Benchmarks

    public func runAll() async throws {
        printHeader()

        try await benchmarkMutualReachability()
        try await benchmarkUMAPGradient()
        try await benchmarkSparseLogTFIDF()

        print("")
        print("═══════════════════════════════════════════════════════════════════════════════════")
        print("  BENCHMARK COMPLETE")
        print("═══════════════════════════════════════════════════════════════════════════════════")
        print("")
    }

    public func benchmarkMutualReachability() async throws {
        print("Initializing MutualReachabilityKernel...")
        let context = try await Metal4Context()
        let kernel = try await MutualReachabilityKernel(context: context)

        // Note: CPU baseline is O(n² × D) - very slow for large N
        // These scales give meaningful results in reasonable time
        let expectedSpeedups: [(scale: String, n: Int, expected: Double)] = [
            ("50 docs", 50, 10),
            ("100 docs", 100, 25),
            ("200 docs", 200, 50),
            ("300 docs", 300, 75)
        ]

        printTableHeader("MutualReachabilityKernel Benchmark")

        for (scale, n, expected) in expectedSpeedups {
            let embeddings = generateEmbeddings(n: n, d: dimension)
            let coreDistances = generateCoreDistances(n: n)

            // CPU benchmark
            let cpuMs = measureCPU(iterations: cpuIterations) {
                _ = CPUBaselines.mutualReachability(embeddings: embeddings, coreDistances: coreDistances)
            }

            // GPU benchmark
            let gpuMs = try await measureGPU(iterations: gpuIterations, warmup: warmupIterations) {
                _ = try await kernel.compute(embeddings: embeddings, coreDistances: coreDistances)
            }

            let speedup = cpuMs / gpuMs
            printRow(scale: scale, cpuMs: cpuMs, gpuMs: gpuMs, speedup: speedup, expected: expected)
        }

        printTableFooter()
    }

    public func benchmarkUMAPGradient() async throws {
        print("Initializing UMAPGradientKernel...")
        let context = try await Metal4Context()
        let kernel = try await UMAPGradientKernel(context: context)

        let outputDimension = 2
        let kNeighbors = 15

        // UMAP at small scales - GPU shines with more edges
        let expectedSpeedups: [(scale: String, n: Int, expected: Double)] = [
            ("200 docs", 200, 2),
            ("500 docs", 500, 5),
            ("1000 docs", 1000, 10)
        ]

        printTableHeader("UMAPGradientKernel Benchmark (1 epoch)")

        for (scale, n, expected) in expectedSpeedups {
            let embedding = generateEmbeddings(n: n, d: outputDimension)

            var edges: [UMAPEdge] = []
            for i in 0..<n {
                for _ in 0..<kNeighbors {
                    let j = Int.random(in: 0..<n)
                    if i != j {
                        edges.append(UMAPEdge(source: i, target: j, weight: Float.random(in: 0.1...1.0)))
                    }
                }
            }

            let sortedEdges = kernel.sortEdgesBySource(edges)
            let cpuEdges = sortedEdges.map { (source: Int($0.source), target: Int($0.target), weight: $0.weight) }

            // CPU benchmark
            let cpuMs = measureCPU(iterations: cpuIterations) {
                _ = CPUBaselines.umapGradient(embedding: embedding, edges: cpuEdges)
            }

            // GPU benchmark
            let gpuMs = try await measureGPU(iterations: gpuIterations, warmup: warmupIterations) {
                var embeddingCopy = embedding
                try await kernel.optimizeEpoch(
                    embedding: &embeddingCopy,
                    edges: sortedEdges,
                    params: .default
                )
            }

            let speedup = cpuMs / gpuMs
            printRow(scale: scale, cpuMs: cpuMs, gpuMs: gpuMs, speedup: speedup, expected: expected)
        }

        printTableFooter()
    }

    public func benchmarkSparseLogTFIDF() async throws {
        print("Initializing SparseLogTFIDFKernel...")
        let context = try await Metal4Context()
        let kernel = try await SparseLogTFIDFKernel(context: context)

        let vocabSize = 10_000
        let termsPerCluster = 100

        let expectedSpeedups: [(scale: String, numClusters: Int, expected: Double)] = [
            ("50 clusters", 50, 3),
            ("100 clusters", 100, 5),
            ("200 clusters", 200, 8)
        ]

        printTableHeader("SparseLogTFIDFKernel Benchmark")

        for (scale, numClusters, expected) in expectedSpeedups {
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

            let corpusFreqs = (0..<vocabSize).map { _ in Float.random(in: 1...1000) }
            let avgClusterSize = Float(500)

            let cpuTermFreqs = clusterTerms.map { cluster in
                zip(cluster.termIndices, cluster.frequencies).map { (Int($0), $1) }
            }

            // CPU benchmark
            let cpuMs = measureCPU(iterations: cpuIterations * 2) {
                _ = CPUBaselines.ctfidf(
                    termFreqs: cpuTermFreqs,
                    corpusFreqs: corpusFreqs,
                    avgClusterSize: avgClusterSize
                )
            }

            // GPU benchmark
            let gpuMs = try await measureGPU(iterations: gpuIterations * 2, warmup: warmupIterations) {
                _ = try await kernel.compute(
                    clusterTerms: clusterTerms,
                    corpusFrequencies: corpusFreqs,
                    avgClusterSize: avgClusterSize
                )
            }

            let speedup = cpuMs / gpuMs
            printRow(scale: scale, cpuMs: cpuMs, gpuMs: gpuMs, speedup: speedup, expected: expected)
        }

        printTableFooter()
    }
}

// MARK: - Command Line Entry Point

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public func runSwiftTopicsBenchmarks() async throws {
    let runner = SwiftTopicsBenchmarkRunner()
    try await runner.runAll()
}
