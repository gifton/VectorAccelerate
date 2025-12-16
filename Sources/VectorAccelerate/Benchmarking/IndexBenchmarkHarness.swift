//
//  IndexBenchmarkHarness.swift
//  VectorAccelerate
//
//  ANN-benchmarks style evaluation framework for AcceleratedVectorIndex.
//
//  Provides:
//  - Recall@K vs brute force
//  - Latency p50/p95/p99
//  - Throughput (queries/second)
//  - Memory footprint
//  - Build time
//

import Foundation
import QuartzCore

// MARK: - Index Benchmark Result

/// Comprehensive benchmark result for index operations.
public struct IndexBenchmarkResult: Sendable {
    /// Benchmark identifier
    public let name: String

    /// Configuration used
    public let configuration: IndexBenchmarkConfiguration

    /// Recall@K metric (fraction of ground truth in top-k results)
    public let recall: Float

    /// Latency statistics in seconds
    public let latencyStats: LatencyStats

    /// Throughput in queries per second
    public let throughput: Double

    /// Memory usage in bytes
    public let memoryUsageBytes: Int

    /// Build time in seconds (for index construction)
    public let buildTimeSeconds: TimeInterval

    /// Additional metrics
    public let additionalMetrics: [String: Double]

    public init(
        name: String,
        configuration: IndexBenchmarkConfiguration,
        recall: Float,
        latencyStats: LatencyStats,
        throughput: Double,
        memoryUsageBytes: Int,
        buildTimeSeconds: TimeInterval,
        additionalMetrics: [String: Double] = [:]
    ) {
        self.name = name
        self.configuration = configuration
        self.recall = recall
        self.latencyStats = latencyStats
        self.throughput = throughput
        self.memoryUsageBytes = memoryUsageBytes
        self.buildTimeSeconds = buildTimeSeconds
        self.additionalMetrics = additionalMetrics
    }
}

// MARK: - Latency Statistics

/// Latency percentile statistics.
public struct LatencyStats: Sendable {
    /// Minimum latency
    public let min: TimeInterval

    /// 50th percentile (median)
    public let p50: TimeInterval

    /// 95th percentile
    public let p95: TimeInterval

    /// 99th percentile
    public let p99: TimeInterval

    /// Maximum latency
    public let max: TimeInterval

    /// Mean latency
    public let mean: TimeInterval

    /// Standard deviation
    public let stdDev: TimeInterval

    /// All raw samples (for further analysis)
    public let samples: [TimeInterval]

    public init(samples: [TimeInterval]) {
        let sorted = samples.sorted()
        let count = sorted.count

        self.samples = samples
        self.min = sorted.first ?? 0
        self.max = sorted.last ?? 0

        // Percentiles
        self.p50 = Self.percentile(sorted, 0.50)
        self.p95 = Self.percentile(sorted, 0.95)
        self.p99 = Self.percentile(sorted, 0.99)

        // Mean and std dev
        let sum = samples.reduce(0, +)
        let meanValue = count > 0 ? sum / Double(count) : 0
        self.mean = meanValue

        let variance = count > 0 ? samples.reduce(0) { $0 + pow($1 - meanValue, 2) } / Double(count) : 0
        self.stdDev = sqrt(variance)
    }

    private static func percentile(_ sorted: [TimeInterval], _ p: Double) -> TimeInterval {
        guard !sorted.isEmpty else { return 0 }
        let index = Int(Double(sorted.count - 1) * p)
        return sorted[Swift.min(index, sorted.count - 1)]
    }
}

// MARK: - Benchmark Configuration

/// Configuration for index benchmarks.
public struct IndexBenchmarkConfiguration: Sendable {
    /// Vector dimension
    public let dimension: Int

    /// Number of vectors in the index
    public let datasetSize: Int

    /// Number of queries to run
    public let numQueries: Int

    /// K value for search
    public let k: Int

    /// Index type to benchmark
    public let indexType: IndexConfiguration.IndexType

    /// Number of warmup iterations
    public let warmupIterations: Int

    /// Number of measurement iterations
    public let measurementIterations: Int

    /// Whether to compute recall (requires ground truth computation)
    public let computeRecall: Bool

    /// Random seed for reproducibility
    public let seed: UInt64

    public init(
        dimension: Int = 128,
        datasetSize: Int = 10_000,
        numQueries: Int = 100,
        k: Int = 10,
        indexType: IndexConfiguration.IndexType = .flat,
        warmupIterations: Int = 5,
        measurementIterations: Int = 50,
        computeRecall: Bool = true,
        seed: UInt64 = 42
    ) {
        self.dimension = dimension
        self.datasetSize = datasetSize
        self.numQueries = numQueries
        self.k = k
        self.indexType = indexType
        self.warmupIterations = warmupIterations
        self.measurementIterations = measurementIterations
        self.computeRecall = computeRecall
        self.seed = seed
    }

    /// Predefined configuration for small-scale testing
    public static let small = IndexBenchmarkConfiguration(
        dimension: 64,
        datasetSize: 1_000,
        numQueries: 50,
        k: 10
    )

    /// Predefined configuration for medium-scale testing
    public static let medium = IndexBenchmarkConfiguration(
        dimension: 128,
        datasetSize: 10_000,
        numQueries: 100,
        k: 10
    )

    /// Predefined configuration for large-scale testing
    public static let large = IndexBenchmarkConfiguration(
        dimension: 384,
        datasetSize: 100_000,
        numQueries: 200,
        k: 10
    )
}

// MARK: - Benchmark Workload

/// Types of benchmark workloads.
public enum BenchmarkWorkload: Sendable {
    /// Pure search on a static index
    case pureSearch

    /// Mixed insert/search operations
    case mixedInsertSearch(insertRatio: Double)

    /// Mixed insert/search/delete operations
    case mixedAll(insertRatio: Double, deleteRatio: Double)

    /// Batch search (multiple queries at once)
    case batchSearch(batchSize: Int)
}

// MARK: - Index Benchmark Harness

/// ANN-benchmarks style evaluation harness for AcceleratedVectorIndex.
public actor IndexBenchmarkHarness {

    private var rng: SeededRandomNumberGenerator

    public init(seed: UInt64 = 42) {
        self.rng = SeededRandomNumberGenerator(seed: seed)
    }

    // MARK: - Main Benchmark API

    /// Run a comprehensive benchmark suite.
    ///
    /// - Parameter configurations: Array of configurations to benchmark
    /// - Returns: Array of benchmark results
    public func runBenchmarkSuite(
        configurations: [IndexBenchmarkConfiguration]
    ) async throws -> [IndexBenchmarkResult] {
        var results: [IndexBenchmarkResult] = []

        for config in configurations {
            let result = try await runBenchmark(configuration: config)
            results.append(result)
        }

        return results
    }

    /// Run a single benchmark with the given configuration.
    ///
    /// - Parameter configuration: Benchmark configuration
    /// - Returns: Benchmark result
    public func runBenchmark(
        configuration: IndexBenchmarkConfiguration
    ) async throws -> IndexBenchmarkResult {
        let startBuild = CACurrentMediaTime()

        // Generate dataset
        let dataset = generateDataset(
            count: configuration.datasetSize,
            dimension: configuration.dimension
        )

        // Generate queries
        let queries = generateDataset(
            count: configuration.numQueries,
            dimension: configuration.dimension
        )

        // Build index
        let index = try await buildIndex(
            dataset: dataset,
            configuration: configuration
        )

        let buildTime = CACurrentMediaTime() - startBuild

        // Compute ground truth if needed
        var groundTruth: [[Int]]?
        if configuration.computeRecall {
            groundTruth = await computeGroundTruth(
                queries: queries,
                dataset: dataset,
                k: configuration.k
            )
        }

        // Warmup
        for _ in 0..<configuration.warmupIterations {
            for query in queries.prefix(10) {
                _ = try await index.search(query: query, k: configuration.k)
            }
        }

        // Measure latency
        var latencySamples: [TimeInterval] = []
        latencySamples.reserveCapacity(configuration.numQueries * configuration.measurementIterations)

        var allResults: [[[IndexSearchResult]]] = []

        for _ in 0..<configuration.measurementIterations {
            var iterationResults: [[IndexSearchResult]] = []

            for query in queries {
                let start = CACurrentMediaTime()
                let results = try await index.search(query: query, k: configuration.k)
                let elapsed = CACurrentMediaTime() - start

                latencySamples.append(elapsed)
                iterationResults.append(results)
            }

            allResults.append(iterationResults)
        }

        // Calculate recall
        let recall: Float
        if let gt = groundTruth {
            recall = await calculateRecall(
                results: allResults[0],  // Use first iteration's results
                groundTruth: gt,
                index: index
            )
        } else {
            recall = 1.0  // Assume perfect recall if not computing
        }

        // Calculate metrics
        let latencyStats = LatencyStats(samples: latencySamples)
        let totalQueries = Double(configuration.numQueries * configuration.measurementIterations)
        let totalTime = latencySamples.reduce(0, +)
        let throughput = totalTime > 0 ? totalQueries / totalTime : 0

        // Estimate memory usage
        let stats = await index.statistics()
        let memoryBytes = stats.gpuVectorMemoryBytes + stats.cpuMetadataMemoryBytes

        return IndexBenchmarkResult(
            name: benchmarkName(for: configuration),
            configuration: configuration,
            recall: recall,
            latencyStats: latencyStats,
            throughput: throughput,
            memoryUsageBytes: memoryBytes,
            buildTimeSeconds: buildTime,
            additionalMetrics: [
                "vectorCount": Double(stats.vectorCount),
                "deletedSlots": Double(stats.deletedSlots)
            ]
        )
    }

    /// Run a workload-specific benchmark.
    ///
    /// - Parameters:
    ///   - workload: Type of workload to benchmark
    ///   - configuration: Base configuration
    /// - Returns: Benchmark result
    public func runWorkloadBenchmark(
        workload: BenchmarkWorkload,
        configuration: IndexBenchmarkConfiguration
    ) async throws -> IndexBenchmarkResult {
        switch workload {
        case .pureSearch:
            return try await runBenchmark(configuration: configuration)

        case .batchSearch(let batchSize):
            return try await runBatchBenchmark(
                batchSize: batchSize,
                configuration: configuration
            )

        case .mixedInsertSearch(let insertRatio):
            return try await runMixedBenchmark(
                insertRatio: insertRatio,
                deleteRatio: 0,
                configuration: configuration
            )

        case .mixedAll(let insertRatio, let deleteRatio):
            return try await runMixedBenchmark(
                insertRatio: insertRatio,
                deleteRatio: deleteRatio,
                configuration: configuration
            )
        }
    }

    // MARK: - Specialized Benchmarks

    /// Run batch search benchmark.
    private func runBatchBenchmark(
        batchSize: Int,
        configuration: IndexBenchmarkConfiguration
    ) async throws -> IndexBenchmarkResult {
        let startBuild = CACurrentMediaTime()

        // Generate dataset
        let dataset = generateDataset(
            count: configuration.datasetSize,
            dimension: configuration.dimension
        )

        // Generate queries in batches
        let numBatches = (configuration.numQueries + batchSize - 1) / batchSize
        var queryBatches: [[[Float]]] = []
        for _ in 0..<numBatches {
            let batch = generateDataset(count: batchSize, dimension: configuration.dimension)
            queryBatches.append(batch)
        }

        // Build index
        let index = try await buildIndex(dataset: dataset, configuration: configuration)
        let buildTime = CACurrentMediaTime() - startBuild

        // Warmup
        for batch in queryBatches.prefix(2) {
            _ = try await index.search(queries: batch, k: configuration.k)
        }

        // Measure batch latency
        var latencySamples: [TimeInterval] = []

        for _ in 0..<configuration.measurementIterations {
            for batch in queryBatches {
                let start = CACurrentMediaTime()
                _ = try await index.search(queries: batch, k: configuration.k)
                let elapsed = CACurrentMediaTime() - start

                // Record per-query latency
                let perQueryLatency = elapsed / Double(batch.count)
                for _ in 0..<batch.count {
                    latencySamples.append(perQueryLatency)
                }
            }
        }

        let latencyStats = LatencyStats(samples: latencySamples)
        let totalQueries = Double(numBatches * batchSize * configuration.measurementIterations)
        let totalTime = latencySamples.reduce(0, +)
        let throughput = totalTime > 0 ? totalQueries / totalTime : 0

        let stats = await index.statistics()
        let memoryBytes = stats.gpuVectorMemoryBytes + stats.cpuMetadataMemoryBytes

        return IndexBenchmarkResult(
            name: "BatchSearch-\(batchSize)-\(benchmarkName(for: configuration))",
            configuration: configuration,
            recall: 1.0,  // Skip recall for batch benchmark
            latencyStats: latencyStats,
            throughput: throughput,
            memoryUsageBytes: memoryBytes,
            buildTimeSeconds: buildTime,
            additionalMetrics: [
                "batchSize": Double(batchSize),
                "numBatches": Double(numBatches)
            ]
        )
    }

    /// Run mixed workload benchmark.
    private func runMixedBenchmark(
        insertRatio: Double,
        deleteRatio: Double,
        configuration: IndexBenchmarkConfiguration
    ) async throws -> IndexBenchmarkResult {
        let startBuild = CACurrentMediaTime()

        // Start with half the dataset
        let initialSize = configuration.datasetSize / 2
        let initialDataset = generateDataset(
            count: initialSize,
            dimension: configuration.dimension
        )

        // Build initial index
        let index = try await buildIndex(dataset: initialDataset, configuration: configuration)
        let buildTime = CACurrentMediaTime() - startBuild

        // Generate additional vectors for inserts
        let insertVectors = generateDataset(
            count: configuration.datasetSize / 2,
            dimension: configuration.dimension
        )

        // Generate queries
        let queries = generateDataset(
            count: configuration.numQueries,
            dimension: configuration.dimension
        )

        // Track handles for deletion
        var handles = await index.allHandles()
        var insertIndex = 0

        // Mixed operations
        var latencySamples: [TimeInterval] = []
        let totalOperations = configuration.numQueries * configuration.measurementIterations

        for _ in 0..<totalOperations {
            let roll = Float.random(in: 0...1, using: &rng)

            if roll < Float(insertRatio) && insertIndex < insertVectors.count {
                // Insert operation
                let start = CACurrentMediaTime()
                let handle = try await index.insert(insertVectors[insertIndex])
                let elapsed = CACurrentMediaTime() - start

                latencySamples.append(elapsed)
                handles.append(handle)
                insertIndex += 1

            } else if roll < Float(insertRatio + deleteRatio) && handles.count > 10 {
                // Delete operation
                let deleteIndex = Int.random(in: 0..<handles.count, using: &rng)
                let handle = handles[deleteIndex]

                let start = CACurrentMediaTime()
                try await index.remove(handle)
                let elapsed = CACurrentMediaTime() - start

                latencySamples.append(elapsed)
                handles.remove(at: deleteIndex)

            } else {
                // Search operation
                let queryIndex = Int.random(in: 0..<queries.count, using: &rng)
                let query = queries[queryIndex]

                let start = CACurrentMediaTime()
                _ = try await index.search(query: query, k: configuration.k)
                let elapsed = CACurrentMediaTime() - start

                latencySamples.append(elapsed)
            }
        }

        let latencyStats = LatencyStats(samples: latencySamples)
        let totalTime = latencySamples.reduce(0, +)
        let throughput = totalTime > 0 ? Double(totalOperations) / totalTime : 0

        let stats = await index.statistics()
        let memoryBytes = stats.gpuVectorMemoryBytes + stats.cpuMetadataMemoryBytes

        return IndexBenchmarkResult(
            name: "Mixed-\(Int(insertRatio*100))i\(Int(deleteRatio*100))d-\(benchmarkName(for: configuration))",
            configuration: configuration,
            recall: 1.0,  // Skip recall for mixed benchmark
            latencyStats: latencyStats,
            throughput: throughput,
            memoryUsageBytes: memoryBytes,
            buildTimeSeconds: buildTime,
            additionalMetrics: [
                "insertRatio": insertRatio,
                "deleteRatio": deleteRatio,
                "finalVectorCount": Double(stats.vectorCount)
            ]
        )
    }

    // MARK: - Helper Methods

    private func buildIndex(
        dataset: [[Float]],
        configuration: IndexBenchmarkConfiguration
    ) async throws -> AcceleratedVectorIndex {
        let indexConfig: IndexConfiguration

        switch configuration.indexType {
        case .flat:
            indexConfig = .flat(
                dimension: configuration.dimension,
                capacity: configuration.datasetSize * 2
            )
        case .ivf(let nlist, let nprobe, let minTrainingVectors):
            indexConfig = .ivf(
                dimension: configuration.dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: configuration.datasetSize * 2,
                minTrainingVectors: minTrainingVectors
            )
        }

        let index = try await AcceleratedVectorIndex(configuration: indexConfig)

        // Batch insert
        _ = try await index.insert(dataset)

        return index
    }

    private func generateDataset(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }
    }

    private func computeGroundTruth(
        queries: [[Float]],
        dataset: [[Float]],
        k: Int
    ) async -> [[Int]] {
        // Brute-force ground truth computation
        queries.map { query in
            let distances = dataset.enumerated().map { (index, vector) in
                (index: index, distance: l2Distance(query, vector))
            }
            let sorted = distances.sorted { $0.distance < $1.distance }
            return Array(sorted.prefix(k).map { $0.index })
        }
    }

    private func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { sum, pair in
            let diff = pair.0 - pair.1
            return sum + diff * diff
        }
    }

    private func calculateRecall(
        results: [[IndexSearchResult]],
        groundTruth: [[Int]],
        index: AcceleratedVectorIndex
    ) async -> Float {
        guard results.count == groundTruth.count else { return 0 }

        var totalRecall: Float = 0

        for (searchResults, gt) in zip(results, groundTruth) {
            // Convert handles to slots for comparison using public API
            var resultSlots = Set<UInt32>()
            for result in searchResults {
                if let slot = await index.slot(for: result.handle) {
                    resultSlots.insert(slot)
                }
            }

            let gtSet = Set(gt.map { UInt32($0) })
            let intersection = resultSlots.intersection(gtSet)

            let recall = Float(intersection.count) / Float(gt.count)
            totalRecall += recall
        }

        return totalRecall / Float(results.count)
    }

    private func benchmarkName(for config: IndexBenchmarkConfiguration) -> String {
        let indexName: String
        switch config.indexType {
        case .flat:
            indexName = "Flat"
        case .ivf(let nlist, let nprobe, _):
            indexName = "IVF\(nlist)p\(nprobe)"
        }

        return "\(indexName)-D\(config.dimension)-N\(config.datasetSize)-K\(config.k)"
    }
}

// MARK: - Seeded Random Number Generator

/// Deterministic random number generator for reproducibility.
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// MARK: - Benchmark Report Extension

extension BenchmarkReport {

    /// Generate console report for index benchmarks.
    public static func indexReport(results: [IndexBenchmarkResult]) -> String {
        var report = """
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                    VectorAccelerate Index Benchmark Report                   ║
        ╠══════════════════════════════════════════════════════════════════════════════╣

        """

        for result in results {
            report += """
            ┌──────────────────────────────────────────────────────────────────────────────┐
            │ \(result.name.padding(toLength: 76, withPad: " ", startingAt: 0)) │
            ├──────────────────────────────────────────────────────────────────────────────┤
            │ Recall@K:      \(String(format: "%.2f%%", result.recall * 100).padding(toLength: 62, withPad: " ", startingAt: 0)) │
            │ Throughput:    \(String(format: "%.0f queries/sec", result.throughput).padding(toLength: 62, withPad: " ", startingAt: 0)) │
            │ Build Time:    \(String(format: "%.3f sec", result.buildTimeSeconds).padding(toLength: 62, withPad: " ", startingAt: 0)) │
            │ Memory:        \(formatBytes(result.memoryUsageBytes).padding(toLength: 62, withPad: " ", startingAt: 0)) │
            ├──────────────────────────────────────────────────────────────────────────────┤
            │ Latency Percentiles:                                                         │
            │   p50:  \(String(format: "%.3f ms", result.latencyStats.p50 * 1000).padding(toLength: 70, withPad: " ", startingAt: 0)) │
            │   p95:  \(String(format: "%.3f ms", result.latencyStats.p95 * 1000).padding(toLength: 70, withPad: " ", startingAt: 0)) │
            │   p99:  \(String(format: "%.3f ms", result.latencyStats.p99 * 1000).padding(toLength: 70, withPad: " ", startingAt: 0)) │
            └──────────────────────────────────────────────────────────────────────────────┘

            """
        }

        return report
    }

    /// Generate JSON report for index benchmarks.
    public static func indexJsonReport(results: [IndexBenchmarkResult]) throws -> Data {
        let reportData = results.map { result in
            [
                "name": result.name,
                "recall": result.recall,
                "throughput": result.throughput,
                "buildTimeSeconds": result.buildTimeSeconds,
                "memoryUsageBytes": result.memoryUsageBytes,
                "latency": [
                    "p50": result.latencyStats.p50,
                    "p95": result.latencyStats.p95,
                    "p99": result.latencyStats.p99,
                    "mean": result.latencyStats.mean,
                    "min": result.latencyStats.min,
                    "max": result.latencyStats.max
                ],
                "configuration": [
                    "dimension": result.configuration.dimension,
                    "datasetSize": result.configuration.datasetSize,
                    "numQueries": result.configuration.numQueries,
                    "k": result.configuration.k
                ],
                "additionalMetrics": result.additionalMetrics
            ] as [String: Any]
        }

        return try JSONSerialization.data(withJSONObject: reportData, options: .prettyPrinted)
    }

    private static func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB"]
        var value = Double(bytes)
        var unitIndex = 0

        while value >= 1024 && unitIndex < units.count - 1 {
            value /= 1024
            unitIndex += 1
        }

        return String(format: "%.2f %@", value, units[unitIndex])
    }
}
