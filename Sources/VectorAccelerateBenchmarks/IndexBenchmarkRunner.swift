//
//  IndexBenchmarkRunner.swift
//  VectorAccelerateBenchmarks
//
//  Runner for the IndexBenchmarkHarness
//

import Foundation
import VectorAccelerate

/// Run the index benchmark harness and print results
public struct IndexBenchmarkRunner {

    public static func run() async throws {
        print("=" .repeated(80))
        print("VectorAccelerate Index Benchmark Harness")
        print("=" .repeated(80))
        print()

        let harness = IndexBenchmarkHarness(seed: 42)

        // Define configurations to test (optimized for quick results)
        let configurations: [IndexBenchmarkConfiguration] = [
            // Small flat index
            IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 500,
                numQueries: 20,
                k: 10,
                indexType: .flat,
                warmupIterations: 2,
                measurementIterations: 10,
                computeRecall: true,
                seed: 42
            ),
            // Medium flat index
            IndexBenchmarkConfiguration(
                dimension: 128,
                datasetSize: 1_000,
                numQueries: 20,
                k: 10,
                indexType: .flat,
                warmupIterations: 2,
                measurementIterations: 10,
                computeRecall: true,
                seed: 42
            ),
            // Small IVF index
            IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 500,
                numQueries: 20,
                k: 10,
                indexType: .ivf(nlist: 8, nprobe: 4, minTrainingVectors: 50),
                warmupIterations: 2,
                measurementIterations: 10,
                computeRecall: true,
                seed: 42
            ),
            // Medium IVF index
            IndexBenchmarkConfiguration(
                dimension: 128,
                datasetSize: 1_000,
                numQueries: 20,
                k: 10,
                indexType: .ivf(nlist: 8, nprobe: 4, minTrainingVectors: 100),
                warmupIterations: 2,
                measurementIterations: 10,
                computeRecall: true,
                seed: 42
            ),
        ]

        print("Running \(configurations.count) benchmark configurations...")
        print()

        var results: [IndexBenchmarkResult] = []

        for (index, config) in configurations.enumerated() {
            print("[\(index + 1)/\(configurations.count)] Running: \(configDescription(config))...")

            do {
                let result = try await harness.runBenchmark(configuration: config)
                results.append(result)
                printResultSummary(result)
            } catch {
                print("  ERROR: \(error)")
            }
            print()
        }

        // Print full report
        print()
        print(BenchmarkReport.indexReport(results: results))

        // Also test batch workload
        print("=" .repeated(80))
        print("Batch Search Workload Comparison")
        print("=" .repeated(80))
        print()

        let batchConfig = IndexBenchmarkConfiguration(
            dimension: 128,
            datasetSize: 1_000,
            numQueries: 40,
            k: 10,
            indexType: .flat,
            warmupIterations: 2,
            measurementIterations: 5,
            computeRecall: false,
            seed: 42
        )

        print("Single query workload...")
        let singleResult = try await harness.runWorkloadBenchmark(
            workload: .pureSearch,
            configuration: batchConfig
        )
        printResultSummary(singleResult)

        print()
        print("Batch query workload (batch=10)...")
        let batchResult = try await harness.runWorkloadBenchmark(
            workload: .batchSearch(batchSize: 10),
            configuration: batchConfig
        )
        printResultSummary(batchResult)

        print()
        print("Speedup from batching: \(String(format: "%.2fx", singleResult.latencyStats.p50 / batchResult.latencyStats.p50))")
    }

    private static func configDescription(_ config: IndexBenchmarkConfiguration) -> String {
        let indexName: String
        switch config.indexType {
        case .flat:
            indexName = "Flat"
        case .ivf(let nlist, let nprobe, _):
            indexName = "IVF(nlist=\(nlist), nprobe=\(nprobe))"
        }
        return "\(indexName) D=\(config.dimension) N=\(config.datasetSize) K=\(config.k)"
    }

    private static func printResultSummary(_ result: IndexBenchmarkResult) {
        print("  Recall@K:    \(String(format: "%.1f%%", result.recall * 100))")
        print("  Throughput:  \(String(format: "%.0f", result.throughput)) queries/sec")
        print("  Latency p50: \(String(format: "%.3f", result.latencyStats.p50 * 1000)) ms")
        print("  Latency p95: \(String(format: "%.3f", result.latencyStats.p95 * 1000)) ms")
        print("  Latency p99: \(String(format: "%.3f", result.latencyStats.p99 * 1000)) ms")
        print("  Build time:  \(String(format: "%.3f", result.buildTimeSeconds)) sec")
        print("  Memory:      \(formatBytes(result.memoryUsageBytes))")
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

private extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
