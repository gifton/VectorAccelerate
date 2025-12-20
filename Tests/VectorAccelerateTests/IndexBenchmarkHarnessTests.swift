//
//  IndexBenchmarkHarnessTests.swift
//  VectorAccelerateTests
//
//  Tests for the IndexBenchmarkHarness
//

import XCTest
@testable import VectorAccelerate

final class IndexBenchmarkHarnessTests: XCTestCase {

    func testBenchmarkHarnessFlatIndex() async throws {
        let harness = IndexBenchmarkHarness(seed: 42)

        let config = IndexBenchmarkConfiguration(
            dimension: 64,
            datasetSize: 200,
            numQueries: 10,
            k: 5,
            indexType: .flat,
            warmupIterations: 1,
            measurementIterations: 3,
            computeRecall: true,
            seed: 42
        )

        let result = try await harness.runBenchmark(configuration: config)

        print("\n=== Flat Index Benchmark Results ===")
        print("Name:        \(result.name)")
        print("Recall@K:    \(String(format: "%.1f%%", result.recall * 100))")
        print("Throughput:  \(String(format: "%.0f", result.throughput)) queries/sec")
        print("Latency p50: \(String(format: "%.3f", result.latencyStats.p50 * 1000)) ms")
        print("Latency p95: \(String(format: "%.3f", result.latencyStats.p95 * 1000)) ms")
        print("Latency p99: \(String(format: "%.3f", result.latencyStats.p99 * 1000)) ms")
        print("Build time:  \(String(format: "%.3f", result.buildTimeSeconds)) sec")
        print("Memory:      \(formatBytes(result.memoryUsageBytes))")

        // Verify basic sanity
        XCTAssertEqual(result.recall, 1.0, accuracy: 0.01, "Flat index should have 100% recall")
        XCTAssertGreaterThan(result.throughput, 0)
        XCTAssertGreaterThan(result.latencyStats.p50, 0)
    }

    func testBenchmarkHarnessIVFIndex() async throws {
        let harness = IndexBenchmarkHarness(seed: 42)

        let config = IndexBenchmarkConfiguration(
            dimension: 64,
            datasetSize: 200,
            numQueries: 10,
            k: 5,
            indexType: .ivf(nlist: 4, nprobe: 4, minTrainingVectors: 50),
            warmupIterations: 1,
            measurementIterations: 3,
            computeRecall: true,
            seed: 42
        )

        let result = try await harness.runBenchmark(configuration: config)

        print("\n=== IVF Index Benchmark Results ===")
        print("Name:        \(result.name)")
        print("Recall@K:    \(String(format: "%.1f%%", result.recall * 100))")
        print("Throughput:  \(String(format: "%.0f", result.throughput)) queries/sec")
        print("Latency p50: \(String(format: "%.3f", result.latencyStats.p50 * 1000)) ms")
        print("Latency p95: \(String(format: "%.3f", result.latencyStats.p95 * 1000)) ms")
        print("Latency p99: \(String(format: "%.3f", result.latencyStats.p99 * 1000)) ms")
        print("Build time:  \(String(format: "%.3f", result.buildTimeSeconds)) sec")
        print("Memory:      \(formatBytes(result.memoryUsageBytes))")

        // IVF with nprobe=nlist should have high recall
        XCTAssertGreaterThan(result.recall, 0.9, "IVF with nprobe=nlist should have >90% recall")
        XCTAssertGreaterThan(result.throughput, 0)
    }

    func testBenchmarkHarnessBatchWorkload() async throws {
        let harness = IndexBenchmarkHarness(seed: 42)

        let config = IndexBenchmarkConfiguration(
            dimension: 64,
            datasetSize: 200,
            numQueries: 20,
            k: 5,
            indexType: .flat,
            warmupIterations: 1,
            measurementIterations: 3,
            computeRecall: false,
            seed: 42
        )

        print("\n=== Batch vs Single Query Comparison ===")

        let singleResult = try await harness.runWorkloadBenchmark(
            workload: .pureSearch,
            configuration: config
        )
        print("\nSingle query:")
        print("  Throughput:  \(String(format: "%.0f", singleResult.throughput)) queries/sec")
        print("  Latency p50: \(String(format: "%.3f", singleResult.latencyStats.p50 * 1000)) ms")

        let batchResult = try await harness.runWorkloadBenchmark(
            workload: .batchSearch(batchSize: 10),
            configuration: config
        )
        print("\nBatch query (batch=10):")
        print("  Throughput:  \(String(format: "%.0f", batchResult.throughput)) queries/sec")
        print("  Latency p50: \(String(format: "%.3f", batchResult.latencyStats.p50 * 1000)) ms")

        let speedup = singleResult.latencyStats.p50 / batchResult.latencyStats.p50
        print("\nSpeedup from batching: \(String(format: "%.2fx", speedup))")

        XCTAssertGreaterThan(speedup, 1.0, "Batching should provide some speedup")
    }

    func testBenchmarkHarnessLatencyStats() async throws {
        // Test LatencyStats computation
        let samples: [TimeInterval] = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
        let stats = LatencyStats(samples: samples)

        print("\n=== LatencyStats Test ===")
        print("Samples: \(samples.map { String(format: "%.3f", $0 * 1000) }.joined(separator: ", ")) ms")
        print("Min:     \(String(format: "%.3f", stats.min * 1000)) ms")
        print("Max:     \(String(format: "%.3f", stats.max * 1000)) ms")
        print("Mean:    \(String(format: "%.3f", stats.mean * 1000)) ms")
        print("StdDev:  \(String(format: "%.3f", stats.stdDev * 1000)) ms")
        print("p50:     \(String(format: "%.3f", stats.p50 * 1000)) ms")
        print("p95:     \(String(format: "%.3f", stats.p95 * 1000)) ms")
        print("p99:     \(String(format: "%.3f", stats.p99 * 1000)) ms")

        XCTAssertEqual(stats.min, 0.001, accuracy: 0.0001)
        XCTAssertEqual(stats.max, 0.010, accuracy: 0.0001)
        XCTAssertEqual(stats.mean, 0.0055, accuracy: 0.0001)
    }

    private func formatBytes(_ bytes: Int) -> String {
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
