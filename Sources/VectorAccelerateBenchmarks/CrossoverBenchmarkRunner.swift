//
//  CrossoverBenchmarkRunner.swift
//  VectorAccelerateBenchmarks
//
//  Sweeps batch sizes to find the CPU/GPU breakeven point for each
//  distance metric and dimension. Outputs the crossover batch size
//  where GPU becomes faster than CPU.
//

import Foundation
import VectorAccelerate
import VectorCore

public struct CrossoverBenchmarkRunner {

    // MARK: - Configuration

    /// Dimensions to test (includes optimized kernel boundaries)
    private static let dimensions = [128, 384, 768]

    /// Batch sizes to sweep (logarithmic scale)
    private static let batchSizes = [1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000]

    /// Iterations per measurement point
    private static let iterations = 20

    /// Warmup iterations
    private static let warmup = 5

    // MARK: - Result Types

    struct CrossoverResult {
        let metric: String
        let dimension: Int
        let measurements: [BatchMeasurement]
        let crossoverBatchSize: Int?
    }

    struct BatchMeasurement {
        let batchSize: Int
        let cpuMedianMs: Double
        let gpuMedianMs: Double
        let gpuSpeedup: Double
    }

    // MARK: - Entry Point

    public static func run() async throws {
        print(String(repeating: "=", count: 80))
        print("VectorAccelerate CPU/GPU Crossover Benchmark")
        print(String(repeating: "=", count: 80))
        print()

        guard ComputeDevice.gpu().isAvailable else {
            print("Metal is not available on this system")
            return
        }

        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)
        let batchEngine = try await BatchDistanceEngine(context: context)

        print("Sweeping batch sizes: \(batchSizes)")
        print("Dimensions: \(dimensions)")
        print("Iterations per point: \(iterations) (warmup: \(warmup))")
        print()

        var results: [CrossoverResult] = []

        // Euclidean distance crossover
        for dim in dimensions {
            let result = try await measureCrossover(
                metric: "Euclidean",
                dimension: dim,
                engine: engine,
                batchEngine: batchEngine,
                cpuOp: { query, candidates in
                    AccelerateFallback.batchEuclideanDistance(query: query, candidates: candidates)
                },
                gpuOp: { query, candidates in
                    try await batchEngine.batchEuclideanDistance(query: query, candidates: candidates, useGPU: true)
                }
            )
            results.append(result)
            printResult(result)
        }

        // Cosine similarity crossover
        for dim in dimensions {
            let result = try await measureCrossover(
                metric: "Cosine",
                dimension: dim,
                engine: engine,
                batchEngine: batchEngine,
                cpuOp: { query, candidates in
                    AccelerateFallback.batchCosineSimilarity(query: query, candidates: candidates)
                },
                gpuOp: { query, candidates in
                    try await batchEngine.batchCosineSimilarity(query: query, candidates: candidates, useGPU: true)
                }
            )
            results.append(result)
            printResult(result)
        }

        // Dot product crossover
        for dim in dimensions {
            let result = try await measureCrossover(
                metric: "DotProduct",
                dimension: dim,
                engine: engine,
                batchEngine: batchEngine,
                cpuOp: { query, candidates in
                    AccelerateFallback.batchDotProduct(query: query, candidates: candidates)
                },
                gpuOp: { query, candidates in
                    try await batchEngine.batchDotProduct(query: query, candidates: candidates, useGPU: true)
                }
            )
            results.append(result)
            printResult(result)
        }

        // Summary table
        printSummary(results)

        // Save JSON
        try saveJSON(results)
    }

    // MARK: - Measurement

    private static func measureCrossover(
        metric: String,
        dimension: Int,
        engine: Metal4ComputeEngine,
        batchEngine: BatchDistanceEngine,
        cpuOp: @Sendable ([Float], [[Float]]) -> [Float],
        gpuOp: @Sendable ([Float], [[Float]]) async throws -> [Float]
    ) async throws -> CrossoverResult {
        print("  \(metric) dim=\(dimension) ...", terminator: "")

        let clock = ContinuousClock()
        var measurements: [BatchMeasurement] = []
        var crossover: Int? = nil

        for batchSize in batchSizes {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
            let candidates = (0..<batchSize).map { _ in
                (0..<dimension).map { _ in Float.random(in: -1...1) }
            }

            // Measure CPU
            var cpuSamples: [Double] = []
            for i in 0..<(warmup + iterations) {
                let elapsed = clock.measure {
                    blackHole(cpuOp(query, candidates))
                }
                if i >= warmup {
                    cpuSamples.append(durationToSeconds(elapsed))
                }
            }

            // Measure GPU
            var gpuSamples: [Double] = []
            for i in 0..<(warmup + iterations) {
                let elapsed = try await clock.measure {
                    blackHole(try await gpuOp(query, candidates))
                }
                if i >= warmup {
                    gpuSamples.append(durationToSeconds(elapsed))
                }
            }

            let cpuMedian = median(cpuSamples) * 1000
            let gpuMedian = median(gpuSamples) * 1000
            let speedup = cpuMedian / gpuMedian

            measurements.append(BatchMeasurement(
                batchSize: batchSize,
                cpuMedianMs: cpuMedian,
                gpuMedianMs: gpuMedian,
                gpuSpeedup: speedup
            ))

            // Detect first crossover point where GPU is consistently faster
            if crossover == nil && speedup > 1.0 {
                crossover = batchSize
            }
        }

        print(" done")

        return CrossoverResult(
            metric: metric,
            dimension: dimension,
            measurements: measurements,
            crossoverBatchSize: crossover
        )
    }

    // MARK: - Output

    private static func printResult(_ result: CrossoverResult) {
        let crossoverStr = result.crossoverBatchSize.map { "N=\($0)" } ?? "GPU never faster"
        print("    Crossover: \(crossoverStr)")
        print()
        print("    Batch |   CPU (ms) |   GPU (ms) | Speedup")
        print("    ------|------------|------------|--------")
        for m in result.measurements {
            let marker = m.gpuSpeedup > 1.0 ? " <--" : ""
            print(String(format: "    %5d | %10.3f | %10.3f | %5.2fx%@",
                         m.batchSize, m.cpuMedianMs, m.gpuMedianMs, m.gpuSpeedup, marker))
        }
        print()
    }

    private static func printSummary(_ results: [CrossoverResult]) {
        print(String(repeating: "=", count: 80))
        print("CROSSOVER SUMMARY")
        print(String(repeating: "=", count: 80))
        print()
        print(String(format: "%-12s | %6s | %s", "Metric", "Dim", "Crossover Batch Size"))
        print(String(repeating: "-", count: 50))
        for r in results {
            let crossoverStr = r.crossoverBatchSize.map { String($0) } ?? "never"
            print(String(format: "%-12s | %6d | %@", r.metric, r.dimension, crossoverStr))
        }
        print()
        print("Crossover = smallest batch size where GPU median < CPU median.")
        print("Use these values to calibrate GPUDecisionEngine thresholds.")
        print()
    }

    private static func saveJSON(_ results: [CrossoverResult]) throws {
        let jsonResults = results.map { result -> [String: Any] in
            [
                "metric": result.metric,
                "dimension": result.dimension,
                "crossoverBatchSize": result.crossoverBatchSize as Any,
                "measurements": result.measurements.map { m -> [String: Any] in
                    [
                        "batchSize": m.batchSize,
                        "cpuMedianMs": m.cpuMedianMs,
                        "gpuMedianMs": m.gpuMedianMs,
                        "gpuSpeedup": m.gpuSpeedup
                    ]
                }
            ]
        }

        let data = try JSONSerialization.data(withJSONObject: jsonResults, options: .prettyPrinted)
        let url = URL(fileURLWithPath: "crossover_results.json")
        try data.write(to: url)
        print("Results saved to: \(url.path)")
    }

    // MARK: - Utilities

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        guard count > 0 else { return 0 }
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        }
        return sorted[count / 2]
    }

    private static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }
}
