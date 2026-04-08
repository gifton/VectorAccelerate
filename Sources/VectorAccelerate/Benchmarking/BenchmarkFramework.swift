//
//  BenchmarkFramework.swift
//  VectorAccelerate
//
//  Benchmarking infrastructure for measuring GPU acceleration performance.
//  Uses Metal 4 exclusively (iOS 26+, macOS 26+).
//

import Foundation
import VectorCore

// MARK: - Dead-Code Elimination Prevention

/// Prevents the compiler from eliminating benchmark operations whose results are unused.
///
/// Without this, the optimizer may remove the entire operation being benchmarked
/// if its return value is discarded (e.g., `_ = try await engine.euclideanDistance(...)`).
@inline(never)
public func blackHole<T>(_ value: T) {
    // The @inline(never) attribute ensures the compiler cannot see through this function
    // and must assume it has side effects, preventing DCE of the argument expression.
    withExtendedLifetime(value) {}
}

// MARK: - Benchmark Result

/// Results from a benchmark run
public struct BenchmarkResult: Sendable {
    public let name: String
    public let samples: [TimeInterval]
    public let configuration: BenchmarkConfiguration

    /// Latency statistics with percentiles (p50/p95/p99).
    /// Reuses `LatencyStats` from `IndexBenchmarkHarness` for consistency.
    public let latencyStats: LatencyStats

    /// GPU-side timing samples (seconds), captured from MTLCommandBuffer.gpuStartTime/gpuEndTime.
    /// Separates actual GPU compute time from Swift-side submission overhead.
    /// Empty if GPU timing was unavailable.
    public let gpuSamples: [TimeInterval]

    /// GPU latency statistics (nil if no GPU timing was captured)
    public let gpuLatencyStats: LatencyStats?

    public init(name: String, samples: [TimeInterval], configuration: BenchmarkConfiguration, gpuSamples: [TimeInterval] = []) {
        self.name = name
        self.samples = samples
        self.configuration = configuration
        self.latencyStats = LatencyStats(samples: samples)
        self.gpuSamples = gpuSamples
        self.gpuLatencyStats = gpuSamples.isEmpty ? nil : LatencyStats(samples: gpuSamples)
    }

    /// Average execution time
    public var averageTime: TimeInterval {
        latencyStats.mean
    }

    /// Median execution time (more robust than average)
    public var medianTime: TimeInterval {
        latencyStats.p50
    }

    /// Standard deviation
    public var standardDeviation: TimeInterval {
        latencyStats.stdDev
    }

    /// 95th percentile latency
    public var p95Time: TimeInterval {
        latencyStats.p95
    }

    /// 99th percentile latency
    public var p99Time: TimeInterval {
        latencyStats.p99
    }

    /// Median GPU execution time (nil if no GPU timing)
    public var gpuMedianTime: TimeInterval? {
        gpuLatencyStats?.p50
    }

    /// Submission overhead: wall-clock median minus GPU median.
    /// Represents time spent in Swift (buffer alloc, actor hops, command encoding).
    public var submissionOverhead: TimeInterval? {
        guard let gpuMedian = gpuMedianTime else { return nil }
        return medianTime - gpuMedian
    }

    /// Minimum execution time
    public var minTime: TimeInterval {
        latencyStats.min
    }

    /// Maximum execution time
    public var maxTime: TimeInterval {
        latencyStats.max
    }

    /// Operations per second (based on median time)
    public var operationsPerSecond: Double {
        medianTime > 0 ? 1.0 / medianTime : 0
    }

    /// Throughput in vectors processed per second (if applicable)
    public var vectorsPerSecond: Double? {
        guard let vectorCount = configuration.additionalInfo["vectorCount"].flatMap(Double.init) else {
            return nil
        }
        return medianTime > 0 ? vectorCount / medianTime : nil
    }
}

// MARK: - Benchmark Configuration

/// Configuration for benchmark runs
public struct BenchmarkConfiguration: Sendable {
    public let iterations: Int
    public let warmupIterations: Int
    public let dimension: Int
    public let batchSize: Int
    public let additionalInfo: [String: String]

    public init(
        iterations: Int = 100,
        warmupIterations: Int = 10,
        dimension: Int = 128,
        batchSize: Int = 1,
        additionalInfo: [String: String] = [:]
    ) {
        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.dimension = dimension
        self.batchSize = batchSize
        self.additionalInfo = additionalInfo
    }

    public static let `default` = BenchmarkConfiguration()
}

// MARK: - Benchmark Suite

/// Comprehensive benchmark suite for VectorAccelerate using Metal 4
public actor BenchmarkSuite {
    private let engine: Metal4ComputeEngine
    private let context: Metal4Context

    public init(engine: Metal4ComputeEngine, context: Metal4Context) {
        self.engine = engine
        self.context = context
    }

    /// Convenience initializer that creates its own context and engine
    public init() async throws {
        self.context = try await Metal4Context()
        self.engine = try await Metal4ComputeEngine(context: context)
    }

    /// Run all benchmarks and return results
    public func runAllBenchmarks(
        configurations: [BenchmarkConfiguration] = [.default]
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for config in configurations {
            // Distance computation benchmarks
            results.append(try await benchmarkEuclideanDistance(config: config))
            results.append(try await benchmarkCosineDistance(config: config))
            results.append(try await benchmarkDotProduct(config: config))

            // Batch operation benchmarks
            results.append(try await benchmarkBatchEuclideanDistance(config: config))

            // Vector operation benchmarks
            results.append(try await benchmarkNormalize(config: config))
            results.append(try await benchmarkScale(config: config))

            // Matrix operation benchmarks
            results.append(try await benchmarkMatrixVectorMultiply(config: config))
        }

        return results
    }

    // MARK: - Individual Benchmarks

    public func benchmarkEuclideanDistance(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vectorA = generateRandomVector(dimension: config.dimension)
        let vectorB = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Euclidean Distance (dim: \(config.dimension))",
            configuration: config
        ) {
            blackHole(try await engine.euclideanDistance(vectorA, vectorB))
        }
    }

    public func benchmarkCosineDistance(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vectorA = generateRandomVector(dimension: config.dimension)
        let vectorB = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Cosine Distance (dim: \(config.dimension))",
            configuration: config
        ) {
            blackHole(try await engine.cosineDistance(vectorA, vectorB))
        }
    }

    public func benchmarkDotProduct(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vectorA = generateRandomVector(dimension: config.dimension)
        let vectorB = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Dot Product (dim: \(config.dimension))",
            configuration: config
        ) {
            blackHole(try await engine.dotProduct(vectorA, vectorB))
        }
    }

    public func benchmarkBatchEuclideanDistance(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let query = generateRandomVector(dimension: config.dimension)
        let candidates = (0..<config.batchSize).map { _ in
            generateRandomVector(dimension: config.dimension)
        }

        let modifiedConfig = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: config.warmupIterations,
            dimension: config.dimension,
            batchSize: config.batchSize,
            additionalInfo: config.additionalInfo.merging(["vectorCount": String(config.batchSize)]) { _, new in new }
        )

        return try await measurePerformance(
            name: "Batch Euclidean Distance (dim: \(config.dimension), batch: \(config.batchSize))",
            configuration: modifiedConfig
        ) {
            blackHole(try await engine.batchEuclideanDistance(query: query, candidates: candidates))
        }
    }

    public func benchmarkNormalize(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vector = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Vector Normalize (dim: \(config.dimension))",
            configuration: config
        ) {
            blackHole(try await engine.normalize(vector))
        }
    }

    public func benchmarkScale(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vector = generateRandomVector(dimension: config.dimension)
        let scalar: Float = 2.5

        return try await measurePerformance(
            name: "Vector Scale (dim: \(config.dimension))",
            configuration: config
        ) {
            blackHole(try await engine.scale(vector, by: scalar))
        }
    }

    public func benchmarkMatrixVectorMultiply(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let rows = config.dimension
        let cols = config.dimension
        let matrix = (0..<rows).map { _ in generateRandomVector(dimension: cols) }
        let vector = generateRandomVector(dimension: cols)

        return try await measurePerformance(
            name: "Matrix-Vector Multiply (\(rows)x\(cols))",
            configuration: config
        ) {
            blackHole(try await engine.matrixVectorMultiply(matrix: matrix, vector: vector))
        }
    }

    // MARK: - Helper Methods

    private func measurePerformance(
        name: String,
        configuration: BenchmarkConfiguration,
        operation: () async throws -> Void
    ) async throws -> BenchmarkResult {
        let clock = ContinuousClock()
        var samples: [TimeInterval] = []
        var gpuSamples: [TimeInterval] = []
        samples.reserveCapacity(configuration.iterations)
        gpuSamples.reserveCapacity(configuration.iterations)

        // Warmup runs
        for _ in 0..<configuration.warmupIterations {
            try await operation()
        }

        // Measured runs using ContinuousClock (monotonic, not affected by NTP/system clock changes)
        for _ in 0..<configuration.iterations {
            let elapsed = try await clock.measure {
                try await operation()
            }
            samples.append(Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18)

            // Capture GPU-side timing from the most recent command buffer
            if let gpuTiming = await context.lastGPUTiming {
                gpuSamples.append(gpuTiming.duration)
            }
        }

        return BenchmarkResult(name: name, samples: samples, configuration: configuration, gpuSamples: gpuSamples)
    }

    private func generateRandomVector(dimension: Int) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }
}

// MARK: - Benchmark Comparison

/// Compare CPU vs GPU performance
public struct PerformanceComparison: Sendable {
    public let cpuResult: BenchmarkResult
    public let gpuResult: BenchmarkResult

    public init(cpu: BenchmarkResult, gpu: BenchmarkResult) {
        self.cpuResult = cpu
        self.gpuResult = gpu
    }

    /// Speedup factor (GPU vs CPU)
    public var speedupFactor: Double {
        cpuResult.medianTime / gpuResult.medianTime
    }

    /// Performance improvement percentage
    public var improvementPercent: Double {
        ((cpuResult.medianTime - gpuResult.medianTime) / cpuResult.medianTime) * 100
    }

    /// Whether GPU is faster
    public var gpuIsFaster: Bool {
        speedupFactor > 1.0
    }
}

// MARK: - Benchmark Report

/// Generate formatted benchmark reports
public struct BenchmarkReport {

    /// Generate console-friendly report
    public static func consoleReport(results: [BenchmarkResult]) -> String {
        var report = "# VectorAccelerate Benchmark Results (Metal 4)\n\n"

        for result in results {
            report += "## \(result.name)\n"
            report += "- Iterations: \(result.configuration.iterations)\n"
            report += "- Wall-clock p50: \(String(format: "%.3f", result.medianTime * 1000))ms\n"
            report += "- Wall-clock p95: \(String(format: "%.3f", result.p95Time * 1000))ms\n"
            report += "- Wall-clock p99: \(String(format: "%.3f", result.p99Time * 1000))ms\n"

            if let gpuStats = result.gpuLatencyStats {
                report += "- GPU compute p50: \(String(format: "%.3f", gpuStats.p50 * 1000))ms\n"
                report += "- GPU compute p95: \(String(format: "%.3f", gpuStats.p95 * 1000))ms\n"
                if let overhead = result.submissionOverhead {
                    report += "- Submission overhead: \(String(format: "%.3f", overhead * 1000))ms\n"
                }
            }

            report += "- Mean:         \(String(format: "%.3f", result.averageTime * 1000))ms\n"
            report += "- Std Dev:      \(String(format: "%.3f", result.standardDeviation * 1000))ms\n"
            report += "- Min/Max:      \(String(format: "%.3f", result.minTime * 1000))/\(String(format: "%.3f", result.maxTime * 1000))ms\n"
            report += "- Ops/sec:      \(String(format: "%.0f", result.operationsPerSecond))\n"

            if let vectorsPerSec = result.vectorsPerSecond {
                report += "- Vectors/sec:  \(String(format: "%.0f", vectorsPerSec))\n"
            }

            report += "\n"
        }

        return report
    }

    /// Generate JSON report
    public static func jsonReport(results: [BenchmarkResult]) throws -> Data {
        let reportData = results.map { result in
            var entry: [String: Any] = [
                "name": result.name,
                "wallClock": [
                    "p50": result.medianTime,
                    "p95": result.p95Time,
                    "p99": result.p99Time,
                    "mean": result.averageTime,
                    "stdDev": result.standardDeviation,
                    "min": result.minTime,
                    "max": result.maxTime
                ],
                "operationsPerSecond": result.operationsPerSecond,
                "configuration": [
                    "iterations": result.configuration.iterations,
                    "dimension": result.configuration.dimension,
                    "batchSize": result.configuration.batchSize
                ]
            ]

            if let gpuStats = result.gpuLatencyStats {
                entry["gpuCompute"] = [
                    "p50": gpuStats.p50,
                    "p95": gpuStats.p95,
                    "p99": gpuStats.p99,
                    "mean": gpuStats.mean,
                    "min": gpuStats.min,
                    "max": gpuStats.max
                ]
                if let overhead = result.submissionOverhead {
                    entry["submissionOverheadMs"] = overhead * 1000
                }
            }

            return entry
        }

        return try JSONSerialization.data(withJSONObject: reportData, options: .prettyPrinted)
    }
}
