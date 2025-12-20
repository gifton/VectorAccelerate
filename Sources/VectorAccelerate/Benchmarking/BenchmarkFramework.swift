//
//  BenchmarkFramework.swift
//  VectorAccelerate
//
//  Benchmarking infrastructure for measuring GPU acceleration performance.
//  Uses Metal 4 exclusively (iOS 26+, macOS 26+).
//

import Foundation
import VectorCore

// MARK: - Benchmark Result

/// Results from a benchmark run
public struct BenchmarkResult: Sendable {
    public let name: String
    public let samples: [TimeInterval]
    public let configuration: BenchmarkConfiguration

    public init(name: String, samples: [TimeInterval], configuration: BenchmarkConfiguration) {
        self.name = name
        self.samples = samples
        self.configuration = configuration
    }

    /// Average execution time
    public var averageTime: TimeInterval {
        samples.reduce(0, +) / Double(samples.count)
    }

    /// Median execution time (more robust than average)
    public var medianTime: TimeInterval {
        let sorted = samples.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            return sorted[count / 2]
        }
    }

    /// Standard deviation
    public var standardDeviation: TimeInterval {
        let avg = averageTime
        let variance = samples.reduce(0) { $0 + pow($1 - avg, 2) } / Double(samples.count)
        return sqrt(variance)
    }

    /// Minimum execution time
    public var minTime: TimeInterval {
        samples.min() ?? 0
    }

    /// Maximum execution time
    public var maxTime: TimeInterval {
        samples.max() ?? 0
    }

    /// Operations per second (based on median time)
    public var operationsPerSecond: Double {
        1.0 / medianTime
    }

    /// Throughput in vectors processed per second (if applicable)
    public var vectorsPerSecond: Double? {
        guard let vectorCount = configuration.additionalInfo["vectorCount"].flatMap(Double.init) else {
            return nil
        }
        return vectorCount / medianTime
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
            _ = try await engine.euclideanDistance(vectorA, vectorB)
        }
    }

    public func benchmarkCosineDistance(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vectorA = generateRandomVector(dimension: config.dimension)
        let vectorB = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Cosine Distance (dim: \(config.dimension))",
            configuration: config
        ) {
            _ = try await engine.cosineDistance(vectorA, vectorB)
        }
    }

    public func benchmarkDotProduct(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vectorA = generateRandomVector(dimension: config.dimension)
        let vectorB = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Dot Product (dim: \(config.dimension))",
            configuration: config
        ) {
            _ = try await engine.dotProduct(vectorA, vectorB)
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
            _ = try await engine.batchEuclideanDistance(query: query, candidates: candidates)
        }
    }

    public func benchmarkNormalize(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vector = generateRandomVector(dimension: config.dimension)

        return try await measurePerformance(
            name: "Vector Normalize (dim: \(config.dimension))",
            configuration: config
        ) {
            _ = try await engine.normalize(vector)
        }
    }

    public func benchmarkScale(config: BenchmarkConfiguration) async throws -> BenchmarkResult {
        let vector = generateRandomVector(dimension: config.dimension)
        let scalar: Float = 2.5

        return try await measurePerformance(
            name: "Vector Scale (dim: \(config.dimension))",
            configuration: config
        ) {
            _ = try await engine.scale(vector, by: scalar)
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
            _ = try await engine.matrixVectorMultiply(matrix: matrix, vector: vector)
        }
    }

    // MARK: - Helper Methods

    private func measurePerformance(
        name: String,
        configuration: BenchmarkConfiguration,
        operation: () async throws -> Void
    ) async throws -> BenchmarkResult {
        var samples: [TimeInterval] = []
        samples.reserveCapacity(configuration.iterations)

        // Warmup runs
        for _ in 0..<configuration.warmupIterations {
            try await operation()
        }

        // Measured runs
        for _ in 0..<configuration.iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            try await operation()
            let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
            samples.append(elapsedTime)
        }

        return BenchmarkResult(name: name, samples: samples, configuration: configuration)
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
            report += "- Median Time: \(String(format: "%.3f", result.medianTime * 1000))ms\n"
            report += "- Average Time: \(String(format: "%.3f", result.averageTime * 1000))ms\n"
            report += "- Std Deviation: \(String(format: "%.3f", result.standardDeviation * 1000))ms\n"
            report += "- Min/Max: \(String(format: "%.3f", result.minTime * 1000))/\(String(format: "%.3f", result.maxTime * 1000))ms\n"
            report += "- Operations/sec: \(String(format: "%.0f", result.operationsPerSecond))\n"

            if let vectorsPerSec = result.vectorsPerSecond {
                report += "- Vectors/sec: \(String(format: "%.0f", vectorsPerSec))\n"
            }

            report += "\n"
        }

        return report
    }

    /// Generate JSON report
    public static func jsonReport(results: [BenchmarkResult]) throws -> Data {
        let reportData = results.map { result in
            [
                "name": result.name,
                "medianTime": result.medianTime,
                "averageTime": result.averageTime,
                "standardDeviation": result.standardDeviation,
                "minTime": result.minTime,
                "maxTime": result.maxTime,
                "operationsPerSecond": result.operationsPerSecond,
                "configuration": [
                    "iterations": result.configuration.iterations,
                    "dimension": result.configuration.dimension,
                    "batchSize": result.configuration.batchSize
                ]
            ] as [String : Any]
        }

        return try JSONSerialization.data(withJSONObject: reportData, options: .prettyPrinted)
    }
}
