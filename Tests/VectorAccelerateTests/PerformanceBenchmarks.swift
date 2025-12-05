//
//  PerformanceBenchmarks.swift
//  VectorAccelerate
//
//  Performance benchmarks for AcceleratedVectorIndex.
//

import XCTest
@testable import VectorAccelerate
import VectorAccelerate
import VectorCore

/// Performance benchmarks for AcceleratedVectorIndex.
final class PerformanceBenchmarks: XCTestCase {

    // MARK: - Flat Index Benchmarks

    /// Benchmark flat index search with 10K vectors, dim=128.
    func testFlatSearch10K_128D() async throws {
        let dimension = 128
        let vectorCount = 10_000
        let k = 10

        let config = IndexConfiguration.flat(dimension: dimension, capacity: vectorCount)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        print("Inserting \(vectorCount) vectors...")
        let insertStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<vectorCount {
            var vector = [Float](repeating: 0, count: dimension)
            for j in 0..<dimension {
                vector[j] = Float.random(in: -1...1)
            }
            _ = try await index.insert(vector)
        }

        let insertTime = CFAbsoluteTimeGetCurrent() - insertStart
        print("Insert time: \(String(format: "%.2f", insertTime))s (\(String(format: "%.0f", Double(vectorCount) / insertTime)) vectors/sec)")

        // Warmup
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
        for _ in 0..<5 {
            _ = try await index.search(query: query, k: k)
        }

        // Benchmark search
        let iterations = 100
        var times: [Double] = []

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await index.search(query: query, k: k)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let avgTime = times.reduce(0, +) / Double(iterations)
        let medianTime = times.sorted()[iterations / 2]
        let qps = 1.0 / avgTime

        print("\n=== Flat Search Benchmark (10K vectors, 128D) ===")
        print("Average: \(String(format: "%.3f", avgTime * 1000))ms")
        print("Median: \(String(format: "%.3f", medianTime * 1000))ms")
        print("QPS: \(String(format: "%.0f", qps))")
    }

    /// Benchmark flat index search with 10K vectors, dim=768.
    func testFlatSearch10K_768D() async throws {
        let dimension = 768
        let vectorCount = 10_000
        let k = 10

        let config = IndexConfiguration.flat(dimension: dimension, capacity: vectorCount)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        print("Inserting \(vectorCount) vectors (dim=\(dimension))...")
        let insertStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<vectorCount {
            var vector = [Float](repeating: 0, count: dimension)
            for j in 0..<dimension {
                vector[j] = Float.random(in: -1...1)
            }
            _ = try await index.insert(vector)
        }

        let insertTime = CFAbsoluteTimeGetCurrent() - insertStart
        print("Insert time: \(String(format: "%.2f", insertTime))s")

        // Warmup
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
        for _ in 0..<3 {
            _ = try await index.search(query: query, k: k)
        }

        // Benchmark search
        let iterations = 50
        var times: [Double] = []

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await index.search(query: query, k: k)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let avgTime = times.reduce(0, +) / Double(iterations)
        let medianTime = times.sorted()[iterations / 2]
        let qps = 1.0 / avgTime

        print("\n=== Flat Search Benchmark (10K vectors, 768D) ===")
        print("Average: \(String(format: "%.3f", avgTime * 1000))ms")
        print("Median: \(String(format: "%.3f", medianTime * 1000))ms")
        print("QPS: \(String(format: "%.0f", qps))")
    }

    // MARK: - IVF Index Benchmarks

    /// Benchmark IVF index search with 10K vectors.
    func testIVFSearch10K_128D() async throws {
        let dimension = 128
        let vectorCount = 10_000
        let k = 10
        let nlist = 32
        let nprobe = 4

        let config = IndexConfiguration.ivf(dimension: dimension, nlist: nlist, nprobe: nprobe)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors
        print("Inserting \(vectorCount) vectors into IVF...")
        let insertStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<vectorCount {
            var vector = [Float](repeating: 0, count: dimension)
            for j in 0..<dimension {
                vector[j] = Float.random(in: -1...1)
            }
            _ = try await index.insert(vector)
        }

        let insertTime = CFAbsoluteTimeGetCurrent() - insertStart
        print("Insert time: \(String(format: "%.2f", insertTime))s")

        // Warmup
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
        for _ in 0..<5 {
            _ = try await index.search(query: query, k: k)
        }

        // Benchmark search
        let iterations = 100
        var times: [Double] = []

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await index.search(query: query, k: k)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let avgTime = times.reduce(0, +) / Double(iterations)
        let medianTime = times.sorted()[iterations / 2]
        let qps = 1.0 / avgTime

        print("\n=== IVF Search Benchmark (10K vectors, 128D, nlist=\(nlist), nprobe=\(nprobe)) ===")
        print("Average: \(String(format: "%.3f", avgTime * 1000))ms")
        print("Median: \(String(format: "%.3f", medianTime * 1000))ms")
        print("QPS: \(String(format: "%.0f", qps))")
    }

    // MARK: - Batch Insert Benchmark

    /// Benchmark batch insert performance.
    func testBatchInsertPerformance() async throws {
        let dimension = 128
        let batchSize = 1000

        let config = IndexConfiguration.flat(dimension: dimension, capacity: batchSize * 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Create batch
        var vectors: [[Float]] = []
        for _ in 0..<batchSize {
            vectors.append((0..<dimension).map { _ in Float.random(in: -1...1) })
        }

        // Warmup
        _ = try await index.insert(vectors)

        // Clear and benchmark
        let index2 = try await AcceleratedVectorIndex(configuration: config)

        let start = CFAbsoluteTimeGetCurrent()
        _ = try await index2.insert(vectors)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let vectorsPerSec = Double(batchSize) / elapsed

        print("\n=== Batch Insert Benchmark (\(batchSize) vectors, 128D) ===")
        print("Time: \(String(format: "%.3f", elapsed * 1000))ms")
        print("Throughput: \(String(format: "%.0f", vectorsPerSec)) vectors/sec")
    }

    // MARK: - Summary Benchmark

    /// Run all benchmarks and print summary.
    func testBenchmarkSummary() async throws {
        print("\n")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║        VectorAccelerate Performance Benchmarks        ║")
        print("╠══════════════════════════════════════════════════════════════╣")

        // Quick benchmark suite
        let dimensions = [128, 768]
        let vectorCount = 5000
        let k = 8

        for dim in dimensions {
            let config = IndexConfiguration.flat(dimension: dim, capacity: vectorCount)
            let index = try await AcceleratedVectorIndex(configuration: config)

            // Insert
            let insertStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<vectorCount {
                var v = [Float](repeating: 0, count: dim)
                for j in 0..<dim { v[j] = Float.random(in: -1...1) }
                _ = try await index.insert(v)
            }
            let insertTime = CFAbsoluteTimeGetCurrent() - insertStart

            // Search
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }
            for _ in 0..<3 { _ = try await index.search(query: query, k: k) }

            var searchTimes: [Double] = []
            for _ in 0..<50 {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await index.search(query: query, k: k)
                searchTimes.append(CFAbsoluteTimeGetCurrent() - start)
            }
            let medianSearch = searchTimes.sorted()[25]

            print("║ Dim \(dim): Insert \(String(format: "%5.0f", Double(vectorCount)/insertTime)) vec/s | Search \(String(format: "%6.2f", medianSearch * 1000))ms ║")
        }

        // IVF benchmark with smaller dataset to avoid capacity issue
        let ivfVectorCount = 500
        let ivfConfig = IndexConfiguration.ivf(dimension: 128, nlist: 8, nprobe: 2)
        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

        for _ in 0..<ivfVectorCount {
            var v = [Float](repeating: 0, count: 128)
            for j in 0..<128 { v[j] = Float.random(in: -1...1) }
            _ = try await ivfIndex.insert(v)
        }

        let ivfQuery = (0..<128).map { _ in Float.random(in: -1...1) }
        for _ in 0..<3 { _ = try await ivfIndex.search(query: ivfQuery, k: k) }

        var ivfTimes: [Double] = []
        for _ in 0..<50 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await ivfIndex.search(query: ivfQuery, k: k)
            ivfTimes.append(CFAbsoluteTimeGetCurrent() - start)
        }
        let medianIVF = ivfTimes.sorted()[25]

        print("║ IVF (nlist=8, nprobe=2):                   Search \(String(format: "%6.2f", medianIVF * 1000))ms ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print("")
    }
}
