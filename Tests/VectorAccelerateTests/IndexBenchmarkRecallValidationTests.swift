//
//  IndexBenchmarkRecallValidationTests.swift
//  VectorAccelerateTests
//
//  Rigorous validation of recall calculation in IndexBenchmarkHarness
//

import XCTest
@testable import VectorAccelerate

final class IndexBenchmarkRecallValidationTests: XCTestCase {

    // MARK: - Recall Validation Tests

    /// Test that recall calculation is actually working correctly
    /// by manually verifying ground truth vs search results
    func testRecallCalculationCorrectness() async throws {
        // Create a simple dataset where we know the ground truth
        let dimension = 16
        let datasetSize = 100
        let k = 5

        // Create index
        let config = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let index = try await AcceleratedVectorIndex(configuration: config)

        // Insert vectors: vector[i] = [i, i, i, ..., i] (all same value)
        // This makes distances predictable: dist(query=[0,...,0], vector[i]) = i^2 * dimension
        var vectors: [[Float]] = []
        for i in 0..<datasetSize {
            let value = Float(i)
            let vector = [Float](repeating: value, count: dimension)
            vectors.append(vector)
            _ = try await index.insert(vector)
        }

        // Query with [0, 0, ..., 0] - nearest should be vectors 0, 1, 2, 3, 4
        let query = [Float](repeating: 0.0, count: dimension)
        let results = try await index.search(query: query, k: k)

        print("\n=== Recall Calculation Validation ===")
        print("Query: [0, 0, ..., 0]")
        print("Expected nearest: slots 0, 1, 2, 3, 4")
        print("Got results:")

        var resultSlots: [UInt32] = []
        for (i, result) in results.enumerated() {
            if let slot = await index.slot(for: result.handle) {
                resultSlots.append(slot)
                print("  [\(i)] slot=\(slot), distance=\(result.distance)")
            }
        }

        // Verify we got the correct nearest neighbors
        let expectedSlots: Set<UInt32> = [0, 1, 2, 3, 4]
        let actualSlots = Set(resultSlots)
        let intersection = actualSlots.intersection(expectedSlots)
        let recall = Float(intersection.count) / Float(k)

        print("Expected slots: \(expectedSlots.sorted())")
        print("Actual slots: \(actualSlots.sorted())")
        print("Recall: \(recall * 100)%")

        XCTAssertEqual(recall, 1.0, "Flat index should find exact nearest neighbors")
    }

    /// Test IVF with nprobe < nlist to see realistic recall
    func testIVFRecallWithPartialProbe() async throws {
        let dimension = 32
        let datasetSize = 1000
        let nlist = 16
        let k = 10

        print("\n=== IVF Recall vs nprobe (N=\(datasetSize), nlist=\(nlist), K=\(k)) ===")

        // Create flat index for ground truth
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Generate random dataset with seed for reproducibility
        var rng = SeededRNG(seed: 42)
        var dataset: [[Float]] = []
        for _ in 0..<datasetSize {
            let vector = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            dataset.append(vector)
            _ = try await flatIndex.insert(vector)
        }

        // Generate queries
        let numQueries = 50
        var queries: [[Float]] = []
        for _ in 0..<numQueries {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            queries.append(query)
        }

        // Get ground truth from flat index
        var groundTruth: [[UInt32]] = []
        for query in queries {
            let results = try await flatIndex.search(query: query, k: k)
            var slots: [UInt32] = []
            for result in results {
                if let slot = await flatIndex.slot(for: result.handle) {
                    slots.append(slot)
                }
            }
            groundTruth.append(slots)
        }

        // Test different nprobe values
        let nprobeValues = [1, 2, 4, 8, 16]

        for nprobe in nprobeValues {
            // Create IVF index
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                minTrainingVectors: 100
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            // Insert same dataset
            _ = try await ivfIndex.insert(dataset)

            // Search and compute recall
            var totalRecall: Float = 0
            for (i, query) in queries.enumerated() {
                let results = try await ivfIndex.search(query: query, k: k)
                var resultSlots: Set<UInt32> = []
                for result in results {
                    if let slot = await ivfIndex.slot(for: result.handle) {
                        resultSlots.insert(slot)
                    }
                }

                let gtSet = Set(groundTruth[i])
                let intersection = resultSlots.intersection(gtSet)
                let queryRecall = Float(intersection.count) / Float(k)
                totalRecall += queryRecall
            }

            let avgRecall = totalRecall / Float(numQueries)
            let probePercent = Float(nprobe) / Float(nlist) * 100

            print("nprobe=\(nprobe)/\(nlist) (\(String(format: "%.0f", probePercent))%): Recall = \(String(format: "%.1f", avgRecall * 100))%")

            // Verify expected behavior
            if nprobe == nlist {
                XCTAssertGreaterThan(avgRecall, 0.95, "nprobe=nlist should give >95% recall")
            }
        }
    }

    /// Test with larger dataset to see scaling behavior
    func testRecallAtScale() async throws {
        let dimension = 64
        let k = 10
        let nlist = 32
        let nprobe = 4  // Only probe 12.5% of clusters

        print("\n=== IVF Recall at Scale (nlist=\(nlist), nprobe=\(nprobe)) ===")

        let datasetSizes = [500, 1000, 2000, 5000]

        for datasetSize in datasetSizes {
            // Create IVF index
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                minTrainingVectors: min(200, datasetSize / 2)
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            // Create flat index for ground truth
            let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
            let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

            // Generate dataset
            var rng = SeededRNG(seed: 123)
            var dataset: [[Float]] = []
            for _ in 0..<datasetSize {
                let vector = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
                dataset.append(vector)
            }

            // Insert into both indexes
            _ = try await ivfIndex.insert(dataset)
            _ = try await flatIndex.insert(dataset)

            // Generate queries and measure recall
            let numQueries = 20
            var totalRecall: Float = 0

            for _ in 0..<numQueries {
                let query = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }

                // Ground truth from flat
                let gtResults = try await flatIndex.search(query: query, k: k)
                var gtSlots: Set<UInt32> = []
                for result in gtResults {
                    if let slot = await flatIndex.slot(for: result.handle) {
                        gtSlots.insert(slot)
                    }
                }

                // IVF results
                let ivfResults = try await ivfIndex.search(query: query, k: k)
                var ivfSlots: Set<UInt32> = []
                for result in ivfResults {
                    if let slot = await ivfIndex.slot(for: result.handle) {
                        ivfSlots.insert(slot)
                    }
                }

                let intersection = ivfSlots.intersection(gtSlots)
                let queryRecall = Float(intersection.count) / Float(k)
                totalRecall += queryRecall
            }

            let avgRecall = totalRecall / Float(numQueries)
            print("N=\(String(format: "%5d", datasetSize)): Recall = \(String(format: "%.1f", avgRecall * 100))%")

            // With nprobe=4 out of 32, we expect lower recall
            // But it should still be reasonable (>50%) due to cluster locality
        }
    }

    /// Test the benchmark harness with realistic IVF settings
    func testBenchmarkHarnessRealisticIVF() async throws {
        let harness = IndexBenchmarkHarness(seed: 42)

        print("\n=== Benchmark Harness with Realistic IVF Settings ===")

        // Configuration with nprobe < nlist (realistic ANN setting)
        let configs: [(name: String, config: IndexBenchmarkConfiguration)] = [
            ("Flat-D64-N1000", IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 1000,
                numQueries: 30,
                k: 10,
                indexType: .flat,
                warmupIterations: 1,
                measurementIterations: 3,
                computeRecall: true,
                seed: 42
            )),
            ("IVF-nprobe=nlist", IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 1000,
                numQueries: 30,
                k: 10,
                indexType: .ivf(nlist: 16, nprobe: 16, minTrainingVectors: 100),
                warmupIterations: 1,
                measurementIterations: 3,
                computeRecall: true,
                seed: 42
            )),
            ("IVF-nprobe=50%", IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 1000,
                numQueries: 30,
                k: 10,
                indexType: .ivf(nlist: 16, nprobe: 8, minTrainingVectors: 100),
                warmupIterations: 1,
                measurementIterations: 3,
                computeRecall: true,
                seed: 42
            )),
            ("IVF-nprobe=25%", IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 1000,
                numQueries: 30,
                k: 10,
                indexType: .ivf(nlist: 16, nprobe: 4, minTrainingVectors: 100),
                warmupIterations: 1,
                measurementIterations: 3,
                computeRecall: true,
                seed: 42
            )),
            ("IVF-nprobe=12.5%", IndexBenchmarkConfiguration(
                dimension: 64,
                datasetSize: 1000,
                numQueries: 30,
                k: 10,
                indexType: .ivf(nlist: 16, nprobe: 2, minTrainingVectors: 100),
                warmupIterations: 1,
                measurementIterations: 3,
                computeRecall: true,
                seed: 42
            )),
        ]

        print("\n| Configuration | Recall | Throughput | p50 Latency |")
        print("|---------------|--------|------------|-------------|")

        for (name, config) in configs {
            let result = try await harness.runBenchmark(configuration: config)
            print("| \(name.padding(toLength: 13, withPad: " ", startingAt: 0)) | \(String(format: "%5.1f%%", result.recall * 100)) | \(String(format: "%6.0f q/s", result.throughput)) | \(String(format: "%6.3f ms", result.latencyStats.p50 * 1000)) |")
        }
    }

    /// Test with batch search to see if batching affects recall
    func testBatchSearchRecall() async throws {
        let dimension = 64
        let datasetSize = 1000
        let k = 10
        let numQueries = 50

        print("\n=== Batch Search Recall Consistency ===")

        // Create flat index
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let index = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Generate and insert dataset
        var rng = SeededRNG(seed: 999)
        for _ in 0..<datasetSize {
            let vector = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            _ = try await index.insert(vector)
        }

        // Generate queries
        var queries: [[Float]] = []
        for _ in 0..<numQueries {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            queries.append(query)
        }

        // Get single-query results (ground truth)
        var singleQueryResults: [[UInt32]] = []
        for query in queries {
            let results = try await index.search(query: query, k: k)
            var slots: [UInt32] = []
            for result in results {
                if let slot = await index.slot(for: result.handle) {
                    slots.append(slot)
                }
            }
            singleQueryResults.append(slots)
        }

        // Get batch query results
        let batchResults = try await index.search(queries: queries, k: k)
        var batchQueryResults: [[UInt32]] = []
        for queryResults in batchResults {
            var slots: [UInt32] = []
            for result in queryResults {
                if let slot = await index.slot(for: result.handle) {
                    slots.append(slot)
                }
            }
            batchQueryResults.append(slots)
        }

        // Compare single vs batch
        var totalRecall: Float = 0
        var mismatchCount = 0

        for i in 0..<numQueries {
            let singleSet = Set(singleQueryResults[i])
            let batchSet = Set(batchQueryResults[i])
            let intersection = singleSet.intersection(batchSet)
            let recall = Float(intersection.count) / Float(k)
            totalRecall += recall

            if recall < 1.0 {
                mismatchCount += 1
            }
        }

        let avgRecall = totalRecall / Float(numQueries)
        print("Single vs Batch consistency: \(String(format: "%.1f", avgRecall * 100))%")
        print("Queries with differences: \(mismatchCount)/\(numQueries)")

        // Flat index batch should match single query exactly
        XCTAssertEqual(avgRecall, 1.0, accuracy: 0.01, "Batch search should match single query search")
    }
}

// MARK: - Seeded RNG for Reproducibility

private struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
