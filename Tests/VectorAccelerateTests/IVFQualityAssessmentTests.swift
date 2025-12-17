//
//  IVFQualityAssessmentTests.swift
//  VectorAccelerateTests
//
//  Assessment of IVF quality against industry benchmarks
//

import XCTest
@testable import VectorAccelerate

final class IVFQualityAssessmentTests: XCTestCase {

    /// Compare against FAISS-like expectations
    /// FAISS IVF typically achieves:
    /// - nprobe=10% of nlist: 70-80% recall
    /// - nprobe=20% of nlist: 85-90% recall
    /// - nprobe=50% of nlist: 95%+ recall
    func testIVFQualityVsFAISSBenchmarks() async throws {
        let dimension = 128  // More realistic dimension
        let datasetSize = 2000
        let nlist = 32  // sqrt(2000) ≈ 45, so 32 is reasonable
        let k = 10
        let numQueries = 100

        print("\n" + "=" .repeated(70))
        print("IVF Quality Assessment vs Industry Benchmarks")
        print("=" .repeated(70))
        print("Dataset: N=\(datasetSize), D=\(dimension), nlist=\(nlist), K=\(k)")
        print("-" .repeated(70))

        // Generate clustered data (more realistic than pure random)
        var rng = SeededRNG(seed: 12345)
        let dataset = generateClusteredData(
            numClusters: 20,
            pointsPerCluster: datasetSize / 20,
            dimension: dimension,
            clusterSpread: 0.3,
            rng: &rng
        )

        // Create flat index for ground truth
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        _ = try await flatIndex.insert(dataset)

        // Generate queries (sample from dataset + noise for realistic queries)
        var queries: [[Float]] = []
        for _ in 0..<numQueries {
            let baseIdx = Int.random(in: 0..<dataset.count, using: &rng)
            var query = dataset[baseIdx]
            // Add small noise
            for j in 0..<dimension {
                query[j] += Float.random(in: -0.1...0.1, using: &rng)
            }
            queries.append(query)
        }

        // Get ground truth
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
        let nprobeConfigs: [(nprobe: Int, expectedRecall: Float, label: String)] = [
            (3, 0.70, "~10%"),   // 3/32 ≈ 10%
            (6, 0.80, "~20%"),   // 6/32 ≈ 20%
            (16, 0.92, "50%"),   // 16/32 = 50%
            (32, 0.99, "100%"),  // 32/32 = 100%
        ]

        print("\n| nprobe | % clusters | Expected | Actual | Status |")
        print("|--------|------------|----------|--------|--------|")

        var allPassed = true

        for (nprobe, expectedRecall, label) in nprobeConfigs {
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                minTrainingVectors: 200
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
            _ = try await ivfIndex.insert(dataset)

            // Measure recall
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
                totalRecall += Float(intersection.count) / Float(k)
            }
            let avgRecall = totalRecall / Float(numQueries)

            let status: String
            let tolerance: Float = 0.15  // Allow 15% below expected
            if avgRecall >= expectedRecall - tolerance {
                status = "✓ OK"
            } else {
                status = "✗ LOW"
                allPassed = false
            }

            print("| \(String(format: "%6d", nprobe)) | \(label.padding(toLength: 10, withPad: " ", startingAt: 0)) | \(String(format: "%5.0f%%", expectedRecall * 100))    | \(String(format: "%5.1f%%", avgRecall * 100))  | \(status) |")
        }

        print("-" .repeated(70))

        if allPassed {
            print("✓ IVF recall is within acceptable range of industry benchmarks")
        } else {
            print("⚠ IVF recall is below expected - may need tuning")
        }
    }

    /// Test recall with different dataset characteristics
    func testRecallWithDifferentDataDistributions() async throws {
        let dimension = 64
        let datasetSize = 1000
        let nlist = 16
        let nprobe = 8  // 50%
        let k = 10
        let numQueries = 50

        print("\n" + "=" .repeated(70))
        print("Recall vs Data Distribution (nprobe=50%, N=1000)")
        print("=" .repeated(70))

        var rng = SeededRNG(seed: 42)

        let distributions: [(name: String, data: [[Float]])] = [
            ("Uniform Random", generateUniformData(count: datasetSize, dimension: dimension, rng: &rng)),
            ("Gaussian Clusters", generateClusteredData(numClusters: 10, pointsPerCluster: 100, dimension: dimension, clusterSpread: 0.2, rng: &rng)),
            ("Sparse Clusters", generateClusteredData(numClusters: 5, pointsPerCluster: 200, dimension: dimension, clusterSpread: 0.5, rng: &rng)),
        ]

        print("\n| Distribution | Recall | Notes |")
        print("|--------------|--------|-------|")

        for (name, data) in distributions {
            // Create indexes
            let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
            let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                minTrainingVectors: 100
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            // Insert data
            _ = try await flatIndex.insert(data)
            _ = try await ivfIndex.insert(data)

            // Generate queries from data
            var totalRecall: Float = 0
            for _ in 0..<numQueries {
                let queryIdx = Int.random(in: 0..<data.count, using: &rng)
                var query = data[queryIdx]
                for j in 0..<dimension {
                    query[j] += Float.random(in: -0.05...0.05, using: &rng)
                }

                // Ground truth
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
                totalRecall += Float(intersection.count) / Float(k)
            }

            let avgRecall = totalRecall / Float(numQueries)
            let notes = avgRecall > 0.85 ? "Good" : (avgRecall > 0.70 ? "Acceptable" : "Needs work")
            print("| \(name.padding(toLength: 12, withPad: " ", startingAt: 0)) | \(String(format: "%5.1f%%", avgRecall * 100)) | \(notes) |")
        }
    }

    /// Throughput comparison to see if IVF provides speedup at scale
    func testIVFThroughputVsFlat() async throws {
        let dimension = 128
        let k = 10
        let numQueries = 50

        print("\n" + "=" .repeated(70))
        print("IVF vs Flat Throughput Comparison")
        print("=" .repeated(70))

        var rng = SeededRNG(seed: 999)

        let sizes = [1000, 2000, 5000]

        print("\n| N | Flat (q/s) | IVF (q/s) | Speedup | IVF Recall |")
        print("|------|------------|-----------|---------|------------|")

        for size in sizes {
            let nlist = max(8, Int(sqrt(Double(size))))  // sqrt(N) rule
            let nprobe = max(1, nlist / 4)  // 25% probe

            // Generate data
            let data = generateClusteredData(
                numClusters: 20,
                pointsPerCluster: size / 20,
                dimension: dimension,
                clusterSpread: 0.3,
                rng: &rng
            )

            // Create indexes
            let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: size * 2)
            let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: size * 2,
                minTrainingVectors: min(200, size / 2)
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            _ = try await flatIndex.insert(data)
            _ = try await ivfIndex.insert(data)

            // Generate queries
            var queries: [[Float]] = []
            for _ in 0..<numQueries {
                let query = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
                queries.append(query)
            }

            // Benchmark flat
            let flatStart = CFAbsoluteTimeGetCurrent()
            for query in queries {
                _ = try await flatIndex.search(query: query, k: k)
            }
            let flatTime = CFAbsoluteTimeGetCurrent() - flatStart
            let flatThroughput = Double(numQueries) / flatTime

            // Benchmark IVF and measure recall
            var totalRecall: Float = 0
            let ivfStart = CFAbsoluteTimeGetCurrent()
            for query in queries {
                let gtResults = try await flatIndex.search(query: query, k: k)
                var gtSlots: Set<UInt32> = []
                for result in gtResults {
                    if let slot = await flatIndex.slot(for: result.handle) {
                        gtSlots.insert(slot)
                    }
                }

                let ivfResults = try await ivfIndex.search(query: query, k: k)
                var ivfSlots: Set<UInt32> = []
                for result in ivfResults {
                    if let slot = await ivfIndex.slot(for: result.handle) {
                        ivfSlots.insert(slot)
                    }
                }

                let intersection = ivfSlots.intersection(gtSlots)
                totalRecall += Float(intersection.count) / Float(k)
            }
            let ivfTime = CFAbsoluteTimeGetCurrent() - ivfStart
            let ivfThroughput = Double(numQueries) / ivfTime
            let avgRecall = totalRecall / Float(numQueries)

            let speedup = ivfThroughput / flatThroughput

            print("| \(String(format: "%4d", size)) | \(String(format: "%10.0f", flatThroughput)) | \(String(format: "%9.0f", ivfThroughput)) | \(String(format: "%6.2fx", speedup)) | \(String(format: "%9.1f%%", avgRecall * 100)) |")
        }

        print("\nNote: IVF benefits increase with larger datasets (N > 10K)")
    }

    // MARK: - Data Generation Helpers

    private func generateUniformData(count: Int, dimension: Int, rng: inout SeededRNG) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }
    }

    private func generateClusteredData(
        numClusters: Int,
        pointsPerCluster: Int,
        dimension: Int,
        clusterSpread: Float,
        rng: inout SeededRNG
    ) -> [[Float]] {
        var data: [[Float]] = []

        // Generate cluster centers
        var centers: [[Float]] = []
        for _ in 0..<numClusters {
            let center = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            centers.append(center)
        }

        // Generate points around each center
        for center in centers {
            for _ in 0..<pointsPerCluster {
                var point = center
                for j in 0..<dimension {
                    point[j] += Float.random(in: -clusterSpread...clusterSpread, using: &rng)
                }
                data.append(point)
            }
        }

        return data
    }
}

// MARK: - Seeded RNG

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

private extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
