//
//  IVFQualityAssessmentTests.swift
//  VectorAccelerateTests
//
//  Assessment of IVF quality against industry benchmarks
//
//  IMPORTANT: These tests use routingThreshold: 0 to force IVF search
//  instead of falling back to flat search for small datasets.
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
        let dimension = 128
        let datasetSize = 2000
        let nlist = 32  // sqrt(2000) ≈ 45, so 32 is reasonable
        let k = 10
        let numQueries = 100

        print("\n" + String(repeating: "=", count: 70))
        print("IVF Quality Assessment vs Industry Benchmarks")
        print(String(repeating: "=", count: 70))
        print("Dataset: N=\(datasetSize), D=\(dimension), nlist=\(nlist), K=\(k)")
        print("routingThreshold: 0 (IVF search forced)")
        print(String(repeating: "-", count: 70))

        // Generate TRUE UNIFORM random data (hardest case for IVF)
        var gen = TestDataGenerator(seed: 12345)
        let dataset = gen.uniformVectors(count: datasetSize, dimension: dimension)

        // Print dataset statistics
        let stats = TestDataGenerator.statistics(for: dataset)
        print("Data distribution: UNIFORM RANDOM")
        print(stats.description)
        print(String(repeating: "-", count: 70))

        // Create flat index for ground truth
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        _ = try await flatIndex.insert(dataset)

        // Generate queries as perturbed dataset points
        let queries = gen.perturbedQueries(from: dataset, count: numQueries, noiseStdDev: 0.1)

        // Get ground truth using handle's internal index for comparison
        var groundTruth: [[UInt32]] = []
        for query in queries {
            let results = try await flatIndex.search(query: query, k: k)
            groundTruth.append(results.map { $0.handle.stableID })
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
            // CRITICAL: Set routingThreshold to 0 to force IVF search
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                routingThreshold: 0  // Disable flat search fallback
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
            _ = try await ivfIndex.insert(dataset)

            // Verify IVF is trained
            let ivfStats = await ivfIndex.statistics()
            print("  [nprobe=\(nprobe)] IVF trained: \(ivfStats.ivfStats?.isTrained ?? false), vectors: \(ivfStats.vectorCount)")

            // Measure recall using index values (not full handles)
            var totalRecall: Float = 0
            for (i, query) in queries.enumerated() {
                let results = try await ivfIndex.search(query: query, k: k)
                let resultIndices = Set(results.map { $0.handle.stableID })
                let gtSet = Set(groundTruth[i])
                let intersection = resultIndices.intersection(gtSet)
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

        print(String(repeating: "-", count: 70))

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

        print("\n" + String(repeating: "=", count: 70))
        print("Recall vs Data Distribution (nprobe=50%, N=1000)")
        print("routingThreshold: 0 (IVF search forced)")
        print(String(repeating: "=", count: 70))

        // Create generators with different streams for independent sequences
        var uniformGen = TestDataGenerator(seed: 42, stream: 0)
        var gaussianGen = TestDataGenerator(seed: 42, stream: 1)
        var sparseGen = TestDataGenerator(seed: 42, stream: 2)

        let distributions: [(name: String, data: [[Float]], stats: DatasetStatistics)] = [
            {
                let data = uniformGen.uniformVectors(count: datasetSize, dimension: dimension)
                return ("Uniform Rand", data, TestDataGenerator.statistics(for: data))
            }(),
            {
                let data = gaussianGen.gaussianClusters(
                    numClusters: 10,
                    pointsPerCluster: 100,
                    dimension: dimension,
                    clusterSpread: 2.0,
                    clusterStdDev: 0.3
                )
                return ("Gaussian Clu", data, TestDataGenerator.statistics(for: data))
            }(),
            {
                let data = sparseGen.gaussianClusters(
                    numClusters: 5,
                    pointsPerCluster: 200,
                    dimension: dimension,
                    clusterSpread: 3.0,
                    clusterStdDev: 0.5
                )
                return ("Sparse Clust", data, TestDataGenerator.statistics(for: data))
            }(),
        ]

        print("\n| Distribution | Recall | IVF Trained | Norm (mean±std) |")
        print("|--------------|--------|-------------|-----------------|")

        for (name, data, dataStats) in distributions {
            // Create indexes
            let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
            let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

            // CRITICAL: Set routingThreshold to 0 to force IVF search
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                routingThreshold: 0  // Disable flat search fallback
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            // Insert data
            _ = try await flatIndex.insert(data)
            _ = try await ivfIndex.insert(data)

            // Check if IVF is trained
            let ivfStats = await ivfIndex.statistics()
            let isTrained = ivfStats.ivfStats?.isTrained ?? false

            // Generate queries from data using a fresh generator
            var queryGen = TestDataGenerator(seed: 999, stream: 3)
            var totalRecall: Float = 0
            for _ in 0..<numQueries {
                let queries = queryGen.perturbedQueries(from: data, count: 1, noiseStdDev: 0.05)
                let query = queries[0]

                // Ground truth (use index for comparison)
                let gtResults = try await flatIndex.search(query: query, k: k)
                let gtIndices = Set(gtResults.map { $0.handle.stableID })

                // IVF results
                let ivfResults = try await ivfIndex.search(query: query, k: k)
                let ivfIndices = Set(ivfResults.map { $0.handle.stableID })

                let intersection = ivfIndices.intersection(gtIndices)
                totalRecall += Float(intersection.count) / Float(k)
            }

            let avgRecall = totalRecall / Float(numQueries)
            let normStr = String(format: "%.2f±%.2f", dataStats.meanNorm, dataStats.stdDevNorm)
            print("| \(name.padding(toLength: 12, withPad: " ", startingAt: 0)) | \(String(format: "%5.1f%%", avgRecall * 100)) | \(isTrained ? "Yes" : "No ") | \(normStr.padding(toLength: 15, withPad: " ", startingAt: 0)) |")
        }
    }

    /// Throughput comparison to see if IVF provides speedup at scale
    func testIVFThroughputVsFlat() async throws {
        let dimension = 128
        let k = 10
        let numQueries = 50

        print("\n" + String(repeating: "=", count: 70))
        print("IVF vs Flat Throughput Comparison")
        print("routingThreshold: 0 (IVF search forced)")
        print(String(repeating: "=", count: 70))

        let sizes = [1000, 2000, 5000]

        print("\n|    N | nlist | nprobe | Flat (q/s) | IVF (q/s) | Speedup | IVF Recall |")
        print("|------|-------|--------|------------|-----------|---------|------------|")

        for (sizeIdx, size) in sizes.enumerated() {
            var gen = TestDataGenerator(seed: 999, stream: UInt64(sizeIdx))

            let nlist = max(8, Int(sqrt(Double(size))))  // sqrt(N) rule
            let nprobe = max(1, nlist / 4)  // 25% probe

            // Generate uniform random data (harder for IVF)
            let data = gen.uniformVectors(count: size, dimension: dimension)

            // Create indexes
            let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: size * 2)
            let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

            // CRITICAL: Set routingThreshold to 0 to force IVF search
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: size * 2,
                routingThreshold: 0  // Disable flat search fallback
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)

            _ = try await flatIndex.insert(data)
            _ = try await ivfIndex.insert(data)

            // Generate random queries (independent of dataset)
            let queries = gen.randomQueries(count: numQueries, dimension: dimension)

            // Pre-compute ground truth BEFORE timing
            var groundTruth: [Set<UInt32>] = []
            for query in queries {
                let gtResults = try await flatIndex.search(query: query, k: k)
                groundTruth.append(Set(gtResults.map { $0.handle.stableID }))
            }

            // Benchmark flat (pure timing, no recall calculation)
            let flatStart = CFAbsoluteTimeGetCurrent()
            for query in queries {
                _ = try await flatIndex.search(query: query, k: k)
            }
            let flatTime = CFAbsoluteTimeGetCurrent() - flatStart
            let flatThroughput = Double(numQueries) / flatTime

            // Benchmark IVF (pure timing)
            let ivfStart = CFAbsoluteTimeGetCurrent()
            var ivfResults: [[IndexSearchResult]] = []
            for query in queries {
                let results = try await ivfIndex.search(query: query, k: k)
                ivfResults.append(results)
            }
            let ivfTime = CFAbsoluteTimeGetCurrent() - ivfStart
            let ivfThroughput = Double(numQueries) / ivfTime

            // Calculate recall AFTER timing
            var totalRecall: Float = 0
            for (i, results) in ivfResults.enumerated() {
                let ivfIndices = Set(results.map { $0.handle.stableID })
                let intersection = ivfIndices.intersection(groundTruth[i])
                totalRecall += Float(intersection.count) / Float(k)
            }
            let avgRecall = totalRecall / Float(numQueries)

            let speedup = ivfThroughput / flatThroughput

            print("| \(String(format: "%4d", size)) | \(String(format: "%5d", nlist)) | \(String(format: "%6d", nprobe)) | \(String(format: "%10.0f", flatThroughput)) | \(String(format: "%9.0f", ivfThroughput)) | \(String(format: "%6.2fx", speedup)) | \(String(format: "%9.1f%%", avgRecall * 100)) |")
        }

        print("\nNote: IVF benefits increase with larger datasets (N > 10K)")
        print("      At small N, IVF overhead may exceed flat search time.")
    }

    /// Probe quality with a larger K to help diagnose small-K emission issues
    func testIVFQualityWithLargerK() async throws {
        // Enable IVF debug prints for this test run
        IVFSearchPipeline.debugEnabled = true
        defer { IVFSearchPipeline.debugEnabled = false }

        let dimension = 128
        let datasetSize = 2000
        let nlist = 32
        let kLarge = 32  // larger K to bypass any small-K kernel path quirks
        let numQueries = 50

        print("\n" + String(repeating: "=", count: 70))
        print("Large-K IVF Quality Probe (K=\(kLarge))")
        print("routingThreshold: 0 (IVF search forced)")
        print(String(repeating: "=", count: 70))

        // Generate UNIFORM random data for stress test
        var gen = TestDataGenerator(seed: 777)
        let dataset = gen.uniformVectors(count: datasetSize, dimension: dimension)

        // Print dataset statistics
        let stats = TestDataGenerator.statistics(for: dataset)
        print("Data distribution: UNIFORM RANDOM")
        print(stats.description)
        print(String(repeating: "-", count: 70))

        // Flat index for ground truth
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        _ = try await flatIndex.insert(dataset)

        // IVF index - probe all lists to approximate flat
        let nprobe = nlist
        let ivfConfig = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            capacity: datasetSize * 2,
            routingThreshold: 0
        )
        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
        _ = try await ivfIndex.insert(dataset)

        // Queries from dataset with slight noise
        let queries = gen.perturbedQueries(from: dataset, count: numQueries, noiseStdDev: 0.05)

        // Ground truth
        var groundTruth: [Set<UInt32>] = []
        for q in queries {
            let gt = try await flatIndex.search(query: q, k: kLarge)
            groundTruth.append(Set(gt.map { $0.handle.stableID }))
        }

        // IVF results and recall
        var totalRecall: Float = 0
        for (i, q) in queries.enumerated() {
            let res = try await ivfIndex.search(query: q, k: kLarge)
            let idxSet = Set(res.map { $0.handle.stableID })
            let inter = idxSet.intersection(groundTruth[i])
            totalRecall += Float(inter.count) / Float(kLarge)
        }
        let avgRecall = totalRecall / Float(numQueries)
        print("\nLarge-K Recall (nprobe=\(nprobe), K=\(kLarge)): \(String(format: "%5.1f%%", avgRecall * 100))")
    }
}
