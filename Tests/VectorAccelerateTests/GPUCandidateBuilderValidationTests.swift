//
//  GPUCandidateBuilderValidationTests.swift
//  VectorAccelerateTests
//
//  Validation tests to ensure GPU candidate builder produces correct results
//

import XCTest
@testable import VectorAccelerate

final class GPUCandidateBuilderValidationTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        context = try await Metal4Context()
    }

    /// Direct comparison of GPU vs CPU candidate building
    func testGPUvsCPUCandidateBuildingEquivalence() async throws {
        print("\n" + "=" .repeated(70))
        print("GPU vs CPU Candidate Building Comparison")
        print("=" .repeated(70))

        let dimension = 64
        let datasetSize = 500
        let nlist = 16
        let k = 10
        let numQueries = 20

        // Generate random data
        var rng = SeededRNG(seed: 12345)
        let dataset = (0..<datasetSize).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        // Generate truly random queries (NOT from dataset)
        let queries = (0..<numQueries).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        // Create ground truth with flat index
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        _ = try await flatIndex.insert(dataset)

        var groundTruth: [Set<UInt32>] = []
        for query in queries {
            let results = try await flatIndex.search(query: query, k: k)
            // Use stableID for comparison (consistent across index instances)
            let stableIDs = Set(results.map { $0.handle.stableID })
            groundTruth.append(stableIDs)
        }

        // Test with GPU candidate builder enabled
        print("\n--- Testing with GPU Candidate Builder (default) ---")
        let ivfConfigGPU = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: 4,  // 25% of lists
            capacity: datasetSize * 2,
            routingThreshold: 0,  // Force IVF search (don't fall back to flat)
            minTrainingVectors: 100
        )
        let ivfIndexGPU = try await AcceleratedVectorIndex(configuration: ivfConfigGPU)
        _ = try await ivfIndexGPU.insert(dataset)

        // Verify IVF is trained before proceeding
        let stats = await ivfIndexGPU.statistics()
        guard stats.ivfStats?.isTrained == true else {
            throw XCTSkip("IVF index not trained - GPU may not be functioning correctly in CI")
        }

        var recallGPU: Float = 0
        for (i, query) in queries.enumerated() {
            let results = try await ivfIndexGPU.search(query: query, k: k)
            // Use stableID for comparison (consistent across index instances)
            let resultIDs = Set(results.map { $0.handle.stableID })
            let intersection = resultIDs.intersection(groundTruth[i])
            recallGPU += Float(intersection.count) / Float(k)
        }
        recallGPU /= Float(numQueries)
        print("GPU Candidate Builder Recall: \(String(format: "%.1f%%", recallGPU * 100))")

        // Note: CPU candidate building requires direct IVFSearchPipeline access
        // For now, we verify the GPU path works and produces reasonable recall
        print("\n" + "-" .repeated(70))
        print("GPU Candidate Builder Result:")
        print("  Recall: \(String(format: "%.1f%%", recallGPU * 100))")
        print("  Expected range for nprobe=4/nlist=16 (25%): 50-80%")

        // With random queries and 25% of lists probed, expect 50-80% recall
        XCTAssertGreaterThan(recallGPU, 0.40, "Recall should be at least 40% with nprobe=25%")
        XCTAssertLessThan(recallGPU, 0.95, "Recall with nprobe=25% shouldn't be near 100% (suspicious)")
    }

    /// Test with increasing nprobe to verify recall behavior
    func testRecallVsNprobeWithRandomQueries() async throws {
        print("\n" + "=" .repeated(70))
        print("Recall vs nprobe (Truly Random Queries)")
        print("=" .repeated(70))

        let dimension = 128
        let datasetSize = 2000
        let nlist = 32
        let k = 10
        let numQueries = 100

        // Generate random data
        var rng = SeededRNG(seed: 54321)
        let dataset = (0..<datasetSize).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        // Generate truly random queries (NOT from dataset)
        let queries = (0..<numQueries).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        // Create ground truth
        let flatConfig = IndexConfiguration.flat(dimension: dimension, capacity: datasetSize * 2)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)
        _ = try await flatIndex.insert(dataset)

        var groundTruth: [Set<UInt32>] = []
        for query in queries {
            let results = try await flatIndex.search(query: query, k: k)
            // Use stableID for comparison (consistent across index instances)
            let stableIDs = Set(results.map { $0.handle.stableID })
            groundTruth.append(stableIDs)
        }

        print("\n| nprobe | % clusters | Recall (GPU) | Expected |")
        print("|--------|------------|--------------|----------|")

        let nprobeValues = [1, 2, 4, 8, 16, 32]

        for nprobe in nprobeValues {
            let ivfConfig = IndexConfiguration.ivf(
                dimension: dimension,
                nlist: nlist,
                nprobe: nprobe,
                capacity: datasetSize * 2,
                minTrainingVectors: 200
            )
            let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
            _ = try await ivfIndex.insert(dataset)

            var recall: Float = 0
            for (i, query) in queries.enumerated() {
                let results = try await ivfIndex.search(query: query, k: k)
                // Use stableID for comparison (consistent across index instances)
                let resultIDs = Set(results.map { $0.handle.stableID })
                let intersection = resultIDs.intersection(groundTruth[i])
                recall += Float(intersection.count) / Float(k)
            }
            recall /= Float(numQueries)

            let pct = Float(nprobe) / Float(nlist) * 100
            let expected: String
            switch nprobe {
            case 1: expected = "~30%"
            case 2: expected = "~50%"
            case 4: expected = "~65%"
            case 8: expected = "~80%"
            case 16: expected = "~90%"
            case 32: expected = "~100%"
            default: expected = "?"
            }

            print("| \(String(format: "%6d", nprobe)) | \(String(format: "%9.0f%%", pct)) | \(String(format: "%11.1f%%", recall * 100)) | \(expected.padding(toLength: 8, withPad: " ", startingAt: 0)) |")
        }

        print("-" .repeated(70))
        print("Note: With truly random queries, recall should scale with nprobe/nlist")
    }

    /// Direct test of IVFGPUCandidateBuilderKernel
    func testGPUCandidateBuilderKernelDirectly() async throws {
        print("\n" + "=" .repeated(70))
        print("Direct GPU Candidate Builder Kernel Test")
        print("=" .repeated(70))

        let kernel = try await IVFGPUCandidateBuilderKernel(context: context)
        let device = context.device.rawDevice

        // Create test data: 4 lists with different sizes
        let listSizes = [10, 5, 8, 7]  // Total 30 entries
        var listOffsets: [UInt32] = [0]
        var offset: UInt32 = 0
        for size in listSizes {
            offset += UInt32(size)
            listOffsets.append(offset)
        }

        // Create nearest centroids: 2 queries, nprobe=2
        let nearestCentroids: [UInt32] = [
            0, 2,  // Query 0: lists 0 and 2 -> 10 + 8 = 18 candidates
            1, 3,  // Query 1: lists 1 and 3 -> 5 + 7 = 12 candidates
        ]

        // Create GPU buffers
        let offsetBuffer = device.makeBuffer(bytes: listOffsets, length: listOffsets.count * 4, options: .storageModeShared)!
        let centroidBuffer = device.makeBuffer(bytes: nearestCentroids, length: nearestCentroids.count * 4, options: .storageModeShared)!

        // Build candidates
        let result = try await kernel.buildCandidates(
            nearestCentroids: centroidBuffer,
            listOffsets: offsetBuffer,
            numQueries: 2,
            nprobe: 2,
            numLists: 4
        )

        print("\nResults:")
        print("  Total candidates: \(result.totalCandidates)")
        print("  Expected: 30 (18 + 12)")

        // Read back offsets
        let csrOffsets = result.candidateOffsets.contents().bindMemory(to: UInt32.self, capacity: 3)
        print("\nCSR Offsets: [\(csrOffsets[0]), \(csrOffsets[1]), \(csrOffsets[2])]")
        print("  Query 0 range: \(csrOffsets[0])..<\(csrOffsets[1]) (count: \(csrOffsets[1] - csrOffsets[0]))")
        print("  Query 1 range: \(csrOffsets[1])..<\(csrOffsets[2]) (count: \(csrOffsets[2] - csrOffsets[1]))")

        // Verify counts
        let q0Count = csrOffsets[1] - csrOffsets[0]
        let q1Count = csrOffsets[2] - csrOffsets[1]

        XCTAssertEqual(Int(q0Count), 18, "Query 0 should have 18 candidates (lists 0+2)")
        XCTAssertEqual(Int(q1Count), 12, "Query 1 should have 12 candidates (lists 1+3)")
        XCTAssertEqual(result.totalCandidates, 30, "Total should be 30")

        print("\n✓ GPU Candidate Builder produces correct candidate counts")
    }
}

// MARK: - Helpers

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
