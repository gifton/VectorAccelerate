//
//  SparseLogTFIDFKernelTests.swift
//  VectorAccelerateTests
//
//  Tests for SparseLogTFIDFKernel - c-TF-IDF computation.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class SparseLogTFIDFKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: SparseLogTFIDFKernel!

    override func setUp() async throws {
        try await super.setUp()

        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        context = try await Metal4Context()
        kernel = try await SparseLogTFIDFKernel(context: context)
    }

    override func tearDown() async throws {
        kernel = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - Basic Correctness Tests

    /// Test basic c-TF-IDF formula correctness.
    func testCTFIDFCorrectness() async throws {
        // Simple test case from spec
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [0, 1], frequencies: [5.0, 3.0]),
            ClusterTermFrequencies(termIndices: [1, 2], frequencies: [2.0, 4.0])
        ]
        let corpusFreqs: [Float] = [10.0, 20.0, 5.0]  // 3-term vocabulary
        let avgClusterSize: Float = 10.0

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        // Expected: tf * log(1 + avg / corpusTf)
        // Cluster 0, term 0: 5 * log(1 + 10/10) = 5 * log(2) ≈ 3.466
        // Cluster 0, term 1: 3 * log(1 + 10/20) = 3 * log(1.5) ≈ 1.216
        // Cluster 1, term 1: 2 * log(1 + 10/20) = 2 * log(1.5) ≈ 0.811
        // Cluster 1, term 2: 4 * log(1 + 10/5) = 4 * log(3) ≈ 4.394

        XCTAssertEqual(result.scoresPerCluster.count, 2)
        XCTAssertEqual(result.scoresPerCluster[0].count, 2)
        XCTAssertEqual(result.scoresPerCluster[1].count, 2)

        let expected00 = Float(5.0 * log(2.0))
        let expected01 = Float(3.0 * log(1.5))
        let expected10 = Float(2.0 * log(1.5))
        let expected11 = Float(4.0 * log(3.0))

        XCTAssertEqual(result.scoresPerCluster[0][0], expected00, accuracy: 1e-5)
        XCTAssertEqual(result.scoresPerCluster[0][1], expected01, accuracy: 1e-5)
        XCTAssertEqual(result.scoresPerCluster[1][0], expected10, accuracy: 1e-5)
        XCTAssertEqual(result.scoresPerCluster[1][1], expected11, accuracy: 1e-5)
    }

    /// Test that vectorized kernel produces same results as scalar.
    func testVectorizedMatchesScalar() async throws {
        // Create data that's divisible by 4 (triggers vectorized path)
        // and data that's not (triggers scalar path)
        let clusterTerms16 = [
            ClusterTermFrequencies(
                termIndices: Array(0..<16).map { UInt32($0) },
                frequencies: Array(0..<16).map { Float($0 + 1) }
            )
        ]

        let clusterTerms17 = [
            ClusterTermFrequencies(
                termIndices: Array(0..<17).map { UInt32($0) },
                frequencies: Array(0..<17).map { Float($0 + 1) }
            )
        ]

        let corpusFreqs: [Float] = Array(0..<20).map { Float($0 + 5) }
        let avgClusterSize: Float = 50.0

        let result16 = try await kernel.compute(
            clusterTerms: clusterTerms16,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        let result17 = try await kernel.compute(
            clusterTerms: clusterTerms17,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        // First 16 elements should match between both
        for i in 0..<16 {
            XCTAssertEqual(
                result16.scoresPerCluster[0][i],
                result17.scoresPerCluster[0][i],
                accuracy: 1e-5,
                "Mismatch at index \(i)"
            )
        }
    }

    // MARK: - Top-K Extraction Tests

    /// Test top-K term extraction.
    func testTopKExtraction() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(
                termIndices: [0, 1, 2, 3, 4],
                frequencies: [1.0, 5.0, 2.0, 4.0, 3.0]
            )
        ]
        // Uniform corpus frequencies means TF determines ranking
        let corpusFreqs: [Float] = [10.0, 10.0, 10.0, 10.0, 10.0]

        let topK = try await kernel.topKPerCluster(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0,
            k: 3
        )

        // Highest tf values: 5 (term 1), 4 (term 3), 3 (term 4)
        XCTAssertEqual(topK.topKPerCluster.count, 1)
        XCTAssertEqual(topK.topKPerCluster[0].count, 3)
        XCTAssertEqual(topK.topKPerCluster[0][0].termIndex, 1, "Highest score should be term 1")
        XCTAssertEqual(topK.topKPerCluster[0][1].termIndex, 3, "Second highest should be term 3")
        XCTAssertEqual(topK.topKPerCluster[0][2].termIndex, 4, "Third highest should be term 4")

        // Verify scores are in descending order
        XCTAssertGreaterThan(topK.topKPerCluster[0][0].score, topK.topKPerCluster[0][1].score)
        XCTAssertGreaterThan(topK.topKPerCluster[0][1].score, topK.topKPerCluster[0][2].score)
    }

    /// Test top-K with multiple clusters.
    func testTopKMultipleClusters() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [0, 1, 2], frequencies: [5.0, 3.0, 1.0]),
            ClusterTermFrequencies(termIndices: [2, 3, 4], frequencies: [2.0, 6.0, 4.0])
        ]
        let corpusFreqs: [Float] = [10.0, 10.0, 10.0, 10.0, 10.0]

        let topK = try await kernel.topKPerCluster(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0,
            k: 2
        )

        XCTAssertEqual(topK.topKPerCluster.count, 2)

        // Cluster 0: term 0 (5.0) > term 1 (3.0)
        XCTAssertEqual(topK.topKPerCluster[0][0].termIndex, 0)
        XCTAssertEqual(topK.topKPerCluster[0][1].termIndex, 1)

        // Cluster 1: term 3 (6.0) > term 4 (4.0)
        XCTAssertEqual(topK.topKPerCluster[1][0].termIndex, 3)
        XCTAssertEqual(topK.topKPerCluster[1][1].termIndex, 4)
    }

    /// Test when K is larger than available terms.
    func testTopKLargerThanAvailable() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [0, 1], frequencies: [3.0, 5.0])
        ]
        let corpusFreqs: [Float] = [10.0, 10.0, 10.0]

        let topK = try await kernel.topKPerCluster(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0,
            k: 5  // Only 2 terms available
        )

        XCTAssertEqual(topK.topKPerCluster.count, 1)
        // Should only return 2 valid entries
        XCTAssertEqual(topK.topKPerCluster[0].count, 2)
        XCTAssertEqual(topK.topKPerCluster[0][0].termIndex, 1)  // Higher score
        XCTAssertEqual(topK.topKPerCluster[0][1].termIndex, 0)
    }

    // MARK: - Edge Cases

    /// Test empty cluster handling.
    func testEmptyCluster() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [], frequencies: []),
            ClusterTermFrequencies(termIndices: [0, 1], frequencies: [2.0, 3.0])
        ]
        let corpusFreqs: [Float] = [10.0, 10.0]

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0
        )

        XCTAssertEqual(result.scoresPerCluster.count, 2)
        XCTAssertEqual(result.scoresPerCluster[0].count, 0)  // Empty cluster
        XCTAssertEqual(result.scoresPerCluster[1].count, 2)

        // Top-K with empty cluster
        let topK = try await kernel.topKPerCluster(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0,
            k: 3
        )

        XCTAssertEqual(topK.topKPerCluster[0].count, 0)  // Empty cluster returns nothing
        XCTAssertEqual(topK.topKPerCluster[1].count, 2)
    }

    /// Test all empty clusters.
    func testAllEmptyClusters() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [], frequencies: []),
            ClusterTermFrequencies(termIndices: [], frequencies: [])
        ]
        let corpusFreqs: [Float] = [10.0, 10.0]

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0
        )

        XCTAssertEqual(result.nnz, 0)
        XCTAssertEqual(result.scoresPerCluster.count, 2)
        XCTAssertEqual(result.scoresPerCluster[0].count, 0)
        XCTAssertEqual(result.scoresPerCluster[1].count, 0)
    }

    /// Test single term in cluster.
    func testSingleTerm() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [5], frequencies: [10.0])
        ]
        let corpusFreqs: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 25.0]

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 20.0
        )

        // tf=10, corpusTf=25, avg=20
        // score = 10 * log(1 + 20/25) = 10 * log(1.8)
        let expected = Float(10.0 * log(1.8))
        XCTAssertEqual(result.scoresPerCluster[0][0], expected, accuracy: 1e-5)
    }

    /// Test division by zero protection (corpus frequency = 0).
    func testDivisionByZeroProtection() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [0], frequencies: [5.0])
        ]
        // Corpus frequency of 0 - should use max(corpusTf, 1.0) = 1.0
        let corpusFreqs: [Float] = [0.0]

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: 10.0
        )

        // Should use corpusTf=1.0: 5 * log(1 + 10/1) = 5 * log(11)
        let expected = Float(5.0 * log(11.0))
        XCTAssertEqual(result.scoresPerCluster[0][0], expected, accuracy: 1e-5)
        XCTAssertFalse(result.scoresPerCluster[0][0].isNaN)
        XCTAssertFalse(result.scoresPerCluster[0][0].isInfinite)
    }

    // MARK: - Supporting Types Tests

    /// Test ClusterTermFrequencies initialization from dictionary.
    func testClusterTermFrequenciesFromDictionary() {
        let dict: [Int: Int] = [0: 5, 3: 10, 7: 2]
        let ctf = ClusterTermFrequencies(from: dict)

        XCTAssertEqual(ctf.count, 3)
        XCTAssertEqual(ctf.termIndices.count, 3)
        XCTAssertEqual(ctf.frequencies.count, 3)

        // Verify values are correctly converted
        for (idx, termIdx) in ctf.termIndices.enumerated() {
            let freq = ctf.frequencies[idx]
            XCTAssertEqual(Float(dict[Int(termIdx)]!), freq)
        }
    }

    /// Test ClusterTermFrequencies from Float dictionary.
    func testClusterTermFrequenciesFromFloatDictionary() {
        let dict: [Int: Float] = [1: 2.5, 4: 3.7]
        let ctf = ClusterTermFrequencies(fromFloats: dict)

        XCTAssertEqual(ctf.count, 2)
    }

    // MARK: - Performance Tests

    /// Test with realistic topic modeling workload.
    func testRealisticWorkload() async throws {
        // 100 clusters, ~50 terms per cluster on average, 10K vocabulary
        let numClusters = 100
        let vocabSize = 10_000

        var clusterTerms: [ClusterTermFrequencies] = []
        clusterTerms.reserveCapacity(numClusters)

        for _ in 0..<numClusters {
            // Random number of terms (30-70)
            let numTerms = Int.random(in: 30...70)
            var termSet = Set<UInt32>()
            while termSet.count < numTerms {
                termSet.insert(UInt32.random(in: 0..<UInt32(vocabSize)))
            }
            let terms = Array(termSet)
            let freqs = (0..<numTerms).map { _ in Float.random(in: 1...100) }
            clusterTerms.append(ClusterTermFrequencies(termIndices: terms, frequencies: freqs))
        }

        let corpusFreqs = (0..<vocabSize).map { _ in Float.random(in: 1...1000) }
        let avgClusterSize: Float = 500.0

        // Measure compute time
        let computeResult = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        print("Realistic workload (100 clusters, ~50 terms/cluster):")
        print("  nnz: \(computeResult.nnz)")
        print("  Compute time: \(String(format: "%.3f", computeResult.executionTime * 1000))ms")
        print("  Throughput: \(String(format: "%.2f", computeResult.throughputGBps)) GB/s")

        // Measure top-K time
        let topKResult = try await kernel.topKPerCluster(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize,
            k: 10
        )

        print("  Top-10 extraction time: \(String(format: "%.3f", topKResult.executionTime * 1000))ms")

        XCTAssertEqual(computeResult.clusterCount, numClusters)
        XCTAssertEqual(topKResult.clusterCount, numClusters)

        // Basic sanity: most clusters should have 10 top terms (unless < 10 terms)
        for (i, topK) in topKResult.topKPerCluster.enumerated() {
            let clusterSize = clusterTerms[i].count
            let expectedCount = min(clusterSize, 10)
            XCTAssertEqual(topK.count, expectedCount,
                           "Cluster \(i) should have \(expectedCount) top terms, got \(topK.count)")
        }
    }

    /// Stress test with larger workload.
    func testLargerWorkload() async throws {
        let numClusters = 200
        let vocabSize = 50_000

        var clusterTerms: [ClusterTermFrequencies] = []
        clusterTerms.reserveCapacity(numClusters)

        for _ in 0..<numClusters {
            let numTerms = Int.random(in: 50...150)
            var termSet = Set<UInt32>()
            while termSet.count < numTerms {
                termSet.insert(UInt32.random(in: 0..<UInt32(vocabSize)))
            }
            let terms = Array(termSet)
            let freqs = (0..<numTerms).map { _ in Float.random(in: 1...100) }
            clusterTerms.append(ClusterTermFrequencies(termIndices: terms, frequencies: freqs))
        }

        let corpusFreqs = (0..<vocabSize).map { _ in Float.random(in: 1...1000) }
        let avgClusterSize: Float = 1000.0

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        print("Larger workload (200 clusters, ~100 terms/cluster, 50K vocab):")
        print("  nnz: \(result.nnz)")
        print("  Compute time: \(String(format: "%.3f", result.executionTime * 1000))ms")
        print("  Throughput: \(String(format: "%.2f", result.throughputGBps)) GB/s")

        XCTAssertEqual(result.clusterCount, numClusters)
    }

    // MARK: - Encode API Tests

    /// Test encode API for kernel fusion via buffer API.
    func testEncodeAPI() async throws {
        // Use buffer API which internally uses the encode API
        let termIndices: [UInt32] = [0, 1, 2, 3]
        let termFreqs: [Float] = [1.0, 2.0, 3.0, 4.0]
        let corpusFreqs: [Float] = [10.0, 10.0, 10.0, 10.0]
        let nnz = termIndices.count

        let device = context.device.rawDevice

        guard let indicesBuffer = termIndices.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }),
        let freqsBuffer = termFreqs.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }),
        let corpusBuffer = corpusFreqs.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Use the buffer API which tests the encode path internally
        let scoresBuffer = try await kernel.compute(
            termIndices: indicesBuffer,
            termFreqs: freqsBuffer,
            corpusFreqs: corpusBuffer,
            avgClusterSize: 10.0,
            nnz: nnz
        )

        // Verify results
        let scoresPtr = scoresBuffer.contents().bindMemory(to: Float.self, capacity: nnz)
        for i in 0..<nnz {
            let expected = Float(Float(i + 1) * log(2.0))  // tf * log(1 + 10/10)
            XCTAssertEqual(scoresPtr[i], expected, accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }

    // MARK: - Reference Implementation

    /// CPU reference for validation.
    private func cpuCTFIDF(
        termFreq: Float,
        corpusFreq: Float,
        avgClusterSize: Float
    ) -> Float {
        let idf = log(1.0 + avgClusterSize / max(corpusFreq, 1.0))
        return termFreq * idf
    }

    /// Comprehensive correctness test vs CPU reference.
    func testAgainstCPUReference() async throws {
        let clusterTerms = [
            ClusterTermFrequencies(termIndices: [0, 2, 5], frequencies: [3.0, 7.0, 1.0]),
            ClusterTermFrequencies(termIndices: [1, 3, 4, 6], frequencies: [2.0, 4.0, 8.0, 5.0])
        ]
        let corpusFreqs: [Float] = [15.0, 8.0, 25.0, 12.0, 3.0, 50.0, 20.0]
        let avgClusterSize: Float = 30.0

        let result = try await kernel.compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFreqs,
            avgClusterSize: avgClusterSize
        )

        // Verify against CPU reference
        for (clusterIdx, cluster) in clusterTerms.enumerated() {
            for (termIdx, termId) in cluster.termIndices.enumerated() {
                let tf = cluster.frequencies[termIdx]
                let corpusTf = corpusFreqs[Int(termId)]
                let expected = cpuCTFIDF(termFreq: tf, corpusFreq: corpusTf, avgClusterSize: avgClusterSize)
                let actual = result.scoresPerCluster[clusterIdx][termIdx]

                XCTAssertEqual(actual, expected, accuracy: 1e-5,
                    "Mismatch at cluster \(clusterIdx), term \(termId): expected \(expected), got \(actual)")
            }
        }
    }
}
