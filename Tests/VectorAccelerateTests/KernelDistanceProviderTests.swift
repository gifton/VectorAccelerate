//
//  KernelDistanceProviderTests.swift
//  VectorAccelerateTests
//
//  Tests for the VectorCore DistanceProvider implementations backed by Metal4 kernels.
//

import XCTest
@testable import VectorAccelerate
import VectorCore

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class KernelDistanceProviderTests: XCTestCase {

    var context: Metal4Context!

    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
    }

    override func tearDown() async throws {
        context = nil
        try await super.tearDown()
    }

    // MARK: - L2 Distance Provider Tests

    func testL2KernelDistanceProvider_SingleDistance() async throws {
        let provider = try await L2KernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)
        XCTAssertEqual(distance, sqrt(2), accuracy: 1e-4)
    }

    func testL2KernelDistanceProvider_SameVector() async throws {
        let provider = try await L2KernelDistanceProvider(context: context)

        let v = DynamicVector([1.0, 2.0, 3.0])

        let distance = try await provider.distance(from: v, to: v, metric: .euclidean)
        XCTAssertEqual(distance, 0.0, accuracy: 1e-5)
    }

    func testL2KernelDistanceProvider_BatchDistance() async throws {
        let provider = try await L2KernelDistanceProvider(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let candidates = [
            DynamicVector([1.0, 0.0, 0.0]),  // Distance = 0
            DynamicVector([0.0, 1.0, 0.0]),  // Distance = sqrt(2)
            DynamicVector([2.0, 0.0, 0.0])   // Distance = 1
        ]

        let distances = try await provider.batchDistance(from: query, to: candidates, metric: .euclidean)

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1], sqrt(2), accuracy: 1e-4)
        XCTAssertEqual(distances[2], 1.0, accuracy: 1e-4)
    }

    func testL2KernelDistanceProvider_RejectsWrongMetric() async throws {
        let provider = try await L2KernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        do {
            _ = try await provider.distance(from: v1, to: v2, metric: .cosine)
            XCTFail("Should reject non-euclidean metric")
        } catch {
            // Expected
        }
    }

    // MARK: - Cosine Distance Provider Tests

    func testCosineKernelDistanceProvider_SingleDistance() async throws {
        let provider = try await CosineKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([1.0, 0.0, 0.0])

        // Same direction -> distance = 0 (similarity = 1)
        let distance = try await provider.distance(from: v1, to: v2, metric: .cosine)
        XCTAssertEqual(distance, 0.0, accuracy: 1e-4)
    }

    func testCosineKernelDistanceProvider_OrthogonalVectors() async throws {
        let provider = try await CosineKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        // Orthogonal -> distance = 1 (similarity = 0)
        let distance = try await provider.distance(from: v1, to: v2, metric: .cosine)
        XCTAssertEqual(distance, 1.0, accuracy: 1e-4)
    }

    func testCosineKernelDistanceProvider_OppositeVectors() async throws {
        let provider = try await CosineKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([-1.0, 0.0, 0.0])

        // Opposite -> distance = 2 (similarity = -1)
        let distance = try await provider.distance(from: v1, to: v2, metric: .cosine)
        XCTAssertEqual(distance, 2.0, accuracy: 1e-4)
    }

    func testCosineKernelDistanceProvider_BatchDistance() async throws {
        let provider = try await CosineKernelDistanceProvider(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let candidates = [
            DynamicVector([1.0, 0.0, 0.0]),   // Same -> distance = 0
            DynamicVector([0.0, 1.0, 0.0]),   // Orthogonal -> distance = 1
            DynamicVector([-1.0, 0.0, 0.0])   // Opposite -> distance = 2
        ]

        let distances = try await provider.batchDistance(from: query, to: candidates, metric: .cosine)

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(distances[1], 1.0, accuracy: 1e-4)
        XCTAssertEqual(distances[2], 2.0, accuracy: 1e-4)
    }

    // MARK: - Dot Product Distance Provider Tests

    func testDotProductKernelDistanceProvider_SingleDistance() async throws {
        let provider = try await DotProductKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 2.0, 3.0])
        let v2 = DynamicVector([1.0, 1.0, 1.0])

        // Dot product = 1 + 2 + 3 = 6
        // Distance = -6 (negated for distance semantics)
        let distance = try await provider.distance(from: v1, to: v2, metric: .dotProduct)
        XCTAssertEqual(distance, -6.0, accuracy: 1e-4)
    }

    func testDotProductKernelDistanceProvider_BatchDistance() async throws {
        let provider = try await DotProductKernelDistanceProvider(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let candidates = [
            DynamicVector([1.0, 0.0, 0.0]),  // Dot = 1, distance = -1
            DynamicVector([2.0, 0.0, 0.0]),  // Dot = 2, distance = -2
            DynamicVector([0.0, 1.0, 0.0])   // Dot = 0, distance = 0
        ]

        let distances = try await provider.batchDistance(from: query, to: candidates, metric: .dotProduct)

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], -1.0, accuracy: 1e-4)
        XCTAssertEqual(distances[1], -2.0, accuracy: 1e-4)
        XCTAssertEqual(distances[2], 0.0, accuracy: 1e-4)
    }

    // MARK: - Minkowski Distance Provider Tests

    func testMinkowskiKernelDistanceProvider_ManhattanDistance() async throws {
        let provider = try await MinkowskiKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 2.0, 3.0])
        let v2 = DynamicVector([4.0, 5.0, 6.0])

        // Manhattan: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        let distance = try await provider.distance(from: v1, to: v2, metric: .manhattan)
        XCTAssertEqual(distance, 9.0, accuracy: 0.1)
    }

    func testMinkowskiKernelDistanceProvider_EuclideanDistance() async throws {
        let provider = try await MinkowskiKernelDistanceProvider(context: context)

        let v1 = DynamicVector([0.0, 0.0, 0.0])
        let v2 = DynamicVector([3.0, 4.0, 0.0])

        // Euclidean: sqrt(9 + 16) = 5
        let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)
        XCTAssertEqual(distance, 5.0, accuracy: 0.1)
    }

    func testMinkowskiKernelDistanceProvider_BatchDistance() async throws {
        let provider = try await MinkowskiKernelDistanceProvider(context: context)

        let query = DynamicVector([0.0, 0.0, 0.0])
        let candidates = [
            DynamicVector([1.0, 0.0, 0.0]),  // Manhattan = 1
            DynamicVector([1.0, 1.0, 0.0]),  // Manhattan = 2
            DynamicVector([1.0, 1.0, 1.0])   // Manhattan = 3
        ]

        let distances = try await provider.batchDistance(from: query, to: candidates, metric: .manhattan)

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 1.0, accuracy: 0.1)
        XCTAssertEqual(distances[1], 2.0, accuracy: 0.1)
        XCTAssertEqual(distances[2], 3.0, accuracy: 0.1)
    }

    // MARK: - Universal Distance Provider Tests

    func testUniversalKernelDistanceProvider_AllMetrics() async throws {
        let provider = try await UniversalKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        // Euclidean
        let euclidean = try await provider.distance(from: v1, to: v2, metric: .euclidean)
        XCTAssertEqual(euclidean, sqrt(2), accuracy: 1e-4)

        // Cosine
        let cosine = try await provider.distance(from: v1, to: v2, metric: .cosine)
        XCTAssertEqual(cosine, 1.0, accuracy: 1e-4)  // Orthogonal

        // Dot Product
        let dot = try await provider.distance(from: v1, to: v2, metric: .dotProduct)
        XCTAssertEqual(dot, 0.0, accuracy: 1e-4)  // -0 = 0

        // Manhattan
        let manhattan = try await provider.distance(from: v1, to: v2, metric: .manhattan)
        XCTAssertEqual(manhattan, 2.0, accuracy: 0.1)  // |1| + |1|
    }

    func testUniversalKernelDistanceProvider_BatchDistance() async throws {
        let provider = try await UniversalKernelDistanceProvider(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let candidates = [
            DynamicVector([1.0, 0.0, 0.0]),
            DynamicVector([0.0, 1.0, 0.0])
        ]

        // Test batch with different metrics
        let euclideanDistances = try await provider.batchDistance(from: query, to: candidates, metric: .euclidean)
        XCTAssertEqual(euclideanDistances[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(euclideanDistances[1], sqrt(2), accuracy: 1e-4)

        let cosineDistances = try await provider.batchDistance(from: query, to: candidates, metric: .cosine)
        XCTAssertEqual(cosineDistances[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(cosineDistances[1], 1.0, accuracy: 1e-4)
    }

    func testUniversalKernelDistanceProvider_KernelCaching() async throws {
        let provider = try await UniversalKernelDistanceProvider(context: context)

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        // Call multiple times - kernels should be cached
        for _ in 0..<5 {
            _ = try await provider.distance(from: v1, to: v2, metric: .euclidean)
            _ = try await provider.distance(from: v1, to: v2, metric: .cosine)
        }

        // Should complete without error (kernels are reused)
    }

    // MARK: - Jaccard Distance Provider Tests

    func testJaccardKernelDistanceProvider_SingleDistance() async throws {
        let provider = try await JaccardKernelDistanceProvider(context: context)

        // Binary-like vectors
        let v1 = DynamicVector([1.0, 1.0, 0.0, 0.0])
        let v2 = DynamicVector([1.0, 0.0, 1.0, 0.0])

        // Intersection: {0} -> 1
        // Union: {0, 1, 2} -> 3
        // Jaccard similarity: 1/3
        // Jaccard distance: 1 - 1/3 = 2/3
        let distance = try await provider.distance(v1, v2)
        XCTAssertEqual(distance, 2.0/3.0, accuracy: 0.1)
    }

    func testJaccardKernelDistanceProvider_SameVector() async throws {
        let provider = try await JaccardKernelDistanceProvider(context: context)

        let v = DynamicVector([1.0, 1.0, 0.0, 0.0])

        // Same vector -> distance = 0
        let distance = try await provider.distance(v, v)
        XCTAssertEqual(distance, 0.0, accuracy: 1e-4)
    }

    // MARK: - Context Extension Tests

    func testContextUniversalDistanceProvider() async throws {
        let provider = await context.universalDistanceProvider()

        let v1 = DynamicVector([1.0, 0.0, 0.0])
        let v2 = DynamicVector([0.0, 1.0, 0.0])

        let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)
        XCTAssertEqual(distance, sqrt(2), accuracy: 1e-4)
    }

    // MARK: - Empty Input Tests

    func testBatchDistanceWithEmptyCandidates() async throws {
        let provider = try await L2KernelDistanceProvider(context: context)

        let query = DynamicVector([1.0, 0.0, 0.0])
        let candidates: [DynamicVector] = []

        let distances = try await provider.batchDistance(from: query, to: candidates, metric: .euclidean)
        XCTAssertTrue(distances.isEmpty)
    }
}
