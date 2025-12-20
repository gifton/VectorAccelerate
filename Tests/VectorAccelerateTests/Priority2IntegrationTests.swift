// Priority 2 Integration Tests
// Validates the new infrastructure components

import XCTest
@testable import VectorAccelerate
import VectorCore
@preconcurrency import Metal

final class Priority2IntegrationTests: XCTestCase {
    var metalContext: Metal4Context!
    var batchEngine: BatchDistanceEngine!

    override func setUp() async throws {
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        metalContext = try await Metal4Context()
        batchEngine = try await BatchDistanceEngine(context: metalContext)
    }

    override func tearDown() async throws {
        metalContext = nil
        batchEngine = nil
    }

    // MARK: - BufferPool Tests

    func testBufferPoolAllocation() async throws {
        let pool = await metalContext.bufferPool

        // Test acquiring buffers of various sizes
        let token1 = try await pool.getBuffer(size: 1024)
        XCTAssertNotNil(token1.buffer)

        let token2 = try await pool.getBuffer(for: Float.self, count: 256)
        XCTAssertNotNil(token2.buffer)

        // Test with data initialization
        let testData: [Float] = (0..<128).map { Float($0) }
        let token3 = try await pool.getBuffer(with: testData)
        XCTAssertNotNil(token3.buffer)

        // Check statistics
        let stats = await pool.getStatistics()
        XCTAssertGreaterThan(stats.allocationCount, 0)
        print("Buffer pool hit rate: \(stats.hitRate)")
    }

    func testBufferPoolReuse() async throws {
        let pool = await metalContext.bufferPool

        // With RAII BufferToken, we need to manually return to demonstrate reuse
        // Allocate a token, use it, then manually return it
        let token1 = try await pool.getBuffer(size: 4096)
        _ = token1.buffer  // Use the buffer
        token1.returnToPool()  // Explicitly return for immediate reuse

        // Wait briefly for async return to complete
        try await Task.sleep(nanoseconds: 10_000_000)  // 10ms

        // Now allocate again - should get the same buffer (cache hit)
        let token2 = try await pool.getBuffer(size: 4096)
        _ = token2.buffer

        // Check that we got a cache hit
        let stats = await pool.getStatistics()
        XCTAssertGreaterThan(stats.hitCount, 0, "Buffer pool should have cache hits after reuse")
    }

    // MARK: - BatchDistanceOperations Tests

    func testBatchEuclideanDistanceGPU() async throws {
        let query = Array(repeating: Float(1.0), count: 128)
        let candidates = (0..<100).map { i in
            Array(repeating: Float(i) / 100.0, count: 128)
        }

        let distances = try await batchEngine.batchEuclideanDistance(
            query: query,
            candidates: candidates,
            useGPU: true
        )

        XCTAssertEqual(distances.count, candidates.count)

        // Verify distances are reasonable
        for distance in distances {
            XCTAssertGreaterThanOrEqual(distance, 0)
            XCTAssertLessThan(distance, Float.infinity)
        }
    }

    func testBatchCosineSimilarity() async throws {
        let query = [1.0, 0.0, 0.0].map { Float($0) }
        let candidates = [
            [1.0, 0.0, 0.0],  // Same direction
            [0.0, 1.0, 0.0],  // Orthogonal
            [-1.0, 0.0, 0.0], // Opposite direction
        ].map { $0.map { Float($0) } }

        let similarities = try await batchEngine.batchCosineSimilarity(
            query: query,
            candidates: candidates,
            useGPU: false  // Test CPU path
        )

        XCTAssertEqual(similarities.count, 3)
        XCTAssertEqual(similarities[0], 1.0, accuracy: 0.01)  // Same direction
        XCTAssertEqual(similarities[1], 0.0, accuracy: 0.01)  // Orthogonal
        XCTAssertEqual(similarities[2], -1.0, accuracy: 0.01) // Opposite
    }

    func testKNearestNeighbors() async throws {
        let query = Array(repeating: Float(0.5), count: 64)
        let candidates = (0..<20).map { i in
            Array(repeating: Float(i) / 20.0, count: 64)
        }

        let neighbors = try await batchEngine.kNearestNeighbors(
            query: query,
            candidates: candidates,
            k: 5,
            metric: .euclidean
        )

        XCTAssertEqual(neighbors.count, 5)

        // Verify results are sorted by distance
        for i in 1..<neighbors.count {
            XCTAssertLessThanOrEqual(neighbors[i-1].distance, neighbors[i].distance)
        }
    }

    func testGPUvsCPURouting() async throws {
        // Small batch - should use CPU
        let smallQuery = Array(repeating: Float(1.0), count: 32)
        let smallCandidates = (0..<50).map { _ in
            (0..<32).map { _ in Float.random(in: 0...1) }
        }

        let _ = try await batchEngine.batchEuclideanDistance(
            query: smallQuery,
            candidates: smallCandidates
            // useGPU not specified, should auto-route to CPU
        )

        // Large batch - should use GPU
        let largeQuery = Array(repeating: Float(1.0), count: 128)
        let largeCandidates = (0..<2000).map { _ in
            (0..<128).map { _ in Float.random(in: 0...1) }
        }

        let _ = try await batchEngine.batchEuclideanDistance(
            query: largeQuery,
            candidates: largeCandidates
            // useGPU not specified, should auto-route to GPU
        )

        // Check performance metrics
        let metrics = await batchEngine.getPerformanceMetrics()
        print("GPU Speedup: \(metrics.gpuSpeedup)x")
    }

    // MARK: - AccelerateFallback Tests

    func testAccelerateFallbackOperations() async throws {
        let a = (0..<256).map { Float($0) / 256.0 }
        let b = (0..<256).map { Float(1.0) - Float($0) / 256.0 }

        // Test distance operations
        let euclidean = AccelerateFallback.euclideanDistance(a, b)
        XCTAssertGreaterThan(euclidean, 0)

        let cosine = AccelerateFallback.cosineSimilarity(a, b)
        XCTAssertGreaterThanOrEqual(cosine, -1.0)
        XCTAssertLessThanOrEqual(cosine, 1.0)

        let manhattan = AccelerateFallback.manhattanDistance(a, b)
        XCTAssertGreaterThan(manhattan, 0)

        // Test vector operations
        let normalized = AccelerateFallback.normalize(a)
        let norm = AccelerateFallback.dotProduct(normalized, normalized)
        XCTAssertEqual(norm, 1.0, accuracy: 0.001)

        // Test batch operations
        let candidates = [a, b, normalized]
        let batchDistances = AccelerateFallback.batchEuclideanDistance(
            query: a,
            candidates: candidates
        )
        XCTAssertEqual(batchDistances.count, 3)
        XCTAssertEqual(batchDistances[0], 0.0, accuracy: 0.001) // Distance to itself
    }

    func testAccelerateMatrixOperations() async throws {
        // Test matrix-vector multiplication
        let matrix: [Float] = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]
        let vector: [Float] = [1, 0, 1]

        let result = AccelerateFallback.matrixVectorMultiply(
            matrix: matrix,
            vector: vector,
            rows: 3,
            columns: 3
        )

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 4, accuracy: 0.001)  // 1*1 + 2*0 + 3*1 = 4
        XCTAssertEqual(result[1], 10, accuracy: 0.001) // 4*1 + 5*0 + 6*1 = 10
        XCTAssertEqual(result[2], 16, accuracy: 0.001) // 7*1 + 8*0 + 9*1 = 16

        // Test matrix transpose
        let transposed = AccelerateFallback.transpose(
            matrix: matrix,
            rows: 3,
            columns: 3
        )

        XCTAssertEqual(transposed[0], 1)
        XCTAssertEqual(transposed[1], 4)
        XCTAssertEqual(transposed[2], 7)
    }

    func testAccelerateStatisticalOperations() async throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let mean = AccelerateFallback.mean(data)
        XCTAssertEqual(mean, 5.5, accuracy: 0.001)

        let variance = AccelerateFallback.variance(data)
        XCTAssertEqual(variance, 8.25, accuracy: 0.1)

        let stdDev = AccelerateFallback.standardDeviation(data)
        XCTAssertEqual(stdDev, sqrt(8.25), accuracy: 0.1)
    }

    // MARK: - Integration Tests

    func testFullPipelineIntegration() async throws {
        // Create a complete pipeline using Priority 2 components
        let pool = await metalContext.bufferPool

        // Generate test data
        let dimension = 512
        let numCandidates = 1000
        let query = (0..<dimension).map { Float($0) / Float(dimension) }
        let candidates = (0..<numCandidates).map { i in
            (0..<dimension).map { j in
                Float(sin(Double(i * j)) + cos(Double(i + j)))
            }
        }

        // Process using batch operations
        let _ = try await batchEngine.batchEuclideanDistance(
            query: query,
            candidates: candidates
        )

        // Find k-nearest neighbors
        let knn = try await batchEngine.kNearestNeighbors(
            query: query,
            candidates: candidates,
            k: 10,
            metric: .euclidean
        )

        XCTAssertEqual(knn.count, 10)

        // Verify pool statistics show reuse
        let poolStats = await pool.getStatistics()
        print("Integration test - Buffer pool utilization: \(poolStats.memoryUtilization)")
    }
}
