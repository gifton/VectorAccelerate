// L2 Distance Kernel Tests
// Comprehensive testing for GPU-accelerated L2 distance computation

import XCTest
import Metal
@testable import VectorAccelerate
import VectorCore

final class L2DistanceKernelTests: XCTestCase {
    
    var device: (any MTLDevice)?
    var kernel: L2DistanceKernel?
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        kernel = try L2DistanceKernel(device: device!)
    }
    
    // MARK: - Correctness Tests
    
    func testBasicL2Distance() async throws {
        // Test with simple 3D vectors
        let queries: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        let database: [[Float]] = [
            [1.0, 2.0, 3.0],  // Same as query[0], distance = 0
            [4.0, 5.0, 6.0],  // Same as query[1], distance = 0
            [7.0, 8.0, 9.0]   // Different
        ]
        
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 3,
            computeSqrt: true
        )
        
        // Check dimensions
        XCTAssertEqual(distances.count, 2)
        XCTAssertEqual(distances[0].count, 3)
        
        // Check expected distances
        XCTAssertEqual(distances[0][0], 0.0, accuracy: 1e-6, "Same vectors should have distance 0")
        XCTAssertEqual(distances[1][1], 0.0, accuracy: 1e-6, "Same vectors should have distance 0")
        
        // Check query[0] to database[2]: sqrt((7-1)² + (8-2)² + (9-3)²) = sqrt(108) ≈ 10.392
        let expected: Float = sqrt(36.0 + 36.0 + 36.0)
        XCTAssertEqual(distances[0][2], expected, accuracy: 1e-4)
    }
    
    func testSquaredDistance() async throws {
        let queries: [[Float]] = [[3.0, 4.0]]  // Simple 2D vector
        let database: [[Float]] = [[0.0, 0.0]]  // Origin
        
        // Without sqrt: (3-0)² + (4-0)² = 9 + 16 = 25
        let squaredDistances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 2,
            computeSqrt: false
        )
        
        XCTAssertEqual(squaredDistances[0][0], 25.0, accuracy: 1e-6)
        
        // With sqrt: sqrt(25) = 5
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 2,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], 5.0, accuracy: 1e-6)
    }
    
    func testZeroVectors() async throws {
        let queries: [[Float]] = [[0.0, 0.0, 0.0]]
        let database: [[Float]] = [[0.0, 0.0, 0.0]]
        
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 3,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], 0.0, accuracy: 1e-6, "Zero vectors should have distance 0")
    }
    
    func testOrthogonalVectors() async throws {
        let queries: [[Float]] = [[1.0, 0.0, 0.0]]  // Unit vector along x
        let database: [[Float]] = [[0.0, 1.0, 0.0]]  // Unit vector along y
        
        // Distance = sqrt(1² + 1²) = sqrt(2) ≈ 1.414
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 3,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], sqrt(2.0), accuracy: 1e-6)
    }
    
    // MARK: - Dimension-Specific Tests

    /// Test 384-dimension optimized kernel (MiniLM/Sentence-BERT - VectorCore 0.1.5)
    func testDimension384() async throws {
        // Create vectors of dimension 384 to test optimized kernel path
        let queries: [[Float]] = [Array(repeating: 1.0, count: 384)]
        let database: [[Float]] = [Array(repeating: 0.0, count: 384)]

        // Distance = sqrt(384 * 1²) = sqrt(384) ≈ 19.596
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 384,
            computeSqrt: true
        )

        XCTAssertEqual(distances[0][0], sqrt(384.0), accuracy: 1e-3)
    }

    func testDimension512() async throws {
        // Create random vectors of dimension 512
        let queries: [[Float]] = [Array(repeating: 1.0, count: 512)]
        let database: [[Float]] = [Array(repeating: 0.0, count: 512)]
        
        // Distance = sqrt(512 * 1²) = sqrt(512) ≈ 22.627
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 512,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], sqrt(512.0), accuracy: 1e-3)
    }
    
    func testDimension768() async throws {
        // Create normalized vectors
        let value: Float = 1.0 / sqrt(768.0)
        let queries: [[Float]] = [Array(repeating: value, count: 768)]
        let database: [[Float]] = [Array(repeating: -value, count: 768)]
        
        // Distance = sqrt(768 * (2*value)²) = 2
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 768,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], 2.0, accuracy: 1e-3)
    }
    
    func testDimension1536() async throws {
        // Test with sparse vectors
        var query = Array(repeating: Float(0.0), count: 1536)
        var db = Array(repeating: Float(0.0), count: 1536)
        
        // Set a few non-zero elements
        query[0] = 1.0
        query[767] = 1.0
        query[1535] = 1.0
        
        db[0] = 1.0  // One matching element
        
        // Distance = sqrt((0)² + (1)² + (1)²) = sqrt(2)
        let distances = try await kernel!.compute(
            queries: [query],
            database: [db],
            dimension: 1536,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances[0][0], sqrt(2.0), accuracy: 1e-3)
    }
    
    // MARK: - Batch Processing Tests
    
    func testBatchProcessing() async throws {
        let numQueries = 10
        let numDatabase = 20
        let dimension = 128
        
        // Generate random test data
        var queries: [[Float]] = []
        var database: [[Float]] = []
        
        for _ in 0..<numQueries {
            queries.append((0..<dimension).map { _ in Float.random(in: -1...1) })
        }
        
        for _ in 0..<numDatabase {
            database.append((0..<dimension).map { _ in Float.random(in: -1...1) })
        }
        
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: dimension,
            computeSqrt: true
        )
        
        // Verify dimensions
        XCTAssertEqual(distances.count, numQueries)
        XCTAssertEqual(distances[0].count, numDatabase)
        
        // Verify all distances are non-negative
        for row in distances {
            for distance in row {
                XCTAssertGreaterThanOrEqual(distance, 0.0, "L2 distance must be non-negative")
            }
        }
        
        // Verify triangle inequality for a sample
        if numDatabase >= 3 {
            let d01 = distances[0][0]
            let d02 = distances[0][1]
            let d12 = cpuL2Distance(database[0], database[1], sqrt: true)
            
            // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            XCTAssertLessThanOrEqual(d02, d01 + d12 + 1e-3, "Triangle inequality violated")
        }
    }
    
    // MARK: - Edge Cases
    
    func testNonAlignedDimensions() async throws {
        // Test dimensions that are not multiples of 4
        let dimensions = [1, 3, 5, 7, 13, 17, 31, 63, 127, 255, 513]
        
        for dim in dimensions {
            let queries: [[Float]] = [Array(repeating: 1.0, count: dim)]
            let database: [[Float]] = [Array(repeating: 0.0, count: dim)]
            
            let distances = try await kernel!.compute(
                queries: queries,
                database: database,
                dimension: dim,
                computeSqrt: true
            )
            
            let expected = sqrt(Float(dim))
            XCTAssertEqual(distances[0][0], expected, accuracy: 1e-3,
                          "Failed for dimension \(dim)")
        }
    }
    
    func testLargeValues() async throws {
        let largeValue: Float = 1e10
        let queries: [[Float]] = [[largeValue, largeValue]]
        let database: [[Float]] = [[-largeValue, -largeValue]]
        
        // Distance = sqrt((2*1e10)² + (2*1e10)²) = 2*1e10*sqrt(2)
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 2,
            computeSqrt: true
        )
        
        let expected = 2.0 * largeValue * sqrt(2.0)
        XCTAssertEqual(distances[0][0], expected, accuracy: expected * 1e-5)
    }
    
    func testSmallValues() async throws {
        let smallValue: Float = 1e-10
        let queries: [[Float]] = [[smallValue, smallValue]]
        let database: [[Float]] = [[0.0, 0.0]]
        
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            dimension: 2,
            computeSqrt: true
        )
        
        let expected = smallValue * sqrt(2.0)
        XCTAssertEqual(distances[0][0], expected, accuracy: 1e-15)
    }
    
    // MARK: - Performance Tests
    
    func testPerformance512() throws {
        let numQueries = 100
        let numDatabase = 1000
        let dimension = 512
        
        // Prepare data
        let queryData = (0..<numQueries * dimension).map { _ in Float.random(in: -1...1) }
        let databaseData = (0..<numDatabase * dimension).map { _ in Float.random(in: -1...1) }
        
        let queryBuffer = device!.makeBuffer(
            bytes: queryData,
            length: queryData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        let databaseBuffer = device!.makeBuffer(
            bytes: databaseData,
            length: databaseData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        let distanceBuffer = device!.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let params = L2DistanceKernel.Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: true
        )
        
        measure {
            let commandBuffer = device!.makeCommandQueue()!.makeCommandBuffer()!
            
            try! kernel!.compute(
                queryVectors: queryBuffer,
                databaseVectors: databaseBuffer,
                distances: distanceBuffer,
                parameters: params,
                commandBuffer: commandBuffer
            )
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
    
    // MARK: - Helper Functions
    
    private func cpuL2Distance(_ a: [Float], _ b: [Float], sqrt: Bool) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt ? sum.squareRoot() : sum
    }
    
    // MARK: - VectorCore Integration Tests
    
    func testVectorCoreIntegration() async throws {
        // Test with VectorCore types
        let queries = [
            try Vector512(Array(repeating: Float(1.0), count: 512)),
            try Vector512(Array(repeating: Float(0.0), count: 512))
        ]

        let database = [
            try Vector512(Array(repeating: Float(0.0), count: 512)),
            try Vector512(Array(repeating: Float(1.0), count: 512))
        ]
        
        let distances = try await kernel!.compute(
            queries: queries,
            database: database,
            computeSqrt: true
        )
        
        XCTAssertEqual(distances.count, 2)
        XCTAssertEqual(distances[0].count, 2)
        
        // Check diagonal elements (same vectors)
        XCTAssertEqual(distances[0][1], distances[1][0], accuracy: 1e-6,
                      "Distance matrix should be symmetric")
    }
}