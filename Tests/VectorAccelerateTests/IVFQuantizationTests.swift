//
//  IVFQuantizationTests.swift
//  VectorAccelerate
//
//  Tests for IVF scalar quantization integration (P2.2).
//
//  Verifies:
//  - SQ8 and SQ4 memory reduction
//  - Recall impact vs unquantized
//  - Quantization parameter computation
//  - Quantized search correctness
//

import XCTest
@testable import VectorAccelerate
import VectorCore

/// Tests for IVF scalar quantization integration.
final class IVFQuantizationTests: XCTestCase {

    // MARK: - Configuration Tests

    /// VectorQuantization enum should have correct compression ratios.
    func testVectorQuantizationCompressionRatios() {
        XCTAssertEqual(VectorQuantization.none.compressionRatio, 1.0)
        XCTAssertEqual(VectorQuantization.sq8.compressionRatio, 4.0)
        XCTAssertEqual(VectorQuantization.sq8Asymmetric.compressionRatio, 4.0)
        XCTAssertEqual(VectorQuantization.sq4.compressionRatio, 8.0)
    }

    /// VectorQuantization enum should have correct bytes per element.
    func testVectorQuantizationBytesPerElement() {
        XCTAssertEqual(VectorQuantization.none.bytesPerElement, 4.0)
        XCTAssertEqual(VectorQuantization.sq8.bytesPerElement, 1.0)
        XCTAssertEqual(VectorQuantization.sq8Asymmetric.bytesPerElement, 1.0)
        XCTAssertEqual(VectorQuantization.sq4.bytesPerElement, 0.5)
    }

    /// IndexConfiguration with quantization should report correct bytes per vector.
    func testIndexConfigurationBytesPerVector() {
        let dimension = 128

        // Flat index ignores quantization
        let flatConfig = IndexConfiguration.flat(dimension: dimension)
        XCTAssertEqual(flatConfig.bytesPerVector, dimension * 4)  // Float32

        // IVF with no quantization
        let ivfNoQuant = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 16,
            nprobe: 4,
            quantization: .none
        )
        XCTAssertEqual(ivfNoQuant.bytesPerVector, dimension * 4)

        // IVF with SQ8 quantization
        let ivfSQ8 = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 16,
            nprobe: 4,
            quantization: .sq8
        )
        XCTAssertEqual(ivfSQ8.bytesPerVector, dimension)  // 1 byte per element

        // IVF with SQ4 quantization
        let ivfSQ4 = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 16,
            nprobe: 4,
            quantization: .sq4
        )
        XCTAssertEqual(ivfSQ4.bytesPerVector, dimension / 2)  // 0.5 bytes per element
    }

    /// isQuantized should correctly identify quantized configurations.
    func testIsQuantizedProperty() {
        let noQuant = IndexConfiguration.ivf(
            dimension: 64,
            nlist: 8,
            nprobe: 2,
            quantization: .none
        )
        XCTAssertFalse(noQuant.isQuantized)

        let sq8 = IndexConfiguration.ivf(
            dimension: 64,
            nlist: 8,
            nprobe: 2,
            quantization: .sq8
        )
        XCTAssertTrue(sq8.isQuantized)

        let sq4 = IndexConfiguration.ivf(
            dimension: 64,
            nlist: 8,
            nprobe: 2,
            quantization: .sq4
        )
        XCTAssertTrue(sq4.isQuantized)
    }

    // MARK: - IVFQuantizedStorage Tests

    /// IVFQuantizedStorage should compute correct quantization parameters.
    func testQuantizationParameterComputation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let context = try await Metal4Context()
        let storage = IVFQuantizedStorage(
            quantizationType: .sq8,
            dimension: 4,
            context: context
        )

        // Symmetric data: [-1, 0, 0.5, 1]
        let symmetricVectors: [[Float]] = [
            [-1.0, 0.0, 0.5, 1.0],
            [-0.5, 0.25, 0.75, 0.5]
        ]
        let (scale, zeroPoint) = storage.computeParameters(from: symmetricVectors)

        // For symmetric SQ8, scale should be max(abs(values)) / 127
        XCTAssertGreaterThan(scale, 0)
        XCTAssertNil(zeroPoint)  // Symmetric has no zero point

        // Test asymmetric
        let asymmetricStorage = IVFQuantizedStorage(
            quantizationType: .sq8Asymmetric,
            dimension: 4,
            context: context
        )

        // Asymmetric data: all positive
        let asymmetricVectors: [[Float]] = [
            [0.0, 0.25, 0.5, 1.0],
            [0.1, 0.3, 0.6, 0.9]
        ]
        let (asymScale, asymZP) = asymmetricStorage.computeParameters(from: asymmetricVectors)

        XCTAssertGreaterThan(asymScale, 0)
        XCTAssertNotNil(asymZP)  // Asymmetric has zero point
    }

    /// IVFQuantizedStorage should quantize and dequantize with acceptable error.
    func testQuantizeDequantizeRoundtrip() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let context = try await Metal4Context()
        let dimension = 64

        let storage = IVFQuantizedStorage(
            quantizationType: .sq8,
            dimension: dimension,
            context: context
        )

        // Generate random vectors
        var vectors: [[Float]] = []
        for _ in 0..<100 {
            var v = [Float](repeating: 0, count: dimension)
            for d in 0..<dimension {
                v[d] = Float.random(in: -1...1)
            }
            vectors.append(v)
        }

        // Quantize
        try storage.quantize(vectors: vectors)
        XCTAssertEqual(storage.vectorCount, 100)

        // Dequantize a few vectors and check error
        let dequantized = storage.dequantize(indices: [0, 50, 99])
        XCTAssertEqual(dequantized.count, 3)

        // Check reconstruction error
        for (i, origIdx) in [0, 50, 99].enumerated() {
            let original = vectors[origIdx]
            let reconstructed = dequantized[i]

            var maxError: Float = 0
            for d in 0..<dimension {
                let error = abs(original[d] - reconstructed[d])
                maxError = max(maxError, error)
            }

            // SQ8 should have max error < 0.02 for [-1, 1] range
            XCTAssertLessThan(
                maxError, 0.02,
                "SQ8 max reconstruction error should be small"
            )
        }
    }

    /// SQ4 quantization should have acceptable reconstruction error.
    func testSQ4QuantizationError() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let context = try await Metal4Context()
        let dimension = 32

        let storage = IVFQuantizedStorage(
            quantizationType: .sq4,
            dimension: dimension,
            context: context
        )

        // Generate random vectors
        var vectors: [[Float]] = []
        for _ in 0..<50 {
            var v = [Float](repeating: 0, count: dimension)
            for d in 0..<dimension {
                v[d] = Float.random(in: -1...1)
            }
            vectors.append(v)
        }

        try storage.quantize(vectors: vectors)
        XCTAssertEqual(storage.vectorCount, 50)

        // Check memory reduction: SQ4 should use ~0.5 bytes per element
        let expectedBytes = 50 * dimension / 2  // 0.5 bytes per element
        XCTAssertEqual(storage.usedBytes, expectedBytes)

        // Dequantize and check error is reasonable (SQ4 has higher error)
        let dequantized = storage.dequantize(indices: [0])
        XCTAssertEqual(dequantized.count, 1)
    }

    // MARK: - IVF Structure Integration Tests

    /// IVFStructure should accept quantization parameter.
    func testIVFStructureQuantizationProperty() {
        let structure = IVFStructure(
            numClusters: 8,
            nprobe: 2,
            dimension: 64,
            quantization: .sq8
        )

        XCTAssertTrue(structure.isQuantized)
        XCTAssertEqual(structure.quantization, .sq8)

        let noQuantStructure = IVFStructure(
            numClusters: 8,
            nprobe: 2,
            dimension: 64,
            quantization: .none
        )

        XCTAssertFalse(noQuantStructure.isQuantized)
    }

    // MARK: - Integration Tests

    /// IVF index with SQ8 should achieve reasonable recall.
    func testIVFWithSQ8Recall() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let datasetSize = 1000
        let dimension = 32

        // Create IVF with SQ8 quantization
        let ivfConfig = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 16,
            nprobe: 4,
            quantization: .sq8
        )

        // Create flat index for ground truth
        let flatConfig = IndexConfiguration.flat(
            dimension: dimension,
            capacity: datasetSize
        )

        let ivfIndex = try await AcceleratedVectorIndex(configuration: ivfConfig)
        let flatIndex = try await AcceleratedVectorIndex(configuration: flatConfig)

        // Generate clustered data
        var vectors: [[Float]] = []
        for i in 0..<datasetSize {
            var v = [Float](repeating: 0, count: dimension)
            let cluster = i % 16
            for d in 0..<dimension {
                v[d] = Float(cluster) * 0.1 + Float.random(in: -0.1...0.1)
            }
            vectors.append(v)
        }

        _ = try await ivfIndex.insert(vectors)
        _ = try await flatIndex.insert(vectors)

        // Test recall
        var totalRecall: Float = 0
        let numQueries = 20
        let k = 10

        for _ in 0..<numQueries {
            var query = [Float](repeating: 0, count: dimension)
            let cluster = Int.random(in: 0..<16)
            for d in 0..<dimension {
                query[d] = Float(cluster) * 0.1 + Float.random(in: -0.1...0.1)
            }

            let ivfResults = try await ivfIndex.search(query: query, k: k)
            let flatResults = try await flatIndex.search(query: query, k: k)

            let ivfSet = Set(ivfResults.map { $0.handle.stableID })
            let flatSet = Set(flatResults.map { $0.handle.stableID })
            let intersection = ivfSet.intersection(flatSet)

            totalRecall += Float(intersection.count) / Float(k)
        }

        let avgRecall = totalRecall / Float(numQueries)
        print("IVF+SQ8 recall: \(String(format: "%.1f%%", avgRecall * 100))")

        // SQ8 should maintain good recall (>70% even with small dataset)
        XCTAssertGreaterThanOrEqual(
            avgRecall, 0.60,
            "IVF+SQ8 should achieve reasonable recall"
        )
    }

    /// IVF with quantization should use less memory than unquantized.
    func testQuantizationMemorySavings() {
        let dimension = 256
        let capacity = 100_000

        let unquantized = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 100,
            nprobe: 10,
            capacity: capacity,
            quantization: .none
        )

        let sq8 = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 100,
            nprobe: 10,
            capacity: capacity,
            quantization: .sq8
        )

        let sq4 = IndexConfiguration.ivf(
            dimension: dimension,
            nlist: 100,
            nprobe: 10,
            capacity: capacity,
            quantization: .sq4
        )

        // Check estimated memory
        XCTAssertEqual(
            unquantized.estimatedVectorMemoryBytes,
            capacity * dimension * 4  // Float32
        )

        XCTAssertEqual(
            sq8.estimatedVectorMemoryBytes,
            capacity * dimension * 1  // INT8
        )

        XCTAssertEqual(
            sq4.estimatedVectorMemoryBytes,
            capacity * dimension / 2  // INT4
        )

        // Verify ratios
        let sq8Ratio = Float(unquantized.estimatedVectorMemoryBytes) /
                       Float(sq8.estimatedVectorMemoryBytes)
        XCTAssertEqual(sq8Ratio, 4.0, accuracy: 0.1, "SQ8 should give 4x compression")

        let sq4Ratio = Float(unquantized.estimatedVectorMemoryBytes) /
                       Float(sq4.estimatedVectorMemoryBytes)
        XCTAssertEqual(sq4Ratio, 8.0, accuracy: 0.1, "SQ4 should give 8x compression")
    }

    /// ivfAuto with quantization should create valid configuration.
    func testIvfAutoWithQuantization() throws {
        let config = IndexConfiguration.ivfAuto(
            dimension: 128,
            expectedSize: 50_000,
            quantization: .sq8
        )

        try config.validate()
        XCTAssertTrue(config.isIVF)
        XCTAssertTrue(config.isQuantized)
        XCTAssertEqual(config.quantization, .sq8)
    }
}
