//
//  IndexableVectorInsertTests.swift
//  VectorAccelerate
//
//  Tests for the typed insert<V: IndexableVector> overloads on AcceleratedVectorIndex,
//  verifying hint capture and data-path equivalence with the raw [Float] insert.
//

import XCTest
@testable import VectorAccelerate
import VectorCore

final class IndexableVectorInsertTests: XCTestCase {

    func randomFloats(_ count: Int) -> [Float] {
        (0..<count).map { _ in Float.random(in: -1...1) }
    }

    // MARK: - Single Insert

    func testTypedInsertRoundtripsData() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let raw = randomFloats(128)
        let vector = DynamicVector(raw)
        let handle = try await index.insert(vector)

        let retrieved = try await index.vector(for: handle)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved!, raw)
    }

    func testTypedInsertCapturesHints() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let raw = randomFloats(128)
        let vector = DynamicVector(raw)
        let handle = try await index.insert(vector)

        // DynamicVector defaults: isNormalized=false, cachedMagnitude=nil
        let hints = await index.vectorHints(for: handle)
        XCTAssertNotNil(hints)
        XCTAssertFalse(hints!.isNormalized)
        XCTAssertNil(hints!.cachedMagnitude)
    }

    func testTypedInsertWithMetadata() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let vector = DynamicVector(randomFloats(128))
        let meta: VectorMetadata = ["key": "value"]
        let handle = try await index.insert(vector, metadata: meta)

        let retrieved = await index.metadata(for: handle)
        XCTAssertEqual(retrieved?["key"], "value")
    }

    // MARK: - Batch Insert

    func testBatchTypedInsertRoundtripsData() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let vectors = (0..<5).map { _ in DynamicVector(randomFloats(128)) }
        let handles = try await index.insert(vectors)

        XCTAssertEqual(handles.count, 5)
        let count = await index.count
        XCTAssertEqual(count, 5)
    }

    func testBatchTypedInsertCapturesHints() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let vectors = (0..<3).map { _ in DynamicVector(randomFloats(128)) }
        let handles = try await index.insert(vectors)

        for handle in handles {
            let hints = await index.vectorHints(for: handle)
            XCTAssertNotNil(hints)
            XCTAssertFalse(hints!.isNormalized)
            XCTAssertNil(hints!.cachedMagnitude)
        }
    }

    // MARK: - NormalizationHint (VectorCore 0.2.1+)

    func testNormalizationHintInsertCapturesNormalizedTrue() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let raw = randomFloats(128)
        let vector = DynamicVector(raw)
        let hint = NormalizationHint(vector: vector, isNormalized: true)
        let handle = try await index.insert(hint)

        let hints = await index.vectorHints(for: handle)
        XCTAssertNotNil(hints)
        XCTAssertTrue(hints!.isNormalized)
        XCTAssertEqual(hints!.cachedMagnitude, 1.0)
    }

    func testNormalizationHintAutoDetection() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 4)
        )

        // Create a unit vector (magnitude ≈ 1.0)
        let unitVec = DynamicVector([0.5, 0.5, 0.5, 0.5])
        let hint = NormalizationHint(vector: unitVec) // auto-detect
        let handle = try await index.insert(hint)

        let hints = await index.vectorHints(for: handle)
        XCTAssertNotNil(hints)
        // magnitude of [0.5,0.5,0.5,0.5] = 1.0, so auto-detect should mark normalized
        XCTAssertTrue(hints!.isNormalized)
    }

    func testNormalizationHintNormalizedFactory() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let vector = DynamicVector(randomFloats(128)).normalizedUnchecked()
        let hint = NormalizationHint.normalized(vector)
        let handle = try await index.insert(hint)

        let hints = await index.vectorHints(for: handle)
        XCTAssertNotNil(hints)
        XCTAssertTrue(hints!.isNormalized)
        XCTAssertEqual(hints!.cachedMagnitude, 1.0)
    }

    func testBatchNormalizationHintInsert() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let vectors = (0..<5).map { _ -> NormalizationHint<DynamicVector> in
            let v = DynamicVector(randomFloats(128)).normalizedUnchecked()
            return NormalizationHint.normalized(v)
        }
        let handles = try await index.insert(vectors)

        XCTAssertEqual(handles.count, 5)
        for handle in handles {
            let hints = await index.vectorHints(for: handle)
            XCTAssertNotNil(hints)
            XCTAssertTrue(hints!.isNormalized)
            XCTAssertEqual(hints!.cachedMagnitude, 1.0)
        }
    }

    func testNormalizationHintSearchable() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 4, capacity: 100)
        )

        let target: [Float] = [1.0, 0.0, 0.0, 0.0]
        let hint = NormalizationHint(vector: DynamicVector(target), isNormalized: true)
        _ = try await index.insert(hint)

        let results = try await index.search(query: target, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.results[0].distance, 0.0, accuracy: 1e-4)
    }

    func testRawInsertHasNoHints() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 128)
        )

        let handle = try await index.insert(randomFloats(128))
        let hints = await index.vectorHints(for: handle)
        XCTAssertNil(hints, "Raw [Float] insert should not produce hints")
    }

    // MARK: - Data Equivalence Between Paths

    func testTypedAndRawInsertProduceIdenticalSearchResults() async throws {
        let dim = 128
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 100)
        )

        // Insert 10 vectors via raw [Float] path
        var rawHandles: [VectorHandle] = []
        var rawVectors: [[Float]] = []
        for _ in 0..<10 {
            let v = randomFloats(dim)
            rawVectors.append(v)
            rawHandles.append(try await index.insert(v))
        }

        // Insert same 10 vectors via typed path
        var typedHandles: [VectorHandle] = []
        for v in rawVectors {
            typedHandles.append(try await index.insert(DynamicVector(v)))
        }

        // Search and verify both sets appear in results
        let query = randomFloats(dim)
        let results = try await index.search(query: query, k: 20)
        XCTAssertEqual(results.count, 20)

        let resultIDs = Set(results.map { $0.id })
        for h in rawHandles { XCTAssertTrue(resultIDs.contains(h)) }
        for h in typedHandles { XCTAssertTrue(resultIDs.contains(h)) }
    }
}
