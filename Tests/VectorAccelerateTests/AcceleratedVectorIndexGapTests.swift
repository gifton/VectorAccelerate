//
//  AcceleratedVectorIndexGapTests.swift
//  VectorAccelerate
//
//  Tests for three gap fixes: WAL factory threading, GPUDecisionEngine routing,
//  and IndexableVector insert overloads.
//

import XCTest
@testable import VectorAccelerate
import VectorCore

// MARK: - Test Helper

/// Minimal IndexableVector conformer with configurable normalization hints.
private struct HintedVector: IndexableVector, @unchecked Sendable {
    typealias Scalar = Float
    typealias Storage = [Float]

    var storage: [Float]
    var scalarCount: Int { storage.count }

    let isNormalized: Bool
    let cachedMagnitude: Float?

    init() {
        storage = []
        isNormalized = false
        cachedMagnitude = nil
    }

    init(_ array: [Float]) throws {
        storage = array
        isNormalized = false
        cachedMagnitude = nil
    }

    init(repeating value: Float) {
        storage = []
        isNormalized = false
        cachedMagnitude = nil
    }

    init(_ array: [Float], isNormalized: Bool, cachedMagnitude: Float? = nil) {
        self.storage = array
        self.isNormalized = isNormalized
        self.cachedMagnitude = cachedMagnitude
    }

    func toArray() -> [Float] { storage }

    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { (ptr: inout UnsafeMutableBufferPointer<Float>) in
            try body(UnsafeMutableBufferPointer(start: ptr.baseAddress, count: ptr.count))
        }
    }

    var startIndex: Int { 0 }
    var endIndex: Int { storage.count }
    func index(after i: Int) -> Int { i + 1 }
    subscript(position: Int) -> Float { storage[position] }
}

// MARK: - Gap 1: WAL Factory Threading

final class WALFactoryTests: XCTestCase {

    func testFlatFactoryThreadsWAL() {
        let walDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-wal-\(UUID().uuidString)")
        let config = IndexConfiguration.flat(
            dimension: 4,
            walConfiguration: .enabled(directory: walDir)
        )
        XCTAssertTrue(config.walConfiguration.enabled)
        XCTAssertEqual(config.walConfiguration.directory, walDir)
    }

    func testIVFFactoryThreadsWAL() {
        let walDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-wal-\(UUID().uuidString)")
        let config = IndexConfiguration.ivf(
            dimension: 4,
            nlist: 4,
            nprobe: 2,
            walConfiguration: .enabled(directory: walDir, syncMode: .immediate)
        )
        XCTAssertTrue(config.walConfiguration.enabled)
        XCTAssertEqual(config.walConfiguration.directory, walDir)
        XCTAssertEqual(config.walConfiguration.syncMode, .immediate)
    }

    func testIVFAutoFactoryThreadsWAL() {
        let walDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-wal-\(UUID().uuidString)")
        let config = IndexConfiguration.ivfAuto(
            dimension: 4,
            expectedSize: 10_000,
            walConfiguration: .enabled(directory: walDir, autoCheckpointThreshold: 5000)
        )
        XCTAssertTrue(config.walConfiguration.enabled)
        XCTAssertEqual(config.walConfiguration.directory, walDir)
        XCTAssertEqual(config.walConfiguration.autoCheckpointThreshold, 5000)
    }

    func testFactoryDefaultsToDisabledWAL() {
        let flat = IndexConfiguration.flat(dimension: 4)
        XCTAssertFalse(flat.walConfiguration.enabled)

        let ivf = IndexConfiguration.ivf(dimension: 4, nlist: 4, nprobe: 2)
        XCTAssertFalse(ivf.walConfiguration.enabled)

        let ivfAuto = IndexConfiguration.ivfAuto(dimension: 4, expectedSize: 10_000)
        XCTAssertFalse(ivfAuto.walConfiguration.enabled)
    }
}

// MARK: - Gap 2: GPUDecisionEngine Routing

final class DecisionEngineRoutingTests: XCTestCase {

    func testCPUFallbackProducesCorrectResults() async throws {
        let engine = GPUDecisionEngine(
            thresholds: GPUActivationThresholds(minVectorsForGPU: Int.max)
        )
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(
            configuration: config,
            decisionEngine: engine
        )

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])
        _ = try await index.insert([0.0, 1.0, 0.0, 0.0])
        _ = try await index.insert([1.0, 1.0, 0.0, 0.0])

        let results = try await index.search(query: [1.0, 0.0, 0.0, 0.0], k: 3)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results.results[0].distance, 0.0, accuracy: 1e-4)
        XCTAssertEqual(results.results[1].distance, 1.0, accuracy: 1e-4)
        XCTAssertEqual(results.results[2].distance, 2.0, accuracy: 1e-4)
    }

    func testFilteredCPUFallback() async throws {
        let engine = GPUDecisionEngine(
            thresholds: GPUActivationThresholds(minVectorsForGPU: Int.max)
        )
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(
            configuration: config,
            decisionEngine: engine
        )

        let h1 = try await index.insert([1.0, 0.0, 0.0, 0.0], metadata: ["label": "keep"])
        _ = try await index.insert([0.0, 1.0, 0.0, 0.0], metadata: ["label": "skip"])
        let h3 = try await index.insert([1.0, 1.0, 0.0, 0.0], metadata: ["label": "keep"])

        let results = try await index.search(
            query: [1.0, 0.0, 0.0, 0.0],
            k: 10,
            filter: { _, meta in meta?["label"] == "keep" }
        )

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.results[0].id, h1)
        XCTAssertEqual(results.results[1].id, h3)
    }

    func testNilEngineUsesGPUPath() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        _ = try await index.insert([1.0, 0.0, 0.0, 0.0])
        _ = try await index.insert([0.0, 1.0, 0.0, 0.0])

        let results = try await index.search(query: [1.0, 0.0, 0.0, 0.0], k: 2)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.results[0].distance, 0.0, accuracy: 1e-4)
    }
}

// MARK: - Gap 3: IndexableVector Insert Overloads

final class IndexableVectorInsertTests: XCTestCase {

    func testNormalizedHintPreserved() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vec = HintedVector([0.5, 0.5, 0.5, 0.5], isNormalized: true)
        let handle = try await index.insert(vec)

        let normalized = await index.isVectorNormalized(handle)
        XCTAssertEqual(normalized, true)
    }

    func testUnnormalizedHintPreserved() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vec = HintedVector([2.0, 3.0, 4.0, 5.0], isNormalized: false, cachedMagnitude: 7.348)
        let handle = try await index.insert(vec)

        let normalized = await index.isVectorNormalized(handle)
        XCTAssertEqual(normalized, false)
    }

    func testBatchInsertPreservesHints() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vectors = [
            HintedVector([1.0, 0.0, 0.0, 0.0], isNormalized: true),
            HintedVector([2.0, 3.0, 0.0, 0.0], isNormalized: false),
            HintedVector([0.0, 0.0, 1.0, 0.0], isNormalized: true),
        ]
        let handles = try await index.insert(vectors)

        let n0 = await index.isVectorNormalized(handles[0])
        let n1 = await index.isVectorNormalized(handles[1])
        let n2 = await index.isVectorNormalized(handles[2])
        XCTAssertEqual(n0, true)
        XCTAssertEqual(n1, false)
        XCTAssertEqual(n2, true)
    }

    func testPlainFloatInsertDefaultsFalse() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let handle = try await index.insert([1.0, 0.0, 0.0, 0.0])

        let normalized = await index.isVectorNormalized(handle)
        XCTAssertEqual(normalized, false)
    }

    func testGenericInsertSearchable() async throws {
        let config = IndexConfiguration.flat(dimension: 4, capacity: 100)
        let index = try await AcceleratedVectorIndex(configuration: config)

        let vec = HintedVector([1.0, 0.0, 0.0, 0.0], isNormalized: true)
        _ = try await index.insert(vec)

        let results = try await index.search(query: [1.0, 0.0, 0.0, 0.0], k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.results[0].distance, 0.0, accuracy: 1e-4)
    }
}
