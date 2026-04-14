//
//  DecisionEngineIndexTests.swift
//  VectorAccelerate
//
//  Tests for GPUDecisionEngine integration in AcceleratedVectorIndex.
//  Verifies that the CPU top-K fallback produces results matching the GPU path.
//

import XCTest
@testable import VectorAccelerate
import VectorCore

final class DecisionEngineIndexTests: XCTestCase {

    func randomVector(_ dimension: Int) -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }

    /// Thresholds that force every decision to "no GPU" (CPU fallback).
    func alwaysCPUThresholds() -> GPUActivationThresholds {
        GPUActivationThresholds(
            minVectorsForGPU: Int.max,
            minCandidatesForGPU: Int.max,
            minOperationsForGPU: Int.max
        )
    }

    // MARK: - Default Behavior (no engine)

    func testDefaultBehaviorUsesGPU() async throws {
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: 64, capacity: 500)
        )

        // Insert vectors and search — should work (GPU path, no engine)
        for _ in 0..<20 {
            _ = try await index.insert(randomVector(64))
        }
        let results = try await index.search(query: randomVector(64), k: 5)
        XCTAssertEqual(results.count, 5)
    }

    // MARK: - CPU Fallback: Single Query

    func testCPUFallbackProducesCorrectResults() async throws {
        let dim = 64
        let engine = GPUDecisionEngine(thresholds: alwaysCPUThresholds())
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500),
            decisionEngine: engine
        )

        // Insert known vectors
        var vectors: [[Float]] = []
        var handles: [VectorHandle] = []
        for _ in 0..<30 {
            let v = randomVector(dim)
            vectors.append(v)
            handles.append(try await index.insert(v))
        }

        // Search via CPU fallback
        let query = randomVector(dim)
        let results = try await index.search(query: query, k: 5)

        XCTAssertEqual(results.count, 5)
        // Results should be sorted by distance ascending
        let distances = results.map { $0.distance }
        XCTAssertEqual(distances, distances.sorted())
    }

    // MARK: - CPU vs GPU Equivalence

    func testCPUAndGPUResultsMatch() async throws {
        let dim = 64
        let vectors = (0..<50).map { _ in randomVector(dim) }
        let query = randomVector(dim)

        // GPU index (no decision engine = always GPU)
        let gpuIndex = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500)
        )
        for v in vectors {
            _ = try await gpuIndex.insert(v)
        }
        let gpuResults = try await gpuIndex.search(query: query, k: 10)

        // CPU index (decision engine forces CPU)
        let engine = GPUDecisionEngine(thresholds: alwaysCPUThresholds())
        let cpuIndex = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500),
            decisionEngine: engine
        )
        for v in vectors {
            _ = try await cpuIndex.insert(v)
        }
        let cpuResults = try await cpuIndex.search(query: query, k: 10)

        // Same count
        XCTAssertEqual(gpuResults.count, cpuResults.count)

        // Same handles in same order (both return exact L2² distances)
        for (gpu, cpu) in zip(gpuResults, cpuResults) {
            XCTAssertEqual(gpu.id, cpu.id, "Handle mismatch: GPU=\(gpu.id) CPU=\(cpu.id)")
            XCTAssertEqual(gpu.distance, cpu.distance, accuracy: 1e-4,
                "Distance mismatch for handle \(gpu.id): GPU=\(gpu.distance) CPU=\(cpu.distance)")
        }
    }

    // MARK: - CPU Fallback: Batch Query

    func testBatchCPUFallbackProducesCorrectResults() async throws {
        let dim = 64
        let engine = GPUDecisionEngine(thresholds: alwaysCPUThresholds())
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500),
            decisionEngine: engine
        )

        for _ in 0..<30 {
            _ = try await index.insert(randomVector(dim))
        }

        let queries = (0..<3).map { _ in randomVector(dim) }
        let batchResults = try await index.search(queries: queries, k: 5)

        XCTAssertEqual(batchResults.count, 3)
        for results in batchResults {
            XCTAssertEqual(results.count, 5)
            let distances = results.map { $0.distance }
            XCTAssertEqual(distances, distances.sorted())
        }
    }

    // MARK: - CPU Fallback with Deletions

    func testCPUFallbackRespectsDeletedVectors() async throws {
        let dim = 64
        let engine = GPUDecisionEngine(thresholds: alwaysCPUThresholds())
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500),
            decisionEngine: engine
        )

        var handles: [VectorHandle] = []
        for _ in 0..<10 {
            handles.append(try await index.insert(randomVector(dim)))
        }

        // Delete some vectors
        try await index.remove(handles[2])
        try await index.remove(handles[5])

        let results = try await index.search(query: randomVector(dim), k: 10)
        let resultIDs = Set(results.map { $0.id })

        // Deleted handles must not appear
        XCTAssertFalse(resultIDs.contains(handles[2]))
        XCTAssertFalse(resultIDs.contains(handles[5]))
        XCTAssertEqual(results.count, 8) // 10 - 2 deleted
    }

    // MARK: - Nil Engine Preserves GPU Path

    func testNilEnginePreservesExistingGPUBehavior() async throws {
        let dim = 64
        let index = try await AcceleratedVectorIndex(
            configuration: .flat(dimension: dim, capacity: 500),
            decisionEngine: nil
        )

        for _ in 0..<20 {
            _ = try await index.insert(randomVector(dim))
        }

        // Should work identically to the no-argument init
        let results = try await index.search(query: randomVector(dim), k: 5)
        XCTAssertEqual(results.count, 5)
    }
}
