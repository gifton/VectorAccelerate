//
//  GPUDecisionEngineTests.swift
//  VectorAccelerateTests
//
//  Tests for GPUDecisionEngine adaptive GPU/CPU routing
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

final class GPUDecisionEngineTests: XCTestCase {

    private var engine: GPUDecisionEngine!

    override func setUp() async throws {
        try await super.setUp()
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }
        engine = GPUDecisionEngine()
    }

    override func tearDown() {
        engine = nil
        super.tearDown()
    }

    // MARK: - Threshold-Based Decision Tests

    func test_shouldUseGPU_belowMinVectors() async throws {
        // vectorCount=100 is below default minVectorsForGPU=1000
        let result = await engine.shouldUseGPU(
            operation: .l2Distance,
            vectorCount: 100,
            candidateCount: 2000,
            k: 50
        )
        XCTAssertFalse(result)
    }

    func test_shouldUseGPU_belowMinCandidates() async throws {
        // candidateCount=100 is below default minCandidatesForGPU=500
        let result = await engine.shouldUseGPU(
            operation: .l2Distance,
            vectorCount: 5000,
            candidateCount: 100,
            k: 50
        )
        XCTAssertFalse(result)
    }

    func test_shouldUseGPU_kGatesIgnoredForDistanceOps() async throws {
        // AUDIT-2 VA2-003: distance-shaped operations have no k — the k-gates apply only to
        // selection ops. Before the fix, batchDistance's k=0 consultations were unconditionally
        // refused, making the provider's batch GPU path unreachable. Any k value must be
        // irrelevant here (complexity gates on q·n·dimension instead).
        for k in [0, 5, 2000] {
            let result = await engine.shouldUseGPU(
                operation: .l2Distance,
                vectorCount: 5000,
                candidateCount: 2000,
                k: k,
                dimension: 128
            )
            XCTAssertTrue(result, "distance op refused with k=\(k) — k-gates must not apply")
        }
    }

    func test_shouldUseGPU_belowMinK_selectionOp() async throws {
        // k=5 is below default minKForGPU=10; the k-gate still applies to selection ops.
        let result = await engine.shouldUseGPU(
            operation: .topKSelection,
            vectorCount: 5000,
            candidateCount: 2000,
            k: 5
        )
        XCTAssertFalse(result)
    }

    func test_shouldUseGPU_aboveMaxK_selectionOp() async throws {
        // k=2000 exceeds default maxKForGPU=1000; the k-gate still applies to selection ops.
        let result = await engine.shouldUseGPU(
            operation: .topKSelection,
            vectorCount: 5000,
            candidateCount: 2000,
            k: 2000
        )
        XCTAssertFalse(result)
    }

    func test_shouldUseGPU_belowMinOperations() async throws {
        // Distance-shaped complexity gate: operationCount = queryCount(1) * candidateCount(500)
        // * dimension(64) = 32_000, below default minOperationsForGPU=50_000.
        let result = await engine.shouldUseGPU(
            operation: .l2Distance,
            vectorCount: 5000,
            candidateCount: 500,
            k: 0,
            dimension: 64
        )
        XCTAssertFalse(result)
    }

    func test_shouldUseGPU_allThresholdsMet() async throws {
        // All thresholds satisfied:
        //   vectorCount=5000 >= 1000
        //   candidateCount=2000 >= 500
        //   k=50 in [10, 1000]
        //   operationCount = 1 * 2000 * 50 = 100000 >= 50000
        //   l2Distance specific: candidateCount(2000) * dimension(128) = 256000 >= 100000
        let result = await engine.shouldUseGPU(
            operation: .l2Distance,
            vectorCount: 5000,
            candidateCount: 2000,
            k: 50,
            dimension: 128
        )
        XCTAssertTrue(result)
    }

    // MARK: - Performance Recording Tests

    func test_recordPerformance_updatesStats() async throws {
        // Record 5 performance samples for l2Distance
        for i in 1...5 {
            await engine.recordPerformance(
                operation: .l2Distance,
                cpuTime: Double(i) * 0.01,
                gpuTime: Double(i) * 0.005
            )
        }

        let stats = await engine.getPerformanceStats()
        XCTAssertEqual(stats.totalOperations, 5)
        XCTAssertNotNil(stats.avgGPUTimes[GPUOperation.l2Distance.rawValue])
        XCTAssertNotNil(stats.avgCPUTimes[GPUOperation.l2Distance.rawValue])
    }

    func test_reset_clearsHistory() async throws {
        // Record some performance data
        await engine.recordPerformance(
            operation: .l2Distance,
            cpuTime: 0.01,
            gpuTime: 0.005
        )
        await engine.recordPerformance(
            operation: .cosineSimilarity,
            cpuTime: 0.02,
            gpuTime: 0.01
        )

        // Verify data exists
        let statsBefore = await engine.getPerformanceStats()
        XCTAssertGreaterThan(statsBefore.totalOperations, 0)

        // Reset and verify cleared
        await engine.reset()

        let statsAfter = await engine.getPerformanceStats()
        XCTAssertEqual(statsAfter.totalOperations, 0)
        XCTAssertTrue(statsAfter.avgGPUTimes.isEmpty)
        XCTAssertTrue(statsAfter.avgCPUTimes.isEmpty)
    }

    // MARK: - Custom Thresholds Tests

    func test_customThresholds() async throws {
        let customEngine = GPUDecisionEngine(thresholds: .batchOptimized)

        let thresholds = await customEngine.getThresholds()
        XCTAssertEqual(thresholds.minVectorsForGPU, 500)
        XCTAssertEqual(thresholds.minCandidatesForGPU, 256)
        XCTAssertEqual(thresholds.minKForGPU, 5)
        XCTAssertEqual(thresholds.maxKForGPU, 2000)
        XCTAssertEqual(thresholds.minOperationsForGPU, 25_000)
        XCTAssertEqual(thresholds.maxGPUMemoryMB, 2048)
        XCTAssertEqual(thresholds.minBatchSizeForGPU, 2)
    }

    // MARK: - Device Information Tests

    func test_isGPUAvailable() async throws {
        let metalAvailable = MTLCreateSystemDefaultDevice() != nil
        let engineAvailable = await engine.isGPUAvailable
        XCTAssertEqual(engineAvailable, metalAvailable)
    }
}
