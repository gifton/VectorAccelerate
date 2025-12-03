//
//  HNSWLevelAssignmentKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW node level assignment.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Level Assignment Result

/// Result from level assignment operation.
public struct LevelAssignmentResult: Sendable {
    /// Assigned levels per node
    public let levels: [UInt32]

    /// Count of nodes at each level
    public let levelCounts: [UInt32]

    /// Random probabilities used (if tracking enabled)
    public let probabilities: [Float]?

    /// Execution time
    public let executionTime: TimeInterval

    /// Level distribution statistics
    public let levelDistribution: LevelDistribution

    /// Average level
    public var averageLevel: Float {
        levelDistribution.averageLevel
    }

    /// Maximum assigned level
    public var maxAssignedLevel: Int {
        levelDistribution.maxAssignedLevel
    }
}

/// Level distribution statistics.
public struct LevelDistribution: Sendable {
    /// Count per level
    public let counts: [UInt32]

    /// Percentage per level
    public let percentages: [Float]

    /// Average level
    public let averageLevel: Float

    /// Maximum assigned level
    public let maxAssignedLevel: Int

    /// Expected theoretical distribution
    public let expectedDistribution: [Float]

    /// KL divergence from expected
    public var klDivergence: Float {
        var divergence: Float = 0
        for i in 0..<min(percentages.count, expectedDistribution.count) {
            if percentages[i] > 0 && expectedDistribution[i] > 0 {
                divergence += percentages[i] * log(percentages[i] / expectedDistribution[i])
            }
        }
        return divergence
    }
}

/// Batch of nodes for level assignment.
public struct NodeBatch: Sendable {
    /// Optional node IDs for seeding
    public let nodeIDs: [UInt32]?

    /// Number of nodes
    public let count: Int

    public init(count: Int, nodeIDs: [UInt32]? = nil) {
        self.count = count
        self.nodeIDs = nodeIDs
    }

    /// Create sequential batch
    public static func sequential(startID: UInt32, count: Int) -> NodeBatch {
        let nodeIDs = (startID..<(startID + UInt32(count))).map { $0 }
        return NodeBatch(count: count, nodeIDs: nodeIDs)
    }
}

// MARK: - HNSW Level Assignment Kernel

/// Metal 4 kernel for GPU-accelerated HNSW hierarchical level assignment.
///
/// Uses PCG random number generator to assign levels to nodes based on the
/// HNSW probability distribution: P(level=l) = (1/M)^l * (1 - 1/M)
///
/// ## Features
/// - GPU-parallel level assignment
/// - PCG32 random number generation
/// - Per-node seeding for reproducibility
/// - Level distribution tracking
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWLevelAssignmentKernel(context: context, configuration: config)
/// let result = try await kernel.assignLevels(batch: NodeBatch(count: 10000))
/// ```
public final class HNSWLevelAssignmentKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWLevelAssignmentKernel"

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: HNSWLevelConfiguration

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: HNSWLevelConfiguration = HNSWLevelConfiguration()
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library and create pipeline
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "hnsw_assign_node_levels") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_assign_node_levels. Ensure HNSW shaders are compiled."
            )
        }

        self.pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline is created in init
    }

    // MARK: - Level Assignment

    /// Assign levels to a batch of nodes.
    ///
    /// - Parameter batch: The batch of nodes to assign levels to
    /// - Returns: Assignment result with levels and statistics
    public func assignLevels(batch: NodeBatch) async throws -> LevelAssignmentResult {
        let startTime = CACurrentMediaTime()

        guard batch.count > 0 else {
            return LevelAssignmentResult(
                levels: [],
                levelCounts: [],
                probabilities: nil,
                executionTime: 0,
                levelDistribution: LevelDistribution(
                    counts: [],
                    percentages: [],
                    averageLevel: 0,
                    maxAssignedLevel: 0,
                    expectedDistribution: []
                )
            )
        }

        let device = context.device.rawDevice

        // Allocate output buffer for levels
        guard let levelsBuffer = device.makeBuffer(
            length: batch.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: batch.count * MemoryLayout<UInt32>.size)
        }
        levelsBuffer.label = "HNSWLevelAssignment.levels"

        // Create seeds buffer if needed
        let seedsBuffer: (any MTLBuffer)?
        if configuration.useNodeSeeds, let nodeIDs = batch.nodeIDs {
            let buffer = device.makeBuffer(
                bytes: nodeIDs,
                length: nodeIDs.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            )
            buffer?.label = "HNSWLevelAssignment.seeds"
            seedsBuffer = buffer
        } else {
            seedsBuffer = nil
        }

        // Allocate level counts buffer (zero-initialized)
        let countSize = configuration.maxLevel + 1
        guard let levelCountsBuffer = device.makeBuffer(
            length: countSize * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: countSize * MemoryLayout<UInt32>.size)
        }
        levelCountsBuffer.label = "HNSWLevelAssignment.counts"
        levelCountsBuffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: countSize)

        // Allocate probabilities buffer if tracking
        let probabilitiesBuffer: (any MTLBuffer)?
        if configuration.trackProbabilities {
            let buffer = device.makeBuffer(
                length: batch.count * MemoryLayout<Float>.size,
                options: .storageModeShared
            )
            buffer?.label = "HNSWLevelAssignment.probabilities"
            probabilitiesBuffer = buffer
        } else {
            probabilitiesBuffer = nil
        }

        // Execute kernel
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "HNSW Level Assignment"

            // Set buffers
            encoder.setBuffer(levelsBuffer, offset: 0, index: 0)
            encoder.setBuffer(seedsBuffer, offset: 0, index: 1)
            encoder.setBuffer(levelCountsBuffer, offset: 0, index: 2)
            encoder.setBuffer(probabilitiesBuffer, offset: 0, index: 3)

            // Set constants
            var mlFactor = configuration.mlFactor
            var batchCount = UInt32(batch.count)
            var maxLevel = UInt32(configuration.maxLevel)
            var globalSeed = configuration.globalSeed

            encoder.setBytes(&mlFactor, length: MemoryLayout<Float>.size, index: 4)
            encoder.setBytes(&batchCount, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&maxLevel, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&globalSeed, length: MemoryLayout<UInt32>.size, index: 7)

            // Dispatch
            let threadsPerGroup = configuration.threadgroupSize
            let threadgroups = (batch.count + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }

        // Extract results
        let result = extractResults(
            levelsBuffer: levelsBuffer,
            levelCountsBuffer: levelCountsBuffer,
            probabilitiesBuffer: probabilitiesBuffer,
            batchSize: batch.count
        )

        let executionTime = CACurrentMediaTime() - startTime

        return LevelAssignmentResult(
            levels: result.levels,
            levelCounts: result.levelCounts,
            probabilities: result.probabilities,
            executionTime: executionTime,
            levelDistribution: result.levelDistribution
        )
    }

    /// Batch assign levels to multiple node batches.
    public func batchAssignLevels(
        batches: [NodeBatch],
        parallel: Bool = true
    ) async throws -> [LevelAssignmentResult] {
        if parallel {
            return try await withThrowingTaskGroup(of: LevelAssignmentResult.self) { group in
                for batch in batches {
                    group.addTask {
                        try await self.assignLevels(batch: batch)
                    }
                }

                var results: [LevelAssignmentResult] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
        } else {
            var results: [LevelAssignmentResult] = []
            for batch in batches {
                let result = try await assignLevels(batch: batch)
                results.append(result)
            }
            return results
        }
    }

    // MARK: - Private Helpers

    private func extractResults(
        levelsBuffer: any MTLBuffer,
        levelCountsBuffer: any MTLBuffer,
        probabilitiesBuffer: (any MTLBuffer)?,
        batchSize: Int
    ) -> (levels: [UInt32], levelCounts: [UInt32], probabilities: [Float]?, levelDistribution: LevelDistribution) {
        // Extract levels
        let levelsPtr = levelsBuffer.contents().bindMemory(to: UInt32.self, capacity: batchSize)
        let levels = Array(UnsafeBufferPointer(start: levelsPtr, count: batchSize))

        // Extract level counts
        let countSize = configuration.maxLevel + 1
        let countsPtr = levelCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: countSize)
        let levelCounts = Array(UnsafeBufferPointer(start: countsPtr, count: countSize))

        // Extract probabilities if tracked
        let probabilities: [Float]?
        if let probBuffer = probabilitiesBuffer {
            let probPtr = probBuffer.contents().bindMemory(to: Float.self, capacity: batchSize)
            probabilities = Array(UnsafeBufferPointer(start: probPtr, count: batchSize))
        } else {
            probabilities = nil
        }

        // Calculate distribution
        let distribution = calculateDistribution(levelCounts: levelCounts, totalNodes: batchSize)

        return (levels, levelCounts, probabilities, distribution)
    }

    private func calculateDistribution(
        levelCounts: [UInt32],
        totalNodes: Int
    ) -> LevelDistribution {
        let total = Float(totalNodes)
        let percentages = levelCounts.map { Float($0) / total }

        // Calculate average level
        var avgLevel: Float = 0
        for (level, count) in levelCounts.enumerated() {
            avgLevel += Float(level) * Float(count)
        }
        avgLevel /= total

        // Find max assigned level
        var maxAssigned = 0
        for (level, count) in levelCounts.enumerated().reversed() {
            if count > 0 {
                maxAssigned = level
                break
            }
        }

        // Calculate expected distribution
        let expectedDist = calculateExpectedDistribution(maxLevel: configuration.maxLevel)

        return LevelDistribution(
            counts: levelCounts,
            percentages: percentages,
            averageLevel: avgLevel,
            maxAssignedLevel: maxAssigned,
            expectedDistribution: expectedDist
        )
    }

    private func calculateExpectedDistribution(maxLevel: Int) -> [Float] {
        let p = 1.0 / Float(configuration.M)
        var expected: [Float] = []

        for level in 0...maxLevel {
            if level == 0 {
                expected.append(1.0 - p)
            } else {
                let prob = pow(p, Float(level)) * (1.0 - p)
                expected.append(prob)
            }
        }

        return expected
    }
}
