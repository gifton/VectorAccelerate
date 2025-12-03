//
//  HNSWVisitedSetKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW visited set management.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Operation Result

/// Result from visited set operations.
public struct VisitedSetOperationResult: Sendable {
    /// Execution time
    public let executionTime: TimeInterval

    /// Number of nodes processed
    public let nodesProcessed: Int

    /// Throughput in GB/s
    public let throughputGBps: Double?

    public init(
        executionTime: TimeInterval,
        nodesProcessed: Int,
        bytesProcessed: Int? = nil
    ) {
        self.executionTime = executionTime
        self.nodesProcessed = nodesProcessed

        if let bytes = bytesProcessed, executionTime > 0 {
            self.throughputGBps = Double(bytes) / (executionTime * 1_000_000_000)
        } else {
            self.throughputGBps = nil
        }
    }
}

// MARK: - HNSW Visited Set Kernel

/// Metal 4 kernel for GPU-accelerated visited set management.
///
/// Provides efficient clear and merge operations for bit-packed visited sets
/// used during HNSW graph traversal.
///
/// ## Features
/// - Vectorized operations using uint4
/// - Fast parallel clear
/// - Bitwise OR merge
/// - Throughput tracking
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWVisitedSetKernel(context: context)
/// let visitedSet = try await kernel.createVisitedSet(nodeCount: 10000)
/// try await kernel.clearVisitedSet(visitedSet)
/// ```
public final class HNSWVisitedSetKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWVisitedSetKernel"

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: VisitedSetConfiguration

    private let clearPipeline: any MTLComputePipelineState
    private let mergePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: VisitedSetConfiguration = VisitedSetConfiguration()
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let clearFunc = library.makeFunction(name: "hnsw_clear_visited_flags") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_clear_visited_flags. Ensure HNSW shaders are compiled."
            )
        }

        guard let mergeFunc = library.makeFunction(name: "hnsw_merge_visited_sets") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_merge_visited_sets. Ensure HNSW shaders are compiled."
            )
        }

        let device = context.device.rawDevice
        self.clearPipeline = try await device.makeComputePipelineState(function: clearFunc)
        self.mergePipeline = try await device.makeComputePipelineState(function: mergeFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines are created in init
    }

    // MARK: - Visited Set Creation

    /// Create a new visited set.
    ///
    /// - Parameter nodeCount: Number of nodes to track
    /// - Returns: The created visited set
    public func createVisitedSet(nodeCount: Int) async throws -> VisitedSet {
        let wordCount = (nodeCount + 31) / 32
        let byteSize = wordCount * MemoryLayout<UInt32>.size

        // Allocate buffer (16-byte aligned for vectorization)
        let alignedSize = ((byteSize + 15) / 16) * 16

        guard let buffer = context.device.rawDevice.makeBuffer(
            length: alignedSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: alignedSize)
        }
        buffer.label = "HNSWVisitedSet.\(nodeCount)"

        // Zero-initialize
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: alignedSize)

        return VisitedSet(buffer: buffer, nodeCount: nodeCount)
    }

    // MARK: - Clear Operation

    /// Clear all visited flags to zero.
    ///
    /// - Parameter visitedSet: The visited set to clear
    /// - Returns: Operation result with timing info
    public func clearVisitedSet(_ visitedSet: VisitedSet) async throws -> VisitedSetOperationResult {
        let startTime = CACurrentMediaTime()

        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(clearPipeline)
            encoder.label = "Clear Visited Set"

            // Set buffers
            encoder.setBuffer(visitedSet.buffer, offset: 0, index: 0)

            // Set constants
            var numNodes = UInt32(visitedSet.nodeCount)
            var numWords = UInt32(visitedSet.wordCount)

            encoder.setBytes(&numNodes, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&numWords, length: MemoryLayout<UInt32>.size, index: 2)

            // Dispatch (process uint4 vectors)
            let numVec4 = (visitedSet.wordCount + 3) / 4
            let threadsPerGroup = configuration.threadgroupSize
            let threadgroups = (numVec4 + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }

        let executionTime = CACurrentMediaTime() - startTime

        return VisitedSetOperationResult(
            executionTime: executionTime,
            nodesProcessed: visitedSet.nodeCount,
            bytesProcessed: visitedSet.sizeInBytes
        )
    }

    // MARK: - Merge Operation

    /// Merge two visited sets using bitwise OR.
    ///
    /// - Parameters:
    ///   - setA: First visited set
    ///   - setB: Second visited set
    ///   - destination: Optional destination set (created if nil)
    /// - Returns: Merged set and operation metrics
    public func mergeVisitedSets(
        setA: VisitedSet,
        setB: VisitedSet,
        into destination: VisitedSet? = nil
    ) async throws -> (result: VisitedSet, metrics: VisitedSetOperationResult) {
        let startTime = CACurrentMediaTime()

        // Validate compatible sizes
        guard setA.nodeCount == setB.nodeCount else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: setA.nodeCount,
                got: setB.nodeCount
            )
        }

        // Create or validate destination
        let mergedSet: VisitedSet
        if let dest = destination {
            guard dest.nodeCount == setA.nodeCount else {
                throw IndexAccelerationError.dimensionMismatch(
                    expected: setA.nodeCount,
                    got: dest.nodeCount
                )
            }
            mergedSet = dest
        } else {
            mergedSet = try await createVisitedSet(nodeCount: setA.nodeCount)
        }

        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(mergePipeline)
            encoder.label = "Merge Visited Sets"

            // Set buffers
            encoder.setBuffer(setA.buffer, offset: 0, index: 0)
            encoder.setBuffer(setB.buffer, offset: 0, index: 1)
            encoder.setBuffer(mergedSet.buffer, offset: 0, index: 2)

            // Set constants
            var numWords = UInt32(setA.wordCount)
            encoder.setBytes(&numWords, length: MemoryLayout<UInt32>.size, index: 3)

            // Dispatch (process uint4 vectors)
            let numVec4 = (setA.wordCount + 3) / 4
            let threadsPerGroup = configuration.threadgroupSize
            let threadgroups = (numVec4 + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }

        let executionTime = CACurrentMediaTime() - startTime
        let bytesProcessed = setA.sizeInBytes + setB.sizeInBytes + mergedSet.sizeInBytes

        let metrics = VisitedSetOperationResult(
            executionTime: executionTime,
            nodesProcessed: setA.nodeCount,
            bytesProcessed: bytesProcessed
        )

        return (mergedSet, metrics)
    }

    // MARK: - Batch Operations

    /// Clear multiple visited sets in parallel.
    ///
    /// - Parameter visitedSets: Sets to clear
    /// - Returns: Results for each set
    public func batchClear(_ visitedSets: [VisitedSet]) async throws -> [VisitedSetOperationResult] {
        guard !visitedSets.isEmpty else { return [] }

        let startTime = CACurrentMediaTime()

        try await context.executeAndWait { [self] _, encoder in
            for visitedSet in visitedSets {
                encoder.setComputePipelineState(clearPipeline)

                encoder.setBuffer(visitedSet.buffer, offset: 0, index: 0)

                var numNodes = UInt32(visitedSet.nodeCount)
                var numWords = UInt32(visitedSet.wordCount)

                encoder.setBytes(&numNodes, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.setBytes(&numWords, length: MemoryLayout<UInt32>.size, index: 2)

                let numVec4 = (visitedSet.wordCount + 3) / 4
                let threadsPerGroup = configuration.threadgroupSize
                let threadgroups = (numVec4 + threadsPerGroup - 1) / threadsPerGroup

                encoder.dispatchThreadgroups(
                    MTLSize(width: threadgroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
                )
            }
        }

        let totalTime = CACurrentMediaTime() - startTime
        let timePerSet = totalTime / Double(visitedSets.count)

        return visitedSets.map { set in
            VisitedSetOperationResult(
                executionTime: timePerSet,
                nodesProcessed: set.nodeCount,
                bytesProcessed: set.sizeInBytes
            )
        }
    }
}
