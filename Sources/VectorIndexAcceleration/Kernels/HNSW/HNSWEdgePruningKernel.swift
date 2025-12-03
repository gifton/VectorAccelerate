//
//  HNSWEdgePruningKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW parallel edge pruning.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Pruning Flags

/// Flags indicating which edges to prune.
public struct PruningFlags: @unchecked Sendable {
    /// Flags buffer (0 = keep, 1 = prune)
    public let flags: any MTLBuffer

    /// Total number of edges
    public let totalEdges: Int

    public init(flags: any MTLBuffer, totalEdges: Int) {
        self.flags = flags
        self.totalEdges = totalEdges
    }
}

// MARK: - Pruning Result

/// Result from edge pruning operation.
public struct EdgePruningResult: @unchecked Sendable {
    /// Compacted edges buffer
    public let compactedEdges: any MTLBuffer

    /// Compacted distances buffer (optional)
    public let compactedDistances: (any MTLBuffer)?

    /// New offsets buffer
    public let newOffsets: any MTLBuffer

    /// Number of edges removed
    public let edgesRemoved: Int

    /// Number of edges retained
    public let edgesRetained: Int

    /// Compression ratio
    public let compressionRatio: Double

    /// Execution time
    public let executionTime: TimeInterval

    /// Pruning rate
    public var pruningRate: Double {
        let total = edgesRemoved + edgesRetained
        return total > 0 ? Double(edgesRemoved) / Double(total) : 0
    }
}

// MARK: - HNSW Edge Pruning Kernel

/// Metal 4 kernel for GPU-accelerated HNSW edge pruning.
///
/// Uses a 3-pass parallel algorithm for order-preserving edge pruning:
/// - Pass 1: Count surviving edges per node
/// - Pass 2: Global scan (prefix sum) for new offsets
/// - Pass 3: Local scan and compaction
///
/// ## Features
/// - Order-preserving compaction
/// - Multiple pruning strategies
/// - Blelloch scan for prefix sums
/// - Distance preservation option
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWEdgePruningKernel(context: context)
/// let flags = try await kernel.generatePruningFlags(graph: graph, distances: distances, targetDegree: 32)
/// let result = try await kernel.pruneEdges(graph: graph, pruningFlags: flags)
/// ```
public final class HNSWEdgePruningKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWEdgePruningKernel"

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: HNSWPruningConfiguration

    private let pass1Pipeline: any MTLComputePipelineState
    private let pass3Pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: HNSWPruningConfiguration = HNSWPruningConfiguration()
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let pass1Func = library.makeFunction(name: "hnsw_k42_pass1_count_surviving") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_k42_pass1_count_surviving. Ensure HNSW shaders are compiled."
            )
        }

        guard let pass3Func = library.makeFunction(name: "hnsw_k42_pass3_scan_and_compact") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_k42_pass3_scan_and_compact. Ensure HNSW shaders are compiled."
            )
        }

        let device = context.device.rawDevice
        self.pass1Pipeline = try await device.makeComputePipelineState(function: pass1Func)
        self.pass3Pipeline = try await device.makeComputePipelineState(function: pass3Func)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines are created in init
    }

    // MARK: - Edge Pruning Pipeline

    /// Prune edges from HNSW graph using 3-pass pipeline.
    ///
    /// - Parameters:
    ///   - graph: The graph to prune
    ///   - pruningFlags: Flags indicating which edges to prune
    ///   - distances: Optional distance buffer
    /// - Returns: Pruning result with compacted graph data
    public func pruneEdges(
        graph: HNSWMutableGraph,
        pruningFlags: PruningFlags,
        distances: (any MTLBuffer)? = nil
    ) async throws -> EdgePruningResult {
        let startTime = CACurrentMediaTime()

        // Validate constraints
        try validateConstraints(graph: graph)

        let device = context.device.rawDevice

        // Pass 1: Count surviving edges per node
        guard let newCounts = device.makeBuffer(
            length: graph.nodeCount * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: graph.nodeCount * MemoryLayout<UInt32>.size)
        }
        newCounts.label = "HNSWPruning.newCounts"

        try await executePass1(
            pruningFlags: pruningFlags.flags,
            oldOffsets: graph.edgeOffsets,
            newCounts: newCounts,
            nodeCount: graph.nodeCount
        )

        // Pass 2: Global scan to compute new offsets (CPU implementation)
        guard let newOffsets = device.makeBuffer(
            length: (graph.nodeCount + 1) * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: (graph.nodeCount + 1) * MemoryLayout<UInt32>.size)
        }
        newOffsets.label = "HNSWPruning.newOffsets"

        executeCPUScan(input: newCounts, output: newOffsets, count: graph.nodeCount)

        // Get total surviving edges
        let totalSurviving = getTotalSurvivingEdges(offsetsBuffer: newOffsets, nodeCount: graph.nodeCount)

        // Allocate output buffers
        guard let compactedEdges = device.makeBuffer(
            length: totalSurviving * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalSurviving * MemoryLayout<UInt32>.size)
        }
        compactedEdges.label = "HNSWPruning.compactedEdges"

        var compactedDistances: (any MTLBuffer)?
        if configuration.preserveDistances && distances != nil {
            compactedDistances = device.makeBuffer(
                length: totalSurviving * MemoryLayout<Float>.size,
                options: .storageModeShared
            )
            compactedDistances?.label = "HNSWPruning.compactedDistances"
        }

        // Pass 3: Local scan and compaction
        try await executePass3(
            oldEdges: graph.edges,
            oldDistances: distances,
            pruningFlags: pruningFlags.flags,
            oldOffsets: graph.edgeOffsets,
            newOffsets: newOffsets,
            compactedEdges: compactedEdges,
            compactedDistances: compactedDistances,
            nodeCount: graph.nodeCount
        )

        let executionTime = CACurrentMediaTime() - startTime

        // Calculate statistics
        let edgesRemoved = pruningFlags.totalEdges - totalSurviving
        let compressionRatio = Double(pruningFlags.totalEdges) / Double(max(1, totalSurviving))

        return EdgePruningResult(
            compactedEdges: compactedEdges,
            compactedDistances: compactedDistances,
            newOffsets: newOffsets,
            edgesRemoved: edgesRemoved,
            edgesRetained: totalSurviving,
            compressionRatio: compressionRatio,
            executionTime: executionTime
        )
    }

    // MARK: - Pruning Flag Generation

    /// Generate pruning flags based on strategy.
    ///
    /// - Parameters:
    ///   - graph: The graph to prune
    ///   - distances: Optional distance buffer
    ///   - targetDegree: Target maximum degree per node
    /// - Returns: Pruning flags
    public func generatePruningFlags(
        graph: HNSWMutableGraph,
        distances: (any MTLBuffer)?,
        targetDegree: Int
    ) async throws -> PruningFlags {
        switch configuration.pruningStrategy {
        case .distance:
            return try await generateDistanceBasedFlags(
                graph: graph,
                distances: distances,
                targetDegree: targetDegree
            )

        case .heuristic:
            return try await generateHeuristicFlags(
                graph: graph,
                distances: distances,
                targetDegree: targetDegree
            )

        case .random:
            return try await generateRandomFlags(
                graph: graph,
                targetDegree: targetDegree
            )

        case .custom:
            throw IndexAccelerationError.invalidInput(message: "Custom pruning requires user-provided flags")
        }
    }

    // MARK: - Private Pass Execution

    private func executePass1(
        pruningFlags: any MTLBuffer,
        oldOffsets: any MTLBuffer,
        newCounts: any MTLBuffer,
        nodeCount: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pass1Pipeline)
            encoder.label = "HNSW Pruning Pass 1: Count Surviving"

            encoder.setBuffer(pruningFlags, offset: 0, index: 3)
            encoder.setBuffer(oldOffsets, offset: 0, index: 8)
            encoder.setBuffer(newCounts, offset: 0, index: 2)

            var n = UInt32(nodeCount)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)

            let threadsPerGroup = configuration.threadgroupSize
            let threadgroups = (nodeCount + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }
    }

    private func executePass3(
        oldEdges: any MTLBuffer,
        oldDistances: (any MTLBuffer)?,
        pruningFlags: any MTLBuffer,
        oldOffsets: any MTLBuffer,
        newOffsets: any MTLBuffer,
        compactedEdges: any MTLBuffer,
        compactedDistances: (any MTLBuffer)?,
        nodeCount: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pass3Pipeline)
            encoder.label = "HNSW Pruning Pass 3: Scan and Compact"

            encoder.setBuffer(oldDistances, offset: 0, index: 0)
            encoder.setBuffer(oldEdges, offset: 0, index: 1)
            encoder.setBuffer(pruningFlags, offset: 0, index: 3)
            encoder.setBuffer(oldOffsets, offset: 0, index: 8)
            encoder.setBuffer(newOffsets, offset: 0, index: 5)
            encoder.setBuffer(compactedEdges, offset: 0, index: 4)
            encoder.setBuffer(compactedDistances, offset: 0, index: 9)

            var n = UInt32(nodeCount)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)

            // TGS must be 256 for Blelloch scan
            let threadsPerGroup = 256
            let threadgroups = (nodeCount + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }
    }

    // MARK: - Private Helpers

    private func validateConstraints(graph: HNSWMutableGraph) throws {
        let offsetsPtr = graph.edgeOffsets.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.nodeCount + 1
        )

        for i in 0..<graph.nodeCount {
            let degree = Int(offsetsPtr[i + 1] - offsetsPtr[i])
            if degree > configuration.maxDegree {
                throw IndexAccelerationError.invalidInput(
                    message: "Node \(i) has degree \(degree) exceeding max \(configuration.maxDegree)"
                )
            }
        }
    }

    private func getTotalSurvivingEdges(
        offsetsBuffer: any MTLBuffer,
        nodeCount: Int
    ) -> Int {
        let pointer = offsetsBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: nodeCount + 1
        )
        return Int(pointer[nodeCount])
    }

    private func executeCPUScan(
        input: any MTLBuffer,
        output: any MTLBuffer,
        count: Int
    ) {
        let inputPtr = input.contents().bindMemory(to: UInt32.self, capacity: count)
        let outputPtr = output.contents().bindMemory(to: UInt32.self, capacity: count + 1)

        // Exclusive scan
        outputPtr[0] = 0
        for i in 0..<count {
            outputPtr[i + 1] = outputPtr[i] + inputPtr[i]
        }
    }

    // MARK: - Flag Generation Methods

    private func generateDistanceBasedFlags(
        graph: HNSWMutableGraph,
        distances: (any MTLBuffer)?,
        targetDegree: Int
    ) async throws -> PruningFlags {
        guard let distances = distances else {
            throw IndexAccelerationError.invalidInput(message: "Distance-based pruning requires distance buffer")
        }

        let offsetsPtr = graph.edgeOffsets.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.nodeCount + 1
        )
        let distPtr = distances.contents().bindMemory(
            to: Float.self,
            capacity: graph.totalCapacity
        )

        var flags: [UInt32] = Array(repeating: 0, count: graph.totalCapacity)

        for nodeId in 0..<graph.nodeCount {
            let start = Int(offsetsPtr[nodeId])
            let end = Int(offsetsPtr[nodeId + 1])
            let degree = end - start

            if degree > targetDegree {
                var edges: [(index: Int, distance: Float)] = []
                for i in start..<end {
                    edges.append((i, distPtr[i]))
                }
                edges.sort { $0.distance < $1.distance }

                for i in targetDegree..<edges.count {
                    flags[edges[i].index] = 1
                }
            }
        }

        let device = context.device.rawDevice
        guard let flagBuffer = device.makeBuffer(
            bytes: flags,
            length: flags.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flags.count * MemoryLayout<UInt32>.size)
        }
        flagBuffer.label = "HNSWPruning.flags"

        return PruningFlags(flags: flagBuffer, totalEdges: graph.totalCapacity)
    }

    private func generateHeuristicFlags(
        graph: HNSWMutableGraph,
        distances: (any MTLBuffer)?,
        targetDegree: Int
    ) async throws -> PruningFlags {
        // For now, delegate to distance-based
        return try await generateDistanceBasedFlags(
            graph: graph,
            distances: distances,
            targetDegree: targetDegree
        )
    }

    private func generateRandomFlags(
        graph: HNSWMutableGraph,
        targetDegree: Int
    ) async throws -> PruningFlags {
        let offsetsPtr = graph.edgeOffsets.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.nodeCount + 1
        )

        var flags: [UInt32] = Array(repeating: 0, count: graph.totalCapacity)

        for nodeId in 0..<graph.nodeCount {
            let start = Int(offsetsPtr[nodeId])
            let end = Int(offsetsPtr[nodeId + 1])
            let degree = end - start

            if degree > targetDegree {
                let toPrune = degree - targetDegree
                let indices = Array(start..<end).shuffled()
                for i in 0..<toPrune {
                    flags[indices[i]] = 1
                }
            }
        }

        let device = context.device.rawDevice
        guard let flagBuffer = device.makeBuffer(
            bytes: flags,
            length: flags.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flags.count * MemoryLayout<UInt32>.size)
        }
        flagBuffer.label = "HNSWPruning.flags"

        return PruningFlags(flags: flagBuffer, totalEdges: graph.totalCapacity)
    }
}
