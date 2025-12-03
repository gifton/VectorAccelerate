//
//  HNSWEdgeInsertionKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW batch edge insertion.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Edge Batch

/// Batch of edges to insert.
public struct EdgeBatch: Sendable {
    /// Source node IDs [B]
    public let sourceNodes: [UInt32]

    /// Target node IDs per source [B][M]
    public let targetNodes: [[UInt32]]

    /// Batch size
    public let batchSize: Int

    /// Maximum edges per node
    public let maxEdgesPerNode: Int

    public init(sourceNodes: [UInt32], targetNodes: [[UInt32]]) {
        self.sourceNodes = sourceNodes
        self.targetNodes = targetNodes
        self.batchSize = sourceNodes.count
        self.maxEdgesPerNode = targetNodes.map { $0.count }.max() ?? 0
    }

    /// Create from edge list format.
    public static func fromEdgeList(_ edges: [(source: UInt32, target: UInt32)]) -> EdgeBatch {
        var grouped: [UInt32: [UInt32]] = [:]
        for (source, target) in edges {
            grouped[source, default: []].append(target)
        }

        let sourceNodes = Array(grouped.keys).sorted()
        let targetNodes = sourceNodes.map { grouped[$0] ?? [] }

        return EdgeBatch(sourceNodes: sourceNodes, targetNodes: targetNodes)
    }
}

// MARK: - Insertion Result

/// Result from batch edge insertion.
public struct EdgeInsertionResult: Sendable {
    /// Number of edges inserted
    public let edgesInserted: Int

    /// Number of edges skipped
    public let edgesSkipped: Int

    /// Nodes requiring reallocation
    public let nodesRequiringReallocation: [UInt32]

    /// Execution time
    public let executionTime: TimeInterval

    /// Check if reallocation is needed
    public var needsReallocation: Bool {
        !nodesRequiringReallocation.isEmpty
    }

    /// Success rate
    public var successRate: Double {
        let total = edgesInserted + edgesSkipped
        return total > 0 ? Double(edgesInserted) / Double(total) : 0
    }
}

// MARK: - HNSW Edge Insertion Kernel

/// Metal 4 kernel for GPU-accelerated HNSW edge insertion.
///
/// Performs atomic batch edge insertion into CSR graph structure with
/// automatic capacity overflow detection.
///
/// ## Features
/// - Atomic edge counting per node
/// - Capacity overflow detection
/// - Bidirectional edge support
/// - Graph structure reallocation
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWEdgeInsertionKernel(context: context)
/// let graph = try await kernel.createGraph(nodeCount: 10000)
/// let result = try await kernel.insertEdges(batch: edges, graph: graph)
/// ```
public final class HNSWEdgeInsertionKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWEdgeInsertionKernel"

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: HNSWEdgeInsertionConfiguration

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: HNSWEdgeInsertionConfiguration = HNSWEdgeInsertionConfiguration()
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library and create pipeline
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "hnsw_batch_insert_edges") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_batch_insert_edges. Ensure HNSW shaders are compiled."
            )
        }

        self.pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline is created in init
    }

    // MARK: - Graph Creation

    /// Create a new mutable graph structure.
    ///
    /// - Parameter nodeCount: Number of nodes
    /// - Returns: The created graph
    public func createGraph(nodeCount: Int) async throws -> HNSWMutableGraph {
        let capacityPerNode = configuration.initialCapacity
        let totalCapacity = nodeCount * capacityPerNode

        let device = context.device.rawDevice

        // Allocate edge storage (CSR format)
        guard let edges = device.makeBuffer(
            length: totalCapacity * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalCapacity * MemoryLayout<UInt32>.size)
        }
        edges.label = "HNSWGraph.edges"

        // Initialize with sentinel values
        let edgesPtr = edges.contents().bindMemory(to: UInt32.self, capacity: totalCapacity)
        for i in 0..<totalCapacity {
            edgesPtr[i] = 0xFFFFFFFF
        }

        // Allocate atomic edge counts (zero-initialized)
        guard let edgeCounts = device.makeBuffer(
            length: nodeCount * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: nodeCount * MemoryLayout<UInt32>.size)
        }
        edgeCounts.label = "HNSWGraph.counts"
        edgeCounts.contents().initializeMemory(as: UInt32.self, repeating: 0, count: nodeCount)

        // Allocate capacity array
        let capacities = Array(repeating: UInt32(capacityPerNode), count: nodeCount)
        guard let edgeCapacity = device.makeBuffer(
            bytes: capacities,
            length: capacities.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: capacities.count * MemoryLayout<UInt32>.size)
        }
        edgeCapacity.label = "HNSWGraph.capacity"

        // Calculate offsets (uniform stride initially)
        var offsets: [UInt32] = []
        for i in 0...nodeCount {
            offsets.append(UInt32(i * capacityPerNode))
        }
        guard let edgeOffsets = device.makeBuffer(
            bytes: offsets,
            length: offsets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: offsets.count * MemoryLayout<UInt32>.size)
        }
        edgeOffsets.label = "HNSWGraph.offsets"

        return HNSWMutableGraph(
            edges: edges,
            edgeCounts: edgeCounts,
            edgeCapacity: edgeCapacity,
            edgeOffsets: edgeOffsets,
            nodeCount: nodeCount,
            totalCapacity: totalCapacity
        )
    }

    // MARK: - Edge Insertion

    /// Insert a batch of edges into the graph.
    ///
    /// - Parameters:
    ///   - batch: The edges to insert
    ///   - graph: The graph to insert into
    /// - Returns: Insertion result
    public func insertEdges(
        batch: EdgeBatch,
        graph: HNSWMutableGraph
    ) async throws -> EdgeInsertionResult {
        let startTime = CACurrentMediaTime()

        guard batch.batchSize > 0 else {
            return EdgeInsertionResult(
                edgesInserted: 0,
                edgesSkipped: 0,
                nodesRequiringReallocation: [],
                executionTime: 0
            )
        }

        let device = context.device.rawDevice

        // Create source buffer
        guard let sourceBuffer = device.makeBuffer(
            bytes: batch.sourceNodes,
            length: batch.sourceNodes.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: batch.sourceNodes.count * MemoryLayout<UInt32>.size)
        }
        sourceBuffer.label = "HNSWEdgeInsertion.sources"

        // Create target buffer (flattened with padding)
        let sentinel: UInt32 = 0xFFFFFFFF
        var flatTargets: [UInt32] = []
        for nodeTargets in batch.targetNodes {
            flatTargets.append(contentsOf: nodeTargets)
            let padding = batch.maxEdgesPerNode - nodeTargets.count
            if padding > 0 {
                flatTargets.append(contentsOf: Array(repeating: sentinel, count: padding))
            }
        }

        guard let targetBuffer = device.makeBuffer(
            bytes: flatTargets,
            length: flatTargets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatTargets.count * MemoryLayout<UInt32>.size)
        }
        targetBuffer.label = "HNSWEdgeInsertion.targets"

        // Allocate reallocation flags (zero-initialized)
        guard let reallocFlags = device.makeBuffer(
            length: graph.nodeCount * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: graph.nodeCount * MemoryLayout<UInt32>.size)
        }
        reallocFlags.label = "HNSWEdgeInsertion.reallocFlags"
        reallocFlags.contents().initializeMemory(as: UInt32.self, repeating: 0, count: graph.nodeCount)

        // Execute kernel
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "HNSW Batch Edge Insertion"

            // Set buffers
            encoder.setBuffer(graph.edges, offset: 0, index: 0)
            encoder.setBuffer(graph.edgeCounts, offset: 0, index: 1)
            encoder.setBuffer(graph.edgeCapacity, offset: 0, index: 2)
            encoder.setBuffer(targetBuffer, offset: 0, index: 3)
            encoder.setBuffer(sourceBuffer, offset: 0, index: 4)
            encoder.setBuffer(graph.edgeOffsets, offset: 0, index: 5)
            encoder.setBuffer(reallocFlags, offset: 0, index: 6)

            // Set constants
            var b = UInt32(batch.batchSize)
            var m = UInt32(batch.maxEdgesPerNode)
            var bidirectional = UInt32(configuration.bidirectional ? 1 : 0)

            encoder.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&bidirectional, length: MemoryLayout<UInt32>.size, index: 10)

            // Dispatch 2D grid
            let gridSize = MTLSize(
                width: batch.maxEdgesPerNode,
                height: batch.batchSize,
                depth: 1
            )

            let threadgroupSize = MTLSize(
                width: configuration.threadgroupWidth,
                height: configuration.threadgroupHeight,
                depth: 1
            )

            let threadgroupsPerGrid = MTLSize(
                width: (gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
                height: (gridSize.height + threadgroupSize.height - 1) / threadgroupSize.height,
                depth: 1
            )

            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
        }

        // Extract results
        let result = extractResults(
            reallocFlags: reallocFlags,
            graph: graph,
            batch: batch
        )

        let executionTime = CACurrentMediaTime() - startTime

        return EdgeInsertionResult(
            edgesInserted: result.edgesInserted,
            edgesSkipped: result.edgesSkipped,
            nodesRequiringReallocation: result.nodesRequiringReallocation,
            executionTime: executionTime
        )
    }

    // MARK: - Graph Reallocation

    /// Reallocate graph capacity for nodes that exceeded limits.
    ///
    /// - Parameters:
    ///   - graph: The graph to reallocate
    ///   - nodesNeedingReallocation: Nodes that need more capacity
    /// - Returns: New graph with increased capacity
    public func reallocateGraph(
        graph: HNSWMutableGraph,
        nodesNeedingReallocation: [UInt32]
    ) async throws -> HNSWMutableGraph {
        let device = context.device.rawDevice

        // Calculate new capacities
        let capacityPtr = graph.edgeCapacity.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.nodeCount
        )
        let countsPtr = graph.edgeCounts.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.nodeCount
        )

        var newCapacities = Array(UnsafeBufferPointer(start: capacityPtr, count: graph.nodeCount))

        for nodeId in nodesNeedingReallocation {
            let currentCount = countsPtr[Int(nodeId)]
            let newCapacity = UInt32(Float(currentCount) * configuration.capacityGrowthFactor)
            newCapacities[Int(nodeId)] = newCapacity
        }

        // Calculate new offsets
        var newOffsets: [UInt32] = [0]
        for capacity in newCapacities {
            newOffsets.append(newOffsets.last! + capacity)
        }
        let totalNewCapacity = Int(newOffsets.last!)

        // Allocate new buffers
        guard let newEdges = device.makeBuffer(
            length: totalNewCapacity * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalNewCapacity * MemoryLayout<UInt32>.size)
        }
        newEdges.label = "HNSWGraph.edges.reallocated"

        guard let newEdgeCapacity = device.makeBuffer(
            bytes: newCapacities,
            length: newCapacities.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: newCapacities.count * MemoryLayout<UInt32>.size)
        }
        newEdgeCapacity.label = "HNSWGraph.capacity.reallocated"

        guard let newEdgeOffsets = device.makeBuffer(
            bytes: newOffsets,
            length: newOffsets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: newOffsets.count * MemoryLayout<UInt32>.size)
        }
        newEdgeOffsets.label = "HNSWGraph.offsets.reallocated"

        // Copy existing edges (CPU operation)
        copyEdges(from: graph, to: newEdges, newOffsets: newOffsets)

        return HNSWMutableGraph(
            edges: newEdges,
            edgeCounts: graph.edgeCounts,
            edgeCapacity: newEdgeCapacity,
            edgeOffsets: newEdgeOffsets,
            nodeCount: graph.nodeCount,
            totalCapacity: totalNewCapacity
        )
    }

    // MARK: - Private Helpers

    private func extractResults(
        reallocFlags: any MTLBuffer,
        graph: HNSWMutableGraph,
        batch: EdgeBatch
    ) -> (edgesInserted: Int, edgesSkipped: Int, nodesRequiringReallocation: [UInt32]) {
        let flagsPtr = reallocFlags.contents().bindMemory(to: UInt32.self, capacity: graph.nodeCount)
        let countsPtr = graph.edgeCounts.contents().bindMemory(to: UInt32.self, capacity: graph.nodeCount)

        var nodesNeedingRealloc: [UInt32] = []

        for i in 0..<graph.nodeCount {
            if flagsPtr[i] != 0 {
                nodesNeedingRealloc.append(UInt32(i))
            }
        }

        // Estimate edges inserted
        let totalEdges = batch.targetNodes.flatMap { $0 }.count
        let multiplier = configuration.bidirectional ? 2 : 1
        let potentialEdges = totalEdges * multiplier

        // Count actual insertions
        var actualInserted = 0
        for sourceId in batch.sourceNodes {
            actualInserted += Int(countsPtr[Int(sourceId)])
        }

        let skipped = potentialEdges - actualInserted

        return (actualInserted, skipped, nodesNeedingRealloc)
    }

    private func copyEdges(
        from oldGraph: HNSWMutableGraph,
        to newEdges: any MTLBuffer,
        newOffsets: [UInt32]
    ) {
        let oldEdgesPtr = oldGraph.edges.contents().bindMemory(
            to: UInt32.self,
            capacity: oldGraph.totalCapacity
        )
        let newEdgesPtr = newEdges.contents().bindMemory(
            to: UInt32.self,
            capacity: Int(newOffsets.last!)
        )
        let countsPtr = oldGraph.edgeCounts.contents().bindMemory(
            to: UInt32.self,
            capacity: oldGraph.nodeCount
        )
        let oldOffsetsPtr = oldGraph.edgeOffsets.contents().bindMemory(
            to: UInt32.self,
            capacity: oldGraph.nodeCount + 1
        )

        // Initialize new buffer with sentinels
        for i in 0..<Int(newOffsets.last!) {
            newEdgesPtr[i] = 0xFFFFFFFF
        }

        // Copy existing edges
        for nodeId in 0..<oldGraph.nodeCount {
            let edgeCount = Int(countsPtr[nodeId])
            let oldOffset = Int(oldOffsetsPtr[nodeId])
            let newOffset = Int(newOffsets[nodeId])

            for i in 0..<edgeCount {
                newEdgesPtr[newOffset + i] = oldEdgesPtr[oldOffset + i]
            }
        }
    }
}
