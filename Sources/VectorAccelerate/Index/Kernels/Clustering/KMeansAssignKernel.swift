//
//  KMeansAssignKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for K-Means point assignment phase.
//
//  Assigns each vector to its nearest centroid using GPU-accelerated
//  tiled distance computation. Optimized for Apple Silicon SIMD-group architecture.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means Assign Kernel

/// Metal 4 kernel for K-Means point assignment.
public final class KMeansAssignKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansAssignKernel"

    // MARK: - Private Properties

    /// Fallback kernel using FusedL2TopK with k=1
    private let fusedL2TopK: FusedL2TopKKernel

    /// Optimized Tiled Assignment pipeline
    private var tiledAssignmentPipeline: (any MTLComputePipelineState)?

    /// Whether to use the optimized tiled kernels
    private let useOptimizedKernels: Bool

    // MARK: - Initialization

    /// Create a K-Means assignment kernel.
    public init(context: Metal4Context) async throws {
        self.context = context
        self.fusedL2TopK = try await FusedL2TopKKernel(context: context)

        // Try to load optimized tiled assignment kernel
        var loadedOptimized = false
        do {
            let library = try await context.shaderCompiler.getDefaultLibrary()

            if let function = library.makeFunction(name: "kmeans_assign_points") {
                self.tiledAssignmentPipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
                loadedOptimized = true
                VectorLogDebug("Loaded high-performance tiled KMeans kernel", category: "KMeansAssign")
            }
        } catch {
            VectorLogDebug("Failed to load optimized kernel, using fallback: \(error)", category: "KMeansAssign")
        }

        self.useOptimizedKernels = loadedOptimized
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        try await fusedL2TopK.warmUp()
    }

    // MARK: - Assignment

    /// Assign vectors to nearest centroids.
    public func assign(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws -> KMeansAssignmentResult {
        let startTime = CACurrentMediaTime()

        guard numVectors > 0, numCentroids > 0 else {
            throw IndexError.invalidInput(message: "numVectors and numCentroids must be positive")
        }

        // Use optimized tiled kernel when available
        if useOptimizedKernels, let pipeline = tiledAssignmentPipeline {
            return try await assignTiled(
                vectors: vectors,
                centroids: centroids,
                numVectors: numVectors,
                numCentroids: numCentroids,
                dimension: dimension,
                pipeline: pipeline,
                startTime: startTime
            )
        }

        // Fallback: Use fused L2 + Top-K with k=1 to find nearest centroid
        return try await assignFusedL2TopK(
            vectors: vectors,
            centroids: centroids,
            numVectors: numVectors,
            numCentroids: numCentroids,
            dimension: dimension,
            startTime: startTime
        )
    }

    // MARK: - Optimized Tiled Assignment

    /// Assign vectors using the optimized tiled shared memory kernel.
    private func assignTiled(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int,
        pipeline: any MTLComputePipelineState,
        startTime: CFTimeInterval
    ) async throws -> KMeansAssignmentResult {
        // Allocate output buffers
        let assignmentsToken = try await context.getBuffer(size: numVectors * MemoryLayout<UInt32>.size)
        let assignmentsBuffer = assignmentsToken.buffer
        assignmentsBuffer.label = "KMeansAssign.assignments"

        let distancesToken = try await context.getBuffer(size: numVectors * MemoryLayout<Float>.size)
        let distancesBuffer = distancesToken.buffer
        distancesBuffer.label = "KMeansAssign.distances"

        // Create command buffer and encoder
        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IndexError.bufferError(
                operation: "assignTiled",
                reason: "Failed to create command buffer or encoder"
            )
        }

        encoder.setComputePipelineState(pipeline)
        
        // Bind buffers at absolute offset 0 for UMA zero-copy
        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(centroids, offset: 0, index: 1)
        encoder.setBuffer(assignmentsBuffer, offset: 0, index: 2)
        encoder.setBuffer(distancesBuffer, offset: 0, index: 3)

        var nVecs = UInt32(numVectors)
        var nCents = UInt32(numCentroids)
        var dim = UInt32(dimension)
        
        encoder.setBytes(&nVecs, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nCents, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 6)

        // Apple Silicon Memory Safety: Calculate dynamic tile capacity (limit to 32KB shared memory)
        let maxSharedMem = pipeline.device.maxThreadgroupMemoryLength
        let bytesPerCentroid = dimension * MemoryLayout<Float>.stride
        
        // Capacitance logic: Aim for 32 centroids per tile, but mathematically bound by hardware 32KB limit
        let safeCentroidsPerTile = maxSharedMem / bytesPerCentroid
        let tileCapacity = max(1, min(32, safeCentroidsPerTile)) 
        
        var tCap = UInt32(tileCapacity)
        encoder.setBytes(&tCap, length: MemoryLayout<UInt32>.size, index: 7)
        
        let threadgroupMemoryLength = tileCapacity * bytesPerCentroid
        encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)

        // Dynamic Grid Configuration: Align threadgroups with SIMD-group width (32)
        let w = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
        let numThreadgroupsX = (numVectors + w - 1) / w
        let threadgroupsPerGrid = MTLSizeMake(numThreadgroupsX, 1, 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        // Anchor lifetimes to command buffer completion
        assignmentsToken.keepAlive(until: commandBuffer)
        distancesToken.keepAlive(until: commandBuffer)

        await commandBuffer.commitAndWait()

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansAssignmentResult(
            assignmentsToken: assignmentsToken,
            distancesToken: distancesToken,
            numVectors: numVectors,
            executionTime: executionTime
        )
    }

    // MARK: - Fallback FusedL2TopK Assignment

    private func assignFusedL2TopK(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int,
        startTime: CFTimeInterval
    ) async throws -> KMeansAssignmentResult {
        let params = try FusedL2TopKParameters(
            numQueries: numVectors,
            numDataset: numCentroids,
            dimension: dimension,
            k: 1
        )

        let result = try await fusedL2TopK.execute(
            queries: vectors,
            dataset: centroids,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansAssignmentResult(
            assignments: result.indices,
            distances: result.distances!,
            numVectors: numVectors,
            executionTime: executionTime
        )
    }

    // MARK: - High-Level Array API

    public func assign(
        vectors: [[Float]],
        centroids: [[Float]]
    ) async throws -> (assignments: [Int], distances: [Float]) {
        guard !vectors.isEmpty else { return ([], []) }
        guard !centroids.isEmpty else {
            throw IndexError.invalidInput(message: "Centroids cannot be empty")
        }

        let dimension = vectors[0].count
        let flatVectors = vectors.flatMap { $0 }
        let vectorToken = try await context.getBuffer(for: flatVectors)
        
        let flatCentroids = centroids.flatMap { $0 }
        let centroidToken = try await context.getBuffer(for: flatCentroids)

        let result = try await assign(
            vectors: vectorToken.buffer,
            centroids: centroidToken.buffer,
            numVectors: vectors.count,
            numCentroids: centroids.count,
            dimension: dimension
        )

        let assignPtr = result.assignments.contents().bindMemory(to: UInt32.self, capacity: vectors.count)
        let distPtr = result.distances.contents().bindMemory(to: Float.self, capacity: vectors.count)

        let assignments = (0..<vectors.count).map { Int(assignPtr[$0]) }
        let distances = (0..<vectors.count).map { distPtr[$0] }

        return (assignments, distances)
    }

    // MARK: - Cluster Counts

    public func computeClusterCounts(
        assignments: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int
    ) async throws -> BufferToken {
        let countsToken = try await context.getBuffer(size: numCentroids * MemoryLayout<UInt32>.size)
        let countsBuffer = countsToken.buffer
        countsBuffer.label = "KMeansAssign.counts"

        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        for i in 0..<numCentroids { countsPtr[i] = 0 }

        let assignPtr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        for i in 0..<numVectors {
            let cluster = Int(assignPtr[i])
            if cluster < numCentroids {
                countsPtr[cluster] += 1
            }
        }

        return countsToken
    }
}
