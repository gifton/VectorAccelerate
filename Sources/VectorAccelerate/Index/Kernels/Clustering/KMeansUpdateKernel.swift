//
//  KMeansUpdateKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for K-Means centroid update phase.
//
//  Computes the mean of vectors assigned to each cluster to update
//  centroid positions. Uses high-performance 2-pass GPU orchestration.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means Update Kernel

/// Metal 4 kernel for K-Means centroid update.
public final class KMeansUpdateKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansUpdateKernel"

    // MARK: - Private Properties

    /// Accumulation pipeline (Pass 1)
    private let accumulatePipelineState: MTLComputePipelineState

    /// Normalization pipeline (Pass 2)
    private let normalizePipelineState: MTLComputePipelineState

    // MARK: - Initialization

    /// Create a K-Means update kernel.
    public init(context: Metal4Context) async throws {
        self.context = context
        
        let library = try await context.shaderCompiler.getDefaultLibrary()
        
        guard let accFunc = library.makeFunction(name: "kmeans_update_accumulate"),
              let normFunc = library.makeFunction(name: "kmeans_update_normalize") else {
            throw VectorError.shaderNotFound(
                name: "kmeans_update_accumulate or kmeans_update_normalize. Ensure ClusteringShaders.metal is compiled."
            )
        }
        
        let device = context.device.rawDevice
        self.accumulatePipelineState = try await device.makeComputePipelineState(function: accFunc)
        self.normalizePipelineState = try await device.makeComputePipelineState(function: normFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Already done in init
    }

    // MARK: - Update

    /// Update centroids based on vector assignments.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - assignments: Assignment buffer [numVectors]
    ///   - currentCentroids: Current centroid buffer (for empty clusters)
    ///   - numVectors: Number of vectors
    ///   - numCentroids: Number of clusters (K)
    ///   - dimension: Vector dimension
    /// - Returns: Tuple containing updated centroids token, counts token, and empty cluster count
    public func update(
        vectors: any MTLBuffer,
        assignments: any MTLBuffer,
        currentCentroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws -> (centroidsToken: BufferToken, countsToken: BufferToken, emptyClusters: Int) {
        
        // 1. Dynamic Allocation via BufferPool
        let sumsSize = numCentroids * dimension * MemoryLayout<Float>.size
        let countsSize = numCentroids * MemoryLayout<UInt32>.size
        
        let clusterSumsToken = try await context.getBuffer(size: sumsSize)
        let clusterCountsToken = try await context.getBuffer(size: countsSize)
        
        // Final output buffer
        let newCentroidsToken = try await context.getBuffer(size: sumsSize)
        
        // 2. Execution via Command Buffer
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw IndexError.bufferError(operation: "update", reason: "Failed to create command buffer")
        }
        
        // Phase A: Clear Intermediaries (Pool memory may be recycled)
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw IndexError.bufferError(operation: "update", reason: "Failed to create blit encoder")
        }
        blitEncoder.fill(buffer: clusterSumsToken.buffer, range: 0..<sumsSize, value: 0)
        blitEncoder.fill(buffer: clusterCountsToken.buffer, range: 0..<countsSize, value: 0)
        blitEncoder.endEncoding()
        
        // Phase B: PASS 1 - Accumulate (Cooperative Gather)
        guard let accEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IndexError.bufferError(operation: "update", reason: "Failed to create compute encoder")
        }
        accEncoder.setComputePipelineState(accumulatePipelineState)
        accEncoder.label = "KMeansUpdate.accumulate"
        
        accEncoder.setBuffer(vectors, offset: 0, index: 0)
        accEncoder.setBuffer(assignments, offset: 0, index: 1)
        accEncoder.setBuffer(clusterSumsToken.buffer, offset: 0, index: 2)
        accEncoder.setBuffer(clusterCountsToken.buffer, offset: 0, index: 3)
        
        var nVecs = UInt32(numVectors)
        var dim = UInt32(dimension)
        accEncoder.setBytes(&nVecs, length: MemoryLayout<UInt32>.size, index: 4)
        accEncoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 5)
        
        // Execution Configuration: Use 4 warps (128 threads) for excellent gathering occupancy
        let wAcc = accumulatePipelineState.threadExecutionWidth
        let accThreadsX = min(wAcc * 4, accumulatePipelineState.maxTotalThreadsPerThreadgroup)
        let accThreadsPerTG = MTLSizeMake(accThreadsX, 1, 1)
        
        // 2D Grid: x = clusters, y = dimension chunks (float4)
        let float4Chunks = (dimension + 3) / 4
        let accThreadgroups = MTLSizeMake(numCentroids, float4Chunks, 1)
        
        accEncoder.dispatchThreadgroups(accThreadgroups, threadsPerThreadgroup: accThreadsPerTG)
        accEncoder.endEncoding()
        
        // Phase C: PASS 2 - Normalize (Mean Calculation)
        guard let normEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IndexError.bufferError(operation: "update", reason: "Failed to create compute encoder")
        }
        normEncoder.setComputePipelineState(normalizePipelineState)
        normEncoder.label = "KMeansUpdate.normalize"
        
        normEncoder.setBuffer(clusterSumsToken.buffer, offset: 0, index: 0)
        normEncoder.setBuffer(clusterCountsToken.buffer, offset: 0, index: 1)
        normEncoder.setBuffer(newCentroidsToken.buffer, offset: 0, index: 2)
        normEncoder.setBuffer(currentCentroids, offset: 0, index: 3)
        normEncoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 4)
        
        let wNorm = normalizePipelineState.threadExecutionWidth
        let normThreadsX = min(wNorm * 4, normalizePipelineState.maxTotalThreadsPerThreadgroup)
        let normThreadsPerTG = MTLSizeMake(normThreadsX, 1, 1)
        let normThreadgroups = MTLSizeMake(numCentroids, 1, 1)
        
        normEncoder.dispatchThreadgroups(normThreadgroups, threadsPerThreadgroup: normThreadsPerTG)
        normEncoder.endEncoding()
        
        // Anchor Token Lifecycles
        clusterSumsToken.keepAlive(until: commandBuffer)
        clusterCountsToken.keepAlive(until: commandBuffer)
        newCentroidsToken.keepAlive(until: commandBuffer)
        
        // Wait for completion
        await commandBuffer.commitAndWait()
        
        // 3. Post-Process (Empty Clusters & Read Counts)
        let countsPtr = clusterCountsToken.buffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        var emptyClusters = 0
        for i in 0..<numCentroids {
            if countsPtr[i] == 0 {
                emptyClusters += 1
                // Empty cluster logic: HandleEmptyClusters will be called by pipeline if needed
            }
        }
        
        return (centroidsToken: newCentroidsToken, countsToken: clusterCountsToken, emptyClusters: emptyClusters)
    }

    // MARK: - Legacy / Helper Methods

    /// Convenience method for Swift arrays.
    public func update(
        vectors: [[Float]],
        assignments: [Int],
        currentCentroids: [[Float]]
    ) async throws -> (centroids: [[Float]], counts: [Int], emptyClusters: Int) {
        guard !vectors.isEmpty else { return ([], [], 0) }
        
        let dimension = vectors[0].count
        let numCentroids = currentCentroids.count
        
        // Create buffers
        let flatVectors = vectors.flatMap { $0 }
        let vectorToken = try await context.getBuffer(for: flatQueries(flatVectors)) // Using dummy name from existing code
        
        // Actual logic: flatten properly
        let vToken = try await context.getBuffer(for: flatVectors)
        let aToken = try await context.getBuffer(for: assignments.map { UInt32($0) })
        let cToken = try await context.getBuffer(for: currentCentroids.flatMap { $0 })
        
        let result = try await update(
            vectors: vToken.buffer,
            assignments: aToken.buffer,
            currentCentroids: cToken.buffer,
            numVectors: vectors.count,
            numCentroids: numCentroids,
            dimension: dimension
        )
        
        // Extract results
        let centroidsBuffer = result.centroidsToken.buffer
        let countsBuffer = result.countsToken.buffer
        
        let centroidsPtr = centroidsBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        
        var newCentroids: [[Float]] = []
        var counts: [Int] = []
        
        for k in 0..<numCentroids {
            var centroid: [Float] = []
            for d in 0..<dimension {
                centroid.append(centroidsPtr[k * dimension + d])
            }
            newCentroids.append(centroid)
            counts.append(Int(countsPtr[k]))
        }
        
        return (newCentroids, counts, result.emptyClusters)
    }
    
    // Internal helper for old API
    private func flatQueries(_ arr: [Float]) -> [Float] { return arr }

    /// Handle empty clusters by re-initializing them.
    /// This is called when a cluster has no vectors assigned to it.
    public func handleEmptyClusters(
        centroids: any MTLBuffer,
        vectors: any MTLBuffer,
        assignments: any MTLBuffer,
        distances: any MTLBuffer,
        counts: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) -> Int {
        // This remains a CPU-side operation for now as it involves finding the vector 
        // with the maximum distance to its current centroid and using it as a new centroid.
        let countsPtr = counts.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: numVectors)
        let vectorsPtr = vectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        let centroidsPtr = centroids.contents().bindMemory(to: Float.self, capacity: numCentroids * dimension)
        
        var emptyIndices: [Int] = []
        for k in 0..<numCentroids {
            if countsPtr[k] == 0 {
                emptyIndices.append(k)
            }
        }
        
        if emptyIndices.isEmpty { return 0 }
        
        // Find vectors with largest distances to replace empty centroids
        // This is a simple version of the K-means++ strategy
        var usedVectorIndices = Set<Int>()
        
        for k in emptyIndices {
            var maxDist: Float = -1.0
            var bestVec = -1
            
            for i in 0..<numVectors {
                if !usedVectorIndices.contains(i) && distPtr[i] > maxDist {
                    maxDist = distPtr[i]
                    bestVec = i
                }
            }
            
            if bestVec != -1 {
                usedVectorIndices.insert(bestVec)
                // Copy vector to centroid
                let vecOffset = bestVec * dimension
                let centOffset = k * dimension
                for d in 0..<dimension {
                    centroidsPtr[centOffset + d] = vectorsPtr[vecOffset + d]
                }
                // Set count to 1 so it's not considered empty anymore
                countsPtr[k] = 1
            }
        }
        
        return usedVectorIndices.count
    }
}
