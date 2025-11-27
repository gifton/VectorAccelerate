//
//  SupportedDistanceMetrics.swift
//  VectorAccelerate
//
//  Additional distance metric implementations for ComputeEngine
//

import Foundation
import VectorCore
@preconcurrency import Metal

public extension ComputeEngine {
    
    // MARK: - Manhattan Distance
    
    /// Compute Manhattan (L1) distance between two vectors
    func manhattanDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "manhattanDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Ensure at least 32 threads for proper reduction, but no more than 256
            let threadsPerGroup = MTLSize(width: min(256, max(32, dimension)), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    // MARK: - Chebyshev Distance
    
    /// Compute Chebyshev (L∞) distance between two vectors
    func chebyshevDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "chebyshevDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Ensure at least 32 threads for proper reduction, but no more than 256
            let threadsPerGroup = MTLSize(width: min(256, max(32, dimension)), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    // MARK: - Minkowski Distance
    
    /// Compute Minkowski distance between two vectors (generalized Lp norm)
    func minkowskiDistance(
        _ vectorA: [Float],
        _ vectorB: [Float],
        p: Float = 2.0
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        guard p > 0 else {
            throw VectorError.invalidInput("Minkowski p parameter must be positive")
        }
        
        // For very large p values, approximate with Chebyshev distance (L∞)
        if p > 50 {
            return try await chebyshevDistance(vectorA, vectorB)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "minkowskiDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            var pValue = p
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&pValue, length: MemoryLayout<Float>.size, index: 4)
            
            // Ensure at least 32 threads for proper reduction, but no more than 256
            let threadsPerGroup = MTLSize(width: min(256, max(32, dimension)), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    // MARK: - Hamming Distance
    
    /// Compute Hamming distance between two binary vectors
    /// Note: Vectors are treated as binary (0.0 = false, non-zero = true)
    func hammingDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "hammingDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Ensure at least 32 threads for proper reduction, but no more than 256
            let threadsPerGroup = MTLSize(width: min(256, max(32, dimension)), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    // MARK: - Jaccard Distance
    
    /// Compute Jaccard distance between two vectors (1 - Jaccard similarity)
    /// Vectors are treated as sets (non-zero elements are present)
    func jaccardDistance(
        _ vectorA: [Float],
        _ vectorB: [Float]
    ) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.dimensionMismatch(expected: vectorA.count, actual: vectorB.count)
        }
        
        let dimension = vectorA.count
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: vectorA)
        let bufferB = try await context.getBuffer(for: vectorB)
        let resultBuffer = try await context.getBuffer(size: MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "jaccardDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            // Ensure at least 32 threads for proper reduction, but no more than 256
            let threadsPerGroup = MTLSize(width: min(256, max(32, dimension)), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        // Read result
        let result = resultBuffer.copyData(as: Float.self)
        return result[0]
    }
    
    // MARK: - Batch Distance Operations
    
    /// Compute Manhattan distances from query to multiple candidates
    func batchManhattanDistance(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard candidates.allSatisfy({ $0.count == query.count }) else {
            throw VectorError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }
        
        let dimension = query.count
        let candidateCount = candidates.count
        
        // Flatten candidates into single buffer
        var flatCandidates: [Float] = []
        flatCandidates.reserveCapacity(candidateCount * dimension)
        for candidate in candidates {
            flatCandidates.append(contentsOf: candidate)
        }
        
        // Get buffers
        let queryBuffer = try await context.getBuffer(for: query)
        let candidatesBuffer = try await context.getBuffer(for: flatCandidates)
        let distancesBuffer = try await context.getBuffer(size: candidateCount * MemoryLayout<Float>.size)
        
        // Get shader
        let pipelineState = try await shaderManager.getPipelineState(functionName: "batchManhattanDistance")
        
        // Execute computation
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(distancesBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(dimension)
            var count = UInt32(candidateCount)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            
            // 2D thread configuration for batch processing
            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (candidateCount + 15) / 16,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        // Read results
        return distancesBuffer.copyData(as: Float.self)
    }
}

// MARK: - Distance Metric Extensions

public extension ComputeEngine {
    
    /// Generic distance computation using SupportedDistanceMetric enum
    func distance(
        _ vectorA: [Float],
        _ vectorB: [Float],
        metric: SupportedDistanceMetric
    ) async throws -> Float {
        switch metric {
        case .euclidean:
            return try await euclideanDistance(vectorA, vectorB)
        case .cosine:
            return try await cosineDistance(vectorA, vectorB)
        case .dotProduct:
            return try await dotProduct(vectorA, vectorB)
        case .manhattan:
            return try await manhattanDistance(vectorA, vectorB)
        case .chebyshev:
            return try await chebyshevDistance(vectorA, vectorB)
        }
    }
    
    /// Generic batch distance computation
    func batchDistance(
        query: [Float],
        candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        switch metric {
        case .euclidean:
            return try await batchEuclideanDistance(query: query, candidates: candidates)
        case .manhattan:
            return try await batchManhattanDistance(query: query, candidates: candidates)
        case .cosine, .dotProduct, .chebyshev:
            // Fall back to sequential computation for metrics without batch implementation
            var distances: [Float] = []
            for candidate in candidates {
                let distance = try await self.distance(candidate, query, metric: metric)
                distances.append(distance)
            }
            return distances
        }
    }
}