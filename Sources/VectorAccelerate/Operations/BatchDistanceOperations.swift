// VectorAccelerate: Batch Distance Operations
//
// High-performance batch distance computations with GPU acceleration
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Batch distance computation engine
public actor BatchDistanceEngine {
    private let metalContext: MetalContext
    private let shaderManager: ShaderManager
    private let bufferPool: SmartBufferPool
    
    // Performance thresholds
    private let gpuThreshold = 1000  // Use GPU for batches > 1000 vectors
    private let simdThreshold = 100  // Use SIMD for batches > 100 vectors
    
    public init(metalContext: MetalContext) async throws {
        self.metalContext = metalContext
        self.shaderManager = try await ShaderManager(device: metalContext.device)
        self.bufferPool = SmartBufferPool(device: metalContext.device)
    }
    
    // MARK: - Batch Euclidean Distance
    
    /// Compute Euclidean distances between a query and multiple candidates
    public func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard query.count == candidates[0].count else {
            throw AccelerationError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }
        
        let shouldUseGPU = useGPU ?? (candidates.count >= gpuThreshold)
        
        if shouldUseGPU {
            return try await batchEuclideanDistanceGPU(query: query, candidates: candidates)
        } else if candidates.count >= simdThreshold {
            return try await batchEuclideanDistanceSIMD(query: query, candidates: candidates)
        } else {
            return batchEuclideanDistanceCPU(query: query, candidates: candidates)
        }
    }
    
    private func batchEuclideanDistanceGPU(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        let dimension = query.count
        let candidateCount = candidates.count
        
        // Flatten candidates array
        let flatCandidates = candidates.flatMap { $0 }
        
        // Allocate buffers
        let queryBuffer = try await bufferPool.acquire(with: query)
        let candidatesBuffer = try await bufferPool.acquire(with: flatCandidates)
        let resultBuffer = try await bufferPool.acquire(for: Float.self, count: candidateCount)
        
        // Get pipeline
        let pipeline = try await shaderManager.getPipelineState(functionName: "batchEuclideanDistance")
        
        // Create command buffer
        guard let commandBuffer = await metalContext.makeCommandBuffer() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
        
        var dim = UInt32(dimension)
        var count = UInt32(candidateCount)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        
        // Dispatch threads
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (candidateCount + 255) / 256,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let results = resultBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: results, count: candidateCount))
    }
    
    private func batchEuclideanDistanceSIMD(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        await withTaskGroup(of: (Int, Float).self) { group in
            for (index, candidate) in candidates.enumerated() {
                group.addTask {
                    let distance = AccelerateFallback.euclideanDistance(query, candidate)
                    return (index, distance)
                }
            }
            
            var results = [Float](repeating: 0, count: candidates.count)
            for await (index, distance) in group {
                results[index] = distance
            }
            return results
        }
    }
    
    private func batchEuclideanDistanceCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { candidate in
            var sum: Float = 0
            for i in 0..<query.count {
                let diff = query[i] - candidate[i]
                sum += diff * diff
            }
            return sqrt(sum)
        }
    }
    
    // MARK: - Batch Cosine Similarity
    
    /// Compute cosine similarities between a query and multiple candidates
    public func batchCosineSimilarity(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard query.count == candidates[0].count else {
            throw AccelerationError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }
        
        let shouldUseGPU = useGPU ?? (candidates.count >= gpuThreshold)
        
        if shouldUseGPU {
            return try await batchCosineSimilarityGPU(query: query, candidates: candidates)
        } else if candidates.count >= simdThreshold {
            return try await batchCosineSimilaritySIMD(query: query, candidates: candidates)
        } else {
            return batchCosineSimilarityCPU(query: query, candidates: candidates)
        }
    }
    
    private func batchCosineSimilarityGPU(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        let dimension = query.count
        let candidateCount = candidates.count
        
        // Prepare data
        let flatCandidates = candidates.flatMap { $0 }
        
        // Allocate buffers
        let queryBuffer = try await bufferPool.acquire(with: query)
        let candidatesBuffer = try await bufferPool.acquire(with: flatCandidates)
        let resultBuffer = try await bufferPool.acquire(for: Float.self, count: candidateCount)
        
        // Get pipeline
        let pipeline = try await shaderManager.getPipelineState(functionName: "batchCosineSimilarity")
        
        // Execute
        guard let commandBuffer = try await metalContext.makeCommandBuffer() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
        
        var dim = UInt32(dimension)
        var count = UInt32(candidateCount)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (candidateCount + 255) / 256,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let results = resultBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: results, count: candidateCount))
    }
    
    private func batchCosineSimilaritySIMD(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        // Normalize query once
        let queryNorm = sqrt(query.reduce(0) { $0 + $1 * $1 })
        let normalizedQuery = query.map { $0 / queryNorm }
        
        return await withTaskGroup(of: (Int, Float).self) { group in
            for (index, candidate) in candidates.enumerated() {
                group.addTask {
                    let candidateNorm = sqrt(candidate.reduce(0) { $0 + $1 * $1 })
                    let normalizedCandidate = candidate.map { $0 / candidateNorm }
                    
                    let dotProduct = zip(normalizedQuery, normalizedCandidate)
                        .reduce(0) { $0 + $1.0 * $1.1 }
                    
                    return (index, dotProduct)
                }
            }
            
            var results = [Float](repeating: 0, count: candidates.count)
            for await (index, similarity) in group {
                results[index] = similarity
            }
            return results
        }
    }
    
    private func batchCosineSimilarityCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        let queryNorm = sqrt(query.reduce(0) { $0 + $1 * $1 })
        
        return candidates.map { candidate in
            let candidateNorm = sqrt(candidate.reduce(0) { $0 + $1 * $1 })
            let dotProduct = zip(query, candidate).reduce(0) { $0 + $1.0 * $1.1 }
            
            if queryNorm > 0 && candidateNorm > 0 {
                return dotProduct / (queryNorm * candidateNorm)
            } else {
                return 0
            }
        }
    }
    
    // MARK: - Batch Dot Product
    
    /// Compute dot products between a query and multiple candidates
    public func batchDotProduct(
        query: [Float],
        candidates: [[Float]],
        useGPU: Bool? = nil
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        guard query.count == candidates[0].count else {
            throw AccelerationError.dimensionMismatch(expected: query.count, actual: candidates[0].count)
        }
        
        let shouldUseGPU = useGPU ?? (candidates.count >= gpuThreshold)
        
        if shouldUseGPU {
            return try await batchDotProductGPU(query: query, candidates: candidates)
        } else {
            return batchDotProductCPU(query: query, candidates: candidates)
        }
    }
    
    private func batchDotProductGPU(
        query: [Float],
        candidates: [[Float]]
    ) async throws -> [Float] {
        let dimension = query.count
        let candidateCount = candidates.count
        
        // Prepare data
        let flatCandidates = candidates.flatMap { $0 }
        
        // Allocate buffers
        let queryBuffer = try await bufferPool.acquire(with: query)
        let candidatesBuffer = try await bufferPool.acquire(with: flatCandidates)
        let resultBuffer = try await bufferPool.acquire(for: Float.self, count: candidateCount)
        
        // Get pipeline
        let pipeline = try await shaderManager.getPipelineState(functionName: "batchDotProduct")
        
        // Execute
        guard let commandBuffer = try await metalContext.makeCommandBuffer() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
        
        var dim = UInt32(dimension)
        var count = UInt32(candidateCount)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (candidateCount + 255) / 256,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let results = resultBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: results, count: candidateCount))
    }
    
    private func batchDotProductCPU(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { candidate in
            zip(query, candidate).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }
    
    // MARK: - K-Nearest Neighbors
    
    /// Find k-nearest neighbors using specified distance metric
    public func kNearestNeighbors(
        query: [Float],
        candidates: [[Float]],
        k: Int,
        metric: DistanceMetricType = .euclidean
    ) async throws -> [(index: Int, distance: Float)] {
        let distances: [Float]
        
        switch metric {
        case .euclidean:
            distances = try await batchEuclideanDistance(query: query, candidates: candidates)
        case .cosine:
            let similarities = try await batchCosineSimilarity(query: query, candidates: candidates)
            distances = similarities.map { 1.0 - $0 } // Convert similarity to distance
        case .dotProduct:
            let products = try await batchDotProduct(query: query, candidates: candidates)
            distances = products.map { -$0 } // Negate for distance (higher dot product = closer)
        default:
            throw AccelerationError.unsupportedOperation("Metric \(metric) not supported for batch operations")
        }
        
        // Find k smallest distances
        let indexed = distances.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 < $1.1 }
        let topK = Array(sorted.prefix(k))
        
        return topK
    }
    
    // MARK: - Performance Metrics
    
    public struct PerformanceMetrics: Sendable {
        public let averageGPUTime: TimeInterval
        public let averageCPUTime: TimeInterval
        public let gpuSpeedup: Double
    }
    
    private var gpuTimes: [TimeInterval] = []
    private var cpuTimes: [TimeInterval] = []
    
    public func getPerformanceMetrics() -> PerformanceMetrics {
        let avgGPU = gpuTimes.isEmpty ? 0 : gpuTimes.reduce(0, +) / Double(gpuTimes.count)
        let avgCPU = cpuTimes.isEmpty ? 0 : cpuTimes.reduce(0, +) / Double(cpuTimes.count)
        let speedup = avgCPU > 0 ? avgCPU / avgGPU : 1.0
        
        return PerformanceMetrics(
            averageGPUTime: avgGPU,
            averageCPUTime: avgCPU,
            gpuSpeedup: speedup
        )
    }
}

// Note: Distance metric type defined elsewhere in the codebase