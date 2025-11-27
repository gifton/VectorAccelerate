// Warp-Optimized Selection Kernel
// Highly optimized top-k selection for small k values using SIMD shuffle operations

import Metal
import QuartzCore
import Foundation
import VectorCore

// MARK: - Warp-Optimized Selection Kernel

/// GPU-accelerated top-k selection optimized for small k values (k ≤ 32)
/// Uses warp-level primitives and SIMD shuffle operations for maximum performance
public final class WarpOptimizedSelectionKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let warpSmallKAscending: any MTLComputePipelineState
    private let warpSmallKDescending: any MTLComputePipelineState
    private let batchSelectAscending: any MTLComputePipelineState
    private let batchSelectDescending: any MTLComputePipelineState
    
    // Configuration constants
    private let WARP_SIZE = 32 // Metal SIMD group size
    private let MAX_SMALL_K = 32
    private let MAX_BATCH_K = 128
    
    /// Selection mode for top-k
    public enum SelectionMode {
        case ascending  // Select k smallest
        case descending // Select k largest
        
        var kernelSuffix: String {
            switch self {
            case .ascending: return "ascending"
            case .descending: return "descending"
            }
        }
    }
    
    /// Result from warp-optimized selection
    /// - Note: Uses @unchecked Sendable because MTLBuffer is thread-safe for reads
    public struct WarpResult: @unchecked Sendable {
        public let indices: any MTLBuffer
        public let values: (any MTLBuffer)?
        public let queryCount: Int
        public let k: Int
        public let mode: SelectionMode
        
        /// Extract results for a specific query
        public func results(for queryIndex: Int) -> [(index: Int, value: Float)] {
            guard queryIndex < queryCount else { return [] }
            
            let offset = queryIndex * k
            let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: (queryIndex + 1) * k)
            let valuePtr = values?.contents().bindMemory(to: Float.self, capacity: (queryIndex + 1) * k)
            
            var results: [(index: Int, value: Float)] = []
            for i in 0..<k {
                let idx = Int(indexPtr[offset + i])
                if idx != 0xFFFFFFFF {
                    let val = valuePtr?[offset + i] ?? 0
                    results.append((index: idx, value: val))
                }
            }
            return results
        }
    }
    
    /// Batch result for multiple batches of queries
    /// - Note: Uses @unchecked Sendable because MTLBuffer is thread-safe for reads
    public struct BatchResult: @unchecked Sendable {
        public let indices: any MTLBuffer
        public let values: (any MTLBuffer)?
        public let batchSize: Int
        public let queryCount: Int
        public let k: Int
        public let mode: SelectionMode
        
        /// Extract results for a specific batch and query
        public func results(batch: Int, query: Int) -> [(index: Int, value: Float)] {
            guard batch < batchSize && query < queryCount else { return [] }
            
            let offset = (batch * queryCount + query) * k
            let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: batchSize * queryCount * k)
            let valuePtr = values?.contents().bindMemory(to: Float.self, capacity: batchSize * queryCount * k)
            
            var results: [(index: Int, value: Float)] = []
            for i in 0..<k {
                let idx = Int(indexPtr[offset + i])
                if idx != 0xFFFFFFFF {
                    let val = valuePtr?[offset + i] ?? 0
                    results.append((index: idx, value: val))
                }
            }
            return results
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load warp-optimized kernels for small k
        guard let warpAsc = library.makeFunction(name: "warp_select_small_k_ascending"),
              let warpDesc = library.makeFunction(name: "warp_select_small_k_descending") else {
            throw VectorError.shaderNotFound(name: "Warp selection kernels not found")
        }
        
        // Load batch selection kernels for general k
        guard let batchAsc = library.makeFunction(name: "batch_select_k_nearest_ascending"),
              let batchDesc = library.makeFunction(name: "batch_select_k_nearest_descending") else {
            throw VectorError.shaderNotFound(name: "Batch selection kernels not found")
        }
        
        self.warpSmallKAscending = try device.makeComputePipelineState(function: warpAsc)
        self.warpSmallKDescending = try device.makeComputePipelineState(function: warpDesc)
        self.batchSelectAscending = try device.makeComputePipelineState(function: batchAsc)
        self.batchSelectDescending = try device.makeComputePipelineState(function: batchDesc)
    }
    
    // MARK: - Warp-Optimized Selection (k ≤ 32)
    
    /// Perform warp-optimized top-k selection for small k values
    /// - Parameters:
    ///   - distances: Distance/score matrix [Q × N]
    ///   - queryCount: Number of queries
    ///   - candidateCount: Number of candidates per query
    ///   - k: Number of top elements to select (must be ≤ 32)
    ///   - mode: Selection mode (ascending/descending)
    ///   - includeValues: Whether to return distance/score values
    ///   - commandBuffer: Command buffer for execution
    /// - Returns: Selection results
    public func warpSelectSmallK(
        distances: any MTLBuffer,
        queryCount: Int,
        candidateCount: Int,
        k: Int,
        mode: SelectionMode = .ascending,
        includeValues: Bool = true,
        commandBuffer: any MTLCommandBuffer
    ) throws -> WarpResult {
        guard k <= MAX_SMALL_K else {
            throw VectorError.invalidInput("K must be ≤ \(MAX_SMALL_K) for warp optimization")
        }
        
        // Allocate output buffers
        let indicesBuffer = device.makeBuffer(
            length: queryCount * k * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        )
        
        let valuesBuffer = includeValues ? device.makeBuffer(
            length: queryCount * k * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) : nil
        
        guard let indices = indicesBuffer else {
            throw VectorError.bufferCreationFailed("Failed to create indices buffer")
        }
        
        // Select appropriate kernel
        let pipelineState = mode == .ascending ? warpSmallKAscending : warpSmallKDescending
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }
        
        encoder.label = "WarpSelectSmallK_\(mode.kernelSuffix)"
        encoder.setComputePipelineState(pipelineState)
        
        // Set buffers
        encoder.setBuffer(distances, offset: 0, index: 0)
        encoder.setBuffer(indices, offset: 0, index: 1)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 2)

        // Set parameters - individual buffers to match Metal shader signature
        var paramQueryCount = UInt32(queryCount)
        var paramCandidateCount = UInt32(candidateCount)
        var paramK = UInt32(k)
        encoder.setBytes(&paramQueryCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&paramCandidateCount, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&paramK, length: MemoryLayout<UInt32>.size, index: 5)

        // Dispatch with warp size (32 threads per query)
        let threadsPerThreadgroup = MTLSize(width: WARP_SIZE, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: queryCount, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return WarpResult(
            indices: indices,
            values: valuesBuffer,
            queryCount: queryCount,
            k: k,
            mode: mode
        )
    }
    
    // MARK: - Batch Selection (General k)
    
    /// Perform batch selection for multiple queries with general k
    /// - Parameters:
    ///   - distances: Distance/score matrix [B × Q × N]
    ///   - batchSize: Number of batches
    ///   - queryCount: Number of queries per batch
    ///   - candidateCount: Number of candidates per query
    ///   - k: Number of top elements to select
    ///   - mode: Selection mode
    ///   - includeValues: Whether to return values
    ///   - threadgroupSize: Threadgroup size (auto-selected if nil)
    ///   - commandBuffer: Command buffer
    /// - Returns: Batch selection results
    public func batchSelect(
        distances: any MTLBuffer,
        batchSize: Int,
        queryCount: Int,
        candidateCount: Int,
        k: Int,
        mode: SelectionMode = .ascending,
        includeValues: Bool = true,
        threadgroupSize: Int? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws -> BatchResult {
        guard k <= MAX_BATCH_K else {
            throw VectorError.invalidInput("K must be ≤ \(MAX_BATCH_K)")
        }
        
        // Allocate output buffers
        let totalResults = batchSize * queryCount * k
        let indicesBuffer = device.makeBuffer(
            length: totalResults * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        )
        
        let valuesBuffer = includeValues ? device.makeBuffer(
            length: totalResults * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) : nil
        
        guard let indices = indicesBuffer else {
            throw VectorError.bufferCreationFailed("Failed to create indices buffer")
        }
        
        // Select kernel
        let pipelineState = mode == .ascending ? batchSelectAscending : batchSelectDescending
        
        // Determine threadgroup size
        let tgs = threadgroupSize ?? min(32, pipelineState.maxTotalThreadsPerThreadgroup)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }
        
        encoder.label = "BatchSelect_\(mode.kernelSuffix)"
        encoder.setComputePipelineState(pipelineState)

        // Set buffers
        encoder.setBuffer(distances, offset: 0, index: 0)
        encoder.setBuffer(indices, offset: 0, index: 1)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 2)

        // Set parameters - individual buffers to match Metal shader signature
        var paramBatchSize = UInt32(batchSize)
        var paramQueryCount = UInt32(queryCount)
        var paramCandidateCount = UInt32(candidateCount)
        var paramK = UInt32(k)
        encoder.setBytes(&paramBatchSize, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&paramQueryCount, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&paramCandidateCount, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&paramK, length: MemoryLayout<UInt32>.size, index: 6)

        // 3D dispatch
        let threadsPerThreadgroup = MTLSize(width: tgs, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: queryCount, depth: batchSize)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return BatchResult(
            indices: indices,
            values: valuesBuffer,
            batchSize: batchSize,
            queryCount: queryCount,
            k: k,
            mode: mode
        )
    }
    
    // MARK: - Convenience Methods
    
    /// Select top-k from array using warp optimization when possible
    public func selectTopK(
        from values: [[Float]],
        k: Int,
        mode: SelectionMode = .ascending
    ) async throws -> [[(index: Int, value: Float)]] {
        guard !values.isEmpty else {
            throw VectorError.invalidInput("Empty input")
        }
        
        let queryCount = values.count
        let candidateCount = values[0].count
        
        // Flatten values
        let flatValues = values.flatMap { $0 }
        guard let distanceBuffer = kernelContext.createBuffer(
            from: flatValues,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create distance buffer")
        }

        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        // Choose optimal kernel based on k
        if k <= MAX_SMALL_K {
            // Use warp-optimized kernel
            let result = try warpSelectSmallK(
                distances: distanceBuffer,
                queryCount: queryCount,
                candidateCount: candidateCount,
                k: k,
                mode: mode,
                commandBuffer: commandBuffer
            )

            commandBuffer.commit()
            await commandBuffer.completed()

            return (0..<queryCount).map { result.results(for: $0) }
        } else {
            // Use batch selection with single batch
            let result = try batchSelect(
                distances: distanceBuffer,
                batchSize: 1,
                queryCount: queryCount,
                candidateCount: candidateCount,
                k: k,
                mode: mode,
                commandBuffer: commandBuffer
            )

            commandBuffer.commit()
            await commandBuffer.completed()

            return (0..<queryCount).map { result.results(batch: 0, query: $0) }
        }
    }
    
    /// Batch process multiple selection tasks
    public func batchProcessSelections(
        batches: [[[Float]]],
        k: Int,
        mode: SelectionMode = .ascending
    ) async throws -> [[[(index: Int, value: Float)]]] {
        guard !batches.isEmpty else { return [] }
        
        let batchSize = batches.count
        let queryCount = batches[0].count
        let candidateCount = batches[0][0].count
        
        // Flatten all batches
        let flatValues = batches.flatMap { batch in
            batch.flatMap { $0 }
        }
        
        guard let distanceBuffer = kernelContext.createBuffer(
            from: flatValues,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create distance buffer")
        }

        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        let result = try batchSelect(
            distances: distanceBuffer,
            batchSize: batchSize,
            queryCount: queryCount,
            candidateCount: candidateCount,
            k: k,
            mode: mode,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        // Extract results
        var allResults: [[[(index: Int, value: Float)]]] = []
        for b in 0..<batchSize {
            var batchResults: [[(index: Int, value: Float)]] = []
            for q in 0..<queryCount {
                batchResults.append(result.results(batch: b, query: q))
            }
            allResults.append(batchResults)
        }
        
        return allResults
    }
    
    // MARK: - VectorCore Integration
    
    /// Select top-k using VectorCore types
    public func selectTopK<V: VectorProtocol>(
        from vectors: [V],
        k: Int,
        mode: SelectionMode = .ascending
    ) async throws -> [(index: Int, value: Float)] where V.Scalar == Float {
        // Convert to single query with vectors as candidates
        let values = vectors.map { Array($0.toArray()) }
        let transposed = transposeMatrix(values) // Treat each vector as a candidate
        
        let results = try await selectTopK(from: [transposed[0]], k: k, mode: mode)
        return results[0]
    }
    
    // MARK: - Performance Analysis
    
    public struct WarpPerformanceMetrics: Sendable {
        public let kernelType: String
        public let executionTime: TimeInterval
        public let throughput: Double // selections per second
        public let warpEfficiency: Double // 0-1, how well warps are utilized
    }
    
    /// Benchmark warp optimization performance
    public func benchmarkWarpOptimization(
        queryCount: Int = 1000,
        candidateCount: Int = 10000,
        k: Int = 10,
        iterations: Int = 100
    ) async throws -> WarpPerformanceMetrics {
        // Generate test data
        let values = (0..<queryCount).map { _ in
            (0..<candidateCount).map { _ in Float.random(in: 0...1) }
        }
        
        var times: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            _ = try await selectTopK(from: values, k: k)
            times.append(CACurrentMediaTime() - start)
        }
        
        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(queryCount) / avgTime
        
        // Calculate warp efficiency (simplified)
        let kernelType = k <= MAX_SMALL_K ? "Warp-Optimized" : "General"
        let efficiency = k <= MAX_SMALL_K ? 0.95 : 0.75 // Warp kernel is more efficient
        
        return WarpPerformanceMetrics(
            kernelType: kernelType,
            executionTime: avgTime,
            throughput: throughput,
            warpEfficiency: efficiency
        )
    }
    
    /// Compare performance between warp-optimized and general kernels
    public func compareKernelPerformance(
        queryCount: Int = 100,
        candidateCount: Int = 10000,
        kValues: [Int] = [5, 10, 20, 32, 50, 100]
    ) async throws -> [(k: Int, warpTime: TimeInterval?, generalTime: TimeInterval?)] {
        var results: [(k: Int, warpTime: TimeInterval?, generalTime: TimeInterval?)] = []
        
        // Generate test data once
        let flatValues = (0..<queryCount * candidateCount).map { _ in Float.random(in: 0...1) }
        guard let distanceBuffer = kernelContext.createBuffer(
            from: flatValues,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create distance buffer")
        }

        for k in kValues {
            var warpTime: TimeInterval? = nil
            var generalTime: TimeInterval? = nil
            
            // Benchmark warp kernel if applicable
            if k <= MAX_SMALL_K {
                let start = CACurrentMediaTime()
                let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
                _ = try warpSelectSmallK(
                    distances: distanceBuffer,
                    queryCount: queryCount,
                    candidateCount: candidateCount,
                    k: k,
                    commandBuffer: commandBuffer
                )
                commandBuffer.commit()
                await commandBuffer.completed()
                warpTime = CACurrentMediaTime() - start
            }

            // Benchmark general kernel
            let start = CACurrentMediaTime()
            let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
            _ = try batchSelect(
                distances: distanceBuffer,
                batchSize: 1,
                queryCount: queryCount,
                candidateCount: candidateCount,
                k: k,
                commandBuffer: commandBuffer
            )
            commandBuffer.commit()
            await commandBuffer.completed()
            generalTime = CACurrentMediaTime() - start
            
            results.append((k: k, warpTime: warpTime, generalTime: generalTime))
        }
        
        return results
    }
    
    // MARK: - Private Helpers
    
    private func transposeMatrix(_ matrix: [[Float]]) -> [[Float]] {
        guard !matrix.isEmpty else { return [] }
        let rows = matrix.count
        let cols = matrix[0].count
        
        var transposed = Array(repeating: Array(repeating: Float(0), count: rows), count: cols)
        for i in 0..<rows {
            for j in 0..<cols {
                transposed[j][i] = matrix[i][j]
            }
        }
        return transposed
    }
}