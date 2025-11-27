// Jaccard Distance Kernel
// GPU-accelerated Jaccard distance computation for set similarity

import Metal
import Foundation
import VectorCore
import QuartzCore

// MARK: - Jaccard Distance Kernel

/// GPU-accelerated Jaccard distance computation
/// Optimized for binary vector comparisons with reduction pattern
public final class JaccardDistanceKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLComputePipelineState
    
    // Kernel configuration matching Metal shader requirements
    private let THREADS_PER_TG: Int = 256
    
    // MARK: - Result Types
    
    /// Result from single Jaccard distance computation
    public struct JaccardResult: Sendable {
        public let distance: Float
        public let similarity: Float
        public let intersectionSize: Int
        public let unionSize: Int
        public let executionTime: TimeInterval
        
        /// Jaccard coefficient (same as similarity)
        public var coefficient: Float { similarity }
        
        /// Check if vectors are identical
        public var isIdentical: Bool { distance == 0.0 }
        
        /// Check if vectors are completely disjoint
        public var isDisjoint: Bool { similarity == 0.0 }
    }
    
    /// Result from batch Jaccard distance computation
    public struct BatchJaccardResult: Sendable {
        public let distances: [Float]
        public let rows: Int
        public let cols: Int
        public let totalExecutionTime: TimeInterval
        public let averageDistance: Double
        
        /// Get distance at (row, col)
        public func distance(row: Int, col: Int) -> Float {
            guard row >= 0 && row < rows && col >= 0 && col < cols else {
                fatalError("Index (\(row), \(col)) out of bounds for \(rows)×\(cols) matrix")
            }
            return distances[row * cols + col]
        }
        
        /// Extract full distance matrix
        public func asMatrix() -> [[Float]] {
            var matrix = Array(repeating: [Float](repeating: 0.0, count: cols), count: rows)
            for r in 0..<rows {
                for c in 0..<cols {
                    matrix[r][c] = distances[r * cols + c]
                }
            }
            return matrix
        }
        
        /// Get row of distances
        public func row(_ index: Int) -> [Float] {
            guard index >= 0 && index < rows else {
                fatalError("Row index \(index) out of bounds for \(rows) rows")
            }
            let start = index * cols
            let end = start + cols
            return Array(distances[start..<end])
        }
    }
    
    /// Configuration for Jaccard distance computation
    public struct JaccardConfig: Sendable {
        public let threshold: Float          // Threshold for binary conversion
        public let useGPUReduction: Bool     // Use GPU for intersection/union counting
        public let batchSize: Int            // Optimal batch size for operations
        
        public init(
            threshold: Float = 0.0,
            useGPUReduction: Bool = true,
            batchSize: Int = 1024
        ) {
            self.threshold = threshold
            self.useGPUReduction = useGPUReduction
            self.batchSize = batchSize
        }
        
        public static let `default` = JaccardConfig()
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw VectorError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        guard let function = library.makeFunction(name: "jaccardDistance") else {
            throw VectorError.shaderNotFound(name: "jaccardDistance")
        }
        
        do {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            throw VectorError.computeFailed(reason: "Failed to create pipeline state: \(error)")
        }
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < THREADS_PER_TG {
            throw VectorError.unsupportedGPUOperation(
                "Device does not support required threadgroup size: \(THREADS_PER_TG)"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Compute Jaccard distance between two vectors
    /// - Parameters:
    ///   - vectorA: First input vector
    ///   - vectorB: Second input vector
    ///   - config: Computation configuration
    /// - Returns: Jaccard distance result with performance metrics
    public func computeDistance(
        vectorA: [Float],
        vectorB: [Float],
        config: JaccardConfig = .default
    ) throws -> JaccardResult {
        guard vectorA.count == vectorB.count else {
            throw VectorError.countMismatch(
                expected: vectorA.count,
                actual: vectorB.count
            )
        }
        
        if vectorA.isEmpty {
            return JaccardResult(
                distance: 0.0,
                similarity: 1.0,
                intersectionSize: 0,
                unionSize: 0,
                executionTime: 0.0
            )
        }
        
        let dimension = UInt32(vectorA.count)
        
        // Create buffers
        guard let bufferA = device.makeBuffer(
            bytes: vectorA,
            length: vectorA.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: vectorA.count * MemoryLayout<Float>.stride)
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: vectorB,
            length: vectorB.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: vectorB.count * MemoryLayout<Float>.stride)
        }
        
        guard let resultBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<Float>.stride)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        // Set parameters
        var params = dimension
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size, index: 3)
        
        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 4)
        
        // Configure thread groups
        let threadgroupSize = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(dimension) + THREADS_PER_TG - 1) / THREADS_PER_TG,
            height: 1,
            depth: 1
        )
        
        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw VectorError.computeFailed(reason: "Jaccard distance computation failed: \(error)")
        }
        
        // Extract result
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        let distance = resultPointer.pointee
        
        // Calculate additional metrics (CPU fallback for detailed analysis)
        let metrics = calculateJaccardMetrics(vectorA: vectorA, vectorB: vectorB, threshold: config.threshold)
        
        return JaccardResult(
            distance: distance,
            similarity: 1.0 - distance,
            intersectionSize: metrics.intersection,
            unionSize: metrics.union,
            executionTime: executionTime
        )
    }
    
    /// Compute pairwise Jaccard distances between two sets of vectors
    /// - Parameters:
    ///   - vectorsA: First set of vectors (M vectors)
    ///   - vectorsB: Second set of vectors (N vectors)  
    ///   - config: Computation configuration
    /// - Returns: M×N distance matrix with performance metrics
    public func computeDistanceMatrix(
        vectorsA: [[Float]],
        vectorsB: [[Float]],
        config: JaccardConfig = .default
    ) throws -> BatchJaccardResult {
        guard let dimension = vectorsA.first?.count,
              dimension > 0,
              vectorsB.first?.count == dimension else {
            throw VectorError.invalidInput("Input vectors must be non-empty and share the same dimension")
        }
        
        // Validate all vectors have same dimension
        for (_, vector) in vectorsA.enumerated() {
            guard vector.count == dimension else {
                throw VectorError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        for (_, vector) in vectorsB.enumerated() {
            guard vector.count == dimension else {
                throw VectorError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        
        let rows = vectorsA.count
        let cols = vectorsB.count
        
        // Flatten input vectors
        let flatA = vectorsA.flatMap { $0 }
        let flatB = vectorsB.flatMap { $0 }
        
        // Create input buffers
        guard let bufferA = device.makeBuffer(
            bytes: flatA,
            length: flatA.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatA.count * MemoryLayout<Float>.stride)
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: flatB,
            length: flatB.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatB.count * MemoryLayout<Float>.stride)
        }
        
        let resultSize = rows * cols * MemoryLayout<Float>.stride
        guard let resultBuffer = device.makeBuffer(length: resultSize, options: MTLResourceOptions.storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: resultSize)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.computeFailed(reason: "Failed to create command encoder")
        }
        
        encoder.setComputePipelineState(pipelineState)
        
        // Set constant parameters
        var params = UInt32(dimension)
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size, index: 3)
        
        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 4)
        
        // Configure thread groups (constant for all iterations)
        let threadgroupSize = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (dimension + THREADS_PER_TG - 1) / THREADS_PER_TG,
            height: 1,
            depth: 1
        )
        
        let vectorStride = dimension * MemoryLayout<Float>.stride
        
        // Execute all pairwise computations in single command buffer
        let startTime = CACurrentMediaTime()
        
        for r in 0..<rows {
            let offsetA = r * vectorStride
            encoder.setBuffer(bufferA, offset: offsetA, index: 0)
            
            for c in 0..<cols {
                let offsetB = c * vectorStride
                let resultOffset = (r * cols + c) * MemoryLayout<Float>.stride
                
                encoder.setBuffer(bufferB, offset: offsetB, index: 1)
                encoder.setBuffer(resultBuffer, offset: resultOffset, index: 2)
                
                encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            }
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let totalExecutionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw VectorError.computeFailed(reason: "Batch Jaccard computation failed: \(error)")
        }
        
        // Extract results
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let distances = Array(UnsafeBufferPointer(start: resultPointer, count: rows * cols))
        
        // Calculate average distance
        let averageDistance = distances.reduce(0.0) { $0 + Double($1) } / Double(distances.count)
        
        return BatchJaccardResult(
            distances: distances,
            rows: rows,
            cols: cols,
            totalExecutionTime: totalExecutionTime,
            averageDistance: averageDistance
        )
    }
    
    // MARK: - Async Operations
    
    /// Async version of single distance computation
    public func computeDistanceAsync(
        vectorA: [Float],
        vectorB: [Float],
        config: JaccardConfig = .default
    ) async throws -> JaccardResult {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try computeDistance(vectorA: vectorA, vectorB: vectorB, config: config)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    /// Async version of batch distance computation
    public func computeDistanceMatrixAsync(
        vectorsA: [[Float]],
        vectorsB: [[Float]],
        config: JaccardConfig = .default
    ) async throws -> BatchJaccardResult {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try computeDistanceMatrix(vectorsA: vectorsA, vectorsB: vectorsB, config: config)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - VectorCore Integration

    /// Compute Jaccard distance using VectorCore protocol types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func computeDistance<V: VectorProtocol>(
        _ vectorA: V,
        _ vectorB: V,
        config: JaccardConfig = .default
    ) throws -> JaccardResult where V.Scalar == Float {
        guard vectorA.count == vectorB.count else {
            throw VectorError.countMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        if vectorA.count == 0 {
            return JaccardResult(
                distance: 0.0,
                similarity: 1.0,
                intersectionSize: 0,
                unionSize: 0,
                executionTime: 0.0
            )
        }

        let dimension = UInt32(vectorA.count)

        // Zero-copy buffer creation using withUnsafeBufferPointer
        guard let bufferA = vectorA.withUnsafeBufferPointer({ ptr -> (any MTLBuffer)? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }) else {
            throw VectorError.bufferAllocationFailed(size: vectorA.count * MemoryLayout<Float>.stride)
        }

        guard let bufferB = vectorB.withUnsafeBufferPointer({ ptr -> (any MTLBuffer)? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }) else {
            throw VectorError.bufferAllocationFailed(size: vectorB.count * MemoryLayout<Float>.stride)
        }

        guard let resultBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<Float>.stride)
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.computeFailed(reason: "Failed to create command encoder")
        }

        // Configure compute pass
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)

        // Set parameters
        var params = dimension
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size, index: 3)

        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 4)

        // Configure thread groups
        let threadgroupSize = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(dimension) + THREADS_PER_TG - 1) / THREADS_PER_TG,
            height: 1,
            depth: 1
        )

        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let executionTime = CACurrentMediaTime() - startTime

        // Read result
        let resultPtr = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        let distance = resultPtr.pointee

        return JaccardResult(
            distance: distance,
            similarity: 1.0 - distance,
            intersectionSize: 0, // Not computed in GPU kernel
            unionSize: 0,        // Not computed in GPU kernel
            executionTime: executionTime
        )
    }

    /// Batch computation using VectorCore protocol types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func computeDistanceMatrix<V: VectorProtocol>(
        vectorsA: [V],
        vectorsB: [V],
        config: JaccardConfig = .default
    ) throws -> BatchJaccardResult where V.Scalar == Float {
        guard !vectorsA.isEmpty && !vectorsB.isEmpty else {
            throw VectorError.invalidInput("Empty input arrays")
        }

        let dimension = vectorsA[0].count
        guard vectorsA.allSatisfy({ $0.count == dimension }) &&
              vectorsB.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.countMismatch(expected: dimension, actual: nil)
        }

        // Zero-copy buffer creation: flatten vectors directly into Metal buffer
        guard let bufferA = createBufferFromVectors(vectorsA, dimension: dimension) else {
            throw VectorError.bufferCreationFailed("Failed to create vectorsA buffer")
        }

        guard let bufferB = createBufferFromVectors(vectorsB, dimension: dimension) else {
            throw VectorError.bufferCreationFailed("Failed to create vectorsB buffer")
        }

        // Create result buffer for M x N distances
        let resultSize = vectorsA.count * vectorsB.count * MemoryLayout<Float>.stride
        guard let resultBuffer = device.makeBuffer(length: resultSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: resultSize)
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.computeFailed(reason: "Failed to create command encoder")
        }

        // Configure compute pass
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)

        // Set parameters
        var dim = UInt32(dimension)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 4)

        // For batch: we need to dispatch for each pair
        let totalPairs = vectorsA.count * vectorsB.count
        let threadgroupSize = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (totalPairs + THREADS_PER_TG - 1) / THREADS_PER_TG,
            height: 1,
            depth: 1
        )

        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let executionTime = CACurrentMediaTime() - startTime

        // Read results
        let resultPtr = resultBuffer.contents().bindMemory(to: Float.self, capacity: totalPairs)
        let distances = Array(UnsafeBufferPointer(start: resultPtr, count: totalPairs))

        let avgDistance = distances.isEmpty ? 0.0 : Double(distances.reduce(0, +)) / Double(distances.count)

        return BatchJaccardResult(
            distances: distances,
            rows: vectorsA.count,
            cols: vectorsB.count,
            totalExecutionTime: executionTime,
            averageDistance: avgDistance
        )
    }

    /// Create a Metal buffer directly from VectorProtocol array without intermediate allocations
    private func createBufferFromVectors<V: VectorProtocol>(
        _ vectors: [V],
        dimension: Int
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        guard !vectors.isEmpty else { return nil }

        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.stride

        // Create buffer
        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            return nil
        }

        // Copy each vector directly using withUnsafeBufferPointer
        let destination = buffer.contents().bindMemory(to: Float.self, capacity: totalCount)
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            vector.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                let dst = destination.advanced(by: offset)
                dst.update(from: srcBase, count: min(srcPtr.count, dimension))
            }
        }

        return buffer
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark Jaccard distance computation for different vector sizes
    public func benchmark(sizes: [Int]) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for size in sizes {
            // Generate random binary-like vectors
            let vectorA = (0..<size).map { _ in Float.random(in: 0...1) > 0.5 ? Float(1.0) : Float(0.0) }
            let vectorB = (0..<size).map { _ in Float.random(in: 0...1) > 0.5 ? Float(1.0) : Float(0.0) }
            
            // Warm-up run
            _ = try computeDistance(vectorA: vectorA, vectorB: vectorB)
            
            // Timed runs
            var times: [TimeInterval] = []
            for _ in 0..<5 {
                let result = try computeDistance(vectorA: vectorA, vectorB: vectorB)
                times.append(result.executionTime)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let throughput = Double(size) / avgTime / 1e6 // Million elements per second
            
            results.append(BenchmarkResult(
                vectorSize: size,
                executionTime: avgTime,
                throughputMEPS: throughput
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let vectorSize: Int
        public let executionTime: TimeInterval
        public let throughputMEPS: Double // Million elements per second
    }
    
    // MARK: - Private Utilities
    
    /// CPU fallback for calculating intersection and union counts
    private func calculateJaccardMetrics(
        vectorA: [Float],
        vectorB: [Float],
        threshold: Float
    ) -> (intersection: Int, union: Int) {
        var intersection = 0
        var union = 0
        
        for (a, b) in zip(vectorA, vectorB) {
            let aPresent = a > threshold
            let bPresent = b > threshold
            
            if aPresent || bPresent {
                union += 1
                if aPresent && bPresent {
                    intersection += 1
                }
            }
        }
        
        return (intersection, union)
    }
}
