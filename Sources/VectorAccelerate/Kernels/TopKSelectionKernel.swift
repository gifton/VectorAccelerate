// Top-K Selection Kernel
// GPU-accelerated selection of K largest or smallest elements

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Top-K Selection Kernel

/// Top-K selection kernel for GPU acceleration
/// Finds the K largest or smallest elements from a dataset along with their indices
public final class TopKSelectionKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let pipelineStateBatch: any MTLComputePipelineState

    /// Maximum K supported by the kernel (must match MAX_K in Metal code)
    public static let MAX_K_SUPPORTED = 128

    /// Selection mode for top-k operation
    public enum SelectionMode: UInt8 {
        case minimum = 0  // Top-K smallest
        case maximum = 1  // Top-K largest
    }

    /// Result from top-k selection
    public struct TopKResult {
        public let values: [Float]
        public let indices: [Int]
        public let k: Int
        public let sorted: Bool
        
        /// Get the best (first) element
        public var best: (value: Float, index: Int)? {
            guard !values.isEmpty else { return nil }
            return (values[0], indices[0])
        }
        
        /// Get the worst (last) element in the top-k
        public var worst: (value: Float, index: Int)? {
            guard !values.isEmpty else { return nil }
            let last = min(k, values.count) - 1
            return (values[last], indices[last])
        }
    }

    // Parameters matching Metal TopKBatchParams
    private struct BatchParameters {
        var batch_size: UInt32
        var num_elements: UInt32
        var k: UInt32
        var input_stride: UInt32
        var output_stride: UInt32
        var mode: UInt8
        var sorted: UInt8
        var padding: (UInt8, UInt8) = (0, 0)
    }

    // MARK: - Initialization

    /// Initialize the TopKSelectionKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }

        guard let kernelBatch = library.makeFunction(name: "topk_select_batch_kernel") else {
            throw AccelerationError.shaderNotFound(name: "Could not find top-k selection kernel")
        }

        do {
            self.pipelineStateBatch = try device.makeComputePipelineState(function: kernelBatch)
        } catch {
            throw AccelerationError.pipelineCreationFailed("Failed to create pipeline state: \(error)")
        }
    }

    // MARK: - Compute Methods

    /// Batch top-k selection for multiple queries
    /// - Parameters:
    ///   - distances: Buffer containing distance/similarity matrix [Q × N]
    ///   - k: Number of top elements to select
    ///   - batchSize: Number of queries (Q)
    ///   - numElements: Number of elements per query (N)
    ///   - mode: Selection mode (minimum or maximum)
    ///   - sorted: Whether to sort the output
    ///   - commandBuffer: Command buffer for GPU execution
    /// - Returns: Tuple of value and index buffers
    public func selectBatch(
        from distances: any MTLBuffer,
        k: Int,
        batchSize: Int,
        numElements: Int,
        mode: SelectionMode = .minimum,
        sorted: Bool = true,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (values: any MTLBuffer, indices: any MTLBuffer) {
        
        // Validation
        if batchSize <= 0 || numElements <= 0 || k <= 0 {
            // Handle empty or invalid input gracefully
            let emptyVal = device.makeBuffer(length: 0, options: MTLResourceOptions.storageModeShared)!
            let emptyIdx = device.makeBuffer(length: 0, options: MTLResourceOptions.storageModeShared)!
            return (emptyVal, emptyIdx)
        }

        if k > TopKSelectionKernel.MAX_K_SUPPORTED {
            throw AccelerationError.invalidInput("K > \(TopKSelectionKernel.MAX_K_SUPPORTED) is not supported")
        }
        
        // Prepare output buffers
        let outputLength = batchSize * k * MemoryLayout<Float>.stride
        let indicesLength = batchSize * k * MemoryLayout<UInt32>.stride
        
        guard let outValBuffer = device.makeBuffer(length: outputLength, options: MTLResourceOptions.storageModeShared),
              let outIdxBuffer = device.makeBuffer(length: indicesLength, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferCreationFailed("Failed to create output buffers")
        }
        
        // Prepare parameters (assuming dense packing)
        var params = BatchParameters(
            batch_size: UInt32(batchSize),
            num_elements: UInt32(numElements),
            k: UInt32(k),
            input_stride: UInt32(numElements),
            output_stride: UInt32(k),
            mode: mode.rawValue,
            sorted: sorted ? 1 : 0
        )

        // Encoding
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        encoder.label = "TopKSelectionBatch (K=\(k))"
        
        encoder.setComputePipelineState(pipelineStateBatch)
        
        // Set buffers: distances (0), topk_values (1), topk_indices (2), params (3)
        encoder.setBuffer(distances, offset: 0, index: 0)
        encoder.setBuffer(outValBuffer, offset: 0, index: 1)
        encoder.setBuffer(outIdxBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<BatchParameters>.stride, index: 3)

        // Dispatch (1D Grid: 1 thread per query/batch item)
        let gridSize = MTLSize(width: batchSize, height: 1, depth: 1)
        
        // Determine optimal threadgroup size
        let w = pipelineStateBatch.threadExecutionWidth
        let maxThreads = pipelineStateBatch.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroupSize = min(batchSize, min(maxThreads, w * (maxThreads / w)))
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupSize, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return (outValBuffer, outIdxBuffer)
    }

    // MARK: - Convenience Methods

    /// Find K nearest neighbors from distance matrix
    /// - Parameters:
    ///   - distances: 2D array of distances [queries][database]
    ///   - k: Number of neighbors to find
    ///   - mode: Selection mode
    /// - Returns: Array of top-k results per query
    public func findKNearest(
        distances: [[Float]],
        k: Int,
        mode: SelectionMode = .minimum
    ) async throws -> [TopKResult] {
        guard !distances.isEmpty else { return [] }
        
        let batchSize = distances.count
        let numElements = distances[0].count
        
        // Validate all rows have same length
        guard distances.allSatisfy({ $0.count == numElements }) else {
            throw AccelerationError.invalidInput("Inconsistent row sizes in distance matrix")
        }
        
        // Create buffer from flattened matrix
        let flatDistances = distances.flatMap { $0 }
        let distanceBuffer = kernelContext.createBuffer(
            from: flatDistances,
            options: MTLResourceOptions.storageModeShared
        )
        guard let distanceBuffer = distanceBuffer else {
            throw AccelerationError.bufferCreationFailed("Failed to create distance buffer")
        }
        
        // Execute
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        let (valBuffer, idxBuffer) = try selectBatch(
            from: distanceBuffer,
            k: k,
            batchSize: batchSize,
            numElements: numElements,
            mode: mode,
            sorted: true,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let valPtr = valBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * k)
        let idxPtr = idxBuffer.contents().bindMemory(to: UInt32.self, capacity: batchSize * k)
        
        var results: [TopKResult] = []
        for i in 0..<batchSize {
            let offset = i * k
            let values = Array(UnsafeBufferPointer<Float>(start: valPtr + offset, count: k))
            let indices = Array(UnsafeBufferPointer(start: idxPtr + offset, count: k)).map { Int($0) }
            results.append(TopKResult(values: values, indices: indices, k: k, sorted: true))
        }
        
        return results
    }

    /// Select top-k from a single array
    /// - Parameters:
    ///   - array: Input array
    ///   - k: Number of elements to select
    ///   - mode: Selection mode
    ///   - sorted: Whether to sort the output
    /// - Returns: Top-k result
    public func select(
        from array: [Float],
        k: Int,
        mode: SelectionMode = .minimum,
        sorted: Bool = true
    ) async throws -> TopKResult {
        let distances = [array] // Treat as batch of 1
        let results = try await findKNearest(distances: distances, k: k, mode: mode)
        return results.first ?? TopKResult(values: [], indices: [], k: k, sorted: sorted)
    }

    /// Find K most similar vectors using VectorCore types
    public func findMostSimilar<V: VectorProtocol>(
        to query: V,
        in database: [V],
        k: Int,
        distanceMetric: (V, V) -> Float
    ) async throws -> [(vector: V, distance: Float, index: Int)] where V.Scalar == Float {
        // Compute distances
        let distances = database.map { distanceMetric(query, $0) }
        
        // Find top-k
        let result = try await select(from: distances, k: k, mode: .minimum, sorted: true)
        
        // Return vectors with distances
        var results: [(vector: V, distance: Float, index: Int)] = []
        for i in 0..<min(k, result.indices.count) {
            let idx = result.indices[i]
            if idx < database.count && idx != Int(UInt32.max) {
                results.append((database[idx], result.values[i], idx))
            }
        }
        return results
    }

    // MARK: - Performance Extensions

    /// Performance statistics for top-k selection
    public struct PerformanceStats: Sendable {
        public let selectionsPerSecond: Double
        public let executionTime: TimeInterval
        public let throughput: Double // Elements/second
    }

    /// Select batch with performance monitoring
    public func selectBatchWithStats(
        from distances: any MTLBuffer,
        k: Int,
        batchSize: Int,
        numElements: Int,
        mode: SelectionMode = .minimum,
        commandBuffer: any MTLCommandBuffer
    ) async throws -> (values: any MTLBuffer, indices: any MTLBuffer, stats: PerformanceStats) {
        let startTime = CACurrentMediaTime()
        
        let (values, indices) = try selectBatch(
            from: distances,
            k: k,
            batchSize: batchSize,
            numElements: numElements,
            mode: mode,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let endTime = CACurrentMediaTime()
        let executionTime = endTime - startTime
        
        let stats = PerformanceStats(
            selectionsPerSecond: Double(batchSize) / executionTime,
            executionTime: executionTime,
            throughput: Double(batchSize * numElements) / executionTime
        )
        
        return (values, indices, stats)
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation
public func cpuTopK(_ array: [Float], k: Int, ascending: Bool = true) -> (values: [Float], indices: [Int]) {
    let indexed = array.enumerated().map { ($0.element, $0.offset) }
    let sorted = indexed.sorted { ascending ? $0.0 < $1.0 : $0.0 > $1.0 }
    let topK = Array(sorted.prefix(k))
    return (topK.map { $0.0 }, topK.map { $0.1 })
}