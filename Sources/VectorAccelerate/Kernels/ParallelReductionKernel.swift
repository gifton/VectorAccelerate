// Parallel Reduction Kernel
// GPU-accelerated aggregation operations

import Metal
import QuartzCore
import Foundation
import VectorCore

// MARK: - Parallel Reduction Kernel

/// Parallel reduction kernel for GPU acceleration
/// Performs aggregation operations (sum, min, max, etc.) across large datasets
public final class ParallelReductionKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let pipelineStateGeneric: any MTLComputePipelineState
    private let THREADGROUP_SIZE: Int

    /// Supported reduction operations
    public enum ReductionOperation: UInt8 {
        case sum = 0
        case minimum = 2
        case maximum = 3
        case argMin = 4
        case argMax = 5
        
        var initialValue: Float {
            switch self {
            case .sum: return 0.0
            case .minimum, .argMin: return Float.infinity
            case .maximum, .argMax: return -Float.infinity
            }
        }
        
        var returnsIndex: Bool {
            switch self {
            case .argMin, .argMax: return true
            default: return false
            }
        }
    }

    /// Result from reduction operation
    public struct ReductionResult {
        public let valueBuffer: any MTLBuffer
        public let indexBuffer: (any MTLBuffer)?
        
        /// Extract single value result
        public var value: Float {
            return valueBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
        }
        
        /// Extract single index result (if available)
        public var index: Int? {
            guard let buffer = indexBuffer else { return nil }
            return Int(buffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
        }
    }
    
    /// Statistics computed from data
    public struct Statistics {
        public let sum: Float
        public let mean: Float
        public let min: Float
        public let max: Float
        public let minIndex: Int
        public let maxIndex: Int
        public let count: Int
        
        public var range: Float { max - min }
    }

    // Parameters matching Metal ReductionParams
    private struct Parameters {
        var num_elements: UInt32
        var stride: UInt32
        var operation: UInt8
        var return_index: UInt8
        var padding: (UInt8, UInt8) = (0, 0)
        var initial_value: Float
    }
    
    // Matches Metal IndexedValue size
    private struct IndexedValue {
        var value: Float
        var index: UInt32
    }

    // MARK: - Initialization

    /// Initialize the ParallelReductionKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)

        guard let kernelGeneric = library.makeFunction(name: "parallel_reduce_kernel") else {
            throw VectorError.shaderNotFound(name: "Could not find parallel reduction kernel")
        }

        do {
            self.pipelineStateGeneric = try device.makeComputePipelineState(function: kernelGeneric)
        } catch {
            throw VectorError.pipelineCreationFailed("Failed to create pipeline state: \(error)")
        }
        
        // Determine optimal threadgroup size. Must be power of 2 and >= 32 for kernel optimizations.
        let maxThreads = self.pipelineStateGeneric.maxTotalThreadsPerThreadgroup
        var tgs = 32
        while (tgs * 2) <= maxThreads {
            tgs *= 2
        }
        self.THREADGROUP_SIZE = tgs
    }

    // MARK: - Compute Methods

    /// Perform reduction on buffer using a multi-pass approach
    /// - Parameters:
    ///   - input: Input buffer to reduce
    ///   - operation: Reduction operation to perform
    ///   - count: Number of elements
    ///   - commandBuffer: Command buffer for GPU execution
    /// - Returns: Reduction result with value and optional index
    public func reduce(
        _ input: any MTLBuffer,
        operation: ReductionOperation,
        count: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws -> ReductionResult {
        if count <= 0 {
            throw VectorError.invalidInput("Count must be > 0")
        }
        
        var currentInputValue = input
        var currentInputIndices: (any MTLBuffer)? = nil // Indices are only provided starting from the 2nd pass
        var currentCount = count
        
        // Track buffers to manage memory
        var intermediateValueBuffers: [any MTLBuffer] = []
        var intermediateIndexBuffers: [any MTLBuffer] = []

        // Multi-pass reduction loop
        while currentCount > 1 {
            let numGroups = (currentCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
            
            // Create output buffers for this pass (GPU Private for efficiency)
            guard let outputValues = device.makeBuffer(
                length: numGroups * MemoryLayout<Float>.stride,
                options: MTLResourceOptions.storageModePrivate
            ) else {
                throw VectorError.bufferCreationFailed("Failed to create value buffer")
            }
            intermediateValueBuffers.append(outputValues)
            
            var outputIndices: (any MTLBuffer)? = nil
            if operation.returnsIndex {
                guard let indicesBuffer = device.makeBuffer(
                    length: numGroups * MemoryLayout<UInt32>.stride,
                    options: MTLResourceOptions.storageModePrivate
                ) else {
                    throw VectorError.bufferCreationFailed("Failed to create index buffer")
                }
                outputIndices = indicesBuffer
                intermediateIndexBuffers.append(indicesBuffer)
            }

            // Execute the pass
            try encodeReductionPass(
                inputValues: currentInputValue,
                inputIndices: currentInputIndices,
                outputValues: outputValues,
                outputIndices: outputIndices,
                count: currentCount,
                operation: operation,
                commandBuffer: commandBuffer
            )
            
            // The output of this pass becomes the input for the next pass
            currentInputValue = outputValues
            currentInputIndices = outputIndices // Crucial for multi-pass ArgMin/ArgMax
            currentCount = numGroups
        }
        
        // The final result (size 1) is in GPU Private memory. Blit to Shared for CPU access.
        return try blitToShared(
            valueBuffer: currentInputValue,
            indexBuffer: currentInputIndices,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Convenience Methods

    /// Reduce array with single value result
    /// - Parameters:
    ///   - array: Input array
    ///   - operation: Reduction operation
    /// - Returns: Tuple of value and optional index
    public func reduce(
        _ array: [Float],
        operation: ReductionOperation
    ) async throws -> (value: Float, index: Int?) {
        guard !array.isEmpty else {
            throw VectorError.invalidInput("Empty input array")
        }
        
        guard let inputBuffer = kernelContext.createBuffer(from: array, options: MTLResourceOptions.storageModeShared) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }
        
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        let result = try reduce(inputBuffer, operation: operation, count: array.count, commandBuffer: commandBuffer)
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        return (result.value, result.index)
    }

    /// Compute comprehensive statistics
    /// - Parameter array: Input array
    /// - Returns: Statistics structure with multiple metrics
    public func computeStatistics(_ array: [Float]) async throws -> Statistics {
        guard !array.isEmpty else {
            throw VectorError.invalidInput("Empty input array")
        }
        
        guard let inputBuffer = kernelContext.createBuffer(from: array, options: MTLResourceOptions.storageModeShared) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        // Run multiple reductions in same command buffer
        let sumResult = try reduce(inputBuffer, operation: .sum, count: array.count, commandBuffer: commandBuffer)
        let minResult = try reduce(inputBuffer, operation: .argMin, count: array.count, commandBuffer: commandBuffer)
        let maxResult = try reduce(inputBuffer, operation: .argMax, count: array.count, commandBuffer: commandBuffer)
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        return Statistics(
            sum: sumResult.value,
            mean: sumResult.value / Float(array.count),
            min: minResult.value,
            max: maxResult.value,
            minIndex: minResult.index ?? 0,
            maxIndex: maxResult.index ?? 0,
            count: array.count
        )
    }

    /// Compute norms for vectors
    /// - Parameters:
    ///   - vectors: Array of vectors
    ///   - normType: Norm type (1 for L1, 2 for L2)
    /// - Returns: Array of norms
    public func computeNorms(
        _ vectors: [[Float]],
        normType: Int = 2
    ) async throws -> [Float] {
        var norms: [Float] = []
        
        for vector in vectors {
            let absVector = normType == 1 ? vector.map { abs($0) } : vector.map { $0 * $0 }
            let (sum, _) = try await reduce(absVector, operation: .sum)
            let norm = normType == 2 ? sqrt(sum) : sum
            norms.append(norm)
        }
        
        return norms
    }

    /// Reduce vectors using VectorCore types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func reduce<V: VectorProtocol>(
        _ vector: V,
        operation: ReductionOperation
    ) async throws -> (value: Float, index: Int?) where V.Scalar == Float {
        guard vector.count > 0 else {
            throw VectorError.invalidInput("Empty vector")
        }

        // Zero-copy buffer creation using withUnsafeBufferPointer
        guard let inputBuffer = kernelContext.createAlignedBufferFromVector(
            vector,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }

        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        let result = try reduce(inputBuffer, operation: operation, count: vector.count, commandBuffer: commandBuffer)

        commandBuffer.commit()
        await commandBuffer.completed()

        return (result.value, result.index)
    }

    // MARK: - Performance Extensions

    /// Performance statistics for reduction
    public struct PerformanceStats: Sendable {
        public let elementsPerSecond: Double
        public let reductionPasses: Int
        public let executionTime: TimeInterval
    }

    /// Reduce with performance monitoring
    public func reduceWithStats(
        _ array: [Float],
        operation: ReductionOperation
    ) async throws -> (value: Float, index: Int?, stats: PerformanceStats) {
        let startTime = CACurrentMediaTime()
        
        // Calculate number of passes
        var passes = 0
        var currentCount = array.count
        while currentCount > 1 {
            passes += 1
            currentCount = (currentCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
        }
        
        let (value, index) = try await reduce(array, operation: operation)
        
        let endTime = CACurrentMediaTime()
        let stats = PerformanceStats(
            elementsPerSecond: Double(array.count) / (endTime - startTime),
            reductionPasses: passes,
            executionTime: endTime - startTime
        )
        
        return (value, index, stats)
    }

    // MARK: - Private Helper Methods

    private func encodeReductionPass(
        inputValues: any MTLBuffer,
        inputIndices: (any MTLBuffer)?,
        outputValues: any MTLBuffer,
        outputIndices: (any MTLBuffer)?,
        count: Int,
        operation: ReductionOperation,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }
        
        encoder.label = "ParallelReductionPass (Count=\(count))"
        encoder.setComputePipelineState(pipelineStateGeneric)

        // Prepare parameters (stride=1 assumed by kernel)
        var params = Parameters(
            num_elements: UInt32(count),
            stride: 1,
            operation: operation.rawValue,
            return_index: operation.returnsIndex ? 1 : 0,
            initial_value: operation.initialValue
        )
        
        // Set buffers: input_values (0), input_indices (1), output_values (2), output_indices (3), params (4)
        encoder.setBuffer(inputValues, offset: 0, index: 0)
        encoder.setBuffer(inputIndices, offset: 0, index: 1)
        encoder.setBuffer(outputValues, offset: 0, index: 2)
        encoder.setBuffer(outputIndices, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 4)
        
        // Set threadgroup memory (shared memory)
        let sharedMemorySize = THREADGROUP_SIZE * MemoryLayout<IndexedValue>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)

        // Dispatch configuration
        let threadsPerThreadgroup = MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        let numThreadgroups = (count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
        let gridSize = MTLSize(width: numThreadgroups * THREADGROUP_SIZE, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
    
    // Helper to copy the final result from GPU Private memory to CPU Shared memory
    private func blitToShared(
        valueBuffer: any MTLBuffer,
        indexBuffer: (any MTLBuffer)?,
        commandBuffer: any MTLCommandBuffer
    ) throws -> ReductionResult {
        guard let sharedValueBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create shared value buffer")
        }
        
        var sharedIndexBuffer: (any MTLBuffer)? = nil
        if indexBuffer != nil {
            sharedIndexBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: MTLResourceOptions.storageModeShared
            )
            if sharedIndexBuffer == nil {
                throw VectorError.bufferCreationFailed("Failed to create shared index buffer")
            }
        }

        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }
        
        blitEncoder.label = "ReductionResultBlit"
        blitEncoder.copy(
            from: valueBuffer,
            sourceOffset: 0,
            to: sharedValueBuffer,
            destinationOffset: 0,
            size: MemoryLayout<Float>.stride
        )
        
        if let indexSrc = indexBuffer, let indexDst = sharedIndexBuffer {
            blitEncoder.copy(
                from: indexSrc,
                sourceOffset: 0,
                to: indexDst,
                destinationOffset: 0,
                size: MemoryLayout<UInt32>.stride
            )
        }
        
        blitEncoder.endEncoding()
        
        return ReductionResult(valueBuffer: sharedValueBuffer, indexBuffer: sharedIndexBuffer)
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation
public func cpuReduce(_ array: [Float], operation: ParallelReductionKernel.ReductionOperation) -> (value: Float, index: Int?) {
    guard !array.isEmpty else { return (0, nil) }
    
    switch operation {
    case .sum:
        return (array.reduce(0, +), nil)
    case .minimum:
        let min = array.min() ?? 0
        return (min, nil)
    case .maximum:
        let max = array.max() ?? 0
        return (max, nil)
    case .argMin:
        let enumerated = array.enumerated()
        let min = enumerated.min { $0.element < $1.element }
        return (min?.element ?? 0, min?.offset)
    case .argMax:
        let enumerated = array.enumerated()
        let max = enumerated.max { $0.element < $1.element }
        return (max?.element ?? 0, max?.offset)
    }
}
