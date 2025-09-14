// Element-wise Operations Kernel
// GPU-accelerated mathematical operations on vectors

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Element-wise Kernel

/// Element-wise operations kernel for GPU acceleration
/// Performs mathematical operations on vectors element-by-element
public final class ElementwiseKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let pipelineState: any MTLComputePipelineState
    private let pipelineStateInplace: any MTLComputePipelineState

    // MARK: - Operation Enum

    /// Supported element-wise operations
    public enum Operation {
        // Binary
        case add, subtract, multiply, divide, power, maximum, minimum
        // Scalar
        case addScalar(Float), multiplyScalar(Float), powerScalar(Float)
        case clamp(min: Float, max: Float)
        // Unary
        case absolute, square, sqrt, reciprocal, negate, exp, log

        var rawValue: UInt8 {
            switch self {
            case .add: return 0
            case .subtract: return 1
            case .multiply: return 2
            case .divide: return 3
            case .power: return 4
            case .maximum: return 5
            case .minimum: return 6
            case .addScalar: return 10
            case .multiplyScalar: return 11
            case .powerScalar: return 12
            case .clamp: return 13
            case .absolute: return 20
            case .square: return 21
            case .sqrt: return 22
            case .reciprocal: return 23
            case .negate: return 24
            case .exp: return 25
            case .log: return 26
            }
        }

        var isBinary: Bool {
            // Binary operations require a second buffer (inputB/operand)
            switch self {
            case .add, .subtract, .multiply, .divide, .power, .maximum, .minimum:
                return true
            default:
                return false
            }
        }
    }

    // MARK: - Parameters Struct

    /// Parameters for element-wise kernel execution
    private struct Parameters {
        var num_elements: UInt32
        var stride_a: UInt32
        var stride_b: UInt32
        var stride_output: UInt32
        var scalar_value: Float
        var scalar_value2: Float
        var scalar_value3: Float
        var operation: UInt8
        var use_fast_math: UInt8
        var padding: (UInt8, UInt8) = (0, 0)
    }

    // MARK: - Initialization

    /// Initialize the ElementwiseKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }
        
        guard let kernel = library.makeFunction(name: "elementwise_operation_kernel"),
              let kernelInplace = library.makeFunction(name: "elementwise_inplace_kernel") else {
            throw AccelerationError.shaderNotFound(name: "Could not find element-wise kernel functions")
        }
        
        do {
            self.pipelineState = try device.makeComputePipelineState(function: kernel)
            self.pipelineStateInplace = try device.makeComputePipelineState(function: kernelInplace)
        } catch {
            throw AccelerationError.pipelineCreationFailed("Failed to create pipeline state: \(error)")
        }
    }

    // MARK: - Compute Methods

    /// Perform element-wise operation (out-of-place or in-place)
    /// - Parameters:
    ///   - inputA: First input buffer
    ///   - inputB: Second input buffer (for binary operations)
    ///   - output: Output buffer
    ///   - operation: Operation to perform
    ///   - count: Number of elements
    ///   - useFastMath: Use fast math variants when available
    ///   - strideA: Stride for input A
    ///   - strideB: Stride for input B
    ///   - strideOutput: Stride for output
    ///   - commandBuffer: Command buffer for GPU execution
    public func compute(
        inputA: any MTLBuffer,
        inputB: (any MTLBuffer)? = nil,
        output: any MTLBuffer,
        operation: Operation,
        count: Int,
        useFastMath: Bool = false,
        strideA: Int = 1,
        strideB: Int = 1,
        strideOutput: Int = 1,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        if count == 0 { return }
        
        // Validation
        if operation.isBinary && inputB == nil {
            throw AccelerationError.invalidInput("Binary operation requires second input buffer")
        }

        let isInPlace = (inputA === output)

        // Prepare Parameters
        var params = prepareParameters(
            operation: operation,
            count: count,
            useFastMath: useFastMath,
            strideA: strideA,
            strideB: strideB,
            strideOutput: strideOutput
        )

        // Encoding
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "ElementwiseKernel"

        let pipeline: any MTLComputePipelineState
        
        if isInPlace {
            if strideA != strideOutput {
                throw AccelerationError.invalidInput("strideA must equal strideOutput for in-place operations")
            }
            pipeline = pipelineStateInplace
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputA, offset: 0, index: 0) // data
            encoder.setBuffer(inputB, offset: 0, index: 1) // operand
            // Parameters at index 2 for in-place
            encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 2)
        } else {
            pipeline = pipelineState
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputA, offset: 0, index: 0) // input_a
            encoder.setBuffer(inputB, offset: 0, index: 1) // input_b
            encoder.setBuffer(output, offset: 0, index: 2) // output
            // Parameters at index 3 for out-of-place
            encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 3)
        }

        // Dispatch
        dispatch(encoder: encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }
    
    // MARK: - Convenience Methods
    
    /// Perform element-wise operation on arrays
    /// - Parameters:
    ///   - a: First input array
    ///   - b: Second input array (for binary operations)
    ///   - operation: Operation to perform
    ///   - useFastMath: Use fast math variants when available
    /// - Returns: Result array
    public func compute(
        _ a: [Float],
        _ b: [Float]? = nil,
        operation: Operation,
        useFastMath: Bool = false
    ) async throws -> [Float] {
        guard !a.isEmpty else { return [] }
        
        if operation.isBinary {
            guard let b = b, b.count == a.count else {
                throw AccelerationError.invalidInput("Binary operation requires matching array sizes")
            }
        }
        
        // Create buffers
        guard let inputBufferA = kernelContext.createBuffer(
            from: a,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        let inputBufferB: (any MTLBuffer)? = try b.map { data in
            guard let buffer = kernelContext.createBuffer(from: data, options: MTLResourceOptions.storageModeShared) else {
                throw AccelerationError.bufferCreationFailed("Failed to create buffer B")
            }
            return buffer
        }
        
        guard let outputBuffer = device.makeBuffer(
            length: a.count * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
        }
        
        // Execute
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            inputA: inputBufferA,
            inputB: inputBufferB,
            output: outputBuffer,
            operation: operation,
            count: a.count,
            useFastMath: useFastMath,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: a.count)
        return Array(UnsafeBufferPointer(start: pointer, count: a.count))
    }
    
    /// Perform element-wise operation using VectorCore types
    public func compute<V: VectorProtocol>(
        _ a: V,
        _ b: V? = nil,
        operation: Operation,
        useFastMath: Bool = false
    ) async throws -> V where V.Scalar == Float {
        let arrayA = Array(a.toArray())
        let arrayB = b.map { Array($0.toArray()) }
        let result = try await compute(arrayA, arrayB, operation: operation, useFastMath: useFastMath)
        
        return try V(result)
    }
    
    // MARK: - Performance Extensions
    
    /// Performance statistics for element-wise operations
    public struct PerformanceStats: Sendable {
        public let elementsPerSecond: Double
        public let bandwidth: Double // GB/s
        public let executionTime: TimeInterval
    }
    
    /// Compute with performance monitoring
    public func computeWithStats(
        _ a: [Float],
        _ b: [Float]? = nil,
        operation: Operation,
        useFastMath: Bool = false
    ) async throws -> ([Float], PerformanceStats) {
        let startTime = CACurrentMediaTime()
        
        let result = try await compute(a, b, operation: operation, useFastMath: useFastMath)
        
        let endTime = CACurrentMediaTime()
        let executionTime = endTime - startTime
        
        let bytesProcessed = a.count * MemoryLayout<Float>.size * (b != nil ? 3 : 2) // Read + Write
        
        let stats = PerformanceStats(
            elementsPerSecond: Double(a.count) / executionTime,
            bandwidth: Double(bytesProcessed) / (executionTime * 1e9), // GB/s
            executionTime: executionTime
        )
        
        return (result, stats)
    }
    
    // MARK: - Helper Methods
    
    private func prepareParameters(
        operation: Operation,
        count: Int,
        useFastMath: Bool,
        strideA: Int,
        strideB: Int,
        strideOutput: Int
    ) -> Parameters {
        var scalar1: Float = 0.0
        var scalar2: Float = 0.0
        var scalar3: Float = 0.0
        
        switch operation {
        case .addScalar(let v), .multiplyScalar(let v), .powerScalar(let v):
            scalar1 = v
        case .clamp(let minV, let maxV):
            scalar2 = minV
            scalar3 = maxV
        default:
            break
        }
        
        return Parameters(
            num_elements: UInt32(count),
            stride_a: UInt32(strideA),
            stride_b: UInt32(strideB),
            stride_output: UInt32(strideOutput),
            scalar_value: scalar1,
            scalar_value2: scalar2,
            scalar_value3: scalar3,
            operation: operation.rawValue,
            use_fast_math: useFastMath ? 1 : 0
        )
    }
    
    private func dispatch(encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState, count: Int) {
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let w = pipeline.threadExecutionWidth
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroupSize = min(count, min(maxThreads, w * (maxThreads / w)))
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}

// MARK: - CPU Reference Implementations

/// CPU reference implementation for validation
public func cpuElementwise(_ a: [Float], _ b: [Float]? = nil, operation: ElementwiseKernel.Operation) -> [Float] {
    switch operation {
    case .add:
        return zip(a, b!).map { $0 + $1 }
    case .subtract:
        return zip(a, b!).map { $0 - $1 }
    case .multiply:
        return zip(a, b!).map { $0 * $1 }
    case .divide:
        return zip(a, b!).map { $0 / $1 }
    case .power:
        return zip(a, b!).map { pow($0, $1) }
    case .maximum:
        return zip(a, b!).map { max($0, $1) }
    case .minimum:
        return zip(a, b!).map { min($0, $1) }
    case .addScalar(let s):
        return a.map { $0 + s }
    case .multiplyScalar(let s):
        return a.map { $0 * s }
    case .powerScalar(let s):
        return a.map { pow($0, s) }
    case .clamp(let minV, let maxV):
        return a.map { max(minV, min(maxV, $0)) }
    case .absolute:
        return a.map { abs($0) }
    case .square:
        return a.map { $0 * $0 }
    case .sqrt:
        return a.map { sqrt($0) }
    case .reciprocal:
        return a.map { 1.0 / $0 }
    case .negate:
        return a.map { -$0 }
    case .exp:
        return a.map { exp($0) }
    case .log:
        return a.map { log($0) }
    }
}