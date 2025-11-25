// Scalar Quantization Kernel
// GPU-accelerated quantization for memory-efficient vector storage

import Metal
import Foundation
import VectorCore

// MARK: - Scalar Quantization Kernel

/// Scalar quantization kernel for GPU acceleration
/// Reduces memory footprint by converting float32 to int8/int4 with scale and offset
public final class ScalarQuantizationKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext

    private let quantizeInt8Pipeline: any MTLComputePipelineState
    private let dequantizeInt8Pipeline: any MTLComputePipelineState
    private let quantizeInt4Pipeline: any MTLComputePipelineState
    private let dequantizeInt4Pipeline: any MTLComputePipelineState

    // MARK: - Enums and Structs

    /// Quantization scheme type
    public enum QuantizationType: UInt8, Sendable {
        case symmetric = 0      // Zero-point = 0
        case asymmetric = 1     // With zero-point
        case perChannel = 2     // Different scale/zero-point per channel
        case dynamic = 3        // Compute parameters from data
    }

    /// Quantization bit width
    public enum BitWidth: UInt8, Sendable {
        case int8 = 8
        case int4 = 4
        
        var bytesPerElement: Float {
            switch self {
            case .int8: return 1.0
            case .int4: return 0.5
            }
        }
    }

    /// Quantization context containing configuration and buffers
    public struct QuantizationContext {
        public let quantizedData: (any MTLBuffer)?  // Nil during quantization, required for dequantization
        public let scales: (any MTLBuffer)?         // Per-channel scales
        public let zeroPoints: (any MTLBuffer)?     // Per-channel zero points
        public let bitWidth: BitWidth
        public let type: QuantizationType
        public let numElements: Int
        public let numChannels: Int
        // Fallback values if buffers are not provided (static/global quantization)
        public let globalScale: Float
        public let globalZeroPoint: Int8
        
        /// Elements per channel
        public var elementsPerChannel: Int {
            (numChannels > 0) ? numElements / numChannels : numElements
        }
        
        public init(
            quantizedData: (any MTLBuffer)? = nil,
            scales: (any MTLBuffer)? = nil,
            zeroPoints: (any MTLBuffer)? = nil,
            bitWidth: BitWidth,
            type: QuantizationType,
            numElements: Int,
            numChannels: Int = 1,
            globalScale: Float = 1.0,
            globalZeroPoint: Int8 = 0
        ) {
            self.quantizedData = quantizedData
            self.scales = scales
            self.zeroPoints = zeroPoints
            self.bitWidth = bitWidth
            self.type = type
            self.numElements = numElements
            self.numChannels = numChannels
            self.globalScale = globalScale
            self.globalZeroPoint = globalZeroPoint
        }
    }
    
    /// Quantization result with metadata
    public struct QuantizationResult: Sendable {
        public let quantizedData: Data
        public let scale: Float
        public let zeroPoint: Int8?
        public let bitWidth: BitWidth
        public let compressionRatio: Float
    }
    
    /// Quantization quality metrics
    public struct QuantizationMetrics: Sendable {
        public let mse: Float              // Mean Squared Error
        public let maxError: Float         // Maximum absolute error
        public let compressionRatio: Float // Compression achieved
    }
    
    // Parameters matching Metal struct
    private struct Parameters {
        var num_elements: UInt32
        var num_channels: UInt32
        var elements_per_channel: UInt32
        var global_scale: Float
        var global_zero_point: Int8
        var quantization_type: UInt8
        var bit_width: UInt8
        var padding: UInt8 = 0
    }

    // MARK: - Initialization

    /// Initialize the ScalarQuantizationKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)

        guard let qInt8 = library.makeFunction(name: "quantize_int8_kernel"),
              let dqInt8 = library.makeFunction(name: "dequantize_int8_kernel"),
              let qInt4 = library.makeFunction(name: "quantize_int4_kernel"),
              let dqInt4 = library.makeFunction(name: "dequantize_int4_kernel") else {
            throw AccelerationError.shaderNotFound(name: "Could not find quantization kernel functions")
        }

        do {
            self.quantizeInt8Pipeline = try device.makeComputePipelineState(function: qInt8)
            self.dequantizeInt8Pipeline = try device.makeComputePipelineState(function: dqInt8)
            self.quantizeInt4Pipeline = try device.makeComputePipelineState(function: qInt4)
            self.dequantizeInt4Pipeline = try device.makeComputePipelineState(function: dqInt4)
        } catch {
            throw AccelerationError.pipelineCreationFailed("Failed to create pipeline state: \(error)")
        }
    }

    // MARK: - Compute Methods

    /// Quantize float data to INT8/INT4
    /// - Parameters:
    ///   - input: Input buffer containing float data
    ///   - output: Output buffer for quantized data
    ///   - context: Quantization context with configuration
    ///   - commandBuffer: Command buffer for GPU execution
    public func quantize(
        input: any MTLBuffer,
        output: any MTLBuffer,
        context: QuantizationContext,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        if context.numElements == 0 { return }
        
        // Validation
        if input.length < context.numElements * MemoryLayout<Float>.stride {
            throw AccelerationError.invalidInput("Input buffer too small")
        }

        // Prepare Parameters
        var params = prepareParameters(context: context)

        // Encoding
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        encoder.label = "Quantize-\(context.bitWidth.rawValue)bit"

        let pipeline: any MTLComputePipelineState
        let dispatchCount: Int
        
        switch context.bitWidth {
        case .int8:
            pipeline = quantizeInt8Pipeline
            dispatchCount = context.numElements
        case .int4:
            pipeline = quantizeInt4Pipeline
            // Grid size is the number of output bytes (N+1)/2
            dispatchCount = (context.numElements + 1) / 2
        }
                
        encoder.setComputePipelineState(pipeline)
        
        // Signature: input (0), output (1), scales (2), zero_points (3), params (4)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(context.scales, offset: 0, index: 2)
        encoder.setBuffer(context.zeroPoints, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 4)

        // Dispatch
        dispatch(encoder: encoder, pipeline: pipeline, count: dispatchCount)
        encoder.endEncoding()
    }

    /// Dequantize INT8/INT4 data back to float
    /// - Parameters:
    ///   - context: Quantization context with quantized data
    ///   - output: Output buffer for float data
    ///   - commandBuffer: Command buffer for GPU execution
    public func dequantize(
        context: QuantizationContext,
        output: any MTLBuffer,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        if context.numElements == 0 { return }

        guard let inputData = context.quantizedData else {
            throw AccelerationError.invalidInput("Quantized data buffer is missing")
        }
        
        if output.length < context.numElements * MemoryLayout<Float>.stride {
            throw AccelerationError.invalidInput("Output buffer too small")
        }

        // Prepare Parameters
        var params = prepareParameters(context: context)

        // Encoding
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        encoder.label = "Dequantize-\(context.bitWidth.rawValue)bit"

        let pipeline: any MTLComputePipelineState
        let dispatchCount: Int
        
        switch context.bitWidth {
        case .int8:
            pipeline = dequantizeInt8Pipeline
            dispatchCount = context.numElements
        case .int4:
            pipeline = dequantizeInt4Pipeline
            // Grid size is the number of input bytes (N+1)/2
            dispatchCount = (context.numElements + 1) / 2
        }
        
        encoder.setComputePipelineState(pipeline)
        
        // Signature: input (0), output (1), scales (2), zero_points (3), params (4)
        encoder.setBuffer(inputData, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(context.scales, offset: 0, index: 2)
        encoder.setBuffer(context.zeroPoints, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 4)

        // Dispatch
        dispatch(encoder: encoder, pipeline: pipeline, count: dispatchCount)
        encoder.endEncoding()
    }
    
    // MARK: - Convenience Methods
    
    /// Quantize array to INT8/INT4
    /// - Parameters:
    ///   - data: Input float array
    ///   - bitWidth: Quantization bit width
    ///   - type: Quantization type
    /// - Returns: Quantization result with metadata
    public func quantize(
        _ data: [Float],
        bitWidth: BitWidth = .int8,
        type: QuantizationType = .symmetric
    ) async throws -> QuantizationResult {
        guard !data.isEmpty else {
            throw AccelerationError.invalidInput("Empty input data")
        }
        
        guard let inputBuffer = kernelContext.createBuffer(
            from: data,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        let outputSize = bitWidth == .int8 ? data.count : (data.count + 1) / 2
        guard let outputBuffer = device.makeBuffer(
            length: outputSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
        }
        
        // Compute scale and zero point
        let (scale, zeroPoint) = computeQuantizationParams(data, type: type)
        
        let context = QuantizationContext(
            bitWidth: bitWidth,
            type: type,
            numElements: data.count,
            globalScale: scale,
            globalZeroPoint: zeroPoint ?? 0
        )
        
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try quantize(
            input: inputBuffer,
            output: outputBuffer,
            context: context,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        // Extract quantized data
        let quantizedData = Data(bytes: outputBuffer.contents(), count: outputSize)
        let compressionRatio = Float(data.count * 4) / Float(outputSize)
        
        return QuantizationResult(
            quantizedData: quantizedData,
            scale: scale,
            zeroPoint: type == .asymmetric ? zeroPoint : nil,
            bitWidth: bitWidth,
            compressionRatio: compressionRatio
        )
    }
    
    /// Dequantize data back to float array
    public func dequantize(
        _ result: QuantizationResult,
        count: Int
    ) async throws -> [Float] {
        let quantizedBuffer = device.makeBuffer(
            bytes: [UInt8](result.quantizedData),
            length: result.quantizedData.count,
            options: MTLResourceOptions.storageModeShared
        )!
        
        let outputBuffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        )!
        
        let context = QuantizationContext(
            quantizedData: quantizedBuffer,
            bitWidth: result.bitWidth,
            type: result.zeroPoint != nil ? .asymmetric : .symmetric,
            numElements: count,
            globalScale: result.scale,
            globalZeroPoint: result.zeroPoint ?? 0
        )
        
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try dequantize(
            context: context,
            output: outputBuffer,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    /// Quantize with quality metrics
    public func quantizeWithMetrics(
        _ data: [Float],
        bitWidth: BitWidth = .int8,
        type: QuantizationType = .symmetric
    ) async throws -> (result: QuantizationResult, metrics: QuantizationMetrics) {
        // Quantize
        let result = try await quantize(data, bitWidth: bitWidth, type: type)
        
        // Dequantize to measure error
        let reconstructed = try await dequantize(result, count: data.count)
        
        // Compute metrics
        var mse: Float = 0
        var maxError: Float = 0
        for i in 0..<data.count {
            let error = abs(data[i] - reconstructed[i])
            mse += error * error
            maxError = max(maxError, error)
        }
        mse /= Float(data.count)
        
        let metrics = QuantizationMetrics(
            mse: mse,
            maxError: maxError,
            compressionRatio: result.compressionRatio
        )
        
        return (result, metrics)
    }
    
    /// Quantize vectors using VectorCore types
    public func quantize<V: VectorProtocol>(
        _ vector: V,
        bitWidth: BitWidth = .int8,
        type: QuantizationType = .symmetric
    ) async throws -> QuantizationResult where V.Scalar == Float {
        let array = Array(vector.toArray())
        return try await quantize(array, bitWidth: bitWidth, type: type)
    }
    
    // MARK: - Helper Methods
    
    private func prepareParameters(context: QuantizationContext) -> Parameters {
        return Parameters(
            num_elements: UInt32(context.numElements),
            num_channels: UInt32(context.numChannels),
            elements_per_channel: UInt32(context.elementsPerChannel),
            global_scale: context.globalScale,
            global_zero_point: context.globalZeroPoint,
            quantization_type: context.type.rawValue,
            bit_width: context.bitWidth.rawValue
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
    
    private func computeQuantizationParams(_ data: [Float], type: QuantizationType) -> (scale: Float, zeroPoint: Int8?) {
        let minVal = data.min() ?? 0
        let maxVal = data.max() ?? 0
        
        switch type {
        case .symmetric:
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = absMax / 127.0
            return (scale, nil)
            
        case .asymmetric:
            let scale = (maxVal - minVal) / 255.0
            let zeroPoint = Int8(max(-128, min(127, -round(minVal / scale))))
            return (scale, zeroPoint)
            
        default:
            // For per-channel and dynamic, use symmetric as default
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = absMax / 127.0
            return (scale, nil)
        }
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation
public func cpuQuantizeInt8(_ data: [Float], scale: Float, zeroPoint: Int8 = 0) -> [Int8] {
    return data.map { value in
        let scaled = value / scale + Float(zeroPoint)
        let clamped = max(-128, min(127, Int(roundf(scaled))))
        return Int8(clamped)
    }
}

/// CPU reference dequantization
public func cpuDequantizeInt8(_ quantized: [Int8], scale: Float, zeroPoint: Int8 = 0) -> [Float] {
    return quantized.map { q in
        (Float(q) - Float(zeroPoint)) * scale
    }
}
