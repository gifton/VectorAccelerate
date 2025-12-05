//
//  ScalarQuantizationKernel.swift
//  VectorAccelerate
//
//  Metal 4 Scalar Quantization kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 3, Priority 3
//
//  Features:
//  - INT8 and INT4 quantization/dequantization
//  - Symmetric and asymmetric quantization modes
//  - Per-channel quantization support
//  - Fusible with distance kernels via encode() API

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Quantization Types

/// Quantization scheme type
public enum Metal4QuantizationType: UInt8, Sendable {
    /// Zero-point = 0, symmetric around origin
    case symmetric = 0
    /// With non-zero zero-point for asymmetric data
    case asymmetric = 1
    /// Different scale/zero-point per channel
    case perChannel = 2
    /// Compute parameters dynamically from data
    case dynamic = 3
}

/// Quantization bit width
public enum Metal4BitWidth: UInt8, Sendable {
    case int8 = 8
    case int4 = 4

    /// Bytes per element (0.5 for INT4)
    public var bytesPerElement: Float {
        switch self {
        case .int8: return 1.0
        case .int4: return 0.5
        }
    }

    /// Output size for given element count
    public func outputSize(for count: Int) -> Int {
        switch self {
        case .int8: return count
        case .int4: return (count + 1) / 2
        }
    }
}

// MARK: - Parameters

/// Parameters for Scalar Quantization kernel.
///
/// Memory layout must match the Metal shader's parameters struct.
public struct ScalarQuantizationParameters: Sendable {
    /// Total number of elements
    public var numElements: UInt32
    /// Number of channels (for per-channel quantization)
    public var numChannels: UInt32
    /// Elements per channel
    public var elementsPerChannel: UInt32
    /// Global scale factor
    public var globalScale: Float
    /// Global zero point
    public var globalZeroPoint: Int8
    /// Quantization type
    public var quantizationType: UInt8
    /// Bit width
    public var bitWidth: UInt8
    /// Padding for alignment
    private var padding: UInt8 = 0

    public init(
        numElements: Int,
        numChannels: Int = 1,
        globalScale: Float = 1.0,
        globalZeroPoint: Int8 = 0,
        quantizationType: Metal4QuantizationType = .symmetric,
        bitWidth: Metal4BitWidth = .int8
    ) {
        self.numElements = UInt32(numElements)
        self.numChannels = UInt32(numChannels)
        self.elementsPerChannel = numChannels > 0 ? UInt32(numElements / numChannels) : UInt32(numElements)
        self.globalScale = globalScale
        self.globalZeroPoint = globalZeroPoint
        self.quantizationType = quantizationType.rawValue
        self.bitWidth = bitWidth.rawValue
    }
}

// MARK: - Result Types

/// Result from quantization operation
public struct Metal4QuantizationResult: Sendable {
    /// Quantized data as raw bytes
    public let quantizedData: Data
    /// Scale used for quantization
    public let scale: Float
    /// Zero point (nil for symmetric)
    public let zeroPoint: Int8?
    /// Bit width used
    public let bitWidth: Metal4BitWidth
    /// Compression ratio achieved
    public let compressionRatio: Float

    /// Original byte count (assuming float32)
    public var originalBytes: Int {
        let elementCount: Int
        switch bitWidth {
        case .int8:
            elementCount = quantizedData.count
        case .int4:
            elementCount = quantizedData.count * 2
        }
        return elementCount * MemoryLayout<Float>.size
    }
}

/// Quality metrics from quantization
public struct Metal4QuantizationMetrics: Sendable {
    /// Mean Squared Error
    public let mse: Float
    /// Maximum absolute error
    public let maxError: Float
    /// Compression ratio achieved
    public let compressionRatio: Float
    /// Signal-to-Noise Ratio in dB
    public var snr: Float {
        mse > 0 ? -10 * log10(mse) : Float.infinity
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Scalar Quantization kernel.
///
/// Converts float32 vectors to INT8 or INT4 representation for memory-efficient
/// storage. Supports multiple quantization schemes with automatic parameter
/// computation.
///
/// ## Quantization Modes
///
/// - **Symmetric**: Zero-centered, scale only: `q = round(x / scale)`
/// - **Asymmetric**: With offset: `q = round(x / scale) + zero_point`
/// - **Per-channel**: Different parameters per feature dimension
///
/// ## Compression Ratios
///
/// - INT8: 4:1 (32-bit float → 8-bit int)
/// - INT4: 8:1 (32-bit float → 4-bit int, packed)
///
/// ## Usage
///
/// ```swift
/// let kernel = try await ScalarQuantizationKernel(context: context)
///
/// // Quantize with automatic parameter selection
/// let result = try await kernel.quantize(vectors, bitWidth: .int8)
///
/// // Dequantize back to floats
/// let reconstructed = try await kernel.dequantize(result, count: vectors.count)
/// ```
public final class ScalarQuantizationKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "ScalarQuantizationKernel"

    // MARK: - Pipelines

    private let quantizeInt8Pipeline: any MTLComputePipelineState
    private let dequantizeInt8Pipeline: any MTLComputePipelineState
    private let quantizeInt4Pipeline: any MTLComputePipelineState
    private let dequantizeInt4Pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Scalar Quantization kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let qInt8 = library.makeFunction(name: "quantize_int8_kernel"),
              let dqInt8 = library.makeFunction(name: "dequantize_int8_kernel"),
              let qInt4 = library.makeFunction(name: "quantize_int4_kernel"),
              let dqInt4 = library.makeFunction(name: "dequantize_int4_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Scalar quantization kernels. Ensure QuantizationShaders.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.quantizeInt8Pipeline = try await device.makeComputePipelineState(function: qInt8)
        self.dequantizeInt8Pipeline = try await device.makeComputePipelineState(function: dqInt8)
        self.quantizeInt4Pipeline = try await device.makeComputePipelineState(function: qInt4)
        self.dequantizeInt4Pipeline = try await device.makeComputePipelineState(function: dqInt4)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode quantization into an existing encoder.
    @discardableResult
    public func encodeQuantize(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        scales: (any MTLBuffer)?,
        zeroPoints: (any MTLBuffer)?,
        parameters: ScalarQuantizationParameters
    ) -> Metal4EncodingResult {
        let bitWidth = Metal4BitWidth(rawValue: parameters.bitWidth) ?? .int8
        let pipeline: any MTLComputePipelineState
        let dispatchCount: Int

        switch bitWidth {
        case .int8:
            pipeline = quantizeInt8Pipeline
            dispatchCount = Int(parameters.numElements)
        case .int4:
            pipeline = quantizeInt4Pipeline
            dispatchCount = (Int(parameters.numElements) + 1) / 2
        }

        encoder.setComputePipelineState(pipeline)
        encoder.label = "ScalarQuantize.\(bitWidth.rawValue)bit"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBuffer(zeroPoints, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<ScalarQuantizationParameters>.size, index: 4)

        let config = Metal4ThreadConfiguration.linear(count: dispatchCount, pipeline: pipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "quantize_\(bitWidth == .int8 ? "int8" : "int4")_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode dequantization into an existing encoder.
    @discardableResult
    public func encodeDequantize(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        scales: (any MTLBuffer)?,
        zeroPoints: (any MTLBuffer)?,
        parameters: ScalarQuantizationParameters
    ) -> Metal4EncodingResult {
        let bitWidth = Metal4BitWidth(rawValue: parameters.bitWidth) ?? .int8
        let pipeline: any MTLComputePipelineState
        let dispatchCount: Int

        switch bitWidth {
        case .int8:
            pipeline = dequantizeInt8Pipeline
            dispatchCount = Int(parameters.numElements)
        case .int4:
            pipeline = dequantizeInt4Pipeline
            dispatchCount = (Int(parameters.numElements) + 1) / 2
        }

        encoder.setComputePipelineState(pipeline)
        encoder.label = "ScalarDequantize.\(bitWidth.rawValue)bit"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBuffer(zeroPoints, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<ScalarQuantizationParameters>.size, index: 4)

        let config = Metal4ThreadConfiguration.linear(count: dispatchCount, pipeline: pipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "dequantize_\(bitWidth == .int8 ? "int8" : "int4")_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute quantization as standalone operation.
    public func executeQuantize(
        input: any MTLBuffer,
        parameters: ScalarQuantizationParameters
    ) async throws -> any MTLBuffer {
        let bitWidth = Metal4BitWidth(rawValue: parameters.bitWidth) ?? .int8
        let outputSize = bitWidth.outputSize(for: Int(parameters.numElements))

        guard let outputBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "ScalarQuantize.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encodeQuantize(
                into: encoder,
                input: input,
                output: outputBuffer,
                scales: nil,
                zeroPoints: nil,
                parameters: parameters
            )
        }

        return outputBuffer
    }

    /// Execute dequantization as standalone operation.
    public func executeDequantize(
        input: any MTLBuffer,
        parameters: ScalarQuantizationParameters
    ) async throws -> any MTLBuffer {
        let outputSize = Int(parameters.numElements) * MemoryLayout<Float>.size

        guard let outputBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "ScalarDequantize.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encodeDequantize(
                into: encoder,
                input: input,
                output: outputBuffer,
                scales: nil,
                zeroPoints: nil,
                parameters: parameters
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Quantize float array to INT8/INT4.
    ///
    /// Automatically computes optimal scale and zero-point parameters.
    ///
    /// - Parameters:
    ///   - data: Input float array
    ///   - bitWidth: Target bit width (default: INT8)
    ///   - type: Quantization type (default: symmetric)
    /// - Returns: Quantization result with metadata
    public func quantize(
        _ data: [Float],
        bitWidth: Metal4BitWidth = .int8,
        type: Metal4QuantizationType = .symmetric
    ) async throws -> Metal4QuantizationResult {
        guard !data.isEmpty else {
            throw VectorError.invalidInput("Empty input data")
        }

        let device = context.device.rawDevice

        guard let inputBuffer = device.makeBuffer(
            bytes: data,
            length: data.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: data.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "ScalarQuantize.input"

        // Compute quantization parameters
        let (scale, zeroPoint) = computeQuantizationParams(data, type: type, bitWidth: bitWidth)

        let parameters = ScalarQuantizationParameters(
            numElements: data.count,
            globalScale: scale,
            globalZeroPoint: zeroPoint ?? 0,
            quantizationType: type,
            bitWidth: bitWidth
        )

        let outputBuffer = try await executeQuantize(input: inputBuffer, parameters: parameters)

        // Extract quantized data
        let outputSize = bitWidth.outputSize(for: data.count)
        let quantizedData = Data(bytes: outputBuffer.contents(), count: outputSize)
        let compressionRatio = Float(data.count * MemoryLayout<Float>.size) / Float(outputSize)

        return Metal4QuantizationResult(
            quantizedData: quantizedData,
            scale: scale,
            zeroPoint: type == .asymmetric ? zeroPoint : nil,
            bitWidth: bitWidth,
            compressionRatio: compressionRatio
        )
    }

    /// Dequantize data back to float array.
    public func dequantize(
        _ result: Metal4QuantizationResult,
        count: Int
    ) async throws -> [Float] {
        let device = context.device.rawDevice

        guard let inputBuffer = device.makeBuffer(
            bytes: [UInt8](result.quantizedData),
            length: result.quantizedData.count,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: result.quantizedData.count)
        }
        inputBuffer.label = "ScalarDequantize.input"

        let type: Metal4QuantizationType = result.zeroPoint != nil ? .asymmetric : .symmetric
        let parameters = ScalarQuantizationParameters(
            numElements: count,
            globalScale: result.scale,
            globalZeroPoint: result.zeroPoint ?? 0,
            quantizationType: type,
            bitWidth: result.bitWidth
        )

        let outputBuffer = try await executeDequantize(input: inputBuffer, parameters: parameters)

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Quantize with quality metrics.
    ///
    /// Returns both the quantized result and error metrics comparing
    /// original vs reconstructed data.
    public func quantizeWithMetrics(
        _ data: [Float],
        bitWidth: Metal4BitWidth = .int8,
        type: Metal4QuantizationType = .symmetric
    ) async throws -> (result: Metal4QuantizationResult, metrics: Metal4QuantizationMetrics) {
        let result = try await quantize(data, bitWidth: bitWidth, type: type)
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

        let metrics = Metal4QuantizationMetrics(
            mse: mse,
            maxError: maxError,
            compressionRatio: result.compressionRatio
        )

        return (result, metrics)
    }

    /// Quantize using VectorProtocol types.
    public func quantize<V: VectorProtocol>(
        _ vector: V,
        bitWidth: Metal4BitWidth = .int8,
        type: Metal4QuantizationType = .symmetric
    ) async throws -> Metal4QuantizationResult where V.Scalar == Float {
        let data: [Float] = vector.withUnsafeBufferPointer { Array($0) }
        return try await quantize(data, bitWidth: bitWidth, type: type)
    }

    // MARK: - Private Helpers

    private func computeQuantizationParams(
        _ data: [Float],
        type: Metal4QuantizationType,
        bitWidth: Metal4BitWidth
    ) -> (scale: Float, zeroPoint: Int8?) {
        let minVal = data.min() ?? 0
        let maxVal = data.max() ?? 0

        // Use correct range based on bit width
        let symmetricMax: Float = bitWidth == .int4 ? 7.0 : 127.0
        let asymmetricMax: Float = bitWidth == .int4 ? 15.0 : 255.0

        switch type {
        case .symmetric, .perChannel, .dynamic:
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = max(absMax / symmetricMax, 1e-8)  // Avoid division by zero
            return (scale, nil)

        case .asymmetric:
            let range = maxVal - minVal
            let scale = max(range / asymmetricMax, 1e-8)
            let zeroPoint = Int8(clamping: Int(-round(minVal / scale)))
            return (scale, zeroPoint)
        }
    }
}
