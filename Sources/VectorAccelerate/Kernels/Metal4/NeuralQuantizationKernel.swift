//
//  NeuralQuantizationKernel.swift
//  VectorAccelerate
//
//  Metal 4 Neural Quantization kernel with learned encoder/decoder.
//
//  Phase 4: ML Integration - Neural Quantization (P3)
//
//  Features:
//  - Learned encoder for dimensionality reduction to latent space
//  - INT8 quantization in latent space (better fidelity than raw quantization)
//  - Learned decoder for reconstruction
//  - Supports multiple encoder architectures (linear, ReLU-activated)
//  - TensorManager integration for weight loading/management
//
//  Key Advantages over Scalar Quantization:
//  - Learns data-specific compression that preserves semantic structure
//  - Latent space optimized for quantization (trained end-to-end)
//  - Better reconstruction quality for complex data distributions

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for neural quantization.
public struct Metal4NeuralQuantizationConfig: Sendable {
    /// Input dimension of vectors
    public let inputDimension: Int
    /// Latent dimension (typically smaller than input)
    public let latentDimension: Int
    /// Whether to use ReLU activation after encoding
    public let useActivation: Bool
    /// Whether to normalize latent vectors before quantization
    public let normalizeLatent: Bool

    public init(
        inputDimension: Int,
        latentDimension: Int,
        useActivation: Bool = true,
        normalizeLatent: Bool = false
    ) {
        self.inputDimension = inputDimension
        self.latentDimension = latentDimension
        self.useActivation = useActivation
        self.normalizeLatent = normalizeLatent
    }

    /// Common configuration: 768 -> 64 (high compression)
    public static func highCompression(inputDim: Int = 768) -> Metal4NeuralQuantizationConfig {
        Metal4NeuralQuantizationConfig(
            inputDimension: inputDim,
            latentDimension: 64,
            useActivation: true,
            normalizeLatent: true
        )
    }

    /// Common configuration: 768 -> 128 (balanced)
    public static func balanced(inputDim: Int = 768) -> Metal4NeuralQuantizationConfig {
        Metal4NeuralQuantizationConfig(
            inputDimension: inputDim,
            latentDimension: 128,
            useActivation: true,
            normalizeLatent: false
        )
    }

    /// Common configuration: 384 -> 64 (for smaller models like MiniLM)
    public static func miniLM() -> Metal4NeuralQuantizationConfig {
        Metal4NeuralQuantizationConfig(
            inputDimension: 384,
            latentDimension: 64,
            useActivation: true,
            normalizeLatent: false
        )
    }

    /// Compression ratio achieved
    public var compressionRatio: Float {
        Float(inputDimension * MemoryLayout<Float>.size) /
        Float(latentDimension * MemoryLayout<Int8>.size)
    }
}

// MARK: - Parameters

/// Parameters for neural quantization kernel (matches Metal struct).
public struct NeuralQuantizationParameters: Sendable {
    public var numVectors: UInt32
    public var inputDimension: UInt32
    public var latentDimension: UInt32
    public var stride: UInt32
    public var useActivation: UInt8
    public var normalizeLatent: UInt8
    private var padding: (UInt8, UInt8) = (0, 0)

    public init(
        numVectors: Int,
        config: Metal4NeuralQuantizationConfig
    ) {
        self.numVectors = UInt32(numVectors)
        self.inputDimension = UInt32(config.inputDimension)
        self.latentDimension = UInt32(config.latentDimension)
        self.stride = UInt32(config.inputDimension)
        self.useActivation = config.useActivation ? 1 : 0
        self.normalizeLatent = config.normalizeLatent ? 1 : 0
    }
}

// MARK: - Result Types

/// Result from neural encoding.
public struct Metal4NeuralEncodingResult: Sendable {
    /// Encoded latent vectors as INT8
    public let latentCodes: Data
    /// Number of vectors encoded
    public let numVectors: Int
    /// Latent dimension per vector
    public let latentDimension: Int
    /// Scale factor used for quantization
    public let scale: Float
    /// Encoding time
    public let encodingTime: TimeInterval

    /// Bytes per vector in encoded form
    public var bytesPerVector: Int { latentDimension }

    /// Total compressed size
    public var compressedSize: Int { latentCodes.count }
}

/// Metrics for neural quantization quality.
public struct Metal4NeuralQuantizationMetrics: Sendable {
    /// Mean Squared Error in reconstruction
    public let mse: Float
    /// Maximum absolute error
    public let maxError: Float
    /// Cosine similarity between original and reconstructed
    public let cosineSimilarity: Float
    /// Compression ratio achieved
    public let compressionRatio: Float
    /// Encoding throughput (vectors/second)
    public let encodingThroughput: Double
    /// Decoding throughput (vectors/second)
    public let decodingThroughput: Double

    /// Signal-to-Noise Ratio in dB
    public var snr: Float {
        mse > 0 ? -10 * log10(mse) : Float.infinity
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Neural Quantization kernel with learned encoder/decoder.
///
/// Neural quantization uses learned transformations to achieve better compression
/// quality than scalar or product quantization. The encoder projects vectors to a
/// lower-dimensional latent space optimized for quantization, while the decoder
/// reconstructs vectors from quantized latent codes.
///
/// ## Architecture
///
/// ```
/// Input (D) → Encoder (D→L) → [ReLU] → Quantize → Latent (L, INT8)
/// Latent (L, INT8) → Dequantize → Decoder (L→D) → Output (D)
/// ```
///
/// ## Weight Format
///
/// - Encoder weights: [latentDim, inputDim] row-major float32
/// - Decoder weights: [inputDim, latentDim] row-major float32
/// - Optional bias terms: [latentDim] and [inputDim]
///
/// ## Usage
///
/// ```swift
/// let kernel = try await NeuralQuantizationKernel(context: context)
///
/// // Load pre-trained weights
/// try await kernel.loadWeights(
///     encoderURL: encoderWeightsURL,
///     decoderURL: decoderWeightsURL,
///     config: .balanced(inputDim: 768)
/// )
///
/// // Encode vectors to compact latent representation
/// let encoded = try await kernel.encode(vectors)
///
/// // Decode back to full vectors
/// let reconstructed = try await kernel.decode(encoded)
/// ```
///
/// ## Training Weights
///
/// Encoder/decoder weights should be trained with a reconstruction objective,
/// optionally with quantization-aware training for best results.
public final class NeuralQuantizationKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "NeuralQuantizationKernel"

    // MARK: - Pipelines

    private let encodePipeline: any MTLComputePipelineState
    private let decodePipeline: any MTLComputePipelineState
    private let encodeQuantizePipeline: any MTLComputePipelineState
    private let dequantizeDecodePipeline: any MTLComputePipelineState

    // MARK: - Weight Management

    private let tensorManager: TensorManager
    private var currentConfig: Metal4NeuralQuantizationConfig?
    private var encoderWeights: TensorBuffer?
    private var decoderWeights: TensorBuffer?
    private var encoderBias: TensorBuffer?
    private var decoderBias: TensorBuffer?

    // MARK: - Initialization

    /// Create a Metal 4 Neural Quantization kernel.
    ///
    /// - Parameter context: Metal 4 context for execution
    /// - Throws: `VectorError` if shader compilation fails
    public init(context: Metal4Context) async throws {
        self.context = context
        self.tensorManager = TensorManager(device: context.device.rawDevice)

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let encodeFunc = library.makeFunction(name: "neural_encode_kernel"),
              let decodeFunc = library.makeFunction(name: "neural_decode_kernel"),
              let encodeQuantizeFunc = library.makeFunction(name: "neural_encode_quantize_kernel"),
              let dequantizeDecodeFunc = library.makeFunction(name: "neural_dequantize_decode_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Neural quantization kernels. Ensure NeuralQuantization.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.encodePipeline = try await device.makeComputePipelineState(function: encodeFunc)
        self.decodePipeline = try await device.makeComputePipelineState(function: decodeFunc)
        self.encodeQuantizePipeline = try await device.makeComputePipelineState(function: encodeQuantizeFunc)
        self.dequantizeDecodePipeline = try await device.makeComputePipelineState(function: dequantizeDecodeFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Weight Loading

    /// Load encoder and decoder weights from files.
    ///
    /// Weight format: binary float32, row-major.
    /// - Encoder: [latentDim, inputDim]
    /// - Decoder: [inputDim, latentDim]
    ///
    /// - Parameters:
    ///   - encoderURL: URL to encoder weight file
    ///   - decoderURL: URL to decoder weight file
    ///   - config: Neural quantization configuration
    public func loadWeights(
        encoderURL: URL,
        decoderURL: URL,
        config: Metal4NeuralQuantizationConfig
    ) async throws {
        let encoderShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: config.latentDimension
        )
        let decoderShape = TensorShape.projection(
            inputDim: config.latentDimension,
            outputDim: config.inputDimension
        )

        encoderWeights = try await tensorManager.loadWeights(
            from: encoderURL,
            name: "neural_encoder",
            shape: encoderShape,
            dataType: .float32
        )

        decoderWeights = try await tensorManager.loadWeights(
            from: decoderURL,
            name: "neural_decoder",
            shape: decoderShape,
            dataType: .float32
        )

        currentConfig = config
    }

    /// Load weights from Data objects.
    public func loadWeights(
        encoderData: Data,
        decoderData: Data,
        config: Metal4NeuralQuantizationConfig
    ) async throws {
        let encoderShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: config.latentDimension
        )
        let decoderShape = TensorShape.projection(
            inputDim: config.latentDimension,
            outputDim: config.inputDimension
        )

        encoderWeights = try await tensorManager.loadWeights(
            from: encoderData,
            name: "neural_encoder",
            shape: encoderShape,
            dataType: .float32
        )

        decoderWeights = try await tensorManager.loadWeights(
            from: decoderData,
            name: "neural_decoder",
            shape: decoderShape,
            dataType: .float32
        )

        currentConfig = config
    }

    /// Load weights from Float arrays.
    ///
    /// - Parameters:
    ///   - encoderWeights: Encoder weights [latentDim * inputDim], row-major
    ///   - decoderWeights: Decoder weights [inputDim * latentDim], row-major
    ///   - config: Neural quantization configuration
    public func loadWeights(
        encoderWeights: [Float],
        decoderWeights: [Float],
        config: Metal4NeuralQuantizationConfig
    ) async throws {
        let encoderShape = TensorShape.projection(
            inputDim: config.inputDimension,
            outputDim: config.latentDimension
        )
        let decoderShape = TensorShape.projection(
            inputDim: config.latentDimension,
            outputDim: config.inputDimension
        )

        self.encoderWeights = try await tensorManager.createTensor(
            from: encoderWeights,
            name: "neural_encoder",
            shape: encoderShape
        )

        self.decoderWeights = try await tensorManager.createTensor(
            from: decoderWeights,
            name: "neural_decoder",
            shape: decoderShape
        )

        currentConfig = config
    }

    /// Create random weights for testing/initialization.
    ///
    /// Uses Xavier/Glorot initialization for proper scaling.
    public func createRandomWeights(
        config: Metal4NeuralQuantizationConfig
    ) async throws {
        encoderWeights = try await tensorManager.createRandomProjection(
            inputDim: config.inputDimension,
            outputDim: config.latentDimension,
            name: "neural_encoder"
        )

        decoderWeights = try await tensorManager.createRandomProjection(
            inputDim: config.latentDimension,
            outputDim: config.inputDimension,
            name: "neural_decoder"
        )

        currentConfig = config
    }

    /// Check if weights are loaded.
    public var hasWeights: Bool {
        encoderWeights != nil && decoderWeights != nil
    }

    /// Unload weights to free memory.
    public func unloadWeights() async {
        await tensorManager.unload(name: "neural_encoder")
        await tensorManager.unload(name: "neural_decoder")
        await tensorManager.unload(name: "encoder_bias")
        await tensorManager.unload(name: "decoder_bias")
        encoderWeights = nil
        decoderWeights = nil
        encoderBias = nil
        decoderBias = nil
        currentConfig = nil
    }

    // MARK: - Encode API

    /// Encode vectors to latent codes (float, not quantized).
    ///
    /// Useful for inspecting latent space before quantization.
    @discardableResult
    public func encodeEncode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        parameters: NeuralQuantizationParameters
    ) throws -> Metal4EncodingResult {
        guard let encoderWeights = encoderWeights else {
            throw VectorError.invalidOperation("Encoder weights not loaded")
        }

        encoder.setComputePipelineState(encodePipeline)
        encoder.label = "NeuralEncode"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(encoderWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(encoderBias?.buffer, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.size, index: 4)

        // 2D dispatch: (numVectors, latentDim)
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numVectors),
            numDatabase: Int(parameters.latentDimension),
            pipeline: encodePipeline
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "neural_encode_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode vectors and quantize to INT8 in one pass.
    @discardableResult
    public func encodeEncodeQuantize(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        scale: any MTLBuffer,
        parameters: NeuralQuantizationParameters
    ) throws -> Metal4EncodingResult {
        guard let encoderWeights = encoderWeights else {
            throw VectorError.invalidOperation("Encoder weights not loaded")
        }

        encoder.setComputePipelineState(encodeQuantizePipeline)
        encoder.label = "NeuralEncodeQuantize"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(encoderWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(scale, offset: 0, index: 3)
        encoder.setBuffer(encoderBias?.buffer, offset: 0, index: 4)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.size, index: 5)

        let numVectors = Int(parameters.numVectors)
        let config = Metal4ThreadConfiguration.linear(count: numVectors, pipeline: encodeQuantizePipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "neural_encode_quantize_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Decode API

    /// Decode latent codes (float) back to full vectors.
    @discardableResult
    public func encodeDecode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        parameters: NeuralQuantizationParameters
    ) throws -> Metal4EncodingResult {
        guard let decoderWeights = decoderWeights else {
            throw VectorError.invalidOperation("Decoder weights not loaded")
        }

        encoder.setComputePipelineState(decodePipeline)
        encoder.label = "NeuralDecode"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(decoderWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(decoderBias?.buffer, offset: 0, index: 3)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.size, index: 4)

        // 2D dispatch: (numVectors, inputDim)
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numVectors),
            numDatabase: Int(parameters.inputDimension),
            pipeline: decodePipeline
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "neural_decode_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Dequantize and decode in one pass.
    @discardableResult
    public func encodeDequantizeDecode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        scale: any MTLBuffer,
        output: any MTLBuffer,
        parameters: NeuralQuantizationParameters
    ) throws -> Metal4EncodingResult {
        guard let decoderWeights = decoderWeights else {
            throw VectorError.invalidOperation("Decoder weights not loaded")
        }

        encoder.setComputePipelineState(dequantizeDecodePipeline)
        encoder.label = "NeuralDequantizeDecode"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(decoderWeights.buffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(decoderBias?.buffer, offset: 0, index: 4)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<NeuralQuantizationParameters>.size, index: 5)

        let numVectors = Int(parameters.numVectors)
        let config = Metal4ThreadConfiguration.linear(count: numVectors, pipeline: dequantizeDecodePipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "neural_dequantize_decode_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - High-Level API

    /// Encode vectors to quantized latent codes.
    ///
    /// - Parameter vectors: Input vectors [N, inputDim]
    /// - Returns: Encoding result with INT8 latent codes
    public func encode(_ vectors: [[Float]]) async throws -> Metal4NeuralEncodingResult {
        guard let config = currentConfig else {
            throw VectorError.invalidOperation("Weights not loaded. Call loadWeights first.")
        }

        guard !vectors.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let inputDim = vectors[0].count
        guard inputDim == config.inputDimension else {
            throw VectorError.countMismatch(expected: config.inputDimension, actual: inputDim)
        }

        let device = context.device.rawDevice
        let numVectors = vectors.count
        let latentDim = config.latentDimension

        // Create input buffer
        let flatInput = vectors.flatMap { $0 }
        guard let inputBuffer = device.makeBuffer(
            bytes: flatInput,
            length: flatInput.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatInput.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "NeuralQuantize.input"

        // Create output buffer (INT8)
        let outputSize = numVectors * latentDim
        guard let outputBuffer = device.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "NeuralQuantize.output"

        // Create scale buffer (one scale per vector)
        guard let scaleBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numVectors * MemoryLayout<Float>.size)
        }
        scaleBuffer.label = "NeuralQuantize.scale"

        let parameters = NeuralQuantizationParameters(
            numVectors: numVectors,
            config: config
        )

        let startTime = CACurrentMediaTime()

        try await context.executeAndWait { [self] _, encoder in
            try self.encodeEncodeQuantize(
                into: encoder,
                input: inputBuffer,
                output: outputBuffer,
                scale: scaleBuffer,
                parameters: parameters
            )
        }

        let encodingTime = CACurrentMediaTime() - startTime

        // Extract results
        let latentCodes = Data(bytes: outputBuffer.contents(), count: outputSize)
        let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        let avgScale = (0..<numVectors).reduce(0.0) { $0 + scalePtr[$1] } / Float(numVectors)

        return Metal4NeuralEncodingResult(
            latentCodes: latentCodes,
            numVectors: numVectors,
            latentDimension: latentDim,
            scale: avgScale,
            encodingTime: encodingTime
        )
    }

    /// Decode quantized latent codes back to full vectors.
    ///
    /// - Parameter encoded: Encoding result from `encode()`
    /// - Returns: Reconstructed vectors [N, inputDim]
    public func decode(_ encoded: Metal4NeuralEncodingResult) async throws -> [[Float]] {
        guard let config = currentConfig else {
            throw VectorError.invalidOperation("Weights not loaded. Call loadWeights first.")
        }

        let device = context.device.rawDevice
        let numVectors = encoded.numVectors
        let inputDim = config.inputDimension

        // Create input buffer (INT8 codes)
        guard let inputBuffer = device.makeBuffer(
            bytes: [UInt8](encoded.latentCodes),
            length: encoded.latentCodes.count,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: encoded.latentCodes.count)
        }
        inputBuffer.label = "NeuralDequantize.input"

        // Create scale buffer
        guard let scaleBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numVectors * MemoryLayout<Float>.size)
        }
        scaleBuffer.label = "NeuralDequantize.scale"

        // Initialize scales (use stored average for now)
        let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
        for i in 0..<numVectors {
            scalePtr[i] = encoded.scale
        }

        // Create output buffer
        let outputSize = numVectors * inputDim * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "NeuralDequantize.output"

        let parameters = NeuralQuantizationParameters(
            numVectors: numVectors,
            config: config
        )

        try await context.executeAndWait { [self] _, encoder in
            try self.encodeDequantizeDecode(
                into: encoder,
                input: inputBuffer,
                scale: scaleBuffer,
                output: outputBuffer,
                parameters: parameters
            )
        }

        // Extract results
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: numVectors * inputDim)
        var results: [[Float]] = []
        results.reserveCapacity(numVectors)
        for i in 0..<numVectors {
            var row: [Float] = []
            row.reserveCapacity(inputDim)
            for j in 0..<inputDim {
                row.append(outputPtr[i * inputDim + j])
            }
            results.append(row)
        }

        return results
    }

    /// Encode and decode with quality metrics.
    ///
    /// - Parameter vectors: Input vectors to quantize
    /// - Returns: Tuple of reconstructed vectors and quality metrics
    public func encodeDecodeWithMetrics(
        _ vectors: [[Float]]
    ) async throws -> (reconstructed: [[Float]], metrics: Metal4NeuralQuantizationMetrics) {
        guard let config = currentConfig else {
            throw VectorError.invalidOperation("Weights not loaded. Call loadWeights first.")
        }

        let encodeStart = CACurrentMediaTime()
        let encoded = try await encode(vectors)
        let encodeTime = CACurrentMediaTime() - encodeStart

        let decodeStart = CACurrentMediaTime()
        let reconstructed = try await decode(encoded)
        let decodeTime = CACurrentMediaTime() - decodeStart

        // Compute metrics
        var mse: Float = 0
        var maxError: Float = 0
        var dotSum: Float = 0
        var normOrigSum: Float = 0
        var normReconSum: Float = 0

        for (orig, recon) in zip(vectors, reconstructed) {
            var dot: Float = 0
            var normOrig: Float = 0
            var normRecon: Float = 0

            for (o, r) in zip(orig, recon) {
                let error = abs(o - r)
                mse += error * error
                maxError = max(maxError, error)

                dot += o * r
                normOrig += o * o
                normRecon += r * r
            }

            dotSum += dot
            normOrigSum += normOrig
            normReconSum += normRecon
        }

        let totalElements = Float(vectors.count * vectors[0].count)
        mse /= totalElements

        let cosineSim = dotSum / (sqrt(normOrigSum) * sqrt(normReconSum) + 1e-8)

        let compressionRatio = config.compressionRatio

        let metrics = Metal4NeuralQuantizationMetrics(
            mse: mse,
            maxError: maxError,
            cosineSimilarity: cosineSim,
            compressionRatio: compressionRatio,
            encodingThroughput: Double(vectors.count) / encodeTime,
            decodingThroughput: Double(vectors.count) / decodeTime
        )

        return (reconstructed, metrics)
    }

    // MARK: - Statistics

    /// Get tensor manager statistics.
    public func getTensorStatistics() async -> TensorManagerStatistics {
        await tensorManager.getStatistics()
    }
}
