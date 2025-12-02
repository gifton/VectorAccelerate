//
//  L2NormalizationKernel.swift
//  VectorAccelerate
//
//  Metal 4 L2 Normalization kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 5, Priority 6
//
//  Features:
//  - Dimension-specific optimized pipelines (512, 768, 1536)
//  - In-place and out-of-place normalization
//  - Optional norm storage
//  - Fusible with distance kernels via encode() API

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Parameters

/// Parameters for L2 Normalization kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct L2NormalizationParameters: Sendable {
    /// Number of vectors to normalize
    public let numVectors: UInt32
    /// Vector dimension
    public let dimension: UInt32
    /// Input stride between vectors
    public let inputStride: UInt32
    /// Output stride between vectors
    public let outputStride: UInt32
    /// Epsilon for numerical stability
    public let epsilon: Float
    /// Whether to store norms
    public let storeNorms: UInt8
    /// Padding for alignment
    private let padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

    public init(
        numVectors: Int,
        dimension: Int,
        epsilon: Float = 1e-8,
        storeNorms: Bool = false,
        inputStride: Int? = nil,
        outputStride: Int? = nil
    ) {
        self.numVectors = UInt32(numVectors)
        self.dimension = UInt32(dimension)
        self.epsilon = epsilon
        self.storeNorms = storeNorms ? 1 : 0
        self.inputStride = UInt32(inputStride ?? dimension)
        self.outputStride = UInt32(outputStride ?? dimension)
    }
}

// MARK: - Result Type

/// Result from L2 normalization operation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4L2NormalizationResult: Sendable {
    /// Normalized vectors buffer
    public let normalizedVectors: any MTLBuffer
    /// Original norms buffer (optional)
    public let norms: (any MTLBuffer)?
    /// Number of vectors
    public let numVectors: Int
    /// Vector dimension
    public let dimension: Int
    /// Execution time
    public let executionTime: TimeInterval
    /// Throughput in GB/s
    public let throughputGBps: Double

    /// Extract normalized vectors as 2D array.
    public func asArrays() -> [[Float]] {
        let ptr = normalizedVectors.contents().bindMemory(to: Float.self, capacity: numVectors * dimension)
        var results: [[Float]] = []
        results.reserveCapacity(numVectors)
        for i in 0..<numVectors {
            let start = i * dimension
            results.append(Array(UnsafeBufferPointer(start: ptr + start, count: dimension)))
        }
        return results
    }

    /// Extract norms as array.
    public func normsAsArray() -> [Float]? {
        guard let norms = norms else { return nil }
        let ptr = norms.contents().bindMemory(to: Float.self, capacity: numVectors)
        return Array(UnsafeBufferPointer(start: ptr, count: numVectors))
    }
}

// MARK: - Kernel Implementation

/// Metal 4 L2 Normalization kernel.
///
/// Normalizes vectors to unit length using the L2 (Euclidean) norm:
/// ```
/// v_normalized = v / ||v||₂
/// ```
///
/// ## Dimension Optimizations
///
/// Provides optimized pipelines for common embedding dimensions:
/// - 512: Common for smaller models
/// - 768: BERT, DistilBERT
/// - 1536: OpenAI text-embedding-3-large
///
/// ## Fusion Pattern
///
/// Often used before cosine similarity to convert dot product to cosine:
/// ```swift
/// try await context.executeAndWait { _, encoder in
///     normKernel.encode(into: encoder, ...)
///     encoder.memoryBarrier(scope: .buffers)
///     cosineKernel.encode(into: encoder, ..., inputsNormalized: true)
/// }
/// ```
///
/// ## Usage
///
/// ```swift
/// let kernel = try await L2NormalizationKernel(context: context)
///
/// // Normalize and get norms
/// let (normalized, norms) = try await kernel.normalize(vectors, storeNorms: true)
///
/// // In-place normalization
/// try await kernel.normalizeInPlace(&vectors)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class L2NormalizationKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "L2NormalizationKernel"
    public let fusibleWith: [String] = ["CosineSimilarity", "DotProduct", "Distance"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let pipelineGeneral: any MTLComputePipelineState
    private let pipeline512: any MTLComputePipelineState
    private let pipeline768: any MTLComputePipelineState
    private let pipeline1536: any MTLComputePipelineState
    private let pipelineInplace: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 L2 Normalization kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let generalFunc = library.makeFunction(name: "l2_normalize_general_kernel"),
              let func512 = library.makeFunction(name: "l2_normalize_512_kernel"),
              let func768 = library.makeFunction(name: "l2_normalize_768_kernel"),
              let func1536 = library.makeFunction(name: "l2_normalize_1536_kernel"),
              let inplaceFunc = library.makeFunction(name: "l2_normalize_inplace_kernel") else {
            throw VectorError.shaderNotFound(
                name: "L2 normalization kernels. Ensure L2Normalization.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipelineGeneral = try await device.makeComputePipelineState(function: generalFunc)
        self.pipeline512 = try await device.makeComputePipelineState(function: func512)
        self.pipeline768 = try await device.makeComputePipelineState(function: func768)
        self.pipeline1536 = try await device.makeComputePipelineState(function: func1536)
        self.pipelineInplace = try await device.makeComputePipelineState(function: inplaceFunc)
    }

    // MARK: - Pipeline Selection

    private func selectPipeline(for dimension: UInt32, inputStride: UInt32, outputStride: UInt32) -> any MTLComputePipelineState {
        // Optimized pipelines require dense packing
        guard inputStride == dimension && outputStride == dimension else {
            return pipelineGeneral
        }

        switch dimension {
        case 512: return pipeline512
        case 768: return pipeline768
        case 1536: return pipeline1536
        default: return pipelineGeneral
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode L2 normalization into an existing encoder (out-of-place).
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        norms: (any MTLBuffer)?,
        parameters: L2NormalizationParameters
    ) -> Metal4EncodingResult {
        let pipeline = selectPipeline(
            for: parameters.dimension,
            inputStride: parameters.inputStride,
            outputStride: parameters.outputStride
        )

        encoder.setComputePipelineState(pipeline)
        encoder.label = "L2Normalize (dim=\(parameters.dimension))"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(norms, offset: 0, index: 2)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<L2NormalizationParameters>.stride, index: 3)

        // One thread per vector
        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numVectors),
            pipeline: pipeline
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "l2_normalize_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    /// Encode in-place L2 normalization into an existing encoder.
    @discardableResult
    public func encodeInPlace(
        into encoder: any MTLComputeCommandEncoder,
        vectors: any MTLBuffer,
        norms: (any MTLBuffer)?,
        parameters: L2NormalizationParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipelineInplace)
        encoder.label = "L2NormalizeInPlace (dim=\(parameters.dimension))"

        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(norms, offset: 0, index: 1)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<L2NormalizationParameters>.stride, index: 2)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numVectors),
            pipeline: pipelineInplace
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "l2_normalize_inplace_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute L2 normalization as standalone operation.
    public func execute(
        input: any MTLBuffer,
        parameters: L2NormalizationParameters
    ) async throws -> Metal4L2NormalizationResult {
        let device = context.device.rawDevice
        let numVectors = Int(parameters.numVectors)
        let dimension = Int(parameters.dimension)

        let outputSize = numVectors * dimension * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "L2Normalize.output"

        let normsBuffer: (any MTLBuffer)?
        if parameters.storeNorms == 1 {
            let normsSize = numVectors * MemoryLayout<Float>.size
            normsBuffer = device.makeBuffer(length: normsSize, options: .storageModeShared)
            normsBuffer?.label = "L2Normalize.norms"
        } else {
            normsBuffer = nil
        }

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                input: input,
                output: outputBuffer,
                norms: normsBuffer,
                parameters: parameters
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        let bytesProcessed = Double(numVectors * dimension * 2 * MemoryLayout<Float>.size)
        let throughputGBps = (bytesProcessed / 1e9) / executionTime

        return Metal4L2NormalizationResult(
            normalizedVectors: outputBuffer,
            norms: normsBuffer,
            numVectors: numVectors,
            dimension: dimension,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    // MARK: - High-Level API

    /// Normalize vectors to unit length.
    public func normalize(
        _ vectors: [[Float]],
        storeNorms: Bool = false,
        epsilon: Float = 1e-8
    ) async throws -> Metal4L2NormalizationResult {
        guard !vectors.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let numVectors = vectors.count
        let dimension = vectors[0].count

        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let device = context.device.rawDevice
        let flatVectors = vectors.flatMap { $0 }

        guard let inputBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: flatVectors.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatVectors.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "L2Normalize.input"

        let parameters = L2NormalizationParameters(
            numVectors: numVectors,
            dimension: dimension,
            epsilon: epsilon,
            storeNorms: storeNorms
        )

        return try await execute(input: inputBuffer, parameters: parameters)
    }

    /// Normalize using VectorProtocol types.
    public func normalize<V: VectorProtocol>(
        _ vectors: [V],
        storeNorms: Bool = false,
        epsilon: Float = 1e-8
    ) async throws -> Metal4L2NormalizationResult where V.Scalar == Float {
        guard !vectors.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let floatVectors = vectors.map { v in
            v.withUnsafeBufferPointer { Array($0) }
        }

        return try await normalize(floatVectors, storeNorms: storeNorms, epsilon: epsilon)
    }

    /// Normalize single vector.
    public func normalizeSingle(
        _ vector: [Float],
        epsilon: Float = 1e-8
    ) async throws -> (normalized: [Float], norm: Float) {
        let result = try await normalize([vector], storeNorms: true, epsilon: epsilon)
        let normalized = result.asArrays().first ?? []
        let norm = result.normsAsArray()?.first ?? 0
        return (normalized, norm)
    }
}
