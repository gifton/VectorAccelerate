//
//  BinaryQuantizationKernel.swift
//  VectorAccelerate
//
//  Metal 4 Binary Quantization kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 3, Priority 3
//
//  Features:
//  - Float to binary bit-packed conversion
//  - Hamming distance computation
//  - 32:1 compression ratio
//  - Efficient popcount-based distance

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Binary Vector Types

/// Bit-packed binary vector representation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BinaryVector: Sendable {
    /// Bit-packed data (32 bits per UInt32 word)
    public let data: [UInt32]
    /// Original vector dimension
    public let dimension: Int
    /// Number of UInt32 words needed
    public let numWords: Int

    public init(data: [UInt32], dimension: Int) {
        self.data = data
        self.dimension = dimension
        self.numWords = (dimension + 31) / 32
    }

    /// Extract as bool array (LSB first ordering).
    public func asBoolArray() -> [Bool] {
        var bools = [Bool](repeating: false, count: dimension)
        for i in 0..<dimension {
            let wordIndex = i / 32
            let bitIndex = i % 32
            if wordIndex < data.count && (data[wordIndex] & (1 << bitIndex)) != 0 {
                bools[i] = true
            }
        }
        return bools
    }

    /// Get bit at specific index.
    public func bit(at index: Int) -> Bool {
        guard index >= 0 && index < dimension else { return false }
        let wordIndex = index / 32
        let bitIndex = index % 32
        guard wordIndex < data.count else { return false }
        return (data[wordIndex] & (1 << bitIndex)) != 0
    }

    /// Convert to float array (0.0 or 1.0).
    public func asFloatArray() -> [Float] {
        asBoolArray().map { $0 ? 1.0 : 0.0 }
    }

    /// CPU Hamming distance to another vector.
    public func hammingDistance(to other: Metal4BinaryVector) -> Int {
        guard dimension == other.dimension else { return -1 }
        var distance = 0
        for i in 0..<numWords {
            let word1 = i < data.count ? data[i] : 0
            let word2 = i < other.data.count ? other.data[i] : 0
            distance += (word1 ^ word2).nonzeroBitCount
        }
        return distance
    }
}

/// Batch of binary vectors for efficient GPU operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BinaryVectorBatch: Sendable {
    public let vectors: [Metal4BinaryVector]
    public let dimension: Int
    public let numWords: Int

    public init(vectors: [Metal4BinaryVector]) throws {
        guard let first = vectors.first else {
            throw VectorError.invalidInput("Cannot create empty batch")
        }
        guard vectors.allSatisfy({ $0.dimension == first.dimension }) else {
            throw VectorError.invalidInput("All vectors must have same dimension")
        }
        self.vectors = vectors
        self.dimension = first.dimension
        self.numWords = first.numWords
    }

    public var count: Int { vectors.count }

    public func vector(at index: Int) -> Metal4BinaryVector? {
        guard index >= 0 && index < vectors.count else { return nil }
        return vectors[index]
    }
}

// MARK: - Configuration

/// Configuration for binary quantization.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BinaryQuantizationConfig: Sendable {
    /// Threshold for binary conversion (values > threshold → 1)
    public let threshold: Float
    /// Use sign bit instead of threshold
    public let useSignBit: Bool
    /// Normalize input vectors before quantization
    public let normalizeInput: Bool

    public init(
        threshold: Float = 0.0,
        useSignBit: Bool = false,
        normalizeInput: Bool = false
    ) {
        self.threshold = threshold
        self.useSignBit = useSignBit
        self.normalizeInput = normalizeInput
    }

    public static let `default` = Metal4BinaryQuantizationConfig()
}

// MARK: - Result Types

/// Result from binary quantization.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BinaryQuantizationResult: Sendable {
    public let binaryVectors: Metal4BinaryVectorBatch
    public let compressionRatio: Float
    public let originalBytes: Int
    public let compressedBytes: Int

    /// Bits per original float (always 1 for binary)
    public var bitsPerFloat: Float {
        Float(compressedBytes * 8) / Float(originalBytes / MemoryLayout<Float>.stride)
    }

    /// Space savings percentage
    public var spaceSavings: Float {
        compressionRatio > 0 ? (1.0 - 1.0 / compressionRatio) * 100.0 : 0.0
    }
}

/// Result from Hamming distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HammingDistanceResult: Sendable {
    public let distances: [Float]
    public let minDistance: Float
    public let maxDistance: Float
    public let meanDistance: Float

    public var standardDeviation: Float {
        guard !distances.isEmpty else { return 0 }
        let variance = distances.reduce(0.0) { sum, distance in
            let diff = distance - meanDistance
            return sum + diff * diff
        } / Float(distances.count)
        return sqrt(variance)
    }

    public var integerDistances: [Int] {
        distances.map { Int($0) }
    }
}

// MARK: - Parameters

/// Parameters for binary quantization kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct Metal4BinaryQuantizationParams: Sendable {
    var dimension: UInt32
    var numVectors: UInt32
    var useSignBit: UInt32
    var padding: UInt32 = 0
}

/// Parameters for Hamming distance kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct Metal4HammingParams: Sendable {
    var numWords: UInt32
    var numCandidates: UInt32
}

// MARK: - Kernel Implementation

/// Metal 4 Binary Quantization kernel.
///
/// Converts float vectors to binary (1-bit) representation with bit packing.
/// Achieves 32:1 compression ratio (float32 → 1 bit per dimension).
///
/// ## Binary Conversion
///
/// Each float is converted to a single bit:
/// - **Sign bit mode**: `bit = (x >= 0) ? 1 : 0`
/// - **Threshold mode**: `bit = (x > threshold) ? 1 : 0`
///
/// ## Hamming Distance
///
/// Binary vectors enable ultra-fast Hamming distance via XOR + popcount:
/// ```
/// distance = popcount(a XOR b)
/// ```
///
/// ## Usage
///
/// ```swift
/// let kernel = try await BinaryQuantizationKernel(context: context)
///
/// // Quantize vectors to binary
/// let result = try await kernel.quantize(vectors: floatVectors)
///
/// // Compute Hamming distances
/// let distances = try await kernel.computeHammingDistances(
///     query: result.binaryVectors.vectors[0],
///     candidates: result.binaryVectors
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class BinaryQuantizationKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "BinaryQuantizationKernel"

    // MARK: - Pipelines

    private let quantizePipeline: any MTLComputePipelineState
    private let hammingPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Binary Quantization kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let quantizeFunc = library.makeFunction(name: "binaryQuantize") else {
            throw VectorError.shaderNotFound(
                name: "Binary quantize kernel. Ensure QuantizationShaders.metal is compiled."
            )
        }

        guard let hammingFunc = library.makeFunction(name: "binaryHammingDistance") else {
            throw VectorError.shaderNotFound(
                name: "Binary Hamming distance kernel. Ensure QuantizationShaders.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.quantizePipeline = try await device.makeComputePipelineState(function: quantizeFunc)
        self.hammingPipeline = try await device.makeComputePipelineState(function: hammingFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode binary quantization into an existing encoder.
    @discardableResult
    public func encodeQuantize(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        dimension: Int,
        numVectors: Int,
        config: Metal4BinaryQuantizationConfig
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(quantizePipeline)
        encoder.label = "BinaryQuantize"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        var params = SIMD4<UInt32>(
            UInt32(dimension),
            UInt32(numVectors),
            config.useSignBit ? 1 : 0,
            0
        )
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)

        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 3)

        // One thread per vector
        let config2 = Metal4ThreadConfiguration.linear(count: numVectors, pipeline: quantizePipeline)
        encoder.dispatchThreadgroups(config2.threadgroups, threadsPerThreadgroup: config2.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "binaryQuantize",
            threadgroups: config2.threadgroups,
            threadsPerThreadgroup: config2.threadsPerThreadgroup
        )
    }

    /// Encode Hamming distance computation into an existing encoder.
    @discardableResult
    public func encodeHammingDistance(
        into encoder: any MTLComputeCommandEncoder,
        query: any MTLBuffer,
        candidates: any MTLBuffer,
        output: any MTLBuffer,
        numWords: Int,
        numCandidates: Int
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(hammingPipeline)
        encoder.label = "BinaryHammingDistance"

        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(candidates, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = SIMD2<UInt32>(UInt32(numWords), UInt32(numCandidates))
        encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)

        // One thread per candidate
        let config = Metal4ThreadConfiguration.linear(count: numCandidates, pipeline: hammingPipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "binaryHammingDistance",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - High-Level API

    /// Quantize float vectors to binary.
    public func quantize(
        vectors: [[Float]],
        config: Metal4BinaryQuantizationConfig = .default
    ) async throws -> Metal4BinaryQuantizationResult {
        guard !vectors.isEmpty, let dimension = vectors.first?.count, dimension > 0 else {
            throw VectorError.invalidInput("Input vectors cannot be empty")
        }
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("All vectors must have same dimension")
        }

        let device = context.device.rawDevice
        let numVectors = vectors.count
        let numWords = (dimension + 31) / 32
        let originalBytes = numVectors * dimension * MemoryLayout<Float>.stride
        let compressedBytes = numVectors * numWords * MemoryLayout<UInt32>.stride

        let flatVectors = vectors.flatMap { $0 }
        guard let inputBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: originalBytes,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: originalBytes)
        }
        inputBuffer.label = "BinaryQuantize.input"

        guard let outputBuffer = device.makeBuffer(
            length: compressedBytes,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: compressedBytes)
        }
        outputBuffer.label = "BinaryQuantize.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encodeQuantize(
                into: encoder,
                input: inputBuffer,
                output: outputBuffer,
                dimension: dimension,
                numVectors: numVectors,
                config: config
            )
        }

        // Extract results
        let resultPointer = outputBuffer.contents().bindMemory(to: UInt32.self, capacity: numVectors * numWords)
        var binaryVectors: [Metal4BinaryVector] = []
        binaryVectors.reserveCapacity(numVectors)

        for i in 0..<numVectors {
            let start = i * numWords
            let vectorData = Array(UnsafeBufferPointer(start: resultPointer + start, count: numWords))
            binaryVectors.append(Metal4BinaryVector(data: vectorData, dimension: dimension))
        }

        let batch = try Metal4BinaryVectorBatch(vectors: binaryVectors)
        let compressionRatio = Float(originalBytes) / Float(compressedBytes)

        return Metal4BinaryQuantizationResult(
            binaryVectors: batch,
            compressionRatio: compressionRatio,
            originalBytes: originalBytes,
            compressedBytes: compressedBytes
        )
    }

    /// Quantize single vector.
    public func quantize(
        vector: [Float],
        config: Metal4BinaryQuantizationConfig = .default
    ) async throws -> Metal4BinaryVector {
        let result = try await quantize(vectors: [vector], config: config)
        guard let first = result.binaryVectors.vector(at: 0) else {
            throw VectorError.invalidInput("Failed to quantize vector")
        }
        return first
    }

    /// Compute Hamming distances (1 vs N).
    public func computeHammingDistances(
        query: Metal4BinaryVector,
        candidates: Metal4BinaryVectorBatch
    ) async throws -> Metal4HammingDistanceResult {
        guard query.dimension == candidates.dimension else {
            throw VectorError.invalidInput("Query and candidate dimensions must match")
        }

        let numCandidates = candidates.count
        guard numCandidates > 0 else {
            return Metal4HammingDistanceResult(
                distances: [],
                minDistance: 0,
                maxDistance: 0,
                meanDistance: 0
            )
        }

        let device = context.device.rawDevice

        guard let queryBuffer = device.makeBuffer(
            bytes: query.data,
            length: query.numWords * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: query.numWords * MemoryLayout<UInt32>.stride)
        }
        queryBuffer.label = "HammingDistance.query"

        let flatCandidates = candidates.vectors.flatMap { $0.data }
        let candidatesSize = flatCandidates.count * MemoryLayout<UInt32>.stride
        guard let candidatesBuffer = device.makeBuffer(
            bytes: flatCandidates,
            length: candidatesSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: candidatesSize)
        }
        candidatesBuffer.label = "HammingDistance.candidates"

        let resultSize = numCandidates * MemoryLayout<Float>.stride
        guard let resultBuffer = device.makeBuffer(
            length: resultSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: resultSize)
        }
        resultBuffer.label = "HammingDistance.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encodeHammingDistance(
                into: encoder,
                query: queryBuffer,
                candidates: candidatesBuffer,
                output: resultBuffer,
                numWords: query.numWords,
                numCandidates: numCandidates
            )
        }

        // Extract results
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: numCandidates)
        let distances = Array(UnsafeBufferPointer(start: resultPointer, count: numCandidates))

        // Compute statistics
        let minVal = distances.min() ?? 0
        let maxVal = distances.max() ?? 0
        let meanVal = distances.reduce(0, +) / Float(numCandidates)

        return Metal4HammingDistanceResult(
            distances: distances,
            minDistance: minVal,
            maxDistance: maxVal,
            meanDistance: meanVal
        )
    }

    /// Quantize using VectorProtocol types.
    public func quantize<V: VectorProtocol>(
        vectors: [V],
        config: Metal4BinaryQuantizationConfig = .default
    ) async throws -> Metal4BinaryQuantizationResult where V.Scalar == Float {
        let floatVectors = vectors.map { v in
            v.withUnsafeBufferPointer { Array($0) }
        }
        return try await quantize(vectors: floatVectors, config: config)
    }

    // MARK: - Utility

    /// Calculate compression ratio for given dimension.
    public static func compressionRatio(for dimension: Int) -> Float {
        let originalSize = Float(dimension * MemoryLayout<Float>.stride)
        let numWords = (dimension + 31) / 32
        let compressedSize = Float(numWords * MemoryLayout<UInt32>.stride)
        return compressedSize > 0 ? originalSize / compressedSize : 0
    }
}
