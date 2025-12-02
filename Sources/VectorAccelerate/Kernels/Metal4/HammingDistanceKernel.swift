//
//  HammingDistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 Hamming Distance kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 6a, Priority 5
//
//  Features:
//  - Bit-packed binary vector support
//  - Float vector binarization
//  - Normalized distance output option
//  - Pairs with BinaryQuantization for binary vector search

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for Hamming distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HammingConfig: Sendable {
    /// Return normalized distance (0.0-1.0)
    public let normalized: Bool
    /// Threshold for binarizing float vectors
    public let binarizationThreshold: Float

    public init(
        normalized: Bool = false,
        binarizationThreshold: Float = 0.0
    ) {
        self.normalized = normalized
        self.binarizationThreshold = binarizationThreshold
    }

    public static let `default` = Metal4HammingConfig()
    public static let normalizedDefault = Metal4HammingConfig(normalized: true)
}

// MARK: - Result Types

/// Result from Hamming distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4HammingResult: Sendable {
    /// Distance buffer
    public let distances: any MTLBuffer
    /// Number of query vectors
    public let queryCount: Int
    /// Number of dataset vectors
    public let datasetCount: Int
    /// Whether distances are normalized
    public let normalized: Bool
    /// Execution time
    public let executionTime: TimeInterval

    /// Extract distance matrix as 2D array.
    public func asMatrix() -> [[Float]] {
        var matrix: [[Float]] = []
        matrix.reserveCapacity(queryCount)

        if normalized {
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * datasetCount)
            for q in 0..<queryCount {
                var row: [Float] = []
                row.reserveCapacity(datasetCount)
                for n in 0..<datasetCount {
                    row.append(ptr[q * datasetCount + n])
                }
                matrix.append(row)
            }
        } else {
            let ptr = distances.contents().bindMemory(to: UInt32.self, capacity: queryCount * datasetCount)
            for q in 0..<queryCount {
                var row: [Float] = []
                row.reserveCapacity(datasetCount)
                for n in 0..<datasetCount {
                    row.append(Float(ptr[q * datasetCount + n]))
                }
                matrix.append(row)
            }
        }

        return matrix
    }

    /// Get distance for specific query-dataset pair.
    public func distance(query: Int, dataset: Int) -> Float {
        guard query < queryCount && dataset < datasetCount else { return Float.infinity }

        if normalized {
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * datasetCount)
            return ptr[query * datasetCount + dataset]
        } else {
            let ptr = distances.contents().bindMemory(to: UInt32.self, capacity: queryCount * datasetCount)
            return Float(ptr[query * datasetCount + dataset])
        }
    }

    /// Get raw integer distance (for non-normalized results).
    public func rawDistance(query: Int, dataset: Int) -> Int {
        guard query < queryCount && dataset < datasetCount else { return Int.max }

        if normalized {
            // Convert back to integer estimate
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * datasetCount)
            return Int(ptr[query * datasetCount + dataset])
        } else {
            let ptr = distances.contents().bindMemory(to: UInt32.self, capacity: queryCount * datasetCount)
            return Int(ptr[query * datasetCount + dataset])
        }
    }

    /// Find k nearest neighbors for each query.
    public func nearestNeighbors(k: Int) -> [[(index: Int, distance: Float)]] {
        let matrix = asMatrix()
        return matrix.map { row in
            let indexed = row.enumerated().map { (index: $0.offset, distance: $0.element) }
            return Array(indexed.sorted { $0.distance < $1.distance }.prefix(k))
        }
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Hamming Distance kernel.
///
/// Computes Hamming distance between binary vectors - the number of positions
/// where corresponding bits differ.
///
/// ## Binary Representations
///
/// - **Bit-packed**: UInt32 arrays where each bit represents a dimension
/// - **Float with binarization**: Float vectors converted to binary using threshold
///
/// ## Integration with BinaryQuantization
///
/// ```swift
/// // Quantize float vectors to binary
/// let binaryVectors = try await binaryQuantKernel.quantize(floatVectors)
///
/// // Compute Hamming distances on binary vectors
/// let distances = try await hammingKernel.computeBitPacked(
///     queries: binaryQueries,
///     dataset: binaryVectors,
///     bitsPerVector: dimension
/// )
/// ```
///
/// ## Usage
///
/// ```swift
/// let kernel = try await HammingDistanceKernel(context: context)
///
/// // Binary arrays
/// let dist = try await kernel.distance([true, false, true], [true, true, true])
///
/// // Float vectors with binarization
/// let result = try await kernel.computeFloat(queries, dataset, dimension: 768)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class HammingDistanceKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HammingDistanceKernel"
    public let fusibleWith: [String] = ["BinaryQuantization", "TopKSelection"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let batchPipeline: any MTLComputePipelineState
    private let floatPipeline: any MTLComputePipelineState
    private let singlePipeline: any MTLComputePipelineState
    private let normalizedPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Hamming Distance kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let batchFunc = library.makeFunction(name: "hamming_distance_batch"),
              let floatFunc = library.makeFunction(name: "hamming_distance_float"),
              let singleFunc = library.makeFunction(name: "hamming_distance_single"),
              let normalizedFunc = library.makeFunction(name: "hamming_distance_normalized") else {
            throw VectorError.shaderNotFound(
                name: "Hamming distance kernels. Ensure HammingDistance.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.batchPipeline = try await device.makeComputePipelineState(function: batchFunc)
        self.floatPipeline = try await device.makeComputePipelineState(function: floatFunc)
        self.singlePipeline = try await device.makeComputePipelineState(function: singleFunc)
        self.normalizedPipeline = try await device.makeComputePipelineState(function: normalizedFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API (Bit-Packed)

    /// Encode bit-packed Hamming distance computation into an existing encoder.
    @discardableResult
    public func encodeBitPacked(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        output: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        wordsPerVector: Int,
        bitsPerVector: Int,
        normalized: Bool
    ) -> Metal4EncodingResult {
        let pipeline = normalized ? normalizedPipeline : batchPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.label = "HammingDistance_BitPacked\(normalized ? "_Normalized" : "")"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        // Pass parameters individually to match Metal kernel signature
        var q = UInt32(queryCount)
        var n = UInt32(datasetCount)
        var dWords = UInt32(wordsPerVector)
        encoder.setBytes(&q, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dWords, length: MemoryLayout<UInt32>.size, index: 5)

        // Normalized kernel also needs D_bits at index 6
        if normalized {
            var dBits = UInt32(bitsPerVector)
            encoder.setBytes(&dBits, length: MemoryLayout<UInt32>.size, index: 6)
        }

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (datasetCount + 15) / 16,
            height: (queryCount + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: normalized ? "hamming_distance_normalized" : "hamming_distance_batch",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Encode API (Float)

    /// Encode float-based Hamming distance computation into an existing encoder.
    @discardableResult
    public func encodeFloat(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        output: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        threshold: Float
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(floatPipeline)
        encoder.label = "HammingDistance_Float"

        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        struct FloatParams {
            var Q: UInt32
            var N: UInt32
            var D: UInt32
            var threshold: Float
        }
        var params = FloatParams(
            Q: UInt32(queryCount),
            N: UInt32(datasetCount),
            D: UInt32(dimension),
            threshold: threshold
        )
        encoder.setBytes(&params, length: MemoryLayout<FloatParams>.stride, index: 3)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (datasetCount + 15) / 16,
            height: (queryCount + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "hamming_distance_float",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API (Bit-Packed)

    /// Compute Hamming distance matrix for bit-packed binary vectors.
    public func computeBitPacked(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        bitsPerVector: Int,
        config: Metal4HammingConfig = .default
    ) async throws -> Metal4HammingResult {
        let wordsPerVector = (bitsPerVector + 31) / 32
        let device = context.device.rawDevice

        let outputBuffer: any MTLBuffer
        if config.normalized {
            let size = queryCount * datasetCount * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: size)
            }
            outputBuffer = buffer
        } else {
            let size = queryCount * datasetCount * MemoryLayout<UInt32>.size
            guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: size)
            }
            outputBuffer = buffer
        }
        outputBuffer.label = "HammingDistance.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeBitPacked(
                into: encoder,
                queries: queries,
                dataset: dataset,
                output: outputBuffer,
                queryCount: queryCount,
                datasetCount: datasetCount,
                wordsPerVector: wordsPerVector,
                bitsPerVector: bitsPerVector,
                normalized: config.normalized
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        return Metal4HammingResult(
            distances: outputBuffer,
            queryCount: queryCount,
            datasetCount: datasetCount,
            normalized: config.normalized,
            executionTime: executionTime
        )
    }

    // MARK: - Execute API (Float)

    /// Compute Hamming distance matrix for float vectors with binarization.
    public func computeFloat(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        config: Metal4HammingConfig = .default
    ) async throws -> Metal4HammingResult {
        let device = context.device.rawDevice

        let outputSize = queryCount * datasetCount * MemoryLayout<UInt32>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "HammingDistance.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeFloat(
                into: encoder,
                queries: queries,
                dataset: dataset,
                output: outputBuffer,
                queryCount: queryCount,
                datasetCount: datasetCount,
                dimension: dimension,
                threshold: config.binarizationThreshold
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        return Metal4HammingResult(
            distances: outputBuffer,
            queryCount: queryCount,
            datasetCount: datasetCount,
            normalized: false,  // Float kernel always outputs raw counts
            executionTime: executionTime
        )
    }

    // MARK: - High-Level API

    /// Pack binary array into bit-packed format.
    public func packBinary(_ binary: [Bool]) -> [UInt32] {
        let wordsNeeded = (binary.count + 31) / 32
        var packed = Array<UInt32>(repeating: 0, count: wordsNeeded)

        for (i, bit) in binary.enumerated() {
            if bit {
                let wordIndex = i / 32
                let bitIndex = i % 32
                packed[wordIndex] |= (1 << bitIndex)
            }
        }

        return packed
    }

    /// Compute Hamming distance between two binary arrays.
    public func distance(
        _ a: [Bool],
        _ b: [Bool]
    ) async throws -> Int {
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }

        let packedA = packBinary(a)
        let packedB = packBinary(b)
        let device = context.device.rawDevice

        guard let bufferA = device.makeBuffer(
            bytes: packedA,
            length: packedA.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: packedA.count * MemoryLayout<UInt32>.size)
        }

        guard let bufferB = device.makeBuffer(
            bytes: packedB,
            length: packedB.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: packedB.count * MemoryLayout<UInt32>.size)
        }

        guard let outputBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<UInt32>.size)
        }

        // Initialize to 0
        outputBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0

        let wordsCount = packedA.count
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(singlePipeline)
            encoder.label = "HammingDistance_Single"

            encoder.setBuffer(bufferA, offset: 0, index: 0)
            encoder.setBuffer(bufferB, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)

            var words = UInt32(wordsCount)
            encoder.setBytes(&words, length: MemoryLayout<UInt32>.size, index: 3)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let numThreadgroups = (wordsCount + 15) / 16
            let gridSize = MTLSize(width: numThreadgroups, height: 1, depth: 1)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        }

        return Int(outputBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }

    /// Compute Hamming distance matrix for binary arrays.
    public func distanceMatrix(
        queries: [[Bool]],
        dataset: [[Bool]],
        normalized: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input arrays")
        }

        let dimension = queries[0].count
        guard dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in dataset vectors")
        }

        let device = context.device.rawDevice

        // Pack all vectors
        let packedQueries = queries.flatMap { packBinary($0) }
        let packedDataset = dataset.flatMap { packBinary($0) }

        guard let queryBuffer = device.makeBuffer(
            bytes: packedQueries,
            length: packedQueries.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: packedQueries.count * MemoryLayout<UInt32>.size)
        }

        guard let datasetBuffer = device.makeBuffer(
            bytes: packedDataset,
            length: packedDataset.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: packedDataset.count * MemoryLayout<UInt32>.size)
        }

        let config = Metal4HammingConfig(normalized: normalized)
        let result = try await computeBitPacked(
            queries: queryBuffer,
            dataset: datasetBuffer,
            queryCount: queries.count,
            datasetCount: dataset.count,
            bitsPerVector: dimension,
            config: config
        )

        return result.asMatrix()
    }

    /// Compute Hamming distance for float vectors with binarization.
    public func distanceMatrix(
        queries: [[Float]],
        dataset: [[Float]],
        threshold: Float = 0.0
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input arrays")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let device = context.device.rawDevice

        let flatQueries = queries.flatMap { $0 }
        let flatDataset = dataset.flatMap { $0 }

        guard let queryBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }

        guard let datasetBuffer = device.makeBuffer(
            bytes: flatDataset,
            length: flatDataset.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDataset.count * MemoryLayout<Float>.size)
        }

        let config = Metal4HammingConfig(binarizationThreshold: threshold)
        let result = try await computeFloat(
            queries: queryBuffer,
            dataset: datasetBuffer,
            queryCount: queries.count,
            datasetCount: dataset.count,
            dimension: dimension,
            config: config
        )

        return result.asMatrix()
    }

    // MARK: - VectorCore Integration

    /// Compute Hamming distance using VectorCore binary vectors.
    public func distance<V: VectorProtocol>(
        _ a: V,
        _ b: V,
        threshold: Float = 0.0
    ) async throws -> Int where V.Scalar == Float {
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }

        // Binarize directly from vector storage
        let binaryA: [Bool] = a.withUnsafeBufferPointer { ptr in
            ptr.map { $0 > threshold }
        }
        let binaryB: [Bool] = b.withUnsafeBufferPointer { ptr in
            ptr.map { $0 > threshold }
        }

        return try await distance(binaryA, binaryB)
    }
}
