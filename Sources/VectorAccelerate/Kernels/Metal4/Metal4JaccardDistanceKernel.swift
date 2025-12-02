//
//  Metal4JaccardDistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 Jaccard Distance kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 6a, Priority 5
//
//  Features:
//  - Jaccard distance for set similarity
//  - Configurable binarization threshold
//  - Batch and single vector pair modes
//  - Fusible with TopK selection

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for Jaccard distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4JaccardConfig: Sendable {
    /// Threshold for binarizing float vectors
    public let threshold: Float
    /// Optimal batch size for operations
    public let batchSize: Int

    public init(
        threshold: Float = 0.0,
        batchSize: Int = 1024
    ) {
        self.threshold = threshold
        self.batchSize = batchSize
    }

    public static let `default` = Metal4JaccardConfig()
}

// MARK: - Result Types

/// Result from single Jaccard distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4JaccardResult: Sendable {
    /// Jaccard distance (1 - similarity)
    public let distance: Float
    /// Jaccard similarity (intersection / union)
    public let similarity: Float
    /// Size of set intersection
    public let intersectionSize: Int
    /// Size of set union
    public let unionSize: Int
    /// Execution time
    public let executionTime: TimeInterval

    /// Jaccard coefficient (same as similarity)
    public var coefficient: Float { similarity }

    /// Check if vectors are identical
    public var isIdentical: Bool { distance == 0.0 }

    /// Check if vectors are completely disjoint
    public var isDisjoint: Bool { similarity == 0.0 }
}

/// Result from batch Jaccard distance computation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchJaccardResult: Sendable {
    /// Flattened distance matrix
    public let distances: [Float]
    /// Number of query vectors (M)
    public let rows: Int
    /// Number of dataset vectors (N)
    public let cols: Int
    /// Total execution time
    public let totalExecutionTime: TimeInterval
    /// Average distance across all pairs
    public let averageDistance: Double

    /// Get distance at (row, col).
    public func distance(row: Int, col: Int) -> Float {
        guard row >= 0 && row < rows && col >= 0 && col < cols else {
            return Float.infinity
        }
        return distances[row * cols + col]
    }

    /// Extract full distance matrix.
    public func asMatrix() -> [[Float]] {
        var matrix: [[Float]] = []
        matrix.reserveCapacity(rows)
        for r in 0..<rows {
            var rowData: [Float] = []
            rowData.reserveCapacity(cols)
            for c in 0..<cols {
                rowData.append(distances[r * cols + c])
            }
            matrix.append(rowData)
        }
        return matrix
    }

    /// Get row of distances.
    public func row(_ index: Int) -> [Float] {
        guard index >= 0 && index < rows else { return [] }
        let start = index * cols
        let end = start + cols
        return Array(distances[start..<end])
    }

    /// Find k nearest neighbors for each query.
    public func nearestNeighbors(k: Int) -> [[(index: Int, distance: Float)]] {
        var results: [[(index: Int, distance: Float)]] = []
        results.reserveCapacity(rows)

        for r in 0..<rows {
            let rowDistances = row(r)
            let indexed = rowDistances.enumerated().map { (index: $0.offset, distance: $0.element) }
            results.append(Array(indexed.sorted { $0.distance < $1.distance }.prefix(k)))
        }

        return results
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Jaccard Distance kernel.
///
/// Computes Jaccard distance for set similarity:
/// ```
/// J(A, B) = |A ∩ B| / |A ∪ B|
/// distance = 1 - J(A, B)
/// ```
///
/// ## Set Representation
///
/// Vectors are binarized using a threshold - values above threshold are
/// considered "present" in the set.
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4JaccardDistanceKernel(context: context)
///
/// // Single pair distance
/// let result = try await kernel.computeDistance(vectorA, vectorB)
/// print("Similarity: \(result.similarity)")
///
/// // Distance matrix
/// let batch = try await kernel.computeDistanceMatrix(queries, dataset)
/// let nearest = batch.nearestNeighbors(k: 5)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4JaccardDistanceKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4JaccardDistanceKernel"
    public let fusibleWith: [String] = ["TopKSelection"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Constants

    private let threadsPerThreadgroup: Int = 256

    // MARK: - Pipeline

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Jaccard Distance kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "jaccardDistance") else {
            throw VectorError.shaderNotFound(
                name: "Jaccard distance kernel. Ensure DistanceShaders.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipeline = try await device.makeComputePipelineState(function: function)

        // Validate hardware support
        if pipeline.maxTotalThreadsPerThreadgroup < threadsPerThreadgroup {
            throw VectorError.unsupportedGPUOperation(
                "Device does not support required threadgroup size: \(threadsPerThreadgroup)"
            )
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline created in init
    }

    // MARK: - Encode API

    /// Encode Jaccard distance computation into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        vectorA: any MTLBuffer,
        vectorB: any MTLBuffer,
        output: any MTLBuffer,
        dimension: Int,
        threshold: Float
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "JaccardDistance"

        encoder.setBuffer(vectorA, offset: 0, index: 0)
        encoder.setBuffer(vectorB, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var dim = UInt32(dimension)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

        var thresh = threshold
        encoder.setBytes(&thresh, length: MemoryLayout<Float>.size, index: 4)

        let threadgroupSize = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (dimension + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "jaccardDistance",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Compute Jaccard distance between two vectors.
    public func computeDistance(
        vectorA: [Float],
        vectorB: [Float],
        config: Metal4JaccardConfig = .default
    ) async throws -> Metal4JaccardResult {
        guard vectorA.count == vectorB.count else {
            throw VectorError.countMismatch(expected: vectorA.count, actual: vectorB.count)
        }

        if vectorA.isEmpty {
            return Metal4JaccardResult(
                distance: 0.0,
                similarity: 1.0,
                intersectionSize: 0,
                unionSize: 0,
                executionTime: 0.0
            )
        }

        let dimension = vectorA.count
        let device = context.device.rawDevice

        guard let bufferA = device.makeBuffer(
            bytes: vectorA,
            length: vectorA.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: vectorA.count * MemoryLayout<Float>.size)
        }

        guard let bufferB = device.makeBuffer(
            bytes: vectorB,
            length: vectorB.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: vectorB.count * MemoryLayout<Float>.size)
        }

        guard let resultBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<Float>.size)
        }

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                vectorA: bufferA,
                vectorB: bufferB,
                output: resultBuffer,
                dimension: dimension,
                threshold: config.threshold
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        let distance = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee

        // Calculate intersection and union for detailed metrics
        let metrics = calculateJaccardMetrics(vectorA: vectorA, vectorB: vectorB, threshold: config.threshold)

        return Metal4JaccardResult(
            distance: distance,
            similarity: 1.0 - distance,
            intersectionSize: metrics.intersection,
            unionSize: metrics.union,
            executionTime: executionTime
        )
    }

    /// Compute pairwise Jaccard distances between two sets of vectors.
    public func computeDistanceMatrix(
        vectorsA: [[Float]],
        vectorsB: [[Float]],
        config: Metal4JaccardConfig = .default
    ) async throws -> Metal4BatchJaccardResult {
        guard let dimension = vectorsA.first?.count,
              dimension > 0,
              vectorsB.first?.count == dimension else {
            throw VectorError.invalidInput("Input vectors must be non-empty and share the same dimension")
        }

        // Validate all vectors have same dimension
        for vector in vectorsA {
            guard vector.count == dimension else {
                throw VectorError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        for vector in vectorsB {
            guard vector.count == dimension else {
                throw VectorError.countMismatch(expected: dimension, actual: vector.count)
            }
        }

        let rows = vectorsA.count
        let cols = vectorsB.count
        let device = context.device.rawDevice

        // Flatten input vectors
        let flatA = vectorsA.flatMap { $0 }
        let flatB = vectorsB.flatMap { $0 }

        guard let bufferA = device.makeBuffer(
            bytes: flatA,
            length: flatA.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatA.count * MemoryLayout<Float>.size)
        }

        guard let bufferB = device.makeBuffer(
            bytes: flatB,
            length: flatB.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatB.count * MemoryLayout<Float>.size)
        }

        let resultSize = rows * cols * MemoryLayout<Float>.size
        guard let resultBuffer = device.makeBuffer(length: resultSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: resultSize)
        }

        let vectorStride = dimension * MemoryLayout<Float>.size
        let startTime = CACurrentMediaTime()

        // Execute all pairwise computations
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "JaccardDistanceMatrix"

            var dim = UInt32(dimension)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)

            var thresh = config.threshold
            encoder.setBytes(&thresh, length: MemoryLayout<Float>.size, index: 4)

            let threadgroupSize = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
            let threadgroups = MTLSize(
                width: (dimension + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
                height: 1,
                depth: 1
            )

            for r in 0..<rows {
                let offsetA = r * vectorStride
                encoder.setBuffer(bufferA, offset: offsetA, index: 0)

                for c in 0..<cols {
                    let offsetB = c * vectorStride
                    let resultOffset = (r * cols + c) * MemoryLayout<Float>.size

                    encoder.setBuffer(bufferB, offset: offsetB, index: 1)
                    encoder.setBuffer(resultBuffer, offset: resultOffset, index: 2)

                    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
                }
            }
        }

        let totalExecutionTime = CACurrentMediaTime() - startTime

        // Extract results
        let resultPtr = resultBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let distances = Array(UnsafeBufferPointer(start: resultPtr, count: rows * cols))
        let averageDistance = distances.reduce(0.0) { $0 + Double($1) } / Double(distances.count)

        return Metal4BatchJaccardResult(
            distances: distances,
            rows: rows,
            cols: cols,
            totalExecutionTime: totalExecutionTime,
            averageDistance: averageDistance
        )
    }

    // MARK: - VectorCore Integration

    /// Compute Jaccard distance using VectorCore protocol types.
    public func computeDistance<V: VectorProtocol>(
        _ vectorA: V,
        _ vectorB: V,
        config: Metal4JaccardConfig = .default
    ) async throws -> Metal4JaccardResult where V.Scalar == Float {
        let arrayA: [Float] = vectorA.withUnsafeBufferPointer { Array($0) }
        let arrayB: [Float] = vectorB.withUnsafeBufferPointer { Array($0) }
        return try await computeDistance(vectorA: arrayA, vectorB: arrayB, config: config)
    }

    /// Batch computation using VectorCore protocol types.
    public func computeDistanceMatrix<V: VectorProtocol>(
        vectorsA: [V],
        vectorsB: [V],
        config: Metal4JaccardConfig = .default
    ) async throws -> Metal4BatchJaccardResult where V.Scalar == Float {
        let floatA = vectorsA.map { v in v.withUnsafeBufferPointer { Array($0) } }
        let floatB = vectorsB.map { v in v.withUnsafeBufferPointer { Array($0) } }
        return try await computeDistanceMatrix(vectorsA: floatA, vectorsB: floatB, config: config)
    }

    // MARK: - Convenience Methods

    /// Compute Jaccard similarity (1 - distance).
    public func similarity(
        _ vectorA: [Float],
        _ vectorB: [Float],
        threshold: Float = 0.0
    ) async throws -> Float {
        let result = try await computeDistance(
            vectorA: vectorA,
            vectorB: vectorB,
            config: Metal4JaccardConfig(threshold: threshold)
        )
        return result.similarity
    }

    /// Find k most similar vectors.
    public func findMostSimilar(
        query: [Float],
        dataset: [[Float]],
        k: Int,
        config: Metal4JaccardConfig = .default
    ) async throws -> [(index: Int, similarity: Float)] {
        let result = try await computeDistanceMatrix(
            vectorsA: [query],
            vectorsB: dataset,
            config: config
        )

        let distances = result.row(0)
        let indexed = distances.enumerated().map { (index: $0.offset, similarity: 1.0 - $0.element) }
        return Array(indexed.sorted { $0.similarity > $1.similarity }.prefix(k))
    }

    // MARK: - Private Helpers

    /// CPU fallback for calculating intersection and union counts.
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
