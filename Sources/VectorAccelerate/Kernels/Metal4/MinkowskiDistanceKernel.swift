//
//  MinkowskiDistanceKernel.swift
//  VectorAccelerate
//
//  Metal 4 Minkowski Distance kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 6a, Priority 5
//
//  Features:
//  - Minkowski distance (Lp norm) computation
//  - Optimizations for p=1 (Manhattan), p=2 (Euclidean), p→∞ (Chebyshev)
//  - Pairwise and single vector pair modes
//  - Fusible with TopK selection

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for Minkowski distance computation.
public struct Metal4MinkowskiConfig: Sendable {
    /// Minkowski parameter (p value)
    public let p: Float
    /// Computation mode
    public let mode: Mode
    /// Use numerically stable computation for large p
    public let useStableComputation: Bool

    public enum Mode: UInt32, Sendable {
        case pairwise = 0  // Compute full M×N distance matrix
        case single = 1    // Compute single vector pair distance
    }

    public init(
        p: Float,
        mode: Mode = .pairwise,
        useStableComputation: Bool? = nil
    ) {
        self.p = p
        self.mode = mode
        // Auto-determine stable computation if not specified
        self.useStableComputation = useStableComputation ?? (p > 10.0 && p <= 30.0)
    }

    /// Returns true if this is effectively Manhattan distance (L1)
    public var isManhattan: Bool { abs(p - 1.0) < 0.001 }

    /// Returns true if this is effectively Euclidean distance (L2)
    public var isEuclidean: Bool { abs(p - 2.0) < 0.001 }

    /// Returns true if this approximates Chebyshev distance (L∞)
    public var isChebyshev: Bool { p > 30.0 }

    /// Get a descriptive name for the metric
    public var metricName: String {
        if isManhattan { return "Manhattan (L1)" }
        if isEuclidean { return "Euclidean (L2)" }
        if isChebyshev { return "Chebyshev (L∞)" }
        return "Minkowski (L\(p))"
    }

    // Common presets
    public static let manhattan = Metal4MinkowskiConfig(p: 1.0)
    public static let euclidean = Metal4MinkowskiConfig(p: 2.0)
    public static let chebyshev = Metal4MinkowskiConfig(p: 100.0)  // Approximates L∞
}

// MARK: - Parameters

/// Parameters for Minkowski distance kernel.
internal struct Metal4MinkowskiParams: Sendable {
    var p: Float
    var M: UInt32
    var N: UInt32
    var D: UInt32
    var mode: UInt32
}

// MARK: - Result Types

/// Result from Minkowski distance computation.
public struct Metal4MinkowskiResult: Sendable {
    /// Distance matrix buffer
    public let distances: any MTLBuffer
    /// Number of query vectors (M)
    public let rows: Int
    /// Number of dataset vectors (N)
    public let cols: Int
    /// Minkowski parameter used
    public let p: Float
    /// Execution time
    public let executionTime: TimeInterval
    /// Throughput in distance computations per second
    public let throughput: Double

    /// Extract distance matrix as 2D array.
    public func asMatrix() -> [[Float]] {
        let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
        var matrix: [[Float]] = []
        matrix.reserveCapacity(rows)

        for i in 0..<rows {
            var row: [Float] = []
            row.reserveCapacity(cols)
            for j in 0..<cols {
                row.append(ptr[i * cols + j])
            }
            matrix.append(row)
        }

        return matrix
    }

    /// Get distance for specific pair.
    public func distance(row: Int, col: Int) -> Float {
        guard row < rows && col < cols else { return Float.infinity }
        let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
        return ptr[row * cols + col]
    }

    /// Find minimum distance and its indices.
    public func minimum() -> (value: Float, row: Int, col: Int) {
        let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
        var minVal = Float.infinity
        var minRow = 0
        var minCol = 0

        for i in 0..<rows {
            for j in 0..<cols {
                let val = ptr[i * cols + j]
                if val < minVal {
                    minVal = val
                    minRow = i
                    minCol = j
                }
            }
        }

        return (minVal, minRow, minCol)
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

/// Metal 4 Minkowski Distance kernel.
///
/// Computes Minkowski distance (Lp norm) between vectors:
/// ```
/// d(x, y) = (Σ |x_i - y_i|^p)^(1/p)
/// ```
///
/// ## Special Cases
///
/// - **p = 1**: Manhattan distance (L1 norm)
/// - **p = 2**: Euclidean distance (L2 norm)
/// - **p → ∞**: Chebyshev distance (L∞ norm / max absolute difference)
///
/// ## Usage
///
/// ```swift
/// let kernel = try await MinkowskiDistanceKernel(context: context)
///
/// // Euclidean distance
/// let euclidean = try await kernel.computeDistances(queries, dataset, config: .euclidean)
///
/// // Manhattan distance
/// let manhattan = try await kernel.computeDistances(queries, dataset, config: .manhattan)
///
/// // Custom p value
/// let custom = try await kernel.computeDistances(queries, dataset, config: Metal4MinkowskiConfig(p: 3.0))
/// ```
public final class MinkowskiDistanceKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "MinkowskiDistanceKernel"
    public let fusibleWith: [String] = ["TopKSelection"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipeline

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Minkowski Distance kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "minkowski_distance_batch") else {
            throw VectorError.shaderNotFound(
                name: "Minkowski distance kernel. Ensure MinkowskiDistance.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipeline = try await device.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline created in init
    }

    // MARK: - Encode API

    /// Encode Minkowski distance computation into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        vectorsA: any MTLBuffer,
        vectorsB: any MTLBuffer,
        output: any MTLBuffer,
        M: Int,
        N: Int,
        D: Int,
        config: Metal4MinkowskiConfig
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "MinkowskiDistance (\(config.metricName))"

        encoder.setBuffer(vectorsA, offset: 0, index: 0)
        encoder.setBuffer(vectorsB, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        // Pass parameters individually to match Metal kernel signature
        var p = config.p
        var numQueries = UInt32(M)
        var numDataset = UInt32(N)
        var dimension = UInt32(D)
        encoder.setBytes(&p, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&numQueries, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&numDataset, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 6)

        // Dispatch with 16×16 threadgroups (matching TILE_M × TILE_N)
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (N + 15) / 16,
            height: (M + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "minkowski_distance",
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Execute Minkowski distance computation as standalone operation.
    public func execute(
        vectorsA: any MTLBuffer,
        vectorsB: any MTLBuffer,
        M: Int,
        N: Int,
        D: Int,
        config: Metal4MinkowskiConfig
    ) async throws -> Metal4MinkowskiResult {
        guard M > 0 && N > 0 && D > 0 else {
            throw VectorError.invalidInput("Dimensions must be positive")
        }

        guard config.p > 0 else {
            throw VectorError.invalidInput("Minkowski parameter p must be positive")
        }

        let device = context.device.rawDevice
        let outputSize = M * N * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "MinkowskiDistance.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                vectorsA: vectorsA,
                vectorsB: vectorsB,
                output: outputBuffer,
                M: M,
                N: N,
                D: D,
                config: config
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        let numDistances = Double(M * N)
        let throughput = numDistances / executionTime

        return Metal4MinkowskiResult(
            distances: outputBuffer,
            rows: M,
            cols: N,
            p: config.p,
            executionTime: executionTime,
            throughput: throughput
        )
    }

    // MARK: - High-Level API

    /// Compute distance matrix for arrays of vectors.
    public func computeDistances(
        queries: [[Float]],
        dataset: [[Float]],
        config: Metal4MinkowskiConfig = .euclidean
    ) async throws -> Metal4MinkowskiResult {
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

        guard let queriesBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        queriesBuffer.label = "MinkowskiDistance.queries"

        guard let datasetBuffer = device.makeBuffer(
            bytes: flatDataset,
            length: flatDataset.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDataset.count * MemoryLayout<Float>.size)
        }
        datasetBuffer.label = "MinkowskiDistance.dataset"

        return try await execute(
            vectorsA: queriesBuffer,
            vectorsB: datasetBuffer,
            M: queries.count,
            N: dataset.count,
            D: dimension,
            config: config
        )
    }

    /// Compute distance between two vectors.
    public func distance(
        _ a: [Float],
        _ b: [Float],
        p: Float = 2.0
    ) async throws -> Float {
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }

        let config = Metal4MinkowskiConfig(p: p, mode: .single)
        let result = try await computeDistances(queries: [a], dataset: [b], config: config)
        return result.distance(row: 0, col: 0)
    }

    /// Find k nearest neighbors using Minkowski distance.
    public func findNearestNeighbors(
        query: [Float],
        dataset: [[Float]],
        k: Int,
        p: Float = 2.0
    ) async throws -> [(index: Int, distance: Float)] {
        let config = Metal4MinkowskiConfig(p: p)
        let result = try await computeDistances(queries: [query], dataset: dataset, config: config)
        return result.nearestNeighbors(k: k)[0]
    }

    // MARK: - VectorCore Integration

    /// Compute Minkowski distance using VectorCore types.
    public func distance<V: VectorProtocol>(
        _ a: V,
        _ b: V,
        p: Float = 2.0
    ) async throws -> Float where V.Scalar == Float {
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }

        let arrayA: [Float] = a.withUnsafeBufferPointer { Array($0) }
        let arrayB: [Float] = b.withUnsafeBufferPointer { Array($0) }

        return try await distance(arrayA, arrayB, p: p)
    }

    /// Compute distance matrix using VectorCore types.
    public func computeDistances<V: VectorProtocol>(
        queries: [V],
        dataset: [V],
        config: Metal4MinkowskiConfig = .euclidean
    ) async throws -> Metal4MinkowskiResult where V.Scalar == Float {
        let floatQueries = queries.map { v in v.withUnsafeBufferPointer { Array($0) } }
        let floatDataset = dataset.map { v in v.withUnsafeBufferPointer { Array($0) } }
        return try await computeDistances(queries: floatQueries, dataset: floatDataset, config: config)
    }

    // MARK: - Convenience Methods

    /// Compute Manhattan (L1) distance matrix.
    public func manhattanDistances(
        queries: [[Float]],
        dataset: [[Float]]
    ) async throws -> Metal4MinkowskiResult {
        return try await computeDistances(queries: queries, dataset: dataset, config: .manhattan)
    }

    /// Compute Euclidean (L2) distance matrix.
    public func euclideanDistances(
        queries: [[Float]],
        dataset: [[Float]]
    ) async throws -> Metal4MinkowskiResult {
        return try await computeDistances(queries: queries, dataset: dataset, config: .euclidean)
    }

    /// Compute Chebyshev (L∞) distance matrix.
    public func chebyshevDistances(
        queries: [[Float]],
        dataset: [[Float]]
    ) async throws -> Metal4MinkowskiResult {
        return try await computeDistances(queries: queries, dataset: dataset, config: .chebyshev)
    }
}
