//
//  TopKSelectionKernel.swift
//  VectorAccelerate
//
//  Metal 4 Top-K Selection kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 2, Priority 2
//
//  Features:
//  - Batch top-k selection from distance/similarity matrices
//  - Supports both minimum (nearest) and maximum (farthest) selection
//  - Fusible with distance kernels via encode() API
//  - Thread-safe for concurrent encoding

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Selection Mode

/// Mode for top-k selection
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4SelectionMode: UInt8, Sendable {
    /// Select K smallest values (nearest neighbors for distance metrics)
    case minimum = 0

    /// Select K largest values (best matches for similarity metrics)
    case maximum = 1
}

// MARK: - Parameters

/// Parameters for Top-K Selection kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct TopKParameters: Sendable {
    /// Number of queries (batch size)
    public let batchSize: UInt32

    /// Number of elements per query (candidates)
    public let numElements: UInt32

    /// Number of top elements to select
    public let k: UInt32

    /// Stride between query rows in input
    public let inputStride: UInt32

    /// Stride between query rows in output
    public let outputStride: UInt32

    /// Selection mode: 0 = minimum, 1 = maximum
    public let mode: UInt8

    /// Whether to sort output: 0 = unsorted, 1 = sorted
    public let sorted: UInt8

    /// Padding for alignment
    private let padding: (UInt8, UInt8) = (0, 0)

    /// Maximum K supported by the kernel
    public static let maxK: Int = 128

    /// Create parameters for dense input layout.
    ///
    /// - Parameters:
    ///   - batchSize: Number of queries
    ///   - numElements: Number of candidates per query
    ///   - k: Number of top elements to select
    ///   - mode: Selection mode (minimum or maximum)
    ///   - sorted: Whether to sort output by value
    public init(
        batchSize: Int,
        numElements: Int,
        k: Int,
        mode: Metal4SelectionMode = .minimum,
        sorted: Bool = true
    ) {
        self.batchSize = UInt32(batchSize)
        self.numElements = UInt32(numElements)
        self.k = UInt32(min(k, Self.maxK))
        self.inputStride = UInt32(numElements)
        self.outputStride = UInt32(min(k, Self.maxK))
        self.mode = mode.rawValue
        self.sorted = sorted ? 1 : 0
    }

    /// Create parameters with custom strides.
    public init(
        batchSize: Int,
        numElements: Int,
        k: Int,
        inputStride: Int,
        outputStride: Int,
        mode: Metal4SelectionMode = .minimum,
        sorted: Bool = true
    ) {
        self.batchSize = UInt32(batchSize)
        self.numElements = UInt32(numElements)
        self.k = UInt32(min(k, Self.maxK))
        self.inputStride = UInt32(inputStride)
        self.outputStride = UInt32(outputStride)
        self.mode = mode.rawValue
        self.sorted = sorted ? 1 : 0
    }
}

// MARK: - Result Type

/// Result from top-k selection operation
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4TopKResult: Sendable {
    /// Buffer containing selected values [batchSize × k]
    public let values: any MTLBuffer

    /// Buffer containing selected indices [batchSize × k]
    public let indices: any MTLBuffer

    /// Number of queries in batch
    public let batchSize: Int

    /// K value used
    public let k: Int

    /// Extract results for a specific query
    public func results(for queryIndex: Int) -> [(index: Int, value: Float)] {
        guard queryIndex < batchSize else { return [] }

        let offset = queryIndex * k
        let valuePtr = values.contents().bindMemory(to: Float.self, capacity: batchSize * k)
        let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: batchSize * k)

        var results: [(index: Int, value: Float)] = []
        results.reserveCapacity(k)
        for i in 0..<k {
            let idx = indexPtr[offset + i]
            if idx != 0xFFFFFFFF {  // Skip sentinel values
                results.append((index: Int(idx), value: valuePtr[offset + i]))
            }
        }
        return results
    }

    /// Get all results as 2D array
    public func allResults() -> [[(index: Int, value: Float)]] {
        (0..<batchSize).map { results(for: $0) }
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Top-K Selection kernel.
///
/// Selects the K largest or smallest elements from each row of an input matrix.
/// Commonly used after distance computation to find nearest neighbors.
///
/// ## Fusion Pattern
///
/// This kernel is designed to be fused with distance kernels:
/// ```swift
/// try await context.executeAndWait { commandBuffer, encoder in
///     // 1. Compute distances
///     distanceKernel.encode(into: encoder, ...)
///
///     // 2. Barrier before selection
///     encoder.memoryBarrier(scope: .buffers)
///
///     // 3. Select top-k from distances
///     topKKernel.encode(into: encoder, ...)
/// }
/// ```
///
/// ## Performance Notes
///
/// - Uses heap-based selection for K << N (K up to 128)
/// - Single-pass algorithm avoids sorting entire input
/// - Batch processing for multiple queries in parallel
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class TopKSelectionKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "TopKSelectionKernel"

    public let fusibleWith: [String] = ["L2Distance", "CosineSimilarity", "DotProduct"]
    public let requiresBarrierAfter: Bool = false  // Output is final, no subsequent reads expected

    // MARK: - Pipeline

    private let batchPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Top-K Selection kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let batchFunc = library.makeFunction(name: "topk_select_batch_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Top-K selection kernel. Ensure AdvancedTopK.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.batchPipeline = try await device.makeComputePipelineState(function: batchFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline created in init
    }

    // MARK: - Encode API

    /// Encode top-k selection into an existing encoder.
    ///
    /// **Important**: If the input buffer was written by a previous kernel in the same
    /// command buffer, insert `encoder.memoryBarrier(scope: .buffers)` before this call.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - input: Input values buffer [batchSize × numElements]
    ///   - outputValues: Output buffer for selected values [batchSize × k]
    ///   - outputIndices: Output buffer for selected indices [batchSize × k]
    ///   - parameters: Selection parameters
    /// - Returns: Encoding result
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        outputValues: any MTLBuffer,
        outputIndices: any MTLBuffer,
        parameters: TopKParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(batchPipeline)
        encoder.label = "TopKSelection.batch (K=\(parameters.k))"

        // Bind buffers
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(outputValues, offset: 0, index: 1)
        encoder.setBuffer(outputIndices, offset: 0, index: 2)

        // Bind parameters
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<TopKParameters>.size, index: 3)

        // One threadgroup per query
        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.batchSize),
            pipeline: batchPipeline
        )

        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )

        return Metal4EncodingResult(
            pipelineName: "topk_select_batch_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute top-k selection as standalone operation.
    ///
    /// - Parameters:
    ///   - input: Input values buffer [batchSize × numElements]
    ///   - parameters: Selection parameters
    /// - Returns: Top-K result with values and indices
    public func execute(
        input: any MTLBuffer,
        parameters: TopKParameters
    ) async throws -> Metal4TopKResult {
        let device = context.device.rawDevice
        let k = Int(parameters.k)
        let batchSize = Int(parameters.batchSize)

        // Allocate output buffers
        let valuesSize = batchSize * k * MemoryLayout<Float>.size
        let indicesSize = batchSize * k * MemoryLayout<UInt32>.size

        guard let valuesBuffer = device.makeBuffer(length: valuesSize, options: .storageModeShared),
              let indicesBuffer = device.makeBuffer(length: indicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: valuesSize + indicesSize)
        }
        valuesBuffer.label = "TopKSelection.values"
        indicesBuffer.label = "TopKSelection.indices"

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                input: input,
                outputValues: valuesBuffer,
                outputIndices: indicesBuffer,
                parameters: parameters
            )
        }

        return Metal4TopKResult(
            values: valuesBuffer,
            indices: indicesBuffer,
            batchSize: batchSize,
            k: k
        )
    }

    // MARK: - High-Level API

    /// Select top-k from a 2D array of values.
    ///
    /// - Parameters:
    ///   - values: Input values [batchSize][numElements]
    ///   - k: Number of top elements to select
    ///   - mode: Selection mode (minimum or maximum)
    ///   - sorted: Whether to sort output
    /// - Returns: Array of (index, value) pairs for each query
    public func select(
        from values: [[Float]],
        k: Int,
        mode: Metal4SelectionMode = .minimum,
        sorted: Bool = true
    ) async throws -> [[(index: Int, value: Float)]] {
        guard !values.isEmpty else {
            return []
        }

        let batchSize = values.count
        let numElements = values[0].count

        guard values.allSatisfy({ $0.count == numElements }) else {
            throw VectorError.invalidInput("All rows must have same length")
        }

        guard k > 0 && k <= TopKParameters.maxK else {
            throw VectorError.invalidInput("K must be between 1 and \(TopKParameters.maxK)")
        }

        let device = context.device.rawDevice
        let flatValues = values.flatMap { $0 }

        guard let inputBuffer = device.makeBuffer(
            bytes: flatValues,
            length: flatValues.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatValues.count * MemoryLayout<Float>.size)
        }

        let parameters = TopKParameters(
            batchSize: batchSize,
            numElements: numElements,
            k: k,
            mode: mode,
            sorted: sorted
        )

        let result = try await execute(input: inputBuffer, parameters: parameters)
        return result.allResults()
    }

    /// Select top-k from a single query.
    public func selectSingle(
        from values: [Float],
        k: Int,
        mode: Metal4SelectionMode = .minimum,
        sorted: Bool = true
    ) async throws -> [(index: Int, value: Float)] {
        let results = try await select(from: [values], k: k, mode: mode, sorted: sorted)
        return results.first ?? []
    }
}

// MARK: - Fused Distance + TopK Helper

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension TopKSelectionKernel {
    /// Convenience method for fused distance + top-k pipeline.
    ///
    /// This encodes both operations in a single command buffer with proper barrier.
    ///
    /// - Parameters:
    ///   - distanceKernel: The distance kernel to use
    ///   - queries: Query vectors buffer
    ///   - database: Database vectors buffer
    ///   - distanceParams: Distance kernel parameters
    ///   - k: Number of nearest neighbors
    ///   - mode: Selection mode
    /// - Returns: Top-K result
    public func fusedDistanceTopK<K: Metal4DistanceKernel>(
        distanceKernel: K,
        queries: any MTLBuffer,
        database: any MTLBuffer,
        distanceParams: K.Parameters,
        k: Int,
        mode: Metal4SelectionMode = .minimum
    ) async throws -> Metal4TopKResult where K.Parameters: Sendable {
        let device = context.device.rawDevice

        // Extract dimensions from distance params (assuming standard layout)
        // This is a simplification - real implementation would use protocol witness
        let numQueries: Int
        let numDatabase: Int

        // Use reflection or require protocol method
        if let l2Params = distanceParams as? L2DistanceParameters {
            numQueries = Int(l2Params.numQueries)
            numDatabase = Int(l2Params.numDatabase)
        } else if let cosineParams = distanceParams as? CosineSimilarityParameters {
            numQueries = Int(cosineParams.numQueries)
            numDatabase = Int(cosineParams.numDatabase)
        } else if let dotParams = distanceParams as? DotProductParameters {
            numQueries = Int(dotParams.numQueries)
            numDatabase = Int(dotParams.numDatabase)
        } else {
            throw VectorError.invalidOperation("Unknown distance parameter type")
        }

        // Allocate distance buffer
        let distanceSize = numQueries * numDatabase * MemoryLayout<Float>.size
        guard let distanceBuffer = device.makeBuffer(length: distanceSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distanceSize)
        }
        distanceBuffer.label = "FusedDistanceTopK.distances"

        // Allocate top-k output buffers
        let actualK = min(k, TopKParameters.maxK)
        let valuesSize = numQueries * actualK * MemoryLayout<Float>.size
        let indicesSize = numQueries * actualK * MemoryLayout<UInt32>.size

        guard let valuesBuffer = device.makeBuffer(length: valuesSize, options: .storageModeShared),
              let indicesBuffer = device.makeBuffer(length: indicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: valuesSize + indicesSize)
        }

        let topKParams = TopKParameters(
            batchSize: numQueries,
            numElements: numDatabase,
            k: actualK,
            mode: mode,
            sorted: true
        )

        // Execute fused pipeline
        try await context.executeAndWait { [self] _, encoder in
            // 1. Distance computation
            try distanceKernel.encode(
                into: encoder,
                queries: queries,
                database: database,
                distances: distanceBuffer,
                parameters: distanceParams
            )

            // 2. Barrier
            encoder.memoryBarrier(scope: .buffers)

            // 3. Top-K selection
            self.encode(
                into: encoder,
                input: distanceBuffer,
                outputValues: valuesBuffer,
                outputIndices: indicesBuffer,
                parameters: topKParams
            )
        }

        return Metal4TopKResult(
            values: valuesBuffer,
            indices: indicesBuffer,
            batchSize: numQueries,
            k: actualK
        )
    }
}
