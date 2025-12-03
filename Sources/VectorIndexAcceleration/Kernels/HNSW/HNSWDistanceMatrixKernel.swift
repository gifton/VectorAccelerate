//
//  HNSWDistanceMatrixKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for computing distance matrices in HNSW operations.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Distance Matrix Configuration

/// Configuration for HNSW distance matrix computation.
public struct HNSWDistanceMatrixConfiguration: Sendable {
    /// Number of query vectors
    public let queryCount: Int

    /// Number of candidate vectors
    public let candidateCount: Int

    /// Vector dimension
    public let dimension: Int

    /// Distance metric to use
    public let metric: SupportedDistanceMetric

    /// Batch size for processing
    public let batchSize: Int

    /// Tile size for shared memory caching
    public let tileSize: Int

    public init(
        queryCount: Int,
        candidateCount: Int,
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        batchSize: Int = 32,
        tileSize: Int = 32
    ) {
        self.queryCount = queryCount
        self.candidateCount = candidateCount
        self.dimension = dimension
        self.metric = metric
        self.batchSize = batchSize
        self.tileSize = tileSize
    }

    /// Validate configuration
    public func validate() throws {
        guard queryCount > 0 else {
            throw IndexAccelerationError.invalidInput(message: "Query count must be positive")
        }
        guard candidateCount > 0 else {
            throw IndexAccelerationError.invalidInput(message: "Candidate count must be positive")
        }
        guard dimension > 0 && dimension <= 4096 else {
            throw IndexAccelerationError.invalidInput(message: "Dimension must be between 1 and 4096")
        }
        guard tileSize > 0 && tileSize <= 64 else {
            throw IndexAccelerationError.invalidInput(message: "Tile size must be between 1 and 64")
        }
    }
}

// MARK: - Distance Matrix Result

/// Result from distance matrix computation.
public struct HNSWDistanceMatrixResult: Sendable {
    /// Distance matrix [numQueries][numCandidates]
    public let distances: [[Float]]

    /// Execution time
    public let executionTime: TimeInterval

    /// Configuration used
    public let configuration: HNSWDistanceMatrixConfiguration

    /// Get distance between query i and candidate j
    public func distance(query: Int, candidate: Int) -> Float {
        distances[query][candidate]
    }

    /// Get k nearest candidates for a query
    public func kNearest(for query: Int, k: Int) -> [(index: Int, distance: Float)] {
        let queryDistances = distances[query]
        return queryDistances.enumerated()
            .sorted { $0.element < $1.element }
            .prefix(k)
            .map { (index: $0.offset, distance: $0.element) }
    }
}

// MARK: - HNSW Distance Matrix Kernel

/// Metal 4 kernel for computing distance matrices in HNSW operations.
///
/// This kernel computes pairwise distances between query vectors and candidate
/// vectors, optimized for HNSW graph construction and search operations.
///
/// ## Features
/// - Multiple distance metrics (L2, Cosine, DotProduct)
/// - Tiled computation with shared memory caching
/// - Batch processing for large datasets
/// - Support for kernel fusion via encode() API
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWDistanceMatrixKernel(context: context)
/// let result = try await kernel.compute(
///     queries: queryVectors,
///     candidates: candidateVectors,
///     configuration: config
/// )
/// ```
public final class HNSWDistanceMatrixKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWDistanceMatrixKernel"

    public let fusibleWith: [String] = ["TopKSelection", "HNSWSearch"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Private Properties

    private let pipeline: any MTLComputePipelineState
    private let maxDimension = 4096
    private let maxBatchSize = 256

    // MARK: - Initialization

    /// Create a new HNSW distance matrix kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    /// - Throws: `VectorError.shaderNotFound` if kernel function is missing
    public init(context: Metal4Context) async throws {
        self.context = context

        // Load library and create pipeline
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "hnsw_distance_matrix") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_distance_matrix. Ensure HNSW shaders are compiled."
            )
        }

        self.pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline is created in init, so this is a no-op
        // Could be extended to trigger initial dispatch for GPU shader cache
    }

    // MARK: - Encode API (for Fusion)

    /// Encode distance matrix computation into an existing encoder.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder
    ///   - queries: Query vectors buffer [N, D]
    ///   - candidates: Candidate vectors buffer [M, D]
    ///   - distances: Output buffer [N, M]
    ///   - parameters: Kernel parameters
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        queries: any MTLBuffer,
        candidates: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: HNSWDistanceMatrixShaderArgs
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.label = "HNSW Distance Matrix"

        // Bind buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(candidates, offset: 0, index: 1)
        encoder.setBuffer(distances, offset: 0, index: 2)

        // Bind parameters
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<HNSWDistanceMatrixShaderArgs>.size, index: 3)

        // Calculate thread configuration
        let config = Metal4ThreadConfiguration.forDistanceKernel(
            numQueries: Int(parameters.numQueries),
            numDatabase: Int(parameters.numCandidates),
            pipeline: pipeline
        )

        // Dispatch
        encoder.dispatchThreadgroups(
            config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute distance matrix computation as standalone operation.
    ///
    /// - Parameters:
    ///   - queries: Query vectors buffer [N, D]
    ///   - candidates: Candidate vectors buffer [M, D]
    ///   - parameters: Kernel parameters
    /// - Returns: Buffer containing distances [N, M]
    public func execute(
        queries: any MTLBuffer,
        candidates: any MTLBuffer,
        parameters: HNSWDistanceMatrixShaderArgs
    ) async throws -> any MTLBuffer {
        // Allocate output buffer
        let outputSize = Int(parameters.numQueries) * Int(parameters.numCandidates) * MemoryLayout<Float>.size
        guard let distanceBuffer = context.device.rawDevice.makeBuffer(
            length: outputSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        distanceBuffer.label = "HNSWDistanceMatrix.output"

        // Execute via context
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                queries: queries,
                candidates: candidates,
                distances: distanceBuffer,
                parameters: parameters
            )
        }

        return distanceBuffer
    }

    // MARK: - High-Level API

    /// Compute distance matrix from Float arrays.
    ///
    /// - Parameters:
    ///   - queries: Query vectors [numQueries][dimension]
    ///   - candidates: Candidate vectors [numCandidates][dimension]
    ///   - configuration: Computation configuration
    /// - Returns: Distance matrix result
    public func compute(
        queries: [[Float]],
        candidates: [[Float]],
        configuration: HNSWDistanceMatrixConfiguration
    ) async throws -> HNSWDistanceMatrixResult {
        // Validate configuration
        try configuration.validate()

        // Validate input dimensions
        guard queries.count == configuration.queryCount else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: configuration.queryCount,
                got: queries.count
            )
        }
        guard candidates.count == configuration.candidateCount else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: configuration.candidateCount,
                got: candidates.count
            )
        }

        let startTime = CACurrentMediaTime()

        // Flatten input data
        let queryData = queries.flatMap { $0 }
        let candidateData = candidates.flatMap { $0 }

        // Create buffers
        let device = context.device.rawDevice

        guard let queryBuffer = device.makeBuffer(
            bytes: queryData,
            length: queryData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: queryData.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "HNSWDistanceMatrix.queries"

        guard let candidateBuffer = device.makeBuffer(
            bytes: candidateData,
            length: candidateData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: candidateData.count * MemoryLayout<Float>.size)
        }
        candidateBuffer.label = "HNSWDistanceMatrix.candidates"

        // Create parameters
        let parameters = HNSWDistanceMatrixShaderArgs(
            dimension: configuration.dimension,
            numQueries: configuration.queryCount,
            numCandidates: configuration.candidateCount,
            metric: configuration.metric,
            tileSize: configuration.tileSize
        )

        // Execute
        let distanceBuffer = try await execute(
            queries: queryBuffer,
            candidates: candidateBuffer,
            parameters: parameters
        )

        // Extract results
        let distances = extractDistances(
            from: distanceBuffer,
            queryCount: configuration.queryCount,
            candidateCount: configuration.candidateCount
        )

        let executionTime = CACurrentMediaTime() - startTime

        return HNSWDistanceMatrixResult(
            distances: distances,
            executionTime: executionTime,
            configuration: configuration
        )
    }

    /// Compute batched distance matrices for large datasets.
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - candidates: Candidate vectors
    ///   - configuration: Computation configuration
    /// - Returns: Distance matrix result
    public func computeBatched(
        queries: [[Float]],
        candidates: [[Float]],
        configuration: HNSWDistanceMatrixConfiguration
    ) async throws -> HNSWDistanceMatrixResult {
        try configuration.validate()

        let batchSize = min(configuration.batchSize, maxBatchSize)
        var allDistances: [[Float]] = Array(repeating: [], count: queries.count)
        var totalExecutionTime: TimeInterval = 0

        // Process queries in batches
        for queryBatchStart in stride(from: 0, to: queries.count, by: batchSize) {
            let queryBatchEnd = min(queryBatchStart + batchSize, queries.count)
            let queryBatch = Array(queries[queryBatchStart..<queryBatchEnd])

            // Create batch configuration
            let batchConfig = HNSWDistanceMatrixConfiguration(
                queryCount: queryBatch.count,
                candidateCount: configuration.candidateCount,
                dimension: configuration.dimension,
                metric: configuration.metric,
                batchSize: configuration.batchSize,
                tileSize: configuration.tileSize
            )

            // Compute batch distances
            let batchResult = try await compute(
                queries: queryBatch,
                candidates: candidates,
                configuration: batchConfig
            )

            // Store results
            for (i, distances) in batchResult.distances.enumerated() {
                allDistances[queryBatchStart + i] = distances
            }

            totalExecutionTime += batchResult.executionTime
        }

        return HNSWDistanceMatrixResult(
            distances: allDistances,
            executionTime: totalExecutionTime,
            configuration: configuration
        )
    }

    // MARK: - Private Helpers

    private func extractDistances(
        from buffer: any MTLBuffer,
        queryCount: Int,
        candidateCount: Int
    ) -> [[Float]] {
        let distancePointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: queryCount * candidateCount
        )

        var results: [[Float]] = []
        results.reserveCapacity(queryCount)

        for q in 0..<queryCount {
            var row: [Float] = []
            row.reserveCapacity(candidateCount)
            for c in 0..<candidateCount {
                row.append(distancePointer[q * candidateCount + c])
            }
            results.append(row)
        }

        return results
    }
}
