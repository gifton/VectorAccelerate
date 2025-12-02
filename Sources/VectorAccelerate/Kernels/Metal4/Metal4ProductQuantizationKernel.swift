//
//  Metal4ProductQuantizationKernel.swift
//  VectorAccelerate
//
//  Metal 4 Product Quantization kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 3, Priority 3
//
//  Features:
//  - K-means based codebook training
//  - Vector encoding to uint8 codes
//  - Asymmetric Distance Computation (ADC)
//  - Configurable M subspaces and K centroids

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for Product Quantization.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4PQConfig: Sendable {
    /// Number of subspaces (vector is split into M parts)
    public let M: Int
    /// Number of centroids per subspace (typically 256 for uint8)
    public let K: Int
    /// Full vector dimension
    public let dimension: Int
    /// Number of training iterations
    public let trainIterations: Int
    /// Convergence threshold for training
    public let convergenceThreshold: Float

    /// Dimension of each subspace (D / M)
    public var D_sub: Int { dimension / M }

    /// Compression ratio (float32 to uint8 codes)
    public var compressionRatio: Float {
        Float(dimension * 32) / Float(M * 8)  // 32 bits per float / 8 bits per code
    }

    public init(
        dimension: Int,
        M: Int = 8,
        K: Int = 256,
        trainIterations: Int = 25,
        convergenceThreshold: Float = 0.001
    ) {
        precondition(dimension % M == 0, "Dimension must be divisible by M")
        precondition(K <= 256, "K must be <= 256 for uint8 encoding")

        self.dimension = dimension
        self.M = M
        self.K = K
        self.trainIterations = trainIterations
        self.convergenceThreshold = convergenceThreshold
    }
}

// MARK: - Model

/// Trained PQ model containing codebooks.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4PQModel: @unchecked Sendable {
    /// Codebook buffer [M × K × D_sub]
    public let codebooks: any MTLBuffer
    /// Configuration used for training
    public let config: Metal4PQConfig

    private let device: any MTLDevice

    internal init(codebooks: any MTLBuffer, config: Metal4PQConfig, device: any MTLDevice) {
        self.codebooks = codebooks
        self.config = config
        self.device = device
    }

    /// Compression ratio achieved
    public var compressionRatio: Float { config.compressionRatio }

    /// Memory usage of codebooks in bytes
    public var codebookMemoryBytes: Int {
        config.M * config.K * config.D_sub * MemoryLayout<Float>.size
    }

    /// Decode a single code to its centroid values.
    public func decodeCentroid(subspace m: Int, code: UInt8) -> [Float] {
        guard m < config.M else { return [] }

        let ptr = codebooks.contents().bindMemory(
            to: Float.self,
            capacity: config.M * config.K * config.D_sub
        )

        let centroidStart = (m * config.K + Int(code)) * config.D_sub
        var values: [Float] = []
        values.reserveCapacity(config.D_sub)

        for d in 0..<config.D_sub {
            values.append(ptr[centroidStart + d])
        }
        return values
    }
}

// MARK: - Encoded Vectors

/// Encoded vectors result.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4EncodedVectors: Sendable {
    /// Code buffer [N × M] uint8
    public let codes: any MTLBuffer
    /// Number of encoded vectors
    public let count: Int
    /// Configuration used
    public let config: Metal4PQConfig

    /// Memory usage in bytes
    public var memoryBytes: Int { count * config.M }

    /// Decode specific vector using model.
    public func decode(index: Int, using model: Metal4PQModel) -> [Float] {
        guard index < count else { return [] }

        let codesPtr = codes.contents().bindMemory(to: UInt8.self, capacity: count * config.M)
        let codebooksPtr = model.codebooks.contents().bindMemory(
            to: Float.self,
            capacity: config.M * config.K * config.D_sub
        )

        var decoded = Array<Float>(repeating: 0, count: config.dimension)

        for m in 0..<config.M {
            let code = codesPtr[index * config.M + m]
            let centroidStart = (m * config.K + Int(code)) * config.D_sub

            for d in 0..<config.D_sub {
                decoded[m * config.D_sub + d] = codebooksPtr[centroidStart + d]
            }
        }

        return decoded
    }
}

// MARK: - Internal Parameters

/// PQ kernel parameters structure (matches Metal shader).
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct Metal4PQParams: Sendable {
    var N: UInt32      // Number of vectors
    var D: UInt32      // Full dimension
    var M: UInt32      // Number of subspaces
    var K: UInt32      // Centroids per subspace
    var D_sub: UInt32  // Dimension per subspace

    init(config: Metal4PQConfig, count: Int = 0) {
        self.N = UInt32(count)
        self.D = UInt32(config.dimension)
        self.M = UInt32(config.M)
        self.K = UInt32(config.K)
        self.D_sub = UInt32(config.D_sub)
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Product Quantization kernel.
///
/// Implements Product Quantization for approximate nearest neighbor search.
/// Vectors are split into M subspaces, and each subspace is quantized using
/// K-means with K centroids (codebook).
///
/// ## Algorithm Overview
///
/// 1. **Training**: Learn M codebooks via k-means clustering
/// 2. **Encoding**: Assign each subvector to nearest centroid (uint8 code)
/// 3. **Search**: Use Asymmetric Distance Computation (ADC)
///
/// ## Memory Efficiency
///
/// - Original: D × 4 bytes (float32)
/// - Encoded: M bytes (uint8 codes)
/// - Typical: 128-dim → 8 bytes (16:1 compression)
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4ProductQuantizationKernel(context: context)
///
/// // Train model on dataset
/// let model = try await kernel.train(data: trainingData, config: config)
///
/// // Encode database vectors
/// let encoded = try await kernel.encode(vectors: database, model: model)
///
/// // Search
/// let neighbors = try await kernel.findNearestNeighbors(
///     query: query,
///     encodedDatabase: encoded,
///     model: model,
///     k: 10
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4ProductQuantizationKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4ProductQuantizationKernel"

    // MARK: - Pipelines

    private let assignmentPipeline: any MTLComputePipelineState
    private let updateAccumulatePipeline: any MTLComputePipelineState
    private let updateFinalizePipeline: any MTLComputePipelineState
    private let precomputeDistancePipeline: any MTLComputePipelineState
    private let computeDistancesPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Product Quantization kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let assignFunc = library.makeFunction(name: "pq_assignment_or_encoding"),
              let updateAccFunc = library.makeFunction(name: "pq_train_update_accumulate"),
              let updateFinFunc = library.makeFunction(name: "pq_train_update_finalize"),
              let precompFunc = library.makeFunction(name: "pq_precompute_distance_table"),
              let distFunc = library.makeFunction(name: "pq_compute_distances_adc") else {
            throw VectorError.shaderNotFound(
                name: "Product quantization kernels. Ensure ProductQuantization.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.assignmentPipeline = try await device.makeComputePipelineState(function: assignFunc)
        self.updateAccumulatePipeline = try await device.makeComputePipelineState(function: updateAccFunc)
        self.updateFinalizePipeline = try await device.makeComputePipelineState(function: updateFinFunc)
        self.precomputeDistancePipeline = try await device.makeComputePipelineState(function: precompFunc)
        self.computeDistancesPipeline = try await device.makeComputePipelineState(function: distFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Training

    /// Train PQ model on training data.
    ///
    /// - Parameters:
    ///   - data: Training vectors buffer [count × dimension]
    ///   - count: Number of training vectors
    ///   - config: PQ configuration
    ///   - progressHandler: Optional callback for training progress
    /// - Returns: Trained PQ model
    public func train(
        data: any MTLBuffer,
        count: Int,
        config: Metal4PQConfig,
        progressHandler: ((Int, Float) -> Void)? = nil
    ) async throws -> Metal4PQModel {
        let device = context.device.rawDevice

        // Allocate codebook buffer [M × K × D_sub]
        let codebookSize = config.M * config.K * config.D_sub * MemoryLayout<Float>.stride
        guard let codebooks = device.makeBuffer(length: codebookSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: codebookSize)
        }
        codebooks.label = "PQ.codebooks"

        // Initialize codebooks with random vectors from training data
        initializeCodebooks(codebooks: codebooks, data: data, count: count, config: config)

        // Allocate training buffers
        let assignmentSize = count * config.M * MemoryLayout<UInt8>.stride
        guard let assignments = device.makeBuffer(length: assignmentSize, options: .storageModePrivate) else {
            throw VectorError.bufferAllocationFailed(size: assignmentSize)
        }
        assignments.label = "PQ.assignments"

        let accumSize = config.M * config.K * config.D_sub * MemoryLayout<Float>.stride
        let countSize = config.M * config.K * MemoryLayout<UInt32>.stride
        let convergenceSize = config.M * config.K * MemoryLayout<Float>.stride

        guard let centroidsAccum = device.makeBuffer(length: accumSize, options: .storageModeShared),
              let centroidCounts = device.makeBuffer(length: countSize, options: .storageModeShared),
              let convergenceBuffer = device.makeBuffer(length: convergenceSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: accumSize + countSize + convergenceSize)
        }
        centroidsAccum.label = "PQ.accumulator"
        centroidCounts.label = "PQ.counts"
        convergenceBuffer.label = "PQ.convergence"

        let params = Metal4PQParams(config: config, count: count)

        // Training loop
        for iteration in 0..<config.trainIterations {
            // E-step: Assignment
            try await context.executeAndWait { [self] _, encoder in
                encoder.setComputePipelineState(assignmentPipeline)
                encoder.label = "PQ.Assignment"

                encoder.setBuffer(data, offset: 0, index: 0)
                encoder.setBuffer(codebooks, offset: 0, index: 1)
                encoder.setBuffer(assignments, offset: 0, index: 2)
                var p = params
                encoder.setBytes(&p, length: MemoryLayout<Metal4PQParams>.size, index: 3)

                let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
                let gridSize = MTLSize(width: count, height: config.M, depth: 1)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            }

            // Clear accumulators
            memset(centroidsAccum.contents(), 0, accumSize)
            memset(centroidCounts.contents(), 0, countSize)

            // M-step: Update centroids
            try await context.executeAndWait { [self] _, encoder in
                // Accumulate
                encoder.setComputePipelineState(updateAccumulatePipeline)
                encoder.label = "PQ.UpdateAccumulate"

                encoder.setBuffer(data, offset: 0, index: 0)
                encoder.setBuffer(assignments, offset: 0, index: 1)
                encoder.setBuffer(centroidsAccum, offset: 0, index: 2)
                encoder.setBuffer(centroidCounts, offset: 0, index: 3)
                var p1 = params
                encoder.setBytes(&p1, length: MemoryLayout<Metal4PQParams>.size, index: 4)

                let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
                let gridSize = MTLSize(width: count, height: config.M, depth: 1)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

                // Barrier
                encoder.memoryBarrier(scope: .buffers)

                // Finalize
                encoder.setComputePipelineState(updateFinalizePipeline)
                encoder.label = "PQ.UpdateFinalize"

                encoder.setBuffer(codebooks, offset: 0, index: 0)
                encoder.setBuffer(centroidsAccum, offset: 0, index: 1)
                encoder.setBuffer(centroidCounts, offset: 0, index: 2)
                encoder.setBuffer(convergenceBuffer, offset: 0, index: 3)
                var p2 = params
                encoder.setBytes(&p2, length: MemoryLayout<Metal4PQParams>.size, index: 4)

                let finalizeGrid = MTLSize(width: config.K, height: config.M, depth: 1)
                encoder.dispatchThreads(finalizeGrid, threadsPerThreadgroup: threadgroupSize)
            }

            // Check convergence
            let movement = calculateTotalMovement(convergenceBuffer: convergenceBuffer, config: config)
            progressHandler?(iteration, movement)

            if movement < config.convergenceThreshold {
                break
            }
        }

        return Metal4PQModel(codebooks: codebooks, config: config, device: device)
    }

    // MARK: - Encoding

    /// Encode vectors using trained model.
    public func encode(
        vectors: any MTLBuffer,
        count: Int,
        model: Metal4PQModel
    ) async throws -> Metal4EncodedVectors {
        let device = context.device.rawDevice
        let config = model.config

        let codesSize = count * config.M * MemoryLayout<UInt8>.stride
        guard let codes = device.makeBuffer(length: codesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: codesSize)
        }
        codes.label = "PQ.codes"

        let params = Metal4PQParams(config: config, count: count)

        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(assignmentPipeline)
            encoder.label = "PQ.Encode"

            encoder.setBuffer(vectors, offset: 0, index: 0)
            encoder.setBuffer(model.codebooks, offset: 0, index: 1)
            encoder.setBuffer(codes, offset: 0, index: 2)
            var p = params
            encoder.setBytes(&p, length: MemoryLayout<Metal4PQParams>.size, index: 3)

            let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let gridSize = MTLSize(width: count, height: config.M, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        }

        return Metal4EncodedVectors(codes: codes, count: count, config: config)
    }

    // MARK: - Distance Computation

    /// Compute approximate distances using ADC.
    public func computeDistances(
        query: any MTLBuffer,
        encodedVectors: Metal4EncodedVectors,
        model: Metal4PQModel
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let config = model.config

        // Allocate distance table [M × K]
        let tableSize = config.M * config.K * MemoryLayout<Float>.stride
        guard let distanceTable = device.makeBuffer(length: tableSize, options: .storageModePrivate) else {
            throw VectorError.bufferAllocationFailed(size: tableSize)
        }
        distanceTable.label = "PQ.distanceTable"

        // Allocate output distances
        let distancesSize = encodedVectors.count * MemoryLayout<Float>.stride
        guard let distances = device.makeBuffer(length: distancesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: distancesSize)
        }
        distances.label = "PQ.distances"

        let params = Metal4PQParams(config: config, count: encodedVectors.count)

        try await context.executeAndWait { [self] _, encoder in
            // Step 1: Precompute distance table
            encoder.setComputePipelineState(precomputeDistancePipeline)
            encoder.label = "PQ.PrecomputeDistanceTable"

            encoder.setBuffer(query, offset: 0, index: 0)
            encoder.setBuffer(model.codebooks, offset: 0, index: 1)
            encoder.setBuffer(distanceTable, offset: 0, index: 2)
            var p1 = params
            encoder.setBytes(&p1, length: MemoryLayout<Metal4PQParams>.size, index: 3)

            let tableThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            let tableGrid = MTLSize(width: config.K, height: config.M, depth: 1)
            encoder.dispatchThreads(tableGrid, threadsPerThreadgroup: tableThreadgroup)

            // Barrier
            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Compute distances via lookup
            encoder.setComputePipelineState(computeDistancesPipeline)
            encoder.label = "PQ.ComputeDistances"

            encoder.setBuffer(encodedVectors.codes, offset: 0, index: 0)
            encoder.setBuffer(distanceTable, offset: 0, index: 1)
            encoder.setBuffer(distances, offset: 0, index: 2)
            var p2 = params
            encoder.setBytes(&p2, length: MemoryLayout<Metal4PQParams>.size, index: 3)

            // Shared memory for distance table
            let sharedMemorySize = config.M * config.K * MemoryLayout<Float>.stride
            encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)

            let distConfig = Metal4ThreadConfiguration.linear(
                count: encodedVectors.count,
                pipeline: computeDistancesPipeline
            )
            encoder.dispatchThreadgroups(
                distConfig.threadgroups,
                threadsPerThreadgroup: distConfig.threadsPerThreadgroup
            )
        }

        return distances
    }

    // MARK: - High-Level API

    /// Train and encode in one call.
    public func trainAndEncode(
        data: [[Float]],
        config: Metal4PQConfig
    ) async throws -> (model: Metal4PQModel, encoded: Metal4EncodedVectors) {
        let device = context.device.rawDevice

        let flatData = data.flatMap { $0 }
        guard let dataBuffer = device.makeBuffer(
            bytes: flatData,
            length: flatData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatData.count * MemoryLayout<Float>.size)
        }
        dataBuffer.label = "PQ.trainingData"

        let model = try await train(data: dataBuffer, count: data.count, config: config)
        let encoded = try await encode(vectors: dataBuffer, count: data.count, model: model)

        return (model, encoded)
    }

    /// Find approximate nearest neighbors.
    public func findNearestNeighbors(
        query: [Float],
        encodedDatabase: Metal4EncodedVectors,
        model: Metal4PQModel,
        k: Int
    ) async throws -> [(index: Int, distance: Float)] {
        let device = context.device.rawDevice

        guard let queryBuffer = device.makeBuffer(
            bytes: query,
            length: query.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: query.count * MemoryLayout<Float>.size)
        }
        queryBuffer.label = "PQ.query"

        let distances = try await computeDistances(
            query: queryBuffer,
            encodedVectors: encodedDatabase,
            model: model
        )

        // Extract and sort
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: encodedDatabase.count)
        var indexed: [(index: Int, distance: Float)] = []
        indexed.reserveCapacity(encodedDatabase.count)

        for i in 0..<encodedDatabase.count {
            indexed.append((index: i, distance: sqrt(distPtr[i])))  // Convert from squared
        }

        indexed.sort { $0.distance < $1.distance }
        return Array(indexed.prefix(k))
    }

    // MARK: - Private Helpers

    private func initializeCodebooks(
        codebooks: any MTLBuffer,
        data: any MTLBuffer,
        count: Int,
        config: Metal4PQConfig
    ) {
        let codebooksPtr = codebooks.contents().bindMemory(
            to: Float.self,
            capacity: config.M * config.K * config.D_sub
        )
        let dataPtr = data.contents().bindMemory(
            to: Float.self,
            capacity: count * config.dimension
        )

        // Random initialization from data
        for m in 0..<config.M {
            for k in 0..<config.K {
                let randomIdx = Int.random(in: 0..<count)
                let srcStart = randomIdx * config.dimension + m * config.D_sub
                let dstStart = (m * config.K + k) * config.D_sub

                for d in 0..<config.D_sub {
                    codebooksPtr[dstStart + d] = dataPtr[srcStart + d]
                }
            }
        }
    }

    private func calculateTotalMovement(
        convergenceBuffer: any MTLBuffer,
        config: Metal4PQConfig
    ) -> Float {
        let ptr = convergenceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: config.M * config.K
        )
        var total: Float = 0

        for i in 0..<(config.M * config.K) {
            total += ptr[i]
        }

        return sqrt(total / Float(config.M * config.K))
    }
}
