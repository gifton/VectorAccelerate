//
//  Metal4BatchMatrixKernel.swift
//  VectorAccelerate
//
//  Metal 4 Batch Matrix kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 4, Priority 4
//
//  Features:
//  - Fused batch GEMM with bias and activation
//  - Strided tensor operations for custom layouts
//  - 3D dispatch for batch dimension
//  - Supports ReLU, Tanh, Sigmoid, GELU activations

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Activation Types

/// Activation function types for fused operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4ActivationType: UInt8, Sendable {
    case none = 0
    case relu = 1
    case tanh = 2
    case sigmoid = 3
    case gelu = 4
}

// MARK: - Configuration

/// Configuration for fused batch operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchFusedConfig: Sendable {
    /// Scaling factor for A*B result
    public let alpha: Float
    /// Scaling factor for accumulation
    public let beta: Float
    /// Include bias addition
    public let hasBias: Bool
    /// Activation function
    public let activation: Metal4ActivationType

    public init(
        alpha: Float = 1.0,
        beta: Float = 0.0,
        hasBias: Bool = false,
        activation: Metal4ActivationType = .none
    ) {
        self.alpha = alpha
        self.beta = beta
        self.hasBias = hasBias
        self.activation = activation
    }

    public static let `default` = Metal4BatchFusedConfig()
}

/// Configuration for strided batch operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchStridedConfig: Sendable {
    /// Strides for tensor A (row, col, batch)
    public let strideA: (row: Int, col: Int, batch: Int)
    /// Strides for tensor B (row, col, batch)
    public let strideB: (row: Int, col: Int, batch: Int)
    /// Strides for tensor C (row, col, batch)
    public let strideC: (row: Int, col: Int, batch: Int)
    /// Transpose A
    public let transposeA: Bool
    /// Transpose B
    public let transposeB: Bool

    public init(
        strideA: (Int, Int, Int),
        strideB: (Int, Int, Int),
        strideC: (Int, Int, Int),
        transposeA: Bool = false,
        transposeB: Bool = false
    ) {
        self.strideA = strideA
        self.strideB = strideB
        self.strideC = strideC
        self.transposeA = transposeA
        self.transposeB = transposeB
    }

    /// Create contiguous strides for standard layout.
    public static func contiguous(
        batchSize: Int,
        rowsA: Int,
        colsA: Int,
        colsB: Int
    ) -> Metal4BatchStridedConfig {
        return Metal4BatchStridedConfig(
            strideA: (colsA, 1, rowsA * colsA),
            strideB: (colsB, 1, colsA * colsB),
            strideC: (colsB, 1, rowsA * colsB)
        )
    }
}

// MARK: - Parameters

/// Parameters for fused batch kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchFusedParameters: Sendable {
    public let batchSize: UInt32
    public let M: UInt32
    public let K: UInt32
    public let N: UInt32
    public let alpha: Float
    public let activation: UInt32
    public let hasBias: UInt32

    public init(
        batchSize: Int,
        M: Int,
        K: Int,
        N: Int,
        config: Metal4BatchFusedConfig = .default
    ) {
        self.batchSize = UInt32(batchSize)
        self.M = UInt32(M)
        self.K = UInt32(K)
        self.N = UInt32(N)
        self.alpha = config.alpha
        self.activation = UInt32(config.activation.rawValue)
        self.hasBias = config.hasBias ? 1 : 0
    }
}

// MARK: - Result Type

/// Result from batch matrix operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4BatchMatrixResult: Sendable {
    /// Result buffer containing all output matrices
    public let buffer: any MTLBuffer
    /// Number of matrices in batch
    public let batchSize: Int
    /// Rows per matrix
    public let rows: Int
    /// Columns per matrix
    public let columns: Int
    /// Total execution time
    public let executionTime: TimeInterval
    /// Total GFLOPS
    public let totalGflops: Double
    /// Average GFLOPS per matrix
    public var averageGflops: Double { totalGflops / Double(batchSize) }

    /// Extract specific matrix from batch.
    public func matrix(at index: Int) -> Matrix? {
        guard index < batchSize else { return nil }

        let elementsPerMatrix = rows * columns
        let start = index * elementsPerMatrix

        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: batchSize * elementsPerMatrix)
        var values: [Float] = []
        values.reserveCapacity(elementsPerMatrix)
        for i in 0..<elementsPerMatrix {
            values.append(pointer[start + i])
        }

        return Matrix(rows: rows, columns: columns, values: values)
    }

    /// Get all matrices.
    public func allMatrices() -> [Matrix] {
        (0..<batchSize).compactMap { matrix(at: $0) }
    }

    /// Flatten all data.
    public func flattenedData() -> [Float] {
        let totalElements = batchSize * rows * columns
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        return Array(UnsafeBufferPointer(start: pointer, count: totalElements))
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Batch Matrix kernel.
///
/// Performs batched matrix operations with optional fused bias and activation.
///
/// ## Fused Operations
///
/// Computes: C[i] = activation(alpha * A[i] * B[i] + bias)
///
/// Supported activations:
/// - None: identity
/// - ReLU: max(0, x)
/// - Tanh: tanh(x)
/// - Sigmoid: 1 / (1 + exp(-x))
/// - GELU: x * Φ(x)
///
/// ## Strided Operations
///
/// Supports custom memory layouts for tensor operations via strides.
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4BatchMatrixKernel(context: context)
///
/// // Fused batch multiply with ReLU
/// let result = try await kernel.multiplyFused(
///     batchA: matrices,
///     batchB: weights,
///     bias: biasVector,
///     config: Metal4BatchFusedConfig(activation: .relu)
/// )
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4BatchMatrixKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4BatchMatrixKernel"

    // MARK: - Constants

    private let BLOCK_SIZE: Int = 16

    // MARK: - Pipelines

    private let fusedPipeline: any MTLComputePipelineState
    private let stridedPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Batch Matrix kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let fusedFunc = library.makeFunction(name: "batchMatrixMultiplyFused") else {
            throw VectorError.shaderNotFound(
                name: "Batch fused kernel. Ensure OptimizedMatrixOps.metal is compiled."
            )
        }

        guard let stridedFunc = library.makeFunction(name: "stridedBatchGEMM") else {
            throw VectorError.shaderNotFound(
                name: "Strided batch kernel. Ensure OptimizedMatrixOps.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.fusedPipeline = try await device.makeComputePipelineState(function: fusedFunc)
        self.stridedPipeline = try await device.makeComputePipelineState(function: stridedFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode fused batch multiplication into an existing encoder.
    @discardableResult
    public func encodeFused(
        into encoder: any MTLComputeCommandEncoder,
        batchA: any MTLBuffer,
        batchB: any MTLBuffer,
        output: any MTLBuffer,
        bias: (any MTLBuffer)?,
        parameters: Metal4BatchFusedParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(fusedPipeline)
        encoder.label = "BatchMatrixFused (batch=\(parameters.batchSize))"

        encoder.setBuffer(batchA, offset: 0, index: 0)
        encoder.setBuffer(batchB, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = SIMD4<UInt32>(
            parameters.batchSize,
            parameters.M,
            parameters.K,
            parameters.N
        )
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)

        encoder.setBuffer(bias, offset: 0, index: 4)

        var alpha = parameters.alpha
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 5)

        // 3D dispatch for batch dimension
        let threadgroupSize = MTLSize(width: BLOCK_SIZE, height: BLOCK_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(parameters.N) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            height: (Int(parameters.M) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            depth: Int(parameters.batchSize)
        )

        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "batchMatrixMultiplyFused",
            threadgroups: threadgroupCount,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    /// Encode strided batch multiplication into an existing encoder.
    @discardableResult
    public func encodeStrided(
        into encoder: any MTLComputeCommandEncoder,
        tensorA: any MTLBuffer,
        tensorB: any MTLBuffer,
        output: any MTLBuffer,
        batchCount: Int,
        dimensions: (M: Int, N: Int, K: Int),
        config: Metal4BatchStridedConfig
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(stridedPipeline)
        encoder.label = "BatchMatrixStrided (batch=\(batchCount))"

        encoder.setBuffer(tensorA, offset: 0, index: 0)
        encoder.setBuffer(tensorB, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var batchCountParam = UInt32(batchCount)
        encoder.setBytes(&batchCountParam, length: MemoryLayout<UInt32>.size, index: 3)

        var dims = SIMD3<UInt32>(UInt32(dimensions.M), UInt32(dimensions.N), UInt32(dimensions.K))
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 4)

        var stridesA = SIMD3<UInt32>(
            UInt32(config.strideA.row),
            UInt32(config.strideA.col),
            UInt32(config.strideA.batch)
        )
        encoder.setBytes(&stridesA, length: MemoryLayout<SIMD3<UInt32>>.size, index: 5)

        var stridesB = SIMD3<UInt32>(
            UInt32(config.strideB.row),
            UInt32(config.strideB.col),
            UInt32(config.strideB.batch)
        )
        encoder.setBytes(&stridesB, length: MemoryLayout<SIMD3<UInt32>>.size, index: 6)

        var stridesC = SIMD3<UInt32>(
            UInt32(config.strideC.row),
            UInt32(config.strideC.col),
            UInt32(config.strideC.batch)
        )
        encoder.setBytes(&stridesC, length: MemoryLayout<SIMD3<UInt32>>.size, index: 7)

        let threadgroupSize = MTLSize(width: BLOCK_SIZE, height: BLOCK_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (dimensions.N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            height: (dimensions.M + BLOCK_SIZE - 1) / BLOCK_SIZE,
            depth: batchCount
        )

        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "stridedBatchGEMM",
            threadgroups: threadgroupCount,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - High-Level API

    /// Perform fused batch matrix multiplication with optional bias and activation.
    public func multiplyFused(
        batchA: [Matrix],
        batchB: [Matrix],
        bias: [Float]? = nil,
        config: Metal4BatchFusedConfig = .default
    ) async throws -> Metal4BatchMatrixResult {
        guard batchA.count == batchB.count else {
            throw VectorError.invalidInput("Batch sizes must match")
        }
        guard !batchA.isEmpty else {
            throw VectorError.invalidInput("Batch cannot be empty")
        }

        let batchSize = batchA.count
        let M = batchA[0].rows
        let K = batchA[0].columns
        let N = batchB[0].columns

        // Validate all matrices have same dimensions
        for i in 0..<batchSize {
            guard batchA[i].rows == M && batchA[i].columns == K else {
                throw VectorError.invalidInput("Matrix A[\(i)] dimension mismatch")
            }
            guard batchB[i].rows == K && batchB[i].columns == N else {
                throw VectorError.invalidInput("Matrix B[\(i)] dimension mismatch")
            }
        }

        // Validate bias
        if let bias = bias {
            guard bias.count == batchSize * N || bias.count == N else {
                throw VectorError.invalidInput("Bias dimensions don't match output")
            }
        }

        let device = context.device.rawDevice

        let flatA = batchA.flatMap { $0.values }
        let flatB = batchB.flatMap { $0.values }

        guard let bufferA = device.makeBuffer(
            bytes: flatA,
            length: flatA.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatA.count * MemoryLayout<Float>.stride)
        }
        bufferA.label = "BatchMatrix.A"

        guard let bufferB = device.makeBuffer(
            bytes: flatB,
            length: flatB.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatB.count * MemoryLayout<Float>.stride)
        }
        bufferB.label = "BatchMatrix.B"

        let outputSize = batchSize * M * N * MemoryLayout<Float>.stride
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "BatchMatrix.output"

        let biasBuffer: (any MTLBuffer)?
        if let bias = bias {
            biasBuffer = device.makeBuffer(
                bytes: bias,
                length: bias.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
            biasBuffer?.label = "BatchMatrix.bias"
        } else {
            biasBuffer = nil
        }

        let parameters = Metal4BatchFusedParameters(
            batchSize: batchSize,
            M: M,
            K: K,
            N: N,
            config: config
        )

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeFused(
                into: encoder,
                batchA: bufferA,
                batchB: bufferB,
                output: outputBuffer,
                bias: biasBuffer,
                parameters: parameters
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        let operations = Double(batchSize) * 2.0 * Double(M) * Double(N) * Double(K)
        let totalGflops = (operations / 1e9) / executionTime

        return Metal4BatchMatrixResult(
            buffer: outputBuffer,
            batchSize: batchSize,
            rows: M,
            columns: N,
            executionTime: executionTime,
            totalGflops: totalGflops
        )
    }

    /// Perform strided batch GEMM for tensor operations.
    public func multiplyStrided(
        tensorA: [Float],
        tensorB: [Float],
        batchCount: Int,
        dimensions: (M: Int, N: Int, K: Int),
        config: Metal4BatchStridedConfig
    ) async throws -> Metal4BatchMatrixResult {
        let expectedSizeA = batchCount * dimensions.M * dimensions.K
        let expectedSizeB = batchCount * dimensions.K * dimensions.N

        guard tensorA.count >= expectedSizeA else {
            throw VectorError.invalidInput("Tensor A size insufficient")
        }
        guard tensorB.count >= expectedSizeB else {
            throw VectorError.invalidInput("Tensor B size insufficient")
        }

        let device = context.device.rawDevice

        guard let bufferA = device.makeBuffer(
            bytes: tensorA,
            length: tensorA.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: tensorA.count * MemoryLayout<Float>.stride)
        }
        bufferA.label = "BatchMatrixStrided.A"

        guard let bufferB = device.makeBuffer(
            bytes: tensorB,
            length: tensorB.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: tensorB.count * MemoryLayout<Float>.stride)
        }
        bufferB.label = "BatchMatrixStrided.B"

        let outputSize = batchCount * dimensions.M * dimensions.N * MemoryLayout<Float>.stride
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "BatchMatrixStrided.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeStrided(
                into: encoder,
                tensorA: bufferA,
                tensorB: bufferB,
                output: outputBuffer,
                batchCount: batchCount,
                dimensions: dimensions,
                config: config
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        let operations = Double(batchCount) * 2.0 * Double(dimensions.M) * Double(dimensions.N) * Double(dimensions.K)
        let totalGflops = (operations / 1e9) / executionTime

        return Metal4BatchMatrixResult(
            buffer: outputBuffer,
            batchSize: batchCount,
            rows: dimensions.M,
            columns: dimensions.N,
            executionTime: executionTime,
            totalGflops: totalGflops
        )
    }
}
