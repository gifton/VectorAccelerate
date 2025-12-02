//
//  MatrixMultiplyKernel.swift
//  VectorAccelerate
//
//  Metal 4 Matrix Multiply kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 4, Priority 4
//
//  Features:
//  - Tiled GEMM with 32x32x8 tiling strategy
//  - Dimension-specific optimized pipelines
//  - GEMM: C = alpha * A * B + beta * C
//  - Supports kernel fusion via encode() API

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for matrix multiplication.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4MatrixMultiplyConfig: Sendable {
    /// Scaling factor for A*B result
    public let alpha: Float
    /// Scaling factor for initial C
    public let beta: Float
    /// Transpose matrix A before multiply
    public let transposeA: Bool
    /// Transpose matrix B before multiply
    public let transposeB: Bool

    public init(
        alpha: Float = 1.0,
        beta: Float = 0.0,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) {
        self.alpha = alpha
        self.beta = beta
        self.transposeA = transposeA
        self.transposeB = transposeB
    }

    public static let `default` = Metal4MatrixMultiplyConfig()
}

// MARK: - Parameters

/// Parameters for matrix multiply kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct MatrixMultiplyParameters: Sendable {
    /// Number of rows in A (or A^T)
    public let M: UInt32
    /// Shared dimension (cols of A, rows of B)
    public let K: UInt32
    /// Number of columns in B (or B^T)
    public let N: UInt32
    /// Alpha scaling factor
    public let alpha: Float
    /// Beta scaling factor
    public let beta: Float
    /// Flags: bit 0 = transposeA, bit 1 = transposeB
    public let flags: UInt32

    public init(
        M: Int,
        K: Int,
        N: Int,
        config: Metal4MatrixMultiplyConfig = .default
    ) {
        self.M = UInt32(M)
        self.K = UInt32(K)
        self.N = UInt32(N)
        self.alpha = config.alpha
        self.beta = config.beta
        self.flags = (config.transposeA ? 1 : 0) | (config.transposeB ? 2 : 0)
    }
}

// MARK: - Result Type

/// Result from matrix multiplication.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4MatrixMultiplyResult: Sendable {
    /// Result matrix as buffer [M × N]
    public let buffer: any MTLBuffer
    /// Number of rows
    public let rows: Int
    /// Number of columns
    public let columns: Int
    /// Execution time in seconds
    public let executionTime: TimeInterval
    /// Performance in GFLOPS
    public let gflops: Double

    /// Convert to Matrix type.
    public func asMatrix() -> Matrix {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: rows * columns)
        let values = Array(UnsafeBufferPointer(start: pointer, count: rows * columns))
        return Matrix(rows: rows, columns: columns, values: values)
    }

    /// Get element at (row, col).
    public func element(row: Int, col: Int) -> Float {
        guard row < rows && col < columns else { return 0 }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: rows * columns)
        return pointer[row * columns + col]
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Matrix Multiply kernel.
///
/// Performs General Matrix Multiplication (GEMM):
/// ```
/// C = alpha * A * B + beta * C
/// ```
///
/// ## Tiling Strategy
///
/// Uses 32×32×8 tiling optimized for Apple Silicon:
/// - Threadgroup size: 32×32 (1024 threads)
/// - Tile dimensions: 32×32 output tile
/// - K-dimension unrolling: 8 elements
///
/// ## Performance Optimizations
///
/// - Dimension-specific pipelines for 512, 768, 1536
/// - Shared memory tiling to reduce global memory access
/// - Bank conflict avoidance with padding
///
/// ## Usage
///
/// ```swift
/// let kernel = try await MatrixMultiplyKernel(context: context)
///
/// // Standard multiply: C = A × B
/// let result = try await kernel.multiply(matrixA, matrixB)
///
/// // GEMM: C = alpha * A × B + beta * C
/// let result = try await kernel.gemm(A, B, C, config: config)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class MatrixMultiplyKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "MatrixMultiplyKernel"

    // MARK: - Constants

    private let TILE_M: Int = 32
    private let TILE_N: Int = 32
    private let TILE_K: Int = 8

    // MARK: - Pipelines

    private let genericPipeline: any MTLComputePipelineState
    private let pipeline512: (any MTLComputePipelineState)?
    private let pipeline768: (any MTLComputePipelineState)?
    private let pipeline1536: (any MTLComputePipelineState)?

    // MARK: - Initialization

    /// Create a Metal 4 Matrix Multiply kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let genericFunc = library.makeFunction(name: "tiledMatrixMultiply") else {
            throw VectorError.shaderNotFound(
                name: "Matrix multiply kernel. Ensure OptimizedMatrixOps.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.genericPipeline = try await device.makeComputePipelineState(function: genericFunc)

        // Load dimension-specific pipelines (optional)
        if let func512 = library.makeFunction(name: "tiledMatrixMultiply_512") {
            self.pipeline512 = try await device.makeComputePipelineState(function: func512)
        } else {
            self.pipeline512 = nil
        }

        if let func768 = library.makeFunction(name: "tiledMatrixMultiply_768") {
            self.pipeline768 = try await device.makeComputePipelineState(function: func768)
        } else {
            self.pipeline768 = nil
        }

        if let func1536 = library.makeFunction(name: "tiledMatrixMultiply_1536") {
            self.pipeline1536 = try await device.makeComputePipelineState(function: func1536)
        } else {
            self.pipeline1536 = nil
        }
    }

    // MARK: - Pipeline Selection

    private func selectPipeline(for K: Int) -> any MTLComputePipelineState {
        switch K {
        case 512 where pipeline512 != nil:
            return pipeline512!
        case 768 where pipeline768 != nil:
            return pipeline768!
        case 1536 where pipeline1536 != nil:
            return pipeline1536!
        default:
            return genericPipeline
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode matrix multiplication into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        matrixA: any MTLBuffer,
        matrixB: any MTLBuffer,
        matrixC: any MTLBuffer,
        parameters: MatrixMultiplyParameters
    ) -> Metal4EncodingResult {
        let pipeline = selectPipeline(for: Int(parameters.K))

        encoder.setComputePipelineState(pipeline)
        encoder.label = "MatrixMultiply (\(parameters.M)×\(parameters.K)×\(parameters.N))"

        encoder.setBuffer(matrixA, offset: 0, index: 0)
        encoder.setBuffer(matrixB, offset: 0, index: 1)
        encoder.setBuffer(matrixC, offset: 0, index: 2)

        var dims = SIMD3<UInt32>(parameters.M, parameters.K, parameters.N)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 3)

        // Configure threadgroups
        let threadgroupSize = MTLSize(width: TILE_N, height: TILE_M, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(parameters.N) + TILE_N - 1) / TILE_N,
            height: (Int(parameters.M) + TILE_M - 1) / TILE_M,
            depth: 1
        )

        // Allocate shared memory for tiling
        // sharedA stores TILE_M × TILE_K elements (rows of A tile)
        // sharedB stores TILE_K × TILE_N elements (cols of B tile)
        let sharedASizeBytes = TILE_M * TILE_K * MemoryLayout<Float>.stride
        let sharedBSizeBytes = TILE_K * TILE_N * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedASizeBytes, index: 0)
        encoder.setThreadgroupMemoryLength(sharedBSizeBytes, index: 1)

        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: "tiledMatrixMultiply",
            threadgroups: threadgroupCount,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Execute matrix multiplication as standalone operation.
    public func execute(
        matrixA: any MTLBuffer,
        matrixB: any MTLBuffer,
        parameters: MatrixMultiplyParameters
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let outputSize = Int(parameters.M) * Int(parameters.N) * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "MatrixMultiply.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                matrixA: matrixA,
                matrixB: matrixB,
                matrixC: outputBuffer,
                parameters: parameters
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Multiply two matrices: C = A × B
    public func multiply(
        _ matrixA: Matrix,
        _ matrixB: Matrix,
        config: Metal4MatrixMultiplyConfig = .default
    ) async throws -> Metal4MatrixMultiplyResult {
        // Validate dimensions
        let effectiveColsA = config.transposeA ? matrixA.rows : matrixA.columns
        let effectiveRowsB = config.transposeB ? matrixB.columns : matrixB.rows

        guard effectiveColsA == effectiveRowsB else {
            throw VectorError.countMismatch(expected: effectiveColsA, actual: effectiveRowsB)
        }

        let M = config.transposeA ? matrixA.columns : matrixA.rows
        let K = effectiveColsA
        let N = config.transposeB ? matrixB.rows : matrixB.columns

        let device = context.device.rawDevice

        guard let bufferA = device.makeBuffer(
            bytes: matrixA.values,
            length: matrixA.values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: matrixA.values.count * MemoryLayout<Float>.stride)
        }
        bufferA.label = "MatrixMultiply.A"

        guard let bufferB = device.makeBuffer(
            bytes: matrixB.values,
            length: matrixB.values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: matrixB.values.count * MemoryLayout<Float>.stride)
        }
        bufferB.label = "MatrixMultiply.B"

        let parameters = MatrixMultiplyParameters(M: M, K: K, N: N, config: config)

        let startTime = CACurrentMediaTime()
        let outputBuffer = try await execute(matrixA: bufferA, matrixB: bufferB, parameters: parameters)
        let executionTime = CACurrentMediaTime() - startTime

        // Apply alpha scaling if needed
        if config.alpha != 1.0 {
            let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: M * N)
            for i in 0..<(M * N) {
                pointer[i] *= config.alpha
            }
        }

        let operations = 2.0 * Double(M) * Double(N) * Double(K)
        let gflops = (operations / 1e9) / executionTime

        return Metal4MatrixMultiplyResult(
            buffer: outputBuffer,
            rows: M,
            columns: N,
            executionTime: executionTime,
            gflops: gflops
        )
    }

    /// General Matrix Multiply: C = alpha * A × B + beta * C
    public func gemm(
        _ matrixA: Matrix,
        _ matrixB: Matrix,
        _ matrixC: Matrix,
        config: Metal4MatrixMultiplyConfig = .default
    ) async throws -> Metal4MatrixMultiplyResult {
        let M = config.transposeA ? matrixA.columns : matrixA.rows
        let N = config.transposeB ? matrixB.rows : matrixB.columns

        guard matrixC.rows == M && matrixC.columns == N else {
            throw VectorError.invalidInput("Matrix C dimensions must be \(M)×\(N)")
        }

        // Compute A × B
        let abResult = try await multiply(
            matrixA,
            matrixB,
            config: Metal4MatrixMultiplyConfig(
                alpha: 1.0,
                beta: 0.0,
                transposeA: config.transposeA,
                transposeB: config.transposeB
            )
        )

        // Apply GEMM formula: C = alpha * (A×B) + beta * C
        let pointer = abResult.buffer.contents().bindMemory(to: Float.self, capacity: M * N)
        for i in 0..<(M * N) {
            pointer[i] = config.alpha * pointer[i] + config.beta * matrixC.values[i]
        }

        return Metal4MatrixMultiplyResult(
            buffer: abResult.buffer,
            rows: M,
            columns: N,
            executionTime: abResult.executionTime,
            gflops: abResult.gflops
        )
    }

    /// Multiply from flat arrays.
    public func multiply(
        matrixA: [Float],
        rowsA: Int,
        colsA: Int,
        matrixB: [Float],
        rowsB: Int,
        colsB: Int,
        config: Metal4MatrixMultiplyConfig = .default
    ) async throws -> Metal4MatrixMultiplyResult {
        let matA = Matrix(rows: rowsA, columns: colsA, values: matrixA)
        let matB = Matrix(rows: rowsB, columns: colsB, values: matrixB)
        return try await multiply(matA, matB, config: config)
    }

    /// Multiply multiple matrix pairs in batch.
    public func multiplyBatch(
        matricesA: [Matrix],
        matricesB: [Matrix],
        config: Metal4MatrixMultiplyConfig = .default
    ) async throws -> [Metal4MatrixMultiplyResult] {
        guard matricesA.count == matricesB.count else {
            throw VectorError.invalidInput("Matrix batch sizes must match")
        }

        var results: [Metal4MatrixMultiplyResult] = []
        for (matA, matB) in zip(matricesA, matricesB) {
            results.append(try await multiply(matA, matB, config: config))
        }

        return results
    }
}
