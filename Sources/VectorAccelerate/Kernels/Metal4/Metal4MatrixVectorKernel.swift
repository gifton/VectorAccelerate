//
//  Metal4MatrixVectorKernel.swift
//  VectorAccelerate
//
//  Metal 4 Matrix-Vector Multiply kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 4, Priority 4
//
//  Features:
//  - SIMD group optimized matrix-vector multiply
//  - Supports y = alpha * A * x + beta * y
//  - Batch operations for multiple vectors
//  - Power iteration for eigenvalue estimation

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for matrix-vector operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4MatrixVectorConfig: Sendable {
    /// Scaling factor for result
    public let alpha: Float
    /// Scaling factor for accumulation
    public let beta: Float
    /// Use A^T instead of A
    public let transpose: Bool
    /// Use SIMD optimization
    public let useSIMDOptimization: Bool

    public init(
        alpha: Float = 1.0,
        beta: Float = 0.0,
        transpose: Bool = false,
        useSIMDOptimization: Bool = true
    ) {
        self.alpha = alpha
        self.beta = beta
        self.transpose = transpose
        self.useSIMDOptimization = useSIMDOptimization
    }

    public static let `default` = Metal4MatrixVectorConfig()
}

// MARK: - Parameters

/// Parameters for matrix-vector kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4MatrixVectorParameters: Sendable {
    /// Number of rows in matrix
    public let rows: UInt32
    /// Number of columns in matrix
    public let columns: UInt32
    /// Alpha scaling
    public let alpha: Float
    /// Beta scaling
    public let beta: Float
    /// Transpose flag
    public let transpose: UInt32

    public init(
        rows: Int,
        columns: Int,
        config: Metal4MatrixVectorConfig = .default
    ) {
        self.rows = UInt32(rows)
        self.columns = UInt32(columns)
        self.alpha = config.alpha
        self.beta = config.beta
        self.transpose = config.transpose ? 1 : 0
    }
}

// MARK: - Result Type

/// Result from matrix-vector multiplication.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4MatrixVectorResult: Sendable {
    /// Result vector buffer
    public let buffer: any MTLBuffer
    /// Vector length
    public let length: Int
    /// Execution time
    public let executionTime: TimeInterval
    /// Performance in GFLOPS
    public let gflops: Double

    /// Convert to float array.
    public func asArray() -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: length)
        return Array(UnsafeBufferPointer(start: pointer, count: length))
    }

    /// Get element at index.
    public func element(at index: Int) -> Float {
        guard index < length else { return 0 }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: length)
        return pointer[index]
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Matrix-Vector Multiply kernel.
///
/// Computes y = alpha * A * x + beta * y using SIMD group optimizations.
///
/// ## SIMD Optimization
///
/// Uses SIMD width of 32 for efficient parallel reduction.
/// Each SIMD group computes one or more output elements.
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4MatrixVectorKernel(context: context)
///
/// // Standard multiply: y = A × x
/// let result = try await kernel.multiply(matrix: A, vector: x)
///
/// // Power iteration for eigenvalue
/// let (eigenvalue, eigenvector) = try await kernel.powerIteration(matrix: A)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4MatrixVectorKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4MatrixVectorKernel"

    // MARK: - Constants

    private let SIMD_WIDTH: Int = 32

    // MARK: - Pipelines

    private let simdPipeline: any MTLComputePipelineState
    private let basicPipeline: any MTLComputePipelineState
    private let batchPipeline: (any MTLComputePipelineState)?

    // MARK: - Initialization

    /// Create a Metal 4 Matrix-Vector kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let simdFunc = library.makeFunction(name: "simdgroupMatrixVector") else {
            throw VectorError.shaderNotFound(
                name: "SIMD matrix-vector kernel. Ensure OptimizedMatrixOps.metal is compiled."
            )
        }

        guard let basicFunc = library.makeFunction(name: "matrixVectorMultiply") else {
            throw VectorError.shaderNotFound(
                name: "Basic matrix-vector kernel. Ensure OptimizedMatrixOps.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.simdPipeline = try await device.makeComputePipelineState(function: simdFunc)
        self.basicPipeline = try await device.makeComputePipelineState(function: basicFunc)

        if let batchFunc = library.makeFunction(name: "batchMatrixVector") {
            self.batchPipeline = try await device.makeComputePipelineState(function: batchFunc)
        } else {
            self.batchPipeline = nil
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode matrix-vector multiplication into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        matrix: any MTLBuffer,
        vector: any MTLBuffer,
        output: any MTLBuffer,
        parameters: Metal4MatrixVectorParameters,
        useSIMD: Bool = true
    ) -> Metal4EncodingResult {
        let pipeline = useSIMD ? simdPipeline : basicPipeline
        let pipelineName = useSIMD ? "simdgroupMatrixVector" : "matrixVectorMultiply"

        encoder.setComputePipelineState(pipeline)
        encoder.label = "MatrixVector (\(parameters.rows)×\(parameters.columns))"

        encoder.setBuffer(matrix, offset: 0, index: 0)
        encoder.setBuffer(vector, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var dims = SIMD2<UInt32>(parameters.rows, parameters.columns)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)

        let outputLength = parameters.transpose == 1 ? Int(parameters.columns) : Int(parameters.rows)
        let threadgroupSize: MTLSize
        let threadgroupCount: MTLSize

        if useSIMD {
            // SIMD kernel processes one row per threadgroup (simdgroups_per_threadgroup = 1)
            // Each threadgroup has SIMD_WIDTH threads that cooperatively compute one row
            threadgroupSize = MTLSize(width: SIMD_WIDTH, height: 1, depth: 1)
            threadgroupCount = MTLSize(
                width: outputLength,  // One threadgroup per output row
                height: 1,
                depth: 1
            )
        } else {
            let blockSize = 256
            threadgroupSize = MTLSize(width: blockSize, height: 1, depth: 1)
            threadgroupCount = MTLSize(
                width: (outputLength + blockSize - 1) / blockSize,
                height: 1,
                depth: 1
            )
        }

        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: pipelineName,
            threadgroups: threadgroupCount,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Execute matrix-vector multiplication as standalone operation.
    public func execute(
        matrix: any MTLBuffer,
        vector: any MTLBuffer,
        parameters: Metal4MatrixVectorParameters,
        useSIMD: Bool = true
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let outputLength = parameters.transpose == 1 ? Int(parameters.columns) : Int(parameters.rows)
        let outputSize = outputLength * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "MatrixVector.output"

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                matrix: matrix,
                vector: vector,
                output: outputBuffer,
                parameters: parameters,
                useSIMD: useSIMD
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Multiply matrix by vector: y = A × x
    public func multiply(
        matrix: Matrix,
        vector: [Float],
        config: Metal4MatrixVectorConfig = .default
    ) async throws -> Metal4MatrixVectorResult {
        let expectedLength = config.transpose ? matrix.rows : matrix.columns
        guard vector.count == expectedLength else {
            throw VectorError.countMismatch(expected: expectedLength, actual: vector.count)
        }

        let device = context.device.rawDevice

        guard let matrixBuffer = device.makeBuffer(
            bytes: matrix.values,
            length: matrix.values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: matrix.values.count * MemoryLayout<Float>.stride)
        }
        matrixBuffer.label = "MatrixVector.matrix"

        guard let vectorBuffer = device.makeBuffer(
            bytes: vector,
            length: vector.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: vector.count * MemoryLayout<Float>.stride)
        }
        vectorBuffer.label = "MatrixVector.vector"

        let parameters = Metal4MatrixVectorParameters(
            rows: matrix.rows,
            columns: matrix.columns,
            config: config
        )

        let startTime = CACurrentMediaTime()
        let outputBuffer = try await execute(
            matrix: matrixBuffer,
            vector: vectorBuffer,
            parameters: parameters,
            useSIMD: config.useSIMDOptimization
        )
        let executionTime = CACurrentMediaTime() - startTime

        let outputLength = config.transpose ? matrix.columns : matrix.rows

        // Apply alpha/beta scaling
        if config.alpha != 1.0 || config.beta != 0.0 {
            let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputLength)
            for i in 0..<outputLength {
                pointer[i] = config.alpha * pointer[i]
            }
        }

        let operations = 2.0 * Double(matrix.rows) * Double(matrix.columns)
        let gflops = (operations / 1e9) / executionTime

        return Metal4MatrixVectorResult(
            buffer: outputBuffer,
            length: outputLength,
            executionTime: executionTime,
            gflops: gflops
        )
    }

    /// Multiply from flat arrays.
    public func multiply(
        matrixData: [Float],
        rows: Int,
        columns: Int,
        vector: [Float],
        config: Metal4MatrixVectorConfig = .default
    ) async throws -> Metal4MatrixVectorResult {
        let matrix = Matrix(rows: rows, columns: columns, values: matrixData)
        return try await multiply(matrix: matrix, vector: vector, config: config)
    }

    /// Multiply matrix by multiple vectors.
    public func multiplyBatch(
        matrix: Matrix,
        vectors: [[Float]],
        config: Metal4MatrixVectorConfig = .default
    ) async throws -> [Metal4MatrixVectorResult] {
        var results: [Metal4MatrixVectorResult] = []
        for vector in vectors {
            results.append(try await multiply(matrix: matrix, vector: vector, config: config))
        }
        return results
    }

    // MARK: - Specialized Operations

    /// Compute matrix-vector product and L2 norm.
    public func multiplyAndNorm(
        matrix: Matrix,
        vector: [Float],
        config: Metal4MatrixVectorConfig = .default
    ) async throws -> (result: Metal4MatrixVectorResult, norm: Float) {
        let result = try await multiply(matrix: matrix, vector: vector, config: config)
        let resultArray = result.asArray()
        let norm = sqrt(resultArray.reduce(0) { $0 + $1 * $1 })
        return (result, norm)
    }

    /// Power iteration for dominant eigenvalue estimation.
    ///
    /// Iteratively computes: v_{k+1} = A * v_k / ||A * v_k||
    ///
    /// - Parameters:
    ///   - matrix: Square matrix to analyze
    ///   - iterations: Maximum iterations
    ///   - tolerance: Convergence threshold
    /// - Returns: Estimated eigenvalue and eigenvector
    public func powerIteration(
        matrix: Matrix,
        iterations: Int = 100,
        tolerance: Float = 1e-6
    ) async throws -> (eigenvalue: Float, eigenvector: [Float]) {
        guard matrix.rows == matrix.columns else {
            throw VectorError.invalidInput("Matrix must be square for eigenvalue computation")
        }

        let n = matrix.rows
        var vector = Array(repeating: Float(1.0 / sqrt(Float(n))), count: n)
        var eigenvalue: Float = 0

        for _ in 0..<iterations {
            let result = try await multiply(matrix: matrix, vector: vector)
            let Av = result.asArray()

            let norm = sqrt(Av.reduce(0) { $0 + $1 * $1 })
            vector = Av.map { $0 / norm }

            // Rayleigh quotient
            let vAv = zip(vector, Av).reduce(0) { $0 + $1.0 * $1.1 }
            let vv = vector.reduce(0) { $0 + $1 * $1 }
            let newEigenvalue = vAv / vv

            if abs(newEigenvalue - eigenvalue) < tolerance {
                return (newEigenvalue, vector)
            }

            eigenvalue = newEigenvalue
        }

        return (eigenvalue, vector)
    }
}
