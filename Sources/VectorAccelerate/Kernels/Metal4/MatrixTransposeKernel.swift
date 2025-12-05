//
//  MatrixTransposeKernel.swift
//  VectorAccelerate
//
//  Metal 4 Matrix Transpose kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 4, Priority 4
//
//  Features:
//  - Tiled transpose with bank conflict avoidance
//  - 16x17 padding strategy for shared memory
//  - Optional in-place transpose for square matrices
//  - Batch transpose operations

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Configuration

/// Configuration for transpose operation.
public struct Metal4TransposeConfig: Sendable {
    /// Conjugate transpose (for complex numbers, future)
    public let conjugate: Bool
    /// Attempt in-place transpose if matrix is square
    public let inPlace: Bool

    public init(conjugate: Bool = false, inPlace: Bool = false) {
        self.conjugate = conjugate
        self.inPlace = inPlace
    }

    public static let `default` = Metal4TransposeConfig()
}

// MARK: - Parameters

/// Parameters for transpose kernel.
public struct TransposeParameters: Sendable {
    /// Number of rows in input matrix
    public let rows: UInt32
    /// Number of columns in input matrix
    public let columns: UInt32

    public init(rows: Int, columns: Int) {
        self.rows = UInt32(rows)
        self.columns = UInt32(columns)
    }
}

// MARK: - Result Type

/// Result from transpose operation.
public struct Metal4TransposeResult: Sendable {
    /// Transposed matrix buffer [columns × rows]
    public let buffer: any MTLBuffer
    /// Output rows (input columns)
    public let rows: Int
    /// Output columns (input rows)
    public let columns: Int
    /// Execution time
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s
    public let throughputGBps: Double

    /// Convert to Matrix type.
    public func asMatrix() -> Matrix {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: rows * columns)
        let values = Array(UnsafeBufferPointer(start: pointer, count: rows * columns))
        return Matrix(rows: rows, columns: columns, values: values)
    }

    /// Get element at (row, col) in transposed matrix.
    public func element(row: Int, col: Int) -> Float {
        guard row < rows && col < columns else { return 0 }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: rows * columns)
        return pointer[row * columns + col]
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Matrix Transpose kernel.
///
/// Transposes a matrix using tiled algorithm with bank conflict avoidance.
///
/// ## Bank Conflict Avoidance
///
/// Uses 16×17 tile (16×16 + 1 padding) to avoid shared memory bank conflicts.
/// The extra column of padding ensures coalesced memory access patterns.
///
/// ## Performance
///
/// - Memory-bound operation: limited by GPU memory bandwidth
/// - Optimal for matrices with dimensions that are multiples of 16
///
/// ## Usage
///
/// ```swift
/// let kernel = try await MatrixTransposeKernel(context: context)
///
/// // Transpose: B = A^T
/// let result = try await kernel.transpose(matrixA)
///
/// // Verify: (A^T)^T == A
/// let valid = try await kernel.validate(matrixA)
/// ```
public final class MatrixTransposeKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "MatrixTransposeKernel"

    // MARK: - Constants

    private let TILE_SIZE: Int = 16
    private let TILE_SIZE_PADDED: Int = 17  // +1 for bank conflict avoidance

    // MARK: - Pipelines

    private let transposePipeline: any MTLComputePipelineState
    private let inPlacePipeline: (any MTLComputePipelineState)?

    // MARK: - Initialization

    /// Create a Metal 4 Matrix Transpose kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let transposeFunc = library.makeFunction(name: "tiledTranspose") else {
            throw VectorError.shaderNotFound(
                name: "Tiled transpose kernel. Ensure DataTransformations.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.transposePipeline = try await device.makeComputePipelineState(function: transposeFunc)

        if let inPlaceFunc = library.makeFunction(name: "tiledTransposeInPlace") {
            self.inPlacePipeline = try await device.makeComputePipelineState(function: inPlaceFunc)
        } else {
            self.inPlacePipeline = nil
        }
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API

    /// Encode transpose into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        parameters: TransposeParameters,
        inPlace: Bool = false
    ) -> Metal4EncodingResult {
        let canDoInPlace = inPlace && parameters.rows == parameters.columns && inPlacePipeline != nil
        let pipeline = canDoInPlace ? inPlacePipeline! : transposePipeline
        let pipelineName = canDoInPlace ? "tiledTransposeInPlace" : "tiledTranspose"

        encoder.setComputePipelineState(pipeline)
        encoder.label = "MatrixTranspose (\(parameters.rows)×\(parameters.columns))"

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(canDoInPlace ? input : output, offset: 0, index: 1)

        var dims = SIMD2<UInt32>(parameters.rows, parameters.columns)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)

        // Allocate shared memory with padding
        let sharedMemorySize = TILE_SIZE_PADDED * TILE_SIZE * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)

        let threadgroupSize = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(parameters.columns) + TILE_SIZE - 1) / TILE_SIZE,
            height: (Int(parameters.rows) + TILE_SIZE - 1) / TILE_SIZE,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

        return Metal4EncodingResult(
            pipelineName: pipelineName,
            threadgroups: threadgroupCount,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Execute API

    /// Execute transpose as standalone operation.
    public func execute(
        input: any MTLBuffer,
        parameters: TransposeParameters,
        config: Metal4TransposeConfig = .default
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let outputSize = Int(parameters.rows) * Int(parameters.columns) * MemoryLayout<Float>.size

        let canDoInPlace = config.inPlace && parameters.rows == parameters.columns && inPlacePipeline != nil

        let outputBuffer: any MTLBuffer
        if canDoInPlace {
            outputBuffer = input
        } else {
            guard let buffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: outputSize)
            }
            buffer.label = "MatrixTranspose.output"
            outputBuffer = buffer
        }

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                input: input,
                output: outputBuffer,
                parameters: parameters,
                inPlace: canDoInPlace
            )
        }

        return outputBuffer
    }

    // MARK: - High-Level API

    /// Transpose a matrix.
    public func transpose(
        _ matrix: Matrix,
        config: Metal4TransposeConfig = .default
    ) async throws -> Metal4TransposeResult {
        let device = context.device.rawDevice

        guard let inputBuffer = device.makeBuffer(
            bytes: matrix.values,
            length: matrix.values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: matrix.values.count * MemoryLayout<Float>.stride)
        }
        inputBuffer.label = "MatrixTranspose.input"

        let parameters = TransposeParameters(rows: matrix.rows, columns: matrix.columns)

        let startTime = CACurrentMediaTime()
        let outputBuffer = try await execute(input: inputBuffer, parameters: parameters, config: config)
        let executionTime = CACurrentMediaTime() - startTime

        // Calculate throughput (read + write bandwidth)
        let bytesProcessed = Double(matrix.rows * matrix.columns * 2 * MemoryLayout<Float>.stride)
        let throughputGBps = (bytesProcessed / 1e9) / executionTime

        return Metal4TransposeResult(
            buffer: outputBuffer,
            rows: matrix.columns,  // Swapped
            columns: matrix.rows,  // Swapped
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Transpose from flat array.
    public func transpose(
        data: [Float],
        rows: Int,
        columns: Int,
        config: Metal4TransposeConfig = .default
    ) async throws -> Metal4TransposeResult {
        let matrix = Matrix(rows: rows, columns: columns, values: data)
        return try await transpose(matrix, config: config)
    }

    /// Transpose multiple matrices in batch.
    public func transposeBatch(
        _ matrices: [Matrix],
        config: Metal4TransposeConfig = .default
    ) async throws -> [Metal4TransposeResult] {
        var results: [Metal4TransposeResult] = []
        for matrix in matrices {
            results.append(try await transpose(matrix, config: config))
        }
        return results
    }

    // MARK: - Specialized Operations

    /// Double transpose (should return original).
    public func doubleTranspose(_ matrix: Matrix) async throws -> Matrix {
        let first = try await transpose(matrix)
        let second = try await transpose(first.asMatrix())
        return second.asMatrix()
    }

    /// Validate transpose correctness.
    ///
    /// Verifies that (A^T)^T == A within tolerance.
    public func validate(_ matrix: Matrix, tolerance: Float = 1e-6) async throws -> Bool {
        let transposed = try await transpose(matrix)
        let doubleTransposed = try await transpose(transposed.asMatrix())

        // Compare with original
        for i in 0..<matrix.values.count {
            if abs(matrix.values[i] - doubleTransposed.asMatrix().values[i]) > tolerance {
                return false
            }
        }

        // Check specific elements
        for row in 0..<min(matrix.rows, 10) {
            for col in 0..<min(matrix.columns, 10) {
                let original = matrix[row, col]
                let transposedValue = transposed.element(row: col, col: row)
                if abs(original - transposedValue) > tolerance {
                    return false
                }
            }
        }

        return true
    }
}
