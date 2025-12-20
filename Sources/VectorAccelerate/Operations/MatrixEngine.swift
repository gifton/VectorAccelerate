//
//  MatrixEngine.swift
//  VectorAccelerate
//
//  High-performance matrix operations with GPU acceleration using Metal 4.
//

import Foundation
@preconcurrency import Metal
import Accelerate
import VectorCore

/// Configuration for matrix operations
public struct MatrixConfiguration: Sendable {
    public let useTiledMultiplication: Bool
    public let tileSize: Int
    public let preferGPUThreshold: Int
    public let enableAsyncExecution: Bool

    public init(
        useTiledMultiplication: Bool = true,
        tileSize: Int = 32,
        preferGPUThreshold: Int = 256,
        enableAsyncExecution: Bool = false
    ) {
        self.useTiledMultiplication = useTiledMultiplication
        self.tileSize = tileSize
        self.preferGPUThreshold = preferGPUThreshold
        self.enableAsyncExecution = enableAsyncExecution
    }

    public static let `default` = MatrixConfiguration()
    public static let performance = MatrixConfiguration(
        tileSize: 64,
        preferGPUThreshold: 128,
        enableAsyncExecution: true
    )
}

/// Matrix layout for memory organization
public enum MatrixLayout: Sendable {
    case rowMajor
    case columnMajor

    var isTransposed: Bool {
        self == .columnMajor
    }
}

/// Matrix descriptor
public struct MatrixDescriptor: Sendable {
    public let rows: Int
    public let columns: Int
    public let layout: MatrixLayout
    public let leadingDimension: Int?

    public init(rows: Int, columns: Int, layout: MatrixLayout = .rowMajor, leadingDimension: Int? = nil) {
        self.rows = rows
        self.columns = columns
        self.layout = layout
        self.leadingDimension = leadingDimension ?? columns
    }

    public var elementCount: Int {
        rows * columns
    }

    public var byteSize: Int {
        elementCount * MemoryLayout<Float>.stride
    }
}

/// Main matrix operations engine using Metal 4
///
/// This actor orchestrates matrix operations using Metal 4 specialized kernels.
/// It automatically selects between CPU (Accelerate) and GPU (Metal 4) execution
/// based on matrix size and configuration.
///
/// ## Metal 4 Only
/// This implementation requires Metal 4 (iOS 26+, macOS 26+). There is no
/// fallback to older Metal versions.
///
/// ## Usage
///
/// ```swift
/// let context = try await Metal4Context()
/// let engine = await MatrixEngine(context: context)
///
/// let result = try await engine.multiply(matrixA, descriptorA: descA,
///                                         matrixB, descriptorB: descB)
/// ```
public actor MatrixEngine {
    private let context: Metal4Context
    private let configuration: MatrixConfiguration
    private let logger: Logger

    // Metal 4 Kernels (lazy initialized)
    private var matrixMultiplyKernel: MatrixMultiplyKernel?
    private var matrixTransposeKernel: MatrixTransposeKernel?

    // Performance tracking
    private var operationCount: Int = 0
    private var totalComputeTime: TimeInterval = 0

    // MARK: - Initialization

    /// Create a MatrixEngine with a Metal 4 context.
    ///
    /// - Parameters:
    ///   - context: Metal 4 context for GPU operations
    ///   - configuration: Matrix operation configuration
    public init(
        context: Metal4Context,
        configuration: MatrixConfiguration = .default
    ) async {
        self.context = context
        self.configuration = configuration
        self.logger = Logger.shared
    }

    // MARK: - Kernel Initialization (Lazy)

    private func getMatrixMultiplyKernel() async throws -> MatrixMultiplyKernel {
        if let kernel = matrixMultiplyKernel {
            return kernel
        }
        let kernel = try await MatrixMultiplyKernel(context: context)
        matrixMultiplyKernel = kernel
        return kernel
    }

    private func getMatrixTransposeKernel() async throws -> MatrixTransposeKernel {
        if let kernel = matrixTransposeKernel {
            return kernel
        }
        let kernel = try await MatrixTransposeKernel(context: context)
        matrixTransposeKernel = kernel
        return kernel
    }

    // MARK: - Matrix Multiplication

    /// Perform matrix multiplication: C = A * B
    ///
    /// Computes the matrix product C = AB where:
    /// - A is an m×n matrix
    /// - B is an n×p matrix
    /// - C is the resulting m×p matrix
    ///
    /// Mathematical formula:
    /// ```
    /// C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
    /// ```
    ///
    /// - Complexity: O(m·n·p) for naive implementation, O(m·n·p/√cache) with tiling
    /// - Memory: O(m·p) for output matrix
    ///
    /// - Parameters:
    ///   - matrixA: First matrix (row-major order)
    ///   - descriptorA: Dimensions of matrix A
    ///   - matrixB: Second matrix (row-major order)
    ///   - descriptorB: Dimensions of matrix B
    /// - Returns: Result matrix C in row-major order
    /// - Throws: `VectorError.dimensionMismatch` if A.columns ≠ B.rows
    public func multiply(
        _ matrixA: [Float],
        descriptorA: MatrixDescriptor,
        _ matrixB: [Float],
        descriptorB: MatrixDescriptor
    ) async throws -> [Float] {
        // Validate dimensions
        guard descriptorA.columns == descriptorB.rows else {
            throw VectorError.dimensionMismatch(
                expected: descriptorA.columns,
                actual: descriptorB.rows
            )
        }

        let measureToken = await logger.startMeasure("matrixMultiply")
        defer { measureToken.end() }

        // Decide execution path based on size
        let totalElements = descriptorA.rows * descriptorB.columns

        let startTime = CFAbsoluteTimeGetCurrent()
        let result: [Float]

        if totalElements < configuration.preferGPUThreshold {
            // Use CPU for small matrices
            result = try cpuMatrixMultiply(
                matrixA, descriptorA: descriptorA,
                matrixB, descriptorB: descriptorB
            )
        } else {
            // Use GPU for large matrices
            result = try await gpuMatrixMultiply(
                matrixA, descriptorA: descriptorA,
                matrixB, descriptorB: descriptorB
            )
        }

        // Track performance metrics
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        operationCount += 1
        totalComputeTime += elapsed

        return result
    }

    /// GPU-accelerated matrix multiplication using Metal 4 kernel
    private func gpuMatrixMultiply(
        _ matrixA: [Float],
        descriptorA: MatrixDescriptor,
        _ matrixB: [Float],
        descriptorB: MatrixDescriptor
    ) async throws -> [Float] {
        let kernel = try await getMatrixMultiplyKernel()

        let result = try await kernel.multiply(
            matrixA: matrixA,
            rowsA: descriptorA.rows,
            colsA: descriptorA.columns,
            matrixB: matrixB,
            rowsB: descriptorB.rows,
            colsB: descriptorB.columns
        )

        return result.asMatrix().values
    }

    /// CPU-optimized matrix multiplication using Accelerate
    private func cpuMatrixMultiply(
        _ matrixA: [Float],
        descriptorA: MatrixDescriptor,
        _ matrixB: [Float],
        descriptorB: MatrixDescriptor
    ) throws -> [Float] {
        let m = descriptorA.rows
        let n = descriptorB.columns
        let k = descriptorA.columns

        var result = [Float](repeating: 0, count: m * n)

        // Use vDSP for optimal CPU performance
        matrixA.withUnsafeBufferPointer { ptrA in
            matrixB.withUnsafeBufferPointer { ptrB in
                result.withUnsafeMutableBufferPointer { ptrC in
                    vDSP_mmul(
                        ptrA.baseAddress!, vDSP_Stride(1),
                        ptrB.baseAddress!, vDSP_Stride(1),
                        ptrC.baseAddress!, vDSP_Stride(1),
                        vDSP_Length(m),
                        vDSP_Length(n),
                        vDSP_Length(k)
                    )
                }
            }
        }

        return result
    }

    // MARK: - Matrix Transpose

    /// Transpose a matrix
    ///
    /// Computes the transpose A^T of matrix A where:
    /// ```
    /// A^T[i,j] = A[j,i]
    /// ```
    ///
    /// For an m×n matrix A, the transpose A^T is an n×m matrix.
    ///
    /// - Complexity: O(m·n) with cache-oblivious algorithm for optimal cache usage
    /// - Memory: O(m·n) for output matrix
    ///
    /// - Parameters:
    ///   - matrix: Input matrix in row-major order
    ///   - descriptor: Dimensions of the input matrix
    /// - Returns: Transposed matrix in row-major order
    public func transpose(
        _ matrix: [Float],
        descriptor: MatrixDescriptor
    ) async throws -> [Float] {
        let measureToken = await logger.startMeasure("matrixTranspose")
        defer { measureToken.end() }

        let startTime = CFAbsoluteTimeGetCurrent()

        // For small matrices, use CPU; otherwise use GPU
        let result: [Float]
        if descriptor.elementCount < configuration.preferGPUThreshold {
            result = cpuTranspose(matrix, descriptor: descriptor)
        } else {
            result = try await gpuTranspose(matrix, descriptor: descriptor)
        }

        // Track performance metrics
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        operationCount += 1
        totalComputeTime += elapsed

        return result
    }

    /// GPU-accelerated transpose using Metal 4 kernel
    private func gpuTranspose(
        _ matrix: [Float],
        descriptor: MatrixDescriptor
    ) async throws -> [Float] {
        let kernel = try await getMatrixTransposeKernel()

        let result = try await kernel.transpose(
            data: matrix,
            rows: descriptor.rows,
            columns: descriptor.columns
        )

        return result.asMatrix().values
    }

    /// CPU transpose using Accelerate
    private func cpuTranspose(_ matrix: [Float], descriptor: MatrixDescriptor) -> [Float] {
        var result = [Float](repeating: 0, count: descriptor.elementCount)

        matrix.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vDSP_mtrans(
                    src.baseAddress!,
                    1,
                    dst.baseAddress!,
                    1,
                    vDSP_Length(descriptor.columns),
                    vDSP_Length(descriptor.rows)
                )
            }
        }

        return result
    }

    // MARK: - Batch Operations

    /// Batch matrix-vector multiplication
    public func batchMatrixVectorMultiply(
        matrices: [[Float]],
        descriptor: MatrixDescriptor,
        vectors: [[Float]]
    ) async throws -> [[Float]] {
        guard matrices.count == vectors.count else {
            throw VectorError.dimensionMismatch(
                expected: matrices.count,
                actual: vectors.count
            )
        }

        let measureToken = await logger.startMeasure("batchMatrixVectorMultiply")
        measureToken.addMetadata("batchSize", value: "\(matrices.count)")
        defer { measureToken.end() }

        // Process in parallel
        return try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            for (index, (matrix, vector)) in zip(matrices, vectors).enumerated() {
                group.addTask {
                    let result = try await self.matrixVectorMultiply(
                        matrix: matrix,
                        descriptor: descriptor,
                        vector: vector
                    )
                    return (index, result)
                }
            }

            // Collect results in order
            var results = [[Float]](repeating: [], count: matrices.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }

    /// Single matrix-vector multiplication
    private func matrixVectorMultiply(
        matrix: [Float],
        descriptor: MatrixDescriptor,
        vector: [Float]
    ) async throws -> [Float] {
        guard vector.count == descriptor.columns else {
            throw VectorError.dimensionMismatch(
                expected: descriptor.columns,
                actual: vector.count
            )
        }

        // Use vDSP for efficient computation
        var result = [Float](repeating: 0, count: descriptor.rows)

        matrix.withUnsafeBufferPointer { matPtr in
            vector.withUnsafeBufferPointer { vecPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_mmul(
                        matPtr.baseAddress!, vDSP_Stride(1),
                        vecPtr.baseAddress!, vDSP_Stride(1),
                        resPtr.baseAddress!, vDSP_Stride(1),
                        vDSP_Length(descriptor.rows),
                        vDSP_Length(1),
                        vDSP_Length(descriptor.columns)
                    )
                }
            }
        }

        return result
    }

    // MARK: - Performance Metrics

    public func getPerformanceMetrics() -> (operations: Int, averageTime: TimeInterval) {
        let avgTime = operationCount > 0 ? totalComputeTime / Double(operationCount) : 0
        return (operationCount, avgTime)
    }
}

// MARK: - Convenience Extensions

public extension MatrixEngine {
    /// Create with default Metal 4 context
    static func createDefault() async throws -> MatrixEngine {
        guard let context = await Metal4Context.createDefault() else {
            throw VectorError.deviceInitializationFailed("Metal 4 not available")
        }
        return await MatrixEngine(context: context)
    }
}
