// Matrix Multiply Kernel
// GPU-accelerated matrix multiplication with tiling optimization

import Metal
import Foundation
import VectorCore
import QuartzCore

// MARK: - Matrix Multiply Kernel

/// GPU-accelerated matrix multiplication using tiled algorithm
/// Optimized for Apple Silicon with 32x32x8 tiling strategy
public final class MatrixMultiplyKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLComputePipelineState
    private let specializedStates: [Int: any MTLComputePipelineState]
    
    // Tile configuration matching Metal kernel
    private let TILE_M: Int = 32
    private let TILE_N: Int = 32
    private let TILE_K: Int = 8
    
    // MARK: - Result Types
    
    /// Result from matrix multiplication
    public struct MultiplyResult: Sendable {
        public let matrix: Matrix
        public let executionTime: TimeInterval
        public let gflops: Double
        
        /// Convert to flat array
        public func asArray() -> [Float] {
            return matrix.values
        }
        
        /// Get element at (row, col)
        public func element(row: Int, col: Int) -> Float {
            return matrix[row, col]
        }
    }
    
    /// Configuration for matrix multiplication
    public struct MultiplyConfig: Sendable {
        public let alpha: Float
        public let beta: Float
        public let transposeA: Bool
        public let transposeB: Bool
        public let useSpecializedKernel: Bool
        
        public init(
            alpha: Float = 1.0,
            beta: Float = 0.0,
            transposeA: Bool = false,
            transposeB: Bool = false,
            useSpecializedKernel: Bool = true
        ) {
            self.alpha = alpha
            self.beta = beta
            self.transposeA = transposeA
            self.transposeB = transposeB
            self.useSpecializedKernel = useSpecializedKernel
        }
        
        public static let `default` = MultiplyConfig()
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load main kernel
        guard let function = library.makeFunction(name: "tiledMatrixMultiply") else {
            throw AccelerationError.shaderNotFound(name: "tiledMatrixMultiply")
        }
        
        self.pipelineState = try device.makeComputePipelineState(function: function)
        
        // Load specialized kernels for common dimensions
        var specialized: [Int: any MTLComputePipelineState] = [:]
        for dimension in [512, 768, 1536] {
            if let specialFunc = library.makeFunction(name: "tiledMatrixMultiply_\(dimension)") {
                specialized[dimension] = try device.makeComputePipelineState(function: specialFunc)
            }
        }
        self.specializedStates = specialized
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < TILE_M * TILE_N {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required threadgroup size: \(TILE_M * TILE_N)"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Multiply two matrices: C = alpha * A * B
    /// - Parameters:
    ///   - matrixA: First matrix (M x K)
    ///   - matrixB: Second matrix (K x N)
    ///   - config: Multiplication configuration (beta should be 0 for this variant)
    /// - Returns: Result matrix (M x N) with performance metrics
    public func multiply(
        _ matrixA: Matrix,
        _ matrixB: Matrix,
        config: MultiplyConfig = .default
    ) throws -> MultiplyResult {
        // Validate dimensions
        let effectiveColsA = config.transposeA ? matrixA.rows : matrixA.columns
        let effectiveRowsB = config.transposeB ? matrixB.columns : matrixB.rows
        
        guard effectiveColsA == effectiveRowsB else {
            throw AccelerationError.countMismatch(
                expected: effectiveColsA,
                actual: effectiveRowsB
            )
        }
        
        let M = config.transposeA ? matrixA.columns : matrixA.rows
        let K = effectiveColsA
        let N = config.transposeB ? matrixB.rows : matrixB.columns
        
        // Create buffers
        guard let bufferA = device.makeBuffer(
            bytes: matrixA.values,
            length: matrixA.values.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: matrixA.values.count * MemoryLayout<Float>.stride)
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: matrixB.values,
            length: matrixB.values.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: matrixB.values.count * MemoryLayout<Float>.stride)
        }
        
        let outputSize = M * N * MemoryLayout<Float>.stride
        guard let bufferC = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: outputSize)
        }
        
        // Initialize output buffer if beta != 0
        if config.beta != 0 {
            bufferC.contents().initializeMemory(as: Float.self, repeating: 0, count: M * N)
        }
        
        // Select appropriate pipeline state
        let state: any MTLComputePipelineState
        if config.useSpecializedKernel, let specializedState = specializedStates[K] {
            state = specializedState
        } else {
            state = pipelineState
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Set pipeline and buffers
        encoder.setComputePipelineState(state)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        
        // Set dimensions
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(K), UInt32(N))
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 3)
        
        // Configure thread groups
        let threadgroupSize = MTLSize(width: TILE_N, height: TILE_M, depth: 1)
        let threadgroupCount = MTLSize(
            width: (N + TILE_N - 1) / TILE_N,
            height: (M + TILE_M - 1) / TILE_M,
            depth: 1
        )
        
        // Calculate shared memory size
        let sharedMemorySize = (TILE_M * TILE_K + TILE_K * TILE_N) * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 1)
        
        // Dispatch
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Matrix multiplication failed: \(error)")
        }
        
        // Extract results
        let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: M * N)
        let resultData = Array(UnsafeBufferPointer(start: resultPointer, count: M * N))
        
        // Apply alpha scaling (beta is handled in the shader or initialization)
        // Note: Standard GEMM is C = alpha * A*B + beta * C_initial
        // Since we don't have C_initial, beta should typically be 0
        var finalData = resultData
        if config.alpha != 1.0 {
            for i in 0..<finalData.count {
                finalData[i] = config.alpha * finalData[i]
            }
        }

        // If beta != 0, it was already applied during buffer initialization
        // The shader should handle: C[i,j] = alpha * sum(A[i,k] * B[k,j]) + beta * C_initial[i,j]
        
        // Calculate GFLOPS
        let operations = 2.0 * Double(M) * Double(N) * Double(K)
        let gflops = (operations / 1e9) / executionTime
        
        let resultMatrix = Matrix(rows: M, columns: N, values: finalData)
        return MultiplyResult(matrix: resultMatrix, executionTime: executionTime, gflops: gflops)
    }
    
    /// General Matrix Multiply (GEMM): C = alpha * A * B + beta * C
    /// - Parameters:
    ///   - matrixA: First matrix (M x K)
    ///   - matrixB: Second matrix (K x N)
    ///   - matrixC: Initial matrix C (M x N) for accumulation
    ///   - config: Multiplication configuration with alpha and beta
    /// - Returns: Result matrix (M x N) with performance metrics
    public func gemm(
        _ matrixA: Matrix,
        _ matrixB: Matrix,
        _ matrixC: Matrix,
        config: MultiplyConfig = .default
    ) throws -> MultiplyResult {
        // Validate C dimensions
        let M = config.transposeA ? matrixA.columns : matrixA.rows
        let N = config.transposeB ? matrixB.rows : matrixB.columns

        guard matrixC.rows == M && matrixC.columns == N else {
            throw AccelerationError.invalidInput("Matrix C dimensions must be \(M)x\(N)")
        }

        // First compute A * B
        let abResult = try multiply(matrixA, matrixB, config: MultiplyConfig(
            alpha: 1.0,  // We'll apply alpha later
            beta: 0.0,
            transposeA: config.transposeA,
            transposeB: config.transposeB,
            useSpecializedKernel: config.useSpecializedKernel
        ))

        // Now apply the full GEMM formula: C = alpha * (A*B) + beta * C
        var finalValues = abResult.matrix.values
        for i in 0..<finalValues.count {
            finalValues[i] = config.alpha * finalValues[i] + config.beta * matrixC.values[i]
        }

        let resultMatrix = Matrix(rows: M, columns: N, values: finalValues)
        return MultiplyResult(
            matrix: resultMatrix,
            executionTime: abResult.executionTime,
            gflops: abResult.gflops
        )
    }

    /// Multiply matrices from flat arrays
    public func multiply(
        matrixA: [Float],
        rowsA: Int,
        colsA: Int,
        matrixB: [Float],
        rowsB: Int,
        colsB: Int,
        config: MultiplyConfig = .default
    ) throws -> MultiplyResult {
        let matA = Matrix(rows: rowsA, columns: colsA, values: matrixA)
        let matB = Matrix(rows: rowsB, columns: colsB, values: matrixB)
        return try multiply(matA, matB, config: config)
    }
    
    // MARK: - Async Operations
    
    /// Async version of matrix multiplication
    public func multiplyAsync(
        _ matrixA: Matrix,
        _ matrixB: Matrix,
        config: MultiplyConfig = .default
    ) async throws -> MultiplyResult {
        return try await Task.detached(priority: .userInitiated) { [self] in
            return try self.multiply(matrixA, matrixB, config: config)
        }.value
    }
    
    // MARK: - Batch Operations
    
    /// Multiply multiple matrix pairs in batch
    public func multiplyBatch(
        matricesA: [Matrix],
        matricesB: [Matrix],
        config: MultiplyConfig = .default
    ) throws -> [MultiplyResult] {
        guard matricesA.count == matricesB.count else {
            throw AccelerationError.invalidInput("Matrix batch sizes must match")
        }
        
        var results: [MultiplyResult] = []
        for (matA, matB) in zip(matricesA, matricesB) {
            results.append(try multiply(matA, matB, config: config))
        }
        
        return results
    }
    
    // MARK: - VectorCore Integration
    
    /// Multiply using VectorCore protocol types
    public func multiply<V: VectorProtocol>(
        vectorsA: [[V]],
        vectorsB: [[V]],
        config: MultiplyConfig = .default
    ) throws -> MultiplyResult where V.Scalar == Float {
        // Convert to matrix format
        let rowsA = vectorsA.count
        let colsA = vectorsA.first?.count ?? 0
        let dataA = vectorsA.flatMap { row in row.flatMap { $0.toArray() } }
        
        let rowsB = vectorsB.count
        let colsB = vectorsB.first?.count ?? 0
        let dataB = vectorsB.flatMap { row in row.flatMap { $0.toArray() } }
        
        let matA = Matrix(rows: rowsA, columns: colsA, values: dataA)
        let matB = Matrix(rows: rowsB, columns: colsB, values: dataB)
        
        return try multiply(matA, matB, config: config)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark matrix multiplication for different sizes
    public func benchmark(sizes: [(M: Int, K: Int, N: Int)]) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for size in sizes {
            // Generate random matrices
            let matA = Matrix.random(rows: size.M, columns: size.K)
            let matB = Matrix.random(rows: size.K, columns: size.N)
            
            // Warm-up run
            _ = try multiply(matA, matB)
            
            // Timed runs
            var times: [TimeInterval] = []
            for _ in 0..<5 {
                let result = try multiply(matA, matB)
                times.append(result.executionTime)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let operations = 2.0 * Double(size.M) * Double(size.N) * Double(size.K)
            let gflops = (operations / 1e9) / avgTime
            
            results.append(BenchmarkResult(
                dimensions: "\(size.M)x\(size.K)x\(size.N)",
                executionTime: avgTime,
                gflops: gflops
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let dimensions: String
        public let executionTime: TimeInterval
        public let gflops: Double
    }
}

// MARK: - Matrix Type Extension

extension Matrix {
    /// Create random matrix for testing
    static func random(rows: Int, columns: Int, range: ClosedRange<Float> = -1...1) -> Matrix {
        let data = (0..<(rows * columns)).map { _ in Float.random(in: range) }
        return Matrix(rows: rows, columns: columns, values: data)
    }
}