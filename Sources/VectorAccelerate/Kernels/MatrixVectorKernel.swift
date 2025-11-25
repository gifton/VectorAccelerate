// Matrix-Vector Kernel
// GPU-accelerated matrix-vector multiplication using SIMD groups

import Metal
import Foundation
import VectorCore
import QuartzCore

// MARK: - Matrix-Vector Kernel

/// GPU-accelerated matrix-vector multiplication
/// Optimized using SIMD group operations for efficient reduction
public final class MatrixVectorKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let simdgroupKernel: any MTLComputePipelineState
    private let basicKernel: any MTLComputePipelineState
    private let batchKernel: (any MTLComputePipelineState)?
    
    // SIMD configuration for Apple Silicon
    private let SIMD_WIDTH: Int = 32
    
    // MARK: - Result Types
    
    /// Result from matrix-vector multiplication
    public struct MatrixVectorResult: Sendable {
        public let vector: [Float]
        public let executionTime: TimeInterval
        public let gflops: Double
        
        /// Get element at index
        public func element(at index: Int) -> Float {
            guard index < vector.count else { return 0 }
            return vector[index]
        }
        
        /// Convert to VectorCore type
        public func asVector<V: VectorProtocol>() -> V where V.Scalar == Float {
            return try! V(vector)
        }
    }
    
    /// Batch result for multiple operations
    public struct BatchResult: Sendable {
        public let vectors: [[Float]]
        public let totalExecutionTime: TimeInterval
        public let averageGflops: Double
        
        /// Get specific result vector
        public func vector(at index: Int) -> [Float]? {
            guard index < vectors.count else { return nil }
            return vectors[index]
        }
    }
    
    /// Configuration for matrix-vector operations
    public struct MatrixVectorConfig: Sendable {
        public let alpha: Float           // Scaling factor for result
        public let beta: Float            // Scaling factor for accumulation
        public let transpose: Bool        // Use A^T instead of A
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
        
        public static let `default` = MatrixVectorConfig()
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
        
        // Load SIMD-optimized kernel
        guard let simdFunc = library.makeFunction(name: "simdgroupMatrixVector") else {
            throw AccelerationError.shaderNotFound(name: "simdgroupMatrixVector")
        }
        self.simdgroupKernel = try device.makeComputePipelineState(function: simdFunc)
        
        // Load basic kernel as fallback
        guard let basicFunc = library.makeFunction(name: "matrixVectorMultiply") else {
            throw AccelerationError.shaderNotFound(name: "matrixVectorMultiply")
        }
        self.basicKernel = try device.makeComputePipelineState(function: basicFunc)
        
        // Try to load batch kernel
        if let batchFunc = library.makeFunction(name: "batchMatrixVector") {
            self.batchKernel = try device.makeComputePipelineState(function: batchFunc)
        } else {
            self.batchKernel = nil
        }
        
        // Validate SIMD support
        let maxThreadsPerThreadgroup = simdgroupKernel.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < SIMD_WIDTH {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required SIMD width: \(SIMD_WIDTH)"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Multiply matrix by vector: y = alpha * A * x + beta * y
    /// - Parameters:
    ///   - matrix: Input matrix (M x N)
    ///   - vector: Input vector (N x 1)
    ///   - config: Operation configuration
    /// - Returns: Result vector (M x 1) with performance metrics
    public func multiply(
        matrix: Matrix,
        vector: [Float],
        config: MatrixVectorConfig = .default
    ) throws -> MatrixVectorResult {
        // Validate dimensions
        let expectedVectorLength = config.transpose ? matrix.rows : matrix.columns
        guard vector.count == expectedVectorLength else {
            throw AccelerationError.countMismatch(
                expected: expectedVectorLength,
                actual: vector.count
            )
        }
        
        let outputLength = config.transpose ? matrix.columns : matrix.rows
        
        // Create buffers
        guard let matrixBuffer = device.makeBuffer(
            bytes: matrix.values,
            length: matrix.values.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: matrix.values.count * MemoryLayout<Float>.stride)
        }
        
        guard let vectorBuffer = device.makeBuffer(
            bytes: vector,
            length: vector.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: vector.count * MemoryLayout<Float>.stride)
        }
        
        let outputSize = outputLength * MemoryLayout<Float>.stride
        guard let resultBuffer = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: outputSize)
        }
        
        // Initialize result buffer if beta != 0
        if config.beta != 0 {
            resultBuffer.contents().initializeMemory(as: Float.self, repeating: 0, count: outputLength)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Select kernel based on configuration
        let kernel = config.useSIMDOptimization ? simdgroupKernel : basicKernel
        encoder.setComputePipelineState(kernel)
        
        // Set buffers
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(vectorBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        // Set dimensions (rows, cols)
        var dims = SIMD2<UInt32>(UInt32(matrix.rows), UInt32(matrix.columns))
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        
        // Configure thread groups
        let threadsPerThreadgroup: MTLSize
        let threadgroupCount: MTLSize
        
        if config.useSIMDOptimization {
            // SIMD-optimized configuration
            threadsPerThreadgroup = MTLSize(width: SIMD_WIDTH, height: 1, depth: 1)
            threadgroupCount = MTLSize(
                width: (outputLength + SIMD_WIDTH - 1) / SIMD_WIDTH,
                height: 1,
                depth: 1
            )
        } else {
            // Basic configuration
            let blockSize = 256
            threadsPerThreadgroup = MTLSize(width: blockSize, height: 1, depth: 1)
            threadgroupCount = MTLSize(
                width: (outputLength + blockSize - 1) / blockSize,
                height: 1,
                depth: 1
            )
        }
        
        // Dispatch
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Matrix-vector multiplication failed: \(error)")
        }
        
        // Extract results
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: outputLength)
        var resultData = Array(UnsafeBufferPointer(start: resultPointer, count: outputLength))
        
        // Apply alpha and beta scaling if needed
        if config.alpha != 1.0 || config.beta != 0.0 {
            for i in 0..<resultData.count {
                resultData[i] = config.alpha * resultData[i] + config.beta * resultData[i]
            }
        }
        
        // Calculate GFLOPS
        let operations = 2.0 * Double(matrix.rows) * Double(matrix.columns)
        let gflops = (operations / 1e9) / executionTime
        
        return MatrixVectorResult(
            vector: resultData,
            executionTime: executionTime,
            gflops: gflops
        )
    }
    
    /// Multiply from flat arrays
    public func multiply(
        matrixData: [Float],
        rows: Int,
        columns: Int,
        vector: [Float],
        config: MatrixVectorConfig = .default
    ) throws -> MatrixVectorResult {
        let matrix = Matrix(rows: rows, columns: columns, values: matrixData)
        return try multiply(matrix: matrix, vector: vector, config: config)
    }
    
    // MARK: - Batch Operations
    
    /// Multiply matrix by multiple vectors
    public func multiplyBatch(
        matrix: Matrix,
        vectors: [[Float]],
        config: MatrixVectorConfig = .default
    ) throws -> BatchResult {
        let startTime = CACurrentMediaTime()
        var results: [[Float]] = []
        var totalGflops: Double = 0
        
        // Process each vector
        for vector in vectors {
            let result = try multiply(matrix: matrix, vector: vector, config: config)
            results.append(result.vector)
            totalGflops += result.gflops
        }
        
        let totalTime = CACurrentMediaTime() - startTime
        let avgGflops = totalGflops / Double(vectors.count)
        
        return BatchResult(
            vectors: results,
            totalExecutionTime: totalTime,
            averageGflops: avgGflops
        )
    }
    
    /// Multiply multiple matrices by corresponding vectors
    public func multiplyPairs(
        matrices: [Matrix],
        vectors: [[Float]],
        config: MatrixVectorConfig = .default
    ) throws -> BatchResult {
        guard matrices.count == vectors.count else {
            throw AccelerationError.invalidInput("Number of matrices must match number of vectors")
        }
        
        let startTime = CACurrentMediaTime()
        var results: [[Float]] = []
        var totalGflops: Double = 0
        
        for (matrix, vector) in zip(matrices, vectors) {
            let result = try multiply(matrix: matrix, vector: vector, config: config)
            results.append(result.vector)
            totalGflops += result.gflops
        }
        
        let totalTime = CACurrentMediaTime() - startTime
        let avgGflops = totalGflops / Double(matrices.count)
        
        return BatchResult(
            vectors: results,
            totalExecutionTime: totalTime,
            averageGflops: avgGflops
        )
    }
    
    // MARK: - Async Operations
    
    /// Async matrix-vector multiplication
    public func multiplyAsync(
        matrix: Matrix,
        vector: [Float],
        config: MatrixVectorConfig = .default
    ) async throws -> MatrixVectorResult {
        return try await Task.detached(priority: .userInitiated) { [self] in
            return try self.multiply(matrix: matrix, vector: vector, config: config)
        }.value
    }
    
    // MARK: - VectorCore Integration
    
    /// Multiply using VectorCore protocol types
    public func multiply<V: VectorProtocol>(
        matrix: Matrix,
        vector: V,
        config: MatrixVectorConfig = .default
    ) throws -> MatrixVectorResult where V.Scalar == Float {
        let vectorArray = vector.toArray()
        return try multiply(matrix: matrix, vector: vectorArray, config: config)
    }
    
    // MARK: - Specialized Operations
    
    /// Compute matrix-vector product and norm: ||A * x||
    public func multiplyAndNorm(
        matrix: Matrix,
        vector: [Float],
        config: MatrixVectorConfig = .default
    ) throws -> (result: MatrixVectorResult, norm: Float) {
        let result = try multiply(matrix: matrix, vector: vector, config: config)
        
        // Compute L2 norm of result
        let norm = sqrt(result.vector.reduce(0) { $0 + $1 * $1 })
        
        return (result, norm)
    }
    
    /// Power iteration for eigenvalue estimation
    public func powerIteration(
        matrix: Matrix,
        iterations: Int = 100,
        tolerance: Float = 1e-6
    ) throws -> (eigenvalue: Float, eigenvector: [Float]) {
        guard matrix.rows == matrix.columns else {
            throw AccelerationError.invalidInput("Matrix must be square for eigenvalue computation")
        }
        
        let n = matrix.rows
        var vector = Array(repeating: Float(1.0 / sqrt(Float(n))), count: n)
        var eigenvalue: Float = 0
        
        for _ in 0..<iterations {
            // Multiply A * v
            let result = try multiply(matrix: matrix, vector: vector)
            
            // Compute norm
            let norm = sqrt(result.vector.reduce(0) { $0 + $1 * $1 })
            
            // Normalize
            vector = result.vector.map { $0 / norm }
            
            // Estimate eigenvalue (Rayleigh quotient)
            let Av = result.vector
            let vAv = zip(vector, Av).reduce(0) { $0 + $1.0 * $1.1 }
            let vv = vector.reduce(0) { $0 + $1 * $1 }
            let newEigenvalue = vAv / vv
            
            // Check convergence
            if abs(newEigenvalue - eigenvalue) < tolerance {
                return (newEigenvalue, vector)
            }
            
            eigenvalue = newEigenvalue
        }
        
        return (eigenvalue, vector)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark matrix-vector multiplication
    public func benchmark(sizes: [(rows: Int, cols: Int)]) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for size in sizes {
            // Generate random matrix and vector
            let matrix = Matrix.random(rows: size.rows, columns: size.cols)
            let vector = (0..<size.cols).map { _ in Float.random(in: -1...1) }
            
            // Warm-up run
            _ = try multiply(matrix: matrix, vector: vector)
            
            // Timed runs
            var times: [TimeInterval] = []
            var gflopsValues: [Double] = []
            
            for _ in 0..<5 {
                let result = try multiply(matrix: matrix, vector: vector)
                times.append(result.executionTime)
                gflopsValues.append(result.gflops)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let avgGflops = gflopsValues.reduce(0, +) / Double(gflopsValues.count)
            
            results.append(BenchmarkResult(
                matrixDimensions: "\(size.rows)x\(size.cols)",
                executionTime: avgTime,
                gflops: avgGflops
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let matrixDimensions: String
        public let executionTime: TimeInterval
        public let gflops: Double
    }
}
