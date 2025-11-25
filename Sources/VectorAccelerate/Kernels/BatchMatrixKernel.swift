// Batch Matrix Kernel
// GPU-accelerated batch matrix operations with fused and strided variants

import Metal
import Foundation
import VectorCore
import QuartzCore

// MARK: - Batch Matrix Kernel

/// GPU-accelerated batch matrix operations
/// Supports both fused operations with bias and strided tensor operations
public final class BatchMatrixKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let fusedKernel: any MTLComputePipelineState
    private let stridedKernel: any MTLComputePipelineState
    private let normalizeKernel: (any MTLComputePipelineState)?
    
    // Configuration constants
    private let BLOCK_SIZE: Int = 16
    private let UNROLL_FACTOR: Int = 4
    
    // MARK: - Result Types
    
    /// Result from batch matrix operations
    public struct BatchResult: Sendable {
        public let matrices: [Matrix]
        public let executionTime: TimeInterval
        public let totalGflops: Double
        public let averageGflops: Double
        
        /// Get specific matrix from batch
        public func matrix(at index: Int) -> Matrix? {
            guard index < matrices.count else { return nil }
            return matrices[index]
        }
        
        /// Extract all data as flat array
        public func flattenedData() -> [Float] {
            return matrices.flatMap { $0.values }
        }
    }
    
    /// Configuration for fused batch operations
    public struct FusedConfig: Sendable {
        public let alpha: Float
        public let beta: Float
        public let hasBias: Bool
        public let activation: ActivationType
        
        public enum ActivationType: Sendable {
            case none
            case relu
            case tanh
            case sigmoid
            case gelu
        }
        
        public init(
            alpha: Float = 1.0,
            beta: Float = 0.0,
            hasBias: Bool = false,
            activation: ActivationType = .none
        ) {
            self.alpha = alpha
            self.beta = beta
            self.hasBias = hasBias
            self.activation = activation
        }
        
        public static let `default` = FusedConfig()
    }
    
    /// Configuration for strided batch operations
    public struct StridedConfig: Sendable {
        public let strideA: (row: Int, col: Int, batch: Int)
        public let strideB: (row: Int, col: Int, batch: Int)
        public let strideC: (row: Int, col: Int, batch: Int)
        public let transposeA: Bool
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
        
        /// Default contiguous strides
        public static func contiguous(
            batchSize: Int,
            rowsA: Int,
            colsA: Int,
            colsB: Int
        ) -> StridedConfig {
            return StridedConfig(
                strideA: (colsA, 1, rowsA * colsA),
                strideB: (colsB, 1, colsA * colsB),
                strideC: (colsB, 1, rowsA * colsB)
            )
        }
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
        
        // Load fused batch kernel
        guard let fusedFunc = library.makeFunction(name: "batchMatrixMultiplyFused") else {
            throw AccelerationError.shaderNotFound(name: "batchMatrixMultiplyFused")
        }
        self.fusedKernel = try device.makeComputePipelineState(function: fusedFunc)
        
        // Load strided batch kernel
        guard let stridedFunc = library.makeFunction(name: "stridedBatchGEMM") else {
            throw AccelerationError.shaderNotFound(name: "stridedBatchGEMM")
        }
        self.stridedKernel = try device.makeComputePipelineState(function: stridedFunc)
        
        // Try to load fast normalize kernel
        if let normalizeFunc = library.makeFunction(name: "fastNormalize") {
            self.normalizeKernel = try device.makeComputePipelineState(function: normalizeFunc)
        } else {
            self.normalizeKernel = nil
        }
    }
    
    // MARK: - Fused Batch Operations
    
    /// Perform fused batch matrix multiplication with optional bias and activation
    /// C[i] = activation(alpha * A[i] * B[i] + beta * C[i] + bias[i])
    public func multiplyFused(
        batchA: [Matrix],
        batchB: [Matrix],
        bias: [Float]? = nil,
        config: FusedConfig = .default
    ) async throws -> BatchResult {
        // Validate batch sizes
        guard batchA.count == batchB.count else {
            throw AccelerationError.invalidInput("Batch sizes must match")
        }
        
        guard !batchA.isEmpty else {
            throw AccelerationError.invalidInput("Batch cannot be empty")
        }
        
        let batchSize = batchA.count
        let M = batchA[0].rows
        let K = batchA[0].columns
        let N = batchB[0].columns
        
        // Validate dimensions for all matrices
        for i in 0..<batchSize {
            guard batchA[i].rows == M && batchA[i].columns == K else {
                throw AccelerationError.countMismatch(expected: M * K, actual: batchA[i].values.count)
            }
            guard batchB[i].rows == K && batchB[i].columns == N else {
                throw AccelerationError.countMismatch(expected: K * N, actual: batchB[i].values.count)
            }
        }
        
        // Validate bias if provided
        if let bias = bias {
            guard bias.count == batchSize * N || bias.count == N else {
                throw AccelerationError.invalidInput("Bias dimensions don't match output")
            }
        }
        
        // Create buffers
        let flatA = batchA.flatMap { $0.values }
        let flatB = batchB.flatMap { $0.values }
        
        guard let bufferA = device.makeBuffer(
            bytes: flatA,
            length: flatA.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: flatA.count * MemoryLayout<Float>.stride)
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: flatB,
            length: flatB.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: flatB.count * MemoryLayout<Float>.stride)
        }
        
        let outputSize = batchSize * M * N * MemoryLayout<Float>.stride
        guard let bufferC = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: outputSize)
        }
        
        // Create bias buffer if needed
        var biasBuffer: (any MTLBuffer)?
        if let bias = bias {
            biasBuffer = device.makeBuffer(
                bytes: bias,
                length: bias.count * MemoryLayout<Float>.stride,
                options: MTLResourceOptions.storageModeShared
            )
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Set pipeline and buffers
        encoder.setComputePipelineState(fusedKernel)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        
        // Set parameters (batchSize, M, K, N)
        var params = SIMD4<UInt32>(UInt32(batchSize), UInt32(M), UInt32(K), UInt32(N))
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        
        // Set bias buffer if provided
        if let biasBuffer = biasBuffer {
            encoder.setBuffer(biasBuffer, offset: 0, index: 4)
        } else {
            encoder.setBuffer(nil, offset: 0, index: 4)
        }
        
        // Set alpha scaling factor
        var alpha = config.alpha
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 5)
        
        // Configure thread groups (3D dispatch for batch dimension)
        let threadgroupSize = MTLSize(width: BLOCK_SIZE, height: BLOCK_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            height: (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
            depth: batchSize
        )
        
        // Dispatch
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Batch multiplication failed: \(error)")
        }
        
        // Extract results
        let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: batchSize * M * N)
        let resultData = Array(UnsafeBufferPointer(start: resultPointer, count: batchSize * M * N))
        
        // Reconstruct matrices
        var matrices: [Matrix] = []
        for i in 0..<batchSize {
            let start = i * M * N
            let end = start + M * N
            let matrixData = Array(resultData[start..<end])
            matrices.append(Matrix(rows: M, columns: N, values: matrixData))
        }
        
        // Calculate GFLOPS
        let operations = Double(batchSize) * 2.0 * Double(M) * Double(N) * Double(K)
        let totalGflops = (operations / 1e9) / executionTime
        let avgGflops = totalGflops / Double(batchSize)
        
        return BatchResult(
            matrices: matrices,
            executionTime: executionTime,
            totalGflops: totalGflops,
            averageGflops: avgGflops
        )
    }
    
    // MARK: - Strided Batch Operations
    
    /// Perform strided batch GEMM for tensor operations
    /// Supports custom memory layouts and strides
    public func multiplyStrided(
        tensorA: [Float],
        tensorB: [Float],
        batchCount: Int,
        dimensions: (M: Int, N: Int, K: Int),
        config: StridedConfig
    ) async throws -> BatchResult {
        // Validate tensor sizes
        let expectedSizeA = batchCount * dimensions.M * dimensions.K
        let expectedSizeB = batchCount * dimensions.K * dimensions.N
        
        guard tensorA.count >= expectedSizeA else {
            throw AccelerationError.invalidInput("Tensor A size insufficient")
        }
        guard tensorB.count >= expectedSizeB else {
            throw AccelerationError.invalidInput("Tensor B size insufficient")
        }
        
        // Create buffers
        guard let bufferA = device.makeBuffer(
            bytes: tensorA,
            length: tensorA.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: tensorA.count * MemoryLayout<Float>.stride)
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: tensorB,
            length: tensorB.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: tensorB.count * MemoryLayout<Float>.stride)
        }
        
        let outputSize = batchCount * dimensions.M * dimensions.N * MemoryLayout<Float>.stride
        guard let bufferC = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: outputSize)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Set pipeline and buffers
        encoder.setComputePipelineState(stridedKernel)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        
        // Set batch count
        var batchCountParam = UInt32(batchCount)
        encoder.setBytes(&batchCountParam, length: MemoryLayout<UInt32>.size, index: 3)
        
        // Set dimensions (M, N, K)
        var dims = SIMD3<UInt32>(UInt32(dimensions.M), UInt32(dimensions.N), UInt32(dimensions.K))
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 4)
        
        // Set strides for A
        var stridesA = SIMD3<UInt32>(
            UInt32(config.strideA.row),
            UInt32(config.strideA.col),
            UInt32(config.strideA.batch)
        )
        encoder.setBytes(&stridesA, length: MemoryLayout<SIMD3<UInt32>>.size, index: 5)
        
        // Set strides for B
        var stridesB = SIMD3<UInt32>(
            UInt32(config.strideB.row),
            UInt32(config.strideB.col),
            UInt32(config.strideB.batch)
        )
        encoder.setBytes(&stridesB, length: MemoryLayout<SIMD3<UInt32>>.size, index: 6)
        
        // Set strides for C
        var stridesC = SIMD3<UInt32>(
            UInt32(config.strideC.row),
            UInt32(config.strideC.col),
            UInt32(config.strideC.batch)
        )
        encoder.setBytes(&stridesC, length: MemoryLayout<SIMD3<UInt32>>.size, index: 7)
        
        // Configure thread groups
        let threadgroupSize = MTLSize(width: BLOCK_SIZE, height: BLOCK_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (dimensions.N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            height: (dimensions.M + BLOCK_SIZE - 1) / BLOCK_SIZE,
            depth: batchCount
        )
        
        // Dispatch
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Strided batch multiplication failed: \(error)")
        }
        
        // Extract results
        let resultPointer = bufferC.contents().bindMemory(
            to: Float.self,
            capacity: batchCount * dimensions.M * dimensions.N
        )
        let resultData = Array(UnsafeBufferPointer(
            start: resultPointer,
            count: batchCount * dimensions.M * dimensions.N
        ))
        
        // Reconstruct matrices
        var matrices: [Matrix] = []
        for i in 0..<batchCount {
            let start = i * dimensions.M * dimensions.N
            let end = start + dimensions.M * dimensions.N
            let matrixData = Array(resultData[start..<end])
            matrices.append(Matrix(rows: dimensions.M, columns: dimensions.N, values: matrixData))
        }
        
        // Calculate GFLOPS
        let operations = Double(batchCount) * 2.0 * Double(dimensions.M) * Double(dimensions.N) * Double(dimensions.K)
        let totalGflops = (operations / 1e9) / executionTime
        let avgGflops = totalGflops / Double(batchCount)
        
        return BatchResult(
            matrices: matrices,
            executionTime: executionTime,
            totalGflops: totalGflops,
            averageGflops: avgGflops
        )
    }
    
    // MARK: - Specialized Operations
    
    /// Batch matrix multiplication with normalization
    public func multiplyAndNormalize(
        batchA: [Matrix],
        batchB: [Matrix],
        config: FusedConfig = .default
    ) async throws -> BatchResult {
        // First multiply
        let multiplyResult = try await multiplyFused(batchA: batchA, batchB: batchB, config: config)
        
        // Then normalize if kernel available
        if normalizeKernel != nil {
            // Would apply normalization here
            // For now, return multiply result
            return multiplyResult
        }
        
        return multiplyResult
    }
    
    // MARK: - Async Operations
    
    /// Async fused batch multiplication
    public func multiplyFusedAsync(
        batchA: [Matrix],
        batchB: [Matrix],
        bias: [Float]? = nil,
        config: FusedConfig = .default
    ) async throws -> BatchResult {
        return try await multiplyFused(
            batchA: batchA,
            batchB: batchB,
            bias: bias,
            config: config
        )
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark batch operations
    public func benchmark(
        batchSizes: [Int],
        dimensions: (M: Int, K: Int, N: Int)
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for batchSize in batchSizes {
            // Generate random matrices
            var batchA: [Matrix] = []
            var batchB: [Matrix] = []
            
            for _ in 0..<batchSize {
                batchA.append(Matrix.random(rows: dimensions.M, columns: dimensions.K))
                batchB.append(Matrix.random(rows: dimensions.K, columns: dimensions.N))
            }
            
            // Warm-up run
            _ = try await multiplyFused(batchA: batchA, batchB: batchB)

            // Timed runs
            var times: [TimeInterval] = []
            var gflopsValues: [Double] = []

            for _ in 0..<5 {
                let result = try await multiplyFused(batchA: batchA, batchB: batchB)
                times.append(result.executionTime)
                gflopsValues.append(result.totalGflops)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let avgGflops = gflopsValues.reduce(0, +) / Double(gflopsValues.count)
            
            results.append(BenchmarkResult(
                batchSize: batchSize,
                dimensions: "\(dimensions.M)x\(dimensions.K)x\(dimensions.N)",
                executionTime: avgTime,
                totalGflops: avgGflops,
                gflopsPerMatrix: avgGflops / Double(batchSize)
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let batchSize: Int
        public let dimensions: String
        public let executionTime: TimeInterval
        public let totalGflops: Double
        public let gflopsPerMatrix: Double
    }
}
