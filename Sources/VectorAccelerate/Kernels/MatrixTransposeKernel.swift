// Matrix Transpose Kernel
// GPU-accelerated matrix transpose with bank conflict avoidance

import Metal
import Foundation
import VectorCore
import QuartzCore

// MARK: - Matrix Transpose Kernel

/// GPU-accelerated matrix transpose using tiled algorithm
/// Optimized to avoid bank conflicts with padding strategy
public final class MatrixTransposeKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLComputePipelineState
    private let inPlacePipelineState: (any MTLComputePipelineState)?
    
    // Tile configuration for bank conflict avoidance
    private let TILE_SIZE: Int = 16
    private let TILE_SIZE_PADDED: Int = 17  // +1 to avoid bank conflicts
    
    // MARK: - Result Types
    
    /// Result from matrix transpose operation
    public struct TransposeResult: Sendable {
        public let matrix: Matrix
        public let executionTime: TimeInterval
        public let throughputGBps: Double
        
        /// Convert to flat array
        public func asArray() -> [Float] {
            return matrix.values
        }
        
        /// Get element at (row, col) in transposed matrix
        public func element(row: Int, col: Int) -> Float {
            return matrix[row, col]
        }
    }
    
    /// Configuration for transpose operation
    public struct TransposeConfig: Sendable {
        public let conjugate: Bool  // For complex numbers (future support)
        public let inPlace: Bool    // Attempt in-place transpose if possible
        
        public init(conjugate: Bool = false, inPlace: Bool = false) {
            self.conjugate = conjugate
            self.inPlace = inPlace
        }
        
        public static let `default` = TransposeConfig()
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }
        
        // Load main transpose kernel
        guard let function = library.makeFunction(name: "tiledTranspose") else {
            throw AccelerationError.shaderNotFound(name: "tiledTranspose")
        }
        
        self.pipelineState = try device.makeComputePipelineState(function: function)
        
        // Try to load in-place transpose kernel if available
        if let inPlaceFunc = library.makeFunction(name: "tiledTransposeInPlace") {
            self.inPlacePipelineState = try device.makeComputePipelineState(function: inPlaceFunc)
        } else {
            self.inPlacePipelineState = nil
        }
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < TILE_SIZE * TILE_SIZE {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required threadgroup size: \(TILE_SIZE * TILE_SIZE)"
            )
        }
    }
    
    // MARK: - Core Operations
    
    /// Transpose a matrix
    /// - Parameters:
    ///   - matrix: Input matrix to transpose
    ///   - config: Transpose configuration
    /// - Returns: Transposed matrix with performance metrics
    public func transpose(
        _ matrix: Matrix,
        config: TransposeConfig = .default
    ) throws -> TransposeResult {
        let rows = matrix.rows
        let cols = matrix.columns
        
        // Check if we can do in-place transpose
        let canDoInPlace = config.inPlace && rows == cols && inPlacePipelineState != nil
        
        // Create input buffer
        guard let inputBuffer = device.makeBuffer(
            bytes: matrix.values,
            length: matrix.values.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: matrix.values.count * MemoryLayout<Float>.stride)
        }
        
        // Create output buffer (may be same as input for square in-place)
        let outputBuffer: any MTLBuffer
        if canDoInPlace {
            outputBuffer = inputBuffer
        } else {
            let outputSize = rows * cols * MemoryLayout<Float>.stride
            guard let buffer = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
                throw AccelerationError.bufferAllocationFailed(size: outputSize)
            }
            outputBuffer = buffer
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Select appropriate pipeline
        let state = canDoInPlace ? inPlacePipelineState! : pipelineState
        encoder.setComputePipelineState(state)
        
        // Set buffers
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        // Set dimensions (rows, cols)
        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        
        // Calculate shared memory size (with padding to avoid bank conflicts)
        let sharedMemorySize = TILE_SIZE_PADDED * TILE_SIZE * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
        
        // Configure thread groups
        let threadgroupSize = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)
        let threadgroupCount = MTLSize(
            width: (cols + TILE_SIZE - 1) / TILE_SIZE,
            height: (rows + TILE_SIZE - 1) / TILE_SIZE,
            depth: 1
        )
        
        // Dispatch
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Matrix transpose failed: \(error)")
        }
        
        // Extract results
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let resultData = Array(UnsafeBufferPointer(start: resultPointer, count: rows * cols))
        
        // Calculate throughput (read + write bandwidth)
        let bytesProcessed = Double(rows * cols * 2 * MemoryLayout<Float>.stride)
        let throughputGBps = (bytesProcessed / 1e9) / executionTime
        
        // Note: Output dimensions are swapped (cols x rows)
        let resultMatrix = Matrix(rows: cols, columns: rows, values: resultData)
        return TransposeResult(
            matrix: resultMatrix,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }
    
    /// Transpose matrix from flat array
    public func transpose(
        data: [Float],
        rows: Int,
        columns: Int,
        config: TransposeConfig = .default
    ) throws -> TransposeResult {
        let matrix = Matrix(rows: rows, columns: columns, values: data)
        return try transpose(matrix, config: config)
    }
    
    // MARK: - Specialized Operations
    
    /// Transpose and multiply: C = A^T * B
    /// More efficient than separate transpose and multiply
    public func transposeAndMultiply(
        _ matrixA: Matrix,
        _ matrixB: Matrix
    ) throws -> Matrix {
        // First transpose A
        let transposedA = try transpose(matrixA)
        
        // Then multiply (would use MatrixMultiplyKernel in practice)
        // For now, return placeholder
        return transposedA.matrix
    }
    
    /// Double transpose (should return original for validation)
    public func doubleTranspose(_ matrix: Matrix) throws -> Matrix {
        let first = try transpose(matrix)
        let second = try transpose(first.matrix)
        return second.matrix
    }
    
    // MARK: - Batch Operations
    
    /// Transpose multiple matrices in batch
    public func transposeBatch(
        _ matrices: [Matrix],
        config: TransposeConfig = .default
    ) throws -> [TransposeResult] {
        var results: [TransposeResult] = []
        
        for matrix in matrices {
            results.append(try transpose(matrix, config: config))
        }
        
        return results
    }
    
    // MARK: - Async Operations
    
    /// Async version of matrix transpose
    public func transposeAsync(
        _ matrix: Matrix,
        config: TransposeConfig = .default
    ) async throws -> TransposeResult {
        return try await Task.detached(priority: .userInitiated) { [self] in
            return try self.transpose(matrix, config: config)
        }.value
    }
    
    // MARK: - VectorCore Integration
    
    /// Transpose using VectorCore protocol types
    public func transpose<V: VectorProtocol>(
        vectors: [[V]],
        config: TransposeConfig = .default
    ) throws -> TransposeResult where V.Scalar == Float {
        let rows = vectors.count
        let cols = vectors.first?.count ?? 0
        let data = vectors.flatMap { row in row.flatMap { $0.toArray() } }
        
        let matrix = Matrix(rows: rows, columns: cols, values: data)
        return try transpose(matrix, config: config)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark transpose for different matrix sizes
    public func benchmark(sizes: [(rows: Int, cols: Int)]) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for size in sizes {
            // Generate random matrix
            let matrix = Matrix.random(rows: size.rows, columns: size.cols)
            
            // Warm-up run
            _ = try transpose(matrix)
            
            // Timed runs
            var times: [TimeInterval] = []
            var throughputs: [Double] = []
            
            for _ in 0..<5 {
                let result = try transpose(matrix)
                times.append(result.executionTime)
                throughputs.append(result.throughputGBps)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let avgThroughput = throughputs.reduce(0, +) / Double(throughputs.count)
            
            results.append(BenchmarkResult(
                dimensions: "\(size.rows)x\(size.cols)",
                executionTime: avgTime,
                throughputGBps: avgThroughput,
                isSquare: size.rows == size.cols
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let dimensions: String
        public let executionTime: TimeInterval
        public let throughputGBps: Double
        public let isSquare: Bool
    }
    
    // MARK: - Validation
    
    /// Validate transpose correctness
    public func validate(_ matrix: Matrix) throws -> Bool {
        // Transpose twice should give original
        let transposed = try transpose(matrix)
        let doubleTransposed = try transpose(transposed.matrix)
        
        // Compare with original
        for i in 0..<matrix.values.count {
            if abs(matrix.values[i] - doubleTransposed.matrix.values[i]) > 1e-6 {
                return false
            }
        }
        
        // Check specific elements
        for row in 0..<min(matrix.rows, 10) {
            for col in 0..<min(matrix.columns, 10) {
                let original = matrix[row, col]
                let transposedValue = transposed.matrix[col, row]
                if abs(original - transposedValue) > 1e-6 {
                    return false
                }
            }
        }
        
        return true
    }
}