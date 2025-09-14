//
//  MatrixEngine.swift
//  VectorAccelerate
//
//  High-performance matrix operations with GPU acceleration
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

/// Main matrix operations engine
/// Orchestrates matrix operations using specialized kernel wrappers
public actor MatrixEngine {
    private let context: MetalContext
    private let configuration: MatrixConfiguration
    private let logger: Logger
    
    // Kernel instances for direct GPU operations
    private var multiplyKernel: MatrixMultiplyKernel?
    private var transposeKernel: MatrixTransposeKernel?
    private var vectorKernel: MatrixVectorKernel?
    private var batchKernel: BatchMatrixKernel?
    
    // Legacy shader cache for fallback operations
    private var matrixMultiplyShader: (any MTLComputePipelineState)?
    private var matrixTransposeShader: (any MTLComputePipelineState)?
    private var batchMatrixMultiplyShader: (any MTLComputePipelineState)?
    
    // Performance tracking
    private var operationCount: Int = 0
    private var totalComputeTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    public init(
        context: MetalContext,
        configuration: MatrixConfiguration = .default
    ) async {
        self.context = context
        self.configuration = configuration
        self.logger = Logger.shared
        
        // Initialize kernel wrappers
        do {
            // Get the actual MTLDevice from MetalContext
            let mtlDevice = await context.device.getDevice() // MetalContext.device.getDevice() gives MTLDevice
            
            self.multiplyKernel = try MatrixMultiplyKernel(device: mtlDevice)
            self.transposeKernel = try MatrixTransposeKernel(device: mtlDevice)
            self.vectorKernel = try MatrixVectorKernel(device: mtlDevice)
            self.batchKernel = try BatchMatrixKernel(device: mtlDevice)
            await logger.info("Matrix kernels initialized successfully")
        } catch {
            await logger.warning("Failed to initialize matrix kernels: \(error)")
            // Set to nil if initialization fails - will use legacy shaders or CPU fallback
            self.multiplyKernel = nil
            self.transposeKernel = nil
            self.vectorKernel = nil
            self.batchKernel = nil
        }
        
        // Pre-compile legacy shaders as fallback
        await precompileShaders()
    }
    
    private func precompileShaders() async {
        // Only compile legacy shaders if kernel wrappers failed to initialize
        guard multiplyKernel == nil || transposeKernel == nil || batchKernel == nil else {
            await logger.debug("Skipping legacy shader compilation - kernels available")
            return
        }
        
        do {
            // Try to load pre-compiled legacy shaders as fallback
            matrixMultiplyShader = try await context.loadShader(functionName: "matrixMultiply")
            matrixTransposeShader = try await context.loadShader(functionName: "matrixTranspose")
            batchMatrixMultiplyShader = try await context.loadShader(functionName: "batchMatrixMultiply")
            await logger.info("Legacy matrix shaders compiled successfully")
        } catch {
            await logger.warning("Failed to precompile legacy matrix shaders: \(error)")
        }
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
    /// - Throws: `AccelerationError.dimensionMismatch` if A.columns ≠ B.rows
    public func multiply(
        _ matrixA: [Float],
        descriptorA: MatrixDescriptor,
        _ matrixB: [Float],
        descriptorB: MatrixDescriptor
    ) async throws -> [Float] {
        // Validate dimensions
        guard descriptorA.columns == descriptorB.rows else {
            throw AccelerationError.dimensionMismatch(
                expected: descriptorA.columns,
                actual: descriptorB.rows
            )
        }
        
        let measureToken = await logger.startMeasure("matrixMultiply")
        defer { measureToken.end() }
        
        // Decide execution path based on size
        let totalElements = descriptorA.rows * descriptorB.columns
        
        if totalElements < configuration.preferGPUThreshold {
            // Use CPU for small matrices
            return try cpuMatrixMultiply(
                matrixA, descriptorA: descriptorA,
                matrixB, descriptorB: descriptorB
            )
        } else {
            // Use GPU for large matrices
            return try await gpuMatrixMultiply(
                matrixA, descriptorA: descriptorA,
                matrixB, descriptorB: descriptorB
            )
        }
    }
    
    /// GPU-accelerated matrix multiplication
    private func gpuMatrixMultiply(
        _ matrixA: [Float],
        descriptorA: MatrixDescriptor,
        _ matrixB: [Float],
        descriptorB: MatrixDescriptor
    ) async throws -> [Float] {
        let outputRows = descriptorA.rows
        let outputCols = descriptorB.columns
        let sharedDim = descriptorA.columns
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: matrixA)
        let bufferB = try await context.getBuffer(for: matrixB)
        let outputBuffer = try await context.getBuffer(size: outputRows * outputCols * MemoryLayout<Float>.stride)
        
        // Ensure shader is loaded
        if matrixMultiplyShader == nil {
            matrixMultiplyShader = try await loadMatrixMultiplyShader()
        }
        
        guard let shader = matrixMultiplyShader else {
            throw AccelerationError.shaderNotFound(name: "matrixMultiply")
        }
        
        // Execute on GPU
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(shader)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 2)
            
            var params = (
                rowsA: UInt32(outputRows),
                colsA: UInt32(sharedDim),
                colsB: UInt32(outputCols)
            )
            encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 3)
            
            // Configure thread groups for tiled execution
            let threadgroupSize = MTLSize(
                width: min(configuration.tileSize, outputCols),
                height: min(configuration.tileSize, outputRows),
                depth: 1
            )
            
            let threadgroupCount = MTLSize(
                width: (outputCols + threadgroupSize.width - 1) / threadgroupSize.width,
                height: (outputRows + threadgroupSize.height - 1) / threadgroupSize.height,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        }
        
        // Read results
        return outputBuffer.copyData(as: Float.self, count: outputRows * outputCols)
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
        
        // Use vDSP for optimal CPU performance (modern replacement for deprecated cblas_sgemm)
        matrixA.withUnsafeBufferPointer { ptrA in
            matrixB.withUnsafeBufferPointer { ptrB in
                result.withUnsafeMutableBufferPointer { ptrC in
                    // Using vDSP_mmul for matrix multiplication
                    // Note: vDSP_mmul expects column-major for B, so we use the transposed version
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
    /// Implementation uses cache blocking for matrices larger than L2 cache
    /// to minimize cache misses during the transpose operation.
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
        
        // For small matrices, use CPU
        if descriptor.elementCount < configuration.preferGPUThreshold {
            return cpuTranspose(matrix, descriptor: descriptor)
        } else {
            // Use new kernel wrapper if available
            if let kernel = transposeKernel {
                let mat = Matrix(
                    rows: descriptor.rows,
                    columns: descriptor.columns,
                    values: matrix
                )
                
                let config = MatrixTransposeKernel.TransposeConfig(
                    conjugate: false,
                    inPlace: descriptor.rows == descriptor.columns
                )
                
                let result = try kernel.transpose(mat, config: config)
                
                operationCount += 1
                totalComputeTime += result.executionTime
                
                await logger.debug("Matrix transpose completed: \(result.throughputGBps) GB/s")
                
                return result.matrix.values
            } else {
                return try await gpuTranspose(matrix, descriptor: descriptor)
            }
        }
    }
    
    /// GPU-accelerated transpose (legacy fallback)
    private func gpuTranspose(
        _ matrix: [Float],
        descriptor: MatrixDescriptor
    ) async throws -> [Float] {
        // This method is now legacy - used only if kernel wrapper is unavailable
        // The new transposeKernel should handle all GPU transpose operations
        await logger.warning("Using legacy GPU transpose - kernel wrapper unavailable")
        
        // Fallback to CPU implementation
        return cpuTranspose(matrix, descriptor: descriptor)
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
            throw AccelerationError.dimensionMismatch(
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
            throw AccelerationError.dimensionMismatch(
                expected: descriptor.columns,
                actual: vector.count
            )
        }
        
        // Use vDSP for efficient computation (modern replacement for deprecated cblas_sgemv)
        var result = [Float](repeating: 0, count: descriptor.rows)
        
        matrix.withUnsafeBufferPointer { matPtr in
            vector.withUnsafeBufferPointer { vecPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    // Using vDSP_mmul treating vector as a column matrix
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
    
    // MARK: - Shader Loading
    
    private func loadMatrixMultiplyShader() async throws -> any MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrixMultiply(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C [[buffer(2)]],
            constant uint3& params [[buffer(3)]],  // rowsA, colsA, colsB
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint rowsA = params.x;
            uint colsA = params.y;
            uint colsB = params.z;
            
            uint row = gid.y;
            uint col = gid.x;
            
            if (row >= rowsA || col >= colsB) return;
            
            float sum = 0.0f;
            for (uint k = 0; k < colsA; k++) {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            
            C[row * colsB + col] = sum;
        }
        """
        
        return try await context.compileShader(source: source, functionName: "matrixMultiply")
    }
    
    private func loadTransposeShader() async throws -> any MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrixTranspose(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint2& dims [[buffer(2)]],  // rows, cols
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint rows = dims.x;
            uint cols = dims.y;
            
            uint row = gid.y;
            uint col = gid.x;
            
            if (row >= rows || col >= cols) return;
            
            // Input: row-major [row][col]
            // Output: row-major transposed [col][row]
            output[col * rows + row] = input[row * cols + col];
        }
        """
        
        return try await context.compileShader(source: source, functionName: "matrixTranspose")
    }
    
    // MARK: - Performance Metrics
    
    public func getPerformanceMetrics() -> (operations: Int, averageTime: TimeInterval) {
        let avgTime = operationCount > 0 ? totalComputeTime / Double(operationCount) : 0
        return (operationCount, avgTime)
    }
}

// MARK: - Convenience Extensions

public extension MatrixEngine {
    /// Create with default context
    static func createDefault() async throws -> MatrixEngine {
        guard let context = await MetalContext.createDefault() else {
            throw AccelerationError.deviceInitializationFailed("Metal not available")
        }
        return await MatrixEngine(context: context)
    }
}
