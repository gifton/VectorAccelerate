// VectorAccelerate Dot Product Kernel
// High-performance GPU-accelerated dot product computation
// Includes specialized GEMV optimization for single-query scenarios

import Metal
import QuartzCore
import simd
import QuartzCore
import VectorCore
import QuartzCore

/// Dot Product computation kernel for GPU acceleration
public final class DotProductKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    
    // GEMM (General Matrix-Matrix style) pipelines
    private let pipelineState: any MTLComputePipelineState
    private let pipelineState512: any MTLComputePipelineState
    private let pipelineState768: any MTLComputePipelineState
    private let pipelineState1536: any MTLComputePipelineState
    
    // GEMV (General Matrix-Vector style) pipeline
    private let pipelineStateGEMV: any MTLComputePipelineState

    /// Parameters for dot product kernel execution
    public struct Parameters {
        public var numQueries: UInt32
        public var numDatabase: UInt32
        public var dimension: UInt32
        public var strideQuery: UInt32
        public var strideDatabase: UInt32
        public var strideOutput: UInt32
        public var absoluteValue: UInt8  // 0 = normal, 1 = absolute
        // Padding to ensure alignment matches the Metal struct (3 bytes)
        private var padding: (UInt8, UInt8, UInt8) = (0, 0, 0)

        public init(
            numQueries: Int,
            numDatabase: Int,
            dimension: Int,
            absoluteValue: Bool = false,
            strideQuery: Int? = nil,
            strideDatabase: Int? = nil,
            strideOutput: Int? = nil
        ) {
            self.numQueries = UInt32(numQueries)
            self.numDatabase = UInt32(numDatabase)
            self.dimension = UInt32(dimension)
            // Allow custom strides, defaulting to dense packing if nil
            self.strideQuery = UInt32(strideQuery ?? dimension)
            self.strideDatabase = UInt32(strideDatabase ?? dimension)
            // For GEMV (N=1), output stride is often irrelevant/1. For GEMM (N>1), it's usually numDatabase.
            self.strideOutput = UInt32(strideOutput ?? (numQueries > 1 ? numDatabase : 1))
            self.absoluteValue = absoluteValue ? 1 : 0
        }
    }

    /// Initialize the DotProductKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)

        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)

        // Load and create pipeline states
        self.pipelineState = try Self.makePipelineState(
            device: device,
            library: library,
            name: "dot_product_kernel"
        )
        self.pipelineState512 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "dot_product_512_kernel"
        )
        self.pipelineState768 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "dot_product_768_kernel"
        )
        self.pipelineState1536 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "dot_product_1536_kernel"
        )
        self.pipelineStateGEMV = try Self.makePipelineState(
            device: device,
            library: library,
            name: "dot_product_gemv_kernel"
        )
    }

    private static func makePipelineState(
        device: any MTLDevice,
        library: any MTLLibrary,
        name: String
    ) throws -> any MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw AccelerationError.shaderNotFound(name: "Could not find kernel function '\(name)'")
        }
        return try device.makeComputePipelineState(function: function)
    }

    /// Validates buffer sizes and dimensions (Covers both GEMM and GEMV scenarios)
    private func validateInputs(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        dotProducts: any MTLBuffer,
        parameters: Parameters
    ) throws {
        // 1. Validate dimensions
        if parameters.dimension == 0 {
            if parameters.numQueries > 0 || parameters.numDatabase > 0 {
                throw AccelerationError.invalidInput("Dimension cannot be zero when counts are greater than zero")
            }
            return // Trivial case
        }

        // 2. Validate strides
        if parameters.strideQuery < parameters.dimension || parameters.strideDatabase < parameters.dimension {
            throw AccelerationError.invalidInput("Strides cannot be smaller than the dimension")
        }

        let floatSize = MemoryLayout<Float>.stride

        // 3. Validate input buffer sizes
        if parameters.numQueries > 0 {
            // Required size: ((N-1) * stride + dimension) * element_size
            let requiredQuerySize = (Int(parameters.numQueries - 1) * Int(parameters.strideQuery) + Int(parameters.dimension)) * floatSize
            if queryVectors.length < requiredQuerySize {
                throw AccelerationError.invalidInput("Query buffer too small. Required: \(requiredQuerySize), got: \(queryVectors.length)")
            }
        }

        if parameters.numDatabase > 0 {
            let requiredDatabaseSize = (Int(parameters.numDatabase - 1) * Int(parameters.strideDatabase) + Int(parameters.dimension)) * floatSize
            if databaseVectors.length < requiredDatabaseSize {
                throw AccelerationError.invalidInput("Database buffer too small. Required: \(requiredDatabaseSize), got: \(databaseVectors.length)")
            }
        }

        // 4. Validate output buffer size
        if parameters.numQueries > 0 && parameters.numDatabase > 0 {
            if parameters.numQueries == 1 {
                // GEMV Case: Output size is simply numDatabase * element_size (assuming dense output)
                let requiredOutputSize = Int(parameters.numDatabase) * floatSize
                if dotProducts.length < requiredOutputSize {
                    throw AccelerationError.invalidInput("DotProducts buffer too small (GEMV). Required: \(requiredOutputSize), got: \(dotProducts.length)")
                }
            } else {
                // GEMM Case
                if parameters.strideOutput < parameters.numDatabase {
                    throw AccelerationError.invalidInput("Output stride cannot be smaller than numDatabase for batch processing")
                }
                // Output size calculation: ((N_queries-1) * strideOutput + N_database) * element_size
                let requiredOutputSize = (Int(parameters.numQueries - 1) * Int(parameters.strideOutput) + Int(parameters.numDatabase)) * floatSize
                if dotProducts.length < requiredOutputSize {
                    throw AccelerationError.invalidInput("DotProducts buffer too small (GEMM). Required: \(requiredOutputSize), got: \(dotProducts.length)")
                }
            }
        }
    }

    /// Compute dot products between query and database vectors
    /// Automatically selects GEMV or GEMM path based on number of queries
    /// - Parameters:
    ///   - queryVectors: Buffer containing query vectors [N, D]
    ///   - databaseVectors: Buffer containing database vectors [M, D]
    ///   - dotProducts: Output buffer for dot products [N, M] or [M] for GEMV
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func compute(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        dotProducts: any MTLBuffer,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        
        // 1. Validate inputs
        try validateInputs(
            queryVectors: queryVectors,
            databaseVectors: databaseVectors,
            dotProducts: dotProducts,
            parameters: parameters
        )

        // Handle trivial case
        if parameters.numQueries == 0 || parameters.numDatabase == 0 || parameters.dimension == 0 {
            return
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }

        // 2. Select optimal kernel (GEMV vs GEMM)
        let selectedPipeline: any MTLComputePipelineState
        let gridSize: MTLSize
        let isGEMV = parameters.numQueries == 1
        
        if isGEMV {
            // Use the optimized GEMV path (Spec Section 3.3, 7.2)
            selectedPipeline = pipelineStateGEMV
            // 1D Grid size (M)
            gridSize = MTLSize(width: Int(parameters.numDatabase), height: 1, depth: 1)
            encoder.label = "DotProductGEMV"
        } else {
            // Use the general GEMM path
            encoder.label = "DotProductGEMM"
            // 2D Grid size (N x M)
            gridSize = MTLSize(
                width: Int(parameters.numQueries),
                height: Int(parameters.numDatabase),
                depth: 1
            )
            
            // Select optimized GEMM kernel if densely packed
            let isDenselyPacked = (parameters.strideQuery == parameters.dimension && 
                                  parameters.strideDatabase == parameters.dimension)

            switch parameters.dimension {
            case 512 where isDenselyPacked:
                selectedPipeline = pipelineState512
            case 768 where isDenselyPacked:
                selectedPipeline = pipelineState768
            case 1536 where isDenselyPacked:
                selectedPipeline = pipelineState1536
            default:
                // Fallback to the general kernel for other dimensions or non-dense layouts
                selectedPipeline = pipelineState
            }
        }

        encoder.setComputePipelineState(selectedPipeline)

        // 3. Set buffers
        encoder.setBuffer(queryVectors, offset: 0, index: 0)
        encoder.setBuffer(databaseVectors, offset: 0, index: 1)
        encoder.setBuffer(dotProducts, offset: 0, index: 2)

        // 4. Set parameters (using .stride to ensure correct size including padding)
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 3)

        // 5. Dispatch configuration
        // Determine optimal threadgroup size dynamically based on pipeline characteristics
        
        let threadsPerThreadgroup: MTLSize
        let maxThreads = selectedPipeline.maxTotalThreadsPerThreadgroup
        let executionWidth = selectedPipeline.threadExecutionWidth

        if isGEMV {
            // For GEMV (1D dispatch), maximize width for better memory coalescing and occupancy
            // Ensure it's a multiple of the thread execution width
            let width = executionWidth * (maxThreads / executionWidth)
            threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        } else {
            // For GEMM (2D dispatch), balance width and height
            let h = max(1, maxThreads / executionWidth)
            threadsPerThreadgroup = MTLSize(width: executionWidth, height: h, depth: 1)
        }

        // Use dispatchThreads. This handles edge cases automatically (requires bounds checking in the kernel).
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
    }

    /// Compute dot product for a single query vector against multiple database vectors
    /// Optimized GEMV path for maximum performance
    /// - Parameters:
    ///   - query: Single query vector
    ///   - database: Array of database vectors
    ///   - absoluteValue: If true, return absolute value of dot products
    /// - Returns: Array of dot products
    public func computeSingle(
        query: [Float],
        database: [[Float]],
        absoluteValue: Bool = false
    ) async throws -> [Float] {
        guard !database.isEmpty else {
            throw AccelerationError.invalidInput("Empty database vectors")
        }
        
        let dimension = query.count
        guard database.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.invalidInput("Dimension mismatch in database vectors")
        }
        
        let numDatabase = database.count
        
        // Allocate buffers
        guard let queryBuffer = kernelContext.createBuffer(
            from: query,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = kernelContext.createBuffer(
            from: database.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create database buffer")
        }
        
        let resultBuffer = device.makeBuffer(
            length: numDatabase * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        )!
        
        // Create parameters for GEMV
        let parameters = Parameters(
            numQueries: 1,
            numDatabase: numDatabase,
            dimension: dimension,
            absoluteValue: absoluteValue
        )
        
        // Execute computation
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            dotProducts: resultBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        // Extract results
        let resultPointer = resultBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numDatabase
        )
        
        return Array(UnsafeBufferPointer(start: resultPointer, count: numDatabase))
    }

    /// Compute dot products for multiple query vectors (batch processing)
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - database: Array of database vectors
    ///   - absoluteValue: If true, return absolute value of dot products
    /// - Returns: 2D array of dot products [queries.count][database.count]
    public func computeBatch(
        queries: [[Float]],
        database: [[Float]],
        absoluteValue: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !database.isEmpty else {
            throw AccelerationError.invalidInput("Empty input vectors")
        }
        
        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              database.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.invalidInput("Dimension mismatch in input vectors")
        }
        
        // Special case: single query - use GEMV kernel
        if queries.count == 1 {
            let results = try await computeSingle(
                query: queries[0],
                database: database,
                absoluteValue: absoluteValue
            )
            return [results]
        }
        
        let numQueries = queries.count
        let numDatabase = database.count
        
        // Allocate buffers
        guard let queryBuffer = kernelContext.createBuffer(
            from: queries.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = kernelContext.createBuffer(
            from: database.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create database buffer")
        }
        
        let dotProductBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        )!
        
        // Create parameters
        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            absoluteValue: absoluteValue
        )
        
        // Execute computation
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            dotProducts: dotProductBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        // Extract results
        let dotProductPointer = dotProductBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )
        
        var results: [[Float]] = []
        for i in 0..<numQueries {
            var row: [Float] = []
            for j in 0..<numDatabase {
                row.append(dotProductPointer[i * numDatabase + j])
            }
            results.append(row)
        }
        
        return results
    }

    /// Compute dot products using VectorCore types
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - database: Database vectors
    ///   - absoluteValue: If true, return absolute value of dot products
    /// - Returns: Dot product matrix
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        absoluteValue: Bool = false
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty && !database.isEmpty else {
            throw AccelerationError.invalidInput("Empty input vectors")
        }
        
        let dimension = queries.first!.count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              database.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.invalidInput("Dimension mismatch in input vectors")
        }
        
        // Convert to arrays and compute
        let queryArrays = queries.map { Array($0.toArray()) }
        let databaseArrays = database.map { Array($0.toArray()) }
        
        return try await computeBatch(
            queries: queryArrays,
            database: databaseArrays,
            absoluteValue: absoluteValue
        )
    }
}

// MARK: - Performance Extensions

extension DotProductKernel {
    /// Performance statistics for kernel execution
    public struct PerformanceStats: Sendable {
        public let computeTime: TimeInterval
        public let throughput: Double  // GFLOPS
        public let bandwidth: Double   // GB/s
        public let kernelType: String  // "GEMV" or "GEMM-512" etc.
    }
    
    /// Compute with performance monitoring
    public func computeWithStats(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        dotProducts: any MTLBuffer,
        parameters: Parameters
    ) async throws -> PerformanceStats {
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        // Determine kernel type for reporting
        let isGEMV = parameters.numQueries == 1
        let kernelType: String
        if isGEMV {
            kernelType = "GEMV"
        } else {
            let isDenselyPacked = (parameters.strideQuery == parameters.dimension && 
                                  parameters.strideDatabase == parameters.dimension)
            switch parameters.dimension {
            case 512 where isDenselyPacked:
                kernelType = "GEMM-512"
            case 768 where isDenselyPacked:
                kernelType = "GEMM-768"
            case 1536 where isDenselyPacked:
                kernelType = "GEMM-1536"
            default:
                kernelType = "GEMM-General"
            }
        }
        
        // Add GPU timing
        let startTime = CACurrentMediaTime()
        
        try compute(
            queryVectors: queryVectors,
            databaseVectors: databaseVectors,
            dotProducts: dotProducts,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        _ = await commandBuffer.completed
        
        let endTime = CACurrentMediaTime()
        let computeTime = endTime - startTime
        
        // Calculate performance metrics
        // Dot product: 2N operations per element (multiply + add)
        let numOps = Int(parameters.numQueries) * Int(parameters.numDatabase) * Int(parameters.dimension) * 2
        let throughput = Double(numOps) / (computeTime * 1e9)  // GFLOPS
        
        let bytesRead = (Int(parameters.numQueries) + Int(parameters.numDatabase)) * Int(parameters.dimension) * 4
        let bytesWritten = Int(parameters.numQueries) * Int(parameters.numDatabase) * 4
        let bandwidth = Double(bytesRead + bytesWritten) / (computeTime * 1e9)  // GB/s
        
        return PerformanceStats(
            computeTime: computeTime,
            throughput: throughput,
            bandwidth: bandwidth,
            kernelType: kernelType
        )
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation and testing
public func cpuDotProduct(a: [Float], b: [Float], absoluteValue: Bool = false) -> Float {
    guard a.count == b.count else {
        fatalError("Vector dimensions must match for dot product calculation")
    }
    
    var sum: Float = 0
    
    // For high-performance CPU validation, could use Accelerate framework
    for i in 0..<a.count {
        sum += a[i] * b[i]
    }
    
    return absoluteValue ? abs(sum) : sum
}
