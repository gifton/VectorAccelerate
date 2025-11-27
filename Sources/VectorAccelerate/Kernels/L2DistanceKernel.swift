// VectorAccelerate L2 Distance Kernel
// High-performance GPU-accelerated L2 distance computation

import Metal
import QuartzCore
import simd
import QuartzCore
import VectorCore
import QuartzCore

/// L2 Distance computation kernel for GPU acceleration
public final class L2DistanceKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let pipelineState: any MTLComputePipelineState
    private let pipelineState384: any MTLComputePipelineState
    private let pipelineState512: any MTLComputePipelineState
    private let pipelineState768: any MTLComputePipelineState
    private let pipelineState1536: any MTLComputePipelineState
    
    /// Parameters for L2 distance kernel execution
    public struct Parameters {
        public let numQueries: UInt32
        public let numDatabase: UInt32
        public let dimension: UInt32
        public let strideQuery: UInt32
        public let strideDatabase: UInt32
        public let strideOutput: UInt32
        public let computeSqrt: UInt8
        private let padding: (UInt8, UInt8, UInt8) = (0, 0, 0)
        
        public init(
            numQueries: Int,
            numDatabase: Int,
            dimension: Int,
            computeSqrt: Bool = true
        ) {
            self.numQueries = UInt32(numQueries)
            self.numDatabase = UInt32(numDatabase)
            self.dimension = UInt32(dimension)
            self.strideQuery = UInt32(dimension)
            self.strideDatabase = UInt32(dimension)
            self.strideOutput = UInt32(numDatabase)
            self.computeSqrt = computeSqrt ? 1 : 0
        }
        
        /// Initialize with custom strides for non-contiguous memory layouts
        public init(
            numQueries: Int,
            numDatabase: Int,
            dimension: Int,
            strideQuery: Int,
            strideDatabase: Int,
            strideOutput: Int,
            computeSqrt: Bool = true
        ) {
            self.numQueries = UInt32(numQueries)
            self.numDatabase = UInt32(numDatabase)
            self.dimension = UInt32(dimension)
            self.strideQuery = UInt32(strideQuery)
            self.strideDatabase = UInt32(strideDatabase)
            self.strideOutput = UInt32(strideOutput)
            self.computeSqrt = computeSqrt ? 1 : 0
        }
    }
    
    /// Initialize L2 distance kernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)

        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load kernel functions
        guard let kernelFunction = library.makeFunction(name: "l2_distance_kernel"),
              let kernel384 = library.makeFunction(name: "l2_distance_384_kernel"),
              let kernel512 = library.makeFunction(name: "l2_distance_512_kernel"),
              let kernel768 = library.makeFunction(name: "l2_distance_768_kernel"),
              let kernel1536 = library.makeFunction(name: "l2_distance_1536_kernel") else {
            throw VectorError.shaderNotFound(name: "L2 distance kernels not found in Metal library")
        }

        // Create pipeline states
        self.pipelineState = try device.makeComputePipelineState(function: kernelFunction)
        self.pipelineState384 = try device.makeComputePipelineState(function: kernel384)
        self.pipelineState512 = try device.makeComputePipelineState(function: kernel512)
        self.pipelineState768 = try device.makeComputePipelineState(function: kernel768)
        self.pipelineState1536 = try device.makeComputePipelineState(function: kernel1536)
    }
    
    /// Compute L2 distances between query and database vectors
    /// - Parameters:
    ///   - queryVectors: Buffer containing query vectors [N, D]
    ///   - databaseVectors: Buffer containing database vectors [M, D]
    ///   - distances: Output buffer for distances [N, M]
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func compute(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }
        
        // Select optimized kernel based on dimension
        let selectedPipeline: any MTLComputePipelineState
        switch parameters.dimension {
        case 384:
            selectedPipeline = pipelineState384
        case 512:
            selectedPipeline = pipelineState512
        case 768:
            selectedPipeline = pipelineState768
        case 1536:
            selectedPipeline = pipelineState1536
        default:
            selectedPipeline = pipelineState
        }
        
        encoder.setComputePipelineState(selectedPipeline)
        encoder.label = "L2DistanceKernel"
        
        // Set buffers
        encoder.setBuffer(queryVectors, offset: 0, index: 0)
        encoder.setBuffer(databaseVectors, offset: 0, index: 1)
        encoder.setBuffer(distances, offset: 0, index: 2)
        
        // Set parameters
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.size, index: 3)
        
        // Calculate thread groups for 2D dispatch based on device capabilities
        let _ = selectedPipeline.threadExecutionWidth
        let maxThreads = selectedPipeline.maxTotalThreadsPerThreadgroup

        // Optimize for 2D dispatch - aim for square-ish thread groups
        let threadsPerSide = Int(sqrt(Double(min(maxThreads, 256))))
        let threadWidth = min(threadsPerSide, Int(parameters.numQueries))
        let threadHeight = min(threadsPerSide, Int(parameters.numDatabase))

        let threadsPerThreadgroup = MTLSize(
            width: threadWidth,
            height: threadHeight,
            depth: 1
        )

        let threadgroups = MTLSize(
            width: (Int(parameters.numQueries) + threadWidth - 1) / threadWidth,
            height: (Int(parameters.numDatabase) + threadHeight - 1) / threadHeight,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        
        encoder.endEncoding()
    }
    
    /// Compute L2 distances with automatic buffer management
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - database: Array of database vectors
    ///   - dimension: Vector dimension
    ///   - computeSqrt: Whether to compute square root (true for Euclidean distance)
    /// - Returns: 2D array of distances [queries.count][database.count]
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        dimension: Int,
        computeSqrt: Bool = true
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }
        
        guard queries.allSatisfy({ $0.count == dimension }) &&
              database.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }
        
        let numQueries = queries.count
        let numDatabase = database.count
        
        // Allocate buffers with alignment for SIMD operations
        let flatQueries = queries.flatMap { $0 }
        let flatDatabase = database.flatMap { $0 }

        // Use aligned buffer creation for better SIMD performance
        guard let queryBuffer = kernelContext.createAlignedBuffer(
            from: flatQueries,
            options: MTLResourceOptions.storageModeShared,
            alignment: 16  // Align for float4 SIMD operations
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = kernelContext.createAlignedBuffer(
            from: flatDatabase,
            options: MTLResourceOptions.storageModeShared,
            alignment: 16  // Align for float4 SIMD operations
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create database buffer")
        }

        // Validate SIMD compatibility for optimal performance
        if dimension >= 4 {  // Only check if we can use SIMD
            if !KernelContext.isBufferSIMDCompatible(queryBuffer, elementSize: MemoryLayout<Float>.stride) {
                print("Warning: Query buffer may not be optimally aligned for SIMD operations")
            }
            if !KernelContext.isBufferSIMDCompatible(databaseBuffer, elementSize: MemoryLayout<Float>.stride) {
                print("Warning: Database buffer may not be optimally aligned for SIMD operations")
            }
        }
        
        let distanceBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        )!
        
        // Create parameters
        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )
        
        // Execute computation
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            distances: distanceBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        // Extract results
        let distancePointer = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )
        
        var results: [[Float]] = []
        for i in 0..<numQueries {
            var row: [Float] = []
            for j in 0..<numDatabase {
                row.append(distancePointer[i * numDatabase + j])
            }
            results.append(row)
        }
        
        return results
    }
    
    /// Compute L2 distances using VectorCore types
    ///
    /// This method uses zero-copy buffer creation via `withUnsafeBufferPointer`
    /// to avoid intermediate array allocations from `.toArray()`.
    ///
    /// - Parameters:
    ///   - queries: Query vectors (VectorProtocol conforming)
    ///   - database: Database vectors (VectorProtocol conforming)
    ///   - computeSqrt: Whether to compute square root (true for L2, false for squared L2)
    /// - Returns: Distance matrix [numQueries x numDatabase]
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        computeSqrt: Bool = true
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty && !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let dimension = queries.first!.count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              database.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        let numQueries = queries.count
        let numDatabase = database.count

        // Zero-copy buffer creation: uses withUnsafeBufferPointer internally
        // to copy directly from VectorProtocol storage to Metal buffer
        guard let queryBuffer = kernelContext.createAlignedBufferFromVectors(
            queries,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = kernelContext.createAlignedBufferFromVectors(
            database,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create database buffer")
        }

        // Validate SIMD compatibility for optimal performance
        if dimension >= 4 {
            if !KernelContext.isBufferSIMDCompatible(queryBuffer, elementSize: MemoryLayout<Float>.stride) {
                print("Warning: Query buffer may not be optimally aligned for SIMD operations")
            }
            if !KernelContext.isBufferSIMDCompatible(databaseBuffer, elementSize: MemoryLayout<Float>.stride) {
                print("Warning: Database buffer may not be optimally aligned for SIMD operations")
            }
        }

        let distanceBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // Create parameters
        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )

        // Execute computation
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            distances: distanceBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        // Extract results
        let distancePointer = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )

        var results: [[Float]] = []
        results.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numDatabase)
            for j in 0..<numDatabase {
                row.append(distancePointer[i * numDatabase + j])
            }
            results.append(row)
        }

        return results
    }

    // MARK: - StaticDimension Optimized Methods

    /// Compute L2 distances using compile-time dimensioned vectors.
    ///
    /// This method provides type-safe distance computation for vectors with known
    /// compile-time dimensions. The dimension is known at compile time via `D.value`,
    /// enabling:
    /// - Type safety: Cannot mix vectors of different dimensions
    /// - Compile-time optimization hints
    /// - Automatic kernel selection for optimized dimensions (384, 512, 768, 1536)
    ///
    /// - Parameters:
    ///   - queries: Query vectors with compile-time dimension D
    ///   - database: Database vectors with same dimension D
    ///   - computeSqrt: Whether to compute square root (true for L2, false for squared L2)
    /// - Returns: Distance matrix [numQueries x numDatabase]
    ///
    /// ## Example Usage
    /// ```swift
    /// let queries: [Vector<Dim384>] = [...]  // MiniLM embeddings
    /// let database: [Vector<Dim384>] = [...]
    /// let distances = try await kernel.compute(queries: queries, database: database)
    /// ```
    public func compute<D: StaticDimension>(
        queries: [Vector<D>],
        database: [Vector<D>],
        computeSqrt: Bool = true
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !database.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        // Dimension is known at compile time
        let dimension = D.value

        let numQueries = queries.count
        let numDatabase = database.count

        // Zero-copy buffer creation using compile-time known dimension
        guard let queryBuffer = kernelContext.createAlignedBufferFromVectors(
            queries,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let databaseBuffer = kernelContext.createAlignedBufferFromVectors(
            database,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create database buffer")
        }

        let distanceBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // Create parameters with compile-time known dimension
        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            computeSqrt: computeSqrt
        )

        // Execute computation
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            distances: distanceBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        // Extract results
        let distancePointer = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )

        var results: [[Float]] = []
        results.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            var row: [Float] = []
            row.reserveCapacity(numDatabase)
            for j in 0..<numDatabase {
                row.append(distancePointer[i * numDatabase + j])
            }
            results.append(row)
        }

        return results
    }

    /// Convenience type aliases for common embedding dimensions
    public typealias Vector384 = Vector<Dim384>
    public typealias Vector512 = Vector<Dim512>
    public typealias Vector768 = Vector<Dim768>
    public typealias Vector1536 = Vector<Dim1536>
}

// MARK: - Performance Extensions

extension L2DistanceKernel {
    /// Performance statistics for kernel execution
    public struct PerformanceStats: Sendable {
        public let computeTime: TimeInterval
        public let throughput: Double  // GFLOPS
        public let bandwidth: Double   // GB/s
    }
    
    /// Compute with performance monitoring
    public func computeWithStats(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        distances: any MTLBuffer,
        parameters: Parameters
    ) async throws -> PerformanceStats {
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        // Add GPU timing
        let startTime = CACurrentMediaTime()
        
        try compute(
            queryVectors: queryVectors,
            databaseVectors: databaseVectors,
            distances: distances,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        let endTime = CACurrentMediaTime()
        let computeTime = endTime - startTime
        
        // Calculate performance metrics
        let numOps = Int(parameters.numQueries) * Int(parameters.numDatabase) * Int(parameters.dimension) * 2
        let throughput = Double(numOps) / (computeTime * 1e9)  // GFLOPS
        
        let bytesRead = (Int(parameters.numQueries) + Int(parameters.numDatabase)) * Int(parameters.dimension) * 4
        let bytesWritten = Int(parameters.numQueries) * Int(parameters.numDatabase) * 4
        let bandwidth = Double(bytesRead + bytesWritten) / (computeTime * 1e9)  // GB/s
        
        return PerformanceStats(
            computeTime: computeTime,
            throughput: throughput,
            bandwidth: bandwidth
        )
    }
}
