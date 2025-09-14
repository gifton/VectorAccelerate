// VectorAccelerate Cosine Similarity Kernel
// High-performance GPU-accelerated cosine similarity/distance computation

import Metal
import QuartzCore
import simd
import QuartzCore
import VectorCore
import QuartzCore

/// Cosine Similarity computation kernel for GPU acceleration
public final class CosineSimilarityKernel {
    private let device: any MTLDevice
    private let computeEngine: ComputeEngine
    
    // General purpose pipelines
    private let pipelineStateNormalized: any MTLComputePipelineState
    private let pipelineStateGeneral: any MTLComputePipelineState
    
    // Optimized pipelines
    private let pipelineState512: any MTLComputePipelineState
    private let pipelineState768: any MTLComputePipelineState
    private let pipelineState1536: any MTLComputePipelineState

    /// Parameters for cosine similarity kernel execution
    public struct Parameters {
        public var numQueries: UInt32
        public var numDatabase: UInt32
        public var dimension: UInt32
        public var strideQuery: UInt32
        public var strideDatabase: UInt32
        public var strideOutput: UInt32
        public var outputDistance: UInt8  // 0 = similarity, 1 = distance
        public var inputsNormalized: UInt8  // 0 = no, 1 = yes
        // Padding to ensure alignment matches the Metal struct (2 bytes)
        private var padding: (UInt8, UInt8) = (0, 0)

        public init(
            numQueries: Int,
            numDatabase: Int,
            dimension: Int,
            outputDistance: Bool = false,
            inputsNormalized: Bool = false,
            strideQuery: Int? = nil,
            strideDatabase: Int? = nil,
            strideOutput: Int? = nil
        ) {
            self.numQueries = UInt32(numQueries)
            self.numDatabase = UInt32(numDatabase)
            self.count = UInt32(dimension)
            // Allow custom strides, defaulting to dense packing if nil
            self.strideQuery = UInt32(strideQuery ?? dimension)
            self.strideDatabase = UInt32(strideDatabase ?? dimension)
            self.strideOutput = UInt32(strideOutput ?? numDatabase)
            self.outputDistance = outputDistance ? 1 : 0
            self.inputsNormalized = inputsNormalized ? 1 : 0
        }
    }

    /// Initialize the CosineSimilarityKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.computeEngine = try ComputeEngine(context: MetalContext(device: device))

        // Load the shader library
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }

        // Load and create pipeline states
        self.pipelineStateNormalized = try Self.makePipelineState(
            device: device,
            library: library,
            name: "cosine_similarity_normalized_kernel"
        )
        self.pipelineStateGeneral = try Self.makePipelineState(
            device: device,
            library: library,
            name: "cosine_similarity_general_kernel"
        )
        self.pipelineState512 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "cosine_similarity_512_kernel"
        )
        self.pipelineState768 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "cosine_similarity_768_kernel"
        )
        self.pipelineState1536 = try Self.makePipelineState(
            device: device,
            library: library,
            name: "cosine_similarity_1536_kernel"
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

    /// Validates buffer sizes and dimensions
    private func validateInputs(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        similarities: any MTLBuffer,
        parameters: Parameters
    ) throws {
        // 1. Validate dimensions
        if parameters.count == 0 {
            if parameters.numQueries > 0 || parameters.numDatabase > 0 {
                throw AccelerationError.invalidInput("Dimension cannot be zero when counts are greater than zero")
            }
            return // Trivial case
        }

        // 2. Validate strides
        if parameters.strideQuery < parameters.count || parameters.strideDatabase < parameters.count {
            throw AccelerationError.invalidInput("Strides cannot be smaller than the dimension")
        }

        let floatSize = MemoryLayout<Float>.stride

        // 3. Validate buffer sizes
        // The required size is precisely calculated as ((N-1) * stride + dimension) * element_size
        if parameters.numQueries > 0 {
            let requiredQuerySize = (Int(parameters.numQueries - 1) * Int(parameters.strideQuery) + Int(parameters.count)) * floatSize
            if queryVectors.length < requiredQuerySize {
                throw AccelerationError.invalidInput("Query buffer too small. Required: \(requiredQuerySize), got: \(queryVectors.length)")
            }
        }

        if parameters.numDatabase > 0 {
            let requiredDatabaseSize = (Int(parameters.numDatabase - 1) * Int(parameters.strideDatabase) + Int(parameters.count)) * floatSize
            if databaseVectors.length < requiredDatabaseSize {
                throw AccelerationError.invalidInput("Database buffer too small. Required: \(requiredDatabaseSize), got: \(databaseVectors.length)")
            }
        }

        // Output buffer validation
        if parameters.numQueries > 0 && parameters.strideOutput < parameters.numDatabase {
            throw AccelerationError.invalidInput("Output stride cannot be smaller than numDatabase")
        }

        if parameters.numQueries > 0 && parameters.numDatabase > 0 {
            let requiredOutputSize = (Int(parameters.numQueries - 1) * Int(parameters.strideOutput) + Int(parameters.numDatabase)) * floatSize
            if similarities.length < requiredOutputSize {
                throw AccelerationError.invalidInput("Similarities buffer too small. Required: \(requiredOutputSize), got: \(similarities.length)")
            }
        }
    }

    /// Compute cosine similarities between query and database vectors
    /// - Parameters:
    ///   - queryVectors: Buffer containing query vectors [N, D]
    ///   - databaseVectors: Buffer containing database vectors [M, D]
    ///   - similarities: Output buffer for similarities/distances [N, M]
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func compute(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        similarities: any MTLBuffer,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        
        // 1. Validate inputs
        try validateInputs(
            queryVectors: queryVectors,
            databaseVectors: databaseVectors,
            similarities: similarities,
            parameters: parameters
        )

        // Handle trivial case
        if parameters.numQueries == 0 || parameters.numDatabase == 0 || parameters.count == 0 {
            return
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }

        encoder.label = "CosineSimilarityKernel"

        // 2. Select optimal kernel
        // Optimized kernels require dense packing (stride == dimension) because their address calculation is hardcoded
        let selectedPipeline: any MTLComputePipelineState
        let isDenselyPacked = (parameters.strideQuery == parameters.count && 
                               parameters.strideDatabase == parameters.count)

        switch parameters.count {
        case 512 where isDenselyPacked:
            selectedPipeline = pipelineState512
        case 768 where isDenselyPacked:
            selectedPipeline = pipelineState768
        case 1536 where isDenselyPacked:
            selectedPipeline = pipelineState1536
        default:
            // Fallback for other dimensions or non-dense (padded) layouts
            // Select the appropriate general kernel based on normalization status
            if parameters.inputsNormalized != 0 {
                selectedPipeline = pipelineStateNormalized
            } else {
                selectedPipeline = pipelineStateGeneral
            }
        }

        encoder.setComputePipelineState(selectedPipeline)

        // 3. Set buffers
        encoder.setBuffer(queryVectors, offset: 0, index: 0)
        encoder.setBuffer(databaseVectors, offset: 0, index: 1)
        encoder.setBuffer(similarities, offset: 0, index: 2)

        // 4. Set parameters (using .stride to ensure correct size including padding)
        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 3)

        // 5. Dispatch configuration
        // Grid size is the total number of computations (N x M)
        let gridSize = MTLSize(
            width: Int(parameters.numQueries),
            height: Int(parameters.numDatabase),
            depth: 1
        )

        // Determine optimal threadgroup size dynamically based on pipeline characteristics
        let w = selectedPipeline.threadExecutionWidth
        // Ensure height is at least 1
        let h = max(1, selectedPipeline.maxTotalThreadsPerThreadgroup / w)
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)

        // Use dispatchThreads for efficient execution and automatic handling of edges
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
    }

    /// Compute cosine similarities with automatic buffer management
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - database: Array of database vectors
    ///   - dimension: Vector dimension
    ///   - outputDistance: If true, output distance (1 - similarity)
    ///   - inputsNormalized: If true, vectors are pre-normalized
    /// - Returns: 2D array of similarities/distances [queries.count][database.count]
    public func compute(
        queries: [[Float]],
        database: [[Float]],
        dimension: Int,
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !database.isEmpty else {
            throw AccelerationError.invalidInput("Empty input vectors")
        }
        
        guard queries.allSatisfy({ $0.count == dimension }) &&
              database.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.invalidInput("Dimension mismatch in input vectors")
        }
        
        let numQueries = queries.count
        let numDatabase = database.count
        
        // Allocate buffers
        let queryBuffer = try computeEngine.createBuffer(
            from: queries.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        )
        
        let databaseBuffer = try computeEngine.createBuffer(
            from: database.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        )
        
        let similarityBuffer = device.makeBuffer(
            length: numQueries * numDatabase * MemoryLayout<Float>.size,
            options: MTLResourceOptions.storageModeShared
        )!
        
        // Create parameters
        let parameters = Parameters(
            numQueries: numQueries,
            numDatabase: numDatabase,
            dimension: dimension,
            outputDistance: outputDistance,
            inputsNormalized: inputsNormalized
        )
        
        // Execute computation
        let commandBuffer = computeEngine.commandQueue.makeCommandBuffer()!
        try compute(
            queryVectors: queryBuffer,
            databaseVectors: databaseBuffer,
            similarities: similarityBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let similarityPointer = similarityBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numDatabase
        )
        
        var results: [[Float]] = []
        for i in 0..<numQueries {
            var row: [Float] = []
            for j in 0..<numDatabase {
                row.append(similarityPointer[i * numDatabase + j])
            }
            results.append(row)
        }
        
        return results
    }

    /// Compute cosine similarities using VectorCore types
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - database: Database vectors
    ///   - outputDistance: If true, output distance (1 - similarity)
    ///   - inputsNormalized: If true, vectors are pre-normalized
    /// - Returns: Similarity/distance matrix
    public func compute<V: VectorProtocol>(
        queries: [V],
        database: [V],
        outputDistance: Bool = false,
        inputsNormalized: Bool = false
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
        
        return try await compute(
            queries: queryArrays,
            database: databaseArrays,
            dimension: dimension,
            outputDistance: outputDistance,
            inputsNormalized: inputsNormalized
        )
    }
}

// MARK: - Performance Extensions

extension CosineSimilarityKernel {
    /// Performance statistics for kernel execution
    public struct PerformanceStats {
        public let computeTime: TimeInterval
        public let throughput: Double  // GFLOPS
        public let bandwidth: Double   // GB/s
    }
    
    /// Compute with performance monitoring
    public func computeWithStats(
        queryVectors: any MTLBuffer,
        databaseVectors: any MTLBuffer,
        similarities: any MTLBuffer,
        parameters: Parameters
    ) async throws -> PerformanceStats {
        let commandBuffer = computeEngine.commandQueue.makeCommandBuffer()!
        
        // Add GPU timing
        let startTime = CACurrentMediaTime()
        
        try compute(
            queryVectors: queryVectors,
            databaseVectors: databaseVectors,
            similarities: similarities,
            parameters: parameters,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let endTime = CACurrentMediaTime()
        let computeTime = endTime - startTime
        
        // Calculate performance metrics
        // For normalized: just dot products (2N ops per element)
        // For general: dot products + 2 norms (6N ops per element)
        let opsPerElement = parameters.inputsNormalized != 0 ? 2 : 6
        let numOps = Int(parameters.numQueries) * Int(parameters.numDatabase) * Int(parameters.count) * opsPerElement
        let throughput = Double(numOps) / (computeTime * 1e9)  // GFLOPS
        
        let bytesRead = (Int(parameters.numQueries) + Int(parameters.numDatabase)) * Int(parameters.count) * 4
        let bytesWritten = Int(parameters.numQueries) * Int(parameters.numDatabase) * 4
        let bandwidth = Double(bytesRead + bytesWritten) / (computeTime * 1e9)  // GB/s
        
        return PerformanceStats(
            computeTime: computeTime,
            throughput: throughput,
            bandwidth: bandwidth
        )
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation and testing
public func cpuCosineSimilarity(a: [Float], b: [Float], outputDistance: Bool = false) -> Float {
    guard a.count == b.count else {
        fatalError("Vector dimensions must match for cosine similarity calculation")
    }
    
    var dotProduct: Float = 0
    var normA_sq: Float = 0
    var normB_sq: Float = 0

    for i in 0..<a.count {
        let valA = a[i]
        let valB = b[i]
        dotProduct += valA * valB
        normA_sq += valA * valA
        normB_sq += valB * valB
    }

    let denominator = sqrt(normA_sq * normB_sq)
    
    // Handle zero vectors (Epsilon matches the GPU implementation)
    let epsilon: Float = 1e-8
    guard denominator > epsilon else { 
        return outputDistance ? 1.0 : 0.0 
    }

    let similarity = dotProduct / denominator
    
    // Clamp for stability
    let clampedSimilarity = max(-1.0, min(1.0, similarity))
    
    return outputDistance ? (1.0 - clampedSimilarity) : clampedSimilarity
}