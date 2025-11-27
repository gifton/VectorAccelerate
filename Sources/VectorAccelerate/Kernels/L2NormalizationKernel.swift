import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - L2 Normalization Kernel

/// L2 Normalization kernel for GPU acceleration
/// Normalizes vectors to unit length using the L2 (Euclidean) norm
public final class L2NormalizationKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext

    // Pipeline states for different variants
    private let pipelineStateGeneral: any MTLComputePipelineState
    private let pipelineState512: any MTLComputePipelineState
    private let pipelineState768: any MTLComputePipelineState
    private let pipelineState1536: any MTLComputePipelineState
    private let pipelineStateInplace: any MTLComputePipelineState

    // MARK: - Parameters Struct

    /// Parameters for L2 normalization kernel execution
    public struct Parameters {
        public var numVectors: UInt32
        public var dimension: UInt32
        public var inputStride: UInt32
        public var outputStride: UInt32
        public var epsilon: Float
        public var computeStats: UInt8
        public var storeNorms: UInt8
        private var padding: (UInt8, UInt8) = (0, 0)

        public init(
            numVectors: Int,
            dimension: Int,
            epsilon: Float = 1e-8,
            computeStats: Bool = false,
            storeNorms: Bool = false,
            inputStride: Int? = nil,
            outputStride: Int? = nil
        ) {
            self.numVectors = UInt32(numVectors)
            self.dimension = UInt32(dimension)
            self.epsilon = epsilon
            self.computeStats = computeStats ? 1 : 0
            self.storeNorms = storeNorms ? 1 : 0
            self.inputStride = UInt32(inputStride ?? dimension)
            self.outputStride = UInt32(outputStride ?? dimension)
        }
    }

    // MARK: - Initialization

    /// Initialize the L2NormalizationKernel with Metal device
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)

        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)

        // Load kernel functions
        guard let kernelGeneral = library.makeFunction(name: "l2_normalize_general_kernel"),
              let kernel512 = library.makeFunction(name: "l2_normalize_512_kernel"),
              let kernel768 = library.makeFunction(name: "l2_normalize_768_kernel"),
              let kernel1536 = library.makeFunction(name: "l2_normalize_1536_kernel"),
              let kernelInplace = library.makeFunction(name: "l2_normalize_inplace_kernel")
        else {
            throw VectorError.shaderNotFound(name: "Could not find L2 normalization kernel functions")
        }

        // Create pipeline states
        do {
            self.pipelineStateGeneral = try device.makeComputePipelineState(function: kernelGeneral)
            self.pipelineState512 = try device.makeComputePipelineState(function: kernel512)
            self.pipelineState768 = try device.makeComputePipelineState(function: kernel768)
            self.pipelineState1536 = try device.makeComputePipelineState(function: kernel1536)
            self.pipelineStateInplace = try device.makeComputePipelineState(function: kernelInplace)
        } catch {
            throw VectorError.pipelineCreationFailed("Failed to create compute pipeline state: \(error)")
        }
    }

    // MARK: - Compute Methods

    /// Normalize vectors out-of-place
    /// - Parameters:
    ///   - input: Buffer containing input vectors
    ///   - output: Buffer for normalized vectors
    ///   - norms: Optional buffer to store computed norms
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func normalize(
        input: any MTLBuffer,
        output: any MTLBuffer,
        norms: (any MTLBuffer)? = nil,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validateBuffers(input: input, output: output, norms: norms, parameters: parameters)

        if parameters.computeStats != 0 {
            throw VectorError.invalidInput("Compute stats (batch kernel) is not implemented")
        }
        
        // Select optimal pipeline
        let pipeline = selectPipeline(for: parameters)

        // Encode computation
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        encoder.label = "L2NormalizationKernel"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        
        var params = parameters
        
        // Handle optional norms buffer
        if let normsBuffer = norms {
            encoder.setBuffer(normsBuffer, offset: 0, index: 2)
        } else {
            // If no buffer is provided, ensure the GPU doesn't try to write norms
            params.storeNorms = 0
            encoder.setBuffer(nil, offset: 0, index: 2)
        }

        // Use .stride to include padding in the size calculation
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 3)

        // Dispatch configuration (1D Grid: 1 thread per vector)
        let gridSize = MTLSize(width: Int(parameters.numVectors), height: 1, depth: 1)
        
        // Determine optimal threadgroup size for 1D dispatch to maximize occupancy
        let w = pipeline.threadExecutionWidth
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        // Ensure the threadgroup size is a multiple of the execution width
        let threadsPerThreadgroup = MTLSize(width: min(maxThreads, w * (maxThreads / w)), height: 1, depth: 1)

        // Use dispatchThreads for exact control over the grid size
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    /// Normalize vectors in-place
    /// - Parameters:
    ///   - vectors: Buffer containing vectors to normalize in-place
    ///   - norms: Optional buffer to store computed norms
    ///   - parameters: Kernel execution parameters
    ///   - commandBuffer: Command buffer for GPU execution
    public func normalizeInPlace(
        vectors: any MTLBuffer,
        norms: (any MTLBuffer)? = nil,
        parameters: Parameters,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        // Validation: In-place requires input and output strides to be the same
        guard parameters.inputStride == parameters.outputStride else {
            throw VectorError.invalidInput("Input and output strides must match for in-place normalization")
        }
        
        try validateBuffers(input: vectors, output: vectors, norms: norms, parameters: parameters)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        let pipeline = pipelineStateInplace
        encoder.label = "L2NormalizationKernel-InPlace"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vectors, offset: 0, index: 0)
        
        var params = parameters
        
        if let normsBuffer = norms {
            encoder.setBuffer(normsBuffer, offset: 0, index: 1)
        } else {
            params.storeNorms = 0
            encoder.setBuffer(nil, offset: 0, index: 1)
        }

        // Note the index difference for params in the in-place signature (Index 2)
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 2)

        // Dispatch configuration
        let gridSize = MTLSize(width: Int(parameters.numVectors), height: 1, depth: 1)
        let w = pipeline.threadExecutionWidth
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSize(width: min(maxThreads, w * (maxThreads / w)), height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    // MARK: - Convenience Methods

    /// Convenience method for array-based normalization
    /// - Parameter vectors: Array of vectors to normalize
    /// - Returns: Tuple of normalized vectors and their original norms
    public func normalize(_ vectors: [[Float]]) async throws -> (normalized: [[Float]], norms: [Float]) {
        guard !vectors.isEmpty else {
            return ([], [])
        }

        let numVectors = vectors.count
        let dimension = vectors[0].count
        
        guard dimension > 0 else {
            throw VectorError.invalidInput("Vector dimension must be greater than zero")
        }
        
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        // Create buffers
        let flattenedVectors = vectors.flatMap { $0 }
        guard let inputBuffer = kernelContext.createBuffer(
            from: flattenedVectors,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }

        let totalElements = numVectors * dimension
        guard let outputBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create output buffer")
        }
        
        guard let normsBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create norms buffer")
        }

        // Configure parameters
        let parameters = Parameters(
            numVectors: numVectors,
            dimension: dimension,
            storeNorms: true
        )

        // Execute
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        try normalize(
            input: inputBuffer,
            output: outputBuffer,
            norms: normsBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()
        
        // Extract results
        let outputPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: totalElements
        )
        let normsPointer = normsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numVectors
        )

        // Reshape the output buffer back into a 2D array
        var normalizedVectors: [[Float]] = []
        normalizedVectors.reserveCapacity(numVectors)
        
        for i in 0..<numVectors {
            let start = i * dimension
            let vector = Array(UnsafeBufferPointer(
                start: outputPointer + start,
                count: dimension
            ))
            normalizedVectors.append(vector)
        }

        let norms = Array(UnsafeBufferPointer(start: normsPointer, count: numVectors))

        return (normalizedVectors, norms)
    }

    /// Normalize vectors using VectorCore types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    ///
    /// - Parameter vectors: Array of vectors conforming to VectorProtocol
    /// - Returns: Tuple of normalized vectors and their original norms
    public func normalize<V: VectorProtocol>(
        _ vectors: [V]
    ) async throws -> (normalized: [V], norms: [Float]) where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return ([], [])
        }

        let numVectors = vectors.count
        let dimension = vectors[0].count

        guard dimension > 0 else {
            throw VectorError.invalidInput("Vector dimension must be greater than zero")
        }

        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        // Zero-copy buffer creation using withUnsafeBufferPointer
        guard let inputBuffer = kernelContext.createAlignedBufferFromVectors(
            vectors,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create input buffer")
        }

        let totalElements = numVectors * dimension
        guard let outputBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create output buffer")
        }

        guard let normsBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create norms buffer")
        }

        // Configure parameters
        let parameters = Parameters(
            numVectors: numVectors,
            dimension: dimension,
            storeNorms: true
        )

        // Execute
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!

        try normalize(
            input: inputBuffer,
            output: outputBuffer,
            norms: normsBuffer,
            parameters: parameters,
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        // Extract results
        let outputPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: totalElements
        )
        let normsPointer = normsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numVectors
        )

        // Reshape the output buffer into VectorProtocol types
        var normalizedVectors: [V] = []
        normalizedVectors.reserveCapacity(numVectors)

        for i in 0..<numVectors {
            let start = i * dimension
            let vector = Array(UnsafeBufferPointer(
                start: outputPointer + start,
                count: dimension
            ))
            let v = try V(vector)
            normalizedVectors.append(v)
        }

        let norms = Array(UnsafeBufferPointer(start: normsPointer, count: numVectors))

        return (normalizedVectors, norms)
    }

    // MARK: - Performance Extensions

    /// Performance statistics for kernel execution
    public struct PerformanceStats: Sendable {
        public let vectorsPerSecond: Double
        public let executionTime: TimeInterval
        public let throughput: Double  // GB/s
    }
    
    /// Normalize with performance monitoring
    public func normalizeWithStats(
        _ vectors: [[Float]]
    ) async throws -> (normalized: [[Float]], norms: [Float], stats: PerformanceStats) {
        let startTime = CACurrentMediaTime()
        
        let (normalized, norms) = try await normalize(vectors)
        
        let endTime = CACurrentMediaTime()
        let executionTime = endTime - startTime
        
        let numVectors = Double(vectors.count)
        let dimension = vectors.isEmpty ? 0 : vectors[0].count
        let bytesProcessed = vectors.count * dimension * MemoryLayout<Float>.stride * 2 // Read + Write
        
        let stats = PerformanceStats(
            vectorsPerSecond: numVectors / executionTime,
            executionTime: executionTime,
            throughput: Double(bytesProcessed) / (executionTime * 1e9)  // GB/s
        )
        
        return (normalized, norms, stats)
    }

    // MARK: - Helper Methods

    private func selectPipeline(for parameters: Parameters) -> any MTLComputePipelineState {
        // Optimized pipelines require dense packing (stride == dimension) for both input and output
        let dimension = parameters.dimension
        guard parameters.inputStride == dimension && parameters.outputStride == dimension else {
            return pipelineStateGeneral
        }

        switch dimension {
        case 512:
            return pipelineState512
        case 768:
            return pipelineState768
        case 1536:
            return pipelineState1536
        default:
            // Fallback for non-optimized dimensions
            return pipelineStateGeneral
        }
    }

    private func validateBuffers(
        input: any MTLBuffer,
        output: any MTLBuffer,
        norms: (any MTLBuffer)?,
        parameters: Parameters
    ) throws {
        let floatSize = MemoryLayout<Float>.stride
        let numVectors = Int(parameters.numVectors)
        let dimension = Int(parameters.dimension)

        if numVectors == 0 {
            return // Nothing to do
        }

        guard dimension > 0 else {
            throw VectorError.invalidInput("Dimension must be greater than zero")
        }

        // Check strides
        if Int(parameters.inputStride) < dimension || Int(parameters.outputStride) < dimension {
            throw VectorError.invalidInput("Strides cannot be smaller than the dimension")
        }

        // Calculate required buffer sizes (accounting for strides)
        // Required Elements = (N-1) * Stride + Dimension
        let requiredInputElements = (numVectors - 1) * Int(parameters.inputStride) + dimension
        
        if input.length < requiredInputElements * floatSize {
            throw VectorError.invalidInput("Input buffer too small for the specified vectors, dimension, and stride")
        }

        // Check output buffer size only if not in-place
        if input !== output {
            let requiredOutputElements = (numVectors - 1) * Int(parameters.outputStride) + dimension
            if output.length < requiredOutputElements * floatSize {
                throw VectorError.invalidInput("Output buffer too small")
            }
        }

        if parameters.storeNorms == 1 {
            guard let normsBuffer = norms else {
                throw VectorError.invalidInput("storeNorms is requested but norms buffer is nil")
            }
            if normsBuffer.length < numVectors * floatSize {
                throw VectorError.invalidInput("Norms buffer too small")
            }
        }
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation and testing
public func cpuL2Normalize(_ vector: [Float], epsilon: Float = 1e-8) -> (normalized: [Float], norm: Float) {
    var normSquared: Float = 0
    
    for value in vector {
        normSquared += value * value
    }
    
    let norm = sqrt(normSquared)
    let invNorm = (norm > epsilon) ? (1.0 / norm) : 0.0
    
    let normalized = vector.map { $0 * invNorm }
    
    return (normalized, norm)
}
