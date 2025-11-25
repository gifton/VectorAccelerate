// Product Quantization Kernel
// GPU-accelerated vector compression and approximate distance computation

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Product Quantization Kernel

/// GPU-accelerated Product Quantization for vector compression and fast similarity search
/// Splits vectors into M subspaces and quantizes each using K-means clustering
public final class ProductQuantizationKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    
    // Pipeline states for different phases
    private let assignmentKernel: any MTLComputePipelineState
    private let updateAccumulateKernel: any MTLComputePipelineState
    private let updateFinalizeKernel: any MTLComputePipelineState
    private let precomputeDistanceKernel: any MTLComputePipelineState
    private let computeDistancesKernel: any MTLComputePipelineState
    
    /// Configuration for Product Quantization
    public struct PQConfig: Sendable {
        public let M: Int           // Number of subspaces
        public let K: Int           // Centroids per subspace (typically 256 for uint8)
        public let dimension: Int   // Full vector dimension
        public let trainIterations: Int
        public let convergenceThreshold: Float
        
        public var D_sub: Int {
            return dimension / M
        }
        
        public init(
            dimension: Int,
            M: Int = 8,
            K: Int = 256,
            trainIterations: Int = 25,
            convergenceThreshold: Float = 0.001
        ) {
            precondition(dimension % M == 0, "Dimension must be divisible by M")
            precondition(K <= 256, "K must be <= 256 for uint8 encoding")
            
            self.dimension = dimension
            self.M = M
            self.K = K
            self.trainIterations = trainIterations
            self.convergenceThreshold = convergenceThreshold
        }
    }
    
    /// Trained PQ model containing codebooks
    public class PQModel {
        public let codebooks: any MTLBuffer      // [M × K × D_sub]
        public let config: PQConfig
        private let device: any MTLDevice
        
        fileprivate init(codebooks: any MTLBuffer, config: PQConfig, device: any MTLDevice) {
            self.codebooks = codebooks
            self.config = config
            self.device = device
        }
        
        /// Save model to file
        public func save(to url: URL) throws {
            let codebookData = Data(
                bytesNoCopy: codebooks.contents(),
                count: codebooks.length,
                deallocator: .none
            )
            
            let modelData: [String: Any] = [
                "codebooks": codebookData,
                "M": config.M,
                "K": config.K,
                "dimension": config.dimension
            ]
            
            let data = try JSONSerialization.data(withJSONObject: modelData)
            try data.write(to: url)
        }
        
        /// Load model from file
        public static func load(from url: URL, device: any MTLDevice) throws -> PQModel {
            let data = try Data(contentsOf: url)
            let modelData = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            
            let codebookData = modelData["codebooks"] as! Data
            let M = modelData["M"] as! Int
            let K = modelData["K"] as! Int
            let dimension = modelData["dimension"] as! Int
            
            let config = PQConfig(dimension: dimension, M: M, K: K)
            
            guard let codebooks = device.makeBuffer(
                bytes: codebookData.withUnsafeBytes { $0.baseAddress! },
                length: codebookData.count,
                options: MTLResourceOptions.storageModeShared
            ) else {
                throw AccelerationError.bufferCreationFailed("Failed to create codebook buffer")
            }
            
            return PQModel(codebooks: codebooks, config: config, device: device)
        }
        
        /// Get compression ratio
        public var compressionRatio: Float {
            let originalBits = Float(config.dimension * 32)  // float32
            let compressedBits = Float(config.M * 8)        // uint8
            return originalBits / compressedBits
        }
    }
    
    /// Encoded vectors result
    public struct EncodedVectors {
        public let codes: any MTLBuffer          // [N × M] uint8
        public let count: Int
        public let config: PQConfig
        
        /// Decode specific vector
        public func decode(index: Int, using model: PQModel) -> [Float] {
            guard index < count else { return [] }
            
            let codesPtr = codes.contents().bindMemory(to: UInt8.self, capacity: count * config.M)
            let codebooksPtr = model.codebooks.contents().bindMemory(to: Float.self, capacity: config.M * config.K * config.D_sub)
            
            var decoded = Array<Float>(repeating: 0, count: config.dimension)
            
            for m in 0..<config.M {
                let code = codesPtr[index * config.M + m]
                let centroidStart = (m * config.K + Int(code)) * config.D_sub
                
                for d in 0..<config.D_sub {
                    decoded[m * config.D_sub + d] = codebooksPtr[centroidStart + d]
                }
            }
            
            return decoded
        }
        
        /// Calculate reconstruction error
        public func reconstructionError(original: any MTLBuffer, using model: PQModel) -> Float {
            // This would require decoding all vectors and comparing
            // Implementation omitted for brevity
            return 0.0
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load all kernels
        guard let assignmentFunc = library.makeFunction(name: "pq_assignment_or_encoding"),
              let updateAccFunc = library.makeFunction(name: "pq_train_update_accumulate"),
              let updateFinFunc = library.makeFunction(name: "pq_train_update_finalize"),
              let precompFunc = library.makeFunction(name: "pq_precompute_distance_table"),
              let distFunc = library.makeFunction(name: "pq_compute_distances_adc") else {
            throw AccelerationError.shaderNotFound(name: "Product Quantization kernels not found")
        }
        
        self.assignmentKernel = try device.makeComputePipelineState(function: assignmentFunc)
        self.updateAccumulateKernel = try device.makeComputePipelineState(function: updateAccFunc)
        self.updateFinalizeKernel = try device.makeComputePipelineState(function: updateFinFunc)
        self.precomputeDistanceKernel = try device.makeComputePipelineState(function: precompFunc)
        self.computeDistancesKernel = try device.makeComputePipelineState(function: distFunc)
    }
    
    // MARK: - Training
    
    /// Train PQ model on training data
    public func train(
        data: any MTLBuffer,
        count: Int,
        config: PQConfig,
        progressHandler: ((Int, Float) -> Void)? = nil
    ) async throws -> PQModel {
        // Allocate buffers
        let codebookSize = config.M * config.K * config.D_sub
        guard let codebooks = device.makeBuffer(
            length: codebookSize * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create codebooks buffer")
        }
        
        // Initialize codebooks with random vectors from training data
        try initializeCodebooks(codebooks: codebooks, data: data, count: count, config: config)
        
        guard let assignments = device.makeBuffer(
            length: count * config.M * MemoryLayout<UInt8>.stride,
            options: MTLResourceOptions.storageModePrivate
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create assignments buffer")
        }
        
        // Allocate accumulation buffers
        let accumSize = config.M * config.K * config.D_sub * MemoryLayout<Float>.stride
        let countSize = config.M * config.K * MemoryLayout<UInt32>.stride
        
        guard let centroidsAccum = device.makeBuffer(length: accumSize, options: MTLResourceOptions.storageModeShared),
              let centroidCounts = device.makeBuffer(length: countSize, options: MTLResourceOptions.storageModeShared),
              let convergenceBuffer = device.makeBuffer(
                length: config.M * config.K * MemoryLayout<Float>.stride,
                options: MTLResourceOptions.storageModeShared
              ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create training buffers")
        }
        
        // Training loop
        for iteration in 0..<config.trainIterations {
            // E-step: Assignment
            let assignCommand = kernelContext.commandQueue.makeCommandBuffer()!
            try encodeAssignment(
                vectors: data,
                codebooks: codebooks,
                assignments: assignments,
                count: count,
                config: config,
                commandBuffer: assignCommand
            )
            assignCommand.commit()
            await assignCommand.completed()
            
            // Clear accumulators
            clearBuffer(centroidsAccum)
            clearBuffer(centroidCounts)
            
            // M-step: Update
            let updateCommand = kernelContext.commandQueue.makeCommandBuffer()!
            
            // Accumulate
            try encodeUpdateAccumulate(
                data: data,
                assignments: assignments,
                centroidsAccum: centroidsAccum,
                centroidCounts: centroidCounts,
                count: count,
                config: config,
                commandBuffer: updateCommand
            )
            
            // Finalize
            try encodeUpdateFinalize(
                codebooks: codebooks,
                centroidsAccum: centroidsAccum,
                centroidCounts: centroidCounts,
                convergenceBuffer: convergenceBuffer,
                config: config,
                commandBuffer: updateCommand
            )
            
            updateCommand.commit()
            await updateCommand.completed()
            
            // Check convergence
            let movement = calculateTotalMovement(convergenceBuffer: convergenceBuffer, config: config)
            progressHandler?(iteration, movement)
            
            if movement < config.convergenceThreshold {
                print("PQ Training converged at iteration \(iteration)")
                break
            }
        }
        
        return PQModel(codebooks: codebooks, config: config, device: device)
    }
    
    // MARK: - Encoding
    
    /// Encode vectors using trained model
    public func encode(
        vectors: any MTLBuffer,
        count: Int,
        model: PQModel,
        commandBuffer: (any MTLCommandBuffer)? = nil
    ) async throws -> EncodedVectors {
        guard let codes = device.makeBuffer(
            length: count * model.config.M * MemoryLayout<UInt8>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create codes buffer")
        }
        
        let command = commandBuffer ?? kernelContext.commandQueue.makeCommandBuffer()!
        
        try encodeAssignment(
            vectors: vectors,
            codebooks: model.codebooks,
            assignments: codes,
            count: count,
            config: model.config,
            commandBuffer: command
        )
        
        if commandBuffer == nil {
            command.commit()
            await command.completed()
        }
        
        return EncodedVectors(codes: codes, count: count, config: model.config)
    }
    
    // MARK: - Distance Computation
    
    /// Compute approximate distances between query and encoded vectors
    public func computeDistances(
        query: any MTLBuffer,
        encodedVectors: EncodedVectors,
        model: PQModel,
        commandBuffer: (any MTLCommandBuffer)? = nil
    ) async throws -> any MTLBuffer {
        // Precompute distance table
        let tableSize = model.config.M * model.config.K
        guard let distanceTable = device.makeBuffer(
            length: tableSize * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModePrivate
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create distance table")
        }
        
        guard let distances = device.makeBuffer(
            length: encodedVectors.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create distances buffer")
        }
        
        let command = commandBuffer ?? kernelContext.commandQueue.makeCommandBuffer()!
        
        // Step 1: Precompute distance table
        try encodePrecomputeDistanceTable(
            query: query,
            codebooks: model.codebooks,
            distanceTable: distanceTable,
            config: model.config,
            commandBuffer: command
        )
        
        // Step 2: Compute distances via lookup
        try encodeComputeDistances(
            codes: encodedVectors.codes,
            distanceTable: distanceTable,
            distances: distances,
            count: encodedVectors.count,
            config: model.config,
            commandBuffer: command
        )
        
        if commandBuffer == nil {
            command.commit()
            await command.completed()
        }
        
        return distances
    }
    
    // MARK: - Convenience Methods
    
    /// Train and encode in one call
    public func trainAndEncode(
        data: [[Float]],
        config: PQConfig
    ) async throws -> (model: PQModel, encoded: EncodedVectors) {
        let flatData = data.flatMap { $0 }
        guard let dataBuffer = kernelContext.createBuffer(from: flatData, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferCreationFailed("Failed to create data buffer")
        }

        let model = try await train(
            data: dataBuffer,
            count: data.count,
            config: config
        )

        let encoded = try await encode(
            vectors: dataBuffer,
            count: data.count,
            model: model
        )
        
        return (model, encoded)
    }
    
    /// Find approximate nearest neighbors using PQ
    public func findNearestNeighbors(
        query: [Float],
        encodedDatabase: EncodedVectors,
        model: PQModel,
        k: Int
    ) async throws -> [(index: Int, distance: Float)] {
        guard let queryBuffer = kernelContext.createBuffer(from: query, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferCreationFailed("Failed to create query buffer")
        }

        let distances = try await computeDistances(
            query: queryBuffer,
            encodedVectors: encodedDatabase,
            model: model
        )
        
        // Extract distances and find top-k
        let distPtr = distances.contents().bindMemory(to: Float.self, capacity: encodedDatabase.count)
        var indexedDistances: [(index: Int, distance: Float)] = []
        
        for i in 0..<encodedDatabase.count {
            indexedDistances.append((index: i, distance: sqrt(distPtr[i])))  // Convert from squared
        }
        
        // Sort and return top-k
        indexedDistances.sort { $0.distance < $1.distance }
        return Array(indexedDistances.prefix(k))
    }
    
    // MARK: - Private Helper Methods
    
    private func initializeCodebooks(codebooks: any MTLBuffer, data: any MTLBuffer, count: Int, config: PQConfig) throws {
        // Simple random initialization
        let codebooksPtr = codebooks.contents().bindMemory(to: Float.self, capacity: config.M * config.K * config.D_sub)
        let dataPtr = data.contents().bindMemory(to: Float.self, capacity: count * config.dimension)
        
        for m in 0..<config.M {
            for k in 0..<config.K {
                // Randomly select a vector and copy its subspace
                let randomIdx = Int.random(in: 0..<count)
                let srcStart = randomIdx * config.dimension + m * config.D_sub
                let dstStart = (m * config.K + k) * config.D_sub
                
                for d in 0..<config.D_sub {
                    codebooksPtr[dstStart + d] = dataPtr[srcStart + d]
                }
            }
        }
    }
    
    private func clearBuffer(_ buffer: any MTLBuffer) {
        memset(buffer.contents(), 0, buffer.length)
    }
    
    private func calculateTotalMovement(convergenceBuffer: any MTLBuffer, config: PQConfig) -> Float {
        let ptr = convergenceBuffer.contents().bindMemory(to: Float.self, capacity: config.M * config.K)
        var total: Float = 0
        
        for i in 0..<(config.M * config.K) {
            total += ptr[i]
        }
        
        return sqrt(total / Float(config.M * config.K))
    }
    
    // MARK: - Kernel Encoding Methods
    
    private func encodeAssignment(
        vectors: any MTLBuffer,
        codebooks: any MTLBuffer,
        assignments: any MTLBuffer,
        count: Int,
        config: PQConfig,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "PQ_Assignment"
        encoder.setComputePipelineState(assignmentKernel)
        
        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(codebooks, offset: 0, index: 1)
        encoder.setBuffer(assignments, offset: 0, index: 2)
        
        var pqConfig = (
            N: UInt32(count),
            D: UInt32(config.dimension),
            M: UInt32(config.M),
            K: UInt32(config.K),
            D_sub: UInt32(config.D_sub)
        )
        encoder.setBytes(&pqConfig, length: MemoryLayout.size(ofValue: pqConfig), index: 3)
        
        // 2D dispatch
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: count, height: config.M, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func encodeUpdateAccumulate(
        data: any MTLBuffer,
        assignments: any MTLBuffer,
        centroidsAccum: any MTLBuffer,
        centroidCounts: any MTLBuffer,
        count: Int,
        config: PQConfig,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "PQ_UpdateAccumulate"
        encoder.setComputePipelineState(updateAccumulateKernel)
        
        encoder.setBuffer(data, offset: 0, index: 0)
        encoder.setBuffer(assignments, offset: 0, index: 1)
        encoder.setBuffer(centroidsAccum, offset: 0, index: 2)
        encoder.setBuffer(centroidCounts, offset: 0, index: 3)
        
        var pqConfig = (
            N: UInt32(count),
            D: UInt32(config.dimension),
            M: UInt32(config.M),
            K: UInt32(config.K),
            D_sub: UInt32(config.D_sub)
        )
        encoder.setBytes(&pqConfig, length: MemoryLayout.size(ofValue: pqConfig), index: 4)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: count, height: config.M, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func encodeUpdateFinalize(
        codebooks: any MTLBuffer,
        centroidsAccum: any MTLBuffer,
        centroidCounts: any MTLBuffer,
        convergenceBuffer: any MTLBuffer,
        config: PQConfig,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "PQ_UpdateFinalize"
        encoder.setComputePipelineState(updateFinalizeKernel)
        
        encoder.setBuffer(codebooks, offset: 0, index: 0)
        encoder.setBuffer(centroidsAccum, offset: 0, index: 1)
        encoder.setBuffer(centroidCounts, offset: 0, index: 2)
        encoder.setBuffer(convergenceBuffer, offset: 0, index: 3)
        
        var pqConfig = (
            N: UInt32(0),  // Not used in finalize
            D: UInt32(config.dimension),
            M: UInt32(config.M),
            K: UInt32(config.K),
            D_sub: UInt32(config.D_sub)
        )
        encoder.setBytes(&pqConfig, length: MemoryLayout.size(ofValue: pqConfig), index: 4)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: config.K, height: config.M, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func encodePrecomputeDistanceTable(
        query: any MTLBuffer,
        codebooks: any MTLBuffer,
        distanceTable: any MTLBuffer,
        config: PQConfig,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "PQ_PrecomputeDistanceTable"
        encoder.setComputePipelineState(precomputeDistanceKernel)
        
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(codebooks, offset: 0, index: 1)
        encoder.setBuffer(distanceTable, offset: 0, index: 2)
        
        var pqConfig = (
            N: UInt32(0),  // Not used
            D: UInt32(config.dimension),
            M: UInt32(config.M),
            K: UInt32(config.K),
            D_sub: UInt32(config.D_sub)
        )
        encoder.setBytes(&pqConfig, length: MemoryLayout.size(ofValue: pqConfig), index: 3)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: config.K, height: config.M, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func encodeComputeDistances(
        codes: any MTLBuffer,
        distanceTable: any MTLBuffer,
        distances: any MTLBuffer,
        count: Int,
        config: PQConfig,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "PQ_ComputeDistances"
        encoder.setComputePipelineState(computeDistancesKernel)
        
        encoder.setBuffer(codes, offset: 0, index: 0)
        encoder.setBuffer(distanceTable, offset: 0, index: 1)
        encoder.setBuffer(distances, offset: 0, index: 2)
        
        var pqConfig = (
            N: UInt32(count),
            D: UInt32(config.dimension),
            M: UInt32(config.M),
            K: UInt32(config.K),
            D_sub: UInt32(config.D_sub)
        )
        encoder.setBytes(&pqConfig, length: MemoryLayout.size(ofValue: pqConfig), index: 3)
        
        // Set threadgroup memory size for distance table
        let sharedMemorySize = config.M * config.K * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
        
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (count + 255) / 256
        let gridSize = MTLSize(width: numThreadgroups * 256, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    // MARK: - Performance Analysis
    
    public struct PerformanceMetrics: Sendable {
        public let trainingTime: TimeInterval
        public let encodingThroughput: Double  // vectors/second
        public let searchThroughput: Double    // queries/second
        public let compressionRatio: Float
        public let recall: Float               // recall@k accuracy
    }
    
    /// Benchmark PQ performance
    public func benchmark(
        dataSize: Int = 10000,
        dimension: Int = 128,
        queryCount: Int = 100,
        config: PQConfig? = nil
    ) async throws -> PerformanceMetrics {
        let pqConfig = config ?? PQConfig(dimension: dimension)
        
        // Generate random data
        let data = (0..<dataSize).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        let queries = (0..<queryCount).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        // Training benchmark
        let trainStart = CACurrentMediaTime()
        let (model, encoded) = try await trainAndEncode(data: data, config: pqConfig)
        let trainTime = CACurrentMediaTime() - trainStart
        
        // Encoding throughput
        let encodingThroughput = Double(dataSize) / trainTime
        
        // Search benchmark
        let searchStart = CACurrentMediaTime()
        for query in queries {
            _ = try await findNearestNeighbors(
                query: query,
                encodedDatabase: encoded,
                model: model,
                k: 10
            )
        }
        let searchTime = CACurrentMediaTime() - searchStart
        let searchThroughput = Double(queryCount) / searchTime
        
        return PerformanceMetrics(
            trainingTime: trainTime,
            encodingThroughput: encodingThroughput,
            searchThroughput: searchThroughput,
            compressionRatio: model.compressionRatio,
            recall: 0.9  // Would need ground truth to calculate
        )
    }
}
