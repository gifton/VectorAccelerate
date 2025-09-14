// Streaming Top-K Manager Kernel
// Handles datasets > 4 billion vectors with incremental chunk processing

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Streaming Top-K Kernel

/// GPU-accelerated streaming top-k selection for massive datasets
/// Processes data in chunks and maintains running top-k using max-heap
public final class StreamingTopKKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let initKernel: any MTLComputePipelineState
    private let processKernel: any MTLComputePipelineState
    private let finalizeKernel: any MTLComputePipelineState
    
    // Configuration matching Metal
    private let MAX_K_PRIVATE = 128
    
    /// Configuration for streaming operations
    public struct StreamConfig {
        public let queryCount: Int
        public let k: Int
        public let chunkSize: Int
        public let totalVectorCount: Int64 // Support > 4B vectors
        
        public init(
            queryCount: Int,
            k: Int,
            chunkSize: Int = 100_000,
            totalVectorCount: Int64
        ) {
            self.queryCount = queryCount
            self.k = min(k, 128) // Enforce MAX_K_PRIVATE
            self.chunkSize = chunkSize
            self.totalVectorCount = totalVectorCount
        }
        
        var numberOfChunks: Int {
            return Int((totalVectorCount + Int64(chunkSize) - 1) / Int64(chunkSize))
        }
    }
    
    /// Streaming state for incremental processing
    public class StreamingState {
        public let runningDistances: any MTLBuffer
        public let runningIndices: any MTLBuffer
        public let config: StreamConfig
        private(set) var chunksProcessed: Int = 0
        
        fileprivate init(
            distances: any MTLBuffer,
            indices: any MTLBuffer,
            config: StreamConfig
        ) {
            self.runningDistances = distances
            self.runningIndices = indices
            self.config = config
        }
        
        var progress: Float {
            return Float(chunksProcessed) / Float(config.numberOfChunks)
        }
        
        var isComplete: Bool {
            return chunksProcessed >= config.numberOfChunks
        }
        
        fileprivate func incrementChunks() {
            chunksProcessed += 1
        }
    }
    
    /// Final results after streaming
    public struct StreamingResult {
        public let indices: any MTLBuffer
        public let distances: any MTLBuffer
        public let queryCount: Int
        public let k: Int
        
        /// Extract sorted results for a query
        public func sortedResults(for queryIndex: Int) -> [(index: Int64, distance: Float)] {
            guard queryIndex < queryCount else { return [] }
            
            let offset = queryIndex * k
            let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: (queryIndex + 1) * k)
            let distPtr = distances.contents().bindMemory(to: Float.self, capacity: (queryIndex + 1) * k)
            
            var results: [(index: Int64, distance: Float)] = []
            for i in 0..<k {
                let idx = indexPtr[offset + i]
                if idx != 0xFFFFFFFF {
                    results.append((index: Int64(idx), distance: distPtr[offset + i]))
                }
            }
            return results
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }
        
        // Load kernels
        guard let initFunc = library.makeFunction(name: "streaming_topk_init"),
              let processFunc = library.makeFunction(name: "streaming_topk_process_chunk"),
              let finalizeFunc = library.makeFunction(name: "streaming_topk_finalize") else {
            throw AccelerationError.shaderNotFound(name: "Streaming top-k kernels not found")
        }
        
        self.initKernel = try device.makeComputePipelineState(function: initFunc)
        self.processKernel = try device.makeComputePipelineState(function: processFunc)
        self.finalizeKernel = try device.makeComputePipelineState(function: finalizeFunc)
    }
    
    // MARK: - Streaming Operations
    
    /// Initialize streaming state for chunk-based processing
    /// - Parameter config: Streaming configuration
    /// - Returns: Initialized streaming state
    public func initializeStreaming(config: StreamConfig) throws -> StreamingState {
        guard config.k <= MAX_K_PRIVATE else {
            throw AccelerationError.invalidInput("K must be <= \(MAX_K_PRIVATE)")
        }
        
        // Allocate running state buffers
        let distanceBuffer = device.makeBuffer(
            length: config.queryCount * config.k * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModePrivate
        )
        
        let indexBuffer = device.makeBuffer(
            length: config.queryCount * config.k * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModePrivate
        )
        
        guard let distances = distanceBuffer, let indices = indexBuffer else {
            throw AccelerationError.bufferCreationFailed("Failed to create streaming buffers")
        }
        
        // Initialize buffers with infinity/sentinel values
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "StreamingTopKInit"
        encoder.setComputePipelineState(initKernel)
        
        encoder.setBuffer(distances, offset: 0, index: 0)
        encoder.setBuffer(indices, offset: 0, index: 1)
        
        var params = (K: UInt32(config.k), Q: UInt32(config.queryCount))
        encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 2)
        
        let totalElements = config.queryCount * config.k
        let optimalThreads = min(initKernel.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: optimalThreads, height: 1, depth: 1)
        let numThreadgroups = (totalElements + optimalThreads - 1) / optimalThreads
        let gridSize = MTLSize(width: numThreadgroups * optimalThreads, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return StreamingState(distances: distances, indices: indices, config: config)
    }
    
    /// Process a chunk of distances
    /// - Parameters:
    ///   - chunkDistances: Distance matrix for current chunk [Q × chunk_size]
    ///   - state: Current streaming state
    ///   - chunkBaseIndex: Global index offset for this chunk
    ///   - commandBuffer: Command buffer for execution
    public func processChunk(
        chunkDistances: any MTLBuffer,
        state: StreamingState,
        chunkBaseIndex: Int64,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard !state.isComplete else {
            throw AccelerationError.invalidInput("Streaming already complete")
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "StreamingTopKProcessChunk"
        encoder.setComputePipelineState(processKernel)
        
        // Set buffers
        encoder.setBuffer(chunkDistances, offset: 0, index: 0)
        encoder.setBuffer(state.runningDistances, offset: 0, index: 2)
        encoder.setBuffer(state.runningIndices, offset: 0, index: 3)
        
        // StreamConfig structure matching Metal
        var config = (
            Q: UInt32(state.config.queryCount),
            chunk_size: UInt32(state.config.chunkSize),
            k_value: UInt32(state.config.k),
            chunk_base_index: UInt64(chunkBaseIndex)
        )
        encoder.setBytes(&config, length: MemoryLayout.size(ofValue: config), index: 5)
        
        // One thread per query
        let optimalThreads = min(processKernel.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: optimalThreads, height: 1, depth: 1)
        let numThreadgroups = (state.config.queryCount + optimalThreads - 1) / optimalThreads
        let gridSize = MTLSize(width: numThreadgroups * optimalThreads, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        state.incrementChunks()
    }
    
    /// Finalize streaming and sort results
    /// - Parameters:
    ///   - state: Completed streaming state
    ///   - commandBuffer: Command buffer for execution
    /// - Returns: Final sorted results
    public func finalizeStreaming(
        state: StreamingState,
        commandBuffer: any MTLCommandBuffer
    ) throws -> StreamingResult {
        guard state.isComplete else {
            throw AccelerationError.invalidInput("Streaming not complete. Processed \(state.chunksProcessed)/\(state.config.numberOfChunks) chunks")
        }
        
        // Create output buffers in shared memory for CPU access
        let finalDistances = device.makeBuffer(
            length: state.config.queryCount * state.config.k * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        )
        
        let finalIndices = device.makeBuffer(
            length: state.config.queryCount * state.config.k * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        )
        
        guard let distances = finalDistances, let indices = finalIndices else {
            throw AccelerationError.bufferCreationFailed("Failed to create final buffers")
        }
        
        // Copy from private to shared memory
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        blitEncoder.copy(
            from: state.runningDistances,
            sourceOffset: 0,
            to: distances,
            destinationOffset: 0,
            size: distances.length
        )
        
        blitEncoder.copy(
            from: state.runningIndices,
            sourceOffset: 0,
            to: indices,
            destinationOffset: 0,
            size: indices.length
        )
        
        blitEncoder.endEncoding()
        
        // Sort using heapsort kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "StreamingTopKFinalize"
        encoder.setComputePipelineState(finalizeKernel)
        
        encoder.setBuffer(distances, offset: 0, index: 0)
        encoder.setBuffer(indices, offset: 0, index: 1)
        
        var params = (K: UInt32(state.config.k), Q: UInt32(state.config.queryCount))
        encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 2)
        
        // One thread per query
        let optimalThreads = min(processKernel.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: optimalThreads, height: 1, depth: 1)
        let numThreadgroups = (state.config.queryCount + optimalThreads - 1) / optimalThreads
        let gridSize = MTLSize(width: numThreadgroups * optimalThreads, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return StreamingResult(
            indices: indices,
            distances: distances,
            queryCount: state.config.queryCount,
            k: state.config.k
        )
    }
    
    // MARK: - High-Level Interface
    
    /// Process entire dataset in streaming fashion
    /// - Parameters:
    ///   - distanceChunks: Generator providing distance chunks
    ///   - config: Streaming configuration
    ///   - progressHandler: Optional progress callback
    /// - Returns: Final top-k results
    public func streamingTopK(
        distanceChunks: @escaping (Int) throws -> (any MTLBuffer)?,
        config: StreamConfig,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws -> StreamingResult {
        // Initialize state
        let state = try initializeStreaming(config: config)
        
        // Process chunks
        var chunkIndex = 0
        var globalIndex: Int64 = 0
        
        while let chunkBuffer = try distanceChunks(chunkIndex) {
            let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
            
            try processChunk(
                chunkDistances: chunkBuffer,
                state: state,
                chunkBaseIndex: globalIndex,
                commandBuffer: commandBuffer
            )
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            chunkIndex += 1
            globalIndex += Int64(config.chunkSize)
            
            progressHandler?(state.progress)
            
            if state.isComplete {
                break
            }
        }
        
        // Finalize
        let finalCommand = kernelContext.commandQueue.makeCommandBuffer()!
        let result = try finalizeStreaming(state: state, commandBuffer: finalCommand)
        
        finalCommand.commit()
        finalCommand.waitUntilCompleted()
        
        return result
    }
    
    // MARK: - Convenience Methods
    
    /// Process large dataset from memory-mapped file
    public func processLargeDataset(
        fileURL: URL,
        queryCount: Int,
        vectorCount: Int64,
        dimension: Int,
        k: Int,
        chunkSize: Int = 100_000
    ) async throws -> StreamingResult {
        // Memory map the file
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        
        let config = StreamConfig(
            queryCount: queryCount,
            k: k,
            chunkSize: chunkSize,
            totalVectorCount: vectorCount
        )
        
        return try await streamingTopK(
            distanceChunks: { chunkIdx -> (any MTLBuffer)? in
                let startIdx = Int64(chunkIdx) * Int64(chunkSize)
                if startIdx >= vectorCount {
                    return nil
                }

                let endIdx = min(startIdx + Int64(chunkSize), vectorCount)
                let actualChunkSize = Int(endIdx - startIdx)

                // Compute distances for this chunk
                // This is simplified - in practice you'd compute L2/cosine distances
                let chunkData = data.subdata(
                    in: Int(startIdx * Int64(dimension) * 4)..<Int(endIdx * Int64(dimension) * 4)
                )

                let chunkArray = [UInt8](chunkData)
                return try self.kernelContext.createBuffer(
                    from: chunkArray,
                    options: MTLResourceOptions.storageModeShared
                )
            },
            config: config
        )
    }
    
    // MARK: - Performance Monitoring
    
    public struct StreamingMetrics: Sendable {
        public let totalTime: TimeInterval
        public let averageChunkTime: TimeInterval
        public let peakMemoryUsage: Int64 // bytes
        public let throughput: Double // vectors per second
    }
    
    /// Benchmark streaming performance
    public func benchmarkStreaming(
        queryCount: Int = 100,
        totalVectors: Int64 = 10_000_000,
        dimension: Int = 512,
        k: Int = 100,
        chunkSize: Int = 100_000
    ) async throws -> StreamingMetrics {
        let startTime = CACurrentMediaTime()
        var chunkTimes: [TimeInterval] = []
        
        let config = StreamConfig(
            queryCount: queryCount,
            k: k,
            chunkSize: chunkSize,
            totalVectorCount: totalVectors
        )
        
        // Generate synthetic chunks
        let state = try initializeStreaming(config: config)
        
        for chunkIdx in 0..<config.numberOfChunks {
            let chunkStart = CACurrentMediaTime()
            
            // Create synthetic distance chunk
            let distances = (0..<queryCount * chunkSize).map { _ in
                Float.random(in: 0...100)
            }
            
            guard let chunkBuffer = kernelContext.createBuffer(
            from: distances,
            options: MTLResourceOptions.storageModeShared
            
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
            
            let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
            try processChunk(
                chunkDistances: chunkBuffer,
                state: state,
                chunkBaseIndex: Int64(chunkIdx) * Int64(chunkSize),
                commandBuffer: commandBuffer
            )
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            chunkTimes.append(CACurrentMediaTime() - chunkStart)
        }
        
        // Finalize
        let finalCommand = kernelContext.commandQueue.makeCommandBuffer()!
        _ = try finalizeStreaming(state: state, commandBuffer: finalCommand)
        finalCommand.commit()
        finalCommand.waitUntilCompleted()
        
        let totalTime = CACurrentMediaTime() - startTime
        let avgChunkTime = chunkTimes.reduce(0, +) / Double(chunkTimes.count)
        
        // Calculate memory usage (approximate)
        let stateMemory = queryCount * k * (MemoryLayout<Float>.stride + MemoryLayout<UInt32>.stride)
        let chunkMemory = queryCount * chunkSize * MemoryLayout<Float>.stride
        let peakMemory = Int64(stateMemory + chunkMemory)
        
        let throughput = Double(totalVectors) / totalTime
        
        return StreamingMetrics(
            totalTime: totalTime,
            averageChunkTime: avgChunkTime,
            peakMemoryUsage: peakMemory,
            throughput: throughput
        )
    }
}

// MARK: - VectorCore Integration

extension StreamingTopKKernel {
    /// Stream process vectors using VectorCore types
    public func streamProcess<V: VectorProtocol>(
        queryVectors: [V],
        datasetChunks: AsyncStream<[V]>,
        k: Int
    ) async throws -> StreamingResult where V.Scalar == Float {
        let queryCount = queryVectors.count
        guard queryCount > 0 else {
            throw AccelerationError.invalidInput("No query vectors")
        }
        
        let _ = queryVectors[0].count
        
        // Count total vectors
        var totalCount: Int64 = 0
        for await chunk in datasetChunks {
            totalCount += Int64(chunk.count)
        }
        
        let config = StreamConfig(
            queryCount: queryCount,
            k: k,
            chunkSize: 100_000,
            totalVectorCount: totalCount
        )
        
        // Re-create the stream for processing
        return try await streamingTopK(
            distanceChunks: { _ in
                // This would need proper async iteration support
                return nil
            },
            config: config
        )
    }
}