// Fused L2 Distance + Top-K Selection Kernel
// Single-pass GPU kernel that computes L2 distances and selects top-k in one operation

import Foundation
import Metal
import QuartzCore
import VectorCore

// MARK: - Fused L2 Top-K Kernel

/// GPU-accelerated kernel that fuses L2 distance computation with top-k selection
/// Achieves significant performance improvement by avoiding intermediate distance matrix
public final class FusedL2TopKKernel {
    private let device: any MTLDevice
    private let kernelContext: KernelContext
    private let fusedKernel: any MTLComputePipelineState
    private let streamingUpdateKernel: any MTLComputePipelineState
    
    // Configuration constants matching Metal
    private let K_PRIVATE = 8
    private let MAX_TGS = 256
    private let MAX_D = 512
    
    /// Result from fused L2 top-k operation
    public struct FusedResult {
        public let indices: any MTLBuffer
        public let distances: (any MTLBuffer)?
        public let queryCount: Int
        public let k: Int
        
        /// Extract results for a specific query
        public func results(for queryIndex: Int) -> [(index: Int, distance: Float)] {
            guard queryIndex < queryCount else { return [] }
            
            let offset = queryIndex * k
            let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: (queryIndex + 1) * k)
            let distPtr = distances?.contents().bindMemory(to: Float.self, capacity: (queryIndex + 1) * k)
            
            var results: [(index: Int, distance: Float)] = []
            for i in 0..<k {
                let idx = Int(indexPtr[offset + i])
                if idx != 0xFFFFFFFF { // Skip sentinel values
                    let dist = distPtr?[offset + i] ?? 0
                    results.append((index: idx, distance: dist))
                }
            }
            return results
        }
    }
    
    /// Configuration for fused operation
    public struct FusedConfig {
        public let includeDistances: Bool
        public let threadgroupSize: Int
        
        public init(includeDistances: Bool = true, threadgroupSize: Int = 256) {
            self.includeDistances = includeDistances
            self.threadgroupSize = min(threadgroupSize, 256) // Enforce MAX_TGS
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load fused kernel
        guard let fusedFunction = library.makeFunction(name: "fused_l2_topk") else {
            throw VectorError.shaderNotFound(name: "fused_l2_topk")
        }
        
        // Load streaming update kernel
        guard let streamingFunction = library.makeFunction(name: "streaming_l2_topk_update") else {
            throw VectorError.shaderNotFound(name: "streaming_l2_topk_update")
        }
        
        self.fusedKernel = try device.makeComputePipelineState(function: fusedFunction)
        self.streamingUpdateKernel = try device.makeComputePipelineState(function: streamingFunction)
    }
    
    // MARK: - Compute Methods
    
    /// Perform fused L2 distance computation and top-k selection
    /// - Parameters:
    ///   - queries: Query vectors [Q × D]
    ///   - dataset: Dataset vectors [N × D]
    ///   - k: Number of nearest neighbors to find
    ///   - config: Operation configuration
    ///   - commandBuffer: Command buffer for execution
    /// - Returns: Fused result with indices and optional distances
    public func fusedL2TopK(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        k: Int,
        config: FusedConfig = FusedConfig(),
        commandBuffer: any MTLCommandBuffer
    ) throws -> FusedResult {
        // Validation
        guard dimension <= MAX_D else {
            throw VectorError.invalidInput("Dimension \(dimension) exceeds maximum \(MAX_D)")
        }
        
        guard k > 0 && k <= datasetCount else {
            throw VectorError.invalidInput("K must be between 1 and \(datasetCount)")
        }
        
        // Allocate output buffers
        let indicesBuffer = device.makeBuffer(
            length: queryCount * k * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        )
        
        let distancesBuffer = config.includeDistances ? device.makeBuffer(
            length: queryCount * k * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) : nil
        
        guard let indices = indicesBuffer else {
            throw VectorError.bufferCreationFailed("Failed to create indices buffer")
        }
        
        // Encode kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        encoder.label = "FusedL2TopK"
        encoder.setComputePipelineState(fusedKernel)

        // Set buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(indices, offset: 0, index: 2)
        encoder.setBuffer(distancesBuffer, offset: 0, index: 3)

        // Set parameters - each as individual buffer to match Metal shader signature
        // Metal shader expects: constant uint& Q [[buffer(4)]], N [[buffer(5)]], D [[buffer(6)]], K [[buffer(7)]]
        var paramQ = UInt32(queryCount)
        var paramN = UInt32(datasetCount)
        var paramD = UInt32(dimension)
        var paramK = UInt32(k)
        encoder.setBytes(&paramQ, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&paramN, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&paramD, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&paramK, length: MemoryLayout<UInt32>.size, index: 7)
        
        // Dispatch with 1 threadgroup per query
        let threadsPerThreadgroup = MTLSize(width: config.threadgroupSize, height: 1, depth: 1)
        let threadgroups = MTLSize(width: queryCount, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return FusedResult(
            indices: indices,
            distances: distancesBuffer,
            queryCount: queryCount,
            k: k
        )
    }
    
    /// Process dataset in chunks for very large datasets
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - dataset: Dataset vectors
    ///   - chunkSize: Number of vectors to process per chunk
    ///   - Other parameters same as fusedL2TopK
    /// - Returns: Final top-k results after processing all chunks
    public func chunkedFusedL2TopK(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        k: Int,
        chunkSize: Int = 100_000,
        config: FusedConfig = FusedConfig()
    ) async throws -> FusedResult {
        // Initialize with first chunk
        let firstChunkSize = min(chunkSize, datasetCount)
        
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        let result = try fusedL2TopK(
            queries: queries,
            dataset: dataset,
            queryCount: queryCount,
            datasetCount: firstChunkSize,
            dimension: dimension,
            k: k,
            config: config,
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        // Process remaining chunks
        var processedCount = firstChunkSize
        while processedCount < datasetCount {
            let remainingCount = datasetCount - processedCount
            let currentChunkSize = min(chunkSize, remainingCount)
            
            // Create buffer for chunk of dataset
            let offset = processedCount * dimension * MemoryLayout<Float>.stride
            let length = currentChunkSize * dimension * MemoryLayout<Float>.stride
            let pointer = dataset.contents().advanced(by: offset)
            guard let chunkBuffer = device.makeBuffer(
                bytes: pointer,
                length: length,
                options: MTLResourceOptions.storageModeShared
            ) else {
                throw VectorError.bufferCreationFailed("Failed to create chunk buffer")
            }
            
            let updateCommand = kernelContext.commandQueue.makeCommandBuffer()!
            try updateWithChunk(
                queries: queries,
                chunk: chunkBuffer,
                currentTopK: result,
                queryCount: queryCount,
                chunkSize: currentChunkSize,
                dimension: dimension,
                k: k,
                chunkOffset: UInt32(processedCount),
                commandBuffer: updateCommand
            )
            
            updateCommand.commit()
            await updateCommand.completed()
            
            processedCount += currentChunkSize
        }
        
        return result
    }
    
    // MARK: - Convenience Methods
    
    /// Find k nearest neighbors for query vectors
    public func findNearestNeighbors(
        queries: [[Float]],
        dataset: [[Float]],
        k: Int,
        includeDistances: Bool = true
    ) async throws -> [[(index: Int, distance: Float)]] {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input arrays")
        }
        
        let dimension = queries[0].count
        guard dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.countMismatch()
        }
        
        // Create buffers
        guard let queryBuffer = kernelContext.createBuffer(
            from: queries.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create buffer")
        }
        
        guard let datasetBuffer = kernelContext.createBuffer(
            from: dataset.flatMap { $0 },
            options: MTLResourceOptions.storageModeShared
        
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create buffer")
        }
        
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!
        
        let result = try fusedL2TopK(
            queries: queryBuffer,
            dataset: datasetBuffer,
            queryCount: queries.count,
            datasetCount: dataset.count,
            dimension: dimension,
            k: k,
            config: FusedConfig(includeDistances: includeDistances),
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        // Extract results
        var allResults: [[(index: Int, distance: Float)]] = []
        for q in 0..<queries.count {
            allResults.append(result.results(for: q))
        }
        
        return allResults
    }
    
    /// Search with VectorCore types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func search<Q: VectorProtocol, D: VectorProtocol>(
        queries: [Q],
        dataset: [D],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] where Q.Scalar == Float, D.Scalar == Float {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw VectorError.invalidInput("Empty input vectors")
        }

        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              dataset.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.invalidInput("Dimension mismatch in input vectors")
        }

        // Zero-copy buffer creation using withUnsafeBufferPointer
        guard let queryBuffer = kernelContext.createAlignedBufferFromVectors(
            queries,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create query buffer")
        }

        guard let datasetBuffer = kernelContext.createAlignedBufferFromVectors(
            dataset,
            options: .storageModeShared,
            alignment: 16
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create dataset buffer")
        }

        // Use the fused kernel with the buffers
        let commandBuffer = kernelContext.commandQueue.makeCommandBuffer()!

        let result = try fusedL2TopK(
            queries: queryBuffer,
            dataset: datasetBuffer,
            queryCount: queries.count,
            datasetCount: dataset.count,
            dimension: dimension,
            k: k,
            config: FusedConfig(includeDistances: true),
            commandBuffer: commandBuffer
        )

        commandBuffer.commit()
        await commandBuffer.completed()

        // Extract results
        var allResults: [[(index: Int, distance: Float)]] = []
        for q in 0..<queries.count {
            allResults.append(result.results(for: q))
        }

        return allResults
    }
    
    // MARK: - Private Methods
    
    /// Parameters struct matching Metal's StreamingL2Params layout
    private struct StreamingL2Params {
        var Q: UInt32
        var chunkSize: UInt32
        var D: UInt32
        var K: UInt32
        var offset: UInt32
    }

    private func updateWithChunk(
        queries: any MTLBuffer,
        chunk: any MTLBuffer,
        currentTopK: FusedResult,
        queryCount: Int,
        chunkSize: Int,
        dimension: Int,
        k: Int,
        chunkOffset: UInt32,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.encoderCreationFailed()
        }

        encoder.label = "StreamingL2TopKUpdate"
        encoder.setComputePipelineState(streamingUpdateKernel)

        // Set buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(chunk, offset: 0, index: 1)
        encoder.setBuffer(currentTopK.indices, offset: 0, index: 2)
        encoder.setBuffer(currentTopK.distances, offset: 0, index: 3)

        // Set parameters - struct matches Metal's StreamingL2Params
        var params = StreamingL2Params(
            Q: UInt32(queryCount),
            chunkSize: UInt32(chunkSize),
            D: UInt32(dimension),
            K: UInt32(k),
            offset: chunkOffset
        )
        encoder.setBytes(&params, length: MemoryLayout<StreamingL2Params>.size, index: 4)

        // Dispatch
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: queryCount, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
    
    // MARK: - Performance Analysis
    
    public struct PerformanceMetrics: Sendable {
        public let totalTime: TimeInterval
        public let distanceComputeTime: TimeInterval
        public let topKSelectionTime: TimeInterval
        public let throughput: Double // queries per second
        public let effectiveMemoryBandwidth: Double // GB/s
    }
    
    /// Benchmark the fused kernel performance
    public func benchmark(
        queryCount: Int = 100,
        datasetSize: Int = 1_000_000,
        dimension: Int = 512,
        k: Int = 100,
        iterations: Int = 10
    ) async throws -> PerformanceMetrics {
        // Generate random test data
        let queries = (0..<queryCount).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        let dataset = (0..<min(datasetSize, 10000)).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        var times: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            _ = try await findNearestNeighbors(
                queries: queries,
                dataset: dataset,
                k: k
            )
            times.append(CACurrentMediaTime() - start)
        }
        
        // Calculate metrics
        let avgTime = times.reduce(0, +) / Double(times.count)
        let throughput = Double(queryCount) / avgTime
        
        // Estimate memory bandwidth (simplified)
        let bytesRead = Double(queryCount * dimension + datasetSize * dimension) * 4
        let bytesWritten = Double(queryCount * k) * 8 // indices + distances
        let bandwidth = (bytesRead + bytesWritten) / avgTime / 1e9
        
        return PerformanceMetrics(
            totalTime: avgTime,
            distanceComputeTime: avgTime * 0.7, // Estimate
            topKSelectionTime: avgTime * 0.3, // Estimate
            throughput: throughput,
            effectiveMemoryBandwidth: bandwidth
        )
    }
}

// MARK: - Extensions

extension FusedL2TopKKernel {
    /// Create a sub-buffer view for chunked processing
    fileprivate func createSubBuffer(
        from buffer: any MTLBuffer,
        offset: Int,
        length: Int
    ) throws -> any MTLBuffer {
        // Create a new buffer with data from the specified offset
        let pointer = buffer.contents().advanced(by: offset)
        guard let subBuffer = device.makeBuffer(
            bytes: pointer,
            length: length,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw VectorError.bufferCreationFailed("Failed to create sub-buffer")
        }
        return subBuffer
    }
}
