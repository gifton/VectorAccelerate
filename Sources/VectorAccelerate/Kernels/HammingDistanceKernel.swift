// Hamming Distance Kernel
// GPU-accelerated Hamming distance computation for binary vectors

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Hamming Distance Kernel

/// GPU-accelerated Hamming distance computation
/// Supports bit-packed binary vectors and float vectors with binarization
public final class HammingDistanceKernel {
    private let device: any MTLDevice
    private let computeEngine: ComputeEngine
    private let batchKernel: any MTLComputePipelineState
    private let floatKernel: any MTLComputePipelineState
    private let singleKernel: any MTLComputePipelineState
    private let normalizedKernel: any MTLComputePipelineState
    
    /// Result from Hamming distance computation
    public struct HammingResult {
        public let distances: any MTLBuffer
        public let queryCount: Int
        public let datasetCount: Int
        public let normalized: Bool
        
        /// Extract distance matrix as 2D array
        public func asMatrix() -> [[Float]] {
            if normalized {
                let ptr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * datasetCount)
                var matrix: [[Float]] = []
                for q in 0..<queryCount {
                    var row: [Float] = []
                    for n in 0..<datasetCount {
                        row.append(ptr[q * datasetCount + n])
                    }
                    matrix.append(row)
                }
                return matrix
            } else {
                let ptr = distances.contents().bindMemory(to: UInt32.self, capacity: queryCount * datasetCount)
                var matrix: [[Float]] = []
                for q in 0..<queryCount {
                    var row: [Float] = []
                    for n in 0..<datasetCount {
                        row.append(Float(ptr[q * datasetCount + n]))
                    }
                    matrix.append(row)
                }
                return matrix
            }
        }
        
        /// Get distance for specific query-dataset pair
        public func distance(query: Int, dataset: Int) -> Float {
            guard query < queryCount && dataset < datasetCount else { return Float.infinity }
            
            if normalized {
                let ptr = distances.contents().bindMemory(to: Float.self, capacity: queryCount * datasetCount)
                return ptr[query * datasetCount + dataset]
            } else {
                let ptr = distances.contents().bindMemory(to: UInt32.self, capacity: queryCount * datasetCount)
                return Float(ptr[query * datasetCount + dataset])
            }
        }
    }
    
    /// Configuration for Hamming distance computation
    public struct HammingConfig {
        public let normalized: Bool
        public let binarizationThreshold: Float
        
        public init(normalized: Bool = false, binarizationThreshold: Float = 0.0) {
            self.normalized = normalized
            self.binarizationThreshold = binarizationThreshold
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        self.computeEngine = try ComputeEngine(context: MetalContext(device: device))
        
        guard let library = device.makeDefaultLibrary() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create Metal library")
        }
        
        // Load kernels
        guard let batchFunc = library.makeFunction(name: "hamming_distance_batch"),
              let floatFunc = library.makeFunction(name: "hamming_distance_float"),
              let singleFunc = library.makeFunction(name: "hamming_distance_single"),
              let normalizedFunc = library.makeFunction(name: "hamming_distance_normalized") else {
            throw AccelerationError.shaderNotFound(name: "Hamming distance kernels not found")
        }
        
        self.batchKernel = try device.makeComputePipelineState(function: batchFunc)
        self.floatKernel = try device.makeComputePipelineState(function: floatFunc)
        self.singleKernel = try device.makeComputePipelineState(function: singleFunc)
        self.normalizedKernel = try device.makeComputePipelineState(function: normalizedFunc)
    }
    
    // MARK: - Compute Methods
    
    /// Compute Hamming distance matrix for bit-packed binary vectors
    /// - Parameters:
    ///   - queries: Query vectors (bit-packed)
    ///   - dataset: Dataset vectors (bit-packed)
    ///   - queryCount: Number of query vectors
    ///   - datasetCount: Number of dataset vectors
    ///   - bitsPerVector: Number of bits per vector
    ///   - config: Configuration options
    ///   - commandBuffer: Command buffer for execution
    /// - Returns: Hamming distance result
    public func computeBitPacked(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        bitsPerVector: Int,
        config: HammingConfig = HammingConfig(),
        commandBuffer: any MTLCommandBuffer
    ) throws -> HammingResult {
        let wordsPerVector = (bitsPerVector + 31) / 32
        
        // Allocate output buffer
        let outputBuffer: any MTLBuffer
        if config.normalized {
            guard let buffer = device.makeBuffer(
                length: queryCount * datasetCount * MemoryLayout<Float>.stride,
                options: MTLResourceOptions.storageModeShared
            ) else {
                throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
            }
            outputBuffer = buffer
        } else {
            guard let buffer = device.makeBuffer(
                length: queryCount * datasetCount * MemoryLayout<UInt32>.stride,
                options: MTLResourceOptions.storageModeShared
            ) else {
                throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
            }
            outputBuffer = buffer
        }
        
        // Encode kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        let kernel = config.normalized ? normalizedKernel : batchKernel
        encoder.label = "HammingDistance_BitPacked"
        encoder.setComputePipelineState(kernel)
        
        // Set buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        // Set parameters
        var params = (
            Q: UInt32(queryCount),
            N: UInt32(datasetCount),
            D_words: UInt32(wordsPerVector)
        )
        encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 3)
        
        if config.normalized {
            var D_bits = UInt32(bitsPerVector)
            encoder.setBytes(&D_bits, length: MemoryLayout<UInt32>.stride, index: 6)
        }
        
        // Dispatch with 16x16 threadgroups
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: datasetCount,
            height: queryCount,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return HammingResult(
            distances: outputBuffer,
            queryCount: queryCount,
            datasetCount: datasetCount,
            normalized: config.normalized
        )
    }
    
    /// Compute Hamming distance for float vectors with binarization
    /// - Parameters:
    ///   - queries: Query float vectors
    ///   - dataset: Dataset float vectors
    ///   - queryCount: Number of queries
    ///   - datasetCount: Number of dataset vectors
    ///   - dimension: Vector dimension
    ///   - config: Configuration with threshold
    ///   - commandBuffer: Command buffer
    /// - Returns: Hamming distance result
    public func computeFloat(
        queries: any MTLBuffer,
        dataset: any MTLBuffer,
        queryCount: Int,
        datasetCount: Int,
        dimension: Int,
        config: HammingConfig = HammingConfig(),
        commandBuffer: any MTLCommandBuffer
    ) throws -> HammingResult {
        // Allocate output buffer
        guard let outputBuffer = device.makeBuffer(
            length: queryCount * datasetCount * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
        }
        
        // Encode kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "HammingDistance_Float"
        encoder.setComputePipelineState(floatKernel)
        
        // Set buffers
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(dataset, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        // Set parameters
        var params = (
            Q: UInt32(queryCount),
            N: UInt32(datasetCount),
            D: UInt32(dimension),
            threshold: config.binarizationThreshold
        )
        encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 3)
        
        // Dispatch
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: datasetCount,
            height: queryCount,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return HammingResult(
            distances: outputBuffer,
            queryCount: queryCount,
            datasetCount: datasetCount,
            normalized: false
        )
    }
    
    // MARK: - Convenience Methods
    
    /// Pack binary array into bit-packed format
    public func packBinary(_ binary: [Bool]) -> [UInt32] {
        let wordsNeeded = (binary.count + 31) / 32
        var packed = Array<UInt32>(repeating: 0, count: wordsNeeded)
        
        for (i, bit) in binary.enumerated() {
            if bit {
                let wordIndex = i / 32
                let bitIndex = i % 32
                packed[wordIndex] |= (1 << bitIndex)
            }
        }
        
        return packed
    }
    
    /// Compute Hamming distance between binary arrays
    public func distance(
        _ a: [Bool],
        _ b: [Bool]
    ) async throws -> Int {
        guard a.count == b.count else {
            throw AccelerationError.countMismatch
        }
        
        let packedA = packBinary(a)
        let packedB = packBinary(b)
        
        let bufferA = try computeEngine.createBuffer(from: packedA, options: MTLResourceOptions.storageModeShared)
        let bufferB = try computeEngine.createBuffer(from: packedB, options: MTLResourceOptions.storageModeShared)
        
        // Use single vector kernel
        guard let outputBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create output buffer")
        }
        
        // Initialize to 0
        outputBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        
        let commandBuffer = computeEngine.commandQueue.makeCommandBuffer()!
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.setComputePipelineState(singleKernel)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        var wordsCount = UInt32(packedA.count)
        encoder.setBytes(&wordsCount, length: MemoryLayout<UInt32>.stride, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = (packedA.count + 15) / 16  // 16 words per thread
        let gridSize = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return Int(outputBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
    
    /// Compute Hamming distance matrix for binary arrays
    public func distanceMatrix(
        queries: [[Bool]],
        dataset: [[Bool]],
        normalized: Bool = false
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw AccelerationError.invalidInput("Empty input arrays")
        }
        
        let dimension = queries[0].count
        guard dataset.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.countMismatch
        }
        
        // Pack all vectors
        let packedQueries = queries.flatMap { packBinary($0) }
        let packedDataset = dataset.flatMap { packBinary($0) }
        
        let queryBuffer = try computeEngine.createBuffer(
            from: packedQueries,
            options: MTLResourceOptions.storageModeShared
        )
        
        let datasetBuffer = try computeEngine.createBuffer(
            from: packedDataset,
            options: MTLResourceOptions.storageModeShared
        )
        
        let commandBuffer = computeEngine.commandQueue.makeCommandBuffer()!
        
        let result = try computeBitPacked(
            queries: queryBuffer,
            dataset: datasetBuffer,
            queryCount: queries.count,
            datasetCount: dataset.count,
            bitsPerVector: dimension,
            config: HammingConfig(normalized: normalized),
            commandBuffer: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result.asMatrix()
    }
    
    // MARK: - VectorCore Integration
    
    /// Compute Hamming distance using VectorCore binary vectors
    public func distance<V: VectorProtocol>(
        _ a: V,
        _ b: V,
        threshold: Float = 0.0
    ) async throws -> Int where V.Scalar == Float {
        let arrayA = Array(a.toArray())
        let arrayB = Array(b.toArray())
        
        // Binarize based on threshold
        let binaryA = arrayA.map { $0 > threshold }
        let binaryB = arrayB.map { $0 > threshold }
        
        return try await distance(binaryA, binaryB)
    }
    
    // MARK: - Performance Analysis
    
    public struct PerformanceMetrics {
        public let totalTime: TimeInterval
        public let throughput: Double // comparisons per second
        public let bitsPerSecond: Double
    }
    
    /// Benchmark Hamming distance performance
    public func benchmark(
        queryCount: Int = 100,
        datasetSize: Int = 10000,
        dimension: Int = 512,
        iterations: Int = 10
    ) async throws -> PerformanceMetrics {
        // Generate random binary data
        let queries = (0..<queryCount).map { _ in
            (0..<dimension).map { _ in Bool.random() }
        }
        
        let dataset = (0..<datasetSize).map { _ in
            (0..<dimension).map { _ in Bool.random() }
        }
        
        var times: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            _ = try await distanceMatrix(queries: queries, dataset: dataset)
            times.append(CACurrentMediaTime() - start)
        }
        
        let avgTime = times.reduce(0, +) / Double(times.count)
        let comparisons = Double(queryCount * datasetSize)
        let throughput = comparisons / avgTime
        let bitsPerSecond = throughput * Double(dimension)
        
        return PerformanceMetrics(
            totalTime: avgTime,
            throughput: throughput,
            bitsPerSecond: bitsPerSecond
        )
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation
public func cpuHammingDistance(_ a: [Bool], _ b: [Bool]) -> Int {
    guard a.count == b.count else { return Int.max }
    
    var distance = 0
    for i in 0..<a.count {
        if a[i] != b[i] {
            distance += 1
        }
    }
    
    return distance
}