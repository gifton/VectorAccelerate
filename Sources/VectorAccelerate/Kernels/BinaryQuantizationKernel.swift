// Binary Quantization Kernel
// GPU-accelerated binary quantization with bit-packing and Hamming distance computation

import Metal
import Foundation
import VectorCore
import QuartzCore
import Accelerate

// MARK: - Binary Quantization Kernel

/// GPU-accelerated binary quantization and Hamming distance computation
/// Optimized for bit-packed storage and efficient binary operations
public final class BinaryQuantizationKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let quantizeKernel: any MTLComputePipelineState
    private let hammingKernel: any MTLComputePipelineState
    
    // MARK: - Types
    
    /// Bit-packed binary vector representation
    public struct BinaryVector: Sendable {
        public let data: [UInt32]  // Use Swift array for Sendable compliance
        public let dimension: Int
        public let numWords: Int
        
        /// Initialize from raw data
        public init(data: [UInt32], dimension: Int) {
            self.data = data
            self.dimension = dimension
            self.numWords = (dimension + 31) / 32
        }
        
        /// Extract as bool array (CPU operation). Assumes LSB first ordering (per spec).
        public func asBoolArray() -> [Bool] {
            var bools = [Bool](repeating: false, count: dimension)
            
            for i in 0..<dimension {
                let wordIndex = i / 32
                let bitIndex = i % 32
                if wordIndex < data.count && (data[wordIndex] & (1 << bitIndex)) != 0 {
                    bools[i] = true
                }
            }
            return bools
        }
        
        /// Get bit at specific dimension (CPU operation)
        public func bit(at index: Int) -> Bool {
            guard index >= 0 && index < dimension else { return false }
            let wordIndex = index / 32
            let bitIndex = index % 32
            
            guard wordIndex < data.count else { return false }
            return (data[wordIndex] & (1 << bitIndex)) != 0
        }
        
        /// Convert to float array (0.0 or 1.0 values)
        public func asFloatArray() -> [Float] {
            return asBoolArray().map { $0 ? 1.0 : 0.0 }
        }
        
        /// Hamming distance to another binary vector
        public func hammingDistance(to other: BinaryVector) -> Int {
            guard dimension == other.dimension else { return -1 }
            
            var distance = 0
            for i in 0..<numWords {
                let word1 = i < data.count ? data[i] : 0
                let word2 = i < other.data.count ? other.data[i] : 0
                distance += (word1 ^ word2).nonzeroBitCount
            }
            
            return distance
        }
    }
    
    /// Batch of binary vectors
    public struct BinaryVectorBatch: Sendable {
        public let vectors: [BinaryVector]
        public let dimension: Int
        public let numWords: Int
        
        /// Initialize from vectors
        public init(vectors: [BinaryVector]) throws {
            guard let first = vectors.first else {
                throw AccelerationError.invalidInput("Cannot create empty batch")
            }
            
            // Validate all vectors have same dimension
            for vector in vectors {
                guard vector.dimension == first.dimension else {
                    throw AccelerationError.countMismatch(
                        expected: first.dimension,
                        actual: vector.dimension
                    )
                }
            }
            
            self.vectors = vectors
            self.dimension = first.dimension
            self.numWords = first.numWords
        }
        
        /// Number of vectors in batch
        public var count: Int { vectors.count }
        
        /// Get specific vector
        public func vector(at index: Int) throws -> BinaryVector {
            guard index >= 0 && index < vectors.count else {
                throw AccelerationError.invalidInput("Index \(index) out of bounds for \(vectors.count) vectors")
            }
            return vectors[index]
        }
        
        /// Extract all as float arrays
        public func asFloatArrays() -> [[Float]] {
            return vectors.map { $0.asFloatArray() }
        }
    }
    
    /// Result from quantization operation
    public struct QuantizationResult: Sendable {
        public let binaryVectors: BinaryVectorBatch
        public let compressionRatio: Float
        public let originalBytes: Int
        public let compressedBytes: Int
        public let executionTime: TimeInterval
        
        /// Bits per original float
        public var bitsPerFloat: Float {
            Float(compressedBytes * 8) / Float(originalBytes / MemoryLayout<Float>.stride)
        }
        
        /// Space savings percentage
        public var spaceSavings: Float {
            compressionRatio > 0 ? (1.0 - 1.0/compressionRatio) * 100.0 : 0.0
        }
    }
    
    /// Result from Hamming distance computation
    public struct HammingDistanceResult: Sendable {
        public let distances: [Float]
        public let minDistance: Float
        public let maxDistance: Float
        public let meanDistance: Float
        public let executionTime: TimeInterval
        
        /// Standard deviation of distances
        public var stdDeviation: Float {
            let variance = distances.reduce(0.0) { sum, distance in
                let diff = distance - meanDistance
                return sum + diff * diff
            } / Float(distances.count)
            return sqrt(variance)
        }
        
        /// Get distances as integers (since Hamming distances are always whole numbers)
        public var integerDistances: [Int] {
            return distances.map { Int($0) }
        }
    }
    
    /// Result from pairwise distance computation
    public struct PairwiseDistanceResult: Sendable {
        public let distanceMatrix: [[Float]]
        public let rows: Int
        public let cols: Int
        public let executionTime: TimeInterval
        public let averageDistance: Double
        
        /// Get distance at (row, col)
        public func distance(row: Int, col: Int) -> Float {
            guard row >= 0 && row < rows && col >= 0 && col < cols else {
                return -1.0
            }
            return distanceMatrix[row][col]
        }
        
        /// Flatten matrix to array
        public func asArray() -> [Float] {
            return distanceMatrix.flatMap { $0 }
        }
    }
    
    /// Configuration for quantization operations
    public struct QuantizationConfig: Sendable {
        public let threshold: Float           // Threshold for binary conversion
        public let useSignBit: Bool          // Use sign bit (vs threshold comparison)
        public let normalizeInput: Bool      // Normalize input vectors before quantization
        public let batchSize: Int            // Optimal batch size
        
        public init(
            threshold: Float = 0.0,
            useSignBit: Bool = false,
            normalizeInput: Bool = false,
            batchSize: Int = 1024
        ) {
            self.threshold = threshold
            self.useSignBit = useSignBit
            self.normalizeInput = normalizeInput
            self.batchSize = batchSize
        }
        
        public static let `default` = QuantizationConfig()
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.deviceInitializationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        // Load quantization kernel
        guard let quantizeFunc = library.makeFunction(name: "binaryQuantize") else {
            throw AccelerationError.shaderNotFound(name: "binaryQuantize")
        }
        
        // Load Hamming distance kernel
        guard let hammingFunc = library.makeFunction(name: "binaryHammingDistance") else {
            throw AccelerationError.shaderNotFound(name: "binaryHammingDistance")
        }
        
        do {
            self.quantizeKernel = try device.makeComputePipelineState(function: quantizeFunc)
            self.hammingKernel = try device.makeComputePipelineState(function: hammingFunc)
        } catch {
            throw AccelerationError.computeFailed(reason: "Failed to create pipeline states: \(error)")
        }
        
        // Validate hardware support
        let maxThreadsPerThreadgroup = quantizeKernel.maxTotalThreadsPerThreadgroup
        if maxThreadsPerThreadgroup < 32 {
            throw AccelerationError.unsupportedOperation(
                "Device does not support required threadgroup size for binary operations"
            )
        }
    }
    
    // MARK: - Quantization Operations
    
    /// Quantize float vectors to binary (bit-packed)
    /// - Parameters:
    ///   - vectors: Input float vectors
    ///   - config: Quantization configuration
    /// - Returns: Quantized binary vectors with compression metrics
    public func quantize(
        vectors: [[Float]],
        config: QuantizationConfig = .default
    ) throws -> QuantizationResult {
        guard !vectors.isEmpty, let dimension = vectors.first?.count, dimension > 0 else {
            throw AccelerationError.invalidInput("Input vectors cannot be empty")
        }
        
        // Validate all vectors have same dimension
        for (_, vector) in vectors.enumerated() {
            guard vector.count == dimension else {
                throw AccelerationError.countMismatch(expected: dimension, actual: vector.count)
            }
        }
        
        let numVectors = vectors.count
        let numWords = (dimension + 31) / 32
        let originalBytes = numVectors * dimension * MemoryLayout<Float>.stride
        let compressedBytes = numVectors * numWords * MemoryLayout<UInt32>.stride
        
        // Prepare input data
        let flatVectors = vectors.flatMap { $0 }
        
        // Create buffers
        guard let inputBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: originalBytes,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: originalBytes)
        }
        
        guard let outputBuffer = device.makeBuffer(
            length: compressedBytes,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: compressedBytes)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(quantizeKernel)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        // Set parameters
        var params = SIMD4<UInt32>(
            UInt32(dimension),          // dimension
            UInt32(numVectors),         // numVectors
            config.useSignBit ? 1 : 0,  // useSignBit flag
            0                           // padding
        )
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
        
        var threshold = config.threshold
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 3)
        
        // Configure thread groups (one thread per vector)
        let threadgroupSize = MTLSize(width: min(numVectors, quantizeKernel.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (numVectors + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Binary quantization failed: \(error)")
        }
        
        // Extract results
        let resultPointer = outputBuffer.contents().bindMemory(to: UInt32.self, capacity: numVectors * numWords)
        var binaryVectors: [BinaryVector] = []
        
        for i in 0..<numVectors {
            let start = i * numWords
            let vectorData = Array(UnsafeBufferPointer(start: resultPointer + start, count: numWords))
            binaryVectors.append(BinaryVector(data: vectorData, dimension: dimension))
        }
        
        let batch = try BinaryVectorBatch(vectors: binaryVectors)
        let compressionRatio = Float(originalBytes) / Float(compressedBytes)
        
        return QuantizationResult(
            binaryVectors: batch,
            compressionRatio: compressionRatio,
            originalBytes: originalBytes,
            compressedBytes: compressedBytes,
            executionTime: executionTime
        )
    }
    
    /// Quantize single vector
    public func quantize(
        vector: [Float],
        config: QuantizationConfig = .default
    ) throws -> BinaryVector {
        let result = try quantize(vectors: [vector], config: config)
        return try result.binaryVectors.vector(at: 0)
    }
    
    // MARK: - Distance Operations
    
    /// Compute Hamming distances (1 vs N)
    /// - Parameters:
    ///   - query: Query binary vector
    ///   - candidates: Candidate binary vectors
    /// - Returns: Hamming distances with statistics
    public func computeHammingDistances(
        query: BinaryVector,
        candidates: BinaryVectorBatch
    ) throws -> HammingDistanceResult {
        guard query.dimension == candidates.dimension else {
            throw AccelerationError.countMismatch(
                expected: query.dimension,
                actual: candidates.dimension
            )
        }
        
        let numCandidates = candidates.count
        guard numCandidates > 0 else {
            return HammingDistanceResult(
                distances: [],
                minDistance: 0,
                maxDistance: 0,
                meanDistance: 0,
                executionTime: 0
            )
        }
        
        // Flatten candidate data for GPU buffer
        let flatCandidates = candidates.vectors.flatMap { $0.data }
        
        // Create buffers
        guard let queryBuffer = device.makeBuffer(
            bytes: query.data,
            length: query.numWords * MemoryLayout<UInt32>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: query.numWords * MemoryLayout<UInt32>.stride)
        }
        
        let candidatesSize = flatCandidates.count * MemoryLayout<UInt32>.stride
        guard let candidatesBuffer = device.makeBuffer(
            bytes: flatCandidates,
            length: candidatesSize,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferAllocationFailed(size: candidatesSize)
        }
        
        let resultSize = numCandidates * MemoryLayout<Float>.stride
        guard let resultBuffer = device.makeBuffer(length: resultSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferAllocationFailed(size: resultSize)
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.computeFailed(reason: "Failed to create command encoder")
        }
        
        // Configure compute pass
        encoder.setComputePipelineState(hammingKernel)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        // Set parameters
        var params = SIMD2<UInt32>(UInt32(query.numWords), UInt32(numCandidates))
        encoder.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        
        // Configure thread groups (one thread per candidate)
        let threadgroupSize = MTLSize(width: min(numCandidates, hammingKernel.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (numCandidates + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        // Execute
        let startTime = CACurrentMediaTime()
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let executionTime = CACurrentMediaTime() - startTime
        
        // Check for errors
        if let error = commandBuffer.error {
            throw AccelerationError.computeFailed(reason: "Hamming distance computation failed: \(error)")
        }
        
        // Extract results and calculate statistics using Accelerate
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: numCandidates)
        let distances = Array(UnsafeBufferPointer(start: resultPointer, count: numCandidates))
        
        var minVal: Float = 0
        var maxVal: Float = 0
        var meanVal: Float = 0
        
        if !distances.isEmpty {
            vDSP_minv(distances, 1, &minVal, vDSP_Length(numCandidates))
            vDSP_maxv(distances, 1, &maxVal, vDSP_Length(numCandidates))
            vDSP_meanv(distances, 1, &meanVal, vDSP_Length(numCandidates))
        }
        
        return HammingDistanceResult(
            distances: distances,
            minDistance: minVal,
            maxDistance: maxVal,
            meanDistance: meanVal,
            executionTime: executionTime
        )
    }
    
    /// Compute pairwise Hamming distances (M vs N)
    /// - Parameters:
    ///   - vectorsA: First set of binary vectors (M vectors)
    ///   - vectorsB: Second set of binary vectors (N vectors)
    /// - Returns: M×N distance matrix with performance metrics
    public func computePairwiseDistances(
        vectorsA: BinaryVectorBatch,
        vectorsB: BinaryVectorBatch
    ) throws -> PairwiseDistanceResult {
        guard vectorsA.dimension == vectorsB.dimension else {
            throw AccelerationError.countMismatch(
                expected: vectorsA.dimension,
                actual: vectorsB.dimension
            )
        }
        
        let M = vectorsA.count
        let N = vectorsB.count
        
        let startTime = CACurrentMediaTime()
        
        // Compute distances row by row for better memory access patterns
        var distanceMatrix: [[Float]] = []
        var totalDistance: Double = 0.0
        
        for i in 0..<M {
            let query = vectorsA.vectors[i]
            let result = try computeHammingDistances(query: query, candidates: vectorsB)
            distanceMatrix.append(result.distances)
            totalDistance += result.distances.reduce(0.0) { $0 + Double($1) }
        }
        
        let executionTime = CACurrentMediaTime() - startTime
        let averageDistance = totalDistance / Double(M * N)
        
        return PairwiseDistanceResult(
            distanceMatrix: distanceMatrix,
            rows: M,
            cols: N,
            executionTime: executionTime,
            averageDistance: averageDistance
        )
    }
    
    // MARK: - Utility Methods
    
    /// Calculate compression ratio for given dimension
    public static func calculateCompressionRatio(dimension: Int) -> Float {
        let originalSize = Float(dimension * MemoryLayout<Float>.stride)
        let numWords = (dimension + 31) / 32
        let compressedSize = Float(numWords * MemoryLayout<UInt32>.stride)
        return compressedSize > 0 ? originalSize / compressedSize : 0
    }
    
    /// Validate that two binary vectors can be compared
    public static func canCompare(_ vector1: BinaryVector, _ vector2: BinaryVector) -> Bool {
        return vector1.dimension == vector2.dimension
    }
    
    // MARK: - Async Operations
    
    /// Async version of quantization
    public func quantizeAsync(
        vectors: [[Float]],
        config: QuantizationConfig = .default
    ) async throws -> QuantizationResult {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try quantize(vectors: vectors, config: config)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    /// Async version of Hamming distance computation
    public func computeHammingDistancesAsync(
        query: BinaryVector,
        candidates: BinaryVectorBatch
    ) async throws -> HammingDistanceResult {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try computeHammingDistances(query: query, candidates: candidates)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - VectorCore Integration
    
    /// Quantize using VectorCore protocol types
    public func quantize<V: VectorProtocol>(
        vectors: [V],
        config: QuantizationConfig = .default
    ) throws -> QuantizationResult where V.Scalar == Float {
        let floatArrays = vectors.map { $0.toArray() }
        return try quantize(vectors: floatArrays, config: config)
    }
    
    /// Single vector quantization using VectorCore
    public func quantize<V: VectorProtocol>(
        vector: V,
        config: QuantizationConfig = .default
    ) throws -> BinaryVector where V.Scalar == Float {
        return try quantize(vector: vector.toArray(), config: config)
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark quantization performance for different vector sizes
    public func benchmark(dimensions: [Int], vectorCount: Int = 1000) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        for dimension in dimensions {
            // Generate random test vectors
            let testVectors = (0..<vectorCount).map { _ in
                (0..<dimension).map { _ in Float.random(in: -1...1) }
            }
            
            // Warm-up run
            _ = try quantize(vectors: Array(testVectors.prefix(10)))
            
            // Timed runs
            var times: [TimeInterval] = []
            for _ in 0..<3 {
                let result = try quantize(vectors: testVectors)
                times.append(result.executionTime)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            let throughput = Double(vectorCount * dimension) / avgTime / 1e6 // Million elements per second
            let compressionRatio = Self.calculateCompressionRatio(dimension: dimension)
            
            results.append(BenchmarkResult(
                dimension: dimension,
                vectorCount: vectorCount,
                executionTime: avgTime,
                throughputMEPS: throughput,
                compressionRatio: compressionRatio
            ))
        }
        
        return results
    }
    
    public struct BenchmarkResult: Sendable {
        public let dimension: Int
        public let vectorCount: Int
        public let executionTime: TimeInterval
        public let throughputMEPS: Double
        public let compressionRatio: Float
    }
}
