// Minkowski Distance Kernel
// GPU-accelerated Minkowski distance (Lp norm) computation

import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Minkowski Distance Kernel

/// GPU-accelerated Minkowski distance computation
/// Supports arbitrary p values with optimizations for common cases (p=1, p=2, p→∞)
public final class MinkowskiDistanceKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    
    /// Configuration for Minkowski distance computation
    public struct Configuration {
        public let p: Float                    // Minkowski parameter
        public let mode: Mode                  // Computation mode
        public let useStableComputation: Bool  // Force stable computation for large p
        
        public enum Mode: UInt32 {
            case pairwise = 0  // Compute full M×N distance matrix
            case single = 1    // Compute single vector pair distance
        }
        
        public init(p: Float, mode: Mode = .pairwise, useStableComputation: Bool? = nil) {
            self.p = p
            self.mode = mode
            // Auto-determine stable computation if not specified
            self.useStableComputation = useStableComputation ?? (p > 10.0 && p <= 30.0)
        }
        
        /// Returns true if this is effectively Manhattan distance
        public var isManhattan: Bool {
            return abs(p - 1.0) < 0.001
        }
        
        /// Returns true if this is effectively Euclidean distance
        public var isEuclidean: Bool {
            return abs(p - 2.0) < 0.001
        }
        
        /// Returns true if this approximates Chebyshev distance
        public var isChebyshev: Bool {
            return p > 30.0
        }
        
        /// Get a descriptive name for the metric
        public var metricName: String {
            if isManhattan { return "Manhattan (L1)" }
            if isEuclidean { return "Euclidean (L2)" }
            if isChebyshev { return "Chebyshev (L∞)" }
            return "Minkowski (L\(p))"
        }
    }
    
    /// Result from Minkowski distance computation
    public struct DistanceResult {
        public let distances: any MTLBuffer
        public let rows: Int    // M dimension
        public let cols: Int    // N dimension
        public let p: Float
        
        /// Extract distance matrix as 2D array
        public func asMatrix() -> [[Float]] {
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
            var matrix: [[Float]] = []
            
            for i in 0..<rows {
                var row: [Float] = []
                for j in 0..<cols {
                    row.append(ptr[i * cols + j])
                }
                matrix.append(row)
            }
            
            return matrix
        }
        
        /// Get distance for specific pair
        public func distance(row: Int, col: Int) -> Float {
            guard row < rows && col < cols else { return Float.infinity }
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
            return ptr[row * cols + col]
        }
        
        /// Find minimum distance and its indices
        public func minimum() -> (value: Float, row: Int, col: Int) {
            let ptr = distances.contents().bindMemory(to: Float.self, capacity: rows * cols)
            var minVal = Float.infinity
            var minRow = 0
            var minCol = 0
            
            for i in 0..<rows {
                for j in 0..<cols {
                    let val = ptr[i * cols + j]
                    if val < minVal {
                        minVal = val
                        minRow = i
                        minCol = j
                    }
                }
            }
            
            return (minVal, minRow, minCol)
        }
    }
    
    // MARK: - Initialization
    
    public init(device: any MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw AccelerationError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)
        
        guard let function = library.makeFunction(name: "minkowski_distance") else {
            throw AccelerationError.shaderNotFound(name: "minkowski_distance")
        }
        
        self.pipeline = try device.makeComputePipelineState(function: function)
    }
    
    // MARK: - Core Computation
    
    /// Compute Minkowski distance matrix
    /// - Parameters:
    ///   - vectorsA: First set of vectors [M × D]
    ///   - vectorsB: Second set of vectors [N × D]
    ///   - M: Number of vectors in A
    ///   - N: Number of vectors in B
    ///   - D: Dimension of vectors
    ///   - config: Distance configuration
    ///   - commandBuffer: Command buffer for execution
    /// - Returns: Distance result
    public func computeDistances(
        vectorsA: any MTLBuffer,
        vectorsB: any MTLBuffer,
        M: Int,
        N: Int,
        D: Int,
        config: Configuration,
        commandBuffer: (any MTLCommandBuffer)? = nil
    ) throws -> DistanceResult {
        // Validate inputs
        guard M > 0 && N > 0 && D > 0 else {
            throw AccelerationError.invalidInput("Dimensions must be positive")
        }
        
        guard config.p > 0 else {
            throw AccelerationError.invalidInput("Minkowski parameter p must be positive")
        }
        
        // Allocate output buffer
        let outputSize = M * N * MemoryLayout<Float>.stride
        guard let distances = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        // Create or use provided command buffer
        let cmdBuffer = commandBuffer ?? commandQueue.makeCommandBuffer()!
        
        // Encode kernel
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw AccelerationError.encoderCreationFailed
        }
        
        encoder.label = "MinkowskiDistance_\(config.metricName)"
        encoder.setComputePipelineState(pipeline)
        
        // Set buffers
        encoder.setBuffer(vectorsA, offset: 0, index: 0)
        encoder.setBuffer(vectorsB, offset: 0, index: 1)
        encoder.setBuffer(distances, offset: 0, index: 2)
        
        // Set parameters
        var p = config.p
        var m = UInt32(M)
        var n = UInt32(N)
        var d = UInt32(D)
        var mode = config.mode.rawValue
        
        encoder.setBytes(&p, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&mode, length: MemoryLayout<UInt32>.stride, index: 7)
        
        // Dispatch with 16×16 threadgroups (matching TILE_M × TILE_N)
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (N + 15) / 16 * 16,
            height: (M + 15) / 16 * 16,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        // Execute if not using external command buffer
        if commandBuffer == nil {
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        
        return DistanceResult(distances: distances, rows: M, cols: N, p: config.p)
    }
    
    // MARK: - Convenience Methods
    
    /// Compute distance between two vectors
    public func distance(
        _ a: [Float],
        _ b: [Float],
        p: Float
    ) async throws -> Float {
        guard a.count == b.count else {
            throw AccelerationError.countMismatch(expected: a.count, actual: b.count)
        }
        
        let dimension = a.count
        
        // Create buffers
        guard let bufferA = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: b,
            length: b.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        let config = Configuration(p: p, mode: .pairwise)
        let result = try computeDistances(
            vectorsA: bufferA,
            vectorsB: bufferB,
            M: 1,
            N: 1,
            D: dimension,
            config: config
        )
        
        return result.distance(row: 0, col: 0)
    }
    
    /// Compute distance matrix for arrays of vectors
    public func distanceMatrix(
        queries: [[Float]],
        dataset: [[Float]],
        p: Float
    ) async throws -> [[Float]] {
        guard !queries.isEmpty && !dataset.isEmpty else {
            throw AccelerationError.invalidInput("Empty input arrays")
        }
        
        let dimension = queries[0].count
        guard queries.allSatisfy({ $0.count == dimension }) &&
              dataset.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.countMismatch(expected: dimension, actual: nil)
        }
        
        // Flatten vectors
        let flatQueries = queries.flatMap { $0 }
        let flatDataset = dataset.flatMap { $0 }
        
        // Create buffers
        guard let queriesBuffer = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        guard let datasetBuffer = device.makeBuffer(
            bytes: flatDataset,
            length: flatDataset.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        let config = Configuration(p: p)
        let result = try computeDistances(
            vectorsA: queriesBuffer,
            vectorsB: datasetBuffer,
            M: queries.count,
            N: dataset.count,
            D: dimension,
            config: config
        )
        
        return result.asMatrix()
    }
    
    /// Find k nearest neighbors using Minkowski distance
    public func findNearestNeighbors(
        query: [Float],
        dataset: [[Float]],
        k: Int,
        p: Float
    ) async throws -> [(index: Int, distance: Float)] {
        let distances = try await distanceMatrix(
            queries: [query],
            dataset: dataset,
            p: p
        )
        
        // Get distances for the single query
        let queryDistances = distances[0]
        
        // Create indexed array and sort
        var indexedDistances: [(index: Int, distance: Float)] = []
        for (index, distance) in queryDistances.enumerated() {
            indexedDistances.append((index: index, distance: distance))
        }
        
        // Sort by distance and return top k
        indexedDistances.sort { $0.distance < $1.distance }
        return Array(indexedDistances.prefix(k))
    }
    
    // MARK: - VectorCore Integration
    
    /// Compute Minkowski distance using VectorCore types
    public func distance<V: VectorProtocol>(
        _ a: V,
        _ b: V,
        p: Float
    ) async throws -> Float where V.Scalar == Float {
        let arrayA = Array(a.toArray())
        let arrayB = Array(b.toArray())
        return try await distance(arrayA, arrayB, p: p)
    }
    
    /// Compute distance matrix using VectorCore types
    public func distanceMatrix<V: VectorProtocol>(
        queries: [V],
        dataset: [V],
        p: Float
    ) async throws -> [[Float]] where V.Scalar == Float {
        let queryArrays = queries.map { Array($0.toArray()) }
        let datasetArrays = dataset.map { Array($0.toArray()) }
        return try await distanceMatrix(
            queries: queryArrays,
            dataset: datasetArrays,
            p: p
        )
    }
    
    // MARK: - Performance Analysis
    
    public struct PerformanceMetrics: Sendable {
        public let executionTime: TimeInterval
        public let throughput: Double // distances per second
        public let effectiveMemoryBandwidth: Double // GB/s
        public let metricType: String
    }
    
    /// Benchmark Minkowski distance performance for different p values
    public func benchmark(
        M: Int = 100,
        N: Int = 1000,
        D: Int = 128,
        pValues: [Float] = [1.0, 2.0, 3.0, 10.0, 30.0],
        iterations: Int = 10
    ) async throws -> [Float: PerformanceMetrics] {
        // Generate random test data
        let vectorsA = (0..<M).flatMap { _ in
            (0..<D).map { _ in Float.random(in: -1...1) }
        }
        
        let vectorsB = (0..<N).flatMap { _ in
            (0..<D).map { _ in Float.random(in: -1...1) }
        }
        
        guard let bufferA = device.makeBuffer(
            bytes: vectorsA,
            length: vectorsA.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        guard let bufferB = device.makeBuffer(
            bytes: vectorsB,
            length: vectorsB.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw AccelerationError.bufferCreationFailed("Failed to create buffer")
        }
        
        var results: [Float: PerformanceMetrics] = [:]
        
        for p in pValues {
            let config = Configuration(p: p)
            var times: [TimeInterval] = []
            
            // Warm-up
            _ = try computeDistances(
                vectorsA: bufferA,
                vectorsB: bufferB,
                M: M,
                N: N,
                D: D,
                config: config
            )
            
            // Benchmark iterations
            for _ in 0..<iterations {
                let start = CACurrentMediaTime()
                
                let commandBuffer = commandQueue.makeCommandBuffer()!
                _ = try computeDistances(
                    vectorsA: bufferA,
                    vectorsB: bufferB,
                    M: M,
                    N: N,
                    D: D,
                    config: config,
                    commandBuffer: commandBuffer
                )
                commandBuffer.commit()
                _ = await commandBuffer.completed
                
                times.append(CACurrentMediaTime() - start)
            }
            
            // Calculate metrics
            let avgTime = times.reduce(0, +) / Double(times.count)
            let numDistances = Double(M * N)
            let throughput = numDistances / avgTime
            
            // Memory bandwidth (simplified): reading M×D + N×D floats, writing M×N floats
            let bytesRead = Double((M + N) * D * MemoryLayout<Float>.stride)
            let bytesWritten = Double(M * N * MemoryLayout<Float>.stride)
            let bandwidth = (bytesRead + bytesWritten) / avgTime / 1e9
            
            results[p] = PerformanceMetrics(
                executionTime: avgTime,
                throughput: throughput,
                effectiveMemoryBandwidth: bandwidth,
                metricType: config.metricName
            )
        }
        
        return results
    }
    
    // MARK: - Validation
    
    /// Validate Minkowski distance properties
    public func validateMetricProperties(
        vectors: [[Float]],
        p: Float,
        tolerance: Float = 1e-5
    ) async throws -> Bool {
        let distances = try await distanceMatrix(
            queries: vectors,
            dataset: vectors,
            p: p
        )
        
        let n = vectors.count
        
        // Check non-negativity
        for i in 0..<n {
            for j in 0..<n {
                if distances[i][j] < -tolerance {
                    print("Non-negativity violated at (\(i), \(j)): \(distances[i][j])")
                    return false
                }
            }
        }
        
        // Check identity (d(x,x) = 0)
        for i in 0..<n {
            if abs(distances[i][i]) > tolerance {
                print("Identity violated at \(i): \(distances[i][i])")
                return false
            }
        }
        
        // Check symmetry (d(x,y) = d(y,x))
        for i in 0..<n {
            for j in 0..<n {
                if abs(distances[i][j] - distances[j][i]) > tolerance {
                    print("Symmetry violated at (\(i), \(j))")
                    return false
                }
            }
        }
        
        // Check triangle inequality (only for p >= 1)
        if p >= 1.0 {
            for i in 0..<n {
                for j in 0..<n {
                    for k in 0..<n {
                        let direct = distances[i][j]
                        let indirect = distances[i][k] + distances[k][j]
                        if direct > indirect + tolerance {
                            print("Triangle inequality violated: d(\(i),\(j))=\(direct) > d(\(i),\(k))+d(\(k),\(j))=\(indirect)")
                            return false
                        }
                    }
                }
            }
        }
        
        return true
    }
}

// MARK: - CPU Reference Implementation

// Note: CPU reference implementation is available in MinkowskiCalculator.swift
