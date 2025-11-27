import Metal
import QuartzCore
import Foundation
import QuartzCore
import VectorCore
import QuartzCore

// MARK: - Enhanced Minkowski Calculator

public class MinkowskiCalculator {
    let device: any MTLDevice
    let commandQueue: any MTLCommandQueue
    var pipelineState: (any MTLComputePipelineState)!

    // Constants matching the Metal kernel TILE_M/TILE_N
    private let TILE_SIZE = 16

    public enum CalculatorError: Error, LocalizedError {
        case initializationFailed(String)
        case invalidInputDimensions
        case executionFailed(String)
        case bufferCreationFailed
        
        public var errorDescription: String? {
            switch self {
            case .initializationFailed(let msg): return "Initialization failed: \(msg)"
            case .invalidInputDimensions: return "Invalid input dimensions"
            case .executionFailed(let msg): return "Execution failed: \(msg)"
            case .bufferCreationFailed: return "Failed to create Metal buffer"
            }
        }
    }
    
    // MARK: - Result Types
    
    public struct DistanceMatrix: Sendable {
        public let data: [Float]
        public let rows: Int  // M
        public let cols: Int  // N
        
        /// Access distance at (row, col)
        public subscript(row: Int, col: Int) -> Float {
            guard row < rows && col < cols else { return Float.infinity }
            return data[row * cols + col]
        }
        
        /// Get row as array
        public func row(_ index: Int) -> [Float] {
            guard index < rows else { return [] }
            let start = index * cols
            return Array(data[start..<start + cols])
        }
        
        /// Find minimum distance and its indices
        public func minimum() -> (value: Float, row: Int, col: Int) {
            var minVal = Float.infinity
            var minRow = 0
            var minCol = 0
            
            for i in 0..<rows {
                for j in 0..<cols {
                    if data[i * cols + j] < minVal {
                        minVal = data[i * cols + j]
                        minRow = i
                        minCol = j
                    }
                }
            }
            return (minVal, minRow, minCol)
        }
    }
    
    public struct PerformanceMetrics: Sendable {
        public let executionTime: TimeInterval
        public let throughput: Double  // distances per second
        public let pValue: Float
        public let metricName: String
    }

    // MARK: - Initialization

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CalculatorError.initializationFailed("Metal is not supported on this device")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw CalculatorError.initializationFailed("Could not create command queue")
        }
        self.commandQueue = queue

        try setupPipeline()
    }

    private func setupPipeline() throws {
        // Load the shader library using shared loader with fallback support
        let library = try KernelContext.getSharedLibrary(for: device)

        guard let kernelFunction = library.makeFunction(name: "minkowski_distance") else {
            throw CalculatorError.initializationFailed("Could not find kernel function 'minkowski_distance'")
        }

        do {
            pipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            throw CalculatorError.initializationFailed("Failed to create pipeline state: \(error)")
        }
        
        // Validate hardware support for the required 16x16 (256 threads) threadgroup size.
        if pipelineState.maxTotalThreadsPerThreadgroup < TILE_SIZE * TILE_SIZE {
            print("Warning: Hardware may not optimally support the 256 threads per threadgroup required by this kernel.")
        }
    }

    // MARK: - Core Calculation (Your Original Method)

    /// Calculates the pairwise Minkowski distances between two sets of vectors.
    public func calculateDistances(vectorA: [Float], vectorB: [Float], M: UInt32, N: UInt32, D: UInt32, p: Float) throws -> [Float] {
        
        // 1. Validate dimensions
        guard vectorA.count == Int(M * D) && vectorB.count == Int(N * D) else {
            throw CalculatorError.invalidInputDimensions
        }

        // 2. Create Buffers
        let floatStride = MemoryLayout<Float>.stride
        // Using MTLResourceOptions.storageModeShared for unified memory architectures (e.g., Apple Silicon)
        let bufferA = device.makeBuffer(bytes: vectorA, length: vectorA.count * floatStride, options: MTLResourceOptions.storageModeShared)
        let bufferB = device.makeBuffer(bytes: vectorB, length: vectorB.count * floatStride, options: MTLResourceOptions.storageModeShared)

        let outputLength = Int(M * N)
        let outputSize = outputLength * floatStride
        guard let bufferOutput = device.makeBuffer(length: outputSize, options: MTLResourceOptions.storageModeShared) else {
            throw CalculatorError.executionFailed("Could not allocate output buffer")
        }

        // Prepare scalar parameters
        var p_param = p
        var M_param = M
        var N_param = N
        var D_param = D
        var mode_param: UInt32 = 0 // Pairwise mode

        // 3. Create Command Buffer and Encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CalculatorError.executionFailed("Failed to create command buffer or encoder")
        }

        commandEncoder.setComputePipelineState(pipelineState)

        // Set buffers (matching [[buffer(index)]] in Metal)
        commandEncoder.setBuffer(bufferA, offset: 0, index: 0)
        commandEncoder.setBuffer(bufferB, offset: 0, index: 1)
        commandEncoder.setBuffer(bufferOutput, offset: 0, index: 2)

        // Set scalar parameters using setBytes
        let uintStride = MemoryLayout<UInt32>.stride
        commandEncoder.setBytes(&p_param, length: floatStride, index: 3)
        commandEncoder.setBytes(&M_param, length: uintStride, index: 4)
        commandEncoder.setBytes(&N_param, length: uintStride, index: 5)
        commandEncoder.setBytes(&D_param, length: uintStride, index: 6)
        commandEncoder.setBytes(&mode_param, length: uintStride, index: 7)

        // 4. Configure Threadgroups and Grid (As per specification)
        // Thread Configuration: 16x16 threadgroup
        let threadgroupSize = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)

        /* 
         Thread Configuration as specified in the requirements:
         let gridSize = MTLSize(
             width: (N + 15) / 16 * 16,
             height: (M + 15) / 16 * 16,
             depth: 1
         )
         We calculate the number of threadgroups required to achieve this padded grid size.
        */
        let numThreadgroups = MTLSize(
            width: (Int(N) + TILE_SIZE - 1) / TILE_SIZE,
            height: (Int(M) + TILE_SIZE - 1) / TILE_SIZE,
            depth: 1
        )

        // Dispatch the threadgroups. The kernel includes boundary checks (if (row < M && col < N)).
        commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadgroupSize)

        // 5. Dispatch and Wait
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // 6. Retrieve Results
        let dataPointer = bufferOutput.contents().bindMemory(to: Float.self, capacity: outputLength)
        let results = Array(UnsafeBufferPointer(start: dataPointer, count: outputLength))

        return results
    }
    
    // MARK: - Enhanced Methods
    
    /// Calculate distance matrix with structured result and validation
    public func calculateDistanceMatrix(vectorsA: [[Float]], vectorsB: [[Float]], p: Float) throws -> DistanceMatrix {
        guard !vectorsA.isEmpty && !vectorsB.isEmpty else {
            throw CalculatorError.invalidInputDimensions
        }
        
        let M = UInt32(vectorsA.count)
        let N = UInt32(vectorsB.count)
        let D = UInt32(vectorsA[0].count)
        
        // Validate all vectors have same dimension
        guard vectorsA.allSatisfy({ $0.count == Int(D) }) &&
              vectorsB.allSatisfy({ $0.count == Int(D) }) else {
            throw CalculatorError.invalidInputDimensions
        }
        
        // Flatten vectors
        let flatA = vectorsA.flatMap { $0 }
        let flatB = vectorsB.flatMap { $0 }
        
        let distances = try calculateDistances(
            vectorA: flatA,
            vectorB: flatB,
            M: M,
            N: N,
            D: D,
            p: p
        )
        
        return DistanceMatrix(data: distances, rows: Int(M), cols: Int(N))
    }
    
    /// Calculate distance between two single vectors
    public func calculateDistance(vectorA: [Float], vectorB: [Float], p: Float) throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw CalculatorError.invalidInputDimensions
        }
        
        let distances = try calculateDistances(
            vectorA: vectorA,
            vectorB: vectorB,
            M: 1,
            N: 1,
            D: UInt32(vectorA.count),
            p: p
        )
        
        return distances[0]
    }
    
    /// Find k nearest neighbors with efficient sorting
    public func findNearestNeighbors(query: [Float], dataset: [[Float]], k: Int, p: Float) throws -> [(index: Int, distance: Float)] {
        guard k > 0 && k <= dataset.count else {
            throw CalculatorError.invalidInputDimensions
        }
        
        let matrix = try calculateDistanceMatrix(
            vectorsA: [query],
            vectorsB: dataset,
            p: p
        )
        
        // Get distances for the query
        let distances = matrix.row(0)
        
        // Create indexed array and use partial sorting for efficiency
        var indexedDistances = distances.enumerated().map { (index: $0, distance: $1) }
        let partitionIndex = min(k, indexedDistances.count)
        indexedDistances.partialSort(partitionIndex, by: { $0.distance < $1.distance })
        
        return Array(indexedDistances.prefix(partitionIndex))
    }
    
    // MARK: - Async Variants
    
    /// Async version using continuation
    public func calculateDistancesAsync(vectorA: [Float], vectorB: [Float], M: UInt32, N: UInt32, D: UInt32, p: Float) async throws -> [Float] {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try calculateDistances(
                    vectorA: vectorA,
                    vectorB: vectorB,
                    M: M,
                    N: N,
                    D: D,
                    p: p
                )
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - Performance Analysis
    
    /// Benchmark performance for different p values with statistical analysis
    public func benchmark(M: Int = 100, N: Int = 1000, D: Int = 128, pValues: [Float] = [1.0, 2.0, 3.0, 10.0, 30.0], iterations: Int = 3) throws -> [PerformanceMetrics] {
        // Generate random test data
        let vectorsA = (0..<M*D).map { _ in Float.random(in: -1...1) }
        let vectorsB = (0..<N*D).map { _ in Float.random(in: -1...1) }
        
        var metrics: [PerformanceMetrics] = []
        
        for p in pValues {
            // Warm-up run
            _ = try calculateDistances(
                vectorA: vectorsA,
                vectorB: vectorsB,
                M: UInt32(M),
                N: UInt32(N),
                D: UInt32(D),
                p: p
            )
            
            // Multiple timed runs for statistical accuracy
            var times: [TimeInterval] = []
            for _ in 0..<iterations {
                let start = CACurrentMediaTime()
                _ = try calculateDistances(
                    vectorA: vectorsA,
                    vectorB: vectorsB,
                    M: UInt32(M),
                    N: UInt32(N),
                    D: UInt32(D),
                    p: p
                )
                times.append(CACurrentMediaTime() - start)
            }
            
            // Use median for stable measurement
            let executionTime = times.sorted()[times.count / 2]
            
            let throughput = Double(M * N) / executionTime
            let metricName = getMetricName(p: p)
            
            metrics.append(PerformanceMetrics(
                executionTime: executionTime,
                throughput: throughput,
                pValue: p,
                metricName: metricName
            ))
        }
        
        return metrics
    }
    
    private func getMetricName(p: Float) -> String {
        if abs(p - 1.0) < 0.001 { return "Manhattan (L1)" }
        if abs(p - 2.0) < 0.001 { return "Euclidean (L2)" }
        if p > 30.0 { return "Chebyshev (L∞)" }
        return "Minkowski (L\(p))"
    }
    
    // MARK: - VectorCore Integration

    /// Calculate distance using VectorCore protocol types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func calculateDistance<V: VectorProtocol>(_ a: V, _ b: V, p: Float) throws -> Float where V.Scalar == Float {
        guard a.count == b.count else {
            throw CalculatorError.invalidInputDimensions
        }

        let dimension = UInt32(a.count)

        // Zero-copy buffer creation using withUnsafeBufferPointer
        guard let bufferA = a.withUnsafeBufferPointer({ ptr -> (any MTLBuffer)? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }) else {
            throw CalculatorError.bufferCreationFailed
        }

        guard let bufferB = b.withUnsafeBufferPointer({ ptr -> (any MTLBuffer)? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }) else {
            throw CalculatorError.bufferCreationFailed
        }

        let distances = try calculateDistances(
            vectorA: bufferA,
            vectorB: bufferB,
            M: 1,
            N: 1,
            D: dimension,
            p: p
        )

        return distances[0]
    }

    /// Calculate distance matrix using VectorCore protocol types
    ///
    /// Uses zero-copy buffer creation via `withUnsafeBufferPointer`.
    public func calculateDistanceMatrix<V: VectorProtocol>(vectorsA: [V], vectorsB: [V], p: Float) throws -> DistanceMatrix where V.Scalar == Float {
        guard !vectorsA.isEmpty && !vectorsB.isEmpty else {
            throw CalculatorError.invalidInputDimensions
        }

        let M = UInt32(vectorsA.count)
        let N = UInt32(vectorsB.count)
        let D = UInt32(vectorsA[0].count)

        // Validate all vectors have same dimension
        guard vectorsA.allSatisfy({ $0.count == Int(D) }) &&
              vectorsB.allSatisfy({ $0.count == Int(D) }) else {
            throw CalculatorError.invalidInputDimensions
        }

        // Zero-copy buffer creation: flatten vectors directly into Metal buffer
        guard let bufferA = createBufferFromVectors(vectorsA, dimension: Int(D)) else {
            throw CalculatorError.bufferCreationFailed
        }

        guard let bufferB = createBufferFromVectors(vectorsB, dimension: Int(D)) else {
            throw CalculatorError.bufferCreationFailed
        }

        let distances = try calculateDistances(
            vectorA: bufferA,
            vectorB: bufferB,
            M: M,
            N: N,
            D: D,
            p: p
        )

        return DistanceMatrix(data: distances, rows: Int(M), cols: Int(N))
    }

    /// Create a Metal buffer directly from VectorProtocol array without intermediate allocations
    private func createBufferFromVectors<V: VectorProtocol>(
        _ vectors: [V],
        dimension: Int
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        guard !vectors.isEmpty else { return nil }

        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.stride

        // Create buffer
        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            return nil
        }

        // Copy each vector directly using withUnsafeBufferPointer
        let destination = buffer.contents().bindMemory(to: Float.self, capacity: totalCount)
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            vector.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                let dst = destination.advanced(by: offset)
                dst.update(from: srcBase, count: min(srcPtr.count, dimension))
            }
        }

        return buffer
    }

    /// Internal helper to calculate distances from Metal buffers
    private func calculateDistances(
        vectorA: any MTLBuffer,
        vectorB: any MTLBuffer,
        M: UInt32,
        N: UInt32,
        D: UInt32,
        p: Float
    ) throws -> [Float] {
        let outputLength = Int(M) * Int(N)
        guard let bufferOutput = device.makeBuffer(
            length: outputLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw CalculatorError.bufferCreationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CalculatorError.executionFailed("Failed to create command encoder")
        }

        var p_param = p
        var M_param = M
        var N_param = N
        var D_param = D
        var mode_param: UInt32 = 0

        let floatStride = MemoryLayout<Float>.stride
        let uintStride = MemoryLayout<UInt32>.stride

        commandEncoder.setComputePipelineState(pipelineState)
        commandEncoder.setBuffer(vectorA, offset: 0, index: 0)
        commandEncoder.setBuffer(vectorB, offset: 0, index: 1)
        commandEncoder.setBuffer(bufferOutput, offset: 0, index: 2)
        commandEncoder.setBytes(&p_param, length: floatStride, index: 3)
        commandEncoder.setBytes(&M_param, length: uintStride, index: 4)
        commandEncoder.setBytes(&N_param, length: uintStride, index: 5)
        commandEncoder.setBytes(&D_param, length: uintStride, index: 6)
        commandEncoder.setBytes(&mode_param, length: uintStride, index: 7)

        let threadgroupSize = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)
        let numThreadgroups = MTLSize(
            width: (Int(N) + TILE_SIZE - 1) / TILE_SIZE,
            height: (Int(M) + TILE_SIZE - 1) / TILE_SIZE,
            depth: 1
        )

        commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadgroupSize)
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let dataPointer = bufferOutput.contents().bindMemory(to: Float.self, capacity: outputLength)
        return Array(UnsafeBufferPointer(start: dataPointer, count: outputLength))
    }
    
    // MARK: - Batch Processing
    
    /// Process large datasets in batches to manage memory
    public func calculateDistancesBatched(
        vectorsA: [[Float]],
        vectorsB: [[Float]],
        p: Float,
        batchSize: Int = 1000
    ) async throws -> DistanceMatrix {
        let M = vectorsA.count
        let N = vectorsB.count
        guard !vectorsA.isEmpty && !vectorsB.isEmpty else {
            throw CalculatorError.invalidInputDimensions
        }
        
        let _ = vectorsA[0].count
        var allDistances: [Float] = []
        
        // Process in batches
        for i in stride(from: 0, to: M, by: batchSize) {
            let endI = min(i + batchSize, M)
            let batchA = Array(vectorsA[i..<endI])
            
            for j in stride(from: 0, to: N, by: batchSize) {
                let endJ = min(j + batchSize, N)
                let batchB = Array(vectorsB[j..<endJ])
                
                let batchMatrix = try await calculateDistanceMatrixAsync(
                    vectorsA: batchA,
                    vectorsB: batchB,
                    p: p
                )
                
                allDistances.append(contentsOf: batchMatrix.data)
            }
        }
        
        return DistanceMatrix(data: allDistances, rows: M, cols: N)
    }
    
    /// Async version of calculateDistanceMatrix
    private func calculateDistanceMatrixAsync(
        vectorsA: [[Float]],
        vectorsB: [[Float]],
        p: Float
    ) async throws -> DistanceMatrix {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try calculateDistanceMatrix(
                    vectorsA: vectorsA,
                    vectorsB: vectorsB,
                    p: p
                )
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

// MARK: - Array Extension for Partial Sorting

private extension Array {
    mutating func partialSort(_ k: Int, by areInIncreasingOrder: (Element, Element) -> Bool) {
        guard k > 0 && k <= count else { return }
        
        // Use nth_element algorithm for O(n) average complexity
        func partition(low: Int, high: Int) -> Int {
            let pivot = self[high]
            var i = low - 1
            
            for j in low..<high {
                if areInIncreasingOrder(self[j], pivot) {
                    i += 1
                    swapAt(i, j)
                }
            }
            swapAt(i + 1, high)
            return i + 1
        }
        
        func quickSelect(low: Int, high: Int, k: Int) {
            if low < high {
                let pi = partition(low: low, high: high)
                
                if pi == k {
                    return
                } else if pi < k {
                    quickSelect(low: pi + 1, high: high, k: k)
                } else {
                    quickSelect(low: low, high: pi - 1, k: k)
                }
            }
        }
        
        quickSelect(low: 0, high: count - 1, k: k - 1)
        
        // Sort the first k elements
        self[0..<k].sort(by: areInIncreasingOrder)
    }
}

// MARK: - CPU Reference Implementation

/// CPU reference implementation for validation
public func cpuMinkowskiDistance(_ a: [Float], _ b: [Float], p: Float) -> Float {
    guard a.count == b.count else { return Float.infinity }
    
    if abs(p - 1.0) < 0.001 {
        // Manhattan distance
        return zip(a, b).reduce(0.0) { $0 + abs($1.0 - $1.1) }
    } else if abs(p - 2.0) < 0.001 {
        // Euclidean distance
        let sum = zip(a, b).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        return sqrt(sum)
    } else if p > 30.0 {
        // Chebyshev distance approximation
        return zip(a, b).reduce(0.0) { max($0, abs($1.0 - $1.1)) }
    } else {
        // General Minkowski
        let sum = zip(a, b).reduce(0.0) { $0 + pow(abs($1.0 - $1.1), p) }
        return pow(sum, 1.0 / p)
    }
}
