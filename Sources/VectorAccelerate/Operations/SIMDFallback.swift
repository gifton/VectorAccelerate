//
//  SIMDFallback.swift
//  VectorAccelerate
//
//  High-performance CPU fallback using Accelerate and SIMD
//

import Foundation
import Accelerate
import simd
import VectorCore

/// Configuration for SIMD operations
public struct SIMDConfiguration: Sendable {
    public let vectorWidth: Int
    public let useAccelerate: Bool
    public let parallelThreshold: Int
    public let enablePrefetch: Bool
    
    public init(
        vectorWidth: Int = 8,  // 8 floats for AVX/NEON
        useAccelerate: Bool = true,
        parallelThreshold: Int = 1024,
        enablePrefetch: Bool = true
    ) {
        self.vectorWidth = vectorWidth
        self.useAccelerate = useAccelerate
        self.parallelThreshold = parallelThreshold
        self.enablePrefetch = enablePrefetch
    }
    
    public static let `default` = SIMDConfiguration()
    public static let performance = SIMDConfiguration(
        vectorWidth: 16,
        parallelThreshold: 512,
        enablePrefetch: true
    )
}

/// SIMD-optimized CPU operations
public final class SIMDFallback: @unchecked Sendable {
    private let configuration: SIMDConfiguration
    private let logger: Logger
    private let queue: DispatchQueue
    
    // Performance tracking
    private var operationCount: Int = 0
    private var totalTime: TimeInterval = 0
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    public init(configuration: SIMDConfiguration = .default) {
        self.configuration = configuration
        self.logger = Logger.shared
        self.queue = DispatchQueue(
            label: "com.vectoraccelerate.simd",
            qos: .userInitiated,
            attributes: .concurrent
        )
    }
    
    // MARK: - Vector Operations
    
    /// Compute dot product using SIMD
    ///
    /// Calculates the inner product of two vectors:
    /// ```
    /// result = Σ(i=0 to n-1) a[i] * b[i]
    /// ```
    ///
    /// Uses either Accelerate framework's optimized BLAS routines or
    /// manual SIMD vectorization with 8-wide float operations.
    ///
    /// - Complexity: O(n) with SIMD acceleration achieving ~8x speedup
    /// - Memory: O(1) additional space
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector (must be same length as a)
    /// - Returns: Scalar dot product
    /// - Throws: `VectorError.dimensionMismatch` if vectors have different lengths
    public func dotProduct(_ a: [Float], _ b: [Float]) async throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }
        
        if configuration.useAccelerate {
            return accelerateDotProduct(a, b)
        } else {
            return simdDotProduct(a, b)
        }
    }
    
    /// Accelerate-based dot product
    private func accelerateDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
    
    /// Manual SIMD dot product
    private func simdDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        let count = a.count
        // Use SIMD8 regardless of configuration to avoid out-of-bounds access
        let vectorWidth = min(configuration.vectorWidth, 8)
        var sum: Float = 0
        
        // Process vectors in chunks
        let fullVectors = count / vectorWidth
        
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                var vectorSum = SIMD8<Float>.zero
                
                // Vectorized loop
                for i in 0..<fullVectors {
                    let offset = i * vectorWidth
                    var va = SIMD8<Float>.zero
                    var vb = SIMD8<Float>.zero
                    for j in 0..<vectorWidth {
                        va[j] = aPtr[offset + j]
                        vb[j] = bPtr[offset + j]
                    }
                    vectorSum += va * vb
                }
                
                // Reduce vector sum
                sum = vectorSum.sum()
                
                // Handle remaining elements
                for i in (fullVectors * vectorWidth)..<count {
                    sum += aPtr[i] * bPtr[i]
                }
            }
        }
        
        return sum
    }
    
    /// Compute Euclidean distance using SIMD
    public func euclideanDistance(_ a: [Float], _ b: [Float]) async throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }
        
        if configuration.useAccelerate {
            return accelerateEuclideanDistance(a, b)
        } else {
            return simdEuclideanDistance(a, b)
        }
    }
    
    /// Accelerate-based Euclidean distance
    private func accelerateEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var distance: Float = 0
        vDSP_distancesq(a, 1, b, 1, &distance, vDSP_Length(a.count))
        return sqrt(distance)
    }
    
    /// Manual SIMD Euclidean distance
    private func simdEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        let count = a.count
        // Use SIMD8 regardless of configuration to avoid out-of-bounds access
        let vectorWidth = min(configuration.vectorWidth, 8)
        var sumSquared: Float = 0

        let fullVectors = count / vectorWidth

        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                var vectorSum = SIMD8<Float>.zero

                for i in 0..<fullVectors {
                    let offset = i * vectorWidth
                    var va = SIMD8<Float>.zero
                    var vb = SIMD8<Float>.zero
                    for j in 0..<vectorWidth {
                        va[j] = aPtr[offset + j]
                        vb[j] = bPtr[offset + j]
                    }
                    let diff = va - vb
                    vectorSum += diff * diff
                }

                sumSquared = vectorSum.sum()

                // Handle remaining elements
                for i in (fullVectors * vectorWidth)..<count {
                    let diff = aPtr[i] - bPtr[i]
                    sumSquared += diff * diff
                }
            }
        }

        return sqrt(sumSquared)
    }

    // MARK: - Manhattan Distance (VectorCore 0.1.5 SIMD-optimized)

    /// Compute Manhattan (L1) distance using VectorCore's SIMD4-optimized implementation
    ///
    /// VectorCore 0.1.5 provides 3-4x faster Manhattan distance computation using SIMD4 vectorization.
    /// This method delegates to VectorCore's optimized `ManhattanDistance` for best performance.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector (must be same length as a)
    /// - Returns: L1 distance (sum of absolute differences)
    /// - Throws: `VectorError.dimensionMismatch` if vectors have different lengths
    public func manhattanDistance(_ a: [Float], _ b: [Float]) async throws -> Float {
        guard a.count == b.count else {
            throw VectorError.dimensionMismatch(expected: a.count, actual: b.count)
        }

        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }

        // Delegate to VectorCore's SIMD4-optimized ManhattanDistance (0.1.5)
        // Use DynamicVector for optimal performance with VectorCore's optimized path
        let vectorA = DynamicVector(a)
        let vectorB = DynamicVector(b)
        return ManhattanDistance().distance(vectorA, vectorB)
    }

    /// Compute Manhattan distance for VectorProtocol types using VectorCore's SIMD optimization
    ///
    /// This generic version works with any VectorProtocol conforming type, leveraging
    /// VectorCore 0.1.5's SIMD4-vectorized Manhattan distance implementation.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: L1 distance
    public func manhattanDistance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        return ManhattanDistance().distance(a, b)
    }
    
    /// Normalize vector using SIMD
    public func normalize(_ vector: [Float]) async -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }
        
        if configuration.useAccelerate {
            return accelerateNormalize(vector)
        } else {
            return simdNormalize(vector)
        }
    }
    
    /// Accelerate-based normalization
    private func accelerateNormalize(_ vector: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        
        guard norm > Float.ulpOfOne else {
            return vector  // Avoid division by zero
        }
        
        var result = [Float](repeating: 0, count: vector.count)
        var invNorm = 1.0 / norm
        vDSP_vsmul(vector, 1, &invNorm, &result, 1, vDSP_Length(vector.count))
        
        return result
    }
    
    /// Manual SIMD normalization
    private func simdNormalize(_ vector: [Float]) -> [Float] {
        let count = vector.count
        // Use SIMD8 regardless of configuration to avoid out-of-bounds access
        let vectorWidth = min(configuration.vectorWidth, 8)
        
        // Compute norm
        var norm: Float = 0
        let fullVectors = count / vectorWidth
        
        vector.withUnsafeBufferPointer { ptr in
            var vectorSum = SIMD8<Float>.zero
            
            for i in 0..<fullVectors {
                let offset = i * vectorWidth
                var v = SIMD8<Float>.zero
                for j in 0..<vectorWidth {
                    v[j] = ptr[offset + j]
                }
                vectorSum += v * v
            }
            
            norm = vectorSum.sum()
            
            for i in (fullVectors * vectorWidth)..<count {
                norm += ptr[i] * ptr[i]
            }
        }
        
        norm = sqrt(norm)
        
        guard norm > Float.ulpOfOne else {
            return vector
        }
        
        // Normalize
        var result = [Float](repeating: 0, count: count)
        let invNorm = 1.0 / norm
        let invNormVec = SIMD8<Float>(repeating: invNorm)
        
        vector.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                for i in 0..<fullVectors {
                    let offset = i * vectorWidth
                    var v = SIMD8<Float>.zero
                    for j in 0..<vectorWidth {
                        v[j] = src[offset + j]
                    }
                    let normalized = v * invNormVec
                    for j in 0..<vectorWidth {
                        dst[offset + j] = normalized[j]
                    }
                }
                
                for i in (fullVectors * vectorWidth)..<count {
                    dst[i] = src[i] * invNorm
                }
            }
        }
        
        return result
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix-vector multiplication using SIMD
    public func matrixVectorMultiply(
        matrix: [Float],
        rows: Int,
        columns: Int,
        vector: [Float]
    ) async throws -> [Float] {
        guard vector.count == columns else {
            throw VectorError.dimensionMismatch(expected: columns, actual: vector.count)
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }
        
        if configuration.useAccelerate {
            return accelerateMatrixVector(matrix: matrix, rows: rows, columns: columns, vector: vector)
        } else {
            return simdMatrixVector(matrix: matrix, rows: rows, columns: columns, vector: vector)
        }
    }
    
    /// Accelerate-based matrix-vector multiplication
    private func accelerateMatrixVector(
        matrix: [Float],
        rows: Int,
        columns: Int,
        vector: [Float]
    ) -> [Float] {
        var result = [Float](repeating: 0, count: rows)
        
        // Use vDSP_mmul for matrix-vector multiplication (modern replacement for deprecated cblas_sgemv)
        // Treat vector as a column matrix (columns x 1)
        matrix.withUnsafeBufferPointer { matrixPtr in
            vector.withUnsafeBufferPointer { vectorPtr in
                result.withUnsafeMutableBufferPointer { resultPtr in
                    vDSP_mmul(
                        matrixPtr.baseAddress!, vDSP_Stride(1),
                        vectorPtr.baseAddress!, vDSP_Stride(1),
                        resultPtr.baseAddress!, vDSP_Stride(1),
                        vDSP_Length(rows),
                        vDSP_Length(1),
                        vDSP_Length(columns)
                    )
                }
            }
        }
        
        return result
    }
    
    /// Manual SIMD matrix-vector multiplication
    private func simdMatrixVector(
        matrix: [Float],
        rows: Int,
        columns: Int,
        vector: [Float]
    ) -> [Float] {
        let vectorWidth = configuration.vectorWidth
        
        // Use concurrent map approach to avoid mutation issues
        let indices = Array(0..<rows)
        let results = indices.map { row -> Float in
            let rowStart = row * columns
            var sum: Float = 0
            
            // Vectorized dot product for this row
            let fullVectors = columns / vectorWidth
            
            for i in 0..<fullVectors {
                let offset = i * vectorWidth
                var matrixChunk = SIMD8<Float>.zero
                var vectorChunk = SIMD8<Float>.zero
                
                for j in 0..<min(vectorWidth, columns - offset) {
                    matrixChunk[j] = matrix[rowStart + offset + j]
                    vectorChunk[j] = vector[offset + j]
                }
                
                sum += (matrixChunk * vectorChunk).sum()
            }
            
            // Handle remaining elements
            for i in (fullVectors * vectorWidth)..<columns {
                sum += matrix[rowStart + i] * vector[i]
            }
            
            return sum
        }
        
        return results
    }
    
    // MARK: - Batch Operations
    
    /// Batch distance computation using SIMD
    public func batchEuclideanDistance(
        vectors: [[Float]],
        reference: [Float]
    ) async throws -> [Float] {
        guard !vectors.isEmpty else {
            return []
        }
        
        let dimension = reference.count
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: -1)
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        defer { trackOperation(start: start) }
        
        if vectors.count < configuration.parallelThreshold {
            // Sequential processing for small batches
            return try await sequentialBatchDistance(vectors: vectors, reference: reference)
        } else {
            // Parallel processing for large batches
            return try await parallelBatchDistance(vectors: vectors, reference: reference)
        }
    }
    
    private func sequentialBatchDistance(
        vectors: [[Float]],
        reference: [Float]
    ) async throws -> [Float] {
        var results = [Float](repeating: 0, count: vectors.count)
        
        for (index, vector) in vectors.enumerated() {
            results[index] = accelerateEuclideanDistance(vector, reference)
        }
        
        return results
    }
    
    private func parallelBatchDistance(
        vectors: [[Float]],
        reference: [Float]
    ) async throws -> [Float] {
        var results = [Float](repeating: 0, count: vectors.count)
        
        await withTaskGroup(of: (Int, Float).self) { group in
            for (index, vector) in vectors.enumerated() {
                group.addTask { [self] in
                    let distance = self.accelerateEuclideanDistance(vector, reference)
                    return (index, distance)
                }
            }
            
            for await (index, distance) in group {
                results[index] = distance
            }
        }
        
        return results
    }
    
    // MARK: - Performance Tracking
    
    private func trackOperation(start: CFAbsoluteTime) {
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        lock.lock()
        operationCount += 1
        totalTime += elapsed
        lock.unlock()
    }
    
    public func getPerformanceMetrics() -> (operations: Int, averageTime: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        let avgTime = operationCount > 0 ? totalTime / Double(operationCount) : 0
        return (operationCount, avgTime)
    }
    
    public func resetMetrics() {
        lock.lock()
        operationCount = 0
        totalTime = 0
        lock.unlock()
    }
}

// MARK: - SIMD Extensions

private extension SIMD8 where Scalar == Float {
    func sum() -> Float {
        return self[0] + self[1] + self[2] + self[3] + 
               self[4] + self[5] + self[6] + self[7]
    }
}