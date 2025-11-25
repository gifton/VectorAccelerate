// VectorAccelerate: Accelerate Framework Fallback
//
// CPU-optimized operations using Apple's Accelerate framework
//

import Foundation
import Accelerate
import VectorCore

/// High-performance CPU operations using Accelerate framework
public struct AccelerateFallback {
    
    // MARK: - Distance Operations
    
    /// Compute Euclidean distance using Accelerate
    public static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.nan }
        
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return sqrt(result)
    }
    
    /// Compute cosine similarity using Accelerate
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.nan }
        
        // Compute dot product
        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        
        // Compute norms
        var normA: Float = 0
        var normB: Float = 0
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(a.count))
        
        normA = sqrt(normA)
        normB = sqrt(normB)
        
        if normA > 0 && normB > 0 {
            return dotProduct / (normA * normB)
        }
        return 0
    }
    
    /// Compute dot product using Accelerate
    public static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.nan }
        
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
    
    /// Compute Manhattan distance using Accelerate
    public static func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.nan }
        
        // Compute absolute differences
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        vDSP_vabs(diff, 1, &diff, 1, vDSP_Length(a.count))
        
        // Sum the absolute differences
        var result: Float = 0
        vDSP_sve(diff, 1, &result, vDSP_Length(a.count))
        return result
    }
    
    // MARK: - Vector Operations
    
    /// Normalize a vector using Accelerate
    public static func normalize(_ vector: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        
        guard norm > 0 else { return vector }
        
        var result = [Float](repeating: 0, count: vector.count)
        var divisor = norm
        vDSP_vsdiv(vector, 1, &divisor, &result, 1, vDSP_Length(vector.count))
        return result
    }
    
    /// Add two vectors using Accelerate
    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Subtract two vectors using Accelerate
    public static func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Scale a vector using Accelerate
    public static func scale(_ vector: [Float], by scalar: Float) -> [Float] {
        var result = [Float](repeating: 0, count: vector.count)
        var s = scalar
        vDSP_vsmul(vector, 1, &s, &result, 1, vDSP_Length(vector.count))
        return result
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix-vector multiplication using Accelerate
    public static func matrixVectorMultiply(
        matrix: [Float],
        vector: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        guard matrix.count == rows * columns && vector.count == columns else { return [] }
        
        var result = [Float](repeating: 0, count: rows)

        // Use vDSP for matrix-vector multiplication (treating vector as Nx1 matrix)
        // vDSP_mmul performs C = A * B where:
        // A is matrix (rows x columns), B is vector (columns x 1), C is result (rows x 1)
        vDSP_mmul(
            matrix,           // Input matrix A
            1,                // Stride for A
            vector,           // Input vector B (as column matrix)
            1,                // Stride for B
            &result,          // Output C
            1,                // Stride for C
            vDSP_Length(rows),     // M: rows in A (and C)
            vDSP_Length(1),        // N: columns in B (and C) = 1
            vDSP_Length(columns)   // P: columns in A = rows in B
        )

        return result
    }
    
    /// Matrix multiplication using Accelerate
    public static func matrixMultiply(
        a: [Float],
        b: [Float],
        rowsA: Int,
        colsA: Int,
        colsB: Int
    ) -> [Float] {
        let rowsB = colsA  // Must match for multiplication
        guard a.count == rowsA * colsA && b.count == rowsB * colsB else { return [] }
        
        var result = [Float](repeating: 0, count: rowsA * colsB)

        // Use vDSP for matrix multiplication
        // vDSP_mmul performs C = A * B where:
        // A is (rowsA x colsA), B is (colsA x colsB), C is (rowsA x colsB)
        vDSP_mmul(
            a,                      // Input matrix A
            1,                      // Stride for A
            b,                      // Input matrix B
            1,                      // Stride for B
            &result,                // Output matrix C
            1,                      // Stride for C
            vDSP_Length(rowsA),     // M: rows in A (and C)
            vDSP_Length(colsB),     // N: columns in B (and C)
            vDSP_Length(colsA)      // P: columns in A = rows in B
        )

        return result
    }
    
    /// Matrix transpose using Accelerate
    public static func transpose(
        matrix: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        guard matrix.count == rows * columns else { return [] }
        
        var result = [Float](repeating: 0, count: rows * columns)
        vDSP_mtrans(matrix, 1, &result, 1, vDSP_Length(columns), vDSP_Length(rows))
        return result
    }
    
    // MARK: - Batch Operations
    
    /// Batch Euclidean distance using Accelerate
    public static func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { euclideanDistance(query, $0) }
    }
    
    /// Batch cosine similarity using Accelerate
    public static func batchCosineSimilarity(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        // Pre-compute query norm
        var queryNorm: Float = 0
        vDSP_svesq(query, 1, &queryNorm, vDSP_Length(query.count))
        queryNorm = sqrt(queryNorm)
        
        guard queryNorm > 0 else {
            return [Float](repeating: 0, count: candidates.count)
        }
        
        return candidates.map { candidate in
            // Compute dot product
            var dotProduct: Float = 0
            vDSP_dotpr(query, 1, candidate, 1, &dotProduct, vDSP_Length(query.count))
            
            // Compute candidate norm
            var candidateNorm: Float = 0
            vDSP_svesq(candidate, 1, &candidateNorm, vDSP_Length(candidate.count))
            candidateNorm = sqrt(candidateNorm)
            
            if candidateNorm > 0 {
                return dotProduct / (queryNorm * candidateNorm)
            }
            return 0
        }
    }
    
    /// Batch dot product using Accelerate
    public static func batchDotProduct(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        candidates.map { dotProduct(query, $0) }
    }
    
    // MARK: - Statistical Operations
    
    /// Compute mean using Accelerate
    public static func mean(_ vector: [Float]) -> Float {
        var result: Float = 0
        vDSP_meanv(vector, 1, &result, vDSP_Length(vector.count))
        return result
    }
    
    /// Compute variance using Accelerate
    public static func variance(_ vector: [Float]) -> Float {
        let meanValue = mean(vector)
        
        // Subtract mean from each element
        var centered = [Float](repeating: 0, count: vector.count)
        var negativeMean = -meanValue
        vDSP_vsadd(vector, 1, &negativeMean, &centered, 1, vDSP_Length(vector.count))
        
        // Square the differences
        var squared = [Float](repeating: 0, count: vector.count)
        vDSP_vsq(centered, 1, &squared, 1, vDSP_Length(vector.count))
        
        // Sum and divide by count
        var sum: Float = 0
        vDSP_sve(squared, 1, &sum, vDSP_Length(vector.count))
        
        return sum / Float(vector.count)
    }
    
    /// Compute standard deviation using Accelerate
    public static func standardDeviation(_ vector: [Float]) -> Float {
        sqrt(variance(vector))
    }
    
    // MARK: - Element-wise Operations
    
    /// Element-wise multiplication using Accelerate
    public static func elementwiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vmul(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Element-wise division using Accelerate
    public static func elementwiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vdiv(b, 1, a, 1, &result, 1, vDSP_Length(a.count))
        return result
    }
    
    /// Apply function element-wise using Accelerate
    public static func applyFunction(_ vector: [Float], function: (Float) -> Float) -> [Float] {
        vector.map(function)
    }
    
    // MARK: - Performance Utilities
    
    /// Check if Accelerate should be used based on vector size
    public static func shouldUseAccelerate(for size: Int) -> Bool {
        // Accelerate is efficient for vectors larger than 32 elements
        size >= 32
    }
    
    /// Optimal chunk size for batch processing
    public static func optimalChunkSize(for totalSize: Int) -> Int {
        // Balance between cache efficiency and parallelism
        let cacheLineSize = 64  // bytes
        let floatSize = MemoryLayout<Float>.size
        let elementsPerCacheLine = cacheLineSize / floatSize
        
        // Use multiples of cache line size
        let baseChunkSize = elementsPerCacheLine * 64  // 1024 elements
        
        if totalSize <= baseChunkSize {
            return totalSize
        }
        
        // Divide into reasonable chunks
        let numChunks = (totalSize + baseChunkSize - 1) / baseChunkSize
        return totalSize / numChunks
    }
}

// MARK: - Accelerate Extensions

extension Array where Element == Float {
    /// Normalize this vector using Accelerate
    public func normalizedAccelerate() -> [Float] {
        AccelerateFallback.normalize(self)
    }
    
    /// Compute dot product with another vector using Accelerate
    public func dotAccelerate(with other: [Float]) -> Float {
        AccelerateFallback.dotProduct(self, other)
    }
    
    /// Compute Euclidean distance to another vector using Accelerate
    public func euclideanDistanceAccelerate(to other: [Float]) -> Float {
        AccelerateFallback.euclideanDistance(self, other)
    }
    
    /// Compute cosine similarity with another vector using Accelerate
    public func cosineSimilarityAccelerate(with other: [Float]) -> Float {
        AccelerateFallback.cosineSimilarity(self, other)
    }
}