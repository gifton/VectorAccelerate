//
//  MatrixEngineEnhancedTests.swift
//  VectorAccelerateTests
//
//  Comprehensive tests for MatrixEngine high-performance matrix operations
//

import XCTest
@testable import VectorAccelerate
@preconcurrency import Foundation
import VectorCore
import Accelerate
import VectorCore

@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
final class MatrixEngineEnhancedTests: XCTestCase {
    
    var engine: MatrixEngine!
    var context: MetalContext!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        context = try await MetalContext()
        engine = await MatrixEngine(context: context, configuration: .default)
    }
    
    override func tearDown() async throws {
        await context?.cleanup()
        context = nil
        engine = nil
        try await super.tearDown()
    }
    
    // MARK: - Test Utilities
    
    private static func generateMatrixStatic(rows: Int, columns: Int, pattern: MatrixPattern = .random) -> [Float] {
        switch pattern {
        case .random:
            return (0..<rows * columns).map { _ in Float.random(in: -1...1) }
        case .identity:
            precondition(rows == columns, "Identity matrix must be square")
            var matrix = [Float](repeating: 0, count: rows * columns)
            for i in 0..<rows {
                matrix[i * columns + i] = 1.0
            }
            return matrix
        case .constant(let value):
            return [Float](repeating: value, count: rows * columns)
        case .diagonal(let values):
            precondition(rows == columns && values.count == rows, "Invalid diagonal specification")
            var matrix = [Float](repeating: 0, count: rows * columns)
            for (i, value) in values.enumerated() {
                matrix[i * columns + i] = value
            }
            return matrix
        case .sequential:
            return (0..<rows * columns).map { Float($0) + 1 }
        case .alternating:
            return (0..<rows * columns).map { $0 % 2 == 0 ? 1.0 : -1.0 }
        }
    }
    
    private func generateMatrix(rows: Int, columns: Int, pattern: MatrixPattern = .random) -> [Float] {
        Self.generateMatrixStatic(rows: rows, columns: columns, pattern: pattern)
    }
    
    private enum MatrixPattern {
        case random
        case identity
        case constant(Float)
        case diagonal([Float])
        case sequential
        case alternating
    }
    
    private func matrixEquals(_ a: [Float], _ b: [Float], accuracy: Float = 1e-5) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { abs($0 - $1) <= accuracy }
    }
    
    private func printMatrix(_ matrix: [Float], rows: Int, columns: Int, label: String = "") {
        if !label.isEmpty {
            print("\(label):")
        }
        for row in 0..<min(rows, 4) { // Print only first 4 rows
            let rowStart = row * columns
            let rowEnd = min(rowStart + columns, matrix.count)
            let rowData = Array(matrix[rowStart..<rowEnd])
            let rowString = rowData.prefix(min(4, columns)).map { String(format: "%8.3f", $0) }.joined(separator: " ")
            print("  [\(rowString)\(columns > 4 ? " ..." : "")]")
        }
        if rows > 4 {
            print("  ...")
        }
    }
    
    private func referenceMatrixMultiply(_ a: [Float], rowsA: Int, colsA: Int,
                                       _ b: [Float], rowsB: Int, colsB: Int) -> [Float] {
        precondition(colsA == rowsB, "Matrix dimensions must be compatible")
        
        var result = [Float](repeating: 0, count: rowsA * colsB)
        
        for i in 0..<rowsA {
            for j in 0..<colsB {
                var sum: Float = 0
                for k in 0..<colsA {
                    sum += a[i * colsA + k] * b[k * colsB + j]
                }
                result[i * colsB + j] = sum
            }
        }
        
        return result
    }
    
    private func referenceTranspose(_ matrix: [Float], rows: Int, columns: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows * columns)
        
        for i in 0..<rows {
            for j in 0..<columns {
                result[j * rows + i] = matrix[i * columns + j]
            }
        }
        
        return result
    }
    
    // MARK: - Initialization Tests
    
    func testInitializationWithDifferentConfigurations() async throws {
        let configurations = [
            MatrixConfiguration(),
            MatrixConfiguration.default,
            MatrixConfiguration.performance,
            MatrixConfiguration(useTiledMultiplication: false, tileSize: 16),
            MatrixConfiguration(preferGPUThreshold: 64, enableAsyncExecution: true)
        ]
        
        for (idx, config) in configurations.enumerated() {
            let testEngine = await MatrixEngine(context: context, configuration: config)
            XCTAssertNotNil(testEngine, "Failed to initialize with configuration \(idx)")
            
            // Test basic operation
            let matrixA = generateMatrix(rows: 4, columns: 4, pattern: .identity)
            let matrixB = generateMatrix(rows: 4, columns: 4, pattern: .constant(2.0))
            
            let result = try await testEngine.multiply(
                matrixA, descriptorA: MatrixDescriptor(rows: 4, columns: 4),
                matrixB, descriptorB: MatrixDescriptor(rows: 4, columns: 4)
            )
            
            XCTAssertEqual(result.count, 16)
        }
    }
    
    func testCreateDefaultEngine() async throws {
        let defaultEngine = try await MatrixEngine.createDefault()
        XCTAssertNotNil(defaultEngine)
        
        // Test basic functionality
        let matrix = generateMatrix(rows: 2, columns: 2, pattern: .identity)
        let transposed = try await defaultEngine.transpose(
            matrix,
            descriptor: MatrixDescriptor(rows: 2, columns: 2)
        )
        
        XCTAssertEqual(transposed.count, 4)
    }
    
    // MARK: - Matrix Multiplication Tests
    
    func testBasicMatrixMultiplication() async throws {
        // Test 3x2 * 2x4 = 3x4
        let matrixA: [Float] = [
            1, 2,
            3, 4,
            5, 6
        ]
        let matrixB: [Float] = [
            7, 8, 9, 10,
            11, 12, 13, 14
        ]
        
        let descriptorA = MatrixDescriptor(rows: 3, columns: 2)
        let descriptorB = MatrixDescriptor(rows: 2, columns: 4)
        
        let result = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        
        // Expected result: 3x4 matrix
        // [1*7+2*11, 1*8+2*12, 1*9+2*13, 1*10+2*14]  = [29, 32, 35, 38]
        // [3*7+4*11, 3*8+4*12, 3*9+4*13, 3*10+4*14]  = [65, 72, 79, 86]
        // [5*7+6*11, 5*8+6*12, 5*9+6*13, 5*10+6*14]  = [101, 112, 123, 134]
        let expected: [Float] = [
            29, 32, 35, 38,
            65, 72, 79, 86,
            101, 112, 123, 134
        ]
        
        XCTAssertEqual(result.count, 12)
        XCTAssertTrue(matrixEquals(result, expected), 
                     "Matrix multiplication result incorrect")
        
        printMatrix(result, rows: 3, columns: 4, label: "Result")
        printMatrix(expected, rows: 3, columns: 4, label: "Expected")
    }
    
    func testIdentityMatrixMultiplication() async throws {
        let size = 4
        let identity = generateMatrix(rows: size, columns: size, pattern: .identity)
        let testMatrix = generateMatrix(rows: size, columns: size, pattern: .sequential)
        
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // I * A should equal A
        let result1 = try await engine!.multiply(identity, descriptorA: descriptor, testMatrix, descriptorB: descriptor)
        XCTAssertTrue(matrixEquals(result1, testMatrix), "I * A should equal A")
        
        // A * I should equal A
        let result2 = try await engine!.multiply(testMatrix, descriptorA: descriptor, identity, descriptorB: descriptor)
        XCTAssertTrue(matrixEquals(result2, testMatrix), "A * I should equal A")
    }
    
    func testMatrixMultiplicationDifferentSizes() async throws {
        let testCases = [
            (2, 3, 4), // 2x3 * 3x4
            (1, 5, 1), // 1x5 * 5x1 (vector multiplication)
            (8, 8, 8), // 8x8 * 8x8 (larger square)
            (10, 7, 15), // 10x7 * 7x15 (non-square)
        ]
        
        for (m, k, n) in testCases {
            let matrixA = generateMatrix(rows: m, columns: k, pattern: .random)
            let matrixB = generateMatrix(rows: k, columns: n, pattern: .random)
            
            let descriptorA = MatrixDescriptor(rows: m, columns: k)
            let descriptorB = MatrixDescriptor(rows: k, columns: n)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            XCTAssertEqual(result.count, m * n)
            
            // Compare with reference implementation
            let reference = referenceMatrixMultiply(matrixA, rowsA: m, colsA: k, matrixB, rowsB: k, colsB: n)
            XCTAssertTrue(matrixEquals(result, reference, accuracy: 1e-4),
                         "Matrix multiplication \(m)x\(k) * \(k)x\(n) failed")
            
            print("Matrix multiplication \(m)x\(k) * \(k)x\(n): \(elapsed * 1000)ms")
        }
    }
    
    func testMatrixMultiplicationGPUvsCPUThreshold() async throws {
        let smallSize = 8 // Below GPU threshold
        let largeSize = 64 // Above GPU threshold
        
        // Small matrix (should use CPU)
        let smallA = generateMatrix(rows: smallSize, columns: smallSize)
        let smallB = generateMatrix(rows: smallSize, columns: smallSize)
        let smallDescriptor = MatrixDescriptor(rows: smallSize, columns: smallSize)
        
        let smallResult = try await engine!.multiply(smallA, descriptorA: smallDescriptor, smallB, descriptorB: smallDescriptor)
        XCTAssertEqual(smallResult.count, smallSize * smallSize)
        
        // Large matrix (should use GPU)
        let largeA = generateMatrix(rows: largeSize, columns: largeSize)
        let largeB = generateMatrix(rows: largeSize, columns: largeSize)
        let largeDescriptor = MatrixDescriptor(rows: largeSize, columns: largeSize)
        
        let largeStart = CFAbsoluteTimeGetCurrent()
        let largeResult = try await engine!.multiply(largeA, descriptorA: largeDescriptor, largeB, descriptorB: largeDescriptor)
        let largeTime = CFAbsoluteTimeGetCurrent() - largeStart
        
        XCTAssertEqual(largeResult.count, largeSize * largeSize)
        
        print("Large matrix (\(largeSize)x\(largeSize)) multiplication time: \(largeTime * 1000)ms")
        
        // Verify correctness with reference
        let reference = referenceMatrixMultiply(largeA, rowsA: largeSize, colsA: largeSize, largeB, rowsB: largeSize, colsB: largeSize)
        XCTAssertTrue(matrixEquals(largeResult, reference, accuracy: 1e-3))
    }
    
    func testMatrixMultiplicationDimensionMismatch() async throws {
        let matrixA = generateMatrix(rows: 3, columns: 2)
        let matrixB = generateMatrix(rows: 4, columns: 3) // Wrong dimensions
        
        let descriptorA = MatrixDescriptor(rows: 3, columns: 2)
        let descriptorB = MatrixDescriptor(rows: 4, columns: 3)
        
        do {
            _ = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Dimension mismatch verified by kind check
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
    
    // MARK: - Matrix Transpose Tests
    
    func testBasicTranspose() async throws {
        // 3x2 matrix
        let matrix: [Float] = [
            1, 2,
            3, 4,
            5, 6
        ]
        
        let descriptor = MatrixDescriptor(rows: 3, columns: 2)
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        // Expected 2x3 matrix (transposed)
        let expected: [Float] = [
            1, 3, 5,
            2, 4, 6
        ]
        
        XCTAssertEqual(result.count, 6)
        XCTAssertTrue(matrixEquals(result, expected), "Transpose result incorrect")
        
        printMatrix(matrix, rows: 3, columns: 2, label: "Original")
        printMatrix(result, rows: 2, columns: 3, label: "Transposed")
    }
    
    func testSquareMatrixTranspose() async throws {
        let size = 4
        let matrix = generateMatrix(rows: size, columns: size, pattern: .sequential)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        // Compare with reference implementation
        let reference = referenceTranspose(matrix, rows: size, columns: size)
        XCTAssertTrue(matrixEquals(result, reference), "Square matrix transpose failed")
        
        // Verify transpose property: (A^T)^T = A
        let doubleTranspose = try await engine!.transpose(result, descriptor: MatrixDescriptor(rows: size, columns: size))
        XCTAssertTrue(matrixEquals(doubleTranspose, matrix), "Double transpose should equal original")
    }
    
    func testTransposeDifferentSizes() async throws {
        let testCases = [
            (1, 10), // Row vector to column vector
            (10, 1), // Column vector to row vector
            (5, 3),  // Non-square
            (16, 16), // Large square (should use GPU)
            (100, 50), // Large non-square
        ]
        
        for (rows, cols) in testCases {
            let matrix = generateMatrix(rows: rows, columns: cols, pattern: .random)
            let descriptor = MatrixDescriptor(rows: rows, columns: cols)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await engine!.transpose(matrix, descriptor: descriptor)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            XCTAssertEqual(result.count, rows * cols)
            
            // Compare with reference
            let reference = referenceTranspose(matrix, rows: rows, columns: cols)
            XCTAssertTrue(matrixEquals(result, reference, accuracy: 1e-5),
                         "Transpose \(rows)x\(cols) failed")
            
            print("Transpose \(rows)x\(cols): \(elapsed * 1000)ms")
        }
    }
    
    func testTransposeSpecialMatrices() async throws {
        let size = 4
        
        // Test identity matrix transpose (should equal itself)
        let identity = generateMatrix(rows: size, columns: size, pattern: .identity)
        let identityDescriptor = MatrixDescriptor(rows: size, columns: size)
        let identityTranspose = try await engine!.transpose(identity, descriptor: identityDescriptor)
        XCTAssertTrue(matrixEquals(identityTranspose, identity), "Identity transpose should equal itself")
        
        // Test diagonal matrix
        let diagonal = generateMatrix(rows: size, columns: size, pattern: .diagonal([1, 2, 3, 4]))
        let diagonalTranspose = try await engine!.transpose(diagonal, descriptor: identityDescriptor)
        XCTAssertTrue(matrixEquals(diagonalTranspose, diagonal), "Diagonal transpose should equal itself")
        
        // Test zero matrix
        let zero = generateMatrix(rows: size, columns: size, pattern: .constant(0))
        let zeroTranspose = try await engine!.transpose(zero, descriptor: identityDescriptor)
        XCTAssertTrue(matrixEquals(zeroTranspose, zero), "Zero transpose should equal itself")
    }
    
    // MARK: - Batch Operations Tests
    
    func testBatchMatrixVectorMultiply() async throws {
        let matrixRows = 4
        let matrixCols = 3
        let batchSize = 5
        
        let matrices = (0..<batchSize).map { _ in
            generateMatrix(rows: matrixRows, columns: matrixCols, pattern: .random)
        }
        let vectors = (0..<batchSize).map { _ in
            generateMatrix(rows: matrixCols, columns: 1, pattern: .random)
        }
        
        let descriptor = MatrixDescriptor(rows: matrixRows, columns: matrixCols)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let results = try await engine!.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        XCTAssertEqual(results.count, batchSize)
        
        // Verify each result
        for i in 0..<batchSize {
            XCTAssertEqual(results[i].count, matrixRows)
            
            // Compare with reference single matrix-vector multiplication
            let matrix = matrices[i]
            let vector = vectors[i]
            let reference = referenceMatrixMultiply(matrix, rowsA: matrixRows, colsA: matrixCols, 
                                                  vector, rowsB: matrixCols, colsB: 1)
            
            XCTAssertTrue(matrixEquals(results[i], reference, accuracy: 1e-4),
                         "Batch result \(i) incorrect")
        }
        
        print("Batch matrix-vector multiply (\(batchSize) operations): \(elapsed * 1000)ms")
        print("Average per operation: \(elapsed * 1000 / Double(batchSize))ms")
    }
    
    func testBatchMatrixVectorDimensionMismatch() async throws {
        let matrices = [generateMatrix(rows: 3, columns: 3)]
        let vectors = [generateMatrix(rows: 2, columns: 1)] // Wrong dimension
        let descriptor = MatrixDescriptor(rows: 3, columns: 3)
        
        do {
            _ = try await engine!.batchMatrixVectorMultiply(
                matrices: matrices,
                descriptor: descriptor,
                vectors: vectors
            )
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected error
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
    
    func testBatchMatrixVectorCountMismatch() async throws {
        let matrices = [generateMatrix(rows: 3, columns: 3)]
        let vectors = [
            generateMatrix(rows: 3, columns: 1),
            generateMatrix(rows: 3, columns: 1)
        ] // Different count
        let descriptor = MatrixDescriptor(rows: 3, columns: 3)
        
        do {
            _ = try await engine!.batchMatrixVectorMultiply(
                matrices: matrices,
                descriptor: descriptor,
                vectors: vectors
            )
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Expected error
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
    
    func testBatchOperationPerformanceComparison() async throws {
        let matrixRows = 8
        let matrixCols = 6
        let batchSize = 20
        
        let matrices = (0..<batchSize).map { _ in
            generateMatrix(rows: matrixRows, columns: matrixCols, pattern: .random)
        }
        let vectors = (0..<batchSize).map { _ in
            generateMatrix(rows: matrixCols, columns: 1, pattern: .random)
        }
        let descriptor = MatrixDescriptor(rows: matrixRows, columns: matrixCols)
        
        // Sequential operations
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        var sequentialResults: [[Float]] = []
        for i in 0..<batchSize {
            // Simulate single matrix-vector multiply using matrix multiply
            let vectorDescriptor = MatrixDescriptor(rows: matrixCols, columns: 1)
            let result = try await engine!.multiply(
                matrices[i], descriptorA: descriptor,
                vectors[i], descriptorB: vectorDescriptor
            )
            sequentialResults.append(result)
        }
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStart
        
        // Batch operation
        let batchStart = CFAbsoluteTimeGetCurrent()
        let batchResults = try await engine!.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        print("Performance comparison (\(batchSize) operations):")
        print("  Sequential: \(sequentialTime * 1000)ms")
        print("  Batch: \(batchTime * 1000)ms")
        print("  Speedup: \(sequentialTime / batchTime)x")
        
        // Results should be equivalent
        XCTAssertEqual(sequentialResults.count, batchResults.count)
        for (sequential, batch) in zip(sequentialResults, batchResults) {
            XCTAssertTrue(matrixEquals(sequential, batch, accuracy: 1e-4),
                         "Sequential and batch results should be equivalent")
        }
        
        // Batch should be at least as fast (though not guaranteed for small operations)
        if batchSize >= 10 {
            print("Batch processing utilized")
        }
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceScaling() async throws {
        let sizes = [16, 32, 64, 128]
        
        for size in sizes {
            let matrixA = generateMatrix(rows: size, columns: size)
            let matrixB = generateMatrix(rows: size, columns: size)
            let descriptor = MatrixDescriptor(rows: size, columns: size)
            
            // Matrix multiplication performance
            let multiplyStart = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
            let multiplyTime = CFAbsoluteTimeGetCurrent() - multiplyStart
            
            // Transpose performance
            let transposeStart = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.transpose(matrixA, descriptor: descriptor)
            let transposeTime = CFAbsoluteTimeGetCurrent() - transposeStart
            
            let operations = Double(size * size * size) // O(n³) for multiplication
            let throughput = operations / multiplyTime
            
            print("Performance for \(size)x\(size) matrices:")
            print("  Multiplication: \(multiplyTime * 1000)ms (\(throughput / 1e6) MFLOPS)")
            print("  Transpose: \(transposeTime * 1000)ms")
            
            // Performance should be reasonable
            XCTAssertLessThan(multiplyTime, Double(size) * 0.01, "Multiplication should be efficient")
            XCTAssertLessThan(transposeTime, 0.1, "Transpose should be fast")
        }
    }
    
    func testMemoryUsageMonitoring() async throws {
        let size = 100
        let numOperations = 10
        
        let matrixA = generateMatrix(rows: size, columns: size)
        let matrixB = generateMatrix(rows: size, columns: size)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // Perform multiple operations and monitor performance
        for i in 0..<numOperations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            
            print("Operation \(i + 1): \(elapsed * 1000)ms")
            
            // Performance should remain consistent (no memory leaks)
            XCTAssertLessThan(elapsed, 1.0, "Performance should remain consistent")
        }
        
        let metrics = await engine!.getPerformanceMetrics()
        print("Engine metrics: \(metrics.operations) operations, avg time: \(metrics.averageTime * 1000)ms")
        
        XCTAssertEqual(metrics.operations, numOperations)
        XCTAssertGreaterThan(metrics.averageTime, 0)
    }
    
    // MARK: - Numerical Accuracy Tests
    
    func testNumericalAccuracy() async throws {
        // Test with known mathematical properties
        
        // 1. Associativity: (A * B) * C = A * (B * C)
        let size = 4
        let matrixA = generateMatrix(rows: size, columns: size, pattern: .random)
        let matrixB = generateMatrix(rows: size, columns: size, pattern: .random)
        let matrixC = generateMatrix(rows: size, columns: size, pattern: .random)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // Compute (A * B) * C
        let ab = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
        let abc1 = try await engine!.multiply(ab, descriptorA: descriptor, matrixC, descriptorB: descriptor)
        
        // Compute A * (B * C)
        let bc = try await engine!.multiply(matrixB, descriptorA: descriptor, matrixC, descriptorB: descriptor)
        let abc2 = try await engine!.multiply(matrixA, descriptorA: descriptor, bc, descriptorB: descriptor)
        
        XCTAssertTrue(matrixEquals(abc1, abc2, accuracy: 1e-3), 
                     "Matrix multiplication should be associative")
        
        // 2. Transpose property: (A * B)^T = B^T * A^T
        let ab_transpose = try await engine!.transpose(ab, descriptor: descriptor)
        
        let a_transpose = try await engine!.transpose(matrixA, descriptor: descriptor)
        let b_transpose = try await engine!.transpose(matrixB, descriptor: descriptor)
        let bt_at = try await engine!.multiply(b_transpose, descriptorA: descriptor, a_transpose, descriptorB: descriptor)
        
        XCTAssertTrue(matrixEquals(ab_transpose, bt_at, accuracy: 1e-3),
                     "Transpose property should hold: (AB)^T = B^T A^T")
    }
    
    func testPrecisionWithLargeNumbers() async throws {
        // Test with large numbers that might cause precision issues
        let size = 4
        let largeMatrix = generateMatrix(rows: size, columns: size, pattern: .constant(1e6))
        let smallMatrix = generateMatrix(rows: size, columns: size, pattern: .constant(1e-6))
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        let result = try await engine!.multiply(largeMatrix, descriptorA: descriptor, smallMatrix, descriptorB: descriptor)
        
        // Result should be approximately 1e6 * 1e-6 * size = size
        let expected = generateMatrix(rows: size, columns: size, pattern: .constant(Float(size)))
        XCTAssertTrue(matrixEquals(result, expected, accuracy: 1e-3),
                     "Should handle large number ranges correctly")
    }
    
    // MARK: - Edge Cases and Error Handling
    
    func testEmptyMatrixOperations() async throws {
        // Test with zero-dimension matrices
        let emptyMatrix: [Float] = []
        let emptyDescriptor = MatrixDescriptor(rows: 0, columns: 0)
        
        // Empty matrix multiplication
        let emptyResult = try await engine!.multiply(
            emptyMatrix, descriptorA: emptyDescriptor,
            emptyMatrix, descriptorB: emptyDescriptor
        )
        XCTAssertEqual(emptyResult.count, 0)
        
        // Empty matrix transpose
        let emptyTranspose = try await engine!.transpose(emptyMatrix, descriptor: emptyDescriptor)
        XCTAssertEqual(emptyTranspose.count, 0)
    }
    
    func testSingleElementMatrix() async throws {
        let singleElement: [Float] = [5.0]
        let descriptor = MatrixDescriptor(rows: 1, columns: 1)
        
        // 1x1 matrix multiplication
        let result = try await engine!.multiply(singleElement, descriptorA: descriptor, singleElement, descriptorB: descriptor)
        XCTAssertEqual(result, [25.0])
        
        // 1x1 matrix transpose
        let transpose = try await engine!.transpose(singleElement, descriptor: descriptor)
        XCTAssertEqual(transpose, singleElement)
    }
    
    func testVectorOperations() async throws {
        // Row vector (1x4)
        let rowVector: [Float] = [1, 2, 3, 4]
        let rowDescriptor = MatrixDescriptor(rows: 1, columns: 4)
        
        // Column vector (4x1)
        let colVector: [Float] = [5, 6, 7, 8]
        let colDescriptor = MatrixDescriptor(rows: 4, columns: 1)
        
        // Inner product (1x4 * 4x1 = 1x1)
        let innerProduct = try await engine!.multiply(rowVector, descriptorA: rowDescriptor, colVector, descriptorB: colDescriptor)
        XCTAssertEqual(innerProduct.count, 1)
        let expectedSum: Float = Float(1*5) + Float(2*6) + Float(3*7) + Float(4*8) // = 70
        XCTAssertEqual(innerProduct[0], expectedSum)
        
        // Outer product (4x1 * 1x4 = 4x4)
        let outerProduct = try await engine!.multiply(colVector, descriptorA: colDescriptor, rowVector, descriptorB: rowDescriptor)
        XCTAssertEqual(outerProduct.count, 16)
    }
    
    func testSpecialValues() async throws {
        let size = 3
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // Test with matrices containing special values
        let infinityMatrix = generateMatrix(rows: size, columns: size, pattern: .constant(Float.infinity))
        let zeroMatrix = generateMatrix(rows: size, columns: size, pattern: .constant(0))
        
        // Infinity * 0 should be handled gracefully (result may be NaN)
        let result = try await engine!.multiply(infinityMatrix, descriptorA: descriptor, zeroMatrix, descriptorB: descriptor)
        XCTAssertEqual(result.count, size * size)
        // Don't assert on specific values as they may be NaN or implementation-dependent
        
        // Very small numbers
        let tinyMatrix = generateMatrix(rows: size, columns: size, pattern: .constant(Float.leastNormalMagnitude))
        let tinyResult = try await engine!.multiply(tinyMatrix, descriptorA: descriptor, tinyMatrix, descriptorB: descriptor)
        XCTAssertEqual(tinyResult.count, size * size)
        
        // All results should be finite (no crashes)
        for value in tinyResult {
            XCTAssertTrue(value.isFinite || value.isZero, "Result should be finite or zero")
        }
    }
    
    // MARK: - Configuration-Specific Tests
    
    func testTiledMultiplication() async throws {
        let tiledConfig = MatrixConfiguration(useTiledMultiplication: true, tileSize: 32)
        let tiledEngine = await MatrixEngine(context: context, configuration: tiledConfig)
        
        let nonTiledConfig = MatrixConfiguration(useTiledMultiplication: false)
        let nonTiledEngine = await MatrixEngine(context: context, configuration: nonTiledConfig)
        
        let size = 64
        let matrixA = generateMatrix(rows: size, columns: size, pattern: .random)
        let matrixB = generateMatrix(rows: size, columns: size, pattern: .random)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // Both engines should produce the same result
        let tiledResult = try await tiledEngine.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
        let nonTiledResult = try await nonTiledEngine.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
        
        XCTAssertTrue(matrixEquals(tiledResult, nonTiledResult, accuracy: 1e-4),
                     "Tiled and non-tiled multiplication should produce same results")
    }
    
    func testAsyncExecution() async throws {
        let asyncConfig = MatrixConfiguration(enableAsyncExecution: true)
        let asyncEngine = await MatrixEngine(context: context, configuration: asyncConfig)
        
        let size = 32
        let matrices = (0..<5).map { _ in generateMatrix(rows: size, columns: size) }
        let vectors = (0..<5).map { _ in generateMatrix(rows: size, columns: 1) }
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        let start = CFAbsoluteTimeGetCurrent()
        let results = try await asyncEngine.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        XCTAssertEqual(results.count, 5)
        for result in results {
            XCTAssertEqual(result.count, size)
        }
        
        print("Async batch execution: \(elapsed * 1000)ms")
    }
    
    // MARK: - Concurrent Operations Tests
    
    func testConcurrentMatrixOperations() async throws {
        let numTasks = 10
        let size = 16
        
        await withTaskGroup(of: (Int, Bool).self) { group in
            for i in 0..<numTasks {
                group.addTask { [engine] in
                    do {
                        let matrixA = MatrixEngineEnhancedTests.generateMatrixStatic(rows: size, columns: size, pattern: .random)
                        let matrixB = MatrixEngineEnhancedTests.generateMatrixStatic(rows: size, columns: size, pattern: .random)
                        let descriptor = MatrixDescriptor(rows: size, columns: size)
                        
                        let result = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
                        let transpose = try await engine!.transpose(result, descriptor: descriptor)
                        
                        return (i, result.count == size * size && transpose.count == size * size)
                    } catch {
                        return (i, false)
                    }
                }
            }
            
            var completedTasks = 0
            var successfulTasks = 0
            
            for await (_, success) in group {
                completedTasks += 1
                if success {
                    successfulTasks += 1
                }
            }
            
            XCTAssertEqual(completedTasks, numTasks, "All tasks should complete")
            XCTAssertEqual(successfulTasks, numTasks, "All tasks should succeed")
        }
        
        // Engine should still be functional
        let testMatrix = generateMatrix(rows: 4, columns: 4)
        let testDescriptor = MatrixDescriptor(rows: 4, columns: 4)
        let finalResult = try await engine!.transpose(testMatrix, descriptor: testDescriptor)
        XCTAssertEqual(finalResult.count, 16)
    }
}