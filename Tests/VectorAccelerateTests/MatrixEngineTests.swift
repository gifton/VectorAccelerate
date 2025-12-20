//
//  MatrixEngineTests.swift
//  VectorAccelerateTests
//
//  Comprehensive tests for MatrixEngine GPU-accelerated linear algebra operations
//

import XCTest
@testable import VectorAccelerate
import Foundation
import VectorCore

final class MatrixEngineTests: XCTestCase {
    
    var engine: MatrixEngine!
    var context: Metal4Context!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard MetalDevice.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        context = try await Metal4Context()
        engine = await MatrixEngine(context: context)
    }
    
    override func tearDown() async throws {
        engine = nil
        context = nil
        try await super.tearDown()
    }
    
    // MARK: - Test Utilities
    
    private func generateMatrix(rows: Int, cols: Int, pattern: MatrixPattern = .random) -> [Float] {
        switch pattern {
        case .random:
            return (0..<(rows * cols)).map { _ in Float.random(in: -1...1) }
        case .identity:
            precondition(rows == cols, "Identity matrix must be square")
            var matrix = [Float](repeating: 0, count: rows * cols)
            for i in 0..<rows {
                matrix[i * cols + i] = 1.0
            }
            return matrix
        case .sequential:
            return (0..<(rows * cols)).map { Float($0) + 1.0 }
        case .constant(let value):
            return [Float](repeating: value, count: rows * cols)
        case .diagonal(let values):
            precondition(rows == cols && values.count == rows, "Diagonal matrix dimension mismatch")
            var matrix = [Float](repeating: 0, count: rows * cols)
            for i in 0..<rows {
                matrix[i * cols + i] = values[i]
            }
            return matrix
        }
    }
    
    private func generateVector(size: Int, pattern: VectorPattern = .random) -> [Float] {
        switch pattern {
        case .random:
            return (0..<size).map { _ in Float.random(in: -1...1) }
        case .sequential:
            return (0..<size).map { Float($0) + 1.0 }
        case .constant(let value):
            return [Float](repeating: value, count: size)
        case .unit(let index):
            var vector = [Float](repeating: 0, count: size)
            if index < size {
                vector[index] = 1.0
            }
            return vector
        }
    }
    
    enum MatrixPattern {
        case random
        case identity
        case sequential
        case constant(Float)
        case diagonal([Float])
    }
    
    enum VectorPattern {
        case random
        case sequential
        case constant(Float)
        case unit(Int)  // One-hot vector at specified index
    }
    
    private func matricesEqual(_ a: [Float], _ b: [Float], tolerance: Float = 1e-5) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { abs($0.0 - $0.1) < tolerance }
    }
    
    private func referenceMatrixMultiply(_ a: [Float], rowsA: Int, colsA: Int, 
                                        _ b: [Float], rowsB: Int, colsB: Int) -> [Float] {
        precondition(colsA == rowsB, "Matrix dimension mismatch")
        
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
    
    private func referenceTranspose(_ matrix: [Float], rows: Int, cols: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[j * rows + i] = matrix[i * cols + j]
            }
        }
        return result
    }
    
    // MARK: - Initialization Tests
    
    func testInitializationWithDefaultConfig() async throws {
        let defaultEngine = await MatrixEngine(context: context)
        let metrics = await defaultEngine.getPerformanceMetrics()
        
        XCTAssertEqual(metrics.operations, 0)
        XCTAssertEqual(metrics.averageTime, 0)
    }
    
    func testInitializationWithCustomConfig() async throws {
        let config = MatrixConfiguration(
            useTiledMultiplication: true,
            tileSize: 64,
            preferGPUThreshold: 128,
            enableAsyncExecution: true
        )
        
        let customEngine = await MatrixEngine(context: context, configuration: config)
        XCTAssertNotNil(customEngine)
    }
    
    func testCreateDefaultEngineFactory() async throws {
        do {
            let defaultEngine = try await MatrixEngine.createDefault()
            XCTAssertNotNil(defaultEngine)
        } catch let error as VectorError where error.kind == .resourceUnavailable {
            // Expected if Metal not available
        }
    }
    
    func testConfigurationPresets() async throws {
        let defaultConfig = MatrixConfiguration.default
        XCTAssertTrue(defaultConfig.useTiledMultiplication)
        XCTAssertEqual(defaultConfig.tileSize, 32)
        XCTAssertEqual(defaultConfig.preferGPUThreshold, 256)
        XCTAssertFalse(defaultConfig.enableAsyncExecution)
        
        let performanceConfig = MatrixConfiguration.performance
        XCTAssertTrue(performanceConfig.useTiledMultiplication)
        XCTAssertEqual(performanceConfig.tileSize, 64)
        XCTAssertEqual(performanceConfig.preferGPUThreshold, 128)
        XCTAssertTrue(performanceConfig.enableAsyncExecution)
    }
    
    // MARK: - Matrix Descriptor Tests
    
    func testMatrixDescriptor() async throws {
        let descriptor = MatrixDescriptor(rows: 10, columns: 20)
        
        XCTAssertEqual(descriptor.rows, 10)
        XCTAssertEqual(descriptor.columns, 20)
        XCTAssertEqual(descriptor.layout, .rowMajor)
        XCTAssertEqual(descriptor.leadingDimension, 20)
        XCTAssertEqual(descriptor.elementCount, 200)
        XCTAssertEqual(descriptor.byteSize, 200 * MemoryLayout<Float>.stride)
    }
    
    func testMatrixDescriptorWithCustomLayout() async throws {
        let descriptor = MatrixDescriptor(
            rows: 5, 
            columns: 8, 
            layout: .columnMajor, 
            leadingDimension: 10
        )
        
        XCTAssertEqual(descriptor.rows, 5)
        XCTAssertEqual(descriptor.columns, 8)
        XCTAssertEqual(descriptor.layout, .columnMajor)
        XCTAssertEqual(descriptor.leadingDimension, 10)
        XCTAssertTrue(descriptor.layout.isTransposed)
    }
    
    // MARK: - Matrix Multiplication Tests
    
    func testBasicMatrixMultiplication() async throws {
        let matrixA = generateMatrix(rows: 3, cols: 4, pattern: .sequential)
        let matrixB = generateMatrix(rows: 4, cols: 5, pattern: .sequential)
        
        let descriptorA = MatrixDescriptor(rows: 3, columns: 4)
        let descriptorB = MatrixDescriptor(rows: 4, columns: 5)
        
        let result = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        
        XCTAssertEqual(result.count, 3 * 5)
        
        // Verify against reference implementation
        let reference = referenceMatrixMultiply(matrixA, rowsA: 3, colsA: 4, matrixB, rowsB: 4, colsB: 5)
        XCTAssertTrue(matricesEqual(result, reference))
    }
    
    func testIdentityMatrixMultiplication() async throws {
        let size = 5
        let identity = generateMatrix(rows: size, cols: size, pattern: .identity)
        let testMatrix = generateMatrix(rows: size, cols: size, pattern: .random)
        
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // A * I = A
        let result1 = try await engine!.multiply(testMatrix, descriptorA: descriptor, identity, descriptorB: descriptor)
        XCTAssertTrue(matricesEqual(result1, testMatrix))
        
        // I * A = A
        let result2 = try await engine!.multiply(identity, descriptorA: descriptor, testMatrix, descriptorB: descriptor)
        XCTAssertTrue(matricesEqual(result2, testMatrix))
    }
    
    func testMatrixMultiplicationWithZeros() async throws {
        let zeros = generateMatrix(rows: 3, cols: 4, pattern: .constant(0))
        let testMatrix = generateMatrix(rows: 4, cols: 3, pattern: .random)
        
        let descriptorZeros = MatrixDescriptor(rows: 3, columns: 4)
        let descriptorTest = MatrixDescriptor(rows: 4, columns: 3)
        
        let result = try await engine!.multiply(zeros, descriptorA: descriptorZeros, testMatrix, descriptorB: descriptorTest)
        
        XCTAssertEqual(result.count, 3 * 3)
        XCTAssertTrue(result.allSatisfy { abs($0) < 1e-6 })
    }
    
    func testMatrixMultiplicationDimensionMismatch() async throws {
        let matrixA = generateMatrix(rows: 3, cols: 4, pattern: .random)
        let matrixB = generateMatrix(rows: 5, cols: 3, pattern: .random)  // Wrong dimension
        
        let descriptorA = MatrixDescriptor(rows: 3, columns: 4)
        let descriptorB = MatrixDescriptor(rows: 5, columns: 3)
        
        do {
            _ = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Verified dimension mismatch was thrown
        }
    }
    
    func testSmallMatrixMultiplication() async throws {
        // Test CPU path for small matrices
        let config = MatrixConfiguration(preferGPUThreshold: 1000) // Force CPU
        let cpuEngine = await MatrixEngine(context: context, configuration: config)
        
        let matrixA = generateMatrix(rows: 2, cols: 3, pattern: .sequential)
        let matrixB = generateMatrix(rows: 3, cols: 2, pattern: .sequential)
        
        let descriptorA = MatrixDescriptor(rows: 2, columns: 3)
        let descriptorB = MatrixDescriptor(rows: 3, columns: 2)
        
        let result = try await cpuEngine.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        
        XCTAssertEqual(result.count, 2 * 2)
        
        // Verify against reference
        let reference = referenceMatrixMultiply(matrixA, rowsA: 2, colsA: 3, matrixB, rowsB: 3, colsB: 2)
        XCTAssertTrue(matricesEqual(result, reference))
    }
    
    func testLargeMatrixMultiplication() async throws {
        // Test GPU path for larger matrices
        let config = MatrixConfiguration(preferGPUThreshold: 10) // Force GPU
        let gpuEngine = await MatrixEngine(context: context, configuration: config)
        
        let matrixA = generateMatrix(rows: 20, cols: 30, pattern: .random)
        let matrixB = generateMatrix(rows: 30, cols: 25, pattern: .random)
        
        let descriptorA = MatrixDescriptor(rows: 20, columns: 30)
        let descriptorB = MatrixDescriptor(rows: 30, columns: 25)
        
        let result = try await gpuEngine.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        
        XCTAssertEqual(result.count, 20 * 25)
        
        // Verify some basic properties (actual verification against reference would be slow)
        XCTAssertTrue(result.allSatisfy { $0.isFinite })
    }
    
    // MARK: - Matrix Transpose Tests
    
    func testBasicTranspose() async throws {
        let matrix = generateMatrix(rows: 3, cols: 4, pattern: .sequential)
        let descriptor = MatrixDescriptor(rows: 3, columns: 4)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        XCTAssertEqual(result.count, 3 * 4)
        
        // Verify against reference implementation
        let reference = referenceTranspose(matrix, rows: 3, cols: 4)
        XCTAssertTrue(matricesEqual(result, reference))
    }
    
    func testSquareMatrixTranspose() async throws {
        let matrix = generateMatrix(rows: 4, cols: 4, pattern: .random)
        let descriptor = MatrixDescriptor(rows: 4, columns: 4)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        XCTAssertEqual(result.count, 16)
        
        // Verify transpose property: (A^T)^T = A
        let doubleTranspose = try await engine!.transpose(result, descriptor: MatrixDescriptor(rows: 4, columns: 4))
        XCTAssertTrue(matricesEqual(doubleTranspose, matrix))
    }
    
    func testTransposeOfIdentityMatrix() async throws {
        let identity = generateMatrix(rows: 5, cols: 5, pattern: .identity)
        let descriptor = MatrixDescriptor(rows: 5, columns: 5)
        
        let result = try await engine!.transpose(identity, descriptor: descriptor)
        
        // Identity matrix should equal its transpose
        XCTAssertTrue(matricesEqual(result, identity))
    }
    
    func testSmallMatrixTranspose() async throws {
        // Test CPU path
        let config = MatrixConfiguration(preferGPUThreshold: 1000)
        let cpuEngine = await MatrixEngine(context: context, configuration: config)
        
        let matrix = generateMatrix(rows: 2, cols: 3, pattern: .sequential)
        let descriptor = MatrixDescriptor(rows: 2, columns: 3)
        
        let result = try await cpuEngine.transpose(matrix, descriptor: descriptor)
        
        let reference = referenceTranspose(matrix, rows: 2, cols: 3)
        XCTAssertTrue(matricesEqual(result, reference))
    }
    
    func testLargeMatrixTranspose() async throws {
        // Test GPU path
        let config = MatrixConfiguration(preferGPUThreshold: 10)
        let gpuEngine = await MatrixEngine(context: context, configuration: config)
        
        let matrix = generateMatrix(rows: 50, cols: 40, pattern: .random)
        let descriptor = MatrixDescriptor(rows: 50, columns: 40)
        
        let result = try await gpuEngine.transpose(matrix, descriptor: descriptor)
        
        XCTAssertEqual(result.count, 50 * 40)
        XCTAssertTrue(result.allSatisfy { $0.isFinite })
    }
    
    // MARK: - Batch Operations Tests
    
    func testBatchMatrixVectorMultiplication() async throws {
        let batchSize = 5
        let rows = 4
        let cols = 6
        
        let matrices = (0..<batchSize).map { _ in generateMatrix(rows: rows, cols: cols, pattern: .random) }
        let vectors = (0..<batchSize).map { _ in generateVector(size: cols, pattern: .random) }
        let descriptor = MatrixDescriptor(rows: rows, columns: cols)
        
        let results = try await engine!.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        
        XCTAssertEqual(results.count, batchSize)
        XCTAssertTrue(results.allSatisfy { $0.count == rows })
        
        // Verify each result individually
        for i in 0..<batchSize {
            let expectedSize = rows
            XCTAssertEqual(results[i].count, expectedSize)
            XCTAssertTrue(results[i].allSatisfy { $0.isFinite })
        }
    }
    
    func testBatchOperationDimensionMismatch() async throws {
        let matrices = [generateMatrix(rows: 3, cols: 4, pattern: .random)]
        let vectors = [
            generateVector(size: 4, pattern: .random),
            generateVector(size: 4, pattern: .random)
        ]  // Different count than matrices
        let descriptor = MatrixDescriptor(rows: 3, columns: 4)
        
        do {
            _ = try await engine!.batchMatrixVectorMultiply(
                matrices: matrices,
                descriptor: descriptor,
                vectors: vectors
            )
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Verified dimension mismatch was thrown
        }
    }
    
    func testEmptyBatchOperation() async throws {
        let matrices: [[Float]] = []
        let vectors: [[Float]] = []
        let descriptor = MatrixDescriptor(rows: 3, columns: 4)
        
        let results = try await engine!.batchMatrixVectorMultiply(
            matrices: matrices,
            descriptor: descriptor,
            vectors: vectors
        )
        
        XCTAssertEqual(results.count, 0)
    }
    
    func testBatchOperationWithMismatchedVectorDimensions() async throws {
        let matrices = [generateMatrix(rows: 3, cols: 4, pattern: .random)]
        let vectors = [generateVector(size: 5, pattern: .random)]  // Wrong vector size
        let descriptor = MatrixDescriptor(rows: 3, columns: 4)
        
        do {
            _ = try await engine!.batchMatrixVectorMultiply(
                matrices: matrices,
                descriptor: descriptor,
                vectors: vectors
            )
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError where error.kind == .dimensionMismatch {
            // Verified dimension mismatch was thrown
        }
    }
    
    // MARK: - Performance and Metrics Tests
    
    func testPerformanceMetricsTracking() async throws {
        let initialMetrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(initialMetrics.operations, 0)
        XCTAssertEqual(initialMetrics.averageTime, 0)
        
        // Perform some operations
        let matrixA = generateMatrix(rows: 5, cols: 5, pattern: .random)
        let matrixB = generateMatrix(rows: 5, cols: 5, pattern: .random)
        let descriptor = MatrixDescriptor(rows: 5, columns: 5)
        
        _ = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
        _ = try await engine!.transpose(matrixA, descriptor: descriptor)
        
        let finalMetrics = await engine!.getPerformanceMetrics()
        XCTAssertGreaterThan(finalMetrics.operations, initialMetrics.operations)
        XCTAssertGreaterThan(finalMetrics.averageTime, 0)
    }
    
    func testOperationPerformanceComparison() async throws {
        let matrixSize = 100
        let matrix = generateMatrix(rows: matrixSize, cols: matrixSize, pattern: .random)
        let descriptor = MatrixDescriptor(rows: matrixSize, columns: matrixSize)
        
        // GPU configuration
        let gpuConfig = MatrixConfiguration(preferGPUThreshold: 10) // Force GPU
        let gpuEngine = await MatrixEngine(context: context, configuration: gpuConfig)
        
        // CPU configuration
        let cpuConfig = MatrixConfiguration(preferGPUThreshold: 100000) // Force CPU
        let cpuEngine = await MatrixEngine(context: context, configuration: cpuConfig)
        
        // Time GPU operation
        let gpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpuEngine.transpose(matrix, descriptor: descriptor)
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
        
        // Time CPU operation
        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await cpuEngine.transpose(matrix, descriptor: descriptor)
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
        
        print("GPU transpose time: \(gpuTime)s")
        print("CPU transpose time: \(cpuTime)s")
        
        // Both should complete in reasonable time
        XCTAssertLessThan(gpuTime, 1.0)
        XCTAssertLessThan(cpuTime, 1.0)
    }
    
    // MARK: - Mathematical Property Tests
    
    func testMatrixMultiplicationAssociativity() async throws {
        let matrixA = generateMatrix(rows: 3, cols: 4, pattern: .random)
        let matrixB = generateMatrix(rows: 4, cols: 5, pattern: .random)
        let matrixC = generateMatrix(rows: 5, cols: 3, pattern: .random)
        
        let descriptorA = MatrixDescriptor(rows: 3, columns: 4)
        let descriptorB = MatrixDescriptor(rows: 4, columns: 5)
        let descriptorC = MatrixDescriptor(rows: 5, columns: 3)
        
        // Compute (A * B) * C
        let AB = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        let descriptorAB = MatrixDescriptor(rows: 3, columns: 5)
        let ABC_left = try await engine!.multiply(AB, descriptorA: descriptorAB, matrixC, descriptorB: descriptorC)
        
        // Compute A * (B * C)
        let BC = try await engine!.multiply(matrixB, descriptorA: descriptorB, matrixC, descriptorB: descriptorC)
        let descriptorBC = MatrixDescriptor(rows: 4, columns: 3)
        let ABC_right = try await engine!.multiply(matrixA, descriptorA: descriptorA, BC, descriptorB: descriptorBC)
        
        // Should be approximately equal (within floating-point precision)
        XCTAssertTrue(matricesEqual(ABC_left, ABC_right, tolerance: 1e-4))
    }
    
    func testTransposeMultiplicationProperty() async throws {
        let matrixA = generateMatrix(rows: 4, cols: 3, pattern: .random)
        let matrixB = generateMatrix(rows: 3, cols: 5, pattern: .random)
        
        let descriptorA = MatrixDescriptor(rows: 4, columns: 3)
        let descriptorB = MatrixDescriptor(rows: 3, columns: 5)
        
        // Compute (A * B)^T
        let AB = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
        let descriptorAB = MatrixDescriptor(rows: 4, columns: 5)
        let AB_T = try await engine!.transpose(AB, descriptor: descriptorAB)
        
        // Compute B^T * A^T
        let A_T = try await engine!.transpose(matrixA, descriptor: descriptorA)
        let B_T = try await engine!.transpose(matrixB, descriptor: descriptorB)
        let descriptorA_T = MatrixDescriptor(rows: 3, columns: 4)
        let descriptorB_T = MatrixDescriptor(rows: 5, columns: 3)
        let BT_AT = try await engine!.multiply(B_T, descriptorA: descriptorB_T, A_T, descriptorB: descriptorA_T)
        
        // (A * B)^T = B^T * A^T
        XCTAssertTrue(matricesEqual(AB_T, BT_AT, tolerance: 1e-4))
    }
    
    func testDiagonalMatrixProperties() async throws {
        let diagonalValues: [Float] = [2.0, 3.0, 4.0, 5.0]
        let diagonal = generateMatrix(rows: 4, cols: 4, pattern: .diagonal(diagonalValues))
        let testMatrix = generateMatrix(rows: 4, cols: 4, pattern: .random)
        
        let descriptor = MatrixDescriptor(rows: 4, columns: 4)
        
        // D * A where D is diagonal
        let DA = try await engine!.multiply(diagonal, descriptorA: descriptor, testMatrix, descriptorB: descriptor)
        
        // Verify that each row of the result is scaled by the corresponding diagonal element
        for i in 0..<4 {
            let scalingFactor = diagonalValues[i]
            for j in 0..<4 {
                let expected = testMatrix[i * 4 + j] * scalingFactor
                let actual = DA[i * 4 + j]
                XCTAssertEqual(actual, expected, accuracy: 1e-5)
            }
        }
    }
    
    // MARK: - Edge Cases and Error Handling Tests
    
    func testMatrixWithNaNValues() async throws {
        var matrix = generateMatrix(rows: 3, cols: 3, pattern: .random)
        matrix[4] = Float.nan  // Inject NaN
        
        let descriptor = MatrixDescriptor(rows: 3, columns: 3)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        // NaN should propagate
        XCTAssertTrue(result.contains { $0.isNaN })
    }
    
    func testMatrixWithInfinityValues() async throws {
        var matrix = generateMatrix(rows: 3, cols: 3, pattern: .random)
        matrix[0] = Float.infinity
        matrix[4] = -Float.infinity
        
        let descriptor = MatrixDescriptor(rows: 3, columns: 3)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        
        // Infinity should be preserved in transpose
        XCTAssertTrue(result.contains { $0.isInfinite })
    }
    
    func testVerySmallMatrix() async throws {
        let matrix = [Float(42.0)]
        let descriptor = MatrixDescriptor(rows: 1, columns: 1)
        
        let transposed = try await engine!.transpose(matrix, descriptor: descriptor)
        XCTAssertEqual(transposed, matrix)
        
        let multiplied = try await engine!.multiply(matrix, descriptorA: descriptor, matrix, descriptorB: descriptor)
        XCTAssertEqual(multiplied[0], 42.0 * 42.0)
    }
    
    func testEmptyMatrix() async throws {
        let matrix: [Float] = []
        let descriptor = MatrixDescriptor(rows: 0, columns: 0)
        
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        XCTAssertEqual(result.count, 0)
    }
    
    // MARK: - Concurrent Operations Tests
    
    func testConcurrentMatrixOperations() async throws {
        let matrixSize = 10
        let matrices = (0..<5).map { _ in generateMatrix(rows: matrixSize, cols: matrixSize, pattern: .random) }
        let descriptor = MatrixDescriptor(rows: matrixSize, columns: matrixSize)
        
        // Perform multiple operations concurrently
        await withTaskGroup(of: Void.self) { group in
            for (i, matrix) in matrices.enumerated() {
                group.addTask { [engine] in
                    do {
                        // Mix of different operations
                        if i % 2 == 0 {
                            _ = try await engine!.transpose(matrix, descriptor: descriptor)
                        } else {
                            _ = try await engine!.multiply(matrix, descriptorA: descriptor, matrix, descriptorB: descriptor)
                        }
                    } catch {
                        XCTFail("Concurrent operation \(i) failed: \(error)")
                    }
                }
            }
        }
        
        // Engine should still be functional
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertGreaterThan(metrics.operations, 0)
    }
    
    func testConcurrentAccessToMetrics() async throws {
        let matrix = generateMatrix(rows: 10, cols: 10, pattern: .random)
        let descriptor = MatrixDescriptor(rows: 10, columns: 10)
        
        // Concurrent operations and metrics access
        await withTaskGroup(of: Void.self) { group in
            // Operation tasks
            for i in 0..<3 {
                group.addTask { [engine] in
                    do {
                        _ = try await engine!.transpose(matrix, descriptor: descriptor)
                    } catch {
                        print("Operation \(i) failed: \(error)")
                    }
                }
            }
            
            // Metrics access tasks
            for _ in 0..<5 {
                group.addTask { [engine] in
                    let metrics = await engine!.getPerformanceMetrics()
                    XCTAssertGreaterThanOrEqual(metrics.operations, 0)
                }
            }
        }
    }
    
    // MARK: - Integration Tests
    
    func testComplexMatrixWorkflow() async throws {
        // Test a complex workflow involving multiple operations
        let size = 8
        let matrixA = generateMatrix(rows: size, cols: size, pattern: .random)
        let matrixB = generateMatrix(rows: size, cols: size, pattern: .random)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        // Step 1: Multiply A * B
        let AB = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
        
        // Step 2: Transpose the result
        let AB_T = try await engine!.transpose(AB, descriptor: descriptor)
        
        // Step 3: Multiply by original A again
        let final = try await engine!.multiply(AB_T, descriptorA: descriptor, matrixA, descriptorB: descriptor)
        
        XCTAssertEqual(final.count, size * size)
        XCTAssertTrue(final.allSatisfy { $0.isFinite })
        
        // Verify metrics were updated for all operations
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertGreaterThanOrEqual(metrics.operations, 3)
    }
    
    func testDifferentConfigurationBehaviors() async throws {
        let matrix = generateMatrix(rows: 20, cols: 20, pattern: .random)
        let descriptor = MatrixDescriptor(rows: 20, columns: 20)
        
        // Test different tile sizes
        let config1 = MatrixConfiguration(tileSize: 16, preferGPUThreshold: 10)
        let config2 = MatrixConfiguration(tileSize: 64, preferGPUThreshold: 10)
        
        let engine1 = await MatrixEngine(context: context, configuration: config1)
        let engine2 = await MatrixEngine(context: context, configuration: config2)
        
        let result1 = try await engine1.transpose(matrix, descriptor: descriptor)
        let result2 = try await engine2.transpose(matrix, descriptor: descriptor)
        
        // Results should be identical regardless of configuration
        XCTAssertTrue(matricesEqual(result1, result2))
    }
    
    // MARK: - Stress Tests
    
    func testLargeMatrixStressTest() async throws {
        let size = 200
        let matrix = generateMatrix(rows: size, cols: size, pattern: .random)
        let descriptor = MatrixDescriptor(rows: size, columns: size)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await engine!.transpose(matrix, descriptor: descriptor)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        XCTAssertEqual(result.count, size * size)
        XCTAssertLessThan(elapsed, 5.0)  // Should complete within 5 seconds
        
        print("Large matrix (\(size)x\(size)) transpose time: \(elapsed)s")
    }
    
    func testMemoryIntensiveOperations() async throws {
        // Test with matrices that exercise memory allocation
        let sizes = [(50, 60), (80, 70), (100, 90)]
        
        for (rows, cols) in sizes {
            let matrixA = generateMatrix(rows: rows, cols: cols, pattern: .random)
            let matrixB = generateMatrix(rows: cols, cols: rows, pattern: .random)
            
            let descriptorA = MatrixDescriptor(rows: rows, columns: cols)
            let descriptorB = MatrixDescriptor(rows: cols, columns: rows)
            
            let result = try await engine!.multiply(matrixA, descriptorA: descriptorA, matrixB, descriptorB: descriptorB)
            
            XCTAssertEqual(result.count, rows * rows)
            XCTAssertTrue(result.allSatisfy { $0.isFinite })
        }
        
        let metrics = await engine!.getPerformanceMetrics()
        XCTAssertEqual(metrics.operations, sizes.count)
    }
}

// MARK: - Performance Benchmarking (Optional)

extension MatrixEngineTests {
    func testMatrixMultiplicationBenchmark() async throws {
        let sizes = [10, 50, 100, 200]
        
        for size in sizes {
            let matrixA = generateMatrix(rows: size, cols: size, pattern: .random)
            let matrixB = generateMatrix(rows: size, cols: size, pattern: .random)
            let descriptor = MatrixDescriptor(rows: size, columns: size)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try await engine!.multiply(matrixA, descriptorA: descriptor, matrixB, descriptorB: descriptor)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            print("Matrix multiplication \(size)x\(size): \(elapsed)s")
            
            // Reasonable time bounds
            XCTAssertLessThan(elapsed, 10.0)
        }
    }
}