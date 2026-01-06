//
//  BatchMaxKernelTests.swift
//  VectorAccelerate
//
//  Tests for GPU-accelerated element-wise maximum operations.
//
//  Test coverage:
//  - max3 correctness (small arrays)
//  - max2 correctness (small arrays)
//  - Vectorized matches scalar (count % 4 == 0)
//  - Non-vectorized path (count % 4 != 0)
//  - In-place operations
//  - Edge cases (empty, single element, all equal, negative values)
//  - Inf and NaN handling
//  - Encode API for kernel fusion
//  - Performance benchmark
//
//  Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - BatchMaxKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class BatchMaxKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: BatchMaxKernel!

    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
        kernel = try await BatchMaxKernel(context: context)
    }

    override func tearDown() async throws {
        kernel = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - max3 Correctness Tests

    func testMax3Correctness() async throws {
        // Simple test case from spec
        let a: [Float] = [1.0, 5.0, 3.0, 2.0]
        let b: [Float] = [4.0, 2.0, 6.0, 1.0]
        let c: [Float] = [2.0, 3.0, 1.0, 7.0]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        XCTAssertEqual(output, [4.0, 5.0, 6.0, 7.0])
    }

    func testMax3WithNegativeValues() async throws {
        let a: [Float] = [-1.0, -5.0, -3.0, 0.0]
        let b: [Float] = [-4.0, -2.0, -6.0, -1.0]
        let c: [Float] = [-2.0, -3.0, -1.0, -7.0]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        XCTAssertEqual(output, [-1.0, -2.0, -1.0, 0.0])
    }

    func testMax3AllEqual() async throws {
        let a: [Float] = [3.0, 3.0, 3.0, 3.0]
        let b: [Float] = [3.0, 3.0, 3.0, 3.0]
        let c: [Float] = [3.0, 3.0, 3.0, 3.0]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        XCTAssertEqual(output, [3.0, 3.0, 3.0, 3.0])
    }

    // MARK: - max2 Correctness Tests

    func testMax2Correctness() async throws {
        let a: [Float] = [1.0, 5.0, 3.0]
        let b: [Float] = [4.0, 2.0, 6.0]

        let result = try await kernel.max2(a: a, b: b)
        let output = result.asArray()

        XCTAssertEqual(output, [4.0, 5.0, 6.0])
    }

    // MARK: - Vectorized vs Scalar Tests

    func testVectorizedMatchesScalar() async throws {
        // Vectorized path: count % 4 == 0 and count >= 16
        let count = 1024
        let a = (0..<count).map { _ in Float.random(in: -10...10) }
        let b = (0..<count).map { _ in Float.random(in: -10...10) }
        let c = (0..<count).map { _ in Float.random(in: -10...10) }

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()
        let expected = zip(zip(a, b), c).map { max(max($0.0, $0.1), $1) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<count {
            XCTAssertEqual(output[i], expected[i], accuracy: 1e-6,
                           "Mismatch at index \(i): output=\(output[i]), expected=\(expected[i])")
        }
    }

    func testNonVectorizedPath() async throws {
        // Non-vectorized path: count % 4 != 0
        let count = 1023  // Not divisible by 4
        let a = (0..<count).map { _ in Float.random(in: -10...10) }
        let b = (0..<count).map { _ in Float.random(in: -10...10) }
        let c = (0..<count).map { _ in Float.random(in: -10...10) }

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()
        let expected = zip(zip(a, b), c).map { max(max($0.0, $0.1), $1) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<count {
            XCTAssertEqual(output[i], expected[i], accuracy: 1e-6)
        }
    }

    func testSmallCountDoesNotVectorize() async throws {
        // count < 16 should use scalar path even if divisible by 4
        let count = 8
        let a = (0..<count).map { _ in Float.random(in: -10...10) }
        let b = (0..<count).map { _ in Float.random(in: -10...10) }
        let c = (0..<count).map { _ in Float.random(in: -10...10) }

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()
        let expected = zip(zip(a, b), c).map { max(max($0.0, $0.1), $1) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<count {
            XCTAssertEqual(output[i], expected[i], accuracy: 1e-6)
        }
    }

    // MARK: - Edge Cases

    func testSingleElement() async throws {
        let a: [Float] = [5.0]
        let b: [Float] = [3.0]
        let c: [Float] = [4.0]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        XCTAssertEqual(output, [5.0])
    }

    func testMismatchedArraySizesThrows() async throws {
        let a: [Float] = [1.0, 2.0]
        let b: [Float] = [3.0]
        let c: [Float] = [4.0, 5.0]

        do {
            _ = try await kernel.max3(a: a, b: b, c: c)
            XCTFail("Expected error for mismatched array sizes")
        } catch {
            // Expected
        }
    }

    func testEmptyArraysThrows() async throws {
        let a: [Float] = []
        let b: [Float] = []
        let c: [Float] = []

        do {
            _ = try await kernel.max3(a: a, b: b, c: c)
            XCTFail("Expected error for empty arrays")
        } catch {
            // Expected
        }
    }

    // MARK: - Special Values Tests

    func testInfinityHandling() async throws {
        let a: [Float] = [1.0, .infinity, -.infinity, 0.0]
        let b: [Float] = [.infinity, 2.0, 0.0, -.infinity]
        let c: [Float] = [0.0, 0.0, 0.0, 0.0]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        XCTAssertEqual(output[0], .infinity)
        XCTAssertEqual(output[1], .infinity)
        XCTAssertEqual(output[2], 0.0)
        XCTAssertEqual(output[3], 0.0)
    }

    func testNaNHandling() async throws {
        // Metal's max(x, NaN) returns x (not NaN like IEEE 754 standard)
        // This is documented Metal behavior for performance
        let a: [Float] = [Float.nan, 1.0, 2.0]
        let b: [Float] = [1.0, Float.nan, 2.0]
        let c: [Float] = [2.0, 2.0, Float.nan]

        let result = try await kernel.max3(a: a, b: b, c: c)
        let output = result.asArray()

        // Metal's max() propagates the non-NaN value
        // max(NaN, 1.0, 2.0) -> max(max(NaN, 1.0), 2.0) -> max(1.0, 2.0) -> 2.0
        XCTAssertEqual(output[0], 2.0)
        // max(1.0, NaN, 2.0) -> max(max(1.0, NaN), 2.0) -> max(1.0, 2.0) -> 2.0
        XCTAssertEqual(output[1], 2.0)
        // max(2.0, 2.0, NaN) -> max(max(2.0, 2.0), NaN) -> max(2.0, NaN) -> 2.0
        XCTAssertEqual(output[2], 2.0)
    }

    // MARK: - In-Place Operations Tests

    func testMaxInplace() async throws {
        let device = context.device.rawDevice
        let a: [Float] = [1.0, 5.0, 3.0, 2.0]
        let b: [Float] = [4.0, 2.0, 6.0, 1.0]

        guard let aBuffer = a.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffer")
            return
        }

        guard let bBuffer = b.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffer")
            return
        }

        try await kernel.maxInplace(a: aBuffer, b: bBuffer, count: 4)

        let ptr = aBuffer.contents().bindMemory(to: Float.self, capacity: 4)
        let output = Array(UnsafeBufferPointer(start: ptr, count: 4))

        XCTAssertEqual(output, [4.0, 5.0, 6.0, 2.0])
    }

    func testMax3Inplace() async throws {
        let device = context.device.rawDevice
        let a: [Float] = [1.0, 5.0, 3.0, 2.0]
        let b: [Float] = [4.0, 2.0, 6.0, 1.0]
        let c: [Float] = [2.0, 3.0, 1.0, 7.0]

        guard let aBuffer = a.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffer")
            return
        }

        guard let bBuffer = b.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffer")
            return
        }

        guard let cBuffer = c.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffer")
            return
        }

        try await kernel.max3Inplace(a: aBuffer, b: bBuffer, c: cBuffer, count: 4)

        let ptr = aBuffer.contents().bindMemory(to: Float.self, capacity: 4)
        let output = Array(UnsafeBufferPointer(start: ptr, count: 4))

        XCTAssertEqual(output, [4.0, 5.0, 6.0, 7.0])
    }

    // MARK: - Encode API Tests

    func testEncodeAPIForFusion() async throws {
        // Test that vectorized path is used for count divisible by 4 and >= 16
        // We verify this indirectly by checking correctness at a vectorized count
        let count = 256

        let a = (0..<count).map { _ in Float.random(in: -10...10) }
        let b = (0..<count).map { _ in Float.random(in: -10...10) }
        let c = (0..<count).map { _ in Float.random(in: -10...10) }

        // Use the buffer-based API which internally uses encode
        let device = context.device.rawDevice

        guard let aBuffer = a.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }),
        let bBuffer = b.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }),
        let cBuffer = c.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create buffers")
            return
        }

        let result = try await kernel.max3(a: aBuffer, b: bBuffer, c: cBuffer, count: count)
        let output = result.asArray()
        let expected = zip(zip(a, b), c).map { max(max($0.0, $0.1), $1) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<count {
            XCTAssertEqual(output[i], expected[i], accuracy: 1e-6)
        }

        // Verify we got a result (the kernel executed successfully)
        XCTAssertGreaterThan(result.throughputGBps, 0)
    }

    // MARK: - FusibleKernel Conformance

    func testFusibleKernelConformance() {
        XCTAssertEqual(kernel.name, "BatchMaxKernel")
        XCTAssertTrue(kernel.fusibleWith.contains("L2Distance"))
        XCTAssertTrue(kernel.fusibleWith.contains("MutualReachability"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }

    // MARK: - Performance Tests

    func testPerformance1M() async throws {
        let count = 1_000_000
        let a = (0..<count).map { _ in Float.random(in: -10...10) }
        let b = (0..<count).map { _ in Float.random(in: -10...10) }
        let c = (0..<count).map { _ in Float.random(in: -10...10) }

        let result = try await kernel.max3(a: a, b: b, c: c)

        // Verify correctness on a sample
        let output = result.asArray()
        for i in stride(from: 0, to: count, by: 10000) {
            let expected = max(max(a[i], b[i]), c[i])
            XCTAssertEqual(output[i], expected, accuracy: 1e-6)
        }

        // Log performance
        print("BatchMax3 1M elements:")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")

        // Performance threshold generous for CI runner variability
        XCTAssertLessThan(result.executionTime, 0.015, "Performance regression: > 15ms for 1M elements")
    }

    func testMutualReachabilityScenario() async throws {
        // Simulate mutual reachability: max(core_a, core_b, dist)
        // For N points, we have N×N distances
        let n = 500
        let count = n * n

        // Simulate expanded core distances (same value repeated in rows/columns)
        let coreDistances = (0..<n).map { _ in Float.random(in: 0.1...1.0) }

        // Expand to N×N matrices
        var coreA = [Float](repeating: 0, count: count)
        var coreB = [Float](repeating: 0, count: count)
        for i in 0..<n {
            for j in 0..<n {
                coreA[i * n + j] = coreDistances[i]
                coreB[i * n + j] = coreDistances[j]
            }
        }

        // Simulate pairwise distances
        let distances = (0..<count).map { _ in Float.random(in: 0...2.0) }

        let result = try await kernel.max3(a: coreA, b: coreB, c: distances)
        let output = result.asArray()

        // Verify a few entries
        for _ in 0..<10 {
            let i = Int.random(in: 0..<n)
            let j = Int.random(in: 0..<n)
            let idx = i * n + j
            let expected = max(max(coreDistances[i], coreDistances[j]), distances[idx])
            XCTAssertEqual(output[idx], expected, accuracy: 1e-6)
        }

        print("Mutual reachability scenario (\(n)×\(n) = \(count) elements):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")
    }
}
