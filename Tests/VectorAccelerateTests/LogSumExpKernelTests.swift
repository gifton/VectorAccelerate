//
//  LogSumExpKernelTests.swift
//  VectorAccelerate
//
//  Tests for GPU-accelerated log-sum-exp and softmax operations.
//
//  Test coverage:
//  - Row-wise logsumexp correctness
//  - Full reduction correctness
//  - Numerical stability with large values
//  - Vectorized matches scalar
//  - Edge cases (single element, single row, all equal)
//  - Special values (-Inf, +Inf, NaN)
//  - Softmax rows sum to 1.0
//  - Encode API for kernel fusion
//  - FusibleKernel conformance
//  - Performance benchmarks
//
//  Note: Requires macOS 26.0+ to run.

import XCTest
@testable import VectorAccelerate
@preconcurrency import Metal
import VectorCore

// MARK: - Helper Functions

/// Compute logsumexp on CPU for verification
private func cpuLogSumExp(_ values: [Float]) -> Float {
    guard !values.isEmpty else { return -.infinity }
    let maxVal = values.max()!
    if maxVal == -.infinity { return -.infinity }
    if maxVal == .infinity { return .infinity }
    let sumExp = values.reduce(Float(0)) { $0 + expf($1 - maxVal) }
    return maxVal + logf(sumExp)
}

/// Compute softmax on CPU for verification
private func cpuSoftmax(_ values: [Float]) -> [Float] {
    let lse = cpuLogSumExp(values)
    return values.map { expf($0 - lse) }
}

// MARK: - LogSumExpKernel Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class LogSumExpKernelTests: XCTestCase {

    var context: Metal4Context!
    var kernel: LogSumExpKernel!

    override func setUp() async throws {
        try await super.setUp()
        context = try await Metal4Context()
        kernel = try await LogSumExpKernel(context: context)
    }

    override func tearDown() async throws {
        kernel = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - Row-wise LogSumExp Correctness Tests

    func testRowwiseCorrectness() async throws {
        // Test case from spec
        let input: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        // Compute expected using CPU helper
        let expected = input.map { cpuLogSumExp($0) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<output.count {
            XCTAssertEqual(Double(output[i]), Double(expected[i]), accuracy: 1e-5,
                           "Mismatch at row \(i): output=\(output[i]), expected=\(expected[i])")
        }
    }

    func testRowwiseSingleColumn() async throws {
        // With d=1, logsumexp(x) = x
        let input: [[Float]] = [
            [5.0],
            [-3.0],
            [0.0]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        XCTAssertEqual(Double(output[0]), 5.0, accuracy: 1e-6)
        XCTAssertEqual(Double(output[1]), -3.0, accuracy: 1e-6)
        XCTAssertEqual(Double(output[2]), 0.0, accuracy: 1e-6)
    }

    func testRowwiseSingleRow() async throws {
        let input: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        let expected = cpuLogSumExp(input[0])
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(Double(output[0]), Double(expected), accuracy: 1e-5)
    }

    func testRowwiseAllEqual() async throws {
        let input: [[Float]] = [
            [3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        // logsumexp([c, c, c, c]) = c + log(4)
        XCTAssertEqual(Double(output[0]), Double(3.0 + logf(4.0)), accuracy: 1e-5)
        XCTAssertEqual(Double(output[1]), Double(0.0 + logf(4.0)), accuracy: 1e-5)
    }

    // MARK: - Full Reduction Tests

    func testReduceCorrectness() async throws {
        let input: [Float] = [1.0, 2.0, 3.0]

        let result = try await kernel.reduce(input: input)

        let expected = cpuLogSumExp(input)
        XCTAssertEqual(Double(result.value), Double(expected), accuracy: 1e-5)
    }

    func testReduceSingleElement() async throws {
        let input: [Float] = [42.0]

        let result = try await kernel.reduce(input: input)

        // logsumexp of single element is that element
        XCTAssertEqual(Double(result.value), 42.0, accuracy: 1e-5)
    }

    func testReduceAllEqual() async throws {
        let input: [Float] = [5.0, 5.0, 5.0, 5.0, 5.0]

        let result = try await kernel.reduce(input: input)

        // logsumexp([c, c, ..., c]) with n elements = c + log(n)
        let expected: Float = 5.0 + logf(5.0)
        XCTAssertEqual(Double(result.value), Double(expected), accuracy: 1e-5)
    }

    // MARK: - Numerical Stability Tests

    func testNumericalStabilityLargeValues() async throws {
        // Large values that would overflow naive exp
        let input: [Float] = [1000, 1001, 1002]

        let result = try await kernel.reduce(input: input)

        // Should not be Inf or NaN
        XCTAssertFalse(result.value.isInfinite, "Result should not be infinite")
        XCTAssertFalse(result.value.isNaN, "Result should not be NaN")

        // Expected: 1002 + log(1 + e^(-1) + e^(-2)) ~ 1002.41
        let expected: Float = 1002.0 + logf(1.0 + expf(-1.0) + expf(-2.0))
        XCTAssertEqual(Double(result.value), Double(expected), accuracy: 0.01)
    }

    func testNumericalStabilityNegativeLargeValues() async throws {
        // Large negative values
        let input: [Float] = [-1000, -1001, -1002]

        let result = try await kernel.reduce(input: input)

        XCTAssertFalse(result.value.isNaN, "Result should not be NaN")

        // Expected: -1000 + log(1 + e^(-1) + e^(-2)) ~ -999.59
        let expected: Float = -1000.0 + logf(1.0 + expf(-1.0) + expf(-2.0))
        XCTAssertEqual(Double(result.value), Double(expected), accuracy: 0.01)
    }

    func testNumericalStabilityRowwise() async throws {
        // Large values in rows
        let input: [[Float]] = [
            [1000, 1001, 1002, 1003],
            [-500, -499, -498, -497]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        XCTAssertFalse(output[0].isInfinite)
        XCTAssertFalse(output[0].isNaN)
        XCTAssertFalse(output[1].isInfinite)
        XCTAssertFalse(output[1].isNaN)

        // Check against CPU implementation
        let expected0 = cpuLogSumExp(input[0])
        let expected1 = cpuLogSumExp(input[1])

        XCTAssertEqual(Double(output[0]), Double(expected0), accuracy: 0.01)
        XCTAssertEqual(Double(output[1]), Double(expected1), accuracy: 0.01)
    }

    // MARK: - Vectorized vs Scalar Tests

    func testVectorizedMatchesScalar() async throws {
        // Vectorized path: d % 4 == 0 and d >= 16
        let n = 100
        let d = 128  // Divisible by 4, >= 16

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -5...5) })
        }

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        // Compute expected using CPU
        let expected = input.map { cpuLogSumExp($0) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<n {
            XCTAssertEqual(Double(output[i]), Double(expected[i]), accuracy: 1e-4,
                           "Mismatch at row \(i): output=\(output[i]), expected=\(expected[i])")
        }
    }

    func testNonVectorizedPath() async throws {
        // Non-vectorized path: d % 4 != 0
        let n = 50
        let d = 13  // Not divisible by 4

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -5...5) })
        }

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        // Compute expected using CPU
        let expected = input.map { cpuLogSumExp($0) }

        XCTAssertEqual(output.count, expected.count)
        for i in 0..<n {
            XCTAssertEqual(Double(output[i]), Double(expected[i]), accuracy: 1e-4)
        }
    }

    // MARK: - Special Values Tests

    func testAllNegativeInfinity() async throws {
        let input: [Float] = [-.infinity, -.infinity, -.infinity]

        let result = try await kernel.reduce(input: input)

        // logsumexp of all -inf should be -inf
        XCTAssertEqual(result.value, -.infinity)
    }

    func testContainsPositiveInfinity() async throws {
        let input: [Float] = [1.0, .infinity, 2.0]

        let result = try await kernel.reduce(input: input)

        // If any element is +inf, result should be +inf
        XCTAssertEqual(result.value, .infinity)
    }

    func testRowwiseInfinityHandling() async throws {
        let input: [[Float]] = [
            [-.infinity, -.infinity],
            [1.0, .infinity],
            [0.0, 0.0]
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        XCTAssertEqual(output[0], -.infinity)
        XCTAssertEqual(output[1], .infinity)
        XCTAssertFalse(output[2].isInfinite)
    }

    func testMixedInfinities() async throws {
        let input: [[Float]] = [
            [-.infinity, 1.0, 2.0],  // -inf is ignored, result ~2.31
            [1.0, -.infinity, 2.0]   // Same result
        ]

        let result = try await kernel.rowwise(input: input)
        let output = result.asArray()

        // exp(-inf) = 0, so these rows are equivalent to [1.0, 2.0]
        let expected = cpuLogSumExp([1.0, 2.0])
        XCTAssertEqual(Double(output[0]), Double(expected), accuracy: 1e-4)
        XCTAssertEqual(Double(output[1]), Double(expected), accuracy: 1e-4)
    }

    // MARK: - Softmax Tests

    func testSoftmaxSumsToOne() async throws {
        let input: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0]
        ]

        let result = try await kernel.softmax(input: input)
        let output = result.asArray()

        for (rowIdx, row) in output.enumerated() {
            let sum = row.reduce(Float(0), +)
            XCTAssertEqual(Double(sum), 1.0, accuracy: 1e-5,
                           "Row \(rowIdx) sums to \(sum), expected 1.0")
        }
    }

    func testSoftmaxMonotonicity() async throws {
        // Larger input values should give larger softmax values
        let input: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0]
        ]

        let result = try await kernel.softmax(input: input)
        let output = result.asArray()

        XCTAssertEqual(output.count, 1)
        let row = output[0]

        // Check monotonicity: softmax[i] < softmax[i+1] when input[i] < input[i+1]
        for i in 0..<3 {
            XCTAssertLessThan(row[i], row[i + 1],
                              "softmax[\(i)]=\(row[i]) should be < softmax[\(i+1)]=\(row[i+1])")
        }
    }

    func testSoftmaxUniformInputs() async throws {
        // Equal inputs should give uniform distribution
        let input: [[Float]] = [
            [0.0, 0.0, 0.0, 0.0]
        ]

        let result = try await kernel.softmax(input: input)
        let output = result.asArray()[0]

        let expectedValue: Float = 0.25  // 1/4
        for i in 0..<4 {
            XCTAssertEqual(Double(output[i]), Double(expectedValue), accuracy: 1e-5)
        }
    }

    func testSoftmaxNumericalStability() async throws {
        // Large values that would overflow naive softmax
        let input: [[Float]] = [
            [1000, 1001, 1002, 1003]
        ]

        let result = try await kernel.softmax(input: input)
        let output = result.asArray()[0]

        // Should not contain NaN or Inf
        for (i, val) in output.enumerated() {
            XCTAssertFalse(val.isNaN, "softmax[\(i)] is NaN")
            XCTAssertFalse(val.isInfinite, "softmax[\(i)] is infinite")
            XCTAssertGreaterThanOrEqual(val, 0, "softmax[\(i)] should be >= 0")
            XCTAssertLessThanOrEqual(val, 1, "softmax[\(i)] should be <= 1")
        }

        // Should sum to 1
        let sum = output.reduce(Float(0), +)
        XCTAssertEqual(Double(sum), 1.0, accuracy: 1e-5)

        // Largest input should have largest probability
        let maxIdx = output.enumerated().max(by: { $0.1 < $1.1 })!.0
        XCTAssertEqual(maxIdx, 3)
    }

    func testSoftmaxInfinityHandling() async throws {
        // Row with +inf should put all mass on that element
        let input: [[Float]] = [
            [1.0, .infinity, 2.0, 3.0]
        ]

        let result = try await kernel.softmax(input: input)
        let output = result.asArray()[0]

        XCTAssertEqual(Double(output[0]), 0.0, accuracy: 1e-5)
        XCTAssertEqual(Double(output[1]), 1.0, accuracy: 1e-5)  // +inf element
        XCTAssertEqual(Double(output[2]), 0.0, accuracy: 1e-5)
        XCTAssertEqual(Double(output[3]), 0.0, accuracy: 1e-5)
    }

    // MARK: - Edge Cases

    func testEmptyInputThrows() async throws {
        let input: [[Float]] = []

        do {
            _ = try await kernel.rowwise(input: input)
            XCTFail("Expected error for empty input")
        } catch {
            // Expected
        }
    }

    func testEmptyRowsThrows() async throws {
        let input: [[Float]] = [[]]

        do {
            _ = try await kernel.rowwise(input: input)
            XCTFail("Expected error for empty rows")
        } catch {
            // Expected
        }
    }

    func testMismatchedRowLengthsThrows() async throws {
        let input: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0]  // Different length
        ]

        do {
            _ = try await kernel.rowwise(input: input)
            XCTFail("Expected error for mismatched row lengths")
        } catch {
            // Expected
        }
    }

    // MARK: - FusibleKernel Conformance

    func testFusibleKernelConformance() {
        XCTAssertEqual(kernel.name, "LogSumExpKernel")
        XCTAssertTrue(kernel.fusibleWith.contains("L2Distance"))
        XCTAssertTrue(kernel.fusibleWith.contains("Any"))
        XCTAssertTrue(kernel.requiresBarrierAfter)
    }

    // MARK: - Encode API Tests

    func testEncodeAPIRowwise() async throws {
        let n = 100
        let d = 64

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -3...3) })
        }

        let flat = input.flatMap { $0 }
        let device = context.device.rawDevice

        guard let inputBuffer = flat.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            XCTFail("Failed to create input buffer")
            return
        }

        guard let outputBuffer = device.makeBuffer(
            length: n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create output buffer")
            return
        }

        // Test encode API
        try await context.executeAndWait { [kernel] _, encoder in
            let result = kernel!.encodeRowwise(
                into: encoder,
                input: inputBuffer,
                output: outputBuffer,
                n: n,
                d: d
            )

            // Verify encoding result
            XCTAssertTrue(result.pipelineName.contains("logsumexp"))
        }

        // Verify output
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: n)
        let output = Array(UnsafeBufferPointer(start: ptr, count: n))

        // Compute expected
        let expected = input.map { cpuLogSumExp($0) }

        for i in 0..<n {
            XCTAssertEqual(Double(output[i]), Double(expected[i]), accuracy: 1e-4)
        }
    }

    // MARK: - Convenience API Tests

    func testRowwiseArrayConvenience() async throws {
        let input: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        let output = try await kernel.rowwiseArray(input: input)

        XCTAssertEqual(output.count, 2)
    }

    func testReduceValueConvenience() async throws {
        let input: [Float] = [1.0, 2.0, 3.0]

        let value = try await kernel.reduceValue(input: input)

        let expected = cpuLogSumExp(input)
        XCTAssertEqual(Double(value), Double(expected), accuracy: 1e-5)
    }

    func testSoftmaxArrayConvenience() async throws {
        let input: [[Float]] = [
            [1.0, 2.0, 3.0]
        ]

        let output = try await kernel.softmaxArray(input: input)

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].count, 3)

        let sum = output[0].reduce(Float(0), +)
        XCTAssertEqual(Double(sum), 1.0, accuracy: 1e-5)
    }

    // MARK: - Performance Tests

    func testPerformanceRowwise1000x50() async throws {
        let n = 1000
        let d = 50

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -5...5) })
        }

        let result = try await kernel.rowwise(input: input)

        print("LogSumExp rowwise (\(n)x\(d)):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")

        // Expected: ~0.1ms according to spec
        XCTAssertLessThan(result.executionTime, 0.01, "Performance regression: > 10ms")
    }

    func testPerformanceRowwise10000x100() async throws {
        let n = 10000
        let d = 100

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -5...5) })
        }

        let result = try await kernel.rowwise(input: input)

        print("LogSumExp rowwise (\(n)x\(d)):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")

        // Expected: ~1ms according to spec
        XCTAssertLessThan(result.executionTime, 0.05, "Performance regression: > 50ms")
    }

    func testPerformanceReduce1M() async throws {
        let count = 1_000_000
        let input = (0..<count).map { _ in Float.random(in: -10...10) }

        let result = try await kernel.reduce(input: input)

        print("LogSumExp reduce (\(count) elements):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")
        print("  Result: \(result.value)")

        // Expected: ~0.5ms according to spec
        XCTAssertLessThan(result.executionTime, 0.02, "Performance regression: > 20ms")
    }

    func testPerformanceSoftmax1000x50() async throws {
        let n = 1000
        let d = 50

        var input: [[Float]] = []
        for _ in 0..<n {
            input.append((0..<d).map { _ in Float.random(in: -5...5) })
        }

        let result = try await kernel.softmax(input: input)

        print("Softmax (\(n)x\(d)):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")

        // Verify correctness on sample
        let output = result.asArray()
        for i in stride(from: 0, to: n, by: 100) {
            let sum = output[i].reduce(Float(0), +)
            XCTAssertEqual(Double(sum), 1.0, accuracy: 1e-4, "Row \(i) doesn't sum to 1.0")
        }

        // Expected: ~0.2ms according to spec
        XCTAssertLessThan(result.executionTime, 0.02, "Performance regression: > 20ms")
    }

    // MARK: - Topic Modeling Scenario

    func testTopicProbabilityDistribution() async throws {
        // Simulate topic scores for documents
        let numDocs = 500
        let numTopics = 50

        var topicScores: [[Float]] = []
        for _ in 0..<numDocs {
            // Random topic scores (like output from a topic model)
            topicScores.append((0..<numTopics).map { _ in Float.random(in: -3...3) })
        }

        // Convert to probabilities using softmax
        let result = try await kernel.softmax(input: topicScores)
        let topicProbs = result.asArray()

        // Verify all rows sum to 1
        for (docIdx, probs) in topicProbs.enumerated() {
            let sum = probs.reduce(Float(0), +)
            XCTAssertEqual(Double(sum), 1.0, accuracy: 1e-4,
                           "Document \(docIdx) topic probabilities don't sum to 1.0")

            // Verify all probabilities are in [0, 1]
            for (topicIdx, prob) in probs.enumerated() {
                XCTAssertGreaterThanOrEqual(prob, 0,
                    "Doc \(docIdx) Topic \(topicIdx) has negative probability")
                XCTAssertLessThanOrEqual(prob, 1,
                    "Doc \(docIdx) Topic \(topicIdx) has probability > 1")
            }
        }

        print("Topic probability distribution (\(numDocs) docs, \(numTopics) topics):")
        print("  Time: \(String(format: "%.3f", result.executionTime * 1000)) ms")
        print("  Throughput: \(String(format: "%.1f", result.throughputGBps)) GB/s")
    }
}
