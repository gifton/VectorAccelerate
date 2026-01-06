//
//  LogSumExpKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for numerically stable log-sum-exp and softmax operations.
//
//  Features:
//  - Row-wise logsumexp: Essential for probability distributions
//  - Full reduction: Single scalar logsumexp over array
//  - Softmax: Stable softmax via logsumexp
//  - Vectorized float4 versions for better memory throughput
//  - FusibleKernel conformance for kernel fusion
//
//  Mathematical Background:
//  The naive logsumexp(x) = log(sum(exp(x_i))) overflows for large x values.
//  The stable form is: logsumexp(x) = max(x) + log(sum(exp(x_i - max(x))))
//
//  Primary use cases:
//  - Topic probability distributions
//  - Log-probability normalization
//  - Attention mechanism scores
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Result Types

/// Result from logsumexp row-wise operation.
public struct LogSumExpRowResult: Sendable {
    /// Output buffer containing logsumexp per row [N]
    public let output: any MTLBuffer
    /// Number of rows processed
    public let rowCount: Int
    /// Columns per row
    public let columnCount: Int
    /// Execution time in seconds
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s
    public let throughputGBps: Double

    /// Extract result as Float array.
    public func asArray() -> [Float] {
        let ptr = output.contents().bindMemory(to: Float.self, capacity: rowCount)
        return Array(UnsafeBufferPointer(start: ptr, count: rowCount))
    }
}

/// Result from logsumexp full reduction operation.
public struct LogSumExpReduceResult: Sendable {
    /// The computed logsumexp value
    public let value: Float
    /// Number of elements processed
    public let count: Int
    /// Execution time in seconds
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s
    public let throughputGBps: Double
}

/// Result from softmax operation.
public struct SoftmaxResult: Sendable {
    /// Output buffer containing softmax values [N, D]
    public let output: any MTLBuffer
    /// Number of rows
    public let rowCount: Int
    /// Number of columns
    public let columnCount: Int
    /// Execution time in seconds
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s
    public let throughputGBps: Double

    /// Extract result as 2D Float array.
    public func asArray() -> [[Float]] {
        let totalCount = rowCount * columnCount
        let ptr = output.contents().bindMemory(to: Float.self, capacity: totalCount)
        var result: [[Float]] = []
        result.reserveCapacity(rowCount)
        for i in 0..<rowCount {
            let rowStart = i * columnCount
            let row = Array(UnsafeBufferPointer(start: ptr.advanced(by: rowStart), count: columnCount))
            result.append(row)
        }
        return result
    }

    /// Extract result as flat Float array.
    public func asFlatArray() -> [Float] {
        let totalCount = rowCount * columnCount
        let ptr = output.contents().bindMemory(to: Float.self, capacity: totalCount)
        return Array(UnsafeBufferPointer(start: ptr, count: totalCount))
    }
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for numerically stable log-sum-exp operations.
///
/// Computes the numerically stable log-sum-exp function:
/// ```
/// logsumexp(x) = log(sum(exp(x_i)))
///              = max(x) + log(sum(exp(x_i - max(x))))  // stable form
/// ```
///
/// ## Performance
///
/// Row-wise operations are optimized for matrix inputs where each row represents
/// a probability distribution (e.g., topic scores, attention logits).
///
/// ## Primary Use Cases
///
/// ### Topic Probability Distribution
/// ```swift
/// let topicScores: [[Float]] = ... // [numDocs, numTopics]
/// let topicProbs = try await kernel.softmax(input: topicScores)
/// // Each row now sums to 1.0
/// ```
///
/// ### Log-Probability Normalization
/// ```swift
/// let logProbs: [Float] = ... // Unnormalized log-probabilities
/// let logNormalizer = try await kernel.reduce(input: logProbs)
/// let normalized = logProbs.map { $0 - logNormalizer.value }
/// ```
///
/// ## Kernel Fusion
///
/// Use `encode()` methods to fuse with other operations:
/// ```swift
/// try await context.executeAndWait { _, encoder in
///     // Compute scores first
///     scoreKernel.encode(into: encoder, ...)
///     encoder.memoryBarrier(scope: .buffers)
///
///     // Then compute softmax
///     logSumExpKernel.encodeSoftmax(into: encoder, ...)
/// }
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class LogSumExpKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "LogSumExpKernel"
    public let fusibleWith: [String] = ["L2Distance", "DotProduct", "CosineSimilarity", "Any"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let rowPipeline: any MTLComputePipelineState
    private let rowVectorizedPipeline: any MTLComputePipelineState
    private let reducePass1Pipeline: any MTLComputePipelineState
    private let reducePass2Pipeline: any MTLComputePipelineState
    private let softmaxPipeline: any MTLComputePipelineState
    private let softmaxEfficientPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a LogSumExp kernel.
    ///
    /// - Parameter context: The Metal 4 context to use.
    /// - Throws: If pipeline creation fails.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let rowFunc = library.makeFunction(name: "logsumexp_row_kernel"),
              let rowVecFunc = library.makeFunction(name: "logsumexp_row_vectorized_kernel"),
              let pass1Func = library.makeFunction(name: "logsumexp_reduce_pass1_kernel"),
              let pass2Func = library.makeFunction(name: "logsumexp_reduce_pass2_kernel"),
              let softmaxFunc = library.makeFunction(name: "softmax_row_kernel"),
              let softmaxEffFunc = library.makeFunction(name: "softmax_row_efficient_kernel") else {
            throw VectorError.shaderNotFound(
                name: "LogSumExp kernels. Ensure LogSumExp.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.rowPipeline = try await device.makeComputePipelineState(function: rowFunc)
        self.rowVectorizedPipeline = try await device.makeComputePipelineState(function: rowVecFunc)
        self.reducePass1Pipeline = try await device.makeComputePipelineState(function: pass1Func)
        self.reducePass2Pipeline = try await device.makeComputePipelineState(function: pass2Func)
        self.softmaxPipeline = try await device.makeComputePipelineState(function: softmaxFunc)
        self.softmaxEfficientPipeline = try await device.makeComputePipelineState(function: softmaxEffFunc)
    }

    // MARK: - Warm Up

    /// Warm up the kernel pipelines.
    ///
    /// All pipelines are created in init, so this is a no-op.
    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Row-wise LogSumExp (Encode API)

    /// Encode row-wise logsumexp for kernel fusion.
    ///
    /// Computes `output[i] = logsumexp(input[i, :])` for each row i.
    /// Automatically selects vectorized kernel when d is divisible by 4 and >= 16.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into.
    ///   - input: Input buffer [N, D] (row-major).
    ///   - output: Output buffer [N].
    ///   - n: Number of rows.
    ///   - d: Number of columns.
    /// - Returns: Encoding result for debugging/profiling.
    @discardableResult
    public func encodeRowwise(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        n: Int,
        d: Int
    ) -> Metal4EncodingResult {
        // Use vectorized version if d is divisible by 4 and large enough
        if d % 4 == 0 && d >= 16 {
            encoder.setComputePipelineState(rowVectorizedPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var nU32 = UInt32(n)
            var d4 = UInt32(d / 4)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&d4, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: n, pipeline: rowVectorizedPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "logsumexp_row_vectorized_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        } else {
            encoder.setComputePipelineState(rowPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var nU32 = UInt32(n)
            var dU32 = UInt32(d)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&dU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: n, pipeline: rowPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "logsumexp_row_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        }
    }

    // MARK: - Row-wise LogSumExp (Execute API)

    /// Compute logsumexp along each row of a matrix.
    ///
    /// - Parameters:
    ///   - input: Input buffer [N, D] (row-major).
    ///   - n: Number of rows.
    ///   - d: Number of columns.
    /// - Returns: Result containing buffer of N logsumexp values.
    /// - Throws: If execution fails.
    public func rowwise(
        input: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> LogSumExpRowResult {
        let device = context.device.rawDevice
        let outputSize = n * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "LogSumExp.rowOutput"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeRowwise(into: encoder, input: input, output: outputBuffer, n: n, d: d)
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Calculate throughput: N*D reads + N writes
        let totalBytes = (n * d + n) * MemoryLayout<Float>.size
        let throughputGBps = Double(totalBytes) / (1e9 * executionTime)

        return LogSumExpRowResult(
            output: outputBuffer,
            rowCount: n,
            columnCount: d,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute logsumexp along each row for Swift 2D array.
    ///
    /// - Parameter input: 2D array [N][D].
    /// - Returns: Result containing logsumexp values.
    /// - Throws: If input is empty or execution fails.
    public func rowwise(input: [[Float]]) async throws -> LogSumExpRowResult {
        guard !input.isEmpty, let firstRow = input.first, !firstRow.isEmpty else {
            throw VectorError.invalidInput("Input array is empty")
        }

        let n = input.count
        let d = firstRow.count

        // Validate all rows have same length
        for (i, row) in input.enumerated() {
            guard row.count == d else {
                throw VectorError.invalidInput("Row \(i) has \(row.count) elements, expected \(d)")
            }
        }

        let flat = input.flatMap { $0 }
        let device = context.device.rawDevice

        guard let inputBuffer = flat.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: flat.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "LogSumExp.input"

        return try await rowwise(input: inputBuffer, n: n, d: d)
    }

    // MARK: - Full Reduction (Encode API)

    /// Encode full reduction for kernel fusion.
    ///
    /// Computes logsumexp of entire input array using two-pass algorithm.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - input: Input buffer.
    ///   - partialMax: Buffer for partial max values.
    ///   - partialSum: Buffer for partial sum values.
    ///   - output: Output buffer [1].
    ///   - count: Number of elements.
    ///   - numGroups: Number of threadgroups.
    public func encodeReduce(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        partialMax: any MTLBuffer,
        partialSum: any MTLBuffer,
        output: any MTLBuffer,
        count: Int,
        numGroups: Int
    ) {
        // Pass 1: Compute partial logsumexp per group
        encoder.setComputePipelineState(reducePass1Pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(partialMax, offset: 0, index: 1)
        encoder.setBuffer(partialSum, offset: 0, index: 2)
        var countU32 = UInt32(count)
        var numGroupsU32 = UInt32(numGroups)
        encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )

        encoder.memoryBarrier(scope: .buffers)

        // Pass 2: Combine partials
        encoder.setComputePipelineState(reducePass2Pipeline)
        encoder.setBuffer(partialMax, offset: 0, index: 0)
        encoder.setBuffer(partialSum, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&numGroupsU32, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    // MARK: - Full Reduction (Execute API)

    /// Compute logsumexp of entire array.
    ///
    /// - Parameters:
    ///   - input: Input buffer.
    ///   - count: Number of elements.
    /// - Returns: Result containing single logsumexp value.
    /// - Throws: If execution fails.
    public func reduce(
        input: any MTLBuffer,
        count: Int
    ) async throws -> LogSumExpReduceResult {
        guard count > 0 else {
            throw VectorError.invalidInput("Input count must be positive")
        }

        let device = context.device.rawDevice

        // For small arrays, use single threadgroup
        let numGroups = min((count + 255) / 256, 256)

        let partialMaxSize = numGroups * MemoryLayout<Float>.size
        let partialSumSize = numGroups * MemoryLayout<Float>.size
        let outputSize = MemoryLayout<Float>.size

        guard let partialMax = device.makeBuffer(length: partialMaxSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: partialMaxSize)
        }
        partialMax.label = "LogSumExp.partialMax"

        guard let partialSum = device.makeBuffer(length: partialSumSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: partialSumSize)
        }
        partialSum.label = "LogSumExp.partialSum"

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "LogSumExp.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeReduce(
                into: encoder,
                input: input,
                partialMax: partialMax,
                partialSum: partialSum,
                output: outputBuffer,
                count: count,
                numGroups: numGroups
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Read result
        let value = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee

        // Calculate throughput (approximate: 2 reads per element for two passes)
        let totalBytes = count * 2 * MemoryLayout<Float>.size
        let throughputGBps = Double(totalBytes) / (1e9 * executionTime)

        return LogSumExpReduceResult(
            value: value,
            count: count,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute logsumexp of Swift array.
    ///
    /// - Parameter input: Input array.
    /// - Returns: Result containing logsumexp value.
    /// - Throws: If array is empty or execution fails.
    public func reduce(input: [Float]) async throws -> LogSumExpReduceResult {
        guard !input.isEmpty else {
            throw VectorError.invalidInput("Input array is empty")
        }

        let device = context.device.rawDevice

        guard let inputBuffer = input.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: input.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "LogSumExp.input"

        return try await reduce(input: inputBuffer, count: input.count)
    }

    // MARK: - Softmax (Encode API)

    /// Encode softmax operation for kernel fusion.
    ///
    /// Computes softmax(x)_i = exp(x_i - logsumexp(x)) along rows.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - input: Input buffer [N, D].
    ///   - output: Output buffer [N, D].
    ///   - n: Number of rows.
    ///   - d: Number of columns.
    ///   - useEfficient: If true, uses one-thread-per-row variant (better for small d).
    /// - Returns: Encoding result.
    @discardableResult
    public func encodeSoftmax(
        into encoder: any MTLComputeCommandEncoder,
        input: any MTLBuffer,
        output: any MTLBuffer,
        n: Int,
        d: Int,
        useEfficient: Bool = false
    ) -> Metal4EncodingResult {
        var nU32 = UInt32(n)
        var dU32 = UInt32(d)

        if useEfficient || d <= 64 {
            // Use efficient kernel: one thread per row
            encoder.setComputePipelineState(softmaxEfficientPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&dU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: n, pipeline: softmaxEfficientPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "softmax_row_efficient_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        } else {
            // Use parallel kernel: one thread per element
            encoder.setComputePipelineState(softmaxPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&dU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.forDistanceKernel(
                numQueries: d,
                numDatabase: n,
                pipeline: softmaxPipeline
            )
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "softmax_row_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        }
    }

    // MARK: - Softmax (Execute API)

    /// Compute softmax along each row.
    ///
    /// softmax(x)_i = exp(x_i) / sum(exp(x_j))
    ///             = exp(x_i - logsumexp(x))
    ///
    /// - Parameters:
    ///   - input: Input buffer [N, D].
    ///   - n: Number of rows.
    ///   - d: Number of columns.
    /// - Returns: Result containing softmax output [N, D].
    /// - Throws: If execution fails.
    public func softmax(
        input: any MTLBuffer,
        n: Int,
        d: Int
    ) async throws -> SoftmaxResult {
        let device = context.device.rawDevice
        let outputSize = n * d * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "Softmax.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeSoftmax(into: encoder, input: input, output: outputBuffer, n: n, d: d)
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Calculate throughput: N*D reads + N*D writes
        let totalBytes = 2 * n * d * MemoryLayout<Float>.size
        let throughputGBps = Double(totalBytes) / (1e9 * executionTime)

        return SoftmaxResult(
            output: outputBuffer,
            rowCount: n,
            columnCount: d,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute softmax for Swift 2D array.
    ///
    /// - Parameter input: 2D array [N][D].
    /// - Returns: Result containing softmax values.
    /// - Throws: If input is empty or execution fails.
    public func softmax(input: [[Float]]) async throws -> SoftmaxResult {
        guard !input.isEmpty, let firstRow = input.first, !firstRow.isEmpty else {
            throw VectorError.invalidInput("Input array is empty")
        }

        let n = input.count
        let d = firstRow.count

        // Validate all rows have same length
        for (i, row) in input.enumerated() {
            guard row.count == d else {
                throw VectorError.invalidInput("Row \(i) has \(row.count) elements, expected \(d)")
            }
        }

        let flat = input.flatMap { $0 }
        let device = context.device.rawDevice

        guard let inputBuffer = flat.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: flat.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "Softmax.input"

        return try await softmax(input: inputBuffer, n: n, d: d)
    }

    // MARK: - Convenience APIs

    /// Compute row-wise logsumexp and return as array.
    ///
    /// - Parameter input: 2D array [N][D].
    /// - Returns: Array of N logsumexp values.
    public func rowwiseArray(input: [[Float]]) async throws -> [Float] {
        let result = try await rowwise(input: input)
        return result.asArray()
    }

    /// Compute logsumexp of array and return scalar.
    ///
    /// - Parameter input: Input array.
    /// - Returns: Logsumexp value.
    public func reduceValue(input: [Float]) async throws -> Float {
        let result = try await reduce(input: input)
        return result.value
    }

    /// Compute softmax and return as 2D array.
    ///
    /// - Parameter input: 2D array [N][D].
    /// - Returns: 2D array of softmax values.
    public func softmaxArray(input: [[Float]]) async throws -> [[Float]] {
        let result = try await softmax(input: input)
        return result.asArray()
    }
}
