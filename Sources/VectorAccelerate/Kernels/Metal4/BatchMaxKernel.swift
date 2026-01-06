//
//  BatchMaxKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for element-wise maximum operations.
//
//  Features:
//  - Three-array max: max(a, b, c) - for mutual reachability
//  - Two-array max: max(a, b)
//  - In-place variants
//  - Vectorized float4 versions for better memory throughput
//  - FusibleKernel conformance for kernel fusion
//
//  Primary use case: Mutual reachability distance computation
//    mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Result Type

/// Result from batch max operation.
public struct BatchMaxResult: Sendable {
    /// Output buffer containing element-wise maximum
    public let output: any MTLBuffer
    /// Number of elements processed
    public let count: Int
    /// Execution time in seconds
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s
    public let throughputGBps: Double

    /// Extract result as Float array.
    public func asArray() -> [Float] {
        let ptr = output.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for element-wise maximum operations.
///
/// Computes element-wise maximum of two or three arrays with GPU acceleration.
/// Includes vectorized variants for optimal memory throughput.
///
/// ## Performance
///
/// This kernel is **memory-bound** (3 reads + 1 write per element = 16 bytes).
/// Vectorization with float4 improves memory coalescing and throughput.
///
/// ## Primary Use Case: Mutual Reachability
///
/// ```swift
/// // Mutual reachability: max(core_dist[a], core_dist[b], euclidean_dist(a, b))
/// let mutualReach = try await kernel.max3(
///     a: coreDistancesA,   // Expanded core distances for rows
///     b: coreDistancesB,   // Expanded core distances for columns
///     c: euclideanDist,    // Pairwise euclidean distances
///     count: n * n
/// )
/// ```
///
/// ## Kernel Fusion
///
/// Use `encode()` methods to fuse with other operations:
/// ```swift
/// try await context.executeAndWait { _, encoder in
///     // Compute distances first
///     distanceKernel.encode(into: encoder, ...)
///     encoder.memoryBarrier(scope: .buffers)
///
///     // Then compute mutual reachability max
///     batchMaxKernel.encode(into: encoder, a: coreA, b: coreB, c: distances, output: result, count: n)
/// }
/// ```
public final class BatchMaxKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "BatchMaxKernel"
    public let fusibleWith: [String] = ["L2Distance", "MutualReachability", "Any"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let max3Pipeline: any MTLComputePipelineState
    private let max3VectorizedPipeline: any MTLComputePipelineState
    private let max2Pipeline: any MTLComputePipelineState
    private let max2VectorizedPipeline: any MTLComputePipelineState
    private let maxInplacePipeline: any MTLComputePipelineState
    private let max3InplacePipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a BatchMax kernel.
    ///
    /// - Parameter context: The Metal 4 context to use.
    /// - Throws: If pipeline creation fails.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let max3Func = library.makeFunction(name: "batch_max3_kernel"),
              let max3VecFunc = library.makeFunction(name: "batch_max3_vectorized_kernel"),
              let max2Func = library.makeFunction(name: "batch_max2_kernel"),
              let max2VecFunc = library.makeFunction(name: "batch_max2_vectorized_kernel"),
              let inplaceFunc = library.makeFunction(name: "batch_max_inplace_kernel"),
              let inplace3Func = library.makeFunction(name: "batch_max3_inplace_kernel") else {
            throw VectorError.shaderNotFound(
                name: "BatchMax kernels. Ensure BatchMax.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.max3Pipeline = try await device.makeComputePipelineState(function: max3Func)
        self.max3VectorizedPipeline = try await device.makeComputePipelineState(function: max3VecFunc)
        self.max2Pipeline = try await device.makeComputePipelineState(function: max2Func)
        self.max2VectorizedPipeline = try await device.makeComputePipelineState(function: max2VecFunc)
        self.maxInplacePipeline = try await device.makeComputePipelineState(function: inplaceFunc)
        self.max3InplacePipeline = try await device.makeComputePipelineState(function: inplace3Func)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Three-Array Maximum (Encode API)

    /// Encode max3 operation for kernel fusion.
    ///
    /// Computes `output[i] = max(a[i], b[i], c[i])`.
    /// Automatically selects vectorized kernel when count is divisible by 4 and >= 16.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder to encode into.
    ///   - a: First input buffer.
    ///   - b: Second input buffer.
    ///   - c: Third input buffer.
    ///   - output: Output buffer.
    ///   - count: Number of elements.
    /// - Returns: Encoding result for debugging/profiling.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        a: any MTLBuffer,
        b: any MTLBuffer,
        c: any MTLBuffer,
        output: any MTLBuffer,
        count: Int
    ) -> Metal4EncodingResult {
        // Use vectorized version if count is divisible by 4 and large enough
        if count % 4 == 0 && count >= 16 {
            encoder.setComputePipelineState(max3VectorizedPipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var count4 = UInt32(count / 4)
            encoder.setBytes(&count4, length: MemoryLayout<UInt32>.size, index: 4)

            let config = Metal4ThreadConfiguration.linear(count: count / 4, pipeline: max3VectorizedPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "batch_max3_vectorized_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        } else {
            encoder.setComputePipelineState(max3Pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 4)

            let config = Metal4ThreadConfiguration.linear(count: count, pipeline: max3Pipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "batch_max3_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        }
    }

    // MARK: - Three-Array Maximum (Execute API)

    /// Compute element-wise max(a, b, c).
    ///
    /// - Parameters:
    ///   - a: First input buffer.
    ///   - b: Second input buffer.
    ///   - c: Third input buffer.
    ///   - count: Number of elements.
    /// - Returns: Result containing output buffer and performance metrics.
    /// - Throws: If execution fails.
    public func max3(
        a: any MTLBuffer,
        b: any MTLBuffer,
        c: any MTLBuffer,
        count: Int
    ) async throws -> BatchMaxResult {
        let device = context.device.rawDevice
        let outputSize = count * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "BatchMax.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(into: encoder, a: a, b: b, c: c, output: outputBuffer, count: count)
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Calculate throughput: 3 reads + 1 write = 16 bytes per element
        let totalBytes = count * 4 * MemoryLayout<Float>.size
        let throughputGBps = Double(totalBytes) / (1e9 * executionTime)

        return BatchMaxResult(
            output: outputBuffer,
            count: count,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute element-wise max(a, b, c) for Swift arrays.
    ///
    /// - Parameters:
    ///   - a: First input array.
    ///   - b: Second input array.
    ///   - c: Third input array.
    /// - Returns: Result containing output and performance metrics.
    /// - Throws: If arrays have different sizes or execution fails.
    public func max3(
        a: [Float],
        b: [Float],
        c: [Float]
    ) async throws -> BatchMaxResult {
        guard a.count == b.count && b.count == c.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }
        guard !a.isEmpty else {
            throw VectorError.invalidInput("Input arrays are empty")
        }

        let device = context.device.rawDevice
        let count = a.count

        guard let aBuffer = a.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: count * MemoryLayout<Float>.size)
        }
        aBuffer.label = "BatchMax.a"

        guard let bBuffer = b.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: count * MemoryLayout<Float>.size)
        }
        bBuffer.label = "BatchMax.b"

        guard let cBuffer = c.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: count * MemoryLayout<Float>.size)
        }
        cBuffer.label = "BatchMax.c"

        return try await max3(a: aBuffer, b: bBuffer, c: cBuffer, count: count)
    }

    // MARK: - Two-Array Maximum (Encode API)

    /// Encode max2 operation for kernel fusion.
    ///
    /// Computes `output[i] = max(a[i], b[i])`.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - a: First input buffer.
    ///   - b: Second input buffer.
    ///   - output: Output buffer.
    ///   - count: Number of elements.
    /// - Returns: Encoding result.
    @discardableResult
    public func encodeMax2(
        into encoder: any MTLComputeCommandEncoder,
        a: any MTLBuffer,
        b: any MTLBuffer,
        output: any MTLBuffer,
        count: Int
    ) -> Metal4EncodingResult {
        if count % 4 == 0 && count >= 16 {
            encoder.setComputePipelineState(max2VectorizedPipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var count4 = UInt32(count / 4)
            encoder.setBytes(&count4, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: count / 4, pipeline: max2VectorizedPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "batch_max2_vectorized_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        } else {
            encoder.setComputePipelineState(max2Pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: count, pipeline: max2Pipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "batch_max2_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        }
    }

    // MARK: - Two-Array Maximum (Execute API)

    /// Compute element-wise max(a, b).
    ///
    /// - Parameters:
    ///   - a: First input buffer.
    ///   - b: Second input buffer.
    ///   - count: Number of elements.
    /// - Returns: Result containing output buffer.
    /// - Throws: If execution fails.
    public func max2(
        a: any MTLBuffer,
        b: any MTLBuffer,
        count: Int
    ) async throws -> BatchMaxResult {
        let device = context.device.rawDevice
        let outputSize = count * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "BatchMax.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encodeMax2(into: encoder, a: a, b: b, output: outputBuffer, count: count)
        }
        let executionTime = CACurrentMediaTime() - startTime

        // 2 reads + 1 write = 12 bytes per element
        let totalBytes = count * 3 * MemoryLayout<Float>.size
        let throughputGBps = Double(totalBytes) / (1e9 * executionTime)

        return BatchMaxResult(
            output: outputBuffer,
            count: count,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute element-wise max(a, b) for Swift arrays.
    public func max2(
        a: [Float],
        b: [Float]
    ) async throws -> BatchMaxResult {
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }
        guard !a.isEmpty else {
            throw VectorError.invalidInput("Input arrays are empty")
        }

        let device = context.device.rawDevice
        let count = a.count

        guard let aBuffer = a.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: count * MemoryLayout<Float>.size)
        }
        aBuffer.label = "BatchMax.a"

        guard let bBuffer = b.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: count * MemoryLayout<Float>.size)
        }
        bBuffer.label = "BatchMax.b"

        return try await max2(a: aBuffer, b: bBuffer, count: count)
    }

    // MARK: - In-Place Operations

    /// Compute a[i] = max(a[i], b[i]) in-place.
    ///
    /// - Parameters:
    ///   - a: Input/output buffer (modified in place).
    ///   - b: Second input buffer.
    ///   - count: Number of elements.
    /// - Throws: If execution fails.
    public func maxInplace(
        a: any MTLBuffer,
        b: any MTLBuffer,
        count: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(maxInplacePipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 2)

            let config = Metal4ThreadConfiguration.linear(count: count, pipeline: maxInplacePipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
    }

    /// Compute a[i] = max(a[i], b[i], c[i]) in-place.
    ///
    /// - Parameters:
    ///   - a: Input/output buffer (modified in place).
    ///   - b: Second input buffer.
    ///   - c: Third input buffer.
    ///   - count: Number of elements.
    /// - Throws: If execution fails.
    public func max3Inplace(
        a: any MTLBuffer,
        b: any MTLBuffer,
        c: any MTLBuffer,
        count: Int
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(max3InplacePipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            var countU32 = UInt32(count)
            encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.size, index: 3)

            let config = Metal4ThreadConfiguration.linear(count: count, pipeline: max3InplacePipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }
    }

    // MARK: - Convenience Array APIs

    /// Compute max(a, b, c) and return as array.
    public func max3Array(
        a: [Float],
        b: [Float],
        c: [Float]
    ) async throws -> [Float] {
        let result = try await max3(a: a, b: b, c: c)
        return result.asArray()
    }

    /// Compute max(a, b) and return as array.
    public func max2Array(
        a: [Float],
        b: [Float]
    ) async throws -> [Float] {
        let result = try await max2(a: a, b: b)
        return result.asArray()
    }
}
