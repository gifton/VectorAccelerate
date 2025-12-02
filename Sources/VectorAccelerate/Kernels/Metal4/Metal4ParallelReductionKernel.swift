//
//  Metal4ParallelReductionKernel.swift
//  VectorAccelerate
//
//  Metal 4 Parallel Reduction kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 5, Priority 6
//
//  Features:
//  - Multi-pass parallel reduction
//  - Sum, Min, Max, ArgMin, ArgMax operations
//  - Comprehensive statistics computation
//  - Thread-safe for concurrent encoding

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Reduction Operations

/// Supported reduction operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4ReductionOperation: UInt8, Sendable {
    case sum = 0
    case minimum = 2
    case maximum = 3
    case argMin = 4
    case argMax = 5

    /// Initial value for reduction
    public var initialValue: Float {
        switch self {
        case .sum: return 0.0
        case .minimum, .argMin: return Float.infinity
        case .maximum, .argMax: return -Float.infinity
        }
    }

    /// Whether operation returns an index
    public var returnsIndex: Bool {
        switch self {
        case .argMin, .argMax: return true
        default: return false
        }
    }
}

// MARK: - Result Types

/// Result from reduction operation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4ReductionResult: Sendable {
    /// Result value
    public let value: Float
    /// Result index (for argMin/argMax)
    public let index: Int?
    /// Execution time
    public let executionTime: TimeInterval
    /// Number of reduction passes used
    public let passes: Int
}

/// Comprehensive statistics result.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4Statistics: Sendable {
    public let count: Int
    public let sum: Float
    public let mean: Float
    public let minimum: Float
    public let maximum: Float
    public let minIndex: Int
    public let maxIndex: Int
    public let executionTime: TimeInterval

    public var range: Float { maximum - minimum }
}

// MARK: - Parameters

/// Parameters for reduction kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
internal struct Metal4ReductionParams: Sendable {
    var numElements: UInt32
    var stride: UInt32
    var operation: UInt8
    var returnIndex: UInt8
    var padding: (UInt8, UInt8) = (0, 0)
    var initialValue: Float
}

// MARK: - Kernel Implementation

/// Metal 4 Parallel Reduction kernel.
///
/// Performs efficient parallel aggregation operations across large datasets
/// using a multi-pass tree reduction algorithm.
///
/// ## Algorithm
///
/// Uses tree reduction with O(log n) passes:
/// - Pass 1: Reduce n → n/threadgroupSize partial results
/// - Pass 2: Reduce partials → smaller set
/// - Continue until single result
///
/// ## Supported Operations
///
/// - **sum**: Add all elements
/// - **minimum**: Find smallest element
/// - **maximum**: Find largest element
/// - **argMin**: Find index of smallest element
/// - **argMax**: Find index of largest element
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4ParallelReductionKernel(context: context)
///
/// // Find minimum with its index
/// let (value, index, _) = try await kernel.reduce(array, operation: .argMin)
///
/// // Compute multiple statistics efficiently
/// let stats = try await kernel.computeStatistics(array)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4ParallelReductionKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4ParallelReductionKernel"

    // MARK: - Constants

    private let threadgroupSize: Int

    // MARK: - Pipelines

    private let reductionPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Parallel Reduction kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let reduceFunc = library.makeFunction(name: "parallel_reduce_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Parallel reduction kernel. Ensure BasicOperations.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.reductionPipeline = try await device.makeComputePipelineState(function: reduceFunc)

        // Determine optimal threadgroup size (must be power of 2 and >= 32)
        let maxThreads = reductionPipeline.maxTotalThreadsPerThreadgroup
        var tgs = 32
        while (tgs * 2) <= maxThreads {
            tgs *= 2
        }
        self.threadgroupSize = tgs
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline created in init
    }

    // MARK: - Low-Level API

    /// Perform reduction with buffers (multi-pass).
    public func reduce(
        input: any MTLBuffer,
        operation: Metal4ReductionOperation,
        count: Int
    ) async throws -> Metal4ReductionResult {
        guard count > 0 else {
            throw VectorError.invalidInput("Count must be > 0")
        }

        let device = context.device.rawDevice
        var currentInput = input
        var currentIndices: (any MTLBuffer)? = nil
        var currentCount = count
        var passes = 0

        let startTime = CACurrentMediaTime()

        // Multi-pass reduction
        while currentCount > 1 {
            let numGroups = (currentCount + threadgroupSize - 1) / threadgroupSize

            // Create output buffers (private memory for efficiency)
            guard let outputValues = device.makeBuffer(
                length: numGroups * MemoryLayout<Float>.size,
                options: .storageModePrivate
            ) else {
                throw VectorError.bufferAllocationFailed(size: numGroups * MemoryLayout<Float>.size)
            }

            var outputIndices: (any MTLBuffer)? = nil
            if operation.returnsIndex {
                outputIndices = device.makeBuffer(
                    length: numGroups * MemoryLayout<UInt32>.size,
                    options: .storageModePrivate
                )
            }

            // Execute pass
            try await encodeReductionPass(
                inputValues: currentInput,
                inputIndices: currentIndices,
                outputValues: outputValues,
                outputIndices: outputIndices,
                count: currentCount,
                operation: operation
            )

            currentInput = outputValues
            currentIndices = outputIndices
            currentCount = numGroups
            passes += 1
        }

        // Blit final result to shared memory for CPU access
        let result = try await blitToShared(
            valueBuffer: currentInput,
            indexBuffer: currentIndices
        )

        let executionTime = CACurrentMediaTime() - startTime

        return Metal4ReductionResult(
            value: result.value,
            index: result.index,
            executionTime: executionTime,
            passes: passes
        )
    }

    // MARK: - High-Level API

    /// Reduce array with specified operation.
    public func reduce(
        _ array: [Float],
        operation: Metal4ReductionOperation
    ) async throws -> Metal4ReductionResult {
        guard !array.isEmpty else {
            throw VectorError.invalidInput("Empty input array")
        }

        let device = context.device.rawDevice
        guard let inputBuffer = device.makeBuffer(
            bytes: array,
            length: array.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: array.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "Reduction.input"

        return try await reduce(input: inputBuffer, operation: operation, count: array.count)
    }

    /// Compute comprehensive statistics for array.
    public func computeStatistics(_ array: [Float]) async throws -> Metal4Statistics {
        guard !array.isEmpty else {
            throw VectorError.invalidInput("Empty input array")
        }

        let device = context.device.rawDevice
        guard let inputBuffer = device.makeBuffer(
            bytes: array,
            length: array.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: array.count * MemoryLayout<Float>.size)
        }
        inputBuffer.label = "Statistics.input"

        let startTime = CACurrentMediaTime()

        // Run multiple reductions
        async let sumResult = reduce(input: inputBuffer, operation: .sum, count: array.count)
        async let minResult = reduce(input: inputBuffer, operation: .argMin, count: array.count)
        async let maxResult = reduce(input: inputBuffer, operation: .argMax, count: array.count)

        let (sum, argMin, argMax) = try await (sumResult, minResult, maxResult)

        let executionTime = CACurrentMediaTime() - startTime

        return Metal4Statistics(
            count: array.count,
            sum: sum.value,
            mean: sum.value / Float(array.count),
            minimum: argMin.value,
            maximum: argMax.value,
            minIndex: argMin.index ?? 0,
            maxIndex: argMax.index ?? 0,
            executionTime: executionTime
        )
    }

    /// Reduce using VectorProtocol types.
    public func reduce<V: VectorProtocol>(
        _ vector: V,
        operation: Metal4ReductionOperation
    ) async throws -> Metal4ReductionResult where V.Scalar == Float {
        guard vector.count > 0 else {
            throw VectorError.invalidInput("Empty vector")
        }

        let array: [Float] = vector.withUnsafeBufferPointer { Array($0) }
        return try await reduce(array, operation: operation)
    }

    // MARK: - Convenience Methods

    /// Compute sum of array.
    public func sum(_ array: [Float]) async throws -> Float {
        let result = try await reduce(array, operation: .sum)
        return result.value
    }

    /// Find minimum value.
    public func minimum(_ array: [Float]) async throws -> Float {
        let result = try await reduce(array, operation: .minimum)
        return result.value
    }

    /// Find maximum value.
    public func maximum(_ array: [Float]) async throws -> Float {
        let result = try await reduce(array, operation: .maximum)
        return result.value
    }

    /// Find minimum value and its index.
    public func argMin(_ array: [Float]) async throws -> (value: Float, index: Int) {
        let result = try await reduce(array, operation: .argMin)
        return (result.value, result.index ?? 0)
    }

    /// Find maximum value and its index.
    public func argMax(_ array: [Float]) async throws -> (value: Float, index: Int) {
        let result = try await reduce(array, operation: .argMax)
        return (result.value, result.index ?? 0)
    }

    // MARK: - Private Helpers

    private func encodeReductionPass(
        inputValues: any MTLBuffer,
        inputIndices: (any MTLBuffer)?,
        outputValues: any MTLBuffer,
        outputIndices: (any MTLBuffer)?,
        count: Int,
        operation: Metal4ReductionOperation
    ) async throws {
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(reductionPipeline)
            encoder.label = "ReductionPass (count=\(count))"

            var params = Metal4ReductionParams(
                numElements: UInt32(count),
                stride: 1,
                operation: operation.rawValue,
                returnIndex: operation.returnsIndex ? 1 : 0,
                initialValue: operation.initialValue
            )

            encoder.setBuffer(inputValues, offset: 0, index: 0)
            encoder.setBuffer(inputIndices, offset: 0, index: 1)
            encoder.setBuffer(outputValues, offset: 0, index: 2)
            encoder.setBuffer(outputIndices, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<Metal4ReductionParams>.stride, index: 4)

            // Shared memory for indexed values
            let sharedMemorySize = threadgroupSize * (MemoryLayout<Float>.size + MemoryLayout<UInt32>.size)
            encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)

            let numThreadgroups = (count + threadgroupSize - 1) / threadgroupSize
            encoder.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
            )
        }
    }

    private func blitToShared(
        valueBuffer: any MTLBuffer,
        indexBuffer: (any MTLBuffer)?
    ) async throws -> (value: Float, index: Int?) {
        let device = context.device.rawDevice

        guard let sharedValue = device.makeBuffer(
            length: MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: MemoryLayout<Float>.size)
        }

        let sharedIndex: (any MTLBuffer)?
        if indexBuffer != nil {
            sharedIndex = device.makeBuffer(
                length: MemoryLayout<UInt32>.size,
                options: .storageModeShared
            )
        } else {
            sharedIndex = nil
        }

        // Use executeBlitAndWait for blit operations
        // (cannot use executeAndWait which creates a compute encoder)
        try await context.executeBlitAndWait { _, blitEncoder in
            blitEncoder.copy(
                from: valueBuffer,
                sourceOffset: 0,
                to: sharedValue,
                destinationOffset: 0,
                size: MemoryLayout<Float>.size
            )

            if let src = indexBuffer, let dst = sharedIndex {
                blitEncoder.copy(
                    from: src,
                    sourceOffset: 0,
                    to: dst,
                    destinationOffset: 0,
                    size: MemoryLayout<UInt32>.size
                )
            }
        }

        let value = sharedValue.contents().bindMemory(to: Float.self, capacity: 1).pointee
        let index: Int? = sharedIndex.flatMap { buffer in
            Int(buffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
        }

        return (value, index)
    }
}
