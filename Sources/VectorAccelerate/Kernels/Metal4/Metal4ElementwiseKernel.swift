//
//  Metal4ElementwiseKernel.swift
//  VectorAccelerate
//
//  Metal 4 Elementwise Operations kernel with ArgumentTable support.
//
//  Phase 5: Kernel Migrations - Batch 5, Priority 6
//
//  Features:
//  - Binary operations (add, subtract, multiply, divide, power, max, min)
//  - Scalar operations (addScalar, multiplyScalar, powerScalar, clamp)
//  - Unary operations (abs, square, sqrt, reciprocal, negate, exp, log)
//  - In-place and out-of-place variants
//  - Fusible with any pipeline as pre/post processing

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Operation Enum

/// Supported element-wise operations.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4ElementwiseOperation: Sendable {
    // Binary operations (require two input buffers)
    case add
    case subtract
    case multiply
    case divide
    case power
    case maximum
    case minimum

    // Scalar operations (apply scalar to each element)
    case addScalar(Float)
    case multiplyScalar(Float)
    case powerScalar(Float)
    case clamp(min: Float, max: Float)

    // Unary operations (single input)
    case absolute
    case square
    case sqrt
    case reciprocal
    case negate
    case exp
    case log

    /// Raw value for shader dispatch.
    public var rawValue: UInt8 {
        switch self {
        case .add: return 0
        case .subtract: return 1
        case .multiply: return 2
        case .divide: return 3
        case .power: return 4
        case .maximum: return 5
        case .minimum: return 6
        case .addScalar: return 10
        case .multiplyScalar: return 11
        case .powerScalar: return 12
        case .clamp: return 13
        case .absolute: return 20
        case .square: return 21
        case .sqrt: return 22
        case .reciprocal: return 23
        case .negate: return 24
        case .exp: return 25
        case .log: return 26
        }
    }

    /// Whether operation requires a second input buffer.
    public var isBinary: Bool {
        switch self {
        case .add, .subtract, .multiply, .divide, .power, .maximum, .minimum:
            return true
        default:
            return false
        }
    }

    /// Whether operation uses scalar values.
    public var isScalar: Bool {
        switch self {
        case .addScalar, .multiplyScalar, .powerScalar, .clamp:
            return true
        default:
            return false
        }
    }

    /// Whether operation is unary (single input, no scalar).
    public var isUnary: Bool {
        switch self {
        case .absolute, .square, .sqrt, .reciprocal, .negate, .exp, .log:
            return true
        default:
            return false
        }
    }
}

// MARK: - Parameters

/// Parameters for elementwise kernel.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4ElementwiseParameters: Sendable {
    /// Number of elements to process
    public let numElements: UInt32
    /// Stride for input A
    public let strideA: UInt32
    /// Stride for input B
    public let strideB: UInt32
    /// Stride for output
    public let strideOutput: UInt32
    /// Primary scalar value (for scalar operations)
    public let scalarValue: Float
    /// Secondary scalar value (for clamp min)
    public let scalarValue2: Float
    /// Tertiary scalar value (for clamp max)
    public let scalarValue3: Float
    /// Operation type
    public let operation: UInt8
    /// Use fast math approximations
    public let useFastMath: UInt8
    /// Padding for alignment
    private let padding: (UInt8, UInt8) = (0, 0)

    public init(
        numElements: Int,
        operation: Metal4ElementwiseOperation,
        useFastMath: Bool = false,
        strideA: Int = 1,
        strideB: Int = 1,
        strideOutput: Int = 1
    ) {
        self.numElements = UInt32(numElements)
        self.strideA = UInt32(strideA)
        self.strideB = UInt32(strideB)
        self.strideOutput = UInt32(strideOutput)
        self.operation = operation.rawValue
        self.useFastMath = useFastMath ? 1 : 0

        // Extract scalar values from operation
        switch operation {
        case .addScalar(let v), .multiplyScalar(let v), .powerScalar(let v):
            self.scalarValue = v
            self.scalarValue2 = 0
            self.scalarValue3 = 0
        case .clamp(let minV, let maxV):
            self.scalarValue = 0
            self.scalarValue2 = minV
            self.scalarValue3 = maxV
        default:
            self.scalarValue = 0
            self.scalarValue2 = 0
            self.scalarValue3 = 0
        }
    }
}

// MARK: - Result Type

/// Result from elementwise operation.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct Metal4ElementwiseResult: Sendable {
    /// Output buffer
    public let output: any MTLBuffer
    /// Number of elements
    public let numElements: Int
    /// Execution time
    public let executionTime: TimeInterval
    /// Throughput in GB/s
    public let throughputGBps: Double

    /// Extract result as array.
    public func asArray() -> [Float] {
        let ptr = output.contents().bindMemory(to: Float.self, capacity: numElements)
        return Array(UnsafeBufferPointer(start: ptr, count: numElements))
    }
}

// MARK: - Kernel Implementation

/// Metal 4 Elementwise Operations kernel.
///
/// Performs mathematical operations on vectors element-by-element with
/// GPU acceleration and support for kernel fusion.
///
/// ## Operation Types
///
/// - **Binary**: Combine two buffers (add, subtract, multiply, etc.)
/// - **Scalar**: Apply scalar to each element (addScalar, clamp, etc.)
/// - **Unary**: Transform each element (abs, sqrt, exp, etc.)
///
/// ## Fusion Pattern
///
/// Commonly used as pre/post processing in pipelines:
/// ```swift
/// try await context.executeAndWait { _, encoder in
///     // Scale inputs before distance
///     elementwiseKernel.encode(into: encoder, ..., operation: .multiplyScalar(0.5))
///     encoder.memoryBarrier(scope: .buffers)
///     distanceKernel.encode(into: encoder, ...)
/// }
/// ```
///
/// ## Usage
///
/// ```swift
/// let kernel = try await Metal4ElementwiseKernel(context: context)
///
/// // Binary operation
/// let sum = try await kernel.execute(a, b, operation: .add)
///
/// // Scalar operation
/// let scaled = try await kernel.execute(a, operation: .multiplyScalar(2.0))
///
/// // In-place operation
/// try await kernel.executeInPlace(&buffer, operation: .absolute)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class Metal4ElementwiseKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "Metal4ElementwiseKernel"
    public let fusibleWith: [String] = ["Any"]  // Can fuse with any kernel
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let pipelineOutOfPlace: any MTLComputePipelineState
    private let pipelineInPlace: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a Metal 4 Elementwise kernel.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let outOfPlaceFunc = library.makeFunction(name: "elementwise_operation_kernel"),
              let inPlaceFunc = library.makeFunction(name: "elementwise_inplace_kernel") else {
            throw VectorError.shaderNotFound(
                name: "Elementwise kernels. Ensure BasicOperations.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.pipelineOutOfPlace = try await device.makeComputePipelineState(function: outOfPlaceFunc)
        self.pipelineInPlace = try await device.makeComputePipelineState(function: inPlaceFunc)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API (Out-of-Place)

    /// Encode elementwise operation into an existing encoder.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        inputA: any MTLBuffer,
        inputB: (any MTLBuffer)?,
        output: any MTLBuffer,
        parameters: Metal4ElementwiseParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipelineOutOfPlace)
        encoder.label = "Elementwise (op=\(parameters.operation))"

        encoder.setBuffer(inputA, offset: 0, index: 0)
        encoder.setBuffer(inputB, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Metal4ElementwiseParameters>.stride, index: 3)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numElements),
            pipeline: pipelineOutOfPlace
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "elementwise_operation_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Encode API (In-Place)

    /// Encode in-place elementwise operation into an existing encoder.
    @discardableResult
    public func encodeInPlace(
        into encoder: any MTLComputeCommandEncoder,
        data: any MTLBuffer,
        operand: (any MTLBuffer)?,
        parameters: Metal4ElementwiseParameters
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(pipelineInPlace)
        encoder.label = "ElementwiseInPlace (op=\(parameters.operation))"

        encoder.setBuffer(data, offset: 0, index: 0)
        encoder.setBuffer(operand, offset: 0, index: 1)

        var params = parameters
        encoder.setBytes(&params, length: MemoryLayout<Metal4ElementwiseParameters>.stride, index: 2)

        let config = Metal4ThreadConfiguration.linear(
            count: Int(parameters.numElements),
            pipeline: pipelineInPlace
        )

        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "elementwise_inplace_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API

    /// Execute elementwise operation as standalone operation.
    public func execute(
        inputA: any MTLBuffer,
        inputB: (any MTLBuffer)?,
        parameters: Metal4ElementwiseParameters
    ) async throws -> Metal4ElementwiseResult {
        let device = context.device.rawDevice
        let numElements = Int(parameters.numElements)

        let outputSize = numElements * MemoryLayout<Float>.size
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: outputSize)
        }
        outputBuffer.label = "Elementwise.output"

        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                inputA: inputA,
                inputB: inputB,
                output: outputBuffer,
                parameters: parameters
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Calculate throughput (read input(s) + write output)
        let bytesRead = numElements * MemoryLayout<Float>.size * (inputB != nil ? 2 : 1)
        let bytesWritten = numElements * MemoryLayout<Float>.size
        let throughputGBps = Double(bytesRead + bytesWritten) / (1e9 * executionTime)

        return Metal4ElementwiseResult(
            output: outputBuffer,
            numElements: numElements,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    // MARK: - High-Level API

    /// Perform binary operation on two arrays.
    public func execute(
        _ a: [Float],
        _ b: [Float],
        operation: Metal4ElementwiseOperation,
        useFastMath: Bool = false
    ) async throws -> Metal4ElementwiseResult {
        guard !a.isEmpty else {
            throw VectorError.invalidInput("Input array A is empty")
        }
        guard operation.isBinary else {
            throw VectorError.invalidInput("Operation \(operation) is not a binary operation")
        }
        guard a.count == b.count else {
            throw VectorError.countMismatch(expected: a.count, actual: b.count)
        }

        let device = context.device.rawDevice

        guard let inputA = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: a.count * MemoryLayout<Float>.size)
        }
        inputA.label = "Elementwise.inputA"

        guard let inputB = device.makeBuffer(
            bytes: b,
            length: b.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: b.count * MemoryLayout<Float>.size)
        }
        inputB.label = "Elementwise.inputB"

        let parameters = Metal4ElementwiseParameters(
            numElements: a.count,
            operation: operation,
            useFastMath: useFastMath
        )

        return try await execute(inputA: inputA, inputB: inputB, parameters: parameters)
    }

    /// Perform scalar/unary operation on array.
    public func execute(
        _ a: [Float],
        operation: Metal4ElementwiseOperation,
        useFastMath: Bool = false
    ) async throws -> Metal4ElementwiseResult {
        guard !a.isEmpty else {
            throw VectorError.invalidInput("Input array is empty")
        }
        guard !operation.isBinary else {
            throw VectorError.invalidInput("Binary operation \(operation) requires two inputs")
        }

        let device = context.device.rawDevice

        guard let inputA = device.makeBuffer(
            bytes: a,
            length: a.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: a.count * MemoryLayout<Float>.size)
        }
        inputA.label = "Elementwise.input"

        let parameters = Metal4ElementwiseParameters(
            numElements: a.count,
            operation: operation,
            useFastMath: useFastMath
        )

        return try await execute(inputA: inputA, inputB: nil, parameters: parameters)
    }

    /// Perform operation using VectorProtocol types.
    public func execute<V: VectorProtocol>(
        _ a: V,
        _ b: V? = nil,
        operation: Metal4ElementwiseOperation,
        useFastMath: Bool = false
    ) async throws -> Metal4ElementwiseResult where V.Scalar == Float {
        let arrayA: [Float] = a.withUnsafeBufferPointer { Array($0) }

        if operation.isBinary {
            guard let b = b else {
                throw VectorError.invalidInput("Binary operation requires second input")
            }
            let arrayB: [Float] = b.withUnsafeBufferPointer { Array($0) }
            return try await execute(arrayA, arrayB, operation: operation, useFastMath: useFastMath)
        } else {
            return try await execute(arrayA, operation: operation, useFastMath: useFastMath)
        }
    }

    // MARK: - Convenience Methods

    /// Add two arrays element-wise.
    public func add(_ a: [Float], _ b: [Float]) async throws -> [Float] {
        let result = try await execute(a, b, operation: .add)
        return result.asArray()
    }

    /// Subtract b from a element-wise.
    public func subtract(_ a: [Float], _ b: [Float]) async throws -> [Float] {
        let result = try await execute(a, b, operation: .subtract)
        return result.asArray()
    }

    /// Multiply two arrays element-wise.
    public func multiply(_ a: [Float], _ b: [Float]) async throws -> [Float] {
        let result = try await execute(a, b, operation: .multiply)
        return result.asArray()
    }

    /// Divide a by b element-wise.
    public func divide(_ a: [Float], _ b: [Float]) async throws -> [Float] {
        let result = try await execute(a, b, operation: .divide)
        return result.asArray()
    }

    /// Scale array by scalar value.
    public func scale(_ a: [Float], by scalar: Float) async throws -> [Float] {
        let result = try await execute(a, operation: .multiplyScalar(scalar))
        return result.asArray()
    }

    /// Add scalar to each element.
    public func add(_ a: [Float], scalar: Float) async throws -> [Float] {
        let result = try await execute(a, operation: .addScalar(scalar))
        return result.asArray()
    }

    /// Clamp values to range.
    public func clamp(_ a: [Float], min: Float, max: Float) async throws -> [Float] {
        let result = try await execute(a, operation: .clamp(min: min, max: max))
        return result.asArray()
    }

    /// Compute absolute value of each element.
    public func abs(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .absolute)
        return result.asArray()
    }

    /// Compute square of each element.
    public func square(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .square)
        return result.asArray()
    }

    /// Compute square root of each element.
    public func sqrt(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .sqrt)
        return result.asArray()
    }

    /// Compute exponential of each element.
    public func exp(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .exp)
        return result.asArray()
    }

    /// Compute natural logarithm of each element.
    public func log(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .log)
        return result.asArray()
    }

    /// Negate each element.
    public func negate(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .negate)
        return result.asArray()
    }

    /// Compute reciprocal of each element.
    public func reciprocal(_ a: [Float]) async throws -> [Float] {
        let result = try await execute(a, operation: .reciprocal)
        return result.asArray()
    }
}
