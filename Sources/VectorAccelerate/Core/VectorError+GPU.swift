// VectorAccelerate: GPU Error Extensions
//
// Extends VectorCore's VectorError with GPU-specific factory methods
// for Metal acceleration operations.
//

import Foundation
import VectorCore

// MARK: - GPU-Specific Error Factory Methods

public extension VectorError {

    // MARK: - Resource Availability Errors

    /// Metal is not available on this device
    static func metalNotAvailable(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("Metal is not available on this device")
            .parameter("resource", value: "Metal")
            .build()
    }

    /// Failed to initialize Metal device
    static func deviceInitializationFailed(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("Failed to initialize Metal device: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Shader not found in Metal library
    static func shaderNotFound(
        name: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("Shader '\(name)' not found")
            .parameter("shader_name", value: name)
            .build()
    }

    // MARK: - Buffer Allocation Errors

    /// Failed to allocate Metal buffer
    static func bufferAllocationFailed(
        size: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.allocationFailed, file: file, line: line, function: function)
            .message("Failed to allocate buffer of size \(size)")
            .parameter("requested_size", value: String(size))
            .build()
    }

    /// Buffer creation failed with reason
    static func bufferCreationFailed(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.allocationFailed, file: file, line: line, function: function)
            .message("Buffer creation failed: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Invalid buffer size requested
    static func invalidBufferSize(
        requested: Int,
        maximum: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message("Requested buffer size \(requested) exceeds maximum \(maximum)")
            .parameter("requested_size", value: String(requested))
            .parameter("maximum_size", value: String(maximum))
            .build()
    }

    // MARK: - Resource Exhaustion Errors

    /// Buffer pool has no available buffers
    static func bufferPoolExhausted(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceExhausted, file: file, line: line, function: function)
            .message("Buffer pool has no available buffers")
            .parameter("resource", value: "BufferPool")
            .build()
    }

    /// System is under memory pressure
    static func memoryPressure(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceExhausted, file: file, line: line, function: function)
            .message("System is under memory pressure")
            .parameter("resource", value: "Memory")
            .build()
    }

    // MARK: - Pipeline/Shader Errors

    /// Shader compilation failed
    static func shaderCompilationFailed(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Shader compilation failed: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Pipeline creation failed
    static func pipelineCreationFailed(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Pipeline creation failed: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Failed to create Metal command queue
    static func commandQueueCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create Metal command queue")
            .build()
    }

    /// Failed to create Metal shader library
    static func libraryCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create Metal shader library")
            .build()
    }

    /// Failed to create compute command encoder
    static func encoderCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create compute command encoder")
            .build()
    }

    // MARK: - Compute Errors

    /// Compute operation failed
    static func computeFailed(
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Compute operation failed: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Operation not supported on this device
    static func unsupportedGPUOperation(
        _ operation: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.unsupportedOperation, file: file, line: line, function: function)
            .message("Operation '\(operation)' is not supported on this device")
            .parameter("operation", value: operation)
            .build()
    }

    // MARK: - Input Validation Errors

    /// Invalid input for GPU operation
    static func invalidInput(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message("Invalid input: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Count mismatch between operands
    static func countMismatch(
        expected: Int? = nil,
        actual: Int? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        let message: String
        if let expected = expected, let actual = actual {
            message = "Count mismatch: expected \(expected), got \(actual)"
        } else {
            message = "Count mismatch between operands"
        }

        var builder = ErrorBuilder(.dimensionMismatch, file: file, line: line, function: function)
            .message(message)

        if let expected = expected {
            builder = builder.parameter("expected_count", value: String(expected))
        }
        if let actual = actual {
            builder = builder.parameter("actual_count", value: String(actual))
        }

        return builder.build()
    }

    // MARK: - File I/O Errors

    /// File not found at path
    static func fileNotFound(
        _ path: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("File not found: \(path)")
            .parameter("path", value: path)
            .build()
    }

    /// Invalid data format
    static func invalidDataFormat(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message("Invalid data format: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    /// Invalid operation (generic reason)
    static func invalidOperation(
        _ reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidOperation, file: file, line: line, function: function)
            .message(reason)
            .parameter("reason", value: reason)
            .build()
    }
}

// MARK: - Legacy Compatibility

/// Type alias for backward compatibility during migration.
/// @available(*, deprecated, renamed: "VectorError", message: "Use VectorError with GPU-specific factory methods instead")
@available(*, deprecated, renamed: "VectorError", message: "Use VectorError with GPU-specific factory methods instead. See VectorError+GPU.swift for available methods.")
public typealias GPUAccelerationError = VectorError
