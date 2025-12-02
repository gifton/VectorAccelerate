//
//  Metal4Error.swift
//  VectorAccelerate
//
//  Metal 4 specific error factory methods extending VectorError
//

import Foundation
import VectorCore

// MARK: - Metal 4 Specific Error Factory Methods

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension VectorError {

    // MARK: - Setup Errors

    /// Metal 4 is not supported on this device
    static func metal4NotSupported(
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("Metal 4 not supported: \(reason)")
            .parameter("reason", value: reason)
            .parameter("api_version", value: "Metal4")
            .build()
    }

    /// Failed to create Metal 4 command queue
    static func metal4CommandQueueCreationFailed(
        underlying: (any Error)? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        var builder = ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create MTL4CommandQueue")
            .parameter("component", value: "MTL4CommandQueue")

        if let underlying = underlying {
            builder = builder.parameter("underlying_error", value: underlying.localizedDescription)
        }

        return builder.build()
    }

    /// Failed to create Metal 4 compiler
    static func metal4CompilerCreationFailed(
        underlying: (any Error)? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        var builder = ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create MTL4Compiler")
            .parameter("component", value: "MTL4Compiler")

        if let underlying = underlying {
            builder = builder.parameter("underlying_error", value: underlying.localizedDescription)
        }

        return builder.build()
    }

    // MARK: - Residency Errors

    /// Failed to create residency set
    static func residencySetCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create MTLResidencySet")
            .parameter("component", value: "MTLResidencySet")
            .build()
    }

    /// Residency set is full
    static func residencySetFull(
        capacity: Int,
        requested: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceExhausted, file: file, line: line, function: function)
            .message("Residency set is full (capacity: \(capacity), requested: \(requested))")
            .parameter("capacity", value: String(capacity))
            .parameter("requested", value: String(requested))
            .build()
    }

    /// Residency commit failed
    static func residencyCommitFailed(
        underlying: any Error,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to commit residency set: \(underlying.localizedDescription)")
            .parameter("underlying_error", value: underlying.localizedDescription)
            .build()
    }

    /// Buffer not in residency set
    static func allocationNotInResidencySet(
        bufferLabel: String?,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        let label = bufferLabel ?? "unknown"
        return ErrorBuilder(.invalidOperation, file: file, line: line, function: function)
            .message("Buffer '\(label)' is not in residency set")
            .parameter("buffer_label", value: label)
            .build()
    }

    // MARK: - Argument Table Errors

    /// Failed to create argument table
    static func argumentTableCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create MTL4ArgumentTable")
            .parameter("component", value: "MTL4ArgumentTable")
            .build()
    }

    /// Argument table pool exhausted
    static func argumentTablePoolExhausted(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceExhausted, file: file, line: line, function: function)
            .message("Argument table pool exhausted")
            .parameter("resource", value: "ArgumentTablePool")
            .build()
    }

    // MARK: - Compilation Errors

    /// Metal 4 shader compilation failed
    static func metal4ShaderCompilationFailed(
        shader: String,
        error: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Metal 4 shader compilation failed: \(error)")
            .parameter("shader", value: shader)
            .parameter("error", value: error)
            .build()
    }

    /// Function not found in library
    static func metal4FunctionNotFound(
        name: String,
        library: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.resourceUnavailable, file: file, line: line, function: function)
            .message("Function '\(name)' not found in library '\(library)'")
            .parameter("function_name", value: name)
            .parameter("library", value: library)
            .build()
    }

    /// Pipeline creation failed
    static func metal4PipelineCreationFailed(
        functionName: String,
        error: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Pipeline creation failed for '\(functionName)': \(error)")
            .parameter("function", value: functionName)
            .parameter("error", value: error)
            .build()
    }

    // MARK: - Execution Errors

    /// Command buffer execution error
    static func metal4CommandBufferError(
        status: String,
        error: (any Error)?,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        var builder = ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Command buffer error (status: \(status))")
            .parameter("status", value: status)

        if let error = error {
            builder = builder.parameter("error", value: error.localizedDescription)
        }

        return builder.build()
    }

    /// Dispatch failed
    static func metal4DispatchFailed(
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Dispatch failed: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }

    // MARK: - Feature Errors

    /// Feature not supported on this device
    static func metal4FeatureNotSupported(
        feature: String,
        requiredFamily: String? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        var builder = ErrorBuilder(.unsupportedOperation, file: file, line: line, function: function)
            .message("Feature '\(feature)' is not supported on this device")
            .parameter("feature", value: feature)

        if let family = requiredFamily {
            builder = builder.parameter("required_family", value: family)
        }

        return builder.build()
    }

    /// ML Tensor not available
    static func mlTensorNotAvailable(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.unsupportedOperation, file: file, line: line, function: function)
            .message("MTLTensor is not available on this device")
            .parameter("feature", value: "MLTensor")
            .build()
    }

    // MARK: - Tensor Errors

    /// Tensor size mismatch
    static func tensorSizeMismatch(
        expected: Int,
        got: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.dimensionMismatch, file: file, line: line, function: function)
            .message("Tensor size mismatch: expected \(expected), got \(got)")
            .parameter("expected", value: String(expected))
            .parameter("actual", value: String(got))
            .build()
    }

    // MARK: - Timeout Errors

    /// Operation timed out
    static func metal4Timeout(
        operation: String,
        durationMs: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Operation '\(operation)' timed out after \(durationMs)ms")
            .parameter("operation", value: operation)
            .parameter("timeout_ms", value: String(durationMs))
            .build()
    }

    // MARK: - Shared Event Errors

    /// Failed to create shared event
    static func sharedEventCreationFailed(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.operationFailed, file: file, line: line, function: function)
            .message("Failed to create MTLSharedEvent")
            .parameter("component", value: "MTLSharedEvent")
            .build()
    }
}
