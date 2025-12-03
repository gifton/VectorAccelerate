//
//  IndexAccelerationError.swift
//  VectorIndexAcceleration
//
//  Error types specific to index acceleration operations.
//  Extends VectorAccelerate's error hierarchy.
//

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - Index Acceleration Errors

/// Errors specific to GPU-accelerated index operations.
///
/// These errors extend VectorAccelerate's `VectorError` with index-specific cases.
/// Use the convenience static methods for creating properly formatted errors.
public enum IndexAccelerationError: Error, Sendable {

    // MARK: - Initialization Errors

    /// GPU context not initialized before accelerated operation.
    case gpuNotInitialized(operation: String)

    /// Failed to create GPU resources for index.
    case gpuResourceCreationFailed(index: String, reason: String)

    // MARK: - Index Compatibility Errors

    /// Index type does not support GPU acceleration.
    case indexNotAccelerable(indexType: String)

    /// Index is empty and cannot be searched.
    case emptyIndex(indexType: String)

    /// Index dimension mismatch with query.
    case dimensionMismatch(expected: Int, got: Int)

    // MARK: - Kernel Errors

    /// Required kernel not available for operation.
    case kernelNotAvailable(kernelName: String, reason: String)

    /// Kernel execution failed.
    case kernelExecutionFailed(kernelName: String, underlying: any Error)

    // MARK: - Buffer Errors

    /// Failed to create or access GPU buffer.
    case bufferError(operation: String, reason: String)

    /// Buffer size exceeds GPU memory limits.
    case bufferTooLarge(requested: Int, available: Int)

    // MARK: - Search Errors

    /// Search operation failed.
    case searchFailed(indexType: String, reason: String)

    /// Batch search operation failed.
    case batchSearchFailed(queryIndex: Int, reason: String)

    // MARK: - Configuration Errors

    /// Invalid configuration parameter.
    case invalidConfiguration(parameter: String, reason: String)

    // MARK: - Input Errors

    /// Invalid input provided to operation.
    case invalidInput(message: String)
}

// MARK: - LocalizedError Conformance

extension IndexAccelerationError: LocalizedError {

    public var errorDescription: String? {
        switch self {
        case .gpuNotInitialized(let operation):
            return "GPU not initialized for operation: \(operation). Call prepareForGPU() first."

        case .gpuResourceCreationFailed(let index, let reason):
            return "Failed to create GPU resources for \(index): \(reason)"

        case .indexNotAccelerable(let indexType):
            return "Index type '\(indexType)' does not support GPU acceleration"

        case .emptyIndex(let indexType):
            return "\(indexType) is empty and cannot be searched"

        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: index expects \(expected), got \(got)"

        case .kernelNotAvailable(let kernelName, let reason):
            return "Kernel '\(kernelName)' not available: \(reason)"

        case .kernelExecutionFailed(let kernelName, let underlying):
            return "Kernel '\(kernelName)' execution failed: \(underlying.localizedDescription)"

        case .bufferError(let operation, let reason):
            return "Buffer error during \(operation): \(reason)"

        case .bufferTooLarge(let requested, let available):
            return "Buffer too large: requested \(requested) bytes, available \(available) bytes"

        case .searchFailed(let indexType, let reason):
            return "\(indexType) search failed: \(reason)"

        case .batchSearchFailed(let queryIndex, let reason):
            return "Batch search failed at query \(queryIndex): \(reason)"

        case .invalidConfiguration(let parameter, let reason):
            return "Invalid configuration for '\(parameter)': \(reason)"

        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}

// MARK: - Convenience Initializers

public extension IndexAccelerationError {

    /// Create a GPU not initialized error for the given operation.
    static func notInitialized(for operation: String) -> IndexAccelerationError {
        .gpuNotInitialized(operation: operation)
    }

    /// Create a dimension mismatch error.
    static func wrongDimension(expected: Int, got: Int) -> IndexAccelerationError {
        .dimensionMismatch(expected: expected, got: got)
    }
}
