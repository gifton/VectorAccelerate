//
//  IndexAccelerationError.swift
//  VectorIndexAcceleration
//
//  Error types specific to index acceleration operations.
//

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - Index Acceleration Errors

/// Errors specific to GPU-accelerated index operations.
///
/// These errors cover GPU initialization, buffer management, kernel execution,
/// and configuration validation for AcceleratedVectorIndex operations.
public enum IndexAccelerationError: Error, Sendable {

    // MARK: - Initialization Errors

    /// GPU context not initialized before accelerated operation.
    case gpuNotInitialized(operation: String)

    /// Failed to create GPU resources for index.
    case gpuResourceCreationFailed(index: String, reason: String)

    // MARK: - Dimension Errors

    /// Index dimension mismatch with input vector.
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

    // MARK: - Configuration Errors

    /// Invalid configuration parameter.
    case invalidConfiguration(parameter: String, reason: String)

    // MARK: - Input Errors

    /// Invalid input provided to operation.
    case invalidInput(message: String)

    // MARK: - Training Errors

    /// IVF training failed.
    case trainingFailed(reason: String)
}

// MARK: - LocalizedError Conformance

extension IndexAccelerationError: LocalizedError {

    public var errorDescription: String? {
        switch self {
        case .gpuNotInitialized(let operation):
            return "GPU not initialized for operation: \(operation)"

        case .gpuResourceCreationFailed(let index, let reason):
            return "Failed to create GPU resources for \(index): \(reason)"

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

        case .invalidConfiguration(let parameter, let reason):
            return "Invalid configuration for '\(parameter)': \(reason)"

        case .invalidInput(let message):
            return "Invalid input: \(message)"

        case .trainingFailed(let reason):
            return "IVF training failed: \(reason)"
        }
    }
}
