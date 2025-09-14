// VectorAccelerate: Error Types
//
// Error handling for GPU acceleration operations
//

import Foundation

public enum AccelerationError: LocalizedError {
    case metalNotAvailable
    case deviceInitializationFailed(String)
    case bufferAllocationFailed(size: Int)
    case shaderNotFound(name: String)
    case shaderCompilationFailed(String)
    case pipelineCreationFailed(String)
    case computeFailed(reason: String)
    case bufferPoolExhausted
    case invalidBufferSize(requested: Int, maximum: Int)
    case memoryPressure
    case unsupportedOperation(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case invalidInput(String)
    case bufferCreationFailed(String)
    case encoderCreationFailed
    case countMismatch(expected: Int? = nil, actual: Int? = nil)
    case commandQueueCreationFailed
    case libraryCreationFailed
    
    public var errorDescription: String? {
        switch self {
        case .metalNotAvailable:
            return "Metal is not available on this device"
        case .deviceInitializationFailed(let reason):
            return "Failed to initialize Metal device: \(reason)"
        case .bufferAllocationFailed(let size):
            return "Failed to allocate buffer of size \(size)"
        case .shaderNotFound(let name):
            return "Shader '\(name)' not found"
        case .shaderCompilationFailed(let reason):
            return "Shader compilation failed: \(reason)"
        case .pipelineCreationFailed(let reason):
            return "Pipeline creation failed: \(reason)"
        case .computeFailed(let reason):
            return "Compute operation failed: \(reason)"
        case .bufferPoolExhausted:
            return "Buffer pool has no available buffers"
        case .invalidBufferSize(let requested, let maximum):
            return "Requested buffer size \(requested) exceeds maximum \(maximum)"
        case .memoryPressure:
            return "System is under memory pressure"
        case .unsupportedOperation(let operation):
            return "Operation '\(operation)' is not supported on this device"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .bufferCreationFailed(let reason):
            return "Buffer creation failed: \(reason)"
        case .encoderCreationFailed:
            return "Failed to create compute command encoder"
        case .countMismatch(let expected, let actual):
            if let expected = expected, let actual = actual {
                return "Count mismatch: expected \(expected), got \(actual)"
            } else {
                return "Count mismatch between operands"
            }
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .libraryCreationFailed:
            return "Failed to create Metal shader library"
        }
    }
}