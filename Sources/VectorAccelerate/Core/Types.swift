// VectorAccelerate: Core Types
//
// Essential type definitions for Metal acceleration
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Metal Buffer Wrapper

/// Wrapper for Metal buffers with metadata for safe, efficient operations
public struct MetalBuffer: @unchecked Sendable {
    /// The underlying Metal buffer
    public let buffer: any MTLBuffer
    
    /// Number of elements in the buffer
    public let count: Int
    
    /// Type of elements stored in the buffer
    public let elementType: MetalElementType
    
    /// Storage mode for the buffer
    public let storageMode: MTLStorageMode
    
    /// Create a new MetalBuffer wrapper
    public init(buffer: any MTLBuffer, count: Int, elementType: MetalElementType = .float32) {
        self.buffer = buffer
        self.count = count
        self.elementType = elementType
        self.storageMode = buffer.storageMode
    }
    
    /// Total byte length of the buffer
    public var byteLength: Int {
        count * elementType.stride
    }
    
    /// Check if buffer is accessible from CPU
    public var isCPUAccessible: Bool {
        #if os(macOS)
        return storageMode == .shared || storageMode == .managed
        #else
        return storageMode == .shared
        #endif
    }
    
    /// Get typed pointer for CPU access (only for shared/managed buffers)
    public func contents<T>() -> UnsafeMutablePointer<T>? {
        guard isCPUAccessible else { return nil }
        return buffer.contents().bindMemory(to: T.self, capacity: count)
    }
}

// MARK: - Metal Element Types

/// Supported element types for Metal buffers
public enum MetalElementType: String, Sendable, CaseIterable {
    case float32 = "float"
    case float16 = "half"
    case int32 = "int"
    case uint32 = "uint"
    case uint8 = "uchar"
    
    /// Size in bytes for each element type
    public var stride: Int {
        switch self {
        case .float32, .int32, .uint32: return 4
        case .float16: return 2
        case .uint8: return 1
        }
    }
    
    /// Metal type string for shader code
    public var metalType: String {
        rawValue
    }
    
    /// Whether this type supports atomic operations
    public var supportsAtomics: Bool {
        switch self {
        case .int32, .uint32: return true
        case .float32, .float16, .uint8: return false
        }
    }
}

// MARK: - Distance Metrics

// MARK: - VectorAccelerate Distance Metric Extensions

/// VectorAccelerate-specific distance metric for metrics not in VectorCore
public enum VectorAccelerateDistanceMetric: String, Sendable, CaseIterable {
    case hamming

    /// Metal shader function name for this metric
    public var metalFunctionName: String {
        switch self {
        case .hamming: return "hamming_distance"
        }
    }

    /// Whether this metric returns a similarity (higher = more similar) or distance (lower = more similar)
    public var isSimilarity: Bool {
        return false
    }

    /// Range of possible values for this metric
    public var valueRange: ClosedRange<Float> {
        return 0.0...1.0
    }
}

/// Extensions to VectorCore's SupportedDistanceMetric for VectorAccelerate features
public extension SupportedDistanceMetric {
    /// Metal shader function name for this metric
    var metalFunctionName: String {
        switch self {
        case .euclidean: return "euclidean_distance"
        case .cosine: return "cosine_similarity"
        case .dotProduct: return "dot_product"
        case .manhattan: return "manhattan_distance"
        case .chebyshev: return "chebyshev_distance"
        }
    }

    /// Whether this metric returns a similarity (higher = more similar) or distance (lower = more similar)
    var isSimilarity: Bool {
        switch self {
        case .cosine, .dotProduct: return true
        case .euclidean, .manhattan, .chebyshev: return false
        }
    }

    /// Range of possible values for this metric
    var valueRange: ClosedRange<Float> {
        switch self {
        case .cosine: return -1.0...1.0
        case .dotProduct: return -Float.infinity...Float.infinity
        case .euclidean, .manhattan, .chebyshev: return 0.0...Float.infinity
        }
    }
}

// MARK: - Quantization Support

/// Quantization schemes for vector compression
public enum QuantizationScheme: Sendable, Equatable {
    /// Scalar quantization with specified bit width
    case scalar(bits: Int)
    
    /// Product quantization with specified number of codebooks
    case product(codebooks: Int, bitsPerCode: Int)
    
    /// Binary quantization (1 bit per dimension)
    case binary
    
    /// Custom quantization with user-defined parameters
    case custom(name: String, parameters: [String: String])
    
    /// Compression ratio achieved by this scheme
    public var compressionRatio: Float {
        switch self {
        case .scalar(let bits):
            return 32.0 / Float(bits)
        case .product(_, let bitsPerCode):
            return 32.0 / Float(bitsPerCode)
        case .binary:
            return 32.0
        case .custom:
            return 1.0 // Unknown
        }
    }
}

/// Quantized vector representation
public struct QuantizedVector: Sendable {
    /// Quantized data
    public let data: Data
    
    /// Original dimension count
    public let dimensions: Int
    
    /// Quantization scheme used
    public let scheme: QuantizationScheme
    
    /// Optional metadata (e.g., scale factors, centroids)
    public let metadata: [String: String]?
    
    public init(data: Data, dimensions: Int, scheme: QuantizationScheme, metadata: [String: String]? = nil) {
        self.data = data
        self.dimensions = dimensions
        self.scheme = scheme
        self.metadata = metadata
    }
}

// MARK: - Matrix Support

/// Matrix representation for batch operations
public struct Matrix: Sendable {
    /// Number of rows
    public let rows: Int
    
    /// Number of columns
    public let columns: Int
    
    /// Row-major storage of values
    public let values: [Float]
    
    /// Whether the matrix is stored in column-major order (for Metal)
    public let isColumnMajor: Bool
    
    public init(rows: Int, columns: Int, values: [Float], isColumnMajor: Bool = false) {
        precondition(values.count == rows * columns, "Invalid matrix dimensions")
        self.rows = rows
        self.columns = columns
        self.values = values
        self.isColumnMajor = isColumnMajor
    }
    
    /// Get value at specific position
    public subscript(row: Int, column: Int) -> Float {
        precondition(row < rows && column < columns, "Index out of bounds")
        if isColumnMajor {
            return values[column * rows + row]
        } else {
            return values[row * columns + column]
        }
    }
    
    /// Convert to Metal-friendly column-major format
    public func toColumnMajor() -> Matrix {
        guard !isColumnMajor else { return self }

        var columnMajorValues = [Float](repeating: 0, count: values.count)
        for row in 0..<rows {
            for col in 0..<columns {
                columnMajorValues[col * rows + row] = values[row * columns + col]
            }
        }

        return Matrix(rows: rows, columns: columns, values: columnMajorValues, isColumnMajor: true)
    }

    /// Create a matrix with random values in range [0, 1)
    public static func random(rows: Int, columns: Int, range: ClosedRange<Float> = 0...1) -> Matrix {
        let values = (0..<(rows * columns)).map { _ in
            Float.random(in: range)
        }
        return Matrix(rows: rows, columns: columns, values: values)
    }

    /// Create a zero matrix
    public static func zeros(rows: Int, columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, values: [Float](repeating: 0, count: rows * columns))
    }

    /// Create an identity matrix
    public static func identity(size: Int) -> Matrix {
        var values = [Float](repeating: 0, count: size * size)
        for i in 0..<size {
            values[i * size + i] = 1.0
        }
        return Matrix(rows: size, columns: size, values: values)
    }
}

// MARK: - Performance Metrics

/// Performance metrics for acceleration operations
public struct PerformanceMetrics: Sendable {
    /// Total execution time in seconds
    public let executionTime: TimeInterval
    
    /// Time spent in GPU operations
    public let gpuTime: TimeInterval?
    
    /// Time spent in CPU operations
    public let cpuTime: TimeInterval?
    
    /// Memory allocated during operation
    public let memoryAllocated: Int
    
    /// Memory peak during operation
    public let memoryPeak: Int
    
    /// Number of operations performed
    public let operationCount: Int
    
    /// Throughput (operations per second)
    public var throughput: Double {
        guard executionTime > 0 else { return 0 }
        return Double(operationCount) / executionTime
    }
    
    /// GPU utilization percentage
    public var gpuUtilization: Double? {
        guard let gpuTime = gpuTime, executionTime > 0 else { return nil }
        return (gpuTime / executionTime) * 100.0
    }
}

// MARK: - Tensor Shape Support

/// Shape representation for multi-dimensional tensors
public struct TensorShape: Sendable, Equatable, Codable {
    /// Dimensions of the tensor
    public let dimensions: [Int]

    /// Total number of elements
    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    /// Number of dimensions (rank)
    public var rank: Int {
        dimensions.count
    }

    public init(_ dimensions: Int...) {
        self.dimensions = dimensions
    }

    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions
    }

    public init(dimensions: [Int]) {
        self.dimensions = dimensions
    }

    /// Check if shape is compatible for broadcast
    public func isBroadcastCompatible(with other: TensorShape) -> Bool {
        // Simple broadcasting rules
        let minRank = min(rank, other.rank)
        for i in 0..<minRank {
            let dim1 = dimensions[rank - 1 - i]
            let dim2 = other.dimensions[other.rank - 1 - i]
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false
            }
        }
        return true
    }

    // MARK: - Factory Methods

    /// Create shape for projection matrix [outputDim, inputDim]
    public static func projection(inputDim: Int, outputDim: Int) -> TensorShape {
        TensorShape([outputDim, inputDim])
    }

    /// Create shape for 1D vector
    public static func vector(_ size: Int) -> TensorShape {
        TensorShape([size])
    }

    /// Create shape for 2D matrix
    public static func matrix(rows: Int, cols: Int) -> TensorShape {
        TensorShape([rows, cols])
    }
}

// MARK: - Type Aliases for VectorCore Compatibility

// VectorCore types are available through the import at the top of this file

// MARK: - Configuration Extensions

// Configuration types to be defined as needed