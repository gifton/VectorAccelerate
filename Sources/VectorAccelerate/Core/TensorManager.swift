//
//  TensorManager.swift
//  VectorAccelerate
//
//  Manages MTLTensor buffers for ML weights and learned projections
//
//  This is part of Phase 4 (Experimental ML Integration).
//  All features are gated behind Metal4Capabilities.supportsMLTensor.
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Tensor Data Type

/// Supported data types for tensor operations
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum TensorDataType: String, Codable, Sendable {
    case float32
    case float16
    case bfloat16
    case int8
    case uint8

    /// Size in bytes per element
    public var elementSize: Int {
        switch self {
        case .float32: return 4
        case .float16, .bfloat16: return 2
        case .int8, .uint8: return 1
        }
    }

    /// Corresponding MTLDataType (when available)
    public var metalDataType: MTLDataType {
        switch self {
        case .float32: return .float
        case .float16: return .half
        case .bfloat16: return .bfloat
        case .int8: return .char
        case .uint8: return .uchar
        }
    }
}

// MARK: - TensorShape Extensions for ML

/// Extension to add byte size calculation for TensorDataType
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension TensorShape {
    /// Size in bytes for given data type
    func byteSize(dataType: TensorDataType) -> Int {
        elementCount * dataType.elementSize
    }
}

// MARK: - Tensor Buffer

/// A Metal buffer containing tensor data with shape metadata
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct TensorBuffer: Sendable {
    /// The underlying Metal buffer
    public let buffer: any MTLBuffer

    /// Shape of the tensor
    public let shape: TensorShape

    /// Data type of elements
    public let dataType: TensorDataType

    /// Optional name for debugging
    public let name: String?

    /// Size in bytes
    public var byteSize: Int {
        shape.byteSize(dataType: dataType)
    }

    /// GPU address for argument table binding
    public var gpuAddress: UInt64 {
        buffer.gpuAddress
    }

    public init(buffer: any MTLBuffer, shape: TensorShape, dataType: TensorDataType, name: String? = nil) {
        self.buffer = buffer
        self.shape = shape
        self.dataType = dataType
        self.name = name
    }
}

// MARK: - Tensor Metadata

/// Metadata for serialized tensor files
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct TensorMetadata: Codable, Sendable {
    public let name: String
    public let shape: TensorShape
    public let dataType: TensorDataType
    public let byteOffset: Int
    public let byteSize: Int
    public let checksum: String?

    public init(
        name: String,
        shape: TensorShape,
        dataType: TensorDataType,
        byteOffset: Int = 0,
        byteSize: Int? = nil,
        checksum: String? = nil
    ) {
        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.byteOffset = byteOffset
        self.byteSize = byteSize ?? shape.byteSize(dataType: dataType)
        self.checksum = checksum
    }
}

// MARK: - Weight File Format

/// Format for weight files (header + binary data)
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct WeightFileManifest: Codable, Sendable {
    public let version: String
    public let tensors: [TensorMetadata]
    public let totalSize: Int
    public let createdAt: Date

    public init(version: String = "1.0.0", tensors: [TensorMetadata]) {
        self.version = version
        self.tensors = tensors
        self.totalSize = tensors.reduce(0) { $0 + $1.byteSize }
        self.createdAt = Date()
    }
}

// MARK: - Tensor Manager Statistics

/// Statistics for tensor manager monitoring
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct TensorManagerStatistics: Sendable {
    public let loadedTensors: Int
    public let totalMemoryBytes: Int
    public let loadCount: Int
    public let unloadCount: Int

    public var totalMemoryMB: Double {
        Double(totalMemoryBytes) / (1024 * 1024)
    }
}

// MARK: - Tensor Manager

/// Manages MTLTensor buffers for ML weights and learned transformations
///
/// TensorManager provides:
/// - Weight loading from files or memory
/// - Buffer lifecycle management
/// - Memory tracking
/// - GPU address resolution for shader binding
///
/// Example:
/// ```swift
/// let manager = TensorManager(device: device)
///
/// // Load projection weights
/// let projection = try await manager.loadWeights(
///     from: weightsURL,
///     name: "projection",
///     shape: TensorShape.projection(inputDim: 384, outputDim: 128)
/// )
///
/// // Use in shader
/// encoder.setBuffer(projection.buffer, offset: 0, index: 2)
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor TensorManager {
    // MARK: - Properties

    private let device: any MTLDevice
    private var tensors: [String: TensorBuffer] = [:]
    private var totalMemoryBytes: Int = 0

    // Statistics
    private var loadCount: Int = 0
    private var unloadCount: Int = 0

    // MARK: - Initialization

    /// Create a tensor manager for the given device
    public init(device: any MTLDevice) {
        self.device = device
    }

    // MARK: - Loading from File

    /// Load tensor weights from a binary file
    ///
    /// - Parameters:
    ///   - url: URL to the weights file
    ///   - name: Name to identify this tensor
    ///   - shape: Expected shape of the tensor
    ///   - dataType: Data type of elements (default: float32)
    /// - Returns: TensorBuffer containing the loaded weights
    public func loadWeights(
        from url: URL,
        name: String,
        shape: TensorShape,
        dataType: TensorDataType = .float32
    ) async throws -> TensorBuffer {
        let expectedSize = shape.byteSize(dataType: dataType)

        // Load file data
        let data = try Data(contentsOf: url)

        guard data.count >= expectedSize else {
            throw VectorError.tensorSizeMismatch(
                expected: expectedSize,
                actual: data.count
            )
        }

        return try await loadWeights(
            from: data,
            name: name,
            shape: shape,
            dataType: dataType
        )
    }

    /// Load tensor weights from Data
    public func loadWeights(
        from data: Data,
        name: String,
        shape: TensorShape,
        dataType: TensorDataType = .float32
    ) async throws -> TensorBuffer {
        let expectedSize = shape.byteSize(dataType: dataType)

        guard data.count >= expectedSize else {
            throw VectorError.tensorSizeMismatch(
                expected: expectedSize,
                actual: data.count
            )
        }

        // Create Metal buffer
        let buffer = try data.withUnsafeBytes { bytes -> any MTLBuffer in
            guard let buffer = device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: expectedSize,
                options: .storageModeShared
            ) else {
                throw VectorError.bufferAllocationFailed(size: expectedSize)
            }
            buffer.label = "TensorManager.\(name)"
            return buffer
        }

        let tensor = TensorBuffer(
            buffer: buffer,
            shape: shape,
            dataType: dataType,
            name: name
        )

        // Store and track
        tensors[name] = tensor
        totalMemoryBytes += expectedSize
        loadCount += 1

        return tensor
    }

    /// Load multiple tensors from a weight file with manifest
    public func loadWeightFile(from url: URL) async throws -> [String: TensorBuffer] {
        let manifestURL = url.appendingPathExtension("json")
        let dataURL = url

        // Load manifest
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(WeightFileManifest.self, from: manifestData)

        // Load binary data
        let fileData = try Data(contentsOf: dataURL)

        guard fileData.count >= manifest.totalSize else {
            throw VectorError.tensorSizeMismatch(
                expected: manifest.totalSize,
                actual: fileData.count
            )
        }

        // Load each tensor
        var loaded: [String: TensorBuffer] = [:]

        for metadata in manifest.tensors {
            let tensorData = fileData.subdata(
                in: metadata.byteOffset..<(metadata.byteOffset + metadata.byteSize)
            )

            let tensor = try await loadWeights(
                from: tensorData,
                name: metadata.name,
                shape: metadata.shape,
                dataType: metadata.dataType
            )

            loaded[metadata.name] = tensor
        }

        return loaded
    }

    // MARK: - Creating from Arrays

    /// Create a tensor from a Float array
    public func createTensor(
        from data: [Float],
        name: String,
        shape: TensorShape
    ) async throws -> TensorBuffer {
        let expectedCount = shape.elementCount

        guard data.count == expectedCount else {
            throw VectorError.tensorSizeMismatch(
                expected: expectedCount * MemoryLayout<Float>.size,
                actual: data.count * MemoryLayout<Float>.size
            )
        }

        let byteSize = data.count * MemoryLayout<Float>.stride

        let buffer = data.withUnsafeBytes { bytes -> (any MTLBuffer)? in
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteSize,
                options: .storageModeShared
            )
        }

        guard let buffer = buffer else {
            throw VectorError.bufferAllocationFailed(size: byteSize)
        }

        buffer.label = "TensorManager.\(name)"

        let tensor = TensorBuffer(
            buffer: buffer,
            shape: shape,
            dataType: .float32,
            name: name
        )

        tensors[name] = tensor
        totalMemoryBytes += byteSize
        loadCount += 1

        return tensor
    }

    /// Create a 2D projection matrix tensor
    public func createProjectionMatrix(
        from weights: [[Float]],
        name: String
    ) async throws -> TensorBuffer {
        guard !weights.isEmpty, !weights[0].isEmpty else {
            throw VectorError.invalidDimension(0, reason: "Projection matrix cannot be empty")
        }

        let rows = weights.count
        let cols = weights[0].count

        // Flatten row-major
        let flattened = weights.flatMap { $0 }

        return try await createTensor(
            from: flattened,
            name: name,
            shape: TensorShape.matrix(rows: rows, cols: cols)
        )
    }

    /// Create a random projection matrix (for testing/initialization)
    public func createRandomProjection(
        inputDim: Int,
        outputDim: Int,
        name: String,
        scale: Float = 0.02
    ) async throws -> TensorBuffer {
        // Xavier/Glorot initialization
        let fanIn = Float(inputDim)
        let fanOut = Float(outputDim)
        let stddev = scale * sqrt(2.0 / (fanIn + fanOut))

        var weights = [Float](repeating: 0, count: inputDim * outputDim)
        for i in 0..<weights.count {
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: 0.0001...1.0)
            let u2 = Float.random(in: 0...1)
            let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            weights[i] = z * stddev
        }

        return try await createTensor(
            from: weights,
            name: name,
            shape: TensorShape.projection(inputDim: inputDim, outputDim: outputDim)
        )
    }

    // MARK: - Retrieval

    /// Get a loaded tensor by name
    public func getTensor(name: String) -> TensorBuffer? {
        tensors[name]
    }

    /// Get all loaded tensor names
    public var loadedTensorNames: [String] {
        Array(tensors.keys)
    }

    /// Check if a tensor is loaded
    public func isLoaded(_ name: String) -> Bool {
        tensors[name] != nil
    }

    // MARK: - Unloading

    /// Unload a tensor and free memory
    public func unload(name: String) {
        if let tensor = tensors.removeValue(forKey: name) {
            totalMemoryBytes -= tensor.byteSize
            unloadCount += 1
        }
    }

    /// Unload all tensors
    public func unloadAll() {
        let count = tensors.count
        tensors.removeAll()
        totalMemoryBytes = 0
        unloadCount += count
    }

    // MARK: - Statistics

    /// Get current statistics
    public func getStatistics() -> TensorManagerStatistics {
        TensorManagerStatistics(
            loadedTensors: tensors.count,
            totalMemoryBytes: totalMemoryBytes,
            loadCount: loadCount,
            unloadCount: unloadCount
        )
    }

    /// Reset statistics counters
    public func resetStatistics() {
        loadCount = 0
        unloadCount = 0
    }

    // MARK: - Validation

    /// Validate tensor shape matches expected
    public func validateShape(name: String, expected: TensorShape) -> Bool {
        guard let tensor = tensors[name] else { return false }
        return tensor.shape == expected
    }

    /// Validate all required tensors are loaded
    public func validateRequired(_ names: [String]) -> [String] {
        names.filter { tensors[$0] == nil }
    }
}

// MARK: - Weight File Utilities

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum WeightFileUtils {
    /// Create a weight file from tensors
    public static func createWeightFile(
        tensors: [(name: String, data: [Float], shape: TensorShape)],
        outputURL: URL
    ) throws {
        var metadata: [TensorMetadata] = []
        var allData = Data()
        var offset = 0

        for (name, data, shape) in tensors {
            let byteSize = data.count * MemoryLayout<Float>.stride

            let tensorData = data.withUnsafeBytes { Data($0) }
            allData.append(tensorData)

            metadata.append(TensorMetadata(
                name: name,
                shape: shape,
                dataType: .float32,
                byteOffset: offset,
                byteSize: byteSize
            ))

            offset += byteSize
        }

        // Write binary data
        try allData.write(to: outputURL)

        // Write manifest
        let manifest = WeightFileManifest(tensors: metadata)
        let manifestData = try JSONEncoder().encode(manifest)
        try manifestData.write(to: outputURL.appendingPathExtension("json"))
    }
}

// MARK: - VectorError Extension

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension VectorError {
    /// Tensor size mismatch error
    static func tensorSizeMismatch(expected: Int, actual: Int) -> VectorError {
        VectorError.invalidOperation(
            "Tensor size mismatch: expected \(expected) bytes, got \(actual) bytes"
        )
    }
}
