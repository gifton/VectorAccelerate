//
//  IVFQuantizedStorage.swift
//  VectorAccelerate
//
//  Manages quantized vector storage for IVF indexes.
//
//  P2.2: Scalar Quantization Integration
//
//  Supports INT8 and INT4 quantization for 4x-8x memory reduction.
//  Vectors are quantized after K-Means training and dequantized
//  on-the-fly during search for distance computation.
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - IVF Quantized Storage

/// Manages quantized vector storage for IVF indexes.
///
/// Provides memory-efficient storage by converting float32 vectors to
/// INT8 or INT4 representation. Supports both symmetric and asymmetric
/// quantization modes.
///
/// ## Quantization Process
/// 1. After K-Means training, compute quantization parameters (scale, zero-point)
/// 2. Quantize all training vectors to compressed format
/// 3. During search, dequantize candidate vectors for distance computation
///
/// ## Memory Layout
/// - INT8: 1 byte per element, direct storage
/// - INT4: 4 bits per element, packed 2 elements per byte
final class IVFQuantizedStorage: @unchecked Sendable {

    // MARK: - Configuration

    /// Quantization type being used.
    let quantizationType: VectorQuantization

    /// Vector dimension.
    let dimension: Int

    // MARK: - Quantization Parameters

    /// Scale factor for quantization.
    /// x_quantized = round(x / scale) + zeroPoint
    private(set) var scale: Float = 1.0

    /// Zero point for asymmetric quantization.
    /// nil for symmetric quantization.
    private(set) var zeroPoint: Int8? = nil

    /// Per-vector scales for per-vector quantization (optional).
    private var perVectorScales: [Float]? = nil

    // MARK: - Storage

    /// Quantized vector storage buffer.
    /// Layout: [numVectors × compressedDim] where compressedDim depends on bit width.
    private var quantizedBuffer: (any MTLBuffer)?

    /// Number of vectors currently stored.
    private(set) var vectorCount: Int = 0

    /// GPU context for Metal operations.
    private weak var context: Metal4Context?

    // MARK: - Statistics

    /// Total bytes used by quantized storage.
    var usedBytes: Int {
        quantizedBuffer?.length ?? 0
    }

    /// Compression ratio achieved.
    var compressionRatio: Float {
        quantizationType.compressionRatio
    }

    // MARK: - Initialization

    /// Create an IVF quantized storage.
    ///
    /// - Parameters:
    ///   - quantizationType: Type of quantization to use
    ///   - dimension: Vector dimension
    ///   - context: Metal 4 context for GPU operations
    init(
        quantizationType: VectorQuantization,
        dimension: Int,
        context: Metal4Context
    ) {
        self.quantizationType = quantizationType
        self.dimension = dimension
        self.context = context
    }

    // MARK: - Parameter Computation

    /// Compute optimal quantization parameters from training data.
    ///
    /// Analyzes the value distribution to determine scale and zero-point
    /// that minimize quantization error.
    ///
    /// - Parameter vectors: Training vectors
    /// - Returns: Tuple of (scale, zeroPoint) for asymmetric, or (scale, nil) for symmetric
    func computeParameters(from vectors: [[Float]]) -> (scale: Float, zeroPoint: Int8?) {
        guard !vectors.isEmpty else {
            return (scale: 1.0, zeroPoint: nil)
        }

        // Find global min/max across all vectors
        var globalMin: Float = .infinity
        var globalMax: Float = -.infinity

        for vector in vectors {
            for val in vector {
                globalMin = min(globalMin, val)
                globalMax = max(globalMax, val)
            }
        }

        let bitWidth = metalBitWidth
        let symmetricMax: Float = bitWidth == .int4 ? 7.0 : 127.0
        let asymmetricMax: Float = bitWidth == .int4 ? 15.0 : 255.0

        switch quantizationType {
        case .none:
            return (scale: 1.0, zeroPoint: nil)

        case .sq8, .sq4:
            // Symmetric quantization: scale based on absolute max
            let absMax = max(abs(globalMin), abs(globalMax))
            let scale = max(absMax / symmetricMax, 1e-8)
            return (scale: scale, zeroPoint: nil)

        case .sq8Asymmetric:
            // Asymmetric quantization: use full range
            let range = globalMax - globalMin
            let scale = max(range / asymmetricMax, 1e-8)
            let zeroPoint = Int8(clamping: Int(-round(globalMin / scale)))
            return (scale: scale, zeroPoint: zeroPoint)
        }
    }

    // MARK: - Quantization

    /// Quantize vectors and store in GPU buffer.
    ///
    /// Computes quantization parameters if not already set, then quantizes
    /// all input vectors to the compressed format.
    ///
    /// - Parameters:
    ///   - vectors: Vectors to quantize [numVectors × dimension]
    ///   - existingParams: Optional pre-computed parameters (scale, zeroPoint)
    /// - Throws: `VectorError` if quantization fails
    func quantize(
        vectors: [[Float]],
        existingParams: (scale: Float, zeroPoint: Int8?)? = nil
    ) throws {
        guard !vectors.isEmpty else { return }
        guard let context = context else {
            throw VectorError.invalidInput("Metal context not available for quantization")
        }

        // Compute or use existing parameters
        let params = existingParams ?? computeParameters(from: vectors)
        self.scale = params.scale
        self.zeroPoint = params.zeroPoint

        // Calculate buffer size
        let numVectors = vectors.count
        let bytesPerVector = quantizedBytesPerVector
        let totalBytes = numVectors * bytesPerVector

        // Allocate buffer
        guard let buffer = context.device.rawDevice.makeBuffer(
            length: totalBytes,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalBytes)
        }
        buffer.label = "IVFQuantizedStorage.quantizedVectors"

        // Quantize vectors
        let ptr = buffer.contents()

        switch quantizationType {
        case .none:
            // Should not happen, but handle gracefully
            break

        case .sq8:
            quantizeToInt8Symmetric(vectors: vectors, output: ptr)

        case .sq8Asymmetric:
            quantizeToInt8Asymmetric(vectors: vectors, output: ptr)

        case .sq4:
            quantizeToInt4(vectors: vectors, output: ptr)
        }

        self.quantizedBuffer = buffer
        self.vectorCount = numVectors
    }

    /// Append new vectors to existing quantized storage.
    ///
    /// Uses the existing quantization parameters. Should only be called
    /// after initial quantization has set the parameters.
    ///
    /// - Parameter vectors: New vectors to append
    /// - Throws: `VectorError` if append fails
    func appendQuantized(vectors: [[Float]]) throws {
        guard !vectors.isEmpty else { return }
        guard let context = context else {
            throw VectorError.invalidInput("Metal context not available for quantization")
        }

        let newCount = vectors.count
        let totalCount = vectorCount + newCount
        let bytesPerVector = quantizedBytesPerVector
        let totalBytes = totalCount * bytesPerVector

        // Allocate new buffer
        guard let newBuffer = context.device.rawDevice.makeBuffer(
            length: totalBytes,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: totalBytes)
        }
        newBuffer.label = "IVFQuantizedStorage.quantizedVectors"

        let newPtr = newBuffer.contents()

        // Copy existing data
        if let existingBuffer = quantizedBuffer, vectorCount > 0 {
            let existingBytes = vectorCount * bytesPerVector
            memcpy(newPtr, existingBuffer.contents(), existingBytes)
        }

        // Quantize and append new vectors
        let appendPtr = newPtr.advanced(by: vectorCount * bytesPerVector)

        switch quantizationType {
        case .none:
            break

        case .sq8:
            quantizeToInt8Symmetric(vectors: vectors, output: appendPtr)

        case .sq8Asymmetric:
            quantizeToInt8Asymmetric(vectors: vectors, output: appendPtr)

        case .sq4:
            quantizeToInt4(vectors: vectors, output: appendPtr)
        }

        self.quantizedBuffer = newBuffer
        self.vectorCount = totalCount
    }

    // MARK: - Dequantization

    /// Get the quantized buffer for GPU operations.
    ///
    /// - Returns: Metal buffer containing quantized vectors
    func getQuantizedBuffer() -> (any MTLBuffer)? {
        return quantizedBuffer
    }

    /// Get quantization parameters for use in shaders.
    ///
    /// - Returns: Tuple of (scale, zeroPoint, bitWidth)
    func getQuantizationParams() -> (scale: Float, zeroPoint: Int8, bitWidth: Metal4BitWidth) {
        return (
            scale: scale,
            zeroPoint: zeroPoint ?? 0,
            bitWidth: metalBitWidth
        )
    }

    /// Dequantize specific vector indices to float32.
    ///
    /// CPU-based dequantization for small number of vectors.
    /// For large-scale operations, use GPU dequantization.
    ///
    /// - Parameter indices: Indices of vectors to dequantize
    /// - Returns: Dequantized float vectors
    func dequantize(indices: [Int]) -> [[Float]] {
        guard let buffer = quantizedBuffer else { return [] }

        let ptr = buffer.contents()
        let bytesPerVector = quantizedBytesPerVector

        var result: [[Float]] = []
        result.reserveCapacity(indices.count)

        for idx in indices {
            guard idx >= 0 && idx < vectorCount else { continue }

            let vectorPtr = ptr.advanced(by: idx * bytesPerVector)
            var vector = [Float](repeating: 0, count: dimension)

            switch quantizationType {
            case .none:
                // Should not happen
                break

            case .sq8, .sq8Asymmetric:
                let int8Ptr = vectorPtr.bindMemory(to: Int8.self, capacity: dimension)
                let zp = Float(zeroPoint ?? 0)
                for d in 0..<dimension {
                    vector[d] = (Float(int8Ptr[d]) - zp) * scale
                }

            case .sq4:
                // INT4: 2 elements per byte
                let uint8Ptr = vectorPtr.bindMemory(to: UInt8.self, capacity: (dimension + 1) / 2)
                for d in 0..<dimension {
                    let byteIdx = d / 2
                    let byte = uint8Ptr[byteIdx]
                    let nibble: UInt8 = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F)
                    // INT4 symmetric: range [-8, 7]
                    let signed = Int8(bitPattern: nibble > 7 ? nibble | 0xF0 : nibble)
                    vector[d] = Float(signed) * scale
                }
            }

            result.append(vector)
        }

        return result
    }

    /// Dequantize all vectors to a GPU buffer.
    ///
    /// Creates a float32 buffer containing all dequantized vectors.
    /// Used for distance computation during search.
    ///
    /// - Returns: Metal buffer with dequantized float32 vectors
    /// - Throws: `VectorError` if dequantization fails
    func dequantizeAll() throws -> (any MTLBuffer)? {
        guard let context = context,
              let quantized = quantizedBuffer,
              vectorCount > 0 else {
            return nil
        }

        let outputBytes = vectorCount * dimension * MemoryLayout<Float>.size
        guard let outputBuffer = context.device.rawDevice.makeBuffer(
            length: outputBytes,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: outputBytes)
        }
        outputBuffer.label = "IVFQuantizedStorage.dequantized"

        // CPU dequantization (could be optimized to GPU)
        let inPtr = quantized.contents()
        let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: vectorCount * dimension)
        let bytesPerVector = quantizedBytesPerVector
        let zp = Float(zeroPoint ?? 0)

        for v in 0..<vectorCount {
            let vectorPtr = inPtr.advanced(by: v * bytesPerVector)
            let outOffset = v * dimension

            switch quantizationType {
            case .none:
                break

            case .sq8, .sq8Asymmetric:
                let int8Ptr = vectorPtr.bindMemory(to: Int8.self, capacity: dimension)
                for d in 0..<dimension {
                    outPtr[outOffset + d] = (Float(int8Ptr[d]) - zp) * scale
                }

            case .sq4:
                let uint8Ptr = vectorPtr.bindMemory(to: UInt8.self, capacity: (dimension + 1) / 2)
                for d in 0..<dimension {
                    let byteIdx = d / 2
                    let byte = uint8Ptr[byteIdx]
                    let nibble: UInt8 = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F)
                    let signed = Int8(bitPattern: nibble > 7 ? nibble | 0xF0 : nibble)
                    outPtr[outOffset + d] = Float(signed) * scale
                }
            }
        }

        return outputBuffer
    }

    // MARK: - Private Helpers

    /// Bytes per vector in quantized format.
    private var quantizedBytesPerVector: Int {
        switch quantizationType {
        case .none:
            return dimension * MemoryLayout<Float>.size
        case .sq8, .sq8Asymmetric:
            return dimension
        case .sq4:
            return (dimension + 1) / 2
        }
    }

    /// Metal bit width for this quantization type.
    private var metalBitWidth: Metal4BitWidth {
        switch quantizationType {
        case .none, .sq8, .sq8Asymmetric:
            return .int8
        case .sq4:
            return .int4
        }
    }

    /// Quantize vectors to INT8 using symmetric quantization.
    private func quantizeToInt8Symmetric(vectors: [[Float]], output: UnsafeMutableRawPointer) {
        let int8Ptr = output.bindMemory(to: Int8.self, capacity: vectors.count * dimension)
        let invScale = 1.0 / scale

        for (v, vector) in vectors.enumerated() {
            let offset = v * dimension
            for (d, val) in vector.enumerated() {
                let quantized = round(val * invScale)
                let clamped = max(-128, min(127, quantized))
                int8Ptr[offset + d] = Int8(clamped)
            }
        }
    }

    /// Quantize vectors to INT8 using asymmetric quantization.
    private func quantizeToInt8Asymmetric(vectors: [[Float]], output: UnsafeMutableRawPointer) {
        let int8Ptr = output.bindMemory(to: Int8.self, capacity: vectors.count * dimension)
        let invScale = 1.0 / scale
        let zp = Float(zeroPoint ?? 0)

        for (v, vector) in vectors.enumerated() {
            let offset = v * dimension
            for (d, val) in vector.enumerated() {
                let quantized = round(val * invScale) + zp
                let clamped = max(-128, min(127, quantized))
                int8Ptr[offset + d] = Int8(clamped)
            }
        }
    }

    /// Quantize vectors to INT4 (packed, 2 elements per byte).
    private func quantizeToInt4(vectors: [[Float]], output: UnsafeMutableRawPointer) {
        let bytesPerVector = (dimension + 1) / 2
        let uint8Ptr = output.bindMemory(to: UInt8.self, capacity: vectors.count * bytesPerVector)
        let invScale = 1.0 / scale

        for (v, vector) in vectors.enumerated() {
            let offset = v * bytesPerVector
            for d in stride(from: 0, to: dimension, by: 2) {
                let val0 = vector[d]
                let q0 = Int(round(val0 * invScale))
                let clamped0 = UInt8(clamping: max(-8, min(7, q0)) + 8) & 0x0F

                var byte: UInt8 = clamped0
                if d + 1 < dimension {
                    let val1 = vector[d + 1]
                    let q1 = Int(round(val1 * invScale))
                    let clamped1 = UInt8(clamping: max(-8, min(7, q1)) + 8) & 0x0F
                    byte |= (clamped1 << 4)
                }

                uint8Ptr[offset + d / 2] = byte
            }
        }
    }

    // MARK: - Reset

    /// Clear all stored data and reset parameters.
    func reset() {
        quantizedBuffer = nil
        vectorCount = 0
        scale = 1.0
        zeroPoint = nil
        perVectorScales = nil
    }
}
