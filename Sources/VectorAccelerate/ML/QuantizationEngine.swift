//
//  QuantizationEngine.swift
//  VectorAccelerate
//
//  Vector quantization for model compression and efficient storage
//

import Foundation
@preconcurrency import Metal
import Accelerate
import VectorCore

/// Quantization method
public enum QuantizationMethod: Sendable {
    case scalar(bits: Int)  // 4, 8, 16 bit scalar quantization
    case product(codebooks: Int, bitsPerCode: Int)  // Product quantization
    case binary  // Binary quantization (1 bit)
    case adaptive(targetCompression: Float)  // Adaptive based on target
}

/// Quantization configuration
public struct QuantizationConfiguration: Sendable {
    public let method: QuantizationMethod
    public let preserveNorm: Bool
    public let useSymmetric: Bool  // Symmetric vs asymmetric quantization
    public let useGPU: Bool
    
    public init(
        method: QuantizationMethod = .scalar(bits: 8),
        preserveNorm: Bool = false,
        useSymmetric: Bool = true,
        useGPU: Bool = true
    ) {
        self.method = method
        self.preserveNorm = preserveNorm
        self.useSymmetric = useSymmetric
        self.useGPU = useGPU
    }
}

// QuantizedVector is imported from Core/Types.swift
// Additional helper for compression ratio calculation
extension QuantizedVector {
    /// Calculate compression ratio
    public var compressionRatio: Float {
        let originalSize = Float(dimensions * MemoryLayout<Float>.size)
        let quantizedSize = Float(data.count)
        return originalSize / quantizedSize
    }
}

/// Product quantization codebook
public struct ProductCodebook: Sendable {
    public let subvectorSize: Int
    public let numCentroids: Int
    public let centroids: [[Float]]  // [numCentroids][subvectorSize]
    public let index: Int
}

/// Main quantization engine using Metal 4
public actor QuantizationEngine {
    private let context: Metal4Context?
    private let configuration: QuantizationConfiguration
    private let logger: Logger

    // Product quantization state
    private var productCodebooks: [ProductCodebook] = []
    private var isProductQuantizationTrained = false

    // Performance tracking
    private var quantizedVectors: Int = 0
    private var totalCompressionRatio: Float = 0

    // MARK: - Initialization

    public init(
        configuration: QuantizationConfiguration,
        context: Metal4Context? = nil
    ) async {
        self.configuration = configuration
        if configuration.useGPU {
            if let ctx = context {
                self.context = ctx
            } else {
                self.context = await Metal4Context.createDefault()
            }
        } else {
            self.context = nil
        }
        self.logger = Logger.shared

        await logger.info("Initialized QuantizationEngine with method: \(configuration.method)",
                         category: "QuantizationEngine")
    }
    
    // MARK: - Scalar Quantization
    
    /// Quantize vector using scalar quantization
    ///
    /// Maps floating-point values to fixed-point representation
    /// - Parameters:
    ///   - vector: Input vector to quantize
    ///   - bits: Number of bits per element (4, 8, or 16)
    /// - Returns: Quantized vector representation
    public func scalarQuantize(vector: [Float], bits: Int = 8) async throws -> QuantizedVector {
        let measureToken = await logger.startMeasure("scalarQuantize")
        measureToken.addMetadata("bits", value: "\(bits)")
        defer { measureToken.end() }

        guard [4, 8, 16].contains(bits) else {
            throw VectorError.unsupportedGPUOperation("Scalar quantization supports 4, 8, or 16 bits")
        }

        // Handle empty vector
        guard !vector.isEmpty else {
            return QuantizedVector(
                data: Data(),
                dimensions: 0,
                scheme: .scalar(bits: bits),
                metadata: ["scale": "1.0", "offset": "0.0", "bits": String(bits)]
            )
        }

        // Sanitize input: replace NaN/Inf with finite bounds
        // This allows quantization to proceed with best-effort for edge case data
        // Also clamp extremely large values to prevent overflow during dequantization
        // Using a safe maximum that leaves headroom for scale multiplication
        let safeMaxMagnitude: Float = Float.greatestFiniteMagnitude / 256.0  // Safe headroom for 8-bit scale ops
        var sanitizedVector = vector
        var hasNonFinite = false
        for i in 0..<sanitizedVector.count {
            if !sanitizedVector[i].isFinite {
                hasNonFinite = true
                if sanitizedVector[i].isNaN {
                    sanitizedVector[i] = 0.0  // Replace NaN with zero
                } else if sanitizedVector[i] == .infinity {
                    sanitizedVector[i] = safeMaxMagnitude
                } else {  // -infinity
                    sanitizedVector[i] = -safeMaxMagnitude
                }
            } else if sanitizedVector[i] > safeMaxMagnitude {
                // Clamp extremely large positive values to prevent overflow during dequantization
                hasNonFinite = true
                sanitizedVector[i] = safeMaxMagnitude
            } else if sanitizedVector[i] < -safeMaxMagnitude {
                // Clamp extremely large negative values to prevent overflow during dequantization
                hasNonFinite = true
                sanitizedVector[i] = -safeMaxMagnitude
            }
        }

        if hasNonFinite {
            await logger.warning("Vector contains non-finite or extreme values; sanitized for quantization", category: "QuantizationEngine")
        }

        // Find min and max for quantization range
        var minVal: Float = Float.infinity
        var maxVal: Float = -Float.infinity

        vDSP_minv(sanitizedVector, 1, &minVal, vDSP_Length(sanitizedVector.count))
        vDSP_maxv(sanitizedVector, 1, &maxVal, vDSP_Length(sanitizedVector.count))

        // Calculate scale and offset
        let range = maxVal - minVal
        var scale: Float
        var offset: Float

        if configuration.useSymmetric {
            // Symmetric quantization around zero
            let absMax = max(abs(minVal), abs(maxVal))
            scale = absMax / Float((1 << (bits - 1)) - 1)
            offset = 0
        } else {
            // Asymmetric quantization
            scale = range / Float((1 << bits) - 1)
            offset = minVal
        }

        // Handle edge case: all values identical (range = 0) or scale becomes 0/NaN
        if !scale.isFinite || scale == 0 {
            scale = 1.0  // Use unit scale when range is zero
        }
        
        // Quantize values using sanitized vector
        var quantizedData = Data()

        switch bits {
        case 4:
            // Pack two 4-bit values per byte
            var buffer: UInt8 = 0
            for (idx, value) in sanitizedVector.enumerated() {
                let quantized = quantizeValue(value, scale: scale, offset: offset, bits: bits)
                if idx % 2 == 0 {
                    buffer = UInt8(quantized & 0xF)
                } else {
                    buffer |= UInt8((quantized & 0xF) << 4)
                    quantizedData.append(buffer)
                }
            }
            if sanitizedVector.count % 2 != 0 {
                quantizedData.append(buffer)
            }

        case 8:
            // One byte per value
            for value in sanitizedVector {
                let quantized = quantizeValue(value, scale: scale, offset: offset, bits: bits)
                quantizedData.append(UInt8(quantized & 0xFF))
            }

        case 16:
            // Two bytes per value
            for value in sanitizedVector {
                let quantized = quantizeValue(value, scale: scale, offset: offset, bits: bits)
                var value16 = UInt16(quantized & 0xFFFF)
                quantizedData.append(Data(bytes: &value16, count: 2))
            }

        default:
            throw VectorError.unsupportedGPUOperation("Unsupported bit width: \(bits)")
        }
        
        // Update metrics
        quantizedVectors += 1
        let ratio = Float(vector.count * MemoryLayout<Float>.size) / Float(quantizedData.count)
        totalCompressionRatio += ratio
        
        let metadata = [
            "scale": String(scale),
            "offset": String(offset),
            "bits": String(bits)
        ]
        
        return QuantizedVector(
            data: quantizedData,
            dimensions: vector.count,
            scheme: .scalar(bits: bits),
            metadata: metadata
        )
    }
    
    /// Dequantize scalar quantized vector
    public func scalarDequantize(quantized: QuantizedVector) async throws -> [Float] {
        guard case .scalar(let bits) = quantized.scheme else {
            throw VectorError.unsupportedGPUOperation("Expected scalar quantized vector")
        }
        
        guard let scaleStr = quantized.metadata?["scale"],
              let offsetStr = quantized.metadata?["offset"],
              let scale = Float(scaleStr),
              let offset = Float(offsetStr) else {
            throw VectorError.unsupportedGPUOperation("Missing quantization metadata")
        }
        
        var result = [Float](repeating: 0, count: quantized.dimensions)
        
        quantized.data.withUnsafeBytes { bytes in
            let ptr = bytes.bindMemory(to: UInt8.self).baseAddress!
            
            switch bits {
            case 4:
                // Unpack two 4-bit values per byte
                for i in 0..<quantized.dimensions {
                    let byteIdx = i / 2
                    let nibble = i % 2
                    let byte = ptr[byteIdx]
                    let quantizedVal = nibble == 0 ? (byte & 0xF) : ((byte >> 4) & 0xF)
                    result[i] = dequantizeValue(Int(quantizedVal), scale: scale, offset: offset, bits: bits)
                }
                
            case 8:
                // One byte per value
                for i in 0..<quantized.dimensions {
                    let quantizedVal = Int(ptr[i])
                    result[i] = dequantizeValue(quantizedVal, scale: scale, offset: offset, bits: bits)
                }
                
            case 16:
                // Two bytes per value
                let ptr16 = bytes.bindMemory(to: UInt16.self).baseAddress!
                for i in 0..<quantized.dimensions {
                    let quantizedVal = Int(ptr16[i])
                    result[i] = dequantizeValue(quantizedVal, scale: scale, offset: offset, bits: bits)
                }
                
            default:
                break
            }
        }
        
        return result
    }
    
    // MARK: - Product Quantization
    
    /// Train product quantization codebooks
    ///
    /// Divides vectors into subvectors and learns codebooks for each
    /// - Parameters:
    ///   - trainingData: Vectors to train on
    ///   - numCodebooks: Number of codebooks (subvectors)
    ///   - centroids: Number of centroids per codebook
    public func trainProductQuantization(
        trainingData: [[Float]],
        numCodebooks: Int = 8,
        centroids: Int = 256
    ) async throws {
        guard !trainingData.isEmpty else {
            throw VectorError.invalidOperation("Training data is empty")
        }
        
        let dimension = trainingData[0].count
        guard dimension % numCodebooks == 0 else {
            throw VectorError.invalidOperation("Dimension must be divisible by number of codebooks")
        }
        
        let measureToken = await logger.startMeasure("trainProductQuantization")
        measureToken.addMetadata("samples", value: "\(trainingData.count)")
        measureToken.addMetadata("codebooks", value: "\(numCodebooks)")
        defer { measureToken.end() }
        
        let subvectorSize = dimension / numCodebooks
        productCodebooks = []
        
        // Train each codebook
        for codebookIdx in 0..<numCodebooks {
            let startIdx = codebookIdx * subvectorSize
            let endIdx = startIdx + subvectorSize
            
            // Extract subvectors
            let subvectors = trainingData.map { vector in
                Array(vector[startIdx..<endIdx])
            }
            
            // Run k-means on subvectors
            let codebook = try await trainCodebook(
                subvectors: subvectors,
                numCentroids: centroids,
                index: codebookIdx
            )
            
            productCodebooks.append(codebook)
            
            await logger.debug("Trained codebook \(codebookIdx + 1)/\(numCodebooks)", 
                             category: "QuantizationEngine")
        }
        
        isProductQuantizationTrained = true
    }
    
    /// Quantize using product quantization
    public func productQuantize(vector: [Float]) async throws -> QuantizedVector {
        guard isProductQuantizationTrained else {
            throw VectorError.invalidOperation("Product quantization not trained")
        }
        
        let measureToken = await logger.startMeasure("productQuantize")
        defer { measureToken.end() }
        
        var codes = Data()
        let bitsPerCode = Int(ceil(log2(Double(productCodebooks[0].numCentroids))))
        
        // Encode each subvector
        for (idx, codebook) in productCodebooks.enumerated() {
            let startIdx = idx * codebook.subvectorSize
            let endIdx = startIdx + codebook.subvectorSize
            let subvector = Array(vector[startIdx..<endIdx])
            
            // Find nearest centroid
            var minDist = Float.infinity
            var bestCode = 0
            
            for (centroidIdx, centroid) in codebook.centroids.enumerated() {
                let dist = euclideanDistance(subvector, centroid)
                if dist < minDist {
                    minDist = dist
                    bestCode = centroidIdx
                }
            }
            
            // Store code (simplified - should pack bits efficiently)
            if bitsPerCode <= 8 {
                codes.append(UInt8(bestCode))
            } else {
                var code16 = UInt16(bestCode)
                codes.append(Data(bytes: &code16, count: 2))
            }
        }
        
        // Update metrics
        quantizedVectors += 1
        let ratio = Float(vector.count * MemoryLayout<Float>.size) / Float(codes.count)
        totalCompressionRatio += ratio
        
        let metadata = [
            "numCodebooks": String(productCodebooks.count),
            "bitsPerCode": String(bitsPerCode)
        ]
        
        return QuantizedVector(
            data: codes,
            dimensions: vector.count,
            scheme: .product(codebooks: productCodebooks.count, bitsPerCode: bitsPerCode),
            metadata: metadata
        )
    }
    
    /// Dequantize product quantized vector
    public func productDequantize(quantized: QuantizedVector) async throws -> [Float] {
        guard case .product(_, let bitsPerCode) = quantized.scheme else {
            throw VectorError.unsupportedGPUOperation("Expected product quantized vector")
        }
        
        guard isProductQuantizationTrained else {
            throw VectorError.invalidOperation("Product quantization not trained")
        }
        
        var result = [Float](repeating: 0, count: quantized.dimensions)
        
        quantized.data.withUnsafeBytes { bytes in
            let ptr = bytes.bindMemory(to: UInt8.self).baseAddress!
            
            for (idx, codebook) in productCodebooks.enumerated() {
                let code: Int
                if bitsPerCode <= 8 {
                    code = Int(ptr[idx])
                } else {
                    let ptr16 = bytes.bindMemory(to: UInt16.self).baseAddress!
                    code = Int(ptr16[idx])
                }
                
                // Copy centroid values
                let centroid = codebook.centroids[code]
                let startIdx = idx * codebook.subvectorSize
                for i in 0..<codebook.subvectorSize {
                    result[startIdx + i] = centroid[i]
                }
            }
        }
        
        return result
    }
    
    // MARK: - Binary Quantization
    
    /// Binary quantization (1-bit per dimension)
    ///
    /// Extremely efficient for Hamming distance computations
    public func binaryQuantize(vector: [Float]) async throws -> QuantizedVector {
        let measureToken = await logger.startMeasure("binaryQuantize")
        defer { measureToken.end() }
        
        // Calculate mean for thresholding
        var mean: Float = 0
        vDSP_meanv(vector, 1, &mean, vDSP_Length(vector.count))
        
        // Pack bits
        let numBytes = (vector.count + 7) / 8
        var data = Data(count: numBytes)
        
        data.withUnsafeMutableBytes { bytes in
            let ptr = bytes.bindMemory(to: UInt8.self).baseAddress!
            
            for (idx, value) in vector.enumerated() {
                let byteIdx = idx / 8
                let bitIdx = idx % 8
                
                if value > mean {
                    ptr[byteIdx] |= (1 << bitIdx)
                }
            }
        }
        
        // Update metrics
        quantizedVectors += 1
        let ratio = Float(vector.count * MemoryLayout<Float>.size) / Float(data.count)
        totalCompressionRatio += ratio
        
        let metadata = [
            "threshold": String(mean)
        ]
        
        return QuantizedVector(
            data: data,
            dimensions: vector.count,
            scheme: .binary,
            metadata: metadata
        )
    }
    
    /// Dequantize binary quantized vector
    public func binaryDequantize(quantized: QuantizedVector) async throws -> [Float] {
        guard case .binary = quantized.scheme else {
            throw VectorError.unsupportedGPUOperation("Expected binary quantized vector")
        }
        
        guard let thresholdStr = quantized.metadata?["threshold"],
              let threshold = Float(thresholdStr) else {
            throw VectorError.unsupportedGPUOperation("Missing threshold metadata")
        }
        
        var result = [Float](repeating: 0, count: quantized.dimensions)
        
        quantized.data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            let ptr = bytes.bindMemory(to: UInt8.self).baseAddress!
            
            for i in 0..<quantized.dimensions {
                let byteIdx = i / 8
                let bitIdx = i % 8
                let bit = (ptr[byteIdx] >> bitIdx) & 1
                
                // Reconstruct as mean +/- fixed value
                result[i] = threshold + (bit == 1 ? 0.5 : -0.5)
            }
        }
        
        return result
    }
    
    // MARK: - Utility Functions
    
    /// Quantize a single value
    /// - Note: Returns 0 for non-finite values (NaN/Inf) as a safety fallback
    private func quantizeValue(_ value: Float, scale: Float, offset: Float, bits: Int) -> Int {
        // Safety check: ensure value is finite before processing
        guard value.isFinite else {
            return 0  // Safe fallback for any non-finite values that slip through
        }

        if configuration.useSymmetric && offset == 0 {
            // Symmetric quantization uses signed integers
            let normalized = scale != 0 ? value / scale : 0
            let maxVal = Float((1 << (bits - 1)) - 1)
            let minVal = -maxVal
            let quantized = round(min(max(normalized, minVal), maxVal))
            // Safety: ensure result is finite before Int conversion
            guard quantized.isFinite else { return Int(maxVal) }
            // Convert to unsigned for storage
            return Int(quantized + maxVal)
        } else {
            // Asymmetric quantization
            let normalized = scale != 0 ? (value - offset) / scale : 0
            let maxVal = Float((1 << bits) - 1)
            let quantized = round(min(max(normalized, 0), maxVal))
            // Safety: ensure result is finite before Int conversion
            guard quantized.isFinite else { return 0 }
            return Int(quantized)
        }
    }
    
    /// Dequantize a single value
    private func dequantizeValue(_ quantized: Int, scale: Float, offset: Float, bits: Int) -> Float {
        if configuration.useSymmetric && offset == 0 {
            // Symmetric quantization - convert back from unsigned storage
            let maxVal = Float((1 << (bits - 1)) - 1)
            let signed = Float(quantized) - maxVal
            return signed * scale
        } else {
            // Asymmetric quantization
            return Float(quantized) * scale + offset
        }
    }
    
    /// Train a single codebook using k-means
    private func trainCodebook(
        subvectors: [[Float]],
        numCentroids: Int,
        index: Int
    ) async throws -> ProductCodebook {
        // Simple k-means implementation
        var centroids = [[Float]]()
        
        // Initialize centroids randomly
        for _ in 0..<numCentroids {
            let randomIdx = Int.random(in: 0..<subvectors.count)
            centroids.append(subvectors[randomIdx])
        }
        
        // Lloyd's algorithm (simplified)
        for _ in 0..<10 {  // Fixed iterations for simplicity
            var newCentroids = [[Float]](repeating: [Float](repeating: 0, count: subvectors[0].count), 
                                        count: numCentroids)
            var counts = [Int](repeating: 0, count: numCentroids)
            
            // Assignment step
            for subvector in subvectors {
                var minDist = Float.infinity
                var bestCentroid = 0
                
                for (idx, centroid) in centroids.enumerated() {
                    let dist = euclideanDistance(subvector, centroid)
                    if dist < minDist {
                        minDist = dist
                        bestCentroid = idx
                    }
                }
                
                // Add to new centroid
                for (i, val) in subvector.enumerated() {
                    newCentroids[bestCentroid][i] += val
                }
                counts[bestCentroid] += 1
            }
            
            // Update centroids
            for i in 0..<numCentroids {
                if counts[i] > 0 {
                    for j in 0..<newCentroids[i].count {
                        newCentroids[i][j] /= Float(counts[i])
                    }
                }
            }
            
            centroids = newCentroids
        }
        
        return ProductCodebook(
            subvectorSize: subvectors[0].count,
            numCentroids: numCentroids,
            centroids: centroids,
            index: index
        )
    }
    
    /// Simple Euclidean distance
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    // MARK: - Performance Metrics
    
    /// Measure quantization error
    public func measureError(original: [Float], quantized: QuantizedVector) async throws -> Float {
        let reconstructed = try await dequantize(quantized: quantized)
        
        var mse: Float = 0
        for i in 0..<original.count {
            let diff = original[i] - reconstructed[i]
            mse += diff * diff
        }
        mse /= Float(original.count)
        
        return sqrt(mse)  // RMSE
    }
    
    /// General dequantization method
    public func dequantize(quantized: QuantizedVector) async throws -> [Float] {
        switch quantized.scheme {
        case .scalar:
            return try await scalarDequantize(quantized: quantized)
        case .product:
            return try await productDequantize(quantized: quantized)
        case .binary:
            return try await binaryDequantize(quantized: quantized)
        case .custom:
            throw VectorError.unsupportedGPUOperation("Custom dequantization not implemented")
        }
    }
    
    public func getPerformanceMetrics() -> (
        vectorsQuantized: Int,
        averageCompressionRatio: Float
    ) {
        let avgRatio = quantizedVectors > 0 ? totalCompressionRatio / Float(quantizedVectors) : 0
        return (quantizedVectors, avgRatio)
    }
}