# Swift Wrapper Specifications for Existing Metal Kernels

## Overview
This document provides detailed specifications for Swift wrappers needed for three existing Metal kernels in VectorAccelerate. These kernels are already implemented in Metal but lack Swift interface classes.

## Common Requirements
- All wrappers should follow the existing pattern used in the project (see HammingDistanceKernel.swift, L2DistanceKernel.swift as references)
- Use `AccelerationError` enum for error handling (defined in Core/AccelerationError.swift)
- Import: `Metal`, `Foundation`, `VectorCore`
- Use `ComputeEngine` for Metal device management when available
- Support both synchronous and async operations where appropriate
- Include result structs for structured return values
- Follow the project's SIMD optimization patterns

---

## 1. Jaccard Distance Kernel Wrapper

### Metal Kernel Signature
**Location:** `Sources/VectorAccelerate/Metal/Shaders/DistanceShaders.metal` (lines 150-198)

```metal
kernel void jaccardDistance(
    constant float* vectorA [[buffer(0)]],      // Input vector A
    constant float* vectorB [[buffer(1)]],      // Input vector B  
    device float* result [[buffer(2)]],         // Output: single float distance value
    constant uint& dimension [[buffer(3)]],     // Vector dimension
    uint id [[thread_position_in_grid]]
)
```

### Kernel Behavior
- Treats input vectors as binary sets (non-zero = present, zero = absent)
- Computes Jaccard distance = 1 - (intersection / union)
- Returns 0.0 when union is empty (both vectors are all zeros)
- Uses threadgroup shared memory with 256 threads for reduction

### Swift Wrapper Specification

```swift
// File: Sources/VectorAccelerate/Kernels/JaccardDistanceKernel.swift

public final class JaccardDistanceKernel {
    private let device: MTLDevice
    private let computeEngine: ComputeEngine
    private let pipelineState: MTLComputePipelineState
    
    // MARK: - Result Types
    
    public struct JaccardResult {
        public let distance: Float
        public let similarity: Float  // Jaccard similarity = 1 - distance
        public let intersectionSize: Int
        public let unionSize: Int
    }
    
    public struct BatchJaccardResult {
        public let distances: MTLBuffer
        public let rows: Int  // Number of vectors in set A
        public let cols: Int  // Number of vectors in set B
        
        /// Get distance at (row, col)
        public func distance(row: Int, col: Int) -> Float
        
        /// Extract full distance matrix
        public func asMatrix() -> [[Float]]
    }
    
    // MARK: - Initialization
    
    public init(device: MTLDevice) throws {
        // Load kernel "jaccardDistance" from default library
        // Create pipeline state
    }
    
    // MARK: - Core Operations
    
    /// Compute Jaccard distance between two vectors
    /// - Parameters:
    ///   - vectorA: First vector (treated as binary set)
    ///   - vectorB: Second vector (treated as binary set)
    /// - Returns: Jaccard distance result with metrics
    public func computeDistance(vectorA: [Float], vectorB: [Float]) throws -> JaccardResult
    
    /// Compute pairwise Jaccard distances (batch operation)
    /// - Parameters:
    ///   - vectorsA: First set of vectors [M x D]
    ///   - vectorsB: Second set of vectors [N x D]
    /// - Returns: Distance matrix [M x N]
    public func computeDistanceMatrix(vectorsA: [[Float]], vectorsB: [[Float]]) throws -> BatchJaccardResult
    
    /// Async version
    public func computeDistanceAsync(vectorA: [Float], vectorB: [Float]) async throws -> JaccardResult
    
    // MARK: - VectorCore Integration
    
    /// Compute distance using VectorCore protocol types
    public func computeDistance<V: VectorProtocol>(_ a: V, _ b: V) throws -> JaccardResult 
        where V.Scalar == Float
}
```

### Implementation Notes
- Jaccard distance expects binary vectors but accepts float inputs (non-zero = 1, zero = 0)
- For batch operations, consider implementing tiled computation for large matrices
- Thread configuration: Use 256 threads per threadgroup (matches kernel's shared memory size)
- Grid size: `(dimension + 255) / 256` threadgroups
- Consider adding threshold parameter for binarization (values > threshold = 1)

---

## 2. Binary Quantization Kernel Wrapper

### Metal Kernel Signatures
**Location:** `Sources/VectorAccelerate/Metal/Shaders/QuantizationShaders.metal`

```metal
// Quantization kernel (lines 126-154)
kernel void binaryQuantize(
    constant float* vectors [[buffer(0)]],      // Input: float vectors [numVectors x dimension]
    device uint* binaryVectors [[buffer(1)]],   // Output: bit-packed vectors [numVectors x numWords]
    constant uint& vectorDimension [[buffer(2)]],
    uint id [[thread_position_in_grid]]         // Vector index
)

// Hamming distance kernel (lines 157-181)
kernel void binaryHammingDistance(
    constant uint* queryBinary [[buffer(0)]],     // Query vector (bit-packed)
    constant uint* candidateBinary [[buffer(1)]], // Candidate vectors (bit-packed)
    device float* distances [[buffer(2)]],        // Output distances
    constant uint& numWords [[buffer(3)]],        // Number of 32-bit words per vector
    uint id [[thread_position_in_grid]]           // Candidate index
)
```

### Kernel Behavior
- **binaryQuantize**: Packs float vectors into bit vectors (32x compression)
  - Each dimension becomes 1 bit (positive = 1, non-positive = 0)
  - Packs 32 dimensions into each uint32
  - Number of words needed = `(dimension + 31) / 32`
  
- **binaryHammingDistance**: Computes Hamming distance between bit-packed vectors
  - Uses XOR and popcount for efficient computation
  - Returns distance as float for consistency

### Swift Wrapper Specification

```swift
// File: Sources/VectorAccelerate/Kernels/BinaryQuantizationKernel.swift

public final class BinaryQuantizationKernel {
    private let device: MTLDevice
    private let computeEngine: ComputeEngine
    private let quantizeKernel: MTLComputePipelineState
    private let hammingKernel: MTLComputePipelineState
    
    // MARK: - Types
    
    /// Bit-packed binary vector representation
    public struct BinaryVector {
        public let buffer: MTLBuffer
        public let dimension: Int
        public let numWords: Int  // Number of uint32 words
        
        /// Extract as bool array
        public func asBoolArray() -> [Bool]
        
        /// Get bit at specific dimension
        public func bit(at dimension: Int) -> Bool
    }
    
    /// Batch of binary vectors
    public struct BinaryVectorBatch {
        public let buffer: MTLBuffer
        public let count: Int
        public let dimension: Int
        public let numWords: Int
        
        /// Get specific vector
        public func vector(at index: Int) -> BinaryVector
    }
    
    public struct QuantizationResult {
        public let binaryVectors: BinaryVectorBatch
        public let compressionRatio: Float  // Should be ~32x
        public let originalBytes: Int
        public let compressedBytes: Int
    }
    
    public struct HammingDistanceResult {
        public let distances: [Float]
        public let minDistance: Float
        public let maxDistance: Float
        public let meanDistance: Float
    }
    
    // MARK: - Initialization
    
    public init(device: MTLDevice) throws {
        // Load kernels "binaryQuantize" and "binaryHammingDistance"
        // Create pipeline states
    }
    
    // MARK: - Quantization Operations
    
    /// Quantize float vectors to binary (bit-packed)
    /// - Parameter vectors: Input vectors [N x D]
    /// - Returns: Bit-packed binary vectors with metrics
    public func quantize(vectors: [[Float]]) throws -> QuantizationResult
    
    /// Quantize single vector
    public func quantize(vector: [Float]) throws -> BinaryVector
    
    // MARK: - Distance Operations
    
    /// Compute Hamming distances between binary vectors
    /// - Parameters:
    ///   - query: Query binary vector
    ///   - candidates: Candidate binary vectors
    /// - Returns: Hamming distances with statistics
    public func computeHammingDistances(
        query: BinaryVector,
        candidates: BinaryVectorBatch
    ) throws -> HammingDistanceResult
    
    /// Compute pairwise Hamming distances
    public func computePairwiseDistances(
        vectorsA: BinaryVectorBatch,
        vectorsB: BinaryVectorBatch
    ) throws -> [[Float]]
    
    // MARK: - Utility Methods
    
    /// Dequantize binary vector back to float (0.0 or 1.0 values)
    public func dequantize(binaryVector: BinaryVector) -> [Float]
    
    /// Calculate memory savings
    public static func calculateCompressionRatio(dimension: Int) -> Float
    
    // MARK: - Async Operations
    
    public func quantizeAsync(vectors: [[Float]]) async throws -> QuantizationResult
}
```

### Implementation Notes
- Thread configuration for quantize: One thread per vector
- Thread configuration for Hamming: One thread per candidate vector
- Memory layout: Bit-packed vectors stored as arrays of uint32
- Bit ordering: LSB first (bit 0 = dimension 0, bit 31 = dimension 31)
- For large batches, consider chunking to avoid memory pressure
- Compression ratio = `(dimension * 4) / ((dimension + 31) / 32 * 4)` ≈ 32x

---

## 3. Quantization Statistics Kernel Wrapper

### Metal Kernel Signature
**Location:** `Sources/VectorAccelerate/Metal/Shaders/QuantizationShaders.metal` (lines 186-224)

```metal
kernel void computeQuantizationStats(
    constant float* original [[buffer(0)]],     // Original vectors [numVectors x dimension]
    constant float* quantized [[buffer(1)]],    // Quantized vectors [numVectors x dimension]
    device float* mse [[buffer(2)]],           // Output: MSE per vector
    device float* psnr [[buffer(3)]],          // Output: PSNR per vector (in dB)
    constant uint& vectorDimension [[buffer(4)]],
    uint id [[thread_position_in_grid]]        // Vector index
)
```

### Kernel Behavior
- Computes per-vector quantization quality metrics
- MSE (Mean Squared Error): Average squared difference
- PSNR (Peak Signal-to-Noise Ratio): 20 * log10(maxValue / sqrt(MSE))
- Returns INFINITY for PSNR when perfect reconstruction (MSE = 0)
- Each thread processes one complete vector

### Swift Wrapper Specification

```swift
// File: Sources/VectorAccelerate/Kernels/QuantizationStatisticsKernel.swift

public final class QuantizationStatisticsKernel {
    private let device: MTLDevice
    private let computeEngine: ComputeEngine
    private let pipelineState: MTLComputePipelineState
    
    // MARK: - Result Types
    
    /// Quality metrics for a single vector
    public struct VectorMetrics {
        public let mse: Float              // Mean Squared Error
        public let psnr: Float             // Peak Signal-to-Noise Ratio (dB)
        public let rmse: Float             // Root Mean Squared Error
        public let maxError: Float        // Maximum absolute error
    }
    
    /// Aggregate statistics for a batch
    public struct BatchStatistics {
        public let perVectorMetrics: [VectorMetrics]
        public let averageMSE: Float
        public let averagePSNR: Float
        public let averageRMSE: Float
        public let worstCaseMSE: Float
        public let bestCaseMSE: Float
        public let percentile95MSE: Float  // 95th percentile MSE
        
        /// Vectors with MSE above threshold
        public func poorQualityIndices(mseThreshold: Float) -> [Int]
        
        /// Generate quality report
        public func qualityReport() -> String
    }
    
    /// Configuration for statistics computation
    public struct StatisticsConfig {
        public let computePercentiles: Bool
        public let computeMaxError: Bool
        
        public init(
            computePercentiles: Bool = true,
            computeMaxError: Bool = false
        ) {
            self.computePercentiles = computePercentiles
            self.computeMaxError = computeMaxError
        }
    }
    
    // MARK: - Initialization
    
    public init(device: MTLDevice) throws {
        // Load kernel "computeQuantizationStats"
        // Create pipeline state
    }
    
    // MARK: - Core Operations
    
    /// Compute quantization statistics for vectors
    /// - Parameters:
    ///   - original: Original float vectors
    ///   - quantized: Quantized/reconstructed vectors (same shape as original)
    ///   - config: Configuration options
    /// - Returns: Batch statistics with per-vector metrics
    public func computeStatistics(
        original: [[Float]],
        quantized: [[Float]],
        config: StatisticsConfig = StatisticsConfig()
    ) throws -> BatchStatistics
    
    /// Compute statistics for single vector pair
    public func computeStatistics(
        original: [Float],
        quantized: [Float]
    ) throws -> VectorMetrics
    
    // MARK: - Specialized Methods
    
    /// Compare multiple quantization methods
    /// - Parameters:
    ///   - original: Original vectors
    ///   - quantizedVersions: Dictionary of method name -> quantized vectors
    /// - Returns: Comparison report
    public func compareQuantizationMethods(
        original: [[Float]],
        quantizedVersions: [String: [[Float]]]
    ) throws -> ComparisonReport
    
    /// Monitor quantization quality over time (for streaming)
    public func createQualityMonitor(
        windowSize: Int = 100
    ) -> QualityMonitor
    
    // MARK: - Async Operations
    
    public func computeStatisticsAsync(
        original: [[Float]],
        quantized: [[Float]]
    ) async throws -> BatchStatistics
    
    // MARK: - Utility Types
    
    public struct ComparisonReport {
        public let methods: [String]
        public let statistics: [String: BatchStatistics]
        public let ranking: [(method: String, averagePSNR: Float)]
        
        public func bestMethod() -> String
        public func printComparison()
    }
    
    public class QualityMonitor {
        public func update(original: [Float], quantized: [Float]) throws
        public func currentStatistics() -> BatchStatistics
        public func isQualityDegrading(threshold: Float) -> Bool
    }
}
```

### Implementation Notes
- Thread configuration: One thread per vector (not per dimension)
- Grid size: Number of vectors
- Memory requirements: 2 floats output per vector (MSE and PSNR)
- PSNR interpretation:
  - > 40 dB: Excellent quality
  - 30-40 dB: Good quality
  - 20-30 dB: Acceptable quality
  - < 20 dB: Poor quality
- Consider batching for very large vector sets
- The kernel processes each vector independently, making it embarrassingly parallel

---

## Common Implementation Patterns

### Error Handling
```swift
// Use the project's AccelerationError enum
throw AccelerationError.invalidDimensions("Vectors must have same dimension")
throw AccelerationError.bufferCreationFailed("Failed to create Metal buffer")
throw AccelerationError.kernelExecutionFailed("Kernel execution failed")
```

### Buffer Creation Pattern
```swift
guard let buffer = device.makeBuffer(
    bytes: data,
    length: data.count * MemoryLayout<Float>.stride,
    options: .storageModeShared
) else {
    throw AccelerationError.bufferCreationFailed("Failed to create buffer")
}
```

### Command Buffer Pattern
```swift
guard let commandBuffer = computeEngine.commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else {
    throw AccelerationError.executionFailed("Failed to create command encoder")
}

encoder.setComputePipelineState(pipelineState)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)
encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 2)

let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
let threadgroups = MTLSize(
    width: (count + 255) / 256,
    height: 1,
    depth: 1
)

encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
encoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

### VectorCore Protocol Integration
All kernels should support VectorCore protocol types where applicable:
```swift
public func compute<V: VectorProtocol>(_ vector: V) throws -> Result 
    where V.Scalar == Float {
    let array = Array(vector.scalars)
    return try compute(array)
}
```

## Testing Recommendations

1. **Unit Tests**: Test with various dimensions (32, 64, 128, 512, 1024, 1536)
2. **Edge Cases**: Empty vectors, single dimension, very large dimensions
3. **Accuracy Tests**: Compare with CPU reference implementations
4. **Performance Tests**: Benchmark against CPU and measure speedup
5. **Memory Tests**: Verify no memory leaks with large batches

## Performance Considerations

1. **Batch Operations**: Always prefer batch operations over individual calls
2. **Memory Transfer**: Minimize CPU-GPU memory transfers
3. **Pipeline States**: Reuse pipeline states across calls
4. **Thread Configuration**: Match kernel's threadgroup size (usually 256)
5. **Async Operations**: Use async variants for better concurrency

## Integration Notes

- These wrappers should be added to the project's Package.swift targets
- Import them in the main VectorAccelerate module exports
- Add documentation with usage examples
- Consider adding convenience methods for common use cases
- Follow the project's existing naming conventions and patterns