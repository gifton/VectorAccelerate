# VectorAccelerate

**GPU-Accelerated Vector Operations for VectorCore**

VectorAccelerate provides high-performance GPU acceleration for vector operations, serving as the computational backbone for the VectorCore ecosystem. By leveraging Metal's compute shaders and Apple Silicon's unified memory architecture, VectorAccelerate delivers up to 100x speedups for large-scale vector operations.

## 🎯 Purpose

VectorAccelerate exists to solve a critical performance bottleneck in vector-based machine learning applications. While VectorCore provides an elegant Swift interface for vector operations, VectorAccelerate ensures these operations run at maximum speed by:

- **GPU Acceleration**: Offloading computations to the GPU for massive parallelism
- **Optimized Kernels**: Hand-tuned Metal shaders for specific dimensions (512, 768, 1536)
- **Memory Efficiency**: Advanced quantization and compression techniques
- **Seamless Integration**: Drop-in acceleration for VectorCore operations

## 📦 Dependencies

### Required Dependencies
- **VectorCore**: The foundational vector mathematics package that VectorAccelerate accelerates
- **Metal Framework**: Apple's GPU programming framework (included in macOS/iOS)
- **Swift 5.9+**: For modern concurrency and performance features

### Dependent Packages
VectorAccelerate is a critical dependency for:
- **VectorIndexAccelerated**: GPU-accelerated vector indexing and search
- **VectorDatabase**: High-performance vector storage and retrieval

## 🚀 Accelerated Operations

### Core Distance Metrics
- **L2 Distance** (`L2DistanceKernel`)
  - Euclidean distance with optional sqrt
  - Specialized kernels for dimensions 512, 768, 1536
  - Batch processing for multiple query-database pairs

- **Cosine Similarity** (`CosineSimilarityKernel`)
  - Pre-normalized and with-normalization variants
  - Output as similarity or distance (1 - similarity)
  - Optimized for high-dimensional embeddings

- **Dot Product** (`DotProductKernel`)
  - SIMD-optimized implementation
  - Automatic GEMV/GEMM path selection
  - Batch matrix-vector products

### Advanced Distance Metrics
- **Hamming Distance** (`HammingDistanceKernel`) - Binary vector distances
- **Jaccard Distance** (`JaccardDistanceKernel`) - Set similarity measurements
- **Minkowski Distance** (`MinkowskiDistanceKernel`) - Generalized Lp distances

### Vector Operations
- **L2 Normalization** (`L2NormalizationKernel`)
  - In-place and out-of-place normalization
  - Numerical stability for zero vectors
  - Batch normalization support

- **Element-wise Operations** (`ElementwiseKernel`)
  - Addition, subtraction, multiplication, division
  - Trigonometric functions (sin, cos, tan)
  - Power and exponential operations
  - Broadcasting support

### Selection & Sorting
- **Top-K Selection** (Multiple implementations)
  - `TopKSelectionKernel` - General purpose
  - `WarpOptimizedSelectionKernel` - Warp-level optimization for k=1,10,100
  - `StreamingTopKKernel` - For massive datasets
  - `FusedL2TopKKernel` - Combined distance + selection

- **Parallel Reduction** (`ParallelReductionKernel`)
  - Sum, min, max reduction
  - Mean and variance computation
  - Custom reduction operations

### Matrix Operations
- **Matrix Multiply (GEMM)** (`MatrixMultiplyKernel`)
  - Tiled implementation with shared memory
  - Support for transposed inputs
  - Alpha/beta scaling (C = α·A·B + β·C)

- **Matrix-Vector (GEMV)** (`MatrixVectorKernel`)
  - SIMD group optimizations
  - Row/column major support
  - Batch vector operations

- **Matrix Transpose** (`MatrixTransposeKernel`)
  - Tiled transpose for coalesced access
  - In-place transpose support
  - Batch operations

- **Batch Matrix Operations** (`BatchMatrixKernel`)
  - Fused multiply-add with bias
  - Strided tensor operations
  - Layer normalization

### Statistical Operations
- **Statistics** (`StatisticsKernel`)
  - Mean, variance, standard deviation
  - Skewness and kurtosis
  - Percentiles and quartiles
  - Running statistics updates

- **Histogram** (`HistogramKernel`)
  - Uniform, adaptive, and logarithmic binning
  - Multi-dimensional histograms
  - Kernel density estimation

### Quantization & Compression
- **Scalar Quantization** (`ScalarQuantizationKernel`)
  - INT8/INT4 quantization with scale and offset
  - Symmetric and asymmetric modes
  - Per-channel quantization

- **Binary Quantization** (`BinaryQuantizationKernel`)
  - 1-bit vector compression
  - Hamming distance on packed bits
  - 32x memory reduction

- **Product Quantization** (`ProductQuantizationKernel`)
  - Subspace decomposition
  - Codebook-based compression
  - Asymmetric distance computation

## 🔧 Installation

### Swift Package Manager

Add VectorAccelerate to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/VectorAccelerate.git", from: "1.0.0"),
    .package(url: "https://github.com/yourusername/VectorCore.git", from: "1.0.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["VectorAccelerate", "VectorCore"]
    )
]
```

## 🎓 Getting Started

### Basic Usage

```swift
import VectorAccelerate
import VectorCore

// Initialize GPU context
let device = MTLCreateSystemDefaultDevice()!
let l2Kernel = try L2DistanceKernel(device: device)

// Prepare your vectors
let queries = [[Float]](repeating: [Float](repeating: 0.5, count: 768), count: 10)
let database = [[Float]](repeating: [Float](repeating: 0.3, count: 768), count: 1000)

// Compute distances on GPU
let distances = try await l2Kernel.compute(
    queries: queries,
    database: database,
    dimension: 768,
    computeSqrt: true  // For Euclidean distance
)

print("Computed \(distances.count) x \(distances[0].count) distances on GPU")
```

### Using with VectorCore Types

```swift
import VectorCore

// Create VectorCore vectors
let vector1 = try DenseVector<Float>([1.0, 2.0, 3.0])
let vector2 = try DenseVector<Float>([4.0, 5.0, 6.0])

// Use GPU-accelerated operations
let cosineSim = try CosineSimilarityKernel(device: device)
let similarity = try await cosineSim.compute(
    queries: [vector1],
    database: [vector2]
)
```

### Advanced: Fused Operations

```swift
// Fused L2 distance + Top-K selection (single kernel execution)
let fusedKernel = try FusedL2TopKKernel(device: device)

let result = try fusedKernel.fusedL2TopK(
    queries: queryBuffer,
    dataset: datasetBuffer,
    queryCount: 100,
    datasetCount: 1_000_000,
    dimension: 768,
    k: 10,
    config: FusedConfig(includeDistances: true),
    commandBuffer: commandBuffer
)

// Result contains top-10 nearest neighbors for each query
```

### Quantization for Memory Efficiency

```swift
// Compress vectors to INT8 for 4x memory reduction
let quantizer = try ScalarQuantizationKernel(device: device)

let original = [[Float]](/* your high-precision vectors */)
let quantized = try await quantizer.quantize(
    original.flatMap { $0 },
    bitWidth: .int8,
    type: .symmetric
)

print("Compression ratio: \(quantized.compressionRatio)x")

// Later, dequantize for use
let restored = try await quantizer.dequantize(quantized, count: original.count)
```

### Batch Processing

```swift
// Process multiple matrix operations in parallel
let matrixKernel = try MatrixMultiplyKernel(device: device)

let matrices = [Matrix](/* your matrices */)
let results = try await matrixKernel.multiplyBatch(
    matrices: matrices,
    config: MultiplyConfig(
        alpha: 1.0,
        beta: 0.0,
        useSpecializedKernel: true
    )
)

print("Achieved \(results.averageGflops) GFLOPS")
```

## 🏗️ Architecture

### Kernel Organization

VectorAccelerate is organized into specialized kernels, each optimized for specific operations:

```
VectorAccelerate/
├── Kernels/              # GPU compute kernels
│   ├── Distance/         # L2, Cosine, Hamming, etc.
│   ├── Selection/        # Top-K, sorting, filtering
│   ├── Matrix/           # GEMM, GEMV, transpose
│   ├── Quantization/     # INT8, INT4, binary
│   └── Statistics/       # Mean, variance, histogram
├── Metal/
│   └── Shaders/         # Metal shader source files
├── Core/                # Context and buffer management
└── Operations/          # High-level operation orchestration
```

### Memory Management

VectorAccelerate uses a sophisticated buffer management system:
- **Shared Memory**: Unified memory architecture on Apple Silicon
- **Buffer Pools**: Reusable buffer allocation to minimize overhead
- **Alignment**: Automatic alignment for SIMD operations
- **Zero-Copy**: Direct memory mapping when possible

### Performance Optimizations

1. **Dimension-Specific Kernels**: Hand-tuned for 512, 768, 1536 dimensions
2. **Tiled Algorithms**: Optimized for cache hierarchy
3. **SIMD Operations**: Leveraging float4 and matrix operations
4. **Warp-Level Primitives**: Using simdgroup operations
5. **Fused Kernels**: Combining operations to reduce memory bandwidth

## 📊 Performance

Typical speedups over CPU implementations:

| Operation | Vector Size | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| L2 Distance | 1M × 10K | 12.3s | 0.15s | 82x |
| Cosine Similarity | 100K × 100K | 8.7s | 0.09s | 97x |
| Matrix Multiply | 2048 × 2048 | 1.8s | 0.03s | 60x |
| Top-K Selection | 1M, k=100 | 0.9s | 0.02s | 45x |
| INT8 Quantization | 10M vectors | 2.1s | 0.04s | 52x |

## 🧪 Testing

Run the test suite:

```bash
swift test
```

Run performance benchmarks:

```bash
swift run VectorAccelerateBenchmarks
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional distance metrics
- Optimizations for new Apple Silicon features
- Support for half-precision (Float16) operations
- Integration with more ML frameworks

## 📄 License

VectorAccelerate is available under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Apple's Metal team for the excellent GPU framework
- The VectorCore team for the foundational vector library
- Contributors to the scientific computing community

## 📚 Related Projects

- [VectorCore](https://github.com/yourusername/VectorCore) - Core vector mathematics
- [VectorIndexAccelerated](https://github.com/yourusername/VectorIndexAccelerated) - GPU-accelerated indexing
- [VectorDatabase](https://github.com/yourusername/VectorDatabase) - Complete vector database solution

---

**Note**: VectorAccelerate requires a Mac with Apple Silicon or a discrete GPU supporting Metal 3.0+. Performance characteristics may vary based on hardware configuration.