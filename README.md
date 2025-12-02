# VectorAccelerate

**GPU-Accelerated Vector Operations for VectorCore — Metal 4 Edition**

VectorAccelerate provides high-performance GPU acceleration for vector operations, serving as the computational backbone for the VectorCore ecosystem. By leveraging Metal 4's compute shaders, unified command encoding, and Apple Silicon's unified memory architecture, VectorAccelerate delivers up to 100x speedups for large-scale vector operations.

> **⚠️ Version 0.2.0 Breaking Change**: This version requires **Metal 4** (macOS 26.0+, iOS 26.0+, visionOS 3.0+). For older OS support, use VectorAccelerate 0.1.x.

## 🎯 Purpose

VectorAccelerate exists to solve a critical performance bottleneck in vector-based machine learning applications. While VectorCore provides an elegant Swift interface for vector operations, VectorAccelerate ensures these operations run at maximum speed by:

- **Metal 4 Acceleration**: Leveraging unified command encoding and tensor operations
- **Optimized Kernels**: Hand-tuned Metal shaders for specific dimensions (512, 768, 1536)
- **Memory Efficiency**: Advanced quantization and compression techniques
- **ML Integration**: Experimental learned distance metrics with MLTensor support
- **Seamless Integration**: Drop-in acceleration for VectorCore operations

## 📦 Requirements

### System Requirements
- **macOS 26.0+** / **iOS 26.0+** / **tvOS 26.0+** / **visionOS 3.0+**
- **Metal 4** capable device (Apple Silicon)
- **Swift 6.0+**

### Dependencies
- **VectorCore 0.1.5+**: The foundational vector mathematics package

### Dependent Packages
VectorAccelerate is a critical dependency for:
- **VectorIndexAccelerated**: GPU-accelerated vector indexing and search
- **VectorDatabase**: High-performance vector storage and retrieval

## 🚀 Accelerated Operations

All kernels require a `Metal4Context` for initialization.

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
- **Manhattan Distance** (`ManhattanDistanceKernel`) - L1 distance
- **Chebyshev Distance** (`ChebyshevDistanceKernel`) - L∞ distance
- **Hamming Distance** (`HammingDistanceKernel`) - Binary vector distances
- **Minkowski Distance** (`MinkowskiDistanceKernel`) - Generalized Lp distances

### Experimental ML Features
- **Learned Distance** (`LearnedDistanceKernel`)
  - Projection-based learned metrics
  - MLTensor integration for neural distance computation
  - Automatic fallback to standard L2 when unavailable

- **Attention Similarity** (`AttentionSimilarityKernel`)
  - Attention-weighted similarity computation
  - Multi-head attention support

- **Neural Quantization** (`NeuralQuantizationKernel`)
  - Learned quantization with neural networks
  - Adaptive codebook generation

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
- **Top-K Selection** (`TopKSelectionKernel`)
  - General purpose top-k with configurable k
  - Warp-level optimization for common k values
  - Streaming support for massive datasets

- **Fused L2 + Top-K** (`FusedL2TopKKernel`)
  - Combined distance computation and selection
  - Reduced memory bandwidth
  - Optimal for nearest neighbor search

- **Parallel Reduction** (`ParallelReductionKernel`)
  - Sum, min, max reduction
  - Mean and variance computation
  - Custom reduction operations

### Matrix Operations
- **Matrix Multiply (GEMM)** (`MatrixMultiplyKernel`)
  - Tiled implementation with shared memory (32×32×8 tiles)
  - Support for transposed inputs
  - Alpha/beta scaling (C = α·A·B + β·C)

- **Matrix-Vector (GEMV)** (`MatrixVectorKernel`)
  - SIMD group optimizations
  - Row/column major support
  - Batch vector operations

- **Matrix Transpose** (`MatrixTransposeKernel`)
  - Tiled transpose for coalesced access
  - Optimized shared memory usage

- **Batch Matrix Operations** (`BatchMatrixKernel`)
  - Fused multiply-add with bias
  - Strided tensor operations

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
    .package(url: "https://github.com/gifton/VectorAccelerate.git", from: "0.2.0"),
    .package(url: "https://github.com/gifton/VectorCore.git", from: "0.1.5")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["VectorAccelerate", "VectorCore"]
    )
]
```

> **Note**: Version 0.2.0+ requires Metal 4. For macOS 15 / iOS 18 support, use version 0.1.x.

## 🎓 Getting Started

### Basic Usage

```swift
import VectorAccelerate
import VectorCore

// Initialize Metal 4 context (async)
let context = try await Metal4Context()
let l2Kernel = try await L2DistanceKernel(context: context)

// Prepare your vectors
let queries = [[Float]](repeating: [Float](repeating: 0.5, count: 768), count: 10)
let database = [[Float]](repeating: [Float](repeating: 0.3, count: 768), count: 1000)

// Compute distances on GPU
let distances = try await l2Kernel.compute(
    queries: queries,
    database: database,
    computeSqrt: true  // For Euclidean distance
)

print("Computed \(distances.count) x \(distances[0].count) distances on GPU")
```

### Using with VectorCore Types

```swift
import VectorCore

// Create VectorCore vectors
let vector1 = Vector<D768>([Float](repeating: 1.0, count: 768))
let vector2 = Vector<D768>([Float](repeating: 0.5, count: 768))

// Use GPU-accelerated operations
let context = try await Metal4Context()
let cosineSim = try await CosineSimilarityKernel(context: context)
let similarity = try await cosineSim.compute(
    queries: [vector1],
    database: [vector2]
)
```

### Advanced: Fused Operations

```swift
// Fused L2 distance + Top-K selection (single kernel execution)
let context = try await Metal4Context()
let fusedKernel = try await FusedL2TopKKernel(context: context)

let result = try await fusedKernel.compute(
    queries: queries,
    database: database,
    k: 10
)

// Result contains top-10 nearest neighbors for each query
print("Found \(result.indices[0].count) nearest neighbors per query")
```

### Learned Distance (Experimental ML)

```swift
// Use learned projections for distance computation
let context = try await Metal4Context()
let config = AccelerationConfiguration(enableExperimentalML: true)
let service = try await LearnedDistanceService(context: context, configuration: config)

// Load projection weights (e.g., from a trained model)
try await service.loadProjection(
    from: weightsURL,
    inputDim: 768,
    outputDim: 128
)

// Compute distances with learned projection
let (distances, mode) = try await service.computeL2(
    queries: queries,
    database: database
)

print("Computed using \(mode) mode")  // .learned or .standard (fallback)
```

### Matrix Operations

```swift
// GPU-accelerated matrix multiplication
let context = try await Metal4Context()
let matrixKernel = try await MatrixMultiplyKernel(context: context)

let a = Matrix.random(rows: 1024, columns: 512)
let b = Matrix.random(rows: 512, columns: 256)

let result = try await matrixKernel.multiply(a: a, b: b)
print("Result: \(result.rows) x \(result.columns)")
```

## 🏗️ Architecture

### Kernel Organization

VectorAccelerate is organized into specialized Metal 4 kernels:

```
VectorAccelerate/
├── Core/                    # Metal 4 infrastructure
│   ├── Metal4Context.swift      # Unified context management
│   ├── Metal4ComputeEngine.swift # Command encoding
│   ├── ResidencyManager.swift    # Memory residency
│   ├── PipelineCache.swift       # Pipeline state caching
│   └── TensorManager.swift       # Tensor operations
├── Kernels/
│   └── Metal4/              # All Metal 4 kernel implementations
│       ├── L2DistanceKernel.swift
│       ├── CosineSimilarityKernel.swift
│       ├── MatrixMultiplyKernel.swift
│       └── ... (20+ kernels)
├── Metal/
│   └── Shaders/             # Metal shader source files (.metal)
└── Operations/              # High-level operation orchestration
```

### Metal 4 Features Used

- **Unified Command Encoding**: Single encoder for compute + blit operations
- **Residency Sets**: Explicit memory management for optimal GPU utilization
- **Argument Tables**: Efficient parameter passing to shaders
- **Pipeline Harvesting**: Background pipeline compilation

### Performance Optimizations

1. **Dimension-Specific Kernels**: Hand-tuned for 512, 768, 1536 dimensions
2. **Tiled Algorithms**: 32×32×8 tiles optimized for Apple Silicon cache
3. **SIMD Operations**: Leveraging float4 and simdgroup matrix operations
4. **Fused Kernels**: Combined distance + selection for reduced memory bandwidth
5. **Async Pipeline Creation**: Non-blocking kernel initialization

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
- Optimizations for new Apple Silicon features (M4, M5)
- Enhanced ML integration
- Performance benchmarks and comparisons

## 📄 License

VectorAccelerate is available under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Apple's Metal team for the excellent GPU framework
- The VectorCore team for the foundational vector library
- Contributors to the scientific computing community

## 📚 Related Projects

- [VectorCore](https://github.com/gifton/VectorCore) - Core vector mathematics
- [VectorIndexAccelerated](https://github.com/gifton/VectorIndexAccelerated) - GPU-accelerated indexing
- [VectorDatabase](https://github.com/gifton/VectorDatabase) - Complete vector database solution

---

**Requirements**: VectorAccelerate 0.2.0+ requires **Metal 4** (macOS 26.0+, iOS 26.0+, visionOS 3.0+) and Apple Silicon. For older OS versions, use VectorAccelerate 0.1.x.