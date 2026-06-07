# VectorAccelerate

**GPU-Accelerated Vector Operations for VectorCore — Metal 4 Edition**

VectorAccelerate provides high-performance GPU acceleration for vector operations, serving as the computational backbone for the VectorCore ecosystem. By leveraging Metal 4's compute shaders, unified command encoding, and Apple Silicon's unified memory architecture, VectorAccelerate delivers up to 100x speedups for large-scale vector operations.

> **⚠️ Version 0.5.0**: Requires **Metal 4** (macOS 26.0+, iOS 26.0+, visionOS 3.0+). For older OS support, use VectorAccelerate 0.2.x
> 
> **⚠️ This package is still experimental, with development and real-world testing in progress** for Production grade Vector operations see VectorCore and VectorIndex's CPU-bound implementation

## 🎯 Purpose

VectorAccelerate exists to solve a critical performance bottleneck in vector-based machine learning applications on in storage, thermal, and power constrained environments. While VectorCore provides an elegant Swift interface for vector operations, VectorAccelerate ensures these operations run at maximum speed by:

- **Metal 4 Acceleration**: Leveraging unified command encoding and tensor operations
- **Optimized Kernels**: Hand-tuned Metal shaders for specific dimensions (512, 768, 1536)
- **Memory Efficiency**: Advanced quantization and compression techniques
- **ML Integration**: Experimental learned distance metrics with MLTensor support
- **Seamless Integration**: Drop-in acceleration for VectorCore operations

## 🏛️ Two-Layer API Architecture

VectorAccelerate provides **two complementary API layers** designed for different use cases:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: High-Level API (AcceleratedVectorIndex)           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Complete vector search solution                     │  │
│  │  • insert(), search(), remove(), compact()             │  │
│  │  • Automatic GPU/CPU routing                           │  │
│  │  • Best for: Applications needing similarity search    │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Low-Level API (25+ GPU Kernel Primitives)         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Direct GPU kernel access                            │  │
│  │  • Distance, Selection, Quantization, Matrix kernels   │  │
│  │  • Pipeline composition with encode() methods          │  │
│  │  • Best for: Custom ML pipelines, fine-grained control │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: High-Level API

For most applications, use `AcceleratedVectorIndex`:

```swift
import VectorAccelerate

// Create a flat index (exact search)
let index = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768, capacity: 10_000)
)

// Insert vectors and get handles
let handle = try await index.insert(embedding)

// Search for nearest neighbors
let results = try await index.search(query: queryVector, k: 10)
for result in results {
    print("Handle: \(result.id), Distance: \(result.distance)")
}
```

### Layer 2: Low-Level Kernel Primitives

For custom pipelines or maximum control, use kernels directly:

```swift
import VectorAccelerate

let context = try await Metal4Context()

// Use individual kernels
let l2Kernel = try await L2DistanceKernel(context: context)
let distances = try await l2Kernel.compute(
    queries: queryVectors,
    database: databaseVectors,
    computeSqrt: true
)

// Compose kernels in a pipeline
let normKernel = try await L2NormalizationKernel(context: context)
let cosineKernel = try await CosineSimilarityKernel(context: context)
let topKKernel = try await TopKSelectionKernel(context: context)

try await context.executeAndWait { _, encoder in
    normKernel.encode(into: encoder, ...)
    encoder.memoryBarrier(scope: .buffers)
    cosineKernel.encode(into: encoder, ...)
    encoder.memoryBarrier(scope: .buffers)
    topKKernel.encode(into: encoder, ...)
}
```

### Convenient Type Aliases

```swift
// Context
let context: GPUContext = try await Metal4Context()

// Distance kernels
let l2 = try await L2Kernel(context: context)
let cosine = try await CosineKernel(context: context)
let dot = try await DotKernel(context: context)

// Selection kernels
let topK = try await TopKKernel(context: context)
let fused = try await FusedTopKKernel(context: context)

// Quantization kernels
let binary = try await BinaryQuantKernel(context: context)
let scalar = try await ScalarQuantKernel(context: context)
```

## 📦 Requirements

### System Requirements
- **macOS 26.0+** / **iOS 26.0+** / **tvOS 26.0+** / **visionOS 3.0+**
- **Metal 4** capable device (Apple Silicon)
- **Swift 6.0+**

### Dependencies
- **VectorCore 0.2.0**: The foundational vector mathematics package

### Products
- **VectorAccelerate**: Core GPU acceleration library

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
- **Hamming Distance** (`HammingDistanceKernel`) - Binary vector distances
- **Minkowski Distance** (`MinkowskiDistanceKernel`) - Generalized Lp distances (includes L1/Manhattan and L∞/Chebyshev)
- **Jaccard Distance** (`JaccardDistanceKernel`) - Set similarity for sparse data

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

## 🗂️ GPU-Accelerated Vector Index

GPU-first vector index for high-performance similarity search.

### Quick Start

```swift
import VectorAccelerate

// Create a GPU-accelerated flat index
let config = IndexConfiguration.flat(dimension: 768, capacity: 10_000)
let index = try await AcceleratedVectorIndex(configuration: config)

// Insert vectors
let handle = try await index.insert(embedding, metadata: ["type": "document"])

// Search (returns L2² distances - native GPU format)
let results = try await index.search(query: queryVector, k: 10)

// Filtered search
let filtered = try await index.search(query: queryVector, k: 10) { handle, meta in
    meta?["type"] == "document"
}
```

### Index Types

| Type | Use Case | Performance |
|------|----------|-------------|
| **Flat** | Small-medium datasets (<100K) | Exact results, ~0.3ms search |
| **IVF** | Large datasets (100K+) | Approximate, faster at scale |

### Handle Stability

`VectorHandle` instances are **stable** — they remain valid across `compact()` operations:

```swift
let handle = try await index.insert(embedding)

// Delete some vectors
try await index.remove(otherHandle)

// Compact to reclaim space
try await index.compact()

// Original handle still works!
let vector = try await index.vector(for: handle)  // ✓ Valid
```

Internally, handles use an indirection table that maps stable IDs to storage slots, so compaction can relocate vectors without invalidating user-held handles.

### Performance

- **Insert**: ~21K vectors/sec (128D), ~3.7K vectors/sec (768D)
- **Search**: 0.30ms (128D), 0.73ms (768D) on 5K vectors
- **Sub-millisecond** latency for typical workloads

## 🔧 Installation

### Swift Package Manager

Add VectorAccelerate to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/gifton/VectorAccelerate.git", from: "0.5.0"),
    .package(url: "https://github.com/gifton/VectorCore.git", from: "0.2.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            "VectorAccelerate",
            "VectorCore"
        ]
    )
]
```

> **Note**: Version 0.4.2+ requires Metal 4. For macOS 15 / iOS 18 support, use version 0.1.x.

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

// Prepare query and dataset vectors
let queries = [[Float]](repeating: [Float](repeating: 0.5, count: 768), count: 10)
let dataset = [[Float]](repeating: [Float](repeating: 0.3, count: 768), count: 1000)

let results = try await fusedKernel.findNearestNeighbors(
    queries: queries,
    dataset: dataset,
    k: 10
)

// Result contains top-10 nearest neighbors for each query
for (queryIndex, neighbors) in results.enumerated() {
    print("Query \(queryIndex): \(neighbors.results.count) neighbors found")
}
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

### VectorCore Integration

VectorAccelerate provides GPU-accelerated `DistanceProvider` implementations:

```swift
import VectorAccelerate
import VectorCore

// Create a kernel-backed distance provider
let context = try await Metal4Context()
let provider = try await L2KernelDistanceProvider(context: context)

// Works with VectorCore's DynamicVector
let v1 = DynamicVector([1.0, 0.0, 0.0])
let v2 = DynamicVector([0.0, 1.0, 0.0])

let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)

// Batch distance computation
let candidates = [v1, v2, DynamicVector([0.5, 0.5, 0.0])]
let distances = try await provider.batchDistance(from: v1, to: candidates, metric: .euclidean)
```

**Universal Distance Provider** - handles all metrics automatically:

```swift
// Dispatches to the optimal kernel for each metric
let provider = context.universalDistanceProvider()

let euclidean = try await provider.distance(from: v1, to: v2, metric: .euclidean)
let cosine = try await provider.distance(from: v1, to: v2, metric: .cosine)
let manhattan = try await provider.distance(from: v1, to: v2, metric: .manhattan)
```

**Available Distance Providers:**

| Provider | Metric | Features |
|----------|--------|----------|
| `L2KernelDistanceProvider` | Euclidean | Dimension-optimized (384, 512, 768, 1536) |
| `CosineKernelDistanceProvider` | Cosine | Auto-normalization |
| `DotProductKernelDistanceProvider` | Dot Product | GEMV optimization |
| `MinkowskiKernelDistanceProvider` | Manhattan, Chebyshev | Configurable p-norm |
| `JaccardKernelDistanceProvider` | Jaccard | Set similarity (specialized API) |
| `HammingKernelDistanceProvider` | Hamming | Binary vectors (specialized API) |
| `UniversalKernelDistanceProvider` | All | Auto-dispatch |

## 📋 Choosing the Right Kernel

### Distance Computation

| Use Case | Recommended Kernel | Notes |
|----------|-------------------|-------|
| Nearest neighbor search | `L2DistanceKernel` | Best for embeddings |
| Semantic similarity | `CosineSimilarityKernel` | Direction-based |
| Maximum inner product | `DotProductKernel` | For unnormalized vectors |
| Sparse data / sets | `JaccardDistanceKernel` | Document fingerprints |
| Binary vectors | `HammingDistanceKernel` | After binary quantization |
| Custom Lp norm | `MinkowskiDistanceKernel` | Configurable p |

### Selection

| Use Case | Recommended Kernel | Notes |
|----------|-------------------|-------|
| Standard top-k | `TopKSelectionKernel` | General purpose |
| Memory-constrained | `FusedL2TopKKernel` | Avoids full distance matrix, auto-fallback for K > 8 |
| Large datasets | `FusedL2TopKKernel` | Uses chunked GPU merge for memory efficiency |
| SIMD-optimized | `WarpOptimizedSelectionKernel` | Small k values (k ≤ 32) |

> **Note**: `StreamingTopKKernel` is deprecated due to known correctness issues. Use `FusedL2TopKKernel` which automatically handles large datasets via chunked GPU merge.

### Quantization

| Use Case | Recommended Kernel | Compression |
|----------|-------------------|-------------|
| Fast approximate search | `BinaryQuantizationKernel` | 32x |
| Quality-preserving | `ScalarQuantizationKernel` | 4-8x |
| High compression | `ProductQuantizationKernel` | 32-64x |

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

1. **Hierarchical SIMD Reductions**: Overhauled L2 and Cosine kernels using a 4-phase reduction model (`Local -> Warp -> Threadgroup -> Global`), maximizing 128-bit memory bus saturation and eliminating global atomic stalls.
2. **Tiled Shared Memory Algorithms**: KMeans assignment dynamically scales centroid tiles to fit within hardware limits (32KB), using register-cached compute loops to reduce global memory pressure by 32x.
3. **2-Pass GPU Orchestration**: K-Means update logic uses a "Cooperative Gather" topology, collaboratively summing dimensions via SIMD-group operations for maximum throughput.
4. **Enforced Asynchronous Execution**: Fully non-blocking execution model using `await commitAndWait()` and Swift 6 concurrency, ensuring zero OS thread stalls during GPU work.
5. **Dynamic Buffer Pooling**: Ring-buffer strategy with Power-of-2 bucketing eliminates allocation overhead in hot loops, with `BufferToken` anchoring for safe asynchronous memory reclamation.
6. **Eager Pipeline Pre-compilation**: Background pre-compilation of critical path kernels during initialization to eliminate cold-start latency.
7. **Tiled GEMM Neural Encoder**: High-performance 2-pass neural encoder using a Full-D register loop and shared memory padding to eliminate bank conflicts and global atomic bottlenecks.
8. **Vectorized Transposed Decoder**: Optimized dequantization path using dual-accumulator latency hiding and dimension-specific loop unrolling for a 2x throughput gain.

## 📊 Performance

Typical speedups over CPU implementations:

| Operation | Vector Size | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| L2 Distance | 1M × 10K | 12.3s | 0.15s | 82x |
| Cosine Similarity | 100K × 100K | 8.7s | 0.09s | 97x |
| Matrix Multiply | 2048 × 2048 | 1.8s | 0.03s | 60x |
| Top-K Selection | 1M, k=100 | 0.9s | 0.02s | 45x |
| INT8 Quantization | 10M vectors | 2.1s | 0.04s | 52x |

## ⚠️ Known Limitations

### Fused L2 Top-K Kernel

The `FusedL2TopKKernel` uses different strategies based on K:

| K Range | Strategy | Memory |
|---------|----------|--------|
| K ≤ 8 | Fused single-pass | O(Q × K) - no distance matrix |
| 8 < K ≤ 32 | Two-pass with warp selection | O(Q × N) distance matrix |
| K > 32 | Two-pass with standard selection | O(Q × N) distance matrix |
| Large N | Chunked GPU merge | Bounded by `maxDistanceMatrixBytes` |

For K > 8, the kernel automatically falls back to a two-pass approach. For very large datasets, it uses chunked processing with GPU-side merge to stay within memory bounds.

### IVF Index (Work in Progress)

The IVF index has known correctness issues being addressed:
- Batch search may not correctly isolate per-query candidate lists
- Training with many deletions may cause duplicate entries

For production use, prefer the **flat index** until these are resolved (see `QUALITY_IMPROVEMENT_ROADMAP.md` P0.1-P0.4).

### Metal Shader Compilation

Shaders are validated with `-std=metal3.0` in CI for syntax checking, but the runtime requires **Metal 4** features (macOS 26.0+). This is intentional — Metal 3 syntax is a subset of Metal 4.

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
- [VectorIndex](https://github.com/gifton/VectorIndex) - CPU-based vector indexing algorithms
- [VectorDatabase](https://github.com/gifton/VectorDatabase) - Complete vector database solution

---

**Requirements**: VectorAccelerate 0.4.2+ requires **Metal 4** (macOS 26.0+, iOS 26.0+, visionOS 3.0+) and Apple Silicon. For older OS versions, use VectorAccelerate 0.1.x.
