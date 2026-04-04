# VectorAccelerate File Inventory

> **Purpose:** Complete file listing for reference during Metal 4 migration.

## Summary Statistics

| Category | Files | Lines |
|----------|-------|-------|
| Core Infrastructure | 11 | ~5,200 |
| Kernels | 25 | ~14,500 |
| Metal Shaders | 18 | ~6,800 |
| Operations | 7 | ~3,500 |
| ML/Integration | 4 | ~2,500 |
| Configuration | 3 | ~800 |
| **Total** | **68** | **~33,300** |

---

## Core Infrastructure

| File | Lines | Purpose | Metal 4 Impact |
|------|-------|---------|----------------|
| `Core/ShaderManager.swift` | 964 | Shader compilation, pipeline caching | **HIGH** - MTL4Compiler |
| `Core/ComputeEngine.swift` | 667 | High-level compute operations | **HIGH** - Unified encoder |
| `Core/SharedMetalContext.swift` | 599 | Shared context management | **MEDIUM** - Update patterns |
| `Core/MetalContext.swift` | 529 | Main context actor | **HIGH** - Queue/buffer changes |
| `Core/BufferPool.swift` | 514 | Token-based buffer pooling | **HIGH** - Residency sets |
| `Core/MetalDevice.swift` | 445 | Device wrapper, factories | **HIGH** - Compiler, options |
| `Core/MetalBufferFactory.swift` | 357 | Buffer creation utilities | **MEDIUM** - GPU addresses |
| `Core/KernelContext.swift` | 326 | Shared kernel context | **MEDIUM** - Update patterns |
| `Core/Types.swift` | 323 | Type definitions | **LOW** - Minor updates |
| `Core/DistanceMetrics.swift` | 338 | Distance metric types | **LOW** - No changes |
| `Core/Logger.swift` | 312 | Logging utilities | **NONE** |

---

## Kernels (Swift)

| File | Lines | Purpose | Metal 4 Impact |
|------|-------|---------|----------------|
| `Kernels/HistogramKernel.swift` | 910 | Histogram computation | **MEDIUM** |
| `Kernels/StatisticsKernel.swift` | 837 | Statistical operations | **MEDIUM** |
| `Kernels/BinaryQuantizationKernel.swift` | 774 | Binary quantization | **MEDIUM** |
| `Kernels/DotProductKernel.swift` | 766 | Dot product (6 variants) | **MEDIUM** |
| `Kernels/QuantizationStatisticsKernel.swift` | 725 | Quant statistics | **MEDIUM** |
| `Kernels/ProductQuantizationKernel.swift` | 702 | Product quantization | **MEDIUM** |
| `Kernels/JaccardDistanceKernel.swift` | 692 | Jaccard distance | **MEDIUM** |
| `Kernels/MinkowskiCalculator.swift` | 652 | Minkowski distance | **MEDIUM** |
| `Kernels/ProductQuantizationKernelEnhanced.swift` | 637 | Enhanced PQ | **MEDIUM** |
| `Kernels/CosineSimilarityKernel.swift` | 637 | Cosine similarity (4 variants) | **MEDIUM** |
| `Kernels/MinkowskiDistanceKernel.swift` | 626 | Minkowski kernel | **MEDIUM** |
| `Kernels/WarpOptimizedSelectionKernel.swift` | 539 | Warp-optimized selection | **MEDIUM** |
| `Kernels/StreamingTopKKernel.swift` | 535 | Streaming top-K | **MEDIUM** |
| `Kernels/BatchMatrixKernel.swift` | 532 | Batch matrix ops | **MEDIUM** |
| `Kernels/L2DistanceKernel.swift` | 526 | L2 distance (4 variants) | **MEDIUM** |
| `Kernels/ScalarQuantizationKernel.swift` | 523 | Scalar quantization | **MEDIUM** |
| `Kernels/L2NormalizationKernel.swift` | 513 | L2 normalization | **MEDIUM** |
| `Kernels/FusedL2TopKKernel.swift` | 496 | Fused L2 + top-K | **MEDIUM** |
| `Kernels/HammingDistanceKernel.swift` | 480 | Hamming distance | **MEDIUM** |
| `Kernels/ParallelReductionKernel.swift` | 465 | Parallel reduction | **MEDIUM** |
| `Kernels/MatrixVectorKernel.swift` | 444 | Matrix-vector multiply | **MEDIUM** |
| `Kernels/ElementwiseKernel.swift` | 440 | Elementwise ops | **MEDIUM** |
| `Kernels/MatrixMultiplyKernel.swift` | 402 | Matrix multiply | **MEDIUM** |
| `Kernels/MatrixTransposeKernel.swift` | 338 | Matrix transpose | **MEDIUM** |
| `Kernels/TopKSelectionKernel.swift` | 324 | Top-K selection | **MEDIUM** |

---

## Metal Shaders

| File | Lines | Kernels | Metal 4 Impact |
|------|-------|---------|----------------|
| `Shaders/AdvancedTopK.metal` | 938 | 6 | **MEDIUM** - MSL 4.0 |
| `Shaders/CosineSimilarity.metal` | 550 | 4 | **LOW** - MSL 4.0 |
| `Shaders/BasicOperations.metal` | 547 | 7 | **LOW** - MSL 4.0 |
| `Shaders/MinkowskiDistance.metal` | 451 | 4 | **LOW** - MSL 4.0 |
| `Shaders/StatisticsShaders.metal` | 446 | 5 | **LOW** - MSL 4.0 |
| `Shaders/DotProduct.metal` | 430 | 5 | **LOW** - MSL 4.0 |
| `Shaders/DistanceShaders.metal` | 418 | 4 | **LOW** - MSL 4.0 |
| `Shaders/ChebyshevDistance.metal` | 417 | 3 | **LOW** - MSL 4.0 |
| `Shaders/ClusteringShaders.metal` | 398 | 3 | **LOW** - MSL 4.0 |
| `Shaders/ManhattanDistance.metal` | 345 | 3 | **LOW** - MSL 4.0 |
| `Shaders/L2Distance.metal` | 317 | 4 | **LOW** - MSL 4.0 |
| `Shaders/HammingDistance.metal` | 317 | 3 | **LOW** - MSL 4.0 |
| `Shaders/ProductQuantization.metal` | ~300 | 4 | **LOW** - MSL 4.0 |
| `Shaders/QuantizationShaders.metal` | ~280 | 4 | **LOW** - MSL 4.0 |
| `Shaders/L2Normalization.metal` | ~250 | 2 | **LOW** - MSL 4.0 |
| `Shaders/DataTransformations.metal` | ~200 | 3 | **LOW** - MSL 4.0 |
| `Shaders/OptimizedMatrixOps.metal` | ~200 | 2 | **LOW** - MSL 4.0 |
| `Shaders/SearchAndRetrieval.metal` | ~180 | 2 | **LOW** - MSL 4.0 |

---

## Operations

| File | Lines | Purpose | Metal 4 Impact |
|------|-------|---------|----------------|
| `Operations/MemoryMapManager.swift` | 624 | Memory-mapped files | **LOW** |
| `Operations/MatrixEngine.swift` | 573 | Matrix operations | **MEDIUM** |
| `Operations/SIMDFallback.swift` | 525 | CPU fallback | **NONE** |
| `Operations/BatchDistanceOperations.swift` | 493 | Batch operations | **MEDIUM** |
| `Operations/BatchProcessor.swift` | 449 | Batch processing | **MEDIUM** |
| `Operations/AccelerateFallback.swift` | 345 | Accelerate fallback | **NONE** |
| `Operations/SIMDOptimizer.swift` | 332 | SIMD optimization | **NONE** |

---

## ML & Integration

| File | Lines | Purpose | Metal 4 Impact |
|------|-------|---------|----------------|
| `Integration/VectorCoreIntegration.swift` | 840 | VectorCore bridge | **MEDIUM** |
| `ML/QuantizationEngine.swift` | 671 | Quantization engine | **MEDIUM** |
| `ML/EmbeddingEngine.swift` | 642 | Embedding operations | **HIGH** - MTLTensor |
| `Integration/CoreMLBridge.swift` | 404 | CoreML bridge | **MEDIUM** |

---

## Configuration & Utilities

| File | Lines | Purpose | Metal 4 Impact |
|------|-------|---------|----------------|
| `Configuration/AdaptiveThresholds.swift` | ~250 | Adaptive config | **LOW** |
| `Configuration/PerformanceMonitor.swift` | ~200 | Perf monitoring | **LOW** |
| `Configuration/AccelerationConfiguration.swift` | ~180 | Accel config | **LOW** |
| `Benchmarking/BenchmarkFramework.swift` | 342 | Benchmarking | **LOW** |
| `Core/AccelerationError.swift` | ~150 | Error types | **LOW** |
| `Core/ConcurrencyShims.swift` | ~100 | Async helpers | **HIGH** - Events |
| `Core/VectorError+GPU.swift` | ~80 | Error extensions | **LOW** |
| `VectorAccelerate.swift` | ~50 | Module exports | **NONE** |

---

## Migration Priority Order

### Phase 1: Foundation (High Impact)
1. `Core/MetalContext.swift`
2. `Core/MetalDevice.swift`
3. `Core/BufferPool.swift`
4. `Core/ConcurrencyShims.swift`

### Phase 2: Command Encoding (High Impact)
1. `Core/ComputeEngine.swift`
2. `Core/ShaderManager.swift`
3. `Core/KernelContext.swift`

### Phase 3: Shaders (Medium Impact)
1. All `Shaders/*.metal` files (MSL 4.0 headers)

### Phase 4: Kernels (Medium Impact, High Volume)
1. All `Kernels/*.swift` files (argument table binding)

### Phase 5: ML Integration (New Features)
1. `ML/EmbeddingEngine.swift`
2. New tensor-based kernels

---

## File Paths Reference

```
Sources/VectorAccelerate/
├── Core/
│   ├── AccelerationError.swift
│   ├── BufferPool.swift
│   ├── ComputeEngine.swift
│   ├── ConcurrencyShims.swift
│   ├── DistanceMetrics.swift
│   ├── KernelContext.swift
│   ├── Logger.swift
│   ├── MetalBufferFactory.swift
│   ├── MetalContext.swift
│   ├── MetalDevice.swift
│   ├── MetalTypes.swift
│   ├── SharedMetalContext.swift
│   ├── ShaderLibrary.swift
│   ├── ShaderManager.swift
│   ├── SmartBufferPool.swift
│   ├── Types.swift
│   └── VectorError+GPU.swift
├── Kernels/
│   ├── [25 kernel files]
├── Metal/Shaders/
│   ├── [18 shader files]
├── Operations/
│   ├── [7 operation files]
├── ML/
│   ├── EmbeddingEngine.swift
│   └── QuantizationEngine.swift
├── Integration/
│   ├── CoreMLBridge.swift
│   └── VectorCoreIntegration.swift
├── Configuration/
│   ├── [3 config files]
├── Benchmarking/
│   └── BenchmarkFramework.swift
└── VectorAccelerate.swift
```
