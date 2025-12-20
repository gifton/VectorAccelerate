# Kernel Migration Index

## Overview

This directory contains migration specifications for each VectorAccelerate kernel.
Each kernel requires updating both Swift (buffer binding) and Metal shader (MSL 4.0) code.

Use [TEMPLATE_KERNEL_TASK.md](TEMPLATE_KERNEL_TASK.md) to create task files for each kernel.

## Migration Priority

Kernels are prioritized by usage frequency and complexity.

### Legend

**Metal 4 Features:**
- `AT` = ArgumentTable (required for all)
- `SM` = SimdgroupMatrix (8x8/16x16 matrix ops)
- `MT` = MLTensor (Phase 4, optional)
- `B` = Barriers (between fused operations)

**Fusibility:** Other kernels that can share a command buffer/encoder for efficiency.

---

### Priority 1: Core Distance Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| L2Distance | `Metal4L2DistanceKernel.swift` | `L2Distance.metal` | AT | TopK, Normalize | **Complete** |
| CosineSimilarity | `Metal4CosineSimilarityKernel.swift` | `CosineSimilarity.metal` | AT | TopK, Normalize | **Complete** |
| DotProduct | `Metal4DotProductKernel.swift` | `DotProduct.metal` | AT, SM | TopK, MatrixOps | **Complete** |

**Notes:**
- These are the most frequently used kernels
- All can be fused with TopK selection for search pipelines
- DotProduct can benefit from simdgroup matrix ops

---

### Priority 2: Selection Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| TopKSelection | `Metal4TopKSelectionKernel.swift` | `AdvancedTopK.metal` | AT, B | Distance kernels | **Complete** |
| FusedL2TopK | `Metal4FusedL2TopKKernel.swift` | `AdvancedTopK.metal` | AT, B | (self-contained) | **Complete** |
| StreamingTopK | `Metal4StreamingTopKKernel.swift` | `AdvancedTopK.metal` | AT, B | Distance, Merge | **Complete** |
| WarpOptimizedSelection | `Metal4WarpOptimizedSelectionKernel.swift` | `AdvancedTopK.metal` | AT | Distance kernels | **Complete** |

**Notes:**
- TopK needs barriers when following distance computation
- StreamingTopK has complex barrier requirements (read/write same buffer)
- FusedL2TopK already combines distance + selection

---

### Priority 3: Quantization Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| ScalarQuantization | `Metal4ScalarQuantizationKernel.swift` | `QuantizationShaders.metal` | AT, MT* | Dequantize, Distance | **Complete** |
| BinaryQuantization | `Metal4BinaryQuantizationKernel.swift` | `QuantizationShaders.metal` | AT | Hamming | **Complete** |
| ProductQuantization | `Metal4ProductQuantizationKernel.swift` | `ProductQuantization.metal` | AT, SM, MT* | PQDistance | **Complete** |

**Notes:**
- `MT*` = MLTensor candidate for Phase 4 neural quantization
- ProductQuantization can benefit from simdgroup matrix for codebook ops
- Quantize → Distance → Dequantize is common pipeline

---

### Priority 4: Matrix Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| MatrixMultiply | `Metal4MatrixMultiplyKernel.swift` | `OptimizedMatrixOps.metal` | AT, SM | Accumulate | **Complete** |
| MatrixVector | `Metal4MatrixVectorKernel.swift` | `OptimizedMatrixOps.metal` | AT, SM | Distance | **Complete** |
| MatrixTranspose | `Metal4MatrixTransposeKernel.swift` | `DataTransformations.metal` | AT | MatrixMultiply | **Complete** |
| BatchMatrix | `Metal4BatchMatrixKernel.swift` | `OptimizedMatrixOps.metal` | AT, SM | (batched) | **Complete** |

**Notes:**
- All matrix ops should use simdgroup matrix (SM) for performance
- MatrixTranspose is often fused before MatrixMultiply
- BatchMatrix processes multiple independent operations

---

### Priority 5: Other Distance Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| Manhattan | (via Minkowski p=1) | `MinkowskiDistance.metal` | AT | TopK | **Complete** |
| Chebyshev | (via Minkowski p→∞) | `MinkowskiDistance.metal` | AT | TopK | **Complete** |
| Minkowski | `Metal4MinkowskiDistanceKernel.swift` | `MinkowskiDistance.metal` | AT | TopK | **Complete** |
| Hamming | `Metal4HammingDistanceKernel.swift` | `HammingDistance.metal` | AT | BinaryQuant, TopK | **Complete** |
| Jaccard | `Metal4JaccardDistanceKernel.swift` | `DistanceShaders.metal` | AT | TopK | **Complete** |

**Notes:**
- Lower priority due to less frequent usage
- All follow same pattern as core distance kernels
- Hamming specifically pairs with BinaryQuantization

---

### Priority 6: Utility Kernels

| Kernel | Swift File | Metal Shader | Metal 4 Features | Fusible With | Status |
|--------|------------|--------------|------------------|--------------|--------|
| L2Normalization | `Metal4L2NormalizationKernel.swift` | `L2Normalization.metal` | AT, B | Distance (pre-step) | **Complete** |
| ParallelReduction | `Metal4ParallelReductionKernel.swift` | `BasicOperations.metal` | AT | Statistics | **Complete** |
| Elementwise | `Metal4ElementwiseKernel.swift` | `BasicOperations.metal` | AT | Any (pre/post) | **Complete** |
| Statistics | `Metal4StatisticsKernel.swift` | `StatisticsShaders.metal` | AT | Reduction | **Complete** |
| Histogram | `Metal4HistogramKernel.swift` | `StatisticsShaders.metal` | AT | Statistics | **Complete** |

**Notes:**
- L2Normalization often fused before CosineSimilarity
- Elementwise ops (add, mul, etc.) can be fused into many pipelines
- Statistics/Histogram are standalone operations

---

## Generating Context Bundle

Use the bundle script to generate context for any kernel:

```bash
# From docs/metal4/scripts/
./bundle-kernel.sh L2Distance > ../05-kernel-migrations/l2-distance-bundle.md
./bundle-kernel.sh CosineSimilarity > ../05-kernel-migrations/cosine-bundle.md
```

## Common Migration Pattern

All kernels follow a similar migration pattern:

### Swift Changes
```swift
// Before (Metal 3)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)
encoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)

// After (Metal 4)
let argTable = context.argumentTablePool.acquire()
argTable.setAddress(inputBuffer.gpuAddress, index: 0)
argTable.setAddress(outputBuffer.gpuAddress, index: 1)
argTable.setAddress(paramsBuffer.gpuAddress, index: 2)
encoder.setArgumentTable(argTable, stages: .compute)
```

### Metal Shader Changes
```metal
// Before
#include <metal_stdlib>
using namespace metal;

// After
#include <metal_stdlib>
#include <metal_tensor>  // Only if using tensor ops
using namespace metal;

// Shader body typically unchanged
// threadgroup_barrier remains compatible
```

## Barrier Reference

When fusing kernels, barriers are required between operations that write and read the same buffer:

| Transition | Barrier | Example |
|------------|---------|---------|
| Distance → TopK | Required | `barrier(distances, .dispatch→.dispatch)` |
| Normalize → Distance | Required | `barrier(normalized, .dispatch→.dispatch)` |
| TopK → TopK (streaming) | Required | `barrier(runningTopK, .dispatch→.dispatch)` |
| Quantize → Dequantize | Not fused | Separate operations |

## Testing Each Kernel

After migration, run kernel-specific tests:

```bash
swift test --filter "L2DistanceTests"
swift test --filter "CosineSimilarityTests"
# etc.
```

## Batch Migration

For efficiency, kernels can be migrated in batches:

| Batch | Kernels | Rationale | Status |
|-------|---------|-----------|--------|
| **1** | L2Distance, CosineSimilarity, DotProduct | Core distance - similar patterns | **Complete** |
| **2** | TopKSelection, FusedL2TopK, StreamingTopK | Selection - shared shader file | **Complete** |
| **3** | ScalarQuantization, BinaryQuantization, ProductQuantization | Quantization - similar memory patterns | **Complete** |
| **4** | MatrixMultiply, MatrixVector, MatrixTranspose, BatchMatrix | Matrix ops - all use simdgroup matrix | **Complete** |
| **5** | L2Normalization, ParallelReduction, Elementwise, Statistics | Utility kernels - common patterns | **Complete** |
| **6a** | Minkowski, Hamming, Jaccard, Histogram | Other distance + histogram | **Complete** |

## Task File Status

| Priority | Task File | Status |
|----------|-----------|--------|
| P1 | `task-l2-distance.md` | Not Created |
| P1 | `task-cosine-similarity.md` | Not Created |
| P1 | `task-dot-product.md` | Not Created |
| P2 | `task-topk-selection.md` | Not Created |
| ... | ... | ... |

Use [TEMPLATE_KERNEL_TASK.md](TEMPLATE_KERNEL_TASK.md) to create task files.
