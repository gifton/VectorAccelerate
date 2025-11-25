# VectorAccelerate — Kernel Specification (Core Primitives)

Document Version: 1.0

## Scope and Ownership
VectorAccelerate provides general‑purpose GPU primitives used across projects. It intentionally excludes index/search‑specific algorithms.

Owned here:
- General distance metrics (pairwise and SIMD/batch variants): Euclidean/squared, cosine, dot, Manhattan (L1), Chebyshev (L∞), Minkowski, Hamming, Jaccard.
- Quantization: scalar (int8), product quantization (encode/decode), binary quantization, quantization statistics.
- Basic operations: normalization, vector ops (scale/add/sub), matrix‑vector, small reductions.
- Matrix ops: GEMM, transpose, batch multiply.
- Utilities: statistics (mean/variance), histograms.

Not owned here (see ../VectorIndexAccelerated/KERNEL_SPECIFICATION.md):
- HNSW/IVF/K‑Means index algorithms, search‑shaped tiled Q×N kernels, Top‑K selection stacks, fused distance+Top‑K.

## General Requirements
- Alignment: 16‑byte alignment for vectorized access.
- Threading: Grid‑stride loops; threadgroup size typically 256 (configurable).
- Numerics: Use FMA where applicable; handle NaN/Inf; avoid div‑by‑zero.

## Distance Kernels
- Pairwise distances:
  - `euclideanDistance`, `squaredEuclideanDistance`
  - `cosineDistance`, `dotProduct`
  - `manhattanDistance`, `chebyshevDistance`, `minkowskiDistance`
  - `hammingDistance`, `jaccardDistance`
- SIMD/batch variants:
  - `euclideanDistanceSIMD`, `cosineDistanceSIMD`
  - `batchEuclideanDistanceSIMD`, `batchCosineSimilaritySIMD`, `batchDotProductSIMD`

Recommended approach:
- Patterns: grid‑stride loops for arbitrary D; per‑thread partials reduced in shared memory; use power‑of‑two TG sizes (256) for clean reductions.
- Vectorization: prefer float4 inputs; for SIMD variants, pass `dimension4 = D/4` and handle scalar tail when needed.
- Cosine: compute dot and norms in one pass; clamp denominators with epsilon; return 1‑cosine for distance variants if needed by callers.
- Hamming/Jaccard: map floats to binary presence (or accept bit‑packed inputs in future); use `popcount` where applicable for bit‑packed variants.

Alignment with existing code:
- Mirror `Metal/Shaders/DistanceShaders.metal` and `BasicOperations.metal` structures and naming; keep buffer shapes consistent (row‑major, contiguous dims).

## Quantization Kernels
- Scalar quantization (int8): `scalarQuantize`, `scalarDequantize`, `computeQuantizationStats`
- Product quantization (PQ): `productQuantize`, `productDequantize`
- Binary quantization & Hamming distance: `binaryQuantize`, `binaryHammingDistance`

Recommended approach:
- Scalar: compute per‑vector or global scales; use vDSP for host‑side de/quant when faster; in GPU path, ensure saturation and symmetric ranges; store scale/zero metadata.
- PQ: stage codebook tiles in shared memory; process subspaces sequentially; choose `M` and `K=256` for uint8 codes; ensure coalesced code writes.
- Binary: thresholding + bit‑packing; hamming via XOR + popcount on `uint32`/`uint64` lanes.

Alignment with existing code:
- Mirror `Metal/Shaders/QuantizationShaders.metal`; keep codebook layout `[M × K × D_sub]`; ensure alignment.

## Basic Ops & Matrix Ops
- Core ops: `batchNormalize`, `batchNormalize2D`, `vectorNormalize`, `vectorScale`, `vectorAdd`, `vectorSubtract`, `matrixVectorMultiply`
- Matrix ops:
  - Dedicated shaders for `matrixMultiply`, `matrixTranspose`; optional inline kernels in Swift for small matrices.
  - Batch multiply when applicable.

Recommended approach:
- Matmul: 16×16 or 32×32 tiles depending on D; float4 loads/stores; avoid bank conflicts; unroll inner loops; accumulate in registers.
- Transpose: tile with padding to avoid bank conflicts; handle boundary tiles; coalesced read/write.

Alignment with existing code:
- Keep consistent with `Operations/MatrixEngine.swift` inline kernels where present; prefer moving heavy paths into shaders for reuse.

## Utilities & Statistics
- Statistics: `compute_mean_variance`
- Histograms: `compute_histogram` with local shared histogram + global merge

Recommended approach:
- Stats: single‑pass Kahan‑like compensated sums if required; otherwise FP32 sufficient for common D; atomics for cross‑threadgroup accumulation.
- Histogram: shared `NUM_BINS` atomic array per TG; flush to global with `atomic_add` on non‑zero bins.

## Implementation Guidelines
- Prefer float4 packing for throughput; avoid bank conflicts; keep atomics to a minimum and relaxed when ordering isn’t needed.
- Provide scalar fallback paths for unaligned inputs; validate buffer alignment in wrappers.
- Keep threadgroup memory under 16KB per TG to maintain occupancy.

## Testing and Performance
- CPU parity tests for each metric; edge cases (zeros, NaN/Inf, different dimensions).
- Benchmarks across typical D values; report occupancy and bandwidth; test float4‑aligned vs unaligned cases.

## Interoperability Guidelines
- Shapes and layouts must match VectorIndexAccelerated expectations when reused:
  - Row‑major, contiguous dims; stable tie‑breakers handled by consumers when needed.
  - Avoid introducing alternate memory layouts without explicit adapters.

## Open Questions
- Typical D ranges for core ops? Any need for half‑precision variants to double throughput?
- Do we want a bit‑packed Hamming path exposed broadly (uint32/uint64 inputs) beyond current float‑based variant?
- Should cosine kernels return similarity or distance by default for interoperability with search packages?

## Spec → Implementation Mapping
- Distances: `Sources/VectorAccelerate/Metal/Shaders/DistanceShaders.metal` (✅ implemented)
- Quantization: `Sources/VectorAccelerate/Metal/Shaders/QuantizationShaders.metal` (✅ implemented)
- Basic Ops: `Sources/VectorAccelerate/Metal/Shaders/BasicOperations.metal` (✅ implemented)
- Matrix Ops (shader/inline): `Sources/VectorAccelerate/Metal/Shaders/OptimizedMatrixOps.metal`, `Sources/VectorAccelerate/Operations/MatrixEngine.swift` (✅ implemented)

Note: Search/index‑specific kernels live in VectorIndexAccelerated and should not be duplicated here.

## Implementation Status Summary
- Distances (pairwise + SIMD/batch listed above): ✅ Complete
  - Note: Bit‑packed Hamming (uint32/uint64) fast path: ⏳ Planned (current float‑based and binary variants exist)
- Quantization (scalar, PQ, binary, stats): ✅ Complete
- Basic ops (normalize, vector ops, matrix‑vector): ✅ Complete
- Matrix ops (GEMM, transpose, batch): ✅ Complete (further tiling specializations optional)
- Utilities
  - Statistics (mean/variance): ⏳ Planned
  - Histogram: ⏳ Planned
