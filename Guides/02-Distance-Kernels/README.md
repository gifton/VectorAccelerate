# Chapter 2: Distance Kernels

> **Accelerating the hot path—GPU-powered distance computation.**

Distance computation is the heart of similarity search. Every query requires computing distances to potentially millions of vectors. This chapter shows how VectorAccelerate accelerates this critical operation.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [2.1 L2 Distance on GPU](./01-L2-Distance-On-GPU.md) | Euclidean distance kernel | float4 vectorization, FMA, dimension optimization |
| [2.2 Cosine Similarity on GPU](./02-Cosine-On-GPU.md) | Cosine similarity/distance | Normalization, pre-normalized fast path |
| [2.3 Batch Distance Matrix](./03-Batch-Distance-Matrix.md) | Computing Q×N matrices | 2D dispatch, memory layout |
| [2.4 Dimension-Optimized Kernels](./04-Dimension-Optimized-Kernels.md) | Specialized kernels for 384/512/768/1536 | Loop unrolling, ILP |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- How to translate CPU SIMD distance code to GPU kernels
- Why dimension-specific optimizations matter
- The memory access patterns that maximize GPU throughput
- How to choose between generic and optimized kernels
- Real performance comparisons between CPU and GPU

---

## The Hot Path

In vector search, distance computation dominates runtime:

```
Search 10K queries × 1M database × 768D:

┌─────────────────────────────────────────────────────────────────────┐
│                    TIME BREAKDOWN (CPU)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Distance computation:  ████████████████████████████████████  95%   │
│  Top-K selection:       ██                                    3%    │
│  Result processing:     █                                     2%    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Accelerating distance computation is the highest-impact optimization.
```

---

## Distance Metrics in VectorAccelerate

VectorAccelerate supports multiple distance metrics:

| Metric | Formula | Use Case | Kernel |
|--------|---------|----------|--------|
| **L2 (Euclidean)** | √Σ(qᵢ - dᵢ)² | General embeddings | `L2DistanceKernel` |
| **L2 Squared** | Σ(qᵢ - dᵢ)² | Same ranking as L2, faster | `L2DistanceKernel` |
| **Cosine Similarity** | q·d / (‖q‖‖d‖) | Normalized embeddings | `CosineSimilarityKernel` |
| **Cosine Distance** | 1 - cosine | Distance form | `CosineSimilarityKernel` |
| **Dot Product** | Σ qᵢ × dᵢ | Maximum inner product search | `DotProductKernel` |
| **Hamming** | popcount(q ⊕ d) | Binary vectors | `HammingDistanceKernel` |

---

## Kernel Architecture

VectorAccelerate kernels follow a consistent pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KERNEL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Swift Wrapper (L2DistanceKernel.swift)                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Parameter validation                                      │    │
│  │  • Buffer management                                         │    │
│  │  • Pipeline selection (384/512/768/1536/generic)            │    │
│  │  • Thread configuration                                      │    │
│  │  • Encoding & dispatch                                       │    │
│  │  • Result extraction                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                           ▼                                          │
│  Metal Shader (L2Distance.metal)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • l2_distance_kernel        (generic, any dimension)       │    │
│  │  • l2_distance_384_kernel    (MiniLM, Sentence-BERT)       │    │
│  │  • l2_distance_512_kernel    (Small BERT)                  │    │
│  │  • l2_distance_768_kernel    (BERT-base, DistilBERT)       │    │
│  │  • l2_distance_1536_kernel   (OpenAI ada-002)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Summary

Typical speedups on Apple Silicon:

| Operation | CPU (VectorCore) | GPU (VectorAccelerate) | Speedup |
|-----------|-----------------|------------------------|---------|
| 1K × 10K × 768D L2 | 680 ms | 12 ms | 57× |
| 100 × 1M × 768D L2 | 6,800 ms | 85 ms | 80× |
| 1K × 10K × 384D Cosine | 340 ms | 8 ms | 42× |
| 100 × 1M × 1536D L2 | 13,600 ms | 145 ms | 94× |

*Measured on M2 Max, batch queries amortize overhead*

---

## Prerequisites Check

This chapter assumes you understand:

From Chapter 1:
- [ ] GPU thread hierarchy (threads, SIMD groups, threadgroups)
- [ ] Memory coalescing and buffer management
- [ ] When GPU acceleration helps

From VectorCore:
- [ ] How SIMD4 accelerates distance computation
- [ ] The mathematical formulas for L2 and cosine

From VectorIndex:
- [ ] Why distance is the hot path in search

---

## Start Here

**[→ 2.1 L2 Distance on GPU](./01-L2-Distance-On-GPU.md)**

---

*Chapter 2 of 6 • VectorAccelerate Learning Guide*
