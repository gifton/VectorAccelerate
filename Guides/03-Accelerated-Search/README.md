# Chapter 3: Accelerated Search

> **From distances to results—GPU-powered Top-K selection and search strategies.**

Computing distances is only half the story. We also need to find the K nearest neighbors from potentially millions of distances. This chapter shows how VectorAccelerate accelerates the complete search pipeline.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [3.1 Flat Search on GPU](./01-Flat-Search-On-GPU.md) | Brute-force GPU search | When exhaustive search wins |
| [3.2 Top-K Selection on GPU](./02-TopK-On-GPU.md) | Parallel selection algorithms | Heap, bitonic sort, warp selection |
| [3.3 Fused Distance+TopK](./03-Fused-Distance-TopK.md) | Single-pass search | Avoiding the distance matrix |
| [3.4 IVF Acceleration](./04-IVF-Acceleration.md) | Approximate search on GPU | Centroid search, list scanning |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- How to parallelize Top-K selection across thousands of threads
- Why fused kernels are faster than separate distance + selection
- When GPU flat search beats CPU approximate algorithms
- How to accelerate IVF index search

---

## The Search Pipeline

A complete similarity search involves:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SEARCH PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Query Vector(s)                                                     │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Distance Compute │  O(Q × N × D)                                  │
│  │ (Chapter 2)      │  Parallel across all (query, database) pairs   │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │ Top-K Selection │  O(Q × N)                                       │
│  │ (Chapter 3)      │  Find K smallest per query                     │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  Search Results                                                      │
│  [(index, distance), ...]                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Problem with Two Passes

```
Two-pass approach:
  Pass 1: Compute Q × N distances → Q × N × 4 bytes output
  Pass 2: Select top-K from each row

  For Q=100, N=1M:
    Distance buffer: 100 × 1M × 4 = 400 MB
    Must write 400 MB, then read 400 MB
    Memory bandwidth: 800 MB

Fused single-pass:
  Each threadgroup maintains its own top-K heap
  No distance matrix materialized
    Memory bandwidth: Much less!
```

VectorAccelerate provides both approaches:
- **Two-pass**: Maximum flexibility, kernel reuse
- **Fused**: Maximum performance for common cases

---

## Selection Algorithm Overview

| Algorithm | Best For | Complexity | GPU Parallelism |
|-----------|----------|------------|-----------------|
| **Partial Sort** | Small K (<32) | O(N) | Warp-level |
| **Heap Selection** | Medium K (32-512) | O(N log K) | Per-query |
| **Bitonic Sort** | Large K (>512) | O(N log² N) | Fully parallel |
| **Streaming TopK** | Very large N | O(N) amortized | Chunk-parallel |

---

## Performance Summary

| Search Type | Dataset | K | CPU (VectorIndex) | GPU (VectorAccelerate) | Speedup |
|-------------|---------|---|------------------|------------------------|---------|
| Flat | 100K × 768D | 10 | 68 ms | 1.2 ms | 57× |
| Flat | 1M × 768D | 10 | 680 ms | 12 ms | 57× |
| IVF (nprobe=16) | 1M × 768D | 10 | 15 ms | 2.1 ms | 7× |
| Top-K only | 1M distances | 100 | 45 ms | 0.8 ms | 56× |

*Measured on M2 Max with batch of 100 queries*

---

## Prerequisites Check

This chapter assumes you understand:

From Chapter 2:
- [ ] How distance kernels compute Q×N matrices
- [ ] Memory layout of distance outputs
- [ ] Thread dispatch configurations

From VectorIndex:
- [ ] How FlatIndex performs brute-force search
- [ ] IVF index structure (clusters, inverted lists)
- [ ] The recall vs. latency tradeoff

---

## Start Here

**[→ 3.1 Flat Search on GPU](./01-Flat-Search-On-GPU.md)**

---

*Chapter 3 of 6 • VectorAccelerate Learning Guide*
