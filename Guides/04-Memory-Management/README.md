# Chapter 4: Memory Management

> **Efficient GPU memory usage—the key to production performance.**

GPU memory is precious. This chapter teaches you how to manage buffers, handle large indices, and leverage Apple Silicon's unified memory architecture.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [4.1 Buffer Strategies](./01-Buffer-Strategies.md) | Buffer types and pooling | Shared vs. private, buffer reuse |
| [4.2 Unified Memory Patterns](./02-Unified-Memory-Patterns.md) | Apple Silicon advantages | Zero-copy, managed buffers |
| [4.3 Streaming Large Indices](./03-Streaming-Large-Indices.md) | When index > GPU memory | Chunked processing, streaming |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- Different Metal buffer storage modes and when to use each
- How buffer pooling eliminates allocation overhead
- Apple Silicon's unified memory advantages
- Residency management in Metal 4
- Strategies for indices larger than GPU memory

---

## Memory is the Bottleneck

Most GPU vector search workloads are **memory-bound**:

```
L2 distance arithmetic intensity:

Memory reads:  D × 2 × 4 bytes (query + database vector)
Compute:       D × 3 FLOPs (sub, mul, add)

For D=768:
  Memory: 6 KB per distance
  Compute: 2.3 KFLOPs per distance
  Intensity: 0.38 FLOP/byte

Apple M2 Max:
  Memory BW: 400 GB/s
  Peak compute: 13,600 GFLOPS
  Balance point: 34 FLOP/byte

We're 100× below the balance point → memory bound!
```

This means **memory management directly impacts performance**.

---

## GPU Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                     METAL MEMORY HIERARCHY                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Registers (per thread)                                             │
│  ├── Fastest: ~1 cycle                                              │
│  ├── Size: ~256 32-bit registers per thread                        │
│  └── Use: Local variables, accumulators                            │
│                                                                      │
│  Threadgroup Memory (per threadgroup)                               │
│  ├── Fast: ~10-20 cycles                                            │
│  ├── Size: ~32 KB per threadgroup                                   │
│  └── Use: Shared data, reduction scratch space                      │
│                                                                      │
│  Device Memory (unified)                                            │
│  ├── Slow: ~200-400 cycles                                          │
│  ├── Size: System RAM (8-192 GB)                                    │
│  └── Use: Input/output buffers, index data                          │
│                                                                      │
│  Cache Hierarchy                                                    │
│  ├── L1: Per-GPU core, ~16-32 KB                                    │
│  └── L2: Shared, ~4-48 MB                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Budget Planning

For production deployments, plan your memory budget:

```
Example: 1M vectors × 768D embeddings

Vector storage:    1M × 768 × 4 = 3.07 GB
Index structures:  ~200 MB (IVF centroids, metadata)
Query buffers:     ~50 MB (batch of 1000 queries)
Result buffers:    ~40 MB (1000 queries × K=100 × 8 bytes)
Working memory:    ~200 MB (scratch space, intermediate results)
──────────────────────────────────────────────────────────────────
Total:             ~3.6 GB

On 16 GB system: Comfortable
On 8 GB system:  May need streaming strategies
```

---

## Prerequisites Check

This chapter assumes you understand:

From Chapter 1:
- [ ] GPU vs. CPU memory characteristics
- [ ] Unified memory basics

From Chapter 3:
- [ ] Distance kernel memory access patterns
- [ ] Fused kernel memory savings

---

## Start Here

**[→ 4.1 Buffer Strategies](./01-Buffer-Strategies.md)**

---

*Chapter 4 of 6 • VectorAccelerate Learning Guide*
