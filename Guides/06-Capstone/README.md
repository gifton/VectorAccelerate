# Chapter 6: Capstone

> **Putting it all together—a complete GPU-accelerated vector search system.**

This final chapter brings together everything you've learned into a comprehensive understanding of AcceleratedVectorIndex, VectorAccelerate's flagship high-level API.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [6.1 AcceleratedVectorIndex Deep Dive](./01-AcceleratedVectorIndex-Deep-Dive.md) | The complete implementation | Architecture, data flow, design decisions |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- How all the pieces (kernels, memory, pipelines) fit together
- The design decisions behind AcceleratedVectorIndex
- How to extend VectorAccelerate for your use cases
- Best practices for production deployments

---

## The Journey So Far

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR LEARNING PATH                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Chapter 1: GPU Fundamentals                                        │
│  ├── Why GPU accelerates search                                     │
│  ├── Metal execution model                                          │
│  └── Memory hierarchy                                               │
│                                                                      │
│  Chapter 2: Distance Kernels                                        │
│  ├── L2 and Cosine on GPU                                          │
│  ├── Batch distance matrices                                        │
│  └── Dimension optimization                                         │
│                                                                      │
│  Chapter 3: Accelerated Search                                      │
│  ├── GPU flat search                                                │
│  ├── Parallel Top-K selection                                       │
│  ├── Fused kernels                                                  │
│  └── IVF acceleration                                               │
│                                                                      │
│  Chapter 4: Memory Management                                       │
│  ├── Buffer strategies                                              │
│  ├── Unified memory                                                 │
│  └── Streaming large indices                                        │
│                                                                      │
│  Chapter 5: Pipeline Optimization                                   │
│  ├── Kernel fusion                                                  │
│  ├── Async compute                                                  │
│  ├── Profiling                                                      │
│  └── CPU/GPU decision                                               │
│                                                                      │
│  Chapter 6: Capstone (You Are Here)                                 │
│  └── Complete system implementation                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The VSK Series Complete

With this chapter, you've completed the VectorAccelerate Learning Guide:

```
VSK Educational Series:

Volume 1: VectorCore
  "How do I make ONE distance computation fast?"
  → SIMD, memory layout, numerical stability

Volume 2: VectorIndex
  "How do I search MILLIONS of vectors on CPU?"
  → Flat, IVF, HNSW, PQ, recall/latency tradeoffs

Volume 3: VectorAccelerate ← Complete!
  "How do I use the GPU to go EVEN FASTER?"
  → Metal kernels, GPU memory, pipeline optimization
```

---

## Start Here

**[→ 6.1 AcceleratedVectorIndex Deep Dive](./01-AcceleratedVectorIndex-Deep-Dive.md)**

---

*Chapter 6 of 6 • VectorAccelerate Learning Guide*
