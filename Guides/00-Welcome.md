# VectorAccelerate Learning Guide

> **Unleash the GPU—taking vector search from fast to massively parallel.**

Welcome to the VectorAccelerate Learning Guide. This is the **third volume** in the VSK (Vector Search Kit) educational series, designed to teach you how to leverage Apple Silicon GPUs for dramatic performance improvements in similarity search.

---

## Prerequisites: VectorCore + VectorIndex Foundations

**This guide assumes you have completed both previous volumes:**
- [VectorCore Learning Guide](../../VectorCore/Guides/00-Welcome.md) (Volume 1)
- [VectorIndex Learning Guide](../../VectorIndex/Guides/00-Welcome.md) (Volume 2)

The concepts build directly on each other:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VectorCore (Volume 1)                            │
│                                                                         │
│   Memory → SIMD → Numerical → Unsafe → Performance → Capstone           │
│                                                                         │
│   "How do I make ONE distance computation fast?"                        │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       VectorIndex (Volume 2)                            │
│                                                                         │
│   Similarity → Flat → IVF → HNSW → PQ → Tuning → Capstone               │
│                                                                         │
│   "How do I search MILLIONS of vectors on CPU?"                         │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     VectorAccelerate (Volume 3)                         │
│                                                                         │
│   GPU Basics → Distance Kernels → Search → Memory → Pipelines → Capstone│
│                                                                         │
│   "How do I use the GPU to go EVEN FASTER?"                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Metal 4 & MSL 4.0

VectorAccelerate is built on **Metal 4**, Apple's latest GPU programming framework. This brings several key improvements:

### Platform Requirements

```
Target Platforms:
  macOS 26.0+       (Tahoe)
  iOS 26.0+
  visionOS 3.0+

MSL Version: 4.0 (Metal Shading Language)
```

### Metal 4 Features Used in VectorAccelerate

| Feature | How It's Used | Benefit |
|---------|--------------|---------|
| **SIMD Group Intrinsics** | `simd_shuffle_xor`, `simd_sum`, `simd_min` | Efficient warp-level communication for TopK selection |
| **Unified Memory** | `.storageModeShared` buffers | Zero-copy CPU↔GPU on Apple Silicon |
| **ResidencySet** | Explicit memory residency | Predictable performance, controlled eviction |
| **Argument Tables** | Efficient buffer binding | Reduced CPU overhead per dispatch |
| **MSL 4.0 Compiler** | Improved optimization | Better code generation for complex kernels |

### Version Detection in Shaders

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/Metal4Common.h:38-50

#if __METAL_VERSION__ >= 400
    #define VA_MSL_4_0 1
    #define VA_METAL_4_AVAILABLE 1
#else
    #define VA_MSL_4_0 0
    #define VA_METAL_4_AVAILABLE 0
#endif
```

VectorAccelerate requires Metal 4 and has no backwards compatibility with Metal 3 or earlier. Ensure your target platforms meet the requirements above.

---

### From VectorCore, You'll Use:

| VectorCore Concept | How It Applies to GPU |
|-------------------|----------------------|
| SIMD operations | GPU SIMD lanes work similarly—but with 32+ lanes instead of 4 |
| Memory layout & alignment | GPU buffers need proper alignment for coalesced access |
| Numerical stability | Parallel reductions require careful accumulation order |
| Unsafe pointers | Metal buffers use raw memory, similar patterns apply |

### From VectorIndex, You'll Use:

| VectorIndex Concept | How It's Accelerated |
|--------------------|---------------------|
| Distance metrics (L2, Cosine) | GPU distance kernels compute millions of pairs in parallel |
| Flat index brute force | GPU handles exhaustive search efficiently for <100K vectors |
| IVF clustering | GPU-accelerated K-Means and centroid assignment |
| Top-K selection | Parallel selection algorithms on GPU |
| Recall vs. latency tradeoffs | GPU adds a new dimension: transfer cost vs. compute benefit |

---

## What You'll Learn

This guide teaches **GPU programming for vector search**—knowledge that applies to any compute-intensive workload on Apple Silicon:

| Chapter | You'll Learn | Why It Matters |
|---------|-------------|----------------|
| [1. GPU Fundamentals](./01-GPU-Fundamentals/README.md) | Metal compute basics, threadgroups, memory | Understanding the GPU execution model |
| [2. Distance Kernels](./02-Distance-Kernels/README.md) | GPU distance computation, SIMD lanes, reduction | The hot path of similarity search |
| [3. Accelerated Search](./03-Accelerated-Search/README.md) | Top-K on GPU, fused kernels, hybrid strategies | Complete search acceleration |
| [4. Memory Management](./04-Memory-Management/README.md) | Buffers, residency, unified memory | Efficient GPU resource usage |
| [5. Pipeline Optimization](./05-Pipeline-Optimization/README.md) | Kernel fusion, async compute, profiling | Production-ready performance |
| [6. Capstone](./06-Capstone/README.md) | AcceleratedVectorIndex deep dive | Building a complete GPU-first index |

---

## The Core Problem

VectorIndex taught you algorithms that search millions of vectors on CPU. But what if even *those* aren't fast enough?

```
VectorCore (CPU SIMD):
  Single dot product:           ~100 ns

VectorIndex (CPU):
  1M × 768D brute force:        ~850 ms
  HNSW search (ef=64):          ~2 ms
  IVF search (nprobe=16):       ~15 ms

VectorAccelerate (GPU):
  1M × 768D brute force:        ~12 ms   (70× faster!)
  10K candidate reranking:      ~0.3 ms  (7× faster!)
```

The GPU isn't magic—it's **massive parallelism**. Where your CPU has 8-12 cores, Apple Silicon GPUs have thousands of execution units running in lockstep.

---

## When GPU Acceleration Helps

GPU acceleration isn't always the answer. Understanding *when* to use it is crucial:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    GPU ACCELERATION DECISION TREE                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Is your workload compute-bound?                                       │
│       │                                                                │
│       ├── NO → CPU is probably fine                                    │
│       │                                                                │
│       └── YES → Continue ↓                                             │
│                    │                                                   │
│  Is the data already on GPU or easily batched?                         │
│       │                                                                │
│       ├── NO → Transfer costs may dominate                             │
│       │         Consider: Can you batch queries?                       │
│       │                                                                │
│       └── YES → Continue ↓                                             │
│                    │                                                   │
│  Is there enough parallelism?                                          │
│       │                                                                │
│       ├── < 1K operations → GPU overhead too high                      │
│       │                                                                │
│       └── > 10K operations → GPU likely wins                           │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### When to Use GPU

✅ **Batch queries**: Amortize transfer overhead across many queries
✅ **Large candidate sets**: >10K vectors to compare
✅ **High dimensions**: 512D+ embeddings where FLOPS dominate
✅ **Data already on GPU**: Index stored in GPU memory
✅ **Throughput over latency**: Maximize queries/second

### When to Stay on CPU

❌ **Single low-latency queries**: GPU launch overhead too high (~50-100μs)
❌ **Small indices**: <10K vectors, CPU is fast enough
❌ **Memory-constrained devices**: GPU memory is limited
❌ **Complex branching logic**: GPUs dislike divergent control flow
❌ **HNSW graph traversal**: Inherently sequential, hard to parallelize

---

## VectorAccelerate Architecture

VectorAccelerate provides two layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL API                                        │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              AcceleratedVectorIndex                              │   │
│   │                                                                  │   │
│   │   • GPU-first vector storage                                     │   │
│   │   • Flat and IVF index types                                     │   │
│   │   • Automatic CPU/GPU routing                                    │   │
│   │   • Handle-based vector identification                           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    LOW-LEVEL API                                        │
│                                                                         │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│   │ Distance     │ │ Selection    │ │ Quantization │ │ Matrix       │   │
│   │ Kernels      │ │ Kernels      │ │ Kernels      │ │ Kernels      │   │
│   │              │ │              │ │              │ │              │   │
│   │ • L2         │ │ • TopK       │ │ • Scalar     │ │ • Multiply   │   │
│   │ • Cosine     │ │ • FusedL2TopK│ │ • Binary     │ │ • Transpose  │   │
│   │ • DotProduct │ │ • WarpSelect │ │ • Product    │ │ • MatVec     │   │
│   │ • Hamming    │ │              │ │ • Neural     │ │ • Batch      │   │
│   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              Metal4Context                                       │   │
│   │                                                                  │   │
│   │   • Device & command queue management                            │   │
│   │   • Buffer pools & residency                                     │   │
│   │   • Pipeline caching                                             │   │
│   │   • Shader compilation                                           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How to Use This Guide

### The Sequential Path

Each chapter builds on the previous. If you're new to GPU programming:

```
Chapter 1 ──→ Chapter 2 ──→ Chapter 3 ──→ Chapter 4 ──→ Chapter 5 ──→ Chapter 6
GPU Basics   Distance     Search       Memory       Pipelines    Capstone
             Kernels      Acceleration Management   & Profiling
```

### The Reference Path

If you're already familiar with Metal or GPU programming:

- **"I want to understand VectorAccelerate's kernels"** → [Chapter 2](./02-Distance-Kernels/README.md)
- **"How do I accelerate an existing index?"** → [Chapter 3](./03-Accelerated-Search/README.md)
- **"I'm hitting memory issues"** → [Chapter 4](./04-Memory-Management/README.md)
- **"I want to see the full implementation"** → [Chapter 6](./06-Capstone/README.md)

### Each Guide Follows This Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  THE CONCEPT                                                    │
│  What's the GPU technique? Plain English, diagrams.             │
├─────────────────────────────────────────────────────────────────┤
│  WHY IT MATTERS                                                 │
│  What performance problem does this solve?                      │
├─────────────────────────────────────────────────────────────────┤
│  THE TECHNIQUE                                                  │
│  Step-by-step with Metal shader code and Swift wrappers.        │
├─────────────────────────────────────────────────────────────────┤
│  IN VECTORACCELERATE                                            │
│  Where is this implemented? Links to actual source.             │
├─────────────────────────────────────────────────────────────────┤
│  🔗 VECTORCORE CONNECTION                                       │
│  How does this relate to CPU SIMD from Volume 1?                │
├─────────────────────────────────────────────────────────────────┤
│  🔗 VECTORINDEX CONNECTION                                      │
│  How does this accelerate algorithms from Volume 2?             │
├─────────────────────────────────────────────────────────────────┤
│  KEY TAKEAWAYS                                                  │
│  What should stick? The transferable lessons.                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## VectorAccelerate Source Locations

Throughout this guide, we reference actual implementation code:

| Topic | File Path |
|-------|-----------|
| L2 Distance Kernel | `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift` |
| L2 Distance Shader | `Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal` |
| Cosine Similarity Kernel | `Sources/VectorAccelerate/Kernels/Metal4/CosineSimilarityKernel.swift` |
| Top-K Selection | `Sources/VectorAccelerate/Kernels/Metal4/TopKSelectionKernel.swift` |
| Fused L2+TopK | `Sources/VectorAccelerate/Kernels/Metal4/FusedL2TopKKernel.swift` |
| Streaming TopK | `Sources/VectorAccelerate/Kernels/Metal4/StreamingTopKKernel.swift` |
| Metal4 Context | `Sources/VectorAccelerate/Core/Metal4Context.swift` |
| Residency Manager | `Sources/VectorAccelerate/Core/ResidencyManager.swift` |
| Buffer Pool | `Sources/VectorAccelerate/Core/BufferPool.swift` |
| Accelerated Index | `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift` |
| IVF Pipeline | `Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift` |

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| `Q` | Number of query vectors |
| `N` | Number of database vectors |
| `D` | Vector dimension |
| `K` | Number of nearest neighbors |
| `📍 See:` | Link to VectorAccelerate source code |
| `🔗 VectorCore:` | Connection to VectorCore concept |
| `🔗 VectorIndex:` | Connection to VectorIndex concept |
| `⚠️` | Common mistake or pitfall |
| `💡` | Key insight or tip |
| `🎯` | Performance optimization |

---

## Hardware Requirements

VectorAccelerate requires **Metal 4** features available on:

- **macOS 26.0+** on Apple Silicon (M1 or later)
- **iOS 26.0+** on A14 Bionic or later
- **visionOS 3.0+**

Key capabilities used:

| Feature | Why We Need It |
|---------|---------------|
| Unified Memory | Zero-copy CPU↔GPU data sharing |
| ArgumentTable | Efficient buffer binding |
| MTLSharedEvent | CPU/GPU synchronization |
| SIMD Group Operations | Warp-level reductions |
| Compute Pipelines | Shader execution |

---

## Known Limitations

VectorAccelerate is under active development. Current limitations:

| Limitation | Details |
|------------|---------|
| **GPU Distance Metrics** | Only `.euclidean` is fully GPU-accelerated. Other metrics fall back to CPU. |
| **Fused Top-K** | `FusedL2TopKKernel` works best with K ≤ 8 (private heap size). Larger K uses fallback strategies. |
| **IVF Index** | IVF support is functional but work-in-progress. Auto-routing to flat search helps with small datasets. |
| **Quantization** | Scalar quantization (SQ8, SQ4) is available for IVF indexes only. |

---

## Let's Begin

Ready to unleash the GPU?

**[→ Chapter 1: GPU Fundamentals](./01-GPU-Fundamentals/README.md)**

---

*VectorAccelerate Learning Guide • Volume 3 of the VSK Educational Series • Dec 2024*
