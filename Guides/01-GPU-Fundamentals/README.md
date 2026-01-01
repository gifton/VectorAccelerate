# Chapter 1: GPU Fundamentals

> **Understanding the GPU execution model—the foundation for everything that follows.**

Before we can write GPU-accelerated search, we need to understand *how* GPUs work differently from CPUs. This chapter builds the mental model you'll use throughout the rest of the guide.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [1.1 Why GPU for Search](./01-Why-GPU-For-Search.md) | CPU vs GPU architecture | Parallelism, throughput vs latency |
| [1.2 Metal Compute Basics](./02-Metal-Compute-Basics.md) | Metal programming model | Threads, threadgroups, SIMD groups |
| [1.3 Memory and Transfer Costs](./03-Memory-Transfer-Costs.md) | GPU memory hierarchy | Unified memory, buffer strategies |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- Why GPUs are fundamentally different from CPUs (not just "more cores")
- How Metal organizes computation into threads, threadgroups, and grids
- What SIMD groups are and why they matter for performance
- How Apple Silicon's unified memory changes the transfer cost equation
- When GPU acceleration will help—and when it won't

---

## The CPU → GPU Mindset Shift

If you've been writing CPU code (even optimized VectorCore code), you're used to thinking about:

```
CPU Mindset:
├── Sequential execution with some parallelism
├── Few powerful cores (8-12)
├── Deep cache hierarchies
├── Branch prediction and speculation
└── Minimize work per operation
```

GPU programming requires a different mindset:

```
GPU Mindset:
├── Massively parallel execution
├── Thousands of simple execution units
├── Explicit memory management
├── Avoid divergent branches at all costs
└── Maximize work per memory access
```

The same algorithm that's fast on CPU may be terrible on GPU, and vice versa.

---

## Key Terms

Before diving into the guides, familiarize yourself with these terms:

| Term | Definition |
|------|------------|
| **Thread** | A single execution instance in a kernel |
| **Threadgroup** | A group of threads that can share memory and synchronize |
| **SIMD Group** | 32 threads that execute in lockstep (like a "warp" in CUDA) |
| **Grid** | The total collection of all threads for a dispatch |
| **Kernel** | A GPU function that runs on many threads |
| **Buffer** | GPU-accessible memory (like MTLBuffer) |
| **Pipeline State** | Compiled shader ready for execution |
| **Command Buffer** | Container for GPU commands to execute |
| **Dispatch** | Launching a kernel on the GPU |

---

## Prerequisites Check

This chapter assumes you understand from VectorCore:

- [ ] How SIMD4 operations process 4 floats at once
- [ ] Why memory alignment matters for vectorized access
- [ ] The concept of memory bandwidth vs. compute throughput

And from VectorIndex:

- [ ] The distance computation is the hot path in search
- [ ] Brute force search is O(N×D) per query
- [ ] Why we need algorithms like IVF and HNSW for scale

If any of these are unclear, review the relevant chapters first.

---

## Start Here

**[→ 1.1 Why GPU for Search](./01-Why-GPU-For-Search.md)**

---

*Chapter 1 of 6 • VectorAccelerate Learning Guide*
