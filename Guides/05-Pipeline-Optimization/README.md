# Chapter 5: Pipeline Optimization

> **Production-ready performance—profiling, tuning, and decision frameworks.**

You've learned the individual pieces. This chapter shows how to put them together for maximum performance and how to make informed decisions about when to use GPU acceleration.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [5.1 Kernel Fusion](./01-Kernel-Fusion.md) | Combining operations | Reducing memory traffic |
| [5.2 Async Compute](./02-Async-Compute.md) | Overlapping CPU/GPU | Double buffering, pipelining |
| [5.3 Profiling with Instruments](./03-Profiling-With-Instruments.md) | Performance analysis | GPU trace, bottleneck identification |
| [5.4 CPU vs GPU Decision Framework](./04-CPU-vs-GPU-Decision.md) | When to accelerate | Heuristics, adaptive routing |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- How to identify and eliminate pipeline bottlenecks
- When kernel fusion helps (and when it doesn't)
- Techniques for overlapping CPU and GPU work
- How to use Instruments to profile GPU code
- Decision frameworks for CPU vs. GPU routing

---

## The Optimization Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. MEASURE BASELINE                                                │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  Profile with realistic workload                         │     │
│     │  Identify: Total time, GPU vs CPU, memory traffic        │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
│  2. IDENTIFY BOTTLENECK                                             │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  Is it memory-bound or compute-bound?                   │     │
│     │  Is GPU underutilized? Is CPU waiting?                  │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
│  3. APPLY OPTIMIZATION                                              │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  Memory-bound: Reduce traffic (fusion, better layout)   │     │
│     │  Compute-bound: Better algorithms, more parallelism     │     │
│     │  Underutilized: Batch more work, overlap with CPU       │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
│  4. MEASURE AGAIN                                                   │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  Verify improvement                                      │     │
│     │  Repeat until target met                                 │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Common Bottlenecks

| Bottleneck | Symptoms | Solution |
|------------|----------|----------|
| Memory bandwidth | GPU utilization high but slow | Kernel fusion, better access patterns |
| GPU launch overhead | Many small operations | Batch operations, fuse kernels |
| CPU waiting for GPU | CPU idle during GPU work | Async operations, double buffering |
| GPU waiting for CPU | GPU idle between dispatches | Pre-compute on CPU, pipeline |
| Poor GPU utilization | GPU active but slow | Increase parallelism, better workload |

---

## Prerequisites Check

This chapter assumes you understand:

From Chapters 1-4:
- [ ] GPU execution model
- [ ] Distance and selection kernels
- [ ] Memory management patterns

---

## Start Here

**[→ 5.1 Kernel Fusion](./01-Kernel-Fusion.md)**

---

*Chapter 5 of 6 • VectorAccelerate Learning Guide*
