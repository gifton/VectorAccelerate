# 5.3 Profiling with Instruments

> **Finding and fixing GPU bottlenecks with Apple's profiling tools.**

---

## The Concept

Instruments is Apple's profiling suite. The **GPU Trace** instrument shows exactly what your GPU is doing:

```
GPU Trace shows:
┌─────────────────────────────────────────────────────────────────────┐
│  Timeline                                                           │
│  ├── Command Buffer submissions                                     │
│  ├── Kernel execution times                                        │
│  ├── Memory operations                                              │
│  └── GPU utilization                                                │
│                                                                      │
│  Per-Kernel Statistics                                              │
│  ├── Execution time                                                 │
│  ├── Threads dispatched                                             │
│  ├── Memory read/write bytes                                        │
│  └── Occupancy                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why It Matters

Without profiling, you're optimizing blind:

```
Common assumptions (often wrong!):
  "The kernel is slow" → Actually waiting for memory
  "Need more parallelism" → Already GPU-bound
  "Fusion will help" → Already compute-bound

Profiling tells you:
  - Where time is actually spent
  - What the bottleneck is
  - Whether optimization helped
```

---

## The Technique: Using GPU Trace

### Step 1: Enable Profiling

```swift
// 📍 See: Sources/VectorAccelerate/Core/Metal4Context.swift:17-41

let config = Metal4Configuration(
    enableProfiling: true,  // Enable timing
    commandQueueLabel: "VectorAccelerate.Profiled"
)

let context = try await Metal4Context(configuration: config)
```

### Step 2: Capture a Trace

1. Open Xcode → Product → Profile (⌘I)
2. Select "Metal System Trace" or "GPU Trace"
3. Run your workload
4. Stop capture and analyze

### Step 3: Analyze the Timeline

```
Timeline View:
┌─────────────────────────────────────────────────────────────────────┐
│ Time:     0ms        5ms        10ms       15ms       20ms          │
│                                                                      │
│ CPU:      [prepare]──────────[wait]──────────[process]              │
│                                                                      │
│ GPU:                 [l2_distance_768]      [topk_select]           │
│                      ██████████████████      █████████              │
│                                                                      │
│ Memory:              ▓▓▓ read ▓▓▓           ▓ read                  │
│                            ▓▓▓ write                                │
└─────────────────────────────────────────────────────────────────────┘

This shows:
- CPU waiting for GPU (could overlap work)
- Distance kernel is the bottleneck
- Memory write between kernels (could fuse)
```

---

## Interpreting GPU Metrics

### GPU Utilization

```
High Utilization (>90%):
  ✓ GPU is working hard
  ✓ Good parallelism
  ⚠️ May be compute-bound

Low Utilization (<50%):
  ⚠️ GPU not fully used
  Causes:
    - Not enough work dispatched
    - Memory stalls
    - Kernel launch overhead
```

### Memory Bandwidth

```
Close to Peak Bandwidth:
  ⚠️ Memory-bound workload
  Optimizations:
    - Kernel fusion
    - Better memory access patterns
    - Reduce data size (quantization)

Well Below Peak:
  ✓ Not memory-bound
  ✓ Room for more memory ops if needed
```

### Occupancy

```
Occupancy = Active threads / Max possible threads

Low Occupancy (<50%):
  ⚠️ Register pressure or shared memory limiting threads
  Check:
    - Reduce register usage
    - Reduce threadgroup memory
    - Adjust threadgroup size

High Occupancy (>80%):
  ✓ Good thread utilization
```

---

## Common Bottlenecks and Solutions

### Bottleneck: Memory-Bound

**Symptoms:**
- High memory bandwidth utilization
- Low ALU utilization
- GPU utilization fluctuates

**Solutions:**
```swift
// 1. Fuse kernels to eliminate intermediate writes
let fused = try await FusedL2TopKKernel(context: context)

// 2. Use dimension-optimized kernels (better cache usage)
// Automatic via L2DistanceKernel

// 3. Quantize data
let quantized = try await scalarQuantKernel.quantize(vectors)
```

### Bottleneck: Launch Overhead

**Symptoms:**
- Many small GPU operations in trace
- Gaps between operations
- CPU time significant

**Solutions:**
```swift
// 1. Batch operations
let results = try await index.search(
    queries: batchedQueries,  // Not one at a time!
    k: k
)

// 2. Encode multiple operations in one command buffer
try await context.executeAndWait { cb, encoder in
    kernel1.encode(into: encoder, ...)
    encoder.memoryBarrier(scope: .buffers)
    kernel2.encode(into: encoder, ...)
}
```

### Bottleneck: CPU/GPU Serialization

**Symptoms:**
- CPU idle while GPU works
- GPU idle while CPU works
- Trace shows alternating activity

**Solutions:**
```swift
// Double buffering (see 5.2 Async Compute)
// Pipeline command buffers
// Overlap preparation with execution
```

---

## Metal Performance Counters

Enable detailed counters in code:

```swift
// Add to command buffer
commandBuffer.addCompletedHandler { cb in
    if let gpuTime = cb.gpuExecutionTime {
        print("GPU time: \(gpuTime * 1000) ms")
    }

    if let startTime = cb.gpuStartTime,
       let endTime = cb.gpuEndTime {
        print("Total: \((endTime - startTime) * 1000) ms")
    }
}
```

---

## VectorAccelerate Built-in Profiling

```swift
// 📍 See: Sources/VectorAccelerate/Core/Metal4Context.swift:493-510

// Enable profiling
let config = Metal4Configuration(enableProfiling: true)
let context = try await Metal4Context(configuration: config)

// Run operations...

// Get stats
let stats = await context.getPerformanceStats()
print("Total GPU time: \(stats.totalComputeTime) s")
print("Operations: \(stats.operationCount)")
print("Average: \(stats.averageOperationTime * 1000) ms")

// Reset for next measurement
await context.resetPerformanceStats()
```

---

## Profiling Checklist

Before optimizing:

- [ ] Capture trace with realistic workload
- [ ] Identify hotspot kernels
- [ ] Check GPU utilization
- [ ] Check memory bandwidth
- [ ] Check for CPU/GPU serialization

After optimizing:

- [ ] Re-capture trace
- [ ] Verify improvement in target metric
- [ ] Check for new bottlenecks
- [ ] Measure end-to-end latency

---

## Key Takeaways

1. **Profile before optimizing**: Know what's actually slow

2. **GPU Trace is essential**: See exactly what GPU does

3. **Utilization tells the story**: Low = not enough work, high = check memory

4. **Memory-bound is common**: Vector search has low arithmetic intensity

5. **Iterate**: Optimize, measure, repeat

---

## Next Up

Making the CPU vs GPU decision:

**[→ 5.4 CPU vs GPU Decision Framework](./04-CPU-vs-GPU-Decision.md)**

---

*Guide 5.3 of 5.4 • Chapter 5: Pipeline Optimization*
