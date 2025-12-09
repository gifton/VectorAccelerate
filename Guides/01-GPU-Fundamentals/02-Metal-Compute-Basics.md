# 1.2 Metal Compute Basics

> **Understanding Metal's execution model—threads, threadgroups, and SIMD groups.**

---

## The Concept

Metal organizes GPU computation into a hierarchy:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DISPATCH GRID                               │
│                                                                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │   Threadgroup   │ │   Threadgroup   │ │   Threadgroup   │  ...   │
│  │     (0,0)       │ │     (1,0)       │ │     (2,0)       │        │
│  │                 │ │                 │ │                 │        │
│  │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │        │
│  │  │SIMD Group │  │ │  │SIMD Group │  │ │  │SIMD Group │  │        │
│  │  │ 32 threads│  │ │  │ 32 threads│  │ │  │ 32 threads│  │        │
│  │  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │        │
│  │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │        │
│  │  │SIMD Group │  │ │  │SIMD Group │  │ │  │SIMD Group │  │        │
│  │  │ 32 threads│  │ │  │ 32 threads│  │ │  │ 32 threads│  │        │
│  │  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │        │
│  │       ...       │ │       ...       │ │       ...       │        │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                      │
│  Grid: All threads for the entire dispatch                           │
│  Threadgroup: Threads that can share memory and synchronize          │
│  SIMD Group: 32 threads executing in lockstep                        │
│  Thread: Single execution instance                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why It Matters

Understanding this hierarchy is essential because:

1. **SIMD groups execute in lockstep**: All 32 threads run the same instruction. Divergent branches cause serialization.

2. **Threadgroups share memory**: Threads in the same threadgroup can communicate via `threadgroup` memory.

3. **Grid size determines parallelism**: More threads = more work done in parallel (up to hardware limits).

4. **Wrong configurations kill performance**: Misaligned thread counts or poor threadgroup sizes waste GPU resources.

---

## The Technique: Metal Kernel Structure

### Basic Kernel Anatomy

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:34-91

kernel void l2_distance_kernel(
    // Buffer arguments
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],

    // Constant arguments (small, uniform across threads)
    constant L2DistanceParams& params [[buffer(3)]],

    // Thread position
    uint3 tid [[thread_position_in_grid]]  // (x, y, z) in the grid
) {
    // Extract position
    uint queryIdx = tid.x;
    uint dbIdx = tid.y;

    // Bounds check (required for non-uniform thread counts)
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // ... compute distance ...
}
```

### Thread Position Attributes

Metal provides several attributes to identify a thread's position:

```metal
// Grid-level position (across ALL threads)
uint3 tid [[thread_position_in_grid]];          // (x, y, z) in grid
uint  linear_id [[thread_position_in_grid]];    // 1D version

// Threadgroup-level position
uint3 tgid [[threadgroup_position_in_grid]];    // Which threadgroup
uint3 local [[thread_position_in_threadgroup]]; // Position in threadgroup
uint  local_id [[thread_index_in_threadgroup]]; // 1D position in threadgroup

// SIMD group position
uint simd_lane [[thread_index_in_simdgroup]];   // 0-31 within SIMD group
uint simd_id [[simdgroup_index_in_threadgroup]];// Which SIMD group
```

### Visualization

```
Grid (100 × 100 threads):
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│    tid.x = 0        tid.x = 1        tid.x = 2                  │
│    ↓                ↓                ↓                          │
│    ┌─────────────────────────────────────────────────┐          │
│  0 │ (0,0) (1,0) (2,0) ... (99,0)                    │ tid.y=0  │
│    │ (0,1) (1,1) (2,1) ... (99,1)                    │ tid.y=1  │
│    │ (0,2) (1,2) (2,2) ... (99,2)                    │ tid.y=2  │
│    │  ...   ...   ...  ...  ...                      │   ...    │
│ 99 │ (0,99)(1,99)(2,99)... (99,99)                   │ tid.y=99 │
│    └─────────────────────────────────────────────────┘          │
│                                                                  │
│    10,000 threads total, each computing one distance             │
└──────────────────────────────────────────────────────────────────┘
```

---

## SIMD Groups: The Real Parallelism Unit

The **SIMD group** (32 threads) is the fundamental unit of execution:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SIMD GROUP (32 threads)                      │
│                                                                      │
│  Lane 0   Lane 1   Lane 2   ...   Lane 30  Lane 31                  │
│    │        │        │              │        │                      │
│    ▼        ▼        ▼              ▼        ▼                      │
│  ┌────┐  ┌────┐  ┌────┐          ┌────┐  ┌────┐                    │
│  │ T0 │  │ T1 │  │ T2 │   ...    │T30 │  │T31 │                    │
│  └────┘  └────┘  └────┘          └────┘  └────┘                    │
│    │        │        │              │        │                      │
│    └────────┴────────┴──────────────┴────────┘                      │
│                        │                                            │
│                        ▼                                            │
│              All execute SAME instruction                           │
│              at the SAME time                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Matters: Branch Divergence

```metal
// ⚠️ BAD: Divergent branch causes serialization
kernel void divergent_example(uint tid [[thread_position_in_grid]]) {
    if (tid % 2 == 0) {
        // Half the threads do this
        expensive_operation_A();
    } else {
        // Other half does this
        expensive_operation_B();
    }
    // Both branches execute serially within SIMD group!
}

// ✅ GOOD: All threads take same path
kernel void uniform_example(uint tid [[thread_position_in_grid]]) {
    // All threads do the same work
    float result = compute_distance(tid);
    output[tid] = result;
}
```

### SIMD Group Communication (Metal 4 / MSL 4.0)

Threads in a SIMD group can communicate efficiently using **SIMD group intrinsics**. These are key Metal 4 features used extensively in VectorAccelerate:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/Metal4Common.h:165-183

#include <metal_simdgroup>  // Required for SIMD intrinsics

// Sum across all 32 threads in SIMD group
float total = simd_sum(partial);

// Broadcast from lane 0 to all lanes
float value = simd_broadcast(myValue, 0);

// Shuffle: get value from another lane
float neighbor = simd_shuffle(myValue, (lane_id + 1) % 32);

// XOR shuffle: exchange with partner at XOR distance
// Used in parallel reduction (stride 1, 2, 4, 8, 16)
float partner = simd_shuffle_xor(myValue, stride);

// Reduction: find minimum/maximum
float minVal = simd_min(myValue);
float maxVal = simd_max(myValue);

// Prefix sum (scan)
float prefix = simd_prefix_exclusive_sum(myValue);
```

### Why SIMD Intrinsics Matter

These intrinsics enable **register-level communication** without going through memory:

```
Traditional approach (through threadgroup memory):
  Thread 0 writes to shared[0]  → Memory write
  Barrier                       → Synchronization cost
  Thread 1 reads from shared[0] → Memory read
  Total: ~20-40 cycles

SIMD shuffle approach:
  simd_shuffle_xor(value, 1)    → Direct register exchange
  Total: ~1-2 cycles

SIMD intrinsics are 10-20× faster for intra-warp communication!
```

This is why the TopK kernels in VectorAccelerate use `simd_shuffle_xor` for parallel merging.

---

## Threadgroups: Shared Memory and Synchronization

Threads in a threadgroup can:
1. Share memory via `threadgroup` qualifier
2. Synchronize via `threadgroup_barrier`

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:204-235

kernel void fused_l2_topk(
    device const float* queries [[buffer(0)]],
    device const float* dataset [[buffer(1)]],
    // ...
    uint q_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    // Shared memory: visible to all threads in threadgroup
    threadgroup float query_cached[MAX_D];
    threadgroup Candidate shared_candidates[MAX_SHARED_CANDIDATES_POT];

    // Cooperatively load query into shared memory
    for (uint d = tid; d < D; d += tgs) {
        query_cached[d] = queries[q_id * D + d];
    }

    // BARRIER: Wait for all threads to finish loading
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now all threads can read the cached query
    // ...
}
```

### Threadgroup Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MEMORY HIERARCHY                               │
│                                                                      │
│  Fastest ──────────────────────────────────────────────── Slowest   │
│                                                                      │
│  ┌────────────┐  ┌────────────────────┐  ┌────────────────────────┐ │
│  │  Registers │  │  Threadgroup Mem   │  │    Device Memory       │ │
│  │            │  │                    │  │                        │ │
│  │ Per-thread │  │  Per-threadgroup   │  │    Global (GPU RAM)    │ │
│  │ ~256 regs  │  │  ~32 KB            │  │    Unified Memory      │ │
│  │            │  │                    │  │                        │ │
│  │ ~1 cycle   │  │  ~10-20 cycles     │  │    ~200-400 cycles     │ │
│  └────────────┘  └────────────────────┘  └────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dispatch Configuration

In Swift, you configure how many threads to launch:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:270-280

// Calculate thread configuration
let config = Metal4ThreadConfiguration.forDistanceKernel(
    numQueries: Int(parameters.numQueries),
    numDatabase: Int(parameters.numDatabase),
    pipeline: pipeline
)

// Dispatch
encoder.dispatchThreadgroups(
    config.threadgroups,                  // How many threadgroups
    threadsPerThreadgroup: config.threadsPerThreadgroup  // Threads per group
)
```

### Two Dispatch Methods

```swift
// Method 1: dispatchThreadgroups
// You specify: threadgroups × threadsPerThreadgroup = total threads
encoder.dispatchThreadgroups(
    MTLSize(width: 100, height: 100, depth: 1),  // 10,000 threadgroups
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)  // 256 each
)
// Total: 2,560,000 threads

// Method 2: dispatchThreads (requires bounds checking in shader)
// You specify: exact thread count, Metal figures out threadgroups
encoder.dispatchThreads(
    MTLSize(width: 100, height: 10000, depth: 1),  // 1M threads
    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
)
```

### Optimal Threadgroup Sizes

```swift
// Query the pipeline for optimal size
let maxThreads = pipeline.maxTotalThreadsPerThreadgroup  // e.g., 1024
let threadExecutionWidth = pipeline.threadExecutionWidth  // e.g., 32 (SIMD width)

// Good threadgroup sizes (multiples of SIMD width):
// 32, 64, 128, 256, 512, 1024

// For 2D distance computation:
// threadsPerThreadgroup: (16, 16, 1) = 256 threads
// or (32, 8, 1) = 256 threads
```

---

## 🔗 VectorCore Connection

VectorCore's SIMD operations map conceptually to GPU execution:

```swift
// VectorCore: 4-wide SIMD on CPU
func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    var sum = SIMD4<Float>.zero
    for i in stride(from: 0, to: a.count, by: 4) {
        sum += SIMD4(a[i...]) * SIMD4(b[i...])
    }
    return sum.sum()  // Horizontal reduction
}
```

```metal
// VectorAccelerate: 32-wide SIMD groups on GPU
kernel void dot_product_parallel(
    device const float* a,
    device const float* b,
    device float* result,
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // Each thread handles one pair element
    float partial = a[tid] * b[tid];

    // SIMD group reduction (32 threads → 1 value)
    float sum = simd_sum(partial);

    // Lane 0 writes result
    if (lane == 0) {
        result[tid / 32] = sum;
    }
}
```

| Concept | VectorCore | VectorAccelerate |
|---------|-----------|------------------|
| SIMD width | 4 (SIMD4) | 32 (SIMD group) |
| Reduction | `.sum()` | `simd_sum()` |
| Memory access | Sequential | Coalesced across threads |
| Parallelism | 8-12 cores | Thousands of threads |

---

## 🔗 VectorIndex Connection

VectorIndex's batch operations map naturally to GPU dispatches:

```swift
// VectorIndex: Batch distance on CPU
func computeDistances(queries: [[Float]], database: [[Float]]) -> [[Float]] {
    queries.map { query in
        database.map { db in
            l2Distance(query, db)
        }
    }
}
```

```swift
// VectorAccelerate: Same logic, GPU execution
func computeDistances(queries: [[Float]], database: [[Float]]) async throws -> [[Float]] {
    // Single GPU dispatch computes ALL Q×N distances
    return try await l2Kernel.compute(queries: queries, database: database)
}
```

The nested loops become a 2D grid dispatch where each thread handles one (query, database) pair.

---

## In VectorAccelerate

The thread configuration logic lives in:

📍 See: `Sources/VectorAccelerate/Core/MetalTypes.swift`

```swift
/// Thread configuration for compute kernels
public struct Metal4ThreadConfiguration: Sendable {
    public let threadgroups: MTLSize
    public let threadsPerThreadgroup: MTLSize

    /// Configuration for distance kernels (Q × N grid)
    public static func forDistanceKernel(
        numQueries: Int,
        numDatabase: Int,
        pipeline: any MTLComputePipelineState
    ) -> Metal4ThreadConfiguration {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup

        // Use 2D dispatch for Q × N work
        let tgWidth = min(16, numQueries)
        let tgHeight = min(maxThreads / tgWidth, numDatabase)

        let numTgX = (numQueries + tgWidth - 1) / tgWidth
        let numTgY = (numDatabase + tgHeight - 1) / tgHeight

        return Metal4ThreadConfiguration(
            threadgroups: MTLSize(width: numTgX, height: numTgY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1)
        )
    }
}
```

---

## Key Takeaways

1. **Threads are the basic unit**: Each thread runs your kernel function once

2. **SIMD groups (32 threads) execute in lockstep**: Avoid branch divergence within a SIMD group

3. **Threadgroups enable cooperation**: Shared memory and barriers let threads work together

4. **Configuration matters**: Wrong threadgroup sizes waste GPU resources

5. **Think in grids**: Map your problem to a 1D, 2D, or 3D grid of independent work items

---

## Next Up

Understanding GPU memory is crucial for performance:

**[→ 1.3 Memory and Transfer Costs](./03-Memory-Transfer-Costs.md)**

---

*Guide 1.2 of 1.3 • Chapter 1: GPU Fundamentals*
