# 2.4 Dimension-Optimized Kernels

> **Hand-tuned kernels for common embedding dimensions—squeezing out every last bit of performance.**

---

## The Concept

Different embedding models produce different dimensions:

| Dimension | Models | Use Cases |
|-----------|--------|-----------|
| **384** | MiniLM, all-MiniLM-L6-v2, Sentence-BERT | Lightweight semantic search |
| **512** | Small BERT variants | Mid-range applications |
| **768** | BERT-base, DistilBERT, MPNet, E5 | General-purpose embeddings |
| **1536** | OpenAI ada-002, text-embedding-3-large | High-quality embeddings |

Since these dimensions are fixed at model selection time, we can create **specialized kernels** that are faster than the generic version.

---

## Why It Matters

Generic kernels must handle any dimension:

```metal
// Generic: Runtime loop with bounds checking
for (uint i = 0; i < dimension; i += 4) {
    // Can't fully unroll - dimension unknown
    // Can't optimize register allocation
    // Branch at end for remainder handling
}
```

Specialized kernels know the dimension at compile time:

```metal
// D=768 specialized: Compile-time constants
#pragma unroll 24  // Exactly 192 float4s = 768 floats
for (uint i = 0; i < 192; ++i) {
    // Compiler can fully unroll
    // Perfect register allocation
    // No remainder handling needed
}
```

Typical speedup: **15-30%** over generic kernels.

---

## The Technique: Compile-Time Optimization

### Loop Unrolling

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:202-259
// Note: Code below is restructured for clarity; actual implementation uses condensed format

kernel void l2_distance_768_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // ... bounds check ...

    // Hardcoded pointers - compiler knows the stride
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 768);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 768);

    // Three interleaved accumulators for maximum ILP
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);

    // 768 / 4 = 192 float4s
    // Unroll by 12: process 48 floats per iteration, 16 iterations
    for (uint i = 0; i < 192; i += 12) {
        // Load 12 float4s from query and database
        float4 q0 = query4[i+0];  float4 d0 = database4[i+0];
        float4 q1 = query4[i+1];  float4 d1 = database4[i+1];
        float4 q2 = query4[i+2];  float4 d2 = database4[i+2];
        float4 q3 = query4[i+3];  float4 d3 = database4[i+3];
        float4 q4 = query4[i+4];  float4 d4 = database4[i+4];
        float4 q5 = query4[i+5];  float4 d5 = database4[i+5];
        float4 q6 = query4[i+6];  float4 d6 = database4[i+6];
        float4 q7 = query4[i+7];  float4 d7 = database4[i+7];
        float4 q8 = query4[i+8];  float4 d8 = database4[i+8];
        float4 q9 = query4[i+9];  float4 d9 = database4[i+9];
        float4 q10 = query4[i+10]; float4 d10 = database4[i+10];
        float4 q11 = query4[i+11]; float4 d11 = database4[i+11];

        // Compute differences and accumulate
        float4 diff0 = q0 - d0; acc0 = fma(diff0, diff0, acc0);
        float4 diff1 = q1 - d1; acc1 = fma(diff1, diff1, acc1);
        float4 diff2 = q2 - d2; acc2 = fma(diff2, diff2, acc2);
        float4 diff3 = q3 - d3; acc0 = fma(diff3, diff3, acc0);
        float4 diff4 = q4 - d4; acc1 = fma(diff4, diff4, acc1);
        float4 diff5 = q5 - d5; acc2 = fma(diff5, diff5, acc2);
        float4 diff6 = q6 - d6; acc0 = fma(diff6, diff6, acc0);
        float4 diff7 = q7 - d7; acc1 = fma(diff7, diff7, acc1);
        float4 diff8 = q8 - d8; acc2 = fma(diff8, diff8, acc2);
        float4 diff9 = q9 - d9; acc0 = fma(diff9, diff9, acc0);
        float4 diff10 = q10 - d10; acc1 = fma(diff10, diff10, acc1);
        float4 diff11 = q11 - d11; acc2 = fma(diff11, diff11, acc2);
    }

    // Final reduction
    float4 total = acc0 + acc1 + acc2;
    float sum = total.x + total.y + total.z + total.w;

    float distance = params.computeSqrt ? sqrt(sum) : sum;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}
```

### Why Three Accumulators?

```
FMA latency: ~3-4 cycles on Apple Silicon
FMA throughput: 1 per cycle

With 1 accumulator:
  fma(diff0, diff0, acc)  → 3 cycles latency
  fma(diff1, diff1, acc)  → must wait, +3 cycles
  fma(diff2, diff2, acc)  → must wait, +3 cycles
  ...
  Total: N × 3 cycles (latency bound)

With 3 accumulators (interleaved):
  fma(diff0, diff0, acc0) → starts
  fma(diff1, diff1, acc1) → starts (different accumulator!)
  fma(diff2, diff2, acc2) → starts (different accumulator!)
  fma(diff3, diff3, acc0) → acc0 now ready
  ...
  Total: N × 1 cycle (throughput bound)

3 accumulators → 3× faster!
```

---

## Dimension-Specific Configurations

### D=384 (MiniLM)

```metal
// 384 / 4 = 96 float4s
// Unroll by 8: 12 iterations
// 2 accumulators (384 is smaller, don't need 3)

for (uint i = 0; i < 96; i += 8) {
    // 8 loads per iteration
    acc0 = fma(diff0, diff0, acc0);
    acc1 = fma(diff1, diff1, acc1);
    // ... alternating ...
}
```

### D=512 (Small BERT)

```metal
// 512 / 4 = 128 float4s
// Unroll by 8: 16 iterations
// 2 accumulators

for (uint i = 0; i < 128; i += 8) {
    // Similar to 384
}
```

### D=768 (BERT-base)

```metal
// 768 / 4 = 192 float4s
// Unroll by 12: 16 iterations
// 3 accumulators (larger dimension benefits from more ILP)

for (uint i = 0; i < 192; i += 12) {
    // As shown above
}
```

### D=1536 (OpenAI ada-002)

```metal
// 1536 / 4 = 384 float4s
// Unroll by 16: 24 iterations
// 4 accumulators (largest dimension needs maximum ILP)

float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

for (uint i = 0; i < 384; i += 16) {
    // 16 loads per iteration
    // Cycle through 4 accumulators
    acc0 = fma(diff0, diff0, acc0);
    acc1 = fma(diff1, diff1, acc1);
    acc2 = fma(diff2, diff2, acc2);
    acc3 = fma(diff3, diff3, acc3);
    // ... continue ...
}
```

---

## Register Pressure

More accumulators and unrolling uses more registers:

```
Register usage per configuration:

D=384 (unroll 8, 2 acc):
  - 16 float4 temps (q0-q7, d0-d7)
  - 2 float4 accumulators
  Total: 18 × 4 = 72 registers

D=768 (unroll 12, 3 acc):
  - 24 float4 temps (q0-q11, d0-d11)
  - 3 float4 accumulators
  Total: 27 × 4 = 108 registers

D=1536 (unroll 16, 4 acc):
  - 32 float4 temps
  - 4 float4 accumulators
  Total: 36 × 4 = 144 registers

Apple Silicon has ~256 registers per thread
All configurations fit comfortably
```

If register pressure is too high, threads get "spilled" to slower memory, killing performance.

---

## Automatic Kernel Selection

The Swift wrapper selects the optimal kernel automatically:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:203-218

private func selectPipeline(for dimension: UInt32) -> (
    pipeline: any MTLComputePipelineState,
    name: String
) {
    switch dimension {
    case 384:  return (pipeline384, "l2_distance_384_kernel")
    case 512:  return (pipeline512, "l2_distance_512_kernel")
    case 768:  return (pipeline768, "l2_distance_768_kernel")
    case 1536: return (pipeline1536, "l2_distance_1536_kernel")
    default:   return (genericPipeline, "l2_distance_kernel")
    }
}
```

### Pipeline Pre-Warming

All pipelines are compiled at init time:

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:176-199

public init(context: Metal4Context) async throws {
    // Load ALL kernels upfront
    let library = try await context.shaderCompiler.getDefaultLibrary()

    // Generic
    guard let genericFunc = library.makeFunction(name: "l2_distance_kernel"),
    // Specialized
          let func384 = library.makeFunction(name: "l2_distance_384_kernel"),
          let func512 = library.makeFunction(name: "l2_distance_512_kernel"),
          let func768 = library.makeFunction(name: "l2_distance_768_kernel"),
          let func1536 = library.makeFunction(name: "l2_distance_1536_kernel")
    else {
        throw VectorError.shaderNotFound(name: "L2 distance kernels")
    }

    // Compile all pipelines
    let device = context.device.rawDevice
    self.genericPipeline = try await device.makeComputePipelineState(function: genericFunc)
    self.pipeline384 = try await device.makeComputePipelineState(function: func384)
    // ... etc ...
}
```

---

## Performance Comparison

Measured on M2 Max, 100 queries × 100K database:

| Dimension | Generic Kernel | Specialized | Speedup |
|-----------|---------------|-------------|---------|
| 384 | 5.2 ms | 4.1 ms | 27% |
| 512 | 6.8 ms | 5.5 ms | 24% |
| 768 | 9.4 ms | 7.8 ms | 21% |
| 1536 | 18.1 ms | 14.9 ms | 22% |

The speedup is consistent across dimensions: **20-27%** improvement.

---

## 🔗 VectorCore Connection

VectorCore uses similar optimization strategies on CPU:

```swift
// VectorCore: Dimension-specific SIMD kernels
@_specialize(where D == D384)
@_specialize(where D == D768)
public func l2DistanceSquared<D: StaticDimension>(
    _ a: Vector<D>, _ b: Vector<D>
) -> Float {
    // Swift compiler specializes based on dimension
    // Enables loop unrolling and better register allocation
}
```

The GPU takes this further with hand-written kernels that the Metal compiler can optimize more aggressively.

---

## 🔗 VectorIndex Connection

VectorIndex benefits transparently:

```swift
// VectorIndex: Uses VectorCore's optimized kernels
let index = FlatIndex<D768>(metric: .euclidean)

// VectorAccelerate: GPU-accelerated search with dimension optimization
let acceleratedIndex = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768, capacity: 100_000)
)

// Both automatically use dimension-specific optimizations
```

The dimension is specified at index creation, enabling optimal kernel selection.

---

## Adding New Dimensions

To add optimization for a new dimension (e.g., 1024):

1. **Analyze the dimension**:
   ```
   1024 / 4 = 256 float4s
   Unroll factor: 16 (256 / 16 = 16 iterations)
   Accumulators: 4 (similar to 1536)
   ```

2. **Write the kernel**:
   ```metal
   kernel void l2_distance_1024_kernel(...) {
       float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

       for (uint i = 0; i < 256; i += 16) {
           // Unrolled accumulation
       }
   }
   ```

3. **Update the Swift wrapper**:
   ```swift
   case 1024: return (pipeline1024, "l2_distance_1024_kernel")
   ```

4. **Benchmark** to verify improvement over generic.

---

## Key Takeaways

1. **Compile-time constants enable optimization**: Known dimension allows full unrolling

2. **Multiple accumulators hide latency**: 2-4 accumulators based on dimension

3. **Unroll factor matters**: Too little = missed ILP, too much = register pressure

4. **20-30% speedup is typical**: Worth the effort for common dimensions

5. **Automatic selection**: Users get optimization without manual intervention

---

## Chapter Summary

You've learned how VectorAccelerate accelerates distance computation:

- ✅ L2 distance with vectorization and FMA
- ✅ Cosine similarity with fast path for normalized vectors
- ✅ Batch distance matrices with 2D dispatch
- ✅ Dimension-specific kernel optimization

Next, we'll use these distance kernels to build complete search pipelines.

**[→ Chapter 3: Accelerated Search](../03-Accelerated-Search/README.md)**

---

*Guide 2.4 of 2.4 • Chapter 2: Distance Kernels*
