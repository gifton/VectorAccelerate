# Neural Encode Kernel Optimization Plan

**Status**: Planning
**Target**: `neural_encode_quantize_kernel`
**Goal**: 10-50x throughput improvement via tiled GEMM with weight caching
**Approach**: Multi-phase implementation to manage complexity

---

## Executive Summary

The neural encoder is the bottleneck in the encode→quantize→decode pipeline, running 35x slower than the decoder. The root cause is redundant weight reads: each vector independently reads the entire 192KB weight matrix from device memory. By caching weight tiles in threadgroup memory and processing multiple vectors cooperatively, we can achieve 10-50x speedup.

---

## Problem Analysis

### Current Performance

| Metric | Value |
|--------|-------|
| Throughput | ~60,000 vectors/sec |
| Time per 10K vectors | 16.7 ms |
| Bottleneck | Memory bandwidth (weight reads) |

### Current Implementation

**File**: `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal`
**Kernel**: `neural_encode_quantize_kernel` (lines 167-237)

```metal
// Dispatch: (numVectors, 1, 1) - one thread per vector
kernel void neural_encode_quantize_kernel(..., uint tid) {
    // Each thread processes ONE vector, reading ALL weights
    for (uint j = 0; j < latentDim; ++j) {
        device const float4* weight4 = (device const float4*)(encoderWeights + j * inputDim);

        float4 acc = float4(0.0f);
        for (uint i = 0; i < inputDim/4; ++i) {
            acc = fma(input4[i], weight4[i], acc);  // Weight read per vector!
        }
        latent[j] = reduce(acc);
    }
    // ... quantization ...
}
```

### Why It's Slow

**Weight Matrix Size** (768→128 config):
- Weights: 768 × 128 × 4 bytes = **393 KB**
- Per vector: reads entire 393 KB
- For 10,000 vectors: 10,000 × 393 KB = **3.93 GB** of weight reads

**Memory Bandwidth Calculation**:
```
Weight reads:     3.93 GB
Input reads:      10,000 × 768 × 4 = 30.7 MB
Output writes:    10,000 × 128 × 1 = 1.28 MB
Total:            ~4 GB

At 200 GB/s:      4 GB / 200 GB/s = 20 ms
Actual:           16.7 ms (close to bandwidth limit)
```

**The Problem**: Weights are read N times (once per vector) instead of once.

### Theoretical Optimum

If weights are cached and read only once:
```
Weight reads:     393 KB (once)
Input reads:      30.7 MB
Output writes:    1.28 MB
Total:            ~32 MB

At 200 GB/s:      32 MB / 200 GB/s = 0.16 ms
Speedup:          16.7 ms / 0.16 ms = 104x theoretical max
```

Realistic target with overhead: **10-50x improvement**

---

## Solution: Tiled GEMM with Weight Caching

### Core Insight

Neural encoding is matrix multiplication:
```
Output[N, L] = Input[N, D] × Weights[D, L]^T

Where:
  N = number of vectors (batch size)
  D = input dimension (768)
  L = latent dimension (128)
```

Classic tiled matrix multiplication caches tiles of both matrices in fast threadgroup memory, enabling massive reuse.

### Algorithm Overview

```
For each weight tile [tileL × tileD]:
    1. Cooperatively load weight tile into threadgroup memory
    2. Barrier (ensure tile is fully loaded)
    3. Each thread processes multiple vectors using cached weights
    4. Barrier (ensure all threads done before next tile)
    5. Repeat for next tile

After all tiles:
    6. Apply activation (ReLU)
    7. Compute quantization scale
    8. Quantize to INT8
```

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    Device Memory                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Weights   │  │   Inputs    │  │   Outputs   │     │
│  │  [L × D]    │  │  [N × D]    │  │  [N × L]    │     │
│  │   393 KB    │  │   30.7 MB   │  │   1.28 MB   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                              │
│         ▼                ▼                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Threadgroup Memory (32 KB)            │   │
│  │  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │  Weight Tile  │  │  Partial Accumulators │   │   │
│  │  │ [tileL × tileD]│  │    [V × tileL]       │   │   │
│  │  │    ~8-16 KB   │  │      ~8-16 KB        │   │   │
│  │  └───────────────┘  └───────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
│                        │                               │
│                        ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Thread Registers                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │ Input   │  │  Acc    │  │ Latent  │         │   │
│  │  │ Cache   │  │ float4  │  │ float[] │         │   │
│  │  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Tile Size Selection

**Constraints**:
- Threadgroup memory: 32 KB max
- Weight tile: `tileL × tileD × 4` bytes
- Need space for accumulators too

**Recommended Configuration** (768→128):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `TILE_L` | 32 | Divides 128 evenly, good parallelism |
| `TILE_D` | 64 | Divides 768 evenly (768/64 = 12 tiles) |
| Weight tile size | 32 × 64 × 4 = 8 KB | Fits comfortably |
| Vectors per threadgroup | 32 | Balance parallelism vs memory |
| Accumulator space | 32 × 32 × 4 = 4 KB | Per-vector partial sums |
| Total threadgroup mem | ~12 KB | Well under 32 KB limit |

**Tile iteration**:
- D dimension: 768 / 64 = 12 tiles
- L dimension: 128 / 32 = 4 tiles
- Total: 12 × 4 = 48 tile loads (vs 10,000 full weight reads before)

---

## Implementation Phases

### Phase 1: Infrastructure & Basic Tiled Kernel

**Goal**: Working tiled kernel with correct output, may not be optimized

**Scope**:
1. Add new kernel signature with threadgroup memory declarations
2. Implement cooperative weight tile loading
3. Implement basic accumulation loop
4. Output float latent (no quantization yet)
5. Add Swift dispatch code
6. Add correctness test

**Deliverables**:
- `neural_encode_tiled_kernel` in Metal
- `encodeTiled()` method in Swift
- Test comparing output to reference implementation

**Key Code Structure**:
```metal
kernel void neural_encode_tiled_kernel(
    device const float* inputVectors      [[buffer(0)]],  // [N, D]
    device const float* encoderWeights    [[buffer(1)]],  // [L, D]
    device float* latentVectors           [[buffer(2)]],  // [N, L]
    device const float* encoderBias       [[buffer(3)]],  // [L]
    constant TiledEncodeParams& params    [[buffer(4)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint  tii  [[thread_index_in_threadgroup]]
) {
    // Threadgroup memory for weight tile
    threadgroup float weightTile[TILE_L][TILE_D];

    // Each thread handles subset of vectors in this threadgroup
    // ... implementation ...
}
```

**Validation**: Output matches `neural_encode_kernel` within 1e-5 tolerance

---

### Phase 2: Quantization Integration

**Goal**: Add INT8 quantization to tiled kernel

**Scope**:
1. Add per-vector scale computation after encoding
2. Add INT8 quantization output path
3. Handle ReLU activation
4. Optimize scale computation (parallel reduction)

**Deliverables**:
- `neural_encode_quantize_tiled_kernel`
- Full encode+quantize path with tiling

**Key Addition**:
```metal
// After accumulation complete:
// 1. Apply ReLU (if enabled)
// 2. Compute max absolute value for scale
// 3. Quantize to INT8

// Use threadgroup reduction for scale computation
threadgroup float tgMaxAbs[VECTORS_PER_TG];
// ... parallel reduction ...
```

**Validation**: Output matches `neural_encode_quantize_kernel` exactly

---

### Phase 3: Memory Access Optimization

**Goal**: Optimize memory coalescing and access patterns

**Scope**:
1. Ensure coalesced weight tile loads (adjacent threads load adjacent addresses)
2. Optimize input vector access pattern
3. Add float4 vectorized loads where beneficial
4. Consider weight layout transposition if helpful

**Deliverables**:
- Optimized load functions
- Potential transposed weight buffer
- Measurable throughput improvement

**Key Optimizations**:
```metal
// Coalesced tile loading: thread i loads element i
// For 32×64 tile with 256 threads:
// Each thread loads 8 elements in strided pattern

uint elementsPerThread = (TILE_L * TILE_D) / THREADS_PER_TG;
for (uint i = 0; i < elementsPerThread; ++i) {
    uint flatIdx = tii + i * THREADS_PER_TG;
    uint tileRow = flatIdx / TILE_D;
    uint tileCol = flatIdx % TILE_D;

    // Coalesced read from device memory
    weightTile[tileRow][tileCol] = encoderWeights[...];
}
```

**Validation**: Correctness maintained, throughput improved

---

### Phase 4: Compute Optimization

**Goal**: Maximize ALU utilization and hide latency

**Scope**:
1. Add dual/quad accumulators for FMA latency hiding
2. Unroll inner loops for known dimensions
3. Use `dot()` instruction where applicable
4. Optimize register allocation

**Deliverables**:
- Latency-hidden accumulation
- Dimension-specialized paths (768→128, 768→64, 384→64)

**Key Pattern**:
```metal
// Process 4 latent dimensions with 4 accumulators
float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

for (uint d = 0; d < TILE_D; d += 4) {
    float4 in = input4[d/4];

    // 4 weight rows, each contributes to different latent dim
    acc0 = fma(in, weightTile4[0][d/4], acc0);
    acc1 = fma(in, weightTile4[1][d/4], acc1);
    acc2 = fma(in, weightTile4[2][d/4], acc2);
    acc3 = fma(in, weightTile4[3][d/4], acc3);
}
```

**Validation**: Correctness maintained, compute throughput improved

---

### Phase 5: Dispatch & Occupancy Tuning

**Goal**: Optimize GPU occupancy and kernel dispatch

**Scope**:
1. Tune threadgroup size for target hardware
2. Tune vectors-per-threadgroup for occupancy
3. Add runtime configuration selection
4. Handle edge cases (non-divisible batch sizes)

**Deliverables**:
- Auto-tuned dispatch configuration
- Edge case handling
- Final performance validation

**Configuration Space**:
```
Threadgroup sizes to test: 64, 128, 256, 512
Vectors per threadgroup:   8, 16, 32, 64
Tile sizes:               Various combinations

Select best via micro-benchmarking or heuristics
```

**Validation**: All batch sizes work correctly, optimal config selected

---

### Phase 6: Integration & Cleanup

**Goal**: Production-ready integration

**Scope**:
1. Update `NeuralQuantizationKernel.swift` to use tiled kernel
2. Add fallback to original kernel if tiled unavailable
3. Update documentation
4. Add performance regression tests
5. Clean up experimental code

**Deliverables**:
- Seamless integration with existing API
- No breaking changes
- Performance tests in CI

---

## Detailed Design: Phase 1

### Kernel Parameters

```metal
struct TiledEncodeParams {
    uint numVectors;        // N - total vectors to encode
    uint inputDimension;    // D - input dim (768)
    uint latentDimension;   // L - latent dim (128)
    uint stride;            // Input stride (usually = inputDimension)
    uint vectorsPerTG;      // Vectors processed per threadgroup
    uint8_t useActivation;  // Apply ReLU
    uint8_t padding[3];
};
```

### Dispatch Configuration

```
Threadgroup size:    (1, 256, 1)  // 256 threads per TG
Vectors per TG:      32
Threadgroups:        (ceil(N/32), 1, 1)

Each threadgroup:
  - Processes 32 vectors
  - 256 threads cooperate on tile loads and computation
  - Each thread responsible for 32/256 = 0.125 vectors?
    → Actually: threads split work differently

Better model:
  - 256 threads / 32 vectors = 8 threads per vector
  - Or: each thread computes multiple latent dims for one vector
```

### Thread Work Assignment

**Option A**: Each thread owns subset of vectors
```
Thread 0-7:   vector 0
Thread 8-15:  vector 1
...
Thread 248-255: vector 31

Within vector group: split latent dimensions
8 threads × 16 latent dims = 128 total
```

**Option B**: Each thread owns subset of latent dimensions across all vectors
```
Thread 0:   latent dims 0-3 for all 32 vectors
Thread 1:   latent dims 4-7 for all 32 vectors
...

Requires more complex reduction for quantization
```

**Recommended**: Option A - simpler quantization, good locality

### Pseudocode

```metal
kernel void neural_encode_tiled_kernel(...) {
    // Constants
    constexpr uint TILE_L = 32;
    constexpr uint TILE_D = 64;
    constexpr uint THREADS_PER_TG = 256;
    constexpr uint VECTORS_PER_TG = 32;
    constexpr uint THREADS_PER_VECTOR = THREADS_PER_TG / VECTORS_PER_TG;  // 8

    // Threadgroup memory
    threadgroup float weightTile[TILE_L][TILE_D];  // 8 KB

    // Determine which vector this thread works on
    uint localVectorIdx = tii / THREADS_PER_VECTOR;  // 0-31
    uint globalVectorIdx = tgp.x * VECTORS_PER_TG + localVectorIdx;
    uint laneInVector = tii % THREADS_PER_VECTOR;    // 0-7

    if (globalVectorIdx >= params.numVectors) return;

    // This thread computes latent dims: laneInVector*16 to laneInVector*16+15
    // (128 latent dims / 8 threads = 16 dims per thread)
    uint latentStart = laneInVector * (TILE_L / THREADS_PER_VECTOR);
    uint latentCount = TILE_L / THREADS_PER_VECTOR;  // 4 if TILE_L=32, 8 threads

    // Actually: 32 latent per tile / 8 threads = 4 latent dims per thread per tile

    // Thread-local accumulators for assigned latent dimensions
    float acc[4] = {0, 0, 0, 0};  // For 4 latent dims

    // Get input pointer
    device const float* input = inputVectors + globalVectorIdx * params.stride;

    // Iterate over D dimension in tiles
    for (uint tileD = 0; tileD < params.inputDimension; tileD += TILE_D) {

        // Iterate over L dimension in tiles
        for (uint tileL = 0; tileL < params.latentDimension; tileL += TILE_L) {

            // === Cooperative tile load ===
            // 256 threads load 32×64 = 2048 elements = 8 per thread
            uint elemsPerThread = (TILE_L * TILE_D) / THREADS_PER_TG;
            for (uint e = 0; e < elemsPerThread; ++e) {
                uint flatIdx = tii * elemsPerThread + e;
                uint row = flatIdx / TILE_D;
                uint col = flatIdx % TILE_D;

                uint globalL = tileL + row;
                uint globalD = tileD + col;

                if (globalL < params.latentDimension && globalD < params.inputDimension) {
                    weightTile[row][col] = encoderWeights[globalL * params.inputDimension + globalD];
                } else {
                    weightTile[row][col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // === Compute with cached tile ===
            // Only if this tile covers our assigned latent dimensions
            uint myLatentInTile = laneInVector * (TILE_L / THREADS_PER_VECTOR);

            for (uint localL = 0; localL < latentCount; ++localL) {
                float sum = 0.0f;

                for (uint localD = 0; localD < TILE_D; ++localD) {
                    uint globalD = tileD + localD;
                    if (globalD < params.inputDimension) {
                        sum += input[globalD] * weightTile[myLatentInTile + localL][localD];
                    }
                }

                // Accumulate across D tiles
                acc[localL] += sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write results for this thread's latent dimensions
    device float* output = latentVectors + globalVectorIdx * params.latentDimension;
    for (uint i = 0; i < latentCount; ++i) {
        uint globalL = laneInVector * latentCount + i;
        if (globalL < params.latentDimension) {
            float val = acc[i];
            if (encoderBias) val += encoderBias[globalL];
            if (params.useActivation) val = max(val, 0.0f);
            output[globalL] = val;
        }
    }
}
```

**Note**: The above pseudocode has issues - the tile iteration is wrong. We should iterate D tiles in outer loop, and accumulate into thread-local storage that covers ALL latent dims, not per-tile. Let me correct:

### Corrected Algorithm

```metal
// Each thread computes a FIXED set of latent dimensions for ONE vector
// Accumulates across ALL D-tiles

kernel void neural_encode_tiled_kernel(...) {
    constexpr uint TILE_D = 64;      // D elements per tile
    constexpr uint VECTORS_PER_TG = 32;
    constexpr uint THREADS_PER_VECTOR = 8;
    constexpr uint LATENT_PER_THREAD = 16;  // 128 / 8 = 16

    threadgroup float weightTile[LATENT_PER_THREAD * THREADS_PER_VECTOR][TILE_D];
    // = [128][64] = 32 KB - too big!

    // Need smaller tile...
}
```

Actually, let me reconsider the tiling strategy. The constraint is 32KB threadgroup memory.

### Revised Tiling Strategy

**Problem**: Full [128][64] weight tile = 32KB, leaves no room for anything else.

**Solution**: Smaller L-tile, iterate over L as well

```
TILE_L = 16   (iterate 128/16 = 8 times)
TILE_D = 64   (iterate 768/64 = 12 times)
Weight tile: 16 × 64 × 4 = 4 KB

With 32 vectors, each needing 16 latent accumulators:
Accumulators: 32 × 16 × 4 = 2 KB

Total: 6 KB - plenty of headroom
```

### Final Phase 1 Design

```
Configuration:
  TILE_L = 16
  TILE_D = 64
  VECTORS_PER_TG = 32
  THREADS_PER_TG = 256

Memory:
  weightTile[16][64] = 4 KB
  accumulators[32][16] = 2 KB (in threadgroup memory)
  Total: 6 KB

Iteration:
  Outer loop: L tiles (8 iterations for 128 latent)
  Inner loop: D tiles (12 iterations for 768 input)

  For each (L-tile, D-tile):
    1. Load weight tile cooperatively
    2. Each thread accumulates for its vector
    3. After all D-tiles for this L-tile: store partial to accumulators
    4. After all L-tiles: apply activation, compute scale, quantize
```

---

## Files to Modify

### Phase 1

| File | Changes |
|------|---------|
| `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal` | Add `TiledEncodeParams`, `neural_encode_tiled_kernel` |
| `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift` | Add pipeline loading, `encodeTiled()` method |
| `Tests/VectorAccelerateTests/NeuralQuantizationTests.swift` | Add correctness test |

### Phases 2-6

Additional changes will be specified in each phase's detailed plan.

---

## Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Correctness | Output matches reference within 1e-5 |
| 2 | Correctness | INT8 output matches reference exactly |
| 3 | Throughput | 2x improvement over baseline |
| 4 | Throughput | 5x improvement over baseline |
| 5 | Throughput | 10x+ improvement, all batch sizes work |
| 6 | Integration | No API changes, tests pass, CI green |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Threadgroup memory limits | Conservative tile sizes, validated calculations |
| Register pressure | Monitor via Metal profiler, reduce if needed |
| Occupancy issues | Test multiple configurations, auto-tune |
| Precision loss | Validate against reference at each phase |
| Context exhaustion during impl | Small, focused phases with clear boundaries |

---

## References

- `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal` - Current encoder kernels
- `Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal` - Example of optimized tiled kernel
- `docs/NEURAL_DECODE_OPTIMIZATION_PLAN.md` - Similar optimization effort (decoder)
- Apple Metal Best Practices Guide - Threadgroup memory optimization
- CUTLASS/cuBLAS - Reference tiled GEMM implementations

---

## Appendix: Memory Calculations

### Weight Read Comparison

| Approach | Weight Reads | Bandwidth |
|----------|--------------|-----------|
| Current (per-vector) | N × 393 KB | 3.93 GB for 10K vectors |
| Tiled (shared) | 393 KB total | 0.4 MB for any batch size |
| **Reduction** | **10,000x fewer** | |

### Threadgroup Memory Budget

```
Available:           32 KB
Weight tile:         16 × 64 × 4 = 4,096 bytes
Accumulators:        32 × 16 × 4 = 2,048 bytes
Scale reduction:     32 × 4 = 128 bytes
Misc:                ~512 bytes
─────────────────────────────────
Total used:          ~6.7 KB
Remaining:           ~25 KB (headroom for tuning)
```

### Expected Performance

```
Current:
  Weight bandwidth: 3.93 GB / 16.7 ms = 235 GB/s (near limit)

Tiled:
  Weight bandwidth: 0.4 MB / X ms
  Input bandwidth:  30.7 MB / X ms
  Output bandwidth: 1.28 MB / X ms
  Total: ~32 MB

  At 200 GB/s: 32 MB / 200 GB/s = 0.16 ms
  With overhead (2-3x): 0.3 - 0.5 ms

  Speedup: 16.7 ms / 0.4 ms = 40x
```
