# Neural Decode Kernel Optimization Plan

**Status**: Planning
**Target**: `neural_dequantize_decode_2d_transposed_kernel`
**Issue**: Transposed kernel is 2x slower than non-transposed on Apple Silicon

---

## Problem Analysis

### Current Performance (CI Runner)

| Kernel | Time | Throughput |
|--------|------|------------|
| Non-transposed (128 threads) | 2.47 ms | 404,875 vec/s |
| Transposed (coalesced) | 4.91 ms | 203,488 vec/s |

**The "optimized" transposed kernel is 2x SLOWER.**

### Root Cause

The transposed kernel sacrificed vectorization for memory coalescing:

**Non-transposed (fast):**
```metal
threadgroup float4 tgLatent4[32];  // Packed float4 storage
device const float4* w4 = ...;

for (uint i4 = 0; i4 < 32; ++i4) {
    sum += dot(tgLatent4[i4], w4[i4]);  // 1 instruction per 4 elements
}
// 32 iterations total
```

**Transposed (slow):**
```metal
threadgroup float tgLatent[128];  // Scalar storage

for (uint i = 0; i < 128; ++i) {
    float w = decoderWeightsT[i * inputDim + outIdx];  // Scalar load
    sum += tgLatent[i] * w;  // Scalar FMA
}
// 128 iterations total (4x more)
```

### Why This Matters on Apple Silicon

| Factor | Non-transposed | Transposed |
|--------|----------------|------------|
| Operations per iteration | 1 (float4 dot) | 1 (scalar fma) |
| Elements per iteration | 4 | 1 |
| Total iterations | 32 | 128 |
| Memory loads | 32 × float4 | 128 × float |
| **Effective throughput** | **4x** | **1x** |

Apple Silicon's unified memory provides ~200 GB/s bandwidth regardless of access pattern. The "coalesced" benefit is minimal compared to losing 4x compute throughput from vectorization.

---

## Solution: Vectorized Transposed Kernel

### Key Insight

We can have BOTH coalesced memory access AND float4 vectorization by:
1. Reading **4 consecutive output dimensions** per thread (coalesced)
2. Processing **4 latent dimensions** at a time (vectorized)
3. Using **multiple accumulators** to hide FMA latency

### Memory Layout

**Transposed weights**: `[latentDim, inputDim]`
- Row `i` contains all weights for latent dimension `i`
- Adjacent output indices = adjacent memory addresses (coalesced)

**Access pattern for 4 outputs at once:**
```
Thread processes outputs: outIdx, outIdx+1, outIdx+2, outIdx+3
Weight reads:
  latent[0]: w[0*inputDim + outIdx], w[0*inputDim + outIdx+1], ... (coalesced!)
  latent[1]: w[1*inputDim + outIdx], w[1*inputDim + outIdx+1], ... (coalesced!)
  ...
```

### Algorithm Design

```
For each vector (threadgroup X dimension):
  1. Load scale cooperatively (1 thread)
  2. Dequantize latent codes to threadgroup memory (128 threads, 1 element each)
  3. Barrier

For each thread:
  4. Compute 4 output dimensions (outIdx*4 + 0,1,2,3)
  5. For each group of 4 latent dimensions:
     - Load 4×4 = 16 weights (4 float4 loads, coalesced)
     - Load 4 latent values from threadgroup (broadcast to all 4 outputs)
     - 4 float4 FMA operations (interleaved for latency hiding)
  6. Write 4 output values
```

### Dispatch Configuration

| Parameter | Value |
|-----------|-------|
| Threadgroup size | (1, 32, 1) |
| Outputs per thread | 4 |
| Outputs per threadgroup | 128 |
| Threadgroups (768-dim) | (numVectors, 6, 1) |

**Thread efficiency:**
- 32 threads × 4 outputs = 128 outputs per threadgroup
- 768 outputs / 128 = 6 threadgroups per vector
- Full GPU occupancy maintained

---

## Implementation Plan

### Phase 1: New Kernel Structure

**File:** `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal`

```metal
/// Vectorized transposed decode: processes 4 outputs per thread with float4 operations.
///
/// Key optimizations:
/// 1. Each thread computes 4 adjacent output dimensions
/// 2. Weight loads are coalesced (adjacent threads access adjacent memory)
/// 3. float4 operations for 4x compute throughput
/// 4. Multiple accumulators hide FMA latency
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/128), 1)
///                threadsPerThreadgroup = (1, 32, 1)
kernel void neural_dequantize_decode_2d_transposed_v2_kernel(
    device const char*  latentCodes     [[buffer(0)]],
    device const float* scales          [[buffer(1)]],
    device const float* decoderWeightsT [[buffer(2)]],  // [latentDim, inputDim]
    device float*       outputVectors   [[buffer(3)]],
    device const float* decoderBias     [[buffer(4)]],
    constant NeuralQuantParams& params  [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]]
);
```

### Phase 2: Core Implementation

**Threadgroup memory layout:**
```metal
threadgroup float tgLatent[128];   // Dequantized latent codes
threadgroup float tgScale;          // Per-vector scale
```

**Per-thread computation (4 outputs):**
```metal
const uint outBase = tgp.y * 128 + tptg.y * 4;  // Base output index

// Initialize 4 accumulators (one per output)
float4 acc = float4(0.0f);  // acc.x = out[0], acc.y = out[1], etc.

// Process latent dimensions in groups of 4
for (uint i = 0; i < latentDim; i += 4) {
    // Load 4 latent values (broadcast to all 4 outputs)
    float4 lat = float4(tgLatent[i], tgLatent[i+1], tgLatent[i+2], tgLatent[i+3]);

    // Load 4×4 weights (coalesced reads)
    // For latent[i], read weights for outputs outBase+0,1,2,3
    float4 w0 = *((device const float4*)(decoderWeightsT + (i+0)*inputDim + outBase));
    float4 w1 = *((device const float4*)(decoderWeightsT + (i+1)*inputDim + outBase));
    float4 w2 = *((device const float4*)(decoderWeightsT + (i+2)*inputDim + outBase));
    float4 w3 = *((device const float4*)(decoderWeightsT + (i+3)*inputDim + outBase));

    // Compute: each latent contributes to all 4 outputs
    // acc += lat[0] * w0 + lat[1] * w1 + lat[2] * w2 + lat[3] * w3
    acc = fma(float4(lat.x), w0, acc);
    acc = fma(float4(lat.y), w1, acc);
    acc = fma(float4(lat.z), w2, acc);
    acc = fma(float4(lat.w), w3, acc);
}

// Write 4 outputs
*((device float4*)(outputVectors + vectorIdx * inputDim + outBase)) = acc;
```

### Phase 3: Latency Hiding with Multiple Accumulators

For 128-dim latent (32 groups of 4), use 2 independent accumulator sets:

```metal
float4 acc0 = float4(0.0f);  // Processes latent groups 0,2,4,...
float4 acc1 = float4(0.0f);  // Processes latent groups 1,3,5,...

for (uint i = 0; i < latentDim; i += 8) {
    // Group 1: latent[i..i+3]
    float4 lat0 = float4(tgLatent[i], tgLatent[i+1], tgLatent[i+2], tgLatent[i+3]);
    float4 w00 = *((device const float4*)(decoderWeightsT + (i+0)*inputDim + outBase));
    float4 w01 = *((device const float4*)(decoderWeightsT + (i+1)*inputDim + outBase));
    float4 w02 = *((device const float4*)(decoderWeightsT + (i+2)*inputDim + outBase));
    float4 w03 = *((device const float4*)(decoderWeightsT + (i+3)*inputDim + outBase));

    // Group 2: latent[i+4..i+7]
    float4 lat1 = float4(tgLatent[i+4], tgLatent[i+5], tgLatent[i+6], tgLatent[i+7]);
    float4 w10 = *((device const float4*)(decoderWeightsT + (i+4)*inputDim + outBase));
    float4 w11 = *((device const float4*)(decoderWeightsT + (i+5)*inputDim + outBase));
    float4 w12 = *((device const float4*)(decoderWeightsT + (i+6)*inputDim + outBase));
    float4 w13 = *((device const float4*)(decoderWeightsT + (i+7)*inputDim + outBase));

    // Interleaved FMA (hides 3-4 cycle latency)
    acc0 = fma(float4(lat0.x), w00, acc0);
    acc1 = fma(float4(lat1.x), w10, acc1);
    acc0 = fma(float4(lat0.y), w01, acc0);
    acc1 = fma(float4(lat1.y), w11, acc1);
    acc0 = fma(float4(lat0.z), w02, acc0);
    acc1 = fma(float4(lat1.z), w12, acc1);
    acc0 = fma(float4(lat0.w), w03, acc0);
    acc1 = fma(float4(lat1.w), w13, acc1);
}

float4 acc = acc0 + acc1;  // Final reduction
```

### Phase 4: Dimension-Optimized Variants

Create specialized kernels for common configurations:

| Config | Latent | Output | Unroll | Accumulators |
|--------|--------|--------|--------|--------------|
| 768→128 | 128 | 768 | 8 | 2 |
| 768→64 | 64 | 768 | 8 | 2 |
| 384→64 | 64 | 384 | 8 | 2 |

**Example: 768→128 specialized kernel**
```metal
kernel void neural_dequantize_decode_768_128_transposed_kernel(...) {
    constexpr uint LATENT_DIM = 128;
    constexpr uint OUTPUT_DIM = 768;
    constexpr uint OUTPUTS_PER_THREAD = 4;

    // Fully unrolled for maximum performance
    #pragma unroll
    for (uint i = 0; i < LATENT_DIM; i += 8) {
        // ... unrolled computation ...
    }
}
```

---

## Expected Performance

### Theoretical Analysis

| Metric | Old Transposed | New Vectorized |
|--------|----------------|----------------|
| Iterations (128-dim latent) | 128 | 16 (8 groups × 2 acc) |
| FMA ops per iteration | 1 | 8 |
| Memory loads per iteration | 1 × float | 8 × float4 |
| Latency hiding | None | 2 accumulators |
| **Expected speedup** | 1x | **4-6x** |

### Target Performance

| Kernel | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Non-transposed | 2.47 ms | - | baseline |
| Transposed (old) | 4.91 ms | - | 0.5x (slower) |
| Transposed (new) | - | <2.0 ms | **2.5x faster** |

**Goal:** New transposed kernel should be **faster than non-transposed** due to:
1. Better memory coalescing on weight reads
2. Equivalent vectorization (float4)
3. Better latency hiding (2 accumulators)

---

## Implementation Checklist

### Files to Modify

- [ ] `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal`
  - Add `neural_dequantize_decode_2d_transposed_v2_kernel`
  - Add dimension-specialized variants (768→128, 768→64, 384→64)

- [ ] `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`
  - Load new kernel pipelines
  - Update `encodeDequantizeDecodeTransposed` to use v2 kernel
  - Add auto-selection logic for specialized kernels

- [ ] `Tests/VectorAccelerateTests/NeuralQuantizationTests.swift`
  - Add correctness tests for new kernel
  - Verify numerical equivalence with non-transposed

- [ ] `Tests/VectorAccelerateTests/MLIntegrationBenchmarkTests.swift`
  - Update performance assertions
  - Add comparison benchmarks

### Validation Steps

1. **Correctness**: Output matches non-transposed kernel within 1e-4 tolerance
2. **Performance**: Transposed v2 is faster than non-transposed
3. **CI Stability**: Tests pass on GitHub runner hardware

---

## Appendix: Apple Silicon Optimization Patterns

### Key Patterns Used

1. **float4 vectorization**: 4x compute throughput
2. **Multiple accumulators**: Hide 3-4 cycle FMA latency
3. **Coalesced memory access**: Adjacent threads read adjacent addresses
4. **Threadgroup memory**: Cache dequantized latent for reuse
5. **Compile-time unrolling**: Dimension-specialized kernels

### Memory Bandwidth Considerations

```
768-dim output × 128-dim latent × 4 bytes = 393 KB weights per vector
1000 vectors × 393 KB = 393 MB total weight reads

At 200 GB/s bandwidth: 393 MB / 200 GB/s = 1.96 ms theoretical minimum

Current non-transposed: 2.47 ms (79% efficiency)
Target: <2.0 ms (>98% efficiency with better coalescing)
```

---

## References

- `Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal` - Multiple accumulator pattern
- `Sources/VectorAccelerate/Metal/Shaders/Metal4Common.h` - float4 helpers
- `Guides/02-Distance-Kernels/04-Dimension-Optimized-Kernels.md` - FMA latency hiding
