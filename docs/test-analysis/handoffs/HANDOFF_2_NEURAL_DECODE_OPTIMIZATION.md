# Handoff 2: Neural Quantization Decode Optimization

## Problem Summary

The neural quantization decode path is 8.8x slower than encode, creating an asymmetric bottleneck. This affects real-time retrieval when vectors must be dequantized for re-ranking or downstream processing.

## Current Performance

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Neural Encode | 83,291 vec/s | Fast - good |
| Neural Decode | 9,491 vec/s | **8.8x slower** |
| Target Decode | >50,000 vec/s | 5x improvement needed |

Test source: `NeuralQuantizationBenchmarkTests.testEncodeDecodeQuality`

## Architecture Overview

Neural quantization uses a learned encoder/decoder:
```
Encode: Input[N, 768] -> Encoder[768, 128] -> Latent[N, 128] -> Quantize -> INT8[N, 128]
Decode: INT8[N, 128] -> Dequantize -> Latent[N, 128] -> Decoder[128, 768] -> Output[N, 768]
```

The encode is a `768 -> 128` projection (compression).
The decode is a `128 -> 768` projection (expansion).

## Root Cause Analysis

### Encode Kernel (FAST)
- Grid dispatch: `(numVectors, 1, 1)` - one thread per vector
- Each thread processes full 768-dim input sequentially
- SIMD-optimized inner loop with float4 operations
- Good memory access pattern (coalesced reads of input)

### Decode Kernel (SLOW)
```metal
kernel void neural_dequantize_decode_kernel(
    device const char* latentCodes [[buffer(0)]],      // [N, 128]
    device const float* scales [[buffer(1)]],          // [N]
    device const float* decoderWeights [[buffer(2)]],  // [768, 128]
    device float* outputVectors [[buffer(3)]],         // [N, 768]
    ...
    uint tid [[thread_position_in_grid]])
{
    // One thread per vector - processes ALL 768 output dimensions
    for (uint j = 0; j < 768; ++j) {
        // Read weight row [128 floats] for each output dimension
        device const float* weightRow = decoderWeights + (j * latentDim);

        // PROBLEM: 768 iterations, each reading 128-float weight row
        // Total: 768 * 128 * 4 = 393KB weight reads PER VECTOR
        for (uint i = 0; i < 128; ++i) {
            sum = fma(latent[i], weightRow[i], sum);
        }
        output[j] = sum;
    }
}
```

**Problem**: Decode reads the full decoder weight matrix (768 * 128 * 4 = 393KB) per vector. With poor cache utilization, this becomes memory-bandwidth bound.

## Relevant Source Files

### Swift Kernel Wrapper
```
Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift
```

### Metal Shader
```
Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal
```

## Optimization Strategies

### Strategy A: Transpose Decoder Weights (Memory Layout)

Current: `decoderWeights[768, 128]` - weight row per output dimension
Proposed: `decoderWeights[128, 768]` - weight column per latent dimension

```metal
// Transposed decode - better memory access pattern
kernel void neural_decode_transposed(
    device const char* latentCodes [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* decoderWeightsT [[buffer(2)]],  // [128, 768] transposed!
    device float* outputVectors [[buffer(3)]],
    constant NeuralQuantParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    // Initialize output accumulator
    float output[768] = {0};

    // For each latent dimension, add its contribution to ALL outputs
    for (uint i = 0; i < 128; ++i) {
        float latentVal = float(codes[i]) * scale;
        device const float* weightCol = decoderWeightsT + (i * 768);

        // Vectorized accumulation
        for (uint j = 0; j < 768; j += 4) {
            float4 w = *((device const float4*)(weightCol + j));
            output[j]   += latentVal * w.x;
            output[j+1] += latentVal * w.y;
            output[j+2] += latentVal * w.z;
            output[j+3] += latentVal * w.w;
        }
    }

    // Write output
    for (uint j = 0; j < 768; ++j) {
        outputVectors[tid * 768 + j] = output[j];
    }
}
```

This reads weight matrix once per vector (128 * 768 * 4 = 393KB) but with sequential access pattern.

### Strategy B: 2D Grid Dispatch

Parallelize across both vectors AND output dimensions:

```metal
// Grid: (numVectors, 768, 1)
kernel void neural_decode_2d(
    device const char* latentCodes [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* decoderWeights [[buffer(2)]],  // [768, 128]
    device float* outputVectors [[buffer(3)]],
    constant NeuralQuantParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint vectorIdx = tid.x;
    uint outputIdx = tid.y;

    device const char* codes = latentCodes + (vectorIdx * 128);
    float scale = scales[vectorIdx];
    device const float* weightRow = decoderWeights + (outputIdx * 128);

    // Each thread computes ONE output element
    float sum = 0.0f;
    for (uint i = 0; i < 128; i += 4) {
        float4 l = float4(
            float(codes[i]) * scale,
            float(codes[i+1]) * scale,
            float(codes[i+2]) * scale,
            float(codes[i+3]) * scale
        );
        float4 w = *((device const float4*)(weightRow + i));
        sum += dot(l, w);
    }

    outputVectors[vectorIdx * 768 + outputIdx] = sum;
}
```

Pros: Maximum parallelism, good for large batch sizes
Cons: Redundant scale reads, may overwhelm thread scheduler for small batches

### Strategy C: Tiled Shared Memory Approach

Use threadgroup shared memory to cache weight tiles:

```metal
constant uint TILE_VECTORS = 16;
constant uint TILE_OUTPUTS = 64;
constant uint TILE_LATENT = 32;

kernel void neural_decode_tiled(
    device const char* latentCodes [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* decoderWeights [[buffer(2)]],
    device float* outputVectors [[buffer(3)]],
    constant NeuralQuantParams& params [[buffer(4)]],
    threadgroup float* sharedWeights [[threadgroup(0)]],  // [TILE_OUTPUTS, TILE_LATENT]
    threadgroup float* sharedLatent [[threadgroup(1)]],   // [TILE_VECTORS, TILE_LATENT]
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    // Cooperative loading of weight tile and latent tile
    // Process tile of outputs for tile of vectors
    // ...
}
```

This minimizes global memory bandwidth but adds complexity.

## Swift Integration Points

### NeuralQuantizationKernel.swift Changes

```swift
public struct NeuralQuantizationKernel: Metal4Kernel {
    // Add transposed weight support
    private var decoderWeightsTransposed: MTLBuffer?

    // Add kernel selection based on batch size
    public func decode(
        latentCodes: MTLBuffer,
        scales: MTLBuffer,
        outputVectors: MTLBuffer,
        params: NeuralQuantParams
    ) async throws {
        // Select optimal kernel:
        // - Small batch (<32): Use 2D grid
        // - Medium batch: Use transposed weights
        // - Large batch (>1K): Use tiled approach
    }

    // Precompute transposed weights during initialization
    public func setDecoderWeights(_ weights: MTLBuffer, inputDim: Int, latentDim: Int) {
        // Transpose on CPU or GPU
        self.decoderWeightsTransposed = transpose(weights, rows: inputDim, cols: latentDim)
    }
}
```

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Decode throughput | 9,491 vec/s | >50,000 vec/s |
| Encode/Decode ratio | 8.8x | <2x |
| Memory overhead | 1x weights | 2x (transposed copy acceptable) |

## Testing Commands

```bash
# Run neural quantization benchmarks
swift test --filter "NeuralQuantization"
swift test --filter "testEncodeDecodeQuality"

# Run ML integration benchmarks
swift test --filter "MLIntegrationBenchmark"
```

## Key Constraints

1. **Weight Matrix Size**: Decoder is 768 * 128 * 4 = 393KB (fits in L2 cache on most Apple Silicon)
2. **Latent Dimension**: Fixed at 128 for INT8 quantization
3. **Output Dimension**: Typically 768 (BERT/sentence-transformers) or 384 (MiniLM)
4. **Backwards Compatibility**: Must support existing weight format; transposition can be done at load time

## Deliverables

1. Optimized `NeuralQuantization.metal` with new decode kernel
2. Updated `NeuralQuantizationKernel.swift` with kernel selection logic
3. Optional: Weight transposition utility in Swift
4. Benchmark showing 5x+ improvement in decode throughput
