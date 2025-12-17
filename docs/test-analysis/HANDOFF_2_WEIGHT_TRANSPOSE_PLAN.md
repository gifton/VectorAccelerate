# Weight Transpose Optimization for Neural Decode Kernel

## Context

This plan was created during HANDOFF_2 (Neural Decode Optimization) implementation. The initial 2D dispatch optimization achieved ~40% improvement (9.5k → 13k vec/s end-to-end). Threadgroup size experiments showed 128 threads is optimal (2.57x kernel speedup). This plan implements the weight transpose approach for additional gains.

**Branch:** `gifton/api-qulity-and-correctness`

## Current State

- Decoder weights: `[inputDim, latentDim]` = `[768, 128]` row-major
- Best kernel: 128-thread 2D dispatch (~3M vec/s kernel-only, ~13k vec/s end-to-end)
- Encode/decode ratio: 4.0x (down from 8.8x baseline)

## Goal

- Transpose weights to `[latentDim, inputDim]` = `[128, 768]`
- Adjacent threads read adjacent weight values (coalesced memory access)
- Target: >4M vec/s kernel-only, >20k vec/s end-to-end

## Implementation Phases

### Phase 1: TensorManager Transpose (Low Risk)

**File:** `Sources/VectorAccelerate/Core/TensorManager.swift`

Add CPU transpose using Accelerate framework:

```swift
import Accelerate

extension TensorManager {
    public func createTransposedTensor(
        from source: TensorBuffer,
        name: String
    ) throws -> TensorBuffer {
        guard source.shape.rank == 2 else {
            throw VectorError.invalidDimension(source.shape.rank, reason: "Transpose requires 2D tensor")
        }
        guard source.dataType == .float32 else {
            throw VectorError.invalidOperation("Transpose currently supports float32 only")
        }

        let rows = source.shape.dimensions[0]
        let cols = source.shape.dimensions[1]
        let transposedShape = TensorShape([cols, rows])

        let sourcePtr = source.buffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        var transposed = [Float](repeating: 0, count: rows * cols)

        // Use vDSP for efficient CPU transpose
        vDSP_mtrans(sourcePtr, 1, &transposed, 1, vDSP_Length(cols), vDSP_Length(rows))

        return try createTensor(from: transposed, name: name, shape: transposedShape)
    }
}
```

### Phase 2: Metal Kernel (Medium Risk)

**File:** `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal`

Add new kernel `neural_dequantize_decode_2d_transposed_kernel`:

- Same 2D dispatch pattern (128 threads per threadgroup)
- Same threadgroup-cached latent optimization
- **Key change:** Access transposed weights with coalesced reads

```metal
kernel void neural_dequantize_decode_2d_transposed_kernel(
    device const char*  latentCodes    [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* decoderWeightsT [[buffer(2)]],  // [latentDim, inputDim] TRANSPOSED
    device float*       outputVectors  [[buffer(3)]],
    device const float* decoderBias    [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    // Same setup as neural_dequantize_decode_2d_tg128_kernel...
    // Same threadgroup-cached latent loading...

    // KEY CHANGE: Access transposed weights with coalesced reads
    // For transposed layout [latentDim, inputDim]:
    // Weight for (latentIdx, outputIdx) at: latentIdx * inputDim + outputIdx
    // Adjacent threads (adjacent outIdx) read adjacent memory locations

    float sum = decoderBias ? decoderBias[outIdx] : 0.0f;

    for (uint i = 0; i < latentDim; ++i) {
        float latent = /* from threadgroup cache */;
        float w = decoderWeightsT[i * inputDim + outIdx];  // Coalesced!
        sum += latent * w;
    }

    outputVectors[vectorIdx * inputDim + outIdx] = sum;
}
```

### Phase 3: Swift Integration (Medium Risk)

**File:** `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

1. Add `decoderWeightsTransposed: TensorBuffer?` property
2. Add `optimizedDecodeTransposedPipeline128` pipeline
3. Auto-create transposed weights on load (in weight loading methods)
4. Update `encodeDequantizeDecode` to prefer transposed kernel when available

```swift
// New property
private var decoderWeightsTransposed: TensorBuffer?
private var optimizedDecodeTransposedPipeline128: (any MTLComputePipelineState)?

// In init - load transposed kernel
if let funcTransposed = library.makeFunction(name: "neural_dequantize_decode_2d_transposed_kernel") {
    self.optimizedDecodeTransposedPipeline128 = try await device.makeComputePipelineState(function: funcTransposed)
}

// Helper to create transposed weights
private func createTransposedWeights() async throws {
    guard let decoderWeights = decoderWeights else { return }
    decoderWeightsTransposed = try await tensorManager.createTransposedTensor(
        from: decoderWeights,
        name: "neural_decoder_transposed"
    )
}

// Update encodeDequantizeDecode to prefer transposed kernel
if let transposedWeights = decoderWeightsTransposed,
   let transposedPipeline = optimizedDecodeTransposedPipeline128 {
    encoder.setBuffer(transposedWeights.buffer, offset: 0, index: 2)
    // ... dispatch with transposed kernel
}
```

### Phase 4: Benchmarking (Low Risk)

**File:** `Tests/VectorAccelerateTests/MLIntegrationBenchmarkTests.swift`

Add comparison test `testTransposedDecodePerformance()`:
- Compare transposed vs non-transposed throughput
- Verify numerical correctness (results match within 1e-4 tolerance)

## Critical Files

| File | Changes |
|------|---------|
| `Sources/VectorAccelerate/Core/TensorManager.swift` | Add `createTransposedTensor()` method |
| `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal` | Add transposed decode kernel |
| `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift` | Integrate transposed weights and kernel |
| `Tests/VectorAccelerateTests/MLIntegrationBenchmarkTests.swift` | Add benchmark test |

## Memory Analysis

- Original weights: 768 * 128 * 4 = 393KB
- Transposed copy: 768 * 128 * 4 = 393KB
- **Total overhead: 393KB** (acceptable per handoff requirements)

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Kernel-only throughput | ~3M vec/s | >4M vec/s |
| End-to-end throughput | ~13k vec/s | >20k vec/s |
| Numerical accuracy | - | Match within 1e-4 |

## Backwards Compatibility

- Transposed weights created as additional copy (original unchanged)
- Graceful fallback if transposed kernel unavailable
- Existing public API unchanged

## Reference Documents

- `docs/test-analysis/handoffs/HANDOFF_2_NEURAL_DECODE_OPTIMIZATION.md` - Original handoff with problem analysis
- `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal` - Existing kernels including 2D variants
