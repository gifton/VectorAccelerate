// VectorAccelerate: Neural Quantization Kernels
//
// GPU kernels for learned encoder/decoder quantization.
// Provides higher quality compression than scalar quantization by learning
// data-specific latent representations optimized for quantization.
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Phase 4: ML Integration - Neural Quantization
//
// Architecture:
// - Encoder: Projects input vectors to lower-dimensional latent space
// - Quantization: INT8 quantization of latent vectors
// - Decoder: Reconstructs original vectors from latent codes

#include "Metal4Common.h"

// MARK: - Parameter Structures

/// Parameters for neural quantization kernels
struct NeuralQuantParams {
    uint32_t numVectors;       // Number of vectors to process
    uint32_t inputDimension;   // Input vector dimension (D)
    uint32_t latentDimension;  // Latent space dimension (L)
    uint32_t stride;           // Stride between input vectors
    uint8_t  useActivation;    // 1 = apply ReLU after encoding
    uint8_t  normalizeLatent;  // 1 = L2 normalize latent vectors
    uint8_t  padding[2];       // Alignment padding
};

// MARK: - Helper Functions

/// Project vector through weight matrix: output[j] = sum_i(input[i] * weights[j * inDim + i])
inline void projectToLatent(
    device const float* input,
    device const float* weights,
    thread float* output,
    uint inputDim,
    uint outputDim
) {
    for (uint j = 0; j < outputDim; ++j) {
        device const float* weightRow = weights + (j * inputDim);

        // SIMD-optimized dot product
        const uint simd_blocks = inputDim / 4;
        device const float4* input4 = (device const float4*)input;
        device const float4* weight4 = (device const float4*)weightRow;

        float4 acc = float4(0.0f);
        for (uint i = 0; i < simd_blocks; ++i) {
            acc = fma(input4[i], weight4[i], acc);
        }
        float sum = acc.x + acc.y + acc.z + acc.w;

        // Handle remainder
        for (uint i = simd_blocks * 4; i < inputDim; ++i) {
            sum = fma(input[i], weightRow[i], sum);
        }

        output[j] = sum;
    }
}

/// Apply ReLU activation in-place
inline void applyReLU(thread float* vec, uint dim) {
    for (uint i = 0; i < dim; ++i) {
        vec[i] = max(vec[i], 0.0f);
    }
}

/// Compute L2 norm
inline float computeNorm(thread float* vec, uint dim) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        sum = fma(vec[i], vec[i], sum);
    }
    return sqrt(sum);
}

/// Normalize vector in-place
inline void normalizeVector(thread float* vec, uint dim) {
    float norm = computeNorm(vec, dim);
    float invNorm = (norm > VA_EPSILON) ? (1.0f / norm) : 0.0f;
    for (uint i = 0; i < dim; ++i) {
        vec[i] *= invNorm;
    }
}

/// Compute scale factor for symmetric INT8 quantization
inline float computeScale(thread float* vec, uint dim) {
    float maxAbs = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        maxAbs = max(maxAbs, abs(vec[i]));
    }
    return max(maxAbs / 127.0f, VA_EPSILON);
}

// MARK: - Encode Kernel (float -> float latent)

/// Encode input vectors to latent space (float output).
///
/// Useful for inspecting latent representations before quantization.
///
/// Grid dispatch: (numVectors, latentDimension, 1)
kernel void neural_encode_kernel(
    device const float* inputVectors [[buffer(0)]],     // [N, D]
    device const float* encoderWeights [[buffer(1)]],   // [L, D]
    device float* latentVectors [[buffer(2)]],          // [N, L]
    device const float* encoderBias [[buffer(3)]],      // [L] or null
    constant NeuralQuantParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vectorIdx = tid.x;
    const uint latentIdx = tid.y;

    if (vectorIdx >= params.numVectors || latentIdx >= params.latentDimension) {
        return;
    }

    // Get input vector
    device const float* input = inputVectors + (vectorIdx * params.stride);
    device const float* weightRow = encoderWeights + (latentIdx * params.inputDimension);

    // Compute dot product for this output dimension
    const uint inputDim = params.inputDimension;
    const uint simd_blocks = inputDim / 4;

    device const float4* input4 = (device const float4*)input;
    device const float4* weight4 = (device const float4*)weightRow;

    float4 acc = float4(0.0f);
    for (uint i = 0; i < simd_blocks; ++i) {
        acc = fma(input4[i], weight4[i], acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    for (uint i = simd_blocks * 4; i < inputDim; ++i) {
        sum = fma(input[i], weightRow[i], sum);
    }

    // Add bias if present
    if (encoderBias) {
        sum += encoderBias[latentIdx];
    }

    // Apply ReLU activation
    if (params.useActivation) {
        sum = max(sum, 0.0f);
    }

    // Store result
    latentVectors[vectorIdx * params.latentDimension + latentIdx] = sum;
}

// MARK: - Encode + Quantize Kernel (float -> INT8)

/// Encode and quantize to INT8 in one pass.
///
/// For each vector:
/// 1. Project through encoder weights
/// 2. Apply optional ReLU activation
/// 3. Compute per-vector scale factor
/// 4. Quantize to INT8
///
/// Grid dispatch: (numVectors, 1, 1)
kernel void neural_encode_quantize_kernel(
    device const float* inputVectors [[buffer(0)]],     // [N, D]
    device const float* encoderWeights [[buffer(1)]],   // [L, D]
    device char* latentCodes [[buffer(2)]],             // [N, L] as INT8
    device float* scales [[buffer(3)]],                 // [N] per-vector scale
    device const float* encoderBias [[buffer(4)]],      // [L] or null
    constant NeuralQuantParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numVectors) {
        return;
    }

    const uint vectorIdx = tid;
    device const float* input = inputVectors + (vectorIdx * params.stride);

    const uint inputDim = params.inputDimension;
    const uint latentDim = params.latentDimension;

    // Thread-local storage for latent vector (max 256 dimensions)
    float latent[256];
    const uint effectiveLatentDim = min(latentDim, 256u);

    // Encode: project through encoder weights
    for (uint j = 0; j < effectiveLatentDim; ++j) {
        device const float* weightRow = encoderWeights + (j * inputDim);

        const uint simd_blocks = inputDim / 4;
        device const float4* input4 = (device const float4*)input;
        device const float4* weight4 = (device const float4*)weightRow;

        float4 acc = float4(0.0f);
        for (uint i = 0; i < simd_blocks; ++i) {
            acc = fma(input4[i], weight4[i], acc);
        }
        float sum = acc.x + acc.y + acc.z + acc.w;

        for (uint i = simd_blocks * 4; i < inputDim; ++i) {
            sum = fma(input[i], weightRow[i], sum);
        }

        // Add bias if present
        if (encoderBias) {
            sum += encoderBias[j];
        }

        latent[j] = sum;
    }

    // Apply ReLU activation
    if (params.useActivation) {
        applyReLU(latent, effectiveLatentDim);
    }

    // Optionally normalize
    if (params.normalizeLatent) {
        normalizeVector(latent, effectiveLatentDim);
    }

    // Compute scale for INT8 quantization
    float scale = computeScale(latent, effectiveLatentDim);
    scales[vectorIdx] = scale;

    // Quantize to INT8
    float invScale = 1.0f / scale;
    device char* output = latentCodes + (vectorIdx * latentDim);
    for (uint i = 0; i < effectiveLatentDim; ++i) {
        float quantized = round(latent[i] * invScale);
        output[i] = (char)clamp(quantized, -127.0f, 127.0f);
    }
}

// MARK: - Decode Kernel (float latent -> float output)

/// Decode latent vectors (float) to output vectors.
///
/// Grid dispatch: (numVectors, inputDimension, 1)
kernel void neural_decode_kernel(
    device const float* latentVectors [[buffer(0)]],    // [N, L]
    device const float* decoderWeights [[buffer(1)]],   // [D, L]
    device float* outputVectors [[buffer(2)]],          // [N, D]
    device const float* decoderBias [[buffer(3)]],      // [D] or null
    constant NeuralQuantParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vectorIdx = tid.x;
    const uint outputIdx = tid.y;

    if (vectorIdx >= params.numVectors || outputIdx >= params.inputDimension) {
        return;
    }

    device const float* latent = latentVectors + (vectorIdx * params.latentDimension);
    device const float* weightRow = decoderWeights + (outputIdx * params.latentDimension);

    const uint latentDim = params.latentDimension;
    const uint simd_blocks = latentDim / 4;

    device const float4* latent4 = (device const float4*)latent;
    device const float4* weight4 = (device const float4*)weightRow;

    float4 acc = float4(0.0f);
    for (uint i = 0; i < simd_blocks; ++i) {
        acc = fma(latent4[i], weight4[i], acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    for (uint i = simd_blocks * 4; i < latentDim; ++i) {
        sum = fma(latent[i], weightRow[i], sum);
    }

    // Add bias if present
    if (decoderBias) {
        sum += decoderBias[outputIdx];
    }

    outputVectors[vectorIdx * params.inputDimension + outputIdx] = sum;
}

// MARK: - Dequantize + Decode Kernel (INT8 -> float output)

/// Dequantize INT8 latent codes and decode to full vectors in one pass.
///
/// Grid dispatch: (numVectors, 1, 1)
kernel void neural_dequantize_decode_kernel(
    device const char* latentCodes [[buffer(0)]],       // [N, L] as INT8
    device const float* scales [[buffer(1)]],           // [N] per-vector scale
    device const float* decoderWeights [[buffer(2)]],   // [D, L]
    device float* outputVectors [[buffer(3)]],          // [N, D]
    device const float* decoderBias [[buffer(4)]],      // [D] or null
    constant NeuralQuantParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numVectors) {
        return;
    }

    const uint vectorIdx = tid;
    const uint latentDim = params.latentDimension;
    const uint outputDim = params.inputDimension;

    device const char* codes = latentCodes + (vectorIdx * latentDim);
    float scale = scales[vectorIdx];

    // Thread-local storage for dequantized latent
    float latent[256];
    const uint effectiveLatentDim = min(latentDim, 256u);

    // Dequantize
    for (uint i = 0; i < effectiveLatentDim; ++i) {
        latent[i] = float(codes[i]) * scale;
    }

    // Decode: project through decoder weights
    device float* output = outputVectors + (vectorIdx * outputDim);

    for (uint j = 0; j < outputDim; ++j) {
        device const float* weightRow = decoderWeights + (j * latentDim);

        float sum = 0.0f;

        // SIMD-optimized for latent dimension
        const uint simd_blocks = effectiveLatentDim / 4;
        for (uint i = 0; i < simd_blocks; ++i) {
            float4 l = float4(latent[i*4], latent[i*4+1], latent[i*4+2], latent[i*4+3]);
            float4 w = float4(weightRow[i*4], weightRow[i*4+1], weightRow[i*4+2], weightRow[i*4+3]);
            sum += dot(l, w);
        }

        for (uint i = simd_blocks * 4; i < effectiveLatentDim; ++i) {
            sum = fma(latent[i], weightRow[i], sum);
        }

        // Add bias if present
        if (decoderBias) {
            sum += decoderBias[j];
        }

        output[j] = sum;
    }
}

// MARK: - Tiled GEMM Encoder (Phase 1)

/// Parameters for tiled encode kernel
struct TiledEncodeParams {
    uint32_t numVectors;       // N - total vectors to encode
    uint32_t inputDimension;   // D - input dim (e.g., 768)
    uint32_t latentDimension;  // L - latent dim (e.g., 128)
    uint32_t stride;           // Input stride (usually = inputDimension)
    uint32_t vectorsPerTG;     // Vectors per threadgroup (e.g., 32)
    uint8_t  useActivation;    // Apply ReLU
    uint8_t  padding[3];       // Alignment padding
};

/// Tiled GEMM encoder with weight caching in threadgroup memory.
///
/// This kernel achieves 10-50x speedup over the naive encoder by:
/// 1. Caching weight tiles in threadgroup memory (read once, use 32x)
/// 2. Processing multiple vectors per threadgroup cooperatively
/// 3. Amortizing weight bandwidth across vectors
///
/// Memory layout:
/// - Weight tile: [TILE_L][TILE_D] = 16×64 = 4KB in threadgroup memory
/// - Each threadgroup processes VECTORS_PER_TG=32 vectors
/// - 256 threads cooperate on tile loading and computation
///
/// Grid dispatch: threadgroups = (ceil(numVectors/32), 1, 1)
///                threadsPerThreadgroup = (256, 1, 1)
///
/// Output: Float latent vectors (not quantized - Phase 2 adds quantization)
kernel void neural_encode_tiled_kernel(
    device const float* inputVectors      [[buffer(0)]],  // [N, D]
    device const float* encoderWeights    [[buffer(1)]],  // [L, D] row-major
    device float* latentVectors           [[buffer(2)]],  // [N, L] float output
    device const float* encoderBias       [[buffer(3)]],  // [L] or null
    constant TiledEncodeParams& params    [[buffer(4)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint  tii  [[thread_index_in_threadgroup]]
) {
    // ========== Configuration ==========
    // These are compile-time constants for the kernel
    constexpr uint TILE_L = 16;          // Latent dimensions per tile
    constexpr uint TILE_D = 64;          // Input dimensions per tile
    constexpr uint VECTORS_PER_TG = 32;  // Vectors per threadgroup
    constexpr uint THREADS_PER_TG = 256; // Threads per threadgroup
    constexpr uint THREADS_PER_VECTOR = THREADS_PER_TG / VECTORS_PER_TG;  // 8
    constexpr uint LATENT_PER_THREAD = TILE_L / THREADS_PER_VECTOR;       // 2

    // ========== Threadgroup Memory ==========
    // Weight tile: [TILE_L][TILE_D] = 16×64×4 = 4KB
    threadgroup float weightTile[TILE_L][TILE_D];

    // ========== Thread Assignment ==========
    // 256 threads / 32 vectors = 8 threads per vector
    // Each thread computes LATENT_PER_THREAD=2 latent dims per L-tile
    const uint localVectorIdx = tii / THREADS_PER_VECTOR;   // 0-31
    const uint globalVectorIdx = tgp.x * VECTORS_PER_TG + localVectorIdx;
    const uint laneInVector = tii % THREADS_PER_VECTOR;     // 0-7

    // Early exit for out-of-bounds vectors
    // Note: All threads in TG participate in tile loading, so we must be careful
    // Only skip computation, not tile loading
    const bool validVector = globalVectorIdx < params.numVectors;

    // This thread computes latent indices: laneInVector * LATENT_PER_THREAD + (0 to LATENT_PER_THREAD-1)
    // For 8 threads per vector, LATENT_PER_THREAD=2: thread 0 → dims 0,1; thread 1 → dims 2,3; etc.
    const uint latentOffset = laneInVector * LATENT_PER_THREAD;

    // Get input pointer for this thread's vector
    device const float* input = inputVectors + globalVectorIdx * params.stride;

    const uint inputDim = params.inputDimension;
    const uint latentDim = params.latentDimension;

    // ========== Iterate Over L-Tiles ==========
    // For 128-dim latent with TILE_L=16: 8 iterations
    for (uint tileL = 0; tileL < latentDim; tileL += TILE_L) {
        // Accumulators for this L-tile (reset for each L-tile)
        float acc[LATENT_PER_THREAD];
        for (uint i = 0; i < LATENT_PER_THREAD; ++i) {
            acc[i] = 0.0f;
        }

        // ========== Iterate Over D-Tiles ==========
        // For 768-dim input with TILE_D=64: 12 iterations
        for (uint tileD = 0; tileD < inputDim; tileD += TILE_D) {

            // ========== Cooperative Tile Load ==========
            // 256 threads load 16×64 = 1024 elements = 4 elements per thread
            constexpr uint TILE_ELEMENTS = TILE_L * TILE_D;
            constexpr uint ELEMS_PER_THREAD = TILE_ELEMENTS / THREADS_PER_TG;  // 4

            for (uint e = 0; e < ELEMS_PER_THREAD; ++e) {
                const uint flatIdx = tii + e * THREADS_PER_TG;
                const uint row = flatIdx / TILE_D;  // 0-15 (L dimension)
                const uint col = flatIdx % TILE_D;  // 0-63 (D dimension)

                const uint globalL = tileL + row;
                const uint globalD = tileD + col;

                // Bounds check and load
                if (globalL < latentDim && globalD < inputDim) {
                    weightTile[row][col] = encoderWeights[globalL * inputDim + globalD];
                } else {
                    weightTile[row][col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ========== Compute with Cached Tile ==========
            // Only compute if this thread has a valid vector
            if (validVector) {
                // For each latent dimension this thread is responsible for
                for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                    const uint tileRow = latentOffset + localL;

                    // Skip if this latent dim is out of bounds
                    if (tileL + tileRow >= latentDim) continue;

                    // Accumulate: sum over D-tile
                    float sum = 0.0f;

                    // Vectorized inner loop: process 4 elements at a time
                    const uint validD = min(TILE_D, inputDim - tileD);
                    const uint d4 = validD / 4;

                    for (uint d = 0; d < d4; ++d) {
                        const uint dBase = d * 4;
                        const uint globalD = tileD + dBase;

                        // Load 4 input values
                        const float4 in4 = *((device const float4*)(input + globalD));

                        // Load 4 weight values from tile
                        const float4 w4 = float4(
                            weightTile[tileRow][dBase + 0],
                            weightTile[tileRow][dBase + 1],
                            weightTile[tileRow][dBase + 2],
                            weightTile[tileRow][dBase + 3]
                        );

                        sum += dot(in4, w4);
                    }

                    // Handle remainder
                    for (uint d = d4 * 4; d < validD; ++d) {
                        const uint globalD = tileD + d;
                        sum += input[globalD] * weightTile[tileRow][d];
                    }

                    acc[localL] += sum;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ========== Write Results for This L-Tile ==========
        if (validVector) {
            device float* output = latentVectors + globalVectorIdx * latentDim;

            for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                const uint globalL = tileL + latentOffset + localL;

                if (globalL < latentDim) {
                    float val = acc[localL];

                    // Add bias if present
                    if (encoderBias) {
                        val += encoderBias[globalL];
                    }

                    // Apply ReLU activation
                    if (params.useActivation) {
                        val = max(val, 0.0f);
                    }

                    output[globalL] = val;
                }
            }
        }
    }
}

// MARK: - Tiled GEMM Encoder with INT8 Quantization (Phase 2)

/// Tiled GEMM encoder with INT8 quantization output.
///
/// Extends the tiled encoder (Phase 1) to output INT8 latent codes with per-vector scales.
/// Combines weight caching benefits of tiled kernel with quantization in one pass.
///
/// Key differences from Phase 1:
/// 1. Accumulates full latent vectors in threadgroup memory (not just per-tile)
/// 2. Computes per-vector scale factor via parallel reduction
/// 3. Quantizes to INT8 and writes to output
///
/// Memory layout:
/// - Weight tile: [TILE_L][TILE_D] = 16×64 = 4KB
/// - Latent accumulators: [VECTORS_PER_TG][LATENT_DIM] = 32×128 = 16KB
/// - Scale reduction: [VECTORS_PER_TG][THREADS_PER_VECTOR] = 32×8 = 1KB
/// - Total: ~21KB (well under 32KB limit)
///
/// Grid dispatch: threadgroups = (ceil(numVectors/32), 1, 1)
///                threadsPerThreadgroup = (256, 1, 1)
kernel void neural_encode_quantize_tiled_kernel(
    device const float* inputVectors      [[buffer(0)]],  // [N, D]
    device const float* encoderWeights    [[buffer(1)]],  // [L, D] row-major
    device char* latentCodes              [[buffer(2)]],  // [N, L] INT8 output
    device float* scales                  [[buffer(3)]],  // [N] per-vector scale
    device const float* encoderBias       [[buffer(4)]],  // [L] or null
    constant TiledEncodeParams& params    [[buffer(5)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint  tii  [[thread_index_in_threadgroup]]
) {
    // ========== Configuration ==========
    constexpr uint TILE_L = 16;          // Latent dimensions per tile
    constexpr uint TILE_D = 64;          // Input dimensions per tile
    constexpr uint VECTORS_PER_TG = 32;  // Vectors per threadgroup
    constexpr uint THREADS_PER_TG = 256; // Threads per threadgroup
    constexpr uint THREADS_PER_VECTOR = THREADS_PER_TG / VECTORS_PER_TG;  // 8
    constexpr uint LATENT_PER_THREAD = TILE_L / THREADS_PER_VECTOR;       // 2
    constexpr uint MAX_LATENT_DIM = 128; // Maximum supported latent dimension

    // ========== Threadgroup Memory ==========
    // Weight tile: [TILE_L][TILE_D] = 16×64×4 = 4KB
    threadgroup float weightTile[TILE_L][TILE_D];

    // Latent accumulators: [VECTORS_PER_TG][MAX_LATENT_DIM] = 32×128×4 = 16KB
    threadgroup float latentAccum[VECTORS_PER_TG][MAX_LATENT_DIM];

    // Partial max values for reduction: [VECTORS_PER_TG][THREADS_PER_VECTOR] = 32×8×4 = 1KB
    threadgroup float partialMax[VECTORS_PER_TG][THREADS_PER_VECTOR];

    // Per-vector scales
    threadgroup float tgScales[VECTORS_PER_TG];

    // ========== Thread Assignment ==========
    const uint localVectorIdx = tii / THREADS_PER_VECTOR;   // 0-31
    const uint globalVectorIdx = tgp.x * VECTORS_PER_TG + localVectorIdx;
    const uint laneInVector = tii % THREADS_PER_VECTOR;     // 0-7

    const bool validVector = globalVectorIdx < params.numVectors;

    // Get input pointer
    device const float* input = inputVectors + globalVectorIdx * params.stride;

    const uint inputDim = params.inputDimension;
    const uint latentDim = params.latentDimension;
    const uint effectiveLatentDim = min(latentDim, MAX_LATENT_DIM);

    // ========== Phase 1: Initialize Accumulators ==========
    // Each thread initializes its portion of the latent accumulator
    for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
        latentAccum[localVectorIdx][l] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 2: Tiled GEMM (same as Phase 1 kernel) ==========
    for (uint tileL = 0; tileL < latentDim; tileL += TILE_L) {
        // Local accumulators for this L-tile
        float acc[LATENT_PER_THREAD];
        for (uint i = 0; i < LATENT_PER_THREAD; ++i) {
            acc[i] = 0.0f;
        }

        const uint latentOffset = laneInVector * LATENT_PER_THREAD;

        // Iterate over D-tiles
        for (uint tileD = 0; tileD < inputDim; tileD += TILE_D) {
            // Cooperative tile load
            constexpr uint TILE_ELEMENTS = TILE_L * TILE_D;
            constexpr uint ELEMS_PER_THREAD = TILE_ELEMENTS / THREADS_PER_TG;

            for (uint e = 0; e < ELEMS_PER_THREAD; ++e) {
                const uint flatIdx = tii + e * THREADS_PER_TG;
                const uint row = flatIdx / TILE_D;
                const uint col = flatIdx % TILE_D;

                const uint globalL = tileL + row;
                const uint globalD = tileD + col;

                if (globalL < latentDim && globalD < inputDim) {
                    weightTile[row][col] = encoderWeights[globalL * inputDim + globalD];
                } else {
                    weightTile[row][col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute with cached tile
            if (validVector) {
                for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                    const uint tileRow = latentOffset + localL;

                    if (tileL + tileRow >= latentDim) continue;

                    float sum = 0.0f;

                    const uint validD = min(TILE_D, inputDim - tileD);
                    const uint d4 = validD / 4;

                    for (uint d = 0; d < d4; ++d) {
                        const uint dBase = d * 4;
                        const uint globalD = tileD + dBase;

                        const float4 in4 = *((device const float4*)(input + globalD));
                        const float4 w4 = float4(
                            weightTile[tileRow][dBase + 0],
                            weightTile[tileRow][dBase + 1],
                            weightTile[tileRow][dBase + 2],
                            weightTile[tileRow][dBase + 3]
                        );

                        sum += dot(in4, w4);
                    }

                    for (uint d = d4 * 4; d < validD; ++d) {
                        const uint globalD = tileD + d;
                        sum += input[globalD] * weightTile[tileRow][d];
                    }

                    acc[localL] += sum;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store results to threadgroup accumulator (not device memory)
        if (validVector) {
            for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                const uint globalL = tileL + latentOffset + localL;

                if (globalL < latentDim) {
                    float val = acc[localL];

                    // Add bias if present
                    if (encoderBias) {
                        val += encoderBias[globalL];
                    }

                    // Apply ReLU activation
                    if (params.useActivation) {
                        val = max(val, 0.0f);
                    }

                    latentAccum[localVectorIdx][globalL] = val;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 3: Scale Computation via Parallel Reduction ==========
    // Each thread computes max over its assigned latent dims
    float localMax = 0.0f;
    if (validVector) {
        for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
            localMax = max(localMax, abs(latentAccum[localVectorIdx][l]));
        }
    }

    // Store to partial max array
    partialMax[localVectorIdx][laneInVector] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane 0 does final reduction for each vector
    if (laneInVector == 0 && validVector) {
        float maxAbs = 0.0f;
        for (uint i = 0; i < THREADS_PER_VECTOR; ++i) {
            maxAbs = max(maxAbs, partialMax[localVectorIdx][i]);
        }
        tgScales[localVectorIdx] = max(maxAbs / 127.0f, VA_EPSILON);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 4: Quantize and Write Output ==========
    if (validVector) {
        const float scale = tgScales[localVectorIdx];
        const float invScale = 1.0f / scale;

        device char* output = latentCodes + globalVectorIdx * latentDim;

        // Each thread quantizes its assigned latent dims
        for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
            float val = latentAccum[localVectorIdx][l];
            float quantized = round(val * invScale);
            output[l] = (char)clamp(quantized, -127.0f, 127.0f);
        }

        // Lane 0 writes the scale
        if (laneInVector == 0) {
            scales[globalVectorIdx] = scale;
        }
    }
}

// MARK: - Tiled GEMM Encoder with Dual Accumulators (Phase 3)

/// Optimized tiled GEMM encoder with dual accumulators for latency hiding.
///
/// This kernel extends Phase 2's tiled encoder with:
/// 1. Dual accumulator sets to hide 3-4 cycle FMA latency on Apple Silicon
/// 2. Processing 8 D-elements per iteration (vs 4 in V1) with interleaved FMAs
/// 3. Same threadgroup memory layout and cooperative patterns as V1
///
/// Performance improvement comes from:
/// - While sum0 += dot(in0, w0) computes, load in1/w1 can proceed
/// - While sum1 += dot(in1, w1) computes, the result of sum0 is ready
/// - Effectively doubles compute throughput by filling FMA pipeline bubbles
///
/// Expected speedup: 1.5-2x over Phase 2 (V1) kernel
///
/// Grid dispatch: threadgroups = (ceil(numVectors/32), 1, 1)
///                threadsPerThreadgroup = (256, 1, 1)
kernel void neural_encode_quantize_tiled_v2_kernel(
    device const float* inputVectors      [[buffer(0)]],  // [N, D]
    device const float* encoderWeights    [[buffer(1)]],  // [L, D] row-major
    device char* latentCodes              [[buffer(2)]],  // [N, L] INT8 output
    device float* scales                  [[buffer(3)]],  // [N] per-vector scale
    device const float* encoderBias       [[buffer(4)]],  // [L] or null
    constant TiledEncodeParams& params    [[buffer(5)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint  tii  [[thread_index_in_threadgroup]]
) {
    // ========== Configuration ==========
    constexpr uint TILE_L = 16;          // Latent dimensions per tile
    constexpr uint TILE_D = 64;          // Input dimensions per tile
    constexpr uint VECTORS_PER_TG = 32;  // Vectors per threadgroup
    constexpr uint THREADS_PER_TG = 256; // Threads per threadgroup
    constexpr uint THREADS_PER_VECTOR = THREADS_PER_TG / VECTORS_PER_TG;  // 8
    constexpr uint LATENT_PER_THREAD = TILE_L / THREADS_PER_VECTOR;       // 2
    constexpr uint MAX_LATENT_DIM = 128; // Maximum supported latent dimension

    // ========== Threadgroup Memory ==========
    // Weight tile: [TILE_L][TILE_D] = 16×64×4 = 4KB
    threadgroup float weightTile[TILE_L][TILE_D];

    // Latent accumulators: [VECTORS_PER_TG][MAX_LATENT_DIM] = 32×128×4 = 16KB
    threadgroup float latentAccum[VECTORS_PER_TG][MAX_LATENT_DIM];

    // Partial max values for reduction: [VECTORS_PER_TG][THREADS_PER_VECTOR] = 32×8×4 = 1KB
    threadgroup float partialMax[VECTORS_PER_TG][THREADS_PER_VECTOR];

    // Per-vector scales
    threadgroup float tgScales[VECTORS_PER_TG];

    // ========== Thread Assignment ==========
    const uint localVectorIdx = tii / THREADS_PER_VECTOR;   // 0-31
    const uint globalVectorIdx = tgp.x * VECTORS_PER_TG + localVectorIdx;
    const uint laneInVector = tii % THREADS_PER_VECTOR;     // 0-7

    const bool validVector = globalVectorIdx < params.numVectors;

    // Get input pointer
    device const float* input = inputVectors + globalVectorIdx * params.stride;

    const uint inputDim = params.inputDimension;
    const uint latentDim = params.latentDimension;
    const uint effectiveLatentDim = min(latentDim, MAX_LATENT_DIM);

    // ========== Phase 1: Initialize Accumulators ==========
    for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
        latentAccum[localVectorIdx][l] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 2: Tiled GEMM with Dual Accumulators ==========
    for (uint tileL = 0; tileL < latentDim; tileL += TILE_L) {
        // Local accumulators for this L-tile (DUAL for latency hiding)
        float acc0[LATENT_PER_THREAD];  // Even D-groups
        float acc1[LATENT_PER_THREAD];  // Odd D-groups
        for (uint i = 0; i < LATENT_PER_THREAD; ++i) {
            acc0[i] = 0.0f;
            acc1[i] = 0.0f;
        }

        const uint latentOffset = laneInVector * LATENT_PER_THREAD;

        // Iterate over D-tiles
        for (uint tileD = 0; tileD < inputDim; tileD += TILE_D) {
            // Cooperative tile load (same as V1)
            constexpr uint TILE_ELEMENTS = TILE_L * TILE_D;
            constexpr uint ELEMS_PER_THREAD = TILE_ELEMENTS / THREADS_PER_TG;

            for (uint e = 0; e < ELEMS_PER_THREAD; ++e) {
                const uint flatIdx = tii + e * THREADS_PER_TG;
                const uint row = flatIdx / TILE_D;
                const uint col = flatIdx % TILE_D;

                const uint globalL = tileL + row;
                const uint globalD = tileD + col;

                if (globalL < latentDim && globalD < inputDim) {
                    weightTile[row][col] = encoderWeights[globalL * inputDim + globalD];
                } else {
                    weightTile[row][col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ========== Compute with DUAL Accumulators ==========
            if (validVector) {
                for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                    const uint tileRow = latentOffset + localL;

                    if (tileL + tileRow >= latentDim) continue;

                    const uint validD = min(TILE_D, inputDim - tileD);
                    const uint d8 = validD / 8;  // Process 8 elements per iteration
                    const uint d4 = validD / 4;  // For remainder

                    // ===== Main loop: 8 elements per iteration with dual accumulators =====
                    for (uint d = 0; d < d8; ++d) {
                        const uint dBase0 = d * 8;        // First group of 4
                        const uint dBase1 = d * 8 + 4;    // Second group of 4
                        const uint globalD0 = tileD + dBase0;
                        const uint globalD1 = tileD + dBase1;

                        // Load input values for both groups
                        const float4 in0 = *((device const float4*)(input + globalD0));
                        const float4 in1 = *((device const float4*)(input + globalD1));

                        // Load weight values from tile for both groups
                        const float4 w0 = float4(
                            weightTile[tileRow][dBase0 + 0],
                            weightTile[tileRow][dBase0 + 1],
                            weightTile[tileRow][dBase0 + 2],
                            weightTile[tileRow][dBase0 + 3]
                        );
                        const float4 w1 = float4(
                            weightTile[tileRow][dBase1 + 0],
                            weightTile[tileRow][dBase1 + 1],
                            weightTile[tileRow][dBase1 + 2],
                            weightTile[tileRow][dBase1 + 3]
                        );

                        // INTERLEAVED accumulation - hides FMA latency
                        acc0[localL] += dot(in0, w0);
                        acc1[localL] += dot(in1, w1);
                    }

                    // ===== Handle middle remainder (4-7 elements) =====
                    const uint remainStart = d8 * 2;  // Start of remainder in d4 units
                    if (remainStart < d4) {
                        const uint dBase = remainStart * 4;
                        const uint globalD = tileD + dBase;

                        const float4 in4 = *((device const float4*)(input + globalD));
                        const float4 w4 = float4(
                            weightTile[tileRow][dBase + 0],
                            weightTile[tileRow][dBase + 1],
                            weightTile[tileRow][dBase + 2],
                            weightTile[tileRow][dBase + 3]
                        );

                        acc0[localL] += dot(in4, w4);
                    }

                    // ===== Handle final remainder (0-3 elements) =====
                    for (uint d = d4 * 4; d < validD; ++d) {
                        const uint globalD = tileD + d;
                        acc0[localL] += input[globalD] * weightTile[tileRow][d];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ===== Merge dual accumulators and store to threadgroup memory =====
        if (validVector) {
            for (uint localL = 0; localL < LATENT_PER_THREAD; ++localL) {
                const uint globalL = tileL + latentOffset + localL;

                if (globalL < latentDim) {
                    // Merge both accumulator sets
                    float val = acc0[localL] + acc1[localL];

                    // Add bias if present
                    if (encoderBias) {
                        val += encoderBias[globalL];
                    }

                    // Apply ReLU activation
                    if (params.useActivation) {
                        val = max(val, 0.0f);
                    }

                    latentAccum[localVectorIdx][globalL] = val;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 3: Scale Computation via Parallel Reduction ==========
    float localMax = 0.0f;
    if (validVector) {
        for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
            localMax = max(localMax, abs(latentAccum[localVectorIdx][l]));
        }
    }

    partialMax[localVectorIdx][laneInVector] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane 0 does final reduction
    if (laneInVector == 0 && validVector) {
        float maxAbs = 0.0f;
        for (uint i = 0; i < THREADS_PER_VECTOR; ++i) {
            maxAbs = max(maxAbs, partialMax[localVectorIdx][i]);
        }
        tgScales[localVectorIdx] = max(maxAbs / 127.0f, VA_EPSILON);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 4: Quantize and Write Output ==========
    if (validVector) {
        const float scale = tgScales[localVectorIdx];
        const float invScale = 1.0f / scale;

        device char* output = latentCodes + globalVectorIdx * latentDim;

        for (uint l = laneInVector; l < effectiveLatentDim; l += THREADS_PER_VECTOR) {
            float val = latentAccum[localVectorIdx][l];
            float quantized = round(val * invScale);
            output[l] = (char)clamp(quantized, -127.0f, 127.0f);
        }

        if (laneInVector == 0) {
            scales[globalVectorIdx] = scale;
        }
    }
}

// MARK: - Optimized Kernels for Common Configurations

/// Optimized encoder for 768 -> 64 (high compression)
kernel void neural_encode_768_to_64_kernel(
    device const float* inputVectors [[buffer(0)]],
    device const float* encoderWeights [[buffer(1)]],
    device char* latentCodes [[buffer(2)]],
    device float* scales [[buffer(3)]],
    device const float* encoderBias [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numVectors) {
        return;
    }

    constexpr uint INPUT_DIM = 768;
    constexpr uint LATENT_DIM = 64;
    constexpr uint INPUT_BLOCKS = INPUT_DIM / 4;

    device const float* input = inputVectors + (tid * INPUT_DIM);
    device const float4* input4 = (device const float4*)input;

    float latent[LATENT_DIM];

    // Encode with loop unrolling for 64 outputs
    for (uint j = 0; j < LATENT_DIM; j += 4) {
        device const float4* w0 = (device const float4*)(encoderWeights + (j+0) * INPUT_DIM);
        device const float4* w1 = (device const float4*)(encoderWeights + (j+1) * INPUT_DIM);
        device const float4* w2 = (device const float4*)(encoderWeights + (j+2) * INPUT_DIM);
        device const float4* w3 = (device const float4*)(encoderWeights + (j+3) * INPUT_DIM);

        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        float4 acc3 = float4(0.0f);

        for (uint i = 0; i < INPUT_BLOCKS; ++i) {
            float4 in = input4[i];
            acc0 = fma(in, w0[i], acc0);
            acc1 = fma(in, w1[i], acc1);
            acc2 = fma(in, w2[i], acc2);
            acc3 = fma(in, w3[i], acc3);
        }

        latent[j+0] = acc0.x + acc0.y + acc0.z + acc0.w;
        latent[j+1] = acc1.x + acc1.y + acc1.z + acc1.w;
        latent[j+2] = acc2.x + acc2.y + acc2.z + acc2.w;
        latent[j+3] = acc3.x + acc3.y + acc3.z + acc3.w;

        if (encoderBias) {
            latent[j+0] += encoderBias[j+0];
            latent[j+1] += encoderBias[j+1];
            latent[j+2] += encoderBias[j+2];
            latent[j+3] += encoderBias[j+3];
        }
    }

    // ReLU
    if (params.useActivation) {
        for (uint i = 0; i < LATENT_DIM; ++i) {
            latent[i] = max(latent[i], 0.0f);
        }
    }

    // Compute scale and quantize
    float maxAbs = 0.0f;
    for (uint i = 0; i < LATENT_DIM; ++i) {
        maxAbs = max(maxAbs, abs(latent[i]));
    }
    float scale = max(maxAbs / 127.0f, VA_EPSILON);
    scales[tid] = scale;

    float invScale = 1.0f / scale;
    device char* output = latentCodes + (tid * LATENT_DIM);
    for (uint i = 0; i < LATENT_DIM; ++i) {
        output[i] = (char)clamp(round(latent[i] * invScale), -127.0f, 127.0f);
    }
}

/// Optimized encoder for 384 -> 64 (MiniLM configuration)
kernel void neural_encode_384_to_64_kernel(
    device const float* inputVectors [[buffer(0)]],
    device const float* encoderWeights [[buffer(1)]],
    device char* latentCodes [[buffer(2)]],
    device float* scales [[buffer(3)]],
    device const float* encoderBias [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numVectors) {
        return;
    }

    constexpr uint INPUT_DIM = 384;
    constexpr uint LATENT_DIM = 64;
    constexpr uint INPUT_BLOCKS = INPUT_DIM / 4;

    device const float* input = inputVectors + (tid * INPUT_DIM);
    device const float4* input4 = (device const float4*)input;

    float latent[LATENT_DIM];

    for (uint j = 0; j < LATENT_DIM; ++j) {
        device const float4* weight4 = (device const float4*)(encoderWeights + j * INPUT_DIM);

        float4 acc = float4(0.0f);
        for (uint i = 0; i < INPUT_BLOCKS; ++i) {
            acc = fma(input4[i], weight4[i], acc);
        }

        float sum = acc.x + acc.y + acc.z + acc.w;
        if (encoderBias) {
            sum += encoderBias[j];
        }

        latent[j] = params.useActivation ? max(sum, 0.0f) : sum;
    }

    // Compute scale and quantize
    float maxAbs = 0.0f;
    for (uint i = 0; i < LATENT_DIM; ++i) {
        maxAbs = max(maxAbs, abs(latent[i]));
    }
    float scale = max(maxAbs / 127.0f, VA_EPSILON);
    scales[tid] = scale;

    float invScale = 1.0f / scale;
    device char* output = latentCodes + (tid * LATENT_DIM);
    for (uint i = 0; i < LATENT_DIM; ++i) {
        output[i] = (char)clamp(round(latent[i] * invScale), -127.0f, 127.0f);
    }
}

// MARK: - Optimized 2D Decode Kernel

/// Optimized dequantize + decode kernel using 2D grid dispatch and threadgroup caching.
///
/// This kernel addresses the performance bottleneck in the original 1D dispatch:
/// - Original: 1 thread per vector → all 768 outputs computed sequentially → ~9.5k vec/s
/// - Optimized: 2D grid parallelizes across output dimensions → target >50k vec/s
///
/// Key optimizations:
/// 1. **Threadgroup-cached latent**: INT8→Float32 dequantization done once per output tile,
///    not once per output element. Reduces redundant work by ~32×.
/// 2. **2D parallelism**: Each thread computes one output element, maximizing GPU occupancy.
/// 3. **Vectorized dot products**: Uses float4 operations for weight × latent computation.
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/32), 1)
///                threadsPerThreadgroup = (1, 32, 1)
///
/// Buffer layout (same as original):
/// - buffer(0): INT8 latent codes [N, latentDim]
/// - buffer(1): scales [N]
/// - buffer(2): decoder weights [inputDim, latentDim] row-major
/// - buffer(3): output [N, inputDim]
/// - buffer(4): decoder bias [inputDim] or null
/// - buffer(5): NeuralQuantParams
kernel void neural_dequantize_decode_2d_tg_kernel(
    device const char*  latentCodes    [[buffer(0)]],  // [N, latentDim] int8
    device const float* scales         [[buffer(1)]],  // [N]
    device const float* decoderWeights [[buffer(2)]],  // [inputDim, latentDim] row-major
    device float*       outputVectors  [[buffer(3)]],  // [N, inputDim] row-major
    device const float* decoderBias    [[buffer(4)]],  // [inputDim] or null
    constant NeuralQuantParams& params [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    // One vector per threadgroup in X dimension
    const uint vectorIdx = tgp.x;
    if (vectorIdx >= params.numVectors) {
        return;  // Uniform early exit - safe before barriers
    }

    const uint inputDim  = params.inputDimension;
    const uint latentDim = params.latentDimension;

    // Threadgroup cache for dequantized latent packed as float4 (max 128 dims → 32 float4s)
    threadgroup float4 tgLatent4[32];

    // Load scale once per threadgroup (cooperative load, thread 0,0 writes)
    // Initialize to 0 to silence uninitialized warning (barrier ensures correct value is read)
    threadgroup float tgScale = 0.0f;
    if (tptg.x == 0 && tptg.y == 0) {
        tgScale = scales[vectorIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = tgScale;

    // Convert latent INT8 → Float32 * scale, once per output tile.
    // Cooperative load: threads in Y dimension share this work.
    // Requires latentDim % 4 == 0 (true for 64, 128).
    const uint latentDim4 = latentDim >> 2;

    // Safety check for unsupported configurations (latentDim > 128)
    if (latentDim4 > 32) {
        return;
    }

    for (uint i4 = tptg.y; i4 < latentDim4; i4 += tgs.y) {
        const uint codeOffset = vectorIdx * latentDim + i4 * 4;

        // Load 4 INT8 codes and dequantize to float4
        // Safe because latentDim is multiple of 4
        const char4 c = *((device const char4*)(latentCodes + codeOffset));
        tgLatent4[i4] = float4(c) * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute output index this thread is responsible for.
    // Tiled across threadgroups in Y, threads in threadgroup.y handle different outputs.
    const uint outIdx = tgp.y * tgs.y + tptg.y;

    // IMPORTANT: Only early-return after all barriers are complete
    if (outIdx >= inputDim) {
        return;
    }

    // Weight row for this output element: [latentDim] floats, cast to float4 for vectorized dot
    device const float4* w4 = (device const float4*)(decoderWeights + outIdx * latentDim);

    // Initialize with bias if present
    float sum = decoderBias ? decoderBias[outIdx] : 0.0f;

    // Fast paths for common latent dimensions with unrolled loops
    if (latentDim == 128) {
        #pragma unroll
        for (uint i4 = 0; i4 < 32; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else if (latentDim == 64) {
        #pragma unroll
        for (uint i4 = 0; i4 < 16; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else {
        // Generic path for other latent dimensions
        for (uint i4 = 0; i4 < latentDim4; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    }

    outputVectors[vectorIdx * inputDim + outIdx] = sum;
}

// MARK: - Threadgroup Size Variants

/// 64-thread variant: processes 64 output dimensions per threadgroup.
/// Better for larger output dimensions (768) where more parallelism helps.
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/64), 1)
///                threadsPerThreadgroup = (1, 64, 1)
kernel void neural_dequantize_decode_2d_tg64_kernel(
    device const char*  latentCodes    [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* decoderWeights [[buffer(2)]],
    device float*       outputVectors  [[buffer(3)]],
    device const float* decoderBias    [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    const uint vectorIdx = tgp.x;
    if (vectorIdx >= params.numVectors) {
        return;
    }

    const uint inputDim  = params.inputDimension;
    const uint latentDim = params.latentDimension;

    threadgroup float4 tgLatent4[32];
    threadgroup float tgScale = 0.0f;

    if (tptg.x == 0 && tptg.y == 0) {
        tgScale = scales[vectorIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = tgScale;
    const uint latentDim4 = latentDim >> 2;

    if (latentDim4 > 32) {
        return;
    }

    // With 64 threads, we can load the full 128-dim latent in 2 iterations (32 float4s / 64 threads)
    // or the 64-dim latent in 1 iteration
    for (uint i4 = tptg.y; i4 < latentDim4; i4 += 64) {
        const uint codeOffset = vectorIdx * latentDim + i4 * 4;
        const char4 c = *((device const char4*)(latentCodes + codeOffset));
        tgLatent4[i4] = float4(c) * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint outIdx = tgp.y * 64 + tptg.y;

    if (outIdx >= inputDim) {
        return;
    }

    device const float4* w4 = (device const float4*)(decoderWeights + outIdx * latentDim);
    float sum = decoderBias ? decoderBias[outIdx] : 0.0f;

    if (latentDim == 128) {
        #pragma unroll
        for (uint i4 = 0; i4 < 32; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else if (latentDim == 64) {
        #pragma unroll
        for (uint i4 = 0; i4 < 16; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else {
        for (uint i4 = 0; i4 < latentDim4; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    }

    outputVectors[vectorIdx * inputDim + outIdx] = sum;
}

/// 128-thread variant: processes 128 output dimensions per threadgroup.
/// Maximum parallelism for large output dimensions.
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/128), 1)
///                threadsPerThreadgroup = (1, 128, 1)
kernel void neural_dequantize_decode_2d_tg128_kernel(
    device const char*  latentCodes    [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* decoderWeights [[buffer(2)]],
    device float*       outputVectors  [[buffer(3)]],
    device const float* decoderBias    [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    const uint vectorIdx = tgp.x;
    if (vectorIdx >= params.numVectors) {
        return;
    }

    const uint inputDim  = params.inputDimension;
    const uint latentDim = params.latentDimension;

    threadgroup float4 tgLatent4[32];
    threadgroup float tgScale = 0.0f;

    if (tptg.x == 0 && tptg.y == 0) {
        tgScale = scales[vectorIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = tgScale;
    const uint latentDim4 = latentDim >> 2;

    if (latentDim4 > 32) {
        return;
    }

    // With 128 threads, we can load the full 128-dim latent (32 float4s) in one go
    // Each of first 32 threads loads one float4
    if (tptg.y < latentDim4) {
        const uint codeOffset = vectorIdx * latentDim + tptg.y * 4;
        const char4 c = *((device const char4*)(latentCodes + codeOffset));
        tgLatent4[tptg.y] = float4(c) * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint outIdx = tgp.y * 128 + tptg.y;

    if (outIdx >= inputDim) {
        return;
    }

    device const float4* w4 = (device const float4*)(decoderWeights + outIdx * latentDim);
    float sum = decoderBias ? decoderBias[outIdx] : 0.0f;

    if (latentDim == 128) {
        #pragma unroll
        for (uint i4 = 0; i4 < 32; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else if (latentDim == 64) {
        #pragma unroll
        for (uint i4 = 0; i4 < 16; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else {
        for (uint i4 = 0; i4 < latentDim4; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    }

    outputVectors[vectorIdx * inputDim + outIdx] = sum;
}

/// 256-thread variant: processes 256 output dimensions per threadgroup.
/// Reduces threadgroup count for very large batches.
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/256), 1)
///                threadsPerThreadgroup = (1, 256, 1)
kernel void neural_dequantize_decode_2d_tg256_kernel(
    device const char*  latentCodes    [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* decoderWeights [[buffer(2)]],
    device float*       outputVectors  [[buffer(3)]],
    device const float* decoderBias    [[buffer(4)]],
    constant NeuralQuantParams& params [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    const uint vectorIdx = tgp.x;
    if (vectorIdx >= params.numVectors) {
        return;
    }

    const uint inputDim  = params.inputDimension;
    const uint latentDim = params.latentDimension;

    threadgroup float4 tgLatent4[32];
    threadgroup float tgScale = 0.0f;

    if (tptg.x == 0 && tptg.y == 0) {
        tgScale = scales[vectorIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = tgScale;
    const uint latentDim4 = latentDim >> 2;

    if (latentDim4 > 32) {
        return;
    }

    // With 256 threads, first 32 threads load the latent
    if (tptg.y < latentDim4) {
        const uint codeOffset = vectorIdx * latentDim + tptg.y * 4;
        const char4 c = *((device const char4*)(latentCodes + codeOffset));
        tgLatent4[tptg.y] = float4(c) * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint outIdx = tgp.y * 256 + tptg.y;

    if (outIdx >= inputDim) {
        return;
    }

    device const float4* w4 = (device const float4*)(decoderWeights + outIdx * latentDim);
    float sum = decoderBias ? decoderBias[outIdx] : 0.0f;

    if (latentDim == 128) {
        #pragma unroll
        for (uint i4 = 0; i4 < 32; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else if (latentDim == 64) {
        #pragma unroll
        for (uint i4 = 0; i4 < 16; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    } else {
        for (uint i4 = 0; i4 < latentDim4; ++i4) {
            sum += dot(tgLatent4[i4], w4[i4]);
        }
    }

    outputVectors[vectorIdx * inputDim + outIdx] = sum;
}

// MARK: - Vectorized Transposed Decode

/// Vectorized transposed decode: processes 4 outputs per thread with float4 operations.
///
/// Key optimizations over the original transposed kernel:
/// 1. Each thread computes 4 adjacent output dimensions (4x work per thread)
/// 2. Weight loads are float4 and coalesced (adjacent threads access adjacent memory)
/// 3. float4 FMA operations provide 4x compute throughput vs scalar
/// 4. Same threadgroup memory reuse for dequantized latent codes
///
/// Performance comparison (128-dim latent):
/// - Original transposed: 128 scalar iterations, 128 scalar loads
/// - This kernel: 32 iterations, 128 float4 loads (4x fewer iterations, same memory)
///
/// Memory access pattern for 4 outputs at thread t:
///   outBase = tgp.y * 128 + t * 4
///   For latent[i]: loads weights[i*inputDim + outBase : +4] as float4
///   Adjacent threads (t, t+1) load adjacent float4s = coalesced!
///
/// Grid dispatch: threadgroups = (numVectors, ceil(inputDim/128), 1)
///                threadsPerThreadgroup = (1, 32, 1)
kernel void neural_dequantize_decode_2d_transposed_v2_kernel(
    device const char*  latentCodes     [[buffer(0)]],  // [N, latentDim] int8
    device const float* scales          [[buffer(1)]],  // [N]
    device const float* decoderWeightsT [[buffer(2)]],  // [latentDim, inputDim] TRANSPOSED
    device float*       outputVectors   [[buffer(3)]],  // [N, inputDim]
    device const float* decoderBias     [[buffer(4)]],  // [inputDim] or null
    constant NeuralQuantParams& params  [[buffer(5)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgp  [[threadgroup_position_in_grid]]
) {
    const uint vectorIdx = tgp.x;
    if (vectorIdx >= params.numVectors) {
        return;
    }

    const uint inputDim  = params.inputDimension;
    const uint latentDim = params.latentDimension;

    // Threadgroup cache for dequantized latent codes
    // Using float4 storage for efficient loading
    threadgroup float4 tgLatent4[32];  // Supports up to 128-dim latent
    threadgroup float tgScale;

    // ========== Phase 1: Cooperative Latent Loading ==========

    // Load scale (single thread)
    if (tptg.y == 0) {
        tgScale = scales[vectorIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = tgScale;
    const uint latentDim4 = (latentDim + 3) >> 2;  // Ceil division by 4

    // Each of first 32 threads loads one float4 of latent codes
    if (tptg.y < latentDim4) {
        const uint codeOffset = vectorIdx * latentDim + tptg.y * 4;
        // Handle potential out-of-bounds for non-multiple-of-4 latent dims
        if (tptg.y * 4 + 3 < latentDim) {
            const char4 c = *((device const char4*)(latentCodes + codeOffset));
            tgLatent4[tptg.y] = float4(c) * scale;
        } else {
            // Partial load for last chunk if latentDim not multiple of 4
            float4 partial = float4(0.0f);
            for (uint j = 0; j < 4 && tptg.y * 4 + j < latentDim; ++j) {
                partial[j] = float(latentCodes[codeOffset + j]) * scale;
            }
            tgLatent4[tptg.y] = partial;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 2: Compute 4 Outputs Per Thread ==========

    // Each thread computes 4 adjacent outputs
    // 32 threads × 4 outputs = 128 outputs per threadgroup
    const uint outBase = tgp.y * 128 + tptg.y * 4;

    // Early exit if all 4 outputs are out of bounds
    if (outBase >= inputDim) {
        return;
    }

    // Determine how many outputs this thread actually computes (1-4)
    const uint numOutputs = min(4u, inputDim - outBase);

    // Initialize accumulators with bias
    float4 acc;
    if (decoderBias) {
        // Safe load of bias values
        acc.x = (numOutputs > 0) ? decoderBias[outBase + 0] : 0.0f;
        acc.y = (numOutputs > 1) ? decoderBias[outBase + 1] : 0.0f;
        acc.z = (numOutputs > 2) ? decoderBias[outBase + 2] : 0.0f;
        acc.w = (numOutputs > 3) ? decoderBias[outBase + 3] : 0.0f;
    } else {
        acc = float4(0.0f);
    }

    // ========== Phase 3: Vectorized Matrix-Vector Product ==========
    //
    // Process 4 latent dimensions per iteration using float4 operations.
    // For 128-dim latent: 32 iterations (vs 128 in scalar kernel = 4x fewer!)
    //
    // Memory access pattern: Adjacent threads load adjacent float4s = coalesced!

    if (latentDim == 128) {
        // Fully unrolled for 128-dim (common case)
        #pragma unroll
        for (uint i4 = 0; i4 < 32; ++i4) {
            const uint i = i4 * 4;

            // Load 4 latent values (broadcast to all 4 outputs)
            const float4 lat = tgLatent4[i4];

            // Load 4×4 weights as 4 float4s (COALESCED!)
            // Adjacent threads load adjacent float4s in memory
            const float4 w0 = *((device const float4*)(decoderWeightsT + (i + 0) * inputDim + outBase));
            const float4 w1 = *((device const float4*)(decoderWeightsT + (i + 1) * inputDim + outBase));
            const float4 w2 = *((device const float4*)(decoderWeightsT + (i + 2) * inputDim + outBase));
            const float4 w3 = *((device const float4*)(decoderWeightsT + (i + 3) * inputDim + outBase));

            // 4 float4 FMAs: each latent contributes to all 4 outputs
            // acc[j] += lat.x*w0[j] + lat.y*w1[j] + lat.z*w2[j] + lat.w*w3[j]
            acc = fma(float4(lat.x), w0, acc);
            acc = fma(float4(lat.y), w1, acc);
            acc = fma(float4(lat.z), w2, acc);
            acc = fma(float4(lat.w), w3, acc);
        }
    } else if (latentDim == 64) {
        // Fully unrolled for 64-dim
        #pragma unroll
        for (uint i4 = 0; i4 < 16; ++i4) {
            const uint i = i4 * 4;
            const float4 lat = tgLatent4[i4];

            const float4 w0 = *((device const float4*)(decoderWeightsT + (i + 0) * inputDim + outBase));
            const float4 w1 = *((device const float4*)(decoderWeightsT + (i + 1) * inputDim + outBase));
            const float4 w2 = *((device const float4*)(decoderWeightsT + (i + 2) * inputDim + outBase));
            const float4 w3 = *((device const float4*)(decoderWeightsT + (i + 3) * inputDim + outBase));

            acc = fma(float4(lat.x), w0, acc);
            acc = fma(float4(lat.y), w1, acc);
            acc = fma(float4(lat.z), w2, acc);
            acc = fma(float4(lat.w), w3, acc);
        }
    } else {
        // Generic path for other latent dimensions
        for (uint i4 = 0; i4 < latentDim4; ++i4) {
            const uint i = i4 * 4;
            const float4 lat = tgLatent4[i4];

            // Handle last iteration if latentDim not multiple of 4
            const uint validLatent = min(4u, latentDim - i);

            if (validLatent >= 1) {
                const float4 w0 = *((device const float4*)(decoderWeightsT + (i + 0) * inputDim + outBase));
                acc = fma(float4(lat.x), w0, acc);
            }
            if (validLatent >= 2) {
                const float4 w1 = *((device const float4*)(decoderWeightsT + (i + 1) * inputDim + outBase));
                acc = fma(float4(lat.y), w1, acc);
            }
            if (validLatent >= 3) {
                const float4 w2 = *((device const float4*)(decoderWeightsT + (i + 2) * inputDim + outBase));
                acc = fma(float4(lat.z), w2, acc);
            }
            if (validLatent >= 4) {
                const float4 w3 = *((device const float4*)(decoderWeightsT + (i + 3) * inputDim + outBase));
                acc = fma(float4(lat.w), w3, acc);
            }
        }
    }

    // ========== Phase 4: Write Outputs ==========

    // Write outputs (handle boundary for non-multiple-of-4 inputDim)
    device float* outPtr = outputVectors + vectorIdx * inputDim + outBase;
    if (numOutputs == 4) {
        // Fast path: write all 4 as float4
        *((device float4*)outPtr) = acc;
    } else {
        // Boundary: write individual floats
        if (numOutputs > 0) outPtr[0] = acc.x;
        if (numOutputs > 1) outPtr[1] = acc.y;
        if (numOutputs > 2) outPtr[2] = acc.z;
    }
}
