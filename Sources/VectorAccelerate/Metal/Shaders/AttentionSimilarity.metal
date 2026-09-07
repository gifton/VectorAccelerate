// VectorAccelerate: Attention-based Similarity Kernels
//
// GPU kernels for computing similarity using scaled dot-product attention
// with learned query and key projections.
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Phase 4: ML Integration - Attention Similarity
//
// Architecture:
// - Query projection: Q = input @ Wq
// - Key projection: K = input @ Wk
// - Similarity: S = (Q @ K^T) / temperature
//
// Supports single-head and multi-head attention patterns.

#include "Metal4Common.h"

// MARK: - Parameter Structures

/// Parameters for attention similarity kernels
struct AttentionParams {
    uint32_t numQueries;           // Number of query vectors (N)
    uint32_t numKeys;              // Number of key vectors (M)
    uint32_t inputDimension;       // Input vector dimension (D)
    uint32_t headDimension;        // Head dimension (H)
    uint32_t numHeads;             // Number of attention heads
    uint32_t strideQuery;          // Stride between query vectors
    uint32_t strideKey;            // Stride between key vectors
    uint32_t strideOutput;         // Stride for output matrix (numKeys)
    float temperature;             // Scaling factor (typically sqrt(headDim))
    uint8_t normalizeSimilarities; // 1 = apply sigmoid to normalize to [0,1]
    uint8_t padding[3];            // Alignment padding
};

// MARK: - Helper Functions
// Note: Prefixed with attn_ to avoid symbol collisions when combined with other shaders

/// Project a vector through weight matrix for a single dimension
/// output = dot(input, weights[dimIdx])
inline float attn_projectDimension(
    device const float* input,
    device const float* weights,
    uint inputDim,
    uint dimIdx
) {
    device const float* weightRow = weights + ((ulong)dimIdx * inputDim);

    const uint simd_blocks = inputDim / 4;
    device const packed_float4* input4 = (device const packed_float4*)input;
    device const packed_float4* weight4 = (device const packed_float4*)weightRow;

    float4 acc = float4(0.0f);
    for (uint i = 0; i < simd_blocks; ++i) {
        acc = fma(input4[i], weight4[i], acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    for (uint i = simd_blocks * 4; i < inputDim; ++i) {
        sum = fma(input[i], weightRow[i], sum);
    }

    return sum;
}

/// Project entire vector through weight matrix
inline void attn_projectVector(
    device const float* input,
    device const float* weights,
    thread float* output,
    uint inputDim,
    uint outputDim
) {
    for (uint j = 0; j < outputDim; ++j) {
        output[j] = attn_projectDimension(input, weights, inputDim, j);
    }
}

/// Compute dot product of two thread-local vectors
inline float attn_dotProduct(thread float* a, thread float* b, uint dim) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        sum = fma(a[i], b[i], sum);
    }
    return sum;
}

/// Sigmoid function for normalization
inline float attn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// MARK: - Single-Head Attention Similarity Kernel

/// Compute attention-based similarity scores (single head).
///
/// For each (query, key) pair:
/// 1. Project query through Wq: q = query @ Wq
/// 2. Project key through Wk: k = key @ Wk
/// 3. Compute scaled dot product: similarity = (q · k) / temperature
///
/// Grid dispatch: (numQueries, numKeys, 1)
kernel void attention_similarity_kernel(
    device const float* queries [[buffer(0)]],          // [N, D]
    device const float* keys [[buffer(1)]],             // [M, D]
    device const float* queryProjection [[buffer(2)]],  // [H, D]
    device const float* keyProjection [[buffer(3)]],    // [H, D]
    device float* similarities [[buffer(4)]],           // [N, M]
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint keyIdx = tid.y;

    if (queryIdx >= params.numQueries || keyIdx >= params.numKeys) {
        return;
    }

    device const float* query = queries + ((ulong)queryIdx * params.strideQuery);
    device const float* key = keys + ((ulong)keyIdx * params.strideKey);

    const uint inputDim = params.inputDimension;
    const uint headDim = params.headDimension;

    // Thread-local storage for projected vectors (max 256 dims)
    float projQuery[256];
    float projKey[256];
    const uint effectiveHeadDim = min(headDim, 256u);

    // Project query and key
    attn_projectVector(query, queryProjection, projQuery, inputDim, effectiveHeadDim);
    attn_projectVector(key, keyProjection, projKey, inputDim, effectiveHeadDim);

    // Compute scaled dot product
    float similarity = attn_dotProduct(projQuery, projKey, effectiveHeadDim);
    similarity /= params.temperature;

    // Optional normalization to [0, 1]
    if (params.normalizeSimilarities) {
        similarity = attn_sigmoid(similarity);
    }

    // Store result
    similarities[(ulong)queryIdx * params.strideOutput + keyIdx] = similarity;
}

// MARK: - Multi-Head Attention Similarity Kernel

/// Compute attention-based similarity scores (multi-head).
///
/// For each (query, key) pair:
/// 1. For each head h:
///    - Project query: q_h = query @ Wq[h]
///    - Project key: k_h = key @ Wk[h]
///    - Compute: score_h = (q_h · k_h) / temperature
/// 2. Average across heads: similarity = mean(score_h)
///
/// Grid dispatch: (numQueries, numKeys, 1)
kernel void multihead_attention_similarity_kernel(
    device const float* queries [[buffer(0)]],          // [N, D]
    device const float* keys [[buffer(1)]],             // [M, D]
    device const float* queryProjection [[buffer(2)]],  // [numHeads * H, D]
    device const float* keyProjection [[buffer(3)]],    // [numHeads * H, D]
    device float* similarities [[buffer(4)]],           // [N, M]
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint keyIdx = tid.y;

    if (queryIdx >= params.numQueries || keyIdx >= params.numKeys) {
        return;
    }

    device const float* query = queries + ((ulong)queryIdx * params.strideQuery);
    device const float* key = keys + ((ulong)keyIdx * params.strideKey);

    const uint inputDim = params.inputDimension;
    const uint headDim = params.headDimension;
    const uint numHeads = params.numHeads;

    // Accumulate similarity across heads
    float totalSimilarity = 0.0f;

    // Thread-local storage for projected vectors
    float projQuery[64];  // Max head dimension for multi-head
    float projKey[64];
    const uint effectiveHeadDim = min(headDim, 64u);

    for (uint head = 0; head < numHeads; ++head) {
        // Get weight matrices for this head
        device const float* wq = queryProjection + ((ulong)head * headDim * inputDim);
        device const float* wk = keyProjection + ((ulong)head * headDim * inputDim);

        // Project query and key for this head
        for (uint j = 0; j < effectiveHeadDim; ++j) {
            device const float* wqRow = wq + ((ulong)j * inputDim);
            device const float* wkRow = wk + ((ulong)j * inputDim);

            const uint simd_blocks = inputDim / 4;
            device const packed_float4* q4 = (device const packed_float4*)query;
            device const packed_float4* k4 = (device const packed_float4*)key;
            device const packed_float4* wq4 = (device const packed_float4*)wqRow;
            device const packed_float4* wk4 = (device const packed_float4*)wkRow;

            float4 accQ = float4(0.0f);
            float4 accK = float4(0.0f);

            for (uint i = 0; i < simd_blocks; ++i) {
                accQ = fma(q4[i], wq4[i], accQ);
                accK = fma(k4[i], wk4[i], accK);
            }

            projQuery[j] = accQ.x + accQ.y + accQ.z + accQ.w;
            projKey[j] = accK.x + accK.y + accK.z + accK.w;

            for (uint i = simd_blocks * 4; i < inputDim; ++i) {
                projQuery[j] = fma(query[i], wqRow[i], projQuery[j]);
                projKey[j] = fma(key[i], wkRow[i], projKey[j]);
            }
        }

        // Compute dot product for this head
        float headSimilarity = attn_dotProduct(projQuery, projKey, effectiveHeadDim);
        totalSimilarity += headSimilarity;
    }

    // Average across heads and scale
    float similarity = (totalSimilarity / float(numHeads)) / params.temperature;

    // Optional normalization
    if (params.normalizeSimilarities) {
        similarity = attn_sigmoid(similarity);
    }

    similarities[(ulong)queryIdx * params.strideOutput + keyIdx] = similarity;
}

