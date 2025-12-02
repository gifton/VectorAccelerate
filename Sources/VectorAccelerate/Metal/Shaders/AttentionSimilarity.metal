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
    device const float* weightRow = weights + (dimIdx * inputDim);

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

    device const float* query = queries + (queryIdx * params.strideQuery);
    device const float* key = keys + (keyIdx * params.strideKey);

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
    similarities[queryIdx * params.strideOutput + keyIdx] = similarity;
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

    device const float* query = queries + (queryIdx * params.strideQuery);
    device const float* key = keys + (keyIdx * params.strideKey);

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
        device const float* wq = queryProjection + (head * headDim * inputDim);
        device const float* wk = keyProjection + (head * headDim * inputDim);

        // Project query and key for this head
        for (uint j = 0; j < effectiveHeadDim; ++j) {
            device const float* wqRow = wq + (j * inputDim);
            device const float* wkRow = wk + (j * inputDim);

            const uint simd_blocks = inputDim / 4;
            device const float4* q4 = (device const float4*)query;
            device const float4* k4 = (device const float4*)key;
            device const float4* wq4 = (device const float4*)wqRow;
            device const float4* wk4 = (device const float4*)wkRow;

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

    similarities[queryIdx * params.strideOutput + keyIdx] = similarity;
}

// MARK: - Optimized Kernel for 768-dim Input, 64-dim Head

/// Optimized attention similarity for transformer embeddings (768 -> 64).
kernel void attention_similarity_768_to_64_kernel(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* queryProjection [[buffer(2)]],
    device const float* keyProjection [[buffer(3)]],
    device float* similarities [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint keyIdx = tid.y;

    if (queryIdx >= params.numQueries || keyIdx >= params.numKeys) {
        return;
    }

    constexpr uint INPUT_DIM = 768;
    constexpr uint HEAD_DIM = 64;
    constexpr uint INPUT_BLOCKS = INPUT_DIM / 4;

    device const float* query = queries + (queryIdx * INPUT_DIM);
    device const float* key = keys + (keyIdx * INPUT_DIM);

    device const float4* q4 = (device const float4*)query;
    device const float4* k4 = (device const float4*)key;

    // Compute similarity directly without storing projected vectors
    float similarity = 0.0f;

    // Process each head dimension
    for (uint j = 0; j < HEAD_DIM; j += 4) {
        device const float4* wq0 = (device const float4*)(queryProjection + (j+0) * INPUT_DIM);
        device const float4* wq1 = (device const float4*)(queryProjection + (j+1) * INPUT_DIM);
        device const float4* wq2 = (device const float4*)(queryProjection + (j+2) * INPUT_DIM);
        device const float4* wq3 = (device const float4*)(queryProjection + (j+3) * INPUT_DIM);

        device const float4* wk0 = (device const float4*)(keyProjection + (j+0) * INPUT_DIM);
        device const float4* wk1 = (device const float4*)(keyProjection + (j+1) * INPUT_DIM);
        device const float4* wk2 = (device const float4*)(keyProjection + (j+2) * INPUT_DIM);
        device const float4* wk3 = (device const float4*)(keyProjection + (j+3) * INPUT_DIM);

        float4 accQ0 = float4(0.0f), accQ1 = float4(0.0f);
        float4 accQ2 = float4(0.0f), accQ3 = float4(0.0f);
        float4 accK0 = float4(0.0f), accK1 = float4(0.0f);
        float4 accK2 = float4(0.0f), accK3 = float4(0.0f);

        for (uint i = 0; i < INPUT_BLOCKS; ++i) {
            float4 q = q4[i];
            float4 k = k4[i];

            accQ0 = fma(q, wq0[i], accQ0);
            accQ1 = fma(q, wq1[i], accQ1);
            accQ2 = fma(q, wq2[i], accQ2);
            accQ3 = fma(q, wq3[i], accQ3);

            accK0 = fma(k, wk0[i], accK0);
            accK1 = fma(k, wk1[i], accK1);
            accK2 = fma(k, wk2[i], accK2);
            accK3 = fma(k, wk3[i], accK3);
        }

        float pQ0 = accQ0.x + accQ0.y + accQ0.z + accQ0.w;
        float pQ1 = accQ1.x + accQ1.y + accQ1.z + accQ1.w;
        float pQ2 = accQ2.x + accQ2.y + accQ2.z + accQ2.w;
        float pQ3 = accQ3.x + accQ3.y + accQ3.z + accQ3.w;

        float pK0 = accK0.x + accK0.y + accK0.z + accK0.w;
        float pK1 = accK1.x + accK1.y + accK1.z + accK1.w;
        float pK2 = accK2.x + accK2.y + accK2.z + accK2.w;
        float pK3 = accK3.x + accK3.y + accK3.z + accK3.w;

        similarity = fma(pQ0, pK0, similarity);
        similarity = fma(pQ1, pK1, similarity);
        similarity = fma(pQ2, pK2, similarity);
        similarity = fma(pQ3, pK3, similarity);
    }

    // Scale by temperature
    similarity /= params.temperature;

    if (params.normalizeSimilarities) {
        similarity = attn_sigmoid(similarity);
    }

    similarities[queryIdx * params.strideOutput + keyIdx] = similarity;
}

// MARK: - Batch Attention for Multiple Queries Against Same Keys

/// Compute attention similarity for a batch of queries against shared keys.
/// Optimized for the common case of searching a database.
///
/// Grid dispatch: (numQueries, 1, 1) - each thread handles one query against all keys
kernel void batch_attention_similarity_kernel(
    device const float* queries [[buffer(0)]],          // [N, D]
    device const float* keys [[buffer(1)]],             // [M, D]
    device const float* queryProjection [[buffer(2)]],  // [H, D]
    device const float* keyProjection [[buffer(3)]],    // [H, D]
    device float* similarities [[buffer(4)]],           // [N, M]
    constant AttentionParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numQueries) {
        return;
    }

    const uint queryIdx = tid;
    device const float* query = queries + (queryIdx * params.strideQuery);

    const uint inputDim = params.inputDimension;
    const uint headDim = params.headDimension;
    const uint numKeys = params.numKeys;

    // Project query once
    float projQuery[256];
    const uint effectiveHeadDim = min(headDim, 256u);
    attn_projectVector(query, queryProjection, projQuery, inputDim, effectiveHeadDim);

    // Thread-local storage for projected key
    float projKey[256];

    // Compute similarity against all keys
    for (uint keyIdx = 0; keyIdx < numKeys; ++keyIdx) {
        device const float* key = keys + (keyIdx * params.strideKey);

        // Project key
        attn_projectVector(key, keyProjection, projKey, inputDim, effectiveHeadDim);

        // Compute scaled dot product
        float similarity = attn_dotProduct(projQuery, projKey, effectiveHeadDim);
        similarity /= params.temperature;

        if (params.normalizeSimilarities) {
            similarity = attn_sigmoid(similarity);
        }

        similarities[queryIdx * params.strideOutput + keyIdx] = similarity;
    }
}
