// VectorAccelerate: Cosine Similarity Computation Kernels
//
// High-performance GPU kernels for cosine similarity/distance computation
// Optimized for dimensions 512, 768, and 1536
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameters Structure (Spec Section 2)

// Parameter structure for kernel configuration
struct CosineSimilarityParams {
    uint32_t numQueries;        // Number of query vectors (N)
    uint32_t numDatabase;       // Number of database vectors (M)
    uint32_t dimension;         // Vector dimension (D)
    uint32_t strideQuery;       // Stride between query vectors
    uint32_t strideDatabase;    // Stride between database vectors
    uint32_t strideOutput;      // Stride for output matrix
    uint8_t  outputDistance;    // 0 = similarity, 1 = distance (1 - similarity)
    uint8_t  inputsNormalized;  // 0 = need normalization, 1 = pre-normalized
    uint8_t  padding[2];        // Alignment padding
};

// Define a small constant for numerical stability checks
constant float EPSILON = 1e-8f;

// MARK: - Helper Functions

// Helper function to calculate the final result from dot product and norms
inline float calculate_similarity(float dotProduct, float queryNormSq, float databaseNormSq, bool outputDistance) {
    // Compute cosine similarity with numerical stability
    float denominator = sqrt(queryNormSq * databaseNormSq);

    float similarity;
    if (denominator > EPSILON) {
        similarity = dotProduct / denominator;
        // Clamp to handle potential floating-point inaccuracies
        similarity = clamp(similarity, -1.0f, 1.0f);
    } else {
        // Handle case where one or both vectors are zero vectors
        similarity = 0.0f;
    }

    return outputDistance ? (1.0f - similarity) : similarity;
}

// Helper function for normalized inputs (dot product only)
inline float calculate_similarity_normalized(float dotProduct, bool outputDistance) {
    // Clamp to handle potential floating-point inaccuracies, assuming norms are 1.
    float similarity = clamp(dotProduct, -1.0f, 1.0f);
    return outputDistance ? (1.0f - similarity) : similarity;
}


// MARK: - General Kernels (Spec Section 3.1, 3.2)

// 3.1 Fast path: Pre-normalized vectors (Dot Product). Handles arbitrary dimensions/strides.
kernel void cosine_similarity_normalized_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    // Bounds checking (required for dispatchThreads)
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Use float4 accumulator for better ILP and vectorization
    float4 dot_acc = float4(0.0f);
    const uint dimension = params.dimension;
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

    // Process 4 elements at a time
    for (uint i = 0; i < simd_blocks; ++i) {
        // Use explicit FMA (Fused Multiply-Add)
        dot_acc = fma(query4[i], database4[i], dot_acc);
    }

    // Horizontal reduction
    float dotProduct = dot_acc.x + dot_acc.y + dot_acc.z + dot_acc.w;

    // Handle remaining elements
    if (remainder > 0) {
        device const float* query_tail = query + (simd_blocks * 4);
        device const float* database_tail = database + (simd_blocks * 4);

        for (uint i = 0; i < remainder; ++i) {
            dotProduct = fma(query_tail[i], database_tail[i], dotProduct);
        }
    }

    float result = calculate_similarity_normalized(dotProduct, params.outputDistance);
    similarities[queryIdx * params.strideOutput + dbIdx] = result;
}

// 3.2 General path: Non-normalized vectors. Handles arbitrary dimensions/strides.
kernel void cosine_similarity_general_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Accumulators for dot product and squared norms
    float4 dot_acc = float4(0.0f);
    float4 qnorm_acc = float4(0.0f);
    float4 dnorm_acc = float4(0.0f);

    const uint dimension = params.dimension;
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

    // Process 4 elements at a time, computing all three metrics simultaneously
    for (uint i = 0; i < simd_blocks; ++i) {
        float4 q = query4[i];
        float4 d = database4[i];

        dot_acc = fma(q, d, dot_acc);
        qnorm_acc = fma(q, q, qnorm_acc);
        dnorm_acc = fma(d, d, dnorm_acc);
    }

    // Horizontal reduction
    float dotProduct = dot_acc.x + dot_acc.y + dot_acc.z + dot_acc.w;
    float queryNormSq = qnorm_acc.x + qnorm_acc.y + qnorm_acc.z + qnorm_acc.w;
    float databaseNormSq = dnorm_acc.x + dnorm_acc.y + dnorm_acc.z + dnorm_acc.w;

    // Handle remaining elements
    if (remainder > 0) {
        device const float* query_tail = query + (simd_blocks * 4);
        device const float* database_tail = database + (simd_blocks * 4);

        for (uint i = 0; i < remainder; ++i) {
            float q = query_tail[i];
            float d = database_tail[i];

            dotProduct = fma(q, d, dotProduct);
            queryNormSq = fma(q, q, queryNormSq);
            databaseNormSq = fma(d, d, databaseNormSq);
        }
    }

    float result = calculate_similarity(dotProduct, queryNormSq, databaseNormSq, params.outputDistance);
    similarities[queryIdx * params.strideOutput + dbIdx] = result;
}

// MARK: - Optimized Kernels (Spec Section 3.3)
// These kernels assume dense packing (stride == dimension) and use ILP optimization.

// Optimized for D=384 (96 float4 ops).
// Critical for MiniLM and Sentence-BERT embeddings (VectorCore 0.1.5 Vector384Optimized)
kernel void cosine_similarity_384_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Hardcoded strides for optimized address calculation
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 384);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 384);

    // Unroll by 8. Use 2 interleaved accumulators for ILP.
    float4 dot_acc0 = float4(0.0f);
    float4 dot_acc1 = float4(0.0f);

    if (params.inputsNormalized) {
        // Fast Path (Normalized)
        for (uint i = 0; i < 96; i += 8) {
            // Interleaved FMA accumulation
            dot_acc0 = fma(query4[i+0], database4[i+0], dot_acc0);
            dot_acc1 = fma(query4[i+1], database4[i+1], dot_acc1);
            dot_acc0 = fma(query4[i+2], database4[i+2], dot_acc0);
            dot_acc1 = fma(query4[i+3], database4[i+3], dot_acc1);
            dot_acc0 = fma(query4[i+4], database4[i+4], dot_acc0);
            dot_acc1 = fma(query4[i+5], database4[i+5], dot_acc1);
            dot_acc0 = fma(query4[i+6], database4[i+6], dot_acc0);
            dot_acc1 = fma(query4[i+7], database4[i+7], dot_acc1);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1;
        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;

        float result = calculate_similarity_normalized(dotProduct, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;

    } else {
        // General Path (Non-Normalized)
        float4 qnorm_acc0 = float4(0.0f);
        float4 qnorm_acc1 = float4(0.0f);
        float4 dnorm_acc0 = float4(0.0f);
        float4 dnorm_acc1 = float4(0.0f);

        for (uint i = 0; i < 96; i += 8) {
            // Load data
            float4 q0=query4[i+0]; float4 d0=database4[i+0];
            float4 q1=query4[i+1]; float4 d1=database4[i+1];
            float4 q2=query4[i+2]; float4 d2=database4[i+2];
            float4 q3=query4[i+3]; float4 d3=database4[i+3];
            float4 q4=query4[i+4]; float4 d4=database4[i+4];
            float4 q5=query4[i+5]; float4 d5=database4[i+5];
            float4 q6=query4[i+6]; float4 d6=database4[i+6];
            float4 q7=query4[i+7]; float4 d7=database4[i+7];

            // Interleaved FMA accumulation
            dot_acc0 = fma(q0, d0, dot_acc0); qnorm_acc0 = fma(q0, q0, qnorm_acc0); dnorm_acc0 = fma(d0, d0, dnorm_acc0);
            dot_acc1 = fma(q1, d1, dot_acc1); qnorm_acc1 = fma(q1, q1, qnorm_acc1); dnorm_acc1 = fma(d1, d1, dnorm_acc1);
            dot_acc0 = fma(q2, d2, dot_acc0); qnorm_acc0 = fma(q2, q2, qnorm_acc0); dnorm_acc0 = fma(d2, d2, dnorm_acc0);
            dot_acc1 = fma(q3, d3, dot_acc1); qnorm_acc1 = fma(q3, q3, qnorm_acc1); dnorm_acc1 = fma(d3, d3, dnorm_acc1);
            dot_acc0 = fma(q4, d4, dot_acc0); qnorm_acc0 = fma(q4, q4, qnorm_acc0); dnorm_acc0 = fma(d4, d4, dnorm_acc0);
            dot_acc1 = fma(q5, d5, dot_acc1); qnorm_acc1 = fma(q5, q5, qnorm_acc1); dnorm_acc1 = fma(d5, d5, dnorm_acc1);
            dot_acc0 = fma(q6, d6, dot_acc0); qnorm_acc0 = fma(q6, q6, qnorm_acc0); dnorm_acc0 = fma(d6, d6, dnorm_acc0);
            dot_acc1 = fma(q7, d7, dot_acc1); qnorm_acc1 = fma(q7, q7, qnorm_acc1); dnorm_acc1 = fma(d7, d7, dnorm_acc1);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1;
        float4 total_qnorm = qnorm_acc0 + qnorm_acc1;
        float4 total_dnorm = dnorm_acc0 + dnorm_acc1;

        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;
        float queryNormSq = total_qnorm.x + total_qnorm.y + total_qnorm.z + total_qnorm.w;
        float databaseNormSq = total_dnorm.x + total_dnorm.y + total_dnorm.z + total_dnorm.w;

        float result = calculate_similarity(dotProduct, queryNormSq, databaseNormSq, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;
    }
}

// Optimized for D=512 (128 float4 ops).
kernel void cosine_similarity_512_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Hardcoded strides for optimized address calculation
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 512);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 512);

    // Unroll by 8. Use 2 interleaved accumulators for ILP.
    float4 dot_acc0 = float4(0.0f);
    float4 dot_acc1 = float4(0.0f);

    if (params.inputsNormalized) {
        // Fast Path (Normalized)
        for (uint i = 0; i < 128; i += 8) {
            // Interleaved FMA accumulation
            dot_acc0 = fma(query4[i+0], database4[i+0], dot_acc0);
            dot_acc1 = fma(query4[i+1], database4[i+1], dot_acc1);
            dot_acc0 = fma(query4[i+2], database4[i+2], dot_acc0);
            dot_acc1 = fma(query4[i+3], database4[i+3], dot_acc1);
            dot_acc0 = fma(query4[i+4], database4[i+4], dot_acc0);
            dot_acc1 = fma(query4[i+5], database4[i+5], dot_acc1);
            dot_acc0 = fma(query4[i+6], database4[i+6], dot_acc0);
            dot_acc1 = fma(query4[i+7], database4[i+7], dot_acc1);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1;
        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;

        float result = calculate_similarity_normalized(dotProduct, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;

    } else {
        // General Path (Non-Normalized)
        float4 qnorm_acc0 = float4(0.0f);
        float4 qnorm_acc1 = float4(0.0f);
        float4 dnorm_acc0 = float4(0.0f);
        float4 dnorm_acc1 = float4(0.0f);

        for (uint i = 0; i < 128; i += 8) {
            // Load data
            float4 q0=query4[i+0]; float4 d0=database4[i+0];
            float4 q1=query4[i+1]; float4 d1=database4[i+1];
            float4 q2=query4[i+2]; float4 d2=database4[i+2];
            float4 q3=query4[i+3]; float4 d3=database4[i+3];
            float4 q4=query4[i+4]; float4 d4=database4[i+4];
            float4 q5=query4[i+5]; float4 d5=database4[i+5];
            float4 q6=query4[i+6]; float4 d6=database4[i+6];
            float4 q7=query4[i+7]; float4 d7=database4[i+7];

            // Interleaved FMA accumulation
            dot_acc0 = fma(q0, d0, dot_acc0); qnorm_acc0 = fma(q0, q0, qnorm_acc0); dnorm_acc0 = fma(d0, d0, dnorm_acc0);
            dot_acc1 = fma(q1, d1, dot_acc1); qnorm_acc1 = fma(q1, q1, qnorm_acc1); dnorm_acc1 = fma(d1, d1, dnorm_acc1);
            dot_acc0 = fma(q2, d2, dot_acc0); qnorm_acc0 = fma(q2, q2, qnorm_acc0); dnorm_acc0 = fma(d2, d2, dnorm_acc0);
            dot_acc1 = fma(q3, d3, dot_acc1); qnorm_acc1 = fma(q3, q3, qnorm_acc1); dnorm_acc1 = fma(d3, d3, dnorm_acc1);
            dot_acc0 = fma(q4, d4, dot_acc0); qnorm_acc0 = fma(q4, q4, qnorm_acc0); dnorm_acc0 = fma(d4, d4, dnorm_acc0);
            dot_acc1 = fma(q5, d5, dot_acc1); qnorm_acc1 = fma(q5, q5, qnorm_acc1); dnorm_acc1 = fma(d5, d5, dnorm_acc1);
            dot_acc0 = fma(q6, d6, dot_acc0); qnorm_acc0 = fma(q6, q6, qnorm_acc0); dnorm_acc0 = fma(d6, d6, dnorm_acc0);
            dot_acc1 = fma(q7, d7, dot_acc1); qnorm_acc1 = fma(q7, q7, qnorm_acc1); dnorm_acc1 = fma(d7, d7, dnorm_acc1);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1;
        float4 total_qnorm = qnorm_acc0 + qnorm_acc1;
        float4 total_dnorm = dnorm_acc0 + dnorm_acc1;

        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;
        float queryNormSq = total_qnorm.x + total_qnorm.y + total_qnorm.z + total_qnorm.w;
        float databaseNormSq = total_dnorm.x + total_dnorm.y + total_dnorm.z + total_dnorm.w;

        float result = calculate_similarity(dotProduct, queryNormSq, databaseNormSq, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;
    }
}

// Optimized for D=768 (192 float4 ops).
kernel void cosine_similarity_768_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 768);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 768);

    // Unroll by 12. Use 3 interleaved accumulators.
    float4 dot_acc0 = float4(0.0f);
    float4 dot_acc1 = float4(0.0f);
    float4 dot_acc2 = float4(0.0f);

    if (params.inputsNormalized) {
        // Fast Path (Normalized)
        for (uint i = 0; i < 192; i += 12) {
            dot_acc0 = fma(query4[i+0], database4[i+0], dot_acc0);
            dot_acc1 = fma(query4[i+1], database4[i+1], dot_acc1);
            dot_acc2 = fma(query4[i+2], database4[i+2], dot_acc2);
            dot_acc0 = fma(query4[i+3], database4[i+3], dot_acc0);
            dot_acc1 = fma(query4[i+4], database4[i+4], dot_acc1);
            dot_acc2 = fma(query4[i+5], database4[i+5], dot_acc2);
            dot_acc0 = fma(query4[i+6], database4[i+6], dot_acc0);
            dot_acc1 = fma(query4[i+7], database4[i+7], dot_acc1);
            dot_acc2 = fma(query4[i+8], database4[i+8], dot_acc2);
            dot_acc0 = fma(query4[i+9], database4[i+9], dot_acc0);
            dot_acc1 = fma(query4[i+10], database4[i+10], dot_acc1);
            dot_acc2 = fma(query4[i+11], database4[i+11], dot_acc2);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1 + dot_acc2;
        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;

        float result = calculate_similarity_normalized(dotProduct, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;

    } else {
        // General Path (Non-Normalized)
        float4 qnorm_acc0=0, qnorm_acc1=0, qnorm_acc2=0;
        float4 dnorm_acc0=0, dnorm_acc1=0, dnorm_acc2=0;

        for (uint i = 0; i < 192; i += 12) {
            // Load data (12 blocks)
            float4 q0=query4[i+0]; float4 d0=database4[i+0];
            float4 q1=query4[i+1]; float4 d1=database4[i+1];
            float4 q2=query4[i+2]; float4 d2=database4[i+2];
            float4 q3=query4[i+3]; float4 d3=database4[i+3];
            float4 q4=query4[i+4]; float4 d4=database4[i+4];
            float4 q5=query4[i+5]; float4 d5=database4[i+5];
            float4 q6=query4[i+6]; float4 d6=database4[i+6];
            float4 q7=query4[i+7]; float4 d7=database4[i+7];
            float4 q8=query4[i+8]; float4 d8=database4[i+8];
            float4 q9=query4[i+9]; float4 d9=database4[i+9];
            float4 q10=query4[i+10]; float4 d10=database4[i+10];
            float4 q11=query4[i+11]; float4 d11=database4[i+11];

            // Interleaved FMA accumulation
            dot_acc0 = fma(q0, d0, dot_acc0); qnorm_acc0 = fma(q0, q0, qnorm_acc0); dnorm_acc0 = fma(d0, d0, dnorm_acc0);
            dot_acc1 = fma(q1, d1, dot_acc1); qnorm_acc1 = fma(q1, q1, qnorm_acc1); dnorm_acc1 = fma(d1, d1, dnorm_acc1);
            dot_acc2 = fma(q2, d2, dot_acc2); qnorm_acc2 = fma(q2, q2, qnorm_acc2); dnorm_acc2 = fma(d2, d2, dnorm_acc2);

            dot_acc0 = fma(q3, d3, dot_acc0); qnorm_acc0 = fma(q3, q3, qnorm_acc0); dnorm_acc0 = fma(d3, d3, dnorm_acc0);
            dot_acc1 = fma(q4, d4, dot_acc1); qnorm_acc1 = fma(q4, q4, qnorm_acc1); dnorm_acc1 = fma(d4, d4, dnorm_acc1);
            dot_acc2 = fma(q5, d5, dot_acc2); qnorm_acc2 = fma(q5, q5, qnorm_acc2); dnorm_acc2 = fma(d5, d5, dnorm_acc2);

            dot_acc0 = fma(q6, d6, dot_acc0); qnorm_acc0 = fma(q6, q6, qnorm_acc0); dnorm_acc0 = fma(d6, d6, dnorm_acc0);
            dot_acc1 = fma(q7, d7, dot_acc1); qnorm_acc1 = fma(q7, q7, qnorm_acc1); dnorm_acc1 = fma(d7, d7, dnorm_acc1);
            dot_acc2 = fma(q8, d8, dot_acc2); qnorm_acc2 = fma(q8, q8, qnorm_acc2); dnorm_acc2 = fma(d8, d8, dnorm_acc2);

            dot_acc0 = fma(q9, d9, dot_acc0); qnorm_acc0 = fma(q9, q9, qnorm_acc0); dnorm_acc0 = fma(d9, d9, dnorm_acc0);
            dot_acc1 = fma(q10, d10, dot_acc1); qnorm_acc1 = fma(q10, q10, qnorm_acc1); dnorm_acc1 = fma(d10, d10, dnorm_acc1);
            dot_acc2 = fma(q11, d11, dot_acc2); qnorm_acc2 = fma(q11, q11, qnorm_acc2); dnorm_acc2 = fma(d11, d11, dnorm_acc2);
        }

         // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1 + dot_acc2;
        float4 total_qnorm = qnorm_acc0 + qnorm_acc1 + qnorm_acc2;
        float4 total_dnorm = dnorm_acc0 + dnorm_acc1 + dnorm_acc2;

        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;
        float queryNormSq = total_qnorm.x + total_qnorm.y + total_qnorm.z + total_qnorm.w;
        float databaseNormSq = total_dnorm.x + total_dnorm.y + total_dnorm.z + total_dnorm.w;

        float result = calculate_similarity(dotProduct, queryNormSq, databaseNormSq, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;
    }
}

// Optimized for D=1536 (384 float4 ops).
kernel void cosine_similarity_1536_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 1536);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 1536);

    // Unroll by 16. Use 4 interleaved accumulators to maximize throughput.
    float4 dot_acc0=0, dot_acc1=0, dot_acc2=0, dot_acc3=0;

    if (params.inputsNormalized) {
        // Fast Path (Normalized)
        for (uint i = 0; i < 384; i += 16) {
             dot_acc0 = fma(query4[i+0], database4[i+0], dot_acc0);
             dot_acc1 = fma(query4[i+1], database4[i+1], dot_acc1);
             dot_acc2 = fma(query4[i+2], database4[i+2], dot_acc2);
             dot_acc3 = fma(query4[i+3], database4[i+3], dot_acc3);

             dot_acc0 = fma(query4[i+4], database4[i+4], dot_acc0);
             dot_acc1 = fma(query4[i+5], database4[i+5], dot_acc1);
             dot_acc2 = fma(query4[i+6], database4[i+6], dot_acc2);
             dot_acc3 = fma(query4[i+7], database4[i+7], dot_acc3);

             dot_acc0 = fma(query4[i+8], database4[i+8], dot_acc0);
             dot_acc1 = fma(query4[i+9], database4[i+9], dot_acc1);
             dot_acc2 = fma(query4[i+10], database4[i+10], dot_acc2);
             dot_acc3 = fma(query4[i+11], database4[i+11], dot_acc3);

             dot_acc0 = fma(query4[i+12], database4[i+12], dot_acc0);
             dot_acc1 = fma(query4[i+13], database4[i+13], dot_acc1);
             dot_acc2 = fma(query4[i+14], database4[i+14], dot_acc2);
             dot_acc3 = fma(query4[i+15], database4[i+15], dot_acc3);
        }
        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1 + dot_acc2 + dot_acc3;
        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;

        float result = calculate_similarity_normalized(dotProduct, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;

    } else {
        // General Path (Non-Normalized)
        float4 qnorm_acc0=0, qnorm_acc1=0, qnorm_acc2=0, qnorm_acc3=0;
        float4 dnorm_acc0=0, dnorm_acc1=0, dnorm_acc2=0, dnorm_acc3=0;

        for (uint i = 0; i < 384; i += 16) {
            // Load data (16 blocks)
            float4 q0=query4[i+0]; float4 d0=database4[i+0]; float4 q1=query4[i+1]; float4 d1=database4[i+1];
            float4 q2=query4[i+2]; float4 d2=database4[i+2]; float4 q3=query4[i+3]; float4 d3=database4[i+3];
            float4 q4=query4[i+4]; float4 d4=database4[i+4]; float4 q5=query4[i+5]; float4 d5=database4[i+5];
            float4 q6=query4[i+6]; float4 d6=database4[i+6]; float4 q7=query4[i+7]; float4 d7=database4[i+7];
            float4 q8=query4[i+8]; float4 d8=database4[i+8]; float4 q9=query4[i+9]; float4 d9=database4[i+9];
            float4 q10=query4[i+10]; float4 d10=database4[i+10]; float4 q11=query4[i+11]; float4 d11=database4[i+11];
            float4 q12=query4[i+12]; float4 d12=database4[i+12]; float4 q13=query4[i+13]; float4 d13=database4[i+13];
            float4 q14=query4[i+14]; float4 d14=database4[i+14]; float4 q15=query4[i+15]; float4 d15=database4[i+15];

            // Interleaved FMA accumulation
            dot_acc0 = fma(q0, d0, dot_acc0); qnorm_acc0 = fma(q0, q0, qnorm_acc0); dnorm_acc0 = fma(d0, d0, dnorm_acc0);
            dot_acc1 = fma(q1, d1, dot_acc1); qnorm_acc1 = fma(q1, q1, qnorm_acc1); dnorm_acc1 = fma(d1, d1, dnorm_acc1);
            dot_acc2 = fma(q2, d2, dot_acc2); qnorm_acc2 = fma(q2, q2, qnorm_acc2); dnorm_acc2 = fma(d2, d2, dnorm_acc2);
            dot_acc3 = fma(q3, d3, dot_acc3); qnorm_acc3 = fma(q3, q3, qnorm_acc3); dnorm_acc3 = fma(d3, d3, dnorm_acc3);

            dot_acc0 = fma(q4, d4, dot_acc0); qnorm_acc0 = fma(q4, q4, qnorm_acc0); dnorm_acc0 = fma(d4, d4, dnorm_acc0);
            dot_acc1 = fma(q5, d5, dot_acc1); qnorm_acc1 = fma(q5, q5, qnorm_acc1); dnorm_acc1 = fma(d5, d5, dnorm_acc1);
            dot_acc2 = fma(q6, d6, dot_acc2); qnorm_acc2 = fma(q6, q6, qnorm_acc2); dnorm_acc2 = fma(d6, d6, dnorm_acc2);
            dot_acc3 = fma(q7, d7, dot_acc3); qnorm_acc3 = fma(q7, q7, qnorm_acc3); dnorm_acc3 = fma(d7, d7, dnorm_acc3);

            dot_acc0 = fma(q8, d8, dot_acc0); qnorm_acc0 = fma(q8, q8, qnorm_acc0); dnorm_acc0 = fma(d8, d8, dnorm_acc0);
            dot_acc1 = fma(q9, d9, dot_acc1); qnorm_acc1 = fma(q9, q9, qnorm_acc1); dnorm_acc1 = fma(d9, d9, dnorm_acc1);
            dot_acc2 = fma(q10, d10, dot_acc2); qnorm_acc2 = fma(q10, q10, qnorm_acc2); dnorm_acc2 = fma(d10, d10, dnorm_acc2);
            dot_acc3 = fma(q11, d11, dot_acc3); qnorm_acc3 = fma(q11, q11, qnorm_acc3); dnorm_acc3 = fma(d11, d11, dnorm_acc3);

            dot_acc0 = fma(q12, d12, dot_acc0); qnorm_acc0 = fma(q12, q12, qnorm_acc0); dnorm_acc0 = fma(d12, d12, dnorm_acc0);
            dot_acc1 = fma(q13, d13, dot_acc1); qnorm_acc1 = fma(q13, q13, qnorm_acc1); dnorm_acc1 = fma(d13, d13, dnorm_acc1);
            dot_acc2 = fma(q14, d14, dot_acc2); qnorm_acc2 = fma(q14, q14, qnorm_acc2); dnorm_acc2 = fma(d14, d14, dnorm_acc2);
            dot_acc3 = fma(q15, d15, dot_acc3); qnorm_acc3 = fma(q15, q15, qnorm_acc3); dnorm_acc3 = fma(d15, d15, dnorm_acc3);
        }

        // Final reduction
        float4 total_dot = dot_acc0 + dot_acc1 + dot_acc2 + dot_acc3;
        float4 total_qnorm = qnorm_acc0 + qnorm_acc1 + qnorm_acc2 + qnorm_acc3;
        float4 total_dnorm = dnorm_acc0 + dnorm_acc1 + dnorm_acc2 + dnorm_acc3;

        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;
        float queryNormSq = total_qnorm.x + total_qnorm.y + total_qnorm.z + total_qnorm.w;
        float databaseNormSq = total_dnorm.x + total_dnorm.y + total_dnorm.z + total_dnorm.w;

        float result = calculate_similarity(dotProduct, queryNormSq, databaseNormSq, params.outputDistance);
        similarities[queryIdx * params.strideOutput + dbIdx] = result;
    }
}