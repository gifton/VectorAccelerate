// VectorAccelerate: Learned Distance Computation Kernels
//
// High-performance GPU kernels for learned distance metrics using projection matrices.
// These kernels project vectors through a learned transformation before computing distances.
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Phase 4: Experimental ML Integration
// - Learned projection matrices for dimensionality reduction
// - Maintains semantic structure while reducing computation
// - Common use cases:
//   - Matryoshka embeddings (768 -> 128)
//   - Learned metric spaces for domain-specific similarity
//   - Efficient approximate nearest neighbor search

#include "Metal4Common.h"

// MARK: - Parameters Structures

/// Parameters for learned distance kernel
struct LearnedDistanceParams {
    uint32_t numQueries;          // Number of query vectors (N)
    uint32_t numDatabase;         // Number of database vectors (M)
    uint32_t inputDimension;      // Input vector dimension (D_in)
    uint32_t projectedDimension;  // Output dimension after projection (D_out)
    uint32_t strideQuery;         // Stride between query vectors
    uint32_t strideDatabase;      // Stride between database vectors
    uint32_t strideOutput;        // Stride for output matrix
    uint8_t  computeSqrt;         // 0 = squared distance, 1 = apply sqrt
    uint8_t  normalizeProjected;  // 1 = L2 normalize after projection
    uint8_t  padding[2];          // Alignment padding
};

/// Parameters for batch projection kernel
struct ProjectionParams {
    uint32_t numVectors;          // Number of vectors to project
    uint32_t inputDimension;      // Input dimension
    uint32_t outputDimension;     // Output dimension after projection
    uint32_t stride;              // Stride between vectors
    uint8_t  normalize;           // 1 = L2 normalize output
    uint8_t  padding[3];          // Alignment padding
};

// MARK: - Helper Functions

/// Project a single vector through weight matrix (row-major: [D_out, D_in])
/// Output: projected[j] = sum_i(input[i] * weights[j * D_in + i])
inline void projectVector(
    device const float* input,
    device const float* weights,
    thread float* output,
    uint inputDim,
    uint outputDim
) {
    for (uint j = 0; j < outputDim; ++j) {
        float sum = 0.0f;
        device const float* weightRow = weights + (j * inputDim);

        // SIMD-friendly inner loop
        const uint simd_blocks = inputDim / 4;
//        const uint remainder = inputDim % 4;

        device const float4* input4 = (device const float4*)input;
        device const float4* weight4 = (device const float4*)weightRow;

        float4 acc = float4(0.0f);
        for (uint i = 0; i < simd_blocks; ++i) {
            acc = fma(input4[i], weight4[i], acc);
        }
        sum = acc.x + acc.y + acc.z + acc.w;

        // Handle remainder
        for (uint i = simd_blocks * 4; i < inputDim; ++i) {
            sum = fma(input[i], weightRow[i], sum);
        }

        output[j] = sum;
    }
}

/// Compute L2 norm of a thread-local array
inline float computeL2Norm(thread float* vec, uint dim) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        sum = fma(vec[i], vec[i], sum);
    }
    return sqrt(sum);
}

/// Normalize a thread-local array in-place
inline void normalizeInPlace(thread float* vec, uint dim) {
    float norm = computeL2Norm(vec, dim);
    float invNorm = (norm > VA_EPSILON) ? (1.0f / norm) : 0.0f;
    for (uint i = 0; i < dim; ++i) {
        vec[i] *= invNorm;
    }
}

/// Compute L2 distance squared between two thread-local arrays
inline float computeL2Squared(thread float* a, thread float* b, uint dim) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum = fma(diff, diff, sum);
    }
    return sum;
}

/// Compute dot product between two thread-local arrays
inline float computeDotProduct(thread float* a, thread float* b, uint dim) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        sum = fma(a[i], b[i], sum);
    }
    return sum;
}

// MARK: - Learned L2 Distance Kernel

/// Compute L2 distance in learned projected space.
///
/// Projects both query and database vectors through a learned weight matrix,
/// then computes L2 distance in the projected space. This enables:
/// - Dimensionality reduction (e.g., 768 -> 128)
/// - Learned metric adaptation for domain-specific similarity
///
/// Weight matrix layout: row-major [D_out, D_in]
/// - weights[j * D_in + i] is the weight for input[i] -> output[j]
///
/// Grid dispatch: (numQueries, numDatabase, 1)
kernel void learned_l2_distance_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device const float* projectionWeights [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant LearnedDistanceParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    // Bounds check
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Get input vector pointers
    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Thread-local storage for projected vectors
    // Note: For dimensions > 256, consider using threadgroup memory
    float projQuery[256];
    float projDb[256];

    const uint inputDim = params.inputDimension;
    const uint outputDim = min(params.projectedDimension, 256u);

    // Project both vectors
    projectVector(query, projectionWeights, projQuery, inputDim, outputDim);
    projectVector(database, projectionWeights, projDb, inputDim, outputDim);

    // Optionally normalize projected vectors
    if (params.normalizeProjected) {
        normalizeInPlace(projQuery, outputDim);
        normalizeInPlace(projDb, outputDim);
    }

    // Compute L2 distance in projected space
    float distSq = computeL2Squared(projQuery, projDb, outputDim);
    float distance = params.computeSqrt ? sqrt(distSq) : distSq;

    // Store result
    const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
    distances[outputIdx] = distance;
}

// MARK: - Learned Cosine Similarity Kernel

/// Compute cosine similarity in learned projected space.
///
/// Projects both vectors, normalizes them, then computes dot product.
/// This is equivalent to angular distance in the projected space.
///
/// Grid dispatch: (numQueries, numDatabase, 1)
kernel void learned_cosine_similarity_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device const float* projectionWeights [[buffer(2)]],
    device float* similarities [[buffer(3)]],
    constant LearnedDistanceParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    float projQuery[256];
    float projDb[256];

    const uint inputDim = params.inputDimension;
    const uint outputDim = min(params.projectedDimension, 256u);

    // Project both vectors
    projectVector(query, projectionWeights, projQuery, inputDim, outputDim);
    projectVector(database, projectionWeights, projDb, inputDim, outputDim);

    // Normalize for cosine similarity (always normalize, ignore params flag)
    normalizeInPlace(projQuery, outputDim);
    normalizeInPlace(projDb, outputDim);

    // Compute cosine similarity (dot product of normalized vectors)
    float similarity = computeDotProduct(projQuery, projDb, outputDim);

    // Clamp to [-1, 1] for numerical stability
    similarity = clamp(similarity, -1.0f, 1.0f);

    const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
    similarities[outputIdx] = similarity;
}

// MARK: - Batch Projection Kernel

/// Project multiple vectors through weight matrix in parallel.
///
/// Useful for pre-projecting a database of vectors for subsequent
/// fast distance computation in the projected space.
///
/// Grid dispatch: (numVectors, outputDimension, 1)
/// - tid.x: vector index
/// - tid.y: output dimension index
kernel void batch_projection_kernel(
    device const float* inputVectors [[buffer(0)]],
    device const float* projectionWeights [[buffer(1)]],
    device float* outputVectors [[buffer(2)]],
    constant ProjectionParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vectorIdx = tid.x;
    const uint outputDimIdx = tid.y;

    if (vectorIdx >= params.numVectors || outputDimIdx >= params.outputDimension) {
        return;
    }

    device const float* input = inputVectors + (vectorIdx * params.stride);
    device const float* weightRow = projectionWeights + (outputDimIdx * params.inputDimension);

    // Compute single output element: output[j] = dot(input, weights[j])
    const uint inputDim = params.inputDimension;
    const uint simd_blocks = inputDim / 4;
//    const uint remainder = inputDim % 4;

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

    // Store result
    device float* output = outputVectors + (vectorIdx * params.outputDimension);
    output[outputDimIdx] = sum;
}

/// Normalize projected vectors in-place.
/// Run after batch_projection_kernel if normalization is needed.
///
/// Grid dispatch: (numVectors, 1, 1)
kernel void batch_normalize_kernel(
    device float* vectors [[buffer(0)]],
    constant ProjectionParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numVectors) {
        return;
    }

    device float* vec = vectors + (tid * params.outputDimension);
    const uint dim = params.outputDimension;

    // Compute L2 norm
    float4 acc = float4(0.0f);
    const uint simd_blocks = dim / 4;
//    const uint remainder = dim % 4;

    device float4* vec4 = (device float4*)vec;
    for (uint i = 0; i < simd_blocks; ++i) {
        acc = fma(vec4[i], vec4[i], acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    for (uint i = simd_blocks * 4; i < dim; ++i) {
        sum = fma(vec[i], vec[i], sum);
    }

    float norm = sqrt(sum);
    float invNorm = (norm > VA_EPSILON) ? (1.0f / norm) : 0.0f;

    // Normalize in-place
    for (uint i = 0; i < simd_blocks; ++i) {
        vec4[i] *= invNorm;
    }
    for (uint i = simd_blocks * 4; i < dim; ++i) {
        vec[i] *= invNorm;
    }
}

// MARK: - Optimized Kernels for Common Dimensions

/// Optimized learned L2 for 768 -> 128 projection (common Matryoshka configuration)
kernel void learned_l2_768_to_128_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device const float* projectionWeights [[buffer(2)]],  // [128, 768]
    device float* distances [[buffer(3)]],
    constant LearnedDistanceParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Fixed dimensions for optimized path
    constexpr uint INPUT_DIM = 768;
    constexpr uint OUTPUT_DIM = 128;
    constexpr uint INPUT_BLOCKS = INPUT_DIM / 4;  // 192

    device const float* query = queryVectors + (queryIdx * INPUT_DIM);
    device const float* database = databaseVectors + (dbIdx * INPUT_DIM);

    device const float4* query4 = (device const float4*)query;
    device const float4* db4 = (device const float4*)database;

    // Project and compute distance simultaneously
    float distSq = 0.0f;

    for (uint j = 0; j < OUTPUT_DIM; ++j) {
        device const float4* weight4 = (device const float4*)(projectionWeights + j * INPUT_DIM);

        // Dual accumulation for query and database projection
        float4 accQ = float4(0.0f);
        float4 accD = float4(0.0f);

        for (uint i = 0; i < INPUT_BLOCKS; ++i) {
            accQ = fma(query4[i], weight4[i], accQ);
            accD = fma(db4[i], weight4[i], accD);
        }

        float projQ = accQ.x + accQ.y + accQ.z + accQ.w;
        float projD = accD.x + accD.y + accD.z + accD.w;

        float diff = projQ - projD;
        distSq = fma(diff, diff, distSq);
    }

    float distance = params.computeSqrt ? sqrt(distSq) : distSq;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}

/// Optimized learned L2 for 384 -> 64 projection (MiniLM to compact)
kernel void learned_l2_384_to_64_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device const float* projectionWeights [[buffer(2)]],  // [64, 384]
    device float* distances [[buffer(3)]],
    constant LearnedDistanceParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    constexpr uint INPUT_DIM = 384;
    constexpr uint OUTPUT_DIM = 64;
    constexpr uint INPUT_BLOCKS = INPUT_DIM / 4;  // 96

    device const float* query = queryVectors + (queryIdx * INPUT_DIM);
    device const float* database = databaseVectors + (dbIdx * INPUT_DIM);

    device const float4* query4 = (device const float4*)query;
    device const float4* db4 = (device const float4*)database;

    float distSq = 0.0f;

    // Unroll outer loop by 4 for better instruction-level parallelism
    for (uint j = 0; j < OUTPUT_DIM; j += 4) {
        device const float4* w0 = (device const float4*)(projectionWeights + (j+0) * INPUT_DIM);
        device const float4* w1 = (device const float4*)(projectionWeights + (j+1) * INPUT_DIM);
        device const float4* w2 = (device const float4*)(projectionWeights + (j+2) * INPUT_DIM);
        device const float4* w3 = (device const float4*)(projectionWeights + (j+3) * INPUT_DIM);

        float4 accQ0 = float4(0.0f), accQ1 = float4(0.0f);
        float4 accQ2 = float4(0.0f), accQ3 = float4(0.0f);
        float4 accD0 = float4(0.0f), accD1 = float4(0.0f);
        float4 accD2 = float4(0.0f), accD3 = float4(0.0f);

        for (uint i = 0; i < INPUT_BLOCKS; ++i) {
            float4 q = query4[i];
            float4 d = db4[i];

            accQ0 = fma(q, w0[i], accQ0);
            accQ1 = fma(q, w1[i], accQ1);
            accQ2 = fma(q, w2[i], accQ2);
            accQ3 = fma(q, w3[i], accQ3);

            accD0 = fma(d, w0[i], accD0);
            accD1 = fma(d, w1[i], accD1);
            accD2 = fma(d, w2[i], accD2);
            accD3 = fma(d, w3[i], accD3);
        }

        float pQ0 = accQ0.x + accQ0.y + accQ0.z + accQ0.w;
        float pQ1 = accQ1.x + accQ1.y + accQ1.z + accQ1.w;
        float pQ2 = accQ2.x + accQ2.y + accQ2.z + accQ2.w;
        float pQ3 = accQ3.x + accQ3.y + accQ3.z + accQ3.w;

        float pD0 = accD0.x + accD0.y + accD0.z + accD0.w;
        float pD1 = accD1.x + accD1.y + accD1.z + accD1.w;
        float pD2 = accD2.x + accD2.y + accD2.z + accD2.w;
        float pD3 = accD3.x + accD3.y + accD3.z + accD3.w;

        float diff0 = pQ0 - pD0;
        float diff1 = pQ1 - pD1;
        float diff2 = pQ2 - pD2;
        float diff3 = pQ3 - pD3;

        distSq = fma(diff0, diff0, distSq);
        distSq = fma(diff1, diff1, distSq);
        distSq = fma(diff2, diff2, distSq);
        distSq = fma(diff3, diff3, distSq);
    }

    float distance = params.computeSqrt ? sqrt(distSq) : distSq;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}
