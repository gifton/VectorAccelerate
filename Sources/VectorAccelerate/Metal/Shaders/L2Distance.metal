// VectorAccelerate: L2 Distance Computation Kernels
//
// High-performance GPU kernels for L2 (Euclidean) distance computation
// Optimized for embedding dimensions: 384, 512, 768, and 1536
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Dimension optimizations:
// - D=384:  MiniLM, all-MiniLM-L6-v2, Sentence-BERT
// - D=512:  Small BERT variants
// - D=768:  BERT-base, DistilBERT, MPNet
// - D=1536: OpenAI ada-002

#include "Metal4Common.h"

// MARK: - Parameters Structure (Spec Section 2)

// Parameter structure for kernel configuration
struct L2DistanceParams {
    uint32_t numQueries;        // Number of query vectors (N)
    uint32_t numDatabase;       // Number of database vectors (M)
    uint32_t dimension;         // Vector dimension (D)
    uint32_t strideQuery;       // Stride between query vectors
    uint32_t strideDatabase;    // Stride between database vectors
    uint32_t strideOutput;      // Stride for output matrix
    uint8_t  computeSqrt;       // 0 = squared distance, 1 = apply sqrt
    uint8_t  padding[3];        // Alignment padding
};

// MARK: - General L2 Distance Kernel (Spec Section 3.1)

// Handles arbitrary dimensions and strides.
kernel void l2_distance_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]] // (x=queryIdx, y=dbIdx)
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    // Bounds checking (required when using dispatchThreads)
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Calculate vector pointers using strides
    device const float* query = queryVectors + ((ulong)queryIdx * params.strideQuery);
    device const float* database = databaseVectors + ((ulong)dbIdx * params.strideDatabase);

    // Use a float4 accumulator to improve ILP and vectorize the accumulation
    float4 sum4 = float4(0.0f);
    const uint dimension = params.dimension;

    // SIMD optimization
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const packed_float4* query4 = (device const packed_float4*)query;
    device const packed_float4* database4 = (device const packed_float4*)database;

    // Process 4 elements at a time
    for (uint i = 0; i < simd_blocks; ++i) {
        float4 diff = query4[i] - database4[i];
        // Use explicit fma (fused multiply-add) for precision and performance
        sum4 = fma(diff, diff, sum4);
    }

    // Horizontal reduction of the vector accumulator
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;

    // Handle remaining elements (if dimension is not a multiple of 4)
    if (remainder > 0) {
        device const float* query_tail = query + (simd_blocks * 4);
        device const float* database_tail = database + (simd_blocks * 4);

        for (uint i = 0; i < remainder; ++i) {
            float diff = query_tail[i] - database_tail[i];
            sum = fma(diff, diff, sum);
        }
    }

    // Apply sqrt if requested
    float distance = params.computeSqrt ? va_euclidean_finalize(sum, query, database, dimension) : sum;

    // Store result
    const ulong outputIdx = (ulong)queryIdx * params.strideOutput + dbIdx;
    distances[outputIdx] = distance;
}

