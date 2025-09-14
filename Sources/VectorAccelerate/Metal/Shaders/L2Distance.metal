// VectorAccelerate: L2 Distance Computation Kernels
//
// High-performance GPU kernels for L2 (Euclidean) distance computation
// Optimized for dimensions 512, 768, and 1536
//

#include <metal_stdlib>
using namespace metal;

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
    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Use a float4 accumulator to improve ILP and vectorize the accumulation
    float4 sum4 = float4(0.0f);
    const uint dimension = params.dimension;

    // SIMD optimization
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

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
    float distance = params.computeSqrt ? sqrt(sum) : sum;

    // Store result
    const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
    distances[outputIdx] = distance;
}

// MARK: - Optimized Kernels (Spec Section 3.2)

// Optimized for D=512 (128 float4 ops). Assumes dense packing (stride=512).
kernel void l2_distance_512_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Hardcoding the stride allows the compiler to optimize address calculation
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 512);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 512);

    // Unroll by 8. Use 2 accumulators and interleave instructions to hide latency and maximize ILP.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    // 128 iterations total (16 loops)
    for (uint i = 0; i < 128; i += 8) {
        float4 diff0 = query4[i+0] - database4[i+0];
        float4 diff1 = query4[i+1] - database4[i+1];
        float4 diff2 = query4[i+2] - database4[i+2];
        float4 diff3 = query4[i+3] - database4[i+3];
        float4 diff4 = query4[i+4] - database4[i+4];
        float4 diff5 = query4[i+5] - database4[i+5];
        float4 diff6 = query4[i+6] - database4[i+6];
        float4 diff7 = query4[i+7] - database4[i+7];

        // Interleaved FMA accumulation
        acc0 = fma(diff0, diff0, acc0);
        acc1 = fma(diff1, diff1, acc1);
        acc0 = fma(diff2, diff2, acc0);
        acc1 = fma(diff3, diff3, acc1);
        acc0 = fma(diff4, diff4, acc0);
        acc1 = fma(diff5, diff5, acc1);
        acc0 = fma(diff6, diff6, acc0);
        acc1 = fma(diff7, diff7, acc1);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1;
    float sum = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    float distance = params.computeSqrt ? sqrt(sum) : sum;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}

// Optimized for D=768 (192 float4 ops). Assumes dense packing (stride=768).
kernel void l2_distance_768_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 768);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 768);

    // Unroll by 12. Use 3 accumulators for better throughput on this dimension.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);

    // 192 iterations total (16 loops)
    for (uint i = 0; i < 192; i += 12) {
        // Compute differences (12 blocks)
        float4 diff0=query4[i+0]-database4[i+0]; float4 diff1=query4[i+1]-database4[i+1];
        float4 diff2=query4[i+2]-database4[i+2]; float4 diff3=query4[i+3]-database4[i+3];
        float4 diff4=query4[i+4]-database4[i+4]; float4 diff5=query4[i+5]-database4[i+5];
        float4 diff6=query4[i+6]-database4[i+6]; float4 diff7=query4[i+7]-database4[i+7];
        float4 diff8=query4[i+8]-database4[i+8]; float4 diff9=query4[i+9]-database4[i+9];
        float4 diff10=query4[i+10]-database4[i+10]; float4 diff11=query4[i+11]-database4[i+11];

        // Interleaved FMA accumulation
        acc0 = fma(diff0, diff0, acc0);
        acc1 = fma(diff1, diff1, acc1);
        acc2 = fma(diff2, diff2, acc2);

        acc0 = fma(diff3, diff3, acc0);
        acc1 = fma(diff4, diff4, acc1);
        acc2 = fma(diff5, diff5, acc2);

        acc0 = fma(diff6, diff6, acc0);
        acc1 = fma(diff7, diff7, acc1);
        acc2 = fma(diff8, diff8, acc2);

        acc0 = fma(diff9, diff9, acc0);
        acc1 = fma(diff10, diff10, acc1);
        acc2 = fma(diff11, diff11, acc2);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1 + acc2;
    float sum = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    float distance = params.computeSqrt ? sqrt(sum) : sum;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}

// Optimized for D=1536 (384 float4 ops). Assumes dense packing (stride=1536).
kernel void l2_distance_1536_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 1536);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 1536);

    // Unroll by 16. Use 4 accumulators to maximize throughput and hide latency.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    // 384 iterations total (24 loops)
    for (uint i = 0; i < 384; i += 16) {
        // Compute differences (16 blocks)
        float4 diff0=query4[i+0]-database4[i+0]; float4 diff1=query4[i+1]-database4[i+1];
        float4 diff2=query4[i+2]-database4[i+2]; float4 diff3=query4[i+3]-database4[i+3];
        float4 diff4=query4[i+4]-database4[i+4]; float4 diff5=query4[i+5]-database4[i+5];
        float4 diff6=query4[i+6]-database4[i+6]; float4 diff7=query4[i+7]-database4[i+7];
        float4 diff8=query4[i+8]-database4[i+8]; float4 diff9=query4[i+9]-database4[i+9];
        float4 diff10=query4[i+10]-database4[i+10]; float4 diff11=query4[i+11]-database4[i+11];
        float4 diff12=query4[i+12]-database4[i+12]; float4 diff13=query4[i+13]-database4[i+13];
        float4 diff14=query4[i+14]-database4[i+14]; float4 diff15=query4[i+15]-database4[i+15];

        // Interleaved FMA accumulation
        acc0 = fma(diff0, diff0, acc0);
        acc1 = fma(diff1, diff1, acc1);
        acc2 = fma(diff2, diff2, acc2);
        acc3 = fma(diff3, diff3, acc3);

        acc0 = fma(diff4, diff4, acc0);
        acc1 = fma(diff5, diff5, acc1);
        acc2 = fma(diff6, diff6, acc2);
        acc3 = fma(diff7, diff7, acc3);

        acc0 = fma(diff8, diff8, acc0);
        acc1 = fma(diff9, diff9, acc1);
        acc2 = fma(diff10, diff10, acc2);
        acc3 = fma(diff11, diff11, acc3);

        acc0 = fma(diff12, diff12, acc0);
        acc1 = fma(diff13, diff13, acc1);
        acc2 = fma(diff14, diff14, acc2);
        acc3 = fma(diff15, diff15, acc3);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1 + acc2 + acc3;
    float sum = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    float distance = params.computeSqrt ? sqrt(sum) : sum;
    distances[queryIdx * params.strideOutput + dbIdx] = distance;
}