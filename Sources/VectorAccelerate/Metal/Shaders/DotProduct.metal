// VectorAccelerate: Dot Product Computation Kernels
//
// High-performance GPU kernels for dot product computation
// Optimized for embedding dimensions: 384, 512, 768, and 1536
// Includes specialized GEMV kernel for single-query optimization
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Parameters Structure (Spec Section 2)

// Parameter structure for kernel configuration
struct DotProductParams {
    uint32_t numQueries;        // Number of query vectors (N)
    uint32_t numDatabase;       // Number of database vectors (M)
    uint32_t dimension;         // Vector dimension (D)
    uint32_t strideQuery;       // Stride between query vectors
    uint32_t strideDatabase;    // Stride between database vectors
    uint32_t strideOutput;      // Stride for output matrix
    uint8_t  absoluteValue;     // 0 = normal, 1 = absolute value of result
    uint8_t  padding[3];        // Alignment padding
};

// MARK: - General Dot Product Kernel (Spec Section 3.1)

// Handles arbitrary dimensions and strides.
kernel void dot_product_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant DotProductParams& params [[buffer(3)]],
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

    const uint dimension = params.dimension;
    float dotProduct = 0.0f;

    // Spec Section 3.1: Choose strategy based on dimension
    if (dimension <= 16) {
        // Small vectors: optimized for low latency, compiler will likely unroll.
        for (uint i = 0; i < dimension; ++i) {
            // Use explicit fma (fused multiply-add)
            dotProduct = fma(query[i], database[i], dotProduct);
        }
    } else {
        // Larger vectors: optimized for throughput using SIMD
        const uint simd_blocks = dimension / 4;
        const uint remainder = dimension % 4;

        device const float4* query4 = (device const float4*)query;
        device const float4* database4 = (device const float4*)database;

        // Use float4 accumulator to improve ILP and vectorize the accumulation
        float4 acc4 = float4(0.0f);

        // Process 4 elements at a time
        for (uint i = 0; i < simd_blocks; ++i) {
            acc4 = fma(query4[i], database4[i], acc4);
        }

        // Horizontal reduction
        dotProduct = acc4.x + acc4.y + acc4.z + acc4.w;

        // Handle remaining elements (if dimension is not a multiple of 4)
        if (remainder > 0) {
            device const float* query_tail = query + (simd_blocks * 4);
            device const float* database_tail = database + (simd_blocks * 4);

            for (uint i = 0; i < remainder; ++i) {
                dotProduct = fma(query_tail[i], database_tail[i], dotProduct);
            }
        }
    }

    // Apply absolute value if requested
    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }

    // Store result
    const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
    dotProducts[outputIdx] = dotProduct;
}

// MARK: - Optimized Kernels (Spec Section 3.2)
// These kernels assume dense packing (stride == dimension) and maximize ILP.

// Optimized for D=384 (96 float4 ops).
// Critical for MiniLM and Sentence-BERT embeddings (VectorCore 0.1.5 Vector384Optimized)
kernel void dot_product_384_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant DotProductParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // Hardcoding the stride allows the compiler to optimize address calculation
    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 384);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 384);

    // Unroll by 8. Use 2 accumulators and interleave instructions to maximize ILP.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    // 96 iterations total (12 loops of 8)
    for (uint i = 0; i < 96; i += 8) {
        // Interleaved FMA accumulation
        acc0 = fma(query4[i+0], database4[i+0], acc0);
        acc1 = fma(query4[i+1], database4[i+1], acc1);
        acc0 = fma(query4[i+2], database4[i+2], acc0);
        acc1 = fma(query4[i+3], database4[i+3], acc1);
        acc0 = fma(query4[i+4], database4[i+4], acc0);
        acc1 = fma(query4[i+5], database4[i+5], acc1);
        acc0 = fma(query4[i+6], database4[i+6], acc0);
        acc1 = fma(query4[i+7], database4[i+7], acc1);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1;
    float dotProduct = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }
    dotProducts[queryIdx * params.strideOutput + dbIdx] = dotProduct;
}

// Optimized for D=512 (128 float4 ops).
kernel void dot_product_512_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant DotProductParams& params [[buffer(3)]],
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

    // Unroll by 16. Use 4 accumulators and interleave instructions to maximize ILP.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    // 128 iterations total (8 loops)
    for (uint i = 0; i < 128; i += 16) {
        // Interleaved FMA accumulation
        acc0 = fma(query4[i+0], database4[i+0], acc0);
        acc1 = fma(query4[i+1], database4[i+1], acc1);
        acc2 = fma(query4[i+2], database4[i+2], acc2);
        acc3 = fma(query4[i+3], database4[i+3], acc3);

        acc0 = fma(query4[i+4], database4[i+4], acc0);
        acc1 = fma(query4[i+5], database4[i+5], acc1);
        acc2 = fma(query4[i+6], database4[i+6], acc2);
        acc3 = fma(query4[i+7], database4[i+7], acc3);

        acc0 = fma(query4[i+8], database4[i+8], acc0);
        acc1 = fma(query4[i+9], database4[i+9], acc1);
        acc2 = fma(query4[i+10], database4[i+10], acc2);
        acc3 = fma(query4[i+11], database4[i+11], acc3);

        acc0 = fma(query4[i+12], database4[i+12], acc0);
        acc1 = fma(query4[i+13], database4[i+13], acc1);
        acc2 = fma(query4[i+14], database4[i+14], acc2);
        acc3 = fma(query4[i+15], database4[i+15], acc3);
    }

    // Final reduction
    float4 total_acc = acc0 + acc1 + acc2 + acc3;
    float dotProduct = total_acc.x + total_acc.y + total_acc.z + total_acc.w;

    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }
    dotProducts[queryIdx * params.strideOutput + dbIdx] = dotProduct;
}

// Optimized for D=768 (192 float4 ops). (Spec Section 3.2 Implementation)
kernel void dot_product_768_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant DotProductParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 768);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 768);

    // Unroll by 16. Use 4 accumulators for maximum ILP.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    // 192 iterations = 12 loops of 16
    for (uint i = 0; i < 192; i += 16) {
        // Group 1
        acc0 = fma(query4[i+0], database4[i+0], acc0);
        acc1 = fma(query4[i+1], database4[i+1], acc1);
        acc2 = fma(query4[i+2], database4[i+2], acc2);
        acc3 = fma(query4[i+3], database4[i+3], acc3);

        // Group 2
        acc0 = fma(query4[i+4], database4[i+4], acc0);
        acc1 = fma(query4[i+5], database4[i+5], acc1);
        acc2 = fma(query4[i+6], database4[i+6], acc2);
        acc3 = fma(query4[i+7], database4[i+7], acc3);

        // Group 3
        acc0 = fma(query4[i+8], database4[i+8], acc0);
        acc1 = fma(query4[i+9], database4[i+9], acc1);
        acc2 = fma(query4[i+10], database4[i+10], acc2);
        acc3 = fma(query4[i+11], database4[i+11], acc3);

        // Group 4
        acc0 = fma(query4[i+12], database4[i+12], acc0);
        acc1 = fma(query4[i+13], database4[i+13], acc1);
        acc2 = fma(query4[i+14], database4[i+14], acc2);
        acc3 = fma(query4[i+15], database4[i+15], acc3);
    }

    // Final reduction
    float4 total = acc0 + acc1 + acc2 + acc3;
    float dotProduct = total.x + total.y + total.z + total.w;

    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }

    dotProducts[queryIdx * params.strideOutput + dbIdx] = dotProduct;
}

// Optimized for D=1536 (384 float4 ops).
kernel void dot_product_1536_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant DotProductParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 1536);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 1536);

    // Unroll by 32. Use 8 accumulators for extreme ILP optimization on this large dimension.
    float4 acc0=0, acc1=0, acc2=0, acc3=0;
    float4 acc4=0, acc5=0, acc6=0, acc7=0;

    // 384 iterations = 12 loops of 32
    for (uint i = 0; i < 384; i += 32) {
        // Group 1
        acc0 = fma(query4[i+0], database4[i+0], acc0);
        acc1 = fma(query4[i+1], database4[i+1], acc1);
        acc2 = fma(query4[i+2], database4[i+2], acc2);
        acc3 = fma(query4[i+3], database4[i+3], acc3);
        acc4 = fma(query4[i+4], database4[i+4], acc4);
        acc5 = fma(query4[i+5], database4[i+5], acc5);
        acc6 = fma(query4[i+6], database4[i+6], acc6);
        acc7 = fma(query4[i+7], database4[i+7], acc7);

        // Group 2
        acc0 = fma(query4[i+8], database4[i+8], acc0);
        acc1 = fma(query4[i+9], database4[i+9], acc1);
        acc2 = fma(query4[i+10], database4[i+10], acc2);
        acc3 = fma(query4[i+11], database4[i+11], acc3);
        acc4 = fma(query4[i+12], database4[i+12], acc4);
        acc5 = fma(query4[i+13], database4[i+13], acc5);
        acc6 = fma(query4[i+14], database4[i+14], acc6);
        acc7 = fma(query4[i+15], database4[i+15], acc7);

        // Group 3
        acc0 = fma(query4[i+16], database4[i+16], acc0);
        acc1 = fma(query4[i+17], database4[i+17], acc1);
        acc2 = fma(query4[i+18], database4[i+18], acc2);
        acc3 = fma(query4[i+19], database4[i+19], acc3);
        acc4 = fma(query4[i+20], database4[i+20], acc4);
        acc5 = fma(query4[i+21], database4[i+21], acc5);
        acc6 = fma(query4[i+22], database4[i+22], acc6);
        acc7 = fma(query4[i+23], database4[i+23], acc7);

        // Group 4
        acc0 = fma(query4[i+24], database4[i+24], acc0);
        acc1 = fma(query4[i+25], database4[i+25], acc1);
        acc2 = fma(query4[i+26], database4[i+26], acc2);
        acc3 = fma(query4[i+27], database4[i+27], acc3);
        acc4 = fma(query4[i+28], database4[i+28], acc4);
        acc5 = fma(query4[i+29], database4[i+29], acc5);
        acc6 = fma(query4[i+30], database4[i+30], acc6);
        acc7 = fma(query4[i+31], database4[i+31], acc7);
    }

    // Final reduction
    float4 total1 = acc0 + acc1 + acc2 + acc3;
    float4 total2 = acc4 + acc5 + acc6 + acc7;
    float4 total = total1 + total2;
    float dotProduct = total.x + total.y + total.z + total.w;

    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }

    dotProducts[queryIdx * params.strideOutput + dbIdx] = dotProduct;
}


// MARK: - GEMV Variant (Spec Section 3.3)

// Optimized for Matrix-Vector multiplication (Single Query, Multiple Database).
kernel void dot_product_gemv_kernel(
    device const float* vector [[buffer(0)]],          // [1, D] single query vector
    device const float* matrix [[buffer(1)]],          // [M, D] database matrix
    device float* results [[buffer(2)]],               // [M] output dot products
    constant DotProductParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]               // 1D Thread ID (one per database vector)
) {
    const uint dbIdx = tid;

    // Bounds checking for 1D dispatch
    if (dbIdx >= params.numDatabase) {
        return;
    }

    const uint dimension = params.dimension;
    device const float* matrixRow = matrix + (dbIdx * params.strideDatabase);

    // SIMD optimization
    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* vector4 = (device const float4*)vector;
    device const float4* row4 = (device const float4*)matrixRow;

    // Use 4 accumulators for ILP. Unroll by 16.
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    uint i = 0;
    // Main loop unrolled by 16 (64 elements per iteration)
    for (; i + 15 < simd_blocks; i += 16) {
        // 'vector4' reads are broadcast efficiently across the threadgroup.
        acc0 = fma(vector4[i+0], row4[i+0], acc0);
        acc1 = fma(vector4[i+1], row4[i+1], acc1);
        acc2 = fma(vector4[i+2], row4[i+2], acc2);
        acc3 = fma(vector4[i+3], row4[i+3], acc3);

        acc0 = fma(vector4[i+4], row4[i+4], acc0);
        acc1 = fma(vector4[i+5], row4[i+5], acc1);
        acc2 = fma(vector4[i+6], row4[i+6], acc2);
        acc3 = fma(vector4[i+7], row4[i+7], acc3);

        acc0 = fma(vector4[i+8], row4[i+8], acc0);
        acc1 = fma(vector4[i+9], row4[i+9], acc1);
        acc2 = fma(vector4[i+10], row4[i+10], acc2);
        acc3 = fma(vector4[i+11], row4[i+11], acc3);

        acc0 = fma(vector4[i+12], row4[i+12], acc0);
        acc1 = fma(vector4[i+13], row4[i+13], acc1);
        acc2 = fma(vector4[i+14], row4[i+14], acc2);
        acc3 = fma(vector4[i+15], row4[i+15], acc3);
    }

    // Combine main accumulators
    float4 acc = acc0 + acc1 + acc2 + acc3;

    // Handle remaining SIMD blocks (0-15 blocks left)
    for (; i < simd_blocks; ++i) {
        acc = fma(vector4[i], row4[i], acc);
    }

    // Horizontal reduction
    float dotProduct = acc.x + acc.y + acc.z + acc.w;

    // Handle remainder elements (0-3 elements left)
    if (remainder > 0) {
        device const float* vector_tail = vector + (simd_blocks * 4);
        device const float* row_tail = matrixRow + (simd_blocks * 4);

        for (uint j = 0; j < remainder; ++j) {
            dotProduct = fma(vector_tail[j], row_tail[j], dotProduct);
        }
    }

    if (params.absoluteValue) {
        dotProduct = abs(dotProduct);
    }

    results[dbIdx] = dotProduct;
}