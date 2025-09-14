//
//  OptimizedMatrixOps.metal
//  VectorAccelerate
//
//  Optimized matrix operations using tiling and shared memory
//

#include <metal_stdlib>
using namespace metal;

// Tile sizes optimized for Apple Silicon
constant int TILE_M = 32;  // Rows per tile
constant int TILE_N = 32;  // Columns per tile
constant int TILE_K = 8;   // Depth per tile

/// Optimized tiled matrix multiplication with shared memory
/// Uses 2D tiling strategy for better cache utilization
kernel void tiledMatrixMultiply(
    device const float* A [[buffer(0)]],  // M x K matrix
    device const float* B [[buffer(1)]],  // K x N matrix
    device float* C [[buffer(2)]],        // M x N result
    constant uint3& dims [[buffer(3)]],   // M, K, N dimensions
    threadgroup float* sharedA [[threadgroup(0)]],  // Shared memory for A tile
    threadgroup float* sharedB [[threadgroup(1)]],  // Shared memory for B tile
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint M = dims.x;
    const uint K = dims.y;
    const uint N = dims.z;
    
    // Global row and column
    const uint globalRow = tgid.y * TILE_M + tid.y;
    const uint globalCol = tgid.x * TILE_N + tid.x;
    
    // Accumulator for this thread's result
    float acc = 0.0f;
    
    // Number of tiles to iterate over
    const uint numTiles = (K + TILE_K - 1) / TILE_K;
    
    // Iterate over tiles
    for (uint tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // Load tile from A into shared memory
        const uint aRow = globalRow;
        const uint aCol = tileIdx * TILE_K + tid.x;
        
        if (aRow < M && aCol < K) {
            sharedA[tid.y * TILE_K + tid.x] = A[aRow * K + aCol];
        } else {
            sharedA[tid.y * TILE_K + tid.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        const uint bRow = tileIdx * TILE_K + tid.y;
        const uint bCol = globalCol;
        
        if (bRow < K && bCol < N) {
            sharedB[tid.y * TILE_N + tid.x] = B[bRow * N + bCol];
        } else {
            sharedB[tid.y * TILE_N + tid.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_K && (tileIdx * TILE_K + k) < K; ++k) {
            acc += sharedA[tid.y * TILE_K + k] * sharedB[k * TILE_N + tid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result to global memory
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = acc;
    }
}

/// Optimized matrix-vector multiplication using simdgroup operations
kernel void simdgroupMatrixVector(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],  // rows, cols
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;
    
    // Each SIMD group processes one or more rows
    const uint simdSize = 32;  // Apple Silicon SIMD width
    const uint simdgroups_per_threadgroup = 1;  // Typically 1 for simple kernels
    const uint row = tgid * simdgroups_per_threadgroup + sid;
    
    if (row >= rows) return;
    
    // Accumulate dot product across SIMD lanes
    float sum = 0.0f;
    
    // Process vector elements in chunks of SIMD width
    for (uint i = tid; i < cols; i += simdSize) {
        sum += matrix[row * cols + i] * vector[i];
    }
    
    // Reduce across SIMD group using shuffle operations
    sum = simd_sum(sum);
    
    // First thread writes result
    if (tid == 0) {
        result[row] = sum;
    }
}

/// Cache-optimized matrix transpose with 2D tiling
kernel void tiledTranspose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],  // rows, cols
    threadgroup float* tile [[threadgroup(0)]],  // Shared memory tile
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;
    
    const uint TILE_SIZE = 16;
    
    // Input indices
    uint inRow = tgid.y * TILE_SIZE + tid.y;
    uint inCol = tgid.x * TILE_SIZE + tid.x;
    
    // Load tile into shared memory (coalesced read)
    if (inRow < rows && inCol < cols) {
        tile[tid.y * (TILE_SIZE + 1) + tid.x] = input[inRow * cols + inCol];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Output indices (transposed)
    uint outRow = tgid.x * TILE_SIZE + tid.y;
    uint outCol = tgid.y * TILE_SIZE + tid.x;
    
    // Write transposed tile (coalesced write)
    if (outRow < cols && outCol < rows) {
        output[outRow * rows + outCol] = tile[tid.x * (TILE_SIZE + 1) + tid.y];
    }
}

/// Batch matrix multiplication with fused operations
kernel void batchMatrixMultiplyFused(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint4& params [[buffer(3)]],  // batchSize, M, K, N
    device const float* bias [[buffer(4)]],  // Optional bias
    constant float& alpha [[buffer(5)]],     // Scaling factor
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch = gid.z;
    const uint row = gid.y;
    const uint col = gid.x;
    
    const uint batchSize = params.x;
    const uint M = params.y;
    const uint K = params.z;
    const uint N = params.w;
    
    if (batch >= batchSize || row >= M || col >= N) return;
    
    // Compute offset for this batch
    const uint aOffset = batch * M * K;
    const uint bOffset = batch * K * N;
    const uint cOffset = batch * M * N;
    
    // Compute dot product for C[row][col]
    float sum = 0.0f;
    
    // Unrolled loop for better performance
    uint k = 0;
    for (; k + 3 < K; k += 4) {
        sum += A[aOffset + row * K + k + 0] * B[bOffset + (k + 0) * N + col];
        sum += A[aOffset + row * K + k + 1] * B[bOffset + (k + 1) * N + col];
        sum += A[aOffset + row * K + k + 2] * B[bOffset + (k + 2) * N + col];
        sum += A[aOffset + row * K + k + 3] * B[bOffset + (k + 3) * N + col];
    }
    
    // Handle remainder
    for (; k < K; ++k) {
        sum += A[aOffset + row * K + k] * B[bOffset + k * N + col];
    }
    
    // Apply scaling and bias if provided
    sum *= alpha;
    if (bias) {
        sum += bias[row * N + col];
    }
    
    // Write result
    C[cOffset + row * N + col] = sum;
}

/// High-performance vector normalization with fast inverse sqrt
kernel void fastNormalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint vectorIdx = gid.x;
    const uint offset = vectorIdx * dimension;
    
    // First pass: compute squared norm
    float sqNorm = 0.0f;
    for (uint i = 0; i < dimension; ++i) {
        float val = input[offset + i];
        sqNorm += val * val;
    }
    
    // Fast inverse square root approximation
    // More accurate than rsqrt() on some hardware
    float invNorm = rsqrt(sqNorm + 1e-8f);
    
    // Second pass: normalize
    for (uint i = 0; i < dimension; ++i) {
        output[offset + i] = input[offset + i] * invNorm;
    }
}

/// Strided batch GEMM for tensor operations
kernel void stridedBatchGEMM(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batchCount [[buffer(3)]],
    constant uint3& dims [[buffer(4)]],     // M, N, K
    constant uint3& stridesA [[buffer(5)]],  // Strides for A
    constant uint3& stridesB [[buffer(6)]],  // Strides for B
    constant uint3& stridesC [[buffer(7)]],  // Strides for C
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch = gid.z;
    if (batch >= batchCount) return;
    
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;
    
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Calculate batch offsets using strides
    const uint aOffset = batch * stridesA.z;
    const uint bOffset = batch * stridesB.z;
    const uint cOffset = batch * stridesC.z;
    
    // Compute matrix multiplication with custom strides
    float sum = 0.0f;
    
    for (uint k = 0; k < K; ++k) {
        uint aIdx = aOffset + row * stridesA.x + k * stridesA.y;
        uint bIdx = bOffset + k * stridesB.x + col * stridesB.y;
        sum += A[aIdx] * B[bIdx];
    }
    
    uint cIdx = cOffset + row * stridesC.x + col * stridesC.y;
    C[cIdx] = sum;
}