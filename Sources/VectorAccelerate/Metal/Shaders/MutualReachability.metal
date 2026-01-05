// VectorAccelerate: Mutual Reachability Distance Kernels
//
// Computes mutual reachability distances for HDBSCAN clustering.
// Formula: mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Kernels:
// - mutual_reachability_dense_kernel:  N×N matrix output
// - mutual_reachability_sparse_kernel: Specific pairs only

#include <metal_stdlib>
using namespace metal;

// Inlined helper: Compute L2 squared distance using float4 vectorization
// (Metal4Common.h may be stripped during runtime compilation)
inline float mutual_reach_l2_squared(
    device const float* vec_a,
    device const float* vec_b,
    uint dimension
) {
    float4 acc = float4(0.0f);

    const uint simd_blocks = dimension / 4;
    const uint remainder = dimension % 4;

    device const float4* a4 = reinterpret_cast<device const float4*>(vec_a);
    device const float4* b4 = reinterpret_cast<device const float4*>(vec_b);

    for (uint i = 0; i < simd_blocks; ++i) {
        float4 diff = a4[i] - b4[i];
        acc = fma(diff, diff, acc);
    }

    float sum = acc.x + acc.y + acc.z + acc.w;

    // Handle remainder
    if (remainder > 0) {
        device const float* a_tail = vec_a + (simd_blocks * 4);
        device const float* b_tail = vec_b + (simd_blocks * 4);
        for (uint i = 0; i < remainder; ++i) {
            float diff = a_tail[i] - b_tail[i];
            sum = fma(diff, diff, sum);
        }
    }

    return sum;
}

// MARK: - Parameters Structure

/// Parameters for mutual reachability kernel configuration.
/// Memory layout must match Swift's MutualReachabilityParams struct.
struct MutualReachabilityParams {
    uint32_t n;              // Number of points
    uint32_t d;              // Embedding dimension
    uint32_t strideEmbed;    // Stride between embeddings (default = d)
    uint32_t pairCount;      // Number of pairs (sparse mode only)
};

// MARK: - Dense Kernel (Generic)

/// Computes full N×N mutual reachability matrix.
///
/// Each thread computes one entry (i, j) of the output matrix.
/// Uses 2D dispatch: tid.x = row index (i), tid.y = column index (j)
///
/// - Inputs:
///   - embeddings: [N, D] row-major embedding matrix
///   - coreDistances: [N] pre-computed core distances
///   - params: kernel parameters
/// - Output:
///   - output: [N, N] symmetric mutual reachability matrix
kernel void mutual_reachability_dense_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i = tid.x;
    const uint j = tid.y;

    // Bounds check
    if (i >= params.n || j >= params.n) return;

    // Diagonal is always 0 (distance from point to itself)
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Compute L2 squared distance using vectorized helper
    device const float* vecI = embeddings + i * params.strideEmbed;
    device const float* vecJ = embeddings + j * params.strideEmbed;

    float distSq = mutual_reach_l2_squared(vecI, vecJ, params.d);
    float dist = sqrt(distSq);

    // Mutual reachability = max(core_i, core_j, euclidean_dist)
    float coreI = coreDistances[i];
    float coreJ = coreDistances[j];
    float mutualReach = max(max(coreI, coreJ), dist);

    output[i * params.n + j] = mutualReach;
}

// MARK: - Sparse Kernel

/// Computes mutual reachability for specific pairs only.
///
/// Each thread computes one pair. Uses 1D dispatch: tid = pair index.
/// More memory-efficient when only a subset of pairs are needed (e.g., MST edges).
///
/// - Inputs:
///   - embeddings: [N, D] row-major embedding matrix
///   - coreDistances: [N] pre-computed core distances
///   - pairs: [P, 2] array of (i, j) index pairs as packed uint2
///   - params: kernel parameters
/// - Output:
///   - output: [P] mutual reachability for each pair
kernel void mutual_reachability_sparse_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device const uint2* pairs           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    constant MutualReachabilityParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Bounds check
    if (tid >= params.pairCount) return;

    // Load pair indices
    uint2 pair = pairs[tid];
    uint i = pair.x;
    uint j = pair.y;

    // Handle same-point pairs (should be 0)
    if (i == j) {
        output[tid] = 0.0f;
        return;
    }

    // Compute L2 squared distance
    device const float* vecI = embeddings + i * params.strideEmbed;
    device const float* vecJ = embeddings + j * params.strideEmbed;

    float distSq = mutual_reach_l2_squared(vecI, vecJ, params.d);
    float dist = sqrt(distSq);

    // Mutual reachability = max(core_i, core_j, euclidean_dist)
    float coreI = coreDistances[i];
    float coreJ = coreDistances[j];
    float mutualReach = max(max(coreI, coreJ), dist);

    output[tid] = mutualReach;
}

// MARK: - Dimension-Optimized Kernels (Phase 2)

/// Optimized for D=384 (MiniLM, Sentence-BERT).
/// Uses 2 accumulators with 8x unrolling for maximum ILP.
kernel void mutual_reachability_384_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i = tid.x;
    const uint j = tid.y;

    if (i >= params.n || j >= params.n) return;
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Hardcoded stride for compiler optimization
    device const float4* vecI = (device const float4*)(embeddings + i * 384);
    device const float4* vecJ = (device const float4*)(embeddings + j * 384);

    // 2 accumulators with 8x unrolling (96 float4s = 12 iterations)
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint k = 0; k < 96; k += 8) {
        float4 d0 = vecI[k+0] - vecJ[k+0];
        float4 d1 = vecI[k+1] - vecJ[k+1];
        float4 d2 = vecI[k+2] - vecJ[k+2];
        float4 d3 = vecI[k+3] - vecJ[k+3];
        float4 d4 = vecI[k+4] - vecJ[k+4];
        float4 d5 = vecI[k+5] - vecJ[k+5];
        float4 d6 = vecI[k+6] - vecJ[k+6];
        float4 d7 = vecI[k+7] - vecJ[k+7];

        // Interleaved FMA for ILP
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc0 = fma(d2, d2, acc0);
        acc1 = fma(d3, d3, acc1);
        acc0 = fma(d4, d4, acc0);
        acc1 = fma(d5, d5, acc1);
        acc0 = fma(d6, d6, acc0);
        acc1 = fma(d7, d7, acc1);
    }

    float4 total = acc0 + acc1;
    float distSq = total.x + total.y + total.z + total.w;
    float dist = sqrt(distSq);

    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[i * params.n + j] = mutualReach;
}

/// Optimized for D=512 (small BERT variants).
/// Uses 2 accumulators with 8x unrolling.
kernel void mutual_reachability_512_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i = tid.x;
    const uint j = tid.y;

    if (i >= params.n || j >= params.n) return;
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Hardcoded stride for compiler optimization
    device const float4* vecI = (device const float4*)(embeddings + i * 512);
    device const float4* vecJ = (device const float4*)(embeddings + j * 512);

    // 2 accumulators with 8x unrolling (128 float4s = 16 iterations)
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint k = 0; k < 128; k += 8) {
        float4 d0 = vecI[k+0] - vecJ[k+0];
        float4 d1 = vecI[k+1] - vecJ[k+1];
        float4 d2 = vecI[k+2] - vecJ[k+2];
        float4 d3 = vecI[k+3] - vecJ[k+3];
        float4 d4 = vecI[k+4] - vecJ[k+4];
        float4 d5 = vecI[k+5] - vecJ[k+5];
        float4 d6 = vecI[k+6] - vecJ[k+6];
        float4 d7 = vecI[k+7] - vecJ[k+7];

        // Interleaved FMA for ILP
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc0 = fma(d2, d2, acc0);
        acc1 = fma(d3, d3, acc1);
        acc0 = fma(d4, d4, acc0);
        acc1 = fma(d5, d5, acc1);
        acc0 = fma(d6, d6, acc0);
        acc1 = fma(d7, d7, acc1);
    }

    float4 total = acc0 + acc1;
    float distSq = total.x + total.y + total.z + total.w;
    float dist = sqrt(distSq);

    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[i * params.n + j] = mutualReach;
}

/// Optimized for D=768 (BERT-base, DistilBERT, MPNet).
/// Uses 3 accumulators with 12x unrolling for better throughput.
kernel void mutual_reachability_768_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i = tid.x;
    const uint j = tid.y;

    if (i >= params.n || j >= params.n) return;
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Hardcoded stride for compiler optimization
    device const float4* vecI = (device const float4*)(embeddings + i * 768);
    device const float4* vecJ = (device const float4*)(embeddings + j * 768);

    // 3 accumulators with 12x unrolling (192 float4s = 16 iterations)
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);

    for (uint k = 0; k < 192; k += 12) {
        float4 d0 = vecI[k+0] - vecJ[k+0];
        float4 d1 = vecI[k+1] - vecJ[k+1];
        float4 d2 = vecI[k+2] - vecJ[k+2];
        float4 d3 = vecI[k+3] - vecJ[k+3];
        float4 d4 = vecI[k+4] - vecJ[k+4];
        float4 d5 = vecI[k+5] - vecJ[k+5];
        float4 d6 = vecI[k+6] - vecJ[k+6];
        float4 d7 = vecI[k+7] - vecJ[k+7];
        float4 d8 = vecI[k+8] - vecJ[k+8];
        float4 d9 = vecI[k+9] - vecJ[k+9];
        float4 d10 = vecI[k+10] - vecJ[k+10];
        float4 d11 = vecI[k+11] - vecJ[k+11];

        // Interleaved FMA with 3 accumulators
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc2 = fma(d2, d2, acc2);

        acc0 = fma(d3, d3, acc0);
        acc1 = fma(d4, d4, acc1);
        acc2 = fma(d5, d5, acc2);

        acc0 = fma(d6, d6, acc0);
        acc1 = fma(d7, d7, acc1);
        acc2 = fma(d8, d8, acc2);

        acc0 = fma(d9, d9, acc0);
        acc1 = fma(d10, d10, acc1);
        acc2 = fma(d11, d11, acc2);
    }

    float4 total = acc0 + acc1 + acc2;
    float distSq = total.x + total.y + total.z + total.w;
    float dist = sqrt(distSq);

    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[i * params.n + j] = mutualReach;
}

/// Optimized for D=1536 (OpenAI ada-002).
/// Uses 4 accumulators with 16x unrolling to maximize throughput.
kernel void mutual_reachability_1536_kernel(
    device const float* embeddings      [[buffer(0)]],
    device const float* coreDistances   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant MutualReachabilityParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i = tid.x;
    const uint j = tid.y;

    if (i >= params.n || j >= params.n) return;
    if (i == j) {
        output[i * params.n + j] = 0.0f;
        return;
    }

    // Hardcoded stride for compiler optimization
    device const float4* vecI = (device const float4*)(embeddings + i * 1536);
    device const float4* vecJ = (device const float4*)(embeddings + j * 1536);

    // 4 accumulators with 16x unrolling (384 float4s = 24 iterations)
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    for (uint k = 0; k < 384; k += 16) {
        float4 d0 = vecI[k+0] - vecJ[k+0];
        float4 d1 = vecI[k+1] - vecJ[k+1];
        float4 d2 = vecI[k+2] - vecJ[k+2];
        float4 d3 = vecI[k+3] - vecJ[k+3];
        float4 d4 = vecI[k+4] - vecJ[k+4];
        float4 d5 = vecI[k+5] - vecJ[k+5];
        float4 d6 = vecI[k+6] - vecJ[k+6];
        float4 d7 = vecI[k+7] - vecJ[k+7];
        float4 d8 = vecI[k+8] - vecJ[k+8];
        float4 d9 = vecI[k+9] - vecJ[k+9];
        float4 d10 = vecI[k+10] - vecJ[k+10];
        float4 d11 = vecI[k+11] - vecJ[k+11];
        float4 d12 = vecI[k+12] - vecJ[k+12];
        float4 d13 = vecI[k+13] - vecJ[k+13];
        float4 d14 = vecI[k+14] - vecJ[k+14];
        float4 d15 = vecI[k+15] - vecJ[k+15];

        // Interleaved FMA with 4 accumulators
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc2 = fma(d2, d2, acc2);
        acc3 = fma(d3, d3, acc3);

        acc0 = fma(d4, d4, acc0);
        acc1 = fma(d5, d5, acc1);
        acc2 = fma(d6, d6, acc2);
        acc3 = fma(d7, d7, acc3);

        acc0 = fma(d8, d8, acc0);
        acc1 = fma(d9, d9, acc1);
        acc2 = fma(d10, d10, acc2);
        acc3 = fma(d11, d11, acc3);

        acc0 = fma(d12, d12, acc0);
        acc1 = fma(d13, d13, acc1);
        acc2 = fma(d14, d14, acc2);
        acc3 = fma(d15, d15, acc3);
    }

    float4 total = acc0 + acc1 + acc2 + acc3;
    float distSq = total.x + total.y + total.z + total.w;
    float dist = sqrt(distSq);

    float mutualReach = max(max(coreDistances[i], coreDistances[j]), dist);
    output[i * params.n + j] = mutualReach;
}
