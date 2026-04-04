// VectorAccelerate: Distance Computation Shaders
//
// GPU kernels for various distance metrics
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Parameter Structures

/// Parameters for L2 distance kernel.
struct L2DistanceParams {
    uint32_t numQueries;
    uint32_t numDatabase;
    uint32_t dimension;
    uint32_t strideQuery;
    uint32_t strideDatabase;
    uint32_t strideOutput;
    uint8_t  computeSqrt;
    uint8_t  padding[3];
};

/// Parameters for Cosine Similarity kernel.
struct CosineSimilarityParams {
    uint32_t numQueries;
    uint32_t numDatabase;
    uint32_t dimension;
    uint32_t strideQuery;
    uint32_t strideDatabase;
    uint32_t strideOutput;
    uint8_t  outputDistance;
    uint8_t  inputsNormalized;
    uint8_t  padding[2];
};

// MARK: - Manhattan Distance

kernel void manhattanDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    // Each thread computes partial sum
    threadgroup float partial_sums[256];
    uint tid = id % 256;
    
    float local_sum = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        local_sum += abs(vectorA[i] - vectorB[i]);
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    if (tid == 0) {
        float total = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total += partial_sums[i];
        }
        atomic_store_explicit((device atomic_float*)result, total, memory_order_relaxed);
    }
}

// MARK: - Chebyshev Distance

kernel void chebyshevDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup float partial_max[256];
    uint tid = id % 256;
    
    float local_max = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        local_max = max(local_max, abs(vectorA[i] - vectorB[i]));
    }
    
    partial_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to find maximum
    if (tid == 0) {
        float maximum = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            maximum = max(maximum, partial_max[i]);
        }
        atomic_store_explicit((device atomic_float*)result, maximum, memory_order_relaxed);
    }
}

// MARK: - Minkowski Distance

kernel void minkowskiDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant float& p [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup float partial_sums[256];
    uint tid = id % 256;
    
    float local_sum = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        float diff = abs(vectorA[i] - vectorB[i]);
        local_sum += pow(diff, p);
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        float total = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total += partial_sums[i];
        }
        atomic_store_explicit((device atomic_float*)result, total, memory_order_relaxed);
    }
}

// MARK: - Jaccard Distance

kernel void jaccardDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup float partial_intersection[256];
    threadgroup float partial_union[256];
    uint tid = id % 256;
    
    float local_intersection = 0.0;
    float local_union = 0.0;
    
    for (uint i = id; i < dimension; i += 256) {
        float a = vectorA[i];
        float b = vectorB[i];
        local_intersection += min(a, b);
        local_union += max(a, b);
    }
    
    partial_intersection[tid] = local_intersection;
    partial_union[tid] = local_union;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        float total_intersection = 0.0;
        float total_union = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total_intersection += partial_intersection[i];
            total_union += partial_union[i];
        }
        
        float jaccard = (total_union > VA_EPSILON) ? 
            (1.0 - (total_intersection / total_union)) : 1.0;
        atomic_store_explicit((device atomic_float*)result, jaccard, memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// L2 Distance (Hierarchical SIMD Reduction)
// 1 Threadgroup = 1 Vector Pair Evaluation (or row/column in cross-product)
// -----------------------------------------------------------------------------
kernel void l2_distance(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant L2DistanceParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 threads_per_tg [[threads_per_threadgroup]]
) {
    uint q_idx = tgid.x;
    uint d_idx = tgid.y;

    if (q_idx >= params.numQueries || d_idx >= params.numDatabase) return;

    device const float* q = queries + (ulong)q_idx * (ulong)params.strideQuery;
    device const float* t = targets + (ulong)d_idx * (ulong)params.strideDatabase;

    uint dimension = params.dimension;
    uint vec_dim = dimension / 4;
    device const float4* q4 = (device const float4*)q;
    device const float4* t4 = (device const float4*)t;

    float sq_diff = 0.0;

    // Phase 1: Local float4 squared difference
    for (uint i = lid.x; i < vec_dim; i += threads_per_tg.x) {
        float4 diff = q4[i] - t4[i];
        sq_diff += dot(diff, diff);
    }

    // Phase 1b: Remainder handling
    uint rem_start = vec_dim * 4;
    for (uint i = rem_start + lid.x; i < dimension; i += threads_per_tg.x) {
        float diff = q[i] - t[i];
        sq_diff += diff * diff;
    }

    // Phase 2: SIMD-group reduction
    float simd_result = simd_sum(sq_diff);

    // Phase 3: Threadgroup consolidation
    threadgroup float shared_sums[32];
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_result;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Single write per threadgroup
    if (lid.x == 0) {
        uint active_simd_groups = (threads_per_tg.x + 31) / 32;
        float final_sum = 0.0;
        for (uint i = 0; i < active_simd_groups; i++) {
            final_sum += shared_sums[i];
        }

        uint out_idx = q_idx * params.numDatabase + d_idx;
        distances[out_idx] = (params.computeSqrt != 0) ? sqrt(final_sum) : final_sum;
    }
}

// -----------------------------------------------------------------------------
// Cosine Similarity (Hierarchical SIMD Reduction)
// -----------------------------------------------------------------------------
kernel void cosine_similarity(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 threads_per_tg [[threads_per_threadgroup]]
) {
    uint q_idx = tgid.x;
    uint d_idx = tgid.y;

    if (q_idx >= params.numQueries || d_idx >= params.numDatabase) return;

    device const float* q = queries + (ulong)q_idx * (ulong)params.strideQuery;
    device const float* t = targets + (ulong)d_idx * (ulong)params.strideDatabase;

    uint dimension = params.dimension;
    uint vec_dim = dimension / 4;
    device const float4* q4 = (device const float4*)q;
    device const float4* t4 = (device const float4*)t;

    float dot_ab = 0.0;
    float dot_aa = 0.0;
    float dot_bb = 0.0;

    // Phase 1: Local float4 accumulation
    for (uint i = lid.x; i < vec_dim; i += threads_per_tg.x) {
        float4 q_val = q4[i];
        float4 t_val = t4[i];
        dot_ab += dot(q_val, t_val);
        dot_aa += dot(q_val, q_val);
        dot_bb += dot(t_val, t_val);
    }

    // Phase 1b: Remainder handling
    uint rem_start = vec_dim * 4;
    for (uint i = rem_start + lid.x; i < dimension; i += threads_per_tg.x) {
        float q_val = q[i];
        float t_val = t[i];
        dot_ab += q_val * t_val;
        dot_aa += q_val * q_val;
        dot_bb += t_val * t_val;
    }

    // Phase 2: SIMD-group reduction
    float3 results = float3(simd_sum(dot_ab), simd_sum(dot_aa), simd_sum(dot_bb));

    // Phase 3: Threadgroup consolidation
    threadgroup float3 shared_sums[32];
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = results;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final calculation and single write
    if (lid.x == 0) {
        uint active_simd_groups = (threads_per_tg.x + 31) / 32;
        float3 final_sums = 0.0;
        for (uint i = 0; i < active_simd_groups; i++) {
            final_sums += shared_sums[i];
        }

        float similarity = 0.0;
        float denom = sqrt(final_sums.y * final_sums.z);
        if (denom > 1e-8f) {
            similarity = final_sums.x / denom;
        }

        uint out_idx = q_idx * params.numDatabase + d_idx;
        output[out_idx] = (params.outputDistance != 0) ? (1.0f - similarity) : similarity;
    }
}

// MARK: - Legacy Tiled Distance Computation

kernel void batchCosineSimilaritySIMD(
    constant float4* query [[buffer(0)]],
    constant float4* candidates [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dimension4 [[buffer(3)]],
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    uint candidateOffset = id * dimension4;
    float4 dot_accum = 0.0;
    float4 query_norm_accum = 0.0;
    float4 candidate_norm_accum = 0.0;
    for (uint i = 0; i < dimension4; i++) {
        float4 q = query[i];
        float4 c = candidates[candidateOffset + i];
        dot_accum += q * c;
        query_norm_accum += q * q;
        candidate_norm_accum += c * c;
    }
    float dot_product = dot_accum.x + dot_accum.y + dot_accum.z + dot_accum.w;
    float query_norm = sqrt(query_norm_accum.x + query_norm_accum.y + query_norm_accum.z + query_norm_accum.w);
    float candidate_norm = sqrt(candidate_norm_accum.x + candidate_norm_accum.y + candidate_norm_accum.z + candidate_norm_accum.w);
    if (query_norm > 0.0 && candidate_norm > 0.0) {
        similarities[id] = dot_product / (query_norm * candidate_norm);
    } else {
        similarities[id] = 0.0;
    }
}

kernel void batchDotProductSIMD(
    constant float4* query [[buffer(0)]],
    constant float4* candidates [[buffer(1)]],
    device float* dotProducts [[buffer(2)]],
    constant uint& dimension4 [[buffer(3)]],
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    uint candidateOffset = id * dimension4;
    float4 dot_accum = 0.0;
    for (uint i = 0; i < dimension4; i++) {
        float4 q = query[i];
        float4 c = candidates[candidateOffset + i];
        dot_accum += q * c;
    }
    dotProducts[id] = dot_accum.x + dot_accum.y + dot_accum.z + dot_accum.w;
}

kernel void batchEuclideanDistanceSIMD(
    constant float4* query [[buffer(0)]],
    constant float4* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension4 [[buffer(3)]],
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    uint candidateOffset = id * dimension4;
    float4 sum_squared = 0.0;
    for (uint i = 0; i < dimension4; i++) {
        float4 diff = query[i] - candidates[candidateOffset + i];
        sum_squared += diff * diff;
    }
    float squared_distance = sum_squared.x + sum_squared.y + sum_squared.z + sum_squared.w;
    distances[id] = sqrt(squared_distance);
}
