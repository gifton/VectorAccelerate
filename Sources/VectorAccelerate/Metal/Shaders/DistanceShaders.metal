// VectorAccelerate: Distance Computation Shaders
//
// GPU kernels for various distance metrics
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include <metal_stdlib>
using namespace metal;

#define VA_EPSILON 1e-8f

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
// 1 Threadgroup = 1 Vector Pair Evaluation
//
// Buffers:
//   [0] queries    — float[numQueries * dimension]
//   [1] targets    — float[numQueries * dimension]  (1:1 pairing with queries)
//   [2] distances  — float[numQueries]              (one result per pair)
//   [3] dimension  — uint
//   [4] compute_sqrt — uint  (0 = squared L2, 1 = Euclidean)
// -----------------------------------------------------------------------------
kernel void l2_distance(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& compute_sqrt [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    uint query_idx = tgid;

    device const float* q = queries + (ulong)query_idx * (ulong)dimension;
    device const float* t = targets + (ulong)query_idx * (ulong)dimension;

    uint vec_dim = dimension / 4;
    device const float4* q4 = (device const float4*)q;
    device const float4* t4 = (device const float4*)t;

    float sq_diff = 0.0;

    // Phase 1: Local float4 squared difference
    for (uint i = lid; i < vec_dim; i += threads_per_tg) {
        float4 diff = q4[i] - t4[i];
        sq_diff += dot(diff, diff);
    }

    // Phase 1b: Remainder handling
    uint rem_start = vec_dim * 4;
    for (uint i = rem_start + lid; i < dimension; i += threads_per_tg) {
        float diff = q[i] - t[i];
        sq_diff += diff * diff;
    }

    // Phase 2: SIMD-group reduction (Execution Width = 32)
    float simd_result = simd_sum(sq_diff);

    // Phase 3: Threadgroup consolidation
    threadgroup float shared_sums[32];
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_result;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Single write per threadgroup
    if (lid == 0) {
        uint active_simd_groups = (threads_per_tg + 31) / 32;
        float final_sum = 0.0;
        for (uint i = 0; i < active_simd_groups; i++) {
            final_sum += shared_sums[i];
        }

        distances[query_idx] = (compute_sqrt != 0) ? sqrt(final_sum) : final_sum;
    }
}

// -----------------------------------------------------------------------------
// Cosine Similarity (Float4 Optimized + In-Shader Finalization)
// 1 Threadgroup = 1 Vector Pair Evaluation
//
// Buffers:
//   [0] queries         — float[numQueries * dimension]
//   [1] targets         — float[numQueries * dimension]  (1:1 pairing)
//   [2] similarities    — float[numQueries]              (one result per pair)
//   [3] dimension       — uint
//   [4] output_distance — uint  (0 = similarity [-1,1], 1 = distance [0,2])
// -----------------------------------------------------------------------------
kernel void cosine_similarity(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& output_distance [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    uint query_idx = tgid;

    device const float* q = queries + (ulong)query_idx * (ulong)dimension;
    device const float* t = targets + (ulong)query_idx * (ulong)dimension;

    uint vec_dim = dimension / 4;
    device const float4* q4 = (device const float4*)q;
    device const float4* t4 = (device const float4*)t;

    // Accumulate A·B, A·A, B·B
    float3 local_sums = float3(0.0);

    for (uint i = lid; i < vec_dim; i += threads_per_tg) {
        float4 a_val = q4[i];
        float4 b_val = t4[i];

        local_sums.x += dot(a_val, b_val);
        local_sums.y += dot(a_val, a_val);
        local_sums.z += dot(b_val, b_val);
    }

    uint rem_start = vec_dim * 4;
    for (uint i = rem_start + lid; i < dimension; i += threads_per_tg) {
        float a_val = q[i];
        float b_val = t[i];
        local_sums.x += a_val * b_val;
        local_sums.y += a_val * a_val;
        local_sums.z += b_val * b_val;
    }

    // Phase 2: SIMD sums
    float3 simd_result = float3(simd_sum(local_sums.x), simd_sum(local_sums.y), simd_sum(local_sums.z));

    // Phase 3: Consolidate
    threadgroup float3 shared_sums[32];
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_result;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final output with in-shader mathematical finalization
    if (lid == 0) {
        uint active_simd_groups = (threads_per_tg + 31) / 32;
        float3 total = float3(0.0);
        for (uint i = 0; i < active_simd_groups; i++) {
            total += shared_sums[i];
        }

        float dot_ab = total.x;
        float dot_aa = total.y;
        float dot_bb = total.z;

        // Use sqrt(a)*sqrt(b) instead of sqrt(a*b) to avoid intermediate underflow
        float norm_a = sqrt(dot_aa);
        float norm_b = sqrt(dot_bb);
        float denom = norm_a * norm_b;
        // Clamp to the valid cosine range so FP drift past ±1 can't produce a negative
        // "1 - similarity" distance (which would break the nonnegativity invariant that
        // downstream min-heap / top-k selection relies on). NaN inputs must still propagate,
        // so only finite values are clamped: Metal's clamp() uses fmax and would otherwise
        // turn a NaN into -1.
        // Absolute-domain floor (FLT_MIN = leastNormalMagnitude) for the zero-vector test,
        // matching VectorCore's BE3 fix: a precision-relative 1e-8 wrongly rejects valid dense
        // micro-vectors (denom > 0 but < 1e-8) as "zero".
        float raw = (denom < FLT_MIN) ? 0.0f : (dot_ab / denom);
        float similarity = isnan(raw) ? raw : clamp(raw, -1.0f, 1.0f);

        similarities[query_idx] = (output_distance != 0) ? (1.0f - similarity) : similarity;
    }
}

// MARK: - Legacy Tiled Distance Computation (Retained for Internal Dispatch Compatibility)

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
