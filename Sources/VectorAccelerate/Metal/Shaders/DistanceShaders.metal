// VectorAccelerate: Distance Computation Shaders
//
// GPU kernels for various distance metrics
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"
#include <metal_stdlib>
using namespace metal;

// File-local floor for the jaccard union gate below. Deliberately NOT named VA_EPSILON:
// in the combined runtime TU (KernelContext.makeLibraryFromBundleSources) a file-scope
// redefinition of a shared macro changes the value for every file compiled after this one —
// that was AUDIT-3 VA3-012, which silently ran the downstream normalize/scale/histogram
// gates at 1e-8 in release builds vs 1e-7 in the debug metallib. The 1e-8 value is this
// file's historical policy, preserved identically on both build paths.
// EpsilonCompileParityTests.testSingleEpsilonAuthority enforces the naming rule corpus-wide.
#define VA_JACCARD_UNION_EPSILON 1e-8f

// MARK: - Manhattan Distance

// Single-pair Manhattan distance: Σ |a_i − b_i|.
//
// Dispatch contract: exactly ONE threadgroup, any width — only the first min(tgSize, 256)
// lanes carry reduction state, the same contract as `jaccardDistance` below (extra
// threadgroups would each redundantly compute and publish the identical full result).
// The pre-AUDIT-3 version derived its lane from `thread_position_in_grid % 256` with a
// hardcoded 256 stride, returned before the barrier for out-of-range threads, and let every
// threadgroup `atomic_store` its own partial total into result[0] — correct only under the
// engine's exact `min(256, dimension)` × 1 dispatch, wrong or racy under any other geometry
// (AUDIT-3 VA3-013, the VA3-001 defect family). All threads now reach the barrier
// unconditionally and thread 0 alone publishes, so no atomics are needed.
kernel void manhattanDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[256];
    const uint lanes = min(tgSize, 256u);

    if (tid < lanes) {
        float local_sum = 0.0f;
        for (uint i = tid; i < dimension; i += lanes) {
            local_sum += abs(vectorA[i] - vectorB[i]);
        }
        partial_sums[tid] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < lanes; i++) {
            total += partial_sums[i];
        }
        result[0] = total;
    }
}

// MARK: - Chebyshev Distance

// Single-pair Chebyshev distance: max |a_i − b_i|. Same dispatch contract and history as
// `manhattanDistance` above (AUDIT-3 VA3-013).
kernel void chebyshevDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partial_max[256];
    const uint lanes = min(tgSize, 256u);

    if (tid < lanes) {
        float local_max = 0.0f;
        for (uint i = tid; i < dimension; i += lanes) {
            local_max = max(local_max, abs(vectorA[i] - vectorB[i]));
        }
        partial_max[tid] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maximum = 0.0f;
        for (uint i = 0; i < lanes; i++) {
            maximum = max(maximum, partial_max[i]);
        }
        result[0] = maximum;
    }
}

// MARK: - Jaccard Distance

// Weighted Jaccard distance for one vector pair: 1 − Σ min(a_i, b_i) / Σ max(a_i, b_i).
//
// Dispatch contract: exactly ONE threadgroup (any size; only the first min(tgSize, 256)
// lanes carry reduction state). The pre-AUDIT-3 version strided by a hardcoded 256 across
// ceil(dimension/256) threadgroups: every group `atomic_store`d its own partial total into
// result[0] — nondeterministically wrong for dimension > 256 — and out-of-range threads
// returned before the barrier, which is barrier divergence (undefined behavior) for any
// dimension not a multiple of the group size (AUDIT-3 VA3-001). All threads now reach the
// barrier unconditionally and thread 0 alone publishes the result, so no atomics are needed.
kernel void jaccardDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partial_intersection[256];
    threadgroup float partial_union[256];
    const uint lanes = min(tgSize, 256u);

    if (tid < lanes) {
        float local_intersection = 0.0f;
        float local_union = 0.0f;
        for (uint i = tid; i < dimension; i += lanes) {
            float a = vectorA[i];
            float b = vectorB[i];
            local_intersection += min(a, b);
            local_union += max(a, b);
        }
        partial_intersection[tid] = local_intersection;
        partial_union[tid] = local_union;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total_intersection = 0.0f;
        float total_union = 0.0f;
        for (uint i = 0; i < lanes; i++) {
            total_intersection += partial_intersection[i];
            total_union += partial_union[i];
        }

        float jaccard = (total_union > VA_JACCARD_UNION_EPSILON) ?
            (1.0f - (total_intersection / total_union)) : 1.0f;
        result[0] = jaccard;
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
    device const packed_float4* q4 = (device const packed_float4*)q;
    device const packed_float4* t4 = (device const packed_float4*)t;

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

        distances[query_idx] = (compute_sqrt != 0) ? va_euclidean_finalize(final_sum, q, t, dimension) : final_sum;
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
    device const packed_float4* q4 = (device const packed_float4*)q;
    device const packed_float4* t4 = (device const packed_float4*)t;

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

        // Overflow/underflow rescue (AUDIT-2 VA2-008): finite inputs with components outside
        // ~[1e-19, 1e19] overflow Σv² to +Inf or collapse it to 0 under flush-to-zero; the
        // shared rescue (Metal4Common.h) recomputes the pair serially in the pre-scaled
        // normalization domain. Cold path — thread 0 only, and only for accumulator states the
        // cooperative fast path above cannot represent.
        if (va_cosine_accumulators_unreliable(dot_ab, dot_aa, dot_bb)) {
            float3 rescued = va_cosine_rescaled_terms(q, t, dimension);
            dot_ab = rescued.x;
            dot_aa = rescued.y;
            dot_bb = rescued.z;
        }

        // Shared finalization (Metal4Common.h): two-stage precise::divide — the single
        // sqrt(a)*sqrt(b) product is the measured-UNSAFE form fast math reassociates into
        // sqrt(a·b) and overflows — FLT_MIN absolute floor for the zero-vector test (BE3 4.5
        // — a precision-relative 1e-8 wrongly rejects valid dense micro-vectors), and the
        // NaN-propagating clamp to [-1, 1] so FP drift past ±1 can't produce a negative
        // "1 - similarity" distance.
        float similarity = va_cosine_similarity_finalize(dot_ab, dot_aa, dot_bb);

        similarities[query_idx] = (output_distance != 0) ? (1.0f - similarity) : similarity;
    }
}

