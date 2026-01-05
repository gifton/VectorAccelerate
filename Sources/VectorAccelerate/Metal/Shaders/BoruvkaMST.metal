// VectorAccelerate: Boruvka's MST Kernels
//
// GPU kernels for Minimum Spanning Tree computation using Boruvka's algorithm.
// Designed for HDBSCAN clustering over mutual reachability distances.
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+
//
// Algorithm:
// 1. Each iteration processes ALL components in parallel
// 2. Each component finds its minimum outgoing edge
// 3. Edges are added to MST and components merged
// 4. ~log(N) iterations until single component remains
//
// Kernels:
// - boruvka_find_min_edge_kernel:     Find minimum outgoing edge per point
// - boruvka_component_reduce_kernel:  Reduce to per-component minimum
// - boruvka_merge_kernel:             Add edges to MST and merge components

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameter Structures

struct BoruvkaParams {
    uint n;              // Number of points
    uint d;              // Embedding dimension
    uint iteration;      // Current iteration (for debugging)
    uint _padding;       // Alignment padding
};

struct MSTEdge {
    uint source;
    uint target;
    float weight;
};

// MARK: - Kernel 1: Find Minimum Outgoing Edge

/// Each thread handles one point, finding the minimum-weight edge to a different component.
///
/// Computes mutual reachability on-the-fly: max(core_i, core_j, euclidean(i, j))
/// This avoids storing the O(N^2) distance matrix.
kernel void boruvka_find_min_edge_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, D] row-major
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current component assignments
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;  // Self means no valid edge found

    // Check all other points for minimum edge to different component
    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;  // Same component, skip

        // Compute L2 distance using vectorized accumulation
        float distSq = 0.0f;

        // Process 4 elements at a time for better memory throughput
        const uint simd_blocks = params.d / 4;
        const uint remainder = params.d % 4;

        device const float4* vec_i = (device const float4*)(embeddings + tid * params.d);
        device const float4* vec_j = (device const float4*)(embeddings + j * params.d);

        float4 acc = float4(0.0f);
        for (uint k = 0; k < simd_blocks; k++) {
            float4 diff = vec_i[k] - vec_j[k];
            acc = fma(diff, diff, acc);
        }
        distSq = acc.x + acc.y + acc.z + acc.w;

        // Handle remainder
        if (remainder > 0) {
            device const float* tail_i = embeddings + tid * params.d + (simd_blocks * 4);
            device const float* tail_j = embeddings + j * params.d + (simd_blocks * 4);
            for (uint k = 0; k < remainder; k++) {
                float diff = tail_i[k] - tail_j[k];
                distSq = fma(diff, diff, distSq);
            }
        }

        float dist = sqrt(distSq);

        // Mutual reachability = max(core_i, core_j, dist)
        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}

// MARK: - Kernel 2: Component-Level Reduce

/// Reduce per-point minimums to per-component minimums.
/// Only the "representative" of each component (smallest index) performs the reduction.
///
/// This ensures exactly one thread per component outputs the minimum edge.
kernel void boruvka_component_reduce_kernel(
    device const uint* componentIds         [[buffer(0)]],  // [N]
    device const float* pointMinWeight      [[buffer(1)]],  // [N] per-point min
    device const uint* pointMinTarget       [[buffer(2)]],  // [N] per-point target
    device float* componentMinWeight        [[buffer(3)]],  // [N] per-component min
    device uint* componentMinSource         [[buffer(4)]],  // [N] source of min edge
    device uint* componentMinTarget         [[buffer(5)]],  // [N] target of min edge
    constant BoruvkaParams& params          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];

    // Only the "representative" of each component performs the reduction
    // Representative = smallest index in component
    bool isRepresentative = true;
    for (uint i = 0; i < tid; i++) {
        if (componentIds[i] == myComponent) {
            isRepresentative = false;
            break;
        }
    }

    if (!isRepresentative) {
        componentMinWeight[tid] = INFINITY;
        return;
    }

    // Find minimum across all points in this component
    float bestWeight = INFINITY;
    uint bestSource = tid;
    uint bestTarget = tid;

    for (uint i = 0; i < params.n; i++) {
        if (componentIds[i] != myComponent) continue;
        if (pointMinWeight[i] < bestWeight) {
            bestWeight = pointMinWeight[i];
            bestSource = i;
            bestTarget = pointMinTarget[i];
        }
    }

    componentMinWeight[tid] = bestWeight;
    componentMinSource[tid] = bestSource;
    componentMinTarget[tid] = bestTarget;
}

// MARK: - Kernel 3: Collect Candidate Edges

/// Collect candidate edges from component representatives.
///
/// This kernel adds ALL candidate edges to the output buffer. Duplicate edges
/// are handled by the CPU merge step using Union-Find, which naturally skips
/// edges between already-connected components.
///
/// Key invariants:
/// - Only component representatives add edges
/// - Atomic edge count increment for thread-safe insertion
/// - Duplicates handled by CPU Union-Find merge
kernel void boruvka_merge_kernel(
    device const uint* componentIds         [[buffer(0)]],  // [N] read-only
    device const float* componentMinWeight  [[buffer(1)]],  // [N]
    device const uint* componentMinSource   [[buffer(2)]],  // [N]
    device const uint* componentMinTarget   [[buffer(3)]],  // [N]
    device MSTEdge* mstEdges                [[buffer(4)]],  // [N-1] output edges
    device atomic_uint* edgeCount           [[buffer(5)]],  // [1] current edge count
    constant BoruvkaParams& params          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    // Only representatives add edges (non-representatives have INFINITY weight)
    if (componentMinWeight[tid] == INFINITY) return;

    uint source = componentMinSource[tid];
    uint target = componentMinTarget[tid];
    float weight = componentMinWeight[tid];

    // Verify this edge connects different components
    uint sourceComp = componentIds[source];
    uint targetComp = componentIds[target];
    if (sourceComp == targetComp) return;  // Same component, skip

    // Add edge to candidate buffer atomically
    // Note: This may add duplicate edges (A→B and B→A), which are
    // deduplicated by the CPU Union-Find merge step
    uint idx = atomic_fetch_add_explicit(edgeCount, 1, memory_order_relaxed);
    mstEdges[idx].source = source;
    mstEdges[idx].target = target;
    mstEdges[idx].weight = weight;

    // NOTE: Component merging and deduplication done on CPU after this kernel.
}

// MARK: - Dimension-Optimized Find Min Edge Kernels

/// 384-dimensional variant (most common for sentence embeddings like MiniLM)
/// Processes 96 float4 vectors per distance computation with 4x unrolling
kernel void boruvka_find_min_edge_384_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, 384] row-major
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current component assignments
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    // Precompute base pointer for this point (384 = 96 float4)
    device const float4* vec_i = (device const float4*)(embeddings + tid * 384);

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* vec_j = (device const float4*)(embeddings + j * 384);

        // Unrolled 384-dim distance: 96 float4 iterations with 4x unrolling (24 iterations)
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);

        #pragma unroll
        for (uint k = 0; k < 24; k++) {
            uint base = k * 4;
            float4 d0 = vec_i[base + 0] - vec_j[base + 0];
            float4 d1 = vec_i[base + 1] - vec_j[base + 1];
            float4 d2 = vec_i[base + 2] - vec_j[base + 2];
            float4 d3 = vec_i[base + 3] - vec_j[base + 3];
            acc0 = fma(d0, d0, acc0);
            acc1 = fma(d1, d1, acc1);
            acc2 = fma(d2, d2, acc2);
            acc3 = fma(d3, d3, acc3);
        }

        float4 sum = acc0 + acc1 + acc2 + acc3;
        float distSq = sum.x + sum.y + sum.z + sum.w;
        float dist = sqrt(distSq);

        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}

/// 512-dimensional variant (common for some BERT variants)
/// Processes 128 float4 vectors with 4x unrolling (32 iterations)
kernel void boruvka_find_min_edge_512_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, 512] row-major
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current component assignments
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    // Precompute base pointer for this point (512 = 128 float4)
    device const float4* vec_i = (device const float4*)(embeddings + tid * 512);

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* vec_j = (device const float4*)(embeddings + j * 512);

        // Unrolled 512-dim distance: 128 float4 iterations with 4x unrolling (32 iterations)
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);

        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            uint base = k * 4;
            float4 d0 = vec_i[base + 0] - vec_j[base + 0];
            float4 d1 = vec_i[base + 1] - vec_j[base + 1];
            float4 d2 = vec_i[base + 2] - vec_j[base + 2];
            float4 d3 = vec_i[base + 3] - vec_j[base + 3];
            acc0 = fma(d0, d0, acc0);
            acc1 = fma(d1, d1, acc1);
            acc2 = fma(d2, d2, acc2);
            acc3 = fma(d3, d3, acc3);
        }

        float4 sum = acc0 + acc1 + acc2 + acc3;
        float distSq = sum.x + sum.y + sum.z + sum.w;
        float dist = sqrt(distSq);

        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}

/// 768-dimensional variant (BERT-base, DistilBERT, MPNet)
/// Processes 192 float4 vectors with 4x unrolling (48 iterations)
kernel void boruvka_find_min_edge_768_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, 768] row-major
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current component assignments
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    // Precompute base pointer for this point (768 = 192 float4)
    device const float4* vec_i = (device const float4*)(embeddings + tid * 768);

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* vec_j = (device const float4*)(embeddings + j * 768);

        // Unrolled 768-dim distance: 192 float4 iterations with 4x unrolling (48 iterations)
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);

        #pragma unroll
        for (uint k = 0; k < 48; k++) {
            uint base = k * 4;
            float4 d0 = vec_i[base + 0] - vec_j[base + 0];
            float4 d1 = vec_i[base + 1] - vec_j[base + 1];
            float4 d2 = vec_i[base + 2] - vec_j[base + 2];
            float4 d3 = vec_i[base + 3] - vec_j[base + 3];
            acc0 = fma(d0, d0, acc0);
            acc1 = fma(d1, d1, acc1);
            acc2 = fma(d2, d2, acc2);
            acc3 = fma(d3, d3, acc3);
        }

        float4 sum = acc0 + acc1 + acc2 + acc3;
        float distSq = sum.x + sum.y + sum.z + sum.w;
        float dist = sqrt(distSq);

        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}

/// 1536-dimensional variant (OpenAI ada-002)
/// Processes 384 float4 vectors with 4x unrolling (96 iterations)
kernel void boruvka_find_min_edge_1536_kernel(
    device const float* embeddings          [[buffer(0)]],  // [N, 1536] row-major
    device const float* coreDistances       [[buffer(1)]],  // [N]
    device const uint* componentIds         [[buffer(2)]],  // [N] current component assignments
    device float* minEdgeWeight             [[buffer(3)]],  // [N] output: min weight per point
    device uint* minEdgeTarget              [[buffer(4)]],  // [N] output: target of min edge
    constant BoruvkaParams& params          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint myComponent = componentIds[tid];
    float myCore = coreDistances[tid];

    float bestWeight = INFINITY;
    uint bestTarget = tid;

    // Precompute base pointer for this point (1536 = 384 float4)
    device const float4* vec_i = (device const float4*)(embeddings + tid * 1536);

    for (uint j = 0; j < params.n; j++) {
        if (componentIds[j] == myComponent) continue;

        device const float4* vec_j = (device const float4*)(embeddings + j * 1536);

        // Unrolled 1536-dim distance: 384 float4 iterations with 4x unrolling (96 iterations)
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);

        #pragma unroll
        for (uint k = 0; k < 96; k++) {
            uint base = k * 4;
            float4 d0 = vec_i[base + 0] - vec_j[base + 0];
            float4 d1 = vec_i[base + 1] - vec_j[base + 1];
            float4 d2 = vec_i[base + 2] - vec_j[base + 2];
            float4 d3 = vec_i[base + 3] - vec_j[base + 3];
            acc0 = fma(d0, d0, acc0);
            acc1 = fma(d1, d1, acc1);
            acc2 = fma(d2, d2, acc2);
            acc3 = fma(d3, d3, acc3);
        }

        float4 sum = acc0 + acc1 + acc2 + acc3;
        float distSq = sum.x + sum.y + sum.z + sum.w;
        float dist = sqrt(distSq);

        float mutualReach = max(max(myCore, coreDistances[j]), dist);

        if (mutualReach < bestWeight) {
            bestWeight = mutualReach;
            bestTarget = j;
        }
    }

    minEdgeWeight[tid] = bestWeight;
    minEdgeTarget[tid] = bestTarget;
}
