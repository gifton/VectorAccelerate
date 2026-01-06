//
//  UMAPGradient.metal
//  VectorAccelerate
//
//  GPU kernels for UMAP gradient computation using segmented reduction.
//
//  Phase 1: Core kernels ✅
//  Phase 2: Optimization API ✅
//  Phase 3: GPU Target Gradient Accumulation ✅
//
//  Kernels:
//  - umap_edge_gradient_kernel: Compute per-edge gradients (attractive force)
//  - umap_segment_reduce_kernel: Reduce edge gradients to point gradients
//  - umap_apply_gradient_kernel: Apply gradients to embedding
//  - umap_negative_sample_kernel: Compute repulsive gradients from negative samples
//  - umap_accumulate_target_gradients_kernel: Atomically accumulate target gradients

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameter Structures

#ifndef VA_UMAP_PARAMS_DEFINED
#define VA_UMAP_PARAMS_DEFINED

struct UMAPParams {
    float a;              // Curve parameter a (default: 1.929)
    float b;              // Curve parameter b (default: 0.7915)
    float learningRate;   // Learning rate
    float epsilon;        // Small constant for numerical stability (0.001)
    uint n;               // Number of points
    uint d;               // Embedding dimension (typically 2-50)
    uint edgeCount;       // Number of edges
    uint negSampleRate;   // Negative samples per point
};

struct UMAPEdge {
    uint source;
    uint target;
    float weight;
};

#endif // VA_UMAP_PARAMS_DEFINED

// MARK: - Kernel 1: Edge Gradient Computation

/// Computes attractive gradients for all edges in parallel.
///
/// Each thread processes one edge and computes the gradient contribution
/// for both source and target points.
///
/// Output:
/// - edgeGradients: Gradient contribution for the source point
/// - targetGradients: Gradient contribution for the target point (negated)
kernel void umap_edge_gradient_kernel(
    device const float* embedding       [[buffer(0)]],  // [N, D]
    device const UMAPEdge* edges        [[buffer(1)]],  // [E]
    device float* edgeGradients         [[buffer(2)]],  // [E, D] grads for source
    device float* targetGradients       [[buffer(3)]],  // [E, D] grads for target (negated)
    constant UMAPParams& params         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.edgeCount) return;

    UMAPEdge edge = edges[tid];
    uint i = edge.source;
    uint j = edge.target;
    float weight = edge.weight;

    // Compute squared distance in low-dim space
    float distSq = 0.0f;
    for (uint k = 0; k < params.d; k++) {
        float diff = embedding[i * params.d + k] - embedding[j * params.d + k];
        distSq = fma(diff, diff, distSq);
    }

    // Attractive gradient coefficient
    // grad_coeff = -2ab × d^(2b-2) / (1 + a×d^(2b)) × weight × lr
    //
    // Note: d^(2b-2) = (d²)^(b-1) and d^(2b) = (d²)^b
    float distSqPowB = pow(max(distSq, params.epsilon), params.b);
    float distSqPowBm1 = pow(max(distSq, params.epsilon), params.b - 1.0f);
    float denom = 1.0f + params.a * distSqPowB;

    float gradCoeff = -2.0f * params.a * params.b * distSqPowBm1 / denom;
    gradCoeff *= weight * params.learningRate;

    // Clamp gradient coefficient to prevent numerical issues
    gradCoeff = clamp(gradCoeff, -4.0f, 4.0f);

    // Compute and store gradient for this edge
    for (uint k = 0; k < params.d; k++) {
        float diff = embedding[i * params.d + k] - embedding[j * params.d + k];
        float grad = gradCoeff * diff;
        edgeGradients[tid * params.d + k] = grad;
        targetGradients[tid * params.d + k] = -grad;  // Newton's third law
    }
}

// MARK: - Kernel 2: Segmented Reduction

/// Reduces per-edge gradients to per-point gradients using segment information.
///
/// Each thread handles one point and sums all edge gradients in its segment.
/// Edges must be sorted by source for this to work correctly.
kernel void umap_segment_reduce_kernel(
    device const float* edgeGradients   [[buffer(0)]],  // [E, D]
    device const uint* segmentStarts    [[buffer(1)]],  // [N]
    device const uint* segmentCounts    [[buffer(2)]],  // [N]
    device float* pointGradients        [[buffer(3)]],  // [N, D]
    constant UMAPParams& params         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    uint start = segmentStarts[tid];
    uint count = segmentCounts[tid];

    // Initialize gradient to zero
    for (uint k = 0; k < params.d; k++) {
        pointGradients[tid * params.d + k] = 0.0f;
    }

    // Sum all edge gradients in this segment
    for (uint e = 0; e < count; e++) {
        uint edgeIdx = start + e;
        for (uint k = 0; k < params.d; k++) {
            pointGradients[tid * params.d + k] += edgeGradients[edgeIdx * params.d + k];
        }
    }
}

// MARK: - Kernel 3: Apply Gradients

/// Applies accumulated gradients to the embedding.
///
/// Processes N×D elements in parallel (one thread per element).
kernel void umap_apply_gradient_kernel(
    device float* embedding             [[buffer(0)]],  // [N, D] in/out
    device const float* gradients       [[buffer(1)]],  // [N, D]
    constant UMAPParams& params         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n * params.d) return;

    embedding[tid] += gradients[tid];
}

// MARK: - Kernel 4: Negative Sampling

/// Computes repulsive gradients from random negative samples.
///
/// Each point is pushed away from randomly selected non-neighbor points.
/// Updates are applied directly to embedding (no accumulation needed since
/// each point has its own unique set of negative samples).
kernel void umap_negative_sample_kernel(
    device float* embedding             [[buffer(0)]],  // [N, D]
    device const uint* randomTargets    [[buffer(1)]],  // [N × negRate] random indices
    constant UMAPParams& params         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.n) return;

    for (uint s = 0; s < params.negSampleRate; s++) {
        uint j = randomTargets[tid * params.negSampleRate + s];
        if (j == tid) continue;  // Skip self

        // Compute squared distance
        float distSq = 0.0f;
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            distSq = fma(diff, diff, distSq);
        }

        // Repulsive gradient coefficient
        // grad_coeff = 2b / ((ε + d²) × (1 + a×d^(2b))) × lr
        float distSqPowB = pow(max(distSq, params.epsilon), params.b);
        float denom = (params.epsilon + distSq) * (1.0f + params.a * distSqPowB);
        float gradCoeff = 2.0f * params.b / denom * params.learningRate;

        // Clamp gradient coefficient
        gradCoeff = clamp(gradCoeff, -4.0f, 4.0f);

        // Apply repulsive gradient directly
        for (uint k = 0; k < params.d; k++) {
            float diff = embedding[tid * params.d + k] - embedding[j * params.d + k];
            embedding[tid * params.d + k] += gradCoeff * diff;
        }
    }
}

// MARK: - Kernel 5: Target Gradient Accumulation

/// Accumulates target gradients using atomic adds.
///
/// This kernel atomically adds each edge's target gradient to the corresponding
/// target point. While atomics have contention, this is typically faster than
/// CPU-side accumulation for large edge counts.
///
/// For very high-contention scenarios (many edges to same target), consider
/// the segment-reduce approach used for source gradients.
kernel void umap_accumulate_target_gradients_kernel(
    device const float* targetGradients   [[buffer(0)]],  // [E, D] per-edge target grads
    device const UMAPEdge* edges          [[buffer(1)]],  // [E]
    device atomic_float* pointGradients   [[buffer(2)]],  // [N, D] accumulated output
    constant UMAPParams& params           [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.edgeCount) return;

    UMAPEdge edge = edges[tid];
    uint j = edge.target;  // Target point index

    // Atomically accumulate gradient for target point
    for (uint k = 0; k < params.d; k++) {
        float grad = targetGradients[tid * params.d + k];
        atomic_fetch_add_explicit(
            &pointGradients[j * params.d + k],
            grad,
            memory_order_relaxed
        );
    }
}
