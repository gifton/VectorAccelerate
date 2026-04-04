# Kernel Spec: SIMD-Accelerated Distance Metrics (L2, Cosine)

## Objective
Refactor existing distance kernels in `Sources/VectorAccelerate/Metal/Shaders/DistanceShaders.metal` to replace per-thread atomic operations with hierarchical SIMD reductions.

## Current State
Distance kernels like `l2_distance` currently calculate their squared differences and then attempt to write to a global distance matrix using `atomic_fetch_add_explicit`. This is extremely slow for large-batch queries.

## Target Specification
1.  **L2 Distance Optimization:**
    *   **Phase 1:** Calculate squared difference per thread (`float diff = a[i] - b[i]; float sq_diff = diff * diff;`).
    *   **Phase 2:** Sum within the SIMD-group (32 threads) using `float simd_result = simd_sum(sq_diff);`.
    *   **Phase 3:** Use a single thread per SIMD-group (`if (simd_lane_id == 0)`) to consolidate the `simd_result` into a `threadgroup float shared_sums[32]`.
    *   **Phase 4:** Perform a final reduction across the threadgroup and write the total sum to global memory **exactly once** per threadgroup.

2.  **Cosine Similarity Optimization:**
    *   Implement similar three-tier reductions for the three separate dot products (A·B, A·A, B·B) needed for the cosine calculation.
    *   **Optimization Hint:** Use `float3` or `float4` SIMD types for vector loads to maximize memory bus utilization.

## Swift Dispatch Requirements
*   **Kernel Class:** `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift` (and Cosine variant).
*   **Grid Sizing:** The grid should be 1D, partitioned by queries and vector dimension. 
*   **Execution Width:** Use the `pipelineState.threadExecutionWidth` (always 32 on Apple Silicon) as the `threadsPerThreadgroup.x` value.
