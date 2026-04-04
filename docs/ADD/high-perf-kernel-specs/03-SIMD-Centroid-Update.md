# Kernel Spec: SIMD-Based K-Means Centroid Update

## Objective
Implement a high-performance K-Means centroid update kernel in `Sources/VectorAccelerate/Metal/Shaders/ClusteringShaders.metal` using hierarchical SIMD reductions.

## Current State
`KMeansUpdateKernel.swift` currently performs most of its math on the CPU or uses simple individual atomic writes. This is a massive bottleneck for high-dimensional vectors (e.g., 768d or 1536d).

## Target Specification
1.  **SIMD-Accelerated Mean Calculation:**
    *   **Phase 1:** Sum the elements of all vectors assigned to a given cluster (`centroid_id`).
    *   **Phase 2:** Sum within the SIMD-group (32 threads) using `float simd_result = simd_sum(vector_dim_value);`.
    *   **Phase 3:** Write to a `threadgroup float shared_dim_sums[32]`.
    *   **Phase 4:** Perform exactly one `atomic_fetch_add` per dimension *per threadgroup*.

2.  **Shared Memory Counter:**
    *   Keep an atomic counter in shared memory for the number of vectors per cluster assigned within the threadgroup.
    *   Sync only the final threadgroup counter to global memory once.

3.  **Optimization Hint:**
    *   Use `float4` for vector accumulation where possible to reduce memory transactions.
    *   Consider a 2-pass approach: 
        *   Pass 1: Accumulate sums and counts. 
        *   Pass 2: Divide final sums by final counts to compute the mean centroids.

## Swift Dispatch Requirements
*   **Kernel Class:** `Sources/VectorAccelerate/Index/Kernels/Clustering/KMeansUpdateKernel.swift`.
*   **Grid Sizing:** Grid should be 2D: `(numClusters, dimension)`.
*   **Threadgroup Size:** Optimize for dimension (e.g., `MTLSize(32, 8, 1)` if dimension allows).
