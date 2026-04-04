# Kernel Spec: Optimized K-Means Assignment (Tiled)

## Objective
Implement a high-performance K-Means assignment kernel in `Sources/VectorAccelerate/Metal/Shaders/ClusteringShaders.metal` that uses tiled shared memory to reduce global memory reads.

## Current State
The current `kmeans_assign_points` kernel performs a naive L2 distance check for every vector against every centroid. This is memory-bound and incurs redundant centroid reads from global memory.

## Target Specification
1.  **Shared Memory Centroid Tiling:**
    *   Load a subset of centroids (e.g., 32 centroids) into `threadgroup float tile_centroids[32 * dimension]` shared memory.
    *   Each thread within the threadgroup then calculates distances from its vector to all 32 centroids in the tile, entirely from shared memory.
    *   **Synchronization:** Use `threadgroup_barrier(mem_flags::mem_threadgroup)` after loading the tile.
    *   **Iteration:** Iterate through all centroids in tiles of size 32.

2.  **Top-1 Minimization (Argmin):**
    *   Track the `min_distance` and `min_centroid_id` across all tiles.
    *   Finalize by writing exactly one `UInt32` to the `assignments` buffer and one `Float` to the `distances` buffer per vector.

3.  **Optimization Hint:**
    *   Utilize `float4` for vector-centroid dot product calculations where possible.
    *   Ensure threadgroups align with SIMD-group width (32) for coalesced vector reads.

## Swift Dispatch Requirements
*   **Kernel Class:** `Sources/VectorAccelerate/Index/Kernels/Clustering/KMeansAssignKernel.swift`.
*   **Grid Sizing:** Grid should be 2D: `(numVectors, 1)`. 
*   **Threadgroup Size:** Calculate dynamically as `MTLSize(w, 1, 1)` where `w` is the execution width.
