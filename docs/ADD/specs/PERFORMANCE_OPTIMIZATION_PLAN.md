# VectorAccelerate Performance Optimization Plan

**Status:** Planned  
**Target:** VectorAccelerate Core & Critical Path Kernels  

## Executive Summary
This plan outlines a focused refactoring of VectorAccelerate to address command buffer/pipeline thrashing, suboptimal threadgroup sizing/reductions, and synchronous GPU blocking. Following the constraint that breaking changes to existing classes are acceptable provided test coverage is maintained, we will focus initially on the critical paths (core distance metrics, clustering) to minimize risk while delivering the highest impact.

## Phase 1: Enforce Asynchronous GPU Execution (Finding 3)
**Goal:** Eliminate all synchronous GPU blocking to allow Swift Concurrency to properly suspend and free up CPU threads.
*   **Audit & Replace `commit()`:** Direct `.commit()` calls exist in `Metal4Context`, `KMeansAssignKernel`, `QuantizationStatisticsKernel`, and `LearnedDistanceKernel`. These must be updated to use the existing `commitAndWait()` shim or `Metal4Context.executeAndWait()` which wrap the commit in a `withCheckedContinuation`.
*   **Remove `waitUntilCompleted()`:** Ensure absolutely no `waitUntilCompleted()` calls exist in Swift source code (it appears mostly clean already, but tests and legacy operations will be strictly audited).
*   **Tie Buffer Releases to Completion:** Update the `BufferToken` lifecycle. Currently, buffers return to the pool on `deinit`. We will update the architecture so the buffer is securely returned to the pool *only after* the `commandBuffer.addCompletedHandler` fires, preventing race conditions or CPU-side overwrites while the GPU is executing.

## Phase 2: MetalContext & Buffer Pooling Refactor (Finding 1)
**Goal:** Prevent GPU initialization latency and driver-level allocation bottlenecks.
*   **Pre-compilation in `Metal4Context`:** Refactor `Metal4Context` (and `ArchivePipelineCache` / `Metal4ShaderCompiler`) to eagerly pre-compile and cache the critical path `.metallib` pipelines (e.g., L2, Cosine, TopK, KMeans, IVF) during framework initialization, rather than compiling them lazily when kernels are initialized.
*   **Buffer Pool Enhancement:** Modify the existing `BufferPool` to function strictly as a robust object pool (Ring Buffer). Instead of calling `device.makeBuffer()` in hot loops (currently found excessively in `BatchMaxKernel`, `KMeans`, and `IVF` kernels), we will enforce that all kernels request pre-allocated `.storageModeShared` buffers from the `Metal4Context.bufferPool`.

## Phase 3: SIMD-Group Reductions & Threadgroup Sizing (Finding 2)
**Goal:** Maximize Apple Silicon GPU core utilization and eliminate atomic bottlenecks. Focus will be strictly on the critical path kernels (`DistanceShaders.metal`, `ClusteringShaders.metal`, `BasicOperations.metal`).
*   **Threadgroup Dispatch Calculation:** Update the Swift kernel dispatch logic (e.g., inside `L2DistanceKernel`, `CosineSimilarityKernel`) to dynamically calculate grid sizes. We will utilize:
    ```swift
    let w = pipelineState.threadExecutionWidth
    let h = pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    ```
*   **SIMD `simd_sum()` Replacements:** Refactor the `.metal` shaders in the critical path. We will replace slow `atomic_fetch_add_explicit` operations (currently over 90 instances across the codebase) with native `simd_sum()` operations and cooperative threadgroup memory reductions. This will exponentially speed up distance accumulations and K-Means centroid updates.

## Execution Order
1.  **Phase 1** (Enforcing `commitAndWait`) should be executed first as it is your stated top priority and establishes the correct concurrency foundation.
2.  **Phase 2** (Pipeline & Buffer pooling) follows, as fixing memory allocations will immediately stabilize latency spikes across all operations.
3.  **Phase 3** (SIMD & Threadgroups) will be executed last, focused strictly on the critical path distance and clustering kernels to manage the scope of shader rewrites.

## Success Criteria
*   No `.commit()` calls without an associated `.addCompletedHandler` and continuation.
*   No `device.makeBuffer` calls inside the `execute` or `compute` loops of critical path kernels.
*   All test suites pass successfully without a drop in coverage.