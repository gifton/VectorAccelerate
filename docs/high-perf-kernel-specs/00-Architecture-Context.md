# VectorAccelerate High-Performance Kernel Environment

## Overview
VectorAccelerate is a high-throughput vector database and mathematics library targeting **iOS 26+ and macOS 26+ (Metal 4 Only)**. This environment is strictly Apple Silicon-native and assumes a Unified Memory Architecture (UMA).

## Architectural Constraints
*   **Metal 4 Core:** We utilize `MTLSharedEvent` for synchronization and `MTLSharedEventListener` for Swift Task suspension. 
*   **Swift 6 Concurrency:** All stateful components are Actors. Kernels are `@unchecked Sendable` classes.
*   **Buffer Lifecycle:** We use a `BufferPool` (Ring Buffer) with `BufferToken` objects. 
    *   Memory is recycled when the `BufferToken` is deallocated.
    *   **Crucial:** To prevent premature recycling, you must capture the token in the `commandBuffer.addCompletedHandler` or ensure the Swift task is suspended via `await context.executeAndWait`.
*   **Memory Layout:** 
    *   Standard vectors are `Float` (FP32).
    *   Indices and Assignments are `UInt32`.
    *   Memory is strictly `.storageModeShared` for zero-copy CPU/GPU access.

## The Performance Bottleneck
The current implementation relies heavily on `atomic_fetch_add_explicit` for global reductions (e.g., calculating distance sums or centroid updates). On Apple Silicon, this causes significant pipeline stalls and cache-coherency overhead.

## Target Reduction Strategy
All new or refactored kernels must follow a two-tier reduction strategy:
1.  **SIMD-Group Level:** Use `simd_sum(value)` to reduce within a 32-thread warp (execution width).
2.  **Threadgroup Level:** Use `threadgroup float shared_data[32]` to consolidate results from up to 32 SIMD-groups.
3.  **Global Level:** Perform a single `atomic_fetch_add` per *threadgroup* (not per thread) to the final output buffer, or use a separate multi-pass reduction kernel for massive datasets.

## Dispatch Configuration
Swift-side dispatch should no longer use hardcoded threadgroup sizes. It must query the pipeline state:
```swift
let w = pipelineState.threadExecutionWidth
let h = pipelineState.maxTotalThreadsPerThreadgroup / w
let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
```
