# Kernel Spec: Tiled Neural Encoder (GEMM with Weight Caching)

## Objective
Implement a high-performance neural encoding kernel in `Sources/VectorAccelerate/Metal/Shaders/NeuralQuantization.metal` that uses **Tiled Matrix Multiplication** to achieve a 10-50x throughput improvement by caching weights in threadgroup memory.

## Project Context (Version 0.4.1)
*   **Target:** Apple Silicon (M1-M4), Metal 4, Unified Memory.
*   **Buffer Strategy:** Uses `BufferPool` with `BufferToken`. Memory is recycled asynchronously.
*   **Lifecycle:** You MUST use `.keepAlive(until: commandBuffer)` for any intermediate buffers.
*   **Constraint:** Threadgroup memory is strictly **32KB**. Vectors are typically 768d or 1536d.

## The Bottleneck
The current `neural_encode_quantize_kernel` is memory-bound. Every input vector independently reads the entire weight matrix (e.g., 393KB for 768→128). For 10,000 vectors, this is **~4GB** of redundant weight reads. 

## Target Specification: Tiled GEMM
The encoder performs: `Output[N, L] = Input[N, D] × Weights[L, D]^T`.

### 1. Cooperative Weight Tiling
*   **Tile Strategy:** Partition the `L` (latent) and `D` (input) dimensions into tiles (e.g., 16x64).
*   **Cooperative Load:** Threads in the threadgroup must collaboratively load a weight tile into `threadgroup float weightTile[16][64]` (4KB).
*   **Synchronization:** Use `threadgroup_barrier(mem_flags::mem_threadgroup)` after every tile load.

### 2. Register-Cached Accumulation
*   **Vector Batching:** Each threadgroup should process a batch of 32 vectors simultaneously.
*   **Local Sums:** Each thread handles a subset of the latent dimensions for its assigned vector, accumulating results in local registers to minimize shared memory bank conflicts.

### 3. Quantization & Activation (2-Pass Orchestration)
*   **ReLU:** Apply ReLU activation after the full dot product is accumulated.
*   **Pass 2 (Normalize/Quantize):** Similar to our KMeans update, use a second lightweight pass to compute the per-vector `max_abs` scale and quantize the resulting floats into `int8_t` (latent codes).

## Swift Dispatch Requirements
*   **Kernel Class:** `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`.
*   **Dynamic Tile Calculation:**
    ```swift
    let maxSharedMem = 32768 // 32KB
    let weightTileSize = tileL * tileD * 4
    let accSize = vectorsPerTG * tileL * 4
    // Ensure weightTileSize + accSize < maxSharedMem
    ```
*   **Buffer Cleaning:** You must use `blitEncoder.fill` to zero out any intermediate accumulation buffers before dispatching Pass 1.

## Performance Hint
Use `float4` for all input vector and weight tile loads. Apple Silicon's 128-bit memory bus is most efficient when threads perform coalesced 16-byte reads.
