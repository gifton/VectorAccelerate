//
//  BatchMax.metal
//  VectorAccelerate
//
//  GPU kernels for element-wise maximum operations.
//
//  Kernels:
//  - batch_max3_kernel: Element-wise max(a, b, c) - scalar version
//  - batch_max3_vectorized_kernel: Element-wise max(a, b, c) - float4 vectorized
//  - batch_max2_kernel: Element-wise max(a, b)
//  - batch_max_inplace_kernel: In-place a[i] = max(a[i], b[i])
//
//  Primary use case: Mutual reachability distance computation
//    mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Three-Array Maximum

/// Element-wise maximum of three arrays.
///
/// output[i] = max(a[i], b[i], c[i])
///
/// Used for mutual reachability: max(core_a, core_b, dist)
kernel void batch_max3_kernel(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* c       [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& count        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = max(max(a[tid], b[tid]), c[tid]);
}

/// Vectorized version for better memory throughput.
///
/// Processes 4 elements per thread using float4.
/// Requires count to be divisible by 4.
kernel void batch_max3_vectorized_kernel(
    device const float4* a      [[buffer(0)]],
    device const float4* b      [[buffer(1)]],
    device const float4* c      [[buffer(2)]],
    device float4* output       [[buffer(3)]],
    constant uint& count4       [[buffer(4)]],  // count / 4
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count4) return;
    output[tid] = max(max(a[tid], b[tid]), c[tid]);
}

// MARK: - Two-Array Maximum

/// Element-wise maximum of two arrays.
///
/// output[i] = max(a[i], b[i])
kernel void batch_max2_kernel(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = max(a[tid], b[tid]);
}

/// Vectorized two-array maximum.
kernel void batch_max2_vectorized_kernel(
    device const float4* a      [[buffer(0)]],
    device const float4* b      [[buffer(1)]],
    device float4* output       [[buffer(2)]],
    constant uint& count4       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count4) return;
    output[tid] = max(a[tid], b[tid]);
}

// MARK: - In-Place Operations

/// In-place maximum: a[i] = max(a[i], b[i])
kernel void batch_max_inplace_kernel(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    a[tid] = max(a[tid], b[tid]);
}

/// In-place three-way maximum: a[i] = max(a[i], b[i], c[i])
kernel void batch_max3_inplace_kernel(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* c       [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    a[tid] = max(max(a[tid], b[tid]), c[tid]);
}
