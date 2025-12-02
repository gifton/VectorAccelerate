// VectorAccelerate: Distance Computation Shaders
//
// GPU kernels for various distance metrics
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Manhattan Distance

kernel void manhattanDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    // Each thread computes partial sum
    threadgroup float partial_sums[256];
    uint tid = id % 256;
    
    float local_sum = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        local_sum += abs(vectorA[i] - vectorB[i]);
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    if (tid == 0) {
        float total = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total += partial_sums[i];
        }
        atomic_store_explicit((device atomic_float*)result, total, memory_order_relaxed);
    }
}

// MARK: - Chebyshev Distance

kernel void chebyshevDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup float partial_max[256];
    uint tid = id % 256;
    
    float local_max = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        local_max = max(local_max, abs(vectorA[i] - vectorB[i]));
    }
    
    partial_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to find maximum
    if (tid == 0) {
        float maximum = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            maximum = max(maximum, partial_max[i]);
        }
        atomic_store_explicit((device atomic_float*)result, maximum, memory_order_relaxed);
    }
}

// MARK: - Minkowski Distance

kernel void minkowskiDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant float& p [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup float partial_sums[256];
    uint tid = id % 256;
    
    float local_sum = 0.0;
    for (uint i = id; i < dimension; i += 256) {
        float diff = abs(vectorA[i] - vectorB[i]);
        local_sum += pow(diff, p);
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        float total = 0.0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total += partial_sums[i];
        }
        float final_result = pow(total, 1.0 / p);
        atomic_store_explicit((device atomic_float*)result, final_result, memory_order_relaxed);
    }
}

// MARK: - Hamming Distance

kernel void hammingDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup uint partial_counts[256];
    uint tid = id % 256;
    
    uint local_count = 0;
    for (uint i = id; i < dimension; i += 256) {
        // Treat as binary: non-zero values are 1
        bool a_bit = (vectorA[i] != 0.0);
        bool b_bit = (vectorB[i] != 0.0);
        if (a_bit != b_bit) {
            local_count++;
        }
    }
    
    partial_counts[tid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        uint total = 0;
        uint limit = min(256u, dimension);
        for (uint i = 0; i < limit; i++) {
            total += partial_counts[i];
        }
        atomic_store_explicit((device atomic_float*)result, float(total), memory_order_relaxed);
    }
}

// MARK: - Jaccard Distance

kernel void jaccardDistance(
    constant float* vectorA [[buffer(0)]],
    constant float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dimension) return;
    
    threadgroup uint intersection_counts[256];
    threadgroup uint union_counts[256];
    uint tid = id % 256;
    
    uint local_intersection = 0;
    uint local_union = 0;
    
    for (uint i = id; i < dimension; i += 256) {
        bool a_present = (vectorA[i] != 0.0);
        bool b_present = (vectorB[i] != 0.0);
        
        if (a_present && b_present) {
            local_intersection++;
        }
        if (a_present || b_present) {
            local_union++;
        }
    }
    
    intersection_counts[tid] = local_intersection;
    union_counts[tid] = local_union;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        uint total_intersection = 0;
        uint total_union = 0;
        uint limit = min(256u, dimension);
        
        for (uint i = 0; i < limit; i++) {
            total_intersection += intersection_counts[i];
            total_union += union_counts[i];
        }
        
        float jaccard_sim = (total_union > 0) ? 
            float(total_intersection) / float(total_union) : 0.0;
        float jaccard_dist = 1.0 - jaccard_sim;
        
        atomic_store_explicit((device atomic_float*)result, jaccard_dist, memory_order_relaxed);
    }
}

// MARK: - Batch Distance Operations

kernel void batchManhattanDistance(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& candidateCount [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint candidateIdx = id.x;
    if (candidateIdx >= candidateCount) return;
    
    uint candidateOffset = candidateIdx * dimension;
    float distance = 0.0;
    
    for (uint d = 0; d < dimension; d++) {
        distance += abs(query[d] - candidates[candidateOffset + d]);
    }
    
    distances[candidateIdx] = distance;
}

// MARK: - Optimized Euclidean with SIMD

kernel void euclideanDistanceSIMD(
    constant float4* vectorA [[buffer(0)]],
    constant float4* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension4 [[buffer(3)]],  // dimension / 4
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[256];
    
    float local_sum = 0.0;
    uint threads_per_group = tg_size;
    
    for (uint i = tid; i < dimension4; i += threads_per_group) {
        float4 diff = vectorA[i] - vectorB[i];
        float4 squared = diff * diff;
        local_sum += squared.x + squared.y + squared.z + squared.w;
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        *result = sqrt(partial_sums[0]);
    }
}

// MARK: - Optimized Cosine with SIMD

kernel void cosineDistanceSIMD(
    constant float4* vectorA [[buffer(0)]],
    constant float4* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float dot_products[256];
    threadgroup float norm_a_sq[256];
    threadgroup float norm_b_sq[256];
    
    float local_dot = 0.0;
    float local_norm_a = 0.0;
    float local_norm_b = 0.0;
    uint threads_per_group = tg_size;
    
    for (uint i = tid; i < dimension4; i += threads_per_group) {
        float4 a = vectorA[i];
        float4 b = vectorB[i];
        
        float4 products = a * b;
        local_dot += products.x + products.y + products.z + products.w;
        
        float4 a_squared = a * a;
        local_norm_a += a_squared.x + a_squared.y + a_squared.z + a_squared.w;
        
        float4 b_squared = b * b;
        local_norm_b += b_squared.x + b_squared.y + b_squared.z + b_squared.w;
    }
    
    dot_products[tid] = local_dot;
    norm_a_sq[tid] = local_norm_a;
    norm_b_sq[tid] = local_norm_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for all three values
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            dot_products[tid] += dot_products[tid + stride];
            norm_a_sq[tid] += norm_a_sq[tid + stride];
            norm_b_sq[tid] += norm_b_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        float dot = dot_products[0];
        float norm_a = sqrt(norm_a_sq[0]);
        float norm_b = sqrt(norm_b_sq[0]);
        
        float cosine_sim = (norm_a > 0.0 && norm_b > 0.0) ? 
            dot / (norm_a * norm_b) : 0.0;
        *result = 1.0 - cosine_sim;
    }
}

// MARK: - Optimized Batch Operations with SIMD

/// High-performance batch cosine similarity using SIMD
kernel void batchCosineSimilaritySIMD(
    constant float4* query [[buffer(0)]],           // Query vector (float4 packed)
    constant float4* candidates [[buffer(1)]],      // Candidate vectors (float4 packed)
    device float* similarities [[buffer(2)]],       // Output similarities
    constant uint& dimension4 [[buffer(3)]],        // dimension / 4
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    
    // Calculate offset for this candidate
    uint candidateOffset = id * dimension4;
    
    float4 dot_accum = 0.0;
    float4 query_norm_accum = 0.0;
    float4 candidate_norm_accum = 0.0;
    
    // Process vectors in chunks of 4 elements
    for (uint i = 0; i < dimension4; i++) {
        float4 q = query[i];
        float4 c = candidates[candidateOffset + i];
        
        // Accumulate dot product and norms using SIMD
        dot_accum += q * c;
        query_norm_accum += q * q;
        candidate_norm_accum += c * c;
    }
    
    // Reduce float4 accumulators to scalars
    float dot_product = dot_accum.x + dot_accum.y + dot_accum.z + dot_accum.w;
    float query_norm = sqrt(query_norm_accum.x + query_norm_accum.y + 
                           query_norm_accum.z + query_norm_accum.w);
    float candidate_norm = sqrt(candidate_norm_accum.x + candidate_norm_accum.y + 
                               candidate_norm_accum.z + candidate_norm_accum.w);
    
    // Compute cosine similarity
    if (query_norm > 0.0 && candidate_norm > 0.0) {
        similarities[id] = dot_product / (query_norm * candidate_norm);
    } else {
        similarities[id] = 0.0;
    }
}

/// High-performance batch dot product using SIMD
kernel void batchDotProductSIMD(
    constant float4* query [[buffer(0)]],           // Query vector (float4 packed)
    constant float4* candidates [[buffer(1)]],      // Candidate vectors (float4 packed)
    device float* dotProducts [[buffer(2)]],        // Output dot products
    constant uint& dimension4 [[buffer(3)]],        // dimension / 4
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    
    // Calculate offset for this candidate
    uint candidateOffset = id * dimension4;
    
    float4 dot_accum = 0.0;
    
    // Process vectors in chunks of 4 elements
    for (uint i = 0; i < dimension4; i++) {
        float4 q = query[i];
        float4 c = candidates[candidateOffset + i];
        
        // SIMD multiply-add
        dot_accum += q * c;
    }
    
    // Reduce float4 to scalar
    dotProducts[id] = dot_accum.x + dot_accum.y + dot_accum.z + dot_accum.w;
}

/// High-performance batch Euclidean distance using SIMD
kernel void batchEuclideanDistanceSIMD(
    constant float4* query [[buffer(0)]],           // Query vector (float4 packed)
    constant float4* candidates [[buffer(1)]],      // Candidate vectors (float4 packed)
    device float* distances [[buffer(2)]],          // Output distances
    constant uint& dimension4 [[buffer(3)]],        // dimension / 4
    constant uint& candidateCount [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= candidateCount) return;
    
    // Calculate offset for this candidate
    uint candidateOffset = id * dimension4;
    
    float4 sum_squared = 0.0;
    
    // Process vectors in chunks of 4 elements
    for (uint i = 0; i < dimension4; i++) {
        float4 diff = query[i] - candidates[candidateOffset + i];
        sum_squared += diff * diff;
    }
    
    // Reduce and compute final distance
    float squared_distance = sum_squared.x + sum_squared.y + sum_squared.z + sum_squared.w;
    distances[id] = sqrt(squared_distance);
}