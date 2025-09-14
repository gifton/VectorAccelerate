// VectorAccelerate: Basic Operations Shaders
//
// Core GPU kernels for fundamental vector and matrix operations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

constant float EPSILON = 1e-7f;

// MARK: - Basic Distance Operations

/// Compute Euclidean distance between two vectors
/// Uses parallel reduction for optimal performance
kernel void euclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = sqrt(partialSums[0]);
    }
}

/// Compute squared Euclidean distance (no sqrt for performance)
kernel void squaredEuclideanDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = partialSums[0];
    }
}

/// Compute cosine distance between two vectors
/// Returns 1 - cosine_similarity for distance metric
kernel void cosineDistance(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float dotProducts[256];
    threadgroup float normA[256];
    threadgroup float normB[256];
    
    float localDot = 0.0f;
    float localNormA = 0.0f;
    float localNormB = 0.0f;
    
    for (uint i = tid; i < dimension; i += tgSize) {
        float a = vectorA[i];
        float b = vectorB[i];
        localDot += a * b;
        localNormA += a * a;
        localNormB += b * b;
    }
    
    dotProducts[tid] = localDot;
    normA[tid] = localNormA;
    normB[tid] = localNormB;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for all three values
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            dotProducts[tid] += dotProducts[tid + stride];
            normA[tid] += normA[tid + stride];
            normB[tid] += normB[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        float dot = dotProducts[0];
        float magA = sqrt(normA[0]);
        float magB = sqrt(normB[0]);
        
        if (magA > EPSILON && magB > EPSILON) {
            float cosineSim = dot / (magA * magB);
            result[0] = 1.0f - cosineSim;
        } else {
            result[0] = 1.0f;
        }
    }
}

/// Compute dot product between two vectors
kernel void dotProduct(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        sum += vectorA[i] * vectorB[i];
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = partialSums[0];
    }
}

// MARK: - Vector Operations

/// Batch normalization of multiple vectors in parallel
/// Each threadgroup processes one vector
kernel void batchNormalize(
    device const float* input [[buffer(0)]],      // [num_vectors, dimension]
    device float* output [[buffer(1)]],           // [num_vectors, dimension] 
    constant uint& num_vectors [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    threadgroup float* shared_sums [[threadgroup(0)]], // Shared memory for reduction
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each threadgroup processes one vector
    const uint vector_idx = tgid;
    
    if (vector_idx >= num_vectors) return;
    
    const uint vector_offset = vector_idx * dimension;
    
    // Phase 1: Compute magnitude for this vector (parallel reduction)
    float local_sum = 0.0f;
    for (uint d = tid; d < dimension; d += tg_size) {
        float val = input[vector_offset + d];
        local_sum += val * val;
    }
    
    // Shared memory reduction
    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Get final magnitude
    threadgroup float magnitude;
    if (tid == 0) {
        magnitude = sqrt(shared_sums[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Normalize all dimensions of this vector
    for (uint d = tid; d < dimension; d += tg_size) {
        if (magnitude > EPSILON) {
            output[vector_offset + d] = input[vector_offset + d] / magnitude;
        } else {
            output[vector_offset + d] = input[vector_offset + d];
        }
    }
}

/// Optimized batch normalization for contiguous vectors
/// Uses 2D grid for efficient parallel processing
kernel void batchNormalize2D(
    device const float* input [[buffer(0)]],      // [num_vectors, dimension]
    device float* output [[buffer(1)]],           // [num_vectors, dimension]
    device float* magnitudes [[buffer(2)]],       // [num_vectors] - optional output
    constant uint& num_vectors [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint vector_idx = gid.x;
    const uint chunk_size = 32; // Process in chunks for better cache usage
    const uint chunk_idx = gid.y;
    
    if (vector_idx >= num_vectors) return;
    
    const uint vector_offset = vector_idx * dimension;
    const uint start_dim = chunk_idx * chunk_size;
    const uint end_dim = min(start_dim + chunk_size, dimension);
    
    // Step 1: Compute partial sum for this chunk
    float partial_sum = 0.0f;
    for (uint d = start_dim; d < end_dim; d++) {
        float val = input[vector_offset + d];
        partial_sum += val * val;
    }
    
    // Use atomic to accumulate across chunks (simple for small chunk counts)
    device atomic_float* atomic_magnitude = (device atomic_float*)&magnitudes[vector_idx];
    atomic_fetch_add_explicit(atomic_magnitude, partial_sum, memory_order_relaxed);
    
    // Synchronize using threadgroup barrier if within same threadgroup
    threadgroup_barrier(mem_flags::mem_device);
    
    // Step 2: Normalize this chunk
    float magnitude = sqrt(magnitudes[vector_idx]);
    for (uint d = start_dim; d < end_dim; d++) {
        if (magnitude > EPSILON) {
            output[vector_offset + d] = input[vector_offset + d] / magnitude;
        } else {
            output[vector_offset + d] = input[vector_offset + d];
        }
    }
}

/// Normalize vector to unit length (single vector)
/// Two-pass algorithm for numerical stability
kernel void vectorNormalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    // First pass: compute magnitude
    float localSum = 0.0f;
    for (uint i = threadId; i < dimension; i += tgSize) {
        float val = input[i];
        localSum += val * val;
    }
    
    partialSums[threadId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for magnitude
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            partialSums[threadId] += partialSums[threadId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup float magnitude;
    if (threadId == 0) {
        magnitude = sqrt(partialSums[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Second pass: normalize each element
    if (tid < dimension) {
        if (magnitude > EPSILON) {
            output[tid] = input[tid] / magnitude;
        } else {
            output[tid] = input[tid];
        }
    }
}

/// Scale vector by scalar value
kernel void vectorScale(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = input[tid] * scalar;
}

/// Add two vectors element-wise
kernel void vectorAdd(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] + vectorB[tid];
}

/// Subtract two vectors element-wise
kernel void vectorSubtract(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] - vectorB[tid];
}

// MARK: - Matrix Operations

/// Matrix-vector multiplication (y = Ax)
/// Each thread computes one output element
kernel void matrixVectorMultiply(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= rows) return;
    
    float sum = 0.0f;
    uint rowOffset = tid * cols;
    
    // Unroll loop for better performance with small vectors
    uint i = 0;
    for (; i + 3 < cols; i += 4) {
        sum += matrix[rowOffset + i] * vector[i];
        sum += matrix[rowOffset + i + 1] * vector[i + 1];
        sum += matrix[rowOffset + i + 2] * vector[i + 2];
        sum += matrix[rowOffset + i + 3] * vector[i + 3];
    }
    
    // Handle remaining elements
    for (; i < cols; i++) {
        sum += matrix[rowOffset + i] * vector[i];
    }
    
    output[tid] = sum;
}

// MARK: - Batch Operations

/// Batch Euclidean distance computation
/// Compute distances from one query to multiple database vectors
kernel void batchEuclideanDistance(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint dbIdx = id.x;
    if (dbIdx >= numDatabase) return;
    
    float sum = 0.0f;
    uint dbOffset = dbIdx * dimension;
    
    // Unrolled loop for better performance
    uint i = 0;
    for (; i + 3 < dimension; i += 4) {
        float diff0 = query[i] - database[dbOffset + i];
        float diff1 = query[i + 1] - database[dbOffset + i + 1];
        float diff2 = query[i + 2] - database[dbOffset + i + 2];
        float diff3 = query[i + 3] - database[dbOffset + i + 3];
        
        sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    
    // Handle remaining elements
    for (; i < dimension; i++) {
        float diff = query[i] - database[dbOffset + i];
        sum += diff * diff;
    }
    
    distances[dbIdx] = sqrt(sum);
}

/// Batch cosine similarity computation
kernel void batchCosineSimilarity(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numDatabase [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numDatabase) return;
    
    float dotProduct = 0.0f;
    float queryNorm = 0.0f;
    float dbNorm = 0.0f;
    
    uint dbOffset = id * dimension;
    
    for (uint i = 0; i < dimension; i++) {
        float q = query[i];
        float d = database[dbOffset + i];
        
        dotProduct += q * d;
        queryNorm += q * q;
        dbNorm += d * d;
    }
    
    queryNorm = sqrt(queryNorm);
    dbNorm = sqrt(dbNorm);
    
    if (queryNorm > EPSILON && dbNorm > EPSILON) {
        similarities[id] = dotProduct / (queryNorm * dbNorm);
    } else {
        similarities[id] = 0.0f;
    }
}

// MARK: - Utility Operations

/// Compute L2 norm of a vector
kernel void vectorNorm(
    device const float* vector [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += tgSize) {
        float val = vector[i];
        sum += val * val;
    }
    
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = sqrt(partialSums[0]);
    }
}

/// Element-wise multiplication (Hadamard product)
kernel void elementwiseMultiply(
    device const float* vectorA [[buffer(0)]],
    device const float* vectorB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dimension) return;
    output[tid] = vectorA[tid] * vectorB[tid];
}

// MARK: - Shader Aliases for Compatibility

/// Alias for vectorNormalize - some code expects "normalizeVectors"
/// Note: This duplicates the vectorNormalize code since kernels can't call other kernels
kernel void normalizeVectors(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float partialSums[256];
    
    // First pass: compute magnitude
    float localSum = 0.0f;
    for (uint i = threadId; i < dimension; i += tgSize) {
        float val = input[i];
        localSum += val * val;
    }
    
    partialSums[threadId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for magnitude
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            partialSums[threadId] += partialSums[threadId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup float magnitude;
    if (threadId == 0) {
        magnitude = sqrt(partialSums[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Second pass: normalize each element
    if (tid < dimension) {
        if (magnitude > EPSILON) {
            output[tid] = input[tid] / magnitude;
        } else {
            output[tid] = input[tid];
        }
    }
}