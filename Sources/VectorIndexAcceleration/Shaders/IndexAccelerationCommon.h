//
//  IndexAccelerationCommon.h
//  VectorIndexAcceleration
//
//  Common definitions for index acceleration Metal shaders.
//  This file provides shared types and utilities for HNSW, IVF, and clustering kernels.
//

#ifndef IndexAccelerationCommon_h
#define IndexAccelerationCommon_h

#include <metal_stdlib>
using namespace metal;

// MARK: - HNSW Structures

/// Parameters for HNSW search kernel
struct HNSWSearchParams {
    uint dimension;        // Vector dimension
    uint efSearch;         // Number of candidates to explore
    uint k;                // Number of results to return
    uint layer;            // Current layer being searched
    uint maxConnections;   // M parameter - max connections per node
    uint _padding[3];      // Alignment padding
};

/// Parameters for HNSW distance matrix computation
struct HNSWDistanceMatrixParams {
    uint dimension;
    uint numQueries;
    uint numCandidates;
    uint metric;          // 0=L2, 1=Cosine, 2=DotProduct
    uint tileSize;
    uint _padding[3];
};

/// HNSW edge representation
struct HNSWEdge {
    uint sourceId;
    uint targetId;
    float distance;
    uint layer;
};

/// Parameters for HNSW level assignment
struct HNSWLevelParams {
    float mlFactor;       // 1/ln(M)
    uint maxLevel;
    uint batchSize;
    uint globalSeed;
};

/// Parameters for HNSW edge insertion
struct HNSWEdgeInsertionParams {
    uint batchSize;
    uint maxEdgesPerNode;
    uint bidirectional;   // 0=directed, 1=bidirectional
    uint _padding;
};

/// Parameters for HNSW pruning
struct HNSWPruningParams {
    uint nodeCount;
    uint maxDegree;
    uint preserveDistances;
    uint _padding;
};

/// Parameters for distance cache
struct HNSWDistanceCacheParams {
    uint cacheCapacity;
    uint totalNodes;
    uint _padding[2];
};

// MARK: - IVF Structures

/// Parameters for IVF coarse quantization
struct IVFCoarseParams {
    uint dimension;
    uint numQueries;
    uint numCentroids;
    uint nprobe;
};

/// Parameters for IVF list search
struct IVFListSearchParams {
    uint dimension;
    uint k;
    uint listIndex;
    uint listSize;
};

/// IVF inverted list entry
struct IVFListEntry {
    uint vectorIndex;      // Index in original dataset
    uint listId;           // Which inverted list this belongs to
};

// MARK: - Clustering Structures

/// Parameters for KMeans point assignment
struct KMeansAssignParams {
    uint dimension;
    uint numVectors;
    uint numCentroids;
    uint _padding;
};

/// Parameters for KMeans centroid update
struct KMeansUpdateParams {
    uint dimension;
    uint numCentroids;
};

/// Parameters for KMeans convergence check
struct KMeansConvergenceParams {
    uint numCentroids;
    uint dimension;
    float thresholdSquared;
    uint _padding;
};

// MARK: - Utility Functions

/// Compute squared L2 distance between two vectors
inline float l2_distance_squared(
    device const float* a,
    device const float* b,
    uint dimension
) {
    float sum = 0.0f;
    for (uint i = 0; i < dimension; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/// Compute squared L2 distance using float4 vectorization
inline float l2_distance_squared_vec4(
    device const float4* a,
    device const float4* b,
    uint vec4Count
) {
    float4 sum = float4(0.0f);
    for (uint i = 0; i < vec4Count; i++) {
        float4 diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum.x + sum.y + sum.z + sum.w;
}

/// Compute cosine similarity between two vectors
inline float cosine_similarity(
    device const float* a,
    device const float* b,
    uint dimension
) {
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (uint i = 0; i < dimension; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    float denom = sqrt(normA * normB);
    return denom > 0.0f ? dot / denom : 0.0f;
}

/// Atomic min for float (using integer atomics)
inline void atomic_min_float(
    device atomic_uint* addr,
    float value
) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;

    do {
        float current = as_type<float>(expected);
        if (value >= current) break;
        desired = as_type<uint>(value);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    ));
}

#endif /* IndexAccelerationCommon_h */
