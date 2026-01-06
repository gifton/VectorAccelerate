//
//  SparseLogTFIDF.metal
//  VectorAccelerate
//
//  GPU kernels for class-based TF-IDF (c-TF-IDF) computation.
//
//  Kernels:
//  - sparse_ctfidf_kernel: Basic c-TF-IDF computation for sparse term frequencies
//  - sparse_ctfidf_vectorized_kernel: float4 vectorized version for aligned data
//  - ctfidf_topk_per_cluster_kernel: Top-K term extraction per cluster
//
//  Primary use case: Topic keyword extraction in topic modeling
//    c-TF-IDF(term, cluster) = tf(term, cluster) * log(1 + avgClusterSize / tf(term, corpus))
//

#include <metal_stdlib>
using namespace metal;

#ifndef VA_SPARSE_CTFIDF_GUARD
#define VA_SPARSE_CTFIDF_GUARD

// MARK: - Parameter Structures

struct CTFIDFParams {
    float avgClusterSize;
    uint nnz;
};

// MARK: - Core c-TF-IDF Computation

/// Computes c-TF-IDF scores for sparse term-frequency data.
///
/// Each thread handles one non-zero entry in the sparse matrix.
/// Formula: c-TF-IDF = tf(term, cluster) * log(1 + avgClusterSize / tf(term, corpus))
///
/// - Parameters:
///   - termIndices: [nnz] term indices (vocabulary IDs)
///   - termFreqs: [nnz] term frequencies in clusters
///   - corpusFreqs: [V] corpus-wide term frequencies
///   - scores: [nnz] output c-TF-IDF scores
///   - params: avgClusterSize and nnz
kernel void sparse_ctfidf_kernel(
    device const uint* termIndices      [[buffer(0)]],
    device const float* termFreqs       [[buffer(1)]],
    device const float* corpusFreqs     [[buffer(2)]],
    device float* scores                [[buffer(3)]],
    constant CTFIDFParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.nnz) return;

    uint termIdx = termIndices[tid];
    float tf = termFreqs[tid];
    float corpusTf = corpusFreqs[termIdx];

    // c-TF-IDF formula with division-by-zero protection
    float idf = log(1.0f + params.avgClusterSize / max(corpusTf, 1.0f));
    scores[tid] = tf * idf;
}

/// Vectorized c-TF-IDF for aligned data.
///
/// Processes 4 elements per thread using float4.
/// Requires nnz to be divisible by 4.
kernel void sparse_ctfidf_vectorized_kernel(
    device const uint4* termIndices     [[buffer(0)]],
    device const float4* termFreqs      [[buffer(1)]],
    device const float* corpusFreqs     [[buffer(2)]],
    device float4* scores               [[buffer(3)]],
    constant CTFIDFParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint baseIdx = tid * 4;
    if (baseIdx >= params.nnz) return;

    uint4 indices = termIndices[tid];
    float4 tf = termFreqs[tid];

    // Gather corpus frequencies (non-contiguous access)
    float4 corpusTf = float4(
        corpusFreqs[indices.x],
        corpusFreqs[indices.y],
        corpusFreqs[indices.z],
        corpusFreqs[indices.w]
    );

    // c-TF-IDF computation
    float4 idf = log(1.0f + params.avgClusterSize / max(corpusTf, float4(1.0f)));
    scores[tid] = tf * idf;
}

// MARK: - Top-K Extraction

/// Per-cluster reduction to find top-K terms by c-TF-IDF score.
///
/// Uses insertion sort which is efficient for small K (typically 10-20).
/// Each thread processes one cluster.
///
/// - Parameters:
///   - scores: [nnz] precomputed c-TF-IDF scores
///   - termIndices: [nnz] term indices
///   - clusterOffsets: [numClusters+1] start offset for each cluster's terms
///   - topKIndices: [numClusters, topK] output term indices
///   - topKScores: [numClusters, topK] output scores
///   - numClusters: total number of clusters
///   - topK: number of top terms to extract per cluster
kernel void ctfidf_topk_per_cluster_kernel(
    device const float* scores          [[buffer(0)]],
    device const uint* termIndices      [[buffer(1)]],
    device const uint* clusterOffsets   [[buffer(2)]],
    device uint* topKIndices            [[buffer(3)]],
    device float* topKScores            [[buffer(4)]],
    constant uint& numClusters          [[buffer(5)]],
    constant uint& topK                 [[buffer(6)]],
    uint cid [[thread_position_in_grid]]
) {
    if (cid >= numClusters) return;

    uint start = clusterOffsets[cid];
    uint end = clusterOffsets[cid + 1];
    uint count = end - start;

    // Output pointers for this cluster
    device uint* outIndices = topKIndices + cid * topK;
    device float* outScores = topKScores + cid * topK;

    // Initialize with -inf (invalid entries)
    for (uint k = 0; k < topK; k++) {
        outScores[k] = -INFINITY;
        outIndices[k] = 0xFFFFFFFF;  // Invalid sentinel
    }

    // Scan all terms and maintain sorted top-K using insertion
    for (uint i = 0; i < count; i++) {
        float score = scores[start + i];
        uint termIdx = termIndices[start + i];

        // Check if this score beats the minimum in top-K
        if (score > outScores[topK - 1]) {
            // Find insertion position (linear scan, fine for small K)
            uint insertPos = topK - 1;
            while (insertPos > 0 && score > outScores[insertPos - 1]) {
                // Shift down
                outScores[insertPos] = outScores[insertPos - 1];
                outIndices[insertPos] = outIndices[insertPos - 1];
                insertPos--;
            }
            outScores[insertPos] = score;
            outIndices[insertPos] = termIdx;
        }
    }
}

#endif // VA_SPARSE_CTFIDF_GUARD
