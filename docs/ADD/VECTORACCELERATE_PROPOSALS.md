# VectorAccelerate Enhancement Proposals for SwiftTopics

**Document Version**: 1.0
**SwiftTopics Version**: 0.1.0-beta.1
**Date**: January 2026

## Executive Summary

SwiftTopics is a pure-Swift topic modeling library implementing BERTopic-style algorithms (HDBSCAN, UMAP, c-TF-IDF, NPMI). This document outlines GPU kernel proposals that would significantly improve SwiftTopics performance while being generally useful for other VectorAccelerate consumers.

### Current VectorAccelerate Usage

SwiftTopics already leverages:
- `L2DistanceKernel` - Pairwise distance computation
- `FusedL2TopKKernel` - k-NN search for core distances
- `MatrixMultiplyKernel` - PCA covariance and projection
- `StatisticsKernel` - Mean centering

### Performance Bottlenecks

| Operation | Current | Bottleneck | Proposed Kernel |
|-----------|---------|------------|-----------------|
| Mutual reachability | CPU O(n²) on-demand | Distance + max per edge | `MutualReachabilityKernel` |
| UMAP optimization | CPU gradient loop | Per-edge gradient compute | `UMAPGradientKernel` |
| c-TF-IDF scoring | CPU dictionary ops | Sparse matrix × log | `SparseLogTFIDFKernel` |
| NPMI coherence | CPU nested loops | Parallel counting | `CooccurrenceCountKernel` |

---

## Priority 1: MutualReachabilityKernel

### Problem Statement

HDBSCAN requires computing mutual reachability distances:

```
mutual_reach(a, b) = max(core_dist[a], core_dist[b], euclidean_dist(a, b))
```

Currently, we compute this on-demand during MST construction. For dense graphs or full pairwise computation, this is O(n²) and CPU-bound.

### Proposed Kernel

```metal
kernel void mutualReachabilityKernel(
    device const float* embeddings [[buffer(0)]],     // [N, D] row-major
    device const float* coreDistances [[buffer(1)]],  // [N]
    device float* mutualReachMatrix [[buffer(2)]],    // [N, N] output
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint j = tid.y;

    if (i >= N || j >= N) return;
    if (i == j) {
        mutualReachMatrix[i * N + j] = 0.0;
        return;
    }

    // Compute L2 distance
    float dist = 0.0;
    for (uint d = 0; d < D; d++) {
        float diff = embeddings[i * D + d] - embeddings[j * D + d];
        dist += diff * diff;
    }
    dist = sqrt(dist);

    // Mutual reachability = max(core_a, core_b, dist)
    float mutual = max(max(coreDistances[i], coreDistances[j]), dist);
    mutualReachMatrix[i * N + j] = mutual;
}
```

### Swift API

```swift
public struct MutualReachabilityKernel {
    /// Computes mutual reachability matrix.
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix.
    ///   - coreDistances: N core distances (from k-NN).
    /// - Returns: N×N mutual reachability distance matrix.
    public func compute(
        embeddings: [[Float]],
        coreDistances: [Float]
    ) async throws -> [[Float]]

    /// Computes mutual reachability for specific pairs (sparse).
    ///
    /// - Parameters:
    ///   - embeddings: N×D embedding matrix.
    ///   - coreDistances: N core distances.
    ///   - pairs: Pairs of indices to compute.
    /// - Returns: Distances for each pair.
    public func computeSparse(
        embeddings: [[Float]],
        coreDistances: [Float],
        pairs: [(Int, Int)]
    ) async throws -> [Float]
}
```

### Impact

- **Current**: O(n²) CPU time for distance computation
- **With kernel**: O(n²/threads) GPU time, ~10-50x speedup for n > 1000
- **Memory**: Same O(n²) but can tile for larger datasets

---

## Priority 2: UMAPGradientKernel

### Problem Statement

UMAP optimization computes attractive and repulsive gradients for each edge in the fuzzy simplicial set. The current implementation processes edges sequentially, which is inefficient for large edge counts.

### Current Algorithm (per edge)

```swift
// Attractive gradient (pull similar points together)
let distSquared = squaredDistance(embedding[i], embedding[j])
let gradCoeff = -2.0 * a * b * pow(distSquared, b - 1) / (1.0 + a * pow(distSquared, b))
for d in 0..<dims {
    gradient[d] = gradCoeff * (embedding[i][d] - embedding[j][d])
}

// Repulsive gradient (push dissimilar points apart)
let repelCoeff = 2.0 * b / ((0.001 + distSquared) * (1.0 + a * pow(distSquared, b)))
```

### Proposed Kernel

```metal
struct UMAPEdge {
    uint source;
    uint target;
    float weight;
};

kernel void umapGradientKernel(
    device const float* embedding [[buffer(0)]],           // [N, D]
    device const UMAPEdge* edges [[buffer(1)]],            // [E] edges
    device float* gradients [[buffer(2)]],                 // [N, D] accumulated
    constant UMAPParams& params [[buffer(3)]],             // a, b, learningRate
    constant uint& D [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    UMAPEdge edge = edges[tid];
    uint i = edge.source;
    uint j = edge.target;

    // Compute squared distance
    float distSq = 0.0;
    for (uint d = 0; d < D; d++) {
        float diff = embedding[i * D + d] - embedding[j * D + d];
        distSq += diff * diff;
    }

    // Attractive gradient coefficient
    float powDistB = pow(distSq, params.b);
    float gradCoeff = -2.0 * params.a * params.b * pow(distSq, params.b - 1.0)
                      / (1.0 + params.a * powDistB);
    gradCoeff *= params.learningRate * edge.weight;

    // Accumulate gradients (atomic add)
    for (uint d = 0; d < D; d++) {
        float grad = gradCoeff * (embedding[i * D + d] - embedding[j * D + d]);
        atomic_fetch_add_explicit(&gradients[i * D + d], grad, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradients[j * D + d], -grad, memory_order_relaxed);
    }
}
```

### Swift API

```swift
public struct UMAPGradientKernel {
    /// Computes UMAP gradients for all edges in parallel.
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding.
    ///   - edges: Edge list with weights.
    ///   - params: UMAP parameters (a, b, learningRate).
    /// - Returns: Accumulated gradients for each point.
    public func computeGradients(
        embedding: [[Float]],
        edges: [(source: Int, target: Int, weight: Float)],
        params: UMAPParameters
    ) async throws -> [[Float]]

    /// Performs full epoch with gradient computation and update.
    public func optimizeEpoch(
        embedding: inout [[Float]],
        edges: [(source: Int, target: Int, weight: Float)],
        params: UMAPParameters,
        negativeSampleRate: Int
    ) async throws
}
```

### Impact

- **Current**: O(epochs × edges) sequential gradient computation
- **With kernel**: Parallel edge processing, ~5-20x speedup
- **Bonus**: Enables larger `nEpochs` for better embedding quality

---

## Priority 3: SparseLogTFIDFKernel

### Problem Statement

c-TF-IDF computes:
```
score(term, cluster) = tf(term, cluster) × log(1 + avgClusterSize / tf(term, corpus))
```

This involves sparse matrix operations (term frequencies are sparse) and log transforms.

### Current Implementation

```swift
// For each cluster
for (termIdx, clusterFreq) in clusterTermFreqs[c] {
    let corpusFreq = Float(corpusTermFreq[termIdx] ?? 1)
    let score = Float(clusterFreq) * log(1.0 + avgTokensPerCluster / corpusFreq)
    scores[termIdx] = score
}
```

### Proposed Kernel

```metal
kernel void sparseLogTFIDFKernel(
    device const uint* clusterTermIndices [[buffer(0)]],   // CSR column indices
    device const uint* clusterTermOffsets [[buffer(1)]],   // CSR row pointers
    device const float* clusterTermFreqs [[buffer(2)]],    // CSR values (tf in cluster)
    device const float* corpusTermFreqs [[buffer(3)]],     // [V] corpus frequencies
    device float* scores [[buffer(4)]],                    // [nnz] output scores
    constant float& avgClusterSize [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint termIdx = clusterTermIndices[tid];
    float tf = clusterTermFreqs[tid];
    float corpusTf = corpusTermFreqs[termIdx];

    scores[tid] = tf * log(1.0 + avgClusterSize / max(corpusTf, 1.0));
}
```

### Swift API

```swift
public struct SparseLogTFIDFKernel {
    /// Computes c-TF-IDF scores for sparse term-frequency matrix.
    ///
    /// - Parameters:
    ///   - clusterTermMatrix: CSR sparse matrix of cluster term frequencies.
    ///   - corpusFrequencies: Global term frequencies.
    ///   - avgClusterSize: Average tokens per cluster.
    /// - Returns: c-TF-IDF scores (same sparsity pattern).
    public func compute(
        clusterTermMatrix: SparseMatrix,
        corpusFrequencies: [Float],
        avgClusterSize: Float
    ) async throws -> SparseMatrix
}
```

### Impact

- **Current**: O(k × avgTermsPerCluster) CPU
- **With kernel**: Parallel across all non-zero entries
- **Note**: Benefit scales with vocabulary size and cluster count

---

## Priority 4: CooccurrenceCountKernel

### Problem Statement

NPMI coherence requires counting word co-occurrences within sliding windows:

```swift
for each document:
    for each window position:
        for each word pair in window:
            pairCounts[word1, word2] += 1
```

This is O(documents × avgLength × windowSize²) and memory-bound.

### Proposed Approach

Parallelize across documents, use atomic operations for count accumulation.

```metal
kernel void cooccurrenceCountKernel(
    device const uint* tokenizedDocs [[buffer(0)]],        // Concatenated token IDs
    device const uint* docOffsets [[buffer(1)]],           // Start offset per doc
    device const uint* docLengths [[buffer(2)]],           // Length per doc
    device atomic_uint* pairCounts [[buffer(3)]],          // [V × V] or hash table
    constant uint& windowSize [[buffer(4)]],
    constant uint& V [[buffer(5)]],                        // Vocabulary size
    uint docIdx [[thread_position_in_grid]]
) {
    uint start = docOffsets[docIdx];
    uint length = docLengths[docIdx];

    for (uint i = 0; i < length; i++) {
        uint word1 = tokenizedDocs[start + i];
        uint windowEnd = min(i + windowSize, length);

        for (uint j = i + 1; j < windowEnd; j++) {
            uint word2 = tokenizedDocs[start + j];

            // Ensure consistent ordering
            uint minWord = min(word1, word2);
            uint maxWord = max(word1, word2);
            uint pairIdx = minWord * V + maxWord;

            atomic_fetch_add_explicit(&pairCounts[pairIdx], 1, memory_order_relaxed);
        }
    }
}
```

### Swift API

```swift
public struct CooccurrenceCountKernel {
    /// Counts word co-occurrences within sliding windows.
    ///
    /// - Parameters:
    ///   - documents: Tokenized documents (as vocabulary indices).
    ///   - windowSize: Sliding window size.
    ///   - vocabularySize: Total vocabulary size.
    /// - Returns: Sparse matrix of co-occurrence counts.
    public func count(
        documents: [[Int]],
        windowSize: Int,
        vocabularySize: Int
    ) async throws -> SparseCooccurrenceMatrix
}
```

### Impact

- **Current**: O(D × L × W²) CPU, often the slowest part of coherence
- **With kernel**: Parallel across documents, ~10-50x for large corpora
- **Memory**: Consider hash-based storage for large vocabularies

---

## Priority 5: Additional Utility Kernels

### 5.1 BatchMaxKernel

Compute element-wise maximum across batches:

```swift
// For mutual reachability: max(coreA, coreB, dist)
public func batchMax3(a: [Float], b: [Float], c: [Float]) -> [Float]
```

### 5.2 SparseMatrixVectorMultiply

For topic-document scoring:

```swift
public func sparseMV(
    matrix: SparseMatrix,  // Topics × Terms
    vector: [Float]        // Term weights
) -> [Float]               // Topic scores
```

### 5.3 LogSumExpKernel

Numerically stable softmax for topic probabilities:

```swift
public func logSumExp(values: [[Float]], axis: Int) -> [Float]
```

---

## Implementation Recommendations

### Phase 1: Core HDBSCAN (High Impact)
1. `MutualReachabilityKernel` - Enables GPU-accelerated HDBSCAN
2. Extend `FusedL2TopKKernel` for self-query (core distances)

### Phase 2: UMAP Optimization
3. `UMAPGradientKernel` - Major speedup for dimensionality reduction

### Phase 3: NLP Operations
4. `SparseLogTFIDFKernel` - Benefits any TF-IDF-based system
5. `CooccurrenceCountKernel` - General NLP utility

---

## Benchmarking Baselines

For validation, SwiftTopics can provide:

| Dataset | Documents | Embeddings | Current Time | Target |
|---------|-----------|------------|--------------|--------|
| Synthetic blobs | 1,000 | 384-dim | 2.5s | <0.5s |
| Medium corpus | 5,000 | 384-dim | 45s | <5s |
| Large corpus | 10,000 | 384-dim | 3min | <30s |

Test on: M1 Pro, M2 Max, M3 Max, M4

---

## Contact

For questions or collaboration on these proposals:

- **Library**: SwiftTopics 0.1.0-beta.1
- **Repository**: [TBD - internal path]
- **Integration Point**: `Sources/SwiftTopics/Acceleration/GPUContext.swift`

---

*This document reflects SwiftTopics performance profiling as of January 2026.*
