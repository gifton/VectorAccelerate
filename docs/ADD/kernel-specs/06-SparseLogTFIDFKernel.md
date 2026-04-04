# SparseLogTFIDFKernel Specification

**Version**: 1.0
**Status**: Approved (Low Priority)
**Priority**: Phase 4
**Estimated LOC**: ~600 (60 Metal + 300 Swift + 240 support)

---

## 1. Overview

### 1.1 Purpose

The SparseLogTFIDFKernel computes class-based TF-IDF (c-TF-IDF) scores for topic keyword extraction. This kernel operates on sparse term-frequency matrices since most clusters only contain a subset of vocabulary terms.

### 1.2 Mathematical Definition

```
c-TF-IDF(term, cluster) = tf(term, cluster) × log(1 + avgClusterSize / tf(term, corpus))
```

Where:
- `tf(term, cluster)` = frequency of term in all documents of cluster
- `avgClusterSize` = average number of tokens per cluster
- `tf(term, corpus)` = frequency of term across entire corpus

### 1.3 Priority Assessment

Given SwiftTopics benchmarks show c-TF-IDF takes only ~50ms for 10K docs on CPU, this is **low priority**. However, the kernel design is documented for future optimization if needed.

---

## 2. Data Representation

### 2.1 Sparse Matrix Formats

**Option A: CSR (Compressed Sparse Row)** - Standard format
```
rowPointers: [0, 3, 5, 8]  // Start index of each row's data
colIndices: [0, 2, 4, 1, 3, 0, 2, 5]  // Column indices
values: [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0]  // Non-zero values
```

**Option B: Flattened Per-Cluster Arrays** - Simpler for this use case
```swift
struct ClusterTerms {
    var termIndices: [UInt32]  // Which terms appear
    var frequencies: [Float]   // Their frequencies
}
clusters: [ClusterTerms]  // One per cluster
```

**Recommendation**: Use Option B (per-cluster arrays) for simplicity. SwiftTopics likely already has per-cluster term counts in this format.

### 2.2 Input Specification

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `clusterTermIndices` | `MTLBuffer` (UInt32) | [nnz] | Term indices (flattened) |
| `clusterTermFreqs` | `MTLBuffer` (Float32) | [nnz] | Term frequencies (flattened) |
| `clusterOffsets` | `MTLBuffer` (UInt32) | [K+1] | Start offset per cluster |
| `corpusFreqs` | `MTLBuffer` (Float32) | [V] | Corpus-wide term frequencies |
| `avgClusterSize` | `Float32` | scalar | Average tokens per cluster |
| `nnz` | `UInt32` | scalar | Total non-zero entries |

### 2.3 Output Specification

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `scores` | `MTLBuffer` (Float32) | [nnz] | c-TF-IDF scores (same sparsity) |

---

## 3. Metal Shader Design

### 3.1 File Location

```
Sources/VectorAccelerate/Shaders/SparseLogTFIDF.metal
```

### 3.2 Kernel Implementation

```metal
#include <metal_stdlib>
using namespace metal;

struct CTFIDFParams {
    float avgClusterSize;
    uint nnz;
};

/// Computes c-TF-IDF scores for sparse term-frequency data.
/// Each thread handles one non-zero entry.
kernel void sparse_ctfidf_kernel(
    device const uint* termIndices      [[buffer(0)]],  // [nnz] term indices
    device const float* termFreqs       [[buffer(1)]],  // [nnz] term frequencies
    device const float* corpusFreqs     [[buffer(2)]],  // [V] corpus frequencies
    device float* scores                [[buffer(3)]],  // [nnz] output scores
    constant CTFIDFParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.nnz) return;

    uint termIdx = termIndices[tid];
    float tf = termFreqs[tid];
    float corpusTf = corpusFreqs[termIdx];

    // c-TF-IDF formula
    // Avoid division by zero: use max(corpusTf, 1.0)
    float idf = log(1.0f + params.avgClusterSize / max(corpusTf, 1.0f));
    scores[tid] = tf * idf;
}

/// Vectorized version for aligned data.
kernel void sparse_ctfidf_vectorized_kernel(
    device const uint4* termIndices     [[buffer(0)]],  // [nnz/4]
    device const float4* termFreqs      [[buffer(1)]],  // [nnz/4]
    device const float* corpusFreqs     [[buffer(2)]],  // [V]
    device float4* scores               [[buffer(3)]],  // [nnz/4]
    constant CTFIDFParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid * 4 >= params.nnz) return;

    uint4 indices = termIndices[tid];
    float4 tf = termFreqs[tid];

    // Gather corpus frequencies
    float4 corpusTf = float4(
        corpusFreqs[indices.x],
        corpusFreqs[indices.y],
        corpusFreqs[indices.z],
        corpusFreqs[indices.w]
    );

    // c-TF-IDF
    float4 idf = log(1.0f + params.avgClusterSize / max(corpusTf, float4(1.0f)));
    scores[tid] = tf * idf;
}

/// Per-cluster reduction to find top-K terms.
/// Uses heap-based selection within each cluster's term set.
kernel void ctfidf_topk_per_cluster_kernel(
    device const float* scores          [[buffer(0)]],  // [nnz]
    device const uint* termIndices      [[buffer(1)]],  // [nnz]
    device const uint* clusterOffsets   [[buffer(2)]],  // [K+1]
    device uint* topKIndices            [[buffer(3)]],  // [K, topK]
    device float* topKScores            [[buffer(4)]],  // [K, topK]
    constant uint& numClusters          [[buffer(5)]],
    constant uint& topK                 [[buffer(6)]],
    uint cid [[thread_position_in_grid]]
) {
    if (cid >= numClusters) return;

    uint start = clusterOffsets[cid];
    uint end = clusterOffsets[cid + 1];
    uint count = end - start;

    // Simple selection sort for small topK (typically 10-20)
    // For production, use heap for better complexity

    device uint* outIndices = topKIndices + cid * topK;
    device float* outScores = topKScores + cid * topK;

    // Initialize with -inf
    for (uint k = 0; k < topK; k++) {
        outScores[k] = -INFINITY;
        outIndices[k] = 0;
    }

    // Find top-K
    for (uint i = 0; i < count; i++) {
        float score = scores[start + i];
        uint termIdx = termIndices[start + i];

        // Check if this score beats the minimum in top-K
        if (score > outScores[topK - 1]) {
            // Insert in sorted position
            uint insertPos = topK - 1;
            while (insertPos > 0 && score > outScores[insertPos - 1]) {
                outScores[insertPos] = outScores[insertPos - 1];
                outIndices[insertPos] = outIndices[insertPos - 1];
                insertPos--;
            }
            outScores[insertPos] = score;
            outIndices[insertPos] = termIdx;
        }
    }
}
```

---

## 4. Swift API Design

### 4.1 File Location

```
Sources/VectorAccelerate/Kernels/NLP/SparseLogTFIDFKernel.swift
```

### 4.2 Supporting Types

```swift
/// Represents sparse term frequencies for a cluster.
public struct ClusterTermFrequencies: Sendable {
    /// Term indices (vocabulary IDs).
    public var termIndices: [UInt32]
    /// Corresponding term frequencies.
    public var frequencies: [Float]

    public init(termIndices: [UInt32], frequencies: [Float]) {
        precondition(termIndices.count == frequencies.count)
        self.termIndices = termIndices
        self.frequencies = frequencies
    }

    public init(from dictionary: [Int: Int]) {
        self.termIndices = dictionary.keys.map { UInt32($0) }
        self.frequencies = dictionary.values.map { Float($0) }
    }
}

/// Result of c-TF-IDF computation.
public struct CTFIDFResult: Sendable {
    /// Sparse scores matching input sparsity pattern.
    public let scores: [Float]

    /// Top-K terms per cluster (if requested).
    public let topKPerCluster: [[(termIndex: Int, score: Float)]]?
}
```

### 4.3 Public Interface

```swift
import Metal

/// Computes class-based TF-IDF (c-TF-IDF) scores for topic keyword extraction.
///
/// c-TF-IDF is defined as:
/// ```
/// c-TF-IDF(term, cluster) = tf(term, cluster) × log(1 + avgClusterSize / tf(term, corpus))
/// ```
///
/// This highlights terms that are:
/// - Frequent within the cluster (high tf in cluster)
/// - Rare across the corpus (low tf in corpus)
///
/// ## Example
/// ```swift
/// let kernel = try SparseLogTFIDFKernel(context: context)
///
/// let result = try await kernel.compute(
///     clusterTerms: clusterTermFreqs,
///     corpusFrequencies: corpusTermCounts,
///     avgClusterSize: avgTokensPerCluster
/// )
///
/// // Get top 10 keywords per cluster
/// let topKeywords = try await kernel.topKPerCluster(
///     clusterTerms: clusterTermFreqs,
///     corpusFrequencies: corpusTermCounts,
///     avgClusterSize: avgTokensPerCluster,
///     k: 10
/// )
/// ```
public struct SparseLogTFIDFKernel: Metal4Kernel {

    // MARK: - Properties

    private let context: Metal4Context
    private let ctfidfPipeline: MTLComputePipelineState
    private let topKPipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(context: Metal4Context) throws {
        self.context = context
        self.ctfidfPipeline = try context.makePipeline(function: "sparse_ctfidf_kernel")
        self.topKPipeline = try context.makePipeline(function: "ctfidf_topk_per_cluster_kernel")
    }

    // MARK: - Compute c-TF-IDF

    /// Computes c-TF-IDF scores for sparse cluster term frequencies.
    ///
    /// - Parameters:
    ///   - clusterTerms: Per-cluster term indices and frequencies.
    ///   - corpusFrequencies: Corpus-wide term frequency counts.
    ///   - avgClusterSize: Average number of tokens per cluster.
    /// - Returns: c-TF-IDF scores in same sparse format as input.
    public func compute(
        clusterTerms: [ClusterTermFrequencies],
        corpusFrequencies: [Float],
        avgClusterSize: Float
    ) async throws -> [[Float]] {
        // Flatten cluster data
        var allIndices: [UInt32] = []
        var allFreqs: [Float] = []
        var offsets: [UInt32] = [0]

        for cluster in clusterTerms {
            allIndices.append(contentsOf: cluster.termIndices)
            allFreqs.append(contentsOf: cluster.frequencies)
            offsets.append(UInt32(allIndices.count))
        }

        let nnz = allIndices.count
        if nnz == 0 {
            return clusterTerms.map { _ in [] }
        }

        // Create buffers
        let indicesBuffer = try context.makeBuffer(bytes: allIndices, label: "CTFIDF.indices")
        let freqsBuffer = try context.makeBuffer(bytes: allFreqs, label: "CTFIDF.freqs")
        let corpusBuffer = try context.makeBuffer(bytes: corpusFrequencies, label: "CTFIDF.corpus")
        let scoresBuffer = try context.makeBuffer(
            length: nnz * MemoryLayout<Float>.size,
            label: "CTFIDF.scores"
        )

        // Execute kernel
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(ctfidfPipeline)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 0)
            encoder.setBuffer(freqsBuffer, offset: 0, index: 1)
            encoder.setBuffer(corpusBuffer, offset: 0, index: 2)
            encoder.setBuffer(scoresBuffer, offset: 0, index: 3)

            var params = CTFIDFParamsGPU(avgClusterSize: avgClusterSize, nnz: UInt32(nnz))
            encoder.setBytes(&params, length: MemoryLayout<CTFIDFParamsGPU>.size, index: 4)

            let config = Metal4ThreadConfiguration.grid1D(count: nnz, pipeline: ctfidfPipeline)
            encoder.dispatchThreads(config.threadsPerGrid, threadsPerThreadgroup: config.threadsPerThreadgroup)
        }

        // Read back and split by cluster
        let scoresPtr = scoresBuffer.contents().bindMemory(to: Float.self, capacity: nnz)
        var result: [[Float]] = []

        for i in 0..<clusterTerms.count {
            let start = Int(offsets[i])
            let end = Int(offsets[i + 1])
            result.append(Array(UnsafeBufferPointer(start: scoresPtr + start, count: end - start)))
        }

        return result
    }

    // MARK: - Top-K Extraction

    /// Computes c-TF-IDF and extracts top-K terms per cluster.
    ///
    /// - Parameters:
    ///   - clusterTerms: Per-cluster term indices and frequencies.
    ///   - corpusFrequencies: Corpus-wide term frequency counts.
    ///   - avgClusterSize: Average tokens per cluster.
    ///   - k: Number of top terms to extract per cluster.
    /// - Returns: Top-K (termIndex, score) pairs per cluster.
    public func topKPerCluster(
        clusterTerms: [ClusterTermFrequencies],
        corpusFrequencies: [Float],
        avgClusterSize: Float,
        k: Int
    ) async throws -> [[(termIndex: Int, score: Float)]] {
        let numClusters = clusterTerms.count
        if numClusters == 0 { return [] }

        // Flatten data
        var allIndices: [UInt32] = []
        var allFreqs: [Float] = []
        var offsets: [UInt32] = [0]

        for cluster in clusterTerms {
            allIndices.append(contentsOf: cluster.termIndices)
            allFreqs.append(contentsOf: cluster.frequencies)
            offsets.append(UInt32(allIndices.count))
        }

        let nnz = allIndices.count

        // Create buffers
        let indicesBuffer = try context.makeBuffer(bytes: allIndices, label: "CTFIDF.indices")
        let freqsBuffer = try context.makeBuffer(bytes: allFreqs, label: "CTFIDF.freqs")
        let corpusBuffer = try context.makeBuffer(bytes: corpusFrequencies, label: "CTFIDF.corpus")
        let offsetsBuffer = try context.makeBuffer(bytes: offsets, label: "CTFIDF.offsets")
        let scoresBuffer = try context.makeBuffer(
            length: nnz * MemoryLayout<Float>.size,
            label: "CTFIDF.scores"
        )
        let topKIndicesBuffer = try context.makeBuffer(
            length: numClusters * k * MemoryLayout<UInt32>.size,
            label: "CTFIDF.topKIndices"
        )
        let topKScoresBuffer = try context.makeBuffer(
            length: numClusters * k * MemoryLayout<Float>.size,
            label: "CTFIDF.topKScores"
        )

        try await context.executeAndWait { commandBuffer, encoder in
            // Step 1: Compute c-TF-IDF scores
            encoder.setComputePipelineState(ctfidfPipeline)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 0)
            encoder.setBuffer(freqsBuffer, offset: 0, index: 1)
            encoder.setBuffer(corpusBuffer, offset: 0, index: 2)
            encoder.setBuffer(scoresBuffer, offset: 0, index: 3)
            var params = CTFIDFParamsGPU(avgClusterSize: avgClusterSize, nnz: UInt32(nnz))
            encoder.setBytes(&params, length: MemoryLayout<CTFIDFParamsGPU>.size, index: 4)
            let config1 = Metal4ThreadConfiguration.grid1D(count: nnz, pipeline: ctfidfPipeline)
            encoder.dispatchThreads(config1.threadsPerGrid, threadsPerThreadgroup: config1.threadsPerThreadgroup)

            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Extract top-K per cluster
            encoder.setComputePipelineState(topKPipeline)
            encoder.setBuffer(scoresBuffer, offset: 0, index: 0)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 1)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
            encoder.setBuffer(topKIndicesBuffer, offset: 0, index: 3)
            encoder.setBuffer(topKScoresBuffer, offset: 0, index: 4)
            var numClustersU32 = UInt32(numClusters)
            var kU32 = UInt32(k)
            encoder.setBytes(&numClustersU32, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&kU32, length: MemoryLayout<UInt32>.size, index: 6)
            let config2 = Metal4ThreadConfiguration.grid1D(count: numClusters, pipeline: topKPipeline)
            encoder.dispatchThreads(config2.threadsPerGrid, threadsPerThreadgroup: config2.threadsPerThreadgroup)
        }

        // Read back results
        let indicesPtr = topKIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: numClusters * k)
        let scoresPtr = topKScoresBuffer.contents().bindMemory(to: Float.self, capacity: numClusters * k)

        var result: [[(termIndex: Int, score: Float)]] = []
        for c in 0..<numClusters {
            var clusterTopK: [(termIndex: Int, score: Float)] = []
            for j in 0..<k {
                let idx = c * k + j
                let score = scoresPtr[idx]
                if score > -.infinity {  // Valid entry
                    clusterTopK.append((termIndex: Int(indicesPtr[idx]), score: score))
                }
            }
            result.append(clusterTopK)
        }

        return result
    }
}

// MARK: - GPU Structures

struct CTFIDFParamsGPU {
    var avgClusterSize: Float
    var nnz: UInt32
}
```

---

## 5. Complexity Analysis

### 5.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Score computation | O(nnz) | Fully parallel |
| Top-K extraction | O(K × nnz_per_cluster × log(topK)) | Per-cluster |

### 5.2 Space Complexity

| Buffer | Size |
|--------|------|
| Flattened indices | O(nnz) |
| Flattened frequencies | O(nnz) |
| Corpus frequencies | O(V) |
| Scores | O(nnz) |
| Top-K results | O(K × topK) |

---

## 6. Performance Considerations

### 6.1 When GPU is Beneficial

- Large vocabulary (V > 10K)
- Many clusters (K > 100)
- Dense term overlap (nnz > 100K)

### 6.2 When CPU May Be Faster

- Small vocabulary (V < 5K)
- Few clusters (K < 30)
- Sparse data (nnz < 10K)

Given SwiftTopics targets V < 10K and K ≈ 30, CPU may be sufficient. This kernel is provided for future scaling.

### 6.3 Expected Performance

| Scenario | CPU Time | GPU Time | Winner |
|----------|----------|----------|--------|
| V=5K, K=30, nnz=10K | ~10ms | ~5ms + overhead | CPU |
| V=50K, K=100, nnz=500K | ~200ms | ~20ms | GPU |
| V=100K, K=200, nnz=2M | ~1s | ~50ms | GPU |

---

## 7. Testing Requirements

### 7.1 Correctness Tests

```swift
func testCTFIDFCorrectness() async throws {
    // Simple test case
    let clusterTerms = [
        ClusterTermFrequencies(termIndices: [0, 1], frequencies: [5.0, 3.0]),
        ClusterTermFrequencies(termIndices: [1, 2], frequencies: [2.0, 4.0])
    ]
    let corpusFreqs: [Float] = [10.0, 20.0, 5.0]  // 3-term vocabulary
    let avgClusterSize: Float = 10.0

    let result = try await kernel.compute(
        clusterTerms: clusterTerms,
        corpusFrequencies: corpusFreqs,
        avgClusterSize: avgClusterSize
    )

    // Expected: tf × log(1 + avg / corpusTf)
    // Cluster 0, term 0: 5 × log(1 + 10/10) = 5 × log(2) ≈ 3.47
    // Cluster 0, term 1: 3 × log(1 + 10/20) = 3 × log(1.5) ≈ 1.22

    XCTAssertEqual(result[0][0], 5.0 * log(2.0), accuracy: 1e-5)
    XCTAssertEqual(result[0][1], 3.0 * log(1.5), accuracy: 1e-5)
}

func testTopKExtraction() async throws {
    let clusterTerms = [
        ClusterTermFrequencies(
            termIndices: [0, 1, 2, 3, 4],
            frequencies: [1.0, 5.0, 2.0, 4.0, 3.0]
        )
    ]
    let corpusFreqs: [Float] = [10.0, 10.0, 10.0, 10.0, 10.0]

    let topK = try await kernel.topKPerCluster(
        clusterTerms: clusterTerms,
        corpusFrequencies: corpusFreqs,
        avgClusterSize: 10.0,
        k: 3
    )

    // Highest tf values: 5 (term 1), 4 (term 3), 3 (term 4)
    XCTAssertEqual(topK[0].count, 3)
    XCTAssertEqual(topK[0][0].termIndex, 1)  // Highest score
    XCTAssertEqual(topK[0][1].termIndex, 3)
    XCTAssertEqual(topK[0][2].termIndex, 4)
}
```

---

## 8. Integration Notes

### 8.1 With SwiftTopics TopicRepresenter

```swift
// In SwiftTopics cTFIDF implementation
public class cTFIDFRepresenter: TopicRepresenter {
    private let kernel: SparseLogTFIDFKernel?

    public func extractKeywords(
        clusters: [Cluster],
        documents: [TokenizedDocument]
    ) async throws -> [Topic] {
        // Build cluster term frequencies
        let clusterTerms = clusters.map { cluster in
            buildTermFrequencies(cluster: cluster, documents: documents)
        }

        // Compute corpus frequencies
        let corpusFreqs = computeCorpusFrequencies(documents: documents)
        let avgSize = Float(totalTokens) / Float(clusters.count)

        // Use GPU kernel if available, else CPU
        let topKeywords: [[(Int, Float)]]
        if let kernel = kernel {
            topKeywords = try await kernel.topKPerCluster(
                clusterTerms: clusterTerms,
                corpusFrequencies: corpusFreqs,
                avgClusterSize: avgSize,
                k: 10
            )
        } else {
            topKeywords = cpuTopKPerCluster(...)
        }

        // Convert to Topic objects
        return zip(clusters, topKeywords).map { cluster, keywords in
            Topic(
                id: cluster.id,
                keywords: keywords.map { (vocabulary[$0.0], $0.1) }
            )
        }
    }
}
```

---

## 9. Future Considerations

### 9.1 MMR Diversification

Add Maximal Marginal Relevance to reduce keyword redundancy:

```swift
public func topKWithMMR(
    clusterTerms: [ClusterTermFrequencies],
    termEmbeddings: [[Float]],  // For similarity computation
    lambda: Float = 0.5,
    k: Int
) async throws -> [[(termIndex: Int, score: Float)]]
```

### 9.2 Batched Processing

For very large vocabularies, process in batches to reduce memory:

```swift
public func computeBatched(
    clusterTerms: [ClusterTermFrequencies],
    corpusFrequencies: [Float],
    batchSize: Int = 100_000
) async throws -> [[Float]]
```
