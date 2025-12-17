# VectorAccelerate Next Phase - Agent Handoff Prompt

## Project Context

**Repository:** `/Users/goftin/dev/gsuite/VSK/future/VectorAccelerate`
**Type:** Swift GPU-accelerated vector similarity search library using Metal
**Branch:** `gifton.learning-guides`

VectorAccelerate provides:
- **Flat index**: Brute-force GPU-accelerated L2 distance + TopK selection
- **IVF index**: Inverted File index for approximate nearest neighbor search
- **Stable handles**: Persist across compaction operations
- **Metal 4 kernel infrastructure**: Modern GPU compute patterns

---

## Current State Summary

### All Critical Bugs Fixed ✅

| Item | Description | Status |
|------|-------------|--------|
| P0.1 | IVF GPU Indirection (CSR candidates) | ✅ Done |
| P0.2 | IVF Training Duplicates Fix | ✅ Done |
| P0.3 | Batch Search Union Bug | ✅ Done |
| P0.4-P0.9 | Various correctness fixes | ✅ Done |
| P1.1-P1.4 | Performance improvements | ✅ Done |
| P2.3 | Benchmarking Harness | ✅ Done |

### Test Status
- **851 tests passing** (847 original + 4 new benchmark tests)
- IVF Recall: 92% (nprobe=4/nlist=8), 100% (nprobe=8/nlist=8)

### IVF Quality Assessment Results

**Recall vs FAISS Benchmarks (N=2000, D=128, nlist=32):**
| nprobe | % clusters | VectorAccelerate | Status |
|--------|------------|------------------|--------|
| 3 | ~10% | 77.4% | ✓ Meets FAISS benchmark |
| 6 | ~20% | 87.7% | ✓ Exceeds benchmark |
| 16 | 50% | 96.7% | ✓ Exceeds benchmark |
| 32 | 100% | 100.0% | ✓ Perfect recall |

**Key Finding:** IVF recall is production-ready and matches industry standards.

---

## Next Phase: Performance Improvements

### Priority Order

| Priority | Item | Impact | Effort | Description |
|----------|------|--------|--------|-------------|
| **1** | P2.1 | High | Medium | GPU Coarse Quantization - Remove CPU bottleneck |
| **2** | P4.1 | Medium | Low | K-Means++ Initialization - +5-10% recall |
| **3** | P2.2 | Medium | Medium | Scalar Quantization Integration - 4x memory reduction |
| **4** | P4.2 | Low | Low | Adaptive nlist Selection - Better UX |

---

## P2.1: GPU-Native Coarse Quantization

**Goal:** Move IVF coarse search (finding nearest centroids) from CPU to GPU.

### Current Flow (CPU Bottleneck)
```
Query → [CPU] Find nprobe nearest centroids
      → [CPU] Build CSR candidate list
      → [GPU] ivf_distance_with_indirection kernel
      → [GPU] TopK selection
      → Results
```

### Target Flow (Fully GPU)
```
Query → [GPU] Coarse search kernel (query × centroids)
      → [GPU] Candidate list construction kernel
      → [GPU] ivf_distance_with_indirection kernel
      → [GPU] TopK selection
      → Results
```

### Implementation Steps

**Step 1: Create GPU Coarse Search Kernel**

File: `Sources/VectorAccelerate/Metal/Shaders/IVFCoarseSearch.metal`

```metal
struct CoarseSearchParams {
    uint32_t num_queries;
    uint32_t num_centroids;
    uint32_t dimension;
    uint32_t nprobe;
};

// For each query, find nprobe nearest centroids
kernel void ivf_coarse_search(
    device const float* queries [[buffer(0)]],       // [Q × D]
    device const float* centroids [[buffer(1)]],     // [nlist × D]
    device float* distances [[buffer(2)]],           // [Q × nlist] output
    device uint* nearest_indices [[buffer(3)]],      // [Q × nprobe] output
    constant CoarseSearchParams& params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint q = tid.y;  // query index
    uint c = tid.x;  // centroid index

    if (q >= params.num_queries || c >= params.num_centroids) return;

    // Compute L2 distance between query[q] and centroid[c]
    float dist = 0.0f;
    for (uint d = 0; d < params.dimension; ++d) {
        float diff = queries[q * params.dimension + d] - centroids[c * params.dimension + d];
        dist += diff * diff;
    }

    distances[q * params.num_centroids + c] = dist;
}

// After distances computed, select top-nprobe per query
// Can reuse existing TopKSelectionKernel or write specialized version
```

**Step 2: Create Swift Wrapper**

File: `Sources/VectorAccelerate/Index/Kernels/IVF/IVFGPUCoarseKernel.swift`

```swift
public final class IVFGPUCoarseKernel: Metal4Kernel {
    public let context: Metal4Context
    private let distancePipeline: MTLComputePipelineState
    private let topKKernel: TopKSelectionKernel

    public struct CoarseResult {
        let nearestCentroids: any MTLBuffer  // [Q × nprobe] UInt32
        let centroidDistances: any MTLBuffer // [Q × nprobe] Float (optional)
    }

    public func execute(
        queries: any MTLBuffer,
        centroids: any MTLBuffer,
        numQueries: Int,
        numCentroids: Int,
        dimension: Int,
        nprobe: Int
    ) async throws -> CoarseResult {
        // 1. Compute all query-centroid distances on GPU
        // 2. Use TopKSelectionKernel to find nprobe nearest per query
        // 3. Return indices buffer
    }
}
```

**Step 3: Create GPU Candidate List Builder**

File: `Sources/VectorAccelerate/Metal/Shaders/IVFCandidateBuilder.metal`

```metal
// Build CSR-format candidate list on GPU
// Input: nearest centroids per query, IVF list offsets
// Output: flat candidate indices, per-query offsets, query IDs

kernel void ivf_build_candidate_list(
    device const uint* nearest_centroids [[buffer(0)]],  // [Q × nprobe]
    device const uint* list_offsets [[buffer(1)]],       // [nlist + 1]
    device const uint* list_indices [[buffer(2)]],       // [total_vectors]
    device uint* candidate_ivf_indices [[buffer(3)]],    // Output: flat list
    device uint* candidate_query_ids [[buffer(4)]],      // Output: query ID per candidate
    device atomic_uint* candidate_counts [[buffer(5)]],  // Per-query counts (atomic)
    constant CandidateParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
);
```

**Step 4: Integrate into IVFSearchPipeline**

File: `Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift`

Replace CPU coarse search in `search()` method:
```swift
// Before (CPU):
let coarseResult = try await coarseQuantizer.execute(...)

// After (GPU):
let gpuCoarseResult = try await gpuCoarseKernel.execute(
    queries: queryBuffer,
    centroids: structure.centroids,
    numQueries: numQueries,
    numCentroids: structure.numCentroids,
    dimension: dimension,
    nprobe: nprobe
)
```

### Verification
- Run `IVFTests` - all should pass
- Run `IVFQualityAssessmentTests` - recall should be unchanged
- Profile to verify reduced CPU-GPU synchronization

---

## P4.1: K-Means++ Initialization

**Goal:** Improve centroid initialization for better clustering quality.

### Current Implementation
File: `Sources/VectorAccelerate/Index/Kernels/Clustering/KMeansPipeline.swift`

Currently uses random sampling for initial centroids.

### K-Means++ Algorithm

```swift
public struct KMeansPlusPlusInitializer {

    /// Initialize centroids using K-means++ algorithm
    /// - Parameters:
    ///   - vectors: Training vectors [N × D]
    ///   - k: Number of centroids (nlist)
    ///   - dimension: Vector dimension
    /// - Returns: Initial centroids [k × D]
    public func initialize(
        vectors: [[Float]],
        k: Int,
        dimension: Int
    ) -> [[Float]] {
        var centroids: [[Float]] = []
        let n = vectors.count

        // 1. Choose first centroid uniformly at random
        let firstIdx = Int.random(in: 0..<n)
        centroids.append(vectors[firstIdx])

        // 2. For each subsequent centroid
        for _ in 1..<k {
            // Compute D(x)² = squared distance to nearest centroid
            var distances = [Float](repeating: Float.infinity, count: n)
            var totalDistance: Float = 0

            for i in 0..<n {
                for centroid in centroids {
                    let dist = squaredL2Distance(vectors[i], centroid)
                    distances[i] = min(distances[i], dist)
                }
                totalDistance += distances[i]
            }

            // Choose next centroid with probability proportional to D(x)²
            let threshold = Float.random(in: 0..<totalDistance)
            var cumulative: Float = 0
            var nextIdx = 0

            for i in 0..<n {
                cumulative += distances[i]
                if cumulative >= threshold {
                    nextIdx = i
                    break
                }
            }

            centroids.append(vectors[nextIdx])
        }

        return centroids
    }

    private func squaredL2Distance(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { sum, pair in
            let diff = pair.0 - pair.1
            return sum + diff * diff
        }
    }
}
```

### GPU-Accelerated Version (Optional)

For large datasets, compute distances on GPU:

```metal
// Compute distance from each vector to nearest centroid
kernel void kmeans_pp_distances(
    device const float* vectors [[buffer(0)]],      // [N × D]
    device const float* centroids [[buffer(1)]],    // [k × D]
    device float* min_distances [[buffer(2)]],      // [N] output
    constant KMeansPPParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);
```

### Integration

File: `Sources/VectorAccelerate/Index/Kernels/Clustering/KMeansPipeline.swift`

```swift
public func train(vectors: [[Float]]) async throws -> [[Float]] {
    // Use K-means++ instead of random init
    let initializer = KMeansPlusPlusInitializer()
    var centroids = initializer.initialize(
        vectors: vectors,
        k: numClusters,
        dimension: dimension
    )

    // Continue with existing K-means iterations
    for _ in 0..<maxIterations {
        // ... existing assignment and update code
    }

    return centroids
}
```

### Verification
- Add test: `testKMeansPlusPlusConvergence`
- Run `IVFQualityAssessmentTests` - expect +5-10% recall improvement

---

## P2.2: Scalar Quantization Integration

**Goal:** Integrate existing scalar quantization into AcceleratedVectorIndex for 4x memory reduction.

### Current State
- `Sources/VectorAccelerate/Quantization/ScalarQuantization.swift` exists
- Not integrated with `AcceleratedVectorIndex`

### Implementation

**Step 1: Add Quantization Option to IndexConfiguration**

```swift
public enum StorageType: Sendable {
    case float32           // Default, full precision
    case int8Symmetric     // 4x compression, symmetric quantization
    case int8Asymmetric    // 4x compression, asymmetric quantization
}

public static func flat(
    dimension: Int,
    capacity: Int? = nil,
    storage: StorageType = .float32  // New parameter
) -> IndexConfiguration
```

**Step 2: Create QuantizedVectorStorage**

```swift
public actor QuantizedVectorStorage {
    private let quantizer: ScalarQuantizationKernel
    private var quantizedBuffer: MTLBuffer?  // Int8 storage
    private var scaleFactors: [Float]        // Per-dimension scale
    private var zeroPoints: [Float]          // For asymmetric

    public func insert(_ vector: [Float]) async throws -> UInt32
    public func readVector(at slot: UInt32) async throws -> [Float]  // Dequantize
}
```

**Step 3: Asymmetric Distance Computation**

For search, use asymmetric distance (query in float32, dataset in int8):

```metal
// Asymmetric L2 distance: float query vs int8 dataset
kernel void asymmetric_l2_distance(
    device const float* query [[buffer(0)]],        // [D] float32
    device const int8_t* vectors [[buffer(1)]],     // [N × D] int8
    device const float* scales [[buffer(2)]],       // [D] scale factors
    device const float* zero_points [[buffer(3)]],  // [D] zero points
    device float* distances [[buffer(4)]],          // [N] output
    constant AsymmetricParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_vectors) return;

    float dist = 0.0f;
    for (uint d = 0; d < params.dimension; ++d) {
        // Dequantize on-the-fly
        float v = float(vectors[tid * params.dimension + d]) * scales[d] + zero_points[d];
        float diff = query[d] - v;
        dist += diff * diff;
    }

    distances[tid] = dist;
}
```

### Verification
- Add test: `testQuantizedIndexRecall` - should be within 1-2% of float32
- Add test: `testQuantizedIndexMemory` - verify 4x reduction
- Benchmark throughput (may be faster due to memory bandwidth)

---

## P4.2: Adaptive nlist Selection

**Goal:** Provide sensible defaults so users don't need to tune nlist/nprobe.

### Implementation

File: `Sources/VectorAccelerate/Index/Types/IndexConfiguration.swift`

```swift
public struct IVFRecommendations {
    /// Recommended nlist based on dataset size
    /// Rule of thumb: nlist ≈ sqrt(N), clamped to [8, 4096]
    public static func recommendedNlist(for datasetSize: Int) -> Int {
        let sqrtN = Int(sqrt(Double(datasetSize)))
        return max(8, min(sqrtN, 4096))
    }

    /// Recommended nprobe for target recall
    /// - Parameters:
    ///   - nlist: Number of clusters
    ///   - targetRecall: Desired recall (0.0-1.0)
    /// - Returns: Recommended nprobe value
    public static func recommendedNprobe(
        for nlist: Int,
        targetRecall: Float = 0.9
    ) -> Int {
        // Empirical mapping based on benchmarks:
        // 80% recall ≈ 10% of nlist
        // 90% recall ≈ 20% of nlist
        // 95% recall ≈ 40% of nlist
        let fraction: Float
        switch targetRecall {
        case ..<0.8: fraction = 0.05
        case 0.8..<0.9: fraction = 0.10
        case 0.9..<0.95: fraction = 0.20
        case 0.95..<0.99: fraction = 0.40
        default: fraction = 1.0
        }

        return max(1, Int(Float(nlist) * fraction))
    }
}

// Convenience initializer
extension IndexConfiguration {
    /// Create IVF index with auto-tuned parameters
    public static func ivfAuto(
        dimension: Int,
        expectedDatasetSize: Int,
        targetRecall: Float = 0.9,
        capacity: Int? = nil
    ) -> IndexConfiguration {
        let nlist = IVFRecommendations.recommendedNlist(for: expectedDatasetSize)
        let nprobe = IVFRecommendations.recommendedNprobe(for: nlist, targetRecall: targetRecall)

        return .ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            capacity: capacity,
            minTrainingVectors: max(nlist * 10, 256)
        )
    }
}
```

### Verification
- Add test: `testIVFAutoConfiguration`
- Add test: `testRecommendedNlistScaling`
- Run quality assessment with auto-tuned params

---

## Key Files Reference

### Index Implementation
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift` - Main index API
- `Sources/VectorAccelerate/Index/Types/IndexConfiguration.swift` - Configuration
- `Sources/VectorAccelerate/Index/Kernels/IVF/IVFSearchPipeline.swift` - IVF search

### Metal Shaders
- `Sources/VectorAccelerate/Metal/Shaders/SearchAndRetrieval.metal` - Distance kernels
- `Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal` - TopK selection

### Tests
- `Tests/VectorAccelerateTests/IVFTests.swift` - IVF functionality
- `Tests/VectorAccelerateTests/IVFQualityAssessmentTests.swift` - Recall benchmarks
- `Tests/VectorAccelerateTests/IndexBenchmarkHarnessTests.swift` - Benchmark harness

### Documentation
- `docs/QUALITY_IMPROVEMENT_ROADMAP.md` - Full roadmap with completion status

---

## Running Tests

```bash
# All tests
swift test

# IVF tests only
swift test --filter "IVFTests"

# Quality assessment
swift test --filter "IVFQualityAssessmentTests"

# Benchmark harness
swift test --filter "IndexBenchmarkHarnessTests"

# Build release for benchmarks
swift build -c release
.build/release/VectorAccelerateBenchmarks --index
```

---

## Success Criteria

### P2.1 GPU Coarse Quantization
- [ ] All existing tests pass
- [ ] No CPU-side centroid distance computation in search path
- [ ] Profile shows reduced CPU-GPU sync points
- [ ] Recall unchanged from baseline

### P4.1 K-Means++ Initialization
- [ ] New test for K-means++ convergence
- [ ] Quality assessment shows +5-10% recall improvement
- [ ] Training time increase < 2x

### P2.2 Scalar Quantization
- [ ] New quantized storage option works
- [ ] Recall within 1-2% of float32
- [ ] Memory usage reduced by ~4x
- [ ] Throughput equal or better

### P4.2 Adaptive nlist
- [ ] `ivfAuto()` convenience initializer works
- [ ] Recommendations match quality assessment data
- [ ] Documentation updated

---

## Notes for Agent

1. **Read the roadmap first**: `docs/QUALITY_IMPROVEMENT_ROADMAP.md` has full context
2. **Run tests frequently**: 851 tests should always pass
3. **Use benchmark harness**: `IndexBenchmarkHarness` for recall/throughput measurements
4. **Metal 4 patterns**: Follow existing kernel structure in `Sources/VectorAccelerate/Kernels/Metal4/`
5. **Buffer pool**: Use `context.getBuffer()` for transient allocations (P1.3 already done)

Good luck! 🚀
