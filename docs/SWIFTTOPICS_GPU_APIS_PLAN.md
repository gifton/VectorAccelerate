# VectorAccelerate GPU Clustering APIs for SwiftTopics

> **Status:** Planned
> **Priority:** P2
> **Estimated Effort:** 6-9 days
> **Related:** [VECTORACCELERATE_GPU_OPPORTUNITIES.md](./VECTORACCELERATE_GPU_OPPORTUNITIES.md)

## Summary

Implement 3 GPU-accelerated clustering APIs to support SwiftTopics post-clustering operations:

| API | Effort | Description |
|-----|--------|-------------|
| **Soft Membership** | 1-2 days | Compose existing kernels (L2Distance + LogSumExp) |
| **Centroid Distances** | 3-4 days | New scatter-add kernel + distance computation |
| **Stability Accumulation** | 2-3 days | New CSR-style segment reduction kernel |

**Target Dimensions:** 512 (Apple NLContextualEmbedding), 384 (MiniLM)

---

## Phase 1: Soft Membership Probabilities (1-2 days)

### API
```swift
public func computeSoftMembership(
    embeddings: MTLBuffer,    // [N × D]
    centroids: MTLBuffer,     // [K × D]
    pointCount: Int,
    clusterCount: Int,
    dimension: Int,
    temperature: Float = 1.0
) async throws -> MTLBuffer  // [N × K] probabilities
```

### Implementation
**Composition only - no new Metal shaders needed:**
1. `L2DistanceKernel` → N×K distance matrix
2. Negate and scale by temperature
3. `LogSumExpKernel.encodeSoftmax()` → row-wise softmax

### Files to Create
- `/Sources/VectorAccelerate/Kernels/Metal4/SoftMembershipKernel.swift`
- `/Tests/VectorAccelerateTests/SoftMembershipKernelTests.swift`

### Test Cases
- Row-wise probability sum = 1.0
- Temperature scaling effect
- GPU vs CPU reference comparison
- Dimension-optimized pipeline selection (384, 512)

---

## Phase 2: Cluster Centroid Distances (3-4 days)

### API
```swift
public func computeClusterCentroidDistances(
    embeddings: MTLBuffer,     // [N × D]
    clusterLabels: MTLBuffer,  // [N] Int32, -1 = outlier
    clusterCount: Int,
    dimension: Int
) async throws -> MTLBuffer  // [N] Float distances
```

### Three-Phase GPU Pipeline
1. **Scatter-Add:** Atomic accumulation of embeddings per cluster label
2. **Normalize:** Divide centroid sums by cluster counts
3. **Distance:** L2 distance from each point to its assigned centroid

### Metal Kernels (new shader file)
```metal
// ClusterCentroidDistances.metal
kernel void scatter_add_centroids(...)     // Phase 1: atomic accumulation
kernel void normalize_centroids(...)       // Phase 2: divide by counts
kernel void point_to_centroid_distances(...)  // Phase 3: compute distances
kernel void point_to_centroid_distances_512(...)  // Optimized for D=512
kernel void point_to_centroid_distances_384(...)  // Optimized for D=384
```

### Files to Create
- `/Sources/VectorAccelerate/Metal/Shaders/ClusterCentroidDistances.metal`
- `/Sources/VectorAccelerate/Kernels/Metal4/ClusterCentroidDistanceKernel.swift`
- `/Tests/VectorAccelerateTests/ClusterCentroidDistanceKernelTests.swift`

### Test Cases
- Outlier handling (label = -1)
- Empty cluster handling
- GPU vs CPU reference comparison
- Large-scale performance (10K+ points)

---

## Phase 3: Stability Accumulation (2-3 days)

### API
```swift
public func computeStabilities(
    birthLevels: MTLBuffer,        // [M] Float
    deathLevels: MTLBuffer,        // [M] Float
    leafDeathDistances: MTLBuffer, // Flattened array
    leafOffsets: MTLBuffer,        // [M+1] CSR-style offsets
    nodeCount: Int
) async throws -> MTLBuffer  // [M] Float stabilities
```

### CSR-Style Segment Reduction
Each threadgroup processes one node's variable-length segment:
```
stability[node] = lambda * sum(leafDeathDistances[offset[node]:offset[node+1]])
```

### Metal Kernel
```metal
// StabilityAccumulation.metal
kernel void segment_stability_reduction(
    // Each threadgroup = one node
    // Grid-stride loop over segment + SIMD reduction
)
```

### Files to Create
- `/Sources/VectorAccelerate/Metal/Shaders/StabilityAccumulation.metal`
- `/Sources/VectorAccelerate/Kernels/Metal4/StabilityAccumulationKernel.swift`
- `/Tests/VectorAccelerateTests/StabilityAccumulationKernelTests.swift`

### Test Cases
- Variable-length segments
- Empty segments (offset[i] == offset[i+1])
- Large segments requiring grid-stride
- GPU vs CPU reference comparison

---

## Phase 4: Integration (1 day)

1. Export public APIs from `VectorAccelerate.swift`
2. Add benchmarks to existing benchmark suite
3. Update module documentation

---

## Critical Reference Files

| Purpose | File |
|---------|------|
| Kernel protocols | `Kernels/Metal4/KernelProtocol.swift` |
| Dimension optimization pattern | `Kernels/Metal4/L2DistanceKernel.swift` |
| Softmax composition | `Kernels/Metal4/LogSumExpKernel.swift` |
| Atomic accumulation pattern | `Metal/Shaders/ClusteringShaders.metal` |
| Common shader utilities | `Metal/Shaders/Metal4Common.h` |
| Test patterns | `Tests/VectorAccelerateTests/ClusteringKernelTests.swift` |

---

## Implementation Checklist

### Phase 1: Soft Membership
- [ ] Create `SoftMembershipKernel.swift` composing L2Distance + LogSumExp
- [ ] Implement `encode()` API for kernel fusion
- [ ] Add dimension optimization passthrough
- [ ] Write `SoftMembershipKernelTests.swift`
- [ ] Add benchmarks

### Phase 2: Centroid Distances
- [ ] Create `ClusterCentroidDistances.metal` with 3 kernels
- [ ] Implement dimension-optimized distance variants (384, 512)
- [ ] Create `ClusterCentroidDistanceKernel.swift` wrapper
- [ ] Write `ClusterCentroidDistanceKernelTests.swift`
- [ ] Add benchmarks

### Phase 3: Stability Accumulation
- [ ] Create `StabilityAccumulation.metal` with segment reduction
- [ ] Create `StabilityAccumulationKernel.swift` wrapper
- [ ] Write `StabilityAccumulationKernelTests.swift`
- [ ] Add benchmarks

### Phase 4: Integration
- [ ] Export public APIs
- [ ] Update documentation
- [ ] Final validation with SwiftTopics integration test
