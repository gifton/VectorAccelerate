# VectorIndexAccelerated → VectorAccelerate Migration Plan

**Version**: 1.0
**Created**: 2024-12-02
**Target VA Version**: 0.3.0
**Status**: Phase 7 Complete - Ready for Release ✅

---

## Executive Summary

This document provides a comprehensive plan for migrating VectorIndexAccelerated (VIA) functionality into VectorAccelerate (VA). The goal is to consolidate all GPU acceleration code into a single package, eliminating ~70% code duplication and simplifying the dependency graph.

### Current State
```
VectorCore (0.1.6)
    ↑
VectorIndex (0.1.3) ←──────────────┐
    ↑                              │
VectorAccelerate (0.2.0)           │
    ↑                              │
VectorIndexAccelerated ────────────┘
```

### Target State
```
VectorCore (0.1.6)
    ↑
VectorIndex (0.1.3)
    ↑
VectorAccelerate (0.3.0) ← All GPU code consolidated here
```

---

## Table of Contents

1. [Pre-Migration Analysis](#1-pre-migration-analysis)
2. [File Classification](#2-file-classification)
3. [Dependency Changes](#3-dependency-changes)
4. [Directory Structure](#4-directory-structure)
5. [Migration Phases](#5-migration-phases)
6. [Code Transformations](#6-code-transformations)
7. [API Surface](#7-api-surface)
8. [Test Migration](#8-test-migration)
9. [Breaking Changes](#9-breaking-changes)
10. [Rollback Plan](#10-rollback-plan)
11. [Validation Checklist](#11-validation-checklist)

---

## 1. Pre-Migration Analysis

### 1.1 VIA Codebase Statistics

| Category | Files | Lines | Action |
|----------|-------|-------|--------|
| **Total** | ~100 | ~48,350 | - |
| Deprecated (`.deprecated/`) | 22 | ~10,500 | DELETE |
| Duplicate with VA | 18 | ~8,200 | DELETE |
| Unique - Migrate | 35 | ~15,000 | MIGRATE |
| Network/Distributed | 15 | ~6,500 | DEFER (separate package) |
| Utilities/Support | 10 | ~2,000 | EVALUATE |

### 1.2 VIA Dependencies on VA

VIA currently duplicates these VA components:
- Distance kernels (L2, Cosine, DotProduct, Hamming)
- Buffer pool management
- Metal device/context infrastructure
- Pipeline caching
- TopK selection

### 1.3 Unique VIA Components Worth Migrating

1. **HNSW GPU Kernels** (~3,500 lines)
   - HNSWDistanceMatrixKernel
   - HNSWEdgeInsertionKernel
   - HNSWEdgePruningKernel
   - HNSWVisitedSetKernel
   - HNSWLevelAssignmentKernel
   - HNSWSearchKernel
   - HNSWDistanceCacheKernel

2. **IVF GPU Components** (~1,600 lines)
   - IVFGPUProcessorEnhanced
   - IVFIndexAccelerated wrapper

3. **KMeans GPU Kernels** (~1,500 lines)
   - KMeansAssignPointsKernel
   - KMeansUpdateCentroidsKernel
   - KMeansConvergenceKernel

4. **Accelerated Index Wrappers** (~1,600 lines)
   - HNSWIndexAccelerated
   - FlatIndexAccelerated
   - IVFIndexAccelerated

5. **MPS Integration** (~2,000 lines)
   - MPSDistanceEngine
   - MPSSelectionEngine
   - MPSVectorProcessor

---

## 2. File Classification

### 2.1 DELETE - Deprecated Code

```
.deprecated/
├── BatchProcessor.swift
├── GPUMemoryManager.swift
├── HNSWGraphBuilder.swift
├── HNSWIndexAccelerated.swift (old version)
├── IVFIndexAccelerated.swift (old version)
├── IndexMPSKernels.swift
├── LargeBatch/
│   ├── BufferPool.swift
│   ├── DoubleBufferManager.swift
│   ├── LargeBatchBenchmark.swift
│   ├── LargeBatchMemoryStrategy.swift
│   ├── LargeBatchProcessor.swift
│   └── StreamingEngine.swift
└── Streaming/
    ├── DMATransferEngine.swift
    ├── MetalPipelineManager.swift
    ├── PrefetchEngine.swift
    ├── StreamingBenchmark.swift
    ├── StreamingPipeline.swift
    ├── TripleBufferCoordinator.swift
    └── WriteBehindBuffer.swift
```

### 2.2 DELETE - Duplicates with VA

```
Metal/Core/
├── MetalDevice.swift          → Use Metal4Context
├── MetalBufferManager.swift   → Use MetalBufferFactory
├── MetalPipelineManager.swift → Use PipelineCache
├── MetalDeviceCapabilities.swift → Use Metal4Capabilities
├── MetalKernelLoader.swift    → Use Metal4ShaderCompiler
└── MetalError.swift           → Use Metal4Error + VectorError

Metal/Kernels/
├── L2DistanceKernel.swift     → Use VA's L2DistanceKernel
├── CosineSimilarityKernel.swift → Use VA's CosineSimilarityKernel
├── DotProductKernel.swift     → Use VA's DotProductKernel
├── HammingDistanceKernel.swift → Use VA's HammingDistanceKernel
├── BitonicSortKernel.swift    → Use VA's TopKSelectionKernel
├── RadixSortKernel.swift      → Evaluate vs VA's selection kernels
├── KSelectionKernel.swift     → Use VA's TopKSelectionKernel
└── PartialSortTopKKernel.swift → Use VA's StreamingTopKKernel

GPU/
├── GPUBufferPool.swift        → Use VA's BufferPool/SmartBufferPool
├── GPUDecisionEngine.swift    → Use VA's HybridExecutionStrategy
└── GPUErrorHandling.swift     → Use VA's error types
```

### 2.3 MIGRATE - Unique Index Acceleration Code

```
# HNSW Kernels → VA/Kernels/IndexAcceleration/HNSW/
Metal/Kernels/
├── HNSWDistanceCacheKernel.swift
├── HNSWEdgeInsertionKernel.swift
├── HNSWEdgePruningKernel.swift
├── HNSWVisitedSetKernel.swift
├── HNSWLevelAssignmentKernel.swift
└── HNSWSearchKernel.swift

Kernels/HNSW/
└── HNSWDistanceMatrixKernel.swift

# IVF Components → VA/Kernels/IndexAcceleration/IVF/
GPU/
└── IVFGPUProcessorEnhanced.swift

# KMeans Kernels → VA/Kernels/IndexAcceleration/Clustering/
Metal/Kernels/
├── KMeansAssignPointsKernel.swift
├── KMeansUpdateCentroidsKernel.swift
├── KMeansConvergenceKernel.swift
└── ListCompactionKernel.swift

# Accelerated Index Wrappers → VA/IndexAcceleration/
Indexes/
└── HNSWIndexAccelerated.swift
FlatIndexAccelerated.swift
IVFIndexAccelerated.swift

# MPS Integration → VA/MPS/
GPU/
├── MPSDistanceEngine.swift
├── MPSSelectionEngine.swift
├── MPSVectorProcessor.swift
├── MPSBufferManager.swift
└── MPSError.swift

# Advanced Query Processing → VA/IndexAcceleration/
Advanced/
├── AsyncOperationManager.swift
├── BatchQueryProcessor.swift
├── QueryOptimizer.swift
└── PerformanceMetrics.swift

# Memory Optimization → VA/Optimization/
Memory/
├── MemoryLayoutOptimizer.swift
├── MemoryAccessAnalyzer.swift
└── BatchPackingOptimizer.swift

Performance/
├── ZeroCopyTransfer.swift
├── BandwidthOptimizer.swift
└── SIMDSerializer.swift
```

### 2.4 DEFER - Network/Distributed (Separate Package Later)

```
# These should become a separate VectorDistributed package
Network/
├── NetworkManager.swift
├── FaultTolerance.swift
├── ResilientConnectionManager.swift
├── NetworkObservability.swift
├── TLSConfiguration.swift
├── ProductionTLSManager.swift
├── MutualTLSManager.swift
├── CertificateParser.swift
├── CRLValidator.swift
├── OCSPValidator.swift
└── MessageEncryption.swift

Distributed/
└── DistributedCoordinator.swift

Partitioning/
└── PartitionManager.swift

Persistence/
├── AdvancedPersistence.swift
├── WriteAheadLog.swift
├── DeltaManager.swift
└── DataCompression.swift
```

---

## 3. Dependency Changes

### 3.1 VectorAccelerate Package.swift Changes

```swift
// Before (VA 0.2.0)
let package = Package(
    name: "VectorAccelerate",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        // ...
    ],
    dependencies: [
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.6")
    ],
    // ...
)

// After (VA 0.3.0)
let package = Package(
    name: "VectorAccelerate",
    platforms: [
        .macOS(.v15),  // Runtime requires macOS 26 for Metal 4
        .iOS(.v18),    // Runtime requires iOS 26 for Metal 4
        // ...
    ],
    products: [
        .library(name: "VectorAccelerate", targets: ["VectorAccelerate"]),
        // New: Separate product for index acceleration
        .library(name: "VectorIndexAcceleration", targets: ["VectorIndexAcceleration"]),
    ],
    dependencies: [
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.6"),
        // NEW: Add VectorIndex dependency
        .package(url: "https://github.com/gifton/VectorIndex", from: "0.1.3")
    ],
    targets: [
        // Core GPU acceleration (no VectorIndex dependency)
        .target(
            name: "VectorAccelerate",
            dependencies: ["VectorCore"],
            // ...
        ),
        // Index acceleration (depends on VectorIndex)
        .target(
            name: "VectorIndexAcceleration",
            dependencies: [
                "VectorAccelerate",
                .product(name: "VectorIndex", package: "VectorIndex")
            ],
            path: "Sources/VectorIndexAcceleration"
        ),
        // ...
    ]
)
```

### 3.2 Rationale for Separate Target

Keeping `VectorIndexAcceleration` as a separate target/product allows:
1. Users who only need vector operations can use `VectorAccelerate` without VectorIndex
2. Users who need accelerated indices import `VectorIndexAcceleration`
3. No circular dependencies
4. Clear separation of concerns

---

## 4. Directory Structure

### 4.1 New VectorAccelerate Structure

```
VectorAccelerate/
├── Sources/
│   ├── VectorAccelerate/           # Core GPU acceleration (existing)
│   │   ├── Core/
│   │   │   ├── Metal4Context.swift
│   │   │   ├── MetalDevice.swift
│   │   │   ├── BufferPool.swift
│   │   │   ├── SmartBufferPool.swift
│   │   │   ├── PipelineCache.swift
│   │   │   └── ...
│   │   ├── Kernels/
│   │   │   ├── Metal4/
│   │   │   │   ├── L2DistanceKernel.swift
│   │   │   │   ├── CosineSimilarityKernel.swift
│   │   │   │   ├── TopKSelectionKernel.swift
│   │   │   │   ├── FusedL2TopKKernel.swift
│   │   │   │   └── ...
│   │   │   └── KernelProtocol.swift
│   │   ├── Operations/
│   │   ├── ML/
│   │   ├── Integration/
│   │   │   └── VectorCoreIntegration.swift
│   │   ├── Configuration/
│   │   └── Benchmarking/
│   │
│   ├── VectorIndexAcceleration/    # NEW: Index acceleration module
│   │   ├── Core/
│   │   │   ├── AcceleratedIndexProtocol.swift
│   │   │   └── IndexAccelerationContext.swift
│   │   ├── Kernels/
│   │   │   ├── HNSW/
│   │   │   │   ├── HNSWDistanceMatrixKernel.swift
│   │   │   │   ├── HNSWEdgeInsertionKernel.swift
│   │   │   │   ├── HNSWEdgePruningKernel.swift
│   │   │   │   ├── HNSWVisitedSetKernel.swift
│   │   │   │   ├── HNSWLevelAssignmentKernel.swift
│   │   │   │   ├── HNSWSearchKernel.swift
│   │   │   │   └── HNSWDistanceCacheKernel.swift
│   │   │   ├── IVF/
│   │   │   │   ├── IVFGPUProcessor.swift
│   │   │   │   └── IVFSearchKernel.swift
│   │   │   └── Clustering/
│   │   │       ├── KMeansAssignPointsKernel.swift
│   │   │       ├── KMeansUpdateCentroidsKernel.swift
│   │   │       ├── KMeansConvergenceKernel.swift
│   │   │       └── ListCompactionKernel.swift
│   │   ├── Indexes/
│   │   │   ├── HNSWIndexAccelerated.swift
│   │   │   ├── IVFIndexAccelerated.swift
│   │   │   └── FlatIndexAccelerated.swift
│   │   ├── MPS/
│   │   │   ├── MPSDistanceEngine.swift
│   │   │   ├── MPSSelectionEngine.swift
│   │   │   └── MPSVectorProcessor.swift
│   │   ├── Optimization/
│   │   │   ├── QueryOptimizer.swift
│   │   │   ├── BatchQueryProcessor.swift
│   │   │   ├── MemoryLayoutOptimizer.swift
│   │   │   └── ZeroCopyTransfer.swift
│   │   └── Extensions/
│   │       └── AccelerableIndexExtensions.swift
│   │
│   └── VectorAccelerateBenchmarks/
│
├── Tests/
│   ├── VectorAccelerateTests/
│   └── VectorIndexAccelerationTests/  # NEW
│
└── docs/
    ├── metal4_migration_guide.md
    └── index_acceleration.md          # NEW
```

---

## 5. Migration Phases

### Phase 1: Preparation (VA 0.2.1)
**Duration**: 1-2 days
**Risk**: Low

1. [ ] Create `VectorIndexAcceleration` target in VA Package.swift
2. [ ] Add VectorIndex dependency (from: "0.1.3")
3. [ ] Create directory structure under `Sources/VectorIndexAcceleration/`
4. [ ] Add placeholder files to verify build
5. [ ] Update CI to build both targets
6. [ ] **Checkpoint**: VA builds with new structure, no VIA code yet

### Phase 2: Core Index Kernels (VA 0.2.2)
**Duration**: 2-3 days
**Risk**: Medium

1. [ ] Migrate HNSW kernels:
   - [ ] HNSWDistanceMatrixKernel.swift
   - [ ] HNSWEdgeInsertionKernel.swift
   - [ ] HNSWEdgePruningKernel.swift
   - [ ] HNSWVisitedSetKernel.swift
   - [ ] HNSWLevelAssignmentKernel.swift
   - [ ] HNSWSearchKernel.swift
   - [ ] HNSWDistanceCacheKernel.swift
2. [ ] Adapt to use Metal4Context instead of VIA's MetalDevice
3. [ ] Conform to Metal4Kernel protocol where applicable
4. [ ] Migrate corresponding .metal shader files
5. [ ] **Checkpoint**: HNSW kernels compile and basic tests pass

### Phase 3: IVF & Clustering Kernels (VA 0.2.3)
**Duration**: 1-2 days
**Risk**: Medium

1. [ ] Migrate KMeans kernels:
   - [ ] KMeansAssignPointsKernel.swift
   - [ ] KMeansUpdateCentroidsKernel.swift
   - [ ] KMeansConvergenceKernel.swift
   - [ ] ListCompactionKernel.swift
2. [ ] Migrate IVF components:
   - [ ] IVFGPUProcessor.swift (from IVFGPUProcessorEnhanced)
3. [ ] **Checkpoint**: All core kernels migrated

### Phase 4: Accelerated Index Wrappers (VA 0.2.4)
**Duration**: 2-3 days
**Risk**: High (API-facing)

1. [ ] Migrate index wrappers:
   - [ ] HNSWIndexAccelerated.swift
   - [ ] IVFIndexAccelerated.swift
   - [ ] FlatIndexAccelerated.swift
2. [ ] Update to use VA's Metal4Context
3. [ ] Update to use VA's distance kernels
4. [ ] Implement AccelerableIndex protocol extensions
5. [ ] **Checkpoint**: Can create and search accelerated indices

### Phase 5: MPS Integration (VA 0.2.5)
**Duration**: 1 day
**Risk**: Low

1. [ ] Migrate MPS components:
   - [ ] MPSDistanceEngine.swift
   - [ ] MPSSelectionEngine.swift
   - [ ] MPSVectorProcessor.swift
2. [ ] Integrate with Metal4Context
3. [ ] **Checkpoint**: MPS fallback works on older devices

### Phase 6: Optimization & Advanced Features (VA 0.2.6)
**Duration**: 2 days
**Risk**: Medium

1. [ ] Migrate optimization code:
   - [ ] QueryOptimizer.swift
   - [ ] BatchQueryProcessor.swift
   - [ ] MemoryLayoutOptimizer.swift
   - [ ] ZeroCopyTransfer.swift
   - [ ] BandwidthOptimizer.swift
2. [ ] Integrate with VA's existing performance infrastructure
3. [ ] **Checkpoint**: Full feature parity with VIA

### Phase 7: Testing & Documentation (VA 0.2.7)
**Duration**: 2-3 days
**Risk**: Low

1. [ ] Migrate all VIA tests to VA
2. [ ] Add integration tests for AccelerableIndex extensions
3. [ ] Benchmark comparison: new VA vs old VIA
4. [ ] Write migration guide for VIA users
5. [ ] Update all documentation
6. [ ] **Checkpoint**: All tests pass, docs complete

### Phase 8: Release (VA 0.3.0)
**Duration**: 1 day
**Risk**: Low

1. [ ] Final code review
2. [ ] Update CHANGELOG
3. [ ] Tag release 0.3.0
4. [ ] Deprecate VIA package (add deprecation notice)
5. [ ] **Checkpoint**: VA 0.3.0 released with index acceleration

---

## 6. Code Transformations

### 6.1 MetalDevice → Metal4Context

**Before (VIA):**
```swift
public actor HNSWIndexAccelerated {
    private var device: MetalDevice?

    public func prepareForGPU() async throws {
        device = try await MetalDevice.createDefault()
        // ...
    }

    private func executeGPUSearch(...) async throws {
        guard let commandBuffer = await device.createCommandBuffer() else {
            throw MetalError.commandBufferCreationFailed
        }
        // ...
    }
}
```

**After (VA):**
```swift
@available(macOS 26.0, iOS 26.0, *)
public actor HNSWIndexAccelerated {
    private var context: Metal4Context?

    public func prepareForGPU() async throws {
        context = try await Metal4Context()
        // ...
    }

    private func executeGPUSearch(...) async throws {
        guard let context = context else {
            throw VectorError.notInitialized("GPU context")
        }

        try await context.executeAndWait { commandBuffer, encoder in
            // ...
        }
    }
}
```

### 6.2 Buffer Management

**Before (VIA):**
```swift
let queriesBuffer = try await device.allocateBuffer(with: flatQueries)
let datasetBuffer = try await device.allocateBuffer(with: flatDataset)
```

**After (VA):**
```swift
let queriesBuffer = try await context.bufferPool.getBuffer(
    for: flatQueries,
    label: "queries"
)
let datasetBuffer = try await context.bufferPool.getBuffer(
    for: flatDataset,
    label: "dataset"
)
```

### 6.3 Distance Kernel Usage

**Before (VIA):**
```swift
let l2Kernel = L2DistanceKernel(device: device, configuration: config)
try await l2Kernel.setup()
let result = try await l2Kernel.computeDistances(queries: queries, dataset: dataset)
```

**After (VA):**
```swift
let l2Kernel = try await L2DistanceKernel(context: context)
let distanceBuffer = try await l2Kernel.execute(
    queries: queriesBuffer,
    database: datasetBuffer,
    parameters: L2DistanceParameters(
        numQueries: queries.count,
        numDatabase: dataset.count,
        dimension: dimension
    )
)
```

### 6.4 Kernel Fusion

**Before (VIA):** Separate distance + TopK calls
```swift
let distances = try await distanceKernel.compute(...)
let topK = try await selectionKernel.select(distances, k: k)
```

**After (VA):** Fused single-pass
```swift
let fusedKernel = try await FusedL2TopKKernel(context: context)
let (indices, distances) = try await fusedKernel.execute(
    queries: queriesBuffer,
    database: datasetBuffer,
    parameters: FusedL2TopKParameters(
        numQueries: queries.count,
        numDataset: dataset.count,
        dimension: dimension,
        k: k
    )
)
```

### 6.5 AccelerableIndex Extension

**New code to add:**
```swift
// Sources/VectorIndexAcceleration/Extensions/AccelerableIndexExtensions.swift

import VectorIndex
import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension AccelerableIndex {

    /// GPU-accelerated search using Metal 4 kernels
    ///
    /// This extension automatically accelerates any index conforming to AccelerableIndex
    /// using VectorAccelerate's optimized GPU kernels.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of results
    ///   - filter: Optional metadata filter
    ///   - context: Optional pre-created Metal4Context (created if nil)
    /// - Returns: Search results sorted by distance
    public func searchAccelerated(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil,
        context: Metal4Context? = nil
    ) async throws -> [SearchResult] {
        // Check if acceleration is beneficial
        let candidateEstimate = await estimateCandidateCount(for: k)
        guard await shouldAccelerate(queryCount: 1, candidateCount: candidateEstimate, k: k) else {
            // Fall back to CPU search
            return try await search(query: query, k: k, filter: filter)
        }

        // Get or create context
        let ctx = try context ?? await Metal4Context()

        // Get candidates from index
        let candidates = try await getCandidates(query: query, k: k, filter: filter)

        // Use fused L2+TopK for efficiency
        let fusedKernel = try await FusedL2TopKKernel(context: ctx)

        // Create buffers
        let queryBuffer = try ctx.bufferFactory.makeBuffer(
            from: query,
            label: "query"
        )

        let candidateBuffer = try candidates.withUnsafeVectorBuffer { ptr in
            try ctx.bufferFactory.makeBuffer(
                bytes: ptr.baseAddress!,
                length: ptr.count * MemoryLayout<Float>.size,
                label: "candidates"
            )
        }

        // Execute fused kernel
        let (indices, distances) = try await fusedKernel.execute(
            queries: queryBuffer,
            database: candidateBuffer,
            parameters: FusedL2TopKParameters(
                numQueries: 1,
                numDataset: candidates.vectorCount,
                dimension: candidates.dimension,
                k: k
            )
        )

        // Convert back to SearchResults
        let acceleratedResults = AcceleratedResults(
            indices: indices[0],
            distances: distances[0]
        )

        return await finalizeResults(
            candidates: candidates,
            results: acceleratedResults,
            filter: filter
        )
    }

    /// Batch GPU-accelerated search
    public func batchSearchAccelerated(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil,
        context: Metal4Context? = nil
    ) async throws -> [[SearchResult]] {
        let ctx = try context ?? await Metal4Context()

        // Get candidates for all queries
        let batchCandidates = try await getBatchCandidates(
            queries: queries,
            k: k,
            filter: filter
        )

        // Process in GPU
        var allResults: [[SearchResult]] = []

        for (query, candidates) in zip(queries, batchCandidates) {
            let fusedKernel = try await FusedL2TopKKernel(context: ctx)

            let queryBuffer = try ctx.bufferFactory.makeBuffer(from: query, label: "query")
            let candidateBuffer = try candidates.withUnsafeVectorBuffer { ptr in
                try ctx.bufferFactory.makeBuffer(
                    bytes: ptr.baseAddress!,
                    length: ptr.count * MemoryLayout<Float>.size,
                    label: "candidates"
                )
            }

            let (indices, distances) = try await fusedKernel.execute(
                queries: queryBuffer,
                database: candidateBuffer,
                parameters: FusedL2TopKParameters(
                    numQueries: 1,
                    numDataset: candidates.vectorCount,
                    dimension: candidates.dimension,
                    k: k
                )
            )

            let acceleratedResults = AcceleratedResults(
                indices: indices[0],
                distances: distances[0]
            )

            let results = await finalizeResults(
                candidates: candidates,
                results: acceleratedResults,
                filter: filter
            )

            allResults.append(results)
        }

        return allResults
    }

    private func estimateCandidateCount(for k: Int) async -> Int {
        // Heuristic based on index structure
        switch await getIndexStructure() {
        case .hnsw(let structure):
            return min(structure.maxLevel * 100 + k * 10, 10000)
        case .ivf(let structure):
            return structure.nprobe * (structure.invertedLists.first?.count ?? 100)
        case .flat:
            return 10000 // Will be refined by actual index size
        }
    }
}
```

---

## 7. API Surface

### 7.1 New Public Types in VectorIndexAcceleration

```swift
// Index Wrappers
public actor HNSWIndexAccelerated: VectorIndexProtocol
public actor IVFIndexAccelerated: VectorIndexProtocol
public actor FlatIndexAccelerated: VectorIndexProtocol

// HNSW Kernels (internal use, but public for advanced users)
@available(macOS 26.0, iOS 26.0, *)
public final class HNSWDistanceMatrixKernel: Metal4Kernel
public final class HNSWSearchKernel: Metal4Kernel
public final class HNSWEdgeInsertionKernel: Metal4Kernel
// ... etc

// Configuration
public struct IndexAccelerationConfiguration: Sendable {
    public var useGPU: Bool
    public var gpuThreshold: Int
    public var batchSize: Int
    public var useFusedKernels: Bool
    public var enableProfiling: Bool
}

// Extensions on VectorIndex types
extension AccelerableIndex {
    public func searchAccelerated(...) async throws -> [SearchResult]
    public func batchSearchAccelerated(...) async throws -> [[SearchResult]]
}
```

### 7.2 Deprecation Path for VIA Users

VIA will be deprecated but maintained for one release cycle. Users should migrate:

```swift
// Old (VIA)
import VectorIndexAccelerated
let index = HNSWIndexAccelerated(baseIndex: hnswIndex)

// New (VA 0.3.0)
import VectorIndex
import VectorIndexAcceleration

// Option 1: Use extension on existing index
let results = try await hnswIndex.searchAccelerated(query: query, k: 10)

// Option 2: Use explicit wrapper
let acceleratedIndex = HNSWIndexAccelerated(baseIndex: hnswIndex)
```

---

## 8. Test Migration

### 8.1 Tests to Migrate

```
VIA Tests → VA Tests Location
─────────────────────────────────────────────────────────
HNSWAcceleratedTests.swift      → VectorIndexAccelerationTests/
IVFAcceleratedTests.swift       → VectorIndexAccelerationTests/
FlatAcceleratedTests.swift      → VectorIndexAccelerationTests/
GPUDistanceTests.swift          → DELETE (covered by VA tests)
BufferPoolTests.swift           → DELETE (covered by VA tests)
MetalDeviceTests.swift          → DELETE (covered by VA tests)
PerformanceBenchmarks.swift     → VectorIndexAccelerationTests/Benchmarks/
IntegrationTests.swift          → VectorIndexAccelerationTests/
```

### 8.2 New Tests to Add

```swift
// Tests/VectorIndexAccelerationTests/AccelerableIndexExtensionTests.swift

@available(macOS 26.0, iOS 26.0, *)
final class AccelerableIndexExtensionTests: XCTestCase {

    func testHNSWAcceleratedSearch() async throws {
        let index = try HNSWIndex(dimension: 128, metric: .euclidean)

        // Add vectors
        for i in 0..<10000 {
            let vector = (0..<128).map { _ in Float.random(in: -1...1) }
            try await index.add(id: VectorID(), vector: vector)
        }

        // Test accelerated search
        let query = (0..<128).map { _ in Float.random(in: -1...1) }
        let results = try await index.searchAccelerated(query: query, k: 10)

        XCTAssertEqual(results.count, 10)

        // Verify results match CPU search
        let cpuResults = try await index.search(query: query, k: 10, filter: nil)
        XCTAssertEqual(results.map(\.id), cpuResults.map(\.id))
    }

    func testFusedKernelPerformance() async throws {
        // Benchmark fused vs separate kernels
        // ...
    }
}
```

---

## 9. Breaking Changes

### 9.1 For VIA Users

| Change | Migration |
|--------|-----------|
| Package renamed | `import VectorIndexAcceleration` instead of `import VectorIndexAccelerated` |
| MetalDevice removed | Use `Metal4Context` from VectorAccelerate |
| GPUBufferPool removed | Use `BufferPool` from VectorAccelerate |
| Distance kernels removed | Use kernels from VectorAccelerate |
| Namespace changes | `VectorIndexAccelerated.X` → `VectorIndexAcceleration.X` |

### 9.2 Compatibility Shim (Optional)

If needed, provide a thin compatibility layer in VIA for one release:

```swift
// In deprecated VIA package
@available(*, deprecated, renamed: "VectorIndexAcceleration")
public typealias VectorIndexAccelerated = VectorIndexAcceleration

@available(*, deprecated, message: "Use Metal4Context from VectorAccelerate")
public typealias MetalDevice = VectorAccelerate.Metal4Context
```

---

## 10. Rollback Plan

If critical issues are discovered post-migration:

1. **Phase 1-4 Issues**: Revert commits, continue using VIA
2. **Phase 5-7 Issues**: Release VA 0.3.0-beta, fix issues, re-release
3. **Post-Release Issues**:
   - VA 0.3.1 hotfix for critical bugs
   - VIA can remain usable (just deprecated)

### Rollback Triggers

- Test failure rate > 5%
- Performance regression > 20%
- Critical security vulnerability
- Incompatibility with VectorCore/VectorIndex

---

## 11. Validation Checklist

### Pre-Migration
- [ ] VectorAccelerate 0.2.0 is stable
- [ ] VectorIndex 0.1.3 is stable
- [ ] VectorCore 0.1.6 is stable
- [ ] All VIA tests pass on current main
- [ ] Benchmark baseline established

### Phase Checkpoints
- [x] Phase 1: VA builds with new target structure
- [x] Phase 2: Core types implemented
- [x] Phase 3: Flat index implementation
- [x] Phase 4: IVF index implementation
- [x] Phase 5: Cleanup old code
- [x] Phase 6: All optimizations integrated
- [x] Phase 7: 100% test pass rate (871 tests)
- [ ] Phase 8: Release (in progress)

### Pre-Release
- [x] All unit tests pass (871 tests)
- [x] Integration tests pass
- [x] Performance benchmarks meet or exceed VIA
- [x] Memory leak tests pass
- [x] API documentation generated
- [x] CHANGELOG updated
- [x] Migration guide published (see CLEAN_API_REDESIGN.md)

### Post-Release
- [ ] GitHub release created
- [ ] VIA deprecation notice published
- [ ] Monitor for issues (1 week)
- [ ] Gather user feedback

---

## Appendix A: File-by-File Migration Map

| VIA File | Action | VA Destination | Notes |
|----------|--------|----------------|-------|
| `Metal/Core/MetalDevice.swift` | DELETE | N/A | Use Metal4Context |
| `Metal/Core/MetalBufferManager.swift` | DELETE | N/A | Use MetalBufferFactory |
| `Metal/Core/MetalPipelineManager.swift` | DELETE | N/A | Use PipelineCache |
| `Metal/Core/MetalDeviceCapabilities.swift` | DELETE | N/A | Use Metal4Capabilities |
| `Metal/Core/MetalKernelLoader.swift` | DELETE | N/A | Use Metal4ShaderCompiler |
| `Metal/Core/MetalError.swift` | DELETE | N/A | Use Metal4Error |
| `Metal/Core/MetalResultTypes.swift` | MIGRATE | Core/ | Useful result types |
| `Metal/Core/MetalResultExtractors.swift` | MIGRATE | Core/ | Result extraction |
| `Metal/Kernels/L2DistanceKernel.swift` | DELETE | N/A | Use VA's version |
| `Metal/Kernels/CosineSimilarityKernel.swift` | DELETE | N/A | Use VA's version |
| `Metal/Kernels/DotProductKernel.swift` | DELETE | N/A | Use VA's version |
| `Metal/Kernels/HammingDistanceKernel.swift` | DELETE | N/A | Use VA's version |
| `Metal/Kernels/BitonicSortKernel.swift` | DELETE | N/A | Use TopKSelectionKernel |
| `Metal/Kernels/RadixSortKernel.swift` | DELETE | N/A | Use TopKSelectionKernel |
| `Metal/Kernels/KSelectionKernel.swift` | DELETE | N/A | Use TopKSelectionKernel |
| `Metal/Kernels/PartialSortTopKKernel.swift` | DELETE | N/A | Use StreamingTopKKernel |
| `Metal/Kernels/HNSWDistanceCacheKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/HNSWEdgeInsertionKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/HNSWEdgePruningKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/HNSWVisitedSetKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/HNSWLevelAssignmentKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/HNSWSearchKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `Metal/Kernels/KMeansAssignPointsKernel.swift` | MIGRATE | Kernels/Clustering/ | Unique |
| `Metal/Kernels/KMeansUpdateCentroidsKernel.swift` | MIGRATE | Kernels/Clustering/ | Unique |
| `Metal/Kernels/KMeansConvergenceKernel.swift` | MIGRATE | Kernels/Clustering/ | Unique |
| `Metal/Kernels/ListCompactionKernel.swift` | MIGRATE | Kernels/Clustering/ | Unique |
| `Kernels/HNSW/HNSWDistanceMatrixKernel.swift` | MIGRATE | Kernels/HNSW/ | Unique |
| `GPU/GPUBufferPool.swift` | DELETE | N/A | Use BufferPool |
| `GPU/GPUDecisionEngine.swift` | DELETE | N/A | Use HybridExecutionStrategy |
| `GPU/GPUErrorHandling.swift` | DELETE | N/A | Use VectorError |
| `GPU/MPSDistanceEngine.swift` | MIGRATE | MPS/ | MPS support |
| `GPU/MPSSelectionEngine.swift` | MIGRATE | MPS/ | MPS support |
| `GPU/MPSVectorProcessor.swift` | MIGRATE | MPS/ | MPS support |
| `GPU/MPSBufferManager.swift` | MIGRATE | MPS/ | MPS support |
| `GPU/MPSError.swift` | MIGRATE | MPS/ | MPS support |
| `GPU/IVFGPUProcessorEnhanced.swift` | MIGRATE | Kernels/IVF/ | Unique |
| `GPU/MemoryProfiler.swift` | EVALUATE | Optimization/ | May merge |
| `GPU/MemoryPressureHandler.swift` | EVALUATE | Optimization/ | May merge |
| `GPU/AdaptivePoolingStrategy.swift` | DELETE | N/A | Use SmartBufferPool |
| `Indexes/HNSWIndexAccelerated.swift` | MIGRATE | Indexes/ | Core wrapper |
| `IVFIndexAccelerated.swift` | MIGRATE | Indexes/ | Core wrapper |
| `FlatIndexAccelerated.swift` | MIGRATE | Indexes/ | Core wrapper |
| `Advanced/AsyncOperationManager.swift` | MIGRATE | Optimization/ | Useful |
| `Advanced/BatchQueryProcessor.swift` | MIGRATE | Optimization/ | Useful |
| `Advanced/QueryOptimizer.swift` | MIGRATE | Optimization/ | Useful |
| `Advanced/PerformanceMetrics.swift` | MERGE | Configuration/ | Merge with VA's |
| `Memory/MemoryLayoutOptimizer.swift` | MIGRATE | Optimization/ | Useful |
| `Memory/MemoryAccessAnalyzer.swift` | MIGRATE | Optimization/ | Useful |
| `Memory/BatchPackingOptimizer.swift` | MIGRATE | Optimization/ | Useful |
| `Performance/ZeroCopyTransfer.swift` | MIGRATE | Optimization/ | Useful |
| `Performance/BandwidthOptimizer.swift` | MIGRATE | Optimization/ | Useful |
| `Performance/SIMDSerializer.swift` | MIGRATE | Optimization/ | Useful |
| `Core/AcceleratedOperations.swift` | EVALUATE | N/A | May be redundant |
| `DataStructures/MinHeap.swift` | DELETE | N/A | Use stdlib/VA's |
| `Config.swift` | MERGE | Configuration/ | Merge with VA's |
| `DistanceUtils.swift` | DELETE | N/A | Use VectorCore |
| `Persistence.swift` | DEFER | N/A | Separate package |
| `IndexMapping.swift` | MIGRATE | Core/ | May be useful |
| `Network/*` | DEFER | N/A | VectorDistributed |
| `Distributed/*` | DEFER | N/A | VectorDistributed |
| `Partitioning/*` | DEFER | N/A | VectorDistributed |
| `Persistence/*` | DEFER | N/A | VectorDistributed |
| `.deprecated/*` | DELETE | N/A | Already deprecated |

---

## Appendix B: Metal Shader Migration

### Shaders to Migrate

```
VIA Shaders → VA Location
──────────────────────────────────
Sources/VectorIndexAccelerated/Metal/Kernels/IndexBuilding/HNSW/*.metal
  → Sources/VectorIndexAcceleration/Shaders/HNSW/

Sources/VectorIndexAccelerated/Metal/Kernels/IndexBuilding/IVF/*.metal
  → Sources/VectorIndexAcceleration/Shaders/IVF/

Sources/VectorIndexAccelerated/Metal/Kernels/Clustering/*.metal
  → Sources/VectorIndexAcceleration/Shaders/Clustering/

Sources/VectorIndexAccelerated/Metal/Kernels/Distance/*.metal
  → DELETE (use VA's)

Sources/VectorIndexAccelerated/Metal/Kernels/Selection/*.metal
  → DELETE (use VA's)

Sources/VectorIndexAccelerated/Metal/Kernels/Sorting/*.metal
  → DELETE (use VA's)
```

### Shader Modifications

1. Update include paths to reference VA's `Metal4Common.h`
2. Align naming conventions with VA's kernel naming
3. Add Metal 4 availability guards where needed

---

## Appendix C: Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Preparation | 1-2 days | 2 days |
| Phase 2: HNSW Kernels | 2-3 days | 5 days |
| Phase 3: IVF & Clustering | 1-2 days | 7 days |
| Phase 4: Index Wrappers | 2-3 days | 10 days |
| Phase 5: MPS Integration | 1 day | 11 days |
| Phase 6: Optimization | 2 days | 13 days |
| Phase 7: Testing & Docs | 2-3 days | 16 days |
| Phase 8: Release | 1 day | 17 days |

**Total Estimated Duration**: 2.5-3 weeks

---

## Appendix D: Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code reduction | 70% less LOC | Compare VIA + VA 0.2.0 vs VA 0.3.0 |
| Test coverage | ≥90% | Code coverage tools |
| Performance parity | ±5% of VIA | Benchmark suite |
| Build time | ≤VA 0.2.0 + 20% | CI metrics |
| Package size | ≤VA 0.2.0 + 30% | Built artifact size |
| API surface | Document 100% | DocC coverage |

---

*End of Migration Plan*
