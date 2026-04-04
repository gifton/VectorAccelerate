# VectorIndexAccelerated → VectorAccelerate Migration Plan

## Executive Summary

This document outlines features from the deprecated **VectorIndexAccelerated** (VIA) package that should be migrated to **VectorAccelerate** (VA) to create a unified, production-ready GPU-accelerated vector search library.

### Current Architecture

```
VectorCore (../../VectorCore)     - Base protocols, types, distance metrics
     ↓
VectorIndex (../../VectorIndex)   - CPU-based indices (FlatIndex, HNSWIndex)
     ↓
VectorAccelerate (this repo)      - Metal 4 GPU kernels, AcceleratedVectorIndex (IVF only)
     ↓
VectorIndexAccelerated (../VectorIndexAccelerated) - [DEPRECATED] GPU wrappers for VectorIndex
```

### Key Finding: HNSW Gap

**VectorAccelerate has NO HNSW support.** The only approximate nearest neighbor (ANN) algorithm currently available is IVF (Inverted File Index).

| Package | HNSW Status |
|---------|-------------|
| `VectorIndex` | CPU-based `HNSWIndex` implementation |
| `VectorIndexAccelerated` | GPU-accelerated wrapper (`HNSWIndexAccelerated`) |
| `VectorAccelerate` | **NONE** - only IVF support |

This is a **critical gap**. HNSW is the industry-standard algorithm for high-recall approximate nearest neighbor search.

---

## Migration Tiers Overview

| Tier | Priority | Items | Effort | Timeline |
|------|----------|-------|--------|----------|
| **Tier 1** | Critical | 3 | ~3 weeks | Weeks 1-3 |
| **Tier 2** | High | 5 | ~2 weeks | Weeks 4-5 |
| **Tier 3** | Medium-Low | 5 | ~2 weeks | As needed |

---

## Tier 1: Critical Priority - Production Essentials

### 1.1 HNSW GPU Support [CRITICAL]

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Indexes/`, `VIA/Metal/Kernels/IndexBuilding/HNSW/` |
| **Lines** | ~2,500 |
| **Effort** | High (10-15 days) |
| **Impact** | Critical |

#### Why Critical

HNSW (Hierarchical Navigable Small World) typically outperforms IVF in:
- **Recall accuracy** at equivalent search speed
- **Query latency** for single queries
- **Memory efficiency** for high-dimensional vectors

Without HNSW, VectorAccelerate cannot compete with production vector databases like Pinecone, Weaviate, or Qdrant.

#### Current State in VIA

```
Sources/VectorIndexAccelerated/
├── Indexes/
│   └── HNSWIndexAccelerated.swift           # GPU wrapper (735 lines)
│
├── Kernels/HNSW/
│   └── HNSWDistanceMatrixKernel.swift       # Swift kernel wrapper
│
└── Metal/Kernels/
    ├── HNSWDistanceCacheKernel.swift
    ├── HNSWSearchKernel.swift
    ├── HNSWEdgeInsertionKernel.swift
    ├── HNSWEdgePruningKernel.swift
    ├── HNSWVisitedSetKernel.swift
    ├── HNSWLevelAssignmentKernel.swift
    │
    └── IndexBuilding/HNSW/
        ├── HNSWDistanceMatrix.metal          # Distance computation with caching
        ├── HNSWSearchLayer.metal             # Layer-by-layer graph search
        ├── HNSWEdgePruning.metal             # Heuristic neighbor selection
        ├── HNSWHeuristicSelection.metal      # Advanced pruning heuristics
        ├── HNSWBatchInsertEdges.metal        # Batch edge insertion
        ├── HNSWIncrementalDistances.metal    # Incremental distance updates
        ├── HNSWNodeLevelAssignment.metal     # Exponential level distribution
        ├── HNSWVisitedSetManagement.metal    # Visited set for search
        ├── HNSWDistanceCacheManagement.metal # Cache management
        └── HNSWBitonicSort.metal             # Sorting for neighbor selection
```

#### Migration Approach

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Port 10 Metal shaders to MSL 4.0 | 3-4 days |
| 2 | Port 6 Swift kernel wrappers to `Metal4Context` | 2-3 days |
| 3 | Create `HNSWStructure` (similar to `IVFStructure`) | 2-3 days |
| 4 | Integrate into `AcceleratedVectorIndex` | 1-2 days |
| 5 | Testing & benchmarking | 2-3 days |

#### Target API

```swift
// New IndexType case
enum IndexType {
    case flat
    case ivf(nlist: Int, nprobe: Int, minTrainingVectors: Int?)
    case hnsw(m: Int, efConstruction: Int, efSearch: Int)  // NEW
}

// Usage
let index = try await AcceleratedVectorIndex(
    configuration: .hnsw(
        dimension: 768,
        m: 16,               // Max edges per node (higher = better recall, more memory)
        efConstruction: 200, // Construction quality (higher = better graph)
        efSearch: 64         // Search quality (higher = better recall, slower)
    )
)

// Search - same API as IVF
let results = try await index.search(query: embedding, k: 10)
```

#### Files to Create

```
Sources/VectorAccelerate/
├── Index/
│   ├── HNSWStructure.swift              # Graph storage, adjacency lists
│   ├── HNSWSearchPipeline.swift         # Multi-layer search coordination
│   └── HNSWConfiguration.swift          # HNSW-specific config
│
├── Kernels/Metal4/HNSW/
│   ├── HNSWDistanceMatrixKernel.swift
│   ├── HNSWSearchKernel.swift
│   ├── HNSWEdgeInsertionKernel.swift
│   ├── HNSWEdgePruningKernel.swift
│   ├── HNSWLevelAssignmentKernel.swift
│   └── HNSWVisitedSetKernel.swift
│
└── Metal/Shaders/HNSW/
    ├── HNSWDistanceMatrix.metal
    ├── HNSWSearchLayer.metal
    ├── HNSWEdgePruning.metal
    ├── HNSWHeuristicSelection.metal
    ├── HNSWBatchInsertEdges.metal
    ├── HNSWIncrementalDistances.metal
    ├── HNSWNodeLevelAssignment.metal
    ├── HNSWVisitedSetManagement.metal
    ├── HNSWDistanceCacheManagement.metal
    └── HNSWBitonicSort.metal
```

---

### 1.2 GPUDecisionEngine

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/GPU/GPUDecisionEngine.swift` |
| **Lines** | 290 |
| **Effort** | Low (4-8 hours) |
| **Impact** | High |

#### Why Important

VA currently uses hardcoded thresholds for GPU/CPU routing. VIA's `GPUDecisionEngine` provides:

- **Adaptive thresholds** based on runtime performance
- **Per-operation tracking** (search, training, selection, etc.)
- **Memory estimation** before execution
- **Performance history** with automatic tuning

#### Key Components

```swift
public struct GPUActivationThresholds: Sendable {
    public let minVectorsForGPU: Int          // Default: 1000
    public let minCandidatesForGPU: Int       // Default: 500
    public let minKForGPU: Int                // Default: 10
    public let maxKForGPU: Int                // Default: 1000
    public let minOperationsForGPU: Int       // Default: 50,000
    public let maxGPUMemoryMB: Int            // Default: 1024
    public var gpuPerformanceRatio: Float     // Updated at runtime
}

public actor GPUDecisionEngine {
    /// Determines whether to use GPU for a given operation
    func shouldUseGPU(
        operation: GPUOperation,
        vectorCount: Int,
        candidateCount: Int,
        k: Int,
        queryCount: Int,
        dimension: Int
    ) async -> Bool

    /// Records performance and updates adaptive thresholds
    func recordPerformance(operation:, cpuTime:, gpuTime:) async

    /// Get current performance statistics
    func getPerformanceStats() async -> GPUPerformanceStats
}
```

#### Migration Approach

1. Copy `GPUDecisionEngine.swift` to `Sources/VectorAccelerate/Core/`
2. Update to use `Metal4Context` for device queries
3. Integrate into `AcceleratedVectorIndex` for routing decisions
4. Add telemetry hooks to kernel execution paths

---

### 1.3 Write-Ahead Log (WAL)

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Persistence/WriteAheadLog.swift` |
| **Lines** | 507 |
| **Effort** | Medium (1-2 days) |
| **Impact** | High |

#### Why Important

VA has **no persistence/crash-recovery story**. For production use, indices need:

- **Durability** - Survive process crashes
- **Point-in-time recovery** - Replay from last checkpoint
- **Incremental updates** - Avoid full index rebuilds

#### Key Features

```swift
public actor WriteAheadLog {
    // Configuration
    struct Configuration {
        var maxSegmentSize: Int = 10_000_000  // 10MB per segment
        var maxSegments: Int = 10              // Rolling window
        var syncMode: SyncMode = .periodic     // immediate | periodic | batch
        var checksumEnabled: Bool = true
        var compressionEnabled: Bool = false
    }

    enum SyncMode {
        case immediate  // Sync after every write (slowest, safest)
        case periodic   // Sync every 1 second (balanced)
        case batch      // Sync after 100 entries (fastest, less safe)
    }

    // Core operations
    func append<T: Codable>(_ operation: T, type: String) async throws -> UInt64
    func flush() async throws
    func checkpoint() async throws -> UInt64
    func replay(from: UInt64) async throws -> [WALEntry]
    func truncate(after: UInt64) async throws
    func compact() async throws
    func getStatistics() async -> WALStatistics
}
```

#### Migration Approach

1. Copy `WriteAheadLog.swift` to `Sources/VectorAccelerate/Persistence/`
2. Define operation types for VA (insert, remove, compact, etc.)
3. Integrate with `AcceleratedVectorIndex`:
   - Log operations before GPU execution
   - Checkpoint after successful batches
   - Replay on index load
4. Add `PersistenceConfiguration` to `IndexConfiguration`

---

## Tier 2: High Priority - Performance & Reliability

### 2.1 GPUHealthMonitor

| Attribute | Value |
|-----------|-------|
| **Source** | Embedded in `VIA/GPU/IVFGPUProcessorEnhanced.swift` |
| **Lines** | ~150 |
| **Effort** | Low (4-6 hours) |
| **Impact** | Medium |

#### Purpose

Track GPU health and trigger automatic fallbacks:
- Failure count tracking per operation type
- Recovery attempt monitoring
- Performance degradation detection
- Automatic CPU fallback triggers

#### Migration Approach

1. Extract health monitoring logic into standalone `GPUHealthMonitor` actor
2. Integrate with `Metal4Context` for error tracking
3. Add callbacks for fallback triggers

---

### 2.2 Advanced IVF List Compaction

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Metal/Kernels/IndexBuilding/IVF/` |
| **Lines** | ~800 |
| **Effort** | Medium (1-2 days) |
| **Impact** | Medium |

#### Files

- `IVFListCompaction.metal` - Base implementation
- `IVFListCompactionOptimized.metal` - Block-size variants:
  - 64 threads (small lists)
  - 128 threads
  - 256 threads
  - 512 threads
  - 1024 threads (large lists)

#### Purpose

After deletions, IVF inverted lists become sparse. Compaction kernels efficiently remove gaps without full rebuild. Optimized variants select block size based on list length.

#### Current VA Status

Basic compaction exists but lacks block-size optimization.

---

### 2.3 Two-Pass Candidate Gathering

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Metal/Kernels/IndexBuilding/IVF/IVFCandidateGatheringPass2.metal` |
| **Lines** | ~200 |
| **Effort** | Low (4-8 hours) |
| **Impact** | Medium |

#### Purpose

Memory-efficient candidate collection for large nprobe values:
- **Pass 1:** Count candidates per inverted list
- **Pass 2:** Gather with exact buffer sizing (no over-allocation)

#### Current VA Status

Single-pass with pre-allocated buffers. Wastes memory for variable-size lists.

---

### 2.4 Streaming Pipeline

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/.deprecated/Streaming/` |
| **Lines** | ~1,200 |
| **Effort** | High (3-5 days) |
| **Impact** | Medium |

#### Files

- `StreamingPipeline.swift` - Main coordinator
- `StreamingEngine.swift` - Core engine
- `DMATransferEngine.swift` - Overlapped CPU↔GPU transfers
- `PrefetchEngine.swift` - Memory latency hiding
- `TripleBufferCoordinator.swift` - Triple buffering for pipelining

#### Purpose

Process datasets larger than GPU memory:
- Chunk data into GPU-sized batches
- Overlap compute with data transfer (hide latency)
- Triple buffering for continuous processing

#### Current VA Status

`StreamingTopKKernel` handles some cases, but lacks pipelined transfers and prefetching.

---

### 2.5 Large Batch Processor

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/.deprecated/LargeBatch/` |
| **Lines** | ~800 |
| **Effort** | Medium (2-3 days) |
| **Impact** | Medium |

#### Files

- `LargeBatchProcessor.swift` - Chunked processing coordinator
- `LargeBatchMemoryStrategy.swift` - Memory strategy selection
- `BufferPool.swift` - Buffer reuse (note: VA has its own `BufferPool`)
- `DoubleBufferManager.swift` - Double buffering coordination

#### Purpose

Handle very large batch operations (100K+ vectors):
- Automatic chunking with configurable overlap
- Concurrent chunk processing (up to 3 in flight)
- Progressive result aggregation
- Memory strategy selection (standard vs aggressive)

---

## Tier 3: Medium-Low Priority - Specialized Features

### 3.1 Memory Optimization Utilities

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Memory/` |
| **Lines** | ~600 |
| **Effort** | Medium (2-3 days) |
| **Impact** | Low |

#### Files

- `MemoryLayoutOptimizer.swift` - Align data for coalesced GPU access
- `MemoryAccessAnalyzer.swift` - Identify optimization opportunities
- `BatchPackingOptimizer.swift` - Minimize wasted compute cycles

#### Use Case

Advanced performance tuning for memory-bound workloads. Useful for users pushing maximum throughput.

---

### 3.2 Delta Encoding

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Persistence/DeltaManager.swift` |
| **Lines** | ~300 |
| **Effort** | Low (1 day) |
| **Impact** | Low |

#### Purpose

Track changes incrementally for:
- Reduced storage overhead
- Faster incremental saves
- Efficient change propagation

#### Use Case

Long-running indices with frequent updates where full saves are expensive.

---

### 3.3 Advanced Persistence

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Persistence/AdvancedPersistence.swift` |
| **Lines** | ~400 |
| **Effort** | Medium (1-2 days) |
| **Impact** | Low |

#### Features

- Compression support (multiple algorithms)
- Incremental persistence
- Optimized save/load operations
- Partial index loading

---

### 3.4 Distributed Infrastructure

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Distributed/`, `VIA/Network/` |
| **Lines** | ~3,000 |
| **Effort** | Very High (2-4 weeks) |
| **Impact** | Specialized |

#### Components

- `DistributedCoordinator.swift` - Multi-node coordination
- `NetworkManager.swift` - Connection management with pooling
- `PartitionManager.swift` - Data partitioning and rebalancing
- `FaultTolerance.swift` - Replica failover
- Raft-like consensus mechanism

#### Use Case

Building a distributed vector database service. Out of scope for single-node library unless specifically requested.

---

### 3.5 mTLS/Security

| Attribute | Value |
|-----------|-------|
| **Source** | `VIA/Network/` |
| **Lines** | ~1,500 |
| **Effort** | High (1-2 weeks) |
| **Impact** | Specialized |

#### Files

- `MutualTLSManager.swift` - mTLS setup
- `ProductionTLSManager.swift` - Production TLS config
- `CertificateParser.swift` - X.509 parsing
- `OCSPValidator.swift` - Online revocation checking
- `CRLValidator.swift` - CRL-based revocation
- `MessageEncryption.swift` - End-to-end encryption

#### Use Case

Secure distributed deployments. Only needed if Tier 3.4 is implemented.

---

## Implementation Roadmap

```
Week 1-2: Foundation
├── [ ] GPUDecisionEngine migration
├── [ ] Write-Ahead Log migration
└── [ ] GPUHealthMonitor extraction

Week 3-4: HNSW Core
├── [ ] Port 10 HNSW Metal shaders to MSL 4.0
├── [ ] Port 6 Swift kernel wrappers to Metal4Context
└── [ ] Create HNSWStructure with graph storage

Week 5: HNSW Integration
├── [ ] Integrate into AcceleratedVectorIndex
├── [ ] Add IndexType.hnsw configuration
└── [ ] Basic testing and validation

Week 6: HNSW Optimization
├── [ ] Recall/performance benchmarking
├── [ ] Tune GPU utilization
└── [ ] Documentation and examples

Week 7-8: Tier 2 (as needed)
├── [ ] Advanced list compaction
├── [ ] Two-pass candidate gathering
└── [ ] Streaming pipeline (if large datasets needed)
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| **HNSW Recall@10** | ≥95% at 1ms query latency |
| **HNSW Build Speed** | ≥10K vectors/second on M1 Pro |
| **GPU Decision Accuracy** | ≥90% optimal routing decisions |
| **WAL Recovery Time** | <1s for 1M vector index |
| **IVF Search Improvement** | ≥20% speedup with optimized compaction |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MSL 4.0 shader compatibility issues | Medium | High | Test on multiple Apple Silicon variants |
| HNSW graph memory pressure | Medium | Medium | Implement graph compression |
| WAL performance overhead | Low | Medium | Use batch sync mode by default |
| Breaking changes to existing API | Low | High | Maintain backward compatibility |

---

## Dependencies

The migration assumes:
- VectorAccelerate 0.4.1+ (Metal 4 only)
- macOS 26.0+ / iOS 26.0+
- Apple Silicon (M1 or later)

No new external dependencies required.

---

## References

- [VectorIndexAccelerated Source](../VectorIndexAccelerated/)
- [VectorIndex Source](../../VectorIndex/) - CPU HNSWIndex implementation
- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2016
- [VectorAccelerate Metal 4 Architecture](./metal4/)
- [Existing IVF Implementation](../Sources/VectorAccelerate/Index/)

---

## Appendix: File Inventory

### Files to Create (Tier 1)

```
Sources/VectorAccelerate/
├── Core/
│   ├── GPUDecisionEngine.swift          # From VIA/GPU/
│   └── GPUHealthMonitor.swift           # Extracted from VIA
│
├── Persistence/
│   ├── WriteAheadLog.swift              # From VIA/Persistence/
│   ├── WALSegment.swift                 # Supporting type
│   └── WALTypes.swift                   # Entry, Statistics, Error types
│
├── Index/
│   ├── HNSWStructure.swift              # Graph storage
│   ├── HNSWSearchPipeline.swift         # Search coordination
│   └── HNSWConfiguration.swift          # Config types
│
├── Kernels/Metal4/HNSW/
│   ├── HNSWDistanceMatrixKernel.swift
│   ├── HNSWSearchKernel.swift
│   ├── HNSWEdgeInsertionKernel.swift
│   ├── HNSWEdgePruningKernel.swift
│   ├── HNSWLevelAssignmentKernel.swift
│   └── HNSWVisitedSetKernel.swift
│
└── Metal/Shaders/HNSW/
    ├── HNSWDistanceMatrix.metal
    ├── HNSWSearchLayer.metal
    ├── HNSWEdgePruning.metal
    ├── HNSWHeuristicSelection.metal
    ├── HNSWBatchInsertEdges.metal
    ├── HNSWIncrementalDistances.metal
    ├── HNSWNodeLevelAssignment.metal
    ├── HNSWVisitedSetManagement.metal
    ├── HNSWDistanceCacheManagement.metal
    └── HNSWBitonicSort.metal
```

### Files to Modify (Tier 1)

```
Sources/VectorAccelerate/
├── Index/
│   ├── IndexConfiguration.swift         # Add IndexType.hnsw
│   └── AcceleratedVectorIndex.swift     # Add HNSW support
│
└── Core/
    └── Metal4Context.swift              # Add decision engine integration
```
