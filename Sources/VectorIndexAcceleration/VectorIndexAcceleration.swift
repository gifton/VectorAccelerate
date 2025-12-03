//
//  VectorIndexAcceleration.swift
//  VectorAccelerate
//
//  GPU-first vector index with direct data ownership.
//
//  This module provides AcceleratedVectorIndex, a clean GPU-first API for
//  similarity search that owns vector data directly on the GPU.
//
//  ## Key Features
//  - Direct GPU data ownership (no CPU/GPU duplication)
//  - Opaque handle-based vector identification
//  - Generation-based stale handle detection
//  - Support for Flat and IVF index types
//  - Native GPU distances (L2² for euclidean)
//
//  ## Quick Start
//  ```swift
//  // Create index
//  let index = try await AcceleratedVectorIndex(
//      configuration: .flat(dimension: 768, capacity: 10_000)
//  )
//
//  // Insert vectors
//  let handle = try await index.insert(embedding)
//
//  // Search
//  let results = try await index.search(query: queryVector, k: 10)
//  ```
//

// MARK: - Module Exports

// Re-export dependencies for convenience
@_exported import VectorAccelerate
@_exported import VectorIndex
@_exported import VectorCore

// MARK: - Public API
//
// Main Types:
// - AcceleratedVectorIndex: GPU-first vector index (actor)
// - VectorHandle: Opaque handle to a vector
// - SearchResult: Search result with handle and distance
// - IndexConfiguration: Index configuration (.flat, .ivf)
// - GPUIndexStats: Index statistics and health metrics
// - VectorMetadata: Per-vector key-value metadata

// MARK: - Index Types
//
// Flat Index:
//   - Exhaustive search using FusedL2TopKKernel
//   - Best for < 10K vectors
//   - O(N) search complexity
//
// IVF Index:
//   - Inverted file index with K-Means clustering
//   - Best for > 10K vectors
//   - O(nprobe * N/nlist) search complexity
//   - Requires training (automatic or manual)

// MARK: - Internal Components
//
// Kernels (Clustering):
// - KMeansPipeline: K-Means clustering for IVF training
// - KMeansAssignKernel, KMeansUpdateKernel, KMeansConvergenceKernel
//
// Kernels (IVF):
// - IVFSearchPipeline: Complete IVF search pipeline
// - IVFCoarseQuantizerKernel: Find nearest centroids
// - IVFKernels: Supporting IVF types and configurations
//
// Internal:
// - GPUVectorStorage: GPU buffer management
// - HandleAllocator: Handle lifecycle with generation tracking
// - DeletionMask: Bitset-based deletion tracking
// - MetadataStore: Sparse CPU-side metadata storage
// - IVFStructure: IVF-specific cluster management

// MARK: - Version Information

/// VectorIndexAcceleration module version
public enum VectorIndexAccelerationVersion {
    /// Current version string
    public static let current = "0.2.0"

    /// Minimum required VectorAccelerate version
    public static let minimumVAVersion = "0.2.0"

    /// Minimum required VectorIndex version
    public static let minimumVIVersion = "0.1.3"

    /// Minimum required VectorCore version
    public static let minimumVCVersion = "0.1.6"
}
