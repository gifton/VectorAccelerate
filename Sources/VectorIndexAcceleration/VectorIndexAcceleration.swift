//
//  VectorIndexAcceleration.swift
//  VectorAccelerate
//
//  GPU-accelerated index operations for VectorIndex types.
//  Provides Metal 4 acceleration for HNSW, IVF, and Flat indices.
//
//  Created as part of VIA → VA migration.
//  Target Version: VA 0.3.0
//

// MARK: - Module Exports

// Re-export dependencies for convenience
@_exported import VectorAccelerate
@_exported import VectorIndex
@_exported import VectorCore

// MARK: - Core
// Core infrastructure for index acceleration

// MARK: - Kernels
// Metal 4 kernels for index operations
// - HNSW: Graph traversal, distance matrix, edge insertion/pruning
// - IVF: Coarse quantization, inverted list search
// - Clustering: KMeans for IVF centroid training

// MARK: - Indexes
// GPU-accelerated index wrappers
// - HNSWIndexAccelerated
// - IVFIndexAccelerated
// - FlatIndexAccelerated

// MARK: - Extensions
// Protocol extensions for AccelerableIndex

// MARK: - Version Information

/// VectorIndexAcceleration module version
public enum VectorIndexAccelerationVersion {
    /// Current version string
    public static let current = "0.1.0-alpha"

    /// Minimum required VectorAccelerate version
    public static let minimumVAVersion = "0.2.0"

    /// Minimum required VectorIndex version
    public static let minimumVIVersion = "0.1.3"

    /// Minimum required VectorCore version
    public static let minimumVCVersion = "0.1.6"
}
