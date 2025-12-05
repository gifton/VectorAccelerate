//
//  GPUIndexStats.swift
//  VectorAccelerate
//
//  Statistics and introspection data for AcceleratedVectorIndex.
//

import Foundation
import VectorCore

// MARK: - GPU Index Statistics

/// Statistics about an AcceleratedVectorIndex.
///
/// Provides information about the current state of the index including
/// vector count, memory usage, and configuration details.
///
/// ## Usage
/// ```swift
/// let stats = index.statistics()
/// print("Vectors: \(stats.vectorCount)")
/// print("GPU Memory: \(stats.gpuMemoryBytes / 1024 / 1024) MB")
/// print("Fragmentation: \(stats.fragmentationRatio * 100)%")
/// ```
public struct GPUIndexStats: Sendable, Equatable {

    // MARK: - Core Counts

    /// Total number of active (non-deleted) vectors in the index.
    public let vectorCount: Int

    /// Total slots allocated in the GPU buffer.
    /// May be greater than vectorCount if vectors have been deleted.
    public let allocatedSlots: Int

    /// Number of deleted slots awaiting compaction.
    public let deletedSlots: Int

    // MARK: - Configuration

    /// Vector dimension.
    public let dimension: Int

    /// Distance metric used by this index.
    public let metric: SupportedDistanceMetric

    /// Index type (flat or IVF).
    public let indexType: IndexConfiguration.IndexType

    /// Current capacity (maximum vectors before growth).
    public let capacity: Int

    // MARK: - Memory Usage

    /// Total GPU memory used by vector data in bytes.
    public let gpuVectorMemoryBytes: Int

    /// Total GPU memory used by index structures (IVF lists, etc.) in bytes.
    public let gpuIndexStructureBytes: Int

    /// Total GPU memory used in bytes.
    public var gpuMemoryBytes: Int {
        gpuVectorMemoryBytes + gpuIndexStructureBytes
    }

    /// CPU memory used for metadata storage in bytes (estimated).
    public let cpuMetadataMemoryBytes: Int

    // MARK: - Health Metrics

    /// Ratio of deleted slots to total allocated slots.
    /// Higher values indicate more fragmentation; consider calling `compact()`.
    public var fragmentationRatio: Float {
        guard allocatedSlots > 0 else { return 0 }
        return Float(deletedSlots) / Float(allocatedSlots)
    }

    /// Buffer utilization ratio (active vectors / capacity).
    public var utilizationRatio: Float {
        guard capacity > 0 else { return 0 }
        return Float(vectorCount) / Float(capacity)
    }

    /// Whether compaction is recommended (fragmentation > 25%).
    public var shouldCompact: Bool {
        fragmentationRatio > 0.25
    }

    // MARK: - IVF-Specific Stats (Optional)

    /// IVF-specific statistics (nil for flat indexes).
    public let ivfStats: IVFStats?

    /// IVF-specific statistics.
    public struct IVFStats: Sendable, Equatable {
        /// Number of clusters (nlist).
        public let numClusters: Int

        /// Number of clusters probed during search (nprobe).
        public let nprobe: Int

        /// Whether the index has been trained.
        public let isTrained: Bool

        /// Average vectors per cluster (for load balancing analysis).
        public let averageVectorsPerCluster: Float

        /// Standard deviation of vectors per cluster.
        public let clusterSizeStdDev: Float

        public init(
            numClusters: Int,
            nprobe: Int,
            isTrained: Bool,
            averageVectorsPerCluster: Float,
            clusterSizeStdDev: Float
        ) {
            self.numClusters = numClusters
            self.nprobe = nprobe
            self.isTrained = isTrained
            self.averageVectorsPerCluster = averageVectorsPerCluster
            self.clusterSizeStdDev = clusterSizeStdDev
        }
    }

    // MARK: - Initialization

    /// Create GPU index statistics.
    public init(
        vectorCount: Int,
        allocatedSlots: Int,
        deletedSlots: Int,
        dimension: Int,
        metric: SupportedDistanceMetric,
        indexType: IndexConfiguration.IndexType,
        capacity: Int,
        gpuVectorMemoryBytes: Int,
        gpuIndexStructureBytes: Int = 0,
        cpuMetadataMemoryBytes: Int = 0,
        ivfStats: IVFStats? = nil
    ) {
        self.vectorCount = vectorCount
        self.allocatedSlots = allocatedSlots
        self.deletedSlots = deletedSlots
        self.dimension = dimension
        self.metric = metric
        self.indexType = indexType
        self.capacity = capacity
        self.gpuVectorMemoryBytes = gpuVectorMemoryBytes
        self.gpuIndexStructureBytes = gpuIndexStructureBytes
        self.cpuMetadataMemoryBytes = cpuMetadataMemoryBytes
        self.ivfStats = ivfStats
    }
}

// MARK: - CustomStringConvertible

extension GPUIndexStats: CustomStringConvertible {
    public var description: String {
        var parts: [String] = [
            "GPUIndexStats(",
            "  vectors: \(vectorCount)/\(capacity)",
            "  dimension: \(dimension)",
            "  metric: \(metric)",
            "  gpuMemory: \(formatBytes(gpuMemoryBytes))",
            "  fragmentation: \(String(format: "%.1f%%", fragmentationRatio * 100))"
        ]

        if let ivf = ivfStats {
            parts.append("  ivf: \(ivf.numClusters) clusters, nprobe=\(ivf.nprobe), trained=\(ivf.isTrained)")
        }

        parts.append(")")
        return parts.joined(separator: "\n")
    }

    private func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
        } else if bytes >= 1_048_576 {
            return String(format: "%.2f MB", Double(bytes) / 1_048_576)
        } else if bytes >= 1024 {
            return String(format: "%.2f KB", Double(bytes) / 1024)
        } else {
            return "\(bytes) B"
        }
    }
}
