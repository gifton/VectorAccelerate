//
//  WALOperations.swift
//  VectorAccelerate
//
//  WAL operation types for AcceleratedVectorIndex crash recovery.
//

import Foundation

// MARK: - Operation Type Enum

/// WAL operation types for AcceleratedVectorIndex.
///
/// These types identify the kind of operation stored in each WAL entry,
/// enabling proper deserialization and replay during crash recovery.
public enum IndexWALOperationType: String, Codable, Sendable {
    /// Single vector insert
    case insert
    /// Batch vector insert
    case batchInsert
    /// Single vector remove
    case remove
    /// Batch vector remove
    case batchRemove
    /// Compaction started
    case compact
    /// Compaction completed
    case compactComplete
    /// Training started (IVF only)
    case trainStart
    /// Training completed (IVF only)
    case trainComplete
    /// Checkpoint marker
    case checkpoint
}

// MARK: - Insert Operations

/// Single vector insert operation for WAL replay.
///
/// Captures all data needed to replay an insert operation:
/// - The stable ID assigned to the vector
/// - The vector data itself
/// - Any associated metadata
///
/// ## Replay Behavior
/// During recovery, the index will:
/// 1. Allocate a slot using the stored stableID
/// 2. Write the vector to GPU storage
/// 3. Store associated metadata
public struct InsertOperation: Codable, Sendable {
    /// Stable ID of the inserted vector
    public let stableID: UInt32
    /// The vector data
    public let vector: [Float]
    /// Optional metadata
    public let metadata: [String: String]?
    /// When the operation occurred
    public let timestamp: Date

    public init(
        stableID: UInt32,
        vector: [Float],
        metadata: [String: String]?,
        timestamp: Date = Date()
    ) {
        self.stableID = stableID
        self.vector = vector
        self.metadata = metadata
        self.timestamp = timestamp
    }
}

/// Batch insert operation for WAL replay.
///
/// Stores multiple vectors in a single WAL entry for efficiency.
/// The vectors are stored as a flattened array to reduce JSON overhead.
///
/// ## Data Layout
/// `vectors` contains `vectorCount * dimension` floats, where each
/// consecutive `dimension` floats represent one vector.
///
/// ## Replay Behavior
/// During recovery, vectors are reconstructed from the flattened array
/// and inserted in order with their corresponding stable IDs.
public struct BatchInsertOperation: Codable, Sendable {
    /// Stable IDs assigned to the batch (in order)
    public let stableIDs: [UInt32]
    /// Number of vectors in batch
    public let vectorCount: Int
    /// Vector dimension
    public let dimension: Int
    /// Flattened vector data (vectorCount * dimension floats)
    public let vectors: [Float]
    /// When the operation occurred
    public let timestamp: Date

    public init(
        stableIDs: [UInt32],
        vectorCount: Int,
        dimension: Int,
        vectors: [Float],
        timestamp: Date = Date()
    ) {
        self.stableIDs = stableIDs
        self.vectorCount = vectorCount
        self.dimension = dimension
        self.vectors = vectors
        self.timestamp = timestamp
    }

    /// Reconstruct individual vectors from the flattened array.
    ///
    /// - Returns: Array of individual vectors
    public func unflattenVectors() -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(vectorCount)
        for i in 0..<vectorCount {
            let start = i * dimension
            let end = start + dimension
            result.append(Array(vectors[start..<end]))
        }
        return result
    }
}

// MARK: - Remove Operations

/// Single remove operation for WAL replay.
///
/// Captures the stable ID of a removed vector for replay.
/// During recovery, the vector with this stable ID will be marked as deleted.
public struct RemoveOperation: Codable, Sendable {
    /// Stable ID of the removed vector
    public let stableID: UInt32
    /// When the operation occurred
    public let timestamp: Date

    public init(stableID: UInt32, timestamp: Date = Date()) {
        self.stableID = stableID
        self.timestamp = timestamp
    }
}

/// Batch remove operation for WAL replay.
///
/// Stores multiple stable IDs to remove in a single WAL entry.
public struct BatchRemoveOperation: Codable, Sendable {
    /// Stable IDs of the removed vectors
    public let stableIDs: [UInt32]
    /// When the operation occurred
    public let timestamp: Date

    public init(stableIDs: [UInt32], timestamp: Date = Date()) {
        self.stableIDs = stableIDs
        self.timestamp = timestamp
    }
}

// MARK: - Compact Operations

/// Compact start operation - captures state for recovery.
///
/// Logged BEFORE compaction begins. If recovery finds a CompactStartOperation
/// without a matching CompactCompleteOperation, the compaction was interrupted
/// and should be re-run.
public struct CompactStartOperation: Codable, Sendable {
    /// Slot indices that will be removed during compaction
    public let deletedSlots: [Int]
    /// Number of vectors before compaction
    public let vectorCountBefore: Int
    /// When the operation started
    public let timestamp: Date

    public init(
        deletedSlots: [Int],
        vectorCountBefore: Int,
        timestamp: Date = Date()
    ) {
        self.deletedSlots = deletedSlots
        self.vectorCountBefore = vectorCountBefore
        self.timestamp = timestamp
    }
}

/// Compact complete operation - marks successful completion.
///
/// Contains the slot mapping so recovery can verify or replay the compaction.
public struct CompactCompleteOperation: Codable, Sendable {
    /// Mapping from old slot indices to new slot indices
    public let slotMapping: [Int: Int]
    /// Number of vectors after compaction
    public let vectorCountAfter: Int
    /// When the operation completed
    public let timestamp: Date

    public init(
        slotMapping: [Int: Int],
        vectorCountAfter: Int,
        timestamp: Date = Date()
    ) {
        self.slotMapping = slotMapping
        self.vectorCountAfter = vectorCountAfter
        self.timestamp = timestamp
    }
}

// MARK: - Train Operations

/// Training start operation - marks beginning of IVF training.
///
/// Logged BEFORE training begins. If recovery finds TrainStartOperation
/// without TrainCompleteOperation, training was interrupted and the
/// index should be retrained.
public struct TrainStartOperation: Codable, Sendable {
    /// Number of vectors used for training
    public let vectorCount: Int
    /// Target number of clusters
    public let numClusters: Int
    /// Vector dimension
    public let dimension: Int
    /// When training started
    public let timestamp: Date

    public init(
        vectorCount: Int,
        numClusters: Int,
        dimension: Int,
        timestamp: Date = Date()
    ) {
        self.vectorCount = vectorCount
        self.numClusters = numClusters
        self.dimension = dimension
        self.timestamp = timestamp
    }
}

/// Training complete operation - contains trained centroids.
///
/// This captures the full training result so it can be restored on recovery
/// without re-running the expensive K-Means clustering.
public struct TrainCompleteOperation: Codable, Sendable {
    /// Trained centroids [numClusters × dimension], flattened
    public let centroids: [Float]
    /// Number of clusters
    public let numClusters: Int
    /// Vector dimension
    public let dimension: Int
    /// When training completed
    public let timestamp: Date

    public init(
        centroids: [Float],
        numClusters: Int,
        dimension: Int,
        timestamp: Date = Date()
    ) {
        self.centroids = centroids
        self.numClusters = numClusters
        self.dimension = dimension
        self.timestamp = timestamp
    }

    /// Reconstruct centroids as 2D array.
    public func unflattenCentroids() -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(numClusters)
        for i in 0..<numClusters {
            let start = i * dimension
            let end = start + dimension
            result.append(Array(centroids[start..<end]))
        }
        return result
    }
}
