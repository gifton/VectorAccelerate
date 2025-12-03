//
//  IVFStructure.swift
//  VectorIndexAcceleration
//
//  Internal IVF (Inverted File) index structure management.
//
//  Manages:
//  - Centroids (cluster centers)
//  - Inverted lists (vectors assigned to each centroid)
//  - Training state and cluster assignments
//  - GPU structure preparation for search
//

import Foundation
@preconcurrency import Metal
import VectorCore
import VectorAccelerate

// MARK: - IVF Training State

/// State of IVF index training.
enum IVFTrainingState: Sendable {
    /// Not yet trained - collecting vectors for training
    case untrained

    /// Currently training (K-Means running)
    case training

    /// Training complete, index is ready for search
    case trained

    /// Training failed
    case failed(any Error)
}

// MARK: - Inverted List Entry

/// Entry in an inverted list.
struct IVFListEntry: Sendable {
    /// Slot index in the main vector storage
    let slotIndex: UInt32

    /// Handle generation at time of insertion
    let generation: UInt16
}

// MARK: - IVF Structure

/// Internal structure for managing IVF index data.
///
/// Maintains:
/// - Centroids from K-Means training
/// - Inverted lists mapping centroids to assigned vectors
/// - GPU-ready structure for efficient search
///
/// ## Training Flow
/// 1. Vectors are inserted into staging buffer until minTrainingVectors reached
/// 2. K-Means training runs to compute centroids
/// 3. Staged vectors are assigned to clusters
/// 4. New inserts go directly to appropriate cluster
///
/// ## Thread Safety
/// This class is not thread-safe. Access should be synchronized by the owning actor.
final class IVFStructure: @unchecked Sendable {

    // MARK: - Configuration

    /// Number of clusters (nlist)
    let numClusters: Int

    /// Number of clusters to probe during search (nprobe)
    let nprobe: Int

    /// Vector dimension
    let dimension: Int

    /// Minimum vectors before training
    let minTrainingVectors: Int

    // MARK: - State

    /// Current training state
    private(set) var trainingState: IVFTrainingState = .untrained

    /// Centroids [numClusters × dimension]
    private var centroids: [[Float]] = []

    /// Inverted lists - one list per centroid
    private var invertedLists: [[IVFListEntry]] = []

    /// Staging buffer for vectors before training
    private var stagingSlots: [UInt32] = []

    /// GPU-prepared structure (cached after first search)
    private var gpuStructure: IVFGPUIndexStructure?

    /// Flag indicating GPU structure needs rebuild
    private var gpuStructureDirty: Bool = true

    // MARK: - Statistics

    /// Total vectors in all inverted lists
    var totalVectors: Int {
        invertedLists.reduce(0) { $0 + $1.count }
    }

    /// Whether the index is trained and ready for search
    var isTrained: Bool {
        if case .trained = trainingState { return true }
        return false
    }

    /// Number of vectors in staging (pre-training)
    var stagingCount: Int {
        stagingSlots.count
    }

    /// Cluster size distribution
    var clusterSizes: [Int] {
        invertedLists.map { $0.count }
    }

    /// Average vectors per cluster
    var averageClusterSize: Float {
        guard numClusters > 0 else { return 0 }
        return Float(totalVectors) / Float(numClusters)
    }

    /// Standard deviation of cluster sizes
    var clusterSizeStdDev: Float {
        guard numClusters > 0 else { return 0 }
        let avg = averageClusterSize
        let variance = clusterSizes.reduce(0.0) { sum, size in
            let diff = Float(size) - avg
            return sum + diff * diff
        } / Float(numClusters)
        return sqrt(variance)
    }

    // MARK: - Initialization

    /// Create an IVF structure.
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters (nlist)
    ///   - nprobe: Number of clusters to probe during search
    ///   - dimension: Vector dimension
    ///   - minTrainingVectors: Minimum vectors before training (default: 10 * numClusters)
    init(numClusters: Int, nprobe: Int, dimension: Int, minTrainingVectors: Int? = nil) {
        self.numClusters = numClusters
        self.nprobe = nprobe
        self.dimension = dimension
        self.minTrainingVectors = minTrainingVectors ?? max(numClusters * 10, 1000)

        // Pre-allocate inverted lists
        self.invertedLists = Array(repeating: [], count: numClusters)
    }

    // MARK: - Training

    /// Train the IVF index using K-Means.
    ///
    /// - Parameters:
    ///   - vectors: Training vectors
    ///   - context: Metal context for GPU operations
    /// - Throws: `IndexAccelerationError` if training fails
    func train(vectors: [[Float]], context: Metal4Context) async throws {
        guard !vectors.isEmpty else {
            throw IndexAccelerationError.invalidInput(message: "Cannot train with empty vectors")
        }

        guard vectors.count >= numClusters else {
            throw IndexAccelerationError.invalidInput(
                message: "Need at least \(numClusters) vectors for \(numClusters) clusters, got \(vectors.count)"
            )
        }

        trainingState = .training

        do {
            // Configure K-Means
            let kmeansConfig = KMeansConfiguration(
                numClusters: numClusters,
                dimension: dimension,
                maxIterations: 25,
                convergenceThreshold: 0.001,
                metric: .euclidean
            )

            // Run K-Means clustering
            let pipeline = try await KMeansPipeline(context: context, configuration: kmeansConfig)
            let result = try await pipeline.fit(vectors: vectors)

            // Extract centroids
            centroids = result.extractCentroids()

            // Assign staged vectors to clusters
            let assignments = result.extractAssignments()
            for (i, clusterIdx) in assignments.enumerated() {
                if i < stagingSlots.count {
                    let entry = IVFListEntry(slotIndex: stagingSlots[i], generation: 0)
                    invertedLists[clusterIdx].append(entry)
                }
            }

            stagingSlots.removeAll()
            gpuStructureDirty = true
            trainingState = .trained

        } catch {
            trainingState = .failed(error)
            throw error
        }
    }

    /// Check if we have enough vectors to train.
    var canTrain: Bool {
        stagingSlots.count >= minTrainingVectors
    }

    // MARK: - Insert

    /// Add a vector to staging (pre-training).
    ///
    /// - Parameter slotIndex: Slot index in main storage
    func addToStaging(_ slotIndex: UInt32) {
        stagingSlots.append(slotIndex)
        gpuStructureDirty = true
    }

    /// Assign a vector to the nearest cluster.
    ///
    /// - Parameters:
    ///   - vector: Vector data
    ///   - slotIndex: Slot index in main storage
    ///   - generation: Handle generation
    /// - Returns: Cluster index the vector was assigned to
    func assignToCluster(vector: [Float], slotIndex: UInt32, generation: UInt16) -> Int {
        let clusterIdx = findNearestCentroid(vector: vector)
        let entry = IVFListEntry(slotIndex: slotIndex, generation: generation)
        invertedLists[clusterIdx].append(entry)
        gpuStructureDirty = true
        return clusterIdx
    }

    /// Find the nearest centroid to a vector.
    ///
    /// - Parameter vector: Query vector
    /// - Returns: Index of nearest centroid
    func findNearestCentroid(vector: [Float]) -> Int {
        guard !centroids.isEmpty else { return 0 }

        var bestIdx = 0
        var bestDist = Float.infinity

        for (i, centroid) in centroids.enumerated() {
            let dist = squaredL2Distance(vector, centroid)
            if dist < bestDist {
                bestDist = dist
                bestIdx = i
            }
        }

        return bestIdx
    }

    // MARK: - Remove

    /// Remove a vector from the index.
    ///
    /// - Parameters:
    ///   - slotIndex: Slot index to remove
    ///   - generation: Expected generation
    /// - Returns: true if found and removed
    @discardableResult
    func removeVector(slotIndex: UInt32, generation: UInt16) -> Bool {
        // Check staging first
        if let idx = stagingSlots.firstIndex(of: slotIndex) {
            stagingSlots.remove(at: idx)
            gpuStructureDirty = true
            return true
        }

        // Check all inverted lists
        for listIdx in 0..<invertedLists.count {
            if let entryIdx = invertedLists[listIdx].firstIndex(where: {
                $0.slotIndex == slotIndex && $0.generation == generation
            }) {
                invertedLists[listIdx].remove(at: entryIdx)
                gpuStructureDirty = true
                return true
            }
        }

        return false
    }

    // MARK: - GPU Structure

    /// Prepare GPU structure for search.
    ///
    /// - Parameters:
    ///   - storage: GPU vector storage
    ///   - device: Metal device
    /// - Returns: GPU-ready structure
    /// - Throws: `IndexAccelerationError` if preparation fails
    func prepareGPUStructure(
        storage: GPUVectorStorage,
        device: any MTLDevice
    ) throws -> IVFGPUIndexStructure {
        // Return cached if valid
        if let cached = gpuStructure, !gpuStructureDirty {
            return cached
        }

        // Create centroid buffer
        let flatCentroids = centroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: max(flatCentroids.count * MemoryLayout<Float>.size, 4),
            options: .storageModeShared
        ) else {
            throw IndexAccelerationError.bufferError(
                operation: "prepareGPUStructure",
                reason: "Failed to create centroid buffer"
            )
        }
        centroidBuffer.label = "IVFStructure.centroids"

        // Build CSR structure
        var listOffsets: [UInt32] = [0]
        var vectorIndices: [UInt32] = []
        var totalVecs = 0

        for list in invertedLists {
            for entry in list {
                vectorIndices.append(entry.slotIndex)
                totalVecs += 1
            }
            listOffsets.append(UInt32(totalVecs))
        }

        // Create offset buffer
        guard let offsetBuffer = device.makeBuffer(
            bytes: listOffsets,
            length: listOffsets.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw IndexAccelerationError.bufferError(
                operation: "prepareGPUStructure",
                reason: "Failed to create offset buffer"
            )
        }
        offsetBuffer.label = "IVFStructure.listOffsets"

        // Create index buffer
        let indexBuffer: any MTLBuffer
        if vectorIndices.isEmpty {
            guard let buf = device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw IndexAccelerationError.bufferError(
                    operation: "prepareGPUStructure",
                    reason: "Failed to create empty index buffer"
                )
            }
            indexBuffer = buf
        } else {
            guard let buf = device.makeBuffer(
                bytes: vectorIndices,
                length: vectorIndices.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared
            ) else {
                throw IndexAccelerationError.bufferError(
                    operation: "prepareGPUStructure",
                    reason: "Failed to create index buffer"
                )
            }
            indexBuffer = buf
        }
        indexBuffer.label = "IVFStructure.vectorIndices"

        // Use the main vector storage buffer
        guard let vectorBuffer = storage.buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "prepareGPUStructure")
        }

        let structure = IVFGPUIndexStructure(
            centroids: centroidBuffer,
            numCentroids: numClusters,
            listVectors: vectorBuffer,
            vectorIndices: indexBuffer,
            listOffsets: offsetBuffer,
            totalVectors: totalVecs,
            dimension: dimension
        )

        gpuStructure = structure
        gpuStructureDirty = false

        return structure
    }

    /// Invalidate cached GPU structure.
    func invalidateGPUStructure() {
        gpuStructureDirty = true
        gpuStructure = nil
    }

    // MARK: - Compaction

    /// Update structure after compaction.
    ///
    /// - Parameter indexMapping: Mapping from old slot indices to new slot indices
    func compact(using indexMapping: [UInt32: UInt32]) {
        // Update staging
        stagingSlots = stagingSlots.compactMap { indexMapping[$0] }

        // Update inverted lists
        for listIdx in 0..<invertedLists.count {
            invertedLists[listIdx] = invertedLists[listIdx].compactMap { entry in
                guard let newSlot = indexMapping[entry.slotIndex] else { return nil }
                return IVFListEntry(slotIndex: newSlot, generation: entry.generation &+ 1)
            }
        }

        gpuStructureDirty = true
    }

    // MARK: - Reset

    /// Reset to initial state.
    func reset() {
        trainingState = .untrained
        centroids = []
        invertedLists = Array(repeating: [], count: numClusters)
        stagingSlots = []
        gpuStructure = nil
        gpuStructureDirty = true
    }

    // MARK: - Statistics

    /// Get IVF-specific statistics.
    func getStats() -> GPUIndexStats.IVFStats {
        GPUIndexStats.IVFStats(
            numClusters: numClusters,
            nprobe: nprobe,
            isTrained: isTrained,
            averageVectorsPerCluster: averageClusterSize,
            clusterSizeStdDev: clusterSizeStdDev
        )
    }

    // MARK: - Private Helpers

    private func squaredL2Distance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
    }
}
