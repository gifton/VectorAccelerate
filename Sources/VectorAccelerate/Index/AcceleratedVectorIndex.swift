//
//  AcceleratedVectorIndex.swift
//  VectorAccelerate
//
//  GPU-first vector index with direct GPU data ownership.
//
//  This is the main entry point for the GPU index API. It provides:
//  - Direct GPU buffer ownership (no CPU copy)
//  - Stable handle-based vector identification (P0.8)
//  - Lazy deletion with on-demand compaction
//  - Native GPU distances (L2² for euclidean)
//  - Support for both Flat and IVF index types
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Metadata Type

/// Metadata for a vector (optional per-vector key-value pairs).
public typealias VectorMetadata = [String: String]

// MARK: - Accelerated Vector Index

/// GPU-accelerated vector index with direct data ownership.
///
/// `AcceleratedVectorIndex` provides a clean, GPU-first API for similarity search.
/// Unlike wrapper-based approaches, this index owns the vector data directly on
/// the GPU, eliminating redundant CPU copies.
///
/// ## Stable Handles (P0.8)
/// Handles remain valid across `compact()` operations. You never need to remap
/// handles after compaction - the index maintains an internal indirection table
/// that maps stable IDs to current storage slots.
///
/// ## Index Types
/// - **Flat**: Exhaustive search, best for < 10K vectors
/// - **IVF**: Inverted file index with clustering, best for > 10K vectors
///
/// ## Features
/// - **Direct GPU ownership**: Vectors live on GPU, no duplication
/// - **Stable handles**: `VectorHandle` remains valid across compaction
/// - **Native distances**: Returns L2² for euclidean (no expensive sqrt)
/// - **Lazy deletion**: Fast removes with on-demand compaction
/// - **Filtered search**: CPU-side metadata filtering with iterative fetch
///
/// ## Supported Metrics
/// Currently, only `.euclidean` metric is supported for GPU-accelerated search.
/// The search returns L2² (squared euclidean distance) for efficiency.
///
/// ## Flat Index Usage
/// ```swift
/// let index = try await AcceleratedVectorIndex(
///     configuration: .flat(dimension: 768, capacity: 10_000)
/// )
/// let handle = try await index.insert(embedding)
/// let results = try await index.search(query: queryVector, k: 10)
/// ```
///
/// ## IVF Index Usage
/// ```swift
/// let index = try await AcceleratedVectorIndex(
///     configuration: .ivf(dimension: 768, nlist: 256, nprobe: 16, capacity: 100_000)
/// )
/// // Insert vectors - training happens automatically
/// for vector in vectors {
///     _ = try await index.insert(vector)
/// }
/// // Or trigger training manually
/// try await index.train()
/// // Search
/// let results = try await index.search(query: queryVector, k: 10)
/// ```
public actor AcceleratedVectorIndex {

    // MARK: - Configuration

    /// Index configuration.
    public let configuration: IndexConfiguration

    // MARK: - Internal Components

    /// GPU context for compute operations.
    private let context: Metal4Context

    /// GPU buffer storage for vectors.
    private var storage: GPUVectorStorage

    /// Handle allocation and stable ID management.
    private var handleAllocator: HandleAllocator

    /// Deletion mask for lazy deletion.
    private var deletionMask: DeletionMask

    /// CPU-side metadata storage (keyed by stableID).
    private var metadataStore: MetadataStore

    /// Fused L2 Top-K kernel for flat search.
    private var fusedL2TopKKernel: FusedL2TopKKernel?

    // MARK: - IVF Components (Optional)

    /// IVF structure for inverted file index (nil for flat index).
    private var ivfStructure: IVFStructure?

    /// IVF search pipeline (nil for flat index).
    private var ivfSearchPipeline: IVFSearchPipeline?

    /// Whether auto-training is enabled for IVF.
    private var autoTrainEnabled: Bool = true

    // MARK: - Initialization

    /// Create an accelerated vector index with the given configuration.
    ///
    /// - Parameter configuration: Index configuration specifying dimension, metric, capacity, and type
    /// - Throws: `IndexError` if configuration is invalid or GPU initialization fails
    ///
    /// - Note: Currently only `.euclidean` metric is supported for GPU-accelerated search.
    public init(configuration: IndexConfiguration) async throws {
        try configuration.validate()
        try Self.validateMetricSupport(configuration.metric)

        self.configuration = configuration
        self.context = try await Metal4Context()

        // Initialize core components
        self.storage = try GPUVectorStorage(
            device: context.device.rawDevice,
            dimension: configuration.dimension,
            capacity: configuration.capacity
        )
        self.handleAllocator = HandleAllocator(initialSlotCapacity: configuration.capacity)
        self.deletionMask = DeletionMask(capacity: configuration.capacity)
        self.metadataStore = MetadataStore(capacity: configuration.capacity / 10)

        // Initialize index-type-specific components
        switch configuration.indexType {
        case .flat:
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)

        case .ivf(let nlist, let nprobe, let minTrainingVectors):
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)
            self.ivfStructure = IVFStructure(
                numClusters: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                minTrainingVectors: minTrainingVectors,
                quantization: configuration.quantization
            )
            let ivfConfig = IVFSearchConfiguration(
                numCentroids: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                metric: configuration.metric
            )
            self.ivfSearchPipeline = try await IVFSearchPipeline(
                context: context,
                configuration: ivfConfig
            )
        }
    }

    /// Create an accelerated vector index with an existing Metal context.
    ///
    /// Use this initializer when you want to share a context across multiple indexes.
    ///
    /// - Parameters:
    ///   - configuration: Index configuration
    ///   - context: Existing Metal 4 context to use
    /// - Throws: `IndexError` if configuration is invalid
    public init(configuration: IndexConfiguration, context: Metal4Context) async throws {
        try configuration.validate()
        try Self.validateMetricSupport(configuration.metric)

        self.configuration = configuration
        self.context = context

        // Initialize core components
        self.storage = try GPUVectorStorage(
            device: context.device.rawDevice,
            dimension: configuration.dimension,
            capacity: configuration.capacity
        )
        self.handleAllocator = HandleAllocator(initialSlotCapacity: configuration.capacity)
        self.deletionMask = DeletionMask(capacity: configuration.capacity)
        self.metadataStore = MetadataStore(capacity: configuration.capacity / 10)

        // Initialize index-type-specific components
        switch configuration.indexType {
        case .flat:
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)

        case .ivf(let nlist, let nprobe, let minTrainingVectors):
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)
            self.ivfStructure = IVFStructure(
                numClusters: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                minTrainingVectors: minTrainingVectors,
                quantization: configuration.quantization
            )
            let ivfConfig = IVFSearchConfiguration(
                numCentroids: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                metric: configuration.metric
            )
            self.ivfSearchPipeline = try await IVFSearchPipeline(
                context: context,
                configuration: ivfConfig
            )
        }
    }

    /// Validate that the metric is supported for GPU acceleration.
    private static func validateMetricSupport(_ metric: SupportedDistanceMetric) throws {
        switch metric {
        case .euclidean:
            break // Supported
        case .cosine, .dotProduct, .manhattan, .chebyshev:
            throw IndexError.invalidConfiguration(
                parameter: "metric",
                reason: "'\(metric)' is not yet supported. Currently only .euclidean is available for GPU-accelerated search."
            )
        }
    }

    // MARK: - Properties

    /// Number of active (non-deleted) vectors in the index.
    public var count: Int {
        handleAllocator.occupiedCount
    }

    /// Vector dimension.
    public var dimension: Int {
        configuration.dimension
    }

    /// Whether the index is empty.
    public var isEmpty: Bool {
        handleAllocator.occupiedCount == 0
    }

    /// Distance metric used by this index.
    public var metric: SupportedDistanceMetric {
        configuration.metric
    }

    /// Current capacity of the index.
    public var capacity: Int {
        storage.capacity
    }

    /// Whether the index uses IVF.
    public var isIVF: Bool {
        ivfStructure != nil
    }

    /// Whether the IVF index is trained (always true for flat index).
    public var isTrained: Bool {
        ivfStructure?.isTrained ?? true
    }

    /// Whether search will route to flat search due to the routing threshold.
    ///
    /// Returns `true` when:
    /// - The index is configured as IVF
    /// - Routing threshold is enabled (> 0)
    /// - Current vector count is below the threshold
    ///
    /// Useful for testing and debugging routing behavior.
    public var willRouteToFlat: Bool {
        guard ivfStructure != nil else { return false }  // Already flat
        guard configuration.routingThreshold > 0 else { return false }  // Routing disabled
        return handleAllocator.occupiedCount < configuration.routingThreshold
    }

    /// The routing threshold configuration.
    public var routingThreshold: Int {
        configuration.routingThreshold
    }

    // MARK: - Statistics

    /// Get statistics about the index.
    public func statistics() -> GPUIndexStats {
        let ivfStats = ivfStructure?.getStats()

        return GPUIndexStats(
            vectorCount: handleAllocator.occupiedCount,
            allocatedSlots: storage.allocatedSlots,
            deletedSlots: deletionMask.deletedCount,
            dimension: configuration.dimension,
            metric: configuration.metric,
            indexType: configuration.indexType,
            capacity: storage.capacity,
            gpuVectorMemoryBytes: storage.usedBytes,
            gpuIndexStructureBytes: 0,
            cpuMetadataMemoryBytes: metadataStore.estimatedMemoryBytes,
            ivfStats: ivfStats
        )
    }

    // MARK: - IVF Training

    /// Train the IVF index.
    ///
    /// For IVF indexes, this runs K-Means clustering on the inserted vectors
    /// to compute centroids. Training is required before search returns accurate results.
    ///
    /// For flat indexes, this method does nothing.
    ///
    /// - Note: Training happens automatically when enough vectors are inserted
    ///         (if auto-training is enabled).
    /// - Throws: `IndexError` if training fails
    public func train() async throws {
        guard let ivf = ivfStructure else { return }
        guard !ivf.isTrained else { return }

        // Gather all vectors for training
        var trainingVectors: [[Float]] = []
        trainingVectors.reserveCapacity(handleAllocator.occupiedCount)

        // Iterate only over actually allocated slots (not the full mask capacity)
        for slotIndex in deletionMask {
            // Skip slots beyond what's actually allocated in storage
            guard slotIndex < storage.allocatedSlots else { continue }
            let vector = try storage.readVector(at: slotIndex)
            trainingVectors.append(vector)
        }

        guard !trainingVectors.isEmpty else {
            throw IndexError.invalidInput(
                message: "Cannot train IVF index with no vectors"
            )
        }

        try await ivf.train(vectors: trainingVectors, context: context)
        // Note: IVFStructure.train() already assigns staged vectors to clusters,
        // so no need to re-assign here.
    }

    /// Enable or disable auto-training for IVF indexes.
    ///
    /// When enabled, the index automatically trains when enough vectors are inserted.
    ///
    /// - Parameter enabled: Whether auto-training should be enabled
    public func setAutoTraining(_ enabled: Bool) {
        autoTrainEnabled = enabled
    }

    // MARK: - Insert Operations

    /// Insert a single vector.
    ///
    /// - Parameters:
    ///   - vector: Vector data (must match index dimension)
    ///   - metadata: Optional metadata for filtering
    /// - Returns: Stable handle to the inserted vector
    /// - Throws: `IndexError` if dimension mismatch or GPU error
    public func insert(
        _ vector: consuming [Float],
        metadata: VectorMetadata? = nil
    ) async throws -> VectorHandle {
        guard vector.count == configuration.dimension else {
            throw IndexError.dimensionMismatch(
                expected: configuration.dimension,
                got: vector.count
            )
        }

        // Ensure capacity
        try storage.ensureCapacity(storage.allocatedSlots + 1)
        deletionMask.ensureCapacity(storage.allocatedSlots + 1)

        // Allocate handle (returns handle with new stableID, allocates new slot internally)
        let handle = handleAllocator.allocate()

        // Get the slot for this handle
        guard let slotIndex = handleAllocator.slot(for: handle) else {
            throw IndexError.bufferError(
                operation: "insert",
                reason: "Failed to get slot for newly allocated handle"
            )
        }

        // Write vector to GPU
        try storage.writeVector(vector, at: Int(slotIndex))

        // Store metadata if provided (keyed by stableID via handle)
        if let metadata = metadata {
            metadataStore[handle] = metadata
        }

        // Handle IVF-specific logic
        if let ivf = ivfStructure {
            if ivf.isTrained {
                // Assign to cluster
                _ = ivf.assignToCluster(
                    vector: vector,
                    slotIndex: slotIndex
                )
            } else {
                // Add to staging
                ivf.addToStaging(slotIndex)

                // Auto-train if we have enough vectors
                if autoTrainEnabled && ivf.canTrain {
                    try await train()
                }
            }
        }

        return handle
    }

    /// Batch insert multiple vectors.
    ///
    /// More efficient than individual inserts for large batches.
    /// Uses optimized batch GPU write operations.
    ///
    /// - Parameters:
    ///   - vectors: Array of vectors (each must match index dimension)
    ///   - metadata: Optional metadata array (must match vectors count if provided)
    /// - Returns: Array of stable handles for inserted vectors
    /// - Throws: `IndexError` if dimension mismatch or GPU error
    public func insert(
        _ vectors: [[Float]],
        metadata: [VectorMetadata?]? = nil
    ) async throws -> [VectorHandle] {
        guard !vectors.isEmpty else { return [] }

        // Validate dimensions
        for vector in vectors {
            guard vector.count == configuration.dimension else {
                throw IndexError.dimensionMismatch(
                    expected: configuration.dimension,
                    got: vector.count
                )
            }
        }

        if let metadata = metadata, metadata.count != vectors.count {
            throw IndexError.invalidInput(
                message: "Metadata count (\(metadata.count)) must match vectors count (\(vectors.count))"
            )
        }

        // Ensure capacity
        try storage.ensureCapacity(storage.allocatedSlots + vectors.count)
        deletionMask.ensureCapacity(storage.allocatedSlots + vectors.count)

        // Allocate handles
        let handles = handleAllocator.allocate(count: vectors.count)

        // Get start slot for batch write
        guard let startSlot = handleAllocator.slot(for: handles[0]) else {
            throw IndexError.bufferError(
                operation: "insert",
                reason: "Failed to get slot for batch insert"
            )
        }

        // Batch write vectors to GPU (optimized)
        try storage.writeVectors(vectors, startingAt: Int(startSlot))

        // Store metadata
        if let metadata = metadata {
            for (i, meta) in metadata.enumerated() {
                if let meta = meta {
                    metadataStore[handles[i]] = meta
                }
            }
        }

        // Handle IVF-specific logic
        if let ivf = ivfStructure {
            if ivf.isTrained {
                // Assign each vector to a cluster
                for (i, vector) in vectors.enumerated() {
                    if let slot = handleAllocator.slot(for: handles[i]) {
                        _ = ivf.assignToCluster(
                            vector: vector,
                            slotIndex: slot
                        )
                    }
                }
            } else {
                // Add all to staging
                for handle in handles {
                    if let slot = handleAllocator.slot(for: handle) {
                        ivf.addToStaging(slot)
                    }
                }

                // Auto-train if we have enough vectors
                if autoTrainEnabled && ivf.canTrain {
                    try await train()
                }
            }
        }

        return handles
    }

    // MARK: - Vector Retrieval

    /// Retrieve the vector data for a handle.
    ///
    /// - Parameter handle: Handle to the vector
    /// - Returns: Vector data, or nil if handle is invalid
    /// - Throws: `IndexError` if GPU read fails
    public func vector(for handle: VectorHandle) throws -> [Float]? {
        guard let slot = handleAllocator.slot(for: handle) else { return nil }
        return try storage.readVector(at: Int(slot))
    }

    /// Retrieve multiple vectors by their handles.
    ///
    /// - Parameter handles: Array of handles
    /// - Returns: Array of optional vectors (nil for invalid handles)
    /// - Throws: `IndexError` if GPU read fails
    public func vectors(for handles: [VectorHandle]) throws -> [[Float]?] {
        try handles.map { handle in
            guard let slot = handleAllocator.slot(for: handle) else { return nil }
            return try storage.readVector(at: Int(slot))
        }
    }

    // MARK: - Metadata Operations

    /// Get metadata for a handle.
    ///
    /// - Parameter handle: Handle to the vector
    /// - Returns: Metadata if set, nil otherwise. Returns nil for invalid handles.
    public func metadata(for handle: VectorHandle) -> VectorMetadata? {
        guard handleAllocator.validate(handle) else { return nil }
        return metadataStore[handle]
    }

    /// Update metadata for a handle.
    ///
    /// - Parameters:
    ///   - metadata: New metadata (nil to remove)
    ///   - handle: Handle to the vector
    /// - Throws: `IndexError.invalidInput` if handle is invalid
    public func setMetadata(_ metadata: VectorMetadata?, for handle: VectorHandle) throws {
        guard handleAllocator.validate(handle) else {
            throw IndexError.invalidInput(
                message: "Invalid handle: \(handle)"
            )
        }
        metadataStore[handle] = metadata
    }

    // MARK: - Remove Operations

    /// Mark a vector as deleted (lazy deletion).
    ///
    /// The vector is not immediately removed from the GPU buffer.
    /// Call `compact()` to reclaim space from deleted vectors.
    ///
    /// - Parameter handle: Handle to the vector to remove
    /// - Throws: `IndexError.invalidInput` if handle is invalid
    public func remove(_ handle: VectorHandle) throws {
        guard let slot = handleAllocator.slot(for: handle) else {
            throw IndexError.invalidInput(
                message: "Invalid handle: \(handle)"
            )
        }

        handleAllocator.markDeleted(handle)
        deletionMask.markDeleted(Int(slot))
        metadataStore.remove(for: handle)

        // Remove from IVF structure
        ivfStructure?.removeVector(slotIndex: slot)
    }

    /// Mark multiple vectors as deleted (lazy deletion).
    ///
    /// More efficient than individual removes for bulk operations.
    ///
    /// - Parameter handles: Handles to remove
    /// - Returns: Number of vectors actually removed (excludes invalid handles)
    @discardableResult
    public func remove(_ handles: [VectorHandle]) -> Int {
        var removedCount = 0
        for handle in handles {
            if let slot = handleAllocator.slot(for: handle) {
                handleAllocator.markDeleted(handle)
                deletionMask.markDeleted(Int(slot))
                metadataStore.remove(for: handle)
                ivfStructure?.removeVector(slotIndex: slot)
                removedCount += 1
            }
        }
        return removedCount
    }

    /// Compact the index to reclaim space from deleted vectors.
    ///
    /// This operation rebuilds the GPU buffer without deleted vectors.
    /// With stable handles (P0.8), handles remain valid after compaction -
    /// you do not need to update any handle references.
    ///
    /// - Note: This is an expensive operation. Call sparingly when fragmentation is high.
    public func compact() async throws {
        guard deletionMask.deletedCount > 0 else { return }

        // Get keep mask
        let keepMask = deletionMask.keepMask()

        // Compact storage and get slot mapping (Int -> Int)
        let intSlotMapping = try storage.compact(keepMask: keepMask)

        // Convert to UInt32 -> UInt32 for handle allocator and IVF
        var slotMapping: [UInt32: UInt32] = [:]
        slotMapping.reserveCapacity(intSlotMapping.count)
        for (oldSlot, newSlot) in intSlotMapping {
            slotMapping[UInt32(oldSlot)] = UInt32(newSlot)
        }

        // Apply compaction to handle allocator (updates indirection tables)
        handleAllocator.applyCompaction(
            slotMapping: slotMapping,
            newSlotCount: storage.allocatedSlots
        )

        // Compact IVF structure (updates slot references)
        ivfStructure?.compact(using: slotMapping)

        // Reset deletion mask
        deletionMask.resetAfterCompaction(newCapacity: storage.allocatedSlots)

        // Note: MetadataStore doesn't need compaction since it's keyed by stableID
    }

    // MARK: - Search Operations

    /// Search for nearest neighbors.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of results to return
    ///   - filter: Optional filter predicate on metadata
    /// - Returns: Array of search results sorted by distance (ascending)
    /// - Throws: `IndexError` if dimension mismatch or GPU error
    public func search(
        query: consuming [Float],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)? = nil
    ) async throws -> [IndexSearchResult] {
        guard query.count == configuration.dimension else {
            throw IndexError.dimensionMismatch(
                expected: configuration.dimension,
                got: query.count
            )
        }

        guard handleAllocator.occupiedCount > 0 else {
            return []
        }

        // Use IVF search if available and trained, unless below routing threshold
        if let ivf = ivfStructure, ivf.isTrained {
            let vectorCount = handleAllocator.occupiedCount
            let threshold = configuration.routingThreshold

            // Route to flat search for small datasets if threshold is enabled
            if threshold > 0 && vectorCount < threshold {
                VectorLogDebug(
                    "Routing to flat search: \(vectorCount) vectors < threshold \(threshold)",
                    category: "Search"
                )
                // Fall through to flat search
            } else {
                return try await searchIVF(query: query, k: k, filter: filter)
            }
        }

        // Fall back to flat search
        guard let filter = filter else {
            return try await searchUnfiltered(query: query, k: k)
        }

        return try await searchFiltered(query: query, k: k, filter: filter)
    }

    /// Batch search for multiple queries.
    ///
    /// More efficient than individual searches for multiple queries.
    /// Uses true GPU batching for flat index without filter.
    ///
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of results per query
    ///   - filter: Optional filter predicate on metadata
    /// - Returns: Array of result arrays (one per query)
    /// - Throws: `IndexError` if dimension mismatch or GPU error
    public func search(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)? = nil
    ) async throws -> [[IndexSearchResult]] {
        guard !queries.isEmpty else { return [] }

        // Validate dimensions
        for query in queries {
            guard query.count == configuration.dimension else {
                throw IndexError.dimensionMismatch(
                    expected: configuration.dimension,
                    got: query.count
                )
            }
        }

        guard handleAllocator.occupiedCount > 0 else {
            return queries.map { _ in [] }
        }

        // Fast path: flat index without filter uses true GPU batch
        if case .flat = configuration.indexType, filter == nil {
            return try await searchBatchGPU(queries: queries, k: k)
        }

        // Use IVF batch search if available and trained, unless below routing threshold
        if let ivf = ivfStructure, ivf.isTrained, filter == nil {
            let vectorCount = handleAllocator.occupiedCount
            let threshold = configuration.routingThreshold

            // Route to flat search for small datasets if threshold is enabled
            if threshold > 0 && vectorCount < threshold {
                VectorLogDebug(
                    "Routing batch to flat search: \(vectorCount) vectors < threshold \(threshold)",
                    category: "Search"
                )
                // Fall through to sequential flat search
            } else {
                return try await searchIVFBatch(queries: queries, k: k)
            }
        }

        // Fall back to sequential search for IVF or filtered
        var results: [[IndexSearchResult]] = []
        results.reserveCapacity(queries.count)

        for query in queries {
            let result = try await search(query: query, k: k, filter: filter)
            results.append(result)
        }

        return results
    }

    // MARK: - Index Introspection

    /// Get all valid handles in the index.
    ///
    /// - Returns: Array of all valid handles
    public func allHandles() -> [VectorHandle] {
        var handles: [VectorHandle] = []
        handles.reserveCapacity(handleAllocator.occupiedCount)

        for slotIndex in deletionMask {
            if let handle = handleAllocator.handle(for: UInt32(slotIndex)) {
                handles.append(handle)
            }
        }

        return handles
    }

    /// Check if a handle exists in the index.
    ///
    /// - Parameter handle: Handle to check
    /// - Returns: true if handle is valid and exists
    public func contains(_ handle: VectorHandle) -> Bool {
        handleAllocator.validate(handle)
    }

    /// Iterate over all vectors with their handles and metadata.
    ///
    /// - Parameter body: Closure called for each vector
    /// - Throws: Rethrows any error from the closure or GPU read
    public func forEach(_ body: (VectorHandle, [Float], VectorMetadata?) throws -> Void) throws {
        for slotIndex in deletionMask {
            guard let handle = handleAllocator.handle(for: UInt32(slotIndex)) else { continue }
            let vector = try storage.readVector(at: slotIndex)
            let metadata = metadataStore[handle]
            try body(handle, vector, metadata)
        }
    }

    /// Get handles matching a metadata predicate.
    ///
    /// - Parameter predicate: Filter function on metadata
    /// - Returns: Array of matching handles
    public func handles(where predicate: (VectorMetadata?) -> Bool) -> [VectorHandle] {
        var result: [VectorHandle] = []

        for slotIndex in deletionMask {
            guard let handle = handleAllocator.handle(for: UInt32(slotIndex)) else { continue }
            let metadata = metadataStore[handle]
            if predicate(metadata) {
                result.append(handle)
            }
        }

        return result
    }

    // MARK: - Lifecycle

    /// Release GPU resources.
    ///
    /// After calling this, the index cannot be used.
    public func releaseResources() {
        storage.release()
        fusedL2TopKKernel = nil
        ivfSearchPipeline = nil
        ivfStructure?.reset()
        metadataStore.reset()
        handleAllocator.reset()
        deletionMask.reset()
    }

    // MARK: - Handle Validation

    /// Check if a handle is valid.
    ///
    /// - Parameter handle: Handle to validate
    /// - Returns: true if handle is valid for this index
    public func isHandleValid(_ handle: VectorHandle) -> Bool {
        handleAllocator.validate(handle)
    }

    /// Get the storage slot for a handle (internal use).
    ///
    /// - Parameter handle: Handle to look up
    /// - Returns: Slot index, or nil if handle is invalid
    public func slot(for handle: VectorHandle) -> UInt32? {
        handleAllocator.slot(for: handle)
    }

    /// Get a valid handle for a slot index (if occupied).
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Valid handle, or nil if slot is not occupied
    public func handle(for slotIndex: UInt32) -> VectorHandle? {
        handleAllocator.handle(for: slotIndex)
    }

    // MARK: - Private Implementation - Flat Search

    private func searchUnfiltered(query: [Float], k: Int) async throws -> [IndexSearchResult] {
        guard let kernel = fusedL2TopKKernel,
              let datasetBuffer = storage.buffer else {
            throw IndexError.gpuNotInitialized(operation: "search")
        }

        let effectiveK = min(k, handleAllocator.occupiedCount)

        // Create query buffer (from pool for transient allocation)
        let queryToken = try await context.getBuffer(for: query)
        let queryBuffer = queryToken.buffer
        queryBuffer.label = "AcceleratedVectorIndex.query"

        // Request more results than needed to account for deleted vectors
        let fetchK = min(effectiveK + deletionMask.deletedCount, storage.allocatedSlots)

        let parameters = try FusedL2TopKParameters(
            numQueries: 1,
            numDataset: storage.allocatedSlots,
            dimension: configuration.dimension,
            k: fetchK
        )

        let gpuResult = try await kernel.execute(
            queries: queryBuffer,
            dataset: datasetBuffer,
            parameters: parameters
        )

        // Convert GPU results to IndexSearchResults, filtering deleted vectors
        var results: [IndexSearchResult] = []
        results.reserveCapacity(effectiveK)

        for (rawIndex, distance) in gpuResult.results(for: 0) {
            // Skip deleted vectors
            if deletionMask.isDeleted(rawIndex) { continue }

            // Get valid handle (maps slot -> stableID)
            guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

            results.append(IndexSearchResult(handle: handle, distance: distance))

            if results.count >= effectiveK { break }
        }

        return results
    }

    private func searchFiltered(
        query: [Float],
        k: Int,
        filter: @Sendable (VectorHandle, VectorMetadata?) -> Bool
    ) async throws -> [IndexSearchResult] {
        guard let kernel = fusedL2TopKKernel,
              let datasetBuffer = storage.buffer else {
            throw IndexError.gpuNotInitialized(operation: "search")
        }

        // Create query buffer (from pool for transient allocation)
        let queryToken = try await context.getBuffer(for: query)
        let queryBuffer = queryToken.buffer
        queryBuffer.label = "AcceleratedVectorIndex.query"

        // Iterative fetch strategy: start with 2x, double until we have enough
        // Track which indices we've already seen to avoid duplicates across iterations
        var results: [IndexSearchResult] = []
        var seenIndices = Set<Int>()
        let maxFetch = storage.allocatedSlots
        var fetchK = min(k * 2, maxFetch)

        while results.count < k {
            let parameters = try FusedL2TopKParameters(
                numQueries: 1,
                numDataset: maxFetch,
                dimension: configuration.dimension,
                k: fetchK
            )

            let gpuResult = try await kernel.execute(
                queries: queryBuffer,
                dataset: datasetBuffer,
                parameters: parameters
            )

            // Process all GPU results, skipping ones we've already seen
            let gpuResults = gpuResult.results(for: 0)

            for (rawIndex, distance) in gpuResults {
                // Skip already-processed indices
                if seenIndices.contains(rawIndex) { continue }
                seenIndices.insert(rawIndex)

                // Skip deleted vectors
                if deletionMask.isDeleted(rawIndex) { continue }

                // Get valid handle (maps slot -> stableID)
                guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

                let meta = metadataStore[handle]

                if filter(handle, meta) {
                    results.append(IndexSearchResult(handle: handle, distance: distance))
                    if results.count >= k { break }
                }
            }

            if results.count >= k { break }

            // Double fetch size for next iteration
            let nextFetchK = min(fetchK * 2, maxFetch)

            // Stop if we can't fetch more (already at max or no progress)
            if nextFetchK == fetchK || seenIndices.count >= maxFetch { break }

            fetchK = nextFetchK
        }

        return results
    }

    // MARK: - Private Implementation - GPU Batch Search (Flat Index)

    /// Perform true GPU batch search for flat index.
    ///
    /// Uses a single GPU dispatch for all queries, which is more efficient
    /// than sequential searches when querying with multiple vectors.
    ///
    /// - Parameters:
    ///   - queries: Array of query vectors (all must have matching dimension)
    ///   - k: Number of results per query
    /// - Returns: Array of result arrays (one per query)
    private func searchBatchGPU(queries: [[Float]], k: Int) async throws -> [[IndexSearchResult]] {
        guard let kernel = fusedL2TopKKernel,
              let datasetBuffer = storage.buffer else {
            throw IndexError.gpuNotInitialized(operation: "searchBatchGPU")
        }

        let numQueries = queries.count
        let effectiveK = min(k, handleAllocator.occupiedCount)

        // Flatten queries into contiguous buffer (from pool for transient allocation)
        let flatQueries = queries.flatMap { $0 }
        let queryToken = try await context.getBuffer(for: flatQueries)
        let queryBuffer = queryToken.buffer
        queryBuffer.label = "AcceleratedVectorIndex.batchQueries"

        // Request more results than needed to account for deleted vectors
        let fetchK = min(effectiveK + deletionMask.deletedCount, storage.allocatedSlots)

        let parameters = try FusedL2TopKParameters(
            numQueries: numQueries,
            numDataset: storage.allocatedSlots,
            dimension: configuration.dimension,
            k: fetchK
        )

        let gpuResult = try await kernel.execute(
            queries: queryBuffer,
            dataset: datasetBuffer,
            parameters: parameters
        )

        // Convert GPU results to IndexSearchResults, filtering deleted vectors
        var allResults: [[IndexSearchResult]] = []
        allResults.reserveCapacity(numQueries)

        for queryIdx in 0..<numQueries {
            var results: [IndexSearchResult] = []
            results.reserveCapacity(effectiveK)

            for (rawIndex, distance) in gpuResult.results(for: queryIdx) {
                // Skip deleted vectors
                if deletionMask.isDeleted(rawIndex) { continue }

                // Get valid handle (maps slot -> stableID)
                guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

                results.append(IndexSearchResult(handle: handle, distance: distance))

                if results.count >= effectiveK { break }
            }

            allResults.append(results)
        }

        return allResults
    }

    // MARK: - Private Implementation - IVF Search

    private func searchIVF(
        query: [Float],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)?
    ) async throws -> [IndexSearchResult] {
        guard let ivf = ivfStructure,
              let pipeline = ivfSearchPipeline else {
            throw IndexError.gpuNotInitialized(operation: "searchIVF")
        }

        // Prepare GPU structure
        let gpuStructure = try ivf.prepareGPUStructure(
            storage: storage,
            device: context.device.rawDevice
        )

        // Execute IVF search
        let ivfResult = try await pipeline.search(
            queries: [query],
            structure: gpuStructure,
            k: filter != nil ? k * 3 : k  // Over-fetch for filtering
        )

        // Convert to IndexSearchResults
        var results: [IndexSearchResult] = []
        results.reserveCapacity(k)

        for (rawIndex, distance) in ivfResult.results(for: 0) {
            // Skip deleted vectors
            if deletionMask.isDeleted(rawIndex) { continue }

            // Get valid handle (maps slot -> stableID)
            guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

            // Apply filter if provided
            if let filter = filter {
                let meta = metadataStore[handle]
                if !filter(handle, meta) { continue }
            }

            results.append(IndexSearchResult(handle: handle, distance: distance))

            if results.count >= k { break }
        }

        return results
    }

    private func searchIVFBatch(queries: [[Float]], k: Int) async throws -> [[IndexSearchResult]] {
        guard let ivf = ivfStructure,
              let pipeline = ivfSearchPipeline else {
            throw IndexError.gpuNotInitialized(operation: "searchIVFBatch")
        }

        // Prepare GPU structure
        let gpuStructure = try ivf.prepareGPUStructure(
            storage: storage,
            device: context.device.rawDevice
        )

        // Execute batch IVF search
        let ivfResult = try await pipeline.search(
            queries: queries,
            structure: gpuStructure,
            k: k
        )

        // Convert all results
        var allResults: [[IndexSearchResult]] = []
        allResults.reserveCapacity(queries.count)

        for queryIdx in 0..<queries.count {
            var results: [IndexSearchResult] = []
            results.reserveCapacity(k)

            for (rawIndex, distance) in ivfResult.results(for: queryIdx) {
                // Skip deleted vectors
                if deletionMask.isDeleted(rawIndex) { continue }

                // Get valid handle (maps slot -> stableID)
                guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

                results.append(IndexSearchResult(handle: handle, distance: distance))

                if results.count >= k { break }
            }

            allResults.append(results)
        }

        return allResults
    }
}

// MARK: - Async Sequence Support

extension AcceleratedVectorIndex {
    /// Async iterator for vectors in the index.
    public struct VectorIterator: AsyncIteratorProtocol {
        let index: AcceleratedVectorIndex
        var slotIndices: [Int]
        var currentPosition: Int = 0

        init(index: AcceleratedVectorIndex, slotIndices: [Int]) {
            self.index = index
            self.slotIndices = slotIndices
        }

        public mutating func next() async throws -> (VectorHandle, [Float], VectorMetadata?)? {
            while currentPosition < slotIndices.count {
                let slotIndex = slotIndices[currentPosition]
                currentPosition += 1

                guard let handle = await index.handle(for: UInt32(slotIndex)) else { continue }
                let vector = try await index.vector(for: handle)
                guard let vector = vector else { continue }
                let metadata = await index.metadata(for: handle)

                return (handle, vector, metadata)
            }
            return nil
        }
    }

    /// Create an async iterator over all vectors.
    ///
    /// - Returns: Async iterator yielding (handle, vector, metadata) tuples
    public func makeAsyncIterator() -> VectorIterator {
        VectorIterator(index: self, slotIndices: deletionMask.activeIndices())
    }
}
